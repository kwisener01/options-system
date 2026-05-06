"""
Options paper trading — manual control script.

Usage:
    python run_options_paper.py status         # show current P&L and open position
    python run_options_paper.py open           # open a new spread now (runs strategy selector)
    python run_options_paper.py check          # check take-profit on open position
    python run_options_paper.py settle         # settle expired position (run Friday PM)
    python run_options_paper.py reset          # wipe state and start fresh (prompts)

The scheduler in main.py runs these automatically:
  Friday 9:45 AM ET  → open
  Mon-Thu 3:45 PM ET → check
  Friday 3:45 PM ET  → check + settle
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.logger import setup_logging
from src.backtest.multi_strategy_engine import MultiStrategyEngine
from src.live.options_paper_trader import (
    open_weekly, preview_weekly, check_tp, settle_expiry, status,
    _load, _save, STATE_PATH, _try_alpaca_open,
)


def parse_args():
    p = argparse.ArgumentParser(description="Options paper trading control")
    p.add_argument("action", choices=["status", "open", "check", "settle", "reset", "backfill-alpaca"],
                   nargs="?", default="status")
    p.add_argument("--capital",     type=float, default=None,  help="Override starting capital for reset")
    p.add_argument("--otm",         type=float, default=0.02)
    p.add_argument("--width",       type=float, default=0.01)
    p.add_argument("--max-vix",     type=float, default=25.0)
    p.add_argument("--low-vol",     type=float, default=18.0)
    p.add_argument("--risk-pct",    type=float, default=0.10)
    p.add_argument("--take-profit", type=float, default=0.50)
    p.add_argument("--dry-run", action="store_true",
                   help="Preview strategy/strikes without saving state or placing orders")
    return p.parse_args()


def _make_engine(args) -> MultiStrategyEngine:
    return MultiStrategyEngine(
        short_otm_pct=args.otm,
        spread_width_pct=args.width,
        call_otm_pct=args.otm,
        max_risk_pct=args.risk_pct,
        max_vix_entry=args.max_vix,
        low_vol_threshold=args.low_vol,
        take_profit_pct=args.take_profit,
    )


def fmt_status(s: dict):
    print()
    print("=" * 56)
    print("  OPTIONS PAPER TRADING — STATUS")
    print("=" * 56)
    print(f"  Account value    : ${s['account_value']:>12,.2f}")
    print(f"  Total P&L        : ${s['total_pnl']:>+12,.2f}")
    w, l = s["n_wins"], s["n_losses"]
    wr   = w / (w + l) if (w + l) > 0 else 0
    print(f"  Record           :  {w}W / {l}L  ({wr:.0%} win rate)")

    pos = s.get("open_position")
    if pos:
        print()
        print("  OPEN POSITION:")
        print(f"    Strategy  : {pos['strategy']}")
        print(f"    Entry     : {pos['entry_date']}  SPY=${pos['spot_entry']:.2f}  VIX={pos['vix_at_entry']:.1f}")
        print(f"    Expiry    : {pos['expiry_date']}")
        nc = pos["net_credit"]
        c  = pos["contracts"]
        if pos["strategy"] == "IRON_CONDOR":
            print(f"    Strikes   : Put {pos['put_short']:.1f}/{pos['put_long']:.1f}"
                  f" + Call {pos['call_short']:.1f}/{pos['call_long']:.1f}")
        else:
            print(f"    Strikes   : {pos['short_strike']:.1f} / {pos['long_strike']:.1f}")
        print(f"    Credit    : ${nc:.3f}/sh × {c} contracts = ${nc*100*c:,.2f}")
        print(f"    Max risk  : ${pos['max_risk']*100*c:,.2f}")

        if s.get("unrealized_pnl") is not None:
            unreal = s["unrealized_pnl"]
            cur    = s["current_value"]
            pct    = (1 - cur / abs(nc)) * 100 if nc != 0 else 0
            print(f"    Current   : ${cur:.3f}/sh  ({pct:.0f}% decayed)")
            print(f"    Unrealized: ${unreal:+,.2f}")
    else:
        print()
        print("  No open position.")
    print("=" * 56)
    print()


def fmt_preview(r: dict):
    from src.backtest.multi_strategy_engine import STRATEGY_IRON_CONDOR, STRATEGY_BULL_CALL_DEBIT
    print()
    print("=" * 56)
    print("  OPTIONS DRY-RUN — THIS WEEK'S CANDIDATE")
    print("=" * 56)
    print(f"  SPY        : ${r['spot']:.2f}")
    print(f"  VIX        : {r['vix']:.2f}")
    print(f"  Regime     : {r['regime']}  |  Trend: {r['spy_trend']}")
    print(f"  Macro      : {r['macro_summary']}")
    print(f"  Strategy   : {r['strategy']}")

    pos = r.get("position")
    if pos:
        print()
        if r["strategy"] == STRATEGY_IRON_CONDOR:
            print(f"  Put spread : {pos['put_short']:.1f} / {pos['put_long']:.1f}  "
                  f"(credit ${pos['put_credit']:.3f}/sh)")
            print(f"  Call spread: {pos['call_short']:.1f} / {pos['call_long']:.1f}  "
                  f"(credit ${pos['call_credit']:.3f}/sh)")
        elif r["strategy"] == STRATEGY_BULL_CALL_DEBIT:
            print(f"  Call debit : {pos['short_strike']:.1f} / {pos['long_strike']:.1f}")
        else:
            print(f"  Strikes    : {pos['short_strike']:.1f} / {pos['long_strike']:.1f}")
        nc = pos["net_credit"]
        c  = pos["contracts"]
        mr = pos["max_risk"]
        print(f"  Net credit : ${nc:.3f}/sh × {c} contracts = ${nc*100*c:,.2f}")
        print(f"  Max risk   : ${mr*100*c:,.2f}")
        print(f"  Expiry     : {pos['expiry_date']}")
        print(f"  Width      : ${pos['spread_width']:.2f}  |  Sigma: {pos['sigma']*100:.1f}%")
        print()
        print("  [DRY RUN — no state saved, no orders placed]")
    else:
        print()
        print("  No trade — conditions call for CASH this week.")
    print("=" * 56)
    print()


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    args   = parse_args()

    if args.action == "status":
        s = status()
        fmt_status(s)

    elif args.action == "open":
        engine = _make_engine(args)
        if args.dry_run:
            r = preview_weekly(engine)
            fmt_preview(r)
        else:
            pos = open_weekly(engine)
            if pos:
                print(f"\nOpened {pos['strategy']}  "
                      f"{pos['short_strike']:.1f}/{pos['long_strike']:.1f}  "
                      f"credit=${pos['net_credit']:.3f}  expiry={pos['expiry_date']}\n")
            else:
                print("\nNo position opened (CASH or already open).\n")

    elif args.action == "check":
        result = check_tp(tp_pct=args.take_profit)
        if result:
            print(f"\nTake-profit triggered — P&L=${result['pnl']:+,.2f}\n")
        else:
            s = status()
            print(f"\nNo TP yet. Unrealized: ${s.get('unrealized_pnl', 0):+,.2f}\n")

    elif args.action == "settle":
        result = settle_expiry()
        if result:
            print(f"\nSettled — P&L=${result['pnl']:+,.2f}  outcome={result['outcome']}\n")
        else:
            print("\nNothing to settle.\n")

    elif args.action == "backfill-alpaca":
        state = _load()
        pos   = state.get("open_position")
        if pos is None:
            print("\nNo open position to backfill.\n")
        elif pos.get("alpaca_short_symbol"):
            print(f"\nAlpaca orders already placed: {pos['alpaca_short_symbol']} / {pos['alpaca_long_symbol']}\n")
        else:
            print(f"\nPlacing Alpaca orders for existing {pos['strategy']} position...")
            _try_alpaca_open(pos)
            _save(state)
            ss = pos.get("alpaca_short_symbol", "FAILED")
            ls = pos.get("alpaca_long_symbol",  "FAILED")
            print(f"Short leg: {ss}")
            print(f"Long  leg: {ls}\n")

    elif args.action == "reset":
        cap = args.capital or 10_000.0
        yn  = input(f"Reset paper account to ${cap:,.2f}? [y/N] ").strip().lower()
        if yn == "y":
            _save({
                "account_value": cap,
                "open_position": None,
                "closed_trades": [],
                "total_pnl": 0.0,
                "n_wins": 0,
                "n_losses": 0,
            })
            print(f"Reset. State saved to {STATE_PATH}\n")
        else:
            print("Cancelled.\n")


if __name__ == "__main__":
    main()
