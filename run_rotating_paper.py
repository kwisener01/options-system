"""
Intraday rotating options paper trader — CLI.

Commands
--------
  python run_rotating_paper.py status          # show P&L and open position
  python run_rotating_paper.py tick            # run one evaluation cycle now
  python run_rotating_paper.py run             # loop every 30 min until Ctrl-C
  python run_rotating_paper.py reset           # wipe state and start fresh
  python run_rotating_paper.py reset --capital 2000 --symbol XSP

Scheduler hook (add to main.py or cron):
  Every 30 min during market hours  ->  python run_rotating_paper.py tick
  Friday 3:45 PM                    ->  position force-closes automatically via tick
"""
import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from src.logger import setup_logging
from src.live.rotating_paper_trader import (
    detect_symbol, tick, status, reset,
    _DEFAULT_ACCOUNT,
)


def fmt_status(s: dict):
    w,  l  = s["n_wins"], s["n_losses"]
    wr     = w / (w + l) if (w + l) else 0
    print()
    print("=" * 58)
    print("  ROTATING OPTIONS PAPER TRADER")
    print("=" * 58)
    print(f"  Symbol         : {s['symbol']}")
    print(f"  Account value  : ${s['account_value']:>12,.2f}")
    print(f"  Total P&L      : ${s['total_pnl']:>+12,.2f}")
    print(f"  Record         :  {w}W / {l}L  ({wr:.0%} win rate)")
    print(f"  Day trades     :  {s['day_trades_used']}/3 this week")

    pos = s.get("open_position")
    if pos:
        print()
        print("  OPEN POSITION:")
        print(f"    Strategy  : {pos['strategy']}")
        print(f"    Entry     : {pos['entry_time'][:16]}  "
              f"SPY/XSP=${pos['spot_entry']:.2f}  VIX={pos['vix_entry']:.1f}")
        print(f"    IV Rank   : {pos.get('iv_rank_entry', '?')}")
        print(f"    Legs      :")
        for l in pos["legs"]:
            sym = l.get("occ_symbol") or "(paper)"
            print(f"              {l['action'].upper():4s} {l['qty']}x  "
                  f"{l['kind']:4s}  K={l['strike']:.1f}  "
                  f"exp={l['expiry']}  entry=${l['entry_price']:.3f}  {sym}")
        nc = pos["net_credit_total"]
        mr = pos["max_risk_total"]
        print(f"    Net credit: ${nc:+,.2f}   Max risk: ${mr:,.2f}")
        if s.get("unrealized_pnl") is not None:
            print(f"    Unrealized: ${s['unrealized_pnl']:+,.2f}")
    else:
        print()
        print("  No open position.")
    print("=" * 58)
    print()


def parse_args():
    p = argparse.ArgumentParser(description="Rotating options paper trader")
    p.add_argument("action", nargs="?", default="status",
                   choices=["status", "tick", "run", "reset"])
    p.add_argument("--capital", type=float, default=_DEFAULT_ACCOUNT)
    p.add_argument("--symbol",  type=str,   default=None,
                   help="XSP or SPY (default: auto-detect)")
    p.add_argument("--interval", type=int,  default=30,
                   help="Minutes between ticks in 'run' mode (default 30)")
    return p.parse_args()


def main():
    setup_logging()
    args = parse_args()
    log  = logging.getLogger(__name__)

    if args.action == "status":
        fmt_status(status())

    elif args.action == "tick":
        sym    = args.symbol or detect_symbol()
        result = tick(symbol=sym)
        print(f"\nAction: {result['action']}")
        if "strategy" in result:
            print(f"Strategy: {result['strategy']}")
        if result.get("pnl_unreal") is not None:
            print(f"Unrealized P&L: ${result['pnl_unreal']:+,.2f}")
        acct = result.get('account_value')
        print(f"Account: ${acct:,.2f}" if isinstance(acct, (int, float)) else "Account: n/a")
        if "iv_rank" in result:
            print(f"IV Rank: {result['iv_rank']:.0f}  Trend: {result.get('trend', 0):+.2f}")
        print()

    elif args.action == "run":
        sym      = args.symbol or detect_symbol()
        interval = args.interval * 60
        print(f"\nRunning rotating paper trader on {sym}, every {args.interval} min.")
        print("Press Ctrl-C to stop.\n")
        while True:
            try:
                result = tick(symbol=sym)
                print(f"[{_now_et()}]  action={result.get('action'):<14}  "
                      f"account=${result.get('account_value', 0):,.2f}  "
                      f"unreal=${result.get('pnl_unreal', 0):+,.2f}")
                time.sleep(interval)
            except KeyboardInterrupt:
                print("\nStopped.")
                break
            except Exception as e:
                log.error("Tick error: %s", e)
                time.sleep(60)

    elif args.action == "reset":
        sym = args.symbol or detect_symbol()
        yn  = input(f"Reset to ${args.capital:,.2f} on {sym}? [y/N] ").strip().lower()
        if yn == "y":
            reset(args.capital, sym)
            print(f"Reset complete. Symbol: {sym}  Capital: ${args.capital:,.2f}\n")
        else:
            print("Cancelled.\n")


def _now_et():
    import pytz
    from datetime import datetime
    return datetime.now(pytz.timezone("America/New_York")).strftime("%H:%M")


if __name__ == "__main__":
    main()
