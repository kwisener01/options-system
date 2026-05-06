"""
Manual rebalance trigger — run anytime to see ML signals and optionally execute.

Usage:
    python rebalance_now.py               # dry run (show orders, don't execute)
    python rebalance_now.py --execute     # place real paper trades
    python rebalance_now.py --top-n 5    # concentrate into 5 positions
    python rebalance_now.py --min-score 0.6  # only buy if ML score >= 0.6
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.logger import setup_logging
from src.client.alpaca_client import AlpacaClient
from src.analysis.macro_scanner import get_macro_snapshot
from src.signals.daily_runner import run_daily_selection
from config.settings import ML_TOP_N, ACCOUNT_RISK_PCT, IS_PAPER


def parse_args():
    p = argparse.ArgumentParser(description="Manual ML rebalance")
    p.add_argument("--execute", action="store_true", help="Place actual orders (default: dry run)")
    p.add_argument("--top-n", type=int, default=ML_TOP_N, help="Number of positions to hold")
    p.add_argument("--min-score", type=float, default=0.0, help="Minimum ML score to buy (0-1)")
    p.add_argument("--cash-pct", type=float, default=0.95, help="Fraction of buying power to deploy")
    return p.parse_args()


def fmt_currency(v: float) -> str:
    return f"${v:,.2f}"


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    args = parse_args()

    mode = "PAPER" if IS_PAPER else "LIVE"
    run_mode = "DRY RUN" if not args.execute else f"EXECUTING on {mode}"
    logger.info("=== REBALANCE NOW [%s] ===", run_mode)

    # ── Macro check ──────────────────────────────────────────────────────────
    print("\nFetching macro snapshot...")
    macro = get_macro_snapshot()
    print(f"  Regime : {macro.regime}")
    print(f"  Summary: {macro.summary}")

    if macro.regime == "BEAR":
        print("\n  [BEAR REGIME] — ML selector returns no buys. All positions will be closed.")

    # ── Account status ───────────────────────────────────────────────────────
    print("\nConnecting to Alpaca...")
    client = AlpacaClient()
    portfolio_value = client.get_portfolio_value()
    buying_power = client.get_buying_power()
    current_positions = client.get_positions()

    print(f"  Account       : {mode}")
    print(f"  Portfolio     : {fmt_currency(portfolio_value)}")
    print(f"  Buying Power  : {fmt_currency(buying_power)}")
    print(f"  Open Positions: {len(current_positions)}")

    if current_positions:
        print()
        print("  Current Holdings:")
        for pos in current_positions:
            pnl = float(pos.unrealized_pl)
            pnl_pct = float(pos.unrealized_plpc) * 100
            print(f"    {pos.symbol:<6}  qty={float(pos.qty):.4f}  "
                  f"value={fmt_currency(float(pos.market_value))}  "
                  f"P&L={fmt_currency(pnl)} ({pnl_pct:+.1f}%)")

    # ── ML selection ─────────────────────────────────────────────────────────
    print("\nRunning ML selection (this may take a few minutes)...")
    selected, scores = run_daily_selection()

    # Apply min-score filter
    if args.min_score > 0:
        selected = [s for s in selected if scores.get(s, 0) >= args.min_score]
        scores = {s: v for s, v in scores.items() if s in selected}
        print(f"  After min-score={args.min_score:.2f} filter: {len(selected)} symbols")

    # Apply top-n cap
    if len(selected) > args.top_n:
        selected = selected[: args.top_n]
        scores = {s: scores[s] for s in selected}

    # ── Build order plan ─────────────────────────────────────────────────────
    current_symbols = {p.symbol for p in current_positions}
    sells = [p for p in current_positions if p.symbol not in selected]
    buys = [s for s in selected if s not in current_symbols]

    # Score-weighted allocation
    total_alloc = buying_power * args.cash_pct
    # After sells, buying power increases — approximate as: existing_bp + sell_proceeds
    approx_sell_proceeds = sum(float(p.market_value) for p in sells)
    total_alloc = min((buying_power + approx_sell_proceeds) * args.cash_pct,
                      portfolio_value * ACCOUNT_RISK_PCT * ML_TOP_N)
    score_sum = sum(scores.get(s, 1.0) for s in buys)

    print()
    print("=" * 56)
    print("  REBALANCE PLAN")
    print("=" * 56)

    if not sells and not buys:
        print("  No changes needed — portfolio matches ML selection.")
    else:
        if sells:
            print(f"\n  CLOSE ({len(sells)} positions):")
            for pos in sells:
                print(f"    SELL  {pos.symbol:<6}  {fmt_currency(float(pos.market_value))}")

        if buys:
            print(f"\n  OPEN ({len(buys)} positions, total ~{fmt_currency(total_alloc)}):")
            for sym in buys:
                weight = scores.get(sym, 1.0) / score_sum if score_sum > 0 else 1.0 / len(buys)
                alloc = total_alloc * weight
                sc = scores.get(sym, 0)
                bar = "#" * int(sc * 20)
                print(f"    BUY   {sym:<6}  {fmt_currency(alloc)}  score={sc:.3f}  {bar}")

    print()
    print(f"  Full ML scores (top selection):")
    for sym in selected:
        sc = scores.get(sym, 0)
        held = " [held]" if sym in current_symbols else ""
        print(f"    {sym:<6}  {sc:.3f}{held}")

    print("=" * 56)

    if not args.execute:
        print("\n  [DRY RUN] — no orders placed. Add --execute to trade.\n")
        return

    # ── Execute ───────────────────────────────────────────────────────────────
    print("\n  Executing...")

    for pos in sells:
        try:
            client.sell(pos.symbol)
            print(f"    SOLD  {pos.symbol}")
        except Exception as e:
            print(f"    SELL {pos.symbol} FAILED: {e}")

    for sym in buys:
        weight = scores.get(sym, 1.0) / score_sum if score_sum > 0 else 1.0 / len(buys)
        alloc = round(total_alloc * weight, 2)
        if alloc < 10:
            print(f"    SKIP  {sym} (alloc {fmt_currency(alloc)} too small)")
            continue
        try:
            client.buy(sym, alloc)
            print(f"    BOUGHT {sym}  {fmt_currency(alloc)}")
        except Exception as e:
            print(f"    BUY {sym} FAILED: {e}")

    print("\n  Done.\n")
    logger.info("=== REBALANCE COMPLETE ===")


if __name__ == "__main__":
    main()
