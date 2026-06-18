#!/usr/bin/env python3
"""
CLI tool — dump trade journal + strategy attribution to console.

Usage:
    python run_attribution.py                # full report
    python run_attribution.py --month 2026-06  # filter to a specific month
    python run_attribution.py --json         # machine-readable output
"""
import argparse
import json
import sys

from src.analysis.trade_journal import (
    get_trades,
    monthly_summary,
    strategy_attribution,
)


def _print_trades(trades: list[dict], limit: int = 20):
    if not trades:
        print("  (no closed trades)")
        return

    for t in trades[-limit:][::-1]:
        icon = "W" if t["pnl"] > 0 else "L"
        strat = t.get("strategy_clean", "?")
        hold = f"{t['hold_days']}d" if t["hold_days"] is not None else "?"
        ror = f"{t['return_on_risk']*100:+.1f}%" if t.get("return_on_risk") else ""
        short_s = t.get("short_strike") or t.get("put_short") or "?"
        long_s = t.get("long_strike") or t.get("put_long") or "?"
        sym = t.get("symbol", t.get("underlying", "?"))
        print(
            f"  [{icon}] {t.get('close_date','?'):10s}  {strat:24s}  "
            f"{sym:5s} ${short_s}/{long_s}  {hold:>4s}  "
            f"P&L ${t['pnl']:+8,.2f}  {ror}"
        )


def _print_attribution(rows: list[dict]):
    if not rows:
        print("  (no data)")
        return

    print(f"  {'Strategy':24s} {'Trades':>6s} {'W/L':>7s} {'Win%':>5s} "
          f"{'Total P&L':>10s} {'Avg P&L':>9s} {'Best':>9s} {'Worst':>9s} "
          f"{'Avg Hold':>8s} {'Contrib':>7s}")
    print(f"  {'-'*24} {'-'*6} {'-'*7} {'-'*5} "
          f"{'-'*10} {'-'*9} {'-'*9} {'-'*9} {'-'*8} {'-'*7}")

    for r in rows:
        print(
            f"  {r['strategy_clean']:24s} {r['trades']:6d} "
            f"{r['wins']}W/{r['losses']}L {r['win_rate']*100:4.0f}% "
            f"${r['total_pnl']:+9,.2f} ${r['avg_pnl']:+8,.2f} "
            f"${r['best_trade']:+8,.2f} ${r['worst_trade']:+8,.2f} "
            f"{r['avg_hold_days']:7.1f}d {r['contribution']*100:+5.0f}%"
        )


def main():
    parser = argparse.ArgumentParser(description="Trade journal & strategy attribution")
    parser.add_argument("--month", help="Filter to month (YYYY-MM)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("-n", type=int, default=20, help="Number of recent trades to show")
    args = parser.parse_args()

    if args.json:
        json.dump(monthly_summary() if not args.month else {
            "trades": get_trades(args.month),
            "attribution": strategy_attribution(args.month),
        }, sys.stdout, indent=2, default=str)
        print()
        return

    trades = get_trades(args.month)
    attr = strategy_attribution(args.month)
    all_trades = get_trades()

    scope = f"month {args.month}" if args.month else "all time"
    print(f"\n{'='*80}")
    print(f"  TRADE JOURNAL & STRATEGY ATTRIBUTION — {scope.upper()}")
    print(f"{'='*80}\n")

    # Totals
    total_pnl = sum(t["pnl"] for t in trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = len(trades) - wins
    wr = (wins / len(trades) * 100) if trades else 0
    print(f"  Trades: {len(trades)}   Wins: {wins}   Losses: {losses}   "
          f"Win rate: {wr:.0f}%   Total P&L: ${total_pnl:+,.2f}\n")

    # Recent trades
    print(f"  RECENT TRADES (last {args.n})")
    print(f"  {'-'*76}")
    _print_trades(trades, args.n)

    # Attribution
    print(f"\n  STRATEGY ATTRIBUTION")
    print(f"  {'-'*76}")
    _print_attribution(attr)

    # Verdict
    if attr:
        print(f"\n  VERDICT")
        print(f"  {'-'*76}")
        for r in attr:
            if r["trades"] < 3:
                tag = "TOO FEW TRADES"
            elif r["win_rate"] >= 0.60 and r["total_pnl"] > 0:
                tag = "KEEP"
            elif r["win_rate"] >= 0.45 and r["total_pnl"] > 0:
                tag = "WATCH — marginal win rate"
            elif r["total_pnl"] > 0 and r["win_rate"] < 0.45:
                tag = "WATCH — low win rate, net positive on outliers"
            elif r["total_pnl"] <= 0 and r["trades"] >= 5:
                tag = "STOP — negative total P&L"
            elif r["total_pnl"] <= 0:
                tag = "REVIEW — underwater, small sample"
            else:
                tag = "REVIEW"
            print(f"  {r['strategy_clean']:24s}  -> {tag}")

    print()


if __name__ == "__main__":
    main()
