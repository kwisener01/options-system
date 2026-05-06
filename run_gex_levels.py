"""
On-demand /ES futures battle plan from current GEX levels.

Usage:
    python run_gex_levels.py                # live scan
    python run_gex_levels.py --no-short     # long-only setups
    python run_gex_levels.py --slack        # also post to Slack
    python run_gex_levels.py --date 2026-04-30  # replay cached chain

Prints the /ES battle plan to stdout and optionally sends to Slack.
"""
import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.logger import setup_logging
from src.analysis.gex_scanner import scan
from src.analysis.futures_levels import build_battle_plan, format_battle_plan
from datetime import date


def parse_args():
    p = argparse.ArgumentParser(description="/ES futures battle plan from GEX")
    p.add_argument("--no-short",  action="store_true", help="Long-only setups")
    p.add_argument("--slack",     action="store_true", help="Post to Slack")
    p.add_argument("--date",      type=str, default=None,
                   help="Replay cached chain for this date (YYYY-MM-DD)")
    return p.parse_args()


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    args   = parse_args()

    as_of = date.fromisoformat(args.date) if args.date else None
    allow_short = not args.no_short

    print(f"\nFetching GEX data{f' for {as_of}' if as_of else ''}...")
    result = scan(as_of=as_of)
    if result is None:
        print("ERROR: GEX scan failed — market may be closed or no chain data available.")
        sys.exit(1)

    plan = build_battle_plan(result, allow_short=allow_short, as_of=as_of or date.today())
    msg  = format_battle_plan(plan)

    # Strip Slack markdown for clean terminal output
    clean = (msg
             .replace("*", "")
             .replace("`", "")
             .replace(":chess_pawn:", "")
             .replace(":pushpin:", "")
             .replace(":green_circle:", "[+GEX]")
             .replace(":red_circle:", "[-GEX]")
             .replace(":chart_with_upward_trend:", "[^]")
             .replace(":chart_with_downwards_trend:", "[v]")
             .replace(":white_circle:", "[ ]")
             .replace(":arrow_up_small:", "[^]")
             .replace(":arrow_down_small:", "[v]")
             .replace(":triangular_flag_on_post:", "")
             .replace(":large_green_circle:", "[LONG]")
             .replace(":fire:", "[HIGH]")
             .replace(":blue_circle:", "[MED]")
             .replace(":bulb:", "")
             .replace(":rotating_light:", "[!]")
             .replace(":warning:", "[!]")
             .replace(":shield:", "[SUPPORT]")
             .replace(":dart:", "[PIN]"))
    print(clean)

    if args.slack:
        try:
            from src.notifications.slack_notifier import send_message
            send_message(msg)
            print("\nPosted to Slack.")
        except Exception as e:
            print(f"\nSlack post failed: {e}")

    # Summary table
    print("\n" + "=" * 55)
    print("  /ES KEY LEVELS SUMMARY")
    print("=" * 55)
    for lv in sorted(plan.levels, key=lambda l: l.es_price, reverse=True):
        here = " <-- HERE" if abs(lv.es_price - plan.es_spot) < 5 else ""
        print(f"  {lv.es_price:>8.2f}  {lv.name:<15}{here}")
    print(f"  {'':>8}  {'(/ES current: ~' + str(plan.es_spot) + ')':}")
    print("=" * 55)

    if plan.setups:
        print("\n  SETUPS:")
        for s in plan.setups:
            print(f"    {s.direction:<5} {s.name}")
            print(f"      Entry {s.es_entry:.2f}  Stop {s.es_stop:.2f}  Target {s.es_target:.2f}")
            print(f"      Risk: /ES ${s.es_risk_per_contract:,.0f}  /MES ${s.mes_risk_per_contract:,.0f}  R:R {s.rr:.1f}x  [{s.confidence}]")
    else:
        print("\n  No setups in range — monitor key levels above")

    print()


if __name__ == "__main__":
    main()
