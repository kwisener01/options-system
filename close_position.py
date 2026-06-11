"""
Close a single stock or ETF position at market.

Usage:
  python close_position.py --ticker VBR
  python close_position.py --ticker VBR --dry-run
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.live.alpaca_options import _trading
from src.notifications.slack_notifier import send_message


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker",  required=True, help="Symbol to close, e.g. VBR")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run():
    args  = parse_args()
    sym   = args.ticker.upper()
    dry   = args.dry_run

    client = _trading()

    # Find position
    try:
        positions = client.get_all_positions()
        pos = next((p for p in positions if p.symbol == sym), None)
    except Exception as e:
        print(f"[ERROR] Could not fetch positions: {e}")
        sys.exit(1)

    if pos is None:
        print(f"[ERROR] No open position found for {sym}")
        sys.exit(1)

    qty      = float(pos.qty)
    mkt_val  = float(getattr(pos, "market_value",    0) or 0)
    unreal   = float(getattr(pos, "unrealized_pl",   0) or 0)
    pct      = float(getattr(pos, "unrealized_plpc", 0) or 0) * 100
    avg_cost = float(getattr(pos, "avg_entry_price", 0) or 0)
    cur_px   = float(getattr(pos, "current_price",   0) or 0)

    print(f"\n=== CLOSE POSITION ===")
    print(f"Mode     : {'DRY RUN' if dry else '*** LIVE ***'}")
    print(f"Symbol   : {sym}")
    print(f"Qty      : {qty:.4f} shares")
    print(f"Avg cost : ${avg_cost:.2f}")
    print(f"Current  : ${cur_px:.2f}")
    print(f"Mkt value: ${mkt_val:,.2f}")
    print(f"Unreal PL: ${unreal:+,.2f}  ({pct:+.1f}%)\n")

    if dry:
        print("Dry run — no order placed.")
        return

    confirm = input(f"Type YES to sell all {qty:.4f} shares of {sym}: ").strip()
    if confirm != "YES":
        print("Aborted.")
        return

    try:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        req = MarketOrderRequest(
            symbol=sym,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order = client.submit_order(req)
        print(f"\n[OK] Market sell submitted -> order_id={order.id}")

        icon = ":green_circle:" if unreal >= 0 else ":red_circle:"
        send_message(
            f"{icon} *Closed `{sym}`*\n"
            f"  Qty {qty:.4f}  |  Mkt value ${mkt_val:,.2f}\n"
            f"  P&L ${unreal:+,.2f} ({pct:+.1f}%)\n"
            f"  Order ID: `{order.id}`"
        )
        print("Slack notification sent.")

    except Exception as e:
        print(f"\n[FAIL] Order rejected: {e}")
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    run()
