"""
Close all open option legs for a given underlying ticker as a single simultaneous
multi-leg order.

Usage:
  python close_bwb.py --ticker KO
  python close_bwb.py --ticker KO --dry-run

KEY LESSON: Submitting legs one at a time causes Alpaca to misidentify each
SELL_TO_CLOSE as a new cash-secured short put (requiring collateral).
Submitting all legs together in a single multi-leg order bypasses this.
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.live.alpaca_options import _trading, get_mid_price
from src.notifications.slack_notifier import send_message


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker",  required=True, help="Underlying symbol, e.g. KO")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run():
    args   = parse_args()
    ticker = args.ticker.upper()
    dry    = args.dry_run

    from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, PositionIntent

    print(f"\n=== CLOSE OPTION SPREAD: {ticker} ===")
    print(f"Mode: {'DRY RUN' if dry else '*** LIVE ***'}\n")

    # -- Find all open option legs for this underlying -------------------------
    client    = _trading()
    positions = client.get_all_positions()
    opt_legs  = [
        p for p in positions
        if getattr(p, "asset_class", "") in ("us_option", "option")
        and p.symbol.upper().startswith(ticker)
    ]

    if not opt_legs:
        print(f"[ERROR] No open option positions found for {ticker}.")
        sys.exit(1)

    print(f"Found {len(opt_legs)} open leg(s) for {ticker}:\n")

    acct = client.get_account()
    bp   = float(getattr(acct, "options_buying_power", 0) or 0)
    cash = float(getattr(acct, "cash", 0) or 0)
    print(f"Options buying power : ${bp:,.2f}")
    print(f"Cash                 : ${cash:,.2f}\n")

    net_debit   = 0.0
    leg_objects = []
    leg_display = []

    for pos in opt_legs:
        sym    = pos.symbol
        qty    = float(pos.qty)   # positive = long, negative = short
        ratio  = int(abs(qty))
        mid    = get_mid_price(sym)
        unreal = float(getattr(pos, "unrealized_pl", 0) or 0)

        if mid is None:
            print(f"  [WARN] No quote for {sym} — skipping")
            continue

        if qty > 0:
            # Long position -> SELL_TO_CLOSE (receive credit)
            side_enum   = OrderSide.SELL
            intent_enum = PositionIntent.SELL_TO_CLOSE
            side_str    = "SELL_TO_CLOSE"
            limit       = round(mid * 0.95, 2)
            flow        = -limit * ratio * 100
        else:
            # Short position -> BUY_TO_CLOSE (pay debit)
            side_enum   = OrderSide.BUY
            intent_enum = PositionIntent.BUY_TO_CLOSE
            side_str    = "BUY_TO_CLOSE "
            limit       = round(mid * 1.05, 2)
            flow        = limit * ratio * 100

        net_debit += flow

        print(f"  {side_str} {sym}  x{ratio}  mid=${mid:.2f}  limit=${limit:.2f}  unrealizedPL=${unreal:+.2f}")

        leg_objects.append(OptionLegRequest(
            symbol=sym,
            ratio_qty=ratio,
            side=side_enum,
            position_intent=intent_enum,
        ))
        leg_display.append((side_str, sym, ratio, limit, unreal))

    if not leg_objects:
        print("\n[ERROR] Could not price any legs. Check option symbols are valid and market is open.")
        sys.exit(1)

    is_debit      = net_debit > 0
    net_per_share = round(abs(net_debit) / 100, 2)
    total_unreal  = sum(ld[4] for ld in leg_display)

    print(f"\nEstimated net {'debit' if is_debit else 'credit'} : ${net_per_share:.2f} per share")
    print(f"Total unrealized P&L : ${total_unreal:+,.2f}")

    if dry:
        print("\nDry run — no orders placed.")
        return

    confirm = input("\nType YES to place multi-leg close order: ").strip()
    if confirm != "YES":
        print("Aborted.")
        return

    try:
        req = LimitOrderRequest(
            symbol=ticker,
            qty=1,
            side=OrderSide.BUY if is_debit else OrderSide.SELL,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            limit_price=net_per_share,
            legs=leg_objects,
        )
        order = client.submit_order(req)
        print(f"\n[OK] Multi-leg close order submitted -> order_id={order.id}")

        icon     = ":green_circle:" if total_unreal >= 0 else ":red_circle:"
        legs_txt = "\n".join(
            f"  {ld[0].strip()} {ld[1]} x{ld[2]} @ ${ld[3]:.2f}" for ld in leg_display
        )
        send_message(
            f"{icon} *Closed option spread `{ticker}`*\n"
            f"{legs_txt}\n"
            f"  Net {'debit' if is_debit else 'credit'} ${net_per_share:.2f}  |  "
            f"Unrealized P&L ${total_unreal:+,.2f}\n"
            f"  Order ID: `{order.id}`"
        )
        print("Slack notification sent.")

    except Exception as e:
        print(f"\n[FAIL] Multi-leg close order: {e}")
        print("\nFallback — provide these exact trades to Alpaca support:")
        for ld in leg_display:
            print(f"  {ld[0].strip()} {ld[2]}x {ld[1]}")
        sys.exit(1)

    print("\nDone. Check Alpaca dashboard for fill status.")


if __name__ == "__main__":
    run()
