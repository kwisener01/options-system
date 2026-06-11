"""
Place a bull put credit spread as a single simultaneous multi-leg order.

Usage:
  python place_spread.py --ticker KO --short 82.5 --long 80.0 --expiry 2026-07-17
  python place_spread.py --ticker KO --short 82.5 --long 80.0 --expiry 2026-07-17 --qty 2
  python place_spread.py --ticker KO --short 82.5 --long 80.0 --expiry 2026-07-17 --dry-run

KEY: submits both legs simultaneously so Alpaca recognises it as a spread,
not two naked puts. This is the same lesson learned closing the INTC BWB.
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from datetime import date
from src.live.alpaca_options import _trading, get_mid_price, occ_symbol
from src.notifications.slack_notifier import send_message


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker",  required=True,  help="Underlying symbol, e.g. KO")
    p.add_argument("--short",   required=True,  type=float, help="Short put strike (higher)")
    p.add_argument("--long",    required=True,  type=float, help="Long put strike (lower)")
    p.add_argument("--expiry",  required=True,  help="Expiry date YYYY-MM-DD")
    p.add_argument("--qty",     default=1,      type=int,   help="Number of spreads (default 1)")
    p.add_argument("--dry-run", action="store_true", help="Show trade without placing")
    return p.parse_args()


def run():
    args = parse_args()
    dry_run = args.dry_run

    try:
        expiry = date.fromisoformat(args.expiry)
    except ValueError:
        print(f"[ERROR] Invalid expiry date: {args.expiry}  (use YYYY-MM-DD)")
        sys.exit(1)

    if args.short <= args.long:
        print(f"[ERROR] Short strike ({args.short}) must be higher than long strike ({args.long})")
        sys.exit(1)

    short_sym = occ_symbol(args.ticker, expiry, "PUT", args.short)
    long_sym  = occ_symbol(args.ticker, expiry, "PUT", args.long)
    width     = round(args.short - args.long, 2)
    dte       = (expiry - date.today()).days

    print(f"\n=== BULL PUT CREDIT SPREAD ===")
    print(f"Mode    : {'DRY RUN' if dry_run else '*** LIVE ***'}")
    print(f"Ticker  : {args.ticker.upper()}")
    print(f"Expiry  : {args.expiry}  ({dte} DTE)")
    print(f"Width   : ${width:.2f}\n")

    # -- Live quotes ----------------------------------------------------------
    short_mid = get_mid_price(short_sym)
    long_mid  = get_mid_price(long_sym)

    if short_mid is None or long_mid is None:
        print(f"[ERROR] Could not fetch quotes:")
        print(f"  Short {short_sym}: {short_mid}")
        print(f"  Long  {long_sym}:  {long_mid}")
        sys.exit(1)

    # Limit prices: collect slightly below mid on short, pay slightly above mid on long
    short_limit = round(short_mid * 0.97, 2)   # 3% below mid — realistic fill
    long_limit  = round(long_mid  * 1.03, 2)   # 3% above mid

    net_credit    = round(short_limit - long_limit, 2)
    credit_pct    = round(net_credit / width * 100, 1)
    max_profit    = round(net_credit * 100 * args.qty, 2)
    max_loss      = round((width - net_credit) * 100 * args.qty, 2)

    if net_credit <= 0:
        print(f"[ERROR] Net credit is ${net_credit:.2f} — spread is not a credit. Aborting.")
        sys.exit(1)

    if credit_pct < 50:
        print(f"[WARN] Credit/width is {credit_pct:.1f}% — below 50% threshold. Proceed carefully.")

    # -- Account check --------------------------------------------------------
    acct = _trading().get_account()
    bp   = float(getattr(acct, "options_buying_power", 0) or 0)
    required_bp = max_loss  # max risk = buying power needed

    print(f"Short put : {short_sym}")
    print(f"  Mid ${short_mid:.2f}  ->  Limit ${short_limit:.2f} (sell)")
    print(f"Long put  : {long_sym}")
    print(f"  Mid ${long_mid:.2f}  ->  Limit ${long_limit:.2f} (buy)")
    print()
    print(f"Net credit    : +${net_credit:.2f}  ({credit_pct:.1f}% of ${width:.2f} width)")
    print(f"Max profit    : +${max_profit:.2f}  per {args.qty} contract(s)")
    print(f"Max loss      : -${max_loss:.2f}  per {args.qty} contract(s)")
    print(f"Options BP    : ${bp:,.2f}  (need ${required_bp:.2f})")
    print()

    if bp < required_bp:
        print(f"[ERROR] Insufficient buying power: have ${bp:.2f}, need ${required_bp:.2f}")
        sys.exit(1)

    if dry_run:
        print("Dry run — no order placed.")
        return

    # -- Confirm --------------------------------------------------------------
    confirm = input("Type YES to place order: ").strip()
    if confirm != "YES":
        print("Aborted.")
        return

    # -- Submit multi-leg order -----------------------------------------------
    from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, PositionIntent

    legs = [
        OptionLegRequest(
            symbol=short_sym,
            ratio_qty=args.qty,
            side=OrderSide.SELL,
            position_intent=PositionIntent.SELL_TO_OPEN,
        ),
        OptionLegRequest(
            symbol=long_sym,
            ratio_qty=args.qty,
            side=OrderSide.BUY,
            position_intent=PositionIntent.BUY_TO_OPEN,
        ),
    ]

    try:
        req = LimitOrderRequest(
            symbol=args.ticker.upper(),
            qty=1,
            side=OrderSide.SELL,        # credit spread = net sell
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            limit_price=net_credit,     # net credit per share
            legs=legs,
        )
        order = _trading().submit_order(req)
        print(f"\n[OK] Order submitted -> order_id={order.id}")

        # Slack confirmation
        msg = (
            f":white_check_mark: *Trade Placed — Bull Put Spread*\n"
            f"  `{args.ticker.upper()}` SELL ${args.short}P / BUY ${args.long}P\n"
            f"  Expiry {args.expiry} ({dte} DTE)  |  Qty {args.qty}\n"
            f"  Credit +${net_credit:.2f} ({credit_pct:.1f}%)  "
            f"|  Max profit +${max_profit:.0f}  |  Max loss -${max_loss:.0f}\n"
            f"  Order ID: `{order.id}`"
        )
        send_message(msg)
        print("Slack notification sent.")

    except Exception as e:
        print(f"\n[FAIL] Order rejected: {e}")
        print("\nIf rejected for buying power, try contacting Alpaca support with:")
        print(f"  SELL_TO_OPEN {args.qty}x {short_sym}")
        print(f"  BUY_TO_OPEN  {args.qty}x {long_sym}")
        sys.exit(1)

    print("\nDone. Monitor position on Alpaca dashboard.")


if __name__ == "__main__":
    run()
