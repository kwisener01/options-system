"""
Close the open INTC BWB position (Jun 12 expiry) using a simultaneous multi-leg order.

KEY LESSON: Submitting legs one at a time causes Alpaca to misidentify each
SELL_TO_CLOSE as a new cash-secured short put (requiring $11,000 collateral).
Submitting all legs together in a single multi-leg order bypasses this.

Run:  python close_bwb.py [--dry-run]
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.live.alpaca_options import _trading, get_mid_price

DRY_RUN = "--dry-run" in sys.argv

# BWB legs: (occ_symbol, side, ratio_qty, position_intent_str)
LEGS = [
    ("INTC260612P00113000", "sell", 1, "sell_to_close"),
    ("INTC260612P00110000", "buy",  2, "buy_to_close"),
    ("INTC260612P00105000", "sell", 1, "sell_to_close"),
]

UNDERLYING = "INTC"


def run():
    from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, PositionIntent

    print("\n=== INTC BWB CLOSE (multi-leg) ===")
    print(f"Mode: {'DRY RUN' if DRY_RUN else '*** LIVE ***'}\n")

    acct = _trading().get_account()
    bp   = float(getattr(acct, "options_buying_power", 0) or 0)
    cash = float(getattr(acct, "cash", 0) or 0)
    print(f"Options buying power : ${bp:,.2f}")
    print(f"Cash                 : ${cash:,.2f}\n")

    net_debit = 0.0
    leg_objects = []

    for symbol, side, ratio_qty, intent_str in LEGS:
        mid = get_mid_price(symbol)
        if mid is None:
            print(f"  [WARN] No quote for {symbol} — cannot price order")
            continue

        limit = round(mid * 1.05, 2) if side == "buy" else round(mid * 0.95, 2)
        flow  = limit * ratio_qty * 100 * (1 if side == "buy" else -1)
        net_debit += flow

        intent = (PositionIntent.BUY_TO_CLOSE if side == "buy"
                  else PositionIntent.SELL_TO_CLOSE)

        print(f"  {'BUY_TO_CLOSE ' if side == 'buy' else 'SELL_TO_CLOSE'} {symbol}  x{ratio_qty}  mid=${mid:.2f}  limit=${limit:.2f}")

        leg_objects.append(OptionLegRequest(
            symbol=symbol,
            ratio_qty=ratio_qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            position_intent=intent,
        ))

    print(f"\nEstimated net debit  : ${net_debit:,.2f}")

    if DRY_RUN:
        print("\nDry run — no orders placed.")
        return

    confirm = input("\nType YES to place multi-leg close order: ").strip()
    if confirm != "YES":
        print("Aborted.")
        return

    # Net limit price for the spread (positive = debit, negative = credit)
    net_limit = round(abs(net_debit) / 100, 2)

    try:
        req = LimitOrderRequest(
            symbol=UNDERLYING,
            qty=1,
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            limit_price=net_limit,
            legs=leg_objects,
        )
        order = _trading().submit_order(req)
        print(f"\n  [OK] Multi-leg order submitted -> order_id={order.id}")
    except Exception as e:
        print(f"\n  [FAIL] Multi-leg order: {e}")
        print("\n  Falling back to Alpaca support — provide them these exact trades:")
        for symbol, side, ratio_qty, intent_str in LEGS:
            print(f"    {intent_str.upper()} {ratio_qty}x {symbol}")

    print("\nDone. Check Alpaca dashboard for fill status.")


if __name__ == "__main__":
    run()
