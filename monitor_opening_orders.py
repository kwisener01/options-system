"""
Monitor option orders placed during the first 15 minutes of market open (9:30-9:45 AM ET).

Alpaca paper accounts receive 15-minute delayed option quotes, so limit orders
placed at 9:30 AM are priced against 9:15 AM data. This script flags that delay
and shows whether orders filled, are still pending, or were rejected.

Usage:
    python monitor_opening_orders.py               # snapshot of today's opening window
    python monitor_opening_orders.py --watch       # live polling every 30 s until 9:45
    python monitor_opening_orders.py --date 2026-05-01   # replay a past day
    python monitor_opening_orders.py --window 30   # extend window to 30 minutes
    python monitor_opening_orders.py --all         # all option orders today (no window)
"""
import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

import pytz

sys.path.insert(0, os.path.dirname(__file__))

from src.logger import setup_logging
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY, IS_PAPER

ET = pytz.timezone("America/New_York")

# Alpaca paper accounts use 15-min delayed option quotes for pricing
PAPER_QUOTE_DELAY_MIN = 15


# -- OCC symbol parser ---------------------------------------------------------

@dataclass
class OccInfo:
    underlying: str
    expiry:     date
    opt_type:   str    # CALL | PUT
    strike:     float
    raw:        str


def parse_occ(symbol: str) -> Optional[OccInfo]:
    """
    Parse an OCC option symbol into its components.
    Format: {underlying}{YYMMDD}{C|P}{strike*1000:08d}
    Example: SPY260507P00704000 -> SPY, 2026-05-07, PUT, 704.00
    """
    try:
        i = 0
        while i < len(symbol) and not symbol[i].isdigit():
            i += 1
        if i >= len(symbol) - 14:
            return None
        underlying = symbol[:i]
        date_str   = symbol[i:i+6]
        opt_char   = symbol[i+6]
        strike_str = symbol[i+7:i+15]
        expiry     = date(2000 + int(date_str[:2]), int(date_str[2:4]), int(date_str[4:6]))
        strike     = int(strike_str) / 1000
        opt_type   = "CALL" if opt_char.upper() == "C" else "PUT"
        return OccInfo(underlying, expiry, opt_type, strike, symbol)
    except Exception:
        return None


def is_option_symbol(symbol: str) -> bool:
    return len(symbol) > 10 and any(c.isdigit() for c in symbol[3:])


# -- Alpaca client -------------------------------------------------------------

def _trading():
    from alpaca.trading.client import TradingClient
    return TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=IS_PAPER)


# -- order fetching ------------------------------------------------------------

def fetch_orders(after: datetime, until: datetime, all_today: bool = False) -> list:
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus

    client = _trading()
    req = GetOrdersRequest(
        status=QueryOrderStatus.ALL,
        after=after,
        until=until,
        limit=200,
        direction="asc",
    )
    try:
        orders = client.get_orders(req)
    except Exception as e:
        print(f"ERROR fetching orders: {e}")
        return []

    if all_today:
        return orders
    return [o for o in orders if is_option_symbol(str(o.symbol))]


# -- display helpers -----------------------------------------------------------

STATUS_ICON = {
    "filled":           "[FILLED]   ",
    "partially_filled": "[PARTIAL]  ",
    "new":              "[PENDING]  ",
    "accepted":         "[PENDING]  ",
    "pending_new":      "[PENDING]  ",
    "held":             "[PENDING]  ",
    "cancelled":        "[CANCELLED]",
    "canceled":         "[CANCELLED]",
    "rejected":         "[REJECTED] ",
    "expired":          "[EXPIRED]  ",
    "replaced":         "[REPLACED] ",
}


def _ts_et(ts) -> str:
    if ts is None:
        return "--"
    if hasattr(ts, "astimezone"):
        return ts.astimezone(ET).strftime("%H:%M:%S")
    return str(ts)


def _fill_price(order) -> str:
    fp = getattr(order, "filled_avg_price", None)
    if fp is None:
        return "--"
    return f"${float(fp):.2f}"


def _limit_price(order) -> str:
    lp = getattr(order, "limit_price", None)
    if lp is None:
        return "(market)"
    return f"${float(lp):.2f}"


def _slippage(order) -> str:
    lp = getattr(order, "limit_price", None)
    fp = getattr(order, "filled_avg_price", None)
    if lp is None or fp is None:
        return ""
    diff  = float(fp) - float(lp)
    side  = str(getattr(order, "side", "")).lower()
    slip  = diff if side == "buy" else -diff
    if abs(slip) < 0.005:
        return ""
    flag = "overpaid" if slip > 0 else "improved"
    return f"  [{flag} {abs(slip):+.2f}]"


def _filled_qty(order) -> str:
    filled = getattr(order, "filled_qty", None) or "0"
    total  = getattr(order, "qty", "?")
    return f"{filled}/{total}"


def print_order(order, delay_note: str = "") -> None:
    sym    = str(order.symbol)
    occ    = parse_occ(sym)
    status = str(getattr(order, "status", "unknown")).lower()
    icon   = STATUS_ICON.get(status, "[UNKNOWN]  ")
    side   = str(getattr(order, "side", "")).upper()
    oid    = str(getattr(order, "id", ""))[:8]

    submitted = _ts_et(getattr(order, "submitted_at", None))
    filled_at = _ts_et(getattr(order, "filled_at", None))

    print(f"\n  {icon}  {side} {sym}")
    if occ:
        print(f"              {occ.underlying} {occ.expiry} {occ.opt_type} ${occ.strike:.2f}")
    print(f"              Status   : {status}")
    print(f"              Qty      : {_filled_qty(order)} contracts")
    print(f"              Limit    : {_limit_price(order)}")
    print(f"              Fill     : {_fill_price(order)}{_slippage(order)}")
    print(f"              Submitted: {submitted}{delay_note}")
    if filled_at != "--":
        print(f"              Filled at: {filled_at}")
    if status in ("rejected", "cancelled", "canceled", "expired"):
        reason = getattr(order, "failed_at", None) or getattr(order, "cancel_requested_at", None)
        if reason:
            print(f"              Reason   : {reason}")
    print(f"              Order ID : {oid}...")


def print_summary(orders: list, window_start: datetime, window_end: datetime,
                  args) -> None:
    now_et = datetime.now(ET)

    print("\n" + "=" * 62)
    print("  OPTION ORDER MONITOR -- OPENING WINDOW")
    print("=" * 62)
    print(f"  Account  : {'PAPER (15-min delayed quotes)' if IS_PAPER else 'LIVE'}")
    print(f"  Window   : {window_start.strftime('%H:%M')} - {window_end.strftime('%H:%M')} ET")
    print(f"  Scanned  : {now_et.strftime('%H:%M:%S')} ET")

    if IS_PAPER:
        effective_quote_time = window_start - timedelta(minutes=PAPER_QUOTE_DELAY_MIN)
        print()
        print(f"  [!] PAPER DELAY: Alpaca paper option quotes are {PAPER_QUOTE_DELAY_MIN} min delayed.")
        print(f"      Orders placed at 9:30 AM used quote data from ~{effective_quote_time.strftime('%H:%M')} AM.")
        print(f"      Limit orders may stay pending until {(window_start + timedelta(minutes=PAPER_QUOTE_DELAY_MIN)).strftime('%H:%M')} ET")
        print(f"      when the live feed catches up to your limit price.")
        delay_note = f"  (quotes ~{PAPER_QUOTE_DELAY_MIN}m delayed)"
    else:
        delay_note = ""

    if not orders:
        print(f"\n  No option orders found in window.")
        print("=" * 62)
        return

    filled    = [o for o in orders if str(getattr(o, "status", "")).lower() == "filled"]
    partial   = [o for o in orders if str(getattr(o, "status", "")).lower() == "partially_filled"]
    pending   = [o for o in orders if str(getattr(o, "status", "")).lower()
                 in ("new", "accepted", "pending_new", "held")]
    cancelled = [o for o in orders if str(getattr(o, "status", "")).lower()
                 in ("cancelled", "canceled", "expired", "replaced")]
    rejected  = [o for o in orders if str(getattr(o, "status", "")).lower() == "rejected"]

    print()
    print(f"  Orders total : {len(orders)}")
    print(f"  Filled       : {len(filled)}")
    print(f"  Partial fill : {len(partial)}")
    print(f"  Pending      : {len(pending)}")
    print(f"  Cancelled    : {len(cancelled)}")
    print(f"  Rejected     : {len(rejected)}")

    groups = [
        ("FILLED",           filled),
        ("PARTIALLY FILLED", partial),
        ("PENDING",          pending),
        ("CANCELLED",        cancelled),
        ("REJECTED",         rejected),
    ]

    for label, group in groups:
        if not group:
            continue
        print(f"\n  --- {label} ({len(group)}) ---")
        for o in group:
            print_order(o, delay_note)

    if IS_PAPER and pending:
        catchup = window_start + timedelta(minutes=PAPER_QUOTE_DELAY_MIN)
        print(f"\n  [i] Pending orders waiting for delayed quotes to catch up.")
        print(f"      Expected fill window: ~{catchup.strftime('%H:%M')} ET")
        print(f"      Run with --watch to track fills in real time.")

    print("\n" + "=" * 62)


# -- watch mode ----------------------------------------------------------------

def watch_loop(window_start: datetime, window_end: datetime, args,
               poll_secs: int = 30) -> None:
    print(f"\nWatch mode: refreshing every {poll_secs}s until {window_end.strftime('%H:%M')} ET")
    print("Press Ctrl+C to stop.\n")

    while True:
        now = datetime.now(ET)
        os.system("cls" if os.name == "nt" else "clear")
        orders = fetch_orders(window_start, min(now, window_end), args.all)
        print_summary(orders, window_start, window_end, args)

        if now >= window_end:
            print("\nWindow closed.")
            break
        remaining = int((window_end - now).total_seconds())
        print(f"\n  Next refresh in {poll_secs}s  |  {remaining}s until window closes")
        try:
            time.sleep(poll_secs)
        except KeyboardInterrupt:
            print("\nStopped.")
            break


# -- entry ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Monitor option orders in the opening window")
    p.add_argument("--watch",  action="store_true",
                   help="Poll live every 30 s until the window closes")
    p.add_argument("--date",   type=str, default=None,
                   help="Date to inspect (YYYY-MM-DD, default=today)")
    p.add_argument("--window", type=int, default=15,
                   help="Window length in minutes after open (default 15)")
    p.add_argument("--all",    action="store_true",
                   help="Show all option orders today, not just the opening window")
    p.add_argument("--poll",   type=int, default=30,
                   help="Poll interval in seconds for --watch mode (default 30)")
    return p.parse_args()


def main():
    setup_logging()
    logging.getLogger().setLevel(logging.WARNING)
    args = parse_args()

    target_date = date.fromisoformat(args.date) if args.date else date.today()

    market_open = ET.localize(datetime(target_date.year, target_date.month, target_date.day,
                                       9, 30, 0))
    window_end  = market_open + timedelta(minutes=args.window)
    query_end   = (ET.localize(datetime(target_date.year, target_date.month, target_date.day,
                                        16, 30, 0)) if args.all else window_end)

    now_et = datetime.now(ET)

    if IS_PAPER and target_date == date.today():
        delay_cutoff = market_open + timedelta(minutes=PAPER_QUOTE_DELAY_MIN)
        if now_et < delay_cutoff:
            mins_left = int((delay_cutoff - now_et).total_seconds() / 60)
            print(f"\n[!] {mins_left}m until Alpaca paper quotes catch up to open prices.")
            print(f"    Opening orders may still be pending. Use --watch to track fills.\n")

    if args.watch and target_date == date.today() and now_et < window_end:
        watch_loop(market_open, window_end, args, poll_secs=args.poll)
        return

    orders = fetch_orders(market_open, query_end, args.all)
    print_summary(orders, market_open, window_end, args)


if __name__ == "__main__":
    main()
