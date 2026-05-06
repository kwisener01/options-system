"""
Alpaca options order execution layer.

Places real option legs on the Alpaca paper/live account so positions
appear in the Alpaca dashboard and portfolio P&L.

Each spread is two single-leg limit orders:
  1. SELL the short put/call  (collect premium)
  2. BUY  the long  put/call  (define max risk)

Limit prices are set at mid-price from the live option quote.
Orders are IOC (immediate-or-cancel) during market hours, so if the
mid isn't fillable we fall back to aggressive limit (ask for buys,
bid for sells) and log a warning.
"""
import logging
import math
from datetime import date, timedelta
from typing import Optional

from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY, IS_PAPER

logger = logging.getLogger(__name__)


# ── OCC symbol builder ────────────────────────────────────────────────────────

def occ_symbol(underlying: str, expiry: date, option_type: str, strike: float) -> str:
    """
    Build an OCC option symbol.
    e.g. SPY 704.50 Put exp 2026-05-09 → 'SPY260509P00704500'
    """
    strike_int = round(strike * 1000)
    return (
        f"{underlying.upper()}"
        f"{expiry.strftime('%y%m%d')}"
        f"{'P' if option_type.upper() == 'PUT' else 'C'}"
        f"{strike_int:08d}"
    )


# ── Alpaca clients (lazy init) ────────────────────────────────────────────────

_trading_client = None
_data_client    = None


def _trading():
    global _trading_client
    if _trading_client is None:
        from alpaca.trading.client import TradingClient
        _trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=IS_PAPER)
    return _trading_client


def _data():
    global _data_client
    if _data_client is None:
        try:
            from alpaca.data.historical.option import OptionHistoricalDataClient
            _data_client = OptionHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        except ImportError:
            from alpaca.data.historical import OptionHistoricalDataClient
            _data_client = OptionHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    return _data_client


# ── contract lookup ───────────────────────────────────────────────────────────

def find_contract(underlying: str, expiry: date, option_type: str,
                  strike: float, tolerance: float = 1.0) -> Optional[str]:
    """
    Search Alpaca's option chain and return the OCC symbol of the best
    matching contract, or None if not found.
    """
    try:
        from alpaca.trading.requests import GetOptionContractsRequest
        from alpaca.trading.enums import ContractType
        ct = ContractType.PUT if option_type.upper() == "PUT" else ContractType.CALL
        req = GetOptionContractsRequest(
            underlying_symbols=[underlying],
            expiration_date_gte=str(expiry - timedelta(days=1)),
            expiration_date_lte=str(expiry + timedelta(days=1)),
            type=ct,
            strike_price_gte=str(strike - tolerance),
            strike_price_lte=str(strike + tolerance),
        )
        resp = _trading().get_option_contracts(req)
        # SDK returns OptionContractsResponse; unwrap to list
        contracts = getattr(resp, "option_contracts", None) or resp
        if not contracts:
            logger.warning("No contracts found for %s %s %s %.2f", underlying, expiry, option_type, strike)
            return None

        def _strike(c):
            v = c.get("strike_price") if isinstance(c, dict) else getattr(c, "strike_price", None)
            return float(v) if v is not None else 0.0

        def _sym(c):
            return c.get("symbol") if isinstance(c, dict) else getattr(c, "symbol", None)

        best = min(contracts, key=lambda c: abs(_strike(c) - strike))
        sym  = _sym(best)
        logger.info("Found contract: %s (strike=%.2f expiry=%s)", sym, _strike(best), expiry)
        return sym
    except Exception as e:
        logger.error("Contract lookup failed: %s", e)
        # Fall back to synthesized OCC symbol
        sym = occ_symbol(underlying, expiry, option_type, strike)
        logger.warning("Using synthesized symbol: %s", sym)
        return sym


# ── live quote ────────────────────────────────────────────────────────────────

def get_mid_price(symbol: str) -> Optional[float]:
    """Return the mid-price of a live option quote, or None on failure."""
    try:
        try:
            from alpaca.data.requests import OptionLatestQuoteRequest
        except ImportError:
            from alpaca.data.requests import OptionSnapshotRequest as OptionLatestQuoteRequest

        req = OptionLatestQuoteRequest(symbol_or_symbols=[symbol])
        quotes = _data().get_option_latest_quote(req)
        q = quotes.get(symbol)
        if q and q.bid_price is not None and q.ask_price is not None:
            return (float(q.bid_price) + float(q.ask_price)) / 2
    except Exception as e:
        logger.warning("Quote fetch failed for %s: %s", symbol, e)
    return None


# ── order placement ───────────────────────────────────────────────────────────

def place_single_leg(symbol: str, side: str, qty: int,
                     limit_price: float) -> Optional[str]:
    """
    Place a single-leg limit option order.
    side: 'buy' | 'sell'
    Returns order_id or None on failure.
    """
    try:
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce, OrderType

        req = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            limit_price=round(limit_price, 2),
        )
        order = _trading().submit_order(req)
        logger.info("%s %s x%d @ $%.2f -> order_id=%s", side.upper(), symbol, qty, limit_price, order.id)
        return str(order.id)
    except Exception as e:
        logger.error("Order failed %s %s: %s", side, symbol, e)
        return None


def place_spread(short_symbol: str, long_symbol: str,
                 qty: int, short_credit: float, long_debit: float) -> dict:
    """
    Place both legs of a credit spread.
    Returns dict with order IDs for both legs.
    """
    results = {"short_order_id": None, "long_order_id": None, "success": False}

    # Get live quotes — fall back to BS price if quote unavailable
    short_mid = get_mid_price(short_symbol) or short_credit
    long_mid  = get_mid_price(long_symbol)  or long_debit

    # Sell the short leg first (reduces capital risk if long doesn't fill)
    results["short_order_id"] = place_single_leg(short_symbol, "sell", qty,
                                                  limit_price=max(short_mid * 0.95, 0.01))
    # Buy the long leg (protection — always fill even if slightly above mid)
    results["long_order_id"]  = place_single_leg(long_symbol, "buy", qty,
                                                  limit_price=long_mid * 1.05)

    results["success"] = (results["short_order_id"] is not None and
                          results["long_order_id"]  is not None)
    return results


def close_spread(short_symbol: str, long_symbol: str, qty: int,
                 short_bid: Optional[float] = None,
                 long_ask:  Optional[float] = None) -> dict:
    """
    Close both legs of an open spread (buy back short, sell long).
    """
    results = {"short_order_id": None, "long_order_id": None, "success": False}

    short_mid = get_mid_price(short_symbol)
    long_mid  = get_mid_price(long_symbol)

    # Buy back the short (pay to close)
    if short_mid is not None and short_mid > 0.01:
        results["short_order_id"] = place_single_leg(short_symbol, "buy", qty,
                                                      limit_price=short_mid * 1.05)
    # Sell the long (collect remaining value)
    if long_mid is not None and long_mid > 0.01:
        results["long_order_id"] = place_single_leg(long_symbol, "sell", qty,
                                                     limit_price=long_mid * 0.95)

    results["success"] = results["short_order_id"] is not None
    return results


def get_option_positions() -> list:
    """Return all open option positions from Alpaca."""
    try:
        positions = _trading().get_all_positions()
        return [p for p in positions if getattr(p, "asset_class", "") == "us_option"
                or len(p.symbol) > 10]   # option symbols are long
    except Exception as e:
        logger.error("Failed to fetch option positions: %s", e)
        return []


def cancel_all_option_orders():
    """Cancel any open option orders (cleanup)."""
    try:
        orders = _trading().get_orders()
        for o in orders:
            if len(o.symbol) > 10:  # option symbol
                _trading().cancel_order_by_id(o.id)
                logger.info("Cancelled option order %s", o.id)
    except Exception as e:
        logger.error("Cancel orders failed: %s", e)
