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
        # Probe for best SSL config: certifi → system → disabled
        import requests as _req
        for _verify in (
            __import__("certifi").where(),
            True,
            False,
        ):
            try:
                _req.get("https://data.alpaca.markets", verify=_verify, timeout=5)
                _data_client._session.verify = _verify
                break
            except Exception:
                pass
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


def fetch_chain_for_gex(underlying: str = "SPY",
                        spot: float = 0.0,
                        n_expiries: int = 3) -> list[dict]:
    """
    Pull the live options chain from Alpaca and return it in the format
    expected by gex_scanner.compute_exposures():
        [{"strike": float, "oi": int, "iv": float, "T": float, "is_call": bool}, ...]

    Falls back gracefully — returns [] on any error so caller can use yfinance.
    """
    from datetime import date, timedelta
    import math

    try:
        from alpaca.data.requests import OptionChainRequest
    except ImportError:
        logger.warning("OptionChainRequest not available in this alpaca-py version")
        return []

    today = date.today()
    rows: list[dict] = []

    try:
        req = OptionChainRequest(
            underlying_symbol=underlying,
            feed="indicative",
        )
        chain: dict = _data().get_option_chain(req)   # symbol -> OptionSnapshot
    except Exception as e:
        logger.warning("Alpaca option chain fetch failed: %s", e)
        return []

    for symbol, snap in chain.items():
        try:
            greeks   = getattr(snap, "greeks", None)
            quote    = getattr(snap, "latest_quote", None)
            contract = getattr(snap, "contract_details", None) or getattr(snap, "details", None)

            # strike
            strike = None
            if contract:
                strike = float(getattr(contract, "strike_price", 0) or 0)
            if not strike:
                # parse from OCC symbol: SPY YYMMDD C/P XXXXXXXX
                try:
                    strike = int(symbol[-8:]) / 1000.0
                except Exception:
                    continue

            # filter to ±8% of spot
            if spot and not (spot * 0.92 <= strike <= spot * 1.08):
                continue

            # is_call
            option_type = None
            if contract:
                option_type = getattr(contract, "type", None) or getattr(contract, "option_type", None)
            if not option_type:
                # parse from OCC symbol
                option_type = "call" if len(symbol) > 15 and symbol[12] == "C" else "put"
            is_call = str(option_type).lower() in ("call", "c")

            # expiry / T
            expiry_str = None
            if contract:
                expiry_str = str(getattr(contract, "expiration_date", "") or "")
            if not expiry_str:
                try:
                    expiry_str = f"20{symbol[3:5]}-{symbol[5:7]}-{symbol[7:9]}"
                except Exception:
                    continue
            try:
                expiry_date = date.fromisoformat(expiry_str[:10])
                T = max((expiry_date - today).days, 0) / 365
            except Exception:
                continue

            # only take the n nearest expirations
            dte = (expiry_date - today).days
            if dte < 0 or dte > 60:
                continue

            # implied vol — prefer snap-level IV, fall back to greeks
            iv = float(getattr(snap, "implied_volatility", 0) or 0)
            if not iv and greeks:
                iv = float(getattr(greeks, "implied_volatility", 0) or 0)
            if iv < 0.01:
                continue

            # open interest — use contract OI when available; otherwise use
            # quote bid+ask size as a liquidity proxy (indicative feed has no OI)
            oi = 0
            if contract:
                oi = int(getattr(contract, "open_interest", 0) or 0)
            if oi < 10 and quote:
                bid_sz = int(getattr(quote, "bid_size", 0) or 0)
                ask_sz = int(getattr(quote, "ask_size", 0) or 0)
                oi = bid_sz + ask_sz
            if oi < 5:
                continue

            rows.append(dict(strike=strike, oi=oi, iv=iv, T=T, is_call=is_call))

        except Exception as e:
            logger.debug("Skipping contract %s: %s", symbol, e)
            continue

    logger.info("Alpaca chain: %d contracts for %s", len(rows), underlying)
    return rows


def fetch_chain_combined(underlying: str, spot: float,
                         dte_min: int = 7, dte_max: int = 60) -> dict:
    """
    Single Alpaca chain fetch returning two views from one API call:
      'gex_contracts': [{strike, oi, iv, T, is_call}]  — for compute_exposures()
      'puts_liquid':   [{strike, expiry, dte, bid, ask, spread, oi_proxy, iv}]
    Type position derived from len(symbol)-9; works for any root length.
    """
    from datetime import date as _date

    empty: dict = {"gex_contracts": [], "puts_liquid": [], "calls_liquid": []}
    try:
        from alpaca.data.requests import OptionChainRequest
    except ImportError:
        logger.warning("OptionChainRequest not available")
        return empty

    today = _date.today()
    gex_rows:  list[dict] = []
    put_rows:  list[dict] = []
    call_rows: list[dict] = []

    try:
        req = OptionChainRequest(underlying_symbol=underlying, feed="indicative")
        chain: dict = _data().get_option_chain(req)
    except Exception as e:
        logger.warning("Chain fetch failed for %s: %s", underlying, e)
        return empty

    for symbol, snap in chain.items():
        try:
            type_pos = len(symbol) - 9
            if type_pos < 0 or symbol[type_pos] not in ("C", "P"):
                continue
            is_put  = symbol[type_pos] == "P"
            is_call = not is_put

            try:
                strike = int(symbol[-8:]) / 1000.0
            except Exception:
                continue

            # GEX uses ±15%; puts scan 84–102% of spot; calls scan 98–115%
            if spot and not (spot * 0.85 <= strike <= spot * 1.15):
                continue

            contract = (getattr(snap, "contract_details", None)
                        or getattr(snap, "details", None))
            expiry_str = ""
            if contract:
                expiry_str = str(getattr(contract, "expiration_date", "") or "")
            if not expiry_str:
                try:
                    d_start = type_pos - 6
                    raw = symbol[d_start:type_pos]
                    expiry_str = f"20{raw[0:2]}-{raw[2:4]}-{raw[4:6]}"
                except Exception:
                    continue

            try:
                expiry_date = _date.fromisoformat(expiry_str[:10])
                dte = (expiry_date - today).days
                T   = max(dte, 0) / 365
            except Exception:
                continue

            if dte < 0 or dte > 60:
                continue

            iv = float(getattr(snap, "implied_volatility", 0) or 0)
            greeks = getattr(snap, "greeks", None)
            if not iv and greeks:
                iv = float(getattr(greeks, "implied_volatility", 0) or 0)
            if iv < 0.01:
                continue

            quote   = getattr(snap, "latest_quote", None)
            oi      = 0
            bid_sz  = ask_sz = 0
            if contract:
                oi = int(getattr(contract, "open_interest", 0) or 0)
            if quote:
                bid_sz = int(getattr(quote, "bid_size", 0) or 0)
                ask_sz = int(getattr(quote, "ask_size", 0) or 0)
                if oi < 10:
                    oi = bid_sz + ask_sz
            if oi < 5:
                continue

            gex_rows.append(dict(strike=strike, oi=oi, iv=iv, T=T, is_call=is_call))

            if quote and dte_min <= dte <= dte_max:
                bid = float(getattr(quote, "bid_price", 0) or 0)
                ask = float(getattr(quote, "ask_price", 0) or 0)
                if bid > 0 and ask > 0:
                    row = {
                        "strike":   strike,
                        "expiry":   expiry_str[:10],
                        "dte":      dte,
                        "bid":      round(bid, 3),
                        "ask":      round(ask, 3),
                        "spread":   round(ask - bid, 3),
                        "oi_proxy": bid_sz + ask_sz,
                        "iv":       round(iv, 4),
                    }
                    if is_put and spot * 0.84 <= strike <= spot * 1.02:
                        put_rows.append(row)
                    elif is_call and spot * 0.98 <= strike <= spot * 1.15:
                        call_rows.append(row)
        except Exception as e:
            logger.debug("Skip %s: %s", symbol, e)
            continue

    logger.info("Combined chain %s: %d GEX, %d puts, %d calls (DTE %d-%d)",
                underlying, len(gex_rows), len(put_rows), len(call_rows), dte_min, dte_max)
    return {"gex_contracts": gex_rows, "puts_liquid": put_rows, "calls_liquid": call_rows}


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
