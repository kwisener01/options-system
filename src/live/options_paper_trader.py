"""
Simulated options paper trader.

Tracks open spreads in data/options_paper_state.json using real SPY/VIX prices.
No actual option orders are placed — P&L is computed via Black-Scholes against
live market data, which is accurate enough for short-dated spreads.

Workflow (daily rolling)
────────────────────────
Mon-Fri 9:45 AM : daily_cycle() — close yesterday's position (or TP), then open new 3DTE spread
Any time        : check_tp()   — close early if 50% profit captured
Fri 3:45 PM     : settle_expiry() — settle any position expiring today

Workflow (weekly, legacy)
────────────────────────
Friday AM  : open_weekly()   — select strategy, price spread, record position
Any day    : check_tp()      — close early if 50% profit captured
Friday 3:45: settle_expiry() — settle any position expiring today
"""
import json
import math
import logging
import os
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from config.settings import DATA_DIR
from src.analysis.macro_scanner import get_macro_snapshot
from src.analysis.gex_scanner import scan as gex_scan, GEXResult
from src.backtest.wheel_engine import bs_put, bs_call
from src.backtest.multi_strategy_engine import (
    MultiStrategyEngine,
    STRATEGY_CASH, STRATEGY_BULL_PUT, STRATEGY_BEAR_CALL,
    STRATEGY_IRON_CONDOR, STRATEGY_IRON_BUTTERFLY, STRATEGY_BULL_CALL_DEBIT,
    STRATEGY_PUT_RATIO, STRATEGY_BREAKOUT_STRANGLE,
)
from src.notifications.slack_notifier import send_message

logger = logging.getLogger(__name__)

STATE_PATH = os.path.join(DATA_DIR, "options_paper_state.json")
_SHARES    = 100
_RF        = 0.045


def _nearest_spy_expiry(min_dte: int = 1) -> date:
    """
    Return the nearest SPY expiry that is at least min_dte calendar days away.
    SPY has daily expirations Mon-Fri. Falls back to min_dte days from today on error.
    """
    today = date.today()
    target = today + timedelta(days=min_dte)
    try:
        t = yf.Ticker("SPY")
        expirations = t.options  # tuple of "YYYY-MM-DD" strings
        for exp_str in expirations:
            exp = date.fromisoformat(exp_str)
            if exp >= target:
                return exp
    except Exception as e:
        logger.warning("Could not fetch SPY expirations: %s — using %dd fallback", e, min_dte)
    return target


# ── state I/O ─────────────────────────────────────────────────────────────────

def _load() -> dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    return {
        "account_value": 10_000.0,
        "open_position": None,
        "closed_trades": [],
        "total_pnl": 0.0,
        "n_wins": 0,
        "n_losses": 0,
    }


def _save(state: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ── live market data ──────────────────────────────────────────────────────────

def _spy_price() -> float:
    try:
        t   = yf.Ticker("SPY")
        px  = t.fast_info.get("last_price") or t.fast_info.get("regularMarketPrice")
        if px:
            return float(px)
    except Exception:
        pass
    # fallback: last daily close
    df = yf.download("SPY", period="2d", interval="1d", auto_adjust=True, progress=False)
    col = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
    return float(col.iloc[-1].item() if hasattr(col.iloc[-1], "item") else col.iloc[-1])


def _vix() -> float:
    try:
        t  = yf.Ticker("^VIX")
        px = t.fast_info.get("last_price") or t.fast_info.get("regularMarketPrice")
        if px:
            return float(px)
    except Exception:
        pass
    df = yf.download("^VIX", period="2d", interval="1d", auto_adjust=True, progress=False)
    col = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
    return float(col.iloc[-1].item() if hasattr(col.iloc[-1], "item") else col.iloc[-1])


def _sigma(vix: float, iv_premium: float = 1.20) -> float:
    return vix / 100.0 * iv_premium


# ── option valuation ──────────────────────────────────────────────────────────

def _spread_value(pos: dict, spot: float, vix: float, today: date) -> float:
    """Current Black-Scholes value of the open spread (cost to close)."""
    expiry = date.fromisoformat(pos["expiry_date"])
    T      = max((expiry - today).days, 0) / 365
    sig    = _sigma(vix)
    s      = pos["strategy"]

    K1, K2 = pos["short_strike"], pos["long_strike"]

    if s == STRATEGY_BULL_PUT:
        return bs_put(spot, K1, T, sig) - bs_put(spot, K2, T, sig)
    if s == STRATEGY_BEAR_CALL:
        return bs_call(spot, K1, T, sig) - bs_call(spot, K2, T, sig)
    if s == STRATEGY_IRON_CONDOR:
        put_v  = bs_put(spot,  pos["put_short"],  T, sig) - bs_put(spot,  pos["put_long"],  T, sig)
        call_v = bs_call(spot, pos["call_short"], T, sig) - bs_call(spot, pos["call_long"], T, sig)
        return put_v + call_v
    if s == STRATEGY_IRON_BUTTERFLY:
        # put_short = call_short = ATM strike (the body)
        put_v  = bs_put(spot,  pos["put_short"],  T, sig) - bs_put(spot,  pos["put_long"],  T, sig)
        call_v = bs_call(spot, pos["call_short"], T, sig) - bs_call(spot, pos["call_long"], T, sig)
        return put_v + call_v
    if s == STRATEGY_BULL_CALL_DEBIT:
        return bs_call(spot, K1, T, sig) - bs_call(spot, K2, T, sig)
    if s == STRATEGY_PUT_RATIO:
        return 2*bs_put(spot, pos["put_short"], T, sig) - bs_put(spot, pos["put_long"], T, sig)
    if s == STRATEGY_BREAKOUT_STRANGLE:
        return bs_call(spot, pos["call_long"], T, sig) + bs_put(spot, pos["put_long"], T, sig)
    return 0.0


def _settle_pnl_ps(pos: dict, spot: float) -> float:
    """P&L per share at expiry or forced close."""
    s  = pos["strategy"]
    K1, K2 = pos["short_strike"], pos["long_strike"]
    cr = pos["net_credit"]    # >0 credit, <0 debit

    if s == STRATEGY_BULL_PUT:
        return cr - max(0.0, K1 - spot) + max(0.0, K2 - spot)
    if s == STRATEGY_BEAR_CALL:
        return cr - max(0.0, spot - K1) + max(0.0, spot - K2)
    if s == STRATEGY_IRON_CONDOR:
        put_pnl  = pos["put_credit"]  - max(0.0, pos["put_short"]  - spot) + max(0.0, pos["put_long"]  - spot)
        call_pnl = pos["call_credit"] - max(0.0, spot - pos["call_short"]) + max(0.0, spot - pos["call_long"])
        return put_pnl + call_pnl
    if s == STRATEGY_IRON_BUTTERFLY:
        # put_short = call_short = ATM body strike
        put_pnl  = pos["put_credit"]  - max(0.0, pos["put_short"]  - spot) + max(0.0, pos["put_long"]  - spot)
        call_pnl = pos["call_credit"] - max(0.0, spot - pos["call_short"]) + max(0.0, spot - pos["call_long"])
        return put_pnl + call_pnl
    if s == STRATEGY_BULL_CALL_DEBIT:
        return max(0.0, spot - K1) - max(0.0, spot - K2) + cr
    if s == STRATEGY_PUT_RATIO:
        return cr - 2*max(0.0, pos["put_short"] - spot) + max(0.0, pos["put_long"] - spot)
    if s == STRATEGY_BREAKOUT_STRANGLE:
        debit = abs(cr)
        return max(0.0, spot - pos["call_long"]) + max(0.0, pos["put_long"] - spot) - debit
    return 0.0


# ── GEX-guided strike selection ───────────────────────────────────────────────

def _gex_strikes(gex: GEXResult, spot: float, engine: MultiStrategyEngine):
    """
    Return (put_strike, call_strike) anchored to GEX walls.
    Falls back to mechanical OTM% if a wall is outside the usable range
    (too close or too far from spot).
    """
    def rs(p):
        return round(p / 0.5) * 0.5

    min_put  = spot * (1 - 0.03)
    max_put  = spot * (1 - 0.005)
    min_call = spot * (1 + 0.005)
    max_call = spot * (1 + 0.03)

    put_strike  = rs(gex.put_wall)  if min_put  <= gex.put_wall  <= max_put  else rs(spot * (1 - engine.short_otm_pct))
    call_strike = rs(gex.call_wall) if min_call <= gex.call_wall <= max_call else rs(spot * (1 + engine.call_otm_pct))

    return put_strike, call_strike


# ── strategy builder ──────────────────────────────────────────────────────────

def _build_position(strategy: str, spot: float, vix: float,
                    engine: MultiStrategyEngine, today: date,
                    account: float,
                    gex: Optional[GEXResult] = None,
                    expiry_date: Optional[date] = None) -> Optional[dict]:
    sig = _sigma(vix, engine.iv_premium)

    if expiry_date is None:
        expiry_date = today + timedelta(days=7)   # legacy weekly default
    expiry = expiry_date
    T = max((expiry - today).days, 1) / 365

    def rs(p):
        t = 0.5
        return round(p / t) * t

    pct_width = rs(spot * engine.spread_width_pct)
    if engine.max_dollar_risk is not None:
        cap_width = rs(engine.max_dollar_risk / 100)
        pct_width = min(pct_width, cap_width)
    width = max(pct_width, 1.0)

    # Resolve GEX-guided strikes (falls back to mechanical OTM% if no GEX)
    if gex is not None:
        gex_put, gex_call = _gex_strikes(gex, spot, engine)
    else:
        gex_put  = rs(spot * (1 - engine.short_otm_pct))
        gex_call = rs(spot * (1 + engine.call_otm_pct))

    if strategy == STRATEGY_BULL_PUT:
        K1  = gex_put
        K2  = K1 - width
        cr  = max(bs_put(spot, K1, T, sig) - bs_put(spot, K2, T, sig), 0.01)
        mr  = width - cr
        pos = dict(strategy=strategy, short_strike=K1, long_strike=K2,
                   net_credit=cr, spread_width=width, max_risk=mr,
                   put_short=K1, put_long=K2, call_short=None, call_long=None,
                   put_credit=cr, call_credit=0.0)

    elif strategy == STRATEGY_BEAR_CALL:
        K1  = gex_call
        K2  = K1 + width
        cr  = max(bs_call(spot, K1, T, sig) - bs_call(spot, K2, T, sig), 0.01)
        mr  = width - cr
        pos = dict(strategy=strategy, short_strike=K1, long_strike=K2,
                   net_credit=cr, spread_width=width, max_risk=mr,
                   put_short=None, put_long=None, call_short=K1, call_long=K2,
                   put_credit=0.0, call_credit=cr)

    elif strategy == STRATEGY_IRON_CONDOR:
        Kp1 = gex_put;   Kp2 = Kp1 - width
        Kc1 = gex_call;  Kc2 = Kc1 + width
        pc  = max(bs_put(spot,  Kp1, T, sig) - bs_put(spot,  Kp2, T, sig), 0.005)
        cc  = max(bs_call(spot, Kc1, T, sig) - bs_call(spot, Kc2, T, sig), 0.005)
        cr  = pc + cc
        mr  = width - min(pc, cc)
        pos = dict(strategy=strategy, short_strike=Kp1, long_strike=Kp2,
                   net_credit=cr, spread_width=width, max_risk=mr,
                   put_short=Kp1, put_long=Kp2, call_short=Kc1, call_long=Kc2,
                   put_credit=pc, call_credit=cc)

    elif strategy == STRATEGY_IRON_BUTTERFLY:
        # Body: sell ATM put + ATM call at same strike; wings protect each side
        K_atm      = rs(spot)
        wing_width = max(rs(spot * engine.butterfly_wing_pct), 1.0)
        K_put_long  = K_atm - wing_width
        K_call_long = K_atm + wing_width
        pc  = max(bs_put(spot,  K_atm, T, sig) - bs_put(spot,  K_put_long,  T, sig), 0.005)
        cc  = max(bs_call(spot, K_atm, T, sig) - bs_call(spot, K_call_long, T, sig), 0.005)
        cr  = pc + cc
        mr  = wing_width - cr   # max loss = one full wing minus total credit
        pos = dict(strategy=strategy, short_strike=K_atm, long_strike=K_atm,
                   net_credit=cr, spread_width=wing_width, max_risk=mr,
                   put_short=K_atm, put_long=K_put_long,
                   call_short=K_atm, call_long=K_call_long,
                   put_credit=pc, call_credit=cc,
                   atm_strike=K_atm, wing_width=wing_width)

    elif strategy == STRATEGY_PUT_RATIO:
        K_short = rs(spot * (1 - engine.short_otm_pct))
        K_long  = rs(spot * (1 - engine.ratio_long_otm_pct))
        if K_long >= K_short:
            K_long = K_short - width
        cr  = max(2*bs_put(spot, K_short, T, sig) - bs_put(spot, K_long, T, sig), 0.005)
        mr  = max(2*(K_short - K_long) - cr, 0.10)
        pos = dict(strategy=strategy, short_strike=K_short, long_strike=K_long,
                   net_credit=cr, spread_width=K_short-K_long, max_risk=mr,
                   put_short=K_short, put_long=K_long,
                   call_short=None, call_long=None,
                   put_credit=cr, call_credit=0.0, ratio=2)

    elif strategy == STRATEGY_BREAKOUT_STRANGLE:
        K_call = rs(spot * (1 + engine.breakout_otm_pct))
        K_put  = rs(spot * (1 - engine.breakout_otm_pct))
        debit  = max(bs_call(spot, K_call, T, sig) + bs_put(spot, K_put, T, sig), 0.01)
        mr     = debit
        pos = dict(strategy=strategy, short_strike=K_call, long_strike=K_put,
                   net_credit=-debit, spread_width=debit*4, max_risk=mr,
                   call_long=K_call, put_long=K_put,
                   call_short=None, put_short=None,
                   call_credit=0.0, put_credit=0.0)

    elif strategy == STRATEGY_BULL_CALL_DEBIT:
        K1  = rs(spot * (1 + engine.debit_long_otm_pct))
        K2  = rs(spot * (1 + engine.debit_short_otm_pct))
        if K2 <= K1:
            K2 = K1 + width
        db  = max(bs_call(spot, K1, T, sig) - bs_call(spot, K2, T, sig), 0.01)
        mr  = db
        pos = dict(strategy=strategy, short_strike=K1, long_strike=K2,
                   net_credit=-db, spread_width=K2 - K1, max_risk=mr,
                   put_short=None, put_long=None, call_short=K2, call_long=K1,
                   put_credit=0.0, call_credit=-db)
    else:
        return None

    if mr <= 0:
        return None

    contracts = max(int(account * engine.max_risk_pct // (mr * _SHARES)), 1)
    pos.update(dict(
        entry_date=str(today),
        expiry_date=str(expiry),
        spot_entry=round(spot, 2),
        contracts=contracts,
        sigma=round(sig, 4),
        vix_at_entry=round(vix, 2),
    ))
    return pos


# ── Alpaca live order integration ─────────────────────────────────────────────

def _try_alpaca_open(pos: dict) -> None:
    """
    Place real Alpaca option orders for a newly opened position.
    Mutates pos in place to add alpaca_*_symbol and alpaca_*_order_id keys.
    Logs errors but never raises — simulated state is always saved regardless.
    """
    try:
        from src.live.alpaca_options import find_contract, place_spread
    except ImportError as e:
        logger.warning("alpaca_options unavailable: %s", e)
        return

    strategy = pos["strategy"]
    expiry   = date.fromisoformat(pos["expiry_date"])
    qty      = pos["contracts"]

    try:
        if strategy == STRATEGY_BULL_PUT:
            ss = find_contract("SPY", expiry, "PUT", pos["short_strike"])
            ls = find_contract("SPY", expiry, "PUT", pos["long_strike"])
            if ss and ls:
                r = place_spread(ss, ls, qty, pos["net_credit"], pos["net_credit"] * 0.3)
                pos.update(alpaca_short_symbol=ss, alpaca_long_symbol=ls,
                           alpaca_short_order_id=r["short_order_id"],
                           alpaca_long_order_id=r["long_order_id"])

        elif strategy == STRATEGY_BEAR_CALL:
            ss = find_contract("SPY", expiry, "CALL", pos["short_strike"])
            ls = find_contract("SPY", expiry, "CALL", pos["long_strike"])
            if ss and ls:
                r = place_spread(ss, ls, qty, pos["net_credit"], pos["net_credit"] * 0.3)
                pos.update(alpaca_short_symbol=ss, alpaca_long_symbol=ls,
                           alpaca_short_order_id=r["short_order_id"],
                           alpaca_long_order_id=r["long_order_id"])

        elif strategy == STRATEGY_IRON_CONDOR:
            ps = find_contract("SPY", expiry, "PUT",  pos["put_short"])
            pl = find_contract("SPY", expiry, "PUT",  pos["put_long"])
            cs = find_contract("SPY", expiry, "CALL", pos["call_short"])
            cl = find_contract("SPY", expiry, "CALL", pos["call_long"])
            if ps and pl:
                r1 = place_spread(ps, pl, qty, pos["put_credit"], pos["put_credit"] * 0.3)
                pos.update(alpaca_put_short_symbol=ps, alpaca_put_long_symbol=pl,
                           alpaca_put_short_order_id=r1["short_order_id"],
                           alpaca_put_long_order_id=r1["long_order_id"])
            if cs and cl:
                r2 = place_spread(cs, cl, qty, pos["call_credit"], pos["call_credit"] * 0.3)
                pos.update(alpaca_call_short_symbol=cs, alpaca_call_long_symbol=cl,
                           alpaca_call_short_order_id=r2["short_order_id"],
                           alpaca_call_long_order_id=r2["long_order_id"])

        elif strategy == STRATEGY_BULL_CALL_DEBIT:
            # long_strike = K2 (higher, sold); short_strike = K1 (lower, bought)
            ss = find_contract("SPY", expiry, "CALL", pos["long_strike"])   # sell K2
            ls = find_contract("SPY", expiry, "CALL", pos["short_strike"])  # buy  K1
            if ss and ls:
                debit = abs(pos["net_credit"])
                r = place_spread(ss, ls, qty, debit * 0.3, debit)
                pos.update(alpaca_short_symbol=ss, alpaca_long_symbol=ls,
                           alpaca_short_order_id=r["short_order_id"],
                           alpaca_long_order_id=r["long_order_id"])

        logger.info("Alpaca orders placed for %s", strategy)

    except Exception as e:
        logger.error("Alpaca open failed (%s): %s", strategy, e)


def _try_alpaca_close(pos: dict) -> None:
    """
    Close real Alpaca option legs for a position being closed.
    Reads alpaca_*_symbol keys written by _try_alpaca_open.
    Logs errors but never raises.
    """
    try:
        from src.live.alpaca_options import close_spread
    except ImportError as e:
        logger.warning("alpaca_options unavailable: %s", e)
        return

    strategy = pos["strategy"]
    qty      = pos["contracts"]

    try:
        if strategy in (STRATEGY_BULL_PUT, STRATEGY_BEAR_CALL, STRATEGY_BULL_CALL_DEBIT):
            ss = pos.get("alpaca_short_symbol")
            ls = pos.get("alpaca_long_symbol")
            if ss and ls:
                close_spread(ss, ls, qty)
            else:
                logger.warning("No Alpaca symbols stored for %s — skipping close", strategy)

        elif strategy == STRATEGY_IRON_CONDOR:
            ps = pos.get("alpaca_put_short_symbol")
            pl = pos.get("alpaca_put_long_symbol")
            cs = pos.get("alpaca_call_short_symbol")
            cl = pos.get("alpaca_call_long_symbol")
            if ps and pl:
                close_spread(ps, pl, qty)
            if cs and cl:
                close_spread(cs, cl, qty)

        logger.info("Alpaca close orders placed for %s", strategy)

    except Exception as e:
        logger.error("Alpaca close failed (%s): %s", strategy, e)


# ── event calendar guard ──────────────────────────────────────────────────────

# Known high-impact dates — add FOMC/CPI/NFP dates here as they are announced.
# Format: "YYYY-MM-DD"
_HIGH_IMPACT_DATES: set[str] = {
    "2026-05-07",   # FOMC rate decision
    "2026-06-10",   # FOMC
    "2026-07-29",   # FOMC
    "2026-09-16",   # FOMC
    "2026-10-28",   # FOMC
    "2026-12-09",   # FOMC
}


def _is_near_event(today: date) -> bool:
    """Return True if today or tomorrow is a high-impact macro event day."""
    for d in (_HIGH_IMPACT_DATES):
        event = date.fromisoformat(d)
        if 0 <= (event - today).days <= 1:
            return True
    return False


# ── public API ────────────────────────────────────────────────────────────────

def preview_weekly(engine: Optional[MultiStrategyEngine] = None) -> dict:
    """
    Dry-run: fetch live data, run strategy selector, price the spread,
    and return a result dict — without saving state or placing any orders.
    """
    if engine is None:
        engine = MultiStrategyEngine()

    spot   = _spy_price()
    vix    = _vix()
    macro  = get_macro_snapshot()
    is_bull = macro.regime == "BULL"
    is_bear = macro.regime == "BEAR"
    spy_tr  = {"UP": 1, "FLAT": 0, "DOWN": -1}.get(macro.spy_trend, 0)

    state    = _load()
    prev_vix = None
    if state["closed_trades"]:
        prev_vix = state["closed_trades"][-1].get("vix_at_entry")

    today  = date.today()

    gex = None
    try:
        gex = gex_scan()
    except Exception as e:
        logger.warning("GEX scan failed in preview — using mechanical strikes: %s", e)

    gex_regime   = gex.gex_regime    if gex else "UNKNOWN"
    vanna_signal = gex.vanna_signal  if gex else "NEUTRAL"
    near_flip    = (abs(spot - gex.flip_level) / spot < 0.005) if gex else False
    near_event   = _is_near_event(today)

    strategy = engine._select(vix, is_bull, is_bear, spy_tr, prev_vix,
                               gex_regime=gex_regime, vanna_signal=vanna_signal,
                               near_event=near_event, near_flip=near_flip)

    result = {
        "strategy":  strategy,
        "spot":      round(spot, 2),
        "vix":       round(vix, 2),
        "regime":    macro.regime,
        "spy_trend": macro.spy_trend,
        "macro_summary": macro.summary,
        "gex_put_wall":  round(gex.put_wall,  2) if gex else None,
        "gex_call_wall": round(gex.call_wall, 2) if gex else None,
        "gex_regime":    gex_regime,
        "vanna_signal":  vanna_signal,
        "near_event":    near_event,
        "near_flip":     near_flip,
        "position":  None,
    }

    if strategy != STRATEGY_CASH:
        pos = _build_position(strategy, spot, vix, engine, today, state["account_value"], gex=gex)
        result["position"] = pos

    return result


def open_weekly(engine: Optional[MultiStrategyEngine] = None) -> Optional[dict]:
    """
    Select a strategy for this week, price the spread, and save it.
    Returns the position dict or None if CASH / already open.
    """
    if engine is None:
        engine = MultiStrategyEngine()

    state = _load()
    if state["open_position"] is not None:
        logger.info("Position already open — skipping entry")
        return None

    spot   = _spy_price()
    vix    = _vix()
    macro  = get_macro_snapshot()
    is_bull = macro.regime == "BULL"
    is_bear = macro.regime == "BEAR"
    spy_tr  = {"UP": 1, "FLAT": 0, "DOWN": -1}.get(macro.spy_trend, 0)

    # prev_vix: load last closed trade's vix if available
    prev_vix = None
    if state["closed_trades"]:
        prev_vix = state["closed_trades"][-1].get("vix_at_entry")

    gex = None
    try:
        gex = gex_scan()
        logger.info("GEX scan: put_wall=%.2f call_wall=%.2f regime=%s",
                    gex.put_wall, gex.call_wall, gex.gex_regime)
    except Exception as e:
        logger.warning("GEX scan failed — using mechanical strikes: %s", e)

    today    = date.today()

    gex_regime   = gex.gex_regime   if gex else "UNKNOWN"
    vanna_signal = gex.vanna_signal if gex else "NEUTRAL"
    near_flip    = (abs(spot - gex.flip_level) / spot < 0.005) if gex else False
    near_event   = _is_near_event(today)

    strategy = engine._select(vix, is_bull, is_bear, spy_tr, prev_vix,
                               gex_regime=gex_regime, vanna_signal=vanna_signal,
                               near_event=near_event, near_flip=near_flip)

    logger.info("Live signal: SPY=%.2f VIX=%.1f regime=%s -> %s",
                spot, vix, macro.regime, strategy)

    if strategy == STRATEGY_CASH:
        msg = f":no_entry: *Options Paper* — {today} | CASH (VIX={vix:.1f}, {macro.regime})"
        send_message(msg)
        return None

    pos = _build_position(strategy, spot, vix, engine, today, state["account_value"], gex=gex)
    if pos is None:
        logger.warning("Could not price spread — skipping")
        return None

    state["open_position"] = pos
    _try_alpaca_open(pos)   # places real Alpaca orders; adds OCC symbols into pos
    _save(state)            # persist including alpaca symbols

    # Slack notification
    nc = pos["net_credit"];  c = pos["contracts"];  mr = pos["max_risk"]
    if strategy in (STRATEGY_IRON_CONDOR, STRATEGY_IRON_BUTTERFLY):
        legs = (f"Put {pos['put_short']:.1f}/{pos['put_long']:.1f} + "
                f"Call {pos['call_short']:.1f}/{pos['call_long']:.1f}")
    elif strategy == STRATEGY_BULL_CALL_DEBIT:
        legs = f"Call debit {pos['short_strike']:.1f}/{pos['long_strike']:.1f}"
    elif strategy == STRATEGY_PUT_RATIO:
        legs = f"2x short ${pos['short_strike']:.1f} put / 1x long ${pos['long_strike']:.1f} put"
    elif strategy == STRATEGY_BREAKOUT_STRANGLE:
        legs = f"Buy ${pos['put_long']:.1f}p + ${pos['call_long']:.1f}c"
    else:
        legs = f"{pos['short_strike']:.1f}/{pos['long_strike']:.1f}"

    msg = (
        f":green_circle: *Options Paper — OPEN* | {today}\n"
        f"> Strategy : `{strategy}`\n"
        f"> SPY      : ${spot:.2f}  |  VIX: {vix:.1f}\n"
        f"> Strikes  : {legs}\n"
        f"> Credit   : ${nc:.3f}/share × {c} contracts = ${nc*_SHARES*c:,.2f}\n"
        f"> Max risk : ${mr*_SHARES*c:,.2f}  |  Expiry: {pos['expiry_date']}\n"
        f"> Account  : ${state['account_value']:,.2f}"
    )
    send_message(msg)
    logger.info("Opened: %s", strategy)
    return pos


def open_daily(engine: Optional[MultiStrategyEngine] = None,
               dte: int = 1) -> Optional[dict]:
    """
    Open a short-dated spread using the nearest SPY expiry >= dte days away.
    For PDT-safe compounding: open today, close tomorrow = overnight hold, not a day trade.
    Default dte=1 = tomorrow's expiry: maximum theta capture, daily capital recycling.
    Higher dte (2-3) adds gamma buffer at the cost of slower compounding.
    """
    if engine is None:
        engine = MultiStrategyEngine()

    state = _load()
    if state["open_position"] is not None:
        logger.info("Position already open — skipping daily entry")
        return None

    spot   = _spy_price()
    vix    = _vix()
    macro  = get_macro_snapshot()
    is_bull = macro.regime == "BULL"
    is_bear = macro.regime == "BEAR"
    spy_tr  = {"UP": 1, "FLAT": 0, "DOWN": -1}.get(macro.spy_trend, 0)

    prev_vix = None
    if state["closed_trades"]:
        prev_vix = state["closed_trades"][-1].get("vix_at_entry")

    today  = date.today()
    expiry = _nearest_spy_expiry(min_dte=dte)

    gex = None
    try:
        gex = gex_scan()
        logger.info("GEX scan: put_wall=%.2f call_wall=%.2f regime=%s",
                    gex.put_wall, gex.call_wall, gex.gex_regime)
    except Exception as e:
        logger.warning("GEX scan failed — using mechanical strikes: %s", e)

    gex_regime   = gex.gex_regime   if gex else "UNKNOWN"
    vanna_signal = gex.vanna_signal if gex else "NEUTRAL"
    near_flip    = (abs(spot - gex.flip_level) / spot < 0.005) if gex else False
    near_event   = _is_near_event(today)

    strategy = engine._select(vix, is_bull, is_bear, spy_tr, prev_vix,
                               gex_regime=gex_regime, vanna_signal=vanna_signal,
                               near_event=near_event, near_flip=near_flip)

    logger.info("Daily open: SPY=%.2f VIX=%.1f regime=%s expiry=%s -> %s",
                spot, vix, macro.regime, expiry, strategy)

    if strategy == STRATEGY_CASH:
        msg = (f":no_entry: *Daily Options — CASH* | {today}\n"
               f"> VIX={vix:.1f}  {macro.regime}  |  No trade today")
        send_message(msg)
        return None

    pos = _build_position(strategy, spot, vix, engine, today,
                          state["account_value"], gex=gex, expiry_date=expiry)
    if pos is None:
        logger.warning("Could not price daily spread — skipping")
        return None

    state["open_position"] = pos
    _try_alpaca_open(pos)
    _save(state)

    nc = pos["net_credit"];  c = pos["contracts"];  mr = pos["max_risk"]
    dte_actual = (expiry - today).days
    if strategy in (STRATEGY_IRON_CONDOR, STRATEGY_IRON_BUTTERFLY):
        legs = (f"Put {pos['put_short']:.1f}/{pos['put_long']:.1f} + "
                f"Call {pos['call_short']:.1f}/{pos['call_long']:.1f}")
    elif strategy == STRATEGY_BULL_CALL_DEBIT:
        legs = f"Call debit {pos['short_strike']:.1f}/{pos['long_strike']:.1f}"
    elif strategy == STRATEGY_PUT_RATIO:
        legs = f"2x short ${pos['short_strike']:.1f} put / 1x long ${pos['long_strike']:.1f} put"
    elif strategy == STRATEGY_BREAKOUT_STRANGLE:
        legs = f"Buy ${pos['put_long']:.1f}p + ${pos['call_long']:.1f}c"
    else:
        legs = f"{pos['short_strike']:.1f}/{pos['long_strike']:.1f}"

    msg = (
        f":green_circle: *Daily Options — OPEN* | {today}  ({dte_actual}DTE)\n"
        f"> Strategy : `{strategy}`\n"
        f"> SPY      : ${spot:.2f}  |  VIX: {vix:.1f}\n"
        f"> Strikes  : {legs}\n"
        f"> Credit   : ${nc:.3f}/share × {c} contracts = ${nc*_SHARES*c:,.2f}\n"
        f"> Max risk : ${mr*_SHARES*c:,.2f}  |  Expiry: {expiry}\n"
        f"> Account  : ${state['account_value']:,.2f}"
    )
    send_message(msg)
    logger.info("Daily opened: %s  expiry=%s  credit=%.3f", strategy, expiry, nc)
    return pos


def daily_cycle(engine: Optional[MultiStrategyEngine] = None,
                dte: int = 1, tp_pct: float = 0.50) -> dict:
    """
    Full daily cycle: check TP → force-close any position from a prior day → open fresh spread.
    Designed to run at 9:45 AM ET Mon-Fri. PDT-safe: overnight hold = not a day trade.
    Returns a summary dict with keys 'closed' and 'opened'.
    """
    if engine is None:
        engine = MultiStrategyEngine()

    result = {"closed": None, "opened": None}

    # 1. Try take-profit first (position held at least since yesterday)
    tp = check_tp(tp_pct=tp_pct)
    if tp:
        result["closed"] = tp
        logger.info("daily_cycle: TP triggered — P&L=%.2f", tp["pnl"])
    else:
        # 2. Force-close any open position from a prior day (max hold = 1 overnight)
        state = _load()
        pos   = state.get("open_position")
        if pos and pos.get("entry_date") != str(date.today()):
            spot  = _spy_price()
            today = date.today()
            cur   = _spread_value(pos, spot, _vix(), today)
            nc    = pos["net_credit"]
            c     = pos["contracts"]
            if nc > 0:
                pnl = (nc - cur) * _SHARES * c
            else:
                pnl = (cur - abs(nc)) * _SHARES * c
            closed = _record_close(state, pos, spot, pnl, "DAILY_CLOSE", today)
            result["closed"] = closed
            logger.info("daily_cycle: force-closed %s  P&L=%.2f", pos["strategy"], pnl)

    # 3. Open a fresh spread for today
    opened = open_daily(engine=engine, dte=dte)
    result["opened"] = opened
    return result


def check_tp(engine: Optional[MultiStrategyEngine] = None,
             tp_pct: float = 0.50) -> Optional[dict]:
    """
    Check if the open spread has hit the take-profit threshold.
    Returns the closed trade dict if TP was triggered, else None.
    """
    state = _load()
    pos   = state["open_position"]
    if pos is None:
        return None

    today = date.today()
    if str(today) == pos["entry_date"]:
        return None    # don't TP same day as entry

    spot = _spy_price()
    vix  = _vix()
    cur  = _spread_value(pos, spot, vix, today)
    nc   = pos["net_credit"]

    tp_hit = False
    if nc > 0:                             # credit trade
        tp_hit = cur <= nc * (1 - tp_pct)
    else:                                  # debit trade
        max_p  = pos["spread_width"] - abs(nc)
        target = abs(nc) + tp_pct * max_p
        tp_hit = cur >= target

    if not tp_hit:
        days_left = (date.fromisoformat(pos["expiry_date"]) - today).days
        logger.info("TP check: cur=%.3f credit=%.3f (%.0f%% decayed) — hold (%dd left)",
                    cur, abs(nc), (1 - cur / abs(nc)) * 100 if nc != 0 else 0, days_left)
        return None

    # Close at residual value
    c   = pos["contracts"]
    if nc > 0:
        pnl = (nc - cur) * _SHARES * c
    else:
        pnl = (cur - abs(nc)) * _SHARES * c

    return _record_close(state, pos, spot, pnl, "TAKE_PROFIT", today)


def settle_expiry() -> Optional[dict]:
    """
    Settle an expired spread at today's SPY close.
    Call every Friday around 4 PM ET.
    """
    state = _load()
    pos   = state["open_position"]
    if pos is None:
        return None

    today  = date.today()
    expiry = date.fromisoformat(pos["expiry_date"])
    if today < expiry:
        logger.info("Position not expired yet (expiry=%s)", expiry)
        return None

    spot   = _spy_price()
    pnl_ps = _settle_pnl_ps(pos, spot)
    c      = pos["contracts"]
    pnl    = pnl_ps * _SHARES * c

    nc = pos["net_credit"]
    if nc > 0:
        outcome = ("FULL_PROFIT" if pnl >= nc * _SHARES * c * 0.95
                   else "MAX_LOSS" if pnl <= -(pos["max_risk"]) * _SHARES * c * 0.95
                   else "PARTIAL_LOSS")
    else:
        outcome = "FULL_PROFIT" if pnl > 0 else "MAX_LOSS"

    return _record_close(state, pos, spot, pnl, outcome, today)


def _record_close(state, pos, spot, pnl, outcome, today):
    _try_alpaca_close(pos)  # close real Alpaca orders before updating local state
    state["account_value"] += pnl
    state["total_pnl"]     += pnl
    if pnl > 0:
        state["n_wins"]   = state.get("n_wins", 0)   + 1
    else:
        state["n_losses"] = state.get("n_losses", 0) + 1

    closed = {**pos, "close_date": str(today), "spot_close": round(spot, 2),
              "pnl": round(pnl, 2), "outcome": outcome}
    state["closed_trades"].append(closed)
    state["open_position"] = None
    _save(state)

    emoji = ":white_check_mark:" if pnl > 0 else ":x:"
    msg = (
        f"{emoji} *Options Paper — CLOSE* ({outcome}) | {today}\n"
        f"> Strategy  : `{pos['strategy']}`\n"
        f"> SPY close : ${spot:.2f}\n"
        f"> P&L       : ${pnl:+,.2f}\n"
        f"> Account   : ${state['account_value']:,.2f}  "
        f"(total {state['total_pnl']:+,.2f}  |  "
        f"{state.get('n_wins',0)}W / {state.get('n_losses',0)}L)"
    )
    send_message(msg)
    logger.info("Closed %s: P&L=$%.2f  outcome=%s", pos["strategy"], pnl, outcome)
    return closed


def status() -> dict:
    """Return current paper trading state summary."""
    state = _load()
    pos   = state["open_position"]
    today = date.today()

    summary = {
        "account_value": state["account_value"],
        "total_pnl":     state["total_pnl"],
        "n_wins":        state.get("n_wins", 0),
        "n_losses":      state.get("n_losses", 0),
        "open_position": None,
        "current_value": None,
        "unrealized_pnl": None,
    }

    if pos:
        spot = _spy_price()
        vix  = _vix()
        cur  = _spread_value(pos, spot, vix, today)
        nc   = pos["net_credit"]
        c    = pos["contracts"]
        if nc > 0:
            unreal = (nc - cur) * _SHARES * c
        else:
            unreal = (cur - abs(nc)) * _SHARES * c
        summary["open_position"]  = pos
        summary["current_value"]  = round(cur, 4)
        summary["unrealized_pnl"] = round(unreal, 2)

    return summary
