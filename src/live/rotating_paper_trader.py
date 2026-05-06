"""
Intraday rotating options paper trader.

Evaluates market regime every ~30 min and rotates between strategy types:
  iron condor, butterfly, bull/bear credit spreads, BWB, jade lizard, debit spreads.

Designed for XSP (auto-falls back to SPY) on a ~$2,000 account.
Day-trade budget is tracked to stay under the PDT 3-per-5-days limit.

State: data/rotating_paper_state.json
Slack: fires on open / morph / close events.

Public API
----------
    detect_symbol()          -> "XSP" or "SPY"
    build_regime(symbol)     -> Regime
    tick(symbol, engine)     -> RotationCycle   # one evaluation pass
    status()                 -> dict
    reset(capital, symbol)   -> None
"""
import json
import logging
import math
import os
from datetime import date, datetime, timedelta
from typing import List, Optional

import pytz
import yfinance as yf

from config.settings import DATA_DIR
from src.backtest.wheel_engine import bs_put, bs_call
from src.live.rotating_engine import (
    Leg, Position, Regime, StrategyType,
    RotatingEngine, RotationCycle, MorphPlan,
)
from src.notifications.slack_notifier import send_message

logger  = logging.getLogger(__name__)
_RF     = 0.045
_SHARES = 100
_STATE  = os.path.join(DATA_DIR, "rotating_paper_state.json")
_ET     = pytz.timezone("America/New_York")

_DEFAULT_ACCOUNT  = 2_000.0
_DEFAULT_RISK_PCT = 0.25
_DEFAULT_DTE      = 7
_WING_WIDTH       = 2.0   # $2-wide spreads on XSP/SPY
_OTM_PCT          = 0.04  # 4% OTM for short strikes (~8-12 delta)
_CLOSE_HOUR_ET    = 15    # force-close open positions after 3 PM ET


# ── state I/O ─────────────────────────────────────────────────────────────────

def _load() -> dict:
    if os.path.exists(_STATE):
        with open(_STATE) as f:
            return json.load(f)
    return _blank_state()


def _save(state: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(_STATE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def _blank_state(capital: float = _DEFAULT_ACCOUNT, symbol: str = "XSP") -> dict:
    return {
        "symbol":         symbol,
        "account_value":  capital,
        "day_trades_used": 0,
        "week_start":     str(_monday()),
        "open_position":  None,
        "closed_trades":  [],
        "total_pnl":      0.0,
        "n_wins":         0,
        "n_losses":       0,
    }


# ── week / PDT tracking ───────────────────────────────────────────────────────

def _monday() -> date:
    today = date.today()
    return today - timedelta(days=today.weekday())


def _reset_dt_budget_if_new_week(state: dict) -> dict:
    if state.get("week_start") != str(_monday()):
        state["day_trades_used"] = 0
        state["week_start"]      = str(_monday())
    return state


# ── market data ───────────────────────────────────────────────────────────────

def detect_symbol() -> str:
    """Try XSP first; fall back to SPY if price data is unavailable."""
    try:
        t  = yf.Ticker("XSP")
        px = t.fast_info.get("last_price") or t.fast_info.get("regularMarketPrice")
        if px and float(px) > 0:
            logger.info("Symbol: XSP (px=%.2f)", float(px))
            return "XSP"
    except Exception:
        pass
    logger.info("XSP unavailable, using SPY")
    return "SPY"


def _spot(symbol: str) -> float:
    t  = yf.Ticker(symbol)
    px = t.fast_info.get("last_price") or t.fast_info.get("regularMarketPrice")
    if px:
        return float(px)
    df = yf.download(symbol, period="2d", interval="1d", auto_adjust=True, progress=False)
    return float(df["Close"].iloc[-1])


def _vix() -> float:
    t  = yf.Ticker("^VIX")
    px = t.fast_info.get("last_price") or t.fast_info.get("regularMarketPrice")
    if px:
        return float(px)
    df = yf.download("^VIX", period="2d", interval="1d", auto_adjust=True, progress=False)
    return float(df["Close"].iloc[-1])


def _iv_rank(current_vix: float) -> float:
    try:
        df = yf.download("^VIX", period="1y", interval="1d", auto_adjust=True, progress=False)
        lo = float(df["Close"].min())
        hi = float(df["Close"].max())
        if hi <= lo:
            return 50.0
        return max(0.0, min(100.0, (current_vix - lo) / (hi - lo) * 100))
    except Exception:
        return 50.0


def _intraday_trend(symbol: str) -> float:
    """
    Intraday VWAP deviation scaled to [-1, +1].
    +1 = 5% above VWAP (strong bull), -1 = 5% below (strong bear).
    """
    try:
        df = yf.download(symbol, period="1d", interval="5m", auto_adjust=True, progress=False)
        if len(df) < 3:
            return 0.0
        typical = (df["High"] + df["Low"] + df["Close"]) / 3
        vwap    = (typical * df["Volume"]).cumsum() / df["Volume"].cumsum()
        cur     = float(df["Close"].iloc[-1])
        v       = float(vwap.iloc[-1])
        return max(-1.0, min(1.0, (cur - v) / v * 20))
    except Exception:
        return 0.0


def _realized_vol(symbol: str) -> float:
    try:
        df   = yf.download(symbol, period="60d", interval="1d", auto_adjust=True, progress=False)
        rets = df["Close"].pct_change().dropna()
        return float(rets.tail(20).std() * math.sqrt(252))
    except Exception:
        return 0.20


def _tod() -> float:
    """Fraction of trading day elapsed: 0.0 = 9:30 ET, 1.0 = 16:00 ET."""
    now       = datetime.now(_ET)
    open_min  = 9 * 60 + 30
    close_min = 16 * 60
    cur_min   = now.hour * 60 + now.minute
    return max(0.0, min(1.0, (cur_min - open_min) / (close_min - open_min)))


def _is_market_open() -> bool:
    now = datetime.now(_ET)
    if now.weekday() >= 5:
        return False
    t = now.hour * 60 + now.minute
    return 9 * 60 + 30 <= t < 16 * 60


def build_regime(symbol: str) -> Regime:
    vix = _vix()
    return Regime(
        iv_rank=_iv_rank(vix),
        vix=vix,
        trend=_intraday_trend(symbol),
        time_of_day=_tod(),
        realized_vol=_realized_vol(symbol),
    )


# ── strategy pricing ──────────────────────────────────────────────────────────

def _rs(p: float, tick: float = 0.5) -> float:
    return round(p / tick) * tick


def price_legs(
    strategy:   StrategyType,
    spot:       float,
    vix:        float,
    dte:        int,
    contracts:  int,
    wing:       float = _WING_WIDTH,
    otm_pct:    float = _OTM_PCT,
) -> Optional[List[dict]]:
    """
    Returns a list of leg dicts ready to become Leg objects, or None.
    entry_price is Black-Scholes fair value (per share).
    """
    sig    = vix / 100.0 * 1.15
    T      = max(dte, 0.5) / 365
    expiry = str(date.today() + timedelta(days=dte))

    def leg(kind, action, strike, qty=contracts):
        fn = bs_put if kind == "put" else bs_call
        px = max(fn(spot, strike, T, sig), 0.01)
        return dict(kind=kind, action=action, strike=strike,
                    expiry=expiry, qty=qty, entry_price=round(px, 3),
                    occ_symbol="", order_id="")

    s = strategy

    if s == StrategyType.BULL_PUT:
        K1 = _rs(spot * (1 - otm_pct));  K2 = K1 - wing
        return [leg("put","sell",K1), leg("put","buy",K2)]

    if s == StrategyType.BEAR_CALL:
        K1 = _rs(spot * (1 + otm_pct));  K2 = K1 + wing
        return [leg("call","sell",K1), leg("call","buy",K2)]

    if s == StrategyType.IRON_CONDOR:
        Kp1 = _rs(spot*(1-otm_pct));  Kp2 = Kp1 - wing
        Kc1 = _rs(spot*(1+otm_pct));  Kc2 = Kc1 + wing
        return [leg("put","sell",Kp1), leg("put","buy",Kp2),
                leg("call","sell",Kc1), leg("call","buy",Kc2)]

    if s == StrategyType.BUTTERFLY:
        K = _rs(spot)
        # buy lower, sell 2× ATM, buy upper — net debit
        return [leg("put","buy",K - wing),
                leg("put","sell",K, qty=contracts*2),
                leg("put","buy",K + wing)]

    if s == StrategyType.BWB_PUT:
        # buy 1 ATM, sell 2 OTM, buy 1 wider OTM → net credit
        K  = _rs(spot)
        K1 = _rs(spot - wing)
        K2 = _rs(spot - wing * 2.5)
        return [leg("put","buy",K),
                leg("put","sell",K1, qty=contracts*2),
                leg("put","buy",K2)]

    if s == StrategyType.BULL_CALL_DEBIT:
        K1 = _rs(spot * 1.005);  K2 = K1 + wing
        return [leg("call","buy",K1), leg("call","sell",K2)]

    if s == StrategyType.BEAR_PUT_DEBIT:
        K1 = _rs(spot * 0.995);  K2 = K1 - wing
        return [leg("put","buy",K1), leg("put","sell",K2)]

    if s == StrategyType.JADE_LIZARD:
        # Defined-risk: wide put spread (more credit) + narrow call spread
        Kp1 = _rs(spot*(1-otm_pct*1.5));  Kp2 = Kp1 - wing*1.5
        Kc1 = _rs(spot*(1+otm_pct));       Kc2 = Kc1 + wing
        return [leg("put","sell",Kp1), leg("put","buy",Kp2),
                leg("call","sell",Kc1), leg("call","buy",Kc2)]

    if s == StrategyType.REV_JADE_LIZARD:
        Kc1 = _rs(spot*(1+otm_pct*1.5));  Kc2 = Kc1 + wing*1.5
        Kp1 = _rs(spot*(1-otm_pct));       Kp2 = Kp1 - wing
        return [leg("call","sell",Kc1), leg("call","buy",Kc2),
                leg("put","sell",Kp1), leg("put","buy",Kp2)]

    return None


_MAX_CONTRACTS = 10  # practical cap: avoids 147-leg Alpaca orders on cheap structures

def _contracts_for(account: float, risk_pct: float,
                   strategy: StrategyType, legs: List[dict]) -> int:
    """How many contracts fit in the risk budget (min 1, max _MAX_CONTRACTS)."""
    pos_dict     = {"legs": legs}
    max_risk_dol = account * risk_pct
    risk_per     = _max_risk(pos_dict)
    if risk_per <= 0:
        return 1
    return max(1, min(_MAX_CONTRACTS, int(max_risk_dol / risk_per)))


# ── leg current value (Black-Scholes mark-to-market) ──────────────────────────

def _mark_legs(legs: List[dict], spot: float, vix: float) -> List[dict]:
    """Return legs with updated current_price via B-S."""
    sig = vix / 100.0 * 1.15
    for l in legs:
        expiry = date.fromisoformat(l["expiry"])
        T      = max((expiry - date.today()).days, 0) / 365
        fn     = bs_put if l["kind"] == "put" else bs_call
        l      = dict(l)  # copy
        l["current_price"] = round(max(fn(spot, l["strike"], T, sig), 0.005), 4)
    return legs


def _position_pnl(pos_dict: dict, spot: float, vix: float) -> float:
    """Unrealized P&L for the open position."""
    legs = _mark_legs(pos_dict["legs"], spot, vix)
    pnl  = 0.0
    for l in legs:
        sign = 1 if l["action"] == "sell" else -1
        pnl += sign * (l["entry_price"] - l["current_price"]) * l["qty"] * _SHARES
    return pnl


def _close_cost(pos_dict: dict, spot: float, vix: float) -> float:
    """Cost to close all legs at current B-S prices."""
    legs = _mark_legs(pos_dict["legs"], spot, vix)
    cost = 0.0
    for l in legs:
        if l["action"] == "sell":
            cost += l["current_price"] * l["qty"] * _SHARES
        else:
            cost -= l["current_price"] * l["qty"] * _SHARES
    return cost


def _net_credit(pos_dict: dict) -> float:
    cr = 0.0
    for l in pos_dict["legs"]:
        sign = 1 if l["action"] == "sell" else -1
        cr  += sign * l["entry_price"] * l["qty"] * _SHARES
    return cr


def _max_risk(pos_dict: dict) -> float:
    """
    Max loss in dollars via numerical payoff scan at expiry.
    Correct for simple spreads, condors, butterflies, and BWB.
    Scans 500 spot prices between 85%-115% of the strike range.
    """
    legs = pos_dict["legs"]
    if not legs:
        return 0.0
    strikes = sorted(set(l["strike"] for l in legs))
    lo   = strikes[0]  * 0.85
    hi   = strikes[-1] * 1.15
    step = (hi - lo) / 500
    min_pnl = 0.0
    s = lo
    while s <= hi + step * 0.5:
        pnl = 0.0
        for l in legs:
            intr = max(l["strike"] - s, 0) if l["kind"] == "put" else max(s - l["strike"], 0)
            sign = 1 if l["action"] == "sell" else -1
            pnl += sign * (l["entry_price"] - intr) * l["qty"] * _SHARES
        min_pnl = min(min_pnl, pnl)
        s += step
    return abs(min_pnl)


# ── open / close / morph ──────────────────────────────────────────────────────

def _open_position(state: dict, strategy: StrategyType,
                   spot: float, vix: float, regime: Regime,
                   dte: int = _DEFAULT_DTE) -> Optional[dict]:
    legs = price_legs(strategy, spot, vix, dte,
                      contracts=1,  # placeholder; recalc below
                      wing=_WING_WIDTH, otm_pct=_OTM_PCT)
    if legs is None:
        return None

    contracts = _contracts_for(state["account_value"], _DEFAULT_RISK_PCT, strategy, legs)
    for l in legs:
        l["qty"] = contracts if l["qty"] == 1 else contracts * (l["qty"] // 1)

    nc = sum((l["entry_price"] if l["action"]=="sell" else -l["entry_price"])
             * l["qty"] * _SHARES for l in legs)
    mr = _max_risk({"legs": legs})

    pos = dict(
        strategy=strategy.value,
        entry_time=datetime.now(_ET).isoformat(),
        spot_entry=round(spot, 2),
        vix_entry=round(vix, 2),
        iv_rank_entry=round(regime.iv_rank, 1),
        dte_entry=dte,
        legs=legs,
        net_credit_total=round(nc, 2),
        max_risk_total=round(mr, 2),
    )
    _try_alpaca_open(pos)
    return pos


def _close_position(state: dict, pos: dict, spot: float, vix: float,
                    reason: str) -> dict:
    _try_alpaca_close(pos)
    nc  = _net_credit(pos)
    pnl = _position_pnl(pos, spot, vix)

    state["account_value"] += pnl
    state["total_pnl"]     += pnl
    if pnl >= 0:
        state["n_wins"]   = state.get("n_wins",   0) + 1
    else:
        state["n_losses"] = state.get("n_losses", 0) + 1

    state["closed_trades"].append({
        **pos,
        "close_time": datetime.now(_ET).isoformat(),
        "spot_close": round(spot, 2),
        "pnl":        round(pnl, 2),
        "reason":     reason,
    })
    state["open_position"] = None

    emoji = ":white_check_mark:" if pnl >= 0 else ":x:"
    msg = (
        f"{emoji} *Rotating Paper — CLOSE* ({reason})\n"
        f"> Strategy  : `{pos['strategy']}`\n"
        f"> Spot      : ${spot:.2f}  VIX: {vix:.1f}\n"
        f"> P&L       : ${pnl:+,.2f}\n"
        f"> Account   : ${state['account_value']:,.2f}  "
        f"(total {state['total_pnl']:+,.2f}  |  "
        f"{state.get('n_wins',0)}W / {state.get('n_losses',0)}L)"
    )
    send_message(msg)
    return state


def _apply_morph(state: dict, plan: MorphPlan, spot: float, vix: float,
                 regime: Regime) -> None:
    """Add new legs to the open position and update strategy type."""
    pos = state["open_position"]
    if pos is None:
        return

    dte    = (date.fromisoformat(pos["legs"][0]["expiry"]) - date.today()).days
    new_legs = price_legs(
        plan.target_strategy, spot, vix, dte,
        contracts=pos["legs"][0]["qty"],
        wing=_WING_WIDTH, otm_pct=_OTM_PCT,
    )
    if new_legs is None:
        return

    # Filter to only the legs that match the morph plan additions
    plan_kinds   = {(l["kind"], l["action"]) for l in plan.legs_to_add}
    legs_to_add  = [l for l in new_legs if (l["kind"], l["action"]) in plan_kinds]

    pos["legs"].extend(legs_to_add)
    pos["strategy"] = plan.target_strategy.value
    pos["net_credit_total"] = round(_net_credit(pos), 2)
    pos["max_risk_total"]   = round(_max_risk(pos), 2)

    state["day_trades_used"] = state.get("day_trades_used", 0) + plan.day_trades_used

    # Place Alpaca orders for only the new legs
    _try_alpaca_open_legs(legs_to_add)


# ── Alpaca order hooks (best-effort, never raise) ─────────────────────────────

def _try_alpaca_open(pos: dict) -> None:
    try:
        from src.live.alpaca_options import find_contract, place_single_leg, get_mid_price
        expiry = date.fromisoformat(pos["legs"][0]["expiry"])
        sym    = pos.get("symbol", "XSP")   # default XSP

        # Can't easily get symbol from pos dict; use the underlying from state later
        # For now, place each leg individually
        for l in pos["legs"]:
            occ = find_contract(sym, expiry, l["kind"].upper(), l["strike"])
            if occ:
                l["occ_symbol"] = occ
                mid = get_mid_price(occ) or l["entry_price"]
                if l["action"] == "sell":
                    place_single_leg(occ, "sell", l["qty"], mid * 0.98)
                else:
                    place_single_leg(occ, "buy",  l["qty"], mid * 1.02)
    except Exception as e:
        logger.warning("Alpaca open skipped: %s", e)


def _try_alpaca_open_legs(legs: List[dict]) -> None:
    try:
        from src.live.alpaca_options import find_contract, place_single_leg, get_mid_price
        for l in legs:
            expiry = date.fromisoformat(l["expiry"])
            occ    = find_contract("XSP", expiry, l["kind"].upper(), l["strike"])
            if not occ:
                occ = find_contract("SPY", expiry, l["kind"].upper(), l["strike"])
            if occ:
                l["occ_symbol"] = occ
                mid = get_mid_price(occ) or l["entry_price"]
                side_price = mid * 0.98 if l["action"] == "sell" else mid * 1.02
                place_single_leg(occ, l["action"], l["qty"], side_price)
    except Exception as e:
        logger.warning("Alpaca morph open skipped: %s", e)


def _try_alpaca_close(pos: dict) -> None:
    try:
        from src.live.alpaca_options import place_single_leg, get_mid_price
        for l in pos["legs"]:
            occ = l.get("occ_symbol")
            if not occ:
                continue
            mid = get_mid_price(occ)
            if mid and mid > 0.01:
                close_side  = "buy"  if l["action"] == "sell" else "sell"
                close_price = mid * 1.02 if close_side == "buy" else mid * 0.98
                place_single_leg(occ, close_side, l["qty"], close_price)
    except Exception as e:
        logger.warning("Alpaca close skipped: %s", e)


# ── main tick ─────────────────────────────────────────────────────────────────

def tick(symbol: Optional[str] = None,
         engine: Optional[RotatingEngine] = None) -> dict:
    """
    One evaluation cycle. Safe to call on a cron or in a loop.

    Returns a summary dict with keys: action, strategy, pnl_unreal, account_value.
    """
    if not _is_market_open():
        logger.info("Market closed — skipping tick")
        return {"action": "market_closed"}

    state  = _load()
    state  = _reset_dt_budget_if_new_week(state)
    symbol = symbol or state.get("symbol") or detect_symbol()
    state["symbol"] = symbol

    if engine is None:
        engine = RotatingEngine(
            symbol=symbol,
            account_size=state["account_value"],
            max_risk_pct=_DEFAULT_RISK_PCT,
            day_trade_budget=3 - state.get("day_trades_used", 0),
        )

    spot   = _spot(symbol)
    vix    = _vix()
    regime = build_regime(symbol)

    # ── force-close after 3 PM ET ────────────────────────────────────────────
    now_et = datetime.now(_ET)
    if now_et.hour >= _CLOSE_HOUR_ET and state["open_position"] is not None:
        state = _close_position(state, state["open_position"], spot, vix, "EOD_CLOSE")
        _save(state)
        return {"action": "eod_close", "account_value": state["account_value"]}

    # ── restore engine state from open position ───────────────────────────────
    if state["open_position"]:
        pos_dict = state["open_position"]
        legs     = [
            Leg(kind=l["kind"], action=l["action"], strike=l["strike"],
                expiry=date.fromisoformat(l["expiry"]), qty=l["qty"],
                entry_price=l["entry_price"], occ_symbol=l.get("occ_symbol",""),
                current_price=l.get("current_price", l["entry_price"]))
            for l in _mark_legs(pos_dict["legs"], spot, vix)
        ]
        engine.current_position = Position(
            strategy=StrategyType(pos_dict["strategy"]),
            symbol=symbol,
            legs=legs,
            entry_time=datetime.fromisoformat(pos_dict["entry_time"]),
            spot_entry=pos_dict["spot_entry"],
            vix_entry=pos_dict["vix_entry"],
        )

    # ── evaluate ──────────────────────────────────────────────────────────────
    cycle = engine.evaluate(regime)
    action = cycle.action

    if action in ("hold", "stand_aside"):
        unreal = (_position_pnl(state["open_position"], spot, vix)
                  if state["open_position"] else 0.0)
        logger.info("tick: %s | unrealized=$%.2f", action, unreal)
        _save(state)
        return {"action": action, "account_value": state["account_value"],
                "pnl_unreal": round(unreal, 2)}

    if action in ("close_tp", "close_sl", "close_reopen"):
        if state["open_position"]:
            reason = "TP" if action == "close_tp" else ("SL" if action == "close_sl" else "ROTATE")
            state  = _close_position(state, state["open_position"], spot, vix, reason)
            state["day_trades_used"] = state.get("day_trades_used", 0) + 1

    if action in ("open", "close_reopen"):
        # Skip if the available premium is too thin to overcome commission drag.
        # At VIX < 20 a 2-wide credit spread collects ~$25-30 — below this threshold
        # commissions eat the entire edge, so stand aside instead.
        _MIN_CREDIT = 25.0
        if cycle.recommended.value not in ("CASH", "BUTTERFLY",
                                            "BULL_CALL_DEBIT", "BEAR_PUT_DEBIT"):
            probe = price_legs(cycle.recommended, spot, vix, _DEFAULT_DTE, 1)
            if probe is not None:
                probe_nc = sum(
                    (l["entry_price"] if l["action"] == "sell" else -l["entry_price"])
                    * l["qty"] * _SHARES for l in probe
                )
                if probe_nc < _MIN_CREDIT:
                    logger.info(
                        "Premium too thin (%.2f < %.0f) — standing aside instead of %s",
                        probe_nc, _MIN_CREDIT, cycle.recommended.value,
                    )
                    _save(state)
                    return {"action": "stand_aside_low_premium",
                            "account_value": state["account_value"],
                            "pnl_unreal": 0.0,
                            "iv_rank": round(regime.iv_rank, 1),
                            "vix": round(vix, 1)}

        pos = _open_position(state, cycle.recommended, spot, vix, regime)
        if pos:
            pos["symbol"]        = symbol
            state["open_position"] = pos
            msg = (
                f":green_circle: *Rotating Paper — OPEN*\n"
                f"> Strategy : `{cycle.recommended.value}`\n"
                f"> Spot     : ${spot:.2f}  VIX: {vix:.1f}  IV Rank: {regime.iv_rank:.0f}\n"
                f"> Trend    : {regime.trend:+.2f}  ToD: {regime.time_of_day:.0%}\n"
                f"> Credit   : ${pos['net_credit_total']:+,.2f}  "
                f"Max risk: ${pos['max_risk_total']:,.2f}\n"
                f"> Account  : ${state['account_value']:,.2f}  "
                f"| DT used: {state.get('day_trades_used',0)}/3"
            )
            send_message(msg)

    if action == "morph" and cycle.morph_plan:
        _apply_morph(state, cycle.morph_plan, spot, vix, regime)
        pos = state["open_position"]
        msg = (
            f":arrows_counterclockwise: *Rotating Paper — MORPH*\n"
            f"> {cycle.current.value} -> `{cycle.morph_plan.target_strategy.value}`\n"
            f"> Spot: ${spot:.2f}  VIX: {vix:.1f}  | {cycle.reason}\n"
            f"> DT used: {state.get('day_trades_used',0)}/3"
        )
        send_message(msg)

    _save(state)
    unreal = (_position_pnl(state["open_position"], spot, vix)
              if state["open_position"] else 0.0)
    return {
        "action":        action,
        "strategy":      cycle.recommended.value,
        "account_value": state["account_value"],
        "pnl_unreal":    round(unreal, 2),
        "iv_rank":       round(regime.iv_rank, 1),
        "trend":         round(regime.trend, 2),
    }


# ── public helpers ────────────────────────────────────────────────────────────

def status() -> dict:
    state  = _load()
    symbol = state.get("symbol", "XSP")
    pos    = state.get("open_position")
    out    = {
        "symbol":        symbol,
        "account_value": state["account_value"],
        "total_pnl":     state["total_pnl"],
        "n_wins":        state.get("n_wins", 0),
        "n_losses":      state.get("n_losses", 0),
        "day_trades_used": state.get("day_trades_used", 0),
        "open_position": None,
        "unrealized_pnl": None,
    }
    if pos:
        try:
            spot = _spot(symbol)
            vix  = _vix()
            out["open_position"]  = pos
            out["unrealized_pnl"] = round(_position_pnl(pos, spot, vix), 2)
        except Exception:
            out["open_position"] = pos
    return out


def reset(capital: float = _DEFAULT_ACCOUNT, symbol: Optional[str] = None) -> None:
    sym = symbol or detect_symbol()
    _save(_blank_state(capital, sym))
    logger.info("State reset: capital=$%.2f symbol=%s", capital, sym)
