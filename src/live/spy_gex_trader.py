"""
Automated SPY options executor for GEX trade signals.

Translates directional GEX signals into SPY debit spreads:
  LONG  signal -> Bull Call Debit Spread (buy ATM call, sell call at target)
  SHORT signal -> Bear Put  Debit Spread (buy ATM put,  sell put  at target)

Advantages over equity bracket orders on a small account:
  - Max loss = debit paid (no stop-loss order needed)
  - Leverage: 1 contract controls 100 shares for a fraction of the cost
  - Defined risk: can size precisely to 1% account risk

Lifecycle:
  9:45 AM  -> open_trade(): place debit spread
  3:15 PM  -> sync_state(): check TP (50% of max gain), record auto-fills
  3:45 PM  -> close_trade(): close any remaining legs at market

State: data/spy_gex_state.json
"""
import json
import logging
import math
import os
from datetime import date, datetime, timedelta
from typing import Optional

import pytz
import yfinance as yf

from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY, IS_PAPER, DATA_DIR, GEX_RISK_PCT
from src.analysis.gex_trader import TradeSignal
from src.notifications.slack_notifier import send_message

logger     = logging.getLogger(__name__)
ET         = pytz.timezone("America/New_York")
STATE_PATH = os.path.join(DATA_DIR, "spy_gex_state.json")

TP_PCT = 0.50   # close when unrealized gain >= 50% of max gain


# ── state I/O ─────────────────────────────────────────────────────────────────

def _load() -> dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"open_trade": None, "closed_trades": [],
            "total_pnl": 0.0, "n_wins": 0, "n_losses": 0}


def _save(state: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ── Alpaca client (lazy) ──────────────────────────────────────────────────────

_client = None


def _trading():
    global _client
    if _client is None:
        from alpaca.trading.client import TradingClient
        _client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=IS_PAPER)
    return _client


# ── helpers ───────────────────────────────────────────────────────────────────

def _spy_price() -> float:
    try:
        info = yf.Ticker("SPY").fast_info
        px   = info.get("last_price") or info.get("regularMarketPrice")
        if px:
            return float(px)
    except Exception:
        pass
    df  = yf.download("SPY", period="2d", interval="1d", auto_adjust=True, progress=False)
    col = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
    v   = col.iloc[-1]
    return float(v.item() if hasattr(v, "item") else v)


def _next_spy_expiry(today: date) -> date:
    """SPY options expire Mon / Wed / Fri. Return today if it's one, else next one."""
    wd = today.weekday()   # 0=Mon 1=Tue 2=Wed 3=Thu 4=Fri
    if wd in (0, 2, 4):
        return today
    return today + timedelta(days=1)   # Tue->Wed, Thu->Fri


def _bs_price(spot: float, strike: float, T: float, sigma: float, is_call: bool) -> float:
    """Black-Scholes price for a European option (r=RF_RATE)."""
    from scipy.stats import norm
    RF = 0.045
    if T <= 0 or sigma <= 0:
        intrinsic = max(spot - strike, 0) if is_call else max(strike - spot, 0)
        return intrinsic
    d1 = (math.log(spot / strike) + (RF + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if is_call:
        return spot * norm.cdf(d1) - strike * math.exp(-RF * T) * norm.cdf(d2)
    else:
        return strike * math.exp(-RF * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)


def _build_spread(signal: TradeSignal, spot: float, vix: float,
                  equity: float) -> Optional[dict]:
    """
    Compute the option legs for a debit spread.
    Returns a dict with strike/price/contract details, or None if unviable.
    """
    today  = date.today()
    expiry = _next_spy_expiry(today)
    T      = max((expiry - today).days + 0.5, 0.5) / 365   # at least half a day
    sigma  = (vix / 100) * 1.20

    is_call = signal.direction == "LONG"

    if is_call:
        # Bull call: buy lower strike (ATM), sell higher strike (at target)
        long_strike  = round(spot)
        short_strike = round(signal.target)
        if short_strike <= long_strike:
            short_strike = long_strike + 1
        long_price  = _bs_price(spot, long_strike,  T, sigma, True)
        short_price = _bs_price(spot, short_strike, T, sigma, True)
        spread_type = "BULL_CALL"
        opt_type    = "CALL"
    else:
        # Bear put: buy higher strike (ATM), sell lower strike (at target)
        long_strike  = round(spot)
        short_strike = round(signal.target)
        if short_strike >= long_strike:
            short_strike = long_strike - 1
        long_price  = _bs_price(spot, long_strike,  T, sigma, False)
        short_price = _bs_price(spot, short_strike, T, sigma, False)
        spread_type = "BEAR_PUT"
        opt_type    = "PUT"

    debit        = max(long_price - short_price, 0.01)
    spread_width = abs(short_strike - long_strike)
    max_gain     = spread_width - debit

    if max_gain <= 0:
        logger.warning("Spread max_gain <= 0 (debit=%.2f width=%d) — skipping", debit, spread_width)
        return None

    # Size: risk 1% of equity = debit × 100 × contracts
    risk_dollars = equity * GEX_RISK_PCT
    contracts    = max(int(risk_dollars / (debit * 100)), 1)

    return {
        "spread_type":   spread_type,
        "opt_type":      opt_type,
        "long_strike":   long_strike,
        "short_strike":  short_strike,
        "expiry":        str(expiry),
        "debit_paid":    round(debit, 2),
        "spread_width":  spread_width,
        "max_gain":      round(max_gain, 2),
        "contracts":     contracts,
        "tp_target":     round(debit + max_gain * TP_PCT, 2),   # spread value at TP
        "sigma":         round(sigma, 4),
    }


# ── open ──────────────────────────────────────────────────────────────────────

def open_trade(signal: TradeSignal, equity: float,
               vix: float = 17.0,
               gex_regime: str = "", vanna_signal: str = "") -> Optional[dict]:
    """
    Build and place a SPY debit spread for the given signal.
    Returns the trade dict saved to state, or None on failure.
    """
    state = _load()
    if state["open_trade"] is not None:
        logger.warning("GEX options trade already open — skipping")
        return None

    spot   = _spy_price()
    spread = _build_spread(signal, spot, vix, equity)
    if spread is None:
        return None

    expiry   = date.fromisoformat(spread["expiry"])
    opt_type = spread["opt_type"]

    from src.live.alpaca_options import find_contract, place_spread

    long_sym  = find_contract("SPY", expiry, opt_type, spread["long_strike"],  tolerance=1.5)
    short_sym = find_contract("SPY", expiry, opt_type, spread["short_strike"], tolerance=1.5)

    if not long_sym or not short_sym:
        logger.error("Could not find SPY option contracts — aborting trade")
        send_message(":rotating_light: *GEX Options* — contract lookup failed, no trade placed")
        return None

    # place_spread: sells short_sym, buys long_sym
    result = place_spread(
        short_symbol=short_sym,
        long_symbol=long_sym,
        qty=spread["contracts"],
        short_credit=spread["debit_paid"] * 0.35,
        long_debit=spread["debit_paid"],
    )

    trade = {
        "setup":           signal.setup,
        "direction":       signal.direction,
        "spread_type":     spread["spread_type"],
        "long_symbol":     long_sym,
        "short_symbol":    short_sym,
        "long_strike":     spread["long_strike"],
        "short_strike":    spread["short_strike"],
        "expiry":          spread["expiry"],
        "contracts":       spread["contracts"],
        "debit_paid":      spread["debit_paid"],
        "spread_width":    spread["spread_width"],
        "max_gain":        spread["max_gain"],
        "tp_target":       spread["tp_target"],
        "long_order_id":   result.get("long_order_id"),
        "short_order_id":  result.get("short_order_id"),
        "spot_at_entry":   round(spot, 2),
        "rr":              round(signal.rr, 2),
        "confidence":      signal.confidence,
        "opened_at":       datetime.now(ET).isoformat(),
        "gex_regime":      gex_regime,
        "vanna_signal":    vanna_signal,
    }

    state["open_trade"] = trade
    _save(state)

    max_loss  = round(spread["debit_paid"] * 100 * spread["contracts"], 2)
    max_profit = round(spread["max_gain"]  * 100 * spread["contracts"], 2)
    logger.info("GEX options trade opened: %s %s  debit=%.2f  contracts=%d",
                spread["spread_type"], signal.setup, spread["debit_paid"], spread["contracts"])

    send_message(
        f":dart: *GEX Options — OPEN* ({spread['spread_type']})\n"
        f">  Setup     : `{signal.setup}` ({signal.confidence} confidence)\n"
        f">  SPY       : ${spot:.2f}  |  Expiry: {spread['expiry']}\n"
        f">  Strikes   : ${spread['long_strike']} / ${spread['short_strike']}\n"
        f">  Debit     : `${spread['debit_paid']:.2f}/sh` x {spread['contracts']} contracts\n"
        f">  Max loss  : `${max_loss:,.2f}`  |  Max gain: `${max_profit:,.2f}`\n"
        f">  TP target : spread worth `${spread['tp_target']:.2f}/sh`"
    )
    return trade


# ── sync / TP check ───────────────────────────────────────────────────────────

def sync_state() -> Optional[dict]:
    """
    Check live option prices for TP. If expired/worthless, record the outcome.
    Returns closed trade dict if position was closed, else None.
    """
    state = _load()
    trade = state.get("open_trade")
    if trade is None:
        return None

    from src.live.alpaca_options import get_mid_price

    long_mid  = get_mid_price(trade["long_symbol"])
    short_mid = get_mid_price(trade["short_symbol"])

    # If quotes unavailable, check if expiry has passed
    if long_mid is None and short_mid is None:
        expiry = date.fromisoformat(trade["expiry"])
        if date.today() > expiry:
            # Expired — settle at intrinsic
            spot   = _spy_price()
            return _settle_at_expiry(state, trade, spot)
        logger.warning("Could not fetch option quotes for sync — skipping")
        return None

    long_mid  = long_mid  or 0.0
    short_mid = short_mid or 0.0
    current_spread_value = long_mid - short_mid   # current value of the spread

    debit    = trade["debit_paid"]
    max_gain = trade["max_gain"]
    unrealized = (current_spread_value - debit) * 100 * trade["contracts"]

    logger.info("GEX options sync: spread_value=%.2f debit=%.2f unrealized=%.2f",
                current_spread_value, debit, unrealized)

    # TP check: close when spread worth >= tp_target (50% of max gain captured)
    if current_spread_value >= trade["tp_target"]:
        logger.info("GEX options TP hit: %.2f >= %.2f", current_spread_value, trade["tp_target"])
        exit_value = current_spread_value
        pnl = (exit_value - debit) * 100 * trade["contracts"]
        return _record_close(state, trade, exit_value, pnl, "TAKE_PROFIT")

    return None


def _settle_at_expiry(state: dict, trade: dict, spot: float) -> dict:
    """Compute intrinsic P&L at expiry."""
    ls, ss = trade["long_strike"], trade["short_strike"]
    if trade["spread_type"] == "BULL_CALL":
        long_val  = max(spot - ls, 0.0)
        short_val = max(spot - ss, 0.0)
    else:
        long_val  = max(ls - spot, 0.0)
        short_val = max(ss - spot, 0.0)

    exit_value = long_val - short_val
    pnl = (exit_value - trade["debit_paid"]) * 100 * trade["contracts"]
    outcome = "EXPIRED_PROFIT" if pnl > 0 else "EXPIRED_LOSS"
    return _record_close(state, trade, exit_value, pnl, outcome)


# ── force close ───────────────────────────────────────────────────────────────

def close_trade(reason: str = "FORCE_CLOSE") -> Optional[dict]:
    """
    Close both option legs at current mid prices.
    Always tries sync first (TP may already have fired).
    """
    already = sync_state()
    if already:
        return already

    state = _load()
    trade = state.get("open_trade")
    if trade is None:
        return None

    from src.live.alpaca_options import close_spread, get_mid_price

    close_spread(trade["short_symbol"], trade["long_symbol"], trade["contracts"])

    long_mid  = get_mid_price(trade["long_symbol"])  or 0.0
    short_mid = get_mid_price(trade["short_symbol"]) or 0.0
    exit_value = max(long_mid - short_mid, 0.0)
    pnl = (exit_value - trade["debit_paid"]) * 100 * trade["contracts"]
    return _record_close(state, trade, exit_value, pnl, reason)


# ── record & notify ───────────────────────────────────────────────────────────

def _record_close(state: dict, trade: dict, exit_value: float,
                  pnl: float, outcome: str) -> dict:
    closed = {
        **trade,
        "exit_spread_value": round(exit_value, 2),
        "pnl":               round(pnl, 2),
        "outcome":           outcome,
        "closed_at":         datetime.now(ET).isoformat(),
    }
    state["closed_trades"].append(closed)
    state["total_pnl"]  = round(state["total_pnl"] + pnl, 2)
    state["n_wins"]     = state.get("n_wins",   0) + (1 if pnl > 0 else 0)
    state["n_losses"]   = state.get("n_losses", 0) + (1 if pnl <= 0 else 0)
    state["open_trade"] = None
    _save(state)

    icon = ":white_check_mark:" if pnl > 0 else ":x:"
    w, l = state["n_wins"], state["n_losses"]
    max_loss = trade["debit_paid"] * 100 * trade["contracts"]
    send_message(
        f"{icon} *GEX Options — CLOSE* ({outcome})\n"
        f">  {trade['spread_type']}  {trade['long_strike']}/{trade['short_strike']}\n"
        f">  Entry debit `${trade['debit_paid']:.2f}`  ->  Exit `${exit_value:.2f}`\n"
        f">  P&L: `${pnl:+,.2f}`  (max was ${max_loss:,.2f})\n"
        f">  Total: `${state['total_pnl']:+,.2f}`  |  {w}W / {l}L"
    )
    logger.info("GEX options closed: %s P&L=%.2f outcome=%s", trade["spread_type"], pnl, outcome)
    return closed


def get_status() -> dict:
    state = _load()
    return {
        "open_trade": state["open_trade"],
        "total_pnl":  state["total_pnl"],
        "n_wins":     state.get("n_wins",   0),
        "n_losses":   state.get("n_losses", 0),
        "n_closed":   len(state["closed_trades"]),
    }
