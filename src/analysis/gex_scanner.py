"""
Intraday dealer Greeks exposure scanner for SPY.

Computes Gamma Exposure (GEX), Vanna, and Charm from the live SPY options
chain to identify intraday price dynamics and key support/resistance levels.

GEX > 0  → dealers long gamma  → they sell rips / buy dips → vol suppression, price pins
GEX < 0  → dealers short gamma → they chase moves          → vol amplification, trending
Vanna + VIX drop → dealer buying  (bullish intraday tailwind)
Vanna + VIX rise → dealer selling (bearish intraday headwind)
Charm near expiry → OTM delta decay → dealer unwind flows (AM typically buying on put-heavy Fridays)
"""
import logging
import math
import os
import pickle
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import numpy as np
import yfinance as yf
from scipy.stats import norm

from config.settings import GEX_CHAIN_DIR

logger = logging.getLogger(__name__)


# ── chain cache (one file per calendar date) ──────────────────────────────────

def _cache_path(as_of: date) -> str:
    os.makedirs(GEX_CHAIN_DIR, exist_ok=True)
    return os.path.join(GEX_CHAIN_DIR, f"spy_chain_{as_of}.pkl")


def save_chain(contracts: list[dict], as_of: date | None = None) -> None:
    """Persist a fetched options chain so subsequent calls skip the API."""
    as_of = as_of or date.today()
    with open(_cache_path(as_of), "wb") as f:
        pickle.dump({"date": as_of, "contracts": contracts}, f)
    logger.info("Chain cached: %d contracts for %s", len(contracts), as_of)


def load_chain(as_of: date | None = None) -> list[dict] | None:
    """Return cached contracts for as_of date, or None if not cached."""
    as_of = as_of or date.today()
    p = _cache_path(as_of)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "rb") as f:
            data = pickle.load(f)
        contracts = data.get("contracts", [])
        logger.info("Chain loaded from cache: %d contracts for %s", len(contracts), as_of)
        return contracts
    except Exception as e:
        logger.warning("Cache read failed (%s): %s — will re-fetch", p, e)
        return None

_RF = 0.045    # risk-free rate
_SHARES = 100  # shares per contract


# ── BS greeks ─────────────────────────────────────────────────────────────────

def _d1d2(S: float, K: float, T: float, sig: float):
    if T <= 1e-6 or sig <= 1e-6:
        return 0.0, 0.0
    d1 = (math.log(S / K) + (_RF + 0.5 * sig**2) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    return d1, d2


def _gamma_bs(S: float, K: float, T: float, sig: float) -> float:
    d1, _ = _d1d2(S, K, T, sig)
    return norm.pdf(d1) / (S * sig * math.sqrt(T)) if T > 1e-6 and sig > 1e-6 else 0.0


def _vanna_bs(S: float, K: float, T: float, sig: float) -> float:
    """dDelta/dVol — same sign for calls and puts."""
    d1, d2 = _d1d2(S, K, T, sig)
    return -norm.pdf(d1) * d2 / sig if T > 1e-6 and sig > 1e-6 else 0.0


def _charm_bs(S: float, K: float, T: float, sig: float, is_call: bool) -> float:
    """dDelta/dTime — delta bleed per day."""
    if T <= 1e-6 or sig <= 1e-6:
        return 0.0
    d1, d2 = _d1d2(S, K, T, sig)
    sqrtT = math.sqrt(T)
    # ∂d1/∂T = (r + σ²/2)/(σ√T) - d1/(2T) normalised to per-day
    dd1_dT = (_RF / (sig * sqrtT)) - (d1 / (2 * T))
    charm = norm.pdf(d1) * dd1_dT / 365  # scale to per-day
    return charm if is_call else -charm


# ── data fetch ────────────────────────────────────────────────────────────────

def _spot_and_vix():
    """Return (spot, vix_now, vix_prev_close)."""
    spy_info = yf.Ticker("SPY").fast_info
    spot     = float(spy_info.get("last_price") or spy_info.get("regularMarketPrice") or 0)
    if not spot:
        df   = yf.download("SPY", period="2d", interval="1d", auto_adjust=True, progress=False)
        col  = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
        spot = float(col.iloc[-1].item() if hasattr(col.iloc[-1], "item") else col.iloc[-1])

    vix_df = yf.download("^VIX", period="5d", interval="1d", auto_adjust=True, progress=False)
    col    = vix_df["Close"] if "Close" in vix_df.columns else vix_df.iloc[:, 0]
    vix_now  = float(col.iloc[-1].item() if hasattr(col.iloc[-1], "item") else col.iloc[-1])
    vix_prev = float(col.iloc[-2].item() if hasattr(col.iloc[-2], "item") else col.iloc[-2]) if len(col) > 1 else vix_now

    return spot, vix_now, vix_prev


def _fetch_chain_from_api(spot: float, n_expiries: int = 3) -> list[dict]:
    """Pull SPY option chain from yfinance. Call sparingly — use load/save_chain instead."""
    ticker = yf.Ticker("SPY")
    today  = date.today()
    rows   = []

    try:
        exps = list(ticker.options)[:n_expiries]
    except Exception as e:
        logger.error("Failed to fetch SPY expirations: %s", e)
        return rows

    for exp_str in exps:
        try:
            exp_date = date.fromisoformat(exp_str)
            T = max((exp_date - today).days, 0) / 365
            chain = ticker.option_chain(exp_str)

            for df, is_call in [(chain.calls, True), (chain.puts, False)]:
                if df is None or df.empty:
                    continue
                for _, row in df.iterrows():
                    strike = float(row["strike"])
                    if not (spot * 0.92 <= strike <= spot * 1.08):
                        continue
                    oi = int(row.get("openInterest") or 0)
                    iv = float(row.get("impliedVolatility") or 0)
                    if oi < 10 or iv < 0.01:
                        continue
                    rows.append(dict(strike=strike, oi=oi, iv=iv, T=T, is_call=is_call))
        except Exception as e:
            logger.warning("Chain fetch failed for %s: %s", exp_str, e)

    return rows


def _fetch_chain(spot: float, n_expiries: int = 3,
                 as_of: date | None = None) -> list[dict]:
    """
    Return SPY option chain contracts, using today's disk cache when available.
    First call of the day hits yfinance and saves; all subsequent calls are instant.
    Pass as_of for historical replay (backtesting).
    """
    as_of = as_of or date.today()
    cached = load_chain(as_of)
    if cached is not None:
        return cached
    rows = _fetch_chain_from_api(spot, n_expiries)
    if rows:
        save_chain(rows, as_of)
    return rows


# ── exposure aggregation ──────────────────────────────────────────────────────

@dataclass
class GEXResult:
    spot:             float
    vix:              float
    vix_prev:         float
    net_gex_bn:       float          # net GEX in $ billions (positive = long gamma)
    gex_regime:       str            # POSITIVE_GAMMA | NEGATIVE_GAMMA
    gamma_wall:       float          # strike nearest spot with highest net GEX
    put_wall:         float          # highest put GEX below spot
    call_wall:        float          # highest call GEX above spot
    flip_level:       float          # where cumulative GEX crosses zero
    net_vanna_bn:     float          # net Vanna in $ billions
    vanna_signal:     str            # BULLISH | BEARISH | NEUTRAL
    net_charm:        float          # net charm notional (+ = dealer buying, - = selling)
    charm_signal:     str            # BUYING_PRESSURE | SELLING_PRESSURE | NEUTRAL
    top_levels:       list = field(default_factory=list)  # [(strike, gex_bn), ...]
    dte_nearest:      int  = 0


def compute_exposures(spot: float, vix: float, vix_prev: float,
                      contracts: list[dict]) -> GEXResult:
    """
    Aggregate gamma/vanna/charm exposures across the options chain.
    """
    gex_by_strike:   dict[float, float] = {}
    vanna_by_strike: dict[float, float] = {}
    charm_total = 0.0

    call_gex_by_strike: dict[float, float] = {}
    put_gex_by_strike:  dict[float, float] = {}

    dte_nearest = 999

    for c in contracts:
        K, oi, iv, T, is_call = c["strike"], c["oi"], c["iv"], c["T"], c["is_call"]
        dte = round(T * 365)
        dte_nearest = min(dte_nearest, dte)

        gx  = _gamma_bs(spot, K, T, iv) * oi * _SHARES * (spot ** 2) / 1e9
        vn  = _vanna_bs(spot, K, T, iv) * oi * _SHARES * spot / 1e9
        ch  = _charm_bs(spot, K, T, iv, is_call) * oi * _SHARES

        # Dealer convention: dealers typically short calls to retail (positive call OI = dealer short)
        # Standard GEX: calls contribute positive, puts negative
        signed_gex = gx if is_call else -gx
        gex_by_strike[K]   = gex_by_strike.get(K, 0.0)   + signed_gex
        vanna_by_strike[K] = vanna_by_strike.get(K, 0.0) + (vn if is_call else -vn)
        charm_total        += ch if is_call else -ch

        if is_call:
            call_gex_by_strike[K] = call_gex_by_strike.get(K, 0.0) + gx
        else:
            put_gex_by_strike[K]  = put_gex_by_strike.get(K, 0.0) + gx

    if not gex_by_strike:
        return GEXResult(spot=spot, vix=vix, vix_prev=vix_prev,
                         net_gex_bn=0, gex_regime="UNKNOWN",
                         gamma_wall=spot, put_wall=spot, call_wall=spot, flip_level=spot,
                         net_vanna_bn=0, vanna_signal="NEUTRAL",
                         net_charm=0, charm_signal="NEUTRAL")

    net_gex    = sum(gex_by_strike.values())
    net_vanna  = sum(vanna_by_strike.values())
    gex_regime = "POSITIVE_GAMMA" if net_gex > 0 else "NEGATIVE_GAMMA"

    # Gamma wall: strongest positive-GEX strike near spot (pin level)
    near_pos = {k: v for k, v in gex_by_strike.items()
                if abs(k - spot) / spot < 0.06 and v > 0}
    gamma_wall = max(near_pos, key=lambda k: near_pos[k], default=spot)

    # Call wall: strongest call GEX above spot (resistance)
    calls_above = {k: v for k, v in call_gex_by_strike.items() if k > spot}
    call_wall   = max(calls_above, key=lambda k: calls_above[k], default=spot * 1.02)

    # Put wall: strongest put GEX below spot (support)
    puts_below = {k: v for k, v in put_gex_by_strike.items() if k < spot}
    put_wall   = max(puts_below, key=lambda k: puts_below[k], default=spot * 0.98)

    # Flip level: strike where cumulative GEX (sorted ascending) changes sign
    flip_level = spot
    cum = 0.0
    prev_k = None
    prev_cum = 0.0
    for k, gex in sorted(gex_by_strike.items()):
        prev_cum = cum
        cum += gex
        if prev_k is not None and prev_cum * cum < 0:
            flip_level = (prev_k + k) / 2   # midpoint of sign-change bracket
            break
        prev_k = k

    # Vanna signal
    vix_chg_pct = (vix - vix_prev) / vix_prev if vix_prev else 0
    if abs(vix_chg_pct) < 0.02 or abs(net_vanna) < 0.5:
        vanna_signal = "NEUTRAL"
    elif vix_chg_pct < 0 and net_vanna > 0:
        vanna_signal = "BULLISH"   # VIX falling + positive vanna → dealer buying
    elif vix_chg_pct > 0 and net_vanna > 0:
        vanna_signal = "BEARISH"   # VIX rising  + positive vanna → dealer selling
    elif vix_chg_pct < 0 and net_vanna < 0:
        vanna_signal = "BEARISH"
    else:
        vanna_signal = "BULLISH"

    # Charm signal
    if abs(charm_total) < 1000:
        charm_signal = "NEUTRAL"
    elif charm_total > 0:
        charm_signal = "BUYING_PRESSURE"
    else:
        charm_signal = "SELLING_PRESSURE"

    # Top 6 levels by absolute GEX
    top_levels = sorted(gex_by_strike.items(), key=lambda x: abs(x[1]), reverse=True)[:6]

    return GEXResult(
        spot=spot, vix=vix, vix_prev=vix_prev,
        net_gex_bn=round(net_gex, 3), gex_regime=gex_regime,
        gamma_wall=gamma_wall, put_wall=put_wall, call_wall=call_wall,
        flip_level=flip_level,
        net_vanna_bn=round(net_vanna, 3), vanna_signal=vanna_signal,
        net_charm=round(charm_total, 0), charm_signal=charm_signal,
        top_levels=top_levels, dte_nearest=dte_nearest if dte_nearest < 999 else 0,
    )


def scan(as_of: date | None = None,
         spot: float | None = None,
         vix: float | None = None,
         vix_prev: float | None = None) -> Optional[GEXResult]:
    """
    Fetch live data and return a GEXResult.
    Pass as_of + spot/vix/vix_prev for historical replay (backtesting from cached chains).
    Returns None on failure.
    """
    try:
        if spot is None or vix is None:
            spot, vix, vix_prev = _spot_and_vix()
        if not spot or not vix:
            logger.error("Could not fetch SPY/VIX prices")
            return None
        contracts = _fetch_chain(spot, as_of=as_of)
        if not contracts:
            logger.warning("No option chain data — chain may be empty or market closed")
            return None
        result = compute_exposures(spot, vix, vix_prev or vix, contracts)
        logger.info("GEX scan: net_gex=%.2fbn regime=%s gamma_wall=%.1f vanna=%s",
                    result.net_gex_bn, result.gex_regime, result.gamma_wall, result.vanna_signal)
        return result
    except Exception as e:
        logger.error("GEX scan failed: %s", e)
        return None


# ── Slack message formatter ───────────────────────────────────────────────────

def format_gex_message(result: GEXResult, session: str = "morning") -> str:
    """
    Build a Slack message summarising dealer Greek exposures.
    session: 'morning' | 'afternoon'
    """
    session_label = "Morning Scan" if session == "morning" else "Afternoon Scan"
    vix_chg  = result.vix - result.vix_prev
    vix_arrow = ":arrow_up:" if vix_chg > 0.3 else (":arrow_down:" if vix_chg < -0.3 else ":left_right_arrow:")
    gex_arrow = ":green_circle:" if result.gex_regime == "POSITIVE_GAMMA" else ":red_circle:"
    vanna_arrow = {"BULLISH": ":chart_with_upward_trend:", "BEARISH": ":chart_with_downwards_trend:", "NEUTRAL": ":white_circle:"}[result.vanna_signal]
    charm_arrow = {"BUYING_PRESSURE": ":arrow_up_small:", "SELLING_PRESSURE": ":arrow_down_small:", "NEUTRAL": ":white_circle:"}[result.charm_signal]

    gex_desc = ("vol suppressor — dealers sell rips / buy dips"
                if result.gex_regime == "POSITIVE_GAMMA"
                else "vol amplifier — dealers chase moves")

    vanna_desc = {
        "BULLISH":  f"VIX {vix_chg:+.1f} -> dealer BUYING pressure",
        "BEARISH":  f"VIX {vix_chg:+.1f} -> dealer SELLING pressure",
        "NEUTRAL":  f"VIX {vix_chg:+.1f} (small, limited flow)",
    }[result.vanna_signal]

    charm_desc = {
        "BUYING_PRESSURE":  "OTM puts losing delta -> dealer hedge unwind (buy)",
        "SELLING_PRESSURE": "OTM calls losing delta -> dealer hedge unwind (sell)",
        "NEUTRAL":          "Minimal charm effect today",
    }[result.charm_signal]

    lines = [
        f":bar_chart: *SPY Greeks — {session_label}* | {datetime.now().strftime('%Y-%m-%d %H:%M')} ET",
        f"",
        f"*SPY:* ${result.spot:.2f}  |  *VIX:* {result.vix:.1f} {vix_arrow} ({vix_chg:+.1f})",
        f"",
        f"{gex_arrow} *Gamma Exposure (GEX)*",
        f">  Net GEX     : `{result.net_gex_bn:+.2f}B`  ({gex_desc})",
        f">  Gamma wall  : `${result.gamma_wall:.1f}`",
        f">  Call wall   : `${result.call_wall:.1f}`  (resistance)",
        f">  Put wall    : `${result.put_wall:.1f}`  (support)",
        f">  Flip level  : `${result.flip_level:.1f}`  (regime change if crossed)",
        f"",
        f"{vanna_arrow} *Vanna*",
        f">  Net Vanna   : `{result.net_vanna_bn:+.2f}B`  |  {vanna_desc}",
        f"",
        f"{charm_arrow} *Charm*  (DTE: {result.dte_nearest}d)",
        f">  {charm_desc}",
        f"",
        f":pushpin: *Top Gamma Levels*",
    ]

    for strike, gex in result.top_levels[:5]:
        marker = " <- SPY" if abs(strike - result.spot) / result.spot < 0.005 else ""
        bar = (":green_square:" if gex > 0 else ":red_square:") * min(int(abs(gex) * 3), 5)
        lines.append(f">  `${strike:.1f}` : `{gex:+.3f}B` {bar}{marker}")

    # Session-specific advisory
    if session == "morning":
        if result.gex_regime == "POSITIVE_GAMMA" and result.vanna_signal == "BULLISH":
            advisory = ":bulb: *AM bias:* Pinned + vanna tailwind — low vol drift higher likely"
        elif result.gex_regime == "NEGATIVE_GAMMA" and result.vanna_signal == "BEARISH":
            advisory = ":warning: *AM bias:* Negative gamma + vanna headwind — wide ranges, sell stops vulnerable"
        elif result.gex_regime == "NEGATIVE_GAMMA":
            advisory = ":warning: *AM bias:* Negative gamma — expect larger-than-normal moves"
        else:
            advisory = ":bulb: *AM bias:* Positive gamma — expect mean-reversion intraday"
    else:
        if result.dte_nearest == 0:
            advisory = ":fire: *PM bias:* 0DTE — gamma explosion near strikes; avoid selling into the close"
        elif result.gex_regime == "POSITIVE_GAMMA":
            advisory = ":bulb: *PM bias:* Positive gamma into close — price likely to pin near gamma wall"
        else:
            advisory = ":warning: *PM bias:* Negative gamma near close — late-day directional risk elevated"

    lines.append(f"")
    lines.append(advisory)

    return "\n".join(lines)
