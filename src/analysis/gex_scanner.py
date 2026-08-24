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
from datetime import date, datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import norm

from config.settings import GEX_CHAIN_DIR

logger = logging.getLogger(__name__)


# ── chain cache (one file per calendar date) ──────────────────────────────────

def _cache_path(as_of: date) -> str:
    os.makedirs(GEX_CHAIN_DIR, exist_ok=True)
    return os.path.join(GEX_CHAIN_DIR, f"spy_chain_{as_of}.pkl")


def chain_is_balanced(contracts: list[dict]) -> bool:
    """A valid SPY chain near the money has real OI on BOTH calls and puts.

    A one-sided chain (e.g. yfinance silently dropping calls — the June 2026 bug)
    makes net GEX = call_gex − put_gex collapse to −put_gex, pinning the regime
    to NEGATIVE_GAMMA forever. Reject such chains everywhere."""
    if not contracts:
        return False
    call_n = sum(1 for c in contracts if c.get("is_call"))
    put_n  = sum(1 for c in contracts if not c.get("is_call"))
    call_oi = sum(c.get("oi", 0) for c in contracts if c.get("is_call"))
    put_oi  = sum(c.get("oi", 0) for c in contracts if not c.get("is_call"))
    return call_n >= 5 and put_n >= 5 and call_oi > 0 and put_oi > 0


def save_chain(contracts: list[dict], as_of: date | None = None) -> None:
    """Persist a fetched options chain so subsequent calls skip the API."""
    as_of = as_of or date.today()
    if not chain_is_balanced(contracts):
        logger.error("Refusing to cache one-sided chain for %s "
                     "(calls=%d puts=%d) — would corrupt GEX regime",
                     as_of,
                     sum(1 for c in contracts if c.get("is_call")),
                     sum(1 for c in contracts if not c.get("is_call")))
        return
    with open(_cache_path(as_of), "wb") as f:
        pickle.dump({"date": as_of, "contracts": contracts}, f)
    logger.info("Chain cached: %d contracts for %s", len(contracts), as_of)


def load_chain(as_of: date | None = None) -> list[dict] | None:
    """Return cached contracts for as_of date, or None if not cached / corrupt."""
    as_of = as_of or date.today()
    p = _cache_path(as_of)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "rb") as f:
            data = pickle.load(f)
        contracts = data.get("contracts", [])
        if not chain_is_balanced(contracts):
            logger.warning("Cached chain %s is one-sided (corrupt) — ignoring, will re-fetch", p)
            return None
        logger.info("Chain loaded from cache: %d contracts for %s", len(contracts), as_of)
        return contracts
    except Exception as e:
        logger.warning("Cache read failed (%s): %s — will re-fetch", p, e)
        return None

_RF = 0.045    # risk-free rate
_SHARES = 100  # shares per contract
_ET = ZoneInfo("America/New_York")


def time_to_expiry_years(dte: int) -> float:
    """Calendar-year fraction until expiry.

    For 0DTE use the hours remaining to the 4 PM ET close instead of 0 —
    T=0 zeroes every greek, silently dropping the largest gamma expiry
    on SPY (0DTE OI dominates intraday dealer gamma)."""
    if dte > 0:
        return dte / 365
    now = datetime.now(_ET)
    hours_left = max(16 - now.hour - now.minute / 60, 0.5)
    return hours_left / (24 * 365)


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


def _vega_bs(S: float, K: float, T: float, sig: float) -> float:
    """dValue/dVol per 1.00 (100%) vol -- identical formula/sign for calls and
    puts (an option's value always rises with IV, unlike delta/gamma). Signed
    +call/-put at aggregation time below purely for consistency with how this
    module signs every other Greek (GEX, vanna, charm) -- net_vega_bn is a
    call-vs-put vega IMBALANCE, not total dealer vol-risk magnitude (that
    would be the unsigned sum, which this does not compute)."""
    d1, _ = _d1d2(S, K, T, sig)
    return S * norm.pdf(d1) * math.sqrt(T) if T > 1e-6 and sig > 1e-6 else 0.0


def _theta_bs(S: float, K: float, T: float, sig: float, is_call: bool) -> float:
    """Option value change per CALENDAR DAY passing (negative = decay, the
    usual convention). Unlike gamma/vanna/charm, calls and puts are NOT
    symmetric here (the r*K*e^-rT*N(d2) term flips sign, not just the whole
    expression), so is_call must be passed in rather than signed at
    aggregation time. Verified against a direct one-day price difference
    (not just an epsilon derivative) across several strikes/DTEs."""
    if T <= 1e-6 or sig <= 1e-6:
        return 0.0
    d1, d2 = _d1d2(S, K, T, sig)
    term1 = -S * norm.pdf(d1) * sig / (2 * math.sqrt(T))
    term2 = (-_RF * K * math.exp(-_RF * T) * norm.cdf(d2) if is_call
             else _RF * K * math.exp(-_RF * T) * norm.cdf(-d2))
    return (term1 + term2) / 365


def _charm_bs(S: float, K: float, T: float, sig: float) -> float:
    """∂Delta/∂T per day — identical for calls and puts under q=0
    (Δ_put = Δ_call − 1). The dealer sign convention is applied at
    aggregation time, same as GEX."""
    if T <= 1e-6 or sig <= 1e-6:
        return 0.0
    d1, _ = _d1d2(S, K, T, sig)
    sqrtT = math.sqrt(T)
    # ∂d1/∂T = (r + σ²/2)/(σ√T) − d1/(2T), verified vs finite differences
    dd1_dT = ((_RF + 0.5 * sig**2) / (sig * sqrtT)) - (d1 / (2 * T))
    return norm.pdf(d1) * dd1_dT / 365  # per calendar day


# ── data fetch ────────────────────────────────────────────────────────────────

_YF_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _yf_get(url: str, session=None) -> dict:
    """GET to Yahoo Finance with SSL verification disabled."""
    import requests, urllib3
    urllib3.disable_warnings()
    if session is not None:
        return session.get(url, headers=_YF_HEADERS, timeout=15, verify=False).json()
    return requests.get(url, headers=_YF_HEADERS, timeout=15, verify=False).json()


def _yf_session_with_crumb():
    """
    Return (curl_cffi session, crumb) for Yahoo Finance v7 authenticated requests.
    Uses curl_cffi so verify=False works (requests module has SSL issues on this host).
    Returns (None, None) if crumb cannot be obtained.
    """
    try:
        import curl_cffi.requests as ccr
        s = ccr.Session(verify=False)
        s.get("https://finance.yahoo.com/quote/SPY", verify=False)
        cr = s.get("https://query1.finance.yahoo.com/v1/test/getcrumb", verify=False)
        crumb = cr.text.strip()
        if cr.status_code != 200 or not crumb or crumb.startswith("{"):
            logger.warning("Crumb fetch failed (status %d): %s", cr.status_code, crumb[:60])
            return None, None
        return s, crumb
    except Exception as e:
        logger.warning("Crumb session setup failed: %s", e)
        return None, None


def _spot_and_vix():
    """Return (spot, vix_now, vix_prev_close) via Yahoo Finance v8 chart API."""
    spy_meta = _yf_get(
        "https://query1.finance.yahoo.com/v8/finance/chart/SPY?interval=1d&range=2d"
    )["chart"]["result"][0]["meta"]
    spot = float(spy_meta["regularMarketPrice"])

    vix_data = _yf_get(
        "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX?interval=1d&range=5d"
    )["chart"]["result"][0]
    closes   = vix_data["indicators"]["quote"][0]["close"]
    closes   = [c for c in closes if c is not None]
    vix_now  = float(closes[-1])
    vix_prev = float(closes[-2]) if len(closes) > 1 else vix_now

    return spot, vix_now, vix_prev


def _fetch_chain_from_api(spot: float, n_expiries: int = 3) -> list[dict]:
    """Pull SPY option chain via Yahoo Finance v7 (real OI). Returns [] on auth failure."""
    today = date.today()
    rows: list[dict] = []

    session, crumb = _yf_session_with_crumb()
    if session is None:
        logger.warning("Cannot fetch YF chain — crumb unavailable (rate-limited or market closed)")
        return rows

    base = f"https://query1.finance.yahoo.com/v7/finance/options/SPY?crumb={crumb}"

    try:
        resp0 = session.get(base, headers=_YF_HEADERS, verify=False).json()
        if "optionChain" not in resp0:
            logger.warning("YF v7 response missing optionChain: %s",
                           str(resp0)[:120])
            return rows
        all_exp_ts: list[int] = resp0["optionChain"]["result"][0].get("expirationDates", [])
    except Exception as e:
        logger.error("Failed to fetch SPY expirations: %s", e)
        return rows

    for ts in all_exp_ts[:n_expiries]:
        try:
            # YF expiry timestamps are midnight UTC — converting through local
            # (ET) time shifts them to the previous day, making every DTE off
            # by one and turning today's expiry into "yesterday".
            exp_date = datetime.fromtimestamp(ts, tz=timezone.utc).date()
            T = time_to_expiry_years((exp_date - today).days)
            resp = session.get(f"{base}&date={ts}", headers=_YF_HEADERS,
                               verify=False).json()
            chain_data = resp["optionChain"]["result"][0]["options"][0]

            for contracts, is_call in [
                (chain_data.get("calls", []), True),
                (chain_data.get("puts",  []), False),
            ]:
                for c in contracts:
                    strike = float(c.get("strike", 0))
                    if not (spot * 0.92 <= strike <= spot * 1.08):
                        continue
                    oi = int(c.get("openInterest", 0) or 0)
                    iv = float(c.get("impliedVolatility", 0) or 0)
                    if oi < 10 or iv < 0.01:
                        continue
                    rows.append(dict(strike=strike, oi=oi, iv=iv, T=T, is_call=is_call))
        except Exception as e:
            logger.warning("Chain fetch failed for ts=%s: %s", ts, e)

    return rows


def _fetch_chain(spot: float, n_expiries: int = 3,
                 as_of: date | None = None) -> tuple[list[dict], str]:
    """
    Return SPY option chain contracts, using today's disk cache when available.

    Source priority:
      1. Yahoo Finance v7 (real OI) — authoritative for GEX
      2. Alpaca indicative feed — fallback with bid>0 filter to avoid garbage entries
         (Alpaca never returns real OI for SPY; without the bid>0 filter the GEX
         bloats to -230B from large ask_sz on zero-bid deep-OTM contracts)

    Pass as_of for historical replay.
    """
    as_of = as_of or date.today()
    cached = load_chain(as_of)
    if cached is not None:
        return cached, "HIGH"   # cached chains were validated at save time

    # Primary: Yahoo Finance v7 (real open interest)
    rows = _fetch_chain_from_api(spot, n_expiries)
    if rows:
        logger.info("GEX chain: Yahoo Finance v7 (%d contracts, real OI)", len(rows))
        save_chain(rows, as_of)
        return rows, "HIGH"

    # Fallback: Alpaca indicative — bid/ask size is NOT real OI; regime confidence is LOW
    logger.warning("Yahoo Finance unavailable — falling back to Alpaca indicative (LOW confidence)")
    try:
        from src.live.alpaca_options import fetch_chain_for_gex
        rows = fetch_chain_for_gex("SPY", spot=spot, n_expiries=n_expiries)
        if rows:
            logger.info("GEX chain: Alpaca fallback (%d contracts, LOW confidence)", len(rows))
            save_chain(rows, as_of)
    except Exception as e:
        logger.warning("Alpaca chain also unavailable: %s", e)
    return rows, "LOW"


# ── exposure aggregation ──────────────────────────────────────────────────────

@dataclass
class GEXResult:
    spot:             float
    vix:              float
    vix_prev:         float
    net_gex_bn:       float          # net GEX, $bn per 1% move (positive = long gamma)
    gex_regime:       str            # POSITIVE_GAMMA | NEGATIVE_GAMMA | UNKNOWN
    gamma_wall:       float          # strike nearest spot with highest net GEX
    put_wall:         float          # highest put GEX below spot
    call_wall:        float          # highest call GEX above spot
    flip_level:       float          # where cumulative GEX crosses zero
    net_vanna_bn:     float          # net Vanna, $bn per 1 vol-pt
    vanna_signal:     str            # BULLISH | BEARISH | NEUTRAL
    net_charm:        float          # net charm notional (+ = dealer buying, - = selling)
    charm_signal:     str            # BUYING_PRESSURE | SELLING_PRESSURE | NEUTRAL
    net_vega_bn:      float = 0.0    # call-vs-put vega imbalance, $bn per 1 vol-pt (see _vega_bs)
    net_theta_bn:     float = 0.0    # call-vs-put theta imbalance, $bn decay per calendar day (see _theta_bs)
    top_levels:       list = field(default_factory=list)  # [(strike, gex_bn), ...]
    top_vanna_levels: list = field(default_factory=list)  # [(strike, vanna_bn), ...]
    dte_nearest:      int  = 0
    # HIGH = Yahoo Finance real OI; LOW = Alpaca indicative (bid/ask size proxy, not real OI)
    regime_confidence: str = "HIGH"


def compute_exposures(spot: float, vix: float, vix_prev: float,
                      contracts: list[dict]) -> GEXResult:
    """
    Aggregate gamma/vanna/charm exposures across the options chain.
    """
    gex_by_strike:   dict[float, float] = {}
    vanna_by_strike: dict[float, float] = {}
    charm_total = 0.0
    vega_total  = 0.0
    theta_total = 0.0

    call_gex_by_strike: dict[float, float] = {}
    put_gex_by_strike:  dict[float, float] = {}

    dte_nearest = 999

    for c in contracts:
        K, oi, iv, T, is_call = c["strike"], c["oi"], c["iv"], c["T"], c["is_call"]
        dte = round(T * 365)
        dte_nearest = min(dte_nearest, dte)

        # Use Alpaca's pre-computed gamma when available (more accurate than BS recalc)
        native_gamma = c.get("gamma")
        # $bn of dealer delta-hedge flow per 1% spot move (standard GEX convention)
        gx = (native_gamma if native_gamma and native_gamma > 0
              else _gamma_bs(spot, K, T, iv)) * oi * _SHARES * (spot ** 2) * 0.01 / 1e9
        # $bn of dealer delta-hedge flow per 1 vol-point move
        vn  = _vanna_bs(spot, K, T, iv) * oi * _SHARES * spot * 0.01 / 1e9
        ch  = _charm_bs(spot, K, T, iv) * oi * _SHARES
        vg  = _vega_bs(spot, K, T, iv)  * oi * _SHARES * 0.01 / 1e9
        # theta is NOT call/put symmetric (unlike gamma/vanna/charm), so is_call
        # goes into the formula itself rather than being applied as a sign flip
        th  = _theta_bs(spot, K, T, iv, is_call) * oi * _SHARES / 1e9

        # Dealer convention: dealers typically short calls to retail (positive call OI = dealer short)
        # Standard GEX: calls contribute positive, puts negative
        signed_gex = gx if is_call else -gx
        gex_by_strike[K]   = gex_by_strike.get(K, 0.0)   + signed_gex
        vanna_by_strike[K] = vanna_by_strike.get(K, 0.0) + (vn if is_call else -vn)
        charm_total        += ch if is_call else -ch
        vega_total          += vg if is_call else -vg
        theta_total         += th if is_call else -th

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

    # Guard: a one-sided chain (missing calls or puts) makes net_gex meaningless
    # and would falsely report NEGATIVE_GAMMA. Report UNKNOWN instead.
    total_call_oi = sum(c["oi"] for c in contracts if c["is_call"])
    total_put_oi  = sum(c["oi"] for c in contracts if not c["is_call"])
    if total_call_oi <= 0 or total_put_oi <= 0:
        logger.error("One-sided chain (callOI=%d putOI=%d) — regime set UNKNOWN, not NEGATIVE",
                     total_call_oi, total_put_oi)
        gex_regime = "UNKNOWN"
    else:
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

    # Vanna signal (threshold in $bn per vol-pt — 0.5 in the old per-100-pt units)
    vix_chg_pct = (vix - vix_prev) / vix_prev if vix_prev else 0
    if abs(vix_chg_pct) < 0.02 or abs(net_vanna) < 0.005:
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

    # Top 6 levels by absolute GEX / absolute Vanna
    top_levels       = sorted(gex_by_strike.items(),   key=lambda x: abs(x[1]), reverse=True)[:6]
    top_vanna_levels = sorted(vanna_by_strike.items(), key=lambda x: abs(x[1]), reverse=True)[:6]

    return GEXResult(
        spot=spot, vix=vix, vix_prev=vix_prev,
        net_gex_bn=round(net_gex, 3), gex_regime=gex_regime,
        gamma_wall=gamma_wall, put_wall=put_wall, call_wall=call_wall,
        flip_level=flip_level,
        net_vanna_bn=round(net_vanna, 3), vanna_signal=vanna_signal,
        net_charm=round(charm_total, 0), charm_signal=charm_signal,
        net_vega_bn=round(vega_total, 3), net_theta_bn=round(theta_total, 4),
        top_levels=top_levels, top_vanna_levels=top_vanna_levels,
        dte_nearest=dte_nearest if dte_nearest < 999 else 0,
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
        contracts, confidence = _fetch_chain(spot, as_of=as_of)
        if not contracts:
            logger.warning("No option chain data — chain may be empty or market closed")
            return None
        result = compute_exposures(spot, vix, vix_prev or vix, contracts)
        result.regime_confidence = confidence
        # When OI data is low-confidence (Alpaca proxy), the raw regime is unreliable.
        # Callers should treat regime as neutral / informational only.
        if confidence == "LOW":
            logger.warning("GEX regime is LOW confidence (Alpaca proxy OI, not real OI) — "
                           "regime=%s may not reflect actual dealer positioning", result.gex_regime)
        logger.info("GEX scan: net_gex=%.2fbn regime=%s [%s] gamma_wall=%.1f vanna=%s",
                    result.net_gex_bn, result.gex_regime, confidence,
                    result.gamma_wall, result.vanna_signal)
        return result
    except Exception as e:
        logger.error("GEX scan failed: %s", e)
        return None


# ── Slack message formatter ───────────────────────────────────────────────────

def format_gex_message(result: GEXResult, session: str = "morning",
                       ticker: str = "SPY") -> str:
    """
    Build a Slack message summarising dealer Greek exposures.
    session: 'morning' | 'afternoon'
    ticker: underlying symbol displayed in the header and level labels
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
        f":bar_chart: *{ticker} Greeks — {session_label}* | {datetime.now().strftime('%Y-%m-%d %H:%M')} ET",
        f"",
        f"*{ticker}:* ${result.spot:.2f}  |  *VIX:* {result.vix:.1f} {vix_arrow} ({vix_chg:+.1f})",
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

    max_lvl = max((abs(g) for _, g in result.top_levels[:5]), default=0) or 1
    for strike, gex in result.top_levels[:5]:
        marker = f" <- {ticker}" if abs(strike - result.spot) / result.spot < 0.005 else ""
        bar = (":green_square:" if gex > 0 else ":red_square:") * max(int(abs(gex) / max_lvl * 5), 1)
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
