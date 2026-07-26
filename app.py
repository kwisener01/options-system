"""
GEX Dashboard — Flask web app.

Routes:
  GET  /          — dashboard HTML
  GET  /api/gex   — JSON scan data (cached 5 min; ?force=true to refresh)
  POST /api/slack — post current data to Slack
  GET  /health    — Render health check
"""
import hashlib
import hmac
import json
import logging
import os
import sys
import threading
import time
import warnings
from datetime import datetime, date, timedelta
from datetime import time as dt_time

import pytz
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, os.path.dirname(__file__))

from src.logger import setup_logging

warnings.filterwarnings("ignore")
setup_logging()
logger = logging.getLogger(__name__)

ET = pytz.timezone("America/New_York")
app = Flask(__name__)

# ── cache ──────────────────────────────────────────────────────────────────────
_lock = threading.Lock()
_cache: dict = {"data": None, "ts": 0.0}
CACHE_TTL = 300  # 5 minutes

_scan_lock = threading.Lock()
_scan_cache: dict = {"data": None, "ts": 0.0, "tier": None}
SCAN_CACHE_TTL = 900  # 15 minutes

_wl_lock = threading.Lock()
_wl_cache: dict = {"data": None, "ts": 0.0}
WL_CACHE_TTL = 1800  # 30 minutes

_fa_lock = threading.Lock()
_fa_cache: dict = {"data": None, "ts": 0.0}
FA_CACHE_TTL = 3600  # 1 hour — price data is daily, no need to refresh often

_bp_lock = threading.Lock()
_bp_cache: dict = {"data": None, "ts": 0.0}
BP_CACHE_TTL = 900   # 15 minutes — chain prices move during the day


# ── data fetching ──────────────────────────────────────────────────────────────

def _yget(url: str) -> dict:
    import urllib3
    import requests as _req
    urllib3.disable_warnings()
    return _req.get(url, headers={"User-Agent": "Mozilla/5.0"},
                    timeout=20, verify=False).json()


def _fetch_prices() -> tuple:
    def meta(sym: str) -> dict:
        encoded = sym.replace("^", "%5E")
        return _yget(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}"
            f"?interval=1d&range=5d"
        )["chart"]["result"][0]["meta"]

    spy = meta("SPY")
    qqq = meta("QQQ")
    iwm = meta("IWM")
    vix = meta("^VIX")
    return (
        spy["regularMarketPrice"], spy["chartPreviousClose"],
        qqq["regularMarketPrice"], qqq["chartPreviousClose"],
        iwm["regularMarketPrice"], iwm["chartPreviousClose"],
        vix["regularMarketPrice"], vix["chartPreviousClose"],
    )


def _r_to_dict(r) -> dict | None:
    if r is None:
        return None
    return {
        "gex":          round(r.net_gex_bn, 1),
        "regime":       r.gex_regime,
        "gamma_wall":   r.gamma_wall,
        "flip_level":   r.flip_level,
        "put_wall":     r.put_wall,
        "call_wall":    r.call_wall,
        "vanna_signal": r.vanna_signal,
        "charm_signal": r.charm_signal,
        "dte_nearest":  r.dte_nearest,
        "top_levels":   [[k, round(g, 3)] for k, g in r.top_levels[:8]],
    }


def _scan_ticker(ticker: str, spot: float, prev: float,
                 vix_now: float, vix_prev: float) -> dict:
    from src.live.alpaca_options import fetch_chain_for_gex
    from src.analysis.gex_scanner import compute_exposures

    contracts = fetch_chain_for_gex(ticker, spot=spot, n_expiries=6)

    c_by_dte: dict[int, list] = {}
    for c in contracts:
        dte = round(c["T"] * 365)
        c_by_dte.setdefault(dte, []).append(c)

    c0 = c_by_dte.get(0) or contracts
    c1 = c_by_dte.get(1, [])

    # Weekly: nearest expiry 4–14 DTE (standard weekly option)
    w_dte = next((d for d in sorted(c_by_dte) if 4 <= d <= 14), None)
    c_wk  = c_by_dte.get(w_dte, [])

    # Monthly: nearest expiry 15–60 DTE (standard monthly)
    m_dte = next((d for d in sorted(c_by_dte) if 15 <= d <= 60), None)
    c_mo  = c_by_dte.get(m_dte, [])

    def _gex(cl):
        return compute_exposures(spot, vix_now, vix_prev, cl) if cl else None

    def _rdict(r, dte=None):
        d = _r_to_dict(r)
        if d is not None and dte is not None:
            d["dte"] = dte
        return d

    return {
        "spot":       spot,
        "prev_close": prev,
        "change":     round(spot - prev, 2),
        "change_pct": round((spot - prev) / prev * 100, 2),
        "today":      _rdict(_gex(c0), 0),
        "tomorrow":   _rdict(_gex(c1), 1),
        "weekly":     _rdict(_gex(c_wk), w_dte),
        "monthly":    _rdict(_gex(c_mo), m_dte),
        "full_chain": _r_to_dict(_gex(contracts)),
    }


def _build_live_trade_idea(spy: dict, vix_now: float, vix_prev: float) -> dict:
    """
    Real-priced trade idea: fetches actual Alpaca mid-prices for each leg,
    runs BWB analyzer + credit spread math, recommends the better structure.
    Falls back to heuristic on any error.
    """
    try:
        return _live_trade_impl(spy, vix_now, vix_prev)
    except Exception as e:
        logger.warning("Live trade idea failed (%s) — using heuristic fallback", e)
        old = _build_trade_idea(spy, vix_now, vix_prev)
        if not old:
            return {}
        tr = {
            "type": old.get("type", ""), "side": "put",
            "is_credit": old.get("direction") != "BEARISH",
            "legs": [
                {"action": "BUY",  "strike": old.get("long_leg",  "?"), "opt": "P", "mid": None},
                {"action": "SELL", "strike": old.get("short_leg", "?"), "opt": "P", "mid": None},
            ],
            "width": old.get("width"), "credit": None,
            "max_profit_usd": None, "max_risk_usd": None, "rr_ratio": None,
            "rating": None, "score": None, "lower_breakeven": None, "exit_plan": None,
            "recommended": True,
        }
        return {
            "direction": old.get("direction", "BULLISH"),
            "bearish_signals": 0,
            "expiry": old.get("expiry", "1DTE"), "dte": 1,
            "spot": spy["spot"], "prices_live": False,
            "trades": [tr], "thesis": old.get("thesis", []),
        }


def _live_trade_impl(spy: dict, vix_now: float, vix_prev: float) -> dict:
    from src.live.alpaca_options import fetch_chain_combined
    from src.analysis.bwb_analyzer import BWBInputs, analyze

    r         = spy.get("full_chain") or spy.get("tomorrow") or spy.get("today") or {}
    spot      = spy["spot"]
    flip      = r.get("flip_level") or spot - 10
    put_wall  = r.get("put_wall")   or spot - 15
    call_wall = r.get("call_wall")  or spot + 15
    regime    = r.get("regime", "UNKNOWN")
    vix_chg   = vix_now - vix_prev

    bearish_signals = sum([
        "NEGATIVE" in regime,
        r.get("vanna_signal") == "BEARISH",
        vix_chg > 0.3,
        spot < flip,
    ])
    bearish  = bearish_signals >= 2
    breakout = spot > call_wall  # price has pushed through call wall

    # Fetch 0–3 DTE SPY puts + calls with live bid/ask
    chain = fetch_chain_combined("SPY", spot, dte_min=0, dte_max=3)
    puts  = chain.get("puts_liquid",  [])
    calls = chain.get("calls_liquid", [])

    def pick_expiry(rows):
        by_exp: dict[str, list] = {}
        for p in rows:
            by_exp.setdefault(p["expiry"], []).append(p)
        for exp in sorted(by_exp.keys()):
            if by_exp[exp][0]["dte"] >= 1:
                return exp, by_exp[exp][0]["dte"], by_exp[exp]
        return None, 1, []

    put_exp,  put_dte,  put_rows  = pick_expiry(puts)
    call_exp, call_dte, call_rows = pick_expiry(calls)

    # Use put-side expiry as canonical; fall back to call-side
    target_exp = put_exp or call_exp or "1DTE"
    target_dte = put_dte if put_exp else call_dte

    put_mids  = {p["strike"]: round((p["bid"] + p["ask"]) / 2, 2) for p in put_rows}
    call_mids = {c["strike"]: round((c["bid"] + c["ask"]) / 2, 2) for c in call_rows}

    def nearest_put(target: float) -> tuple[float, float]:
        if not put_mids:
            return round(target), 0.0
        s = min(put_mids, key=lambda x: abs(x - target))
        return s, put_mids[s]

    def nearest_call(target: float) -> tuple[float, float]:
        if not call_mids:
            return round(target), 0.0
        s = min(call_mids, key=lambda x: abs(x - target))
        return s, call_mids[s]

    trades: list[dict] = []

    # ── PUT-SIDE STRUCTURES ───────────────────────────────────────────────────

    if not bearish:
        short_s, short_mid = nearest_put(round(put_wall))

        # Bull Put Spread: sell put_wall, buy $5 lower
        cs_l_s, cs_l_mid = nearest_put(short_s - 5)
        cs_credit = round(short_mid - cs_l_mid, 2)
        cs_width  = short_s - cs_l_s
        if cs_credit > 0 and cs_width > 0:
            trades.append({
                "type": "Bull Put Spread", "side": "put", "is_credit": True,
                "legs": [
                    {"action": "SELL", "strike": short_s, "opt": "P", "mid": short_mid},
                    {"action": "BUY",  "strike": cs_l_s,  "opt": "P", "mid": cs_l_mid},
                ],
                "width":          cs_width,
                "credit":         cs_credit,
                "max_profit_usd": round(cs_credit * 100),
                "max_risk_usd":   round((cs_width - cs_credit) * 100),
                "rr_ratio":       round(cs_credit / max(cs_width - cs_credit, 0.01), 2),
                "rating": None, "score": None, "lower_breakeven": None, "exit_plan": None,
            })

        # Bull BWB: buy H, sell 2×M, buy L
        h_s,     h_mid     = nearest_put(short_s + 5)
        bwb_l_s, bwb_l_mid = nearest_put(short_s - 10)
        bwb_credit = round(short_mid * 2 - h_mid - bwb_l_mid, 2)
        bwb = analyze(BWBInputs(
            ticker="SPY", spot=spot, dte=target_dte,
            long_upper=h_s, short_strike=short_s, long_lower=bwb_l_s,
            credit=bwb_credit, regime=regime, vix_now=vix_now, vix_prev=vix_prev,
            flip_level=flip, put_wall=put_wall, call_wall=call_wall, major_news=False,
        ))
        if bwb_credit > 0:
            trades.append({
                "type": "Bull BWB", "side": "put", "is_credit": True,
                "legs": [
                    {"action": "BUY",  "strike": h_s,     "opt": "P", "mid": h_mid},
                    {"action": "SELL", "strike": short_s,  "opt": "P", "mid": short_mid, "qty": 2},
                    {"action": "BUY",  "strike": bwb_l_s,  "opt": "P", "mid": bwb_l_mid},
                ],
                "width":            short_s - bwb_l_s,
                "credit":           bwb_credit,
                "max_profit_usd":   round(bwb.max_profit_usd),
                "max_risk_usd":     round(bwb.max_loss_usd),
                "rr_ratio":         round(bwb.rr_ratio, 2),
                "rating":           bwb.rating,
                "score":            bwb.setup_score,
                "lower_breakeven":  round(bwb.lower_breakeven, 2),
                "exit_plan":        bwb.exit_plan,
            })

    else:
        # Bear Put Spread: buy near flip, sell lower
        long_s,  long_mid  = nearest_put(round(flip))
        bps_s,   bps_mid   = nearest_put(max(round(put_wall), round(flip) - 10))
        if long_s - bps_s < 3:
            bps_s, bps_mid = nearest_put(long_s - 5)
        bps_width = long_s - bps_s
        bps_debit = round(long_mid - bps_mid, 2)
        if bps_debit > 0:
            trades.append({
                "type": "Bear Put Spread", "side": "put", "is_credit": False,
                "legs": [
                    {"action": "BUY",  "strike": long_s, "opt": "P", "mid": long_mid},
                    {"action": "SELL", "strike": bps_s,  "opt": "P", "mid": bps_mid},
                ],
                "width":          bps_width,
                "credit":         -bps_debit,
                "max_profit_usd": round((bps_width - bps_debit) * 100),
                "max_risk_usd":   round(bps_debit * 100),
                "rr_ratio":       round((bps_width - bps_debit) / max(bps_debit, 0.01), 2),
                "rating": None, "score": None, "lower_breakeven": None, "exit_plan": None,
            })

    # ── DOUBLE BWB (Batman) — dedicated 7-21 DTE fetch; both sides must be credit
    #
    # Uses a SEPARATE chain fetch (dte_min=7, dte_max=21) so there is enough
    # theta for both BWB sides to collect real premium.  Both sides share the
    # same expiry.  Every leg mid is verified non-zero from Alpaca bid/ask
    # before credits are computed.

    if not bearish and not breakout:
        try:
            from src.analysis.bwb_analyzer import DoubleBWBInputs, analyze_double_bwb

            dbwb_raw = fetch_chain_combined("SPY", spot, dte_min=7, dte_max=21)
            d_puts   = dbwb_raw.get("puts_liquid",  [])
            d_calls  = dbwb_raw.get("calls_liquid", [])

            # Find the nearest common expiry where both put and call sides have data
            p_exps = sorted({p["expiry"] for p in d_puts})
            c_exps = sorted({c["expiry"] for c in d_calls})
            common = [e for e in p_exps if e in set(c_exps)]
            if not common:
                raise ValueError("No common put/call expiry found in 7-21 DTE chain")

            dbwb_exp = common[0]
            dbwb_dte = next(p["dte"] for p in d_puts if p["expiry"] == dbwb_exp)

            # Build price maps — only include strikes with real bid AND ask
            dp_mids = {
                p["strike"]: round((p["bid"] + p["ask"]) / 2, 2)
                for p in d_puts
                if p["expiry"] == dbwb_exp and p["bid"] > 0 and p["ask"] > 0
            }
            dc_mids = {
                c["strike"]: round((c["bid"] + c["ask"]) / 2, 2)
                for c in d_calls
                if c["expiry"] == dbwb_exp and c["bid"] > 0 and c["ask"] > 0
            }

            if not dp_mids or not dc_mids:
                raise ValueError("Insufficient liquid strikes for Double BWB")

            def np_(target):
                s = min(dp_mids, key=lambda x: abs(x - target))
                return s, dp_mids[s]

            def nc_(target):
                s = min(dc_mids, key=lambda x: abs(x - target))
                return s, dc_mids[s]

            # GEX-anchored: short strikes at GEX walls
            mp_s, mp_mid = np_(round(put_wall))
            hp_s, hp_mid = np_(mp_s + 5)
            lp_s, lp_mid = np_(mp_s - 10)
            mc_s, mc_mid = nc_(round(call_wall))
            lc_s, lc_mid = nc_(mc_s - 5)
            hc_s, hc_mid = nc_(mc_s + 10)

            # All six leg mids must be non-zero (real Alpaca prices)
            leg_mids = [hp_mid, mp_mid, lp_mid, lc_mid, mc_mid, hc_mid]
            if any(m == 0 for m in leg_mids):
                raise ValueError("One or more leg mids are zero — strike not in chain")

            # Compute credits from verified live mids
            dbwb_put_credit  = round(mp_mid * 2 - hp_mid - lp_mid, 2)
            dbwb_call_credit = round(mc_mid * 2 - lc_mid - hc_mid, 2)
            dbwb_total       = round(dbwb_put_credit + dbwb_call_credit, 2)
            dbwb_mid_width   = lc_s - hp_s

            # Hard gate: both sides must independently be credits AND total must be credit
            if dbwb_put_credit <= 0:
                raise ValueError(f"Put BWB is a debit ({dbwb_put_credit:.2f}) — not viable")
            if dbwb_call_credit <= 0:
                raise ValueError(f"Call BWB is a debit ({dbwb_call_credit:.2f}) — not viable")
            if dbwb_total <= 0.10:
                raise ValueError(f"Total credit too thin ({dbwb_total:.2f})")
            if dbwb_mid_width < 8:
                raise ValueError(f"Middle zone only {dbwb_mid_width:.0f}pt — too narrow")

            d = analyze_double_bwb(DoubleBWBInputs(
                ticker="SPY", spot=spot, dte=dbwb_dte,
                put_upper=hp_s,  put_short=mp_s,  put_lower=lp_s,
                put_credit=dbwb_put_credit,
                call_lower=lc_s, call_short=mc_s, call_upper=hc_s,
                call_credit=dbwb_call_credit,
                put_upper_mid=hp_mid, put_short_mid=mp_mid, put_lower_mid=lp_mid,
                call_lower_mid=lc_mid, call_short_mid=mc_mid, call_upper_mid=hc_mid,
                regime=regime, vix_now=vix_now, vix_prev=vix_prev,
                flip_level=flip, put_wall=put_wall, call_wall=call_wall,
                major_news=False,
            ))
            trades.append({
                "type": "Double BWB", "side": "both", "is_credit": True,
                "legs": [
                    {"action": "BUY",  "strike": hp_s, "opt": "P", "mid": hp_mid},
                    {"action": "SELL", "strike": mp_s, "opt": "P", "mid": mp_mid, "qty": 2},
                    {"action": "BUY",  "strike": lp_s, "opt": "P", "mid": lp_mid},
                    {"action": "BUY",  "strike": lc_s, "opt": "C", "mid": lc_mid},
                    {"action": "SELL", "strike": mc_s, "opt": "C", "mid": mc_mid, "qty": 2},
                    {"action": "BUY",  "strike": hc_s, "opt": "C", "mid": hc_mid},
                ],
                "expiry_dbwb":        dbwb_exp,
                "dte_dbwb":           dbwb_dte,
                "width":              dbwb_mid_width,
                "credit":             dbwb_total,
                "put_credit":         dbwb_put_credit,
                "call_credit":        dbwb_call_credit,
                "max_profit_usd":     round(max(d.put_peak_profit_usd, d.call_peak_profit_usd)),
                "max_risk_usd":       round(max(abs(d.put_flat_loss_usd), abs(d.call_flat_loss_usd))),
                "rr_ratio":           round(max(d.put_peak_profit_usd, d.call_peak_profit_usd)
                                            / max(max(abs(d.put_flat_loss_usd), abs(d.call_flat_loss_usd)), 1), 2),
                "rating":             d.rating,
                "score":              d.setup_score,
                "lower_breakeven":    d.put_lower_breakeven,
                "exit_plan":          d.exit_plan,
                "middle_width":       d.middle_width,
                "middle_profit_usd":  d.middle_profit_usd,
                "put_peak_profit_usd":  d.put_peak_profit_usd,
                "call_peak_profit_usd": d.call_peak_profit_usd,
                "put_lower_breakeven":  d.put_lower_breakeven,
                "call_upper_breakeven": d.call_upper_breakeven,
                "put_flat_loss_usd":    d.put_flat_loss_usd,
                "call_flat_loss_usd":   d.call_flat_loss_usd,
                "put_strikes":  [lp_s, mp_s, hp_s],
                "call_strikes": [lc_s, mc_s, hc_s],
            })
            logger.info(
                "Double BWB SPY %s/%s/%s P | %s/%s/%s C  "
                "put_cr=%.2f  call_cr=%.2f  total=%.2f  mid=%dpt  DTE=%d  rating=%s",
                hp_s, mp_s, lp_s, lc_s, mc_s, hc_s,
                dbwb_put_credit, dbwb_call_credit, dbwb_total,
                dbwb_mid_width, dbwb_dte, d.rating,
            )
        except Exception as e:
            logger.info("Double BWB skipped: %s", e)

    # ── CALL-SIDE STRUCTURES ──────────────────────────────────────────────────

    if breakout and call_mids:
        # Bull Call Spread: buy just above call wall, sell further OTM (breakout play)
        lc_s, lc_mid = nearest_call(round(call_wall) + 2)
        sc_s, sc_mid = nearest_call(round(call_wall) + 7)
        bcs_debit = round(lc_mid - sc_mid, 2)
        bcs_width = sc_s - lc_s
        if bcs_debit > 0 and bcs_width > 0:
            trades.append({
                "type": "Bull Call Spread", "side": "call", "is_credit": False,
                "legs": [
                    {"action": "BUY",  "strike": lc_s, "opt": "C", "mid": lc_mid},
                    {"action": "SELL", "strike": sc_s, "opt": "C", "mid": sc_mid},
                ],
                "width":          bcs_width,
                "credit":         -bcs_debit,
                "max_profit_usd": round((bcs_width - bcs_debit) * 100),
                "max_risk_usd":   round(bcs_debit * 100),
                "rr_ratio":       round((bcs_width - bcs_debit) / max(bcs_debit, 0.01), 2),
                "rating": None, "score": None, "lower_breakeven": None, "exit_plan": None,
            })

    if not breakout and call_mids:
        # Bear Call Spread: sell near call wall, buy higher (income above resistance)
        sell_s, sell_mid = nearest_call(round(call_wall))
        buy_s,  buy_mid  = nearest_call(round(call_wall) + 5)
        becs_credit = round(sell_mid - buy_mid, 2)
        becs_width  = buy_s - sell_s
        if becs_credit > 0 and becs_width > 0:
            trades.append({
                "type": "Bear Call Spread", "side": "call", "is_credit": True,
                "legs": [
                    {"action": "SELL", "strike": sell_s, "opt": "C", "mid": sell_mid},
                    {"action": "BUY",  "strike": buy_s,  "opt": "C", "mid": buy_mid},
                ],
                "width":          becs_width,
                "credit":         becs_credit,
                "max_profit_usd": round(becs_credit * 100),
                "max_risk_usd":   round((becs_width - becs_credit) * 100),
                "rr_ratio":       round(becs_credit / max(becs_width - becs_credit, 0.01), 2),
                "rating": None, "score": None, "lower_breakeven": None, "exit_plan": None,
            })

        if bearish:
            # Call BWB: buy lower call, sell 2× at call wall, buy higher (bearish income)
            lc_s, lc_mid = nearest_call(round(call_wall) - 5)
            mc_s, mc_mid = nearest_call(round(call_wall))
            hc_s, hc_mid = nearest_call(round(call_wall) + 5)
            cbwb_credit = round(mc_mid * 2 - lc_mid - hc_mid, 2)
            uw = mc_s - lc_s
            lw = hc_s - mc_s
            if cbwb_credit > 0:
                trades.append({
                    "type": "Call BWB", "side": "call", "is_credit": True,
                    "legs": [
                        {"action": "BUY",  "strike": lc_s, "opt": "C", "mid": lc_mid},
                        {"action": "SELL", "strike": mc_s, "opt": "C", "mid": mc_mid, "qty": 2},
                        {"action": "BUY",  "strike": hc_s, "opt": "C", "mid": hc_mid},
                    ],
                    "width":          lw + uw,
                    "credit":         cbwb_credit,
                    "max_profit_usd": round(cbwb_credit * 100),
                    "max_risk_usd":   round(max(lw - cbwb_credit, 0) * 100),
                    "rr_ratio":       None,
                    "rating": None, "score": None, "lower_breakeven": None, "exit_plan": None,
                })

    # ── Recommend the best structure ──────────────────────────────────────────
    def _priority(tr: dict) -> int:
        t = tr["type"]
        if t == "Double BWB"    and tr["rating"] in ("A+", "Acceptable") and tr["credit"] >= 0.20: return 0
        if t == "Bull BWB"      and tr["rating"] in ("A+", "Acceptable") and tr["credit"] >= 0.15: return 1
        if t == "Bull Put Spread"  and tr["credit"] >= 0.10: return 2
        if t == "Bear Call Spread" and tr["credit"] >= 0.10: return 3
        if t == "Bear Put Spread":                            return 4
        if t in ("Bull Call Spread", "Call BWB"):             return 5
        return 9

    if trades:
        best = min(trades, key=_priority)
        for tr in trades:
            tr["recommended"] = (tr is best)

    direction = "BREAKOUT" if breakout else ("BEARISH" if bearish else "BULLISH")
    thesis = [
        f"Put wall ${put_wall:.0f}  |  Call wall ${call_wall:.0f}  |  Flip ${flip:.0f}",
        f"Regime: {regime.replace('_', ' ').title()}  |  VIX {vix_chg:+.2f}",
        f"Vanna: {r.get('vanna_signal','—')}  |  Charm: {r.get('charm_signal','—')}",
    ]

    return {
        "direction":       direction,
        "bearish_signals": bearish_signals,
        "expiry":          target_exp,
        "dte":             target_dte,
        "spot":            spot,
        "prices_live":     bool(put_mids or call_mids),
        "trades":          trades,
        "thesis":          thesis,
    }


def _build_trade_idea(spy: dict, vix_now: float, vix_prev: float) -> dict:
    r = spy.get("full_chain") or spy.get("tomorrow") or spy.get("today")
    if not r:
        return {}
    spot      = spy["spot"]
    flip      = r["flip_level"]
    put_wall  = r["put_wall"]
    vix_chg   = vix_now - vix_prev

    bearish_signals = sum([
        r["regime"] == "NEGATIVE_GAMMA",
        r["vanna_signal"] == "BEARISH",
        vix_chg > 0.3,
        spot < flip,
    ])

    if bearish_signals >= 2:
        lp = round(flip)
        sp = max(round(put_wall), lp - 10)
        w  = lp - sp
        if w < 3:
            sp = lp - 5
            w  = 5
        return {
            "type":         "Bear Put Spread",
            "direction":    "BEARISH",
            "long_leg":     f"Buy ${lp}P",
            "short_leg":    f"Sell ${sp}P",
            "width":        w,
            "expiry":       "1DTE",
            "stop_level":   round(flip + 2),
            "target_level": sp,
            "thesis": [
                f"Negative gamma ({r['gex']:.0f}B) — breaks amplify lower",
                f"Vanna: {r['vanna_signal']}  |  VIX {vix_chg:+.2f}",
                f"SPY near flip ${flip:.0f} — thin support below",
            ],
        }
    else:
        sp = round(put_wall)
        lp = sp - 5
        return {
            "type":         "Bull Put Spread",
            "direction":    "BULLISH",
            "long_leg":     f"Buy ${lp}P",
            "short_leg":    f"Sell ${sp}P",
            "width":        5,
            "expiry":       "1DTE",
            "stop_level":   sp,
            "target_level": None,
            "thesis": [
                f"Put wall ${put_wall:.0f} = GEX-defined support floor",
                f"Vanna: {r['vanna_signal']}  |  VIX {vix_chg:+.2f}",
                "Collect premium above GEX support",
            ],
        }


# ── SPY trade idea: shared formatting + change-detection ─────────────────────────
#
# These power both the periodic scans and the 5-minute SPY trade monitor.  The
# monitor alerts ONLY when the recommended high-probability trade changes to a
# different structure/strikes, so the channel stays quiet on unchanged days.

def _spy_recommended(t: dict):
    """Return the single recommended trade dict from a trade_idea, or None."""
    trades = (t or {}).get("trades") or []
    return next((tr for tr in trades if tr.get("recommended")), None)


def _spy_trade_signature(t: dict):
    """Stable fingerprint of the recommended trade. Two ideas with the same
    structure + strikes + direction + expiry share a signature, so re-running
    the scan on unchanged conditions produces no new alert."""
    rec = _spy_recommended(t)
    if not rec:
        return None
    strikes = "/".join(str(lg.get("strike")) for lg in rec.get("legs", []))
    return f"{t.get('direction','')}|{rec.get('type','')}|{strikes}|{t.get('expiry','')}"


def _spy_trade_quality(t: dict):
    """Grade the recommended SPY trade. Returns (tier, reasons).

    tier ∈ {HIGH, MODERATE, LOW}. HIGH = high-probability setup worth an alert:
    premium-selling structure, supportive GEX/VIX read, healthy R/R, and a
    short strike comfortably OTM (probability-of-profit proxy)."""
    rec = _spy_recommended(t)
    if not rec:
        return "LOW", ["no recommended structure"]

    reasons: list[str] = []
    score = 0

    if rec.get("is_credit"):
        score += 1
        reasons.append("credit / positive theta")

    rating = rec.get("rating")
    if rating in ("A+", "Acceptable"):
        score += 1
        reasons.append(f"BWB rating {rating}")

    rr = rec.get("rr_ratio") or 0
    if rr and rr >= 0.25:
        score += 1
        reasons.append(f"R/R {rr:.2f}")

    spot = t.get("spot") or 0
    short_leg = next((lg for lg in rec.get("legs", []) if lg.get("action") == "SELL"), None)
    if short_leg and spot:
        k = short_leg.get("strike") or 0
        otm = ((spot - k) if short_leg.get("opt") == "P" else (k - spot)) / spot * 100
        if otm >= 0.5:
            score += 1
            reasons.append(f"short {otm:.1f}% OTM")

    tier = "HIGH" if score >= 3 else "MODERATE" if score >= 2 else "LOW"
    return tier, reasons


def _fmt_spy_trade_block(t: dict) -> str:
    """Multi-trade block used inside the GEX dashboard post and the unified scan."""
    if not t:
        return ""
    direction = t.get("direction", "")
    color     = ":red_circle:" if direction == "BEARISH" else ":green_circle:"
    live_tag  = " _(live prices)_" if t.get("prices_live") else ""
    trades    = t.get("trades") or []
    rec_trade = _spy_recommended(t)
    rec_label = rec_trade["type"] if rec_trade else "—"

    trade_lines = []
    for tr in trades:
        star   = "★ " if tr.get("recommended") else "  "
        legs   = "/".join(f"${lg['strike']}" for lg in tr.get("legs", []))
        cr     = tr.get("credit", 0) or 0
        profit = tr.get("max_profit_usd") or 0
        risk   = tr.get("max_risk_usd") or 0
        rating = f"  [{tr['rating']}]" if tr.get("rating") else ""
        trade_lines.append(
            f">  {star}{tr['type']}  {legs}  "
            f"{'Cr' if cr >= 0 else 'Dr'} ${abs(cr):.2f}  "
            f"Profit ${profit}  Risk ${risk}{rating}"
        )

    return (
        f"{color} *SPY Trade Ideas — {direction}  |  Recommended: {rec_label}"
        f"  |  Expiry {t.get('expiry','')} ({t.get('dte','')}DTE){live_tag}*\n"
        + "\n".join(trade_lines)
        + ("\n" if trade_lines else "")
        + "\n".join(f">  - {pt}" for pt in t.get("thesis", []))
    )


def _fmt_spy_signal(t: dict, tier: str, reasons: list, changed: bool) -> str:
    """Focused 'trade this now' alert for the 5-minute monitor — one structure,
    full leg detail with live mids, sizing, and confidence."""
    rec       = _spy_recommended(t)
    direction = t.get("direction", "")
    color     = ":red_circle:" if direction == "BEARISH" else ":green_circle:"
    header    = ":rotating_light: *SPY Trade Changed*" if changed else ":dart: *SPY Trade Signal*"

    leg_lines = []
    for lg in rec.get("legs", []):
        q     = lg.get("qty", 1) or 1
        qtxt  = f"{q}x " if q != 1 else ""
        mid   = lg.get("mid")
        midtx = f" @ ${mid:.2f}" if mid else ""
        leg_lines.append(f"  {lg['action']:<4} {qtxt}${lg['strike']}{lg['opt']}{midtx}")

    cr     = rec.get("credit", 0) or 0
    crtxt  = f"{'Credit' if cr >= 0 else 'Debit'} ${abs(cr):.2f}"
    profit = rec.get("max_profit_usd") or 0
    risk   = rec.get("max_risk_usd") or 0
    rr     = rec.get("rr_ratio")
    rrtxt  = f"  |  R/R {rr:.2f}" if rr else ""
    rate   = f"  |  Rating {rec['rating']}" if rec.get("rating") else ""

    lines = [
        header,
        f"{color} *{rec['type']} — {direction}*  ({t.get('expiry','')}, {t.get('dte','')}DTE)  "
        f"SPY ${t.get('spot', 0):.2f}",
        *leg_lines,
        f"{crtxt}  |  Max profit ${profit}  |  Max risk ${risk}{rrtxt}{rate}",
        f":dart: Confidence *{tier}* — " + ", ".join(reasons),
    ]
    return "\n".join(lines)


# Persisted across Render restarts so a redeploy doesn't re-fire the same trade.
SPY_STATE_PATH = os.path.join(os.path.dirname(__file__), "data", "spy_trade_state.json")
_spy_state_lock = threading.Lock()


def _load_spy_state() -> dict:
    try:
        with open(SPY_STATE_PATH) as f:
            return json.load(f)
    except Exception:
        return {"sig": None, "tier": None, "ts": 0}


def _save_spy_state(state: dict) -> None:
    try:
        os.makedirs(os.path.dirname(SPY_STATE_PATH), exist_ok=True)
        with open(SPY_STATE_PATH, "w") as f:
            json.dump(state, f)
    except Exception as e:
        logger.warning("Could not persist SPY trade state: %s", e)


def _market_is_open() -> bool:
    """Alpaca clock is authoritative (handles holidays/half-days); fall back to
    an ET time window only if the API is unreachable."""
    try:
        from src.live.alpaca_options import _trading
        return bool(_trading().get_clock().is_open)
    except Exception as e:
        logger.warning("Clock check failed (%s) — using ET time window", e)
        now = datetime.now(ET)
        if now.weekday() >= 5:
            return False
        return dt_time(9, 30) <= now.time() <= dt_time(16, 0)


def _build_data() -> dict:
    spy_spot, spy_prev, qqq_spot, qqq_prev, iwm_spot, iwm_prev, vix_now, vix_prev = _fetch_prices()
    spy   = _scan_ticker("SPY", spy_spot, spy_prev, vix_now, vix_prev)
    qqq   = _scan_ticker("QQQ", qqq_spot, qqq_prev, vix_now, vix_prev)
    iwm   = _scan_ticker("IWM", iwm_spot, iwm_prev, vix_now, vix_prev)
    trade = _build_live_trade_idea(spy, vix_now, vix_prev)
    return {
        "timestamp":  datetime.now(ET).strftime("%Y-%m-%d %H:%M ET"),
        "spy":        spy,
        "qqq":        qqq,
        "iwm":        iwm,
        "vix":        {
            "now":    vix_now,
            "prev":   vix_prev,
            "change": round(vix_now - vix_prev, 2),
        },
        "trade_idea": trade,
    }


def get_data(force: bool = False) -> dict:
    with _lock:
        age = time.time() - _cache["ts"]
        if force or _cache["data"] is None or age > CACHE_TTL:
            logger.info("Running GEX scan (force=%s age=%.0fs)", force, age)
            _cache["data"] = _build_data()
            _cache["ts"]   = time.time()
        return _cache["data"]


# ── routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/gex")
def api_gex():
    force = request.args.get("force", "false").lower() == "true"
    try:
        return jsonify({"ok": True, "data": get_data(force=force)})
    except Exception as e:
        logger.exception("GEX scan failed")
        return jsonify({"ok": False, "error": str(e)}), 500


_news_lock = threading.Lock()
_news_cache: dict = {"data": None, "ts": 0.0}
NEWS_CACHE_TTL = 300  # 5 minutes


@app.route("/api/news")
def api_news():
    """Recent scored headlines for held symbols + SPY, for the dashboard."""
    with _news_lock:
        if _news_cache["data"] is not None and time.time() - _news_cache["ts"] < NEWS_CACHE_TTL:
            return jsonify({"ok": True, "data": _news_cache["data"], "cached": True})
    try:
        from src.live.news import news_for
        symbols = ["SPY"]
        try:
            from src.live.alpaca_options import _trading
            for p in _trading().get_all_positions():
                base = p.symbol if len(p.symbol) <= 6 else _occ_parse(p.symbol)[0] if _occ_parse(p.symbol) else None
                if base and base not in symbols:
                    symbols.append(base)
        except Exception as e:
            logger.warning("api_news positions fetch failed: %s", e)
        data = news_for(symbols, limit_each=5)
        with _news_lock:
            _news_cache["data"], _news_cache["ts"] = data, time.time()
        return jsonify({"ok": True, "data": data, "cached": False})
    except Exception as e:
        logger.exception("api_news failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/expected-move")
def api_expected_move():
    """Straddle-based expected move for a chosen DTE (0 = today / 0DTE)."""
    try:
        dte = int(request.args.get("dte", 0))
        sym = request.args.get("symbol", "SPY").upper()
        from src.analysis.expected_move import compute
        em = compute(sym, dte)
        if not em:
            return jsonify({"ok": False, "error": "no chain data (market closed or no quotes)"}), 200
        return jsonify({"ok": True, **em})
    except Exception as e:
        logger.exception("expected-move failed")
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


@app.route("/api/close-prompt/<ticker>", methods=["GET", "POST"])
def api_close_prompt(ticker):
    """Post a Close/Hold button for a ticker's option structure to Slack so it can
    be closed with a tap (no slash command needed). Exposes NO account data; the
    close itself still requires the signature-verified Slack button tap."""
    try:
        ticker = ticker.upper()
        from src.live.alpaca_options import _trading
        legs = [p for p in _trading().get_all_positions()
                if (getattr(p, "asset_class", "") in ("us_option", "option") or len(p.symbol) > 8)
                and p.symbol.upper().startswith(ticker)]
        if not legs:
            return jsonify({"ok": False, "error": "no open option legs"}), 404
        label = f"{ticker} spread ({len(legs)} legs)"
        text  = f"*Close {label}?*  _tap to submit a closing order_"
        # restart-proof: button value carries the ticker, not a stored record id
        from src.notifications.slack_blocks import exit_blocks
        from src.notifications.slack_notifier import send_blocks
        blocks, fb = exit_blocks(":outbox_tray: *Close position*",
                                 [{"tid": f"close_ticker:{ticker}", "text": text, "label": label}])
        send_blocks(blocks, fb)
        return jsonify({"ok": True, "posted": label})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


@app.route("/api/mode")
def api_mode():
    """LIVE vs PAPER, derived from ALPACA_BASE_URL only. No account data exposed."""
    from config.settings import IS_PAPER, ALPACA_BASE_URL
    host = ALPACA_BASE_URL.split("//")[-1]
    return jsonify({"ok": True, "mode": "PAPER" if IS_PAPER else "LIVE",
                    "paper": IS_PAPER, "host": host})


@app.route("/api/slack", methods=["POST"])
def api_slack():
    try:
        d  = get_data()
        from src.notifications.slack_notifier import send_message
        ok = send_message(_fmt_slack(d))
        return jsonify({"ok": ok})
    except Exception as e:
        logger.exception("Slack post failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/bwb", methods=["POST"])
def api_bwb():
    try:
        body = request.get_json(force=True)
        from src.analysis.bwb_analyzer import BWBInputs, analyze

        # Pull live GEX context so the user doesn't have to type it in
        d  = get_data()
        fc = d["spy"].get("full_chain") or {}

        # Use explicitly supplied spot (for non-SPY tickers) or fall back to SPY
        ticker = body.get("ticker", "SPY")
        if "spot" in body and body["spot"]:
            spot_price = float(body["spot"])
        elif ticker.upper() in ("SPY", "XSP", "QQQ"):
            spot_price = d[ticker.lower()]["spot"] if ticker.lower() in d else d["spy"]["spot"]
        else:
            spot_price = d["spy"]["spot"]  # use SPY as market proxy for individual stocks

        # Use per-stock GEX context if the scanner computed it; fall back to SPY
        stock_gex = body.get("gex_context") or {}
        gex_src   = stock_gex if stock_gex else fc  # fc = SPY full-chain GEX

        inp = BWBInputs(
            ticker       = ticker,
            spot         = spot_price,
            dte          = int(body["dte"]),
            long_upper   = float(body["long_upper"]),
            short_strike = float(body["short_strike"]),
            long_lower   = float(body["long_lower"]),
            credit       = float(body["credit"]),
            regime       = gex_src.get("regime", "UNKNOWN"),
            vix_now      = d["vix"]["now"],
            vix_prev     = d["vix"]["prev"],
            flip_level   = gex_src.get("flip_level", 0.0),
            put_wall     = gex_src.get("put_wall", 0.0),
            call_wall    = gex_src.get("call_wall", 0.0),
            major_news   = bool(body.get("major_news", False)),
        )
        r = analyze(inp)

        return jsonify({"ok": True, "result": {
            "upper_wing":     r.upper_wing,
            "lower_wing":     r.lower_wing,
            "extra_risk":     r.extra_risk,
            "max_profit_usd": r.max_profit_usd,
            "max_loss_usd":   r.max_loss_usd,
            "lower_breakeven":r.lower_breakeven,
            "rr_ratio":       round(r.rr_ratio, 2),
            "checks":         r.checks,
            "setup_score":    r.setup_score,
            "cs_max_risk_usd":r.cs_max_risk_usd,
            "cs_credit_est":  r.cs_credit_est,
            "bwb_vs_cs":      r.bwb_vs_cs,
            "rating":         r.rating,
            "main_risk":      r.main_risk,
            "exit_plan":      r.exit_plan,
            "overnight_ok":   r.overnight_ok,
            "overnight_reason":r.overnight_reason,
            "summary":        r.summary,
            # echo back context used
            "context": {
                "spot":       spot_price,
                "regime":     gex_src.get("regime"),
                "flip_level": gex_src.get("flip_level"),
                "put_wall":   gex_src.get("put_wall"),
                "vanna":      gex_src.get("vanna_signal"),
                "charm":      gex_src.get("charm_signal"),
                "vix":        d["vix"]["now"],
                "gex_source": "stock" if stock_gex else "SPY",
            },
        }})
    except KeyError as e:
        return jsonify({"ok": False, "error": f"Missing field: {e}"}), 400
    except Exception as e:
        logger.exception("BWB analysis failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/watchlist", methods=["GET"])
def api_watchlist():
    force = request.args.get("force", "false").lower() == "true"
    try:
        signals = _get_watchlist(force=force)
        return jsonify({"ok": True, "signals": signals,
                        "timestamp": datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")})
    except Exception as e:
        logger.exception("Watchlist scan failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/watchlist/alert", methods=["POST"])
def api_watchlist_alert():
    """Post STRONG + WATCH signals to Slack."""
    try:
        d   = get_data()
        fc  = d["spy"].get("full_chain") or {}
        with _wl_lock:
            signals = _wl_cache["data"]
        if not signals:
            from src.analysis.value_watchlist import scan_watchlist
            signals = scan_watchlist(
                spy_regime = fc.get("regime", "UNKNOWN"),
                vix_now    = d["vix"]["now"],
                vix_prev   = d["vix"]["prev"],
            )
        from src.analysis.value_watchlist import fmt_slack
        from src.notifications.slack_notifier import send_message
        ts  = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
        msg = fmt_slack(signals, fc.get("regime", "UNKNOWN"), d["vix"]["now"], ts)
        ok  = send_message(msg)
        return jsonify({"ok": ok})
    except Exception as e:
        logger.exception("Watchlist alert failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/bwb/alert", methods=["POST"])
def api_bwb_alert():
    """Post BWB scan results to Slack."""
    try:
        tier = request.args.get("tier", "low_risk")
        with _scan_lock:
            results = _scan_cache["data"]
        if not results:
            from src.analysis.bwb_scanner import scan
            vix = get_data()["vix"]
            results = scan(tier=tier, vix_now=vix["now"], vix_prev=vix["prev"])
        d  = get_data()
        fc = d["spy"].get("full_chain") or {}
        from src.analysis.bwb_scanner import fmt_slack as bwb_fmt
        from src.notifications.slack_notifier import send_message
        ts  = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
        msg = bwb_fmt(results, tier, fc.get("regime", "UNKNOWN"), d["vix"]["now"], ts)
        ok  = send_message(msg)
        return jsonify({"ok": ok})
    except Exception as e:
        logger.exception("BWB alert failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/fallen-angels", methods=["GET"])
def api_fallen_angels():
    """Scan for beaten-down large-caps near multi-year floor."""
    force = request.args.get("force", "false").lower() == "true"
    try:
        with _fa_lock:
            age = time.time() - _fa_cache["ts"]
            if force or _fa_cache["data"] is None or age > FA_CACHE_TTL:
                logger.info("Running fallen angel scan (force=%s age=%.0fs)", force, age)
                from src.analysis.fallen_angel_scanner import scan_fallen_angels
                _fa_cache["data"] = scan_fallen_angels()
                _fa_cache["ts"]   = time.time()
        return jsonify({"ok": True, "angels": _fa_cache["data"],
                        "timestamp": datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")})
    except Exception as e:
        logger.exception("Fallen angel scan failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/fallen-angels/alert", methods=["POST"])
def api_fallen_angels_alert():
    """Post STRONG + WATCH fallen angel signals to Slack."""
    try:
        with _fa_lock:
            angels = _fa_cache["data"]
        if not angels:
            from src.analysis.fallen_angel_scanner import scan_fallen_angels
            angels = scan_fallen_angels()
        from src.analysis.fallen_angel_scanner import fmt_slack as fa_fmt
        from src.notifications.slack_notifier import send_message
        ts  = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
        msg = fa_fmt(angels, ts)
        ok  = send_message(msg)
        return jsonify({"ok": ok})
    except Exception as e:
        logger.exception("Fallen angel alert failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/bull-put/scan", methods=["GET"])
def api_bull_put_scan():
    force = request.args.get("force", "false").lower() == "true"
    try:
        with _bp_lock:
            age = time.time() - _bp_cache["ts"]
            if force or _bp_cache["data"] is None or age > BP_CACHE_TTL:
                logger.info("Running bull put scan (force=%s age=%.0fs)", force, age)
                from src.analysis.bull_put_scanner import scan as bp_scan
                vix = get_data()["vix"]
                _bp_cache["data"] = bp_scan(vix_now=vix["now"], vix_prev=vix["prev"])
                _bp_cache["ts"]   = time.time()
        return jsonify({"ok": True, "results": _bp_cache["data"],
                        "timestamp": datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")})
    except Exception as e:
        logger.exception("Bull put scan failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/bull-put/alert", methods=["POST"])
def api_bull_put_alert():
    try:
        with _bp_lock:
            results = _bp_cache["data"]
        if not results:
            from src.analysis.bull_put_scanner import scan as bp_scan
            vix = get_data()["vix"]
            results = bp_scan(vix_now=vix["now"], vix_prev=vix["prev"])
        d   = get_data()
        fc  = d["spy"].get("full_chain") or {}
        from src.analysis.bull_put_scanner import fmt_slack as bp_fmt
        from src.notifications.slack_notifier import send_message
        ts  = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
        msg = bp_fmt(results, fc.get("regime", "UNKNOWN"), d["vix"]["now"], ts)
        ok  = send_message(msg)
        return jsonify({"ok": ok})
    except Exception as e:
        logger.exception("Bull put alert failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/eod-report", methods=["POST"])
def api_eod_report():
    """Manually trigger the EOD report + close suggestions to Slack."""
    try:
        _eod_report_job()
        return jsonify({"ok": True})
    except Exception as e:
        logger.exception("Manual EOD report failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/bwb/scan", methods=["GET"])
def api_bwb_scan():
    tier  = request.args.get("tier", "low_risk")
    force = request.args.get("force", "false").lower() == "true"
    try:
        with _scan_lock:
            age = time.time() - _scan_cache["ts"]
            if force or _scan_cache["data"] is None or age > SCAN_CACHE_TTL or _scan_cache["tier"] != tier:
                logger.info("Running BWB scan tier=%s (force=%s age=%.0fs)", tier, force, age)
                from src.analysis.bwb_scanner import scan
                vix = get_data()["vix"]
                _scan_cache["data"] = scan(tier=tier,
                                           vix_now=vix["now"],
                                           vix_prev=vix["prev"])
                _scan_cache["ts"]   = time.time()
                _scan_cache["tier"] = tier
        return jsonify({"ok": True, "tier": tier, "results": _scan_cache["data"]})
    except Exception as e:
        logger.exception("BWB scan failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/analyze")
def analyze_page():
    return render_template("analyze.html")


@app.route("/api/risk", methods=["POST"])
def api_risk():
    """Analyze a hypothetical or live option position (1-6 legs)."""
    try:
        body     = request.get_json(force=True)
        raw_legs = body.get("legs", [])
        if not raw_legs or len(raw_legs) > 6:
            return jsonify({"ok": False, "error": "Provide 1-6 legs"}), 400

        from risk_profile import (
            auto_group_spreads, enrich_legs, build_scenario_table,
            build_expiry_table, max_risk_per_position, _build_occ, SCENARIO_MOVES,
        )
        from datetime import date as _date

        legs = []
        for r in raw_legs:
            underlying = str(r.get("underlying", "")).upper().strip()
            expiry_str = str(r.get("expiry", "")).strip()
            is_call    = bool(r.get("is_call", True))
            strike     = float(r.get("strike", 0))
            qty        = int(r.get("qty", 1))
            entry      = float(r.get("entry_price", 0))
            if not underlying or not expiry_str or strike <= 0:
                continue
            try:
                exp_date = _date.fromisoformat(expiry_str)
            except ValueError:
                continue
            sym = _build_occ(underlying, exp_date, is_call, strike)
            legs.append({
                "source":      "analyze",
                "id":          sym,
                "label":       (f"{underlying} {'C' if is_call else 'P'}"
                                f"{strike:.0f} {exp_date.strftime('%b%y')}"),
                "symbol":      sym,
                "underlying":  underlying,
                "strike":      strike,
                "expiry":      expiry_str,
                "is_call":     is_call,
                "qty":         qty,
                "entry_price": entry,
            })

        if not legs:
            return jsonify({"ok": False, "error": "No valid legs parsed"}), 400

        legs     = auto_group_spreads(legs)
        enriched = enrich_legs(legs, use_api=True)
        if not enriched:
            return jsonify({"ok": False, "error": "Could not fetch live data"}), 500

        for leg in enriched:
            if leg["entry_price"] == 0.0 and leg["mid"] > 0:
                leg["entry_price"] = leg["mid"]

        scenarios    = build_scenario_table(enriched)
        expiry_tbl   = build_expiry_table(enriched)
        max_risks    = max_risk_per_position(enriched)
        u            = enriched[0]["underlying"]
        spot         = enriched[0]["spot"]

        def _fmt_money(v):
            if v is None:
                return "unlimited"
            return f"+${v:,.2f}" if v >= 0 else f"-${abs(v):,.2f}"

        return jsonify({
            "ok":         True,
            "underlying": u,
            "spot":       spot,
            "moves":      SCENARIO_MOVES,
            "legs": [{
                "label":       l["label"],
                "qty":         l["qty"],
                "strike":      l["strike"],
                "is_call":     l["is_call"],
                "expiry":      l["expiry"],
                "entry_price": round(l["entry_price"], 3),
                "mid":         round(l["mid"], 3),
                "iv":          l["iv"],
                "delta":       l["delta"],
                "gamma":       l["gamma"],
                "theta":       l["theta"],
                "vega":        l["vega"],
                "bs_source":   l["bs_source"],
            } for l in enriched],
            "scenarios":    {str(k): v for k, v in (scenarios.get(u) or {}).items()},
            "expiry_table": {str(k): v for k, v in (expiry_tbl.get(u) or {}).items()},
            "max_risks": [{
                "label":    r["label"],
                "max_gain": _fmt_money(r["max_gain"]),
                "max_loss": _fmt_money(r["max_loss"]),
            } for r in max_risks],
            "net_greeks": {
                "delta": round(sum(l["pos_delta"] for l in enriched), 2),
                "gamma": round(sum(l["pos_gamma"] for l in enriched), 2),
                "theta": round(sum(l["pos_theta"] for l in enriched), 2),
                "vega":  round(sum(l["pos_vega"]  for l in enriched), 2),
            },
        })
    except Exception as e:
        logger.exception("Risk analysis failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/cron/manage", methods=["GET", "POST"])
def cron_manage():
    """External-cron heartbeat for the AUTO-EXITS. Hit this every ~5 min during
    market hours (cron-job.org / UptimeRobot) so the 50%/1-DTE and trim/trail
    exits fire even if the in-process scheduler is asleep or was restarted.

    Runs only the RTH-guarded, idempotent management jobs (each no-ops when the
    market is closed and never double-closes via client_order_id), so it is safe
    to run alongside the scheduler and to call repeatedly. Token-protected
    because it can place closing orders.
    """
    token = request.args.get("token") or request.headers.get("X-Cron-Token", "")
    expected = os.getenv("CRON_TOKEN", "")
    if not expected:
        return jsonify({"ok": False, "error": "CRON_TOKEN not configured"}), 503
    if token != expected:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    ran, errs = [], {}
    for name, fn in (("auto_manage", _auto_manage_job),
                     ("fa_manage", _fa_manage_job),
                     ("zerodte_manage", _zerodte_manage_job),
                     ("burrito_manage", _burrito_manage)):
        try:
            fn()
            ran.append(name)
        except Exception as e:
            errs[name] = str(e)
            logger.error("cron_manage %s failed: %s", name, e)
    return jsonify({"ok": not errs, "ran": ran, "errors": errs})


@app.route("/cron/swing", methods=["GET", "POST"])
def cron_swing():
    """External-cron trigger for the GEX swing scanner (advisory only — it scans
    the watchlist and posts setups to Slack, it does NOT place orders). Runs in a
    background thread (the scan takes ~30-90s) so the request returns immediately;
    point cron-job.org at it a few times during market hours. Token-protected."""
    token = request.args.get("token") or request.headers.get("X-Cron-Token", "")
    expected = os.getenv("CRON_TOKEN", "")
    if not expected:
        return jsonify({"ok": False, "error": "CRON_TOKEN not configured"}), 503
    if token != expected:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    if not _alerts_cfg().get("swing", True):
        return jsonify({"ok": True, "paused": True})

    def _run():
        try:
            import swing_scanner            # lazy import: avoids import-time side effects at boot
            swing_scanner.run_scan()
        except Exception as e:
            logger.error("cron_swing run failed: %s", e)
            try:
                from src.notifications.slack_notifier import send_message
                send_message(f":rotating_light: Swing scan error: {e}")
            except Exception:
                pass

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True, "started": True})


@app.route("/cron/shadow", methods=["GET", "POST"])
def cron_shadow():
    """External-cron trigger for the bull-put SHADOW logger (paper, no orders).
    Logs today's STRONG scanner picks, marks the running book to market, and posts
    both to Slack (Slack = durable record since Render's disk is ephemeral). Drive
    it once each market day ~10:05 ET. Token-protected."""
    token = request.args.get("token") or request.headers.get("X-Cron-Token", "")
    expected = os.getenv("CRON_TOKEN", "")
    if not expected:
        return jsonify({"ok": False, "error": "CRON_TOKEN not configured"}), 503
    if token != expected:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    if not _alerts_cfg().get("shadow", True):
        return jsonify({"ok": True, "paused": True})

    def _run():
        # Only log once mid-morning — guards against a too-frequent cron spamming
        # all day. Outside the window it's a silent no-op regardless of schedule.
        t = datetime.now(ET).time()
        if not (dt_time(10, 0) <= t <= dt_time(10, 12)):
            return
        try:
            import shadow_bullput
            added = shadow_bullput.log_picks()
            book  = shadow_bullput.mark()
            if not added:                       # nothing new -> stay quiet
                return
            from src.notifications.slack_notifier import send_message
            head = (f":ledger: *Shadow bull-put* — logged {len(added)} new pick(s)\n"
                    + "\n".join(f"  • {r['ticker']} {r['short']}/{r['long']}P {r['expiry']} "
                                f"credit ${r['entry_credit']} ({r['credit_pct']}%)" for r in added))
            send_message(head + "\n" + book)
        except Exception as e:
            logger.error("cron_shadow failed: %s", e)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True, "started": True})


def _dca_run():
    """Income-core DCA: buy weekly_budget split by config/dca.json weights as
    fractional MARKET orders (RTH only), idempotent per ISO week. Respects the
    risk-engine cash floor but intentionally keeps buying through drawdowns."""
    from src.live.alpaca_options import _trading
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    path = os.path.join(os.path.dirname(__file__), "config", "dca.json")
    cfg = {"enabled": False, "weekly_budget": 20.0, "respect_cash_floor": True, "weights": {}}
    try:
        with open(path) as f:
            cfg.update(json.load(f))
    except Exception:
        pass
    if not cfg.get("enabled") or not cfg.get("weights"):
        return [":information_source: DCA disabled or no weights configured."]
    if not _market_is_open():
        return [":hourglass: DCA skipped — market closed (fractional orders need RTH)."]
    client = _trading()
    acct   = client.get_account()
    equity = float(acct.equity); cash = float(getattr(acct, "cash", 0) or 0)
    budget = float(cfg["weekly_budget"])
    if cfg.get("respect_cash_floor", True):
        floor = _risk_config().get("cash_floor_pct", 0.30) * equity
        if cash - budget < floor:
            return [f":octagonal_sign: DCA skipped — would breach cash floor "
                    f"(cash ${cash:.0f}, floor ${floor:.0f}, budget ${budget:.0f})."]
    y, wk, _ = datetime.now(ET).isocalendar()
    period   = f"{y}W{wk:02d}"
    lines = [f":moneybag: *Income-core DCA* — ${budget:.0f} this week ({period})"]
    for tk, wt in cfg["weights"].items():
        notional = round(budget * float(wt), 2)
        if notional < 1:
            continue
        coid = f"dca-{tk}-{period}"
        try:
            if client.get_order_by_client_id(coid):
                lines.append(f"  • {tk}: already bought this week"); continue
        except Exception:
            pass
        try:
            o = client.submit_order(MarketOrderRequest(
                symbol=tk, notional=notional, side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY, client_order_id=coid))
            lines.append(f"  • {tk}: bought ${notional:.2f}  (`{str(o.id)[:8]}`)")
        except Exception as e:
            lines.append(f"  • {tk}: failed — {e}")
    return lines


@app.route("/cron/dca", methods=["GET", "POST"])
def cron_dca():
    """External-cron trigger for the income-core DCA. Drive once a week during
    RTH (e.g. Wed 11:00 ET). Token-protected."""
    token = request.args.get("token") or request.headers.get("X-Cron-Token", "")
    expected = os.getenv("CRON_TOKEN", "")
    if not expected:
        return jsonify({"ok": False, "error": "CRON_TOKEN not configured"}), 503
    if token != expected:
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    def _run():
        try:
            lines = _dca_run()
            from src.notifications.slack_notifier import send_message
            send_message("\n".join(lines))
        except Exception as e:
            logger.error("cron_dca failed: %s", e)
            try:
                from src.notifications.slack_notifier import send_message
                send_message(f":rotating_light: DCA error: {e}")
            except Exception:
                pass

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True, "started": True})


# ── Burrito Butterfly (SPY swing) ─────────────────────────────────────────────
# ATM butterfly + directional debit spread as ONE 4-leg MLEG. Proposed via Slack
# Take button; auto-wrapped + managed off real Alpaca P&L in /cron/manage.

def _burrito_config() -> dict:
    path = os.path.join(os.path.dirname(__file__), "config", "burrito.json")
    cfg = {"enabled": True, "width": 5, "dte_min": 4, "dte_max": 9, "direction": "auto",
           "sma_days": 20, "profit_target_pct": 0.25, "stop_pct": 0.50,
           "wrap_trigger_pct": 0.10, "force_close_dte": 1, "max_loss": 150}
    try:
        with open(path) as f:
            cfg.update(json.load(f))
    except Exception:
        pass
    return cfg


def _spy_sma(days: int):
    """Simple moving average of SPY daily closes (Alpaca), or None."""
    import requests as _rq
    from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY
    H = {"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY}
    end = (datetime.now(ET).date() - timedelta(days=1)).isoformat()
    start = (datetime.now(ET).date() - timedelta(days=days * 2 + 15)).isoformat()
    try:
        r = _rq.get("https://data.alpaca.markets/v2/stocks/SPY/bars", headers=H,
                    params={"timeframe": "1Day", "start": start, "end": end,
                            "limit": 1000, "adjustment": "raw"}, timeout=20)
        closes = [b["c"] for b in (r.json().get("bars") or [])] if r.status_code == 200 else []
    except Exception:
        closes = []
    return (sum(closes[-days:]) / min(len(closes), days)) if closes else None


def _burrito_expiry(cfg) -> date:
    """Nearest Friday in the DTE window (SPY always lists Friday expirations)."""
    today = datetime.now(ET).date()
    for d in range(int(cfg["dte_min"]), int(cfg["dte_max"]) + 1):
        cand = today + timedelta(days=d)
        if cand.weekday() == 4:
            return cand
    return today + timedelta(days=int(cfg["dte_max"]))


def _burrito_candidate():
    """Build the SPY directional butterfly (fly + debit spread). Returns (cand, reason)."""
    cfg = _burrito_config()
    if not cfg.get("enabled"):
        return None, "disabled"
    W = float(cfg["width"])
    spot = _stock_price("SPY")
    if not spot:
        return None, "no SPY price"
    mode = cfg.get("direction", "auto")
    if mode == "bull":
        bull = True
    elif mode == "bear":
        bull = False
    else:
        sma = _spy_sma(int(cfg["sma_days"]))
        bull = (sma is None) or (spot >= sma)
    K = float(round(spot))
    exp = _burrito_expiry(cfg)
    if bull:                                          # fly + call debit spread
        legs = [{"action": "BUY",  "strike": K - W,     "opt": "C", "qty": 1},
                {"action": "SELL", "strike": K,         "opt": "C", "qty": 2},
                {"action": "BUY",  "strike": K + W,     "opt": "C", "qty": 2},
                {"action": "SELL", "strike": K + 2 * W, "opt": "C", "qty": 1}]
    else:                                             # fly + put debit spread (mirror)
        legs = [{"action": "BUY",  "strike": K + W,     "opt": "P", "qty": 1},
                {"action": "SELL", "strike": K,         "opt": "P", "qty": 2},
                {"action": "BUY",  "strike": K - W,     "opt": "P", "qty": 2},
                {"action": "SELL", "strike": K - 2 * W, "opt": "P", "qty": 1}]
    net, _ = _reprice_net(_legs_to_occ("SPY", exp, legs))
    if net is None:
        return None, "no quotes"
    if net > 0:
        return None, "priced as a credit (unexpected) — skipped"
    max_loss = round(abs(net) * 100, 2)               # 0-delta wings -> max loss = net debit
    if max_loss > float(cfg["max_loss"]):
        return None, f"max loss ${max_loss:.0f} > cap ${cfg['max_loss']:.0f}"
    dirn = "BULL" if bull else "BEAR"
    dte  = (exp - datetime.now(ET).date()).days
    label = f"Burrito {dirn} SPY {K:.0f}±{W:.0f} {exp}"
    text  = (f"*{label}*  net debit ${max_loss/100:.2f}  (max loss ${max_loss:.0f})  {dte}DTE "
             f"— fly + {'call' if bull else 'put'} debit spread")
    return {"legs": legs, "expiry": exp, "net": net, "max_loss": max_loss,
            "label": label, "text": text, "bull": bull, "K": K, "W": W}, "ok"


def _burrito_propose():
    cand, reason = _burrito_candidate()
    if not cand:
        return f":burrito: *Burrito* — no trade ({reason})"
    tid = register_entry("burrito", "SPY", cand["expiry"], cand["legs"], 1,
                         cand["net"], cand["label"], cand["text"])
    _post_entry_approvals([{"tid": tid, "text": cand["text"], "label": cand["label"]}])
    return f":burrito: *Burrito proposed* — {cand['label']} · max loss ${cand['max_loss']:.0f} · tap *Take*"


@app.route("/cron/levitation", methods=["GET", "POST"])
def cron_levitation():
    """Propose the daily SPY Levitation trade (0DTE bull-put + temp hedge).
    Hit once per market day inside the 9:45-11:30 ET entry window."""
    token = request.args.get("token") or request.headers.get("X-Cron-Token", "")
    expected = os.getenv("CRON_TOKEN", "")
    if not expected:
        return jsonify({"ok": False, "error": "CRON_TOKEN not configured"}), 503
    if token != expected:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    force = str(request.args.get("force", "")).lower() in ("1", "true", "yes")

    def _run():
        try:
            from src.notifications.slack_notifier import send_message
            from src.analysis.levitation_scanner import scan_levitation, fmt_slack
            if not _market_is_open() and not force:
                send_message(":feather: *Levitation* — market closed, no proposal "
                             "(use &force=1 to preview)")
                return
            vix = get_data()["vix"]
            r = scan_levitation(vix_now=vix["now"], vix_prev=vix["prev"])
            send_message(fmt_slack(r))
        except Exception as e:
            logger.error("cron_levitation failed: %s", e)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True, "started": True})


@app.route("/cron/burrito", methods=["GET", "POST"])
def cron_burrito():
    """Propose a SPY Burrito Butterfly (Slack Take button). Hit once/market day."""
    token = request.args.get("token") or request.headers.get("X-Cron-Token", "")
    expected = os.getenv("CRON_TOKEN", "")
    if not expected:
        return jsonify({"ok": False, "error": "CRON_TOKEN not configured"}), 503
    if token != expected:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    force = str(request.args.get("force", "")).lower() in ("1", "true", "yes")

    def _run():
        try:
            from src.notifications.slack_notifier import send_message
            if not _market_is_open() and not force:
                send_message(":burrito: *Burrito* — market closed, no proposal (use &force=1 to preview)")
                return
            send_message(_burrito_propose())
        except Exception as e:
            logger.error("cron_burrito failed: %s", e)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True, "started": True})


def _burrito_close(client, entry_legs, reason: str, exp) -> bool:
    """Close a burrito by reversing its entry legs as one signed-limit MLEG."""
    from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, PositionIntent, OrderClass
    order_legs = [{"symbol": lg.symbol,
                   "side": "BUY" if str(lg.side).endswith("SELL") else "SELL",   # reverse to close
                   "ratio": int(lg.ratio_qty)} for lg in entry_legs]
    net, priced = _reprice_net(order_legs)
    if net is None:
        return False
    leg_objs = [OptionLegRequest(
                    symbol=lg["symbol"], ratio_qty=lg["ratio"],
                    side=OrderSide.SELL if lg["side"] == "SELL" else OrderSide.BUY,
                    position_intent=(PositionIntent.SELL_TO_CLOSE if lg["side"] == "SELL"
                                     else PositionIntent.BUY_TO_CLOSE))
                for lg in priced]
    try:
        client.submit_order(LimitOrderRequest(
            qty=1, order_class=OrderClass.MLEG, side=OrderSide.SELL if net > 0 else OrderSide.BUY,
            type=OrderType.LIMIT, time_in_force=TimeInForce.DAY,
            limit_price=round(-net, 2), legs=leg_objs,
            client_order_id=f"burritoclose-SPY-{exp}"))
        from src.notifications.slack_notifier import send_message
        send_message(f":burrito: *Burrito closed* ({exp}) — {reason}")
        logger.info("Burrito close %s — %s", exp, reason)
        return True
    except Exception as e:
        logger.error("Burrito close failed %s: %s", exp, e)
        return False


def _burrito_manage():
    """Every RTH tick (via /cron/manage): take profit / stop / force-close at 1-DTE
    on open burritos, keyed off real Alpaca P&L. (Auto-wrap: phase 2.)"""
    if not _market_is_open():
        return
    cfg = _burrito_config()
    from src.live.alpaca_options import _trading
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    client = _trading()
    try:
        orders = client.get_orders(filter=GetOrdersRequest(
            status=QueryOrderStatus.ALL, limit=300, after=datetime.now(ET) - timedelta(days=21)))
    except Exception as e:
        logger.warning("burrito-manage get_orders: %s", e); return
    positions = {p.symbol: p for p in client.get_all_positions()}
    closed_exp = {(getattr(o, "client_order_id", "") or "").replace("burritoclose-SPY-", "")
                  for o in orders if (getattr(o, "client_order_id", "") or "").startswith("burritoclose-SPY-")}
    for o in orders:
        coid = getattr(o, "client_order_id", "") or ""
        if not coid.startswith("burrito-SPY-") or not str(o.status).endswith("FILLED"):
            continue
        legs = [lg for lg in (getattr(o, "legs", None) or []) if lg.symbol in positions]
        if not legs:
            continue                                   # already closed / expired
        p0 = _occ_parse(legs[0].symbol)
        if not p0:
            continue
        exp = p0[1]
        if str(exp) in closed_exp:
            continue
        dte = (exp - datetime.now(ET).date()).days
        pnl = sum(float(getattr(positions[lg.symbol], "unrealized_pl", 0) or 0) for lg in legs)
        risk = abs(float(getattr(o, "filled_avg_price", 0) or 0)) * 100 or float(cfg["max_loss"])
        if dte <= int(cfg["force_close_dte"]):
            _burrito_close(client, legs, f"1-DTE force-close (assignment/valley guard)", exp)
        elif pnl >= float(cfg["profit_target_pct"]) * risk:
            _burrito_close(client, legs, f"+{cfg['profit_target_pct']*100:.0f}% target (${pnl:+.0f})", exp)
        elif pnl <= -float(cfg["stop_pct"]) * risk:
            _burrito_close(client, legs, f"-{cfg['stop_pct']*100:.0f}% stop (${pnl:+.0f})", exp)


# ── slack formatter ─────────────────────────────────────────────────────────────

def _fmt_slack(d: dict) -> str:
    spy = d["spy"]
    qqq = d["qqq"]
    vix = d["vix"]
    t   = d.get("trade_idea") or {}
    ts  = d["timestamp"]

    vix_arrow = (":arrow_up:" if vix["change"] > 0.3
                 else ":arrow_down:" if vix["change"] < -0.3
                 else ":left_right_arrow:")

    def ticker_block(name: str, td: dict) -> str:
        fc  = td.get("full_chain") or {}
        t1  = td.get("tomorrow")   or {}
        chg = td["change"]
        arr = ":arrow_up_small:" if chg >= 0 else ":arrow_down_small:"
        lines = [f"*{name}* ${td['spot']:.2f} {arr} ({chg:+.2f})"]
        if fc:
            lines.append(
                f">  Full chain  -- Flip: ${fc['flip_level']:.0f}  "
                f"Put: ${fc['put_wall']:.0f}  Call: ${fc['call_wall']:.0f}  "
                f"GEX: {fc['gex']:+.0f}B"
            )
        if t1:
            lines.append(
                f">  Tomorrow    -- Flip: ${t1['flip_level']:.0f}  "
                f"Put: ${t1['put_wall']:.0f}  Call: ${t1['call_wall']:.0f}  "
                f"GEX: {t1['gex']:+.0f}B"
            )
        regime = (fc or t1).get("regime", "")
        badge  = ":red_circle:" if "NEGATIVE" in regime else ":green_circle:"
        lines.append(f">  Regime: {badge} {regime.replace('_', ' ').title()}")
        return "\n".join(lines)

    trade_block = ("\n" + _fmt_spy_trade_block(t)) if t else ""

    return (
        f":bar_chart: *GEX Dashboard -- {ts}*\n"
        f":zap: VIX {vix['now']:.2f} {vix_arrow} ({vix['change']:+.2f})\n\n"
        + ticker_block("SPY", spy) + "\n\n"
        + ticker_block("QQQ", qqq)
        + trade_block
    )


# ── Bull Put Scheduled Scanner (HITL alerts) ──────────────────────────────────

def _structure_name(legs: list) -> str:
    """Human name for an option structure from its legs (each: qty, strike,
    otype 'C'/'P'). Legs are one (underlying, expiry) group, so a vertical is 2,
    a butterfly/BWB is 3, a condor/iron condor is 4. Calendars span expiries and
    show up as separate single/vertical groups (Alpaca doesn't tag structures)."""
    n = len(legs)
    types = {lg["otype"] for lg in legs}
    if n == 1:
        lg   = legs[0]
        side = "long" if lg["qty"] > 0 else "short"
        return f"{side} {'call' if lg['otype'] == 'C' else 'put'}"
    if n == 2:
        if types == {"C", "P"}:                       # straddle/strangle/combo
            strikes = {lg["strike"] for lg in legs}
            return "straddle" if len(strikes) == 1 else "strangle"
        return "vertical"
    if n == 3:
        sl = sorted(legs, key=lambda x: x["strike"])
        w1 = sl[1]["strike"] - sl[0]["strike"]
        w2 = sl[2]["strike"] - sl[1]["strike"]
        return "butterfly" if abs(w1 - w2) < 0.01 else "BWB"
    if n == 4:
        return "iron condor" if types == {"C", "P"} else "condor"
    return f"{n}-leg spread"


def _fmt_strikes(legs: list) -> str:
    """Compact strike label for a structure, e.g. '5895/5900P' or '5850/5860/5940/5950CP'."""
    ks   = "/".join(f"{lg['strike']:g}" for lg in sorted(legs, key=lambda x: x["strike"]))
    kind = "".join(sorted({lg["otype"] for lg in legs}))   # C, P, or CP
    return f"{ks}{kind}"


def _build_position_summary() -> tuple[str, str]:
    """
    Returns (close_block, positions_block) for open positions.
    close_block  — urgent close suggestions with runnable commands.
    positions_block — one line per position: equities individually, and option
    legs grouped into their structure (vertical/butterfly/condor/…) with the
    structure's combined market value and P&L.
    """
    from src.live.alpaca_options import _trading
    from datetime import date

    try:
        client    = _trading()
        positions = client.get_all_positions()
        acct      = client.get_account()
        bp        = float(getattr(acct, "options_buying_power", 0) or 0)
    except Exception as e:
        logger.warning("Position fetch failed in scan job: %s", e)
        return "", ""

    if not positions:
        return "", f"  _No open positions_  |  BP ${bp:,.2f}"

    today      = date.today()
    pos_lines  = []
    close_tips = []

    # -- split equities from option legs, grouping options by (underlying, expiry) --
    equities: list       = []
    opt_groups: dict     = {}
    for p in positions:
        sym    = p.symbol
        parsed = _occ_parse(sym) if len(sym) > 6 else None
        if parsed is None:
            equities.append(p)
            continue
        underlying, exp, strike, otype = parsed
        opt_groups.setdefault((underlying, exp), []).append({
            "qty":    float(p.qty),
            "strike": strike, "otype": otype,
            "mkt":    float(getattr(p, "market_value", 0) or 0),
            "unreal": float(getattr(p, "unrealized_pl", 0) or 0),
            "cost":   float(getattr(p, "cost_basis", 0) or 0),
        })

    # -- equities: one line each -------------------------------------------
    for p in equities:
        sym    = p.symbol
        unreal = float(getattr(p, "unrealized_pl",   0) or 0)
        pct    = float(getattr(p, "unrealized_plpc", 0) or 0) * 100
        mkt    = float(getattr(p, "market_value",    0) or 0)
        icon   = ":green_circle:" if unreal >= 0 else ":red_circle:"
        pos_lines.append(
            f"  {icon} `{sym}`  ${mkt:,.2f}  P&L ${unreal:+,.2f} ({pct:+.1f}%)"
        )
        if pct <= -8:
            close_tips.append(
                f":rotating_light: *`{sym}`* — down {pct:.1f}%, stop-loss zone.\n"
                f"  > `python close_position.py --ticker {sym}`"
            )
        elif pct >= 20:
            close_tips.append(
                f":moneybag: *`{sym}`* — up {pct:.1f}%, consider trimming.\n"
                f"  > `python close_position.py --ticker {sym}`"
            )

    # -- option structures: one line per (underlying, expiry) group --------
    for (underlying, exp), legs in sorted(opt_groups.items(), key=lambda kv: kv[0][1]):
        dte     = (exp - today).days
        mkt     = sum(lg["mkt"]    for lg in legs)
        unreal  = sum(lg["unreal"] for lg in legs)
        cost    = sum(lg["cost"]   for lg in legs)          # signed; <0 = net credit
        pct     = (unreal / abs(cost) * 100) if cost else 0.0
        icon    = ":green_circle:" if unreal >= 0 else ":red_circle:"
        name    = _structure_name(legs)
        strikes = _fmt_strikes(legs)
        pos_lines.append(
            f"  {icon} *{underlying}* {name} `{strikes}` {exp.strftime('%m/%d')}"
            f"  ${mkt:,.2f}  P&L ${unreal:+,.2f} ({pct:+.1f}%)  {dte}DTE"
        )

        if dte <= 3:
            close_tips.append(
                f":warning: *{underlying} {name}* ({strikes}) — {dte} DTE, expires soon.\n"
                f"  > `python close_bwb.py --ticker {underlying}`  _(or let expire — defined-risk)_"
            )
        # Credit structure at 50%+ of max profit captured
        net_credit = -cost
        if net_credit > 0 and unreal >= 0.5 * net_credit:
            profit_pct = unreal / net_credit * 100
            close_tips.append(
                f":moneybag: *{underlying} {name}* ({strikes}) — {profit_pct:.0f}% of max profit captured.\n"
                f"  > `python close_bwb.py --ticker {underlying}`"
            )

    pos_block   = "\n".join(pos_lines) + f"\n  BP available: ${bp:,.2f}"
    close_block = "\n\n".join(close_tips) if close_tips else ""
    return close_block, pos_block


def _premarket_prep_job():
    """Runs at 8:30 AM ET Mon-Fri.
    Uses prior-day cached GEX (real OI, HIGH confidence) + fresh pre-market
    prices from yfinance. No options chain fetch needed — walls from yesterday
    remain valid until the market opens and new OI prints.
    """
    try:
        from src.analysis.gex_scanner import load_chain, compute_exposures, format_gex_message
        from src.notifications.slack_notifier import send_message

        ts = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")

        # -- Most recent cached chain (prior trading day) ---------------------
        contracts  = None
        chain_date = None
        for offset in range(5):
            d = date.today() - timedelta(days=offset + 1)
            c = load_chain(d)
            if c:
                contracts  = c
                chain_date = d
                break

        # -- Pre-market prices via yfinance -----------------------------------
        prices      = {}   # sym -> latest price
        prev_closes = {}   # sym -> prior close
        try:
            import yfinance as yf
            tickers = yf.download(
                "SPY QQQ IWM ^VIX", period="3d", auto_adjust=True,
                progress=False, threads=True
            )
            closes = tickers["Close"]
            for sym in ["SPY", "QQQ", "IWM", "^VIX"]:
                col = sym
                if col in closes.columns and len(closes[col].dropna()) >= 2:
                    vals = closes[col].dropna()
                    prices[sym]      = float(vals.iloc[-1])
                    prev_closes[sym] = float(vals.iloc[-2])
        except Exception as e:
            logger.warning("yfinance pre-market fetch failed: %s", e)

        # Fallback: use get_data() VIX if yfinance empty
        if not prices:
            try:
                d_data = get_data()
                vx = d_data.get("vix", {})
                prices["^VIX"]      = vx.get("now", 0)
                prev_closes["^VIX"] = vx.get("prev", 0)
                sp = d_data.get("spy", {}).get("spot", 0)
                if sp:
                    prices["SPY"] = sp
            except Exception:
                pass

        spy_spot = prices.get("SPY", 0)
        vix_now  = prices.get("^VIX", 0)
        vix_prev = prev_closes.get("^VIX", vix_now)

        # -- GEX from cached chain --------------------------------------------
        gex_block = ""
        if contracts and spy_spot:
            try:
                result     = compute_exposures(spy_spot, vix_now, vix_prev, contracts)
                result.regime_confidence = "HIGH"
                gex_block  = format_gex_message(result, session="morning")
                gex_block += f"\n  _GEX from {chain_date} cache (real OI)_"
            except Exception as e:
                logger.warning("GEX compute in prep job failed: %s", e)

        # -- Index price lines ------------------------------------------------
        index_lines = []
        for sym in ["SPY", "QQQ", "IWM"]:
            px   = prices.get(sym)
            prev = prev_closes.get(sym)
            if px and prev:
                chg  = (px - prev) / prev * 100
                icon = ":green_circle:" if chg >= 0 else ":red_circle:"
                index_lines.append(f"  {icon} *{sym}* ${px:.2f}  ({chg:+.1f}% prev close)")
            elif px:
                index_lines.append(f"  *{sym}* ${px:.2f}")

        if vix_now:
            vix_chg   = vix_now - vix_prev if vix_prev else 0
            vix_icon  = (":green_circle:" if vix_now < 18
                         else ":large_yellow_circle:" if vix_now < 25
                         else ":red_circle:")
            vix_label = ("LOW — favorable for credit spreads" if vix_now < 18
                         else "ELEVATED — reduce size" if vix_now < 25
                         else "HIGH — caution, widen strikes")
            index_lines.append(
                f"  {vix_icon} *VIX* {vix_now:.1f} ({vix_chg:+.1f})  _{vix_label}_"
            )

        # -- Expected move (straddle-based) -----------------------------------
        em_lines = []
        try:
            from src.analysis.expected_move import compute as _em
            for lbl, want in (("0DTE", 0), ("Weekly", 7)):
                em = _em("SPY", want)
                if em:
                    em_lines.append(
                        f"  *{lbl}* ({em['expiry']}, {em['dte']}DTE): ±${em['em_dollars']:.2f} "
                        f"({em['em_pct']:.1f}%)  → {em['lower']:.0f}–{em['upper']:.0f}")
        except Exception as e:
            logger.warning("premarket expected-move failed: %s", e)

        # -- Open positions snapshot ------------------------------------------
        _, pos_block = _build_position_summary()

        # -- Assemble message -------------------------------------------------
        sections = [f":clipboard: *Pre-Market Prep — {ts}*"]

        if index_lines:
            sections += ["", "*--- MARKET ---*"] + index_lines

        if em_lines:
            sections += ["", "*--- EXPECTED MOVE (SPY, ATM straddle) ---*"] + em_lines

        if gex_block:
            sections += ["", "*--- GEX WALLS (prior close, HIGH confidence) ---*", gex_block]

        if pos_block:
            sections += ["", "*--- POSITIONS (pre-market) ---*", pos_block]

        sections.append("\n_Live options scan fires at 9:45 AM ET._")

        send_message("\n".join(sections))
        logger.info("Pre-market prep sent — spy=%.2f vix=%.2f gex_cached=%s",
                    spy_spot, vix_now, bool(contracts))

    except Exception as e:
        logger.error("_premarket_prep_job error: %s", e)


# ── Stock-holding rotation: sell laggards, rotate into faster movers ────────────
#
# Daily-gain rotation. For each equity holding we estimate trailing daily return
# and trend; against the best STRONG value-watchlist candidate we hold. A swap is
# only suggested when the candidate's avg daily return beats the laggard by a
# clear margin AND the laggard has a reason to exit (sell trigger / below trend).

ROTATION_EDGE_PCT   = 0.30   # candidate must out-gain holding by ≥0.30 %/day
ROTATION_LOOKBACK   = 10     # sessions for trailing avg daily return


def _get_watchlist(force: bool = False) -> list:
    """Cached value-watchlist scan (30-min TTL) — shared by the route and rotation."""
    with _wl_lock:
        age = time.time() - _wl_cache["ts"]
        if force or _wl_cache["data"] is None or age > WL_CACHE_TTL:
            d  = get_data()
            fc = d["spy"].get("full_chain") or {}
            from src.analysis.value_watchlist import scan_watchlist
            _wl_cache["data"] = scan_watchlist(
                spy_regime=fc.get("regime", "UNKNOWN"),
                vix_now=d["vix"]["now"], vix_prev=d["vix"]["prev"],
            )
            _wl_cache["ts"] = time.time()
        return _wl_cache["data"] or []


def _stock_daily_metrics(tickers: list) -> dict:
    """Trailing daily-return metrics from Alpaca daily bars.
    Returns {ticker: {avg_daily_ret, mom_pct, last, above_ma}}."""
    out: dict = {}
    tickers = [t for t in dict.fromkeys(tickers) if t]
    if not tickers:
        return out
    try:
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY

        client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        req = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=TimeFrame.Day,
            start=datetime.now(ET) - timedelta(days=40),
        )
        bars = client.get_stock_bars(req).data
    except Exception as e:
        logger.warning("Rotation: daily-bar fetch failed: %s", e)
        return out

    for tk in tickers:
        rows = bars.get(tk) or []
        closes = [float(b.close) for b in rows if getattr(b, "close", None)]
        if len(closes) < ROTATION_LOOKBACK + 1:
            continue
        rets = [(closes[i] / closes[i - 1] - 1) * 100 for i in range(1, len(closes))]
        avg_daily = sum(rets[-ROTATION_LOOKBACK:]) / ROTATION_LOOKBACK
        mom_pct   = (closes[-1] / closes[-min(20, len(closes))] - 1) * 100
        ma        = sum(closes[-20:]) / min(20, len(closes))
        out[tk] = {
            "avg_daily_ret": round(avg_daily, 3),
            "mom_pct":       round(mom_pct, 2),
            "last":          round(closes[-1], 2),
            "above_ma":      closes[-1] >= ma,
        }
    return out


def _stock_rotation_analysis():
    """Return a rotation suggestion dict or None.

    {sell, buy, sell_metrics, buy_metrics, edge, reasons, sell_cmd, buy_note}"""
    try:
        from src.live.alpaca_options import _trading
        client    = _trading()
        positions = client.get_all_positions()
    except Exception as e:
        logger.warning("Rotation: position fetch failed: %s", e)
        return None

    holdings = [
        p.symbol.upper() for p in positions
        if getattr(p, "asset_class", "") in ("us_equity", "equity")
    ]
    if not holdings:
        return None

    # Candidate pool = STRONG/WATCH watchlist names we don't already hold.
    vw = _get_watchlist()
    candidates = [
        r for r in vw
        if r.get("signal") in ("STRONG", "WATCH") and r.get("ticker", "").upper() not in holdings
    ]
    if not candidates:
        return None

    cand_syms = [r["ticker"].upper() for r in candidates]
    metrics   = _stock_daily_metrics(holdings + cand_syms)
    if not metrics:
        return None

    held_pnl = {
        p.symbol.upper(): float(getattr(p, "unrealized_plpc", 0) or 0) * 100
        for p in positions
    }

    # Weakest holding by trailing daily return.
    held_m = [(h, metrics[h]) for h in holdings if h in metrics]
    if not held_m:
        return None
    sell_tk, sell_m = min(held_m, key=lambda kv: kv[1]["avg_daily_ret"])

    sell_reasons = []
    if not sell_m["above_ma"]:
        sell_reasons.append("below 20-day trend")
    if sell_m["mom_pct"] < 0:
        sell_reasons.append(f"20d momentum {sell_m['mom_pct']:+.1f}%")
    if sell_m["avg_daily_ret"] < 0:
        sell_reasons.append(f"avg {sell_m['avg_daily_ret']:+.2f}%/day")
    if held_pnl.get(sell_tk, 0) < -5:
        sell_reasons.append(f"position {held_pnl[sell_tk]:+.1f}%")
    if not sell_reasons:
        return None   # no real reason to exit anything

    # Best candidate by trailing daily return.
    cand_m = [(c, metrics[c]) for c in cand_syms if c in metrics]
    if not cand_m:
        return None
    buy_tk, buy_m = max(cand_m, key=lambda kv: kv[1]["avg_daily_ret"])

    edge = round(buy_m["avg_daily_ret"] - sell_m["avg_daily_ret"], 3)
    if edge < ROTATION_EDGE_PCT or not buy_m["above_ma"]:
        return None   # no clear daily-gain edge

    return {
        "sell": sell_tk, "buy": buy_tk,
        "sell_metrics": sell_m, "buy_metrics": buy_m,
        "edge": edge, "reasons": sell_reasons,
        "sell_cmd": f"/close_position {sell_tk}",
        "buy_note": f"buy {buy_tk} (~+{buy_m['avg_daily_ret']:.2f}%/day, mom {buy_m['mom_pct']:+.1f}%)",
    }


def _fmt_rotation(rot: dict) -> str:
    sm, bm = rot["sell_metrics"], rot["buy_metrics"]
    return "\n".join([
        ":arrows_counterclockwise: *Rotation Idea — higher daily gain*",
        f"  SELL *{rot['sell']}*  ({sm['avg_daily_ret']:+.2f}%/day, mom {sm['mom_pct']:+.1f}%)",
        f"    reasons: {', '.join(rot['reasons'])}",
        f"  BUY  *{rot['buy']}*  ({bm['avg_daily_ret']:+.2f}%/day, mom {bm['mom_pct']:+.1f}%)",
        f"  Edge: *+{rot['edge']:.2f}%/day*",
        f"  :point_right: `{rot['sell_cmd']}`  then buy {rot['buy']}",
    ])


def _alerts_cfg() -> dict:
    path = os.path.join(os.path.dirname(__file__), "config", "alerts.json")
    # Focus mode (2026-07-15): only market updates, FA verticals, and the SPY
    # Burrito are active. These flags gate the rest; absent = prior behavior.
    cfg = {"bwb": False, "xsp": False, "hourly_status": True,
           "bull_put_entries": True, "spy_idea": True, "value_alerts": True,
           "swing": True, "shadow": True}
    try:
        with open(path) as f:
            cfg.update(json.load(f))
    except Exception:
        pass
    return cfg


def _status_hourly_job():
    """Compact SPY / QQQ / VIX snapshot, once an hour during market hours."""
    try:
        if not _market_is_open() or not _alerts_cfg().get("hourly_status", True):
            return
        from src.notifications.slack_notifier import send_message
        d = get_data()
        spy, qqq, vix = d["spy"], d["qqq"], d["vix"]
        ts = datetime.now(ET).strftime("%H:%M ET")
        send_message(
            f":clock1: *Market Status — {ts}*\n"
            f"  SPY ${spy['spot']:.2f} ({spy['change']:+.2f})   "
            f"QQQ ${qqq['spot']:.2f} ({qqq['change']:+.2f})   "
            f"VIX {vix['now']:.2f} ({vix['change']:+.2f})")
    except Exception as e:
        logger.error("hourly status failed: %s", e)


def _adverse_position_alert(state, now):
    """Immediate ping (max once/hour) when open OPTIONS P&L falls to/below the
    adverse threshold — 'a trade has gone against me'. Runs every monitor tick."""
    try:
        thr = float(_alerts_cfg().get("adverse_alert_usd", -100))
        from src.live.alpaca_options import _trading
        opts = [p for p in _trading().get_all_positions() if len(p.symbol) > 8]
        if not opts:
            return
        total = sum(float(getattr(p, "unrealized_pl", 0) or 0) for p in opts)
        last  = float(state.get("adverse_ts", 0) or 0)
        if total <= thr and (time.time() - last) > 3600:
            reds = sorted((p for p in opts if float(getattr(p, "unrealized_pl", 0) or 0) < 0),
                          key=lambda p: float(p.unrealized_pl))
            lines = [f":rotating_light: *Position alert — {now}* — options P&L ${total:+,.0f}"]
            for p in reds[:8]:
                lines.append(f"  :red_circle: `{p.symbol}`  ${float(p.unrealized_pl):+,.0f}")
            from src.notifications.slack_notifier import send_message
            send_message("\n".join(lines))
            state["adverse_ts"] = time.time()
    except Exception as e:
        logger.warning("adverse-position alert failed: %s", e)


def _spy_trade_monitor_job():
    """Every 5 min during RTH. Silent by default — only pings on a trade going
    against you (_adverse_position_alert). The recurring setup chatter (SPY idea /
    GEX fly / condor / rotation) runs only when config/alerts.json monitor_setups
    is true. Original change-gated alerts:
      1. SPY options trade   — fires only when the recommended HIGH-probability
         structure/strikes change from the last alert.
      2. GEX-pinned butterfly — positive-gamma pin play; fires when the best
         fly's body/wing/expiry changes.
      3. GEX-anchored condor  — positive-gamma premium play; fires when the
         short put/call strikes change.
      4. Stock rotation       — fires only when a new sell→buy suggestion appears.
    Unchanged conditions stay silent."""
    try:
        if not _market_is_open():
            return

        from src.notifications.slack_notifier import send_message
        state = _load_spy_state()
        now   = datetime.now(ET).strftime("%H:%M ET")

        # Always: ping if a trade has gone against us (the only routine 5-min alert).
        _adverse_position_alert(state, now)

        # The recurring setup chatter below is silenced unless explicitly enabled.
        if not _alerts_cfg().get("monitor_setups", False):
            state["ts"] = time.time()
            with _spy_state_lock:
                _save_spy_state(state)
            return

        # 1 ── SPY options trade ------------------------------------------------
        d   = get_data()
        t   = d.get("trade_idea") or {}
        sig = _spy_trade_signature(t)
        # stability: how many consecutive ticks this exact idea has held (×5 min).
        # A persistent idea is decision-worthy; one that flickers every tick is noise.
        if sig and sig == state.get("stable_sig"):
            state["stable_n"] = int(state.get("stable_n", 0)) + 1
        else:
            state["stable_sig"], state["stable_n"] = sig, 1
        stable_min = state["stable_n"] * 5
        if sig:
            tier, reasons = _spy_trade_quality(t)
            if tier == "HIGH" and state.get("sig") != sig:
                changed = state.get("sig") is not None
                reasons = list(reasons) + [f"stable {stable_min}m"]
                send_message(_fmt_spy_signal(t, tier, reasons, changed))
                logger.info("SPY monitor: trade alert %s (prev=%s)", sig, state.get("sig"))
                state["sig"] = sig
                state["tier"] = tier
            elif tier != "HIGH":
                logger.info("SPY monitor: tier=%s — not high-probability, hold", tier)

        # 2 ── GEX-pinned butterfly + condor (positive-gamma plays) -------------
        try:
            from src.analysis.gex_scanner import scan as gex_scan
            from src.analysis.butterfly_scanner import scan as fly_scan, fmt_slack as fly_fmt
            from src.analysis.condor_scanner import scan as condor_scan, fmt_slack as condor_fmt
            gx = gex_scan()

            # Bucket strikes so tiny (1–2pt) wiggles don't re-fire the same idea.
            # Re-alert only on a meaningful strike move (bucket grid) / expiry change.
            def _b(x, n):
                try:
                    return int(round(float(x) / n) * n)
                except Exception:
                    return x

            fly = fly_scan(gx.spot, gx, dte_min=0, dte_max=10) if gx else {}
            cands = fly.get("candidates") or []
            top = cands[0] if cands else None
            fly_sig = (f"{top['ticker']}|{_b(top['short_body'],2)}|{top['wing']}|{top['expiry']}"
                       if top else None)
            if fly_sig and state.get("fly_sig") != fly_sig:
                send_message(f":alarm_clock: _{now}_\n" + fly_fmt(fly))
                logger.info("SPY monitor: fly alert %s (prev=%s)", fly_sig, state.get("fly_sig"))
                state["fly_sig"] = fly_sig
            elif not fly_sig:
                state["fly_sig"] = None   # no pin play (e.g. negative gamma) — reset

            con = condor_scan(gx.spot, gx, dte_min=0, dte_max=7) if gx else {}
            cc = con.get("candidate")
            con_sig = (f"{cc['ticker']}|{_b(cc['short_put'],5)}|{_b(cc['short_call'],5)}|{cc['expiry']}"
                       if cc else None)
            if con_sig and state.get("con_sig") != con_sig:
                send_message(f":alarm_clock: _{now}_\n" + condor_fmt(con))
                logger.info("SPY monitor: condor alert %s (prev=%s)", con_sig, state.get("con_sig"))
                state["con_sig"] = con_sig
            elif not con_sig:
                state["con_sig"] = None

            from src.analysis.batman_scanner import scan as batman_scan, fmt_slack as batman_fmt
            bat = batman_scan(gx.spot, gx, dte_min=5, dte_max=21) if gx else {}
            bc = bat.get("candidate")
            bat_sig = (f"{_b(bc['put_short'],5)}|{_b(bc['call_short'],5)}|{bc['outer_wing']}|{bc['expiry']}"
                       if bc else None)
            if bat_sig and state.get("bat_sig") != bat_sig:
                send_message(f":alarm_clock: _{now}_\n" + batman_fmt(bat))
                logger.info("SPY monitor: batman alert %s (prev=%s)", bat_sig, state.get("bat_sig"))
                state["bat_sig"] = bat_sig
            elif not bat_sig:
                state["bat_sig"] = None
        except Exception as e:
            logger.warning("SPY monitor: fly/condor/batman check failed: %s", e)

        # 3 ── Stock rotation ---------------------------------------------------
        rot = _stock_rotation_analysis()
        rot_sig = f"{rot['sell']}->{rot['buy']}" if rot else None
        if rot_sig and state.get("rot_sig") != rot_sig:
            send_message(f":alarm_clock: _{now}_\n" + _fmt_rotation(rot))
            logger.info("SPY monitor: rotation alert %s (prev=%s)", rot_sig, state.get("rot_sig"))
            state["rot_sig"] = rot_sig
        elif not rot_sig:
            state["rot_sig"] = None   # cleared — allow a future suggestion to re-fire

        state["ts"] = time.time()
        with _spy_state_lock:
            _save_spy_state(state)

    except Exception as e:
        logger.error("_spy_trade_monitor_job error: %s", e)


def _unified_scan_job():
    """Runs at 9:45 AM and 12:30 PM ET Mon-Fri.
    Runs all strategy scanners in parallel. Only sends Slack when there is
    something actionable — a STRONG/WATCH signal from any strategy, or a
    position that needs closing. Silent days produce no noise.
    """
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from src.analysis.bull_put_scanner       import scan as bp_scan
        from src.analysis.fallen_angel_scanner   import scan_fallen_angels as fa_scan
        from src.analysis.fallen_angel_scanner   import fmt_slack as fa_fmt
        from src.analysis.value_watchlist        import scan_watchlist as vw_scan
        from src.analysis.value_watchlist        import fmt_slack as vw_fmt
        from src.analysis.bwb_scanner            import scan as bwb_scan
        from src.analysis.butterfly_scanner      import scan as fly_scan, fmt_slack as fly_fmt
        from src.analysis.condor_scanner         import scan as condor_scan, fmt_slack as condor_fmt
        from src.analysis.batman_scanner         import scan as batman_scan, fmt_slack as batman_fmt
        from src.analysis.gex_scanner            import scan as gex_scan, format_gex_message
        from src.notifications.slack_notifier    import send_message

        session = "Morning" if datetime.now(ET).hour < 12 else "Midday"
        ts      = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
        d       = get_data()
        vix     = d["vix"]
        fc      = d["spy"].get("full_chain") or {}
        t       = d.get("trade_idea") or {}
        regime  = fc.get("regime", "UNKNOWN")

        # -- Run all scanners in parallel -------------------------------------
        bp_results = []
        fa_results = []
        vw_results = []
        bwb_results = []
        gex_result = None

        def _run_bp():
            return bp_scan(vix_now=vix["now"], vix_prev=vix["prev"])
        def _run_fa():
            return fa_scan()
        def _run_vw():
            return vw_scan(spy_regime=regime, vix_now=vix["now"])
        def _run_bwb():
            return bwb_scan(tier="all", vix_now=vix["now"], vix_prev=vix["prev"])
        def _run_gex():
            return gex_scan()

        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = {
                ex.submit(_run_bp):  "bp",
                ex.submit(_run_fa):  "fa",
                ex.submit(_run_vw):  "vw",
                ex.submit(_run_bwb): "bwb",
                ex.submit(_run_gex): "gex",
            }
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    result = fut.result()
                    if key == "bp":
                        bp_results = result
                    elif key == "fa":
                        fa_results = result
                    elif key == "vw":
                        vw_results = result
                    elif key == "bwb":
                        bwb_results = result
                    elif key == "gex":
                        gex_result = result
                except Exception as e:
                    logger.warning("Scanner %s failed: %s", key, e)

        # Refresh bull put cache
        with _bp_lock:
            _bp_cache["data"] = bp_results
            _bp_cache["ts"]   = time.time()

        # GEX-pinned butterfly + GEX-anchored condor (positive-gamma only)
        fly_result = {}
        condor_result = {}
        batman_result = {}
        if gex_result:
            try:
                fly_result = fly_scan(gex_result.spot, gex_result, dte_min=0, dte_max=10)
            except Exception as e:
                logger.warning("Butterfly scan failed: %s", e)
            try:
                condor_result = condor_scan(gex_result.spot, gex_result, dte_min=0, dte_max=7)
            except Exception as e:
                logger.warning("Condor scan failed: %s", e)
            try:
                batman_result = batman_scan(gex_result.spot, gex_result, dte_min=5, dte_max=21)
            except Exception as e:
                logger.warning("Batman scan failed: %s", e)
                batman_result = {}

        # -- Check if anything is actionable ---------------------------------
        close_block, pos_block = _build_position_summary()

        bp_hits = [r for r in bp_results if r.get("signal") in ("STRONG", "WATCH")]
        fa_hits = [r for r in fa_results if r.get("signal") in ("STRONG", "WATCH")]
        vw_hits = [r for r in vw_results if r.get("signal") in ("STRONG", "WATCH", "SELL")]

        # BWB watchlist: only high-quality, non-earnings candidates count as hits
        bwb_hits = [
            r for r in bwb_results
            if r.get("candidate")
            and (r["candidate"].get("analysis") or {}).get("rating") in ("A+", "Acceptable")
            and r["candidate"].get("earnings_warning") is not True
        ]

        # SPY BWB: a recommended, HIGH-quality butterfly should fire on its own
        rec_spy        = _spy_recommended(t)
        spy_tier, _    = _spy_trade_quality(t)
        spy_bwb_hit    = bool(
            rec_spy
            and rec_spy.get("type") in ("Bull BWB", "Double BWB", "Call BWB")
            and spy_tier == "HIGH"
        )

        fly_hits = fly_result.get("candidates") or []
        condor_hit = condor_result.get("candidate")
        batman_hit = batman_result.get("candidate")

        if not any([close_block, bp_hits, fa_hits, vw_hits, bwb_hits,
                    spy_bwb_hit, fly_hits, condor_hit, batman_hit]):
            logger.info("Unified scan: nothing actionable — skipping Slack alert")
            return

        # -- News / earnings guard (equity premium plays) --------------------
        earn_flags: dict = {}
        try:
            from src.live.news import earnings_guard
            for r in bp_hits + bwb_hits:
                tk = r.get("ticker")
                if tk and tk not in earn_flags:
                    earn_flags[tk] = earnings_guard(tk)
        except Exception as e:
            logger.warning("News guard failed: %s", e)

        def _earn_note(tk: str) -> str:
            g = earn_flags.get(tk) or {}
            return ("\n  :warning: _news: " + g["reason"] +
                    " — verify before selling premium_") if g.get("flagged") else ""

        # -- GEX / SPY context -----------------------------------------------
        gex_block = ""
        try:
            if gex_result:
                gex_block = format_gex_message(gex_result, session=session.lower())
        except Exception as e:
            logger.warning("GEX format failed: %s", e)
            gex_block = (
                f":bar_chart: *SPY Greeks*\n"
                f"  VIX {vix['now']:.2f} ({vix['change']:+.2f})  |  "
                f"Regime: {regime.replace('_',' ').title()}"
            )

        # -- SPY trade idea --------------------------------------------------
        trade_block = _fmt_spy_trade_block(t)

        # -- Bull put /place commands ----------------------------------------
        bp_place_lines = []
        for r in bp_hits:
            c   = r.get("candidate")
            if not c:
                continue
            sig  = r["signal"]
            icon = ":white_check_mark:" if sig == "STRONG" else ":large_yellow_circle:"
            cmd  = f"/place {r['ticker']} {c['short_strike']} {c['long_strike']} {c['expiry']}"
            bp_place_lines.append(
                f"{icon} *{r['ticker']} ${r['spot']:.2f}* ({r.get('change_pct',0):+.1f}%)  "
                f"[{sig} {c['score']:.0f}/9]\n"
                f"  SELL ${c['short_strike']}P / BUY ${c['long_strike']}P  "
                f"{c['dte']}DTE  Credit +${c['credit']:.2f} ({c['credit_pct']:.0f}%)  "
                f"Max loss -${c['max_loss_usd']}\n"
                f"  :point_right: `{cmd}`" + _earn_note(r['ticker'])
            )

        # -- BWB watchlist candidates ----------------------------------------
        bwb_lines = []
        for r in sorted(bwb_hits, key=lambda x: x["candidate"]["analysis"]["score"], reverse=True):
            c = r["candidate"]
            a = c["analysis"]
            icon = ":white_check_mark:" if a["rating"] == "A+" else ":large_yellow_circle:"
            bwb_lines.append(
                f"{icon} *{r['ticker']} ${r['spot']:.2f}* ({r.get('change_pct',0):+.1f}%)  "
                f"[{a['rating']} {a['score']}/10]\n"
                f"  BUY ${c['long_upper']:.0f}P / SELL 2x ${c['short_strike']:.0f}P / "
                f"BUY ${c['long_lower']:.0f}P  {c['dte']}DTE  "
                f"Credit +${c.get('credit',0):.2f}  R/R {a['rr_ratio']}  "
                f"Max profit +${a['max_profit_usd']}  Max loss -${a['max_loss_usd']}"
                + _earn_note(r['ticker'])
            )

        # -- Assemble message ------------------------------------------------
        _acfg = _alerts_cfg()
        sections = [f":mag: *{session} Scan — {ts}*"]

        if gex_block:
            sections += ["", gex_block]
        if trade_block and _acfg.get("spy_idea", True):
            sections += ["", "*--- SPY TRADE IDEA ---*", trade_block]
        if close_block:
            sections += ["", "*--- CLOSE / ACTION ---*", close_block]
        if pos_block:
            sections += ["", "*--- OPEN POSITIONS ---*", pos_block]
        if fa_hits:
            sections += ["", fa_fmt(fa_results, ts)]
        if vw_hits and _acfg.get("value_alerts", True):
            sections += ["", vw_fmt(vw_results, regime, vix["now"], ts)]
        if bp_place_lines and _acfg.get("bull_put_entries", True):
            sections += ["", ":moneybag: *Bull Put Spreads — place trade*"] + bp_place_lines
        if bwb_lines and _acfg.get("bwb", False):
            sections += ["", ":butterfly: *BWB Watchlist Candidates*"] + bwb_lines
        if _acfg.get("monitor_setups", False):
            fly_block = fly_fmt(fly_result)
            if fly_block:
                sections += ["", fly_block]
            condor_block = condor_fmt(condor_result)
            if condor_block:
                sections += ["", condor_block]
        if _acfg.get("xsp", False):
            batman_block = batman_fmt(batman_result)
            if batman_block:
                sections += ["", batman_block]

        send_message("\n".join(sections))

        # -- Interactive entry approvals (Alpaca-executable structures) -------
        # Focus mode: bull-put/bwb/fly/condor buttons are gated by alerts.json.
        # FA verticals (below) and the SPY Burrito (/cron/burrito) stay active.
        try:
            entry_cands = []
            for r in (bp_hits if _acfg.get("bull_put_entries", True) else []):
                c = r.get("candidate")
                if not c:
                    continue
                legs = [
                    {"action": "SELL", "strike": c["short_strike"], "opt": "P", "qty": 1},
                    {"action": "BUY",  "strike": c["long_strike"],  "opt": "P", "qty": 1},
                ]
                label = f"Bull put {r['ticker']} {c['short_strike']}/{c['long_strike']} {c['expiry']}"
                tid = register_entry("bull_put", r["ticker"], c["expiry"], legs,
                                     qty=1, ref_net=float(c["credit"]), label=label,
                                     text=f"*{label}*  {c['dte']}DTE  credit +${c['credit']:.2f} "
                                          f"({c['credit_pct']:.0f}%)  max loss -${c['max_loss_usd']}"
                                          + _earn_note(r["ticker"]))
                entry_cands.append({"tid": tid,
                                    "text": f"*{label}*  {c['dte']}DTE  credit +${c['credit']:.2f} "
                                            f"({c['credit_pct']:.0f}%)  max loss -${c['max_loss_usd']}"
                                            + _earn_note(r["ticker"]),
                                    "label": label})
            for r in (sorted(bwb_hits, key=lambda x: x["candidate"]["analysis"]["score"], reverse=True)
                      if _acfg.get("bwb", False) else []):
                c = r["candidate"]
                legs = [
                    {"action": "BUY",  "strike": c["long_upper"],   "opt": "P", "qty": 1},
                    {"action": "SELL", "strike": c["short_strike"], "opt": "P", "qty": 2},
                    {"action": "BUY",  "strike": c["long_lower"],   "opt": "P", "qty": 1},
                ]
                label = f"BWB {r['ticker']} {c['long_upper']:.0f}/{c['short_strike']:.0f}/{c['long_lower']:.0f} {c['expiry']}"
                tid = register_entry("bwb", r["ticker"], c["expiry"], legs, qty=1,
                                     ref_net=float(c.get("credit", 0)), label=label,
                                     text=f"*{label}*  {c['dte']}DTE" + _earn_note(r["ticker"]))
                entry_cands.append({"tid": tid, "text": f"*{label}*  {c['dte']}DTE", "label": label})
            _setups_on = _acfg.get("monitor_setups", False)
            for c in (fly_hits[:3] if _setups_on else []):
                label = f"Pin fly {c['ticker']} {c['long_upper']:.0f}/{c['short_body']:.0f}/{c['long_lower']:.0f} {c['expiry']}"
                tid = register_entry("fly", c["ticker"], c["expiry"], c["legs"], qty=1,
                                     ref_net=-float(c["debit"]), label=label,
                                     text=f"*{label}*  {c['dte']}DTE  debit -${c['debit']:.2f}  "
                                          f"R/R {c['rr']}  max profit +${c['max_profit_usd']}")
                entry_cands.append({"tid": tid,
                                    "text": f"*{label}*  debit -${c['debit']:.2f}  R/R {c['rr']}",
                                    "label": label})
            if condor_hit and _setups_on:
                c = condor_hit
                label = (f"Condor {c['ticker']} {c['short_put']:.0f}/{c['long_put']:.0f}P "
                         f"{c['short_call']:.0f}/{c['long_call']:.0f}C {c['expiry']}")
                tid = register_entry("condor", c["ticker"], c["expiry"], c["legs"], qty=1,
                                     ref_net=float(c["credit"]), label=label,
                                     text=f"*{label}*  {c['dte']}DTE  credit +${c['credit']:.2f}  "
                                          f"POP {c.get('pop_pct','?')}%")
                entry_cands.append({"tid": tid,
                                    "text": f"*{label}*  credit +${c['credit']:.2f}  POP {c.get('pop_pct','?')}%",
                                    "label": label})
            _post_entry_approvals(entry_cands)
        except Exception as e:
            logger.warning("Entry-approval post failed: %s", e)

        # -- Fallen-angel SPREAD approvals (managed verticals, not stock buys) --
        # Bull-put credit spread on every name at score >= spread_min_score, plus
        # a bull-call debit spread at score >= call_min_score. HITL: each posts a
        # Take button that re-prices live and submits an MLEG order on approval.
        try:
            facfg = _fa_config()
            if facfg.get("enabled") and facfg.get("fa_spreads_enabled", True):
                min_sc = int(facfg.get("spread_min_score", 6))
                fa_spread_cands = []
                for a in fa_results:
                    if int(a.get("score", 0)) < min_sc or not a.get("spot"):
                        continue
                    try:
                        verts = _fa_spread_candidates(a["ticker"], float(a["spot"]),
                                                      int(a["score"]), facfg)
                    except Exception as e:
                        logger.warning("FA spread build failed for %s: %s", a.get("ticker"), e)
                        continue
                    for v in verts:
                        if v["kind"] == "bull_put":
                            label = f"FA bull-put {a['ticker']} {v['short']:g}/{v['long']:g}P {v['expiry']}"
                            text  = (f"*{label}*  {v['dte']}DTE  credit +${v['ref_net']:.2f}  "
                                     f"max loss -${v['max_loss']}  _(FA score {a['score']}/13)_")
                        else:
                            label = f"FA bull-call {a['ticker']} {v['long']:g}/{v['short']:g}C {v['expiry']}"
                            text  = (f"*{label}*  {v['dte']}DTE  debit -${abs(v['ref_net']):.2f}  "
                                     f"max profit +${v['max_profit']}  _(FA score {a['score']}/13)_")
                        tid = register_entry(v["kind"], a["ticker"], v["expiry"], v["legs"],
                                             qty=1, ref_net=v["ref_net"], label=label, text=text)
                        fa_spread_cands.append({"tid": tid, "text": text, "label": label})
                _post_entry_approvals(fa_spread_cands)
        except Exception as e:
            logger.warning("FA-spread approval post failed: %s", e)

        # -- Value-watchlist Buy approvals (STRONG + earnings-clear) ---------
        try:
            if _fa_config().get("value_buy_enabled", True):
                val_cands = []
                for v in vw_results:
                    if v.get("signal") != "STRONG" or not v.get("earnings_clear", True):
                        continue
                    label = f"{v['ticker']} ${v['spot']:.2f}"
                    text  = (f"*{label}*  score {v.get('score','?')}  RSI {v.get('rsi','?')}  "
                             f"P/E {v.get('pe_forward') or v.get('pe_trailing','?')}")
                    val_cands.append({"tid": f"val-{v['ticker']}", "text": text, "label": label,
                                      "buy_value": f"val_buy:{str(v['ticker']).upper()}:"})
                _post_value_approvals(val_cands)
        except Exception as e:
            logger.warning("Value-approval post failed: %s", e)

        logger.info(
            "Unified scan alert sent — BP:%d FA:%d VW:%d BWB:%d spyBWB:%s fly:%d "
            "condor:%s batman:%s close=%s",
            len(bp_hits), len(fa_hits), len(vw_hits), len(bwb_hits),
            spy_bwb_hit, len(fly_hits), bool(condor_hit), bool(batman_hit), bool(close_block),
        )

    except Exception as e:
        logger.error("_unified_scan_job error: %s", e)


# ── Manage-at-50% (3:30 PM ET) ───────────────────────────────────────────────

def _occ_parse(sym: str):
    """Return (underlying, expiry_date, strike, 'C'|'P') from an OCC symbol, or None."""
    try:
        tp  = len(sym) - 9
        raw = sym[tp - 6:tp]
        exp = date(2000 + int(raw[0:2]), int(raw[2:4]), int(raw[4:6]))
        return sym[:tp - 6], exp, int(sym[-8:]) / 1000.0, sym[tp]
    except Exception:
        return None


def _profit_targets():
    """Group open option legs into structures and compute profit vs. the 50%
    target. Returns a list of flagged structures that have hit it.

    - credit structures (condors, BWBs, credit spreads): target = 50% of credit
    - long butterflies (3 legs, 1/2/1, net debit): target = 50% of max profit
    - other long debit: target = +50% on the debit paid

    Legs are grouped by (underlying, expiry); Alpaca doesn't tag which legs
    belong to which strategy, so multiple structures on the same underlying AND
    expiry are treated as one net book (the net-credit % is still a sane signal).
    """
    from src.live.alpaca_options import _trading
    client    = _trading()
    positions = client.get_all_positions()
    opts = [p for p in positions
            if getattr(p, "asset_class", "") in ("us_option", "option") or len(p.symbol) > 10]

    groups: dict = {}
    for p in opts:
        parsed = _occ_parse(p.symbol)
        if not parsed:
            continue
        underlying, exp, strike, otype = parsed
        leg = {
            "sym":    p.symbol,
            "qty":    float(p.qty),
            "entry":  float(getattr(p, "avg_entry_price", 0) or 0),
            "unreal": float(getattr(p, "unrealized_pl", 0) or 0),
            "strike": strike, "otype": otype,
        }
        groups.setdefault((underlying, exp), []).append(leg)

    flagged = []
    today = date.today()
    for (underlying, exp), legs in groups.items():
        dte    = (exp - today).days
        net_cf = sum(-lg["qty"] * lg["entry"] * 100 for lg in legs)   # + = credit
        unreal = sum(lg["unreal"] for lg in legs)
        n      = len(legs)

        if net_cf > 0:                                   # credit structure
            maxp, kind = net_cf, "credit"
        else:                                            # debit structure
            debit = -net_cf
            maxp, kind = debit, "debit"                  # default: 50% on premium
            if n == 3:                                   # maybe a long butterfly
                sl     = sorted(legs, key=lambda x: x["strike"])
                qtys   = [abs(x["qty"]) for x in sl]
                strks  = [x["strike"] for x in sl]
                even   = abs((strks[1] - strks[0]) - (strks[2] - strks[1])) < 0.01
                ratio  = qtys[1] >= 2 * qtys[0] - 1e-6
                if even and ratio:
                    wing = strks[1] - strks[0]
                    base = qtys[0]
                    mp   = wing * 100 * base - debit
                    if mp > 0:
                        maxp, kind = mp, "fly"

        target = 0.5 * maxp
        if maxp > 0 and unreal >= target:
            flagged.append({
                "underlying": underlying, "exp": str(exp), "dte": dte,
                "unreal": round(unreal), "maxp": round(maxp),
                "pct": round(unreal / maxp * 100), "kind": kind,
                "n_legs": n, "single": legs[0]["sym"] if n == 1 else None,
            })
    return flagged


def _position_adjustments():
    """During-market-hours actionable position flags (so they can actually be
    traded before the close, unlike the post-close EOD report):
      - stock down ≤ -8%   → CLOSE (stop-loss)
      - stock up   ≥ +20%  → TRIM (sell half)
      - option ≤ 3 DTE     → REVIEW (close structure or let expire)
    Returns a list of dicts ready to render as text + an exit button.
    """
    from src.live.alpaca_options import _trading
    positions = _trading().get_all_positions()
    today = date.today()
    out = []
    for p in positions:
        sym    = p.symbol
        unreal = float(getattr(p, "unrealized_pl", 0) or 0)
        pct    = float(getattr(p, "unrealized_plpc", 0) or 0) * 100
        if len(sym) <= 6:                                  # stock / ETF
            if pct <= -8:
                out.append({"underlying": sym, "symbol": sym, "exit_kind": "equity",
                            "action": "CLOSE", "icon": ":rotating_light:", "pct": pct,
                            "btn": "🛑 Close", "reason": f"down {pct:.1f}% — stop-loss zone"})
            elif pct >= 20:
                out.append({"underlying": sym, "symbol": sym, "exit_kind": "equity_trim",
                            "action": "TRIM", "icon": ":moneybag:", "pct": pct,
                            "btn": "✂️ Trim ½", "reason": f"up {pct:.1f}% — take partial profit"})
        else:                                              # option leg
            parsed = _occ_parse(sym)
            if not parsed:
                continue
            underlying, exp, _, _ = parsed
            dte = (exp - today).days
            if dte <= 3:
                out.append({"underlying": underlying, "symbol": None, "exit_kind": "structure",
                            "action": "REVIEW", "icon": ":warning:", "pct": pct,
                            "btn": "💰 Close", "reason": f"{dte} DTE — expires soon"})
    # de-dupe option REVIEWs by underlying (one structure close covers all its legs)
    seen, deduped = set(), []
    for a in out:
        key = (a["underlying"], a["exit_kind"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(a)
    return deduped


def _manage_5050_job():
    """3:30 PM ET Mon-Fri. End-of-day actionable sweep, done while the market is
    still open so flags can be traded before the close:
      - option structures at the 50% profit target (take winners)
      - stock stop-loss / trim and options near expiry
    Each flag carries a Close/Trim button. Silent if nothing qualifies."""
    try:
        if not _market_is_open():
            return
        from src.notifications.slack_notifier import send_message
        flags       = _profit_targets()
        adjustments = _position_adjustments()
        if not flags and not adjustments:
            logger.info("Manage-3:30: nothing to flag")
            return
        ts = datetime.now(ET).strftime("%H:%M ET")
        lines = [f":dart: *3:30 Position Check — {ts}*  _act before the close_"]
        exit_cands = []

        for f in sorted(flags, key=lambda x: -x["pct"]):
            close = (f"`/close_position {f['single']}`" if f["single"]
                     else f"`python close_bwb.py --ticker {f['underlying']}`")
            lines.append(
                f"  :white_check_mark: *{f['underlying']}* {f['exp']} ({f['dte']}DTE, "
                f"{f['kind']})  +${f['unreal']} of ${f['maxp']} max  *{f['pct']}%*\n"
                f"    :point_right: {close}"
            )
            label = f"{f['underlying']} {f['exp']} ({f['kind']})"
            text  = (f"*{label}*  +${f['unreal']} of ${f['maxp']} max  *{f['pct']}%*  "
                     f"({f['dte']}DTE)")
            tid = register_exit("structure", f["underlying"], label, text)
            exit_cands.append({"tid": tid, "text": text, "label": label})

        for a in adjustments:
            lines.append(f"  {a['icon']} *{a['action']} `{a['underlying']}`* — {a['reason']}")
            label = f"{a['action']} {a['underlying']}"
            text  = f"*{a['action']} `{a['underlying']}`* — {a['reason']}"
            tid = register_exit(a["exit_kind"], a["underlying"], label, text, symbol=a["symbol"])
            exit_cands.append({"tid": tid, "text": text, "label": label, "close_label": a["btn"]})

        send_message("\n".join(lines))

        try:
            _post_exit_approvals(exit_cands)
        except Exception as e:
            logger.warning("Exit-approval post failed: %s", e)

        logger.info("Manage-3:30 alert: %d profit-target, %d adjustment(s)",
                    len(flags), len(adjustments))
    except Exception as e:
        logger.error("_manage_5050_job error: %s", e)


# ── EOD Report + Close Suggestions ───────────────────────────────────────────

def _eod_report_job():
    """Runs at 4:05 PM ET Mon-Fri. Pure post-close recap: daily P&L, 30-day P&L,
    and positions. Actionable trims/stops live in the 3:30 PM manage check."""
    try:
        from src.live.alpaca_options import _trading
        from src.notifications.slack_notifier import send_message
        from alpaca.trading.requests import GetPortfolioHistoryRequest
        from datetime import date, timedelta

        client  = _trading()
        acct    = client.get_account()
        equity  = float(acct.equity)
        cash    = float(acct.cash)
        bp      = float(getattr(acct, "options_buying_power", 0) or 0)
        now_str = datetime.now(ET).strftime("%A %b %d, %Y")

        # -- Portfolio history: daily P&L and 30-day P&L ----------------------
        try:
            hist_1m = client.get_portfolio_history(
                GetPortfolioHistoryRequest(period="1M", timeframe="1D")
            )
            eq_series = hist_1m.equity or []
            pl_series = hist_1m.profit_loss or []
            start_eq  = eq_series[0]  if eq_series  else equity
            daily_pl  = pl_series[-1] if pl_series  else 0.0
            full_pl   = equity - start_eq
            daily_pct = (daily_pl / (equity - daily_pl) * 100) if equity != daily_pl else 0
            full_pct  = (full_pl  / start_eq * 100)            if start_eq            else 0
        except Exception as e:
            logger.warning("Portfolio history fetch failed: %s", e)
            daily_pl = full_pl = daily_pct = full_pct = 0.0
            start_eq = equity

        # -- Positions --------------------------------------------------------
        positions = client.get_all_positions()
        stocks  = [p for p in positions if len(p.symbol) <= 6]
        options = [p for p in positions if len(p.symbol) > 6]

        def _fmt_pos(p):
            unreal = float(getattr(p, "unrealized_pl",    0) or 0)
            pct    = float(getattr(p, "unrealized_plpc",  0) or 0) * 100
            mkt    = float(getattr(p, "market_value",     0) or 0)
            icon   = ":green_circle:" if unreal >= 0 else ":red_circle:"
            return f"  {icon} `{p.symbol}`  mkt=${mkt:,.2f}  P&L=${unreal:+,.2f} ({pct:+.1f}%)"

        stock_lines  = "\n".join(_fmt_pos(p) for p in stocks)  or "  _none_"
        option_lines = "\n".join(_fmt_pos(p) for p in options) or "  _none_"

        # -- Build message ----------------------------------------------------
        # Actionable close/trim/stop suggestions are NOT here — they fire at the
        # 3:30 PM manage check (with buttons) while the market is still open, so
        # they can actually be traded. The EOD report is a pure post-close recap.
        d_icon = ":chart_with_upwards_trend:" if daily_pl >= 0 else ":chart_with_downwards_trend:"
        f_icon = ":chart_with_upwards_trend:" if full_pl  >= 0 else ":chart_with_downwards_trend:"

        msg = (
            f":bar_chart: *EOD Report — {now_str}*\n\n"
            f"*Portfolio:* ${equity:,.2f}   "
            f"*Cash:* ${cash:,.2f}   "
            f"*Options BP:* ${bp:,.2f}\n\n"
            f"{d_icon} *Today's P&L:*  ${daily_pl:+,.2f}  ({daily_pct:+.1f}%)\n"
            f"{f_icon} *30-Day P&L:*   ${full_pl:+,.2f}  ({full_pct:+.1f}%)  "
            f"_(start ${start_eq:,.2f})_\n\n"
            f"*Stocks ({len(stocks)}):*\n{stock_lines}\n\n"
            f"*Options ({len(options)}):*\n{option_lines}\n\n"
            f"_Actionable trims/stops fire at the 3:30 PM check (market open)._"
        )
        send_message(msg)
        logger.info("EOD report sent")

    except Exception as e:
        logger.error("_eod_report_job error: %s", e)
        from src.notifications.slack_notifier import send_message as _sm
        _sm(f":rotating_light: EOD report error: {e}")


# ── Interactive trade approvals (Slack buttons) ──────────────────────────────
#
# Scans/manage jobs post an approval message with Take/Skip (entry) or
# Close/Hold (exit) buttons. A tap hits /slack/interactive, which re-prices the
# structure live, confirms it hasn't drifted past tolerance, and only then
# submits. Pending trades live in data/pending_trades.json (survive restarts).

REPRICE_TOLERANCE = 0.20   # reject if net credit/debit drifts >20% against us


def _as_date(v):
    """Coerce a scanner 'expiry' (date or 'YYYY-MM-DD' string) to a date."""
    if isinstance(v, date):
        return v
    return date.fromisoformat(str(v)[:10])


def _legs_to_occ(underlying: str, expiry, legs: list) -> list:
    """Convert scanner legs [{action,strike,opt,qty}] to stored order legs with
    OCC symbols. opt is 'P'/'C' (or PUT/CALL)."""
    from src.live.alpaca_options import occ_symbol
    exp = _as_date(expiry)
    out = []
    for lg in legs:
        opt = str(lg.get("opt", "P")).upper()
        otype = "CALL" if opt.startswith("C") else "PUT"
        out.append({
            "symbol": occ_symbol(underlying, exp, otype, float(lg["strike"])),
            "side":   str(lg["action"]).upper(),       # BUY / SELL
            "ratio":  int(lg.get("qty", 1)),
            "ref_mid": float(lg.get("mid", 0) or 0),
        })
    return out


ENTRY_FILL_FRAC = 0.60   # how far from mid toward the natural (fillable) price;
                         # 0 = mid (often won't fill), 1 = natural/cross (fills now)


def _reprice_net(legs: list, fill_frac: float = ENTRY_FILL_FRAC):
    """Re-fetch live quotes and price each leg `fill_frac` of the way from mid
    toward its NATURAL fill side (SELL→bid, BUY→ask), so the combo limit is
    actually marketable. Returns (net_per_lot, priced_legs); net>0 = credit.
    Falls back to mid if only a mid is available; (None,[]) if unpriceable."""
    from src.live.alpaca_options import get_quote, get_mid_price
    net = 0.0
    priced = []
    for lg in legs:
        bid, ask = get_quote(lg["symbol"])
        if bid is not None and ask is not None and ask >= bid:
            mid = (bid + ask) / 2
            # toward the side that gets us filled: sell lower (toward bid), buy higher (toward ask)
            limit = (mid - (mid - bid) * fill_frac) if lg["side"] == "SELL" \
                    else (mid + (ask - mid) * fill_frac)
            limit = round(limit, 2)
        else:
            mid = get_mid_price(lg["symbol"])
            if mid is None:
                return None, []
            limit = round(mid * (0.97 if lg["side"] == "SELL" else 1.03), 2)
        if lg["side"] == "SELL":
            net += limit * lg["ratio"]
        else:
            net -= limit * lg["ratio"]
        priced.append({**lg, "mid": mid, "limit": limit})
    return round(net, 2), priced


def register_entry(strategy: str, underlying: str, expiry, legs: list,
                   qty: int, ref_net: float, label: str, text: str) -> str:
    """Store a pending ENTRY and return its id. ref_net > 0 = credit per lot."""
    from src.notifications import pending_store
    order_legs = _legs_to_occ(underlying, expiry, legs)
    return pending_store.add({
        "kind": "entry", "strategy": strategy, "underlying": underlying.upper(),
        "label": label, "text": text,
        "order": {"legs": order_legs, "qty": int(qty),
                  "ref_net": round(float(ref_net), 2), "tolerance": REPRICE_TOLERANCE},
    })


def register_exit(kind: str, underlying: str, label: str, text: str,
                  symbol: str | None = None) -> str:
    """Store a pending EXIT. kind='equity' (market-sell `symbol`) or
    kind='structure' (multi-leg close of all option legs for `underlying`)."""
    from src.notifications import pending_store
    return pending_store.add({
        "kind": "exit", "exit_kind": kind, "underlying": underlying.upper(),
        "symbol": symbol, "label": label, "text": text, "order": {},
    })


def _drifted(ref_net: float, live_net: float, tol: float) -> bool:
    """True if the live net moved against us beyond tolerance, or flipped sign."""
    if ref_net >= 0:                                   # expected a credit
        return live_net <= 0 or live_net < ref_net * (1 - tol)
    else:                                              # expected a debit
        return live_net >= 0 or abs(live_net) > abs(ref_net) * (1 + tol)


def _execute_entry(rec: dict, resp_url: str):
    """Re-price + confirm + submit a pending entry, then update the Slack msg."""
    from src.notifications import pending_store
    from src.live.alpaca_options import _trading
    from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, PositionIntent, OrderClass
    tid = rec["id"]
    try:
        allowed, reason = _risk_check_entry(None)
        if not allowed:
            pending_store.update_status(tid, "skipped", note=f"risk: {reason}")
            _slack_replace(resp_url, f":octagonal_sign: *Blocked by risk rule* — {reason}. Not placed.")
            return
        order   = rec["order"]
        qty     = int(order["qty"])
        ref_net = float(order["ref_net"])
        tol     = float(order.get("tolerance", REPRICE_TOLERANCE))

        live_net, priced = _reprice_net(order["legs"])
        if live_net is None:
            pending_store.update_status(tid, "failed", note="no_quotes")
            _slack_replace(resp_url, f":x: *Skipped {rec['label']}* — could not re-price (no live quotes).")
            return
        if _drifted(ref_net, live_net, tol):
            pending_store.update_status(tid, "skipped", note="drift", live_net=live_net)
            kind = "credit" if ref_net >= 0 else "debit"
            _slack_replace(resp_url,
                f":no_entry: *Stood down — {rec['label']}*\n"
                f"  {kind} moved from ${abs(ref_net):.2f} to ${abs(live_net):.2f} "
                f"(past {tol:.0%} tolerance). Not submitted.")
            return

        is_credit = live_net > 0
        leg_objs = [
            OptionLegRequest(
                symbol=lg["symbol"], ratio_qty=lg["ratio"] * qty,
                side=OrderSide.SELL if lg["side"] == "SELL" else OrderSide.BUY,
                position_intent=(PositionIntent.SELL_TO_OPEN if lg["side"] == "SELL"
                                 else PositionIntent.BUY_TO_OPEN),
            )
            for lg in priced
        ]
        client = _trading()
        _strat = rec.get("strategy", "opt")
        _coid  = f"{_strat}-{rec['underlying']}-{datetime.now(ET):%Y%m%d}-{tid[:4]}"
        order_obj = client.submit_order(LimitOrderRequest(
            qty=1, order_class=OrderClass.MLEG,
            side=OrderSide.SELL if is_credit else OrderSide.BUY,
            type=OrderType.LIMIT, time_in_force=TimeInForce.DAY,
            limit_price=round(-live_net, 2), legs=leg_objs,   # signed: credit<0, debit>0
            client_order_id=_coid,
        ))
        # Accepted ≠ filled. Poll the order and report the TRUTH (a resting limit
        # that never fills and cancels at the close looks "placed" but isn't).
        word = "credit" if is_credit else "debit"
        status, filled, note = _confirm_fill(client, order_obj.id)
        if status == "filled":
            pending_store.update_status(tid, "taken", order_id=str(order_obj.id), live_net=live_net)
            _slack_replace(resp_url,
                f":white_check_mark: *FILLED — {rec['label']}*  (qty {qty})\n"
                f"  Net {word} ${abs(live_net):.2f}  _(alert ${abs(ref_net):.2f})_\n"
                f"  Order ID: `{order_obj.id}`")
        elif status in ("rejected", "canceled"):
            pending_store.update_status(tid, "rejected", order_id=str(order_obj.id), note=note)
            _slack_replace(resp_url,
                f":rotating_light: *Not filled — {rec['label']}* ({status})\n"
                f"  {note or 'order did not fill'}\n"
                f"  Order ID: `{order_obj.id}`")
        else:   # still working / unfilled limit
            pending_store.update_status(tid, "working", order_id=str(order_obj.id), live_net=live_net)
            _slack_replace(resp_url,
                f":hourglass_flowing_sand: *Working (not yet filled) — {rec['label']}*\n"
                f"  Limit net {word} ${abs(live_net):.2f}; resting at the broker. "
                f"Fills if the market reaches it, else expires at the close.\n"
                f"  Order ID: `{order_obj.id}`")
    except Exception as e:
        pending_store.update_status(tid, "failed", note=str(e))
        _slack_replace(resp_url, f":rotating_light: *Place failed — {rec['label']}*: {e}")


def _confirm_fill(client, order_id, waits=(2.0, 3.0, 3.0, 4.0, 4.0)):
    """Poll an order to its real outcome. Returns (status, filled_qty, note) where
    status ∈ {filled, rejected, canceled, working}. Accepted-but-unfilled limits
    return 'working' so we never report a false fill."""
    import time as _t
    last = None
    for w in waits:
        _t.sleep(w)
        try:
            o = client.get_order_by_id(order_id)
        except Exception:
            return ("working", 0.0, "")
        st     = str(getattr(o, "status", "")).lower()
        filled = float(getattr(o, "filled_qty", 0) or 0)
        last   = o
        if "filled" in st and filled > 0 and "partial" not in st:
            return ("filled", filled, "")
        if "reject" in st or "cancel" in st:
            reason = ""
            for f in ("rejected_reason", "reject_reason", "failure_reason"):
                v = getattr(o, f, None)
                if v:
                    reason = str(v); break
            if not reason:
                reason = ("no fill at the limit; canceled" if "cancel" in st
                          else "rejected by broker (see Alpaca)")
            return ("canceled" if "cancel" in st else "rejected", filled, reason)
    # window elapsed, still live → working limit (will fill later or expire at close)
    fq = float(getattr(last, "filled_qty", 0) or 0) if last else 0.0
    return ("filled" if fq > 0 else "working", fq, "")


def _execute_exit(rec: dict, resp_url: str):
    """Submit a pending exit (equity market-sell or multi-leg structure close)."""
    from src.notifications import pending_store
    from src.live.alpaca_options import _trading, get_mid_price
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, OptionLegRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, PositionIntent, OrderClass
    tid = rec["id"]
    try:
        client = _trading()
        if rec.get("exit_kind") in ("equity", "equity_trim"):
            sym = rec.get("symbol") or rec["underlying"]
            pos = next((p for p in client.get_all_positions() if p.symbol == sym), None)
            if pos is None:
                pending_store.update_status(tid, "failed", note="no_position")
                _slack_replace(resp_url, f":x: No open position for `{sym}` — nothing to close.")
                return
            unreal   = float(getattr(pos, "unrealized_pl", 0) or 0)
            full_qty = float(pos.qty)
            if rec["exit_kind"] == "equity_trim":
                # sell half; keep whole shares whole, allow fractional for fractionable assets
                sell_qty = float(int(full_qty) // 2) if full_qty == int(full_qty) else round(full_qty / 2, 4)
                if sell_qty <= 0:
                    pending_store.update_status(tid, "failed", note="too_small_to_trim")
                    _slack_replace(resp_url,
                        f":x: `{sym}` is only {full_qty:g} share(s) — too small to trim; use a full close.")
                    return
                verb = f"Trimmed {sell_qty:g} of {full_qty:g}"
            else:
                sell_qty = full_qty
                verb = "Closed"
            order_obj = client.submit_order(MarketOrderRequest(
                symbol=sym, qty=sell_qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY))
            pending_store.update_status(tid, "taken", order_id=str(order_obj.id))
            icon = ":green_circle:" if unreal >= 0 else ":red_circle:"
            _slack_replace(resp_url,
                f"{icon} *{verb} `{sym}`*  P&L ${unreal:+,.2f}\n  Order ID: `{order_obj.id}`")
            return

        # structure: close every open option leg for the underlying as one order
        underlying = rec["underlying"]
        opt_legs = [p for p in client.get_all_positions()
                    if getattr(p, "asset_class", "") in ("us_option", "option")
                    and p.symbol.upper().startswith(underlying)]
        if not opt_legs:
            pending_store.update_status(tid, "failed", note="no_legs")
            _slack_replace(resp_url, f":x: No open option legs for `{underlying}` — nothing to close.")
            return
        net_debit, leg_objs, total_unreal = 0.0, [], 0.0
        for pos in opt_legs:
            mid = get_mid_price(pos.symbol)
            if mid is None:
                continue
            qn    = float(pos.qty)
            ratio = int(abs(qn))
            total_unreal += float(getattr(pos, "unrealized_pl", 0) or 0)
            if qn > 0:
                side, intent, limit = OrderSide.SELL, PositionIntent.SELL_TO_CLOSE, round(mid * 0.95, 2)
                net_debit -= limit * ratio * 100
            else:
                side, intent, limit = OrderSide.BUY, PositionIntent.BUY_TO_CLOSE, round(mid * 1.05, 2)
                net_debit += limit * ratio * 100
            leg_objs.append(OptionLegRequest(symbol=pos.symbol, ratio_qty=ratio,
                                             side=side, position_intent=intent))
        if not leg_objs:
            pending_store.update_status(tid, "failed", note="no_quotes")
            _slack_replace(resp_url, f":x: Could not price any `{underlying}` legs — not submitted.")
            return
        is_debit = net_debit > 0
        net_share = round(abs(net_debit) / 100, 2)
        order_obj = client.submit_order(LimitOrderRequest(
            qty=1, order_class=OrderClass.MLEG,
            side=OrderSide.BUY if is_debit else OrderSide.SELL,
            type=OrderType.LIMIT, time_in_force=TimeInForce.DAY,
            limit_price=(net_share if is_debit else -net_share), legs=leg_objs))
        pending_store.update_status(tid, "taken", order_id=str(order_obj.id))
        icon = ":green_circle:" if total_unreal >= 0 else ":red_circle:"
        _slack_replace(resp_url,
            f"{icon} *Closed structure `{underlying}`*  ({len(leg_objs)} legs)\n"
            f"  Net {'debit' if is_debit else 'credit'} ${net_share:.2f}  |  "
            f"P&L ${total_unreal:+,.2f}\n  Order ID: `{order_obj.id}`")
    except Exception as e:
        pending_store.update_status(tid, "failed", note=str(e))
        _slack_replace(resp_url, f":rotating_light: *Close failed — {rec.get('label','')}*: {e}")


def _slack_replace(url: str, text: str):
    """Replace the original interactive message (drops the buttons)."""
    import requests as _r
    from src.notifications.slack_blocks import resolved_blocks
    try:
        _r.post(url, json={"replace_original": True, "text": text,
                           "blocks": resolved_blocks(text)}, timeout=10)
    except Exception as e:
        logger.error("Slack replace_original failed: %s", e)


def _post_entry_approvals(candidates: list):
    """candidates: [{"tid","text","label"}]. Posts one approval message."""
    if not candidates:
        return
    from src.notifications.slack_blocks import entry_blocks
    from src.notifications.slack_notifier import send_blocks
    header = ":inbox_tray: *Take these trades?*  _re-priced live on tap; auto-skipped if it drifts >20%_"
    blocks, fallback = entry_blocks(header, candidates)
    send_blocks(blocks, fallback)


def _post_exit_approvals(candidates: list):
    if not candidates:
        return
    from src.notifications.slack_blocks import exit_blocks
    from src.notifications.slack_notifier import send_blocks
    header = ":outbox_tray: *Close these?*  _taps submit a live closing order_"
    blocks, fallback = exit_blocks(header, candidates)
    send_blocks(blocks, fallback)


@app.route("/slack/interactive", methods=["POST"])
def slack_interactive():
    if not _verify_slack(request):
        return "", 403
    from src.notifications import pending_store
    from src.notifications.slack_blocks import (ACTION_TAKE, ACTION_SKIP, ACTION_CLOSE,
                                                 ACTION_HOLD, ACTION_BUY)
    try:
        payload = json.loads(request.form.get("payload", "{}"))
    except Exception:
        return "", 200
    action   = (payload.get("actions") or [{}])[0]
    aid      = action.get("action_id", "")
    tid      = action.get("value", "")
    resp_url = payload.get("response_url", "")

    # Restart-proof close: value "close_ticker:SYM" carries the ticker directly,
    # so it works even after the ephemeral pending-store is wiped on a redeploy.
    if aid == ACTION_CLOSE and isinstance(tid, str) and tid.startswith("close_ticker:"):
        sym = tid.split(":", 1)[1].upper()
        threading.Thread(target=lambda: _close_option_structure(sym, resp_url), daemon=True).start()
        return "", 200

    # Restart-proof fallen-angel BUY: value "fa_buy:SYM:LOW52W" carries everything
    # needed, so it still works after the ephemeral pending-store is wiped on a
    # redeploy. _execute_stock_buy re-quotes + re-sizes live and is idempotent
    # (client_order_id fa-<ticker>-<date>), so a reconstructed rec is safe.
    if aid == ACTION_BUY and isinstance(tid, str) and (tid.startswith("fa_buy:") or tid.startswith("val_buy:")):
        strat = "value" if tid.startswith("val_buy:") else "fa"
        parts = tid.split(":")
        sym = parts[1].upper() if len(parts) > 1 else ""
        low = float(parts[2]) if len(parts) > 2 and parts[2] else 0.0
        if sym:
            btn = "valbtn" if strat == "value" else "fabtn"
            rec = {"id": f"{btn}-{sym}-{datetime.now(ET):%Y%m%d}", "kind": "stock_buy",
                   "ticker": sym, "low_52w": low, "entry": None, "label": sym}
            threading.Thread(target=lambda: _execute_stock_buy(rec, resp_url, strat=strat),
                             daemon=True).start()
        return "", 200

    pending_store.purge_expired()
    rec = pending_store.get(tid)
    if rec is None:
        if resp_url:
            _slack_replace(resp_url, ":grey_question: That trade is no longer available.")
        return "", 200

    if aid == ACTION_SKIP:
        pending_store.update_status(tid, "skipped")
        threading.Thread(target=lambda: _slack_replace(resp_url, f":x: *Skipped — {rec['label']}*"),
                         daemon=True).start()
        return "", 200
    if aid == ACTION_HOLD:
        pending_store.update_status(tid, "held")
        threading.Thread(target=lambda: _slack_replace(resp_url, f":pause_button: *Holding — {rec['label']}*"),
                         daemon=True).start()
        return "", 200

    ok, reason = pending_store.is_actionable(tid)
    if not ok:
        nice = {"expired": "expired (alert went stale)", "already_taken": "already placed",
                "already_skipped": "already skipped", "already_held": "already held"}.get(reason, reason)
        threading.Thread(target=lambda: _slack_replace(resp_url, f":hourglass: *{rec['label']}* — {nice}."),
                         daemon=True).start()
        return "", 200

    if aid == ACTION_TAKE:
        threading.Thread(target=lambda: _execute_entry(rec, resp_url), daemon=True).start()
    elif aid == ACTION_CLOSE:
        threading.Thread(target=lambda: _execute_exit(rec, resp_url), daemon=True).start()
    elif aid == ACTION_BUY:
        threading.Thread(target=lambda: _execute_stock_buy(rec, resp_url), daemon=True).start()
    return "", 200


# ── Slack Slash Commands ──────────────────────────────────────────────────────
#
# Register ALL commands in your Slack app pointing to ONE URL:
#   https://<your-render-app>.onrender.com/slack/command
#
# Commands: /close  /place  /scan  /eod  /positions  /help

def _verify_slack(req) -> bool:
    secret = os.environ.get("SLACK_SIGNING_SECRET", "")
    if not secret:
        return True  # dev — skip verification
    ts  = req.headers.get("X-Slack-Request-Timestamp", "0")
    sig = req.headers.get("X-Slack-Signature", "")
    if abs(time.time() - float(ts)) > 300:
        return False
    base     = f"v0:{ts}:{req.get_data(as_text=True)}"
    expected = "v0=" + hmac.new(secret.encode(), base.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig)


def _slack_respond(url: str, text: str):
    """Post a delayed response back to Slack via response_url."""
    import requests as _r
    try:
        _r.post(url, json={"text": text, "response_type": "in_channel"}, timeout=10)
    except Exception as e:
        logger.error("Slack response_url post failed: %s", e)


def _cmd_positions(resp_url: str):
    close_block, pos_block = _build_position_summary()
    sections = [":clipboard: *Open Positions*", pos_block or "  _none_"]
    if close_block:
        sections += ["", "*Suggestions:*", close_block]
    _slack_respond(resp_url, "\n".join(sections))


def _close_option_structure(ticker: str, resp_url: str):
    """Close all open option legs for an underlying as one multi-leg order
    (mirrors close_bwb.py). A closing order frees buying power, so it works even
    when new opens are rejected for insufficient BP."""
    from src.live.alpaca_options import _trading, get_mid_price
    from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, PositionIntent, OrderClass
    try:
        client = _trading()
        # Cancel any working orders on this underlying first — a stale resting
        # close from a prior attempt causes wash-trade/duplicate rejections.
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            for o in client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=200)):
                osym = (getattr(o, "symbol", "") or "").upper()
                legs = getattr(o, "legs", None) or []
                touches = osym == ticker or any((getattr(l, "symbol", "") or "").upper().startswith(ticker)
                                                for l in legs)
                if touches:
                    try:
                        client.cancel_order_by_id(o.id)
                    except Exception:
                        pass
        except Exception as e:
            logger.warning("close: cancel-existing %s failed: %s", ticker, e)

        legs_pos = [p for p in client.get_all_positions()
                    if (getattr(p, "asset_class", "") in ("us_option", "option") or len(p.symbol) > 8)
                    and p.symbol.upper().startswith(ticker)]
        if not legs_pos:
            _slack_respond(resp_url, f":x: No open option legs for `{ticker}`.")
            return
        from src.live.alpaca_options import get_quote
        net_debit, leg_objs, unreal = 0.0, [], 0.0
        CLOSE_FRAC = 0.85   # price aggressively toward the natural side so the exit fills
        for pos in legs_pos:
            bid, ask = get_quote(pos.symbol)
            if bid is not None and ask is not None and ask >= bid:
                mid = (bid + ask) / 2
            else:
                mid = get_mid_price(pos.symbol)
                if mid is None:
                    continue
                bid, ask = mid * 0.9, mid * 1.1
            qn, ratio = float(pos.qty), int(abs(float(pos.qty)))
            unreal += float(getattr(pos, "unrealized_pl", 0) or 0)
            if qn > 0:   # long leg → SELL to close, price toward the bid (fills)
                limit = round(mid - (mid - bid) * CLOSE_FRAC, 2)
                side, intent = OrderSide.SELL, PositionIntent.SELL_TO_CLOSE
                net_debit -= limit * ratio * 100
            else:        # short leg → BUY to close, price toward the ask (fills)
                limit = round(mid + (ask - mid) * CLOSE_FRAC, 2)
                side, intent = OrderSide.BUY, PositionIntent.BUY_TO_CLOSE
                net_debit += limit * ratio * 100
            leg_objs.append(OptionLegRequest(symbol=pos.symbol, ratio_qty=ratio,
                                             side=side, position_intent=intent))
        if not leg_objs:
            _slack_respond(resp_url, f":x: Could not price `{ticker}` legs (market closed?).")
            return
        is_debit  = net_debit > 0
        net_share = round(abs(net_debit) / 100, 2)
        order = client.submit_order(LimitOrderRequest(
            qty=1, order_class=OrderClass.MLEG,
            side=OrderSide.BUY if is_debit else OrderSide.SELL,
            type=OrderType.LIMIT, time_in_force=TimeInForce.DAY,
            limit_price=(net_share if is_debit else -net_share),
            legs=leg_objs, client_order_id=f"close-{ticker}-{datetime.now(ET):%Y%m%d%H%M%S}"))
        status, filled, note = _confirm_fill(client, order.id)
        icon = ":green_circle:" if unreal >= 0 else ":red_circle:"
        if status == "filled":
            _slack_respond(resp_url,
                f"{icon} *Closed `{ticker}` spread* ({len(leg_objs)} legs)\n"
                f"  Net {'debit' if is_debit else 'credit'} ${net_share:.2f}  |  P&L ${unreal:+,.2f}\n"
                f"  Order ID: `{order.id}`")
        elif status in ("rejected", "canceled"):
            _slack_respond(resp_url,
                f":rotating_light: *Close {status} — `{ticker}`*\n  {note or 'broker refused'}\n"
                f"  Order ID: `{order.id}`")
        else:
            _slack_respond(resp_url,
                f":hourglass_flowing_sand: *Close working — `{ticker}`* ({len(leg_objs)} legs)\n"
                f"  Limit net {'debit' if is_debit else 'credit'} ${net_share:.2f} resting; "
                f"fills if the market reaches it.\n  Order ID: `{order.id}`")
    except Exception as e:
        _slack_respond(resp_url, f":rotating_light: Close failed for `{ticker}`: {e}")


def _cmd_close(ticker: str, resp_url: str):
    from src.live.alpaca_options import _trading
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    try:
        client    = _trading()
        positions = client.get_all_positions()
        pos       = next((p for p in positions if p.symbol == ticker.upper()), None)
        if pos is None:
            # no stock position by that name — maybe it's an option structure
            if any(getattr(p, "asset_class", "") in ("us_option", "option") or len(p.symbol) > 8
                   for p in positions if p.symbol.upper().startswith(ticker.upper())):
                _close_option_structure(ticker.upper(), resp_url)
            else:
                _slack_respond(resp_url, f":x: No open position found for `{ticker.upper()}`.")
            return
        qty    = float(pos.qty)
        unreal = float(getattr(pos, "unrealized_pl",   0) or 0)
        pct    = float(getattr(pos, "unrealized_plpc", 0) or 0) * 100
        mkt    = float(getattr(pos, "market_value",    0) or 0)
        req = MarketOrderRequest(
            symbol=ticker.upper(), qty=qty,
            side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
        )
        order = client.submit_order(req)
        icon  = ":green_circle:" if unreal >= 0 else ":red_circle:"
        _slack_respond(resp_url,
            f"{icon} *Closed `{ticker.upper()}`*\n"
            f"  Qty {qty:.4f}  |  Mkt ${mkt:,.2f}  |  P&L ${unreal:+,.2f} ({pct:+.1f}%)\n"
            f"  Order ID: `{order.id}`"
        )
    except Exception as e:
        _slack_respond(resp_url, f":rotating_light: Close failed: {e}")


def _cmd_place(args: str, resp_url: str):
    """args: TICKER SHORT LONG EXPIRY [QTY]"""
    from src.live.alpaca_options import _trading, get_mid_price, occ_symbol
    from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, PositionIntent, OrderClass
    from datetime import date
    try:
        parts = args.split()
        if len(parts) < 4:
            _slack_respond(resp_url,
                ":x: Usage: `/place TICKER SHORT LONG EXPIRY [QTY]`\n"
                "  e.g. `/place KO 82.5 80 2026-07-17`")
            return
        ticker  = parts[0].upper()
        short_k = float(parts[1])
        long_k  = float(parts[2])
        expiry  = date.fromisoformat(parts[3])
        qty     = int(parts[4]) if len(parts) > 4 else 1

        short_sym = occ_symbol(ticker, expiry, "PUT", short_k)
        long_sym  = occ_symbol(ticker, expiry, "PUT", long_k)
        width     = round(short_k - long_k, 2)
        dte       = (expiry - date.today()).days

        short_mid = get_mid_price(short_sym)
        long_mid  = get_mid_price(long_sym)
        if not short_mid or not long_mid:
            _slack_respond(resp_url, f":x: Could not fetch quotes for `{ticker}` options.")
            return

        short_lim  = round(short_mid * 0.97, 2)
        long_lim   = round(long_mid  * 1.03, 2)
        net_credit = round(short_lim - long_lim, 2)
        credit_pct = round(net_credit / width * 100, 1)
        max_profit = round(net_credit * 100 * qty, 2)
        max_loss   = round((width - net_credit) * 100 * qty, 2)

        if net_credit <= 0:
            _slack_respond(resp_url, f":x: Net credit is ${net_credit:.2f} — not a credit spread.")
            return

        acct = _trading().get_account()
        bp   = float(getattr(acct, "options_buying_power", 0) or 0)
        if bp < max_loss:
            _slack_respond(resp_url,
                f":x: Insufficient BP — need ${max_loss:.2f}, have ${bp:.2f}.")
            return

        legs = [
            OptionLegRequest(symbol=short_sym, ratio_qty=qty,
                             side=OrderSide.SELL, position_intent=PositionIntent.SELL_TO_OPEN),
            OptionLegRequest(symbol=long_sym,  ratio_qty=qty,
                             side=OrderSide.BUY,  position_intent=PositionIntent.BUY_TO_OPEN),
        ]
        order = _trading().submit_order(LimitOrderRequest(
            qty=1, order_class=OrderClass.MLEG, side=OrderSide.SELL,
            type=OrderType.LIMIT, time_in_force=TimeInForce.DAY,
            limit_price=round(-net_credit, 2), legs=legs,   # signed: credit<0
        ))
        _slack_respond(resp_url,
            f":white_check_mark: *Placed — Bull Put Spread*\n"
            f"  `{ticker}` SELL ${short_k}P / BUY ${long_k}P  {dte}DTE  Qty {qty}\n"
            f"  Credit +${net_credit:.2f} ({credit_pct:.1f}%)  "
            f"Max profit +${max_profit:.0f}  Max loss -${max_loss:.0f}\n"
            f"  Order ID: `{order.id}`"
        )
    except Exception as e:
        _slack_respond(resp_url, f":rotating_light: Place failed: {e}")


def _cmd_scan(resp_url: str):
    _unified_scan_job()
    # scan job sends its own Slack message; no need to double-post


def _cmd_spy(resp_url: str):
    """On-demand SPY trade signal + rotation check (does not touch monitor state)."""
    d = get_data()
    t = d.get("trade_idea") or {}
    sections = []
    if _spy_recommended(t):
        tier, reasons = _spy_trade_quality(t)
        sections.append(_fmt_spy_signal(t, tier, reasons, changed=False))
    else:
        sections.append(":warning: No SPY trade idea available right now.")
    rot = _stock_rotation_analysis()
    if rot:
        sections += ["", _fmt_rotation(rot)]
    _slack_respond(resp_url, "\n".join(sections))


def _cmd_fly(resp_url: str):
    """On-demand GEX-pinned butterfly scan."""
    try:
        from src.analysis.gex_scanner import scan as gex_scan
        from src.analysis.butterfly_scanner import scan as fly_scan, fmt_slack as fly_fmt
        gx = gex_scan()
        if not gx:
            _slack_respond(resp_url, ":warning: GEX scan unavailable right now.")
            return
        fly = fly_scan(gx.spot, gx, dte_min=0, dte_max=10)
        block = fly_fmt(fly)
        if block:
            _slack_respond(resp_url, block)
        else:
            _slack_respond(resp_url,
                f":no_entry: No pin butterfly — {fly.get('note','')}\n"
                f"  Regime: {gx.gex_regime}  |  gamma wall ${gx.gamma_wall:.0f}  "
                f"call wall ${gx.call_wall:.0f}")
    except Exception as e:
        _slack_respond(resp_url, f":rotating_light: Fly scan failed: {e}")


def _cmd_condor(resp_url: str):
    """On-demand GEX-anchored iron condor. Optional arg: delta (0.10 / 0.16)."""
    try:
        from src.analysis.gex_scanner import scan as gex_scan
        from src.analysis.condor_scanner import scan as condor_scan, fmt_slack as condor_fmt
        gx = gex_scan()
        if not gx:
            _slack_respond(resp_url, ":warning: GEX scan unavailable right now.")
            return
        con = condor_scan(gx.spot, gx, dte_min=0, dte_max=7)
        block = condor_fmt(con)
        if block:
            _slack_respond(resp_url, block)
        else:
            _slack_respond(resp_url,
                f":no_entry: No condor — {con.get('note','')}\n"
                f"  Regime: {gx.gex_regime}  |  put wall ${gx.put_wall:.0f}  "
                f"call wall ${gx.call_wall:.0f}")
    except Exception as e:
        _slack_respond(resp_url, f":rotating_light: Condor scan failed: {e}")


def _cmd_batman(resp_url: str):
    """On-demand GEX-anchored Batman (double broken-wing butterfly) for XSP."""
    try:
        from src.analysis.gex_scanner import scan as gex_scan
        from src.analysis.batman_scanner import scan as batman_scan, fmt_slack as batman_fmt
        gx = gex_scan()
        if not gx:
            _slack_respond(resp_url, ":warning: GEX scan unavailable right now.")
            return
        bat = batman_scan(gx.spot, gx, dte_min=5, dte_max=21)
        block = batman_fmt(bat)
        if block:
            _slack_respond(resp_url, block)
        else:
            _slack_respond(resp_url,
                f":no_entry: No Batman — {bat.get('note','')}\n"
                f"  Regime: {gx.gex_regime}  |  put wall ${gx.put_wall:.0f}  "
                f"call wall ${gx.call_wall:.0f}")
    except Exception as e:
        _slack_respond(resp_url, f":rotating_light: Batman scan failed: {e}")


def _cmd_manage(resp_url: str):
    """On-demand 50% profit check across open structures."""
    try:
        flags = _profit_targets()
        if not flags:
            _slack_respond(resp_url, ":hourglass_flowing_sand: No open structures at the 50% target yet.")
            return
        lines = [":dart: *50% Profit Check*"]
        for f in sorted(flags, key=lambda x: -x["pct"]):
            close = (f"`/close_position {f['single']}`" if f["single"]
                     else f"`python close_bwb.py --ticker {f['underlying']}`")
            lines.append(
                f"  :white_check_mark: *{f['underlying']}* {f['exp']} ({f['dte']}DTE, "
                f"{f['kind']})  +${f['unreal']} of ${f['maxp']} max  *{f['pct']}%*\n"
                f"    :point_right: {close}"
            )
        _slack_respond(resp_url, "\n".join(lines))
    except Exception as e:
        _slack_respond(resp_url, f":rotating_light: Manage check failed: {e}")


def _cmd_eod(resp_url: str):
    _eod_report_job()


def _cmd_strategies(resp_url: str):
    """Render the full strategy registry (the bot's playbook)."""
    try:
        from src.strategies import format_overview
        _slack_respond(resp_url, format_overview())
    except Exception as e:
        _slack_respond(resp_url, f":rotating_light: Strategy registry failed: {e}")


def _spy_idea_line() -> str:
    """One-line confidence + stability read of the current SPY trade idea."""
    try:
        t = get_data().get("trade_idea") or {}
        rec = _spy_recommended(t)
        if not rec:
            return "  SPY idea: _none right now_"
        tier, _ = _spy_trade_quality(t)
        st = _load_spy_state()
        sig = _spy_trade_signature(t)
        stable = f"{int(st.get('stable_n',0))*5}m" if st.get("stable_sig") == sig else "new"
        icon = {"HIGH": ":green_circle:", "MODERATE": ":large_yellow_circle:"}.get(tier, ":red_circle:")
        return (f"  SPY idea: {icon} *{tier}* — {rec.get('type','?')} "
                f"({t.get('dte','?')}DTE) · stable {stable}"
                + ("  ← *decision-worthy*" if tier == "HIGH" and stable != "new" else ""))
    except Exception:
        return "  SPY idea: _n/a_"


def _cmd_risk(resp_url: str):
    """`/risk` — fund risk status: equity, today's P&L vs limit, drawdown, budget."""
    try:
        snap = _risk_snapshot()
        cfg  = _risk_config()
        allowed, reason = _risk_gate(snap, cfg, None)
        budget = cfg["risk_pct_per_trade"] * snap["equity"]
        dl, dd = abs(cfg["daily_loss_limit_pct"]), abs(cfg["max_drawdown_pct"])
        state = ":green_circle: *clear to trade*" if allowed else f":octagonal_sign: *HALTED* — {reason}"
        am = _account_mode()
        mode_icon = ":red_circle:" if am["mode"] == "LIVE" else ":large_blue_circle:"
        _slack_respond(resp_url, "\n".join([
            ":shield: *Risk status*",
            f"  Account: {mode_icon} *{am['mode']}* {am['acct'] or ''}",
            f"  {state}",
            f"  Equity: ${snap['equity']:,.2f}",
            f"  Today: ${snap['day_pl']:+,.2f} ({snap['day_pl_pct']*100:+.1f}%)  _halt at -{dl*100:.0f}%_",
            f"  Drawdown: {snap['drawdown_pct']*100:.1f}% below HWM ${snap['hwm']:,.0f}  _halt at {dd*100:.0f}%_",
            f"  VIX: {snap.get('vix') if snap.get('vix') is not None else '?'}  → size ×"
            f"{_vix_size_factor(snap.get('vix'), cfg):g}  "
            f"_(½ ≥{cfg.get('vix_half_size','?')}, cash ≥{cfg.get('vix_stand_down','?')})_",
            (lambda v: f"  IV vs RV: VIX {v['vix']:.0f} vs RV {v['rv']:.0f} → spread {v['spread']:+.0f}  "
                       f"_{'rich · sell OK' if v['spread'] >= cfg.get('vrp_min_points',0) else 'cheap · no premium-sells'}_"
                       if v.get('spread') is not None else "  IV vs RV: _n/a_")(_vrp_snapshot()),
            f"  Per-trade budget: ${budget:,.0f}  ({cfg['risk_pct_per_trade']*100:.0f}% of equity)",
            _spy_idea_line(),
        ]))
    except Exception as e:
        _slack_respond(resp_url, f":rotating_light: Risk status failed: {e}")


def _cmd_attribution(resp_url: str):
    """`/attribution` — realized P&L by strategy, from Alpaca order tags."""
    try:
        _slack_respond(resp_url, _fmt_attribution(_attribution_report()))
    except Exception as e:
        _slack_respond(resp_url, f":rotating_light: Attribution failed: {e}")


def _cmd_allocation(resp_url: str):
    """`/allocation` — stock / option / cash split vs the caps."""
    try:
        _slack_respond(resp_url, _fmt_allocation(_allocation_snapshot(), _risk_config()))
    except Exception as e:
        _slack_respond(resp_url, f":rotating_light: Allocation failed: {e}")


def _cmd_autotrade(arg: str, resp_url: str):
    """`/autotrade on|off|status` — toggle or inspect the one-shot auto-trade."""
    arg = (arg or "").strip().lower()
    if arg == "off":
        _auto_runtime_set(False)
        _slack_respond(resp_url,
            ":no_entry: *Auto-trade DISABLED.* It will not place a trade. "
            "Re-enable with `/autotrade on`.\n"
            "_Note: this holds while the app stays awake; for a guaranteed overnight "
            "cancel also set `AUTO_TRADE_KILL=1` in Render._")
        return
    if arg == "on":
        _auto_runtime_set(True)
        cfg = _auto_trade_config()
        extra = "  :warning: _but env AUTO_TRADE_KILL is set — that overrides this._" \
            if os.environ.get("AUTO_TRADE_KILL") else ""
        cad = "every market day" if cfg.get("recurring") else f"one-shot {cfg.get('armed_date')}"
        _slack_respond(resp_url,
            f":white_check_mark: *Auto-trade ENABLED.*{extra}\n"
            f"  {cad}, max loss ${cfg.get('max_loss')}, "
            f"10:00–12:30 ET, no 0DTE, exit 50%/1-DTE.")
        return
    # status (default)
    cfg   = _auto_trade_config()
    today = datetime.now(ET).strftime("%Y-%m-%d")
    state = "ENABLED :green_circle:" if cfg.get("enabled") else "DISABLED :red_circle:"
    cadence = ("recurring — every market day" if cfg.get("recurring")
               else ("TODAY" if cfg.get("armed_date") == today
                     else f"one-shot {cfg.get('armed_date')} (not today)"))
    lines = [
        ":robot_face: *Auto-trade status*",
        f"  State: *{state}*" + ("  _(env kill set)_" if os.environ.get("AUTO_TRADE_KILL") else ""),
        f"  Cadence: {cadence}",
        f"  Max loss: ${cfg.get('max_loss')}  |  window 10:00–12:30 ET  |  no 0DTE",
        f"  Exit: auto-close 50% / force-close 1 DTE 3:30 ET",
        "  Toggle: `/autotrade on` · `/autotrade off`",
    ]
    try:
        if _auto_fired_today(today.replace('-', '')):
            lines.append("  :heavy_check_mark: *Already traded today.*")
    except Exception:
        pass
    _slack_respond(resp_url, "\n".join(lines))


HELP_TEXT = (
    ":robot_face: *Trader Bot Commands*\n\n"
    "`/positions`  — show all open positions + close suggestions\n"
    "`/close_position TICKER`  — market-sell a stock/ETF position\n"
    "  _e.g._ `/close_position VBR`\n\n"
    "`/place TICKER SHORT LONG EXPIRY [QTY]`  — place bull put credit spread\n"
    "  _e.g._ `/place KO 82.5 80 2026-07-17`\n\n"
    "`/scan`  — run bull put scanner now + send results\n"
    "`/spy`   — current SPY trade signal + stock-rotation check\n"
    "`/fly`   — GEX-pinned butterfly (positive-gamma pin play)\n"
    "`/condor` — GEX-anchored iron condor (high-POP premium play)\n"
    "`/batman` — GEX-anchored Batman for XSP (positive-cowl double BWB)\n"
    "`/manage` — check open structures at the 50% profit target\n"
    "`/autotrade on|off|status` — toggle / inspect the one-shot auto-trade\n"
    "`/risk`  — fund risk status (equity, day P&L vs limit, drawdown, budget)\n"
    "`/performance` — fund metrics: CAGR, max DD, Sharpe, win rate, profit factor\n"
    "`/attribution` — realized P&L by strategy (which strategy makes money)\n"
    "`/allocation` — stock / option / cash split vs caps\n"
    "`/eod`   — generate EOD P&L report now\n"
    "`/strategies` — the bot's full strategy playbook (registry)\n"
    "`/help`  — show this message"
)


@app.route("/slack/command", methods=["POST"])
def slack_command():
    if not _verify_slack(request):
        return jsonify({"text": ":x: Invalid Slack signature."}), 403

    command  = request.form.get("command", "").lstrip("/").lower()
    text     = request.form.get("text", "").strip()
    resp_url = request.form.get("response_url", "")

    # Normalise — /close_position is the Slack-safe alias for /close
    if command == "close_position":
        command = "close"

    # Acknowledge immediately — Slack requires response within 3 seconds
    ack_map = {
        "positions": ":hourglass: Fetching positions...",
        "close":     f":hourglass: Closing `{text.upper()}`...",
        "place":     f":hourglass: Placing spread `{text}`...",
        "scan":      ":hourglass: Running bull put scan...",
        "spy":       ":hourglass: Checking SPY trade + rotation...",
        "fly":       ":hourglass: Scanning GEX-pinned butterflies...",
        "condor":    ":hourglass: Scanning GEX-anchored condors...",
        "batman":    ":hourglass: Scanning XSP Batman setups...",
        "manage":    ":hourglass: Checking 50% profit targets...",
        "autotrade": ":hourglass: Auto-trade...",
        "risk":      ":hourglass: Checking risk status...",
        "performance": ":hourglass: Computing performance...",
        "perf":      ":hourglass: Computing performance...",
        "attribution": ":hourglass: Computing attribution...",
        "attr":      ":hourglass: Computing attribution...",
        "allocation": ":hourglass: Computing allocation...",
        "alloc":     ":hourglass: Computing allocation...",
        "eod":       ":hourglass: Generating EOD report...",
        "strategies": ":hourglass: Loading strategy registry...",
        "strats":    ":hourglass: Loading strategy registry...",
        "help":      None,
    }

    if command not in ack_map:
        return jsonify({"text": f":x: Unknown command `/{command}`. Try `/help`."}), 200

    if command == "help":
        return jsonify({"text": HELP_TEXT, "response_type": "ephemeral"}), 200

    # Fire background thread, respond with ack immediately
    dispatch = {
        "positions": lambda: _cmd_positions(resp_url),
        "close":     lambda: _cmd_close(text, resp_url),
        "place":     lambda: _cmd_place(text, resp_url),
        "scan":      lambda: _cmd_scan(resp_url),
        "spy":       lambda: _cmd_spy(resp_url),
        "fly":       lambda: _cmd_fly(resp_url),
        "condor":    lambda: _cmd_condor(resp_url),
        "batman":    lambda: _cmd_batman(resp_url),
        "manage":    lambda: _cmd_manage(resp_url),
        "autotrade": lambda: _cmd_autotrade(text, resp_url),
        "risk":      lambda: _cmd_risk(resp_url),
        "performance": lambda: _cmd_performance(resp_url),
        "perf":      lambda: _cmd_performance(resp_url),
        "attribution": lambda: _cmd_attribution(resp_url),
        "attr":      lambda: _cmd_attribution(resp_url),
        "allocation": lambda: _cmd_allocation(resp_url),
        "alloc":     lambda: _cmd_allocation(resp_url),
        "eod":       lambda: _cmd_eod(resp_url),
        "strategies": lambda: _cmd_strategies(resp_url),
        "strats":    lambda: _cmd_strategies(resp_url),
    }
    threading.Thread(target=dispatch[command], daemon=True).start()
    return jsonify({"text": ack_map[command], "response_type": "in_channel"}), 200


# ── Risk engine (fund-style capital protection) ──────────────────────────────
#
# Gates EVERY new entry (auto-trade and button) on portfolio-level circuit
# breakers, and sizes the auto-trade as a % of equity so it scales as the fund
# grows. Equity / today's P&L / high-water mark all come from Alpaca, so the
# numbers are durable and restart-proof.

def _risk_config() -> dict:
    path = os.path.join(os.path.dirname(__file__), "config", "risk.json")
    cfg = {"enabled": True, "risk_pct_per_trade": 0.05,
           "daily_loss_limit_pct": 0.05, "max_drawdown_pct": 0.15}
    try:
        with open(path) as f:
            cfg.update(json.load(f))
    except Exception:
        pass
    return cfg


def _account_mode() -> dict:
    """Whether we're pointed at the LIVE or PAPER Alpaca account, with a masked
    account number for confirmation. Derived from ALPACA_BASE_URL."""
    from config.settings import IS_PAPER, ALPACA_BASE_URL
    out = {"paper": IS_PAPER, "mode": "PAPER" if IS_PAPER else "LIVE",
           "base": ALPACA_BASE_URL, "acct": None}
    try:
        from src.live.alpaca_options import _trading
        num = str(getattr(_trading().get_account(), "account_number", "") or "")
        out["acct"] = ("…" + num[-4:]) if num else None
    except Exception as e:
        logger.warning("account mode fetch failed: %s", e)
    return out


def _risk_snapshot() -> dict:
    """Equity, today's P&L, and drawdown from high-water mark — all from Alpaca."""
    from src.live.alpaca_options import _trading
    from alpaca.trading.requests import GetPortfolioHistoryRequest
    client = _trading()
    acct   = client.get_account()
    equity = float(acct.equity)
    last   = float(getattr(acct, "last_equity", equity) or equity)
    day_pl = equity - last
    hwm    = equity
    try:
        h  = client.get_portfolio_history(GetPortfolioHistoryRequest(period="1A", timeframe="1D"))
        eq = [float(e) for e in (h.equity or []) if e]
        if eq:
            hwm = max(max(eq), equity)
    except Exception as e:
        logger.warning("risk: portfolio history failed: %s", e)
    try:
        vix = (get_data().get("vix") or {}).get("now")
    except Exception:
        vix = None
    cash = float(getattr(acct, "cash", 0) or 0)
    return {
        "equity": equity, "last_equity": last, "day_pl": day_pl,
        "day_pl_pct": (day_pl / last) if last else 0.0,
        "hwm": hwm, "drawdown_pct": ((hwm - equity) / hwm) if hwm else 0.0,
        "vix": vix,
        "cash": cash, "cash_pct": (cash / equity) if equity else 1.0,
    }


_vrp_cache: dict = {"ts": 0.0, "data": None}


def _rv_from_closes(closes: list, lookback: int = 20):
    """Pure: annualized realized vol (%) from daily closes over `lookback` days."""
    import math
    closes = [float(c) for c in closes if c]
    if len(closes) < lookback + 1:
        return None
    seg  = closes[-(lookback + 1):]
    rets = [math.log(seg[i] / seg[i - 1]) for i in range(1, len(seg)) if seg[i - 1]]
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    var  = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return round(math.sqrt(var) * math.sqrt(252) * 100, 1)


def _realized_vol(symbol: str = "SPY", lookback: int = 20):
    try:
        from src.live.yahoo import fetch_daily_ohlcv
        df = fetch_daily_ohlcv(symbol, "3mo")
        return _rv_from_closes(list(df["close"]) if not df.empty else [], lookback)
    except Exception as e:
        logger.warning("realized vol failed: %s", e)
        return None


def _vrp_snapshot() -> dict:
    """Variance risk premium: VIX (implied) vs SPY realized vol. spread>0 = premium
    is rich (sell-favourable). Cached 5 min."""
    now = time.time()
    if _vrp_cache["data"] and now - _vrp_cache["ts"] < 300:
        return _vrp_cache["data"]
    vix = None
    try:
        vix = (get_data().get("vix") or {}).get("now")
    except Exception:
        pass
    rv = _realized_vol("SPY", 20)
    spread = round(vix - rv, 1) if (vix is not None and rv is not None) else None
    out = {"vix": vix, "rv": rv, "spread": spread}
    _vrp_cache.update(ts=now, data=out)
    return out


def _premium_ok():
    """(ok, reason, snapshot) — is implied > realized enough to sell premium?
    Fail-open if vol data is unavailable (don't block on a fetch error)."""
    snap = _vrp_snapshot()
    vmin = _risk_config().get("vrp_min_points", 0)
    if snap["spread"] is None:
        return True, "vrp n/a", snap
    if snap["spread"] < vmin:
        return (False,
                f"IV≤RV — VIX {snap['vix']:.0f} vs RV {snap['rv']:.0f} "
                f"(spread {snap['spread']:+.0f} < {vmin}); premium too cheap to sell",
                snap)
    return True, "ok", snap


def _vix_size_factor(vix, cfg: dict) -> float:
    """VIX-scaled de-risk: full size in calm vol, half in the elevated band,
    stand down (0) in crisis vol. Validated to cut backtest max DD ~2.7×."""
    if vix is None:
        return 1.0
    if cfg.get("vix_stand_down") and vix >= cfg["vix_stand_down"]:
        return 0.0
    if cfg.get("vix_half_size") and vix >= cfg["vix_half_size"]:
        return 0.5
    return 1.0


def _risk_gate(snap: dict, cfg: dict, max_loss: float | None = None) -> tuple[bool, str]:
    """Pure rule check given a snapshot + config. Returns (allowed, reason)."""
    if not cfg.get("enabled", True):
        return True, "risk checks disabled"
    dl = abs(cfg["daily_loss_limit_pct"])
    if snap["day_pl_pct"] <= -dl:
        return False, (f"daily loss limit — down {snap['day_pl_pct']*100:.1f}% today "
                       f"(limit {dl*100:.0f}%)")
    dd = abs(cfg["max_drawdown_pct"])
    if snap["drawdown_pct"] >= dd:
        return False, (f"drawdown guard — {snap['drawdown_pct']*100:.1f}% below high-water "
                       f"mark (limit {dd*100:.0f}%)")
    vix, sd = snap.get("vix"), cfg.get("vix_stand_down")
    if vix is not None and sd and vix >= sd:
        return False, f"VIX {vix:.1f} ≥ {sd:.0f} — crisis-vol stand-down (no new premium)"
    cf = cfg.get("cash_floor_pct")
    if cf and snap.get("cash_pct") is not None and snap["cash_pct"] < cf:
        return False, (f"cash floor — {snap['cash_pct']*100:.0f}% cash < {cf*100:.0f}% min "
                       f"(keep dry powder)")
    if max_loss is not None:
        budget = cfg["risk_pct_per_trade"] * snap["equity"]
        if max_loss > budget + 1e-9:
            return False, (f"trade risk ${max_loss:.0f} exceeds per-trade budget "
                           f"${budget:.0f} ({cfg['risk_pct_per_trade']*100:.0f}% of "
                           f"${snap['equity']:,.0f})")
    return True, "ok"


def _risk_check_entry(max_loss: float | None = None) -> tuple[bool, str]:
    """Convenience: fetch snapshot + config and apply the gate (button path)."""
    try:
        return _risk_gate(_risk_snapshot(), _risk_config(), max_loss)
    except Exception as e:
        logger.warning("risk check failed (allowing): %s", e)
        return True, "risk check unavailable"


def _allocation_snapshot() -> dict:
    """Stock / option / cash split of the book, from Alpaca positions + account."""
    from src.live.alpaca_options import _trading
    client = _trading()
    acct   = client.get_account()
    equity = float(acct.equity)
    cash   = float(getattr(acct, "cash", 0) or 0)
    stock_val = opt_val = 0.0
    for p in client.get_all_positions():
        mv = abs(float(getattr(p, "market_value", 0) or 0))
        if "option" in str(getattr(p, "asset_class", "") or "") or len(p.symbol) > 8:
            opt_val += mv
        else:
            stock_val += mv
    e = equity or 1.0
    return {"equity": equity, "cash": cash, "stock_value": stock_val, "option_value": opt_val,
            "cash_pct": cash / e, "stock_pct": stock_val / e, "option_pct": opt_val / e,
            "deployed_pct": (equity - cash) / e}


def _fmt_allocation(snap: dict, cfg: dict) -> str:
    cf, sc = cfg.get("cash_floor_pct", 0.30), cfg.get("stock_cap_pct", 0.40)
    cash_ok  = ":green_circle:" if snap["cash_pct"]  >= cf else ":red_circle:"
    stock_ok = ":green_circle:" if snap["stock_pct"] <= sc else ":red_circle:"
    return "\n".join([
        ":balance_scale: *Allocation*",
        f"  Equity ${snap['equity']:,.2f}",
        f"  {cash_ok} Cash:    {snap['cash_pct']*100:4.0f}%  _(floor {cf*100:.0f}%)_  ${snap['cash']:,.0f}",
        f"  {stock_ok} Stocks:  {snap['stock_pct']*100:4.0f}%  _(cap {sc*100:.0f}%)_  ${snap['stock_value']:,.0f}",
        f"  :large_blue_circle: Options (mkt): {snap['option_pct']*100:4.0f}%  ${snap['option_value']:,.0f}",
        f"  Deployed {snap['deployed_pct']*100:.0f}%  ·  target ~40 stock / ~25 opt-at-risk / ~35 cash",
    ])


# ── Performance reporting (fund metrics) ─────────────────────────────────────
#
# Fund-grade metrics computed from Alpaca's durable daily equity series — CAGR,
# max drawdown, Sharpe/Sortino, daily win rate / profit factor, period returns.
# No local storage, so it's restart-proof and always reconciles to the broker.

def _perf_from_series(eq: list, ts: list | None = None) -> dict:
    """Pure metrics from a daily equity series (and optional epoch timestamps)."""
    import math
    eq = [float(e) for e in eq if e]
    out = {"points": len(eq)}
    if len(eq) < 2:
        return out
    start, equity = eq[0], eq[-1]
    rets = [(eq[i] - eq[i - 1]) / eq[i - 1] for i in range(1, len(eq)) if eq[i - 1]]

    out["start_equity"] = start
    out["equity"]       = equity
    out["total_return"] = (equity / start - 1) if start > 0 else 0.0

    # CAGR from elapsed calendar time (fallback to trading-day count)
    days = ((ts[-1] - ts[0]) / 86400.0) if (ts and len(ts) >= 2) else float(len(eq))
    out["days"] = days
    yrs = max(days / 365.0, 1e-9)
    out["cagr"] = ((equity / start) ** (1 / yrs) - 1) if start > 0 else 0.0

    # max drawdown
    peak, mdd = -1e18, 0.0
    for v in eq:
        peak = max(peak, v)
        if peak > 0:
            mdd = max(mdd, (peak - v) / peak)
    out["max_dd"] = mdd

    if rets:
        mean = sum(rets) / len(rets)
        sd   = (sum((r - mean) ** 2 for r in rets) / len(rets)) ** 0.5
        downs = [r for r in rets if r < 0]
        dsd  = (sum(r * r for r in downs) / len(downs)) ** 0.5 if downs else 0.0
        green, red = [r for r in rets if r > 0], [r for r in rets if r < 0]
        out["sharpe"]        = (mean / sd * math.sqrt(252)) if sd > 1e-9 else 0.0
        out["sortino"]       = (mean / dsd * math.sqrt(252)) if dsd > 1e-9 else 0.0
        out["day_win_rate"]  = len(green) / len(rets)
        out["profit_factor"] = (sum(green) / abs(sum(red))) if (red and sum(red) != 0) else float("inf")
        out["best_day"]      = max(rets)
        out["worst_day"]     = min(rets)
        # ~30-trading-day return
        k = min(22, len(eq) - 1)
        out["ret_30d"] = (equity / eq[-1 - k] - 1) if eq[-1 - k] else 0.0
    return out


def _performance_metrics() -> dict:
    from src.live.alpaca_options import _trading
    from alpaca.trading.requests import GetPortfolioHistoryRequest
    client = _trading()
    h = client.get_portfolio_history(GetPortfolioHistoryRequest(period="1A", timeframe="1D"))
    return _perf_from_series(list(h.equity or []), list(h.timestamp or []))


def _fmt_performance(m: dict, title: str = "Performance") -> str:
    if m.get("points", 0) < 2:
        return f":chart_with_upwards_trend: *{title}* — not enough history yet."
    def pct(x): return f"{x*100:+.1f}%"
    pf = m.get("profit_factor", 0)
    pf_s = "∞" if pf == float("inf") else f"{pf:.2f}"
    young = "  _(short history — annualized figures are noisy)_" if m.get("days", 0) < 60 else ""
    return "\n".join([
        f":chart_with_upwards_trend: *{title}*  _({int(m['days'])}d, {m['points']} pts)_{young}",
        f"  Equity ${m['equity']:,.2f}   Total {pct(m['total_return'])}   "
        f"30d {pct(m.get('ret_30d',0))}",
        f"  CAGR {pct(m['cagr'])}   Max DD {m['max_dd']*100:.1f}%",
        f"  Sharpe {m.get('sharpe',0):.2f}   Sortino {m.get('sortino',0):.2f}",
        f"  Daily win {m.get('day_win_rate',0)*100:.0f}%   Profit factor {pf_s}   "
        f"best {pct(m.get('best_day',0))} / worst {pct(m.get('worst_day',0))}",
    ])


def _cmd_performance(resp_url: str):
    try:
        _slack_respond(resp_url, _fmt_performance(_performance_metrics()))
    except Exception as e:
        _slack_respond(resp_url, f":rotating_light: Performance report failed: {e}")


def _monthly_nav_job():
    """1st of the month, 8 AM ET — fund NAV statement to Slack."""
    try:
        from src.notifications.slack_notifier import send_message
        m = _performance_metrics()
        send_message(_fmt_performance(m, title=f"Monthly NAV — {datetime.now(ET):%B %Y}"))
        logger.info("Monthly NAV report sent")
    except Exception as e:
        logger.error("_monthly_nav_job error: %s", e)


# ── Per-strategy attribution ─────────────────────────────────────────────────
#
# Which strategy actually makes money. Reconstructed from Alpaca filled orders
# (durable) using the strategy encoded in each entry's client_order_id, netting
# each (underlying, expiry) structure's cash flows. Realized = round-tripped
# structures (legs no longer open); covers trades placed since strategy tagging.

def _coid_strategy(coid: str):
    """Strategy name from an ENTRY client_order_id, or None for closes/unknown."""
    if not coid or coid.startswith(("autoclose", "fastop", "fatrim", "fatrail",
                                    "valstop", "valtrim", "valtrail")):
        return None
    parts = coid.split("-")
    if parts[0] == "auto" and len(parts) >= 3:
        return parts[1]                       # auto-<strategy>-<date>
    if parts[0] == "fa":
        return "fallen_angel"                 # fa-<ticker>-<date>
    if parts[0] == "val":
        return "value"                        # val-<ticker>-<date>
    if parts[0] == "dca":
        return "income_core"                  # dca-<ticker>-<isoweek>
    if parts[0] in ("condor", "bull_put", "bull_call", "bwb", "fly", "burrito"):
        return parts[0]
    return None


def _order_cash_flow(o) -> float:
    """Signed net dollars of a filled order: SELL +, BUY −. Options ×100. Sums
    legs for multi-leg orders. Over a round-trip, the net = realized P&L."""
    legs  = getattr(o, "legs", None)
    items = list(legs) if legs else [o]
    total = 0.0
    for it in items:
        price = float(getattr(it, "filled_avg_price", 0) or 0)
        qty   = float(getattr(it, "filled_qty", 0) or 0)
        if not price or not qty:
            continue
        sym  = getattr(it, "symbol", "") or ""
        mult = 100 if len(sym) > 8 else 1     # OCC option symbols are long
        sign = 1 if "sell" in str(getattr(it, "side", "")).lower() else -1
        total += sign * price * qty * mult
    return round(total, 2)


def _order_key(o):
    """(underlying, expiry) for an options order, or (symbol, 'stock') for equity."""
    legs = getattr(o, "legs", None)
    sym  = (getattr(legs[0], "symbol", None) if legs else None) or getattr(o, "symbol", "") or ""
    if len(sym) > 8:
        pr = _occ_parse(sym)
        if pr:
            return (pr[0], str(pr[1]))
    return (sym, "stock")


def _attribution(norm: list, open_keys: set) -> dict:
    """Pure aggregator. norm: [{key,strategy,flow}]. Groups by structure key,
    attributes by the entry's strategy, sums realized P&L for closed structures."""
    groups: dict = {}
    for o in norm:
        g = groups.setdefault(o["key"], {"strategy": None, "flow": 0.0})
        g["flow"] += o["flow"]
        if o.get("strategy"):
            g["strategy"] = o["strategy"]
    out: dict = {}
    for key, g in groups.items():
        s = out.setdefault(g["strategy"] or "untagged",
                           {"realized": 0.0, "trades": 0, "wins": 0, "open": 0})
        if key in open_keys:
            s["open"] += 1
        else:
            s["realized"] = round(s["realized"] + g["flow"], 2)
            s["trades"]  += 1
            if g["flow"] > 0:
                s["wins"] += 1
    return out


def _attribution_report(days: int = 120) -> dict:
    from src.live.alpaca_options import _trading
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    client = _trading()
    orders = client.get_orders(filter=GetOrdersRequest(
        status=QueryOrderStatus.ALL, limit=500,
        after=datetime.now(ET) - timedelta(days=days)))
    open_keys = set()
    for p in client.get_all_positions():
        sym = p.symbol
        pr  = _occ_parse(sym) if len(sym) > 8 else None
        open_keys.add((pr[0], str(pr[1])) if pr else (sym, "stock"))
    norm = []
    for o in orders:
        if str(getattr(o, "status", "")).lower() != "filled":
            continue
        flow = _order_cash_flow(o)
        if flow == 0:
            continue
        norm.append({"key": _order_key(o),
                     "strategy": _coid_strategy(getattr(o, "client_order_id", "") or ""),
                     "flow": flow})
    return _attribution(norm, open_keys)


def _fmt_attribution(rep: dict) -> str:
    if not rep:
        return ":bar_chart: *Strategy attribution* — no trades yet (tagging starts now)."
    lines = [":bar_chart: *Strategy attribution*  _realized P&L by strategy (closed round-trips)_"]
    for strat, s in sorted(rep.items(), key=lambda kv: -kv[1]["realized"]):
        wr   = (s["wins"] / s["trades"] * 100) if s["trades"] else 0
        icon = ":green_circle:" if s["realized"] >= 0 else ":red_circle:"
        lines.append(f"  {icon} *{strat}*  ${s['realized']:+,.0f}  "
                     f"({s['trades']} closed · {wr:.0f}% win · {s['open']} open)")
    lines.append("  _realized = net of round-tripped structures; tagged trades only_")
    return "\n".join(lines)


# ── Fallen-angel stock rules ─────────────────────────────────────────────────
#
# STRONG fallen-angel signals get a Slack Buy button. A tap risk-sizes the
# position (1% of equity at the stop), buys, and places a protective sell-stop
# at the broker — stop = the higher of -12% or just below the 52-week low, so a
# bought name is never unprotected. Concentration caps: max positions + total
# category exposure ≤ the stock sleeve (~40%).

def _fa_config() -> dict:
    path = os.path.join(os.path.dirname(__file__), "config", "fallen_angel.json")
    cfg = {"enabled": True, "conviction": "STRONG", "risk_pct_per_position": 0.01,
           "stop_drawdown_pct": 0.12, "stop_below_52w_low_pct": 0.01,
           "max_position_pct": 0.15, "max_positions": 4, "max_category_pct": 0.40,
           "trim_at_gain_pct": 0.20, "trail_pct": 0.15,
           # Fallen-angel names are proposed as MANAGED verticals, not stock buys:
           # a bull-put credit spread on every triggered name, plus a bull-call
           # debit spread when score >= call_min_score (the backtested differentiator).
           "fa_spreads_enabled": True, "spread_min_score": 6, "call_min_score": 6,
           "put_short_otm": 0.05, "put_width_otm": 0.07,
           "call_long_otm": 0.0, "call_short_otm": 0.08,
           "spread_dte_min": 20, "spread_dte_max": 45, "spread_target_dte": 30}
    try:
        with open(path) as f:
            cfg.update(json.load(f))
    except Exception:
        pass
    return cfg


def _build_fa_verticals(spot: float, put_mids: dict, call_mids: dict,
                        score: int, cfg: dict) -> list:
    """Pure: from one expiry's {strike: mid} for puts and calls, build the FA
    verticals. Bull-put credit spread on every name; bull-call debit spread when
    score >= call_min_score. ref_net > 0 = credit, < 0 = debit (register_entry's
    convention). Returns [] if strikes/quotes don't support a spread."""
    def nearest(strikes, target):
        return min(strikes, key=lambda k: abs(k - target)) if strikes else None

    out = []
    # Bull-put credit spread: short ~put_short_otm below spot, long put_width_otm lower.
    ps = sorted(put_mids)
    Ks = nearest([k for k in ps if k <= spot], spot * (1 - cfg["put_short_otm"]))
    if Ks is not None:
        Kl = nearest([k for k in ps if k < Ks],
                     spot * (1 - cfg["put_short_otm"] - cfg["put_width_otm"]))
        if Kl is not None:
            credit = round(put_mids[Ks] - put_mids[Kl], 2)
            width  = round(Ks - Kl, 2)
            if credit > 0 and width > 0:
                out.append({"kind": "bull_put", "short": Ks, "long": Kl, "width": width,
                            "ref_net": credit, "max_loss": round(width * 100 - credit * 100),
                            "legs": [{"action": "SELL", "strike": Ks, "opt": "P", "qty": 1},
                                     {"action": "BUY",  "strike": Kl, "opt": "P", "qty": 1}]})

    # Bull-call debit spread (score-gated): long ~ATM, short call_short_otm above.
    cs = sorted(call_mids)
    if score >= cfg["call_min_score"]:
        Kl = nearest(cs, spot * (1 + cfg["call_long_otm"]))
        if Kl is not None:
            Ku = nearest([k for k in cs if k > Kl], spot * (1 + cfg["call_short_otm"]))
            if Ku is not None:
                debit = round(call_mids[Kl] - call_mids[Ku], 2)
                width = round(Ku - Kl, 2)
                if debit > 0 and width > 0:
                    out.append({"kind": "bull_call", "long": Kl, "short": Ku, "width": width,
                                "ref_net": -debit, "max_profit": round(width * 100 - debit * 100),
                                "legs": [{"action": "BUY",  "strike": Kl, "opt": "C", "qty": 1},
                                         {"action": "SELL", "strike": Ku, "opt": "C", "qty": 1}]})
    return out


def _fa_spread_candidates(ticker: str, spot: float, score: int, cfg: dict) -> list:
    """Live chain → FA vertical candidates (with expiry/dte attached)."""
    from src.live.alpaca_options import fetch_chain_combined
    chain = fetch_chain_combined(ticker, spot,
                                 dte_min=cfg["spread_dte_min"], dte_max=cfg["spread_dte_max"])
    puts  = chain.get("puts_liquid", [])
    calls = chain.get("calls_liquid", [])
    if not puts:
        return []
    exp_dte = {p["expiry"]: p["dte"] for p in puts}
    want_call = score >= cfg["call_min_score"]
    exps = ({p["expiry"] for p in puts} & {c["expiry"] for c in calls}) if want_call else set(exp_dte)
    exps = exps or set(exp_dte)
    exp = min(exps, key=lambda e: abs(exp_dte.get(e, 999) - cfg["spread_target_dte"]))

    def mids(rows):
        return {r["strike"]: round((r["bid"] + r["ask"]) / 2, 3)
                for r in rows if r["expiry"] == exp and r.get("bid") and r.get("ask")}

    verts = _build_fa_verticals(spot, mids(puts), mids(calls), score, cfg)
    for v in verts:
        v["expiry"] = exp
        v["dte"] = exp_dte.get(exp)
    return verts


def _fa_stop_and_size(entry: float, low_52w: float, equity: float, cfg: dict) -> dict:
    """Pure: stop price (higher of -drawdown% and just below 52w low) and
    risk-based share count, capped by max_position_pct of equity."""
    stop = max(entry * (1 - cfg["stop_drawdown_pct"]),
               (low_52w or 0) * (1 - cfg["stop_below_52w_low_pct"]))
    rps  = entry - stop
    out  = {"stop": round(stop, 2), "risk_per_share": round(rps, 4),
            "shares": 0, "notional": 0.0, "risk": 0.0, "capped_by": None}
    if rps <= 0 or entry <= 0 or equity <= 0:
        out["capped_by"] = "bad_stop"
        return out
    shares = int((cfg["risk_pct_per_position"] * equity) // rps)
    capped = "risk"
    max_by_notional = int((cfg["max_position_pct"] * equity) // entry)
    if max_by_notional < shares:
        shares, capped = max_by_notional, "position_pct"
    out["shares"]   = max(shares, 0)
    out["notional"] = round(out["shares"] * entry, 2)
    out["risk"]     = round(out["shares"] * rps, 2)
    out["capped_by"] = capped if out["shares"] > 0 else "too_small"
    return out


def _stock_price(ticker: str):
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestTradeRequest
        from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY
        c = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        r = c.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=ticker.upper()))
        return float(r[ticker.upper()].price)
    except Exception as e:
        logger.warning("stock price fetch failed %s: %s", ticker, e)
        return None


def _fa_open_tickers(prefix: str = "fa-") -> set:
    """Tickers currently held that were bought under a given entry-tag prefix
    ('fa-' fallen angels, 'val-' value picks), reconciled from Alpaca so it's
    restart-proof. The '-' in the prefix excludes stop/trim/trail tags."""
    from src.live.alpaca_options import _trading
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    client = _trading()
    try:
        orders = client.get_orders(filter=GetOrdersRequest(
            status=QueryOrderStatus.ALL, limit=300,
            after=datetime.now(ET) - timedelta(days=120)))
    except Exception as e:
        logger.warning("open-tickers get_orders failed (%s): %s", prefix, e)
        return set()
    open_syms = {p.symbol for p in client.get_all_positions()}
    out = set()
    for o in orders:
        coid = getattr(o, "client_order_id", "") or ""
        if coid.startswith(prefix):         # entry tag only (stop/trim/trail excluded)
            sym = getattr(o, "symbol", None)
            if sym in open_syms:
                out.add(sym)
    return out


def register_fa_buy(ticker: str, entry: float, low_52w: float, label: str, text: str) -> str:
    from src.notifications import pending_store
    return pending_store.add({"kind": "stock_buy", "ticker": ticker.upper(),
                              "entry": entry, "low_52w": low_52w,
                              "label": label, "text": text})


def _post_fa_approvals(cands: list):
    if not cands:
        return
    from src.notifications.slack_blocks import buy_blocks
    from src.notifications.slack_notifier import send_blocks
    header = ":parachute: *Buy these fallen angels?*  _risk-sized, protective stop set on entry_"
    blocks, fb = buy_blocks(header, cands)
    send_blocks(blocks, fb)


def _post_value_approvals(cands: list):
    if not cands:
        return
    from src.notifications.slack_blocks import buy_blocks
    from src.notifications.slack_notifier import send_blocks
    header = ":moneybag: *Buy these value picks?*  _risk-sized, protective stop set on entry_"
    blocks, fb = buy_blocks(header, cands)
    send_blocks(blocks, fb)


def _execute_stock_buy(rec: dict, resp_url: str, strat: str = "fa"):
    """Risk-size + buy a stock pick, then place a broker-side stop. strat='fa'
    (fallen angel) or 'value' (value watchlist) — only the entry tag, position
    bucket, and wording differ; sizing/caps/stop logic are shared."""
    from src.notifications import pending_store
    from src.live.alpaca_options import _trading
    from alpaca.trading.requests import MarketOrderRequest, StopOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    pfx   = "fa" if strat == "fa" else "val"
    kind  = "fallen angel" if strat == "fa" else "value pick"
    emoji = ":parachute:" if strat == "fa" else ":moneybag:"
    tid = rec["id"]
    ticker = rec["ticker"]
    try:
        cfg = _fa_config()
        allowed, reason = _risk_check_entry(None)
        if not allowed:
            pending_store.update_status(tid, "skipped", note=f"risk:{reason}")
            _slack_replace(resp_url, f":octagonal_sign: *Blocked by risk rule* — {reason}. Not bought.")
            return
        client = _trading()
        acct   = client.get_account()
        equity = float(acct.equity)
        cash   = float(getattr(acct, "cash", 0) or 0)

        positions = {p.symbol: p for p in client.get_all_positions()}
        if ticker in positions:              # never double-buy a name held in ANY bucket
            pending_store.update_status(tid, "skipped", note="already_held")
            _slack_replace(resp_url, f":information_source: Already hold `{ticker}` — not adding.")
            return
        open_bucket = _fa_open_tickers(pfx + "-")
        if len(open_bucket) >= cfg["max_positions"]:
            pending_store.update_status(tid, "skipped", note="max_positions")
            _slack_replace(resp_url, f":no_entry: Max {kind} positions "
                                     f"({cfg['max_positions']}) reached.")
            return

        fa_val = sum(float(getattr(positions[s], "market_value", 0) or 0)
                     for s in open_bucket if s in positions)

        entry = _stock_price(ticker) or rec.get("entry")
        if not entry:
            pending_store.update_status(tid, "failed", note="no_price")
            _slack_replace(resp_url, f":x: Could not get a price for `{ticker}`.")
            return

        sz = _fa_stop_and_size(entry, rec.get("low_52w") or 0, equity, cfg)
        shares = sz["shares"]
        # category cap
        room = max(cfg["max_category_pct"] * equity - fa_val, 0)
        if shares * entry > room:
            shares = int(room // entry); sz["capped_by"] = "category_pct"
        # total stock-sleeve cap (allocation rule, all equity positions)
        stock_cap = _risk_config().get("stock_cap_pct", 0.40)
        total_stock = sum(abs(float(getattr(p, "market_value", 0) or 0))
                          for s, p in positions.items() if len(s) <= 8)
        room_stk = max(stock_cap * equity - total_stock, 0)
        if shares * entry > room_stk:
            shares = int(room_stk // entry); sz["capped_by"] = "stock_sleeve"
        # cash cap
        if shares * entry > cash:
            shares = int(cash // entry); sz["capped_by"] = "cash"
        if shares < 1:
            pending_store.update_status(tid, "skipped", note=f"too_small:{sz['capped_by']}")
            _slack_replace(resp_url, f":x: *{ticker}* position too small after caps "
                                     f"({sz['capped_by']}). Not bought.")
            return

        today = datetime.now(ET).strftime("%Y%m%d")
        buy = client.submit_order(MarketOrderRequest(
            symbol=ticker, qty=shares, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
            client_order_id=f"{pfx}-{ticker}-{today}"))
        stop_ok = True
        try:
            client.submit_order(StopOrderRequest(
                symbol=ticker, qty=shares, side=OrderSide.SELL, time_in_force=TimeInForce.GTC,
                stop_price=sz["stop"], client_order_id=f"{pfx}stop-{ticker}-{today}"))
        except Exception as e:
            stop_ok = False
            logger.warning("%s stop placement failed %s: %s", pfx, ticker, e)

        pending_store.update_status(tid, "taken", order_id=str(buy.id))
        stop_pct = (entry - sz["stop"]) / entry * 100
        _slack_replace(resp_url,
            f"{emoji} *BOUGHT `{ticker}`* — {kind}\n"
            f"  {shares} sh @ ~${entry:.2f}  (${shares*entry:,.0f})\n"
            f"  Stop ${sz['stop']:.2f} (-{stop_pct:.1f}%)  ·  risk ${sz['risk']:.0f} "
            f"({cfg['risk_pct_per_position']*100:.0f}% of equity)"
            f"{'' if stop_ok else '  :warning: STOP NOT PLACED — set it manually!'}\n"
            f"  Order ID: `{buy.id}`")
        logger.info("%s buy %s %dsh @%.2f stop %.2f (stop_ok=%s)", pfx, ticker, shares, entry, sz["stop"], stop_ok)
    except Exception as e:
        pending_store.update_status(tid, "failed", note=str(e))
        _slack_replace(resp_url, f":rotating_light: *Buy failed — {ticker}*: {e}")


def _fa_trim_plan(qty) -> dict:
    """Pure: how to split a fallen-angel winner. ≥2 shares → sell half + trail the
    rest; 1 share → can't trim, just trail the whole thing."""
    qty  = int(qty)
    half = qty // 2
    if qty < 2:
        return {"mode": "trail_only", "sell": 0, "keep": qty}
    return {"mode": "trim_trail", "sell": half, "keep": qty - half}


def _fa_already_trimmed(ticker: str) -> bool:
    """True once a trim/trail has been placed for this ticker (idempotent)."""
    from src.live.alpaca_options import _trading
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    try:
        orders = _trading().get_orders(filter=GetOrdersRequest(
            status=QueryOrderStatus.ALL, limit=200,
            after=datetime.now(ET) - timedelta(days=10)))
        for o in orders:
            c = getattr(o, "client_order_id", "") or ""
            if c.startswith((f"fatrim-{ticker}", f"fatrail-{ticker}")):
                return True
    except Exception:
        pass
    return False


def _fa_cancel_stops(client, ticker: str):
    """Cancel the fixed protective stop(s) for a ticker to free the shares."""
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    try:
        for o in client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=200)):
            if getattr(o, "symbol", "") != ticker:
                continue
            ot   = str(getattr(o, "order_type", "") or getattr(o, "type", "")).lower()
            coid = getattr(o, "client_order_id", "") or ""
            if "stop" in ot or coid.startswith("fastop"):
                try:
                    client.cancel_order_by_id(o.id)
                except Exception as e:
                    logger.warning("FA cancel stop %s failed: %s", ticker, e)
    except Exception as e:
        logger.warning("FA cancel-stops lookup %s failed: %s", ticker, e)


def _fa_trim_and_trail(ticker: str, cfg: dict):
    """At +trim_at_gain_pct: re-protect with a trailing stop, THEN sell half — so
    the kept shares are never unprotected. Order matters for safety."""
    from src.live.alpaca_options import _trading
    from alpaca.trading.requests import MarketOrderRequest, TrailingStopOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from src.notifications.slack_notifier import send_message
    client = _trading()
    pos = next((p for p in client.get_all_positions() if p.symbol == ticker), None)
    if pos is None:
        return
    qty   = int(float(pos.qty))
    plan  = _fa_trim_plan(qty)
    trailp = float(cfg["trail_pct"]) * 100
    today  = datetime.now(ET).strftime("%Y%m%d")
    unreal = float(getattr(pos, "unrealized_pl", 0) or 0)

    # 1. release shares held by the fixed stop
    _fa_cancel_stops(client, ticker)
    try:
        # 2. re-establish protection on the KEEP shares FIRST (trailing stop)
        client.submit_order(TrailingStopOrderRequest(
            symbol=ticker, qty=plan["keep"], side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC, trail_percent=round(trailp, 2),
            client_order_id=f"fatrail-{ticker}-{today}"))
    except Exception as e:
        # recovery: protection failed — restore a trailing stop on the FULL position
        logger.error("FA trail placement failed %s: %s", ticker, e)
        try:
            client.submit_order(TrailingStopOrderRequest(
                symbol=ticker, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC,
                trail_percent=round(trailp, 2), client_order_id=f"fatrail-{ticker}-{today}r"))
            send_message(f":warning: *{ticker}* trim aborted — placed a {trailp:.0f}% trailing stop "
                         f"on the full position instead. No shares sold.")
        except Exception as e2:
            send_message(f":rotating_light: *{ticker}* trim FAILED and could not re-protect ({e2}). "
                         f"*Set a stop manually now.*")
        return

    if plan["mode"] == "trail_only":
        send_message(f":chart_with_upwards_trend: *{ticker}* up ≥{cfg['trim_at_gain_pct']*100:.0f}% — "
                     f"only {qty} share, can't trim; placed a {trailp:.0f}% trailing stop. "
                     f"P&L ${unreal:+,.2f}")
        return

    # 3. sell half at market (kept shares already protected by the trail)
    try:
        client.submit_order(MarketOrderRequest(
            symbol=ticker, qty=plan["sell"], side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
            client_order_id=f"fatrim-{ticker}-{today}"))
        send_message(f":scissors: *{ticker} TRIMMED* — sold {plan['sell']}/{qty} at +"
                     f"{cfg['trim_at_gain_pct']*100:.0f}%, {plan['keep']} left on a {trailp:.0f}% "
                     f"trailing stop.  P&L ${unreal:+,.2f}")
    except Exception as e:
        send_message(f":warning: *{ticker}* trail placed on {plan['keep']} sh, but the trim-sell "
                     f"failed ({e}). You still hold {qty} (half untrailed) — verify.")


def _fa_manage_job():
    """Every 15 min RTH. Fallen-angel exit: at +20% trim half + trail the rest
    (15%). The −12%/52w-low stop set at entry covers the downside until then."""
    try:
        if not _market_is_open():
            return
        cfg = _fa_config()
        if not cfg.get("enabled"):
            return
        from src.live.alpaca_options import _trading
        positions = {p.symbol: p for p in _trading().get_all_positions()}
        for ticker in _fa_open_tickers():
            try:
                pos = positions.get(ticker)
                if not pos:
                    continue
                entry = float(getattr(pos, "avg_entry_price", 0) or 0)
                cur   = float(getattr(pos, "current_price", 0) or 0) or (_stock_price(ticker) or 0)
                if not entry or not cur:
                    continue
                if (cur - entry) / entry >= cfg["trim_at_gain_pct"] and not _fa_already_trimmed(ticker):
                    _fa_trim_and_trail(ticker, cfg)
            except Exception as e:
                logger.warning("FA manage %s failed: %s", ticker, e)
    except Exception as e:
        logger.error("_fa_manage_job error: %s", e)


# ── 0DTE put-credit-spread at the expected move (opt-in) ─────────────────────
#
# tastylive-style: sell a SPY 0DTE put spread with the short strike ~1 expected-
# move below the forward, width-wide, manage at 50% / DON'T close losers early /
# force-close before the bell (SPY is physically settled — never let a 0DTE short
# put expire ITM on a small account). OFF by default (config/zerodte.json).

def _zerodte_config() -> dict:
    path = os.path.join(os.path.dirname(__file__), "config", "zerodte.json")
    cfg = {"enabled": False, "underlying": "SPY", "width": 1, "fire_start": "10:00",
           "fire_end": "11:30", "profit_target": 0.50, "force_close": "15:45",
           "min_credit_frac": 0.12}
    try:
        with open(path) as f:
            cfg.update(json.load(f))
    except Exception:
        pass
    return cfg


def _parse_hhmm(s: str):
    h, m = str(s).split(":")
    return dt_time(int(h), int(m))


def _zerodte_strikes(forward: float, em_dollars: float, width: float):
    """Pure: short put ~1 expected-move below the forward (nearest $1 strike),
    long put `width` below it."""
    short_put = float(round(forward - em_dollars))
    return short_put, short_put - width


def _zerodte_place(short_k, long_k, coid, budget, cfg) -> dict:
    """Submit the SPY 0DTE put spread; respects the per-trade budget + min credit."""
    from src.live.alpaca_options import _trading, get_mid_price, occ_symbol
    from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, PositionIntent, OrderClass
    try:
        today = date.today()
        ssym = occ_symbol("SPY", today, "PUT", short_k)
        lsym = occ_symbol("SPY", today, "PUT", long_k)
        sm, lm = get_mid_price(ssym), get_mid_price(lsym)
        if not sm or not lm:
            return {"ok": False, "reason": "no quotes"}
        credit = round(sm * 0.97 - lm * 1.03, 2)
        width  = short_k - long_k
        if credit < cfg.get("min_credit_frac", 0.12) * width:
            return {"ok": False, "reason": f"thin credit ${credit:.2f} on ${width:.0f} wide"}
        max_loss = round((width - credit) * 100, 2)
        if max_loss > budget:
            return {"ok": False, "reason": f"max loss ${max_loss:.0f} > budget ${budget:.0f}"}
        legs = [
            OptionLegRequest(symbol=ssym, ratio_qty=1, side=OrderSide.SELL,
                             position_intent=PositionIntent.SELL_TO_OPEN),
            OptionLegRequest(symbol=lsym, ratio_qty=1, side=OrderSide.BUY,
                             position_intent=PositionIntent.BUY_TO_OPEN),
        ]
        o = _trading().submit_order(LimitOrderRequest(
            qty=1, order_class=OrderClass.MLEG, side=OrderSide.SELL, type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY, limit_price=round(-credit, 2), legs=legs, client_order_id=coid))  # signed: credit<0
        return {"ok": True, "order_id": str(o.id), "credit": credit, "max_loss": max_loss}
    except Exception as e:
        return {"ok": False, "reason": f"submit error: {e}"}


def _zerodte_job():
    """Every 5 min in the entry window. Places one SPY 0DTE put spread/day at the
    expected move, gated by the risk engine + IV>RV. OFF unless config enabled."""
    try:
        cfg = _zerodte_config()
        if not cfg.get("enabled") or not _market_is_open():
            return
        now = datetime.now(ET).time()
        if now < _parse_hhmm(cfg["fire_start"]) or now > _parse_hhmm(cfg["fire_end"]):
            return
        from src.live.alpaca_options import _trading
        date_tag = datetime.now(ET).strftime("%Y%m%d")
        coid = f"0dte-SPY-{date_tag}"
        try:
            if _trading().get_order_by_client_id(coid):
                return   # already placed today
        except Exception:
            pass

        from src.notifications.slack_notifier import send_message
        from config.settings import IS_PAPER
        if IS_PAPER:
            return   # never auto-trade 0DTE on paper-as-live by accident
        snap, rcfg = _risk_snapshot(), _risk_config()
        ok, reason = _risk_gate(snap, rcfg, None)
        if not ok:
            return
        pok, preason, _ = _premium_ok()
        if not pok:
            logger.info("0DTE stand-down: %s", preason)
            return

        from src.analysis.expected_move import compute
        em = compute("SPY", 0)
        if not em or em.get("dte") != 0:
            logger.info("0DTE: no live 0DTE expiry/quotes")
            return
        short_k, long_k = _zerodte_strikes(em["forward"], em["em_dollars"], float(cfg["width"]))
        budget = rcfg["risk_pct_per_trade"] * snap["equity"] * _vix_size_factor(snap.get("vix"), rcfg)
        res = _zerodte_place(short_k, long_k, coid, budget, cfg)
        if not res["ok"]:
            logger.info("0DTE stood down: %s", res["reason"])
            return
        send_message(
            f":zap: *0DTE SPY put spread placed* — at the expected move\n"
            f"  SELL {short_k:.0f}P / BUY {long_k:.0f}P  {em['expiry']} (0DTE)\n"
            f"  credit +${res['credit']:.2f}  max loss ${res['max_loss']:.0f}  "
            f"_short ~1σ below ({em['lower']:.0f})_\n"
            f"  manage 50% · force-close {cfg['force_close']} ET\n"
            f"  Order: `{res['order_id']}`")
        logger.info("0DTE placed: SPY %.0f/%.0f credit %.2f", short_k, long_k, res["credit"])
    except Exception as e:
        logger.error("_zerodte_job error: %s", e)


def _zerodte_manage_job():
    """Every 5 min RTH. Manages the open 0DTE SPY put spread: close at 50% profit,
    DON'T close losers early, force-close before the bell (assignment guard)."""
    try:
        if not _market_is_open():
            return
        cfg = _zerodte_config()
        if not cfg.get("enabled"):
            return
        from src.live.alpaca_options import _trading
        client, today = _trading(), date.today()
        legs = []
        for p in client.get_all_positions():
            if len(p.symbol) <= 8 or not p.symbol.startswith("SPY"):
                continue
            pr = _occ_parse(p.symbol)
            if pr and pr[1] == today and pr[3] == "P":
                legs.append(p)
        if not legs:
            return
        net_credit = sum(-float(p.qty) * float(getattr(p, "avg_entry_price", 0) or 0) * 100 for p in legs)
        unreal     = sum(float(getattr(p, "unrealized_pl", 0) or 0) for p in legs)
        now = datetime.now(ET).time()
        reason = None
        if net_credit > 0 and unreal >= cfg.get("profit_target", 0.5) * net_credit:
            reason = f"50% profit (+${unreal:.0f} of ${net_credit:.0f} credit)"
        elif now >= _parse_hhmm(cfg["force_close"]):
            reason = f"force-close {cfg['force_close']} ET — 0DTE assignment guard"
        if not reason:
            return
        # don't double-submit while a close is already working
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        try:
            for o in client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=50)):
                if (getattr(o, "client_order_id", "") or "").startswith("0dteclose"):
                    return
        except Exception:
            pass
        res = _submit_close(client, legs, f"0dteclose-{datetime.now(ET):%Y%m%d%H%M}")
        if res:
            from src.notifications.slack_notifier import send_message
            icon = ":green_circle:" if res["unreal"] >= 0 else ":red_circle:"
            send_message(f"{icon} *0DTE close* — {reason}\n  P&L ${res['unreal']:+,.2f}  "
                         f"Order: `{res['order_id']}`")
            logger.info("0DTE close (%s) order=%s", reason, res["order_id"])
    except Exception as e:
        logger.error("_zerodte_manage_job error: %s", e)


def _submit_close(client, legs, coid):
    """Close a specific set of option-leg positions as one multi-leg order."""
    from src.live.alpaca_options import get_mid_price
    from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, PositionIntent, OrderClass
    net_debit, leg_objs, unreal, underlying = 0.0, [], 0.0, None
    for pos in legs:
        mid = get_mid_price(pos.symbol)
        if mid is None:
            continue
        qn, ratio = float(pos.qty), int(abs(float(pos.qty)))
        unreal += float(getattr(pos, "unrealized_pl", 0) or 0)
        underlying = _occ_parse(pos.symbol)[0]
        if qn > 0:
            side, intent, limit = OrderSide.SELL, PositionIntent.SELL_TO_CLOSE, round(mid * 0.90, 2)
            net_debit -= limit * ratio * 100
        else:
            side, intent, limit = OrderSide.BUY, PositionIntent.BUY_TO_CLOSE, round(mid * 1.10, 2)
            net_debit += limit * ratio * 100
        leg_objs.append(OptionLegRequest(symbol=pos.symbol, ratio_qty=ratio, side=side, position_intent=intent))
    if not leg_objs:
        return None
    is_debit  = net_debit > 0
    net_share = round(abs(net_debit) / 100, 2)
    o = client.submit_order(LimitOrderRequest(
        qty=1, order_class=OrderClass.MLEG,
        side=OrderSide.BUY if is_debit else OrderSide.SELL,
        type=OrderType.LIMIT, time_in_force=TimeInForce.DAY,
        limit_price=(net_share if is_debit else -net_share),
        legs=leg_objs, client_order_id=coid))
    return {"order_id": str(o.id), "unreal": round(unreal, 2)}


# ── One-shot auto-trade ──────────────────────────────────────────────────────
#
# Fires AT MOST ONE defined-risk structure on the armed date (config/auto_trade.json),
# during [fire_start, fire_end] ET, re-checking every 5 min. Best passing candidate
# in priority condor > bull put > fly, capped at max_loss. Re-priced + drift-guarded
# at fill (same gate as the buttons). Idempotent via Alpaca client_order_id so a
# Render restart can't double-fire. Kill switch: env AUTO_TRADE_KILL=1.

_AUTO_RUNTIME = os.path.join(os.path.dirname(__file__), "data", "auto_trade_runtime.json")
_AUTO_ERR: dict = {}        # de-dupe: last date we posted a submit-error to Slack


def _auto_runtime_get() -> dict | None:
    """Slack-toggle override ({'enabled': bool}). Lives on the (ephemeral) disk —
    reliably holds intraday while the app is warm; reverts to the committed
    config after a restart. For an overnight-durable kill use env AUTO_TRADE_KILL."""
    try:
        with open(_AUTO_RUNTIME) as f:
            return json.load(f)
    except Exception:
        return None


def _auto_runtime_set(enabled: bool):
    try:
        os.makedirs(os.path.dirname(_AUTO_RUNTIME), exist_ok=True)
        with open(_AUTO_RUNTIME, "w") as f:
            json.dump({"enabled": bool(enabled), "ts": time.time()}, f)
    except Exception as e:
        logger.warning("auto runtime write failed: %s", e)


def _auto_trade_config() -> dict:
    path = os.path.join(os.path.dirname(__file__), "config", "auto_trade.json")
    try:
        with open(path) as f:
            cfg = json.load(f)
    except Exception:
        cfg = {"enabled": False}
    if os.environ.get("AUTO_TRADE_DATE"):
        cfg["armed_date"] = os.environ["AUTO_TRADE_DATE"]
    if os.environ.get("AUTO_TRADE_MAX_LOSS"):
        try:
            cfg["max_loss"] = float(os.environ["AUTO_TRADE_MAX_LOSS"])
        except ValueError:
            pass
    # Slack toggle (/autotrade on|off) overrides the committed default…
    rt = _auto_runtime_get()
    if rt is not None and "enabled" in rt:
        cfg["enabled"] = bool(rt["enabled"])
    # …but the env kill switch is the hard master and always wins.
    if os.environ.get("AUTO_TRADE_KILL"):
        cfg["enabled"] = False
    return cfg


def _auto_fired_today(date_tag: str) -> bool:
    """True if today's auto-trade already placed — scans Alpaca for an entry order
    tagged `auto-<strategy>-<date_tag>`. Durable / restart-proof, and strategy-
    agnostic (the strategy is encoded in the tag for attribution)."""
    from src.live.alpaca_options import _trading
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    try:
        orders = _trading().get_orders(filter=GetOrdersRequest(
            status=QueryOrderStatus.ALL, limit=100,
            after=datetime.now(ET) - timedelta(days=2)))
        for o in orders:
            c = getattr(o, "client_order_id", "") or ""
            if c.startswith("auto-") and not c.startswith("autoclose") and date_tag in c:
                return True
    except Exception:
        pass
    return False


def _auto_pick_candidate(max_loss: float):
    """Return the single best structure clearing the $max_loss gate, or None.
    Priority: GEX condor (high POP) > bull put (STRONG first, earnings-clear) > pin fly."""
    from src.analysis.gex_scanner    import scan as gex_scan
    from src.analysis.condor_scanner  import scan as condor_scan
    from src.analysis.butterfly_scanner import scan as fly_scan
    from src.analysis.bull_put_scanner import scan as bp_scan

    d   = get_data()
    vix = d["vix"]
    try:
        gx = gex_scan()
    except Exception:
        gx = None

    # Sell premium (condor / bull put) only when IV > RV; the long fly is exempt
    # (it actually benefits from cheap premium).
    prem_ok, _prem_reason, _ = _premium_ok()
    if not prem_ok:
        logger.info("Auto-trade: skipping premium-sellers — %s", _prem_reason)

    # 1) Iron condor (no 0DTE — dte_min=1)
    if gx and prem_ok:
        try:
            con = condor_scan(gx.spot, gx, dte_min=1, dte_max=7).get("candidate")
            if con and float(con.get("max_loss_usd", 1e9)) <= max_loss and int(con.get("dte", 0)) >= 1:
                return {
                    "strategy": "condor", "underlying": con["ticker"], "expiry": con["expiry"],
                    "legs": con["legs"], "ref_net": float(con["credit"]),
                    "max_loss": con["max_loss_usd"],
                    "label": (f"Condor {con['ticker']} {con['short_put']:.0f}/{con['long_put']:.0f}P "
                              f"{con['short_call']:.0f}/{con['long_call']:.0f}C {con['expiry']}"),
                    "detail": f"credit +${con['credit']:.2f}  POP {con.get('pop_pct','?')}%  max loss ${con['max_loss_usd']}",
                }
        except Exception as e:
            logger.warning("Auto-trade condor check failed: %s", e)

    # 2) Bull put (STRONG before WATCH, higher score first, no earnings catalyst)
    try:
        bp = bp_scan(vix_now=vix["now"], vix_prev=vix["prev"])
    except Exception as e:
        logger.warning("Auto-trade bull-put scan failed: %s", e)
        bp = []
    hits = [r for r in bp if r.get("signal") in ("STRONG", "WATCH") and r.get("candidate")] if prem_ok else []
    hits.sort(key=lambda r: (r["signal"] != "STRONG", -float(r["candidate"].get("score", 0))))
    for r in hits:
        c = r["candidate"]
        if float(c.get("max_loss_usd", 1e9)) > max_loss:
            continue
        if int(c.get("dte", 0)) < 1:          # no 0DTE
            continue
        try:
            from src.live.news import earnings_guard
            if earnings_guard(r["ticker"]).get("flagged"):
                logger.info("Auto-trade skip %s — earnings catalyst", r["ticker"])
                continue
        except Exception:
            pass
        return {
            "strategy": "bull_put", "underlying": r["ticker"], "expiry": c["expiry"],
            "legs": [{"action": "SELL", "strike": c["short_strike"], "opt": "P", "qty": 1},
                     {"action": "BUY",  "strike": c["long_strike"],  "opt": "P", "qty": 1}],
            "ref_net": float(c["credit"]), "max_loss": c["max_loss_usd"],
            "label": f"Bull put {r['ticker']} {c['short_strike']}/{c['long_strike']} {c['expiry']}",
            "detail": f"credit +${c['credit']:.2f} ({c['credit_pct']:.0f}%)  max loss ${c['max_loss_usd']}",
        }

    # 3) Pin fly (cheapest defined risk; no 0DTE — dte_min=1)
    if gx:
        try:
            flies = fly_scan(gx.spot, gx, dte_min=1, dte_max=10).get("candidates") or []
            flies = [f for f in flies
                     if float(f.get("debit_usd", 1e9)) <= max_loss and int(f.get("dte", 0)) >= 1]
            if flies:
                f0 = sorted(flies, key=lambda f: -f["rr"])[0]
                return {
                    "strategy": "fly", "underlying": f0["ticker"], "expiry": f0["expiry"],
                    "legs": f0["legs"], "ref_net": -float(f0["debit"]), "max_loss": f0["debit_usd"],
                    "label": (f"Pin fly {f0['ticker']} {f0['long_upper']:.0f}/{f0['short_body']:.0f}/"
                              f"{f0['long_lower']:.0f} {f0['expiry']}"),
                    "detail": f"debit -${f0['debit']:.2f}  R/R {f0['rr']}  max loss ${f0['debit_usd']}",
                }
        except Exception as e:
            logger.warning("Auto-trade fly check failed: %s", e)
    return None


def _auto_place(cand: dict, client_order_id: str) -> dict:
    """Re-price + drift-guard + submit the chosen structure (qty 1) with a fixed
    client_order_id for idempotency. Returns {ok, reason|order_id, live_net, is_credit}."""
    from src.live.alpaca_options import _trading
    from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, PositionIntent, OrderClass
    try:
        legs = _legs_to_occ(cand["underlying"], cand["expiry"], cand["legs"])
        live_net, priced = _reprice_net(legs)
        if live_net is None:
            return {"ok": False, "reason": "no live quotes to re-price"}
        ref_net = float(cand["ref_net"])
        if _drifted(ref_net, live_net, REPRICE_TOLERANCE):
            return {"ok": False,
                    "reason": f"price drifted (alert ${abs(ref_net):.2f} → live ${abs(live_net):.2f}, past {REPRICE_TOLERANCE:.0%})"}
        is_credit = live_net > 0
        leg_objs = [
            OptionLegRequest(
                symbol=lg["symbol"], ratio_qty=lg["ratio"],
                side=OrderSide.SELL if lg["side"] == "SELL" else OrderSide.BUY,
                position_intent=(PositionIntent.SELL_TO_OPEN if lg["side"] == "SELL"
                                 else PositionIntent.BUY_TO_OPEN))
            for lg in priced
        ]
        client = _trading()
        order = client.submit_order(LimitOrderRequest(
            qty=1, order_class=OrderClass.MLEG,
            side=OrderSide.SELL if is_credit else OrderSide.BUY,
            type=OrderType.LIMIT, time_in_force=TimeInForce.DAY,
            limit_price=round(-live_net, 2), legs=leg_objs,   # signed: credit<0, debit>0
            client_order_id=client_order_id))
        # confirm it actually fills — never report a fill on mere acceptance
        status, filled, note = _confirm_fill(client, order.id)
        return {"ok": True, "order_id": str(order.id), "live_net": live_net,
                "is_credit": is_credit, "status": status, "filled": filled, "note": note}
    except Exception as e:
        return {"ok": False, "reason": f"submit error: {e}"}


def _auto_trade_job():
    """Every 5 min 10:00–12:55 ET (gated to the armed window). Places at most one
    structure on the armed date, then is idempotent for the rest of the day."""
    try:
        cfg = _auto_trade_config()
        if not cfg.get("enabled"):
            return
        today = datetime.now(ET).strftime("%Y-%m-%d")
        # recurring → fires every market day; otherwise only on the armed date
        if not cfg.get("recurring") and cfg.get("armed_date") != today:
            return
        if not _market_is_open():
            return
        now = datetime.now(ET).time()
        if now < dt_time(10, 0) or now >= dt_time(12, 35):
            return

        date_tag = today.replace('-', '')
        if _auto_fired_today(date_tag):
            return   # already placed today (survives restarts via Alpaca)

        from src.notifications.slack_notifier import send_message
        max_loss = float(cfg.get("max_loss", 100))

        # Safety: refuse to fire on a paper account when require_live is set
        from config.settings import IS_PAPER
        if cfg.get("require_live", True) and IS_PAPER:
            if dt_time(10, 0) <= now < dt_time(10, 5):
                send_message(":octagonal_sign: *Auto-trade halted* — account is *PAPER*, not LIVE "
                             "(`require_live`). Set `ALPACA_BASE_URL=https://api.alpaca.markets` "
                             "in Render to go live. No trade.")
            return

        # Risk engine: portfolio circuit breakers + %-of-equity sizing
        try:
            snap = _risk_snapshot()
            rcfg = _risk_config()
            allowed, reason = _risk_gate(snap, rcfg, None)
            if not allowed:
                if dt_time(10, 0) <= now < dt_time(10, 5):
                    send_message(f":octagonal_sign: *Auto-trade halted by risk rule* — {reason}. "
                                 f"No trade today.")
                return
            budget   = rcfg["risk_pct_per_trade"] * snap["equity"]
            factor   = _vix_size_factor(snap.get("vix"), rcfg)   # VIX de-risk (½ / stand-down)
            max_loss = min(max_loss, budget) * factor
        except Exception as e:
            logger.warning("Auto-trade risk gate failed (using hard cap): %s", e)

        # arming notice once, on the first (10:00) tick
        if dt_time(10, 0) <= now < dt_time(10, 5):
            _m = "LIVE" if not IS_PAPER else "PAPER"
            send_message(f":robot_face: *Auto-trade armed* ({_m}) — scanning for the best defined-risk "
                         f"setup ≤ ${max_loss:.0f} max loss until 12:30 ET. One shot.")

        cand = _auto_pick_candidate(max_loss)
        if not cand:
            if now >= dt_time(12, 30):
                send_message(f":robot_face: *Auto-trade stood down* — no setup cleared the "
                             f"${max_loss:.0f} max-loss gate by 12:30 ET. No trade today.")
            return   # otherwise keep checking next tick

        # AUTO-PLACE the one daily trade (fully automatic; the rest of the system
        # is button-approval). Marketable limit + fill confirmation built in.
        res = _auto_place(cand, f"auto-{cand['strategy']}-{date_tag}")
        if not res["ok"]:
            logger.info("Auto-trade: stood down this tick — %s", res["reason"])
            if res["reason"].startswith("submit error") and _AUTO_ERR.get("date") != date_tag:
                _AUTO_ERR["date"] = date_tag
                send_message(f":rotating_light: *Auto-trade submit error* — {res['reason']}\n"
                             f"  ({cand['strategy']} · {cand['label']})")
            return   # drift/quotes may resolve; retry next tick (idempotent)

        word   = "credit" if res["is_credit"] else "debit"
        status = res.get("status", "working")
        if status == "filled":
            send_message(
                f":robot_face: *AUTO-TRADE FILLED* — {cand['strategy']}\n"
                f"  *{cand['label']}*\n  {cand['detail']}\n"
                f"  net {word} ${abs(res['live_net']):.2f}  (max loss ≤ ${max_loss:.0f})\n"
                f"  Order ID: `{res['order_id']}`  ·  managed at 50% / 1-DTE")
            logger.info("Auto-trade FILLED: %s | %s | order=%s",
                        cand["strategy"], cand["label"], res["order_id"])
        elif status in ("rejected", "canceled"):
            send_message(
                f":rotating_light: *Auto-trade not filled* ({status}) — {cand['strategy']} {cand['label']}\n"
                f"  {res.get('note') or 'did not fill at the limit'}\n"
                f"  Order ID: `{res['order_id']}` · daily attempt spent")
            logger.info("Auto-trade %s: %s (%s)", status, cand["label"], res.get("note"))
        else:   # working — resting limit; report once, idempotency stops re-fire
            send_message(
                f":robot_face: *Auto-trade working* — {cand['strategy']} {cand['label']}\n"
                f"  limit net {word} ${abs(res['live_net']):.2f} resting; fills if the market "
                f"reaches it.  Order ID: `{res['order_id']}`")
            logger.info("Auto-trade WORKING: %s | %s", cand["strategy"], cand["label"])
    except Exception as e:
        logger.error("_auto_trade_job error: %s", e)
        try:
            from src.notifications.slack_notifier import send_message
            send_message(f":rotating_light: Auto-trade error: {e}")
        except Exception:
            pass


def _auto_open_structures():
    """Find still-open auto-trade structures, restart-proof. Queries Alpaca for
    filled entry orders tagged client_order_id 'auto-*' (NOT 'autoclose-*') whose
    legs are still open positions. Alpaca is the durable source of truth, so this
    survives any Render restart. Returns [{coid, underlying, expiry(date), legs}]."""
    from src.live.alpaca_options import _trading
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    client = _trading()
    try:
        orders = client.get_orders(filter=GetOrdersRequest(
            status=QueryOrderStatus.ALL, limit=300,
            after=datetime.now(ET) - timedelta(days=21)))
    except Exception as e:
        logger.warning("auto-manage: get_orders failed: %s", e)
        return []
    positions = client.get_all_positions()
    open_syms = {p.symbol for p in positions}
    out = []
    for o in orders:
        coid = getattr(o, "client_order_id", "") or ""
        if not coid.startswith("auto-") or coid.startswith("autoclose"):
            continue
        legs = getattr(o, "legs", None) or []
        leg_syms = [getattr(lg, "symbol", None) for lg in legs]
        still_open = [s for s in leg_syms if s and s in open_syms]
        if not still_open:
            continue
        parsed = _occ_parse(still_open[0])
        if not parsed:
            continue
        out.append({"coid": coid, "underlying": parsed[0], "expiry": parsed[1], "legs": still_open})

    # Manually-added structures: pull open option legs for allowlisted underlyings
    # (config/manual_manage.json) into auto-manage even though they weren't tagged
    # 'auto-*'. Grouped by (underlying, expiry); skips legs already captured above.
    manual = {u.upper() for u in _manual_manage_underlyings()}
    if manual:
        captured = {s for st in out for s in st["legs"]}
        from collections import defaultdict
        groups: dict = defaultdict(list)
        for p in positions:
            s = p.symbol
            if len(s) <= 6 or s in captured:
                continue
            parsed = _occ_parse(s)
            if parsed and parsed[0] in manual:
                groups[(parsed[0], parsed[1])].append(s)
        for (u, exp), syms in groups.items():
            out.append({"coid": f"manual-{u}-{exp}", "underlying": u, "expiry": exp, "legs": syms})
    return out


def _manual_manage_underlyings() -> list:
    """Underlyings whose manually-placed option structures should also be
    auto-managed (config/manual_manage.json)."""
    path = os.path.join(os.path.dirname(__file__), "config", "manual_manage.json")
    try:
        with open(path) as f:
            return json.load(f).get("underlyings", [])
    except Exception:
        return []


def _auto_close_structure(st: dict, reason: str):
    """Close exactly the auto-trade's open legs (not other positions on the same
    underlying) as one multi-leg order, idempotent via client_order_id."""
    from src.live.alpaca_options import _trading, get_mid_price
    from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, PositionIntent, OrderClass
    from src.notifications.slack_notifier import send_message
    client = _trading()
    coid = f"autoclose-{st['underlying']}-{st['expiry']}"
    try:
        if client.get_order_by_client_id(coid):
            return   # close already submitted
    except Exception:
        pass
    positions = {p.symbol: p for p in client.get_all_positions()}
    net_debit, leg_objs, unreal = 0.0, [], 0.0
    for sym in st["legs"]:
        pos = positions.get(sym)
        if not pos:
            continue
        mid = get_mid_price(sym)
        if mid is None:
            continue
        qn    = float(pos.qty)
        ratio = int(abs(qn))
        unreal += float(getattr(pos, "unrealized_pl", 0) or 0)
        if qn > 0:
            side, intent, limit = OrderSide.SELL, PositionIntent.SELL_TO_CLOSE, round(mid * 0.95, 2)
            net_debit -= limit * ratio * 100
        else:
            side, intent, limit = OrderSide.BUY, PositionIntent.BUY_TO_CLOSE, round(mid * 1.05, 2)
            net_debit += limit * ratio * 100
        leg_objs.append(OptionLegRequest(symbol=sym, ratio_qty=ratio, side=side, position_intent=intent))
    if not leg_objs:
        return
    is_debit  = net_debit > 0
    net_share = round(abs(net_debit) / 100, 2)
    order = client.submit_order(LimitOrderRequest(
        qty=1, order_class=OrderClass.MLEG,
        side=OrderSide.BUY if is_debit else OrderSide.SELL,
        type=OrderType.LIMIT, time_in_force=TimeInForce.DAY,
        limit_price=(net_share if is_debit else -net_share), legs=leg_objs, client_order_id=coid))
    icon = ":green_circle:" if unreal >= 0 else ":red_circle:"
    send_message(
        f"{icon} *AUTO-CLOSE `{st['underlying']}`* — {reason}\n"
        f"  {len(leg_objs)} legs, net {'debit' if is_debit else 'credit'} ${net_share:.2f}  |  "
        f"P&L ${unreal:+,.2f}\n  Order ID: `{order.id}`")
    logger.info("Auto-close %s %s — %s order=%s", st["underlying"], st["expiry"], reason, order.id)


def _auto_manage_job():
    """Every 5 min during RTH. Manages open auto-trade structures:
      - take profit automatically at the 50% target
      - force-close at 1 DTE, 3:30 PM ET (never hold into expiration)
    Only ever touches the auto-trade's own legs."""
    try:
        if not _market_is_open():
            return
        structs = _auto_open_structures()
        if not structs:
            return
        now    = datetime.now(ET)
        today  = date.today()
        # 50% targets reconstructed from leg strikes (credit/debit/fly aware)
        pflags = {(f["underlying"], str(f["exp"])): f for f in _profit_targets()}
        for st in structs:
            try:
                dte    = (st["expiry"] - today).days
                reason = None
                f = pflags.get((st["underlying"], str(st["expiry"])))
                if f and f["pct"] >= 50:
                    reason = f"50% profit target ({f['pct']}%)"
                elif dte <= 1 and now.time() >= dt_time(15, 30):
                    reason = f"1 DTE force-close — {dte} DTE, 3:30 ET (no holding into expiry)"
                if reason:
                    _auto_close_structure(st, reason)
            except Exception as e:
                logger.warning("auto-manage: close %s failed: %s", st.get("underlying"), e)
    except Exception as e:
        logger.error("_auto_manage_job error: %s", e)


def _start_scheduler():
    """Start background scheduler once — guarded against double-start in reloaders."""
    from apscheduler.schedulers.background import BackgroundScheduler

    sched = BackgroundScheduler(timezone=ET)
    sched.add_job(_premarket_prep_job, "cron", day_of_week="mon-fri", hour=8, minute=30,
                  id="premarket_prep",     replace_existing=True)
    sched.add_job(_unified_scan_job, "cron", day_of_week="mon-fri", hour=9,  minute=45,
                  id="scan_0945",          replace_existing=True)
    sched.add_job(_unified_scan_job, "cron", day_of_week="mon-fri", hour=10, minute=0,
                  id="scan_1000",          replace_existing=True)
    sched.add_job(_unified_scan_job, "cron", day_of_week="mon-fri", hour=10, minute=30,
                  id="scan_1030",          replace_existing=True)
    sched.add_job(_unified_scan_job, "cron", day_of_week="mon-fri", hour=12, minute=30,
                  id="scan_1230",          replace_existing=True)
    # SPY trade + rotation monitor — every 5 min during RTH (market-open guarded)
    sched.add_job(_spy_trade_monitor_job, "cron", day_of_week="mon-fri",
                  hour="9-15", minute="*/5", id="spy_monitor", replace_existing=True)
    sched.add_job(_status_hourly_job, "cron", day_of_week="mon-fri",
                  hour="9-16", minute=0, id="status_hourly", replace_existing=True)
    sched.add_job(_manage_5050_job,  "cron", day_of_week="mon-fri", hour=15, minute=30,
                  id="manage_1530",        replace_existing=True)
    sched.add_job(_eod_report_job,   "cron", day_of_week="mon-fri", hour=16, minute=5,
                  id="eod_report",         replace_existing=True)
    # One-shot auto-trade: enters 10:00–12:30 (armed date in config), exit 50%/1-DTE
    sched.add_job(_auto_trade_job,   "cron", day_of_week="mon-fri", hour="10-12", minute="*/5",
                  id="auto_trade",         replace_existing=True)
    sched.add_job(_auto_manage_job,  "cron", day_of_week="mon-fri", hour="9-15", minute="*/5",
                  id="auto_manage",        replace_existing=True)
    # Fallen-angel exit: trim half at +20%, trail the rest — every 15 min RTH
    sched.add_job(_fa_manage_job,    "cron", day_of_week="mon-fri", hour="9-15", minute="*/15",
                  id="fa_manage",          replace_existing=True)
    # 0DTE put spread at the expected move (opt-in; gated to its window + manager)
    sched.add_job(_zerodte_job,        "cron", day_of_week="mon-fri", hour="10-11", minute="*/5",
                  id="zerodte",            replace_existing=True)
    sched.add_job(_zerodte_manage_job, "cron", day_of_week="mon-fri", hour="9-15", minute="*/5",
                  id="zerodte_manage",     replace_existing=True)
    # Monthly NAV statement — 1st of the month, 8 AM ET
    sched.add_job(_monthly_nav_job,  "cron", day=1, hour=8, minute=0,
                  id="monthly_nav",        replace_existing=True)
    sched.start()
    logger.info("Scheduler started: prep 8:30 / scan 9:45/10:00/10:30/12:30 / "
                "SPY monitor */5 / manage 3:30 / EOD 4:05 / auto-trade 10:00-12:30 + "
                "auto-manage */5 PM ET (Mon-Fri)")
    return sched


# Start scheduler only in the main process (not in Flask debug reloader child)
if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    _scheduler = _start_scheduler()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
