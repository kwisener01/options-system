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
import logging
import os
import sys
import threading
import time
import warnings
from datetime import datetime, date, timedelta

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
        with _wl_lock:
            age = time.time() - _wl_cache["ts"]
            if force or _wl_cache["data"] is None or age > WL_CACHE_TTL:
                logger.info("Running value watchlist scan (force=%s age=%.0fs)", force, age)
                d   = get_data()
                fc  = d["spy"].get("full_chain") or {}
                from src.analysis.value_watchlist import scan_watchlist
                _wl_cache["data"] = scan_watchlist(
                    spy_regime = fc.get("regime", "UNKNOWN"),
                    vix_now    = d["vix"]["now"],
                    vix_prev   = d["vix"]["prev"],
                )
                _wl_cache["ts"] = time.time()
        return jsonify({"ok": True, "signals": _wl_cache["data"],
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

    trade_block = ""
    if t:
        direction = t.get("direction", "")
        color     = ":red_circle:" if direction == "BEARISH" else ":green_circle:"
        live_tag  = " _(live prices)_" if t.get("prices_live") else ""
        trades    = t.get("trades") or []
        rec_trade = next((tr for tr in trades if tr.get("recommended")), None)
        rec_label = rec_trade["type"] if rec_trade else "—"

        trade_lines = []
        for tr in trades:
            star  = "★ " if tr.get("recommended") else "  "
            legs  = "/".join(f"${lg['strike']}" for lg in tr.get("legs", []))
            cr    = tr.get("credit", 0) or 0
            profit = tr.get("max_profit_usd") or 0
            risk   = tr.get("max_risk_usd") or 0
            rating = f"  [{tr['rating']}]" if tr.get("rating") else ""
            trade_lines.append(
                f">  {star}{tr['type']}  {legs}  "
                f"{'Cr' if cr >= 0 else 'Dr'} ${abs(cr):.2f}  "
                f"Profit ${profit}  Risk ${risk}{rating}"
            )

        trade_block = (
            f"\n{color} *SPY Trade Ideas — {direction}  |  Recommended: {rec_label}"
            f"  |  Expiry {t.get('expiry','')} ({t.get('dte','')}DTE){live_tag}*\n"
            + "\n".join(trade_lines)
            + ("\n" if trade_lines else "")
            + "\n".join(f">  - {pt}" for pt in t.get("thesis", []))
        )

    return (
        f":bar_chart: *GEX Dashboard -- {ts}*\n"
        f":zap: VIX {vix['now']:.2f} {vix_arrow} ({vix['change']:+.2f})\n\n"
        + ticker_block("SPY", spy) + "\n\n"
        + ticker_block("QQQ", qqq)
        + trade_block
    )


# ── Bull Put Scheduled Scanner (HITL alerts) ──────────────────────────────────

def _build_position_summary() -> tuple[str, str]:
    """
    Returns (close_block, positions_block) for open positions.
    close_block  — urgent close suggestions with runnable commands.
    positions_block — one-line summary of every open position.
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

    for p in positions:
        sym    = p.symbol
        qty    = float(p.qty)
        unreal = float(getattr(p, "unrealized_pl",   0) or 0)
        pct    = float(getattr(p, "unrealized_plpc", 0) or 0) * 100
        mkt    = float(getattr(p, "market_value",    0) or 0)
        icon   = ":green_circle:" if unreal >= 0 else ":red_circle:"
        is_opt = len(sym) > 6

        pos_lines.append(
            f"  {icon} `{sym}`  ${mkt:,.2f}  P&L ${unreal:+,.2f} ({pct:+.1f}%)"
        )

        if is_opt:
            # Parse underlying and DTE from OCC symbol
            try:
                tp         = len(sym) - 9
                raw        = sym[tp - 6: tp]
                exp_date   = date(2000 + int(raw[0:2]), int(raw[2:4]), int(raw[4:6]))
                dte        = (exp_date - today).days
                underlying = sym[:tp - 6]
            except Exception:
                dte        = 99
                underlying = sym

            if dte <= 3:
                close_tips.append(
                    f":warning: *`{sym}`* — {dte} DTE, expires soon.\n"
                    f"  > `python close_bwb.py --ticker {underlying}`  _(or let expire — defined-risk)_"
                )
            # Short option at 50%+ profit
            cost = float(getattr(p, "cost_basis", 0) or 0)
            if qty < 0 and cost != 0:
                profit_pct = (abs(cost) - abs(mkt)) / abs(cost) * 100
                if profit_pct >= 50:
                    close_tips.append(
                        f":moneybag: *`{sym}`* — {profit_pct:.0f}% of max profit captured.\n"
                        f"  > `python close_bwb.py --ticker {underlying}`"
                    )
        else:
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

        # -- Open positions snapshot ------------------------------------------
        _, pos_block = _build_position_summary()

        # -- Assemble message -------------------------------------------------
        sections = [f":clipboard: *Pre-Market Prep — {ts}*"]

        if index_lines:
            sections += ["", "*--- MARKET ---*"] + index_lines

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
        bp_results = fa_results = vw_results = []
        gex_result = None

        def _run_bp():
            return bp_scan(vix_now=vix["now"], vix_prev=vix["prev"])
        def _run_fa():
            return fa_scan()
        def _run_vw():
            return vw_scan(spy_regime=regime, vix_now=vix["now"])
        def _run_gex():
            return gex_scan()

        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = {
                ex.submit(_run_bp):  "bp",
                ex.submit(_run_fa):  "fa",
                ex.submit(_run_vw):  "vw",
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
                    elif key == "gex":
                        gex_result = result
                except Exception as e:
                    logger.warning("Scanner %s failed: %s", key, e)

        # Refresh bull put cache
        with _bp_lock:
            _bp_cache["data"] = bp_results
            _bp_cache["ts"]   = time.time()

        # -- Check if anything is actionable ---------------------------------
        close_block, pos_block = _build_position_summary()

        bp_hits = [r for r in bp_results if r.get("signal") in ("STRONG", "WATCH")]
        fa_hits = [r for r in fa_results if r.get("signal") in ("STRONG", "WATCH")]
        vw_hits = [r for r in vw_results if r.get("signal") in ("STRONG", "WATCH", "SELL")]

        if not close_block and not bp_hits and not fa_hits and not vw_hits:
            logger.info("Unified scan: nothing actionable — skipping Slack alert")
            return

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
        trade_block = ""
        if t:
            direction = t.get("direction", "")
            color     = ":red_circle:" if direction == "BEARISH" else ":green_circle:"
            live_tag  = " _(live prices)_" if t.get("prices_live") else ""
            trades    = t.get("trades") or []
            rec_trade = next((tr for tr in trades if tr.get("recommended")), None)
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
            trade_block = (
                f"{color} *SPY Trade Ideas — {direction}  |  Recommended: {rec_label}"
                f"  |  Expiry {t.get('expiry','')} ({t.get('dte','')}DTE){live_tag}*\n"
                + "\n".join(trade_lines)
                + ("\n" if trade_lines else "")
                + "\n".join(f">  - {pt}" for pt in t.get("thesis", []))
            )

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
                f"  :point_right: `{cmd}`"
            )

        # -- Assemble message ------------------------------------------------
        sections = [f":mag: *{session} Scan — {ts}*"]

        if gex_block:
            sections += ["", gex_block]
        if trade_block:
            sections += ["", "*--- SPY TRADE IDEA ---*", trade_block]
        if close_block:
            sections += ["", "*--- CLOSE / ACTION ---*", close_block]
        if pos_block:
            sections += ["", "*--- OPEN POSITIONS ---*", pos_block]
        if fa_hits:
            sections += ["", fa_fmt(fa_results, ts)]
        if vw_hits:
            sections += ["", vw_fmt(vw_results, regime, vix["now"], ts)]
        if bp_place_lines:
            sections += ["", ":moneybag: *Bull Put Spreads — place trade*"] + bp_place_lines

        send_message("\n".join(sections))
        logger.info(
            "Unified scan alert sent — BP:%d FA:%d VW:%d close=%s",
            len(bp_hits), len(fa_hits), len(vw_hits), bool(close_block),
        )

    except Exception as e:
        logger.error("_unified_scan_job error: %s", e)


# ── EOD Report + Close Suggestions ───────────────────────────────────────────

def _eod_report_job():
    """Runs at 4:05 PM ET Mon-Fri. Sends daily P&L, full P&L, and close suggestions."""
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

        # -- Close suggestions ------------------------------------------------
        suggestions = []
        today = date.today()

        for p in stocks:
            pct = float(getattr(p, "unrealized_plpc", 0) or 0) * 100
            if pct <= -8:
                suggestions.append(
                    f":rotating_light: *CLOSE `{p.symbol}`* — down {pct:.1f}% from cost. Stop-loss zone.\n"
                    f"  > `python close_position.py --ticker {p.symbol}`"
                )
            elif pct >= 20:
                suggestions.append(
                    f":moneybag: *TRIM `{p.symbol}`* — up {pct:.1f}%. Consider taking partial profit.\n"
                    f"  > `python close_position.py --ticker {p.symbol}`"
                )

        for p in options:
            unreal = float(getattr(p, "unrealized_pl", 0) or 0)
            mkt    = float(getattr(p, "market_value",  0) or 0)
            qty    = float(p.qty)
            sym    = p.symbol
            # Parse DTE from OCC symbol
            try:
                type_pos = len(sym) - 9
                raw_date = sym[type_pos - 6: type_pos]
                exp_date = date(2000 + int(raw_date[0:2]), int(raw_date[2:4]), int(raw_date[4:6]))
                dte = (exp_date - today).days
            except Exception:
                dte = 99

            # Parse strike and type for close_bwb-style display
            try:
                strike = int(sym[-8:]) / 1000.0
                opt_type = sym[len(sym) - 9]  # C or P
                type_pos = len(sym) - 9
                raw_date = sym[type_pos - 6: type_pos]
                expiry_str = f"20{raw_date[0:2]}-{raw_date[2:4]}-{raw_date[4:6]}"
                underlying = sym[:type_pos - 6]
            except Exception:
                strike, opt_type, expiry_str, underlying = 0, "P", "", sym

            if dte <= 3:
                suggestions.append(
                    f":warning: *REVIEW `{sym}`* — {dte} DTE. Expires soon.\n"
                    f"  > `python close_bwb.py --ticker {underlying}`  _(or let expire — spread is defined-risk)_"
                )

            cost_basis = float(getattr(p, "cost_basis", 0) or 0)
            if qty < 0 and cost_basis != 0:
                profit_pct = (abs(cost_basis) - abs(mkt)) / abs(cost_basis) * 100
                if profit_pct >= 50:
                    suggestions.append(
                        f":white_check_mark: *CLOSE `{sym}`* — {profit_pct:.0f}% of max profit captured. Lock it in.\n"
                        f"  > `python close_bwb.py --ticker {underlying}`"
                    )

        if not suggestions:
            suggestion_block = "  _No action needed — hold current positions._"
        else:
            suggestion_block = "\n\n".join(suggestions)

        # -- Build message ----------------------------------------------------
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
            f"*Close / Action Suggestions:*\n{suggestion_block}"
        )
        send_message(msg)
        logger.info("EOD report sent")

    except Exception as e:
        logger.error("_eod_report_job error: %s", e)
        from src.notifications.slack_notifier import send_message as _sm
        _sm(f":rotating_light: EOD report error: {e}")


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


def _cmd_close(ticker: str, resp_url: str):
    from src.live.alpaca_options import _trading
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    try:
        client    = _trading()
        positions = client.get_all_positions()
        pos       = next((p for p in positions if p.symbol == ticker.upper()), None)
        if pos is None:
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
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, PositionIntent
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
            symbol=ticker, qty=1, side=OrderSide.SELL,
            type=OrderType.LIMIT, time_in_force=TimeInForce.DAY,
            limit_price=net_credit, legs=legs,
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


def _cmd_eod(resp_url: str):
    _eod_report_job()


HELP_TEXT = (
    ":robot_face: *Trader Bot Commands*\n\n"
    "`/positions`  — show all open positions + close suggestions\n"
    "`/close_position TICKER`  — market-sell a stock/ETF position\n"
    "  _e.g._ `/close_position VBR`\n\n"
    "`/place TICKER SHORT LONG EXPIRY [QTY]`  — place bull put credit spread\n"
    "  _e.g._ `/place KO 82.5 80 2026-07-17`\n\n"
    "`/scan`  — run bull put scanner now + send results\n"
    "`/eod`   — generate EOD P&L report now\n"
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
        "eod":       ":hourglass: Generating EOD report...",
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
        "eod":       lambda: _cmd_eod(resp_url),
    }
    threading.Thread(target=dispatch[command], daemon=True).start()
    return jsonify({"text": ack_map[command], "response_type": "in_channel"}), 200


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
    sched.add_job(_eod_report_job,   "cron", day_of_week="mon-fri", hour=16, minute=5,
                  id="eod_report",         replace_existing=True)
    sched.start()
    logger.info("Scheduler started: prep 8:30 / scan 9:45/10:00/10:30/12:30 / EOD 4:05 PM ET (Mon-Fri)")
    return sched


# Start scheduler only in the main process (not in Flask debug reloader child)
if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    _scheduler = _start_scheduler()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
