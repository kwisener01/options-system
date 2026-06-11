"""
GEX Dashboard — Flask web app.

Routes:
  GET  /          — dashboard HTML
  GET  /api/gex   — JSON scan data (cached 5 min; ?force=true to refresh)
  POST /api/slack — post current data to Slack
  GET  /health    — Render health check
"""
import logging
import os
import sys
import threading
import time
import warnings
from datetime import datetime

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

def _bull_put_scan_job():
    """Runs at 9:45 AM and 12:30 PM ET Mon-Fri. Sends Slack alert for STRONG setups only."""
    try:
        from src.analysis.bull_put_scanner import scan as bp_scan, fmt_slack as bp_fmt
        from src.notifications.slack_notifier import send_message

        d   = get_data()
        vix = d["vix"]
        results = bp_scan(vix_now=vix["now"], vix_prev=vix["prev"])

        strong = [r for r in results if r.get("signal") == "STRONG"]
        if not strong:
            logger.info("Bull put scan: no STRONG setups — skipping Slack alert")
            return

        fc  = d["spy"].get("full_chain") or {}
        ts  = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
        msg = bp_fmt(results, fc.get("regime", "UNKNOWN"), vix["now"], ts)
        msg += "\n\n:point_right: *Review above and reply to place a trade.*"
        send_message(msg)
        logger.info("Bull put scan alert sent — %d STRONG setup(s)", len(strong))

        # Refresh cache so dashboard stays current
        with _bp_lock:
            _bp_cache["data"] = results
            _bp_cache["ts"]   = time.time()

    except Exception as e:
        logger.error("_bull_put_scan_job error: %s", e)


def _start_scheduler():
    """Start background scheduler once — guarded against double-start in reloaders."""
    from apscheduler.schedulers.background import BackgroundScheduler

    sched = BackgroundScheduler(timezone=ET)
    sched.add_job(_bull_put_scan_job, "cron", day_of_week="mon-fri", hour=9,  minute=45,
                  id="bull_put_morning",   replace_existing=True)
    sched.add_job(_bull_put_scan_job, "cron", day_of_week="mon-fri", hour=12, minute=30,
                  id="bull_put_midday",    replace_existing=True)
    sched.start()
    logger.info("Bull put scheduler started (9:45 AM + 12:30 PM ET, Mon-Fri)")
    return sched


# Start scheduler only in the main process (not in Flask debug reloader child)
if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    _scheduler = _start_scheduler()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
