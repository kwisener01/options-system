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
    vix = meta("^VIX")
    return (
        spy["regularMarketPrice"], spy["chartPreviousClose"],
        qqq["regularMarketPrice"], qqq["chartPreviousClose"],
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

    contracts = fetch_chain_for_gex(ticker, spot=spot, n_expiries=4)

    c_by_dte: dict[int, list] = {}
    for c in contracts:
        dte = round(c["T"] * 365)
        c_by_dte.setdefault(dte, []).append(c)

    c0 = c_by_dte.get(0) or contracts
    c1 = c_by_dte.get(1, [])

    r0 = compute_exposures(spot, vix_now, vix_prev, c0) if c0 else None
    r1 = compute_exposures(spot, vix_now, vix_prev, c1) if c1 else None
    rA = compute_exposures(spot, vix_now, vix_prev, contracts) if contracts else None

    return {
        "spot":       spot,
        "prev_close": prev,
        "change":     round(spot - prev, 2),
        "change_pct": round((spot - prev) / prev * 100, 2),
        "today":      _r_to_dict(r0),
        "tomorrow":   _r_to_dict(r1),
        "full_chain": _r_to_dict(rA),
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
        return _build_trade_idea(spy, vix_now, vix_prev)


def _live_trade_impl(spy: dict, vix_now: float, vix_prev: float) -> dict:
    from src.live.alpaca_options import fetch_chain_combined
    from src.analysis.bwb_analyzer import BWBInputs, analyze

    r        = spy.get("full_chain") or spy.get("tomorrow") or spy.get("today") or {}
    spot     = spy["spot"]
    flip     = r.get("flip_level") or spot - 10
    put_wall = r.get("put_wall")   or spot - 15
    regime   = r.get("regime", "UNKNOWN")
    vix_chg  = vix_now - vix_prev

    bearish_signals = sum([
        "NEGATIVE" in regime,
        r.get("vanna_signal") == "BEARISH",
        vix_chg > 0.3,
        spot < flip,
    ])
    bearish = bearish_signals >= 2

    # Fetch 0–3 DTE SPY puts with live bid/ask
    chain = fetch_chain_combined("SPY", spot, dte_min=0, dte_max=3)
    puts  = chain.get("puts_liquid", [])

    # Group by expiry; prefer soonest with DTE ≥ 1
    by_exp: dict[str, list] = {}
    for p in puts:
        by_exp.setdefault(p["expiry"], []).append(p)

    target_exp, target_dte = None, 1
    for exp in sorted(by_exp.keys()):
        dte = by_exp[exp][0]["dte"]
        if dte >= 1:
            target_exp, target_dte = exp, dte
            break

    # Build {strike: mid} price map for the chosen expiry
    mids: dict[float, float] = {}
    if target_exp:
        for p in by_exp[target_exp]:
            mids[p["strike"]] = round((p["bid"] + p["ask"]) / 2, 2)

    def nearest(target: float) -> tuple[float, float]:
        """Closest available strike and its mid-price."""
        if not mids:
            return round(target), 0.0
        s = min(mids, key=lambda x: abs(x - target))
        return s, mids[s]

    thesis = [
        f"Put wall ${put_wall:.0f} — GEX-defined support"
        if not bearish else
        f"Negative gamma ({r.get('gex', 0):.0f}B) — dealer amplification",
        f"Regime: {regime.replace('_', ' ').title()}  |  VIX {vix_chg:+.2f}",
        f"Vanna: {r.get('vanna_signal','—')}  |  Charm: {r.get('charm_signal','—')}",
    ]

    # ── BULLISH: credit spread + BWB, both evaluated ──────────────────────────
    if not bearish:
        short_s, short_mid = nearest(round(put_wall))
        h_s,     h_mid     = nearest(short_s + 5)
        cs_l_s,  cs_l_mid  = nearest(short_s - 5)
        bwb_l_s, bwb_l_mid = nearest(short_s - 10)

        # Credit spread (sell short, buy cs_long)
        cs_credit = short_mid - cs_l_mid
        cs_width  = short_s - cs_l_s
        cs_profit = round(cs_credit * 100)
        cs_risk   = round((cs_width - cs_credit) * 100)

        # BWB (buy H, sell 2×M, buy L)
        bwb_credit = round(short_mid * 2 - h_mid - bwb_l_mid, 2)
        bwb_inp = BWBInputs(
            ticker="SPY", spot=spot, dte=target_dte,
            long_upper=h_s, short_strike=short_s, long_lower=bwb_l_s,
            credit=bwb_credit,
            regime=regime, vix_now=vix_now, vix_prev=vix_prev,
            flip_level=flip, put_wall=put_wall,
            call_wall=r.get("call_wall", 0), major_news=False,
        )
        bwb = analyze(bwb_inp)

        # Prefer BWB if it's rated well and credit is meaningful
        rec_bwb = (bwb.rating in ("A+", "Acceptable")
                   and bwb_credit >= 0.15
                   and bwb.max_loss_usd <= 700)

        return {
            "direction":       "BULLISH",
            "bearish_signals": bearish_signals,
            "expiry":          target_exp or "1DTE",
            "dte":             target_dte,
            "spot":            spot,
            "prices_live":     bool(mids),
            "cs": {
                "type":           "Bull Put Spread",
                "long_strike":    cs_l_s,
                "short_strike":   short_s,
                "width":          cs_width,
                "credit":         round(cs_credit, 2),
                "long_mid":       cs_l_mid,
                "short_mid":      short_mid,
                "max_profit_usd": cs_profit,
                "max_risk_usd":   cs_risk,
                "rr_ratio":       round(cs_profit / cs_risk, 2) if cs_risk > 0 else 0,
            },
            "bwb": {
                "type":            "Bull BWB",
                "long_upper":      h_s,
                "short_strike":    short_s,
                "long_lower":      bwb_l_s,
                "credit":          bwb_credit,
                "upper_mid":       h_mid,
                "short_mid":       short_mid,
                "lower_mid":       bwb_l_mid,
                "max_profit_usd":  round(bwb.max_profit_usd),
                "max_loss_usd":    round(bwb.max_loss_usd),
                "lower_breakeven": round(bwb.lower_breakeven, 2),
                "rr_ratio":        round(bwb.rr_ratio, 2),
                "rating":          bwb.rating,
                "score":           bwb.setup_score,
                "exit_plan":       bwb.exit_plan,
                "overnight_ok":    bwb.overnight_ok,
            },
            "recommended": "bwb" if rec_bwb else "cs",
            "thesis": thesis,
        }

    # ── BEARISH: bear put spread (BWB is a bullish-only structure) ────────────
    long_s,  long_mid  = nearest(round(flip))
    short_s, short_mid = nearest(max(round(put_wall), round(flip) - 10))
    if long_s - short_s < 3:
        short_s, short_mid = nearest(long_s - 5)

    width      = long_s - short_s
    net_debit  = round(long_mid - short_mid, 2)
    bear_profit = round((width - net_debit) * 100)
    bear_risk   = round(net_debit * 100)

    return {
        "direction":       "BEARISH",
        "bearish_signals": bearish_signals,
        "expiry":          target_exp or "1DTE",
        "dte":             target_dte,
        "spot":            spot,
        "prices_live":     bool(mids),
        "cs": {
            "type":           "Bear Put Spread",
            "long_strike":    long_s,
            "short_strike":   short_s,
            "width":          width,
            "credit":         -net_debit,
            "long_mid":       long_mid,
            "short_mid":      short_mid,
            "max_profit_usd": bear_profit,
            "max_risk_usd":   bear_risk,
            "rr_ratio":       round(bear_profit / bear_risk, 2) if bear_risk > 0 else 0,
        },
        "bwb":         None,
        "recommended": "cs",
        "thesis":      thesis,
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
    spy_spot, spy_prev, qqq_spot, qqq_prev, vix_now, vix_prev = _fetch_prices()
    spy   = _scan_ticker("SPY", spy_spot, spy_prev, vix_now, vix_prev)
    qqq   = _scan_ticker("QQQ", qqq_spot, qqq_prev, vix_now, vix_prev)
    trade = _build_live_trade_idea(spy, vix_now, vix_prev)
    return {
        "timestamp":  datetime.now(ET).strftime("%Y-%m-%d %H:%M ET"),
        "spy":        spy,
        "qqq":        qqq,
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
        rec       = t.get("recommended", "cs")
        cs        = t.get("cs") or {}
        bwb       = t.get("bwb")
        rec_label = ("BWB" if rec == "bwb" else cs.get("type", "CS"))
        live_tag  = " _(live prices)_" if t.get("prices_live") else ""

        cs_line = (
            f"CS: {cs.get('type','')}  "
            f"${cs.get('long_strike','?')}/{cs.get('short_strike','?')}  "
            f"Credit ${cs.get('credit',0):.2f}  "
            f"Risk ${cs.get('max_risk_usd',0):.0f}"
        ) if cs else ""

        bwb_line = ""
        if bwb:
            bwb_line = (
                f"BWB: {bwb.get('long_upper','?')}/{bwb.get('short_strike','?')}/{bwb.get('long_lower','?')}  "
                f"Credit ${bwb.get('credit',0):.2f}  "
                f"Rating {bwb.get('rating','?')}  MaxLoss ${bwb.get('max_loss_usd',0):.0f}"
            )

        trade_block = (
            f"\n{color} *Trade Idea — {direction}  |  Recommended: {rec_label}"
            f"  |  Expiry {t.get('expiry','')} ({t.get('dte','')}DTE){live_tag}*\n"
            + (f">  {cs_line}\n" if cs_line else "")
            + (f">  {bwb_line}\n" if bwb_line else "")
            + "\n".join(f">  - {pt}" for pt in t.get("thesis", []))
        )

    return (
        f":bar_chart: *GEX Dashboard -- {ts}*\n"
        f":zap: VIX {vix['now']:.2f} {vix_arrow} ({vix['change']:+.2f})\n\n"
        + ticker_block("SPY", spy) + "\n\n"
        + ticker_block("QQQ", qqq)
        + trade_block
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
