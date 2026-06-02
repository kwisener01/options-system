"""
Fallen Angel Scanner — institutional-grade bottom-fishing screen.

Finds large-cap S&P 500 stocks that have suffered major drawdowns and are
currently near their multi-year floor, with early signs of institutional
accumulation (volume expansion, RSI bottoming).

The INTC pattern:
  - Once-dominant company, crushed 50-70% from multi-year high
  - Trading near book value (or below)
  - Smart money quietly accumulating at the floor via CALL OPTIONS before price moves

Scoring (max 13):
  Depth of decline (0-3 pts):
    Down 50%+ from 52-week high : 3 pts
    Down 40-50%                 : 2 pts
    Down 30-40%                 : 1 pt
  Near the floor (0-2 pts):
    Within 10% of 52-week low   : 2 pts
    Within 20% of 52-week low   : 1 pt
  RSI position (0-3 pts):
    Was oversold (<30) in last 20 sessions, now recovering (35-55): 3 pts
    Currently oversold (<35)    : 2 pts
    RSI 35-45                   : 1 pt
  Volume expansion (0-2 pts):
    10d avg volume > 30d avg by 25%+ : 2 pts
    10d avg volume > 30d avg by 10%+ : 1 pt
  Options flow (0-3 pts) — the LEADING indicator:
    Put/call OI ratio < 0.5 (calls dominating)  : 2 pts
    Put/call OI ratio < 0.8                     : 1 pt
    GEX put wall within 5% of current price     : 1 pt
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Curated S&P 500 large-caps across all sectors — institutional quality only
UNIVERSE: list[str] = [
    # Financials
    "BAC", "JPM", "WFC", "GS", "MS", "C", "AXP", "BK", "SCHW", "PRU", "MET",
    # Technology
    "INTC", "QCOM", "CSCO", "IBM", "HPQ", "DELL", "STX", "WDC", "NTAP", "AMAT",
    # Healthcare / Pharma
    "PFE", "MRK", "ABBV", "BMY", "AMGN", "GILD", "CVS", "WBA", "HUM", "CI",
    # Energy
    "CVX", "XOM", "COP", "PSX", "VLO", "HAL", "SLB", "OXY", "DVN", "MPC",
    # Consumer Staples
    "KO", "PEP", "PM", "MO", "KR", "GIS", "K", "CAG", "SJM", "CPB",
    # Communication / Telecom
    "T", "VZ", "PARA", "WBD", "FOX", "IPG", "OMC",
    # Industrials
    "GE", "HON", "MMM", "RTX", "LMT", "BA", "CAT", "DE", "EMR", "ETN",
    # Materials
    "DOW", "LYB", "CF", "MOS", "NUE", "X", "CLF", "FCX", "NEM", "AA",
    # Utilities
    "SO", "D", "DUK", "EXC", "PCG", "XEL", "AES", "NRG",
    # REITs / Real Estate
    "VNO", "SLG", "MPW", "NYCB", "KIM",
]

FLOOR_THRESHOLD  = 0.10   # within 10% of 52w low = "near floor"
FLOOR_NEARBY     = 0.20   # within 20% of 52w low = "nearby"
DECLINE_DEEP     = 0.50   # down 50%+ from 52w high
DECLINE_MID      = 0.40   # down 40-50%
DECLINE_SOFT     = 0.30   # down 30-40%
OVERSOLD_RSI     = 30
RECOVERING_RSI   = (35, 55)
VOL_STRONG       = 1.25
VOL_MILD         = 1.10
PC_BULLISH       = 0.5    # call OI > 2x put OI — institutional call buying
PC_ELEVATED      = 0.8    # calls above average
PUT_WALL_RANGE   = 0.05   # put wall within 5% of spot = dealer floor

# Only fetch options for stocks that scored >= this on price/RSI/volume alone
OPTIONS_MIN_SCORE = 3

STRONG_SCORE = 8
WATCH_SCORE  = 5


@dataclass
class FallenAngel:
    ticker:           str
    spot:             float
    change_pct:       float
    high_52w:         float
    low_52w:          float
    pct_from_high:    float   # negative — how far below 52w high
    pct_from_low:     float   # positive — how far above 52w low
    rsi:              float
    rsi_was_oversold: bool
    vol_ratio:        float
    # Options flow (leading indicator)
    put_call_ratio:   Optional[float]
    put_wall:         Optional[float]
    call_wall:        Optional[float]
    options_regime:   Optional[str]
    score:            int
    signal:           str     # STRONG | WATCH | NONE
    reasons:          list = field(default_factory=list)


# ── Price history ─────────────────────────────────────────────────────────────

def _fetch_history_with_volume(ticker: str) -> dict:
    import urllib3
    import requests as _req
    urllib3.disable_warnings()
    url = (f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?interval=1d&range=1y")
    resp = _req.get(url, headers={"User-Agent": "Mozilla/5.0"},
                    timeout=15, verify=False).json()
    result = resp["chart"]["result"][0]
    meta   = result["meta"]
    quote  = result["indicators"]["quote"][0]
    closes  = [c for c in (quote.get("close")  or []) if c is not None]
    volumes = [v for v in (quote.get("volume") or []) if v is not None]
    return {
        "closes":   closes,
        "volumes":  volumes,
        "high_52w": meta.get("fiftyTwoWeekHigh"),
        "low_52w":  meta.get("fiftyTwoWeekLow"),
    }


def _fetch_all_prices(tickers: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_fetch_history_with_volume, t): t for t in tickers}
        for f in as_completed(futures):
            t = futures[f]
            try:
                out[t] = f.result()
            except Exception as e:
                logger.debug("Price fetch failed for %s: %s", t, e)
    return out


# ── Options flow ──────────────────────────────────────────────────────────────

def _fetch_options_flow(ticker: str, spot: float) -> dict:
    """
    Fetch put/call OI ratio and GEX levels for a single ticker.
    Returns empty dict if options data unavailable.
    """
    try:
        from src.live.alpaca_options import fetch_chain_combined
        from src.analysis.gex_scanner import compute_exposures
        chain = fetch_chain_combined(ticker, spot)
        contracts = chain.get("gex_contracts", [])
        if len(contracts) < 5:
            return {}

        # Put/call OI ratio from the chain
        total_call_oi = sum(c.get("open_interest", 0) or 0
                            for c in contracts if c.get("option_type") == "call")
        total_put_oi  = sum(c.get("open_interest", 0) or 0
                            for c in contracts if c.get("option_type") == "put")
        pc_ratio = round(total_put_oi / total_call_oi, 2) if total_call_oi > 0 else None

        # GEX levels
        vix_now, vix_prev = 20.0, 20.0   # neutral estimate when called standalone
        try:
            from src.live.alpaca_options import get_vix
            vix_now, vix_prev = get_vix()
        except Exception:
            pass

        gex = compute_exposures(spot, vix_now, vix_prev, contracts)
        return {
            "put_call_ratio": pc_ratio,
            "put_wall":       gex.put_wall,
            "call_wall":      gex.call_wall,
            "regime":         gex.gex_regime,
        }
    except Exception as e:
        logger.debug("Options flow failed for %s: %s", ticker, e)
        return {}


def _fetch_all_options(candidates: list[tuple[str, float]]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(_fetch_options_flow, t, spot): t for t, spot in candidates}
        for f in as_completed(futures):
            t = futures[f]
            try:
                out[t] = f.result()
            except Exception:
                out[t] = {}
    return out


# ── Indicators ────────────────────────────────────────────────────────────────

def _rsi_series(closes: list[float], period: int = 14) -> list[float]:
    if len(closes) < period + 2:
        return []
    out = []
    ch = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains  = [max(c, 0) for c in ch]
    losses = [max(-c, 0) for c in ch]
    ag = sum(gains[:period])  / period
    al = sum(losses[:period]) / period
    for i in range(period, len(ch)):
        ag = (ag * (period - 1) + gains[i])  / period
        al = (al * (period - 1) + losses[i]) / period
        rs = ag / al if al else 100.0
        out.append(round(100 - 100 / (1 + rs), 1))
    return out


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score_price(ticker: str, data: dict) -> Optional[dict]:
    """Score price/RSI/volume. Returns None if stock doesn't qualify (< 30% decline)."""
    closes  = data.get("closes", [])
    volumes = data.get("volumes", [])
    if len(closes) < 50 or not data.get("high_52w") or not data.get("low_52w"):
        return None

    spot = closes[-1]
    prev = closes[-2] if len(closes) > 1 else spot
    chg_pct = round((spot - prev) / prev * 100, 2) if prev else 0.0
    hi52 = data["high_52w"]
    lo52 = data["low_52w"]

    pct_from_high = round((spot - hi52) / hi52 * 100, 1)  # negative
    pct_from_low  = round((spot - lo52) / lo52 * 100, 1)  # positive

    if pct_from_high > -DECLINE_SOFT * 100:
        return None  # not beaten down enough

    rsi_series    = _rsi_series(closes)
    rsi_now       = rsi_series[-1] if rsi_series else 50.0
    recent_rsi    = rsi_series[-20:] if len(rsi_series) >= 20 else rsi_series
    was_oversold  = any(r < OVERSOLD_RSI for r in recent_rsi)

    vol_ratio = 1.0
    if len(volumes) >= 30:
        avg10 = sum(volumes[-10:]) / 10
        avg30 = sum(volumes[-30:]) / 30
        vol_ratio = round(avg10 / avg30, 2) if avg30 else 1.0

    score   = 0
    reasons = []

    # 1. Depth of decline (0-3 pts)
    decline = abs(pct_from_high) / 100
    if decline >= DECLINE_DEEP:
        score += 3
        reasons.append(f"Down {abs(pct_from_high):.1f}% from 52w high — deep value territory")
    elif decline >= DECLINE_MID:
        score += 2
        reasons.append(f"Down {abs(pct_from_high):.1f}% from 52w high — significant drawdown")
    elif decline >= DECLINE_SOFT:
        score += 1
        reasons.append(f"Down {abs(pct_from_high):.1f}% from 52w high — off highs")

    # 2. Near the floor (0-2 pts)
    proximity = pct_from_low / 100
    if proximity <= FLOOR_THRESHOLD:
        score += 2
        reasons.append(f"Within {pct_from_low:.1f}% of 52w low ${lo52:.2f} — at the floor")
    elif proximity <= FLOOR_NEARBY:
        score += 1
        reasons.append(f"{pct_from_low:.1f}% above 52w low ${lo52:.2f}")

    # 3. RSI position (0-3 pts)
    lo, hi = RECOVERING_RSI
    if was_oversold and lo <= rsi_now <= hi:
        score += 3
        reasons.append(f"RSI {rsi_now} — was oversold, now recovering (institutional signal)")
    elif rsi_now < OVERSOLD_RSI:
        score += 2
        reasons.append(f"RSI {rsi_now} — actively oversold, accumulation zone")
    elif rsi_now < 45:
        score += 1
        reasons.append(f"RSI {rsi_now} — below midline, room to run")

    # 4. Volume expansion (0-2 pts)
    if vol_ratio >= VOL_STRONG:
        score += 2
        reasons.append(f"Volume {vol_ratio:.2f}x 30d avg — unusual accumulation")
    elif vol_ratio >= VOL_MILD:
        score += 1
        reasons.append(f"Volume {vol_ratio:.2f}x 30d avg — above-average interest")

    return {
        "ticker": ticker, "spot": round(spot, 2), "change_pct": chg_pct,
        "high_52w": round(hi52, 2), "low_52w": round(lo52, 2),
        "pct_from_high": pct_from_high, "pct_from_low": pct_from_low,
        "rsi": rsi_now, "rsi_was_oversold": was_oversold, "vol_ratio": vol_ratio,
        "price_score": score, "reasons": reasons,
    }


def _build_angel(p: dict, opts: dict) -> FallenAngel:
    score   = p["price_score"]
    reasons = list(p["reasons"])

    # 5. Options flow (0-3 pts) — the leading indicator
    pc  = opts.get("put_call_ratio")
    pw  = opts.get("put_wall")
    spot = p["spot"]

    if pc is not None:
        if pc < PC_BULLISH:
            score += 2
            reasons.append(f"P/C ratio {pc:.2f} — calls dominating, smart money loading")
        elif pc < PC_ELEVATED:
            score += 1
            reasons.append(f"P/C ratio {pc:.2f} — elevated call activity")
        else:
            reasons.append(f"P/C ratio {pc:.2f}")

    if pw and spot:
        pct_pw = abs(spot - pw) / spot
        if pct_pw <= PUT_WALL_RANGE:
            score += 1
            reasons.append(f"GEX put wall ${pw:.0f} — dealer floor at current price")

    signal = "STRONG" if score >= STRONG_SCORE else ("WATCH" if score >= WATCH_SCORE else "NONE")

    return FallenAngel(
        ticker=p["ticker"], spot=p["spot"], change_pct=p["change_pct"],
        high_52w=p["high_52w"], low_52w=p["low_52w"],
        pct_from_high=p["pct_from_high"], pct_from_low=p["pct_from_low"],
        rsi=p["rsi"], rsi_was_oversold=p["rsi_was_oversold"], vol_ratio=p["vol_ratio"],
        put_call_ratio=opts.get("put_call_ratio"),
        put_wall=opts.get("put_wall"),
        call_wall=opts.get("call_wall"),
        options_regime=opts.get("regime"),
        score=score, signal=signal, reasons=reasons,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def scan_fallen_angels(universe: Optional[list[str]] = None) -> list[dict]:
    """
    Scan for fallen angel opportunities.
    Phase 1: price/RSI/volume screen across full universe.
    Phase 2: options flow (put/call ratio + GEX put wall) for qualifying candidates only.
    Returns list sorted by score desc, then pct_from_high (most beaten-down first).
    """
    tickers = universe or UNIVERSE
    logger.info("Fallen angel scan: %d tickers", len(tickers))

    # Phase 1 — price data
    raw = _fetch_all_prices(tickers)
    price_scored = []
    for ticker, data in raw.items():
        try:
            p = _score_price(ticker, data)
            if p:
                price_scored.append(p)
        except Exception as e:
            logger.warning("Price score failed for %s: %s", ticker, e)

    logger.info("Phase 1 complete: %d candidates qualify (≥30%% decline)", len(price_scored))

    # Phase 2 — options flow for qualifying candidates only
    opts_candidates = [(p["ticker"], p["spot"]) for p in price_scored
                       if p["price_score"] >= OPTIONS_MIN_SCORE]
    opts_data: dict[str, dict] = {}
    if opts_candidates:
        logger.info("Fetching options flow for %d candidates", len(opts_candidates))
        opts_data = _fetch_all_options(opts_candidates)

    results: list[FallenAngel] = []
    for p in price_scored:
        angel = _build_angel(p, opts_data.get(p["ticker"], {}))
        results.append(angel)

    results.sort(key=lambda a: (-a.score, a.pct_from_high))

    n_strong = sum(1 for a in results if a.signal == "STRONG")
    n_watch  = sum(1 for a in results if a.signal == "WATCH")
    logger.info("Fallen angels: %d STRONG, %d WATCH, %d candidates total",
                n_strong, n_watch, len(results))

    return [_angel_dict(a) for a in results]


def _angel_dict(a: FallenAngel) -> dict:
    return {
        "ticker":           a.ticker,
        "spot":             a.spot,
        "change_pct":       a.change_pct,
        "high_52w":         a.high_52w,
        "low_52w":          a.low_52w,
        "pct_from_high":    a.pct_from_high,
        "pct_from_low":     a.pct_from_low,
        "rsi":              a.rsi,
        "rsi_was_oversold": a.rsi_was_oversold,
        "vol_ratio":        a.vol_ratio,
        "put_call_ratio":   a.put_call_ratio,
        "put_wall":         a.put_wall,
        "call_wall":        a.call_wall,
        "options_regime":   a.options_regime,
        "score":            a.score,
        "signal":           a.signal,
        "reasons":          a.reasons,
    }


def fmt_slack(angels: list[dict], timestamp: str) -> str:
    lines = [
        f":fallen_leaf: *Fallen Angel Scanner — {timestamp}*",
        "_Beaten-down large-caps near multi-year floor with options flow confirmation_",
        "",
    ]
    strong = [a for a in angels if a["signal"] == "STRONG"]
    watch  = [a for a in angels if a["signal"] == "WATCH"]

    if not strong and not watch:
        lines.append("_No fallen angel setups today._")
        return "\n".join(lines)

    if strong:
        lines.append(":rotating_light: *HIGH CONVICTION — Floor + Smart Money Signal*")
        for a in strong:
            pc_str = f"  P/C {a['put_call_ratio']:.2f}" if a["put_call_ratio"] else ""
            pw_str = f"  PW ${a['put_wall']:.0f}" if a["put_wall"] else ""
            lines += [
                f":red_circle: *{a['ticker']} ${a['spot']:.2f}*"
                f"  ★ {a['score']}/13  |  {a['pct_from_high']:.1f}% from top{pc_str}{pw_str}",
            ]
            for r in a["reasons"]:
                lines.append(f"  ✓ {r}")
            lines.append("")

    if watch:
        lines.append(":eyes: *WATCH — Developing Setup*")
        for a in watch:
            pc_str = f"  P/C {a['put_call_ratio']:.2f}" if a["put_call_ratio"] else ""
            lines += [
                f":large_yellow_circle: *{a['ticker']} ${a['spot']:.2f}*"
                f"  ◎ {a['score']}/13  |  {a['pct_from_high']:.1f}% from top{pc_str}",
            ]
            for r in a["reasons"]:
                lines.append(f"  ✓ {r}")
            lines.append("")

    return "\n".join(lines)
