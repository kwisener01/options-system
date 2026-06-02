"""
Fallen Angel Scanner — institutional-grade bottom-fishing screen.

Finds large-cap S&P 500 stocks that have suffered major drawdowns and are
currently near their multi-year floor, with early signs of institutional
accumulation (volume expansion, RSI bottoming).

The INTC pattern:
  - Once-dominant company
  - Stock crushed 50-70% from multi-year high
  - Trading near book value (or below)
  - Catalyst exists: new management, restructuring, M&A potential, sector rotation
  - Smart money quietly accumulating at the floor

Scoring (max 10):
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
    "VNO", "SLG", "MPW", "WPG", "NYCB", "KIM",
]

FLOOR_THRESHOLD  = 0.10  # within 10% of 52w low = "near floor"
FLOOR_NEARBY     = 0.20  # within 20% of 52w low = "nearby"
DECLINE_DEEP     = 0.50  # down 50%+ from 52w high
DECLINE_MID      = 0.40  # down 40-50%
DECLINE_SOFT     = 0.30  # down 30-40%
OVERSOLD_RSI     = 30
RECOVERING_RSI   = (35, 55)  # recovering band after oversold
VOL_STRONG       = 1.25  # 10d avg 25%+ above 30d avg
VOL_MILD         = 1.10  # 10d avg 10%+ above 30d avg

STRONG_SCORE = 7
WATCH_SCORE  = 4


@dataclass
class FallenAngel:
    ticker:          str
    spot:            float
    change_pct:      float
    high_52w:        float
    low_52w:         float
    pct_from_high:   float   # negative — how far below 52w high
    pct_from_low:    float   # positive — how far above 52w low
    rsi:             float
    rsi_was_oversold: bool   # RSI < 30 in past 20 sessions
    vol_ratio:       float   # 10d / 30d avg volume ratio
    score:           int
    signal:          str     # STRONG | WATCH | NONE
    reasons:         list = field(default_factory=list)


def _fetch_history_with_volume(ticker: str) -> dict:
    """Fetch 1-year daily OHLCV via Yahoo Finance v8."""
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


def _fetch_all(tickers: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_fetch_history_with_volume, t): t for t in tickers}
        for f in as_completed(futures):
            t = futures[f]
            try:
                out[t] = f.result()
            except Exception as e:
                logger.debug("Fetch failed for %s: %s", t, e)
    return out


def _rsi(closes: list[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    ch = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains  = [max(c, 0) for c in ch]
    losses = [max(-c, 0) for c in ch]
    ag = sum(gains[:period])  / period
    al = sum(losses[:period]) / period
    for i in range(period, len(ch)):
        ag = (ag * (period - 1) + gains[i])  / period
        al = (al * (period - 1) + losses[i]) / period
    if al == 0:
        return 100.0
    return round(100 - (100 / (1 + ag / al)), 1)


def _rsi_series(closes: list[float], period: int = 14) -> list[float]:
    """Compute RSI for each point — needed to check historical oversold."""
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


def _score_angel(ticker: str, data: dict) -> Optional[FallenAngel]:
    closes  = data.get("closes", [])
    volumes = data.get("volumes", [])
    if len(closes) < 50 or not data.get("high_52w") or not data.get("low_52w"):
        return None

    spot     = closes[-1]
    prev     = closes[-2] if len(closes) > 1 else spot
    chg_pct  = round((spot - prev) / prev * 100, 2) if prev else 0.0
    hi52     = data["high_52w"]
    lo52     = data["low_52w"]

    pct_from_high = round((spot - hi52) / hi52 * 100, 1)  # negative
    pct_from_low  = round((spot - lo52) / lo52 * 100, 1)  # positive

    # Must be down at least 30% from 52-week high to qualify as "fallen"
    if pct_from_high > -DECLINE_SOFT * 100:
        return None

    rsi_series = _rsi_series(closes)
    rsi_now    = rsi_series[-1] if rsi_series else 50.0
    recent_rsi = rsi_series[-20:] if len(rsi_series) >= 20 else rsi_series
    was_oversold = any(r < OVERSOLD_RSI for r in recent_rsi)

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
        reasons.append(f"RSI {rsi_now} — was oversold, now recovering (smart money signal)")
    elif rsi_now < OVERSOLD_RSI:
        score += 2
        reasons.append(f"RSI {rsi_now} — actively oversold, accumulation zone")
    elif rsi_now < 45:
        score += 1
        reasons.append(f"RSI {rsi_now} — below midline, room to run")

    # 4. Volume expansion (0-2 pts) — institutions buy quietly at the floor
    if vol_ratio >= VOL_STRONG:
        score += 2
        reasons.append(f"Volume {vol_ratio:.2f}x 30d avg — unusual accumulation")
    elif vol_ratio >= VOL_MILD:
        score += 1
        reasons.append(f"Volume {vol_ratio:.2f}x 30d avg — above-average interest")

    signal = "STRONG" if score >= STRONG_SCORE else ("WATCH" if score >= WATCH_SCORE else "NONE")

    return FallenAngel(
        ticker=ticker, spot=round(spot, 2), change_pct=chg_pct,
        high_52w=round(hi52, 2), low_52w=round(lo52, 2),
        pct_from_high=pct_from_high, pct_from_low=pct_from_low,
        rsi=rsi_now, rsi_was_oversold=was_oversold, vol_ratio=vol_ratio,
        score=score, signal=signal, reasons=reasons,
    )


def scan_fallen_angels(universe: Optional[list[str]] = None,
                       min_decline_pct: float = 30.0) -> list[dict]:
    """
    Scan for fallen angel opportunities.

    Args:
        universe: list of tickers to scan (default: built-in S&P 500 large-cap list)
        min_decline_pct: minimum % decline from 52-week high to qualify (default 30%)

    Returns:
        List of angel dicts sorted by score desc, then by pct_from_high (most beaten-down first).
    """
    tickers = universe or UNIVERSE
    logger.info("Fallen angel scan: %d tickers (min decline %.0f%%)", len(tickers), min_decline_pct)

    raw = _fetch_all(tickers)

    results: list[FallenAngel] = []
    for ticker, data in raw.items():
        try:
            angel = _score_angel(ticker, data)
            if angel:
                results.append(angel)
        except Exception as e:
            logger.warning("Score failed for %s: %s", ticker, e)

    results.sort(key=lambda a: (-a.score, a.pct_from_high))

    n_strong = sum(1 for a in results if a.signal == "STRONG")
    n_watch  = sum(1 for a in results if a.signal == "WATCH")
    logger.info("Fallen angels: %d STRONG, %d WATCH, %d NONE of %d candidates",
                n_strong, n_watch,
                len(results) - n_strong - n_watch, len(tickers))

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
        "score":            a.score,
        "signal":           a.signal,
        "reasons":          a.reasons,
    }


def fmt_slack(angels: list[dict], timestamp: str) -> str:
    """Format fallen angel results for Slack."""
    lines = [
        f":fallen_leaf: *Fallen Angel Scanner — {timestamp}*",
        "_Beaten-down large-caps near multi-year floors_",
        "",
    ]

    strong = [a for a in angels if a["signal"] == "STRONG"]
    watch  = [a for a in angels if a["signal"] == "WATCH"]

    if not strong and not watch:
        lines.append("_No fallen angel setups today — no stocks near floor with volume confirmation._")
        return "\n".join(lines)

    if strong:
        lines.append(":rotating_light: *HIGH CONVICTION — Near Floor + Accumulation Signal*")
        for a in strong:
            lines += [
                f":red_circle: *{a['ticker']} ${a['spot']:.2f}*"
                f"  ★ {a['score']}/10  |  52w: ${a['low_52w']:.2f}–${a['high_52w']:.2f}"
                f"  |  {a['pct_from_high']:.1f}% from top",
            ]
            for r in a["reasons"]:
                lines.append(f"  ✓ {r}")
            lines.append("")

    if watch:
        lines.append(":eyes: *WATCH LIST — Developing Setup*")
        for a in watch:
            lines += [
                f":large_yellow_circle: *{a['ticker']} ${a['spot']:.2f}*"
                f"  ◎ {a['score']}/10  |  {a['pct_from_high']:.1f}% from 52w top",
            ]
            for r in a["reasons"]:
                lines.append(f"  ✓ {r}")
            lines.append("")

    return "\n".join(lines)
