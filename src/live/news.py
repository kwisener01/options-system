"""
News + event awareness for the trader.

Pulls recent headlines from Alpaca's news API (Benzinga feed, included with the
existing API keys — no new vendor or secret). Used three ways:

  1. earnings guard  — `earnings_guard()` flags an underlying that has an
     earnings/guidance catalyst in recent headlines, so the scanners can annotate
     or suppress premium-selling setups.
  2. FOMC guard      — `fomc_guard()` detects upcoming Fed rate decisions,
     minutes releases, and Powell speeches that inject gap risk into credit
     spreads.
  3. dashboard feed  — `news_for()` returns scored headlines for the web page.

Everything is wrapped defensively: if the news endpoint is unavailable the
trader keeps working, just without the news overlay.
"""
import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# Cheap lexical sentiment — enough to colour a headline, not a trading signal.
_POS = {"beats", "beat", "surge", "soar", "jumps", "rally", "upgrade", "upgraded",
        "raises", "raised", "record", "tops", "growth", "strong", "outperform",
        "buyback", "approval", "wins", "gains", "bullish"}
_NEG = {"miss", "misses", "plunge", "plunges", "falls", "drop", "drops", "cuts",
        "cut", "downgrade", "downgraded", "lawsuit", "probe", "investigation",
        "recall", "warns", "warning", "weak", "layoffs", "bearish", "slump", "halts"}

# Words that mark a scheduled/again-pending earnings or guidance catalyst.
_EARNINGS = {"earnings", "q1", "q2", "q3", "q4", "quarterly", "guidance",
             "results", "eps", "revenue forecast", "to report", "reports earnings"}

# Words that mark FOMC / Fed rate-decision / monetary-policy catalysts.
_FOMC = {"fomc", "federal reserve", "fed meeting", "rate decision", "rate hike",
         "rate cut", "interest rate", "fed minutes", "powell", "monetary policy",
         "basis points", "fed holds", "fed pause", "quantitative tightening",
         "dot plot", "fed statement", "policy rate", "funds rate"}


def _client():
    from alpaca.data.historical.news import NewsClient
    from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY
    return NewsClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)


def _score(text: str) -> str:
    t = text.lower()
    pos = sum(w in t for w in _POS)
    neg = sum(w in t for w in _NEG)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def fetch_news(symbols, limit: int = 10, lookback_hours: int = 48) -> list[dict]:
    """Recent headlines for one or more symbols, newest first. Returns [] on any
    failure (never raises)."""
    if isinstance(symbols, str):
        symbols = [symbols]
    symbols = [s.upper() for s in symbols if s]
    if not symbols:
        return []
    try:
        from alpaca.data.requests import NewsRequest
        start = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        req = NewsRequest(symbols=",".join(symbols), start=start, limit=min(limit, 50),
                          include_content=False, sort="desc")
        resp = _client().get_news(req)
        items = resp.data.get("news", []) if hasattr(resp, "data") else []
        out = []
        for n in items:
            head = getattr(n, "headline", "") or ""
            summ = getattr(n, "summary", "") or ""
            created = getattr(n, "created_at", None)
            out.append({
                "headline": head,
                "summary": summ,
                "source": getattr(n, "source", "") or "",
                "url": getattr(n, "url", "") or "",
                "symbols": list(getattr(n, "symbols", []) or []),
                "created_at": created.isoformat() if hasattr(created, "isoformat") else str(created),
                "sentiment": _score(f"{head} {summ}"),
            })
        return out
    except Exception as e:
        logger.warning("fetch_news failed for %s: %s", symbols, e)
        return []


def news_for(symbols, limit_each: int = 5) -> dict:
    """Map of symbol -> recent scored headlines, for the dashboard."""
    if isinstance(symbols, str):
        symbols = [symbols]
    out = {}
    for s in symbols:
        items = fetch_news(s, limit=limit_each)
        if items:
            out[s.upper()] = items
    return out


def earnings_guard(symbol: str, lookback_hours: int = 72) -> dict:
    """Flag whether `symbol` has an earnings/guidance catalyst in recent
    headlines. Returns {flagged, reason, headlines, net_sentiment}. Defensive:
    an unavailable feed yields flagged=False (fail-open — never blocks a trade
    silently on an outage)."""
    items = fetch_news(symbol, limit=15, lookback_hours=lookback_hours)
    if not items:
        return {"flagged": False, "reason": "", "headlines": [], "net_sentiment": "neutral"}

    earn = [n for n in items
            if any(k in f"{n['headline']} {n['summary']}".lower() for k in _EARNINGS)]
    pos = sum(n["sentiment"] == "positive" for n in items)
    neg = sum(n["sentiment"] == "negative" for n in items)
    net = "positive" if pos > neg else "negative" if neg > pos else "neutral"

    if earn:
        return {
            "flagged": True,
            "reason": f"earnings/guidance headline in last {lookback_hours}h",
            "headlines": [n["headline"] for n in earn[:3]],
            "net_sentiment": net,
        }
    return {"flagged": False, "reason": "", "headlines": [n["headline"] for n in items[:3]],
            "net_sentiment": net}


def fomc_guard(lookback_hours: int = 96) -> dict:
    """Flag whether FOMC / Fed rate decision / Powell speech is imminent.
    Scans broad-market symbols (SPY, QQQ, DIA) for Fed-related headlines.
    Returns {flagged, reason, headlines}. Fail-open on outage."""
    items = []
    for sym in ("SPY", "QQQ", "DIA"):
        items.extend(fetch_news(sym, limit=20, lookback_hours=lookback_hours))

    if not items:
        return {"flagged": False, "reason": "", "headlines": []}

    # Deduplicate by headline text
    seen = set()
    unique = []
    for n in items:
        h = n["headline"]
        if h not in seen:
            seen.add(h)
            unique.append(n)

    hits = [n for n in unique
            if any(k in f"{n['headline']} {n['summary']}".lower() for k in _FOMC)]

    if hits:
        return {
            "flagged": True,
            "reason": f"FOMC/Fed headline in last {lookback_hours}h ({len(hits)} article{'s' if len(hits)!=1 else ''})",
            "headlines": [n["headline"] for n in hits[:5]],
        }
    return {"flagged": False, "reason": "", "headlines": []}
