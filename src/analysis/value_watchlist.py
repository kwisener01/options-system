"""
Value Stock Watchlist — buy signal scanner.

Phase 1 (parallel): fetch price histories + earnings dates + fundamental metrics
Phase 2 (parallel): per-stock GEX computation + technical + value scoring

Technical scoring (max 9):
  SPY positive gamma    : 2 pts
  Above 200MA           : 1–2 pts
  RSI < 50              : 1–2 pts
  Near own put wall     : 1–2 pts
  No earnings < 14d     : 1 pt

Fundamental value scoring (max 6):
  Forward/trailing P/E  : 0–2 pts  (< 12 = cheap, 12–18 = fair value)
  Price/Book            : 0–2 pts  (< 1.5 = near book, 1.5–2.5 = moderate)
  Dividend yield >= 3%  : 1 pt
  FCF yield >= 5%       : 1 pt

Combined max: 15 pts
STRONG threshold: own_now >= 10, own_lower >= 11
WATCH  threshold: own_now >= 6,  own_lower >= 7
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

logger = logging.getLogger(__name__)

VALUE_WATCHLIST: dict[str, list[str]] = {
    "own_now":   ["BAC", "KO", "AXP", "CVX", "T", "BRK.B", "VTV", "SCHD"],
    "own_lower": ["PFE", "INTC", "KR", "SPLG", "VOO"],
}

# ETFs have no meaningful per-stock GEX or P/B on indicative feed
_ETF_TICKERS = {"VTV", "SCHD", "SPLG", "VOO", "SPY", "QQQ"}

STRONG_THRESHOLD = {"own_now": 10, "own_lower": 11}
WATCH_THRESHOLD  = {"own_now": 6,  "own_lower": 7}
EARNINGS_AVOID_DAYS = 14

SELL_RSI_HIGH      = 72
SELL_MA200_EXT_PCT = 25.0
SELL_EARNINGS_DAYS = 7


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class StockSignal:
    ticker:            str
    label:             str
    spot:              float
    change_pct:        float
    ma200:             Optional[float]
    above_ma200:       Optional[bool]
    pct_above_ma200:   Optional[float]
    rsi:               Optional[float]
    put_wall:          Optional[float]
    regime:            Optional[str]
    vanna_signal:      Optional[str]
    earnings_date:     Optional[str]
    earnings_days_out: Optional[int]
    earnings_clear:    bool
    # Fundamental value metrics (facts, not technicals)
    pe_trailing:       Optional[float]
    pe_forward:        Optional[float]
    pb:                Optional[float]
    div_yield:         Optional[float]   # percent, e.g. 3.5
    ev_ebitda:         Optional[float]
    fcf_yield:         Optional[float]   # percent
    score:             int               # combined tech + value (max 15)
    signal:            str               # STRONG | WATCH | NONE | SELL
    reasons:           list = field(default_factory=list)
    blockers:          list = field(default_factory=list)
    sell_triggers:     list = field(default_factory=list)


# ── Data fetchers ─────────────────────────────────────────────────────────────

def _fetch_history(ticker: str) -> list[float]:
    """Daily closes for past year via Yahoo Finance v8 (most recent last)."""
    import urllib3
    import requests as _req
    urllib3.disable_warnings()
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?interval=1d&range=1y")
    resp = _req.get(url, headers={"User-Agent": "Mozilla/5.0"},
                    timeout=15, verify=False).json()
    closes = resp["chart"]["result"][0]["indicators"]["quote"][0]["close"]
    return [c for c in closes if c is not None]


def _fetch_all_histories(tickers: list[str]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(_fetch_history, t): t for t in tickers}
        for f in as_completed(futures):
            t = futures[f]
            try:
                out[t] = f.result()
            except Exception as e:
                logger.warning("History failed for %s: %s", t, e)
    return out


def _fetch_earnings(ticker: str) -> tuple[Optional[str], Optional[int]]:
    """Return (earnings_date_str, days_out) via Yahoo Finance v10."""
    import urllib3
    import requests as _req
    urllib3.disable_warnings()
    try:
        url = (f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
               f"?modules=calendarEvents")
        resp = _req.get(url, headers={"User-Agent": "Mozilla/5.0"},
                        timeout=12, verify=False).json()
        result = (resp.get("quoteSummary", {}).get("result") or [])
        if not result:
            return None, None
        dates = (result[0].get("calendarEvents", {})
                           .get("earnings", {})
                           .get("earningsDate", []))
        if dates:
            ts = dates[0].get("raw")
            if ts:
                ed = datetime.utcfromtimestamp(ts).date()
                return str(ed), (ed - date.today()).days
    except Exception as e:
        logger.debug("Earnings failed for %s: %s", ticker, e)
    return None, None


def _fetch_all_earnings(tickers: list[str]) -> dict[str, tuple]:
    out: dict[str, tuple] = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(_fetch_earnings, t): t for t in tickers}
        for f in as_completed(futures):
            t = futures[f]
            try:
                out[t] = f.result()
            except Exception:
                out[t] = (None, None)
    return out


def _fetch_fundamentals(ticker: str) -> dict:
    """Fetch P/E, P/B, dividend yield, EV/EBITDA, FCF yield from Yahoo Finance."""
    import urllib3
    import requests as _req
    urllib3.disable_warnings()
    try:
        url = (f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
               f"?modules=defaultKeyStatistics,summaryDetail,financialData")
        resp = _req.get(url, headers={"User-Agent": "Mozilla/5.0"},
                        timeout=12, verify=False).json()
        result = (resp.get("quoteSummary", {}).get("result") or [])
        if not result:
            return {}
        r       = result[0]
        stats   = r.get("defaultKeyStatistics", {})
        summary = r.get("summaryDetail", {})
        fin     = r.get("financialData", {})

        pe_trail = (summary.get("trailingPE")         or {}).get("raw")
        pe_fwd   = (stats.get("forwardPE")            or {}).get("raw")
        pb       = (stats.get("priceToBook")          or {}).get("raw")
        dy_raw   = (summary.get("dividendYield")      or {}).get("raw")  # decimal: 0.035 = 3.5%
        ev_ebd   = (stats.get("enterpriseToEbitda")   or {}).get("raw")
        fcf      = (fin.get("freeCashflow")           or {}).get("raw")
        mkt_cap  = (summary.get("marketCap")          or {}).get("raw")

        fcf_yield = None
        if fcf and mkt_cap and mkt_cap > 0:
            fcf_yield = round(fcf / mkt_cap * 100, 1)

        return {
            "pe_trailing": pe_trail,
            "pe_forward":  pe_fwd,
            "pb":          pb,
            "div_yield":   round(dy_raw * 100, 2) if dy_raw is not None else None,
            "ev_ebitda":   ev_ebd,
            "fcf_yield":   fcf_yield,
        }
    except Exception as e:
        logger.debug("Fundamentals failed for %s: %s", ticker, e)
        return {}


def _fetch_all_fundamentals(tickers: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(_fetch_fundamentals, t): t for t in tickers}
        for f in as_completed(futures):
            t = futures[f]
            try:
                out[t] = f.result()
            except Exception:
                out[t] = {}
    return out


# ── Technical indicators ──────────────────────────────────────────────────────

def _ma200(closes: list[float]) -> Optional[float]:
    return sum(closes[-200:]) / 200 if len(closes) >= 200 else None


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


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score_ticker(label: str, spot: float, ma200: Optional[float],
                  rsi: Optional[float], gex: dict, spy_regime: str,
                  earn_days: Optional[int]) -> tuple:
    """Technical scoring only — max 9 pts."""
    score    = 0
    reasons  = []
    blockers = []

    # 1. SPY regime (2 pts)
    if spy_regime and "POSITIVE" in spy_regime:
        score += 2
        reasons.append("SPY positive gamma — market stable")
    else:
        blockers.append("SPY negative gamma — avoid bullish entries")

    # 2. Above 200MA (2 pts)
    if ma200 is not None:
        pct = (spot - ma200) / ma200 * 100
        if spot > ma200:
            pts = 2 if pct <= 12 else 1
            score += pts
            reasons.append(f"Above 200MA ${ma200:.2f} by {pct:.1f}%"
                           + (" — extended" if pts == 1 else ""))
        else:
            blockers.append(f"Below 200MA ${ma200:.2f} — downtrend")
    else:
        blockers.append("< 200 days of price history")

    # 3. RSI (2 pts)
    if rsi is not None:
        if rsi < 35:
            score += 2
            reasons.append(f"RSI {rsi} — oversold, strong entry")
        elif rsi < 50:
            score += 1
            reasons.append(f"RSI {rsi} — room to run")
        else:
            blockers.append(f"RSI {rsi} — overbought, wait for pullback")

    # 4. Put wall (2 pts)
    pw = gex.get("put_wall")
    if pw and pw > 0:
        pct_pw = (spot - pw) / pw * 100
        if 0 < pct_pw <= 3:
            score += 2
            reasons.append(f"Near put wall ${pw:.0f} — GEX structural support")
        elif 0 < pct_pw <= 8:
            score += 1
            reasons.append(f"Above put wall ${pw:.0f} by {pct_pw:.1f}%")
        elif pct_pw <= 0:
            blockers.append(f"Below put wall ${pw:.0f} — GEX floor breached")

    # 5. Earnings clear (1 pt)
    clear = earn_days is None or earn_days < 0 or earn_days > EARNINGS_AVOID_DAYS
    if clear:
        score += 1
        reasons.append("No earnings risk" +
                       (f" ({earn_days}d out)" if earn_days and earn_days > 0 else ""))
    else:
        blockers.append(f"Earnings in {earn_days}d — avoid until after report")

    return score, reasons, blockers, clear


def _score_value_fundamentals(fund: dict, is_etf: bool) -> tuple[int, list, list]:
    """
    Fundamental value scoring — max 6 pts.
    All inputs are reported facts from Yahoo Finance.
    """
    score    = 0
    reasons  = []
    blockers = []

    # P/E — forward preferred (based on next-year estimates), trailing as fallback
    pe = fund.get("pe_forward") or fund.get("pe_trailing")
    if pe is not None and pe > 0:
        tag = "Fwd P/E" if fund.get("pe_forward") else "P/E"
        if pe < 12:
            score += 2
            reasons.append(f"{tag} {pe:.1f} — deeply cheap")
        elif pe < 18:
            score += 1
            reasons.append(f"{tag} {pe:.1f} — fair value")
        elif pe > 25:
            blockers.append(f"{tag} {pe:.1f} — expensive for a value stock")
        # 18–25: neutral, no points, no blocker

    # P/B — skip ETFs (fund P/B is not the same concept as stock P/B)
    if not is_etf:
        pb = fund.get("pb")
        if pb is not None and pb > 0:
            if pb < 1.5:
                score += 2
                reasons.append(f"P/B {pb:.2f} — trading near book value")
            elif pb < 2.5:
                score += 1
                reasons.append(f"P/B {pb:.2f} — moderate book premium")
            elif pb > 4.0:
                blockers.append(f"P/B {pb:.2f} — expensive relative to book")

    # Dividend yield
    dy = fund.get("div_yield")
    if dy is not None:
        if dy >= 3.0:
            score += 1
            reasons.append(f"Yield {dy:.1f}% — income cushion")
        elif dy >= 1.5:
            reasons.append(f"Yield {dy:.1f}%")  # noted but not scored

    # FCF yield — skip ETFs (ETF FCF not meaningful)
    if not is_etf:
        fcfy = fund.get("fcf_yield")
        if fcfy is not None:
            if fcfy >= 5.0:
                score += 1
                reasons.append(f"FCF yield {fcfy:.1f}% — cash-generative")
            elif fcfy < 0:
                blockers.append(f"Negative FCF {fcfy:.1f}% — burning cash")

    return score, reasons, blockers


def _sell_check(spot: float, ma200: Optional[float], rsi: Optional[float],
                gex: dict, spy_regime: str, earn_days: Optional[int]) -> tuple[bool, list[str]]:
    """
    Returns (should_sell, sell_reasons).
    Fires when 2+ exit conditions are met — one alone is noise.
    """
    triggers: list[str] = []

    if rsi is not None and rsi >= SELL_RSI_HIGH:
        triggers.append(f"RSI {rsi} — overbought, momentum fading")

    if ma200 is not None and spot > ma200:
        pct = (spot - ma200) / ma200 * 100
        if pct >= SELL_MA200_EXT_PCT:
            triggers.append(f"{pct:.1f}% above 200MA — stretched, take profits")

    if spy_regime and "NEGATIVE" in spy_regime:
        triggers.append("SPY negative gamma — dealer amplification risk")

    pw = gex.get("put_wall")
    if pw and pw > 0 and spot < pw:
        pct_below = (pw - spot) / pw * 100
        triggers.append(f"Below put wall ${pw:.0f} by {pct_below:.1f}% — support broken")

    if earn_days is not None and 0 <= earn_days <= SELL_EARNINGS_DAYS:
        triggers.append(f"Earnings in {earn_days}d — take profits before report")

    return len(triggers) >= 2, triggers


# ── Per-ticker scan ───────────────────────────────────────────────────────────

def _scan_one(ticker: str, label: str,
              closes: list[float],
              earn_info: tuple,
              fundamentals: dict,
              spy_regime: str,
              vix_now: float, vix_prev: float) -> Optional[StockSignal]:
    if not closes:
        return None
    try:
        spot      = closes[-1]
        prev      = closes[-2] if len(closes) > 1 else spot
        chg_pct   = round((spot - prev) / prev * 100, 2) if prev else 0.0
        ma200_val = _ma200(closes)
        rsi_val   = _rsi(closes)
        above_ma  = (spot > ma200_val) if ma200_val else None
        pct_ma    = round((spot - ma200_val) / ma200_val * 100, 2) if ma200_val else None
        earn_date, earn_days = earn_info
        is_etf    = ticker in _ETF_TICKERS

        # Per-stock GEX (skip ETFs — no meaningful chain on indicative feed)
        gex: dict = {}
        if not is_etf:
            try:
                from src.live.alpaca_options import fetch_chain_combined
                from src.analysis.gex_scanner import compute_exposures
                chain = fetch_chain_combined(ticker, spot)
                if len(chain["gex_contracts"]) >= 10:
                    r = compute_exposures(spot, vix_now, vix_prev, chain["gex_contracts"])
                    gex = {
                        "put_wall":     r.put_wall,
                        "call_wall":    r.call_wall,
                        "regime":       r.gex_regime,
                        "vanna_signal": r.vanna_signal,
                    }
            except Exception as e:
                logger.debug("GEX failed for %s: %s", ticker, e)

        # Technical score (max 9)
        tech_score, tech_reasons, tech_blockers, earn_clear = _score_ticker(
            label, spot, ma200_val, rsi_val, gex, spy_regime, earn_days
        )

        # Fundamental value score (max 6) — gracefully absent if data unavailable
        fund_available = bool(fundamentals)
        val_score, val_reasons, val_blockers = _score_value_fundamentals(fundamentals, is_etf)

        score    = tech_score + val_score
        reasons  = tech_reasons + val_reasons
        blockers = tech_blockers + val_blockers

        # Sell check overrides buy signal — exit takes priority
        should_sell, sell_triggers = _sell_check(
            spot, ma200_val, rsi_val, gex, spy_regime, earn_days
        )

        # Fall back to original 9-pt thresholds when fundamental data is unavailable
        if fund_available:
            thresh_s = STRONG_THRESHOLD.get(label, 10)
            thresh_w = WATCH_THRESHOLD.get(label, 6)
        else:
            thresh_s = {"own_now": 6, "own_lower": 7}.get(label, 6)
            thresh_w = {"own_now": 4, "own_lower": 5}.get(label, 4)
        signal   = "STRONG" if score >= thresh_s else ("WATCH" if score >= thresh_w else "NONE")
        if should_sell:
            signal = "SELL"

        return StockSignal(
            ticker=ticker, label=label,
            spot=round(spot, 2), change_pct=chg_pct,
            ma200=round(ma200_val, 2) if ma200_val else None,
            above_ma200=above_ma, pct_above_ma200=pct_ma,
            rsi=rsi_val,
            put_wall=gex.get("put_wall"), regime=gex.get("regime"),
            vanna_signal=gex.get("vanna_signal"),
            earnings_date=earn_date, earnings_days_out=earn_days,
            earnings_clear=earn_clear,
            pe_trailing=fundamentals.get("pe_trailing"),
            pe_forward=fundamentals.get("pe_forward"),
            pb=fundamentals.get("pb"),
            div_yield=fundamentals.get("div_yield"),
            ev_ebitda=fundamentals.get("ev_ebitda"),
            fcf_yield=fundamentals.get("fcf_yield"),
            score=score, signal=signal, reasons=reasons, blockers=blockers,
            sell_triggers=sell_triggers,
        )
    except Exception as e:
        logger.warning("Scan failed for %s: %s", ticker, e)
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def scan_watchlist(spy_regime: str = "UNKNOWN",
                   vix_now: float = 0.0,
                   vix_prev: float = 0.0) -> list[dict]:
    """
    Scan all value watchlist tickers.
    Returns list of signal dicts sorted STRONG -> WATCH -> NONE.
    """
    tasks = [(t, lbl)
             for lbl, tickers in VALUE_WATCHLIST.items()
             for t in tickers]
    all_tickers = [t for t, _ in tasks]

    logger.info("Value watchlist scan: %d tickers", len(all_tickers))

    # Phase 1: histories, earnings, fundamentals — all parallel
    with ThreadPoolExecutor(max_workers=3) as pre:
        hist_f = pre.submit(_fetch_all_histories,    all_tickers)
        earn_f = pre.submit(_fetch_all_earnings,     all_tickers)
        fund_f = pre.submit(_fetch_all_fundamentals, all_tickers)
    histories    = hist_f.result()
    earnings     = earn_f.result()
    fundamentals = fund_f.result()

    # Phase 2: GEX + scoring per ticker in parallel
    results: list[StockSignal] = []
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {
            ex.submit(
                _scan_one,
                ticker, label,
                histories.get(ticker, []),
                earnings.get(ticker, (None, None)),
                fundamentals.get(ticker, {}),
                spy_regime, vix_now, vix_prev,
            ): ticker
            for ticker, label in tasks
        }
        for f in as_completed(futures):
            try:
                sig = f.result()
                if sig:
                    results.append(sig)
            except Exception as e:
                logger.error("Watchlist future error: %s", e)

    _order = {"STRONG": 0, "WATCH": 1, "NONE": 2}
    results.sort(key=lambda s: (_order.get(s.signal, 9), s.ticker))

    n_strong = sum(1 for s in results if s.signal == "STRONG")
    n_watch  = sum(1 for s in results if s.signal == "WATCH")
    logger.info("Watchlist complete: %d STRONG, %d WATCH, %d NONE",
                n_strong, n_watch, len(results) - n_strong - n_watch)

    return [_as_dict(s) for s in results]


def _as_dict(s: StockSignal) -> dict:
    return {
        "ticker":            s.ticker,
        "label":             s.label,
        "spot":              s.spot,
        "change_pct":        s.change_pct,
        "ma200":             s.ma200,
        "above_ma200":       s.above_ma200,
        "pct_above_ma200":   s.pct_above_ma200,
        "rsi":               s.rsi,
        "put_wall":          s.put_wall,
        "regime":            s.regime,
        "vanna_signal":      s.vanna_signal,
        "earnings_date":     s.earnings_date,
        "earnings_days_out": s.earnings_days_out,
        "earnings_clear":    s.earnings_clear,
        "pe_trailing":       s.pe_trailing,
        "pe_forward":        s.pe_forward,
        "pb":                s.pb,
        "div_yield":         s.div_yield,
        "ev_ebitda":         s.ev_ebitda,
        "fcf_yield":         s.fcf_yield,
        "score":             s.score,
        "signal":            s.signal,
        "reasons":           s.reasons,
        "blockers":          s.blockers,
        "sell_triggers":     s.sell_triggers,
    }


# ── Slack formatter ───────────────────────────────────────────────────────────

def fmt_slack(signals: list[dict], spy_regime: str, vix_now: float,
              timestamp: str) -> str:
    lines = [
        f":chart_with_upwards_trend: *Value Watchlist — {timestamp}*",
        f"SPY: `{spy_regime.replace('_', ' ').title()}` | VIX {vix_now:.2f}",
        "",
    ]

    sell   = [s for s in signals if s["signal"] == "SELL"]
    strong = [s for s in signals if s["signal"] == "STRONG"]
    watch  = [s for s in signals if s["signal"] == "WATCH"]

    if sell:
        lines.append(":rotating_light: *SELL / ROLL ALERTS*")
        for s in sell:
            lines += [
                f":red_circle: *{s['ticker']} ${s['spot']:.2f}*"
                f"  ({s['label'].replace('_',' ').title()})  ▼ SELL  {s['score']}/15",
            ]
            for t in s.get("sell_triggers", []):
                lines.append(f"  ⚠ {t}")
            lines.append("")

    if not strong and not watch:
        lines.append("_No buy signals today — conditions not met across watchlist._")
        return "\n".join(lines)

    for s in strong:
        earn_str = (f"  · Earnings {s['earnings_date']} ({s['earnings_days_out']}d)"
                    if s["earnings_date"] else "")
        lines += [
            f":white_check_mark: *{s['ticker']} ${s['spot']:.2f}*"
            f"  ({s['label'].replace('_',' ').title()})  ★ BUY NOW  {s['score']}/15",
        ]
        for r in s["reasons"]:
            lines.append(f"  ✓ {r}")
        for b in s["blockers"]:
            lines.append(f"  ✗ {b}")
        if earn_str:
            lines.append(earn_str)
        lines.append("")

    for s in watch:
        lines += [
            f":eyes: *{s['ticker']} ${s['spot']:.2f}*"
            f"  ({s['label'].replace('_',' ').title()})  ◎ ACCUMULATE  {s['score']}/15",
        ]
        for r in s["reasons"]:
            lines.append(f"  ✓ {r}")
        for b in s["blockers"][:2]:
            lines.append(f"  ✗ {b}")
        lines.append("")

    return "\n".join(lines)
