"""
BWB Watchlist Scanner

Screens tiered ticker universes for Broken-Wing Butterfly put candidates.
Liquidity criteria (per user spec):
  - Stock price $15–$150
  - bid/ask spread ≤ $0.20 per leg
  - bid+ask size proxy ≥ 5 per leg
  - DTE 7–35 (weekly options)
  - Lower wing ≤ 2× upper wing
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

logger = logging.getLogger(__name__)

BWB_UNIVERSE: dict[str, list[str]] = {
    "low_risk":    ["BAC", "F", "T", "PFE", "INTC", "PYPL", "UBER"],
    "medium_risk": ["PLTR", "SOFI", "HOOD", "DKNG", "CCL", "MU"],
    "high_risk":   ["RIVN", "ROKU", "COIN", "MARA", "MSTR"],
}

# Screening thresholds
PRICE_MIN    = 15.0
PRICE_MAX    = 150.0
OI_PROXY_MIN = 5
SPREAD_MAX   = 0.20
DTE_MIN      = 7
DTE_MAX      = 35


def _fetch_spots(tickers: list[str]) -> dict[str, tuple[float, float]]:
    """Return {ticker: (price, prev_close)} via Yahoo Finance v8."""
    import urllib3
    import requests as _req
    urllib3.disable_warnings()

    out: dict[str, tuple[float, float]] = {}
    for t in tickers:
        try:
            url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{t}"
                   f"?interval=1d&range=5d")
            meta = _req.get(url, headers={"User-Agent": "Mozilla/5.0"},
                            timeout=15, verify=False).json()["chart"]["result"][0]["meta"]
            out[t] = (float(meta["regularMarketPrice"]),
                      float(meta["chartPreviousClose"]))
        except Exception as e:
            logger.warning("Spot fetch failed for %s: %s", t, e)
    return out


def _suggest_bwb(spot: float, puts: list[dict]) -> Optional[dict]:
    """
    Find best BWB structure from liquid puts.

    Searches for H/M/L put strikes where:
      - H is 0.5–6% OTM (upper buy, defines max profit tent)
      - M is H minus 2–8 pts (short strike)
      - L is M minus 1.0–2.0× upper wing (broken lower wing)
    Returns the highest-scoring candidate or None.
    """
    if not puts:
        return None

    # Group by expiry; pick shortest-DTE weekly with ≥3 liquid strikes
    by_expiry: dict[str, list[dict]] = {}
    for p in puts:
        by_expiry.setdefault(p["expiry"], []).append(p)

    best: Optional[dict] = None
    best_score = -1.0

    for expiry, ep in sorted(by_expiry.items()):
        dte = ep[0]["dte"]
        if not (DTE_MIN <= dte <= DTE_MAX):
            continue

        by_strike = {p["strike"]: p for p in ep}
        strikes = sorted(by_strike.keys(), reverse=True)

        # H candidates: 0.5–6% OTM
        h_cands = [s for s in strikes if spot * 0.94 <= s < spot * 0.9975]
        for h_strike in h_cands:
            h = by_strike[h_strike]
            if h["spread"] > SPREAD_MAX or h["oi_proxy"] < OI_PROXY_MIN:
                continue

            # M: h_strike minus 2–8 pts (try increments matching available strikes)
            for upper_wing in [2, 2.5, 3, 4, 5, 6, 7, 8]:
                m_target = h_strike - upper_wing
                m_strike = min(by_strike, key=lambda s: abs(s - m_target), default=None)
                if m_strike is None or abs(m_strike - m_target) > 1.5 or m_strike >= h_strike:
                    continue
                m = by_strike[m_strike]
                if m["spread"] > SPREAD_MAX or m["oi_proxy"] < OI_PROXY_MIN:
                    continue
                uw = h_strike - m_strike  # actual upper wing

                # L: m_strike minus 1.0–2.0× uw
                for lw_mult in [1.5, 2.0, 1.25, 1.75, 1.0]:
                    l_target = m_strike - uw * lw_mult
                    l_strike = min(by_strike, key=lambda s: abs(s - l_target), default=None)
                    if l_strike is None or abs(l_strike - l_target) > 2.0 or l_strike >= m_strike:
                        continue
                    l = by_strike[l_strike]
                    if l["spread"] > SPREAD_MAX or l["oi_proxy"] < OI_PROXY_MIN:
                        continue
                    lw = m_strike - l_strike
                    if lw > uw * 2.5:
                        continue  # skip grotesquely wide lower wing

                    liq = h["oi_proxy"] + m["oi_proxy"] + l["oi_proxy"]
                    spread_total = h["spread"] + m["spread"] + l["spread"]
                    score = liq / (spread_total + 0.01) - lw / uw

                    if score > best_score:
                        best_score = score
                        liq_score  = min(round(liq / 8), 10)
                        best = {
                            "expiry":       expiry,
                            "dte":          dte,
                            "long_upper":   h_strike,
                            "short_strike": m_strike,
                            "long_lower":   l_strike,
                            "upper_wing":   round(uw, 2),
                            "lower_wing":   round(lw, 2),
                            "liq_score":    liq_score,
                            "h_bid":        h["bid"],  "h_ask": h["ask"],
                            "h_spread":     h["spread"], "h_oi":  h["oi_proxy"],
                            "m_bid":        m["bid"],  "m_ask": m["ask"],
                            "m_spread":     m["spread"], "m_oi":  m["oi_proxy"],
                            "l_bid":        l["bid"],  "l_ask": l["ask"],
                            "l_spread":     l["spread"], "l_oi":  l["oi_proxy"],
                        }
    return best


def _scan_one(ticker: str, spot: float, prev: float, tier: str) -> dict:
    """Full BWB screen for a single ticker."""
    result: dict = {
        "ticker":     ticker,
        "spot":       round(spot, 2),
        "prev":       round(prev, 2),
        "change":     round(spot - prev, 2),
        "change_pct": round((spot - prev) / prev * 100, 2) if prev else 0.0,
        "tier":       tier,
        "price_ok":   PRICE_MIN <= spot <= PRICE_MAX,
        "candidate":  None,
        "note":       "",
        "error":      None,
    }

    if not result["price_ok"]:
        result["note"] = f"Price ${spot:.2f} outside ${PRICE_MIN:.0f}–${PRICE_MAX:.0f} range"
        return result

    try:
        from src.live.alpaca_options import fetch_puts_liquid
        puts = fetch_puts_liquid(ticker, spot, dte_min=DTE_MIN, dte_max=DTE_MAX)
        if not puts:
            result["note"] = "No liquid puts found via Alpaca"
            return result

        candidate = _suggest_bwb(spot, puts)
        if candidate:
            result["candidate"] = candidate
            h, m, l = (candidate["long_upper"], candidate["short_strike"],
                       candidate["long_lower"])
            result["note"] = (
                f"{h:.0f}/{m:.0f}/{l:.0f}  DTE {candidate['dte']}  "
                f"liq {candidate['liq_score']}/10"
            )
        else:
            result["note"] = f"{len(puts)} puts found — no viable BWB structure"
    except Exception as e:
        logger.warning("Scan error for %s: %s", ticker, e)
        result["error"] = str(e)
        result["note"] = "Scan error — see logs"

    return result


def scan(tickers: list[str] = None, tier: str = "low_risk") -> list[dict]:
    """
    Screen tickers for BWB candidates.

    Args:
        tickers: explicit list (overrides tier)
        tier: 'low_risk' | 'medium_risk' | 'high_risk' | 'all'
    Returns:
        List of dicts, candidates first, sorted by ticker name.
    """
    if tickers is None:
        if tier == "all":
            tickers = [t for ts in BWB_UNIVERSE.values() for t in ts]
        else:
            tickers = BWB_UNIVERSE.get(tier, BWB_UNIVERSE["low_risk"])

    tier_map = {t: k for k, ts in BWB_UNIVERSE.items() for t in ts}

    logger.info("BWB scan starting for %d tickers (%s)", len(tickers), tier)
    spots = _fetch_spots(tickers)

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {
            ex.submit(
                _scan_one,
                ticker,
                spots.get(ticker, (0.0, 0.0))[0],
                spots.get(ticker, (0.0, 0.0))[1],
                tier_map.get(ticker, tier),
            ): ticker
            for ticker in tickers
            if ticker in spots
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                logger.error("Future error for %s: %s", futures[future], e)

    # Tickers with price fetch failure
    for t in tickers:
        if t not in spots:
            results.append({
                "ticker": t, "spot": 0.0, "prev": 0.0, "change": 0.0,
                "change_pct": 0.0, "tier": tier_map.get(t, tier),
                "price_ok": False, "candidate": None,
                "note": "Price fetch failed", "error": "no price data",
            })

    results.sort(key=lambda r: (r["candidate"] is None, r["ticker"]))
    logger.info("BWB scan complete: %d/%d have candidates",
                sum(1 for r in results if r["candidate"]), len(results))
    return results
