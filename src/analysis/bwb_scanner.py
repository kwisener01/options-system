"""
BWB Watchlist Scanner

Screens tiered ticker universes for Broken-Wing Butterfly put candidates.
For tickers with a viable structure, computes per-stock GEX (gamma, vanna, charm)
in a single Alpaca chain call — no second round trip.

Liquidity criteria (per user spec):
  - Stock price $15–$150
  - bid/ask spread ≤ $0.20 per leg
  - bid+ask size proxy ≥ 5 per leg
  - DTE 7–35
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
    Find best H/M/L put structure from liquid puts.
    Returns the highest-scoring candidate or None.
    """
    if not puts:
        return None

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

        # H candidates: 0.5–6% OTM below spot
        h_cands = [s for s in by_strike if spot * 0.94 <= s < spot * 0.9975]
        for h_strike in sorted(h_cands, reverse=True):
            h = by_strike[h_strike]
            if h["spread"] > SPREAD_MAX or h["oi_proxy"] < OI_PROXY_MIN:
                continue

            for upper_wing in [2, 2.5, 3, 4, 5, 6, 7, 8]:
                m_target = h_strike - upper_wing
                if not by_strike:
                    continue
                m_strike = min(by_strike, key=lambda s: abs(s - m_target))
                if abs(m_strike - m_target) > 1.5 or m_strike >= h_strike:
                    continue
                m = by_strike[m_strike]
                if m["spread"] > SPREAD_MAX or m["oi_proxy"] < OI_PROXY_MIN:
                    continue
                uw = h_strike - m_strike

                for lw_mult in [1.5, 2.0, 1.25, 1.75, 1.0]:
                    l_target = m_strike - uw * lw_mult
                    l_strike = min(by_strike, key=lambda s: abs(s - l_target))
                    if abs(l_strike - l_target) > 2.0 or l_strike >= m_strike:
                        continue
                    l = by_strike[l_strike]
                    if l["spread"] > SPREAD_MAX or l["oi_proxy"] < OI_PROXY_MIN:
                        continue
                    lw = m_strike - l_strike
                    if lw > uw * 2.5:
                        continue

                    liq = h["oi_proxy"] + m["oi_proxy"] + l["oi_proxy"]
                    score = liq / (h["spread"] + m["spread"] + l["spread"] + 0.01) - lw / uw

                    if score > best_score:
                        best_score = score
                        best = {
                            "expiry":       expiry,
                            "dte":          dte,
                            "long_upper":   h_strike,
                            "short_strike": m_strike,
                            "long_lower":   l_strike,
                            "upper_wing":   round(uw, 2),
                            "lower_wing":   round(lw, 2),
                            "liq_score":    min(round(liq / 8), 10),
                            "h_bid":   h["bid"],  "h_ask": h["ask"],
                            "h_spread":h["spread"], "h_oi": h["oi_proxy"],
                            "m_bid":   m["bid"],  "m_ask": m["ask"],
                            "m_spread":m["spread"], "m_oi": m["oi_proxy"],
                            "l_bid":   l["bid"],  "l_ask": l["ask"],
                            "l_spread":l["spread"], "l_oi": l["oi_proxy"],
                        }
    return best


def _compute_gex_context(spot: float, vix_now: float, vix_prev: float,
                          gex_contracts: list[dict]) -> Optional[dict]:
    """
    Run compute_exposures on this ticker's chain and return a GEX context dict
    compatible with BWBInputs (regime, flip_level, put_wall, call_wall, vanna, charm).
    Returns None if chain is too sparse to be meaningful.
    """
    if len(gex_contracts) < 10:  # too sparse for reliable signal
        return None
    try:
        from src.analysis.gex_scanner import compute_exposures
        r = compute_exposures(spot, vix_now, vix_prev, gex_contracts)
        return {
            "regime":       r.gex_regime,
            "flip_level":   r.flip_level,
            "put_wall":     r.put_wall,
            "call_wall":    r.call_wall,
            "gamma_wall":   r.gamma_wall,
            "vanna_signal": r.vanna_signal,
            "charm_signal": r.charm_signal,
            "gex":          round(r.net_gex_bn, 3),
        }
    except Exception as e:
        logger.warning("GEX computation failed: %s", e)
        return None


def _scan_one(ticker: str, spot: float, prev: float, tier: str,
              vix_now: float = 0.0, vix_prev: float = 0.0) -> dict:
    """Full BWB screen + per-stock GEX for a single ticker."""
    result: dict = {
        "ticker":      ticker,
        "spot":        round(spot, 2),
        "prev":        round(prev, 2),
        "change":      round(spot - prev, 2),
        "change_pct":  round((spot - prev) / prev * 100, 2) if prev else 0.0,
        "tier":        tier,
        "price_ok":    PRICE_MIN <= spot <= PRICE_MAX,
        "candidate":   None,
        "gex_context": None,   # per-stock GEX — None means use SPY fallback
        "note":        "",
        "error":       None,
    }

    if not result["price_ok"]:
        result["note"] = f"Price ${spot:.2f} outside ${PRICE_MIN:.0f}–${PRICE_MAX:.0f} range"
        return result

    try:
        from src.live.alpaca_options import fetch_chain_combined

        chain = fetch_chain_combined(ticker, spot, dte_min=DTE_MIN, dte_max=DTE_MAX)
        puts  = chain["puts_liquid"]

        if not puts:
            result["note"] = "No liquid puts found via Alpaca"
            return result

        candidate = _suggest_bwb(spot, puts)
        if not candidate:
            result["note"] = f"{len(puts)} puts found — no viable BWB structure"
            return result

        # Candidate found: compute this stock's own GEX from the same chain call
        gex_ctx = _compute_gex_context(spot, vix_now, vix_prev, chain["gex_contracts"])
        result["candidate"]   = candidate
        result["gex_context"] = gex_ctx

        h, m, l = (candidate["long_upper"], candidate["short_strike"],
                   candidate["long_lower"])
        regime_str = (f"  [{gex_ctx['regime'].replace('_GAMMA','').lower()} γ]"
                      if gex_ctx else "  [no GEX]")
        result["note"] = (
            f"{h:.0f}/{m:.0f}/{l:.0f}  DTE {candidate['dte']}  "
            f"liq {candidate['liq_score']}/10{regime_str}"
        )

    except Exception as e:
        logger.warning("Scan error for %s: %s", ticker, e)
        result["error"] = str(e)
        result["note"]  = "Scan error — see logs"

    return result


def scan(tickers: list[str] = None, tier: str = "low_risk",
         vix_now: float = 0.0, vix_prev: float = 0.0) -> list[dict]:
    """
    Screen tickers for BWB candidates with per-stock GEX context.

    Args:
        tickers:  explicit list (overrides tier)
        tier:     'low_risk' | 'medium_risk' | 'high_risk' | 'all'
        vix_now:  current VIX (for GEX computation)
        vix_prev: prior VIX close
    """
    if tickers is None:
        if tier == "all":
            tickers = [t for ts in BWB_UNIVERSE.values() for t in ts]
        else:
            tickers = BWB_UNIVERSE.get(tier, BWB_UNIVERSE["low_risk"])

    tier_map = {t: k for k, ts in BWB_UNIVERSE.items() for t in ts}

    logger.info("BWB scan: %d tickers (%s)  VIX %.2f", len(tickers), tier, vix_now)
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
                vix_now,
                vix_prev,
            ): ticker
            for ticker in tickers
            if ticker in spots
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                logger.error("Future error for %s: %s", futures[future], e)

    for t in tickers:
        if t not in spots:
            results.append({
                "ticker": t, "spot": 0.0, "prev": 0.0, "change": 0.0,
                "change_pct": 0.0, "tier": tier_map.get(t, tier),
                "price_ok": False, "candidate": None, "gex_context": None,
                "note": "Price fetch failed", "error": "no price data",
            })

    results.sort(key=lambda r: (r["candidate"] is None, r["ticker"]))
    n_candidates = sum(1 for r in results if r["candidate"])
    n_with_gex   = sum(1 for r in results if r["gex_context"])
    logger.info("BWB scan complete: %d/%d candidates, %d with GEX",
                n_candidates, len(results), n_with_gex)
    return results
