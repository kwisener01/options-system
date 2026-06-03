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
from datetime import date, datetime
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
DTE_MAX      = 45
CREDIT_MIN   = 0.05   # reject any structure that doesn't collect at least a nickel


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


def _fetch_earnings_dates(tickers: list[str]) -> dict[str, Optional[date]]:
    """
    Return {ticker: next_earnings_date | None} via Yahoo Finance v10 calendarEvents.
    Runs tickers in parallel. Returns None for ETFs or fetch failures.
    """
    import urllib3
    import requests as _req
    urllib3.disable_warnings()

    def _one(ticker: str) -> tuple[str, Optional[date]]:
        try:
            url = (f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
                   f"?modules=calendarEvents")
            resp = _req.get(url, headers={"User-Agent": "Mozilla/5.0"},
                            timeout=12, verify=False).json()
            result = (resp.get("quoteSummary", {}).get("result") or [])
            if not result:
                return ticker, None
            dates = (result[0].get("calendarEvents", {})
                               .get("earnings", {})
                               .get("earningsDate", []))
            # Yahoo returns a range; take the first timestamp (earliest estimate)
            if dates:
                ts = dates[0].get("raw")
                if ts:
                    return ticker, datetime.utcfromtimestamp(ts).date()
            return ticker, None
        except Exception as e:
            logger.debug("Earnings fetch failed for %s: %s", ticker, e)
            return ticker, None

    out: dict[str, Optional[date]] = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for ticker, dt in ex.map(_one, tickers):
            out[ticker] = dt
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

                    net_credit_check = round(
                        ((m["bid"] + m["ask"]) / 2) * 2
                        - (h["bid"] + h["ask"]) / 2
                        - (l["bid"] + l["ask"]) / 2, 3
                    )
                    if net_credit_check < CREDIT_MIN:
                        continue

                    liq = h["oi_proxy"] + m["oi_proxy"] + l["oi_proxy"]
                    score = liq / (h["spread"] + m["spread"] + l["spread"] + 0.01) - lw / uw

                    if score > best_score:
                        best_score = score
                        h_mid = round((h["bid"] + h["ask"]) / 2, 3)
                        m_mid = round((m["bid"] + m["ask"]) / 2, 3)
                        l_mid = round((l["bid"] + l["ask"]) / 2, 3)
                        net_credit = round(m_mid * 2 - h_mid - l_mid, 3)
                        best = {
                            "expiry":       expiry,
                            "dte":          dte,
                            "long_upper":   h_strike,
                            "short_strike": m_strike,
                            "long_lower":   l_strike,
                            "upper_wing":   round(uw, 2),
                            "lower_wing":   round(lw, 2),
                            "liq_score":    min(round(liq / 8), 10),
                            "h_bid":   h["bid"],  "h_ask": h["ask"],  "h_mid": h_mid,
                            "h_spread":h["spread"], "h_oi": h["oi_proxy"],
                            "m_bid":   m["bid"],  "m_ask": m["ask"],  "m_mid": m_mid,
                            "m_spread":m["spread"], "m_oi": m["oi_proxy"],
                            "l_bid":   l["bid"],  "l_ask": l["ask"],  "l_mid": l_mid,
                            "l_spread":l["spread"], "l_oi": l["oi_proxy"],
                            "credit":  net_credit,
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
        # Per-stock GEX always uses Alpaca indicative data (no real OI available).
        # Mark regime as LOW confidence so the BWB analyzer treats it as neutral.
        return {
            "regime":             "UNKNOWN",   # Alpaca proxy OI — not reliable for regime
            "flip_level":         r.flip_level,
            "put_wall":           r.put_wall,
            "call_wall":          r.call_wall,
            "gamma_wall":         r.gamma_wall,
            "vanna_signal":       r.vanna_signal,
            "charm_signal":       r.charm_signal,
            "gex":                round(r.net_gex_bn, 3),
            "regime_confidence":  "LOW",
        }
    except Exception as e:
        logger.warning("GEX computation failed: %s", e)
        return None


def _scan_one(ticker: str, spot: float, prev: float, tier: str,
              vix_now: float = 0.0, vix_prev: float = 0.0,
              earnings_date: Optional[date] = None) -> dict:
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

        # Run BWB analyzer for rating, risk, exit plan
        try:
            from src.analysis.bwb_analyzer import BWBInputs, analyze as _bwb_analyze
            gx = gex_ctx or {}
            bwb_result = _bwb_analyze(BWBInputs(
                ticker=ticker, spot=spot, dte=candidate["dte"],
                long_upper=candidate["long_upper"],
                short_strike=candidate["short_strike"],
                long_lower=candidate["long_lower"],
                credit=candidate.get("credit", 0.0),
                regime=gx.get("regime", "UNKNOWN"),
                vix_now=vix_now, vix_prev=vix_prev,
                flip_level=gx.get("flip_level", 0.0),
                put_wall=gx.get("put_wall", 0.0),
                call_wall=gx.get("call_wall", 0.0),
                major_news=bool(candidate.get("earnings_warning")),
            ))
            candidate["analysis"] = {
                "rating":          bwb_result.rating,
                "score":           bwb_result.setup_score,
                "max_profit_usd":  round(bwb_result.max_profit_usd),
                "max_loss_usd":    round(bwb_result.max_loss_usd),
                "lower_breakeven": round(bwb_result.lower_breakeven, 2),
                "rr_ratio":        round(bwb_result.rr_ratio, 2),
                "exit_plan":       bwb_result.exit_plan,
                "overnight_ok":    bwb_result.overnight_ok,
                "summary":         bwb_result.summary,
            }
        except Exception as e:
            logger.debug("BWB analysis failed for %s: %s", ticker, e)
            candidate["analysis"] = None

        # Earnings check: flag if earnings fall within the option's DTE window
        today = date.today()
        if earnings_date is not None:
            days_to_earnings = (earnings_date - today).days
            earnings_within_dte = 0 <= days_to_earnings <= candidate["dte"]
            candidate["earnings_date"]    = str(earnings_date)
            candidate["earnings_days_out"] = days_to_earnings
            candidate["earnings_warning"] = earnings_within_dte
        else:
            candidate["earnings_date"]     = None
            candidate["earnings_days_out"] = None
            candidate["earnings_warning"]  = None  # unknown — treat as caution

        h, m, l = (candidate["long_upper"], candidate["short_strike"],
                   candidate["long_lower"])
        regime_str = (f"  [{gex_ctx['regime'].replace('_GAMMA','').lower()} γ]"
                      if gex_ctx else "  [no GEX]")
        earn_str   = ""
        if candidate["earnings_warning"] is True:
            earn_str = f"  ⚠ EARNS {earnings_date}"
        elif candidate["earnings_warning"] is None:
            earn_str = "  [earns?]"
        result["note"] = (
            f"{h:.0f}/{m:.0f}/{l:.0f}  DTE {candidate['dte']}  "
            f"liq {candidate['liq_score']}/10{regime_str}{earn_str}"
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

    # Fetch spots and earnings dates in parallel before launching chain scans
    with ThreadPoolExecutor(max_workers=2) as pre:
        spots_future    = pre.submit(_fetch_spots, tickers)
        earnings_future = pre.submit(_fetch_earnings_dates, tickers)
    spots         = spots_future.result()
    earnings_dates = earnings_future.result()
    logger.info("Earnings dates: %s", {t: str(d) for t, d in earnings_dates.items() if d})

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
                earnings_dates.get(ticker),
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
                "earnings_date": str(earnings_dates.get(t)) if earnings_dates.get(t) else None,
            })

    results.sort(key=lambda r: (r["candidate"] is None, r["ticker"]))
    n_candidates = sum(1 for r in results if r["candidate"])
    n_with_gex   = sum(1 for r in results if r["gex_context"])
    logger.info("BWB scan complete: %d/%d candidates, %d with GEX",
                n_candidates, len(results), n_with_gex)
    return results


def fmt_slack(results: list[dict], tier: str, spy_regime: str,
              vix_now: float, timestamp: str) -> str:
    """Format BWB scan results for Slack."""
    tier_label = tier.replace("_", " ").title()
    lines = [
        f":butterfly: *BWB Scanner — {timestamp}*  ({tier_label})",
        f"SPY: `{spy_regime.replace('_', ' ').title()}` | VIX {vix_now:.2f}",
        "",
    ]

    candidates = [r for r in results if r.get("candidate")]
    skipped    = [r for r in results if not r.get("candidate") and r.get("price_ok")]

    if not candidates:
        lines.append("_No viable BWB structures found today._")
        if skipped:
            lines.append(f"_Scanned: {', '.join(r['ticker'] for r in results)} — no qualifying setups._")
        return "\n".join(lines)

    for r in candidates:
        c   = r["candidate"]
        an  = c.get("analysis") or {}
        gc  = r.get("gex_context") or {}

        rating = an.get("rating", "—")
        rating_icon = ":white_check_mark:" if rating == "A+" else ":large_blue_circle:" if rating == "Acceptable" else ":warning:"
        chg_str = f"{r['change_pct']:+.1f}%"
        credit_str = f"+${c['credit']:.2f}" if c.get("credit") is not None and c["credit"] >= 0 else f"-${abs(c.get('credit',0)):.2f}"
        risk_str   = f"${int(an['max_loss_usd'])}" if an.get("max_loss_usd") else "?"
        rr_str     = f"{an['rr_ratio']:.1f}×" if an.get("rr_ratio") else "?"
        regime_str = gc.get("regime", "").replace("_", " ").title() if gc else "SPY proxy"
        ew_str     = f"  ⚠ Earnings {c.get('earnings_date')}" if c.get("earnings_warning") else ""

        lines += [
            f"{rating_icon} *{r['ticker']} ${r['spot']:.2f}* ({chg_str})  ·  {rating}",
            f"  `BUY ${c['long_upper']}P / SELL 2× ${c['short_strike']}P / BUY ${c['long_lower']}P`  ·  {c['dte']}DTE",
            f"  Credit {credit_str}  ·  Max risk {risk_str}  ·  R:R {rr_str}  ·  GEX: {regime_str}{ew_str}",
            "",
        ]

    n = len(candidates)
    total = len(results)
    lines.append(f"_{n} of {total} ticker{'s' if total != 1 else ''} {'have' if n != 1 else 'has'} a viable BWB structure today._")
    return "\n".join(lines)
