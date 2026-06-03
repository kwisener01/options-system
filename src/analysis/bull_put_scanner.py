"""
Bull Put Credit Spread Scanner

Two rules — both must be true for every trade returned:
  1. Short strike is AT the GEX put wall  (structural dealer support, not arbitrary)
  2. Credit > 50% of spread width         (reward > risk by definition)

Scoring (max 9):
  Credit/width ratio (0-3 pts):
    >= 65% : 3 pts
    55-65% : 2 pts
    50-55% : 1 pt
  Strike proximity to put wall (0-2 pts):
    Within $0.50 : 2 pts
    Within $1.50 : 1 pt
  DTE quality (0-2 pts):
    14-21 DTE : 2 pts  (theta decay sweet spot)
    7-13 DTE  : 1 pt
    22-45 DTE : 1 pt
  Liquidity (0-2 pts):
    OI proxy sum >= 20 : 2 pts
    OI proxy sum >= 10 : 1 pt
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from typing import Optional

logger = logging.getLogger(__name__)

UNIVERSE: list[str] = [
    # BWB universe — already liquid
    "BAC", "F", "T", "PFE", "INTC", "PYPL", "UBER",
    "PLTR", "SOFI", "HOOD", "DKNG", "CCL", "MU",
    # Value watchlist
    "KO", "AXP", "CVX", "KR", "BMY", "ABBV", "GILD",
    # Large-cap with deep option markets
    "JPM", "WFC", "MS", "GS", "C",
    "AAPL", "MSFT", "AMZN", "META", "GOOGL",
    "XOM", "HAL", "SLB",
    "GE", "BA", "CAT",
    # Index ETFs
    "SPY", "QQQ", "IWM",
]

PRICE_MIN    = 10.0
PRICE_MAX    = 600.0
DTE_MIN      = 7
DTE_MAX      = 45
SPREAD_MAX   = 0.25   # max bid/ask spread per leg
OI_MIN       = 3      # min OI proxy per leg
CREDIT_MIN_PCT = 0.50  # must collect > 50% of width
CREDIT_MIN_ABS = 0.05  # must collect at least a nickel

STRONG_SCORE = 6
WATCH_SCORE  = 4

# Spread widths to try, in order of preference
WIDTHS = [1.0, 2.0, 2.5, 3.0, 5.0]


# ── Data fetchers ─────────────────────────────────────────────────────────────

def _fetch_spots(tickers: list[str]) -> dict[str, tuple[float, float]]:
    import urllib3, requests as _req
    urllib3.disable_warnings()
    out: dict[str, tuple[float, float]] = {}
    def _one(t):
        try:
            url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{t}"
                   f"?interval=1d&range=5d")
            meta = _req.get(url, headers={"User-Agent": "Mozilla/5.0"},
                            timeout=15, verify=False).json()["chart"]["result"][0]["meta"]
            return t, (float(meta["regularMarketPrice"]), float(meta["chartPreviousClose"]))
        except Exception as e:
            logger.warning("Spot failed %s: %s", t, e)
            return t, None
    with ThreadPoolExecutor(max_workers=8) as ex:
        for t, v in ex.map(_one, tickers):
            if v:
                out[t] = v
    return out


def _fetch_earnings(tickers: list[str]) -> dict[str, Optional[date]]:
    import urllib3, requests as _req
    urllib3.disable_warnings()
    def _one(ticker):
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
            if dates:
                ts = dates[0].get("raw")
                if ts:
                    return ticker, datetime.utcfromtimestamp(ts).date()
        except Exception:
            pass
        return ticker, None
    out: dict[str, Optional[date]] = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for t, d in ex.map(_one, tickers):
            out[t] = d
    return out


# ── Spread search ─────────────────────────────────────────────────────────────

def _find_best_spread(spot: float, put_wall: float,
                      puts: list[dict]) -> Optional[dict]:
    """
    Search all expirations in the DTE window for the best credit spread
    with short strike at put wall and credit > 50% of width.
    """
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
        strikes   = sorted(by_strike.keys())
        if not strikes:
            continue

        # Find short strike candidates near put wall (within $2)
        short_candidates = [s for s in strikes if abs(s - put_wall) <= 2.0]
        if not short_candidates:
            # Relax to nearest single strike if nothing within $2
            nearest = min(strikes, key=lambda s: abs(s - put_wall))
            short_candidates = [nearest]

        for short_s in short_candidates:
            short = by_strike[short_s]
            if short["spread"] > SPREAD_MAX or short["oi_proxy"] < OI_MIN:
                continue

            short_mid = (short["bid"] + short["ask"]) / 2

            for width in WIDTHS:
                long_target = short_s - width
                # Find the nearest available strike to the long target
                long_s = min(strikes, key=lambda s: abs(s - long_target))
                if abs(long_s - long_target) > 1.0 or long_s >= short_s:
                    continue
                long = by_strike[long_s]
                if long["spread"] > SPREAD_MAX or long["oi_proxy"] < OI_MIN:
                    continue

                long_mid   = (long["bid"] + long["ask"]) / 2
                credit     = round(short_mid - long_mid, 3)
                act_width  = round(short_s - long_s, 2)
                credit_pct = credit / act_width if act_width > 0 else 0

                if credit < CREDIT_MIN_ABS or credit_pct < CREDIT_MIN_PCT:
                    continue

                # Score this candidate
                score = 0.0

                # Credit/width ratio (0-3)
                if credit_pct >= 0.65:
                    score += 3
                elif credit_pct >= 0.55:
                    score += 2
                else:
                    score += 1

                # Proximity to put wall (0-2)
                dist = abs(short_s - put_wall)
                if dist <= 0.50:
                    score += 2
                elif dist <= 1.50:
                    score += 1

                # DTE quality (0-2)
                if 14 <= dte <= 21:
                    score += 2
                elif DTE_MIN <= dte <= 45:
                    score += 1

                # Liquidity (0-2)
                liq = short["oi_proxy"] + long["oi_proxy"]
                if liq >= 20:
                    score += 2
                elif liq >= 10:
                    score += 1

                if score > best_score:
                    best_score = score
                    best = {
                        "expiry":      expiry,
                        "dte":         dte,
                        "short_strike": short_s,
                        "long_strike":  long_s,
                        "width":        act_width,
                        "credit":       credit,
                        "credit_pct":   round(credit_pct * 100, 1),
                        "max_profit_usd": round(credit * 100, 0),
                        "max_loss_usd":   round((act_width - credit) * 100, 0),
                        "put_wall":     put_wall,
                        "pw_distance":  round(abs(short_s - put_wall), 2),
                        "liq_score":    min(round(liq / 8), 10),
                        "short_bid":    short["bid"], "short_ask": short["ask"],
                        "long_bid":     long["bid"],  "long_ask":  long["ask"],
                        "score":        round(score, 1),
                    }

    return best


# ── Per-ticker scan ───────────────────────────────────────────────────────────

def _scan_one(ticker: str, spot: float, prev: float,
              vix_now: float, vix_prev: float,
              earnings_date: Optional[date]) -> dict:
    result: dict = {
        "ticker":     ticker,
        "spot":       round(spot, 2),
        "change":     round(spot - prev, 2),
        "change_pct": round((spot - prev) / prev * 100, 2) if prev else 0.0,
        "price_ok":   PRICE_MIN <= spot <= PRICE_MAX,
        "candidate":  None,
        "put_wall":   None,
        "gex_regime": None,
        "signal":     "NONE",
        "note":       "",
        "earnings_date": str(earnings_date) if earnings_date else None,
        "earnings_warning": False,
    }

    if not result["price_ok"]:
        result["note"] = f"Price ${spot:.2f} outside scan range"
        return result

    try:
        from src.live.alpaca_options import fetch_chain_combined
        from src.analysis.gex_scanner import compute_exposures

        chain = fetch_chain_combined(ticker, spot, dte_min=DTE_MIN, dte_max=DTE_MAX)
        puts  = chain.get("puts_liquid", [])

        if not puts:
            result["note"] = "No liquid puts found"
            return result

        # Get put wall from this ticker's own GEX
        put_wall  = None
        gex_regime = None
        gex_contracts = chain.get("gex_contracts", [])
        if len(gex_contracts) >= 10:
            try:
                r = compute_exposures(spot, vix_now, vix_prev, gex_contracts)
                put_wall   = r.put_wall
                gex_regime = r.gex_regime
            except Exception as e:
                logger.debug("GEX failed for %s: %s", ticker, e)

        if not put_wall:
            result["note"] = "No GEX put wall — cannot anchor short strike"
            return result

        result["put_wall"]   = round(put_wall, 2)
        result["gex_regime"] = gex_regime

        candidate = _find_best_spread(spot, put_wall, puts)
        if not candidate:
            result["note"] = f"Put wall ${put_wall:.2f} — no credit spread clears 50% of width"
            return result

        # Earnings check
        if earnings_date:
            days_to = (earnings_date - date.today()).days
            candidate["earnings_days_out"] = days_to
            if 0 <= days_to <= candidate["dte"]:
                candidate["earnings_warning"] = True
                result["earnings_warning"]    = True
            else:
                candidate["earnings_warning"] = False
        else:
            candidate["earnings_days_out"] = None
            candidate["earnings_warning"]  = False

        score  = candidate["score"]
        signal = "STRONG" if score >= STRONG_SCORE else ("WATCH" if score >= WATCH_SCORE else "NONE")
        result["candidate"] = candidate
        result["signal"]    = signal

    except Exception as e:
        logger.warning("Scan failed for %s: %s", ticker, e)
        result["note"]  = str(e)
        result["error"] = str(e)

    return result


# ── Public API ────────────────────────────────────────────────────────────────

def scan(tickers: list[str] = None,
         vix_now: float = 0.0,
         vix_prev: float = 0.0) -> list[dict]:
    """
    Scan for bull put credit spread setups.
    Returns list sorted: STRONG first, then WATCH, then NONE.
    """
    tickers = tickers or UNIVERSE
    logger.info("Bull put scan: %d tickers  VIX %.2f", len(tickers), vix_now)

    with ThreadPoolExecutor(max_workers=2) as pre:
        spots_f    = pre.submit(_fetch_spots,   tickers)
        earnings_f = pre.submit(_fetch_earnings, tickers)
    spots    = spots_f.result()
    earnings = earnings_f.result()

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {
            ex.submit(
                _scan_one,
                t,
                spots[t][0], spots[t][1],
                vix_now, vix_prev,
                earnings.get(t),
            ): t
            for t in tickers if t in spots
        }
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                logger.error("Future error %s: %s", futures[f], e)

    # Add tickers with no price data
    for t in tickers:
        if t not in spots:
            results.append({"ticker": t, "spot": 0, "candidate": None,
                            "signal": "NONE", "note": "Price unavailable",
                            "price_ok": False})

    _order = {"STRONG": 0, "WATCH": 1, "NONE": 2}
    results.sort(key=lambda r: (_order.get(r.get("signal", "NONE"), 9),
                                -(r.get("candidate") or {}).get("credit_pct", 0)))

    n_strong = sum(1 for r in results if r.get("signal") == "STRONG")
    n_watch  = sum(1 for r in results if r.get("signal") == "WATCH")
    logger.info("Bull put scan: %d STRONG, %d WATCH of %d", n_strong, n_watch, len(results))
    return results


def fmt_slack(results: list[dict], spy_regime: str,
              vix_now: float, timestamp: str) -> str:
    lines = [
        f":moneybag: *Bull Put Scanner — {timestamp}*",
        f"SPY: `{spy_regime.replace('_',' ').title()}` | VIX {vix_now:.2f}",
        "_Short strike AT GEX put wall · Credit > 50% of width_",
        "",
    ]
    candidates = [r for r in results if r.get("candidate") and r.get("signal") != "NONE"]
    if not candidates:
        lines.append("_No qualifying setups today — no spreads clear 50% credit._")
        return "\n".join(lines)

    for r in candidates:
        c = r["candidate"]
        icon = ":white_check_mark:" if r["signal"] == "STRONG" else ":large_yellow_circle:"
        ew   = f"  ⚠ Earnings {r['earnings_date']}" if r.get("earnings_warning") else ""
        lines += [
            f"{icon} *{r['ticker']} ${r['spot']:.2f}*  ({r.get('change_pct',0):+.1f}%)"
            f"  ·  {r['signal']}  ·  score {c['score']}/9",
            f"  `SELL ${c['short_strike']}P / BUY ${c['long_strike']}P`  ·  {c['dte']}DTE  ·  {c['expiry']}",
            f"  Credit +${c['credit']:.2f} ({c['credit_pct']:.0f}% of ${c['width']:.0f} width)"
            f"  ·  Max profit ${c['max_profit_usd']:.0f}  ·  Max loss ${c['max_loss_usd']:.0f}{ew}",
            f"  GEX put wall ${c['put_wall']:.2f}  ·  short strike {c['pw_distance']:.2f} away",
            "",
        ]
    lines.append(f"_{len(candidates)} setup{'s' if len(candidates)!=1 else ''} found today._")
    return "\n".join(lines)
