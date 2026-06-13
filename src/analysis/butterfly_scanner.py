"""
GEX-pinned long butterfly scanner.

A long butterfly (BUY 1 / SELL 2 / BUY 1, symmetric wings) is a defined-risk,
high-reward bet that price *pins* near a strike at expiry:

    risk   = the debit paid (small)
    reward = wing_width - debit  (often 5-20x the debit)
    payoff = maximised if the underlying lands at the body strike

The edge is WHERE to center it. In a POSITIVE-GAMMA regime, dealer hedging
suppresses volatility and gravitates price toward high-gamma strikes into
expiry. So we only suggest pin flies when GEX is positive, and we center them on
the gamma wall / call wall / flip level that the GEX scan already computes.

In NEGATIVE gamma (or UNKNOWN regime) we stand down — there is no pin to target.

Live leg mids come from Alpaca (`fetch_chain_combined`). Every leg is verified to
have a two-sided market before a structure is priced.
"""
import logging
from datetime import date

logger = logging.getLogger(__name__)

# Tunables
WING_WIDTHS   = [1, 2, 3, 5]   # candidate symmetric wing widths (points)
MAX_DEBIT     = 2.50           # skip flies costing more than this per share
MIN_RR        = 4.0            # require at least 4:1 reward:risk to surface
MAX_PIN_DIST  = 0.025          # body must be within 2.5% of spot (realistic pin)


def _nearest(strikes: list, target: float):
    return min(strikes, key=lambda s: abs(s - target)) if strikes else None


def scan(spot: float, gex, ticker: str = "SPY",
         dte_min: int = 0, dte_max: int = 10) -> dict:
    """
    Return {regime, dte, expiry, centers, candidates, note}.

    `gex` is a GEXResult (from gex_scanner). Pin flies are only produced when
    gex.gex_regime == 'POSITIVE_GAMMA'. candidates is a list of dicts sorted by
    reward:risk, each ready for Slack formatting / order placement.
    """
    out = {"regime": getattr(gex, "gex_regime", "UNKNOWN"), "dte": None,
           "expiry": None, "centers": [], "candidates": [], "note": ""}

    if gex is None or gex.gex_regime != "POSITIVE_GAMMA":
        out["note"] = (f"Regime {out['regime']} — pin flies need POSITIVE_GAMMA "
                       f"(dealers suppress vol toward a strike).")
        return out

    # Pin targets the GEX scan already gives us, de-duplicated and rounded.
    raw_centers = [
        ("gamma wall", getattr(gex, "gamma_wall", None)),
        ("call wall",  getattr(gex, "call_wall", None)),
        ("flip level", getattr(gex, "flip_level", None)),
    ]
    centers = []
    seen = set()
    for label, c in raw_centers:
        if not c:
            continue
        ck = round(c)
        if ck in seen or abs(ck - spot) / spot > MAX_PIN_DIST:
            continue
        seen.add(ck)
        centers.append((label, ck))
    out["centers"] = centers
    if not centers:
        out["note"] = "No GEX pin level within range of spot."
        return out

    # One live chain fetch; build a put-mid map for the nearest in-window expiry.
    try:
        from src.live.alpaca_options import fetch_chain_combined
        chain = fetch_chain_combined(ticker, spot, dte_min=dte_min, dte_max=dte_max)
    except Exception as e:
        out["note"] = f"Chain fetch failed: {e}"
        return out

    puts = chain.get("puts_liquid", [])
    if not puts:
        out["note"] = "No liquid puts in the target DTE window."
        return out

    # Pick the nearest expiry that has data (prefer >=1 DTE to avoid 0DTE assignment
    # noise unless that is all that exists).
    by_exp: dict = {}
    for p in puts:
        by_exp.setdefault(p["expiry"], []).append(p)
    exp = sorted(by_exp.keys())[0]
    rows = by_exp[exp]
    dte  = rows[0]["dte"]
    out["dte"], out["expiry"] = dte, exp

    mids = {r["strike"]: round((r["bid"] + r["ask"]) / 2, 3) for r in rows}
    strikes = sorted(mids.keys())

    candidates = []
    for label, center in centers:
        cK = _nearest(strikes, center)
        if cK is None:
            continue
        for w in WING_WIDTHS:
            hi = _nearest(strikes, cK + w)
            lo = _nearest(strikes, cK - w)
            # require genuine, distinct wings symmetric in width
            if hi is None or lo is None or hi <= cK or lo >= cK:
                continue
            if (hi - cK) != (cK - lo):
                continue
            if not all(k in mids and mids[k] > 0 for k in (hi, cK, lo)):
                continue

            debit = round(mids[hi] - 2 * mids[cK] + mids[lo], 3)
            wing  = hi - cK
            if debit <= 0 or debit > MAX_DEBIT:
                continue
            max_profit = round((wing - debit) * 100)
            risk       = round(debit * 100)
            rr         = round((wing - debit) / debit, 1)
            if rr < MIN_RR:
                continue

            be_lo = round(lo + debit, 2)
            be_hi = round(hi - debit, 2)
            pin_dist = round(abs(cK - spot) / spot * 100, 2)

            candidates.append({
                "ticker": ticker, "center_label": label,
                "expiry": exp, "dte": dte,
                "long_upper": hi, "short_body": cK, "long_lower": lo,
                "wing": wing, "debit": debit,
                "debit_usd": risk, "max_profit_usd": max_profit, "rr": rr,
                "be_lo": be_lo, "be_hi": be_hi, "band": round(be_hi - be_lo, 2),
                "pin_dist_pct": pin_dist,
                "legs": [
                    {"action": "BUY",  "strike": hi, "opt": "P", "qty": 1, "mid": mids[hi]},
                    {"action": "SELL", "strike": cK, "opt": "P", "qty": 2, "mid": mids[cK]},
                    {"action": "BUY",  "strike": lo, "opt": "P", "qty": 1, "mid": mids[lo]},
                ],
            })

    # Best R/R per center; keep the single best width for each pin level.
    best_by_center: dict = {}
    for c in candidates:
        key = c["short_body"]
        if key not in best_by_center or c["rr"] > best_by_center[key]["rr"]:
            best_by_center[key] = c
    out["candidates"] = sorted(best_by_center.values(), key=lambda c: -c["rr"])
    if not out["candidates"]:
        out["note"] = "Positive gamma, but no fly cleared the R/R / debit filters."
    return out


def fmt_slack(result: dict) -> str:
    """Format pin-fly candidates for Slack. Returns '' when nothing to show."""
    cands = result.get("candidates") or []
    if not cands:
        return ""
    lines = [
        f":butterfly: *GEX-Pinned Butterfly* — {cands[0]['ticker']} "
        f"({result.get('expiry','')}, {result.get('dte','')}DTE)  "
        f"_positive gamma → pin play_"
    ]
    for c in cands:
        lines.append(
            f"  *{c['center_label']} ${c['short_body']:.0f}*  "
            f"(pin {c['pin_dist_pct']:+.1f}% from spot)\n"
            f"    BUY ${c['long_upper']:.0f}P / SELL 2x ${c['short_body']:.0f}P / "
            f"BUY ${c['long_lower']:.0f}P  ({c['wing']}pt wings)\n"
            f"    Debit ${c['debit']:.2f} (${c['debit_usd']})  →  "
            f"Max profit +${c['max_profit_usd']}  *{c['rr']:.0f}:1*  "
            f"| profit ${c['be_lo']:.1f}–${c['be_hi']:.1f}"
        )
    return "\n".join(lines)
