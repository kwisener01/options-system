"""
GEX-anchored Batman scanner (double broken-wing butterfly) for XSP.

A Batman = a put BWB (left ear) + a call BWB (right ear) sharing one expiry,
with both wings "broken" so the whole thing is entered for a NET CREDIT:

    PUT  BWB:  BUY 1 (Bp+inner)P / SELL 2 Bp P / BUY 1 (Bp-outer)P
    CALL BWB:  BUY 1 (Bc-inner)C / SELL 2 Bc C / BUY 1 (Bc+outer)C

The ears sit at the GEX put wall (Bp) and call wall (Bc). The "cowl" is the
flat middle between the ears.

KEY FACT: when both ears are OTM (put shorts below spot, call shorts above),
every leg is OTM at spot, so **P&L at spot = the net credit**. A *positive
cowl* therefore means a *net credit* — which broken wings make possible but
symmetric butterflies never do (they're always a debit). This scanner only
surfaces structures with a positive cowl (net credit) and defined tail risk.

Only fires in POSITIVE_GAMMA. Priced off the SPY chain on Alpaca (SPY ≈ XSP
numerically, since XSP = SPX/10 ≈ SPY); strikes/prices are presented for XSP
execution at a European / cash-settled broker (no assignment, no pin risk).
"""
import logging

logger = logging.getLogger(__name__)

INNER_WINGS  = [2, 3, 5]            # short -> inner long (toward spot)
OUTER_WINGS  = [10, 15, 20, 25]    # short -> outer long (the broken, wider side)
MIN_COWL     = 0.15                 # net credit per share (positive cowl floor)
MAX_LOSS_CAP = 1500                 # skip structures risking more than this ($)


def scan(spot: float, gex, ticker: str = "SPY", label_as: str = "XSP",
         dte_min: int = 5, dte_max: int = 21) -> dict:
    """Return {regime, dte, expiry, candidate, note}. candidate is the best
    positive-cowl Batman (or None)."""
    out = {"regime": getattr(gex, "gex_regime", "UNKNOWN"), "dte": None,
           "expiry": None, "candidate": None, "note": ""}

    if gex is None or gex.gex_regime != "POSITIVE_GAMMA":
        out["note"] = (f"Regime {out['regime']} — Batman wants POSITIVE_GAMMA "
                       f"(price stays between the walls).")
        return out

    put_wall  = getattr(gex, "put_wall", spot * 0.98)
    call_wall = getattr(gex, "call_wall", spot * 1.02)

    try:
        from src.live.alpaca_options import fetch_chain_combined
        chain = fetch_chain_combined(ticker, spot, dte_min=dte_min, dte_max=dte_max)
    except Exception as e:
        out["note"] = f"Chain fetch failed: {e}"
        return out

    puts, calls = chain.get("puts_liquid", []), chain.get("calls_liquid", [])
    if not puts or not calls:
        out["note"] = "Need both liquid puts and calls in the DTE window."
        return out

    common = sorted({p["expiry"] for p in puts} & {c["expiry"] for c in calls})
    if not common:
        out["note"] = "No common put/call expiry."
        return out
    exp = common[0]
    prows = [p for p in puts  if p["expiry"] == exp]
    crows = [c for c in calls if c["expiry"] == exp]
    dte = prows[0]["dte"]
    out["dte"], out["expiry"] = dte, exp

    pm = {p["strike"]: round((p["bid"] + p["ask"]) / 2, 3) for p in prows}
    cm = {c["strike"]: round((c["bid"] + c["ask"]) / 2, 3) for c in crows}
    p_strikes, c_strikes = sorted(pm), sorted(cm)

    def nearest(strikes, t):
        return min(strikes, key=lambda s: abs(s - t)) if strikes else None

    Bp = nearest([s for s in p_strikes if s < spot], put_wall)   # left ear (short put)
    Bc = nearest([s for s in c_strikes if s > spot], call_wall)  # right ear (short call)
    if Bp is None or Bc is None:
        out["note"] = "No strikes at the put/call walls."
        return out

    best = None
    for inner in INNER_WINGS:
        for outer in OUTER_WINGS:
            if outer <= inner:
                continue
            p_inner = nearest(p_strikes, Bp + inner)   # upper put long (toward spot)
            p_outer = nearest(p_strikes, Bp - outer)   # lower put long (broken)
            c_inner = nearest(c_strikes, Bc - inner)   # lower call long (toward spot)
            c_outer = nearest(c_strikes, Bc + outer)   # upper call long (broken)
            legs_k = [p_outer, Bp, p_inner, c_inner, Bc, c_outer]
            if len(set(legs_k)) != 6:
                continue
            if not (p_outer < Bp < p_inner <= spot <= c_inner < Bc < c_outer):
                continue
            if not all(k in pm for k in (p_inner, Bp, p_outer)) or \
               not all(k in cm for k in (c_inner, Bc, c_outer)):
                continue

            put_credit  = 2 * pm[Bp] - pm[p_inner] - pm[p_outer]
            call_credit = 2 * cm[Bc] - cm[c_inner] - cm[c_outer]
            net = round(put_credit + call_credit, 3)
            if net < MIN_COWL:                  # cowl must be positive
                continue

            def pnl(S):
                iv = (max(p_inner - S, 0) - 2 * max(Bp - S, 0) + max(p_outer - S, 0)
                      + max(S - c_inner, 0) - 2 * max(S - Bc, 0) + max(S - c_outer, 0))
                return (iv + net) * 100

            grid = range(int(p_outer) - 40, int(c_outer) + 40)
            max_loss = min(pnl(s) for s in grid)
            if max_loss < -MAX_LOSS_CAP:
                continue
            cowl = round(pnl(round(spot)))
            ear_l, ear_r = round(pnl(Bp)), round(pnl(Bc))
            prof = [s for s in grid if pnl(s) > 0]
            if not prof:
                continue

            cand = {
                "ticker": label_as, "expiry": exp, "dte": dte,
                "put_inner": p_inner, "put_short": Bp, "put_outer": p_outer,
                "call_inner": c_inner, "call_short": Bc, "call_outer": c_outer,
                "inner_wing": inner, "outer_wing": outer,
                "credit": net, "cowl_usd": cowl,
                "ear_left_usd": ear_l, "ear_right_usd": ear_r,
                "max_loss_usd": round(max_loss),
                "range_lo": min(prof), "range_hi": max(prof),
                "range_pts": max(prof) - min(prof),
                "be_lo": min(prof), "be_hi": max(prof),
                "legs": [
                    {"action": "BUY",  "strike": p_inner, "opt": "P", "qty": 1, "mid": pm[p_inner]},
                    {"action": "SELL", "strike": Bp,      "opt": "P", "qty": 2, "mid": pm[Bp]},
                    {"action": "BUY",  "strike": p_outer, "opt": "P", "qty": 1, "mid": pm[p_outer]},
                    {"action": "BUY",  "strike": c_inner, "opt": "C", "qty": 1, "mid": cm[c_inner]},
                    {"action": "SELL", "strike": Bc,      "opt": "C", "qty": 2, "mid": cm[Bc]},
                    {"action": "BUY",  "strike": c_outer, "opt": "C", "qty": 1, "mid": cm[c_outer]},
                ],
            }
            # Prefer the widest profitable band that keeps a positive cowl and
            # the smallest tail loss; tie-break on higher cowl.
            key = (cand["range_pts"], cand["cowl_usd"], cand["max_loss_usd"])
            if best is None or key > best[0]:
                best = (key, cand)

    if best is None:
        out["note"] = "Positive gamma, but no positive-cowl Batman cleared the filters."
        return out
    out["candidate"] = best[1]
    return out


def fmt_slack(result: dict) -> str:
    c = result.get("candidate")
    if not c:
        return ""
    return "\n".join([
        f":bat: *GEX-Anchored Batman* — {c['ticker']} ({c['expiry']}, {c['dte']}DTE)  "
        f"_positive gamma · execute as XSP (European/cash-settled)_",
        f"  PUT  BWB:  BUY ${c['put_inner']:.0f}P / SELL 2x ${c['put_short']:.0f}P / "
        f"BUY ${c['put_outer']:.0f}P",
        f"  CALL BWB:  BUY ${c['call_inner']:.0f}C / SELL 2x ${c['call_short']:.0f}C / "
        f"BUY ${c['call_outer']:.0f}C",
        f"  Net CREDIT +${c['credit']:.2f}  →  *cowl +${c['cowl_usd']}* (profit even at spot)",
        f"  Ears: ${c['put_short']:.0f} +${c['ear_left_usd']}  |  "
        f"${c['call_short']:.0f} +${c['ear_right_usd']}",
        f"  Profitable ${c['range_lo']:.0f}–${c['range_hi']:.0f} "
        f"({c['range_pts']:.0f} pts)  |  Max loss -${abs(c['max_loss_usd'])}",
        f"  _wings {c['inner_wing']}/{c['outer_wing']}pt · manage at 50%_",
    ])
