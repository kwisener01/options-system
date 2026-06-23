"""
GEX-anchored Iron Condor scanner — the high-win-percentage premium play.

An iron condor sells an OTM put spread AND an OTM call spread:

    SELL put  @ short_put   / BUY put  @ short_put - wing      (put credit spread)
    SELL call @ short_call  / BUY call @ short_call + wing      (call credit spread)

You keep the full credit if price finishes between the short strikes. It is a
positive-theta, defined-risk structure whose win rate is set by the short-strike
delta (≈ probability of finishing ITM): 16Δ ≈ 84% / side, 10Δ ≈ 90% / side.

The edge is twofold and uses the rest of the system:
  1. Only sell in POSITIVE_GAMMA — dealers pin price between the walls, so the
     condor's profit zone aligns with dealer hedging flow.
  2. Anchor the shorts to the GEX put/call walls (or the delta target, whichever
     is FURTHER OTM) so the pin zone sits inside support/resistance.

Vol gauge: condors need premium. VIX bands flag thin-premium days (selling 10Δ
for pennies is poor reward for tail risk). Surfaced, with a soft floor.

`gex` is a GEXResult; live leg mids come from Alpaca (`fetch_chain_combined`).
"""
import logging
import math

from scipy.stats import norm

logger = logging.getLogger(__name__)

_RF = 0.045

# Tunables
DEFAULT_DELTA = 0.16    # short-strike target (~84% POP/side); pass 0.10 for ~90%
DEFAULT_WING  = 5       # spread width (points) on each side
MIN_CREDIT    = 0.20    # skip condors thinner than this (per share, total)
VIX_FLOOR     = 11.0    # below this, premium is too thin to bother
EDGE_MARGIN   = 5       # POP must beat the breakeven win-rate by ≥ this many points
                        # (breakeven POP = 1 − credit/width). Below it the condor has
                        # negative expectancy no matter how high the raw win rate.


def _abs_put_delta(S, K, T, sigma):
    if T <= 0 or sigma <= 0:
        return 1.0 if S < K else 0.0
    d1 = (math.log(S / K) + (_RF + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return float(norm.cdf(-d1))            # |put delta|


def _call_delta(S, K, T, sigma):
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (_RF + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return float(norm.cdf(d1))             # call delta


def _vol_band(vix: float) -> str:
    if vix < 13:   return "LOW"
    if vix <= 20:  return "NORMAL"
    return "RICH"


def scan(spot: float, gex, ticker: str = "SPY",
         dte_min: int = 0, dte_max: int = 7,
         target_delta: float = DEFAULT_DELTA, wing: int = DEFAULT_WING) -> dict:
    """
    Return {regime, dte, expiry, vol_band, candidate, note}.

    candidate is a single best iron condor dict (or None). Only produced in
    POSITIVE_GAMMA with VIX above the thin-premium floor.
    """
    out = {"regime": getattr(gex, "gex_regime", "UNKNOWN"), "dte": None,
           "expiry": None, "vol_band": None, "candidate": None, "note": ""}

    if gex is None or gex.gex_regime != "POSITIVE_GAMMA":
        out["note"] = (f"Regime {out['regime']} — condors want POSITIVE_GAMMA "
                       f"(price pins between the walls).")
        return out

    vix = float(getattr(gex, "vix", 0) or 0)
    out["vol_band"] = _vol_band(vix)
    if vix and vix < VIX_FLOOR:
        out["note"] = f"VIX {vix:.1f} below {VIX_FLOOR:.0f} — premium too thin for a condor."
        return out

    put_wall  = getattr(gex, "put_wall", spot * 0.98)
    call_wall = getattr(gex, "call_wall", spot * 1.02)

    try:
        from src.live.alpaca_options import fetch_chain_combined
        chain = fetch_chain_combined(ticker, spot, dte_min=dte_min, dte_max=dte_max)
    except Exception as e:
        out["note"] = f"Chain fetch failed: {e}"
        return out

    puts  = chain.get("puts_liquid", [])
    calls = chain.get("calls_liquid", [])
    if not puts or not calls:
        out["note"] = "Need both liquid puts and calls in the DTE window."
        return out

    # Nearest expiry that has BOTH sides.
    p_exp = {p["expiry"] for p in puts}
    c_exp = {c["expiry"] for c in calls}
    common = sorted(p_exp & c_exp)
    if not common:
        out["note"] = "No common put/call expiry in the DTE window."
        return out
    exp = common[0]
    prows = [p for p in puts  if p["expiry"] == exp]
    crows = [c for c in calls if c["expiry"] == exp]
    dte   = prows[0]["dte"]
    out["dte"], out["expiry"] = dte, exp
    T = max(dte, 1) / 365

    put_mids  = {p["strike"]: round((p["bid"] + p["ask"]) / 2, 3) for p in prows}
    call_mids = {c["strike"]: round((c["bid"] + c["ask"]) / 2, 3) for c in crows}

    # Short put: delta target, but never less OTM than the put wall.
    put_strikes = sorted(put_mids)
    delta_put = min(put_strikes,
                    key=lambda K: abs(_abs_put_delta(spot, K, T, _iv(prows, K)) - target_delta))
    wall_put  = min(put_strikes, key=lambda K: abs(K - put_wall))
    short_put = min(delta_put, wall_put)          # more OTM of the two

    # Short call: delta target, but never less OTM than the call wall.
    call_strikes = sorted(call_mids)
    delta_call = min(call_strikes,
                     key=lambda K: abs(_call_delta(spot, K, T, _iv(crows, K)) - target_delta))
    wall_call  = min((K for K in call_strikes if K >= call_wall), default=delta_call)
    short_call = max(delta_call, wall_call)        # more OTM of the two

    long_put  = min(put_strikes,  key=lambda K: abs(K - (short_put - wing)))
    long_call = min(call_strikes, key=lambda K: abs(K - (short_call + wing)))

    # Validate distinct, correctly-ordered legs with live mids.
    if not (long_put < short_put < short_call < long_call):
        out["note"] = "Could not place four distinct, ordered legs."
        return out
    legs_k = [long_put, short_put, short_call, long_call]
    if not all(k in put_mids for k in (long_put, short_put)) or \
       not all(k in call_mids for k in (short_call, long_call)):
        out["note"] = "Missing live mid on one or more legs."
        return out

    put_credit  = round(put_mids[short_put]  - put_mids[long_put], 3)
    call_credit = round(call_mids[short_call] - call_mids[long_call], 3)
    credit = round(put_credit + call_credit, 3)
    put_w  = short_put - long_put
    call_w = long_call - short_call
    max_w  = max(put_w, call_w)
    if credit < MIN_CREDIT or credit >= max_w:
        out["note"] = f"Credit ${credit:.2f} too thin / invalid vs width {max_w:.0f}."
        return out

    sp_d = _abs_put_delta(spot, short_put, T, _iv(prows, short_put))
    sc_d = _call_delta(spot, short_call, T, _iv(crows, short_call))
    pop_between = round((1 - sp_d - sc_d) * 100)   # P(finish between shorts)

    # Expectancy gate: breakeven win-rate = 1 − credit/width. If POP doesn't beat
    # it by EDGE_MARGIN, the condor is a negative-EV bet regardless of raw win %.
    breakeven_pop = round((1 - credit / max_w) * 100)
    if pop_between < breakeven_pop + EDGE_MARGIN:
        out["note"] = (f"Negative edge: POP {pop_between}% < breakeven "
                       f"{breakeven_pop}%+{EDGE_MARGIN} (credit ${credit:.2f} / {max_w:.0f} wide).")
        return out

    max_loss = round((max_w - credit) * 100)
    out["candidate"] = {
        "ticker": ticker, "expiry": exp, "dte": dte,
        "short_put": short_put, "long_put": long_put,
        "short_call": short_call, "long_call": long_call,
        "wing": max_w, "credit": credit,
        "put_credit": put_credit, "call_credit": call_credit,
        "max_profit_usd": round(credit * 100), "max_loss_usd": max_loss,
        "rr": round(credit / max(max_w - credit, 0.01), 2),
        "lower_be": round(short_put - credit, 2),
        "upper_be": round(short_call + credit, 2),
        "short_put_delta": round(sp_d, 2), "short_call_delta": round(sc_d, 2),
        "pop_pct": pop_between, "breakeven_pop": breakeven_pop,
        "edge_pts": pop_between - breakeven_pop,
        "vol_band": out["vol_band"], "vix": round(vix, 1),
        "legs": [
            {"action": "BUY",  "strike": long_put,   "opt": "P", "qty": 1, "mid": put_mids[long_put]},
            {"action": "SELL", "strike": short_put,  "opt": "P", "qty": 1, "mid": put_mids[short_put]},
            {"action": "SELL", "strike": short_call, "opt": "C", "qty": 1, "mid": call_mids[short_call]},
            {"action": "BUY",  "strike": long_call,  "opt": "C", "qty": 1, "mid": call_mids[long_call]},
        ],
    }
    return out


def _iv(rows: list, strike: float) -> float:
    """IV for a given strike from chain rows (fallback to 0.18 if missing)."""
    for r in rows:
        if r["strike"] == strike:
            return float(r.get("iv") or 0) or 0.18
    return 0.18


def fmt_slack(result: dict) -> str:
    c = result.get("candidate")
    if not c:
        return ""
    vol = f"VIX {c['vix']} ({c['vol_band']})"
    return "\n".join([
        f":shield: *GEX-Anchored Iron Condor* — {c['ticker']} "
        f"({c['expiry']}, {c['dte']}DTE)  _positive gamma · {vol}_",
        f"  SELL ${c['short_put']:.0f}P / BUY ${c['long_put']:.0f}P   "
        f"(short {c['short_put_delta']:.2f}Δ)",
        f"  SELL ${c['short_call']:.0f}C / BUY ${c['long_call']:.0f}C  "
        f"(short {c['short_call_delta']:.2f}Δ)",
        f"  Credit +${c['credit']:.2f} (${c['max_profit_usd']})  |  "
        f"Max loss -${c['max_loss_usd']}  |  R/R {c['rr']:.2f}",
        f"  POP ~{c['pop_pct']}% vs breakeven {c.get('breakeven_pop','?')}%  "
        f"(*+{c.get('edge_pts','?')}pt edge*)  |  profit ${c['lower_be']:.0f}–${c['upper_be']:.0f}",
        f"  _manage at 50% — close to cut the gamma tail_",
    ])
