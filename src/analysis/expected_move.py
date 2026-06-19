"""
Expected move — the market's implied ±range by a chosen expiry.

Most-accurate method: the **ATM straddle**. The at-the-money straddle price *is*
the market's expected move (it bakes in the real IV at that expiry, including
skew/term structure), scaled to ~1 standard deviation by ×0.85.

  EM ≈ 0.85 × (ATM call mid + ATM put mid)

The ATM strike is found where |call − put| is smallest (put-call parity: at the
forward, call ≈ put) — so no separate spot fetch is needed, and 0DTE works
naturally because the straddle reflects whatever time value remains.
"""
import logging
import time

logger = logging.getLogger(__name__)

_EM_FACTOR = 0.85                      # straddle → 1-SD expected move
_cache: dict = {}                      # underlying -> (ts, by_exp)
_TTL = 60                              # seconds; lets the webpage flip DTE cheaply


def _em_calc(atm_strike: float, call_mid: float, put_mid: float, factor: float = _EM_FACTOR) -> dict:
    """Pure: expected move from an ATM straddle. forward ≈ K + (C − P) by parity."""
    straddle = call_mid + put_mid
    forward  = atm_strike + (call_mid - put_mid)
    em       = factor * straddle
    return {
        "straddle":   round(straddle, 2),
        "forward":    round(forward, 2),
        "em_dollars": round(em, 2),
        "em_pct":     round(em / forward * 100, 2) if forward else 0.0,
        "lower":      round(forward - em, 2),
        "upper":      round(forward + em, 2),
    }


def _fetch_by_exp(underlying: str) -> dict:
    """Build {dte: {exp, strikes:{K:{C,P,iv}}}} from one Alpaca chain fetch."""
    from datetime import date
    from src.live.alpaca_options import _data
    from alpaca.data.requests import OptionChainRequest
    chain = _data().get_option_chain(OptionChainRequest(underlying_symbol=underlying, feed="indicative"))
    today = date.today()
    by_exp: dict = {}
    for sym, snap in chain.items():
        try:
            tp = len(sym) - 9
            if tp < 0 or sym[tp] not in ("C", "P"):
                continue
            is_call = sym[tp] == "C"
            strike  = int(sym[-8:]) / 1000.0
            contract = getattr(snap, "contract_details", None) or getattr(snap, "details", None)
            exp = str(getattr(contract, "expiration_date", "") or "") if contract else ""
            if not exp:
                raw = sym[tp - 6:tp]
                exp = f"20{raw[0:2]}-{raw[2:4]}-{raw[4:6]}"
            exp_d = date.fromisoformat(exp[:10])
            dtex  = (exp_d - today).days
            if dtex < 0:
                continue
            q   = getattr(snap, "latest_quote", None)
            bid = float(getattr(q, "bid_price", 0) or 0) if q else 0
            ask = float(getattr(q, "ask_price", 0) or 0) if q else 0
            if bid <= 0 or ask <= 0:
                continue
            mid = (bid + ask) / 2
            iv  = float(getattr(snap, "implied_volatility", 0) or 0)
            slot = by_exp.setdefault(dtex, {"exp": exp[:10], "strikes": {}})
            cell = slot["strikes"].setdefault(strike, {})
            cell["C" if is_call else "P"] = mid
            if iv:
                cell["iv"] = iv
        except Exception:
            continue
    return by_exp


def _load_chain(underlying: str) -> dict:
    now = time.time()
    c = _cache.get(underlying)
    if c and now - c[0] < _TTL:
        return c[1]
    by_exp = _fetch_by_exp(underlying)
    _cache[underlying] = (now, by_exp)
    return by_exp


def compute(underlying: str = "SPY", dte: int = 0, factor: float = _EM_FACTOR) -> dict | None:
    """Expected move for the expiry nearest `dte` (0 = today / 0DTE)."""
    by_exp = _load_chain(underlying)
    if not by_exp:
        return None
    target = min(by_exp.keys(), key=lambda k: abs(k - dte))
    info   = by_exp[target]
    cands  = [(k, v) for k, v in info["strikes"].items() if "C" in v and "P" in v]
    if not cands:
        return None
    atm_k, atm = min(cands, key=lambda kv: abs(kv[1]["C"] - kv[1]["P"]))
    em = _em_calc(atm_k, atm["C"], atm["P"], factor)
    return {
        "underlying": underlying, "requested_dte": dte, "dte": target,
        "expiry": info["exp"], "atm_strike": atm_k,
        "atm_iv": round(atm.get("iv", 0) * 100, 1),
        "available_dtes": sorted(by_exp.keys())[:14],
        **em,
    }
