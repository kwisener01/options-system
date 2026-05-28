"""
risk_profile.py ? Live option risk profiler for Alpaca positions.

Fetches open positions from the Alpaca account (and optionally pmcc_positions.json),
pulls live greeks from the Alpaca chain (Black-Scholes fallback when feed misses),
then prints:
  - Per-leg greek table
  - Portfolio aggregate greeks (delta / gamma / theta / vega ? 100 multiplier)
  - P&L scenario grid (?2 / 5 / 10 / 15 % underlying moves) via BS repricing
  - Max loss / max gain per position

Usage:
    python risk_profile.py                    # Alpaca account positions only
    python risk_profile.py --pmcc             # also merge pmcc_positions.json
    python risk_profile.py --no-alpaca        # PMCC file only (no API needed)
    python risk_profile.py --ticker SOFI      # filter to one underlying
"""

import argparse
import json
import logging
import math
import os
import sys
from datetime import date, datetime
from typing import Optional

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

POSITIONS_FILE = os.path.join(os.path.dirname(__file__), "pmcc_positions.json")
RISK_FREE_RATE = 0.053   # approx current 3-month T-bill
SCENARIO_MOVES = [-0.15, -0.10, -0.05, -0.02, 0.00, +0.02, +0.05, +0.10, +0.15]


# -- Black-Scholes engine -------------------------------------------------------

def _ncdf(x: float) -> float:
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def bs_price_and_greeks(S: float, K: float, T: float, r: float,
                         sigma: float, is_call: bool) -> dict:
    """Return BS price + greeks. All zeros when T<=0 or sigma<=0."""
    zero = {"price": 0.0, "delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
    if T <= 0 or sigma <= 0.01 or S <= 0 or K <= 0:
        return zero
    d1  = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2  = d1 - sigma * math.sqrt(T)
    nd1 = _ncdf(d1)
    nd2 = _ncdf(d2)
    npd1 = math.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)
    disc = math.exp(-r * T)
    if is_call:
        price = S * nd1 - K * disc * nd2
        delta = nd1
        rho_nd = nd2
    else:
        price = K * disc * _ncdf(-d2) - S * _ncdf(-d1)
        delta = nd1 - 1.0
        rho_nd = -_ncdf(-d2)
    gamma = npd1 / (S * sigma * math.sqrt(T))
    theta = (-(S * npd1 * sigma) / (2 * math.sqrt(T))
             - r * K * disc * (nd2 if is_call else -_ncdf(-d2))) / 365
    vega  = S * npd1 * math.sqrt(T) / 100   # per 1 IV percentage-point
    return {
        "price": max(price, 0.0),
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega":  vega,
    }


# -- OCC symbol builder (inverse of parser) ------------------------------------

def _build_occ(underlying: str, expiry: date, is_call: bool, strike: float) -> str:
    return (f"{underlying.upper()}{expiry.strftime('%y%m%d')}"
            f"{'C' if is_call else 'P'}{round(strike * 1000):08d}")


# -- OCC symbol parser ----------------------------------------------------------

def parse_occ(symbol: str) -> Optional[dict]:
    """
    Parse OCC option symbol for any root length.
    Returns {underlying, expiry (date), is_call, strike} or None.
    """
    try:
        type_pos = len(symbol) - 9
        if type_pos < 1 or symbol[type_pos] not in ("C", "P"):
            return None
        underlying = symbol[:type_pos - 6]
        ds = symbol[type_pos - 6: type_pos]
        expiry = date(2000 + int(ds[0:2]), int(ds[2:4]), int(ds[4:6]))
        return {
            "underlying": underlying,
            "expiry":     expiry,
            "is_call":    symbol[type_pos] == "C",
            "strike":     int(symbol[type_pos + 1:]) / 1000.0,
        }
    except Exception:
        return None


# -- Alpaca clients (lazy) ------------------------------------------------------

_tc = _dc = None


def _clients():
    global _tc, _dc
    if _tc is None:
        from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY, IS_PAPER
        from alpaca.trading.client import TradingClient
        try:
            from alpaca.data.historical.option import OptionHistoricalDataClient
        except ImportError:
            from alpaca.data.historical import OptionHistoricalDataClient
        _tc = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=IS_PAPER)
        _dc = OptionHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    return _tc, _dc


# -- Position loaders -----------------------------------------------------------

def fetch_account_positions() -> list[dict]:
    """Pull open option legs from Alpaca account."""
    try:
        tc, _ = _clients()
        raw = tc.get_all_positions()
    except Exception as e:
        log.warning("Alpaca position fetch failed: %s", e)
        return []
    rows = []
    for p in raw:
        parsed = parse_occ(p.symbol)
        if not parsed:
            continue
        qty = int(float(p.qty))
        rows.append({
            "source":      "alpaca",
            "id":          p.symbol,
            "label":       f"{parsed['underlying']} {'C' if parsed['is_call'] else 'P'}"
                           f"{parsed['strike']:.0f} {parsed['expiry'].strftime('%b%y')}",
            "symbol":      p.symbol,
            "underlying":  parsed["underlying"],
            "strike":      parsed["strike"],
            "expiry":      parsed["expiry"].isoformat(),
            "is_call":     parsed["is_call"],
            "qty":         qty,
            "entry_price": float(p.avg_entry_price or 0),
        })
    return rows


def load_pmcc_positions() -> list[dict]:
    """Load pmcc_positions.json into per-leg dicts."""
    if not os.path.exists(POSITIONS_FILE):
        return []
    with open(POSITIONS_FILE) as f:
        data = json.load(f)
    rows = []
    for pos in data:
        ticker = pos["ticker"]
        n      = pos.get("contracts", 1)
        rows.append({
            "source":      "pmcc",
            "id":          pos["id"] + "_long",
            "group_id":    pos["id"],
            "label":       f"{ticker} LEAPS ${pos['long_strike']:.0f}",
            "symbol":      pos.get("long_symbol", ""),
            "underlying":  ticker,
            "strike":      pos["long_strike"],
            "expiry":      pos["long_expiry"],
            "is_call":     True,
            "qty":         n,
            "entry_price": pos["long_cost"],
        })
        if pos.get("short_symbol"):
            rows.append({
                "source":      "pmcc",
                "id":          pos["id"] + "_short",
                "group_id":    pos["id"],
                "label":       f"{ticker} Short ${pos['short_strike']:.0f}C",
                "symbol":      pos["short_symbol"],
                "underlying":  ticker,
                "strike":      pos["short_strike"],
                "expiry":      pos["short_expiry"],
                "is_call":     True,
                "qty":         -n,
                "entry_price": pos["short_credit"],
            })
    return rows


# -- Live chain / greek fetch ---------------------------------------------------

def fetch_chain_greeks(underlying: str) -> dict:
    """
    Pull full option chain for underlying from Alpaca.
    Returns {occ_symbol: {mid, iv, delta, gamma, theta, vega}}.
    No DTE filter ? we want LEAPS too.
    """
    try:
        from alpaca.data.requests import OptionChainRequest
        _, dc = _clients()
        chain = dc.get_option_chain(
            OptionChainRequest(underlying_symbol=underlying, feed="indicative")
        )
    except Exception as e:
        log.warning("Chain fetch failed %s: %s", underlying, e)
        return {}
    result = {}
    for sym, snap in chain.items():
        g   = getattr(snap, "greeks", None)
        q   = getattr(snap, "latest_quote", None)
        iv  = float(getattr(snap, "implied_volatility", 0) or 0)
        if not iv and g:
            iv = float(getattr(g, "implied_volatility", 0) or 0)
        bid = float(getattr(q, "bid_price", 0) or 0) if q else 0.0
        ask = float(getattr(q, "ask_price", 0) or 0) if q else 0.0
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0
        result[sym] = {
            "mid":   round(mid, 4),
            "iv":    round(iv, 4),
            "delta": round(float(getattr(g, "delta", 0) or 0), 4) if g else 0.0,
            "gamma": round(float(getattr(g, "gamma", 0) or 0), 4) if g else 0.0,
            "theta": round(float(getattr(g, "theta", 0) or 0), 4) if g else 0.0,
            "vega":  round(float(getattr(g, "vega",  0) or 0), 4) if g else 0.0,
        }
    return result


def get_spot(ticker: str) -> float:
    try:
        import yfinance as yf
        return round(float(yf.Ticker(ticker).fast_info.last_price), 2)
    except Exception:
        return 0.0


# -- Leg enrichment -------------------------------------------------------------

def enrich_legs(legs: list[dict], ticker_filter: Optional[str] = None,
                use_api: bool = True) -> list[dict]:
    """
    Add live greeks (Alpaca chain -> BS fallback) to each leg.
    Greeks are signed for position direction (long positive, short negative).
    pos_* fields are scaled by qty and the 100-share multiplier.
    Set use_api=False to skip Alpaca chain fetches (pure BS mode).
    """
    if ticker_filter:
        legs = [l for l in legs if l["underlying"].upper() == ticker_filter.upper()]
    if not legs:
        return []

    underlyings = list({l["underlying"] for l in legs})
    chains: dict = {}
    spots:  dict = {}
    for u in underlyings:
        if use_api:
            print(f"  Fetching chain data for {u}...")
            chains[u] = fetch_chain_greeks(u)
        else:
            chains[u] = {}
        spots[u] = get_spot(u)
        print(f"    {u}: {len(chains[u])} chain contracts, spot=${spots[u]:.2f}")

    today    = date.today()
    enriched = []
    for leg in legs:
        u       = leg["underlying"]
        sym     = leg["symbol"]
        S       = spots.get(u, 0.0)
        K       = leg["strike"]
        T       = max((date.fromisoformat(leg["expiry"]) - today).days, 0) / 365
        qty     = leg["qty"]
        sign    = 1 if qty > 0 else -1
        n       = abs(qty)
        is_call = leg["is_call"]

        cd = chains.get(u, {}).get(sym, {})
        mid   = cd.get("mid",   0.0)
        iv    = cd.get("iv",    0.0)
        delta = cd.get("delta", 0.0)
        gamma = cd.get("gamma", 0.0)
        theta = cd.get("theta", 0.0)
        vega  = cd.get("vega",  0.0)

        bs_source = False
        if abs(delta) < 1e-6 and S > 0 and T > 0:
            iv_for_bs = iv if iv > 0.01 else 0.45  # crude fallback IV
            g = bs_price_and_greeks(S, K, T, RISK_FREE_RATE, iv_for_bs, is_call)
            delta = g["delta"]
            gamma = g["gamma"]
            theta = g["theta"]
            vega  = g["vega"]
            if mid == 0.0:
                mid = g["price"]
            if iv < 0.01:
                iv = iv_for_bs
            bs_source = True

        enriched.append({
            **leg,
            "spot":       S,
            "T":          T,
            "iv_raw":     iv,                       # decimal, for BS repricing
            "iv":         round(iv * 100, 1),       # display %
            "mid":        round(mid, 3),
            "delta":      round(delta * sign, 4),   # per-share, signed
            "gamma":      round(gamma * sign, 4),
            "theta":      round(theta * sign, 4),
            "vega":       round(vega  * sign, 4),
            "pos_delta":  round(delta * sign * n * 100, 2),
            "pos_gamma":  round(gamma * sign * n * 100, 2),
            "pos_theta":  round(theta * sign * n * 100, 2),
            "pos_vega":   round(vega  * sign * n * 100, 2),
            "bs_source":  bs_source,
        })
    return enriched


# -- Scenario P&L --------------------------------------------------------------

def _leg_scenario_pnl(leg: dict, move_pct: float) -> float:
    """
    Estimate TOTAL unrealized P&L (vs entry price) for one leg if the
    underlying moves move_pct% from its current spot right now.

    Long:  pnl = (new_option_price - entry_price) * n * 100
    Short: pnl = (entry_price - new_option_price) * n * 100

    Uses BS repricing when IV is non-trivial and the option is not
    deeply ITM (where chain IV is often unreliable).  Falls back to
    delta-gamma approximation otherwise.
    """
    S_new  = leg["spot"] * (1 + move_pct)
    qty    = leg["qty"]
    sign   = 1 if qty > 0 else -1
    n      = abs(qty)
    entry  = leg["entry_price"]
    iv     = leg["iv_raw"]
    T      = leg["T"]
    S      = leg["spot"]
    K      = leg["strike"]

    # Skip BS repricing when the option is >10% ITM — chain IV is unreliable
    # for deep ITM options and produces nonsense P&L curves.
    itm_frac = max((S - K) / S if leg["is_call"] else (K - S) / S, 0.0)
    use_bs   = T > 0 and iv > 0.01 and S_new > 0 and itm_frac < 0.10

    if use_bs:
        new_price = bs_price_and_greeks(S_new, K, T, RISK_FREE_RATE, iv,
                                         leg["is_call"])["price"]
    else:
        # Delta-gamma approximation from current mid
        dS        = S * move_pct
        d         = leg["delta"]   # already signed for direction
        g         = leg["gamma"]
        delta_pnl = (d * dS + 0.5 * g * dS ** 2) * n * 100
        # Translate to entry-based P&L: add current unrealized
        mid_now   = leg["mid"]
        unrealized = (mid_now - entry) * sign * n * 100
        return round(unrealized + delta_pnl, 2)

    # Entry-based P&L: (new_price - entry) for longs, (entry - new_price) for shorts
    pnl = (new_price - entry) * sign * n * 100
    return round(pnl, 2)


def build_scenario_table(enriched: list[dict]) -> dict:
    """
    Returns {underlying: {move_pct: total_pnl}}.
    Adds a 'PORTFOLIO' key when more than one underlying is present.
    """
    table: dict = {}
    for leg in enriched:
        u = leg["underlying"]
        table.setdefault(u, {m: 0.0 for m in SCENARIO_MOVES})
        for m in SCENARIO_MOVES:
            table[u][m] = round(table[u][m] + _leg_scenario_pnl(leg, m), 2)

    if len(table) > 1:
        portfolio = {m: 0.0 for m in SCENARIO_MOVES}
        for u_data in table.values():
            for m in SCENARIO_MOVES:
                portfolio[m] = round(portfolio[m] + u_data[m], 2)
        table["PORTFOLIO"] = portfolio
    return table


def build_expiry_table(enriched: list[dict]) -> dict:
    """
    Compute P&L AT EXPIRATION for each scenario price (T=0, pure intrinsic).
    This shows the classic tent shape for spreads/iron condors.
    Returns {underlying: {move_pct: expiry_pnl}}.
    """
    table: dict = {}
    for leg in enriched:
        u       = leg["underlying"]
        K       = leg["strike"]
        qty     = leg["qty"]
        sign    = 1 if qty > 0 else -1
        n       = abs(qty)
        entry   = leg["entry_price"]
        is_call = leg["is_call"]
        S       = leg["spot"]
        table.setdefault(u, {m: 0.0 for m in SCENARIO_MOVES})
        for m in SCENARIO_MOVES:
            S_new = S * (1 + m)
            if is_call:
                expiry_price = max(S_new - K, 0.0)
            else:
                expiry_price = max(K - S_new, 0.0)
            pnl = (expiry_price - entry) * sign * n * 100
            table[u][m] = round(table[u][m] + pnl, 2)

    if len(table) > 1:
        portfolio = {m: 0.0 for m in SCENARIO_MOVES}
        for u_data in table.values():
            for m in SCENARIO_MOVES:
                portfolio[m] = round(portfolio[m] + u_data[m], 2)
        table["PORTFOLIO"] = portfolio
    return table


# -- Spread auto-grouping -------------------------------------------------------

def auto_group_spreads(legs: list[dict]) -> list[dict]:
    """
    Detect vertical spread pairs in Alpaca account legs (same underlying +
    same expiry + same option type + one long / one short).  Assigns a shared
    group_id so max_risk_per_position can compute defined-risk payoffs instead
    of treating each leg as a naked position.

    Does NOT touch legs that already have a group_id (e.g., PMCC legs).
    Multiple same-type same-expiry pairs are matched by closest strike.
    """
    from collections import defaultdict

    ungrouped = [l for l in legs if not l.get("group_id")]
    buckets: dict = defaultdict(list)
    for leg in ungrouped:
        key = (leg["underlying"], leg["expiry"], leg["is_call"])
        buckets[key].append(leg)

    paired = 0
    for (u, exp, is_call), bucket in buckets.items():
        longs  = sorted([l for l in bucket if l["qty"] > 0], key=lambda l: l["strike"])
        shorts = sorted([l for l in bucket if l["qty"] < 0], key=lambda l: l["strike"])

        for long_leg, short_leg in zip(longs, shorts):
            lo_k = min(long_leg["strike"], short_leg["strike"])
            hi_k = max(long_leg["strike"], short_leg["strike"])
            typ  = "C" if is_call else "P"
            gid  = f"{u}_{typ}{lo_k:.0f}_{hi_k:.0f}_{exp}"

            # Descriptive spread name
            if is_call:
                name = (f"{u} Bear Call {lo_k:.0f}/{hi_k:.0f}"
                        if short_leg["strike"] < long_leg["strike"]
                        else f"{u} Bull Call {lo_k:.0f}/{hi_k:.0f}")
            else:
                name = (f"{u} Bull Put {lo_k:.0f}/{hi_k:.0f}"
                        if short_leg["strike"] > long_leg["strike"]
                        else f"{u} Bear Put {lo_k:.0f}/{hi_k:.0f}")

            long_leg["group_id"]   = short_leg["group_id"]   = gid
            long_leg["spread_name"] = short_leg["spread_name"] = name
            paired += 1

    if paired:
        print(f"  Auto-grouped {paired} vertical spread pair(s)")
    return legs


# -- Max loss / max gain --------------------------------------------------------

def max_risk_per_position(enriched: list[dict]) -> list[dict]:
    """
    Group legs by group_id (PMCC) or individual id.
    Computes max loss / max gain using payoff math.
    """
    groups: dict = {}
    for leg in enriched:
        gid = leg.get("group_id") or leg["id"]
        groups.setdefault(gid, []).append(leg)

    summaries = []
    for gid, legs in groups.items():
        label      = (legs[0].get("spread_name")
                      or legs[0].get("label")
                      or legs[0]["symbol"])
        underlying = legs[0]["underlying"]
        spot       = legs[0]["spot"]
        n          = abs(legs[0]["qty"])

        if len(legs) == 1:
            leg = legs[0]
            if leg["qty"] > 0:          # long option
                max_loss = round(leg["entry_price"] * n * 100, 2)
                max_gain = None          # unlimited for calls; (K-prem)*100 for puts
            else:                       # short naked
                max_loss = None
                max_gain = round(leg["entry_price"] * n * 100, 2)

        elif len(legs) == 2:
            long_leg  = next((l for l in legs if l["qty"] > 0), None)
            short_leg = next((l for l in legs if l["qty"] < 0), None)
            if long_leg and short_leg:
                net_debit = (long_leg["entry_price"] - short_leg["entry_price"]) * n * 100
                same_exp  = long_leg["expiry"] == short_leg["expiry"]
                width     = abs(long_leg["strike"] - short_leg["strike"]) * n * 100

                if net_debit >= 0:      # net debit (debit spread or PMCC)
                    max_loss = round(net_debit, 2)
                    if same_exp:        # debit vertical
                        max_gain = round(width - net_debit, 2)
                    else:               # diagonal / PMCC
                        max_gain = round(width - net_debit, 2)
                else:                   # net credit (bull put / bear call)
                    net_credit = -net_debit
                    max_gain   = round(net_credit, 2)
                    max_loss   = round(width - net_credit, 2)
            else:
                max_gain = max_loss = None
        else:
            max_gain = max_loss = None

        summaries.append({
            "label":      label,
            "underlying": underlying,
            "spot":       spot,
            "legs":       len(legs),
            "max_gain":   max_gain,
            "max_loss":   max_loss,
        })
    return summaries


# -- Report printer -------------------------------------------------------------

def _money(val: Optional[float]) -> str:
    if val is None:
        return "unlimited"
    prefix = "$" if val >= 0 else "-$"
    return f"{prefix}{abs(val):,.2f}"


def print_risk_report(enriched: list[dict], scenarios: dict,
                      expiry_table: dict, max_risks: list[dict]):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    W = 74

    print(f"\n{'=' * W}")
    print(f"  OPTION RISK PROFILE  --  {now}")
    print(f"{'=' * W}")

    # -- Per-leg greek table ----------------------------------------------------
    print(f"\n  {'POSITION':<32} {'T':<5} {'DTE':>4} {'IV':>6}"
          f"  {'Delta':>7}  {'Gamma':>7}  {'Theta/day':>7}  {'V/pt':>7}")
    print(f"  {'-' * (W - 2)}")
    for leg in enriched:
        t    = "C" if leg["is_call"] else "P"
        side = "L" if leg["qty"] > 0 else "S"
        dte  = max(int(leg["T"] * 365), 0)
        flag = "*" if leg["bs_source"] else " "
        print(f"  {leg['label']:<32} {t}{side:<4} {dte:>4} {leg['iv']:>5.1f}%"
              f"  {leg['delta']:>7.4f}  {leg['gamma']:>7.4f}"
              f"  {leg['theta']:>7.4f}  {leg['vega']:>7.4f}{flag}")
    print(f"\n  * greeks via Black-Scholes (contract not in live feed or IV missing)")

    # -- Aggregate greeks per underlying ---------------------------------------
    print(f"\n{'-' * W}")
    print(f"  PORTFOLIO GREEKS  (per-position, scaled ? qty ? 100 multiplier)")
    print(f"{'-' * W}")
    underlyings = list({l["underlying"] for l in enriched})
    for u in underlyings:
        ul   = [l for l in enriched if l["underlying"] == u]
        spot = ul[0]["spot"]
        ttl  = {k: round(sum(l[f"pos_{k}"] for l in ul), 2)
                for k in ("delta", "gamma", "theta", "vega")}
        dol_move = round(ttl["delta"] * spot * 0.01, 2)  # $ P&L per 1% move
        print(f"\n  {u} @ ${spot:.2f}")
        print(f"    Net Delta  {ttl['delta']:>+10.2f}   (? ${dol_move:+.2f} per 1% move in {u})")
        print(f"    Net Gamma  {ttl['gamma']:>+10.2f}   (delta change per $1 move)")
        print(f"    Net Theta  {ttl['theta']:>+10.2f}   ($/day ? negative = you pay decay)")
        print(f"    Net Vega   {ttl['vega']:>+10.2f}   ($ per +1 IV point)")

    # -- At-expiration payoff (tent shape) -------------------------------------
    print(f"\n{'-' * W}")
    print(f"  PAYOFF AT EXPIRATION  (tent shape -- pure intrinsic at T=0)")
    print(f"{'-' * W}")
    for u, moves in expiry_table.items():
        if u == "PORTFOLIO":
            continue
        ul   = [l for l in enriched if l["underlying"] == u]
        spot = ul[0]["spot"]
        all_pnl = list(moves.values())
        max_abs = max(abs(p) for p in all_pnl) or 1
        print(f"\n  {u}  (current spot ${spot:.2f})")
        print(f"    {'Move':>8}   {'New Price':>10}   {'Expiry P&L':>12}   bar")
        print(f"    {'-' * 55}")
        for m in SCENARIO_MOVES:
            pnl    = moves[m]
            new_px = spot * (1 + m)
            marker = " <-- now" if m == 0 else ""
            bar_w  = int(abs(pnl) / max_abs * 20)
            bar    = ("+" if pnl >= 0 else "-") * bar_w
            print(f"    {m:>+7.0%}   ${new_px:>9.2f}   {_money(pnl):>12}   {bar}{marker}")

    # -- Scenario table (BS, from entry) ---------------------------------------
    print(f"\n{'-' * W}")
    print(f"  P&L SCENARIOS TODAY  (unrealized vs entry; BS repricing, same IV)")
    print(f"{'-' * W}")
    for u, moves in scenarios.items():
        if u == "PORTFOLIO":
            continue
        ul   = [l for l in enriched if l["underlying"] == u]
        spot = ul[0]["spot"]
        print(f"\n  {u}  (current spot ${spot:.2f})")
        print(f"    {'Move':>8}   {'New Price':>10}   {'Today P&L':>12}   bar")
        print(f"    {'-' * 55}")
        all_pnl = list(moves.values())
        max_abs = max(abs(p) for p in all_pnl) or 1
        for m in SCENARIO_MOVES:
            pnl    = moves[m]
            new_px = spot * (1 + m)
            marker = " <-- now" if m == 0 else ""
            bar_w  = int(abs(pnl) / max_abs * 20)
            bar    = ("+" if pnl >= 0 else "-") * bar_w
            print(f"    {m:>+7.0%}   ${new_px:>9.2f}   {_money(pnl):>12}   {bar}{marker}")

    if "PORTFOLIO" in scenarios:
        print(f"\n  PORTFOLIO TOTAL (all underlyings)")
        print(f"    {'Move':>8}   {'Expiry P&L':>12}   {'Today P&L':>12}")
        print(f"    {'-' * 35}")
        for m in SCENARIO_MOVES:
            e_pnl = expiry_table["PORTFOLIO"][m]
            t_pnl = scenarios["PORTFOLIO"][m]
            print(f"    {m:>+7.0%}   {_money(e_pnl):>12}   {_money(t_pnl):>12}")

    # -- Max loss / max gain ----------------------------------------------------
    print(f"\n{'-' * W}")
    print(f"  MAX LOSS / MAX GAIN  (payoff at expiration)")
    print(f"{'-' * W}")
    for r in max_risks:
        gain = _money(r["max_gain"])
        loss = _money(r["max_loss"])
        print(f"  {r['label']:<40}  gain: {gain:>12}   loss: {loss:>12}")

    print(f"\n{'=' * W}\n")


# -- Analyze mode (hypothetical positions) -------------------------------------

def _parse_expiry(s: str) -> Optional[date]:
    today = date.today()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%m-%d-%y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    # MM/DD with no year → nearest future occurrence
    for fmt in ("%m/%d", "%m-%d"):
        try:
            d = datetime.strptime(f"{s}/{today.year}", fmt + "/%Y").date()
            if d < today:
                d = d.replace(year=today.year + 1)
            return d
        except ValueError:
            pass
    return None


def parse_leg_spec(spec: str) -> Optional[dict]:
    """
    Parse a single leg string: SYMBOL EXPIRY TYPE STRIKE SIDE [QTY] [PRICE]
      TYPE  : C or P
      SIDE  : B/BUY/L or S/SELL
      QTY   : default 1
      PRICE : default 0.0 — filled from live chain if omitted
    """
    parts = spec.strip().split()
    if len(parts) < 5:
        return None
    try:
        underlying = parts[0].upper()
        expiry     = _parse_expiry(parts[1])
        if expiry is None:
            return None
        is_call  = parts[2].upper().startswith("C")
        strike   = float(parts[3])
        side_raw = parts[4].upper()
        sign     = 1 if side_raw in ("B", "BUY", "L", "LONG") else -1
        qty      = sign * (int(parts[5]) if len(parts) > 5 else 1)
        price    = float(parts[6]) if len(parts) > 6 else 0.0
        sym      = _build_occ(underlying, expiry, is_call, strike)
        return {
            "source":      "analyze",
            "id":          sym,
            "label":       (f"{underlying} {'C' if is_call else 'P'}{strike:.0f} "
                            f"{expiry.strftime('%b%y')} "
                            f"{'Long' if qty > 0 else 'Short'}"),
            "symbol":      sym,
            "underlying":  underlying,
            "strike":      strike,
            "expiry":      expiry.isoformat(),
            "is_call":     is_call,
            "qty":         qty,
            "entry_price": price,
        }
    except (ValueError, IndexError):
        return None


def analyze_legs_interactive() -> list[dict]:
    """Prompt for up to 4 hypothetical legs and return them."""
    print()
    print("  ANALYZE MODE  —  enter up to 4 legs, blank line when done")
    print()
    print("  Format:  SYMBOL  EXPIRY      TYPE  STRIKE  SIDE  [QTY]  [PRICE]")
    print("  Example: SPY     2026-06-20  C     744     S     1      9.23")
    print("           SPY     06/20/26    P     730     B                    (fetches live price)")
    print()
    legs: list[dict] = []
    while len(legs) < 4:
        try:
            raw = input(f"  Leg {len(legs) + 1}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            if legs:
                break
            continue
        leg = parse_leg_spec(raw)
        if leg:
            side  = "Long" if leg["qty"] > 0 else "Short"
            price = f"@ ${leg['entry_price']:.2f}" if leg["entry_price"] > 0 else "(uses live mid)"
            print(f"    + {side} {leg['label']} {price}")
            legs.append(leg)
        else:
            print("    Could not parse — format: SPY 2026-06-20 C 744 S 1")
    return legs


# -- Main -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Alpaca option risk profiler")
    ap.add_argument("--pmcc",      action="store_true",
                    help="Also load pmcc_positions.json")
    ap.add_argument("--no-alpaca", action="store_true",
                    help="Skip live Alpaca account positions (use PMCC file only)")
    ap.add_argument("--ticker",    default=None,
                    help="Filter to one underlying symbol")
    ap.add_argument("--analyze",   action="store_true",
                    help="Analyze a hypothetical position (enter legs interactively)")
    args = ap.parse_args()

    # ── Analyze mode: hypothetical legs entered interactively ──────────────────
    if args.analyze:
        legs = analyze_legs_interactive()
        if not legs:
            print("No legs entered.")
            sys.exit(0)
        legs = auto_group_spreads(legs)
        print(f"\nFetching live data for {len(legs)} leg(s)...")
        enriched = enrich_legs(legs, use_api=True)
        if not enriched:
            print("Enrichment failed — check API credentials.")
            sys.exit(1)
        # Fill any missing entry prices with current live mid
        for leg in enriched:
            if leg["entry_price"] == 0.0 and leg["mid"] > 0:
                leg["entry_price"] = leg["mid"]
        scenarios    = build_scenario_table(enriched)
        expiry_table = build_expiry_table(enriched)
        max_risks    = max_risk_per_position(enriched)
        print_risk_report(enriched, scenarios, expiry_table, max_risks)
        return

    # ── Normal mode: open positions from Alpaca account / PMCC file ───────────
    legs: list[dict] = []

    if not args.no_alpaca:
        print("Fetching Alpaca account positions...")
        al = fetch_account_positions()
        print(f"  Found {len(al)} option leg(s) in Alpaca account")
        legs.extend(al)

    if args.pmcc or args.no_alpaca:
        print("Loading PMCC positions from pmcc_positions.json...")
        pm = load_pmcc_positions()
        existing = {l["symbol"] for l in legs}
        fresh    = [l for l in pm if l["symbol"] not in existing]
        print(f"  Added {len(fresh)} new leg(s) from PMCC file")
        legs.extend(fresh)

    if not legs:
        print("No positions found.\n"
              "  Use --pmcc to load pmcc_positions.json,\n"
              "  or --no-alpaca --pmcc to run without API access.\n"
              "  Use --analyze to model a hypothetical position.")
        sys.exit(0)

    legs = auto_group_spreads(legs)

    print(f"\nEnriching {len(legs)} leg(s) with live greeks...")
    enriched = enrich_legs(legs, ticker_filter=args.ticker, use_api=not args.no_alpaca)
    if not enriched:
        print("No legs after filtering.")
        sys.exit(0)

    scenarios    = build_scenario_table(enriched)
    expiry_table = build_expiry_table(enriched)
    max_risks    = max_risk_per_position(enriched)
    print_risk_report(enriched, scenarios, expiry_table, max_risks)


if __name__ == "__main__":
    main()
