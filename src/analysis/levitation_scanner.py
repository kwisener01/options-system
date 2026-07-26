"""
Daily SPY "Levitation" trade — 0DTE ATM bull-put spread + cheap temp hedge.

Structure (all same-day expiry puts):
  SELL 1 put at Ks (~ATM)            \\  bull-put spread, credit C
  BUY  1 put at Kl = Ks - W          /
  BUY  1 put at Kh < Kl, cost H <= C/2   -> "temp" hedge, net credit N = C - H

Why the hedge: near expiry the cheap OTM put carries outsized gamma/vega, so
on a sharp down move it gains fast enough that the *T+0* (same-day
mark-to-market) curve stays at-or-above breakeven across the tested range at
entry — the expiry payoff still has a valley (max loss W - N between Kh and
Kl), which is why exits are managed intraday and the hedge is monetized
("temp") when it spikes, per run_levitation_realfill_backtest.py.

A strictly-positive T+0 curve is impossible for any net-credit structure
(worth $0 at entry and >=$0 everywhere would be an arbitrage), so the gate
bounds the dips instead: the entry dip must be small (``t0_entry_dip_usd``)
and by ``t0_check_hour`` every modeled dip must stay inside the stop budget
— meaning the stop realistically caps risk before the expiry valley can.
The T+0 model reprices each leg with Black-Scholes at shifted spots, with a
sticky-strike IV bump of ``vol_beta`` vol-pts per -1% spot move (0DTE skew
response assumption — the weakest link in the model, so it is reported in
the output rather than hidden).

Management (from the real-fill backtest defaults):
  - take profit at +50% of net credit
  - stop at -200% of net credit
  - sell the hedge alone when it reaches 2x its cost (keep the spread)
  - hard close by 15:45 ET; never hold past the cash close
"""
import json
import logging
import math
import os
from datetime import date, datetime
from typing import Optional
from zoneinfo import ZoneInfo

from scipy.stats import norm

logger = logging.getLogger(__name__)
ET  = ZoneInfo("America/New_York")
_RF = 0.045

_CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "config", "levitation.json")

_DEFAULTS = {
    "width":             3.0,    # bull-put spread width (Ks - Kl)
    "hedge_budget_frac": 0.50,   # total hedge cost must be <= this x spread credit
    "min_net_credit":    0.10,   # $/share after buying the hedge
    "vol_beta":          1.5,    # vol-pts of IV rise per -1% spot move (T+0 model)
    # A strictly-positive T+0 curve is impossible for ANY net-credit
    # structure under consistent pricing (worth $0 now and >=$0 everywhere
    # would be an arbitrage). The honest levitation gate instead bounds the
    # dips: a small entry dip, and a checkpoint dip that stays inside the
    # stop budget so the stop caps real risk before the expiry valley can.
    "t0_entry_dip_usd":  50,     # entry T+0 dip may not exceed this ($/lot)
    "t0_check_hour":     13.0,   # ET hour for the checkpoint curve
    "t0_chk_buffer_usd": 10,     # slack allowed beyond the stop at checkpoint
    "t0_range_pct":      0.025,  # test T+0 across +/- this spot range
    "tp_frac":           0.50,   # take profit at this x net credit
    "stop_frac":         2.00,   # stop at this x net credit loss
    "hedge_lock":        2.0,    # sell hedge alone at this x its cost
    "max_loss_usd":      300,    # skip if expiry-valley max loss exceeds this
    "entry_start":       "09:45",
    "entry_end":         "11:30",
}


def _cfg() -> dict:
    cfg = dict(_DEFAULTS)
    try:
        with open(_CFG_PATH) as f:
            cfg.update(json.load(f))
    except Exception:
        pass
    return cfg


# ── pricing ───────────────────────────────────────────────────────────────────

def _bs_put(S: float, K: float, T: float, iv: float) -> float:
    if T <= 1e-8 or iv <= 1e-6:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (_RF + 0.5 * iv ** 2) * T) / (iv * math.sqrt(T))
    d2 = d1 - iv * math.sqrt(T)
    return K * math.exp(-_RF * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _hours_to_close(now: Optional[datetime] = None) -> float:
    now = now or datetime.now(ET)
    return max(16.0 - (now.hour + now.minute / 60.0), 0.25)


def _iv_from_mid(S: float, K: float, T: float, mid: float) -> float:
    """Back out the put IV that reprices to mid under OUR clock (T in years),
    so the T+0 curve is exactly $0 at entry. Returns 0.0 on failure."""
    from scipy.optimize import brentq
    if T <= 1e-8 or mid <= max(K - S, 0.0):
        return 0.0
    try:
        return brentq(lambda v: _bs_put(S, K, T, v) - mid, 1e-4, 10.0,
                      xtol=1e-4, maxiter=60)
    except Exception:
        return 0.0


def t0_curve(legs: list[dict], spot: float, net_credit: float,
             vol_beta: float, range_pct: float,
             hours_left: Optional[float] = None) -> list[tuple[float, float]]:
    """
    Model the same-day mark-to-market P&L ($/lot) of the 3-leg position across
    a spot range, at the current time-to-close.

    legs: [{"K": strike, "iv": entry iv, "qty": +1 long / -1 short}, ...]
    Returns [(pct_move, pnl_usd), ...] for pct in +/- range_pct.
    """
    T = (hours_left if hours_left is not None else _hours_to_close()) / (24 * 365)
    out = []
    steps = 25
    for i in range(-steps, steps + 1):
        pct = range_pct * i / steps
        S = spot * (1 + pct)
        pnl = net_credit
        for leg in legs:
            # sticky-strike + skew response: IV rises as spot falls
            iv = max(leg["iv"] - vol_beta * (pct * 100) / 100, 0.02)
            pnl += leg["qty"] * _bs_put(S, leg["K"], T, iv)
        out.append((round(pct * 100, 2), round(pnl * 100, 2)))
    return out


# ── structure selection ───────────────────────────────────────────────────────

def _pick_structure(spot: float, puts: list[dict], cfg: dict) -> Optional[dict]:
    """
    Pick Ks (ATM short) and Kl (long, W below), then choose the hedge
    (strike x qty 1-3) that MAXIMIZES the modeled T+0 minimum within the
    hedge budget — a nearly-free far put adds no convexity, so "highest
    affordable strike" is the wrong objective.
    """
    if not puts:
        return None
    by_strike = {p["strike"]: p for p in puts}
    strikes = sorted(by_strike)

    ks = min(strikes, key=lambda s: abs(s - spot))
    kl_target = ks - cfg["width"]
    kl = min(strikes, key=lambda s: abs(s - kl_target))
    if kl >= ks or abs(kl - kl_target) > 1.0:
        return None

    s_leg, l_leg = by_strike[ks], by_strike[kl]
    credit = round((s_leg["bid"] + s_leg["ask"]) / 2
                   - (l_leg["bid"] + l_leg["ask"]) / 2, 3)
    if credit <= 0.05:
        return None

    # Calibrate spread-leg IVs once (T+0 anchor at $0 for the spread alone)
    T_now = _hours_to_close() / (24 * 365)
    iv_s = _iv_from_mid(spot, ks, T_now, (s_leg["bid"] + s_leg["ask"]) / 2) or s_leg["iv"]
    iv_l = _iv_from_mid(spot, kl, T_now, (l_leg["bid"] + l_leg["ask"]) / 2) or l_leg["iv"]

    budget = credit * cfg["hedge_budget_frac"]
    hours_chk = max(16.0 - cfg["t0_check_hour"], 0.25)
    best = None            # (min_chk, min_now, net, kh, qty, cost, h_leg, legs)
    for kh in sorted((s for s in strikes if s < kl), reverse=True):
        h_leg  = by_strike[kh]
        h_mid  = (h_leg["bid"] + h_leg["ask"]) / 2
        if h_mid > budget:
            continue
        iv_h = _iv_from_mid(spot, kh, T_now, h_mid) or h_leg["iv"]
        for qty in (1, 2, 3):
            cost = h_mid * qty
            if cost > budget:
                break
            net = round(credit - cost, 3)
            if net < cfg["min_net_credit"]:
                break
            legs = [{"K": ks, "iv": iv_s, "qty": -1},
                    {"K": kl, "iv": iv_l, "qty": +1},
                    {"K": kh, "iv": iv_h, "qty": qty}]
            # Objective: checkpoint-curve minimum first (the hard gate),
            # then the entry dip, then net credit.
            min_chk = min(p for _, p in t0_curve(
                legs, spot, net, cfg["vol_beta"], cfg["t0_range_pct"],
                hours_left=hours_chk))
            min_now = min(p for _, p in t0_curve(
                legs, spot, net, cfg["vol_beta"], cfg["t0_range_pct"]))
            if best is None or (min_chk, min_now, net) > (best[0], best[1], best[2]):
                best = (min_chk, min_now, net, kh, qty, round(cost, 3), h_leg, legs)
    if best is None:
        return None

    min_chk, min_now, net, kh, qty, h_cost, h_leg, legs = best
    return {
        "ks": ks, "kl": kl, "kh": kh, "hedge_qty": qty,
        "credit": credit, "hedge_cost": h_cost, "net_credit": net,
        "width": round(ks - kl, 2),
        "legs": legs,
        "legs_meta": {"short": s_leg, "long": l_leg, "hedge": h_leg},
    }


# ── scan ──────────────────────────────────────────────────────────────────────

def scan_levitation(spot: Optional[float] = None,
                    vix_now: float = 0.0, vix_prev: float = 0.0) -> dict:
    """
    Full daily scan. Returns dict with keys:
      spot, candidate, t0, checks, rating, note
    candidate is None when nothing qualifies (note says why).
    """
    cfg = _cfg()

    if spot is None:
        try:
            import yfinance as yf
            spot = float(yf.Ticker("SPY").fast_info["last_price"])
        except Exception as e:
            return {"spot": 0.0, "candidate": None, "note": f"SPY spot fetch failed: {e}"}

    from src.live.alpaca_options import fetch_chain_combined, occ_symbol
    chain = fetch_chain_combined("SPY", spot, dte_min=0, dte_max=0)
    puts  = [p for p in chain["puts_liquid"] if p["dte"] == 0]
    if not puts:
        return {"spot": spot, "candidate": None,
                "note": "No liquid 0DTE SPY puts (market closed or feed down)"}

    cand = _pick_structure(spot, puts, cfg)
    if cand is None:
        return {"spot": spot, "candidate": None,
                "note": f"{len(puts)} liquid 0DTE puts — no structure met "
                        f"credit/hedge-budget rules"}

    # Legs carry IVs calibrated from mids in _pick_structure, so the entry
    # T+0 curve is anchored at exactly $0. Two curves: at entry, and
    # projected at the checkpoint hour (theta accrued, same IV model).
    curve  = t0_curve(cand["legs"], spot, cand["net_credit"],
                      cfg["vol_beta"], cfg["t0_range_pct"])
    t0_min = min(p for _, p in curve)
    t0_min_at = next(m for m, p in curve if p == t0_min)
    hours_chk = max(16.0 - cfg["t0_check_hour"], 0.25)
    curve_chk = t0_curve(cand["legs"], spot, cand["net_credit"],
                         cfg["vol_beta"], cfg["t0_range_pct"],
                         hours_left=hours_chk)
    chk_min = min(p for _, p in curve_chk)

    # expiry-valley max loss (between Kh and Kl): width - net credit
    max_loss = round((cand["width"] - cand["net_credit"]) * 100, 0)

    now = datetime.now(ET)
    hhmm = now.strftime("%H:%M")
    vix_rising = vix_prev and vix_now > vix_prev * 1.03

    chk_label = f"{int(cfg['t0_check_hour']):02d}:00"
    stop_usd  = cand["net_credit"] * cfg["stop_frac"] * 100
    checks = {
        f"T+0 dip by {chk_label} within stop (${stop_usd:.0f}) across +/-{cfg['t0_range_pct']:.1%}":
            chk_min >= -(stop_usd + cfg["t0_chk_buffer_usd"]),
        f"Entry T+0 dip <= ${cfg['t0_entry_dip_usd']:.0f}":
            t0_min >= -cfg["t0_entry_dip_usd"],
        f"Net credit >= ${cfg['min_net_credit']:.2f}": cand["net_credit"] >= cfg["min_net_credit"],
        "Hedge <= 50% of credit": cand["hedge_cost"] <= cand["credit"] * cfg["hedge_budget_frac"] + 1e-9,
        f"Expiry max loss <= ${cfg['max_loss_usd']:.0f}": max_loss <= cfg["max_loss_usd"],
        "VIX not spiking (+3%)": not vix_rising,
        f"Entry window {cfg['entry_start']}-{cfg['entry_end']} ET":
            cfg["entry_start"] <= hhmm <= cfg["entry_end"],
    }
    score = sum(checks.values())
    if not checks[next(iter(checks))]:          # T+0 floor is a hard gate
        rating = "Skip"
    elif score == len(checks):
        rating = "Take"
    elif score >= len(checks) - 1:
        rating = "Acceptable"
    else:
        rating = "Skip"

    today = date.today()
    cand["symbols"] = {
        "short": occ_symbol("SPY", today, "PUT", cand["ks"]),
        "long":  occ_symbol("SPY", today, "PUT", cand["kl"]),
        "hedge": occ_symbol("SPY", today, "PUT", cand["kh"]),
    }
    tp   = round(cand["net_credit"] * cfg["tp_frac"] * 100)
    stop = round(cand["net_credit"] * cfg["stop_frac"] * 100)
    cand["exit_plan"] = (
        f"TP +${tp} ({cfg['tp_frac']:.0%} of net credit) | "
        f"stop -${stop} | sell hedge alone at {cfg['hedge_lock']:.0f}x its cost | "
        f"hard close 15:45 ET — never hold past the close."
    )

    return {
        "spot":      round(spot, 2),
        "vix_now":   round(vix_now, 2),
        "vix_prev":  round(vix_prev, 2),
        "candidate": cand,
        "t0":        {"min_usd": t0_min, "min_at_pct": t0_min_at,
                      "chk_min_usd": chk_min, "chk_hour": cfg["t0_check_hour"],
                      "vol_beta": cfg["vol_beta"],
                      "curve": curve, "curve_chk": curve_chk},
        "checks":    checks,
        "score":     f"{score}/{len(checks)}",
        "rating":    rating,
        "note":      (f"SPY levitation {cand['ks']:.0f}/{cand['kl']:.0f} put spread "
                      f"+ {cand['hedge_qty']}x{cand['kh']:.0f} hedge | net ${cand['net_credit']:.2f} cr | "
                      f"entry dip ${t0_min:+.0f} @ {t0_min_at:+.1f}% | "
                      f"T+0 min by {chk_label} ${chk_min:+.0f} | "
                      f"max loss ${max_loss:.0f} | {rating}"),
    }


# ── Slack formatter ───────────────────────────────────────────────────────────

def fmt_slack(r: dict) -> str:
    if not r.get("candidate"):
        return f":feather: *SPY Levitation* — no trade: {r.get('note', 'no data')}"

    c, t0 = r["candidate"], r["t0"]
    icon = {"Take": ":large_green_circle:", "Acceptable": ":large_yellow_circle:",
            "Skip": ":white_circle:"}.get(r["rating"], ":white_circle:")
    check_lines = "\n".join(
        f">  {':white_check_mark:' if ok else ':x:'} {name}"
        for name, ok in r["checks"].items())

    return (
        f":feather: *SPY Levitation — 0DTE bull-put + temp hedge*  {icon} *{r['rating']}* ({r['score']})\n"
        f"*SPY* ${r['spot']:.2f}  |  VIX {r['vix_now']:.1f}\n"
        f">  SELL  `{c['symbols']['short']}`  (put ${c['ks']:.0f})\n"
        f">  BUY   `{c['symbols']['long']}`  (put ${c['kl']:.0f})\n"
        f">  BUY x{c.get('hedge_qty', 1)}  `{c['symbols']['hedge']}`  (put ${c['kh']:.0f}, temp hedge ${c['hedge_cost']:.2f})\n"
        f">  Spread credit `${c['credit']:.2f}`  −  hedge  =  *net `${c['net_credit']:.2f}` credit*\n"
        f">  Entry T+0 dip: `${t0['min_usd']:+.0f}` at `{t0['min_at_pct']:+.1f}%`  |  "
        f"T+0 min by {int(t0['chk_hour']):02d}:00: `${t0['chk_min_usd']:+.0f}` "
        f"(vol-beta {t0['vol_beta']:.1f} assumption)\n"
        f"{check_lines}\n"
        f">  :scissors: {c['exit_plan']}"
    )
