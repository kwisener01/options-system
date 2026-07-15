#!/usr/bin/env python3
"""Backtest the ranked trade suggestions captured in a Slack export.

Reads a Slack channel export (the bot's own alerts) and evaluates two families
of ranked suggestion:

  Equity ranks  — Value Watchlist ACCUMULATE + Fallen Angel picks, each with a
                  conviction score. Buy at the alert-day close, measure forward
                  return AND SPY-relative alpha, bucketed by score, to test
                  whether the rank has predictive value.  [#1]

  Option ranks  — BWB Scanner (broken-wing put butterflies) and Iron Butterfly
                  suggestions. Defined-risk structures are held to expiration and
                  settled at intrinsic value from the underlying's close on the
                  expiry date; entry is the net credit stated in the alert. Alpaca
                  option bars are pulled on the entry date to confirm the credit
                  was achievable.  [#2]

  --as-vertical — re-express the equity signals as MANAGED option verticals
                  (bull-put credit / bull-call debit) instead of buying stock.
                  Take profit at a target, else force-close a few days before
                  expiry — never held to expiration, so no assignment risk.
                  Entry and every mark come from real Alpaca option daily bars.

Data comes from Alpaca historical bars (stock + option) — yfinance is not used,
it is unreliable behind corporate SSL. Requires ALPACA_API_KEY / ALPACA_SECRET_KEY
(loaded from .env if present).

Usage:
    python run_suggestion_backtest.py [export.txt]
    python run_suggestion_backtest.py --as-vertical [--source FA|VW|both]
        [--vertical put|call|both] [--target 0.5] [--close-dte 2]
"""
from __future__ import annotations

import os
import re
import sys
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

DEFAULT_EXPORT = "slack_test/ck test — trader connected.txt"


def _flag_value(flag: str, default: str) -> str:
    """Read `--flag value` or `--flag=value` from argv, else default."""
    for i, a in enumerate(sys.argv):
        if a == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return default


# ── credentials ────────────────────────────────────────────────────────────────

def _load_env() -> None:
    if os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"):
        return
    if os.path.exists(".env"):
        for ln in open(".env", encoding="utf-8"):
            ln = ln.strip()
            if ln and not ln.startswith("#") and "=" in ln:
                k, v = ln.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    if not (os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY")):
        sys.exit("ERROR: ALPACA_API_KEY / ALPACA_SECRET_KEY not set (and not in .env)")


# ── parsing the Slack export ─────────────────────────────────────────────────────

_VW_HDR = re.compile(r"Value Watchlist — (\d{4}-\d{2}-\d{2})")
_FA_HDR = re.compile(r"Fallen Angel Scanner — (\d{4}-\d{2}-\d{2})")
_VW_PICK = re.compile(r"^\s*([A-Z]{1,5})\s+\$([\d.]+)\s+\(Own[^)]*\)\s+.*ACCUMULATE\s+(\d+)/(\d+)")
_FA_PICK = re.compile(r"^\s*([A-Z]{1,5})\s+\$([\d.]+)\s+◎\s+(\d+)/(\d+)\s*\|")


def parse_equity(lines: list[str]) -> list[dict]:
    """Ranked equity suggestions: (date, source, ticker, price, score fraction)."""
    out, src = [], None
    for ln in lines:
        if (m := _VW_HDR.search(ln)):
            src = ("VW", m.group(1)); continue
        if (m := _FA_HDR.search(ln)):
            src = ("FA", m.group(1)); continue
        if src is None:
            continue
        pat = _VW_PICK if src[0] == "VW" else _FA_PICK
        if (m := pat.match(ln)):
            out.append(dict(date=src[1], source=src[0], ticker=m.group(1),
                            price=float(m.group(2)),
                            score=int(m.group(3)), score_max=int(m.group(4)),
                            rank=int(m.group(3)) / int(m.group(4))))
    # de-dup identical (date, source, ticker)
    seen, uniq = set(), []
    for r in out:
        k = (r["date"], r["source"], r["ticker"])
        if k not in seen:
            seen.add(k); uniq.append(r)
    return uniq


_BWB_DATE = re.compile(r"BWB Scanner — (\d{4}-\d{2}-\d{2})")
_BWB_TICK = re.compile(r"^\s*([A-Z]{1,5})\s+\$([\d.]+)\s+\([+-]")
_BWB_LEGS = re.compile(r"BUY \$([\d.]+)P / SELL 2× \$([\d.]+)P / BUY \$([\d.]+)P\s+·\s+(\d+)DTE")
_BWB_CRED = re.compile(r"Credit \+?\$(-?[\d.]+)\s+·\s+Max risk \$([\d.]+)")

_IB_DATE = re.compile(r"Bull Put Screener — (\d{4}-\d{2}-\d{2})")
_IB_HDR = re.compile(r"ENTER ([A-Z]+) @ \$([\d.]+) — IRON BUTTERFLY")
_IB_LEGS = re.compile(r"Sell \$([\d.]+) straddle / Buy \$([\d.]+)P \+ \$([\d.]+)C\s+exp (\d{4}-\d{2}-\d{2})")
_IB_NET = re.compile(r"Net \$([\d.]+)\s*\|\s*Risk \$([\d.]+)")


def parse_options(lines: list[str]) -> list[dict]:
    """Ranked option-structure suggestions (BWB + iron butterfly)."""
    out = []
    date = tick = px = None
    for i, ln in enumerate(lines):
        if (m := _BWB_DATE.search(ln)):
            date = m.group(1)
        elif (m := _IB_DATE.search(ln)):
            date = m.group(1)
        if (m := _BWB_TICK.match(ln)):
            tick, px = m.group(1), float(m.group(2))
        if (m := _BWB_LEGS.search(ln)) and tick and date:
            khi, kmid, klo, dte = float(m.group(1)), float(m.group(2)), float(m.group(3)), int(m.group(4))
            cred = _BWB_CRED.search(lines[i + 1]) if i + 1 < len(lines) else None
            if cred:
                out.append(dict(kind="bwb", date=date, ticker=tick, spot=px,
                                strikes=dict(khi=khi, kmid=kmid, klo=klo), dte=dte,
                                credit=float(cred.group(1)) * 100,      # per-share → dollars
                                max_risk=float(cred.group(2)), expiry=None))
        if (m := _IB_HDR.search(ln)):
            tick, px = m.group(1), float(m.group(2))
            legs = _IB_LEGS.search(lines[i + 1]) if i + 1 < len(lines) else None
            net = _IB_NET.search(lines[i + 2]) if i + 2 < len(lines) else None
            if legs and net:
                out.append(dict(kind="iron_fly", date=date, ticker=tick, spot=px,
                                strikes=dict(k=float(legs.group(1)),
                                             kp=float(legs.group(2)), kc=float(legs.group(3))),
                                expiry=legs.group(4),
                                credit=float(net.group(1)),             # already dollars
                                max_risk=float(net.group(2)), dte=None))
    # de-dup structurally identical suggestions, keep earliest entry
    seen, uniq = set(), []
    for r in sorted(out, key=lambda x: x["date"]):
        k = (r["kind"], r["ticker"], tuple(sorted(r["strikes"].items())), r["expiry"], r["dte"])
        if k not in seen:
            seen.add(k); uniq.append(r)
    return uniq


# ── Alpaca data ─────────────────────────────────────────────────────────────────

def fetch_stock_closes(tickers: list[str], start: datetime, end: datetime):
    import pandas as pd
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    cli = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    df = cli.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=tickers, timeframe=TimeFrame.Day, start=start, end=end)).df
    out = {}
    have = set(df.index.get_level_values(0))
    for t in tickers:
        if t in have:
            s = df.xs(t, level=0)["close"]
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            out[t] = s
    return out


def option_entry_credit(struct: dict) -> float | None:
    """Net credit ($) achievable on the entry date from Alpaca option daily bars,
    or None if bars are missing. Validates the alert's stated credit."""
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.requests import OptionBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from src.live.alpaca_options import occ_symbol
    d = datetime.strptime(struct["date"], "%Y-%m-%d")
    exp = _expiry_date(struct)
    if struct["kind"] == "bwb":
        legs = [("PUT", struct["strikes"]["khi"], +1), ("PUT", struct["strikes"]["kmid"], -2),
                ("PUT", struct["strikes"]["klo"], +1)]
    else:
        s = struct["strikes"]
        legs = [("CALL", s["k"], -1), ("PUT", s["k"], -1),
                ("PUT", s["kp"], +1), ("CALL", s["kc"], +1)]
    cli = OptionHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    credit = 0.0
    for typ, strike, qty in legs:
        sym = occ_symbol(struct["ticker"], exp.date(), typ, strike)
        try:
            df = cli.get_option_bars(OptionBarsRequest(
                symbol_or_symbols=sym, timeframe=TimeFrame.Day,
                start=d, end=d + timedelta(days=3))).df
        except Exception:
            return None
        if df is None or len(df) == 0:
            return None
        price = float(df["close"].iloc[0])
        credit += -qty * price * 100        # sell(+qty=-1)→receive, buy→pay
    return round(credit, 2)


# ── settlement P&L (hold to expiry) ──────────────────────────────────────────────

def _expiry_date(struct: dict) -> datetime:
    if struct["expiry"]:
        return datetime.strptime(struct["expiry"], "%Y-%m-%d")
    return datetime.strptime(struct["date"], "%Y-%m-%d") + timedelta(days=struct["dte"])


def _price_at(series, target: datetime, tol_days: int = 4):
    """Close nearest to `target` within tol_days, or None."""
    import pandas as pd
    if series is None or len(series) == 0:
        return None
    diffs = (series.index - pd.Timestamp(target)).to_series().abs()
    j = diffs.values.argmin()
    if diffs.iloc[j] > pd.Timedelta(days=tol_days):
        return None
    return float(series.iloc[j])


def settle_pnl(struct: dict, closes: dict) -> dict | None:
    """P&L of a defined-risk structure held to expiry, settled on the underlying's
    close at expiration. Returns None if not yet expired / no price."""
    import pandas as pd
    exp = _expiry_date(struct)
    series = closes.get(struct["ticker"])
    if series is None:
        return None
    if pd.Timestamp(exp) > series.index.max() + pd.Timedelta(days=1):
        return {"status": "open (not yet expired)"}
    S = _price_at(series, exp)
    if S is None:
        return None
    k = struct["strikes"]
    if struct["kind"] == "bwb":
        intrinsic = (max(k["khi"] - S, 0) - 2 * max(k["kmid"] - S, 0) + max(k["klo"] - S, 0))
    else:
        intrinsic = (max(k["kp"] - S, 0) + max(S - k["kc"], 0)
                     - max(k["k"] - S, 0) - max(S - k["k"], 0))
    pnl = struct["credit"] + intrinsic * 100
    pnl = max(pnl, -struct["max_risk"])           # cap at defined max loss
    return {"status": "expired", "settle_price": round(S, 2),
            "pnl": round(pnl, 2), "ror": pnl / struct["max_risk"]}


# ── reporting ────────────────────────────────────────────────────────────────────

def _stat(vals: list[float]) -> str:
    if not vals:
        return "n=0"
    import statistics as st
    wins = sum(1 for v in vals if v > 0)
    return (f"n={len(vals):3}  mean={sum(vals)/len(vals):+6.2f}  "
            f"median={st.median(vals):+6.2f}  win={wins/len(vals)*100:3.0f}%")


def run_equity(picks: list[dict], closes: dict) -> None:
    import pandas as pd, numpy as np
    spy = closes.get("SPY")

    def fwd(t, date, h):
        s = closes.get(t)
        if s is None:
            return None
        s = s[s.index >= pd.Timestamp(date)]
        if h >= len(s):
            return None
        return (s.iloc[h] / s.iloc[0] - 1) * 100

    for p in picks:
        for h in (10, 21):
            r = fwd(p["ticker"], p["date"], h)
            b = fwd("SPY", p["date"], h)
            p[f"r{h}"] = r
            p[f"a{h}"] = (r - b) if (r is not None and b is not None) else None

    print("\n" + "=" * 74)
    print(f"  #1  EQUITY RANKED SUGGESTIONS  —  {len(picks)} picks "
          f"({sum(p['source']=='FA' for p in picks)} Fallen Angel, "
          f"{sum(p['source']=='VW' for p in picks)} Value Watchlist)")
    print("=" * 74)
    print("  return = buy at alert-day close, hold N trading days")
    print("  alpha  = pick return − SPY return over the same window\n")

    for src, label in [("FA", "Fallen Angel (score /13)"), ("VW", "Value Watchlist (score /9)")]:
        sub = [p for p in picks if p["source"] == src]
        if not sub:
            continue
        print(f"  {label}  ({len(sub)} picks)")
        for h in (10, 21):
            rets = [p[f"r{h}"] for p in sub if p[f"r{h}"] is not None]
            alfa = [p[f"a{h}"] for p in sub if p[f"a{h}"] is not None]
            print(f"    {h:2}d return  {_stat(rets)}")
            print(f"    {h:2}d ALPHA   {_stat(alfa)}")
            rr = [(p['rank'], p[f'a{h}']) for p in sub if p[f'a{h}'] is not None]
            if len(rr) > 2:
                c = np.corrcoef([x[0] for x in rr], [x[1] for x in rr])[0, 1]
                print(f"        corr(score, {h}d alpha) = {c:+.3f}"
                      f"   {'← higher score = more alpha' if c > 0.1 else '← rank not predictive' if abs(c)<=0.1 else '← INVERTED: higher score = less alpha'}")
        print()


def run_options(structs: list[dict], closes: dict, validate: bool) -> None:
    print("=" * 74)
    print(f"  #2  OPTION RANKED SUGGESTIONS  —  {len(structs)} structures")
    print("=" * 74)
    print("  method: hold to expiry, settle at intrinsic on the underlying's")
    print("          expiry-day close; entry = net credit from the alert.\n")
    rows, pnls, rors = [], [], []
    for s in structs:
        res = settle_pnl(s, closes)
        tag = f"{s['ticker']:5} {s['kind']:9} {s['date']}"
        strikes = "/".join(f"{v:g}" for v in s["strikes"].values())
        if res is None:
            print(f"  {tag}  [{strikes}]  — no underlying data"); continue
        if res["status"] != "expired":
            print(f"  {tag}  [{strikes}]  — {res['status']}"); continue
        val = ""
        if validate:
            c = option_entry_credit(s)
            if c is not None:
                val = f"  | mkt credit ${c:+.0f} (alert ${s['credit']:+.0f})"
        print(f"  {tag}  [{strikes}]  exp@${res['settle_price']}  "
              f"P&L ${res['pnl']:+7.2f}  (ROR {res['ror']*100:+5.0f}% of ${s['max_risk']:.0f} risk){val}")
        pnls.append(res["pnl"]); rors.append(res["ror"] * 100)
    if pnls:
        wins = sum(1 for p in pnls if p > 0)
        print(f"\n  Settled: {len(pnls)}   total P&L ${sum(pnls):+.2f}   "
              f"win {wins}/{len(pnls)} ({wins/len(pnls)*100:.0f}%)   "
              f"mean ROR {sum(rors)/len(rors):+.0f}% of risk")
    print()


# ── --as-vertical: re-express an equity signal as a managed vertical ──────────────
#
# Instead of buying the stock, sell a bull-put credit spread (or buy a bull-call
# debit spread) below/around the alert price and MANAGE it — take profit at a
# target, else force-close a few days before expiry. Nothing is ever held through
# expiration, so there is no assignment risk. Entry and every mark come from real
# Alpaca option daily bars.

from calendar import monthcalendar

_OPT = None


def _opt_client():
    global _OPT
    if _OPT is None:
        from alpaca.data.historical.option import OptionHistoricalDataClient
        _OPT = OptionHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    return _OPT


def _third_friday(y: int, m: int) -> datetime:
    fr = [w[4] for w in monthcalendar(y, m) if w[4]]
    return datetime(y, m, fr[2])


def _monthly_after(dt: datetime, min_dte: int = 25) -> datetime:
    e = _third_friday(dt.year, dt.month)
    if e < dt + timedelta(days=min_dte):
        ny, nm = (dt.year, dt.month + 1) if dt.month < 12 else (dt.year + 1, 1)
        e = _third_friday(ny, nm)
    return e


def _strike_grid(spot: float, lo: float, hi: float) -> list[float]:
    inc = 0.5 if spot < 20 else 1.0 if spot < 60 else 2.5 if spot < 150 else 5.0
    import math
    out, k = [], math.floor(spot * lo / inc) * inc
    while k <= spot * hi:
        out.append(round(k, 1)); k += inc
    return out


def _spread_bars(tkr: str, exp: datetime, typ: str, strikes: list[float],
                 start: datetime, end: datetime) -> dict:
    """{strike: close-series (naive daily index)} for candidate strikes that traded."""
    import pandas as pd
    from alpaca.data.requests import OptionBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from src.live.alpaca_options import occ_symbol
    syms = {occ_symbol(tkr, exp.date(), typ, k): k for k in strikes}
    try:
        df = _opt_client().get_option_bars(OptionBarsRequest(
            symbol_or_symbols=list(syms), timeframe=TimeFrame.Day, start=start, end=end)).df
    except Exception:
        return {}
    if df is None or len(df) == 0:
        return {}
    out = {}
    for sym in df.index.get_level_values(0).unique():
        s = df.xs(sym, level=0)["close"]
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        out[syms[sym]] = s
    return out


def _nearest(cands, target):
    return min(cands, key=lambda k: abs(k - target)) if cands else None


def manage_vertical(kind: str, pick: dict, target: float, close_dte: int) -> dict | None:
    """Backtest one managed vertical from an equity alert. `kind` is 'put'
    (bull-put credit) or 'call' (bull-call debit). Returns realized-P&L dict or
    None if the spread can't be built from traded option bars."""
    import pandas as pd
    dt = datetime.strptime(pick["date"], "%Y-%m-%d")
    spot = pick["price"]
    exp = _monthly_after(dt)
    typ = "PUT" if kind == "put" else "CALL"
    lo, hi = (0.80, 1.00) if kind == "put" else (0.98, 1.20)
    fetch_end = min(exp, datetime.now())             # Alpaca returns nothing for a future end
    bars = _spread_bars(pick["ticker"], exp, typ, _strike_grid(spot, lo, hi), dt, fetch_end)
    entry_px = {k: float(s.iloc[0]) for k, s in bars.items() if len(s)}
    if len(entry_px) < 2:
        return None

    if kind == "put":                                   # short ~5% OTM / long lower
        otm = {k: v for k, v in entry_px.items() if k <= spot}
        Ks = _nearest(list(otm), spot * 0.95)
        Kl = _nearest([k for k in otm if k < (Ks or -1)], spot * 0.88)
        if not Ks or not Kl:
            return None
        legs = [(Ks, -1), (Kl, +1)]                     # sell high strike, buy low
        width = (Ks - Kl) * 100
    else:                                               # long ~ATM / short higher
        Kl = _nearest(list(entry_px), spot * 1.00)
        Ku = _nearest([k for k in entry_px if k > (Kl or 1e9)], spot * 1.08)
        if not Kl or not Ku:
            return None
        legs = [(Kl, +1), (Ku, -1)]                     # buy low strike, sell high
        width = (Ku - Kl) * 100

    # align the two legs' daily marks
    ser = {k: bars[k] for k, _ in legs}
    common = sorted(set(ser[legs[0][0]].index) & set(ser[legs[1][0]].index))
    common = [d for d in common if d >= pd.Timestamp(dt)]
    if len(common) < 2:
        return None

    def net(d):                                          # position value to holder
        return sum(q * float(ser[k].loc[d]) for k, q in legs) * 100

    p_entry = net(common[0])
    credit = -p_entry                                    # >0 for the put credit spread
    debit = p_entry                                      # >0 for the call debit spread
    if kind == "put":
        if credit <= 0:
            return None
        max_profit, max_risk = credit, width - credit
    else:
        if debit <= 0:
            return None
        max_profit, max_risk = width - debit, debit
    if max_risk <= 0:
        return None

    force_close = pd.Timestamp(exp) - pd.Timedelta(days=close_dte)
    exit_d, reason = common[-1], "data-end mark"
    for d in common[1:]:
        pnl = net(d) - p_entry
        if pnl >= target * max_profit:
            exit_d, reason = d, f"+{int(target*100)}% target"; break
        if d >= force_close:
            exit_d, reason = d, f"force-close {close_dte}d pre-exp"; break
    pnl = max(min(net(exit_d) - p_entry, max_profit), -max_risk)
    return dict(kind=kind, ticker=pick["ticker"], date=pick["date"],
                strikes="/".join(f"{k:g}" for k, _ in legs),
                entry=p_entry, pnl=round(pnl, 2), max_risk=round(max_risk, 2),
                ror=pnl / max_risk, days=(exit_d - common[0]).days, reason=reason)


def run_verticals(picks: list[dict], closes: dict, sources: set[str],
                  kinds: list[str], target: float, close_dte: int) -> None:
    sel = [p for p in picks if p["source"] in sources]
    src_label = "+".join(sorted(sources))
    print("=" * 74)
    print(f"  --as-vertical  —  {src_label} equity signals re-expressed as MANAGED verticals")
    print("=" * 74)
    print(f"  exit rule: take profit at +{int(target*100)}% of max profit, else force-close")
    print(f"             {close_dte}d before expiry. Never held to expiration (no assignment).")
    print(f"  entry/marks from Alpaca option daily bars; monthly expiry.\n")

    results = {k: [] for k in kinds}
    for p in sel:
        parts = []
        for k in kinds:
            r = manage_vertical(k, p, target, close_dte)
            if r:
                results[k].append(r)
                parts.append(f"{k}: ${r['pnl']:+5.0f}/{r['max_risk']:>3.0f}r "
                             f"{r['ror']*100:+4.0f}% ({r['reason']})")
        if parts:
            print(f"  {p['ticker']:5} {p['date']}  " + "   ".join(parts))

    print(f"\n=== SUMMARY (capital-aware) — {src_label} ===")
    for k in kinds:
        rs = results[k]
        if not rs:
            print(f"  bull-{k:4} n=0"); continue
        pnls = [r["pnl"] for r in rs]
        rors = [r["ror"] * 100 for r in rs]
        risk = sum(r["max_risk"] for r in rs)
        wins = sum(1 for x in pnls if x > 0)
        label = "credit spread" if k == "put" else "debit spread"
        print(f"  bull-{k} {label:14} n={len(rs):3}  "
              f"mean ${sum(pnls)/len(pnls):+5.0f}/trade  mean ROR {sum(rors)/len(rors):+4.0f}%  "
              f"win {wins/len(rs)*100:3.0f}%")
        print(f"    gross P&L ${sum(pnls):+.0f}  across ${risk:,.0f} of cumulative risk "
              f"(NOT concurrent — these overlap and re-alert)")
    print("\n  ⚠️  Gross P&L is a SUM over overlapping, non-independent re-alerts that")
    print("      would need far more capital than the ~$2k account had. Read the")
    print("      PER-TRADE mean/ROR/win as the edge; the gross total is not account P&L.")
    print()


# ── main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    try:                                    # Windows console is cp1252 by default
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    value_flags = {"--source", "--vertical", "--target", "--close-dte"}
    positional, skip = [], False
    for i, a in enumerate(sys.argv[1:]):
        if skip:
            skip = False; continue
        if a in value_flags:                         # consumes the next arg as its value
            skip = True; continue
        if not a.startswith("--"):
            positional.append(a)
    path = positional[0] if positional else DEFAULT_EXPORT
    validate = "--no-validate" not in sys.argv
    if not os.path.exists(path):
        sys.exit(f"export not found: {path}")
    _load_env()
    lines = open(path, encoding="utf-8").read().splitlines()

    as_vertical = "--as-vertical" in sys.argv
    src_arg = _flag_value("--source", "FA")          # FA | VW | both
    kinds_arg = _flag_value("--vertical", "both")    # put | call | both
    target = float(_flag_value("--target", "0.5"))
    close_dte = int(_flag_value("--close-dte", "2"))

    equity = parse_equity(lines)
    options = parse_options(lines)
    print(f"Parsed {len(equity)} ranked equity picks and {len(options)} option structures "
          f"from {os.path.basename(path)}")

    # one stock-bar pull covers equity forward returns + option settlement + SPY
    tickers = sorted({p["ticker"] for p in equity} | {s["ticker"] for s in options} | {"SPY"})
    dates = [p["date"] for p in equity] + [s["date"] for s in options]
    start = datetime.strptime(min(dates), "%Y-%m-%d") - timedelta(days=5)
    end = datetime.now()
    print(f"Fetching Alpaca daily bars for {len(tickers)} symbols "
          f"({start.date()} → {end.date()})...\n")
    closes = fetch_stock_closes(tickers, start, end)

    if as_vertical:
        sources = {"FA", "VW"} if src_arg == "both" else {src_arg}
        kinds = ["put", "call"] if kinds_arg == "both" else [kinds_arg]
        run_verticals(equity, closes, sources, kinds, target, close_dte)
        return

    run_equity(equity, closes)
    run_options(options, closes, validate)

    print("Notes: overlapping re-alerts are not independent; forward windows are")
    print("truncated for recent picks; options are held-to-expiry (no early mgmt).")
    print("Run with --as-vertical to re-express equity signals as managed spreads.")


if __name__ == "__main__":
    main()
