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

Data comes from Alpaca historical bars (stock + option) — yfinance is not used,
it is unreliable behind corporate SSL. Requires ALPACA_API_KEY / ALPACA_SECRET_KEY
(loaded from .env if present).

Usage:
    python run_suggestion_backtest.py [slack_export.txt]
"""
from __future__ import annotations

import os
import re
import sys
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

DEFAULT_EXPORT = "slack_test/ck test — trader connected.txt"


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


# ── main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    try:                                    # Windows console is cp1252 by default
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EXPORT
    validate = "--no-validate" not in sys.argv
    if not os.path.exists(path):
        sys.exit(f"export not found: {path}")
    _load_env()
    lines = open(path, encoding="utf-8").read().splitlines()

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
          f"({start.date()} → {end.date()})...")
    closes = fetch_stock_closes(tickers, start, end)

    run_equity(equity, closes)
    run_options(options, closes, validate)

    print("Notes: overlapping re-alerts are not independent; forward windows are")
    print("truncated for recent picks; options are held-to-expiry (no early mgmt).")


if __name__ == "__main__":
    main()
