"""
Predictive-power test: does time-of-day / day-of-week (the "charm" effect —
delta-decay-driven dealer rehedging) actually predict SPY's next 15-30min
direction historically? No options-chain data needed, just real SPY 5-min
bars + the calendar.

Hypotheses tested (from docs/trading_strategy.md's charm claims):
  H1) Friday mornings (9:30-11:00 ET) skew UP more than chance
      ("put-heavy Fridays: OTM puts decay, dealers buy back hedges, AM buying")
  H2) The last 45 min before close (15:00-15:45 ET) behaves differently from
      the rest of the day (pinning / charm-driven flows into 0DTE expiry)
  H3) General scan: for every 5-min time-of-day bucket, is the hit rate of
      "next N bars return > 0" significantly different from 50%?

For each bucket: n, mean forward return (bps), hit rate, and a binomial
z-test p-value against 50% (proxy for "is this just noise").

Usage: python run_charm_predictor_test.py [--days 500] [--horizon 6]
"""
import argparse, sys, os
from datetime import datetime, timedelta, date, time as dt_time
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(__file__))
import requests
import numpy as np
import pandas as pd
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY

ET, UTC = ZoneInfo("America/New_York"), ZoneInfo("UTC")
STK = "https://data.alpaca.markets/v2/stocks"
HEAD = {"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY}


def _u(d, t):
    return datetime.combine(d, t, ET).astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_bars(start, end):
    out, tok = [], None
    while True:
        p = {"timeframe": "5Min", "start": _u(start, dt_time(9, 30)),
             "end": _u(end, dt_time(16, 0)), "limit": 10000, "feed": "sip"}
        if tok: p["page_token"] = tok
        r = requests.get(f"{STK}/SPY/bars", headers=HEAD, params=p, timeout=30)
        d = r.json() if r.status_code == 200 else {}
        for b in (d.get("bars") or []):
            dt_ = datetime.fromisoformat(b["t"].replace("Z", "+00:00")).astimezone(ET)
            out.append((dt_.date(), dt_.strftime("%H:%M"), dt_.weekday(), b.get("vw", b["c"])))
        tok = d.get("next_page_token")
        if not tok: break
    df = pd.DataFrame(out, columns=["date", "hhmm", "weekday", "px"])
    df = df.drop_duplicates(["date", "hhmm"]).sort_values(["date", "hhmm"])
    return df[(df["hhmm"] >= "09:30") & (df["hhmm"] <= "16:00")]   # RTH only — drop pre/post market


def welch_p(sample, baseline):
    """Two-sided Welch's t-test p-value: does `sample`'s mean differ from `baseline`'s mean?"""
    from scipy.stats import ttest_ind
    if len(sample) < 2 or len(baseline) < 2:
        return float("nan")
    return ttest_ind(sample, baseline, equal_var=False).pvalue


def forward_returns(df, horizon):
    """Non-overlapping forward returns: step by `horizon` bars within each day, so
    consecutive samples don't share bars (avoids inflating significance via autocorrelation)."""
    rows = []
    for d, g in df.groupby("date"):
        g = g.reset_index(drop=True)
        px = g["px"].values
        for i in range(0, len(g) - horizon, horizon):
            fwd = (px[i + horizon] / px[i]) - 1.0
            rows.append((d, g["hhmm"].iloc[i], g["weekday"].iloc[i], fwd))
    return pd.DataFrame(rows, columns=["date", "hhmm", "weekday", "fwd_ret"])


def summarize(sub, label, baseline_rets):
    n = len(sub)
    if n == 0:
        print(f"  {label:45}  n=0"); return
    hits = int((sub["fwd_ret"] > 0).sum())
    mean_bps = sub["fwd_ret"].mean() * 10000
    hit_rate = hits / n
    # test vs the OVERALL RTH mean return (accounts for SPY's drift over the window),
    # not a flat 0bp/50% null — that flat null is what falsely flagged everything before.
    p = welch_p(sub["fwd_ret"].values, baseline_rets)
    flag = "  <-- p<0.05" if p < 0.05 else ""
    print(f"  {label:45}  n={n:5d}  hit={hit_rate*100:5.1f}%  mean={mean_bps:+6.2f}bp  p_vs_overall={p:.3f}{flag}")


def run(days, horizon):
    end = datetime.now(ET).date() - timedelta(days=1)
    start = end - timedelta(days=days)
    print(f"Fetching SPY 5-min bars {start} -> {end} ...")
    df = fetch_bars(start, end)
    print(f"{len(df)} bars across {df['date'].nunique()} days\n")

    fwd = forward_returns(df, horizon)
    horizon_min = horizon * 5
    overall = fwd["fwd_ret"].values     # baseline: the full RTH distribution, captures SPY's drift

    print(f"H1: Friday AM (9:30-11:00) vs overall RTH mean  [{horizon_min}min forward return, "
          f"non-overlapping samples]")
    fri_am = fwd[(fwd["weekday"] == 4) & (fwd["hhmm"] >= "09:30") & (fwd["hhmm"] <= "11:00")]
    summarize(fri_am, "Friday 9:30-11:00", overall)
    print()

    print(f"H2: Last 45min before close (15:00-15:45) vs overall RTH mean  [{horizon_min}min forward return]")
    late = fwd[(fwd["hhmm"] >= "15:00") & (fwd["hhmm"] <= "15:45")]
    summarize(late, "15:00-15:45", overall)
    print()

    print(f"H3: By weekday vs overall RTH mean  [{horizon_min}min forward return]")
    names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    for wd in range(5):
        summarize(fwd[fwd["weekday"] == wd], names[wd], overall)
    print()

    print(f"H3b: By hour-of-day bucket vs overall RTH mean  [{horizon_min}min forward return]")
    fwd["hour"] = fwd["hhmm"].str[:2]
    for h, g in fwd.groupby("hour"):
        summarize(g, f"{h}:00", overall)
    print()

    sig = []
    tested = [("Friday AM", fri_am), ("Last 45min", late)] + \
             [(names[wd], fwd[fwd["weekday"] == wd]) for wd in range(5)] + \
             [(f"{h}:00", g) for h, g in fwd.groupby("hour")]
    for label, sub in tested:
        n = len(sub)
        if n == 0: continue
        p = welch_p(sub["fwd_ret"].values, overall)
        if p < 0.05:
            sig.append((label, n, sub["fwd_ret"].mean() * 10000, p))
    n_tests = len(tested)
    print("=" * 70)
    if sig:
        print(f"BUCKETS WITH p<0.05 vs overall (n={len(sig)}, out of {n_tests} tested — "
              f"expect ~{0.05*n_tests:.1f} false positives by chance alone at this threshold):")
        for label, n, mean_bp, p in sig:
            print(f"  {label:20} n={n:5d}  mean={mean_bp:+.2f}bp  p={p:.4f}")
    else:
        print(f"NO buckets significant at p<0.05 vs overall (out of {n_tests} tested) — "
              "no detectable charm/time-of-day edge in this window.")
    print("=" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=500)
    ap.add_argument("--horizon", type=int, default=6, help="bars ahead for forward return (6 = 30min)")
    a = ap.parse_args()
    run(a.days, a.horizon)
