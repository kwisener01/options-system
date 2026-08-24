"""
Quick honest look: does a Greek move BEFORE spot (leading / predictive), WITH
spot (contemporaneous / just a shared driver), or AFTER spot (reacting to
price, not predicting it)? Checks all three logged Greeks -- net_vanna_bn,
net_gex_bn, net_charm -- since they're not equally likely to lead: gamma and
vanna are both heavily driven by spot itself in the Black-Scholes formulas
(gamma ~ spot^2, vanna ~ spot), so they're expected to be close to
CONTEMPORANEOUS by construction -- a near-1.0 same-tick correlation there is
NOT evidence of predictive power, it's the math. Charm (dDelta/dTime) is
driven primarily by the clock, not spot, so it's the one with a real
theoretical reason to actually lead instead of just track.

Pulls the live /api/level_snapshots feed and, per Greek, computes over
consecutive 5-min ticks:
  - contemporaneous correlation: corr(d_greek[t], d_spot[t])
  - GREEK LEADS:  corr(d_greek[t], d_spot[t+1])   -- does a Greek move predict
                  the NEXT tick's price move?
  - GREEK LAGS:   corr(d_spot[t], d_greek[t+1])   -- or is the Greek just
                  following price that already moved?

This is exploratory on however many days are in the feed so far (each day's
~75 ticks are highly autocorrelated, not independent samples) -- NOT a
statistically powered test until many days have accumulated. Treat it as
"does this look promising enough to keep watching," not a verdict.

Usage: python run_vanna_leadlag_check.py [--date YYYY-MM-DD]
"""
import argparse
import requests
import numpy as np

URL = "https://options-system-2u59.onrender.com/api/level_snapshots"

GREEKS = {
    "vanna": ("net_vanna_bn", "spot-driven (vanna ~ spot) -- expect ~coincident, not leading"),
    "gex":   ("net_gex_bn",   "spot-driven (gamma ~ spot^2) -- expect ~coincident, not leading"),
    "charm": ("net_charm",    "TIME-driven, not spot-driven -- the one actually worth watching for leading"),
    "vega":  ("net_vega_bn",  "IV-driven, call-vs-put imbalance -- no forced hedging flow like the "
                              "others, so a real lead here would be a genuine surprise, not expected"),
    "theta": ("net_theta_bn", "TIME-driven like charm, but decay is P&L, not a delta/hedge-ratio "
                              "change -- no direct forced-flow mechanism, so also not expected to lead"),
}


def corr(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return np.corrcoef(a, b)[0, 1]


def run(date_filter=None):
    r = requests.get(URL, timeout=20)
    rows = r.json().get("rows", [])
    if date_filter:
        rows = [row for row in rows if row["ts"].startswith(date_filter)]
    rows = sorted(rows, key=lambda row: row["ts"])
    if len(rows) < 5:
        print(f"Only {len(rows)} rows -- not enough to look at."); return

    days = sorted(set(row["ts"][:10] for row in rows))
    spot = np.array([float(row["spot"]) for row in rows])
    d_spot = np.diff(spot)
    n = len(d_spot)
    print(f"{len(rows)} ticks across {len(days)} day(s) ({days[0]} to {days[-1]})\n")

    for label, (col, theory) in GREEKS.items():
        missing = [row for row in rows if col not in row or row[col] in ("", None)]
        if len(missing) == len(rows):
            print(f"{label:6} ({col}): not logged yet in this data -- skip\n")
            continue
        vals = np.array([float(row.get(col) or 0.0) for row in rows])
        d_val = np.diff(vals)
        contemp = corr(d_val, d_spot)
        leads   = corr(d_val[:-1], d_spot[1:])
        lags    = corr(d_spot[:-1], d_val[1:])
        print(f"{label:6} ({col}) -- {theory}")
        print(f"  contemporaneous corr(d_{label}[t], d_spot[t])   = {contemp:+.3f}  (n={n})")
        print(f"  {label.upper()} LEADS  corr(d_{label}[t], d_spot[t+1]) = {leads:+.3f}  (n={n-1})")
        print(f"  {label.upper()} LAGS   corr(d_spot[t], d_{label}[t+1]) = {lags:+.3f}  (n={n-1})")
        print()

    print(f"{'='*70}")
    print(f"{len(days)} day(s), {len(rows)} ticks total -- ticks within a day are highly")
    print("autocorrelated, not independent. Needs many days before this is a real test.")
    print("Watch CHARM's lead number specifically as days accumulate -- vanna/gex leading")
    print("would be a bigger surprise than lagging, given how they're computed.")
    print(f"{'='*70}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=str, default=None)
    a = ap.parse_args()
    run(a.date)
