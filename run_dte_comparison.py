"""
DTE comparison backtest: 1DTE vs 2DTE vs 3DTE vs 7DTE bull put spreads on SPY.

Simulates daily rolling credit spreads using Black-Scholes on real SPY/VIX history.
Answers: which DTE maximises compounding from a $2k account while staying PDT-safe?

Usage:
    python run_dte_comparison.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import math
import pandas as pd
import numpy as np
import yfinance as yf

from src.logger import setup_logging
from src.backtest.wheel_engine import bs_put

setup_logging()

# ── parameters ────────────────────────────────────────────────────────────────
INITIAL_CAPITAL  = 2_000.0
SHORT_OTM_PCT    = 0.015   # short strike 1.5% OTM
SPREAD_WIDTH_PCT = 0.01    # spread width ~1% of spot
MAX_RISK_PCT     = 0.10    # risk 10% of account per trade
IV_PREMIUM       = 1.20    # VIX × 1.2 for IV estimate
MAX_VIX          = 25.0    # skip entry if VIX > 25
TP_PCT           = 0.50    # take profit at 50% of max credit
START            = "2022-01-01"
END              = "2025-12-31"

TICK = 0.5


def rs(p: float) -> float:
    return round(p / TICK) * TICK


# ── simulation ────────────────────────────────────────────────────────────────

def run_backtest(spy_df: pd.DataFrame, vix_df: pd.DataFrame,
                 dte: int) -> tuple[pd.Series, pd.DataFrame]:
    """
    Simulate a daily rolling bull put spread at the given DTE.
    - dte=1  : enter today, expire tomorrow (overnight hold — PDT safe)
    - dte=7  : enter every Friday, expire next Friday
    Entry uses today's close as spot proxy; settlement uses expiry-day close.
    """
    dates   = spy_df.index.sort_values()
    account = INITIAL_CAPITAL
    equity  = {}
    trades  = []
    pos     = None   # {entry, expiry, K1, K2, credit, contracts, sigma, vix, entry_spot}

    for i, d in enumerate(dates):
        row_spy = spy_df.loc[d]
        spot    = float(row_spy["Close"])
        vix_row = vix_df.reindex([d], method="ffill")
        vix     = float(vix_row["Close"].iloc[0]) if len(vix_row) > 0 else 20.0

        # ── 1. Check / close open position ────────────────────────────────────
        if pos is not None:
            expired = d >= pos["expiry"]

            # Take-profit: check if residual value ≤ (1-TP_PCT)*credit
            tp_hit = False
            if not expired and d > pos["entry"]:
                t_rem = max((pos["expiry"] - d).days / 365, 1 / 365)
                cur = (bs_put(spot, pos["K1"], t_rem, pos["sigma"]) -
                       bs_put(spot, pos["K2"], t_rem, pos["sigma"]))
                if cur <= pos["credit"] * (1 - TP_PCT):
                    tp_hit = True

            if expired or tp_hit:
                K1, K2   = pos["K1"], pos["K2"]
                cr       = pos["credit"]
                c        = pos["contracts"]
                pnl_ps   = cr - max(0.0, K1 - spot) + max(0.0, K2 - spot)
                pnl      = pnl_ps * 100 * c
                account += pnl

                if tp_hit:
                    outcome = "TAKE_PROFIT"
                elif spot >= K1:
                    outcome = "FULL_PROFIT"
                elif spot <= K2:
                    outcome = "MAX_LOSS"
                else:
                    outcome = "PARTIAL_LOSS"

                trades.append({
                    "entry_date":   pos["entry"],
                    "close_date":   d,
                    "K1": K1, "K2": K2,
                    "credit":       cr,
                    "contracts":    c,
                    "vix_entry":    pos["vix"],
                    "spot_entry":   pos["entry_spot"],
                    "spot_close":   spot,
                    "pnl":          round(pnl, 2),
                    "outcome":      outcome,
                    "account":      round(account, 2),
                })
                pos = None

        # ── 2. Open new position ──────────────────────────────────────────────
        if pos is None:
            # Weekly: only enter on Fridays
            if dte >= 7 and d.weekday() != 4:
                equity[d] = account
                continue

            # Skip if VIX too high
            if vix > MAX_VIX:
                equity[d] = account
                continue

            sigma = vix / 100.0 * IV_PREMIUM
            T     = max(dte, 1) / 365.0

            K1 = rs(spot * (1 - SHORT_OTM_PCT))
            sw = max(rs(spot * SPREAD_WIDTH_PCT), 1.0)
            K2 = K1 - sw

            credit = bs_put(spot, K1, T, sigma) - bs_put(spot, K2, T, sigma)
            credit = max(credit, 0.005)

            max_loss_pc = (sw - credit) * 100
            if max_loss_pc <= 0:
                equity[d] = account
                continue

            contracts = max(int(account * MAX_RISK_PCT // max_loss_pc), 1)

            # Find actual expiry date (next trading day dte steps ahead)
            future = dates[dates > d]
            if len(future) < dte:
                equity[d] = account
                continue
            expiry = future[min(dte - 1, len(future) - 1)]

            pos = {
                "entry":      d,
                "expiry":     expiry,
                "K1":         K1,
                "K2":         K2,
                "credit":     credit,
                "contracts":  contracts,
                "sigma":      sigma,
                "vix":        vix,
                "entry_spot": spot,
            }

        equity[d] = account

    eq     = pd.Series(equity).sort_index()
    tr_df  = pd.DataFrame(trades)
    return eq, tr_df


# ── metrics ───────────────────────────────────────────────────────────────────

def metrics(eq: pd.Series, tr: pd.DataFrame, dte: int) -> dict:
    if eq.empty:
        return {}

    total_return = (eq.iloc[-1] / INITIAL_CAPITAL - 1) * 100
    n_years      = (eq.index[-1] - eq.index[0]).days / 365
    cagr         = ((eq.iloc[-1] / INITIAL_CAPITAL) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

    daily_ret = eq.pct_change().dropna()
    sharpe    = (daily_ret.mean() / daily_ret.std() * math.sqrt(252)) if daily_ret.std() > 0 else 0

    roll_max  = eq.cummax()
    drawdown  = (eq - roll_max) / roll_max
    max_dd    = drawdown.min() * 100

    if tr.empty:
        return {"dte": dte, "trades": 0}

    n         = len(tr)
    win_rate  = (tr["outcome"].isin(["FULL_PROFIT", "TAKE_PROFIT"])).mean() * 100
    avg_pnl   = tr["pnl"].mean()
    avg_win   = tr.loc[tr["pnl"] > 0, "pnl"].mean() if (tr["pnl"] > 0).any() else 0
    avg_loss  = tr.loc[tr["pnl"] < 0, "pnl"].mean() if (tr["pnl"] < 0).any() else 0
    expectancy = avg_win * (win_rate / 100) + avg_loss * (1 - win_rate / 100)

    return {
        "dte":          dte,
        "final_value":  round(eq.iloc[-1], 0),
        "total_return": round(total_return, 1),
        "cagr":         round(cagr, 1),
        "sharpe":       round(sharpe, 2),
        "max_dd":       round(max_dd, 1),
        "n_trades":     n,
        "win_rate":     round(win_rate, 1),
        "avg_pnl":      round(avg_pnl, 2),
        "avg_win":      round(avg_win, 2),
        "avg_loss":     round(avg_loss, 2),
        "expectancy":   round(expectancy, 2),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\nDownloading SPY & VIX ({START} to {END})...")
    spy = yf.download("SPY",  start=START, end=END, auto_adjust=True, progress=False)
    vix = yf.download("^VIX", start=START, end=END, auto_adjust=True, progress=False)

    if spy.empty or vix.empty:
        print("ERROR: Could not download market data.")
        return

    spy.index = pd.to_datetime(spy.index)
    vix.index = pd.to_datetime(vix.index)

    # Flatten MultiIndex columns if present (yfinance returns them for single tickers)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    spy_bm_start = float(spy["Close"].iloc[0])
    spy_bm_end   = float(spy["Close"].iloc[-1])
    spy_bm_ret   = (spy_bm_end / spy_bm_start - 1) * 100

    print(f"SPY buy-and-hold: {spy_bm_ret:+.1f}%  ({spy_bm_start:.0f} to {spy_bm_end:.0f})\n")

    all_metrics = []
    all_equity  = {}

    for dte in [1, 2, 3, 7]:
        print(f"  Running {dte}DTE backtest...", end=" ", flush=True)
        eq, tr = run_backtest(spy, vix, dte=dte)
        m = metrics(eq, tr, dte)
        all_metrics.append(m)
        all_equity[f"{dte}DTE"] = eq
        print(f"${m.get('final_value', 0):,.0f}  ({m.get('n_trades', 0)} trades)")

    # ── summary table ─────────────────────────────────────────────────────────
    print()
    print("=" * 82)
    print(f"  DTE COMPARISON BACKTEST  |  Start ${INITIAL_CAPITAL:,.0f}  |  {START} - {END}")
    print(f"  OTM={SHORT_OTM_PCT*100:.1f}%  Width={SPREAD_WIDTH_PCT*100:.1f}%  MaxVIX={MAX_VIX}  TP={TP_PCT*100:.0f}%  MaxRisk={MAX_RISK_PCT*100:.0f}%/trade")
    print("=" * 82)
    hdr = f"{'Metric':<22}"
    for m in all_metrics:
        hdr += f"  {''+str(m['dte'])+'DTE':>12}"
    print(hdr)
    print("-" * 82)

    rows = [
        ("Final value ($)",    "final_value",  "${:,.0f}"),
        ("Total return (%)",   "total_return", "{:+.1f}%"),
        ("CAGR (%)",           "cagr",         "{:+.1f}%"),
        ("Sharpe ratio",       "sharpe",       "{:.2f}"),
        ("Max drawdown (%)",   "max_dd",       "{:.1f}%"),
        ("# Trades",           "n_trades",     "{:,}"),
        ("Win rate (%)",       "win_rate",     "{:.1f}%"),
        ("Avg P&L / trade",    "avg_pnl",      "${:+.2f}"),
        ("Avg win",            "avg_win",      "${:+.2f}"),
        ("Avg loss",           "avg_loss",     "${:+.2f}"),
        ("Expectancy / trade", "expectancy",   "${:+.2f}"),
    ]

    for label, key, fmt in rows:
        line = f"  {label:<20}"
        for m in all_metrics:
            val = m.get(key, 0)
            try:
                line += f"  {fmt.format(val):>12}"
            except Exception:
                line += f"  {'N/A':>12}"
        print(line)

    print("=" * 82)
    print()

    # ── year-by-year breakdown ────────────────────────────────────────────────
    print("  Year-by-year returns:")
    print(f"  {'Year':<6}", end="")
    for m in all_metrics:
        print(f"  {str(m['dte'])+'DTE':>8}", end="")
    print(f"  {'SPY':>8}")
    print("  " + "-" * 52)

    for year in sorted({d.year for d in spy.index}):
        print(f"  {year:<6}", end="")
        for label, eq in all_equity.items():
            yr = eq[eq.index.year == year]
            if len(yr) >= 2:
                ret = (yr.iloc[-1] / yr.iloc[0] - 1) * 100
                print(f"  {ret:>+7.1f}%", end="")
            else:
                print(f"  {'N/A':>8}", end="")
        spy_yr = spy["Close"][spy.index.year == year]
        if len(spy_yr) >= 2:
            spy_ret = (spy_yr.iloc[-1] / spy_yr.iloc[0] - 1) * 100
            print(f"  {spy_ret:>+7.1f}%")
        else:
            print()

    print()


if __name__ == "__main__":
    main()
