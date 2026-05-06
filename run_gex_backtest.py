"""
GEX strategy backtester.

Replays cached SPY options chains against historical daily OHLC to simulate
the PUT_WALL_BOUNCE / CALL_WALL_FADE / FLIP_BREAKOUT setups.

The backtest needs cached chain files in data/gex_chain/.
Run download_gex_chain.py each morning to build up the dataset.
For historical replay of already-cached days, chains are loaded from disk —
no API calls needed.

Simulation rules:
  - Signal generated at 9:45 AM using that day's cached chain + open price.
  - Trade entered at market open (approximated by day's open price).
  - Stop / target checked against day's low/high.
  - If high >= target first (conservative): WIN (target hit).
  - If low  <= stop  first (conservative): LOSS (stop hit).
  - Else: position closed at day's close (force-close P&L).
  - Only one trade per day. No overnight holds.

Usage:
    python run_gex_backtest.py
    python run_gex_backtest.py --period 3mo --allow-short
    python run_gex_backtest.py --risk-pct 0.01 --no-chart
"""
import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from src.logger import setup_logging
from src.analysis.gex_scanner import scan, load_chain
from src.analysis.gex_trader import generate_signal, TradeSignal
from config.settings import REPORTS_DIR, GEX_CHAIN_DIR


def parse_args():
    p = argparse.ArgumentParser(description="GEX strategy backtest")
    p.add_argument("--period",      type=str,   default="3mo")
    p.add_argument("--capital",     type=float, default=10_000.0)
    p.add_argument("--risk-pct",    type=float, default=0.01,  help="Fraction of equity risked per trade")
    p.add_argument("--allow-short", action="store_true", default=False)
    p.add_argument("--no-chart",    action="store_true")
    return p.parse_args()


@dataclass
class BacktestTrade:
    date:       date
    setup:      str
    direction:  str
    entry:      float
    stop:       float
    target:     float
    exit_price: float
    pnl:        float
    outcome:    str          # TARGET_HIT | STOP_HIT | FORCE_CLOSE | NO_TRADE
    rr:         float
    confidence: str
    gex_regime: str


@dataclass
class BacktestResult:
    trades:         list = field(default_factory=list)
    equity_curve:   pd.Series = None
    initial_capital: float = 10_000.0
    final_value:    float = 10_000.0


def _simulate_day(row: pd.Series, signal: TradeSignal, qty: int) -> tuple[float, str]:
    """
    Given a daily OHLC row and a signal, return (exit_price, outcome).
    Conservative fill assumptions: stop checked against low, target against high.
    Entry at open.
    """
    o, h, l, c = row["open"], row["high"], row["low"], row["close"]

    if signal.direction == "LONG":
        if h >= signal.target and l <= signal.stop:
            # Both hit same day — assume stop hit first (conservative)
            return signal.stop, "STOP_HIT"
        if h >= signal.target:
            return signal.target, "TARGET_HIT"
        if l <= signal.stop:
            return signal.stop, "STOP_HIT"
        return c, "FORCE_CLOSE"

    else:  # SHORT
        if l <= signal.target and h >= signal.stop:
            return signal.stop, "STOP_HIT"
        if l <= signal.target:
            return signal.target, "TARGET_HIT"
        if h >= signal.stop:
            return signal.stop, "STOP_HIT"
        return c, "FORCE_CLOSE"


def run_backtest(spy_ohlc: pd.DataFrame, args) -> BacktestResult:
    logger  = logging.getLogger(__name__)
    capital = args.capital
    result  = BacktestResult(initial_capital=capital)
    equity  = capital
    eq_vals = []

    for dt, row in spy_ohlc.iterrows():
        d = dt.date() if hasattr(dt, "date") else dt

        # Load cached chain for this date
        contracts = load_chain(d)
        if not contracts:
            logger.debug("No cached chain for %s — skipping", d)
            eq_vals.append(equity)
            continue

        # Reconstruct spot from open price (signal fires at 9:45 AM ≈ open)
        spot = float(row["open"])

        # VIX proxy: use closing VIX from prior day if available (simplified: use 0 change)
        vix_row = spy_ohlc.get("vix_close") if "vix_close" in spy_ohlc.columns else None
        vix      = float(vix_row.get(dt, 17.0)) if vix_row is not None else 17.0
        vix_prev = vix   # simplified: no prev-day VIX in standard SPY download

        gex_result = scan(as_of=d, spot=spot, vix=vix, vix_prev=vix_prev)
        if gex_result is None:
            eq_vals.append(equity)
            continue

        signal = generate_signal(gex_result, allow_short=args.allow_short)

        if signal.direction == "NO_TRADE":
            result.trades.append(BacktestTrade(
                date=d, setup=signal.setup, direction="NO_TRADE",
                entry=spot, stop=0, target=0, exit_price=spot,
                pnl=0, outcome="NO_TRADE", rr=0,
                confidence=signal.confidence, gex_regime=gex_result.gex_regime,
            ))
            eq_vals.append(equity)
            continue

        # Size: 1% equity / risk_per_share
        risk_per_share = abs(signal.entry - signal.stop)
        qty = int(equity * args.risk_pct / risk_per_share) if risk_per_share > 0 else 0
        if qty <= 0:
            eq_vals.append(equity)
            continue

        exit_price, outcome = _simulate_day(row, signal, qty)
        mult = 1 if signal.direction == "LONG" else -1
        pnl  = (exit_price - signal.entry) * qty * mult

        equity = round(equity + pnl, 2)
        result.trades.append(BacktestTrade(
            date=d, setup=signal.setup, direction=signal.direction,
            entry=signal.entry, stop=signal.stop, target=signal.target,
            exit_price=exit_price, pnl=pnl, outcome=outcome, rr=signal.rr,
            confidence=signal.confidence, gex_regime=gex_result.gex_regime,
        ))
        eq_vals.append(equity)
        logger.info("%s  %s %s  entry=%.2f exit=%.2f  pnl=%+.2f  %s",
                    d, signal.direction, signal.setup, signal.entry, exit_price, pnl, outcome)

    result.final_value  = equity
    result.equity_curve = pd.Series(eq_vals, index=spy_ohlc.index)
    return result


def print_report(result: BacktestResult, args):
    trades = [t for t in result.trades if t.direction != "NO_TRADE"]
    n      = len(trades)
    no_trade = len([t for t in result.trades if t.direction == "NO_TRADE"])
    wins   = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]

    eq     = result.equity_curve.dropna()
    dr     = eq.pct_change().dropna()
    years  = max(len(eq) / 252, 0.01)
    total  = result.final_value / result.initial_capital - 1
    ann    = (1 + total) ** (1 / years) - 1
    sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    max_dd = float(((eq - eq.cummax()) / eq.cummax()).min())

    setups = {}
    for t in trades:
        setups.setdefault(t.setup, {"n": 0, "wins": 0, "pnl": 0.0})
        setups[t.setup]["n"] += 1
        setups[t.setup]["wins"] += 1 if t.pnl > 0 else 0
        setups[t.setup]["pnl"] += t.pnl

    print()
    print("=" * 62)
    print("  GEX STRATEGY — BACKTEST REPORT")
    print("=" * 62)
    print(f"  Period              : {result.equity_curve.index[0].date()} – {result.equity_curve.index[-1].date()}")
    print(f"  Allow short         : {args.allow_short}")
    print(f"  Risk per trade      : {args.risk_pct:.0%}")
    print(f"  Initial capital     : ${result.initial_capital:>10,.2f}")
    print(f"  Final value         : ${result.final_value:>10,.2f}")
    print(f"  Total return        : {total:>+10.2%}")
    print(f"  Annualised return   : {ann:>+10.2%}")
    print("-" * 62)
    print(f"  Sharpe ratio        : {sharpe:>10.3f}")
    print(f"  Max drawdown        : {max_dd:>+10.2%}")
    print("-" * 62)
    print(f"  Trading days scanned: {len(result.trades):>10d}")
    print(f"  Trades taken        : {n:>10d}  ({n/max(len(result.trades),1):.0%} of days)")
    print(f"  No-trade days       : {no_trade:>10d}")
    print(f"  Win rate            : {len(wins)/n:.1%}" if n else "  Win rate           : n/a")
    print(f"  Avg win             : ${np.mean([t.pnl for t in wins]):>+9.2f}" if wins else "  Avg win            : n/a")
    print(f"  Avg loss            : ${np.mean([t.pnl for t in losses]):>+9.2f}" if losses else "  Avg loss           : n/a")
    print(f"  Avg R:R             : {np.mean([t.rr for t in trades]):>10.2f}" if trades else "")
    print("-" * 62)
    print("  By setup:")
    for setup, s in sorted(setups.items()):
        wr = s["wins"] / s["n"] if s["n"] else 0
        print(f"    {setup:<26} {s['n']:>3} trades  {wr:.0%} WR  ${s['pnl']:>+8.2f}")
    print("=" * 62)

    if n == 0:
        print("\n  NOTE: Zero trades taken. This usually means no cached chain files")
        print("  exist in data/gex_chain/. Run download_gex_chain.py each morning")
        print("  to build up the dataset. Historical backtest requires prior daily")
        print("  snapshots — yfinance does not provide historical options OI.\n")


def plot(result: BacktestResult):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    trades = [t for t in result.trades if t.direction != "NO_TRADE"]
    eq     = result.equity_curve.dropna()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle("GEX Intraday Strategy — Backtest", fontsize=13)

    ax1.plot(eq.index, eq.values, color="steelblue", lw=1.8)
    ax1.set_ylabel("Portfolio ($)")
    ax1.set_title(f"Equity Curve  |  {result.initial_capital:,.0f} -> {result.final_value:,.0f}")
    ax1.grid(alpha=0.3)

    colors = {"TARGET_HIT": "green", "STOP_HIT": "red", "FORCE_CLOSE": "orange"}
    for t in trades:
        ts = pd.Timestamp(t.date)
        ax2.bar(ts, t.pnl, color=colors.get(t.outcome, "gray"), width=1.5, alpha=0.7)
    ax2.axhline(0, color="black", lw=0.7)
    ax2.set_ylabel("Daily P&L ($)")
    ax2.set_title("P&L per Trade")
    ax2.grid(alpha=0.3)

    import matplotlib.patches as mpatches
    legend = [mpatches.Patch(color=v, label=k) for k, v in colors.items()]
    ax2.legend(handles=legend, fontsize=8)

    plt.tight_layout()
    fname = os.path.join(REPORTS_DIR, "gex_backtest_equity.png")
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved -> {fname}")


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    args   = parse_args()

    # Check for cached chains
    chain_files = []
    if os.path.isdir(GEX_CHAIN_DIR):
        chain_files = [f for f in os.listdir(GEX_CHAIN_DIR) if f.endswith(".pkl")]

    logger.info("Found %d cached chain file(s) in %s", len(chain_files), GEX_CHAIN_DIR)
    if not chain_files:
        print("\nNo cached chain files found. Run download_gex_chain.py each morning to build")
        print("up a historical dataset. Today's chain will be automatically cached on first scan.\n")

    logger.info("Downloading SPY OHLC (%s)...", args.period)
    raw = yf.download("SPY", period=args.period, interval="1d",
                      auto_adjust=True, progress=False)
    raw.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in raw.columns]
    spy = raw.dropna(subset=["close"])

    result = run_backtest(spy, args)
    print_report(result, args)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    trade_rows = [vars(t) for t in result.trades if t.direction != "NO_TRADE"]
    if trade_rows:
        pd.DataFrame(trade_rows).to_csv(
            os.path.join(REPORTS_DIR, "gex_backtest_trades.csv"), index=False)

    if not args.no_chart:
        plot(result)


if __name__ == "__main__":
    main()
