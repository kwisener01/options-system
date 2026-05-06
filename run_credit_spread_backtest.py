"""
XSP/SPY bull put spread backtest — weekly, 0DTE, with optional take-profit.

Usage:
    python run_credit_spread_backtest.py                          # weekly, hold to expiry
    python run_credit_spread_backtest.py --dte 0                  # 0DTE, daily
    python run_credit_spread_backtest.py --take-profit 0.50       # close at 50%% max profit
    python run_credit_spread_backtest.py --otm 0.03 --max-vix 20 # tighter filter
"""
import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf

sys.path.insert(0, os.path.dirname(__file__))

from src.logger import setup_logging
from src.analysis.macro_scanner import build_macro_history
from src.backtest.credit_spread_engine import CreditSpreadEngine
from config.settings import REPORTS_DIR


def parse_args():
    p = argparse.ArgumentParser(description="XSP bull put spread backtest")
    p.add_argument("--capital",      type=float, default=10_000.0,  help="Starting capital ($)")
    p.add_argument("--otm",          type=float, default=0.02,      help="Short put OTM%% (default 0.02 = 2%%)")
    p.add_argument("--width",        type=float, default=0.01,      help="Spread width as %% of spot (default 0.01 = 1%%)")
    p.add_argument("--max-vix",      type=float, default=25.0,      help="Skip entry when VIX above this (default 25)")
    p.add_argument("--risk-pct",     type=float, default=0.10,      help="Max %% of account to risk per trade (default 0.10)")
    p.add_argument("--iv-premium",   type=float, default=1.20,      help="IV premium over realized vol (default 1.20)")
    p.add_argument("--dte",          type=int,   default=7,         help="Days to expiration: 7=weekly, 0=0DTE (default 7)")
    p.add_argument("--take-profit",  type=float, default=None,      help="Close at this fraction of max profit, e.g. 0.50 = 50%%")
    p.add_argument("--period",       type=str,   default="2y",      help="Lookback period for yfinance (default 2y)")
    p.add_argument("--no-chart",     action="store_true",           help="Skip chart generation")
    return p.parse_args()


def download_spy(period: str) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    logger.info("Downloading SPY prices (period=%s)", period)
    raw = yf.download("SPY", period=period, interval="1d",
                      auto_adjust=True, progress=False)
    raw.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in raw.columns]
    raw = raw.dropna(subset=["close"])
    logger.info("SPY: %d trading days", len(raw))
    return raw


def compute_metrics(result, args) -> dict:
    eq = result.equity_curve
    daily_ret = eq.pct_change().dropna()

    n_days = len(eq)
    years = n_days / 252

    total_ret = (result.final_value / result.initial_capital) - 1
    ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0

    bm_total = (result.benchmark_curve.iloc[-1] / result.benchmark_curve.iloc[0]) - 1
    bm_ann = (1 + bm_total) ** (1 / years) - 1 if years > 0 else 0

    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0

    roll_max = eq.cummax()
    max_dd = float(((eq - roll_max) / roll_max).min())

    trades = result.trades
    n_closed = result.n_full_profit + result.n_take_profit + result.n_partial_loss + result.n_max_loss
    win_rate = (result.n_full_profit + result.n_take_profit) / n_closed if n_closed > 0 else 0
    avg_credit = trades["credit_per_share"].mean() if not trades.empty else 0
    avg_pnl = trades["pnl"].mean() if not trades.empty else 0

    return {
        "initial_capital": result.initial_capital,
        "final_value":     result.final_value,
        "total_ret":       total_ret,
        "ann_ret":         ann_ret,
        "bm_ann":          bm_ann,
        "alpha":           ann_ret - bm_ann,
        "sharpe":          sharpe,
        "max_dd":          max_dd,
        "n_entered":       result.n_entered,
        "n_skipped":       result.n_skipped,
        "n_full_profit":   result.n_full_profit,
        "n_take_profit":   result.n_take_profit,
        "n_partial_loss":  result.n_partial_loss,
        "n_max_loss":      result.n_max_loss,
        "win_rate":        win_rate,
        "avg_credit":      avg_credit,
        "avg_pnl":         avg_pnl,
        "total_premium":   result.total_premium_collected,
        "total_losses":    result.total_losses,
        "years":           years,
        "otm":             args.otm,
        "width":           args.width,
        "max_vix":         args.max_vix,
        "risk_pct":        args.risk_pct,
        "dte":             args.dte,
        "take_profit":     args.take_profit,
    }


def print_report(m: dict):
    tp_str  = f"{m['take_profit']:.0%} TP" if m["take_profit"] else "hold to expiry"
    dte_str = "0DTE" if m["dte"] == 0 else f"{m['dte']}-DTE weekly"
    print()
    print("=" * 62)
    print("  XSP BULL PUT SPREAD -- BACKTEST REPORT")
    print("=" * 62)
    print(f"  Mode                 : {dte_str}  |  {tp_str}")
    print(f"  Strategy             : {m['otm']*100:.0f}% OTM short / {m['width']*100:.0f}% wide")
    print(f"  VIX gate             : skip entry when VIX > {m['max_vix']:.0f}")
    print(f"  Risk per trade       : {m['risk_pct']*100:.0f}% of account")
    print(f"  Initial Capital      : ${m['initial_capital']:>12,.2f}")
    print(f"  Final Value          : ${m['final_value']:>12,.2f}")
    print(f"  Total Return         : {m['total_ret']:>+12.2%}")
    print(f"  Annualized Return    : {m['ann_ret']:>+12.2%}")
    print(f"  SPY Annualized       : {m['bm_ann']:>+12.2%}")
    print(f"  Alpha vs SPY         : {m['alpha']:>+12.2%}")
    print("-" * 62)
    print(f"  Sharpe Ratio         : {m['sharpe']:>+12.3f}")
    print(f"  Max Drawdown         : {m['max_dd']:>+12.2%}")
    print("-" * 62)
    print(f"  Spreads entered      : {m['n_entered']:>12d}")
    print(f"  Spreads skipped      : {m['n_skipped']:>12d}  (VIX/regime gate)")
    print(f"  Full profit (expiry) : {m['n_full_profit']:>12d}")
    print(f"  Take profit (early)  : {m['n_take_profit']:>12d}")
    print(f"  Win rate             : {m['win_rate']:>12.1%}")
    print(f"  Partial loss         : {m['n_partial_loss']:>12d}")
    print(f"  Max loss             : {m['n_max_loss']:>12d}")
    print(f"  Avg credit/share     : ${m['avg_credit']:>11.3f}")
    print(f"  Avg P&L / trade      : ${m['avg_pnl']:>11.2f}")
    print(f"  Total premium earned : ${m['total_premium']:>11,.2f}")
    print(f"  Total losses         : ${m['total_losses']:>11,.2f}")
    print("=" * 62)


def plot_chart(result, m: dict):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.4)

    eq = result.equity_curve
    bm = result.benchmark_curve

    # Panel 1: equity curves
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(eq.index, eq.values, label="Bull Put Spread", color="steelblue", linewidth=1.5)
    ax1.plot(bm.index, bm.values, label="SPY B&H", color="orange", linestyle="--", linewidth=1.2)
    ax1.set_title(f"XSP Bull Put Spread vs SPY  |  {m['ann_ret']:+.1%} ann  |  Sharpe {m['sharpe']:.2f}")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Panel 2: weekly P&L per trade
    ax2 = fig.add_subplot(gs[1])
    if not result.trades.empty:
        pnl = result.trades.set_index("expiry_date")["pnl"]
        colors = ["green" if v > 0 else "red" for v in pnl.values]
        ax2.bar(pnl.index, pnl.values, color=colors, width=5, alpha=0.7)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("P&L per Spread (at expiration)")
    ax2.set_ylabel("P&L ($)")
    ax2.grid(alpha=0.3)

    # Panel 3: drawdown
    ax3 = fig.add_subplot(gs[2])
    roll_max = eq.cummax()
    dd = (eq - roll_max) / roll_max * 100
    ax3.fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.4)
    ax3.set_title(f"Drawdown (max {m['max_dd']:+.1%})")
    ax3.set_ylabel("Drawdown (%)")
    ax3.grid(alpha=0.3)

    fname = os.path.join(REPORTS_DIR, "credit_spread_equity.png")
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved -> {fname}")


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    args = parse_args()

    logger.info("=== CREDIT SPREAD BACKTEST START ===")

    spy = download_spy(period=args.period)
    if len(spy) < 50:
        logger.error("Insufficient SPY data")
        sys.exit(1)

    logger.info("Building macro history (VIX + regime)...")
    macro = build_macro_history(period=args.period)

    engine = CreditSpreadEngine(
        initial_capital=args.capital,
        short_otm_pct=args.otm,
        spread_width_pct=args.width,
        max_risk_pct=args.risk_pct,
        iv_premium=args.iv_premium,
        max_vix_entry=args.max_vix,
        dte=args.dte,
        take_profit_pct=args.take_profit,
    )

    logger.info("Running simulation...")
    result = engine.run(spy, macro)

    metrics = compute_metrics(result, args)
    print_report(metrics)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    if not result.trades.empty:
        trades_path = os.path.join(REPORTS_DIR, "credit_spread_trades.csv")
        result.trades.to_csv(trades_path, index=False)
        logger.info("Trades saved -> %s", trades_path)

    if not args.no_chart:
        plot_chart(result, metrics)

    logger.info("=== CREDIT SPREAD BACKTEST COMPLETE ===")


if __name__ == "__main__":
    main()
