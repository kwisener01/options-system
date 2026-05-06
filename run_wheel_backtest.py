"""
Wheel strategy backtest on historical price data.

Usage:
    python run_wheel_backtest.py
    python run_wheel_backtest.py --tickers XSP SPY --capital 50000
    python run_wheel_backtest.py --put-otm 0.05 --call-otm 0.03 --period 2y
    python run_wheel_backtest.py --no-chart

Note on XSP:
    XSP (CBOE Mini-SPX) is a cash-settled index — there is no assignment of
    shares.  For backtesting we simulate it as if shares were received (P&L is
    equivalent).  For a true equity wheel, use SPY or any optionable stock.
"""
import argparse
import datetime
import logging
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(__file__))

from src.logger import setup_logging
from src.backtest.wheel_engine import WheelEngine, WheelResult
from config.settings import (
    WHEEL_TICKERS,
    WHEEL_PUT_OTM_PCT,
    WHEEL_CALL_OTM_PCT,
    WHEEL_IV_PREMIUM,
    WHEEL_VOL_WINDOW,
    WHEEL_INITIAL_CAPITAL,
    REPORTS_DIR,
)

logger = logging.getLogger(__name__)


# ── Data download ─────────────────────────────────────────────────────────────

def download_prices(tickers: list, period: str) -> dict:
    prices = {}
    fallbacks = {"XSP": "SPY"}   # XSP cash-settled -> use SPY as price proxy

    for sym in tickers:
        target = sym
        df = _fetch(sym, period)
        if df is None or df.empty:
            alt = fallbacks.get(sym.upper())
            if alt:
                logger.warning(
                    "%s: no yfinance data — falling back to %s as price proxy", sym, alt
                )
                df = _fetch(alt, period)
                target = alt
            if df is None or df.empty:
                logger.error("Could not download price data for %s (or fallback). Skipping.", sym)
                continue

        df.columns = [c.lower() for c in df.columns]
        df = df.dropna(subset=["close"])
        if len(df) < 60:
            logger.warning("%s: only %d trading days — skipping", sym, len(df))
            continue
        prices[sym] = df
        if target != sym:
            prices[target] = df   # keep SPY separately for benchmark

    return prices


def _fetch(sym: str, period: str) -> pd.DataFrame | None:
    try:
        raw = yf.download(sym, period=period, interval="1d", auto_adjust=True, progress=False)
        if raw.empty:
            return None
        # yfinance may return MultiIndex columns when given a single ticker in newer versions
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.index = pd.to_datetime(raw.index)
        return raw
    except Exception as exc:
        logger.debug("yfinance error for %s: %s", sym, exc)
        return None


# ── Report ────────────────────────────────────────────────────────────────────

def compute_metrics(result: WheelResult) -> dict:
    eq = result.equity_curve.dropna()
    if eq.empty or len(eq) < 2:
        return {}

    monthly_rets = eq.pct_change().dropna()
    total_ret = (result.final_value / result.initial_capital) - 1
    n_months = len(monthly_rets)
    ann_ret = (1 + total_ret) ** (12 / n_months) - 1 if n_months else 0.0

    rf_monthly = _RF_ANN / 12
    excess = monthly_rets - rf_monthly
    sharpe = (excess.mean() / excess.std()) * math.sqrt(12) if excess.std() > 0 else 0.0

    rolling_max = eq.cummax()
    drawdown = (eq / rolling_max) - 1
    max_dd = float(drawdown.min())

    bm = result.benchmark_curve.reindex(monthly_rets.index).ffill()
    bm_total = float((bm.iloc[-1] / bm.iloc[0]) - 1) if len(bm) >= 2 else 0.0
    bm_ann = (1 + bm_total) ** (12 / n_months) - 1 if n_months else 0.0
    alpha = ann_ret - bm_ann

    total_cycles = (
        result.n_put_expirations
        + result.n_assignments
        + result.n_call_expirations
        + result.n_call_aways
    )
    assign_rate = result.n_assignments / (result.n_put_expirations + result.n_assignments) if (result.n_put_expirations + result.n_assignments) > 0 else 0.0
    call_rate = result.n_call_aways / (result.n_call_expirations + result.n_call_aways) if (result.n_call_expirations + result.n_call_aways) > 0 else 0.0

    return {
        "initial_capital": result.initial_capital,
        "final_value": result.final_value,
        "total_return_pct": round(total_ret * 100, 2),
        "annualized_return_pct": round(ann_ret * 100, 2),
        "bm_annualized_return_pct": round(bm_ann * 100, 2),
        "alpha_pct": round(alpha * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "annualized_vol_pct": round(float(monthly_rets.std() * math.sqrt(12)) * 100, 2),
        "total_premium": round(result.total_premium, 2),
        "n_assignments": result.n_assignments,
        "n_put_expirations": result.n_put_expirations,
        "n_call_expirations": result.n_call_expirations,
        "n_call_aways": result.n_call_aways,
        "total_cycles": total_cycles,
        "assignment_rate_pct": round(assign_rate * 100, 1),
        "call_away_rate_pct": round(call_rate * 100, 1),
        "n_months": n_months,
    }


_RF_ANN = 0.045


def print_report(metrics: dict, tickers: list, save: bool = True) -> None:
    lines = [
        "",
        "=" * 58,
        "  WHEEL STRATEGY BACKTEST REPORT",
        f"  Tickers: {', '.join(tickers)}",
        "=" * 58,
        f"  Initial Capital       : ${metrics.get('initial_capital', 0):>12,.2f}",
        f"  Final Value           : ${metrics.get('final_value', 0):>12,.2f}",
        f"  Total Return          : {metrics.get('total_return_pct', 0):>+10.2f}%",
        f"  Annualized Return     : {metrics.get('annualized_return_pct', 0):>+10.2f}%",
        f"  Buy-and-Hold Ann.     : {metrics.get('bm_annualized_return_pct', 0):>+10.2f}%",
        f"  Alpha vs B&H          : {metrics.get('alpha_pct', 0):>+10.2f}%",
        "-" * 58,
        f"  Sharpe Ratio          : {metrics.get('sharpe_ratio', 0):>+10.3f}",
        f"  Max Drawdown          : {metrics.get('max_drawdown_pct', 0):>+10.2f}%",
        f"  Annualized Volatility : {metrics.get('annualized_vol_pct', 0):>10.2f}%",
        "-" * 58,
        f"  Total Premium Collected: ${metrics.get('total_premium', 0):>11,.2f}",
        f"  Put Cycles (worthless): {metrics.get('n_put_expirations', 0):>12}",
        f"  Assignments           : {metrics.get('n_assignments', 0):>12}",
        f"  Assignment Rate       : {metrics.get('assignment_rate_pct', 0):>10.1f}%",
        f"  Call Cycles (worthless):{metrics.get('n_call_expirations', 0):>12}",
        f"  Called Away           : {metrics.get('n_call_aways', 0):>12}",
        f"  Call-Away Rate        : {metrics.get('call_away_rate_pct', 0):>10.1f}%",
        f"  Total Option Cycles   : {metrics.get('total_cycles', 0):>12}",
        "-" * 58,
        f"  Months Tested         : {metrics.get('n_months', 0):>12}",
        "=" * 58,
        "",
    ]
    report = "\n".join(lines)
    print(report)

    if save:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        fname = os.path.join(REPORTS_DIR, f"wheel_metrics_{datetime.date.today()}.txt")
        with open(fname, "w") as f:
            f.write(report)
        print(f"  Report saved -> {fname}")


def plot_wheel(result: WheelResult, tickers: list, save: bool = True) -> None:
    eq = result.equity_curve.dropna()
    bm = result.benchmark_curve.reindex(eq.index).ffill()
    rolling_max = eq.cummax()
    drawdown = (eq / rolling_max - 1) * 100

    # Monthly premium from trades
    prem_by_month: dict = {}
    if not result.trades.empty and "action" in result.trades.columns:
        sold = result.trades[result.trades["action"].isin(["EXPIRED_PUT", "EXPIRED_CALL", "CALLED_AWAY"])]
        if not sold.empty:
            sold = sold.copy()
            sold["month"] = pd.to_datetime(sold["date"]).dt.to_period("M").dt.to_timestamp()
            prem_col = "realized_pnl" if "realized_pnl" in sold.columns else "premium_per_share"
            prem_by_month = sold.groupby("month")[prem_col].sum().to_dict()

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), gridspec_kw={"height_ratios": [3, 1.5, 1]})
    fig.suptitle(f"Wheel Strategy Backtest — {', '.join(tickers)}", fontsize=13, fontweight="bold")

    # Panel 1: equity vs benchmark
    ax1 = axes[0]
    ax1.plot(eq.index, eq.values, label="Wheel Strategy", color="#2563EB", linewidth=1.8)
    ax1.plot(bm.index, bm.values, label="Buy & Hold", color="#9CA3AF", linewidth=1.2, linestyle="--")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.setp(ax1.get_xticklabels(), rotation=30)

    # Panel 2: monthly premium income
    ax2 = axes[1]
    if prem_by_month:
        months = pd.Series(prem_by_month).sort_index()
        ax2.bar(months.index, months.values, color="#10B981", alpha=0.8, width=20, label="Premium Collected")
        ax2.set_ylabel("Premium ($)")
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        plt.setp(ax2.get_xticklabels(), rotation=30)
    else:
        ax2.text(0.5, 0.5, "No realized premium data", ha="center", va="center", transform=ax2.transAxes)

    # Panel 3: drawdown
    ax3 = axes[2]
    ax3.fill_between(drawdown.index, drawdown.values, 0, color="#EF4444", alpha=0.5, label="Drawdown")
    ax3.set_ylabel("Drawdown (%)")
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.setp(ax3.get_xticklabels(), rotation=30)

    plt.tight_layout()

    if save:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        path = os.path.join(REPORTS_DIR, "wheel_equity_curve.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Chart saved -> {path}")
    else:
        plt.show()

    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Wheel strategy backtest")
    p.add_argument("--tickers", nargs="+", default=WHEEL_TICKERS, help="Tickers to wheel (default: XSP)")
    p.add_argument("--capital", type=float, default=WHEEL_INITIAL_CAPITAL, help="Starting capital ($)")
    p.add_argument("--put-otm", type=float, default=WHEEL_PUT_OTM_PCT, help="Put OTM fraction (e.g. 0.05 = 5%%)")
    p.add_argument("--call-otm", type=float, default=WHEEL_CALL_OTM_PCT, help="Call OTM fraction above cost basis")
    p.add_argument("--iv-premium", type=float, default=WHEEL_IV_PREMIUM, help="Realized-vol multiplier for IV")
    p.add_argument("--period", default="2y", help="yfinance history period (e.g. 1y, 2y)")
    p.add_argument("--no-chart", action="store_true", help="Skip chart generation")
    return p.parse_args()


def main():
    setup_logging()
    args = parse_args()

    logger.info("=== WHEEL BACKTEST START ===")
    logger.info(
        "Tickers=%s | Capital=$%.0f | PutOTM=%.1f%% | CallOTM=%.1f%% | IVpremium=%.2fx | Period=%s",
        args.tickers, args.capital, args.put_otm * 100, args.call_otm * 100, args.iv_premium, args.period,
    )

    prices = download_prices(args.tickers, args.period)
    if not prices:
        logger.error("No price data downloaded — check tickers and internet connection")
        sys.exit(1)

    engine = WheelEngine(
        initial_capital=args.capital,
        tickers=args.tickers,
        put_otm_pct=args.put_otm,
        call_otm_pct=args.call_otm,
        iv_premium=args.iv_premium,
        vol_window=WHEEL_VOL_WINDOW,
    )

    result = engine.run(prices)

    if result.equity_curve.empty:
        logger.error("Backtest produced no equity curve — not enough data")
        sys.exit(1)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    if not result.trades.empty:
        trades_path = os.path.join(REPORTS_DIR, "wheel_trades_log.csv")
        result.trades.to_csv(trades_path, index=False)
        logger.info("Trades log saved -> %s", trades_path)

    metrics = compute_metrics(result)
    if not metrics:
        logger.error("Metrics computation failed — not enough data points")
        sys.exit(1)

    print_report(metrics, args.tickers)

    if not args.no_chart:
        plot_wheel(result, args.tickers)

    logger.info("=== WHEEL BACKTEST COMPLETE ===")


if __name__ == "__main__":
    main()
