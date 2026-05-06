"""
Multi-strategy XSP/SPY options backtest.

Auto-selects the right spread each week based on macro regime + VIX:
  BULL   + normal VIX  → Bull Put Spread   (sell OTM puts)
  BEAR   + normal VIX  → Bear Call Spread  (sell OTM calls)
  NEUTRAL + low VIX    → Iron Condor       (sell puts AND calls)
  Vol crush (VIX spike then fall) → Bull Call Debit (buy cheap calls)
  High VIX / black-swan           → Cash

Usage:
    python run_multi_strategy_backtest.py
    python run_multi_strategy_backtest.py --dte 0 --take-profit 0.50
    python run_multi_strategy_backtest.py --risk-pct 0.05 --max-vix 20
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
from src.backtest.multi_strategy_engine import (
    MultiStrategyEngine,
    STRATEGY_CASH, STRATEGY_BULL_PUT, STRATEGY_BEAR_CALL,
    STRATEGY_IRON_CONDOR, STRATEGY_BULL_CALL_DEBIT,
)
from config.settings import REPORTS_DIR

COLORS = {
    STRATEGY_BULL_PUT:        "steelblue",
    STRATEGY_BEAR_CALL:       "tomato",
    STRATEGY_IRON_CONDOR:     "mediumpurple",
    STRATEGY_BULL_CALL_DEBIT: "gold",
    STRATEGY_CASH:            "lightgray",
}


def parse_args():
    p = argparse.ArgumentParser(description="Multi-strategy XSP options backtest")
    p.add_argument("--capital",      type=float, default=10_000.0)
    p.add_argument("--otm",          type=float, default=0.02,   help="Short OTM%% for credit spreads (default 0.02)")
    p.add_argument("--width",        type=float, default=0.01,   help="Spread width %% of spot (default 0.01)")
    p.add_argument("--max-vix",      type=float, default=25.0,   help="Hard skip above this VIX (default 25)")
    p.add_argument("--low-vol",      type=float, default=18.0,   help="Below this VIX, prefer iron condor (default 18)")
    p.add_argument("--risk-pct",     type=float, default=0.10,   help="Max account %% at risk per trade (default 0.10)")
    p.add_argument("--iv-premium",   type=float, default=1.20)
    p.add_argument("--dte",          type=int,   default=7,      help="7=weekly, 0=0DTE")
    p.add_argument("--take-profit",  type=float, default=0.50,   help="TP fraction (default 0.50 = 50%%)")
    p.add_argument("--period",       type=str,   default="2y")
    p.add_argument("--no-chart",     action="store_true")
    return p.parse_args()


def download_spy(period):
    logger = logging.getLogger(__name__)
    logger.info("Downloading SPY (%s)", period)
    raw = yf.download("SPY", period=period, interval="1d",
                      auto_adjust=True, progress=False)
    raw.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in raw.columns]
    return raw.dropna(subset=["close"])


def metrics(result, args):
    eq     = result.equity_curve
    dr     = eq.pct_change().dropna()
    years  = len(eq) / 252
    total  = result.final_value / result.initial_capital - 1
    ann    = (1 + total) ** (1 / years) - 1 if years > 0 else 0
    bm_t   = result.benchmark_curve.iloc[-1] / result.benchmark_curve.iloc[0] - 1
    bm_ann = (1 + bm_t) ** (1 / years) - 1 if years > 0 else 0
    sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    max_dd = float(((eq - eq.cummax()) / eq.cummax()).min())

    td = result.trades
    n  = len(td)
    wins = int((td["pnl"] > 0).sum()) if n > 0 else 0
    avg_pnl = td["pnl"].mean() if n > 0 else 0

    return dict(
        initial=result.initial_capital, final=result.final_value,
        total=total, ann=ann, bm_ann=bm_ann, alpha=ann - bm_ann,
        sharpe=sharpe, max_dd=max_dd, years=years,
        n_trades=n, win_rate=wins / n if n > 0 else 0,
        avg_pnl=avg_pnl,
        counts=result.strategy_counts,
        dte=args.dte, tp=args.take_profit, max_vix=args.max_vix,
        low_vol=args.low_vol,
    )


def print_report(m):
    tp_s  = f"{m['tp']:.0%} TP" if m["tp"] else "hold to expiry"
    dte_s = "0DTE" if m["dte"] == 0 else f"{m['dte']}-DTE weekly"
    c     = m["counts"]
    print()
    print("=" * 64)
    print("  MULTI-STRATEGY XSP OPTIONS -- BACKTEST REPORT")
    print("=" * 64)
    print(f"  Mode                   : {dte_s}  |  {tp_s}")
    print(f"  VIX gate / low-vol     : skip>{m['max_vix']:.0f}  |  condor<{m['low_vol']:.0f}")
    print(f"  Initial Capital        : ${m['initial']:>12,.2f}")
    print(f"  Final Value            : ${m['final']:>12,.2f}")
    print(f"  Total Return           : {m['total']:>+12.2%}")
    print(f"  Annualized Return      : {m['ann']:>+12.2%}")
    print(f"  SPY Annualized         : {m['bm_ann']:>+12.2%}")
    print(f"  Alpha vs SPY           : {m['alpha']:>+12.2%}")
    print("-" * 64)
    print(f"  Sharpe Ratio           : {m['sharpe']:>+12.3f}")
    print(f"  Max Drawdown           : {m['max_dd']:>+12.2%}")
    print("-" * 64)
    print(f"  Total trades           : {m['n_trades']:>12d}")
    print(f"  Win rate               : {m['win_rate']:>12.1%}")
    print(f"  Avg P&L / trade        : ${m['avg_pnl']:>11.2f}")
    print("-" * 64)
    print("  Strategy usage:")
    for s, n in c.items():
        bar = "#" * n
        print(f"    {s:<22} {n:>4}   {bar}")
    print("=" * 64)


def plot(result, m):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    eq  = result.equity_curve
    bm  = result.benchmark_curve
    td  = result.trades

    fig = plt.figure(figsize=(15, 11))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.30)

    # ── equity curve ─────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(eq.index, eq.values, color="steelblue", lw=1.8, label="Multi-Strategy")
    ax1.plot(bm.index, bm.values, color="orange",    lw=1.2, ls="--", label="SPY B&H")
    ax1.set_title(f"Multi-Strategy vs SPY  |  {m['ann']:+.1%} ann  |  Sharpe {m['sharpe']:.2f}")
    ax1.set_ylabel("Portfolio ($)")
    ax1.legend(); ax1.grid(alpha=0.3)

    # ── per-trade P&L coloured by strategy ───────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    if not td.empty:
        for _, row in td.iterrows():
            color = COLORS.get(row["strategy"], "gray")
            alpha = 0.8 if row["pnl"] > 0 else 0.5
            ax2.bar(row["close_date"], row["pnl"], color=color, width=4, alpha=alpha)
    ax2.axhline(0, color="black", lw=0.8)
    # legend patches
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=v, label=k) for k, v in COLORS.items() if k != STRATEGY_CASH]
    ax2.legend(handles=patches, fontsize=7, ncol=4)
    ax2.set_title("P&L per Trade (coloured by strategy)")
    ax2.set_ylabel("P&L ($)"); ax2.grid(alpha=0.3)

    # ── drawdown ─────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    dd = (eq - eq.cummax()) / eq.cummax() * 100
    ax3.fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.4)
    ax3.set_title(f"Drawdown (max {m['max_dd']:+.1%})")
    ax3.set_ylabel("DD (%)"); ax3.grid(alpha=0.3)

    # ── strategy usage pie ────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    counts = {k: v for k, v in m["counts"].items() if v > 0}
    ax4.pie(
        counts.values(),
        labels=counts.keys(),
        colors=[COLORS.get(k, "gray") for k in counts],
        autopct="%1.0f%%", startangle=90,
    )
    ax4.set_title("Strategy Mix")

    fname = os.path.join(REPORTS_DIR, "multi_strategy_equity.png")
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved -> {fname}")


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    args   = parse_args()

    logger.info("=== MULTI-STRATEGY BACKTEST START ===")
    spy   = download_spy(args.period)
    macro = build_macro_history(period=args.period)

    engine = MultiStrategyEngine(
        initial_capital=args.capital,
        short_otm_pct=args.otm,
        spread_width_pct=args.width,
        call_otm_pct=args.otm,
        max_risk_pct=args.risk_pct,
        iv_premium=args.iv_premium,
        max_vix_entry=args.max_vix,
        low_vol_threshold=args.low_vol,
        dte=args.dte,
        take_profit_pct=args.take_profit,
    )

    result = engine.run(spy, macro)
    m      = metrics(result, args)
    print_report(m)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    if not result.trades.empty:
        result.trades.to_csv(os.path.join(REPORTS_DIR, "multi_strategy_trades.csv"), index=False)

    if not args.no_chart:
        plot(result, m)

    logger.info("=== MULTI-STRATEGY BACKTEST COMPLETE ===")


if __name__ == "__main__":
    main()
