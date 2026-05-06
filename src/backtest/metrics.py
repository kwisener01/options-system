import os
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config.settings import REPORTS_DIR
from src.backtest.engine import BacktestResult


def compute_metrics(result: BacktestResult, top_n_features: int = 10) -> dict:
    eq = result.equity_curve.dropna()
    wr = result.weekly_returns.dropna()
    bench = result.benchmark_returns.reindex(wr.index).fillna(0)

    if eq.empty or len(wr) < 2:
        return {}

    total_return = (result.final_value / result.initial_capital) - 1
    n_weeks = len(wr)
    ann_return = (1 + total_return) ** (52 / n_weeks) - 1

    rf_weekly = 0.05 / 52
    excess = wr - rf_weekly
    sharpe = (excess.mean() / excess.std()) * np.sqrt(52) if excess.std() > 0 else 0.0

    rolling_max = eq.cummax()
    drawdown = (eq / rolling_max) - 1
    max_dd = float(drawdown.min())

    win_rate = float((wr > 0).mean())

    spy_total = float((1 + bench).prod() - 1)
    spy_ann = (1 + spy_total) ** (52 / n_weeks) - 1 if n_weeks > 0 else 0.0
    alpha = ann_return - spy_ann

    vol_ann = float(wr.std() * np.sqrt(52))

    oos = result.oos_auc_scores or []
    feat_imp = None
    if result.feature_importances is not None:
        feat_imp = result.feature_importances.head(top_n_features)

    return {
        "initial_capital": result.initial_capital,
        "final_value": result.final_value,
        "total_return_pct": round(total_return * 100, 2),
        "annualized_return_pct": round(ann_return * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "win_rate_pct": round(win_rate * 100, 1),
        "annualized_volatility_pct": round(vol_ann * 100, 2),
        "alpha_vs_spy_pct": round(alpha * 100, 2),
        "spy_annualized_return_pct": round(spy_ann * 100, 2),
        "n_weeks": n_weeks,
        "n_trades": len(result.trades_log),
        "oos_auc_mean": round(float(np.mean(oos)), 3) if oos else None,
        "oos_auc_std": round(float(np.std(oos)), 3) if oos else None,
        "oos_auc_n_folds": len(oos),
        "feature_importances": feat_imp,
    }


def print_report(metrics: dict, save: bool = True) -> None:
    auc = metrics.get("oos_auc_mean")
    auc_line = (
        f"  OOS AUC (mean/std)   : {auc:.3f} +/- {metrics.get('oos_auc_std', 0):.3f}"
        f"  ({metrics.get('oos_auc_n_folds', 0)} folds)"
        if auc else "  OOS AUC              : n/a"
    )

    lines = [
        "",
        "=" * 56,
        "  ML STOCK SELECTOR -- BACKTEST REPORT",
        "=" * 56,
        f"  Initial Capital      : ${metrics.get('initial_capital', 0):>12,.2f}",
        f"  Final Value          : ${metrics.get('final_value', 0):>12,.2f}",
        f"  Total Return         : {metrics.get('total_return_pct', 0):>+10.2f}%",
        f"  Annualized Return    : {metrics.get('annualized_return_pct', 0):>+10.2f}%",
        f"  SPY Annualized       : {metrics.get('spy_annualized_return_pct', 0):>+10.2f}%",
        f"  Alpha vs SPY         : {metrics.get('alpha_vs_spy_pct', 0):>+10.2f}%",
        "-" * 56,
        f"  Sharpe Ratio         : {metrics.get('sharpe_ratio', 0):>+10.3f}",
        f"  Max Drawdown         : {metrics.get('max_drawdown_pct', 0):>+10.2f}%",
        f"  Win Rate (weekly)    : {metrics.get('win_rate_pct', 0):>10.1f}%",
        f"  Annualized Volatility: {metrics.get('annualized_volatility_pct', 0):>10.2f}%",
        "-" * 56,
        auc_line,
        "-" * 56,
        f"  Weeks Tested         : {metrics.get('n_weeks', 0):>12}",
        f"  Total Trades         : {metrics.get('n_trades', 0):>12}",
        "=" * 56,
    ]

    feat_imp = metrics.get("feature_importances")
    if feat_imp is not None and not feat_imp.empty:
        lines.append("")
        lines.append("  TOP FEATURE IMPORTANCES (last fold)")
        lines.append("  " + "-" * 44)
        for feat, score in feat_imp.items():
            bar = "#" * int(score * 200)
            lines.append(f"  {feat:<26} {score:.4f}  {bar}")

    lines.append("")
    report = "\n".join(lines)
    print(report)

    if save:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        fname = os.path.join(REPORTS_DIR, f"metrics_{datetime.date.today()}.txt")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"  Report saved -> {fname}")


def plot_equity_curve(result: BacktestResult, save: bool = True) -> None:
    eq = result.equity_curve.dropna()
    bench = result.benchmark_returns.reindex(result.weekly_returns.index).fillna(0)

    spy_eq = (1 + bench).cumprod() * result.initial_capital
    spy_eq = spy_eq.reindex(eq.index).ffill()

    rolling_max = eq.cummax()
    drawdown = (eq / rolling_max - 1) * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("Backtest: ML Stock Selector vs SPY", fontsize=14, fontweight="bold")

    ax1.plot(eq.index, eq.values, label="Strategy", color="#2563EB", linewidth=1.8)
    ax1.plot(spy_eq.index, spy_eq.values, label="SPY (benchmark)", color="#9CA3AF", linewidth=1.2, linestyle="--")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.setp(ax1.get_xticklabels(), rotation=30)

    ax2.fill_between(drawdown.index, drawdown.values, 0, color="#EF4444", alpha=0.5, label="Drawdown")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.setp(ax2.get_xticklabels(), rotation=30)

    plt.tight_layout()

    if save:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        path = os.path.join(REPORTS_DIR, "equity_curve.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Chart saved -> {path}")
    else:
        plt.show()

    plt.close(fig)
