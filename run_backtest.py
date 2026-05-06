"""
Run a full walk-forward backtest of the ML stock selection strategy.

Usage:
    python run_backtest.py
    python run_backtest.py --capital 25000 --top-n 15
    python run_backtest.py --lookback 252
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.logger import setup_logging
from src.backtest.data_prep import prepare_backtest_data
from src.signals.ml_selector import MLSelector
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import compute_metrics, print_report, plot_equity_curve
from config.settings import BACKTEST_INITIAL_CAPITAL, BACKTEST_SLIPPAGE_PCT, ML_TOP_N, ML_LOOKBACK_DAYS, REPORTS_DIR


def parse_args():
    p = argparse.ArgumentParser(description="ML stock selector backtest")
    p.add_argument("--capital", type=float, default=BACKTEST_INITIAL_CAPITAL, help="Starting capital ($)")
    p.add_argument("--top-n", type=int, default=ML_TOP_N, help="Stocks to hold per week")
    p.add_argument("--slippage", type=float, default=BACKTEST_SLIPPAGE_PCT, help="Slippage per side (e.g. 0.0005)")
    p.add_argument("--lookback", type=int, default=ML_LOOKBACK_DAYS, help="History in trading days")
    p.add_argument("--max-universe", type=int, default=500, help="Cap universe size (0=no limit, default 500)")
    p.add_argument("--max-atr", type=float, default=0.0, help="Skip stocks with daily ATR%% above this (0=disabled)")
    p.add_argument("--vix-buffer", type=float, default=0.0, help="Cash fraction held back when VIX>20 (0=disabled)")
    p.add_argument("--no-chart", action="store_true", help="Skip generating the equity curve chart")
    return p.parse_args()


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    args = parse_args()

    logger.info("=== BACKTEST START ===")
    logger.info("Capital=$%.2f | TopN=%d | Slippage=%.4f%%", args.capital, args.top_n, args.slippage * 100)

    # 1. Data
    labeled, prices = prepare_backtest_data(lookback_days=args.lookback, max_symbols=args.max_universe)
    if labeled.empty:
        logger.error("No data prepared — exiting")
        sys.exit(1)

    # 2. Train model (walk-forward handled inside engine)
    selector = MLSelector()

    # 3. Run backtest
    engine = BacktestEngine(
        initial_capital=args.capital,
        slippage_pct=args.slippage,
        top_n=args.top_n,
        max_atr_pct=args.max_atr,
        vix_cash_buffer=args.vix_buffer,
    )
    result = engine.run(labeled, prices, selector)

    if result.equity_curve.empty:
        logger.error("Backtest produced no results — check data")
        sys.exit(1)

    # 4. Save trades log
    os.makedirs(REPORTS_DIR, exist_ok=True)
    if not result.trades_log.empty:
        trades_path = os.path.join(REPORTS_DIR, "trades_log.csv")
        result.trades_log.to_csv(trades_path, index=False)
        logger.info("Trades log saved -> %s", trades_path)

    # 5. Metrics + report
    metrics = compute_metrics(result)
    print_report(metrics)

    # 6. Chart
    if not args.no_chart:
        plot_equity_curve(result)

    logger.info("=== BACKTEST COMPLETE ===")


if __name__ == "__main__":
    main()
