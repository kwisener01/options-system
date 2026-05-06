import logging
import pandas as pd

from config.settings import ML_LOOKBACK_DAYS
from src.universe.sp500_universe import get_sp500_tickers, download_prices, filter_universe
from src.analysis.macro_scanner import get_macro_snapshot, build_macro_history
from src.signals.feature_engineer import build_feature_matrix, add_labels

logger = logging.getLogger(__name__)


def prepare_backtest_data(lookback_days: int = ML_LOOKBACK_DAYS, max_symbols: int = 0) -> tuple:
    """
    Orchestrate the full data pipeline for a backtest run.
    Returns: (labeled_feature_matrix, prices_dict)

    Macro features are computed historically (no lookahead) so the walk-forward
    backtest sees only regime data that was available at each point in time.
    """
    period = "2y" if lookback_days >= 504 else "1y"

    logger.info("Step 1/4 — fetching S&P 500 universe")
    tickers = get_sp500_tickers()

    if max_symbols > 0 and len(tickers) > max_symbols:
        # Always keep SPY for the benchmark; sample the rest randomly so we get
        # a representative cross-section rather than alphabetically biased picks.
        import random
        others = [t for t in tickers if t != "SPY"]
        random.shuffle(others)
        tickers = ["SPY"] + others[: max_symbols - 1]
        logger.info("Universe capped to %d symbols (max_symbols=%d)", len(tickers), max_symbols)

    logger.info("Step 2/4 — downloading price history (%s)", period)
    prices_raw = download_prices(tickers, period=period)
    prices = filter_universe(prices_raw)

    logger.info("Step 3/4 — building historical macro features (no lookahead)")
    macro_history = build_macro_history(period=period)
    macro_snap = get_macro_snapshot()   # used only for live-trading paths, not backtest

    logger.info("Step 4/4 — building feature matrix")
    matrix = build_feature_matrix(prices, macro_snap, options_cache={}, macro_history=macro_history)
    labeled = add_labels(matrix, prices)

    # Drop rows with no label (last 5 days per symbol — future unknown)
    labeled = labeled.dropna(subset=["label", "ret_5d"])

    logger.info(
        "Data ready: %d rows, %d symbols, %d labeled positive (%.1f%%)",
        len(labeled),
        labeled["symbol"].nunique(),
        int(labeled["label"].sum()),
        labeled["label"].mean() * 100,
    )
    return labeled, prices
