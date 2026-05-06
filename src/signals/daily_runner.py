import logging
import pandas as pd

from config.settings import ML_TOP_N
from src.universe.sp500_universe import get_sp500_tickers, download_prices, filter_universe
from src.analysis.macro_scanner import get_macro_snapshot, build_macro_history
from src.signals.feature_engineer import build_feature_matrix, add_labels
from src.signals.ml_selector import MLSelector

logger = logging.getLogger(__name__)

_selector = MLSelector()


def run_daily_selection() -> tuple[list, dict]:
    """
    Called each evening after market close.
    Returns (top_n_symbols, score_dict) where score_dict maps symbol -> ML probability.
    Returns ([], {}) in BEAR regime so the caller closes all positions.
    """
    logger.info("Daily ML selection starting")

    tickers = get_sp500_tickers()
    prices = filter_universe(download_prices(tickers))
    macro = get_macro_snapshot()

    if macro.regime == "BEAR":
        logger.warning("Bear regime detected — returning empty selection (caller will close all positions)")
        return [], {}

    macro_history = build_macro_history(period="1y")
    matrix = build_feature_matrix(prices, macro, options_cache={}, macro_history=macro_history)
    labeled = add_labels(matrix, prices)

    _selector.load_or_train(labeled)

    # Score using only the most recent row per symbol
    latest_date = matrix["date"].max()
    today_features = matrix[matrix["date"] == latest_date].copy()
    today_features = today_features.dropna(subset=["ret_5d"])

    if today_features.empty:
        logger.warning("No features available for today — skipping selection")
        return [], {}

    scores = _selector.score(today_features)
    top_n = _selector.get_top_n(scores, n=ML_TOP_N)
    scores_dict = scores.loc[top_n].to_dict()

    logger.info(
        "Selected %d symbols | top: %s",
        len(top_n),
        ", ".join(f"{s}({v:.2f})" for s, v in list(scores_dict.items())[:5]),
    )
    return top_n, scores_dict


def _momentum_shortlist(prices: dict, n: int = 50) -> list:
    """Rank by 21-day return, return top N."""
    mom = {}
    for sym, df in prices.items():
        if len(df) >= 22:
            ret = df["close"].iloc[-1] / df["close"].iloc[-22] - 1
            mom[sym] = ret
    return sorted(mom, key=mom.get, reverse=True)[:n]
