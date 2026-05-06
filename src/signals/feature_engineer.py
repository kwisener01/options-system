import numpy as np
import pandas as pd
import logging
from typing import Optional

import ta

from config.settings import ML_TARGET_RETURN_PCT
from src.analysis.macro_scanner import MacroSnapshot
from src.analysis.options_analyzer import OptionsSignal

_MACRO_COLS = [
    "regime_bull", "regime_neutral", "regime_bear",
    "vix_level", "breadth_score", "spy_trend_enc", "qqq_trend_enc",
]

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d",
    "vol_ratio_5_20", "vol_ratio_1_5",
    "rsi_14",
    "macd_cross",
    "bb_pct",
    "atr_pct",
    "dist_52w_high",
    "above_sma20", "above_sma50",
    "sma20_slope",
    # macro
    "regime_bull", "regime_neutral", "regime_bear",
    "vix_level", "breadth_score",
    "spy_trend_enc", "qqq_trend_enc",
    # options
    "put_call_ratio", "dist_max_pain_pct", "implied_move_pct", "options_bias_enc",
]


def _trend_enc(trend: str) -> int:
    return {"UP": 1, "FLAT": 0, "DOWN": -1}.get(trend, 0)


def _bias_enc(bias: str) -> int:
    return {"BULLISH": 1, "NEUTRAL": 0, "BEARISH": -1}.get(bias, 0)


def encode_macro(snap: MacroSnapshot) -> dict:
    return {
        "regime_bull": int(snap.regime == "BULL"),
        "regime_neutral": int(snap.regime == "NEUTRAL"),
        "regime_bear": int(snap.regime == "BEAR"),
        "vix_level": snap.vix_level,
        "breadth_score": snap.breadth_score,
        "spy_trend_enc": _trend_enc(snap.spy_trend),
        "qqq_trend_enc": _trend_enc(snap.qqq_trend),
    }


def encode_options(sig: Optional[OptionsSignal]) -> dict:
    if sig is None:
        return {"put_call_ratio": np.nan, "dist_max_pain_pct": np.nan,
                "implied_move_pct": np.nan, "options_bias_enc": 0}
    return {
        "put_call_ratio": sig.put_call_ratio,
        "dist_max_pain_pct": sig.distance_to_max_pain_pct,
        "implied_move_pct": sig.implied_move_pct,
        "options_bias_enc": _bias_enc(sig.bias),
    }


def compute_stock_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute technical features for a single stock's OHLCV dataframe."""
    df = ohlcv.copy()
    close = df["close"]
    volume = df["volume"]

    feat = pd.DataFrame(index=df.index)

    # Momentum
    feat["ret_5d"] = close.pct_change(5)
    feat["ret_10d"] = close.pct_change(10)
    feat["ret_21d"] = close.pct_change(21)
    feat["ret_63d"] = close.pct_change(63)

    # Volume surge
    feat["vol_ratio_5_20"] = volume.rolling(5).mean() / volume.rolling(20).mean()
    feat["vol_ratio_1_5"] = volume / volume.rolling(5).mean()

    # RSI
    feat["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    # MACD cross (signal line cross in last 3 days)
    macd_obj = ta.trend.MACD(close)
    macd_diff = macd_obj.macd_diff()
    feat["macd_cross"] = (
        (macd_diff > 0) & (macd_diff.shift(1) <= 0)
    ).rolling(3).max().fillna(0).astype(int)

    # Bollinger Band %B
    bb = ta.volatility.BollingerBands(close, window=20)
    feat["bb_pct"] = bb.bollinger_pband()

    # ATR as % of price
    atr = ta.volatility.AverageTrueRange(df["high"], df["low"], close, window=14).average_true_range()
    feat["atr_pct"] = atr / close

    # Distance from 52-week high
    feat["dist_52w_high"] = close / close.rolling(252, min_periods=126).max() - 1

    # Trend indicators
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    feat["above_sma20"] = (close > sma20).astype(int)
    feat["above_sma50"] = (close > sma50).astype(int)
    feat["sma20_slope"] = sma20.pct_change(5)

    return feat


def build_feature_matrix(
    prices: dict,
    macro_snap: MacroSnapshot,
    options_cache: dict,
    macro_history: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build a flat feature matrix across all symbols.

    macro_history: if provided (DataFrame indexed by date), merge macro features
    per-date so the backtest has no lookahead in regime signals.  When None,
    the current macro_snap is broadcast to all rows (live-trading mode).

    options_cache: {symbol: OptionsSignal | None}
    """
    current_macro = encode_macro(macro_snap)
    rows = []

    for sym, ohlcv in prices.items():
        try:
            tech = compute_stock_features(ohlcv)
            opts = encode_options(options_cache.get(sym))

            combined = tech.copy()
            for k, v in opts.items():
                combined[k] = v

            # Macro: broadcast snapshot (overridden below when history exists)
            for k, v in current_macro.items():
                combined[k] = v

            combined["symbol"] = sym
            combined = combined.reset_index().rename(
                columns={"index": "date", "Date": "date", "Datetime": "date"}
            )
            combined["date"] = pd.to_datetime(combined["date"])
            rows.append(combined)
        except Exception as e:
            logger.warning("Feature build failed for %s: %s", sym, e)

    if not rows:
        return pd.DataFrame()

    matrix = pd.concat(rows, ignore_index=True)

    # Overwrite macro columns with date-accurate history (no lookahead)
    if macro_history is not None and not macro_history.empty:
        hist = macro_history.copy()
        hist.index.name = "date"
        hist = hist.reset_index()
        hist["date"] = pd.to_datetime(hist["date"])
        available_cols = [c for c in _MACRO_COLS if c in hist.columns]
        matrix = matrix.drop(columns=[c for c in available_cols if c in matrix.columns], errors="ignore")
        matrix = matrix.merge(hist[["date"] + available_cols], on="date", how="left")
        # Forward-fill any trading days missing from macro history
        matrix = matrix.sort_values(["symbol", "date"])
        for col in available_cols:
            matrix[col] = matrix[col].ffill()

    # Fill missing options features with per-day universe median
    for col in ["put_call_ratio", "dist_max_pain_pct", "implied_move_pct"]:
        if col in matrix.columns:
            daily_median = matrix.groupby("date")[col].transform("median")
            matrix[col] = matrix[col].fillna(daily_median)

    return matrix


def add_labels(matrix: pd.DataFrame, prices: dict) -> pd.DataFrame:
    """Attach 5-day forward return and binary label. Must be called after build_feature_matrix."""
    label_rows = []
    for sym, ohlcv in prices.items():
        fwd = ohlcv["close"].pct_change(5).shift(-5)
        sym_df = pd.DataFrame({"date": ohlcv.index, "fwd_return_5d": fwd.values})
        sym_df["symbol"] = sym
        sym_df["date"] = pd.to_datetime(sym_df["date"])
        label_rows.append(sym_df)

    labels = pd.concat(label_rows, ignore_index=True)
    out = matrix.merge(labels, on=["symbol", "date"], how="left")
    out["label"] = (out["fwd_return_5d"] > ML_TARGET_RETURN_PCT).astype(int)
    return out
