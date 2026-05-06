import yfinance as yf
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import Literal, Dict

from config.settings import MACRO_SYMBOLS

logger = logging.getLogger(__name__)

MarketRegime = Literal["BULL", "BEAR", "NEUTRAL"]


@dataclass
class MacroSnapshot:
    regime: MarketRegime
    spy_trend: str        # "UP" | "DOWN" | "FLAT"
    qqq_trend: str
    vix_level: float
    vix_signal: str       # "LOW" | "ELEVATED" | "HIGH"
    breadth_score: float  # 0.0–1.0  (fraction of sector ETFs above 20-SMA)
    summary: str


def _trend(closes: pd.Series, fast: int = 20, slow: int = 50) -> str:
    if len(closes) < slow:
        return "FLAT"
    sma20 = closes.rolling(fast).mean().iloc[-1]
    sma50 = closes.rolling(slow).mean().iloc[-1]
    price = closes.iloc[-1]
    if price > sma20 > sma50:
        return "UP"
    if price < sma20 < sma50:
        return "DOWN"
    return "FLAT"


def _vix_signal(vix: float) -> str:
    if vix < 18:
        return "LOW"
    if vix < 28:
        return "ELEVATED"
    return "HIGH"


def _download_macro_prices(period: str = "2y") -> Dict[str, pd.DataFrame]:
    """Download OHLCV for all macro symbols. Returns {ticker: df_with_lowercase_cols}."""
    tickers = MACRO_SYMBOLS  # already has ^VIX
    raw = yf.download(
        " ".join(tickers),
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )
    prices: Dict[str, pd.DataFrame] = {}
    for sym in tickers:
        try:
            df = raw[sym].copy() if len(tickers) > 1 else raw.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            df = df.dropna(subset=["close"])
            df.index = pd.to_datetime(df.index)
            if not df.empty:
                prices[sym] = df
        except Exception:
            pass
    return prices


def build_macro_history(period: str = "2y") -> pd.DataFrame:
    """
    Compute daily macro-regime features using only historically available data
    (no lookahead).  Returns a DataFrame indexed by date.

    Columns match FEATURE_COLS macro fields:
      regime_bull, regime_neutral, regime_bear,
      vix_level, breadth_score, spy_trend_enc, qqq_trend_enc
    """
    mp = _download_macro_prices(period)

    spy_c = mp.get("SPY", pd.DataFrame()).get("close", pd.Series(dtype=float))
    qqq_c = mp.get("QQQ", pd.DataFrame()).get("close", pd.Series(dtype=float))
    vix_c = mp.get("^VIX", pd.DataFrame()).get("close", pd.Series(dtype=float))

    if spy_c.empty:
        logger.warning("build_macro_history: no SPY data")
        return pd.DataFrame()

    dates = spy_c.index

    # ── SPY trend (vectorized) ────────────────────────────────────────────────
    spy_sma20 = spy_c.rolling(20).mean()
    spy_sma50 = spy_c.rolling(50).mean()
    spy_up   = (spy_c > spy_sma20) & (spy_sma20 > spy_sma50)
    spy_down = (spy_c < spy_sma20) & (spy_sma20 < spy_sma50)
    spy_enc  = spy_up.astype(int) - spy_down.astype(int)

    # ── QQQ trend ─────────────────────────────────────────────────────────────
    if not qqq_c.empty:
        qqq_c    = qqq_c.reindex(dates).ffill()
        qqq_sma20 = qqq_c.rolling(20).mean()
        qqq_sma50 = qqq_c.rolling(50).mean()
        qqq_up   = (qqq_c > qqq_sma20) & (qqq_sma20 > qqq_sma50)
        qqq_down = (qqq_c < qqq_sma20) & (qqq_sma20 < qqq_sma50)
        qqq_enc  = qqq_up.astype(int) - qqq_down.astype(int)
    else:
        qqq_up = qqq_down = pd.Series(False, index=dates)
        qqq_enc = pd.Series(0, index=dates)

    # ── VIX ───────────────────────────────────────────────────────────────────
    if not vix_c.empty:
        vix_level = vix_c.reindex(dates).ffill().fillna(20.0)
    else:
        vix_level = pd.Series(20.0, index=dates)
    vix_low  = (vix_level < 18)
    vix_high = (vix_level >= 28)

    # ── Sector breadth ────────────────────────────────────────────────────────
    sector_etfs = ["XLF", "XLK", "XLE", "XLV", "IWM"]
    above_count = pd.Series(0.0, index=dates)
    for etf in sector_etfs:
        df = mp.get(etf, pd.DataFrame())
        if df.empty:
            continue
        s = df["close"].reindex(dates).ffill()
        above_count += (s > s.rolling(20).mean()).fillna(False).astype(float)
    breadth = above_count / len(sector_etfs)
    breadth_high = (breadth >= 0.6)
    breadth_low  = (breadth <= 0.3)

    # ── Regime ────────────────────────────────────────────────────────────────
    bull_sig = spy_up.astype(int) + qqq_up.astype(int) + vix_low.astype(int) + breadth_high.astype(int)
    bear_sig = spy_down.astype(int) + qqq_down.astype(int) + vix_high.astype(int) + breadth_low.astype(int)
    regime_bull    = (bull_sig >= 3).astype(int)
    regime_bear    = (bear_sig >= 3).astype(int)
    regime_neutral = (1 - regime_bull - regime_bear).clip(0, 1)

    history = pd.DataFrame({
        "regime_bull":    regime_bull,
        "regime_neutral": regime_neutral,
        "regime_bear":    regime_bear,
        "vix_level":      vix_level,
        "breadth_score":  breadth,
        "spy_trend_enc":  spy_enc,
        "qqq_trend_enc":  qqq_enc,
    }, index=dates)

    logger.info("Macro history: %d trading days", len(history))
    return history


def get_macro_snapshot() -> MacroSnapshot:
    """Pull macro ETF data and classify the current market regime."""
    tickers = " ".join(MACRO_SYMBOLS)
    raw = yf.download(tickers, period="3mo", interval="1d", auto_adjust=True, progress=False)

    closes = raw["Close"] if "Close" in raw.columns else raw.xs("Close", axis=1, level=0)

    spy_closes = closes["SPY"].dropna()
    qqq_closes = closes["QQQ"].dropna()
    vix_closes = closes.get("^VIX", closes.get("VIX", pd.Series(dtype=float))).dropna()

    spy_trend = _trend(spy_closes)
    qqq_trend = _trend(qqq_closes)
    vix_level = float(vix_closes.iloc[-1])
    vix_sig = _vix_signal(vix_level)

    sector_etfs = ["XLF", "XLK", "XLE", "XLV", "IWM"]
    above_sma = 0
    for etf in sector_etfs:
        if etf in closes.columns:
            s = closes[etf].dropna()
            if len(s) >= 20 and s.iloc[-1] > s.rolling(20).mean().iloc[-1]:
                above_sma += 1
    breadth = above_sma / len(sector_etfs)

    # Regime classification
    bull_signals = sum([
        spy_trend == "UP",
        qqq_trend == "UP",
        vix_sig == "LOW",
        breadth >= 0.6,
    ])
    bear_signals = sum([
        spy_trend == "DOWN",
        qqq_trend == "DOWN",
        vix_sig == "HIGH",
        breadth <= 0.3,
    ])

    if bull_signals >= 3:
        regime: MarketRegime = "BULL"
    elif bear_signals >= 3:
        regime = "BEAR"
    else:
        regime = "NEUTRAL"

    summary = (
        f"Regime: {regime} | SPY={spy_trend} QQQ={qqq_trend} "
        f"VIX={vix_level:.1f}({vix_sig}) Breadth={breadth:.0%}"
    )
    logger.info(summary)

    return MacroSnapshot(
        regime=regime,
        spy_trend=spy_trend,
        qqq_trend=qqq_trend,
        vix_level=vix_level,
        vix_signal=vix_sig,
        breadth_score=breadth,
        summary=summary,
    )
