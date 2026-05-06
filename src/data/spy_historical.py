"""
SPY historical dataset builder.

Downloads SPY from 1993-01-29 (inception) plus companion instruments, computes
a comprehensive indicator set, and saves everything to data/spy_features.parquet.

Companion instruments downloaded
─────────────────────────────────────────────────────────────────────────────
Price / index
  ^VIX    Volatility index                  (1990)
  ^VXN    Nasdaq volatility                 (2001)
  ^GSPC   S&P 500 index (for spot ref)      (1927, use SPY dates)
  QQQ     Nasdaq-100 ETF                    (1999)
  IWM     Russell 2000 ETF                  (2000)
Rates / fixed income
  ^TNX    10-year Treasury yield            (1962)
  ^FVX    5-year Treasury yield             (1962)
  ^IRX    3-month Treasury yield            (1960)
  TLT     20-year bond ETF                  (2002)
  LQD     Investment-grade bond ETF         (2002)
  HYG     High-yield bond ETF               (2007)
Sectors (all from 1998)
  XLF XLK XLE XLV XLY XLU XLI XLB XLP
Commodities / FX
  GLD     Gold ETF                          (2004)
  USO     Oil ETF                           (2006)
  UUP     Dollar ETF                        (2007)

Computed feature groups
─────────────────────────────────────────────────────────────────────────────
Returns         1d 2d 5d 10d 21d 63d 126d 252d
Momentum        RSI(7,14,21), Stochastic(%K,%D), Williams%R, ROC, CCI
                MACD(line, signal, hist), PPO
Trend           SMA(5,10,20,50,200), EMA(5,12,26), SMA cross flags
                ADX(14), Aroon(up,down,oscillator), Ichimoku (leading A/B)
                TRIX, DPO, KST, Vortex(+/-)
Volatility      ATR(7,14), BBands(width,%B,upper,lower)
                Keltner channels, Donchian channels, Ulcer index
                Realized vol (5d,21d,63d), Vol of vol (vol ratio)
Volume          OBV, MFI(14), CMF(20), Force Index, VWAP daily proxy
                Volume SMA ratio(5/20), EOM, NVI
Cross-asset     VIX level, VIX change, VIX/RV ratio, Yield curve (10Y-3M, 10Y-2Y proxy)
                SPY/TLT ratio, SPY/GLD ratio, HYG/LQD spread proxy
                Sector relative strength (8 sectors vs SPY)
Calendar        day_of_week, month, quarter, is_month_end, is_quarter_end
Target labels   fwd_ret_1d 2d 5d 10d 21d, fwd_vol_5d, direction_5d (binary)
"""
import logging
import os
from datetime import date

import numpy as np
import pandas as pd
import ta
import yfinance as yf

from config.settings import DATA_DIR

logger = logging.getLogger(__name__)

PARQUET_PATH = os.path.join(DATA_DIR, "spy_features.parquet")
CSV_PATH     = os.path.join(DATA_DIR, "spy_features.csv")

# ── companion download symbols ────────────────────────────────────────────────

_COMPANIONS = {
    "^VIX":  "vix",
    "^VXN":  "vxn",
    "QQQ":   "qqq",
    "IWM":   "iwm",
    "^TNX":  "tnx_10y",
    "^FVX":  "tnx_5y",
    "^IRX":  "irx_3m",
    "TLT":   "tlt",
    "LQD":   "lqd",
    "HYG":   "hyg",
    "XLF":   "xlf",
    "XLK":   "xlk",
    "XLE":   "xle",
    "XLV":   "xlv",
    "XLY":   "xly",
    "XLU":   "xlu",
    "XLI":   "xli",
    "XLB":   "xlb",
    "XLP":   "xlp",
    "GLD":   "gld",
    "USO":   "uso",
    "UUP":   "uup",
}


# ── download helpers ──────────────────────────────────────────────────────────

def _download(symbol: str, period: str = "max") -> pd.DataFrame:
    t  = yf.Ticker(symbol)
    df = t.history(period=period, auto_adjust=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.columns = [c.lower() for c in df.columns]
    return df[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])


def _align(spy_index: pd.DatetimeIndex, df: pd.DataFrame,
           col: str = "close") -> pd.Series:
    """Reindex to SPY trading days, forward-fill up to 5 days."""
    s = df[col].reindex(spy_index).ffill(limit=5)
    return s


# ── feature computation ───────────────────────────────────────────────────────

def _returns(close: pd.Series) -> pd.DataFrame:
    periods = [1, 2, 5, 10, 21, 63, 126, 252]
    return pd.DataFrame(
        {f"ret_{p}d": close.pct_change(p) for p in periods}
    )


def _momentum(close: pd.Series, high: pd.Series, low: pd.Series) -> pd.DataFrame:
    out = {}
    for w in [7, 14, 21]:
        out[f"rsi_{w}"] = ta.momentum.RSIIndicator(close, window=w).rsi()
    stoch = ta.momentum.StochasticOscillator(high, low, close)
    out["stoch_k"]    = stoch.stoch()
    out["stoch_d"]    = stoch.stoch_signal()
    out["williams_r"] = ta.momentum.WilliamsRIndicator(high, low, close).williams_r()
    for w in [10, 21]:
        out[f"roc_{w}d"] = ta.momentum.ROCIndicator(close, window=w).roc()
    out["cci_20"]  = ta.trend.CCIIndicator(high, low, close, window=20).cci()
    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    out["macd"]        = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_hist"]   = macd.macd_diff()
    out["macd_cross_bull"] = ((out["macd_hist"] > 0) & (pd.Series(out["macd_hist"]).shift(1) <= 0)).astype(int)
    out["macd_cross_bear"] = ((out["macd_hist"] < 0) & (pd.Series(out["macd_hist"]).shift(1) >= 0)).astype(int)
    out["ppo"] = ta.momentum.PercentagePriceOscillator(close).ppo()
    out["tsi"] = ta.momentum.TSIIndicator(close).tsi()
    out["ultimate_osc"] = ta.momentum.UltimateOscillator(high, low, close).ultimate_oscillator()
    return pd.DataFrame(out, index=close.index)


def _trend(close: pd.Series, high: pd.Series, low: pd.Series) -> pd.DataFrame:
    out = {}
    for w in [5, 10, 20, 50, 100, 200]:
        sma = close.rolling(w).mean()
        out[f"sma_{w}"]       = sma
        out[f"above_sma_{w}"] = (close > sma).astype(int)
        out[f"dist_sma_{w}"]  = (close - sma) / sma
    for span in [5, 12, 26]:
        out[f"ema_{span}"] = close.ewm(span=span, adjust=False).mean()
    # SMA cross signals
    out["sma20_x_sma50_bull"] = (
        (close.rolling(20).mean() > close.rolling(50).mean()) &
        (close.rolling(20).mean().shift(1) <= close.rolling(50).mean().shift(1))
    ).astype(int)
    out["sma20_slope"] = close.rolling(20).mean().pct_change(5)
    out["sma50_slope"] = close.rolling(50).mean().pct_change(10)
    adx = ta.trend.ADXIndicator(high, low, close, window=14)
    out["adx"]    = adx.adx()
    out["adx_pos"] = adx.adx_pos()
    out["adx_neg"] = adx.adx_neg()
    aroon = ta.trend.AroonIndicator(high, low, window=25)
    out["aroon_up"]   = aroon.aroon_up()
    out["aroon_down"] = aroon.aroon_down()
    out["aroon_osc"]  = aroon.aroon_indicator()
    out["trix_15"]    = ta.trend.TRIXIndicator(close, window=15).trix()
    out["dpo_20"]     = ta.trend.DPOIndicator(close, window=20).dpo()
    out["kst"]        = ta.trend.KSTIndicator(close).kst()
    vi = ta.trend.VortexIndicator(high, low, close, window=14)
    out["vortex_pos"] = vi.vortex_indicator_pos()
    out["vortex_neg"] = vi.vortex_indicator_neg()
    out["dist_52w_high"] = close / close.rolling(252, min_periods=126).max() - 1
    out["dist_52w_low"]  = close / close.rolling(252, min_periods=126).min() - 1
    return pd.DataFrame(out, index=close.index)


def _volatility(close: pd.Series, high: pd.Series, low: pd.Series) -> pd.DataFrame:
    out = {}
    for w in [7, 14, 21]:
        out[f"atr_{w}"]     = ta.volatility.AverageTrueRange(high, low, close, window=w).average_true_range()
        out[f"atr_{w}_pct"] = out[f"atr_{w}"] / close
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    out["bb_upper"]  = bb.bollinger_hband()
    out["bb_lower"]  = bb.bollinger_lband()
    out["bb_width"]  = bb.bollinger_wband()
    out["bb_pct"]    = bb.bollinger_pband()
    kc = ta.volatility.KeltnerChannel(high, low, close, window=20)
    out["kc_upper"]  = kc.keltner_channel_hband()
    out["kc_lower"]  = kc.keltner_channel_lband()
    out["kc_width"]  = kc.keltner_channel_wband()
    out["kc_pct"]    = kc.keltner_channel_pband()
    dc = ta.volatility.DonchianChannel(high, low, close, window=20)
    out["dc_upper"]  = dc.donchian_channel_hband()
    out["dc_lower"]  = dc.donchian_channel_lband()
    out["dc_width"]  = dc.donchian_channel_wband()
    out["ulcer"]     = ta.volatility.UlcerIndex(close, window=14).ulcer_index()
    # Realized vol at multiple windows
    log_ret = np.log(close / close.shift(1))
    for w in [5, 10, 21, 63]:
        out[f"rvol_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252)
    out["rvol_ratio_5_21"]  = out["rvol_5d"]  / out["rvol_21d"].replace(0, np.nan)
    out["rvol_ratio_21_63"] = out["rvol_21d"] / out["rvol_63d"].replace(0, np.nan)
    return pd.DataFrame(out, index=close.index)


def _volume(close: pd.Series, high: pd.Series, low: pd.Series,
            volume: pd.Series) -> pd.DataFrame:
    out = {}
    out["obv"]         = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    out["mfi_14"]      = ta.volume.MFIIndicator(high, low, close, volume, window=14).money_flow_index()
    out["cmf_20"]      = ta.volume.ChaikinMoneyFlowIndicator(high, low, close, volume, window=20).chaikin_money_flow()
    out["force_idx"]   = ta.volume.ForceIndexIndicator(close, volume, window=13).force_index()
    out["eom"]         = ta.volume.EaseOfMovementIndicator(high, low, volume, window=14).sma_ease_of_movement()
    out["vpt"]         = ta.volume.VolumePriceTrendIndicator(close, volume).volume_price_trend()
    out["nvi"]         = ta.volume.NegativeVolumeIndexIndicator(close, volume).negative_volume_index()
    out["vol_sma5"]    = volume.rolling(5).mean()
    out["vol_sma20"]   = volume.rolling(20).mean()
    out["vol_ratio_5_20"]  = out["vol_sma5"]  / out["vol_sma20"].replace(0, np.nan)
    out["vol_ratio_1_5"]   = volume           / out["vol_sma5"].replace(0, np.nan)
    out["vwap_proxy"]  = (close * volume).rolling(20).sum() / volume.rolling(20).sum()
    return pd.DataFrame(out, index=close.index)


def _cross_asset(spy_idx: pd.DatetimeIndex, companions: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=spy_idx)
    spy_close = companions.get("spy_close")

    # Rates
    for name in ["vix", "vxn", "tnx_10y", "tnx_5y", "irx_3m"]:
        if name in companions:
            out[name] = companions[name]

    if "vix" in companions:
        out["vix_chg_1d"]  = companions["vix"].pct_change(1)
        out["vix_chg_5d"]  = companions["vix"].pct_change(5)
        out["vix_ma20"]    = companions["vix"].rolling(20).mean()
        out["vix_pct_ma"]  = companions["vix"] / out["vix_ma20"] - 1
        if "rvol_21d" in companions:
            out["vix_rv_ratio"] = companions["vix"] / (companions["rvol_21d"] * 100)

    # Yield curve
    if "tnx_10y" in companions and "irx_3m" in companions:
        out["yield_curve_10y_3m"] = companions["tnx_10y"] - companions["irx_3m"]
        out["yield_curve_inverted"] = (out["yield_curve_10y_3m"] < 0).astype(int)
    if "tnx_10y" in companions and "tnx_5y" in companions:
        out["yield_curve_10y_5y"] = companions["tnx_10y"] - companions["tnx_5y"]

    # Bond/credit spreads
    if spy_close is not None and "tlt" in companions:
        out["spy_tlt_ratio"]     = spy_close / companions["tlt"].replace(0, np.nan)
        out["spy_tlt_ratio_chg"] = out["spy_tlt_ratio"].pct_change(5)
    if "hyg" in companions and "lqd" in companions:
        out["hyg_lqd_ratio"] = companions["hyg"] / companions["lqd"].replace(0, np.nan)
        out["credit_spread_proxy"] = out["hyg_lqd_ratio"].pct_change(5)

    # SPY vs commodities / dollar
    if spy_close is not None:
        for name in ["gld", "uso", "uup"]:
            if name in companions:
                out[f"spy_{name}_ratio"] = spy_close / companions[name].replace(0, np.nan)

    # Relative strength vs sectors
    if spy_close is not None:
        for etf in ["xlf", "xlk", "xle", "xlv", "xly", "xlu", "xli", "xlb", "xlp"]:
            if etf in companions:
                rs = companions[etf] / companions[etf].rolling(20).mean()
                out[f"rs_{etf}"] = rs - (spy_close / spy_close.rolling(20).mean())

    # QQQ / IWM vs SPY (growth vs value, large vs small cap)
    if spy_close is not None:
        for name in ["qqq", "iwm"]:
            if name in companions:
                out[f"spy_{name}_rs"] = (
                    companions[name].pct_change(21) - spy_close.pct_change(21)
                )

    return out


def _calendar(idx: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({
        "day_of_week":     idx.dayofweek,
        "month":           idx.month,
        "quarter":         idx.quarter,
        "is_month_end":    idx.is_month_end.astype(int),
        "is_quarter_end":  idx.is_quarter_end.astype(int),
        "week_of_year":    idx.isocalendar().week.values,
    }, index=idx)


def _labels(close: pd.Series) -> pd.DataFrame:
    out = {}
    for p in [1, 2, 5, 10, 21]:
        ret = close.pct_change(p).shift(-p)
        out[f"fwd_ret_{p}d"]       = ret
        out[f"fwd_direction_{p}d"] = (ret > 0).astype(int)
    log_ret = np.log(close / close.shift(1))
    out["fwd_rvol_5d"] = log_ret.shift(-5).rolling(5).std() * np.sqrt(252)
    return pd.DataFrame(out, index=close.index)


# ── main pipeline ─────────────────────────────────────────────────────────────

def build_dataset(save: bool = True) -> pd.DataFrame:
    """
    Download and build the full SPY feature dataset.
    Returns a DataFrame with ~150 columns indexed by date.
    If save=True, writes spy_features.parquet (fast) and spy_features.csv.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    logger.info("Downloading SPY (max history)...")
    spy = _download("SPY")
    logger.info("SPY: %d rows from %s to %s", len(spy), spy.index.min().date(), spy.index.max().date())

    idx = spy.index  # use SPY trading calendar as the master index

    # Download companions
    companion_data: dict = {"spy_close": spy["close"]}
    logger.info("Downloading %d companion instruments...", len(_COMPANIONS))
    for symbol, name in _COMPANIONS.items():
        try:
            df = _download(symbol)
            companion_data[name] = _align(idx, df)
            logger.info("  %-10s %s  (%d rows aligned)", symbol, name, df[~df["close"].isna()].shape[0])
        except Exception as e:
            logger.warning("  %-10s FAILED: %s", symbol, e)

    # Add realized vol to companions for VIX/RV ratio
    log_ret = np.log(spy["close"] / spy["close"].shift(1))
    companion_data["rvol_21d"] = log_ret.rolling(21).std() * np.sqrt(252) * 100

    # Compute feature groups
    logger.info("Computing technical indicators...")
    parts = [
        _returns(spy["close"]),
        _momentum(spy["close"], spy["high"], spy["low"]),
        _trend(spy["close"], spy["high"], spy["low"]),
        _volatility(spy["close"], spy["high"], spy["low"]),
        _volume(spy["close"], spy["high"], spy["low"], spy["volume"]),
        _cross_asset(idx, companion_data),
        _calendar(idx),
        _labels(spy["close"]),
    ]

    df = spy[["open", "high", "low", "close", "volume"]].copy()
    for part in parts:
        part.index = pd.to_datetime(part.index).tz_localize(None)
        df = df.join(part, how="left")

    df.index.name = "date"
    df = df.sort_index()

    n_features = len(df.columns) - 5  # exclude OHLCV
    n_complete = df.dropna(subset=["sma_200", "rsi_14"]).shape[0]
    logger.info("Dataset: %d rows x %d features (%d fully-warm rows after indicator lookback)",
                len(df), n_features, n_complete)

    if save:
        df.to_parquet(PARQUET_PATH)
        logger.info("Saved: %s", PARQUET_PATH)
        df.to_csv(CSV_PATH)
        logger.info("Saved: %s", CSV_PATH)

    return df


def load_dataset() -> pd.DataFrame:
    """Load the cached parquet, rebuilding if not found."""
    if os.path.exists(PARQUET_PATH):
        df = pd.read_parquet(PARQUET_PATH)
        logger.info("Loaded %s (%d rows x %d cols)", PARQUET_PATH, len(df), len(df.columns))
        return df
    logger.info("No cached dataset found — building now...")
    return build_dataset()


def feature_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Return a catalog of all feature columns with type and non-null count."""
    rows = []
    groups = {
        "price":       ["open","high","low","close","volume"],
        "returns":     [c for c in df.columns if c.startswith("ret_")],
        "momentum":    [c for c in df.columns if any(c.startswith(p) for p in
                        ["rsi_","stoch","williams","roc_","cci_","macd","ppo","tsi","ultimate"])],
        "trend":       [c for c in df.columns if any(c.startswith(p) for p in
                        ["sma_","ema_","above_sma","dist_sma","adx","aroon","trix","dpo","kst",
                         "vortex","dist_52w","sma20_","sma50_"])],
        "volatility":  [c for c in df.columns if any(c.startswith(p) for p in
                        ["atr_","bb_","kc_","dc_","ulcer","rvol"])],
        "volume":      [c for c in df.columns if any(c.startswith(p) for p in
                        ["obv","mfi","cmf","force","eom","vpt","nvi","vol_","vwap"])],
        "cross_asset": [c for c in df.columns if any(c.startswith(p) for p in
                        ["vix","vxn","tnx","irx","yield","spy_tlt","hyg","credit","spy_gld",
                         "spy_uso","spy_uup","rs_","spy_qqq","spy_iwm"])],
        "calendar":    [c for c in df.columns if c in
                        ["day_of_week","month","quarter","is_month_end","is_quarter_end","week_of_year"]],
        "labels":      [c for c in df.columns if c.startswith("fwd_")],
    }
    assigned = set()
    for group, cols in groups.items():
        for col in cols:
            if col in df.columns and col not in assigned:
                assigned.add(col)
                rows.append({
                    "group":    group,
                    "feature":  col,
                    "non_null": int(df[col].notna().sum()),
                    "from_date": str(df[col].first_valid_index().date()) if df[col].notna().any() else "n/a",
                    "dtype":    str(df[col].dtype),
                })
    unassigned = [c for c in df.columns if c not in assigned]
    for col in unassigned:
        rows.append({"group": "other", "feature": col, "non_null": int(df[col].notna().sum()),
                     "from_date": "n/a", "dtype": str(df[col].dtype)})
    return pd.DataFrame(rows).sort_values(["group","feature"])
