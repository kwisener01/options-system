import os
import time
import pickle
import logging
import datetime
import pandas as pd
import yfinance as yf

from config.settings import SP500_CACHE_PATH, PRICE_CACHE_PATH, MIN_STOCK_PRICE, DATA_DIR, \
    ALPACA_API_KEY, ALPACA_SECRET_KEY, IS_PAPER

logger = logging.getLogger(__name__)

_CACHE_AGE_HOURS = 23
_TICKER_REFRESH_DAYS = 7
_BATCH_SIZE = 50
_BATCH_SLEEP = 2.0


def get_sp500_tickers() -> list:
    """
    Return a tradable universe of US equities.  Fetch order:
      1. Local cache (refreshed weekly)
      2. Wikipedia S&P 500 list
      3. Alpaca asset list (fractionable US equities — best proxy for large/mid-caps)
      4. Hardcoded 50-stock fallback
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # ── Cache hit ─────────────────────────────────────────────────────────────
    if os.path.exists(SP500_CACHE_PATH):
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(SP500_CACHE_PATH))
        if (datetime.datetime.now() - mtime).days < _TICKER_REFRESH_DAYS:
            df = pd.read_csv(SP500_CACHE_PATH)
            tickers = df["Symbol"].tolist()
            logger.info("Universe loaded from cache (%d symbols)", len(tickers))
            return tickers

    # ── Wikipedia ─────────────────────────────────────────────────────────────
    logger.info("Fetching S&P 500 tickers from Wikipedia")
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0][["Symbol", "Security"]].copy()
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
        df.to_csv(SP500_CACHE_PATH, index=False)
        logger.info("Cached %d tickers from Wikipedia", len(df))
        return df["Symbol"].tolist()
    except Exception as e:
        logger.warning("Wikipedia failed (%s) — trying Alpaca asset list", e)

    # ── Alpaca asset list ─────────────────────────────────────────────────────
    tickers = _get_tickers_from_alpaca()
    if tickers:
        df = pd.DataFrame({"Symbol": tickers, "Security": ""})
        df.to_csv(SP500_CACHE_PATH, index=False)
        logger.info("Cached %d tickers from Alpaca", len(tickers))
        return tickers

    # ── Hardcoded fallback ────────────────────────────────────────────────────
    logger.warning("All ticker sources failed — using 50-stock fallback list")
    return _FALLBACK_TICKERS


def _get_tickers_from_alpaca() -> list:
    """
    Pull all active, fractionable US equity assets from Alpaca.
    Fractionable stocks are liquid large/mid-caps — a solid ML universe.
    """
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetAssetsRequest
        from alpaca.trading.enums import AssetClass, AssetStatus

        client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=IS_PAPER)
        req = GetAssetsRequest(
            asset_class=AssetClass.US_EQUITY,
            status=AssetStatus.ACTIVE,
        )
        assets = client.get_all_assets(req)

        tickers = []
        for asset in assets:
            if (
                asset.fractionable
                and asset.tradable
                and asset.shortable
                and "." not in asset.symbol   # skip BRK.B style (yfinance uses BRK-B)
                and len(asset.symbol) <= 5
            ):
                tickers.append(asset.symbol)

        # Always include SPY for benchmark
        if "SPY" not in tickers:
            tickers.insert(0, "SPY")

        logger.info("Alpaca returned %d fractionable US equities", len(tickers))
        return sorted(tickers)

    except Exception as e:
        logger.error("Alpaca asset fetch failed: %s", e)
        return []


def download_prices(tickers: list, period: str = "2y") -> dict:
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(PRICE_CACHE_PATH):
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(PRICE_CACHE_PATH))
        age_hours = (datetime.datetime.now() - mtime).total_seconds() / 3600
        if age_hours < _CACHE_AGE_HOURS:
            with open(PRICE_CACHE_PATH, "rb") as f:
                cached = pickle.load(f)
            logger.info("Price cache hit (%d symbols, %.1f hours old)", len(cached), age_hours)
            return cached

    logger.info("Downloading prices for %d symbols in batches of %d", len(tickers), _BATCH_SIZE)
    prices = {}

    for i in range(0, len(tickers), _BATCH_SIZE):
        batch = tickers[i : i + _BATCH_SIZE]
        try:
            raw = yf.download(
                " ".join(batch),
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                group_by="ticker",
            )
            for sym in batch:
                try:
                    if len(batch) == 1:
                        df = raw.copy()
                    else:
                        df = raw[sym].copy()
                    df.columns = [c.lower() for c in df.columns]
                    df = df.dropna(subset=["close"])
                    if len(df) >= 200:
                        prices[sym] = df
                except Exception:
                    pass
        except Exception as e:
            logger.warning("Batch %d-%d failed: %s", i, i + _BATCH_SIZE, e)

        if i + _BATCH_SIZE < len(tickers):
            time.sleep(_BATCH_SLEEP)

    logger.info("Downloaded %d/%d symbols with sufficient history", len(prices), len(tickers))

    with open(PRICE_CACHE_PATH, "wb") as f:
        pickle.dump(prices, f)

    return prices


def filter_universe(prices: dict) -> dict:
    """Remove penny stocks and symbols with too little history."""
    filtered = {
        sym: df
        for sym, df in prices.items()
        if float(df["close"].iloc[-1]) >= MIN_STOCK_PRICE and len(df) >= 200
    }
    logger.info("Universe filtered: %d -> %d symbols", len(prices), len(filtered))
    return filtered


_FALLBACK_TICKERS = [
    "SPY",  # always include benchmark
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "UNH", "JPM", "V",
    "XOM", "LLY", "JNJ", "PG", "MA", "AVGO", "HD", "CVX", "MRK", "ABBV",
    "COST", "PEP", "KO", "WMT", "BAC", "MCD", "TMO", "CSCO", "ACN", "ABT",
    "CRM", "LIN", "NFLX", "AMD", "DIS", "TXN", "PM", "CMCSA", "VZ", "NEE",
    "RTX", "HON", "INTU", "QCOM", "AMGN", "IBM", "GE", "CAT", "SPGI", "BLK",
]
