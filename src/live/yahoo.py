"""
SSL-resilient Yahoo daily-OHLCV fetch.

`yfinance.download()` fails in some environments (SSL handshake / blocked egress),
which is exactly what broke the backtests. This hits Yahoo's chart API directly
with the same verify-fallback the live scanners use (certifi → system → disabled),
returning a pandas DataFrame with lowercase OHLCV columns indexed by date — a
drop-in replacement for `yf.download(...)` for daily bars.
"""
import datetime as _dt
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def fetch_daily_ohlcv(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Daily bars for `ticker` over `period` (e.g. '2y','1y','6mo','3mo','max').
    Returns columns open/high/low/close/volume (close = adjusted). Empty df on
    failure (never raises)."""
    import requests
    sym = ticker.replace("^", "%5E")          # ^VIX → %5EVIX
    url = (f"https://query2.finance.yahoo.com/v8/finance/chart/{sym}"
           f"?range={period}&interval=1d&events=div%2Csplit")
    data = None
    for verify in (__import__("certifi").where(), True, False):
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"},
                             timeout=20, verify=verify)
            r.raise_for_status()
            data = r.json()
            break
        except requests.exceptions.SSLError:
            continue
        except Exception as e:
            logger.debug("yahoo fetch %s failed (verify=%s): %s", ticker, verify, e)
            continue
    if not data:
        logger.warning("yahoo fetch failed for %s", ticker)
        return pd.DataFrame()
    try:
        res = data["chart"]["result"][0]
        ts  = res.get("timestamp") or []
        q   = res["indicators"]["quote"][0]
        adj = (res["indicators"].get("adjclose") or [{}])[0].get("adjclose")
        idx = pd.to_datetime([_dt.datetime.utcfromtimestamp(t) for t in ts])
        df = pd.DataFrame({
            "open":   q.get("open"),
            "high":   q.get("high"),
            "low":    q.get("low"),
            "close":  adj if adj else q.get("close"),
            "volume": q.get("volume"),
        }, index=idx)
        return df.dropna(subset=["close"])
    except Exception as e:
        logger.warning("yahoo parse failed for %s: %s", ticker, e)
        return pd.DataFrame()


def download_many(tickers: list[str], period: str = "2y") -> dict[str, pd.DataFrame]:
    """{ticker: df} via the resilient fetch, run in parallel."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    out: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(fetch_daily_ohlcv, t, period): t for t in tickers}
        for f in as_completed(futs):
            t = futs[f]
            try:
                df = f.result()
                if not df.empty:
                    out[t] = df
            except Exception:
                pass
    return out
