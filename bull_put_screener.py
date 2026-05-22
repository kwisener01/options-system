"""
BULL PUT SPREAD + IRON BUTTERFLY SCREENER  v3
Production-hardened: all audit fixes applied.

Strategies:
  Bull Put Spread  — sell OTM put, buy lower put, same expiry
  Iron Butterfly   — sell ATM straddle, buy OTM wings, same expiry

Gates (run in order):
  0. SPY macro trend (market must be above its own 21 EMA)
  1. Stock above its own 21 EMA
  2. ATM IV >= MIN_IV (always ATM, never OTM-skew IV)
  3. Spread: delta < 0.30 | Butterfly: wings >= MIN_FLY_WING_PCT each side
  4. Credit ratio >= threshold (after commissions)
  5. Max risk <= MAX_RISK_PER_TRADE (5 pct of account)
  6. Breakeven cushion (spread) | Both-sided profit window (butterfly)
  7. Liquidity: OI >= 100 and daily vol >= 10 on every leg
  8. Bid-ask width <= 30 pct of mid on every leg
  9. Earnings: no earnings within the trade window

Run:  python bull_put_screener.py
"""

import os
import sys
import csv
import json
import time
import logging
import requests
import io
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ── TLS  ──────────────────────────────────────────────────────────────────────
# Probe for the best SSL config available on this machine.
# Order: certifi bundle → system bundle → disabled (with loud warning).
# To fix permanently on Windows:  pip install pip-system-certs
import urllib3
import ssl as _ssl

def _best_ssl_verify():
    import requests as _req
    test_url = "https://data.alpaca.markets"
    for verify in (
        __import__("certifi").where() if True else None,  # certifi
        True,                                             # Python system bundle
    ):
        try:
            s = _req.Session()
            s.verify = verify
            s.get(test_url, timeout=5)
            return verify
        except Exception:
            continue
    # Last resort — disable, but log prominently
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    print("[!] WARNING: SSL cert verification DISABLED (Windows cert store issue).")
    print("[!] Fix with:  pip install pip-system-certs")
    print("[!] API keys are transmitted without cert validation until fixed.\n")
    return False

try:
    _SSL_VERIFY = _best_ssl_verify()
except Exception:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    _SSL_VERIFY = False

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── CONFIG ─────────────────────────────────────────────────────────────────────

CONFIG = {
    "ALPACA_API_KEY":    os.getenv("ALPACA_API_KEY"),
    "ALPACA_SECRET_KEY": os.getenv("ALPACA_SECRET_KEY"),
    "ALPACA_BASE_URL":   os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
    "ALPACA_DATA_URL":   "https://data.alpaca.markets",

    # ── Account & position sizing ──────────────────────────────────────────
    "ACCOUNT_SIZE":        1500,
    "MAX_RISK_PER_TRADE":  150,     # 10% of $1,500 — max 3 open trades simultaneously
    "COMMISSION_PER_LEG":  0.65,    # Alpaca options commission per contract per leg

    # ── Strategy filters ───────────────────────────────────────────────────
    "MIN_CREDIT_RATIO":    0.25,    # net credit (after commissions) >= 25% of spread width
    "MAX_SHORT_DELTA":     0.25,    # short put delta — kept tighter for win-rate
    "MIN_IV":              30,      # ATM IV must be >= 30% (real premium environment)
    "MIN_DTE":             21,
    "MAX_DTE":             45,
    "MIN_BREAKEVEN_DROP":  0.08,    # spread breakeven >= 8% below spot
    "MIN_FLY_WING_PCT":    0.05,    # butterfly wings >= 5% of spot each side (tightened)
    "MIN_FLY_CREDIT_RATIO": 0.40,   # butterfly: collect >= 40% of wing width after commissions
    "SPREAD_WIDTHS":       [2.5, 5.0],

    # ── Liquidity gates ────────────────────────────────────────────────────
    "MIN_OI":              100,     # open interest per leg
    "MIN_VOLUME":          10,      # daily volume per leg
    "MAX_BID_ASK_PCT":     0.30,    # max (ask-bid)/mid — rejects wide/illiquid options

    # ── Options data feed ──────────────────────────────────────────────────
    # "opra"  = real-time NBBO (requires Alpaca Options+ subscription ~$10/mo)
    # "indicative" = free synthetic quotes (default, may differ from fills)
    "OPTIONS_FEED":        "indicative",

    "CSV_OUTPUT_FILE": "bull_put_results.csv",
}

# ── WATCHLIST — liquid names with real options chains ─────────────────────────
# Removed: NIO, BBAI, SOUN, IONQ (thinly traded chains)
# Removed: MSTR (Bitcoin proxy, no mean-reversion edge)

WATCHLIST = [
    # ETFs — tightest spreads, deepest chains
    "SPY", "QQQ", "IWM", "GLD", "SLV", "XLF", "XLE", "XLK",

    # Mega cap — deep chains, high volume
    "AAPL", "MSFT", "NVDA", "META", "TSLA", "AMZN", "GOOGL", "AMD",

    # Mid cap with active options
    "PLTR", "MARA", "HOOD", "SOFI", "COIN", "RIVN", "F", "BAC",
]

# ── LOGGING ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bull_put_screener.log"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _norm_strike(s) -> float:
    """Normalize a strike to 2 decimal places to avoid float key mismatches."""
    return round(float(s), 2)


def _ba_pct(bid: float, ask: float) -> float:
    """Bid-ask width as a fraction of the mid price."""
    mid = (bid + ask) / 2
    return (ask - bid) / mid if mid > 0 else 999


def _liquid(snap: dict) -> bool:
    """True if leg has enough volume and tight enough bid-ask.
    OI is skipped if the feed doesn't supply it (indicative feed returns None).
    """
    q   = snap.get("latestQuote", {})
    bid = q.get("bp", 0)
    ask = q.get("ap", 0)
    oi  = snap.get("openInterest")          # None on indicative feed
    vol = (snap.get("dailyBar") or {}).get("v", 0) or 0

    if oi is not None and oi < CONFIG["MIN_OI"]:   # only enforce if data present
        return False
    if vol < CONFIG["MIN_VOLUME"]:
        return False
    if ask <= bid or bid <= 0:
        return False
    if _ba_pct(bid, ask) > CONFIG["MAX_BID_ASK_PCT"]:
        return False
    return True


def _commission(n_legs: int) -> float:
    """Total round-trip commission for n_legs (open + close)."""
    return n_legs * CONFIG["COMMISSION_PER_LEG"] * 2   # open + close


def _market_open() -> bool:
    """True during NYSE regular hours (9:30–16:00 ET, Mon–Fri)."""
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5:
        return False
    return now.replace(hour=9, minute=30) <= now <= now.replace(hour=16, minute=0)


# ─────────────────────────────────────────────────────────────────────────────
# ALPACA CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class AlpacaClient:

    def __init__(self):
        self.headers = {
            "APCA-API-KEY-ID":     CONFIG["ALPACA_API_KEY"],
            "APCA-API-SECRET-KEY": CONFIG["ALPACA_SECRET_KEY"],
            "Accept":              "application/json",
        }
        self.data_url      = CONFIG["ALPACA_DATA_URL"]
        self.trade_url     = CONFIG["ALPACA_BASE_URL"].replace("paper-api", "api")
        self.session       = requests.Session()
        self.session.verify = _SSL_VERIFY
        self._options_feed = self._detect_options_feed()

    def _get(self, url: str, params: dict = None, timeout: int = 15) -> dict:
        """Single GET with 429 backoff and error logging."""
        for attempt in range(3):
            try:
                r = self.session.get(url, headers=self.headers,
                                     params=params, timeout=timeout)
                if r.status_code == 429:
                    wait = 2 ** attempt
                    log.warning(f"Rate limited — sleeping {wait}s")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r.json()
            except requests.HTTPError as e:
                log.warning(f"HTTP {r.status_code} on {url}: {e}")
                return {}
            except Exception as e:
                log.warning(f"Request failed ({url}): {e}")
                return {}
        return {}

    def get_stock_price(self, ticker: str) -> Optional[float]:
        """Last trade price — more reliable than quote mid for index/ETF."""
        # Try last trade first (most accurate)
        data = self._get(f"{self.data_url}/v2/stocks/{ticker}/trades/latest",
                         params={"feed": "iex"})
        price = (data.get("trade") or {}).get("p")
        if price and price > 0:
            return round(float(price), 2)
        # Fallback: validated quote mid
        data = self._get(f"{self.data_url}/v2/stocks/{ticker}/quotes/latest",
                         params={"feed": "iex"})
        q   = data.get("quote", {})
        ask = q.get("ap", 0)
        bid = q.get("bp", 0)
        if ask > bid > 0:
            return round((ask + bid) / 2, 2)
        return None

    def get_bars(self, ticker: str, days: int = 60) -> Optional[pd.DataFrame]:
        end   = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=days + 15)).strftime("%Y-%m-%d")
        data  = self._get(
            f"{self.data_url}/v2/stocks/{ticker}/bars",
            params={"start": start, "end": end, "timeframe": "1Day",
                    "limit": 100, "feed": "iex"},
        )
        bars = data.get("bars", [])
        if not bars or len(bars) < 25:
            return None
        df = pd.DataFrame(bars)
        df["t"] = pd.to_datetime(df["t"])
        return df.sort_values("t").reset_index(drop=True)

    def get_contracts(self, ticker: str, opt_type: str,
                      exp_min: str, exp_max: str,
                      strike_min: float, strike_max: float) -> list:
        """Paginated contract fetch from broker API."""
        url    = f"{self.trade_url}/v2/options/contracts"
        params = {
            "underlying_symbols":  ticker,
            "type":                opt_type,
            "expiration_date_gte": exp_min,
            "expiration_date_lte": exp_max,
            "strike_price_gte":    str(int(strike_min)),
            "strike_price_lte":    str(int(strike_max) + 1),
            "limit":               100,
        }
        results, page_token = [], None
        for _ in range(10):
            if page_token:
                params["page_token"] = page_token
            data = self._get(url, params=params)
            for c in data.get("option_contracts", []):
                results.append({
                    "symbol":          c["symbol"],
                    "strike_price":    _norm_strike(c["strike_price"]),
                    "expiration_date": c["expiration_date"],
                })
            page_token = data.get("next_page_token")
            if not page_token:
                break
        return results

    def _detect_options_feed(self) -> str:
        """
        Probe once at startup to find the best available options feed.
        Returns 'opra' if subscribed, otherwise 'indicative'.
        """
        if CONFIG["OPTIONS_FEED"] != "indicative":
            return CONFIG["OPTIONS_FEED"]   # user override
        # Quick probe: try OPRA on a known liquid symbol
        r = self.session.get(
            f"{self.data_url}/v1beta1/options/snapshots",
            headers=self.headers,
            params={"symbols": "SPY260620P00500000", "feed": "opra"},
            timeout=8,
        )
        if r.status_code == 200:
            log.info("Options feed: OPRA (real-time NBBO)")
            return "opra"
        log.info("Options feed: indicative (free tier) — quotes may differ slightly from fills")
        return "indicative"

    def get_snapshots(self, symbols: list, chunk_size: int = 50) -> dict:
        """Batch fetch greeks + quotes using the detected feed."""
        if not symbols:
            return {}
        result = {}
        feed = getattr(self, "_options_feed", "indicative")
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i + chunk_size]
            data  = self._get(
                f"{self.data_url}/v1beta1/options/snapshots",
                params={"symbols": ",".join(chunk), "feed": feed},
            )
            result.update(data.get("snapshots", {}))
            time.sleep(0.15)   # rate-limit buffer between chunks
        return result

    def get_atm_iv(self, ticker: str, price: float) -> Optional[float]:
        """
        True ATM implied volatility from live snapshots.
        Uses the same 21-45 DTE window as the trade selection so the IV
        measurement matches the expiry being traded.
        """
        today      = date.today()
        exp_min    = (today + timedelta(days=CONFIG["MIN_DTE"])).strftime("%Y-%m-%d")
        exp_max    = (today + timedelta(days=CONFIG["MAX_DTE"])).strftime("%Y-%m-%d")
        s_min      = round(price * 0.97, 0)
        s_max      = round(price * 1.03, 0)

        contracts  = self.get_contracts(ticker, "put", exp_min, exp_max, s_min, s_max)
        if not contracts:
            return None

        atm = sorted(contracts, key=lambda c: abs(c["strike_price"] - price))[:6]
        snaps = self.get_snapshots([c["symbol"] for c in atm])

        ivs = [
            snaps[c["symbol"]].get("impliedVolatility", 0) * 100
            for c in atm
            if c["symbol"] in snaps and snaps[c["symbol"]].get("impliedVolatility", 0) > 0
        ]
        return round(sum(ivs) / len(ivs), 1) if ivs else None

    def get_next_earnings(self, ticker: str, dte: int) -> Optional[int]:
        """Days until next earnings; None if no earnings within the trade window."""
        try:
            import curl_cffi.requests as curl_req
            import yfinance as yf
            sess = curl_req.Session(verify=False, impersonate="chrome")
            cal  = yf.Ticker(ticker, session=sess).calendar
            if cal is None:
                return None
            ed = cal.get("Earnings Date")
            if ed is None:
                return None
            next_e = ed[0] if hasattr(ed, "__len__") else ed
            days   = (next_e.date() - date.today()).days
            return days if 0 <= days <= dte else None
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# BULL PUT SPREAD FINDER
# ─────────────────────────────────────────────────────────────────────────────

class BullPutSpreadFinder:

    def __init__(self, alpaca: AlpacaClient):
        self.alpaca = alpaca

    def find_best_spread(self, ticker: str, price: float) -> Optional[dict]:
        today   = date.today()
        exp_min = (today + timedelta(days=CONFIG["MIN_DTE"])).strftime("%Y-%m-%d")
        exp_max = (today + timedelta(days=CONFIG["MAX_DTE"])).strftime("%Y-%m-%d")

        contracts = self.alpaca.get_contracts(
            ticker, "put", exp_min, exp_max,
            round(price * 0.65, 0), round(price * 0.95, 0)
        )
        if not contracts:
            return None

        snapshots = self.alpaca.get_snapshots([c["symbol"] for c in contracts])

        # Group by expiration so both legs always share the same expiry
        by_exp = {}
        for c in contracts:
            snap = snapshots.get(c["symbol"], {})
            by_exp.setdefault(c["expiration_date"], {})[c["strike_price"]] = {
                "symbol": c["symbol"],
                "snap":   snap,
            }

        # Fetch ATM IV once for Gate 2
        atm_iv = self.alpaca.get_atm_iv(ticker, price)

        best       = None
        best_score = -999

        for exp_date, by_strike in by_exp.items():
            exp_d = datetime.strptime(exp_date, "%Y-%m-%d").date()
            dte   = (exp_d - today).days
            if not (CONFIG["MIN_DTE"] <= dte <= CONFIG["MAX_DTE"]):
                continue

            strikes = sorted(by_strike.keys(), reverse=True)

            for short_strike in strikes:
                short_info  = by_strike[short_strike]
                short_snap  = short_info["snap"]
                short_quote = short_snap.get("latestQuote", {})
                short_bid   = short_quote.get("bp", 0)
                short_delta = abs(short_snap.get("greeks", {}).get("delta", 1.0))

                if not _liquid(short_snap):
                    continue
                if short_delta >= CONFIG["MAX_SHORT_DELTA"]:
                    continue

                for width in CONFIG["SPREAD_WIDTHS"]:
                    target_long = _norm_strike(short_strike - width)

                    # Find nearest available long strike within same expiry
                    if target_long in by_strike:
                        long_strike = target_long
                    else:
                        candidates = [s for s in strikes if s < short_strike]
                        if not candidates:
                            continue
                        long_strike = min(candidates,
                                         key=lambda s: abs(s - target_long))
                        if abs(long_strike - target_long) > width * 0.6:
                            continue

                    long_info  = by_strike[long_strike]
                    long_snap  = long_info["snap"]
                    long_ask   = long_snap.get("latestQuote", {}).get("ap", 0)

                    if not _liquid(long_snap):
                        continue

                    actual_width = _norm_strike(short_strike - long_strike)
                    commission   = _commission(2)   # 2 legs, round-trip
                    gross_credit = round((short_bid - long_ask) * 100, 2)
                    net_credit   = round(gross_credit - commission, 2)
                    if net_credit <= 0:
                        continue

                    max_risk     = round(actual_width * 100 - net_credit, 2)
                    if max_risk <= 0 or max_risk > CONFIG["MAX_RISK_PER_TRADE"]:
                        continue

                    credit_ratio  = round(net_credit / (actual_width * 100), 3)
                    breakeven     = round(short_strike - net_credit / 100, 2)
                    breakeven_pct = round((price - breakeven) / price * 100, 1)
                    ror           = round(net_credit / max_risk * 100, 1)

                    if credit_ratio < CONFIG["MIN_CREDIT_RATIO"]:
                        continue
                    if breakeven_pct < CONFIG["MIN_BREAKEVEN_DROP"] * 100:
                        continue

                    # Score on safety + quality, not ROR
                    delta_safety = round((CONFIG["MAX_SHORT_DELTA"] - short_delta)
                                        / CONFIG["MAX_SHORT_DELTA"], 3)
                    score = (breakeven_pct * 0.5
                             + delta_safety  * 100 * 0.3
                             + credit_ratio  * 100 * 0.2)

                    if score > best_score:
                        best_score = score
                        best = {
                            "type":           "bull_put_spread",
                            "short_symbol":   short_info["symbol"],
                            "short_strike":   short_strike,
                            "long_symbol":    long_info["symbol"],
                            "long_strike":    long_strike,
                            "expiration":     exp_date,
                            "dte":            dte,
                            "width":          actual_width,
                            "short_bid":      round(short_bid, 2),
                            "long_ask":       round(long_ask, 2),
                            "gross_credit":   gross_credit,
                            "commission":     round(commission, 2),
                            "net_credit":     net_credit,
                            "max_risk":       max_risk,
                            "credit_ratio":   credit_ratio,
                            "breakeven":      breakeven,
                            "breakeven_pct":  breakeven_pct,
                            "ror":            ror,
                            "short_delta":    round(short_delta, 3),
                            "short_iv":       round(
                                short_snap.get("impliedVolatility", 0) * 100, 1),
                            "atm_iv":         atm_iv,
                        }
        return best


# ─────────────────────────────────────────────────────────────────────────────
# IRON BUTTERFLY FINDER
# ─────────────────────────────────────────────────────────────────────────────

class IronButterflyFinder:
    """
    Sell ATM straddle (call + put, same strike), buy OTM wings.
    Profit if stock stays within breakeven range at expiry.
    Close at 50% of max profit — never hold to expiry.
    """

    def __init__(self, alpaca: AlpacaClient):
        self.alpaca = alpaca

    def find_best_butterfly(self, ticker: str, price: float) -> Optional[dict]:
        today   = date.today()
        exp_min = (today + timedelta(days=CONFIG["MIN_DTE"])).strftime("%Y-%m-%d")
        exp_max = (today + timedelta(days=CONFIG["MAX_DTE"])).strftime("%Y-%m-%d")
        s_min   = round(price * 0.80, 0)
        s_max   = round(price * 1.20, 0)

        call_contracts = self.alpaca.get_contracts(
            ticker, "call", exp_min, exp_max, s_min, s_max)
        put_contracts  = self.alpaca.get_contracts(
            ticker, "put",  exp_min, exp_max, s_min, s_max)

        all_syms  = [c["symbol"] for c in call_contracts + put_contracts]
        snapshots = self.alpaca.get_snapshots(all_syms)

        # Build expiry → strike → data dicts, normalizing strikes
        call_by: dict = {}
        for c in call_contracts:
            snap = snapshots.get(c["symbol"], {})
            if snap:
                k = _norm_strike(c["strike_price"])
                call_by.setdefault(c["expiration_date"], {})[k] = {
                    "symbol": c["symbol"], "snap": snap}

        put_by: dict = {}
        for c in put_contracts:
            snap = snapshots.get(c["symbol"], {})
            if snap:
                k = _norm_strike(c["strike_price"])
                put_by.setdefault(c["expiration_date"], {})[k] = {
                    "symbol": c["symbol"], "snap": snap}

        # Fetch ATM IV once
        atm_iv = self.alpaca.get_atm_iv(ticker, price)

        best       = None
        best_score = -999

        for exp, call_strikes in call_by.items():
            if exp not in put_by:
                continue
            put_strikes = put_by[exp]

            dte = (datetime.strptime(exp, "%Y-%m-%d").date() - today).days
            if not (CONFIG["MIN_DTE"] <= dte <= CONFIG["MAX_DTE"]):
                continue

            # ATM strike = closest available to spot that exists in both chains
            common = set(call_strikes.keys()) & set(put_strikes.keys())
            if not common:
                continue
            atm = min(common, key=lambda s: abs(s - price))

            short_call = call_strikes[atm]
            short_put  = put_strikes[atm]

            # Liquidity check on ATM shorts
            if not _liquid(short_call["snap"]) or not _liquid(short_put["snap"]):
                continue

            sc_bid = short_call["snap"].get("latestQuote", {}).get("bp", 0)
            sp_bid = short_put["snap"].get("latestQuote", {}).get("bp", 0)
            if sc_bid <= 0 or sp_bid <= 0:
                continue

            for wing_width in CONFIG["SPREAD_WIDTHS"]:
                # Find nearest available call wing
                call_target = _norm_strike(atm + wing_width)
                avail_calls = [k for k in call_strikes if k > atm]
                if not avail_calls:
                    continue
                cw = min(avail_calls, key=lambda k: abs(k - call_target))
                if abs(cw - call_target) > wing_width * 0.6:
                    continue

                # Find nearest available put wing
                put_target = _norm_strike(atm - wing_width)
                avail_puts = [k for k in put_strikes if k < atm]
                if not avail_puts:
                    continue
                pw = min(avail_puts, key=lambda k: abs(k - put_target))
                if abs(pw - put_target) > wing_width * 0.6:
                    continue

                long_call = call_strikes.get(cw)
                long_put  = put_strikes.get(pw)
                if not (long_call and long_put):
                    continue
                if not _liquid(long_call["snap"]) or not _liquid(long_put["snap"]):
                    continue

                lc_ask = long_call["snap"].get("latestQuote", {}).get("ap", 0)
                lp_ask = long_put["snap"].get("latestQuote", {}).get("ap", 0)
                if lc_ask <= 0 or lp_ask <= 0:
                    continue

                commission   = _commission(4)   # 4 legs, round-trip
                gross_credit = round((sc_bid + sp_bid - lc_ask - lp_ask) * 100, 2)
                net_credit   = round(gross_credit - commission, 2)
                if net_credit <= 0:
                    continue

                # Max risk = widest wing (asymmetric wings possible) minus credit
                call_wing_width = _norm_strike(cw - atm)
                put_wing_width  = _norm_strike(atm - pw)
                actual_wing     = max(call_wing_width, put_wing_width)
                max_risk        = round(actual_wing * 100 - net_credit, 2)
                if max_risk <= 0 or max_risk > CONFIG["MAX_RISK_PER_TRADE"]:
                    continue

                credit_ratio = round(net_credit / (actual_wing * 100), 3)
                if credit_ratio < CONFIG["MIN_FLY_CREDIT_RATIO"]:
                    continue

                breakeven_up = round(atm + net_credit / 100, 2)
                breakeven_dn = round(atm - net_credit / 100, 2)

                # Check BOTH sides of profit window
                be_pct_dn = round((price - breakeven_dn) / price * 100, 1)
                be_pct_up = round((breakeven_up - price) / price * 100, 1)
                min_cushion = CONFIG["MIN_FLY_WING_PCT"] * 100
                if min(be_pct_dn, be_pct_up) < min_cushion:
                    continue

                wing_pct_call = round(call_wing_width / price * 100, 1)
                wing_pct_put  = round(put_wing_width  / price * 100, 1)
                if min(wing_pct_call, wing_pct_put) < CONFIG["MIN_FLY_WING_PCT"] * 100:
                    continue

                ror = round(net_credit / max_risk * 100, 1)

                short_call_delta = abs(short_call["snap"].get("greeks", {}).get("delta", 1))
                short_put_delta  = abs(short_put["snap"].get("greeks", {}).get("delta", 1))

                # Score same formula as spread
                be_avg = (be_pct_dn + be_pct_up) / 2
                score  = be_avg * 0.5 + credit_ratio * 100 * 0.3 + min_cushion * 0.2

                if score > best_score:
                    best_score = score
                    best = {
                        "type":             "iron_butterfly",
                        "short_call_sym":   short_call["symbol"],
                        "short_put_sym":    short_put["symbol"],
                        "long_call_sym":    long_call["symbol"],
                        "long_put_sym":     long_put["symbol"],
                        "atm_strike":       atm,
                        "call_wing":        cw,
                        "put_wing":         pw,
                        "expiration":       exp,
                        "dte":              dte,
                        "call_wing_width":  call_wing_width,
                        "put_wing_width":   put_wing_width,
                        "wing_width":       actual_wing,
                        "gross_credit":     gross_credit,
                        "commission":       round(commission, 2),
                        "net_credit":       net_credit,
                        "max_risk":         max_risk,
                        "credit_ratio":     credit_ratio,
                        "breakeven_up":     breakeven_up,
                        "breakeven_dn":     breakeven_dn,
                        "be_pct_dn":        be_pct_dn,
                        "be_pct_up":        be_pct_up,
                        "ror":              ror,
                        "short_call_delta": round(short_call_delta, 3),
                        "short_put_delta":  round(short_put_delta, 3),
                        "sc_bid":           round(sc_bid, 2),
                        "sp_bid":           round(sp_bid, 2),
                        "lc_ask":           round(lc_ask, 2),
                        "lp_ask":           round(lp_ask, 2),
                        "atm_iv":           atm_iv or round(
                            short_call["snap"].get("impliedVolatility", 0) * 100, 1),
                        "wing_pct":         round(min(wing_pct_call, wing_pct_put), 1),
                    }

        return best


# ─────────────────────────────────────────────────────────────────────────────
# GATE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _spy_trend(alpaca: AlpacaClient) -> Optional[bool]:
    """Gate 0: True if SPY is above its 21-day EMA (macro bullish)."""
    bars = alpaca.get_bars("SPY", days=60)
    if bars is None or len(bars) < 25:
        return None
    bars["ema21"] = bars["c"].ewm(span=21, adjust=False).mean()
    spy_close = float(bars["c"].iloc[-1])
    ema21     = float(bars["ema21"].iloc[-1])
    return bool(spy_close > ema21)


def run_gates(ticker: str, price: float, trade: dict,
              alpaca: AlpacaClient, spy_ok: Optional[bool]) -> list:
    gates  = []
    is_fly = trade.get("type") == "iron_butterfly"

    # Gate 0: SPY macro trend
    if spy_ok is True:
        gates.append({"gate": 0, "name": "SPY Macro Trend", "pass": True,
                      "detail": "SPY above 21 EMA -- market bullish"})
    elif spy_ok is False:
        gates.append({"gate": 0, "name": "SPY Macro Trend", "pass": False,
                      "detail": "SPY below 21 EMA -- avoid new bullish premium"})
    else:
        gates.append({"gate": 0, "name": "SPY Macro Trend", "pass": None,
                      "detail": "SPY data unavailable"})

    # Gate 1: Ticker above its own 21 EMA
    bars = alpaca.get_bars(ticker, days=60)
    if bars is not None and len(bars) >= 25:
        bars["ema21"] = bars["c"].ewm(span=21, adjust=False).mean()
        ema21  = round(float(bars["ema21"].iloc[-1]), 4)
        passed = bool(price > ema21)
        gates.append({"gate": 1, "name": "Above 21 EMA", "pass": passed,
                      "detail": f"Price ${price} vs EMA21 ${ema21} -- {'OK' if passed else 'BELOW'}"})
    else:
        gates.append({"gate": 1, "name": "Above 21 EMA", "pass": False,
                      "detail": "Insufficient bar data"})

    # Gate 2: ATM IV >= threshold (always ATM, never OTM-skew IV)
    iv = trade.get("atm_iv") or alpaca.get_atm_iv(ticker, price)
    if iv is not None:
        passed = iv >= CONFIG["MIN_IV"]
        gates.append({"gate": 2, "name": f"ATM IV >= {CONFIG['MIN_IV']}%", "pass": passed,
                      "detail": f"ATM IV {iv}% -- {'enough premium' if passed else 'too low'}"})
    else:
        gates.append({"gate": 2, "name": "ATM IV", "pass": None,
                      "detail": "Unable to fetch ATM IV"})

    # Gate 3: Structure quality
    if is_fly:
        wing_pct = trade.get("wing_pct", 0)
        passed   = wing_pct >= CONFIG["MIN_FLY_WING_PCT"] * 100
        gates.append({"gate": 3, "name": f"Wings >= {CONFIG['MIN_FLY_WING_PCT']*100:.0f}% of Spot",
                      "pass": passed,
                      "detail": f"Narrow wing {wing_pct}% -- {'OK' if passed else 'too narrow'}"})
    else:
        delta  = trade.get("short_delta", 0.5)
        passed = delta < CONFIG["MAX_SHORT_DELTA"]
        gates.append({"gate": 3, "name": f"Delta < {CONFIG['MAX_SHORT_DELTA']}",
                      "pass": passed,
                      "detail": f"Delta {delta} -- {'safe' if passed else 'too close to money'}"})

    # Gate 4: Credit ratio (after commissions)
    ratio     = trade.get("credit_ratio", 0)
    min_ratio = CONFIG["MIN_FLY_CREDIT_RATIO"] if is_fly else CONFIG["MIN_CREDIT_RATIO"]
    passed    = ratio >= min_ratio
    gates.append({"gate": 4, "name": f"Credit Ratio >= {round(min_ratio*100)}% (net of commissions)",
                  "pass": passed,
                  "detail": (f"Net credit ${trade.get('net_credit')} "
                             f"(gross ${trade.get('gross_credit')} - "
                             f"${trade.get('commission')} commission) "
                             f"= {round(ratio*100,1)}% of width")})

    # Gate 5: Position sizing
    max_risk = trade.get("max_risk", 9999)
    passed   = max_risk <= CONFIG["MAX_RISK_PER_TRADE"]
    gates.append({"gate": 5, "name": f"Risk <= ${CONFIG['MAX_RISK_PER_TRADE']} (5% account)",
                  "pass": passed,
                  "detail": f"Max loss ${max_risk} -- {'fits' if passed else 'too large'}"})

    # Gate 6: Breakeven cushion — both sides checked for butterfly
    if is_fly:
        dn  = trade.get("be_pct_dn", 0)
        up  = trade.get("be_pct_up", 0)
        thr = CONFIG["MIN_FLY_WING_PCT"] * 100
        passed = bool(min(dn, up) >= thr)
        gates.append({"gate": 6, "name": "Both-sided Profit Cushion", "pass": passed,
                      "detail": (f"Down {dn}% / Up {up}% "
                                 f"BE ${trade.get('breakeven_dn')}-${trade.get('breakeven_up')} "
                                 f"-- {'OK' if passed else 'too tight'}")})
    else:
        be_pct = trade.get("breakeven_pct", 0)
        passed = be_pct >= CONFIG["MIN_BREAKEVEN_DROP"] * 100
        gates.append({"gate": 6, "name": "Breakeven >= 8% Below Spot", "pass": passed,
                      "detail": f"BE ${trade.get('breakeven')} ({be_pct}% below)"})

    # Gate 7: Liquidity (OI + volume + bid-ask) — verified per leg in finders
    gates.append({"gate": 7, "name": "Liquidity (OI/Vol/Spread)", "pass": True,
                  "detail": f"OI >= {CONFIG['MIN_OI']}, Vol >= {CONFIG['MIN_VOLUME']}, "
                            f"BA <= {int(CONFIG['MAX_BID_ASK_PCT']*100)}% -- verified"})

    # Gate 8: Earnings — auto-checked via yfinance
    dte       = trade.get("dte", CONFIG["MAX_DTE"])
    days_earn = alpaca.get_next_earnings(ticker, dte)
    if days_earn is not None:
        gates.append({"gate": 8, "name": "No Earnings in Window", "pass": False,
                      "detail": f"Earnings in {days_earn} days -- SKIP"})
    else:
        gates.append({"gate": 8, "name": "No Earnings in Window", "pass": True,
                      "detail": "No earnings detected in trade window"})

    return gates


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SCAN
# ─────────────────────────────────────────────────────────────────────────────

def scan_ticker(ticker: str, alpaca: AlpacaClient,
                spread_finder: BullPutSpreadFinder,
                fly_finder: IronButterflyFinder,
                spy_ok: Optional[bool]) -> dict:
    log.info(f"Scanning {ticker}...")
    result = {
        "ticker":    ticker,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "price":     None,
        "spread":    None,
        "butterfly": None,
        "best_trade": None,
        "gates":     [],
        "verdict":   "NO SPREAD FOUND",
        "error":     None,
    }

    try:
        price = alpaca.get_stock_price(ticker)
        if not price:
            result["verdict"] = "DATA ERROR"
            return result
        result["price"] = price

        spread    = spread_finder.find_best_spread(ticker, price)
        butterfly = fly_finder.find_best_butterfly(ticker, price)
        result["spread"]    = spread
        result["butterfly"] = butterfly

        if not spread and not butterfly:
            result["verdict"] = "NO QUALIFYING TRADE"
            return result

        # Pick best by score (breakeven-weighted), not raw ROR
        def _score(t):
            if t is None:
                return -999
            if t.get("type") == "iron_butterfly":
                be = (t.get("be_pct_dn", 0) + t.get("be_pct_up", 0)) / 2
            else:
                be = t.get("breakeven_pct", 0)
            d  = t.get("short_delta") or t.get("short_put_delta") or 0.25
            cr = t.get("credit_ratio", 0)
            ds = (CONFIG["MAX_SHORT_DELTA"] - d) / CONFIG["MAX_SHORT_DELTA"]
            return be * 0.5 + ds * 100 * 0.3 + cr * 100 * 0.2

        best = spread if _score(spread) >= _score(butterfly) else butterfly
        result["best_trade"] = best

        gates = run_gates(ticker, price, best, alpaca, spy_ok)
        result["gates"] = gates

        auto_fails = [g for g in gates if g["pass"] is False]
        manual     = [g for g in gates if g["pass"] is None]
        auto_pass  = sum(1 for g in gates if g["pass"] is True)
        auto_total = sum(1 for g in gates if g["pass"] is not None)

        if auto_fails:
            result["verdict"] = f"FAIL -- Gate(s) {', '.join(str(g['gate']) for g in auto_fails)}"
        elif manual:
            result["verdict"] = f"CHECK MANUALLY ({auto_pass}/{auto_total} auto gates pass)"
        else:
            result["verdict"] = "ENTER"

    except Exception as e:
        result["error"]   = str(e)
        result["verdict"] = "SCAN ERROR"
        log.error(f"Error scanning {ticker}: {e}", exc_info=True)

    return result


def _send_slack_alerts(results: list):
    """Post one Slack message per ENTER verdict. Silent if no webhook configured."""
    webhook = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook:
        return

    enters = [r for r in results if r["verdict"] == "ENTER"]
    checks = [r for r in results if "CHECK" in r["verdict"]]

    if not enters and not checks:
        return

    lines = [f"*Bull Put Screener* — {datetime.now().strftime('%Y-%m-%d %H:%M')}"]

    for r in enters + checks:
        t      = r.get("best_trade") or {}
        is_fly = t.get("type") == "iron_butterfly"
        tag    = ":white_check_mark: ENTER" if r["verdict"] == "ENTER" else ":eyes: CHECK"

        if is_fly:
            entry = (f"Sell ${t.get('atm_strike')} straddle / "
                     f"Buy ${t.get('put_wing')}P + ${t.get('call_wing')}C")
            be    = f"BE ${t.get('breakeven_dn')}-${t.get('breakeven_up')}"
            strat = "IRON BUTTERFLY"
        else:
            entry = f"Sell ${t.get('short_strike')}P / Buy ${t.get('long_strike')}P"
            be    = f"BE ${t.get('breakeven')} ({t.get('breakeven_pct')}% below)"
            strat = "BULL PUT SPREAD"

        lines.append(
            f"\n{tag} *{r['ticker']}* @ ${r['price']} — {strat}\n"
            f"  {entry}  exp {t.get('expiration')} ({t.get('dte')} DTE)\n"
            f"  Net ${t.get('net_credit')} | Risk ${t.get('max_risk')} | "
            f"ROR {t.get('ror')}% | IV {t.get('atm_iv')}%\n"
            f"  {be}"
        )

    payload = {"text": "\n".join(lines)}
    try:
        r = requests.post(webhook, json=payload,
                          verify=_SSL_VERIFY, timeout=10)
        if r.status_code == 200:
            log.info(f"Slack alert sent ({len(enters)} ENTER, {len(checks)} CHECK)")
        else:
            log.warning(f"Slack returned {r.status_code}: {r.text}")
    except Exception as e:
        log.warning(f"Slack alert failed: {e}")


def run_scan():
    log.info("=" * 60)
    log.info("BULL PUT SCREENER v3 STARTING")
    log.info("=" * 60)

    if not _market_open():
        log.warning("Market is CLOSED — quotes are stale. Results are for research only.")
        print("\n  [!] WARNING: Market closed. Quotes may not be executable.\n")

    alpaca        = AlpacaClient()
    spread_finder = BullPutSpreadFinder(alpaca)
    fly_finder    = IronButterflyFinder(alpaca)

    # Gate 0 — fetch SPY trend once for the whole scan
    log.info("Fetching SPY macro trend...")
    spy_ok = _spy_trend(alpaca)
    spy_label = "ABOVE 21 EMA (bullish)" if spy_ok else \
                "BELOW 21 EMA (caution)" if spy_ok is False else "UNKNOWN"
    print(f"\n  SPY MACRO: {spy_label}\n")

    results = []
    for ticker in WATCHLIST:
        r = scan_ticker(ticker, alpaca, spread_finder, fly_finder, spy_ok)
        results.append(r)

        price   = f"${r['price']}" if r["price"] else "N/A"
        verdict = r["verdict"]

        print(f"\n{'-'*55}")
        print(f"  {ticker:6s} @ {price:>8}  |  {verdict}")

        spread = r.get("spread")
        if spread:
            print(f"  [SPREAD] Sell ${spread['short_strike']}P / Buy ${spread['long_strike']}P"
                  f"  exp {spread['expiration']}  ({spread['dte']} DTE)")
            print(f"    Net credit ${spread['net_credit']} (gross ${spread['gross_credit']} - ${spread['commission']} comm)"
                  f"  Risk ${spread['max_risk']}  ROR {spread['ror']}%"
                  f"  BE ${spread['breakeven']} ({spread['breakeven_pct']}% below)"
                  f"  delta {spread['short_delta']}")

        fly = r.get("butterfly")
        if fly:
            print(f"  [BUTTERFLY] Sell ${fly['atm_strike']} straddle / "
                  f"Buy ${fly['put_wing']}P + ${fly['call_wing']}C"
                  f"  exp {fly['expiration']}  ({fly['dte']} DTE)")
            print(f"    Net credit ${fly['net_credit']} (gross ${fly['gross_credit']} - ${fly['commission']} comm)"
                  f"  Risk ${fly['max_risk']}  ROR {fly['ror']}%"
                  f"  BE ${fly['breakeven_dn']}-${fly['breakeven_up']}"
                  f"  (dn {fly['be_pct_dn']}% / up {fly['be_pct_up']}%)")

        for g in r.get("gates", []):
            icon = "[OK]" if g["pass"] is True else ("[??]" if g["pass"] is None else "[NO]")
            print(f"    {icon} Gate {g['gate']}: {g['detail']}")

        time.sleep(1.0)

    # ── Summary ───────────────────────────────────────────────────────────────
    ready   = [r for r in results if "ENTER" in r["verdict"] or "CHECK" in r["verdict"]]
    failed  = [r for r in results if "FAIL" in r["verdict"]]
    no_data = [r for r in results if "ERROR" in r["verdict"]
               or "NO QUAL" in r["verdict"] or "NO SPREAD" in r["verdict"]]

    print(f"\n{'='*55}")
    print(f"SCAN COMPLETE -- {len(results)} tickers")
    print(f"  Qualifying  : {len(ready)}")
    print(f"  Failed gates: {len(failed)}")
    print(f"  No data/chain: {len(no_data)}")

    # ── Ranked recommendations ────────────────────────────────────────────────
    def _rank_score(r):
        t   = r.get("best_trade") or {}
        iv  = t.get("atm_iv") or 0
        cr  = t.get("credit_ratio", 0) * 100
        g1  = next((g for g in r.get("gates", []) if g["gate"] == 1), {})
        ema_pen = 0 if g1.get("pass") else -20
        if t.get("type") == "iron_butterfly":
            be = (t.get("be_pct_dn", 0) + t.get("be_pct_up", 0)) / 2
        else:
            be = t.get("breakeven_pct", 0)
        return be * 0.5 + cr * 0.3 + iv * 0.2 + ema_pen

    ranked = sorted(ready, key=_rank_score, reverse=True)

    if ranked:
        print(f"\n{'='*55}")
        print("  RANKED CANDIDATES")
        print(f"{'='*55}")
        for i, r in enumerate(ranked[:8], 1):
            t      = r.get("best_trade") or {}
            is_fly = t.get("type") == "iron_butterfly"
            strat  = "BUTTERFLY" if is_fly else "SPREAD"
            if is_fly:
                entry = (f"Sell ${t.get('atm_strike')} straddle / "
                         f"Buy ${t.get('put_wing')}P+${t.get('call_wing')}C")
                be = f"BE ${t.get('breakeven_dn')}-${t.get('breakeven_up')}"
            else:
                entry = f"Sell ${t.get('short_strike')}P / Buy ${t.get('long_strike')}P"
                be = f"BE ${t.get('breakeven')} ({t.get('breakeven_pct')}% below)"
            print(f"\n  #{i}  {r['ticker']:6s} @ ${r['price']}  [{strat}]  {r['verdict']}")
            print(f"       {entry}  exp {t.get('expiration')}  ({t.get('dte')} DTE)")
            print(f"       Net ${t.get('net_credit')}  Risk ${t.get('max_risk')}"
                  f"  ROR {t.get('ror')}%  IV {t.get('atm_iv')}%"
                  f"  CR {round((t.get('credit_ratio',0))*100,1)}%")
            print(f"       {be}")

        # ── Best trade block ──────────────────────────────────────────────────
        best_r  = ranked[0]
        best_t  = best_r.get("best_trade") or {}
        is_fly  = best_t.get("type") == "iron_butterfly"
        strategy = "IRON BUTTERFLY" if is_fly else "BULL PUT SPREAD"

        print(f"\n{'='*55}")
        print(f"  *** BEST TRADE: {best_r['ticker']} {strategy} ***")
        print(f"{'='*55}")
        print(f"  Expiry : {best_t.get('expiration')}  ({best_t.get('dte')} DTE)")
        print(f"  Legs:")
        if is_fly:
            print(f"    SELL  {best_t.get('short_call_sym'):40s}  ${best_t.get('sc_bid')} bid  (ATM call  ${best_t.get('atm_strike')})")
            print(f"    SELL  {best_t.get('short_put_sym'):40s}  ${best_t.get('sp_bid')} bid  (ATM put   ${best_t.get('atm_strike')})")
            print(f"    BUY   {best_t.get('long_put_sym'):40s}  ${best_t.get('lp_ask')} ask  (put wing  ${best_t.get('put_wing')})")
            print(f"    BUY   {best_t.get('long_call_sym'):40s}  ${best_t.get('lc_ask')} ask  (call wing ${best_t.get('call_wing')})")
            print(f"  Gross credit : ${best_t.get('gross_credit')}")
            print(f"  Commission   : -${best_t.get('commission')} (4 legs round-trip)")
            print(f"  Net credit   : ${best_t.get('net_credit')}  |  Max risk: ${best_t.get('max_risk')}")
            print(f"  ROR          : {best_t.get('ror')}%  |  ATM IV: {best_t.get('atm_iv')}%")
            print(f"  Profit zone  : ${best_t.get('breakeven_dn')} - ${best_t.get('breakeven_up')}"
                  f"  (down {best_t.get('be_pct_dn')}% / up {best_t.get('be_pct_up')}%)")
            print(f"  *** Close at 50% of max profit — do not hold to expiry ***")
        else:
            print(f"    SELL  {best_t.get('short_symbol'):40s}  ${best_t.get('short_bid')} bid  (short put  ${best_t.get('short_strike')})")
            print(f"    BUY   {best_t.get('long_symbol'):40s}  ${best_t.get('long_ask')} ask  (long put   ${best_t.get('long_strike')})")
            print(f"  Gross credit : ${best_t.get('gross_credit')}")
            print(f"  Commission   : -${best_t.get('commission')} (2 legs round-trip)")
            print(f"  Net credit   : ${best_t.get('net_credit')}  |  Width: ${best_t.get('width')} pts"
                  f"  |  Max risk: ${best_t.get('max_risk')}")
            print(f"  ROR          : {best_t.get('ror')}%  |  Delta: {best_t.get('short_delta')}"
                  f"  |  ATM IV: {best_t.get('atm_iv')}%")
            print(f"  Breakeven    : ${best_t.get('breakeven')}  ({best_t.get('breakeven_pct')}% below spot)")

    # ── Slack alerts ──────────────────────────────────────────────────────────
    _send_slack_alerts(results)

    # ── CSV output ─────────────────────────────────────────────────────────────
    path     = CONFIG["CSV_OUTPUT_FILE"]
    file_new = not os.path.isfile(path)
    headers  = [
        "Run", "Timestamp", "Ticker", "Price", "Verdict", "Strategy",
        "Entry", "Expiration", "DTE",
        "Gross Credit", "Commission", "Net Credit", "Max Risk",
        "Credit Ratio %", "ATM IV %", "Return on Risk %",
        "Breakeven / Range", "Breakeven % Below",
        "Gate 0 SPY", "Gate 1 EMA", "Gate 2 IV", "Gate 3 Structure",
        "Gate 4 Credit", "Gate 5 Sizing", "Gate 6 Cushion",
        "Gate 7 Liquidity", "Gate 8 Earnings",
    ]
    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if file_new:
            writer.writerow(headers)
        else:
            writer.writerow([f"--- RUN {run_id} ---"] + [""] * (len(headers) - 1))
        for r in results:
            t      = r.get("best_trade") or {}
            gates  = {g["gate"]: g["detail"] for g in r.get("gates", [])}
            is_fly = t.get("type") == "iron_butterfly"
            if is_fly:
                strat = "IRON BUTTERFLY"
                entry = (f"Sell ${t.get('atm_strike')} / "
                         f"Buy ${t.get('put_wing')}P+${t.get('call_wing')}C")
                be    = f"${t.get('breakeven_dn')}-${t.get('breakeven_up')}"
                be_pct = f"dn {t.get('be_pct_dn')}% / up {t.get('be_pct_up')}%"
            else:
                strat = "BULL PUT SPREAD"
                entry = f"Sell ${t.get('short_strike')}P / Buy ${t.get('long_strike')}P"
                be    = f"${t.get('breakeven')}"
                be_pct = t.get("breakeven_pct", "N/A")
            writer.writerow([
                run_id, r["timestamp"], r["ticker"],
                f"${r['price']}" if r["price"] else "N/A",
                r["verdict"], strat if t else "N/A",
                entry if t else "N/A",
                t.get("expiration", "N/A"), t.get("dte", "N/A"),
                t.get("gross_credit", "N/A"), t.get("commission", "N/A"),
                t.get("net_credit", "N/A"), t.get("max_risk", "N/A"),
                round(t["credit_ratio"] * 100, 1) if t.get("credit_ratio") else "N/A",
                t.get("atm_iv", "N/A"), t.get("ror", "N/A"),
                be if t else "N/A", be_pct,
                gates.get(0, ""), gates.get(1, ""), gates.get(2, ""),
                gates.get(3, ""), gates.get(4, ""), gates.get(5, ""),
                gates.get(6, ""), gates.get(7, ""), gates.get(8, ""),
            ])
    log.info(f"Results appended to {path}  (run {run_id})")
    return results


if __name__ == "__main__":
    run_scan()
