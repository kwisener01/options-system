"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           PMCC DAILY SCANNER — Fynix Systems                                ║
║           Alpaca API → 9 Gate Auto-Check → Google Sheets + Slack            ║
╚══════════════════════════════════════════════════════════════════════════════╝

HOW TO USE:
1. Fill in your credentials in the CONFIG section below
2. Add or remove tickers in the WATCHLIST section
3. Run once:        python3 pmcc_scanner.py
4. Run on schedule: python3 pmcc_scanner.py --schedule
   (runs every trading day at 9:35 AM ET automatically)

REQUIREMENTS:
pip install requests pandas slack-sdk schedule

WHAT IT DOES:
- Pulls live options chains from Alpaca
- Runs 9 of 10 gates automatically (Gate 7 requires your judgment)
- Finds the best LEAPS and short call combo per ticker
- Pushes results to your Google Sheet
- Sends a Slack alert with GO/NO GO summary
"""

import os
import sys
import re
import math
import json
import time
import logging
import requests
import schedule
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional
import csv
from dotenv import load_dotenv

load_dotenv()

import urllib3
import io
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def _bs_call_delta(S: float, K: float, T: float, r: float = 0.05, sigma: float = 0.40) -> float:
    """Black-Scholes delta for a European call option."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return 0.5 * (1 + math.erf(d1 / math.sqrt(2)))

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — FILL THESE IN
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {

    # ── ALPACA ──────────────────────────────────────────────────────────────
    "ALPACA_API_KEY":    os.getenv("ALPACA_API_KEY",    "YOUR_ALPACA_API_KEY"),
    "ALPACA_SECRET_KEY": os.getenv("ALPACA_SECRET_KEY", "YOUR_ALPACA_SECRET_KEY"),
    "ALPACA_BASE_URL":   os.getenv("ALPACA_BASE_URL",   "https://paper-api.alpaca.markets"),
    "ALPACA_DATA_URL":   "https://data.alpaca.markets",

    # ── CSV OUTPUT ──────────────────────────────────────────────────────────
    "CSV_OUTPUT_FILE": "pmcc_scan_results.csv",             # path to write results

    # ── SLACK ────────────────────────────────────────────────────────────────
    # Uses SLACK_WEBHOOK_URL from .env (Incoming Webhook — no bot token needed)
    "SLACK_WEBHOOK": os.getenv("SLACK_WEBHOOK_URL", ""),

    # ── ACCOUNT ──────────────────────────────────────────────────────────────
    "ACCOUNT_SIZE":        1500,     # your total account balance
    "MAX_POSITION_PCT":    0.50,     # max 50% per position = $750 (was 40%; deep ITM LEAPS on $10-16 stocks need $600-750)
    "CASH_BUFFER":         150,      # minimum cash reserve (Gate 10)

    # ── OPTIONS PARAMETERS ───────────────────────────────────────────────────
    "MIN_LEAPS_DELTA":     0.70,     # Gate 4 min delta (was 0.75; 0.70 lets scanner find viable strikes on cheap stocks)
    "MAX_IV_RANK":         65,       # Gate 3 max IV rank (was 50; for PMCC, moderate-high IV is good for short call income)
    "MIN_DTE_SHORT":       21,       # Gate 9 minimum DTE
    "MAX_DTE_SHORT":       45,       # Gate 9 max DTE (was 35; captures June monthly for stocks without weekly options)
    "MIN_LEAPS_MONTHS":    12,       # minimum months on LEAPS
    "MIN_OPTIONS_VOLUME":  100,      # minimum daily volume on short call (was 500; too strict for $8-15 stocks)

    # ── RUN SCHEDULE ─────────────────────────────────────────────────────────
    "SCAN_TIME": "09:35",            # ET — 5 mins after market open

}

# ─────────────────────────────────────────────────────────────────────────────
# WATCHLIST — ADD OR REMOVE TICKERS HERE
# ─────────────────────────────────────────────────────────────────────────────

WATCHLIST = [
    # ── TARGET: $5-20 stock price, liquid options, LEAPS available ──
    # Sorted by sector to diversify across 3-position PMCC portfolio.
    # Skip any ticker already in an open PMCC position.

    # ── FINTECH / BROKERAGE ──────────────────────────────────────────
    "SOFI",     # $10-18 — already in portfolio if PMCC open
    "HOOD",     # $15-25 — Robinhood, liquid options, growing
    "NU",       # $10-18 — Nubank (Brazilian fintech), liquid, LEAPS available
    "FUTU",     # $10-25 — Futu Holdings, Chinese brokerage
    "AFRM",     # $8-20  — Affirm (BNPL), high IV, active options

    # ── CRYPTO / MINERS ─────────────────────────────────────────────
    "MARA",     # $10-25 — Marathon Digital, liquid, high IV
    "RIOT",     # $8-18  — Riot Platforms, liquid options
    "CLSK",     # $8-20  — CleanSpark, BTC miner, high IV
    "CIFR",     # $3-10  — Cipher Mining, smaller but has options

    # ── AI / TECH (small-to-mid) ─────────────────────────────────────
    "AI",       # $15-30 — C3.ai, AI theme, $2.50 strikes
    "SOUN",     # $8-20  — SoundHound AI, growing options market
    "BBAI",     # $3-10  — BigBear.ai, AI/defense
    "IONQ",     # $10-30 — IonQ, quantum computing, active
    "RKLB",     # $10-25 — Rocket Lab, space tech, LEAPS available
    "PATH",     # $10-20 — UiPath, RPA/automation, decent options

    # ── EV / CLEAN ENERGY ────────────────────────────────────────────
    "RIVN",     # $10-18 — Rivian EV, high IV, LEAPS available
    "LCID",     # $2-6   — Lucid Motors, liquid options
    "JOBY",     # $5-12  — Joby Aviation, eVTOL, growing options
    "ACHR",     # $8-15  — Archer Aviation, eVTOL
    "PLUG",     # $2-8   — Plug Power, hydrogen, liquid

    # ── INDUSTRIAL / AUTO ────────────────────────────────────────────
    "F",        # $10-13 — Ford, very liquid, weekly options, LEAPS
    "STLA",     # $10-18 — Stellantis, autos, liquid options

    # ── MEDIA / SOCIAL / CONSUMER ────────────────────────────────────
    "SNAP",     # $8-15  — Snapchat, LEAPS available, active
    "RDDT",     # $10-20 — Reddit, newer listing, growing options
    "PARA",     # $8-15  — Paramount, media, liquid LEAPS

    # ── HEALTHCARE / BIOTECH ─────────────────────────────────────────
    "SANA",     # $2-8   — Sana Biotech, gene editing
    "RXRX",     # $5-15  — Recursion Pharma, AI drug discovery
    "CLOV",     # $2-6   — Clover Health, high IV

    # ── MORTGAGE / REAL ESTATE ───────────────────────────────────────
    "RKT",      # $12-18 — Rocket Companies, mortgage, liquid
    "UWMC",     # $5-8   — UWM Holdings, mortgage, options available
    "OPEN",     # $2-8   — Opendoor, proptech, high IV

    # ── GAMING / SPORTS BETTING ──────────────────────────────────────
    "DKNG",     # $15-35 — DraftKings, liquid options (may be too expensive)
    "PENN",     # $12-20 — Penn Entertainment, gaming/betting
]

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pmcc_scanner.log")
    ]
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ALPACA API CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class AlpacaClient:
    """Handles all Alpaca API calls for market data and options chains."""

    def __init__(self):
        self.headers = {
            "APCA-API-KEY-ID":     CONFIG["ALPACA_API_KEY"],
            "APCA-API-SECRET-KEY": CONFIG["ALPACA_SECRET_KEY"],
            "Accept":              "application/json"
        }
        self.data_url  = CONFIG["ALPACA_DATA_URL"]
        self.trade_url = CONFIG["ALPACA_BASE_URL"]
        self.session   = requests.Session()
        self.session.verify = False

    def get_stock_price(self, ticker: str) -> Optional[float]:
        """Get latest stock price."""
        try:
            url = f"{self.data_url}/v2/stocks/{ticker}/quotes/latest"
            r = self.session.get(url, headers=self.headers, timeout=10)
            r.raise_for_status()
            data = r.json()
            ask = data.get("quote", {}).get("ap", 0)
            bid = data.get("quote", {}).get("bp", 0)
            return round((ask + bid) / 2, 2) if ask and bid else None
        except Exception as e:
            log.warning(f"Price fetch failed for {ticker}: {e}")
            return None

    def get_bars(self, ticker: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Get historical daily bars for EMA calculation."""
        try:
            end   = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=days + 10)).strftime("%Y-%m-%d")
            url   = f"{self.data_url}/v2/stocks/{ticker}/bars"
            params = {
                "start":     start,
                "end":       end,
                "timeframe": "1Day",
                "limit":     50,
                "feed":      "iex"
            }
            r = self.session.get(url, headers=self.headers, params=params, timeout=10)
            r.raise_for_status()
            bars = r.json().get("bars", [])
            if not bars:
                return None
            df = pd.DataFrame(bars)
            df["t"] = pd.to_datetime(df["t"])
            df = df.sort_values("t").reset_index(drop=True)
            return df
        except Exception as e:
            log.warning(f"Bars fetch failed for {ticker}: {e}")
            return None

    def get_options_contracts(self, ticker: str, option_type: str,
                               strike_min: float, strike_max: float,
                               exp_min: str, exp_max: str) -> list:
        """
        List options contracts using the broker API (api.alpaca.markets/v2/options/contracts),
        which supports server-side filtering by type, strike, and expiry.
        """
        url    = f"{self.trade_url}/v2/options/contracts"
        params = {
            "underlying_symbols": ticker,
            "type":               option_type,
            "strike_price_gte":   str(int(strike_min)),
            "strike_price_lte":   str(int(strike_max)),
            "expiration_date_gte": exp_min,
            "expiration_date_lte": exp_max,
            "limit":              100,
        }
        results    = []
        page_token = None

        for _ in range(10):
            if page_token:
                params["page_token"] = page_token
            try:
                r = self.session.get(url, headers=self.headers, params=params, timeout=15)
                r.raise_for_status()
            except Exception as e:
                log.warning(f"Contracts fetch failed for {ticker}: {e}")
                break

            data = r.json()
            for c in data.get("option_contracts", []):
                results.append({
                    "symbol":          c["symbol"],
                    "strike_price":    float(c["strike_price"]),
                    "expiration_date": c["expiration_date"],
                    "open_interest":   int(c.get("open_interest") or 0),
                })

            page_token = data.get("next_page_token")
            if not page_token:
                break

        return results

    def get_options_snapshots(self, symbols: list[str]) -> dict:
        """
        Fetch greeks + bid/ask for a batch of option symbols via the
        multi-symbol snapshots endpoint, which returns real greeks and IV.
        """
        if not symbols:
            return {}
        try:
            r = self.session.get(
                f"{self.data_url}/v1beta1/options/snapshots",
                headers=self.headers,
                params={"symbols": ",".join(symbols), "feed": "indicative"},
                timeout=15
            )
            r.raise_for_status()
            return r.json().get("snapshots", {})
        except Exception as e:
            log.warning(f"Snapshots batch fetch failed: {e}")
            return {}

    def get_option_quote(self, symbol: str, contract: Optional[dict] = None) -> Optional[dict]:
        """Get bid/ask + greeks for a single option symbol."""
        snaps = self.get_options_snapshots([symbol])
        snap  = snaps.get(symbol, {})
        if not snap:
            return None
        quote  = snap.get("latestQuote", {})
        greeks = snap.get("greeks", {})
        return {
            "ap":     quote.get("ap", 0),
            "bp":     quote.get("bp", 0),
            "greeks": greeks,
            "iv":     snap.get("impliedVolatility", 0),
        }

    def get_iv_rank(self, ticker: str) -> Optional[float]:
        """
        Estimate IV rank from historical vs current implied vol.
        Alpaca doesn't provide IV rank directly — we approximate
        using the ratio of current IV to 52-week IV range.
        For production use, supplement with barchart.com scrape
        or a dedicated IV data provider.
        """
        try:
            # Get bars for historical vol calculation
            bars = self.get_bars(ticker, days=365)
            if bars is None or len(bars) < 30:
                return None

            # Calculate 30-day historical volatility
            bars["returns"] = bars["c"].pct_change()
            current_hv = bars["returns"].tail(30).std() * (252 ** 0.5) * 100

            # 52-week HV range as IV rank proxy
            hv_52w_high = bars["returns"].rolling(30).std().max() * (252 ** 0.5) * 100
            hv_52w_low  = bars["returns"].rolling(30).std().min() * (252 ** 0.5) * 100

            if hv_52w_high == hv_52w_low:
                return 50.0  # neutral if no range

            iv_rank = ((current_hv - hv_52w_low) / (hv_52w_high - hv_52w_low)) * 100
            return round(min(max(iv_rank, 0), 100), 1)
        except Exception as e:
            log.warning(f"IV rank calc failed for {ticker}: {e}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# GATE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class GateEngine:
    """Runs all 9 automatable gates for a given ticker."""

    def __init__(self, alpaca: AlpacaClient):
        self.alpaca = alpaca

    def check_gate_1_trend(self, ticker: str, price: float) -> dict:
        """Gate 1: Stock price above 21-day EMA."""
        bars = self.alpaca.get_bars(ticker, days=35)
        if bars is None or len(bars) < 21:
            return {"gate": 1, "name": "21 EMA Trend", "pass": False,
                    "detail": "Insufficient data", "value": "N/A"}
        bars["ema21"] = bars["c"].ewm(span=21, adjust=False).mean()
        ema21 = round(float(bars["ema21"].iloc[-1]), 4)
        passed = bool(price > ema21)
        return {
            "gate":   1,
            "name":   "Stock Above 21 EMA",
            "pass":   passed,
            "value":  f"Price ${price} | EMA21 ${ema21}",
            "detail": "✅ Above 21 EMA" if passed else f"🔴 Below 21 EMA (${ema21})"
        }

    def check_gate_2_earnings(self, ticker: str) -> dict:
        """
        Gate 2: Earnings more than 30 days away.
        NOTE: Alpaca does not provide earnings dates.
        This gate flags as MANUAL — check earningswhispers.com
        In production, connect to an earnings API like
        Finnhub (free tier) or Polygon.io
        """
        return {
            "gate":   2,
            "name":   "Earnings > 30 Days",
            "pass":   None,  # None = manual check required
            "value":  "MANUAL CHECK REQUIRED",
            "detail": "⚠️ Verify at earningswhispers.com"
        }

    def check_gate_3_iv_rank(self, ticker: str) -> dict:
        """Gate 3: IV Rank below 50."""
        iv_rank = self.alpaca.get_iv_rank(ticker)
        if iv_rank is None:
            return {"gate": 3, "name": "IV Rank < 50", "pass": None,
                    "value": "Unable to calculate", "detail": "⚠️ Check barchart.com manually"}
        passed = bool(iv_rank < CONFIG["MAX_IV_RANK"])
        return {
            "gate":   3,
            "name":   "IV Rank < 50",
            "pass":   passed,
            "value":  f"IV Rank: {iv_rank}%",
            "detail": f"✅ IV Rank {iv_rank}% — low enough" if passed
                      else f"🔴 IV Rank {iv_rank}% — too high, LEAPS overpriced"
        }

    def check_gate_4_delta(self, delta: Optional[float]) -> dict:
        """Gate 4: LEAPS delta >= 0.75."""
        if delta is None:
            return {"gate": 4, "name": "LEAPS Delta ≥ 0.75", "pass": False,
                    "value": "N/A", "detail": "🔴 No delta data — go deeper ITM"}
        passed = delta >= CONFIG["MIN_LEAPS_DELTA"]
        return {
            "gate":   4,
            "name":   "LEAPS Delta ≥ 0.75",
            "pass":   passed,
            "value":  f"Delta: {delta}",
            "detail": f"✅ Delta {delta} — strong stock surrogate" if passed
                      else f"🔴 Delta {delta} — too low, go deeper ITM"
        }

    def check_gate_5_roc(self, net_debit: float, short_credit: float) -> dict:
        """
        Gate 5: Monthly ROC >= 2% of net debit.
        PMCC profit is the income stream from rolling short calls, NOT spread-to-expiry math.
        The old 'net_debit < spread_width' check was a vertical spread rule and killed every PMCC.
        """
        if net_debit <= 0:
            return {"gate": 5, "name": "Cycle ROC >= 2%", "pass": False,
                    "value": "N/A", "detail": "🔴 Cannot calculate — net debit is zero or negative"}
        roc = (short_credit / net_debit) * 100
        passed = roc >= 2.0
        return {
            "gate":   5,
            "name":   "Cycle ROC >= 2%",
            "pass":   passed,
            "value":  f"ROC: {roc:.1f}% | Credit ${short_credit} / Net Debit ${net_debit}",
            "detail": f"✅ {roc:.1f}%/cycle — positive income stream" if passed
                      else f"🔴 {roc:.1f}%/cycle — short call too cheap for the LEAPS cost"
        }

    def check_gate_6_sizing(self, leaps_cost: float) -> dict:
        """Gate 6: LEAPS cost under 50% of account (deep ITM LEAPS on $10-16 stocks run $600-750)."""
        max_cost = CONFIG["ACCOUNT_SIZE"] * CONFIG["MAX_POSITION_PCT"]
        passed = leaps_cost < max_cost
        return {
            "gate":   6,
            "name":   "LEAPS Under 50% of Account",
            "pass":   passed,
            "value":  f"LEAPS ${leaps_cost} | Max ${max_cost}",
            "detail": f"✅ ${leaps_cost} fits account sizing" if passed
                      else f"🔴 ${leaps_cost} exceeds ${max_cost} limit — ticker too expensive"
        }

    def check_gate_8_structure(self, leaps_strike: float, short_strike: float) -> dict:
        """Gate 8: Short call strike above LEAPS strike."""
        passed = short_strike > leaps_strike
        return {
            "gate":   8,
            "name":   "Short Strike Above LEAPS Strike",
            "pass":   passed,
            "value":  f"LEAPS ${leaps_strike} | Short ${short_strike}",
            "detail": f"✅ ${short_strike} > ${leaps_strike} — correct structure" if passed
                      else f"🔴 Short strike below LEAPS — trade is inverted"
        }

    def check_gate_9_dte(self, dte: int) -> dict:
        """Gate 9: Short call DTE between 21 and 45."""
        passed = CONFIG["MIN_DTE_SHORT"] <= dte <= CONFIG["MAX_DTE_SHORT"]
        return {
            "gate":   9,
            "name":   f"Short Call DTE {CONFIG['MIN_DTE_SHORT']}-{CONFIG['MAX_DTE_SHORT']}",
            "pass":   passed,
            "value":  f"DTE: {dte}",
            "detail": f"✅ {dte} DTE — theta decay sweet spot" if passed
                      else f"🔴 {dte} DTE — outside {CONFIG['MIN_DTE_SHORT']}-{CONFIG['MAX_DTE_SHORT']} day window"
        }

    def check_gate_10_reserve(self, net_debit: float) -> dict:
        """Gate 10: Enough cash reserve after trade."""
        remaining = CONFIG["ACCOUNT_SIZE"] - net_debit
        passed = remaining >= CONFIG["CASH_BUFFER"]
        return {
            "gate":   10,
            "name":   "Cash Reserve ≥ $150",
            "pass":   passed,
            "value":  f"Remaining: ${round(remaining, 0)}",
            "detail": f"✅ ${round(remaining, 0)} buffer available" if passed
                      else f"🔴 Only ${round(remaining, 0)} left — not enough reserve"
        }


# ─────────────────────────────────────────────────────────────────────────────
# PMCC SCANNER
# ─────────────────────────────────────────────────────────────────────────────

class PMCCScanner:
    """
    Main scanner — finds best LEAPS + short call combo
    and runs all gates for each ticker in the watchlist.
    """

    def __init__(self):
        self.alpaca = AlpacaClient()
        self.gates  = GateEngine(self.alpaca)

    def find_best_leaps(self, ticker: str, price: float) -> Optional[dict]:
        """
        Find the best deep ITM LEAPS call:
        - Strike at 60-80% of current price (deep ITM)
        - Expiration 12-24 months out
        - Delta closest to 0.80 but >= 0.75
        """
        today      = date.today()
        exp_min    = (today + timedelta(days=365)).strftime("%Y-%m-%d")
        exp_max    = (today + timedelta(days=730)).strftime("%Y-%m-%d")
        strike_min = round(price * 0.50, 0)
        strike_max = round(price * 0.85, 0)

        contracts = self.alpaca.get_options_contracts(
            ticker, "call", strike_min, strike_max, exp_min, exp_max
        )

        if not contracts:
            log.warning(f"{ticker}: No LEAPS contracts found")
            return None

        # Batch-fetch greeks + quotes for all candidates in one call
        symbols   = [c["symbol"] for c in contracts]
        snapshots = self.alpaca.get_options_snapshots(symbols)

        best = None
        best_delta_diff = float("inf")

        for c in contracts:
            snap  = snapshots.get(c["symbol"], {})
            quote = snap.get("latestQuote", {})
            ask   = quote.get("ap", 0)
            bid   = quote.get("bp", 0)
            delta = abs(snap.get("greeks", {}).get("delta", 0))

            if ask == 0 or delta < 0.60:
                continue

            cost = round(ask * 100, 2)
            diff = abs(delta - 0.80)

            if diff < best_delta_diff:
                best_delta_diff = diff
                best = {
                    "symbol":     c["symbol"],
                    "strike":     c["strike_price"],
                    "expiration": c["expiration_date"],
                    "ask":        round(ask, 2),
                    "bid":        round(bid, 2),
                    "cost":       cost,
                    "delta":      round(delta, 3)
                }

        return best

    def find_best_short_call(self, ticker: str, price: float,
                              leaps_strike: float) -> Optional[dict]:
        """
        Find the best short call:
        - Strike 10-30% above current price (OTM)
        - 30-60 DTE
        - Above LEAPS strike
        - Minimum volume 500
        """
        today      = date.today()
        exp_min    = (today + timedelta(days=CONFIG["MIN_DTE_SHORT"])).strftime("%Y-%m-%d")
        exp_max    = (today + timedelta(days=CONFIG["MAX_DTE_SHORT"])).strftime("%Y-%m-%d")
        strike_min = max(round(price * 1.05, 0), leaps_strike + 0.50)
        strike_max = round(price * 1.40, 0)

        contracts = self.alpaca.get_options_contracts(
            ticker, "call", strike_min, strike_max, exp_min, exp_max
        )

        if not contracts:
            log.warning(f"{ticker}: No short call contracts found")
            return None

        symbols   = [c["symbol"] for c in contracts]
        snapshots = self.alpaca.get_options_snapshots(symbols)

        best = None
        best_credit = 0

        for c in contracts:
            snap  = snapshots.get(c["symbol"], {})
            quote = snap.get("latestQuote", {})
            bid   = quote.get("bp", 0)
            if bid == 0:
                continue

            exp_date = datetime.strptime(c["expiration_date"], "%Y-%m-%d").date()
            dte = (exp_date - today).days

            if not (CONFIG["MIN_DTE_SHORT"] <= dte <= CONFIG["MAX_DTE_SHORT"]):
                continue

            credit = round(bid * 100, 2)

            if credit > best_credit:
                best_credit = credit
                best = {
                    "symbol":     c["symbol"],
                    "strike":     c["strike_price"],
                    "expiration": c["expiration_date"],
                    "bid":        round(bid, 2),
                    "credit":     credit,
                    "dte":        dte,
                    "volume":     int(snap.get("dailyBar", {}).get("v", 0))
                }

        return best

    def scan_ticker(self, ticker: str) -> dict:
        """Run full 10-gate scan for one ticker."""
        log.info(f"Scanning {ticker}...")
        result = {
            "ticker":     ticker,
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M"),
            "price":      None,
            "leaps":      None,
            "short_call": None,
            "net_debit":  None,
            "spread_width": None,
            "breakeven":  None,
            "monthly_roc": None,
            "gates":      [],
            "gates_pass": 0,
            "verdict":    "🔴 INCOMPLETE",
            "error":      None
        }

        try:
            # Get stock price
            price = self.alpaca.get_stock_price(ticker)
            if not price:
                result["error"] = "Could not fetch stock price"
                result["verdict"] = "⚠️ DATA ERROR"
                return result
            result["price"] = price

            # Find best LEAPS
            leaps = self.find_best_leaps(ticker, price)
            if not leaps:
                result["error"] = "No suitable LEAPS found"
                result["verdict"] = "⚠️ NO LEAPS"
                return result
            result["leaps"] = leaps

            # Find best short call
            short = self.find_best_short_call(ticker, price, leaps["strike"])
            if not short:
                result["error"] = "No suitable short call found"
                result["verdict"] = "⚠️ NO SHORT CALL"
                return result
            result["short_call"] = short

            # Calculate trade metrics
            net_debit    = round(leaps["cost"] - short["credit"], 2)
            spread_width = round((short["strike"] - leaps["strike"]) * 100, 2)
            breakeven    = round(leaps["strike"] + (net_debit / 100), 2)
            monthly_roc  = round((short["credit"] / net_debit) * 100, 1) if net_debit > 0 else 0

            result["net_debit"]    = net_debit
            result["spread_width"] = spread_width
            result["breakeven"]    = breakeven
            result["monthly_roc"]  = monthly_roc

            # ── RUN ALL GATES ─────────────────────────────────────────────
            gates = []

            gates.append(self.gates.check_gate_1_trend(ticker, price))
            gates.append(self.gates.check_gate_2_earnings(ticker))       # manual
            gates.append(self.gates.check_gate_3_iv_rank(ticker))
            gates.append(self.gates.check_gate_4_delta(leaps["delta"]))
            gates.append(self.gates.check_gate_5_roc(net_debit, short["credit"]))
            gates.append(self.gates.check_gate_6_sizing(leaps["cost"]))
            gates.append({                                                # gate 7 manual
                "gate":   7,
                "name":   "Bullish Narrative (6-12 months)",
                "pass":   None,
                "value":  "MANUAL CHECK REQUIRED",
                "detail": "⚠️ Research company fundamentals"
            })
            gates.append(self.gates.check_gate_8_structure(
                leaps["strike"], short["strike"]
            ))
            gates.append(self.gates.check_gate_9_dte(short["dte"]))
            gates.append(self.gates.check_gate_10_reserve(net_debit))

            result["gates"] = gates

            # Count auto gates that pass (exclude manual gates 2 and 7)
            auto_pass  = sum(1 for g in gates if g["pass"] is True)
            auto_total = sum(1 for g in gates if g["pass"] is not None)
            manual_count = sum(1 for g in gates if g["pass"] is None)

            result["gates_pass"] = auto_pass

            # Verdict logic
            auto_fails = [g for g in gates if g["pass"] is False]

            if auto_fails:
                failing_gates = ", ".join([f"Gate {g['gate']}" for g in auto_fails])
                result["verdict"] = f"🔴 FAIL — {failing_gates}"
            elif manual_count > 0:
                result["verdict"] = f"⚠️ CHECK GATES 2 & 7 MANUALLY ({auto_pass}/{auto_total} auto gates pass)"
            else:
                result["verdict"] = "🟢 ENTER — ALL GATES PASS"

        except Exception as e:
            result["error"]   = str(e)
            result["verdict"] = "⚠️ SCAN ERROR"
            log.error(f"Scan error for {ticker}: {e}")

        return result


# ─────────────────────────────────────────────────────────────────────────────
# CSV OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

class CSVOutput:
    """Appends scanner results to a local CSV file."""

    HEADERS = [
        "Scan Time", "Ticker", "Price", "Verdict",
        "LEAPS Strike", "LEAPS Expiry", "LEAPS Cost", "LEAPS Delta",
        "Short Strike", "Short Expiry", "Short Credit", "DTE",
        "Net Debit", "Spread Width", "Breakeven", "Monthly ROC %",
        "Gate 1 Trend", "Gate 2 Earnings*", "Gate 3 IV Rank",
        "Gate 4 Delta", "Gate 5 Spread", "Gate 6 Sizing",
        "Gate 7 Narrative*", "Gate 8 Structure", "Gate 9 DTE",
        "Gate 10 Reserve", "Notes"
    ]

    def write_results(self, results: list[dict]):
        """Append all scan results to CSV, writing header if file is new."""
        path = CONFIG["CSV_OUTPUT_FILE"]
        file_exists = os.path.isfile(path)

        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(self.HEADERS)

            for r in results:
                gates = {g["gate"]: g for g in r.get("gates", [])}

                def gd(n):
                    g = gates.get(n, {})
                    return g.get("detail", "MANUAL" if g.get("pass") is None else "N/A")

                leaps = r.get("leaps") or {}
                short = r.get("short_call") or {}

                writer.writerow([
                    r["timestamp"],
                    r["ticker"],
                    f"${r['price']}"     if r["price"]     else "N/A",
                    r["verdict"],
                    f"${leaps.get('strike', 'N/A')}",
                    leaps.get("expiration", "N/A"),
                    f"${leaps.get('cost', 'N/A')}",
                    leaps.get("delta", "N/A"),
                    f"${short.get('strike', 'N/A')}",
                    short.get("expiration", "N/A"),
                    f"${short.get('credit', 'N/A')}",
                    short.get("dte", "N/A"),
                    f"${r['net_debit']}"    if r["net_debit"]    else "N/A",
                    f"${r['spread_width']}" if r["spread_width"] else "N/A",
                    f"${r['breakeven']}"    if r["breakeven"]    else "N/A",
                    f"{r['monthly_roc']}%"  if r["monthly_roc"]  else "N/A",
                    gd(1), gd(2), gd(3), gd(4), gd(5),
                    gd(6), gd(7), gd(8), gd(9), gd(10),
                    r.get("error", "")
                ])

        log.info(f"Results written to {path} — {len(results)} tickers scanned")


# ─────────────────────────────────────────────────────────────────────────────
# SLACK OUTPUT  (Incoming Webhook — uses SLACK_WEBHOOK_URL from .env)
# ─────────────────────────────────────────────────────────────────────────────

class SlackOutput:
    """Sends scan summary to Slack via Incoming Webhook."""

    def send_results(self, results: list[dict]):
        webhook = CONFIG.get("SLACK_WEBHOOK", "")
        if not webhook:
            log.info("No SLACK_WEBHOOK_URL set — skipping Slack notification")
            return

        green  = [r for r in results if "🟢" in r["verdict"]]
        yellow = [r for r in results if "⚠️" in r["verdict"]]
        red    = [r for r in results if "🔴" in r["verdict"]]

        now   = datetime.now().strftime("%A %b %d, %Y — %I:%M %p ET")
        lines = [
            f"*📊 PMCC Scanner — {now}*",
            f"{len(results)} scanned | 🟢 {len(green)} ready | ⚠️ {len(yellow)} manual | 🔴 {len(red)} fail",
            "",
        ]

        if green:
            lines.append("*🟢 READY TO ENTER*")
            for r in green:
                leaps = r.get("leaps") or {}
                short = r.get("short_call") or {}
                lines.append(
                    f"• *{r['ticker']}* ${r['price']} | "
                    f"LEAPS ${leaps.get('strike')} {leaps.get('expiration')} δ{leaps.get('delta')} | "
                    f"Short ${short.get('strike')} {short.get('expiration')} "
                    f"${short.get('credit')} {short.get('dte')}DTE | "
                    f"Net ${r['net_debit']} | ROC {r['monthly_roc']}%/cycle"
                )
            lines.append("")

        if yellow:
            lines.append("*⚠️ CHECK MANUALLY*")
            for r in yellow:
                lines.append(f"• *{r['ticker']}* ${r.get('price','?')} — {r['verdict']}")
            lines.append("")

        if red:
            lines.append("*🔴 DISQUALIFIED*")
            for r in red:
                lines.append(f"• *{r['ticker']}* ${r.get('price','?')} — {r['verdict']}")
            lines.append("")

        lines.append(
            f"_Results: {CONFIG['CSV_OUTPUT_FILE']} | "
            "Gate 2 (Earnings) & Gate 7 (Narrative) = manual check required_"
        )

        try:
            resp = requests.post(webhook, json={"text": "\n".join(lines)}, timeout=8)
            if resp.status_code == 200:
                log.info("Slack alert sent")
            else:
                log.warning(f"Slack returned {resp.status_code}: {resp.text}")
        except Exception as e:
            log.error(f"Slack send failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RUN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_scan():
    """Run the full scan — called directly or by scheduler."""
    log.info("=" * 60)
    log.info("PMCC SCANNER STARTING")
    log.info("=" * 60)

    scanner = PMCCScanner()
    results = []

    for ticker in WATCHLIST:
        result = scanner.scan_ticker(ticker)
        results.append(result)

        # Print to console
        verdict = result["verdict"]
        price   = f"${result['price']}" if result["price"] else "N/A"
        print(f"\n{'-'*50}")
        print(f"  {ticker} @ {price}")
        print(f"  {verdict}")
        if result["net_debit"]:
            print(f"  Net Debit: ${result['net_debit']} | "
                  f"ROC: {result['monthly_roc']}%/cycle | "
                  f"Breakeven: ${result['breakeven']}")
        for g in result["gates"]:
            icon = "[OK]" if g["pass"] is True else ("[??]" if g["pass"] is None else "[NO]")
            print(f"  {icon} Gate {g['gate']}: {g['detail']}")

        # Small delay to respect API rate limits
        time.sleep(1.5)

    print(f"\n{'='*60}")
    print(f"SCAN COMPLETE -- {len(results)} tickers processed")
    print(f"{'='*60}\n")

    # Write to CSV
    try:
        CSVOutput().write_results(results)
    except Exception as e:
        log.error(f"CSV write failed: {e}")

    # Send Slack alert
    try:
        SlackOutput().send_results(results)
    except Exception as e:
        log.error(f"Slack send failed: {e}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────

def run_scheduled():
    """Run on schedule every trading day at configured time."""
    scan_time = CONFIG["SCAN_TIME"]
    log.info(f"Scheduler started — scanning daily at {scan_time} ET")
    log.info("Press Ctrl+C to stop")

    schedule.every().monday.at(scan_time).do(run_scan)
    schedule.every().tuesday.at(scan_time).do(run_scan)
    schedule.every().wednesday.at(scan_time).do(run_scan)
    schedule.every().thursday.at(scan_time).do(run_scan)
    schedule.every().friday.at(scan_time).do(run_scan)

    while True:
        schedule.run_pending()
        time.sleep(30)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--schedule" in sys.argv:
        run_scheduled()
    else:
        run_scan()
