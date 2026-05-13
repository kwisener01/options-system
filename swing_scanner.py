"""
SWING SCANNER  v2  — GEX + Levels
Scans a watchlist for 1-2 day directional setups driven by dealer gamma positioning.

Market regime (SPY, once per scan):
  Uses existing gex_scanner.scan() — net GEX, vanna, flip level, walls

Per-ticker signals (need >= 2 aligned to qualify, max score 4):
  G1  Near put wall    : price within 1.5% above put wall → dealers defend → BULLISH
  G2  Near call wall   : price within 1.5% below call wall → dealers cap → BEARISH
  G3  Flip momentum    : net GEX < 0 AND price above flip → trending BULLISH
                         net GEX < 0 AND price below flip → trending BEARISH
  G4  SPY macro align  : SPY vanna = BULLISH confirms long, BEARISH confirms short
                         SPY regime negative = momentum amplifier (bonus if aligned)

Instrument selection per setup:
  1. Options: 2-7 DTE call (bullish) or put (bearish), delta 0.35-0.55, liquid chain
  2. Fallback: stock (shares) if chain is illiquid or no qualifying contract found

Run:  python swing_scanner.py
"""

import io
import logging
import math
import os
import sys
import time
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
import urllib3
from dotenv import load_dotenv
from scipy.stats import norm

load_dotenv()

# ── SSL probe ────────────────────────────────────────────────────────────────

def _best_ssl_verify():
    import requests as _req
    for verify in (
        __import__("certifi").where() if True else None,
        True,
    ):
        try:
            s = _req.Session()
            s.verify = verify
            s.get("https://data.alpaca.markets", timeout=5)
            return verify
        except Exception:
            continue
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    print("[!] WARNING: SSL cert verification DISABLED.")
    return False

try:
    _SSL_VERIFY = _best_ssl_verify()
except Exception:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    _SSL_VERIFY = False

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── CONFIG ───────────────────────────────────────────────────────────────────

CONFIG = {
    "ALPACA_API_KEY":    os.getenv("ALPACA_API_KEY"),
    "ALPACA_SECRET_KEY": os.getenv("ALPACA_SECRET_KEY"),
    "ALPACA_BASE_URL":   os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
    "ALPACA_DATA_URL":   "https://data.alpaca.markets",

    "ACCOUNT_SIZE":       1500,
    "MAX_RISK_PER_TRADE": 150,
    "COMMISSION_PER_LEG": 0.65,

    # GEX signal thresholds
    "NEAR_WALL_PCT":      0.015,   # price within 1.5% of wall to trigger
    "FLIP_ZONE_PCT":      0.010,   # price within 1% of flip level counts as "at flip"
    "GEX_CHAIN_DTE":      30,      # how many DTE to pull for GEX computation
    "GEX_CHAIN_RANGE":    0.14,    # ±14% strike range for GEX chain

    # Options selection (swing entry)
    "MIN_DTE":            2,
    "MAX_DTE":            7,
    "TARGET_DELTA":       0.45,
    "DELTA_RANGE":        0.18,
    "MAX_BID_ASK_PCT":    0.35,
    "MIN_VOLUME":         5,
    "MIN_OI":             50,
    "OPTIONS_FEED":       "indicative",
}

WATCHLIST = [
    "SPY", "QQQ", "IWM", "XLF", "XLE", "XLK",
    "AAPL", "MSFT", "NVDA", "META", "TSLA", "AMZN", "GOOGL", "AMD",
    "PLTR", "MARA", "HOOD", "SOFI", "COIN", "RIVN", "F", "BAC",
]

# ── LOGGING ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("swing_scanner.log")],
)
log = logging.getLogger(__name__)


# ── HELPERS ───────────────────────────────────────────────────────────────────

_RF     = 0.045
_SHARES = 100


def _norm_strike(s) -> float:
    return round(float(s), 2)


def _ba_pct(bid: float, ask: float) -> float:
    mid = (bid + ask) / 2
    return (ask - bid) / mid if mid > 0 else 999


def _market_open() -> bool:
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5:
        return False
    return now.replace(hour=9, minute=30) <= now <= now.replace(hour=16, minute=0)


def _gamma_bs(S: float, K: float, T: float, sig: float) -> float:
    """Black-Scholes gamma."""
    if T <= 1e-6 or sig <= 1e-6 or S <= 0 or K <= 0:
        return 0.0
    d1 = (math.log(S / K) + (_RF + 0.5 * sig ** 2) * T) / (sig * math.sqrt(T))
    return norm.pdf(d1) / (S * sig * math.sqrt(T))


# ── ALPACA CLIENT ─────────────────────────────────────────────────────────────

class AlpacaClient:

    def __init__(self):
        self.headers = {
            "APCA-API-KEY-ID":     CONFIG["ALPACA_API_KEY"],
            "APCA-API-SECRET-KEY": CONFIG["ALPACA_SECRET_KEY"],
            "Accept":              "application/json",
        }
        self.data_url  = CONFIG["ALPACA_DATA_URL"]
        self.trade_url = CONFIG["ALPACA_BASE_URL"].replace("paper-api", "api")
        self.session   = requests.Session()
        self.session.verify = _SSL_VERIFY

    def _get(self, url: str, params: dict = None, timeout: int = 15) -> dict:
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
        data = self._get(f"{self.data_url}/v2/stocks/{ticker}/trades/latest",
                         params={"feed": "iex"})
        price = (data.get("trade") or {}).get("p")
        if price and price > 0:
            return round(float(price), 2)
        data = self._get(f"{self.data_url}/v2/stocks/{ticker}/quotes/latest",
                         params={"feed": "iex"})
        q   = data.get("quote", {})
        ask = q.get("ap", 0)
        bid = q.get("bp", 0)
        if ask > bid > 0:
            return round((ask + bid) / 2, 2)
        return None

    def get_contracts(self, ticker: str, opt_type: str,
                      exp_min: str, exp_max: str,
                      strike_min: float, strike_max: float) -> list:
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
        for _ in range(5):
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

    def get_snapshots(self, symbols: list, chunk_size: int = 50) -> dict:
        if not symbols:
            return {}
        result = {}
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i + chunk_size]
            data  = self._get(
                f"{self.data_url}/v1beta1/options/snapshots",
                params={"symbols": ",".join(chunk), "feed": CONFIG["OPTIONS_FEED"]},
            )
            result.update(data.get("snapshots", {}))
            time.sleep(0.15)
        return result

    def get_gex_chain(self, ticker: str, spot: float) -> tuple[list, dict]:
        """Fetch 0-30 DTE contracts (calls + puts) for GEX computation."""
        today   = date.today()
        exp_min = today.strftime("%Y-%m-%d")
        exp_max = (today + timedelta(days=CONFIG["GEX_CHAIN_DTE"])).strftime("%Y-%m-%d")
        rng     = CONFIG["GEX_CHAIN_RANGE"]
        s_min   = round(spot * (1 - rng), 0)
        s_max   = round(spot * (1 + rng), 0)

        raw_calls = self.get_contracts(ticker, "call", exp_min, exp_max, s_min, s_max)
        raw_puts  = self.get_contracts(ticker, "put",  exp_min, exp_max, s_min, s_max)

        contracts = (
            [{**c, "is_call": True}  for c in raw_calls] +
            [{**c, "is_call": False} for c in raw_puts]
        )
        if not contracts:
            return [], {}

        snapshots = self.get_snapshots([c["symbol"] for c in contracts])
        return contracts, snapshots


# ── PER-TICKER GEX ────────────────────────────────────────────────────────────

def compute_ticker_gex(spot: float, contracts: list, snapshots: dict) -> Optional[dict]:
    """
    Compute GEX levels for a single ticker from Alpaca options chain.
    Returns dict with gamma_wall, call_wall, put_wall, flip_level, net_gex, top_levels.
    Returns None if insufficient chain data.
    """
    today             = date.today()
    gex_by_strike:    dict[float, float] = {}
    call_gex_strike:  dict[float, float] = {}
    put_gex_strike:   dict[float, float] = {}
    n_used = 0

    for c in contracts:
        snap = snapshots.get(c["symbol"])
        if not snap:
            continue

        iv_raw = snap.get("impliedVolatility", 0) or 0
        iv     = float(iv_raw)
        if iv < 0.01:
            continue

        oi_raw = snap.get("openInterest")
        oi     = int(oi_raw) if oi_raw is not None else 1   # fallback: weight=1

        strike  = float(c["strike_price"])
        exp_d   = datetime.strptime(c["expiration_date"], "%Y-%m-%d").date()
        T       = max((exp_d - today).days, 0) / 365
        is_call = bool(c["is_call"])

        # Prefer Greeks from snapshot; fall back to BS
        greeks    = snap.get("greeks") or {}
        gamma_raw = greeks.get("gamma")
        gamma     = float(gamma_raw) if gamma_raw and float(gamma_raw) > 0 \
                    else _gamma_bs(spot, strike, T, iv)

        gex        = gamma * oi * _SHARES * (spot ** 2) / 1e9
        signed_gex = gex if is_call else -gex

        gex_by_strike[strike]  = gex_by_strike.get(strike, 0.0) + signed_gex
        if is_call:
            call_gex_strike[strike] = call_gex_strike.get(strike, 0.0) + gex
        else:
            put_gex_strike[strike]  = put_gex_strike.get(strike, 0.0) + gex
        n_used += 1

    if n_used < 4 or not gex_by_strike:
        return None

    net_gex = sum(gex_by_strike.values())

    # Gamma wall: largest absolute GEX within 8% of spot
    near = {k: v for k, v in gex_by_strike.items() if abs(k - spot) / spot < 0.08}
    gamma_wall = max(near, key=lambda k: abs(near[k]), default=spot) if near else spot

    # Call wall: strongest call GEX above spot (resistance)
    calls_above = {k: v for k, v in call_gex_strike.items() if k > spot}
    call_wall   = max(calls_above, key=lambda k: calls_above[k],
                      default=round(spot * 1.03, 2)) if calls_above else round(spot * 1.03, 2)

    # Put wall: strongest put GEX below spot (support)
    puts_below  = {k: v for k, v in put_gex_strike.items() if k < spot}
    put_wall    = max(puts_below, key=lambda k: puts_below[k],
                     default=round(spot * 0.97, 2)) if puts_below else round(spot * 0.97, 2)

    # Flip level: cumulative GEX sign change (ascending)
    flip_level  = spot
    cum, prev_k, prev_cum = 0.0, None, 0.0
    for k, g in sorted(gex_by_strike.items()):
        prev_cum = cum
        cum += g
        if prev_k is not None and prev_cum * cum < 0:
            flip_level = round((prev_k + k) / 2, 2)
            break
        prev_k = k

    top_levels = sorted(gex_by_strike.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

    return {
        "net_gex":    round(net_gex, 4),
        "regime":     "POSITIVE" if net_gex > 0 else "NEGATIVE",
        "gamma_wall": gamma_wall,
        "call_wall":  call_wall,
        "put_wall":   put_wall,
        "flip_level": flip_level,
        "top_levels": top_levels,
        "n_contracts": n_used,
    }


# ── GEX SIGNAL ENGINE ─────────────────────────────────────────────────────────

def gex_signals(spot: float, gex: dict, spy_gex) -> dict:
    """
    Evaluate GEX-based directional signals. Returns direction + score.

    Signals:
      G1  Near put wall   → BULLISH
      G2  Near call wall  → BEARISH
      G3  Flip momentum   → BULLISH if neg-gamma above flip, BEARISH if below
      G4  SPY macro align → +1 if SPY vanna / regime confirms direction
    """
    near_wall = CONFIG["NEAR_WALL_PCT"]

    pct_above_put  = (spot - gex["put_wall"])  / spot   # positive = above wall
    pct_below_call = (gex["call_wall"] - spot) / spot   # positive = below wall
    pct_vs_flip    = (spot - gex["flip_level"]) / spot  # positive = above flip

    g1_bull = 0 < pct_above_put  <= near_wall           # sitting just above put wall
    g2_bear = 0 < pct_below_call <= near_wall           # sitting just below call wall
    g3_bull = gex["regime"] == "NEGATIVE" and pct_vs_flip > CONFIG["FLIP_ZONE_PCT"]
    g3_bear = gex["regime"] == "NEGATIVE" and pct_vs_flip < -CONFIG["FLIP_ZONE_PCT"]

    # SPY macro alignment (G4)
    g4_bull, g4_bear = False, False
    spy_regime = spy_gex.get("regime", "UNKNOWN") if spy_gex else "UNKNOWN"
    if spy_regime == "NEGATIVE":
        g4_bull = True
        g4_bear = True   # negative gamma amplifies both directions — let other signals decide

    bull_count = sum([g1_bull, g3_bull, g4_bull])
    bear_count = sum([g2_bear, g3_bear, g4_bear])

    direction = None
    score     = 0
    if bull_count >= 2 and bull_count >= bear_count:
        direction = "bull"
        score     = bull_count
    elif bear_count >= 2 and bear_count > bull_count:
        direction = "bear"
        score     = bear_count

    reasons = []
    if g1_bull: reasons.append(f"near put wall ${gex['put_wall']} ({pct_above_put:.1%} above)")
    if g2_bear: reasons.append(f"near call wall ${gex['call_wall']} ({pct_below_call:.1%} below)")
    if g3_bull: reasons.append(f"neg-gamma breakout above flip ${gex['flip_level']}")
    if g3_bear: reasons.append(f"neg-gamma breakdown below flip ${gex['flip_level']}")
    if g4_bull and direction == "bull": reasons.append(f"SPY regime={spy_regime} (negative gamma amplifies)")
    if g4_bear and direction == "bear": reasons.append(f"SPY regime={spy_regime} (negative gamma amplifies)")

    return {
        "direction":  direction,
        "score":      score,
        "reasons":    reasons,
        "g1_bull":    g1_bull,
        "g2_bear":    g2_bear,
        "g3_bull":    g3_bull,
        "g3_bear":    g3_bear,
        "g4_bull":    g4_bull,
        "g4_bear":    g4_bear,
        "pct_above_put":  round(pct_above_put * 100, 2),
        "pct_below_call": round(pct_below_call * 100, 2),
        "pct_vs_flip":    round(pct_vs_flip * 100, 2),
    }


# ── OPTIONS FINDER ────────────────────────────────────────────────────────────

def find_swing_option(ticker: str, price: float, direction: str,
                      alpaca: AlpacaClient) -> Optional[dict]:
    """Find a 2-7 DTE call (bull) or put (bear) near ATM with liquid chain."""
    opt_type  = "call" if direction == "bull" else "put"
    today     = date.today()
    exp_min   = (today + timedelta(days=CONFIG["MIN_DTE"])).strftime("%Y-%m-%d")
    exp_max   = (today + timedelta(days=CONFIG["MAX_DTE"])).strftime("%Y-%m-%d")
    s_min     = round(price * 0.90, 0)
    s_max     = round(price * 1.10, 0)

    contracts = alpaca.get_contracts(ticker, opt_type, exp_min, exp_max, s_min, s_max)
    if not contracts:
        return None

    snapshots = alpaca.get_snapshots([c["symbol"] for c in contracts])

    target_delta = CONFIG["TARGET_DELTA"]
    delta_range  = CONFIG["DELTA_RANGE"]
    best, best_dist = None, 999

    for c in contracts:
        snap = snapshots.get(c["symbol"])
        if not snap:
            continue

        q   = snap.get("latestQuote", {})
        bid = q.get("bp", 0)
        ask = q.get("ap", 0)
        if ask <= bid or bid <= 0:
            continue

        oi  = snap.get("openInterest")
        vol = (snap.get("dailyBar") or {}).get("v", 0) or 0
        if oi is not None and oi < CONFIG["MIN_OI"]:
            continue
        if vol < CONFIG["MIN_VOLUME"]:
            continue
        if _ba_pct(bid, ask) > CONFIG["MAX_BID_ASK_PCT"]:
            continue

        delta_raw = snap.get("greeks", {}).get("delta")
        if delta_raw is None:
            continue
        delta = abs(float(delta_raw))
        if not (target_delta - delta_range <= delta <= target_delta + delta_range):
            continue

        dist = abs(delta - target_delta)
        if dist < best_dist:
            best_dist = dist
            dte = (datetime.strptime(c["expiration_date"], "%Y-%m-%d").date() - today).days
            mid = round((bid + ask) / 2, 2)
            best = {
                "symbol":      c["symbol"],
                "strike":      c["strike_price"],
                "expiration":  c["expiration_date"],
                "dte":         dte,
                "type":        opt_type,
                "bid":         round(bid, 2),
                "ask":         round(ask, 2),
                "mid":         mid,
                "delta":       round(delta, 3),
                "iv":          round(snap.get("impliedVolatility", 0) * 100, 1),
                "oi":          oi,
                "volume":      vol,
                "ba_pct":      round(_ba_pct(bid, ask) * 100, 1),
                "cost_1ct":    round(ask * 100, 2),
                "max_gain_est": round(ask * 100 * 1.5, 2),
            }

    return best


# ── STOCK FALLBACK ────────────────────────────────────────────────────────────

def stock_fallback(ticker: str, price: float, direction: str) -> dict:
    risk   = CONFIG["MAX_RISK_PER_TRADE"]
    qty    = max(1, int(risk / price))
    cost   = round(qty * price, 2)
    side   = "BUY" if direction == "bull" else "SELL SHORT"
    target = round(price * (1.015 if direction == "bull" else 0.985), 2)
    stop   = round(price * (0.990 if direction == "bull" else 1.010), 2)
    return {
        "instrument": "stock",
        "side":       side,
        "ticker":     ticker,
        "price":      price,
        "qty":        qty,
        "cost":       cost,
        "target":     target,
        "stop":       stop,
        "risk_reward": "1.5:1 (1.5% target / 1% stop)",
    }


# ── SCAN LOGIC ────────────────────────────────────────────────────────────────

def scan_ticker(ticker: str, alpaca: AlpacaClient, spy_gex) -> dict:
    result = {
        "ticker":     ticker,
        "price":      None,
        "gex":        None,
        "signals":    None,
        "direction":  None,
        "score":      0,
        "instrument": None,
        "alert":      "NO SETUP",
        "error":      None,
    }
    try:
        price = alpaca.get_stock_price(ticker)
        if not price:
            result["alert"] = "DATA ERROR"
            return result
        result["price"] = price

        # Fetch options chain and compute per-ticker GEX
        contracts, snapshots = alpaca.get_gex_chain(ticker, price)
        gex = compute_ticker_gex(price, contracts, snapshots)
        result["gex"] = gex

        if not gex:
            result["alert"] = "THIN CHAIN (no GEX data)"
            return result

        sigs = gex_signals(price, gex, spy_gex)
        result["signals"]   = sigs
        result["direction"] = sigs["direction"]
        result["score"]     = sigs["score"]

        if sigs["direction"] is None:
            bull_hits = sum([sigs["g1_bull"], sigs["g3_bull"], sigs["g4_bull"]])
            bear_hits = sum([sigs["g2_bear"], sigs["g3_bear"], sigs["g4_bear"]])
            result["alert"] = (
                f"NO EDGE  "
                f"put_wall ${gex['put_wall']} ({sigs['pct_above_put']:+.1f}%)  "
                f"call_wall ${gex['call_wall']} ({sigs['pct_below_call']:+.1f}%)  "
                f"flip ${gex['flip_level']} ({sigs['pct_vs_flip']:+.1f}%)  "
                f"bull={bull_hits} bear={bear_hits}"
            )
            return result

        direction = sigs["direction"]

        # Try options first
        opt = find_swing_option(ticker, price, direction, alpaca)
        if opt:
            result["instrument"] = {**opt, "instrument": "option"}
            side = "BUY CALL" if direction == "bull" else "BUY PUT"
            result["alert"] = (
                f"{'BULLISH' if direction == 'bull' else 'BEARISH'} [{sigs['score']}/4]  "
                f"{side} — {opt['symbol']}  "
                f"${opt['mid']}/contract  delta {opt['delta']}  "
                f"{opt['dte']} DTE  IV {opt['iv']}%"
            )
        else:
            stock = stock_fallback(ticker, price, direction)
            result["instrument"] = stock
            result["alert"] = (
                f"{'BULLISH' if direction == 'bull' else 'BEARISH'} [{sigs['score']}/4]  "
                f"{stock['side']} {stock['qty']} shares @ ${price}  "
                f"target ${stock['target']}  stop ${stock['stop']}  "
                f"(options illiquid — stock fallback)"
            )

    except Exception as e:
        result["error"] = str(e)
        result["alert"] = "SCAN ERROR"
        log.error(f"Error scanning {ticker}: {e}", exc_info=True)

    return result


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_scan():
    log.info("=" * 60)
    log.info("SWING SCANNER v2 (GEX + LEVELS) STARTING")
    log.info("=" * 60)

    if not _market_open():
        print("\n  [!] WARNING: Market closed — quotes are stale.\n")

    alpaca  = AlpacaClient()
    results = []

    # SPY GEX macro context via Alpaca (same pipeline as per-ticker)
    print("Fetching SPY GEX levels from Alpaca...")
    spy_gex = None
    try:
        spy_price = alpaca.get_stock_price("SPY")
        if spy_price:
            spy_contracts, spy_snaps = alpaca.get_gex_chain("SPY", spy_price)
            spy_gex = compute_ticker_gex(spy_price, spy_contracts, spy_snaps)
            if spy_gex:
                spy_gex["spot"] = spy_price
                flip_dist = round((spy_price - spy_gex["flip_level"]) / spy_price * 100, 2)
                print(f"\n  SPY @ ${spy_price}")
                print(f"  GEX Regime  : {spy_gex['regime']}")
                print(f"  Net GEX     : {spy_gex['net_gex']:+.4f}B")
                print(f"  Gamma Wall  : ${spy_gex['gamma_wall']}")
                print(f"  Call Wall   : ${spy_gex['call_wall']}  (resistance)")
                print(f"  Put Wall    : ${spy_gex['put_wall']}  (support)")
                print(f"  Flip Level  : ${spy_gex['flip_level']}  ({flip_dist:+.2f}% from spot)")
                print()
            else:
                print("  [!] SPY chain too thin — macro filter disabled\n")
        else:
            print("  [!] Could not fetch SPY price — macro filter disabled\n")
    except Exception as e:
        print(f"  [!] SPY GEX error: {e} — continuing without macro filter\n")
        log.warning(f"SPY GEX scan failed: {e}")

    for ticker in WATCHLIST:
        log.info(f"Scanning {ticker}...")
        r = scan_ticker(ticker, alpaca, spy_gex)
        results.append(r)

        price    = f"${r['price']}" if r["price"] else "N/A"
        gex      = r.get("gex") or {}
        has_gex  = bool(gex)
        has_setup = r["direction"] is not None
        sigs     = r.get("signals") or {}

        print(f"\n{'─'*65}")
        if has_gex:
            regime_tag = f"GEX {gex.get('regime','?'):8s}"
            gwall  = f"gwall ${gex.get('gamma_wall')}"
            cwall  = f"cwall ${gex.get('call_wall')}"
            pwall  = f"pwall ${gex.get('put_wall')}"
            flip   = f"flip ${gex.get('flip_level')} ({sigs.get('pct_vs_flip', 0):+.1f}%)"
            print(f"  {ticker:6s} @ {price:>8}  {regime_tag}  {gwall}  {cwall}  {pwall}  {flip}")
        else:
            print(f"  {ticker:6s} @ {price:>8}  {r['alert']}")

        if has_setup:
            for reason in sigs.get("reasons", []):
                print(f"    * {reason}")
            inst = r.get("instrument")
            if inst and inst.get("instrument") == "option":
                print(f"    => {r['alert']}")
                print(f"       Strike ${inst['strike']}  exp {inst['expiration']}  ({inst['dte']} DTE)")
                print(f"       Bid/Ask ${inst['bid']}/${inst['ask']}  delta {inst['delta']}  IV {inst['iv']}%")
                print(f"       Cost 1 contract: ${inst['cost_1ct']}  ~50% target: ${inst['max_gain_est']}")
            elif inst and inst.get("instrument") == "stock":
                print(f"    => {r['alert']}")
        elif has_gex:
            print(f"    {r['alert']}")

        time.sleep(0.8)

    # ── Summary ───────────────────────────────────────────────────────────────
    setups = [r for r in results if r["direction"] is not None]
    bull   = [r for r in setups if r["direction"] == "bull"]
    bear   = [r for r in setups if r["direction"] == "bear"]
    ranked = sorted(setups, key=lambda r: r["score"], reverse=True)

    print(f"\n{'='*65}")
    print(f"SCAN COMPLETE — {len(WATCHLIST)} tickers")
    print(f"  GEX setups found: {len(setups)}  ({len(bull)} bullish  {len(bear)} bearish)")

    if ranked:
        print(f"\n{'='*65}")
        print("  TOP SETUPS (ranked by signal strength)")
        print(f"{'='*65}")
        for i, r in enumerate(ranked[:8], 1):
            gex   = r.get("gex") or {}
            sigs  = r.get("signals") or {}
            inst  = r.get("instrument") or {}
            arrow = "↑" if r["direction"] == "bull" else "↓"
            tag   = "OPT  " if inst.get("instrument") == "option" else "STOCK"

            print(f"\n  #{i}  {r['ticker']:6s} @ ${r['price']}  {arrow}  score {r['score']}/4"
                  f"  regime={gex.get('regime')}  flip ${gex.get('flip_level')} ({sigs.get('pct_vs_flip', 0):+.1f}%)")
            for reason in sigs.get("reasons", []):
                print(f"       reason: {reason}")
            if inst.get("instrument") == "option":
                print(f"       [{tag}] {'BUY CALL' if r['direction']=='bull' else 'BUY PUT'}"
                      f" {inst.get('symbol')}  ${inst.get('mid')}/ct  delta {inst.get('delta')}"
                      f"  {inst.get('dte')} DTE  IV {inst.get('iv')}%  cost ${inst.get('cost_1ct')}")
            elif inst.get("instrument") == "stock":
                print(f"       [{tag}] {inst.get('side')} {inst.get('qty')} shares"
                      f"  target ${inst.get('target')}  stop ${inst.get('stop')}")

    print(f"\n  Max position size per trade: ${CONFIG['MAX_RISK_PER_TRADE']} (options) or ~$300 (stock)")
    print(f"  Hold 1-2 days. Exit options at 40-60% gain or at 50% loss.")
    print()

    _send_slack_alerts(ranked, spy_gex)

    return results


# ── SLACK ALERTS ──────────────────────────────────────────────────────────────

def _send_slack_alerts(ranked: list, spy_gex) -> None:
    """Post swing scan results to Slack. Silent if SLACK_WEBHOOK_URL not set."""
    webhook = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook:
        log.info("SLACK_WEBHOOK_URL not set — skipping Slack alert")
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f":bar_chart: *Swing Scanner* — {now}"]

    # SPY GEX regime summary
    if spy_gex:
        regime_icon = ":green_circle:" if spy_gex.get("regime") == "POSITIVE" else ":red_circle:"
        lines.append(
            f"\n{regime_icon} *SPY GEX:* {spy_gex.get('regime')}  {spy_gex.get('net_gex', 0):+.4f}B  "
            f"| flip ${spy_gex.get('flip_level')}  "
            f"call wall ${spy_gex.get('call_wall')}  "
            f"put wall ${spy_gex.get('put_wall')}"
        )
    else:
        lines.append("\n_SPY GEX unavailable — macro filter disabled_")

    if not ranked:
        lines.append("\n_No qualifying setups found._")
    else:
        bull_ct = sum(1 for r in ranked if r["direction"] == "bull")
        bear_ct = sum(1 for r in ranked if r["direction"] == "bear")
        lines.append(f"\n*{len(ranked)} setup(s)* — {bull_ct} bullish  {bear_ct} bearish\n")

        for r in ranked:
            gex   = r.get("gex") or {}
            sigs  = r.get("signals") or {}
            inst  = r.get("instrument") or {}
            arrow = ":arrow_up:" if r["direction"] == "bull" else ":arrow_down:"
            side  = "BUY CALL" if r["direction"] == "bull" else "BUY PUT"

            if inst.get("instrument") == "option":
                entry = (
                    f"`{side}` {inst['symbol']}  "
                    f"${inst['mid']}/ct  delta {inst['delta']}  "
                    f"{inst['dte']} DTE  IV {inst['iv']}%  cost ${inst['cost_1ct']}"
                )
            else:
                entry = (
                    f"`{inst.get('side')} {inst.get('qty')} shares`  "
                    f"target ${inst.get('target')}  stop ${inst.get('stop')}  _(options illiquid)_"
                )

            reasons_txt = "  |  ".join(sigs.get("reasons", []))
            lines.append(
                f"{arrow} *{r['ticker']}* @ ${r['price']}  score {r['score']}/4  "
                f"regime={gex.get('regime')}  flip ${gex.get('flip_level')} ({sigs.get('pct_vs_flip', 0):+.1f}%)\n"
                f">  {entry}\n"
                f">  _{reasons_txt}_"
            )

    payload = {"text": "\n".join(lines)}
    try:
        r = requests.post(webhook, json=payload, verify=_SSL_VERIFY, timeout=10)
        if r.status_code == 200:
            log.info(f"Slack alert sent ({len(ranked)} setup(s))")
        else:
            log.warning(f"Slack returned {r.status_code}: {r.text}")
    except Exception as e:
        log.warning(f"Slack alert failed: {e}")


if __name__ == "__main__":
    run_scan()
