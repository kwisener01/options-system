# Swing Scanner — Build Log

## What It Does

`swing_scanner.py` scans a watchlist of 22 tickers for 1-2 day directional swing setups
driven entirely by dealer gamma positioning (GEX). For each qualifying setup it finds the
best options contract (call or put, 2-7 DTE, delta ~0.45) or falls back to a stock
recommendation. Results are posted to Slack automatically after every scan.

---

## Architecture

### Signal Engine — GEX + Levels

Signals are computed from the live Alpaca options chain, not technical indicators.
Four signals per ticker (need ≥ 2 aligned to qualify):

| Signal | Bullish trigger | Bearish trigger |
|--------|----------------|-----------------|
| G1 | Price within 1.5% above **put wall** | — |
| G2 | — | Price within 1.5% below **call wall** |
| G3 | Net GEX negative AND price above **flip level** | Net GEX negative AND price below flip level |
| G4 | SPY regime = NEGATIVE (amplifies both directions) | SPY regime = NEGATIVE |

**Key levels computed per ticker:**
- **Put Wall** — strike with highest put GEX below spot (dealer support)
- **Call Wall** — strike with highest call GEX above spot (dealer resistance)
- **Gamma Wall** — strike with largest absolute net GEX near spot (pin level)
- **Flip Level** — where cumulative GEX crosses zero (regime boundary)
- **Net GEX** — positive = pinning/mean-reversion, negative = trending/momentum

### SPY Macro Context

SPY GEX is computed first using the same Alpaca pipeline (not yfinance). The regime
(POSITIVE/NEGATIVE) acts as a market-wide amplifier for G4.

### Instrument Selection

For each setup the scanner:
1. Tries to find a **2-7 DTE call (bull) or put (bear)** — delta 0.35–0.55, bid-ask < 35% of mid, OI ≥ 50
2. Falls back to a **stock recommendation** (qty, target +1.5%, stop -1%) if the chain is illiquid

### GEX Math

```
gamma  = from snapshot greeks, or Black-Scholes if unavailable
GEX    = gamma × OI × 100 × spot² / 1e9   (in $billions)
signed = +GEX for calls, -GEX for puts  (dealer convention)
```

---

## Watchlist

```
SPY  QQQ  IWM  XLF  XLE  XLK
AAPL MSFT NVDA META TSLA AMZN GOOGL AMD
PLTR MARA HOOD SOFI COIN RIVN F BAC
```

---

## Config (swing_scanner.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NEAR_WALL_PCT` | 1.5% | How close to a wall to trigger G1/G2 |
| `FLIP_ZONE_PCT` | 1.0% | Dead zone around flip level |
| `GEX_CHAIN_DTE` | 30 days | DTE range for GEX chain fetch |
| `GEX_CHAIN_RANGE` | ±14% | Strike range for GEX chain |
| `MIN_DTE` / `MAX_DTE` | 2 / 7 | DTE window for swing options |
| `TARGET_DELTA` | 0.45 | Ideal option delta |
| `DELTA_RANGE` | ±0.18 | Acceptable delta band |
| `MAX_RISK_PER_TRADE` | $150 | Max cost per options trade |
| `ACCOUNT_SIZE` | $1,500 | Account size for position sizing |

---

## Slack Output

Every scan posts one message containing:
- SPY GEX regime, net GEX, flip/call/put walls
- Each setup: ticker, direction, score, regime, flip distance
- Option details: symbol, mid price, delta, DTE, IV, cost per contract
- Or stock details: side, qty, target, stop

---

## Automation

**Scheduled routine:** Every weekday at **9:45 AM ET** via Claude Code remote agent.

- Platform: Anthropic CCR (Claude Code Routines)
- Routine ID: `trig_01HwUV21PKasWtaEnDy25RiH`
- Manage: https://claude.ai/code/routines/trig_01HwUV21PKasWtaEnDy25RiH
- Repo: https://github.com/kwisener01/options-system
- Model: Claude Sonnet 4.6

The remote agent:
1. Clones the repo fresh each run
2. `pip install pandas requests scipy numpy python-dotenv certifi`
3. Writes `.env` with Alpaca + Slack credentials
4. Runs `python swing_scanner.py`

---

## Environment Variables Required

| Variable | Purpose |
|----------|---------|
| `ALPACA_API_KEY` | Alpaca brokerage key |
| `ALPACA_SECRET_KEY` | Alpaca secret |
| `ALPACA_BASE_URL` | `https://paper-api.alpaca.markets` (paper) or live URL |
| `SLACK_WEBHOOK_URL` | Incoming webhook for scan alerts |

---

## Known Issues & Fixes Applied

| Issue | Fix |
|-------|-----|
| yfinance fails after market close | Replaced SPY GEX with Alpaca chain pipeline — no yfinance dependency |
| IEX feed returns 403 in remote environment | `get_stock_price` now tries `sip` → `iex` → no feed in sequence |
| Slack SSL failure in remote environment | Slack POST now uses `certifi.where()` explicitly |
| Windows SSL cert store issues locally | `_best_ssl_verify()` probes certifi → system → disabled with warning |

---

## Files

| File | Purpose |
|------|---------|
| `swing_scanner.py` | Main scanner — GEX engine, options finder, Slack alerts |
| `bull_put_screener.py` | Separate income strategy scanner (21-45 DTE credit spreads) |
| `src/analysis/gex_scanner.py` | SPY-specific GEX scanner using yfinance (used by other modules) |
| `src/analysis/gex_trader.py` | GEX-based intraday signal generator for SPY |

---

## How to Run Manually

```bash
python swing_scanner.py
```

Takes ~3-4 minutes for the full 22-ticker scan. Slack alert fires automatically at the end.
Best run between 9:45–11:00 AM ET when options markets are liquid.
