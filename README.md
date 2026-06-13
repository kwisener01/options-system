# Alpaca Algorithmic Trader

Live options and equity trading system built on Alpaca Markets, deployed on Render, with Slack alerts.

## What It Does

Runs automated scans every trading day, fires Slack alerts only when actionable setups appear, and provides a web dashboard for GEX (Gamma Exposure) analysis. Supports options spread entry/exit from the command line.

---

## Architecture

```
app.py              Flask web server + APScheduler (single process)
├── /                Dashboard HTML (GEX levels, positions, market status)
├── /api/gex         Live GEX JSON (5-min cache)
├── /slack/command   Slack slash commands (/scan, /positions, /gex, etc.)
├── /health          Render health check (pinged by UptimeRobot every 5 min)
│
├── 8:30 AM ET     → _premarket_prep_job()      Pre-market brief (cached GEX + prices)
├── 9:45 AM ET     → _unified_scan_job()        All scanners in parallel
├── 10:00 AM ET    → _unified_scan_job()
├── 10:30 AM ET    → _unified_scan_job()
├── 12:30 PM ET    → _unified_scan_job()
├── every 5 min    → _spy_trade_monitor_job()   SPY trade + rotation (alert on CHANGE only)
├── 3:30 PM ET     → _manage_5050_job()          Flag open structures at 50% profit target
└── 4:05 PM ET     → _eod_report_job()           EOD P&L + position summary

src/
├── analysis/
│   ├── gex_scanner.py        GEX computation (Gamma × OI × 100 × Spot²)
│   ├── bull_put_scanner.py   Bull put credit spread scanner
│   ├── fallen_angel_scanner.py
│   ├── value_watchlist.py
│   ├── bwb_analyzer.py       Broken Wing Butterfly analysis
│   ├── bwb_scanner.py        Multi-ticker BWB watchlist screen
│   ├── butterfly_scanner.py  GEX-pinned long butterfly (positive-gamma pin play)
│   └── condor_scanner.py     GEX-anchored iron condor (high-POP premium play)
├── live/
│   ├── alpaca_options.py     Options chain fetch, GEX chain cache, order placement
│   └── leading_indicators.py
├── notifications/
│   └── slack_notifier.py
└── signals/
    └── ml_selector.py        LightGBM stock selector
```

---

## Strategies

### Bull Put Credit Spreads
Scans S&P 500 universe for high-IV stocks with GEX put-wall support. Sells put spreads below the GEX put wall. Filters: credit ≥ $0.30, risk/reward ≥ 1:3, stock above 200-day MA.

### Fallen Angels
Stocks down 20–50% from 52-week high with strong fundamentals (low P/E, high ROE). Mean-reversion candidates for stock purchases.

### Value Watchlist
Pre-defined watchlist of quality companies monitored for entry signals.

### Broken Wing Butterfly (BWB)
Unequal-wing put butterfly on SPX/XSP. Structures analyzed:
- **Standard 1/-2/1**: Small credit, max profit between shorts, defined risk
- **1/-3/2**: SELL 1 / BUY 3 / SELL 2 — bounded downside (S-terms cancel below C)
- **BUY 1 / SELL 2 / BUY 2**: Small debit, profits at middle strike AND on crash below lower longs

### GEX-Pinned Butterfly (`butterfly_scanner.py`)
Symmetric long butterfly (BUY 1 / SELL 2 / BUY 1) centered on the GEX pin level — the mirror of the premium-collection strategies. **Risk = the debit (small); reward = wing − debit (often 5–20×).** Only fires in `POSITIVE_GAMMA` regime (dealers suppress vol toward a strike into expiry), centered on the gamma wall / call wall / flip level. Filters: debit ≤ $2.50, R/R ≥ 4:1, body within 2.5% of spot. Stands down in negative/unknown gamma — there's no pin to target. `/fly` runs it on demand.

### GEX-Anchored Iron Condor (`condor_scanner.py`)
The high-win-percentage premium play and the mirror of the butterfly. Sells an OTM put spread + OTM call spread; keeps the full credit if price finishes between the shorts. **Delta = the win-rate dial:** shorts targeted to ~16Δ (≈84%/side) or ~10Δ (≈90%/side). Only fires in `POSITIVE_GAMMA` (price pins between the walls), shorts anchored to the GEX put/call walls (or the delta target, whichever is further OTM). VIX bands flag thin-premium days (soft floor at VIX 11). Reports POP, breakevens, R/R, and a "manage at 50%" reminder to cut the gamma tail. `/condor` runs it on demand.

**Strategy barbell:** butterfly = low-win / high-reward (pin bet); condor = high-win / modest-reward (premium). High win % ≠ high expectancy — the edge is selling when IV > realized, wall-anchored strikes, and managing winners early.

### GEX (Gamma Exposure)
Computed as `Gamma × OI × 100 × Spot²`, **calls positive / puts negative** (SqueezeMetrics convention). `net_gex > 0` → POSITIVE_GAMMA (pinning); `< 0` → NEGATIVE_GAMMA (trending). Uses Yahoo Finance OI (high confidence) cached daily; Alpaca greeks/chain as fallback.

**Regime integrity guard (added after the June 2026 bug):** a one-sided chain (missing calls or puts — which the flaky yfinance fetch produced, collapsing `net_gex` to `−put_gex` and pinning the regime to NEGATIVE forever) is now rejected at three layers — `save_chain` won't persist it, `load_chain` won't return it, and `compute_exposures` reports `UNKNOWN` instead of a false NEGATIVE. The Alpaca OCC type-parse bug (`symbol[12]` → `symbol[-9]`, which misclassified calls as puts) is also fixed. GEX walls act as support/resistance magnets.

---

## Daily Schedule (ET)

| Time | Job | Description |
|------|-----|-------------|
| 8:30 AM | Pre-market prep | Cached GEX walls + Alpaca prices + positions |
| 9:45 AM | Unified scan | All strategies in parallel |
| 10:00 AM | Unified scan | |
| 10:30 AM | Unified scan | |
| 12:30 PM | Unified scan | Midday check |
| every 5 min | SPY trade monitor | High-prob SPY trade + stock rotation |
| 3:30 PM | Profit check | Flags open structures at the 50% profit target |
| 4:05 PM | EOD report | P&L + positions + close suggestions |

### Manage-at-50% (3:30 PM ET)

Captures the win-rate edge so you don't hold short premium into the closing gamma tail. Groups open option legs by (underlying, expiry) and flags any structure that has hit its target — **50% of credit** for credit structures (condors, BWBs, spreads), **50% of max profit** for long butterflies (max profit reconstructed from the leg strikes). Silent if nothing qualifies. Each flag includes the exact close command. `/manage` runs it on demand.

Unified scan is **silent if nothing actionable** — no noise.

### SPY Trade Monitor (every 5 min)

Runs every 5 minutes during regular trading hours (market-open guarded via Alpaca clock). Three independent, **change-gated** alerts — silent unless something changes:

1. **SPY options trade** — recomputes the recommended structure from live GEX + VIX. Fires only when the trade passes a **HIGH-probability** quality gate *and* the structure/strikes differ from the last alert. Quality gate scores: credit/positive-theta, BWB rating (A+/Acceptable), R/R ≥ 0.25, and short strike ≥ 0.5% OTM (POP proxy). Needs ≥ 3 to alert.
2. **GEX-pinned butterfly** — on positive-gamma days, the best pin fly; fires when its body/wing/expiry changes.
3. **GEX-anchored condor** — on positive-gamma days, the delta-targeted condor; fires when the short put/call strikes change.
4. **Stock rotation** — checks equity holdings; if the weakest holding (lowest trailing daily return) has a reason to exit (below 20-day trend / negative momentum / losing position) *and* a STRONG/WATCH watchlist candidate out-gains it by ≥ 0.30 %/day, it suggests a sell→buy swap. Fires only when the sell→buy pair changes.

State persists to `data/spy_trade_state.json` so a Render restart doesn't re-fire the same trade. Tunables: `ROTATION_EDGE_PCT`, `ROTATION_LOOKBACK` in `app.py`.

---

## Setup

### 1. Environment Variables

Create a `.env` file (never commit this):

```
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_BASE_URL=https://api.alpaca.markets      # live
# ALPACA_BASE_URL=https://paper-api.alpaca.markets  # paper

SLACK_WEBHOOK_URL=https://hooks.slack.com/...
SLACK_SIGNING_SECRET=your_signing_secret
```

### 2. Install

```bash
pip install -r requirements.txt
```

### 3. Run locally

```bash
python app.py
# or
gunicorn --workers 1 --threads 4 app:app
```

Dashboard at `http://localhost:5000`

---

## Opening & Closing Trades

### Place a spread
```bash
python place_spread.py
```

### Close all legs for a ticker (multi-leg simultaneous order)
```bash
python close_bwb.py --ticker INTC          # live
python close_bwb.py --ticker INTC --dry-run # preview only
```

**Important:** All legs are submitted as a single multi-leg order. Submitting legs individually causes Alpaca to misidentify SELL_TO_CLOSE as new naked shorts.

---

## Deployment (Render)

- **Service type:** Web Service (free tier keeps alive via UptimeRobot)
- **Build command:** `pip install -r requirements.txt`
- **Start command:** `gunicorn --workers 1 --threads 4 app:app`
- **Single process required** — APScheduler BackgroundScheduler runs inside the app process
- UptimeRobot pings `/health` every 5 minutes to prevent sleep

Set all `.env` variables in Render's Environment dashboard. No redeploy needed for config changes.

---

## Slack Commands

| Command | Description |
|---------|-------------|
| `/scan` | Run unified scan now |
| `/spy` | Current SPY trade signal + stock-rotation check |
| `/fly` | GEX-pinned butterfly (positive-gamma pin play) |
| `/condor` | GEX-anchored iron condor (high-POP premium play) |
| `/manage` | Check open structures at the 50% profit target |
| `/positions` | Current open positions |
| `/place TICKER SHORT LONG EXPIRY [QTY]` | Place a bull put credit spread |
| `/close_position TICKER` | Market-sell a stock/ETF position |
| `/eod` | EOD report |
| `/help` | Show all commands |

---

## Key Files

| File | Purpose |
|------|---------|
| `app.py` | Main Flask app, scheduler, all Slack logic |
| `close_bwb.py` | Generic multi-leg close for any ticker |
| `place_spread.py` | Interactive spread entry |
| `download_gex_chain.py` | Download + cache daily options chain for GEX |
| `src/live/alpaca_options.py` | Alpaca options chain fetch, order helpers |
| `src/analysis/gex_scanner.py` | GEX computation + wall detection |
| `config/settings.py` | All tunable parameters |

---

## GEX Notes

- **Yahoo Finance OI** = real open interest → high-confidence GEX (cached in `data/gex_chain/spy_chain_{date}.pkl`)
- **Alpaca OI** = bid+ask size proxy → low-confidence (used only when Yahoo unavailable)
- **Alpaca greeks** = pre-computed for ~47% of contracts; falls back to Black-Scholes for the rest
- GEX walls remain valid until market open → pre-market brief uses prior-day cached chain

---

## Account

- Broker: Alpaca (live account as of Jun 2026)
- Options buying power: ~$1,930
- XSP preferred over SPXW for defined-risk spreads (1/10 size, European-style, cash-settled)
