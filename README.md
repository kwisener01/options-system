# Alpaca Algorithmic Trader

Live options and equity trading system built on Alpaca Markets, deployed on Render, with Slack alerts.

## What It Does

Runs automated scans every trading day, fires Slack alerts only when actionable setups appear, and provides a web dashboard for GEX (Gamma Exposure) analysis. Supports options spread entry/exit from the command line.

---

## Architecture

```
app.py              Flask web server + APScheduler (single process)
├── /                Dashboard HTML (GEX levels, positions, market status, news)
├── /api/gex         Live GEX JSON (5-min cache)
├── /api/news        Recent scored headlines for holdings + SPY (5-min cache)
├── /api/expected-move  Straddle-based expected move for a chosen DTE (0DTE supported)
├── /slack/command   Slack slash commands (/scan, /positions, /gex, etc.)
├── /slack/interactive  Trade approval button taps (Take/Skip, Close/Hold)
├── /health          Render health check (pinged by UptimeRobot every 5 min)
│
├── 8:30 AM ET     → _premarket_prep_job()      Pre-market brief (cached GEX + prices)
├── 9:45 AM ET     → _unified_scan_job()        All scanners in parallel
├── 10:00 AM ET    → _unified_scan_job()
├── 10:30 AM ET    → _unified_scan_job()
├── 12:30 PM ET    → _unified_scan_job()
├── every 5 min    → _spy_trade_monitor_job()   SPY trade + rotation (alert on CHANGE only)
├── 3:30 PM ET     → _manage_5050_job()          Actionable sweep: 50% targets + stock trim/stop + expiry (buttons)
├── 4:05 PM ET     → _eod_report_job()           Post-close P&L recap (no actions — market's closed)
├── 10:00-12:30    → _auto_trade_job()           Recurring auto-trade (one defined-risk trade/market day, ≤ max_loss)
└── every 5 min    → _auto_manage_job()          Auto-exit the auto-trade: 50% profit / 1-DTE force-close

src/
├── analysis/
│   ├── gex_scanner.py        GEX computation (Gamma × OI × 100 × Spot²)
│   ├── bull_put_scanner.py   Bull put credit spread scanner
│   ├── fallen_angel_scanner.py
│   ├── value_watchlist.py
│   ├── bwb_analyzer.py       Broken Wing Butterfly analysis
│   ├── bwb_scanner.py        Multi-ticker BWB watchlist screen
│   ├── butterfly_scanner.py  GEX-pinned long butterfly (positive-gamma pin play)
│   ├── condor_scanner.py     GEX-anchored iron condor (high-POP premium play)
│   └── batman_scanner.py     GEX-anchored Batman / double BWB for XSP (positive-cowl)
├── live/
│   ├── alpaca_options.py     Options chain fetch, GEX chain cache, order placement
│   ├── news.py               Alpaca news feed: earnings guard + headline sentiment
│   └── leading_indicators.py
├── notifications/
│   ├── slack_notifier.py     Outbound Slack (text + Block Kit button messages)
│   ├── slack_blocks.py       Block Kit builders for Take/Skip & Close/Hold
│   └── pending_store.py      JSON-backed pending-trade store (approval TTL/state)
└── signals/
    └── ml_selector.py        LightGBM stock selector
```

---

## Strategies

### Bull Put Credit Spreads
Scans S&P 500 universe for high-IV stocks with GEX put-wall support. Sells put spreads below the GEX put wall. Filters: credit ≥ $0.30, risk/reward ≥ 1:3, stock above 200-day MA.

### Fallen Angels
Stocks down 30%+ from 52-week high, near their multi-year floor, with early institutional-accumulation signals (RSI bottoming, volume expansion, call-heavy options flow). 13-point screen; STRONG ≥ 8, WATCH ≥ 5.

**Trading rules (`config/fallen_angel.json`):** STRONG signals post a Slack **🪂 Buy** button. A tap:
- **risk-sizes** the position — shares = `1% of equity ÷ (entry − stop)`, capped at 15% of equity per name, `max_positions` (4) concurrent, and total category exposure ≤ 40% of equity (the stock sleeve);
- buys, then places a **protective sell-stop** at the broker — stop = the higher (first-triggered) of **−12%** or **just below the 52-week low**, so a held name is never unprotected.

**Exit** (automatic, `_fa_manage_job` every 15 min RTH): at +20%, sell half and place a 15% trailing stop on the remainder — the trailing stop goes on *before* the half is sold, so the kept shares are never unprotected.

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

### GEX-Anchored Batman (`batman_scanner.py`)
Double broken-wing butterfly = a put BWB (left ear) + a call BWB (right ear) sharing one expiry, ears anchored to the GEX put/call walls. **Key constraint: positive cowl.** When both ears are OTM, P&L at spot = the net credit, so a *positive cowl* (profit even if price sits at spot) means the whole structure is a *net credit* — which broken wings make possible but symmetric butterflies never do. The scanner searches inner/outer wing widths to find a positive-cowl structure with defined tail risk (≤ $1,500), only in `POSITIVE_GAMMA`. Priced off the SPY chain (SPY ≈ XSP numerically) and **presented for XSP execution** at a European/cash-settled broker (no assignment, no pin risk — XSP isn't on Alpaca). `/batman` runs it on demand.

**Strategy barbell:** butterfly = low-win / high-reward (pin bet); condor = high-win / modest-reward (premium). High win % ≠ high expectancy — the edge is selling when IV > realized, wall-anchored strikes, and managing winners early.

### Expected Move (`expected_move.py`)
The market's implied ±range by a chosen expiry, from the **ATM straddle** (the most accurate method — it bakes in the real IV/skew at that expiry): `EM ≈ 0.85 × (ATM call + ATM put)`. The ATM strike is found where |call − put| is smallest (parity), so 0DTE works naturally (the straddle reflects remaining time value). Shown in the **pre-market Slack brief** (SPY 0DTE + weekly) and on the **dashboard** with a DTE selector (`/api/expected-move?symbol=SPY&dte=N`). Note: the 1-SD expected move ≈ the 16Δ strike — so "sell outside the expected move" and "sell at 16Δ" are the same trade.

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
| 3:30 PM | Position check | 50% profit targets + stock trim/stop-loss + near-expiry options, each with a Close/Trim button (market still open) |
| 4:05 PM | EOD report | Post-close P&L + position recap only — actions already fired at 3:30 |
| 10:00 AM–12:30 PM | Auto-trade | Recurring: one defined-risk entry per market day; stands down at 12:30 if nothing qualifies |
| every 5 min | Auto-manage | Auto-exits the auto-trade at 50% profit or force-closes at 1 DTE / 3:30 PM |

### 3:30 PM Position Check (was Manage-at-50%)

The end-of-day **actionable** sweep, run while the market is still open so every flag can actually be traded before the close (the 4:05 EOD report is post-close, so its old suggestions were useless — they moved here). It flags two things, each with a Close/Trim button:

1. **Profit targets** — groups open option legs by (underlying, expiry) and flags any structure that hit its target: **50% of credit** for credit structures (condors, BWBs, spreads), **50% of max profit** for long butterflies (reconstructed from leg strikes). Captures the win-rate edge so you don't hold short premium into the closing gamma tail.
2. **Position adjustments** — stock **down ≤ −8%** → CLOSE (stop-loss), stock **up ≥ +20%** → TRIM (sells half), option **≤ 3 DTE** → REVIEW (close structure or let expire).

Silent if nothing qualifies. `/manage` runs the profit-target half on demand.

Unified scan is **silent if nothing actionable** — no noise.

### Performance Reporting (fund metrics)

Fund-grade metrics computed from Alpaca's **durable** daily equity series (no local storage — always reconciles to the broker, survives restarts):

- **Returns:** total, ~30-day, and annualized **CAGR**.
- **Risk-adjusted:** **max drawdown**, **Sharpe**, **Sortino**.
- **Consistency:** daily **win rate**, **profit factor**, best/worst day.

`/performance` (alias `/perf`) shows it on demand; a **Monthly NAV statement** posts to Slack on the 1st of each month at 8 AM ET. Short-history figures are flagged as noisy. Per-trade and per-strategy attribution is the next layer (tag `client_order_id` by strategy + reconcile Alpaca fills).

### Risk Engine (config/risk.json)

Fund-style capital protection that gates **every** new entry — the auto-trade *and* the Slack Take buttons. Equity, today's P&L, and the high-water mark all come from Alpaca, so the rules are durable and restart-proof.

- **%-of-equity sizing** — the auto-trade's max-loss budget is `risk_pct_per_trade × equity` (default 5%), so position size scales as the account grows. The committed `$` cap still applies; effective risk = the tighter of the two.
- **Daily-loss circuit breaker** — blocks new entries once today's P&L is down `daily_loss_limit_pct` (default 5%) vs. yesterday's close.
- **Drawdown guard** — blocks new entries while equity is `max_drawdown_pct` (default 15%) below its high-water mark.
- **VIX de-risk** — half size at VIX ≥ `vix_half_size` (18), full stand-down ≥ `vix_stand_down` (25). Backtested (10y) to cut max drawdown −38% → −14% while raising Sharpe.
- **Allocation caps** — **cash floor** (≥ `cash_floor_pct`, 30%) blocks all new entries when dry powder runs low; **stock-sleeve cap** (≤ `stock_cap_pct`, 40%) caps fallen-angel exposure. Their sum leaves ~25% for options. `/allocation` shows the live split.
- **IV-vs-RV gate** — the auto-trade sells premium (condor / bull put) only when **VIX > SPY 20-day realized vol** by ≥ `vrp_min_points` (0), i.e. only when implied richness pays for the risk; the long pin fly is exempt (it benefits from cheap premium). The variance risk premium is the actual edge behind premium selling. `/risk` shows the live spread.

A blocked button entry replies `🛑 Blocked by risk rule — …`; the auto-trade posts a halt and stands down. `/risk` shows live status (equity, day P&L vs limit, drawdown, per-trade budget). Defaults are starting values for a small account — tune in `config/risk.json`.

### Recurring Auto-Trade (config/auto_trade.json)

Places **one** defined-risk structure automatically per day — no button tap. Guardrails:

- **Cadence.** `recurring: true` → every market day (one trade/day, idempotent). `recurring: false` → one-shot on `armed_date` (or env `AUTO_TRADE_DATE`) only. Committed config survives Render restarts.
- **Window:** re-checks every 5 min from **10:00 AM** and places the first qualifying setup; **stands down at 12:30** if nothing clears the gate.
- **Best passing candidate**, priority **condor → bull put → fly**, capped at `max_loss` (default **$100**), **no 0DTE**, bull puts skipped on an earnings catalyst (news guard).
- **Re-priced + drift-guarded at fill** (same 20% gate as the buttons) and **idempotent** via Alpaca `client_order_id` (`auto-YYYYMMDD`) — a restart can't double-fire.
- **Automatic exit** (`_auto_manage_job`, every 5 min): auto-closes at the **50% profit target**, and **force-closes at 1 DTE, 3:30 PM ET** so it never holds into expiration. The open structure is re-identified from Alpaca each tick (durable across restarts); only the auto-trade's own legs are closed.
- **Kill switch:** env `AUTO_TRADE_KILL=1` (or `enabled:false`) disables it before it fires. Every action is posted to Slack.

To re-arm another day, change `armed_date` (commit/redeploy) or set `AUTO_TRADE_DATE` in Render.

### Interactive Trade Approvals (Slack buttons)

Scans and the 3:30 manage job don't just *show* trades — they ask. After the
normal alert, a follow-up message offers a button per candidate:

- **Entry** — `✅ Take #N` / `✖ Skip`, numbered when there are several. Tapping
  *Take* re-prices the structure off **live** quotes, rejects it if the net
  credit/debit has drifted more than **20%** against you (or the alert has gone
  stale — 30-min TTL), and only then submits. So a tap is a confirmation, not a
  blind fire of a stale price.
- **Exit** — `💰 Close #N` / `✋ Hold` on each structure flagged at the 50%
  target; *Close* submits a single multi-leg closing order (same logic as
  `close_bwb.py`).

Pending approvals live in `data/pending_trades.json` so a Render restart doesn't
lose them. Auto-executable structures: bull put spreads, BWBs, GEX pin
butterflies, and iron condors. Batman is XSP-only (not on Alpaca), so it stays
informational.

**One-time Slack setup:** in api.slack.com → your app → *Interactivity &
Shortcuts* → turn on and set the Request URL to
`https://<your-render-app>.onrender.com/slack/interactive`. Clicks are verified
with the same `SLACK_SIGNING_SECRET` the slash commands already use — no new
token or secret required.

### News & Catalyst Awareness

Recent headlines (Alpaca's Benzinga news feed — included with your existing API
keys, no new secret) are surfaced two ways:

- **Analysis guard** — equity premium plays (bull put, BWB) are checked for an
  earnings/guidance catalyst in recent headlines; a hit annotates the alert with
  a `⚠️ news` warning so you don't sell defined-risk premium blind into a print.
- **Dashboard** — a *📰 News & Catalysts* panel shows scored (🟢/🔴/⚪)
  headlines for SPY + your holdings, served by `/api/news` (5-min cache).

### SPY Trade Monitor (every 5 min)

Runs every 5 minutes during regular trading hours (market-open guarded via Alpaca clock). Three independent, **change-gated** alerts — silent unless something changes:

1. **SPY options trade** — recomputes the recommended structure from live GEX + VIX. Fires only when the trade passes a **HIGH-probability** quality gate *and* the structure/strikes differ from the last alert. Quality gate scores: credit/positive-theta, BWB rating (A+/Acceptable), R/R ≥ 0.25, and short strike ≥ 0.5% OTM (POP proxy). Needs ≥ 3 to alert.
2. **GEX-pinned butterfly** — on positive-gamma days, the best pin fly; fires when its body/wing/expiry changes.
3. **GEX-anchored condor** — on positive-gamma days, the delta-targeted condor; fires when the short put/call strikes change.
4. **GEX-anchored Batman** — on positive-gamma days, the positive-cowl XSP double BWB; fires when the ears/outer-wing change.
5. **Stock rotation** — checks equity holdings; if the weakest holding (lowest trailing daily return) has a reason to exit (below 20-day trend / negative momentum / losing position) *and* a STRONG/WATCH watchlist candidate out-gains it by ≥ 0.30 %/day, it suggests a sell→buy swap. Fires only when the sell→buy pair changes.

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
| `/batman` | GEX-anchored Batman for XSP (positive-cowl double BWB) |
| `/manage` | Check open structures at the 50% profit target |
| `/autotrade on\|off\|status` | Toggle or inspect the recurring auto-trade (env `AUTO_TRADE_KILL` is the hard override) |
| `/risk` | Fund risk status: equity, day P&L vs limit, drawdown vs high-water mark, per-trade budget |
| `/performance` (`/perf`) | Fund metrics: total/30d/CAGR returns, max drawdown, Sharpe, Sortino, win rate, profit factor |
| `/attribution` (`/attr`) | Realized P&L by strategy — reconciled from Alpaca order tags (which strategy makes money) |
| `/allocation` (`/alloc`) | Stock / option / cash split vs. the cash floor + stock cap |
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
