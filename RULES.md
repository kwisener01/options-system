# Trading Rules — Investment Policy Statement

The operating rules for this **live** Alpaca account. Each rule is implemented by a
config file and enforced in `app.py`. To change a rule: edit it here **and** the
matching config, then deploy.

> Account mode is confirmed at `GET /api/mode` and shown in `/risk`. The live
> endpoint is `api.alpaca.markets`.

---

## 1. Capital protection — every entry (`config/risk.json`)

Gates the auto-trade, the options Take buttons, and fallen-angel buys alike.

| Rule | Default | Effect |
|------|---------|--------|
| Per-trade risk budget | 5% of equity | Caps each trade's max loss |
| Daily-loss circuit breaker | −5% on the day | Blocks **new** entries the rest of the day |
| Drawdown guard | 15% below high-water mark | Blocks **new** entries until recovered |
| VIX de-risk | ½ size at VIX ≥ 18, stand down ≥ 25 | Shrinks/halts premium-selling as vol rises |
| IV-vs-RV gate | sell premium only when VIX − SPY-realized-vol ≥ `vrp_min_points` (0) | No condor/bull-put when premium is cheap vs actual movement; the long fly is exempt |

The VIX de-risk is the highest-impact rule: in a 10-year backtest it cut max
drawdown from −38% to −14% (2.7×) while *raising* Sharpe — same trades, smaller
size in elevated vol.

Equity / P&L / HWM come from Alpaca (durable). Blocked entries say `🛑 Blocked by
risk rule`. Status: `/risk`.

## 2. Recurring auto-trade (`config/auto_trade.json`)

- `recurring: true` → fires **one** defined-risk structure **every market day**
  (idempotent per day), scanning **10:00 AM–12:30 PM ET**; stands down at 12:30
  if nothing qualifies. (`recurring: false` → one-shot on `armed_date` only.)
- Candidate priority **condor → bull put → fly**; max loss = min($100 cap, 5% of equity).
- **No 0DTE**; bull puts skipped on an earnings catalyst.
- Re-priced live at fill; **rejected if drift > 20%**.
- **Exit:** auto-close at **50% profit**, force-close at **1 DTE / 3:30 PM ET**.
- **LIVE only** (`require_live`); idempotent via `client_order_id`.
- Controls: `/autotrade on|off|status`; hard kill `AUTO_TRADE_KILL=1` (Render env).

## 2b. 0DTE put spread at the expected move (`config/zerodte.json`, OFF by default)

Opt-in tastylive-style strategy — **`enabled: false`** until you flip it on.
- Each market day in **10:00–11:30 ET**, sells one **SPY 0DTE put credit spread**,
  short strike ~**1 expected-move below the forward** (from `expected_move.py`),
  **$1-wide** (≈$100 risk).
- Gated by the **risk engine** (5%/VIX/cash-floor/drawdown) **and the IV>RV gate**.
- **Manage at 50% profit; do NOT close losers early; FORCE-CLOSE by 15:45 ET** —
  SPY is American/physically settled, so a 0DTE short put left ITM at expiry gets
  assigned 100 shares (~$74k), which a small account can't hold.
- SPY (not SPX/XSP) because Alpaca has no index options. Short gamma → respect the
  fat tail. Untested against live order mechanics — watch the first enabled run.

## 3. Fallen-angel stocks (`config/fallen_angel.json`)

- Act on **STRONG** signals only (score ≥ 8/13), via the 🪂 **Buy** button.
- **Size:** risk **1% of equity** per name; ≤ **15%** of equity per position;
  **max 4** concurrent; total category ≤ **40%** of equity.
- **Stop (broker-side, set on entry):** higher of **−12%** or **just below the
  52-week low** — a held name is never unprotected.
- **Exit (automatic, every 15 min RTH):** at **+20%**, sell half and put a **15%
  trailing stop** on the remainder. The trailing stop is placed *before* the
  half is sold, so the kept shares are never unprotected.

## 4. Options approvals & end-of-day management

- Scan/manage alerts post **Take / Close** buttons; a tap re-prices live (20%
  drift guard), is risk-gated, and approvals expire after **30 min**.
- **3:30 PM sweep:** 50% profit targets on structures, plus stock **trim at +20%
  / stop at −8%** and **options ≤ 3 DTE** review — each with a button.
- **4:05 PM:** post-close P&L recap only (no actions — market closed).

## 5. Allocation caps (`config/risk.json`)

Target mix **~40% stocks / ~25% options-at-risk / ~35% cash**. Enforced by two
exactly-computable caps (their sum *leaves* ≤30% for options, so the mix falls
out — no flaky "options-at-risk" gauge needed):

| Cap | Default | Effect |
|-----|---------|--------|
| **Cash floor** | ≥ 30% of equity | Blocks **all** new entries when cash is too low — the master "keep dry powder" rule |
| **Stock-sleeve cap** | ≤ 40% of equity | Blocks fallen-angel buys that would push total stock past it |

Cash is ammunition for adjustments, assignment, and dips. `/allocation` shows the
live split vs. the caps.

## 6. Operating discipline

- **Outside capital = regulated.** These rules assume **own capital only**.
  Managing other people's money requires registration (RIA/CPO) + legal counsel.
- **Reporting:** `/risk`, `/performance` (`/perf`), `/attribution` (realized P&L
  by strategy), monthly NAV statement (1st, 8 AM ET), EOD recap.
- **Uptime:** UptimeRobot + a second pinger keep the scheduler alive during market
  hours — required for the auto-exits to fire.

---

## Not yet enforced (honest gaps)

1. **No validated backtest** of the live GEX strategies — backtests cover
   related-but-different variants and model option P&L from price moves, not real
   historical chains. Trust live `/performance` + `/attribution` as the scorecard.
2. **Options-at-risk** is bounded *indirectly* (cash floor + stock cap leave the
   room) rather than measured per-structure — a precise gauge needs max-loss
   reconstruction from chains.
