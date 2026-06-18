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

The VIX de-risk is the highest-impact rule: in a 10-year backtest it cut max
drawdown from −38% to −14% (2.7×) while *raising* Sharpe — same trades, smaller
size in elevated vol.

Equity / P&L / HWM come from Alpaca (durable). Blocked entries say `🛑 Blocked by
risk rule`. Status: `/risk`.

## 2. One-shot auto-trade (`config/auto_trade.json`)

- Fires **at most one** defined-risk structure on `armed_date`, scanning
  **10:00 AM–12:30 PM ET**; stands down at 12:30 if nothing qualifies.
- Candidate priority **condor → bull put → fly**; max loss = min($100 cap, 5% of equity).
- **No 0DTE**; bull puts skipped on an earnings catalyst.
- Re-priced live at fill; **rejected if drift > 20%**.
- **Exit:** auto-close at **50% profit**, force-close at **1 DTE / 3:30 PM ET**.
- **LIVE only** (`require_live`); idempotent via `client_order_id`.
- Controls: `/autotrade on|off|status`; hard kill `AUTO_TRADE_KILL=1` (Render env).

## 3. Fallen-angel stocks (`config/fallen_angel.json`)

- Act on **STRONG** signals only (score ≥ 8/13), via the 🪂 **Buy** button.
- **Size:** risk **1% of equity** per name; ≤ **15%** of equity per position;
  **max 4** concurrent; total category ≤ **40%** of equity.
- **Stop (broker-side, set on entry):** higher of **−12%** or **just below the
  52-week low** — a held name is never unprotected.
- **Exit:** trim half at **+20%**, then a **15% trailing stop** on the remainder.
  *(Entry + stop enforced; the trim/trail manager is pending — until then the
  +20% trim is offered as a button by the 3:30 sweep.)*

## 4. Options approvals & end-of-day management

- Scan/manage alerts post **Take / Close** buttons; a tap re-prices live (20%
  drift guard), is risk-gated, and approvals expire after **30 min**.
- **3:30 PM sweep:** 50% profit targets on structures, plus stock **trim at +20%
  / stop at −8%** and **options ≤ 3 DTE** review — each with a button.
- **4:05 PM:** post-close P&L recap only (no actions — market closed).

## 5. Allocation targets — *guidance, not yet auto-enforced*

- **~40% stocks / ~25% options-at-risk / ~35% cash.** Never exceed ~50% buying-
  power usage. Cash is ammunition for adjustments, assignment, and dips.

## 6. Operating discipline

- **Outside capital = regulated.** These rules assume **own capital only**.
  Managing other people's money requires registration (RIA/CPO) + legal counsel.
- **Reporting:** `/risk`, `/performance` (`/perf`), monthly NAV statement (1st,
  8 AM ET), EOD recap.
- **Uptime:** UptimeRobot + a second pinger keep the scheduler alive during market
  hours — required for the auto-exits to fire.

---

## Not yet enforced (honest gaps)

1. Fallen-angel **trim/trail exit** automation (entry stop *is* set).
2. **Allocation caps** in §5 (targets only).
3. **No validated backtest** of the live combined system — existing backtests
   cover related-but-different strategy variants and model option P&L from price
   moves, not real historical chains.
