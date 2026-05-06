# Algorithmic Trading Strategy
**Account:** Alpaca Paper + Prop Firm Futures  
**Updated:** 2026-05-02  
**SPY at time of writing:** $720.67

---

## Overview

Two-layer system using SPY dealer gamma positioning (GEX) as the primary edge:

| Layer | Instrument | Horizon | Goal |
|-------|-----------|---------|------|
| 1 | /ES or /MES futures (prop firm) | Intraday, same-day exit | Daily income from gamma scalps |
| 2 | SPY options (paper / live) | 1-4 weeks | Directional thesis with defined risk |

The same GEX signals drive both layers — futures exploit intraday dealer behavior at key levels,
options capture the multi-week structural positioning. Futures trades hedge the options book
intraday without buying separate protection.

---

## Core Concepts

### Gamma Exposure (GEX)
`GEX = sum(gamma_call x OI x 100 x spot^2) - sum(gamma_put x OI x 100 x spot^2)`

- **Positive GEX** — dealers are long gamma, they sell rips and buy dips, vol suppression, price pins between walls
- **Negative GEX** — dealers are short gamma, they chase moves, vol amplification, trending behavior

### Key Levels

| Level | Definition | Role |
|-------|-----------|------|
| Gamma wall | Strike with highest positive GEX near spot | Magnetic pin — price gravitates here |
| Call wall | Highest call GEX above spot | Dealer resistance — they sell into it |
| Put wall | Highest put GEX below spot | Dealer support — they buy into it |
| Flip level | Where cumulative GEX crosses zero | Regime boundary — break = regime change |

### Vanna
`Vanna = dDelta/dVol`

- VIX falling + positive vanna: dealer buying pressure (bullish intraday tailwind)
- VIX rising + positive vanna: dealer selling pressure (bearish intraday headwind)
- **Rule:** Never fade against vanna. Bearish vanna = no longs. Bullish vanna = no shorts.

### Charm
`Charm = dDelta/dTime` — delta decay forces dealers to unwind hedges near expiry.

- On put-heavy Fridays: OTM puts decay, dealers buy back hedges, AM buying pressure
- Near 0DTE: gamma explodes near strikes, avoid selling into the close

---

## Layer 1: Futures Day Trading (Prop Firm)

### Daily Routine

```
9:30 AM       Slack battle plan arrives — identify regime and key /ES levels
9:45 AM       First 15-min candle closes — directional bias confirmed
9:45-10:30    Best trading window (charm flows clear, vanna active)
12:00-2:00    Avoid — low volume, no dealer hedging activity
3:00-4:00     Power hour — second window (charm forces 0DTE pinning)
3:45 PM       Exit all positions — never hold into close
```

### Setup A: Positive Gamma — Fade the Walls

**Conditions:** GEX regime = POSITIVE_GAMMA, vanna not against direction.

**Long at put wall:**
- Entry: /ES touches put wall zone (within 2-3 pts), 1-min rejection candle forms
- Stop: 3 pts below put wall
- Target: gamma wall
- Hold: 30-90 min max

**Short at call wall:**
- Entry: /ES touches call wall zone (within 2-3 pts), 1-min rejection candle forms
- Stop: 3 pts above call wall
- Target: gamma wall
- Hold: 30-90 min max

Dealers mechanically buy dips and sell rips at these levels. Price snaps back within
15-30 minutes or the level is breaking — if it doesn't bounce, exit flat.

### Setup B: Negative Gamma — Breakout/Momentum

**Conditions:** GEX regime = NEGATIVE_GAMMA, vanna confirms direction.

**Long above flip level:**
- Entry: /ES closes a 5-min candle above flip level
- Stop: 3 pts below flip level
- Target: call wall
- Hold: follow the move, trail stop

**Short below flip level:**
- Entry: /ES closes a 5-min candle below flip level
- Stop: 3 pts above flip level
- Target: put wall
- Hold: follow the move, trail stop

Dealers amplify moves in negative gamma — once the flip breaks, they chase.
Let the position run further than in positive gamma.

### Sizing by Prop Firm Phase

| Phase | Contract | $/point | 3-pt stop risk |
|-------|---------|---------|---------------|
| Challenge | /MES (Micro E-mini) | $5 | $15/trade |
| Verification | /MES, 2 contracts | $5 | $30/trade |
| Funded | /ES (E-mini) | $50 | $150/trade |

**Rule:** Never risk more than 1% of the prop account drawdown limit per trade.  
**Daily cutoff:** Stop trading after 2 consecutive losses. Edge comes from selectivity, not frequency.

### /ES Level Conversion

`/ES price is approximately SPY x 10`

| SPY Level | /ES Equivalent | Role |
|-----------|---------------|------|
| $700 | ~7000 | Deep support |
| $710 | ~7100 | Put wall / structural floor |
| $720 | ~7200 | Current gamma wall / flip zone |
| $725 | ~7250 | Call wall / hard ceiling |
| $730 | ~7300 | Secondary resistance / breakout target |

---

## Layer 2: Multi-Week Options (SPY)

All positions use defined-risk spreads only — no naked options.

### Setup A: Iron Condor at the GEX Pin

**When to use:** Near-term expiry has GEX > +30B. Positive gamma regime confirmed.

```
Sell put spread below put wall     e.g., sell $710P / buy $705P
Sell call spread above call wall   e.g., sell $726C / buy $731C
Same expiry — target the weekly opex with the strongest GEX concentration
```

The +75B GEX on May 8 is the strongest pin in the chain. Dealers will defend the
$710-$726 range mechanically. Max profit if SPY stays inside both spreads.

**Exit rules:**
- Take profit at 50% of max credit
- Close the threatened side if price approaches within 50% of spread width
- Never hold through expiry if the short strike is breached

### Setup B: Bull Put Spread at Sticky Support

**When to use:** Put wall appears in 3+ expiries (confirmed structural support). Positive gamma regime.

```
Sell put at put wall strike   e.g., sell $710P
Buy put 10 pts below          e.g., buy $700P
Expiry: 2-3 weeks out
```

Sticky put walls ($710 and $700 each appear in 3+ expiries currently) have dealer
buying behind them. The spread collects premium with dealer support as a structural backstop.

**Exit rules:**
- Take profit at 50-60% of max credit
- Stop if SPY closes below the put wall for 2 consecutive days

### Setup C: Regime-Change Straddle

**When to use:** Forward chain shows an upcoming flip from positive to negative gamma.

```
Buy ATM call + Buy ATM put
Same strike, same expiry (the first negative-gamma expiry)
Enter 1-2 weeks before the regime flip date
```

The regime change means SPY stops pinning and starts trending. Direction unknown —
you only need the move to be larger than the market is pricing. Current data shows
May 29 flipping to -12.92B. Enter around May 15 when IV is lower.

**Exit rules:**
- Sell the losing leg when the winner doubles
- Hard stop: exit both legs at 40% loss of total premium paid

---

## Forward Chain Structure (2026-05-02)

| Expiry | DTE | Net GEX | Gamma Wall | Call Wall | Put Wall | P/C Ratio |
|--------|-----|---------|-----------|----------|---------|---------|
| 2026-05-04 | 2d | +29.02B | $725 | $725 | $710 | 1.85 |
| 2026-05-08 | 6d | +75.25B | $718 | $726 | $700 | 2.01 |
| 2026-05-15 | 13d | +23.79B | $730 | $730 | $700 | 1.81 |
| 2026-05-22 | 20d | +8.00B | $740 | $740 | $710 | 1.16 |
| 2026-05-29 | 27d | -12.92B | $717 | $730 | $700 | 1.67 |

### Sticky Multi-Expiry Levels (highest conviction)

| Strike | Appearances | Role |
|--------|------------|------|
| $725 | 5 expiries | Call wall / Gamma wall — hard ceiling |
| $720 | 5 expiries | Flip zone / Gamma wall — current inflection |
| $730 | 5 expiries | Secondary call wall — breakout target |
| $710 | 3 expiries | Put wall — structural floor |
| $700 | 3 expiries | Deep support — breakdown level |

### Current Outlook

- Gamma concentrating near-term — strongest at May 8 (+75B), the dominant expiry this week
- Regime consistently POSITIVE through May 22 — mean-reversion bias, dealers defend the range
- **May 29 flips negative** — regime change, moves will amplify after monthly opex rolls off
- Put/Call ratios 1.5-2.0 across all expiries — heavy institutional hedging, underlying bid intact
- Near-term range: **$710-$725** with spot at $720.67 sitting just below the call wall

---

## How the Two Layers Interact

| Scenario | Futures | Options Effect |
|----------|---------|---------------|
| SPY rallies to $725 call wall | Short /MES scalp at $725 | Iron condor call side at risk — futures profit offsets |
| SPY drops to $710 put wall | Long /MES scalp at $710 | Bull put spread safe, futures adds P&L |
| SPY breaks flip level down | Short /MES breakout | Straddle long put profits |
| SPY grinds between walls all day | No futures setup — stay out | Iron condor collects theta passively |

The futures positions hedge the options book intraday. On quiet days with no clear
setup, the options collect theta without any futures exposure.

---

## Risk Management

### Per-trade rules
- Futures: max loss per trade = 1% of prop account drawdown limit
- Options: max position size = 5% of total account per spread
- Never add to a losing position

### Daily rules
- Futures: stop trading after 2 consecutive losses
- Options: no more than 3 open positions simultaneously
- If VIX spikes above 30: close all short premium, no new iron condors

### Weekly rules
- Run `python run_forward_analysis.py` every Monday morning
- Update battle plan levels if sticky strikes shift by more than $5
- Review all open option positions vs current GEX levels — exit any spread where
  the short strike is now inside a sticky level

---

## Automated System Schedule

| Time (ET) | Job | Trigger |
|-----------|-----|---------|
| 9:30 AM Mon-Fri | /ES futures battle plan to Slack | `futures_battle_plan_job` |
| 9:45 AM Mon-Fri | GEX scan + SPY options signal | `gex_scan_and_trade_job` |
| 9:45 AM Fri | Open weekly SPY options spread | `options_open_job` |
| Every 15 min 9:45-3:45 | /ES proximity alerts | `futures_proximity_job` |
| Every 15 min 9:00-3:45 | Equity position monitor | `monitor_job` |
| 3:15 PM Mon-Fri | Afternoon GEX scan + sync | `gex_afternoon_job` |
| 3:45 PM Mon-Thu | Options take-profit check | `options_tp_job` |
| 3:45 PM Mon-Fri | Force-close GEX options trade | `gex_force_close_job` |
| 3:45 PM Fri | Settle expiring spread + TP check | `options_settle_job` |
| 4:05 PM Mon-Fri | ML equity selection + EOD report | `eod_report_job` |

### Manual Scripts

```bash
# Every morning before open
python run_gex_levels.py --slack          # current /ES battle plan to Slack

# Every Monday
python run_forward_analysis.py --slack    # 8-expiry GEX term structure

# After 9:30 AM — track opening option order fills
python monitor_opening_orders.py --watch  # live refresh every 30s

# Download today's chain if auto-cache missed
python download_gex_chain.py

# Backtest GEX strategy on cached historical chains
python run_gex_backtest.py --period 3mo --allow-short
```

---

## Monthly Return Projections

Starting capital $5,000. Options risk 5% per spread. Futures on /MES during challenge.

| Scenario | Monthly return | Notes |
|----------|---------------|-------|
| Conservative (options only, 50% WR) | 3-5% | Iron condors + bull put spreads only |
| Moderate (both layers, 55% WR) | 8-12% | Add futures scalps on clear setups |
| Aggressive (both, 60% WR, compounding) | 15-20% | Requires consistent daily execution |

To reach $10k/month in profit requires approximately $80-120k account at 8-12% monthly,
or a funded prop account with $250k+ in buying power alongside a $20-30k options account.

---

## Prop Firm Path

1. **Challenge phase:** Pass using /MES only, max 2 contracts, GEX setups only (no discretionary)
2. **Verification phase:** Same rules, prove consistency over 30 days
3. **Funded:** Scale to /ES. Use funded account profits to grow the options account
4. **Target:** Funded /ES account ($100k+) running alongside a $20-30k options account

The prop firm provides the leverage; the options account provides the defined-risk compounding base.

---

## Immediate Trade Plan (week of 2026-05-02)

Based on current GEX data with SPY at $720.67:

### This week's options trade
- **Iron condor expiring May 8** — highest conviction given +75.25B GEX pin
- Sell $710P / Buy $705P + Sell $726C / Buy $731C
- Entry: Monday open (now), collect credit on both sides
- Exit trigger: 50% of max credit or either short strike breached

### Futures watch levels (/ES)
- Short setup: approach to ~7250 (call wall). Stop 7260, target 7200
- Long setup: approach to ~7100 (put wall). Stop 7090, target 7200
- No trade: /ES grinding between 7150-7240 (middle of range, low edge)

### Calendar alert
- **May 15:** Enter straddle for May 29 regime flip while IV is still low
- **May 22:** Monthly opex — watch for gamma roll-off and regime transition
- **May 29:** Negative gamma begins — switch from fade to breakout playbook
