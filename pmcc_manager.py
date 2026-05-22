"""
PMCC POSITION MANAGER  v1
Tracks open PMCC positions, fires close/roll signals, and projects compound growth.

Rules embedded:
  - Close short call when >= 50% of original premium decayed
  - Force roll when DTE drops to <= 14 (gamma risk zone)
  - New short call target: 21-35 DTE, 8-14% OTM from current price
  - Danger alert: stock within 1.30x of LEAPS strike (delta collapse risk)

Income cadence: 21-35 DTE + 50% close = income event every ~10-20 days per position.
Three staggered PMCCs = 1-2 income events per week without weekly option gamma risk.

Run daily:   python pmcc_manager.py
Schedule:    python pmcc_manager.py --schedule
Add ticker:  python pmcc_manager.py --add
"""

import os
import sys
import json
import time
import logging
import requests
import schedule
import io
from datetime import datetime, date, timedelta
from typing import Optional
from dotenv import load_dotenv

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    print("[!] yfinance not installed.  pip install yfinance")

load_dotenv()

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("pmcc_manager.log")]
)
log = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────

CONFIG = {
    "ACCOUNT_SIZE":          1500,    # total account value
    "FREE_CASH":              830,    # cash not in LEAPS (update as you add PMCCs)
    "CLOSE_PROFIT_PCT":        50,    # close short call when this % of premium decayed
    "ROLL_DTE_THRESHOLD":      14,    # force roll at this DTE regardless of profit
    "NEW_SHORT_DTE_MIN":       21,    # target DTE window — min
    "NEW_SHORT_DTE_MAX":       35,    # target DTE window — max
    "NEW_SHORT_OTM_MIN":     0.06,    # min OTM % for new short call (6%)
    "NEW_SHORT_OTM_MAX":     0.14,    # max OTM % for new short call (14%)
    "LEAPS_DANGER_MULTIPLE": 1.30,    # alert if stock < LEAPS_strike × this
    "SCAN_TIME":           "09:40",   # ET — 5 mins after cash open
    "SLACK_WEBHOOK":  os.getenv("SLACK_WEBHOOK_URL", ""),
}

POSITIONS_FILE = os.path.join(os.path.dirname(__file__), "pmcc_positions.json")

# ── DEFAULT POSITIONS — seeded with your live SOFI trade ─────────────────────

DEFAULT_POSITIONS = [
    {
        "id":           "sofi_pmcc_1",
        "ticker":       "SOFI",
        "long_symbol":  "SOFI270115C00010000",
        "long_strike":  10.00,
        "long_expiry":  "2027-01-15",
        "long_cost":    6.70,        # per share paid
        "short_symbol": "SOFI260717C00018000",
        "short_strike": 18.00,
        "short_expiry": "2026-07-17",
        "short_credit": 0.75,        # per share collected
        "opened_date":  "2026-05-01",
        "contracts":    1,
        "notes":        "Next roll: target $17-17.50 strike (8-9% OTM). Current $18 is 12% OTM — too far for max income."
    }
]

# ── POSITIONS I/O ─────────────────────────────────────────────────────────────

def load_positions() -> list:
    if os.path.exists(POSITIONS_FILE):
        with open(POSITIONS_FILE) as f:
            return json.load(f)
    with open(POSITIONS_FILE, "w") as f:
        json.dump(DEFAULT_POSITIONS, f, indent=2)
    return DEFAULT_POSITIONS

def save_positions(positions: list):
    with open(POSITIONS_FILE, "w") as f:
        json.dump(positions, f, indent=2)

# ── MARKET DATA (yfinance) ────────────────────────────────────────────────────

def get_stock_price(ticker: str) -> Optional[float]:
    if not YF_AVAILABLE:
        return None
    try:
        info = yf.Ticker(ticker).fast_info
        return round(float(info.last_price), 2)
    except Exception as e:
        log.debug(f"Stock price failed {ticker}: {e}")
        return None

def get_option_quote(ticker: str, expiry: str, strike: float,
                     opt_type: str = "call") -> Optional[dict]:
    """Return bid/ask/mid for the nearest-strike option on the given expiry."""
    if not YF_AVAILABLE:
        return None
    try:
        stk = yf.Ticker(ticker)
        chain = stk.option_chain(expiry)
        opts = chain.calls.copy() if opt_type == "call" else chain.puts.copy()
        opts["diff"] = abs(opts["strike"] - strike)
        row = opts.nsmallest(1, "diff").iloc[0]
        if abs(row["diff"]) > 0.51:
            return None
        bid = max(float(row.get("bid", 0) or 0), 0)
        ask = max(float(row.get("ask", 0) or 0), 0)
        last = float(row.get("lastPrice", 0) or 0)
        if bid == 0 and ask == 0:
            bid, ask = last * 0.95, last * 1.05
        mid = (bid + ask) / 2
        return {
            "bid":    round(bid, 2),
            "ask":    round(ask, 2),
            "mid":    round(mid, 2),
            "iv":     round(float(row.get("impliedVolatility", 0) or 0), 4),
            "volume": int(row.get("volume", 0) or 0),
            "oi":     int(row.get("openInterest", 0) or 0),
            "strike": float(row["strike"]),
        }
    except Exception as e:
        log.debug(f"Option quote failed {ticker} {expiry} {strike}: {e}")
        return None

def suggest_next_roll(ticker: str, stock_price: float,
                      long_expiry: str, long_strike: float) -> dict:
    """
    Find the best short call to sell next.
    Targets: 21-35 DTE, 8-14% OTM, above LEAPS strike.
    Returns concrete strike + expiry + estimated premium.
    """
    if not YF_AVAILABLE:
        return {"error": "yfinance required"}
    try:
        today         = date.today()
        min_date      = today + timedelta(days=CONFIG["NEW_SHORT_DTE_MIN"])
        max_date      = today + timedelta(days=CONFIG["NEW_SHORT_DTE_MAX"])
        long_exp_date = datetime.strptime(long_expiry, "%Y-%m-%d").date()

        stk       = yf.Ticker(ticker)
        candidate = None
        for exp_str in stk.options:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            if min_date <= exp_date <= max_date and exp_date < long_exp_date:
                candidate = exp_str
                break

        # Fallback: widen to 45 DTE if nothing in 21-35 window
        if candidate is None:
            max_date = today + timedelta(days=50)
            for exp_str in stk.options:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                if min_date <= exp_date <= max_date and exp_date < long_exp_date:
                    candidate = exp_str
                    break

        if candidate is None:
            return {"error": "No valid expiry found — check manually"}

        dte          = (datetime.strptime(candidate, "%Y-%m-%d").date() - today).days
        strike_lo    = max(
            round(stock_price * (1 + CONFIG["NEW_SHORT_OTM_MIN"]) * 2) / 2,
            long_strike + 0.50
        )
        strike_hi    = round(stock_price * (1 + CONFIG["NEW_SHORT_OTM_MAX"]) * 2) / 2

        chain = stk.option_chain(candidate)
        calls = chain.calls.copy()
        calls = calls[(calls["strike"] >= strike_lo) & (calls["strike"] <= strike_hi) & (calls["bid"] > 0)]

        if calls.empty:
            return {
                "expiry":       candidate,
                "dte":          dte,
                "strike_range": f"${strike_lo:.2f}–${strike_hi:.2f}",
                "note":         "No bid data — verify manually",
            }

        best    = calls.nlargest(1, "bid").iloc[0]
        bid     = float(best["bid"])
        strike  = float(best["strike"])
        otm_pct = round((strike - stock_price) / stock_price * 100, 1)

        return {
            "expiry":    candidate,
            "strike":    strike,
            "dte":       dte,
            "bid":       round(bid, 2),
            "credit":    round(bid * 100, 2),
            "otm_pct":   otm_pct,
            "iv":        round(float(best.get("impliedVolatility", 0) or 0), 2),
            "action":    f"Sell {ticker} ${strike}C {candidate} @ ${bid:.2f}  ({dte} DTE, {otm_pct}% OTM)",
        }
    except Exception as e:
        return {"error": str(e)}

# ── POSITION ANALYSIS ────────────────────────────────────────────────────────

def monitor_position(pos: dict) -> dict:
    ticker    = pos["ticker"]
    contracts = pos["contracts"]

    result = {
        "id":            pos["id"],
        "ticker":        ticker,
        "contracts":     contracts,
        "stock_price":   None,
        "long_value":    None,
        "long_pnl":      None,
        "short_value":   None,
        "short_pnl":     None,
        "short_pnl_pct": None,
        "net_pnl":       None,
        "short_dte":     None,
        "action":        "HOLD",
        "urgency":       "LOW",
        "reason":        "Within normal parameters — no action needed.",
        "next_roll":     None,
        "danger":        False,
        "notes":         pos.get("notes", ""),
    }

    # Stock price
    sp = get_stock_price(ticker)
    result["stock_price"] = sp

    # LEAPS current value
    long_q = get_option_quote(ticker, pos["long_expiry"], pos["long_strike"], "call")
    if long_q:
        result["long_value"] = long_q["mid"]
        result["long_pnl"]   = round((long_q["mid"] - pos["long_cost"]) * 100 * contracts, 2)

    # Short call current value + DTE
    today    = date.today()
    exp_date = datetime.strptime(pos["short_expiry"], "%Y-%m-%d").date()
    dte      = max((exp_date - today).days, 0)
    result["short_dte"] = dte

    short_q = get_option_quote(ticker, pos["short_expiry"], pos["short_strike"], "call")
    if short_q:
        short_mid = short_q["mid"]
        pnl_pct   = ((pos["short_credit"] - short_mid) / pos["short_credit"] * 100
                     if pos["short_credit"] > 0 else 0)
        result["short_value"]    = short_mid
        result["short_pnl"]      = round((pos["short_credit"] - short_mid) * 100 * contracts, 2)
        result["short_pnl_pct"]  = round(pnl_pct, 1)
    else:
        pnl_pct = 0

    # Net P&L
    result["net_pnl"] = round((result["long_pnl"] or 0) + (result["short_pnl"] or 0), 2)

    # ── DANGER: stock too close to LEAPS strike ───────────────────────────────
    if sp and sp < pos["long_strike"] * CONFIG["LEAPS_DANGER_MULTIPLE"]:
        result["danger"]  = True
        result["action"]  = "DANGER_CLOSE_LEAPS"
        result["urgency"] = "CRITICAL"
        result["reason"]  = (
            f"{ticker} at ${sp:.2f} is within {CONFIG['LEAPS_DANGER_MULTIPLE']}x of "
            f"LEAPS strike ${pos['long_strike']:.2f}. Delta collapsing fast. "
            f"Evaluate closing the entire PMCC now to limit loss."
        )
        return result

    # ── BREACH: stock approaching short strike ────────────────────────────────
    if sp and sp >= pos["short_strike"] * 0.96:
        result["action"]  = "MANAGE_BREACH"
        result["urgency"] = "HIGH"
        result["reason"]  = (
            f"{ticker} at ${sp:.2f} is within 4% of short strike ${pos['short_strike']:.2f}. "
            f"Roll short call up and/or out to avoid assignment."
        )
        if sp:
            result["next_roll"] = suggest_next_roll(ticker, sp, pos["long_expiry"], pos["long_strike"])
        return result

    # ── PROFIT TARGET HIT ─────────────────────────────────────────────────────
    if pnl_pct >= CONFIG["CLOSE_PROFIT_PCT"]:
        result["action"]  = "CLOSE_AND_ROLL"
        result["urgency"] = "HIGH"
        result["reason"]  = (
            f"Short call at {pnl_pct:.0f}% profit — 50% target reached. "
            f"Buy back the short call, then sell a new 21-35 DTE call immediately."
        )
        if sp:
            result["next_roll"] = suggest_next_roll(ticker, sp, pos["long_expiry"], pos["long_strike"])
        return result

    # ── GAMMA DANGER: DTE too low ─────────────────────────────────────────────
    if dte <= CONFIG["ROLL_DTE_THRESHOLD"]:
        result["action"]  = "ROLL_NOW"
        result["urgency"] = "HIGH"
        result["reason"]  = (
            f"Only {dte} DTE — gamma is surging. Roll to a new 21-35 DTE call now "
            f"even if profit target hasn't been hit."
        )
        if sp:
            result["next_roll"] = suggest_next_roll(ticker, sp, pos["long_expiry"], pos["long_strike"])
        return result

    # ── APPROACHING WINDOW ────────────────────────────────────────────────────
    if dte <= 25 and pnl_pct >= 35:
        result["action"]  = "PREPARE_ROLL"
        result["urgency"] = "MEDIUM"
        result["reason"]  = f"{dte} DTE with {pnl_pct:.0f}% profit — approaching roll window. Prepare next strike."
        if sp:
            result["next_roll"] = suggest_next_roll(ticker, sp, pos["long_expiry"], pos["long_strike"])

    return result

# ── COMPOUND GROWTH PROJECTION ────────────────────────────────────────────────

def compound_projection(positions: list, free_cash: float) -> dict:
    """
    Honest 12-month projection.

    Assumptions (conservative):
      - Each PMCC cycle: sell 21-35 DTE call, close at 50% in ~14 days avg
      - Average net credit per cycle: 2.5% of LEAPS net debit (after commissions)
      - Cycles per month: 2 (bi-weekly cadence)
      - Win rate: 78% (22% of cycles closed at 2x loss to avoid assignment)
      - Free cash in bull put spreads: 4% gross/month, 78% win rate
      - No compounding of free cash into new LEAPS (conservative)
    """
    leaps_deployed = sum(
        (p["long_cost"] - p["short_credit"]) * 100 * p["contracts"]
        for p in positions
    )
    credit_per_cycle_pct = 0.025    # 2.5% of net debit per cycle
    cycles_per_month     = 2.0
    win_rate             = 0.78
    loss_multiple        = 2.0      # pay back 2x credit on losing cycles

    credit_per_cycle = leaps_deployed * credit_per_cycle_pct
    ev_per_cycle     = (win_rate * credit_per_cycle) - ((1 - win_rate) * credit_per_cycle * loss_multiple)
    monthly_pmcc     = max(ev_per_cycle * cycles_per_month, 0)

    # Free cash spread income: 4% gross/month × 78% win rate − 22% × avg loss
    # Net: ~2% of free cash per month in a good environment
    monthly_spreads = free_cash * 0.02

    monthly_total  = monthly_pmcc + monthly_spreads
    starting       = leaps_deployed + free_cash

    balances = []
    bal = starting
    for _ in range(12):
        bal = round(bal + monthly_total, 2)
        balances.append(bal)

    return {
        "starting":          round(starting, 2),
        "leaps_deployed":    round(leaps_deployed, 2),
        "free_cash":         round(free_cash, 2),
        "monthly_pmcc_ev":   round(monthly_pmcc, 2),
        "monthly_spread_ev": round(monthly_spreads, 2),
        "monthly_total_ev":  round(monthly_total, 2),
        "month_6":           balances[5],
        "month_12":          balances[11],
        "return_12mo_pct":   round((balances[11] - starting) / starting * 100, 1),
        "monthly_history":   balances,
    }

# ── OUTPUT ────────────────────────────────────────────────────────────────────

URGENCY_ICON = {"CRITICAL": "🚨", "HIGH": "🔴", "MEDIUM": "🟡", "LOW": "✅"}

def print_report(results: list, projection: dict):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*60}")
    print(f"  PMCC POSITION MONITOR — {now}")
    print(f"{'='*60}")

    for r in results:
        icon  = URGENCY_ICON.get(r["urgency"], "")
        print(f"\n  {icon} {r['ticker']} (×{r['contracts']} contract)")
        print(f"  {'─'*40}")
        if r["stock_price"]:
            print(f"  Stock         ${r['stock_price']:.2f}")

        lv = f"${r['long_value']:.2f}"  if r["long_value"]  else "n/a"
        sv = f"${r['short_value']:.2f}" if r["short_value"] else "n/a"
        print(f"  LEAPS value   {lv}   (paid ${r['long_value'] and round(r['long_value'] - (r['long_pnl'] or 0) / 100 / r['contracts'], 2) or 'n/a'})")
        print(f"  Short call    {sv}   ({r['short_pnl_pct'] or 0:.0f}% profit | {r['short_dte']} DTE)")

        lp = f"{r['long_pnl']:+.2f}"   if r["long_pnl"]  is not None else "n/a"
        sp = f"{r['short_pnl']:+.2f}"  if r["short_pnl"] is not None else "n/a"
        np = f"{r['net_pnl']:+.2f}"    if r["net_pnl"]   is not None else "n/a"
        print(f"  LEAPS P&L     ${lp}")
        print(f"  Short P&L     ${sp}")
        print(f"  Net P&L       ${np}")

        print(f"\n  Action:  {r['action']}")
        print(f"  {r['reason']}")

        if r.get("next_roll"):
            nr = r["next_roll"]
            if "action" in nr:
                print(f"\n  ROLL TARGET: {nr['action']}")
                print(f"  Credit: ${nr.get('credit', 'n/a')} | IV: {nr.get('iv', 'n/a')}")
            elif "error" in nr:
                print(f"\n  Roll lookup: {nr['error']}")
            else:
                rng = nr.get("strike_range", "")
                print(f"\n  ROLL TARGET: {nr.get('expiry','?')} | Strike: {rng} | {nr.get('dte','?')} DTE")

        if r.get("notes"):
            print(f"\n  Note: {r['notes']}")

    # ── PROJECTION ────────────────────────────────────────────────────────────
    p = projection
    print(f"\n{'='*60}")
    print(f"  COMPOUND GROWTH PROJECTION (12 months, conservative)")
    print(f"{'─'*60}")
    print(f"  Starting balance:    ${p['starting']:,.2f}")
    print(f"  LEAPS deployed:      ${p['leaps_deployed']:,.2f}")
    print(f"  Free cash:           ${p['free_cash']:,.2f}")
    print(f"  PMCC income/month:   ${p['monthly_pmcc_ev']:,.2f}  (EV @ 78% win rate)")
    print(f"  Spread income/month: ${p['monthly_spread_ev']:,.2f}  (EV @ 78% win rate)")
    print(f"  Combined EV/month:   ${p['monthly_total_ev']:,.2f}")
    print(f"  Month 6 projection:  ${p['month_6']:,.2f}")
    print(f"  Month 12 projection: ${p['month_12']:,.2f}  ({p['return_12mo_pct']:.1f}% return)")
    print(f"\n  Truth: 78% win rate is not guaranteed. Bad streaks happen. Keep")
    print(f"  individual trade risk below $150 and the math works over time.")
    print(f"{'='*60}\n")

def send_slack_alerts(results: list, projection: dict):
    webhook = CONFIG["SLACK_WEBHOOK"]
    if not webhook:
        return

    actionable = [r for r in results if r["urgency"] in ("CRITICAL", "HIGH")]
    if not actionable:
        return

    lines = ["*PMCC POSITION ALERT*\n"]
    for r in actionable:
        icon = URGENCY_ICON.get(r["urgency"], "")
        lines.append(f"{icon} *{r['ticker']}* — {r['action']}")
        lines.append(f"  {r['reason']}")
        nr = r.get("next_roll", {})
        if nr and "action" in nr:
            lines.append(f"  Roll: {nr['action']}")
        lines.append("")

    lines.append(f"_Account EV: ${projection['monthly_total_ev']:.0f}/mo | 12mo target: ${projection['month_12']:,.0f}_")

    try:
        requests.post(webhook, json={"text": "\n".join(lines)}, timeout=8)
        log.info(f"Slack alert sent for {len(actionable)} position(s)")
    except Exception as e:
        log.warning(f"Slack send failed: {e}")

# ── ADD POSITION INTERACTIVELY ────────────────────────────────────────────────

def add_position_interactive(positions: list) -> list:
    print("\nAdd a new PMCC position")
    print("─" * 40)
    try:
        ticker       = input("Ticker (e.g. MARA): ").strip().upper()
        long_symbol  = input("LEAPS symbol (e.g. MARA270117C00005000): ").strip()
        long_strike  = float(input("LEAPS strike (e.g. 5.0): "))
        long_expiry  = input("LEAPS expiry YYYY-MM-DD: ").strip()
        long_cost    = float(input("LEAPS cost per share (what you paid, e.g. 4.20): "))
        short_symbol = input("Short call symbol (e.g. MARA260717C00012000): ").strip()
        short_strike = float(input("Short strike (e.g. 12.0): "))
        short_expiry = input("Short expiry YYYY-MM-DD: ").strip()
        short_credit = float(input("Credit collected per share (e.g. 0.55): "))
        contracts    = int(input("Contracts (e.g. 1): ") or "1")
        notes        = input("Notes (optional): ").strip()

        pos_id = f"{ticker.lower()}_pmcc_{len(positions)+1}"
        new_pos = {
            "id":           pos_id,
            "ticker":       ticker,
            "long_symbol":  long_symbol,
            "long_strike":  long_strike,
            "long_expiry":  long_expiry,
            "long_cost":    long_cost,
            "short_symbol": short_symbol,
            "short_strike": short_strike,
            "short_expiry": short_expiry,
            "short_credit": short_credit,
            "opened_date":  date.today().isoformat(),
            "contracts":    contracts,
            "notes":        notes,
        }
        positions.append(new_pos)
        save_positions(positions)
        print(f"\n[OK] {ticker} PMCC saved to {POSITIONS_FILE}")
    except Exception as e:
        print(f"[!] Error: {e}")
    return positions

# ── MAIN MONITOR RUN ─────────────────────────────────────────────────────────

def run_monitor():
    log.info("PMCC Manager starting")
    positions = load_positions()
    if not positions:
        print("[!] No positions found. Run with --add to enter your first PMCC.")
        return

    results = []
    for pos in positions:
        try:
            r = monitor_position(pos)
            results.append(r)
        except Exception as e:
            log.error(f"Monitor failed for {pos['ticker']}: {e}")

    proj = compound_projection(positions, CONFIG["FREE_CASH"])
    print_report(results, proj)
    send_slack_alerts(results, proj)
    log.info("Monitor run complete")

def run_scheduled():
    t = CONFIG["SCAN_TIME"]
    log.info(f"Scheduler started — running daily at {t} ET")
    schedule.every().monday.at(t).do(run_monitor)
    schedule.every().tuesday.at(t).do(run_monitor)
    schedule.every().wednesday.at(t).do(run_monitor)
    schedule.every().thursday.at(t).do(run_monitor)
    schedule.every().friday.at(t).do(run_monitor)
    while True:
        schedule.run_pending()
        time.sleep(30)

# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not YF_AVAILABLE:
        print("[!] Install yfinance first:  pip install yfinance")
        sys.exit(1)

    if "--add" in sys.argv:
        positions = load_positions()
        positions = add_position_interactive(positions)
    elif "--schedule" in sys.argv:
        run_scheduled()
    else:
        run_monitor()
