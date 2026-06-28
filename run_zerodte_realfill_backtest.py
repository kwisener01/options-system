"""
REAL-FILL backtest of the 0DTE SPY put-spread-at-the-expected-move strategy.

Unlike run_zerodte_backtest.py (which MODELS the credit with Black-Scholes), this
one uses Alpaca's REAL historical option data since Feb 2024:
  - entry credit  = actual TRADED prices of the two legs near the 10:00 ET entry
  - intraday mgmt = real 5-min trade BARS to detect the 50% profit target
  - exit cost     = real bar price at the 50% target or the 15:45 ET force-close

Limitation we're honest about: Alpaca serves historical TRADES + BARS but NOT
historical bid/ask quotes, so we approximate the cost of crossing the spread with
a per-leg slippage haircut (--slip, default $0.02/leg). Results are reported BOTH
with and without slippage so you can see the sensitivity.

Strategy mirrors config/zerodte.json: SPY $1-wide put credit spread, short ~1
expected-move below the open (VIX-based EM), enter ~10:00 ET, take profit at 50%,
never close losers early, FORCE-CLOSE 15:45 ET (0DTE assignment guard).

Usage:  python run_zerodte_realfill_backtest.py [--days 500] [--slip 0.02]
"""
import argparse, json, math, os, sys, time, hashlib
from datetime import datetime, timedelta, date, time as dt_time
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(__file__))
import requests
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY
from src.live.alpaca_options import occ_symbol

ET   = ZoneInfo("America/New_York")
UTC  = ZoneInfo("UTC")
DATA = "https://data.alpaca.markets/v1beta1/options"
HEAD = {"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY}

WIDTH           = 1.0
MIN_CREDIT_FRAC = 0.12
PROFIT_TARGET   = 0.50
ENTRY_ET        = dt_time(10, 0)
FORCE_ET        = dt_time(15, 45)
T_TO_CLOSE      = 5.5 / (6.5 * 252)     # ~10:00 entry to cash close, in years
OPTIONS_START   = date(2024, 2, 1)      # Alpaca historical options inception

_CACHE = os.path.join(os.path.dirname(__file__), ".cache", "zerodte")
os.makedirs(_CACHE, exist_ok=True)


# ── data helpers (raw REST + on-disk cache) ────────────────────────────────────

def _cached_get(ep: str, params: dict) -> dict:
    key = hashlib.md5(f"{ep}|{json.dumps(params, sort_keys=True)}".encode()).hexdigest()
    fp  = os.path.join(_CACHE, f"{key}.json")
    if os.path.exists(fp):
        with open(fp) as f:
            return json.load(f)
    for attempt in range(4):
        r = requests.get(f"{DATA}/{ep}", headers=HEAD, params=params, timeout=25)
        if r.status_code == 429:
            time.sleep(2 ** attempt); continue
        if r.status_code != 200:
            return {}
        d = r.json()
        with open(fp, "w") as f:
            json.dump(d, f)
        return d
    return {}


def _et_to_utc(d: date, t: dt_time) -> str:
    return datetime.combine(d, t, ET).astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def vwap_trade(sym: str, d: date, t0: dt_time, t1: dt_time):
    """Size-weighted avg traded price of `sym` in [t0,t1] ET on day d, or None."""
    data = _cached_get("trades", {"symbols": sym, "start": _et_to_utc(d, t0),
                                  "end": _et_to_utc(d, t1), "limit": 5000})
    ts = (data.get("trades") or {}).get(sym) or []
    num = sum(float(x["p"]) * float(x["s"]) for x in ts)
    den = sum(float(x["s"]) for x in ts)
    return (num / den) if den > 0 else None


def day_bars(sym: str, d: date):
    """5-min trade bars for `sym` over the RTH of day d → {ET-time: close}."""
    data = _cached_get("bars", {"symbols": sym, "start": _et_to_utc(d, dt_time(9, 30)),
                                "end": _et_to_utc(d, dt_time(16, 0)),
                                "timeframe": "5Min", "limit": 200})
    out = {}
    for b in (data.get("bars") or {}).get(sym, []):
        tt = datetime.fromisoformat(b["t"].replace("Z", "+00:00")).astimezone(ET).time()
        out[tt] = float(b["c"])
    return out


# ── backtest ───────────────────────────────────────────────────────────────────

def run(days: int, slip: float):
    from src.live.yahoo import fetch_daily_ohlcv
    period = "2y" if days > 365 else "1y"
    spy = fetch_daily_ohlcv("SPY", period)
    vix = fetch_daily_ohlcv("^VIX", period)
    if spy.empty or vix.empty:
        print("data fetch failed"); return
    vix_close = {d.date(): float(v) for d, v in vix["close"].items()}
    vix_dates = sorted(vix_close)

    def prev_vix(d):
        prior = [x for x in vix_dates if x < d]
        return vix_close[prior[-1]] if prior else None

    cutoff = max(OPTIONS_START, (datetime.now(ET).date() - timedelta(days=days)))
    rows = [(ts.date(), float(r["open"]), float(r["close"]))
            for ts, r in spy.iterrows() if ts.date() >= cutoff]

    trades, skips = [], {}
    def skip(why): skips[why] = skips.get(why, 0) + 1

    print(f"Scanning {len(rows)} trading days from {cutoff} (real Alpaca option data)...")
    for i, (d, o, c) in enumerate(rows):
        v = prev_vix(d)
        if v is None:
            skip("no_vix"); continue
        em    = o * (v / 100.0) * math.sqrt(T_TO_CLOSE)
        short = round(o - em)
        long_ = short - WIDTH
        ssym  = occ_symbol("SPY", d, "PUT", short)
        lsym  = occ_symbol("SPY", d, "PUT", long_)

        # entry: real traded prices near 10:00 ET
        sp = vwap_trade(ssym, d, ENTRY_ET, dt_time(10, 6))
        lp = vwap_trade(lsym, d, ENTRY_ET, dt_time(10, 6))
        if sp is None or lp is None:
            skip("no_entry_trades"); continue
        gross = sp - lp
        if gross < MIN_CREDIT_FRAC * WIDTH:
            skip("thin_credit"); continue
        credit = gross - 2 * slip                  # received, net of entry slippage

        sbars, lbars = day_bars(ssym, d), day_bars(lsym, d)
        common = sorted(t for t in sbars if t in lbars and t > ENTRY_ET)
        if not common:
            skip("no_mgmt_bars"); continue

        exit_reason, realized, realized_ns = None, None, None
        for t in common:
            spread = sbars[t] - lbars[t]
            cost   = spread + 2 * slip             # to buy the spread back
            if credit - cost >= PROFIT_TARGET * gross:
                exit_reason = "target"
                realized    = credit - cost
                realized_ns = gross - spread        # no-slippage variant
                break
            if t >= FORCE_ET:
                exit_reason = "force_close"
                realized    = credit - cost
                realized_ns = gross - spread
                break
        if realized is None:                        # ran out of bars before 15:45
            t = common[-1]; spread = sbars[t] - lbars[t]
            exit_reason = "eod_bar"
            realized    = credit - (spread + 2 * slip)
            realized_ns = gross - spread

        trades.append({"date": d, "credit": gross, "net_credit": credit,
                       "pnl": realized * 100, "pnl_ns": realized_ns * 100,
                       "win": realized > 0, "reason": exit_reason,
                       "short": short, "breach_close": c < short})
        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{len(rows)} days  ({len(trades)} trades)")
        time.sleep(0.05)

    _report(trades, skips, slip)


def _report(trades, skips, slip):
    if not trades:
        print("\nno trades.  skips:", skips); return
    n      = len(trades)
    wins   = sum(t["win"] for t in trades)
    total  = sum(t["pnl"] for t in trades)
    tot_ns = sum(t["pnl_ns"] for t in trades)
    avgcr  = sum(t["credit"] for t in trades) / n
    best   = max(t["pnl"] for t in trades)
    worst  = min(t["pnl"] for t in trades)
    tgt    = sum(1 for t in trades if t["reason"] == "target")
    forced = sum(1 for t in trades if t["reason"] in ("force_close", "eod_bar"))
    eq = peak = mdd = 0.0
    for t in trades:
        eq += t["pnl"]; peak = max(peak, eq); mdd = min(mdd, eq - peak)

    print(f"\n{'='*66}")
    print(f"  0DTE SPY put-spread @ expected move — REAL-FILL backtest")
    print(f"  (real traded prices since Feb-2024; slippage ${slip:.2f}/leg)")
    print(f"{'='*66}")
    print(f"  Trades                 : {n}")
    print(f"  Win rate               : {wins/n*100:.1f}%   ({wins}W / {n-wins}L)")
    print(f"  Avg entry credit (gross): ${avgcr:.3f}   (max profit ${avgcr*100:.0f}/spread)")
    print(f"  50% target hits        : {tgt} ({tgt/n*100:.0f}%)   force/eod closes: {forced}")
    print(f"  --- P&L per 1-lot/day, REAL fills ---")
    print(f"  WITH slippage  total   : ${total:,.0f}   avg/trade ${total/n:+.2f}")
    print(f"  NO   slippage  total   : ${tot_ns:,.0f}   avg/trade ${tot_ns/n:+.2f}")
    print(f"  Best / worst day       : ${best:+.0f} / ${worst:+.0f}")
    print(f"  Max drawdown (cum)     : ${mdd:,.0f}")
    if skips:
        print(f"  Skipped days           : " +
              ", ".join(f"{k}={v}" for k, v in sorted(skips.items())))
    verdict = "EDGE (positive w/ slippage)" if total > 0 else (
              "marginal (positive only w/o slippage)" if tot_ns > 0 else "NO EDGE (negative even gross)")
    print(f"  VERDICT                : {verdict}")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=500, help="calendar-day lookback (capped at Feb-2024)")
    ap.add_argument("--slip", type=float, default=0.02, help="per-leg slippage haircut ($)")
    a = ap.parse_args()
    run(a.days, a.slip)
