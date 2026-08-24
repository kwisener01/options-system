"""
REAL-FILL backtest of a LONG-premium "gamma scalp" idea (0DTE SPY straddle).

Structure each day:
  - BUY 1 ATM call (Kc ~ round(spot@entry)), BUY 1 ATM put (Kp = Kc)   -> debit D
  - Max loss is capped at D (defined risk, no assignment/hedge-cost drag —
    the failure mode that killed the short-premium "Levitation" idea).

Two entry modes:
  --filter none   Buy every day (baseline — tests whether daily long gamma
                   beats theta decay at all).
  --filter rv     Only buy when trailing realized vol implies the straddle
                   is CHEAP: compare the straddle's breakeven move (D / spot)
                   against the trailing `--rv-days` average daily range. Buy
                   only if breakeven_pct <= avg_range_pct * `--rv-mult`
                   (the market is pricing a smaller move than has been
                   typical lately — the inverse of the IV>RV gate the short-
                   premium strategies use).

Prices are REAL Alpaca traded prices near the 10:00 ET entry (not Black-
Scholes); historical bid/ask isn't available so a per-leg slippage haircut
approximates the spread you cross. Walks 5-min bars and exits at +target%
of debit (scalp the move) or force-closes at the 15:45 ET mark; never lets
it ride to expiry (0DTE settlement/pin risk).

Usage:  python run_gamma_scalp_realfill_backtest.py [--days 180] [--filter rv]
"""
import argparse, json, os, sys, time, hashlib
from datetime import datetime, timedelta, date, time as dt_time
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(__file__))
import requests
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY
from src.live.alpaca_options import occ_symbol

ET, UTC = ZoneInfo("America/New_York"), ZoneInfo("UTC")
DATA = "https://data.alpaca.markets/v1beta1/options"
STK  = "https://data.alpaca.markets/v2/stocks"
HEAD = {"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY}
OPTIONS_START = date(2024, 2, 1)
ENTRY_ET = dt_time(10, 0)
_CACHE = os.path.join(os.path.dirname(__file__), ".cache", "gscalp")
os.makedirs(_CACHE, exist_ok=True)


def _cached_get(params):
    key = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
    fp = os.path.join(_CACHE, f"{key}.json")
    if os.path.exists(fp):
        with open(fp) as f:
            return json.load(f)
    for attempt in range(4):
        r = requests.get(f"{DATA}/bars", headers=HEAD, params=params, timeout=25)
        if r.status_code == 429:
            time.sleep(2 ** attempt); continue
        d = r.json() if r.status_code == 200 else {}
        with open(fp, "w") as f:
            json.dump(d, f)
        return d
    return {}


def _u(d, t):
    return datetime.combine(d, t, ET).astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def session_bars(symbols, d):
    out = {s: {} for s in symbols}
    for i in range(0, len(symbols), 90):
        d_ = _cached_get({"symbols": ",".join(symbols[i:i+90]), "timeframe": "5Min",
                          "start": _u(d, ENTRY_ET), "end": _u(d, dt_time(15, 55)), "limit": 1000})
        for s, bars in (d_.get("bars") or {}).items():
            for b in bars:
                tt = datetime.fromisoformat(b["t"].replace("Z", "+00:00")).astimezone(ET).strftime("%H:%M")
                out[s][tt] = b.get("vw", b["c"])
    return out


def spy_spot_close(start, end):
    """{date: {hhmm: price}} of RTH 5-min SPY bars, real Alpaca prices."""
    out, tok = {}, None
    while True:
        p = {"timeframe": "5Min", "start": _u(date.fromisoformat(start), dt_time(9, 30)),
             "end": _u(date.fromisoformat(end), dt_time(16, 0)), "limit": 10000, "feed": "sip"}
        if tok: p["page_token"] = tok
        r = requests.get(f"{STK}/SPY/bars", headers=HEAD, params=p, timeout=30)
        d = r.json() if r.status_code == 200 else {}
        for b in (d.get("bars") or []):
            dt_ = datetime.fromisoformat(b["t"].replace("Z", "+00:00")).astimezone(ET)
            dd, hhmm = dt_.date(), dt_.strftime("%H:%M")
            out.setdefault(dd, {})[hhmm] = b.get("vw", b["c"])
        tok = d.get("next_page_token")
        if not tok: break
    return out


def _sim(call, put, D, target, slip):
    """Walk 5-min bars; exit at +target% of debit, else force-close at 15:45."""
    times = sorted(set(call) & set(put))
    if not times:
        return None, "no_bars"
    for t in times:
        mark = call[t] + put[t] - 2 * slip - D
        if mark >= target * D:
            return mark * 100, "target"
    last = times[-1]
    mark = call[last] + put[last] - 2 * slip - D
    return mark * 100, "close"


def run(days, target, slip, filt, rv_days, rv_mult):
    cutoff = max(OPTIONS_START, datetime.now(ET).date() - timedelta(days=days))
    end    = (datetime.now(ET).date() - timedelta(days=1))
    sb = spy_spot_close(cutoff.isoformat(), end.isoformat())
    if not sb:
        print("SPY data fetch failed"); return
    dates = sorted(sb)

    # trailing realized daily range (high-low of RTH bars, as % of close) per day, for the rv filter
    daily_range_pct = {}
    for d in dates:
        vals = list(sb[d].values())
        if len(vals) < 2:
            continue
        daily_range_pct[d] = (max(vals) - min(vals)) / vals[-1]

    trades, skips = [], {}
    def skip(w): skips[w] = skips.get(w, 0) + 1
    print(f"Scanning {len(dates)} days from {cutoff} (real fills, filter={filt})...")

    for i, d in enumerate(dates):
        bars = sb[d]
        if ENTRY_ET.strftime("%H:%M") not in bars:
            skip("no_entry_bar"); continue
        spot = bars[ENTRY_ET.strftime("%H:%M")]
        Kc = Kp = round(spot)

        if filt == "rv":
            idx = dates.index(d)
            hist = [daily_range_pct[dd] for dd in dates[max(0, idx - rv_days):idx] if dd in daily_range_pct]
            if len(hist) < max(3, rv_days // 2):
                skip("not_enough_rv_history"); continue
            avg_range = sum(hist) / len(hist)

        call_sym = occ_symbol("SPY", d, "CALL", Kc)
        put_sym  = occ_symbol("SPY", d, "PUT", Kp)
        sess = session_bars([call_sym, put_sym], d)
        call, put = sess[call_sym], sess[put_sym]
        t0 = sorted(set(call) & set(put))
        if not t0:
            skip("no_session_bars"); continue
        t0 = t0[0]
        D = call[t0] + put[t0]
        if D <= 0.10:
            skip("no_debit"); continue

        if filt == "rv":
            breakeven_pct = D / spot
            if not (breakeven_pct <= avg_range * rv_mult):
                skip("rv_filter_reject"); continue

        pnl, reason = _sim(call, put, D, target, slip)
        if pnl is None:
            skip("no_bars"); continue
        trades.append({"date": d, "D": D, "pnl": pnl, "reason": reason})
        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(dates)}  ({len(trades)} trades)")
        time.sleep(0.04)

    _report(trades, skips, target, slip, filt)


def _report(trades, skips, target, slip, filt):
    if not trades:
        print("\nno trades.  skips:", skips); return
    n = len(trades)
    pnls = [t["pnl"] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    tot = sum(pnls)
    eq = peak = mdd = 0.0
    for p in pnls:
        eq += p; peak = max(peak, eq); mdd = min(mdd, eq - peak)
    avgD = sum(t["D"] for t in trades) / n
    reasons = {}
    for t in trades:
        reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1

    print(f"\n{'='*70}")
    print(f"  ATM long straddle — REAL-FILL 0DTE gamma-scalp backtest")
    print(f"  filter={filt} | take +{target:.0%} of debit | force-close 15:45 | slip ${slip}/leg")
    print(f"  avg debit ${avgD:.2f} | exits: {reasons}")
    print(f"{'='*70}")
    print(f"  trades       {n}")
    print(f"  win rate     {wins/n*100:.1f}%")
    print(f"  total P&L    {tot:,.0f}")
    print(f"  avg / trade  {tot/n:+.2f}")
    print(f"  best/worst   {max(pnls):+.0f} / {min(pnls):+.0f}")
    print(f"  max drawdown {mdd:,.0f}")
    if skips:
        print(f"  skips: " + ", ".join(f"{k}={v}" for k, v in sorted(skips.items())))
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--target", type=float, default=0.5, help="take profit at this x debit paid")
    ap.add_argument("--slip", type=float, default=0.02)
    ap.add_argument("--filter", choices=["none", "rv"], default="none")
    ap.add_argument("--rv-days", type=int, default=10)
    ap.add_argument("--rv-mult", type=float, default=1.0)
    a = ap.parse_args()
    run(a.days, a.target, a.slip, a.filter, a.rv_days, a.rv_mult)
