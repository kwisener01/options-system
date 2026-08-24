"""
Real-data test: are the gamma/put/call walls from the pasted Slack GEX Dashboard
posts (18 snapshots, 14 distinct days, real timestamps + real spot) actually
significant support/resistance, using REAL subsequent SPY price action?

For each snapshot: fetch real 5-min SPY bars from the snapshot time to that
day's close. Check whether price APPROACHES each wall (within `tol` of it) and,
if so, whether it BOUNCES (closes the rest of the day on the same side it
approached from, i.e. doesn't break through by more than `tol`) or BREAKS
(closes clearly through it).

Critical control: a "wall" 10-15pts from spot isn't special just because it's a
round-ish number near where price already was — prices bounce off ALL sorts of
levels by chance. So every real wall gets a PLACEBO level at the same distance
from spot but mirrored to the opposite side (e.g. real put_wall distance below
spot -> placebo put level tested using that same distance measured as if it
were a call-side level, and vice versa). If real walls bounce more often than
placebo levels at the same distance, that's the signal. If not, distance from
spot alone explains everything and the wall's specific price has no extra info.

Usage: python run_wall_significance_test.py
"""
import json, os, sys, time
from datetime import datetime, timedelta, date, time as dt_time
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(__file__))
import requests
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY

ET, UTC = ZoneInfo("America/New_York"), ZoneInfo("UTC")
STK = "https://data.alpaca.markets/v2/stocks"
HEAD = {"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY}

SNAPSHOTS_PATH = (r"C:\Users\kwise\AppData\Local\Temp\claude\C--Projects-algorithmic-trader"
                  r"\63c18184-cee3-49d4-910d-2888f87bf47d\scratchpad\parsed.json")
TOL = 0.0035   # 0.35% of spot -- "approached" the level


def _u(d, t):
    return datetime.combine(d, t, ET).astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_rest_of_day(d, start_t):
    p = {"timeframe": "5Min", "start": _u(d, start_t), "end": _u(d, dt_time(16, 0)),
         "limit": 1000, "feed": "sip"}
    r = requests.get(f"{STK}/SPY/bars", headers=HEAD, params=p, timeout=25)
    data = r.json() if r.status_code == 200 else {}
    bars = []
    for b in (data.get("bars") or []):
        dt_ = datetime.fromisoformat(b["t"].replace("Z", "+00:00")).astimezone(ET)
        bars.append({"t": dt_.strftime("%H:%M"), "h": b["h"], "l": b["l"], "c": b["c"]})
    return bars


def approach_and_outcome(bars, level, spot, side):
    """side: 'below' (support-type, e.g. put wall) or 'above' (resistance-type).
    Returns (approached: bool, bounced: bool or None)."""
    tol_abs = TOL * spot
    approached = False
    for b in bars:
        if side == "below" and b["l"] <= level + tol_abs:
            approached = True
        if side == "above" and b["h"] >= level - tol_abs:
            approached = True
    if not approached or not bars:
        return approached, None
    last_close = bars[-1]["c"]
    if side == "below":
        bounced = last_close > level - tol_abs   # stayed above (didn't break down through)
    else:
        bounced = last_close < level + tol_abs   # stayed below (didn't break up through)
    return approached, bounced


def run():
    with open(SNAPSHOTS_PATH) as f:
        snaps = json.load(f)
    print(f"{len(snaps)} snapshots loaded\n")

    results = []
    for s in snaps:
        ts = datetime.fromisoformat(s["ts"])
        d, t0 = ts.date(), ts.time()
        spot = s["spot"]
        put_wall, call_wall = s["put_wall"], s["call_wall"]
        # placebo: mirror the real distance to the OTHER side of spot, so it's
        # equally far away but not the level GEX theory actually flags.
        put_dist, call_dist = spot - put_wall, call_wall - spot
        placebo_below = spot - call_dist   # as far below as the call wall is above
        placebo_above = spot + put_dist    # as far above as the put wall is below

        bars = fetch_rest_of_day(d, t0)
        if not bars:
            print(f"  {d} {t0} -- no bars (weekend/holiday/no data), skip")
            continue

        real_put_app, real_put_bounce   = approach_and_outcome(bars, put_wall, spot, "below")
        real_call_app, real_call_bounce = approach_and_outcome(bars, call_wall, spot, "above")
        plc_below_app, plc_below_bounce = approach_and_outcome(bars, placebo_below, spot, "below")
        plc_above_app, plc_above_bounce = approach_and_outcome(bars, placebo_above, spot, "above")

        results.append(dict(date=str(d), time=str(t0), spot=spot, regime=s["regime"],
                            real_put_app=real_put_app, real_put_bounce=real_put_bounce,
                            real_call_app=real_call_app, real_call_bounce=real_call_bounce,
                            plc_below_app=plc_below_app, plc_below_bounce=plc_below_bounce,
                            plc_above_app=plc_above_app, plc_above_bounce=plc_above_bounce))
        print(f"  {d} {t0}  spot={spot:.1f}  put_wall={put_wall} (app={real_put_app},bounce={real_put_bounce})  "
              f"call_wall={call_wall} (app={real_call_app},bounce={real_call_bounce})")
        time.sleep(0.1)

    print(f"\n{'='*70}\n{len(results)} snapshots with real bar data\n{'='*70}")

    def rate(key_app, key_bounce):
        app = [r for r in results if r[key_app]]
        n = len(app)
        if n == 0:
            return n, None
        bounces = sum(1 for r in app if r[key_bounce])
        return n, bounces / n

    n_put, r_put   = rate("real_put_app", "real_put_bounce")
    n_call, r_call = rate("real_call_app", "real_call_bounce")
    n_plc_b, r_plc_b = rate("plc_below_app", "plc_below_bounce")
    n_plc_a, r_plc_a = rate("plc_above_app", "plc_above_bounce")

    def fmt(n, r):
        return f"n={n:2d}  bounce_rate={r*100:.0f}%" if r is not None else f"n={n:2d}  (never approached)"

    print(f"REAL  put wall  (support):    {fmt(n_put, r_put)}")
    print(f"REAL  call wall (resistance): {fmt(n_call, r_call)}")
    print(f"PLACEBO below (same distance as call wall, mirrored): {fmt(n_plc_b, r_plc_b)}")
    print(f"PLACEBO above (same distance as put wall, mirrored):  {fmt(n_plc_a, r_plc_a)}")

    real_n = (n_put or 0) + (n_call or 0)
    plc_n  = (n_plc_b or 0) + (n_plc_a or 0)
    real_bounces = (r_put or 0) * (n_put or 0) + (r_call or 0) * (n_call or 0)
    plc_bounces  = (r_plc_b or 0) * (n_plc_b or 0) + (r_plc_a or 0) * (n_plc_a or 0)
    print(f"\nPOOLED: real walls {real_bounces:.0f}/{real_n} bounced "
          f"({(real_bounces/real_n*100 if real_n else 0):.0f}%) vs "
          f"placebo {plc_bounces:.0f}/{plc_n} bounced "
          f"({(plc_bounces/plc_n*100 if plc_n else 0):.0f}%)")
    print(f"\nSample size is tiny ({len(results)} snapshots, {len(set(r['date'] for r in results))} "
          f"distinct days) -- treat this as directional/qualitative, NOT a statistically powered result.")


if __name__ == "__main__":
    run()
