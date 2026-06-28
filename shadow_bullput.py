"""
SHADOW LOGGER for the live bull-put strategy.

We can't BACKTEST the bull put (its put-wall-anchored strikes need historical OI
Alpaca doesn't keep). So we FORWARD-test it: each market day log the scanner's
REAL picks (paper, no orders), then mark them to market with real option bars and
the live 50%/1-DTE management. After ~6-8 weeks this is an exact, trustworthy
track record of the actual strategy — no proxy, no guessing.

  python shadow_bullput.py --log     # append today's STRONG bull-put picks
  python shadow_bullput.py --mark    # mark all picks to market, print track record
  python shadow_bullput.py --add 2026-06-25,T,23,22,2026-07-17,0.53   # seed a known pick

Durable store: shadow_picks.csv. On Render's ephemeral disk this resets on
redeploy, so the /cron/shadow endpoint also posts each pick to Slack as a backup.
"""
import argparse, csv, json, os, sys, time, hashlib
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(__file__))
import requests
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY
from src.live.alpaca_options import occ_symbol

ET   = ZoneInfo("America/New_York")
OPT  = "https://data.alpaca.markets/v1beta1/options"
HEAD = {"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY}
CSV  = os.path.join(os.path.dirname(__file__), "shadow_picks.csv")
COLS = ["date", "ticker", "short", "long", "expiry", "width", "entry_credit",
        "credit_pct", "signal", "status", "exit_date", "exit_reason", "pnl"]
PROFIT_TARGET = 0.50
SLIP = 0.03
_CACHE = os.path.join(os.path.dirname(__file__), ".cache", "shadow")
os.makedirs(_CACHE, exist_ok=True)


# ── csv store ──────────────────────────────────────────────────────────────────

def _load():
    if not os.path.exists(CSV):
        return []
    with open(CSV, newline="") as f:
        return list(csv.DictReader(f))


def _save(rows):
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)


def _add_row(row):
    rows = _load()
    key = (row["date"], row["ticker"], row["expiry"])
    if any((r["date"], r["ticker"], r["expiry"]) == key for r in rows):
        return False                                   # de-dupe one pick per ticker/expiry/day
    rows.append({c: row.get(c, "") for c in COLS})
    _save(rows)
    return True


# ── data ───────────────────────────────────────────────────────────────────────

def _opt_daily(occ, start, end):
    key = hashlib.md5(f"{occ}|{start}|{end}".encode()).hexdigest()
    fp  = os.path.join(_CACHE, f"{key}.json")
    data = None
    if os.path.exists(fp) and end < date.today().isoformat():   # cache only closed ranges
        with open(fp) as f:
            data = json.load(f)
    if data is None:
        r = requests.get(f"{OPT}/bars", headers=HEAD, params={
            "symbols": occ, "timeframe": "1Day", "start": start, "end": end, "limit": 1000}, timeout=25)
        data = r.json() if r.status_code == 200 else {}
        with open(fp, "w") as f:
            json.dump(data, f)
    out = {}
    for b in (data.get("bars") or {}).get(occ, []):
        out[b["t"][:10]] = (b["c"], b.get("vw", b["c"]))
    return out


def get_vix():
    try:
        from src.live.yahoo import fetch_daily_ohlcv
        v = fetch_daily_ohlcv("^VIX", "1mo")
        cl = [float(x) for x in v["close"].tolist()]
        return (cl[-1], cl[-2]) if len(cl) >= 2 else (0.0, 0.0)
    except Exception:
        return 0.0, 0.0


# ── log today's picks ───────────────────────────────────────────────────────────

def log_picks(slip=SLIP):
    from src.analysis.bull_put_scanner import scan
    vn, vp = get_vix()
    results = scan(vix_now=vn, vix_prev=vp)
    today = datetime.now(ET).date().isoformat()
    added = []
    for r in results:
        if r.get("signal") != "STRONG" or not r.get("candidate"):
            continue
        c = r["candidate"]
        credit = round(float(c["credit"]) - 2 * slip, 3)     # paper entry, slippage-haircut
        row = {"date": today, "ticker": r["ticker"],
               "short": c["short_strike"], "long": c["long_strike"],
               "expiry": c["expiry"], "width": c["width"],
               "entry_credit": credit, "credit_pct": c["credit_pct"],
               "signal": "STRONG", "status": "open",
               "exit_date": "", "exit_reason": "", "pnl": ""}
        if _add_row(row):
            added.append(row)
    return added


# ── mark to market ───────────────────────────────────────────────────────────────

def mark(slip=SLIP):
    rows = _load()
    yest = (datetime.now(ET).date() - timedelta(days=1)).isoformat()   # options to 'today' need OPRA
    for r in rows:
        if r["status"] == "closed":
            continue
        tk, exp = r["ticker"], r["expiry"]
        sk, lk  = float(r["short"]), float(r["long"])
        w       = float(r["width"]); credit = float(r["entry_credit"])
        ssym = occ_symbol(tk, date.fromisoformat(exp), "PUT", sk)
        lsym = occ_symbol(tk, date.fromisoformat(exp), "PUT", lk)
        end  = min(exp, yest)
        sb, lb = _opt_daily(ssym, r["date"], end), _opt_daily(lsym, r["date"], end)
        hold = sorted(d for d in sb if d in lb and d > r["date"])
        for d in hold:
            spread = sb[d][0] - lb[d][0]
            cost   = spread + 2 * slip
            if credit - cost >= PROFIT_TARGET * (credit + 2 * slip):
                r.update(status="closed", exit_date=d, exit_reason="target",
                         pnl=round((credit - cost) * 100, 2)); break
            if (date.fromisoformat(exp) - date.fromisoformat(d)).days <= 1:
                r.update(status="closed", exit_date=d, exit_reason="force_1dte",
                         pnl=round((credit - cost) * 100, 2)); break
        if r["status"] != "closed" and hold:               # still open → mark unrealized
            d = hold[-1]; spread = sb[d][0] - lb[d][0]
            r["pnl"] = round((credit - (spread + 2 * slip)) * 100, 2)
    _save(rows)
    return _summary(rows)


def _summary(rows):
    closed = [r for r in rows if r["status"] == "closed"]
    openp  = [r for r in rows if r["status"] != "closed"]
    lines = [f"*Shadow bull-put track record* — {len(rows)} picks "
             f"({len(closed)} closed, {len(openp)} open)"]
    if closed:
        n = len(closed)
        wins = sum(1 for r in closed if float(r["pnl"] or 0) > 0)
        tot  = sum(float(r["pnl"] or 0) for r in closed)
        lines.append(f"Closed: {wins}/{n} wins ({wins/n*100:.0f}%)  |  realized ${tot:+,.0f}  "
                     f"|  avg ${tot/n:+.2f}/trade")
    if openp:
        un = sum(float(r["pnl"] or 0) for r in openp if r["pnl"] != "")
        lines.append(f"Open: {len(openp)}  |  unrealized ${un:+,.0f}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--mark", action="store_true")
    ap.add_argument("--add", help="date,ticker,short,long,expiry,credit")
    a = ap.parse_args()
    if a.add:
        d, tk, sk, lk, exp, cr = a.add.split(",")
        w = round(float(sk) - float(lk), 2)
        _add_row({"date": d, "ticker": tk.upper(), "short": sk, "long": lk, "expiry": exp,
                  "width": w, "entry_credit": cr, "credit_pct": round(float(cr)/w*100, 1),
                  "signal": "STRONG", "status": "open", "exit_date": "", "exit_reason": "", "pnl": ""})
        print("added.")
    if a.log:
        added = log_picks()
        print(f"logged {len(added)} new STRONG pick(s):")
        for r in added:
            print(f"  {r['ticker']} {r['short']}/{r['long']}P {r['expiry']} "
                  f"credit ${r['entry_credit']} ({r['credit_pct']}%)")
    if a.mark:
        print(mark())
