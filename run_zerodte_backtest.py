"""
ROUGH backtest of the 0DTE SPY put-spread-at-the-expected-move strategy.

Honest about its limits: there are NO historical option chains here, so the
CREDIT is MODELED (Black-Scholes with VIX as the vol input) — treat the dollar
P&L as indicative, not gospel. The WIN RATE, however, is pure price data: did
SPY close above the 1-SD-down short strike? That answers "is the premise real?"

Each day: enter ~10:00 a $1-wide SPY put spread, short ~1 expected-move below the
open, hold to the cash close (0DTE). Reports win rate, modeled expectancy, worst
day, and max drawdown over the last ~365 days.
"""
import math, os, sys
sys.path.insert(0, os.path.dirname(__file__))

WIDTH        = 1.0
T_TO_CLOSE   = 5.5 / (6.5 * 252)     # ~10:00 entry to cash close, in years
R            = 0.0


def _ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _bs_put(S, K, T, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (R + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-R * T) * _ncdf(-d2) - S * _ncdf(-d1)


def main():
    from src.live.yahoo import fetch_daily_ohlcv
    spy = fetch_daily_ohlcv("SPY", "1y")
    vix = fetch_daily_ohlcv("^VIX", "1y")
    if spy.empty or vix.empty:
        print("data fetch failed"); return
    vix_close = {d.date(): float(v) for d, v in vix["close"].items()}

    trades = []
    for ts, row in spy.iterrows():
        d = ts.date()
        if d not in vix_close:
            continue
        o, lo, c = float(row["open"]), float(row["low"]), float(row["close"])
        sigma = vix_close[d] / 100.0
        em    = o * sigma * math.sqrt(T_TO_CLOSE)        # 1-SD move to close
        short = round(o - em)
        long_ = short - WIDTH
        # modeled entry credit
        credit = max(_bs_put(o, short, T_TO_CLOSE, sigma) - _bs_put(o, long_, T_TO_CLOSE, sigma), 0.0)
        if credit < 0.12 * WIDTH:        # same thin-credit filter as live
            continue
        # settle at cash close (0DTE)
        settle = max(0.0, min(WIDTH, max(short - c, 0) - max(long_ - c, 0)))
        pnl    = (credit - settle) * 100
        trades.append({"date": d, "win": c >= short, "breach": lo < short,
                       "credit": credit, "pnl": pnl})

    if not trades:
        print("no trades"); return
    n      = len(trades)
    wins   = sum(t["win"] for t in trades)
    total  = sum(t["pnl"] for t in trades)
    avg    = total / n
    worst  = min(t["pnl"] for t in trades)
    best   = max(t["pnl"] for t in trades)
    avgcr  = sum(t["credit"] for t in trades) / n
    # equity curve / max drawdown
    eq, peak, mdd = 0.0, 0.0, 0.0
    for t in trades:
        eq += t["pnl"]; peak = max(peak, eq); mdd = min(mdd, eq - peak)
    # recovered: dipped below short intraday but closed above (don't-close-losers payoff)
    recovered = sum(1 for t in trades if t["breach"] and t["win"])

    print(f"\n=== 0DTE SPY put-spread @ expected move — ROUGH backtest (1y) ===")
    print(f"  Trades                 : {n}")
    print(f"  Win rate (close>short) : {wins/n*100:.1f}%   <-- PURE DATA")
    print(f"  Avg modeled credit     : ${avgcr:.2f}  (max profit ${avgcr*100:.0f}/spread)")
    print(f"  --- modeled P&L (credit is BS-estimated, indicative only) ---")
    print(f"  Total per 1-lot/day    : ${total:,.0f}")
    print(f"  Avg / trade            : ${avg:+.2f}")
    print(f"  Best / worst day       : ${best:+.0f} / ${worst:+.0f}")
    print(f"  Max drawdown (cum)     : ${mdd:,.0f}")
    print(f"  Loss days              : {n-wins} ({(n-wins)/n*100:.0f}%)")
    print(f"  Dipped-then-recovered  : {recovered}  (days the 'don't close losers' rule paid)")
    print(f"  NOTE: no 50% management or slippage modeled; hold-to-close only.\n")


if __name__ == "__main__":
    main()
