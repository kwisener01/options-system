#!/usr/bin/env python3
"""
Fallen Angel backtest — walk-forward test of the price-only scoring model.

For each ticker in the universe, downloads 5 years of daily history and
simulates entries when the price score >= entry_threshold (default 5/8).
Exits on the first of:
  - +20% gain from entry price (take-profit)
  - -25% loss from entry price (stop-loss)
  - 180 calendar-day max hold (time stop)

Options flow (Phase 2 scoring) is NOT available historically, so the
backtest uses the 8-point price/RSI/volume score only.

Usage:
    python run_fallen_angel_backtest.py                    # full run
    python run_fallen_angel_backtest.py --years 3          # last 3 years
    python run_fallen_angel_backtest.py --tp 0.30          # 30% TP target
    python run_fallen_angel_backtest.py --ticker INTC PFE  # specific tickers
    python run_fallen_angel_backtest.py --cache-dir data/fa_cache  # use/save cached data
"""
import argparse
import json
import logging
import math
import os
import sys
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

UNIVERSE = [
    "BAC", "JPM", "WFC", "GS", "MS", "C", "AXP", "BK", "SCHW", "PRU", "MET",
    "INTC", "QCOM", "CSCO", "IBM", "HPQ", "DELL", "STX", "WDC", "NTAP", "AMAT",
    "PFE", "MRK", "ABBV", "BMY", "AMGN", "GILD", "CVS", "WBA", "HUM", "CI",
    "CVX", "XOM", "COP", "PSX", "VLO", "HAL", "SLB", "OXY", "DVN", "MPC",
    "KO", "PEP", "PM", "MO", "KR", "GIS", "K", "CAG", "SJM", "CPB",
    "T", "VZ", "PARA", "WBD", "FOX", "IPG", "OMC",
    "GE", "HON", "MMM", "RTX", "LMT", "BA", "CAT", "DE", "EMR", "ETN",
    "DOW", "LYB", "CF", "MOS", "NUE", "X", "CLF", "FCX", "NEM", "AA",
    "SO", "D", "DUK", "EXC", "PCG", "XEL", "AES", "NRG",
    "VNO", "SLG", "MPW", "NYCB", "KIM",
]

DECLINE_DEEP = 0.50
DECLINE_MID  = 0.40
DECLINE_SOFT = 0.30
FLOOR_THRESHOLD = 0.10
FLOOR_NEARBY    = 0.20
OVERSOLD_RSI    = 30
RECOVERING_RSI  = (35, 55)
VOL_STRONG = 1.25
VOL_MILD   = 1.10


@dataclass
class Trade:
    ticker:      str
    entry_date:  str
    entry_price: float
    exit_date:   str = ""
    exit_price:  float = 0.0
    exit_reason: str = ""
    pnl_pct:     float = 0.0
    hold_days:   int = 0
    score:       int = 0


@dataclass
class BacktestResult:
    trades:       list = field(default_factory=list)
    ticker_stats: dict = field(default_factory=dict)
    summary:      dict = field(default_factory=dict)


def _rsi_series(closes: list[float], period: int = 14) -> list[float]:
    if len(closes) < period + 2:
        return [50.0] * len(closes)
    ch = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains  = [max(c, 0) for c in ch]
    losses = [max(-c, 0) for c in ch]
    ag = sum(gains[:period]) / period
    al = sum(losses[:period]) / period
    pad = [50.0] * (period + 1)
    out = []
    for i in range(period, len(ch)):
        ag = (ag * (period - 1) + gains[i]) / period
        al = (al * (period - 1) + losses[i]) / period
        rs = ag / al if al else 100.0
        out.append(round(100 - 100 / (1 + rs), 1))
    return pad + out


def _fetch_yahoo(ticker: str, years: int = 5) -> pd.DataFrame:
    end = int(datetime.now().timestamp())
    start = int((datetime.now() - timedelta(days=years * 365)).timestamp())
    url = (f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?period1={start}&period2={end}&interval=1d")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        import ssl
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            data = json.loads(resp.read())
    except Exception:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())

    result = data["chart"]["result"][0]
    ts = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    df = pd.DataFrame({
        "date":   pd.to_datetime(ts, unit="s").normalize(),
        "close":  quote["close"],
        "volume": quote["volume"],
        "high":   quote["high"],
        "low":    quote["low"],
    }).dropna(subset=["close"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _load_ticker(ticker: str, years: int, cache_dir: str | None) -> pd.DataFrame:
    if cache_dir:
        path = os.path.join(cache_dir, f"{ticker}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["date"])
            logger.info("  %s: loaded %d rows from cache", ticker, len(df))
            return df

    logger.info("  %s: downloading %d years from Yahoo...", ticker, years)
    df = _fetch_yahoo(ticker, years)

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        df.to_csv(os.path.join(cache_dir, f"{ticker}.csv"), index=False)

    return df


def _score_day(closes: list[float], volumes: list[float], rsi_vals: list[float],
               hi_252: float, lo_252: float, idx: int) -> tuple[int, list[str]]:
    spot = closes[idx]
    if hi_252 <= 0 or lo_252 <= 0:
        return 0, []

    pct_from_high = (spot - hi_252) / hi_252
    pct_from_low  = (spot - lo_252) / lo_252

    if pct_from_high > -DECLINE_SOFT:
        return 0, []

    score = 0
    reasons = []
    decline = abs(pct_from_high)

    if decline >= DECLINE_DEEP:
        score += 3
        reasons.append(f"Down {decline*100:.1f}% from 52w high")
    elif decline >= DECLINE_MID:
        score += 2
        reasons.append(f"Down {decline*100:.1f}% from 52w high")
    elif decline >= DECLINE_SOFT:
        score += 1
        reasons.append(f"Down {decline*100:.1f}% from 52w high")

    if pct_from_low <= FLOOR_THRESHOLD:
        score += 2
        reasons.append(f"Within {pct_from_low*100:.1f}% of 52w low")
    elif pct_from_low <= FLOOR_NEARBY:
        score += 1
        reasons.append(f"{pct_from_low*100:.1f}% above 52w low")

    rsi_now = rsi_vals[idx] if idx < len(rsi_vals) else 50.0
    recent_rsi = rsi_vals[max(0, idx-20):idx+1]
    was_oversold = any(r < OVERSOLD_RSI for r in recent_rsi)
    lo_r, hi_r = RECOVERING_RSI

    if was_oversold and lo_r <= rsi_now <= hi_r:
        score += 3
        reasons.append(f"RSI {rsi_now} recovering from oversold")
    elif rsi_now < OVERSOLD_RSI:
        score += 2
        reasons.append(f"RSI {rsi_now} oversold")
    elif rsi_now < 45:
        score += 1
        reasons.append(f"RSI {rsi_now} below midline")

    if idx >= 30 and len(volumes) > idx:
        avg10 = np.mean(volumes[idx-9:idx+1])
        avg30 = np.mean(volumes[idx-29:idx+1])
        vol_ratio = avg10 / avg30 if avg30 > 0 else 1.0
        if vol_ratio >= VOL_STRONG:
            score += 2
            reasons.append(f"Volume {vol_ratio:.2f}x avg")
        elif vol_ratio >= VOL_MILD:
            score += 1
            reasons.append(f"Volume {vol_ratio:.2f}x avg")

    return score, reasons


def backtest_ticker(df: pd.DataFrame, ticker: str,
                    tp_pct: float = 0.20, sl_pct: float = 0.25,
                    max_hold: int = 180, entry_threshold: int = 5,
                    cooldown: int = 20) -> list[Trade]:
    closes  = df["close"].tolist()
    volumes = df["volume"].tolist()
    highs   = df["high"].tolist()
    lows    = df["low"].tolist()
    dates   = df["date"].tolist()
    n = len(closes)

    if n < 252:
        return []

    rsi_vals = _rsi_series(closes)

    trades: list[Trade] = []
    in_trade = False
    entry_price = 0.0
    entry_idx = 0
    entry_score = 0
    last_exit_idx = -cooldown

    for i in range(252, n):
        if in_trade:
            spot = closes[i]
            ret = (spot - entry_price) / entry_price
            hold = (dates[i] - dates[entry_idx]).days

            if ret >= tp_pct:
                trades.append(Trade(
                    ticker=ticker, entry_date=str(dates[entry_idx].date()),
                    entry_price=round(entry_price, 2),
                    exit_date=str(dates[i].date()), exit_price=round(spot, 2),
                    exit_reason="TP", pnl_pct=round(ret * 100, 2),
                    hold_days=hold, score=entry_score,
                ))
                in_trade = False
                last_exit_idx = i
            elif ret <= -sl_pct:
                trades.append(Trade(
                    ticker=ticker, entry_date=str(dates[entry_idx].date()),
                    entry_price=round(entry_price, 2),
                    exit_date=str(dates[i].date()), exit_price=round(spot, 2),
                    exit_reason="SL", pnl_pct=round(ret * 100, 2),
                    hold_days=hold, score=entry_score,
                ))
                in_trade = False
                last_exit_idx = i
            elif hold >= max_hold:
                trades.append(Trade(
                    ticker=ticker, entry_date=str(dates[entry_idx].date()),
                    entry_price=round(entry_price, 2),
                    exit_date=str(dates[i].date()), exit_price=round(spot, 2),
                    exit_reason="TIME", pnl_pct=round(ret * 100, 2),
                    hold_days=hold, score=entry_score,
                ))
                in_trade = False
                last_exit_idx = i
            continue

        if i - last_exit_idx < cooldown:
            continue

        hi_252 = max(highs[max(0, i-252):i])
        lo_252 = min(lows[max(0, i-252):i])

        score, reasons = _score_day(closes, volumes, rsi_vals, hi_252, lo_252, i)

        if score >= entry_threshold:
            in_trade = True
            entry_price = closes[i]
            entry_idx = i
            entry_score = score

    if in_trade:
        spot = closes[-1]
        ret = (spot - entry_price) / entry_price
        hold = (dates[-1] - dates[entry_idx]).days
        trades.append(Trade(
            ticker=ticker, entry_date=str(dates[entry_idx].date()),
            entry_price=round(entry_price, 2),
            exit_date=str(dates[-1].date()), exit_price=round(spot, 2),
            exit_reason="OPEN", pnl_pct=round(ret * 100, 2),
            hold_days=hold, score=entry_score,
        ))

    return trades


def _ticker_summary(trades: list[Trade]) -> dict:
    if not trades:
        return {}
    closed = [t for t in trades if t.exit_reason != "OPEN"]
    if not closed:
        return {"trades": len(trades), "closed": 0}
    wins   = [t for t in closed if t.pnl_pct > 0]
    losses = [t for t in closed if t.pnl_pct <= 0]
    pnls   = [t.pnl_pct for t in closed]
    holds  = [t.hold_days for t in closed]
    tp_hits = sum(1 for t in closed if t.exit_reason == "TP")
    sl_hits = sum(1 for t in closed if t.exit_reason == "SL")
    time_exits = sum(1 for t in closed if t.exit_reason == "TIME")
    return {
        "trades":     len(closed),
        "wins":       len(wins),
        "losses":     len(losses),
        "win_rate":   round(len(wins) / len(closed), 3),
        "avg_pnl":    round(np.mean(pnls), 2),
        "total_pnl":  round(sum(pnls), 2),
        "best":       round(max(pnls), 2),
        "worst":      round(min(pnls), 2),
        "avg_hold":   round(np.mean(holds), 1),
        "tp_exits":   tp_hits,
        "sl_exits":   sl_hits,
        "time_exits": time_exits,
    }


def run_backtest(tickers: list[str] | None = None, years: int = 5,
                 tp_pct: float = 0.20, sl_pct: float = 0.25,
                 max_hold: int = 180, entry_threshold: int = 5,
                 cache_dir: str | None = None) -> BacktestResult:
    tickers = tickers or UNIVERSE
    all_trades: list[Trade] = []
    ticker_stats: dict[str, dict] = {}

    logger.info("=" * 70)
    logger.info("  FALLEN ANGEL BACKTEST")
    logger.info("  %d tickers | %d years | TP +%d%% | SL -%d%% | %dd max hold",
                len(tickers), years, int(tp_pct*100), int(sl_pct*100), max_hold)
    logger.info("  Entry threshold: %d/8 (price score only, no options flow)", entry_threshold)
    logger.info("=" * 70)
    logger.info("")

    failed = []
    for ticker in tickers:
        try:
            df = _load_ticker(ticker, years, cache_dir)
            trades = backtest_ticker(df, ticker, tp_pct, sl_pct, max_hold, entry_threshold)
            all_trades.extend(trades)
            stats = _ticker_summary(trades)
            if stats:
                ticker_stats[ticker] = stats
                ct = stats.get("closed", stats.get("trades", 0))
                wr = stats.get("win_rate", 0)
                tp = stats.get("total_pnl", 0)
                logger.info("  %s: %d trades, %.0f%% win rate, %+.1f%% total", ticker, ct, wr*100, tp)
        except Exception as e:
            failed.append(ticker)
            logger.warning("  %s: FAILED — %s", ticker, e)

    closed = [t for t in all_trades if t.exit_reason != "OPEN"]
    if not closed:
        logger.info("\nNo closed trades found.")
        return BacktestResult(all_trades, ticker_stats, {})

    wins   = [t for t in closed if t.pnl_pct > 0]
    losses = [t for t in closed if t.pnl_pct <= 0]
    pnls   = [t.pnl_pct for t in closed]
    holds  = [t.hold_days for t in closed]
    tp_exits   = sum(1 for t in closed if t.exit_reason == "TP")
    sl_exits   = sum(1 for t in closed if t.exit_reason == "SL")
    time_exits = sum(1 for t in closed if t.exit_reason == "TIME")

    avg_win  = np.mean([t.pnl_pct for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0
    profit_factor = abs(sum(t.pnl_pct for t in wins) / sum(t.pnl_pct for t in losses)) if losses else float("inf")

    summary = {
        "total_trades": len(closed),
        "wins":         len(wins),
        "losses":       len(losses),
        "win_rate":     round(len(wins) / len(closed), 3),
        "avg_pnl":      round(np.mean(pnls), 2),
        "median_pnl":   round(np.median(pnls), 2),
        "total_pnl":    round(sum(pnls), 2),
        "avg_win":      round(avg_win, 2),
        "avg_loss":     round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "best_trade":   round(max(pnls), 2),
        "worst_trade":  round(min(pnls), 2),
        "avg_hold":     round(np.mean(holds), 1),
        "tp_exits":     tp_exits,
        "sl_exits":     sl_exits,
        "time_exits":   time_exits,
        "failed_tickers": failed,
    }

    # Per-sector breakdown
    sectors = {
        "Financials":  ["BAC","JPM","WFC","GS","MS","C","AXP","BK","SCHW","PRU","MET"],
        "Technology":  ["INTC","QCOM","CSCO","IBM","HPQ","DELL","STX","WDC","NTAP","AMAT"],
        "Healthcare":  ["PFE","MRK","ABBV","BMY","AMGN","GILD","CVS","WBA","HUM","CI"],
        "Energy":      ["CVX","XOM","COP","PSX","VLO","HAL","SLB","OXY","DVN","MPC"],
        "Consumer":    ["KO","PEP","PM","MO","KR","GIS","K","CAG","SJM","CPB"],
        "Telecom":     ["T","VZ","PARA","WBD","FOX","IPG","OMC"],
        "Industrials": ["GE","HON","MMM","RTX","LMT","BA","CAT","DE","EMR","ETN"],
        "Materials":   ["DOW","LYB","CF","MOS","NUE","X","CLF","FCX","NEM","AA"],
        "Utilities":   ["SO","D","DUK","EXC","PCG","XEL","AES","NRG"],
        "REITs":       ["VNO","SLG","MPW","NYCB","KIM"],
    }
    sector_stats = {}
    for sec, syms in sectors.items():
        sec_trades = [t for t in closed if t.ticker in syms]
        if sec_trades:
            sec_wins = sum(1 for t in sec_trades if t.pnl_pct > 0)
            sec_pnls = [t.pnl_pct for t in sec_trades]
            sector_stats[sec] = {
                "trades":   len(sec_trades),
                "win_rate": round(sec_wins / len(sec_trades), 3),
                "avg_pnl":  round(np.mean(sec_pnls), 2),
                "total_pnl": round(sum(sec_pnls), 2),
            }
    summary["sector_stats"] = sector_stats

    return BacktestResult(all_trades, ticker_stats, summary)


def print_report(result: BacktestResult, tp_pct: float):
    s = result.summary
    if not s:
        print("No results to report.")
        return

    print(f"\n{'='*70}")
    print(f"  FALLEN ANGEL BACKTEST RESULTS")
    print(f"{'='*70}\n")

    print(f"  Total closed trades:  {s['total_trades']}")
    print(f"  Wins / Losses:        {s['wins']}W / {s['losses']}L")
    print(f"  Win rate:             {s['win_rate']*100:.1f}%")
    print(f"  Avg P&L per trade:    {s['avg_pnl']:+.2f}%")
    print(f"  Median P&L:           {s['median_pnl']:+.2f}%")
    print(f"  Avg win:              {s['avg_win']:+.2f}%")
    print(f"  Avg loss:             {s['avg_loss']:+.2f}%")
    print(f"  Profit factor:        {s['profit_factor']:.2f}")
    print(f"  Best trade:           {s['best_trade']:+.2f}%")
    print(f"  Worst trade:          {s['worst_trade']:+.2f}%")
    print(f"  Avg hold:             {s['avg_hold']:.0f} days")
    print(f"  Exit breakdown:       TP(+{int(tp_pct*100)}%): {s['tp_exits']}  "
          f"SL: {s['sl_exits']}  TIME: {s['time_exits']}")
    print()

    if s.get("sector_stats"):
        print(f"  {'SECTOR BREAKDOWN':^66}")
        print(f"  {'-'*66}")
        print(f"  {'Sector':16s} {'Trades':>7s} {'Win%':>6s} {'Avg P&L':>9s} {'Total P&L':>10s}")
        for sec, st in sorted(s["sector_stats"].items(), key=lambda x: -x[1]["total_pnl"]):
            print(f"  {sec:16s} {st['trades']:7d} {st['win_rate']*100:5.1f}% "
                  f"{st['avg_pnl']:+8.2f}% {st['total_pnl']:+9.2f}%")
        print()

    # Top and bottom tickers
    ranked = sorted(result.ticker_stats.items(), key=lambda x: -x[1].get("total_pnl", 0))
    if ranked:
        top5 = ranked[:5]
        bot5 = ranked[-5:]
        print(f"  {'TOP 5 TICKERS':^66}")
        print(f"  {'-'*66}")
        for tk, st in top5:
            print(f"  {tk:6s}  {st.get('trades',0):3d} trades  "
                  f"{st.get('win_rate',0)*100:5.1f}% win  "
                  f"avg {st.get('avg_pnl',0):+.2f}%  total {st.get('total_pnl',0):+.2f}%")
        print()
        print(f"  {'BOTTOM 5 TICKERS':^66}")
        print(f"  {'-'*66}")
        for tk, st in bot5:
            print(f"  {tk:6s}  {st.get('trades',0):3d} trades  "
                  f"{st.get('win_rate',0)*100:5.1f}% win  "
                  f"avg {st.get('avg_pnl',0):+.2f}%  total {st.get('total_pnl',0):+.2f}%")

    # Verdict
    print(f"\n  {'VERDICT':^66}")
    print(f"  {'-'*66}")
    wr = s["win_rate"]
    pf = s["profit_factor"]
    if wr >= 0.55 and pf >= 1.5:
        print("  VIABLE — positive edge with acceptable win rate and profit factor.")
    elif wr >= 0.45 and pf >= 1.0:
        print("  MARGINAL — small edge, may improve with options flow overlay.")
    else:
        print("  WEAK — price-only scoring lacks sufficient edge. Needs options flow.")

    if s.get("failed_tickers"):
        print(f"\n  Failed tickers ({len(s['failed_tickers'])}): {', '.join(s['failed_tickers'])}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Fallen Angel backtest")
    parser.add_argument("--years", type=int, default=5, help="Years of history (default 5)")
    parser.add_argument("--tp", type=float, default=0.20, help="Take-profit %% (default 0.20)")
    parser.add_argument("--sl", type=float, default=0.25, help="Stop-loss %% (default 0.25)")
    parser.add_argument("--max-hold", type=int, default=180, help="Max hold days (default 180)")
    parser.add_argument("--threshold", type=int, default=5, help="Entry score threshold out of 8 (default 5)")
    parser.add_argument("--ticker", nargs="+", help="Specific tickers to test")
    parser.add_argument("--cache-dir", help="Cache directory for downloaded data")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    result = run_backtest(
        tickers=args.ticker,
        years=args.years,
        tp_pct=args.tp,
        sl_pct=args.sl,
        max_hold=args.max_hold,
        entry_threshold=args.threshold,
        cache_dir=args.cache_dir,
    )

    if args.json:
        out = {
            "summary": result.summary,
            "ticker_stats": result.ticker_stats,
            "trades": [
                {"ticker": t.ticker, "entry_date": t.entry_date,
                 "entry_price": t.entry_price, "exit_date": t.exit_date,
                 "exit_price": t.exit_price, "exit_reason": t.exit_reason,
                 "pnl_pct": t.pnl_pct, "hold_days": t.hold_days, "score": t.score}
                for t in result.trades
            ],
        }
        json.dump(out, sys.stdout, indent=2)
        print()
    else:
        print_report(result, args.tp)


if __name__ == "__main__":
    main()
