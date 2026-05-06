"""
Bull put spread backtest — weekly (7-DTE) or 0DTE, with optional take-profit.

Daily loop logic:
  1. If an open spread exists: check expiry OR intra-cycle take-profit.
  2. If no open spread: check VIX/regime gate, then open a new spread.

DTE modes:
  dte=7  — open every Friday, expire following Friday  (weekly)
  dte=0  — open every trading day, expire same day     (0DTE)

Take-profit:
  take_profit_pct=0.50  — close the spread when its residual value has fallen
                          to (1-0.50)*credit, i.e. we have locked in 50% of max
                          profit.  Frees capital for faster redeployment.
  take_profit_pct=None  — hold all spreads to expiry.

P&L at settlement (per share):
  = credit - max(0, K1-S) + max(0, K2-S)
  i.e. full credit if S>=K1, max-loss (credit-width) if S<=K2, else partial.
"""
import math
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.wheel_engine import bs_put

logger = logging.getLogger(__name__)

_RF      = 0.045
_SHARES  = 100


@dataclass
class SpreadTrade:
    entry_date:       pd.Timestamp
    expiry_date:      pd.Timestamp
    close_date:       pd.Timestamp        # actual close (TP or expiry)
    spot_entry:       float
    spot_close:       float
    short_strike:     float
    long_strike:      float
    spread_width:     float
    credit_per_share: float
    contracts:        int
    sigma:            float
    vix_at_entry:     float
    pnl:              float
    outcome:          str   # FULL_PROFIT | PARTIAL_LOSS | MAX_LOSS | TAKE_PROFIT | SKIPPED


@dataclass
class SpreadResult:
    equity_curve:            pd.Series
    trades:                  pd.DataFrame
    initial_capital:         float
    final_value:             float
    n_entered:               int
    n_skipped:               int
    n_full_profit:           int
    n_take_profit:           int
    n_partial_loss:          int
    n_max_loss:              int
    total_premium_collected: float
    total_losses:            float
    benchmark_curve:         pd.Series


class CreditSpreadEngine:
    def __init__(
        self,
        initial_capital:   float = 10_000.0,
        short_otm_pct:     float = 0.02,
        spread_width_pct:  float = 0.01,
        max_risk_pct:      float = 0.10,
        iv_premium:        float = 1.20,
        max_vix_entry:     float = 25.0,
        vol_window:        int   = 20,
        dte:               int   = 7,
        take_profit_pct:   Optional[float] = None,
    ):
        self.initial_capital  = initial_capital
        self.short_otm_pct    = short_otm_pct
        self.spread_width_pct = spread_width_pct
        self.max_risk_pct     = max_risk_pct
        self.iv_premium       = iv_premium
        self.max_vix_entry    = max_vix_entry
        self.vol_window       = vol_window
        self.dte              = dte
        self.take_profit_pct  = take_profit_pct

    # ── helpers ───────────────────────────────────────────────────────────────

    def _price_at(self, df: pd.DataFrame, date: pd.Timestamp, col: str = "close") -> Optional[float]:
        idx = df.index.get_indexer([date], method="ffill")
        if idx[0] == -1:
            return None
        c = col if col in df.columns else "close"
        return float(df[c].iloc[idx[0]])

    def _rv(self, df: pd.DataFrame, date: pd.Timestamp) -> float:
        sub = df[df.index <= date].tail(self.vol_window + 1)
        if len(sub) < 5:
            return 0.20
        return float(sub["close"].pct_change().dropna().std() * math.sqrt(252))

    def _sigma(self, vix: float, df: pd.DataFrame, date: pd.Timestamp) -> float:
        return max(vix / 100.0 * self.iv_premium, self._rv(df, date))

    def _round_strike(self, price: float, tick: float = 0.50) -> float:
        return round(price / tick) * tick

    def _spread_value(self, spot: float, K1: float, K2: float,
                      T_remaining: float, sigma: float) -> float:
        """Current market value of the short put spread (what we'd pay to close)."""
        if T_remaining <= 0:
            return max(0.0, K1 - spot) - max(0.0, K2 - spot)
        return bs_put(spot, K1, T_remaining, sigma) - bs_put(spot, K2, T_remaining, sigma)

    def _settle_pnl(self, spot: float, K1: float, K2: float,
                    credit: float, contracts: int) -> float:
        pnl_ps = credit - max(0.0, K1 - spot) + max(0.0, K2 - spot)
        return pnl_ps * _SHARES * contracts

    # ── main simulation ───────────────────────────────────────────────────────

    def run(
        self,
        spy_prices:    pd.DataFrame,
        macro_history: pd.DataFrame,
    ) -> SpreadResult:
        macro   = macro_history.reindex(spy_prices.index, method="ffill")
        dates   = spy_prices.index.sort_values()

        account      = self.initial_capital
        equity: dict = {}
        trades: list = []
        open_pos     = None   # SpreadTrade in progress (pnl/outcome/close_date TBD)
        open_sigma   = 0.0    # sigma used at entry (for TP repricing)

        n_entered = n_skipped = 0
        n_full = n_tp = n_partial = n_max = 0
        total_premium = total_losses = 0.0

        entry_time_s  = dict()   # date -> timestamp (seconds since epoch) for DTE calc

        for date in dates:
            spot = self._price_at(spy_prices, date)
            if spot is None:
                equity[date] = account
                continue

            row    = macro.loc[date] if date in macro.index else None
            vix    = float(row["vix_level"])  if row is not None and "vix_level"   in row.index else 20.0
            is_bear= bool(row["regime_bear"]) if row is not None and "regime_bear" in row.index else False

            # ── check / close open position ───────────────────────────────────
            if open_pos is not None:
                K1, K2   = open_pos.short_strike, open_pos.long_strike
                cr       = open_pos.credit_per_share
                contracts= open_pos.contracts
                expired  = date >= open_pos.expiry_date

                # Take-profit check (not applicable on 0DTE — no time left to check)
                tp_hit = False
                if (not expired) and (self.take_profit_pct is not None) and self.dte > 0:
                    days_elapsed = (date - open_pos.entry_date).days
                    if days_elapsed >= 1:   # don't check same day as entry
                        t_entry  = self.dte / 365
                        t_remain = max((open_pos.expiry_date - date).days / 365, 0)
                        current_val = self._spread_value(spot, K1, K2, t_remain, open_sigma)
                        if current_val <= cr * (1 - self.take_profit_pct):
                            tp_hit = True

                if expired or tp_hit:
                    pnl = self._settle_pnl(spot, K1, K2, cr, contracts)
                    account += pnl

                    if tp_hit:
                        outcome = "TAKE_PROFIT"
                        n_tp += 1
                        total_premium += pnl
                    elif spot >= K1:
                        outcome = "FULL_PROFIT"
                        n_full += 1
                        total_premium += pnl
                    elif spot <= K2:
                        outcome = "MAX_LOSS"
                        n_max += 1
                        total_losses += abs(pnl)
                    else:
                        outcome = "PARTIAL_LOSS"
                        n_partial += 1
                        if pnl < 0:
                            total_losses += abs(pnl)
                        else:
                            total_premium += pnl

                    open_pos.close_date = date
                    open_pos.spot_close = spot
                    open_pos.pnl        = pnl
                    open_pos.outcome    = outcome
                    trades.append(open_pos)
                    open_pos = None

            # ── open new position? ────────────────────────────────────────────
            if open_pos is None:
                # Weekly: only enter on Fridays (weekday==4); 0DTE: enter every day
                is_entry_day = (self.dte == 0) or (date.weekday() == 4)

                if not is_entry_day:
                    equity[date] = account
                    continue

                if vix > self.max_vix_entry or is_bear:
                    n_skipped += 1
                    equity[date] = account
                    continue

                sigma = self._sigma(vix, spy_prices, date)
                T     = max(self.dte, 0) / 365 if self.dte > 0 else (1 / 252)

                # 0DTE: price at open, settle at close — realistic intraday simulation
                entry_spot = self._price_at(spy_prices, date, col="open") or spot if self.dte == 0 else spot

                K1 = self._round_strike(entry_spot * (1 - self.short_otm_pct))
                sw = max(self._round_strike(entry_spot * self.spread_width_pct), 1.0)
                K2 = K1 - sw

                credit = bs_put(entry_spot, K1, T, sigma) - bs_put(entry_spot, K2, T, sigma)
                credit = max(credit, 0.005)

                max_loss_pc = (sw - credit) * _SHARES
                if max_loss_pc <= 0:
                    equity[date] = account
                    continue

                contracts = max(int(account * self.max_risk_pct // max_loss_pc), 1)

                if self.dte == 0:
                    expiry = date          # 0DTE: expire same day
                else:
                    # Next Friday within ~10 calendar days
                    future  = dates[dates > date]
                    fridays = future[future.day_of_week == 4]
                    if len(fridays) == 0:
                        expiry = date + pd.Timedelta(days=7)
                    else:
                        week_fridays = fridays[fridays <= date + pd.Timedelta(days=10)]
                        expiry = week_fridays[-1] if len(week_fridays) > 0 else fridays[0]

                open_pos = SpreadTrade(
                    entry_date=date, expiry_date=expiry, close_date=date,
                    spot_entry=entry_spot, spot_close=0.0,
                    short_strike=K1, long_strike=K2, spread_width=sw,
                    credit_per_share=credit, contracts=contracts,
                    sigma=sigma, vix_at_entry=vix,
                    pnl=0.0, outcome="",
                )
                open_sigma = sigma
                n_entered += 1

                # 0DTE: immediately settle at today's close
                if self.dte == 0:
                    pnl = self._settle_pnl(spot, K1, K2, credit, contracts)
                    account += pnl

                    if spot >= K1:
                        outcome = "FULL_PROFIT"; n_full += 1; total_premium += pnl
                    elif spot <= K2:
                        outcome = "MAX_LOSS";    n_max  += 1; total_losses  += abs(pnl)
                    else:
                        outcome = "PARTIAL_LOSS"; n_partial += 1
                        if pnl < 0: total_losses += abs(pnl)
                        else: total_premium += pnl

                    open_pos.close_date = date
                    open_pos.spot_close = spot
                    open_pos.pnl        = pnl
                    open_pos.outcome    = outcome
                    trades.append(open_pos)
                    open_pos = None

            equity[date] = account

        eq = pd.Series(equity).sort_index()

        bm_start  = self._price_at(spy_prices, eq.index[0]) or 1.0
        benchmark = pd.Series({
            d: (self._price_at(spy_prices, d) or bm_start) / bm_start * self.initial_capital
            for d in eq.index
        })

        trades_df = pd.DataFrame([vars(t) for t in trades if t.outcome != "SKIPPED"])

        return SpreadResult(
            equity_curve=eq,
            trades=trades_df,
            initial_capital=self.initial_capital,
            final_value=float(eq.iloc[-1]) if not eq.empty else self.initial_capital,
            n_entered=n_entered,
            n_skipped=n_skipped,
            n_full_profit=n_full,
            n_take_profit=n_tp,
            n_partial_loss=n_partial,
            n_max_loss=n_max,
            total_premium_collected=total_premium,
            total_losses=total_losses,
            benchmark_curve=benchmark,
        )
