"""
Multi-strategy options engine for XSP/SPY.

Regime → Strategy map
─────────────────────────────────────────────────────────────
VIX > 30                            → CASH      (black-swan protection)
BEAR  + VIX > max_vix_entry         → CASH
BEAR  + VIX OK                      → BEAR_CALL_SPREAD  (sell OTM calls)
VOL_CRUSH (prev VIX>30 → now <25)   → BULL_CALL_DEBIT   (buy cheap calls, profit from vol collapse)
BULL  + VIX > max_vix_entry         → CASH
BULL  + VIX OK                      → BULL_PUT_SPREAD   (sell OTM puts)
NEUTRAL + VIX < low_vol_threshold   → IRON_CONDOR       (sell both sides, market going nowhere)
NEUTRAL + VIX OK                    → BULL_PUT_SPREAD   (slight bullish bias)
─────────────────────────────────────────────────────────────

Spread pricing uses Black-Scholes with VIX-implied vol × iv_premium.
All strategies support optional take-profit and weekly or 0DTE modes.
"""
import math
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.backtest.wheel_engine import bs_put, bs_call

logger = logging.getLogger(__name__)

_RF     = 0.045
_SHARES = 100

STRATEGY_CASH            = "CASH"
STRATEGY_BULL_PUT        = "BULL_PUT_SPREAD"
STRATEGY_BEAR_CALL       = "BEAR_CALL_SPREAD"
STRATEGY_IRON_CONDOR     = "IRON_CONDOR"
STRATEGY_BULL_CALL_DEBIT = "BULL_CALL_DEBIT"


# ── data classes ─────────────────────────────────────────────────────────────

@dataclass
class SpreadLeg:
    """One side of a spread (put or call)."""
    kind:       str    # "put" | "call"
    is_short:   bool   # True = sold, False = bought
    strike:     float
    premium:    float  # per share, signed: + received, - paid


@dataclass
class OptionTrade:
    strategy:         str
    entry_date:       pd.Timestamp
    expiry_date:      pd.Timestamp
    close_date:       pd.Timestamp
    spot_entry:       float
    spot_close:       float
    legs:             list          # List[SpreadLeg]
    net_credit:       float         # >0 credit received, <0 debit paid
    spread_width:     float
    contracts:        int
    sigma:            float
    vix_at_entry:     float
    max_risk:         float         # max loss per share (always positive)
    pnl:              float
    outcome:          str           # FULL_PROFIT | TAKE_PROFIT | PARTIAL_LOSS | MAX_LOSS | CASH


@dataclass
class MultiStrategyResult:
    equity_curve:     pd.Series
    trades:           pd.DataFrame
    strategy_counts:  dict          # strategy → count
    initial_capital:  float
    final_value:      float
    benchmark_curve:  pd.Series


# ── engine ───────────────────────────────────────────────────────────────────

class MultiStrategyEngine:
    def __init__(
        self,
        initial_capital:    float = 10_000.0,
        short_otm_pct:      float = 0.02,     # OTM% for short leg of credit spreads
        spread_width_pct:   float = 0.01,     # width of each spread as % of spot
        call_otm_pct:       float = 0.02,     # OTM% for bear-call short leg
        debit_long_otm_pct: float = 0.01,     # debit call: long strike (slightly OTM)
        debit_short_otm_pct:float = 0.05,     # debit call: short strike (cap upside)
        max_risk_pct:       float = 0.10,     # max capital at risk per trade
        iv_premium:         float = 1.20,     # VIX × this = estimated IV
        max_vix_entry:      float = 25.0,     # hard skip above this VIX
        low_vol_threshold:  float = 18.0,     # below this VIX, prefer iron condor
        vol_crush_threshold:float = 30.0,     # prev-week VIX above this triggers debit play
        vol_window:         int   = 20,
        dte:                int   = 7,
        take_profit_pct:    Optional[float] = 0.50,
    ):
        self.initial_capital     = initial_capital
        self.short_otm_pct       = short_otm_pct
        self.spread_width_pct    = spread_width_pct
        self.call_otm_pct        = call_otm_pct
        self.debit_long_otm_pct  = debit_long_otm_pct
        self.debit_short_otm_pct = debit_short_otm_pct
        self.max_risk_pct        = max_risk_pct
        self.iv_premium          = iv_premium
        self.max_vix_entry       = max_vix_entry
        self.low_vol_threshold   = low_vol_threshold
        self.vol_crush_threshold = vol_crush_threshold
        self.vol_window          = vol_window
        self.dte                 = dte
        self.take_profit_pct     = take_profit_pct

    # ── helpers ───────────────────────────────────────────────────────────────

    def _px(self, df, date, col="close"):
        idx = df.index.get_indexer([date], method="ffill")
        if idx[0] == -1:
            return None
        c = col if col in df.columns else "close"
        return float(df[c].iloc[idx[0]])

    def _rv(self, df, date):
        sub = df[df.index <= date].tail(self.vol_window + 1)
        if len(sub) < 5:
            return 0.20
        return float(sub["close"].pct_change().dropna().std() * math.sqrt(252))

    def _sigma(self, vix, df, date):
        return max(vix / 100.0 * self.iv_premium, self._rv(df, date))

    def _tick(self, spot):
        return 0.50  # $0.50 strike increments (standard for XSP-like products)

    def _rs(self, price, spot):
        t = self._tick(spot)
        return round(price / t) * t

    def _width(self, spot):
        return max(self._rs(spot * self.spread_width_pct, spot), 1.0)

    # ── strategy selector ─────────────────────────────────────────────────────

    def _select(self, vix, is_bull, is_bear, spy_trend, prev_vix) -> str:
        # Hard stops
        if vix > 30:
            return STRATEGY_CASH
        if is_bear and vix > self.max_vix_entry:
            return STRATEGY_CASH
        if is_bull and vix > self.max_vix_entry:
            return STRATEGY_CASH

        # Vol-crush recovery: VIX was spiking last week, now normalizing
        if prev_vix is not None and prev_vix > self.vol_crush_threshold and vix < 25:
            return STRATEGY_BULL_CALL_DEBIT

        # Bearish regime: sell OTM calls
        if is_bear:
            return STRATEGY_BEAR_CALL

        # Bullish: sell OTM puts (standard high-probability trade)
        if is_bull:
            return STRATEGY_BULL_PUT

        # Neutral: iron condor if vol is low, else slight bull bias
        if vix < self.low_vol_threshold:
            return STRATEGY_IRON_CONDOR
        return STRATEGY_BULL_PUT

    # ── spread pricers ────────────────────────────────────────────────────────

    def _bull_put(self, spot, sigma, T):
        """Returns (K_short, K_long, credit_per_share, max_risk_per_share)."""
        K1 = self._rs(spot * (1 - self.short_otm_pct), spot)
        w  = self._width(spot)
        K2 = K1 - w
        cr = max(bs_put(spot, K1, T, sigma) - bs_put(spot, K2, T, sigma), 0.005)
        return K1, K2, w, cr, w - cr

    def _bear_call(self, spot, sigma, T):
        """Returns (K_short, K_long, credit_per_share, max_risk_per_share)."""
        K3 = self._rs(spot * (1 + self.call_otm_pct), spot)
        w  = self._width(spot)
        K4 = K3 + w
        cr = max(bs_call(spot, K3, T, sigma) - bs_call(spot, K4, T, sigma), 0.005)
        return K3, K4, w, cr, w - cr

    def _iron_condor(self, spot, sigma, T):
        """Returns put legs, call legs, total credit, max_risk."""
        K1, K2, w, put_cr, _ = self._bull_put(spot, sigma, T)
        K3, K4, _, call_cr, _ = self._bear_call(spot, sigma, T)
        total_cr  = put_cr + call_cr
        max_risk  = w - min(put_cr, call_cr)   # can only lose one side at once
        return K1, K2, K3, K4, w, put_cr, call_cr, total_cr, max_risk

    def _bull_call_debit(self, spot, sigma, T):
        """Returns (K_long, K_short, debit_per_share, max_profit_per_share)."""
        K1 = self._rs(spot * (1 + self.debit_long_otm_pct), spot)
        K2 = self._rs(spot * (1 + self.debit_short_otm_pct), spot)
        if K2 <= K1:
            K2 = K1 + self._width(spot)
        w      = K2 - K1
        debit  = max(bs_call(spot, K1, T, sigma) - bs_call(spot, K2, T, sigma), 0.005)
        return K1, K2, w, debit, w - debit   # max_profit = width - debit

    # ── P&L calculators ───────────────────────────────────────────────────────

    def _pnl_bull_put(self, spot_exp, K1, K2, credit, contracts):
        pnl_ps = credit - max(0.0, K1 - spot_exp) + max(0.0, K2 - spot_exp)
        return pnl_ps * _SHARES * contracts

    def _pnl_bear_call(self, spot_exp, K3, K4, credit, contracts):
        pnl_ps = credit - max(0.0, spot_exp - K3) + max(0.0, spot_exp - K4)
        return pnl_ps * _SHARES * contracts

    def _pnl_iron_condor(self, spot_exp, K1, K2, K3, K4, put_cr, call_cr, contracts):
        put_pnl  = put_cr  - max(0.0, K1 - spot_exp) + max(0.0, K2 - spot_exp)
        call_pnl = call_cr - max(0.0, spot_exp - K3) + max(0.0, spot_exp - K4)
        return (put_pnl + call_pnl) * _SHARES * contracts

    def _pnl_bull_call_debit(self, spot_exp, K1, K2, debit, contracts):
        pnl_ps = max(0.0, spot_exp - K1) - max(0.0, spot_exp - K2) - debit
        return pnl_ps * _SHARES * contracts

    # ── current spread value (for take-profit check) ──────────────────────────

    def _current_value(self, strategy, spot, T_rem, sigma, K1, K2,
                       K3=None, K4=None, net_cr=None):
        """Current spread value in $-per-share (cost to close a credit, gain to close a debit)."""
        if strategy == STRATEGY_BULL_PUT:
            return bs_put(spot, K1, T_rem, sigma) - bs_put(spot, K2, T_rem, sigma)
        if strategy == STRATEGY_BEAR_CALL:
            return bs_call(spot, K1, T_rem, sigma) - bs_call(spot, K2, T_rem, sigma)
        if strategy == STRATEGY_IRON_CONDOR:
            put_v  = bs_put(spot, K1, T_rem, sigma)  - bs_put(spot, K2, T_rem, sigma)
            call_v = bs_call(spot, K3, T_rem, sigma) - bs_call(spot, K4, T_rem, sigma)
            return put_v + call_v
        if strategy == STRATEGY_BULL_CALL_DEBIT:
            return bs_call(spot, K1, T_rem, sigma) - bs_call(spot, K2, T_rem, sigma)
        return 0.0

    # ── main simulation ───────────────────────────────────────────────────────

    def run(self, spy_prices: pd.DataFrame, macro_history: pd.DataFrame) -> MultiStrategyResult:
        macro  = macro_history.reindex(spy_prices.index, method="ffill")
        dates  = spy_prices.index.sort_values()

        account      = self.initial_capital
        equity: dict = {}
        trades: list = []
        open_pos     = None      # OptionTrade (pnl/outcome still TBD)
        prev_vix     = None
        strategy_counts = {s: 0 for s in [
            STRATEGY_CASH, STRATEGY_BULL_PUT, STRATEGY_BEAR_CALL,
            STRATEGY_IRON_CONDOR, STRATEGY_BULL_CALL_DEBIT,
        ]}

        for date in dates:
            spot = self._px(spy_prices, date)
            if spot is None:
                equity[date] = account
                continue

            row      = macro.loc[date] if date in macro.index else None
            vix      = float(row["vix_level"])    if row is not None and "vix_level"    in row.index else 20.0
            is_bull  = bool(row["regime_bull"])   if row is not None and "regime_bull"  in row.index else False
            is_bear  = bool(row["regime_bear"])   if row is not None and "regime_bear"  in row.index else False
            spy_tr   = int(row["spy_trend_enc"])  if row is not None and "spy_trend_enc" in row.index else 0

            # ── check / close open position ───────────────────────────────────
            if open_pos is not None:
                legs = open_pos.legs
                nc   = open_pos.net_credit
                K1   = legs[0].strike
                K2   = legs[1].strike
                K3   = legs[2].strike if len(legs) > 2 else None
                K4   = legs[3].strike if len(legs) > 3 else None
                expired = date >= open_pos.expiry_date

                # Take-profit: check residual value vs initial credit/debit
                tp_hit = False
                if not expired and self.take_profit_pct is not None and self.dte > 0:
                    days_elapsed = (date - open_pos.entry_date).days
                    if days_elapsed >= 1:
                        T_rem = max((open_pos.expiry_date - date).days / 365, 0)
                        cur_val = self._current_value(
                            open_pos.strategy, spot, T_rem, open_pos.sigma,
                            K1, K2, K3, K4,
                        )
                        if open_pos.net_credit > 0:
                            # credit trade: TP when residual ≤ (1-tp_pct) × credit
                            if cur_val <= nc * (1 - self.take_profit_pct):
                                tp_hit = True
                        else:
                            # debit trade: TP when current value ≥ debit + tp_pct × max_profit
                            max_profit = open_pos.spread_width - abs(nc)
                            target = abs(nc) + self.take_profit_pct * max_profit
                            if cur_val >= target:
                                tp_hit = True

                if expired or tp_hit:
                    s = open_pos.strategy
                    c = open_pos.contracts
                    cr = open_pos.net_credit

                    if s == STRATEGY_BULL_PUT:
                        pnl = self._pnl_bull_put(spot, K1, K2, cr, c)
                    elif s == STRATEGY_BEAR_CALL:
                        pnl = self._pnl_bear_call(spot, K1, K2, cr, c)
                    elif s == STRATEGY_IRON_CONDOR:
                        pnl = self._pnl_iron_condor(spot, K1, K2, K3, K4,
                                                     legs[0].premium, legs[2].premium, c)
                    elif s == STRATEGY_BULL_CALL_DEBIT:
                        pnl = self._pnl_bull_call_debit(spot, K1, K2, abs(cr), c)
                    else:
                        pnl = 0.0

                    if tp_hit:
                        # On TP, we close at residual value, not at settlement
                        T_rem = max((open_pos.expiry_date - date).days / 365, 0)
                        cur_val = self._current_value(open_pos.strategy, spot, T_rem,
                                                      open_pos.sigma, K1, K2, K3, K4)
                        if nc > 0:
                            pnl = (nc - cur_val) * _SHARES * c
                        else:
                            pnl = (cur_val - abs(nc)) * _SHARES * c

                    account += pnl

                    if tp_hit:
                        outcome = "TAKE_PROFIT"
                    elif s in (STRATEGY_BULL_PUT, STRATEGY_BEAR_CALL, STRATEGY_IRON_CONDOR):
                        if pnl >= nc * _SHARES * c * 0.95:
                            outcome = "FULL_PROFIT"
                        elif pnl <= -open_pos.max_risk * _SHARES * c * 0.95:
                            outcome = "MAX_LOSS"
                        else:
                            outcome = "PARTIAL_LOSS"
                    else:  # debit
                        outcome = "FULL_PROFIT" if pnl > 0 else "MAX_LOSS"

                    open_pos.close_date  = date
                    open_pos.spot_close  = spot
                    open_pos.pnl         = pnl
                    open_pos.outcome     = outcome
                    trades.append(open_pos)
                    open_pos = None

            # ── open new position ─────────────────────────────────────────────
            if open_pos is None:
                is_entry_day = (self.dte == 0) or (date.weekday() == 4)
                if not is_entry_day:
                    equity[date] = account
                    prev_vix = vix
                    continue

                strategy = self._select(vix, is_bull, is_bear, spy_tr, prev_vix)
                strategy_counts[strategy] += 1

                if strategy == STRATEGY_CASH:
                    equity[date] = account
                    prev_vix = vix
                    continue

                sigma = self._sigma(vix, spy_prices, date)
                entry_spot = self._px(spy_prices, date, col="open") or spot if self.dte == 0 else spot
                T = max(self.dte, 0) / 365 if self.dte > 0 else (1 / 252)

                legs = []
                net_credit = 0.0
                max_risk   = 0.0
                spread_width = 0.0
                K3 = K4 = None

                if strategy == STRATEGY_BULL_PUT:
                    K1, K2, w, cr, mr = self._bull_put(entry_spot, sigma, T)
                    legs = [SpreadLeg("put", True, K1, cr), SpreadLeg("put", False, K2, -cr)]
                    net_credit, max_risk, spread_width = cr, mr, w

                elif strategy == STRATEGY_BEAR_CALL:
                    K1, K2, w, cr, mr = self._bear_call(entry_spot, sigma, T)
                    legs = [SpreadLeg("call", True, K1, cr), SpreadLeg("call", False, K2, -cr)]
                    net_credit, max_risk, spread_width = cr, mr, w

                elif strategy == STRATEGY_IRON_CONDOR:
                    K1, K2, K3, K4, w, put_cr, call_cr, total_cr, mr = \
                        self._iron_condor(entry_spot, sigma, T)
                    legs = [
                        SpreadLeg("put",  True,  K1, put_cr),
                        SpreadLeg("put",  False, K2, -put_cr),
                        SpreadLeg("call", True,  K3, call_cr),
                        SpreadLeg("call", False, K4, -call_cr),
                    ]
                    net_credit, max_risk, spread_width = total_cr, mr, w

                elif strategy == STRATEGY_BULL_CALL_DEBIT:
                    K1, K2, w, db, mp = self._bull_call_debit(entry_spot, sigma, T)
                    legs = [SpreadLeg("call", False, K1, -db), SpreadLeg("call", True, K2, db)]
                    net_credit, max_risk, spread_width = -db, db, w

                if max_risk <= 0:
                    equity[date] = account
                    prev_vix = vix
                    continue

                contracts = max(int(account * self.max_risk_pct // (max_risk * _SHARES)), 1)

                # Find expiry
                if self.dte == 0:
                    expiry = date
                else:
                    future  = dates[dates > date]
                    fridays = future[future.day_of_week == 4]
                    if len(fridays) == 0:
                        expiry = date + pd.Timedelta(days=7)
                    else:
                        wf = fridays[fridays <= date + pd.Timedelta(days=10)]
                        expiry = wf[-1] if len(wf) > 0 else fridays[0]

                open_pos = OptionTrade(
                    strategy=strategy,
                    entry_date=date, expiry_date=expiry, close_date=date,
                    spot_entry=entry_spot, spot_close=0.0,
                    legs=legs,
                    net_credit=net_credit, spread_width=spread_width,
                    contracts=contracts, sigma=sigma, vix_at_entry=vix,
                    max_risk=max_risk, pnl=0.0, outcome="",
                )

                # 0DTE: settle immediately at day's close
                if self.dte == 0:
                    s = strategy
                    cr = net_credit
                    c  = contracts
                    if s == STRATEGY_BULL_PUT:
                        pnl = self._pnl_bull_put(spot, K1, K2, cr, c)
                    elif s == STRATEGY_BEAR_CALL:
                        pnl = self._pnl_bear_call(spot, K1, K2, cr, c)
                    elif s == STRATEGY_IRON_CONDOR:
                        pnl = self._pnl_iron_condor(spot, K1, K2, K3, K4,
                                                     legs[0].premium, legs[2].premium, c)
                    elif s == STRATEGY_BULL_CALL_DEBIT:
                        pnl = self._pnl_bull_call_debit(spot, K1, K2, abs(cr), c)
                    else:
                        pnl = 0.0

                    account += pnl
                    open_pos.close_date = date
                    open_pos.spot_close = spot
                    open_pos.pnl        = pnl
                    open_pos.outcome    = "FULL_PROFIT" if pnl > 0 else ("MAX_LOSS" if pnl < -max_risk * _SHARES * c * 0.95 else "PARTIAL_LOSS")
                    trades.append(open_pos)
                    open_pos = None

                logger.debug("%s | %s | SPY=%.1f VIX=%.1f sigma=%.0f%%",
                             date.date(), strategy, entry_spot, vix, sigma * 100)

            equity[date] = account
            prev_vix = vix

        eq = pd.Series(equity).sort_index()
        bm_start = self._px(spy_prices, eq.index[0]) or 1.0
        benchmark = pd.Series({
            d: (self._px(spy_prices, d) or bm_start) / bm_start * self.initial_capital
            for d in eq.index
        })

        trades_df = pd.DataFrame([
            {k: v for k, v in vars(t).items() if k != "legs"}
            for t in trades if t.outcome not in ("", "CASH")
        ])

        return MultiStrategyResult(
            equity_curve=eq,
            trades=trades_df,
            strategy_counts=strategy_counts,
            initial_capital=self.initial_capital,
            final_value=float(eq.iloc[-1]) if not eq.empty else self.initial_capital,
            benchmark_curve=benchmark,
        )
