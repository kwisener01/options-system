"""
Wheel strategy backtest engine.

Simulates the options wheel on any equity/ETF:
  1. Sell cash-secured puts (CSP) at put_otm_pct below spot.
  2. On assignment: own shares, immediately sell a covered call (CC).
  3. On call-away: shares sold, restart the put-selling cycle.

Premium is priced with Black-Scholes using 20-day realized vol scaled by
iv_premium to approximate the volatility risk premium sellers capture.

Capital accounting:
  free_cash   - cash not committed to any obligation
  reserved    - cash locked as CSP collateral (released on expiry or assignment)
  holdings    - mark-to-market value of assigned shares
  account     = free_cash + reserved + holdings_value
"""
import math
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_RF = 0.045        # annualized risk-free rate
_SHARES = 100      # shares per contract


# ── Black-Scholes helpers ────────────────────────────────────────────────────

def _ncdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2))


def bs_put(S: float, K: float, T: float, sigma: float) -> float:
    """European put price via Black-Scholes. Returns intrinsic value when T≈0."""
    if T < 1e-6 or sigma < 1e-6:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (_RF + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return max(K * math.exp(-_RF * T) * _ncdf(-d2) - S * _ncdf(-d1), 0.0)


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    """European call price via Black-Scholes. Returns intrinsic value when T≈0."""
    if T < 1e-6 or sigma < 1e-6:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (_RF + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return max(S * _ncdf(d1) - K * math.exp(-_RF * T) * _ncdf(d2), 0.0)


def _round_strike(price: float) -> float:
    return round(price)  # $1 increments (standard for most US equities/ETFs)


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class WheelTrade:
    date: pd.Timestamp
    symbol: str
    action: str           # SELL_PUT | EXPIRED_PUT | ASSIGNED | SELL_CALL | EXPIRED_CALL | CALLED_AWAY
    stock_price: float
    strike: float
    iv: float
    premium_per_share: float
    contracts: int
    realized_pnl: float = 0.0


@dataclass
class _Pos:
    symbol: str
    phase: str            # "CSP" | "CC"
    open_date: pd.Timestamp
    expiry_date: pd.Timestamp
    strike: float
    premium: float        # total premium received (all shares)
    contracts: int
    shares: int
    cost_basis: float     # per share (for CC: assignment_price - put_premium_per_share)
    reserved: float       # cash locked as CSP collateral


@dataclass
class WheelResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    initial_capital: float
    final_value: float
    total_premium: float
    n_assignments: int
    n_put_expirations: int
    n_call_expirations: int
    n_call_aways: int
    benchmark_curve: pd.Series


# ── Engine ────────────────────────────────────────────────────────────────────

class WheelEngine:
    def __init__(
        self,
        initial_capital: float = 50_000.0,
        tickers: List[str] = None,
        put_otm_pct: float = 0.05,
        call_otm_pct: float = 0.03,
        iv_premium: float = 1.15,
        vol_window: int = 20,
    ):
        self.initial_capital = initial_capital
        self.tickers = tickers or ["SPY"]
        self.put_otm_pct = put_otm_pct
        self.call_otm_pct = call_otm_pct
        self.iv_premium = iv_premium
        self.vol_window = vol_window

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _price_at(self, df: pd.DataFrame, date: pd.Timestamp) -> Optional[float]:
        idx = df.index.get_indexer([date], method="ffill")
        if idx[0] == -1:
            return None
        return float(df["close"].iloc[idx[0]])

    def _vol_at(self, df: pd.DataFrame, date: pd.Timestamp) -> float:
        subset = df[df.index <= date].tail(self.vol_window + 1)
        if len(subset) < 5:
            return 0.25
        rv = float(subset["close"].pct_change().dropna().std() * math.sqrt(252))
        return max(rv * self.iv_premium, 0.05)

    def _open_cc(
        self,
        sym: str,
        df: pd.DataFrame,
        open_date: pd.Timestamp,
        shares: int,
        contracts: int,
        cost_basis: float,
        trades: list,
    ) -> _Pos:
        """Price and record a new covered-call position."""
        px = self._price_at(df, open_date)
        iv = self._vol_at(df, open_date)
        T = 30 / 365
        cc_strike = _round_strike(max(px * (1 + self.call_otm_pct), cost_basis + 0.50))
        prem_ps = bs_call(px, cc_strike, T, iv)
        prem = prem_ps * shares
        trades.append(WheelTrade(
            date=open_date, symbol=sym, action="SELL_CALL",
            stock_price=px, strike=cc_strike, iv=iv,
            premium_per_share=prem_ps, contracts=contracts,
        ))
        return _Pos(
            symbol=sym, phase="CC",
            open_date=open_date,
            expiry_date=open_date + pd.DateOffset(days=30),
            strike=cc_strike, premium=prem,
            contracts=contracts, shares=shares,
            cost_basis=cost_basis, reserved=0.0,
        )

    # ── Main simulation ───────────────────────────────────────────────────────

    def run(self, prices: Dict[str, pd.DataFrame]) -> WheelResult:
        available = [t for t in self.tickers if t in prices]
        if not available:
            raise ValueError(f"No price data found for: {self.tickers}")

        ref_df = prices[available[0]]
        all_dates = ref_df.index.sort_values()

        # Monthly periods
        s = pd.Series(all_dates)
        periods = s.groupby(s.dt.to_period("M")).agg(["first", "last"])

        free_cash = self.initial_capital
        reserved_cash = 0.0
        positions: Dict[str, _Pos] = {}
        holdings: Dict[str, tuple] = {}   # sym -> (shares, cost_basis)
        equity_curve: Dict[pd.Timestamp, float] = {}
        trades: List[WheelTrade] = []

        total_premium = 0.0
        n_assignments = n_put_exp = n_call_exp = n_call_aways = 0

        target_per = self.initial_capital / len(available)

        for _, row in periods.iterrows():
            month_start: pd.Timestamp = row["first"]
            month_end: pd.Timestamp = row["last"]

            # ── Expire positions that matured this month ──────────────────────
            for sym in list(positions.keys()):
                pos = positions[sym]
                if pos.expiry_date > month_end:
                    continue

                df = prices[sym]
                exp_px = self._price_at(df, pos.expiry_date)
                if exp_px is None:
                    continue

                if pos.phase == "CSP":
                    if exp_px < pos.strike:
                        # Assigned: convert reserved collateral into stock position
                        reserved_cash -= pos.reserved
                        cb = pos.strike - (pos.premium / pos.shares)
                        holdings[sym] = (pos.shares, cb)
                        n_assignments += 1
                        total_premium += pos.premium
                        trades.append(WheelTrade(
                            date=pos.expiry_date, symbol=sym, action="ASSIGNED",
                            stock_price=exp_px, strike=pos.strike, iv=0.0,
                            premium_per_share=pos.premium / pos.shares,
                            contracts=pos.contracts,
                        ))
                        del positions[sym]

                        # Immediately sell a covered call on the assigned shares
                        cc = self._open_cc(sym, df, pos.expiry_date, pos.shares, pos.contracts, cb, trades)
                        free_cash += cc.premium
                        total_premium += cc.premium
                        positions[sym] = cc

                    else:
                        # Put expired worthless: release collateral, keep premium
                        reserved_cash -= pos.reserved
                        free_cash += pos.reserved
                        total_premium += pos.premium
                        n_put_exp += 1
                        trades.append(WheelTrade(
                            date=pos.expiry_date, symbol=sym, action="EXPIRED_PUT",
                            stock_price=exp_px, strike=pos.strike, iv=0.0,
                            premium_per_share=pos.premium / pos.shares,
                            contracts=pos.contracts, realized_pnl=pos.premium,
                        ))
                        del positions[sym]

                elif pos.phase == "CC":
                    shares, cb = holdings[sym]
                    if exp_px > pos.strike:
                        # Called away: receive strike for shares
                        free_cash += pos.strike * shares
                        total_premium += pos.premium
                        pnl = (pos.strike - cb) * shares + pos.premium
                        n_call_aways += 1
                        trades.append(WheelTrade(
                            date=pos.expiry_date, symbol=sym, action="CALLED_AWAY",
                            stock_price=exp_px, strike=pos.strike, iv=0.0,
                            premium_per_share=pos.premium / pos.shares,
                            contracts=pos.contracts, realized_pnl=pnl,
                        ))
                        del holdings[sym]
                        del positions[sym]

                    else:
                        # CC expired worthless: keep shares, sell next CC
                        total_premium += pos.premium
                        n_call_exp += 1
                        trades.append(WheelTrade(
                            date=pos.expiry_date, symbol=sym, action="EXPIRED_CALL",
                            stock_price=exp_px, strike=pos.strike, iv=0.0,
                            premium_per_share=pos.premium / pos.shares,
                            contracts=pos.contracts, realized_pnl=pos.premium,
                        ))
                        del positions[sym]
                        cc = self._open_cc(sym, prices[sym], pos.expiry_date, shares, pos.contracts, cb, trades)
                        free_cash += cc.premium
                        total_premium += cc.premium
                        positions[sym] = cc

            # ── Open CSPs for idle tickers ────────────────────────────────────
            for sym in available:
                if sym in positions:
                    continue

                df = prices[sym]
                px = self._price_at(df, month_start)
                if px is None or px <= 0:
                    continue

                iv = self._vol_at(df, month_start)
                T = 30 / 365
                put_strike = _round_strike(px * (1 - self.put_otm_pct))
                if put_strike <= 0:
                    continue

                collateral_per = put_strike * _SHARES
                # Limit to target allocation but always allow at least what free cash supports
                max_by_target = max(int(target_per // collateral_per), 0)
                max_by_cash = max(int(free_cash // collateral_per), 0)
                # Use the lesser of the two, but ensure at least 1 if cash allows
                contracts = min(max(max_by_target, 1 if max_by_cash >= 1 else 0), max_by_cash)
                if contracts <= 0:
                    logger.warning(
                        "%s: can't afford 1 contract (strike=%.2f, need=$%.0f, free_cash=%.0f)",
                        sym, put_strike, collateral_per, free_cash,
                    )
                    continue

                shares = contracts * _SHARES
                prem_ps = bs_put(px, put_strike, T, iv)
                prem = prem_ps * shares
                collateral = put_strike * shares

                free_cash += prem
                free_cash -= collateral
                reserved_cash += collateral

                positions[sym] = _Pos(
                    symbol=sym, phase="CSP",
                    open_date=month_start,
                    expiry_date=month_start + pd.DateOffset(days=30),
                    strike=put_strike, premium=prem,
                    contracts=contracts, shares=shares,
                    cost_basis=put_strike, reserved=collateral,
                )
                trades.append(WheelTrade(
                    date=month_start, symbol=sym, action="SELL_PUT",
                    stock_price=px, strike=put_strike, iv=iv,
                    premium_per_share=prem_ps, contracts=contracts,
                ))

            # ── Mark to market ────────────────────────────────────────────────
            hold_val = sum(
                (self._price_at(prices[sym], month_end) or 0.0) * sh
                for sym, (sh, _) in holdings.items()
                if sym in prices
            )
            equity_curve[month_end] = free_cash + reserved_cash + hold_val

            logger.debug(
                "%s | free=%.0f rsv=%.0f hold=%.0f total=%.0f",
                month_end.date(), free_cash, reserved_cash, hold_val,
                free_cash + reserved_cash + hold_val,
            )

        eq = pd.Series(equity_curve).sort_index()

        # Benchmark: buy-and-hold the first available ticker (usually SPY proxy)
        bm_sym = "SPY" if "SPY" in prices else available[0]
        bm_df = prices[bm_sym]
        bm_start = self._price_at(bm_df, eq.index[0]) if not eq.empty else 1.0
        benchmark = pd.Series(
            {d: (self._price_at(bm_df, d) or bm_start) / bm_start * self.initial_capital
             for d in eq.index}
        )

        trades_df = pd.DataFrame([vars(t) for t in trades]) if trades else pd.DataFrame()

        return WheelResult(
            equity_curve=eq,
            trades=trades_df,
            initial_capital=self.initial_capital,
            final_value=float(eq.iloc[-1]) if not eq.empty else self.initial_capital,
            total_premium=total_premium,
            n_assignments=n_assignments,
            n_put_expirations=n_put_exp,
            n_call_expirations=n_call_exp,
            n_call_aways=n_call_aways,
            benchmark_curve=benchmark,
        )
