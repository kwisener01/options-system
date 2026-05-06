import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.metrics import roc_auc_score

from config.settings import (
    BACKTEST_INITIAL_CAPITAL,
    BACKTEST_SLIPPAGE_PCT,
    ML_TOP_N,
    ML_MIN_TRAIN_WEEKS,
)
from src.signals.feature_engineer import FEATURE_COLS
from src.signals.ml_selector import MLSelector

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades_log: pd.DataFrame
    weekly_returns: pd.Series
    benchmark_returns: pd.Series
    initial_capital: float
    final_value: float
    oos_auc_scores: list = None
    feature_importances: pd.Series = None


class BacktestEngine:
    def __init__(
        self,
        initial_capital: float = BACKTEST_INITIAL_CAPITAL,
        slippage_pct: float = BACKTEST_SLIPPAGE_PCT,
        top_n: int = ML_TOP_N,
        max_atr_pct: float = 0.0,
        vix_cash_buffer: float = 0.0,
    ):
        self.initial_capital = initial_capital
        self.slippage_pct = slippage_pct
        self.top_n = top_n
        self.max_atr_pct = max_atr_pct      # skip stocks with daily ATR% above this
        self.vix_cash_buffer = vix_cash_buffer  # hold back this fraction of cash when VIX > 20

    def _exec_price(self, price: float, side: str) -> float:
        mult = 1 + self.slippage_pct if side == "buy" else 1 - self.slippage_pct
        return price * mult

    def _get_close(self, prices: dict, sym: str, date: pd.Timestamp) -> float | None:
        df = prices.get(sym)
        if df is None:
            return None
        idx = df.index.get_indexer([date], method="ffill")
        if idx[0] == -1:
            return None
        return float(df["close"].iloc[idx[0]])

    def run(self, labeled: pd.DataFrame, prices: dict, selector: MLSelector) -> BacktestResult:
        """
        Walk-forward backtest: retrain model up to each week, score, rebalance.
        Uses the same expanding-window logic as MLSelector.train() but applies it
        week-by-week to simulate live execution.
        """
        feat_cols = [c for c in FEATURE_COLS if c in labeled.columns]

        # Weekly periods (Friday dates)
        all_dates = sorted(labeled["date"].unique())
        weeks = pd.Series(all_dates).dt.to_period("W").drop_duplicates().sort_values()
        week_end_dates = [w.end_time.normalize() for w in weeks]

        min_train = ML_MIN_TRAIN_WEEKS
        equity = {}
        trades = []
        portfolio_value = self.initial_capital
        cash = self.initial_capital
        holdings: dict = {}  # {symbol: {"qty": float, "cost": float}}
        oos_auc_scores = []
        last_model = None

        # SPY for benchmark
        spy_prices = prices.get("SPY") if prices.get("SPY") is not None else prices.get(list(prices.keys())[0])

        for wi, week_end in enumerate(week_end_dates):
            if wi < min_train:
                equity[week_end] = portfolio_value
                continue

            train_cutoff = week_end_dates[wi - 1]

            # Retrain on all data up to last week
            train = labeled[labeled["date"] <= train_cutoff].dropna(subset=["label"])
            if train.empty:
                equity[week_end] = portfolio_value
                continue

            n_neg = int((train["label"] == 0).sum())
            n_pos = int((train["label"] == 1).sum())
            spw = n_neg / n_pos if n_pos > 0 else 1.0

            from src.signals.ml_selector import _make_model  # returns LGBMClassifier
            model = _make_model(scale_pos_weight=spw)
            model.fit(train[feat_cols].fillna(0), train["label"])
            last_model = model

            # OOS AUC: score next week's data with this fold's model
            next_week_end = week_end_dates[wi + 1] if wi + 1 < len(week_end_dates) else None
            if next_week_end is not None:
                oos_data = labeled[
                    (labeled["date"] > train_cutoff) & (labeled["date"] <= next_week_end)
                ].dropna(subset=["label"])
                if not oos_data.empty and oos_data["label"].nunique() > 1:
                    oos_proba = model.predict_proba(oos_data[feat_cols].fillna(0))[:, 1]
                    try:
                        oos_auc_scores.append(roc_auc_score(oos_data["label"], oos_proba))
                    except Exception:
                        pass

            # Regime / VIX-spike gate: liquidate all and skip if bear or VIX > 30
            # week_end_dates are Sundays; find the nearest prior trading day in labeled.
            regime_row = labeled[labeled["date"] == train_cutoff]
            if regime_row.empty:
                _avail = labeled[labeled["date"] <= train_cutoff]
                if not _avail.empty:
                    regime_row = labeled[labeled["date"] == _avail["date"].max()].head(1)

            is_bear = False
            if not regime_row.empty:
                if "regime_bear" in regime_row.columns and int(regime_row["regime_bear"].iloc[0]) == 1:
                    is_bear = True
                if "vix_level" in regime_row.columns and float(regime_row["vix_level"].iloc[0]) > 30:
                    is_bear = True

            if is_bear:
                for sym in list(holdings.keys()):
                    exit_price_raw = self._get_close(prices, sym, week_end)
                    if exit_price_raw is None:
                        continue
                    exit_price = self._exec_price(exit_price_raw, "sell")
                    qty = holdings[sym]["qty"]
                    proceeds = qty * exit_price
                    cash += proceeds
                    trades.append({"date": week_end, "symbol": sym, "action": "SELL",
                                   "qty": qty, "price": exit_price, "value": proceeds})
                    del holdings[sym]
                equity[week_end] = cash
                vix_val = float(regime_row["vix_level"].iloc[0]) if not regime_row.empty and "vix_level" in regime_row.columns else 0
                logger.info("Week %d/%d | RISK-OFF (VIX=%.1f) — all positions closed, cash=%.2f",
                            wi, len(week_end_dates), vix_val, cash)
                continue

            # Score this week's latest features (no label required)
            week_data = labeled[labeled["date"] == train_cutoff]
            if week_data.empty:
                # Use closest available date
                available = labeled[labeled["date"] <= train_cutoff]
                if available.empty:
                    equity[week_end] = portfolio_value
                    continue
                latest_date = available["date"].max()
                week_data = available[available["date"] == latest_date]

            week_data = week_data.dropna(subset=feat_cols[:5])
            if week_data.empty:
                equity[week_end] = portfolio_value
                continue

            X = week_data.set_index("symbol")[feat_cols].fillna(0)
            proba = model.predict_proba(X)[:, 1]
            scores = pd.Series(proba, index=X.index).sort_values(ascending=False)

            # ATR filter: drop high-volatility candidates before picking top_n
            if self.max_atr_pct > 0 and "atr_pct" in X.columns:
                low_vol = X["atr_pct"] <= self.max_atr_pct
                scores = scores[scores.index.isin(X.index[low_vol])]

            selected = scores.head(self.top_n).index.tolist()

            # Close positions not in new selection
            exited = [s for s in list(holdings.keys()) if s not in selected]
            for sym in exited:
                exit_price_raw = self._get_close(prices, sym, week_end)
                if exit_price_raw is None:
                    continue
                exit_price = self._exec_price(exit_price_raw, "sell")
                qty = holdings[sym]["qty"]
                proceeds = qty * exit_price
                cash += proceeds
                trades.append({
                    "date": week_end,
                    "symbol": sym,
                    "action": "SELL",
                    "qty": qty,
                    "price": exit_price,
                    "value": proceeds,
                })
                del holdings[sym]

            # VIX cash buffer: hold back capital when VIX is elevated
            deployable = cash
            if self.vix_cash_buffer > 0 and "vix_level" in week_data.columns:
                vix_now = float(week_data["vix_level"].iloc[0])
                if vix_now > 20:
                    deployable = cash * (1 - self.vix_cash_buffer)

            # Open new positions with inverse-vol sizing (calmer stocks get more capital)
            new_entries = [s for s in selected if s not in holdings]
            if new_entries:
                if "atr_pct" in X.columns:
                    vols = X.loc[X.index.isin(new_entries), "atr_pct"].reindex(new_entries).clip(lower=0.001)
                    inv_vol = 1.0 / vols
                    weights = inv_vol / inv_vol.sum()
                    alloc_amounts = (weights * deployable).to_dict()
                else:
                    per = deployable / len(new_entries)
                    alloc_amounts = {s: per for s in new_entries}
            else:
                alloc_amounts = {}

            for sym in new_entries:
                entry_price_raw = self._get_close(prices, sym, week_end)
                if entry_price_raw is None or entry_price_raw <= 0:
                    continue
                entry_price = self._exec_price(entry_price_raw, "buy")
                dollar_in = min(alloc_amounts.get(sym, 0), cash)
                if dollar_in < 1:
                    continue
                qty = dollar_in / entry_price
                cash -= dollar_in
                holdings[sym] = {"qty": qty, "cost": entry_price}
                trades.append({
                    "date": week_end,
                    "symbol": sym,
                    "action": "BUY",
                    "qty": qty,
                    "price": entry_price,
                    "value": dollar_in,
                })

            # Mark positions to market
            holdings_value = 0.0
            for sym, pos in holdings.items():
                px = self._get_close(prices, sym, week_end)
                if px:
                    holdings_value += pos["qty"] * px

            portfolio_value = cash + holdings_value
            equity[week_end] = portfolio_value

            if wi % 4 == 0:
                logger.info(
                    "Week %d/%d | Value=$%.2f | Holdings=%d | Cash=$%.2f",
                    wi, len(week_end_dates), portfolio_value, len(holdings), cash,
                )

        equity_curve = pd.Series(equity).sort_index()
        weekly_returns = equity_curve.pct_change().dropna()

        # Benchmark: SPY weekly returns over same period
        spy_weekly = {}
        for date in equity_curve.index:
            px = self._get_close(prices, "SPY", date)
            if px:
                spy_weekly[date] = px
        spy_series = pd.Series(spy_weekly).sort_index()
        benchmark_returns = spy_series.pct_change().dropna()
        benchmark_returns = benchmark_returns.reindex(weekly_returns.index).fillna(0)

        trades_df = pd.DataFrame(trades)

        feat_imp = None
        if last_model is not None:
            raw = last_model.feature_importances_.astype(float)
            total = raw.sum()
            normalized = raw / total if total > 0 else raw
            feat_imp = pd.Series(normalized, index=feat_cols).sort_values(ascending=False)

        if oos_auc_scores:
            logger.info(
                "Walk-forward OOS AUC: mean=%.3f  std=%.3f  over %d folds",
                np.mean(oos_auc_scores), np.std(oos_auc_scores), len(oos_auc_scores),
            )

        return BacktestResult(
            equity_curve=equity_curve,
            trades_log=trades_df,
            weekly_returns=weekly_returns,
            benchmark_returns=benchmark_returns,
            initial_capital=self.initial_capital,
            final_value=float(equity_curve.iloc[-1]) if not equity_curve.empty else self.initial_capital,
            oos_auc_scores=oos_auc_scores,
            feature_importances=feat_imp,
        )
