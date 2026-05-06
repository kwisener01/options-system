"""
Rotating options engine — regime-driven intraday strategy selection.

Evaluates market conditions every N minutes and decides whether to:
  - hold the current structure
  - morph it (add/remove legs without a full close)
  - close and reopen in a different strategy type
  - stand aside

Designed for XSP on a small account (~$2k). Builds on the existing
alpaca_options.py leg execution layer.

Strategy rotation map:
─────────────────────────────────────────────────────────────
  IV Rank > 50 + Neutral       →  Iron Condor / BWB (late day)
  IV Rank > 50 + Bullish       →  Jade Lizard       (no upside risk)
  IV Rank > 50 + Bearish       →  Rev. Jade Lizard  (no downside risk)
  IV Rank 25-50 + Neutral      →  Butterfly / Calendar (early)
  IV Rank 25-50 + Directional  →  Credit spread (put or call side)
  IV Rank < 25 + Neutral       →  Butterfly         (cheap, defined)
  IV Rank < 25 + Directional   →  Debit spread
  VIX > 35                     →  Cash
─────────────────────────────────────────────────────────────

Morphing preference (avoids full close+reopen, saves commissions):
  Bull Put Spread  + bearish move  →  add bear call = Iron Condor
  Bear Call Spread + bullish move  →  add bull put  = Iron Condor
  Iron Condor      + vol collapse  →  tighten to Butterfly
  Strangle                         →  sell wings    = Iron Condor

PDT note for < $25k accounts:
  Opening AND closing an option same day = 1 day trade.
  Morphing (adding a new leg without closing) does NOT count.
  Prefer morphing over close+reopen to preserve day-trade budget.
"""
import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_SHARES = 100


# ── Strategy types ────────────────────────────────────────────────────────────

class StrategyType(Enum):
    CASH            = "CASH"
    IRON_CONDOR     = "IRON_CONDOR"
    BUTTERFLY       = "BUTTERFLY"
    BWB_PUT         = "BWB_PUT"          # broken-wing butterfly, put biased
    BWB_CALL        = "BWB_CALL"
    JADE_LIZARD     = "JADE_LIZARD"      # sell put + sell call spread; no upside risk
    REV_JADE_LIZARD = "REV_JADE_LIZARD"  # sell call + sell put spread; no downside risk
    BULL_PUT        = "BULL_PUT_SPREAD"
    BEAR_CALL       = "BEAR_CALL_SPREAD"
    BULL_CALL_DEBIT = "BULL_CALL_DEBIT"
    BEAR_PUT_DEBIT  = "BEAR_PUT_DEBIT"
    CALENDAR_CALL   = "CALENDAR_CALL"    # sell near, buy far — same strike
    CALENDAR_PUT    = "CALENDAR_PUT"
    RATIO_CALL      = "RATIO_CALL"       # buy 1 ATM call, sell 2 OTM calls
    RATIO_PUT       = "RATIO_PUT"


# ── Leg ───────────────────────────────────────────────────────────────────────

@dataclass
class Leg:
    """One option contract in a multi-leg position."""
    kind:          str    # "call" | "put"
    action:        str    # "buy"  | "sell"
    strike:        float
    expiry:        date
    qty:           int
    entry_price:   float  # per share (option premium)
    occ_symbol:    str = ""
    order_id:      str = ""
    current_price: float = 0.0

    @property
    def signed_premium(self) -> float:
        """Net cash flow at entry. Positive = received, negative = paid."""
        sign = 1 if self.action == "sell" else -1
        return sign * self.entry_price * self.qty * _SHARES

    def pnl(self) -> float:
        """Unrealized P&L based on current_price."""
        price_diff = self.current_price - self.entry_price
        sign = -1 if self.action == "sell" else 1
        return sign * price_diff * self.qty * _SHARES


# ── Position ──────────────────────────────────────────────────────────────────

@dataclass
class Position:
    strategy:   StrategyType
    symbol:     str
    legs:       List[Leg]
    entry_time: datetime
    spot_entry: float
    vix_entry:  float
    notes:      str = ""

    @property
    def net_credit(self) -> float:
        """Total cash received at entry (negative if net debit)."""
        return sum(leg.signed_premium for leg in self.legs)

    @property
    def max_loss(self) -> float:
        """Approximate max loss (positive dollars). Strategy-specific."""
        return _calc_max_loss(self)

    @property
    def breakevens(self) -> List[float]:
        return _calc_breakevens(self)

    @property
    def unrealized_pnl(self) -> float:
        return sum(leg.pnl() for leg in self.legs)

    @property
    def close_cost(self) -> float:
        """Net cost to close all legs at current prices (positive = costs money)."""
        cost = 0.0
        for leg in self.legs:
            if leg.action == "sell":
                cost += leg.current_price * leg.qty * _SHARES
            else:
                cost -= leg.current_price * leg.qty * _SHARES
        return cost

    def tp_pct(self) -> float:
        """How much of max profit has been captured. 1.0 = full profit."""
        if self.net_credit <= 0:
            return 0.0
        return 1.0 - (self.close_cost / self.net_credit)

    def loss_pct(self) -> float:
        """Current loss as fraction of max loss. 1.0 = max loss hit."""
        if self.max_loss <= 0:
            return 0.0
        return max(0.0, -self.unrealized_pnl) / self.max_loss


# ── Market Regime ─────────────────────────────────────────────────────────────

@dataclass
class Regime:
    iv_rank:      float   # 0-100, percentile of VIX vs 52-week range
    vix:          float
    trend:        float   # -1.0 (strong bear) to +1.0 (strong bull), intraday
    time_of_day:  float   # 0.0 = market open, 1.0 = market close
    realized_vol: float   # HV20 as decimal (e.g. 0.15 for 15%)


# ── Strategy Selector ─────────────────────────────────────────────────────────

class StrategySelector:
    """
    Maps a Regime to the best-fit StrategyType.

    High IV → sell premium structures (condor, jade lizard, BWB).
    Low IV  → buy premium structures (butterfly, debit spread, calendar).
    Directional bias shifts from neutral to one-sided structures.
    Late in the day → prefer shorter theta-rich structures.
    """

    def select(self, regime: Regime) -> StrategyType:
        vix  = regime.vix
        iv   = regime.iv_rank
        tr   = regime.trend
        tod  = regime.time_of_day

        if vix > 35:
            return StrategyType.CASH

        # HIGH IV — sell premium
        if iv > 50:
            if abs(tr) < 0.25:                         # neutral
                if tod > 0.6:                          # late session: BWB collects extra theta
                    return StrategyType.BWB_PUT if tr >= 0 else StrategyType.BWB_CALL
                return StrategyType.IRON_CONDOR
            elif tr >= 0.25:                           # bullish bias
                return StrategyType.JADE_LIZARD        # no upside risk
            else:                                      # bearish bias
                return StrategyType.REV_JADE_LIZARD    # no downside risk

        # MEDIUM IV — mixed
        if 25 < iv <= 50:
            if abs(tr) < 0.25:
                if tod < 0.35:                         # early: calendar captures vol term structure
                    return StrategyType.CALENDAR_PUT
                return StrategyType.BUTTERFLY          # tighter, cheaper
            elif tr >= 0.25:
                return StrategyType.BULL_PUT           # sell put spread, bullish
            else:
                return StrategyType.BEAR_CALL          # sell call spread, bearish

        # LOW IV — buy premium
        if abs(tr) < 0.25:
            return StrategyType.BUTTERFLY              # cheap defined risk
        elif tr >= 0.25:
            return StrategyType.BULL_CALL_DEBIT
        else:
            return StrategyType.BEAR_PUT_DEBIT


# ── Position Morpher ──────────────────────────────────────────────────────────

@dataclass
class MorphPlan:
    """Instructions to transition between position structures."""
    legs_to_close:   List[Leg]
    legs_to_add:     List[dict]   # each: {kind, action, strike, expiry, qty}
    target_strategy: StrategyType
    day_trades_used: int          # 0 for pure additions, 1+ for closures
    estimated_net:   float        # positive = net credit for the morph


# Maps (current_strategy, target_strategy) → morph action tag.
# Only includes morphs that require no full position close (≤ 1 day trade).
_MORPH_TABLE = {
    (StrategyType.BULL_PUT,   StrategyType.IRON_CONDOR): "add_bear_call",
    (StrategyType.BEAR_CALL,  StrategyType.IRON_CONDOR): "add_bull_put",
    (StrategyType.IRON_CONDOR, StrategyType.BUTTERFLY):  "tighten_to_fly",
    (StrategyType.BULL_PUT,   StrategyType.BUTTERFLY):   "close_to_fly",  # 1 DT
    (StrategyType.BEAR_CALL,  StrategyType.BUTTERFLY):   "close_to_fly",
}


class PositionMorpher:
    """
    Generates MorphPlans that minimize day-trade usage.
    Pure additions (no closures) use 0 day trades.
    Closures use 1 day trade per pair of legs closed.
    """

    def can_morph(self, current: Position, target: StrategyType) -> bool:
        return (current.strategy, target) in _MORPH_TABLE

    def plan(self, pos: Position, target: StrategyType,
             spot: float, wing_width: float = 2.0) -> Optional[MorphPlan]:
        key    = (pos.strategy, target)
        action = _MORPH_TABLE.get(key)
        if not action:
            return None

        ref_expiry = pos.legs[0].expiry

        if action == "add_bear_call":
            # Bull put → Iron Condor: add call spread above market, no closes
            cs = round(spot * 1.04 / 0.5) * 0.5
            cl = cs + wing_width
            return MorphPlan(
                legs_to_close=[],
                legs_to_add=[
                    dict(kind="call", action="sell", strike=cs, expiry=ref_expiry, qty=1),
                    dict(kind="call", action="buy",  strike=cl, expiry=ref_expiry, qty=1),
                ],
                target_strategy=target,
                day_trades_used=0,
                estimated_net=0.10,
            )

        if action == "add_bull_put":
            # Bear call → Iron Condor: add put spread below market, no closes
            ps = round(spot * 0.96 / 0.5) * 0.5
            pl = ps - wing_width
            return MorphPlan(
                legs_to_close=[],
                legs_to_add=[
                    dict(kind="put", action="sell", strike=ps, expiry=ref_expiry, qty=1),
                    dict(kind="put", action="buy",  strike=pl, expiry=ref_expiry, qty=1),
                ],
                target_strategy=target,
                day_trades_used=0,
                estimated_net=0.10,
            )

        if action == "tighten_to_fly":
            # Iron Condor → Butterfly: buy back the outer wings (1 DT each side)
            outer_legs = [l for l in pos.legs if l.action == "buy"]
            center     = spot
            new_strike = round(center / 0.5) * 0.5
            kind       = "call" if pos.legs[0].kind == "call" else "put"
            return MorphPlan(
                legs_to_close=outer_legs,
                legs_to_add=[
                    dict(kind=kind, action="sell", strike=new_strike,
                         expiry=ref_expiry, qty=2),
                ],
                target_strategy=target,
                day_trades_used=len(outer_legs),
                estimated_net=-0.20,
            )

        return None


# ── Rotation Cycle ────────────────────────────────────────────────────────────

@dataclass
class RotationCycle:
    """Output of one evaluation tick."""
    timestamp:    datetime
    regime:       Regime
    recommended:  StrategyType
    current:      Optional[StrategyType]
    action:       str           # "hold" | "morph" | "close_reopen" | "open" | "stand_aside"
    morph_plan:   Optional[MorphPlan] = None
    reason:       str = ""


# ── Rotating Engine ───────────────────────────────────────────────────────────

class RotatingEngine:
    """
    Top-level controller. Call evaluate() on each tick.

    The engine decides WHAT to do — execution (calling alpaca_options.py)
    is left to the caller so paper/live modes stay separate.

    Small account guardrails:
      - max_risk_pct: never risk more than this fraction of account per position
      - day_trade_budget: remaining PDT allowance this week (reset externally)
      - min_iv_rank_change: ignore regime shifts smaller than this (avoids overtrading)
    """

    def __init__(
        self,
        symbol:              str   = "XSP",
        account_size:        float = 2_000.0,
        max_risk_pct:        float = 0.25,
        day_trade_budget:    int   = 3,
        min_iv_rank_change:  float = 10.0,
        eval_interval_min:   int   = 30,
        tp_threshold:        float = 0.50,   # close at 50% of max profit
        sl_threshold:        float = 1.50,   # close at 150% of credit received
    ):
        self.symbol             = symbol
        self.account_size       = account_size
        self.max_risk_pct       = max_risk_pct
        self.day_trade_budget   = day_trade_budget
        self.min_iv_rank_change = min_iv_rank_change
        self.eval_interval_min  = eval_interval_min
        self.tp_threshold       = tp_threshold
        self.sl_threshold       = sl_threshold

        self.selector           = StrategySelector()
        self.morpher            = PositionMorpher()
        self.current_position:  Optional[Position] = None
        self.history:           List[RotationCycle] = []
        self._last_iv_rank:     float = 50.0

    @property
    def max_risk_dollars(self) -> float:
        return self.account_size * self.max_risk_pct

    def contracts_for_spread(self, wing_width: float, credit: float) -> int:
        """How many contracts fit within the risk budget."""
        max_risk_per = (wing_width - credit) * _SHARES
        if max_risk_per <= 0:
            return 0
        return max(1, int(self.max_risk_dollars / max_risk_per))

    def evaluate(self, regime: Regime) -> RotationCycle:
        recommended  = self.selector.select(regime)
        current_type = self.current_position.strategy if self.current_position else None

        # Check exit conditions first
        if self.current_position:
            pos = self.current_position
            if pos.tp_pct() >= self.tp_threshold:
                cycle = RotationCycle(
                    timestamp=datetime.now(), regime=regime,
                    recommended=recommended, current=current_type,
                    action="close_tp",
                    reason=f"TP hit {pos.tp_pct():.0%}"
                )
                self.history.append(cycle)
                return cycle

            if pos.net_credit > 0 and pos.close_cost > pos.net_credit * self.sl_threshold:
                cycle = RotationCycle(
                    timestamp=datetime.now(), regime=regime,
                    recommended=recommended, current=current_type,
                    action="close_sl",
                    reason=f"SL hit: close_cost > {self.sl_threshold:.0%} of credit"
                )
                self.history.append(cycle)
                return cycle

        iv_shift = abs(regime.iv_rank - self._last_iv_rank)
        self._last_iv_rank = regime.iv_rank

        if recommended == StrategyType.CASH:
            action = "stand_aside"
            reason = f"VIX={regime.vix:.1f} > 35, standing aside"
        elif current_type is None:
            action = "open"
            reason = f"No position open → {recommended.value}"
        elif current_type == recommended:
            action = "hold"
            reason = "Regime unchanged"
        elif iv_shift < self.min_iv_rank_change and abs(regime.trend) < 0.3:
            action = "hold"
            reason = f"Regime shift too small (IV Δ={iv_shift:.1f})"
        elif self.morpher.can_morph(self.current_position, recommended):
            plan = self.morpher.plan(self.current_position, recommended, regime.vix)
            if plan and plan.day_trades_used <= self.day_trade_budget:
                action = "morph"
                reason = f"{current_type.value} → {recommended.value} (morph, {plan.day_trades_used} DT used)"
            else:
                action = "hold"
                reason = "Morph available but day-trade budget exhausted"
        elif self.day_trade_budget >= 1:
            action = "close_reopen"
            reason = f"Full rotation: {current_type.value} → {recommended.value}"
        else:
            action = "hold"
            reason = "Regime change but no day trades remaining"

        morph_plan = None
        if action == "morph" and self.current_position:
            morph_plan = self.morpher.plan(self.current_position, recommended, regime.vix)

        cycle = RotationCycle(
            timestamp=datetime.now(), regime=regime,
            recommended=recommended, current=current_type,
            action=action, morph_plan=morph_plan, reason=reason
        )
        self.history.append(cycle)
        logger.info("[rotate] %s → action=%s | %s", current_type, action, reason)
        return cycle

    def on_morph_executed(self, plan: MorphPlan, new_legs: List[Leg]):
        """Update state after morph execution. Call from your execution layer."""
        if self.current_position:
            for closed_leg in plan.legs_to_close:
                self.current_position.legs = [
                    l for l in self.current_position.legs if l is not closed_leg
                ]
            self.current_position.legs.extend(new_legs)
            self.current_position.strategy = plan.target_strategy
            self.day_trade_budget -= plan.day_trades_used

    def on_opened(self, position: Position):
        self.current_position = position

    def on_closed(self):
        self.current_position = None


# ── P&L helpers (approximate, strategy-specific) ─────────────────────────────

def _calc_max_loss(pos: Position) -> float:
    legs = pos.legs
    st   = pos.strategy

    wing_pairs = {
        StrategyType.BULL_PUT, StrategyType.BEAR_CALL,
        StrategyType.BULL_CALL_DEBIT, StrategyType.BEAR_PUT_DEBIT,
    }
    if st in wing_pairs and len(legs) == 2:
        width = abs(legs[0].strike - legs[1].strike)
        net   = abs(pos.net_credit)
        if pos.net_credit >= 0:
            return (width - net / _SHARES) * legs[0].qty * _SHARES
        else:
            return net

    if st == StrategyType.IRON_CONDOR and len(legs) == 4:
        put_legs  = sorted([l for l in legs if l.kind == "put"],  key=lambda l: l.strike)
        call_legs = sorted([l for l in legs if l.kind == "call"], key=lambda l: l.strike)
        put_w  = put_legs[1].strike  - put_legs[0].strike  if len(put_legs)  == 2 else 0
        call_w = call_legs[1].strike - call_legs[0].strike if len(call_legs) == 2 else 0
        width  = max(put_w, call_w)
        credit = pos.net_credit / _SHARES
        return (width - credit) * legs[0].qty * _SHARES

    # Fallback: sum of bought premiums
    return sum(l.entry_price * l.qty * _SHARES for l in legs if l.action == "buy")


def _calc_breakevens(pos: Position) -> List[float]:
    legs = pos.legs
    st   = pos.strategy

    if st == StrategyType.BULL_PUT and len(legs) == 2:
        short = next(l for l in legs if l.action == "sell")
        credit_ps = pos.net_credit / (short.qty * _SHARES)
        return [short.strike - credit_ps]

    if st == StrategyType.BEAR_CALL and len(legs) == 2:
        short = next(l for l in legs if l.action == "sell")
        credit_ps = pos.net_credit / (short.qty * _SHARES)
        return [short.strike + credit_ps]

    if st == StrategyType.IRON_CONDOR and len(legs) == 4:
        put_short  = next(l for l in legs if l.kind == "put"  and l.action == "sell")
        call_short = next(l for l in legs if l.kind == "call" and l.action == "sell")
        credit_ps  = pos.net_credit / (put_short.qty * _SHARES)
        return [put_short.strike - credit_ps, call_short.strike + credit_ps]

    return []


# ── Compounding projections ───────────────────────────────────────────────────

def compound_table(
    start:          float,
    monthly_pct:    float,
    months:         int = 36,
) -> List[dict]:
    """
    Returns month-by-month projection.
    monthly_pct: e.g. 3.0 means 3% per month.
    """
    rows = []
    val  = start
    for m in range(1, months + 1):
        val *= (1 + monthly_pct / 100)
        rows.append({
            "month": m,
            "value": round(val, 2),
            "gain":  round(val - start, 2),
            "return_pct": round((val / start - 1) * 100, 1),
        })
    return rows


def xsp_account_sizing(
    account:    float = 2_000.0,
    risk_pct:   float = 0.25,
    wing_width: float = 2.0,
    credit:     float = 0.40,
) -> dict:
    """
    Position sizing for XSP spreads on a small account.
    Returns contract count, max risk, max profit, and return-on-risk.
    """
    max_risk_dollars = account * risk_pct
    max_risk_per     = (wing_width - credit) * _SHARES
    contracts        = max(1, int(max_risk_dollars / max_risk_per))
    total_risk       = max_risk_per * contracts
    total_credit     = credit * contracts * _SHARES
    return {
        "contracts":      contracts,
        "bp_used":        round(total_risk, 2),
        "bp_pct":         round(total_risk / account * 100, 1),
        "max_credit":     round(total_credit, 2),
        "return_on_risk": round(total_credit / total_risk * 100, 1),
        "remaining_bp":   round(account - total_risk, 2),
    }
