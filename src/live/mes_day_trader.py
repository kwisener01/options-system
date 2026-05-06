"""
MES/ES futures day trading system for MyFundedFutures (MFF) challenge accounts.

Combines GEX structural levels + cross-asset leading indicators to produce
high-conviction /MES entry/stop/target setups with full prop-firm risk rules.

MFF account presets (all accounts):
  10k  → target $600   | daily $500  | DD $1,000  | 10 days
  25k  → target $1,500 | daily $1,000| DD $1,500  | 10 days
  50k  → target $3,000 | daily $1,500| DD $2,500  | 10 days  ← default
  100k → target $6,000 | daily $3,000| DD $5,000  | 10 days
  150k → target $9,000 | daily $4,500| DD $7,500  | 10 days
  250k → target $15,000| daily $7,500| DD $12,500 | 10 days

Key rules enforced:
  - Trailing max drawdown: anchors to HIGHEST equity ever reached (not starting)
  - Daily loss limit: hard stop — no new trades
  - Soft stop at 60% of daily limit — step aside early
  - No entries before 9:45 AM ET (open-buffer) or after 3:30 PM ET
  - No overnight holds (MFF requires flat by market close)
  - Risk 1% of account per trade; cap at 3 MES contracts
  - Minimum R:R 1.5 before entering any setup
"""
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, time
from typing import Optional

import pytz

from config.settings import DATA_DIR

logger = logging.getLogger(__name__)

ET          = pytz.timezone("America/New_York")
STATE_FILE  = os.path.join(DATA_DIR, "mes_trader_state.json")
SPY_TO_ES   = 10.0
MES_DOLLAR  = 5.0   # $/point
ES_DOLLAR   = 50.0  # $/point
ES_TICK     = 0.25


# ── MFF account presets ────────────────────────────────────────────────────────

MFF_ACCOUNTS: dict[str, dict] = {
    "10k":  dict(account_size=10_000,  profit_target=600,    daily_loss=500,   max_dd=1_000,  min_days=10, max_mes=3),
    "25k":  dict(account_size=25_000,  profit_target=1_500,  daily_loss=1_000, max_dd=1_500,  min_days=10, max_mes=6),
    "50k":  dict(account_size=50_000,  profit_target=3_000,  daily_loss=1_500, max_dd=2_500,  min_days=10, max_mes=10),
    "100k": dict(account_size=100_000, profit_target=6_000,  daily_loss=3_000, max_dd=5_000,  min_days=10, max_mes=20),
    "150k": dict(account_size=150_000, profit_target=9_000,  daily_loss=4_500, max_dd=7_500,  min_days=10, max_mes=30),
    "250k": dict(account_size=250_000, profit_target=15_000, daily_loss=7_500, max_dd=12_500, min_days=10, max_mes=50),
}


# ── dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class MFFRules:
    account_size:         float
    profit_target:        float
    daily_loss:           float   # hard daily loss limit (will be breached if hit)
    max_dd:               float   # trailing drawdown from highest equity
    min_days:             int
    max_mes:              int
    no_overnight:         bool  = True
    soft_stop_pct:        float = 0.60  # stop new entries at this fraction of daily loss
    risk_per_trade_pct:   float = 0.01  # 1% of account_size per trade


@dataclass
class RiskState:
    starting_balance: float
    current_balance:  float
    highest_balance:  float   # trailing DD anchors here
    daily_start:      float   # balance at start of today
    daily_pnl:        float
    gross_profit:     float   # cumulative wins (for profit target)
    days_traded:      int
    trades_today:     int
    is_passed:        bool
    is_breached:      bool
    breach_reason:    str
    last_updated:     str


@dataclass
class MESTradeSignal:
    direction:     str    # LONG | SHORT | NO_TRADE
    setup:         str    # setup name (e.g. PUT_WALL_BOUNCE)
    reason:        str
    confidence:    str    # HIGH | MEDIUM | LOW
    es_entry:      float
    es_stop:       float
    es_target:     float
    stop_pts:      float
    target_pts:    float
    rr:            float
    mes_contracts: int
    mes_risk:      float  # total $ risk on MES position
    es_risk:       float  # equivalent risk for 1 /ES contract
    confluences:   list = field(default_factory=list)
    block_reason:  str = ""


# ── state persistence ──────────────────────────────────────────────────────────

def _load_state(account_size: float) -> RiskState:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                d = json.load(f)
            today = date.today().isoformat()
            if d.get("last_date") != today:
                # New trading day: reset daily fields, keep cumulative
                d["daily_pnl"]    = 0.0
                d["daily_start"]  = d.get("current_balance", account_size)
                # Only count previous day if trades were made
                if d.get("trades_today", 0) > 0:
                    d["days_traded"] = d.get("days_traded", 0) + 1
                d["trades_today"] = 0
                d["last_date"]    = today
            return RiskState(
                starting_balance=d.get("starting_balance", account_size),
                current_balance =d.get("current_balance",  account_size),
                highest_balance =d.get("highest_balance",  account_size),
                daily_start     =d.get("daily_start",      account_size),
                daily_pnl       =d.get("daily_pnl",        0.0),
                gross_profit    =d.get("gross_profit",     0.0),
                days_traded     =d.get("days_traded",      0),
                trades_today    =d.get("trades_today",     0),
                is_passed       =d.get("is_passed",        False),
                is_breached     =d.get("is_breached",      False),
                breach_reason   =d.get("breach_reason",    ""),
                last_updated    =d.get("last_updated",     ""),
            )
        except Exception as e:
            logger.warning("State load failed: %s — starting fresh", e)

    return RiskState(
        starting_balance=account_size, current_balance=account_size,
        highest_balance=account_size,  daily_start=account_size,
        daily_pnl=0.0, gross_profit=0.0, days_traded=0, trades_today=0,
        is_passed=False, is_breached=False, breach_reason="",
        last_updated=datetime.now(ET).isoformat(),
    )


def _save_state(state: RiskState) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    d = asdict(state)
    d["last_date"] = date.today().isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(d, f, indent=2)


# ── risk checks ───────────────────────────────────────────────────────────────

def _dd_floor(state: RiskState, rules: MFFRules) -> float:
    """Lowest allowed balance before trailing drawdown breach."""
    return state.highest_balance - rules.max_dd


def check_can_trade(state: RiskState, rules: MFFRules) -> tuple[bool, str]:
    """
    Returns (ok, reason).  Checks breach → daily loss → soft stop → time.
    """
    if state.is_breached:
        return False, f"Account breached: {state.breach_reason}"
    if state.is_passed:
        return False, "Challenge already passed!"

    floor = _dd_floor(state, rules)
    if state.current_balance <= floor:
        return False, (f"Trailing drawdown floor hit: "
                       f"${state.current_balance:,.0f} <= ${floor:,.0f}")

    if state.daily_pnl <= -rules.daily_loss:
        return False, (f"Daily loss limit: -${-state.daily_pnl:,.0f} "
                       f">= ${rules.daily_loss:,.0f}")

    soft = rules.daily_loss * rules.soft_stop_pct
    if state.daily_pnl <= -soft:
        return False, (f"Soft daily limit ({int(rules.soft_stop_pct*100)}% threshold): "
                       f"-${-state.daily_pnl:,.0f} >= ${soft:,.0f} — standing aside")

    return True, ""


def _size_contracts(rules: MFFRules, stop_pts: float) -> int:
    """1% of account_size risk per trade, capped at 3 and rules.max_mes."""
    if stop_pts <= 0:
        return 1
    risk_budget = rules.account_size * rules.risk_per_trade_pct
    raw = risk_budget / (stop_pts * MES_DOLLAR)
    return max(1, min(int(raw), rules.max_mes, 3))


def _round_es(price: float) -> float:
    return round(price / ES_TICK) * ES_TICK


# ── signal generation ──────────────────────────────────────────────────────────

def generate_mes_signal(gex_result,
                         cross_asset,
                         direction_score,
                         rules: MFFRules,
                         state: RiskState,
                         confluences: Optional[list] = None) -> MESTradeSignal:
    """
    Produce a MES trade signal by combining GEX structural levels with
    cross-asset direction score.

    Entry logic:
      POSITIVE_GAMMA: mean-reversion fades at call_wall / bounces at put_wall
      NEGATIVE_GAMMA: momentum breakouts/breakdowns through flip level
      Direction score must not be HIGH-confidence OPPOSITE
      Minimum R:R 1.5 enforced
    """
    from src.analysis.futures_levels import _to_es

    def _no_trade(reason: str, block: str = "") -> MESTradeSignal:
        return MESTradeSignal(
            direction="NO_TRADE", setup="", reason=reason, confidence="LOW",
            es_entry=0, es_stop=0, es_target=0,
            stop_pts=0, target_pts=0, rr=0,
            mes_contracts=0, mes_risk=0, es_risk=0,
            block_reason=block or reason,
        )

    # ── risk gate ─────────────────────────────────────────────────────────────
    ok, block = check_can_trade(state, rules)
    if not ok:
        return _no_trade(block, "risk_limit")

    # ── time gate ─────────────────────────────────────────────────────────────
    now = datetime.now(ET).time()
    if now < time(9, 45):
        return _no_trade("Wait — first 15 min (9:30-9:45 ET open buffer)", "open_buffer")
    if now >= time(15, 30):
        return _no_trade("After 3:30 PM ET — no new entries (MFF no-overnight rule)", "close_buffer")

    spot         = gex_result.spot
    es_spot      = _to_es(spot)
    regime       = gex_result.gex_regime
    bias         = direction_score.bias
    bias_conf    = direction_score.confidence

    put_wall_es  = _to_es(gex_result.put_wall)
    call_wall_es = _to_es(gex_result.call_wall)
    flip_es      = _to_es(gex_result.flip_level)
    gamma_es     = _to_es(gex_result.gamma_wall)

    NEAR = 4.0    # ES points within which we consider "at a level"
    STOP_BUFFER = 3.0   # ES points of buffer beyond the level for stop placement

    direction  = "NO_TRADE"
    setup_name = ""
    es_entry   = es_spot
    es_stop    = es_spot
    es_target  = es_spot
    reasons: list[str] = []

    # ── LONG setups ───────────────────────────────────────────────────────────
    if bias in ("BULLISH", "NEUTRAL"):

        # Bounce off put wall (pos gamma support)
        if (regime == "POSITIVE_GAMMA"
                and abs(es_spot - put_wall_es) <= NEAR
                and es_spot >= put_wall_es):
            direction  = "LONG"
            setup_name = "PUT_WALL_BOUNCE"
            es_entry   = _round_es(put_wall_es + 1.5)
            es_stop    = _round_es(put_wall_es - STOP_BUFFER)
            es_target  = gamma_es if gamma_es > es_entry else call_wall_es
            reasons.append(f"ES {es_spot - put_wall_es:.1f}pts above put wall {put_wall_es:.2f}")

        # Flip-level breakout (neg gamma momentum long)
        elif (regime == "NEGATIVE_GAMMA"
              and es_spot >= flip_es
              and abs(es_spot - flip_es) <= NEAR):
            direction  = "LONG"
            setup_name = "FLIP_BREAKOUT"
            es_entry   = _round_es(flip_es + 2.0)
            es_stop    = _round_es(flip_es - STOP_BUFFER)
            es_target  = call_wall_es
            reasons.append(f"ES {es_spot - flip_es:.1f}pts above flip {flip_es:.2f}")

    # ── SHORT setups ──────────────────────────────────────────────────────────
    if bias in ("BEARISH", "NEUTRAL") and direction == "NO_TRADE":

        # Fade call wall (pos gamma resistance)
        if (regime == "POSITIVE_GAMMA"
                and abs(es_spot - call_wall_es) <= NEAR
                and es_spot <= call_wall_es):
            direction  = "SHORT"
            setup_name = "CALL_WALL_FADE"
            es_entry   = _round_es(call_wall_es - 1.5)
            es_stop    = _round_es(call_wall_es + STOP_BUFFER)
            es_target  = gamma_es if gamma_es < es_entry else put_wall_es
            reasons.append(f"ES {call_wall_es - es_spot:.1f}pts below call wall {call_wall_es:.2f}")

        # Flip-level breakdown (neg gamma momentum short)
        elif (regime == "NEGATIVE_GAMMA"
              and es_spot <= flip_es
              and abs(es_spot - flip_es) <= NEAR):
            direction  = "SHORT"
            setup_name = "FLIP_BREAKDOWN"
            es_entry   = _round_es(flip_es - 2.0)
            es_stop    = _round_es(flip_es + STOP_BUFFER)
            es_target  = put_wall_es
            reasons.append(f"ES {flip_es - es_spot:.1f}pts below flip {flip_es:.2f}")

    if direction == "NO_TRADE":
        desc = (f"ES {es_spot:.2f} not near any level "
                f"(put {put_wall_es:.2f} / call {call_wall_es:.2f} / flip {flip_es:.2f})")
        return _no_trade(desc, "no_setup")

    # ── Cross-asset veto (HIGH confidence only) ────────────────────────────────
    if direction == "LONG" and bias == "BEARISH" and bias_conf == "HIGH":
        return _no_trade(
            f"{setup_name} vetoed by HIGH-confidence bearish cross-asset score",
            "cross_asset_veto",
        )
    if direction == "SHORT" and bias == "BULLISH" and bias_conf == "HIGH":
        return _no_trade(
            f"{setup_name} vetoed by HIGH-confidence bullish cross-asset score",
            "cross_asset_veto",
        )

    # ── R:R check ─────────────────────────────────────────────────────────────
    stop_pts   = abs(es_entry - es_stop)
    target_pts = abs(es_target - es_entry)

    if stop_pts < 2.0:
        stop_pts = 4.0
        es_stop  = _round_es(es_entry - 4.0 if direction == "LONG" else es_entry + 4.0)

    if target_pts < stop_pts * 1.5:
        # Extend to the next GEX level
        if direction == "LONG":
            es_target  = _round_es(max(call_wall_es, gamma_es))
        else:
            es_target  = _round_es(min(put_wall_es, gamma_es))
        target_pts = abs(es_target - es_entry)

    rr = round(target_pts / stop_pts, 2) if stop_pts > 0 else 0

    if rr < 1.5:
        return _no_trade(
            f"{setup_name} R:R {rr:.2f} < 1.5 minimum (stop {stop_pts:.1f}pts / target {target_pts:.1f}pts)",
            "poor_rr",
        )

    # ── Add cross-asset confirmation reasons ───────────────────────────────────
    confirming = [s.reason for s in direction_score.signals
                  if (s.direction > 0 and direction == "LONG") or
                     (s.direction < 0 and direction == "SHORT")][:3]
    reasons.extend(confirming)

    # ── Confidence: boost when cross-asset aligns ──────────────────────────────
    aligned = (bias == "BULLISH" and direction == "LONG") or (bias == "BEARISH" and direction == "SHORT")
    if aligned and bias_conf == "HIGH":
        final_conf = "HIGH"
    elif aligned and bias_conf == "MEDIUM":
        final_conf = "HIGH"
    elif aligned:
        final_conf = "MEDIUM"
    elif bias == "NEUTRAL":
        final_conf = "MEDIUM"
    else:
        final_conf = "LOW"   # direction score opposes but not high confidence enough to veto

    contracts = _size_contracts(rules, stop_pts)
    mes_risk  = round(contracts * stop_pts * MES_DOLLAR, 0)
    es_risk   = round(stop_pts * ES_DOLLAR, 0)

    return MESTradeSignal(
        direction=direction,
        setup=setup_name,
        reason="; ".join(reasons[:3]),
        confidence=final_conf,
        es_entry=es_entry,
        es_stop=es_stop,
        es_target=es_target,
        stop_pts=stop_pts,
        target_pts=target_pts,
        rr=rr,
        mes_contracts=contracts,
        mes_risk=mes_risk,
        es_risk=es_risk,
        confluences=confluences or [],
    )


# ── trade manager ─────────────────────────────────────────────────────────────

class MESTradeManager:
    """
    Stateful trade manager tracking MFF challenge account P&L, drawdown, progress.
    """

    def __init__(self, account_type: str = "50k"):
        preset = MFF_ACCOUNTS.get(account_type, MFF_ACCOUNTS["50k"])
        self.account_type = account_type
        self.rules = MFFRules(**preset)
        self.state = _load_state(self.rules.account_size)

    def check_risk(self) -> tuple[bool, str]:
        return check_can_trade(self.state, self.rules)

    def on_fill(self, direction: str, es_entry: float, mes_contracts: int) -> None:
        self.state.trades_today += 1
        self.state.last_updated = datetime.now(ET).isoformat()
        _save_state(self.state)
        logger.info("FILL: %s %d MES @ ES %.2f", direction, mes_contracts, es_entry)

    def on_exit(self, direction: str, es_entry: float, es_exit: float,
                mes_contracts: int) -> float:
        pts_pnl = (es_exit - es_entry) if direction == "LONG" else (es_entry - es_exit)
        pnl     = round(pts_pnl * MES_DOLLAR * mes_contracts, 2)

        self.state.daily_pnl        += pnl
        self.state.current_balance  += pnl
        if pnl > 0:
            self.state.gross_profit += pnl

        if self.state.current_balance > self.state.highest_balance:
            self.state.highest_balance = self.state.current_balance

        # Breach checks
        floor = _dd_floor(self.state, self.rules)
        if self.state.current_balance <= floor and not self.state.is_breached:
            self.state.is_breached  = True
            self.state.breach_reason = (
                f"Trailing DD floor: ${self.state.current_balance:,.0f} <= ${floor:,.0f}"
            )
        if self.state.daily_pnl <= -self.rules.daily_loss and not self.state.is_breached:
            self.state.is_breached  = True
            self.state.breach_reason = (
                f"Daily loss ${-self.state.daily_pnl:,.0f} >= limit ${self.rules.daily_loss:,.0f}"
            )

        # Pass check
        if (not self.state.is_breached
                and not self.state.is_passed
                and self.state.gross_profit >= self.rules.profit_target
                and self.state.days_traded  >= self.rules.min_days):
            self.state.is_passed = True

        self.state.last_updated = datetime.now(ET).isoformat()
        _save_state(self.state)

        sign = "+" if pnl >= 0 else ""
        logger.info("EXIT: %s %.2f→%.2f = %.2fpts = %s$%.2f | daily=$%.2f",
                    direction, es_entry, es_exit, pts_pnl, sign, pnl, self.state.daily_pnl)
        return pnl

    def end_of_day(self) -> None:
        if self.state.trades_today > 0:
            self.state.days_traded += 1
        _save_state(self.state)
        logger.info("EOD: days=%d daily_pnl=$%.2f balance=$%.2f",
                    self.state.days_traded, self.state.daily_pnl, self.state.current_balance)

    def reset(self) -> None:
        self.state = RiskState(
            starting_balance=self.rules.account_size,
            current_balance =self.rules.account_size,
            highest_balance =self.rules.account_size,
            daily_start     =self.rules.account_size,
            daily_pnl=0.0, gross_profit=0.0,
            days_traded=0, trades_today=0,
            is_passed=False, is_breached=False, breach_reason="",
            last_updated=datetime.now(ET).isoformat(),
        )
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
        logger.info("State reset — fresh $%s MFF challenge", account_type_label(self.account_type))

    def format_status(self) -> str:
        s = self.state
        r = self.rules
        floor       = _dd_floor(s, r)
        dd_used     = s.highest_balance - s.current_balance
        dd_remain   = r.max_dd - dd_used
        prog_pct    = min(s.gross_profit / r.profit_target * 100, 100) if r.profit_target else 0
        soft_limit  = r.daily_loss * r.soft_stop_pct
        day_remain  = r.daily_loss + s.daily_pnl   # positive = room left
        soft_remain = soft_limit + s.daily_pnl

        icon = (":trophy:" if s.is_passed
                else ":rotating_light:" if s.is_breached
                else ":large_green_circle:" if prog_pct > 50
                else ":blue_circle:")

        lines = [
            f"{icon} MFF Challenge — ${r.account_size:,.0f} ({self.account_type})",
            f"",
            f"  Profit       : ${s.gross_profit:,.2f}  / ${r.profit_target:,.0f}  ({prog_pct:.1f}%)",
            f"  Daily P&L    : ${s.daily_pnl:+,.2f}  (limit -${r.daily_loss:,.0f} | soft -${soft_limit:,.0f})",
            f"  Day room     : ${day_remain:,.2f}  (soft: ${soft_remain:,.2f})",
            f"  Balance      : ${s.current_balance:,.2f}",
            f"",
            f"  DD used      : ${dd_used:,.2f}  / ${r.max_dd:,.0f}  (${dd_remain:,.2f} remaining)",
            f"  DD floor     : ${floor:,.2f}  (highest: ${s.highest_balance:,.2f})",
            f"",
            f"  Days traded  : {s.days_traded} / {r.min_days} min",
            f"  Trades today : {s.trades_today}",
        ]

        if s.is_breached:
            lines += ["", f"  *** BREACHED: {s.breach_reason} ***"]
        elif s.is_passed:
            lines += ["", "  *** CHALLENGE PASSED! ***"]

        return "\n".join(lines)


def account_type_label(t: str) -> str:
    return MFF_ACCOUNTS.get(t, {}).get("account_size", t)


# ── Slack signal formatters ────────────────────────────────────────────────────

def format_slack_signal(signal: MESTradeSignal, gex_result,
                         cross_asset, direction_score) -> str:
    """
    Slack buy/sell alert.  Sent when a valid setup fires — includes entry,
    stop, target, contract count, and the top cross-asset reasons.
    """
    from src.analysis.futures_levels import _to_es
    from datetime import datetime
    import pytz
    now_str = datetime.now(pytz.timezone("America/New_York")).strftime("%H:%M ET")

    dir_icon  = ":large_green_circle:" if signal.direction == "LONG" else ":red_circle:"
    conf_icon = {"HIGH": ":fire:", "MEDIUM": ":blue_circle:", "LOW": ":white_circle:"}.get(signal.confidence, "")
    spy_price = gex_result.spot
    es_spot   = _to_es(spy_price)
    vix       = cross_asset.vix.price
    regime    = gex_result.gex_regime.replace("_", " ")

    lines = [
        f"{dir_icon} *MES {signal.direction} SIGNAL* {conf_icon}  |  {now_str}",
        f"",
        f"*Setup:* `{signal.setup}`  |  *Confidence:* {signal.confidence}",
        f"*SPY:* ${spy_price:.2f}  |  */ES:* {es_spot:.2f}  |  *VIX:* {vix:.1f}  |  *Regime:* {regime}",
        f"",
        f":pushpin: *Levels*",
        f">  Entry  : `/ES {signal.es_entry:.2f}`  (SPY ~${signal.es_entry / 10:.2f})",
        f">  Stop   : `/ES {signal.es_stop:.2f}`  — {signal.stop_pts:.1f} pts = *${signal.mes_risk:,.0f} risk* on {signal.mes_contracts} MES",
        f">  Target : `/ES {signal.es_target:.2f}`  — {signal.target_pts:.1f} pts  R:R *{signal.rr:.1f}x*",
        f"",
        f":globe_with_meridians: *Cross-Asset Score:* {direction_score.bias} ({direction_score.confidence})",
        f">  Bull: {direction_score.bull_signals}  Bear: {direction_score.bear_signals}  Neutral: {direction_score.neutral_signals}",
    ]

    # Top confirming signals
    confirming = [s for s in direction_score.signals
                  if (s.direction > 0 and signal.direction == "LONG") or
                     (s.direction < 0 and signal.direction == "SHORT")]
    confirming.sort(key=lambda s: s.weight, reverse=True)
    for sig in confirming[:3]:
        lines.append(f">  :white_check_mark: [{sig.source}] {sig.reason}")

    # Confluences
    if signal.confluences:
        lines.append(f"")
        lines.append(f":diamond_shape_with_a_dot_inside: *Level Confluences*")
        for c in signal.confluences[:3]:
            dist = f"{c.distance_pts:+.1f}pts"
            lines.append(f">  `{c.price_zone:.2f}` [{dist}]  {c.strength}x: {', '.join(c.levels)}")

    lines += [
        f"",
        f"_/ES=${ES_DOLLAR:.0f}/pt  /MES=${MES_DOLLAR:.0f}/pt  |  MFF $50k challenge_",
    ]
    return "\n".join(lines)


def format_slack_exit(direction: str, es_entry: float, es_exit: float,
                       contracts: int, pnl: float,
                       state: RiskState, rules: MFFRules) -> str:
    icon    = ":white_check_mark:" if pnl >= 0 else ":x:"
    pts_pnl = (es_exit - es_entry) if direction == "LONG" else (es_entry - es_exit)
    prog    = min(state.gross_profit / rules.profit_target * 100, 100) if rules.profit_target else 0
    dd_used = state.highest_balance - state.current_balance

    return (
        f"{icon} *MES {direction} CLOSED*\n"
        f">  Entry `{es_entry:.2f}` -> Exit `{es_exit:.2f}` ({pts_pnl:+.2f} pts)\n"
        f">  {contracts} MES  P&L: *${pnl:+,.2f}*\n"
        f">  Daily P&L: ${state.daily_pnl:+,.2f}  |  Balance: ${state.current_balance:,.2f}\n"
        f">  Challenge: ${state.gross_profit:,.2f} / ${rules.profit_target:,.0f} "
        f"({prog:.1f}%)  DD used: ${dd_used:,.2f}"
    )


def format_slack_morning(gex_result, cross_asset, direction_score,
                          signal: MESTradeSignal, manager: MESTradeManager) -> str:
    """Compact Slack morning briefing — GEX regime + cross-asset bias + primary setup."""
    from src.analysis.futures_levels import _to_es
    from datetime import date as _date

    es_spot   = _to_es(gex_result.spot)
    regime    = gex_result.gex_regime.replace("_", " ")
    vanna     = gex_result.vanna_signal
    vix       = cross_asset.vix.price
    bias      = direction_score.bias
    conf      = direction_score.confidence
    bias_icon = {"BULLISH": ":large_green_circle:", "BEARISH": ":red_circle:",
                 "NEUTRAL": ":white_circle:"}[bias]
    regime_icon = ":green_circle:" if "POSITIVE" in gex_result.gex_regime else ":red_circle:"

    lines = [
        f":chess_pawn: *MES Battle Plan — {_date.today()}*",
        f"",
        f"{regime_icon} *GEX:* {regime}  |  Vanna: {vanna}  |  VIX: {vix:.1f}",
        f"*SPY:* ${gex_result.spot:.2f}  |  */ES:* {es_spot:.2f}",
        f"",
        f":pushpin: *Key /ES Levels*",
    ]

    from src.analysis.futures_levels import _to_es as _t
    levels_sorted = sorted([
        (_t(gex_result.gamma_wall), "GAMMA WALL"),
        (_t(gex_result.call_wall),  "CALL WALL"),
        (_t(gex_result.put_wall),   "PUT WALL"),
        (_t(gex_result.flip_level), "FLIP"),
    ], reverse=True)
    for lv_price, lv_name in levels_sorted:
        here = "  *<-- HERE*" if abs(lv_price - es_spot) < 5 else ""
        lines.append(f">  `{lv_price:.2f}` — {lv_name}{here}")

    lines += [
        f"",
        f"{bias_icon} *Cross-Asset Bias:* {bias} ({conf})",
        f">  Bull: {direction_score.bull_signals}  "
        f"Bear: {direction_score.bear_signals}  "
        f"Neutral: {direction_score.neutral_signals}",
    ]

    # Top 3 signals
    top = sorted([s for s in direction_score.signals if s.direction != 0],
                 key=lambda s: abs(s.direction) * s.weight, reverse=True)[:3]
    for s in top:
        icon = ":large_green_circle:" if s.direction > 0 else ":red_circle:"
        lines.append(f">  {icon} [{s.source}] {s.reason}")

    lines.append("")
    if signal.direction != "NO_TRADE":
        dir_icon  = ":large_green_circle:" if signal.direction == "LONG" else ":red_circle:"
        conf_icon = {"HIGH": ":fire:", "MEDIUM": ":blue_circle:", "LOW": ":white_circle:"}.get(signal.confidence, "")
        lines += [
            f":triangular_flag_on_post: *Primary Setup* {dir_icon} {conf_icon}",
            f">  *{signal.direction}* `{signal.setup}`",
            f">  Entry `{signal.es_entry:.2f}` | Stop `{signal.es_stop:.2f}` ({signal.stop_pts:.1f}pts) | Target `{signal.es_target:.2f}` ({signal.rr:.1f}x R:R)",
            f">  {signal.mes_contracts} MES — ${signal.mes_risk:,.0f} risk",
        ]
    else:
        lines += [
            f":white_circle: *No setup — standing aside*",
            f">  {signal.block_reason or signal.reason}",
        ]

    # MFF challenge progress
    s = manager.state
    r = manager.rules
    prog = min(s.gross_profit / r.profit_target * 100, 100) if r.profit_target else 0
    floor = _dd_floor(s, r)
    lines += [
        f"",
        f":shield: MFF: ${s.gross_profit:,.0f}/${r.profit_target:,.0f} ({prog:.0f}%)  "
        f"Daily: ${s.daily_pnl:+,.0f}  DD floor: ${floor:,.0f}  Days: {s.days_traded}/{r.min_days}",
    ]

    return "\n".join(lines)


# ── morning plan formatter ─────────────────────────────────────────────────────

def format_morning_plan(gex_result, cross_asset, direction_score,
                         signal: MESTradeSignal,
                         manager: MESTradeManager) -> str:
    """Full morning battle plan: cross-asset + GEX + primary signal + MFF status."""
    from src.analysis.futures_levels import build_battle_plan, _to_es
    from src.live.leading_indicators import format_cross_asset_plain

    es_spot = _to_es(gex_result.spot)

    lines = [
        "=" * 65,
        f"  MES/ES BATTLE PLAN  {date.today()}",
        "=" * 65,
        "",
        format_cross_asset_plain(cross_asset, direction_score),
        "",
    ]

    # GEX battle plan (plain text friendly)
    plan = build_battle_plan(gex_result)
    lines += [
        f"  GEX Regime    : {plan.gex_regime}",
        f"  Vanna         : {plan.vanna_signal}",
        f"  Charm         : {plan.charm_signal}",
        f"  ES Spot       : {es_spot:.2f}",
        f"",
        f"  Key /ES Levels:",
    ]
    for lv in sorted(plan.levels, key=lambda l: l.es_price, reverse=True):
        here = " <-- HERE" if abs(lv.es_price - es_spot) < 5 else ""
        lines.append(f"    {lv.es_price:>8.2f}  {lv.name:<14} ({lv.role}){here}")
    lines.append("")

    # Confluences
    if signal.confluences:
        lines.append(f"  Level Confluences ({len(signal.confluences)} zones):")
        for c in signal.confluences[:4]:
            dist = f"{c.distance_pts:+.1f}pts"
            lines.append(f"    {c.price_zone:>8.2f}  [{dist:>8}]  {c.strength}x  {', '.join(c.levels)}")
        lines.append("")

    # Primary signal
    lines.append("  Primary MES Signal:")
    if signal.direction != "NO_TRADE":
        dir_sym = ">>" if signal.direction == "LONG" else "<<"
        lines += [
            f"    {dir_sym} {signal.direction}  [{signal.setup}]  ({signal.confidence})",
            f"    Entry  : /ES {signal.es_entry:.2f}",
            f"    Stop   : /ES {signal.es_stop:.2f}  ({signal.stop_pts:.1f} pts = ${signal.mes_risk:,.0f} on {signal.mes_contracts} MES)",
            f"    Target : /ES {signal.es_target:.2f}  ({signal.target_pts:.1f} pts  R:R {signal.rr:.1f}x)",
            f"    Why    : {signal.reason}",
        ]
    else:
        lines += [
            f"    -- NO TRADE",
            f"    {signal.block_reason or signal.reason}",
        ]

    lines += ["", manager.format_status(), ""]
    return "\n".join(lines)
