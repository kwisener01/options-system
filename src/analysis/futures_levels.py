"""
/ES and /MES futures battle plan generator from SPY GEX levels.

Converts SPY options-derived gamma/vanna levels to /ES equivalent prices
so traders on prop firm accounts can use the same dealer-positioning edge.

Key conversion: /ES price ≈ SPY × 10  (approximate; actual index value)
  /ES  = $50 per point (E-mini S&P 500)
  /MES = $5  per point (Micro E-mini)

Morning battle plan: sent at 9:30 AM before the open
  - Key /ES price levels (gamma wall, call/put walls, flip level)
  - Directional setups with /ES entry / stop / target
  - Dollar risk per contract (/ES and /MES)

Intraday proximity alerts: checked every 15 minutes
  - Alert when /ES is within PROXIMITY_PTS of a key level
  - State file prevents re-alerting until price clears the zone
"""
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import pytz

from src.analysis.gex_scanner import GEXResult
from src.analysis.gex_trader import TradeSignal, generate_signal, _build
from config.settings import DATA_DIR, ALLOW_SHORT_GEX

logger = logging.getLogger(__name__)

ET             = pytz.timezone("America/New_York")
SPY_TO_ES      = 10.0          # SPY price → /ES approximate points
ES_TICK        = 0.25          # /ES minimum tick
ES_DOLLAR      = 50.0          # $/point for /ES
MES_DOLLAR     = 5.0           # $/point for /MES
PROXIMITY_PTS  = 3.0           # /ES points — alert when this close to a key level
ALERT_STATE    = os.path.join(DATA_DIR, "futures_alert_state.json")


# ── dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class FuturesLevel:
    name:      str    # GAMMA_WALL | CALL_WALL | PUT_WALL | FLIP_LEVEL
    es_price:  float
    spy_price: float
    role:      str    # PIN | RESISTANCE | SUPPORT | REGIME_CHANGE


@dataclass
class FuturesSetup:
    name:                 str
    direction:            str     # LONG | SHORT
    es_entry:             float
    es_stop:              float
    es_target:            float
    stop_pts:             float   # abs distance in ES points
    target_pts:           float
    rr:                   float
    confidence:           str
    es_risk_per_contract: float   # stop_pts × $50
    mes_risk_per_contract: float  # stop_pts × $5
    reasons:              list = field(default_factory=list)


@dataclass
class FuturesBattlePlan:
    as_of:        date
    es_spot:      float
    spy_spot:     float
    vix:          float
    gex_regime:   str
    vanna_signal: str
    charm_signal: str
    levels:       list   # list[FuturesLevel]
    setups:       list   # list[FuturesSetup]
    advisory:     str


# ── conversion helpers ────────────────────────────────────────────────────────

def _to_es(spy_price: float) -> float:
    """Round SPY price to nearest /ES tick."""
    raw = spy_price * SPY_TO_ES
    return round(raw / ES_TICK) * ES_TICK


def _all_setups(result: GEXResult, allow_short: bool = True) -> list[FuturesSetup]:
    """
    Build every valid GEX setup as a FuturesSetup (not just the best one).
    Uses the same logic as gex_trader but returns all candidates.
    """
    spot   = result.spot
    regime = result.gex_regime
    vanna  = result.vanna_signal
    raw: list[TradeSignal] = []

    if regime == "POSITIVE_GAMMA":
        pct_above_put  = (spot - result.put_wall)  / spot
        pct_below_call = (result.call_wall - spot) / spot
        NEAR = 0.006

        if 0 < pct_above_put <= NEAR:
            stop   = result.put_wall * 0.998
            target = result.gamma_wall if result.gamma_wall > spot else result.call_wall
            s = _build(result, "LONG", "PUT_WALL_BOUNCE", spot, stop, target,
                       [f"Spot {pct_above_put:.2%} above put wall ${result.put_wall:.1f}"])
            if s:
                raw.append(s)

        if allow_short and 0 < pct_below_call <= NEAR:
            stop   = result.call_wall * 1.002
            target = result.gamma_wall if result.gamma_wall < spot else result.put_wall
            s = _build(result, "SHORT", "CALL_WALL_FADE", spot, stop, target,
                       [f"Spot {pct_below_call:.2%} below call wall ${result.call_wall:.1f}"])
            if s:
                raw.append(s)

    if regime == "NEGATIVE_GAMMA":
        pct_above_flip = (spot - result.flip_level) / spot
        pct_below_flip = (result.flip_level - spot) / spot
        FLIP = 0.008

        if 0 < pct_above_flip <= FLIP:
            stop   = result.flip_level * 0.997
            target = result.call_wall
            s = _build(result, "LONG", "FLIP_BREAKOUT_LONG", spot, stop, target,
                       [f"Spot {pct_above_flip:.2%} above flip ${result.flip_level:.1f}"])
            if s:
                raw.append(s)

        if allow_short and 0 < pct_below_flip <= FLIP:
            stop   = result.flip_level * 1.003
            target = result.put_wall
            s = _build(result, "SHORT", "FLIP_BREAKDOWN_SHORT", spot, stop, target,
                       [f"Spot {pct_below_flip:.2%} below flip ${result.flip_level:.1f}"])
            if s:
                raw.append(s)

    # Vanna filter and confidence boost (mirrors gex_trader logic)
    filtered = []
    for s in raw:
        if s.direction == "LONG"  and vanna == "BEARISH":
            continue
        if s.direction == "SHORT" and vanna == "BULLISH":
            continue
        if (s.direction == "LONG"  and vanna == "BULLISH") or \
           (s.direction == "SHORT" and vanna == "BEARISH"):
            levels_map = {"LOW": "MEDIUM", "MEDIUM": "HIGH", "HIGH": "HIGH"}
            s.confidence = levels_map.get(s.confidence, s.confidence)
            s.reasons.append(f"Vanna {vanna} confirms direction")
        filtered.append(s)

    futures = []
    for s in filtered:
        es_entry  = _to_es(s.entry)
        es_stop   = _to_es(s.stop)
        es_target = _to_es(s.target)
        stop_pts   = abs(es_entry - es_stop)
        target_pts = abs(es_target - es_entry)
        futures.append(FuturesSetup(
            name=s.setup, direction=s.direction,
            es_entry=es_entry, es_stop=es_stop, es_target=es_target,
            stop_pts=stop_pts, target_pts=target_pts,
            rr=round(target_pts / stop_pts, 2) if stop_pts > 0 else 0,
            confidence=s.confidence,
            es_risk_per_contract=round(stop_pts * ES_DOLLAR, 0),
            mes_risk_per_contract=round(stop_pts * MES_DOLLAR, 0),
            reasons=s.reasons,
        ))

    return futures


# ── battle plan ───────────────────────────────────────────────────────────────

def build_battle_plan(result: GEXResult,
                      allow_short: bool = True,
                      as_of: Optional[date] = None) -> FuturesBattlePlan:
    """Convert a GEXResult into an /ES futures battle plan."""
    as_of = as_of or date.today()

    levels = [
        FuturesLevel("GAMMA_WALL", _to_es(result.gamma_wall), result.gamma_wall, "PIN"),
        FuturesLevel("CALL_WALL",  _to_es(result.call_wall),  result.call_wall,  "RESISTANCE"),
        FuturesLevel("PUT_WALL",   _to_es(result.put_wall),   result.put_wall,   "SUPPORT"),
        FuturesLevel("FLIP_LEVEL", _to_es(result.flip_level), result.flip_level, "REGIME_CHANGE"),
    ]

    setups = _all_setups(result, allow_short)

    # Advisory
    if result.gex_regime == "POSITIVE_GAMMA" and result.vanna_signal == "BULLISH":
        advisory = "Positive gamma + vanna tailwind — mean-reversion, drift higher likely. Fade extremes."
    elif result.gex_regime == "POSITIVE_GAMMA" and result.vanna_signal == "BEARISH":
        advisory = "Positive gamma but vanna headwind — dealer selling into rips. Fade highs aggressively."
    elif result.gex_regime == "NEGATIVE_GAMMA" and result.vanna_signal == "BEARISH":
        advisory = "Negative gamma + vanna headwind — trending/breakdown risk. Follow momentum, avoid fades."
    elif result.gex_regime == "NEGATIVE_GAMMA" and result.vanna_signal == "BULLISH":
        advisory = "Negative gamma but vanna support — potential whipsaw. Keep stops tight."
    elif result.gex_regime == "NEGATIVE_GAMMA":
        advisory = "Negative gamma — wide ranges expected. Breakout/breakdown setups preferred over fades."
    else:
        advisory = "Positive gamma — range-bound, mean-reversion bias. Sell rips, buy dips between walls."

    return FuturesBattlePlan(
        as_of=as_of,
        es_spot=_to_es(result.spot),
        spy_spot=result.spot,
        vix=result.vix,
        gex_regime=result.gex_regime,
        vanna_signal=result.vanna_signal,
        charm_signal=result.charm_signal,
        levels=levels,
        setups=setups,
        advisory=advisory,
    )


# ── Slack formatters ──────────────────────────────────────────────────────────

def format_battle_plan(plan: FuturesBattlePlan) -> str:
    """Morning Slack message: /ES futures battle plan."""
    regime_icon = ":green_circle:" if plan.gex_regime == "POSITIVE_GAMMA" else ":red_circle:"
    vanna_icon  = {"BULLISH": ":chart_with_upward_trend:",
                   "BEARISH": ":chart_with_downwards_trend:",
                   "NEUTRAL": ":white_circle:"}[plan.vanna_signal]
    charm_icon  = {"BUYING_PRESSURE": ":arrow_up_small:",
                   "SELLING_PRESSURE": ":arrow_down_small:",
                   "NEUTRAL": ":white_circle:"}[plan.charm_signal]

    lines = [
        f":chess_pawn: *{plan.as_of} — /ES Futures Battle Plan*",
        f"",
        f"*Current /ES:* ~{plan.es_spot:.2f}  |  *SPY:* ${plan.spy_spot:.2f}  |  *VIX:* {plan.vix:.1f}",
        f"",
        f":pushpin: *Key Levels (/ES points)*",
    ]

    level_roles = {"PIN": "magnetic pin", "RESISTANCE": "resistance", "SUPPORT": "support",
                   "REGIME_CHANGE": "regime flip if crossed"}
    for lv in sorted(plan.levels, key=lambda l: l.es_price, reverse=True):
        marker = " *<-- HERE*" if abs(lv.es_price - plan.es_spot) < 5 else ""
        lines.append(f">  `{lv.es_price:.2f}` {lv.name.replace('_', ' ')} ({level_roles[lv.role]}){marker}")

    lines += [
        f"",
        f"{regime_icon} *Regime:* {plan.gex_regime.replace('_', ' ')}",
        f"{vanna_icon} *Vanna:* {plan.vanna_signal}",
        f"{charm_icon} *Charm:* {plan.charm_signal}",
        f"",
    ]

    if plan.setups:
        lines.append(f":triangular_flag_on_post: *Active Setups*")
        for s in plan.setups:
            dir_icon = ":large_green_circle:" if s.direction == "LONG" else ":red_circle:"
            conf_badge = {"HIGH": ":fire:", "MEDIUM": ":blue_circle:", "LOW": ":white_circle:"}.get(s.confidence, "")
            lines += [
                f"{dir_icon} *{s.name}* {conf_badge}",
                f">  Entry  : `{s.es_entry:.2f}` (SPY ${s.es_entry / SPY_TO_ES:.2f})",
                f">  Stop   : `{s.es_stop:.2f}`  ({s.stop_pts:.1f} pts  /ES ${s.es_risk_per_contract:,.0f}  /MES ${s.mes_risk_per_contract:,.0f})",
                f">  Target : `{s.es_target:.2f}`  ({s.target_pts:.1f} pts  R:R {s.rr:.1f}x)",
            ]
            for r in s.reasons:
                lines.append(f">  - {r}")
            lines.append("")
    else:
        lines.append(":white_circle: *No active setups in range right now* — watch key levels above")
        lines.append("")

    lines += [
        f":bulb: *Advisory:* {plan.advisory}",
        f"",
        f"_/ES = $50/pt | /MES = $5/pt | Levels approximate (SPY x10)_",
    ]

    return "\n".join(lines)


def format_proximity_alert(level: FuturesLevel, es_spot: float,
                           direction: str) -> str:
    """Intraday Slack alert when /ES approaches a key gamma level."""
    dist = es_spot - level.es_price
    side = "above" if dist > 0 else "below"
    pts  = abs(dist)

    role_desc = {
        "PIN":           "gamma wall (price magnet)",
        "RESISTANCE":    "call wall (resistance / fade zone)",
        "SUPPORT":       "put wall (support / bounce zone)",
        "REGIME_CHANGE": "flip level (regime change if crossed!)",
    }[level.role]

    icon = {
        "RESISTANCE":    ":warning:",
        "SUPPORT":       ":shield:",
        "PIN":           ":dart:",
        "REGIME_CHANGE": ":rotating_light:",
    }[level.role]

    action = {
        "RESISTANCE": "Watch for fade SHORT setup if price stalls here",
        "SUPPORT":    "Watch for bounce LONG setup if price holds here",
        "PIN":        "Price may pin here into close — reduce size",
        "REGIME_CHANGE": "A CLOSE above/below this level flips the gamma regime!",
    }[level.role]

    return (
        f"{icon} *GEX Level Alert — {level.name.replace('_', ' ')}*\n"
        f">  /ES now at `{es_spot:.2f}` — `{pts:.1f} pts {side}` the {role_desc}\n"
        f">  Level: `{level.es_price:.2f}`  (SPY ${level.spy_price:.2f})\n"
        f">  {action}"
    )


# ── proximity alert state ─────────────────────────────────────────────────────

def _load_alert_state() -> dict:
    if os.path.exists(ALERT_STATE):
        try:
            with open(ALERT_STATE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_alert_state(state: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(ALERT_STATE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def check_proximity_alerts(plan: FuturesBattlePlan,
                           es_spot: float) -> list[str]:
    """
    Compare current /ES price against all key levels.
    Returns list of formatted Slack alert strings for levels now within PROXIMITY_PTS.
    Deduplicates: only fires once per level per approach; resets when price clears by 2×.
    """
    state = _load_alert_state()
    today = str(date.today())
    if state.get("date") != today:
        state = {"date": today, "alerted": {}, "cleared": {}}

    alerts = []
    for lv in plan.levels:
        dist = abs(es_spot - lv.es_price)
        key  = lv.name

        if dist <= PROXIMITY_PTS:
            if not state["alerted"].get(key):
                direction = "above" if es_spot > lv.es_price else "below"
                alerts.append(format_proximity_alert(lv, es_spot, direction))
                state["alerted"][key]  = True
                state["cleared"][key]  = False
        elif dist > PROXIMITY_PTS * 2:
            # Price moved away — reset so it can alert again on next approach
            state["alerted"][key] = False
            state["cleared"][key] = True

    _save_alert_state(state)
    return alerts
