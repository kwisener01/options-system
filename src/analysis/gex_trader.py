"""
GEX-based intraday trade signal generator for SPY.

Setups (positive gamma — mean reversion):
  PUT_WALL_BOUNCE  : spot within 0.6% above put wall  → LONG  to gamma wall
  CALL_WALL_FADE   : spot within 0.6% below call wall → SHORT to gamma wall

Setups (negative gamma — breakout/momentum):
  FLIP_BREAKOUT_LONG    : spot within 0.8% above flip level → LONG  to call wall
  FLIP_BREAKDOWN_SHORT  : spot within 0.8% below flip level → SHORT to put wall

Vanna filter:
  Bearish vanna cancels LONG  setups (VIX rising = dealer selling pressure)
  Bullish vanna cancels SHORT setups (VIX falling = dealer buying pressure)
  Confirming vanna boosts confidence by one level.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

from src.analysis.gex_scanner import GEXResult

logger = logging.getLogger(__name__)

# ── thresholds ────────────────────────────────────────────────────────────────

NEAR_WALL_PCT  = 0.006   # 0.6% — how close to a wall to trigger
FLIP_ZONE_PCT  = 0.008   # 0.8% — how close to flip level to trigger breakout
MIN_RR         = 0.8     # minimum reward / risk ratio
MIN_TARGET_PCT = 0.003   # target must be at least 0.3% from entry
GEX_RISK_PCT   = 0.01    # 1% account equity risked per trade (overridden by settings)


@dataclass
class TradeSignal:
    direction:  str               # LONG | SHORT | NO_TRADE
    setup:      str               # PUT_WALL_BOUNCE | CALL_WALL_FADE |
                                  # FLIP_BREAKOUT_LONG | FLIP_BREAKDOWN_SHORT | NO_SETUP
    entry:      float = 0.0       # spot at signal time
    stop:       float = 0.0
    target:     float = 0.0
    rr:         float = 0.0       # abs(target-entry) / abs(entry-stop)
    confidence: str   = "NONE"    # HIGH | MEDIUM | LOW | NONE
    reasons:    list  = field(default_factory=list)

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry - self.stop)

    @property
    def reward_per_share(self) -> float:
        return abs(self.target - self.entry)


def generate_signal(result: GEXResult,
                    allow_short: bool = True) -> TradeSignal:
    """
    Evaluate all setups against the GEXResult and return the best qualifying signal.
    Returns direction=NO_TRADE if nothing qualifies.
    """
    spot    = result.spot
    regime  = result.gex_regime
    vanna   = result.vanna_signal
    candidates: list[TradeSignal] = []

    # ── positive gamma: mean-reversion setups ─────────────────────────────────
    if regime == "POSITIVE_GAMMA":
        pct_above_put  = (spot - result.put_wall)  / spot
        pct_below_call = (result.call_wall - spot) / spot

        # PUT_WALL_BOUNCE
        if 0 < pct_above_put <= NEAR_WALL_PCT:
            stop   = result.put_wall * 0.998
            target = result.gamma_wall if result.gamma_wall > spot else result.call_wall
            sig    = _build(result, "LONG", "PUT_WALL_BOUNCE", spot, stop, target,
                            [f"Spot {pct_above_put:.2%} above put wall ${result.put_wall:.1f}",
                             "Positive gamma — dealers defend this level"])
            if sig:
                candidates.append(sig)

        # CALL_WALL_FADE
        if allow_short and 0 < pct_below_call <= NEAR_WALL_PCT:
            stop   = result.call_wall * 1.002
            target = result.gamma_wall if result.gamma_wall < spot else result.put_wall
            sig    = _build(result, "SHORT", "CALL_WALL_FADE", spot, stop, target,
                            [f"Spot {pct_below_call:.2%} below call wall ${result.call_wall:.1f}",
                             "Positive gamma — dealers fade this level"])
            if sig:
                candidates.append(sig)

    # ── negative gamma: breakout/momentum setups ──────────────────────────────
    if regime == "NEGATIVE_GAMMA":
        pct_above_flip = (spot - result.flip_level) / spot
        pct_below_flip = (result.flip_level - spot) / spot

        # FLIP_BREAKOUT_LONG
        if 0 < pct_above_flip <= FLIP_ZONE_PCT:
            stop   = result.flip_level * 0.997
            target = result.call_wall
            sig    = _build(result, "LONG", "FLIP_BREAKOUT_LONG", spot, stop, target,
                            [f"Spot {pct_above_flip:.2%} above flip ${result.flip_level:.1f}",
                             "Negative gamma — momentum regime, dealers chase moves"])
            if sig:
                candidates.append(sig)

        # FLIP_BREAKDOWN_SHORT
        if allow_short and 0 < pct_below_flip <= FLIP_ZONE_PCT:
            stop   = result.flip_level * 1.003
            target = result.put_wall
            sig    = _build(result, "SHORT", "FLIP_BREAKDOWN_SHORT", spot, stop, target,
                            [f"Spot {pct_below_flip:.2%} below flip ${result.flip_level:.1f}",
                             "Negative gamma — dealers amplify downside move"])
            if sig:
                candidates.append(sig)

    if not candidates:
        return TradeSignal("NO_TRADE", "NO_SETUP",
                           reasons=[f"No setup in range (regime={regime})"])

    # ── vanna filter / confidence boost ───────────────────────────────────────
    filtered = []
    for s in candidates:
        if s.direction == "LONG"  and vanna == "BEARISH":
            continue   # VIX rising cancels longs
        if s.direction == "SHORT" and vanna == "BULLISH":
            continue   # VIX falling cancels shorts
        # Boost confidence if vanna confirms
        if (s.direction == "LONG"  and vanna == "BULLISH") or \
           (s.direction == "SHORT" and vanna == "BEARISH"):
            s = _boost_confidence(s)
            s.reasons.append(f"Vanna {vanna} confirms direction")
        filtered.append(s)

    if not filtered:
        return TradeSignal("NO_TRADE", "NO_SETUP",
                           reasons=[f"Vanna ({vanna}) cancelled all setups"])

    # Pick highest confidence, break ties by R:R
    _pri = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}
    best = max(filtered, key=lambda s: (_pri.get(s.confidence, 0), s.rr))
    return best


def size_trade(signal: TradeSignal, equity: float, buying_power: float,
               risk_pct: float = GEX_RISK_PCT) -> int:
    """Return share quantity that risks risk_pct of equity at the signal's stop."""
    if signal.direction == "NO_TRADE" or signal.risk_per_share < 0.01:
        return 0
    risk_dollars = equity * risk_pct
    qty = int(risk_dollars / signal.risk_per_share)
    max_bp = int(buying_power / signal.entry) if signal.entry > 0 else 0
    return max(min(qty, max_bp), 0)


def format_signal_message(signal: TradeSignal, qty: int, equity: float) -> str:
    if signal.direction == "NO_TRADE":
        icon = ":white_circle:"
        header = f"{icon} *GEX Signal — NO TRADE* | {signal.reasons[0] if signal.reasons else ''}"
        return header

    icon = ":large_green_circle:" if signal.direction == "LONG" else ":red_circle:"
    conf_badge = {"HIGH": ":fire:", "MEDIUM": ":blue_circle:", "LOW": ":white_circle:"}.get(signal.confidence, "")
    risk_dollars = round(signal.risk_per_share * qty, 2) if qty > 0 else 0
    notional     = round(signal.entry * qty, 2) if qty > 0 else 0

    lines = [
        f"{icon} *GEX Signal — {signal.direction} ({signal.setup})* {conf_badge}",
        f">  Entry   : `${signal.entry:.2f}`",
        f">  Stop    : `${signal.stop:.2f}`  ({abs(signal.entry-signal.stop)/signal.entry:.2%} risk)",
        f">  Target  : `${signal.target:.2f}`  ({abs(signal.target-signal.entry)/signal.entry:.2%} move)",
        f">  R:R     : `{signal.rr:.2f}`",
        f">  Size    : `{qty} shares` @ ${signal.entry:.2f} = ${notional:,.0f}  |  Risk ${risk_dollars:,.0f}",
        f">  Acct 1% risk on ${equity:,.0f}",
    ]
    for r in signal.reasons:
        lines.append(f">  - {r}")
    return "\n".join(lines)


# ── helpers ───────────────────────────────────────────────────────────────────

def _build(result: GEXResult, direction: str, setup: str,
           entry: float, stop: float, target: float,
           reasons: list) -> Optional[TradeSignal]:
    """Validate R:R and min-move, return TradeSignal or None."""
    risk   = abs(entry - stop)
    reward = abs(target - entry)
    if risk < 0.01:
        return None
    if reward / entry < MIN_TARGET_PCT:
        logger.debug("%s target move too small (%.2f%%)", setup, reward / entry * 100)
        return None
    rr = reward / risk
    if rr < MIN_RR:
        logger.debug("%s R:R %.2f below minimum %.2f", setup, rr, MIN_RR)
        return None

    confidence = "HIGH" if rr >= 2.0 else ("MEDIUM" if rr >= 1.2 else "LOW")
    return TradeSignal(direction, setup, entry, stop, target, round(rr, 2), confidence, reasons[:])


def _boost_confidence(sig: TradeSignal) -> TradeSignal:
    levels = ["LOW", "MEDIUM", "HIGH"]
    idx = levels.index(sig.confidence) if sig.confidence in levels else 0
    sig.confidence = levels[min(idx + 1, 2)]
    return sig
