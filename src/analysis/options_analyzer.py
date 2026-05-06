import yfinance as yf
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional

from config.settings import SIGNAL_THRESHOLDS

logger = logging.getLogger(__name__)


@dataclass
class OptionsSignal:
    symbol: str
    put_call_ratio: float
    max_pain: float
    current_price: float
    distance_to_max_pain_pct: float
    gamma_wall_above: Optional[float]
    gamma_wall_below: Optional[float]
    implied_move_pct: float
    bias: str           # "BULLISH" | "BEARISH" | "NEUTRAL"
    action: str         # "BUY" | "SELL" | "HOLD"
    notes: list = field(default_factory=list)


def _calc_max_pain(chain: pd.DataFrame) -> float:
    """Max pain = strike where total option holder loss is minimized."""
    strikes = chain["strike"].unique()
    pain = {}
    for s in strikes:
        call_loss = chain.loc[(chain["strike"] < s) & (chain["optionType"] == "call"), "openInterest"].sum() * (s - chain.loc[chain["strike"] < s, "strike"].values).mean() if any(chain["strike"] < s) else 0
        put_loss = chain.loc[(chain["strike"] > s) & (chain["optionType"] == "put"), "openInterest"].sum() * (chain.loc[chain["strike"] > s, "strike"].values.mean() - s) if any(chain["strike"] > s) else 0
        pain[s] = call_loss + put_loss
    return min(pain, key=pain.get) if pain else 0.0


def _calc_gamma_walls(chain: pd.DataFrame, current_price: float):
    """Strikes with highest call/put OI above and below current price (gamma walls)."""
    calls = chain[chain["optionType"] == "call"].copy()
    puts = chain[chain["optionType"] == "put"].copy()

    wall_above = None
    wall_below = None

    calls_above = calls[calls["strike"] > current_price]
    if not calls_above.empty:
        wall_above = float(calls_above.loc[calls_above["openInterest"].idxmax(), "strike"])

    puts_below = puts[puts["strike"] < current_price]
    if not puts_below.empty:
        wall_below = float(puts_below.loc[puts_below["openInterest"].idxmax(), "strike"])

    return wall_above, wall_below


def analyze_options(symbol: str) -> Optional[OptionsSignal]:
    """Pull and analyze the nearest-expiry options chain for a symbol."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        current_price = info.get("regularMarketPrice") or info.get("currentPrice")
        if not current_price:
            hist = ticker.history(period="1d")
            if hist.empty:
                return None
            current_price = float(hist["Close"].iloc[-1])

        expirations = ticker.options
        if not expirations:
            logger.warning("%s: no options data", symbol)
            return None

        # Use nearest expiry that is at least 7 days out
        import datetime
        today = datetime.date.today()
        valid_expiries = [e for e in expirations if (datetime.date.fromisoformat(e) - today).days >= 7]
        expiry = valid_expiries[0] if valid_expiries else expirations[0]

        opt = ticker.option_chain(expiry)
        calls = opt.calls.copy()
        calls["optionType"] = "call"
        puts = opt.puts.copy()
        puts["optionType"] = "put"
        chain = pd.concat([calls, puts], ignore_index=True)
        chain = chain[chain["openInterest"] > 0]

        if chain.empty:
            return None

        total_call_oi = chain.loc[chain["optionType"] == "call", "openInterest"].sum()
        total_put_oi = chain.loc[chain["optionType"] == "put", "openInterest"].sum()
        put_call_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0

        max_pain = _calc_max_pain(chain)
        dist_to_max_pain = abs(current_price - max_pain) / current_price * 100

        # Implied move from ATM straddle
        atm_calls = calls.iloc[(calls["strike"] - current_price).abs().argsort()[:1]]
        atm_puts = puts.iloc[(puts["strike"] - current_price).abs().argsort()[:1]]
        atm_straddle = 0.0
        if not atm_calls.empty and not atm_puts.empty:
            atm_straddle = float(atm_calls["lastPrice"].iloc[0]) + float(atm_puts["lastPrice"].iloc[0])
        implied_move_pct = atm_straddle / current_price * 100 if current_price > 0 else 0.0

        wall_above, wall_below = _calc_gamma_walls(chain, current_price)

        # Bias
        notes = []
        bullish_count = 0
        bearish_count = 0

        if put_call_ratio < SIGNAL_THRESHOLDS["put_call_ratio_bullish"]:
            bullish_count += 1
            notes.append(f"Low P/C ratio ({put_call_ratio:.2f}) → bullish")
        elif put_call_ratio > SIGNAL_THRESHOLDS["put_call_ratio_bearish"]:
            bearish_count += 1
            notes.append(f"High P/C ratio ({put_call_ratio:.2f}) → bearish")

        if max_pain > current_price * 1.005:
            bullish_count += 1
            notes.append(f"Max pain {max_pain:.2f} above price → bullish pull")
        elif max_pain < current_price * 0.995:
            bearish_count += 1
            notes.append(f"Max pain {max_pain:.2f} below price → bearish pull")

        if wall_above and (wall_above - current_price) / current_price < SIGNAL_THRESHOLDS["distance_to_wall_pct"] / 100:
            bearish_count += 1
            notes.append(f"Gamma wall near above ({wall_above:.2f}) → resistance")

        bias = "BULLISH" if bullish_count > bearish_count else "BEARISH" if bearish_count > bullish_count else "NEUTRAL"
        action = "BUY" if bias == "BULLISH" else "SELL" if bias == "BEARISH" else "HOLD"

        return OptionsSignal(
            symbol=symbol,
            put_call_ratio=round(put_call_ratio, 3),
            max_pain=max_pain,
            current_price=current_price,
            distance_to_max_pain_pct=round(dist_to_max_pain, 2),
            gamma_wall_above=wall_above,
            gamma_wall_below=wall_below,
            implied_move_pct=round(implied_move_pct, 2),
            bias=bias,
            action=action,
            notes=notes,
        )

    except Exception as e:
        logger.error("options_analyzer %s: %s", symbol, e)
        return None
