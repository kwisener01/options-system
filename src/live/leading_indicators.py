"""
Cross-asset leading indicators for /ES day trading.

Spreads level analysis across correlated asset classes — each asset provides
its own Previous Day OHLC, intraday VWAP, and breakout/breakdown flags.
Combined with GEX from gex_scanner.py they produce multi-level confluence zones.

Asset proxies (all free via yfinance):
  TLT  → 20Y Treasuries  — inverse to equity (TLT down = rates up = equity headwind)
  QQQ  → Nasdaq 100      — tech/risk appetite; leads SPY on momentum shifts
  UUP  → Dollar Index    — strong dollar = risk-off = equity headwind
  GLD  → Gold            — safe-haven bid confirms risk-off
  HYG  → High Yield ETF  — credit spreads lead equities (HYG weak = trouble ahead)
  ^VIX → Implied vol     — fear gauge; rising VIX = equity headwind

Direction scoring: each asset casts a weighted vote (+1 bull / -1 bear / 0 neutral).
  VIX + TLT signals carry weight 1.5 (highest correlation)
  GEX regime + Vanna carry weight 2.0 (most direct dealer flow)
  QQQ / HYG carry weight 1.0-1.2
  SPY VWAP position carries weight 1.0
  GLD carries weight 0.8 (context signal only)
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf
import pytz

logger = logging.getLogger(__name__)

ET = pytz.timezone("America/New_York")


# ── dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class AssetSnapshot:
    symbol:           str
    price:            float
    pdc:              float   # previous day close
    pdh:              float   # previous day high
    pdl:              float   # previous day low
    vwap:             float   # intraday VWAP (cumulative from open)
    open_:            float   # today's open (first bar)
    above_vwap:       bool
    above_pdc:        bool
    above_pdh:        bool    # breakout above prior high
    below_pdl:        bool    # breakdown below prior low
    intraday_chg_pct: float   # % change from today's open
    vs_pdc_pct:       float   # % change from previous close


@dataclass
class VixSnapshot:
    price:       float
    pdc:         float
    pdh:         float
    pdl:         float
    change:      float    # absolute from pdc
    change_pct:  float
    trend:       str      # RISING | FALLING | FLAT
    regime:      str      # LOW (<15) | NORMAL (15-20) | ELEVATED (20-30) | EXTREME (>30)


@dataclass
class CrossAssetLevels:
    as_of: datetime
    spy:   AssetSnapshot
    tlt:   AssetSnapshot   # bonds (inverse rate proxy)
    qqq:   AssetSnapshot   # nasdaq (tech / risk appetite)
    uup:   AssetSnapshot   # dollar (risk-off proxy)
    gld:   AssetSnapshot   # gold (safe haven)
    hyg:   AssetSnapshot   # high yield (credit spreads)
    vix:   VixSnapshot


@dataclass
class DirectionSignal:
    source:    str
    direction: int     # +1 bullish ES, -1 bearish, 0 neutral
    reason:    str
    weight:    float   # relative importance


@dataclass
class DirectionScore:
    raw_score:      float
    bias:           str    # BULLISH | BEARISH | NEUTRAL
    confidence:     str    # HIGH | MEDIUM | LOW
    signals:        list = field(default_factory=list)  # list[DirectionSignal]
    bull_signals:   int = 0
    bear_signals:   int = 0
    neutral_signals: int = 0


@dataclass
class LevelConfluence:
    price_zone:   float   # ES price where confluence occurs
    spy_zone:     float   # equivalent SPY price
    distance_pts: float   # distance from current ES price (+ = above)
    levels:       list = field(default_factory=list)  # level names at this zone
    strength:     int = 0  # count of confluent levels


# ── data fetching ──────────────────────────────────────────────────────────────

def _fetch_daily(symbol: str, days: int = 5) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(symbol, period=f"{days}d", interval="1d",
                         auto_adjust=True, progress=False)
        return df if not df.empty and len(df) >= 2 else None
    except Exception as e:
        logger.warning("Daily fetch failed for %s: %s", symbol, e)
        return None


def _fetch_intraday(symbol: str) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(symbol, period="1d", interval="5m",
                         auto_adjust=True, progress=False)
        return df if not df.empty else None
    except Exception as e:
        logger.warning("Intraday fetch failed for %s: %s", symbol, e)
        return None


def _vwap(df: pd.DataFrame) -> float:
    try:
        close  = df["Close"].squeeze()
        high   = df["High"].squeeze()
        low    = df["Low"].squeeze()
        volume = df["Volume"].squeeze()
        tp     = (high + low + close) / 3
        return float((tp * volume).cumsum().iloc[-1] / volume.cumsum().iloc[-1])
    except Exception:
        return 0.0


def _build_snapshot(symbol: str) -> Optional[AssetSnapshot]:
    daily = _fetch_daily(symbol)
    if daily is None:
        return None

    intraday = _fetch_intraday(symbol)

    try:
        pdc = float(daily["Close"].squeeze().iloc[-2])
        pdh = float(daily["High"].squeeze().iloc[-2])
        pdl = float(daily["Low"].squeeze().iloc[-2])

        if intraday is not None and not intraday.empty:
            price  = float(intraday["Close"].squeeze().iloc[-1])
            open_  = float(intraday["Open"].squeeze().iloc[0])
            vwap   = _vwap(intraday)
        else:
            price  = float(daily["Close"].squeeze().iloc[-1])
            open_  = float(daily["Open"].squeeze().iloc[-1])
            vwap   = price

        return AssetSnapshot(
            symbol=symbol, price=price, pdc=pdc, pdh=pdh, pdl=pdl,
            vwap=vwap if vwap > 0 else price, open_=open_,
            above_vwap=(price > vwap and vwap > 0),
            above_pdc=(price > pdc),
            above_pdh=(price > pdh),
            below_pdl=(price < pdl),
            intraday_chg_pct=round((price - open_) / open_ * 100, 3) if open_ else 0,
            vs_pdc_pct=round((price - pdc) / pdc * 100, 3) if pdc else 0,
        )
    except Exception as e:
        logger.warning("Snapshot failed for %s: %s", symbol, e)
        return None


def _build_vix() -> Optional[VixSnapshot]:
    try:
        df = yf.download("^VIX", period="5d", interval="1d",
                         auto_adjust=True, progress=False)
        if df.empty or len(df) < 2:
            return None
        price = float(df["Close"].squeeze().iloc[-1])
        pdc   = float(df["Close"].squeeze().iloc[-2])
        pdh   = float(df["High"].squeeze().iloc[-2])
        pdl   = float(df["Low"].squeeze().iloc[-2])
        chg   = price - pdc
        chg_pct = chg / pdc * 100 if pdc else 0

        trend = "FLAT" if abs(chg_pct) < 1.5 else ("RISING" if chg > 0 else "FALLING")
        regime = ("LOW" if price < 15 else "NORMAL" if price < 20
                  else "ELEVATED" if price < 30 else "EXTREME")

        return VixSnapshot(price=price, pdc=pdc, pdh=pdh, pdl=pdl,
                           change=round(chg, 2), change_pct=round(chg_pct, 2),
                           trend=trend, regime=regime)
    except Exception as e:
        logger.warning("VIX snapshot failed: %s", e)
        return None


def fetch_cross_asset_levels() -> Optional[CrossAssetLevels]:
    """Fetch all cross-asset snapshots. Returns None if SPY fails."""
    logger.info("Fetching cross-asset levels...")

    spy = _build_snapshot("SPY")
    if spy is None:
        logger.error("SPY snapshot failed — cannot continue")
        return None

    vix = _build_vix() or VixSnapshot(
        price=20, pdc=20, pdh=20, pdl=20, change=0, change_pct=0,
        trend="FLAT", regime="NORMAL",
    )

    def _safe(sym: str) -> AssetSnapshot:
        snap = _build_snapshot(sym)
        if snap is None:
            logger.warning("Using SPY fallback for %s", sym)
            return spy
        return snap

    return CrossAssetLevels(
        as_of=datetime.now(ET),
        spy=spy,
        tlt=_safe("TLT"),
        qqq=_safe("QQQ"),
        uup=_safe("UUP"),
        gld=_safe("GLD"),
        hyg=_safe("HYG"),
        vix=vix,
    )


# ── direction scoring ──────────────────────────────────────────────────────────

def compute_direction_score(levels: CrossAssetLevels,
                             gex_regime: str = "",
                             vanna_signal: str = "") -> DirectionScore:
    """
    Score each cross-asset signal and produce a composite direction bias for /ES.
    """
    signals: list[DirectionSignal] = []
    vix = levels.vix
    spy = levels.spy

    # ── VIX (weight 1.5) ──────────────────────────────────────────────────────
    if vix.trend == "FALLING":
        signals.append(DirectionSignal("VIX", +1,
            f"VIX falling {vix.change_pct:+.1f}% — fear receding, equity tailwind", 1.5))
    elif vix.trend == "RISING":
        signals.append(DirectionSignal("VIX", -1,
            f"VIX rising {vix.change_pct:+.1f}% — fear rising, equity headwind", 1.5))
    else:
        signals.append(DirectionSignal("VIX", 0,
            f"VIX flat at {vix.price:.1f}", 1.0))

    if vix.price > vix.pdh:
        signals.append(DirectionSignal("VIX_PDH", -1,
            f"VIX above PDH {vix.pdh:.1f} — vol breakout, bearish", 1.5))
    elif vix.price < vix.pdl:
        signals.append(DirectionSignal("VIX_PDL", +1,
            f"VIX below PDL {vix.pdl:.1f} — vol compression, bullish", 1.5))

    # ── TLT / Bonds (weight 1.5) ──────────────────────────────────────────────
    # TLT falling = rates rising = equity headwind (most direct correlation)
    tlt = levels.tlt
    if tlt.below_pdl:
        signals.append(DirectionSignal("TLT", -1,
            f"TLT breakdown below PDL {tlt.pdl:.2f} — rates rising, equity headwind", 1.5))
    elif tlt.above_pdh:
        # Bond breakout — could be risk-off flight; lean bearish equities
        signals.append(DirectionSignal("TLT", -1,
            f"TLT above PDH {tlt.pdh:.2f} — flight to safety bid (risk-off)", 1.2))
    elif not tlt.above_vwap and not tlt.above_pdc:
        signals.append(DirectionSignal("TLT", -1,
            f"TLT below VWAP + PDC — bond selling = rates up = equity pressure", 1.5))
    elif tlt.above_vwap and tlt.above_pdc and not tlt.above_pdh:
        signals.append(DirectionSignal("TLT", 0,
            f"TLT mild bid above VWAP — neutral for equities", 1.0))
    else:
        signals.append(DirectionSignal("TLT", 0,
            f"TLT mixed ({tlt.vs_pdc_pct:+.2f}% vs PDC)", 1.0))

    # ── QQQ / Nasdaq (weight 1.0) ─────────────────────────────────────────────
    qqq = levels.qqq
    qqq_vs_spy = qqq.vs_pdc_pct - spy.vs_pdc_pct   # relative performance

    if qqq.above_pdh:
        signals.append(DirectionSignal("QQQ", +1,
            f"QQQ breakout above PDH {qqq.pdh:.2f} — tech leading, risk-on", 1.0))
    elif qqq.below_pdl:
        signals.append(DirectionSignal("QQQ", -1,
            f"QQQ breakdown below PDL {qqq.pdl:.2f} — tech leading lower", 1.0))
    elif qqq_vs_spy > 0.20:
        signals.append(DirectionSignal("QQQ", +1,
            f"QQQ outperforming SPY by {qqq_vs_spy:+.2f}% — risk appetite", 1.0))
    elif qqq_vs_spy < -0.20:
        signals.append(DirectionSignal("QQQ", -1,
            f"QQQ lagging SPY by {qqq_vs_spy:+.2f}% — tech weakness, caution", 1.0))
    else:
        signals.append(DirectionSignal("QQQ", 0,
            f"QQQ in line with SPY ({qqq_vs_spy:+.2f}%)", 0.8))

    # ── HYG / High Yield Credit (weight 1.2) ──────────────────────────────────
    hyg = levels.hyg
    if hyg.above_pdh:
        signals.append(DirectionSignal("HYG", +1,
            f"HYG breakout above PDH {hyg.pdh:.2f} — credit spreads tightening, risk-on", 1.2))
    elif hyg.below_pdl:
        signals.append(DirectionSignal("HYG", -1,
            f"HYG breakdown below PDL {hyg.pdl:.2f} — credit spreads widening, risk-off", 1.2))
    elif hyg.above_vwap and hyg.above_pdc:
        signals.append(DirectionSignal("HYG", +1,
            "HYG above VWAP + PDC — healthy credit appetite", 1.2))
    elif not hyg.above_vwap and not hyg.above_pdc:
        signals.append(DirectionSignal("HYG", -1,
            "HYG below VWAP + PDC — credit spreads rising, defensive", 1.2))
    else:
        signals.append(DirectionSignal("HYG", 0,
            f"HYG mixed ({hyg.vs_pdc_pct:+.2f}% vs PDC)", 1.0))

    # ── UUP / Dollar (weight 1.0) ─────────────────────────────────────────────
    uup = levels.uup
    if uup.above_pdh:
        signals.append(DirectionSignal("UUP", -1,
            f"Dollar above PDH {uup.pdh:.3f} — risk-off, equity headwind", 1.0))
    elif uup.below_pdl:
        signals.append(DirectionSignal("UUP", +1,
            f"Dollar below PDL {uup.pdl:.3f} — risk-on, equity tailwind", 1.0))
    elif uup.above_vwap and uup.above_pdc:
        signals.append(DirectionSignal("UUP", -1,
            "Dollar firm above VWAP + PDC — risk-off pressure", 1.0))
    elif not uup.above_vwap and not uup.above_pdc:
        signals.append(DirectionSignal("UUP", +1,
            "Dollar weak below VWAP + PDC — risk-on environment", 1.0))
    else:
        signals.append(DirectionSignal("UUP", 0,
            f"Dollar neutral ({uup.vs_pdc_pct:+.2f}%)", 0.8))

    # ── GLD / Gold (weight 0.8, context only) ─────────────────────────────────
    gld = levels.gld
    if gld.above_pdh and spy.vs_pdc_pct < -0.2:
        signals.append(DirectionSignal("GLD", -1,
            "Gold breaking out + equities weak — safe-haven bid confirms risk-off", 0.8))
    elif gld.below_pdl and spy.vs_pdc_pct > 0.2:
        signals.append(DirectionSignal("GLD", +1,
            "Gold weak + equities strong — risk-on rotation", 0.8))
    else:
        signals.append(DirectionSignal("GLD", 0,
            f"Gold neutral ({gld.vs_pdc_pct:+.2f}%)", 0.5))

    # ── GEX regime + Vanna (weight 2.0, highest) ──────────────────────────────
    if gex_regime == "NEGATIVE_GAMMA":
        if spy.above_pdc:
            signals.append(DirectionSignal("GEX_REGIME", +1,
                "Neg gamma + SPY above PDC — dealers chase moves higher", 2.0))
        else:
            signals.append(DirectionSignal("GEX_REGIME", -1,
                "Neg gamma + SPY below PDC — dealers chase moves lower", 2.0))
    elif gex_regime == "POSITIVE_GAMMA":
        signals.append(DirectionSignal("GEX_REGIME", 0,
            "Pos gamma — range-bound mean-reversion expected", 1.5))

    if vanna_signal == "BULLISH":
        signals.append(DirectionSignal("VANNA", +1,
            "Vanna bullish — dealer buying flow on VIX decline", 2.0))
    elif vanna_signal == "BEARISH":
        signals.append(DirectionSignal("VANNA", -1,
            "Vanna bearish — dealer selling flow on VIX rise", 2.0))

    # ── SPY VWAP position (weight 1.0) ────────────────────────────────────────
    if spy.above_vwap:
        signals.append(DirectionSignal("SPY_VWAP", +1,
            f"SPY above VWAP {spy.vwap:.2f} — bullish intraday structure", 1.0))
    else:
        signals.append(DirectionSignal("SPY_VWAP", -1,
            f"SPY below VWAP {spy.vwap:.2f} — bearish intraday structure", 1.0))

    # ── Composite score ───────────────────────────────────────────────────────
    raw     = sum(s.direction * s.weight for s in signals)
    max_wt  = sum(s.weight for s in signals)
    norm    = raw / max_wt if max_wt else 0.0

    n_bull  = sum(1 for s in signals if s.direction > 0)
    n_bear  = sum(1 for s in signals if s.direction < 0)
    n_neut  = sum(1 for s in signals if s.direction == 0)

    bias = ("BULLISH" if norm > 0.25 else "BEARISH" if norm < -0.25 else "NEUTRAL")
    conf = ("HIGH" if abs(norm) > 0.50 else "MEDIUM" if abs(norm) > 0.25 else "LOW")

    return DirectionScore(
        raw_score=round(raw, 2), bias=bias, confidence=conf,
        signals=signals, bull_signals=n_bull,
        bear_signals=n_bear, neutral_signals=n_neut,
    )


# ── confluence detection ────────────────────────────────────────────────────────

def find_level_confluences(es_price: float,
                            gex_levels: list,
                            cross_asset: CrossAssetLevels,
                            spy_to_es: float = 10.0,
                            tolerance_pts: float = 5.0) -> list[LevelConfluence]:
    """
    Find ES zones where multiple levels from different sources cluster.

    gex_levels: list of (name, es_price) tuples from futures_levels.py
    Returns confluences sorted by proximity to current ES price.
    """
    all_levels: list[tuple[float, str]] = list(gex_levels)

    spy = cross_asset.spy
    all_levels += [
        (spy.pdh  * spy_to_es, "SPY:PDH"),
        (spy.pdl  * spy_to_es, "SPY:PDL"),
        (spy.pdc  * spy_to_es, "SPY:PDC"),
        (spy.vwap * spy_to_es, "SPY:VWAP"),
    ]

    # Round numbers near current price
    base = round(es_price / 25) * 25
    for offset in (-50, -25, 0, 25, 50):
        all_levels.append((base + offset, f"ROUND:{int(base + offset)}"))

    # Cluster levels within tolerance
    clusters: dict[float, list[str]] = {}
    for price, name in all_levels:
        placed = False
        for anchor in clusters:
            if abs(price - anchor) <= tolerance_pts:
                clusters[anchor].append(name)
                placed = True
                break
        if not placed:
            clusters[price] = [name]

    result = []
    for anchor, names in clusters.items():
        if len(names) >= 2:
            result.append(LevelConfluence(
                price_zone=anchor,
                spy_zone=round(anchor / spy_to_es, 2),
                distance_pts=round(anchor - es_price, 2),
                levels=names,
                strength=len(names),
            ))

    return sorted(result, key=lambda c: abs(c.distance_pts))


# ── Slack / console formatter ──────────────────────────────────────────────────

def format_cross_asset_summary(levels: CrossAssetLevels, score: DirectionScore) -> str:
    bias_icon = {"BULLISH": ":large_green_circle:", "BEARISH": ":red_circle:",
                 "NEUTRAL": ":white_circle:"}[score.bias]
    conf_icon = {"HIGH": ":fire:", "MEDIUM": ":blue_circle:", "LOW": ":white_circle:"}[score.confidence]

    lines = [
        f":globe_with_meridians: *Cross-Asset Dashboard* | {levels.as_of.strftime('%H:%M ET')}",
        f"",
        f"{bias_icon} *Direction Bias: {score.bias}* {conf_icon} ({score.confidence})",
        f">  Bull: {score.bull_signals}  Bear: {score.bear_signals}  Neutral: {score.neutral_signals}",
        f"",
        f":chart_with_upward_trend: *Asset Levels*",
    ]

    def _line(label: str, s: AssetSnapshot) -> str:
        vw  = ":arrow_up_small:" if s.above_vwap else ":arrow_down_small:"
        pdh = " [PDH]" if s.above_pdh else ""
        pdl = " [PDL]" if s.below_pdl else ""
        return (f">  `{label:<4}` ${s.price:.2f}  {s.vs_pdc_pct:+.2f}%  "
                f"VWAP ${s.vwap:.2f}{vw}{pdh}{pdl}")

    lines += [
        _line("SPY",  levels.spy),
        _line("TLT",  levels.tlt),
        _line("QQQ",  levels.qqq),
        _line("UUP",  levels.uup),
        _line("GLD",  levels.gld),
        _line("HYG",  levels.hyg),
        (f">  `VIX ` {levels.vix.price:.1f}  {levels.vix.change:+.2f} "
         f"({levels.vix.change_pct:+.1f}%)  [{levels.vix.regime}]  {levels.vix.trend}"),
        f"",
        f":signal_strength: *Key Signals*",
    ]

    active = sorted(
        [s for s in score.signals if s.direction != 0],
        key=lambda s: abs(s.direction) * s.weight, reverse=True
    )[:6]
    for sig in active:
        icon = ":large_green_circle:" if sig.direction > 0 else ":red_circle:"
        lines.append(f">  {icon} [{sig.source}] {sig.reason}")

    return "\n".join(lines)


def format_cross_asset_plain(levels: CrossAssetLevels, score: DirectionScore) -> str:
    """Console-friendly version without Slack emoji."""
    bias_bar = ("^^^" if score.bias == "BULLISH" else
                "vvv" if score.bias == "BEARISH" else "---")

    lines = [
        f"  Cross-Asset Dashboard  {levels.as_of.strftime('%H:%M ET')}",
        f"  Bias: {score.bias} ({score.confidence})  {bias_bar}",
        f"  Bull:{score.bull_signals} Bear:{score.bear_signals} Neutral:{score.neutral_signals}",
        f"",
        f"  {'SYM':<5}  {'Price':>8}  {'vs PDC':>7}  {'VWAP':>8}  {'Flags':<14}",
        f"  {'-'*55}",
    ]

    def _row(label: str, s: AssetSnapshot) -> str:
        flags = []
        if s.above_pdh: flags.append("PDH+")
        if s.below_pdl: flags.append("PDL-")
        if s.above_vwap: flags.append("vwap+")
        else: flags.append("vwap-")
        return (f"  {label:<5}  ${s.price:>7.2f}  {s.vs_pdc_pct:>+6.2f}%  "
                f"${s.vwap:>7.2f}  {' '.join(flags):<14}")

    lines += [
        _row("SPY",  levels.spy),
        _row("TLT",  levels.tlt),
        _row("QQQ",  levels.qqq),
        _row("UUP",  levels.uup),
        _row("GLD",  levels.gld),
        _row("HYG",  levels.hyg),
        (f"  {'VIX':<5}  {levels.vix.price:>8.1f}  {levels.vix.change_pct:>+6.1f}%  "
         f"  {'--':>8}  [{levels.vix.regime}] {levels.vix.trend}"),
        f"",
        f"  Active Signals (top 5):",
    ]

    active = sorted(
        [s for s in score.signals if s.direction != 0],
        key=lambda s: abs(s.direction) * s.weight, reverse=True
    )[:5]
    for sig in active:
        arrow = "+" if sig.direction > 0 else "-"
        lines.append(f"    [{arrow}] {sig.source}: {sig.reason}")

    return "\n".join(lines)
