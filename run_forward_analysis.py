"""
Forward SPY options chain analysis — dealer positioning outlook across expiries.

Fetches 6-8 weekly expiries and computes gamma term structure:
  - Net GEX and key levels (gamma wall, call wall, put wall) per expiry
  - "Sticky" strikes that appear as significant across multiple expiries
  - Outlook narrative based on dealer positioning slope

Usage:
    python run_forward_analysis.py               # fetch fresh (up to 8 expiries)
    python run_forward_analysis.py --expiries 6  # fewer expiries
    python run_forward_analysis.py --slack        # post to Slack
    python run_forward_analysis.py --no-chart     # skip the ASCII gamma profile
"""
import argparse
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import pytz
import yfinance as yf

sys.path.insert(0, os.path.dirname(__file__))

from src.logger import setup_logging
from src.analysis.gex_scanner import (
    compute_exposures, _spot_and_vix, GEXResult
)
from config.settings import GEX_CHAIN_DIR

ET = pytz.timezone("America/New_York")
_RF      = 0.045
_SHARES  = 100


# -- fetch chain per expiry ---------------------------------------------------

def fetch_chain_all_expiries(spot: float, n_expiries: int = 8,
                              otm_pct: float = 0.10) -> dict[str, list[dict]]:
    """
    Fetch SPY option chain for up to n_expiries expiration dates.
    Returns {expiry_str: [contract_dict, ...]}
    """
    ticker  = yf.Ticker("SPY")
    today   = date.today()
    chains  = {}

    try:
        exps = list(ticker.options)[:n_expiries]
    except Exception as e:
        print(f"ERROR: Could not fetch SPY expiration dates: {e}")
        return chains

    for exp_str in exps:
        try:
            exp_date = date.fromisoformat(exp_str)
            T = max((exp_date - today).days, 0) / 365
            chain = ticker.option_chain(exp_str)
            rows = []

            for df, is_call in [(chain.calls, True), (chain.puts, False)]:
                if df is None or df.empty:
                    continue
                for _, row in df.iterrows():
                    strike = float(row["strike"])
                    if not (spot * (1 - otm_pct) <= strike <= spot * (1 + otm_pct)):
                        continue
                    oi = int(row.get("openInterest") or 0)
                    iv = float(row.get("impliedVolatility") or 0)
                    if oi < 5 or iv < 0.01:
                        continue
                    rows.append(dict(strike=strike, oi=oi, iv=iv, T=T, is_call=is_call))

            if rows:
                chains[exp_str] = rows
                print(f"  {exp_str}  ({(exp_date-today).days}d)  {len(rows)} contracts")
        except Exception as e:
            print(f"  {exp_str}: fetch error — {e}")

    return chains


# -- per-expiry analysis ------------------------------------------------------

@dataclass
class ExpiryProfile:
    expiry:       str
    dte:          int
    net_gex_bn:   float
    gamma_wall:   float
    call_wall:    float
    put_wall:     float
    flip_level:   float
    regime:       str
    total_oi:     int
    call_oi:      int
    put_oi:       int
    pc_ratio:     float
    top_strikes:  list = field(default_factory=list)   # [(strike, gex_bn)]


def profile_expiry(exp_str: str, contracts: list[dict],
                   spot: float, vix: float) -> ExpiryProfile:
    dte = int(float(contracts[0]["T"]) * 365) if contracts else 0
    result = compute_exposures(spot, vix, vix, contracts)

    call_oi = sum(c["oi"] for c in contracts if c["is_call"])
    put_oi  = sum(c["oi"] for c in contracts if not c["is_call"])
    total   = call_oi + put_oi
    pc      = put_oi / call_oi if call_oi > 0 else 0

    return ExpiryProfile(
        expiry=exp_str, dte=dte,
        net_gex_bn=result.net_gex_bn,
        gamma_wall=result.gamma_wall,
        call_wall=result.call_wall,
        put_wall=result.put_wall,
        flip_level=result.flip_level,
        regime=result.gex_regime,
        total_oi=total, call_oi=call_oi, put_oi=put_oi, pc_ratio=round(pc, 2),
        top_strikes=result.top_levels[:4],
    )


# -- sticky strike detection --------------------------------------------------

def find_sticky_strikes(profiles: list[ExpiryProfile],
                        tolerance: float = 2.0) -> dict[float, dict]:
    """
    Find strikes that appear as significant GEX levels across >= 2 expiries.
    Returns {canonical_strike: {count, total_gex_bn, role, expiries}}
    """
    counts: dict[float, dict] = {}

    for p in profiles:
        levels = [
            (p.gamma_wall, "GAMMA_WALL"),
            (p.call_wall,  "CALL_WALL"),
            (p.put_wall,   "PUT_WALL"),
            (p.flip_level, "FLIP"),
        ]
        for strike, role in levels:
            if strike <= 0:
                continue
            matched = None
            for existing in counts:
                if abs(existing - strike) <= tolerance:
                    matched = existing
                    break
            key = matched if matched is not None else strike
            if key not in counts:
                counts[key] = {"count": 0, "roles": set(), "expiries": [], "gex": 0.0}
            counts[key]["count"]    += 1
            counts[key]["roles"].add(role)
            counts[key]["expiries"].append(p.expiry)
            counts[key]["gex"]      += abs(p.net_gex_bn)

    return {k: v for k, v in counts.items() if v["count"] >= 2}


# -- outlook narrative --------------------------------------------------------

def build_outlook(profiles: list[ExpiryProfile], spot: float,
                  sticky: dict) -> str:
    if not profiles:
        return "No data available."

    near = profiles[0]
    far  = profiles[-1]

    # GEX slope
    gex_slope = "FLATTENING" if abs(near.net_gex_bn) > abs(far.net_gex_bn) else "BUILDING"

    # Regime consistency
    regimes    = [p.regime for p in profiles]
    pos_count  = regimes.count("POSITIVE_GAMMA")
    neg_count  = regimes.count("NEGATIVE_GAMMA")
    regime_str = "consistently POSITIVE" if pos_count >= len(profiles) * 0.75 \
                 else "consistently NEGATIVE" if neg_count >= len(profiles) * 0.75 \
                 else "MIXED (regime shift in mid-term)"

    # P/C ratio trend
    pc_near = near.pc_ratio
    pc_far  = far.pc_ratio
    pc_str  = "elevated put buying (bearish hedge)" if pc_far > 1.2 \
              else "normal hedging" if pc_far < 0.8 \
              else "balanced"

    # Major sticky levels
    if sticky:
        top = sorted(sticky.items(), key=lambda x: x[1]["count"], reverse=True)[:3]
        sticky_str = ", ".join(f"${k:.0f} ({v['count']} expiries)" for k, v in top)
    else:
        sticky_str = "none identified"

    call_w = near.call_wall
    put_w  = near.put_wall
    flip_l = near.flip_level
    band   = call_w - put_w

    lines = [
        f"Net GEX across expiries is {gex_slope} — gamma is {'concentrating near-term' if gex_slope == 'FLATTENING' else 'building in later weeks'}.",
        f"Gamma regime is {regime_str} across the {len(profiles)} expiries scanned.",
        f"Near-term range: ${put_w:.0f} put wall to ${call_w:.0f} call wall (${band:.0f} wide). Flip level at ${flip_l:.0f}.",
        f"Put/Call ratio in far expiry: {pc_far:.2f} — {pc_str}.",
        f"Sticky multi-expiry levels: {sticky_str}.",
    ]

    if near.regime == "POSITIVE_GAMMA":
        lines.append(f"Dealer positioning SUPPORTS current levels — expect mean reversion within the ${put_w:.0f}–${call_w:.0f} band near-term.")
    else:
        lines.append(f"Negative gamma near-term — dealers will AMPLIFY moves. A break of ${flip_l:.0f} could accelerate.")

    return "\n  ".join(lines)


# -- ASCII gamma profile bar chart --------------------------------------------

def print_gamma_profile(profiles: list[ExpiryProfile], spot: float) -> None:
    print("\n  GAMMA TERM STRUCTURE (net GEX by expiry)")
    print("  " + "-" * 50)
    max_gex = max(abs(p.net_gex_bn) for p in profiles) or 1
    bar_w   = 30
    for p in profiles:
        fill  = int(abs(p.net_gex_bn) / max_gex * bar_w)
        bar   = ("+" if p.net_gex_bn >= 0 else "-") * fill
        print(f"  {p.expiry}  ({p.dte:2d}d)  {p.net_gex_bn:>+7.2f}B  |{bar:<{bar_w}}|  "
              f"P/C={p.pc_ratio:.2f}  {p.regime[:3]}")
    print()


# -- Slack formatter ----------------------------------------------------------

def format_slack(profiles: list[ExpiryProfile], sticky: dict,
                 spot: float, vix: float, outlook: str) -> str:
    near = profiles[0]
    gex_icon  = ":green_circle:" if near.regime == "POSITIVE_GAMMA" else ":red_circle:"
    lines = [
        f":telescope: *SPY Forward Options Analysis* | {datetime.now(ET).strftime('%Y-%m-%d %H:%M')} ET",
        f"SPY ${spot:.2f}  VIX {vix:.1f}",
        f"",
        f":bar_chart: *Gamma Term Structure*",
        f"```",
        f"Expiry       DTE  Net GEX     P/C   Regime",
        f"{'─' * 52}",
    ]
    for p in profiles:
        reg = "POS" if p.regime == "POSITIVE_GAMMA" else "NEG"
        lines.append(f"{p.expiry}   {p.dte:2d}d  {p.net_gex_bn:>+7.2f}B  {p.pc_ratio:.2f}  {reg}")
    lines.append(f"```")

    if sticky:
        lines += ["", f":pushpin: *Multi-Expiry Sticky Levels* (appear in 2+ expiries)"]
        for strike, info in sorted(sticky.items(), key=lambda x: x[1]["count"], reverse=True):
            roles = " / ".join(sorted(info["roles"]))
            lines.append(f">  `${strike:.0f}` — {roles}  ({info['count']} expiries)")

    lines += [
        "",
        f"{gex_icon} *Near-Term Levels* ({near.expiry})",
        f">  Gamma wall : `${near.gamma_wall:.0f}`",
        f">  Call wall  : `${near.call_wall:.0f}`  (resistance)",
        f">  Put wall   : `${near.put_wall:.0f}`  (support)",
        f">  Flip level : `${near.flip_level:.0f}`",
        "",
        f":bulb: *Outlook*",
        f">  {outlook.replace(chr(10)+'  ', chr(10)+'>  ')}",
    ]
    return "\n".join(lines)


# -- main ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SPY forward options analysis")
    p.add_argument("--expiries", type=int, default=8)
    p.add_argument("--slack",    action="store_true")
    p.add_argument("--no-chart", action="store_true")
    return p.parse_args()


def main():
    setup_logging()
    logging.getLogger().setLevel(logging.WARNING)
    args = parse_args()

    print(f"\nFetching SPY spot and VIX...")
    spot, vix, vix_prev = _spot_and_vix()
    print(f"  SPY ${spot:.2f}  VIX {vix:.1f}")
    print(f"\nFetching up to {args.expiries} SPY option expiries (this may take ~20 s)...")

    chains = fetch_chain_all_expiries(spot, n_expiries=args.expiries, otm_pct=0.10)
    if not chains:
        print("ERROR: No chain data returned.")
        sys.exit(1)

    print(f"\nAnalysing {len(chains)} expiries...")
    profiles = []
    for exp_str, contracts in chains.items():
        p = profile_expiry(exp_str, contracts, spot, vix)
        profiles.append(p)

    profiles.sort(key=lambda p: p.dte)

    sticky  = find_sticky_strikes(profiles)
    outlook = build_outlook(profiles, spot, sticky)

    # -- terminal output -------------------------------------------------------
    print("\n" + "=" * 62)
    print("  SPY FORWARD OPTIONS ANALYSIS")
    print("=" * 62)
    print(f"  SPY: ${spot:.2f}   VIX: {vix:.1f}   Date: {date.today()}")
    print()

    if not args.no_chart:
        print_gamma_profile(profiles, spot)

    print("  LEVELS BY EXPIRY")
    print("  " + "-" * 62)
    hdr = f"  {'Expiry':<12} {'DTE':>4}  {'GEX':>8}  {'GammaW':>7}  {'CallW':>7}  {'PutW':>7}  {'Flip':>7}  {'P/C':>5}"
    print(hdr)
    print("  " + "-" * 62)
    for p in profiles:
        reg = "(+)" if p.regime == "POSITIVE_GAMMA" else "(-)"
        print(f"  {p.expiry:<12} {p.dte:>4}  {p.net_gex_bn:>+7.2f}B  "
              f"{p.gamma_wall:>7.1f}  {p.call_wall:>7.1f}  {p.put_wall:>7.1f}  "
              f"{p.flip_level:>7.1f}  {p.pc_ratio:>5.2f}  {reg}")

    if sticky:
        print(f"\n  STICKY MULTI-EXPIRY LEVELS")
        print("  " + "-" * 40)
        for strike, info in sorted(sticky.items(), key=lambda x: x[1]["count"], reverse=True):
            roles = " / ".join(sorted(info["roles"]))
            print(f"  ${strike:>7.1f}  {info['count']}x  {roles}")

    print(f"\n  OUTLOOK")
    print("  " + "-" * 60)
    print(f"  {outlook}")
    print("\n" + "=" * 62)

    if args.slack:
        try:
            from src.notifications.slack_notifier import send_message
            msg = format_slack(profiles, sticky, spot, vix, outlook)
            send_message(msg)
            print("\nPosted to Slack.")
        except Exception as e:
            print(f"\nSlack post failed: {e}")


if __name__ == "__main__":
    main()
