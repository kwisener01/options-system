"""
Broken-Wing Butterfly (BWB) trade analyzer.

Evaluates a put BWB (buy H / sell 2x M / buy L) against a rigorous checklist,
computes full risk math, compares to a plain put credit spread at the same
short strike, and issues a final rating: A+ | Acceptable | Skip | Dangerous.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BWBInputs:
    ticker: str
    spot: float
    dte: int
    long_upper: float     # highest strike — buy 1
    short_strike: float   # middle strike  — sell 2
    long_lower: float     # lowest strike  — buy 1
    credit: float         # + = credit per share,  - = debit per share
    # GEX context (auto-filled from last scan)
    regime: str     = "UNKNOWN"   # POSITIVE_GAMMA | NEGATIVE_GAMMA
    vix_now: float  = 0.0
    vix_prev: float = 0.0
    flip_level: float  = 0.0
    put_wall: float    = 0.0
    call_wall: float   = 0.0
    major_news: bool   = False


@dataclass
class BWBResult:
    # ── risk math ──────────────────────────────────────────────────────────────
    upper_wing: float
    lower_wing: float
    extra_risk: float
    max_profit_usd: float
    max_loss_usd: float
    lower_breakeven: float
    rr_ratio: float

    # ── checklist ──────────────────────────────────────────────────────────────
    checks: dict         # label -> bool
    setup_score: int     # 0–10

    # ── vs plain credit spread ─────────────────────────────────────────────────
    cs_max_risk_usd: float
    cs_credit_est: float
    bwb_vs_cs: str       # short verdict string

    # ── decision ───────────────────────────────────────────────────────────────
    rating: str          # A+ | Acceptable | Skip | Dangerous
    main_risk: str
    exit_plan: str
    overnight_ok: bool
    overnight_reason: str
    summary: str


def analyze(inp: BWBInputs) -> BWBResult:
    """Evaluate a put BWB and return a structured result."""

    # ── 1. Risk math ───────────────────────────────────────────────────────────
    upper_wing = inp.long_upper - inp.short_strike
    lower_wing = inp.short_strike - inp.long_lower
    extra_risk = lower_wing - upper_wing

    # P&L at expiry:
    #   above H         → keep credit
    #   at M            → max profit = upper_wing + credit
    #   at or below L   → loss = extra_risk - credit  (capped by lower long)
    max_profit_usd = (upper_wing + inp.credit) * 100
    max_loss_usd   = max((extra_risk - inp.credit) * 100, 0.0)

    # Lower breakeven (inside the tent, price falling toward M):
    #   solve (price - (2M - H)) * 100 + credit * 100 = 0
    lower_breakeven = 2 * inp.short_strike - inp.long_upper - inp.credit

    rr_ratio = (max_profit_usd / max_loss_usd) if max_loss_usd > 0 else 99.0

    # ── 2. Checklist ───────────────────────────────────────────────────────────
    neg_gamma   = "NEGATIVE" in inp.regime.upper()
    vix_rising  = (inp.vix_now > inp.vix_prev * 1.03) if inp.vix_prev else False
    near_struct = inp.spot <= inp.long_upper + 2          # within $2 of upper long
    wing_ok     = lower_wing <= upper_wing * 2            # lower ≤ 2× upper

    short_near_wall = (
        abs(inp.short_strike - inp.put_wall) <= 5
        if inp.put_wall else True
    )
    be_below_wall = (
        lower_breakeven < inp.put_wall
        if inp.put_wall else True
    )

    checks: dict[str, bool] = {
        "Price above gamma flip":          (inp.spot > inp.flip_level) if inp.flip_level else True,
        "Regime bullish/neutral":          not neg_gamma,
        "Short strike near put wall":      short_near_wall,
        "Opens for credit":                inp.credit > 0,
        "Lower breakeven below put wall":  be_below_wall,
        "DTE is 10–30 days":               10 <= inp.dte <= 30,
        "No major news scheduled":         not inp.major_news,
        "Max loss ≤ $500/lot":             max_loss_usd <= 500,
        "Price safely above structure":    not near_struct,
        "Lower wing ≤ 2× upper wing":      wing_ok,
    }
    setup_score = sum(checks.values())

    # ── 3. Credit spread comparison ───────────────────────────────────────────
    # Comparable CS: sell short_strike put, buy (short_strike − upper_wing) put.
    # Estimate CS credit as: BWB premium minus the extra broken-wing premium.
    # Rough heuristic: broken wing adds ~(extra_risk × 0.05) per share to credit.
    cs_credit_est  = max(inp.credit - extra_risk * 0.05, 0.10)
    cs_max_risk_usd = max((upper_wing - cs_credit_est) * 100, 0.0)

    # BWB is better if max loss isn't much worse and RR is meaningfully higher.
    if max_loss_usd <= cs_max_risk_usd * 1.3 and rr_ratio >= 1.2:
        bwb_vs_cs = (
            "BWB BETTER — extra credit and lower breakeven justify the structure "
            f"vs a plain ${inp.short_strike:.0f}/{inp.short_strike - upper_wing:.0f} credit spread."
        )
    elif max_loss_usd > cs_max_risk_usd * 2.0:
        bwb_vs_cs = (
            f"CS BETTER — BWB tail risk (${max_loss_usd:.0f}) is >2× the credit spread "
            f"(${cs_max_risk_usd:.0f}). Take the simpler spread instead."
        )
    else:
        bwb_vs_cs = (
            "SIMILAR — marginal difference. Use BWB only if tent profit matters; "
            "otherwise the credit spread is simpler and less stressful."
        )

    # ── 4. Overnight decision ─────────────────────────────────────────────────
    o_checks = [
        inp.spot > inp.long_upper + 5,                        # buffer above structure
        (inp.spot > inp.put_wall) if inp.put_wall else True,  # above put wall
        not neg_gamma,                                         # stable regime
        not vix_rising,                                        # VIX not expanding
        not inp.major_news,                                    # no catalyst
        not near_struct,                                       # not near structure
    ]
    # Negative gamma, VIX rising, and near-structure are hard NOs per strategy rules
    overnight_ok = sum(o_checks) >= 4 and not neg_gamma and not vix_rising and not near_struct

    bad = []
    if not o_checks[0]: bad.append(f"only ${inp.spot - inp.long_upper:.1f} above upper long put")
    if not o_checks[2]: bad.append("negative gamma environment")
    if not o_checks[3]: bad.append("VIX rising")
    if not o_checks[4]: bad.append("major news scheduled")
    if not o_checks[5]: bad.append("price too close to structure")

    overnight_reason = (
        "Safe to hold overnight — price is comfortably above structure in stable conditions."
        if overnight_ok else
        "Do NOT hold overnight: " + "; ".join(bad) + "."
    )

    # ── 5. Exit plan ──────────────────────────────────────────────────────────
    profit_target = round(max_profit_usd * 0.30)
    stop_usd      = round(max_loss_usd   * 0.35)
    exit_plan = (
        f"Take profit at +${profit_target} (30% of max). "
        f"Hard stop at −${stop_usd} (35% of max risk) OR if SPY closes below "
        f"${inp.short_strike:.0f}. Exit before 3:30 PM on expiry day."
    )

    # ── 6. Main risk ──────────────────────────────────────────────────────────
    risk_list = []
    if neg_gamma:
        risk_list.append("Negative gamma — dealer amplification; a move into the structure accelerates")
    if extra_risk > upper_wing * 1.5:
        risk_list.append(
            f"Wide lower wing ({lower_wing:.0f}pt vs {upper_wing:.0f}pt upper) — "
            f"tail risk is disproportionate to credit"
        )
    if inp.credit < 0.30:
        risk_list.append("Thin credit — not enough premium to justify the broken-wing tail risk")
    if near_struct:
        risk_list.append(f"SPY within $2 of upper long put ${inp.long_upper:.0f} — too close")
    if vix_rising:
        risk_list.append("VIX expanding — vol buyers have the edge; wait for VIX to stabilise")
    main_risk = risk_list[0] if risk_list else (
        f"Gap below lower breakeven ${lower_breakeven:.2f} on a large overnight move"
    )

    # ── 7. Rating ─────────────────────────────────────────────────────────────
    # Dangerous: structural problems (too close, loss too large, no credit)
    # Skip: environment problems (neg gamma, VIX rising) — per strategy rules
    dangerous = near_struct or max_loss_usd > 800 or inp.credit <= 0
    skip_env  = neg_gamma or vix_rising or setup_score < 5

    if dangerous:
        rating = "Dangerous"
    elif skip_env:
        rating = "Skip"
    elif setup_score >= 8 and max_loss_usd <= 500 and rr_ratio >= 1.0:
        rating = "A+"
    else:
        rating = "Acceptable"

    # ── 8. Summary line ───────────────────────────────────────────────────────
    cr_str = (f"${inp.credit:.2f} credit" if inp.credit >= 0
              else f"${abs(inp.credit):.2f} debit")
    summary = (
        f"{inp.ticker} {inp.long_upper:.0f}/{inp.short_strike:.0f}/{inp.long_lower:.0f} "
        f"put BWB | {cr_str} | "
        f"Max profit ${max_profit_usd:.0f} | Max loss ${max_loss_usd:.0f} | "
        f"Lower BE ${lower_breakeven:.2f} | Score {setup_score}/10 | {rating}"
    )

    return BWBResult(
        upper_wing=upper_wing,
        lower_wing=lower_wing,
        extra_risk=extra_risk,
        max_profit_usd=max_profit_usd,
        max_loss_usd=max_loss_usd,
        lower_breakeven=lower_breakeven,
        rr_ratio=rr_ratio,
        checks=checks,
        setup_score=setup_score,
        cs_max_risk_usd=cs_max_risk_usd,
        cs_credit_est=cs_credit_est,
        bwb_vs_cs=bwb_vs_cs,
        rating=rating,
        main_risk=main_risk,
        exit_plan=exit_plan,
        overnight_ok=overnight_ok,
        overnight_reason=overnight_reason,
        summary=summary,
    )
