"""
Broken-Wing Butterfly (BWB) trade analyzer.

Evaluates a put BWB (buy H / sell 2x M / buy L) against a rigorous checklist,
computes full risk math, compares to a plain put credit spread at the same
short strike, and issues a final rating: A+ | Acceptable | Skip | Dangerous.

Also provides DoubleBWBInputs / analyze_double_bwb for the Batman / Double BWB
structure (put BWB below spot + call BWB above spot), which profits in the
middle zone regardless of which side the market pins to.
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


# ── Double BWB (Batman) ────────────────────────────────────────────────────────

@dataclass
class DoubleBWBInputs:
    ticker:     str
    spot:       float
    dte:        int
    # Put-side BWB  (buy H_put / sell 2× M_put / buy L_put)
    put_upper:  float       # H_put — buy 1  (closest to spot from below)
    put_short:  float       # M_put — sell 2
    put_lower:  float       # L_put — buy 1  (furthest below spot)
    put_credit: float       # net credit from put side
    # Call-side BWB (buy L_call / sell 2× M_call / buy H_call)
    call_lower: float       # L_call — buy 1  (closest to spot from above)
    call_short: float       # M_call — sell 2
    call_upper: float       # H_call — buy 1  (furthest above spot)
    call_credit: float      # net credit from call side
    # Leg mid-prices (optional — for display)
    put_upper_mid:  float = 0.0
    put_short_mid:  float = 0.0
    put_lower_mid:  float = 0.0
    call_lower_mid: float = 0.0
    call_short_mid: float = 0.0
    call_upper_mid: float = 0.0
    # GEX context
    regime:     str   = "UNKNOWN"
    vix_now:    float = 0.0
    vix_prev:   float = 0.0
    flip_level: float = 0.0
    put_wall:   float = 0.0
    call_wall:  float = 0.0
    major_news: bool  = False


@dataclass
class DoubleBWBResult:
    # Per-side wing math
    put_upper_wing:  float
    put_lower_wing:  float
    call_upper_wing: float
    call_lower_wing: float
    # Combined P&L zones (per share, at expiry)
    total_credit:          float    # both sides OTM → keep this in the middle
    middle_width:          float    # call_lower − put_upper (gap between structures)
    middle_profit_usd:     float    # total_credit × 100
    put_peak_profit_usd:   float    # combined P&L when SPY pins M_put
    call_peak_profit_usd:  float    # combined P&L when SPY pins M_call
    put_flat_loss_usd:     float    # combined P&L below L_put (flat zone, 0 if total_credit ≥ extra_risk)
    call_flat_loss_usd:    float    # combined P&L above H_call
    # Combined breakevens (use total_credit — each side benefits from the other's premium)
    put_lower_breakeven:   float    # descending put tent hits zero
    call_upper_breakeven:  float    # descending call tent hits zero
    # Checklist
    checks:      dict
    setup_score: int
    # Decision
    rating:      str
    main_risk:   str
    exit_plan:   str
    overnight_ok: bool
    summary:     str


def analyze_double_bwb(inp: DoubleBWBInputs) -> DoubleBWBResult:
    """
    Evaluate a Batman / Double BWB structure.

    Key insight: both sides are in the same combined position.  The credit
    received from the call BWB lowers the effective breakeven of the put side
    and vice versa.  The combined breakevens therefore use total_credit.

    Payoff zones at expiry:
      S < L_put        → flat: total_credit − put_extra_risk   (may be loss)
      L_put ≤ S ≤ M_put → ascending put tent peak
      M_put < S < H_put → descending put tent
      H_put ≤ S ≤ L_call → MIDDLE: keep total_credit (no intrinsic on either side)
      L_call ≤ S ≤ M_call → ascending call tent peak
      M_call < S < H_call → descending call tent
      S > H_call       → flat: total_credit − call_extra_risk  (may be loss)
    """

    # ── Wing math ─────────────────────────────────────────────────────────────
    put_upper_wing  = inp.put_upper  - inp.put_short        # H_put  − M_put
    put_lower_wing  = inp.put_short  - inp.put_lower        # M_put  − L_put
    put_extra_risk  = put_lower_wing  - put_upper_wing

    call_upper_wing = inp.call_short - inp.call_lower       # M_call − L_call
    call_lower_wing = inp.call_upper - inp.call_short       # H_call − M_call
    call_extra_risk = call_lower_wing - call_upper_wing

    total_credit = inp.put_credit + inp.call_credit
    middle_width = inp.call_lower - inp.put_upper

    # ── P&L in each zone ──────────────────────────────────────────────────────
    middle_profit_usd    = round(total_credit * 100, 2)

    # At M_put: put side = upper_wing_put + put_credit; call side = call_credit
    put_peak_profit_usd  = round((put_upper_wing + total_credit) * 100, 2)

    # At M_call: call side = upper_wing_call + call_credit; put side = put_credit
    call_peak_profit_usd = round((call_upper_wing + total_credit) * 100, 2)

    # Below L_put (flat): total_credit − put_extra_risk
    put_flat_pnl     = total_credit - put_extra_risk
    put_flat_loss_usd  = round(min(put_flat_pnl, 0) * 100, 2)   # negative = loss

    # Above H_call (flat): total_credit − call_extra_risk
    call_flat_pnl    = total_credit - call_extra_risk
    call_flat_loss_usd = round(min(call_flat_pnl, 0) * 100, 2)

    # ── Combined breakevens (on the slope of each tent) ───────────────────────
    # Descending put tent (L_put ≤ S ≤ M_put):
    #   Combined = H_put − 2M_put + S + total_credit = 0
    #   ⟹ S = 2M_put − H_put − total_credit
    put_lower_breakeven  = 2 * inp.put_short  - inp.put_upper  - total_credit

    # Descending call tent (M_call ≤ S ≤ H_call):
    #   Combined = 2M_call − L_call − S + total_credit = 0
    #   ⟹ S = 2M_call − L_call + total_credit
    call_upper_breakeven = 2 * inp.call_short - inp.call_lower + total_credit

    # ── Checklist ─────────────────────────────────────────────────────────────
    neg_gamma    = "NEGATIVE" in inp.regime.upper()
    vix_rising   = (inp.vix_now > inp.vix_prev * 1.03) if inp.vix_prev else False
    spot_in_mid  = inp.put_upper <= inp.spot <= inp.call_lower
    near_put_str = inp.spot <= inp.put_upper + 2
    near_call_str= inp.spot >= inp.call_lower - 2

    put_short_near_wall  = abs(inp.put_short  - inp.put_wall)  <= 5 if inp.put_wall  else True
    call_short_near_wall = abs(inp.call_short - inp.call_wall) <= 5 if inp.call_wall else True

    checks: dict[str, bool] = {
        "Positive gamma regime":          not neg_gamma,
        "SPY inside middle zone":         spot_in_mid,
        "Put short near put wall":        put_short_near_wall,
        "Call short near call wall":      call_short_near_wall,
        "Both sides credit":              inp.put_credit > 0 and inp.call_credit > 0,
        "Middle zone ≥ $10 wide":         middle_width >= 10,
        "Middle zone profitable":         total_credit > 0,
        "No major news":                  not inp.major_news,
        "Not crowded to put structure":   not near_put_str,
        "Not crowded to call structure":  not near_call_str,
    }
    setup_score = sum(checks.values())

    # ── Rating ─────────────────────────────────────────────────────────────────
    dangerous = (inp.put_credit <= 0 or inp.call_credit <= 0
                 or total_credit <= 0 or near_put_str or near_call_str)
    skip_env  = neg_gamma or vix_rising or setup_score < 5

    if dangerous:
        rating = "Dangerous"
    elif skip_env:
        rating = "Skip"
    elif setup_score >= 8 and put_flat_loss_usd >= -600 and call_flat_loss_usd >= -600:
        rating = "A+"
    else:
        rating = "Acceptable"

    # ── Main risk ──────────────────────────────────────────────────────────────
    risk_list = []
    if neg_gamma:
        risk_list.append("Negative gamma — directional move amplified by dealers")
    if not spot_in_mid:
        risk_list.append("SPY already outside the middle zone — asymmetric exposure")
    if middle_width < 10:
        risk_list.append(f"Middle zone only ${middle_width:.0f} wide — narrow buffer")
    if put_flat_loss_usd < -600:
        risk_list.append(f"Put tail loss ${abs(put_flat_loss_usd):.0f} — lower wing too wide relative to credit")
    if call_flat_loss_usd < -600:
        risk_list.append(f"Call tail loss ${abs(call_flat_loss_usd):.0f} — upper wing too wide relative to credit")
    main_risk = risk_list[0] if risk_list else (
        "Gap move through one side at expiry — monitor both breakevens"
    )

    # ── Exit plan ──────────────────────────────────────────────────────────────
    profit_target = round(middle_profit_usd * 0.40)
    exit_plan = (
        f"Target: +${profit_target} (40% of middle credit ${middle_profit_usd:.0f}). "
        f"Danger zone: SPY < ${inp.put_upper:.0f} or SPY > ${inp.call_lower:.0f}. "
        f"Hard stop if SPY closes through put BE ${put_lower_breakeven:.0f} "
        f"or call BE ${call_upper_breakeven:.0f}. "
        f"Exit both legs before 3:30 PM on expiry day."
    )

    overnight_ok = (
        not neg_gamma and not vix_rising and spot_in_mid
        and not near_put_str and not near_call_str
        and inp.spot > inp.put_upper + 5
        and inp.spot < inp.call_lower - 5
    )

    summary = (
        f"Batman SPY  PUT {inp.put_upper:.0f}/{inp.put_short:.0f}/{inp.put_lower:.0f}  "
        f"CALL {inp.call_lower:.0f}/{inp.call_short:.0f}/{inp.call_upper:.0f}  "
        f"Credit ${total_credit:.2f}  Middle ${middle_width:.0f}pt (${middle_profit_usd:.0f})  "
        f"Peaks PUT ${put_peak_profit_usd:.0f} / CALL ${call_peak_profit_usd:.0f}  "
        f"BE ${put_lower_breakeven:.0f}–${call_upper_breakeven:.0f}  {rating}"
    )

    return DoubleBWBResult(
        put_upper_wing=put_upper_wing,
        put_lower_wing=put_lower_wing,
        call_upper_wing=call_upper_wing,
        call_lower_wing=call_lower_wing,
        total_credit=round(total_credit, 3),
        middle_width=middle_width,
        middle_profit_usd=middle_profit_usd,
        put_peak_profit_usd=put_peak_profit_usd,
        call_peak_profit_usd=call_peak_profit_usd,
        put_flat_loss_usd=put_flat_loss_usd,
        call_flat_loss_usd=call_flat_loss_usd,
        put_lower_breakeven=round(put_lower_breakeven, 2),
        call_upper_breakeven=round(call_upper_breakeven, 2),
        checks=checks,
        setup_score=setup_score,
        rating=rating,
        main_risk=main_risk,
        exit_plan=exit_plan,
        overnight_ok=overnight_ok,
        summary=summary,
    )
