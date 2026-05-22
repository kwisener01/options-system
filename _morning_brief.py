"""
Morning brief — runs at 9:45 AM ET.
Sends GEX levels + options dry run to Slack.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.logger import setup_logging
setup_logging()

from src.analysis.gex_scanner import scan as gex_scan
from src.live.options_paper_trader import preview_weekly
from src.notifications.slack_notifier import send_message
from src.backtest.multi_strategy_engine import (
    MultiStrategyEngine, STRATEGY_IRON_CONDOR, STRATEGY_BULL_CALL_DEBIT, STRATEGY_CASH
)

# ── GEX levels ────────────────────────────────────────────────────────────────
gex = gex_scan()

if gex:
    cushion_put  = (gex.spot - gex.put_wall)  / gex.spot * 100
    cushion_call = (gex.call_wall - gex.spot) / gex.spot * 100
    cushion_flip = (gex.spot - gex.flip_level) / gex.spot * 100

    send_message(
        f":chart_with_upwards_trend: *SPY Morning GEX Levels*\n"
        f">  SPY: *${gex.spot:.2f}*  |  VIX: *{gex.vix:.2f}*  |  {gex.gex_regime}\n"
        f">  :red_circle: Call wall  : *${gex.call_wall:.2f}*  (+{cushion_call:.1f}%)\n"
        f">  :yellow_circle: Gamma wall : *${gex.gamma_wall:.2f}*\n"
        f">  :white_circle: Flip level : *${gex.flip_level:.2f}*  ({cushion_flip:+.1f}%)\n"
        f">  :large_green_circle: Put wall   : *${gex.put_wall:.2f}*  (-{cushion_put:.1f}%)\n"
        f">  Vanna: {gex.vanna_signal}  |  Charm: {gex.charm_signal}"
    )
else:
    send_message(":warning: GEX scan failed this morning.")

# ── Dry run ───────────────────────────────────────────────────────────────────
engine = MultiStrategyEngine(
    short_otm_pct=0.02,
    spread_width_pct=0.01,
    call_otm_pct=0.02,
    max_risk_pct=0.10,
    max_vix_entry=25.0,
    low_vol_threshold=20.0,
    take_profit_pct=0.50,
)

r   = preview_weekly(engine)
pos = r.get("position")

if r["strategy"] == STRATEGY_CASH:
    send_message(
        f":no_entry: *Options Dry Run — CASH*\n"
        f">  VIX={r['vix']:.1f}  Regime={r['regime']}  -> No trade today"
    )
else:
    if pos:
        nc  = pos["net_credit"]
        c   = pos["contracts"]
        mr  = pos["max_risk"]

        if r["strategy"] == STRATEGY_IRON_CONDOR:
            legs = (
                f"Put {pos['put_short']:.1f}/{pos['put_long']:.1f}  "
                f"+ Call {pos['call_short']:.1f}/{pos['call_long']:.1f}"
            )
        elif r["strategy"] == STRATEGY_BULL_CALL_DEBIT:
            legs = f"Call debit {pos['short_strike']:.1f}/{pos['long_strike']:.1f}"
        else:
            legs = f"{pos['short_strike']:.1f} / {pos['long_strike']:.1f}"

        gex_line = ""
        if r.get("gex_put_wall"):
            gex_line = f">  GEX walls : Put ${r['gex_put_wall']:.1f}  |  Call ${r['gex_call_wall']:.1f}\n"

        send_message(
            f":mag: *Options Dry Run — {r['strategy']}*\n"
            f">  SPY: ${r['spot']:.2f}  |  VIX: {r['vix']:.2f}  |  {r['regime']}\n"
            f">  Strikes   : *{legs}*\n"
            f">  Credit    : *${nc:.2f}/sh x {c} contracts = ${nc*100*c:,.0f}*\n"
            f">  Max risk  : ${mr*100*c:,.0f}  |  Expiry: {pos['expiry_date']}\n"
            f"{gex_line}"
            f">  :white_check_mark: To open: `python run_options_paper.py open`"
        )
    else:
        send_message(":warning: Dry run: strategy selected but position could not be priced.")
