"""
Alpaca Algorithmic Trader — main entry point.
Runs the 15-minute monitor loop during market hours and sends a daily EOD report.
ML selection runs at 4 PM ET to queue next-day orders.
"""
import logging
import pytz
from apscheduler.schedulers.blocking import BlockingScheduler

from src.logger import setup_logging
from src.client.alpaca_client import AlpacaClient
from src.analysis.macro_scanner import get_macro_snapshot
from src.notifications.slack_notifier import send_monitor_update, send_daily_report, send_message, send_trade_alert
from src.signals.daily_runner import run_daily_selection
from src.analysis.gex_scanner import scan as gex_scan, format_gex_message
from src.analysis.gex_trader import generate_signal, format_signal_message
from src.live.spy_gex_trader import open_trade, sync_state, close_trade, get_status as gex_status
from config.settings import GEX_RISK_PCT, ALLOW_SHORT_GEX
from src.backtest.multi_strategy_engine import MultiStrategyEngine
from src.live.options_paper_trader import open_weekly, check_tp, settle_expiry
from src.analysis.futures_levels import build_battle_plan, format_battle_plan, check_proximity_alerts
from config.settings import MONITOR_INTERVAL_MINUTES, TIMEZONE, ML_TOP_N, ACCOUNT_RISK_PCT

setup_logging()
logger = logging.getLogger(__name__)

client = AlpacaClient()
trades_today: list = []

# Shared options engine (settings can be tuned here)
_options_engine = MultiStrategyEngine(
    short_otm_pct=0.02,
    spread_width_pct=0.01,
    call_otm_pct=0.02,
    max_risk_pct=0.10,
    max_vix_entry=25.0,
    low_vol_threshold=18.0,
    take_profit_pct=0.50,
)


def monitor_job():
    """Runs every 15 minutes during market hours."""
    if not client.is_market_open():
        logger.info("Market closed — skipping monitor cycle")
        return

    try:
        macro = get_macro_snapshot()
        positions = client.get_positions()
        alerts = []

        if macro.regime == "BEAR":
            alerts.append(f"Bear regime detected — {macro.summary}")
        if macro.vix_signal == "HIGH":
            alerts.append(f"VIX elevated at {macro.vix_level:.1f} — exercise caution")

        send_monitor_update(macro.summary, positions, alerts)
        logger.info("Monitor cycle complete — %d positions open", len(positions))

    except Exception as e:
        logger.error("monitor_job error: %s", e)
        send_message(f":rotating_light: Monitor error: {e}")


def eod_report_job():
    """Runs at 4:05 PM ET — ML selection, rebalance, then daily report."""
    try:
        macro = get_macro_snapshot()
        positions = client.get_positions()
        portfolio_value = client.get_portfolio_value()
        buying_power = client.get_buying_power()

        # --- ML selection & rebalance ---
        selected, scores = run_daily_selection()
        current_symbols = {p.symbol for p in positions}

        # Close positions no longer in the selection (or all positions in BEAR regime)
        for pos in positions:
            if pos.symbol not in selected:
                try:
                    client.sell(pos.symbol)
                    trades_today.append({"action": "SELL", "symbol": pos.symbol, "amount": float(pos.market_value)})
                    send_trade_alert("SELL", pos.symbol, float(pos.market_value), float(pos.current_price), "Dropped from ML selection")
                except Exception as e:
                    logger.error("Sell %s failed: %s", pos.symbol, e)

        # Open new positions with score-weighted dollar allocations
        new_buys = [s for s in selected if s not in current_symbols]
        if new_buys and buying_power > 100:
            total_alloc = min(buying_power * 0.95, portfolio_value * ACCOUNT_RISK_PCT * ML_TOP_N)
            score_sum = sum(scores.get(s, 1.0) for s in new_buys)
            for sym in new_buys:
                weight = scores.get(sym, 1.0) / score_sum if score_sum > 0 else 1.0 / len(new_buys)
                alloc = round(total_alloc * weight, 2)
                if alloc < 10:
                    continue
                try:
                    quote = client.get_latest_quote(sym)
                    client.buy(sym, alloc)
                    trades_today.append({"action": "BUY", "symbol": sym, "amount": alloc})
                    send_trade_alert("BUY", sym, alloc, quote["mid"], f"ML score={scores.get(sym, 0):.2f}")
                except Exception as e:
                    logger.error("Buy %s failed: %s", sym, e)

        # EOD report
        positions = client.get_positions()
        send_daily_report(
            portfolio_value=client.get_portfolio_value(),
            buying_power=client.get_buying_power(),
            positions=positions,
            trades_today=trades_today,
            macro_summary=macro.summary,
        )
        trades_today.clear()
        logger.info("EOD report sent")

    except Exception as e:
        logger.error("eod_report_job error: %s", e)
        send_message(f":rotating_light: EOD report error: {e}")


def gex_scan_and_trade_job():
    """Mon-Fri 9:45 AM ET — GEX scan, Slack alert, signal generation, bracket entry."""
    try:
        result = gex_scan()
        if not result:
            return

        send_message(format_gex_message(result, session="morning"))

        signal = generate_signal(result, allow_short=ALLOW_SHORT_GEX)
        equity = client.get_portfolio_value()
        send_message(format_signal_message(signal, 0, equity))

        if signal.direction != "NO_TRADE":
            trade = open_trade(signal, equity=equity, vix=result.vix,
                               gex_regime=result.gex_regime,
                               vanna_signal=result.vanna_signal)
            if trade:
                logger.info("GEX options trade opened: %s", trade.get("spread_type"))

    except Exception as e:
        logger.error("gex_scan_and_trade_job error: %s", e)
        send_message(f":rotating_light: GEX scan/trade error: {e}")


def gex_afternoon_job():
    """Mon-Fri 3:15 PM ET — sync bracket state, afternoon GEX scan."""
    try:
        closed = sync_state()
        if closed:
            logger.info("Afternoon sync: bracket filled %s P&L=%.2f",
                        closed.get("outcome"), closed.get("pnl", 0))

        result = gex_scan()
        if result:
            send_message(format_gex_message(result, session="afternoon"))

    except Exception as e:
        logger.error("gex_afternoon_job error: %s", e)


def gex_force_close_job():
    """Mon-Fri 3:45 PM ET — force-close any open GEX trade before market close."""
    try:
        status = gex_status()
        if status["open_trade"] is None:
            return
        closed = close_trade(reason="FORCE_CLOSE")
        if closed:
            logger.info("GEX force-closed: P&L=%.2f", closed.get("pnl", 0))
    except Exception as e:
        logger.error("gex_force_close_job error: %s", e)
        send_message(f":rotating_light: GEX force-close error: {e}")


def options_open_job():
    """Friday 9:45 AM ET — open this week's options spread."""
    try:
        open_weekly(_options_engine)
    except Exception as e:
        logger.error("options_open_job error: %s", e)
        send_message(f":rotating_light: Options open error: {e}")


def options_tp_job():
    """Mon-Thu 3:45 PM ET — check take-profit on open spread."""
    try:
        check_tp(tp_pct=_options_engine.take_profit_pct)
    except Exception as e:
        logger.error("options_tp_job error: %s", e)


def options_settle_job():
    """Friday 3:45 PM ET — settle expiring spread + check TP."""
    try:
        check_tp(tp_pct=_options_engine.take_profit_pct)
        settle_expiry()
    except Exception as e:
        logger.error("options_settle_job error: %s", e)
        send_message(f":rotating_light: Options settle error: {e}")


def futures_battle_plan_job():
    """Mon-Fri 9:30 AM ET — /ES futures battle plan from GEX levels."""
    try:
        result = gex_scan()
        if not result:
            return
        plan = build_battle_plan(result, allow_short=ALLOW_SHORT_GEX)
        send_message(format_battle_plan(plan))
        logger.info("Futures battle plan sent: %d setup(s)", len(plan.setups))
    except Exception as e:
        logger.error("futures_battle_plan_job error: %s", e)
        send_message(f":rotating_light: Futures battle plan error: {e}")


def futures_proximity_job():
    """Mon-Fri 9:45–3:45 every 15 min — alert when /ES near a key GEX level."""
    try:
        result = gex_scan()
        if not result:
            return
        plan    = build_battle_plan(result, allow_short=ALLOW_SHORT_GEX)
        es_spot = plan.es_spot
        alerts  = check_proximity_alerts(plan, es_spot)
        for msg in alerts:
            send_message(msg)
        if alerts:
            logger.info("Futures proximity: %d alert(s) sent", len(alerts))
    except Exception as e:
        logger.error("futures_proximity_job error: %s", e)


def main():
    tz = pytz.timezone(TIMEZONE)
    scheduler = BlockingScheduler(timezone=tz)

    # ML equity strategy
    scheduler.add_job(monitor_job,    "cron", day_of_week="mon-fri",
                      hour="9-15", minute=f"*/{MONITOR_INTERVAL_MINUTES}")
    scheduler.add_job(eod_report_job, "cron", day_of_week="mon-fri", hour=16, minute=5)

    # GEX intraday trader
    scheduler.add_job(gex_scan_and_trade_job, "cron", day_of_week="mon-fri", hour=9,  minute=45)
    scheduler.add_job(gex_afternoon_job,      "cron", day_of_week="mon-fri", hour=15, minute=15)
    scheduler.add_job(gex_force_close_job,    "cron", day_of_week="mon-fri", hour=15, minute=45)

    # Options paper trading
    scheduler.add_job(options_open_job,   "cron", day_of_week="fri",     hour=9,  minute=45)
    scheduler.add_job(options_tp_job,     "cron", day_of_week="mon-thu", hour=15, minute=45)
    scheduler.add_job(options_settle_job, "cron", day_of_week="fri",     hour=15, minute=45)

    # /ES futures prop-firm signals
    scheduler.add_job(futures_battle_plan_job, "cron", day_of_week="mon-fri", hour=9, minute=30)
    scheduler.add_job(futures_proximity_job,   "cron", day_of_week="mon-fri",
                      hour="9-15", minute=f"*/{MONITOR_INTERVAL_MINUTES}")

    from config.settings import IS_PAPER
    mode = "paper" if IS_PAPER else "LIVE"
    send_message(f":rocket: Alpaca trader starting up — {mode} mode")
    logger.info("Scheduler starting")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")


if __name__ == "__main__":
    main()
