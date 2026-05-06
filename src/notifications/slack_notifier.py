import requests
import logging
from datetime import datetime

from config.settings import SLACK_WEBHOOK_URL

logger = logging.getLogger(__name__)


def _post(payload: dict) -> bool:
    if not SLACK_WEBHOOK_URL:
        logger.warning("SLACK_WEBHOOK_URL not set — skipping notification")
        return False
    try:
        resp = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.error("Slack post failed: %s", e)
        return False


def send_message(text: str) -> bool:
    return _post({"text": text})


def send_trade_alert(action: str, symbol: str, amount: float, price: float, reason: str = "") -> bool:
    emoji = ":green_circle:" if action == "BUY" else ":red_circle:"
    text = (
        f"{emoji} *{action}* `{symbol}` — ${amount:,.2f} @ ${price:.2f}\n"
        f"> {reason}"
    )
    return _post({"text": text})


def send_daily_report(
    portfolio_value: float,
    buying_power: float,
    positions: list,
    trades_today: list,
    macro_summary: str,
) -> bool:
    now = datetime.now().strftime("%A %b %d, %Y")

    pos_lines = "\n".join(
        f"  • `{p.symbol}` qty={p.qty}  P&L=${float(p.unrealized_pl):+,.2f}  ({float(p.unrealized_plpc)*100:+.2f}%)"
        for p in positions
    ) or "  _No open positions_"

    trade_lines = "\n".join(
        f"  • {t['action']} `{t['symbol']}` ${t['amount']:,.2f}" for t in trades_today
    ) or "  _No trades today_"

    total_pl = sum(float(p.unrealized_pl) for p in positions)
    pl_emoji = ":chart_with_upwards_trend:" if total_pl >= 0 else ":chart_with_downwards_trend:"

    report = (
        f":bar_chart: *Daily Report — {now}*\n\n"
        f"*Portfolio Value:* ${portfolio_value:,.2f}   "
        f"*Buying Power:* ${buying_power:,.2f}   "
        f"*Unrealized P&L:* {pl_emoji} ${total_pl:+,.2f}\n\n"
        f"*Macro:* {macro_summary}\n\n"
        f"*Open Positions ({len(positions)}):*\n{pos_lines}\n\n"
        f"*Trades Today ({len(trades_today)}):*\n{trade_lines}"
    )
    return _post({"text": report})


def send_monitor_update(macro_summary: str, positions: list, alerts: list) -> bool:
    alert_lines = "\n".join(f"  :warning: {a}" for a in alerts) or "  _No alerts_"
    pos_summary = ", ".join(f"`{p.symbol}`" for p in positions) or "_none_"
    text = (
        f":clock3: *15-min Monitor Update*\n"
        f"*Macro:* {macro_summary}\n"
        f"*Positions:* {pos_summary}\n"
        f"*Alerts:*\n{alert_lines}"
    )
    return _post({"text": text})
