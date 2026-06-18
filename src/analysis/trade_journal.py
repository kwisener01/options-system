"""
Trade journal & per-strategy attribution.

Reads the paper-trader's closed_trades[] ledger and Alpaca's order history
to produce:
  - per-trade log with computed metrics (hold time, return-on-risk)
  - per-strategy rollup (win rate, avg P&L, total P&L, contribution %)
  - monthly attribution table

Data sources (all restart-proof):
  - options_paper_state.json  → closed_trades[]
  - Alpaca order history       → reconciliation / fallback
"""
import json
import logging
import os
from collections import defaultdict
from datetime import date, datetime
from typing import Optional

from config.settings import DATA_DIR

logger = logging.getLogger(__name__)

STATE_PATH = os.path.join(DATA_DIR, "options_paper_state.json")


def _load_closed_trades() -> list[dict]:
    try:
        with open(STATE_PATH) as f:
            state = json.load(f)
        return state.get("closed_trades", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _enrich(trade: dict) -> dict:
    """Add computed fields to a closed trade dict (non-mutating)."""
    t = dict(trade)
    try:
        entry = date.fromisoformat(str(t.get("entry_date", "")))
        close = date.fromisoformat(str(t.get("close_date", "")))
        t["hold_days"] = (close - entry).days
    except (ValueError, TypeError):
        t["hold_days"] = None

    nc = t.get("net_credit", 0)
    mr = t.get("max_risk", 0)
    pnl = t.get("pnl", 0)
    t["return_on_risk"] = round(pnl / (mr * 100), 4) if mr else 0.0
    t["strategy_clean"] = (t.get("strategy") or "UNKNOWN").replace("_", " ").title()
    return t


def get_trades(month: Optional[str] = None) -> list[dict]:
    """Return enriched closed trades, optionally filtered to a month (YYYY-MM)."""
    trades = [_enrich(t) for t in _load_closed_trades()]
    if month:
        trades = [t for t in trades
                  if str(t.get("close_date", "")).startswith(month)]
    return trades


def strategy_attribution(month: Optional[str] = None) -> list[dict]:
    """Per-strategy rollup sorted by total P&L contribution."""
    trades = get_trades(month)
    if not trades:
        return []

    by_strat: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        by_strat[t.get("strategy") or "UNKNOWN"].append(t)

    total_pnl = sum(t["pnl"] for t in trades) or 1e-9

    rows = []
    for strat, group in by_strat.items():
        wins = [t for t in group if t["pnl"] > 0]
        losses = [t for t in group if t["pnl"] <= 0]
        pnls = [t["pnl"] for t in group]
        s_pnl = sum(pnls)
        hold_days = [t["hold_days"] for t in group if t["hold_days"] is not None]

        rows.append({
            "strategy":       strat,
            "strategy_clean": strat.replace("_", " ").title(),
            "trades":         len(group),
            "wins":           len(wins),
            "losses":         len(losses),
            "win_rate":       round(len(wins) / len(group), 3) if group else 0,
            "total_pnl":      round(s_pnl, 2),
            "avg_pnl":        round(s_pnl / len(group), 2),
            "best_trade":     round(max(pnls), 2),
            "worst_trade":    round(min(pnls), 2),
            "avg_hold_days":  round(sum(hold_days) / len(hold_days), 1) if hold_days else 0,
            "contribution":   round(s_pnl / total_pnl, 3) if total_pnl else 0,
        })

    rows.sort(key=lambda r: r["total_pnl"], reverse=True)
    return rows


def monthly_summary() -> dict:
    """Current-month attribution + fund totals."""
    now = date.today()
    month_key = now.strftime("%Y-%m")
    all_trades = get_trades()
    month_trades = get_trades(month_key)

    def _totals(trades):
        if not trades:
            return {"trades": 0, "pnl": 0, "wins": 0, "losses": 0, "win_rate": 0}
        wins = sum(1 for t in trades if t["pnl"] > 0)
        return {
            "trades":   len(trades),
            "pnl":      round(sum(t["pnl"] for t in trades), 2),
            "wins":     wins,
            "losses":   len(trades) - wins,
            "win_rate": round(wins / len(trades), 3),
        }

    return {
        "month":           month_key,
        "month_totals":    _totals(month_trades),
        "month_by_strat":  strategy_attribution(month_key),
        "all_time_totals": _totals(all_trades),
        "all_time_by_strat": strategy_attribution(),
    }


# ── Slack formatters ─────────────────────────────────────────────────────────

def fmt_journal(n: int = 10) -> str:
    """Format recent trades for Slack."""
    trades = get_trades()
    if not trades:
        return ":ledger: *Trade Journal* — no closed trades yet."

    recent = trades[-n:][::-1]
    lines = [f":ledger: *Trade Journal — Last {len(recent)} Trades*", ""]

    for t in recent:
        icon = ":white_check_mark:" if t["pnl"] > 0 else ":x:"
        strat = t["strategy_clean"]
        hold = f"{t['hold_days']}d" if t["hold_days"] is not None else "?"
        ror = f"{t['return_on_risk']*100:+.1f}%" if t.get("return_on_risk") else ""

        short_s = t.get("short_strike") or t.get("put_short") or "?"
        long_s = t.get("long_strike") or t.get("put_long") or "?"
        lines.append(
            f"{icon} *{t.get('close_date', '?')}*  `{strat}`  "
            f"${short_s}/{long_s}  {hold}  "
            f"P&L ${t['pnl']:+,.2f}  {ror}"
        )

    total = sum(t["pnl"] for t in trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    lines += [
        "",
        f"_All time: {len(trades)} trades  |  {wins}W/{len(trades)-wins}L  |  "
        f"${total:+,.2f} total P&L_",
    ]
    return "\n".join(lines)


def fmt_attribution() -> str:
    """Format per-strategy attribution for Slack."""
    summary = monthly_summary()
    mt = summary["month_totals"]
    at = summary["all_time_totals"]

    lines = [
        f":bar_chart: *Strategy Attribution — {summary['month']}*",
        "",
    ]

    # Month section
    if mt["trades"] == 0:
        lines.append("_No trades this month._")
    else:
        lines.append(
            f"*This month:* {mt['trades']} trades  |  "
            f"{mt['wins']}W/{mt['losses']}L ({mt['win_rate']*100:.0f}%)  |  "
            f"${mt['pnl']:+,.2f}"
        )
        for row in summary["month_by_strat"]:
            pct = f"{row['contribution']*100:+.0f}%" if row["contribution"] else ""
            lines.append(
                f"  `{row['strategy_clean']:24s}`  "
                f"{row['trades']}t  {row['wins']}W/{row['losses']}L  "
                f"${row['total_pnl']:+,.2f}  avg ${row['avg_pnl']:+,.2f}  "
                f"{row['avg_hold_days']:.0f}d avg hold  {pct}"
            )

    lines.append("")

    # All-time section
    if at["trades"] > 0:
        lines.append(
            f"*All time:* {at['trades']} trades  |  "
            f"{at['wins']}W/{at['losses']}L ({at['win_rate']*100:.0f}%)  |  "
            f"${at['pnl']:+,.2f}"
        )
        for row in summary["all_time_by_strat"]:
            pct = f"{row['contribution']*100:+.0f}%" if row["contribution"] else ""
            lines.append(
                f"  `{row['strategy_clean']:24s}`  "
                f"{row['trades']}t  {row['wins']}W/{row['losses']}L  "
                f"${row['total_pnl']:+,.2f}  avg ${row['avg_pnl']:+,.2f}  "
                f"best ${row['best_trade']:+,.2f}  worst ${row['worst_trade']:+,.2f}  {pct}"
            )

    return "\n".join(lines)
