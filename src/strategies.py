"""Single source of truth for every trading strategy this bot runs.

Read this first to get up to speed: it maps each strategy's canonical slug to
its structure, how it's triggered (Slack command / cron route / scheduled job),
its scanner entry point, and its config file. Nothing else in the codebase holds
the full list — the scanners live in ``src/analysis/``, the live wiring in
``app.py``, and the backtests in ``run_*.py``; this file ties them together.

The ``slug`` is the canonical id used everywhere P&L is tracked: it matches the
strategy tag in entry ``client_order_id``s (see ``app._coid_strategy``) and the
first arg to ``app.register_entry``. Keep those in sync with this file.

Structure vocabulary (also what the grouped-P&L view derives from leg counts in
``app._structure_name``):
    single / vertical (2) / butterfly (3) / bwb (3, broken wings) /
    condor · iron_condor (4) / double_bwb (6) / calendar / equity

Usage::

    from src.strategies import STRATEGIES, get, load_scanner
    get("condor").name                 # "Iron Condor"
    scan = load_scanner("condor")      # lazily imports condor_scanner.scan
"""
from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Callable, Optional


@dataclass(frozen=True)
class Strategy:
    slug: str                       # canonical id — matches order-id tags + register_entry
    name: str                       # display name
    structure: str                  # single|vertical|butterfly|bwb|condor|iron_condor|double_bwb|calendar|equity
    legs: int                       # nominal option legs (0 = equity / variable)
    kind: str                       # credit | debit | equity | mixed
    summary: str                    # one-line description
    scanner: Optional[str] = None   # "module:function" that produces candidates
    formatter: Optional[str] = None # "module:function" that renders to Slack
    command: Optional[str] = None   # Slack slash command that runs it on demand
    cron: Optional[str] = None      # cron route / scheduled job that runs it
    config: Optional[str] = None    # config/<file>.json, if parameterised
    status: str = "active"          # active | off | manual (backtest/one-off only)

    def load(self) -> Optional[Callable]:
        """Import and return the scanner callable, or None if not importable."""
        return _resolve(self.scanner)

    def load_formatter(self) -> Optional[Callable]:
        return _resolve(self.formatter)


def _resolve(ref: Optional[str]) -> Optional[Callable]:
    if not ref:
        return None
    mod, _, fn = ref.partition(":")
    return getattr(import_module(mod), fn)


# ── The catalog ───────────────────────────────────────────────────────────────
# Ordered roughly premium-selling options → directional options → equity sleeves.

_ALL: list[Strategy] = [
    Strategy(
        slug="bull_put", name="Bull Put Credit Spread", structure="vertical",
        legs=2, kind="credit",
        summary="Sell an OTM put spread on strong tickers for premium (defined risk).",
        scanner="src.analysis.bull_put_scanner:scan",
        formatter="src.analysis.bull_put_scanner:fmt_slack",
        command="/scan", cron="/api/bull-put/scan"),
    Strategy(
        slug="condor", name="Iron Condor", structure="iron_condor",
        legs=4, kind="credit",
        summary="GEX-anchored put+call spreads between the walls — high-POP premium play.",
        scanner="src.analysis.condor_scanner:scan",
        formatter="src.analysis.condor_scanner:fmt_slack",
        command="/condor"),
    Strategy(
        slug="fly", name="GEX-Pinned Butterfly", structure="butterfly",
        legs=3, kind="debit",
        summary="Debit butterfly pinned to the gamma wall in positive-gamma regimes.",
        scanner="src.analysis.butterfly_scanner:scan",
        formatter="src.analysis.butterfly_scanner:fmt_slack",
        command="/fly"),
    Strategy(
        slug="bwb", name="Broken-Wing Butterfly", structure="bwb",
        legs=3, kind="mixed",
        summary="Butterfly with unequal wings — skewed, often for a small credit.",
        scanner="src.analysis.bwb_scanner:scan",
        formatter="src.analysis.bwb_scanner:fmt_slack",
        cron="/api/bwb/scan"),
    Strategy(
        slug="batman", name="XSP Batman (Double BWB)", structure="double_bwb",
        legs=6, kind="credit",
        summary="Two broken-wing butterflies straddling spot on XSP (positive-cowl payoff).",
        scanner="src.analysis.batman_scanner:scan",
        formatter="src.analysis.batman_scanner:fmt_slack",
        command="/batman"),
    Strategy(
        slug="burrito", name="SPY Burrito Butterfly", structure="butterfly",
        legs=3, kind="debit",
        summary="Boomer Dan 'Burrito Butterfly' on SPY run as a 4-9 DTE swing.",
        scanner="app:_burrito_propose",
        cron="/cron/burrito", config="burrito.json"),
    Strategy(
        slug="double_calendar", name="Double Calendar", structure="calendar",
        legs=4, kind="debit",
        summary="Long-dated vs short-dated straddle legs — sells near-term theta.",
        scanner="src.analysis.double_calendar_scanner:scan",
        status="manual"),
    Strategy(
        slug="xsp_0dte_bwb", name="XSP 0DTE BWB", structure="bwb",
        legs=3, kind="credit",
        summary="Same-day broken-wing butterfly on XSP anchored to the expected move.",
        scanner="src.analysis.xsp_0dte_bwb:scan_xsp_0dte",
        status="manual"),
    Strategy(
        slug="zerodte", name="0DTE Put Credit Spread", structure="vertical",
        legs=2, kind="credit",
        summary="SPY put spread at the downside expected move, closed intraday.",
        cron="_zerodte_job", config="zerodte.json", status="off"),
    Strategy(
        slug="swing", name="GEX Swing", structure="single",
        legs=1, kind="debit",
        summary="Directional single-leg option (or stock fallback) from per-ticker GEX signals.",
        scanner="swing_scanner:scan_ticker",
        cron="/cron/swing"),
    Strategy(
        slug="fallen_angel", name="Fallen-Angel Equity", structure="equity",
        legs=0, kind="equity",
        summary="Buy beaten-down high-conviction stocks near 52w lows with a hard stop.",
        scanner="src.analysis.fallen_angel_scanner:scan_fallen_angels",
        formatter="src.analysis.fallen_angel_scanner:fmt_slack",
        cron="/api/fallen-angels", config="fallen_angel.json"),
    Strategy(
        slug="value", name="Value Watchlist", structure="equity",
        legs=0, kind="equity",
        summary="Rotation into oversold value names (RSI / 200-DMA / fundamentals).",
        scanner="src.analysis.value_watchlist:scan_watchlist",
        formatter="src.analysis.value_watchlist:fmt_slack"),
    Strategy(
        slug="income_core", name="Income-Core DCA", structure="equity",
        legs=0, kind="equity",
        summary="Weekly dollar-cost-average buys across the income/ETF sleeve.",
        scanner="app:_dca_run",
        cron="/cron/dca", config="dca.json"),
]

STRATEGIES: dict[str, Strategy] = {s.slug: s for s in _ALL}


# ── Lookups ────────────────────────────────────────────────────────────────────

def get(slug: str) -> Optional[Strategy]:
    """Strategy by canonical slug (as stored in order-id tags), or None."""
    return STRATEGIES.get(slug)


def all_strategies() -> list[Strategy]:
    return list(_ALL)


def by_command(command: str) -> Optional[Strategy]:
    """Strategy for a Slack slash command, e.g. '/condor' -> Iron Condor."""
    command = command if command.startswith("/") else f"/{command}"
    return next((s for s in _ALL if s.command == command), None)


def for_structure(structure: str) -> list[Strategy]:
    """Strategies whose leg structure matches (as classified in the P&L view)."""
    return [s for s in _ALL if s.structure == structure]


def load_scanner(slug: str) -> Optional[Callable]:
    """Lazily import and return a strategy's scanner callable."""
    s = get(slug)
    return s.load() if s else None


# ── Rendering ───────────────────────────────────────────────────────────────────

def format_overview() -> str:
    """Slack-formatted catalog of every strategy — used by the `/strategies` command."""
    icon = {"active": ":green_circle:", "off": ":white_circle:", "manual": ":wrench:"}
    lines = [":books: *Strategy Registry* — the bot's full playbook\n"]
    for s in _ALL:
        trig = s.command or s.cron or "—"
        shape = f"{s.structure}/{s.legs}-leg, {s.kind}" if s.legs else s.kind
        lines.append(
            f"{icon.get(s.status, ':grey_question:')} *{s.name}* "
            f"(`{s.slug}`, {shape})\n"
            f"   {s.summary}\n"
            f"   _trigger:_ `{trig}`" + (f"  ·  _config:_ `config/{s.config}`" if s.config else "")
        )
    lines.append(
        "\n_:green_circle: active · :white_circle: off (config) · :wrench: manual/backtest only_"
    )
    return "\n".join(lines)
