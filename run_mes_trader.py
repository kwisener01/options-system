"""
MES/ES day trading CLI for MyFundedFutures challenge.

Commands:
    python run_mes_trader.py plan                          # morning battle plan + Slack
    python run_mes_trader.py signal                       # current signal + Slack if active
    python run_mes_trader.py monitor                      # loop every 15 min, Slack on signals
    python run_mes_trader.py status                       # MFF challenge P&L status
    python run_mes_trader.py fill LONG 5285.50 2          # record fill
    python run_mes_trader.py exit LONG 5285.50 5290.00 2  # record exit + Slack P&L alert
    python run_mes_trader.py eod                          # end-of-day mark
    python run_mes_trader.py reset                        # reset challenge state

Options:
    --account  10k|25k|50k|100k|150k|250k  (default: 50k)
    --no-slack                              # skip Slack notifications
"""
import argparse
import os
import sys
import time as _time

sys.path.insert(0, os.path.dirname(__file__))

from src.logger import setup_logging
from src.analysis.gex_scanner import scan as gex_scan
from src.live.leading_indicators import (
    fetch_cross_asset_levels, compute_direction_score,
    find_level_confluences, format_cross_asset_plain,
)
from src.live.mes_day_trader import (
    MESTradeManager, MFF_ACCOUNTS, generate_mes_signal,
    format_morning_plan, format_slack_signal, format_slack_exit,
    format_slack_morning,
)
from src.notifications.slack_notifier import send_message


def _build_signal(manager: MESTradeManager):
    """Fetch live data and return (gex, cross, score, signal)."""
    print("Scanning GEX levels...")
    gex = gex_scan()
    if gex is None:
        print("GEX scan failed — market may be closed or chain unavailable.")
        return None, None, None, None

    print("Fetching cross-asset levels...")
    cross = fetch_cross_asset_levels()
    if cross is None:
        print("Cross-asset fetch failed.")
        return None, None, None, None

    score = compute_direction_score(cross, gex.gex_regime, gex.vanna_signal)

    from src.analysis.futures_levels import _to_es
    es_spot = _to_es(gex.spot)
    gex_pairs = [
        (_to_es(gex.gamma_wall), "GAMMA_WALL"),
        (_to_es(gex.call_wall),  "CALL_WALL"),
        (_to_es(gex.put_wall),   "PUT_WALL"),
        (_to_es(gex.flip_level), "FLIP_LEVEL"),
    ]
    confluences = find_level_confluences(es_spot, gex_pairs, cross)
    signal = generate_mes_signal(gex, cross, score, manager.rules, manager.state, confluences)
    return gex, cross, score, signal


def cmd_plan(manager: MESTradeManager, slack: bool = True):
    gex, cross, score, signal = _build_signal(manager)
    if gex is None:
        return
    print()
    print(format_morning_plan(gex, cross, score, signal, manager))
    if slack:
        msg = format_slack_morning(gex, cross, score, signal, manager)
        sent = send_message(msg)
        print(f"\n[Slack] {'sent' if sent else 'FAILED (check SLACK_WEBHOOK_URL)'}")


def cmd_signal(manager: MESTradeManager, slack: bool = True):
    gex, cross, score, signal = _build_signal(manager)
    if gex is None:
        return

    print()
    print("=" * 55)
    print("  MES TRADE SIGNAL")
    print("=" * 55)
    print(f"  Direction  : {signal.direction}")
    print(f"  Setup      : {signal.setup or '--'}")
    print(f"  Confidence : {signal.confidence}")

    if signal.direction != "NO_TRADE":
        print(f"  Entry      : /ES {signal.es_entry:.2f}")
        print(f"  Stop       : /ES {signal.es_stop:.2f}  ({signal.stop_pts:.1f} pts)")
        print(f"  Target     : /ES {signal.es_target:.2f}  ({signal.target_pts:.1f} pts  R:R {signal.rr:.1f}x)")
        print(f"  Contracts  : {signal.mes_contracts} MES  (${signal.mes_risk:,.0f} risk)")
        print(f"  Reason     : {signal.reason}")
        if signal.confluences:
            print(f"  Confluences:")
            for c in signal.confluences[:3]:
                print(f"    ES {c.price_zone:.2f}  [{c.distance_pts:+.1f}pts]  "
                      f"{c.strength}x: {', '.join(c.levels)}")
        if slack:
            msg = format_slack_signal(signal, gex, cross, score)
            sent = send_message(msg)
            print(f"\n[Slack] signal sent: {'yes' if sent else 'FAILED'}")
    else:
        print(f"  Reason     : {signal.block_reason or signal.reason}")

    print()
    print(format_cross_asset_plain(cross, score))
    print()


def cmd_monitor(manager: MESTradeManager, interval_min: int = 15, slack: bool = True):
    """
    Poll every interval_min.  When a fresh LONG/SHORT signal fires (not seen
    last poll), send it to Slack.  Continues until Ctrl-C.
    """
    import pytz
    from datetime import datetime, time as _t

    ET = pytz.timezone("America/New_York")
    last_setup = ""
    print(f"Monitoring every {interval_min} min.  Press Ctrl-C to stop.")

    while True:
        now = datetime.now(ET)
        # Only run during market hours (9:30 AM – 4:00 PM ET)
        if not (_t(9, 25) <= now.time() <= _t(16, 5)):
            print(f"[{now.strftime('%H:%M')}] Market closed — sleeping 10 min")
            _time.sleep(600)
            continue

        gex, cross, score, signal = _build_signal(manager)
        ts = now.strftime("%H:%M")

        if gex is None:
            print(f"[{ts}] Scan failed — retry in {interval_min} min")
        elif signal.direction != "NO_TRADE":
            # Only alert if it's a new setup (direction or setup name changed)
            key = f"{signal.direction}:{signal.setup}"
            if key != last_setup:
                print(f"[{ts}] NEW SIGNAL: {signal.direction} {signal.setup}  "
                      f"entry={signal.es_entry:.2f}  stop={signal.es_stop:.2f}  "
                      f"target={signal.es_target:.2f}  R:R={signal.rr:.1f}x")
                if slack:
                    msg = format_slack_signal(signal, gex, cross, score)
                    sent = send_message(msg)
                    print(f"[{ts}] Slack: {'sent' if sent else 'FAILED'}")
                last_setup = key
            else:
                print(f"[{ts}] Signal unchanged: {signal.direction} {signal.setup}")
        else:
            if last_setup:
                last_setup = ""   # reset so next signal fires fresh
            print(f"[{ts}] No setup — {signal.block_reason or signal.reason}")

        _time.sleep(interval_min * 60)


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="MES day trading for MFF challenge")
    parser.add_argument("command",
                        choices=["plan", "signal", "monitor", "status",
                                 "fill", "exit", "eod", "reset"])
    parser.add_argument("args", nargs="*")
    parser.add_argument("--account", default="50k", choices=list(MFF_ACCOUNTS.keys()),
                        help="MFF account size (default: 50k)")
    parser.add_argument("--no-slack", action="store_true",
                        help="Skip Slack notifications")
    parser.add_argument("--interval", type=int, default=15,
                        help="Monitor poll interval in minutes (default: 15)")
    args = parser.parse_args()

    slack = not args.no_slack
    manager = MESTradeManager(args.account)

    if args.command == "plan":
        cmd_plan(manager, slack)

    elif args.command == "signal":
        cmd_signal(manager, slack)

    elif args.command == "monitor":
        cmd_monitor(manager, args.interval, slack)

    elif args.command == "status":
        print()
        print(manager.format_status())
        print()

    elif args.command == "fill":
        if len(args.args) < 3:
            print("Usage: fill LONG|SHORT <es_entry> <mes_contracts>")
            sys.exit(1)
        direction = args.args[0].upper()
        es_entry  = float(args.args[1])
        contracts = int(args.args[2])
        manager.on_fill(direction, es_entry, contracts)
        print(f"Fill recorded: {direction} {contracts} MES @ /ES {es_entry:.2f}")

    elif args.command == "exit":
        if len(args.args) < 4:
            print("Usage: exit LONG|SHORT <es_entry> <es_exit> <mes_contracts>")
            sys.exit(1)
        direction  = args.args[0].upper()
        es_entry   = float(args.args[1])
        es_exit    = float(args.args[2])
        contracts  = int(args.args[3])
        pnl = manager.on_exit(direction, es_entry, es_exit, contracts)
        sign = "+" if pnl >= 0 else ""
        print(f"Exit: {direction} {contracts} MES  {es_entry:.2f} -> {es_exit:.2f}  = {sign}${pnl:,.2f}")
        print()
        print(manager.format_status())
        print()
        if slack:
            msg = format_slack_exit(direction, es_entry, es_exit, contracts,
                                     pnl, manager.state, manager.rules)
            sent = send_message(msg)
            print(f"[Slack] exit alert: {'sent' if sent else 'FAILED'}")

    elif args.command == "eod":
        manager.end_of_day()
        print("End-of-day recorded.")
        print()
        print(manager.format_status())
        print()
        if slack:
            s = manager.state
            r = manager.rules
            prog = min(s.gross_profit / r.profit_target * 100, 100) if r.profit_target else 0
            eod_msg = (
                f":moon: *MES EOD Summary*\n"
                f">  Daily P&L : ${s.daily_pnl:+,.2f}\n"
                f">  Trades    : {s.trades_today}\n"
                f">  Balance   : ${s.current_balance:,.2f}\n"
                f">  Challenge : ${s.gross_profit:,.2f} / ${r.profit_target:,.0f}  ({prog:.1f}%)  "
                f"Day {s.days_traded}/{r.min_days}"
            )
            send_message(eod_msg)

    elif args.command == "reset":
        answer = input("Reset MFF challenge state? Clears all P&L history. (y/N): ")
        if answer.strip().lower() == "y":
            manager.reset()
            print("Challenge state reset.")
        else:
            print("Cancelled.")


if __name__ == "__main__":
    main()
