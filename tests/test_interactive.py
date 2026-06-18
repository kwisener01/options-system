"""
Unit tests for the interactive-approval + news plumbing.

Covers only pure logic — no Alpaca/Slack network calls:
  - pending_store round-trip, TTL/expiry, status transitions
  - slack_blocks Block Kit shape
  - news lexical sentiment
  - app._drifted tolerance gate and _legs_to_occ symbol building

Run: python tests/test_interactive.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_pending_store(tmp_path_factory=None):
    from src.notifications import pending_store
    # redirect the store to a temp file so we never touch real data/
    pending_store._STORE_PATH = os.path.join(
        os.path.dirname(__file__), "_tmp_pending_test.json")
    if os.path.exists(pending_store._STORE_PATH):
        os.remove(pending_store._STORE_PATH)

    tid = pending_store.add({"kind": "entry", "label": "x"}, ttl=1000)
    assert pending_store.get(tid)["status"] == "pending"
    ok, reason = pending_store.is_actionable(tid)
    assert ok and reason == "ok"

    pending_store.update_status(tid, "taken", order_id="abc")
    assert pending_store.get(tid)["order_id"] == "abc"
    ok, reason = pending_store.is_actionable(tid)
    assert not ok and reason == "already_taken"

    # expiry path
    tid2 = pending_store.add({"kind": "entry"}, ttl=-1)   # already expired
    ok, reason = pending_store.is_actionable(tid2)
    assert not ok and reason == "expired"
    assert pending_store.get(tid2)["status"] == "expired"

    os.remove(pending_store._STORE_PATH)
    print("ok  pending_store")


def test_blocks():
    from src.notifications import slack_blocks as sb
    cands = [{"tid": "a1", "text": "Trade A", "label": "A"},
             {"tid": "b2", "text": "Trade B", "label": "B"}]
    blocks, fallback = sb.entry_blocks(":x: head", cands)
    # one header + (divider+section+actions) per candidate
    assert blocks[0]["type"] == "section"
    actions = [b for b in blocks if b["type"] == "actions"]
    assert len(actions) == 2
    btn_ids = {e["action_id"] for a in actions for e in a["elements"]}
    assert btn_ids == {sb.ACTION_TAKE, sb.ACTION_SKIP}
    assert actions[0]["elements"][0]["value"] == "a1"
    assert "1. A" in fallback and "2. B" in fallback

    eb, _ = sb.exit_blocks(":x: head", cands)
    eids = {e["action_id"] for b in eb if b["type"] == "actions" for e in b["elements"]}
    assert eids == {sb.ACTION_CLOSE, sb.ACTION_HOLD}
    print("ok  slack_blocks")


def test_news_sentiment():
    from src.live import news
    assert news._score("Company beats earnings, shares surge") == "positive"
    assert news._score("Stock plunges on lawsuit and downgrade") == "negative"
    assert news._score("Company holds annual meeting") == "neutral"
    print("ok  news sentiment")


def test_app_logic():
    import app
    # drift gate
    assert app._drifted(0.50, 0.45, 0.20) is False       # credit within tol
    assert app._drifted(0.50, 0.30, 0.20) is True         # credit collapsed
    assert app._drifted(0.50, -0.10, 0.20) is True        # credit -> debit
    assert app._drifted(-1.00, -1.10, 0.20) is False      # debit within tol
    assert app._drifted(-1.00, -1.40, 0.20) is True       # debit too rich

    # OCC leg building
    legs = app._legs_to_occ("KO", "2026-07-17",
                            [{"action": "SELL", "strike": 82.5, "opt": "P", "qty": 1},
                             {"action": "BUY",  "strike": 80,   "opt": "P", "qty": 1}])
    assert legs[0]["side"] == "SELL"
    assert legs[0]["symbol"].startswith("KO") and "P" in legs[0]["symbol"]
    assert legs[1]["symbol"].endswith("00080000")          # 80.000 strike * 1000
    print("ok  app logic (_drifted / _legs_to_occ)")


def test_auto_trade_config():
    import os
    import app
    base = app._auto_trade_config()
    assert base["max_loss"] == 100 and base["enabled"] is True
    assert base["armed_date"]  # a date string is set

    # env overrides
    os.environ["AUTO_TRADE_DATE"] = "2030-01-02"
    os.environ["AUTO_TRADE_MAX_LOSS"] = "250"
    cfg = app._auto_trade_config()
    assert cfg["armed_date"] == "2030-01-02"
    assert cfg["max_loss"] == 250.0

    # kill switch disables regardless
    os.environ["AUTO_TRADE_KILL"] = "1"
    assert app._auto_trade_config()["enabled"] is False

    for k in ("AUTO_TRADE_DATE", "AUTO_TRADE_MAX_LOSS", "AUTO_TRADE_KILL"):
        os.environ.pop(k, None)
    print("ok  auto-trade config (file + env overrides + kill switch)")


def test_risk_gate():
    import app
    cfg = {"enabled": True, "risk_pct_per_trade": 0.05,
           "daily_loss_limit_pct": 0.05, "max_drawdown_pct": 0.15}
    ok_snap = {"equity": 2000, "day_pl_pct": -0.01, "drawdown_pct": 0.02}
    assert app._risk_gate(ok_snap, cfg, None)[0] is True
    # daily loss breaker
    assert app._risk_gate({**ok_snap, "day_pl_pct": -0.06}, cfg, None)[0] is False
    # drawdown breaker
    assert app._risk_gate({**ok_snap, "drawdown_pct": 0.20}, cfg, None)[0] is False
    # per-trade size: budget = 5% * 2000 = 100
    assert app._risk_gate(ok_snap, cfg, 100)[0] is True
    assert app._risk_gate(ok_snap, cfg, 101)[0] is False
    # disabled → always allowed
    assert app._risk_gate({**ok_snap, "day_pl_pct": -0.5}, {**cfg, "enabled": False}, None)[0] is True
    print("ok  risk gate (daily-loss / drawdown / per-trade budget)")


def test_performance_metrics():
    import app
    # steady 1%/day climb over 10 days
    eq = [1000 * (1.01 ** i) for i in range(11)]
    ts = [i * 86400 for i in range(11)]
    m = app._perf_from_series(eq, ts)
    assert m["points"] == 11
    assert abs(m["total_return"] - (eq[-1] / eq[0] - 1)) < 1e-9
    assert m["max_dd"] == 0.0                      # monotonic up → no drawdown
    assert m["day_win_rate"] == 1.0                # every day green
    assert m["profit_factor"] == float("inf")      # no losing days
    assert m["best_day"] > 0 and m["worst_day"] > 0

    # a drawdown in the middle is detected
    eq2 = [100, 110, 121, 90, 99]                  # peak 121 → trough 90
    m2 = app._perf_from_series(eq2)
    assert abs(m2["max_dd"] - (121 - 90) / 121) < 1e-9
    assert 0 < m2["day_win_rate"] < 1

    # too little history → safe empty
    assert app._perf_from_series([1000])["points"] == 1
    print("ok  performance metrics (returns / drawdown / win rate)")


def test_fa_sizing():
    import app
    cfg = app._fa_config()
    # entry 30, 52w low 27 → stop = max(30*0.88=26.40, 27*0.99=26.73) = 26.73
    s = app._fa_stop_and_size(30.0, 27.0, 2000, cfg)
    assert abs(s["stop"] - 26.73) < 0.01
    # budget = 1% * 2000 = $20; risk/share = 3.27 → 6 shares
    assert s["shares"] == 6 and s["capped_by"] == "risk"
    # tiny risk/share → capped by position notional (15% of 2000 / 10 = 30 sh)
    s2 = app._fa_stop_and_size(10.0, 9.9, 2000, cfg)
    assert s2["shares"] == 30 and s2["capped_by"] == "position_pct"
    # 52w low above entry → stop >= entry → no trade
    s3 = app._fa_stop_and_size(10.0, 11.0, 2000, cfg)
    assert s3["shares"] == 0 and s3["capped_by"] == "bad_stop"
    print("ok  fallen-angel sizing (stop / risk-based shares / caps)")


def test_vix_derisk():
    import app
    cfg = {"vix_half_size": 18, "vix_stand_down": 25}
    assert app._vix_size_factor(12, cfg) == 1.0      # calm → full
    assert app._vix_size_factor(18, cfg) == 0.5      # band → half
    assert app._vix_size_factor(22, cfg) == 0.5
    assert app._vix_size_factor(25, cfg) == 0.0      # crisis → stand down
    assert app._vix_size_factor(30, cfg) == 0.0
    assert app._vix_size_factor(None, cfg) == 1.0    # no data → full
    # risk gate stands down on crisis VIX
    rc = {"enabled": True, "risk_pct_per_trade": 0.05, "daily_loss_limit_pct": 0.05,
          "max_drawdown_pct": 0.15, "vix_stand_down": 25}
    base = {"equity": 2000, "day_pl_pct": -0.01, "drawdown_pct": 0.02}
    assert app._risk_gate({**base, "vix": 30}, rc, None)[0] is False
    assert app._risk_gate({**base, "vix": 15}, rc, None)[0] is True
    print("ok  vix de-risk (size factor + crisis stand-down)")


if __name__ == "__main__":
    test_pending_store()
    test_blocks()
    test_news_sentiment()
    test_app_logic()
    test_auto_trade_config()
    test_risk_gate()
    test_performance_metrics()
    test_fa_sizing()
    test_vix_derisk()
    print("\nALL PASS")
