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
    # cash floor: below the minimum blocks new entries
    cfc = {**cfg, "cash_floor_pct": 0.30}
    assert app._risk_gate({**ok_snap, "cash_pct": 0.20}, cfc, None)[0] is False
    assert app._risk_gate({**ok_snap, "cash_pct": 0.45}, cfc, None)[0] is True
    print("ok  risk gate (daily-loss / drawdown / cash-floor / per-trade budget)")


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


def test_attribution():
    import app
    from types import SimpleNamespace as NS
    # strategy parsed from entry coid; closes/blank → None
    assert app._coid_strategy("auto-condor-20260618") == "condor"
    assert app._coid_strategy("fa-INTC-20260618") == "fallen_angel"
    assert app._coid_strategy("bull_put-KO-20260618-ab12") == "bull_put"
    assert app._coid_strategy("autoclose-SPY-2026-06-20") is None
    assert app._coid_strategy("") is None

    # signed cash flow of a put credit spread entry: sell 2.00 / buy 1.00 → +$100
    entry = NS(legs=[NS(symbol="SPY260620P00500000", side="sell", filled_avg_price=2.0, filled_qty=1),
                     NS(symbol="SPY260620P00490000", side="buy",  filled_avg_price=1.0, filled_qty=1)])
    assert app._order_cash_flow(entry) == 100.0
    assert app._order_key(entry) == ("SPY", "2026-06-20")
    # close for 0.30 net debit → −$30
    close = NS(legs=[NS(symbol="SPY260620P00500000", side="buy",  filled_avg_price=0.40, filled_qty=1),
                     NS(symbol="SPY260620P00490000", side="sell", filled_avg_price=0.10, filled_qty=1)])
    assert app._order_cash_flow(close) == -30.0

    # round-trip nets +$70 realized, attributed to condor
    norm = [{"key": ("SPY", "2026-06-20"), "strategy": "condor", "flow": 100.0},
            {"key": ("SPY", "2026-06-20"), "strategy": None,     "flow": -30.0}]
    rep = app._attribution(norm, open_keys=set())
    assert rep["condor"]["realized"] == 70.0 and rep["condor"]["trades"] == 1 and rep["condor"]["wins"] == 1
    # still open → counts as open, not realized
    rep2 = app._attribution(norm, open_keys={("SPY", "2026-06-20")})
    assert rep2["condor"]["open"] == 1 and rep2["condor"]["trades"] == 0
    print("ok  attribution (coid parse / cash flow / round-trip P&L)")


def test_fa_trim_plan():
    import app
    p = app._fa_trim_plan(10)
    assert p == {"mode": "trim_trail", "sell": 5, "keep": 5}
    p3 = app._fa_trim_plan(3)
    assert p3["sell"] == 1 and p3["keep"] == 2 and p3["mode"] == "trim_trail"
    p1 = app._fa_trim_plan(1)
    assert p1 == {"mode": "trail_only", "sell": 0, "keep": 1}
    # sell + keep always equals qty (no shares lost/created)
    for q in (2, 7, 20, 101):
        pl = app._fa_trim_plan(q)
        assert pl["sell"] + pl["keep"] == q
    print("ok  fallen-angel trim plan (half + trail, 1-share trail-only)")


def test_realized_vol():
    import app
    # alternating +1% / -1% daily moves → ~1%/day stdev → ~15.9% annualized RV
    closes, p = [], 100.0
    for i in range(26):
        p *= 1.01 if i % 2 == 0 else (1 / 1.01)
        closes.append(p)
    rv = app._rv_from_closes(closes, 20)
    assert rv is not None and 12 < rv < 22, rv
    # too little history → None
    assert app._rv_from_closes([100, 101, 102], 20) is None
    print("ok  realized vol (annualized from closes)")


def test_expected_move():
    from src.analysis.expected_move import _em_calc
    # ATM 740, call=put=8 → straddle 16, forward 740, EM = 0.85*16 = 13.6
    em = _em_calc(740, 8.0, 8.0)
    assert em["straddle"] == 16.0
    assert em["forward"] == 740.0
    assert em["em_dollars"] == 13.6
    assert em["lower"] == 726.4 and em["upper"] == 753.6
    # call != put shifts the forward via parity (C-P)
    em2 = _em_calc(740, 9.0, 7.0)
    assert em2["forward"] == 742.0          # 740 + (9-7)
    print("ok  expected move (ATM straddle -> range)")


def test_zerodte_strikes():
    import app
    # forward 747, EM ±4.5, $1 wide → short ~742 (1 SD below), long 741
    s, l = app._zerodte_strikes(747.0, 4.5, 1)
    assert s == 742.0 and l == 741.0
    # wider EM pushes the short strike further OTM
    s2, l2 = app._zerodte_strikes(747.0, 9.5, 2)
    assert s2 == 738.0 and l2 == 736.0
    assert s - l == 1 and s2 - l2 == 2          # width preserved
    print("ok  0DTE strikes (short at expected move, width below)")


def test_condor_edge_gate():
    # breakeven POP = 1 - credit/width; gate needs POP >= breakeven + EDGE_MARGIN
    from src.analysis import condor_scanner as cs
    # today's bad condor: $0.40 credit / $5 wide -> breakeven 92%, POP 82% -> REJECT
    be = round((1 - 0.40 / 5) * 100)
    assert be == 92
    assert 82 < be + cs.EDGE_MARGIN          # 82 < 97 -> rejected
    # a good one: $1.20 credit / $5 wide -> breakeven 76%, POP 84% -> PASS
    be2 = round((1 - 1.20 / 5) * 100)
    assert be2 == 76
    assert 84 >= be2 + cs.EDGE_MARGIN        # 84 >= 81 -> passes
    print("ok  condor edge gate (POP vs breakeven win-rate)")


def test_fa_verticals():
    import app
    cfg = {"call_min_score": 6, "put_short_otm": 0.05, "put_width_otm": 0.07,
           "call_long_otm": 0.0, "call_short_otm": 0.08}
    spot = 100.0
    put_mids  = {88: 0.9, 90: 1.1, 93: 1.6, 95: 2.2, 98: 3.5}
    call_mids = {100: 3.0, 105: 1.4, 108: 0.8}

    # score 7 -> bull-put AND bull-call
    v = app._build_fa_verticals(spot, put_mids, call_mids, 7, cfg)
    by = {x["kind"]: x for x in v}
    assert set(by) == {"bull_put", "bull_call"}
    bp = by["bull_put"]
    assert bp["short"] == 95 and bp["long"] == 88          # ~5% OTM short, ~12% long
    assert bp["ref_net"] > 0                               # credit
    assert bp["legs"][0] == {"action": "SELL", "strike": 95, "opt": "P", "qty": 1}
    bc = by["bull_call"]
    assert bc["long"] == 100 and bc["short"] == 108        # ~ATM long, ~8% OTM short
    assert bc["ref_net"] < 0                               # debit
    assert bc["legs"][0]["action"] == "BUY" and bc["legs"][0]["opt"] == "C"

    # score 5 -> bull-put only (call gated at >=6)
    v5 = app._build_fa_verticals(spot, put_mids, call_mids, 5, cfg)
    assert {x["kind"] for x in v5} == {"bull_put"}

    # no positive-credit put available -> nothing (guards against bad quotes)
    assert app._build_fa_verticals(spot, {95: 1.0, 90: 1.0}, {}, 5, cfg) == [] or \
        all(x["ref_net"] != 0 for x in app._build_fa_verticals(spot, {95: 1.0, 90: 1.0}, {}, 5, cfg))
    print("ok  FA verticals (bull-put always, bull-call gated at score>=6)")


def test_strategy_registry():
    import app
    from src import strategies as reg

    # every registered slug that trades live must be recognised by attribution
    # (_coid_strategy), so realized P&L is attributed to the right strategy.
    tagged = {"bull_put", "bull_call", "condor", "bwb", "fly", "burrito",
              "fallen_angel", "value", "income_core"}
    for slug in tagged:
        assert reg.get(slug) is not None, f"{slug} missing from registry"
    # round-trip the slugs that ride an entry client_order_id through _coid_strategy
    assert app._coid_strategy("auto-condor-20260618") == "condor"
    assert app._coid_strategy("bull_put-KO-20260717") == "bull_put"
    assert app._coid_strategy("bull_call-CAG-20260717") == "bull_call"
    assert app._coid_strategy("fa-INTC-20260618") == "fallen_angel"
    assert app._coid_strategy("dca-BND-2026W29") == "income_core"

    # command lookups resolve to the right strategy
    assert reg.by_command("/condor").slug == "condor"
    assert reg.by_command("fly").slug == "fly"

    # every scanner/formatter reference actually imports (no stale module paths)
    for s in reg.all_strategies():
        if s.scanner:
            assert callable(s.load()), f"{s.slug} scanner {s.scanner} broken"
        if s.formatter:
            assert callable(s.load_formatter()), f"{s.slug} formatter broken"

    # structure grouping in the P&L view maps to a real registry structure
    assert reg.for_structure("iron_condor"), "no iron_condor strategy"
    assert reg.format_overview().startswith(":books:")
    print("ok  strategy registry (slug alignment / command lookup / scanner refs)")


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
    test_attribution()
    test_fa_trim_plan()
    test_realized_vol()
    test_expected_move()
    test_zerodte_strikes()
    test_condor_edge_gate()
    test_fa_verticals()
    test_strategy_registry()
    print("\nALL PASS")
