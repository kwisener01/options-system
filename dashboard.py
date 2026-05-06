"""
Trading dashboard — simple web UI for controlling the Alpaca trader.
Run: python dashboard.py
Open: http://localhost:5000
"""
import json
import logging
import os
import signal
import subprocess
import sys
import threading

from flask import Flask, jsonify, render_template_string, request

sys.path.insert(0, os.path.dirname(__file__))

from src.logger import setup_logging
from src.live.options_paper_trader import (
    preview_weekly, open_weekly, check_tp, settle_expiry, status as options_status,
)
from src.analysis.macro_scanner import get_macro_snapshot
from src.client.alpaca_client import AlpacaClient
from src.backtest.multi_strategy_engine import MultiStrategyEngine, STRATEGY_IRON_CONDOR, STRATEGY_BULL_CALL_DEBIT

setup_logging()
logger = logging.getLogger(__name__)
app = Flask(__name__)

_scheduler_proc: subprocess.Popen = None
_scheduler_lock = threading.Lock()


# ── scheduler process control ─────────────────────────────────────────────────

def _scheduler_running() -> bool:
    global _scheduler_proc
    if _scheduler_proc is None:
        return False
    return _scheduler_proc.poll() is None


# ── HTML template ─────────────────────────────────────────────────────────────

TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Alpaca Trader Dashboard</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; }

    .header { background: #1a1f2e; border-bottom: 1px solid #2d3748; padding: 16px 32px; display: flex; align-items: center; justify-content: space-between; }
    .header h1 { font-size: 1.25rem; font-weight: 600; color: #fff; letter-spacing: 0.5px; }
    .mode-badge { font-size: 0.75rem; padding: 4px 10px; border-radius: 20px; font-weight: 600; letter-spacing: 1px; }
    .mode-paper { background: #1a3a2e; color: #4ade80; border: 1px solid #166534; }
    .mode-live  { background: #3a1a1a; color: #f87171; border: 1px solid #991b1b; }

    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; padding: 28px 32px; }

    .card { background: #1a1f2e; border: 1px solid #2d3748; border-radius: 12px; padding: 22px; }
    .card h2 { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px; color: #718096; margin-bottom: 16px; }

    .stat { margin-bottom: 12px; }
    .stat .label { font-size: 0.8rem; color: #718096; margin-bottom: 2px; }
    .stat .value { font-size: 1.4rem; font-weight: 700; color: #fff; }
    .stat .value.green { color: #4ade80; }
    .stat .value.red   { color: #f87171; }
    .stat .value.yellow { color: #fbbf24; }
    .stat .value.dim   { font-size: 1rem; color: #a0aec0; }

    .regime-bull  { color: #4ade80; font-weight: 700; }
    .regime-bear  { color: #f87171; font-weight: 700; }
    .regime-neutral { color: #fbbf24; font-weight: 700; }

    .btn { display: inline-block; padding: 10px 20px; border-radius: 8px; font-size: 0.9rem; font-weight: 600; cursor: pointer; border: none; transition: opacity 0.15s, transform 0.1s; width: 100%; margin-bottom: 10px; text-align: center; }
    .btn:hover { opacity: 0.85; }
    .btn:active { transform: scale(0.98); }
    .btn:disabled { opacity: 0.4; cursor: not-allowed; }
    .btn-green  { background: #166534; color: #4ade80; border: 1px solid #15803d; }
    .btn-blue   { background: #1e3a5f; color: #60a5fa; border: 1px solid #1d4ed8; }
    .btn-yellow { background: #3d2c00; color: #fbbf24; border: 1px solid #92400e; }
    .btn-red    { background: #3d1010; color: #f87171; border: 1px solid #991b1b; }
    .btn-gray   { background: #2d3748; color: #a0aec0; border: 1px solid #4a5568; }

    .scheduler-on  { color: #4ade80; }
    .scheduler-off { color: #f87171; }

    .result-box { background: #0f1117; border: 1px solid #2d3748; border-radius: 8px; padding: 16px; margin-top: 16px; font-size: 0.85rem; line-height: 1.7; font-family: 'Cascadia Code', 'Fira Code', monospace; white-space: pre-wrap; word-break: break-word; max-height: 340px; overflow-y: auto; display: none; }
    .result-box.visible { display: block; }

    .divider { border: none; border-top: 1px solid #2d3748; margin: 14px 0; }

    .position-box { background: #12192b; border: 1px solid #2d4a7a; border-radius: 8px; padding: 14px; margin-top: 8px; font-size: 0.85rem; line-height: 1.8; }
    .position-box .leg { color: #60a5fa; font-weight: 600; }
    .position-box .credit { color: #4ade80; }
    .position-box .risk   { color: #f87171; }

    .spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid rgba(255,255,255,0.2); border-top-color: #fff; border-radius: 50%; animation: spin 0.6s linear infinite; margin-right: 6px; vertical-align: middle; }
    @keyframes spin { to { transform: rotate(360deg); } }

    .toast { position: fixed; bottom: 24px; right: 24px; padding: 12px 20px; background: #2d3748; border-radius: 8px; font-size: 0.85rem; color: #e2e8f0; z-index: 9999; transform: translateY(80px); transition: transform 0.3s; opacity: 0; border-left: 3px solid #4ade80; }
    .toast.show { transform: translateY(0); opacity: 1; }
    .toast.error { border-left-color: #f87171; }
  </style>
</head>
<body>

<div class="header">
  <h1>Alpaca Trader</h1>
  <span class="mode-badge {{ 'mode-paper' if is_paper else 'mode-live' }}">
    {{ 'PAPER' if is_paper else 'LIVE' }}
  </span>
</div>

<div class="grid">

  <!-- Scheduler -->
  <div class="card">
    <h2>Scheduler</h2>
    <div class="stat">
      <div class="label">Status</div>
      <div class="value" id="sched-status">Loading...</div>
    </div>
    <hr class="divider">
    <button class="btn btn-green" id="btn-start" onclick="schedAction('start')">Start Scheduler</button>
    <button class="btn btn-red"   id="btn-stop"  onclick="schedAction('stop')">Stop Scheduler</button>
    <div style="font-size:0.75rem; color:#4a5568; margin-top:4px;">
      Runs monitor every 15 min + EOD rebalance at 4:05 PM ET
    </div>
  </div>

  <!-- Account -->
  <div class="card">
    <h2>Account</h2>
    <div class="stat">
      <div class="label">Portfolio Value</div>
      <div class="value" id="acct-value">—</div>
    </div>
    <div class="stat">
      <div class="label">Buying Power</div>
      <div class="value dim" id="acct-bp">—</div>
    </div>
    <div class="stat">
      <div class="label">Positions</div>
      <div class="value dim" id="acct-pos">—</div>
    </div>
    <hr class="divider">
    <button class="btn btn-gray" onclick="loadAccount()">Refresh</button>
  </div>

  <!-- Macro -->
  <div class="card">
    <h2>Market Regime</h2>
    <div class="stat">
      <div class="label">Regime</div>
      <div class="value" id="macro-regime">—</div>
    </div>
    <div class="stat">
      <div class="label">VIX</div>
      <div class="value dim" id="macro-vix">—</div>
    </div>
    <div class="stat">
      <div class="label">SPY Trend</div>
      <div class="value dim" id="macro-trend">—</div>
    </div>
    <div style="font-size:0.8rem; color:#718096; margin-top:8px;" id="macro-summary">—</div>
    <hr class="divider">
    <button class="btn btn-gray" onclick="loadMacro()">Refresh</button>
  </div>

  <!-- Options Paper Trading -->
  <div class="card">
    <h2>Options Paper Trading</h2>
    <div class="stat">
      <div class="label">Paper Account</div>
      <div class="value" id="opt-value">—</div>
    </div>
    <div class="stat">
      <div class="label">Total P&amp;L</div>
      <div class="value dim" id="opt-pnl">—</div>
    </div>
    <div class="stat">
      <div class="label">Record</div>
      <div class="value dim" id="opt-record">—</div>
    </div>
    <div id="opt-position"></div>
    <hr class="divider">
    <button class="btn btn-blue"   onclick="optAction('dry-run')">Dry Run — Preview This Week</button>
    <button class="btn btn-green"  onclick="optAction('open')">Open Position</button>
    <button class="btn btn-yellow" onclick="optAction('check')">Check Take-Profit</button>
    <button class="btn btn-gray"   onclick="optAction('settle')">Settle Expiry</button>
    <div class="result-box" id="opt-result"></div>
  </div>

</div>

<div class="toast" id="toast"></div>

<script>
  function toast(msg, isError) {
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.className = 'toast show' + (isError ? ' error' : '');
    setTimeout(() => t.className = 'toast', 3000);
  }

  async function api(path, method = 'GET', body = null) {
    const opts = { method, headers: { 'Content-Type': 'application/json' } };
    if (body) opts.body = JSON.stringify(body);
    const r = await fetch(path, opts);
    return r.json();
  }

  async function loadAccount() {
    try {
      const d = await api('/api/account');
      document.getElementById('acct-value').textContent = d.portfolio_value ? '$' + d.portfolio_value.toLocaleString('en-US', {minimumFractionDigits: 2}) : '—';
      document.getElementById('acct-bp').textContent    = d.buying_power    ? '$' + d.buying_power.toLocaleString('en-US', {minimumFractionDigits: 2})    : '—';
      document.getElementById('acct-pos').textContent   = d.positions !== undefined ? d.positions + ' open' : '—';
    } catch(e) { toast('Account fetch failed', true); }
  }

  async function loadMacro() {
    try {
      const d = await api('/api/macro');
      const el = document.getElementById('macro-regime');
      el.textContent = d.regime || '—';
      el.className = 'value regime-' + (d.regime || 'neutral').toLowerCase();
      document.getElementById('macro-vix').textContent     = d.vix_level ? d.vix_level.toFixed(2) : '—';
      document.getElementById('macro-trend').textContent   = d.spy_trend || '—';
      document.getElementById('macro-summary').textContent = d.summary   || '';
    } catch(e) { toast('Macro fetch failed', true); }
  }

  async function loadOptionsStatus() {
    try {
      const d = await api('/api/options/status');
      const pnl = d.total_pnl || 0;
      document.getElementById('opt-value').textContent  = '$' + (d.account_value || 0).toLocaleString('en-US', {minimumFractionDigits: 2});
      const pnlEl = document.getElementById('opt-pnl');
      pnlEl.textContent = (pnl >= 0 ? '+' : '') + '$' + pnl.toFixed(2);
      pnlEl.className = 'value dim ' + (pnl >= 0 ? 'green' : 'red');
      document.getElementById('opt-record').textContent = d.n_wins + 'W / ' + d.n_losses + 'L';

      const posEl = document.getElementById('opt-position');
      const pos = d.open_position;
      if (pos) {
        let legs = '';
        if (pos.strategy === 'IRON_CONDOR') {
          legs = `Put ${pos.put_short}/${pos.put_long} + Call ${pos.call_short}/${pos.call_long}`;
        } else {
          legs = `${pos.short_strike} / ${pos.long_strike}`;
        }
        const unreal = d.unrealized_pnl;
        posEl.innerHTML = `<div class="position-box">
          <div><b>${pos.strategy}</b>  exp ${pos.expiry_date}</div>
          <div class="leg">${legs}</div>
          <div class="credit">Credit: $${pos.net_credit.toFixed(3)}/sh × ${pos.contracts} contracts</div>
          <div class="risk">Max risk: $${(pos.max_risk * 100 * pos.contracts).toFixed(2)}</div>
          ${unreal !== null ? '<div style="color:#a0aec0">Unrealized: <span class="' + (unreal >= 0 ? 'credit' : 'risk') + '">' + (unreal >= 0 ? '+' : '') + '$' + unreal.toFixed(2) + '</span></div>' : ''}
        </div>`;
      } else {
        posEl.innerHTML = '<div style="font-size:0.8rem;color:#4a5568;margin-top:8px;">No open position</div>';
      }
    } catch(e) { toast('Options status failed', true); }
  }

  async function loadSchedulerStatus() {
    try {
      const d = await api('/api/scheduler/status');
      const el = document.getElementById('sched-status');
      el.textContent = d.running ? 'Running' : 'Stopped';
      el.className   = 'value ' + (d.running ? 'scheduler-on' : 'scheduler-off');
      document.getElementById('btn-start').disabled = d.running;
      document.getElementById('btn-stop').disabled  = !d.running;
    } catch(e) {}
  }

  async function schedAction(action) {
    const btn = document.getElementById('btn-' + action);
    btn.innerHTML = '<span class="spinner"></span>' + (action === 'start' ? 'Starting...' : 'Stopping...');
    btn.disabled = true;
    try {
      const d = await api('/api/scheduler/' + action, 'POST');
      toast(d.message || 'Done');
    } catch(e) { toast('Error', true); }
    await loadSchedulerStatus();
  }

  async function optAction(action) {
    const resultEl = document.getElementById('opt-result');
    resultEl.textContent = 'Loading...';
    resultEl.classList.add('visible');

    try {
      const d = await api('/api/options/' + action, 'POST');
      if (d.error) {
        resultEl.textContent = 'Error: ' + d.error;
        toast(d.error, true);
      } else {
        resultEl.textContent = d.output || JSON.stringify(d, null, 2);
        toast('Done');
        await loadOptionsStatus();
      }
    } catch(e) {
      resultEl.textContent = 'Request failed';
      toast('Request failed', true);
    }
  }

  // Load everything on page load
  Promise.all([loadAccount(), loadMacro(), loadOptionsStatus(), loadSchedulerStatus()]);
  // Refresh status every 60 seconds
  setInterval(() => Promise.all([loadSchedulerStatus(), loadOptionsStatus()]), 60_000);
</script>
</body>
</html>
"""


# ── API routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    from config.settings import IS_PAPER
    return render_template_string(TEMPLATE, is_paper=IS_PAPER)


@app.route("/api/account")
def api_account():
    try:
        client = AlpacaClient()
        return jsonify({
            "portfolio_value": round(client.get_portfolio_value(), 2),
            "buying_power":    round(client.get_buying_power(), 2),
            "positions":       len(client.get_positions()),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/macro")
def api_macro():
    try:
        m = get_macro_snapshot()
        return jsonify({
            "regime":    m.regime,
            "spy_trend": m.spy_trend,
            "vix_level": round(m.vix_level, 2),
            "vix_signal": m.vix_signal,
            "summary":   m.summary,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/options/status")
def api_options_status():
    try:
        return jsonify(options_status())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/options/dry-run", methods=["POST"])
def api_dry_run():
    try:
        engine = MultiStrategyEngine(
            short_otm_pct  = float(request.json.get("otm",        0.02)) if request.json else 0.02,
            spread_width_pct = float(request.json.get("width",    0.01)) if request.json else 0.01,
            call_otm_pct   = float(request.json.get("otm",        0.02)) if request.json else 0.02,
            max_risk_pct   = float(request.json.get("risk_pct",   0.10)) if request.json else 0.10,
            max_vix_entry  = float(request.json.get("max_vix",   25.0))  if request.json else 25.0,
            low_vol_threshold = float(request.json.get("low_vol", 18.0)) if request.json else 18.0,
            take_profit_pct = float(request.json.get("tp",        0.50)) if request.json else 0.50,
        )
        r = preview_weekly(engine)
        pos = r.get("position")
        lines = [
            f"Strategy : {r['strategy']}",
            f"SPY      : ${r['spot']:.2f}",
            f"VIX      : {r['vix']:.2f}",
            f"Regime   : {r['regime']}  |  Trend: {r['spy_trend']}",
            f"Macro    : {r['macro_summary']}",
        ]
        if pos:
            lines.append("")
            if r["strategy"] == STRATEGY_IRON_CONDOR:
                lines.append(f"Put spread : {pos['put_short']:.1f} / {pos['put_long']:.1f}  (credit ${pos['put_credit']:.3f}/sh)")
                lines.append(f"Call spread: {pos['call_short']:.1f} / {pos['call_long']:.1f}  (credit ${pos['call_credit']:.3f}/sh)")
            elif r["strategy"] == STRATEGY_BULL_CALL_DEBIT:
                lines.append(f"Call debit : {pos['short_strike']:.1f} / {pos['long_strike']:.1f}")
            else:
                lines.append(f"Strikes    : {pos['short_strike']:.1f} / {pos['long_strike']:.1f}")
            nc, c, mr = pos["net_credit"], pos["contracts"], pos["max_risk"]
            lines.append(f"Net credit : ${nc:.3f}/sh × {c} contracts = ${nc*100*c:,.2f}")
            lines.append(f"Max risk   : ${mr*100*c:,.2f}  |  Expiry: {pos['expiry_date']}")
            lines.append(f"Width      : ${pos['spread_width']:.2f}  |  Sigma: {pos['sigma']*100:.1f}%")
            lines.append("")
            lines.append("[DRY RUN — nothing saved, no orders placed]")
        else:
            lines.append("")
            lines.append("No trade — conditions call for CASH this week.")
        return jsonify({"output": "\n".join(lines)})
    except Exception as e:
        logger.exception("dry-run error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/options/open", methods=["POST"])
def api_open():
    try:
        pos = open_weekly()
        if pos is None:
            return jsonify({"output": "No position opened — CASH signal or position already open."})
        nc, c, mr = pos["net_credit"], pos["contracts"], pos["max_risk"]
        if pos["strategy"] == STRATEGY_IRON_CONDOR:
            legs = f"Put {pos['put_short']:.1f}/{pos['put_long']:.1f} + Call {pos['call_short']:.1f}/{pos['call_long']:.1f}"
        elif pos["strategy"] == STRATEGY_BULL_CALL_DEBIT:
            legs = f"Call debit {pos['short_strike']:.1f}/{pos['long_strike']:.1f}"
        else:
            legs = f"{pos['short_strike']:.1f} / {pos['long_strike']:.1f}"
        output = (
            f"Opened: {pos['strategy']}\n"
            f"Strikes : {legs}\n"
            f"Credit  : ${nc:.3f}/sh × {c} contracts = ${nc*100*c:,.2f}\n"
            f"Max risk: ${mr*100*c:,.2f}\n"
            f"Expiry  : {pos['expiry_date']}"
        )
        return jsonify({"output": output})
    except Exception as e:
        logger.exception("open error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/options/check", methods=["POST"])
def api_check():
    try:
        result = check_tp()
        if result:
            return jsonify({"output": f"Take-profit triggered!\nP&L: ${result['pnl']:+,.2f}\nOutcome: {result['outcome']}"})
        s = options_status()
        unreal = s.get("unrealized_pnl")
        cur    = s.get("current_value")
        if unreal is None:
            return jsonify({"output": "No open position to check."})
        pos = s["open_position"]
        nc  = pos["net_credit"]
        pct = (1 - cur / abs(nc)) * 100 if nc != 0 else 0
        return jsonify({"output": f"No TP yet.\nCurrent spread value: ${cur:.3f}/sh  ({pct:.0f}% decayed)\nUnrealized P&L: ${unreal:+,.2f}"})
    except Exception as e:
        logger.exception("check error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/options/settle", methods=["POST"])
def api_settle():
    try:
        result = settle_expiry()
        if result:
            return jsonify({"output": f"Settled.\nOutcome: {result['outcome']}\nP&L: ${result['pnl']:+,.2f}"})
        return jsonify({"output": "Nothing to settle — position not expired yet."})
    except Exception as e:
        logger.exception("settle error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/scheduler/status")
def api_scheduler_status():
    return jsonify({"running": _scheduler_running()})


@app.route("/api/scheduler/start", methods=["POST"])
def api_scheduler_start():
    global _scheduler_proc
    with _scheduler_lock:
        if _scheduler_running():
            return jsonify({"message": "Scheduler is already running."})
        _scheduler_proc = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=os.path.dirname(__file__),
        )
    logger.info("Scheduler started (PID %d)", _scheduler_proc.pid)
    return jsonify({"message": f"Scheduler started (PID {_scheduler_proc.pid})."})


@app.route("/api/scheduler/stop", methods=["POST"])
def api_scheduler_stop():
    global _scheduler_proc
    with _scheduler_lock:
        if not _scheduler_running():
            return jsonify({"message": "Scheduler is not running."})
        try:
            _scheduler_proc.terminate()
            _scheduler_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _scheduler_proc.kill()
        _scheduler_proc = None
    logger.info("Scheduler stopped")
    return jsonify({"message": "Scheduler stopped."})


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  Alpaca Trader Dashboard")
    print("  Open: http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
