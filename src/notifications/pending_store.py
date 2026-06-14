"""
Pending-trade store for Slack approval buttons.

A scan/manage job that wants human sign-off writes a *pending* trade here and
puts the returned short id in the Slack button's `value`. When the user taps the
button, the interactive endpoint looks the trade back up, re-prices it live, and
either submits or reports why it stood down.

Backed by a small JSON file so a Render restart doesn't lose in-flight approvals.
Pure-Python / no external deps so it is unit-testable without Alpaca or Slack.
"""
import json
import logging
import os
import threading
import time
import uuid

logger = logging.getLogger(__name__)

_STORE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "pending_trades.json"
)
_lock = threading.Lock()

DEFAULT_TTL = 1800          # 30 min — alert goes stale after this; tap is rejected
MAX_KEEP    = 200           # cap stored history so the file can't grow unbounded


def _load() -> dict:
    try:
        with open(_STORE_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"trades": {}}
    except Exception as e:                       # pragma: no cover - defensive
        logger.warning("pending_store load failed: %s", e)
        return {"trades": {}}


def _save(store: dict) -> None:
    trades = store.get("trades", {})
    # Trim oldest if we blow past the cap.
    if len(trades) > MAX_KEEP:
        for k in sorted(trades, key=lambda k: trades[k].get("created", 0))[: len(trades) - MAX_KEEP]:
            trades.pop(k, None)
    tmp = _STORE_PATH + ".tmp"
    os.makedirs(os.path.dirname(_STORE_PATH), exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(store, f, indent=2, default=str)
    os.replace(tmp, _STORE_PATH)


def add(trade: dict, ttl: int = DEFAULT_TTL) -> str:
    """Store a pending trade and return its short id. `trade` is any JSON-able
    dict describing the order to (re-price and) submit on approval."""
    tid = uuid.uuid4().hex[:8]
    now = time.time()
    record = dict(trade)
    record.update({
        "id": tid,
        "created": now,
        "expires": now + ttl,
        "status": "pending",
    })
    with _lock:
        store = _load()
        store.setdefault("trades", {})[tid] = record
        _save(store)
    return tid


def get(tid: str) -> dict | None:
    with _lock:
        return _load().get("trades", {}).get(tid)


def is_actionable(tid: str) -> tuple[bool, str]:
    """Return (ok, reason). A trade is actionable only if it exists, is still
    pending, and hasn't expired."""
    rec = get(tid)
    if rec is None:
        return False, "not_found"
    if rec.get("status") != "pending":
        return False, f"already_{rec.get('status')}"
    if time.time() > rec.get("expires", 0):
        update_status(tid, "expired")
        return False, "expired"
    return True, "ok"


def update_status(tid: str, status: str, **extra) -> dict | None:
    with _lock:
        store = _load()
        rec = store.get("trades", {}).get(tid)
        if rec is None:
            return None
        rec["status"] = status
        rec["updated"] = time.time()
        rec.update(extra)
        _save(store)
        return rec


def purge_expired() -> int:
    """Mark still-pending trades past their TTL as expired. Returns count."""
    n = 0
    now = time.time()
    with _lock:
        store = _load()
        for rec in store.get("trades", {}).values():
            if rec.get("status") == "pending" and now > rec.get("expires", 0):
                rec["status"] = "expired"
                n += 1
        if n:
            _save(store)
    return n
