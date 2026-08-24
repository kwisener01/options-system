"""
Walk-forward ML study: does price action across a cross-asset universe
(available via Alpaca, no options-chain/OI data needed) predict SPY's own
next-N-minute direction, beyond a naive baseline?

Universe: SPY (target) + QQQ, IWM, DIA (broad indices), VXX (vol proxy),
TLT (rates), HYG (credit), UUP (dollar), GLD (gold), USO (oil), XLK/XLF/XLE
(sectors), AAPL/MSFT/NVDA (mega-cap leaders that can lead cap-weighted SPY).

Design choices, made specifically to avoid the mistakes in the first pass of
run_charm_predictor_test.py:
  - RTH only (09:30-16:00 ET), no extended-hours contamination.
  - Non-overlapping label windows within each day (step by horizon bars) —
    overlapping windows inflate apparent significance via autocorrelation.
  - CHRONOLOGICAL walk-forward folds (expanding window) — never trains on
    the future. Random k-fold would leak information in a time series and
    silently invalidate every result.
  - Reports economic significance (mean forward return, hit rate, net of a
    slippage estimate) on top of accuracy/AUC. A statistically "significant"
    classifier that doesn't clear costs is not tradeable — that's exactly
    what sank the 0DTE Levitation/straddle backtests earlier this session.
  - All features at time t use ONLY bars up to and including t (no lookahead).

Usage: python run_ml_direction_study.py [--days 500] [--horizon 6] [--folds 5]
"""
import argparse, json, os, sys, time, hashlib
from datetime import datetime, timedelta, date, time as dt_time
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(__file__))
import requests
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY

ET, UTC = ZoneInfo("America/New_York"), ZoneInfo("UTC")
STK = "https://data.alpaca.markets/v2/stocks"
HEAD = {"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY}
_CACHE = os.path.join(os.path.dirname(__file__), ".cache", "mlstudy")
os.makedirs(_CACHE, exist_ok=True)

TARGET = "SPY"
UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA",        # broad indices
    "VXX",                              # vol proxy
    "TLT", "HYG", "UUP",                # rates, credit, dollar
    "GLD", "USO",                       # commodities
    "XLK", "XLF", "XLE",                # sectors
    "AAPL", "MSFT", "NVDA",             # mega-cap leaders
]
LOOKBACKS = [1, 3, 6, 12]   # bars (5/15/30/60 min) for lagged-return features


def _u(d, t):
    return datetime.combine(d, t, ET).astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_universe(start, end):
    """{symbol: {(date, hhmm): price}} for the whole UNIVERSE, RTH only, cached."""
    key = hashlib.md5(f"{start}-{end}-{','.join(UNIVERSE)}".encode()).hexdigest()
    fp = os.path.join(_CACHE, f"bars_{key}.parquet")
    if os.path.exists(fp):
        print(f"Loading cached bars from {fp}")
        return pd.read_parquet(fp)

    rows, tok = [], None
    print(f"Fetching {len(UNIVERSE)} symbols, {start} -> {end} ...")
    while True:
        p = {"symbols": ",".join(UNIVERSE), "timeframe": "5Min",
             "start": _u(start, dt_time(9, 30)), "end": _u(end, dt_time(16, 0)),
             "limit": 10000, "feed": "sip"}
        if tok: p["page_token"] = tok
        r = requests.get(f"{STK}/bars", headers=HEAD, params=p, timeout=30)
        d = r.json() if r.status_code == 200 else {}
        for sym, bars in (d.get("bars") or {}).items():
            for b in bars:
                dt_ = datetime.fromisoformat(b["t"].replace("Z", "+00:00")).astimezone(ET)
                rows.append((sym, dt_.date(), dt_.strftime("%H:%M"), b.get("vw", b["c"])))
        tok = d.get("next_page_token")
        if not tok: break
        print(f"  ...{len(rows)} rows so far")

    df = pd.DataFrame(rows, columns=["symbol", "date", "hhmm", "px"])
    df = df[(df["hhmm"] >= "09:30") & (df["hhmm"] <= "16:00")]
    df = df.drop_duplicates(["symbol", "date", "hhmm"])
    df.to_parquet(fp)
    return df


def build_wide(df):
    """Pivot to wide: index=(date,hhmm), columns=symbol, values=price. Days where
    any universe member is missing bars get dropped (keeps the panel rectangular)."""
    wide = df.pivot_table(index=["date", "hhmm"], columns="symbol", values="px")
    wide = wide.sort_index()
    wide = wide.dropna(axis=0, how="any")   # require all universe members present
    return wide


def build_features_labels(wide, horizon):
    """Lagged-return features (all tickers) at time t -> forward-return sign of
    SPY from t to t+horizon, same day only, non-overlapping samples (step=horizon)."""
    feat_rows, labels, meta = [], [], []
    for d, g in wide.groupby(level=0):
        g = g.reset_index(level=0, drop=True)
        n = len(g)
        if n <= max(LOOKBACKS) + horizon:
            continue
        px = {c: g[c].values for c in g.columns}
        for i in range(max(LOOKBACKS), n - horizon, horizon):
            feats = {}
            for sym in UNIVERSE:
                for lb in LOOKBACKS:
                    feats[f"{sym}_ret_{lb}"] = px[sym][i] / px[sym][i - lb] - 1.0
            fwd = px[TARGET][i + horizon] / px[TARGET][i] - 1.0
            feat_rows.append(feats)
            labels.append(1 if fwd > 0 else 0)
            meta.append((d, g.index[i], fwd))
    X = pd.DataFrame(feat_rows)
    y = np.array(labels)
    meta = pd.DataFrame(meta, columns=["date", "hhmm", "fwd_ret"])
    return X, y, meta


def fetch_universe_daily(start, end):
    """{symbol: {date: close}} for the whole UNIVERSE on 1Day bars, cached separately
    from the intraday cache. Daily bars go back much further than 5Min history."""
    key = hashlib.md5(f"daily-{start}-{end}-{','.join(UNIVERSE)}".encode()).hexdigest()
    fp = os.path.join(_CACHE, f"daily_{key}.parquet")
    if os.path.exists(fp):
        print(f"Loading cached daily bars from {fp}")
        return pd.read_parquet(fp)

    rows, tok = [], None
    print(f"Fetching {len(UNIVERSE)} symbols, 1Day, {start} -> {end} ...")
    while True:
        p = {"symbols": ",".join(UNIVERSE), "timeframe": "1Day",
             "start": _u(start, dt_time(0, 0)), "end": _u(end, dt_time(23, 59)),
             "limit": 10000, "feed": "sip"}
        if tok: p["page_token"] = tok
        r = requests.get(f"{STK}/bars", headers=HEAD, params=p, timeout=30)
        d = r.json() if r.status_code == 200 else {}
        for sym, bars in (d.get("bars") or {}).items():
            for b in bars:
                dt_ = datetime.fromisoformat(b["t"].replace("Z", "+00:00")).astimezone(ET)
                rows.append((sym, dt_.date(), b["c"]))
        tok = d.get("next_page_token")
        if not tok: break

    df = pd.DataFrame(rows, columns=["symbol", "date", "close"])
    df = df.drop_duplicates(["symbol", "date"])
    df.to_parquet(fp)
    return df


def build_wide_daily(df):
    wide = df.pivot_table(index="date", columns="symbol", values="close")
    wide = wide.sort_index()
    return wide.dropna(axis=0, how="any")


DAILY_LOOKBACKS = [1, 3, 5, 10, 20]   # trading days


def build_features_labels_daily(wide, horizon_days):
    """Lagged daily-return features (all tickers) as of close t -> sign of SPY's
    close-to-close return from t to t+horizon_days. Non-overlapping (step=horizon_days)."""
    n = len(wide)
    idx = wide.index
    px = {c: wide[c].values for c in wide.columns}
    feat_rows, labels, meta = [], [], []
    for i in range(max(DAILY_LOOKBACKS), n - horizon_days, horizon_days):
        feats = {}
        for sym in UNIVERSE:
            for lb in DAILY_LOOKBACKS:
                feats[f"{sym}_ret_{lb}d"] = px[sym][i] / px[sym][i - lb] - 1.0
        fwd = px[TARGET][i + horizon_days] / px[TARGET][i] - 1.0
        feat_rows.append(feats)
        labels.append(1 if fwd > 0 else 0)
        meta.append((idx[i], fwd))
    X = pd.DataFrame(feat_rows)
    y = np.array(labels)
    meta = pd.DataFrame(meta, columns=["date", "fwd_ret"])
    return X, y, meta


def walk_forward(X, y, meta, n_folds, slip_bps):
    """Expanding-window chronological folds — train on the past, test on the
    next block, never the reverse. Reports per-fold + pooled OOS metrics for
    both a linear (logistic) and a nonlinear (gradient boosting) model."""
    n = len(X)
    fold_edges = np.linspace(0, n, n_folds + 1).astype(int)
    results = {"logreg": [], "gbm": []}
    all_preds = {"logreg": [], "gbm": []}

    for k in range(1, n_folds):
        train_idx = slice(0, fold_edges[k])
        test_idx = slice(fold_edges[k], fold_edges[k + 1])
        Xtr, ytr = X.iloc[train_idx], y[train_idx]
        Xte, yte = X.iloc[test_idx], y[test_idx]
        mte = meta.iloc[test_idx]
        if len(np.unique(ytr)) < 2 or len(Xte) == 0:
            continue

        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

        lr = LogisticRegression(max_iter=1000, C=1.0).fit(Xtr_s, ytr)
        p_lr = lr.predict_proba(Xte_s)[:, 1]

        gbm = HistGradientBoostingClassifier(max_depth=3, max_iter=150,
                                              learning_rate=0.05).fit(Xtr, ytr)
        p_gbm = gbm.predict_proba(Xte)[:, 1]

        for name, p in [("logreg", p_lr), ("gbm", p_gbm)]:
            auc = roc_auc_score(yte, p) if len(np.unique(yte)) > 1 else float("nan")
            acc = accuracy_score(yte, (p > 0.5).astype(int))
            baseline_acc = max(yte.mean(), 1 - yte.mean())
            results[name].append({"fold": k, "n": len(yte), "auc": auc,
                                   "acc": acc, "baseline_acc": baseline_acc})
            all_preds[name].append(pd.DataFrame({"p": p, "fwd_ret": mte["fwd_ret"].values}))

    return results, all_preds


def economic_report(name, preds_by_fold, slip_bps, horizon_label):
    all_p = pd.concat(preds_by_fold, ignore_index=True)
    print(f"\n  --- {name}: economic significance (pooled OOS, {horizon_label} horizon) ---")
    for thresh in [0.55, 0.60, 0.65, 0.70]:
        conf = all_p[(all_p["p"] >= thresh) | (all_p["p"] <= 1 - thresh)]
        if len(conf) == 0:
            print(f"  conf>={thresh:.2f}: no trades"); continue
        direction = np.where(conf["p"] >= 0.5, 1, -1)
        realized = direction * conf["fwd_ret"].values
        mean_bps = realized.mean() * 10000
        hit = (realized > 0).mean()
        net_bps = mean_bps - slip_bps
        flag = "  <-- clears costs" if net_bps > 0 else ""
        print(f"  conf>={thresh:.2f}  n={len(conf):5d}  hit={hit*100:5.1f}%  "
              f"gross={mean_bps:+6.2f}bp  net_of_{slip_bps}bp_slip={net_bps:+6.2f}bp{flag}")


def run(days, horizon, folds, slip_bps, daily=False, horizon_days=1):
    end = datetime.now(ET).date() - timedelta(days=1)
    start = end - timedelta(days=days)

    if daily:
        df = fetch_universe_daily(start, end)
        print(f"{len(df)} raw daily bars, {df['symbol'].nunique()} symbols, {df['date'].nunique()} days")
        wide = build_wide_daily(df)
        print(f"Rectangular panel: {len(wide)} trading days "
              f"(days with any missing symbol dropped)")
        X, y, meta = build_features_labels_daily(wide, horizon_days)
        horizon_min = None
        horizon_label = f"{horizon_days}day"
    else:
        df = fetch_universe(start, end)
        print(f"{len(df)} raw bars, {df['symbol'].nunique()} symbols, {df['date'].nunique()} days")
        wide = build_wide(df)
        print(f"Rectangular panel: {len(wide)} bars across {wide.index.get_level_values(0).nunique()} days "
              f"(days with any missing symbol dropped)")
        X, y, meta = build_features_labels(wide, horizon)
        horizon_min = horizon * 5
        horizon_label = f"{horizon_min}min"

    print(f"Built {len(X)} non-overlapping samples, {X.shape[1]} features, "
          f"base rate up={y.mean()*100:.1f}%\n")

    results, preds = walk_forward(X, y, meta, folds, slip_bps)
    for name in ["logreg", "gbm"]:
        print(f"=== {name} — per-fold OOS (chronological, never trained on the future) ===")
        for r in results[name]:
            edge = r["acc"] - r["baseline_acc"]
            print(f"  fold {r['fold']}  n={r['n']:5d}  AUC={r['auc']:.3f}  "
                  f"acc={r['acc']*100:5.1f}%  baseline={r['baseline_acc']*100:5.1f}%  "
                  f"edge={edge*100:+5.1f}pp")
        if preds[name]:
            economic_report(name, preds[name], slip_bps, horizon_label)
        print()

    print("=" * 70)
    print("Read this as: consistent positive AUC/accuracy-edge AND positive net-of-slippage")
    print("bps ACROSS folds = real signal. A good fold followed by a bad one, or gross")
    print("edge that dies after slippage, means noise/overfitting — same failure mode")
    print("that killed the 0DTE Levitation and straddle backtests earlier.")
    print("=" * 70)

    # Interpretability only — fit on ALL data to see which tickers the model leans
    # on. NOT an OOS result, don't read this as "these predict SPY" — it's diagnostic.
    gbm_full = HistGradientBoostingClassifier(max_depth=3, max_iter=150,
                                               learning_rate=0.05).fit(X, y)
    from sklearn.inspection import permutation_importance
    imp = permutation_importance(gbm_full, X, y, n_repeats=5, random_state=0, n_jobs=-1)
    order = np.argsort(imp.importances_mean)[::-1][:15]
    print("\nTop 15 features by permutation importance (in-sample, diagnostic only):")
    for i in order:
        print(f"  {X.columns[i]:20}  importance={imp.importances_mean[i]:+.5f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=500,
                     help="lookback window in calendar days (intraday: 5Min bar history; "
                          "daily: use a much larger value, e.g. 2000+, for enough trading days")
    ap.add_argument("--horizon", type=int, default=6, help="bars ahead for the label (6=30min, intraday mode)")
    ap.add_argument("--daily", action="store_true", help="use 1Day bars instead of 5Min")
    ap.add_argument("--horizon-days", type=int, default=1, help="trading days ahead for the label (daily mode)")
    ap.add_argument("--folds", type=int, default=5, help="chronological walk-forward folds")
    ap.add_argument("--slip-bps", type=float, default=3.0, help="round-trip slippage estimate, bps")
    a = ap.parse_args()
    run(a.days, a.horizon, a.folds, a.slip_bps, daily=a.daily, horizon_days=a.horizon_days)
