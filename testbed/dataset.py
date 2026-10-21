"""Testbed dataset: 30 s windows with a 10 s step from the raw events of every run, features in the style of the earlier AnoMod windows (request count, error rate,
4xx/5xx, latency mean/p95/p99, log error share, process CPU / memory / threads / loop lag / pool wait and their changes) and labels from the recorded fault schedule.

  window k of a run covers [start + 10 k, start + 10 k + 30), start = meta.start_ts (the moment the load generator started); only windows that lie fully inside the run exist
  label_level   Normal, or the class of the fault (Performance / Service / Database / Code) that overlaps the window for at least half of it (15 s)
  transition    1 when a fault overlaps the window for less than half: the label is ambiguous; these windows are kept (labelled Normal, as specified) but flagged and are
                left out of training and evaluation (see docs/TESTBED_DATASET.md)
  gradual faults: time_to_failure_s = failure time - window end (> 0 before the failure) and prefail_1m / 2m / 3m = the window ends within 1 / 2 / 3 minutes before the failure
  baseline-relative versions rel__<feature>: robust z-score against the run's own past (scripts/universal_features.relative_features: windows that ended before this one began,
  last 60, at least 9; no label, no other run, no future window)

python -m testbed.dataset            -> processed/testbed_windows.parquet (git-ignored) and report_outputs/testbed_dataset_summary.csv"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
RUNS = ROOT / "testbed" / "runs"
OUT = ROOT / "processed" / "testbed_windows.parquet"
SUMMARY = ROOT / "report_outputs" / "testbed_dataset_summary.csv"
WIN, STEP, HALF = 30.0, 10.0, 15.0
LEARN_S = 150.0
METRICS = ["process_cpu_percent", "process_rss_bytes", "process_threads", "event_loop_lag_ms", "db_pool_wait_ms", "inflight_requests"]
DELTA_BASE = ["api_request_count", "api_error_rate", "api_latency_p95_ms", "log_error_count", "m_process_cpu_percent__mean", "m_process_rss_bytes__mean", "m_event_loop_lag_ms__mean", "m_db_pool_wait_ms__mean"]
META = ["window_id", "run_id", "window_index", "window_start", "window_end", "t_offset", "label_level", "label_binary", "fault_id", "fault_type", "severity_level", "fault_kind", "overlap_s", "transition",
        "time_to_failure_s", "prefail_1m", "prefail_2m", "prefail_3m", "in_learning", "run_seed", "profile", "rate"]


def read_jsonl(path):
    return pd.read_json(path, lines=True, convert_dates=False) if path.exists() and path.stat().st_size else pd.DataFrame()


def window_features(run_dir):
    m = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    t0, dur = m["start_ts"], m["duration_s"]
    api, logs, met = (read_jsonl(run_dir / f"{k}.jsonl") for k in ("api", "logs", "metrics"))
    n_win = int((dur - WIN) // STEP) + 1
    starts = t0 + STEP * np.arange(n_win)
    rows = []
    at = api.timestamp.to_numpy(float) if len(api) else np.array([])
    ao = np.argsort(at)
    at = at[ao]
    sc = api.status_code.to_numpy(int)[ao] if len(api) else np.array([], int)
    lat = api.latency_ms.to_numpy(float)[ao] if len(api) else np.array([])
    cl = api.content_length.to_numpy(float)[ao] if len(api) else np.array([])
    lt = np.sort(logs.timestamp.to_numpy(float)) if len(logs) else np.array([])
    lo = np.argsort(logs.timestamp.to_numpy(float)) if len(logs) else np.array([], int)
    lv = logs.level.to_numpy(str)[lo] if len(logs) else np.array([], str)
    series = {}
    for name in METRICS:
        x = met[met.metric == name].sort_values("timestamp") if len(met) else pd.DataFrame()
        series[name] = (x.timestamp.to_numpy(float), x.value.to_numpy(float)) if len(x) else (np.array([]), np.array([]))
    for k, ws in enumerate(starts):
        we = ws + WIN
        f = {}
        i, j = np.searchsorted(at, ws, "left"), np.searchsorted(at, we, "left")
        n = j - i
        s = sc[i:j]
        f["api_request_count"] = float(n)
        f["api_4xx_count"] = float(((s >= 400) & (s < 500)).sum())
        f["api_5xx_count"] = float((s >= 500).sum())
        f["api_error_rate"] = float((s >= 500).mean()) if n else 0.0
        l = lat[i:j]
        f["api_latency_mean_ms"] = float(l.mean()) if n else 0.0
        f["api_latency_p95_ms"] = float(np.percentile(l, 95)) if n else 0.0
        f["api_latency_p99_ms"] = float(np.percentile(l, 99)) if n else 0.0
        f["api_content_length_mean"] = float(cl[i:j].mean()) if n else 0.0
        a, b = np.searchsorted(lt, ws, "left"), np.searchsorted(lt, we, "left")
        lvl = lv[a:b]
        f["log_line_count"] = float(b - a)
        f["log_error_count"] = float(np.isin(lvl, ["ERROR", "CRITICAL"]).sum())
        f["log_warning_count"] = float((lvl == "WARNING").sum())
        f["log_error_share"] = float(f["log_error_count"] / (b - a)) if b > a else 0.0
        for name in METRICS:
            ts, vs = series[name]
            p, q = np.searchsorted(ts, ws, "left"), np.searchsorted(ts, we, "left")
            v = vs[p:q]
            if name == "process_rss_bytes":
                v = v / 1e6
            f[f"m_{name}__mean"] = float(v.mean()) if len(v) else np.nan
            f[f"m_{name}__max"] = float(v.max()) if len(v) else np.nan
            f[f"m_{name}__std"] = float(v.std()) if len(v) else np.nan
            if name == "process_rss_bytes":
                f["m_process_rss_bytes__slope"] = float(np.polyfit(ts[p:q] - ws, v, 1)[0]) if len(v) >= 3 else np.nan      # MB per second inside the window
        f["window_index"], f["window_start"], f["window_end"] = k, ws, we
        rows.append(f)
    df = pd.DataFrame(rows)
    for c in DELTA_BASE:
        df[f"d_{c}"] = df[c] - df[c].shift(3)                      # change against the previous non-overlapping window
    return df, m


def label_run(df, m, sched):
    """labels from the recorded fault schedule (exact start / end / failure timestamps)"""
    n = len(df)
    ws, we = df.window_start.to_numpy(), df.window_end.to_numpy()
    best_ov, best = np.zeros(n), [None] * n
    for r in sched.itertuples():
        end = r.end_ts if pd.notna(r.end_ts) else r.start_ts
        ov = np.clip(np.minimum(we, end) - np.maximum(ws, r.start_ts), 0, None)
        for k in np.where(ov > best_ov)[0]:
            best_ov[k], best[k] = ov[k], r
    df["overlap_s"] = best_ov
    big = best_ov >= HALF
    df["label_level"] = [b.klass if big[k] else "Normal" for k, b in enumerate(best)]
    df["label_binary"] = (df.label_level != "Normal").astype(int)
    df["fault_id"] = [b.fault_id if big[k] else "" for k, b in enumerate(best)]
    df["fault_type"] = [b.fault_type if big[k] else "" for k, b in enumerate(best)]
    df["severity_level"] = [b.severity_level if big[k] else "" for k, b in enumerate(best)]
    df["fault_kind"] = [b.kind if big[k] else "" for k, b in enumerate(best)]
    df["transition"] = ((best_ov > 0) & (best_ov < HALF)).astype(int)
    ttf = np.full(n, np.nan)
    for r in sched.itertuples():
        if pd.notna(r.failure_ts):
            t = r.failure_ts - we
            ttf = np.where(t > 0, np.where(np.isnan(ttf), t, np.minimum(ttf, t)), ttf)
    df["time_to_failure_s"] = ttf
    for nmin in (1, 2, 3):
        df[f"prefail_{nmin}m"] = ((df.time_to_failure_s > 0) & (df.time_to_failure_s <= nmin * 60)).astype(int)
    df["t_offset"] = df.window_end - m["start_ts"]
    df["in_learning"] = (df.t_offset <= LEARN_S).astype(int)
    return df


def build(runs_dir=RUNS):
    import universal_features as U
    parts = []
    for d in sorted(runs_dir.glob("r*")):
        if not (d / "DONE").exists():
            continue
        df, m = window_features(d)
        sched = pd.read_csv(d / "fault_schedule.csv", dtype={"fault_id": str})
        df = label_run(df, m, sched)
        df["run_id"], df["run_seed"], df["profile"], df["rate"] = d.name, m["seed"], m["profile"], m["rate"]
        df["window_id"] = d.name + "|" + df.window_index.astype(str)
        parts.append(df)
    W = pd.concat(parts, ignore_index=True)
    base = [c for c in W.columns if c not in META]
    W[base] = W[base].replace([np.inf, -np.inf], np.nan)
    rel = np.full((len(W), len(base)), np.nan)
    for run, idx in W.groupby("run_id").indices.items():
        idx = np.sort(idx)
        rel[idx] = U.relative_features(W.loc[idx, base].to_numpy(float), None, np.ones(len(idx), bool))     # a run is entirely on one side of every split: past windows of the same run only
    R = pd.DataFrame(rel, columns=[f"rel__{c}" for c in base], index=W.index)
    return pd.concat([W, R], axis=1), base


def feature_columns(W):
    base = [c for c in W.columns if c not in META and not c.startswith("rel__")]
    return base, [f"rel__{c}" for c in base]


def prove_no_future(W, base, n=60, seed=3):
    """overwrite every window that overlaps or follows window k (and nothing else) with garbage: the baseline-relative features of window k must not change"""
    import universal_features as U
    rng = np.random.default_rng(seed)
    runs = list(W.groupby("run_id").indices.items())
    bad = checked = 0
    for run, idx in [runs[i] for i in rng.permutation(len(runs))[:8]]:
        idx = np.sort(idx)
        X = W.loc[idx, base].to_numpy(float)
        tr = np.ones(len(idx), bool)
        ref = U.relative_features(X, None, tr)
        for k in rng.choice(np.arange(U.GUARD + U.MIN_HISTORY, len(idx)), size=min(8, len(idx) - 12), replace=False):
            Xp = X.copy()
            fut = [j for j in range(k - U.GUARD + 1, len(idx)) if j != k]
            Xp[fut] = rng.standard_normal((len(fut), X.shape[1])) * (np.nanmax(np.abs(X), axis=0) + 1) * 100
            checked += 1
            bad += 0 if np.array_equal(ref[k], U.relative_features(Xp, None, tr)[k], equal_nan=True) else 1
    return checked, bad


def main():
    W, base = build()
    OUT.parent.mkdir(exist_ok=True)
    W.to_parquet(OUT, index=False)
    n_chk, n_bad = prove_no_future(W, base)
    s = (W.groupby(["label_level"]).size().rename("windows").reset_index())
    s["share"] = s.windows / s.windows.sum()
    s.to_csv(SUMMARY, index=False)
    print(f"{len(W)} windows from {W.run_id.nunique()} runs, {len(base)} features (+{len(base)} baseline-relative)")
    print(W.groupby("label_level").size().to_string())
    print("transition windows (partial overlap, flagged):", int(W.transition.sum()))
    print(f"no-future proof: {n_chk} windows checked, {n_bad} differences")
    assert n_bad == 0


if __name__ == "__main__":
    main()
