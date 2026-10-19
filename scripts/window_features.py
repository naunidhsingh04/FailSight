"""Window feature code shared by the live monitoring pipeline (app/backend/monitor.py), the replay checks and the equivalence tests.

It implements, as functions over the raw events of ONE 30 s window, exactly the arithmetic of the training window builder
(AnoMod/scripts/02_windows.py, read-only; copy in scripts/old_pipeline/): window k covers [anchor + 10 k, anchor + 10 k + 30);
  api   : request count, 4xx / 5xx counts, error rate, latency mean / p95 / p99, content-length mean
  logs  : physical lines, ERROR lines (level code 3), WARN lines (level code 2)
  metric: mean / max / population std of all samples of a metric name inside the window; *_total counters are first converted to a
          per-second rate between consecutive samples of the same series (first sample, resets and dt <= 0 give no rate).
Equivalence with the training tables is verified by tests/test_window_equivalence.py (and scripts/proofs.py), not assumed.
"""
import math

import numpy as np

STEP, WIN = 10.0, 30.0
API_FEATURES = ("api_error_rate", "api_latency_mean_ms", "api_latency_p95_ms", "api_latency_p99_ms", "api_content_length_mean",
                "api_request_count", "api_4xx_count", "api_5xx_count")
LOG_FEATURES = ("log_line_count", "log_error_count", "log_warning_count")
# log level codes of the tidy tables: 0 continuation line, 1 INFO, 2 WARN, 3 ERROR, 4 DEBUG, 5 OTHER
LEVEL_CODES = {"CONT": 0, "CONTINUATION": 0, "INFO": 1, "WARN": 2, "WARNING": 2, "ERROR": 3, "ERR": 3, "FATAL": 3, "CRITICAL": 3, "DEBUG": 4, "TRACE": 4, "OTHER": 5}


def window_bounds(anchor: float, k: int):
    ws = anchor + STEP * k
    return ws, ws + WIN


def is_counter(metric_name: str) -> bool:
    return metric_name.endswith("_total") and not metric_name.startswith("rate(")


def counter_rate(prev, t: float, v: float) -> float:
    """per-second rate between the previous sample (t_prev, v_prev) of the same series and this one; NaN when there is none, on a reset or dt <= 0"""
    if prev is None or v is None or math.isnan(v):
        return math.nan
    dv, dt = v - prev[1], t - prev[0]
    if dt <= 0 or dv < 0 or math.isnan(dv):
        return math.nan
    return dv / dt


def level_code(level) -> int:
    if isinstance(level, (int, np.integer)):
        return int(level)
    return LEVEL_CODES.get(str(level).strip().upper(), 5)


def api_features(status, latency, content_length) -> dict:
    status = np.asarray(status, dtype=float)
    n = len(status)
    f = {k: math.nan for k in API_FEATURES}
    f["api_request_count"], f["api_4xx_count"], f["api_5xx_count"] = 0.0, 0.0, 0.0
    if n == 0:
        return f
    n4 = int(((status >= 400) & (status < 500)).sum())
    n5 = int((status >= 500).sum())
    f["api_request_count"], f["api_4xx_count"], f["api_5xx_count"] = float(n), float(n4), float(n5)
    f["api_error_rate"] = (n4 + n5) / n
    la = np.asarray(latency, dtype=float)
    la = la[~np.isnan(la)]
    if len(la):
        f["api_latency_mean_ms"], f["api_latency_p95_ms"], f["api_latency_p99_ms"] = float(la.mean()), float(np.percentile(la, 95)), float(np.percentile(la, 99))
    ca = np.asarray(content_length, dtype=float)
    ca = ca[~np.isnan(ca)]
    if len(ca):
        f["api_content_length_mean"] = float(ca.mean())
    return f


def log_features(levels) -> dict:
    lv = np.asarray(levels, dtype=int)
    return {"log_line_count": float(len(lv)), "log_error_count": float((lv == 3).sum()), "log_warning_count": float((lv == 2).sum())}


def metric_features(samples_by_metric: dict) -> dict:
    """samples_by_metric: {metric name: array of values (NaN allowed) inside the window} -> {'m_<name>__mean|max|std': value}"""
    out = {}
    for name, vals in samples_by_metric.items():
        v = np.asarray(vals, dtype=float)
        v = v[~np.isnan(v)]
        if len(v):
            out[f"m_{name}__mean"], out[f"m_{name}__max"], out[f"m_{name}__std"] = float(v.mean()), float(v.max()), float(v.std())
        else:
            out[f"m_{name}__mean"] = out[f"m_{name}__max"] = out[f"m_{name}__std"] = math.nan
    return out


def build_window_features(api=None, logs=None, metrics=None) -> dict:
    """api = (status, latency, content_length) arrays; logs = level codes; metrics = {name: values}. All restricted to the window by the caller."""
    f = {}
    a = api if api is not None else ([], [], [])
    f.update(api_features(*a))
    f.update(log_features(logs if logs is not None else []))
    f.update(metric_features(metrics or {}))
    return f
