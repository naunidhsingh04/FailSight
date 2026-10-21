"""Windows and labels of the testbed dataset: counts inside a window, the half-overlap labelling rule, transition flags and pre-failure targets."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from testbed import dataset as DS  # noqa: E402


def make_run(d, t0=1000.0):
    d.mkdir(parents=True)
    api = [dict(timestamp=t0 + 5 + i, status_code=500 if i % 4 == 0 else 200, latency_ms=10.0 + i, content_length=100) for i in range(0, 40)]
    logs = [dict(timestamp=t0 + 1 + i, level="ERROR" if i % 5 == 0 else "INFO") for i in range(0, 60)]
    met = [dict(timestamp=t0 + 2 * i, metric="process_cpu_percent", value=5.0 + i) for i in range(0, 120)]
    for name, rows in (("api", api), ("logs", logs), ("metrics", met)):
        (d / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (d / "meta.json").write_text(json.dumps(dict(start_ts=t0, duration_s=240.0, seed=1, profile="steady", rate=5.0, run_id="rx")), encoding="utf-8")
    return t0


def test_window_counts_grid_and_deltas(tmp_path):
    t0 = make_run(tmp_path / "rx")
    df, m = DS.window_features(tmp_path / "rx")
    assert len(df) == int((240 - 30) // 10) + 1 and df.window_start.iloc[1] - df.window_start.iloc[0] == 10
    w0 = df.iloc[0]                                           # [t0, t0+30): requests at t0+5 .. t0+29 -> 25
    assert w0.api_request_count == 25 and w0.api_5xx_count == len([i for i in range(0, 25) if i % 4 == 0])
    assert w0.log_line_count == 29 and w0.log_error_count == 6 and abs(w0.log_error_share - 6 / 29) < 1e-9
    assert abs(w0["m_process_cpu_percent__mean"] - np.mean([5.0 + i for i in range(15)])) < 1e-9
    assert df.iloc[3]["d_api_request_count"] == df.iloc[3].api_request_count - df.iloc[0].api_request_count
    assert df.window_end.iloc[-1] <= t0 + 240


def test_half_overlap_rule_transition_and_prefailure(tmp_path):
    t0 = make_run(tmp_path / "rx")
    df, m = DS.window_features(tmp_path / "rx")
    sched = pd.DataFrame([dict(fault_id="f1", fault_type="leak", klass="Code", kind="gradual", severity_level="mild", start_ts=t0 + 60, end_ts=t0 + 130, failure_ts=t0 + 110)])
    df = DS.label_run(df, m, sched)
    ov = df.overlap_s.to_numpy()
    assert (df.label_level == "Code").sum() == (ov >= 15).sum() and (df.label_level == "Normal").sum() == (ov < 15).sum()
    assert ((ov > 0) & (ov < 15)).sum() == df.transition.sum() > 0
    assert (df[df.transition == 1].label_level == "Normal").all()
    k = int(df.index[df.window_end == t0 + 100][0])           # a window ending 10 s before the failure
    assert abs(df.loc[k, "time_to_failure_s"] - 10) < 1e-9 and df.loc[k, "prefail_1m"] == 1
    assert df.loc[df.window_end > t0 + 110, "time_to_failure_s"].isna().all()                # nothing after the failure
    assert df.prefail_3m.sum() >= df.prefail_2m.sum() >= df.prefail_1m.sum() > 0
    assert df.in_learning.iloc[0] == 1 and df.in_learning.iloc[-1] == 0 or df.t_offset.iloc[-1] <= DS.LEARN_S
