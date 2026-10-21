"""The live testbed monitor builds the same windows, baseline-relative features and risks as the offline dataset and the frozen model, learns first, then alerts with reasons."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from testbed import dataset as DS  # noqa: E402

CFG = ROOT / "report_outputs" / "testbed_frozen_config.json"
MODEL = ROOT / "models" / "testbed" / "detector.joblib"
RUN = "r004_s1004"
need = pytest.mark.skipif(not (CFG.exists() and MODEL.exists() and DS.OUT.exists() and (DS.RUNS / RUN / "DONE").exists()), reason="frozen detector, dataset or run not present")


@pytest.fixture(scope="module")
def replay():
    from app.backend import testbed_monitor as TM
    TM.LIVE_DIR = Path(__file__).resolve().parents[1] / "outputs" / "testbed_live_test"
    mon = TM.TestbedMonitor()
    d = DS.RUNS / RUN
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    mon.reset("t", meta["start_ts"])
    for kind in ("api", "logs", "metrics"):
        ev = [json.loads(line) for line in open(d / f"{kind}.jsonl", encoding="utf-8")]
        for i in range(0, len(ev), 2000):
            mon.ingest(kind, "t", ev[i:i + 2000])
    mon.close_ready(meta["start_ts"] + 100000)
    return mon, mon.windows("t"), meta


@need
def test_live_windows_equal_the_offline_dataset_features_and_model_risk(replay):
    mon, wins, meta = replay
    W = pd.read_parquet(DS.OUT)
    ref = W[W.run_id == RUN].sort_values("window_index").reset_index(drop=True)
    assert len(wins) == len(ref) == 58
    base, rel = DS.feature_columns(W)
    from testbed.validate import make_X
    X = make_X(ref, base, rel, mon.cfg["feature_set"])
    risk_off = mon.pipeline.predict_proba(X)[:, 1]
    risk_live = np.array([w["risk"] for w in wins])
    assert np.max(np.abs(risk_off - risk_live)) < 1e-9
    assert all(abs(w["window_start"] - s) < 1e-6 for w, s in zip(wins, ref.window_start))


@need
def test_learning_mode_then_alerts_after_the_fault_with_plain_reasons(replay):
    mon, wins, meta = replay
    learn = [w for w in wins if w["learning"]]
    assert len(learn) == 13 and all(not w["flagged"] and not w["alert"] and w["status"] == "learning" for w in learn)
    assert [round(w["learning_progress"], 2) for w in learn][-1] == 1.0 and learn[0]["learning_progress"] < 0.3
    sched = pd.read_csv(DS.RUNS / RUN / "fault_schedule.csv")
    first_fault = sched.start_ts.min()
    alerts = [w for w in wins if w["new_alert"]]
    assert alerts and all(w["window_end"] > meta["start_ts"] + 150 for w in alerts)
    assert min(w["window_end"] for w in alerts) >= first_fault                    # no alert before the first fault in this run
    a = alerts[0]
    assert a["status"] == "failing" and a["reasons"] and "normally" in a["summary"]
    assert a["model"]["name"] == mon.cfg["model"] and a["model"]["alert_rule"] == "2 of last 3 windows"


@need
def test_two_of_three_rule_and_second_opinion(replay):
    mon, wins, meta = replay
    f = [w["flagged"] for w in wins]
    for k, w in enumerate(wins):
        expect = (not w["learning"]) and sum(f[max(0, k - 2):k + 1]) >= 2
        assert w["alert"] == expect or (w["learning"] and not w["alert"])
    assert all(w["robust_z_score"] is None for w in wins if w["learning"]) and any(w["robust_z_score"] is not None for w in wins)


@need
def test_malformed_events_are_counted_not_fatal_and_baseline_uses_only_the_past(replay):
    mon, wins, meta = replay
    r = mon.ingest("api", "x1", [dict(timestamp="nope"), dict(timestamp=1.0, status_code=200, latency_ms=3), "str", dict(status_code=200)])
    assert r == dict(app="x1", accepted=1, rejected=3)
    with pytest.raises(ValueError):
        mon.ingest("api", "", [])
    # the baseline-relative row of a window equals universal_features.relative_features computed on the past only
    import universal_features as U
    st = mon.apps["t"]
    H = np.array([[row[c] for c in mon.base_cols] for row in st.base_hist], float)
    full = U.relative_features(H, None, np.ones(len(H), bool))
    k = 30
    st2 = type("S", (), {"base_hist": st.base_hist[:k]})()
    live = mon.rel_row(st2, k, st.base_hist[k])
    assert np.array_equal(live, full[k], equal_nan=True)
    future_changed = [dict(r) for r in st.base_hist[:k]] + [dict(st.base_hist[k])] + [{c: 1e9 for c in mon.base_cols}] * 5
    st3 = type("S", (), {"base_hist": future_changed[:k]})()
    assert np.array_equal(mon.rel_row(st3, k, future_changed[k]), live, equal_nan=True)
