"""Run planning (determinism, no overlapping faults, a fault-free learning period) and the fault schedule built from the app's own event log."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from testbed import loadgen, run_generator as RG  # noqa: E402


def test_plan_is_a_function_of_the_seed_alone():
    assert RG.plan_run(3, 1003, 10) == RG.plan_run(3, 1003, 10)
    assert RG.plan_run(3, 1003, 10) != RG.plan_run(3, 1004, 10)


def test_faults_never_overlap_and_leave_a_fault_free_start_and_end():
    zero = 0
    for seed in range(1000, 1200):
        p = RG.plan_run(0, seed, 10)
        zero += not p["faults"]
        fs = p["faults"]
        assert len(fs) <= 3
        for a, b in zip(fs, fs[1:]):
            assert a["start"] + a["duration"] + RG.GAP_S <= b["start"] + 1e-6
        for f in fs:
            assert f["start"] >= RG.LEARN_S and f["start"] + f["duration"] <= p["duration_s"] - RG.END_MARGIN_S + 1e-6
    assert 30 <= zero <= 80                                  # about a quarter of the runs have no fault at all


def test_traffic_profiles_and_mixes():
    for prof in loadgen.PROFILES:
        fn = loadgen.make_rate_fn(prof, 8.0, 600, 5)
        r = [fn(t) for t in range(0, 600, 10)]
        assert min(r) > 0
    assert len({round(loadgen.make_rate_fn("steady", 8, 600, 5)(t), 3) for t in range(0, 600, 10)}) == 1
    assert len({round(x, 2) for x in [loadgen.make_rate_fn("ramp", 8, 600, 5)(t) for t in range(0, 600, 50)]}) > 5
    m = loadgen.random_mix(11)
    assert abs(sum(m.values()) - 1) < 1e-9 and set(m) == set(loadgen.SESSIONS)


def test_schedule_from_events_including_crash_restart_and_gradual_failure(tmp_path):
    ev = [dict(fault_id="a", fault_type="latency", klass="Performance", kind="abrupt", event="start", ts=100.0, params={"added_ms": 400}),
          dict(fault_id="a", fault_type="latency", klass="Performance", kind="abrupt", event="end", ts=160.0, params={}),
          dict(fault_id="b", fault_type="leak", klass="Code", kind="gradual", event="start", ts=300.0, params={"mb_per_s": 2}),
          dict(fault_id="b", fault_type="leak", klass="Code", kind="gradual", event="failure", ts=380.0, params={}),
          dict(fault_id="b", fault_type="leak", klass="Code", kind="gradual", event="end", ts=420.0, params={}),
          dict(fault_id="c", fault_type="crash", klass="Service", kind="abrupt", event="start", ts=500.0, params={"down_s": 20}),
          dict(event="restart", ts=526.5, down_since=501.0)]
    (tmp_path / "fault_events.jsonl").write_text("\n".join(json.dumps(e) for e in ev), encoding="utf-8")
    rows = RG.build_schedule(tmp_path, dict(run_id="r1"))
    by = {r["fault_type"]: r for r in rows}
    assert by["latency"]["start_ts"] == 100.0 and by["latency"]["end_ts"] == 160.0 and by["latency"]["failure_ts"] is None
    assert by["leak"]["failure_ts"] == 380.0 and by["leak"]["kind"] == "gradual"
    assert by["crash"]["end_ts"] == 526.5 and by["crash"]["duration_s"] == 26.5
    assert (tmp_path / "fault_schedule.csv").exists()
