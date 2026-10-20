"""Run planning (determinism, no overlapping faults, a fault-free learning period) and the fault schedule built from the app's own event log."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from testbed import loadgen, run_generator as RG  # noqa: E402


def test_plan_is_a_function_of_the_seed_alone():
    assert RG.plan_all(40, 1000, 10) == RG.plan_all(40, 1000, 10)
    assert RG.plan_all(40, 1000, 10) != RG.plan_all(40, 2000, 10)
    assert [p["seed"] for p in RG.plan_all(5, 1000, 10)] == [1000, 1001, 1002, 1003, 1004]


def test_balanced_plan_every_type_at_least_four_times_no_overlap_fault_free_start():
    plans = RG.plan_all(40, 1000, 10)
    assert len(plans) == 40 and sum(1 for p in plans if not p["faults"]) == 10
    faults = [f for p in plans for f in p["faults"]]
    for t in RG.TYPE_ORDER:
        assert sum(1 for f in faults if f["type"] == t) >= 4, t
    assert sum(1 for f in faults if f["type"] == "db_lock") >= 4
    for p in plans:
        fs = p["faults"]
        assert len(fs) <= RG.MAX_FAULTS_PER_RUN
        for a, b in zip(fs, fs[1:]):
            assert a["start"] + a["duration"] + RG.GAP_S <= b["start"] + 1e-6
        for f in fs:
            assert f["start"] >= RG.LEARN_S and f["start"] + f["duration"] <= p["duration_s"] - RG.END_MARGIN_S + 1e-6
            assert f["level"] in ("mild", "medium", "severe") and 0 <= f["u"] <= 1 and RG.severity_level(f["u"]) == f["level"]


def test_severity_is_stratified_per_type_and_maps_monotonically():
    import random
    faults = [f for p in RG.plan_all(40, 1000, 10) for f in p["faults"]]
    for t in RG.TYPE_ORDER:
        levels = {f["level"] for f in faults if f["type"] == t}
        assert levels == {"mild", "medium", "severe"} or sum(1 for f in faults if f["type"] == t) < 3, t
    r = random.Random(1)
    assert RG.severity_params("latency", 0.0, 60, r)["added_ms"] == 30 and RG.severity_params("latency", 1.0, 60, r)["added_ms"] == 800
    assert RG.severity_params("cpu_burn", 0.0, 60, r)["duty"] == 0.2 and RG.severity_params("cpu_burn", 1.0, 60, r)["duty"] == 1.0
    assert RG.severity_params("route_500", 0.0, 60, r)["prob"] == 0.05 and RG.severity_params("exception_route", 1.0, 60, r)["prob"] == 1.0
    assert RG.severity_params("leak", 1.0, 150, r)["mb_per_s"] > RG.severity_params("leak", 0.0, 150, r)["mb_per_s"]
    assert RG.severity_params("pool_exhaustion", 1.0, 150, r)["every_s"] < RG.severity_params("pool_exhaustion", 0.0, 150, r)["every_s"]
    assert RG.fault_duration("db_lock", 0.0, r) == 15 and RG.fault_duration("db_lock", 1.0, r) == 100
    assert RG.fault_duration("crash", 0.0, r) == 28 and RG.fault_duration("crash", 1.0, r) == 53


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
    ev = [dict(fault_id="a", fault_type="latency", klass="Performance", kind="abrupt", event="start", ts=100.0, params={"added_ms": 400, "severity": 0.5, "severity_level": "medium"}),
          dict(fault_id="a", fault_type="latency", klass="Performance", kind="abrupt", event="end", ts=160.0, params={}),
          dict(fault_id="b", fault_type="leak", klass="Code", kind="gradual", event="start", ts=300.0, params={"mb_per_s": 2}),
          dict(fault_id="b", fault_type="leak", klass="Code", kind="gradual", event="failure", ts=380.0, params={}),
          dict(fault_id="b", fault_type="leak", klass="Code", kind="gradual", event="end", ts=420.0, params={}),
          dict(fault_id="c", fault_type="crash", klass="Service", kind="abrupt", event="start", ts=500.0, params={"down_s": 20}),
          dict(event="restart", ts=526.5, down_since=501.0)]
    (tmp_path / "fault_events.jsonl").write_text("\n".join(json.dumps(e) for e in ev), encoding="utf-8")
    rows = RG.build_schedule(tmp_path, dict(run_id="r1"))
    by = {r["fault_type"]: r for r in rows}
    assert by["latency"]["severity_level"] == "medium" and by["latency"]["severity"] == 0.5
    assert by["latency"]["start_ts"] == 100.0 and by["latency"]["end_ts"] == 160.0 and by["latency"]["failure_ts"] is None
    assert by["leak"]["failure_ts"] == 380.0 and by["leak"]["kind"] == "gradual"
    assert by["crash"]["end_ts"] == 526.5 and by["crash"]["duration_s"] == 26.5
    assert (tmp_path / "fault_schedule.csv").exists()
