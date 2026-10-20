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


def _fake_run(d, dur=600.0, wall=601.0, span=599.0, gap_at=None, late=0.0, crash=False):
    d.mkdir(parents=True, exist_ok=True)
    t0 = 1_000_000.0
    api = [{"timestamp": t0 + i * 5.0} for i in range(int(span / 5) + 1)]
    (d / "api.jsonl").write_text("\n".join(json.dumps(e) for e in api), encoding="utf-8")
    (d / "logs.jsonl").write_text("", encoding="utf-8")
    mt = [{"timestamp": t0 + i * 2.0, "metric": "process_threads"} for i in range(int(dur / 2)) if gap_at is None or not (gap_at <= i * 2.0 < gap_at + 100)]
    (d / "metrics.jsonl").write_text("\n".join(json.dumps(e) for e in mt), encoding="utf-8")
    faults = [dict(type="latency", start=200.0, duration=60.0)]
    (d / "meta.json").write_text(json.dumps(dict(duration_s=dur, start_ts=t0, end_ts=t0 + wall, faults=faults)), encoding="utf-8")
    (d / "fault_schedule.csv").write_text(f"run_id,fault_id,fault_type,klass,kind,severity_level,severity,params,start_ts,end_ts,failure_ts,duration_s\nr,a,latency,Performance,abrupt,mild,0.1,{{}},{t0 + 200 + late},{t0 + 260 + late},,60\n", encoding="utf-8")
    (d / "DONE").write_text("x", encoding="utf-8")


def test_validate_run_accepts_a_clean_run_and_rejects_suspended_or_stalled_ones(tmp_path):
    _fake_run(tmp_path / "ok")
    assert RG.validate_run(tmp_path / "ok") == []
    _fake_run(tmp_path / "wall", wall=21868.0)
    assert any("wall time" in p for p in RG.validate_run(tmp_path / "wall"))
    _fake_run(tmp_path / "gap", gap_at=100.0)
    assert any("no process metrics" in p for p in RG.validate_run(tmp_path / "gap"))
    _fake_run(tmp_path / "late", late=31.0)
    assert any("injected" in p for p in RG.validate_run(tmp_path / "late"))
    _fake_run(tmp_path / "short", span=540.0)
    assert any("span only" in p for p in RG.validate_run(tmp_path / "short"))
    (tmp_path / "ok" / "meta.json").unlink()
    assert RG.validate_run(tmp_path / "ok") == ["missing meta.json"]


def test_quarantine_keeps_the_data_and_drops_the_manifest_row(tmp_path, monkeypatch):
    monkeypatch.setattr(RG, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(RG, "MANIFEST", tmp_path / "manifest.csv")
    (tmp_path / "manifest.csv").write_text("run_id,seed\nrA,1\nrB,2\n", encoding="utf-8")
    (tmp_path / "rB").mkdir()
    (tmp_path / "rB" / "api.jsonl").write_text("x", encoding="utf-8")
    RG.quarantine(tmp_path / "rB")
    assert not (tmp_path / "rB").exists() and list((tmp_path / "_invalid").glob("rB_*/api.jsonl"))
    assert (tmp_path / "manifest.csv").read_text(encoding="utf-8").split() == ["run_id,seed", "rA,1"]
