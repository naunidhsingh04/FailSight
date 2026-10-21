"""Scripted 12-minute live demo of the testbed: a NEW run (seed >= 9000, never used for training or selection) through the real chain
shop + failsight_agent -> monitor (ingestion, window builder, FROZEN detector, alerts) with the fault schedule drawn and recorded BEFORE the run.

  python -m testbed.live_demo --scenario fault_free|abrupt|gradual|all [--minutes 12]

Per scenario: 1. the schedule is drawn from the seed and written (with a timestamp and hash) to report_outputs/testbed_live_schedule_<seed>.json, 2. the shop, the load generator and
the monitor run for 12 minutes (3+ minutes normal, then the fault, then recovery), 3. the monitor's per-window predictions and the app's own fault record are evaluated exactly as
they come out: detection delay, false alarms, misses. Nothing is tuned; a seed is never run twice (use --allow-rerun only to repeat a crashed attempt: it is counted in the CSV)."""
import argparse
import asyncio
import hashlib
import json
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from testbed import loadgen, run_generator as RG  # noqa: E402
from testbed.harness import Testbed  # noqa: E402
from testbed import live_eval as LE  # noqa: E402

REP = ROOT / "report_outputs"
CSV = REP / "testbed_live_validation.csv"
SEEDS = {"fault_free": 9001, "abrupt": 9002, "gradual": 9003}
ABRUPT = ["latency", "cpu_burn", "route_500", "db_slow", "db_lock", "exception_route", "crash"]
GRADUAL = ["latency_ramp", "leak", "pool_exhaustion"]
BACKEND = "http://127.0.0.1:8000"


def draw_schedule(scenario, seed, minutes):
    """everything about the demo, drawn from the seed alone"""
    rng = random.Random(seed)
    profile, rate, mix = rng.choice(loadgen.PROFILES), round(rng.uniform(3.0, 12.0), 2), loadgen.random_mix(seed)
    faults = []
    if scenario != "fault_free":
        ftype = rng.choice(ABRUPT if scenario == "abrupt" else GRADUAL)
        u = rng.random()
        dur = RG.fault_duration(ftype, u, rng)
        faults.append(dict(type=ftype, klass=RG.CATALOGUE[ftype][0], kind=RG.CATALOGUE[ftype][1], u=round(u, 4), level=RG.severity_level(u), duration=round(dur, 1),
                           params=RG.severity_params(ftype, u, dur, rng), start=round(rng.uniform(200, 250), 1)))
    return dict(scenario=scenario, seed=seed, minutes=minutes, duration_s=minutes * 60.0, profile=profile, rate=rate, mix=mix, faults=faults)


def write_schedule(plan):
    plan = dict(plan, written_at=time.strftime("%Y-%m-%d %H:%M:%S"), written_at_epoch=time.time())
    plan["sha256_of_plan"] = hashlib.sha256(json.dumps({k: v for k, v in plan.items() if k not in ("written_at", "written_at_epoch")}, sort_keys=True).encode()).hexdigest()
    p = REP / f"testbed_live_schedule_{plan['seed']}.json"
    if p.exists():
        sys.exit(f"{p} exists: the schedule of seed {plan['seed']} was already drawn and recorded")
    p.write_text(json.dumps(plan, indent=1), encoding="utf-8")
    return plan


def ensure_backend():
    try:
        if httpx.get(BACKEND + "/api/health", timeout=2).status_code == 200:
            return None
    except Exception:
        pass
    env = dict(os.environ, PORT="8000", FAILSIGHT_LOAD_DEEP="0", PYTHONPATH=str(ROOT))
    proc = subprocess.Popen([sys.executable, "-m", "app.backend.serve"], cwd=ROOT, env=env, stdout=open(ROOT / "outputs" / "live_backend.log", "ab"), stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 240:
        try:
            if httpx.get(BACKEND + "/api/testbed/state", timeout=2).status_code == 200:
                return proc
        except Exception:
            time.sleep(2)
    proc.kill()
    raise RuntimeError("the monitor did not start (see outputs/live_backend.log)")


def run_demo(plan, minutes, log=print):
    seed = plan["seed"]
    run_dir = ROOT / "testbed" / "runs" / f"_live_{seed}"
    if run_dir.exists():
        import shutil
        shutil.move(str(run_dir), str(ROOT / "testbed" / "runs" / f"_live_{seed}_attempt_{int(time.time())}"))
    cfg = json.loads((ROOT / "report_outputs" / "testbed_frozen_config.json").read_text(encoding="utf-8"))
    assert plan["written_at_epoch"] > cfg["frozen_at_epoch"], "the schedule must be drawn after the configuration was frozen"
    tb = Testbed(run_dir, seed=seed, shop_id="shop", monitor_url=BACKEND, log=log).start(with_recorder=False)
    time.sleep(2)
    anchor = time.time() + 1.0
    httpx.post(BACKEND + "/api/testbed/reset", json=dict(app_id="shop", start_ts=anchor), timeout=10).raise_for_status()
    stop = threading.Event()
    t_start = anchor

    def clock():
        for f in plan["faults"]:
            while time.time() - t_start < f["start"] and not stop.is_set():
                time.sleep(0.05)
            if stop.is_set():
                return
            log(f"  t+{time.time() - t_start:5.0f}s inject {f['type']} ({f['kind']}, {f['level']}) for {f['duration']:.0f}s {f['params']}")
            tb.inject(f["type"], f["duration"], dict(f["params"], severity=f["u"], severity_level=f["level"]))

    th = threading.Thread(target=clock, daemon=True)
    th.start()
    time.sleep(max(0.0, anchor - time.time()))
    started_at = time.time()
    stats, _ = asyncio.run(loadgen.run(f"http://127.0.0.1:{tb.sp}", plan["duration_s"], plan["profile"], plan["rate"], seed, mix=plan["mix"], stop=stop, log=None))
    stop.set()
    log("  load finished; waiting for the last windows")
    time.sleep(45)
    tb.stop()
    wins = httpx.get(BACKEND + "/api/testbed/windows", params=dict(app_id="shop"), timeout=30).json()
    return run_dir, anchor, started_at, wins, stats, tb


def finish(plan, run_dir, anchor, started_at, wins, stats, tb, cfg):
    meta = dict(plan, run_id=f"live_{plan['seed']}", start_ts=anchor, end_ts=time.time())
    sched = RG.build_schedule(run_dir, dict(plan, run_id=f"live_{plan['seed']}"))
    sched = pd.read_csv(run_dir / "fault_schedule.csv", dtype={"fault_id": str})
    sched.to_csv(REP / f"testbed_live_schedule_actual_{plan['seed']}.csv", index=False)
    wins = [w for w in wins if w["window_end"] <= anchor + plan["duration_s"] + 1e-6]              # windows after the end of the load are not part of the demo
    W = pd.DataFrame([{k: v for k, v in w.items() if k not in ("reasons", "features", "model")} | dict(top_reason=(w["reasons"][0]["sentence"] if w.get("reasons") else "")) for w in wins])
    L = LE.label_windows(W, sched, anchor)
    keep = ["window_index", "window_start", "window_end", "t_offset", "learning", "risk", "threshold", "flagged", "alert", "new_alert", "status", "robust_z_score", "robust_z_flag", "top_reason", "label_level",
            "fault_type", "severity_level", "overlap_s", "transition", "time_to_failure_s"]
    L[keep].to_csv(REP / f"testbed_live_windows_{plan['seed']}.csv", index=False)
    ev = LE.evaluate(L, sched)
    inst = ev.pop("instances")
    first = inst[0] if inst else {}
    planned = plan["faults"][0] if plan["faults"] else {}
    row = dict(scenario=plan["scenario"], seed=plan["seed"], frozen_at=cfg["frozen_at"], schedule_written_at=plan["written_at"], run_started_at=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started_at)),
               model=cfg["model"], feature_set=cfg["feature_set"], threshold=cfg["threshold"], alert_rule=cfg["alert_rule"]["name"], profile=plan["profile"], rate=plan["rate"], requests=stats.requests,
               planned_fault=planned.get("type", "none"), planned_severity=planned.get("level", ""), planned_start_s=planned.get("start", np.nan), planned_duration_s=planned.get("duration", np.nan),
               actual_fault=first.get("fault_type", "none"), actual_start_s=(first["start_ts"] - anchor) if inst else np.nan, actual_end_s=(first["end_ts"] - anchor) if inst else np.nan,
               failure_s=(first["failure_ts"] - anchor) if inst and pd.notna(first.get("failure_ts")) else np.nan, detected=first.get("detected", np.nan), detection_delay_s=first.get("delay_s", np.nan),
               warned_before_failure=first.get("warned_before_failure", np.nan), lead_time_s=first.get("lead_time_s", np.nan), shop_restarts=len(tb.restarts), attempt=1, **ev)
    row["false_alarm_window_end_s"] = json.dumps(row["false_alarm_window_end_s"])
    new = pd.DataFrame([row])
    if CSV.exists():
        old = pd.read_csv(CSV)
        if int(plan["seed"]) in set(old.seed):
            new["attempt"] = int(old[old.seed == plan["seed"]].attempt.max()) + 1
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(CSV, index=False)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=["fault_free", "abrupt", "gradual", "all"], default="all")
    ap.add_argument("--minutes", type=float, default=12.0)
    ap.add_argument("--allow-rerun", action="store_true")
    ap.add_argument("--keep-backend", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="plumbing test with a throw-away seed (9100); outputs go to outputs/live_smoke, nothing is added to the validation")
    a = ap.parse_args()
    global REP, CSV
    if a.smoke:
        REP = ROOT / "outputs" / "live_smoke"
        REP.mkdir(parents=True, exist_ok=True)
        CSV = REP / "testbed_live_validation.csv"
        SEEDS["abrupt"] = 9100
    cfg = json.loads((ROOT / "report_outputs" / "testbed_frozen_config.json").read_text(encoding="utf-8"))
    import testbed.freeze as FZ
    assert FZ.check(), "the frozen model file does not match the frozen configuration"
    scenarios = list(SEEDS) if a.scenario == "all" else [a.scenario]
    for sc in scenarios:
        if SEEDS[sc] < 9000:
            sys.exit("live demo seeds must be 9000 or above")
    if sys.platform == "win32":
        RG.keep_awake()
    if RG.on_battery():
        sys.exit("plug the laptop in first")
    proc = ensure_backend()
    try:
        for sc in scenarios:
            seed = SEEDS[sc]
            if a.allow_rerun and (REP / f"testbed_live_schedule_{seed}.json").exists():
                (REP / f"testbed_live_schedule_{seed}.json").rename(REP / f"testbed_live_schedule_{seed}.attempt1.json")
            plan = write_schedule(draw_schedule(sc, seed, a.minutes))
            print(f"[{time.strftime('%H:%M:%S')}] {sc} (seed {seed}): schedule recorded at {plan['written_at']} before the run: " + (", ".join(f"{f['type']} [{f['level']}] at {f['start']:.0f}s for {f['duration']:.0f}s {f['params']}" for f in plan["faults"]) or "no fault"), flush=True)
            run_dir, anchor, started_at, wins, stats, tb = run_demo(plan, a.minutes, log=lambda s: print(s, flush=True))
            row = finish(plan, run_dir, anchor, started_at, wins, stats, tb, cfg)
            print(f"    result: detected={row['detected']} delay={row['detection_delay_s']} s, false alarms={row['false_alarms']} ({row['false_alarms_per_hour']:.1f}/h), missed fault windows {row['missed_fault_windows']}/{row['fault_windows']}", flush=True)
    finally:
        if proc is not None and not a.keep_backend:
            proc.terminate()


if __name__ == "__main__":
    main()
