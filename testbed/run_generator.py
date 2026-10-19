"""Run generator: produces many labelled testbed runs, resumable, sequential (one run at a time so runs do not disturb each other on an 8 GB laptop).

Every run: a fresh shop + recorder, a random traffic profile / rate / user mix (loadgen), and a random schedule of 0-3 real faults injected into the running app. The raw
events (api.jsonl, logs.jsonl, metrics.jsonl exactly as the agent sent them), fault_events.jsonl, fault_schedule.csv (exact start / end / failure timestamps) and meta.json
are written to testbed/runs/<run_id>/. Only fault_schedule.csv, meta.json and testbed/runs/manifest.csv are tracked by git.

  python -m testbed.run_generator --runs 40 --minutes 10        # estimate is printed first; rerun the same command to resume after an interruption
  python -m testbed.run_generator --runs 40 --dry               # print the plan (profiles, rates, fault schedules) without running anything
"""
import argparse
import asyncio
import csv
import json
import random
import shutil
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from testbed import loadgen  # noqa: E402
from testbed.harness import Testbed  # noqa: E402
from testbed.shop.faults import CATALOGUE  # noqa: E402

RUNS_DIR = ROOT / "testbed" / "runs"
MANIFEST = RUNS_DIR / "manifest.csv"
OVERHEAD_S = 30                       # start-up, shutdown and post-processing per run (measured on the development laptop)
LEARN_S = 150                         # the first 2.5 minutes of every run are fault-free (a per-app detector learns its normal from them)
END_MARGIN_S = 30
GAP_S = 45
N_FAULTS_P = {0: 0.25, 1: 0.30, 2: 0.30, 3: 0.15}
FAULT_WEIGHTS = {"latency": 1, "cpu_burn": 1, "route_500": 1, "crash": 0.7, "db_slow": 1, "db_lock": 1, "exception_route": 1, "latency_ramp": 1.2, "leak": 1.2, "pool_exhaustion": 1.2}


def plan_run(index, seed, minutes):
    """everything random about one run, derived from its seed alone (so a run can be re-planned and checked)"""
    rng = random.Random(seed)
    duration = minutes * 60.0
    profile = rng.choice(loadgen.PROFILES)
    rate = round(rng.uniform(3.0, 12.0), 2)
    mix = loadgen.random_mix(seed)
    n = rng.choices(list(N_FAULTS_P), list(N_FAULTS_P.values()))[0]
    faults, tries = [], 0
    while len(faults) < n and tries < 400:
        tries += 1
        ftype = rng.choices(list(FAULT_WEIGHTS), list(FAULT_WEIGHTS.values()))[0]
        klass, kind = CATALOGUE[ftype]
        down_s = round(rng.uniform(20, 45))
        dur = rng.uniform(120, 220) if kind == "gradual" else (down_s + 8.0 if ftype == "crash" else rng.uniform(40, 110))
        lo, hi = LEARN_S, duration - END_MARGIN_S - dur
        if hi <= lo:
            continue
        start = rng.uniform(lo, hi)
        if any(start < f["start"] + f["duration"] + GAP_S and f["start"] < start + dur + GAP_S for f in faults):
            continue
        params = {}
        if ftype == "crash":
            params = dict(down_s=down_s)                                                         # the supervisor restarts the crashed shop after this many seconds
        elif ftype == "latency":
            params = dict(added_ms=rng.choice([250, 400, 700]))
        elif ftype == "cpu_burn":
            params = dict(threads=rng.choice([1, 2, 3]), duty=rng.choice([0.7, 0.9]))
        elif ftype == "route_500":
            params = dict(route=rng.choice(["search", "book", "pay"]), prob=rng.choice([0.5, 0.8]))
        elif ftype == "db_slow":
            params = dict(extra_ms=rng.choice([200, 350, 600]))
        elif ftype == "exception_route":
            params = dict(route=rng.choice(["search", "book", "pay", "cancel"]), prob=rng.choice([0.6, 1.0]))
        elif ftype == "latency_ramp":
            params = dict(ramp_s=round(dur * 0.8), max_ms=2500)
        elif ftype == "leak":
            params = dict(mb_per_s=round(220 / (dur * rng.uniform(0.6, 0.9)), 2))               # reaches the memory cap at 60-90% of the fault
        elif ftype == "pool_exhaustion":
            params = dict(every_s=round(dur * rng.uniform(0.6, 0.85) / 8, 1))                    # the 8-connection pool is empty at 60-85% of the fault
        faults.append(dict(type=ftype, klass=klass, kind=kind, start=round(start, 1), duration=round(dur, 1), params=params))
    faults.sort(key=lambda f: f["start"])
    return dict(run_id=f"r{index:03d}_s{seed}", index=index, seed=seed, minutes=minutes, duration_s=duration, profile=profile, rate=rate, mix=mix, faults=faults)


def estimate(plans, minutes):
    per = minutes * 60 + OVERHEAD_S
    return len(plans) * per / 3600.0


def build_schedule(run_dir, plan):
    """fault_schedule.csv from the app's own fault_events.jsonl: exact start / end / failure timestamps (epoch and ISO)"""
    ev = [json.loads(line) for line in open(run_dir / "fault_events.jsonl", encoding="utf-8")] if (run_dir / "fault_events.jsonl").exists() else []
    faults = {}
    order = []
    for e in ev:
        fid = e.get("fault_id")
        if e["event"] == "start":
            faults[fid] = dict(fault_id=fid, fault_type=e["fault_type"], klass=e["klass"], kind=e["kind"], params=json.dumps(e["params"]), start_ts=e["ts"], end_ts=None, failure_ts=None)
            order.append(fid)
        elif e["event"] == "end" and fid in faults:
            faults[fid]["end_ts"] = e["ts"]
        elif e["event"] == "failure" and fid in faults:
            faults[fid]["failure_ts"] = e["ts"]
        elif e["event"] == "restart":                                  # the crashed shop is healthy again: the end of the crash fault
            for f in faults.values():
                if f["fault_type"] == "crash" and f["end_ts"] is None:
                    f["end_ts"] = e["ts"]
    rows = []
    for fid in order:
        f = faults[fid]
        f["run_id"] = plan["run_id"]
        f["duration_s"] = round((f["end_ts"] - f["start_ts"]), 2) if f["end_ts"] else None
        rows.append(f)
    cols = ["run_id", "fault_id", "fault_type", "klass", "kind", "params", "start_ts", "end_ts", "failure_ts", "duration_s"]
    with open(run_dir / "fault_schedule.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})
    return rows


def count_lines(p):
    return sum(1 for _ in open(p, encoding="utf-8")) if p.exists() else 0


def execute(plan, log):
    run_dir = RUNS_DIR / plan["run_id"]
    if run_dir.exists():
        shutil.rmtree(run_dir)                                         # a partial run from an interrupted attempt is redone from scratch
    tb = Testbed(run_dir, seed=plan["seed"], log=log).start()
    t_start = time.time()
    stop = threading.Event()

    def fault_clock():
        for f in plan["faults"]:
            while time.time() - t_start < f["start"] and not stop.is_set():
                time.sleep(0.05)
            if stop.is_set():
                return
            log(f"  t+{time.time() - t_start:5.0f}s inject {f['type']} ({f['kind']}) for {f['duration']:.0f}s {f['params']}")
            tb.inject(f["type"], f["duration"], f["params"])

    th = threading.Thread(target=fault_clock, daemon=True)
    th.start()
    stats, info = asyncio.run(loadgen.run(f"http://127.0.0.1:{tb.sp}", plan["duration_s"], plan["profile"], plan["rate"], plan["seed"], mix=plan["mix"], stop=stop, log=None))
    t_end = time.time()
    stop.set()
    time.sleep(3)                                                      # let the agent flush its last batches
    tb.stop()
    rows = build_schedule(run_dir, plan)
    meta = dict(plan, start_ts=t_start, end_ts=t_end, load=dict(requests=stats.requests, by_status={str(k): v for k, v in stats.by_status.items()}, client_errors=stats.client_errors, sessions=stats.sessions, shed=stats.shed),
                events=dict(api=count_lines(run_dir / "api.jsonl"), logs=count_lines(run_dir / "logs.jsonl"), metrics=count_lines(run_dir / "metrics.jsonl")),
                shop_restarts=len(tb.restarts), faults_recorded=len(rows), python=sys.version.split()[0])
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    (run_dir / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    new = not MANIFEST.exists()
    with open(MANIFEST, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["run_id", "seed", "profile", "rate", "duration_s", "n_faults", "fault_types", "start_ts", "requests", "api_events", "log_events", "metric_events", "shop_restarts"])
        w.writerow([plan["run_id"], plan["seed"], plan["profile"], plan["rate"], plan["duration_s"], len(rows), "+".join(r["fault_type"] for r in rows), round(t_start, 3), stats.requests,
                    meta["events"]["api"], meta["events"]["logs"], meta["events"]["metrics"], len(tb.restarts)])
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--seed0", type=int, default=1000, help="run i uses seed seed0 + i; seeds >= 9000 are reserved for the live demo and never used for training")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if a.seed0 + a.runs > 9000:
        sys.exit("seeds >= 9000 are reserved for the live demo")
    plans = [plan_run(i, a.seed0 + i, a.minutes) for i in range(a.runs)]
    todo = [p for p in plans if not (RUNS_DIR / p["run_id"] / "DONE").exists()]
    n_faultfree = sum(1 for p in plans if not p["faults"])
    print(f"plan: {len(plans)} runs of {a.minutes:g} min, {len(todo)} still to do, {n_faultfree} runs without any fault, {sum(len(p['faults']) for p in plans)} faults in total")
    cnt = {}
    for p in plans:
        for f in p["faults"]:
            cnt[f["type"]] = cnt.get(f["type"], 0) + 1
    print("faults by type:", dict(sorted(cnt.items())))
    print(f"ESTIMATED TIME: {estimate(todo, a.minutes):.1f} hours for the {len(todo)} remaining runs ({a.minutes:g} min + about {OVERHEAD_S} s overhead each, one run at a time); disk about {len(todo) * 3} MB")
    if a.dry:
        for p in plans:
            print(f"  {p['run_id']:<14} {p['profile']:<7} rate {p['rate']:>5}  " + ("; ".join(f"{f['type']}@{f['start']:.0f}s+{f['duration']:.0f}s" for f in p["faults"]) or "no fault"))
        return
    if sys.platform == "win32":
        print("keep the laptop awake for the whole run: powercfg /change standby-timeout-ac 0   and   powercfg /change monitor-timeout-ac 0   (undo afterwards)")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for k, p in enumerate(todo):
        print(f"[{time.strftime('%H:%M:%S')}] run {k + 1}/{len(todo)}: {p['run_id']} {p['profile']} rate {p['rate']} faults: " + ("; ".join(f"{f['type']}@{f['start']:.0f}s" for f in p["faults"]) or "none"), flush=True)
        try:
            m = execute(p, lambda s: print(s, flush=True))
            print(f"    done: {m['load']['requests']} requests, events {m['events']}, faults recorded {m['faults_recorded']}", flush=True)
        except Exception as e:                                          # keep going: the run is retried the next time the script is started
            print(f"    FAILED: {e!r}", flush=True)
        left = (len(todo) - k - 1) * (a.minutes * 60 + OVERHEAD_S)
        print(f"    elapsed {(time.time() - t0) / 3600:.2f} h, about {left / 3600:.2f} h left", flush=True)


if __name__ == "__main__":
    main()
