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
FAULT_FREE_SHARE = 0.25               # about a quarter of the runs have no fault at all
MAX_FAULTS_PER_RUN = 3
MIN_PER_TYPE_FRACTION = 0.1           # every fault type gets at least runs * 0.1 instances (4 in 40 runs); a few extra instances are added at random
EXTRA_FRACTION = 0.15
TYPE_ORDER = list(CATALOGUE)


def severity_level(u):
    return "mild" if u < 1 / 3 else "medium" if u < 2 / 3 else "severe"


def severity_params(ftype, u, dur, rng):
    """map the severity u in [0, 1] (drawn from the run's seed) to the parameters of one fault instance: mild -> severe"""
    if ftype == "latency":
        return dict(added_ms=round(30 + u * 770))                                              # +30 ms ... +800 ms on every request
    if ftype == "cpu_burn":
        return dict(threads=1, duty=round(0.2 + 0.8 * u, 2))                                   # 20% ... 100% of one core
    if ftype == "route_500":
        return dict(route=rng.choice(["search", "book", "pay"]), prob=round(0.05 + 0.95 * u, 2))   # 5% ... 100% of the requests of one route fail
    if ftype == "exception_route":
        return dict(route=rng.choice(["search", "book", "pay", "cancel"]), prob=round(0.05 + 0.95 * u, 2))
    if ftype == "db_slow":
        return dict(extra_ms=round(30 + u * 770))                                              # +30 ms ... +800 ms of real database work per request
    if ftype == "db_lock":
        return dict(lock_s=round(dur))                                                         # the write lock is held for the whole fault (15 ... 100 s)
    if ftype == "crash":
        return dict(down_s=round(dur - 8))                                                     # the crashed shop stays down 20 ... 45 s before the supervisor restarts it
    if ftype == "latency_ramp":
        return dict(ramp_s=round(dur * 0.8), max_ms=round(1700 + u * 1500))                    # grows to 1.7 ... 3.2 s; requests time out (the failure) at 1.5 s
    if ftype == "leak":
        return dict(mb_per_s=round(220 / (dur * (0.9 - 0.5 * u)), 2))                          # the 220 MB cap is reached at 90% (mild) ... 40% (severe) of the fault
    if ftype == "pool_exhaustion":
        return dict(every_s=round(dur * (0.9 - 0.5 * u) / 8, 1))                               # the 8-connection pool is empty at 90% (mild) ... 40% (severe) of the fault
    raise ValueError(ftype)


def fault_duration(ftype, u, rng):
    klass, kind = CATALOGUE[ftype]
    if kind == "gradual":
        return rng.uniform(120, 220)
    if ftype == "crash":
        return 8.0 + round(20 + 25 * u)                                                         # down_s 20 ... 45 s (+ 8 s until the shop is healthy again)
    if ftype == "db_lock":
        return 15 + 85 * u                                                                      # lock duration 15 ... 100 s
    return rng.uniform(40, 110)


def plan_all(n_runs, seed0, minutes):
    """The whole plan as a function of (n_runs, seed0, minutes) only: balanced fault types (each at least 10% of the number of runs), fault-free runs, severities, positions.
    Per-run traffic (profile, rate, mix) comes from each run's own seed seed0 + i."""
    duration = minutes * 60.0
    rng = random.Random(seed0 * 7919 + n_runs)
    n_free = round(n_runs * FAULT_FREE_SHARE)
    free = set(rng.sample(range(n_runs), n_free))
    fault_runs = [i for i in range(n_runs) if i not in free]
    per_type = max(int(n_runs * MIN_PER_TYPE_FRACTION), 1)
    types = [t for t in TYPE_ORDER for _ in range(per_type)]
    extra = round(n_runs * EXTRA_FRACTION)
    types += rng.choices(TYPE_ORDER, k=extra)
    instances = []
    thirds = {t: rng.sample([0, 1, 2], 3) for t in TYPE_ORDER}        # severity is stratified within each fault type (mild / medium / severe in rotation), then drawn at random inside its third
    seen = {t: 0 for t in TYPE_ORDER}
    for k, t in enumerate(types):
        u = (thirds[t][seen[t] % 3] + rng.random()) / 3
        seen[t] += 1
        dur = fault_duration(t, u, rng)
        instances.append(dict(type=t, klass=CATALOGUE[t][0], kind=CATALOGUE[t][1], u=round(u, 4), level=severity_level(u), duration=round(dur, 1), params=severity_params(t, u, dur, rng)))
    capacity = duration - LEARN_S - END_MARGIN_S
    rng.shuffle(instances)
    instances.sort(key=lambda f: -f["duration"])                                                # place the long ones first
    load = {i: [] for i in fault_runs}
    for f in instances:
        ok = [i for i in fault_runs if len(load[i]) < MAX_FAULTS_PER_RUN and sum(x["duration"] for x in load[i]) + f["duration"] + GAP_S * len(load[i]) <= capacity]
        if not ok:
            raise RuntimeError("the fault plan does not fit; use more or longer runs")
        low = min(len(load[i]) for i in ok)
        i = rng.choice([i for i in ok if len(load[i]) == low])
        load[i].append(f)
    plans = []
    for i in range(n_runs):
        seed = seed0 + i
        r = random.Random(seed)
        profile, rate, mix = r.choice(loadgen.PROFILES), round(r.uniform(3.0, 12.0), 2), loadgen.random_mix(seed)
        fs = list(load.get(i, []))
        rng.shuffle(fs)
        slack = capacity - sum(f["duration"] for f in fs) - GAP_S * max(len(fs) - 1, 0)
        cuts = sorted(rng.uniform(0, slack) for _ in range(len(fs) + 1))
        gaps = [cuts[0]] + [cuts[k] - cuts[k - 1] for k in range(1, len(fs) + 1)]                # lead-in, extra gaps between faults (the tail is what remains)
        t, out = LEARN_S + gaps[0], []
        for k, f in enumerate(fs):
            out.append(dict(f, start=round(t, 1)))
            t += f["duration"] + GAP_S + (gaps[k + 1] if k + 1 < len(gaps) else 0)
        plans.append(dict(run_id=f"r{i:03d}_s{seed}", index=i, seed=seed, minutes=minutes, duration_s=duration, profile=profile, rate=rate, mix=mix, faults=out))
    return plans


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
            faults[fid] = dict(fault_id=fid, fault_type=e["fault_type"], klass=e["klass"], kind=e["kind"], params=json.dumps(e["params"]), severity=e["params"].get("severity"), severity_level=e["params"].get("severity_level"), start_ts=e["ts"], end_ts=None, failure_ts=None)
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
    cols = ["run_id", "fault_id", "fault_type", "klass", "kind", "severity_level", "severity", "params", "start_ts", "end_ts", "failure_ts", "duration_s"]
    with open(run_dir / "fault_schedule.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})
    return rows


def count_lines(p):
    return sum(1 for _ in open(p, encoding="utf-8")) if p.exists() else 0


REQUIRED_FILES = ["api.jsonl", "logs.jsonl", "metrics.jsonl", "fault_schedule.csv", "meta.json", "DONE"]


def validate_run(run_dir):
    """problems that make a finished run unusable as training data (an empty list = valid): missing files, a run that is truncated or stretched by a suspended
    laptop (wall time, event span), holes in the 2 s process-metric samples outside a crash, or a fault injected late (the planned time was missed)"""
    run_dir = Path(run_dir)
    problems = [f"missing {f}" for f in REQUIRED_FILES if not (run_dir / f).exists()]
    if problems:
        return problems
    m = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    dur = m["duration_s"]
    wall = m["end_ts"] - m["start_ts"]
    if not (dur - 5 <= wall <= dur + 90):
        problems.append(f"wall time {wall:.0f} s for a planned {dur:.0f} s (suspended or stalled machine)")
    api_ts = [json.loads(line)["timestamp"] for line in open(run_dir / "api.jsonl", encoding="utf-8")]
    span = (max(api_ts) - min(api_ts)) if api_ts else 0
    if span < 0.95 * dur:
        problems.append(f"request events span only {span:.0f} s of {dur:.0f} s")
    sched = []
    with open(run_dir / "fault_schedule.csv", encoding="utf-8") as fh:
        sched = list(csv.DictReader(fh))
    if len(sched) != len(m["faults"]):
        problems.append(f"{len(sched)} faults recorded, {len(m['faults'])} planned")
    allowed = []                                                     # a crashed shop sends no metrics while it is down
    for r in sched:
        if r["fault_type"] == "crash" and r["end_ts"]:
            allowed.append((float(r["start_ts"]) - 3, float(r["end_ts"]) + 12))
    mt = sorted(json.loads(line)["timestamp"] for line in open(run_dir / "metrics.jsonl", encoding="utf-8") if '"process_threads"' in line)
    for a, b in zip(mt, mt[1:]):
        if b - a > 12 and not any(lo <= a and b <= hi for lo, hi in allowed):
            problems.append(f"no process metrics for {b - a:.0f} s at +{a - m['start_ts']:.0f} s (machine stalled)")
            break
    for plan_f, r in zip(sorted(m["faults"], key=lambda f: f["start"]), sorted(sched, key=lambda r: float(r["start_ts"]))):
        late = float(r["start_ts"]) - (m["start_ts"] + plan_f["start"])
        if abs(late) > 5:
            problems.append(f"{plan_f['type']} injected {late:+.0f} s from its planned time")
    return problems


def drop_manifest_row(run_id):
    if not MANIFEST.exists():
        return
    rows = list(csv.reader(open(MANIFEST, encoding="utf-8")))
    keep = [rows[0]] + [r for r in rows[1:] if r and r[0] != run_id]
    with open(MANIFEST, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(keep)


def quarantine(run_dir):
    """keep the data of an invalid run for inspection (testbed/runs/_invalid/...), remove it from the manifest, free the name for a clean redo"""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return
    dst = RUNS_DIR / "_invalid" / f"{run_dir.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_dir), str(dst))
    drop_manifest_row(run_dir.name)


def keep_awake():
    """Windows: tell the system not to go to sleep / Modern Standby while this process runs (it does not help against a critical battery or a closed lid)"""
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)       # ES_CONTINUOUS | ES_SYSTEM_REQUIRED


def on_battery():
    try:
        import psutil
        b = psutil.sensors_battery()
        return b is not None and not b.power_plugged
    except Exception:
        return False


def execute(plan, log):
    run_dir = RUNS_DIR / plan["run_id"]
    if run_dir.exists():
        quarantine(run_dir)                                            # a partial or invalid run from an interrupted attempt is kept aside and redone from scratch
    tb = Testbed(run_dir, seed=plan["seed"], log=log).start()
    t_start = time.time()
    stop = threading.Event()

    def fault_clock():
        for f in plan["faults"]:
            while time.time() - t_start < f["start"] and not stop.is_set():
                time.sleep(0.05)
            if stop.is_set():
                return
            log(f"  t+{time.time() - t_start:5.0f}s inject {f['type']} ({f['kind']}, {f['level']}) for {f['duration']:.0f}s {f['params']}")
            tb.inject(f["type"], f["duration"], dict(f["params"], severity=f["u"], severity_level=f["level"]))

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
    problems = validate_run(run_dir)
    if problems:                                                       # never keep a run that the laptop interrupted: it is redone the next time the script starts
        quarantine(run_dir)
        raise RuntimeError("run invalid, set aside and not counted: " + "; ".join(problems))
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
    try:
        plans = plan_all(a.runs, a.seed0, a.minutes)
    except RuntimeError:
        sys.exit(f"{a.runs} runs of {a.minutes:g} minutes are too few to hold every fault type without overlap: use at least 12 runs (the full plan is 40)")
    todo, invalid = [], {}
    for p in plans:
        d = RUNS_DIR / p["run_id"]
        if (d / "DONE").exists():
            probs = validate_run(d)
            if probs:
                invalid[p["run_id"]] = probs
                todo.append(p)
        else:
            todo.append(p)
    for rid, probs in invalid.items():
        print(f"finished run {rid} is INVALID and will be redone: " + "; ".join(probs))
    n_faultfree = sum(1 for p in plans if not p["faults"])
    print(f"plan: {len(plans)} runs of {a.minutes:g} min, {len(todo)} still to do, {n_faultfree} runs without any fault, {sum(len(p['faults']) for p in plans)} faults in total")
    print("\nfault instances by type and severity level (all runs):")
    print(f"  {'fault type':<17}{'class':<13}{'kind':<9}{'mild':>5}{'medium':>8}{'severe':>8}{'total':>7}")
    for t in TYPE_ORDER:
        fs = [f for p in plans for f in p["faults"] if f["type"] == t]
        print(f"  {t:<17}{CATALOGUE[t][0]:<13}{CATALOGUE[t][1]:<9}" + "".join(f"{sum(1 for f in fs if f['level'] == lv):>{w}}" for lv, w in (("mild", 5), ("medium", 8), ("severe", 8))) + f"{len(fs):>7}")
    print()
    print(f"ESTIMATED TIME: {estimate(todo, a.minutes):.1f} hours for the {len(todo)} remaining runs ({a.minutes:g} min + about {OVERHEAD_S} s overhead each, one run at a time); disk about {len(todo) * 3} MB")
    if a.dry:
        for p in plans:
            print(f"  {p['run_id']:<14} {p['profile']:<7} rate {p['rate']:>5}  " + ("; ".join(f"{f['type']}[{f['level']}]@{f['start']:.0f}s+{f['duration']:.0f}s" for f in p["faults"]) or "no fault"))
        return
    if sys.platform == "win32":
        print("keep the laptop awake for the whole run: powercfg /change standby-timeout-ac 0   and   powercfg /change monitor-timeout-ac 0   (undo afterwards)")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if on_battery():
        sys.exit("the laptop is running on battery: plug it in (a critical battery puts it to sleep and corrupts the run), then start again")
    keep_awake()
    t0 = time.time()
    for k, p in enumerate(todo):
        print(f"[{time.strftime('%H:%M:%S')}] run {k + 1}/{len(todo)}: {p['run_id']} {p['profile']} rate {p['rate']} faults: " + ("; ".join(f"{f['type']}[{f['level']}]@{f['start']:.0f}s" for f in p["faults"]) or "none"), flush=True)
        try:
            m = execute(p, lambda s: print(s, flush=True))
            print(f"    done: {m['load']['requests']} requests, events {m['events']}, faults recorded {m['faults_recorded']}", flush=True)
        except Exception as e:                                          # keep going: the run is retried the next time the script is started
            print(f"    FAILED: {e!r}", flush=True)
        per_run = (time.time() - t0) / (k + 1)                       # measured average, including start-up and shutdown
        left = (len(todo) - k - 1) * per_run
        print(f"    elapsed {(time.time() - t0) / 3600:.2f} h, estimated time left {left / 3600:.2f} h (finish about {time.strftime('%H:%M', time.localtime(time.time() + left))})", flush=True)


if __name__ == "__main__":
    main()
