"""Fault self-test: injects every fault into a real running shop under steady load and measures, from the events the agent recorded, that the app's behaviour
really changed (error share, latency, process CPU / memory, loop lag, pool wait). Writes testbed/runs/_fault_check/summary.csv (not committed).

python -m testbed.check_faults [--quick]"""
import argparse
import asyncio
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from testbed import loadgen  # noqa: E402
from testbed.harness import Testbed  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

ABRUPT = [("latency", dict(added_ms=400)), ("cpu_burn", dict(threads=2, duty=0.9)), ("route_500", dict(route="book", prob=0.8)), ("db_slow", dict(extra_ms=300)),
          ("db_lock", {}), ("exception_route", dict(route="search", prob=1.0)), ("crash", dict(down_s=12))]
GRADUAL = [("latency_ramp", dict(ramp_s=50, max_ms=2500), 80), ("leak", dict(mb_per_s=6.0), 80), ("pool_exhaustion", dict(every_s=6), 80)]


def window_stats(d, t0, t1):
    api = pd.DataFrame(d["api"])
    m = pd.DataFrame(d["metrics"])
    a = api[(api.timestamp >= t0) & (api.timestamp < t1)]
    out = dict(requests=len(a), error_share=float((a.status_code >= 500).mean()) if len(a) else np.nan, latency_p95_ms=float(a.latency_ms.quantile(0.95)) if len(a) else np.nan)
    for name, key in (("process_cpu_percent", "cpu_pct"), ("process_rss_bytes", "rss_mb"), ("event_loop_lag_ms", "lag_ms"), ("db_pool_wait_ms", "pool_wait_ms")):
        x = m[(m.metric == name) & (m.timestamp >= t0) & (m.timestamp < t1)].value
        out[key] = float(x.mean() / 1e6 if key == "rss_mb" else x.mean()) if len(x) else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="abrupt faults only")
    a = ap.parse_args()
    d = ROOT / "testbed" / "runs" / "_fault_check"
    for f in d.glob("*.jsonl"):
        f.unlink()
    tb = Testbed(d, seed=3, log=print).start()
    rows, marks = [], []
    stop = threading.Event()
    total = 30 + len(ABRUPT) * 40 + (0 if a.quick else len(GRADUAL) * 105)
    th = threading.Thread(target=lambda: asyncio.run(loadgen.run(f"http://127.0.0.1:{tb.sp}", total, "steady", 8.0, 3, stop=stop, log=print)), daemon=True)
    th.start()
    t_base = time.time()
    time.sleep(30)
    marks.append(("baseline", t_base + 5, time.time()))
    for ftype, params in ABRUPT:
        t0 = time.time()
        tb.inject(ftype, 20, params)
        time.sleep(22 if ftype != "crash" else 22)
        marks.append((ftype, t0 + 2, t0 + 20))
        time.sleep(18)
    if not a.quick:
        for ftype, params, dur in GRADUAL:
            t0 = time.time()
            tb.inject(ftype, dur, params)
            time.sleep(dur + 2)
            marks.append((ftype + " (whole fault)", t0, t0 + dur))
            time.sleep(15)
    stop.set()
    th.join(timeout=20)
    time.sleep(3)
    tb.stop()
    d_ = {k: [json.loads(line) for line in open(d / f"{k}.jsonl", encoding="utf-8")] for k in ("api", "metrics")}
    for name, a0, a1 in marks:
        rows.append(dict(period=name, **window_stats(d_, a0, a1)))
    df = pd.DataFrame(rows)
    df.to_csv(d / "summary.csv", index=False)
    print(df.round(2).to_string(index=False))
    ev = [json.loads(line) for line in open(d / "fault_events.jsonl", encoding="utf-8")]
    print("\nfault events:")
    for e in ev:
        print(" ", e.get("fault_type", e.get("event")), e["event"], round(e["ts"] - t_base, 1), "s", {k: v for k, v in e.items() if k in ("leaked_mb", "added_ms", "held", "down_since")})


if __name__ == "__main__" and "--mild" not in sys.argv:
    main()


# ------------------------------------------------------------------------------------------------ mild severity check
def mild_check():
    """inject every fault type at MILD severity (u = 0.02, the mildest the run generator draws) and measure from the recorded events whether it is still visible:
    z = (mean inside the fault - mean in the fault-free start) / sd of 5 s bins in the fault-free start (floors: 1 ms, 0.5 MB, 0.01 error share)"""
    import random
    from testbed import run_generator as RG
    u = 0.02
    rng = random.Random(5)
    d = ROOT / "testbed" / "runs" / "_fault_check_mild"
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.jsonl"):
        f.unlink()
    plan = []
    for ft, dur in [("latency", 30), ("cpu_burn", 30), ("route_500", 30), ("db_slow", 30), ("db_lock", 15), ("exception_route", 30), ("crash", 28),
                    ("latency_ramp", 100), ("leak", 100), ("pool_exhaustion", 100)]:
        params = RG.severity_params(ft, u, dur, rng)
        if "route" in params:
            params["route"] = "search"
        plan.append((ft, dur, params))
    total = 40 + sum(p[1] + 14 for p in plan)
    tb = Testbed(d, seed=5, log=print).start()
    stop = threading.Event()
    th = threading.Thread(target=lambda: asyncio.run(loadgen.run(f"http://127.0.0.1:{tb.sp}", total, "steady", 8.0, 5, stop=stop, log=print)), daemon=True)
    th.start()
    t_base = time.time()
    time.sleep(40)
    marks = []
    for ft, dur, params in plan:
        t0 = time.time()
        tb.inject(ft, dur, dict(params, severity=u, severity_level="mild"))
        time.sleep(dur + 2)
        marks.append((ft, params, t0, t0 + dur))
        time.sleep(12)
    stop.set()
    th.join(timeout=20)
    time.sleep(3)
    tb.stop()
    api = pd.DataFrame([json.loads(line) for line in open(d / "api.jsonl", encoding="utf-8")])
    met = pd.DataFrame([json.loads(line) for line in open(d / "metrics.jsonl", encoding="utf-8")])
    ev = [json.loads(line) for line in open(d / "fault_events.jsonl", encoding="utf-8")]

    def bins(df, col, how, t0, t1, route=None, step=5.0):
        x = df[(df.timestamp >= t0) & (df.timestamp < t1)]
        if route:
            x = x[x.route == f"/{route}"]
        k = ((x.timestamp - t0) // step).astype(int)
        out = []
        for _, g in x.groupby(k):
            out.append(how(g[col]))
        return np.array(out, float)

    p95 = lambda s: float(s.quantile(0.95))
    err = lambda s: float((s >= 500).mean())
    rows = []
    for ft, params, a0, a1 in marks:
        route = params.get("route")
        if ft in ("latency", "db_slow", "latency_ramp", "pool_exhaustion"):
            name, floor = "p95 latency (ms)", 1.0
            b = bins(api, "latency_ms", p95, t_base + 5, t_base + 40)
            f_ = bins(api, "latency_ms", p95, a0 + 2, a1)
        elif ft == "cpu_burn":
            name, floor = "process CPU (%)", 1.0
            b = bins(met[met.metric == "process_cpu_percent"], "value", np.mean, t_base + 5, t_base + 40)
            f_ = bins(met[met.metric == "process_cpu_percent"], "value", np.mean, a0 + 2, a1)
        elif ft in ("route_500", "exception_route"):
            name, floor = f"error share on /{route}", 0.01
            b = bins(api, "status_code", err, t_base + 5, t_base + 40, route)
            f_ = bins(api, "status_code", err, a0 + 2, a1, route)
        elif ft == "db_lock":
            name, floor = "p95 latency of writes (ms)", 1.0
            wr = api[api.method == "POST"]
            b = bins(wr, "latency_ms", p95, t_base + 5, t_base + 40)
            f_ = bins(wr, "latency_ms", p95, a0 + 2, a1)
        elif ft == "crash":
            name, floor = "requests per 5 s (drop)", 1.0
            b = bins(api, "status_code", len, t_base + 5, t_base + 40)
            ev_c = [e for e in ev if e.get("event") == "restart" and a0 <= e["ts"] <= a1 + 30]
            t_end = ev_c[0]["ts"] if ev_c else a1
            f_ = bins(api, "status_code", len, a0 + 2, t_end)
            if len(f_) == 0:
                f_ = np.array([0.0])
        elif ft == "leak":
            name, floor = "resident memory (MB)", 0.5
            m = met[met.metric == "process_rss_bytes"]
            b = bins(m, "value", lambda s: s.mean() / 1e6, t_base + 5, t_base + 40)
            f_ = bins(m, "value", lambda s: s.mean() / 1e6, a0 + 2, a1)
        sd = max(float(np.std(b)), floor)
        z = (float(np.max(f_)) - float(np.mean(b))) / sd if ft != "crash" else (float(np.mean(b)) - float(np.min(f_))) / sd
        rows.append(dict(fault=ft, severity="mild (u=0.02)", parameters=json.dumps({k: v for k, v in params.items()}), measure=name, baseline=round(float(np.mean(b)), 2), baseline_sd=round(float(np.std(b)), 2),
                         during_fault=round(float(np.max(f_)) if ft != "crash" else float(np.min(f_)), 2), z=round(z, 1), visible=bool(z >= 3)))
    df = pd.DataFrame(rows)
    df.to_csv(d / "mild_summary.csv", index=False)
    print(df.to_string(index=False))
    fe = [e for e in ev if e.get("event") == "failure"]
    print("gradual failure events:", [(e["fault_type"], round(e["ts"] - t_base, 1)) for e in fe])
    print("mild faults with a visible effect (z >= 3):", int(df.visible.sum()), "of", len(df))


if __name__ == "__main__" and "--mild" in sys.argv:
    mild_check()
    sys.exit(0)
