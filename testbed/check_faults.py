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


if __name__ == "__main__":
    main()
