"""3-minute smoke run with the storefront in place, compared with a saved fault-free run (r000_s1000).

python scripts/smoke_storefront.py            (needs ports 8801, 8802 and 5000 free; takes about 3.5 minutes)

It starts the recorder and the shop exactly as the run generator does (testbed.harness.Testbed), the storefront, the steady load generator for 180 s and, on top, a few
real storefront sessions (browse, add to cart, checkout through the storefront). The raw events the shop's agent sent are then compared with the saved normal run: which fields every
event type carries, which metric series exist, events per request, error rates, the share of each route and the latency per route.
Nothing here changes the shop; the comparison is written to report_outputs/storefront_smoke_comparison.md."""
import asyncio
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from testbed import loadgen  # noqa: E402
from testbed.harness import Testbed  # noqa: E402

BASELINE = ROOT / "testbed" / "runs" / "r000_s1000"
RUN = ROOT / "testbed" / "runs" / "_live_smoke_storefront"       # (git-ignored: testbed/runs/_live_*)
OUT = ROOT / "report_outputs" / "storefront_smoke_comparison.md"
DURATION = 180


def read(dirpath, kind):
    rows = []
    with open(dirpath / f"{kind}.jsonl", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def summarize(dirpath):
    api, logs, met = read(dirpath, "api"), read(dirpath, "logs"), read(dirpath, "metrics")
    t = [e["timestamp"] for e in api]
    span = max(t) - min(t)
    status = {}
    by_route = {}
    for e in api:
        status[e["status_code"]] = status.get(e["status_code"], 0) + 1
        by_route.setdefault(f"{e['method']} {e['route']}", []).append(e["latency_ms"])
    n = len(api)
    lv = {}
    for e in logs:
        lv[e["level"]] = lv.get(e["level"], 0) + 1
    series = {}
    for e in met:
        series.setdefault(e["metric"], []).append(e["value"])
    return dict(
        span=span, api=n, logs=len(logs), metrics=len(met),
        api_per_s=n / span, logs_per_api=len(logs) / n, metrics_per_s=len(met) / span,
        keys_api=sorted({k for e in api for k in e if not k.startswith("_")}), keys_logs=sorted({k for e in logs for k in e if not k.startswith("_")}),
        keys_metrics=sorted({k for e in met for k in e if not k.startswith("_")}),
        series=sorted(series), status=status, err5=sum(v for k, v in status.items() if k >= 500) / n, err4=sum(v for k, v in status.items() if 400 <= k < 500) / n,
        route_share={r: len(v) / n for r, v in by_route.items()}, p50={r: float(np.percentile(v, 50)) for r, v in by_route.items()}, p95={r: float(np.percentile(v, 95)) for r, v in by_route.items()},
        levels=lv, series_mean={k: float(np.mean(v)) for k, v in series.items()},
    )


async def storefront_sessions(n, seconds, results):
    rng = random.Random(7)
    async with httpx.AsyncClient(base_url="http://127.0.0.1:5000", follow_redirects=True, timeout=15) as c:
        async def one(delay):
            await asyncio.sleep(delay)
            cl = httpx.AsyncClient(base_url="http://127.0.0.1:5000", follow_redirects=True, timeout=15)
            try:
                o, d = rng.sample(["Berlin", "Paris", "Rome", "Madrid", "Vienna", "Prague", "Oslo", "Lisbon"], 2)
                r = await cl.get("/")
                results.append(("home", r.status_code))
                r = await cl.get("/journeys", params=dict(origin=o, dest=d))
                results.append(("journeys", r.status_code))
                m = re.search(r"/journey/(\d+)\?origin", r.text)
                if not m:
                    return
                await asyncio.sleep(rng.uniform(0.3, 1.5))
                r = await cl.get(f"/journey/{m.group(1)}", params=dict(origin=o, dest=d))
                results.append(("journey", r.status_code))
                f = dict(re.findall(r'name="(\w+)" value="([^"]*)"', r.text))
                f["seats"] = "1"
                r = await cl.post("/cart/add", data=f)
                results.append(("cart", r.status_code))
                if rng.random() < 0.7:
                    r = await cl.post("/checkout", data=dict(name="A. Traveller", email="a@example.com"))
                    results.append(("checkout", r.status_code))
            finally:
                await cl.aclose()
        await asyncio.gather(*[one(rng.uniform(5, seconds - 20)) for _ in range(n)])


def run_smoke():
    if RUN.exists():
        import shutil
        shutil.rmtree(RUN, ignore_errors=True)
    tb = Testbed(RUN, seed=1000, shop_id="shop").start(with_recorder=True)
    store = subprocess.Popen([sys.executable, "-m", "storefront.server"], cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                if httpx.get("http://127.0.0.1:5000/healthz", timeout=1).status_code == 200:
                    break
            except Exception:
                time.sleep(0.5)
        results = []
        t0 = time.time()
        th = threading.Thread(target=lambda: asyncio.run(storefront_sessions(14, DURATION, results)))
        th.start()
        stats, _ = asyncio.run(loadgen.run("http://127.0.0.1:8802", DURATION, "steady", 6.0, 1000))
        th.join()
        time.sleep(3)
        elapsed = time.time() - t0
    finally:
        store.terminate()
        tb.stop()
    return results, stats, elapsed


def main():
    results, stats, elapsed = [], None, 0.0
    if "--summary-only" not in sys.argv:
        results, stats, elapsed = run_smoke()
    base, new = summarize(BASELINE), summarize(RUN)
    sf_ok = sum(1 for k, s in results if s < 400)
    lines = ["# Storefront smoke run compared with a saved normal run", "",
             f"Smoke run: {DURATION} s, steady load 6 requests/s from the load generator plus {len({k for k in range(len(results))})} storefront page and form requests (browse, cart, checkout; {sf_ok} of {len(results)} answered without an error). Baseline: `testbed/runs/r000_s1000` (fault-free, 600 s, daily profile, 9 requests/s).",
             "Rates differ because the traffic profiles differ, so the comparison is made on ratios, field names and per-route behaviour.", ""]
    ok = True

    def row(label, a, b, same=None):
        nonlocal ok
        flag = "" if same is None else ("same" if same else "DIFFERENT")
        if same is False:
            ok = False
        lines.append(f"| {label} | {a} | {b} | {flag} |")

    lines += ["| What | Saved normal run | Smoke run | |", "|---|---|---|---|"]
    row("run length (s)", f"{base['span']:.0f}", f"{new['span']:.0f}")
    row("API events per second", f"{base['api_per_s']:.2f}", f"{new['api_per_s']:.2f}")
    row("log events per API event", f"{base['logs_per_api']:.3f}", f"{new['logs_per_api']:.3f}", abs(base["logs_per_api"] - new["logs_per_api"]) < 0.15)
    row("metric events per second", f"{base['metrics_per_s']:.2f}", f"{new['metrics_per_s']:.2f}", abs(base["metrics_per_s"] - new["metrics_per_s"]) < 0.4)
    row("fields of an API event", ", ".join(base["keys_api"]), ", ".join(new["keys_api"]), base["keys_api"] == new["keys_api"])
    row("fields of a log event", ", ".join(base["keys_logs"]), ", ".join(new["keys_logs"]), base["keys_logs"] == new["keys_logs"])
    row("fields of a metric event", ", ".join(base["keys_metrics"]), ", ".join(new["keys_metrics"]), base["keys_metrics"] == new["keys_metrics"])
    row("metric series", ", ".join(base["series"]), ", ".join(new["series"]), base["series"] == new["series"])
    row("server errors (5xx) share", f"{base['err5'] * 100:.2f}%", f"{new['err5'] * 100:.2f}%", new["err5"] < 0.01)
    row("client errors (4xx) share", f"{base['err4'] * 100:.1f}%", f"{new['err4'] * 100:.1f}%", abs(base["err4"] - new["err4"]) < 0.15)
    row("status codes seen", ", ".join(f"{k}: {v}" for k, v in sorted(base["status"].items())), ", ".join(f"{k}: {v}" for k, v in sorted(new["status"].items())), set(new["status"]) <= set(base["status"]) | {422})
    lines += ["", "## Per route (share of requests, median and 95th percentile latency in ms)", "", "| Route | Saved share | Smoke share | Saved p50 / p95 | Smoke p50 / p95 | |", "|---|---|---|---|---|---|"]
    for r in sorted(set(base["route_share"]) | set(new["route_share"])):
        sb, sn = base["route_share"].get(r, 0), new["route_share"].get(r, 0)
        same = r in base["route_share"] and r in new["route_share"] and new["p50"][r] < 2.5 * base["p50"][r] + 10
        row_txt = f"| {r} | {sb * 100:.1f}% | {sn * 100:.1f}% | {base['p50'].get(r, float('nan')):.0f} / {base['p95'].get(r, float('nan')):.0f} | {new['p50'].get(r, float('nan')):.0f} / {new['p95'].get(r, float('nan')):.0f} | {'same order' if same else 'CHECK'} |"
        if not same:
            ok = False
        lines.append(row_txt)
    lines += ["", "## Metric series (mean value)", "", "| Series | Saved | Smoke |", "|---|---|---|"]
    for k in sorted(base["series_mean"]):
        lines.append(f"| {k} | {base['series_mean'][k]:.3g} | {new['series_mean'].get(k, float('nan')):.3g} |")
    lines += ["", f"Log levels, saved: {base['levels']}; smoke: {new['levels']}.", "", f"**Result: {'telemetry fields, request mix, error rates and latencies look the same as before.' if ok else 'at least one line above is flagged; read it before relying on the storefront.'}**"]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if stats is not None:
        print("load generator: %d requests, status %s; storefront requests %d; elapsed %.0f s" % (stats.requests, dict(sorted(stats.by_status.items())), len(results), elapsed))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
