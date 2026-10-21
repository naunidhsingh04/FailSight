"""One command: python -m testbed.launcher   (or start_testbed.bat on Windows; no Docker)

Starts the monitor (FailSight backend + dashboard on http://127.0.0.1:8000), the demo shop (port 8802, instrumented only with failsight_agent), a steady load generator, and opens the
dashboard on the "Live demo" tab. The monitor learns the shop's normal for 150 s ("Learning normal behaviour"), then scores every 10 s with the frozen detector. Use the fault switches on
the Live demo tab to inject real faults. Ctrl+C stops everything.   Options: --rate 6  --profile steady|bursty|ramp|daily  --no-browser"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from testbed import loadgen, run_generator as RG  # noqa: E402
from testbed.harness import Testbed  # noqa: E402

BACKEND = os.environ.get("FAILSIGHT_MONITOR_URL", "http://127.0.0.1:8000")          # set when the monitor itself starts the shop (Start button on the Demo page)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rate", type=float, default=6.0)
    ap.add_argument("--profile", choices=loadgen.PROFILES, default="steady")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    if not (ROOT / "report_outputs" / "testbed_frozen_config.json").exists() or not (ROOT / "models" / "testbed" / "detector.joblib").exists():
        sys.exit("the frozen detector is missing: run  python -m testbed.freeze  first")
    if not (ROOT / "app" / "frontend" / "dist").exists():
        print("note: the dashboard is not built (cd app/frontend && npm install && npm run build); the API still works")
    RG.keep_awake()
    backend = None
    try:
        httpx.get(BACKEND + "/api/health", timeout=2)
        print("monitor already running on", BACKEND)
    except Exception:
        print("starting the monitor (about 40 s) ...", flush=True)
        env = dict(os.environ, PORT="8000", FAILSIGHT_LOAD_DEEP="0", PYTHONPATH=str(ROOT))
        (ROOT / "outputs").mkdir(exist_ok=True)
        backend = subprocess.Popen([sys.executable, "-m", "app.backend.serve"], cwd=ROOT, env=env, stdout=open(ROOT / "outputs" / "launcher_backend.log", "ab"), stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < 240:
            try:
                if httpx.get(BACKEND + "/api/testbed/state", timeout=2).status_code == 200:
                    break
            except Exception:
                time.sleep(2)
        else:
            backend.kill()
            sys.exit("the monitor did not start; see outputs/launcher_backend.log")
    run_dir = ROOT / "testbed" / "runs" / "_live_session"
    run_dir.mkdir(parents=True, exist_ok=True)
    tb = Testbed(run_dir, seed=1, shop_id="shop", monitor_url=BACKEND, log=print).start(with_recorder=False)
    httpx.post(BACKEND + "/api/testbed/reset", json=dict(app_id="shop", start_ts=time.time() + 1.0), timeout=10)
    stop = threading.Event()
    th = threading.Thread(target=lambda: asyncio.run(loadgen.run(f"http://127.0.0.1:{tb.sp}", 8 * 3600, a.profile, a.rate, int(time.time()) % 8000, stop=stop)), daemon=True)
    th.start()
    url = BACKEND + "/?tab=live"
    print(f"\nshop running (port {tb.sp}), steady load {a.rate} requests/s, monitor learning its normal for 150 s.\nDashboard: {url}\nPress Ctrl+C to stop everything.\n", flush=True)
    if not a.no_browser:
        webbrowser.open(url)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        tb.stop()
        if backend is not None:
            backend.terminate()
        print("stopped")


if __name__ == "__main__":
    main()
