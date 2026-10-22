"""One command: python -m testbed.launcher   (or start_testbed.bat on Windows; no Docker)

Starts the monitor (FailSight backend + dashboard on http://127.0.0.1:8000), the shop (port 8802, instrumented only with failsight_agent), the storefront (Marigold Coaches, http://127.0.0.1:5000)
and a steady load generator, then opens two browser tabs: the storefront and the monitor. Anything that is already running on its port is reused, not started twice. The monitor learns the shop's
normal for 150 s ("Learning what normal looks like"), then scores every 10 s with the frozen detector. Break the shop on the presenter page (http://127.0.0.1:5000/ops). Ctrl+C stops what this
command started.   Options: --rate 6  --profile steady|bursty|ramp|daily  --no-browser"""
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
from testbed import harness, loadgen, run_generator as RG  # noqa: E402
from testbed.harness import Testbed  # noqa: E402

BACKEND = os.environ.get("FAILSIGHT_MONITOR_URL", "http://127.0.0.1:8000")          # set when the monitor itself starts the shop (Start button on the monitor page)
SHOP_PORT = 8802
STORE_PORT = int(os.environ.get("STOREFRONT_PORT", "5000"))
STORE = f"http://127.0.0.1:{STORE_PORT}"


def is_up(url):
    try:
        return httpx.get(url, timeout=2).status_code == 200
    except Exception:
        return False


def traffic_flowing():
    """True when the monitor has received shop events in the last 10 seconds (a load generator is already running)"""
    try:
        t = httpx.get(BACKEND + "/api/testbed/state", timeout=3).json()["apps"].get("shop", {}).get("last_event_at")
        return bool(t) and time.time() - t < 10
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rate", type=float, default=6.0)
    ap.add_argument("--profile", choices=loadgen.PROFILES, default="steady")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    if not (ROOT / "report_outputs" / "testbed_frozen_config.json").exists() or not (ROOT / "models" / "testbed" / "detector.joblib").exists():
        sys.exit("the frozen detector is missing: run  python -m testbed.freeze  first")
    if not (ROOT / "app" / "frontend" / "dist").exists():
        print("note: the monitor page is not built (cd app/frontend && npm install && npm run build); the API still works")
    RG.keep_awake()
    backend = tb = store = None
    (ROOT / "outputs").mkdir(exist_ok=True)
    try:
        # 1. the monitor
        if is_up(BACKEND + "/api/health"):
            print("monitor already running on", BACKEND, "(reusing it)")
        else:
            print("starting the monitor (about 40 s) ...", flush=True)
            env = dict(os.environ, PORT=BACKEND.rsplit(":", 1)[-1], FAILSIGHT_LOAD_DEEP="0", PYTHONPATH=str(ROOT))
            backend = subprocess.Popen([sys.executable, "-m", "app.backend.serve"], cwd=ROOT, env=env, stdout=open(ROOT / "outputs" / "launcher_backend.log", "ab"), stderr=subprocess.STDOUT)
            t0 = time.time()
            while time.time() - t0 < 240:
                if is_up(BACKEND + "/api/testbed/state"):
                    break
                time.sleep(2)
            else:
                backend.kill()
                sys.exit("the monitor did not start; see outputs/launcher_backend.log")
        # 2. the shop
        shop_url = f"http://127.0.0.1:{SHOP_PORT}"
        started_shop = False
        if is_up(shop_url + "/health"):
            print(f"shop already running on port {SHOP_PORT} (reusing it)")
        elif not harness.free_port_check(SHOP_PORT):
            sys.exit(f"port {SHOP_PORT} is in use by a program that is not the shop; close it and run this again")
        else:
            run_dir = ROOT / "testbed" / "runs" / "_live_session"
            run_dir.mkdir(parents=True, exist_ok=True)
            tb = Testbed(run_dir, seed=1, shop_id="shop", monitor_url=BACKEND, log=print).start(with_recorder=False)
            started_shop = True
            httpx.post(BACKEND + "/api/testbed/reset", json=dict(app_id="shop", start_ts=time.time() + 1.0), timeout=10)      # a freshly started shop is learned from the start
            print(f"shop started (port {SHOP_PORT})")
        # 3. the load generator
        stop = threading.Event()
        if started_shop or not traffic_flowing():
            th = threading.Thread(target=lambda: asyncio.run(loadgen.run(shop_url, 8 * 3600, a.profile, a.rate, int(time.time()) % 8000, stop=stop)), daemon=True)
            th.start()
            print(f"steady load {a.rate} requests/s started")
        else:
            print("traffic is already flowing to the shop (reusing it)")
        # 4. the storefront
        if is_up(STORE + "/healthz"):
            print(f"storefront already running on port {STORE_PORT} (reusing it)")
        elif not harness.free_port_check(STORE_PORT):
            print(f"warning: port {STORE_PORT} is in use by another program, so the storefront was not started")
        else:
            store = subprocess.Popen([sys.executable, "-m", "storefront.server"], cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT), STOREFRONT_PORT=str(STORE_PORT)),
                                     stdout=open(ROOT / "outputs" / "storefront.log", "ab"), stderr=subprocess.STDOUT)
            t0 = time.time()
            while time.time() - t0 < 30 and not is_up(STORE + "/healthz"):
                time.sleep(0.5)
            print("storefront started" if is_up(STORE + "/healthz") else "warning: the storefront did not start; see outputs/storefront.log")
        print()
        print("Shop:       " + STORE)
        print("Presenter:  " + STORE + "/ops   (this computer only)")
        print("Monitor:    " + BACKEND + "/")
        print("The monitor needs about 150 s to learn what normal looks like. Press Ctrl+C to stop what this command started.", flush=True)
        if not a.no_browser:
            if is_up(STORE + "/healthz"):
                webbrowser.open(STORE + "/")
            webbrowser.open(BACKEND + "/")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            stop.set()
        except NameError:
            pass
        for proc in (store,):
            if proc is not None and proc.poll() is None:
                proc.terminate()
        if tb is not None:
            tb.stop()
        if backend is not None:
            backend.terminate()
        print("stopped")


if __name__ == "__main__":
    main()
