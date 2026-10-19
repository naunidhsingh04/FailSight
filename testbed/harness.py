"""Process management shared by the run generator and the fault self-test: recorder + shop (with a supervisor that restarts a crashed shop) on an 8 GB Windows laptop, no Docker."""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
RECORDER_PORT, SHOP_PORT = 8801, 8802


def free_port_check(port):
    import socket
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def wait_http(url, timeout=40.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if httpx.get(url, timeout=1.5).status_code == 200:
                return True
        except Exception:
            time.sleep(0.2)
    return False


class Testbed:
    """one run's processes: the recorder (writes the raw events) and the shop; a supervisor thread restarts the shop when it exits (the 'crash' fault)"""

    def __init__(self, run_dir, seed=1, shop_id="shop", recorder_port=RECORDER_PORT, shop_port=SHOP_PORT, monitor_url=None, api_key=None, log=print):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.seed, self.shop_id, self.rp, self.sp, self.log = seed, shop_id, recorder_port, shop_port, log
        self.monitor_url = monitor_url or f"http://127.0.0.1:{recorder_port}"
        self.api_key = api_key
        self.recorder = self.shop = None
        self.restarts = []                               # (exit time, healthy-again time)
        self._stop = threading.Event()
        self._sup = None

    def _spawn_shop(self):
        env = dict(os.environ, PORT=str(self.sp), SHOP_DB=str(self.dir / "shop.db"), SHOP_EVENTS=str(self.dir / "fault_events.jsonl"), FAILSIGHT_URL=self.monitor_url,
                   SHOP_ID=self.shop_id, SHOP_SEED=str(self.seed), PYTHONPATH=str(ROOT))
        if self.api_key:
            env["FAILSIGHT_KEY"] = self.api_key
        self.shop = subprocess.Popen([sys.executable, "-m", "testbed.shop.serve"], cwd=ROOT, env=env, stdout=open(self.dir / "shop.log", "ab"), stderr=subprocess.STDOUT)

    def start(self, with_recorder=True):
        for p in (self.rp if with_recorder else None, self.sp):
            if p and not free_port_check(p):
                raise RuntimeError(f"port {p} is already in use (is another testbed run still active?)")
        if with_recorder:
            self.recorder = subprocess.Popen([sys.executable, "-m", "testbed.recorder", "--dir", str(self.dir), "--port", str(self.rp)], cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)),
                                             stdout=open(self.dir / "recorder.log", "ab"), stderr=subprocess.STDOUT)
            if not wait_http(f"http://127.0.0.1:{self.rp}/stats"):
                raise RuntimeError("recorder did not start")
        self._spawn_shop()
        if not wait_http(f"http://127.0.0.1:{self.sp}/health", 60):
            raise RuntimeError("shop did not start; see " + str(self.dir / "shop.log"))
        self._sup = threading.Thread(target=self._supervise, daemon=True)
        self._sup.start()
        return self

    def _supervise(self):
        while not self._stop.is_set():
            time.sleep(0.2)
            if self.shop is not None and self.shop.poll() is not None and not self._stop.is_set():
                t_exit = time.time()
                down = 1.5                                  # a real supervisor waits before it restarts a crashed worker (back-off): the length is a parameter of the crash fault
                try:
                    for line in open(self.dir / "fault_events.jsonl", encoding="utf-8"):
                        e = json.loads(line)
                        if e.get("fault_type") == "crash" and e["event"] == "start":
                            down = float(e["params"].get("down_s", 1.5))
                except Exception:
                    pass
                time.sleep(down)
                self._spawn_shop()
                ok = wait_http(f"http://127.0.0.1:{self.sp}/health", 40)
                t_up = time.time()
                self.restarts.append((t_exit, t_up))
                self.log(f"shop restarted after a crash: down {t_up - t_exit:.1f}s (healthy={ok})")
                with open(self.dir / "fault_events.jsonl", "a", encoding="utf-8") as f:        # the end of a crash fault = the shop is healthy again
                    f.write(json.dumps(dict(event="restart", ts=t_up, down_since=t_exit)) + "\n")

    def stop(self):
        self._stop.set()
        for p in (self.shop, self.recorder):
            if p is not None and p.poll() is None:
                p.terminate()
        for p in (self.shop, self.recorder):
            if p is not None:
                try:
                    p.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    p.kill()
        time.sleep(0.5)

    def admin(self, path, method="POST", **kw):
        return httpx.request(method, f"http://127.0.0.1:{self.sp}{path}", timeout=5, **kw)

    def inject(self, ftype, duration_s, params=None):
        try:
            return self.admin("/admin/fault", json=dict(type=ftype, duration_s=duration_s, params=params or {})).json()
        except Exception as e:                           # a crash fault kills the process before it answers
            return {"error": repr(e)}
