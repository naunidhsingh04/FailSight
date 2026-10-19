"""Starts / pauses / stops scripts/replay_scenario.py (a separate process that talks to the ingestion API like any client) for the web console."""
import subprocess
import sys
import threading
import time
from collections import deque

from . import config

SCENARIOS = ("Normal", "Performance", "Service", "Database", "Code")
SPEEDS = (1, 10, 60)


class ReplayController:
    def __init__(self):
        self.proc = None
        self.lock = threading.Lock()
        self.state = dict(paused=False, speed=None, stop=False)
        self.info = {}
        self.log = deque(maxlen=200)

    def _reader(self, proc):
        for line in proc.stdout:
            self.log.append(line.rstrip())
        proc.wait()

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, scenario, speed, system, blocks, port, model=None):
        if scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}")
        if speed not in SPEEDS:
            raise ValueError(f"speed must be one of {SPEEDS}")
        if system not in ("SN", "TT", "both"):
            raise ValueError("system must be SN, TT or both")
        if not (1 <= blocks <= 4):
            raise ValueError("blocks must be between 1 and 4")
        with self.lock:
            self.stop(wait=True)
            self.state = dict(paused=False, speed=float(speed), stop=False)
            base = f"http://127.0.0.1:{port}"
            cmd = [sys.executable, str(config.SCRIPTS / "replay_scenario.py"), "--scenario", scenario, "--speed", str(speed), "--system", system, "--blocks", str(blocks),
                   "--server", base, "--control", base + "/api/replay/control"]
            if model:
                cmd += ["--model", model]
            self.log.clear()
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(config.ROOT))
            threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()
            self.info = dict(scenario=scenario, speed=speed, system=system, blocks=blocks, started_at=time.time(), command=" ".join(cmd[1:]))
        return self.status()

    def control(self, paused=None, speed=None, stop=None):
        with self.lock:
            if paused is not None:
                self.state["paused"] = bool(paused)
            if speed is not None:
                if speed not in SPEEDS:
                    raise ValueError(f"speed must be one of {SPEEDS}")
                self.state["speed"] = float(speed)
                self.info["speed"] = speed
            if stop:
                self.stop(wait=False)
        return self.status()

    def stop(self, wait=False):
        self.state["stop"] = True
        p = self.proc
        if p is not None and p.poll() is None:
            if wait:
                try:
                    p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    p.terminate()
                    try:
                        p.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        p.kill()
            else:
                threading.Timer(4.0, lambda: p.poll() is None and p.terminate()).start()

    def status(self):
        return dict(running=self.running(), paused=self.state["paused"], speed=self.state["speed"], info=self.info, log=list(self.log)[-12:])
