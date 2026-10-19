"""Real faults for the shop. Every fault changes what the running app does (it sleeps, burns CPU, fails, locks the database, leaks memory, exhausts the pool, crashes);
the start, end and (for gradual faults) failure time are appended to fault_events.jsonl with exact epoch timestamps."""
import json
import os
import threading
import time
import uuid

# fault type -> (class label, kind)
CATALOGUE = {
    "latency": ("Performance", "abrupt"),
    "cpu_burn": ("Performance", "abrupt"),
    "route_500": ("Service", "abrupt"),
    "crash": ("Service", "abrupt"),
    "db_slow": ("Database", "abrupt"),
    "db_lock": ("Database", "abrupt"),
    "exception_route": ("Code", "abrupt"),
    "latency_ramp": ("Performance", "gradual"),
    "leak": ("Code", "gradual"),
    "pool_exhaustion": ("Service", "gradual"),
}
ROUTES = ("search", "book", "pay", "cancel")
LEAK_CAP_MB = 220                    # a leak that reaches this size makes the app fail (simulated out-of-memory): the failure time of the gradual leak
TIMEOUT_MS = 1500                    # latency at which a request is answered with 504 (the failure time of the latency ramp)


class FaultRegistry:
    def __init__(self, events_path=None, pool=None):
        self.events_path = events_path
        self.pool = pool
        self.lock = threading.Lock()
        self.active = {}              # fault_id -> dict
        self.leak = []
        self.leak_mb = 0.0
        self.crash_hook = lambda: os._exit(1)

    # ------------------------------------------------------------------ event log
    def _log(self, fault, event, ts=None, **extra):
        if not self.events_path:
            return
        row = dict(fault_id=fault["id"], fault_type=fault["type"], klass=fault["klass"], kind=fault["kind"], event=event, ts=ts or time.time(), params=fault["params"], **extra)
        with self.lock:
            with open(self.events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
                f.flush()
                os.fsync(f.fileno())

    # ------------------------------------------------------------------ control
    def start(self, ftype, duration_s, params=None):
        if ftype not in CATALOGUE:
            raise ValueError(f"unknown fault type {ftype!r}; known: {sorted(CATALOGUE)}")
        klass, kind = CATALOGUE[ftype]
        params = dict(params or {})
        fault = dict(id=uuid.uuid4().hex[:8], type=ftype, klass=klass, kind=kind, params=params, duration=float(duration_s), start=time.time(), failed=False, stop=threading.Event())
        fault["until"] = fault["start"] + fault["duration"]
        with self.lock:
            self.active[fault["id"]] = fault
        self._log(fault, "start", ts=fault["start"], duration_s=fault["duration"])
        if ftype == "crash":
            self.crash_hook()                                   # the process dies here; the supervisor restarts it and logs the end of the fault
            return fault["id"]
        worker = getattr(self, f"_run_{ftype}", None)
        threading.Thread(target=self._lifecycle, args=(fault, worker), name=f"fault-{ftype}", daemon=True).start()
        return fault["id"]

    def _lifecycle(self, fault, worker):
        try:
            if worker:
                worker(fault)
            while not fault["stop"].is_set() and time.time() < fault["until"]:
                time.sleep(0.05)
        finally:
            self._end(fault)

    def _end(self, fault):
        fault["stop"].set()
        if fault["type"] == "leak":
            self.leak.clear()
            self.leak_mb = 0.0
        if fault["type"] == "pool_exhaustion" and self.pool is not None:
            self.pool.release_held()
        with self.lock:
            self.active.pop(fault["id"], None)
        self._log(fault, "end")

    def clear(self):
        with self.lock:
            fs = list(self.active.values())
        for f in fs:
            f["stop"].set()

    def mark_failure(self, fault, **extra):
        if not fault["failed"]:
            fault["failed"] = True
            self._log(fault, "failure", **extra)

    def status(self):
        with self.lock:
            return [dict(id=f["id"], type=f["type"], klass=f["klass"], kind=f["kind"], start=f["start"], until=f["until"], failed=f["failed"], params=f["params"]) for f in self.active.values()]

    def of_type(self, ftype):
        with self.lock:
            return [f for f in self.active.values() if f["type"] == ftype]

    # ------------------------------------------------------------------ workers (run in their own thread while the fault is active)
    def _run_cpu_burn(self, fault):
        """busy threads that compete with the request threads for the CPU / interpreter lock"""
        n, duty = int(fault["params"].get("threads", 2)), float(fault["params"].get("duty", 0.9))

        def burn():
            while not fault["stop"].is_set() and time.time() < fault["until"]:
                t = time.perf_counter()
                while time.perf_counter() - t < 0.1 * duty:
                    sum(i * i for i in range(500))
                time.sleep(0.1 * (1 - duty))
        ts = [threading.Thread(target=burn, daemon=True) for _ in range(n)]
        [t.start() for t in ts]
        [t.join() for t in ts]

    def _run_db_lock(self, fault):
        """hold the database write lock: every write waits for it (until the busy timeout makes it fail)"""
        if self.pool is None:
            return
        import queue
        try:
            conn = self.pool.q.get(timeout=2)
        except queue.Empty:
            return
        try:
            conn.execute("BEGIN IMMEDIATE")
            while not fault["stop"].is_set() and time.time() < fault["until"]:
                time.sleep(0.05)
        finally:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            self.pool.q.put(conn)

    def _run_leak(self, fault):
        """allocate memory that is never released while the fault is active; when it reaches LEAK_CAP_MB the app starts to fail (the failure time)"""
        mb_per_s = float(fault["params"].get("mb_per_s", 1.5))
        t0 = time.time()
        while not fault["stop"].is_set() and time.time() < fault["until"]:
            chunk = bytearray(int(mb_per_s * 0.5 * 1024 * 1024))
            for i in range(0, len(chunk), 4096):            # touch the pages so the memory is really used
                chunk[i] = 1
            self.leak.append(chunk)
            self.leak_mb += mb_per_s * 0.5
            if self.leak_mb >= fault["params"].get("cap_mb", LEAK_CAP_MB):
                self.mark_failure(fault, leaked_mb=round(self.leak_mb, 1))
            time.sleep(0.5)

    def _run_pool_exhaustion(self, fault):
        """take one more database connection out of circulation every `every_s` seconds; with none left, requests wait and then fail"""
        every = float(fault["params"].get("every_s", 20))
        nxt = time.time()
        while not fault["stop"].is_set() and time.time() < fault["until"]:
            if time.time() >= nxt and self.pool is not None:
                if self.pool.free() > 0:
                    self.pool.hold_one()
                nxt += every
                if self.pool.free() == 0:
                    self.mark_failure(fault, held=len(self.pool.leaked))
            time.sleep(0.1)

    # ------------------------------------------------------------------ hooks called by the request handlers / data layer
    def before_request(self, route):
        """may sleep or raise: this is how the active faults change what a request experiences"""
        import random
        for f in self.of_type("latency"):
            time.sleep(float(f["params"].get("added_ms", 400)) / 1000.0 * random.uniform(0.7, 1.3))
        for f in self.of_type("latency_ramp"):
            p = f["params"]
            frac = min((time.time() - f["start"]) / max(float(p.get("ramp_s", f["duration"])), 1.0), 1.0)
            extra = frac * float(p.get("max_ms", 2500))
            if route in p.get("routes", ROUTES):
                if extra >= TIMEOUT_MS:
                    self.mark_failure(f, added_ms=round(extra))
                    time.sleep(TIMEOUT_MS / 1000.0)
                    raise TimeoutError("upstream timed out (injected latency ramp)")
                time.sleep(extra / 1000.0)
        for f in self.of_type("route_500"):
            if route == f["params"].get("route", "book") and random.random() < float(f["params"].get("prob", 0.8)):
                raise RuntimeError(f"injected 500 in {route}")
        for f in self.of_type("exception_route"):
            if route == f["params"].get("route", "search") and random.random() < float(f["params"].get("prob", 1.0)):
                return 1 / 0                                    # a real unhandled exception: the framework turns it into a 500 and logs the traceback
        for f in self.of_type("leak"):
            if self.leak_mb >= f["params"].get("cap_mb", LEAK_CAP_MB):
                raise MemoryError("out of memory (injected leak reached its cap)")

    def db_extra(self, conn):
        """called with the connection of the request: the slow-query fault runs real heavy queries on it"""
        from . import db
        for f in self.of_type("db_slow"):
            db.heavy_query(conn, float(f["params"].get("extra_ms", 350)))
