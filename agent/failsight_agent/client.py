"""Batching transport of the FailSight agent: events are queued in memory and POSTed to the monitor in the background (standard library only)."""
import json
import queue
import threading
import time
import urllib.request

KINDS = ("api", "logs", "metrics")


class Client:
    """Collects api / log / metric events and sends them in batches to <monitor_url>/ingest/<kind>.

    Never blocks the application: the queue is bounded and the oldest events are dropped (and counted) when the monitor is unreachable."""

    def __init__(self, app_id, monitor_url, api_key=None, flush_interval=1.0, batch_size=500, max_queue=20000, timeout=3.0):
        self.app_id = app_id
        self.url = monitor_url.rstrip("/")
        self.api_key = api_key
        self.flush_interval = flush_interval
        self.batch_size = batch_size
        self.timeout = timeout
        self.q = {k: queue.Queue(maxsize=max_queue) for k in KINDS}
        self.sent = {k: 0 for k in KINDS}
        self.dropped = {k: 0 for k in KINDS}
        self.errors = 0
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

    def start(self):
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="failsight-sender", daemon=True)
                self._thread.start()
        return self

    def stop(self, flush=True):
        self._stop.set()
        if flush:
            self.flush()

    def emit(self, kind, event):
        try:
            self.q[kind].put_nowait(event)
        except queue.Full:
            try:
                self.q[kind].get_nowait()           # drop the oldest
            except queue.Empty:
                pass
            self.dropped[kind] += 1
            try:
                self.q[kind].put_nowait(event)
            except queue.Full:
                pass

    def _drain(self, kind, limit):
        out = []
        while len(out) < limit:
            try:
                out.append(self.q[kind].get_nowait())
            except queue.Empty:
                break
        return out

    def flush(self):
        for kind in KINDS:
            while True:
                batch = self._drain(kind, self.batch_size)
                if not batch:
                    break
                self._post(kind, batch)

    def _post(self, kind, batch):
        body = json.dumps({"system": self.app_id, "events": batch}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        req = urllib.request.Request(f"{self.url}/ingest/{kind}", data=body, headers=headers, method="POST")
        for attempt in (0, 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    r.read()
                self.sent[kind] += len(batch)
                return True
            except Exception:
                self.errors += 1
                time.sleep(0.2)
        self.dropped[kind] += len(batch)
        return False

    def _run(self):
        while not self._stop.is_set():
            time.sleep(self.flush_interval)
            try:
                self.flush()
            except Exception:
                self.errors += 1
        self.flush()
