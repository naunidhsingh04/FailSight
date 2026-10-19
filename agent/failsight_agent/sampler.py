"""Process sampler: CPU, resident memory, thread count, event-loop lag, DB-pool wait and in-flight requests every `interval` seconds."""
import threading
import time

try:
    import psutil
except ImportError:                                       # the metrics option needs psutil; the request and log features do not
    psutil = None


class Sampler:
    def __init__(self, agent, interval=2.0):
        self.agent = agent
        self.interval = interval
        self._stop = threading.Event()
        self.proc = psutil.Process() if psutil else None

    def start(self):
        if not self.proc:
            return self
        self.proc.cpu_percent(None)
        threading.Thread(target=self._run, name="failsight-sampler", daemon=True).start()
        return self

    def stop(self):
        self._stop.set()

    def sample(self):
        a = self.agent
        now = time.time()
        waits, a._pool_waits = a._pool_waits, []
        lag, a._lag_max = a._lag_max, 0.0
        vals = dict(process_cpu_percent=self.proc.cpu_percent(None), process_rss_bytes=float(self.proc.memory_info().rss), process_threads=float(self.proc.num_threads()),
                    event_loop_lag_ms=lag * 1000.0, db_pool_wait_ms=(sum(waits) / len(waits) * 1000.0) if waits else 0.0, inflight_requests=float(a._inflight))
        for k, v in vals.items():
            a.client.emit("metrics", dict(timestamp=now, metric=k, series=a.app_id, value=round(v, 4)))

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                self.sample()
            except Exception:
                pass
