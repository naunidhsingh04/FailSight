"""FailSight agent: instruments a Python web app and sends request, log and process-metric events to a FailSight monitor.

    from failsight_agent import FailSight
    fs = FailSight(app_id="shop", monitor_url="http://127.0.0.1:8800", api_key="...")      # or FailSight.from_config("failsight.json")
    fs.instrument(app, pool=db_pool)                                                          # ASGI middleware + log handler + metrics sampler

Only the standard library is required; psutil is used for the process metrics when it is installed."""
import asyncio
import json
import logging
import os
import time

from .client import Client
from .loghandler import FailSightLogHandler
from .middleware import FailSightMiddleware
from .sampler import Sampler

__all__ = ["FailSight", "FailSightMiddleware", "FailSightLogHandler"]
__version__ = "0.1.0"


class _Timed:
    """context-manager wrapper that reports how long entering the wrapped context manager took (the wait for a pooled resource)"""

    def __init__(self, cm, report):
        self.cm, self.report = cm, report

    def __enter__(self):
        t0 = time.perf_counter()
        try:
            return self.cm.__enter__()
        finally:
            self.report(time.perf_counter() - t0)

    def __exit__(self, *exc):
        return self.cm.__exit__(*exc)


class FailSight:
    def __init__(self, app_id, monitor_url, api_key=None, send_logs=True, send_metrics=True, metrics_interval=2.0, flush_interval=1.0, log_level=logging.INFO):
        self.app_id = app_id
        self.client = Client(app_id, monitor_url, api_key, flush_interval=flush_interval)
        self.send_logs, self.send_metrics = send_logs, send_metrics
        self.metrics_interval, self.log_level = metrics_interval, log_level
        self._inflight = 0
        self._lag_max = 0.0
        self._pool_waits = []
        self._loop_started = False
        self.sampler = None
        self.handler = None

    @classmethod
    def from_config(cls, path_or_dict):
        cfg = path_or_dict if isinstance(path_or_dict, dict) else json.load(open(path_or_dict, encoding="utf-8"))
        return cls(cfg["app_id"], cfg["monitor_url"], cfg.get("api_key"), cfg.get("send_logs", True), cfg.get("send_metrics", True), cfg.get("metrics_interval", 2.0))

    def instrument(self, app, pool=None, pool_method="acquire"):
        """add the middleware to a Starlette/FastAPI app, forward log records, start the process sampler (and time waits on a connection pool)"""
        self.client.start()
        app.add_middleware(FailSightMiddleware, agent=self)
        if self.send_logs:
            self.handler = FailSightLogHandler(self.client, self.log_level)
            logging.getLogger().addHandler(self.handler)
            if logging.getLogger().level > self.log_level or logging.getLogger().level == logging.NOTSET:
                logging.getLogger().setLevel(self.log_level)
        if pool is not None:
            self.watch_pool(pool, pool_method)
        if self.send_metrics:
            self.sampler = Sampler(self, self.metrics_interval).start()
        return self

    def watch_pool(self, pool, method="acquire"):
        """time every acquisition of a pooled resource (the pool method must return a context manager, or it is timed as a plain call)"""
        orig = getattr(pool, method)

        def timed(*a, **k):
            t0 = time.perf_counter()
            res = orig(*a, **k)
            if hasattr(res, "__enter__"):
                return _Timed(res, self._pool_waits.append)
            self._pool_waits.append(time.perf_counter() - t0)
            return res

        setattr(pool, method, timed)

    # ---- event-loop lag: a task in the app's own loop sleeps 50 ms and measures how late it wakes up
    def _start_loop_monitor(self):
        self._loop_started = True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        async def watch():
            while True:
                t = time.perf_counter()
                await asyncio.sleep(0.05)
                self._lag_max = max(self._lag_max, time.perf_counter() - t - 0.05)

        loop.create_task(watch())

    def close(self):
        if self.sampler:
            self.sampler.stop()
        self.client.stop()
        if self.handler:
            logging.getLogger().removeHandler(self.handler)
