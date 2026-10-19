"""The failsight_agent package (middleware, log handler, sampler, pool timing) and the shop app that uses it, against the recorder (same ingestion protocol as the monitor)."""
import json
import logging
import socket
import sys
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agent"))
from failsight_agent import FailSight  # noqa: E402
from testbed import recorder  # noqa: E402
from testbed.shop import app as shop_app  # noqa: E402


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture()
def rec(tmp_path):
    port = free_port()
    srv = uvicorn.Server(uvicorn.Config(recorder.create_app(tmp_path / "rec"), host="127.0.0.1", port=port, log_level="error"))
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}", tmp_path / "rec"
    srv.should_exit = True
    t.join(timeout=5)


def read(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")] if path.exists() else []


def test_middleware_records_route_status_and_latency(rec):
    url, d = rec
    fs = FailSight("t", url, send_metrics=False, send_logs=False, flush_interval=0.2)
    app = FastAPI()
    fs.instrument(app)

    @app.get("/items/{i}")
    def item(i: int):
        return {"i": i}

    @app.get("/boom")
    def boom():
        raise RuntimeError("x")

    c = TestClient(app, raise_server_exceptions=False)
    assert c.get("/items/3").status_code == 200
    assert c.get("/boom").status_code == 500
    assert c.get("/missing").status_code == 404
    fs.close()
    ev = read(d / "api.jsonl")
    by = {e["status_code"]: e for e in ev}
    assert set(by) == {200, 500, 404}
    assert by[200]["route"] == "/items/{i}" and by[200]["method"] == "GET" and by[200]["latency_ms"] >= 0 and "timestamp" in by[200]


def test_log_handler_and_process_sampler_and_pool_wait(rec):
    url, d = rec
    fs = FailSight("t", url, metrics_interval=0.2, flush_interval=0.2)

    class Pool:
        def acquire(self):
            import contextlib

            @contextlib.contextmanager
            def cm():
                time.sleep(0.05)
                yield 1
            return cm()

    pool = Pool()
    app = FastAPI()
    fs.instrument(app, pool=pool)
    logging.getLogger("myapp").warning("disk almost full")
    with pool.acquire():
        pass
    time.sleep(0.7)
    fs.close()
    logs = read(d / "logs.jsonl")
    assert any(e["level"] == "WARNING" and "disk almost full" in e["message"] for e in logs)
    m = read(d / "metrics.jsonl")
    names = {e["metric"] for e in m}
    assert {"process_cpu_percent", "process_rss_bytes", "process_threads", "event_loop_lag_ms", "db_pool_wait_ms", "inflight_requests"} <= names
    assert max(e["value"] for e in m if e["metric"] == "db_pool_wait_ms") >= 40


def test_agent_never_blocks_or_fails_when_the_monitor_is_down():
    fs = FailSight("t", "http://127.0.0.1:9", send_metrics=False, send_logs=False, flush_interval=0.1)
    app = FastAPI()
    fs.instrument(app)

    @app.get("/x")
    def x():
        return 1

    c = TestClient(app)
    t0 = time.time()
    assert c.get("/x").status_code == 200
    assert time.time() - t0 < 2
    fs.close(); fs.client.stop(flush=False)


def test_shop_endpoints_and_the_app_has_no_reporting_code(tmp_path):
    src = (ROOT / "testbed" / "shop" / "app.py").read_text(encoding="utf-8")
    assert "urllib" not in src and "requests" not in src and "httpx" not in src and "psutil" not in src and ".emit(" not in src     # monitored only through the agent
    app = shop_app.create_app(str(tmp_path / "shop.db"), str(tmp_path / "ev.jsonl"))
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    res = []
    for o, dd in [("Berlin", "Paris"), ("Rome", "Madrid"), ("Vienna", "Prague"), ("Oslo", "Lisbon")]:
        res = c.get("/search", params=dict(origin=o, dest=dd)).json()["results"]
        if res:
            break
    assert res
    b = c.post("/book", json=dict(route_id=res[0]["id"], user="u", seats=1))
    assert b.status_code == 200
    bid = b.json()["booking_id"]
    assert c.post("/pay", json=dict(booking_id=bid, amount=10)).json() == {"paid": True}
    assert c.post("/pay", json=dict(booking_id=bid, amount=10)).status_code == 409           # not payable twice
    assert c.post("/cancel", json=dict(booking_id=bid)).status_code == 200
    assert c.post("/cancel", json=dict(booking_id=bid)).status_code == 404
