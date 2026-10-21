"""Testbed endpoints of the monitor: ingestion for any agent app id, per-app learning state, results and fault control."""
import time

import pytest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
needs_frozen = pytest.mark.skipif(not ((ROOT / "report_outputs" / "testbed_frozen_config.json").exists() and (ROOT / "models" / "testbed" / "detector.joblib").exists()), reason="frozen testbed detector not present")


@needs_frozen
def test_agent_app_ids_go_to_the_testbed_monitor_and_anomod_ids_to_the_replay_monitor(client):
    t0 = time.time() - 40
    r = client.post("/api/testbed/reset", json=dict(app_id="apitest", start_ts=t0))
    assert r.status_code == 200 and r.json()["anchor"] == pytest.approx(t0)
    ev = [dict(timestamp=t0 + i * 0.5, method="GET", route="/search", status_code=200, latency_ms=12.0, content_length=100) for i in range(60)]
    r = client.post("/ingest/api", json=dict(system="apitest", events=ev + [dict(timestamp="bad")]))
    assert r.status_code == 200 and r.json() == dict(app="apitest", accepted=60, rejected=1)
    st = client.get("/api/testbed/state").json()
    assert "apitest" in st["apps"] and st["apps"]["apitest"]["counters"]["events_api"] == 60
    assert st["config"]["alert_rule"] == "2 of last 3 windows" and st["config"]["learning_period_s"] == 150
    # the AnoMod systems still use the original monitor
    assert client.post("/ingest/api", json=dict(system="SN", events=[dict(timestamp=1.7e9, status_code=200)])).status_code == 200
    assert client.post("/ingest/api", json=dict(system="", events=[])).status_code in (422, 400)


@needs_frozen
def test_results_endpoint_states_the_limits_and_serves_the_validation_tables(client):
    r = client.get("/api/testbed/results").json()
    text = " ".join(r["limits"])
    for needle in ("One app", "One machine", "Synthetic traffic and injected faults", "Small held-out test", "Per-type results are unreliable", "Transfer to other apps has not been shown"):
        assert needle in text
    assert r["frozen"]["alert_rule"] == "2 of last 3 windows" and len(r["assignment"]) == 40
    assert any(x["method"].startswith("XGB") for x in r["overall"])


@needs_frozen
def test_fault_endpoints_validate_input_and_report_an_unreachable_shop(client):
    assert client.post("/api/testbed/fault", json=dict(type="nope")).status_code == 422
    assert client.post("/api/testbed/fault", json=dict(type="latency", severity="extreme")).status_code == 422
    r = client.post("/api/testbed/fault", json=dict(type="latency", severity="mild"))
    assert r.status_code in (200, 502)                                       # 502 when no shop is running on port 8802
    assert client.get("/api/testbed/faults").status_code in (200, 502)


@needs_frozen
def test_api_key_check_on_ingestion_is_off_by_default_and_enforced_when_configured(client, monkeypatch):
    ev = dict(timestamp=time.time(), status_code=200, latency_ms=3.0)
    monkeypatch.delenv("FAILSIGHT_API_KEYS", raising=False)
    assert client.post("/ingest/api", json=dict(system="keytest", events=[ev])).status_code == 200            # off by default
    monkeypatch.setenv("FAILSIGHT_API_KEYS", "keytest:s3cret")
    assert client.post("/ingest/api", json=dict(system="keytest", events=[ev])).status_code == 401
    assert client.post("/ingest/api", json=dict(system="keytest", events=[ev]), headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/ingest/api", json=dict(system="keytest", events=[ev]), headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.post("/ingest/api", json=dict(system="SN", events=[dict(timestamp=1.7e9, status_code=200)])).status_code == 200      # the AnoMod replay is not affected
    st = client.get("/api/testbed/state").json()["integration"]
    assert st["api_key_required"] is True and st["apps_with_keys"] == ["keytest"]


@needs_frozen
def test_webhook_endpoint_validates_and_reports_state(client):
    assert client.put("/api/testbed/webhook", json=dict(url="ftp://x")).status_code == 422
    assert client.put("/api/testbed/webhook", json=dict(url="http://127.0.0.1:9/hook")).json() == dict(webhook_configured=True)
    assert client.get("/api/testbed/state").json()["integration"]["webhook_configured"] is True
    assert client.put("/api/testbed/webhook", json=dict(url="")).json() == dict(webhook_configured=False)


@needs_frozen
def test_start_and_stop_endpoints_never_start_a_second_shop(client, monkeypatch):
    import subprocess
    started = []

    class P:
        pid = 0
        def poll(self): return None
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: started.append(a) or P())
    monkeypatch.setattr("app.backend.testbed_api.SHOP_ADMIN", "http://127.0.0.1:1")           # nothing listens there: the shop is "down"
    client.app.state.shop_proc = None
    assert client.post("/api/testbed/start").json() == {"started": True} and len(started) == 1
    assert client.post("/api/testbed/start").json() == {"started": False, "starting": True} and len(started) == 1
    client.app.state.shop_proc = None
    assert client.post("/api/testbed/stop").json()["stopped"] is False
