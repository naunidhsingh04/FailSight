"""A clean clone has no AnoMod models (they are not in git): the backend must still start, say so clearly, and keep the testbed monitor, Proofs and Testbed pages working."""
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from app.backend.main import create_app  # noqa: E402
from app.backend.registry import Registry  # noqa: E402


@pytest.fixture()
def lean(tmp_path):
    reg = Registry(preprocessor_path=tmp_path / "missing_preprocessor.joblib")       # the AnoMod preprocessor / models are absent
    with TestClient(create_app(reg)) as c:
        yield c


def test_backend_starts_without_the_anomod_models_and_says_why(lean):
    h = lean.get("/api/health")
    assert h.status_code == 200 and h.json()["status"] == "testbed-only" and "not found" in h.json()["anomod_error"].lower()
    for path in ("/api/models", "/api/monitor/state", "/api/replay/status"):
        r = lean.get(path)
        assert r.status_code == 503 and "testbed-only mode" in r.json()["error"], path
    assert lean.post("/ingest/api", json=dict(system="SN", events=[dict(timestamp=1.7e9, status_code=200)])).status_code == 503


def test_testbed_and_proofs_endpoints_still_work_without_the_anomod_models(lean):
    assert lean.get("/api/testbed/results").status_code == 200
    if (ROOT / "report_outputs" / "proofs.json").exists():
        assert lean.get("/api/proofs").status_code == 200
    if (ROOT / "models" / "testbed" / "detector.joblib").exists():
        t0 = time.time() - 30
        assert lean.post("/api/testbed/reset", json=dict(app_id="lean", start_ts=t0)).status_code == 200
        r = lean.post("/ingest/api", json=dict(system="lean", events=[dict(timestamp=t0 + 1, status_code=200, latency_ms=5.0)]))
        assert r.status_code == 200 and r.json()["accepted"] == 1
