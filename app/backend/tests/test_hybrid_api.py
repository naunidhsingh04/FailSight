"""The hybrid-model endpoints serve the saved CSV files unchanged and still work (empty, available=false) when the files are missing, and in testbed-only mode."""
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from app.backend import config, hybrid_results  # noqa: E402
from app.backend.main import create_app  # noqa: E402
from app.backend.registry import Registry  # noqa: E402

HAVE = (config.REPORTS / "hybrid_anomod_leaderboard.csv").exists()
needs = pytest.mark.skipif(not HAVE, reason="hybrid result files not present (run scripts/hybrid_anomod.py)")


@needs
def test_hybrid_endpoint_serves_the_csv_numbers(client):
    d = client.get("/api/hybrid").json()
    assert d["available"] and "10-fold" in d["fold_note"] and "5-fold" in d["fold_note"] and "647 test windows" in d["fold_note"]
    lb = pd.read_csv(config.REPORTS / "hybrid_anomod_leaderboard.csv")
    done = lb[lb.status != "not completed"]
    api = {(m["model"], m["label_version"]): m for m in d["anomod"]["models"]}
    for r in done.itertuples():
        assert abs(api[(r.model, r.label_version)]["test_bal_acc"] - r.test_bal_acc) < 1e-9
    assert any("no transfer" in n.lower() or "transfer" in n.lower() for n in d["notes"])


@needs
def test_lo2_part_uses_whole_runs_with_an_untouched_test_part(client):
    d = client.get("/api/hybrid").json()
    f = d["lo2"].get("facts")
    if f and f.get("windows"):
        assert f["unreliable"] is (f["runs"] < 6 or f["windows"] < 150)
        assert f["test_runs"] >= 1 and f["train_runs"] >= 1 and f["train"] + f["test"] == f["windows"]


def test_leaderboard_endpoint_never_fails(client):
    r = client.get("/api/lo2/leaderboard")
    assert r.status_code == 200 and "available" in r.json()


def test_missing_files_give_an_empty_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPORTS", tmp_path)
    monkeypatch.setattr(hybrid_results, "R", lambda name: tmp_path / name)
    h = hybrid_results.hybrids()
    assert h["available"] is False and h["anomod"]["models"] == [] and h["lo2"]["models"] == []
    assert hybrid_results.lo2_leaderboard()["available"] is False


def test_endpoints_work_in_testbed_only_mode(tmp_path):
    reg = Registry(preprocessor_path=tmp_path / "missing_preprocessor.joblib")
    with TestClient(create_app(reg)) as c:
        assert c.get("/api/health").json()["status"] == "testbed-only"
        assert c.get("/api/hybrid").status_code == 200
        assert c.get("/api/lo2/leaderboard").status_code == 200
