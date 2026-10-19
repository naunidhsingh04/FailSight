"""The Proofs page serves report_outputs/proofs.json unchanged; the re-run endpoint streams the terminal output of scripts/proofs.py."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from app.backend import config, proofs_api  # noqa: E402

HAVE = (config.REPORTS / "proofs.json").exists()
needs_report = pytest.mark.skipif(not HAVE, reason="report_outputs/proofs.json not present; run scripts/proofs.py")


@needs_report
def test_report_equals_the_file_on_disk(client):
    r = client.get("/api/proofs")
    assert r.status_code == 200
    assert r.json() == json.loads((config.REPORTS / "proofs.json").read_text(encoding="utf-8"))


@needs_report
def test_report_has_all_ten_sections_and_no_failure(client):
    rep = client.get("/api/proofs").json()
    assert [s["id"] for s in rep["sections"]] == [f"P{i}" for i in range(1, 11)]
    assert rep["totals"].get("FAIL", 0) == 0
    assert rep["text"].strip()


@needs_report
def test_audit_table_is_in_p7_and_c3_stays_a_warning(client):
    p7 = next(s for s in client.get("/api/proofs").json()["sections"] if s["id"] == "P7")
    claims = " ".join(i["claim"] for i in p7["items"])
    for c in ["C1a", "C1b", "C2", "C3", "C4", "C5", "C6", "C7"]:
        assert c in claims, c
    c3 = [i for i in p7["items"] if i["claim"].startswith("C3")]
    assert c3 and all(i["status"] == "WARN" for i in c3)
    assert "since then" in " ".join(" ".join(i["evidence"]) for i in p7["items"] if i["claim"].startswith("C7")).lower()


@needs_report
def test_downloads(client):
    j = client.get("/api/proofs/download.json")
    assert j.status_code == 200 and "attachment" in j.headers["content-disposition"]
    h = client.get("/api/proofs/report.html?download=1")
    assert h.status_code == 200 and h.headers["content-type"].startswith("text/html")
    assert "attachment" in h.headers.get("content-disposition", "")
    assert "P7" in h.text


def test_run_rejects_unknown_sections(client):
    r = client.post("/api/proofs/run", json={"sections": ["P99"]})
    assert r.status_code == 422


def test_run_refuses_a_second_run(client, monkeypatch):
    monkeypatch.setattr(proofs_api, "busy", lambda: True)
    assert client.post("/api/proofs/run", json={"sections": ["P1"]}).status_code == 409


def test_run_streams_the_script_output(client, monkeypatch):
    seen = {}

    def fake(sections=None, quick=False):
        seen["sections"] = sections
        yield "P1 line one\n"
        yield "[finished: exit code 0]\n"

    monkeypatch.setattr(proofs_api, "run_stream", fake)
    with client.stream("POST", "/api/proofs/run", json={"sections": ["P1"]}) as r:
        assert r.status_code == 200
        body = "".join(r.iter_text())
    assert seen["sections"] == ["P1"]
    assert body.startswith("P1 line one") and body.rstrip().endswith("[finished: exit code 0]")


@needs_report
def test_lo2_is_reported_accurately_not_shown_as_a_pass(client):
    rep = client.get("/api/proofs").json()
    p9 = next(s for s in rep["sections"] if s["id"] == "P9")
    assert p9["status"] == "REPORTED ACCURATELY"
    first = p9["items"][0]
    assert first["status"] == "REPORTED ACCURATELY" and first["result"] == "models did not transfer (0.500)"
    assert any(i["status"] == "PASS" for i in p9["items"])          # the recomputation of the LO2 numbers is a real correctness check and keeps PASS
    assert rep["totals"]["REPORTED ACCURATELY"] == 2 and "Result: models did not transfer (0.500)" in rep["text"]
    html = client.get("/api/proofs/report.html").text
    assert "REPORTED ACCURATELY" in html and "Result: models did not transfer (0.500)" in html
