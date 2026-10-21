"""Ingestion API, window cadence, malformed events, model path and SSE of the live monitor (replay of recorded data uses the same endpoints)."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from app.backend import config, monitor as M  # noqa: E402

HAVE_MODELS = (config.MODELS_DIR / "label_binary__XGB.joblib").exists() and config.PREPROCESSOR.exists()
needs_models = pytest.mark.skipif(not HAVE_MODELS, reason="models/ or processed/preprocessor.joblib not present")
T0 = 1_700_000_000.0


def reset(client, system="SN", start=T0, **meta):
    r = client.post("/ingest/reset", json=dict(system=system, start_ts=start, **meta))
    assert r.status_code == 200, r.text
    return r.json()


def logs(t0, n, level="INFO"):
    return [dict(timestamp=t0 + i, level=level, service="svc") for i in range(n)]


# ------------------------------------------------------------------ pure window assembly (no model)
def test_stream_counts_features_exactly_inside_the_window():
    st = M.Stream("SN")
    st.reset(T0)
    st.add_rows("logs", [M.clean_log(e) for e in logs(T0, 45)])
    st.add_rows("logs", [M.clean_log(dict(timestamp=T0 + 5, level="ERROR")), M.clean_log(dict(timestamp=T0 + 5.5, level="WARN")), M.clean_log(dict(timestamp=T0 + 31, level="ERROR"))])
    st.add_rows("api", [M.clean_api(dict(timestamp=T0 + 1, status_code=200, latency_ms=10, content_length=100)),
                        M.clean_api(dict(timestamp=T0 + 2, status_code=404, latency_ms=30)), M.clean_api(dict(timestamp=T0 + 3, status_code=503, latency_ms=50)),
                        M.clean_api(dict(timestamp=T0 + 30.0, status_code=500, latency_ms=999))])    # exactly at the window end: belongs to the NEXT window
    f, (ws, we) = st.window_features(0)
    assert (ws, we) == (T0, T0 + 30)
    assert f["log_line_count"] == 30 + 2                      # 30 lines at T0..T0+29 plus the ERROR and WARN at 5 and 5.5; the one at +31 is outside
    assert f["log_error_count"] == 1 and f["log_warning_count"] == 1
    assert f["api_request_count"] == 3 and f["api_4xx_count"] == 1 and f["api_5xx_count"] == 1
    assert f["api_error_rate"] == pytest.approx(2 / 3) and f["api_latency_mean_ms"] == pytest.approx(30.0)
    assert f["api_content_length_mean"] == pytest.approx(100.0)
    f1, _ = st.window_features(1)                              # window 1 = [T0+10, T0+40): contains the +30 api event and the +31 ERROR
    assert f1["api_request_count"] == 1 and f1["log_error_count"] == 1


def test_counter_metrics_are_converted_to_rates():
    st = M.Stream("TT")
    st.reset(T0)
    rows = [dict(timestamp=T0 + i * 5, metric="node_cpu_seconds_total", series="cpu0", value=100 + i * 10) for i in range(8)]
    rows += [dict(timestamp=T0 + 5, metric="node_load5", series="x", value=2.0), dict(timestamp=T0 + 15, metric="node_load5", series="x", value=4.0)]
    st.add_rows("metrics", [M.clean_metric(e) for e in rows])
    f, _ = st.window_features(0)
    assert f["m_node_cpu_seconds_total__mean"] == pytest.approx(2.0) and f["m_node_cpu_seconds_total__std"] == pytest.approx(0.0)   # 10 per 5 s = 2 per second; first sample has no rate
    assert f["m_node_load5__mean"] == pytest.approx(3.0) and f["m_node_load5__max"] == 4.0 and f["m_node_load5__std"] == pytest.approx(1.0)


def test_cadence_one_window_every_10_seconds_after_the_30_second_window_closes():
    st = M.Stream("SN")
    st.reset(T0)
    built = []
    for sec in range(0, 101):                                   # one log event per second for 100 s of event time
        st.add_rows("logs", [M.clean_log(dict(timestamp=T0 + sec, level="INFO"))])
        while st.ready():
            built.append((sec, st.next_k))
            st.advance()
    # window k is built when the watermark reaches T0 + 10 k + 30 + 1 s (lateness)
    assert [b[0] for b in built] == [31, 41, 51, 61, 71, 81, 91]
    assert [b[1] for b in built] == list(range(7))


def test_late_and_early_events_are_counted_not_used():
    st = M.Stream("SN")
    st.reset(T0)
    st.add_rows("logs", [M.clean_log(dict(timestamp=T0 + 35, level="INFO"))])
    st.add_rows("logs", [M.clean_log(dict(timestamp=T0 - 5, level="INFO"))])
    assert st.counters["before_start"] == 1
    while st.ready():
        st.advance()
    st.add_rows("logs", [M.clean_log(dict(timestamp=T0 + 2, level="INFO"))])      # window 0 already built
    assert st.counters["late_dropped"] == 1


# ------------------------------------------------------------------ HTTP: ingestion and validation
def test_ingest_single_and_batch_for_every_type(client):
    reset(client, "SN")
    r = client.post("/ingest/api", json=dict(system="SN", timestamp=T0 + 1, status_code=200, latency_ms=12.5, content_length=300))
    assert r.status_code == 200 and r.json()["accepted"] == 1 and r.json()["rejected"] == 0
    r = client.post("/ingest/logs", json=dict(system="SN", events=logs(T0, 5, "ERROR")))
    assert r.json()["accepted"] == 5
    r = client.post("/ingest/metrics", json=dict(system="SN", events=[dict(timestamp=T0 + 1, metric="node_load5", series="a", value=1.5), dict(timestamp=T0 + 2, metric="node_load5", value=None)]))
    assert r.json()["accepted"] == 2
    st = client.get("/api/monitor/state").json()["systems"]["SN"]
    assert st["counters"]["events_api"] == 1 and st["counters"]["events_logs"] == 5 and st["counters"]["events_metrics"] == 2


@pytest.mark.parametrize("path,body,status,needle", [
    ("/ingest/api", {"timestamp": T0, "status_code": 200}, 422, "system"),
    ("/ingest/api", {"system": 123, "timestamp": T0, "status_code": 200}, 422, "system"),      # any other string is an agent app id (testbed monitor); a non-string is invalid
    ("/ingest/logs", {"system": "SN", "events": "nope"}, 422, "list"),
    ("/ingest/logs", [1, 2, 3], 422, "JSON object"),
])
def test_ingest_rejects_bad_requests(client, path, body, status, needle):
    r = client.post(path, json=body)
    assert r.status_code == status and needle in json.dumps(r.json())


def test_ingest_rejects_invalid_json_and_oversized_batches(client):
    r = client.post("/ingest/api", content=b"{broken", headers={"content-type": "application/json"})
    assert r.status_code == 400 and "valid JSON" in r.json()["error"]
    r = client.post("/ingest/logs", json=dict(system="SN", events=[dict(timestamp=T0, level="INFO")] * (M.MAX_BATCH + 1)))
    assert r.status_code == 413 and "batch too large" in r.json()["error"]
    big = client.post("/ingest/logs", content=b"x" * (config.MAX_UPLOAD_BYTES + 10), headers={"content-type": "application/json"})
    assert big.status_code == 413


def test_malformed_events_are_rejected_individually_and_the_rest_accepted(client):
    reset(client, "TT")
    ev = [dict(timestamp=T0, status_code=200, latency_ms=5),                                # ok
          dict(timestamp="not a date", status_code=200),                                     # bad timestamp
          dict(timestamp=T0, status_code=99),                                                # status out of range
          dict(timestamp=T0, status_code="200"),                                             # wrong type
          dict(timestamp=T0, status_code=200, latency_ms=-3),                                # negative latency
          dict(status_code=200),                                                             # no timestamp
          "garbage",                                                                         # not an object
          dict(timestamp="2023-11-14T22:13:21Z", status_code=404, latency_ms=7)]             # ISO timestamp ok
    r = client.post("/ingest/api", json=dict(system="TT", events=ev)).json()
    assert r["accepted"] == 2 and r["rejected"] == 6
    assert [e["index"] for e in r["errors"]] == [1, 2, 3, 4, 5, 6]
    assert "timestamp" in r["errors"][0]["error"] and "status_code" in r["errors"][1]["error"] and "JSON object" in r["errors"][5]["error"]
    m = client.post("/ingest/metrics", json=dict(system="TT", events=[dict(timestamp=T0, metric="", value=1), dict(timestamp=T0, metric="m", value="abc"), dict(timestamp=T0, metric="m", value=1)])).json()
    assert m["accepted"] == 1 and m["rejected"] == 2
    lg = client.post("/ingest/logs", json=dict(system="TT", events=[dict(timestamp=T0, level=9), dict(timestamp=T0, level=["x"]), dict(timestamp=T0, level="ERROR")])).json()
    assert lg["accepted"] == 1 and lg["rejected"] == 2


# ------------------------------------------------------------------ HTTP: windows, model, SSE
@needs_models
def test_windows_are_built_every_10s_and_scored_with_the_saved_rule(client):
    reset(client, "SN", true_label_binary=1, true_label_level="Performance", scenario="synthetic")
    client.put("/api/monitor/config", json=dict(model="XGB", label_version="label_binary"))
    ev = logs(T0, 100, "INFO") + [dict(timestamp=T0 + i, level="ERROR", service="s") for i in range(0, 100, 2)]
    r = client.post("/ingest/logs", json=dict(system="SN", events=ev)).json()
    assert r["windows_built"] == 7                                    # k = 0..6 are complete at watermark T0+99
    state = client.get("/api/monitor/state").json()
    s = state["systems"]["SN"]
    assert s["counters"]["windows"] == 7 and s["last"]["window_index"] == 6
    last = s["last"]
    assert last["window_end"] - last["window_start"] == 30 and last["window_start"] == T0 + 60
    assert 0.0 <= last["risk"] <= 1.0 and last["verdict"] in ("Normal", "Anomaly", "Not enough data") and len(last["top_features"]) == 10
    assert last["decision_rule"]["type"] == "threshold" and last["decision_rule"]["value"] == json.loads((config.REPORTS / "thresholds.json").read_text())["label_binary"]["XGB"]["threshold"]
    assert last["missing"]["tier"] in ("normal", "degraded", "unreliable") and last["missing"]["fraction"] > 0.5      # only logs were sent: almost everything is missing
    assert last["truth"]["known_from"] == "recording" and last["truth"]["label_binary"] == "Anomaly"
    # the first two windows are warm-up: scored but never an alert
    hist = {w["window_index"]: w for w in s["history"]}
    assert all(not hist[k]["flagged"] for k in (0, 1))
    c = s["counters"]
    # only logs were sent, so every scored window (k = 2..6) has more than 50% of its features missing: not judged, no alert, counted separately
    assert c["low_data"] == 5 and c["labelled"] == 0 and c["alerts"] == 0 and c["correct_alerts"] + c["false_alarms"] + c["missed"] + c["correct_quiet"] == 0
    assert last["low_data"] is True and last["flagged"] is False and last["status"] == "nodata" and last["verdict"] == "Not enough data"
    assert "Not enough data to judge" in last["summary"] and s["status"] == "nodata"
    assert state["config"]["low_data_share"] == 0.5
    assert client.get("/api/monitor/state").json()["alerts"] == []


@needs_models
def test_the_same_windows_are_judged_when_the_low_data_limit_is_raised(client, monkeypatch):
    monkeypatch.setattr(M, "LOW_DATA_SHARE", 0.99)                       # the model and the saved rule are untouched; only the share that blocks a verdict moves
    reset(client, "SN", true_label_binary=1, true_label_level="Performance", scenario="synthetic")
    client.put("/api/monitor/config", json=dict(model="XGB", label_version="label_binary"))
    ev = logs(T0, 100, "INFO") + [dict(timestamp=T0 + i, level="ERROR", service="s") for i in range(0, 100, 2)]
    client.post("/ingest/logs", json=dict(system="SN", events=ev))
    c = client.get("/api/monitor/state").json()["systems"]["SN"]["counters"]
    assert c["low_data"] == 0 and c["labelled"] == 5 and c["correct_alerts"] + c["missed"] == 5


@needs_models
def test_missing_features_warning_is_reported_with_each_window(client):
    reset(client, "TT")
    client.post("/ingest/logs", json=dict(system="TT", events=logs(T0, 45)))
    w = client.get("/api/monitor/state").json()["systems"]["TT"]["last"]
    assert w["missing"]["tier"] == "unreliable" and set(w["missing"]["by_group"]) == {"api", "log", "metric"}
    assert w["missing"]["by_group"]["metric"]["missing"] == w["missing"]["by_group"]["metric"]["expected"] > 0


@needs_models
def test_flush_builds_the_last_window_and_session_reset_clears_state(client):
    reset(client, "SN")
    client.post("/ingest/logs", json=dict(system="SN", events=logs(T0, 40)))
    n0 = client.get("/api/monitor/state").json()["systems"]["SN"]["counters"]["windows"]
    r = client.post("/ingest/flush", json=dict(system="SN", upto_ts=T0 + 60)).json()
    assert r["windows_built"] >= 1
    assert client.get("/api/monitor/state").json()["systems"]["SN"]["counters"]["windows"] > n0
    reset(client, "SN")
    assert client.get("/api/monitor/state").json()["systems"]["SN"]["counters"]["windows"] == 0
    assert client.post("/ingest/flush", json=dict(system="ZZ")).status_code == 422


@needs_models
def test_sse_stream_delivers_window_events(client):
    reset(client, "SN")
    client.post("/ingest/logs", json=dict(system="SN", events=logs(T0, 45)))
    with client.stream("GET", "/api/stream", params=dict(replay=5, max=1)) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        data = [ln for ln in r.iter_lines() if ln.startswith("data: ")]
    ev = json.loads(data[0][6:])
    assert ev["type"] in ("window", "session", "config")


@needs_models
def test_config_accepts_only_servable_tree_models(client):
    assert client.put("/api/monitor/config", json=dict(model="LGBM_tuned", label_version="label_binary")).status_code == 200
    assert client.put("/api/monitor/config", json=dict(model="Transformer", label_version="label_binary")).status_code == 422
    assert client.put("/api/monitor/config", json=dict(model="XGB", label_version="label_level")).status_code == 422
    client.put("/api/monitor/config", json=dict(model="XGB", label_version="label_binary"))
    st = client.get("/api/monitor/state").json()
    assert st["config"]["model"] == "XGB" and "XGB" in st["models"] and "Transformer" not in st["models"]


@needs_models
def test_counters_continue_across_segments_of_one_replay(client):
    reset(client, "SN", true_label_binary=1, true_label_level="Performance", scenario="a")
    client.post("/ingest/logs", json=dict(system="SN", events=logs(T0, 100, "INFO")))
    n1 = client.get("/api/monitor/state").json()["systems"]["SN"]["counters"]["windows"]
    assert n1 == 7
    client.post("/ingest/reset", json=dict(system="SN", start_ts=T0 + 1000, scenario="b", true_label_binary=1, true_label_level="Performance", keep_counters=True))
    client.post("/ingest/logs", json=dict(system="SN", events=logs(T0 + 1000, 45, "INFO")))
    s = client.get("/api/monitor/state").json()["systems"]["SN"]
    assert s["counters"]["windows"] == n1 + 2 and len(s["history"]) == n1 + 2           # the tally and the timeline continue
    reset(client, "SN")                                                                  # a plain reset starts clean
    assert client.get("/api/monitor/state").json()["systems"]["SN"]["counters"]["windows"] == 0
