"""Backend tests. They use the real saved models (models/ and processed/preprocessor.joblib) and the committed result files, so they check the
numbers the API serves against the CSVs. If the model files are not present the model-dependent tests are skipped with a reason."""
import io, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from app.backend import config                      # noqa: E402
from app.backend.main import create_app             # noqa: E402
from app.backend.registry import Registry, RegistryError   # noqa: E402

HAVE_MODELS = (config.MODELS_DIR / "label_binary__XGB.joblib").exists() and config.PREPROCESSOR.exists()
needs_models = pytest.mark.skipif(not HAVE_MODELS, reason="models/ or processed/preprocessor.joblib not present (they are not stored in git)")
REP = config.REPORTS


@pytest.fixture(scope="session")
def samples(client):
    return client.get("/api/samples").json()["windows"]


def feature_row(sample):
    return dict(sample["features"], system=sample["system"], window_id=sample["window_id"])


# ------------------------------------------------------------------ registry / startup
@needs_models
def test_health_and_models(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["models"] == 36 and h["available"] >= 18
    models = client.get("/api/models").json()["models"]
    assert len(models) == 36
    assert {m["label_version"] for m in models} == {"label_binary", "label_level"}
    assert {m["family"] for m in models} <= {"tree", "linear", "deep", "ensemble"}
    xgb = next(m for m in models if m["name"] == "XGB" and m["label_version"] == "label_binary")
    tm = pd.read_csv(REP / "test_metrics.csv")
    row = tm[(tm.model == "XGB") & (tm.label_version == "label_binary") & (tm.scope == "ALL") & (tm.variant == "official")].iloc[0]
    assert xgb["metrics"]["official/ALL"]["balanced_accuracy"] == pytest.approx(row.balanced_accuracy)
    ci = pd.read_csv(REP / "test_bootstrap_ci.csv")
    r = ci[(ci.model == "XGB") & (ci.label_version == "label_binary") & (ci.scope == "ALL") & (ci.metric == "balanced_accuracy")].iloc[0]
    assert xgb["ci"]["ALL/balanced_accuracy"]["low"] == pytest.approx(r.ci95_low) and xgb["ci"]["ALL/balanced_accuracy"]["high"] == pytest.approx(r.ci95_high)
    assert xgb["corrected_variant"] is True and xgb["family"] == "tree" and xgb["supports_explain"] is True
    sv = next(m for m in models if m["name"] == "SoftVote" and m["label_version"] == "label_binary")
    assert sv["available"] is False and "fold" in sv["unavailable_reason"].lower()


@needs_models
def test_models_filter_and_metric_fields(client):
    ms = client.get("/api/models", params={"label_version": "label_level"}).json()["models"]
    assert len(ms) == 18 and all(m["label_version"] == "label_level" for m in ms)
    for m in ms:
        a = m["metrics"]["official/ALL"]
        assert {"balanced_accuracy", "macro_f1", "normal_recall", "roc_auc", "pr_auc"} <= set(a)
        assert m["ci"]["ALL/macro_f1"]["low"] <= m["ci"]["ALL/macro_f1"]["high"]


def test_registry_reports_missing_model_files(tmp_path):
    prep = config.PREPROCESSOR
    if not prep.exists():
        pytest.skip("preprocessor.joblib not present")
    with pytest.raises(RegistryError) as e:
        Registry(models_dir=tmp_path).load()
    msg = str(e.value)
    assert "missing" in msg and "label_binary__XGB.joblib" in msg and str(tmp_path) in msg


def test_registry_reports_missing_preprocessor(tmp_path):
    with pytest.raises(RegistryError, match="Preprocessor not found"):
        Registry(preprocessor_path=tmp_path / "nope.joblib").load()


def test_app_startup_fails_clearly_when_models_missing(tmp_path):
    if not config.PREPROCESSOR.exists():
        pytest.skip("preprocessor.joblib not present")
    with pytest.raises(RegistryError, match="model file"):
        with TestClient(create_app(Registry(models_dir=tmp_path))):
            pass


@needs_models
def test_tree_only_mode_marks_deep_models_unavailable():
    reg = Registry(load_deep=False).load()
    assert reg.entries[("label_binary", "XGB")]["available"]
    d = reg.entries[("label_binary", "Transformer")]
    assert not d["available"] and "TensorFlow" in d["unavailable_reason"]
    assert not reg.entries[("label_binary", "Stacking")]["available"]
    with TestClient(create_app(reg)) as c:
        r = c.post("/api/predict", json={"model": "Transformer", "sample_id": "sample-01"})
        assert r.status_code == 503 and "cannot be served" in r.json()["error"]
        assert c.post("/api/predict", json={"model": "XGB", "sample_id": "sample-01"}).status_code == 200


# ------------------------------------------------------------------ samples
@needs_models
def test_samples_are_real_balanced_test_windows(client):
    j = client.get("/api/samples").json()
    w = j["windows"]
    assert len(w) == 20
    assert pd.Series([s["label_level"] for s in w]).value_counts().to_dict() == {c: 4 for c in ["Normal", "Performance", "Service", "Database", "Code"]}
    assert {s["system"] for s in w} == {"SN", "TT"}
    sp = pd.read_csv(REP / "split_ids.csv").set_index("window_id")
    assert all(sp.loc[s["window_id"], "split"] == "test" for s in w)
    assert all(s["label_binary"] == ("Normal" if s["label_level"] == "Normal" else "Anomaly") for s in w)
    assert len(w[0]["features"]) == 155


# ------------------------------------------------------------------ predict
@needs_models
@pytest.mark.parametrize("label,name", [("label_binary", "XGB"), ("label_level", "LGBM_tuned"), ("label_binary", "Transformer"), ("label_binary", "Stacking"), ("label_level", "Fusion_XGBhead")])
def test_predict_sample_returns_probabilities_and_threshold(client, label, name):
    r = client.post("/api/predict", json={"model": name, "label_version": label, "sample_id": "sample-05"})
    assert r.status_code == 200, r.text
    j = r.json()
    res = j["results"][0]
    assert sum(res["probabilities"].values()) == pytest.approx(1.0)
    assert set(res["probabilities"]) == set(config.CLASS_NAMES[label])
    thr = json.loads((REP / "thresholds.json").read_text())[label][name]
    if thr["type"] == "threshold":
        assert j["decision_rule"]["value"] == thr["threshold"]
        p = res["probabilities"]["Anomaly"]
        assert res["predicted_class"] == ("Anomaly" if p >= thr["threshold"] else "Normal")
    else:
        assert j["decision_rule"]["weights"] == thr["weights"]
        pr = np.array([res["probabilities"][c] for c in config.CLASS_NAMES[label]]) * np.array(thr["weights"])
        assert res["predicted_class"] == config.CLASS_NAMES[label][int(pr.argmax())]
    assert res["missing"]["reliability"] in ("normal", "degraded", "unreliable")


@needs_models
def test_predict_matches_preprocess_path(client, samples):
    """the API must equal preprocess.transform + the saved model, for the same raw window"""
    sys.path.insert(0, str(config.SCRIPTS))
    import joblib, preprocess as P
    s = samples[3]
    prep = P.load_preprocessor(config.PREPROCESSOR)
    x = prep.transform_raw_window(dict(s["features"], system=s["system"]))[None, :]
    est = joblib.load(config.MODELS_DIR / "label_binary__XGB_tuned.joblib")
    est = est.named_steps["clf"] if hasattr(est, "named_steps") else est
    p = est.predict_proba(x)[0]
    r = client.post("/api/predict", json={"model": "XGB_tuned", "label_version": "label_binary", "window": dict(s["features"], system=s["system"])}).json()["results"][0]
    assert r["probabilities"]["Anomaly"] == pytest.approx(float(p[1]), abs=1e-9)


@needs_models
def test_predict_reports_missing_features_by_group(client):
    r = client.post("/api/predict", json={"model": "XGB", "window": {"system": "TT", "log_line_count": 120, "log_error_count": 3}})
    assert r.status_code == 200
    j = r.json()
    m = j["results"][0]["missing"]
    assert m["reliability"] == "unreliable" and m["missing_fraction"] > 0.9
    assert set(m["by_group"]) == {"api", "log", "metric"}
    assert m["by_group"]["metric"]["missing"] == m["by_group"]["metric"]["expected"] > 0
    assert m["by_group"]["log"]["missing"] < m["by_group"]["log"]["expected"]
    assert any("absent from the input" in w for w in j["warnings"]) and any("unreliable" in w for w in j["warnings"])


@needs_models
@pytest.mark.parametrize("body,status,needle", [
    ({"model": "XGB", "window": {"log_line_count": 1}}, 422, "system"),
    ({"model": "XGB", "window": {"system": "XX", "log_line_count": 1}}, 422, "SN"),
    ({"model": "XGB", "window": {"system": "SN", "foo": 1, "bar": 2}}, 422, "None of the columns"),
    ({"model": "NOPE", "sample_id": "sample-01"}, 404, "unknown model"),
    ({"model": "XGB", "label_version": "label_x", "sample_id": "sample-01"}, 404, "label_version"),
    ({"model": "XGB", "sample_id": "sample-99"}, 404, "sample_id"),
    ({"model": "XGB"}, 400, "Provide"),
    ({"model": "XGB", "windows": []}, 400, "No windows"),
    ({"window": {"system": "SN"}}, 422, "Invalid JSON body"),
])
def test_predict_input_errors(client, body, status, needle):
    r = client.post("/api/predict", json=body)
    assert r.status_code == status, r.text
    assert needle in json.dumps(r.json())


@needs_models
def test_predict_rejects_bad_json_and_wrong_content_type(client):
    assert client.post("/api/predict", content=b"{not json", headers={"content-type": "application/json"}).status_code == 422
    r = client.post("/api/predict", content=b"hello", headers={"content-type": "text/plain"})
    assert r.status_code == 415 and "multipart" in r.json()["error"]


def csv_bytes(samples, n=5, drop=()):
    rows = [feature_row(s) for s in samples[:n]]
    df = pd.DataFrame(rows).drop(columns=list(drop))
    return df.to_csv(index=False).encode()


@needs_models
def test_predict_csv_upload(client, samples):
    r = client.post("/api/predict", data={"model": "XGB", "label_version": "label_binary"}, files={"file": ("w.csv", csv_bytes(samples, 6), "text/csv")})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["n_windows"] == 6 and len(j["results"]) == 6 and sum(j["class_counts"].values()) == 6
    assert j["results"][0]["window_id"] == samples[0]["window_id"]


@needs_models
def test_predict_csv_with_dropped_columns_reports_missing(client, samples):
    r = client.post("/api/predict", data={"model": "LGBM_tuned"}, files={"file": ("w.csv", csv_bytes(samples, 4, drop=["log_line_count", "api_error_rate"]), "text/csv")})
    assert r.status_code == 200
    j = r.json()
    assert j["input"]["feature_columns_missing"] >= 2 and any("absent from the input" in w for w in j["warnings"])


@needs_models
@pytest.mark.parametrize("payload,status,needle", [
    (b"", 400, "empty"),
    (b"a,b,c\n", 400, "no rows"),
    (b"system,foo\nSN,1\n", 422, "None of the columns"),
    (b"\x00\x01\x02binary\x00", 415, "binary"),
    (b"log_line_count\n10\n", 422, "system"),
])
def test_predict_csv_bad_files(client, payload, status, needle):
    r = client.post("/api/predict", data={"model": "XGB"}, files={"file": ("bad.csv", payload, "text/csv")})
    assert r.status_code == status, r.text
    assert needle in json.dumps(r.json())


@needs_models
def test_predict_csv_too_large_and_missing_fields(client):
    big = b"system,log_line_count\n" + b"SN,1\n" * (config.MAX_UPLOAD_BYTES // 5 + 1000)
    r = client.post("/api/predict", data={"model": "XGB"}, files={"file": ("big.csv", big, "text/csv")})
    assert r.status_code == 413 and "5 MB" in r.json()["error"]
    assert client.post("/api/predict", data={"label_version": "label_binary"}, files={"file": ("a.csv", b"x\n1\n", "text/csv")}).status_code == 422
    assert client.post("/api/predict", data={"model": "XGB"}, files={"other": ("a.txt", b"x")}).status_code == 400


@needs_models
def test_predict_csv_row_limit(client):
    rows = "system,log_line_count\n" + "SN,1\n" * (config.MAX_ROWS + 1)
    r = client.post("/api/predict", data={"model": "XGB"}, files={"file": ("many.csv", rows.encode(), "text/csv")})
    assert r.status_code == 413 and "Too many windows" in r.json()["error"]


@needs_models
def test_sequence_models_use_run_id_context(client, samples):
    rows = [dict(feature_row(s), run_id="r1") for s in samples[:3]]
    r = client.post("/api/predict", json={"model": "Transformer", "windows": rows})
    assert r.status_code == 200 and any("grouped by run_id" in w for w in r.json()["warnings"])
    r2 = client.post("/api/predict", json={"model": "Transformer", "windows": [feature_row(samples[0])]})
    assert any("repeated 6 times" in w for w in r2.json()["warnings"])


# ------------------------------------------------------------------ explain
@needs_models
def test_explain_shap_and_lime(client):
    r = client.post("/api/explain", json={"model": "XGB", "label_version": "label_binary", "sample_id": "sample-12"})
    assert r.status_code == 200, r.text
    j = r.json()
    top = j["shap"]["top"]
    assert len(top) == 15 and all(t["group"] in ("api", "log", "metric", "system") for t in top)
    assert [abs(t["shap"]) for t in top] == sorted([abs(t["shap"]) for t in top], reverse=True)
    assert sum(g["share"] for g in j["shap"]["groups"]) == pytest.approx(1.0)
    if (config.ASSETS / "lime_background.npy").exists():
        assert j["lime"]["available"] and len(j["lime"]["top"]) == 15
    else:
        assert j["lime"]["available"] is False and "background" in j["lime"]["reason"]


@needs_models
def test_explain_five_class_and_errors(client):
    r = client.post("/api/explain", json={"model": "LGBM_tuned", "label_version": "label_level", "sample_id": "sample-03", "lime": False})
    assert r.status_code == 200 and r.json()["explained_class"] in config.CLASS_NAMES["label_level"] and r.json()["lime"]["available"] is False
    assert client.post("/api/explain", json={"model": "Transformer", "sample_id": "sample-03"}).status_code == 400
    assert client.post("/api/explain", json={"model": "LR", "sample_id": "sample-03"}).status_code == 400
    assert client.post("/api/explain", json={"model": "XGB", "sample_id": "nope"}).status_code == 404
    assert client.post("/api/explain", json={"model": "XGB", "sample_id": "sample-03", "explain_class": "Banana"}).status_code == 422
    assert client.post("/api/explain", json={"model": "XGB", "window": {"system": "SN", "log_line_count": 3}}).status_code == 200


# ------------------------------------------------------------------ results / data / lo2
@needs_models
def test_results_endpoint_matches_csvs(client):
    j = client.get("/api/results").json()
    lb = pd.DataFrame(j["leaderboards"]["label_binary"]).set_index("model")
    ref = pd.read_csv(REP / "leaderboard_label_binary.csv").set_index("model")
    assert lb.loc["XGB", "test_bal_acc"] == pytest.approx(ref.loc["XGB", "test_bal_acc"])
    assert len(j["leaderboards"]["label_level"]) == 18
    assert len(j["correction"]) == len(pd.read_csv(REP / "before_after_correction.csv"))
    fr = pd.DataFrame(j["first_run_vs_retrain"])
    assert len(fr) > 0 and {"v1_balanced_accuracy", "v2_balanced_accuracy"} <= set(fr.columns)
    assert j["overfit"] and j["paired_bootstrap"]


@needs_models
def test_model_detail_confusion_and_curves(client):
    j = client.get("/api/results/model/label_binary/XGB").json()
    cm = j["confusion"]["ALL/argmax"]
    assert cm["labels"] == ["Normal", "Anomaly"] and np.array(cm["matrix"]).sum() == 647
    assert len(j["curves"]["roc"]) > 10 and len(j["curves"]["pr"]) > 10
    assert client.get("/api/results/model/label_binary/NOPE").status_code == 404


def test_data_endpoint():
    with TestClient(create_app(_StubRegistry())) as c:
        j = c.get("/api/data").json()
    assert j["dataset"]["train"] == 2549 and j["dataset"]["test"] == 647 and j["dataset"]["windows"] == 3196
    assert j["folds"]["n_folds"] == 10 and j["folds"]["group_type"] == "block_id"
    assert any(r["check"] == "duplicates" for r in j["leakage"])
    assert {r["class"] for r in j["smote_final_train"]} >= {"Normal", "Anomaly"}


class _StubRegistry:
    """registry-free app for endpoints that only read result files"""
    entries = {}
    tf_available = False
    load_deep = False
    prep = None

    def load(self):
        return self

    def tree_estimator(self, *a):
        return None


def test_lo2_endpoint_states_the_failure():
    with TestClient(create_app(_StubRegistry())) as c:
        j = c.get("/api/lo2").json()
    assert j["sample"]["windows"] == 28 and j["sample"]["normal_windows"] == 5
    a = j["feature_mapping"]["A_logs_only"]
    assert (a["features_built"], a["features_total"], a["used_features_built"], a["used_features_total"]) == (3, 155, 3, 124)
    assert len(j["models"]) == 18 and all(m["balanced_accuracy"] == pytest.approx(0.5) for m in j["models"])
    assert j["log_volume_check"]["raw_log_line_count_auc"] == pytest.approx(1.0)
    assert "ROC-AUC" in j["summary_markdown"]


def test_unknown_api_path_is_json_404():
    with TestClient(create_app(_StubRegistry())) as c:
        assert c.get("/api/nope").status_code == 404
