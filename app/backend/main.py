"""FailSight API. Start: uvicorn app.backend.main:app  (from the repo root)."""
import asyncio, io, json, logging, os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, results
from .decision import apply_decision
from .explain import Explainer
from .features import InputError, missing_report, sequence_context, to_frame
from .monitor import IngestError, Monitor
from .registry import Registry, RegistryError
from . import proofs_api
from .replay_ctl import ReplayController

log = logging.getLogger("failsight")


class PredictBody(BaseModel):
    model: str
    label_version: str = "label_binary"
    window: dict | None = None
    windows: list[dict] | None = None
    sample_id: str | None = None


class ExplainBody(BaseModel):
    model: str
    label_version: str = "label_binary"
    window: dict | None = None
    sample_id: str | None = None
    explain_class: str | None = None
    lime: bool = True


def create_app(registry: Registry | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        reg = registry or Registry()
        reg.load()                         # raises RegistryError with a clear message when a model file is missing
        app.state.reg = reg
        app.state.explainer = Explainer(reg)
        app.state.monitor = Monitor(reg, app.state.explainer)
        app.state.monitor.hub.loop = asyncio.get_running_loop()
        app.state.replay = ReplayController()
        for (_l, _n) in reg.entries:         # fill the metric cache now so the first page load of Results / Limits is not slow
            results.model_metrics(_l, _n)
        log.info("FailSight: %d model entries, %d available", len(reg.entries), sum(e["available"] for e in reg.entries.values()))
        yield
        app.state.replay.stop(wait=True)

    app = FastAPI(title="FailSight API", version="1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.exception_handler(InputError)
    async def input_error(_, exc: InputError):
        return JSONResponse(status_code=exc.status, content={"error": exc.detail})

    @app.exception_handler(IngestError)
    async def ingest_error(_, exc: IngestError):
        return JSONResponse(status_code=exc.status, content={"error": exc.detail})

    @app.exception_handler(RegistryError)
    async def registry_error(_, exc: RegistryError):
        return JSONResponse(status_code=503, content={"error": str(exc)})

    def reg_of(request: Request) -> Registry:
        return request.app.state.reg

    def samples() -> list:
        p = config.ASSETS / "samples.json"
        if not p.exists():
            raise InputError(503, "samples.json is missing (generate it with app/backend/build_assets.py).")
        return json.loads(p.read_text())["windows"]

    def get_entry(reg: Registry, label: str, name: str) -> dict:
        if label not in config.CLASS_NAMES:
            raise InputError(404, f"unknown label_version '{label}' (use label_binary or label_level)")
        try:
            e = reg.get(label, name)
        except KeyError:
            raise InputError(404, f"unknown model '{name}' for {label}")
        if not e["available"]:
            raise InputError(503, f"model '{name}' cannot be served: {e['unavailable_reason']}")
        return e

    def predict_frame(request: Request, name: str, label: str, X: pd.DataFrame, info: dict) -> dict:
        reg = reg_of(request)
        e = get_entry(reg, label, name)
        prep = reg.prep
        raw = X[prep.raw_cols].to_numpy(float)
        Xp = prep.transform_array(raw, X["system_is_TT"].to_numpy(int))
        ctx = sequence_context(X)
        P = reg.predict_proba(label, name, Xp, Xp[ctx])
        pred, used = apply_decision(label, name, P)
        names = config.CLASS_NAMES[label]
        miss = missing_report(prep, X)
        rows = []
        for i in range(len(P)):
            rows.append(dict(index=i, window_id=(None if "window_id" not in X.columns else str(X["window_id"].iloc[i])), system="TT" if X["system_is_TT"].iloc[i] else "SN",
                             predicted_class=names[int(pred[i])], default_argmax_class=names[int(P[i].argmax())],
                             probabilities={n: float(P[i, k]) for k, n in enumerate(names)}, missing=miss[i]))
        warnings = []
        if info["feature_columns_missing"]:
            warnings.append(f"{info['feature_columns_missing']} of {len(prep.raw_cols)} feature columns are absent from the input and were treated as missing.")
        if info["n_unknown_columns"]:
            warnings.append(f"{info['n_unknown_columns']} unknown column(s) ignored: {', '.join(info['unknown_columns'][:8])}")
        if info["non_numeric_cells"]:
            warnings.append(f"{info['non_numeric_cells']} non-numeric cell(s) were treated as missing.")
        if e["kind"] in ("deep", "head") or (e["kind"] == "ens" and any(reg.get(label, m)["kind"] in ("deep", "head") for m in e["members"])):
            warnings.append("This model reads the last 6 windows of a run. " + ("Rows were grouped by run_id." if "run_id" in X.columns else
                                                                              "No run_id column: every window was repeated 6 times (no history), which these models never saw in training."))
        worst = max((m["missing_fraction"] for m in miss), default=0.0)
        if worst > config.RELIABILITY_BAD:
            warnings.append(f"Up to {worst:.0%} of the features this system should have are missing. LO2 external test: with 3 of 124 model features available every saved model "
                            f"scored balanced accuracy 0.500. Treat this verdict as unreliable.")
        return dict(model=name, label_version=label, family=e["family"], corrected_variant=e["corrected"], decision_rule=used, classes=names, n_windows=len(rows),
                    class_counts={n: int(sum(r["predicted_class"] == n for r in rows)) for n in names}, results=rows, input=info, warnings=warnings,
                    reliability_note="Reliability bands (normal <= 10%, degraded <= 50%, unreliable > 50% of expected features missing) are heuristics, not measured on held-out data.")

    def window_records(body_window, body_windows, sample_id) -> list:
        if sample_id:
            m = [s for s in samples() if s["id"] == sample_id]
            if not m:
                raise InputError(404, f"unknown sample_id '{sample_id}'")
            return [dict(m[0]["features"], system=m[0]["system"], window_id=m[0]["window_id"])]
        if body_windows is not None:
            return body_windows
        if body_window is not None:
            return [body_window]
        raise InputError(400, "Provide 'window' (one raw window), 'windows' (a list), 'sample_id', or upload a CSV file.")

    # ------------------------------------------------------------------ endpoints
    @app.get("/api/health")
    async def health(request: Request):
        reg = reg_of(request)
        return dict(status="ok", models=len(reg.entries), available=sum(e["available"] for e in reg.entries.values()), tensorflow=reg.tf_available and reg.load_deep)

    @app.get("/api/models")
    async def models(request: Request, label_version: str | None = None):
        reg = reg_of(request)
        thr = results.thresholds()
        out = []
        for (label, name), e in reg.entries.items():
            if label_version and label != label_version:
                continue
            mm = results.model_metrics(label, name)
            out.append(dict(name=name, label_version=label, family=e["family"], kind=e["kind"], base=e["base"], imbalance_method=e["imbalance_method"], corrected_variant=e["corrected"],
                            members=e["members"], available=e["available"], unavailable_reason=e["unavailable_reason"], supports_explain=reg.tree_estimator(label, name) is not None and e["family"] == "tree",
                            decision_rule=thr[label][name] if name in thr[label] else None, **mm))
        return dict(models=out, ci_note="95% CI: 1000 block-bootstrap resamples of the test windows (stratified by run) from test_bootstrap_ci.csv.")

    @app.post("/api/predict")
    async def predict(request: Request):
        ctype = request.headers.get("content-type", "")
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > config.MAX_UPLOAD_BYTES + 64 * 1024:
            raise InputError(413, f"Upload too large: limit is {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
        reg = reg_of(request)
        if ctype.startswith("application/json"):
            raw = await request.body()
            if len(raw) > config.MAX_UPLOAD_BYTES:
                raise InputError(413, f"Request too large: limit is {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
            try:
                body = PredictBody(**json.loads(raw))
            except Exception as ex:
                raise InputError(422, f"Invalid JSON body: {ex}")
            recs = window_records(body.window, body.windows, body.sample_id)
            name, label = body.model, body.label_version
        elif ctype.startswith("multipart/form-data"):
            form = await request.form()
            up = form.get("file")
            name, label = form.get("model"), form.get("label_version") or "label_binary"
            if not name:
                raise InputError(422, "Form field 'model' is required.")
            if up is None or not hasattr(up, "read"):
                raise InputError(400, "Form field 'file' (a CSV) is required.")
            data = await up.read()
            if len(data) > config.MAX_UPLOAD_BYTES:
                raise InputError(413, f"Upload too large: limit is {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
            if not data.strip():
                raise InputError(400, "The uploaded file is empty.")
            if b"\x00" in data[:4096]:
                raise InputError(415, "The upload looks like a binary file. Upload a UTF-8 CSV with a header row.")
            try:
                df = pd.read_csv(io.BytesIO(data), encoding="utf-8-sig")
            except Exception as ex:
                raise InputError(400, f"Could not parse the file as CSV: {type(ex).__name__}: {str(ex)[:200]}")
            if df.empty:
                raise InputError(400, "The CSV has a header but no rows.")
            recs = df.to_dict("records")
        else:
            raise InputError(415, "Send application/json (window / windows / sample_id) or multipart/form-data with a CSV 'file'.")
        get_entry(reg, label, name)
        X, info = to_frame(recs, reg.prep)
        return JSONResponse(content=results.clean(predict_frame(request, name, label, X, info)))

    @app.get("/api/samples")
    async def get_samples():
        s = samples()
        return dict(source="20 real windows of the TEST split (report_outputs/split_ids.csv), 4 per class, 2 SN + 2 TT; features are the raw window features (null = missing)",
                    windows=s)

    @app.post("/api/explain")
    async def explain(request: Request, body: ExplainBody):
        reg = reg_of(request)
        ex: Explainer = request.app.state.explainer
        e = get_entry(reg, body.label_version, body.model)
        if not ex.supported(body.label_version, body.model):
            raise InputError(400, f"SHAP/LIME are offered for tree models only (RF, ET, XGB, LGBM and their tuned versions); '{body.model}' is a {e['family']} model.")
        recs = window_records(body.window, None, body.sample_id)
        if len(recs) != 1:
            raise InputError(400, "Explain takes exactly one window.")
        X, info = to_frame(recs, reg.prep)
        pred = predict_frame(request, body.model, body.label_version, X, info)
        r = pred["results"][0]
        names = config.CLASS_NAMES[body.label_version]
        cls = body.explain_class or ("Anomaly" if body.label_version == "label_binary" else r["predicted_class"])
        if cls not in names:
            raise InputError(422, f"explain_class must be one of {names}")
        ci = names.index(cls)
        Xp = reg.prep.transform_array(X[reg.prep.raw_cols].to_numpy(float), X["system_is_TT"].to_numpy(int))
        shap_out = ex.shap_values(body.label_version, body.model, Xp, ci)
        lime_out = ex.lime_values(body.label_version, body.model, Xp, ci) if body.lime else dict(available=False, reason="not requested")
        raw_vals = X[reg.prep.raw_cols].iloc[0].to_dict()
        for blk in (shap_out["top"], lime_out.get("top", [])):
            for t in blk:
                t["raw_value"] = raw_vals.get(t["feature"], None)
        return JSONResponse(content=results.clean(dict(prediction=r, decision_rule=pred["decision_rule"], explained_class=cls, shap=shap_out, lime=lime_out, warnings=pred["warnings"],
                                                      note="SHAP values are in the model's output units (log-odds for boosted trees, probability for random forests); preprocessed_value is the scaled value the model saw.")))

    @app.get("/api/results")
    async def get_results():
        return JSONResponse(content=results.results())

    @app.get("/api/results/model/{label}/{name}")
    async def model_detail(label: str, name: str):
        c = results.curves().get(label, {}).get(name)
        if c is None and not results.confusion(label, name):
            raise InputError(404, f"no result files for model '{name}' ({label})")
        return JSONResponse(content=results.clean(dict(model=name, label_version=label, confusion=results.confusion(label, name), curves=c,
                                                      curve_note="ROC/PR on the test set; five-class curves are micro-averaged one-vs-rest; binary uses Anomaly as the positive class.")))

    @app.get("/api/data")
    async def get_data():
        return JSONResponse(content=results.data())

    @app.get("/api/lo2")
    async def get_lo2():
        return JSONResponse(content=results.lo2())

    # ------------------------------------------------------------------ live monitoring: ingestion, control, SSE
    async def read_json(request: Request) -> dict:
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > config.MAX_UPLOAD_BYTES:
            raise IngestError(413, f"request too large: limit is {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        raw = await request.body()
        if len(raw) > config.MAX_UPLOAD_BYTES:
            raise IngestError(413, f"request too large: limit is {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        try:
            body = json.loads(raw)
        except Exception as ex:
            raise IngestError(400, f"body is not valid JSON: {ex}")
        if not isinstance(body, dict):
            raise IngestError(422, "body must be a JSON object: {\"system\": \"SN\", \"events\": [...]} or one event with a \"system\" field")
        return body

    def mon(request: Request) -> Monitor:
        return request.app.state.monitor

    async def ingest_route(kind: str, request: Request):
        body = await read_json(request)
        system = body.get("system")
        events = body["events"] if "events" in body else [{k: v for k, v in body.items() if k != "system"}]
        return JSONResponse(content=await run_in_threadpool(mon(request).ingest, kind, system, events))

    @app.post("/ingest/api")
    async def ingest_api(request: Request):
        return await ingest_route("api", request)

    @app.post("/ingest/logs")
    async def ingest_logs(request: Request):
        return await ingest_route("logs", request)

    @app.post("/ingest/metrics")
    async def ingest_metrics(request: Request):
        return await ingest_route("metrics", request)

    @app.post("/ingest/reset")
    async def ingest_reset(request: Request):
        b = await read_json(request)
        if b.get("system") not in ("SN", "TT"):
            raise IngestError(422, "system must be SN or TT")
        meta = {k: b[k] for k in ("scenario", "true_label_level", "true_label_binary", "source", "block_ids") if k in b}
        return mon(request).start_session(b["system"], b.get("start_ts"), b.get("session"), meta, bool(b.get("keep_counters", False)))

    @app.post("/ingest/flush")
    async def ingest_flush(request: Request):
        b = await read_json(request)
        return JSONResponse(content=await run_in_threadpool(mon(request).flush, b.get("system"), b.get("upto_ts")))

    @app.get("/api/monitor/state")
    async def monitor_state(request: Request):
        return JSONResponse(content=results.clean(mon(request).state()))

    @app.put("/api/monitor/config")
    async def monitor_config(request: Request):
        b = await read_json(request)
        return mon(request).set_config(b.get("model"), b.get("label_version"))

    @app.get("/api/stream")
    async def stream(request: Request, replay: int = 60, max: int = 0):
        m = mon(request)
        q, snap = m.hub.subscribe(replay)

        async def gen():
            sent = 0
            try:
                yield "retry: 2000\n\n"
                for ev in snap:
                    yield f"data: {json.dumps(results.clean(ev))}\n\n"
                    sent += 1
                    if max and sent >= max:
                        return
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=15)
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
                        continue
                    yield f"data: {json.dumps(results.clean(ev))}\n\n"
                    sent += 1
                    if max and sent >= max:
                        break
            finally:
                m.hub.unsubscribe(q)
        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ------------------------------------------------------------------ proofs (report_outputs/proofs.json is written by scripts/proofs.py)
    @app.get("/api/proofs")
    async def get_proofs():
        rep = proofs_api.load()
        if rep is None:
            raise IngestError(404, "No proof report yet. Run `python scripts/proofs.py` (or use the Re-run button).")
        return JSONResponse(content=rep)

    @app.get("/api/proofs/masking")
    async def get_proofs_masking():
        """the few P13 numbers the Limits page quotes (kept small so that page does not download the whole report)"""
        rep = proofs_api.load()
        sec = next((x for x in (rep or {}).get("sections", []) if x["id"] == "P13"), None)
        d = next((i["data"]["xgb_binary"] for i in (sec or {}).get("items", []) if i.get("data", {}).get("xgb_binary")), None)
        if d is None:
            raise IngestError(404, "No P13 result yet. Run `python scripts/proofs.py --only P13`.")
        return JSONResponse(content=dict(xgb_binary=d, generated_at=sec.get("generated_at")))

    @app.get("/api/proofs/download.json")
    async def proofs_download_json():
        rep = proofs_api.load()
        if rep is None:
            raise IngestError(404, "No proof report yet.")
        return Response(json.dumps(rep, indent=1), media_type="application/json", headers={"Content-Disposition": 'attachment; filename="failsight_proofs.json"'})

    @app.get("/api/proofs/report.html")
    async def proofs_report_html(download: int = 0):
        rep = proofs_api.load()
        if rep is None:
            raise IngestError(404, "No proof report yet.")
        h = {"Content-Disposition": 'attachment; filename="failsight_proof_report.html"'} if download else {}
        return Response(proofs_api.render_html(rep), media_type="text/html; charset=utf-8", headers=h)

    @app.post("/api/proofs/run")
    async def proofs_run(request: Request):
        b = {}
        if request.headers.get("content-length", "0") not in ("0", ""):
            b = await read_json(request)
        secs = b.get("sections")
        if secs is not None and (not isinstance(secs, list) or any(x not in proofs_api.SECTIONS for x in secs)):
            raise IngestError(422, f"sections must be a list of {proofs_api.SECTIONS}")
        if proofs_api.busy():
            raise IngestError(409, "A proof run is already in progress.")
        return StreamingResponse(proofs_api.run_stream(secs, bool(b.get("quick"))), media_type="text/plain; charset=utf-8", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ------------------------------------------------------------------ replay control (the replay itself is scripts/replay_scenario.py, a separate client process)
    @app.post("/api/replay/start")
    async def replay_start(request: Request):
        b = await read_json(request)
        port = request.url.port or int(os.environ.get("PORT", "8000"))
        try:
            return request.app.state.replay.start(b.get("scenario", "Performance"), int(b.get("speed", 10)), b.get("system", "both"), int(b.get("blocks", 2)), port, b.get("model"))
        except (ValueError, TypeError) as ex:
            raise IngestError(422, str(ex))

    @app.get("/api/replay/control")
    async def replay_control_get(request: Request):
        return request.app.state.replay.state

    @app.post("/api/replay/control")
    async def replay_control_set(request: Request):
        b = await read_json(request)
        try:
            return request.app.state.replay.control(b.get("paused"), b.get("speed"), b.get("stop"))
        except ValueError as ex:
            raise IngestError(422, str(ex))

    @app.get("/api/replay/status")
    async def replay_status(request: Request):
        return request.app.state.replay.status()

    # ------------------------------------------------------------------ built frontend (single-container deployment)
    dist = config.FRONTEND_DIST
    if (dist / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        async def spa(path: str):
            if path.startswith("api/"):
                return JSONResponse(status_code=404, content={"error": "not found"})
            f = (dist / path).resolve()
            if path and f.is_file() and dist.resolve() in f.parents:
                return FileResponse(f)
            return FileResponse(dist / "index.html")
    return app


app = create_app()
