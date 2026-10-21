"""HTTP endpoints of the testbed monitor: state, live stream, reset, windows, fault control (proxy to the shop's admin endpoint) and the validation results."""
import asyncio
import json
import math
import os
from pathlib import Path

import pandas as pd
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "report_outputs"
SHOP_ADMIN = os.environ.get("SHOP_ADMIN_URL", "http://127.0.0.1:8802")
LIMITS = [
    "One app: the detector was built and tested on a single small ticket shop.",
    "One machine: the shop, the load generator and the monitor ran on one laptop.",
    "Synthetic traffic and injected faults: the users are a script and the faults are ones we wrote.",
    "Small held-out test: 12 runs and 15 faults.",
    "Per-type results are unreliable: most fault types have only one to three test instances.",
    "Transfer to other apps has not been shown: a new app needs its own learning period and its own check.",
]


def clean(x):
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


def table(name):
    p = REP / name
    if not p.exists():
        return []
    return clean(json.loads(pd.read_csv(p).to_json(orient="records")))


def register(app):
    def mon(request: Request):
        m = getattr(request.app.state, "testbed", None)
        if m is None:
            from .monitor import IngestError
            raise IngestError(503, "The testbed monitor is not available: run `python -m testbed.freeze` to create the frozen detector (models/testbed/detector.joblib).")
        return m

    @app.get("/api/testbed/state")
    async def tb_state(request: Request):
        from .testbed_monitor import parse_keys
        m = mon(request)
        st = m.state()
        keys = parse_keys(os.environ.get("FAILSIGHT_API_KEYS"))
        st["integration"] = dict(api_key_required=bool(keys), apps_with_keys=sorted(keys), webhook_configured=bool(m.webhook_url), webhook_host=(m.webhook_url.split("/")[2] if m.webhook_url and "//" in m.webhook_url else None),
                                 last_webhook=(m.webhook_log[-1] if m.webhook_log else None))
        return JSONResponse(content=clean(st))

    @app.put("/api/testbed/webhook")
    async def tb_webhook(request: Request):
        """set (or clear with an empty url) the Slack-compatible webhook that receives alerts"""
        b = await request.json()
        url = (b.get("url") or "").strip()
        if url and not url.startswith(("http://", "https://")):
            return JSONResponse(status_code=422, content={"error": "the webhook url must start with http:// or https://"})
        mon(request).set_webhook(url)
        return {"webhook_configured": bool(url)}

    @app.get("/api/testbed/stream")
    async def tb_stream(request: Request, replay: int = 40, max: int = 0):
        m = mon(request)
        q, snap = m.hub.subscribe(replay)

        async def gen():
            sent = 0
            try:
                yield "retry: 2000\n\n"
                for ev in snap:
                    yield f"data: {json.dumps(clean(ev), default=float)}\n\n"
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
                    yield f"data: {json.dumps(clean(ev), default=float)}\n\n"
                    sent += 1
                    if max and sent >= max:
                        break
            finally:
                m.hub.unsubscribe(q)
        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/testbed/reset")
    async def tb_reset(request: Request):
        b = await request.json()
        return mon(request).reset(str(b.get("app_id", "shop")), b.get("start_ts"))

    @app.get("/api/testbed/windows")
    async def tb_windows(request: Request, app_id: str = "shop"):
        return JSONResponse(content=clean(mon(request).windows(app_id)))

    async def shop(method, path, **kw):
        try:
            import httpx                               # only needed for the fault switches of the live demo
            async with httpx.AsyncClient(timeout=6) as c:
                r = await c.request(method, SHOP_ADMIN + path, **kw)
            return JSONResponse(status_code=r.status_code, content=r.json())
        except Exception as e:
            return JSONResponse(status_code=502, content={"error": f"the demo shop is not reachable at {SHOP_ADMIN}: {e!r}"})

    SEVERITY_U = {"mild": 0.15, "medium": 0.5, "severe": 0.85}

    @app.post("/api/testbed/fault")
    async def tb_fault(request: Request):
        """inject a real fault into the demo shop. Either explicit params, or a severity (mild / medium / severe) that is mapped to parameters exactly as the run generator does"""
        import random
        import sys
        sys.path.insert(0, str(ROOT))
        from testbed import run_generator as RG
        b = await request.json()
        ftype = b.get("type")
        if ftype not in RG.CATALOGUE:
            return JSONResponse(status_code=422, content={"error": f"unknown fault type {ftype!r}", "known": sorted(RG.CATALOGUE)})
        sev = b.get("severity", "medium")
        if sev not in SEVERITY_U:
            return JSONResponse(status_code=422, content={"error": "severity must be mild, medium or severe"})
        u = SEVERITY_U[sev]
        kind = RG.CATALOGUE[ftype][1]
        dur = float(b.get("duration_s") or (RG.fault_duration(ftype, u, random.Random(1)) if kind == "gradual" or ftype in ("crash", "db_lock") else 60.0))
        params = b.get("params") or RG.severity_params(ftype, u, dur, random.Random(1))
        params = dict(params, severity=u, severity_level=sev)
        return await shop("POST", "/admin/fault", json=dict(type=ftype, duration_s=dur, params=params))

    @app.get("/api/testbed/faults")
    async def tb_faults():
        r = await shop("GET", "/admin/faults")
        if r.status_code == 502:                       # the shop not running is a state to show, not a server error
            return JSONResponse(content=dict(active=[], reachable=False, error=json.loads(r.body)["error"]))
        return r

    @app.post("/api/testbed/clear")
    async def tb_clear():
        return await shop("POST", "/admin/clear")

    @app.get("/api/testbed/results")
    async def tb_results():
        p = REP / "testbed_frozen_config.json"
        cfg = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        small = None
        if cfg is not None:
            small = {k: cfg[k] for k in ("frozen_at", "model", "feature_set", "threshold", "learning_period_s", "train_runs", "test_runs")}
            small["alert_rule"] = cfg["alert_rule"]["name"]
        return JSONResponse(content=dict(limits=LIMITS, frozen=small, assignment=table("testbed_run_assignment.csv"), overall=table("testbed_heldout_overall.csv"), by_fault=table("testbed_heldout_by_fault.csv"),
                                         false_alarms=table("testbed_false_alarms.csv"), prefailure=table("testbed_prefailure.csv"), runid=table("testbed_runid_check.csv"), shuffled=table("testbed_shuffled_control.csv"),
                                         live=table("testbed_live_validation.csv"), rcaeval=table("rcaeval_external_results.csv"), cv_by_fault=table("testbed_cv_by_fault.csv")))
