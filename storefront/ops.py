"""Presenter controls at /ops: plain buttons that really break the shop (through the shop's own fault injector at /admin/*), and a Fix button.
Only reachable from this computer. The fault parameters are computed exactly as the monitor does it (testbed.run_generator.severity_params, medium strength), so a break here is the same break as before."""
import json
import random
import sys
from pathlib import Path

import httpx
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ("127.0.0.1", "::1", "localhost", "testclient")

# the four plain choices; `type` is the real fault that is injected (the only place these names are shown)
CHOICES = [
    dict(id="slow-pages", label="Make pages slow", type="latency", note="every page takes longer to open", duration_s=90),
    dict(id="slow-db", label="Make the database slow", type="db_slow", note="the database answers slowly", duration_s=90),
    dict(id="page-crash", label="Make a page crash", type="exception_route", note="one page shows an error", duration_s=90),
    dict(id="server-crash", label="Make the server crash", type="crash", note="the whole shop stops for about 30 seconds", duration_s=None),
]
U_MEDIUM = 0.5


def register(app, render, shop_url):
    def local_only(request: Request):
        host = request.client.host if request.client else ""
        return host in LOCAL

    def forbidden():
        return JSONResponse(status_code=403, content={"error": "The presenter controls only work on the computer that runs the shop."})

    async def admin(method, path, **kw):
        try:
            async with httpx.AsyncClient(timeout=6) as c:
                r = await c.request(method, shop_url + path, **kw)
            return r.status_code, r.json()
        except Exception as e:                    # a crash fault kills the shop before it answers
            return 502, {"error": repr(e)}

    @app.get("/ops", response_class=HTMLResponse)
    async def ops_page(request: Request):
        if not local_only(request):
            return HTMLResponse("<h1>Not available</h1><p>The presenter controls only work on the computer that runs the shop.</p>", status_code=403)
        return render(request, "ops.html", title="Presenter controls", choices=CHOICES)

    @app.get("/ops/api/status")
    async def ops_status(request: Request):
        if not local_only(request):
            return forbidden()
        code, j = await admin("GET", "/admin/faults")
        if code == 502:
            return {"reachable": False, "active": []}
        return {"reachable": True, "active": [dict(type=f["type"], seconds_left=max(0, round(f["until"] - __import__("time").time())), failed=f.get("failed", False)) for f in j.get("active", [])]}

    @app.post("/ops/api/break")
    async def ops_break(request: Request):
        if not local_only(request):
            return forbidden()
        body = await request.json()
        choice_id = body.get("choice")
        if choice_id == "surprise":
            choice_id = random.choice([c["id"] for c in CHOICES])
        c = next((x for x in CHOICES if x["id"] == choice_id), None)
        if c is None:
            return JSONResponse(status_code=422, content={"error": "unknown choice"})
        sys.path.insert(0, str(ROOT))
        from testbed import run_generator as RG
        ftype = c["type"]
        kind = RG.CATALOGUE[ftype][1]
        dur = float(c["duration_s"] or (RG.fault_duration(ftype, U_MEDIUM, random.Random(1)) if kind == "gradual" or ftype in ("crash", "db_lock") else 60.0))
        params = RG.severity_params(ftype, U_MEDIUM, dur, random.Random(1))
        params = dict(params, severity=U_MEDIUM, severity_level="medium")
        code, j = await admin("POST", "/admin/fault", json=dict(type=ftype, duration_s=dur, params=params))
        if code == 502 and ftype == "crash":      # the shop dies before it can answer: that is the point of this choice
            return {"ok": True, "choice": c["id"], "label": c["label"], "note": c["note"], "type": ftype}
        if code != 200:
            return JSONResponse(status_code=502, content={"error": "The shop is not answering. Start it first (start_testbed.bat)."})
        return {"ok": True, "choice": c["id"], "label": c["label"], "note": c["note"], "type": ftype}

    @app.post("/ops/api/fix")
    async def ops_fix(request: Request):
        if not local_only(request):
            return forbidden()
        code, _ = await admin("POST", "/admin/clear")
        return {"ok": code == 200}
