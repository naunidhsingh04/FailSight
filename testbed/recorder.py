"""Recorder: a tiny stand-in for the monitor that speaks the same ingestion protocol (POST /ingest/api|logs|metrics) and writes every event to JSONL files.
The run generator uses it to record the raw events of a testbed run exactly as the agent sent them. python -m testbed.recorder --dir <run dir> --port 8801"""
import argparse
import asyncio
import json
import sys
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[1]


def create_app(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {k: open(out / f"{k}.jsonl", "a", encoding="utf-8") for k in ("api", "logs", "metrics")}
    lock = threading.Lock()
    app = FastAPI(docs_url=None, openapi_url=None)
    counts = {k: 0 for k in files}

    @app.post("/ingest/{kind}")
    async def ingest(kind: str, request: Request):
        if kind not in files:
            return JSONResponse({"error": "unknown kind"}, status_code=404)
        body = await request.json()
        recv = time.time()
        events = body.get("events", [])
        with lock:
            f = files[kind]
            for e in events:
                e["_recv"] = recv
                f.write(json.dumps(e) + "\n")
            f.flush()
            counts[kind] += len(events)
        return {"accepted": len(events)}

    @app.get("/stats")
    def stats():
        return counts

    @app.on_event("shutdown")
    def close():
        for f in files.values():
            f.close()

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--port", type=int, default=8801)
    a = ap.parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run(create_app(a.dir), host="127.0.0.1", port=a.port, log_level="warning", loop="none", access_log=False)


if __name__ == "__main__":
    main()
