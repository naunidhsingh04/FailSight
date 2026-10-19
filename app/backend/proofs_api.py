"""Proof report for the web app: serves report_outputs/proofs.json (the file scripts/proofs.py writes), re-runs the script and streams its terminal output,
and renders a self-contained printable HTML version of the same JSON. The page never computes numbers of its own."""
import html
import json
import subprocess
import sys
import threading

from . import config

PATH = config.REPORTS / "proofs.json"
SECTIONS = [f"P{i}" for i in range(1, 11)]
_lock = threading.Lock()


def load():
    return json.loads(PATH.read_text(encoding="utf-8")) if PATH.exists() else None


def busy() -> bool:
    return _lock.locked()


def run_stream(sections=None, quick=False):
    """sync generator of terminal output lines; only one run at a time (the caller checks busy() first)"""
    cmd = [sys.executable, "-u", str(config.SCRIPTS / "proofs.py")]
    if sections:
        cmd += ["--only", ",".join(sections)]
    if quick:
        cmd += ["--quick"]
    with _lock:
        yield f"$ python scripts/proofs.py {' '.join(cmd[3:])}\n"
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(config.ROOT), bufsize=1, env={**__import__('os').environ, "PYTHONUNBUFFERED": "1"})
        try:
            for line in proc.stdout:
                if line.startswith(("WARNING", "I0000", "W0000")) or "tf.function retracing" in line or "oneDNN" in line:
                    continue                                   # TensorFlow start-up noise is not part of the report
                yield line
        finally:
            rc = proc.wait()
        yield f"\n[finished: exit code {rc}]\n"


CSS = """
body{font:14px/1.5 Georgia,serif;max-width:60rem;margin:2rem auto;padding:0 1rem;color:#111;background:#fff}
h1{font-size:1.8rem;margin:0 0 .2rem} h2{font-size:1.2rem;margin:2rem 0 .5rem;border-bottom:2px solid #111;padding-bottom:.2rem}
.meta{color:#444;margin:0 0 1rem} .badge{display:inline-block;font:700 11px/1 Arial,sans-serif;padding:3px 7px;border:1px solid #111;border-radius:3px;margin-right:.4rem}
.PASS{background:#fff}.WARN{background:#111;color:#fff}.FAIL{background:#000;color:#fff;text-decoration:underline}.SKIP{background:#eee}
.item{margin:0 0 1rem;page-break-inside:avoid} .claim{font-weight:700}
pre{font:12px/1.35 Consolas,Menlo,monospace;background:#f4f4f4;border:1px solid #ccc;padding:.5rem;white-space:pre-wrap;overflow-wrap:anywhere;margin:.3rem 0}
.src{color:#444;font-size:12px} table{border-collapse:collapse;width:100%;font:12px Arial,sans-serif} td,th{border:1px solid #999;padding:3px 6px;text-align:left;vertical-align:top}
@media print{body{margin:0;max-width:none} .noprint{display:none}}
"""


def render_html(rep: dict) -> str:
    e = html.escape
    out = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>FailSight proof report</title><style>{CSS}</style></head><body>",
           "<h1>FailSight proof report</h1>",
           f"<p class='meta'>Generated {e(rep['generated_at'])} &middot; commit {e(rep.get('git_commit', '')[:10])} &middot; totals {e(json.dumps(rep['totals']))}{' &middot; <b>partial report</b>' if rep.get('partial') else ''}</p>",
           "<p>Every number is recomputed from the data or the saved models, or read from a named result file. The live demo source is a <b>replay of recorded AnoMod data</b>, not a production system.</p>",
           "<p class='noprint'>Use the browser's Print command to save this page as PDF.</p>"]
    for sec in rep["sections"]:
        out.append(f"<h2>{e(sec['id'])} &mdash; {e(sec['title'])} <span class='badge {e(sec['status'])}'>{e(sec['status'])}</span></h2>")
        for it in sec["items"]:
            out.append(f"<div class='item'><span class='badge {e(it['status'])}'>{e(it['status'])}</span><span class='claim'>{e(it['claim'])}</span><pre>{e(chr(10).join(it['evidence']))}</pre><div class='src'>source: {e(it['source'])}</div></div>")
    out.append("<h2>Summary</h2><table><tr><th>#</th><th>claim</th><th>status</th><th>source file</th></tr>")
    for r in rep["summary"]:
        out.append(f"<tr><td>{e(r['section'])}</td><td>{e(r['claim'])}</td><td>{e(r['status'])}</td><td>{e(r['source'])}</td></tr>")
    out.append("</table></body></html>")
    return "".join(out)
