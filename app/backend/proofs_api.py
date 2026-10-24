"""Proof report for the web app: serves report_outputs/proofs.json (the file scripts/proofs.py writes), re-runs the script and streams its terminal output,
and renders a self-contained printable HTML version of the same JSON. The page never computes numbers of its own."""
import html
import json
import subprocess
import sys
import threading

from . import config, print_report

PATH = config.REPORTS / "proofs.json"
SECTIONS = [f"P{i}" for i in range(1, 25)]
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
.PASS{background:#fff}.WARN{background:#111;color:#fff}.FAIL{background:#000;color:#fff;text-decoration:underline}.SKIP{background:#eee}.REPORTED{background:#e8eef5;border-style:dashed}.result{font-style:italic}
.item{margin:0 0 1rem;page-break-inside:avoid} .claim{font-weight:700}
pre{font:12px/1.35 Consolas,Menlo,monospace;background:#f4f4f4;border:1px solid #ccc;padding:.5rem;white-space:pre-wrap;overflow-wrap:anywhere;margin:.3rem 0}
.src{color:#444;font-size:12px} table{border-collapse:collapse;width:100%;font:12px Arial,sans-serif} td,th{border:1px solid #999;padding:3px 6px;text-align:left;vertical-align:top}
.pb{page-break-before:always} .toc ol{columns:2} .note{font-style:italic;color:#333} h3{font-size:1rem;margin:1rem 0 .3rem} dt{font-weight:700;margin-top:.5rem} dd{margin:0 0 0 1rem}
.cover{border:2px solid #111;padding:1rem;margin:1rem 0} .cover b{font-family:Arial,sans-serif}
@page{size:A4;margin:16mm 14mm 18mm;@bottom-center{content:'FailSight proof report - page ' counter(page) ' of ' counter(pages);font:10px Arial}}
@media print{body{margin:0;max-width:none;font-size:11.5px} .noprint{display:none} table{page-break-inside:auto} tr{page-break-inside:avoid} a{color:#111;text-decoration:none}}
"""


def _cover(rep: dict) -> str:
    t = rep.get("totals", {})
    cells = " &middot; ".join(f"<b>{e_(str(v))}</b> {e_(str(k))}" for k, v in t.items())
    return (f"<div class='cover'><p><b>What this report is.</b> Proof that the numbers in the FailSight app are real: the data split, the SMOTE balancing of the 80% practice part, the untouched 20% test part, "
            f"the ten hybrid deep-learning models on AnoMod, on LO2 and on both together, and every check behind them.</p><p><b>Status of the checks:</b> {cells}. "
            f"Generated {e_(rep.get('generated_at', ''))}, commit {e_(rep.get('git_commit', '')[:10])}.</p></div>")


e_ = html.escape


def render_html(rep: dict) -> str:
    e = html.escape
    out = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>FailSight proof report</title><style>{CSS}</style></head><body>",
           "<h1>FailSight proof report</h1>",
           f"<p class='meta'>Generated {e(rep['generated_at'])} &middot; commit {e(rep.get('git_commit', '')[:10])} &middot; totals {e(json.dumps(rep['totals']))}{' &middot; <b>partial report</b>' if rep.get('partial') else ''}</p>",
           "<p>Every number is recomputed from the data or the saved models, or read from a named result file. The live demo source is a <b>replay of recorded AnoMod data</b>, not a production system.</p>",
           "<p class='noprint'>Use the browser's Print command to save this page as PDF.</p>",
           _cover(rep), print_report.overview(rep), "<h2 class='pb'>5. Proof sections</h2>"]
    for sec in rep["sections"]:
        out.append(f"<h2>{e(sec['id'])} &mdash; {e(sec['title'])} <span class='badge {e(sec['status'].split()[0])}'>{e(sec['status'])}</span></h2>")
        for it in sec["items"]:
            out.append(f"<div class='item'><span class='badge {e(it['status'].split()[0])}'>{e(it['status'])}</span><span class='claim'>{e(it['claim'])}</span>{(' <div class=result>Result: ' + e(it['result']) + '</div>') if it.get('result') else ''}<pre>{e(chr(10).join(it['evidence']))}</pre><div class='src'>source: {e(it['source'])}</div></div>")
    out.append("<h2 class='pb'>6. Summary of checks</h2><table><tr><th>#</th><th>claim</th><th>status</th><th>source file</th></tr>")
    for r in rep["summary"]:
        out.append(f"<tr><td>{e(r['section'])}</td><td>{e(r['claim'])}</td><td>{e(r['status'])}</td><td>{e(r['source'])}</td></tr>")
    out.append("</table>" + print_report.closing(rep) + "</body></html>")
    return "".join(out)
