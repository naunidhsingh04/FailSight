"""Executes notebooks/FailSight_Workflow.ipynb in FAST mode the way a fresh Colab runtime would: a clean clone of the committed repository (no AnoMod raw folders, no models folder, no inputs),
the data bundle unzipped next to it, and a virtual environment that contains only what Colab has (see colab_venv_test, created beside the repository).
Writes notebooks/FailSight_Workflow_executed.ipynb (outputs kept, huge ones stripped), notebooks/FailSight_Workflow_executed.html and report_outputs/notebook_check.json (read by proof P23).

python scripts/run_notebook_check.py [--venv PATH] [--timeout 2400]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "FailSight_Workflow.ipynb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venv", default=str(ROOT.parent / "colab_venv_test" / "Scripts" / "python.exe"))
    ap.add_argument("--timeout", type=int, default=2400)
    a = ap.parse_args()
    if not Path(a.venv).exists():
        sys.exit(f"venv python not found: {a.venv}")
    work = Path(tempfile.mkdtemp(prefix="failsight_nb_"))
    bundle = work / "bundle"
    with zipfile.ZipFile(ROOT / "failsight_bundle.zip") as z:
        z.extractall(bundle)
    env = dict(os.environ, FAILSIGHT_CLONE_SOURCE=os.environ.get("FAILSIGHT_CHECK_SOURCE", str(ROOT)), FAILSIGHT_BUNDLE=str(bundle), FAILSIGHT_WORK=str(work), PYTHONIOENCODING="utf-8", TF_CPP_MIN_LOG_LEVEL="3")
    for k in list(env):
        if k.startswith("PYTHON") and k not in ("PYTHONIOENCODING",):
            env.pop(k)
    nb_in = work / "FailSight_Workflow.ipynb"
    shutil.copy2(NB, nb_in)
    t0 = time.time()
    cmd = [a.venv, "-m", "nbconvert", "--to", "notebook", "--execute", "--ExecutePreprocessor.timeout=1800", "--ExecutePreprocessor.allow_errors=True", "--output", "executed.ipynb", str(nb_in)]
    print("running:", " ".join(cmd))
    r = subprocess.run(cmd, env=env, cwd=str(work), capture_output=True, text=True, timeout=a.timeout)
    secs = time.time() - t0
    print(r.stdout[-500:], r.stderr[-1500:])
    ex = work / "executed.ipynb"
    if not ex.exists():
        sys.exit("execution failed: no output notebook")
    nb = json.load(open(ex, encoding="utf-8"))
    n_err, soft, long_outputs = 0, [], 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        keep = []
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                n_err += 1
            txt = "".join(o.get("text", [])) if o.get("output_type") == "stream" else ""
            for ln in txt.splitlines():
                if "could not run" in ln or "missing files (this part is skipped)" in ln:
                    soft.append(ln[:200])
            if len(json.dumps(o)) > 400_000:                                    # huge outputs are stripped to keep the saved notebook small
                long_outputs += 1
                o = dict(output_type="stream", name="stdout", text=["[output removed: too large]\n"])
            if o.get("output_type") == "stream" and txt.count("\n") > 200:
                long_outputs += 1
                o["text"] = (txt.splitlines(True)[:200]) + ["[... output cut at 200 lines]\n"]
            keep.append(o)
        c["outputs"] = keep
    numbers = {}
    nj = work / "FailSight" / "notebook_numbers.json"
    if nj.exists():
        numbers = json.load(open(nj, encoding="utf-8"))
    comps = numbers.get("comparisons", []) if isinstance(numbers, dict) else []
    dest = ROOT / "notebooks" / "FailSight_Workflow_executed.ipynb"
    json.dump(nb, open(dest, "w", encoding="utf-8"), indent=1)
    subprocess.run([a.venv, "-m", "nbconvert", "--to", "html", "--output", "FailSight_Workflow_executed", "--output-dir", str(ROOT / "notebooks"), str(dest)], capture_output=True, text=True, env=env)
    chk = dict(executed_ok=bool(n_err == 0 and not soft), executed_at=time.strftime("%Y-%m-%d %H:%M:%S"), n_cells=len(nb["cells"]), n_errors=n_err, soft_failures=soft, seconds=secs,
               environment="fresh virtual environment (numpy, pandas, scikit-learn, matplotlib, tensorflow, xgboost, lightgbm + the notebook's own pip install of imbalanced-learn, shap, optuna, lime); clean clone of the committed repository; data bundle unzipped; no AnoMod raw folders, no models folder",
               comparisons=comps, outputs_trimmed=long_outputs)
    (ROOT / "report_outputs" / "notebook_check.json").write_text(json.dumps(chk, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in chk.items() if k != "comparisons"}, indent=1)[:1500])
    print("comparisons:", len(comps), "different:", sum(not c["equal"] for c in comps))
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
