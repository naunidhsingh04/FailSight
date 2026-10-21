"""Freeze the testbed detector BEFORE the live demos.

The model is the best supervised model by run-grouped cross-validation on the TRAIN runs (report_outputs/testbed_train_cv.csv; the test runs play no part), refitted on all
28 train runs exactly as in the held-out validation. The alert rule is fixed here as well. Both are written, with a timestamp and the hash of the model file, to
report_outputs/testbed_frozen_config.json and are never changed afterwards: this script refuses to overwrite an existing config (use --check to verify it).

python -m testbed.freeze            # create the frozen config and the model file (models/testbed/detector.joblib, git-ignored)
python -m testbed.freeze --check    # verify that the model file still matches the frozen config"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from testbed import dataset as DS, validate as V  # noqa: E402

CONFIG = ROOT / "report_outputs" / "testbed_frozen_config.json"
MODEL = ROOT / "models" / "testbed" / "detector.joblib"
ALERT_RULE = dict(name="2 of last 3 windows", k=2, n=3, why="decided before any live demo: it cut false alarms in the held-out validation (fault-free runs) at the cost of about 10 s more delay, and it is the rule the Console already uses")
LEARNING_S = 150


def sha256(p):
    h = hashlib.sha256()
    h.update(Path(p).read_bytes())
    return h.hexdigest()


def freeze():
    if CONFIG.exists():
        sys.exit(f"{CONFIG} already exists: the configuration is frozen and is not changed. Use --check to verify it.")
    cv = pd.read_csv(ROOT / "report_outputs" / "testbed_train_cv.csv")
    best = cv.sort_values("cv_balanced_accuracy_tuned", ascending=False).iloc[0]
    W, base, rel = V.load()
    asg = pd.read_csv(ROOT / "report_outputs" / "testbed_run_assignment.csv")
    train = list(asg[asg.split == "train"].run_id)
    use = (W.transition == 0).to_numpy() & W.run_id.isin(train).to_numpy()
    X = V.make_X(W, base, rel, best.feature_set)
    y = W.label_binary.to_numpy(int)
    model = V.make_model(best.model, y[use]).fit(X[use], y[use])
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    cols = list(base) + (list(rel) if best.feature_set in ("rel", "both") else [])
    if best.feature_set == "rel":
        cols = list(rel)
    joblib.dump(dict(model=model, feature_set=best.feature_set, columns=cols, base=list(base), rel_used=best.feature_set in ("rel", "both"), raw_used=best.feature_set in ("raw", "both")), MODEL)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    cfg = dict(frozen_at=time.strftime("%Y-%m-%d %H:%M:%S"), frozen_at_epoch=time.time(), purpose="live demo validation: nothing below is changed after a demo has been run",
               selection=dict(basis="best cv_balanced_accuracy_tuned over the 4 supervised models x 3 feature sets in run-grouped 7-fold cross-validation on the 28 train runs only",
                              source="report_outputs/testbed_train_cv.csv", chosen=best.to_dict(), all_candidates=cv.round(4).to_dict("records")),
               model=best.model, feature_set=best.feature_set, n_features=len(cols), threshold=float(best.tuned_threshold), threshold_source="maximises balanced accuracy of the out-of-fold predictions on the train runs",
               alert_rule=ALERT_RULE, learning_period_s=LEARNING_S, window_s=DS.WIN, step_s=DS.STEP, train_runs=train, train_windows=int(use.sum()),
               test_runs=list(asg[asg.split == "test"].run_id), model_file="models/testbed/detector.joblib (not in git)", model_sha256=sha256(MODEL),
               baseline_rule="robust z-score against the app's own past (scripts/universal_features.py: last 60 windows that ended before this one began, at least 9)",
               live_demo_seeds="9000 and above; never used for training or selection", git_commit=commit)
    CONFIG.write_text(json.dumps(cfg, indent=1, default=str), encoding="utf-8")
    print(f"frozen: {best.model} on {best.feature_set} features, threshold {cfg['threshold']:.3f}, alert rule '{ALERT_RULE['name']}', at {cfg['frozen_at']}")
    return cfg


def check():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    ok = MODEL.exists() and sha256(MODEL) == cfg["model_sha256"]
    print("frozen model file matches the config:", ok)
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    check() if a.check else freeze()
