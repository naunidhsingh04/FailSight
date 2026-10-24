"""Five hybrid deep-learning models on the LO2 slice (all 33 whole runs, 5,676 five-second windows), the SAME protocol as the AnoMod hybrids (hybrid_engine.py).
LO2 is trained completely separately: its own windows (lo2_data.py), its own preprocessing fitted on LO2's training part, its own models in models/hybrid/lo2/.
Nothing of AnoMod (weights, scalers, thresholds, features) is read here (checked by grep in proof P21).

python scripts/hybrid_lo2.py [--models A,B] [--labels ...] [--until HH:MM] [--assemble-only]
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import hybrid_engine as E                      # noqa: E402
from lo2_data import LO2Data                   # noqa: E402
from models_src.hybrid import LO2_HYBRIDS      # noqa: E402

REP, ROOT = E.REP, E.ROOT
PREFIX = "hybrid_lo2_"
STATE = ROOT / "outputs" / "hybrid_state" / "lo2"
MODELS = ROOT / "models" / "hybrid" / "lo2"
NAMES = list(LO2_HYBRIDS)


def logvolume_baseline(D):
    """control: a logistic regression on the log volume (total number of log lines in the window) ALONE, fitted on the 80% training part, scored on the test part"""
    from sklearn.linear_model import LogisticRegression
    j = D.out_names.index("log_total_volume") if "log_total_volume" in D.out_names else None
    rows = []
    W = D.W("final")
    for label in E.LABELS:
        y = D.y[label]
        yte = y[D.test_pos]
        m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(W[D.train_pos][:, [j]], y[D.train_pos])
        P = m.predict_proba(W[D.test_pos][:, [j]])
        mm = E.metrics_of(yte, P, P.argmax(1), D.class_names[label])
        b = E.block_bootstrap(yte, P, P.argmax(1), D.group_of(D.test_pos), 1000)
        rows.append(dict(model="LogisticRegression on log volume only", label_version=label, balanced_accuracy=mm["balanced_accuracy"], ba_ci_low=b["balanced_accuracy"][0], ba_ci_high=b["balanced_accuracy"][1],
                         roc_auc=mm["roc_auc"], auc_ci_low=b["roc_auc"][0], auc_ci_high=b["roc_auc"][1], normal_recall=mm["normal_recall"], n_test=mm["n"]))
    pd.DataFrame(rows).to_csv(REP / f"{PREFIX}logvolume_baseline.csv", index=False, float_format="%.6f")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(NAMES))
    ap.add_argument("--labels", default=",".join(E.LABELS))
    ap.add_argument("--until", default="")
    ap.add_argument("--assemble-only", action="store_true")
    a = ap.parse_args()
    D = LO2Data()
    E.log(f"LO2 data: train {len(D.train_pos)} windows / {D.df.run_id[D.train_pos].nunique()} runs, test {len(D.test_pos)} windows / {D.df.run_id[D.test_pos].nunique()} run, features {D.n_feat_out}, "
          f"graph nodes {D.node_names}, template embedding: {D.embedding_kind}, 5-fold context overlap (val windows with a context window in train) {D.context_overlap}")
    blocks = D.group_of(D.train_pos)
    assert all(len(set(blocks[D.fold5 == k]) & set(blocks[D.fold5 != k])) == 0 for k in range(1, 6))
    dead = None
    if a.until:
        h, m = map(int, a.until.split(":"))
        now = dt.datetime.now()
        t = now.replace(hour=h, minute=m, second=0, microsecond=0)
        dead = (t + dt.timedelta(days=1) if t < now - dt.timedelta(hours=1) else t).timestamp()
    if not a.assemble_only:
        E.imbalance_report(D, PREFIX)
        R = E.Runner(D, LO2_HYBRIDS, STATE, MODELS, PREFIX, deadline=dead, methods=("smote",))   # 80% training part always SMOTE-balanced, 20% test run untouched
        for name in a.models.split(","):
            for label in a.labels.split(","):
                R.run(name, label)
                E.assemble(D, NAMES, PREFIX, STATE, runid=None, n_boot=200)
    logvolume_baseline(D)
    runid = E.run_id_rule(D, None)
    lead = E.assemble(D, NAMES, PREFIX, STATE, runid=runid)
    E.log("assembled:\n" + lead.to_string())


if __name__ == "__main__":
    main()
