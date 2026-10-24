"""Diagnostic for the LO2 binary labels: why is the binary task almost trivially separable on the LO2 slice?
In every LO2 run the 'correct' (normal) scenario is the FIRST of the 54 scenarios (scenario_order_in_run == 1), so a window of that scenario also differs in system state (warm-up) and the
scenario itself runs 60 s instead of 10 s. This script scores the same RF (100 trees, class-balanced) on subsets of the LO2 inputs, trained on the training runs and scored on the test run:
only log counts / only log-template counts / only metric components / log volume only. It reports; it is not used to choose or change any model.
python scripts/lo2_shortcut_check.py -> report_outputs/lo2_shortcut_check.csv"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lo2_data import LO2Data  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # noqa: E402

REP = HERE.parent / "report_outputs"


def main():
    D = LO2Data()
    W = D.W("final")
    names = D.out_names
    y = D.y["label_binary"]
    sets = {"all inputs": list(range(len(names))), "log counts and rates only": [i for i, n in enumerate(names) if n.startswith("log_")], "log-template counts only": [i for i, n in enumerate(names) if n.startswith("tmpl_")],
            "metric components only": [i for i, n in enumerate(names) if n.startswith("m_pc")], "log volume only": [names.index("log_total_volume")] if "log_total_volume" in names else []}
    rows = []
    for k, idx in sets.items():
        if not idx:
            continue
        m = RandomForestClassifier(100, n_jobs=2, random_state=42, class_weight="balanced").fit(W[D.train_pos][:, idx], y[D.train_pos])
        P = m.predict_proba(W[D.test_pos][:, idx])[:, 1]
        pred = (P >= 0.5).astype(int)
        rows.append(dict(inputs=k, n_inputs=len(idx), balanced_accuracy=balanced_accuracy_score(y[D.test_pos], pred), roc_auc=roc_auc_score(y[D.test_pos], P), n_test=len(D.test_pos),
                         note="diagnostic only; test part scored, nothing chosen from it"))
    # controls for the warm-up / run-length confound: the normal scenario is first in a run and lasts 60 s. If that alone made the task easy, (1) normal windows from the late part of the scenario
    # would be missed, (2) anomalies from early scenarios would be missed, (3) a matched subset (late normal windows vs anomalies of scenarios 2-4) would be hard.
    tr, te = D.train_pos, D.test_pos
    wi, od = D.df.window_index.to_numpy(), D.df.scenario_order_in_run.to_numpy()
    allx = list(range(len(names)))
    warn = [i for i, n in enumerate(names) if n.endswith(("_WARN", "_ERROR", "_error_rate", "_warn_rate"))]
    nolog = [i for i, n in enumerate(names) if not n.startswith(("log_", "tmpl_"))]
    def rf(idx, a, b):
        m = RandomForestClassifier(200, n_jobs=2, random_state=42, class_weight="balanced").fit(W[a][:, idx], y[a])
        P = m.predict_proba(W[b][:, idx])[:, 1]
        return P, (P >= 0.5).astype(int)
    P, pr = rf(warn, tr, te)
    rows.append(dict(inputs="WARN / ERROR log counts and rates only", n_inputs=len(warn), balanced_accuracy=balanced_accuracy_score(y[te], pr), roc_auc=roc_auc_score(y[te], P), n_test=len(te), note="diagnostic only"))
    P, pr = rf(allx, tr, te)
    yt = y[te]
    for lab, sel in (("normal windows from the first 30 s of the normal scenario (recall)", (yt == 0) & (wi[te] < 6)), ("normal windows from the late part of the normal scenario (recall)", (yt == 0) & (wi[te] >= 6)),
                     ("anomalies of scenarios 2-3 (recall)", (yt == 1) & (od[te] <= 3)), ("anomalies of scenarios 4-10 (recall)", (yt == 1) & (od[te] >= 4) & (od[te] <= 10)), ("anomalies of scenarios 11-54 (recall)", (yt == 1) & (od[te] >= 11))):
        rows.append(dict(inputs=lab, n_inputs=len(allx), balanced_accuracy=float((pr[sel] == yt[sel]).mean()), roc_auc=np.nan, n_test=int(sel.sum()), note="share of these test windows called correctly (not a balanced accuracy)"))
    mt = ((y[tr] == 0) & (wi[tr] >= 6)) | ((y[tr] == 1) & (od[tr] <= 4))
    ms = ((yt == 0) & (wi[te] >= 6)) | ((yt == 1) & (od[te] <= 4))
    for lab, idx in (("matched subset (late normal windows vs anomalies of scenarios 2-4), all inputs", allx), ("matched subset, metric components only", nolog)):
        P, pr = rf(idx, tr[mt], te[ms])
        rows.append(dict(inputs=lab, n_inputs=len(idx), balanced_accuracy=balanced_accuracy_score(yt[ms], pr), roc_auc=roc_auc_score(yt[ms], P), n_test=int(ms.sum()), note="diagnostic only; same warm-up state on both sides"))
    pd.DataFrame(rows).to_csv(REP / "lo2_shortcut_check.csv", index=False, float_format="%.4f")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
