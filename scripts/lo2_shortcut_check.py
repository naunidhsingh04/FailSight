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
    pd.DataFrame(rows).to_csv(REP / "lo2_shortcut_check.csv", index=False, float_format="%.4f")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
