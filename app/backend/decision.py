"""The saved decision rule of a model (report_outputs/thresholds.json): binary = threshold on P(Anomaly); five-class = per-class weights."""
import numpy as np

from . import results


def apply_decision(label: str, name: str, p: np.ndarray):
    spec = results.thresholds()[label][name]
    if spec["type"] == "threshold":
        pred = (p[:, 1] >= spec["threshold"]).astype(int)
        used = dict(type="threshold", on="P(Anomaly)", value=spec["threshold"])
    else:
        pred = (p * np.array(spec["weights"])).argmax(1)
        used = dict(type="class_weights", weights=spec["weights"])
    used.update(oof_normal_recall=spec["oof_normal_recall"], oof_balanced_accuracy=spec["oof_balanced_accuracy"], oof_balanced_accuracy_default_argmax=spec["oof_balanced_accuracy_default_argmax"],
                normal_recall_ge_0_70_reachable=spec["normal_recall_ge_0_70_reachable"])
    return pred, used
