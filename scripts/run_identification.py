"""Check A: run identification. Can a throw-away classifier tell WHICH of the 26 runs a test window comes from? If yes, a model could score well on the labels simply by
recognising the run (every window of a run carries the run's single label), and the test-set scores would say little about detecting faults.

  1. XGBoost (throw-away, never saved) predicts run_id (26 classes) from the 124 model inputs of a window, trained on the training windows, scored on the test
     windows. Repeated with only the metric inputs, only API + log inputs, and only the 27 generic metric aggregates. Accuracy with 95% block-bootstrap CI.
     (Run-aware where possible: the project's split is by 10-window blocks, so every run has windows on both sides; a held-out-run split cannot recognise a run.)
  2. The "run-identification rule": predict the run, then output that run's own label. Its test balanced accuracy / Normal recall is the ceiling of what pure run
     recognition can reach.
  3. How often the saved final models' test predictions equal the label of the predicted run (agreement), next to their own accuracy.

python scripts/run_identification.py [--boot 1000]        -> report_outputs/run_identification.csv
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
REP = ROOT / "report_outputs"
OUT = REP / "run_identification.csv"
SEED = 20260103


def block_ci(blocks, correct, n_boot, seed=SEED):
    """accuracy and 95% block-bootstrap CI of a 0/1 vector"""
    ids, inv = np.unique(blocks, return_inverse=True)
    n_b = np.bincount(inv, minlength=len(ids)).astype(float)
    c_b = np.bincount(inv, weights=correct.astype(float), minlength=len(ids))
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(ids), (n_boot, len(ids)))
    cnt = np.stack([np.bincount(d, minlength=len(ids)) for d in draw]).astype(float)
    acc = (cnt @ c_b) / (cnt @ n_b)
    return float(c_b.sum() / n_b.sum()), float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))


def rule_metrics(blocks, y, pred, n_classes, n_boot, seed=SEED):
    """balanced accuracy, macro-F1 and Normal recall (class 0) of a decision, with 95% block-bootstrap CIs"""
    ids, inv = np.unique(blocks, return_inverse=True)
    nb = len(ids)
    cm = np.zeros((nb, n_classes, n_classes))
    np.add.at(cm, (inv, y, pred), 1)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, nb, (n_boot, nb))
    cnt = np.stack([np.bincount(d, minlength=nb) for d in draw]).astype(float)

    def f(c):
        tp = np.diagonal(c, axis1=-2, axis2=-1)
        rs, cs = c.sum(-1), c.sum(-2)
        with np.errstate(invalid="ignore", divide="ignore"):
            rec = np.where(rs > 0, tp / rs, np.nan)
            f1 = np.where(rs + cs > 0, 2 * tp / (rs + cs), np.nan)
        return np.nanmean(rec, -1), np.nanmean(f1, -1), rec[..., 0]

    pt = f(cm.sum(0))
    bt = f((cnt @ cm.reshape(nb, -1)).reshape(-1, n_classes, n_classes))
    out = {}
    for name, p, b in zip(("balanced_accuracy", "macro_f1", "normal_recall"), pt, bt):
        out[name] = float(p)
        out[f"{name}_ci_low"], out[f"{name}_ci_high"] = (float(x) for x in np.nanpercentile(b, [2.5, 97.5]))
    return out


def setup():
    sys.argv = ["x", "--stages", "0"]
    import full_pipeline as FP
    import v2_data
    from v2_data import DataV2
    from app.backend.registry import Registry
    v2_data.common.LOG = ROOT / "outputs" / "run_identification_log.txt"
    D = DataV2()
    FP.D = D
    return D, FP, Registry().load()


def feature_sets(D, reg):
    names = reg.prep.output_names
    k = len(reg.prep.kept_cols)
    kept = names[:k]                                    # the 124 model inputs (the 3 was_missing_* indicators and the system flag come after them)
    W = D.W("final")[:, :k]
    idx = lambda pred: [i for i, n in enumerate(kept) if pred(n)]
    g = pd.read_parquet(ROOT / "processed" / "generic_metric_windows.parquet").set_index("window_id").reindex(D.df.window_id)
    G = g.to_numpy(float)
    return {"all 124 model inputs": W, "metric inputs only": W[:, idx(lambda n: n.startswith("m_"))],
            "API + log inputs only": W[:, idx(lambda n: n.startswith(("api_", "log_")))], "generic metric aggregates only (27)": G}


def run_id_model(X, y, tr, te):
    from xgboost import XGBClassifier
    m = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=42, verbosity=0)
    m.fit(X[tr], y[tr])
    return m.predict(X[te])


def compute(n_boot, log=print):
    D, FP, reg = setup()
    df = D.df
    runs = sorted(df.run_id.unique())
    rid = df.run_id.map({r: i for i, r in enumerate(runs)}).to_numpy(int)
    tr, te = D.train_pos, D.test_pos
    blocks = df.block_id.to_numpy()[te]
    run_bin = df.groupby("run_id").label_binary.first().reindex(runs).to_numpy(int)
    run_lvl = df.groupby("run_id").apply(lambda g: D.y["label_level"][g.index[0]]).reindex(runs).to_numpy(int)
    rows = []
    sets = feature_sets(D, reg)
    preds = {}
    for name, X in sets.items():
        p = run_id_model(X, rid, tr, te)
        preds[name] = p
        acc, lo, hi = block_ci(blocks, p == rid[te], n_boot)
        same_sys = float(np.mean(df.system.to_numpy()[te] == np.array([r.split("|")[0] for r in np.array(runs)[p]])))
        rows.append(dict(kind="run_identification", feature_set=name, n_features=X.shape[1], n_train=len(tr), n_test=len(te), accuracy=acc, accuracy_ci_low=lo, accuracy_ci_high=hi,
                         chance_26_runs=1 / 26, system_correct=same_sys))
        log(f"  run id from {name:<40} accuracy {acc:.3f} ({lo:.3f}-{hi:.3f})  system right {same_sys:.3f}")
    # the rule: predicted run -> that run's own label
    p = preds["all 124 model inputs"]
    for lab, arr, C in (("label_binary", run_bin, 2), ("label_level", run_lvl, 5)):
        rule = arr[p]
        m = rule_metrics(blocks, D.yte[lab], rule, C, n_boot)
        rows.append(dict(kind="run_id_rule", label_version=lab, feature_set="all 124 model inputs", accuracy=float((rule == D.yte[lab]).mean()), **m))
        log(f"  run-identification rule {lab}: balanced accuracy {m['balanced_accuracy']:.3f}, Normal recall {m['normal_recall']:.3f}")
    # saved models
    Wf = D.W("final")
    X, seq = Wf[te], D.seq(Wf, te)
    for (lab, name), e in reg.entries.items():
        if not e["available"] or name == "SoftVote":
            continue
        C = 2 if lab == "label_binary" else 5
        pm = reg.predict_proba(lab, name, X, seq).argmax(1)
        arr = run_bin if lab == "label_binary" else run_lvl
        rule = arr[p]
        agree = (pm == rule)
        ok_model = pm == D.yte[lab]
        ok_rule = rule == D.yte[lab]
        a, lo, hi = block_ci(blocks, agree, n_boot)
        mm = rule_metrics(blocks, D.yte[lab], pm, C, n_boot)
        rows.append(dict(kind="saved_model_vs_rule", label_version=lab, model=name, agreement=a, agreement_ci_low=lo, agreement_ci_high=hi, model_accuracy=float(ok_model.mean()),
                         rule_accuracy=float(ok_rule.mean()), share_of_correct_also_correct_by_rule=float((ok_model & ok_rule).sum() / max(ok_model.sum(), 1)),
                         balanced_accuracy=mm["balanced_accuracy"], balanced_accuracy_ci_low=mm["balanced_accuracy_ci_low"], balanced_accuracy_ci_high=mm["balanced_accuracy_ci_high"]))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    t0 = time.time()
    df = compute(a.boot, log=lambda s: print(s, flush=True))
    df.insert(0, "generated", time.strftime("%Y-%m-%d %H:%M:%S"))
    df.to_csv(OUT, index=False, float_format="%.6f")
    print(f"wrote {OUT} ({len(df)} rows, {time.time() - t0:.0f} s)")
    s = df[df.kind == "saved_model_vs_rule"]
    print(s[["label_version", "model", "agreement", "model_accuracy", "rule_accuracy", "balanced_accuracy"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
