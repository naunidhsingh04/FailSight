"""LO2 family (d): the four deep-learning models and their XGBoost-head versions. The deep model is trained with the hybrid protocol (hybrid_engine.Runner); while each fold / the final
model exists, an XGBoost head (150 trees, depth 3) is fitted on the model's pre-head vector of the SMOTE-balanced (or class-weighted) training part and scores the validation / test part.
The head versions are therefore trained on exactly the same folds and training windows as their deep model."""
import json
import time
from pathlib import Path

import numpy as np

import hybrid_engine as E
from models_src.hybrid.dl_models import DL_MODELS
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

STATE = E.ROOT / "outputs" / "hybrid_state" / "lo2_full"
PREFIX = "lo2_full_"
MDIR = E.ROOT / "models" / "hybrid" / "lo2"
HEAD = dict(n_estimators=150, max_depth=3, learning_rate=0.1, subsample=0.7, colsample_bytree=0.6, min_child_weight=3, reg_lambda=5.0)


def head_fit(m, Xs, ys, method):
    em = lambda X: np.asarray(m.embed(X.astype("float32"))) if len(X) <= 256 else np.concatenate([np.asarray(m.embed(X[i:i + 256].astype("float32"))) for i in range(0, len(X), 256)])
    C = int(ys.max()) + 1
    clf = XGBClassifier(tree_method="hist", n_jobs=2, random_state=E.SEED, verbosity=0, **HEAD)
    clf.fit(em(Xs), ys, sample_weight=compute_sample_weight("balanced", ys) if method == "cw" else None)

    def pr(X):
        P = clf.predict_proba(em(X)).astype(np.float32)
        if P.shape[1] != C:
            full = np.zeros((len(X), C), np.float32)
            full[:, clf.classes_] = P
            P = full
        return P
    return pr


def hook_fold(m, Xs, ys, Xtr, Xva, method):
    pr = head_fit(m, Xs, ys, method)
    return dict(head_train=pr(Xtr), head_val=pr(Xva))


def make_hook_final(D):
    def hook_final(m, Xs, ys, Xtr, Xte, tag, name, label, method):
        pr = head_fit(m, Xs, ys, method)
        np.save(STATE / f"{'test_proba' if tag == 'real' else 'shuf_proba'}__{label}__{name}_XGBhead.npy", pr(Xte))
    return hook_final


def head_record(D, name, label, rec):
    """result record of NAME_XGBhead from the head probabilities stored with the folds of the chosen stage"""
    rp = STATE / f"result__{label}__{name}_XGBhead.json"
    if rp.exists() or not (STATE / f"test_proba__{label}__{name}_XGBhead.npy").exists() or not (STATE / f"shuf_proba__{label}__{name}_XGBhead.npy").exists():
        return
    C = len(D.class_names[label])
    ytr = D.y[label][D.train_pos]
    folds = [E.load_pkl(STATE / f"fold__{label}__{name}__{rec['chosen_tag']}__f{k}.pkl") for k in range(1, 6)]
    for f in folds:
        f["train_proba"], f["val_proba"] = f["head_train"], f["head_val"]
    R = E.Runner.__new__(E.Runner)
    R.D = D
    s = E.Runner.cv_summary(R, label, folds)
    s.insert(0, "stage", rec["chosen_tag"])
    oof = E.Runner.oof(R, label, folds)
    np.save(STATE / f"oof__{label}__{name}_XGBhead.npy", oof)
    th = E.tune_threshold(ytr, oof)
    pte = np.load(STATE / f"test_proba__{label}__{name}_XGBhead.npy")
    gap = float(s.train_f1.mean() - s.val_f1.mean())
    r = dict(dataset="lo2", label=label, model=name + "_XGBhead", seed=E.SEED, n_train=len(D.train_pos), n_test=len(D.test_pos), n_folds=5, notes=[f"XGBoost head on the pre-head vector of {name}"], did_not_learn=rec["did_not_learn"],
             screening=rec["screening"], cv_before=dict(cfg=HEAD, method=rec["chosen_method"], val_ba=float(s.val_ba.mean()), val_f1=float(s.val_f1.mean()), train_f1=float(s.train_f1.mean()), gap=gap,
                                                          diagnosis="overfit (train-validation macro-F1 gap > 0.10)" if gap > 0.10 else "ok"),
             chosen_tag="cv", chosen_cfg=dict(hidden=rec["chosen_cfg"]["hidden"], head=HEAD), chosen_method=rec["chosen_method"], cv_rows=s.to_dict("records"), thresholds=th,
             final_smote=rec["final_smote"], final=dict(rec["final"], train_ba=float("nan"), train_f1=float("nan")), secs_total=rec["secs_total"])
    r["cv_rows"] = [dict(row, stage="cv") for row in r["cv_rows"]]
    json.dump(r, open(rp, "w"), indent=1, default=float)


def run_all(D, deadline):
    reg = DL_MODELS
    R = E.Runner(D, reg, STATE, MDIR, PREFIX, deadline=deadline)
    R.hook_fold, R.hook_final = hook_fold, make_hook_final(D)
    for name in reg:
        for label in E.LABELS:
            rec = R.run(name, label)
            if rec is not None:
                head_record(D, name, label, rec)
