"""Classical baselines (LogisticRegression, RandomForest, ExtraTrees, XGBoost, LightGBM): run-aware 10-fold CV engine."""
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from v2_data import *

XGB_DEVICE = "cpu"       # set to "cuda" by stage 0 only if the GPU test passes

DEFAULTS = {
    "LR": dict(C=1.0),
    "RF": dict(n_estimators=300, max_depth=None, min_samples_leaf=1, max_features="sqrt"),
    "ET": dict(n_estimators=300, max_depth=None, min_samples_leaf=1, max_features="sqrt"),
    "XGB": dict(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=1.0, colsample_bytree=1.0, min_child_weight=1, reg_lambda=1.0),
    "LGBM": dict(n_estimators=300, num_leaves=31, max_depth=-1, learning_rate=0.1, subsample=1.0, colsample_bytree=1.0,
                 min_child_samples=20, reg_lambda=0.0),
}


def make_clf(name, params, variant, C, early_stop=False):
    cw = variant == "cw"
    p = dict(params)
    if name == "LR":
        return LogisticRegression(max_iter=3000, class_weight="balanced" if cw else None, **p)
    if name in ("RF", "ET"):
        cls = RandomForestClassifier if name == "RF" else ExtraTreesClassifier
        if "max_samples" in p and name == "ET":
            p["bootstrap"] = True
        return cls(n_jobs=-1, random_state=SEED, class_weight=("balanced_subsample" if name == "RF" else "balanced") if cw else None, **p)
    if name == "XGB":
        kw = dict(tree_method="hist", n_jobs=-1, random_state=SEED, verbosity=0, device=XGB_DEVICE)
        if early_stop:
            kw["early_stopping_rounds"] = 25
        return XGBClassifier(**p, **kw)
    if name == "LGBM":
        if p.get("subsample", 1.0) < 1.0:
            p["subsample_freq"] = 1
        return LGBMClassifier(n_jobs=-1, random_state=SEED, verbose=-1, class_weight="balanced" if cw else None, **p)
    raise ValueError(name)


def _xgb_weights(name, variant, y, C):
    """XGBoost: binary -> scale_pos_weight (set in params by caller); multi-class -> balanced sample weights"""
    return compute_sample_weight("balanced", y) if (name == "XGB" and variant == "cw" and C > 2) else None


def fit_model(name, params, variant, Xtr, ytr, C, pre=None, early_stop=False):
    """returns a fitted estimator with predict_proba. variant 'cw' = class weights, 'smote' = SMOTE on this training part only
    (imblearn Pipeline, or the pre-resampled arrays `pre` = (Xs, ys) when given). early_stop (XGB/LGBM only): 15% inner holdout of the
    training part is used purely to stop adding trees."""
    params = dict(params)
    if name == "XGB" and variant == "cw" and C == 2:
        params["scale_pos_weight"] = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))   # class 1 (Anomaly) is the positive class
    k = min(5, int(np.bincount(ytr).min()) - 1)
    es = early_stop and name in ("XGB", "LGBM")
    if es:
        Xa, Xb, ya, yb = train_test_split(Xtr, ytr, test_size=0.15, stratify=ytr, random_state=SEED)
        if variant == "smote":
            Xa, ya = SMOTE(random_state=SEED, k_neighbors=min(5, int(np.bincount(ya).min()) - 1)).fit_resample(Xa, ya)
        clf = make_clf(name, params, variant, C, early_stop=True)
        sw = _xgb_weights(name, variant, ya, C)
        if name == "XGB":
            clf.fit(Xa, ya, sample_weight=sw, eval_set=[(Xb, yb)], verbose=False)
        else:
            import lightgbm as lgb
            clf.fit(Xa, ya, eval_set=[(Xb, yb)], callbacks=[lgb.early_stopping(25, verbose=False)])
        return clf
    clf = make_clf(name, params, variant, C)
    if variant == "smote":
        if pre is not None:
            clf.fit(pre[0], pre[1])
            return clf
        pipe = Pipeline([("smote", SMOTE(random_state=SEED, k_neighbors=k)), ("clf", clf)])
        pipe.fit(Xtr, ytr)
        return pipe
    sw = _xgb_weights(name, variant, ytr, C)
    if sw is not None:
        clf.fit(Xtr, ytr, sample_weight=sw)
    else:
        clf.fit(Xtr, ytr)
    return clf


_FOLD_CACHE = {}


def fold_data(D, label, k, need_smote=False):
    key = (label, k)
    if key not in _FOLD_CACHE:
        tr, va = D.folds[k - 1]
        W = D.W(k)
        ytr = D.y[label][tr]
        _FOLD_CACHE[key] = dict(Xtr=W[tr], ytr=ytr, Xva=W[va], yva=D.y[label][va], smote=None)
    d = _FOLD_CACHE[key]
    if need_smote and d["smote"] is None:
        d["smote"] = smote_resample(d["Xtr"], d["ytr"])
    return d


def clear_cache():
    _FOLD_CACHE.clear()


def tab_cv(D, label, name, params, variant, use_cache=False, early_stop=False, verbose=False):
    C = len(CLASS_NAMES[label])
    folds, t_all = {}, time.time()
    for k in range(1, 11):
        tr, va = D.folds[k - 1]
        t0 = time.time()
        d = fold_data(D, label, k, need_smote=use_cache and variant == "smote")
        m = fit_model(name, params, variant, d["Xtr"], d["ytr"], C, pre=d["smote"] if use_cache else None, early_stop=early_stop)
        folds[k] = dict(train_rows=tr, val_rows=va, train_proba=m.predict_proba(d["Xtr"]).astype(np.float32),
                        val_proba=m.predict_proba(d["Xva"]).astype(np.float32), secs=time.time() - t0)
    return dict(name=name, label=label, params=params, variant=variant, early_stop=early_stop, folds=folds, secs=time.time() - t_all)
