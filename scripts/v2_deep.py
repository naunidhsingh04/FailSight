"""Deep models (Keras): 10-fold CV with the epoch policy max 150 / EarlyStopping 12 / ReduceLROnPlateau 4, XGBoost heads on embeddings."""
import time
import numpy as np
import tensorflow as tf
import keras
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

import models as M
from v2_data import *

DEEP_CFG = {"orig": dict(units=64, drop=0.3, l2=0.0), "corr": dict(units=32, drop=0.4, l2=1e-4)}
HEAD_ORIG = dict(n_estimators=300, max_depth=6, learning_rate=0.1)                       # as specified
HEAD_CORR = dict(n_estimators=150, max_depth=3, learning_rate=0.1, subsample=0.7, colsample_bytree=0.6, min_child_weight=5, reg_lambda=5.0)


def fold_model_path(label, name, tag, k):
    return MODELS / "folds" / f"{label}__{name}__{tag}__f{k}.keras"


def train_net(name, Xs, ys, Xval, yval, C, tag, seed, method, max_epochs=MAX_EPOCHS, fixed_epochs=None):
    cfg = DEEP_CFG[tag]
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    model = M.build(name, Xs.shape[2], C, cfg["units"], cfg["drop"], DATA.groups, l2=cfg["l2"])
    cw = balanced_class_weight(ys) if method == "cw" else None
    t0 = time.time()
    if fixed_epochs is None:
        cb = [keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=LR_PATIENCE, min_lr=1e-5),
              keras.callbacks.EarlyStopping("val_loss", patience=ES_PATIENCE, restore_best_weights=True)]
        h = model.fit(Xs, ys, batch_size=256, epochs=max_epochs, validation_data=(Xval, yval), callbacks=cb, class_weight=cw, verbose=0)
    else:
        cb = [keras.callbacks.ReduceLROnPlateau("loss", factor=0.5, patience=LR_PATIENCE, min_lr=1e-5)]
        h = model.fit(Xs, ys, batch_size=256, epochs=fixed_epochs, callbacks=cb, class_weight=cw, verbose=0)
    hist = {k: [float(x) for x in v] for k, v in h.history.items() if k in ("loss", "val_loss")}
    n = len(hist["loss"])
    best = int(np.argmin(hist["val_loss"])) + 1 if "val_loss" in hist else n
    return model, dict(hist=hist, epochs=n, best_epoch=best, secs=time.time() - t0, cap=max_epochs,
                       hit_cap=bool(fixed_epochs is None and n >= max_epochs), params=int(model.count_params()), layers=len(model.layers))


DATA = None          # set by the pipeline (DataV2)


def prepare_train(D, label, k, method):
    tr, va = D.folds[k - 1]
    W = D.W(k)
    Xtr, ytr = D.seq(W, tr), D.y[label][tr]
    Xva, yva = D.seq(W, va), D.y[label][va]
    if method == "smote":
        Xs, ys = smote_seq(Xtr, ytr)
    else:
        Xs, ys = Xtr, ytr
    if FAST:                                   # dry-run only: tiny training set so the whole pipeline code path runs in minutes
        keep = np.unique(np.concatenate([np.arange(0, len(ys), 8), [int(np.where(ys == c)[0][0]) for c in np.unique(ys)]]))
        Xs, ys = Xs[keep], ys[keep]
    return Xtr, ytr, Xva, yva, Xs, ys


def deep_fold(D, label, name, tag, method, k, max_epochs):
    """one fold; fold-level result is cached on disk so an interrupted model resumes at the next fold"""
    p = state_path("s3f", label, name, tag, f"f{k}")
    r = load_pkl(p)
    if r is not None:
        return r
    C = len(CLASS_NAMES[label])
    tr, va = D.folds[k - 1]
    Xtr, ytr, Xva, yva, Xs, ys = prepare_train(D, label, k, method)
    model, info = train_net(name, Xs, ys, Xva, yva, C, tag, SEED + 7 * k + DEEP.index(name), method, max_epochs)
    model.save(fold_model_path(label, name, tag, k))
    r = dict(train_rows=tr, val_rows=va, train_proba=model.predict(Xtr, batch_size=512, verbose=0).astype(np.float32),
             val_proba=model.predict(Xva, batch_size=512, verbose=0).astype(np.float32), **info)
    flag = f"  ** {NOT_CONVERGED} (hit {max_epochs} epochs) **" if info["hit_cap"] else ""
    log(f"    [{label}] {name}/{tag} fold {k}: {info['epochs']} epochs run (best {info['best_epoch']}), {info['secs']:.0f}s, "
        f"{info['secs']/info['epochs']:.2f}s/epoch, val BA {quick(yva, r['val_proba'])[1]:.3f}{flag}")
    save_pkl(p, r)
    return r


def deep_cv(D, label, name, tag, method, max_epochs=MAX_EPOCHS):
    t0 = time.time()
    folds = {k: deep_fold(D, label, name, tag, method, k, max_epochs) for k in range(1, 11)}
    return dict(name=name, label=label, tag=tag, variant=method, folds=folds, secs=sum(f["secs"] for f in folds.values()),
                params=folds[1]["params"], layers=folds[1]["layers"], cfg=DEEP_CFG[tag], max_epochs=max_epochs,
                n_not_converged=sum(f["hit_cap"] for f in folds.values()))


def head_cv(D, label, name, tag, method, head_params):
    """XGBoost head on the 64-d embedding of the saved fold models (training embeddings only)"""
    C = len(CLASS_NAMES[label])
    folds, t0 = {}, time.time()
    for k in range(1, 11):
        tr, va = D.folds[k - 1]
        Xtr, ytr, Xva, yva, Xs, ys = prepare_train(D, label, k, method)
        model = keras.models.load_model(fold_model_path(label, name, tag, k))
        em = M.embed_model(model)
        Etr_s, Etr, Eva = (em.predict(a, batch_size=512, verbose=0) for a in (Xs, Xtr, Xva))
        clf = XGBClassifier(tree_method="hist", n_jobs=-1, random_state=SEED, verbosity=0, **head_params)
        clf.fit(Etr_s, ys, sample_weight=compute_sample_weight("balanced", ys) if method == "cw" else None)
        folds[k] = dict(train_rows=tr, val_rows=va, train_proba=clf.predict_proba(Etr).astype(np.float32),
                        val_proba=clf.predict_proba(Eva).astype(np.float32), secs=0.0)
    return dict(name=name + "_XGBhead", label=label, tag=tag, variant=method, folds=folds, secs=time.time() - t0, head_params=head_params)
