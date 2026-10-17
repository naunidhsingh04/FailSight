"""Training pipeline: 10-fold CV with SMOTE inside folds, 4 deep models, XGBoost heads, stacking, soft voting.

usage: python scripts/train_models.py [--smoke] [--epochs N] [--phase train|report|explain|all]
"""
import sys, os, time, pickle, argparse, json
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
import numpy as np, pandas as pd
import tensorflow as tf, keras
from xgboost import XGBClassifier
from common import *
import models as M

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--phase", default="all")
ap.add_argument("--labels", default="label_binary,label_level")
args = ap.parse_args()
EPOCHS = args.epochs
keras.utils.set_random_seed(SEED)
CFGS = {"A": (64, 0.3), "B": (128, 0.4)}
BASE = ["CNN_BiGRU_Att", "Fusion", "Transformer", "CNN_BiLSTM_Att"]
HEAD_OF = {"Fusion": "Fusion_XGBhead", "CNN_BiGRU_Att": "CNN_BiGRU_Att_XGBhead"}
D = None


def xgb_head(Etr, ytr, C):
    clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, tree_method="hist", n_jobs=8,
                        random_state=SEED, verbosity=0)
    clf.fit(Etr, ytr)
    return clf


def fit_net(name, cfg, label, k, Xs, ys, Xval, yval, fixed_epochs=None, want_head=False, evals=()):
    """Train one network; returns dict with probs for each array in `evals` (name->X) plus optional head probs."""
    C = len(CLASS_NAMES[label])
    units, drop = CFGS[cfg]
    keras.backend.clear_session()
    keras.utils.set_random_seed(SEED + (0 if k == "final" else k) * 7 + BASE.index(name))
    model = M.build(name, Xs.shape[2], C, units, drop, (D.n_api, D.n_logs, D.n_met))
    t0 = time.time()
    if fixed_epochs is None:
        cb = [keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=2, min_lr=1e-5),
              keras.callbacks.EarlyStopping("val_loss", patience=5, restore_best_weights=True)]
        h = model.fit(Xs, ys, batch_size=256, epochs=EPOCHS, validation_data=(Xval, yval), callbacks=cb, verbose=0)
    else:
        cb = [keras.callbacks.ReduceLROnPlateau("loss", factor=0.5, patience=2, min_lr=1e-5)]
        h = model.fit(Xs, ys, batch_size=256, epochs=fixed_epochs, callbacks=cb, verbose=0)
    secs = time.time() - t0
    hist = {kk: [float(x) for x in v] for kk, v in h.history.items() if kk in ("loss", "val_loss")}
    n_ep = len(hist["loss"])
    best = int(np.argmin(hist["val_loss"])) + 1 if "val_loss" in hist else n_ep
    res = dict(secs=secs, hist=hist, epochs=n_ep, best_epoch=best, params=int(model.count_params()),
               layers=len(model.layers), probs={n: model.predict(X, batch_size=512, verbose=0) for n, X in evals.items()},
               s_per_epoch=secs / n_ep)
    if want_head:
        t1 = time.time()
        em = M.embed_model(model)
        clf = xgb_head(em.predict(Xs, batch_size=512, verbose=0), ys, C)
        res["head_probs"] = {n: clf.predict_proba(em.predict(X, batch_size=512, verbose=0)) for n, X in evals.items()}
        res["head_secs"] = time.time() - t1
    res["model"] = model
    return res


def fold_job(label, name, cfg, k):
    Xs, ys, c0, c1 = D.smote(label, k)
    Xtr_all, Xte = D.fold_arrays(k)
    tr, va = D.folds[k - 1]
    evals = {"train": Xtr_all[tr], "val": Xtr_all[va], "test": Xte}
    r = fit_net(name, cfg, label, k, Xs, ys, Xtr_all[va], D.ytr[label][va], want_head=name in HEAD_OF, evals=evals)
    r.pop("model")
    return r


def save_state(label, name, obj):
    with open(STATE / f"{label}__{name}.pkl", "wb") as f:
        pickle.dump(obj, f)


def load_state(label, name):
    p = STATE / f"{label}__{name}.pkl"
    return pickle.load(open(p, "rb")) if p.exists() else None


def cv_entry(label, r, k):
    tr, va = D.folds[k - 1]
    return dict(fold=k, train_proba=r["probs"]["train"], val_proba=r["probs"]["val"], test_proba=r["probs"]["test"],
                train_rows=tr, val_rows=va, hist=r["hist"], epochs=r["epochs"], best_epoch=r["best_epoch"], secs=r["secs"])


def run_base(label, name):
    if (st := load_state(label, name)) is not None:
        log(f"[skip] {label} {name} already done"); return
    C = len(CLASS_NAMES[label])
    t_all = time.time()
    sel, sel_scores = {}, {}
    for cfg in CFGS:
        sel[cfg] = {}
        for k in (1, 2):
            sel[cfg][k] = fold_job(label, name, cfg, k)
            log(f"  [{label}] {name} cfg {cfg} fold {k}: {sel[cfg][k]['epochs']} ep, {sel[cfg][k]['s_per_epoch']:.2f}s/ep, "
                f"val macroF1/BA {quick(D.ytr[label][D.folds[k-1][1]], sel[cfg][k]['probs']['val'])}")
        sel_scores[cfg] = float(np.mean([quick(D.ytr[label][D.folds[k - 1][1]], sel[cfg][k]["probs"]["val"])[0] for k in (1, 2)]))
    win = max(sel_scores, key=lambda c: (sel_scores[c], c == "A"))
    log(f"[{label}] {name}: config selection on folds 1-2 val macro-F1 {sel_scores} -> winner {win} {CFGS[win]}")
    runs = {k: sel[win][k] for k in (1, 2)}
    for k in range(3, 11):
        runs[k] = fold_job(label, name, win, k)
        log(f"  [{label}] {name} fold {k}: {runs[k]['epochs']} ep (best {runs[k]['best_epoch']}), {runs[k]['secs']:.0f}s, "
            f"val {quick(D.ytr[label][D.folds[k-1][1]], runs[k]['probs']['val'])}")
    cv_secs = time.time() - t_all
    fin_ep = max(5, int(round(np.mean([runs[k]["best_epoch"] for k in runs]))))
    Xs, ys, _, _ = D.smote(label, "final")
    Xtr_all, Xte = D.fold_arrays("final")
    fr = fit_net(name, win, label, "final", Xs, ys, None, None, fixed_epochs=fin_ep, want_head=name in HEAD_OF,
                 evals={"train": Xtr_all, "test": Xte})
    model = fr.pop("model")
    (OUT / "models").mkdir(exist_ok=True)
    model.save(OUT / "models" / f"{label}__{name}.keras")
    am = M.attn_model(model)
    base = dict(cfg=win, cfg_scores=sel_scores, units_drop=CFGS[win], params=fr["params"], layers=fr["layers"],
                cv_secs=cv_secs, final_secs=fr["secs"], final_epochs=fin_ep, final_hist=fr["hist"])
    folds = {k: cv_entry(label, runs[k], k) for k in runs}
    own = dict(base, kind="own", folds=folds, test_proba=fr["probs"]["test"], final_train_proba=fr["probs"]["train"])
    save_state(label, name, own)
    if name in HEAD_OF:
        hfolds = {}
        for k in runs:
            hr = dict(runs[k]); hr["probs"] = runs[k]["head_probs"]; hfolds[k] = cv_entry(label, hr, k)
        hd = dict(base, kind="xgb_head", folds=hfolds, test_proba=fr["head_probs"]["test"],
                  final_train_proba=fr["head_probs"]["train"], head_secs=sum(runs[k]["head_secs"] for k in runs) + fr["head_secs"])
        save_state(label, HEAD_OF[name], hd)
    log(f"[{label}] {name} DONE in {time.time()-t_all:.0f}s (final epochs {fin_ep})")


def run_ensembles(label):
    C = len(CLASS_NAMES[label])
    st = {n: load_state(label, n) for n in BASE}
    if any(v is None for v in st.values()):
        return
    # ---- soft voting over all fold models of the 4 architectures (OOF-style on validation)
    if load_state(label, "SoftVote") is None:
        folds = {}
        for k in range(1, 11):
            avg = lambda key: np.mean([st[n]["folds"][k][key] for n in BASE], axis=0)
            e = dict(fold=k, train_proba=avg("train_proba"), val_proba=avg("val_proba"), test_proba=avg("test_proba"),
                     train_rows=D.folds[k - 1][0], val_rows=D.folds[k - 1][1], hist=None, epochs=0, best_epoch=0, secs=0)
            folds[k] = e
        test = np.mean([st[n]["folds"][k]["test_proba"] for n in BASE for k in range(1, 11)], axis=0)
        ftrain = np.mean([st[n]["final_train_proba"] for n in BASE], axis=0)
        save_state(label, "SoftVote", dict(kind="ensemble", folds=folds, test_proba=test, final_train_proba=ftrain,
                                            params=0, layers=0, cv_secs=0, final_secs=0, cfg="-"))
        log(f"[{label}] SoftVote done (40 fold models averaged)")
    # ---- stacking: OOF probs of CNN_BiGRU_Att, Fusion, Transformer -> XGBoost meta-learner
    if load_state(label, "Stacking") is None:
        t0 = time.time()
        sb = ["CNN_BiGRU_Att", "Fusion", "Transformer"]
        n = len(D.train_pos)
        oof = {b: np.zeros((n, C)) for b in sb}
        for b in sb:
            for k in range(1, 11):
                oof[b][st[b]["folds"][k]["val_rows"]] = st[b]["folds"][k]["val_proba"]
        Z = np.hstack([oof[b] for b in sb])
        Zte = np.hstack([st[b]["test_proba"] for b in sb])
        Zfin_train = np.hstack([st[b]["final_train_proba"] for b in sb])
        y = D.ytr[label]
        mk = lambda: XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, tree_method="hist", n_jobs=8,
                                   random_state=SEED, verbosity=0)
        folds = {}
        for k in range(1, 11):
            tr, va = D.folds[k - 1]
            m = mk().fit(Z[tr], y[tr])
            folds[k] = dict(fold=k, train_proba=m.predict_proba(Z[tr]), val_proba=m.predict_proba(Z[va]),
                            test_proba=m.predict_proba(Zte), train_rows=tr, val_rows=va, hist=None, epochs=0, best_epoch=0, secs=0)
        m = mk().fit(Z, y)
        save_state(label, "Stacking", dict(kind="ensemble", folds=folds, test_proba=m.predict_proba(Zte),
                                           final_train_proba=m.predict_proba(Z), params=0, layers=0,
                                           cv_secs=time.time() - t0, final_secs=0, cfg="-"))
        log(f"[{label}] Stacking done ({time.time()-t0:.0f}s). Meta-learner trained on OOF probabilities of the 80% only.")


def fold_counts():
    rows = []
    for label in LABELS:
        for k in range(1, 11):
            _, _, c0, c1 = D.smote(label, k)
            for ci, cn in enumerate(CLASS_NAMES[label]):
                rows.append(dict(label_version=label, fold=k, **{"class": cn}, before_smote=int(c0[ci]), after_smote=int(c1[ci])))
    pd.DataFrame(rows).to_csv(OUT / "fold_class_counts_for_graph.csv", index=False)
    log("saved fold_class_counts_for_graph.csv")


def smoke():
    label = "label_level"
    for name in BASE:
        Xs, ys, _, _ = D.smote(label, 1)
        Xtr_all, Xte = D.fold_arrays(1)
        tr, va = D.folds[0]
        t0 = time.time()
        r = fit_net(name, "A", label, 1, Xs, ys, Xtr_all[va], D.ytr[label][va], evals={"val": Xtr_all[va]})
        log(f"SMOKE {name}: {r['epochs']} epochs, {r['s_per_epoch']:.2f} s/epoch, params {r['params']}, "
            f"val {quick(D.ytr[label][va], r['probs']['val'])}")
        smoke.t[name] = (r["s_per_epoch"], r["epochs"])
        if name == "CNN_BiGRU_Att":  # full run of one model on one fold incl. head
            pass
smoke.t = {}


if __name__ == "__main__":
    log("=" * 70); log("start", time.ctime(), "args", vars(args))
    log("NOTE: sequences are indexed by their last window, so validation sequences overlap training windows and CV scores "
        "are optimistic; the TEST score is the main result; each class comes from a single scenario run (per system).")
    log("NOTE: TensorFlow on native Windows has no GPU support -> CPU only.")
    D = Data()
    log(f"split check: train {len(D.train_pos)} test {len(D.test_pos)} ({100*len(D.train_pos)/len(D.w):.2f}% train)")
    log(f"context windows padded (repeat earliest) in {D.n_padded_slots} of {len(D.w)*T} slots; context never crosses train/test")
    log(f"feature groups: api {D.n_api}, logs {D.n_logs}, metrics {D.n_met} (+ system_is_TT)")
    if args.smoke:
        smoke()
        sys.exit()
    if args.phase in ("train", "all"):
        fold_counts()
        for label in args.labels.split(","):
            for name in BASE:
                run_base(label, name)
            run_ensembles(label)
            D._sm.clear()
    if args.phase in ("report", "all"):
        import report; report.main(D)
    if args.phase in ("explain", "all"):
        import explain; explain.main(D)
