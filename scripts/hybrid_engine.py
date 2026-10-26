"""One protocol for the hybrid deep-learning models, used unchanged for AnoMod and for the LO2 slice.

A dataset adapter `D` provides (see hybrid_anomod.py / hybrid_lo2.py):
  D.ds                    "anomod" or "lo2"
  D.df                    window table (window_id, run_id, block_id, ...)
  D.train_pos/test_pos    row positions of the 80% training part / 20% test part (made FIRST, written to a split file)
  D.folds                 list of 5 (train_rows, val_rows), run/block-aware, no group shared across folds
  D.y[label]              integer labels; D.class_names[label]; label 0 = healthy ("Normal")
  D.W(k) / D.seq(W,pos)   windows transformed with the preprocessor fitted on the training rows of fold k (k = 1..5 or "final"), cut into (n, T, F) sequences
  D.spec(label)           dict with T, F, C, node_idx, adj, ... for the model constructors; D.binary_cols = indices of 0/1 columns (rounded after SMOTE)
  D.group_of(pos)         block id of each row (for the bootstrap)
Every result is saved the moment it exists (state dir), so a restart skips finished work.
"""
import gc
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRA_OP_PARALLELISM", "2")
os.environ.setdefault("TF_NUM_INTER_OP_PARALLELISM", "1")
import tensorflow as tf  # noqa: E402
import keras  # noqa: E402
from imblearn.over_sampling import SMOTE, ADASYN  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, average_precision_score, accuracy_score  # noqa: E402

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except RuntimeError:
    pass

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "report_outputs"
SEED = 42
BATCH = 32
MAX_EPOCHS, PATIENCE = 40, 6
SCREEN_EPOCHS = 10
MEM_LIMIT_GB = 5.5
LABELS = ["label_binary", "label_level"]
LOGF = REP / "hybrid_engine_log.txt"


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(LOGF, "a", encoding="utf-8") as f:
        f.write(s + "\n")


def wait_memory(limit=MEM_LIMIT_GB):
    me = psutil.Process()                          # this training process (with children) must stay under 5.5 GB and the machine must keep 0.6 GB free
    for _ in range(60):
        used = (me.memory_info().rss + sum(c.memory_info().rss for c in me.children(recursive=True))) / 2 ** 30
        if used < limit and psutil.virtual_memory().available / 2 ** 30 > 0.6:
            return used
        gc.collect()
        time.sleep(10)
    return used


def save_pkl(p, o):
    tmp = str(p) + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(o, f)
    os.replace(tmp, p)


def load_pkl(p):
    return pickle.load(open(p, "rb")) if Path(p).exists() else None


# ----------------------------------------------------------------------------------------------- imbalance
def smote_safe(X, y, binary_idx=(), seed=SEED):
    """SMOTE on a training part only. Brings every class with >= 2 windows up to the biggest class; k_neighbors = largest valid value (min 1);
    a class with < 2 windows is skipped. Returns Xs, ys, counts before, counts after, skipped classes."""
    C = int(y.max()) + 1
    cnt = np.bincount(y, minlength=C)
    skipped = [c for c in range(C) if 0 < cnt[c] < 2]
    strat = {c: int(cnt.max()) for c in range(C) if cnt[c] >= 2 and cnt[c] < cnt.max()}
    if not strat:
        return X, y, cnt, cnt, skipped
    k = max(1, min(5, min(int(cnt[c]) for c in strat) - 1))
    Xs, ys = SMOTE(sampling_strategy=strat, k_neighbors=k, random_state=seed).fit_resample(X, y)
    return Xs, ys.astype(int), cnt, np.bincount(ys, minlength=C), skipped


def smote_seq(X, y, binary_cols=()):
    n, T, F = X.shape
    Xs, ys, b, a, sk = smote_safe(X.reshape(n, -1), y)
    Xs = Xs.reshape(-1, T, F).astype(np.float32)
    for c in binary_cols:
        Xs[:, :, c] = np.clip(np.round(Xs[:, :, c]), 0, 1)
    return Xs, ys, b, a, sk


def balanced_sample_weight(y):
    c = np.bincount(y)
    w = len(y) / (len(c) * np.maximum(c, 1))
    return w[y].astype("float32")


def cnt_str(c):
    return "/".join(str(int(v)) for v in c)


# ----------------------------------------------------------------------------------------------- training
def train_model(m, Xs, ys, Xv, yv, C, cfg, method, max_epochs, sample_weight=None, fixed_epochs=None, lr_scale=1.0):
    """compile and train one keras model with the protocol shared by every hybrid. Early stopping on the validation fold (patience 6, at most 40 epochs) unless fixed_epochs
    is given (final model, no validation data). Returns the loss history."""
    m.compile(optimizer=keras.optimizers.AdamW(learning_rate=cfg["lr"] * lr_scale, weight_decay=cfg["wd"], clipnorm=1.0),
              loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05))
    Y = keras.utils.to_categorical(ys, C).astype("float32")
    sw = sample_weight if sample_weight is not None else (balanced_sample_weight(ys) if method == "cw" else None)
    if fixed_epochs is None:
        cb = [keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=3, min_lr=1e-5),
              keras.callbacks.EarlyStopping("val_loss", patience=PATIENCE, restore_best_weights=True)]
        h = m.fit(Xs, Y, batch_size=BATCH, epochs=max_epochs, validation_data=(Xv, keras.utils.to_categorical(yv, C).astype("float32")),
                  sample_weight=sw, callbacks=cb, verbose=0, shuffle=True)
    else:
        cb = [keras.callbacks.ReduceLROnPlateau("loss", factor=0.5, patience=3, min_lr=1e-5)]
        h = m.fit(Xs, Y, batch_size=BATCH, epochs=fixed_epochs, sample_weight=sw, callbacks=cb, verbose=0, shuffle=True)
    return {k: [float(x) for x in v] for k, v in h.history.items() if k in ("loss", "val_loss")}


def fit_hybrid(builder, spec, cfg, Xs, ys, Xv, yv, method, max_epochs, seed, sample_weight=None, fixed_epochs=None, lr_scale=1.0):
    """train one hybrid. A two-stage hybrid (builder.two_stage) trains its first algorithm's function, then hands the variables it produced to the second algorithm's function,
    which trains on them; both stages use train_model. Otherwise the builder returns one model that is trained end to end."""
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    C = spec["C"]
    t0 = time.time()
    stages = []
    if getattr(builder, "two_stage", False):
        def fit(model, X, y, Xv_, yv_):
            stages.append(train_model(model, X, y, Xv_, yv_, C, cfg, method, max_epochs, sample_weight, fixed_epochs, lr_scale))
        m = builder(spec, cfg, data=dict(X=Xs, y=ys, Xv=Xv, yv=yv, fit=fit))
    else:
        m = builder(spec, cfg)
        stages.append(train_model(m, Xs, ys, Xv, yv, C, cfg, method, max_epochs, sample_weight, fixed_epochs, lr_scale))
    hist = stages[-1]                                  # the final classifier (the second stage) is the one that is validated
    n = sum(len(h["loss"]) for h in stages)
    best = int(np.argmin(hist["val_loss"])) + 1 if "val_loss" in hist else len(hist["loss"])
    return m, dict(epochs=n, best_epoch=best, secs=time.time() - t0, loss_first=hist["loss"][0], loss_last=hist["loss"][-1], params=int(m.count_params()),
                   val_loss_best=float(min(hist["val_loss"])) if "val_loss" in hist else None)


def predict(m, X):
    return m.predict(X, batch_size=256, verbose=0).astype(np.float32)


def bal_f1(y, P):
    pred = P.argmax(1)
    return float(balanced_accuracy_score(y, pred)), float(f1_score(y, pred, average="macro", labels=np.unique(y), zero_division=0))


def normal_recall(y, P):
    m = y == 0
    return float((P[m].argmax(1) == 0).mean()) if m.any() else float("nan")


# ----------------------------------------------------------------------------------------------- the per-model procedure
class Runner:
    def __init__(self, D, registry, state_dir, model_dir, prefix, deadline=None, model_budget_s=45 * 60, methods=("smote", "cw")):
        self.methods = tuple(methods)          # imbalance methods tried in screening; LO2 passes ("smote",) so its 80% training part is always SMOTE-balanced
        self.D, self.reg, self.state, self.mdir, self.prefix = D, registry, Path(state_dir), Path(model_dir), prefix
        self.state.mkdir(parents=True, exist_ok=True)
        self.mdir.mkdir(parents=True, exist_ok=True)
        self.deadline, self.budget = deadline, model_budget_s
        self.hook_fold = self.hook_final = None          # optional hooks (the XGBoost-head versions of the deep-learning models use them)

    def sp(self, *parts):
        return self.state / ("__".join(str(p) for p in parts))

    # -- one CV fold
    def fold(self, name, label, cfg, method, k, max_epochs, tag, lr_scale=1.0):
        p = self.sp("fold", label, name, tag, f"f{k}.pkl")
        r = load_pkl(p)
        if r is not None:
            return r
        D = self.D
        tr, va = D.folds[k - 1]
        W = D.W(k)
        Xtr, ytr, Xva, yva = D.seq(W, tr), D.y[label][tr], D.seq(W, va), D.y[label][va]
        D.drop_W(k)
        b = np.bincount(ytr, minlength=len(D.class_names[label]))
        if method == "smote":
            Xs, ys, b, a, sk = smote_seq(Xtr, ytr, D.binary_cols)
        else:
            Xs, ys, a, sk = Xtr, ytr, b, []
        hold = np.bincount(yva, minlength=len(b))
        m, info = fit_hybrid(self.reg[name], D.spec(label), cfg, Xs, ys, Xva, yva, method, max_epochs, SEED + 7 * k, lr_scale=lr_scale)
        ptr, pva = predict(m, Xtr), predict(m, Xva)
        extra = self.hook_fold(m, Xs, ys, Xtr, Xva, method) if self.hook_fold else {}
        r = dict(**extra, k=k, train_rows=tr, val_rows=va, train_proba=ptr, val_proba=pva, b=cnt_str(b), a=cnt_str(a), hold=cnt_str(hold), skipped=sk, **info)
        del m, Xs, Xtr, Xva
        keras.backend.clear_session()
        gc.collect()
        save_pkl(p, r)
        vb, vf = bal_f1(yva, pva)
        log(f"    {self.D.ds} {label} {name}/{tag} fold {k}: {info['epochs']} ep (best {info['best_epoch']}), {info['secs']:.0f}s, val BA {vb:.3f} F1 {vf:.3f}, "
            f"train F1 {bal_f1(ytr, ptr)[1]:.3f}  SMOTE b:{r['b']} a:{r['a']} hold:{r['hold']}")
        return r

    def cv(self, name, label, cfg, method, tag, max_epochs=MAX_EPOCHS, lr_scale=1.0, t_model0=None):
        folds = []
        for k in range(1, len(self.D.folds) + 1):
            me = max_epochs
            if t_model0 and time.time() - t_model0 > self.budget * 0.5 and k > 1:
                me = min(me, 20)                      # reduced epochs for the rest of this model (recorded in the fold rows)
            folds.append(self.fold(name, label, cfg, method, k, me, tag, lr_scale))
        return folds

    def cv_summary(self, label, folds):
        D = self.D
        rows = []
        for f in folds:
            yv = D.y[label][f["val_rows"]]
            yt = D.y[label][f["train_rows"]]
            vb, vf = bal_f1(yv, f["val_proba"])
            tb, tf_ = bal_f1(yt, f["train_proba"])
            rows.append(dict(fold=f["k"], val_ba=vb, val_f1=vf, val_normal_recall=normal_recall(yv, f["val_proba"]), train_ba=tb, train_f1=tf_,
                             epochs=f["epochs"], best_epoch=f["best_epoch"], secs=f["secs"], smote_b=f["b"], smote_a=f["a"], hold=f["hold"]))
        d = pd.DataFrame(rows)
        return d

    def oof(self, label, folds):
        D = self.D
        C = len(D.class_names[label])
        idx = {p: i for i, p in enumerate(D.train_pos)}
        oof = np.zeros((len(D.train_pos), C), np.float32)
        for f in folds:
            oof[[idx[p] for p in f["val_rows"]]] = f["val_proba"]
        return oof

    # -- everything for one model and one label version
    def run(self, name, label):
        rp = self.sp("result", label, f"{name}.json")
        if Path(rp).exists():
            log(f"skip {self.D.ds} {label} {name} (done)")
            return json.load(open(rp))
        if self.deadline and time.time() > self.deadline:
            log(f"time box reached: {name} {label} not started")
            return None
        used = wait_memory()
        log(f"=== {self.D.ds} {label} {name}  (memory in use {used:.1f} GB)")
        D = self.D
        t_model0 = time.time()
        spec, C = D.spec(label), len(D.class_names[label])
        rec = dict(dataset=D.ds, label=label, model=name, seed=SEED, n_train=len(D.train_pos), n_test=len(D.test_pos), n_folds=len(D.folds), notes=[])
        # 1. screening on fold 1, validation scores only, at most 10 epochs
        scr = []
        for hid in (32, 64):
            for method in self.methods:
                cfg = dict(hidden=hid, dropout=0.3, wd=1e-4, lr=1e-3)
                r = self.fold(name, label, cfg, method, 1, SCREEN_EPOCHS, f"screen_h{hid}_{method}")
                yv = D.y[label][r["val_rows"]]
                vb, vf = bal_f1(yv, r["val_proba"])
                scr.append(dict(hidden=hid, method=method, val_ba=vb, val_f1=vf, epochs=r["epochs"], loss_first=r["loss_first"], loss_last=r["loss_last"], secs=r["secs"]))
        best = max(scr, key=lambda s: (round(s["val_ba"], 4), -s["hidden"]))
        chance = 1.0 / C
        broken = best["val_ba"] < chance + 0.05 or best["loss_last"] > 0.98 * best["loss_first"]
        lr_scale = 1.0
        if broken:
            rec["notes"].append("screening looked broken (chance-level validation or loss not falling); bug check = shapes/finite outputs/gradients (unit tests) passed; one retry with lr/2")
            cfg = dict(hidden=best["hidden"], dropout=0.3, wd=1e-4, lr=1e-3)
            r = self.fold(name, label, cfg, best["method"], 1, SCREEN_EPOCHS, f"screen_retry_h{best['hidden']}_{best['method']}", lr_scale=0.5)
            vb, vf = bal_f1(D.y[label][r["val_rows"]], r["val_proba"])
            scr.append(dict(hidden=best["hidden"], method=best["method"] + "_retry_lr/2", val_ba=vb, val_f1=vf, epochs=r["epochs"], loss_first=r["loss_first"], loss_last=r["loss_last"], secs=r["secs"]))
            if vb >= chance + 0.05 and r["loss_last"] <= 0.98 * r["loss_first"]:
                lr_scale, broken = 0.5, False
        rec["screening"], rec["did_not_learn"] = scr, bool(broken)
        cfg = dict(hidden=best["hidden"], dropout=0.3, wd=1e-4, lr=1e-3)
        method = best["method"]
        # 2. 5-fold cross-validation of the chosen configuration
        folds = self.cv(name, label, cfg, method, "cv", lr_scale=lr_scale, t_model0=t_model0)
        s0 = self.cv_summary(label, folds)
        gap0 = float(s0.train_f1.mean() - s0.val_f1.mean())
        rec["cv_before"] = dict(cfg=cfg, method=method, val_ba=float(s0.val_ba.mean()), val_f1=float(s0.val_f1.mean()), train_f1=float(s0.train_f1.mean()), gap=gap0,
                                diagnosis="overfit (train-validation macro-F1 gap > 0.10)" if gap0 > 0.10 else "ok")
        chosen, chosen_folds, chosen_tag = cfg, folds, "cv"
        # 3. correction stage if overfit: stronger regularisation, adopted only if cross-validated validation BA is not worse (decided on validation, never on test)
        if gap0 > 0.10:
            cfg2 = dict(hidden=max(16, cfg["hidden"] // 2), dropout=min(0.6, cfg["dropout"] + 0.2), wd=cfg["wd"] * 20, lr=cfg["lr"])
            folds2 = self.cv(name, label, cfg2, method, "corr", lr_scale=lr_scale, t_model0=t_model0)
            s1 = self.cv_summary(label, folds2)
            gap1 = float(s1.train_f1.mean() - s1.val_f1.mean())
            adopt = float(s1.val_ba.mean()) >= float(s0.val_ba.mean()) - 0.01
            rec["cv_after"] = dict(cfg=cfg2, method=method, val_ba=float(s1.val_ba.mean()), val_f1=float(s1.val_f1.mean()), train_f1=float(s1.train_f1.mean()), gap=gap1, adopted=bool(adopt),
                                   diagnosis="overfit (train-validation macro-F1 gap > 0.10)" if gap1 > 0.10 else "ok")
            if adopt:
                chosen, chosen_folds, chosen_tag = cfg2, folds2, "corr"
        rec["chosen_tag"], rec["chosen_cfg"], rec["chosen_method"] = chosen_tag, chosen, method
        sc = self.cv_summary(label, chosen_folds)
        sc.insert(0, "stage", chosen_tag)
        rec["cv_rows"] = sc.to_dict("records")
        if chosen_tag == "corr":
            rec["cv_rows"] += [dict(stage="cv", **r) for r in s0.to_dict("records")]
        oof = self.oof(label, chosen_folds)
        np.save(self.sp("oof", label, f"{name}.npy"), oof)
        ytr = D.y[label][D.train_pos]
        rec["thresholds"] = tune_threshold(ytr, oof)
        epochs_final = int(max(5, np.median([f["best_epoch"] for f in chosen_folds])))
        # 4. freeze the configuration in a file BEFORE the test part is scored
        frozen = dict(dataset=D.ds, label=label, model=name, seed=SEED, config=chosen, method=method, final_epochs=epochs_final, lr_scale=lr_scale, thresholds=rec["thresholds"],
                      chosen_stage=chosen_tag, frozen_at=time.strftime("%Y-%m-%d %H:%M:%S"), note="written before the test windows were scored; every choice was made on training folds only")
        (REP / f"{self.prefix}frozen_{label}__{name}.json").write_text(json.dumps(frozen, indent=1, default=float))
        # 5. final model on the full 80% (SMOTE once on the full training part), then the test part is scored ONCE
        t0 = time.time()
        W = D.W("final")
        Xtr, Xte = D.seq(W, D.train_pos), D.seq(W, D.test_pos)
        D.drop_W("final", keep=True)
        sw = None
        if method == "smote":
            Xs, ys, b, a, sk = smote_seq(Xtr, ytr, D.binary_cols)
        else:
            Xs, ys, b, a, sk = Xtr, ytr, np.bincount(ytr, minlength=C), np.bincount(ytr, minlength=C), []
        rec["final_smote"] = dict(b=cnt_str(b), a=cnt_str(a), skipped=sk, hold_test=cnt_str(np.bincount(D.y[label][D.test_pos], minlength=C)))
        m, info = fit_hybrid(self.reg[name], spec, chosen, Xs, ys, None, None, method, 0, SEED, fixed_epochs=epochs_final, lr_scale=lr_scale)
        pte, ptr_final = predict(m, Xte), predict(m, Xtr)
        m.save_weights(self.mdir / f"{label}__{name}.weights.h5")          # sub-classed models: weights are saved, the architecture is rebuilt from the spec + frozen config
        np.save(self.sp("test_proba", label, f"{name}.npy"), pte)
        if self.hook_final:
            self.hook_final(m, Xs, ys, Xtr, Xte, "real", name, label, method)
        rec["final"] = dict(epochs=epochs_final, secs=time.time() - t0, params=info["params"], train_ba=bal_f1(ytr, ptr_final)[0], train_f1=bal_f1(ytr, ptr_final)[1])
        del m
        keras.backend.clear_session()
        # 6. shuffled-label control: same recipe, training labels permuted
        yshuf = np.random.RandomState(SEED).permutation(ytr)
        if method == "smote":
            Xs2, ys2, *_ = smote_seq(Xtr, yshuf, D.binary_cols)
        else:
            Xs2, ys2 = Xtr, yshuf
        m, _ = fit_hybrid(self.reg[name], spec, chosen, Xs2, ys2, None, None, method, 0, SEED, fixed_epochs=epochs_final, lr_scale=lr_scale)
        np.save(self.sp("shuf_proba", label, f"{name}.npy"), predict(m, Xte))
        if self.hook_final:
            self.hook_final(m, Xs2, ys2, Xtr, Xte, "shuf", name, label, method)
        del m, Xs2, Xs, Xtr, Xte
        keras.backend.clear_session()
        gc.collect()
        rec["secs_total"] = time.time() - t_model0
        json.dump(rec, open(rp, "w"), indent=1, default=float)
        yte = D.y[label][D.test_pos]
        log(f"=== done {name} {label} in {rec['secs_total']/60:.1f} min | CV val BA {rec['cv_before']['val_ba']:.3f} | test BA {bal_f1(yte, pte)[0]:.3f} (reported, not used for any choice)")
        return rec


def tune_threshold(y, oof):
    """out-of-fold only. binary: threshold on P(class 1); multi-class: per-class multipliers by coordinate ascent. Chosen to maximise balanced accuracy."""
    C = oof.shape[1]
    if C == 2:
        best = (-1, 0.5, 0)
        for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
            pred = (oof[:, 1] >= t).astype(int)
            ba = balanced_accuracy_score(y, pred)
            if ba > best[0] + 1e-9:
                best = (ba, float(t), float((pred[y == 0] == 0).mean()))
        return dict(type="threshold", threshold=best[1], oof_balanced_accuracy=best[0], oof_normal_recall=best[2],
                    oof_balanced_accuracy_default_argmax=float(balanced_accuracy_score(y, oof.argmax(1))))
    w = np.ones(C)
    grid = np.exp(np.linspace(-2, 2, 17))
    cur = balanced_accuracy_score(y, (oof * w).argmax(1))
    for _ in range(2):
        for c in range(C):
            for g in grid:
                w2 = w.copy()
                w2[c] = g
                s = balanced_accuracy_score(y, (oof * w2).argmax(1))
                if s > cur + 1e-9:
                    cur, w = s, w2
    return dict(type="class_multipliers", weights=[float(x) for x in w], oof_balanced_accuracy=float(cur), oof_balanced_accuracy_default_argmax=float(balanced_accuracy_score(y, oof.argmax(1))))


def apply_threshold(P, th):
    if th["type"] == "threshold":
        return (P[:, 1] >= th["threshold"]).astype(int)
    return (P * np.array(th["weights"])).argmax(1)


# ----------------------------------------------------------------------------------------------- metrics, bootstrap, assembling the csv files
def metrics_of(y, P, pred, names):
    C = len(names)
    d = dict(n=int(len(y)), accuracy=float(accuracy_score(y, pred)), balanced_accuracy=float(balanced_accuracy_score(y, pred)),
             macro_f1=float(f1_score(y, pred, average="macro", labels=np.unique(y), zero_division=0)))
    m = y == 0
    d["normal_recall"] = float((pred[m] == 0).mean()) if m.any() else float("nan")
    d["support_normal"] = int(m.sum())
    for c in range(C):
        mc = y == c
        d[f"recall_{names[c]}"] = float((pred[mc] == c).mean()) if mc.any() else float("nan")
        d[f"support_{names[c]}"] = int(mc.sum())
    try:
        if C == 2:
            d["roc_auc"], d["pr_auc"] = float(roc_auc_score(y, P[:, 1])), float(average_precision_score(y, P[:, 1]))
        else:
            Pn = P / np.clip(P.sum(1, keepdims=True), 1e-12, None)
            d["roc_auc"] = float(roc_auc_score(y, Pn, multi_class="ovr", average="macro", labels=list(range(C))))
            d["pr_auc"] = float(np.mean([average_precision_score(y == c, Pn[:, c]) for c in range(C) if (y == c).any()]))
    except Exception:
        d["roc_auc"] = d["pr_auc"] = float("nan")
    return d


def block_bootstrap(y, P, pred, blocks, n_boot=1000, seed=SEED):
    """95% intervals of BA, macro-F1, healthy recall and ROC-AUC; resampling whole blocks (neighbouring windows are not independent)"""
    ids, inv = np.unique(blocks, return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(ids))]
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("balanced_accuracy", "macro_f1", "normal_recall", "roc_auc")}
    C = P.shape[1]
    for _ in range(n_boot):
        pick = rng.integers(0, len(ids), len(ids))
        ix = np.concatenate([members[i] for i in pick])
        yb, pb = y[ix], pred[ix]
        out["balanced_accuracy"].append(balanced_accuracy_score(yb, pb))
        out["macro_f1"].append(f1_score(yb, pb, average="macro", labels=np.unique(yb), zero_division=0))
        mm = yb == 0
        out["normal_recall"].append((pb[mm] == 0).mean() if mm.any() else np.nan)
        try:
            out["roc_auc"].append(roc_auc_score(yb, P[ix][:, 1]) if C == 2 else roc_auc_score(yb, P[ix] / P[ix].sum(1, keepdims=True), multi_class="ovr", average="macro", labels=list(range(C))))
        except Exception:
            out["roc_auc"].append(np.nan)
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)), int(np.isfinite(v).sum())) for k, v in out.items()}


def run_id_rule(D, label_of_run, n_boot=1000):
    """P14-style check. A throw-away XGBoost predicts WHICH RUN a window comes from (trained on training windows, scored on test windows); the 'rule' then outputs that run's own label.
    Returns the rule's test predictions per label version and the run-identification accuracy."""
    from xgboost import XGBClassifier
    df = D.df
    runs = sorted(df.run_id.unique())
    rid = df.run_id.map({r: i for i, r in enumerate(runs)}).to_numpy(int)
    W = D.W("final")
    X = W[:, D.runid_cols]
    D.drop_W("final", keep=True)
    tr, te = D.train_pos, D.test_pos
    present = np.unique(rid[tr])
    remap = {r: i for i, r in enumerate(present)}
    ytr_r = np.array([remap[r] for r in rid[tr]])
    m = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=2, random_state=SEED, verbosity=0)
    m.fit(X[tr], ytr_r)
    pr = present[m.predict(X[te])]
    acc = float((pr == rid[te]).mean())
    out = dict(run_id_accuracy=acc, n_runs=len(runs), chance=1 / len(runs), rule={})
    for lab in LABELS:
        per_run = {r: int(np.bincount(D.y[lab][np.where(rid == r)[0]]).argmax()) for r in np.unique(rid)}     # a run's own (majority) label
        out["rule"][lab] = np.array([per_run[r] for r in pr])
    return out


def fold_seconds(state, label, name):
    """training seconds of every fold fit of a model (screening, cross-validation, correction), summed from the saved fold files (so a resumed run keeps its earlier time)"""
    tot = 0.0
    for p in Path(state).glob(f"fold__{label}__{name}__*.pkl"):
        r = load_pkl(p)
        tot += float(r.get("secs", 0.0)) if isinstance(r, dict) else 0.0
    return tot


def assemble(D, names, prefix, state_dir, runid=None, n_boot=1000):
    """build every csv from the saved per-model records and saved test probabilities"""
    state = Path(state_dir)
    tm, ci, cvr, smo, thr, tt, ovf, shuf, scr, rid_rows, lead = [], [], [], [], {}, [], [], [], [], [], []
    blocks = D.group_of(D.test_pos)
    for label in LABELS:
        names_c = D.class_names[label]
        yte = D.y[label][D.test_pos]
        thr[label] = {}
        for name in names:
            rp = state / f"result__{label}__{name}.json"
            if not rp.exists():
                lead.append(dict(model=name, label_version=label, status="not completed"))
                continue
            r = json.load(open(rp))
            P = np.load(state / f"test_proba__{label}__{name}.npy")
            th = r["thresholds"]
            thr[label][name] = th
            for variant, pred in (("official", P.argmax(1)), ("tuned_threshold", apply_threshold(P, th))):
                m = metrics_of(yte, P, pred, names_c)
                tm.append(dict(model=name, label_version=label, scope="ALL", variant=variant, **m))
                if variant == "official":
                    b = block_bootstrap(yte, P, pred, blocks, n_boot)
                    for k, (lo, hi, nv) in b.items():
                        ci.append(dict(model=name, label_version=label, scope="ALL", metric=k, point_estimate=m[k], ci95_low=lo, ci95_high=hi, n_resamples=n_boot, n_valid_resamples=nv, n_blocks=len(np.unique(blocks))))
                    lead.append(dict(model=name, label_version=label, status="did not learn" if r["did_not_learn"] else "done", test_bal_acc=m["balanced_accuracy"],
                                     test_bal_acc_ci_low=b["balanced_accuracy"][0], test_bal_acc_ci_high=b["balanced_accuracy"][1], test_macro_f1=m["macro_f1"], test_normal_recall=m["normal_recall"],
                                     test_roc_auc=m["roc_auc"], cv_val_bal_acc=r["cv_before"]["val_ba"] if r["chosen_tag"] == "cv" else r["cv_after"]["val_ba"],
                                     train_secs=(fold_seconds(state, label, name) + r["final"]["secs"]) or r["secs_total"],
                                     folds=r["n_folds"], hidden=r["chosen_cfg"]["hidden"], method=r["chosen_method"], stage=r["chosen_tag"]))
            for row in r["cv_rows"]:
                cvr.append(dict(model=name, label_version=label, **row))
                smo.append(dict(model=name, label_version=label, stage=row["stage"], fold=row["fold"], before=row["smote_b"], after=row["smote_a"], held_out_fold=row["hold"]))
            smo.append(dict(model=name, label_version=label, stage="final", fold="final(80% train)", before=r["final_smote"]["b"], after=r["final_smote"]["a"], held_out_fold=r["final_smote"]["hold_test"] + " (test, untouched)"))
            tt.append(dict(model=name, label_version=label, cv_total_seconds=sum(x["secs"] for x in r["cv_rows"] if x["stage"] == r["chosen_tag"]), final_model_seconds=r["final"]["secs"],
                           final_epochs=r["final"]["epochs"], total_seconds_incl_screening_and_controls=(fold_seconds(state, label, name) + r["final"]["secs"]) or r["secs_total"], params=r["final"]["params"]))
            for stage, key in (("before_correction", "cv_before"), ("after_correction", "cv_after")):
                if key in r:
                    d = r[key]
                    row = dict(model=name, label_version=label, stage=stage, cv_train_macro_f1=d["train_f1"], cv_val_macro_f1=d["val_f1"], cv_gap_macro_f1=d["gap"], cv_val_bal_acc=d["val_ba"], diagnosis=d["diagnosis"])
                    if (stage == "before_correction" and r["chosen_tag"] == "cv") or (stage == "after_correction" and r["chosen_tag"] == "corr"):
                        mo = next(x for x in tm if x["model"] == name and x["label_version"] == label and x["variant"] == "official")
                        row.update(final_train_macro_f1=r["final"]["train_f1"], test_macro_f1=mo["macro_f1"], train_test_gap_macro_f1=r["final"]["train_f1"] - mo["macro_f1"], adopted=True)
                    else:
                        row["adopted"] = bool(d.get("adopted", False)) if stage == "after_correction" else False
                    ovf.append(row)
            Ps = np.load(state / f"shuf_proba__{label}__{name}.npy")
            ms = metrics_of(yte, Ps, Ps.argmax(1), names_c)
            shuf.append(dict(model=name, label_version=label, shuffled_balanced_accuracy=ms["balanced_accuracy"], shuffled_macro_f1=ms["macro_f1"], shuffled_normal_recall=ms["normal_recall"],
                             shuffled_roc_auc=ms["roc_auc"], real_balanced_accuracy=next(x for x in tm if x["model"] == name and x["label_version"] == label and x["variant"] == "official")["balanced_accuracy"],
                             verdict="PASS (shuffled labels score near chance)" if (ms["balanced_accuracy"] < 0.65 if len(names_c) == 2 else ms["balanced_accuracy"] < 0.45) else "WARN (shuffled labels score above chance)"))
            for s in r["screening"]:
                scr.append(dict(model=name, label_version=label, **s))
            if runid is not None:
                rule = runid["rule"][label]
                pm = P.argmax(1)
                ok_m, ok_r = pm == yte, rule == yte
                rid_rows.append(dict(kind="hybrid_vs_rule", model=name, label_version=label, agreement=float((pm == rule).mean()), model_accuracy=float(ok_m.mean()), rule_accuracy=float(ok_r.mean()),
                                     rule_balanced_accuracy=float(balanced_accuracy_score(yte, rule)), model_balanced_accuracy=float(balanced_accuracy_score(yte, pm)),
                                     share_of_correct_also_correct_by_rule=float((ok_m & ok_r).sum() / max(1, ok_m.sum()))))
    if runid is not None:
        rid_rows.insert(0, dict(kind="run_identification", model="(throw-away XGBoost)", accuracy=runid["run_id_accuracy"], chance=runid["chance"], n_runs=runid["n_runs"]))
    P_ = lambda n: REP / f"{prefix}{n}"
    pd.DataFrame(tm).to_csv(P_("test_metrics.csv"), index=False, float_format="%.6f")
    pd.DataFrame(ci).to_csv(P_("test_bootstrap_ci.csv"), index=False, float_format="%.6f")
    pd.DataFrame(cvr).to_csv(P_("cv_results.csv"), index=False, float_format="%.6f")
    pd.DataFrame(smo).to_csv(P_("smote_counts.csv"), index=False)
    json.dump(thr, open(P_("thresholds.json"), "w"), indent=1)
    pd.DataFrame(tt).to_csv(P_("training_time.csv"), index=False, float_format="%.2f")
    pd.DataFrame(ovf).to_csv(P_("overfit.csv"), index=False, float_format="%.6f")
    pd.DataFrame(shuf).to_csv(P_("shuffled_control.csv"), index=False, float_format="%.6f")
    pd.DataFrame(scr).to_csv(P_("screening.csv"), index=False, float_format="%.6f")
    pd.DataFrame(rid_rows).to_csv(P_("run_identification.csv"), index=False, float_format="%.6f")
    pd.DataFrame(lead).sort_values(["label_version", "test_bal_acc"], ascending=[True, False]).to_csv(P_("leaderboard.csv"), index=False, float_format="%.4f")
    return pd.DataFrame(lead)


def imbalance_report(D, prefix):
    """class-weight and ADASYN dataset-size comparison on the 80% training part (counts only, never used to train)"""
    rows = []
    W = D.W("final")
    Xtr = D.seq(W, D.train_pos).reshape(len(D.train_pos), -1)
    D.drop_W("final", keep=True)
    for label in LABELS:
        y = D.y[label][D.train_pos]
        cnt = np.bincount(y)
        cw = len(y) / (len(cnt) * np.maximum(cnt, 1))
        _, _, b, a, sk = smote_safe(Xtr, y)
        try:
            strat = {c: int(cnt.max()) for c in range(len(cnt)) if cnt[c] >= 2 and cnt[c] < cnt.max()}
            k = max(1, min(5, min(cnt[c] for c in strat) - 1))
            _, ya = ADASYN(sampling_strategy=strat, n_neighbors=k, random_state=SEED).fit_resample(Xtr, y)
            ada = cnt_str(np.bincount(ya, minlength=len(cnt)))
        except Exception as e:
            ada = f"ADASYN not possible: {str(e)[:60]}"
        rows.append(dict(label_version=label, original=cnt_str(cnt), smote_after=cnt_str(a), adasyn_after=ada, class_weights=" / ".join(f"{x:.2f}" for x in cw),
                         smote_total=int(a.sum()), original_total=int(cnt.sum()), skipped_classes=str(sk), note="ADASYN is a size comparison only; it is never used to train"))
    pd.DataFrame(rows).to_csv(REP / f"{prefix}imbalance.csv", index=False)
    return rows
