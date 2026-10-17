"""Full retrain pipeline (step 2). Resumable: every job writes outputs/v2_state/*.pkl and is skipped on restart.

python scripts/full_pipeline.py [--stages 0,1,...,10]   (default: all, in order; label versions binary -> level inside every stage)
Stages: 0 timing/env | 1 baselines | 2 Optuna | 3 deep | 4 heads+ensembles | 5 thresholds | 6 diagnosis | 7 correction |
        8 final fit | 9 test evaluation (once) | 10 timing/architecture/cv tables, OOF parquet, leaderboard
"""
import argparse, json, platform, subprocess
import numpy as np, pandas as pd

from v2_data import *
import v2_data
import v2_tab as TB
import v2_deep as DP
import keras
import models as M

ap = argparse.ArgumentParser()
ap.add_argument("--stages", default="0,1,2,3,4,5,6,7,8,9,10")
ap.add_argument("--labels", default="label_binary,label_level")
ARGS = ap.parse_args()
STAGES = [int(s) for s in ARGS.stages.split(",")]
LABS = ARGS.labels.split(",")
D = None
N_TRIALS = 2 if FAST else 40
OVERFIT_GAP, UNDERFIT_F1, ADOPT_TOL = 0.10, 0.60, 0.01


# ------------------------------------------------------------------------------------------------ registry
def reg_path(label):
    return V2 / f"registry__{label}.json"


def reg(label):
    p = reg_path(label)
    return json.loads(p.read_text()) if p.exists() else {}


def reg_save(label, r):
    reg_path(label).write_text(json.dumps(r, indent=1, default=str))


def reg_set(label, name, **kw):
    r = reg(label)
    r.setdefault(name, {}).update(kw)
    reg_save(label, r)


def res_of(label, name):
    r = reg(label)[name]
    return load_pkl(V2 / r["cv_state"])


def cvmean(label, name, col, scope="ALL"):
    return mean_val(D, label, res_of(label, name), col)


# ------------------------------------------------------------------------------------------------ stage 0
def stage0():
    log("=" * 70); log("STAGE 0: environment, GPU check, 3-epoch timing of every deep model")
    out = []
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except Exception as e:
        smi = f"nvidia-smi failed: {e}"
    log("nvidia-smi:", smi)
    out.append(f"nvidia-smi: {smi}")
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    log("TensorFlow GPUs:", gpus or "none (TensorFlow >= 2.11 has no native-Windows GPU support) -> CPU")
    out.append(f"TensorFlow GPUs: {gpus or 'none -> CPU'}")
    try:
        import warnings as _w
        from xgboost import XGBClassifier
        x = np.random.rand(500, 20); y = (x[:, 0] > .5).astype(int)
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            XGBClassifier(n_estimators=5, device="cuda", tree_method="hist").fit(x, y)
        fell_back = any("No visible GPU" in str(c.message) or "changed from GPU to CPU" in str(c.message) for c in caught)
        msg = "XGBoost: no usable CUDA device (driver 452.56 / MX350 too old for the CUDA-12 build) -> CPU" if fell_back else "XGBoost CUDA works"
    except Exception as e:
        msg = f"XGBoost CUDA unavailable ({type(e).__name__}) -> CPU"
    log(msg + "; LightGBM pip build is CPU-only")
    out.append(msg + "; LightGBM pip build is CPU-only")
    secs = {}
    label = "label_binary"
    tr, va = D.folds[0]
    Xtr, ytr, Xva, yva, Xs, ys = DP.prepare_train(D, label, 1, "smote")
    log(f"timing data: SMOTE train {Xs.shape}, val {Xva.shape}")
    for name in DEEP:
        keras.backend.clear_session()
        model = M.build(name, Xs.shape[2], 2, 64, 0.3, D.groups)
        model.fit(Xs[:256], ys[:256], epochs=1, batch_size=256, verbose=0)           # warm-up / graph build
        t0 = time.time()
        model.fit(Xs, ys, epochs=3, batch_size=256, verbose=0)
        secs[name] = (time.time() - t0) / 3
        log(f"  {name}: {secs[name]:.2f} s/epoch ({model.count_params()} params)")
    avg_epochs = 60                                   # assumption for the estimate (policy: up to 150, stop after 12 without improvement)
    per_label = {n: s * avg_epochs * (10 + 1) for n, s in secs.items()}
    base_h = 2 * sum(per_label.values()) / 3600
    corr_h = base_h * 0.5                                          # upper bound: all deep models need a (smaller, shorter) correction retrain
    log(f"ESTIMATE deep models: {base_h:.1f} h for 10 folds + final x 2 label versions at {avg_epochs} epochs/run on average; "
        f"+ up to {corr_h:.1f} h if every deep model needs the correction retrain (upper bound)")
    log("ESTIMATE trees/Optuna/baselines: roughly 2-4 h (not timed yet)")
    log(f"ESTIMATE TOTAL ~ {base_h + 3:.1f} - {base_h + corr_h + 4:.1f} hours. Continuing without waiting (policy: no shortening of epochs/folds/SMOTE).")
    out += [f"{n}: {s:.2f} s/epoch" for n, s in secs.items()]
    out.append(f"estimated deep hours (60 epochs avg, 11 runs x 2 labels): {base_h:.1f}; with corrections up to +{corr_h:.1f}; trees/Optuna ~2-4 h")
    (REP / "time_estimate.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    lines = [f"Python {platform.python_version()}"]
    import sklearn, imblearn, xgboost, lightgbm, optuna
    lines += [f"TensorFlow {tf.__version__}", f"Keras {keras.__version__}", f"scikit-learn {sklearn.__version__}", f"imbalanced-learn {imblearn.__version__}",
              f"xgboost {xgboost.__version__}", f"lightgbm {lightgbm.__version__}", f"optuna {optuna.__version__}", f"pandas {pd.__version__}", f"numpy {np.__version__}",
              "device: CPU (see time_estimate.txt for the GPU check)", f"seed {SEED}",
              f"epoch policy: max {MAX_EPOCHS}, EarlyStopping(val_loss, patience {ES_PATIENCE}, restore best), ReduceLROnPlateau(0.5, patience {LR_PATIENCE}, min_lr 1e-5)"]
    (REP / "environment_v2.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------------------------------ stage 1
def stage1():
    log("=" * 70); log("STAGE 1: baselines, run-aware 10-fold CV (folds from fold_assignment.csv), variants cw / smote")
    jobs = [(l, n, v) for l in LABS for n in TB.TAB for v in ("cw", "smote")]
    pg = Progress("1 baselines", len(jobs))
    for l, n, v in jobs:
        p = state_path("s1", l, n, v)
        if p.exists():
            pg.skip(f"{l} {n} {v}"); continue
        t0 = time.time()
        res = TB.tab_cv(D, l, n, TB.DEFAULTS[n], v)
        save_pkl(p, res)
        pg.finished(f"{l} {n} {v}: val BA {mean_val(D, l, res):.3f}, macro-F1 {mean_val(D, l, res, 'macro_f1'):.3f}", time.time() - t0)
    rows, sel = [], {}
    for l in LABS:
        sel[l] = {"models": {}}
        score = {"cw": [], "smote": []}
        for n in TB.TAB:
            best = None
            for v in ("cw", "smote"):
                res = load_pkl(state_path("s1", l, n, v))
                fm = fold_metrics(D, l, res)
                ba = fm.balanced_accuracy.mean()
                score[v].append(ba)
                fm.insert(0, "variant", v); fm.insert(0, "label_version", l); fm.insert(0, "model", n)
                rows.append(fm)
                if best is None or ba > best[1]:
                    best = (v, ba)
            sel[l]["models"][n] = dict(variant=best[0], val_balanced_accuracy=best[1])
            reg_set(l, n, kind="tab", base=n, variant=best[0], params=TB.DEFAULTS[n], tag="orig", cv_state=state_path("s1", l, n, best[0]).name)
        m = {v: float(np.mean(s)) for v, s in score.items()}
        sel[l]["deep_method"] = "smote" if m["smote"] >= m["cw"] else "cw"
        sel[l]["mean_val_ba_by_variant"] = m
        log(f"[{l}] mean validation BA over the 5 baselines: class-weight {m['cw']:.4f} vs SMOTE {m['smote']:.4f} -> deep models use '{sel[l]['deep_method']}'")
        for n, s in sel[l]["models"].items():
            log(f"    {n}: better variant {s['variant']} (val BA {s['val_balanced_accuracy']:.4f})")
    allf = pd.concat(rows, ignore_index=True)
    num = ["accuracy", "balanced_accuracy", "macro_f1", "normal_recall", "roc_auc", "train_balanced_accuracy", "train_macro_f1"]
    gb = allf.groupby(["model", "label_version", "variant"], sort=False)[num]
    mean_rows = gb.mean().reset_index().assign(fold="mean")
    std_rows = gb.std().reset_index().assign(fold="std")
    allf["fold"] = allf.fold.astype(object)
    out = pd.concat([allf, mean_rows, std_rows], ignore_index=True)
    out["selected_variant"] = [sel[r.label_version]["models"][r.model]["variant"] == r.variant for r in out.itertuples()]
    out.to_csv(REP / "baseline_cv.csv", index=False)
    (V2 / "s1_selection.json").write_text(json.dumps(sel, indent=1))
    log("saved report_outputs/baseline_cv.csv")


# ------------------------------------------------------------------------------------------------ stage 2
def space(trial, name):
    if name in ("RF", "ET"):
        return dict(n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50), max_depth=trial.suggest_int("max_depth", 3, 40),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20), max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]))
    if name == "XGB":
        return dict(n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50), max_depth=trial.suggest_int("max_depth", 2, 10),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True), subsample=trial.suggest_float("subsample", 0.5, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0), min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True))
    if name == "LGBM":
        return dict(n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50), num_leaves=trial.suggest_int("num_leaves", 8, 128),
                    max_depth=trial.suggest_int("max_depth", 3, 12), learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    min_child_samples=trial.suggest_int("min_child_samples", 5, 60), subsample=trial.suggest_float("subsample", 0.5, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0), reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True))
    raise ValueError(name)


def stage2():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    log("=" * 70); log(f"STAGE 2: Optuna ({N_TRIALS} trials/model) for the top-3 tree models, objective = mean validation balanced accuracy, same folds")
    best_all = json.loads((REP / "best_params.json").read_text()) if (REP / "best_params.json").exists() else {}
    trial_tabs = []
    pg = Progress("2 optuna", len(LABS) * 3)
    for l in LABS:
        ranks = sorted(TB.TREES, key=lambda n: -cvmean(l, n, "balanced_accuracy"))
        top3 = ranks[:3]
        log(f"[{l}] tree ranking by validation BA: " + ", ".join(f"{n} {cvmean(l, n, 'balanced_accuracy'):.4f}" for n in ranks) + f" -> tuning {top3}")
        for n in top3:
            tname = n + "_tuned"
            sp = state_path("s2", l, tname)
            if sp.exists():
                pg.skip(f"{l} {tname}")
                continue
            t0 = time.time()
            variant = reg(l)[n]["variant"]
            TB.clear_cache()
            db = V2 / f"optuna_{l}_{n}.db"
            study = optuna.create_study(study_name=f"{l}_{n}", direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                        storage=f"sqlite:///{db.as_posix()}", load_if_exists=True)

            def objective(trial):
                params = space(trial, n)
                res = TB.tab_cv(D, l, n, params, variant, use_cache=True)
                fm = fold_metrics(D, l, res)
                trial.set_user_attr("val_macro_f1", float(fm.macro_f1.mean()))
                trial.set_user_attr("train_balanced_accuracy", float(fm.train_balanced_accuracy.mean()))
                return float(fm.balanced_accuracy.mean())

            def cb(st, tr):
                log(f"    [{l}] {n} trial {tr.number + 1}/{N_TRIALS}: val BA {tr.value:.4f} (best {st.best_value:.4f}) {tr.params}")
            todo = N_TRIALS - len([t for t in study.trials if t.state.name == "COMPLETE"])
            if todo > 0:
                study.optimize(objective, n_trials=todo, callbacks=[cb])
            best = study.best_trial
            params = dict(best.params)
            best_all.setdefault(l, {})[n] = dict(variant=variant, best_value=best.value, params=params)
            (REP / "best_params.json").write_text(json.dumps(best_all, indent=1))
            tdf = study.trials_dataframe().assign(label_version=l, model=n, variant=variant)
            tdf.to_csv(V2 / f"trials_{l}_{n}.csv", index=False)
            # 10-fold CV with the best params (same folds) -> out-of-fold probabilities of the tuned model
            res = TB.tab_cv(D, l, n, params, variant, use_cache=True)
            res["name"] = tname
            save_pkl(sp, res)
            reg_set(l, tname, kind="tab", base=n, variant=variant, params=params, tag="orig", cv_state=sp.name)
            pg.finished(f"{l} {tname}: val BA {mean_val(D, l, res):.4f} (default {cvmean(l, n, 'balanced_accuracy'):.4f})", time.time() - t0)
    tabs = [pd.read_csv(p) for p in sorted(V2.glob("trials_*.csv"))]
    if tabs:
        pd.concat(tabs, ignore_index=True).to_csv(REP / "optuna_trials.csv", index=False)
    log("saved report_outputs/best_params.json, optuna_trials.csv")


# ------------------------------------------------------------------------------------------------ stage 3
def stage3():
    log("=" * 70); log(f"STAGE 3: deep models, 10-fold, max {MAX_EPOCHS} epochs, EarlyStopping patience {ES_PATIENCE}, ReduceLROnPlateau patience {LR_PATIENCE}")
    sel = json.loads((V2 / "s1_selection.json").read_text())
    pg = Progress("3 deep", len(LABS) * len(DEEP))
    for l in LABS:
        method = sel[l]["deep_method"]
        for n in DEEP:
            sp = state_path("s3", l, n, "orig")
            if sp.exists():
                pg.skip(f"{l} {n}")
                reg_set(l, n, kind="deep", base=n, variant=method, tag="orig", cv_state=sp.name)
                continue
            t0 = time.time()
            res = DP.deep_cv(D, l, n, "orig", method)
            save_pkl(sp, res)
            reg_set(l, n, kind="deep", base=n, variant=method, tag="orig", cv_state=sp.name)
            ep = [f["epochs"] for f in res["folds"].values()]
            log(f"[{l}] {n}: epochs per fold {ep}; folds hitting {MAX_EPOCHS}: {res['n_not_converged']} {'(' + NOT_CONVERGED + ')' if res['n_not_converged'] else ''}")
            pg.finished(f"{l} {n}: val BA {mean_val(D, l, res):.4f}", time.time() - t0)
    rows = []
    for l in LABS:
        for n in DEEP:
            r = load_pkl(state_path("s3", l, n, "orig"))
            if r:
                for k, f in r["folds"].items():
                    rows.append(dict(model=n, label_version=l, tag="orig", fold=k, epochs_run=f["epochs"], best_epoch=f["best_epoch"], max_epochs=f["cap"],
                                     status=NOT_CONVERGED if f["hit_cap"] else "converged (early stopping)", seconds=f["secs"]))
    pd.DataFrame(rows).to_csv(REP / "deep_epochs_per_fold.csv", index=False)


# ------------------------------------------------------------------------------------------------ helpers shared by stages 4-10
from sklearn.linear_model import LogisticRegression
import joblib


def idx(rows):
    return [D.tr_index[p] for p in rows]


def make_ens(l, name, kind, members, meta_C=1.0):
    """Stacking (shallow logistic regression on the out-of-fold probabilities of the members) or SoftVote (mean of the members' fold-k models).
    Both only ever see out-of-fold predictions of their members."""
    t0 = time.time()
    rs = [res_of(l, m) for m in members]
    folds = {}
    if kind == "stack":
        Z = np.hstack([oof_of(D, r) for r in rs])
        y = D.ytr[l]
        for k in range(1, 11):
            tr, va = D.folds[k - 1]
            itr, iva = idx(tr), idx(va)
            meta = LogisticRegression(C=meta_C, max_iter=3000, class_weight="balanced").fit(Z[itr], y[itr])
            folds[k] = dict(train_rows=tr, val_rows=va, train_proba=meta.predict_proba(Z[itr]).astype(np.float32),
                            val_proba=meta.predict_proba(Z[iva]).astype(np.float32), secs=0.0)
    else:
        for k in range(1, 11):
            folds[k] = dict(train_rows=rs[0]["folds"][k]["train_rows"], val_rows=rs[0]["folds"][k]["val_rows"],
                            train_proba=np.mean([r["folds"][k]["train_proba"] for r in rs], axis=0).astype(np.float32),
                            val_proba=np.mean([r["folds"][k]["val_proba"] for r in rs], axis=0).astype(np.float32), secs=0.0)
    return dict(name=name, label=l, members=list(members), meta_C=meta_C, kind=kind, folds=folds, secs=time.time() - t0)


def ens_members(l):
    """stacking members: the tuned tree models + the 3 best deep models by out-of-fold balanced accuracy; soft voting: the 4 deep models"""
    r = reg(l)
    trees = [n for n in r if n.endswith("_tuned")]
    deep = sorted(DP.DEEP, key=lambda n: -cvmean(l, n, "balanced_accuracy"))[:3]
    return trees + deep, list(DP.DEEP)


def cv_scope_table(l, name, res):
    rows = []
    for k, f in sorted(res["folds"].items()):
        yv, yt = D.ytr[l][idx(f["val_rows"])], D.ytr[l][idx(f["train_rows"])]
        tr = scope_scores(yt, f["train_proba"], D.sys[f["train_rows"]])
        va = scope_scores(yv, f["val_proba"], D.sys[f["val_rows"]])
        for sc in ("ALL", "SN", "TT"):
            rows.append(dict(model=name, label_version=l, scope=sc, fold=k, train_macro_f1=tr[sc][0], train_bal_acc=tr[sc][1],
                             val_macro_f1=va[sc][0], val_bal_acc=va[sc][1]))
    return pd.DataFrame(rows)


def diagnose_row(l, name, res, stage):
    out = []
    t = cv_scope_table(l, name, res)
    for sc, g in t.groupby("scope"):
        gap_f1, gap_ba = g.train_macro_f1.mean() - g.val_macro_f1.mean(), g.train_bal_acc.mean() - g.val_bal_acc.mean()
        flags = []
        if g.val_macro_f1.mean() < UNDERFIT_F1:
            flags.append("underfit (validation macro-F1 < 0.60)")
        if gap_f1 > OVERFIT_GAP:
            flags.append("overfit (train-validation macro-F1 gap > 0.10)")
        out.append(dict(model=name, label_version=l, scope=sc, stage=stage, cv_train_macro_f1=g.train_macro_f1.mean(),
                        cv_val_macro_f1=g.val_macro_f1.mean(), cv_gap_macro_f1=gap_f1, cv_train_bal_acc=g.train_bal_acc.mean(),
                        cv_val_bal_acc=g.val_bal_acc.mean(), cv_gap_bal_acc=gap_ba, diagnosis=" + ".join(flags) if flags else "ok"))
    return out


def is_overfit(l, name, res):
    return any(r["scope"] == "ALL" and "overfit" in r["diagnosis"] for r in diagnose_row(l, name, res, "x"))


# ------------------------------------------------------------------------------------------------ stage 4
def stage4():
    log("=" * 70); log("STAGE 4: XGBoost heads on the deep embeddings, stacking (shallow meta-learner on OOF probabilities), soft voting")
    pg = Progress("4 heads+ensembles", len(LABS) * (len(DP.DEEP) + 2))
    for l in LABS:
        method = reg(l)[DP.DEEP[0]]["variant"]
        for n in DP.DEEP:
            hn = n + "_XGBhead"
            sp = state_path("s4h", l, hn, "orig")
            if sp.exists():
                pg.skip(f"{l} {hn}")
            else:
                t0 = time.time()
                res = DP.head_cv(D, l, n, "orig", method, DP.HEAD_ORIG)
                save_pkl(sp, res)
                pg.finished(f"{l} {hn}: val BA {mean_val(D, l, res):.4f} (own output layer {cvmean(l, n, 'balanced_accuracy'):.4f})", time.time() - t0)
            reg_set(l, hn, kind="head", base=n, variant=method, tag="orig", head_params=DP.HEAD_ORIG, cv_state=sp.name)
        stack_m, vote_m = ens_members(l)
        for name, kind, members in (("Stacking", "stack", stack_m), ("SoftVote", "vote", vote_m)):
            sp = state_path("s4e", l, name, "orig")
            if sp.exists():
                pg.skip(f"{l} {name}")
            else:
                t0 = time.time()
                res = make_ens(l, name, kind, members)
                save_pkl(sp, res)
                pg.finished(f"{l} {name} {members}: val BA {mean_val(D, l, res):.4f}", time.time() - t0)
            reg_set(l, name, kind="ens", ens_kind=kind, members=members, meta_C=1.0, tag="orig", cv_state=sp.name)


# ------------------------------------------------------------------------------------------------ stage 5
def tune_decision(y, P, C):
    """decision rule from out-of-fold predictions only. binary: threshold t on P(Anomaly); five-class: per-class weights."""
    def nr_ba(pred):
        from sklearn.metrics import balanced_accuracy_score
        nr = float(((pred == 0) & (y == 0)).sum() / max(1, (y == 0).sum()))
        return nr, float(balanced_accuracy_score(y, pred))
    if C == 2:
        best, any_ok = None, False
        for t in np.round(np.arange(0.01, 1.0, 0.01), 2):
            nr, ba = nr_ba((P[:, 1] >= t).astype(int))
            ok = nr >= 0.70
            any_ok |= ok
            key = (ok, ba if ok else ba, -abs(t - 0.5))
            if best is None or key > best[0]:
                best = (key, float(t), nr, ba)
        return dict(type="threshold", threshold=best[1], oof_normal_recall=best[2], oof_balanced_accuracy=best[3], normal_recall_ge_0_70_reachable=bool(any_ok))
    w = np.ones(C)
    grid = np.geomspace(0.25, 8.0, 25)

    def score(w):
        nr, ba = nr_ba((P * w).argmax(1))
        return ba - 2 * max(0.0, 0.70 - nr), nr, ba
    cur = score(w)
    for _ in range(3):
        for c in range(C):
            for g in grid:
                w2 = w.copy(); w2[c] = g
                s = score(w2)
                if s[0] > cur[0] + 1e-12:
                    w, cur = w2, s
    return dict(type="class_weights", weights=[float(x) for x in w / w.max()], oof_normal_recall=cur[1], oof_balanced_accuracy=cur[2],
                normal_recall_ge_0_70_reachable=bool(cur[1] >= 0.70))


def apply_decision(P, spec):
    """probabilities whose argmax equals the tuned decision (binary: monotone re-centring of P(Anomaly) at the threshold)"""
    P = P / np.clip(P.sum(1, keepdims=True), 1e-12, None)
    if spec["type"] == "threshold":
        t, p = spec["threshold"], P[:, 1]
        q = p * (1 - t) / (p * (1 - t) + (1 - p) * t + 1e-12)
        return np.stack([1 - q, q], 1)
    Q = P * np.array(spec["weights"])
    return Q / Q.sum(1, keepdims=True)


def thresholds(tag):
    out = {}
    for l in LABS:
        out[l] = {}
        for n in reg(l):
            res = res_of(l, n)
            oof = oof_of(D, res)
            y = D.ytr[l]
            spec = tune_decision(y, oof, oof.shape[1])
            from sklearn.metrics import balanced_accuracy_score
            spec["oof_balanced_accuracy_default_argmax"] = float(balanced_accuracy_score(y, oof.argmax(1)))
            out[l][n] = spec
    (REP / "thresholds.json").write_text(json.dumps(out, indent=1))
    log(f"saved thresholds.json ({tag}); computed on out-of-fold predictions only")
    for l in LABS:
        nok = sum(1 for s in out[l].values() if not s["normal_recall_ge_0_70_reachable"])
        log(f"  [{l}] Normal recall >= 0.70 reachable for {len(out[l]) - nok}/{len(out[l])} models (unreachable -> best balanced accuracy, flagged)")
    return out


def stage5():
    log("=" * 70); log("STAGE 5: decision thresholds / class weights from out-of-fold predictions")
    thresholds("before correction")


# ------------------------------------------------------------------------------------------------ stage 6
def stage6():
    log("=" * 70); log("STAGE 6: over/under-fitting diagnosis (gap > 0.10 overfit; validation macro-F1 < 0.60 underfit)")
    rows = []
    for l in LABS:
        for n in reg(l):
            rows += diagnose_row(l, n, res_of(l, n), "before_correction")
    t = pd.DataFrame(rows)
    t.to_csv(REP / "underfit_overfit.csv", index=False)
    a = t[t.scope == "ALL"]
    for l in LABS:
        g = a[a.label_version == l]
        log(f"[{l}] overfit: {g[g.diagnosis.str.contains('overfit')].model.tolist()} | underfit: {g[g.diagnosis.str.contains('underfit')].model.tolist()}")


# ------------------------------------------------------------------------------------------------ stage 7
def regularize_tab(name, p):
    p = dict(p)
    if name == "LR":
        p["C"] = p.get("C", 1.0) / 10
    elif name in ("RF", "ET"):
        d = p.get("max_depth") or 20
        p.update(max_depth=max(3, int(d * 0.5)), min_samples_leaf=max(5, 2 * p.get("min_samples_leaf", 1)), max_samples=0.7)
    elif name == "XGB":
        p.update(max_depth=max(2, int(p.get("max_depth", 6) * 0.5)), min_child_weight=max(5, 3 * p.get("min_child_weight", 1)), subsample=min(0.7, p.get("subsample", 1.0)),
                 colsample_bytree=min(0.6, p.get("colsample_bytree", 1.0)), reg_lambda=max(5.0, 5 * p.get("reg_lambda", 1.0)), n_estimators=min(600, 2 * p.get("n_estimators", 300)))
    elif name == "LGBM":
        md = p.get("max_depth", -1)
        p.update(num_leaves=max(4, p.get("num_leaves", 31) // 2), max_depth=min(5, md if md and md > 0 else 5), min_child_samples=max(30, 3 * p.get("min_child_samples", 20)),
                 subsample=min(0.7, p.get("subsample", 1.0)), colsample_bytree=min(0.6, p.get("colsample_bytree", 1.0)), reg_lambda=max(5.0, 5 * p.get("reg_lambda", 0.0)),
                 n_estimators=min(600, 2 * p.get("n_estimators", 300)))
    return p


def ba_f1(l, res):
    fm = fold_metrics(D, l, res)
    return dict(val_ba=fm.balanced_accuracy.mean(), val_f1=fm.macro_f1.mean(), train_f1=fm.train_macro_f1.mean(), train_ba=fm.train_balanced_accuracy.mean())


def ba_row(l, name, kind, desc, before, after, adopted):
    return dict(model=name, label_version=l, kind=kind, correction=desc,
                val_bal_acc_before=before["val_ba"], val_bal_acc_after=after["val_ba"], val_macro_f1_before=before["val_f1"], val_macro_f1_after=after["val_f1"],
                train_macro_f1_before=before["train_f1"], train_macro_f1_after=after["train_f1"],
                gap_macro_f1_before=before["train_f1"] - before["val_f1"], gap_macro_f1_after=after["train_f1"] - after["val_f1"],
                adopted=bool(adopted), rule=f"adopt if validation balanced accuracy after >= before - {ADOPT_TOL} AND the train-validation macro-F1 gap does not grow")


def stage7():
    log("=" * 70); log("STAGE 7: correction of every model diagnosed as overfit (stronger regularisation, validated on the same folds)")
    rows = []
    done_p = state_path("s7", "rows")
    prev = load_pkl(done_p) or {}
    for l in LABS:
        r = reg(l)
        flagged = [n for n in r if r[n]["kind"] in ("tab", "deep", "head", "ens") and is_overfit(l, n, res_of(l, n))]
        log(f"[{l}] diagnosed overfit before correction: {flagged}")
        TB.clear_cache()
        # --- tab + tuned + deep (base models)
        for n in [n for n in flagged if r[n]["kind"] in ("tab", "deep")]:
            key = f"{l}|{n}"
            if key in prev:
                rows.append(prev[key]); continue
            t0 = time.time()
            before_res = res_of(l, n)
            before = ba_f1(l, before_res)
            sp = state_path("s7", l, n, "corr")
            if r[n]["kind"] == "tab":
                params = regularize_tab(r[n]["base"], r[n]["params"])
                after_res = load_pkl(sp) or TB.tab_cv(D, l, r[n]["base"], params, r[n]["variant"], use_cache=True, early_stop=r[n]["base"] in ("XGB", "LGBM"))
                desc = f"shallower/more regularised trees, subsampling, min leaf/child weight" + (", early stopping" if r[n]["base"] in ("XGB", "LGBM") else "") + f": {params}"
            else:
                cap = max(20, int(round(0.6 * np.mean([f["epochs"] for f in before_res["folds"].values()]))))
                after_res = load_pkl(sp) or DP.deep_cv(D, l, n, "corr", r[n]["variant"], max_epochs=cap)
                desc = f"dropout 0.3->0.4, L2 1e-4, units 64->32, max epochs {cap} (60% of the epochs the original ran)"
            after_res["name"] = n
            save_pkl(sp, after_res)
            after = ba_f1(l, after_res)
            adopt = after["val_ba"] >= before["val_ba"] - ADOPT_TOL and (after["train_f1"] - after["val_f1"]) <= (before["train_f1"] - before["val_f1"])
            row = ba_row(l, n, r[n]["kind"], desc, before, after, adopt)
            rows.append(row); prev[key] = row; save_pkl(done_p, prev)
            if adopt:
                kw = dict(tag="corr", cv_state=sp.name)
                if r[n]["kind"] == "tab":
                    kw["params"] = params; kw["early_stop"] = r[n]["base"] in ("XGB", "LGBM")
                reg_set(l, n, **kw)
            log(f"    [{l}] {n}: val BA {before['val_ba']:.4f} -> {after['val_ba']:.4f}, gap {row['gap_macro_f1_before']:.3f} -> {row['gap_macro_f1_after']:.3f}, "
                f"{'ADOPTED' if adopt else 'not adopted'} ({time.time()-t0:.0f}s)")
        # --- heads follow their deep model; a head diagnosed overfit also gets the regularised XGB head
        r = reg(l)
        for n in DP.DEEP:
            hn = n + "_XGBhead"
            key = f"{l}|{hn}"
            if key in prev:
                rows.append(prev[key]); continue
            before_res = load_pkl(state_path("s4h", l, hn, "orig"))
            if not is_overfit(l, hn, before_res) and r[n]["tag"] == "orig":
                continue
            t0 = time.time()
            before = ba_f1(l, before_res)
            cands = []
            for htag, hp in (("orig", DP.HEAD_ORIG), ("corr", DP.HEAD_CORR)):
                if htag == "orig" and r[n]["tag"] == "orig":
                    continue
                sp = state_path("s7h", l, hn, r[n]["tag"], htag)
                res = load_pkl(sp) or DP.head_cv(D, l, n, r[n]["tag"], r[n]["variant"], hp)
                save_pkl(sp, res)
                cands.append((htag, hp, sp, res, ba_f1(l, res)))
            ok = [c for c in cands if c[4]["val_ba"] >= before["val_ba"] - ADOPT_TOL and (c[4]["train_f1"] - c[4]["val_f1"]) <= (before["train_f1"] - before["val_f1"])]
            pick = min(ok, key=lambda c: c[4]["train_f1"] - c[4]["val_f1"]) if ok else None
            if pick is None and r[n]["tag"] != "orig":
                pick = max(cands, key=lambda c: c[4]["val_ba"])        # the head must follow the adopted corrected deep model
            desc = "XGB head on the corrected deep embedding / regularised head (depth 3, 150 rounds, subsample 0.7, min_child_weight 5, lambda 5): " + ", ".join(
                f"{c[0]} val BA {c[4]['val_ba']:.4f} gap {c[4]['train_f1'] - c[4]['val_f1']:.3f}" for c in cands)
            row = ba_row(l, hn, "head", desc, before, (pick or cands[-1])[4], pick is not None)
            rows.append(row); prev[key] = row; save_pkl(done_p, prev)
            if pick:
                reg_set(l, hn, tag=r[n]["tag"], head_params=pick[1], cv_state=pick[2].name, head_tag=pick[0])
            log(f"    [{l}] {hn}: {desc} -> {'adopted ' + pick[0] if pick else 'kept original'} ({time.time()-t0:.0f}s)")
        # --- ensembles are rebuilt from the (possibly corrected) members; a stacking meta-learner diagnosed overfit also gets C=0.1
        stack_m, vote_m = ens_members(l)
        for name, kind, members in (("Stacking", "stack", stack_m), ("SoftVote", "vote", vote_m)):
            key = f"{l}|{name}"
            if key in prev:
                rows.append(prev[key]); continue
            before_res = load_pkl(state_path("s4e", l, name, "orig"))
            changed = any(reg(l)[m]["tag"] != "orig" for m in members) or members != reg(l)[name]["members"]
            ov = is_overfit(l, name, before_res)
            if not changed and not ov:
                continue
            t0 = time.time()
            before = ba_f1(l, before_res)
            cands = []
            for C_meta in ([1.0, 0.1] if kind == "stack" and ov else [1.0]):
                if C_meta == 1.0 and not changed:
                    continue
                res = make_ens(l, name, kind, members, C_meta)
                cands.append((C_meta, res, ba_f1(l, res)))
            if not cands:
                log(f"    [{l}] {name}: diagnosed overfit but nothing to rebuild (soft voting has no tunable setting; members unchanged) -> kept")
                continue
            ok = [c for c in cands if c[2]["val_ba"] >= before["val_ba"] - ADOPT_TOL and (c[2]["train_f1"] - c[2]["val_f1"]) <= (before["train_f1"] - before["val_f1"])]
            pick = min(ok, key=lambda c: c[2]["train_f1"] - c[2]["val_f1"]) if ok else None
            if pick:
                sp = state_path("s7e", l, name, "corr"); save_pkl(sp, pick[1])
                reg_set(l, name, members=members, meta_C=pick[0], tag="corr", cv_state=sp.name)
            desc = f"rebuilt from corrected members {members}; meta C candidates {[c[0] for c in cands]}"
            row = ba_row(l, name, "ens", desc, before, (pick or cands[-1])[2], pick is not None)
            rows.append(row); prev[key] = row; save_pkl(done_p, prev)
            log(f"    [{l}] {name}: val BA {before['val_ba']:.4f} -> {(pick or cands[-1])[2]['val_ba']:.4f}, {'ADOPTED' if pick else 'kept original'} ({time.time()-t0:.0f}s)")
    pd.DataFrame(rows).drop_duplicates(["model", "label_version"], keep="last").to_csv(REP / "before_after_correction.csv", index=False)
    log(f"saved before_after_correction.csv ({len(rows)} corrected models)")
    thresholds("after correction (final)")


# ------------------------------------------------------------------------------------------------ stage 8
def final_paths(l, n):
    return MODELS / f"{l}__{n}"


def stage8():
    log("=" * 70); log("STAGE 8: final fit of every model on the full 80% training part (corrected settings, SMOTE/weights on this part only)")
    Wf = D.W("final")
    Xtab = Wf[D.train_pos]
    Xseq = D.seq(Wf, D.train_pos)
    pg = Progress("8 final fit", sum(len(reg(l)) for l in LABS))
    for l in LABS:
        C = len(CLASS_NAMES[l])
        y = D.ytr[l]
        r = reg(l)
        for n, s in r.items():
            sp = state_path("s8", l, n)
            if sp.exists():
                pg.skip(f"{l} {n}"); continue
            t0 = time.time()
            cv = res_of(l, n)
            info = dict(name=n, label=l, kind=s["kind"], tag=s["tag"])
            if s["kind"] == "tab":
                m = TB.fit_model(s["base"], s["params"], s["variant"], Xtab, y, C, early_stop=bool(s.get("early_stop")))
                joblib.dump(m, str(final_paths(l, n)) + ".joblib")
                info.update(train_proba=m.predict_proba(Xtab).astype(np.float32), file=str(final_paths(l, n)) + ".joblib")
            elif s["kind"] == "deep":
                ep = max(5, int(round(np.mean([f["best_epoch"] for f in cv["folds"].values()]))))
                if s["variant"] == "smote":
                    Xs, ys = smote_seq(Xseq, y)
                else:
                    Xs, ys = Xseq, y
                model, inf = DP.train_net(n, Xs, ys, None, None, C, s["tag"], SEED + DP.DEEP.index(n), s["variant"], fixed_epochs=ep)
                path = str(final_paths(l, n)) + ".keras"
                model.save(path)
                info.update(train_proba=model.predict(Xseq, batch_size=512, verbose=0).astype(np.float32), file=path, epochs=ep, params=inf["params"],
                            layers=inf["layers"], hist=inf["hist"], units=DP.DEEP_CFG[s["tag"]]["units"])
            elif s["kind"] == "head":
                base = r[s["base"]]
                model = keras.models.load_model(str(final_paths(l, s["base"])) + ".keras")
                em = M.embed_model(model)
                if base["variant"] == "smote":
                    Xs, ys = smote_seq(Xseq, y)
                else:
                    Xs, ys = Xseq, y
                clf = DP.XGBClassifier(tree_method="hist", n_jobs=-1, random_state=SEED, verbosity=0, **s["head_params"])
                clf.fit(em.predict(Xs, batch_size=512, verbose=0), ys,
                        sample_weight=DP.compute_sample_weight("balanced", ys) if base["variant"] == "cw" else None)
                joblib.dump(clf, str(final_paths(l, n)) + ".joblib")
                info.update(train_proba=clf.predict_proba(em.predict(Xseq, batch_size=512, verbose=0)).astype(np.float32), file=str(final_paths(l, n)) + ".joblib")
            else:                                                                       # ensembles
                if s["ens_kind"] == "stack":
                    Z = np.hstack([oof_of(D, res_of(l, m)) for m in s["members"]])
                    meta = LogisticRegression(C=s["meta_C"], max_iter=3000, class_weight="balanced").fit(Z, y)
                    joblib.dump(meta, str(final_paths(l, n)) + ".joblib")
                    info.update(train_proba=meta.predict_proba(Z).astype(np.float32), file=str(final_paths(l, n)) + ".joblib")
                else:
                    info.update(train_proba=np.mean([load_pkl(state_path("s8", l, m))["train_proba"] for m in s["members"]], axis=0).astype(np.float32),
                                file="(average of the saved fold models, see models/folds)")
            info["secs"] = time.time() - t0
            save_pkl(sp, info)
            pg.finished(f"{l} {n}", info["secs"])


# ------------------------------------------------------------------------------------------------ stage 9
def predict_test(l, n, s, cache):
    Wf = D.W("final")
    Xtab, Xseq = Wf[D.test_pos], D.seq(Wf, D.test_pos)
    fin = load_pkl(state_path("s8", l, n))
    if s["kind"] == "tab":
        return joblib.load(fin["file"]).predict_proba(Xtab)
    if s["kind"] == "deep":
        return keras.models.load_model(fin["file"]).predict(Xseq, batch_size=512, verbose=0)
    if s["kind"] == "head":
        em = M.embed_model(keras.models.load_model(load_pkl(state_path("s8", l, s["base"]))["file"]))
        return joblib.load(fin["file"]).predict_proba(em.predict(Xseq, batch_size=512, verbose=0))
    if s["ens_kind"] == "stack":
        return joblib.load(fin["file"]).predict_proba(np.hstack([cache[m] for m in s["members"]]))
    ps = []
    for m in s["members"]:                                         # all fold models of the member architectures
        tag = reg(l)[m]["tag"]
        for k in range(1, 11):
            fm = keras.models.load_model(DP.fold_model_path(l, m, tag, k))
            ps.append(fm.predict(D.seq(D.W(k), D.test_pos), batch_size=512, verbose=0))
    return np.mean(ps, axis=0)


def metric_rows(l, n, P, variant, spec=None):
    names = CLASS_NAMES[l]
    rows = []
    Padj = apply_decision(P, spec) if spec else P
    for sc, m in (("ALL", np.ones(len(D.test_pos), bool)), ("SN", D.sys_te == 0), ("TT", D.sys_te == 1)):
        r = full_metrics(D.yte[l][m], Padj[m], names)
        if spec:                                                   # ranking metrics from the original probabilities
            o = full_metrics(D.yte[l][m], P[m], names)
            r["roc_auc"], r["pr_auc"], r["pr_auc_normal"] = o["roc_auc"], o["pr_auc"], o["pr_auc_normal"]
        rows.append(dict(model=n, label_version=l, scope=sc, variant=variant, **r))
    return rows


def stage9():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
    plt.rcParams["font.family"] = "serif"
    log("=" * 70); log("STAGE 9: ONE evaluation of every final model on the untouched 20% test set (test is read here for the first time)")
    thr = json.loads((REP / "thresholds.json").read_text())
    rows, cm_rows = [], []
    for l in LABS:
        names = CLASS_NAMES[l]
        r = reg(l)
        order = [n for n in r if r[n]["kind"] != "ens"] + [n for n in r if r[n]["kind"] == "ens"]
        cache = {}
        for n in order:
            sp = state_path("s9", l, n)
            if sp.exists():
                P = load_pkl(sp)["test_proba"]
            else:
                t0 = time.time()
                P = predict_test(l, n, r[n], cache).astype(np.float64)
                save_pkl(sp, dict(test_proba=P, secs=time.time() - t0))
            cache[n] = P
            rows += metric_rows(l, n, P, "official")
            rows += metric_rows(l, n, P, "tuned_threshold", thr[l][n])
            for sc, m in (("ALL", np.ones(len(D.test_pos), bool)), ("SN", D.sys_te == 0), ("TT", D.sys_te == 1)):
                for dec, pred in (("argmax", P.argmax(1)), ("tuned", apply_decision(P, thr[l][n]).argmax(1))):
                    cm = confusion_matrix(D.yte[l][m], pred[m], labels=range(len(names)))
                    pd.DataFrame(cm, index=[f"true_{x}" for x in names], columns=[f"pred_{x}" for x in names]).to_csv(
                        FIG / f"cm_{l}_{n}_{sc}_{dec}.csv")
                    if sc == "ALL" and dec == "argmax":
                        fig, ax = plt.subplots(figsize=(4.6, 4.0))
                        cn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
                        ax.imshow(cn, cmap="Greys", vmin=0, vmax=1)
                        for i in range(len(names)):
                            for j in range(len(names)):
                                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cn[i, j] > .5 else "black", fontsize=9)
                        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
                        ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
                        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{n} ({l[6:]}, test)", fontsize=10)
                        fig.tight_layout(); fig.savefig(FIG / f"cm_{l}_{n}.png", dpi=130); plt.close(fig)
        t = pd.DataFrame([x for x in rows if x["label_version"] == l and x["scope"] == "ALL" and x["variant"] == "official"]).set_index("model")
        top = t.balanced_accuracy.sort_values(ascending=False).index[:6]
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
        C = len(names)
        for n in top:
            P = cache[n] / cache[n].sum(1, keepdims=True)
            yy, pp = (D.yte[l], P[:, 1]) if C == 2 else (np.eye(C)[D.yte[l]].ravel(), P.ravel())
            fpr, tpr, _ = roc_curve(yy, pp); pr, rc, _ = precision_recall_curve(yy, pp)
            axs[0].plot(fpr, tpr, label=n); axs[1].plot(rc, pr, label=n)
        axs[0].plot([0, 1], [0, 1], "k:", lw=.8)
        sub = "Anomaly vs Normal" if C == 2 else "micro-average one-vs-rest"
        axs[0].set_title(f"ROC, {l[6:]} ({sub}); top 6 by balanced accuracy", fontsize=9); axs[0].set_xlabel("FPR"); axs[0].set_ylabel("TPR"); axs[0].legend(fontsize=7)
        axs[1].set_title(f"Precision-recall, {l[6:]}", fontsize=9); axs[1].set_xlabel("recall"); axs[1].set_ylabel("precision")
        fig.tight_layout(); fig.savefig(FIG / f"roc_pr_{l}.png", dpi=130); plt.close(fig)
    T = pd.DataFrame(rows)
    T = T[["model", "label_version", "scope", "variant"] + [c for c in T.columns if c not in ("model", "label_version", "scope", "variant")]]
    T.to_csv(REP / "test_metrics.csv", index=False)
    log(f"saved test_metrics.csv ({len(T)} rows: model x label x scope ALL/SN/TT x variant official / tuned_threshold) and figures")


# ------------------------------------------------------------------------------------------------ stage 10
def stage10():
    log("=" * 70); log("STAGE 10: cv_scores, underfit_overfit (final), training_time, model_architectures, oof_predictions.parquet, leaderboard")
    cvs, uo, arch, tim, oofs = [], [], [], [], []
    test = pd.read_csv(REP / "test_metrics.csv")
    for l in LABS:
        C = len(CLASS_NAMES[l])
        for n, s in reg(l).items():
            res = res_of(l, n)
            fin = load_pkl(state_path("s8", l, n))
            cvs.append(cv_scope_table(l, n, res))
            d = pd.DataFrame(diagnose_row(l, n, res, "final"))
            ytr = D.ytr[l]
            for i, rr in d.iterrows():
                m = {"ALL": np.ones(len(ytr), bool), "SN": D.sys_tr == 0, "TT": D.sys_tr == 1}[rr.scope]
                f1, ba = quick(ytr[m], fin["train_proba"][m])
                t = test[(test.model == n) & (test.label_version == l) & (test.scope == rr.scope) & (test.variant == "official")].iloc[0]
                for col, val in (("final_train_macro_f1", f1), ("test_macro_f1", t.macro_f1), ("train_test_gap_macro_f1", f1 - t.macro_f1),
                                 ("final_train_bal_acc", ba), ("test_bal_acc", t.balanced_accuracy), ("train_test_gap_bal_acc", ba - t.balanced_accuracy),
                                 ("final_last_train_loss", fin["hist"]["loss"][-1] if fin.get("hist") else np.nan)):
                    d.loc[i, col] = val
                tg = f1 - t.macro_f1
                extra = ("; test gap > 0.10 (overfit to train)" if tg > OVERFIT_GAP else "") + ("; test macro-F1 < 0.60 (underfit on test)" if t.macro_f1 < UNDERFIT_F1 else "")
                d.loc[i, "diagnosis"] = (rr.diagnosis + extra) if rr.diagnosis != "ok" or extra else "ok"
            uo.append(d)
            f = list(res["folds"].values())
            fsize = os.path.getsize(fin["file"]) / 1e6 if fin.get("file") and os.path.exists(str(fin["file"])) else np.nan
            if s["kind"] == "deep":
                note = f"{res['cfg']}, max {res['max_epochs']} epochs, method {s['variant']}"
                arch.append(dict(model=n, label_version=l, config=s["tag"], units=fin["units"], dropout=DP.DEEP_CFG[s["tag"]]["drop"], layers=fin["layers"],
                                 parameters=fin["params"], size_mb=round(fin["params"] * 4 / 1e6, 3), note=note))
            else:
                p = s.get("params") or s.get("head_params") or {}
                if s["kind"] == "tab":
                    note = f"{s['base']} {p}; imbalance: {s['variant']}"
                elif s["kind"] == "head":
                    note = f"XGBoost head on 64-d embedding of {s['base']} {p}"
                else:
                    note = f"{s['ens_kind']} over {s['members']}"
                arch.append(dict(model=n, label_version=l, config=s["tag"], units=np.nan, dropout=np.nan, layers=np.nan, parameters=np.nan, size_mb=round(fsize, 3) if fsize == fsize else np.nan, note=note))
            eps = [x.get("epochs", np.nan) for x in f]
            secs = [x["secs"] for x in f]
            tim.append(dict(model=n, label_version=l, cv_total_seconds_incl_cfg_selection=res.get("secs"), fold_seconds_mean=np.mean(secs),
                            epochs_mean=np.nanmean(eps) if not np.all(np.isnan(eps)) else np.nan,
                            seconds_per_epoch=(sum(secs) / np.nansum(eps)) if not np.all(np.isnan(eps)) and np.nansum(eps) else np.nan,
                            final_model_seconds=fin["secs"], final_model_epochs=fin.get("epochs"), xgb_head_seconds=res.get("secs") if s["kind"] == "head" else np.nan,
                            folds_not_converged=res.get("n_not_converged", 0), correction_tag=s["tag"]))
            oof = oof_of(D, res)
            o = pd.DataFrame({"window_id": D.df.window_id.to_numpy()[D.train_pos], "label_version": l, "model": n, "tag": s["tag"],
                              "fold": D.df.fold.to_numpy()[D.train_pos], "y_true": D.ytr[l]})
            for c in range(5):
                o[f"p{c}"] = oof[:, c] if c < C else np.nan
            oofs.append(o)
    cv = pd.concat(cvs, ignore_index=True)
    num = ["train_macro_f1", "train_bal_acc", "val_macro_f1", "val_bal_acc"]
    summ = [dict(model=m, label_version=l, scope=sc, fold=st, **getattr(g[num], st)().to_dict())
            for (m, l, sc), g in cv.groupby(["model", "label_version", "scope"], sort=False) for st in ("mean", "std")]
    cv = pd.concat([cv.astype({"fold": object}), pd.DataFrame(summ)], ignore_index=True)
    cv.to_csv(REP / "cv_scores.csv", index=False)
    old = pd.read_csv(REP / "underfit_overfit.csv")
    pd.concat([old[old.stage == "before_correction"], pd.concat(uo, ignore_index=True)], ignore_index=True).to_csv(REP / "underfit_overfit.csv", index=False)
    pd.DataFrame(arch).to_csv(REP / "model_architectures.csv", index=False)
    pd.DataFrame(tim).to_csv(REP / "training_time.csv", index=False)
    pd.concat(oofs, ignore_index=True).to_parquet(REP / "oof_predictions.parquet", index=False)
    # ---- leaderboard
    for l in LABS:
        t = test[(test.label_version == l) & (test.scope == "ALL") & (test.variant == "official")].set_index("model")
        v = cv[(cv.label_version == l) & (cv.scope == "ALL") & (cv.fold == "mean")].set_index("model")
        lb = pd.DataFrame({"test_bal_acc": t.balanced_accuracy, "test_macro_f1": t.macro_f1, "test_normal_recall": t.normal_recall, "test_normal_precision": t.precision_Normal,
                           "test_roc_auc": t.roc_auc, "test_pr_auc": t.pr_auc, "cv_val_bal_acc": v.val_bal_acc, "cv_val_macro_f1": v.val_macro_f1}).sort_values("test_bal_acc", ascending=False)
        lb.insert(0, "label_version", l)
        lb.round(4).to_csv(REP / f"leaderboard_{l}.csv")
        tt = test[(test.label_version == l) & (test.scope == "ALL") & (test.variant == "tuned_threshold")].set_index("model")
        log(f"\nLEADERBOARD {l} (final models, untouched test set, sorted by balanced accuracy)\n" + lb.drop(columns="label_version").round(3).to_string())
        log(f"  with tuned decision rule (thresholds from OOF): best test BA {tt.balanced_accuracy.max():.3f} ({tt.balanced_accuracy.idxmax()}), Normal recall of that model {tt.loc[tt.balanced_accuracy.idxmax(), 'normal_recall']:.3f}")
    files = sorted(p.name for p in REP.glob("*") if p.is_file()) + sorted("figures/" + p.name for p in FIG.glob("*.png"))[:5] + ["figures/ (cm_*.png/csv, roc_pr_*.png)"]
    log("\nFILES WRITTEN to report_outputs: " + ", ".join(files))



if __name__ == "__main__":
    log("=" * 70); log("full_pipeline start", time.ctime(), "stages", STAGES, "labels", LABS)
    log("NOTE: sequences are indexed by their last window, so validation sequences overlap training windows within the same block; folds are "
        "group-aware (block_id) but CV scores can still be optimistic; the TEST score is the main result; each class comes from a few scenario runs.")
    log("NOTE: spec conflict resolved in favour of the EPOCH POLICY: deep early stopping patience 12 (not 5), ReduceLROnPlateau patience 4.")
    D = DataV2()
    DP.DATA = D
    log(f"data: {len(D.df)} windows, train {len(D.train_pos)}, test {len(D.test_pos)} (test untouched until stage 9); model inputs {D.n_feat}; "
        f"groups api/log/metric {D.groups}; context slots padded {D.padded_slots}")
    g = globals()
    for s in STAGES:
        t0 = time.time()
        g[f"stage{s}"]() if f"stage{s}" in g else log(f"stage {s}: not implemented yet")
        log(f"STAGE {s} finished in {(time.time()-t0)/60:.1f} min")
