"""LO2: the other model families, same protocol as the hybrids (hybrid_engine.py), in priority order
  (b) classical machine-learning algorithms: Logistic Regression, Random Forest, ExtraTrees, XGBoost, LightGBM
  (c) Optuna-tuned Random Forest / XGBoost / LightGBM (>= 40 trials each, decided by cross-validation on the training part only)
  (d) deep-learning models CNN+BiGRU+Attention, CNN+BiLSTM+Attention, Fusion, Transformer and their XGBoost-head versions   (models_src/hybrid/dl_models.py)
  (e) stacking, then soft voting (built from the saved out-of-fold / test probabilities of (b))
Everything is fitted on LO2's own training part. Results -> report_outputs/lo2_full_*.csv, state -> outputs/hybrid_state/lo2/ (shared result format with the hybrids).

python scripts/lo2_full.py [--families b,c,d,e] [--until HH:MM]
"""
import argparse
import datetime as dt
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import hybrid_engine as E                      # noqa: E402
from lo2_data import LO2Data                   # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier  # noqa: E402
from xgboost import XGBClassifier              # noqa: E402
from lightgbm import LGBMClassifier            # noqa: E402

REP, ROOT = E.REP, E.ROOT
PREFIX = "lo2_full_"
STATE = ROOT / "outputs" / "hybrid_state" / "lo2_full"
SEED = 42
TAB = ["LR", "RF", "ET", "XGB", "LGBM"]
TUNED = ["RF_tuned", "XGB_tuned", "LGBM_tuned"]
DEEP = ["CNN_BiGRU_Att", "CNN_BiLSTM_Att", "Fusion", "Transformer"]
DEEP_HEAD = [d + "_XGBhead" for d in DEEP]
OPTUNA_TRIALS = 40

DEFAULTS = {"LR": dict(C=1.0), "RF": dict(n_estimators=100, max_features="sqrt"), "ET": dict(n_estimators=100, max_features="sqrt"),
            "XGB": dict(n_estimators=100, max_depth=6, learning_rate=0.1), "LGBM": dict(n_estimators=100, num_leaves=31, learning_rate=0.1, min_child_samples=5)}
CORRECTED = {"LR": dict(C=0.1), "RF": dict(n_estimators=100, max_depth=6, min_samples_leaf=3, max_features="sqrt"), "ET": dict(n_estimators=100, max_depth=6, min_samples_leaf=3, max_features="sqrt"),
             "XGB": dict(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.7, colsample_bytree=0.6, reg_lambda=5.0, min_child_weight=3),
             "LGBM": dict(n_estimators=100, num_leaves=8, learning_rate=0.1, min_child_samples=10, reg_lambda=5.0, subsample=0.7, subsample_freq=1, colsample_bytree=0.6)}


def base_of(name):
    return name.replace("_tuned", "")


def make(name, params, cw, C):
    b = base_of(name)
    p = dict(params)
    if b == "LR":
        return LogisticRegression(max_iter=3000, class_weight="balanced" if cw else None, **p)
    if b in ("RF", "ET"):
        cls = RandomForestClassifier if b == "RF" else ExtraTreesClassifier
        return cls(n_jobs=2, random_state=SEED, class_weight="balanced" if cw else None, **p)
    if b == "XGB":
        return XGBClassifier(tree_method="hist", n_jobs=2, random_state=SEED, verbosity=0, **p)
    return LGBMClassifier(n_jobs=2, random_state=SEED, verbose=-1, class_weight="balanced" if cw else None, **p)


def fit_tab(name, params, method, X, y, binary_cols):
    C = int(y.max()) + 1
    cw = method == "cw"
    sw = None
    if method == "smote":
        X, y, b, a, sk = E.smote_safe(X, y)
        X = np.asarray(X)
        X[:, binary_cols] = np.clip(np.round(X[:, binary_cols]), 0, 1)
    else:
        b = a = np.bincount(y, minlength=C)
        sk = []
    m = make(name, params, cw, C)
    if base_of(name) == "XGB" and cw:
        sw = E.balanced_sample_weight(y)
    if not np.array_equal(np.unique(y), np.arange(C)):                       # a class missing from this training part: map labels so XGBoost accepts them
        m.classes_map = np.unique(y)
    m.fit(X, y, sample_weight=sw) if sw is not None else m.fit(X, y)
    return m, b, a, sk


def proba(m, X, C):
    P = m.predict_proba(X).astype(np.float32)
    if P.shape[1] != C:                                                       # a class absent from the training part gets probability 0
        full = np.zeros((len(X), C), np.float32)
        full[:, m.classes_] = P
        P = full
    return P


class TabRunner:
    def __init__(self, D, deadline=None):
        self.D, self.deadline = D, deadline
        STATE.mkdir(parents=True, exist_ok=True)

    def sp(self, *a):
        return STATE / ("__".join(str(x) for x in a))

    def cv(self, name, label, params, method, tag):
        D = self.D
        C = len(D.class_names[label])
        path = self.sp("cv", label, name, tag, method + ".pkl")
        r = E.load_pkl(path)
        if r is not None:
            return r
        folds = []
        for k in range(1, 6):
            t0 = time.time()
            tr, va = D.folds[k - 1]
            W = D.W(k)
            Xtr, ytr, Xva, yva = W[tr], D.y[label][tr], W[va], D.y[label][va]
            D.drop_W(k)
            m, b, a, sk = fit_tab(name, params, method, Xtr, ytr, D.binary_cols)
            folds.append(dict(k=k, train_rows=tr, val_rows=va, train_proba=proba(m, Xtr, C), val_proba=proba(m, Xva, C), b=E.cnt_str(b), a=E.cnt_str(a), hold=E.cnt_str(np.bincount(yva, minlength=C)),
                              skipped=sk, epochs=0, best_epoch=0, secs=time.time() - t0))
        E.save_pkl(path, folds)
        return folds

    def run(self, name, label, params=None, tuned_info=None):
        rp = self.sp("result", label, f"{name}.json")
        if Path(rp).exists():
            return json.load(open(rp))
        if self.deadline and time.time() > self.deadline:
            E.log(f"time box reached: {name} {label} not started")
            return None
        D = self.D
        C = len(D.class_names[label])
        t_model0 = time.time()
        E.wait_memory()
        E.log(f"=== lo2 {label} {name}")
        params = params or DEFAULTS[base_of(name)]
        R = E.Runner.__new__(E.Runner)
        R.D = D
        rec = dict(dataset="lo2", label=label, model=name, seed=SEED, n_train=len(D.train_pos), n_test=len(D.test_pos), n_folds=5, notes=[], did_not_learn=False, family="tabular")
        if tuned_info:
            rec["tuning"] = tuned_info
        # class weights versus SMOTE, decided on the cross-validated validation score
        scr, runs = [], {}
        for method in ("smote", "cw"):
            f = self.cv(name, label, params, method, "orig")
            s = E.Runner.cv_summary(R, label, f)
            runs[method] = (f, s)
            scr.append(dict(hidden=0, method=method, val_ba=float(s.val_ba.mean()), val_f1=float(s.val_f1.mean()), epochs=0, loss_first=0.0, loss_last=0.0, secs=float(s.secs.sum())))
        method = max(("smote", "cw"), key=lambda m_: (round(runs[m_][1].val_ba.mean(), 4), m_ == "smote"))
        folds, s0 = runs[method]
        gap0 = float(s0.train_f1.mean() - s0.val_f1.mean())
        rec["screening"] = scr
        rec["cv_before"] = dict(cfg=params, method=method, val_ba=float(s0.val_ba.mean()), val_f1=float(s0.val_f1.mean()), train_f1=float(s0.train_f1.mean()), gap=gap0,
                                diagnosis="overfit (train-validation macro-F1 gap > 0.10)" if gap0 > 0.10 else "ok")
        chosen, chosen_folds, chosen_tag = params, folds, "cv"
        if gap0 > 0.10 and "_tuned" not in name:
            cp = CORRECTED[base_of(name)]
            f2 = self.cv(name, label, cp, method, "corr")
            s1 = E.Runner.cv_summary(R, label, f2)
            gap1 = float(s1.train_f1.mean() - s1.val_f1.mean())
            adopt = float(s1.val_ba.mean()) >= float(s0.val_ba.mean()) - 0.01
            rec["cv_after"] = dict(cfg=cp, method=method, val_ba=float(s1.val_ba.mean()), val_f1=float(s1.val_f1.mean()), train_f1=float(s1.train_f1.mean()), gap=gap1, adopted=bool(adopt),
                                   diagnosis="overfit (train-validation macro-F1 gap > 0.10)" if gap1 > 0.10 else "ok")
            if adopt:
                chosen, chosen_folds, chosen_tag = cp, f2, "corr"
        rec["chosen_tag"], rec["chosen_cfg"], rec["chosen_method"] = chosen_tag, dict(hidden=0, params=chosen), method
        sc = E.Runner.cv_summary(R, label, chosen_folds)
        sc.insert(0, "stage", chosen_tag)
        rec["cv_rows"] = sc.to_dict("records")
        if chosen_tag == "corr":
            rec["cv_rows"] += [dict(stage="cv", **r) for r in s0.to_dict("records")]
        oof = E.Runner.oof(R, label, chosen_folds)
        np.save(self.sp("oof", label, f"{name}.npy"), oof)
        ytr = D.y[label][D.train_pos]
        rec["thresholds"] = E.tune_threshold(ytr, oof)
        frozen = dict(dataset="lo2", label=label, model=name, seed=SEED, params=chosen, method=method, thresholds=rec["thresholds"], chosen_stage=chosen_tag, frozen_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                      note="written before the test windows were scored; every choice was made on training folds only")
        (REP / f"{PREFIX}frozen_{label}__{name}.json").write_text(json.dumps(frozen, indent=1, default=float))
        t0 = time.time()
        W = D.W("final")
        Xtr, Xte = W[D.train_pos], W[D.test_pos]
        m, b, a, sk = fit_tab(name, chosen, method, Xtr, ytr, D.binary_cols)
        pte = proba(m, Xte, C)
        ptr = proba(m, Xtr, C)
        np.save(self.sp("test_proba", label, f"{name}.npy"), pte)
        rec["final_smote"] = dict(b=E.cnt_str(b), a=E.cnt_str(a), skipped=sk, hold_test=E.cnt_str(np.bincount(D.y[label][D.test_pos], minlength=C)))
        rec["final"] = dict(epochs=0, secs=time.time() - t0, params=0, train_ba=E.bal_f1(ytr, ptr)[0], train_f1=E.bal_f1(ytr, ptr)[1])
        yshuf = np.random.RandomState(SEED).permutation(ytr)                  # shuffled-label control
        m2, *_ = fit_tab(name, chosen, method, Xtr, yshuf, D.binary_cols)
        np.save(self.sp("shuf_proba", label, f"{name}.npy"), proba(m2, Xte, C))
        del m, m2
        gc.collect()
        rec["secs_total"] = time.time() - t_model0
        json.dump(rec, open(rp, "w"), indent=1, default=float)
        E.log(f"=== done {name} {label} in {rec['secs_total']:.0f} s | CV val BA {rec['cv_before']['val_ba']:.3f} | method {method} | stage {chosen_tag}")
        return rec

    # ---- (c) Optuna: the objective is the mean validation balanced accuracy of 5-fold CV on the training part (SMOTE inside every training fold)
    def tune(self, name, label, trials=OPTUNA_TRIALS):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        D = self.D
        C = len(D.class_names[label])
        b = base_of(name)
        p = self.sp("optuna", label, name + ".json")
        if Path(p).exists():
            return json.load(open(p))

        def space(t):
            if b == "RF":
                return dict(n_estimators=t.suggest_int("n_estimators", 50, 200), max_depth=t.suggest_int("max_depth", 3, 14), min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 8),
                            max_features=t.suggest_categorical("max_features", ["sqrt", "log2", 0.5]))
            if b == "XGB":
                return dict(n_estimators=t.suggest_int("n_estimators", 50, 200), max_depth=t.suggest_int("max_depth", 2, 8), learning_rate=t.suggest_float("learning_rate", 0.03, 0.3, log=True),
                            subsample=t.suggest_float("subsample", 0.5, 1.0), colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0), reg_lambda=t.suggest_float("reg_lambda", 0.1, 10, log=True))
            return dict(n_estimators=t.suggest_int("n_estimators", 50, 200), num_leaves=t.suggest_int("num_leaves", 4, 40), learning_rate=t.suggest_float("learning_rate", 0.03, 0.3, log=True),
                        min_child_samples=t.suggest_int("min_child_samples", 3, 30), reg_lambda=t.suggest_float("reg_lambda", 0.1, 10, log=True), subsample=t.suggest_float("subsample", 0.5, 1.0),
                        subsample_freq=1, colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0))
        prepared = []
        for k in range(1, 6):
            tr, va = D.folds[k - 1]
            W = D.W(k)
            Xs, ys, *_ = E.smote_safe(W[tr], D.y[label][tr])
            Xs = np.asarray(Xs)
            Xs[:, D.binary_cols] = np.clip(np.round(Xs[:, D.binary_cols]), 0, 1)
            prepared.append((Xs, ys, W[va], D.y[label][va]))
            D.drop_W(k)

        def obj(t):
            prm = space(t)
            sc = []
            for Xs, ys, Xv, yv in prepared:
                m = make(name, prm, False, C)
                m.fit(Xs, ys)
                sc.append(E.bal_f1(yv, proba(m, Xv, C))[0])
            return float(np.mean(sc))
        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        st.optimize(obj, n_trials=trials)
        trials_df = st.trials_dataframe()[["number", "value", "state"]].assign(model=name, label_version=label)
        trials_df.to_csv(REP / f"{PREFIX}optuna_trials_{label}__{name}.csv", index=False)
        out = dict(best_params=st.best_params, best_value=st.best_value, n_trials=len(st.trials))
        if b == "LGBM":
            out["best_params"]["subsample_freq"] = 1
        json.dump(out, open(p, "w"))
        return out


def derived(D, label, kind, base_names):
    """(e) stacking: logistic regression on the out-of-fold probabilities of the base models; soft voting: mean of their probabilities. Test probabilities come from the final base models."""
    C = len(D.class_names[label])
    ytr = D.y[label][D.train_pos]
    oofs = [np.load(STATE / f"oof__{label}__{n}.npy") for n in base_names]
    tests = [np.load(STATE / f"test_proba__{label}__{n}.npy") for n in base_names]
    if kind == "SoftVote":
        return np.mean(oofs, 0), np.mean(tests, 0)
    Z, Zt = np.hstack(oofs), np.hstack(tests)
    fold = D.fold5
    oof = np.zeros((len(ytr), C), np.float32)
    for k in range(1, 6):
        m = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Z[fold != k], ytr[fold != k])
        oof[fold == k] = proba(m, Z[fold == k], C)
    m = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Z, ytr)
    return oof, proba(m, Zt, C)


def record_derived(D, label, kind, base_names):
    rp = STATE / f"result__{label}__{kind}.json"
    if rp.exists():
        return
    C = len(D.class_names[label])
    ytr = D.y[label][D.train_pos]
    oof, pte = derived(D, label, kind, base_names)
    np.save(STATE / f"oof__{label}__{kind}.npy", oof)
    np.save(STATE / f"test_proba__{label}__{kind}.npy", pte)
    th = E.tune_threshold(ytr, oof)
    rows = []
    for k in range(1, 6):
        va = D.fold5 == k
        vb, vf = E.bal_f1(ytr[va], oof[va])
        rows.append(dict(stage="cv", fold=k, val_ba=vb, val_f1=vf, val_normal_recall=E.normal_recall(ytr[va], oof[va]), train_ba=float("nan"), train_f1=float("nan"), epochs=0, best_epoch=0, secs=0.0,
                         smote_b="(built from the base models' out-of-fold probabilities)", smote_a="-", hold=E.cnt_str(np.bincount(ytr[va], minlength=C))))
    # shuffled-label control for the combination: the base models' shuffled test probabilities, combined the same way
    sh = [np.load(STATE / f"shuf_proba__{label}__{n}.npy") for n in base_names]
    np.save(STATE / f"shuf_proba__{label}__{kind}.npy", np.mean(sh, 0) if kind == "SoftVote" else LogisticRegression(max_iter=3000).fit(np.hstack([np.load(STATE / f"oof__{label}__{n}.npy") for n in base_names]), np.random.RandomState(SEED).permutation(ytr)).predict_proba(np.hstack(sh)))
    vals = pd.DataFrame(rows)
    rec = dict(dataset="lo2", label=label, model=kind, seed=SEED, n_train=len(D.train_pos), n_test=len(D.test_pos), n_folds=5, notes=[f"built from {base_names}"], did_not_learn=False,
               screening=[], cv_before=dict(cfg={}, method="combination", val_ba=float(vals.val_ba.mean()), val_f1=float(vals.val_f1.mean()), train_f1=float("nan"), gap=0.0, diagnosis="ok (out-of-fold)"),
               chosen_tag="cv", chosen_cfg=dict(hidden=0), chosen_method="combination", cv_rows=vals.to_dict("records"), thresholds=th,
               final_smote=dict(b="-", a="-", skipped=[], hold_test=E.cnt_str(np.bincount(D.y[label][D.test_pos], minlength=C))), final=dict(epochs=0, secs=0.0, params=0, train_ba=float("nan"), train_f1=float("nan")),
               secs_total=0.0)
    json.dump(rec, open(rp, "w"), indent=1, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="b,c,d,e")
    ap.add_argument("--until", default="")
    ap.add_argument("--trials", type=int, default=OPTUNA_TRIALS)
    a = ap.parse_args()
    D = LO2Data()
    dead = None
    if a.until:
        h, m = map(int, a.until.split(":"))
        now = dt.datetime.now()
        t = now.replace(hour=h, minute=m, second=0, microsecond=0)
        dead = (t + dt.timedelta(days=1) if t < now - dt.timedelta(hours=1) else t).timestamp()
    fam = a.families.split(",")
    R = TabRunner(D, dead)
    if "b" in fam:
        for n in TAB:
            for label in E.LABELS:
                R.run(n, label)
    if "c" in fam:
        for n in TUNED:
            for label in E.LABELS:
                if dead and time.time() > dead:
                    break
                info = R.tune(n, label, a.trials)
                E.log(f"optuna {n} {label}: {info['n_trials']} trials, best CV balanced accuracy {info['best_value']:.3f}")
                R.run(n, label, params=info["best_params"], tuned_info=info)
    if "d" in fam:
        import lo2_deep
        lo2_deep.run_all(D, dead)
    if "e" in fam:
        for label in E.LABELS:
            if all((STATE / f"result__{label}__{n}.json").exists() for n in TAB):
                record_derived(D, label, "Stacking", TAB)
                record_derived(D, label, "SoftVote", TAB)
    assemble_all(D)


def assemble_all(D):
    """leaderboard of all LO2 families: the hybrids (hybrid_lo2_*), then (b) to (e); families with no result are listed as 'not completed because of the time limit'"""
    all_names = TAB + TUNED + DEEP + DEEP_HEAD + ["Stacking", "SoftVote"]
    lead = E.assemble(D, [n for n in all_names], PREFIX, STATE, runid=None, n_boot=1000)
    hyb = pd.read_csv(REP / "hybrid_lo2_leaderboard.csv") if (REP / "hybrid_lo2_leaderboard.csv").exists() else pd.DataFrame()
    fam = {**{n: "(b) classical machine-learning algorithm" for n in TAB}, **{n: "(c) Optuna-tuned classical algorithm" for n in TUNED}, **{n: "(d) deep-learning model" for n in DEEP},
           **{n: "(d) deep-learning model + XGBoost head" for n in DEEP_HEAD}, "Stacking": "(e) stacking", "SoftVote": "(e) soft voting"}
    lead["family"] = lead.model.map(fam)
    if not hyb.empty:
        hyb = hyb.assign(family="(a) hybrid deep-learning model")
        lead = pd.concat([hyb, lead], ignore_index=True)
    lead["reliability"] = "unreliable: very small sample (5 runs, 172 test windows, 13 healthy)"
    order = {"(a)": 0, "(b)": 1, "(c)": 2, "(d)": 3, "(e)": 4}
    lead["_o"] = lead.family.str[:3].map(order)
    lead = lead.sort_values(["label_version", "_o"]).drop(columns="_o")
    lead.to_csv(REP / "lo2_full_leaderboard.csv", index=False, float_format="%.4f")
    nc = lead[lead.status == "not completed"][["model", "label_version", "family"]].assign(reason="not completed because of the time limit")
    nc.to_csv(REP / "lo2_full_not_completed.csv", index=False)
    E.log("LO2 leaderboard written; not completed: " + (", ".join(sorted(set(nc.model))) or "none"))
    return lead


if __name__ == "__main__":
    main()
