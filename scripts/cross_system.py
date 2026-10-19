"""Cross-system generalisation test (the key experiment of v2).

Train on ONE system's training split, test on the OTHER system's test windows (SN -> TT and TT -> SN); in-system rows (SN -> SN, TT -> TT) are the reference.
Three feature sets (scripts/universal_features.py): (a) shared API + log, (b) generic metric aggregates, (c) both. Each set twice: baseline-relative ("rel") and
the same columns raw. Models: the saved recipe of XGBoost (as in the final fit: SMOTE on the training part only, early stopping) and the two best other refittable
models of the earlier runs by test balanced accuracy (LGBM_tuned and XGB_tuned; the stacking ensemble is a meta-model of the others and cannot be refitted on a new
feature set). Nothing saved is touched; every model here is a throw-away. Shuffled-label control: the training labels are permuted (2 shuffles), the scoring
labels are real. 95% CIs: 1000 block-bootstrap resamples of the target test windows (a block = 10 consecutive windows of one run).

python scripts/cross_system.py [--boot 1000] [--shuffles 2] [--no-cv]        -> report_outputs/cross_system_results.csv
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
OUT = REP / "cross_system_results.csv"
SEED = 20260102
MODELS = ["XGB", "LGBM_tuned", "XGB_tuned"]
DIRECTIONS = [("SN", "TT"), ("TT", "SN"), ("SN", "SN"), ("TT", "TT")]
SET_NAMES = {"a_shared_api_log": "(a) shared API + log", "b_generic_metrics": "(b) generic metric aggregates", "c_both": "(c) both"}


def load():
    import universal_features as U
    d = pd.read_parquet(U.OUT)
    shared = pd.read_csv(REP / "shared_features.csv")
    return d, U.feature_sets(shared)


def matrices(d, cols, rep, train_mask):
    """feature matrix for the chosen representation. rel: NaN -> 0 (no deviation from the baseline). raw: NaN -> median of the TRAINING rows (train_mask) of the same system"""
    if rep == "rel":
        X = d[[f"rel__{c}" for c in cols]].to_numpy(float)
        return np.nan_to_num(X, nan=0.0)
    X = d[cols].to_numpy(float).copy()
    for s in ("SN", "TT"):
        ms = (d.system == s).to_numpy()
        med = np.nanmedian(X[ms & train_mask], axis=0) if (ms & train_mask).any() else np.zeros(X.shape[1])
        med = np.where(np.isnan(med), 0.0, med)
        sub = X[ms]
        nan = np.isnan(sub)
        sub[nan] = np.take(med, np.where(nan)[1])
        X[ms] = sub
    return X


class Boot:
    def __init__(self, blocks, y, n_boot, seed=SEED):
        self.y = y
        ids, inv = np.unique(blocks, return_inverse=True)
        self.rows = [np.where(inv == b)[0] for b in range(len(ids))]
        rng = np.random.default_rng(seed)
        self.draw = rng.integers(0, len(ids), (n_boot, len(ids)))
        self.inv, self.nb = inv, len(ids)
        self.counts = np.stack([np.bincount(r, minlength=len(ids)) for r in self.draw]).astype(float)

    def metrics(self, P1, pred):
        from sklearn.metrics import roc_auc_score
        y = self.y
        cm = np.zeros((self.nb, 4))                               # tn, fp, fn, tp per block
        for j, (t, p) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            cm[:, j] = np.bincount(self.inv, weights=((y == t) & (pred == p)).astype(float), minlength=self.nb)

        def frm(c):                                                # c [..., 4]
            tn, fp, fn, tp = (c[..., i] for i in range(4))
            with np.errstate(invalid="ignore", divide="ignore"):
                rn, ra = tn / (tn + fp), tp / (tp + fn)
                f1n = 2 * tn / (2 * tn + fp + fn)                  # F1 of class Normal (positive = Normal): TP = tn
                f1a = 2 * tp / (2 * tp + fp + fn)
            return dict(balanced_accuracy=(rn + ra) / 2, macro_f1=(f1n + f1a) / 2, normal_recall=rn)

        point = {k: float(v) for k, v in frm(cm.sum(0)).items()}
        point["roc_auc"] = float(roc_auc_score(y, P1)) if len(np.unique(y)) == 2 else np.nan
        bt = frm(self.counts @ cm)
        aucs = np.full(len(self.counts), np.nan)
        for b, draw in enumerate(self.draw):
            idx = np.concatenate([self.rows[i] for i in draw])
            yy = y[idx]
            if yy.min() != yy.max():
                aucs[b] = roc_auc_score(yy, P1[idx])
        bt["roc_auc"] = aucs
        out = {}
        for k, v in point.items():
            lo, hi = np.nanpercentile(bt[k], [2.5, 97.5]) if np.isfinite(bt[k]).any() else (np.nan, np.nan)
            out[k], out[f"{k}_ci_low"], out[f"{k}_ci_high"] = v, float(lo), float(hi)
        out["boot_valid"] = int(np.isfinite(bt["balanced_accuracy"]).sum())
        return out


def fit_predict(FP, TB, name, Xtr, ytr, Xte):
    spec = FP.reg("label_binary")[name]
    m = TB.fit_model(spec["base"], spec["params"], spec["variant"], Xtr, ytr, 2, early_stop=bool(spec.get("early_stop")))
    return m.predict_proba(Xte)[:, 1]


def run(n_boot, shuffles, do_cv, log=print):
    import v2_tab as TB
    sys.argv = ["x", "--stages", "0"]
    import full_pipeline as FP
    from v2_data import DataV2
    FP.D = DataV2()
    d, sets = load()
    folds = pd.read_csv(REP / "fold_assignment.csv").set_index("window_id").fold
    d["fold"] = d.window_id.map(folds)
    y_all = d.label_binary.to_numpy(int)
    is_train = (d.split == "train").to_numpy()
    rows = []
    t0 = time.time()
    for sname, cols in sets.items():
        valid = d[[f"rel__{c}" for c in cols]].notna().any(axis=1).to_numpy()      # the same windows for rel and raw
        for rep in ("rel", "raw"):
            X = matrices(d, cols, rep, is_train & valid)
            for (src, dst) in DIRECTIONS:
                tr = (d.system == src).to_numpy() & is_train & valid
                te = (d.system == dst).to_numpy() & (~is_train) & valid
                ytr, yte = y_all[tr], y_all[te]
                boot = Boot(d.block_id.to_numpy()[te], yte, n_boot)
                for mi, name in enumerate(MODELS):
                    P1 = fit_predict(FP, TB, name, X[tr], ytr, X[te])
                    base = dict(direction=f"{src}->{dst}", train_system=src, test_system=dst, feature_set=SET_NAMES[sname], representation=rep, model=name, n_features=len(cols),
                                n_train=int(tr.sum()), n_train_normal=int((ytr == 0).sum()), n_test=int(te.sum()), n_test_normal=int((yte == 0).sum()))
                    rows.append(dict(base, control="real labels", **boot.metrics(P1, (P1 >= 0.5).astype(int))))
                    sh = []
                    for r in range(shuffles):
                        ys = np.random.default_rng(SEED + 100 * mi + r).permutation(ytr)
                        Ps = fit_predict(FP, TB, name, X[tr], ys, X[te])
                        sh.append(boot.metrics(Ps, (Ps >= 0.5).astype(int)))
                    agg = {k: float(np.mean([s[k] for s in sh])) for k in sh[0] if not k.endswith(("_ci_low", "_ci_high")) and k != "boot_valid"}
                    rows.append(dict(base, control=f"shuffled labels (mean of {shuffles})", **agg,
                                     balanced_accuracy_ci_low=float(np.min([s["balanced_accuracy"] for s in sh])), balanced_accuracy_ci_high=float(np.max([s["balanced_accuracy"] for s in sh]))))
                    log(f"  {sname:<18} {rep:<3} {src}->{dst} {name:<10} BA {rows[-2]['balanced_accuracy']:.3f} ({rows[-2]['balanced_accuracy_ci_low']:.3f}-{rows[-2]['balanced_accuracy_ci_high']:.3f}) "
                        f"NR {rows[-2]['normal_recall']:.3f} AUC {rows[-2]['roc_auc']:.3f} | shuffled BA {agg['balanced_accuracy']:.3f}  ({time.time() - t0:.0f} s)")
            if do_cv:                                      # in-system validation with the 10 block-grouped training folds (XGBoost), for context
                for src in ("SN", "TT"):
                    tr = (d.system == src).to_numpy() & is_train & valid
                    oof = np.full(len(d), -1)
                    for k in range(1, 11):
                        a, b = tr & (d.fold.to_numpy() != k), tr & (d.fold.to_numpy() == k)
                        if b.sum() == 0 or len(np.unique(y_all[a])) < 2:
                            continue
                        oof[b] = (fit_predict(FP, TB, "XGB", X[a], y_all[a], X[b]) >= 0.5).astype(int)
                    m = oof[tr] >= 0
                    from sklearn.metrics import balanced_accuracy_score
                    rows.append(dict(direction=f"{src}->{src}", train_system=src, test_system=src, feature_set=SET_NAMES[sname], representation=rep, model="XGB", n_features=len(cols), control="10-fold CV on the training split",
                                     n_train=int(tr.sum()), n_test=int(m.sum()), balanced_accuracy=float(balanced_accuracy_score(y_all[tr][m], oof[tr][m]))))
                    log(f"  {sname:<18} {rep:<3} {src} 10-fold CV XGB BA {rows[-1]['balanced_accuracy']:.3f}")
    df = pd.DataFrame(rows)
    df.insert(0, "generated", time.strftime("%Y-%m-%d %H:%M:%S"))
    df.to_csv(OUT, index=False, float_format="%.6f")
    return df


def reproduce(n_boot=1000, log=lambda *_: None):
    """recompute the real-label XGBoost rows of the two cross-system directions for every feature set and representation (same seeds); used by the proofs"""
    import v2_tab as TB
    sys.argv = ["x", "--stages", "0"]
    import full_pipeline as FP
    from v2_data import DataV2
    FP.D = DataV2()
    d, sets = load()
    y_all = d.label_binary.to_numpy(int)
    is_train = (d.split == "train").to_numpy()
    out = []
    for sname, cols in sets.items():
        valid = d[[f"rel__{c}" for c in cols]].notna().any(axis=1).to_numpy()
        for rep in ("rel", "raw"):
            X = matrices(d, cols, rep, is_train & valid)
            for (src, dst) in (("SN", "TT"), ("TT", "SN")):
                tr = (d.system == src).to_numpy() & is_train & valid
                te = (d.system == dst).to_numpy() & (~is_train) & valid
                boot = Boot(d.block_id.to_numpy()[te], y_all[te], n_boot)
                P1 = fit_predict(FP, TB, "XGB", X[tr], y_all[tr], X[te])
                out.append(dict(direction=f"{src}->{dst}", feature_set=SET_NAMES[sname], representation=rep, **boot.metrics(P1, (P1 >= 0.5).astype(int))))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--shuffles", type=int, default=2)
    ap.add_argument("--no-cv", action="store_true")
    a = ap.parse_args()
    df = run(a.boot, a.shuffles, not a.no_cv, log=lambda s: print(s, flush=True))
    print(f"wrote {OUT} ({len(df)} rows)")


if __name__ == "__main__":
    main()
