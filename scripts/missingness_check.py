"""Missingness and shortcut check. Do the saved final models rely on missingness patterns (the was_missing_* indicators, the share of missing readings,
the system flag) instead of real signals?  Nothing saved is modified; every extra model trained here is a throw-away and is never written to disk.

  1. masking      every saved final model (soft voting excluded: it needs 40 fold models) is scored on the 647 test windows after a random 10/30/50/70% of the
                  real feature values of the API, log, metric group (or all three) are set to missing. The was_missing_* indicators are recomputed by the real
                  preprocessor (the group indicator fires when >= 50% of the group's features are missing). Several mask draws per cell; 95% block-bootstrap CI.
  2. indicator-only  a small throw-away classifier on the 3 was_missing_* indicators (and on indicators + per-window missing shares), 10 block-grouped folds
                  on the training split and one fit on the whole training split scored on test, with a shuffled-label control.
  3. no shortcuts a throw-away XGBoost (same settings as the saved one) trained on the training split without the was_missing_* indicators and/or without the
                  system flag, next to the saved XGBoost.

python scripts/missingness_check.py [--reps 3] [--boot 1000]      -> report_outputs/missingness_check.csv"""
import argparse
import json
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
OUT = REP / "missingness_check.csv"
SEED = 20260101
LEVELS_PCT = (10, 30, 50, 70)
GROUPS = ("api", "log", "metric", "all")
GMAP = {"api": "api", "log": "logs", "metric": "metrics"}      # preprocessor group names


# ---------------------------------------------------------------------------------------------------- shared helpers
def balanced_accuracy_from_cm(cm):
    """cm [..., C, C] (rows true, columns predicted); classes absent from the true labels are ignored (as sklearn does)"""
    tot = cm.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.diagonal(cm, axis1=-2, axis2=-1) / tot
    return np.nanmean(np.where(tot > 0, rec, np.nan), axis=-1)


class BlockBoot:
    """95% block bootstrap over the test blocks (a block is 10 consecutive windows of one run); the same resamples for every cell"""

    def __init__(self, blocks, n_boot, seed=SEED):
        self.ids, self.inv = np.unique(blocks, return_inverse=True)
        nb = len(self.ids)
        rng = np.random.default_rng(seed)
        draw = rng.integers(0, nb, (n_boot, nb))
        self.counts = np.stack([np.bincount(d, minlength=nb) for d in draw]).astype(float)       # [B, nb]

    def ba(self, y, preds, n_classes):
        """y [n]; preds [R, n] decisions of R mask draws -> (point = mean over draws, lo, hi)"""
        nb, R = len(self.ids), preds.shape[0]
        cm = np.zeros((R, nb, n_classes, n_classes))
        for r in range(R):
            np.add.at(cm[r], (self.inv, y, preds[r]), 1)
        point = float(np.mean([balanced_accuracy_from_cm(cm[r].sum(0)) for r in range(R)]))
        boot = np.mean([balanced_accuracy_from_cm((self.counts @ cm[r].reshape(nb, -1)).reshape(-1, n_classes, n_classes)) for r in range(R)], axis=0)
        return point, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    def normal_recall(self, y, pred, n_classes):
        nb = len(self.ids)
        cm = np.zeros((nb, n_classes, n_classes))
        np.add.at(cm, (self.inv, y, pred), 1)
        pt = cm.sum(0)[0, 0] / max(cm.sum(0)[0].sum(), 1)
        b = (self.counts @ cm.reshape(nb, -1)).reshape(-1, n_classes, n_classes)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = b[:, 0, 0] / b[:, 0].sum(-1)
        return float(pt), float(np.nanpercentile(r, 2.5)), float(np.nanpercentile(r, 97.5))


def setup(load_models=True):
    sys.argv = ["x", "--stages", "0"]
    import full_pipeline as FP
    import v2_data
    from v2_data import DataV2
    v2_data.common.LOG = ROOT / "outputs" / "missingness_log.txt"
    D = DataV2()
    FP.D = D
    reg = None
    if load_models:
        from app.backend.registry import Registry
        reg = Registry().load()
    return D, FP, reg


def raw_matrix(D, prep):
    X = D.df[prep.raw_cols].to_numpy(float).copy()
    X[~np.isfinite(X)] = np.nan
    return X


def mask_cells(X, prep, group, p, seed):
    """set a random share p of the PRESENT values of `group` ('api'/'log'/'metric'/'all') to missing"""
    if p == 0:
        return X
    cols = np.arange(X.shape[1]) if group == "all" else prep.group_idx[GMAP[group]]
    rng = np.random.default_rng(seed)
    Xm = X.copy()
    sub = Xm[:, cols]
    sub[rng.random(sub.shape) < p] = np.nan
    Xm[:, cols] = sub
    return Xm


def missing_shares(prep, X, sysid):
    """per-window share of the features that exist for the system that are missing: overall, api, log, metric (what the real pipeline's indicators are built from)"""
    n = len(X)
    out = np.zeros((n, 4))
    for s in (0, 1):
        m = sysid == s
        if not m.any():
            continue
        nan = np.isnan(prep._log(X[m]))
        for gi, idx in enumerate([np.arange(X.shape[1])] + [prep.group_idx[GMAP[g]] for g in ("api", "log", "metric")]):
            ex = ~prep.absent[s, idx]
            out[np.where(m)[0], gi] = nan[:, idx][:, ex].mean(1) if ex.any() else 1.0
    return out


# ---------------------------------------------------------------------------------------------------- 1. masking
def masking(D, reg, reps, n_boot, models=None, log=print):
    prep = reg.prep
    X = raw_matrix(D, prep)
    sysid = D.sys
    boot = BlockBoot(D.df.block_id.to_numpy()[D.test_pos], n_boot)
    rows = []
    entries = [(l, n) for (l, n), e in reg.entries.items() if e["available"] and n != "SoftVote" and (models is None or n in models)]
    conds = [("none", 0)] + [(g, p) for g in GROUPS for p in LEVELS_PCT]
    t0 = time.time()
    for ci, (g, pct) in enumerate(conds):
        nrep = 1 if pct == 0 else reps
        preds = {e: [] for e in entries}
        for r in range(nrep):
            Xm = mask_cells(X, prep, g, pct / 100, SEED + 1000 * ci + r)
            W = prep.transform_array(Xm, sysid)
            Xt, St = W[D.test_pos], W[D.ctx[D.test_pos]].astype(np.float32)
            cache = {}
            for (l, n) in entries:
                preds[(l, n)].append(reg.predict_proba(l, n, Xt, St, cache).argmax(1))
        for (l, n) in entries:
            C = 2 if l == "label_binary" else 5
            y = D.yte[l]
            pt, lo, hi = boot.ba(y, np.array(preds[(l, n)]), C)
            P_ = np.array(preds[(l, n)])
            nr = float(np.mean([(p_[y == 0] == 0).mean() for p_ in P_]))                # Normal recall (point)
            rows.append(dict(kind="masking", label_version=l, model=n, group=g, mask_pct=pct, draws=nrep, balanced_accuracy=pt, ci_low=lo, ci_high=hi,
                             normal_recall=nr, share_predicted_anomaly=float(np.mean(P_ != 0))))
        log(f"  masking {g:>6} {pct:>2}%  ({time.time() - t0:.0f} s)")
    return rows


# ---------------------------------------------------------------------------------------------------- 2. indicator-only classifier
def _small_xgb(seed=42):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, min_child_weight=5, subsample=0.8, tree_method="hist", n_jobs=-1, random_state=seed, verbosity=0)


def _fit_small(Xtr, ytr, seed=42):
    from sklearn.utils.class_weight import compute_sample_weight
    clf = _small_xgb(seed)
    clf.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
    return clf


def indicator_only(D, prep, n_boot, n_shuffle=5, log=print):
    from sklearn.metrics import balanced_accuracy_score
    X = raw_matrix(D, prep)
    W = prep.transform_array(X, D.sys)
    ind_cols = [prep.output_names.index(f"was_missing_{GMAP[g]}") for g in ("api", "log", "metric")]
    ind = W[:, ind_cols]
    shares = missing_shares(prep, X, D.sys)
    sets = {"was_missing_* indicators (3 columns)": ind, "indicators + missing shares (overall, api, log, metric: 7 columns)": np.hstack([ind, shares])}
    boot = BlockBoot(D.df.block_id.to_numpy()[D.test_pos], n_boot)
    rows = []
    for l in ("label_binary", "label_level"):
        C = 2 if l == "label_binary" else 5
        y = D.y[l]
        for name, F in sets.items():
            for ctl in ("real labels", "shuffled labels"):
                n_rep = 1 if ctl == "real labels" else n_shuffle
                oof_ba, te_preds = [], []
                for r in range(n_rep):
                    rng = np.random.default_rng(SEED + r)
                    ytr_all = y.copy()
                    if ctl == "shuffled labels":
                        ytr_all[D.train_pos] = rng.permutation(y[D.train_pos])
                    oof = np.full(len(y), -1)
                    for (tr, va) in D.folds:                               # 10 block-grouped folds (fold_assignment.csv), validation labels are the real ones
                        oof[va] = _fit_small(F[tr], ytr_all[tr]).predict(F[va])
                    oof_ba.append(balanced_accuracy_score(y[D.train_pos], oof[D.train_pos]))
                    te_preds.append(_fit_small(F[D.train_pos], ytr_all[D.train_pos]).predict(F[D.test_pos]))
                pt, lo, hi = boot.ba(D.yte[l], np.array(te_preds), C)
                rows.append(dict(kind="indicator_only", label_version=l, model=name, group=ctl, mask_pct=np.nan, draws=n_rep, balanced_accuracy=pt, ci_low=lo, ci_high=hi,
                                 cv_balanced_accuracy=float(np.mean(oof_ba))))
                log(f"  indicator-only {l[6:]:>6} {name[:32]:<32} {ctl:<15} test BA {pt:.3f} ({lo:.3f}-{hi:.3f}), 10-fold {np.mean(oof_ba):.3f}")
    return rows


# ---------------------------------------------------------------------------------------------------- 3. throw-away XGBoost without the shortcuts
def no_shortcut_xgb(D, FP, reg, n_boot, log=print):
    import v2_tab as TB
    prep = reg.prep
    W = D.W("final")
    names = prep.output_names
    ind = [names.index(f"was_missing_{GMAP[g]}") for g in ("api", "log", "metric")]
    sysc = names.index("system_is_TT")
    boot = BlockBoot(D.df.block_id.to_numpy()[D.test_pos], n_boot)
    variants = [("saved XGBoost (all 128 inputs)", None), ("throw-away, all 128 inputs (reproduction)", []), ("throw-away without was_missing_*", ind),
                ("throw-away without system_is_TT", [sysc]), ("throw-away without was_missing_* and without system_is_TT", ind + [sysc])]
    rows = []
    for l in ("label_binary", "label_level"):
        C = 2 if l == "label_binary" else 5
        spec = FP.reg(l)["XGB"]
        for vname, drop in variants:
            if drop is None:
                P = reg.predict_proba(l, "XGB", W[D.test_pos], D.seq(W, D.test_pos))
                n_in = W.shape[1]
            else:
                keep = [i for i in range(W.shape[1]) if i not in drop]
                m = TB.fit_model(spec["base"], spec["params"], spec["variant"], W[D.train_pos][:, keep], D.ytr[l], C, early_stop=bool(spec.get("early_stop")))
                P = m.predict_proba(W[D.test_pos][:, keep])
                n_in = len(keep)
            pred = P.argmax(1)
            pt, lo, hi = boot.ba(D.yte[l], pred[None], C)
            nr, nlo, nhi = boot.normal_recall(D.yte[l], pred, C)
            rows.append(dict(kind="no_shortcut_xgb", label_version=l, model=vname, group=f"{n_in} inputs", mask_pct=np.nan, draws=1, balanced_accuracy=pt, ci_low=lo, ci_high=hi,
                             normal_recall=nr, normal_recall_ci_low=nlo, normal_recall_ci_high=nhi))
            log(f"  xgb {l[6:]:>6} {vname:<58} inputs {n_in}: BA {pt:.3f} ({lo:.3f}-{hi:.3f}), Normal recall {nr:.3f} ({nlo:.3f}-{nhi:.3f})")
    return rows


# ---------------------------------------------------------------------------------------------------- verdict (rules fixed before looking at the numbers)
RULES = ["indicator-only model explains the performance if its test balanced accuracy keeps >= 50% of the saved XGBoost's lift over chance (0.5), binary labels",
         "masking explains the performance if the saved XGBoost still keeps >= 50% of its lift over chance (0.5) when 70% of ALL real values are masked and every indicator fires (binary labels): the call would then come from the missingness pattern",
         "the no-shortcut XGBoost reveals reliance if its test balanced accuracy is lower than the saved XGBoost's by more than 0.05 (binary labels)"]


def verdict(rows):
    d = pd.DataFrame(rows)
    mk = d[d.kind == "masking"]
    xb = mk[(mk.label_version == "label_binary") & (mk.model == "XGB")]
    base = float(xb[xb.mask_pct == 0].balanced_accuracy.iloc[0])
    lift = base - 0.5
    m70 = xb[(xb.group == "all") & (xb.mask_pct == 70)].iloc[0]
    io = d[(d.kind == "indicator_only") & (d.label_version == "label_binary") & (d.group == "real labels")]
    ind_best = io.sort_values("balanced_accuracy", ascending=False).iloc[0]
    ns = d[(d.kind == "no_shortcut_xgb") & (d.label_version == "label_binary")].set_index("model")
    saved = float(ns.loc["saved XGBoost (all 128 inputs)"].balanced_accuracy)
    both = ns.loc["throw-away without was_missing_* and without system_is_TT"]
    out = dict(saved_xgb_ba=base, lift=lift,
               masking_all70_ba=float(m70.balanced_accuracy), masking_all70_lift_kept=float((m70.balanced_accuracy - 0.5) / lift),
               indicator_only_ba=float(ind_best.balanced_accuracy), indicator_only_set=str(ind_best.model), indicator_only_lift_share=float((ind_best.balanced_accuracy - 0.5) / lift),
               no_shortcut_ba=float(both.balanced_accuracy), no_shortcut_drop=float(saved - both.balanced_accuracy))
    out["masking_explains"] = out["masking_all70_lift_kept"] >= 0.5
    out["indicator_explains"] = out["indicator_only_lift_share"] >= 0.5
    out["no_shortcut_relies"] = out["no_shortcut_drop"] > 0.05
    out["explains"] = bool(out["masking_explains"] or out["indicator_explains"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=3, help="mask draws per masking cell")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--models", default="", help="comma list of model names to restrict the masking part to (default: all)")
    a = ap.parse_args()
    t0 = time.time()
    D, FP, reg = setup()
    models = [m for m in a.models.split(",") if m] or None
    print("1. masking", flush=True)
    rows = masking(D, reg, a.reps, a.boot, models, log=lambda s: print(s, flush=True))
    print("2. indicator-only classifiers", flush=True)
    rows += indicator_only(D, reg.prep, a.boot, log=lambda s: print(s, flush=True))
    print("3. XGBoost without the shortcuts", flush=True)
    rows += no_shortcut_xgb(D, FP, reg, a.boot, log=lambda s: print(s, flush=True))
    df = pd.DataFrame(rows)
    df.insert(0, "generated", time.strftime("%Y-%m-%d %H:%M:%S"))
    df.to_csv(OUT, index=False, float_format="%.6f")
    print(f"wrote {OUT} ({len(df)} rows, {time.time() - t0:.0f} s)")
    print(json.dumps(verdict(rows), indent=1, default=float))


if __name__ == "__main__":
    main()
