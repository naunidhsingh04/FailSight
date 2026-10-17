"""Robustness variant "no_overlap": refit every final model after dropping the TRAIN windows whose 30 s span overlaps in time
with any TEST window of the same run. The official 80:20 split, folds and official results are not touched; the variant's
test probabilities are stored as outputs/_state/noov__<label>__<model>.pkl and picked up by report.test_table().

Same config and epoch count as the official final model (no re-selection), same SMOTE (random_state 42), preprocessing refitted on
the kept train rows only. Stacking: meta-learner refitted on the out-of-fold probabilities of the kept rows (the official CV
OOF probabilities, produced by models that still saw the overlapping windows) + variant base models' test probabilities.
SoftVote: average of the four variant final models (the official SoftVote averages 40 fold models; refitting those is not done here).
"""
import sys, os, pickle, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from xgboost import XGBClassifier
import train_models as TM          # reuses fit_net / load_state (module has no side effects without __main__)
from common import *

D = TM.D = Data()
log("=" * 60); log("no_overlap robustness variant started", time.ctime())

# ---- windows to drop: train windows overlapping in time with a test window of the same run
w = D.w
tr_idx = D.train_pos
drop = np.zeros(len(tr_idx), bool)
for run, g in w.groupby("run"):
    ts = g[g.split == "test"]
    if ts.empty:
        continue
    s, e = ts.window_start.to_numpy(), ts.window_end.to_numpy()
    for pos in g.index[g.split == "train"]:
        if ((w.window_start[pos].to_datetime64() < e) & (w.window_end[pos].to_datetime64() > s)).any():
            drop[np.searchsorted(tr_idx, pos)] = True
n_sn, n_tt = int(drop[D.sys_tr == 0].sum()), int(drop[D.sys_tr == 1].sum())
log(f"overlapping train windows dropped: SN {n_sn}, TT {n_tt}, total {int(drop.sum())} of {len(tr_idx)} (leakage_checks.csv: 98 and 102)")
assert (n_sn, n_tt) == (98, 102)
keep = np.where(~drop)[0]
log(f"train rows kept {len(keep)}; test rows unchanged {len(D.test_pos)}")

Wt = D.fit_transform(D.train_pos[keep])           # statistics from kept train rows only
Xtr_all, Xte = D.seq(Wt, D.train_pos), D.seq(Wt, D.test_pos)
from imblearn.over_sampling import SMOTE


def smote(label):
    X, y = Xtr_all[keep], D.ytr[label][keep]
    sm = SMOTE(random_state=SEED, k_neighbors=min(5, int(np.bincount(y).min()) - 1))
    Xs, ys = sm.fit_resample(X.reshape(len(X), -1), y)
    Xs = Xs.reshape(-1, T, X.shape[2]).astype(np.float32)
    Xs[:, :, -1] = np.clip(np.round(Xs[:, :, -1]), 0, 1)
    return Xs, ys.astype(int)


def save(label, name, test_proba, **kw):
    with open(STATE / f"noov__{label}__{name}.pkl", "wb") as f:
        pickle.dump(dict(test_proba=test_proba, n_train_rows=len(keep), dropped=int(drop.sum()), **kw), f)


for label in LABELS:
    Xs, ys = smote(label)
    probs = {}
    for name in TM.BASE:
        off = TM.load_state(label, name)
        t0 = time.time()
        r = TM.fit_net(name, off["cfg"], label, "final", Xs, ys, None, None, fixed_epochs=off["final_epochs"],
                       want_head=name in TM.HEAD_OF, evals={"train": Xtr_all[keep], "test": Xte})
        r.pop("model")
        probs[name] = r["probs"]["test"]
        save(label, name, r["probs"]["test"], epochs=off["final_epochs"], cfg=off["cfg"], secs=time.time() - t0)
        if name in TM.HEAD_OF:
            save(label, TM.HEAD_OF[name], r["head_probs"]["test"], epochs=off["final_epochs"], cfg=off["cfg"], secs=time.time() - t0)
        log(f"[{label}] {name} no_overlap refit {time.time()-t0:.0f}s ({off['final_epochs']} epochs, cfg {off['cfg']})")
    # stacking: OOF of the kept rows
    sb = ["CNN_BiGRU_Att", "Fusion", "Transformer"]
    oof = {b: np.zeros((len(tr_idx), len(CLASS_NAMES[label]))) for b in sb}
    for b in sb:
        st = TM.load_state(label, b)
        for k in range(1, 11):
            oof[b][st["folds"][k]["val_rows"]] = st["folds"][k]["val_proba"]
    Z = np.hstack([oof[b] for b in sb])[keep]
    meta = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, tree_method="hist", n_jobs=8, random_state=SEED, verbosity=0)
    meta.fit(Z, D.ytr[label][keep])
    save(label, "Stacking", meta.predict_proba(np.hstack([probs[b] for b in sb])))
    save(label, "SoftVote", np.mean([probs[n] for n in TM.BASE], axis=0))
    log(f"[{label}] no_overlap Stacking and SoftVote done")
log("no_overlap robustness variant finished")
