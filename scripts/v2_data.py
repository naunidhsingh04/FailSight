"""Shared data / state / logging layer of the full retrain (step 2). Uses preprocess.py, split_ids.csv and fold_assignment.csv."""
import sys, os, time, json, pickle, warnings
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common                                    # metric helpers (full_metrics, quick, scope_scores)
from common import quick, scope_scores, full_metrics, CLASS_NAMES, LABELS, SEED, T
import preprocess as P
from imblearn.over_sampling import SMOTE

ROOT = Path(__file__).resolve().parent.parent
FAST = bool(os.environ.get("V2_FAST"))            # dry-run switch used only to test the pipeline code (separate folders, 2 epochs)
REP_IN = ROOT / "report_outputs"                  # split_ids.csv, fold_assignment.csv (read only)
V2 = ROOT / "outputs" / ("v2_fast" if FAST else "v2_state")
MODELS = ROOT / "outputs" / "models_fast" if FAST else ROOT / "models"
REP = ROOT / "outputs" / "rep_fast" if FAST else REP_IN
FIG = REP / "figures"
for d in (V2, MODELS / "folds", FIG, REP):
    d.mkdir(parents=True, exist_ok=True)
common.LOG = ROOT / "outputs" / ("log_v2_fast.txt" if FAST else "log_v2.txt")     # warnings + progress of this run go here (first run's log.txt untouched)
log = common.log
MAX_EPOCHS, ES_PATIENCE, LR_PATIENCE = (2 if FAST else 150), 12, 4
LEVEL_NAMES = common.LEVEL_NAMES
NOT_CONVERGED = "not converged"

TREES = ["RF", "ET", "XGB", "LGBM"]
TAB = ["LR"] + TREES
DEEP = ["CNN_BiGRU_Att", "CNN_BiLSTM_Att", "Fusion", "Transformer"]


def save_pkl(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def load_pkl(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def state_path(*parts):
    return V2 / ("__".join(str(x) for x in parts) + ".pkl")


class Progress:
    """timestamped progress with a simple ETA from the average job time of the current stage"""
    def __init__(self, stage, total):
        self.stage, self.total, self.done, self.t0, self.secs = stage, total, 0, time.time(), 0.0

    def skip(self, what):
        self.done += 1
        log(f"[{self.stage}] {self.done}/{self.total} skipped (already done): {what}")

    def finished(self, what, secs):
        self.done += 1
        self.ran = getattr(self, "ran", 0) + 1
        self.secs += secs
        left = (self.total - self.done) * self.secs / self.ran
        log(f"[{self.stage}] {self.done}/{self.total} done: {what} in {secs:.0f}s | stage elapsed {(time.time()-self.t0)/60:.1f} min | "
            f"ETA this stage ~{left/60:.1f} min")


class DataV2:
    def __init__(self):
        df = P.load_windows()
        sp = pd.read_csv(REP_IN / "split_ids.csv")
        fa = pd.read_csv(REP_IN / "fold_assignment.csv")
        n0 = len(df)
        df = df.merge(sp[["window_id", "split"]], on="window_id", how="left").merge(fa[["window_id", "fold"]], on="window_id", how="left")
        assert len(df) == n0 and df.split.notna().all() and df.fold.notna().all()
        self.df = df.reset_index(drop=True)
        self.sys = self.df.system_is_TT.to_numpy(int)
        self.train_pos = np.where(self.df.split == "train")[0]
        self.test_pos = np.where(self.df.split == "test")[0]
        assert (len(self.train_pos), len(self.test_pos)) == (2549, 647), (len(self.train_pos), len(self.test_pos))
        fold = self.df.fold.to_numpy(int)
        self.folds = [(self.train_pos[fold[self.train_pos] != k], self.train_pos[fold[self.train_pos] == k]) for k in range(1, 11)]
        assert sorted(np.concatenate([v for _, v in self.folds]).tolist()) == self.train_pos.tolist()
        self.y = {"label_binary": self.df.label_binary.to_numpy(int),
                  "label_level": self.df.label_level.map({n: i for i, n in enumerate(LEVEL_NAMES)}).to_numpy(int)}
        self.ytr = {l: self.y[l][self.train_pos] for l in LABELS}
        self.yte = {l: self.y[l][self.test_pos] for l in LABELS}
        self.sys_tr, self.sys_te = self.sys[self.train_pos], self.sys[self.test_pos]
        self.tr_index = {p: i for i, p in enumerate(self.train_pos)}
        self._ctx()
        self._w = {}
        self.gprep = None
        self.global_prep()

    # ---- last-6-window context: same run, same split, contiguous in time; earliest window repeated at the start of a chain
    def _ctx(self):
        run, spl, wi = self.df.run_id.to_numpy(), self.df.split.to_numpy(), self.df.window_index.to_numpy()
        ctx = np.zeros((len(self.df), T), dtype=int)
        pad = 0
        for i in range(len(self.df)):
            chain = [i]
            for j in range(1, T):
                p = i - j
                if p >= 0 and run[p] == run[i] and spl[p] == spl[i] and wi[p] == wi[chain[-1]] - 1:
                    chain.append(p)
                else:
                    break
            pad += T - len(chain)
            chain += [chain[-1]] * (T - len(chain))
            ctx[i] = chain[::-1]
        self.ctx, self.padded_slots = ctx, pad

    def global_prep(self):
        if self.gprep is None:
            self.gprep = P.fit_preprocessor(self.df.iloc[self.train_pos])
            self.n_feat = self.gprep.n_out
            g = self.gprep.feature_names()["groups"]
            self.groups = (len(g["api"]), len(g["log"]), len(g["metric"]))
            assert sum(self.groups) + 4 == self.n_feat
        return self.gprep

    def prep(self, k):
        """preprocessor fitted on the training rows of fold k (or all train rows for 'final'); same column set as the global one"""
        rows = self.train_pos if k == "final" else self.folds[k - 1][0]
        if k == "final":
            return self.gprep
        p = P.fit_preprocessor(self.df.iloc[rows])
        g = self.gprep
        p.keep, p.kept_cols, p.output_names, p.n_out, p.const_cols = g.keep, g.kept_cols, g.output_names, g.n_out, g.const_cols
        return p

    def W(self, k):
        """all windows (3196 x n_feat) transformed with the preprocessor of fold k"""
        if k not in self._w:
            self._w[k] = self.prep(k).transform(self.df)
        return self._w[k]

    def seq(self, W, pos):
        return W[self.ctx[pos]].astype(np.float32)

    def ylab(self, label, pos):
        return self.y[label][pos]


def smote_resample(X2d, y, round_last=0, seed=SEED):
    """SMOTE with k_neighbors = min(5, smallest class - 1); last `round_last` columns (binary indicators) are rounded back to 0/1"""
    k = min(5, int(np.bincount(y).min()) - 1)
    Xs, ys = SMOTE(random_state=seed, k_neighbors=k).fit_resample(X2d, y)
    return Xs, ys.astype(int)


def smote_seq(X, y):
    n, t, f = X.shape
    Xs, ys = smote_resample(X.reshape(n, -1), y)
    Xs = Xs.reshape(-1, t, f).astype(np.float32)
    Xs[:, :, -4:] = np.clip(np.round(Xs[:, :, -4:]), 0, 1)       # 3 was_missing indicators + system_is_TT stay 0/1
    return Xs, ys


def balanced_class_weight(y):
    c = np.bincount(y)
    return {i: float(len(y) / (len(c) * c[i])) for i in range(len(c))}


def fold_metrics(D, label, res):
    """per-fold table from a CV-result dict (folds[k] = train_rows, val_rows, train_proba, val_proba)"""
    rows = []
    names = CLASS_NAMES[label]
    for k, f in sorted(res["folds"].items()):
        yv, yt = D.ytr[label][[D.tr_index[p] for p in f["val_rows"]]], D.ytr[label][[D.tr_index[p] for p in f["train_rows"]]]
        m = full_metrics(yv, f["val_proba"], names)
        t = quick(yt, f["train_proba"])
        rows.append(dict(fold=k, accuracy=m["accuracy"], balanced_accuracy=m["balanced_accuracy"], macro_f1=m["macro_f1"],
                         normal_recall=m["normal_recall"], roc_auc=m["roc_auc"], train_balanced_accuracy=t[1], train_macro_f1=t[0]))
    return pd.DataFrame(rows)


def oof_of(D, res):
    C = res["folds"][1]["val_proba"].shape[1]
    oof = np.zeros((len(D.train_pos), C))
    for f in res["folds"].values():
        oof[[D.tr_index[p] for p in f["val_rows"]]] = f["val_proba"]
    return oof


def mean_val(D, label, res, col="balanced_accuracy"):
    return float(fold_metrics(D, label, res)[col].mean())
