"""Shared data prep, models and metrics for train_models.py."""
import json, time, warnings, logging, os, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support,
                             roc_auc_score, average_precision_score, confusion_matrix)
from imblearn.over_sampling import SMOTE

ROOT = Path(__file__).resolve().parent.parent
INP, OUT = ROOT / "inputs", ROOT / "outputs"
STATE = OUT / "_state"
for d in (OUT, STATE):
    d.mkdir(exist_ok=True)
SEED, T = 42, 6
LEVEL_NAMES = ["Normal", "Performance", "Service", "Database", "Code"]
BIN_NAMES = ["Normal", "Anomaly"]
CLASS_NAMES = {"label_binary": BIN_NAMES, "label_level": LEVEL_NAMES}
LABELS = ["label_binary", "label_level"]
CLIP = 10.0

LOG = OUT / "log.txt"


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M:%S ") + s + "\n")


def _showwarning(message, category, filename, lineno, file=None, line=None):
    log(f"WARNING {category.__name__}: {message} ({Path(filename).name}:{lineno})")


warnings.showwarning = _showwarning
warnings.simplefilter("default")


# ------------------------------------------------------------------ data
class Data:
    def __init__(self):
        w = pd.read_parquet(INP / "combined_windows.parquet").reset_index(drop=True)
        fl = pd.read_csv(INP / "combined_feature_list.csv")
        order = []
        for g in ("api", "logs", "metrics"):
            order += fl[fl.group == g].feature.tolist()
        self.order = order
        self.groups = {g: fl[fl.group == g].feature.tolist() for g in ("api", "logs", "metrics")}
        self.n_api, self.n_logs, self.n_met = (len(self.groups[g]) for g in ("api", "logs", "metrics"))
        self.feat_names = order + ["system_is_TT"]
        self.F = w[order].to_numpy(dtype=np.float64)
        self.w = w
        self.sys = (w.system == "TT").to_numpy().astype(int)  # 1 = TT
        self.pres = {s: fl.set_index("feature").loc[order, f"present_in_{s}"].to_numpy(dtype=bool) for s in ("SN", "TT")}
        self.is_train = (w.split == "train").to_numpy()
        self.train_pos = np.where(self.is_train)[0]
        self.test_pos = np.where(~self.is_train)[0]
        assert len(self.train_pos) == 2549 and len(self.test_pos) == 647
        self.y = {"label_binary": w.label_binary.to_numpy().astype(int),
                  "label_level": w.label_level.map({n: i for i, n in enumerate(LEVEL_NAMES)}).to_numpy().astype(int)}
        # context: the last T consecutive windows of the same run AND the same split (no test window ever feeds a train
        # sequence and vice versa); positions before the start of the contiguous chain repeat the earliest window.
        run, spl, wi = w.run.to_numpy(), w.split.to_numpy(), w.window_index.to_numpy()
        ctx = np.zeros((len(w), T), dtype=int)
        n_pad = 0
        for i in range(len(w)):
            chain = [i]
            for j in range(1, T):
                p = i - j
                if p >= 0 and run[p] == run[i] and spl[p] == spl[i] and wi[p] == wi[chain[-1]] - 1:
                    chain.append(p)
                else:
                    break
            n_pad += T - len(chain)
            chain = chain + [chain[-1]] * (T - len(chain))
            ctx[i] = chain[::-1]
        self.ctx = ctx
        self.n_padded_slots = n_pad
        fi = json.loads((INP / "combined_fold_indices.json").read_text())["folds"]
        keys = sorted(fi, key=lambda k: int(k))
        self.folds = [(np.array(fi[k]["train_idx"]), np.array(fi[k]["val_idx"])) for k in keys]
        assert len(self.folds) == 10
        # every train position appears in exactly one validation part
        allv = np.sort(np.concatenate([v for _, v in self.folds]))
        assert (allv == np.arange(len(self.train_pos))).all()
        self.ytr = {l: self.y[l][self.train_pos] for l in LABELS}
        self.yte = {l: self.y[l][self.test_pos] for l in LABELS}
        self.sys_tr, self.sys_te = self.sys[self.train_pos], self.sys[self.test_pos]
        self._ft = {}
        self._sm = {}

    def fit_transform(self, rows):
        """Impute (training median) + standardise per system with statistics of `rows` only; absent features -> 0."""
        out = np.zeros((len(self.w), self.F.shape[1] + 1), dtype=np.float32)
        for s, sid in (("SN", 0), ("TT", 1)):
            idx_all = np.where(self.sys == sid)[0]
            pres = self.pres[s]
            r = rows[self.sys[rows] == sid]
            Xr = self.F[r][:, pres]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                med = np.nanmedian(Xr, axis=0)
            med = np.where(np.isnan(med), 0.0, med)
            Xi = np.where(np.isnan(Xr), med, Xr)
            mu, sd = Xi.mean(0), Xi.std(0)
            sd = np.where(sd < 1e-8, 1.0, sd)
            Xa = self.F[idx_all][:, pres]
            Xa = np.where(np.isnan(Xa), med, Xa)
            Xa = np.clip((Xa - mu) / sd, -CLIP, CLIP)
            block = np.zeros((len(idx_all), self.F.shape[1] + 1), dtype=np.float32)
            block[:, np.where(pres)[0]] = Xa
            block[:, -1] = sid
            out[idx_all] = block
        return out

    def seq(self, Wt, pos):
        return Wt[self.ctx[pos]]  # (n, T, F+1)

    def fold_arrays(self, k):
        """k = 1..10 or 'final'. Returns transformed sequences for all train rows and test rows."""
        if k in self._ft:
            return self._ft[k]
        tr = self.folds[k - 1][0] if k != "final" else np.arange(len(self.train_pos))
        Wt = self.fit_transform(self.train_pos[tr])
        res = (self.seq(Wt, self.train_pos), self.seq(Wt, self.test_pos))
        self._ft[k] = res
        return res

    def smote(self, label, k):
        if (label, k) in self._sm:
            return self._sm[(label, k)]
        res = self._smote(label, k)
        self._sm[(label, k)] = res
        return res

    def _smote(self, label, k):
        Xtr_all, _ = self.fold_arrays(k)
        tr = self.folds[k - 1][0] if k != "final" else np.arange(len(self.train_pos))
        X, y = Xtr_all[tr], self.ytr[label][tr]
        cnt = np.bincount(y)
        sm = SMOTE(random_state=SEED, k_neighbors=min(5, int(cnt.min()) - 1))
        Xs, ys = sm.fit_resample(X.reshape(len(X), -1), y)
        Xs = Xs.reshape(-1, T, X.shape[2]).astype(np.float32)
        Xs[:, :, -1] = np.clip(np.round(Xs[:, :, -1]), 0, 1)  # keep system_is_TT binary
        return Xs, ys.astype(int), np.bincount(y, minlength=len(cnt)), np.bincount(ys, minlength=len(cnt))


# ------------------------------------------------------------------ metrics
def quick(y, P):
    pred = P.argmax(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (f1_score(y, pred, average="macro", labels=np.unique(y), zero_division=0),
                balanced_accuracy_score(y, pred))


def scope_scores(y, P, sysarr):
    out = {}
    for sc, m in (("ALL", np.ones(len(y), bool)), ("SN", sysarr == 0), ("TT", sysarr == 1)):
        out[sc] = quick(y[m], P[m])
    return out


def full_metrics(y, P, names):
    C = len(names)
    P = P / np.clip(P.sum(1, keepdims=True), 1e-12, None)
    pred = P.argmax(1)
    d = dict(n=len(y), accuracy=accuracy_score(y, pred))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d["balanced_accuracy"] = balanced_accuracy_score(y, pred)
        d["macro_f1"] = f1_score(y, pred, average="macro", labels=np.unique(y), zero_division=0)
        p, r, f, s = precision_recall_fscore_support(y, pred, labels=list(range(C)), zero_division=0)
    for i, n in enumerate(names):
        d[f"precision_{n}"], d[f"recall_{n}"], d[f"f1_{n}"], d[f"support_{n}"] = p[i], (r[i] if s[i] else np.nan), f[i], s[i]
    d["normal_recall"] = d["recall_Normal"]
    try:
        if C == 2:
            d["roc_auc"] = roc_auc_score(y, P[:, 1]); d["pr_auc"] = average_precision_score(y, P[:, 1])
        else:
            d["roc_auc"] = roc_auc_score(y, P, multi_class="ovr", average="macro", labels=list(range(C)))
            d["pr_auc"] = float(np.mean([average_precision_score(y == c, P[:, c]) for c in range(C)]))
        d["pr_auc_normal"] = average_precision_score(y == 0, P[:, 0])
    except Exception as e:
        d["roc_auc"] = d["pr_auc"] = d["pr_auc_normal"] = np.nan
    return d
