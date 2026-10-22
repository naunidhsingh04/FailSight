"""LO2 slice: window table -> model inputs. Same preprocessing steps as AnoMod, every statistic fitted on training rows only; LO2's own features, nothing from AnoMod.

Steps (fit on the training rows of the fold / of the 80% training part): counters were turned into rates when the windows were built (row-local rule); log1p on heavy-tailed
non-negative features; was_missing flag per group (logs, templates, metrics); median imputation; constant columns dropped; clipping at the 0.5 and 99.5 percentiles; StandardScaler;
then the 478 metric families are compressed to 32 principal components (fitted on the training rows) because 478 columns for ~700 windows would swamp the model.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
REP = ROOT / "report_outputs"
PROC = ROOT / "processed" / "lo2_hybrid"
SEED, T, K_TMPL, N_PC, BLOCK = 42, 3, 48, 32, 3
LEVELS = ["Normal", "HTTP400", "HTTP401", "HTTP404"]
CLASS_NAMES = {"label_binary": ["Normal", "Anomaly"], "label_level": LEVELS}


class LO2Prep:
    def fit(self, X, groups):
        """X: raw feature matrix (n, f) of TRAINING rows only; groups: array of 'logs'/'templates'/'metrics' per column"""
        X = np.array(X, dtype=np.float64, copy=True)
        X[~np.isfinite(X)] = np.nan
        self.groups = np.asarray(groups)
        self.n_fit = len(X)
        self.log_mask = np.zeros(X.shape[1], bool)
        for j in range(X.shape[1]):
            v = X[:, j][~np.isnan(X[:, j])]
            if len(v) >= 20 and v.min() >= 0 and np.ptp(v) > 0 and skew(v) > 2.0:
                self.log_mask[j] = True
        Xl = self._log(X)
        self.miss_groups = ["logs", "templates", "metrics"]
        self.median = np.nanmedian(np.where(np.isnan(Xl).all(0, keepdims=True), 0.0, Xl), axis=0)
        self.median = np.where(np.isnan(self.median), 0.0, self.median)
        Xi = np.where(np.isnan(Xl), self.median, Xl)
        self.keep = Xi.std(0) > 1e-12                                   # constant columns dropped
        self.lo, self.hi = np.percentile(Xi, 0.5, axis=0), np.percentile(Xi, 99.5, axis=0)
        Xc = np.clip(Xi, self.lo, self.hi)
        self.mean, self.scale = Xc.mean(0), np.where(Xc.std(0) < 1e-12, 1.0, Xc.std(0))
        Z = ((Xc - self.mean) / self.scale)[:, self.keep]
        self.kept_groups = self.groups[self.keep]
        mi = np.where(self.kept_groups == "metrics")[0]
        self.pca = PCA(n_components=min(N_PC, len(mi), Z.shape[0] - 1), random_state=SEED).fit(Z[:, mi]) if len(mi) else None
        self.n_pc = self.pca.n_components_ if self.pca is not None else 0
        return self

    def _log(self, X):
        X = X.copy()
        for j in np.where(self.log_mask)[0]:
            X[:, j] = np.log1p(np.clip(X[:, j], 0, None))
        return X

    def transform(self, X):
        X = np.array(X, dtype=np.float64, copy=True)
        X[~np.isfinite(X)] = np.nan
        flags = []
        for g in self.miss_groups:
            cols = self.groups == g
            flags.append((np.isnan(X[:, cols]).mean(1) >= 0.5).astype(np.float32) if cols.any() else np.zeros(len(X), np.float32))
        Xl = self._log(X)
        Xi = np.where(np.isnan(Xl), self.median, Xl)
        Z = ((np.clip(Xi, self.lo, self.hi) - self.mean) / self.scale)[:, self.keep]
        mi = np.where(self.kept_groups == "metrics")[0]
        nm = np.where(self.kept_groups != "metrics")[0]
        parts = [Z[:, nm]]
        if self.pca is not None:
            parts.append(self.pca.transform(Z[:, mi]))
        parts.append(np.stack(flags, 1))
        return np.hstack(parts).astype(np.float32)

    def names(self, raw_names):
        raw = np.asarray(raw_names)
        kept = raw[self.keep]
        nm = list(kept[self.kept_groups != "metrics"])
        return nm + [f"m_pc{i}" for i in range(self.n_pc)] + [f"was_missing_{g}" for g in self.miss_groups]


class LO2Data:
    ds = "lo2"
    class_names = CLASS_NAMES

    def __init__(self):
        df = pd.read_parquet(PROC / "windows.parquet").reset_index(drop=True)
        counts = np.load(PROC / "template_counts.npz")["counts"]
        tm = pd.read_csv(PROC / "templates.csv")
        sp = pd.read_csv(REP / "lo2_split_ids.csv", dtype={"run_id": str})
        assert "window_id" in sp.columns, "run lo2_hybrid_data.py parse first (window-level split file)"
        df = df.merge(sp[["window_id"]], on="window_id")
        self.df = df
        self.train_pos = np.where(df.split == "train")[0]
        self.test_pos = np.where(df.split == "test")[0]
        assert not set(df.run_id[self.train_pos]) & set(df.run_id[self.test_pos])
        self.y = {"label_binary": df.label_binary.to_numpy(int), "label_level": df.label_level.map({n: i for i, n in enumerate(LEVELS)}).to_numpy(int)}
        # model input columns (raw): logs, templates (top K by number of TRAINING windows they appear in), metrics
        log_cols = [c for c in df.columns if c.startswith("log_")]
        pres = (counts[self.train_pos][:, :-1] > 0).sum(0)
        top = np.argsort(-pres, kind="stable")[:K_TMPL]
        top = top[pres[top] > 0]
        self.tmpl_top = top
        other = counts.sum(1) - counts[:, top].sum(1)
        Tm = np.hstack([counts[:, top], other[:, None]])
        self.tmpl_names = [f"tmpl_{tm.source[i]}_{tm.template_id[i]}" for i in top] + ["tmpl_other"]
        met_cols = [c for c in df.columns if c.startswith("m_")]
        self.raw_names = log_cols + self.tmpl_names + met_cols
        self.raw_groups = ["logs"] * len(log_cols) + ["templates"] * len(self.tmpl_names) + ["metrics"] * len(met_cols)
        self.raw = np.hstack([df[log_cols].to_numpy(float), Tm.astype(float), df[met_cols].to_numpy(float)])
        self.tm_top = tm.iloc[top].reset_index(drop=True)
        self._w, self._p = {}, {}
        self._ctx()
        self.make_folds5()
        p = self.prep("final")
        self.out_names = p.names(self.raw_names)
        self.n_feat_out = len(self.out_names)
        n = self.out_names
        self.binary_cols = [i for i, x in enumerate(n) if x.startswith("was_missing")]
        self.runid_cols = [i for i, x in enumerate(n) if not x.startswith("was_missing")]
        self.build_graphs()

    # context: the last T windows of the same scenario (same run, same split, contiguous in time); earliest window repeated at the start
    def _ctx(self):
        run, sc, wi = self.df.run_id.to_numpy(), self.df.scenario.to_numpy(), self.df.window_index.to_numpy()
        ctx = np.zeros((len(self.df), T), dtype=int)
        for i in range(len(self.df)):
            chain = [i]
            for j in range(1, T):
                p = i - j
                if p >= 0 and run[p] == run[i] and sc[p] == sc[i] and wi[p] == wi[chain[-1]] - 1:
                    chain.append(p)
                else:
                    break
            chain += [chain[-1]] * (T - len(chain))
            ctx[i] = chain[::-1]
        self.ctx = ctx

    def block_ids(self):
        d = self.df
        return (d.run_id + "|" + d.scenario + "|b" + (d.window_index // BLOCK).astype(str)).to_numpy()

    def make_folds5(self):
        tr = self.train_pos
        blocks = self.block_ids()[tr]
        yl = self.y["label_level"][tr]
        p = REP / "hybrid_lo2_folds5.csv"
        if p.exists():
            fa = pd.read_csv(p).set_index("window_id").fold
            fold = fa.reindex(self.df.window_id.to_numpy()[tr]).to_numpy(int)
        else:
            fold = None
            for seed in range(42, 142):
                f = np.zeros(len(tr), int)
                for k, (_, va) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(np.zeros(len(tr)), yl, blocks)):
                    f[va] = k + 1
                if all(set(np.unique(yl[f == k])) == set(range(4)) for k in range(1, 6)):
                    fold = f
                    break
            fold = f if fold is None else fold
            pd.DataFrame(dict(window_id=self.df.window_id.to_numpy()[tr], fold=fold, group_type="run|scenario|3-window block", run_id=self.df.run_id.to_numpy()[tr], block_id=blocks)).to_csv(p, index=False)
        self.fold5 = fold
        self.folds = [(tr[fold != k], tr[fold == k]) for k in range(1, 6)]
        # overlap measured: validation windows whose context windows (the 2 before them) lie in the training rows of the same fold
        ov = []
        for tr_k, va_k in self.folds:
            trs = set(tr_k)
            ov.append(int(sum(any(c in trs for c in self.ctx[v][:-1]) for v in va_k)))
        self.context_overlap = ov

    def prep(self, k):
        if k not in self._p:
            rows = self.train_pos if k == "final" else self.folds[k - 1][0]
            self._p[k] = LO2Prep().fit(self.raw[rows], self.raw_groups)
        return self._p[k]

    def W(self, k):
        if k not in self._w:
            self._w[k] = self.prep(k).transform(self.raw)
        return self._w[k]

    def drop_W(self, k, keep=False):
        if not keep:
            self._w.pop(k, None)

    def seq(self, W, pos):
        return W[self.ctx[pos]].astype(np.float32)

    def group_of(self, pos):
        return self.block_ids()[pos]

    # ---- graphs / embeddings used by the hybrids (built from the 80% training part only; never from test windows)
    def build_graphs(self):
        W = self.W("final")[self.train_pos]
        names = self.out_names
        src_of = lambda n: next((s for s in ("client", "code", "key", "refresh-token", "service", "token", "user", "mysql", "locust") if f"_{s}_" in n), None)
        nodes = {}
        for i, n in enumerate(names):
            if n.startswith(("log_", "tmpl_")):
                s = src_of(n)
                nodes.setdefault(s or "other_logs", []).append(i)
            elif n.startswith("m_pc"):
                nodes.setdefault("metrics", []).append(i)
            else:
                nodes.setdefault("flags", []).append(i)
        self.node_names = list(nodes)
        self.node_idx = [nodes[k] for k in self.node_names]
        act = np.stack([W[:, i].mean(1) for i in self.node_idx], 1)
        C = np.nan_to_num(np.corrcoef(act.T))
        adj = (np.abs(C) >= 0.5).astype(int)
        np.fill_diagonal(adj, 0)
        for a, k in enumerate(self.node_names):                     # flags are linked to everything; every node keeps at least its strongest neighbour
            if k == "flags":
                adj[a, :] = 1
                adj[:, a] = 1
                adj[a, a] = 0
            elif adj[a].sum() == 0:
                o = np.argsort(-np.abs(C[a]))
                o = [x for x in o if x != a][0]
                adj[a, o] = adj[o, a] = 1
        self.adj = adj
        self.corr = C
        # template graph + frozen text embeddings of the templates (TF-IDF + truncated SVD: the transformers library / a downloadable BERT is not available here)
        tm = self.tmpl_names
        tidx = [i for i, n in enumerate(names) if n.startswith("tmpl_") and n != "tmpl_other"]
        self.tmpl_idx = tidx
        self.side_idx = [i for i in range(len(names)) if i not in set(tidx)]
        texts = list(self.tm_top.template)
        tf = TfidfVectorizer(token_pattern=r"[A-Za-z_<>*\.]+", lowercase=True).fit(texts)
        M = tf.transform(texts)
        nc = max(2, min(16, M.shape[1] - 1, len(texts) - 1))
        E = TruncatedSVD(n_components=nc, random_state=SEED).fit_transform(M)
        E = E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-9, None)
        self.tmpl_emb = E.astype(np.float32)
        # keep only template columns that survived constant-dropping, in the same order as tidx
        raw_tm = {n: i for i, n in enumerate(self.tmpl_names)}
        order = [raw_tm[names[i]] for i in tidx]
        self.tmpl_emb = self.tmpl_emb[order]
        pres = (self.raw[self.train_pos][:, [self.raw_names.index(names[i]) for i in tidx]] > 0).astype(float)
        inter = pres.T @ pres
        union = pres.sum(0)[:, None] + pres.sum(0)[None] - inter
        J = np.where(union > 0, inter / np.maximum(union, 1), 0)
        np.fill_diagonal(J, 0)
        ta = np.zeros_like(J, dtype=int)                              # k-nearest-neighbour graph (k = 4) on how often two templates appear in the same training windows (Jaccard)
        for a in range(len(ta)):
            for o in np.argsort(-J[a], kind="stable")[:4]:
                if o != a:
                    ta[a, o] = ta[o, a] = 1
        self.tmpl_adj = ta
        self.embedding_kind = "TF-IDF + truncated SVD of the template text (a downloadable BERT encoder / the transformers library is not available here)"
        # modalities for the multimodal hybrid
        met = [i for i, n in enumerate(names) if n.startswith(("m_pc", "was_missing"))]
        self.modal_idx = ([i for i in range(len(names)) if i not in set(met)], met)

    def spec(self, label):
        return dict(T=T, F=self.n_feat_out, C=len(self.class_names[label]), node_idx=self.node_idx, adj=self.adj, tmpl_idx=self.tmpl_idx, side_idx=self.side_idx,
                    tmpl_emb=self.tmpl_emb, tmpl_adj=self.tmpl_adj, modal_idx=self.modal_idx)

    def flat(self, pos, k="final"):
        """tabular (non-sequence) inputs for the classical algorithms: the current window only"""
        return self.W(k)[pos]
