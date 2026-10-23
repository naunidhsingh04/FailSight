"""BOTH data sets in one training set. Same layout idea as the project's earlier SN + TT combined set: the model input is the UNION of the AnoMod inputs and the LO2 inputs; the part a data set does
not have is 0 after scaling, plus a flag column that says which data set a window comes from. Each data set keeps its own preprocessing (fitted on that data set's training rows of the fold), its own
80:20 split and its own folds; the pooled training part = AnoMod training part + LO2 training part, the pooled test part = AnoMod test part + LO2 test part (never resampled). Sequences are the last 3
windows of the same run (LO2 windows are 5 s long, AnoMod windows 30 s). Labels: binary (Normal / Anomaly) only, because the multi-class labels of the two data sets are different things.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import hybrid_anomod as HA          # noqa: E402
from lo2_data import LO2Data        # noqa: E402

T3 = 3


class BothData:
    ds = "both"
    class_names = {"label_binary": ["Normal", "Anomaly"]}

    def __init__(self):
        self.A, self.L = HA.AnoModData(), LO2Data()
        A, L = self.A, self.L
        self.NA, self.NL = len(A.df), len(L.df)
        self.fa, self.fl = A.n_feat_out, L.n_feat_out
        self.F = self.fa + self.fl + 1
        self.df = pd.concat([A.df[["window_id", "run_id", "block_id"]].assign(dataset="AnoMod"), L.df[["window_id", "run_id", "scenario"]].assign(dataset="LO2", block_id=L.block_ids())[["window_id", "run_id", "block_id", "dataset"]]], ignore_index=True)
        self.dataset = self.df.dataset.to_numpy()
        self.train_pos = np.concatenate([A.train_pos, L.train_pos + self.NA])
        self.test_pos = np.concatenate([A.test_pos, L.test_pos + self.NA])
        self.y = {"label_binary": np.concatenate([A.y["label_binary"], L.y["label_binary"]])}
        self.folds = [(np.concatenate([A.folds[k][0], L.folds[k][0] + self.NA]), np.concatenate([A.folds[k][1], L.folds[k][1] + self.NA])) for k in range(len(A.folds))]
        k5 = np.zeros(len(self.train_pos), int)
        idx = {p: i for i, p in enumerate(self.train_pos)}
        for k, (_, va) in enumerate(self.folds, 1):
            k5[[idx[p] for p in va]] = k
        self.fold5 = k5
        self.ctx = np.vstack([A.ctx[:, -T3:], L.ctx + self.NA])
        self._w = {}
        self.binary_cols = list(A.binary_cols) + [self.fa + c for c in L.binary_cols] + [self.F - 1]
        self.runid_cols = list(A.runid_cols) + [self.fa + c for c in L.runid_cols]
        self.node_names = [f"AnoMod:{n}" for n in A.node_names] + [f"LO2:{n}" for n in L.node_names]
        self.node_idx = [list(i) for i in A.node_idx] + [[self.fa + c for c in i] for i in L.node_idx]
        na, nl = len(A.node_idx), len(L.node_idx)
        adj = np.zeros((na + nl, na + nl), int)
        adj[:na, :na], adj[na:, na:] = A.adj, L.adj
        self.adj = adj
        self.tmpl_idx = [self.fa + c for c in L.tmpl_idx]
        self.side_idx = [i for i in range(self.F) if i not in set(self.tmpl_idx)]
        self.tmpl_emb, self.tmpl_adj = L.tmpl_emb, L.tmpl_adj
        an = A.gprep.output_names
        a_logs = [i for i, n in enumerate(an) if n.startswith(("api_", "log_"))]
        a_met = [i for i in range(self.fa) if i not in set(a_logs)]
        l_logs, l_met = L.modal_idx
        self.modal_idx = (a_logs + [self.fa + c for c in l_logs], a_met + [self.fa + c for c in l_met] + [self.F - 1])

    def block_ids(self):
        return self.df.block_id.to_numpy()

    def W(self, k):
        if k not in self._w:
            out = np.zeros((self.NA + self.NL, self.F), np.float32)
            out[:self.NA, :self.fa] = self.A.W(k)
            out[self.NA:, self.fa:self.fa + self.fl] = self.L.W(k)
            out[self.NA:, -1] = 1.0
            self._w[k] = out
            self.A.drop_W(k)
            self.L.drop_W(k)
        return self._w[k]

    def drop_W(self, k, keep=False):
        if not keep:
            self._w.pop(k, None)

    def seq(self, W, pos):
        return W[self.ctx[pos]].astype(np.float32)

    def group_of(self, pos):
        return self.df.block_id.to_numpy()[pos]

    def spec(self, label):
        return dict(T=T3, F=self.F, C=2, node_idx=self.node_idx, adj=self.adj, tmpl_idx=self.tmpl_idx, side_idx=self.side_idx, tmpl_emb=self.tmpl_emb, tmpl_adj=self.tmpl_adj, modal_idx=self.modal_idx)
