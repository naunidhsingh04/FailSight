"""Five hybrid deep-learning models on AnoMod: same windows, same 2,549 / 647 split, same preprocessing and labels as the existing models; 5-fold CV for the new work.

python scripts/hybrid_anomod.py [--models A,B] [--labels label_binary,label_level] [--until HH:MM] [--assemble-only]
Results -> report_outputs/hybrid_anomod_*.csv, frozen configs hybrid_anomod_frozen_*.json, models -> models/hybrid/anomod/ (git-ignored), state -> outputs/hybrid_state/anomod/.
Nothing of the earlier AnoMod models, results or proofs is read for training or changed. Only split_ids.csv and the window tables are read (as the existing models did).
"""
import argparse
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import hybrid_engine as E                      # noqa: E402  (sets the thread limits before TensorFlow is imported)
from v2_data import DataV2                     # noqa: E402  (existing data layer: windows + split_ids.csv; read only)
import v2_data                                 # noqa: E402
from common import CLASS_NAMES, T              # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from models_src.hybrid import ANOMOD_HYBRIDS   # noqa: E402

REP, ROOT = E.REP, E.ROOT
PREFIX = "hybrid_anomod_"
STATE = ROOT / "outputs" / "hybrid_state" / "anomod"
MODELS = ROOT / "models" / "hybrid" / "anomod"
NAMES = list(ANOMOD_HYBRIDS)
v2_data.common.LOG = ROOT / "outputs" / "hybrid_anomod_dataload_log.txt"      # keep the data layer's warnings out of the earlier logs

RESOURCES = {"cpu": ("cpu",), "memory": ("memory", "mem", "resident"), "network": ("network", "net_"), "disk": ("disk", "filesystem", "volume"), "load": ("load",)}


def node_of(name):
    if name.startswith("api_"):
        return "api"
    if name.startswith("log_"):
        return "logs"
    if name.startswith("was_missing") or name == "system_is_TT":
        return "flags"
    n = name[2:] if name.startswith("m_") else name
    for pre, node in (("socialnet_container", "socialnet_containers"), ("system_", "host_system"), ("container_", "containers"), ("node_", "node_exporter"),
                      ("rate(node", "node_exporter"), ("process_", "process"), ("http_", "http_server")):
        if n.startswith(pre):
            return node
    return "other_services"


def build_graph(output_names):
    """nodes = components of the monitored system, taken from the metric feature names (their source prefix: host system, containers, node exporter, process, http server, ...)
    plus the API and log groups; an edge joins two nodes that report on the same resource (cpu, memory, network, disk, load) or are the API / log view of the system
    (api-logs, api-host_system, logs-host_system). was_missing and system flags form one 'flags' node linked to every node."""
    nodes = {}
    for i, n in enumerate(output_names):
        nodes.setdefault(node_of(n), []).append(i)
    order = list(nodes)
    res = {k: set() for k in order}
    for k, idx in nodes.items():
        for i in idx:
            for r, keys in RESOURCES.items():
                if any(s in output_names[i] for s in keys):
                    res[k].add(r)
    N = len(order)
    adj = np.zeros((N, N), dtype=int)
    why = {}
    for a in range(N):
        for b in range(a + 1, N):
            ka, kb = order[a], order[b]
            shared = res[ka] & res[kb]
            reason = None
            if "flags" in (ka, kb):
                reason = "flags are linked to every node"
            elif {ka, kb} in ({"api", "logs"}, {"api", "host_system"}, {"logs", "host_system"}):
                reason = "api/log view of the system"
            elif shared and "api" not in (ka, kb) and "logs" not in (ka, kb):
                reason = "same resource: " + ", ".join(sorted(shared))
            if reason:
                adj[a, b] = adj[b, a] = 1
                why[(ka, kb)] = reason
    return order, [nodes[k] for k in order], adj, why


class AnoModData(DataV2):
    ds = "anomod"

    def __init__(self):
        super().__init__()
        self.class_names = CLASS_NAMES
        self.make_folds5()
        names = self.gprep.output_names
        self.node_names, self.node_idx, self.adj, self.edge_why = build_graph(names)
        self.binary_cols = [i for i, n in enumerate(names) if n.startswith("was_missing") or n == "system_is_TT"]
        self.runid_cols = [i for i, n in enumerate(names) if not (n.startswith("was_missing") or n == "system_is_TT")]
        self.n_feat_out = len(names)

    def make_folds5(self):
        df, tr = self.df, self.train_pos
        p = REP / f"{PREFIX}folds5.csv"
        sub = df.iloc[tr]
        if p.exists():
            fa = pd.read_csv(p).set_index("window_id").fold
            fold = fa.reindex(sub.window_id).to_numpy(int)
        else:
            fold = None
            for seed in range(42, 142):
                sg = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
                f = np.zeros(len(sub), int)
                for k, (_, va) in enumerate(sg.split(np.zeros(len(sub)), self.y["label_level"][tr], sub.block_id.to_numpy())):
                    f[va] = k + 1
                ok = all(set(np.unique(self.y["label_level"][tr][f == k])) == set(range(5)) for k in range(1, 6))
                if ok:
                    fold = f
                    break
            if fold is None:
                fold = f
            pd.DataFrame(dict(window_id=sub.window_id.to_numpy(), fold=fold, group_type="block_id", system=sub.system.to_numpy(), run_id=sub.run_id.to_numpy(), block_id=sub.block_id.to_numpy())).to_csv(p, index=False)
        self.folds = [(tr[fold != k], tr[fold == k]) for k in range(1, 6)]
        self.fold5 = fold

    def spec(self, label):
        return dict(T=T, F=self.n_feat_out, C=len(self.class_names[label]), node_idx=self.node_idx, adj=self.adj)

    def drop_W(self, k, keep=False):
        if not keep:
            self._w.pop(k, None)

    def group_of(self, pos):
        return self.df.block_id.to_numpy()[pos]


def write_graph_doc(D):
    rows = [dict(node=n, n_features=len(i), example_features="; ".join(D.gprep.output_names[j] for j in i[:3])) for n, i in zip(D.node_names, D.node_idx)]
    pd.DataFrame(rows).to_csv(REP / f"{PREFIX}graph_nodes.csv", index=False)
    pd.DataFrame([dict(node_a=a, node_b=b, reason=w) for (a, b), w in D.edge_why.items()]).to_csv(REP / f"{PREFIX}graph_edges.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(NAMES))
    ap.add_argument("--labels", default=",".join(E.LABELS))
    ap.add_argument("--until", default="", help="HH:MM local time after which no new model is started")
    ap.add_argument("--assemble-only", action="store_true")
    a = ap.parse_args()
    D = AnoModData()
    E.log(f"AnoMod data: train {len(D.train_pos)}, test {len(D.test_pos)}, features {D.n_feat_out}, 5 folds {[len(v) for _, v in D.folds]}, graph nodes {D.node_names}")
    write_graph_doc(D)
    # fold integrity: no block (group) in two folds, every class in every validation fold where possible
    blocks = D.group_of(D.train_pos)
    assert all(len(set(blocks[D.fold5 == k]) & set(blocks[D.fold5 != k])) == 0 for k in range(1, 6))
    dead = None
    if a.until:
        h, m = map(int, a.until.split(":"))
        now = dt.datetime.now()
        t = now.replace(hour=h, minute=m, second=0, microsecond=0)
        dead = (t + dt.timedelta(days=1) if t < now - dt.timedelta(hours=1) else t).timestamp()
    if not a.assemble_only:
        E.imbalance_report(D, PREFIX)
        R = E.Runner(D, ANOMOD_HYBRIDS, STATE, MODELS, PREFIX, deadline=dead)
        for name in a.models.split(","):
            for label in a.labels.split(","):
                R.run(name, label)
                E.assemble(D, NAMES, PREFIX, STATE, runid=None, n_boot=200)     # quick interim csv files after every model
    runid = E.run_id_rule(D, None)
    lead = E.assemble(D, NAMES, PREFIX, STATE, runid=runid)
    E.log("assembled:\n" + lead.to_string())


if __name__ == "__main__":
    main()
