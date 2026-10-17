"""Leakage checks on the train/test split -> report_outputs/leakage_checks.csv. Every number is computed and printed."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocess as P

ROWS = []
THR = 0.90


def add(check, scope, item, value, threshold="", flag=False, detail=""):
    ROWS.append(dict(check=check, scope=scope, item=item, value=value, threshold=threshold, flag=bool(flag), detail=detail))
    if check != "single_feature_bal_acc" or flag:
        print(f"[{check}] {scope} | {item}: {value}" + (f"   <-- FLAG {detail}" if flag else ""), flush=True)


def main():
    df = P.load_windows()
    sp = pd.read_csv(P.REPORTS / "split_ids.csv")
    d = df.merge(sp[["window_id", "split"]], on="window_id")
    tr, te = d[d.split == "train"].reset_index(drop=True), d[d.split == "test"].reset_index(drop=True)
    prep = P.load_preprocessor()
    feats = prep.raw_cols
    print(f"train {len(tr)} test {len(te)}; raw features {len(feats)}; model inputs {prep.n_out}")

    # ---- 1. exact duplicates across train / test
    cols = feats + ["system_is_TT"]
    htr = pd.util.hash_pandas_object(tr[cols], index=False).to_numpy()
    hte = pd.util.hash_pandas_object(te[cols], index=False).to_numpy()
    inter = set(htr) & set(hte)
    te_dup, tr_dup = np.isin(hte, list(inter)), np.isin(htr, list(inter))
    add("duplicates", "raw features+system", "test rows identical to a train row", int(te_dup.sum()), "0", te_dup.sum() > 0,
        f"of {len(te)} test rows; {len(inter)} distinct duplicated vectors; train rows involved {int(tr_dup.sum())}")
    lab_conf = 0
    if inter:
        m = tr.assign(h=htr).groupby("h").label_level.agg(set)
        lab_conf = sum(lv not in m[h] for h, lv in zip(hte[te_dup], te.label_level[te_dup]))
    add("duplicates", "raw features+system", "duplicated test rows whose train twin has a different label_level", int(lab_conf), "0", lab_conf > 0)
    add("duplicates", "raw features+system", "duplicate rows inside train", int(pd.Series(htr).duplicated().sum()), "info")
    for s in P.SYSTEMS:
        mm = (te.system == s).to_numpy()
        add("duplicates", s, "test rows identical to a train row (raw)", int(te_dup[mm].sum()), "0", te_dup[mm].sum() > 0, f"of {int(mm.sum())}")
    Ztr, Zte = prep.transform(tr), prep.transform(te)
    ztr_h = pd.util.hash_pandas_object(pd.DataFrame(Ztr), index=False).to_numpy()
    zte_h = pd.util.hash_pandas_object(pd.DataFrame(Zte), index=False).to_numpy()
    zd = np.isin(zte_h, ztr_h)
    add("duplicates", "preprocessed vectors", "test rows identical to a train row after preprocessing", int(zd.sum()), "0", zd.sum() > 0,
        "different raw windows can collapse to one vector after clipping/imputation")

    # ---- 2. single-feature balanced accuracy (depth-4 tree on one preprocessed column, fit on train, scored on test)
    names = prep.output_names
    n_flag = 0
    for lab in ("label_binary", "label_level"):
        ytr, yte = tr[lab].astype(str), te[lab].astype(str)
        best = ("", 0.0)
        for j, n in enumerate(names):
            if np.ptp(Ztr[:, j]) == 0:
                add("single_feature_bal_acc", lab, n, 0.5, f"<={THR}", False, "constant column in train")
                continue
            t = DecisionTreeClassifier(max_depth=4, random_state=P.SEED).fit(Ztr[:, [j]], ytr)
            ba = balanced_accuracy_score(yte, t.predict(Zte[:, [j]]))
            fl = ba > THR
            n_flag += int(fl)
            if ba > best[1]:
                best = (n, ba)
            add("single_feature_bal_acc", lab, n, round(ba, 4), f"<={THR}", fl, "single feature separates the classes on test")
        add("single_feature_bal_acc_max", lab, best[0], round(best[1], 4), f"<={THR}", best[1] > THR)

    # ---- 3. label / scenario columns not among the model inputs
    scen = {s.lower() for s in set(d.scenario) | set(d.run_id)}
    bad = [n for n in names if any(k in n.lower() for k in P.FORBIDDEN_NAMES) or any(s in n.lower() for s in scen)]
    add("forbidden_columns", "model input names", "inputs whose name contains label/scenario/run/split/id/block/class/fold/target or a scenario name", len(bad), "0", len(bad) > 0, str(bad))
    meta_in = [c for c in P.META if c in names and c != "system_is_TT"]
    add("forbidden_columns", "model input names", "metadata columns among inputs (ids, labels, window_index, timestamps)", len(meta_in), "0", len(meta_in) > 0, str(meta_in))
    add("forbidden_columns", "model input names", "system_is_TT among inputs (intended, see README: a model can identify the system from it)", int("system_is_TT" in names), "info")
    add("forbidden_columns", "model input names", "model inputs / raw features / dropped constant", f"{len(names)} / {len(feats)} / {len(prep.const_cols)}")
    Zall = prep.transform(d)
    lv = d.label_level.astype("category").cat.codes.to_numpy(float)
    varying = [j for j in range(len(names)) if np.ptp(Zall[:, j]) > 0]
    eq = [names[j] for j in varying if abs(np.corrcoef(Zall[:, j], d.label_binary)[0, 1]) > 0.999 or abs(np.corrcoef(Zall[:, j], lv)[0, 1]) > 0.999]
    add("forbidden_columns", "model inputs", "inputs with |corr| > 0.999 to label_binary or label_level", len(eq), "0", len(eq) > 0, str(eq))
    wi = d.window_index.to_numpy(float)
    top = sorted(((names[j], round(abs(np.corrcoef(Zall[:, j], wi)[0, 1]), 3)) for j in varying), key=lambda t: -t[1])[:3]
    add("forbidden_columns", "model inputs", "max |corr| of any input with window_index (time position in the run)", top[0][1], "<=0.9", top[0][1] > 0.9, str(top))

    # ---- 4. time overlap train vs test
    for s in P.SYSTEMS:
        t_, e_ = tr[tr.system == s], te[te.system == s]
        ts, tend = t_.window_start.to_numpy(), t_.window_end.to_numpy()
        es, eend = e_.window_start.to_numpy(), e_.window_end.to_numpy()
        ov = (ts[:, None] < eend[None]) & (tend[:, None] > es[None])
        same = t_.run_id.to_numpy()[:, None] == e_.run_id.to_numpy()[None]
        add("time_overlap", s, "train windows overlapping a test window (same system, any run)", int(ov.any(1).sum()), "0", ov.any(1).sum() > 0, f"of {len(t_)}")
        add("time_overlap", s, "train windows overlapping a test window of the SAME run", int((ov & same).any(1).sum()), "0", (ov & same).any(1).sum() > 0, f"of {len(t_)}")
        add("time_overlap", s, "test windows overlapping a train window (any run)", int(ov.any(0).sum()), "0", ov.any(0).sum() > 0, f"of {len(e_)}")
        add("time_overlap", s, "train/test window pairs from DIFFERENT runs overlapping in time", int((ov & ~same).sum()), "0", (ov & ~same).any())
        sec = (np.minimum(tend[:, None], eend[None]) - np.maximum(ts[:, None], es[None])) / np.timedelta64(1, "s")
        add("time_overlap", s, "max overlap in seconds (window 30 s, step 10 s)", float(np.where(ov, sec, 0).max()), "0", ov.any())
        ds = d[d.system == s].sort_values(["run_id", "window_index"])
        nb = int(sum((g.to_numpy()[1:] != g.to_numpy()[:-1]).sum() for _, g in ds.groupby("run_id").split))
        add("time_overlap", s, "train/test block boundaries inside runs (adjacent windows share 20 s there)", nb, "info")

    # ---- 5. run / block overlap
    rt = set(tr.run_id) & set(te.run_id)
    add("run_overlap", "ALL", "runs present in both train and test", len(rt), "0", len(rt) > 0, f"of {d.run_id.nunique()} runs (block split inside every run, by design)")
    for s in P.SYSTEMS:
        add("run_overlap", s, "runs in both train and test", len(set(tr[tr.system == s].run_id) & set(te[te.system == s].run_id)), "0", True, f"of {d[d.system == s].run_id.nunique()} runs")
    bo = set(tr.block_id) & set(te.block_id)
    add("run_overlap", "ALL", "10-window blocks present in both train and test", len(bo), "0", len(bo) > 0)
    add("run_overlap", "ALL", "runs per class (label_level)", str(d.groupby("label_level").run_id.nunique().to_dict()), "info")

    out = pd.DataFrame(ROWS)
    P.REPORTS.mkdir(exist_ok=True)
    out.to_csv(P.REPORTS / "leakage_checks.csv", index=False)
    print(f"\nrows in leakage_checks.csv: {len(out)}; flagged: {int(out.flag.sum())}; single-feature flags (> {THR}): {n_flag}")


if __name__ == "__main__":
    main()
