"""Part B: one combined SN + TT modelling dataset on top of the Part A split. No model is trained.

combined_windows.parquet : union of all feature columns (a feature a system lacks stays NaN for it) + system, run, labels, split.
Preprocessing (copies, fitted on the combined TRAIN only): add system_is_TT; impute NaN with the training median of that
system; a feature that is entirely absent for a system (all-NaN in that system's training rows) becomes exactly 0 after scaling;
standard scaling uses only the rows of systems where the feature exists.
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.model_selection import StratifiedKFold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

SEED, NSPLIT, MIN_CLASS = 42, 10, 10
CD = os.path.join(PROC, "combined")
os.makedirs(CD, exist_ok=True)
PRES = []
FRONT = ["window_id", "system", "run", "scenario", "label_binary", "label_level", "split", "window_index", "window_start", "window_end"]


def pres(step, obj, rin, cin, rout, cout, note=""):
    PRES.append(dict(step=step, system="SN+TT", object=obj, rows_in=rin, cols_in=cin, rows_out=rout, cols_out=cout,
                     rows_lost=(rin - rout) if rin is not None and rout is not None else None, note=note))


def labels(df, version):
    return (df.label_binary.map({0: "Normal", 1: "Anomaly"}) if version == "binary" else df.label_level).astype(str).to_numpy()


def classes_of(version):
    return ["Normal", "Anomaly"] if version == "binary" else LEVELS


def counts(y, classes):
    s = pd.Series(y).value_counts()
    return {c: int(s.get(c, 0)) for c in classes}


class CombPrep:
    def fit(self, X, is_tt):
        n, f = X.shape
        self.med = np.full((2, f), np.nan)
        for s in (0, 1):
            r = is_tt == s
            if r.any():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.med[s] = np.nanmedian(X[r], axis=0)
        self.absent = np.isnan(self.med)                    # [system, feature]: feature entirely absent in that system's training rows
        Xi = self._impute(X, is_tt)
        present = ~self.absent[is_tt]                        # n x f
        cnt = present.sum(0)
        w = np.where(present, Xi, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.mean = np.where(cnt > 0, w.sum(0) / cnt, 0.0)
            var = np.where(cnt > 0, (np.where(present, (Xi - self.mean) ** 2, 0.0)).sum(0) / cnt, 1.0)
        self.std = np.where(var > 0, np.sqrt(var), 1.0)
        return self

    def _impute(self, X, is_tt):
        Xi = X.copy()
        r, c = np.where(np.isnan(Xi))
        m = self.med[is_tt[r], c]
        Xi[r, c] = m                                         # stays NaN where the feature is absent for that system
        return Xi

    def transform(self, X, is_tt):
        Xi = self._impute(X, is_tt)
        Z = (Xi - self.mean) / self.std
        Z[self.absent[is_tt]] = 0.0                          # absent for that system -> 0 after scaling
        assert not np.isnan(Z).any()
        return Z


def sampler(kind, X, y, minc):
    k = min(5, minc - 1)
    smp = SMOTE(random_state=SEED, k_neighbors=k) if kind == "smote" else ADASYN(random_state=SEED, n_neighbors=k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smp.fit_resample(X, y)


def main():
    sn = pd.read_parquet(os.path.join(PROC, "SN", "windows.parquet"))
    tt = pd.read_parquet(os.path.join(PROC, "TT", "windows.parquet"))
    tr_ids = set(pd.read_parquet(os.path.join(PROC, "SN", "train_raw.parquet"), columns=["window_id"]).window_id) | \
        set(pd.read_parquet(os.path.join(PROC, "TT", "train_raw.parquet"), columns=["window_id"]).window_id)
    te_ids = set(pd.read_parquet(os.path.join(PROC, "SN", "test_raw.parquet"), columns=["window_id"]).window_id) | \
        set(pd.read_parquet(os.path.join(PROC, "TT", "test_raw.parquet"), columns=["window_id"]).window_id)

    # ---------------------------------------------------------------- 1. combined windows
    comb = pd.concat([sn, tt], ignore_index=True, sort=False)
    comb["run"] = comb["scenario"]
    comb["split"] = np.where(comb.window_id.isin(tr_ids), "train", "test")
    assert len(comb) == len(sn) + len(tt) and comb.window_id.is_unique
    assert set(comb.window_id) == tr_ids | te_ids and not (tr_ids & te_ids), "split must be the union of the SN and TT splits"
    feats = [c for c in comb.columns if c.startswith(("api_", "m_", "log_"))]
    comb = comb[FRONT + feats]
    comb.to_parquet(os.path.join(CD, "combined_windows.parquet"), index=False)
    sn_f, tt_f = {c for c in sn.columns if c.startswith(("api_", "m_", "log_"))}, {c for c in tt.columns if c.startswith(("api_", "m_", "log_"))}
    fl = pd.DataFrame([dict(feature=f, group=("api" if f.startswith("api_") else "metrics" if f.startswith("m_") else "logs"),
                            present_in_SN=f in sn_f, present_in_TT=f in tt_f) for f in feats])
    fl.to_csv(os.path.join(OUT, "combined_feature_list.csv"), index=False)
    pres("concat SN + TT windows", "SN windows + TT windows -> combined_windows.parquet", len(sn) + len(tt), sn.shape[1] + tt.shape[1], len(comb), comb.shape[1],
         f"union of columns: {len(feats)} feature columns (SN {len(sn_f)}, TT {len(tt_f)}, both {len(sn_f & tt_f)}); every window kept")
    log(f"combined windows: {len(comb)} (SN {len(sn)} + TT {len(tt)}), {comb.shape[1]} columns, {len(feats)} feature columns")
    log(f"features present in: both systems {len(sn_f & tt_f)}, SN only {len(sn_f - tt_f)}, TT only {len(tt_f - sn_f)}")
    log("   by group (both / SN only / TT only): " + str({g: (int(((x.present_in_SN) & (x.present_in_TT)).sum()), int((x.present_in_SN & ~x.present_in_TT).sum()),
                                                            int((~x.present_in_SN & x.present_in_TT).sum())) for g, x in fl.groupby("group")}))

    # ---------------------------------------------------------------- 2. split summary
    train, test = comb[comb.split == "train"], comb[comb.split == "test"]
    pres("split union", "SN split + TT split -> combined split column", len(comb), comb.shape[1], len(train) + len(test), comb.shape[1], f"train {len(train)} + test {len(test)}")
    log(f"\ncombined train {len(train)} ({100 * len(train) / len(comb):.2f}%)   test {len(test)} ({100 * len(test) / len(comb):.2f}%)   [target train 79-81%]")
    rows = []
    for version in ("binary", "level"):
        cl = classes_of(version)
        for name, sub in (("ALL", None), ("SN", "SN"), ("TT", "TT")):
            ctr = counts(labels(train if sub is None else train[train.system == sub], version), cl)
            cte = counts(labels(test if sub is None else test[test.system == sub], version), cl)
            for c in cl:
                rows.append(dict(label_version=version, system=name, **{"class": c}, train=ctr[c], test=cte[c]))
            log(f"[{version}/{name}] train {ctr}  test {cte}")
    sc = pd.DataFrame(rows)
    sc["train_pct_of_split"] = (100 * sc.train / sc.groupby(["label_version", "system"]).train.transform("sum")).round(2)
    sc["test_pct_of_split"] = (100 * sc.test / sc.groupby(["label_version", "system"]).test.transform("sum")).round(2)
    sc.to_csv(os.path.join(OUT, "combined_split_counts.csv"), index=False)
    log("per-system windows within each split: " + str({s: dict(train=int((train.system == s).sum()), test=int((test.system == s).sum())) for s in SYSTEMS}))
    log(f"per-system train share: " + str({s: round(100 * (train.system == s).sum() / (comb.system == s).sum(), 2) for s in SYSTEMS}))

    # ---------------------------------------------------------------- 3. preprocessing (copies)
    def design(df):
        X = df[feats].to_numpy(float)
        tt_flag = (df.system == "TT").to_numpy().astype(int)
        return np.hstack([X, tt_flag[:, None].astype(float)]), tt_flag
    cols = feats + ["system_is_TT"]
    Xtr, ftr = design(train)
    Xte, fte = design(test)
    prep = CombPrep().fit(Xtr, ftr)
    Xtr_s, Xte_s = prep.transform(Xtr, ftr), prep.transform(Xte, fte)
    pres("impute+scale (copy)", "combined train -> combined_train_model", len(train), Xtr.shape[1], len(Xtr_s), Xtr_s.shape[1],
         f"+1 column system_is_TT; NaN cells before {int(np.isnan(Xtr).sum())}, after {int(np.isnan(Xtr_s).sum())}")
    pres("impute+scale (copy)", "combined test -> combined_test_model", len(test), Xte.shape[1], len(Xte_s), Xte_s.shape[1],
         f"fitted on train only; NaN cells before {int(np.isnan(Xte).sum())}, after {int(np.isnan(Xte_s).sum())}")
    for name, X, src in (("combined_train_model", Xtr_s, train), ("combined_test_model", Xte_s, test)):
        pd.concat([src[FRONT].reset_index(drop=True), pd.DataFrame(X, columns=cols)], axis=1).to_parquet(os.path.join(CD, f"{name}.parquet"), index=False)
    json.dump(dict(columns=cols, median_SN=np.where(np.isnan(prep.med[0]), None, prep.med[0]).tolist(), median_TT=np.where(np.isnan(prep.med[1]), None, prep.med[1]).tolist(),
                   absent_in_SN_train=[c for c, a in zip(cols, prep.absent[0]) if a], absent_in_TT_train=[c for c, a in zip(cols, prep.absent[1]) if a],
                   scaler_mean=prep.mean.tolist(), scaler_std=prep.std.tolist()), open(os.path.join(CD, "combined_preprocessing.json"), "w"))
    log(f"features absent (all-NaN) in SN training rows: {int(prep.absent[0].sum())}, in TT training rows: {int(prep.absent[1].sum())} -> set to 0 after scaling for that system")

    # ---------------------------------------------------------------- 4. 10-fold CV on combined train, stratified on system + level
    strat = (train.system + "|" + train.label_level).to_numpy()
    for version in ("binary", "level"):
        small = {c: n for c, n in counts(labels(train, version), classes_of(version)).items() if n < MIN_CLASS}
        if small:
            log(f"STOP: {version}: class(es) with fewer than {MIN_CLASS} training samples: {small}")
            sys.exit(1)
    folds = {}
    fold_rows, adasyn_err = [], []
    skf = StratifiedKFold(n_splits=NSPLIT, shuffle=True, random_state=SEED)
    seen = np.zeros(len(train), int)
    ylev, ybin = labels(train, "level"), labels(train, "binary")
    for k, (itr, iva) in enumerate(skf.split(Xtr, strat), 1):
        folds[str(k)] = dict(train_idx=itr.tolist(), val_idx=iva.tolist())
        seen[iva] += 1
        p = CombPrep().fit(Xtr[itr], ftr[itr])
        Xf = p.transform(Xtr[itr], ftr[itr]); Xv = p.transform(Xtr[iva], ftr[iva])
        for version, yall in (("binary", ybin), ("level", ylev)):
            cl = classes_of(version)
            yf, yv = yall[itr], yall[iva]
            cf, cv = counts(yf, cl), counts(yv, cl)
            minc = min(cf.values())
            if minc < MIN_CLASS:
                log(f"STOP: {version} fold {k}: class with fewer than {MIN_CLASS} training samples: {cf}")
                sys.exit(1)
            _, ys = sampler("smote", Xf, yf, minc)
            cs = counts(ys, cl)
            try:
                _, ya = sampler("adasyn", Xf, yf, minc)
                ca, ast = counts(ya, cl), "ok"
            except Exception as e:
                ca, ast = {c: None for c in cl}, f"FAILED: {type(e).__name__}: {e}"
                adasyn_err.append(dict(label_version=version, fold=k, reason=ast))
            for c in cl:
                fold_rows.append(dict(label_version=version, fold=k, **{"class": c}, train_before=cf[c], train_after_smote=cs[c],
                                      train_after_adasyn=ca[c], validation=cv[c], adasyn_status=ast))
    assert (seen == 1).all()
    pres("10-fold CV", "folds on combined train (stratified on system + level)", len(train), Xtr.shape[1], int(seen.sum()), Xtr.shape[1],
         "each training row in exactly one validation fold; validation never resampled; preprocessing refitted per fold")
    pd.DataFrame(fold_rows).to_csv(os.path.join(OUT, "combined_fold_counts.csv"), index=False)
    note = "indices are row positions among split=='train' rows of combined_windows.parquet (in file order); StratifiedKFold(10, shuffle=True, random_state=42) stratified on system+label_level"
    json.dump(dict(note=note, folds=folds), open(os.path.join(CD, "fold_indices.json"), "w"))
    json.dump(dict(note=note, folds=folds), open(os.path.join(OUT, "combined_fold_indices.json"), "w"))

    # ---------------------------------------------------------------- final SMOTE on the full combined train; ADASYN counts only
    tt_col = cols.index("system_is_TT")
    rb, mix_rows = [], {}
    for version, yall in (("binary", ybin), ("level", ylev)):
        cl = classes_of(version)
        ctr = counts(yall, cl)
        minc = min(ctr.values())
        row = {c: dict(label_version=version, **{"class": c}, train_raw=ctr[c]) for c in cl}
        for kind in ("smote", "adasyn"):
            try:
                Xr, yr = sampler(kind, Xtr_s, yall, minc)
                n0 = len(Xtr_s)
                assert np.allclose(Xr[:n0], Xtr_s) and (yr[:n0] == yall).all()
                if kind == "smote":
                    out = pd.DataFrame(Xr, columns=cols)
                    out.insert(0, "label", yr)
                    out.insert(0, "is_synthetic", np.arange(len(out)) >= n0)
                    out.insert(0, "window_id", list(train.window_id) + [f"synthetic_smote_{i}" for i in range(len(out) - n0)])
                    out.to_parquet(os.path.join(CD, f"train_smote_{version}.parquet"), index=False)
                    flag = Xr[n0:, tt_col] * prep.std[tt_col] + prep.mean[tt_col]           # back to the raw 0/1 indicator
                    mix_rows[version] = int((np.minimum(np.abs(flag), np.abs(flag - 1)) > 1e-6).sum()), len(flag)
                cb = counts(yr, cl)
                pres(f"{kind.upper()} on full combined train", f"{version}: combined_train_model -> " + (f"train_smote_{version}" if kind == "smote" else "counts only"),
                     n0, len(cols), len(Xr), len(cols), f"{len(Xr) - n0} synthetic rows added; all {n0} real rows unchanged (verified)" + ("" if kind == "smote" else "; counts only, not saved"))
                for c in cl:
                    row[c][f"train_{kind}"] = cb[c]
            except Exception as e:
                adasyn_err.append(dict(label_version=version, fold="final", reason=f"{kind}: {type(e).__name__}: {e}"))
                for c in cl:
                    row[c][f"train_{kind}"] = None
                log(f"  {kind} FINAL FAILED for {version}: {e}")
        rb.extend(row.values())
    rbd = pd.DataFrame(rb)
    rbd.to_csv(os.path.join(OUT, "combined_train_raw_vs_balanced.csv"), index=False)
    pd.DataFrame(adasyn_err, columns=["label_version", "fold", "reason"]).to_csv(os.path.join(OUT, "combined_sampler_errors.csv"), index=False)

    # ---------------------------------------------------------------- sample rows + plots
    rng = np.random.default_rng(SEED)
    samp = []
    for version in ("binary", "level"):
        allv = labels(comb, version)
        for c in classes_of(version):
            idx = np.flatnonzero(allv == c)
            s = comb.iloc[rng.choice(idx, size=min(2, len(idx)), replace=False)].copy()
            s.insert(0, "class", c); s.insert(0, "label_version", version)
            samp.append(s)
    pd.concat(samp, ignore_index=True).to_csv(os.path.join(OUT, "combined_sample_rows.csv"), index=False)
    for version in ("binary", "level"):
        cl = classes_of(version); x = np.arange(len(cl)); w = 0.38
        ctr, cte = counts(labels(train, version), cl), counts(labels(test, version), cl)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for a, logy in zip(ax, (False, True)):
            b1 = a.bar(x - w / 2, [ctr[c] for c in cl], w, label="train"); b2 = a.bar(x + w / 2, [cte[c] for c in cl], w, label="test")
            for b in list(b1) + list(b2):
                a.text(b.get_x() + b.get_width() / 2, b.get_height(), int(b.get_height()), ha="center", va="bottom", fontsize=7)
            a.set_xticks(x); a.set_xticklabels(cl, rotation=20); a.set_ylabel("windows")
            a.set_title(f"combined SN+TT / {version}: " + ("log scale" if logy else "linear scale"))
            if logy:
                a.set_yscale("log")
            a.legend()
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"combined_class_distribution_{version}.png"), dpi=130); plt.close(fig)
        fig, a = plt.subplots(figsize=(7, 4)); w = 0.38
        sub = rbd[rbd.label_version == version].set_index("class")
        for j, (key, lab) in enumerate((("train_raw", "raw"), ("train_smote", "SMOTE"))):
            vals = [sub.loc[c, key] for c in cl]
            bars = a.bar(x - w / 2 + j * w, vals, w, label=lab)
            for b, v in zip(bars, vals):
                a.text(b.get_x() + b.get_width() / 2, v, int(v), ha="center", va="bottom", fontsize=7)
        a.set_yscale("log"); a.set_xticks(x); a.set_xticklabels(cl, rotation=20); a.set_ylabel("windows")
        a.set_title(f"combined SN+TT / {version}: training class counts (log scale)"); a.legend()
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"combined_raw_vs_smote_{version}.png"), dpi=130); plt.close(fig)

    # ---------------------------------------------------------------- preservation log (append to the Part A rows)
    old = pd.read_csv(os.path.join(OUT, "data_preservation_log.csv"))
    old = old[old.system != "SN+TT"]
    pd.concat([old, pd.DataFrame(PRES)], ignore_index=True).to_csv(os.path.join(OUT, "data_preservation_log.csv"), index=False)

    # ---------------------------------------------------------------- final print
    log("\n================ PART B KEY COUNTS ================")
    log("\ncombined_split_counts (ALL):\n" + sc[sc.system == "ALL"].to_string(index=False))
    log("\ncombined_train_raw_vs_balanced:\n" + rbd.to_string(index=False))
    log("\nsampler errors:", "none" if not adasyn_err else "\n" + pd.DataFrame(adasyn_err).to_string(index=False))
    log("\nWARNINGS")
    log(f"  - {int((~fl.present_in_SN | ~fl.present_in_TT).sum())} of {len(fl)} feature columns exist in only one system; the other system gets 0 after scaling (see combined_feature_list.csv)")
    log(f"  - SMOTE interpolates between neighbours without knowing the system: synthetic rows whose system_is_TT flag is not exactly 0/1 "
        f"(i.e. built from an SN and a TT neighbour): {mix_rows}")
    log(f"  - system_is_TT is a feature: a model can identify the system (and therefore much of the class mix) from it")
    log("  - feature names do not align between systems for metrics (different monitoring stacks): only api_* and log_* features are shared")
    log(f"  - fold sizes: train rows per fold ~{len(Xtr) - len(Xtr) // NSPLIT}, validation ~{len(Xtr) // NSPLIT}")


if __name__ == "__main__":
    main()
