"""Part A (steps 2-7): API-aware block split (blocks of 10 windows, see split_blocks.py), imputed+scaled copies, 10-fold CV with SMOTE / ADASYN (train part of each fold only),
final balanced training sets, reports. No model is trained. Raw folders are only read.
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa
from split_blocks import block_split

SEED, NSPLIT, TRAIN_FRAC, MIN_CLASS, BLOCK = 42, 10, 0.8, 10, 10
SPLITINFO = {}
SPLITMETA = {}
META = ["window_id", "system", "scenario", "label_binary", "label_level", "window_index", "window_start", "window_end"]
PRES = []      # data preservation log rows
FOLDJSON = {}  # fold_indices.json


def pres(step, system, obj, rin, cin, rout, cout, note=""):
    PRES.append(dict(step=step, system=system, object=obj, rows_in=rin, cols_in=cin, rows_out=rout, cols_out=cout,
                     rows_lost=(rin - rout) if rin is not None and rout is not None else None, note=note))


def label_series(df, version):
    if version == "binary":
        return df.label_binary.map({0: "Normal", 1: "Anomaly"}).astype(str)
    return df.label_level.astype(str)


def class_order(version):
    return ["Normal", "Anomaly"] if version == "binary" else LEVELS


class Prep:
    """median imputation (all-NaN column -> 0) + standard scaling, fitted on the data given to fit()."""
    def fit(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            med = np.nanmedian(X, axis=0)
        self.all_nan = np.isnan(med)
        self.med = np.where(self.all_nan, 0.0, med)
        self.sc = StandardScaler().fit(self.impute(X))
        return self

    def impute(self, X):
        X = X.copy()
        r, c = np.where(np.isnan(X))
        X[r, c] = self.med[c]
        return X

    def transform(self, X):
        return self.sc.transform(self.impute(X))


def counts(y, classes):
    s = pd.Series(y).value_counts()
    return {c: int(s.get(c, 0)) for c in classes}


def run_sampler(kind, X, y, minc):
    k = min(5, minc - 1)
    if kind == "smote":
        smp = SMOTE(random_state=SEED, k_neighbors=k)
    else:
        smp = ADASYN(random_state=SEED, n_neighbors=k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smp.fit_resample(X, y)


def main():
    snap_before = json.load(open(os.path.join(PROC, "raw_snapshot_before.json")))
    rep = dict(split=[], raw_vs_bal=[], fold=[], samples=[], warn=[], adasyn_err=[])
    figs = []
    for system in SYSTEMS:
        d = os.path.join(PROC, system)
        W = pd.read_parquet(os.path.join(d, "windows.parquet"))
        feats = [c for c in W.columns if c.startswith(("api_", "m_", "log_"))]
        log(f"\n================ {system}: {len(W)} windows, {len(feats)} features ================")
        # ---- warnings (never dropped)
        for c in feats:
            if W[c].isna().all():
                rep["warn"].append(dict(system=system, column=c, issue="all-NaN (kept)"))
            elif W[c].nunique(dropna=True) <= 1:
                rep["warn"].append(dict(system=system, column=c, issue="constant (kept)"))
        # ---- PART A split: API-aware blocks of 10 windows, shuffled per group (API / non-API), see scripts/split_blocks.py
        tr_mask, sinfo, pr = block_split(W)
        pr.insert(0, "system", system)
        log("split info: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in sinfo.items()}))
        SPLITMETA[system] = sinfo
        low = pr[pr.test_blocks < 3]
        if len(low):
            rep["warn"].append(dict(system=system, column="(split)", issue=f"{len(low)} runs have fewer than 3 test blocks (min {int(pr.test_blocks.min())}): "
                               f"3 test blocks in every run would force the train share to at most {100 * sinfo['max_train_share_if_3_test_blocks_per_run']:.2f}% (target 80%, band 79-81%); at least 2 test blocks per run were used instead"))
        pr.to_csv(os.path.join(OUT, f"split_per_run_{system}.csv"), index=False)
        train, test = W[tr_mask].copy(), W[~tr_mask].copy()
        train.to_parquet(os.path.join(d, "train_raw.parquet"), index=False)
        test.to_parquet(os.path.join(d, "test_raw.parquet"), index=False)
        pres("windows -> API-aware block split", system, "windows.parquet -> train_raw + test_raw", len(W), W.shape[1], len(train) + len(test), train.shape[1],
             f"train {len(train)} + test {len(test)}; no window dropped or trimmed")
        log(f"train {len(train)} ({len(train) / len(W):.1%})  test {len(test)} ({len(test) / len(W):.1%})")
        log(f"windows with API requests: train {int((train.api_request_count > 0).sum())} / {len(train)}   test {int((test.api_request_count > 0).sum())} / {len(test)}")
        log("windows per run (train / test):\n" + pr[["scenario", "windows", "train", "test", "train_pct"]].to_string(index=False))
        # train windows whose 30 s span overlaps in time with any test window of the same run (not trimmed, reported only)
        ov = 0
        for sc, g in W.groupby("scenario", sort=False):
            m_ = tr_mask[g.index]
            ts_, te_ = g.window_start.to_numpy()[m_], g.window_end.to_numpy()[m_]
            xs, xe = g.window_start.to_numpy()[~m_], g.window_end.to_numpy()[~m_]
            if len(xs):
                ov += int(((ts_[:, None] < xe[None, :]) & (te_[:, None] > xs[None, :])).any(axis=1).sum())
        log(f"train windows that overlap in time with a test window: {ov} of {len(train)}")
        rep["warn"].append(dict(system=system, column="(split)", issue=f"{ov} train windows overlap in time with a test window of the same run (not trimmed)"))
        SPLITINFO[system] = dict(train=len(train), test=len(test), api_train=int((train.api_request_count > 0).sum()),
                                 api_test=int((test.api_request_count > 0).sum()), overlap=ov)

        # ---- copies for modelling: impute + scale, fitted on the 80% train only
        Xtr, Xte = train[feats].to_numpy(float), test[feats].to_numpy(float)
        prep = Prep().fit(Xtr)
        Xtr_s, Xte_s = prep.transform(Xtr), prep.transform(Xte)
        pres("impute+scale (copy)", system, "train_raw -> train_model", len(train), len(feats), len(Xtr_s), Xtr_s.shape[1], f"NaN cells before: {int(np.isnan(Xtr).sum())}, after: {int(np.isnan(Xtr_s).sum())}")
        pres("impute+scale (copy)", system, "test_raw -> test_model", len(test), len(feats), len(Xte_s), Xte_s.shape[1], f"NaN cells before: {int(np.isnan(Xte).sum())}, after: {int(np.isnan(Xte_s).sum())}")
        for name, X, src in (("train_model", Xtr_s, train), ("test_model", Xte_s, test)):
            out = pd.concat([src[META].reset_index(drop=True), pd.DataFrame(X, columns=feats)], axis=1)
            out.to_parquet(os.path.join(d, f"{name}.parquet"), index=False)
        json.dump(dict(features=feats, median_used=prep.med.tolist(), all_nan_in_train=[f for f, a in zip(feats, prep.all_nan) if a],
                       scaler_mean=prep.sc.mean_.tolist(), scaler_scale=prep.sc.scale_.tolist()),
                  open(os.path.join(d, "preprocessing.json"), "w"), indent=1)

        for version in ("binary", "level"):
            classes = class_order(version)
            ytr, yte = label_series(train, version).to_numpy(), label_series(test, version).to_numpy()
            ctr, cte = counts(ytr, classes), counts(yte, classes)
            for c in classes:
                rep["split"].append(dict(system=system, label_version=version, **{"class": c}, train=ctr[c], test=cte[c],
                                         train_pct=round(100 * ctr[c] / len(ytr), 2), test_pct=round(100 * cte[c] / len(yte), 2)))
            log(f"[{system}/{version}] train {ctr}  test {cte}")
            small = {c: n for c, n in ctr.items() if n < MIN_CLASS}
            if small:
                log(f"STOP: {system}/{version} class(es) with fewer than {MIN_CLASS} training samples: {small}")
                sys.exit(1)

            # ---------------- 10-fold CV on the 80%; resample only the fold's training part
            skf = StratifiedKFold(n_splits=NSPLIT, shuffle=True, random_state=SEED)
            FOLDJSON.setdefault(system, {}).setdefault(version, {})
            val_seen = np.zeros(len(ytr), int)
            for k, (itr, iva) in enumerate(skf.split(Xtr, ytr), 1):
                FOLDJSON[system][version][str(k)] = dict(train_idx=itr.tolist(), val_idx=iva.tolist())
                val_seen[iva] += 1
                p = Prep().fit(Xtr[itr])
                Xf, Xv = p.transform(Xtr[itr]), p.transform(Xtr[iva])
                assert not np.isnan(Xf).any() and not np.isnan(Xv).any()
                yf, yv = ytr[itr], ytr[iva]
                cf, cv = counts(yf, classes), counts(yv, classes)
                minc = min(cf.values())
                if minc < MIN_CLASS:
                    log(f"STOP: {system}/{version} fold {k}: class with fewer than {MIN_CLASS} training samples: {cf}")
                    sys.exit(1)
                _, ys = run_sampler("smote", Xf, yf, minc)
                cs = counts(ys, classes)
                try:
                    _, ya = run_sampler("adasyn", Xf, yf, minc)
                    ca, ast = counts(ya, classes), "ok"
                except Exception as e:  # recorded, run continues
                    ca, ast = {c: None for c in classes}, f"FAILED: {type(e).__name__}: {e}"
                    rep["adasyn_err"].append(dict(system=system, label_version=version, fold=k, reason=ast))
                for c in classes:
                    rep["fold"].append(dict(system=system, label_version=version, fold=k, **{"class": c}, train_before=cf[c],
                                            train_after_smote=cs[c], train_after_adasyn=ca[c], validation=cv[c], adasyn_status=ast))
            assert (val_seen == 1).all(), "every training row must be validated exactly once"
            pres("10-fold CV", system, f"{version}: folds on train_raw", len(ytr), Xtr.shape[1], int(val_seen.sum()),
                 Xtr.shape[1], "each training row is in exactly one validation fold; validation folds never resampled")

            # ---------------- final balanced training sets (fit on the full 80%)
            row = {c: dict(system=system, label_version=version, **{"class": c}, train_raw=ctr[c]) for c in classes}
            minc = min(ctr.values())
            for kind in ("smote", "adasyn"):
                try:
                    Xr, yr = run_sampler(kind, Xtr_s, ytr, minc)
                    n0 = len(Xtr_s)
                    assert np.allclose(Xr[:n0], Xtr_s) and (yr[:n0] == ytr).all(), "real rows must be preserved unchanged"
                    out = pd.DataFrame(Xr, columns=feats)
                    out.insert(0, "label", yr)
                    out.insert(0, "is_synthetic", np.arange(len(out)) >= n0)
                    out.insert(0, "scenario", list(train.scenario) + [None] * (len(out) - n0))
                    out.insert(0, "window_id", list(train.window_id) + [f"synthetic_{kind}_{i}" for i in range(len(out) - n0)])
                    if kind == "smote":   # ADASYN: counts only, no file written
                        out.to_parquet(os.path.join(d, f"train_{kind}_{version}.parquet"), index=False)
                    cb = counts(yr, classes)
                    pres(f"{kind.upper()} on full 80%", system, f"{version}: train_model -> train_{kind}_{version}", len(Xtr_s), len(feats), len(Xr), len(feats),
                         f"{len(Xr) - n0} synthetic rows added; all {n0} real rows unchanged (verified)" + ("" if kind == "smote" else "; counts only, not saved"))
                    for c in classes:
                        row[c][f"train_{kind}"] = cb[c]
                except Exception as e:
                    rep["adasyn_err"].append(dict(system=system, label_version=version, fold="final", reason=f"{kind}: {type(e).__name__}: {e}"))
                    for c in classes:
                        row[c][f"train_{kind}"] = None
                    pres(f"{kind.upper()} on full 80%", system, f"{version}", len(Xtr_s), len(feats), None, None, f"FAILED: {type(e).__name__}: {e}")
                    log(f"  {kind} FINAL FAILED for {system}/{version}: {e}")
            rep["raw_vs_bal"].extend(row.values())
            figs.append((system, version, classes, ctr, cte, row))

            # ---------------- sample rows (2 real windows per class, raw values)
            rng = np.random.default_rng(SEED)
            allv = label_series(W, version).to_numpy()
            for c in classes:
                idx = np.flatnonzero(allv == c)
                pick = rng.choice(idx, size=min(2, len(idx)), replace=False)
                s = W.iloc[pick].copy()
                s.insert(0, "class", c); s.insert(0, "label_version", version)
                rep["samples"].append(s)

    # ------------------------------------------------------------------ write reports
    pd.DataFrame(rep["split"]).to_csv(os.path.join(OUT, "split_counts.csv"), index=False)
    rb = pd.DataFrame(rep["raw_vs_bal"])
    rb.to_csv(os.path.join(OUT, "train_raw_vs_balanced.csv"), index=False)
    fc = pd.DataFrame(rep["fold"])
    fc.to_csv(os.path.join(OUT, "fold_counts.csv"), index=False)
    pd.concat(rep["samples"], ignore_index=True).to_csv(os.path.join(OUT, "sample_rows.csv"), index=False)
    pd.DataFrame(rep["warn"]).to_csv(os.path.join(OUT, "column_warnings.csv"), index=False)
    pd.DataFrame(rep["adasyn_err"], columns=["system", "label_version", "fold", "reason"]).to_csv(os.path.join(OUT, "sampler_errors.csv"), index=False)
    json.dump(dict(note="indices are row positions in processed/<system>/train_raw.parquet; seed 42; StratifiedKFold(10, shuffle=True)",
                   folds=FOLDJSON), open(os.path.join(PROC, "fold_indices.json"), "w"))
    json.dump(dict(note="indices are row positions in processed/<system>/train_raw.parquet; seed 42; StratifiedKFold(10, shuffle=True)",
                   folds=FOLDJSON), open(os.path.join(OUT, "fold_indices.json"), "w"))

    # ---- raw data read-only proof + rows->windows steps from the earlier scripts
    mem = pd.read_csv(os.path.join(OUT, "rows_to_windows.csv"))
    for _, r in mem.iterrows():
        pres("tidy -> windows", r.system, f"{r.datatype}: raw rows -> rows in >=1 window", int(r.rows_with_timestamp), None, int(r.rows_in_at_least_one_window), None,
             f"{int(r.rows_without_timestamp)} rows (log lines before the first timestamped record) have no time and cannot be placed; {int(r.windows_built)} windows built")
    pres("TT latency check", "TT", "api_responses records -> api_latency_* features", 98073, 8, 98073, 8,
         "TT records hold only 'timestamp' (no latency/start/end fields; Date header is 1 s server clock); latency stays NaN, columns kept")
    snap_after = raw_snapshot()
    same = {k: ({kk: vv for kk, vv in v.items()} == {kk: vv for kk, vv in snap_before[k].items()}) for k, v in snap_after.items()}
    for k, ok in same.items():
        pres("raw data read-only check", "-", k, None, None, None, None, "IDENTICAL before/after (file count, bytes, mtime" + (", sha256)" if "csv" in k else ")") if ok else "CHANGED!")
    pd.DataFrame(PRES).to_csv(os.path.join(OUT, "data_preservation_log.csv"), index=False)

    # ------------------------------------------------------------------ plots
    for system, version, classes, ctr, cte, row in figs:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        x = np.arange(len(classes)); w = 0.38
        for a, logy in zip(ax, (False, True)):
            b1 = a.bar(x - w / 2, [ctr[c] for c in classes], w, label="train (80%)")
            b2 = a.bar(x + w / 2, [cte[c] for c in classes], w, label="test (20%)")
            for b in list(b1) + list(b2):
                a.text(b.get_x() + b.get_width() / 2, b.get_height(), int(b.get_height()), ha="center", va="bottom", fontsize=7)
            a.set_xticks(x); a.set_xticklabels(classes, rotation=20)
            a.set_title(f"{system} / {version}: " + ("log scale" if logy else "linear scale")); a.set_ylabel("windows")
            if logy:
                a.set_yscale("log")
            a.legend()
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"class_distribution_{system}_{version}.png"), dpi=130); plt.close(fig)
        fig, a = plt.subplots(figsize=(7, 4)); w = 0.27
        for j, (key, lab) in enumerate((("train_raw", "raw"), ("train_smote", "SMOTE"), ("train_adasyn", "ADASYN"))):
            vals = [row[c].get(key) for c in classes]
            vals = [np.nan if v is None else v for v in vals]
            bars = a.bar(x - w + j * w, vals, w, label=lab)
            for b, v in zip(bars, vals):
                if not np.isnan(v):
                    a.text(b.get_x() + b.get_width() / 2, v, int(v), ha="center", va="bottom", fontsize=7)
        a.set_yscale("log"); a.set_xticks(x); a.set_xticklabels(classes, rotation=20)
        a.set_title(f"{system} / {version}: training class counts (log scale)"); a.set_ylabel("windows"); a.legend()
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"raw_vs_smote_adasyn_{system}_{version}.png"), dpi=130); plt.close(fig)

    # ------------------------------------------------------------------ final print
    log("\n================ KEY COUNTS ================")
    log("\nsplit summary (block-shuffled, blocks of 10 windows):")
    for k_, v_ in SPLITINFO.items():
        log(f"  {k_}: {v_}")
        log(f"     split info: {json.dumps({a: (round(b, 4) if isinstance(b, float) else b) for a, b in SPLITMETA[k_].items()})}")
    log("\nsplit_counts:\n" + pd.DataFrame(rep["split"]).to_string(index=False))
    log("\ntrain_raw_vs_balanced:\n" + rb.to_string(index=False))
    log("\nADASYN/SMOTE errors:", "none" if not rep["adasyn_err"] else "\n" + pd.DataFrame(rep["adasyn_err"]).to_string(index=False))
    log("\nraw data unchanged:", all(same.values()), same)
    log("\ncolumn warnings (columns kept):")
    wdf = pd.DataFrame(rep["warn"])
    log(wdf.groupby(["system", "issue"]).size().to_string() if len(wdf) else "none")


if __name__ == "__main__":
    main()
