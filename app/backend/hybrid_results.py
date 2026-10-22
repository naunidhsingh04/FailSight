"""Read-only API data for the hybrid deep-learning models (AnoMod and the LO2 slice) and for the LO2 leaderboard.
Every number comes from the saved CSV files in report_outputs/ (hybrid_anomod_*.csv, hybrid_lo2_*.csv, lo2_full_*.csv, lo2_slice_manifest.csv); nothing is computed except
simple joins and sorting. When a file is missing the corresponding part is returned empty and `available` is false, so the backend still starts in a clean clone."""
import json

import pandas as pd

from . import config
from .results import clean, records

R = lambda name: config.REPORTS / name
FOLD_NOTE = ("Original AnoMod models: 10-fold. Hybrids and LO2 work: 5-fold, chosen because of the time limit. "
             "All AnoMod models are scored on the same 647 test windows.")


def _read(name):
    p = R(name)
    return pd.read_csv(p) if p.exists() else None


def _models(prefix: str) -> list:
    lead = _read(f"{prefix}leaderboard.csv")
    if lead is None:
        return []
    ci = _read(f"{prefix}test_bootstrap_ci.csv")
    shuf = _read(f"{prefix}shuffled_control.csv")
    tt = _read(f"{prefix}training_time.csv")
    ovf = _read(f"{prefix}overfit.csv")
    rows = []
    for r in lead.to_dict("records"):
        name, lab = r["model"], r["label_version"]
        row = dict(r)
        if ci is not None:
            for m in ("normal_recall", "roc_auc", "macro_f1"):
                x = ci[(ci.model == name) & (ci.label_version == lab) & (ci.metric == m)]
                if not x.empty:
                    row[f"{m}_ci_low"], row[f"{m}_ci_high"] = float(x.ci95_low.iloc[0]), float(x.ci95_high.iloc[0])
        if shuf is not None:
            x = shuf[(shuf.model == name) & (shuf.label_version == lab)]
            if not x.empty:
                row["shuffled_balanced_accuracy"], row["shuffled_verdict"] = float(x.shuffled_balanced_accuracy.iloc[0]), x.verdict.iloc[0]
        if tt is not None:
            x = tt[(tt.model == name) & (tt.label_version == lab)]
            if not x.empty:
                row["train_seconds_total"] = float(x.total_seconds_incl_screening_and_controls.iloc[0])
        if ovf is not None:
            x = ovf[(ovf.model == name) & (ovf.label_version == lab)]
            if not x.empty:
                row["overfit_stages"] = records(x[["stage", "cv_train_macro_f1", "cv_val_macro_f1", "cv_gap_macro_f1", "diagnosis", "adopted"]])
        rows.append(row)
    return rows


def _best(models, label):
    done = [m for m in models if m["label_version"] == label and m.get("status") in ("done", "did not learn") and m.get("test_bal_acc") == m.get("test_bal_acc")]
    return max(done, key=lambda m: m["test_bal_acc"]) if done else None


def hybrids() -> dict:
    out = dict(fold_note=FOLD_NOTE, available=False)
    # ---- AnoMod
    am = _models("hybrid_anomod_")
    out["anomod"] = dict(models=am, available=bool(am))
    if am:
        out["available"] = True
        sp = _read("split_ids.csv")
        out["anomod"]["facts"] = dict(windows=int(len(sp)), train=int((sp.split == "train").sum()), test=int((sp.split == "test").sum()), runs=int(sp.run_id.nunique()),
                                      folds=5, healthy_test_windows=36, window_seconds=30, megabytes=None)
        cmp_ = {}
        for lab in ("label_binary", "label_level"):
            lb = _read(f"leaderboard_{lab}.csv")
            b = _best(am, lab)
            if lb is not None and b is not None:
                top = lb.sort_values("test_bal_acc", ascending=False).iloc[0]
                bs = _read("test_bootstrap_ci.csv")
                x = bs[(bs.model == top.model) & (bs.label_version == lab) & (bs.scope == "ALL") & (bs.metric == "balanced_accuracy")]
                lo, hi = (float(x.ci95_low.iloc[0]), float(x.ci95_high.iloc[0])) if not x.empty else (None, None)
                cmp_[lab] = dict(best_hybrid=b["model"], best_hybrid_ba=b["test_bal_acc"], best_hybrid_low=b.get("test_bal_acc_ci_low"), best_hybrid_high=b.get("test_bal_acc_ci_high"),
                                 best_existing=str(top.model), best_existing_ba=float(top.test_bal_acc), best_existing_low=lo, best_existing_high=hi,
                                 hybrid_beats_best_existing=bool(b["test_bal_acc"] > float(top.test_bal_acc)),
                                 ranges_overlap=bool(lo is not None and b.get("test_bal_acc_ci_low") is not None and b["test_bal_acc_ci_low"] <= hi and lo <= b["test_bal_acc_ci_high"]))
        out["anomod"]["comparison"] = cmp_
        out["anomod"]["integrity"] = dict(smote_counts=records((_read("hybrid_anomod_smote_counts.csv") or pd.DataFrame()).head(0)))
    # ---- LO2
    lm = _models("hybrid_lo2_")
    out["lo2"] = dict(models=lm, available=bool(lm))
    man = _read("lo2_slice_manifest.csv")
    if man is not None:
        sp = _read("lo2_split_ids.csv")
        facts = dict(runs=int(len(man)), megabytes=float(man.size_mb.sum()), cap_megabytes=150, share_of_download_percent=round(100 * float(man.size_mb.sum()) / 887.4, 1),
                     folds=5, run_files=records(man[["run_id", "size_mb", "role", "scrape_rows", "normal_rows", "anomalous_rows"]]))
        if sp is not None and "window_id" in sp.columns:
            facts.update(windows=int(len(sp)), train=int((sp.split == "train").sum()), test=int((sp.split == "test").sum()),
                         class_counts=clean(sp.label_level.value_counts().to_dict()), test_class_counts=clean(sp[sp.split == "test"].label_level.value_counts().to_dict()),
                         healthy_test_windows=int(((sp.split == "test") & (sp.label_level == "Normal")).sum()))
        facts["unreliable"] = bool(facts["runs"] < 6 or facts.get("windows", 0) < 150)
        facts["embedding"] = "TF-IDF + truncated SVD (a downloadable BERT-style encoder was not available)"
        out["lo2"]["facts"] = facts
        out["lo2"]["logvolume_baseline"] = records(_read("hybrid_lo2_logvolume_baseline.csv") if _read("hybrid_lo2_logvolume_baseline.csv") is not None else pd.DataFrame())
    out["notes"] = [FOLD_NOTE,
                    "LO2 results are about training on LO2's own data. They say nothing about transfer from AnoMod or to any other system.",
                    "Nothing here forecasts failures or the time until a failure; that is future work."]
    return clean(out)


def lo2_leaderboard() -> dict:
    """the LO2 leaderboard: hybrids first, then the other families in priority order; families not finished are listed as 'not completed because of the time limit'"""
    p = R("lo2_full_leaderboard.csv")
    if not p.exists():
        return dict(available=False, rows=[], not_completed=[], note="No LO2 leaderboard file yet.")
    d = pd.read_csv(p)
    nc = _read("lo2_full_not_completed.csv")
    return clean(dict(available=True, rows=records(d), not_completed=records(nc) if nc is not None else [], note=FOLD_NOTE))
