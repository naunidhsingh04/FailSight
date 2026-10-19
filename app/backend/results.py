"""Read-only access to the result files in report_outputs/, results/v1_16epoch/ and the small derived assets. Every number served by the API comes
from these files; nothing is computed except simple joins, per-fold counts and sorting."""
import json, math
from functools import lru_cache
import numpy as np
import pandas as pd

from . import config

R = lambda name: config.REPORTS / name


def clean(obj):
    """make pandas / numpy objects JSON-safe (NaN -> None)"""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if (math.isnan(float(obj)) or math.isinf(float(obj))) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def records(df: pd.DataFrame) -> list:
    return clean(df.astype(object).where(df.notna(), None).to_dict("records"))


@lru_cache(maxsize=None)
def csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def test_metrics() -> pd.DataFrame:
    return csv(str(R("test_metrics.csv")))


def bootstrap() -> pd.DataFrame:
    return csv(str(R("test_bootstrap_ci.csv")))


def thresholds() -> dict:
    return json.loads(R("thresholds.json").read_text())


def model_metrics(label: str, name: str) -> dict:
    """official (argmax) and tuned-threshold test metrics with the 95% bootstrap CIs, all scopes, exactly as in the CSVs"""
    tm = test_metrics()
    tm = tm[(tm.label_version == label) & (tm.model == name)]
    bs = bootstrap()
    bs = bs[(bs.label_version == label) & (bs.model == name)]
    out = {}
    for variant in ("official", "tuned_threshold"):
        for scope in ("ALL", "SN", "TT"):
            r = tm[(tm.variant == variant) & (tm.scope == scope)]
            if r.empty:
                continue
            r = r.iloc[0]
            out[f"{variant}/{scope}"] = clean({k: r[k] for k in ("n", "accuracy", "balanced_accuracy", "macro_f1", "normal_recall", "roc_auc", "pr_auc", "support_Normal", "precision_Normal") if k in r.index})
    ci = {}
    for scope in ("ALL", "SN", "TT"):
        for m in ("balanced_accuracy", "macro_f1", "normal_recall", "roc_auc"):
            r = bs[(bs.scope == scope) & (bs.metric == m)]
            if not r.empty:
                r = r.iloc[0]
                ci[f"{scope}/{m}"] = clean(dict(point=r.point_estimate, low=r.ci95_low, high=r.ci95_high, n_resamples=int(r.n_resamples)))
    return dict(metrics=out, ci=ci)


def leaderboard(label: str) -> list:
    return records(csv(str(R(f"leaderboard_{label}.csv"))).drop(columns=["label_version"], errors="ignore").reset_index(drop=True))


def confusion(label: str, name: str) -> dict:
    out = {}
    for dec in ("argmax", "tuned"):
        for scope in ("ALL", "SN", "TT"):
            p = R(f"figures/cm_{label}_{name}_{scope}_{dec}.csv")
            if p.exists():
                d = pd.read_csv(p, index_col=0)
                out[f"{scope}/{dec}"] = dict(labels=[c.replace("pred_", "") for c in d.columns], matrix=d.to_numpy().tolist())
    return out


def curves() -> dict:
    p = config.ASSETS / "curves.json"
    return json.loads(p.read_text()) if p.exists() else {}


def correction() -> list:
    return records(csv(str(R("before_after_correction.csv"))))


def overfit_table() -> list:
    d = csv(str(R("underfit_overfit.csv")))
    return records(d[d.scope == "ALL"])


def paired() -> list:
    d = csv(str(R("paired_bootstrap.csv")))
    return records(d[(d.scope == "ALL") & (d.metric == "balanced_accuracy")])


def first_run_vs_retrain() -> list:
    """test metrics (official argmax, all windows) of the models that exist in both runs; the 647 test windows are identical"""
    v1 = csv(str(config.RESULTS_V1 / "test_metrics.csv"))
    v1 = v1[(v1.scope == "ALL") & (v1.variant == "official")]
    v2 = test_metrics()
    v2 = v2[(v2.scope == "ALL") & (v2.variant == "official")]
    rows = []
    for _, a in v1.iterrows():
        b = v2[(v2.model == a.model) & (v2.label_version == a.label_version)]
        if b.empty:
            continue
        b = b.iloc[0]
        rows.append(dict(model=a.model, label_version=a.label_version,
                         **{f"v1_{k}": a[k] for k in ("balanced_accuracy", "macro_f1", "normal_recall", "roc_auc")},
                         **{f"v2_{k}": b[k] for k in ("balanced_accuracy", "macro_f1", "normal_recall", "roc_auc")},
                         delta_balanced_accuracy=b.balanced_accuracy - a.balanced_accuracy, delta_macro_f1=b.macro_f1 - a.macro_f1))
    return clean(rows)


def results() -> dict:
    return dict(
        leaderboards={l: leaderboard(l) for l in ("label_binary", "label_level")},
        paired_bootstrap=paired(), correction=correction(), overfit=overfit_table(), first_run_vs_retrain=first_run_vs_retrain(),
        curves_available=bool(curves()),
        notes=["Test metrics are on the 647 untouched test windows. 95% CIs come from 1000 block bootstrap resamples stratified by run (test_bootstrap_ci.csv).",
               "The first run (results/v1_16epoch) used the same test windows; its cross-validation was row-level and optimistic, so only test scores are compared.",
               "The best model per label was identified on the test set itself."])


# ------------------------------------------------------------------ data & method
def _level_of(window_id: str) -> str:
    system, scenario = window_id.split("|")[:2]
    if scenario.startswith("Normal"):
        return "Normal"
    if system == "SN":
        return {"Perf": "Performance", "Svc": "Service", "DB": "Database", "Code": "Code"}[scenario.split("_")[0]]
    return {"C": "Code", "D": "Database", "P": "Performance", "S": "Service"}[scenario.split("_")[1]]


def data() -> dict:
    sp = csv(str(R("split_ids.csv")))
    fa = csv(str(R("fold_assignment.csv")))
    sp = sp.assign(level=sp.window_id.map(_level_of))
    fnp = config.ROOT / "processed" / "feature_names.json"
    fn = json.loads(fnp.read_text()) if fnp.exists() else {}
    split_counts = records(sp.groupby(["system", "split", "level"]).size().reset_index(name="windows"))
    fa_l = fa[fa.fold > 0].merge(sp[["window_id", "level", "system"]], on="window_id")
    folds = records(fa_l.groupby(["fold", "level"]).size().reset_index(name="validation_windows"))
    smote = records(csv(str(R("old_pipeline/combined_train_raw_vs_balanced.csv"))).drop(columns=["train_adasyn"], errors="ignore"))
    leak = csv(str(R("leakage_checks.csv")))
    leak = leak[leak.check != "single_feature_bal_acc"]
    single = csv(str(R("leakage_checks.csv")))
    single = single[single.check == "single_feature_bal_acc_max"]
    return clean(dict(
        dataset=dict(windows=len(sp), train=int((sp.split == "train").sum()), test=int((sp.split == "test").sum()),
                     systems={s: int((sp.system == s).sum()) for s in ("SN", "TT")}, runs=int(sp.run_id.nunique()), blocks=int(sp.block_id.nunique()),
                     train_share=float((sp.split == "train").mean()), window_seconds=30, step_seconds=10, block_windows=10,
                     raw_features=155, model_inputs=fn.get("n_features"), dropped_constant=len(fn.get("dropped_constant", [])),
                     feature_groups={k: len(v) for k, v in fn.get("groups", {}).items()}, log1p_features=len(fn.get("log1p_features", []))),
        split_counts=split_counts, fold_validation_counts=folds, smote_final_train=smote,
        folds=dict(n_folds=int(fa.fold.max()), group_type=str(fa[fa.fold > 0].group_type.iloc[0]),
                   rows_per_fold=records(fa[fa.fold > 0].groupby("fold").size().reset_index(name="windows"))),
        leakage=records(leak), single_feature_max=records(single[["scope", "item", "value", "threshold", "flag"]]),
        errors=records(csv(str(R("error_normal_summary.csv")))),
        explainability=dict(
            agreement=records(csv(str(R("shap_lime_agreement_summary.csv")))),
            global_top={l: records(csv(str(R(f"shap_global_{l}.csv"))).head(10)) for l in ("label_binary", "label_level")},
            group_share={l: records(csv(str(R(f"shap_group_share_{l}.csv")))) for l in ("label_binary", "label_level")})))


# ------------------------------------------------------------------ LO2
def lo2() -> dict:
    m = csv(str(R("lo2_test_metrics.csv")))
    t = csv(str(R("lo2_threshold_free.csv")))
    mp = csv(str(R("lo2_feature_mapping.csv")))
    sh = csv(str(R("lo2_distribution_shift.csv")))
    vs = csv(str(R("lo2_vs_anomod.csv")))
    man = csv(str(R("lo2_sample_manifest.csv")))
    chk = csv(str(R("lo2_threshold_free_logvolume_check.csv")))
    used = mp[mp.kept_by_preprocessor]
    summary = {}
    for lab, col in (("A_logs_only", "variant_A_logs_only"), ("B_logs_plus_node_metrics", "variant_B_logs_plus_node_metrics")):
        summary[lab] = dict(features_built=int(mp[col].str.startswith("built").sum()), features_total=len(mp),
                            used_features_built=int(used[col].str.startswith("built").sum()), used_features_total=len(used))
    cp = man[man.copied]
    prim_m = m[(m.feature_set == "A_logs_only") & (m.system_flag == "TT")]
    prim_t = t[(t.feature_set == "A_logs_only") & (t.system_flag == "TT")]
    comb = prim_m.merge(prim_t[["model", "roc_auc", "roc_auc_ci95_low", "roc_auc_ci95_high", "pr_auc_anomaly_positive", "pr_auc_anomaly_positive_ci95_low",
                               "pr_auc_anomaly_positive_ci95_high", "mean_p_anomaly_normal_windows", "mean_p_anomaly_anomaly_windows", "mean_p_gap_anomaly_minus_normal"]],
                        on="model", suffixes=("", "_tf"))
    an = vs[(vs.feature_set == "A_logs_only") & (vs.system_flag == "TT") & (vs.metric == "balanced_accuracy") & (vs.anomod_scope == "ALL")][["model", "anomod_test", "drop_lo2_minus_anomod"]]
    comb = comb.merge(an, on="model")
    keep = ["model", "balanced_accuracy", "balanced_accuracy_ci95_low", "balanced_accuracy_ci95_high", "macro_f1", "normal_recall", "anomaly_recall", "share_predicted_anomaly",
            "roc_auc_tf", "roc_auc_ci95_low", "roc_auc_ci95_high", "pr_auc_anomaly_positive", "mean_p_anomaly_normal_windows", "mean_p_anomaly_anomaly_windows",
            "mean_p_gap_anomaly_minus_normal", "anomod_test", "drop_lo2_minus_anomod"]
    comb = comb[[c for c in keep if c in comb.columns]].rename(columns={"roc_auc_tf": "roc_auc"})
    raw_auc = chk[chk.kind == "raw_feature_roc_auc_anomaly_positive"].set_index("name").value
    sp_ = chk[(chk.kind == "spearman_score_vs_log_line_count") & chk.name.str.endswith("|A_logs_only|TT")]
    hi = prim_t[prim_t.roc_auc >= 0.9].model.tolist()
    rho = [float(sp_[sp_.name == f"{x}|A_logs_only|TT"].value.iloc[0]) for x in hi]
    return clean(dict(
        sample=dict(run_id=str(cp.run_id.iloc[0]), scenarios=len(cp), error_scenarios=int(cp.is_error.sum()), megabytes=float(cp.bytes_selected.sum() / 1e6),
                    windows=int(m.n_windows.iloc[0]), normal_windows=int(m.n_normal.iloc[0]), anomaly_windows=int(m.n_anomaly.iloc[0])),
        feature_mapping=summary, primary_setting="variant A (logs only), system flag TT", models=records(comb),
        all_settings=records(m[["model", "feature_set", "system_flag", "balanced_accuracy", "normal_recall", "anomaly_recall", "share_predicted_anomaly"]]),
        threshold_free_all=records(t[["model", "feature_set", "system_flag", "roc_auc", "roc_auc_ci95_low", "roc_auc_ci95_high", "mean_p_gap_anomaly_minus_normal"]]),
        distribution_shift=records(sh[sh.anomod_system == "TT"]),
        log_volume_check=dict(raw_log_line_count_auc=float(raw_auc["log_line_count"]), raw_log_error_count_auc=float(raw_auc["log_error_count"]),
                              high_auc_models=hi, rank_correlation_with_log_line_count=rho,
                              n_high_auc_tracking_log_volume=int(sum(abs(r) > 0.8 for r in rho))),
        summary_markdown=(R("lo2_results_summary.md").read_text(encoding="utf-8") if R("lo2_results_summary.md").exists() else "")))
