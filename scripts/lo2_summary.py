"""Writes report_outputs/lo2_results_summary.md; every number is read from the CSVs produced by lo2_sample.py / lo2_windows.py / lo2_eval.py."""
from pathlib import Path
import pandas as pd

REP = Path(__file__).resolve().parent.parent / "report_outputs"
man = pd.read_csv(REP / "lo2_sample_manifest.csv")
mp = pd.read_csv(REP / "lo2_feature_mapping.csv")
res = pd.read_csv(REP / "lo2_test_metrics.csv")
vs = pd.read_csv(REP / "lo2_vs_anomod.csv")
sh = pd.read_csv(REP / "lo2_distribution_shift.csv")
pred = pd.read_csv(REP / "lo2_window_predictions_A_TT.csv")
cp = man[man.copied]
used = mp[mp.kept_by_preprocessor]
f = lambda x: f"{x:.3f}"
o = ["# LO2 external test: results summary", "",
     "All numbers are read from the CSVs in `report_outputs/` (lo2_sample_manifest, lo2_feature_mapping, lo2_test_metrics, lo2_vs_anomod, lo2_distribution_shift). "
     "No model was trained, tuned or refitted on LO2.", "",
     "## Sample", "",
     f"- Run {cp.run_id.iloc[0]} (earliest run), scenarios in start order 1..{int(cp.scenario_order_in_run.max())}: {len(cp)} whole scenarios ({int(cp.is_error.sum())} error, {int((cp.is_error == 0).sum())} `correct`), "
     f"{cp.bytes_selected.sum() / 1e6:.1f} MB of selected files (service logs, MySQL log, shared node_exporter metrics).",
     f"- Windows built with the AnoMod 30 s / 10 s rule per scenario: {len(pred[pred.model == pred.model.iloc[0]])} windows = "
     f"{int((pred[pred.model == pred.model.iloc[0]].true_label_binary == 0).sum())} Normal (all from the single `correct` scenario) + "
     f"{int((pred[pred.model == pred.model.iloc[0]].true_label_binary == 1).sum())} Anomaly (one window per error scenario, because error scenarios last only 10-18 s).", "",
     "## Feature mapping loses most features", ""]
for lab, col in (("A logs only (requested)", "variant_A_logs_only"), ("B logs + shared node metrics (extra)", "variant_B_logs_plus_node_metrics")):
    nb, nk = int(mp[col].str.startswith("built").sum()), int(used[col].str.startswith("built").sum())
    o.append(f"- Variant {lab}: {nb} of {len(mp)} AnoMod features can be built ({nb / len(mp):.1%}); of the {len(used)} features the models actually use, {nk} ({nk / len(used):.1%}). "
             f"The rest are missing and imputed by the preprocessor.")
o.append(f"- API features: {int((mp.group == 'api').sum())} of {int((mp.group == 'api').sum())} cannot be built (LO2 has no per-request status, latency or size). "
         f"Metric features: {int(mp[(mp.group == 'metrics') & mp.variant_A_logs_only.str.startswith('cannot')].shape[0])} of {int((mp.group == 'metrics').sum())} missing in variant A.")
o += ["", "## Results (binary labels, final models unchanged)", ""]
prim = res[(res.feature_set == "A_logs_only") & (res.system_flag == "TT")].set_index("model")
an = vs[(vs.feature_set == "A_logs_only") & (vs.system_flag == "TT") & (vs.metric == "balanced_accuracy") & (vs.anomod_scope == "ALL")].set_index("model")
o.append(f"- Primary setting (variant A, system flag TT): balanced accuracy is {f(prim.balanced_accuracy.min())} to {f(prim.balanced_accuracy.max())} for all {len(prim)} models; "
         f"{int((prim.share_predicted_anomaly == 1).sum())} models label every window Anomaly (Normal recall 0), {int((prim.share_predicted_anomaly == 0).sum())} label every window Normal (anomaly recall 0), "
         f"{int(((prim.share_predicted_anomaly > 0) & (prim.share_predicted_anomaly < 1)).sum())} give a mix.")
o.append(f"- AnoMod test balanced accuracy of the same models: {f(an.anomod_test.min())} to {f(an.anomod_test.max())}; mean drop LO2 minus AnoMod: {f(an.drop_lo2_minus_anomod.mean())} "
         f"(best AnoMod model {an.anomod_test.idxmax()}: {f(an.anomod_test.max())} -> {f(an.lo2[an.anomod_test.idxmax()])}).")
o.append(f"- Bootstrap 95% CI of balanced accuracy (scenarios resampled within class): {f(prim.balanced_accuracy_ci95_low.min())} to {f(prim.balanced_accuracy_ci95_high.max())} across models. "
         "The interval collapses to a point because the predictions are constant; it does not mean the estimate is precise (see caveats).")
for lab, fs, fl in (("variant B, flag TT", "B_logs_plus_node_metrics", "TT"), ("variant A, flag SN", "A_logs_only", "SN"), ("variant B, flag SN", "B_logs_plus_node_metrics", "SN")):
    r = res[(res.feature_set == fs) & (res.system_flag == fl)]
    o.append(f"- {lab}: balanced accuracy {f(r.balanced_accuracy.min())} to {f(r.balanced_accuracy.max())}; models above 0.5: {int((r.balanced_accuracy > 0.5).sum())} of {len(r)}; "
             f"models predicting Anomaly for every window: {int((r.share_predicted_anomaly == 1).sum())}.")
m = pred.groupby(["model", "true_label_binary"]).p_anomaly.mean().unstack()
o.append(f"- Mean predicted P(Anomaly) on Normal vs Anomaly windows (variant A, flag TT) differs by at most {f((m[1] - m[0]).abs().max())} for any model: the models do not separate the two classes.")
tf_ = pd.read_csv(REP / "lo2_threshold_free.csv")
o += ["", "## Threshold-free metrics (window bootstrap, 1000 resamples, 95% CI)", ""]
pf = tf_[(tf_.feature_set == "A_logs_only") & (tf_.system_flag == "TT")].set_index("model")
o.append(f"- Chance level: ROC-AUC 0.500; PR-AUC with Anomaly as positive class {f(pf.pr_auc_anomaly_chance_level.iloc[0])} (the Anomaly share); PR-AUC with Normal as positive class {f(pf.pr_auc_normal_chance_level.iloc[0])}.")
b = pf.roc_auc.idxmax(); w_ = pf.roc_auc.idxmin()
o.append(f"- Variant A, flag TT: ROC-AUC ranges from {f(pf.roc_auc.min())} ({w_}) to {f(pf.roc_auc.max())} ({b}); models with ROC-AUC above 0.5: {int((pf.roc_auc > 0.5).sum())}, below 0.5: {int((pf.roc_auc < 0.5).sum())}, "
         f"exactly 0.5: {int((pf.roc_auc == 0.5).sum())} of {len(pf)}. Models whose 95% CI lies entirely above 0.5: {int((pf.roc_auc_ci95_low > 0.5).sum())}.")
o.append(f"- Best by ROC-AUC, {b}: ROC-AUC {f(pf.roc_auc[b])} (95% CI {f(pf.roc_auc_ci95_low[b])} to {f(pf.roc_auc_ci95_high[b])}), PR-AUC Anomaly-positive {f(pf.pr_auc_anomaly_positive[b])} "
         f"({f(pf.pr_auc_anomaly_positive_ci95_low[b])} to {f(pf.pr_auc_anomaly_positive_ci95_high[b])}), PR-AUC Normal-positive {f(pf.pr_auc_normal_positive[b])} ({f(pf.pr_auc_normal_positive_ci95_low[b])} to {f(pf.pr_auc_normal_positive_ci95_high[b])}).")
o.append(f"- Mean predicted P(Anomaly), Normal vs Anomaly windows, variant A flag TT: gap (Anomaly minus Normal) ranges from {f(pf.mean_p_gap_anomaly_minus_normal.min())} to {f(pf.mean_p_gap_anomaly_minus_normal.max())}; "
         f"models with a negative gap (higher P(Anomaly) on Normal windows): {int((pf.mean_p_gap_anomaly_minus_normal < 0).sum())} of {len(pf)}; the largest absolute gap is {f(pf.mean_p_gap_anomaly_minus_normal.abs().max())}.")
for lab, fs, fl in (("variant B, flag TT", "B_logs_plus_node_metrics", "TT"), ("variant A, flag SN", "A_logs_only", "SN"), ("variant B, flag SN", "B_logs_plus_node_metrics", "SN")):
    r = tf_[(tf_.feature_set == fs) & (tf_.system_flag == fl)]
    o.append(f"- {lab}: ROC-AUC {f(r.roc_auc.min())} to {f(r.roc_auc.max())}; models with CI entirely above 0.5: {int((r.roc_auc_ci95_low > 0.5).sum())} of {len(r)}; "
             f"models with identical scores for all windows (ROC-AUC exactly 0.5): {int((r.roc_auc == 0.5).sum())}; gap range {f(r.mean_p_gap_anomaly_minus_normal.min())} to {f(r.mean_p_gap_anomaly_minus_normal.max())}.")
lv = pd.read_csv(REP / "lo2_threshold_free_logvolume_check.csv")
raw = lv[lv.kind == "raw_feature_roc_auc_anomaly_positive"].set_index("name").value
sp_ = lv[(lv.kind == "spearman_score_vs_log_line_count") & lv.name.str.endswith("|A_logs_only|TT")].set_index("name").value
hi = pf[pf.roc_auc >= 0.9].index
rho_hi = [sp_[f"{m}|A_logs_only|TT"] for m in hi]
o.append(f"- Log-volume check: the raw log_line_count alone separates the two classes with ROC-AUC {f(raw['log_line_count'])} (Normal windows have more log lines because `correct` runs 60 s, error scenarios 10-18 s); "
         f"raw log_error_count alone: {f(raw['log_error_count'])}. Of the {len(hi)} models with ROC-AUC >= 0.9 (variant A, flag TT), {int((pd.Series(rho_hi).abs() > 0.8).sum())} have a rank correlation with log_line_count above 0.8 in absolute value "
         f"(range {f(min(rho_hi))} to {f(max(rho_hi))}). Log volume therefore explains the ranking of {int((pd.Series(rho_hi).abs() > 0.8).sum())} of them; for the others this analysis does not identify what drives the ranking "
         f"(imputed and clipped inputs, system flag, was_missing flags). Because all Normal windows come from one scenario, any separation cannot be told apart from a scenario-specific artefact and is not evidence of failure detection.")
o.append("- Reading: ROC-AUC is rank-based; with 5 Normal windows from one scenario one ordering decision moves it by 0.2, and the CIs resample windows (as requested), not scenarios, so they understate the real uncertainty.")
o += ["", "## Why: distribution shift of the three log features that can be built", ""]
for _, r in sh[(sh.anomod_system == "TT")].iterrows():
    o.append(f"- {r.feature}, LO2 {r.lo2_class}: median {r.lo2_median:,.0f} vs AnoMod TT train median {r.anomod_train_median:,.0f} (99.5th percentile {r.anomod_train_p995:,.0f}); "
             f"{r.share_lo2_above_anomod_p995:.0%} of LO2 windows exceed the AnoMod TT 99.5th percentile.")
o += ["", "## Caveats", "",
      "- 5 Normal windows from ONE scenario and 23 Anomaly windows from 23 scenarios of ONE run: far too little for any statistical conclusion; confidence intervals cannot represent run-to-run variation.",
      "- LO2 labels are scenario-level (`correct` still contains failing requests); LO2 logs are DEBUG-level Java logs, so log counts are orders of magnitude above AnoMod's.",
      "- Log counts also encode the scenario length (about 60 s for `correct`, 10-18 s for error scenarios) because windows hold whatever the scenario produced.",
      "- The AnoMod models use system_is_TT; LO2 is assigned TT (primary) or SN (sensitivity), neither is correct.", "",
      "Files: lo2_sample_manifest.csv, lo2_feature_mapping.csv, lo2_test_metrics.csv, lo2_vs_anomod.csv, lo2_distribution_shift.csv, lo2_window_predictions.csv, lo2_threshold_free.csv, lo2_threshold_free_logvolume_check.csv, lo2_inspection.txt, figures/step4/.", ""]
(REP / "lo2_results_summary.md").write_text("\n".join(o), encoding="utf-8")
print("\n".join(o))
