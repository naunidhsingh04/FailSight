"""Builds report_outputs/INDEX.md: every result file with a one-line description, where it is used (proof section / app endpoint) and whether it is a final result or an intermediate.
Files are never moved or renamed.   python scripts/build_index.py"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "report_outputs"

# name or prefix -> (description, kind)   kind: final | intermediate | log
D = {
    "split_ids.csv": ("Train/test assignment of every AnoMod window (block split, 647 test windows)", "final"),
    "fold_assignment.csv": ("10 block-grouped cross-validation folds of the AnoMod training windows", "final"),
    "data_integrity.csv": ("SHA-256 of every AnoMod / LO2 file at the start and end of the leakage audit (baseline for 'files unchanged')", "final"),
    "leakage_audit.csv": ("Leakage audit table C1a-C7 (PASS/WARN) for the AnoMod pipeline", "final"),
    "leakage_audit.json": ("Leakage audit in JSON, with measured effects (C3 time overlap) and the hash check", "final"),
    "leakage_checks.csv": ("Single-feature power and run/time-overlap checks behind audit items C3 and C5", "intermediate"),
    "test_metrics.csv": ("Test metrics of every saved final AnoMod model (official and tuned-threshold decisions, ALL/SN/TT)", "final"),
    "test_per_class.csv": ("Per-class precision, recall and F1 of the saved AnoMod models on the test set", "final"),
    "test_bootstrap_ci.csv": ("1000 block-bootstrap 95% confidence intervals of the AnoMod test metrics", "final"),
    "paired_bootstrap.csv": ("Paired bootstrap comparison of each AnoMod model with the best one (Holm-adjusted)", "final"),
    "leaderboard_label_binary.csv": ("AnoMod leaderboard, binary labels", "final"),
    "leaderboard_label_level.csv": ("AnoMod leaderboard, five-class labels", "final"),
    "thresholds.json": ("Saved decision rules (binary threshold / per-class weights) chosen on out-of-fold predictions", "final"),
    "calibration.csv": ("Calibration (reliability) statistics of the AnoMod models", "intermediate"),
    "underfit_overfit.csv": ("Train / cross-validation / test macro-F1 and the overfitting diagnosis per AnoMod model", "final"),
    "before_after_correction.csv": ("Effect of the regularisation correction on overfit AnoMod models", "final"),
    "baseline_cv.csv": ("Cross-validation of the baseline AnoMod models", "intermediate"),
    "cv_scores.csv": ("Per-fold cross-validation scores of the AnoMod models", "intermediate"),
    "best_params.json": ("Optuna best hyper-parameters of the tuned AnoMod models", "intermediate"),
    "optuna_trials.csv": ("All Optuna trials of the AnoMod tuning", "intermediate"),
    "deep_epochs_per_fold.csv": ("Epoch counts used by the deep AnoMod models per fold", "intermediate"),
    "model_architectures.csv": ("Layer summary of the deep AnoMod models", "intermediate"),
    "training_time.csv": ("Training time of the AnoMod models", "intermediate"),
    "oof_predictions.parquet": ("Out-of-fold predictions of the AnoMod models (used to choose decision rules; not tracked: parquet is git-ignored)", "intermediate"),
    "error_normal_": ("Error analysis of misclassified Normal AnoMod test windows (counts, feature shift, SHAP drivers)", "final"),
    "shap_global_": ("Global SHAP importances of the AnoMod models", "final"),
    "shap_group_share_": ("Share of SHAP importance by feature group (api / log / metric)", "final"),
    "shap_lime_agreement_": ("Agreement of SHAP and LIME top features on AnoMod test windows", "final"),
    "shap_lime_local_windows.csv": ("Local SHAP and LIME explanations of individual AnoMod test windows", "intermediate"),
    "lo2_results_summary.md": ("Plain-language summary of the LO2 external test (models did not transfer)", "final"),
    "lo2_test_metrics.csv": ("LO2 external test: metrics of every saved model per feature variant and system flag (balanced accuracy 0.500)", "final"),
    "lo2_threshold_free.csv": ("LO2 threshold-free metrics (ROC-AUC, PR-AUC) and their bootstrap intervals", "final"),
    "lo2_threshold_free_logvolume_check.csv": ("Check that LO2 ROC-AUCs are explained by log volume", "final"),
    "lo2_feature_mapping.csv": ("Which AnoMod features can be built from LO2 (3 of 124 model inputs)", "final"),
    "lo2_distribution_shift.csv": ("Distribution shift of LO2 features against AnoMod training data", "intermediate"),
    "lo2_vs_anomod.csv": ("LO2 versus AnoMod test scores per model", "intermediate"),
    "lo2_window_predictions": ("Per-window LO2 predictions of the saved models", "intermediate"),
    "lo2_sample_manifest.csv": ("Manifest of the LO2 scenarios sampled (read-only data)", "intermediate"),
    "lo2_inspection.txt": ("Inspection notes of the LO2 data", "log"),
    "lo2_log.txt": ("Log of the LO2 evaluation", "log"),
    "missingness_check.csv": ("Missingness and shortcut check: masking, indicator-only classifier, no-shortcut XGBoost (proofs P13)", "final"),
    "shared_features.csv": ("Which of the 155 AnoMod features exist in both systems (8 shared)", "final"),
    "generic_metric_features.csv": ("Generic metric aggregates buildable for both AnoMod systems (9) and those that are not", "final"),
    "cross_system_results.csv": ("Cross-system generalisation test (train on one system, test on the other): all feature sets, models, controls (proofs P11)", "final"),
    "run_identification.csv": ("Run-identification check on AnoMod: runs are identifiable and reproduce the scores (proofs P14)", "final"),
    "fault_times.csv": ("What the AnoMod data say about fault start and end times (none recorded) (proofs P15)", "final"),
    "early_warning_feasibility.md": ("Why early warning and segment labels cannot be built from AnoMod, and what would be needed", "final"),
    "testbed_run_assignment.csv": ("Testbed: which runs are held-out test runs and which train/validation folds", "final"),
    "testbed_train_cv.csv": ("Testbed: run-grouped cross-validation on the train runs (model selection)", "final"),
    "testbed_heldout_overall.csv": ("Testbed held-out results per method (balanced accuracy, macro-F1, Normal recall, ROC-AUC with run-bootstrap CIs)", "final"),
    "testbed_heldout_by_fault.csv": ("Testbed held-out recall by fault type, severity and kind, with instance counts and reliability flag", "final"),
    "testbed_heldout_instances.csv": ("Testbed held-out per fault instance: detection, delay, lead time", "intermediate"),
    "testbed_false_alarms.csv": ("Testbed false alarms per hour on fault-free test runs, per method and alert rule", "final"),
    "testbed_prefailure.csv": ("Testbed pre-failure warning (gradual faults): precision, recall, lead time per horizon", "final"),
    "testbed_prefailure_instances.csv": ("Testbed pre-failure warning per gradual instance", "intermediate"),
    "testbed_shuffled_control.csv": ("Testbed shuffled-label control", "final"),
    "testbed_runid_check.csv": ("Testbed run-identification check (proofs P17)", "final"),
    "testbed_cv_by_fault.csv": ("Testbed: every run held out once, recall by type and severity (supplementary)", "final"),
    "testbed_cv_false_alarms.csv": ("Testbed: false alarms per hour over all fault-free runs (supplementary)", "intermediate"),
    "testbed_cv_instances.csv": ("Testbed: per-instance detection in the every-run-held-out CV", "intermediate"),
    "testbed_dataset_summary.csv": ("Testbed dataset class counts", "final"),
    "testbed_dataset_sample.csv": ("Testbed dataset: two sample windows per class (also in docs/TESTBED_DATASET.md)", "final"),
    "testbed_frozen_config.json": ("FROZEN testbed detector configuration (model, features, threshold, alert rule) written before the live demos", "final"),
    "testbed_live_schedule_actual_": ("Live demo: the shop's own record of the injected fault (exact start / end / failure timestamps)", "final"),
    "testbed_live_schedule_": ("Live demo: fault schedule drawn from the seed and written BEFORE the run (with hash and timestamp)", "final"),
    "testbed_live_windows_": ("Live demo: the monitor's per-window predictions (risk, flag, alert, status)", "final"),
    "testbed_live_validation.csv": ("Live demo results: detection delay, false alarms, misses for the three demos (proofs P12)", "final"),
    "rcaeval_external_results.csv": ("RCAEval external test: detection, delay, false alarms, placebo per system and fault type (proofs P18)", "final"),
    "rcaeval_case_results.csv": ("RCAEval external test: per-case results", "intermediate"),
    "rcaeval_feature_mapping.csv": ("RCAEval: which generic aggregates exist per system", "final"),
    "rcaeval_summary.md": ("RCAEval external test summary (every number read from the CSVs)", "final"),
    "proofs.json": ("The proof report P1-P23 (statuses, evidence, terminal text); shown on the Proofs page", "final"),
    "results_summary.md": ("Plain-language summary of the AnoMod results", "final"),
    "INDEX.md": ("This index", "final"),
    "FINAL_STATUS.md": ("Status of the final task list, key result files and every stated limitation", "final"),
    "log_v2.txt": ("Log of the AnoMod retraining run", "log"),
    "step3_log.txt": ("Log of the AnoMod test robustness step", "log"),
    "preprocess_log.txt": ("Log of preprocessing", "log"),
    "environment_v2.txt": ("Library versions of the AnoMod retraining run", "log"),
    "time_estimate.txt": ("Time estimate of the AnoMod training", "log"),
}

# new work (overnight run): hybrid deep-learning models, the LO2 slice and its other families, the Colab notebook
SUFFIX = {
    "test_metrics.csv": "test metrics (official argmax and tuned-threshold decisions) on the test part",
    "test_bootstrap_ci.csv": "1000 block-bootstrap 95% intervals of the test metrics",
    "cv_results.csv": "5-fold cross-validation results per fold (train / validation scores, epochs, SMOTE b: / a: / hold: counts)",
    "smote_counts.csv": "SMOTE before / after / held-out class counts per fold and for the final model (b: / a: / hold:)",
    "thresholds.json": "decision thresholds / class multipliers chosen on out-of-fold predictions only",
    "training_time.csv": "training times and parameter counts",
    "overfit.csv": "overfitting diagnosis (train minus validation macro-F1) before and after the correction stage",
    "shuffled_control.csv": "shuffled-label control for every model",
    "screening.csv": "screening on one training fold (validation scores only): hidden size x imbalance method",
    "run_identification.csv": "run-identification check ('predict the run' rule) for every model",
    "leaderboard.csv": "leaderboard with 95% ranges, status ('not completed' where unfinished), reliability",
    "imbalance.csv": "class-weight and ADASYN dataset-size comparison (never used to train)",
    "folds5.csv": "the 5 block-aware cross-validation folds of the training part",
    "graph_nodes.csv": "nodes of the component graph used by the GNN hybrid and the features in each",
    "graph_edges.csv": "edges of the component graph with the reason for each edge",
    "logvolume_baseline.csv": "control: logistic regression on log volume alone",
    "explain_groups.csv": "share of each feature group in the best hybrid's gradient x input attribution",
}
for _pre, _what in (("hybrid_anomod_", "AnoMod hybrids"), ("hybrid_lo2_", "LO2 hybrids"), ("lo2_full_", "LO2 other families (b)-(e)")):
    for _suf, _d in SUFFIX.items():
        D[_pre + _suf] = (f"{_what}: {_d}", "final")
    D[_pre + "frozen_"] = (f"{_what}: frozen configuration of one model, written before its test scoring", "final")
    D[_pre + "explain_"] = (f"{_what}: explanation of the best model (gradient x input / attention / SHAP)", "final")
    D[_pre + "optuna_trials_"] = (f"{_what}: Optuna trials of one tuned model", "intermediate")
D.update({
    "lo2_slice_manifest.csv": ("LO2 slice: the whole runs chosen (size in MB, role, window counts, reason)", "final"),
    "lo2_split_ids.csv": ("LO2 slice: run-level split written before any window existed, then the window-level split", "final"),
    "lo2_full_leaderboard.csv": ("LO2 leaderboard of all families: hybrids first, then (b)-(e); 'not completed' rows; every row marked unreliable (very small sample)", "final"),
    "lo2_full_not_completed.csv": ("LO2 model families not completed because of the time limit", "final"),
    "lo2_full_shap_global.csv": ("LO2: SHAP of the best classical tree model on the real test windows", "final"),
    "lo2_full_lime_example.csv": ("LO2: LIME explanation of one real test window", "final"),
    "lo2_shortcut_check.csv": ("LO2 diagnostic: why the binary labels are almost trivial (same forest on subsets of inputs)", "final"),
    "hybrid_engine_log.txt": ("Log of the hybrid training engine (AnoMod and LO2)", "log"),
    "notebook_check.json": ("Result of executing the Colab notebook in a fresh environment (proof P23)", "final"),
    "overnight_log.md": ("Timestamped log of decisions and steps of the overnight run", "log"),
    "OVERNIGHT_STATUS.md": ("Status of the overnight run: every step, the ten hybrids, LO2, tests, limits", "final"),
})


def describe(name):
    if name in D:
        return D[name]
    for k, v in D.items():
        if k.endswith("_") and name.startswith(k) or (not k.endswith((".csv", ".json", ".md", ".txt", ".parquet")) and name.startswith(k)):
            return v
    return ("(undocumented)", "intermediate")


def uses():
    """where each file name appears: proof section (nearest preceding Section(\"Pxx\") in scripts/proofs.py) and app code"""
    proofs = (ROOT / "scripts" / "proofs.py").read_text(encoding="utf-8")
    secs = [(m.start(), m.group(1)) for m in re.finditer(r'Section\("(P\d+)"', proofs)]
    app_src = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in list((ROOT / "app" / "backend").glob("*.py")))
    out = {}
    for f in sorted(p.name for p in REP.iterdir() if p.is_file()):
        keys = {f, re.sub(r"\d{4}", "", f)}
        stem = f.rsplit(".", 1)[0]
        ps = set()
        for pos in [m.start() for k in {f, stem.rstrip("_0123456789")} if len(k) > 5 for m in re.finditer(re.escape(k), proofs)]:
            before = [s for s in secs if s[0] < pos]
            if before:
                ps.add(before[-1][1])
        in_app = [m for m in {f, stem.rstrip("_0123456789")} if len(m) > 5 and m in app_src] or (["f-string"] if f.startswith("leaderboard_label") else [])      # results.py builds this name with an f-string
        u = []
        if ps:
            u.append("proofs " + ", ".join(sorted(ps, key=lambda x: int(x[1:]))))
        if in_app:
            u.append("app")
        out[f] = "; ".join(u) or "report only"
    return out


def main():
    use = uses()
    files = sorted(p.name for p in REP.iterdir() if p.is_file())
    L = ["# Results index", "",
         f"Every result file in `report_outputs/` ({len(files)} files plus the sub-folders below). Nothing was moved or renamed. *final* = a result that is reported or shown; *intermediate* = needed to produce or check a final result; *log* = run log. "
         "'Used by' lists the proof sections (`scripts/proofs.py`) and whether the web app reads the file; 'report only' means it is a stand-alone result file.", "",
         "| file | what it is | used by | status |", "|---|---|---|---|"]
    n = {"final": 0, "intermediate": 0, "log": 0}
    for f in files:
        d, k = describe(f)
        n[k] += 1
        L.append(f"| `{f}` | {d} | {use[f]} | {k} |")
    L += ["", f"{len(files)} files: {n['final']} final, {n['intermediate']} intermediate, {n['log']} logs.", "",
          "## Sub-folders", "", "| folder | what it is | status |", "|---|---|---|",
          f"| `figures/` | {len(list((REP / 'figures').rglob('*.*')))} figures and per-figure CSVs of the AnoMod analysis (confusion matrices, curves, SHAP/LIME plots); the confusion-matrix and curve files are read by the app's Results page | final |",
          f"| `old_pipeline/` | {len(list((REP / 'old_pipeline').rglob('*.*')))} files of the first (16-epoch) AnoMod run, kept for the first-run-versus-retrain comparison (proofs P6) | intermediate |", "",
          "## New in the overnight run (code, documents, notebook)", "",
          "| path | what it is |", "|---|---|",
          "| `models_src/hybrid/` | the ten hybrid model classes (`anomod_hybrids.py`, `lo2_hybrids.py`, shared `base.py`) and the four single deep models (`dl_models.py`) |",
          "| `scripts/hybrid_engine.py` | the shared training protocol (SMOTE inside folds, screening, 5-fold CV, correction stage, frozen config, controls, assembling the result files) |",
          "| `scripts/hybrid_anomod.py`, `scripts/hybrid_lo2.py` | run the five hybrids on AnoMod / on the LO2 slice |",
          "| `scripts/lo2_hybrid_data.py`, `scripts/lo2_data.py` | build the LO2 slice windows from the raw logs (Drain-style templates) and prepare model inputs |",
          "| `scripts/lo2_full.py`, `scripts/lo2_deep.py` | LO2 families (b) classical, (c) Optuna-tuned, (d) deep + XGBoost head, (e) stacking and soft voting |",
          "| `scripts/hybrid_explain.py` | explanations of the best hybrid and of the best LO2 tree model |",
          "| `scripts/proofs_hybrid.py` | proof sections P20 to P23 (called by `scripts/proofs.py`) |",
          "| `scripts/make_notebook.py`, `scripts/make_colab_bundle.py`, `scripts/run_notebook_check.py` | build the Colab notebook, zip the data bundle, execute the notebook in a fresh environment |",
          "| `scripts/build_docs.py` | writes `docs/HYBRID_MODELS.md`, `docs/LO2_SLICE.md`, `docs/NUMBERS_FOR_VIVA.md` from the result files |",
          "| `notebooks/FailSight_Workflow.ipynb` (+ `_executed.ipynb` / `.html`) | the Colab workflow notebook and its executed copy |",
          "| `docs/HYBRID_MODELS.md`, `docs/LO2_SLICE.md`, `docs/COLAB.md`, `docs/NUMBERS_FOR_VIVA.md`, `docs/PRESENT_NOW.md` | documents of the new work |",
          "| `app/backend/hybrid_results.py` | read-only API for the hybrid results (`/api/hybrid`, `/api/lo2/leaderboard`) |",
          "| `app/frontend/src/pages/Hybrids.tsx` | the Hybrid models page |",
          "| `tests/test_hybrid_models.py`, `app/backend/tests/test_hybrid_api.py`, `app/frontend/tests/hybrids.spec.ts` | tests of the new work |", "",
          "## Where to start", "",
          "- The proof report: `proofs.json` (also the Proofs page); the testbed: `testbed_run_assignment.csv`, `testbed_heldout_overall.csv`, `testbed_live_validation.csv`, `docs/TESTBED_DATASET.md`; the external test: `rcaeval_summary.md`; the AnoMod results: `test_metrics.csv`, `leakage_audit.csv`, `run_identification.csv`, `lo2_results_summary.md`.", ""]
    (REP / "INDEX.md").write_text("\n".join(L), encoding="utf-8")
    undoc = [f for f in files if describe(f)[0] == "(undocumented)"]
    print(f"wrote INDEX.md: {len(files)} files, undocumented: {undoc}")


if __name__ == "__main__":
    main()

