# Results index

Every result file in `report_outputs/` (254 files plus the sub-folders below). Nothing was moved or renamed. *final* = a result that is reported or shown; *intermediate* = needed to produce or check a final result; *log* = run log. 'Used by' lists the proof sections (`scripts/proofs.py`) and whether the web app reads the file; 'report only' means it is a stand-alone result file.

| file | what it is | used by | status |
|---|---|---|---|
| `FINAL_STATUS.md` | Status of the final task list, key result files and every stated limitation | proofs P18 | final |
| `INDEX.md` | This index | report only | final |
| `OVERNIGHT_STATUS.md` | Status of the overnight run: every step, the ten hybrids, LO2, tests, limits | report only | final |
| `baseline_cv.csv` | Cross-validation of the baseline AnoMod models | report only | intermediate |
| `before_after_correction.csv` | Effect of the regularisation correction on overfit AnoMod models | proofs P6; app | final |
| `best_params.json` | Optuna best hyper-parameters of the tuned AnoMod models | report only | intermediate |
| `both_full_cv_results.csv` | (undocumented) | report only | intermediate |
| `both_full_frozen_label_binary__CNN_BiGRU_Att.json` | (undocumented) | report only | intermediate |
| `both_full_frozen_label_binary__CNN_BiLSTM_Att.json` | (undocumented) | report only | intermediate |
| `both_full_frozen_label_binary__ET.json` | (undocumented) | report only | intermediate |
| `both_full_frozen_label_binary__Fusion.json` | (undocumented) | report only | intermediate |
| `both_full_frozen_label_binary__LGBM.json` | (undocumented) | report only | intermediate |
| `both_full_frozen_label_binary__LGBM_tuned.json` | (undocumented) | report only | intermediate |
| `both_full_frozen_label_binary__LR.json` | (undocumented) | report only | intermediate |
| `both_full_frozen_label_binary__RF.json` | (undocumented) | report only | intermediate |
| `both_full_frozen_label_binary__RF_tuned.json` | (undocumented) | report only | intermediate |
| `both_full_frozen_label_binary__Transformer.json` | (undocumented) | report only | intermediate |
| `both_full_frozen_label_binary__XGB.json` | (undocumented) | report only | intermediate |
| `both_full_frozen_label_binary__XGB_tuned.json` | (undocumented) | report only | intermediate |
| `both_full_leaderboard.csv` | (undocumented) | report only | intermediate |
| `both_full_optuna_trials_label_binary__LGBM_tuned.csv` | (undocumented) | report only | intermediate |
| `both_full_optuna_trials_label_binary__RF_tuned.csv` | (undocumented) | report only | intermediate |
| `both_full_optuna_trials_label_binary__XGB_tuned.csv` | (undocumented) | report only | intermediate |
| `both_full_overfit.csv` | (undocumented) | report only | intermediate |
| `both_full_run_identification.csv` | (undocumented) | report only | intermediate |
| `both_full_screening.csv` | (undocumented) | report only | intermediate |
| `both_full_shuffled_control.csv` | (undocumented) | report only | intermediate |
| `both_full_smote_counts.csv` | (undocumented) | report only | intermediate |
| `both_full_test_bootstrap_ci.csv` | (undocumented) | report only | intermediate |
| `both_full_test_metrics.csv` | (undocumented) | report only | intermediate |
| `both_full_thresholds.json` | (undocumented) | report only | intermediate |
| `both_full_training_time.csv` | (undocumented) | report only | intermediate |
| `both_leaderboard.csv` | (undocumented) | app | intermediate |
| `calibration.csv` | Calibration (reliability) statistics of the AnoMod models | report only | intermediate |
| `cross_system_results.csv` | Cross-system generalisation test (train on one system, test on the other): all feature sets, models, controls (proofs P11) | proofs P11 | final |
| `cv_scores.csv` | Per-fold cross-validation scores of the AnoMod models | report only | intermediate |
| `data_integrity.csv` | SHA-256 of every AnoMod / LO2 file at the start and end of the leakage audit (baseline for 'files unchanged') | proofs P7 | final |
| `deep_epochs_per_fold.csv` | Epoch counts used by the deep AnoMod models per fold | report only | intermediate |
| `early_warning_feasibility.md` | Why early warning and segment labels cannot be built from AnoMod, and what would be needed | proofs P15 | final |
| `environment_v2.txt` | Library versions of the AnoMod retraining run | report only | log |
| `error_normal_feature_shift.csv` | Error analysis of misclassified Normal AnoMod test windows (counts, feature shift, SHAP drivers) | report only | final |
| `error_normal_misclassified.csv` | Error analysis of misclassified Normal AnoMod test windows (counts, feature shift, SHAP drivers) | report only | final |
| `error_normal_shap_drivers.csv` | Error analysis of misclassified Normal AnoMod test windows (counts, feature shift, SHAP drivers) | report only | final |
| `error_normal_summary.csv` | Error analysis of misclassified Normal AnoMod test windows (counts, feature shift, SHAP drivers) | app | final |
| `fault_times.csv` | What the AnoMod data say about fault start and end times (none recorded) (proofs P15) | proofs P15 | final |
| `fold_assignment.csv` | 10 block-grouped cross-validation folds of the AnoMod training windows | proofs P3, P4; app | final |
| `frontend_redesign_log.md` | (undocumented) | report only | intermediate |
| `generic_metric_features.csv` | Generic metric aggregates buildable for both AnoMod systems (9) and those that are not | proofs P11 | final |
| `hybrid_anomod_cv_results.csv` | AnoMod hybrids: 5-fold cross-validation results per fold (train / validation scores, epochs, SMOTE b: / a: / hold: counts) | report only | final |
| `hybrid_anomod_explain_groups.csv` | AnoMod hybrids: share of each feature group in the best hybrid's gradient x input attribution | report only | final |
| `hybrid_anomod_explain_label_binary.csv` | AnoMod hybrids: explanation of the best model (gradient x input / attention / SHAP) | report only | final |
| `hybrid_anomod_explain_label_level.csv` | AnoMod hybrids: explanation of the best model (gradient x input / attention / SHAP) | report only | final |
| `hybrid_anomod_folds5.csv` | AnoMod hybrids: the 5 block-aware cross-validation folds of the training part | report only | final |
| `hybrid_anomod_frozen_label_binary__AE_LSTM.json` | AnoMod hybrids: frozen configuration of one model, written before its test scoring | report only | final |
| `hybrid_anomod_frozen_label_binary__CNN_LSTM.json` | AnoMod hybrids: frozen configuration of one model, written before its test scoring | report only | final |
| `hybrid_anomod_frozen_label_binary__GNN_Transformer.json` | AnoMod hybrids: frozen configuration of one model, written before its test scoring | report only | final |
| `hybrid_anomod_frozen_label_binary__Transformer_BiLSTM.json` | AnoMod hybrids: frozen configuration of one model, written before its test scoring | report only | final |
| `hybrid_anomod_frozen_label_binary__VAE_Transformer.json` | AnoMod hybrids: frozen configuration of one model, written before its test scoring | report only | final |
| `hybrid_anomod_frozen_label_level__AE_LSTM.json` | AnoMod hybrids: frozen configuration of one model, written before its test scoring | report only | final |
| `hybrid_anomod_frozen_label_level__CNN_LSTM.json` | AnoMod hybrids: frozen configuration of one model, written before its test scoring | report only | final |
| `hybrid_anomod_frozen_label_level__GNN_Transformer.json` | AnoMod hybrids: frozen configuration of one model, written before its test scoring | report only | final |
| `hybrid_anomod_frozen_label_level__Transformer_BiLSTM.json` | AnoMod hybrids: frozen configuration of one model, written before its test scoring | report only | final |
| `hybrid_anomod_frozen_label_level__VAE_Transformer.json` | AnoMod hybrids: frozen configuration of one model, written before its test scoring | report only | final |
| `hybrid_anomod_graph_edges.csv` | AnoMod hybrids: edges of the component graph with the reason for each edge | report only | final |
| `hybrid_anomod_graph_nodes.csv` | AnoMod hybrids: nodes of the component graph used by the GNN hybrid and the features in each | report only | final |
| `hybrid_anomod_imbalance.csv` | AnoMod hybrids: class-weight and ADASYN dataset-size comparison (never used to train) | report only | final |
| `hybrid_anomod_leaderboard.csv` | AnoMod hybrids: leaderboard with 95% ranges, status ('not completed' where unfinished), reliability | report only | final |
| `hybrid_anomod_overfit.csv` | AnoMod hybrids: overfitting diagnosis (train minus validation macro-F1) before and after the correction stage | report only | final |
| `hybrid_anomod_run_identification.csv` | AnoMod hybrids: run-identification check ('predict the run' rule) for every model | report only | final |
| `hybrid_anomod_screening.csv` | AnoMod hybrids: screening on one training fold (validation scores only): hidden size x imbalance method | report only | final |
| `hybrid_anomod_shuffled_control.csv` | AnoMod hybrids: shuffled-label control for every model | report only | final |
| `hybrid_anomod_smote_counts.csv` | AnoMod hybrids: SMOTE before / after / held-out class counts per fold and for the final model (b: / a: / hold:) | report only | final |
| `hybrid_anomod_test_bootstrap_ci.csv` | AnoMod hybrids: 1000 block-bootstrap 95% intervals of the test metrics | report only | final |
| `hybrid_anomod_test_metrics.csv` | AnoMod hybrids: test metrics (official argmax and tuned-threshold decisions) on the test part | report only | final |
| `hybrid_anomod_thresholds.json` | AnoMod hybrids: decision thresholds / class multipliers chosen on out-of-fold predictions only | report only | final |
| `hybrid_anomod_training_time.csv` | AnoMod hybrids: training times and parameter counts | report only | final |
| `hybrid_both_cv_results.csv` | (undocumented) | report only | intermediate |
| `hybrid_both_frozen_label_binary__AE_LSTM.json` | (undocumented) | report only | intermediate |
| `hybrid_both_frozen_label_binary__BERT_GNN.json` | (undocumented) | report only | intermediate |
| `hybrid_both_frozen_label_binary__CNN_LSTM.json` | (undocumented) | report only | intermediate |
| `hybrid_both_frozen_label_binary__GNN_LSTM.json` | (undocumented) | report only | intermediate |
| `hybrid_both_frozen_label_binary__GNN_Transformer.json` | (undocumented) | report only | intermediate |
| `hybrid_both_frozen_label_binary__Multimodal_Transformer_MLP.json` | (undocumented) | report only | intermediate |
| `hybrid_both_frozen_label_binary__Transformer_BiLSTM.json` | (undocumented) | report only | intermediate |
| `hybrid_both_frozen_label_binary__Transformer_LSTM.json` | (undocumented) | report only | intermediate |
| `hybrid_both_frozen_label_binary__VAE_Transformer.json` | (undocumented) | report only | intermediate |
| `hybrid_both_imbalance.csv` | (undocumented) | report only | intermediate |
| `hybrid_both_leaderboard.csv` | (undocumented) | report only | intermediate |
| `hybrid_both_overfit.csv` | (undocumented) | report only | intermediate |
| `hybrid_both_run_identification.csv` | (undocumented) | report only | intermediate |
| `hybrid_both_screening.csv` | (undocumented) | report only | intermediate |
| `hybrid_both_shuffled_control.csv` | (undocumented) | report only | intermediate |
| `hybrid_both_smote_counts.csv` | (undocumented) | report only | intermediate |
| `hybrid_both_test_bootstrap_ci.csv` | (undocumented) | report only | intermediate |
| `hybrid_both_test_metrics.csv` | (undocumented) | report only | intermediate |
| `hybrid_both_thresholds.json` | (undocumented) | report only | intermediate |
| `hybrid_both_training_time.csv` | (undocumented) | report only | intermediate |
| `hybrid_engine_log.txt` | Log of the hybrid training engine (AnoMod and LO2) | report only | log |
| `hybrid_lo2_cv_results.csv` | LO2 hybrids: 5-fold cross-validation results per fold (train / validation scores, epochs, SMOTE b: / a: / hold: counts) | report only | final |
| `hybrid_lo2_explain_attention_label_binary.csv` | LO2 hybrids: explanation of the best model (gradient x input / attention / SHAP) | report only | final |
| `hybrid_lo2_explain_attention_label_level.csv` | LO2 hybrids: explanation of the best model (gradient x input / attention / SHAP) | report only | final |
| `hybrid_lo2_explain_groups.csv` | LO2 hybrids: share of each feature group in the best hybrid's gradient x input attribution | report only | final |
| `hybrid_lo2_explain_label_binary.csv` | LO2 hybrids: explanation of the best model (gradient x input / attention / SHAP) | report only | final |
| `hybrid_lo2_explain_label_level.csv` | LO2 hybrids: explanation of the best model (gradient x input / attention / SHAP) | report only | final |
| `hybrid_lo2_folds5.csv` | LO2 hybrids: the 5 block-aware cross-validation folds of the training part | report only | final |
| `hybrid_lo2_frozen_label_binary__AE_LSTM.json` | LO2 hybrids: frozen configuration of one model, written before its test scoring | report only | final |
| `hybrid_lo2_frozen_label_binary__BERT_GNN.json` | LO2 hybrids: frozen configuration of one model, written before its test scoring | report only | final |
| `hybrid_lo2_frozen_label_binary__GNN_LSTM.json` | LO2 hybrids: frozen configuration of one model, written before its test scoring | report only | final |
| `hybrid_lo2_frozen_label_binary__Multimodal_Transformer_MLP.json` | LO2 hybrids: frozen configuration of one model, written before its test scoring | report only | final |
| `hybrid_lo2_frozen_label_binary__Transformer_LSTM.json` | LO2 hybrids: frozen configuration of one model, written before its test scoring | report only | final |
| `hybrid_lo2_frozen_label_level__AE_LSTM.json` | LO2 hybrids: frozen configuration of one model, written before its test scoring | report only | final |
| `hybrid_lo2_frozen_label_level__BERT_GNN.json` | LO2 hybrids: frozen configuration of one model, written before its test scoring | report only | final |
| `hybrid_lo2_frozen_label_level__GNN_LSTM.json` | LO2 hybrids: frozen configuration of one model, written before its test scoring | report only | final |
| `hybrid_lo2_frozen_label_level__Multimodal_Transformer_MLP.json` | LO2 hybrids: frozen configuration of one model, written before its test scoring | report only | final |
| `hybrid_lo2_frozen_label_level__Transformer_LSTM.json` | LO2 hybrids: frozen configuration of one model, written before its test scoring | report only | final |
| `hybrid_lo2_imbalance.csv` | LO2 hybrids: class-weight and ADASYN dataset-size comparison (never used to train) | report only | final |
| `hybrid_lo2_leaderboard.csv` | LO2 hybrids: leaderboard with 95% ranges, status ('not completed' where unfinished), reliability | report only | final |
| `hybrid_lo2_logvolume_baseline.csv` | LO2 hybrids: control: logistic regression on log volume alone | app | final |
| `hybrid_lo2_overfit.csv` | LO2 hybrids: overfitting diagnosis (train minus validation macro-F1) before and after the correction stage | report only | final |
| `hybrid_lo2_run_identification.csv` | LO2 hybrids: run-identification check ('predict the run' rule) for every model | report only | final |
| `hybrid_lo2_screening.csv` | LO2 hybrids: screening on one training fold (validation scores only): hidden size x imbalance method | report only | final |
| `hybrid_lo2_shuffled_control.csv` | LO2 hybrids: shuffled-label control for every model | report only | final |
| `hybrid_lo2_smote_counts.csv` | LO2 hybrids: SMOTE before / after / held-out class counts per fold and for the final model (b: / a: / hold:) | report only | final |
| `hybrid_lo2_test_bootstrap_ci.csv` | LO2 hybrids: 1000 block-bootstrap 95% intervals of the test metrics | report only | final |
| `hybrid_lo2_test_metrics.csv` | LO2 hybrids: test metrics (official argmax and tuned-threshold decisions) on the test part | report only | final |
| `hybrid_lo2_thresholds.json` | LO2 hybrids: decision thresholds / class multipliers chosen on out-of-fold predictions only | report only | final |
| `hybrid_lo2_training_time.csv` | LO2 hybrids: training times and parameter counts | report only | final |
| `leaderboard_label_binary.csv` | AnoMod leaderboard, binary labels | app | final |
| `leaderboard_label_level.csv` | AnoMod leaderboard, five-class labels | app | final |
| `leakage_audit.csv` | Leakage audit table C1a-C7 (PASS/WARN) for the AnoMod pipeline | proofs P7 | final |
| `leakage_audit.json` | Leakage audit in JSON, with measured effects (C3 time overlap) and the hash check | proofs P7 | final |
| `leakage_checks.csv` | Single-feature power and run/time-overlap checks behind audit items C3 and C5 | app | intermediate |
| `lo2_distribution_shift.csv` | Distribution shift of LO2 features against AnoMod training data | app | intermediate |
| `lo2_feature_mapping.csv` | Which AnoMod features can be built from LO2 (3 of 124 model inputs) | proofs P9; app | final |
| `lo2_full_cv_results.csv` | LO2 other families (b)-(e): 5-fold cross-validation results per fold (train / validation scores, epochs, SMOTE b: / a: / hold: counts) | report only | final |
| `lo2_full_frozen_label_binary__CNN_BiGRU_Att.json` | LO2 other families (b)-(e): frozen configuration of one model, written before its test scoring | report only | final |
| `lo2_full_frozen_label_binary__CNN_BiLSTM_Att.json` | LO2 other families (b)-(e): frozen configuration of one model, written before its test scoring | report only | final |
| `lo2_full_frozen_label_binary__ET.json` | LO2 other families (b)-(e): frozen configuration of one model, written before its test scoring | report only | final |
| `lo2_full_frozen_label_binary__Fusion.json` | LO2 other families (b)-(e): frozen configuration of one model, written before its test scoring | report only | final |
| `lo2_full_frozen_label_binary__LGBM.json` | LO2 other families (b)-(e): frozen configuration of one model, written before its test scoring | report only | final |
| `lo2_full_frozen_label_binary__LGBM_tuned.json` | LO2 other families (b)-(e): frozen configuration of one model, written before its test scoring | report only | final |
| `lo2_full_frozen_label_binary__LR.json` | LO2 other families (b)-(e): frozen configuration of one model, written before its test scoring | report only | final |
| `lo2_full_frozen_label_binary__RF.json` | LO2 other families (b)-(e): frozen configuration of one model, written before its test scoring | report only | final |
| `lo2_full_frozen_label_binary__RF_tuned.json` | LO2 other families (b)-(e): frozen configuration of one model, written before its test scoring | report only | final |
| `lo2_full_frozen_label_binary__Transformer.json` | LO2 other families (b)-(e): frozen configuration of one model, written before its test scoring | report only | final |
| `lo2_full_frozen_label_binary__XGB.json` | LO2 other families (b)-(e): frozen configuration of one model, written before its test scoring | report only | final |
| `lo2_full_frozen_label_binary__XGB_tuned.json` | LO2 other families (b)-(e): frozen configuration of one model, written before its test scoring | report only | final |
| `lo2_full_frozen_label_level__CNN_BiGRU_Att.json` | LO2 other families (b)-(e): frozen configuration of one model, written before its test scoring | report only | final |
| `lo2_full_frozen_label_level__CNN_BiLSTM_Att.json` | LO2 other families (b)-(e): frozen configuration of one model, written before its test scoring | report only | final |
| `lo2_full_frozen_label_level__ET.json` | LO2 other families (b)-(e): frozen configuration of one model, written before its test scoring | report only | final |
| `lo2_full_frozen_label_level__Fusion.json` | LO2 other families (b)-(e): frozen configuration of one model, written before its test scoring | report only | final |
| `lo2_full_frozen_label_level__LGBM.json` | LO2 other families (b)-(e): frozen configuration of one model, written before its test scoring | report only | final |
| `lo2_full_frozen_label_level__LGBM_tuned.json` | LO2 other families (b)-(e): frozen configuration of one model, written before its test scoring | report only | final |
| `lo2_full_frozen_label_level__LR.json` | LO2 other families (b)-(e): frozen configuration of one model, written before its test scoring | report only | final |
| `lo2_full_frozen_label_level__RF.json` | LO2 other families (b)-(e): frozen configuration of one model, written before its test scoring | report only | final |
| `lo2_full_frozen_label_level__RF_tuned.json` | LO2 other families (b)-(e): frozen configuration of one model, written before its test scoring | report only | final |
| `lo2_full_frozen_label_level__Transformer.json` | LO2 other families (b)-(e): frozen configuration of one model, written before its test scoring | report only | final |
| `lo2_full_frozen_label_level__XGB.json` | LO2 other families (b)-(e): frozen configuration of one model, written before its test scoring | report only | final |
| `lo2_full_frozen_label_level__XGB_tuned.json` | LO2 other families (b)-(e): frozen configuration of one model, written before its test scoring | report only | final |
| `lo2_full_leaderboard.csv` | LO2 leaderboard of all families: hybrids first, then (b)-(e); 'not completed' rows; every row marked unreliable (very small sample) | app | final |
| `lo2_full_lime_example.csv` | LO2: LIME explanation of one real test window | report only | final |
| `lo2_full_not_completed.csv` | LO2 model families not completed | app | final |
| `lo2_full_optuna_trials_label_binary__LGBM_tuned.csv` | LO2 other families (b)-(e): Optuna trials of one tuned model | report only | intermediate |
| `lo2_full_optuna_trials_label_binary__RF_tuned.csv` | LO2 other families (b)-(e): Optuna trials of one tuned model | report only | intermediate |
| `lo2_full_optuna_trials_label_binary__XGB_tuned.csv` | LO2 other families (b)-(e): Optuna trials of one tuned model | report only | intermediate |
| `lo2_full_optuna_trials_label_level__LGBM_tuned.csv` | LO2 other families (b)-(e): Optuna trials of one tuned model | report only | intermediate |
| `lo2_full_optuna_trials_label_level__RF_tuned.csv` | LO2 other families (b)-(e): Optuna trials of one tuned model | report only | intermediate |
| `lo2_full_optuna_trials_label_level__XGB_tuned.csv` | LO2 other families (b)-(e): Optuna trials of one tuned model | report only | intermediate |
| `lo2_full_overfit.csv` | LO2 other families (b)-(e): overfitting diagnosis (train minus validation macro-F1) before and after the correction stage | report only | final |
| `lo2_full_run_identification.csv` | LO2 other families (b)-(e): run-identification check ('predict the run' rule) for every model | report only | final |
| `lo2_full_screening.csv` | LO2 other families (b)-(e): screening on one training fold (validation scores only): hidden size x imbalance method | report only | final |
| `lo2_full_shap_global.csv` | LO2: SHAP of the best classical tree model on the real test windows | report only | final |
| `lo2_full_shuffled_control.csv` | LO2 other families (b)-(e): shuffled-label control for every model | report only | final |
| `lo2_full_smote_counts.csv` | LO2 other families (b)-(e): SMOTE before / after / held-out class counts per fold and for the final model (b: / a: / hold:) | report only | final |
| `lo2_full_test_bootstrap_ci.csv` | LO2 other families (b)-(e): 1000 block-bootstrap 95% intervals of the test metrics | report only | final |
| `lo2_full_test_metrics.csv` | LO2 other families (b)-(e): test metrics (official argmax and tuned-threshold decisions) on the test part | report only | final |
| `lo2_full_thresholds.json` | LO2 other families (b)-(e): decision thresholds / class multipliers chosen on out-of-fold predictions only | report only | final |
| `lo2_full_training_time.csv` | LO2 other families (b)-(e): training times and parameter counts | report only | final |
| `lo2_inspection.txt` | Inspection notes of the LO2 data | report only | log |
| `lo2_log.txt` | Log of the LO2 evaluation | proofs P9; app | log |
| `lo2_results_summary.md` | Plain-language summary of the LO2 external test (models did not transfer) | app | final |
| `lo2_sample_manifest.csv` | Manifest of the LO2 scenarios sampled (read-only data) | proofs P9; app | intermediate |
| `lo2_shortcut_check.csv` | LO2 diagnostic: why the binary labels are almost trivial (same forest on subsets of inputs) | report only | final |
| `lo2_slice_manifest.csv` | LO2 slice: the whole runs chosen (size in MB, role, window counts, reason) | app | final |
| `lo2_split_ids.csv` | LO2 slice: run-level split written before any window existed, then the window-level split | app | final |
| `lo2_test_metrics.csv` | LO2 external test: metrics of every saved model per feature variant and system flag (balanced accuracy 0.500) | proofs P9; app | final |
| `lo2_threshold_free.csv` | LO2 threshold-free metrics (ROC-AUC, PR-AUC) and their bootstrap intervals | proofs P9; app | final |
| `lo2_threshold_free_logvolume_check.csv` | Check that LO2 ROC-AUCs are explained by log volume | proofs P9; app | final |
| `lo2_vs_anomod.csv` | LO2 versus AnoMod test scores per model | app | intermediate |
| `lo2_window_predictions.csv` | Per-window LO2 predictions of the saved models | report only | intermediate |
| `lo2_window_predictions_A_TT.csv` | Per-window LO2 predictions of the saved models | report only | intermediate |
| `log_v2.txt` | Log of the AnoMod retraining run | report only | log |
| `missingness_check.csv` | Missingness and shortcut check: masking, indicator-only classifier, no-shortcut XGBoost (proofs P13) | proofs P13 | final |
| `model_architectures.csv` | Layer summary of the deep AnoMod models | report only | intermediate |
| `notebook_check.json` | Result of executing the Colab notebook in a fresh environment (proof P23) | report only | final |
| `oof_predictions.parquet` | Out-of-fold predictions of the AnoMod models (used to choose decision rules; not tracked: parquet is git-ignored) | report only | intermediate |
| `optuna_trials.csv` | All Optuna trials of the AnoMod tuning | report only | intermediate |
| `overnight_log.md` | Timestamped log of decisions and steps of the overnight run | report only | log |
| `overnight_steps.json` | (undocumented) | report only | intermediate |
| `overnight_tests.json` | (undocumented) | report only | intermediate |
| `paired_bootstrap.csv` | Paired bootstrap comparison of each AnoMod model with the best one (Holm-adjusted) | app | final |
| `preprocess_log.txt` | Log of preprocessing | report only | log |
| `proofs.json` | The proof report P1-P24 (statuses, evidence, terminal text); shown on the Proofs page | proofs P8, P9, P10; app | final |
| `rcaeval_case_results.csv` | RCAEval external test: per-case results | proofs P18 | intermediate |
| `rcaeval_external_results.csv` | RCAEval external test: detection, delay, false alarms, placebo per system and fault type (proofs P18) | proofs P18; app | final |
| `rcaeval_feature_mapping.csv` | RCAEval: which generic aggregates exist per system | proofs P18 | final |
| `rcaeval_summary.md` | RCAEval external test summary (every number read from the CSVs) | proofs P18 | final |
| `results_summary.md` | Plain-language summary of the AnoMod results | app | final |
| `run_identification.csv` | Run-identification check on AnoMod: runs are identifiable and reproduce the scores (proofs P14) | proofs P14 | final |
| `shap_global_label_binary.csv` | Global SHAP importances of the AnoMod models | report only | final |
| `shap_global_label_level.csv` | Global SHAP importances of the AnoMod models | report only | final |
| `shap_group_share_label_binary.csv` | Share of SHAP importance by feature group (api / log / metric) | report only | final |
| `shap_group_share_label_level.csv` | Share of SHAP importance by feature group (api / log / metric) | report only | final |
| `shap_lime_agreement_summary.csv` | Agreement of SHAP and LIME top features on AnoMod test windows | proofs P8; app | final |
| `shap_lime_agreement_windows.csv` | Agreement of SHAP and LIME top features on AnoMod test windows | report only | final |
| `shap_lime_local_windows.csv` | Local SHAP and LIME explanations of individual AnoMod test windows | report only | intermediate |
| `shared_features.csv` | Which of the 155 AnoMod features exist in both systems (8 shared) | proofs P11 | final |
| `split_ids.csv` | Train/test assignment of every AnoMod window (block split, 647 test windows) | proofs P2, P3; app | final |
| `step3_log.txt` | Log of the AnoMod test robustness step | report only | log |
| `storefront_log.md` | (undocumented) | report only | intermediate |
| `storefront_smoke_comparison.md` | (undocumented) | report only | intermediate |
| `test_bootstrap_ci.csv` | 1000 block-bootstrap 95% confidence intervals of the AnoMod test metrics | proofs P5; app | final |
| `test_metrics.csv` | Test metrics of every saved final AnoMod model (official and tuned-threshold decisions, ALL/SN/TT) | proofs P5, P6, P7, P9, P11, P14; app | final |
| `test_per_class.csv` | Per-class precision, recall and F1 of the saved AnoMod models on the test set | report only | final |
| `testbed_cv_by_fault.csv` | Testbed: every run held out once, recall by type and severity (supplementary) | proofs P11, P16; app | final |
| `testbed_cv_false_alarms.csv` | Testbed: false alarms per hour over all fault-free runs (supplementary) | report only | intermediate |
| `testbed_cv_instances.csv` | Testbed: per-instance detection in the every-run-held-out CV | report only | intermediate |
| `testbed_dataset_sample.csv` | Testbed dataset: two sample windows per class (also in docs/TESTBED_DATASET.md) | report only | final |
| `testbed_dataset_summary.csv` | Testbed dataset class counts | report only | final |
| `testbed_false_alarms.csv` | Testbed false alarms per hour on fault-free test runs, per method and alert rule | proofs P11, P16; app | final |
| `testbed_frozen_config.json` | FROZEN testbed detector configuration (model, features, threshold, alert rule) written before the live demos | proofs P12, P19; app | final |
| `testbed_heldout_by_fault.csv` | Testbed held-out recall by fault type, severity and kind, with instance counts and reliability flag | proofs P11, P16; app | final |
| `testbed_heldout_instances.csv` | Testbed held-out per fault instance: detection, delay, lead time | proofs P16; app | intermediate |
| `testbed_heldout_overall.csv` | Testbed held-out results per method (balanced accuracy, macro-F1, Normal recall, ROC-AUC with run-bootstrap CIs) | proofs P11, P16; app | final |
| `testbed_live_schedule_9001.json` | Live demo: fault schedule drawn from the seed and written BEFORE the run (with hash and timestamp) | proofs P12 | final |
| `testbed_live_schedule_9002.json` | Live demo: fault schedule drawn from the seed and written BEFORE the run (with hash and timestamp) | proofs P12 | final |
| `testbed_live_schedule_9003.json` | Live demo: fault schedule drawn from the seed and written BEFORE the run (with hash and timestamp) | proofs P12 | final |
| `testbed_live_schedule_actual_9001.csv` | Live demo: the shop's own record of the injected fault (exact start / end / failure timestamps) | proofs P12 | final |
| `testbed_live_schedule_actual_9002.csv` | Live demo: the shop's own record of the injected fault (exact start / end / failure timestamps) | proofs P12 | final |
| `testbed_live_schedule_actual_9003.csv` | Live demo: the shop's own record of the injected fault (exact start / end / failure timestamps) | proofs P12 | final |
| `testbed_live_validation.csv` | Live demo results: detection delay, false alarms, misses for the three demos (proofs P12) | proofs P12; app | final |
| `testbed_live_windows_9001.csv` | Live demo: the monitor's per-window predictions (risk, flag, alert, status) | proofs P12 | final |
| `testbed_live_windows_9002.csv` | Live demo: the monitor's per-window predictions (risk, flag, alert, status) | proofs P12 | final |
| `testbed_live_windows_9003.csv` | Live demo: the monitor's per-window predictions (risk, flag, alert, status) | proofs P12 | final |
| `testbed_prefailure.csv` | Testbed pre-failure warning (gradual faults): precision, recall, lead time per horizon | proofs P11, P16; app | final |
| `testbed_prefailure_instances.csv` | Testbed pre-failure warning per gradual instance | report only | intermediate |
| `testbed_run_assignment.csv` | Testbed: which runs are held-out test runs and which train/validation folds | proofs P11, P16, P18; app | final |
| `testbed_runid_check.csv` | Testbed run-identification check (proofs P17) | proofs P11, P17; app | final |
| `testbed_shuffled_control.csv` | Testbed shuffled-label control | proofs P11, P16; app | final |
| `testbed_train_cv.csv` | Testbed: run-grouped cross-validation on the train runs (model selection) | proofs P11 | final |
| `thresholds.json` | Saved decision rules (binary threshold / per-class weights) chosen on out-of-fold predictions | proofs P5, P8, P18; app | final |
| `time_estimate.txt` | Time estimate of the AnoMod training | report only | log |
| `training_time.csv` | Training time of the AnoMod models | app | intermediate |
| `underfit_overfit.csv` | Train / cross-validation / test macro-F1 and the overfitting diagnosis per AnoMod model | proofs P6; app | final |

254 files: 165 final, 80 intermediate, 9 logs.

## Sub-folders

| folder | what it is | status |
|---|---|---|
| `figures/` | 264 figures and per-figure CSVs of the AnoMod analysis (confusion matrices, curves, SHAP/LIME plots); the confusion-matrix and curve files are read by the app's Results page | final |
| `old_pipeline/` | 34 files of the first (16-epoch) AnoMod run, kept for the first-run-versus-retrain comparison (proofs P6) | intermediate |

## New in the overnight run (code, documents, notebook)

| path | what it is |
|---|---|
| `models_src/hybrid/` | the ten hybrid model classes (`anomod_hybrids.py`, `lo2_hybrids.py`, shared `base.py`) and the four single deep models (`dl_models.py`) |
| `scripts/hybrid_engine.py` | the shared training protocol (SMOTE inside folds, screening, 5-fold CV, correction stage, frozen config, controls, assembling the result files) |
| `scripts/hybrid_anomod.py`, `scripts/hybrid_lo2.py` | run the five hybrids on AnoMod / on the LO2 slice |
| `scripts/lo2_hybrid_data.py`, `scripts/lo2_data.py` | build the LO2 slice windows from the raw logs (Drain-style templates) and prepare model inputs |
| `scripts/lo2_full.py`, `scripts/lo2_deep.py` | LO2 families (b) classical, (c) Optuna-tuned, (d) deep + XGBoost head, (e) stacking and soft voting |
| `scripts/hybrid_explain.py` | explanations of the best hybrid and of the best LO2 tree model |
| `scripts/proofs_hybrid.py` | proof sections P20 to P24 (called by `scripts/proofs.py`) |
| `scripts/make_notebook.py`, `scripts/make_colab_bundle.py`, `scripts/run_notebook_check.py` | build the Colab notebook, zip the data bundle, execute the notebook in a fresh environment |
| `scripts/build_docs.py` | writes `docs/HYBRID_MODELS.md`, `docs/LO2_SLICE.md`, `docs/NUMBERS_FOR_VIVA.md` from the result files |
| `notebooks/FailSight_Workflow.ipynb` (+ `_executed.ipynb` / `.html`) | the Colab workflow notebook and its executed copy |
| `docs/HYBRID_MODELS.md`, `docs/LO2_SLICE.md`, `docs/COLAB.md`, `docs/NUMBERS_FOR_VIVA.md`, `docs/PRESENT_NOW.md` | documents of the new work |
| `app/backend/hybrid_results.py` | read-only API for the hybrid results (`/api/hybrid`, `/api/lo2/leaderboard`) |
| `app/frontend/src/pages/Hybrids.tsx` | the Hybrid models page |
| `tests/test_hybrid_models.py`, `app/backend/tests/test_hybrid_api.py`, `app/frontend/tests/hybrids.spec.ts` | tests of the new work |

## Where to start

- The proof report: `proofs.json` (also the Proofs page); the testbed: `testbed_run_assignment.csv`, `testbed_heldout_overall.csv`, `testbed_live_validation.csv`, `docs/TESTBED_DATASET.md`; the external test: `rcaeval_summary.md`; the AnoMod results: `test_metrics.csv`, `leakage_audit.csv`, `run_identification.csv`, `lo2_results_summary.md`.
