# Step 3 results summary

All numbers below are read from the CSVs in `report_outputs/` by `scripts/step3_eval.py`. The test set (647 windows) was not used for training or tuning; the best model was nevertheless *chosen* on it, so differences in its favour are optimistic. Blocks (10 consecutive windows of one run) were resampled with replacement within each run (stratified by run, so every class is present in every resample), 1000 times, seed 42. Runs themselves are not resampled, so the intervals do not cover run-to-run variation.

## Binary labels

- Best model on test: **XGB**. Balanced accuracy 0.774 (95% CI 0.496 to 0.996), macro-F1 0.819 (0.485 to 0.969), Normal recall 0.556 (0.000 to 1.000), ROC-AUC 0.760 (0.461 to 0.994).
- Paired bootstrap on balanced accuracy, XGB vs the other 17 models: the best model is significantly better after Holm correction (p < 0.05) than 10 of them; not distinguishable from LR, RF, LGBM, LGBM_tuned, XGB_tuned, RF_tuned, CNN_BiGRU_Att_XGBhead.
- SN only, XGB: balanced accuracy 0.812 (0.500 to 1.000), Normal recall 0.625 (0.000 to 1.000).
- TT only, XGB: balanced accuracy 0.741 (0.482 to 1.000), Normal recall 0.500 (0.000 to 1.000).
- Per-class recall of the best model (support): Normal 0.556 (36), Anomaly 0.992 (611).

- Calibration of XGB on test (fitted on out-of-fold predictions): Brier raw 0.032, temperature 0.033 (T = 1.458), isotonic 0.038; ECE raw 0.031, temperature 0.027, isotonic 0.033.
- Across all 18 models, temperature scaling lowered the test Brier score for 16 and isotonic for 14.
- SHAP (XGB): top-5 features m_process_resident_memory_bytes__max (metric, 1.657), m_process_resident_memory_bytes__std (metric, 1.509), m_system_disk_usage_percent__mean (metric, 1.069), m_system_disk_usage_percent__max (metric, 0.994), m_node_filesystem_avail_bytes__std (metric, 0.476); share of total mean |SHAP| by group: API 4.6%, log 6.0%, metric 89.4%, system/indicator 0.0%.
- SHAP vs LIME top-5 agreement: 5 correct + 5 wrong: mean 2.80 of 5 shared features, >=3 shared in 70% of 10 windows; 60 random: mean 2.60 of 5 shared features, >=3 shared in 58% of 60 windows.
- Error analysis (XGB): 16 of 36 Normal test windows are misclassified.
  - predicted as: Anomaly 16
  - feature-space nearest training scenario: SN|Perf_Network_Loss 4, TT|Lv_D_cachelimit 4, TT|Lv_P_DISKIO_preserve 3, TT|Lv_P_NETLOSS_preserve 2, SN|Svc_Kill_Media 1
  - system: TT 10, SN 6
  - position: inside a test block 14, adjacent to a train window (20 s overlap) 2

## Level labels

- Best model on test: **Stacking**. Balanced accuracy 0.890 (95% CI 0.838 to 0.941), macro-F1 0.882 (0.832 to 0.935), Normal recall 0.722 (0.500 to 0.900), ROC-AUC 0.990 (0.981 to 0.997).
- Paired bootstrap on balanced accuracy, Stacking vs the other 17 models: the best model is significantly better after Holm correction (p < 0.05) than 11 of them; not distinguishable from RF, XGB, LGBM, XGB_tuned, LGBM_tuned, RF_tuned.
- SN only, Stacking: balanced accuracy 0.920 (0.881 to 0.953), Normal recall 0.750 (0.667 to 0.800).
- TT only, Stacking: balanced accuracy 0.854 (0.765 to 0.948), Normal recall 0.700 (0.400 to 1.000).
- Per-class recall of the best model (support): Normal 0.722 (36), Performance 0.877 (130), Service 0.887 (160), Database 0.981 (154), Code 0.982 (167).

- Calibration of Stacking on test (fitted on out-of-fold predictions): Brier raw 0.110, temperature 0.113 (T = 1.205), isotonic 0.111; ECE raw 0.027, temperature 0.042, isotonic 0.045.
- Across all 18 models, temperature scaling lowered the test Brier score for 12 and isotonic for 13.
- SHAP (LGBM_tuned): top-5 features m_system_disk_usage_percent__mean (metric, 1.180), m_system_disk_usage_percent__max (metric, 0.681), m_node_filesystem_avail_bytes__mean (metric, 0.650), m_node_filesystem_avail_bytes__max (metric, 0.544), m_process_resident_memory_bytes__max (metric, 0.308); share of total mean |SHAP| by group: API 4.5%, log 2.3%, metric 90.4%, system/indicator 2.9%.
- SHAP vs LIME top-5 agreement: 5 correct + 5 wrong: mean 2.90 of 5 shared features, >=3 shared in 80% of 10 windows; 60 random: mean 2.83 of 5 shared features, >=3 shared in 70% of 60 windows.
- Error analysis (LGBM_tuned): 18 of 36 Normal test windows are misclassified.
  - predicted as: Performance 9, Database 6, Service 3
  - feature-space nearest training scenario: SN|Perf_Network_Loss 4, TT|Lv_D_cachelimit 4, TT|Lv_P_DISKIO_preserve 3, TT|Lv_P_NETLOSS_preserve 2, SN|Svc_Kill_UserTimeline 1
  - system: TT 10, SN 8
  - position: inside a test block 16, adjacent to a train window (20 s overlap) 2

## Caveats (interpretation, not numbers)

- Each class comes from a small number of scenario runs, and most top SHAP features are host-level gauges (disk usage, filesystem space, process memory). They can identify *which run* a window belongs to rather than *what failed*, so high importance is not evidence of a causal failure indicator.
- The test set holds 36 Normal windows (SN 16, TT 20) from two runs, so Normal recall and its confidence intervals rest on very few blocks.
- The best model per label was identified on the test set; the paired-bootstrap p-values therefore favour it. SHAP/LIME explain the best single tree-based model, not the ensemble.

## Files

test_bootstrap_ci.csv, test_per_class.csv, paired_bootstrap.csv, calibration.csv, shap_global_*.csv, shap_group_share_*.csv, shap_lime_local_windows.csv, shap_lime_agreement_windows.csv, shap_lime_agreement_summary.csv, error_normal_*.csv; figures in `report_outputs/figures/step3/`.
