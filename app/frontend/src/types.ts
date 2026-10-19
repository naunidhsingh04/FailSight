export type Label = 'label_binary' | 'label_level'
export const LABEL_NAME: Record<Label, string> = { label_binary: 'Binary (Normal / Anomaly)', label_level: 'Five-class (Normal + four failure levels)' }

export interface CI { point: number | null; low: number | null; high: number | null; n_resamples: number }
export interface MetricSet { n?: number; accuracy?: number; balanced_accuracy?: number; macro_f1?: number; normal_recall?: number; roc_auc?: number; pr_auc?: number }
export interface DecisionRule { type: 'threshold' | 'class_weights'; threshold?: number; weights?: number[]; oof_normal_recall: number; oof_balanced_accuracy: number; oof_balanced_accuracy_default_argmax: number; normal_recall_ge_0_70_reachable: boolean }

export interface ModelInfo {
  name: string
  label_version: Label
  family: 'tree' | 'linear' | 'deep' | 'ensemble'
  kind: string
  base: string | null
  imbalance_method: string | null
  corrected_variant: boolean
  members: string[]
  available: boolean
  unavailable_reason: string
  supports_explain: boolean
  decision_rule: DecisionRule | null
  metrics: Record<string, MetricSet>
  ci: Record<string, CI>
}
export interface ModelsResponse { models: ModelInfo[]; ci_note: string }

export interface Sample { id: string; window_id: string; system: 'SN' | 'TT'; scenario: string; label_level: string; label_binary: string; features: Record<string, number | null> }
export interface SamplesResponse { source: string; windows: Sample[] }

export interface GroupMissing { expected: number; missing: number }
export interface Missing { expected_features: number; missing_features: number; missing_fraction: number; by_group: Record<string, GroupMissing>; reliability: 'normal' | 'degraded' | 'unreliable'; structurally_absent_for_system: number }
export interface PredictResult { index: number; window_id: string | null; system: string; predicted_class: string; default_argmax_class: string; probabilities: Record<string, number>; missing: Missing }
export interface PredictResponse {
  model: string; label_version: Label; family: string; corrected_variant: boolean; decision_rule: DecisionRule & { on?: string; value?: number }
  classes: string[]; n_windows: number; class_counts: Record<string, number>; results: PredictResult[]; warnings: string[]; reliability_note: string
  input: { feature_columns_missing: number; n_unknown_columns: number; non_numeric_cells: number }
}

export interface ShapTop { feature: string; group: string; shap: number; preprocessed_value: number; raw_value: number | null }
export interface LimeTop { feature: string; group: string; weight: number; preprocessed_value: number; raw_value: number | null }
export interface ExplainResponse {
  prediction: PredictResult; decision_rule: PredictResponse['decision_rule']; explained_class: string; warnings: string[]; note: string
  shap: { method: string; base_value: number; top: ShapTop[]; groups: { group: string; total_abs_shap: number; share: number }[]; n_features: number }
  lime: { available: boolean; reason?: string; method?: string; top?: LimeTop[] }
}

export interface LeaderRow { model: string; test_bal_acc: number; test_macro_f1: number; test_normal_recall: number; test_normal_precision: number; test_roc_auc: number; test_pr_auc: number; cv_val_bal_acc: number; cv_val_macro_f1: number }
export interface CorrectionRow { model: string; label_version: Label; kind: string; correction: string; val_bal_acc_before: number; val_bal_acc_after: number; val_macro_f1_before: number; val_macro_f1_after: number; gap_macro_f1_before: number; gap_macro_f1_after: number; adopted: boolean; rule: string }
export interface OverfitRow { model: string; label_version: Label; stage: string; cv_train_macro_f1: number; cv_val_macro_f1: number; cv_gap_macro_f1: number; final_train_macro_f1?: number | null; test_macro_f1?: number | null; train_test_gap_macro_f1?: number | null; diagnosis: string }
export interface PairedRow { label_version: Label; best_model: string; other_model: string; mean_diff: number; ci95_low: number; ci95_high: number; p_holm: number; 'significant_at_0.05_holm': boolean }
export interface V1Row { model: string; label_version: Label; v1_balanced_accuracy: number; v1_macro_f1: number; v1_normal_recall: number; v2_balanced_accuracy: number; v2_macro_f1: number; v2_normal_recall: number; delta_balanced_accuracy: number; delta_macro_f1: number }
export interface ResultsResponse { leaderboards: Record<Label, LeaderRow[]>; paired_bootstrap: PairedRow[]; correction: CorrectionRow[]; overfit: OverfitRow[]; first_run_vs_retrain: V1Row[]; notes: string[] }
export interface ModelDetail { model: string; label_version: Label; confusion: Record<string, { labels: string[]; matrix: number[][] }>; curves: { roc: number[][]; pr: number[][] } | null; curve_note: string }

export interface DataResponse {
  dataset: { windows: number; train: number; test: number; systems: Record<string, number>; runs: number; blocks: number; train_share: number; window_seconds: number; step_seconds: number; block_windows: number; raw_features: number; model_inputs: number; dropped_constant: number; feature_groups: Record<string, number>; log1p_features: number }
  split_counts: { system: string; split: string; level: string; windows: number }[]
  fold_validation_counts: { fold: number; level: string; validation_windows: number }[]
  smote_final_train: { label_version: string; class: string; train_raw: number; train_smote: number }[]
  folds: { n_folds: number; group_type: string; rows_per_fold: { fold: number; windows: number }[] }
  leakage: { check: string; scope: string; item: string; value: string | number; threshold: string | number | null; flag: boolean; detail: string | null }[]
  single_feature_max: { scope: string; item: string; value: number; threshold: string; flag: boolean }[]
  explainability: {
    agreement: { label_version: Label; model: string; window_set: string; outcome: string; n_windows: number; mean_shared_top5: number; share_with_ge3_shared: number }[]
    global_top: Record<Label, { feature: string; group: string; mean_abs_shap: number; rank: number; model: string }[]>
    group_share: Record<Label, { group: string; share_of_total_mean_abs_shap: number; model: string }[]>
  }
}

export interface Lo2Model { model: string; balanced_accuracy: number; balanced_accuracy_ci95_low: number; balanced_accuracy_ci95_high: number; macro_f1: number; normal_recall: number; anomaly_recall: number; share_predicted_anomaly: number; roc_auc: number; roc_auc_ci95_low: number; roc_auc_ci95_high: number; pr_auc_anomaly_positive: number; mean_p_anomaly_normal_windows: number; mean_p_anomaly_anomaly_windows: number; mean_p_gap_anomaly_minus_normal: number; anomod_test: number; drop_lo2_minus_anomod: number }
export interface Lo2Response {
  sample: { run_id: string; scenarios: number; error_scenarios: number; megabytes: number; windows: number; normal_windows: number; anomaly_windows: number }
  feature_mapping: Record<string, { features_built: number; features_total: number; used_features_built: number; used_features_total: number }>
  primary_setting: string
  models: Lo2Model[]
  all_settings: { model: string; feature_set: string; system_flag: string; balanced_accuracy: number; normal_recall: number; anomaly_recall: number; share_predicted_anomaly: number }[]
  distribution_shift: { feature: string; lo2_class: string; anomod_train_median: number; anomod_train_p995: number; lo2_median: number; share_lo2_above_anomod_p995: number }[]
  log_volume_check: { raw_log_line_count_auc: number; raw_log_error_count_auc: number; high_auc_models: string[]; rank_correlation_with_log_line_count: number[]; n_high_auc_tracking_log_volume: number }
  summary_markdown: string
}
