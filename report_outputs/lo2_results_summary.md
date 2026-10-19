# LO2 external test: results summary

All numbers are read from the CSVs in `report_outputs/` (lo2_sample_manifest, lo2_feature_mapping, lo2_test_metrics, lo2_vs_anomod, lo2_distribution_shift). No model was trained, tuned or refitted on LO2.

## Sample

- Run 1719592986 (earliest run), scenarios in start order 1..24: 24 whole scenarios (23 error, 1 `correct`), 98.6 MB of selected files (service logs, MySQL log, shared node_exporter metrics).
- Windows built with the AnoMod 30 s / 10 s rule per scenario: 28 windows = 5 Normal (all from the single `correct` scenario) + 23 Anomaly (one window per error scenario, because error scenarios last only 10-18 s).

## Feature mapping loses most features

- Variant A logs only (requested): 3 of 155 AnoMod features can be built (1.9%); of the 124 features the models actually use, 3 (2.4%). The rest are missing and imputed by the preprocessor.
- Variant B logs + shared node metrics (extra): 60 of 155 AnoMod features can be built (38.7%); of the 124 features the models actually use, 43 (34.7%). The rest are missing and imputed by the preprocessor.
- API features: 8 of 8 cannot be built (LO2 has no per-request status, latency or size). Metric features: 144 of 144 missing in variant A.

## Results (binary labels, final models unchanged)

- Primary setting (variant A, system flag TT): balanced accuracy is 0.500 to 0.500 for all 18 models; 16 models label every window Anomaly (Normal recall 0), 2 label every window Normal (anomaly recall 0), 0 give a mix.
- AnoMod test balanced accuracy of the same models: 0.588 to 0.774; mean drop LO2 minus AnoMod: -0.163 (best AnoMod model XGB: 0.774 -> 0.500).
- Bootstrap 95% CI of balanced accuracy (scenarios resampled within class): 0.500 to 0.500 across models. The interval collapses to a point because the predictions are constant; it does not mean the estimate is precise (see caveats).
- variant B, flag TT: balanced accuracy 0.500 to 0.500; models above 0.5: 0 of 18; models predicting Anomaly for every window: 18.
- variant A, flag SN: balanced accuracy 0.500 to 0.500; models above 0.5: 0 of 18; models predicting Anomaly for every window: 18.
- variant B, flag SN: balanced accuracy 0.500 to 0.500; models above 0.5: 0 of 18; models predicting Anomaly for every window: 18.
- Mean predicted P(Anomaly) on Normal vs Anomaly windows (variant A, flag TT) differs by at most 0.102 for any model: the models do not separate the two classes.

## Why: distribution shift of the three log features that can be built

- log_line_count, LO2 Normal: median 68,713 vs AnoMod TT train median 7 (99.5th percentile 28,804); 100% of LO2 windows exceed the AnoMod TT 99.5th percentile.
- log_line_count, LO2 Anomaly: median 19,598 vs AnoMod TT train median 7 (99.5th percentile 28,804); 0% of LO2 windows exceed the AnoMod TT 99.5th percentile.
- log_error_count, LO2 Normal: median 1,009 vs AnoMod TT train median 0 (99.5th percentile 479); 100% of LO2 windows exceed the AnoMod TT 99.5th percentile.
- log_error_count, LO2 Anomaly: median 309 vs AnoMod TT train median 0 (99.5th percentile 479); 0% of LO2 windows exceed the AnoMod TT 99.5th percentile.
- log_warning_count, LO2 Normal: median 0 vs AnoMod TT train median 0 (99.5th percentile 31); 0% of LO2 windows exceed the AnoMod TT 99.5th percentile.
- log_warning_count, LO2 Anomaly: median 0 vs AnoMod TT train median 0 (99.5th percentile 31); 0% of LO2 windows exceed the AnoMod TT 99.5th percentile.

## Caveats

- 5 Normal windows from ONE scenario and 23 Anomaly windows from 23 scenarios of ONE run: far too little for any statistical conclusion; confidence intervals cannot represent run-to-run variation.
- LO2 labels are scenario-level (`correct` still contains failing requests); LO2 logs are DEBUG-level Java logs, so log counts are orders of magnitude above AnoMod's.
- Log counts also encode the scenario length (about 60 s for `correct`, 10-18 s for error scenarios) because windows hold whatever the scenario produced.
- The AnoMod models use system_is_TT; LO2 is assigned TT (primary) or SN (sensitivity), neither is correct.

Files: lo2_sample_manifest.csv, lo2_feature_mapping.csv, lo2_test_metrics.csv, lo2_vs_anomod.csv, lo2_distribution_shift.csv, lo2_window_predictions_A_TT.csv, lo2_inspection.txt, figures/step4/.
