# The LO2 slice

Written for: examiners who ask what part of LO2 was used and why.

LO2 (Zenodo 10.5281/zenodo.14265858) is a set of 33 runs of an OAuth2 service (`light-oauth2`: seven micro-services, a database, a Locust load generator, Prometheus metrics). Each run executes 54 scenarios back to back (one `correct` scenario and 53 error scenarios); a scenario's folder name is its label. The data are stored read-only on this computer in the folder `LO2` next to the repository (`lo2_raw` holds the downloaded run files, `lo2_tar` the same runs unpacked); SHA-256 of every file was recorded before and after (`outputs/lo2_hash_before.json` / `lo2_hash_after.json`; unchanged).

## Which files, and why

* Every downloaded run is used: **33 whole runs (887.4 MB, 100.0% of the download)**. An earlier version used only 5 runs under a 150 MB cap; that left 13 healthy test windows and unreliable results, so the cap was lifted.
* Whole runs only (no run is ever cut), each with all 54 scenarios (normal and anomalous), covering the 33 collection dates.
* The split was made first, by whole runs: 7 runs (20%, chosen with seed 42) are the test part, the other 26 are the training part. No run and no window is in both parts (proof P21, check 1). The training part is balanced with SMOTE (inside every training fold and for the final model); the test part is never resampled (proof P24).

| run | file size (MB) | role | scrape rows (windows) | normal | anomalous |
|---|---|---|---|---|---|
| 1719592986 | 26.61 | train | 172 | 13 | 159 |
| 1719785649 | 26.88 | train | 172 | 13 | 159 |
| 1719983645 | 27.21 | train | 172 | 13 | 159 |
| 1720178048 | 26.47 | train | 172 | 13 | 159 |
| 1720372448 | 26.78 | train | 172 | 13 | 159 |
| 1720570451 | 26.42 | train | 172 | 13 | 159 |
| 1720764845 | 27.35 | train | 172 | 13 | 159 |
| 1720959241 | 27.12 | train | 172 | 13 | 159 |
| 1721157245 | 26.15 | test | 172 | 13 | 159 |
| 1721351650 | 26.44 | test | 172 | 13 | 159 |
| 1721546043 | 26.97 | train | 172 | 13 | 159 |
| 1721744041 | 27.21 | train | 172 | 13 | 159 |
| 1721938441 | 27.27 | train | 172 | 13 | 159 |
| 1722132848 | 26.71 | train | 172 | 13 | 159 |
| 1722330847 | 27.53 | train | 172 | 13 | 159 |
| 1722543242 | 27.14 | test | 172 | 13 | 159 |
| 1722741254 | 27.15 | train | 172 | 13 | 159 |
| 1722935656 | 27.14 | test | 172 | 13 | 159 |
| 1723130055 | 27.38 | train | 172 | 13 | 159 |
| 1723328076 | 26.11 | test | 172 | 13 | 159 |
| 1723522475 | 26.94 | train | 172 | 13 | 159 |
| 1723716877 | 27.94 | train | 172 | 13 | 159 |
| 1723914878 | 26.51 | train | 172 | 13 | 159 |
| 1724109283 | 27.39 | train | 172 | 13 | 159 |
| 1724303680 | 26.58 | train | 172 | 13 | 159 |
| 1724501678 | 27.10 | train | 172 | 13 | 159 |
| 1724696081 | 26.47 | test | 172 | 13 | 159 |
| 1724890485 | 26.54 | train | 172 | 13 | 159 |
| 1725088482 | 27.19 | train | 172 | 13 | 159 |
| 1725282876 | 27.33 | train | 172 | 13 | 159 |
| 1725477283 | 27.26 | train | 172 | 13 | 159 |
| 1725675301 | 26.75 | test | 172 | 13 | 159 |
| 1725869703 | 25.33 | train | 172 | 13 | 159 |

Total: 33 runs, 887.4 MB, 5676 windows (429 normal, 5247 anomalous). The list of files with the reasons is `report_outputs/lo2_slice_manifest.csv`.

## Windows and class counts

A window is one 5-second scrape row of the LO2 data set (the finest window LO2 offers; logs are aggregated over the 5 seconds before the scrape). Features come from LO2's own signals: per log source (7 services, the database, Locust) the number of lines, continuation (stack-trace) lines, DEBUG / INFO / WARN / ERROR counts, error and warn rates; the total log volume; counts of the 48 most frequent **Drain-style log templates** (a template is a log line with numbers, ids and quoted text masked; templates were mined on the training runs only, test lines that match none go to an `other` count); and 478 metric families (the mean over the series of each Prometheus metric name, counters turned into per-second rates).

| class | train | test |
|---|---|---|
| Normal | 338 | 91 |
| HTTP400 | 2340 | 630 |
| HTTP401 | 390 | 105 |
| HTTP404 | 1404 | 378 |

Total 5676 windows: 4472 train, 1204 test. Binary label: Normal (the `correct` scenario) versus Anomaly; the multi-class label is Normal / HTTP 400 / 401 / 404 (from the scenario name).

## Folds

The training runs give 5 folds of WHOLE runs: no run is in two folds, every class is in every validation fold, and no context window crosses a fold border (proof P22). The test part is whole runs that no fold has seen.

## Is the binary task easy because of a shortcut?

In every LO2 run the normal (`correct`) scenario is the first of the 54 scenarios and lasts 60 s instead of 10 s. That could be a warm-up or run-length shortcut. The same random forest (trained on the training runs, scored on the untouched test runs; `report_outputs/lo2_shortcut_check.csv`; a diagnostic that changed nothing) shows it is not:

| inputs / subset | features | balanced accuracy (or share correct) | ROC-AUC | test windows |
|---|---|---|---|---|
| all inputs | 134 | 1.000 | 1.000 | 1204 |
| log counts and rates only | 50 | 0.992 | 0.993 | 1204 |
| log-template counts only | 49 | 0.959 | 0.994 | 1204 |
| metric components only | 32 | 0.998 | 1.000 | 1204 |
| log volume only | 1 | 0.663 | 0.787 | 1204 |
| WARN / ERROR log counts and rates only | 20 | 0.956 | 0.986 | 1204 |
| normal windows from the first 30 s of the normal scenario (recall) | 134 | 1.000 | - | 42 |
| normal windows from the late part of the normal scenario (recall) | 134 | 1.000 | - | 49 |
| anomalies of scenarios 2-3 (recall) | 134 | 1.000 | - | 42 |
| anomalies of scenarios 4-10 (recall) | 134 | 1.000 | - | 147 |
| anomalies of scenarios 11-54 (recall) | 134 | 1.000 | - | 924 |
| matched subset (late normal windows vs anomalies of scenarios 2-4), all inputs | 134 | 1.000 | 1.000 | 112 |
| matched subset, metric components only | 35 | 0.984 | 1.000 | 112 |

Normal windows from the late part of the scenario are recognised as well as the first ones, anomalies from scenarios 2-3 as well as from scenario 54, and a matched subset (late normal windows against anomalies of scenarios 2-4, same warm-up state) is still separated. The error scenarios leave clear traces: WARN / ERROR log lines alone give about 96%, metric changes about 99%. The high binary score therefore reflects a real, easy-to-see fault signature, within one service with injected HTTP errors. The HTTP 400 / 401 / 404 task is harder and is the more informative LO2 result.

## Separation from AnoMod

LO2 is trained completely separately: its own windows (`scripts/lo2_hybrid_data.py`, `scripts/lo2_data.py`), its own preprocessing fitted on LO2's training part, its own models (`models/hybrid/lo2/`) and results (`report_outputs/hybrid_lo2_*`, `lo2_full_*`). No AnoMod weight, scaler, threshold or feature is used, and no AnoMod and LO2 window is ever mixed in a split or training set. Proof P21 has four explicit PASS / FAIL checks (no window or run in both parts, natural test counts, SMOTE before/after per fold, physical separation and a grep for AnoMod artefacts in the LO2 scripts).

## What this says and does not say

* These results are about training on LO2's own data. They say nothing about transfer from AnoMod to LO2.
* Controls: shuffled labels, a log-volume-only baseline and a run-identification check (the test runs are never in the training data, so run identity cannot help).
* The earlier external test (`lo2_results_summary.md`: saved AnoMod models applied unchanged to one LO2 run) is separate and unchanged.
