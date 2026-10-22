# The LO2 slice

Written for: examiners who ask what part of LO2 was used and why.

LO2 (Zenodo 10.5281/zenodo.14265858) is a set of 33 runs of an OAuth2 service (`light-oauth2`: seven micro-services, a database, a Locust load generator, Prometheus metrics). Each run executes 54 scenarios back to back (one `correct` scenario and 53 error scenarios); a scenario's folder name is its label. The data are stored read-only on this computer in the folder `LO2` next to the repository (`lo2_raw` holds the downloaded run files, `lo2_tar` the same runs unpacked); SHA-256 of every file was recorded before and after (`outputs/lo2_hash_before.json` / `lo2_hash_after.json`; unchanged).

## Which files, and why

* Cap: at most 150 MB of the downloaded run files (about 17% of the 887 MB). Each run file is about 26 MB, so the cap allows **5 whole runs (131.8 MB, 14.9% of the download)**. Six runs would need 157 MB, so the aim of at least 6 runs cannot be met under the cap; the cap wins, and every LO2 result is marked *unreliable: very small sample* (fewer than 6 runs).
* Whole runs only (no run is ever cut), evenly spread over the 33 collection dates (positions 0, 8, 16, 24 and 32 of the sorted run list) so that the slice covers drift over time, each with all 54 scenarios (normal and anomalous).
* The split was made first, by whole runs: one run (chosen with seed 42) is the 20% test part, the other four are the training part. No run and no window is in both parts (proof P21, check 1).

| run | file size (MB) | role | scrape rows (windows) | normal | anomalous |
|---|---|---|---|---|---|
| 1719592986 | 26.61 | train | 172 | 13 | 159 |
| 1721157245 | 26.15 | train | 172 | 13 | 159 |
| 1722741254 | 27.15 | train | 172 | 13 | 159 |
| 1724303680 | 26.58 | test | 172 | 13 | 159 |
| 1725869703 | 25.33 | train | 172 | 13 | 159 |

Total: 5 runs, 131.8 MB, 860 windows (65 normal, 795 anomalous). The list of files with the reasons is `report_outputs/lo2_slice_manifest.csv`.

## Windows and class counts

A window is one 5-second scrape row of the LO2 data set (the finest window LO2 offers; logs are aggregated over the 5 seconds before the scrape). Features come from LO2's own signals: per log source (7 services, the database, Locust) the number of lines, continuation (stack-trace) lines, DEBUG / INFO / WARN / ERROR counts, error and warn rates; the total log volume; counts of the 48 most frequent **Drain-style log templates** (a template is a log line with numbers, ids and quoted text masked; templates were mined on the training runs only, test lines that match none go to an `other` count); and 478 metric families (the mean over the series of each Prometheus metric name, counters turned into per-second rates).

| class | train | test |
|---|---|---|
| Normal | 52 | 13 |
| HTTP400 | 360 | 90 |
| HTTP401 | 60 | 15 |
| HTTP404 | 216 | 54 |

Total 860 windows: 688 train, 172 test. Binary label: Normal (the `correct` scenario) versus Anomaly; the multi-class label is Normal / HTTP 400 / 401 / 404 (from the scenario name).

## Folds

Four training runs cannot give five whole-run folds. The 5 folds are therefore block-aware inside the training runs: a group is a run + scenario + 3-window block; no group is shared across folds; every class is in every validation fold; the overlap of context windows is measured and reported in proof P22. The test part is a whole run that no fold has seen, so it has no overlap with the training data.

## Why the binary labels are almost trivial here

In every LO2 run the normal (`correct`) scenario is the first of the 54 scenarios and lasts 60 s instead of 10 s. The same random forest (100 trees) was scored on subsets of the inputs (trained on the training runs, scored on the test run; `report_outputs/lo2_shortcut_check.csv`; a diagnostic that changed nothing):

| inputs | features | balanced accuracy | ROC-AUC |
|---|---|---|---|
| all inputs | 132 | 1.000 | 1.000 |
| log counts and rates only | 48 | 0.994 | 1.000 |
| log-template counts only | 49 | 0.962 | 1.000 |
| metric components only | 32 | 0.997 | 1.000 |
| log volume only | 1 | 0.713 | 0.843 |

Log counts, template counts or metric components alone each separate the normal scenario almost perfectly, so the binary LO2 task is nearly trivial and a perfect score is **not** strong evidence of fault detection. The multi-class (HTTP 400 / 401 / 404) task is much harder and is the more informative LO2 result; with 13 healthy windows and one test run it is still a very small sample.

## Separation from AnoMod

LO2 is trained completely separately: its own windows (`scripts/lo2_hybrid_data.py`, `scripts/lo2_data.py`), its own preprocessing fitted on LO2's training part, its own models (`models/hybrid/lo2/`) and results (`report_outputs/hybrid_lo2_*`, `lo2_full_*`). No AnoMod weight, scaler, threshold or feature is used, and no AnoMod and LO2 window is ever mixed in a split or training set. Proof P21 has four explicit PASS / FAIL checks (no window or run in both parts, natural test counts, SMOTE before/after per fold, physical separation and a grep for AnoMod artefacts in the LO2 scripts).

## What this says and does not say

* These results are about training on LO2's own data. They say nothing about transfer from AnoMod to LO2.
* Controls: shuffled labels, a log-volume-only baseline and a run-identification check (the test run is never in the training data, so run identity cannot help).
* The earlier external test (`lo2_results_summary.md`: saved AnoMod models applied unchanged to one LO2 run) is separate and unchanged.
