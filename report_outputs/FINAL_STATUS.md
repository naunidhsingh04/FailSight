# FailSight: final status

Written 2026-10-21 at the end of the final task list (started 13:53, finished about 16:40: roughly 2.8 hours of laptop time, inside the 5-hour limit). Every number below is taken from the files named next to it. The saved models, the frozen testbed configuration (`testbed_frozen_config.json`), the stage 5 and live-demo results and the demo seeds (9001-9003) were not changed, and `python -m testbed.live_demo` was not run again.

## Tasks

| # | Task | Result | Commit |
|---|---|---|---|
| 1 | Housekeeping | **Done.** `testbed/runs/_live_*`, `testbed/runs/_invalid/`, `outputs/` and `external/` are ignored. Checked every tracked file: no raw event data, models, parquet or other large files are tracked (largest tracked file 427 KB). | `75ff5f6` chore: ignore temporary demo folders |
| 2 | Connect your app | **Done.** `docs/INTEGRATION.md`; Python ASGI middleware, Node/Express middleware (`agent/node/failsight-express.js`), nginx/app access-log tailer (`python -m failsight_agent.tail`); web **Connect** page with copy buttons, connection state (events received, last event, learning/live) and a webhook form, in Simple and Detailed modes; optional per-app API key on the ingestion endpoints (`FAILSIGHT_API_KEYS`, off by default) and a Slack-compatible alert webhook (`FAILSIGHT_WEBHOOK_URL`); one test each plus tests for the tailer and the Node middleware; proofs **P19** (5 PASS, 1 REPORTED ACCURATELY). The monitor labels every app other than the testbed shop *unvalidated*. | `071c791` docs: connect-your-app page and integration guide |
| 3 | RCAEval external test | **Done (testing only).** Downloaded RE1-OB, RE1-SS, RE1-TT (390 MB in total) into `external/rcaeval/` (git-ignored); 150 cases (10 per fault type per system); every case has an explicit `inject_time.txt`, so no normal-period assumption was needed. The testbed's robust z-score and IsolationForest (thresholds frozen from the testbed train runs) were applied to generic aggregates. Results, per-case results, feature mapping and a summary are in `report_outputs/rcaeval_*`; proofs **P18** (2 PASS, 1 REPORTED ACCURATELY). The supervised testbed models could not be applied (their 78 inputs are testbed-specific) and nothing was trained or tuned on this data. | `0e4e70f` eval: RCAEval external test |
| 4 | Results index | **Done.** `report_outputs/INDEX.md` (every result file, description, where it is used, final / intermediate / log); nothing moved or renamed. | `5b3d32f` docs: results index |
| 5 | Full test pass | **Done, no regression found.** See "Tests" below. The full proof report was re-run (34 min, 2018 s). | `aca3fba` app: final test pass |
| 6 | Clean-clone check | **Done; problems found and fixed.** See "Clean-clone findings" below. | `611fbd7`, `b6dbd64` docs: clean-clone fixes |
| 7 | Final status | **Done** (this file). | docs: final status |

No task failed or was skipped.

## Proof totals (full run of `python scripts/proofs.py`, 2026-10-21, 2018 s)

**65 PASS, 4 WARN, 0 FAIL, 9 REPORTED ACCURATELY** (19 sections).

| Section | Status | | Section | Status |
|---|---|---|---|---|
| P1 data | PASS | | P11 cross-system test | REPORTED ACCURATELY (no generalisation) |
| P2 split | PASS | | P12 live demo validation | REPORTED ACCURATELY (2 of 2 faults detected, 1 false alarm, 0 of 24 fault windows missed) |
| P3 SMOTE | PASS | | P13 missingness and shortcuts | PASS |
| P4 folds | PASS | | P14 run identification (AnoMod) | WARN |
| P5 models recomputed | WARN (tie noise in ranking metrics) | | P15 fault times | REPORTED ACCURATELY (none recorded) |
| P6 overfitting | PASS | | P16 testbed held-out validation | REPORTED ACCURATELY |
| P7 leakage audit | WARN (C3 time overlap, measured effect) | | P17 testbed run identification | PASS |
| P8 explainability | PASS | | P18 RCAEval external test | REPORTED ACCURATELY |
| P9 LO2 | REPORTED ACCURATELY (models did not transfer) | | P19 integration | REPORTED ACCURATELY (checks PASS, scope stated) |
| P10 reproducibility | PASS | | | |

## Tests (final state, 2026-10-21)

| Suite | Result |
|---|---|
| Backend and repository tests (`python -m pytest tests app/backend/tests`) | 128 passed (this machine, 144 s) |
| Playwright: Chromium, Firefox, WebKit at 375, 768, 1280, 1920 px (pages, flows, axe accessibility) | 234 passed, 0 failed, 0 flaky |
| Playwright replay spec (start, pause, alert, explanation, proofs re-run), one worker | 15 passed |
| Lighthouse, 9 pages, mobile and desktop | accessibility 100, best practices 100, SEO 100 on every page; performance 65 to 100 (lowest: Limits on the mobile profile, 65; most pages 77-100) |
| Clean clone, no data or models (`python -m pytest tests app/backend/tests`) | 47 passed, 81 skipped (with reasons), 0 failed |

## Clean-clone findings (task 6)

A fresh clone of the public repository was installed exactly as the README says (venv, `pip install -r app\backend\requirements.txt -r requirements-deep.txt`, `npm install; npm run build`, `python -m app.backend.serve`).

| Finding | Fix |
|---|---|
| The backend did not import: `httpx` is imported by the new testbed endpoints but was not in `requirements.txt` (nor `psutil`, `pyarrow`, `matplotlib`, used by the testbed and its scripts). | Added to `app/backend/requirements.txt`; `httpx` is now imported lazily in the endpoint that needs it. |
| The backend refused to start without the AnoMod models and preprocessor (they are not in git), so nothing, not even the testbed pages, worked in a clean clone. | The backend now starts in **testbed-only mode**: `/api/health` says `testbed-only`, the Proofs, Testbed, Limits and Connect pages and the live testbed monitor work, AnoMod endpoints answer 503 with the exact missing file. Two tests cover it. |
| 15 tests failed or errored without data or models (the README claimed they would skip). | Tests that need AnoMod data or models now skip with a reason. |
| The README did not say what a fresh clone contains. | New README paragraph: what is and is not in git, testbed-only mode, how to reproduce the testbed from scratch (`run_generator`, `dataset`, `validate`, `freeze`, `start_testbed.bat`; about 7 hours). |
| `python -m testbed.run_generator --runs 3` ended in a traceback. | Clear message: use at least 12 runs (full plan 40). |

Verified in the clean clone: the dashboard builds; the backend starts in testbed-only mode; Proofs, Testbed and the UI pages answer 200 and the AnoMod endpoints 503; `run_generator --runs 12 --dry` prints the plan; `start_testbed.bat` / the launcher stops with "the frozen detector is missing: run python -m testbed.freeze first". Not verified in the clone: a full testbed run (7 hours) and anything needing AnoMod data or models.

**Things that should not be public, and what was done**

- Personal Windows paths (`C:\Users\<user>\...`) in two tracked logs, `report_outputs/preprocess_log.txt` and `results/v1_16epoch/log.txt`: redacted in place (the logs are otherwise unchanged and no code reads them). **They remain in the git history** (it was not rewritten; no force-push).
- The git commit metadata carries the author's e-mail address; that is how git works and cannot be removed without rewriting history.
- No keys, tokens, passwords, raw data, models or large files are tracked (searched with `git grep`).

## Key result files for the report

| What | File |
|---|---|
| Testbed dataset document (counts, imbalance, 2 rows per class, figures) | `docs/TESTBED_DATASET.md`, `report_outputs/testbed_dataset_summary.csv`, `testbed_dataset_sample.csv` |
| Held-out validation (whole runs) | `testbed_run_assignment.csv`, `testbed_train_cv.csv`, `testbed_heldout_overall.csv`, `testbed_heldout_by_fault.csv`, `testbed_false_alarms.csv`, `testbed_prefailure.csv`, `testbed_shuffled_control.csv`, `testbed_runid_check.csv`, `testbed_cv_by_fault.csv` |
| Frozen detector and live validation | `testbed_frozen_config.json`, `testbed_live_validation.csv`, `testbed_live_schedule_<seed>.json` (written before each run), `testbed_live_schedule_actual_<seed>.csv`, `testbed_live_windows_<seed>.csv` |
| RCAEval external test | `rcaeval_summary.md`, `rcaeval_external_results.csv`, `rcaeval_case_results.csv`, `rcaeval_feature_mapping.csv` |
| AnoMod results | `test_metrics.csv`, `test_bootstrap_ci.csv`, `paired_bootstrap.csv`, `leaderboard_label_binary.csv`, `leaderboard_label_level.csv`, `thresholds.json`, `results_summary.md`, `lo2_results_summary.md`, `lo2_test_metrics.csv` |
| AnoMod validity checks | `leakage_audit.csv` / `.json`, `data_integrity.csv`, `missingness_check.csv`, `run_identification.csv`, `cross_system_results.csv`, `fault_times.csv`, `early_warning_feasibility.md` |
| Proof report | `proofs.json` (also the Proofs page); index of all files: `INDEX.md` |
| Integration and presentation | `docs/INTEGRATION.md`, `docs/DEMO_SCRIPT.md`, `README.md` |

## Limitations the project states

**AnoMod (two systems)**
- Labels are per run and the data contain no fault start or end times, so the AnoMod test cannot show that failures are really detected; the run-identification rule reproduces the saved binary XGBoost's balanced accuracy (0.773 against 0.774) (P14).
- On healthy windows the best binary model's recall is only 0.556, so it raises many false alarms; the 36 healthy test windows come from two runs.
- Training and test windows overlap in time (audit item C3, WARN; measured effect -0.036 binary, +0.010 five-class).
- The best model was chosen on the test set; its lead is optimistic. The audit's file-hash baseline was first recorded during the audit, so it proves no change since then, not before.
- The saved models rely on the machine metrics (masking API or log values changes nothing); missing metric values push the call toward Anomaly, which is why the Console has a "Not enough data to judge" state.
- The Console replays recorded AnoMod data; it is not a live system, and the model detects current anomalies, it does not forecast.
- The external LO2 test failed (balanced accuracy 0.500 for every model, 3 of 124 features buildable, 28 windows; high ROC-AUC explained by log volume); the cross-system test (train on one system, test on the other) shows no generalisation (best cell 0.521, none above chance).
- Early warning and within-run (segment) labels cannot be built from AnoMod (no fault times).

**Testbed (our own shop)**
- One app, one machine, synthetic traffic and injected faults; the faults are ones we wrote.
- The held-out test has 12 runs and 15 faults; most fault types have 1 to 3 test instances, so per-type numbers are unreliable (marked, rule: at least 5 instances); the 2 fault-free test runs cover 0.25 hours, so false-alarm rates are coarse.
- The three live demos are a demonstration, not a statistic (2 faults detected, 1 false alarm, 0 of 24 fault windows missed).
- Transfer to other apps has not been shown; any other app is labelled *unvalidated* and needs its own learning period and its own check. A claim that FailSight works on any app is not supported.

**External test (RCAEval, testing only)**
- 150 cases of three public systems, metrics only; the unsupervised detectors were applied with thresholds frozen from another application: the robust z-score found 20-36% of the faults (placebo at most 10%); IsolationForest found 68-86% but also alerted at random times in 31-79% of the cases, so most of its detections are not about the fault; the supervised testbed models could not be applied.

**Engineering**
- Lighthouse performance is below 90 on several pages (lowest: Limits on mobile, 65); the proof report takes about 35 minutes; the full testbed pipeline takes about 7 hours of laptop time and the generated runs will not reproduce our numbers exactly.
- A fresh clone has no data or models (not in git); the AnoMod replay console and Results page need the AnoMod models.
- The git history still contains personal paths in two old logs and the author's e-mail address.
