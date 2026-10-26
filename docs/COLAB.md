# Run the whole workflow in Google Colab

Written for: the person who will run the FailSight notebook on Google Colab (no installation needed on their computer).

The notebook `notebooks/FailSight_Workflow.ipynb` runs the whole workflow with its proofs, section by section: the data, preprocessing, the 80:20 split, imbalance handling (SMOTE), cross-validation, the classical machine-learning algorithms, the hybrid deep-learning models (AnoMod, LO2 and both together, each with both label versions), honesty checks, explanations, our testbed, LO2, the outside tests, the proof report (P1 to P24) and the limits. Every section starts with a plain-English "what we do and why" and ends with a "what you should see" line.

## Before you start (once, on the computer with the repository)

The notebook clones the code from GitHub, so the newest code must be on GitHub first.

1. Open a terminal in the repository folder and send the latest commit to GitHub:
   ```
   git status
   git push origin main
   ```
   (If `git status` shows changed files, run `git add -A` and `git commit -m "colab update"` first.)
2. The data bundle is already made: **`failsight_bundle.zip`** in the repository root (about 9 MB). To make it again: `python scripts\make_colab_bundle.py`. It holds only small files: the AnoMod window tables, split and fold files, the testbed windows and frozen detector, the LO2 windows and templates (all 33 runs), every result file, `proofs.json` and the figures. The raw AnoMod / LO2 recordings are not needed.

## Exact steps on Google Colab

1. **Put the bundle on Google Drive.** Unzip `failsight_bundle.zip`. In Google Drive create the folder `MyDrive/FailSight_bundle/` and upload the *contents* of the unzipped folder into it (the folders `inputs`, `processed`, `models`, `report_outputs`). Check: `MyDrive/FailSight_bundle/inputs/SN_windows.parquet` must exist.
2. **Open the notebook** in Colab: <https://colab.research.google.com/github/naunidhsingh04/FailSight/blob/main/notebooks/FailSight_Workflow.ipynb>
   (or in Colab: File > Open notebook > GitHub tab > `naunidhsingh04/FailSight` > `notebooks/FailSight_Workflow.ipynb`).
3. **Runtime > Run all.** When Colab asks for Google Drive access, click *Connect to Google Drive* and allow it (section 0 mounts Drive and reads the bundle). A GPU is optional (Runtime > Change runtime type > T4 GPU makes the small live hybrid a little faster; the CPU is fine).
4. Wait for the last cell. The whole notebook is built to finish in about 30 minutes or less on a fresh runtime. Cells print `missing files (this part is skipped)` instead of failing if a file is absent.
5. Read the last sections: **13 Proof report** (totals, a per-section table and a check that the notebook's numbers equal the proof numbers) and **14 Limits**.
6. To keep a copy of the output: File > Download > Download .ipynb, or File > Print and save as PDF.

`FAST = True` is the required mode (set in the first code cell): live parts are small (5-fold cross-validation, 100 trees, 3 epochs) and the long results are read from saved result files. `FAST = False` is not needed for any claim.

## If something goes wrong

| Problem | What to do |
|---|---|
| A cell says "missing files (this part is skipped)" | The bundle was not found in Drive. Section 0 looks for the folder that holds `inputs/SN_windows.parquet` anywhere up to three levels below My Drive (so `FailSight_bundle/failsight_bundle/inputs/...` also works). Check that file exists, then Runtime > Run all again. |
| "Mounted at /content/drive" never appears | Allow the Drive pop-up (it can open in a second window) and run section 0 again. |
| Old numbers appear (for example LO2 with 5 runs) | The old code is still on GitHub: run `git push origin main`, then in Colab Runtime > Disconnect and delete runtime, and run all again. |
| Out of memory | Runtime > Restart session and run all; do not switch `FAST` to False. |
| A library error right after section 0 | Run section 0 again; it installs only missing libraries (never TensorFlow or NumPy). |

## What to expect per section (and roughly how long)

| Section | What you will see | Time |
|---|---|---|
| 0 Setup | the repository is cloned, only missing libraries are installed, Drive is mounted, versions and a file tree | 1 to 3 min |
| 1 The data | window counts, class labels, feature groups, data types | under 1 min |
| 2 Preprocessing | each step on real rows (before / after), fitted on training windows only, plus a leak check | under 1 min |
| 3 The 80:20 split | train / test counts, no window or block in both parts, class counts | under 1 min |
| 4 Imbalance | SMOTE before / after per fold (`b:` / `a:` / `hold:`), class weights, ADASYN size comparison, charts | 1 min |
| 5 Folds | no block shared across folds, every class in every fold | under 1 min |
| 6 Classical algorithms | five algorithms trained live, test scores with 95% ranges next to the saved full results | 2 to 4 min |
| 7 Hybrids | wiring diagrams, one hybrid trained live in its two stages (3 epochs each: the first algorithm, then the second one using its values), saved results of all of them with ranges and the comparison with the tree models, and the combined AnoMod + LO2 table | 2 to 4 min |
| 8 Honesty checks | overfitting gaps, shuffled-label control, leakage, missingness, run identification, each with its badge | under 1 min |
| 9 Explainability | SHAP for XGBoost on real test windows, LIME for one window, attention of the live hybrid | 2 to 4 min |
| 10 Our testbed | data, fault types, whole-run split, held-out results with ranges, controls, three live demos, one run drawn, a replay | 1 min |
| 11 LO2 | all 33 runs (26 training runs balanced with SMOTE, 7 untouched test runs), both label versions for each hybrid, other families, controls including the shortcut table, the separation / SMOTE checks, the no-transfer note | under 1 min |
| 12 External tests | cross-system and RCAEval results and the honest conclusion | under 1 min |
| 13 Proof report | totals and a per-section table, plus a check that the notebook's numbers equal the proof numbers | under 1 min |
| 14 Limits | the limits in plain words and a "what we claim / do not claim" table | under 1 min |

A pre-executed copy with all outputs is in `notebooks/FailSight_Workflow_executed.ipynb` (and `.html`); it was executed locally in a fresh virtual environment without the raw data folders and without the models folder (proof section P23). Open it first if you only want to look at the results.

## Honesty rules of the notebook

* Every number is computed in a cell or read from a named result file (the file name is printed under each table). Nothing is typed in by hand.
* The test part is only scored, never used to choose anything. SMOTE is fitted on training parts only; test parts keep the counts they were recorded with. All test data is real recorded data.
* Nothing in it shows that FailSight works on other systems, forecasts failures, or gives a time-until-failure.
* The earlier AnoMod models were checked with 10-fold cross-validation and the hybrid models with 5-fold. All AnoMod models, hybrids included, are scored on the same 647 test windows.
* LO2 is one OAuth2 service with injected HTTP errors. Its binary task scores high because the errors leave clear traces in logs and metrics (controls in section 11 show it is not a warm-up or run-length shortcut); that says nothing about other systems.
