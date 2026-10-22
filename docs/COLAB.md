# Run the whole workflow in Google Colab

Written for: someone who wants to repeat the FailSight workflow without installing anything.

The notebook `notebooks/FailSight_Workflow.ipynb` runs the whole workflow with its proofs, section by section: the data, preprocessing, the 80:20 split, imbalance handling, cross-validation, the classical machine-learning algorithms, the hybrid deep-learning models, honesty checks, explanations, our testbed, the LO2 slice, the outside tests, the proof report and the limits. Every section starts with a plain-English "what we do and why" and ends with a "what you should see" line.

## Exact steps

1. **Make the data bundle** (once, on the computer that has the repository):
   `python scripts/make_colab_bundle.py` creates `failsight_bundle.zip` in the repository root (about 20 MB; the script prints its size and contents). It holds only small files: the AnoMod window tables, the split and fold files, the testbed windows and frozen detector, the LO2 slice windows and templates, every result file, `proofs.json` and the figures. The raw AnoMod / LO2 recordings are not needed.
2. **Upload the contents of the zip to Google Drive.** Unzip it and put the *contents* (the folders `inputs`, `processed`, `models`, `report_outputs`) into `MyDrive/FailSight_bundle/` so that, for example, `MyDrive/FailSight_bundle/inputs/SN_windows.parquet` exists.
3. **Open the notebook:** <https://colab.research.google.com/github/naunidhsingh04/FailSight/blob/main/notebooks/FailSight_Workflow.ipynb>
4. **Runtime > Run all.** Allow Drive access when asked (section 0 mounts Google Drive and reads the bundle). A GPU is optional: Runtime > Change runtime type > T4 GPU only makes the 3-epoch live hybrid a little faster.
5. Without the bundle the notebook still runs: sections that need the window tables say "missing files (this part is skipped)" and the rest (result tables read from the repository) carry on.

`FAST = True` is the required mode (set in the first code cell): live parts are small (5-fold cross-validation, 100 trees, 3 epochs) and the long results are read from saved result files. `FAST = False` is not needed for any claim.

## What to expect per section (and roughly how long)

| Section | What you will see | Time |
|---|---|---|
| 0 Setup | the repository is cloned, only missing libraries are installed (never TensorFlow or NumPy), Drive is mounted, versions and a file tree with one line per folder | 1 to 3 min |
| 1 The data | window counts, class labels, feature groups, a table of data types (coverage files and images are not used) | under 1 min |
| 2 Preprocessing | each step on real rows (before / after), fitted on training windows only, plus a leak check | under 1 min |
| 3 The 80:20 split | train / test counts, no window or block in both parts, class counts | under 1 min |
| 4 Imbalance | SMOTE before / after per fold (`b:` / `a:` / `hold:`), class weights, ADASYN size comparison, bar charts | 1 min |
| 5 Folds | no block shared across folds, every class in every fold | under 1 min |
| 6 Classical algorithms | five algorithms trained live (5 folds, 100 trees), test scores with 95% ranges next to the saved full results | 2 to 4 min |
| 7 Hybrids | wiring diagrams, one hybrid trained live for 3 epochs, saved results of all five with ranges and the comparison with the tree models | 2 to 4 min |
| 8 Honesty checks | overfitting gaps, shuffled-label control, leakage, missingness, run identification, each with the badge from the proof report | under 1 min |
| 9 Explainability | SHAP for XGBoost on real test windows, LIME for one window, attention of the live hybrid | 2 to 4 min |
| 10 Our testbed | data, fault types, whole-run split, held-out results with ranges, controls, three live demos, one run drawn, a replay through the frozen detector | 1 min |
| 11 LO2 | slice facts, window counts, hybrids and other families, controls, the four separation / SMOTE checks, the no-transfer note | under 1 min |
| 12 External tests | cross-system and RCAEval results and the honest conclusion | under 1 min |
| 13 Proof report | totals and a per-section table, plus a check that the notebook's numbers equal the proof numbers | under 1 min |
| 14 Limits | the limits in plain words and a "what we claim / do not claim" table | under 1 min |

The whole notebook is built to finish in at most 30 minutes on a fresh Colab runtime. A pre-executed copy with all outputs is in `notebooks/FailSight_Workflow_executed.ipynb` (and `.html`); it was executed locally in a fresh virtual environment without the raw data folders and without the models folder (proof section P23).

## Honesty rules of the notebook

* Every number is computed in a cell or read from a named result file (the file name is printed under each table). Nothing is typed in by hand.
* The test part is only scored, never used to choose anything. All test data is real recorded data.
* Nothing in it shows that FailSight works on other systems, forecasts failures, or gives a time-until-failure.
* Original AnoMod models: 10-fold. Hybrids and LO2 work: 5-fold, chosen because of the time limit. All AnoMod models are scored on the same 647 test windows.
