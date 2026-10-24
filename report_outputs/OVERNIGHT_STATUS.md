# Overnight status

Generated 2026-10-24 06:04 (IST). Task: FailSight overnight master task, deadline 1 PM IST on 23 Oct 2026.

## Anything that failed or was cut

* No step failed. Anything that was cut or partial is listed under its step below and under *Limitations*.

## Steps

| step | status | time (IST) | notes |
|---|---|---|---|
| 1 Storefront | **done** | 23:40-23:50 | cold start OK, break detected in 17 s, ports freed, docs/PRESENT_NOW.md extended; no code changed |
| 2 Five AnoMod hybrids | **done** | 23:56-01:01 | 5 hybrids x binary/five-class, 5-fold; no hybrid beats the best earlier model; P20 PASS. Five hybrid classes share one file instead of five |
| 3 LO2 slice + five LO2 hybrids | **done** | 00:04-01:36 | 5 whole runs, 131.8 MB (6 runs would exceed 150 MB cap); split by whole runs first; P21 four checks PASS; binary LO2 almost trivial (shortcut diagnostic); TF-IDF+SVD instead of BERT |
| 4 LO2 families b-e | **done** | 01:36-02:15 | all 23 models completed, none cut (Optuna 40 trials each) |
| 5 Colab notebook | **done** | 02:17-02:23 | executed in a fresh venv from a clean clone with the bundle: 0 errors, 0 soft failures, 8 numbers equal proofs; P23 PASS. FAST mode only |
| 6 Front end, back end, docs, checks | **done** | 02:25-03:45 | new Hybrid models page, Results top-9/Show all + LO2 tab, Limits/How it works lines, Colab box; pytest 146 passed; Playwright chromium 101 passed; screenshots in docs/screenshots |
| 7 Final | **done** | 03:35-03:50 | front end rebuilt, one quick start check (storefront and monitor open, break detected in 21 s), everything stopped, status written, pushed |

> The earlier AnoMod models were checked with 10-fold cross-validation and the hybrid models with 5-fold. All AnoMod models, hybrids included, are scored on the same 647 test windows.

## The ten hybrid models

| dataset | labels | hybrid | status | folds | test balanced accuracy (95% range) | healthy-window recall | ROC-AUC | training time | reliability |
|---|---|---|---|---|---|---|---|---|---|
| AnoMod | binary | AE_LSTM | done | 5 | 0.631 (0.458-0.959) | 0.333 | 0.822 | 2.8 min | 36 healthy test windows (small) |
| AnoMod | binary | CNN_LSTM | done | 5 | 0.609 (0.452-0.963) | 0.278 | 0.867 | 2.6 min | 36 healthy test windows (small) |
| AnoMod | binary | GNN_Transformer | done | 5 | 0.652 (0.467-0.968) | 0.361 | 0.797 | 6.4 min | 36 healthy test windows (small) |
| AnoMod | binary | Transformer_BiLSTM | done | 5 | 0.650 (0.475-0.938) | 0.333 | 0.801 | 4.8 min | 36 healthy test windows (small) |
| AnoMod | binary | VAE_Transformer | done | 5 | 0.649 (0.488-0.978) | 0.333 | 0.899 | 4.3 min | 36 healthy test windows (small) |
| AnoMod | level | AE_LSTM | done | 5 | 0.772 (0.648-0.877) | 0.583 | 0.957 | 3.7 min | 36 healthy test windows (small) |
| AnoMod | level | CNN_LSTM | done | 5 | 0.775 (0.659-0.886) | 0.500 | 0.958 | 3.3 min | 36 healthy test windows (small) |
| AnoMod | level | GNN_Transformer | done | 5 | 0.767 (0.644-0.882) | 0.583 | 0.936 | 6.7 min | 36 healthy test windows (small) |
| AnoMod | level | Transformer_BiLSTM | done | 5 | 0.717 (0.624-0.816) | 0.389 | 0.954 | 5.4 min | 36 healthy test windows (small) |
| AnoMod | level | VAE_Transformer | done | 5 | 0.703 (0.598-0.806) | 0.361 | 0.939 | 5.2 min | 36 healthy test windows (small) |
| LO2 | binary | AE_LSTM | done | 5 | 1.000 (1.000-1.000) | 1.000 | 1.000 | 5.1 min | 33 whole runs, 7 untouched test runs |
| LO2 | binary | BERT_GNN | done | 5 | 1.000 (1.000-1.000) | 1.000 | 1.000 | 10.3 min | 33 whole runs, 7 untouched test runs |
| LO2 | binary | GNN_LSTM | done | 5 | 1.000 (1.000-1.000) | 1.000 | 1.000 | 6.9 min | 33 whole runs, 7 untouched test runs |
| LO2 | binary | Multimodal_Transformer_MLP | done | 5 | 1.000 (1.000-1.000) | 1.000 | 1.000 | 11.3 min | 33 whole runs, 7 untouched test runs |
| LO2 | binary | Transformer_LSTM | done | 5 | 1.000 (1.000-1.000) | 1.000 | 1.000 | 5.0 min | 33 whole runs, 7 untouched test runs |
| LO2 | level | AE_LSTM | done | 5 | 0.709 (0.668-0.748) | 1.000 | 0.883 | 6.6 min | 33 whole runs, 7 untouched test runs |
| LO2 | level | BERT_GNN | done | 5 | 0.674 (0.632-0.717) | 1.000 | 0.850 | 17.2 min | 33 whole runs, 7 untouched test runs |
| LO2 | level | GNN_LSTM | done | 5 | 0.681 (0.643-0.718) | 1.000 | 0.860 | 5.8 min | 33 whole runs, 7 untouched test runs |
| LO2 | level | Multimodal_Transformer_MLP | done | 5 | 0.720 (0.680-0.759) | 1.000 | 0.877 | 14.8 min | 33 whole runs, 7 untouched test runs |
| LO2 | level | Transformer_LSTM | done | 5 | 0.706 (0.671-0.740) | 1.000 | 0.859 | 9.3 min | 33 whole runs, 7 untouched test runs |

## Comparison with the best earlier models (AnoMod, same 647 test windows)

* binary: best hybrid **GNN_Transformer** 0.652 (0.467-0.968); best of the 36 earlier models **XGB** 0.774 (0.496-0.996). No hybrid beats the best earlier model. The ranges overlap; the earlier best was picked on the test set, so its lead is optimistic. Reason: small tabular data (2,549 training windows, about 128 numbers each) and only 36 healthy test windows, where tree models are hard to beat.
* level: best hybrid **CNN_LSTM** 0.775 (0.659-0.886); best of the 36 earlier models **Stacking** 0.890 (0.838-0.941). No hybrid beats the best earlier model. The ranges overlap; the earlier best was picked on the test set, so its lead is optimistic. Reason: small tabular data (2,549 training windows, about 128 numbers each) and only 36 healthy test windows, where tree models are hard to beat.

## LO2 slice

* 33 whole runs, **887.4 MB** of the downloaded run files (every downloaded run is used; an earlier 5-run slice under a 150 MB cap was too small).
* Scrape-row windows (5 s): 5676 (429 normal, 5247 anomalous).
* Class counts: {'HTTP400': 2970, 'HTTP404': 1782, 'HTTP401': 495, 'Normal': 429}; split by WHOLE RUNS (written before any window existed): 4472 training windows from 26 runs, 1204 test windows from 7 untouched runs (test class counts {'HTTP400': 630, 'HTTP404': 378, 'HTTP401': 105, 'Normal': 91}).
* Folds: 5, whole runs (no run in two folds); the test part is runs no fold has seen; the training part is SMOTE-balanced, the test part untouched.
* Template embedding: TF-IDF + truncated SVD (a downloadable BERT-style encoder was not available).

## LO2 model families

* (a) hybrid deep-learning model: done AE_LSTM, BERT_GNN, GNN_LSTM, Multimodal_Transformer_MLP, Transformer_LSTM
* (b) classical machine-learning algorithm: done ET, LGBM, LR, RF, XGB
* (c) Optuna-tuned classical algorithm: done LGBM_tuned, RF_tuned, XGB_tuned
* (d) deep-learning model: done CNN_BiGRU_Att, CNN_BiLSTM_Att, Fusion, Transformer
* (d) deep-learning model + XGBoost head: done CNN_BiGRU_Att_XGBhead, CNN_BiLSTM_Att_XGBhead, Fusion_XGBhead, Transformer_XGBhead
* (e) soft voting: done SoftVote
* (e) stacking: done Stacking
* Best on LO2 (binary): BERT_GNN 1.000 (1.000-1.000); many models reach the same high score. The log-volume-only baseline is in `hybrid_lo2_logvolume_baseline.csv`.

## Proof totals

* {'PASS': 82, 'WARN': 4, 'REPORTED ACCURATELY': 13} (generated 2026-10-24 06:04:35, partial: False); earlier baseline: 65 PASS / 4 WARN / 0 FAIL / 9 REPORTED ACCURATELY before the new sections.
* P1 PASS, P2 PASS, P3 PASS, P4 PASS, P5 WARN, P6 PASS, P7 WARN, P8 PASS, P9 REPORTED ACCURATELY, P10 PASS, P11 REPORTED ACCURATELY, P12 REPORTED ACCURATELY, P13 PASS, P14 WARN, P15 REPORTED ACCURATELY, P16 REPORTED ACCURATELY, P17 PASS, P18 REPORTED ACCURATELY, P19 REPORTED ACCURATELY, P20 REPORTED ACCURATELY, P21 REPORTED ACCURATELY, P22 PASS, P23 PASS, P24 PASS

## Tests

* `python -m pytest tests app/backend/tests` -> 146 passed
* `npm run build (app/frontend)` -> built, tsc clean
* `npx playwright test --project=chromium hybrids.spec.ts pages.spec.ts (all pages at 375/768/1280/1920 incl. Results, Proof, Hybrid models, Demo, jargon check)` -> 101 passed
* `python scripts/proofs.py --only P20,P21,P22,P23 (P1-P19 untouched)` -> new sections PASS / REPORTED ACCURATELY; totals 78 PASS / 4 WARN / 0 FAIL / 13 REPORTED ACCURATELY
* `python scripts/run_notebook_check.py` -> 0 errors, 0 soft failures, 8 numbers equal
* `cold start (testbed launcher) and break slow pages` -> storefront 200, monitor 200, alert after 21 s, ports freed

## Colab notebook

* Executed in FAST mode in a fresh virtual environment (clean clone, bundle unzipped, no AnoMod raw folders, no models folder): executed_ok = True, cells 55, errors 0, soft failures 0, run time 5.1 min.
* Numbers compared with the proofs / result files: 8; different: 0.
* Files: `notebooks/FailSight_Workflow.ipynb`, `notebooks/FailSight_Workflow_executed.ipynb`, `notebooks/FailSight_Workflow_executed.html`, `docs/COLAB.md`, `failsight_bundle.zip` (git-ignored).

## Website: which models are shown by default

* AnoMod binary: top 9 shown by default: XGB, Stacking, LGBM_tuned, XGB_tuned, RF_tuned, LR, LGBM, CNN_BiLSTM_Att, GNN_Transformer (hybrid). Under *Show all models* (23 in total): Transformer_BiLSTM (hybrid), VAE_Transformer (hybrid), AE_LSTM (hybrid), Fusion_XGBhead, RF, ET, Fusion, SoftVote, CNN_BiLSTM_Att_XGBhead, Transformer_XGBhead, CNN_LSTM (hybrid), CNN_BiGRU_Att, CNN_BiGRU_Att_XGBhead, Transformer.
* AnoMod level: top 9 shown by default: Stacking, LGBM_tuned, XGB, LGBM, XGB_tuned, RF, RF_tuned, CNN_LSTM (hybrid), AE_LSTM (hybrid). Under *Show all models* (23 in total): Fusion_XGBhead, Fusion, GNN_Transformer (hybrid), ET, LR, CNN_BiLSTM_Att, SoftVote, CNN_BiLSTM_Att_XGBhead, Transformer_XGBhead, CNN_BiGRU_Att, CNN_BiGRU_Att_XGBhead, Transformer, Transformer_BiLSTM (hybrid), VAE_Transformer (hybrid).
* LO2 tab: top 9 by overall score shown by default, the rest under *Show all models*; every row marked unreliable (very small sample); unfinished families listed as not completed.

## Limitations (all stated on the Limits page, the Hybrid models page and in the docs)

* The earlier AnoMod models were checked with 10-fold cross-validation and the hybrid models with 5-fold. All AnoMod models, hybrids included, are scored on the same 647 test windows.
* AnoMod test part: only 36 healthy windows, wide ranges; the best model on every list was picked on the test set (lead is optimistic).
* Run identification can reproduce the AnoMod scores (P14, P20): they are not proof of fault detection.
* LO2: fewer than 6 whole runs under the 150 MB cap, one test run (172 windows, 13 healthy); trained on LO2's own data only; no transfer claim; binary LO2 is almost trivially separable (log volume alone scores high; see the baseline).
* Families not finished on LO2 are listed above as not completed.
* No transfer to other systems, no claim that FailSight works on any app, no forecasting of failures, no time-until-failure (not built; future work).
* The template encoder on LO2 is TF-IDF + SVD, not a real BERT. The AnoMod graph is built from the component names in the feature names, not from real service calls.
* Hybrid selection used a single screening fold (validation scores only) to pick hidden size and imbalance method; the CV then used 5 folds; fold models were not averaged.

## How to start the app

```
cd "C:\Users\Lenovo\Documents\AI PROJECT\code"
.\start_testbed.bat
```

| address | what |
|---|---|
| http://127.0.0.1:5000 | the shop |
| http://127.0.0.1:5000/ops | presenter controls (this computer only) |
| http://127.0.0.1:8000 | the live monitor; the Hybrid models page is at http://127.0.0.1:8000/examiners/hybrids |

Results website without the shop: `python -m app.backend.serve` then http://127.0.0.1:8000 . Fallback commit for the storefront demo: see `docs/PRESENT_NOW.md`; the tag `pre-overnight` marks the state before this run.

## git log --oneline -20

```
b9e7519 both-data-sets results, notebook fixed and executed
7bf6484 notebook executed
bff7ee4 combined both-data-sets training, percent display, glossary, notebook 7b
cb91e93 fix unused import
4727bdf hybrid pages: use the models' proper names
317f6c9 plain hybrid comparison page, screenshots
730da75 wording
b313433 plainer hybrid page with side-by-side comparison; wording
070d15d final: status, index, tests record
5e4f2ef front end polish, docs, index, steps
9cbab9d notebook executed copy, proofs P23
06a8b2b notebook: replay with exact fault start, quiet warnings
51517af notebook: fixes
992a582 LO2 families (b)-(e) results, explanations, shortcut diagnostic, proofs P20-P22, docs
2137055 hybrid: LO2 slice results (5 hybrids x 2 label versions)
298f759 hybrid: AnoMod results (5 hybrids x 2 label versions), engine timing fix
ca6468a frontend: hybrid models page, results top-9 + LO2 tab, limits; docs and notebook generator; explain, bundle and notebook-check scripts
ad3d3f4 chore: keep processed LO2 data out of git
2c07e70 hybrid: LO2 slice parser, LO2 data layer, LO2 hybrids, deep/tabular families, proofs P20-P23, API endpoints
1feca8e hybrid: five AnoMod hybrid models, generic engine, AnoMod runner, unit tests
```
