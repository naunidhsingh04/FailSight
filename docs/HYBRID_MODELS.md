# Hybrid deep-learning models

Written for: examiners and students who want to know exactly what the ten hybrid models are, how they are wired, how they were trained and what they scored.

> The earlier AnoMod models were checked with 10-fold cross-validation and the hybrid models with 5-fold. All AnoMod models, hybrids included, are scored on the same 647 test windows.

## What a hybrid is here

Each hybrid is **one model class** (one `keras.Model`). Its code builds a first component and a second component as sub-models; its forward pass calls the first, feeds its output into the second, and a small classification head gives the answer. Both parts are trained together end to end with one loss and one optimiser. It is **not** an ensemble and **not** an average of two separately trained models. Exception: BERT + GNN, where the text encoder is frozen by design (it turns each log template into a fixed vector once, before training) and the graph part consumes those vectors. Each class has a docstring with the first part, the second part and what is passed between them (`models_src/hybrid/anomod_hybrids.py`, `models_src/hybrid/lo2_hybrids.py`; the shared layers are in `models_src/hybrid/base.py`). Graph layers are written by hand with an adjacency-matrix multiplication (`GraphLayer`); no extra libraries. Unit tests (`tests/test_hybrid_models.py`) check output shapes, that probabilities sum to one, that the auxiliary reconstruction losses exist and that gradients reach both parts.

| # | data set | hybrid | first part -> passed on -> second part |
|---|---|---|---|
| 1 | AnoMod | Transformer + BiLSTM | self-attention over the 6 recent windows -> a (6 x hidden) sequence -> bidirectional LSTM |
| 2 | AnoMod | Autoencoder + LSTM | window autoencoder (reconstruction error is an auxiliary loss, weight 0.1) -> (6 x hidden) codes -> LSTM |
| 3 | AnoMod | GNN + Transformer | graph over the system's components (see below) -> one vector per window -> Transformer encoder |
| 4 | AnoMod | CNN + LSTM | two 1-D convolutions with batch normalisation along the windows -> pattern sequence -> LSTM |
| 5 | AnoMod | VAE + Transformer | variational autoencoder per window (rebuild + 0.01 x KL as auxiliary loss) -> codes (means at prediction time) -> Transformer encoder |
| 6 | LO2 | BERT-style encoder (frozen) + GNN | frozen template vectors scaled by the window's template counts -> graph over template co-occurrence -> graph convolutions with attention pooling, joined with the window's other numbers |
| 7 | LO2 | Transformer + LSTM | self-attention over the last 3 windows -> sequence -> LSTM |
| 8 | LO2 | Autoencoder + LSTM | same class as #2, trained separately on LO2 |
| 9 | LO2 | GNN + LSTM | graph over the log sources (7 services, database, load generator, metrics, flags) -> one vector per window -> LSTM |
| 10 | LO2 | Multimodal Transformer + MLP | a logs branch and a metrics branch (two Transformers) -> joined vector -> MLP fusion |

## The AnoMod graph (hybrid 3), exactly how it is built

The metric feature names do not carry per-service names (they are resource readings such as `m_system_cpu_usage__mean` or `m_socialnet_container_memory__max`), so the graph is built over the *components* the names reveal. Each model input is assigned to a node by its name: `api_*` -> api, `log_*` -> logs, `m_socialnet_container_*` -> socialnet_containers, `m_system_*` -> host_system, `m_container_*` -> containers, `m_node_*` and `m_rate(node...)` -> node_exporter, `m_process_*` -> process, `m_http_*` -> http_server, everything else -> other_services, and the was-missing and system flags -> one flags node. Two nodes are joined by an edge when they report on the same resource (cpu, memory, network, disk or load, detected from the feature names), plus api - logs, api - host_system and logs - host_system (the API / log view of the system), and the flags node is joined to every node. The node and edge lists with the reason for every edge are saved in `report_outputs/hybrid_anomod_graph_nodes.csv` and `hybrid_anomod_graph_edges.csv`. For LO2 the graph is built from the training windows only (see below).

## Same protocol on both data sets

* **Split first.** 80:20, written to a split file before anything else (AnoMod: the existing `split_ids.csv`, 2,549 train / 647 test; LO2: `lo2_split_ids.csv`, whole runs). The test part keeps its natural class imbalance, is never resampled and is scored once per model.
* **Preprocessing fitted on the training part only** (AnoMod: `scripts/preprocess.py`; LO2: `scripts/lo2_data.py`, LO2's own features): counters to rates, `log1p` on heavy-tailed features, was-missing flags, median imputation, constant dropping, clipping at the 0.5 and 99.5 percentiles, standardisation (LO2 additionally compresses its 478 metric families to 32 principal components, fitted on the training rows).
* **SMOTE only on training parts:** inside each training fold and once on the full 80% for the final model; `k_neighbors` is the largest valid value (minimum 1); a class with fewer than 2 training windows would be skipped and said so. Class weights versus SMOTE is decided on validation. The per-fold counts are in `hybrid_*_smote_counts.csv` (`b:` before / `a:` after / `hold:` held out). ADASYN is computed for a size comparison only (`hybrid_*_imbalance.csv`) and never used to train.
* **5-fold cross-validation, block-aware** (AnoMod: the project's 10-window blocks; LO2: run + scenario + 3-window blocks), no group shared across folds, every class in every validation fold (proof P22).
* **Screening before the full run:** each hybrid is first tried on ONE training fold for at most 10 epochs with 2 hidden sizes (32, 64) x 2 imbalance methods (SMOTE, class weights), using validation scores only; a clearly broken model gets one retry with half the learning rate and is then reported as `did not learn`, never removed (`hybrid_*_screening.csv`).
* **Training:** AdamW (weight decay 1e-4), gradient clipping, label smoothing 0.05, batch size 32, ReduceLROnPlateau, early stopping on the validation fold (patience 6, at most 40 epochs), seed 42, float32, CPU only.
* **Overfitting diagnosis and correction:** a model is flagged when the train-minus-validation macro-F1 gap exceeds 0.10; it is then re-run with stronger regularisation (half the hidden size, dropout + 0.2, weight decay x 20) and the corrected version is adopted only if validation balanced accuracy does not drop by more than 0.01 (`hybrid_*_overfit.csv`).
* **Thresholds on out-of-fold predictions only** (`hybrid_*_thresholds.json`); the official score uses argmax, the tuned-threshold score is reported next to it in `hybrid_*_test_metrics.csv`.
* **Frozen before testing:** each model's configuration is written to `hybrid_*_frozen_<labels>__<model>.json` before its test windows are scored. Nothing was chosen, dropped or tuned by looking at test scores.
* **Controls:** shuffled-label control for every model (`hybrid_*_shuffled_control.csv`), run-identification check (`hybrid_*_run_identification.csv`), bootstrap 95% intervals on the test part (1000 block-bootstrap resamples, `hybrid_*_test_bootstrap_ci.csv`). LO2 also has a log-volume-only baseline (`hybrid_lo2_logvolume_baseline.csv`).
* **Memory and CPU:** one training process at a time, 2 threads, float32, memory checked before each model (limit 5.5 GB). Models are saved as weights in `models/hybrid/anomod/` and `models/hybrid/lo2/` (git-ignored) and rebuilt from the frozen configuration; proofs P20 and P21 reload them and recompute the scores.

## Results: AnoMod hybrids

**Binary (Normal / Anomaly) labels**

| model | status | test balanced accuracy (95% range) | healthy-window recall | ROC-AUC | CV validation BA | training time | folds |
|---|---|---|---|---|---|---|---|
| GNN_Transformer | done | 0.652 (0.467-0.968) | 0.361 | 0.797 | 0.779 | 6.4 min | 5 |
| Transformer_BiLSTM | done | 0.650 (0.475-0.938) | 0.333 | 0.801 | 0.842 | 4.8 min | 5 |
| VAE_Transformer | done | 0.649 (0.488-0.978) | 0.333 | 0.899 | 0.812 | 4.3 min | 5 |
| AE_LSTM | done | 0.631 (0.458-0.959) | 0.333 | 0.822 | 0.809 | 2.8 min | 5 |
| CNN_LSTM | done | 0.609 (0.452-0.963) | 0.278 | 0.867 | 0.815 | 2.6 min | 5 |

**Multi-class labels**

| model | status | test balanced accuracy (95% range) | healthy-window recall | ROC-AUC | CV validation BA | training time | folds |
|---|---|---|---|---|---|---|---|
| CNN_LSTM | done | 0.775 (0.659-0.886) | 0.500 | 0.958 | 0.795 | 3.3 min | 5 |
| AE_LSTM | done | 0.772 (0.648-0.877) | 0.583 | 0.957 | 0.784 | 3.7 min | 5 |
| GNN_Transformer | done | 0.767 (0.644-0.882) | 0.583 | 0.936 | 0.810 | 6.7 min | 5 |
| Transformer_BiLSTM | done | 0.717 (0.624-0.816) | 0.389 | 0.954 | 0.766 | 5.4 min | 5 |
| VAE_Transformer | done | 0.703 (0.598-0.806) | 0.361 | 0.939 | 0.773 | 5.2 min | 5 |

* **binary labels:** best hybrid GNN_Transformer 0.652 (95% range 0.467-0.968) versus the best of the 36 earlier models, XGB, 0.774 (95% range 0.496-0.996): **no hybrid beats the best earlier model**; the ranges overlap.
* **level labels:** best hybrid CNN_LSTM 0.775 (95% range 0.659-0.886) versus the best of the 36 earlier models, Stacking, 0.890 (95% range 0.838-0.941): **no hybrid beats the best earlier model**; the ranges overlap.

Why the hybrids are not expected to beat the tree models here: the data are small and table-like (2,549 training windows, about 128 numbers each), and the test part has only 36 healthy windows, so every range is wide. Tree models are strong on small tabular data; neural networks need more data. If the hybrids end up below the tree models, that is what is reported.

## Results: LO2 hybrids (all 33 runs, 7 untouched test runs)

**Binary (Normal / Anomaly) labels**

| model | status | test balanced accuracy (95% range) | healthy-window recall | ROC-AUC | CV validation BA | training time | folds |
|---|---|---|---|---|---|---|---|
| BERT_GNN | done | 1.000 (1.000-1.000) | 1.000 | 1.000 | 1.000 | 10.3 min | 5 |
| Transformer_LSTM | done | 1.000 (1.000-1.000) | 1.000 | 1.000 | 1.000 | 5.0 min | 5 |
| AE_LSTM | done | 1.000 (1.000-1.000) | 1.000 | 1.000 | 1.000 | 5.1 min | 5 |
| GNN_LSTM | done | 1.000 (1.000-1.000) | 1.000 | 1.000 | 1.000 | 6.9 min | 5 |
| Multimodal_Transformer_MLP | done | 1.000 (1.000-1.000) | 1.000 | 1.000 | 1.000 | 11.3 min | 5 |

**Multi-class labels**

| model | status | test balanced accuracy (95% range) | healthy-window recall | ROC-AUC | CV validation BA | training time | folds |
|---|---|---|---|---|---|---|---|
| Multimodal_Transformer_MLP | done | 0.720 (0.680-0.759) | 1.000 | 0.877 | 0.720 | 14.8 min | 5 |
| AE_LSTM | done | 0.709 (0.668-0.748) | 1.000 | 0.883 | 0.710 | 6.6 min | 5 |
| Transformer_LSTM | done | 0.706 (0.671-0.740) | 1.000 | 0.859 | 0.712 | 9.3 min | 5 |
| GNN_LSTM | done | 0.681 (0.643-0.718) | 1.000 | 0.860 | 0.700 | 5.8 min | 5 |
| BERT_GNN | done | 0.674 (0.632-0.717) | 1.000 | 0.850 | 0.683 | 17.2 min | 5 |

LO2 results are about training on LO2's own data (33 whole runs, 1204 test windows from 7 untouched runs, 91 healthy) and the 80% training part is balanced with SMOTE while the 20% test part is untouched. They say nothing about transfer from AnoMod. The binary labels score high because LO2's injected HTTP errors leave clear error lines and metric changes; the controls below show it is not a warm-up or run-length shortcut. Compare the log-volume-only baseline in `hybrid_lo2_logvolume_baseline.csv`.

The template encoder is TF-IDF + truncated SVD of the template text, because a downloadable BERT-style encoder (and the `transformers` library) was not available here; this is stated on every page that shows BERT + GNN.

## Explanations of the best hybrid on each data set

The best hybrid is chosen by its cross-validation validation score (never by test scores). Method: gradient x input attribution on real test windows (`scripts/hybrid_explain.py`; `hybrid_*_explain_*.csv`); BERT + GNN also reports its attention over log templates when it is the best. These explain the model, not the real cause of a fault.

* AnoMod, binary labels, best hybrid by cross-validation score = **Transformer_BiLSTM**: top inputs by gradient x input: `was_missing_api`, `m_system_disk_usage_percent__mean`, `api_latency_mean_ms`, `api_latency_p99_ms`, `api_latency_p95_ms`; share by group: api 21%, flag 16%, log 4%, metric 59%
* AnoMod, level labels, best hybrid by cross-validation score = **GNN_Transformer**: top inputs by gradient x input: `m_system_disk_usage_percent__max`, `m_system_disk_usage_percent__mean`, `m_system_cpu_usage__max`, `m_system_load1__max`, `m_system_disk_usage_percent__std`; share by group: api 4%, flag 2%, log 3%, metric 90%
* LO2, binary labels, best hybrid by cross-validation score = **BERT_GNN**: top inputs by gradient x input: `m_pc1`, `m_pc0`, `m_pc4`, `m_pc2`, `m_pc6`; share by group: flag 0%, log 41%, log template 5%, metric 55%
* LO2, level labels, best hybrid by cross-validation score = **BERT_GNN**: top inputs by gradient x input: `m_pc8`, `m_pc1`, `m_pc4`, `m_pc2`, `m_pc0`; share by group: flag 0%, log 40%, log template 2%, metric 58%

* LO2, best classical tree model by cross-validation (**RF**): SHAP (TreeExplainer) on the real test windows, top inputs `m_pc1`, `m_pc4`, `m_pc2`, `log_token_INFO`, `log_token_DEBUG`; LIME for one window in `lo2_full_lime_example.csv`.

## Limits

* The earlier AnoMod models were checked with 10-fold cross-validation and the hybrid models with 5-fold. All AnoMod models, hybrids included, are scored on the same 647 test windows.
* Small samples, wide ranges; the best model on any list was picked on the test set.
* No claim of transfer to other systems, no forecasting, no time-until-failure (not built; future work).
