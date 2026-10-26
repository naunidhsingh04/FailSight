# Hybrid deep-learning models

Written for: examiners and students who want to know exactly what the ten hybrid models are, how they are wired, how they were trained and what they scored.

> The earlier AnoMod models were checked with 10-fold cross-validation and the hybrid models with 5-fold. All AnoMod models, hybrids included, are scored on the same 647 test windows.

## What a hybrid is here

Each hybrid is built in **two stages from two separate algorithm functions** (`models_src/hybrid/stages.py`); the layers themselves are separate functions in `models_src/hybrid/components.py` (`cnn`, `lstm`, `bilstm`, `transformer`, `autoencoder`, `vae`, `gnn`, `bert`, `template_gnn`, `multimodal_transformer`, `mlp`). Nothing combines two algorithms into one function. The **first algorithm** (for example `cnn_stage`) turns the raw windows into a representation; it is trained on its own with a small classification head of its own, then frozen, and it returns a set of variables (its input, its trained layer, an `encode` function that gives its values, its training history). The **second algorithm** (for example `lstm_stage`) is called with those variables, uses the values the first one produced as its input, adds its own layers and does the final classification and the validation (early stopping on the validation fold). A hybrid is only that hand-over, for example `cnn_vars = cnn_stage(spec, cfg, data)` then `lstm_vars = lstm_stage(cnn_vars, spec, cfg, data)` (`models_src/hybrid/anomod_hybrids.py`, `models_src/hybrid/lo2_hybrids.py`). It is **not** an ensemble and **not** an average of two models; it is **not** trained end to end either (the second stage does not change the first). BERT + GNN: the text encoder is frozen by design, so its stage trains nothing and the graph stage consumes its vectors. Graph layers are written by hand with an adjacency-matrix multiplication (`GraphLayer`); no extra libraries. Unit tests (`tests/test_hybrid_models.py`) check output shapes, that probabilities sum to one, that the auxiliary reconstruction losses exist, that the first stage is trained and then left unchanged, and that gradients reach the second part.

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
| Transformer_BiLSTM | done | 0.691 (0.479-0.937) | 0.417 | 0.685 | 0.817 | 11.5 min | 5 |
| CNN_LSTM | done | 0.661 (0.492-0.976) | 0.361 | 0.913 | 0.806 | 7.3 min | 5 |
| GNN_Transformer | done | 0.627 (0.482-0.941) | 0.278 | 0.763 | 0.798 | 13.0 min | 5 |
| VAE_Transformer | done | 0.618 (0.451-0.963) | 0.306 | 0.846 | 0.787 | 21.9 min | 5 |
| AE_LSTM | done | 0.605 (0.462-0.935) | 0.250 | 0.792 | 0.788 | 6.3 min | 5 |

**Multi-class labels**

| model | status | test balanced accuracy (95% range) | healthy-window recall | ROC-AUC | CV validation BA | training time | folds |
|---|---|---|---|---|---|---|---|
| GNN_Transformer | done | 0.779 (0.649-0.879) | 0.694 | 0.926 | 0.837 | 12.6 min | 5 |
| Transformer_BiLSTM | done | 0.760 (0.658-0.857) | 0.500 | 0.936 | 0.800 | 6.3 min | 5 |
| VAE_Transformer | done | 0.758 (0.637-0.870) | 0.500 | 0.913 | 0.812 | 24.0 min | 5 |
| CNN_LSTM | done | 0.757 (0.641-0.864) | 0.528 | 0.934 | 0.791 | 6.1 min | 5 |
| AE_LSTM | done | 0.742 (0.633-0.854) | 0.417 | 0.915 | 0.811 | 6.2 min | 5 |

* **binary labels:** best hybrid Transformer_BiLSTM 0.691 (95% range 0.479-0.937) versus the best of the 36 earlier models, XGB, 0.774 (95% range 0.496-0.996): **no hybrid beats the best earlier model**; the ranges overlap.
* **level labels:** best hybrid GNN_Transformer 0.779 (95% range 0.649-0.879) versus the best of the 36 earlier models, Stacking, 0.890 (95% range 0.838-0.941): **no hybrid beats the best earlier model**; the ranges overlap.

Why the hybrids are not expected to beat the tree models here: the data are small and table-like (2,549 training windows, about 128 numbers each), and the test part has only 36 healthy windows, so every range is wide. Tree models are strong on small tabular data; neural networks need more data. If the hybrids end up below the tree models, that is what is reported.

## Results: LO2 hybrids (all 33 runs, 7 untouched test runs)

**Binary (Normal / Anomaly) labels**

| model | status | test balanced accuracy (95% range) | healthy-window recall | ROC-AUC | CV validation BA | training time | folds |
|---|---|---|---|---|---|---|---|
| BERT_GNN | done | 1.000 (1.000-1.000) | 1.000 | 1.000 | 1.000 | 7.5 min | 5 |
| Transformer_LSTM | done | 1.000 (1.000-1.000) | 1.000 | 1.000 | 1.000 | 6.1 min | 5 |
| AE_LSTM | done | 1.000 (1.000-1.000) | 1.000 | 1.000 | 1.000 | 6.1 min | 5 |
| GNN_LSTM | done | 1.000 (1.000-1.000) | 1.000 | 1.000 | 1.000 | 12.7 min | 5 |
| Multimodal_Transformer_MLP | done | 1.000 (1.000-1.000) | 1.000 | 1.000 | 1.000 | 34.3 min | 5 |

**Multi-class labels**

| model | status | test balanced accuracy (95% range) | healthy-window recall | ROC-AUC | CV validation BA | training time | folds |
|---|---|---|---|---|---|---|---|
| Transformer_LSTM | done | 0.714 (0.673-0.755) | 1.000 | 0.857 | 0.711 | 11.7 min | 5 |
| Multimodal_Transformer_MLP | done | 0.710 (0.669-0.751) | 1.000 | 0.878 | 0.718 | 36.3 min | 5 |
| AE_LSTM | done | 0.684 (0.637-0.728) | 1.000 | 0.837 | 0.685 | 9.5 min | 5 |
| BERT_GNN | done | 0.674 (0.632-0.717) | 1.000 | 0.850 | 0.683 | 9.6 min | 5 |
| GNN_LSTM | done | 0.648 (0.612-0.687) | 1.000 | 0.820 | 0.676 | 16.5 min | 5 |

LO2 results are about training on LO2's own data (33 whole runs, 1204 test windows from 7 untouched runs, 91 healthy) and the 80% training part is balanced with SMOTE while the 20% test part is untouched. They say nothing about transfer from AnoMod. The binary labels score high because LO2's injected HTTP errors leave clear error lines and metric changes; the controls below show it is not a warm-up or run-length shortcut. Compare the log-volume-only baseline in `hybrid_lo2_logvolume_baseline.csv`.

The template encoder is TF-IDF + truncated SVD of the template text, because a downloadable BERT-style encoder (and the `transformers` library) was not available here; this is stated on every page that shows BERT + GNN.

## Explanations of the best hybrid on each data set

The best hybrid is chosen by its cross-validation validation score (never by test scores). Method: gradient x input attribution on real test windows (`scripts/hybrid_explain.py`; `hybrid_*_explain_*.csv`); BERT + GNN also reports its attention over log templates when it is the best. These explain the model, not the real cause of a fault.

* AnoMod, binary labels, best hybrid by cross-validation score = **Transformer_BiLSTM**: top inputs by gradient x input: `m_system_disk_usage_percent__mean`, `m_system_disk_usage_percent__max`, `m_system_cpu_usage__std`, `m_system_cpu_usage__mean`, `m_system_load1__mean`; share by group: api 9%, flag 4%, log 2%, metric 86%
* AnoMod, level labels, best hybrid by cross-validation score = **GNN_Transformer**: top inputs by gradient x input: `m_system_disk_usage_percent__mean`, `m_system_disk_usage_percent__max`, `m_system_load1__max`, `m_system_load1__mean`, `m_system_network_transmit_bytes__mean`; share by group: api 7%, flag 1%, log 2%, metric 90%
* LO2, binary labels, best hybrid by cross-validation score = **BERT_GNN**: top inputs by gradient x input: `m_pc0`, `m_pc2`, `m_pc1`, `m_pc3`, `m_pc4`; share by group: flag 0%, log 42%, log template 6%, metric 52%
* LO2, level labels, best hybrid by cross-validation score = **Multimodal_Transformer_MLP**: top inputs by gradient x input: `m_pc2`, `m_pc3`, `m_pc1`, `m_pc0`, `tmpl_locust_0`; share by group: flag 0%, log 25%, log template 38%, metric 38%

* LO2, best classical tree model by cross-validation (**RF**): SHAP (TreeExplainer) on the real test windows, top inputs `m_pc3`, `m_pc1`, `m_pc2`, `log_locust_ERROR`, `log_locust_error_rate`; LIME for one window in `lo2_full_lime_example.csv`.

## Limits

* The earlier AnoMod models were checked with 10-fold cross-validation and the hybrid models with 5-fold. All AnoMod models, hybrids included, are scored on the same 647 test windows.
* Small samples, wide ranges; the best model on any list was picked on the test set.
* No claim of transfer to other systems, no forecasting, no time-until-failure (not built; future work).
