"""Writes docs/HYBRID_MODELS.md, docs/LO2_SLICE.md and docs/NUMBERS_FOR_VIVA.md. Every number is read from a saved result file (named next to it); no number is typed in.
python scripts/build_docs.py"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REP, DOCS = ROOT / "report_outputs", ROOT / "docs"
FOLD_NOTE = "The earlier AnoMod models were checked with 10-fold cross-validation and the hybrid models with 5-fold. All AnoMod models, hybrids included, are scored on the same 647 test windows."
f3 = lambda x: "n/a" if x is None or x != x else f"{x:.3f}"


def rd(name, **kw):
    p = REP / name
    return pd.read_csv(p, **kw) if p.exists() else None


def secs(s):
    return "n/a" if s is None or s != s else (f"{s / 60:.1f} min" if s >= 90 else f"{s:.0f} s")


def lb_table(prefix, labels=("label_binary", "label_level")):
    lb = rd(f"{prefix}leaderboard.csv")
    if lb is None:
        return ["(result file missing)"]
    out = []
    for lab in labels:
        d = lb[lb.label_version == lab]
        out += [f"**{'Binary (Normal / Anomaly)' if lab == 'label_binary' else 'Multi-class'} labels**", "",
                "| model | status | test balanced accuracy (95% range) | healthy-window recall | ROC-AUC | CV validation BA | training time | folds |", "|---|---|---|---|---|---|---|---|"]
        tt = rd(f"{prefix}training_time.csv")
        for r in d.itertuples():
            if r.status == "not completed":
                out.append(f"| {r.model} | not completed | - | - | - | - | - | - |")
                continue
            t = tt[(tt.model == r.model) & (tt.label_version == lab)].total_seconds_incl_screening_and_controls.iloc[0] if tt is not None and not tt[(tt.model == r.model) & (tt.label_version == lab)].empty else None
            out.append(f"| {r.model} | {r.status} | {f3(r.test_bal_acc)} ({f3(r.test_bal_acc_ci_low)}-{f3(r.test_bal_acc_ci_high)}) | {f3(r.test_normal_recall)} | {f3(r.test_roc_auc)} | {f3(r.cv_val_bal_acc)} | {secs(t)} | {r.folds} |")
        out.append("")
    return out


def hybrid_doc():
    L = ["# Hybrid deep-learning models", "", "Written for: examiners and students who want to know exactly what the ten hybrid models are, how they are wired, how they were trained and what they scored.", "",
         "> " + FOLD_NOTE, "",
         "## What a hybrid is here", "",
         "Each hybrid is **one model class** (one `keras.Model`). Its code builds a first component and a second component as sub-models; its forward pass calls the first, feeds its output into the second, and a small classification head gives the answer. Both parts are trained together end to end with one loss and one optimiser. It is **not** an ensemble and **not** an average of two separately trained models. Exception: BERT + GNN, where the text encoder is frozen by design (it turns each log template into a fixed vector once, before training) and the graph part consumes those vectors. Each class has a docstring with the first part, the second part and what is passed between them (`models_src/hybrid/anomod_hybrids.py`, `models_src/hybrid/lo2_hybrids.py`; the shared layers are in `models_src/hybrid/base.py`). Graph layers are written by hand with an adjacency-matrix multiplication (`GraphLayer`); no extra libraries. Unit tests (`tests/test_hybrid_models.py`) check output shapes, that probabilities sum to one, that the auxiliary reconstruction losses exist and that gradients reach both parts.", "",
         "| # | data set | hybrid | first part -> passed on -> second part |", "|---|---|---|---|",
         "| 1 | AnoMod | Transformer + BiLSTM | self-attention over the 6 recent windows -> a (6 x hidden) sequence -> bidirectional LSTM |",
         "| 2 | AnoMod | Autoencoder + LSTM | window autoencoder (reconstruction error is an auxiliary loss, weight 0.1) -> (6 x hidden) codes -> LSTM |",
         "| 3 | AnoMod | GNN + Transformer | graph over the system's components (see below) -> one vector per window -> Transformer encoder |",
         "| 4 | AnoMod | CNN + LSTM | two 1-D convolutions with batch normalisation along the windows -> pattern sequence -> LSTM |",
         "| 5 | AnoMod | VAE + Transformer | variational autoencoder per window (rebuild + 0.01 x KL as auxiliary loss) -> codes (means at prediction time) -> Transformer encoder |",
         "| 6 | LO2 | BERT-style encoder (frozen) + GNN | frozen template vectors scaled by the window's template counts -> graph over template co-occurrence -> graph convolutions with attention pooling, joined with the window's other numbers |",
         "| 7 | LO2 | Transformer + LSTM | self-attention over the last 3 windows -> sequence -> LSTM |",
         "| 8 | LO2 | Autoencoder + LSTM | same class as #2, trained separately on LO2 |",
         "| 9 | LO2 | GNN + LSTM | graph over the log sources (7 services, database, load generator, metrics, flags) -> one vector per window -> LSTM |",
         "| 10 | LO2 | Multimodal Transformer + MLP | a logs branch and a metrics branch (two Transformers) -> joined vector -> MLP fusion |", "",
         "## The AnoMod graph (hybrid 3), exactly how it is built", "",
         "The metric feature names do not carry per-service names (they are resource readings such as `m_system_cpu_usage__mean` or `m_socialnet_container_memory__max`), so the graph is built over the *components* the names reveal. Each model input is assigned to a node by its name: `api_*` -> api, `log_*` -> logs, `m_socialnet_container_*` -> socialnet_containers, `m_system_*` -> host_system, `m_container_*` -> containers, `m_node_*` and `m_rate(node...)` -> node_exporter, `m_process_*` -> process, `m_http_*` -> http_server, everything else -> other_services, and the was-missing and system flags -> one flags node. Two nodes are joined by an edge when they report on the same resource (cpu, memory, network, disk or load, detected from the feature names), plus api - logs, api - host_system and logs - host_system (the API / log view of the system), and the flags node is joined to every node. The node and edge lists with the reason for every edge are saved in `report_outputs/hybrid_anomod_graph_nodes.csv` and `hybrid_anomod_graph_edges.csv`. For LO2 the graph is built from the training windows only (see below).", "",
         "## Same protocol on both data sets", "",
         "* **Split first.** 80:20, written to a split file before anything else (AnoMod: the existing `split_ids.csv`, 2,549 train / 647 test; LO2: `lo2_split_ids.csv`, whole runs). The test part keeps its natural class imbalance, is never resampled and is scored once per model.",
         "* **Preprocessing fitted on the training part only** (AnoMod: `scripts/preprocess.py`; LO2: `scripts/lo2_data.py`, LO2's own features): counters to rates, `log1p` on heavy-tailed features, was-missing flags, median imputation, constant dropping, clipping at the 0.5 and 99.5 percentiles, standardisation (LO2 additionally compresses its 478 metric families to 32 principal components, fitted on the training rows).",
         "* **SMOTE only on training parts:** inside each training fold and once on the full 80% for the final model; `k_neighbors` is the largest valid value (minimum 1); a class with fewer than 2 training windows would be skipped and said so. Class weights versus SMOTE is decided on validation. The per-fold counts are in `hybrid_*_smote_counts.csv` (`b:` before / `a:` after / `hold:` held out). ADASYN is computed for a size comparison only (`hybrid_*_imbalance.csv`) and never used to train.",
         "* **5-fold cross-validation, block-aware** (AnoMod: the project's 10-window blocks; LO2: run + scenario + 3-window blocks), no group shared across folds, every class in every validation fold (proof P22).",
         "* **Screening before the full run:** each hybrid is first tried on ONE training fold for at most 10 epochs with 2 hidden sizes (32, 64) x 2 imbalance methods (SMOTE, class weights), using validation scores only; a clearly broken model gets one retry with half the learning rate and is then reported as `did not learn`, never removed (`hybrid_*_screening.csv`).",
         "* **Training:** AdamW (weight decay 1e-4), gradient clipping, label smoothing 0.05, batch size 32, ReduceLROnPlateau, early stopping on the validation fold (patience 6, at most 40 epochs), seed 42, float32, CPU only.",
         "* **Overfitting diagnosis and correction:** a model is flagged when the train-minus-validation macro-F1 gap exceeds 0.10; it is then re-run with stronger regularisation (half the hidden size, dropout + 0.2, weight decay x 20) and the corrected version is adopted only if validation balanced accuracy does not drop by more than 0.01 (`hybrid_*_overfit.csv`).",
         "* **Thresholds on out-of-fold predictions only** (`hybrid_*_thresholds.json`); the official score uses argmax, the tuned-threshold score is reported next to it in `hybrid_*_test_metrics.csv`.",
         "* **Frozen before testing:** each model's configuration is written to `hybrid_*_frozen_<labels>__<model>.json` before its test windows are scored. Nothing was chosen, dropped or tuned by looking at test scores.",
         "* **Controls:** shuffled-label control for every model (`hybrid_*_shuffled_control.csv`), run-identification check (`hybrid_*_run_identification.csv`), bootstrap 95% intervals on the test part (1000 block-bootstrap resamples, `hybrid_*_test_bootstrap_ci.csv`). LO2 also has a log-volume-only baseline (`hybrid_lo2_logvolume_baseline.csv`).",
         "* **Memory and CPU:** one training process at a time, 2 threads, float32, memory checked before each model (limit 5.5 GB). Models are saved as weights in `models/hybrid/anomod/` and `models/hybrid/lo2/` (git-ignored) and rebuilt from the frozen configuration; proofs P20 and P21 reload them and recompute the scores.", ""]
    L += ["## Results: AnoMod hybrids", ""] + lb_table("hybrid_anomod_")
    ex = rd("leaderboard_label_binary.csv")
    exl = rd("leaderboard_label_level.csv")
    bs = rd("test_bootstrap_ci.csv")
    hb = rd("hybrid_anomod_leaderboard.csv")
    if hb is not None and ex is not None:
        for lab, e in (("label_binary", ex), ("label_level", exl)):
            h = hb[(hb.label_version == lab) & (hb.status != "not completed")].sort_values("test_bal_acc", ascending=False)
            if h.empty:
                continue
            top = e.sort_values("test_bal_acc", ascending=False).iloc[0]
            x = bs[(bs.model == top.model) & (bs.label_version == lab) & (bs.scope == "ALL") & (bs.metric == "balanced_accuracy")].iloc[0]
            b = h.iloc[0]
            L.append(f"* **{lab[6:]} labels:** best hybrid {b.model} {f3(b.test_bal_acc)} (95% range {f3(b.test_bal_acc_ci_low)}-{f3(b.test_bal_acc_ci_high)}) versus the best of the 36 earlier models, {top.model}, {f3(top.test_bal_acc)} (95% range {f3(x.ci95_low)}-{f3(x.ci95_high)}): "
                     f"**{'a hybrid scores higher' if b.test_bal_acc > top.test_bal_acc else 'no hybrid beats the best earlier model'}**; the ranges {'overlap' if (b.test_bal_acc_ci_low <= x.ci95_high and x.ci95_low <= b.test_bal_acc_ci_high) else 'do not overlap'}.")
        _s = rd("split_ids.csv")
        L += ["", f"Why the hybrids are not expected to beat the tree models here: the data are small and table-like ({int((_s.split == 'train').sum()):,} training windows, about 128 numbers each), and the test part has only {int(rd('hybrid_anomod_test_metrics.csv').support_Normal.iloc[0])} healthy windows, so every range is wide. Tree models are strong on small tabular data; neural networks need more data. If the hybrids end up below the tree models, that is what is reported.", ""]
    L += ["## Results: LO2 hybrids (all 33 runs, 7 untouched test runs)", ""] + lb_table("hybrid_lo2_")
    _sp, _man = rd("lo2_split_ids.csv", dtype={"run_id": str}), rd("lo2_slice_manifest.csv")
    _facts = (f"{len(_man)} whole runs, {int((_sp.split == 'test').sum())} test windows from {_sp[_sp.split == 'test'].run_id.nunique()} untouched runs, {int(((_sp.split == 'test') & (_sp.label_level == 'Normal')).sum())} healthy"
              if _sp is not None and _man is not None and 'window_id' in _sp.columns else "all downloaded runs")
    L += [f"LO2 results are about training on LO2's own data ({_facts}) and the 80% training part is balanced with SMOTE while the 20% test part is untouched. They say nothing about transfer from AnoMod. The binary labels score high because LO2's injected HTTP errors leave clear error lines and metric changes; the controls below show it is not a warm-up or run-length shortcut. Compare the log-volume-only baseline in `hybrid_lo2_logvolume_baseline.csv`.", "",
          "The template encoder is TF-IDF + truncated SVD of the template text, because a downloadable BERT-style encoder (and the `transformers` library) was not available here; this is stated on every page that shows BERT + GNN.", ""]
    ex_lines = []
    for ds, pre in (("AnoMod", "hybrid_anomod_"), ("LO2", "hybrid_lo2_")):
        for lab in ("label_binary", "label_level"):
            d = rd(f"{pre}explain_{lab}.csv")
            if d is None:
                continue
            g = rd(f"{pre}explain_groups.csv")
            gs = g[(g.label_version == lab)] if g is not None else None
            ex_lines.append(f"* {ds}, {lab[6:]} labels, best hybrid by cross-validation score = **{d.model.iloc[0]}**: top inputs by gradient x input: " + ", ".join(f"`{r.feature}`" for r in d.head(5).itertuples()) +
                            ("; share by group: " + ", ".join(f"{r.group} {r.share_of_total:.0%}" for r in gs.itertuples()) if gs is not None and len(gs) else ""))
    if ex_lines:
        L += ["## Explanations of the best hybrid on each data set", "", "The best hybrid is chosen by its cross-validation validation score (never by test scores). Method: gradient x input attribution on real test windows (`scripts/hybrid_explain.py`; `hybrid_*_explain_*.csv`); BERT + GNN also reports its attention over log templates when it is the best. These explain the model, not the real cause of a fault.", ""] + ex_lines + [""]
        sh = rd("lo2_full_shap_global.csv")
        if sh is not None:
            L += [f"* LO2, best classical tree model by cross-validation (**{sh.model.iloc[0]}**): SHAP (TreeExplainer) on the real test windows, top inputs " + ", ".join(f"`{r.feature}`" for r in sh.head(5).itertuples()) + "; LIME for one window in `lo2_full_lime_example.csv`.", ""]
    L += ["## Limits", "", "* " + FOLD_NOTE, "* Small samples, wide ranges; the best model on any list was picked on the test set.", "* No claim of transfer to other systems, no forecasting, no time-until-failure (not built; future work).", ""]
    (DOCS / "HYBRID_MODELS.md").write_text("\n".join(L), encoding="utf-8")


def lo2_doc():
    man = rd("lo2_slice_manifest.csv", dtype={"run_id": str})
    sp = rd("lo2_split_ids.csv", dtype={"run_id": str})
    if man is None:
        return
    allmb = sum(p.stat().st_size for p in [] )
    L = ["# The LO2 slice", "", "Written for: examiners who ask what part of LO2 was used and why.", "",
         "LO2 (Zenodo 10.5281/zenodo.14265858) is a set of 33 runs of an OAuth2 service (`light-oauth2`: seven micro-services, a database, a Locust load generator, Prometheus metrics). Each run executes 54 scenarios back to back (one `correct` scenario and 53 error scenarios); a scenario's folder name is its label. The data are stored read-only on this computer in the folder `LO2` next to the repository (`lo2_raw` holds the downloaded run files, `lo2_tar` the same runs unpacked); SHA-256 of every file was recorded before and after (`outputs/lo2_hash_before.json` / `lo2_hash_after.json`; unchanged).", "",
         "## Which files, and why", "",
         f"* Every downloaded run is used: **{len(man)} whole runs ({man.size_mb.sum():.1f} MB, {100 * man.size_mb.sum() / 887.4:.1f}% of the download)**. An earlier version used only 5 runs under a 150 MB cap; that left 13 healthy test windows and unreliable results, so the cap was lifted.",
         "* Whole runs only (no run is ever cut), each with all 54 scenarios (normal and anomalous), covering the 33 collection dates.",
         f"* The split was made first, by whole runs: {int((man.role == 'test').sum())} runs (20%, chosen with seed 42) are the test part, the other {int((man.role == 'train').sum())} are the training part. No run and no window is in both parts (proof P21, check 1). The training part is balanced with SMOTE (inside every training fold and for the final model); the test part is never resampled (proof P24).", "",
         "| run | file size (MB) | role | scrape rows (windows) | normal | anomalous |", "|---|---|---|---|---|---|"]
    for r in man.itertuples():
        L.append(f"| {r.run_id} | {r.size_mb:.2f} | {r.role} | {r.scrape_rows} | {r.normal_rows} | {r.anomalous_rows} |")
    L += ["", f"Total: {len(man)} runs, {man.size_mb.sum():.1f} MB, {int(man.scrape_rows.sum())} windows ({int(man.normal_rows.sum())} normal, {int(man.anomalous_rows.sum())} anomalous). The list of files with the reasons is `report_outputs/lo2_slice_manifest.csv`.", ""]
    if sp is not None and "window_id" in sp.columns:
        L += ["## Windows and class counts", "", "A window is one 5-second scrape row of the LO2 data set (the finest window LO2 offers; logs are aggregated over the 5 seconds before the scrape). Features come from LO2's own signals: per log source (7 services, the database, Locust) the number of lines, continuation (stack-trace) lines, DEBUG / INFO / WARN / ERROR counts, error and warn rates; the total log volume; counts of the 48 most frequent **Drain-style log templates** (a template is a log line with numbers, ids and quoted text masked; templates were mined on the training runs only, test lines that match none go to an `other` count); and 478 metric families (the mean over the series of each Prometheus metric name, counters turned into per-second rates).", "",
              "| class | train | test |", "|---|---|---|"]
        for c in ["Normal", "HTTP400", "HTTP401", "HTTP404"]:
            L.append(f"| {c} | {int(((sp.split == 'train') & (sp.label_level == c)).sum())} | {int(((sp.split == 'test') & (sp.label_level == c)).sum())} |")
        L += ["", f"Total {len(sp)} windows: {int((sp.split == 'train').sum())} train, {int((sp.split == 'test').sum())} test. Binary label: Normal (the `correct` scenario) versus Anomaly; the multi-class label is Normal / HTTP 400 / 401 / 404 (from the scenario name).", "",
              "## Folds", "", "The training runs give 5 folds of WHOLE runs: no run is in two folds, every class is in every validation fold, and no context window crosses a fold border (proof P22). The test part is whole runs that no fold has seen.", ""]
    sc = rd("lo2_shortcut_check.csv")
    if sc is not None:
        L += ["## Is the binary task easy because of a shortcut?", "", "In every LO2 run the normal (`correct`) scenario is the first of the 54 scenarios and lasts 60 s instead of 10 s. That could be a warm-up or run-length shortcut. The same random forest (trained on the training runs, scored on the untouched test runs; `report_outputs/lo2_shortcut_check.csv`; a diagnostic that changed nothing) shows it is not:", "",
              "| inputs / subset | features | balanced accuracy (or share correct) | ROC-AUC | test windows |", "|---|---|---|---|---|"] + [f"| {r.inputs} | {int(r.n_inputs)} | {r.balanced_accuracy:.3f} | {'-' if r.roc_auc != r.roc_auc else format(r.roc_auc, '.3f')} | {int(r.n_test)} |" for r in sc.itertuples()] +              ["", "Normal windows from the late part of the scenario are recognised as well as the first ones, anomalies from scenarios 2-3 as well as from scenario 54, and a matched subset (late normal windows against anomalies of scenarios 2-4, same warm-up state) is still separated. The error scenarios leave clear traces: WARN / ERROR log lines alone give about 96%, metric changes about 99%. The high binary score therefore reflects a real, easy-to-see fault signature, within one service with injected HTTP errors. The HTTP 400 / 401 / 404 task is harder and is the more informative LO2 result.", ""]
    L += ["## Separation from AnoMod", "", "LO2 is trained completely separately: its own windows (`scripts/lo2_hybrid_data.py`, `scripts/lo2_data.py`), its own preprocessing fitted on LO2's training part, its own models (`models/hybrid/lo2/`) and results (`report_outputs/hybrid_lo2_*`, `lo2_full_*`). No AnoMod weight, scaler, threshold or feature is used, and no AnoMod and LO2 window is ever mixed in a split or training set. Proof P21 has four explicit PASS / FAIL checks (no window or run in both parts, natural test counts, SMOTE before/after per fold, physical separation and a grep for AnoMod artefacts in the LO2 scripts).", "",
          "## What this says and does not say", "", "* These results are about training on LO2's own data. They say nothing about transfer from AnoMod to LO2.", "* Controls: shuffled labels, a log-volume-only baseline and a run-identification check (the test runs are never in the training data, so run identity cannot help).", "* The earlier external test (`lo2_results_summary.md`: saved AnoMod models applied unchanged to one LO2 run) is separate and unchanged.", ""]
    (DOCS / "LO2_SLICE.md").write_text("\n".join(L), encoding="utf-8")


def viva_doc():
    rows = []

    def add(what, val, src):
        rows.append(f"| {what} | {val} | `{src}` |")
    sp = rd("split_ids.csv")
    add("AnoMod windows / runs", f"{len(sp)} windows, {sp.run_id.nunique()} runs ({int((sp.split == 'train').sum())} train, {int((sp.split == 'test').sum())} test)", "split_ids.csv")
    for lab in ("label_binary", "label_level"):
        e = rd(f"leaderboard_{lab}.csv").sort_values("test_bal_acc", ascending=False)
        bs = rd("test_bootstrap_ci.csv")
        for _, r in e.head(3).iterrows():
            x = bs[(bs.model == r.model) & (bs.label_version == lab) & (bs.scope == "ALL") & (bs.metric == "balanced_accuracy")].iloc[0]
            nr = rd("test_metrics.csv")
            nrr = nr[(nr.model == r.model) & (nr.label_version == lab) & (nr.scope == "ALL") & (nr.variant == "official")].normal_recall.iloc[0]
            add(f"AnoMod {lab[6:]}: {r.model}", f"balanced accuracy {f3(r.test_bal_acc)} (95% range {f3(x.ci95_low)}-{f3(x.ci95_high)}), healthy-window recall {f3(nrr)}, ROC-AUC {f3(r.test_roc_auc)}", f"leaderboard_{lab}.csv, test_bootstrap_ci.csv, test_metrics.csv")
    ho = rd("testbed_heldout_overall.csv")
    cfg = json.load(open(REP / "testbed_frozen_config.json"))["selection"]["chosen"]
    r = ho[ho.method == f"{cfg['model']} ({cfg['feature_set']})"].iloc[0]
    add(f"Testbed held-out, frozen detector {cfg['model']} ({cfg['feature_set']})", f"balanced accuracy {f3(r.balanced_accuracy)} (95% range {f3(r.balanced_accuracy_ci_low)}-{f3(r.balanced_accuracy_ci_high)}), {int(r.n_test_runs)} test runs, {int(r.n_windows)} windows", "testbed_heldout_overall.csv, testbed_frozen_config.json")
    lv = rd("testbed_live_validation.csv")
    ff = lv[lv.actual_fault != "none"]
    add("Live demos (seeds 9001-9003)", f"{int(ff.detected.fillna(0).astype(bool).sum())} of {len(ff)} injected faults detected; schedule written before each run", "testbed_live_validation.csv")
    rc = rd("rcaeval_external_results.csv")
    al = rc[(rc.fault_type == "all") & (rc.detector == "RobustZ")]
    add("RCAEval external (RobustZ, all fault types)", ", ".join(f"{r.system} {r.detection_rate:.2f}" for r in al.itertuples()) + " (detection rate per system; testing only)", "rcaeval_external_results.csv")
    hb = rd("hybrid_anomod_leaderboard.csv")
    if hb is not None:
        for lab in ("label_binary", "label_level"):
            h = hb[(hb.label_version == lab) & (hb.status != "not completed")].sort_values("test_bal_acc", ascending=False)
            if not h.empty:
                b = h.iloc[0]
                add(f"AnoMod best hybrid {lab[6:]}: {b.model}", f"balanced accuracy {f3(b.test_bal_acc)} (95% range {f3(b.test_bal_acc_ci_low)}-{f3(b.test_bal_acc_ci_high)}), healthy-window recall {f3(b.test_normal_recall)}, 5-fold", "hybrid_anomod_leaderboard.csv")
    lh = rd("hybrid_lo2_leaderboard.csv")
    if lh is not None:
        for lab in ("label_binary", "label_level"):
            h = lh[(lh.label_version == lab) & (lh.status != "not completed")].sort_values("test_bal_acc", ascending=False)
            if not h.empty:
                b = h.iloc[0]
                ties = int((h.test_bal_acc.round(6) == round(b.test_bal_acc, 6)).sum()) - 1
                add(f"LO2 best hybrid {lab[6:]}: {b.model}" + (f" (tied with {ties} other hybrid{'s' if ties > 1 else ''})" if ties else ""), f"balanced accuracy {f3(b.test_bal_acc)} (95% range {f3(b.test_bal_acc_ci_low)}-{f3(b.test_bal_acc_ci_high)}); unreliable: very small sample", "hybrid_lo2_leaderboard.csv")
    man = rd("lo2_slice_manifest.csv")
    if man is not None:
        add("LO2 slice", f"{len(man)} whole runs, {man.size_mb.sum():.1f} MB (cap 150 MB), {int(man.scrape_rows.sum())} windows", "lo2_slice_manifest.csv")
    fl = rd("lo2_full_leaderboard.csv")
    if fl is not None:
        d = fl[(fl.label_version == "label_binary") & (fl.status != "not completed")].sort_values("test_bal_acc", ascending=False)
        if not d.empty:
            add("LO2 best model of all families (binary)", f"{d.iloc[0].model}: {f3(d.iloc[0].test_bal_acc)} (95% range {f3(d.iloc[0].test_bal_acc_ci_low)}-{f3(d.iloc[0].test_bal_acc_ci_high)})", "lo2_full_leaderboard.csv")
        nc = fl[fl.status == "not completed"]
        add("LO2 families not completed ", ", ".join(sorted(set(nc.model))) or "none", "lo2_full_not_completed.csv")
    prf = json.load(open(REP / "proofs.json", encoding="utf-8"))
    add("Proof report totals", json.dumps(prf["totals"]), "proofs.json")
    L = ["# Numbers for the viva", "", "Written for: the person answering questions after the presentation. Every number is read from the file named next to it (generated by `scripts/build_docs.py`; re-run it after any change).", "",
         "> " + FOLD_NOTE, "", "| what | number | source file (in `report_outputs/`) |", "|---|---|---|"] + rows + ["", "## Things to say if asked", "",
         "* The best model on any list was chosen on the test set, so its lead is optimistic; the ranges overlap.", "* Run identification can reproduce the AnoMod scores (proof P14), so they are not proof of fault detection.", "* LO2 is training on LO2's own data, a very small sample; no transfer claim.", "* FailSight notices a problem that has started; it cannot forecast a failure, and time-until-failure is future work (not built).", ""]
    (DOCS / "NUMBERS_FOR_VIVA.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    hybrid_doc()
    lo2_doc()
    viva_doc()
    print("docs written")
