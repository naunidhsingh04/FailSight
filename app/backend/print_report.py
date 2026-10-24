"""Extra pages of the printable proof report: contents list, how the data were split and balanced (80% SMOTE-balanced / 20% untouched), the hybrid-model results on AnoMod, LO2 and both together,
the checks behind them, plain-English terms and the limits. Every number is read from report_outputs/*.csv through hybrid_results; nothing is computed here except simple counts and formatting."""
import html

import pandas as pd

from . import config, hybrid_results

e = html.escape
LABEL = {"label_binary": "Normal / Anomaly", "label_level": "Five-class"}
TERMS = [
    ("Balanced accuracy", "The average of the share of healthy windows found and the share of faulty windows found. 50% is a coin flip; it is not fooled by having many more faulty than healthy windows."),
    ("95% range", "A bootstrap range: if the test were drawn again many times from the same kind of data, 95 of 100 scores would fall inside it. A wide range means few test windows."),
    ("Healthy-window recall", "Of the windows that really were healthy, how many were called healthy."),
    ("ROC-AUC", "The chance that a random faulty window gets a higher fault score than a random healthy one. 100% is perfect, 50% is chance."),
    ("SMOTE", "Synthetic Minority Over-sampling: makes new, in-between practice windows for the rarer classes so every class has as many practice windows as the biggest class. It is applied to the practice (training) part only; the test part is never touched."),
    ("80% / 20% split", "Whole runs are set aside first. 80% of the data is for practice and for choosing settings (with 5-fold checking inside it); 20% is the test, scored once after the settings were written to a file."),
    ("Shuffled-label control", "The same recipe is trained with the practice answers mixed up. If it still scored well, the score would be a trick of the data; it should fall to about 50%."),
    ("Hybrid model", "Two kinds of neural network joined in one model, for example a Transformer that reads the whole window followed by an LSTM that reads it in order."),
]


def _pct(x, d=1):
    return "-" if x is None or x != x else f"{100 * float(x):.{d}f}%"


def _table(head, rows, cls=""):
    h = "".join(f"<th>{e(str(c))}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{e(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table class='{cls}'><tr>{h}</tr>{b}</table>"


def _model_rows(models, label):
    rows = []
    for m in sorted([m for m in models if m["label_version"] == label], key=lambda m: -(m.get("test_bal_acc") or -1)):
        if m.get("status") not in ("done", "did not learn"):
            rows.append([m["model"], m.get("status", "not completed"), "-", "-", "-", "-", "-"])
            continue
        rows.append([m["model"], f"{_pct(m['test_bal_acc'])} ({_pct(m.get('test_bal_acc_ci_low'))} to {_pct(m.get('test_bal_acc_ci_high'))})", _pct(m.get("test_normal_recall")), _pct(m.get("test_roc_auc")),
                     _pct(m.get("shuffled_balanced_accuracy")), str(m.get("method", "")).replace("cw", "class weights"), f"{(m.get('train_seconds_total') or m.get('train_secs') or 0) / 60:.1f} min"])
    return rows


HEAD = ["model", "test balanced accuracy (95% range)", "healthy-window recall", "ROC-AUC", "shuffled-label control", "imbalance method", "training time"]


def _smote_rows(prefix, labels, names):
    """one row per final model: counts before / after SMOTE; a model that chose class weights shows equal counts (SMOTE not applied to it)"""
    p = config.REPORTS / f"{prefix}smote_counts.csv"
    if not p.exists():
        return []
    d = pd.read_csv(p)
    d = d[d.stage == "final"]
    rows = []
    for r in d.sort_values(["label_version", "model"]).itertuples():
        if r.label_version in labels:
            rows.append([r.model, names[r.label_version], r.before, r.after if r.after != r.before else f"{r.after} (class weights, no SMOTE)", str(r.held_out_fold)])
    return rows


def overview(rep: dict) -> str:
    """contents list and the pages on data split, SMOTE balance and the hybrid-model results, as HTML"""
    hy = hybrid_results.hybrids()
    bo = hybrid_results.both()
    s = []
    toc = ["<nav class='toc'><h2>Contents</h2><ol>",
           "<li>How the data were split and balanced</li><li>Hybrid models on AnoMod</li><li>Hybrid models on LO2</li><li>Both data sets trained together</li>",
           "<li>Proof sections P1 to P24 (every check with its evidence)</li><li>Summary of checks</li><li>Plain-English terms</li><li>Limits</li></ol></nav>"]
    s.append("<h2 class='pb'>1. How the data were split and balanced</h2>")
    af, lf = hy.get("anomod", {}).get("facts"), hy.get("lo2", {}).get("facts")
    rows = []
    if af:
        rows.append(["AnoMod", af["windows"], f"{af['train']} ({100 * af['train'] / af['windows']:.0f}%)", f"{af['test']} ({100 * af['test'] / af['windows']:.0f}%)", af["healthy_test_windows"], "kept as recorded"])
    if lf and "windows" in lf:
        rows.append(["LO2 slice", lf["windows"], f"{lf['train']} ({100 * lf['train'] / lf['windows']:.0f}%) - SMOTE-balanced", f"{lf['test']} ({100 * lf['test'] / lf['windows']:.0f}%)", lf["healthy_test_windows"], "kept as recorded"])
    if bo.get("facts"):
        f = bo["facts"]
        rows.append(["AnoMod + LO2 together", f["train_anomod"] + f["test_anomod"] + f["train_lo2"] + f["test_lo2"], f["train_anomod"] + f["train_lo2"], f["test_anomod"] + f["test_lo2"], f["healthy_test_anomod"] + f["healthy_test_lo2"], "kept as recorded"])
    s.append(_table(["data", "windows", "80% practice part", "20% test part", "healthy test windows", "test part"], rows))
    s.append("<p>Whole runs are assigned to the practice or test part before any window exists. SMOTE is fitted on the practice part only (and inside each practice fold); the test part keeps the class counts it was recorded with. "
             "Counts before / after SMOTE for the final models (classes in label order):</p>")
    sm = []
    lv = {"label_binary": "Normal / Anomaly"}
    for name, prefix, lvl in (("AnoMod", "hybrid_anomod_", "Five-class"), ("LO2", "hybrid_lo2_", "HTTP class"), ("Both", "hybrid_both_", None)):
        names = {**lv, "label_level": lvl}
        for r in _smote_rows(prefix, list(names) if lvl else ["label_binary"], names):
            sm.append([name] + r)
    s.append(_table(["data", "model", "labels", "practice part before SMOTE", "after SMOTE", "test part (untouched)"], sm))
    if lf:
        s.append(f"<p class='note'>LO2: all {lf['runs']} downloaded runs ({lf['megabytes']:.1f} MB); {lf.get('test_runs')} whole runs are the untouched 20% test part ({lf['healthy_test_windows']} healthy test windows).</p>")
    s.append("<h2>The ten hybrid models at a glance (both label versions, each data set trained on its own)</h2>")
    glance = []
    for ds, key in (("AnoMod", "anomod"), ("LO2", "lo2")):
        ms = hy.get(key, {}).get("models", [])
        for n in dict.fromkeys(m["model"] for m in ms):
            cell = lambda lab: next((m for m in ms if m["model"] == n and m["label_version"] == lab and m.get("status") in ("done", "did not learn")), None)
            b, l = cell("label_binary"), cell("label_level")
            fmt = lambda m: "not completed" if m is None else f"{_pct(m['test_bal_acc'])} ({_pct(m.get('test_bal_acc_ci_low'))} to {_pct(m.get('test_bal_acc_ci_high'))})"
            glance.append([ds, n, fmt(b), fmt(l)])
    s.append(_table(["data set", "hybrid model", "Normal / Anomaly: balanced accuracy", "kind of problem: balanced accuracy"], glance))
    for i, (title, key) in enumerate((("2. Hybrid models on AnoMod", "anomod"), ("3. Hybrid models on LO2", "lo2")), 2):
        s.append(f"<h2 class='pb'>{title}</h2>")
        ms = hy.get(key, {}).get("models", [])
        if not ms:
            s.append("<p>No results yet.</p>")
            continue
        for lab in ("label_binary", "label_level"):
            s.append(f"<h3>{e(LABEL[lab] if key == 'anomod' or lab == 'label_binary' else 'HTTP class (Normal / 400 / 401 / 404)')}</h3>" + _table(HEAD, _model_rows(ms, lab)))
        if key == "anomod":
            for lab, c in hy["anomod"].get("comparison", {}).items():
                verdict = "beats" if c["hybrid_beats_best_existing"] else "does not beat"
                s.append(f"<p>{e(LABEL[lab])}: best hybrid <b>{e(c['best_hybrid'])}</b> {_pct(c['best_hybrid_ba'])} {verdict} the best of the 36 earlier models <b>{e(c['best_existing'])}</b> {_pct(c['best_existing_ba'])}"
                         f"{'; the 95% ranges overlap' if c['ranges_overlap'] else ''}.</p>")
        else:
            s.append("<p class='note'>The binary LO2 score is high because faults leave clear error lines and metric changes; controls in P21 show it is not a warm-up or run-length shortcut.</p>")
    s.append("<h2 class='pb'>4. Both data sets trained together</h2>")
    ms = [m for m in bo.get("rows", []) if m.get("family") == "Hybrid model"]
    if ms:
        rows = [[m["model"], _pct(m.get("test_bal_acc")) if m.get("status") == "done" else m.get("status", "not completed"), _pct(m.get("ba_AnoMod")), _pct(m.get("ba_LO2")), _pct(m.get("test_normal_recall")), _pct(m.get("test_roc_auc"))]
                for m in sorted(ms, key=lambda m: -(m.get("test_bal_acc") or -1))]
        s.append(_table(["hybrid model", "pooled test balanced accuracy", "AnoMod part", "LO2 part", "healthy-window recall", "ROC-AUC"], rows))
        s.append(f"<p class='note'>{e(bo.get('note', ''))}</p>")
    else:
        s.append("<p>No combined results yet.</p>")
    return "".join(toc) + "".join(s)


def closing(rep: dict) -> str:
    t = "<h2 class='pb'>Plain-English terms</h2><dl>" + "".join(f"<dt>{e(a)}</dt><dd>{e(b)}</dd>" for a, b in TERMS) + "</dl>"
    t += ("<h2>Limits</h2><ul><li>AnoMod: only 36 healthy test windows, so ranges are wide and no hybrid beats the best earlier model.</li>"
          "<li>LO2: one OAuth2 service with injected HTTP errors. The binary task scores high because errors leave clear traces in logs and metrics (checked against a warm-up / run-length shortcut, P21); it says nothing about other systems.</li>"
          "<li>The earlier AnoMod models used 10-fold checking, the hybrids 5-fold; all AnoMod models are scored on the same 647 test windows.</li>"
          "<li>Training on both data sets together says how one model copes with two systems; it is not a transfer test.</li>"
          "<li>The live demo is a replay of recorded data, not a production system. Nothing here forecasts failures.</li></ul>")
    return t
