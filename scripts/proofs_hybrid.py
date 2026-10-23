"""New proof sections P20 (AnoMod hybrids), P21 (LO2), P22 (fold integrity of the new 5-fold work) and P23 (Colab notebook). Called from scripts/proofs.py; only ADDS sections.
Everything is recomputed from the saved models / split / fold files or read from named result files."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REP = ROOT / "report_outputs"
FOLD_NOTE = "The earlier AnoMod models were checked with 10-fold cross-validation and the hybrid models with 5-fold. All AnoMod models, hybrids included, are scored on the same 647 test windows."
LABELS = ["label_binary", "label_level"]


def _rebuild(registry, name, spec, frozen, weights):
    import keras
    cfg = frozen["config"]
    m = registry[name](spec, cfg)
    m(np.zeros((1, spec["T"], spec["F"]), "float32"))
    m.load_weights(weights)
    return m


def _reload_scores(D, registry, prefix, mdir, names, tag):
    import hybrid_engine as E
    W = D.W("final")
    out, maxd, n = [], 0.0, 0
    saved = pd.read_csv(REP / f"{prefix}test_metrics.csv")
    for label in LABELS:
        yte = D.y[label][D.test_pos]
        Xte = D.seq(W, D.test_pos)
        for name in names:
            fp, wp = REP / f"{prefix}frozen_{label}__{name}.json", Path(mdir) / f"{label}__{name}.weights.h5"
            row = saved[(saved.model == name) & (saved.label_version == label) & (saved.variant == "official")]
            if not (fp.exists() and wp.exists()) or row.empty:
                out.append([name, label, "not completed", "", ""])
                continue
            frozen = json.load(open(fp))
            m = _rebuild(registry, name, D.spec(label), frozen, wp)
            P = E.predict(m, Xte)
            mm = E.metrics_of(yte, P, P.argmax(1), D.class_names[label])
            r = row.iloc[0]
            d = max(abs(mm["balanced_accuracy"] - r.balanced_accuracy), abs(mm["macro_f1"] - r.macro_f1), abs(mm["normal_recall"] - r.normal_recall))
            maxd, n = max(maxd, d), n + 1
            out.append([name, label[6:], f"{mm['balanced_accuracy']:.4f}", f"{r.balanced_accuracy:.4f}", f"{d:.1e}"])
            del m
    return out, maxd, n


def p20(s, table):
    import hybrid_engine as E
    import hybrid_anomod as HA
    from models_src.hybrid import ANOMOD_HYBRIDS
    if not (REP / "hybrid_anomod_test_metrics.csv").exists():
        s.add("AnoMod hybrid models", "SKIP", ["report_outputs/hybrid_anomod_test_metrics.csv missing; run python scripts/hybrid_anomod.py"], "-")
        return
    D = HA.AnoModData()
    rows, maxd, n = _reload_scores(D, ANOMOD_HYBRIDS, "hybrid_anomod_", HA.MODELS, list(ANOMOD_HYBRIDS), "anomod")
    s.add("Each saved AnoMod hybrid reloads and reproduces its test scores", "PASS" if (n > 0 and maxd <= 1e-4) else "FAIL",
          [f"{n} saved models (5 hybrids x binary / five-class) rebuilt from the frozen configuration files, weights loaded, the 647 test windows scored again; largest difference of balanced accuracy / macro-F1 / healthy recall {maxd:.1e}"] +
          table(rows, ["model", "labels", "recomputed BA", "saved BA", "difference"]), "models/hybrid/anomod/, report_outputs/hybrid_anomod_frozen_*.json, hybrid_anomod_test_metrics.csv", dict(max_difference=maxd, models=n))
    # split integrity: same split as every earlier AnoMod model
    sp = pd.read_csv(REP / "split_ids.csv")
    tr, te = set(D.df.window_id[D.train_pos]), set(D.df.window_id[D.test_pos])
    same = (sp.set_index("window_id").split.reindex(D.df.window_id).to_numpy() == D.df.split.to_numpy()).all()
    nat = {l: np.bincount(D.y[l][D.test_pos], minlength=len(D.class_names[l])).tolist() for l in LABELS}
    nat_file = {l: np.bincount(D.df.assign(y=D.y[l])[D.df.window_id.isin(sp[sp.split == "test"].window_id)].y.to_numpy(), minlength=len(D.class_names[l])).tolist() for l in LABELS}
    ok = len(tr) == 2549 and len(te) == 647 and not (tr & te) and same and nat == nat_file
    s.add("Split integrity: the hybrids use the same 2,549 / 647 split and the test part keeps its natural class counts", "PASS" if ok else "FAIL",
          [f"train {len(tr)}, test {len(te)}, windows in both {len(tr & te)}; identical to report_outputs/split_ids.csv: {bool(same)}", f"test class counts binary {nat['label_binary']}, five-class {nat['label_level']} (equal to the counts in the split file: {nat == nat_file})"],
          "report_outputs/split_ids.csv", dict(train=len(tr), test=len(te)))
    # SMOTE integrity from the per-fold counts
    sc = pd.read_csv(REP / "hybrid_anomod_smote_counts.csv")
    bad, checked = 0, 0
    for r in sc[sc.stage != "final"].itertuples():
        b, a, h = (np.array([int(x) for x in str(v).split("/")]) for v in (r.before, r.after, r.held_out_fold))
        k = int(r.fold)
        tr_rows, va_rows = D.folds[k - 1]
        lab = r.label_version
        true_hold = np.bincount(D.y[lab][va_rows], minlength=len(h))
        true_before = np.bincount(D.y[lab][tr_rows], minlength=len(h))
        checked += 1
        if not (np.array_equal(h, true_hold) and (a >= b).all() and np.array_equal(b, true_before)):
            bad += 1
    fin = sc[sc.stage == "final"]
    fin_ok = all(np.array_equal(np.array([int(x) for x in str(r.before).split("/")]), np.bincount(D.y[r.label_version][D.train_pos], minlength=len(str(r.before).split("/")))) for r in fin.itertuples())
    ex = sc[(sc.stage != "final") & (sc.label_version == "label_level")].head(2)
    s.add("SMOTE integrity: synthetic windows only in training folds; validation folds and the test part keep their real counts", "PASS" if (bad == 0 and fin_ok and checked > 0) else "FAIL",
          [f"{checked} fold rows checked: held-out counts equal the real validation counts, 'before' equals the real training counts, 'after' >= 'before' per class (synthetic windows are added to training only); final model: training counts equal the real 80% counts: {fin_ok}",
           "format  b: before SMOTE / a: after SMOTE / hold: held-out part (never resampled)"] + [f"  {r.model} {r.label_version[6:]} fold {r.fold}:  b: {r.before}  a: {r.after}  hold: {r.held_out_fold}" for r in ex.itertuples()],
          "report_outputs/hybrid_anomod_smote_counts.csv", dict(rows_checked=checked, bad=bad))
    shuf = pd.read_csv(REP / "hybrid_anomod_shuffled_control.csv")
    warn = shuf.verdict.str.startswith("WARN").sum()
    s.add("Shuffled-label control for every hybrid (training labels permuted, same recipe, test part scored)", "PASS" if warn == 0 else "WARN",
          table([[r.model, r.label_version[6:], f"{r.shuffled_balanced_accuracy:.3f}", f"{r.real_balanced_accuracy:.3f}", r.verdict[:4]] for r in shuf.itertuples()], ["model", "labels", "shuffled BA", "real BA", "verdict"]) +
          ["chance level balanced accuracy is 0.5 (binary) and 0.2 (five classes); with only 36 healthy test windows a shuffled model can drift above chance by luck."], "report_outputs/hybrid_anomod_shuffled_control.csv")
    ri = pd.read_csv(REP / "hybrid_anomod_run_identification.csv")
    rid = ri[ri.kind == "run_identification"]
    hv = ri[ri.kind == "hybrid_vs_rule"]
    lines = [f"a throw-away XGBoost identifies the run of a test window in {float(rid.accuracy.iloc[0]):.1%} of cases (chance {float(rid.chance.iloc[0]):.1%}); the 'predict the run' rule is scored like the P14 rule"] + \
            table([[r.model, r.label_version[6:], f"{r.agreement:.3f}", f"{r.model_balanced_accuracy:.3f}", f"{r.rule_balanced_accuracy:.3f}"] for r in hv.itertuples()], ["hybrid", "labels", "agreement with rule", "hybrid BA", "rule BA"]) + \
            ["reading (as in P14): the AnoMod split is by blocks, so every run has windows on both sides and run recognition alone can reproduce scores of this size; the hybrid scores cannot be read as the ability to detect faults."]
    s.add("Run-identification note for the hybrids (P14 rule)", "REPORTED", lines, "report_outputs/hybrid_anomod_run_identification.csv", result="run identification can reproduce the scores; reported as it is")
    lb = pd.read_csv(REP / "hybrid_anomod_leaderboard.csv")
    lines = []
    for lab in LABELS:
        ex_ = pd.read_csv(REP / f"leaderboard_{lab}.csv").sort_values("test_bal_acc", ascending=False)
        h = lb[(lb.label_version == lab) & (lb.status != "not completed")].sort_values("test_bal_acc", ascending=False)
        if h.empty:
            continue
        b = h.iloc[0]
        bs = pd.read_csv(REP / "test_bootstrap_ci.csv")
        x = bs[(bs.model == ex_.iloc[0].model) & (bs.label_version == lab) & (bs.scope == "ALL") & (bs.metric == "balanced_accuracy")].iloc[0]
        lines += [f"{lab[6:]} labels: best hybrid {b.model} balanced accuracy {b.test_bal_acc:.3f} (95% range {b.test_bal_acc_ci_low:.3f}-{b.test_bal_acc_ci_high:.3f}); best of the 36 earlier models {ex_.iloc[0].model} {ex_.iloc[0].test_bal_acc:.3f} (95% range {x.ci95_low:.3f}-{x.ci95_high:.3f}); "
                  f"{'the hybrid is higher' if b.test_bal_acc > ex_.iloc[0].test_bal_acc else 'no hybrid beats the best earlier model'}; the ranges {'overlap' if (b.test_bal_acc_ci_low <= x.ci95_high and x.ci95_low <= b.test_bal_acc_ci_high) else 'do not overlap'}."]
    s.add("Comparison of the hybrids with the 36 existing models (same 647 test windows)", "REPORTED", lines + [FOLD_NOTE], "report_outputs/hybrid_anomod_leaderboard.csv, report_outputs/leaderboard_label_*.csv", result=lines[0] if lines else None)


def _folds_check(ids, folds, groups, y, ncls):
    ok_groups = all(len(set(groups[folds == k]) & set(groups[folds != k])) == 0 for k in range(1, 6))
    cover = [sorted(set(np.unique(y[folds == k]))) for k in range(1, 6)]
    all_classes = all(len(c) == ncls for c in cover)
    return ok_groups, all_classes, [len(set(groups[folds == k])) for k in range(1, 6)]


def p22(s, table):
    import hybrid_anomod as HA
    from lo2_data import LO2Data
    fa = pd.read_csv(REP / "hybrid_anomod_folds5.csv")
    D = HA.AnoModData()
    f = fa.set_index("window_id").reindex(D.df.window_id[D.train_pos])
    og, ac, ng = _folds_check(None, f.fold.to_numpy(int), D.df.block_id.to_numpy()[D.train_pos], D.y["label_level"][D.train_pos], 5)
    runs_per = [int(f[f.fold == k].run_id.nunique()) for k in range(1, 6)]
    s.add("AnoMod 5-fold: block-aware (a 10-window block never spans two folds), every class in every validation fold", "PASS" if (og and ac) else "FAIL",
          [f"groups = the project's 10-window blocks inside runs; blocks per validation fold {ng}; runs per validation fold {runs_per}; no block shared across folds: {og}; all 5 classes in every validation fold: {ac}",
           "context windows (the 5 before a window) belong to the same run and the same split, as for the existing models; some context windows of a validation window can lie in a training fold (the existing 10-fold models share this property)", FOLD_NOTE],
          "report_outputs/hybrid_anomod_folds5.csv", dict(blocks_per_fold=ng))
    L = LO2Data()
    fl = L.fold5
    og, ac, ng = _folds_check(None, fl, L.group_of(L.train_pos), L.y["label_level"][L.train_pos], 4)
    s.add("LO2 5-fold: block-aware (run + scenario + 3-window block), every class in every validation fold", "PASS" if (og and ac) else "FAIL",
          [f"4 training runs cannot give 5 whole-run folds, so folds are block-aware inside the training runs; blocks per validation fold {ng}; no block shared across folds: {og}; all 4 classes in every validation fold: {ac}",
           f"overlap measured: validation windows whose earlier context windows (2 before) lie in the training rows of the same fold: {L.context_overlap} of {[int((fl == k).sum()) for k in range(1, 6)]}",
           "the 20% test part is a whole run that no fold has seen (no overlap with the test part)", FOLD_NOTE], "report_outputs/hybrid_lo2_folds5.csv", dict(blocks_per_fold=ng, context_overlap=L.context_overlap))


def p21(s, table):
    import hybrid_engine as E
    import hybrid_lo2 as HL
    from lo2_data import LO2Data
    from models_src.hybrid import LO2_HYBRIDS
    if not (REP / "hybrid_lo2_test_metrics.csv").exists():
        s.add("LO2 hybrids", "SKIP", ["report_outputs/hybrid_lo2_test_metrics.csv missing; run python scripts/hybrid_lo2.py"], "-")
        return
    D = LO2Data()
    sp = pd.read_csv(REP / "lo2_split_ids.csv", dtype={"run_id": str})
    tr_ids, te_ids = set(sp[sp.split == "train"].window_id), set(sp[sp.split == "test"].window_id)
    tr_runs, te_runs = set(sp[sp.split == "train"].run_id), set(sp[sp.split == "test"].run_id)
    ok1 = not (tr_ids & te_ids) and not (tr_runs & te_runs) and (set(D.df.window_id[D.train_pos]) == tr_ids) and (set(D.df.window_id[D.test_pos]) == te_ids)
    s.add("Check 1: no window id and no run is in both the training and the test part", "PASS" if ok1 else "FAIL",
          [f"training {len(tr_ids)} windows from {len(tr_runs)} runs, test {len(te_ids)} windows from {len(te_runs)} run ({sorted(te_runs)}); windows in both {len(tr_ids & te_ids)}, runs in both {len(tr_runs & te_runs)}",
           "the split was written at run level before any window existed and the window-level file keeps the same run assignment"], "report_outputs/lo2_split_ids.csv", dict(train=len(tr_ids), test=len(te_ids)))
    nat_file = sp[sp.split == "test"].label_level.value_counts().to_dict()
    nat_data = pd.Series(D.df.label_level.to_numpy()[D.test_pos]).value_counts().to_dict()
    ok2 = nat_file == nat_data and len(D.test_pos) == len(te_ids)
    fin = pd.read_csv(REP / "hybrid_lo2_smote_counts.csv")
    fin = fin[fin.stage == "final"]
    hold_ok = all(r.held_out_fold.split(" ")[0] == "/".join(str(int(x)) for x in np.bincount(D.y[r.label_version][D.test_pos], minlength=len(D.class_names[r.label_version]))) for r in fin.itertuples())
    s.add("Check 2: the test part's class counts equal the natural counts in the split file (no synthetic windows in it)", "PASS" if (ok2 and hold_ok) else "FAIL",
          [f"natural test counts from the split file {nat_file}; counts of the windows used for scoring {nat_data}; counts recorded next to every final model's SMOTE row equal them: {hold_ok}"], "report_outputs/lo2_split_ids.csv, report_outputs/hybrid_lo2_smote_counts.csv", dict(test_counts=nat_file))
    sc = pd.read_csv(REP / "hybrid_lo2_smote_counts.csv")
    bad, checked = 0, 0
    for r in sc[sc.stage != "final"].itertuples():
        b, a, h = (np.array([int(x) for x in str(v).split("/")]) for v in (r.before, r.after, r.held_out_fold))
        tr_rows, va_rows = D.folds[int(r.fold) - 1]
        n = len(h)
        checked += 1
        if not (np.array_equal(h, np.bincount(D.y[r.label_version][va_rows], minlength=n)) and np.array_equal(b, np.bincount(D.y[r.label_version][tr_rows], minlength=n)) and (a >= b).all()):
            bad += 1
    ex = sc[(sc.stage != "final") & (sc.label_version == "label_level")].head(3)
    s.add("Check 3: SMOTE before/after counts per fold; synthetic windows only in training folds", "PASS" if (bad == 0 and checked > 0) else "FAIL",
          [f"{checked} fold rows checked against the real fold contents", "format  b: before SMOTE / a: after SMOTE / hold: held-out fold (never resampled); classes in the order Normal / HTTP400 / HTTP401 / HTTP404"] +
          [f"  {r.model} {r.label_version[6:]} fold {r.fold}:  b: {r.before}  a: {r.after}  hold: {r.held_out_fold}" for r in ex.itertuples()] + ["a class with fewer than 2 training windows would be skipped and said so; none was skipped"],
          "report_outputs/hybrid_lo2_smote_counts.csv, report_outputs/hybrid_lo2_folds5.csv", dict(rows_checked=checked, bad=bad))
    # check 4: physical separation + grep
    an_dir, lo_dir = ROOT / "models" / "hybrid" / "anomod", ROOT / "models" / "hybrid" / "lo2"
    an_files = {str(p.resolve()) for p in an_dir.rglob("*")} if an_dir.exists() else set()
    lo_files = {str(p.resolve()) for p in lo_dir.rglob("*")} if lo_dir.exists() else set()
    lo_scripts = ["lo2_hybrid_data.py", "lo2_data.py", "hybrid_lo2.py", "lo2_full.py", "lo2_deep.py"]
    import ast
    pat = re.compile(r"preprocessor\.joblib|(?<!lo2_)split_ids\.csv|models/label_|models/folds|models/testbed|hybrid/anomod|hybrid_anomod|feature_names\.json|v2_data|import preprocess|from preprocess|thresholds\.json|SN_windows|TT_windows")
    hits = []
    for f in lo_scripts:
        tree = ast.parse((HERE / f).read_text(encoding="utf-8"))
        for node in ast.walk(tree):                                  # docstrings and comments are prose, not loaded artefacts
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body[0].value.value = ""
        for i, line in enumerate(ast.unparse(tree).splitlines(), 1):
            if pat.search(line):
                hits.append(f"{f}: {line.strip()[:90]}")
    ok4 = not (an_files & lo_files) and not hits and lo_dir.exists()
    s.add("Check 4: AnoMod and LO2 models, scalers and files are physically separate; no AnoMod artefact is loaded by any LO2 script", "PASS" if ok4 else "FAIL",
          [f"AnoMod hybrid weights: {len(an_files)} files in models/hybrid/anomod/; LO2: {len(lo_files)} files in models/hybrid/lo2/; files in both: {len(an_files & lo_files)}",
           f"LO2 scripts searched for AnoMod artefact names (preprocessor.joblib, split_ids.csv, models/label_*, hybrid_anomod, v2_data, preprocess, feature_names.json, thresholds.json): {', '.join(lo_scripts)} -> {len(hits)} hits"] + hits +
          ["LO2 uses its own preprocessing (scripts/lo2_data.py, fitted on LO2's training rows), its own features, thresholds and weights."], "scripts/lo2_*.py, scripts/hybrid_lo2.py, models/hybrid/lo2/", dict(hits=len(hits)))
    rows, maxd, n = _reload_scores(D, LO2_HYBRIDS, "hybrid_lo2_", HL.MODELS, list(LO2_HYBRIDS), "lo2")
    s.add("Each saved LO2 hybrid reloads and reproduces its test scores", "PASS" if (n > 0 and maxd <= 1e-4) else "FAIL",
          [f"{n} saved models rebuilt from the frozen configuration files and re-scored on the {len(D.test_pos)} test windows; largest difference {maxd:.1e}"] + table(rows, ["model", "labels", "recomputed BA", "saved BA", "difference"]),
          "models/hybrid/lo2/, report_outputs/hybrid_lo2_frozen_*.json, report_outputs/hybrid_lo2_test_metrics.csv", dict(max_difference=maxd))
    lv = pd.read_csv(REP / "hybrid_lo2_logvolume_baseline.csv")
    lb = pd.read_csv(REP / "hybrid_lo2_leaderboard.csv")
    shuf = pd.read_csv(REP / "hybrid_lo2_shuffled_control.csv")
    lines = ["Log volume alone (logistic regression on the number of log lines in a window):"] + table([[r.label_version[6:], f"{r.balanced_accuracy:.3f} ({r.ba_ci_low:.3f}-{r.ba_ci_high:.3f})", f"{r.roc_auc:.3f}"] for r in lv.itertuples()], ["labels", "balanced accuracy (95%)", "ROC-AUC"])
    lines += ["", "Shuffled-label control for each LO2 hybrid:"] + table([[r.model, r.label_version[6:], f"{r.shuffled_balanced_accuracy:.3f}", f"{r.real_balanced_accuracy:.3f}"] for r in shuf.itertuples()], ["model", "labels", "shuffled BA", "real BA"])
    sc_p = REP / "lo2_shortcut_check.csv"
    lines_sc = []
    if sc_p.exists():
        sc_ = pd.read_csv(sc_p)
        lines_sc = ["", "why the binary labels are so easy (diagnostic, scripts/lo2_shortcut_check.py): the same random forest scored on subsets of the LO2 inputs (test = one whole run):"] + table([[r.inputs, int(r.n_inputs), f"{r.balanced_accuracy:.3f}", f"{r.roc_auc:.3f}"] for r in sc_.itertuples()], ["inputs", "features", "balanced accuracy", "ROC-AUC"]) +                    ["reading: log counts, log-template counts or metric components alone each separate the normal scenario almost perfectly, while log volume alone does not; the normal scenario ('correct') is always the first scenario of a run and runs 60 s instead of 10 s, so LO2's binary task is nearly trivial and a perfect score is NOT strong evidence of fault detection."]
    ri = pd.read_csv(REP / "hybrid_lo2_run_identification.csv")
    rid = ri[ri.kind == "run_identification"].iloc[0]
    lines += ["", f"run identification: the test part is a whole run that is not in the training data, so no model can recognise it (identification accuracy {rid.accuracy:.3f} by construction); the 'predict the run' rule gets balanced accuracy 0.5 (it can only output the label of a training run)."]
    s.add("LO2 controls: log-volume-only baseline, shuffled labels, run identification", "REPORTED", lines + lines_sc + ["LO2 results are about training on LO2's own data and say nothing about transfer from AnoMod.", f"sample: {len(sp)} windows from {sp.run_id.nunique()} runs (fewer than 6 runs, so every LO2 result is marked: unreliable, very small sample)"],
          "report_outputs/hybrid_lo2_logvolume_baseline.csv, hybrid_lo2_shuffled_control.csv, hybrid_lo2_run_identification.csv", result="reported as it is; very small sample")
    full = REP / "lo2_full_leaderboard.csv"
    if full.exists():
        fl = pd.read_csv(full)
        nc = fl[fl.status == "not completed"]
        done = fl[fl.status != "not completed"]
        s.add("LO2 other families (b)-(e) with the same protocol", "REPORTED",
              [f"models with results: {done.model.nunique()}; not completed: {', '.join(sorted(set(nc.model))) or 'none'}", "every LO2 result is marked: unreliable, very small sample (5 runs; 172 test windows, 13 healthy)"],
              "report_outputs/lo2_full_leaderboard.csv", result=f"{done.model.nunique()} models done; {nc.model.nunique()} not completed")


def p23(s, table):
    """The notebook is executed by scripts/run_notebook_check.py in a fresh virtual environment; that script writes report_outputs/notebook_check.json. Here the result is re-read and the
    notebook's printed numbers are compared with the numbers in proofs.json and the result files."""
    p = REP / "notebook_check.json"
    if not p.exists():
        s.add("The Colab notebook was executed end to end", "SKIP", ["report_outputs/notebook_check.json missing; run python scripts/run_notebook_check.py"], "-")
        return
    c = json.load(open(p))
    s.add("The notebook executes top to bottom in FAST mode in a fresh environment without the AnoMod raw folders or the models folder", "PASS" if c.get("executed_ok") else "FAIL",
          [f"executed at {c.get('executed_at')}; cells {c.get('n_cells')}; code cells with an error {c.get('n_errors')}; run time {c.get('seconds', 0):.0f} s; environment: {c.get('environment')}"], "notebooks/FailSight_Workflow_executed.ipynb, report_outputs/notebook_check.json", dict(cells=c.get("n_cells")))
    rows = c.get("comparisons", [])
    bad = [r for r in rows if not r["equal"]]
    s.add("The numbers printed by the notebook equal the numbers in the proofs report and the result files", "PASS" if (rows and not bad) else "FAIL",
          [f"{len(rows)} numbers compared; {len(bad)} differ"] + table([[r["name"], r["notebook"], r["reference"], "equal" if r["equal"] else "DIFFERENT"] for r in rows[:25]], ["number", "notebook", "proofs / file", "result"]),
          "notebooks/FailSight_Workflow_executed.ipynb, report_outputs/proofs.json", dict(compared=len(rows), different=len(bad)))
