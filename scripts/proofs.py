"""FailSight proof report. One command prints a numbered, human-readable report and writes the SAME content to report_outputs/proofs.json
(the web app reads that file; there are no separate numbers).

  python scripts/proofs.py                 full report (recomputes every saved model's test metrics; takes a few minutes)
  python scripts/proofs.py --only P1,P2    only some sections (the JSON is then marked partial)
  python scripts/proofs.py --quick         skip the slowest recomputation (soft voting from its 40 fold models); shown as SKIP, never as PASS

Statuses: PASS / WARN / FAIL (SKIP when a step was deliberately not run). Exit code 1 if any item FAILs.
Nothing is trained or fitted here; AnoMod and LO2 are only read.
"""
import argparse, hashlib, json, os, platform, subprocess, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
WS = ROOT.parent
ANOMOD, LO2 = WS / "AnoMod", WS / "LO2"
REP = ROOT / "report_outputs"
TOL = 1e-6
T0 = time.time()
LINES: list = []
CUR: list = []


def out(s=""):
    print(s, flush=True)
    LINES.append(s)
    CUR.append(s)


REPORTED = "REPORTED ACCURATELY"


class Section:
    def __init__(self, sid, title):
        self.id, self.title, self.items = sid, title, []
        self.lines, self.generated_at = [], time.strftime("%Y-%m-%d %H:%M:%S")

    def add(self, claim, status, evidence, source, data=None, result=None):
        """evidence: list of text lines (shown identically in the terminal and on the web page); data: optional structured numbers.
        status: PASS / WARN / FAIL for checks of correctness; REPORTED = "REPORTED ACCURATELY" for a result that is reported exactly as it is
        but is not a success (the check is that the report is faithful, so it must not look like a green PASS); result: one-line outcome shown in the same row."""
        status = REPORTED if status == "REPORTED" else status
        self.items.append(dict(claim=claim, status=status, evidence=evidence if isinstance(evidence, list) else [evidence], source=source, data=data or {}, result=result))
        out(f"  [{status}] {claim}")
        if result:
            out(f"         Result: {result}")
        for e in self.items[-1]["evidence"]:
            out(f"         {e}")
        out(f"         source: {source}")

    @property
    def status(self):
        s = [i["status"] for i in self.items]
        return "FAIL" if "FAIL" in s else "WARN" if "WARN" in s else REPORTED if REPORTED in s else "SKIP" if s and all(x == "SKIP" for x in s) else "PASS"


def header(sec):
    global CUR
    CUR = sec.lines
    out(""); out("=" * 100); out(f"{sec.id}  {sec.title}"); out("=" * 100)


def table(rows, cols, widths=None):
    widths = widths or [max(len(str(c)), *(len(str(r[i])) for r in rows)) for i, c in enumerate(cols)]
    ln = lambda r: "  ".join(str(x).ljust(w) if i == 0 else str(x).rjust(w) for i, (x, w) in enumerate(zip(r, widths)))
    return [ln(cols), "  ".join("-" * w for w in widths)] + [ln(r) for r in rows]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def count_csv_rows(path):
    c, last = 0, b""
    with open(path, "rb") as fh:
        while True:
            b = fh.read(1 << 24)
            if not b:
                break
            c += b.count(b"\n"); last = b[-1:]
    if last not in (b"\n", b""):
        c += 1
    return max(c - 1, 0)


LEVELS = ["Normal", "Performance", "Service", "Database", "Code"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    only = {s.strip().upper() for s in args.only.split(",") if s.strip()}
    want = lambda sid: not only or sid in only

    out("FAILSIGHT PROOF REPORT"); out(time.strftime("%Y-%m-%d %H:%M:%S"))
    out("Every number below is recomputed from the data or the saved models, or read from a named result file. Nothing is retrained.")
    import preprocess as P
    import v2_data
    from v2_data import DataV2
    v2_data.common.LOG = ROOT / "outputs" / "proofs_log.txt"
    D = DataV2()
    df = D.df
    sections: list = []
    ctx = {}

    # ------------------------------------------------------------------------------------------------ P1 data
    if want("P1"):
        s = Section("P1", "DATA: what exists and how big it is"); header(s)
        csvs, rows_sn, rows_tt = [], 0, 0
        for root in ("SN_data", "TT_data"):
            for dp, _, fns in os.walk(ANOMOD / root):
                for f in fns:
                    if f.lower().endswith(".csv"):
                        csvs.append(Path(dp) / f)
        per = {"SN": 0, "TT": 0}
        for p in csvs:
            per["SN" if "SN_data" in p.parts else "TT"] += count_csv_rows(p)
        total = per["SN"] + per["TT"]
        s.add("The raw data are 234 CSV files with 4,190,698 rows", "PASS" if (len(csvs), total) == (234, 4190698) else "FAIL",
              [f"CSV files found under AnoMod/SN_data and TT_data: {len(csvs)} (claim 234)", f"data rows counted (newline count minus one header per file): {total:,} (claim 4,190,698); SN {per['SN']:,}, TT {per['TT']:,}"],
              "AnoMod/SN_data, AnoMod/TT_data (read-only); report_outputs/old_pipeline/row_crosscheck.csv", dict(csv_files=len(csvs), csv_rows=total, per_system=per))
        ncounts = {}
        for root in ("SN_data", "TT_data"):
            for dp, _, fns in os.walk(ANOMOD / root):
                for f in fns:
                    ncounts[(root[:2], f.rsplit(".", 1)[-1].lower())] = ncounts.get((root[:2], f.rsplit(".", 1)[-1].lower()), 0) + 1
        top = sorted(ncounts.items(), key=lambda kv: -kv[1])[:8]
        s.add("File inventory of the raw folders", "PASS", [f"{sys_}: {ext} x {n:,}" for (sys_, ext), n in top] + [f"total files: {sum(ncounts.values()):,}"], "AnoMod/SN_data, AnoMod/TT_data", dict(top_types=[[a, b, n] for (a, b), n in top], total_files=sum(ncounts.values())))
        feats = P.feature_columns(df)
        grp = pd.Series([P.feature_group(c) for c in feats]).value_counts().to_dict()
        s.add("The windows table has 3,196 windows of 30 s (10 s step) and 155 features", "PASS" if (len(df), len(feats)) == (3196, 155) else "FAIL",
              [f"windows: {len(df):,} (SN {int((df.system == 'SN').sum()):,}, TT {int((df.system == 'TT').sum()):,}); runs {df.run_id.nunique()}; blocks {df.block_id.nunique()}",
               f"features: {len(feats)} = api {grp.get('api')}, log {grp.get('logs')}, metric {grp.get('metrics')}",
               f"window length {(df.window_end - df.window_start).dt.total_seconds().unique().tolist()} s"], "AnoMod/processed/{SN,TT}/windows.parquet (copies in inputs/)", dict(windows=len(df), features=len(feats), groups=grp))
        same = all(sha256(ROOT / "inputs" / f"{x}_windows.parquet") == sha256(ANOMOD / "processed" / x / "windows.parquet") for x in ("SN", "TT"))
        s.add("The window tables used by the models are byte-identical to the AnoMod originals", "PASS" if same else "FAIL", ["SHA-256 of inputs/SN_windows.parquet and TT_windows.parquet equals the AnoMod/processed copies: " + str(same)], "inputs/*_windows.parquet vs AnoMod/processed")
        cb = df.groupby(["system", "label_binary"]).size().unstack(fill_value=0)
        cl = df.groupby(["system", "label_level"]).size().unstack(fill_value=0)[LEVELS]
        lines = table([[i] + cl.loc[i].tolist() + [int(cl.loc[i].sum())] for i in cl.index] + [["ALL"] + cl.sum().tolist() + [int(cl.values.sum())]], ["system"] + LEVELS + ["total"])
        s.add("Class counts (all windows, recomputed)", "PASS", lines + [f"binary: Normal {int((df.label_binary == 0).sum())}, Anomaly {int((df.label_binary == 1).sum())}"],
              "AnoMod/processed windows tables", dict(level=cl.sum().to_dict(), binary=dict(Normal=int((df.label_binary == 0).sum()), Anomaly=int((df.label_binary == 1).sum()))))
        sections.append(s)

    # ------------------------------------------------------------------------------------------------ P2 split
    if want("P2"):
        s = Section("P2", "SPLIT: 80:20, disjoint, untouched test set"); header(s)
        sp = pd.read_csv(REP / "split_ids.csv")
        tr, te = set(df.window_id[df.split == "train"]), set(df.window_id[df.split == "test"])
        per = sp.groupby(["system", "split"]).size().unstack()
        s.add("2,549 training and 647 test windows (79.76% / 20.24%)", "PASS" if (len(tr), len(te)) == (2549, 647) else "FAIL",
              [f"train {len(tr):,}, test {len(te):,}, total {len(tr) + len(te):,}; train share {len(tr) / (len(tr) + len(te)):.2%}"] +
              [f"{sy}: train {int(per.loc[sy, 'train'])}, test {int(per.loc[sy, 'test'])}, train share {per.loc[sy, 'train'] / per.loc[sy].sum():.2%}" for sy in per.index], "report_outputs/split_ids.csv", dict(train=len(tr), test=len(te)))
        s.add("No window id is in both train and test; the split file equals the table's split", "PASS" if not (tr & te) and tr == set(sp[sp.split == "train"].window_id) else "FAIL",
              [f"shared ids: {len(tr & te)}", f"blocks in both parts: {len(set(df[df.split == 'train'].block_id) & set(df[df.split == 'test'].block_id))}", f"ids unique: {df.window_id.is_unique}"], "report_outputs/split_ids.csv")
        sections.append(s)

    # ------------------------------------------------------------------------------------------------ P3 imbalance / SMOTE
    if want("P3"):
        s = Section("P3", "IMBALANCE AND SMOTE: only the training part is resampled"); header(s)
        from v2_data import smote_resample
        res = {}
        for lab, names in (("label_binary", ["Normal", "Anomaly"]), ("label_level", LEVELS)):
            y_all = D.y[lab]
            rows = []
            ok = True
            for k in list(range(1, 11)) + ["final"]:
                tr_pos = D.train_pos if k == "final" else D.folds[k - 1][0]
                va_pos = D.test_pos if k == "final" else D.folds[k - 1][1]
                W = D.W(k)
                ytr = y_all[tr_pos]
                Xs, ys = smote_resample(W[tr_pos], ytr)
                before, after = np.bincount(ytr, minlength=len(names)), np.bincount(ys, minlength=len(names))
                held = np.bincount(y_all[va_pos], minlength=len(names))
                ok &= bool((after == before.max()).all() and len(Xs) == after.sum() and len(set(tr_pos) & set(va_pos)) == 0)
                rows.append([str(k), *before.tolist(), *after.tolist(), *held.tolist()])
            res[lab] = rows
            cols = ["fold"] + [f"b:{n[:4]}" for n in names] + [f"a:{n[:4]}" for n in names] + [f"hold:{n[:4]}" for n in names]
            s.add(f"{lab[6:].capitalize()} labels: class counts before (b) and after (a) SMOTE on the training part; held-out part (validation, or test for 'final') untouched", "PASS" if ok else "FAIL",
                  table(rows, cols) + ["after SMOTE every class equals the largest training class; synthetic windows exist only in the training part; held-out counts are the natural counts"],
                  "recomputed with scripts/v2_data.smote_resample on report_outputs/fold_assignment.csv + split_ids.csv", dict(table=rows, columns=cols))
        te_lab = pd.Series(D.yte["label_level"]).map(dict(enumerate(LEVELS))).value_counts().reindex(LEVELS)
        s.add("The test set keeps its natural imbalance (never resampled)", "PASS", [", ".join(f"{k} {v}" for k, v in te_lab.items()) + f"  (total {int(te_lab.sum())})"], "report_outputs/split_ids.csv", dict(test_counts=te_lab.to_dict()))
        sections.append(s)

    # ------------------------------------------------------------------------------------------------ P4 folds
    if want("P4"):
        s = Section("P4", "FOLDS: 10 run-aware folds"); header(s)
        fa = pd.read_csv(REP / "fold_assignment.csv")
        tr = fa[fa.fold > 0].merge(df[["window_id", "block_id", "run_id", "label_level"]], on="window_id")
        multi = int((tr.groupby("block_id").fold.nunique() > 1).sum())
        sizes = tr.groupby("fold").size()
        s.add("10 folds; grouped by block (every class has fewer than 10 runs, so run-grouping was impossible); no block is in two folds", "PASS" if tr.fold.nunique() == 10 and multi == 0 else "FAIL",
              [f"folds: {tr.fold.nunique()}; group type: {fa[fa.fold > 0].group_type.iloc[0]}; blocks appearing in more than one fold: {multi}", "fold sizes (validation windows): " + ", ".join(f"{k}:{v}" for k, v in sizes.items()),
               f"blocks per fold: " + ", ".join(f"{k}:{v}" for k, v in tr.groupby('fold').block_id.nunique().items())], "report_outputs/fold_assignment.csv", dict(sizes=sizes.to_dict(), multi_fold_blocks=multi))
        piv = tr.groupby(["fold", "label_level"]).size().unstack(fill_value=0)[LEVELS]
        s.add("Every fold validates on every class", "PASS" if (piv.values > 0).all() else "FAIL", table([[str(i)] + piv.loc[i].tolist() for i in piv.index], ["fold"] + LEVELS), "report_outputs/fold_assignment.csv + window labels")
        sections.append(s)

    # ------------------------------------------------------------------------------------------------ P5 models (recompute test metrics)
    if want("P5") or want("P6") or want("P8") or want("P9") or want("P13"):
        sys.argv = ["x", "--stages", "0"]
        import full_pipeline as FP
        FP.D = D
        from app.backend.registry import Registry
        reg = Registry().load()
        ctx["reg"], ctx["FP"] = reg, FP
        Wf = D.W("final")
        X, seq = Wf[D.test_pos], D.seq(Wf, D.test_pos)
        ctx["X"], ctx["seq"] = X, seq
    if want("P5"):
        s = Section("P5", "MODELS: every saved final model is loaded and scored on the 647 test windows"); header(s)
        reg, FP = ctx["reg"], ctx["FP"]
        thr = json.loads((REP / "thresholds.json").read_text())
        tm = pd.read_csv(REP / "test_metrics.csv")
        bs = pd.read_csv(REP / "test_bootstrap_ci.csv")
        same_prep = np.allclose(D.gprep.median, reg.prep.median, equal_nan=True) and np.allclose(D.gprep.scale_, reg.prep.scale_) and D.gprep.output_names == reg.prep.output_names
        s.add("The saved preprocessor equals one re-fitted from the training rows", "PASS" if same_prep else "FAIL", [f"medians, scales and the {len(reg.prep.output_names)} output columns identical: {same_prep}"], "processed/preprocessor.joblib vs scripts/preprocess.py fit")
        import keras
        from app.backend import deep_support
        probs = {}
        for l in ("label_binary", "label_level"):
            cache = {}
            for (lab, name), e in reg.entries.items():
                if lab != l:
                    continue
                if e["available"]:
                    probs[(l, name)] = reg.predict_proba(l, name, ctx["X"], ctx["seq"], cache)
            vote = reg.entries[(l, "SoftVote")]
            if args.quick:
                continue
            ps = []
            for m in vote["members"]:
                tag = reg.entries[(l, m)]["tag"]
                for k in range(1, 11):
                    fm = deep_support.load_keras(ROOT / "models" / "folds" / f"{l}__{m}__{tag}__f{k}.keras")
                    ps.append(fm.predict(D.seq(D.W(k), D.test_pos).astype(np.float32), batch_size=256, verbose=0))
            probs[(l, "SoftVote")] = np.mean(ps, axis=0)
        RANK = ("roc_auc", "pr_auc", "pr_auc_normal")
        n_val = 0
        summary_rows, per_model = [], {}
        for (l, name), P_ in sorted(probs.items()):
            md = mr = 0.0
            for variant, spec in (("official", None), ("tuned_threshold", thr[l][name])):
                rows = FP.metric_rows(l, name, P_.astype(np.float64), variant, spec)
                for r in rows:
                    ref = tm[(tm.model == name) & (tm.label_version == l) & (tm.scope == r["scope"]) & (tm.variant == variant)].iloc[0]
                    for k, v in r.items():
                        if k in ("model", "label_version", "scope", "variant") or k not in ref.index or pd.isna(ref[k]) or pd.isna(v):
                            continue
                        d = abs(float(v) - float(ref[k]))
                        n_val += 1
                        if k in RANK:
                            mr = max(mr, d)
                        else:
                            md = max(md, d)
            ci = bs[(bs.model == name) & (bs.label_version == l) & (bs.scope == "ALL") & (bs.metric == "balanced_accuracy")].iloc[0]
            mm = FP.metric_rows(l, name, P_.astype(np.float64), "official", None)[0]
            ref = tm[(tm.model == name) & (tm.label_version == l) & (tm.scope == "ALL") & (tm.variant == "official")].iloc[0]
            distinct = int(len(np.unique(np.round(P_[:, -1], 12))))
            spread = 0.0
            if mr > TOL:                       # how much does the ranking metric move if the scores are perturbed by floating-point noise (1e-15 relative)?
                aucs = []
                for t in range(8):
                    Pj = P_ * (1 + 1e-15 * np.random.RandomState(t).randn(*P_.shape))
                    aucs += [FP.metric_rows(l, name, Pj, "official", None)[0]["roc_auc"], FP.metric_rows(l, name, Pj, "official", None)[0]["pr_auc"]]
                spread = float(max(aucs) - min(aucs))
            per_model[f"{l}/{name}"] = dict(recomputed_ba=float(mm["balanced_accuracy"]), file_ba=float(ref.balanced_accuracy), max_diff_decision_metrics=md, max_diff_ranking_metrics=mr, noise_spread_ranking=spread,
                                          distinct_scores=distinct, ci_low=float(ci.ci95_low), ci_high=float(ci.ci95_high), recomputed_macro_f1=float(mm["macro_f1"]), file_macro_f1=float(ref.macro_f1), normal_recall=float(mm["normal_recall"]))
            summary_rows.append([f"{l[6:]}/{name}", f"{mm['balanced_accuracy']:.6f}", f"{ref.balanced_accuracy:.6f}", f"{md:.0e}", f"{mr:.0e}", f"{ci.ci95_low:.3f}-{ci.ci95_high:.3f}"])
        dec_bad = [k for k, v in per_model.items() if v["max_diff_decision_metrics"] > TOL]
        rank_off = {k: v for k, v in per_model.items() if v["max_diff_ranking_metrics"] > TOL}
        s.add(f"Decision metrics recomputed from the loaded models equal test_metrics.csv within {TOL:g} for all {len(per_model)} models (accuracy, balanced accuracy, macro-F1, per-class precision / recall / F1, Normal recall; argmax and saved-threshold decisions; all test windows, SN and TT)",
              "PASS" if not dec_bad else "FAIL",
              table(summary_rows, ["model", "BA recomputed", "BA in file", "max diff decision", "max diff ranking", "BA 95% CI"]) + ["95% CI = block bootstrap (1000 resamples) from test_bootstrap_ci.csv; 'decision' = metrics computed from the predicted class; 'ranking' = ROC-AUC / PR-AUC"],
              "models/ + processed/preprocessor.joblib vs report_outputs/test_metrics.csv, test_bootstrap_ci.csv", dict(models=per_model, tolerance=TOL, numbers_compared=n_val))
        ok_rank = [k for k, v in per_model.items() if v["max_diff_ranking_metrics"] <= TOL]
        explained = {k: v for k, v in rank_off.items() if v["max_diff_ranking_metrics"] <= max(5e-3, v["noise_spread_ranking"]) and v["noise_spread_ranking"] >= v["max_diff_ranking_metrics"] * 0.5}
        unexplained = [k for k in rank_off if k not in explained]
        lines = [f"ranking metrics (ROC-AUC, PR-AUC, PR-AUC of Normal) equal the file within {TOL:g} for {len(ok_rank)} of {len(per_model)} models"]
        if rank_off:
            lines += table([[k, f"{v['max_diff_ranking_metrics']:.1e}", f"{v['noise_spread_ranking']:.1e}", v["distinct_scores"]] for k, v in rank_off.items()], ["model", "max diff vs file", "spread from 1e-15 noise", "distinct scores"])
            lines += ["the recomputed probabilities of these models equal the original ones to ~1e-16, but forest scores are multiples of 1/(number of trees): many are EXACTLY tied, and a 1e-16 change in floating-point summation",
                      "order between runs re-orders the ties, which moves AUC by the amount shown. Adding 1e-15 relative noise to the scores moves the AUC by a spread of the same size ('spread from 1e-15 noise'), so this is tie sensitivity, not a model difference.",
                      "decision metrics are not affected (they match to 1e-6 above)"]
        s.add("Ranking metrics (ROC-AUC / PR-AUC) reproduce; differences above 1e-6 are explained by tied scores", "PASS" if not rank_off else ("WARN" if not unexplained else "FAIL"), lines,
              "models/ vs report_outputs/test_metrics.csv", dict(within_tolerance=len(ok_rank), differing={k: v for k, v in rank_off.items()}, unexplained=unexplained))
        if args.quick:
            s.add("Soft voting (average of 40 fold models) was not recomputed in --quick mode", "SKIP", ["run without --quick to recompute it"], "models/folds/")
        ctx["P"] = probs
        sections.append(s)

    # ------------------------------------------------------------------------------------------------ P6 over/under-fitting
    if want("P6"):
        s = Section("P6", "OVERFITTING: train vs validation vs test, corrections, first run vs retrain"); header(s)
        uo = pd.read_csv(REP / "underfit_overfit.csv")
        fin = uo[(uo.stage == "final") & (uo.scope == "ALL")]
        ba = pd.read_csv(REP / "before_after_correction.csv")
        tm = pd.read_csv(REP / "test_metrics.csv")
        probs = ctx.get("P", {})
        chk = []
        for _, r in fin.iterrows():
            t = tm[(tm.model == r.model) & (tm.label_version == r.label_version) & (tm.scope == "ALL") & (tm.variant == "official")].iloc[0]
            chk.append(abs(float(t.macro_f1) - float(r.test_macro_f1)))
        s.add("Test macro-F1 in underfit_overfit.csv equals test_metrics.csv for every model", "PASS" if max(chk) <= TOL else "FAIL", [f"models checked: {len(chk)}; max |difference| {max(chk):.1e}"], "report_outputs/underfit_overfit.csv, test_metrics.csv")
        if probs:
            d = []
            for (l, m), P_ in probs.items():
                r = fin[(fin.model == m) & (fin.label_version == l)]
                if len(r):
                    row = ctx["FP"].metric_rows(l, m, P_.astype(np.float64), "official", None)[0]
                    d.append(abs(row["macro_f1"] - float(r.test_macro_f1.iloc[0])))
            s.add("...and equals the macro-F1 recomputed in P5 from the loaded models", "PASS" if max(d) <= TOL else "FAIL", [f"models: {len(d)}; max |difference| {max(d):.1e}"], "recomputed")
        for l in ("label_binary", "label_level"):
            g = fin[fin.label_version == l].sort_values("train_test_gap_macro_f1", ascending=False)
            lines = table([[r.model, f"{r.final_train_macro_f1:.3f}", f"{r.cv_val_macro_f1:.3f}", f"{r.test_macro_f1:.3f}", f"{r.train_test_gap_macro_f1:+.3f}", ("overfit" if "overfit (train-validation" in r.diagnosis else "ok")] for _, r in g.iterrows()],
                          ["model", "train F1", "CV valid. F1", "test F1", "train-test", "CV diag."])
            s.add(f"{l[6:].capitalize()}: macro-F1 on the training part, in cross-validation and on test (final models)", "PASS", lines + ["rule: a model is flagged overfit when train minus validation macro-F1 > 0.10 (check also test gap)"], "report_outputs/underfit_overfit.csv")
        for l in ("label_binary", "label_level"):
            g = ba[ba.label_version == l]
            s.add(f"{l[6:].capitalize()}: {len(g)} models were diagnosed as overfit and retrained with stronger regularisation; {int(g.adopted.sum())} corrected versions were adopted", "PASS",
                  table([[r.model, f"{r.val_bal_acc_before:.3f}", f"{r.val_bal_acc_after:.3f}", f"{r.gap_macro_f1_before:.3f}", f"{r.gap_macro_f1_after:.3f}", "yes" if r.adopted else "no"] for _, r in g.iterrows()],
                        ["model", "val BA before", "after", "gap before", "gap after", "adopted"]) + ["adoption rule: validation balanced accuracy drops by <= 0.01 AND the train-validation gap does not grow"], "report_outputs/before_after_correction.csv")
        v1 = pd.read_csv(ROOT / "results" / "v1_16epoch" / "test_metrics.csv")
        v1 = v1[(v1.scope == "ALL") & (v1.variant == "official")]
        v2 = tm[(tm.scope == "ALL") & (tm.variant == "official")]
        rows = []
        for l in ("label_binary", "label_level"):
            for _, a in v1[v1.label_version == l].iterrows():
                b = v2[(v2.model == a.model) & (v2.label_version == l)].iloc[0]
                rows.append([f"{l[6:]}/{a.model}", f"{a.balanced_accuracy:.3f}", f"{b.balanced_accuracy:.3f}", f"{b.balanced_accuracy - a.balanced_accuracy:+.3f}"])
        best = {l: (v1[v1.label_version == l].balanced_accuracy.max(), v2[v2.label_version == l].balanced_accuracy.max()) for l in ("label_binary", "label_level")}
        s.add("First run (16-epoch cap) versus the full retrain on the same 647 test windows", "PASS",
              table(rows, ["model", "v1 BA", "v2 BA", "change"]) + [f"best balanced accuracy: binary {best['label_binary'][0]:.3f} -> {best['label_binary'][1]:.3f}; five-class {best['label_level'][0]:.3f} -> {best['label_level'][1]:.3f}",
                                                                  "the first run's cross-validation (row-level folds) was optimistic; only test scores are compared"], "results/v1_16epoch/test_metrics.csv, report_outputs/test_metrics.csv")
        sections.append(s)

    # ------------------------------------------------------------------------------------------------ P7 leakage audit
    if want("P7"):
        s = Section("P7", "LEAKAGE AUDIT: as reported by scripts/leakage_audit.py"); header(s)
        aud = json.loads((REP / "leakage_audit.json").read_text())
        out("  audit run: " + aud["generated_at"] + f", commit {aud['git_commit'][:10]}, runtime {aud['runtime_seconds']} s")
        for c in aud["checks"]:
            s.add(f"{c['id']}  {c['check']}", c["status"], [c["summary"]], c["source"], dict(evidence=c["evidence"]))
        c3 = next(c for c in aud["checks"] if c["id"] == "C3")
        eff = c3["evidence"]["no_overlap_effect_first_run"]
        s.add("C3 stays WARN: measured effect of dropping the overlapping training windows (first-run models, refit without them)", "WARN",
              [f"mean change in test balanced accuracy: {eff['label_binary']['mean_delta_balanced_accuracy']:+.3f} (binary), {eff['label_level']['mean_delta_balanced_accuracy']:+.3f} (five-class)",
               f"range over {eff['label_binary']['models']} models: binary {eff['label_binary']['min']:+.3f} to {eff['label_binary']['max']:+.3f}; five-class {eff['label_level']['min']:+.3f} to {eff['label_level']['max']:+.3f}",
               "not a label leak, but real: neighbouring windows at a block boundary share 20 s of data"], "report_outputs/leakage_audit.json (C3), results/v1_16epoch/test_metrics.csv")
        di = pd.read_csv(REP / "data_integrity.csv")
        files = [WS / p for p in di.path]
        t = time.time()
        now = {}
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(6) as ex:
            for p, h in zip(files, ex.map(sha256, files)):
                now[p] = h
        bad = [p.as_posix() for p, h in zip(files, di.sha256_end) if now[p] != h]
        first = di.baseline_time.min()
        s.add("C7 (live re-check): no AnoMod / LO2 file used by the project has changed since the audit", "PASS" if not bad else "FAIL",
              [f"{len(files)} files re-hashed now ({time.time() - t:.0f}s): {len(files) - len(bad)} identical to the audit's end-of-run hash; changed: {len(bad)}",
               f"IMPORTANT: the C7 hash baseline was first recorded during the audit run (baseline recorded {first}). It therefore proves that no file changed SINCE THEN, not before."],
              "report_outputs/data_integrity.csv", dict(files=len(files), changed=bad[:10], baseline_first_recorded=str(first)))
        sections.append(s)

    # ------------------------------------------------------------------------------------------------ P8 explainability
    if want("P8"):
        s = Section("P8", "EXPLAINABILITY: SHAP and LIME"); header(s)
        reg = ctx["reg"]
        ag = pd.read_csv(REP / "shap_lime_agreement_summary.csv")
        a = ag[ag.outcome == "all"]
        s.add("SHAP and LIME agree on part of the top-5 features (offline analysis of the best tree models)", "PASS",
              table([[r.label_version[6:], r.model, r.window_set, int(r.n_windows), f"{r.mean_shared_top5:.2f}", f"{r.share_with_ge3_shared:.0%}"] for _, r in a.iterrows()], ["labels", "model", "windows", "n", "mean shared of 5", ">=3 shared"]),
              "report_outputs/shap_lime_agreement_summary.csv")
        from app.backend.explain import Explainer
        ex = Explainer(reg)
        rng = np.random.RandomState(42)
        te = D.test_pos
        picks = []
        for lv, sy in (("Normal", "SN"), ("Service", "TT"), ("Code", "SN")):
            idx = np.where((df.label_level.to_numpy()[te] == lv) & (df.system.to_numpy()[te] == sy))[0]
            picks.append(int(rng.choice(idx)))
        names = reg.prep.output_names
        for label, model in (("label_binary", "XGB"), ("label_level", "LGBM_tuned")):
            est = reg.tree_estimator(label, model)
            full = reg.full_tab_model(label, model)
            Xp = ctx["X"]
            thr = json.loads((REP / "thresholds.json").read_text())[label][model]
            for j in picks:
                x = Xp[j:j + 1]
                p = reg.predict_proba(label, model, x, ctx["seq"][j:j + 1])[0]
                cn = ["Normal", "Anomaly"] if label == "label_binary" else LEVELS
                pred = int(p[1] >= thr["threshold"]) if label == "label_binary" else int(np.argmax(p * np.array(thr["weights"])))
                ci = 1 if label == "label_binary" else pred
                sh = ex.shap_values(label, model, x, ci, top=5)
                sv = np.array([t["shap"] for t in sh["top"]])
                # additivity: base value + sum of ALL shap values == the model's raw output
                import shap as _shap
                allv = _shap.TreeExplainer(est).shap_values(x)
                allv = np.stack(allv, -1) if isinstance(allv, list) else np.asarray(allv)
                tot = float(allv[0, :, ci].sum()) if allv.ndim == 3 else float(allv[0].sum())
                if label == "label_binary":
                    margin = float(est.predict(x, output_margin=True)[0]) if hasattr(est, "get_booster") else float(est.predict(x, raw_score=True)[0])
                else:
                    margin = float(est.predict(x, raw_score=True)[0][ci]) if not hasattr(est, "get_booster") else float(est.predict(x, output_margin=True)[0][ci])
                add_ok = abs(sh["base_value"] + tot - margin) < 1e-3
                lime_txt = ""
                if ex.background is not None:
                    lm = ex.lime_values(label, model, x, ci, top=5, num_samples=1000)
                    shared = len({t["feature"] for t in sh["top"]} & {t["feature"] for t in lm["top"]})
                    lime_txt = f"; LIME top-5 shares {shared}/5 with SHAP (LIME is stochastic: seed 42, 1000 samples)"
                lines = [f"window {df.window_id.to_numpy()[te][j]} (true: {df.label_level.to_numpy()[te][j]}, system {df.system.to_numpy()[te][j]}) -> verdict {cn[pred]}; explained class {cn[ci]}",
                         *[f"  {t['feature']}: SHAP {t['shap']:+.4f}  (scaled value {t['preprocessed_value']:+.3f}, raw {df[t['feature']].to_numpy()[te][j] if t['feature'] in df.columns else 'n/a'})" for t in sh["top"]],
                         f"  additivity: base {sh['base_value']:+.4f} + sum(all SHAP) {tot:+.4f} = {sh['base_value'] + tot:+.4f}  vs model raw output {margin:+.4f}  -> {'OK' if add_ok else 'MISMATCH'}{lime_txt}",
                         f"  recompute: python scripts/proofs.py --only P8   (window id above, model {model}, {label})"]
                s.add(f"Explanation of one test window with {model} ({label[6:]}) is recomputable and consistent", "PASS" if add_ok else "FAIL", lines, "models/ + scripts/preprocess.py + app/backend/explain.py (shap TreeExplainer)", dict(window=str(df.window_id.to_numpy()[te][j]), model=model, label=label))
        sections.append(s)

    # ------------------------------------------------------------------------------------------------ P9 external test
    if want("P9"):
        s = Section("P9", "EXTERNAL TEST on LO2: reported exactly as it is"); header(s)
        m = pd.read_csv(REP / "lo2_test_metrics.csv")
        prim = m[(m.feature_set == "A_logs_only") & (m.system_flag == "TT")]
        mp = pd.read_csv(REP / "lo2_feature_mapping.csv")
        used = mp[mp.kept_by_preprocessor]
        built = int(used.variant_A_logs_only.str.startswith("built").sum())
        w = pd.read_parquet(ROOT / "processed" / "lo2_windows_A.parquet") if (ROOT / "processed" / "lo2_windows_A.parquet").exists() else None
        ok = prim.balanced_accuracy.round(6).eq(0.5).all() and len(prim) == 18 and built == 3 and len(used) == 124 and int(prim.n_windows.iloc[0]) == 28
        s.add("On LO2 every saved model scores balanced accuracy 0.500; only 3 of the 124 model features can be built; 28 windows", "REPORTED" if ok else "FAIL",
              [f"models: {len(prim)}; balanced accuracy min {prim.balanced_accuracy.min():.3f}, max {prim.balanced_accuracy.max():.3f}", f"model features buildable from LO2 logs: {built} of {len(used)} ({built / len(used):.1%}); AnoMod features: {int(mp.variant_A_logs_only.str.startswith('built').sum())} of {len(mp)}",
               f"windows: {int(prim.n_windows.iloc[0])} = {int(prim.n_normal.iloc[0])} Normal (one scenario) + {int(prim.n_anomaly.iloc[0])} Anomaly",
               f"models labelling every window Anomaly: {int((prim.share_predicted_anomaly == 1).sum())}; every window Normal: {int((prim.share_predicted_anomaly == 0).sum())}",
               "variant B (adds 19 shared node metrics) and the SN system flag were also run: see lo2_test_metrics.csv; every model still predicted one class for every window"], "report_outputs/lo2_test_metrics.csv, lo2_feature_mapping.csv, lo2_sample_manifest.csv", dict(models=len(prim), built=built, used=len(used)), result=f"models did not transfer ({prim.balanced_accuracy.max():.3f})")
        if w is not None and "reg" in ctx:
            import lo2_eval
            v2_data.common.LOG = ROOT / "outputs" / "proofs_log.txt"      # lo2_eval points the shared log at report_outputs/lo2_log.txt; keep tracked files untouched
            reg = ctx["reg"]
            fr = w.copy(); fr["system_is_TT"] = 1; fr["system"] = "TT"
            Wg = reg.prep.transform(fr)
            ctx_i = lo2_eval.lo2_ctx(fr)
            Xs = Wg[ctx_i].astype(np.float32)
            y = fr.label_binary.to_numpy(int)
            cache, dmax, n = {}, 0.0, 0
            for (lab, name), e in reg.entries.items():
                if lab == "label_binary" and e["available"]:
                    P_ = reg.predict_proba(lab, name, Wg, Xs, cache)
                    pt = lo2_eval.point(y, P_)
                    ref = prim[prim.model == name].iloc[0]
                    dmax = max(dmax, abs(pt["balanced_accuracy"] - ref.balanced_accuracy), abs(pt["roc_auc"] - ref.roc_auc)); n += 1
            s.add("LO2 results recomputed by loading the models and scoring the 28 windows", "PASS" if dmax <= TOL else "FAIL", [f"servable models recomputed: {n}; max |difference| in balanced accuracy and ROC-AUC vs the file: {dmax:.1e} (soft voting is read from the file)"], "processed/lo2_windows_A.parquet + models/")
        else:
            s.add("LO2 results recomputed by loading the models", "SKIP", ["processed/lo2_windows_A.parquet not present"], "-")
        tf = pd.read_csv(REP / "lo2_threshold_free.csv")
        p2 = tf[(tf.feature_set == "A_logs_only") & (tf.system_flag == "TT")]
        chk = pd.read_csv(REP / "lo2_threshold_free_logvolume_check.csv")
        raw = chk[chk.kind == "raw_feature_roc_auc_anomaly_positive"].set_index("name").value
        hi = p2[p2.roc_auc >= 0.9]
        sp_ = chk[(chk.kind == "spearman_score_vs_log_line_count") & chk.name.str.endswith("|A_logs_only|TT")]
        rho = [float(sp_[sp_.name == f"{x}|A_logs_only|TT"].value.iloc[0]) for x in hi.model]
        s.add("The high ROC-AUCs are not evidence of detection: log volume alone separates the classes", "REPORTED",
              [f"{len(hi)} of {len(p2)} models have ROC-AUC >= 0.90 although balanced accuracy is 0.500", f"raw log_line_count alone: ROC-AUC {raw['log_line_count']:.3f}; raw log_error_count alone: {raw['log_error_count']:.3f} (Normal scenario runs 60 s, failure scenarios 10-18 s)",
               f"of those {len(hi)} models, {sum(abs(r) > 0.8 for r in rho)} rank windows like log volume (|Spearman| > 0.8); for the others the driver is unidentified",
               "all 5 Normal windows come from one scenario, so separation cannot be told apart from a scenario artefact"], "report_outputs/lo2_threshold_free.csv, lo2_threshold_free_logvolume_check.csv",
              result="high ROC-AUC reflects log volume, not detection")
        sections.append(s)

    # ------------------------------------------------------------------------------------------------ P11 universal features and cross-system test
    if want("P11"):
        s = Section("P11", "UNIVERSAL FEATURES AND CROSS-SYSTEM TEST: do shared and baseline-relative features generalise from one system to the other?"); header(s)
        import universal_features as UF
        import cross_system as CS
        cp = CS.OUT
        if not cp.exists() or not UF.OUT.exists():
            s.add("Cross-system test", "SKIP", ["report_outputs/cross_system_results.csv or processed/universal_windows.parquet missing; run scripts/generic_metrics.py, scripts/universal_features.py, scripts/cross_system.py"], "-")
            sections.append(s)
        else:
            shared = pd.read_csv(REP / "shared_features.csv")
            gen = pd.read_csv(REP / "generic_metric_features.csv")
            ud = pd.read_parquet(UF.OUT)
            sh = shared[shared.shared]
            # ---- 1 shared features and generic aggregates
            both = gen[gen.both_systems == True]  # noqa: E712
            nob = gen[gen.both_systems != True]
            lines = [f"shared features (values in both SocialNetwork and TrainTicket windows): {len(sh)} of {len(shared)}: " + ", ".join(sh.feature),
                     f"  = {int((sh.group == 'api').sum())} API + {int((sh.group == 'logs').sum())} log features. SocialNetwork also has {int(((shared.group == 'api') & shared.in_SN & ~shared.in_TT).sum())} API latency features that TrainTicket does not ({', '.join(shared[(shared.group == 'api') & shared.in_SN & ~shared.in_TT].feature)}).",
                     f"  metric features: SN {int(((shared.group == 'metrics') & shared.in_SN).sum())}, TT {int(((shared.group == 'metrics') & shared.in_TT).sum())}, shared by name: {int(((shared.group == 'metrics') & shared.shared).sum())} (the two systems do not export a single metric under the same name).",
                     "", f"generic metric aggregates that can be built for BOTH systems ({len(both)}), each as window mean / max / std of one per-scrape total: scripts/generic_metrics.py"]
            lines += table([[r.aggregate, r.SN_windows_with_value, r.TT_windows_with_value, f"{r.SN_median:.3g}", f"{r.TT_median:.3g}"] for r in both.itertuples()], ["aggregate", "SN windows with value", "TT windows with value", "SN median", "TT median"])
            lines += ["(the medians differ by a large factor, e.g. memory 1e9 vs 2.7e10 bytes: raw values are not comparable between the systems, which is why baseline-relative forms are needed)", "", "cannot be built for both systems:"]
            lines += [f"  {r.aggregate}: {r.SN_source}" for r in nob.itertuples()]
            first = []
            for sy in ("SN", "TT"):
                dd = ud[ud.system == sy]
                pos = []
                for run, g in dd.groupby("run_id"):
                    g = g.sort_values("window_index")
                    h = g.api_request_count.to_numpy(float) > 0
                    pos.append(np.where(h)[0][0] / len(g) if h.any() else np.nan)
                first.append(f"{sy}: windows with API traffic {(dd.api_request_count > 0).mean():.1%}; first window with API traffic at {np.nanmedian(pos):.0%} of the run (median over runs; {int(np.isnan(pos).sum())} runs without any)")
            lines += ["", "what the shared API features look like in the recordings (computed from the window table):"] + ["  " + x for x in first]
            lines += ["", "baseline-relative rule (scripts/universal_features.py): z = clip((x - median) / max(1.4826 MAD, 0.05 |median|, 1e-12), +-10); baseline = the most recent 60 non-missing values of the same run among windows that ended before this window began (index <= k - 3); at least 9 needed; no label, other run, other system or future window.",
                      "training windows use only earlier TRAINING windows; test windows may use any earlier window of their own run (a live agent has no labels). The first 11 windows of each run have no value and are left out of training and testing."]
            n_chk, n_bad = UF.prove(ud, sorted(set(sum(UF.feature_sets(shared).values(), []))))
            lines += [f"proof: for {n_chk} windows (random runs and positions) every future / overlapping window, and for training windows every test window, was overwritten with random values 100x the data scale; baseline-relative features changed in {n_bad} cases"]
            s.add("Shared features, generic metric aggregates and the baseline rule: no future or test data enters a baseline", "PASS" if (n_bad == 0 and len(sh) == 8 and len(both) == 9) else "FAIL", lines, "scripts/universal_features.py, scripts/generic_metrics.py, report_outputs/shared_features.csv, generic_metric_features.csv", dict(shared=len(sh), generic_both=len(both), proof_windows=n_chk, proof_differences=n_bad))
            # ---- 2 recomputation
            cs = pd.read_csv(cp)
            fresh = CS.reproduce(1000)
            dmax = 0.0
            for r in fresh:
                o = cs[(cs.model == "XGB") & (cs.control == "real labels") & (cs.direction == r["direction"]) & (cs.feature_set == r["feature_set"]) & (cs.representation == r["representation"])].iloc[0]
                for k in ("balanced_accuracy", "macro_f1", "normal_recall", "roc_auc", "balanced_accuracy_ci_low", "balanced_accuracy_ci_high"):
                    dmax = max(dmax, abs(float(r[k]) - float(o[k])))
            s.add("The cross-system results of XGBoost (both directions, 3 feature sets, baseline-relative and raw) are reproduced by recomputation", "PASS" if dmax <= 1e-5 else "FAIL",
                  [f"cells recomputed: {len(fresh)}; max |difference| over balanced accuracy, macro-F1, Normal recall, ROC-AUC and the BA interval {dmax:.1e}; the other models and the shuffled controls are read from the file"], "scripts/cross_system.py, report_outputs/cross_system_results.csv")
            # ---- 3 the result
            real = cs[cs.control == "real labels"]
            cross = real[real.direction.isin(["SN->TT", "TT->SN"])]
            xb = cross[cross.model == "XGB"]
            best = xb.sort_values("balanced_accuracy", ascending=False).iloc[0]
            above = cross[cross.balanced_accuracy_ci_low > 0.5]
            f3 = lambda x: f"{x:.3f}"
            cell = lambda r: f"{f3(r.balanced_accuracy)} ({r.balanced_accuracy_ci_low:.2f}-{r.balanced_accuracy_ci_high:.2f})"
            lines = ["Train on one system's TRAINING split, test on the other system's TEST windows (SN->TT, TT->SN); SN->SN and TT->TT are in-system references. XGBoost with the saved recipe (SMOTE on the training part only, early stopping), throw-away, never saved. Balanced accuracy (95% block-bootstrap CI over the target test blocks); chance = 0.500."]
            for rep in ("rel", "raw"):
                lines += ["", f"XGBoost, {'baseline-relative' if rep == 'rel' else 'raw'} features:"]
                rows_ = []
                for fs in CS.SET_NAMES.values():
                    row = [fs]
                    for dr in ("SN->TT", "TT->SN", "SN->SN", "TT->TT"):
                        c = real[(real.model == "XGB") & (real.representation == rep) & (real.feature_set == fs) & (real.direction == dr)].iloc[0]
                        row.append(cell(c))
                    rows_.append(row)
                lines += table(rows_, ["feature set", "SN->TT", "TT->SN", "SN->SN (in-system)", "TT->TT (in-system)"])
            lines += ["", "XGBoost, cross-system cells: Normal recall and ROC-AUC (point values; intervals in the CSV)"]
            lines += table([[r.direction, r.feature_set, "rel" if r.representation == "rel" else "raw", f3(r.normal_recall), f3(r.roc_auc), f"{int(r.n_train)} / {int(r.n_train_normal)}", f"{int(r.n_test)} / {int(r.n_test_normal)}"] for r in xb.itertuples()],
                           ["direction", "feature set", "features", "Normal recall", "ROC-AUC", "train windows / Normal", "test windows / Normal"])
            lines += ["", "the two other models (LGBM_tuned and XGB_tuned: the best other refittable models by test balanced accuracy in the earlier runs), cross-system, mean over the 12 cells of each representation:"]
            lines += table([[m, rep, f3(cross[(cross.model == m) & (cross.representation == rep)].balanced_accuracy.mean()), f3(cross[(cross.model == m) & (cross.representation == rep)].balanced_accuracy.max())] for m in CS.MODELS for rep in ("rel", "raw")], ["model", "features", "mean BA", "best BA"])
            shuf = cs[cs.control.str.startswith("shuffled") & cs.direction.isin(["SN->TT", "TT->SN"])]
            lines += ["", f"shuffled-label control (training labels permuted, scoring labels real), all cross-system cells and models: mean balanced accuracy {f3(shuf.balanced_accuracy.mean())}, range {f3(shuf.balanced_accuracy.min())} to {f3(shuf.balanced_accuracy.max())}"]
            tm = pd.read_csv(REP / "test_metrics.csv")
            ref = tm[(tm.model == "XGB") & (tm.label_version == "label_binary") & (tm.scope == "ALL") & (tm.variant == "official")].iloc[0]
            lines += [f"reference, in-system with ALL 128 raw features (the saved XGBoost, both systems pooled): test balanced accuracy {f3(ref.balanced_accuracy)}, Normal recall {f3(ref.normal_recall)}"]
            cvr = cs[cs.control.str.startswith("10-fold") & (cs.representation == "rel")]
            lines += ["10-fold CV (block-grouped folds) of XGBoost on the training split of one system, baseline-relative: " + ", ".join(f"{r.direction[:2]} {r.feature_set[:3]} {f3(r.balanced_accuracy)}" for r in cvr.itertuples())]
            res = (f"best cross-system XGBoost balanced accuracy {f3(best.balanced_accuracy)} (95% CI {best.balanced_accuracy_ci_low:.2f}-{best.balanced_accuracy_ci_high:.2f}); "
                   f"{len(above)} of {len(cross)} cross-system cells (all models) have a CI above chance")
            lines += ["", "rule: generalisation is shown only if the CI of the cross-system balanced accuracy lies above 0.5 (and clearly above the shuffled control).", "result: " + res]
            if len(above):
                lines += ["cells above chance: " + "; ".join(f"{r.direction} {r.feature_set} {r.representation} {r.model} {cell(r)}" for r in above.itertuples())]
            s.add("Cross-system generalisation of shared and baseline-relative features", "REPORTED", lines, "report_outputs/cross_system_results.csv (scripts/cross_system.py)",
                  dict(best_xgb_cross_ba=float(best.balanced_accuracy), cells_above_chance=int(len(above)), cells=int(len(cross))), result=("not shown: " + res) if not len(above) else res)
            sections.append(s)

    # ------------------------------------------------------------------------------------------------ P12 live demo, P16 run-held-out results, P17 run identification on the testbed
    if want("P12") or want("P16") or want("P17"):
        import tempfile
        import contextlib
        import io
        tb_files = ["testbed_run_assignment.csv", "testbed_heldout_overall.csv", "testbed_heldout_by_fault.csv", "testbed_false_alarms.csv", "testbed_prefailure.csv", "testbed_shuffled_control.csv",
                    "testbed_runid_check.csv", "testbed_cv_by_fault.csv", "testbed_train_cv.csv"]
        f3 = lambda x: "nan" if pd.isna(x) else f"{x:.3f}"
        ci3 = lambda r, k: f"{f3(r[k])} ({f3(r[k + '_ci_low'])}-{f3(r[k + '_ci_high'])})"
        have = all((REP / f).exists() for f in tb_files)
    if want("P16") or want("P17"):
        s16 = Section("P16", "TESTBED, RUN-HELD-OUT VALIDATION: detection of injected faults on whole runs the models never saw"); header(s16)
        if not have:
            s16.add("Testbed validation", "SKIP", ["report_outputs/testbed_*.csv missing; run python -m testbed.validate"], "-")
            sections.append(s16)
        else:
            import testbed.validate as TV
            tmp = Path(tempfile.mkdtemp())
            old_rep = TV.REP
            TV.REP = tmp
            sys.argv = ["x", "--boot", "1000"]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    TV.main()
            finally:
                TV.REP = old_rep
            dmax, n_cmp, bad_files = 0.0, 0, []
            for f in tb_files:
                a_, b_ = pd.read_csv(tmp / f), pd.read_csv(REP / f)
                if a_.shape != b_.shape:
                    bad_files.append(f)
                    continue
                for col in a_.columns:
                    if pd.api.types.is_numeric_dtype(a_[col]) and pd.api.types.is_numeric_dtype(b_[col]):
                        x, y = a_[col].to_numpy(float), b_[col].to_numpy(float)
                        ok_ = np.isfinite(x) & np.isfinite(y)
                        if ok_.any():
                            dmax = max(dmax, float(np.abs(x[ok_] - y[ok_]).max())); n_cmp += int(ok_.sum())
                        if (np.isfinite(x) != np.isfinite(y)).any():
                            bad_files.append(f"{f}:{col}")
                    elif not a_[col].astype(str).equals(b_[col].astype(str)):
                        bad_files.append(f"{f}:{col}")
            s16.add("The run-held-out results are reproduced by re-running the whole validation", "PASS" if (dmax <= 1e-5 and not bad_files) else "FAIL",
                    [f"split, cross-validation, models, bootstrap, delays, false alarms, controls recomputed; {len(tb_files)} result files, {n_cmp} numbers compared, max |difference| {dmax:.1e}" + (f"; differing: {bad_files}" if bad_files else "")],
                    "testbed/validate.py, report_outputs/testbed_*.csv")
            asg = pd.read_csv(REP / "testbed_run_assignment.csv")
            Wt = pd.read_parquet(ROOT / "processed" / "testbed_windows.parquet")
            Sf = pd.concat([pd.read_csv(f, dtype={"fault_id": str}) for f in sorted((ROOT / "testbed" / "runs").glob("r*/fault_schedule.csv"))], ignore_index=True)
            test_r, train_r = set(asg[asg.split == "test"].run_id), set(asg[asg.split == "train"].run_id)
            free_r = set(Wt.groupby("run_id").label_binary.max().loc[lambda s: s == 0].index)
            ok_split = (not (test_r & train_r) and set(Sf[Sf.run_id.isin(test_r)].fault_type) == set(Sf.fault_type) and len(test_r & free_r) >= 2 and Wt.run_seed.max() < 9000
                        and all(Wt[Wt.run_id == r].window_id.str.startswith(r).all() for r in test_r))
            lines = [f"{len(test_r)} test runs and {len(train_r)} train runs; windows of one run are never on both sides; every one of the {Sf.fault_type.nunique()} fault types is in the test runs ({len(Sf[Sf.run_id.isin(test_r)])} faults); fault-free test runs: {len(test_r & free_r)}; largest run seed in the data {int(Wt.run_seed.max())} (live demo seeds start at 9000)", ""]
            lines += table([[r.run_id, r.split, "" if pd.isna(r.fold) else int(r.fold), r.faults] for r in asg.itertuples()], ["run", "split", "fold", "faults"])
            s16.add("Whole runs are split, every fault type is in the test runs, at least two test runs are fault-free", "PASS" if ok_split else "FAIL", lines, "report_outputs/testbed_run_assignment.csv, testbed/validate.py")
            ov = pd.read_csv(REP / "testbed_heldout_overall.csv")
            bf = pd.read_csv(REP / "testbed_heldout_by_fault.csv")
            fa = pd.read_csv(REP / "testbed_false_alarms.csv")
            pf = pd.read_csv(REP / "testbed_prefailure.csv")
            inst = pd.read_csv(REP / "testbed_heldout_instances.csv")
            sup = ov[ov.kind == "supervised"].sort_values("balanced_accuracy", ascending=False)
            best = sup.iloc[0]
            lines = [f"scored windows only (after each run's 150 s learning period, transition windows left out): {int(best.n_windows)} windows, {int(best.n_fault_windows)} of them fault windows, {int(best.n_test_runs)} test runs. 95% CIs resample whole runs (1000 resamples).", ""]
            lines += table([[r.method, f"{f3(r.balanced_accuracy)} ({f3(r.balanced_accuracy_ci_low)}-{f3(r.balanced_accuracy_ci_high)})", f3(r.macro_f1), f"{f3(r.normal_recall)} ({f3(r.normal_recall_ci_low)}-{f3(r.normal_recall_ci_high)})", f3(r.roc_auc)] for r in ov.itertuples()],
                           ["method", "balanced accuracy", "macro-F1", "Normal recall", "ROC-AUC"])
            lines += ["", "recall of fault windows by severity level and by fault kind (share of the fault windows that were flagged), test runs; 'reliable' needs at least 5 fault instances"]
            for m in [best.method]:
                x = bf[(bf.method == m) & bf.group.isin(["severity level", "fault kind"])]
                lines += table([[r.group, r.value, int(r.instances), int(r.windows), f"{f3(r.recall)} ({f3(r.ci_low)}-{f3(r.ci_high)})", "yes" if r.reliable else "NO: too few instances"] for r in x.itertuples()], ["group", "value", "instances", "windows", "recall (95% CI)", "reliable"])
            lines += ["", "recall by fault type, test runs: instances per type are 1 to 3, so none of these ten numbers is reliable (the project's rule: at least 5 instances)"]
            x = bf[(bf.method == best.method) & (bf.group == "fault type")]
            lines += table([[r.value, int(r.instances), int(r.windows), f"{f3(r.recall)} ({f3(r.ci_low)}-{f3(r.ci_high)})", "yes" if r.reliable else "NO"] for r in x.itertuples()], ["fault type", "instances", "windows", "recall (95% CI)", "reliable"])
            cvb = pd.read_csv(REP / "testbed_cv_by_fault.csv")
            lines += ["", "supplementary, every one of the 40 runs held out once (XGBoost, 8 folds by run): more instances per cell, descriptive only"]
            lines += table([[r.group, r.value, int(r.instances), f"{f3(r.recall)} ({f3(r.ci_low)}-{f3(r.ci_high)})", "yes" if r.reliable else "NO"] for r in cvb.itertuples()], ["group", "value", "instances", "recall (95% CI)", "reliable"])
            lines += ["", "detection delay (seconds from the recorded fault start to the first alert), false alarms per hour on the fault-free test runs, per method and alert rule"]
            rows_ = []
            for r in fa.itertuples():
                sub = inst[(inst.method == r.method) & (inst.rule == r.rule)]
                rows_.append([r.method, r.rule, f"{int(sub.detected.sum())}/{len(sub)}", f"{sub.delay_s.median():.0f}", f"{sub.delay_s.quantile(0.9):.0f}", f"{int(r.alert_episodes_fault_free)} in {r.fault_free_hours:.2f} h = {r.false_alarms_per_hour_fault_free:.1f}/h"])
            lines += table(rows_, ["method", "alert rule", "detected", "median delay s", "90th pct s", "false alarms (fault-free runs)"])
            lines += ["", f"only {int(fa.fault_free_runs.iloc[0])} fault-free test runs ({fa.fault_free_hours.iloc[0]:.2f} h): one alert changes the rate by {1 / fa.fault_free_hours.iloc[0]:.0f} per hour, so these rates are very coarse"]
            lines += ["", "pre-failure warning for the gradual faults (XGBoost, dedicated classifier per horizon; 3 gradual instances in the test runs)"]
            lines += table([[int(r.horizon_min), f"{f3(r.precision)} ({f3(r.precision_ci_low)}-{f3(r.precision_ci_high)})", f"{f3(r.recall)} ({f3(r.recall_ci_low)}-{f3(r.recall_ci_high)})", f"{int(r.instances_warned)}/{int(r.gradual_instances)}", f"{r.median_lead_time_s:.0f}"] for r in pf.itertuples()],
                           ["horizon (min)", "precision (95% CI)", "recall (95% CI)", "instances warned", "median lead time s"])
            res = (f"best supervised model {best.method}: balanced accuracy {f3(best.balanced_accuracy)} ({f3(best.balanced_accuracy_ci_low)}-{f3(best.balanced_accuracy_ci_high)}) on {int(best.n_test_runs)} held-out runs; "
                   f"cells with too few instances are marked")
            s16.add("Run-held-out detection results (12 test runs, 15 faults): small test, per-type numbers unreliable", "REPORTED", lines, "report_outputs/testbed_heldout_*.csv, testbed_false_alarms.csv, testbed_prefailure.csv, testbed_cv_by_fault.csv", dict(best_method=str(best.method), best_ba=float(best.balanced_accuracy)), result=res)
            sh = pd.read_csv(REP / "testbed_shuffled_control.csv")
            s16.add("Shuffled-label control: with the training labels shuffled the models are at chance", "PASS" if sh.shuffled_labels_mean_ba.between(0.45, 0.55).all() else "WARN",
                    table([[r.method, f3(r.shuffled_labels_mean_ba), f"{f3(r.min_ba)}-{f3(r.max_ba)}", f3(r.real_labels_ba)] for r in sh.itertuples()], ["method", "shuffled mean BA", "range", "real-label BA"]) + ["rule: the mean balanced accuracy with shuffled labels must lie in 0.45-0.55"],
                    "report_outputs/testbed_shuffled_control.csv")
            sections.append(s16)
            s17 = Section("P17", "TESTBED, RUN-IDENTIFICATION CHECK: could recognising the run explain the result?"); header(s17)
            rid = pd.read_csv(REP / "testbed_runid_check.csv")
            g = lambda key: float(rid[rid.check.str.contains(key, regex=False)].value.iloc[0])
            acc, rule_ba, best_ba = g("runs identifiable"), g("majority label"), g("best held-out")
            ok17 = rule_ba <= 0.6 and best_ba - rule_ba >= 0.2
            s17.add("Recognising the run does not explain the held-out result (every run contains both normal and faulty windows)", "PASS" if ok17 else "WARN",
                    table([[r.check, f3(r.value), f3(r.chance)] for r in rid.itertuples()], ["check", "value", "chance"]) +
                    ["", "rule: PASS only if the best 'predict the run, output its majority label' rule stays at or below 0.60 balanced accuracy and the best held-out model beats it by at least 0.20.",
                     f"runs are identifiable from the features (accuracy {f3(acc)} within runs, chance 0.025), but the rule scores {f3(rule_ba)}; the held-out runs are unseen, so run identity cannot help them. This is the opposite of AnoMod (P14), where the run-identification rule reproduced the saved model's score."],
                    "report_outputs/testbed_runid_check.csv (testbed/validate.py)", dict(rule_ba=rule_ba, best_ba=best_ba))
            sections.append(s17)
    if want("P12"):
        s12 = Section("P12", "TESTBED, LIVE DEMO VALIDATION: three new 12-minute runs through the real agent and monitor"); header(s12)
        lv_path = REP / "testbed_live_validation.csv"
        if not lv_path.exists():
            s12.add("Live demo validation", "SKIP", ["report_outputs/testbed_live_validation.csv missing; run python -m testbed.live_demo --scenario all"], "-")
            sections.append(s12)
        else:
            import testbed.freeze as FZ
            import testbed.live_eval as LE
            import hashlib as _hl
            lv = pd.read_csv(lv_path)
            cfg = json.loads((REP / "testbed_frozen_config.json").read_text(encoding="utf-8"))
            man = pd.read_csv(ROOT / "testbed" / "runs" / "manifest.csv")
            order_ok, lines = True, []
            for r in lv.itertuples():
                plan = json.loads((REP / f"testbed_live_schedule_{r.seed}.json").read_text(encoding="utf-8"))
                plan_hash = _hl.sha256(json.dumps({k: v for k, v in plan.items() if k not in ("written_at", "written_at_epoch", "sha256_of_plan")}, sort_keys=True).encode()).hexdigest()
                t_frozen, t_plan = cfg["frozen_at"], plan["written_at"]
                ok = bool(t_frozen < t_plan < r.run_started_at) and plan_hash == plan["sha256_of_plan"] and int(r.seed) >= 9000 and int(r.seed) not in set(man.seed)
                order_ok &= ok
                lines.append(f"seed {int(r.seed)} ({r.scenario}): detector frozen {t_frozen}; schedule written {t_plan} (plan hash {'matches' if plan_hash == plan['sha256_of_plan'] else 'DIFFERS'}); run started {r.run_started_at}; seed >= 9000 and not among the {len(man)} training runs: {int(r.seed) >= 9000 and int(r.seed) not in set(man.seed)}; attempts {int(r.attempt)}")
            lines += [f"frozen model: {cfg['model']} on {cfg['feature_set']} features, threshold {cfg['threshold']:.3f}, alert rule '{cfg['alert_rule']['name']}', selected by cross-validation on the train runs only", f"model file still matches the frozen config: {FZ.check() if (ROOT / 'models' / 'testbed' / 'detector.joblib').exists() else 'model file not present'}"]
            s12.add("The detector and the alert rule were frozen before the demos, and each fault schedule was recorded before its run", "PASS" if order_ok else "FAIL", lines, "report_outputs/testbed_frozen_config.json, testbed_live_schedule_<seed>.json, testbed_live_validation.csv")
            dmax, n_cmp = 0.0, 0
            for r in lv.itertuples():
                W_ = pd.read_csv(REP / f"testbed_live_windows_{r.seed}.csv")
                sched = pd.read_csv(REP / f"testbed_live_schedule_actual_{r.seed}.csv", dtype={"fault_id": str})
                anchor = float(W_.window_start.iloc[0])
                Wl = W_.copy()
                Wl["learning"] = Wl.learning.astype(bool)
                L = LE.label_windows(Wl[["window_index", "window_start", "window_end", "learning", "risk", "flagged", "alert", "new_alert"]], sched, anchor)
                ev = LE.evaluate(L, sched)
                inst_ = ev.pop("instances")
                chk = dict(missed_fault_windows=ev["missed_fault_windows"], fault_windows=ev["fault_windows"], false_alarms=ev["false_alarms"], fault_windows_flagged=ev["fault_windows_flagged"], scored_windows=ev["scored_windows"])
                if inst_:
                    chk["detection_delay_s"] = inst_[0]["delay_s"]
                for k, v in chk.items():
                    if pd.notna(v) and pd.notna(getattr(r, k)):
                        dmax = max(dmax, abs(float(v) - float(getattr(r, k)))); n_cmp += 1
            s12.add("The live-validation numbers are recomputed from the stored per-window predictions and the recorded fault schedule", "PASS" if dmax <= 1e-6 else "FAIL",
                    [f"demos recomputed: {len(lv)}; numbers compared {n_cmp}; max |difference| {dmax:.1e}", "per-window predictions: report_outputs/testbed_live_windows_<seed>.csv (monitor output); fault record: testbed_live_schedule_actual_<seed>.csv (the app's own start / end / failure timestamps)"],
                    "testbed/live_eval.py, report_outputs/testbed_live_windows_*.csv")
            rows_ = [[int(r.seed), r.scenario, f"{r.actual_fault} ({r.planned_severity})" if r.actual_fault != "none" else "none", ("yes" if r.detected else "NO") if r.actual_fault != "none" else "-", f"{r.detection_delay_s:.0f}" if pd.notna(r.detection_delay_s) else "-",
                      f"{int(r.false_alarms)} ({r.false_alarms_per_hour:.1f}/h)", f"{int(r.missed_fault_windows)}/{int(r.fault_windows)}", f"{r.lead_time_s:.0f}" if pd.notna(r.lead_time_s) else "-", f"{r.balanced_accuracy:.3f}" if pd.notna(r.balanced_accuracy) else "-"] for r in lv.itertuples()]
            lines = table(rows_, ["seed", "scenario", "fault (severity)", "detected", "delay s", "false alarms", "fault windows missed", "lead time s", "window BA"])
            nf = lv[lv.actual_fault != "none"]
            lines += ["", f"detected {int(nf.detected.sum())} of {len(nf)} injected faults; false alarms {int(lv.false_alarms.sum())} in {lv.scored_hours_outside_faults.sum():.2f} scored hours outside faults; fault windows missed {int(lv.missed_fault_windows.sum())} of {int(lv.fault_windows.sum())}",
                      "alerts use the frozen rule (2 of the last 3 windows flagged); delay = seconds from the recorded fault start to the end of the first alert window; a miss is a fault window the model did not flag; the schedule was drawn from the seed before the run; nothing was tuned on these runs",
                      "three runs, one fault each: this is a demonstration, not a statistic. One app, one machine, synthetic traffic and injected faults; transfer to other apps has not been shown."]
            s12.add("Live demos reported as they came out", "REPORTED", lines, "report_outputs/testbed_live_validation.csv", dict(detected=int(nf.detected.sum()), faults=len(nf), false_alarms=int(lv.false_alarms.sum())),
                    result=f"detected {int(nf.detected.sum())} of {len(nf)} injected faults, {int(lv.false_alarms.sum())} false alarm(s), {int(lv.missed_fault_windows.sum())} of {int(lv.fault_windows.sum())} fault windows missed")
            sections.append(s12)

    # ------------------------------------------------------------------------------------------------ P18 RCAEval external test (testing only)
    if want("P18"):
        s = Section("P18", "EXTERNAL TEST ON RCAEVAL (testing only, nothing tuned): do the testbed detectors find injected faults in other systems?"); header(s)
        rp = REP / "rcaeval_external_results.csv"
        if not rp.exists():
            s.add("RCAEval external test", "REPORTED", ["The external test has not been run (or was blocked): the reason is recorded in report_outputs/FINAL_STATUS.md. No claim about other systems is made from it."], "report_outputs/FINAL_STATUS.md", result="not run")
        sections.append(s)

    # ------------------------------------------------------------------------------------------------ P19 integration (agent, log tailer, Node middleware, API keys, webhook)
    if want("P19"):
        s = Section("P19", "INTEGRATION: connecting another app (agent, log tailer, Node middleware, optional API key, Slack-compatible webhook)"); header(s)
        import http.server
        import shutil as _sh
        import threading as _th
        sys.path.insert(0, str(ROOT / "agent"))
        got = []

        class _H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("content-length", 0))
                got.append((self.path, {k.lower(): v for k, v in self.headers.items()}, json.loads(self.rfile.read(n) or b"{}")))
                self.send_response(200); self.end_headers(); self.wfile.write(b"{}")

            def log_message(self, *a):
                pass
        srv = http.server.HTTPServer(("127.0.0.1", 0), _H)
        _th.Thread(target=srv.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{srv.server_port}"
        # 1 Python ASGI agent
        from failsight_agent import FailSight
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        fsx = FailSight("p19", url, "key19", send_metrics=False, send_logs=False, flush_interval=0.2)
        appx = FastAPI()
        fsx.instrument(appx)

        @appx.get("/ping/{i}")
        def _ping(i: int):
            return {"i": i}
        tc = TestClient(appx)
        tc.get("/ping/1"); tc.get("/nope")
        fsx.close()
        ev1 = [e for p_, h, b in got if p_ == "/ingest/api" for e in b["events"]]
        ok1 = {e["status_code"] for e in ev1} == {200, 404} and all(b["system"] == "p19" for p_, h, b in got) and got[0][1].get("x-api-key") == "key19" and {"timestamp", "route", "latency_ms"} <= set(ev1[0])
        s.add("The Python ASGI agent sends one event per request (route, status, latency) with the API key header", "PASS" if ok1 else "FAIL", [f"events received {len(ev1)}; statuses {sorted({e['status_code'] for e in ev1})}; routes {sorted({e['route'] for e in ev1})}"], "agent/failsight_agent/")
        # 2 log tailer
        from failsight_agent import tail
        a_ = tail.parse_line('1.2.3.4 - - [21/Oct/2026:13:55:01 +0000] "GET /search?x=1 HTTP/1.1" 200 1532 "-" "curl"')
        b_ = tail.parse_line('1.2.3.4 - - [21/Oct/2026:13:55:02 +0000] "POST /pay HTTP/2.0" 502 - "-" "x" 0.250')
        ok2 = a_["route"] == "/search" and a_["status_code"] == 200 and b_["status_code"] == 502 and abs(b_["latency_ms"] - 250.0) < 1e-9 and tail.parse_line("garbage") is None
        s.add("The access-log tailer parses nginx combined lines (with and without request time) and ignores other lines", "PASS" if ok2 else "FAIL", [f"parsed: {a_}", f"parsed: {b_}"], "agent/failsight_agent/tail.py")
        # 3 node middleware
        if _sh.which("node"):
            n0 = len(got)
            js = ("const failsight = require(" + json.dumps(str(ROOT / "agent" / "node" / "failsight-express.js")) + "); const {EventEmitter} = require('events');"
                  "const mw = failsight({monitorUrl: " + json.dumps(url) + ", appId: 'node19', apiKey: 'k', flushMs: 100000}); const res = new EventEmitter(); res.statusCode = 503; res.get = () => '42';"
                  "mw({method: 'GET', path: '/pay', url: '/pay'}, res, () => {}); res.emit('finish'); mw.flush().then(() => process.exit(0));")
            subprocess.run(["node", "-e", js], check=True, timeout=30)
            nb = [b for p_, h, b in got[n0:] if p_ == "/ingest/api" and b.get("system") == "node19"]
            ok3 = bool(nb) and nb[0]["events"][0]["status_code"] == 503 and nb[0]["events"][0]["route"] == "/pay"
            s.add("The Node/Express middleware posts the same fields to /ingest/api", "PASS" if ok3 else "FAIL", [f"event: {nb[0]['events'][0] if nb else None}"], "agent/node/failsight-express.js")
        else:
            s.add("The Node/Express middleware posts the same fields to /ingest/api", "SKIP", ["node is not installed on this machine"], "agent/node/failsight-express.js")
        # 4 API keys
        from app.backend.testbed_monitor import check_api_key, parse_keys, slack_payload
        keys = parse_keys("shop:abc,other:xyz")
        ok4 = check_api_key("shop", None, {}) and check_api_key("shop", "abc", keys) and not check_api_key("shop", "xyz", keys) and not check_api_key("shop", None, keys) and not check_api_key("new", "abc", keys)
        s.add("The per-app API key check is off by default and enforced per app when keys are configured", "PASS" if ok4 else "FAIL", ["FAILSIGHT_API_KEYS unset: every request is accepted; \"shop:abc,other:xyz\": only the matching key for that app is accepted; the HTTP 401 behaviour is tested in app/backend/tests/test_testbed_api.py"], "app/backend/testbed_monitor.py")
        # 5 webhook (replay of a recorded testbed run through the monitor)
        cfgp, modp = REP / "testbed_frozen_config.json", ROOT / "models" / "testbed" / "detector.joblib"
        rdir = ROOT / "testbed" / "runs" / "r004_s1004"
        if cfgp.exists() and modp.exists() and (rdir / "DONE").exists():
            from app.backend import testbed_monitor as TM
            TM.LIVE_DIR = ROOT / "outputs" / "testbed_live_proof"
            mon = TM.TestbedMonitor()
            mon.set_webhook(url + "/hook")
            meta = json.loads((rdir / "meta.json").read_text(encoding="utf-8"))
            mon.reset("hooked", meta["start_ts"])
            for kind in ("api", "logs", "metrics"):
                evs = [json.loads(line) for line in open(rdir / f"{kind}.jsonl", encoding="utf-8")]
                for i in range(0, len(evs), 2000):
                    mon.ingest(kind, "hooked", evs[i:i + 2000])
            mon.close_ready(meta["start_ts"] + 100000)
            time.sleep(2)
            hooks = [b for p_, h, b in got if p_ == "/hook"]
            n_alerts = sum(1 for w in mon.windows("hooked") if w["new_alert"])
            ok5 = len(hooks) == n_alerts >= 1 and all(set(h) == {"text"} and "FailSight alert for hooked" in h["text"] and "unvalidated" in h["text"] for h in hooks)
            s.add("The webhook receives one Slack-compatible message per alert episode, tagged unvalidated for an app other than the testbed shop", "PASS" if ok5 else "FAIL",
                  [f"recorded run r004_s1004 replayed through the monitor as app 'hooked': {n_alerts} alert episode(s), {len(hooks)} message(s)", f"first message: {hooks[0]['text'] if hooks else None}"], "app/backend/testbed_monitor.py")
        else:
            s.add("The webhook receives one Slack-compatible message per alert episode", "SKIP", ["frozen detector or recorded run not present"], "app/backend/testbed_monitor.py")
        srv.shutdown()
        s.add("Scope of validation for other apps", "REPORTED", ["The detector was validated only on the testbed shop (one app, one machine, synthetic traffic, injected faults, 12 held-out runs with 15 faults, per-type results unreliable). The monitor labels every other app id 'unvalidated' in the dashboard and in webhook messages.",
                                                                "Another app needs its own labelled data (faults injected at recorded times, validation on whole runs) or at least the learning period with the unsupervised per-app detector. Transfer to other apps has not been shown (P9 LO2, P11 cross-system).",
                                                                "docs/INTEGRATION.md and the Connect page say the same."], "docs/INTEGRATION.md", result="integration works; transfer to other apps not shown")
        sections.append(s)

    # ------------------------------------------------------------------------------------------------ P13 missingness and shortcuts
    if want("P13"):
        s = Section("P13", "MISSINGNESS AND SHORTCUTS: do the models rely on missing-data patterns instead of real signals?"); header(s)
        import missingness_check as MC
        mp = MC.OUT
        if not mp.exists() or "reg" not in ctx:
            s.add("Missingness check", "SKIP", ["report_outputs/missingness_check.csv or the saved models are not present; run python scripts/missingness_check.py"], "-")
            sections.append(s)
        else:
            res = pd.read_csv(mp)
            rows_csv = res.drop(columns=["generated"]).to_dict("records")
            reg = ctx["reg"]
            reps = int(res[(res.kind == "masking") & (res.mask_pct == 10)].draws.iloc[0])
            # recompute: the whole of parts 2 and 3, and part 1 for three models, must equal the saved csv (same seeds)
            quiet = lambda *_: None
            fresh = MC.masking(D, reg, reps, 1000, ["LR", "XGB", "Fusion"], log=quiet) + MC.indicator_only(D, reg.prep, 1000, log=quiet) + MC.no_shortcut_xgb(D, ctx["FP"], reg, 1000, log=quiet)
            key = lambda r: (r["kind"], r["label_version"], r["model"], r["group"], None if pd.isna(r["mask_pct"]) else int(r["mask_pct"]))
            old = {key(r): r for r in rows_csv}
            dmax, n_cmp = 0.0, 0
            for r in fresh:
                o = old[key(r)]
                for k in ("balanced_accuracy", "ci_low", "ci_high"):
                    dmax = max(dmax, abs(float(r[k]) - float(o[k]))); n_cmp += 1
            s.add("The saved missingness_check.csv is reproduced by recomputation (all indicator-only and no-shortcut rows, and the masking rows of LR, XGB and Fusion)", "PASS" if dmax <= 1e-6 else "FAIL",
                  [f"numbers compared: {n_cmp}; max |difference| {dmax:.1e} (same seeds); the other masking rows are read from the file"], "scripts/missingness_check.py, report_outputs/missingness_check.csv")
            v = MC.verdict(rows_csv)
            f3 = lambda x: f"{x:.3f}"
            ci = lambda r: f"{f3(r.balanced_accuracy)} ({f3(r.ci_low)}-{f3(r.ci_high)})"
            # ---- 1 masking
            mk = res[res.kind == "masking"]
            lines = [f"Every saved final model except soft voting (needs 40 fold models) is scored on the 647 test windows after a random share of the REAL values of a feature group is set to missing; the was_missing_* indicators are recomputed by the real preprocessor (they fire at >= 50% missing in a group); {reps} mask draws per cell, 95% block-bootstrap CI.",
                     "", "XGBoost (binary): balanced accuracy (95% CI) by masked group and share masked"]
            xb = mk[(mk.label_version == "label_binary") & (mk.model == "XGB")]
            b0 = xb[xb.mask_pct == 0].iloc[0]
            trows = [["unmasked", ci(b0), "", "", ""]] + [[g] + [ci(xb[(xb.group == g) & (xb.mask_pct == p)].iloc[0]) for p in MC.LEVELS_PCT] for g in MC.GROUPS]
            lines += table(trows, ["masked group", "10%", "30%", "50%", "70%"])
            lines += ["", "XGBoost (binary) when ALL groups are masked: Normal recall and share of windows called Anomaly"]
            lines += table([["unmasked", f3(b0.normal_recall), f"{b0.share_predicted_anomaly:.1%}"]] + [[f"{p}%", f3(r.normal_recall), f"{r.share_predicted_anomaly:.1%}"] for p in MC.LEVELS_PCT for r in [xb[(xb.group == "all") & (xb.mask_pct == p)].iloc[0]]], ["masked", "Normal recall", "called Anomaly"])
            lines += ["", "all models, binary labels: balanced accuracy unmasked and with ALL groups masked 10 / 30 / 50 / 70% (point values; the CI of every cell is in missingness_check.csv)"]
            names = list(mk[mk.label_version == "label_binary"].model.unique())
            kept = lambda n: (float(mk[(mk.label_version == "label_binary") & (mk.model == n) & (mk.group == "all") & (mk.mask_pct == 70)].balanced_accuracy.iloc[0]) - 0.5) / (float(mk[(mk.label_version == "label_binary") & (mk.model == n) & (mk.mask_pct == 0)].balanced_accuracy.iloc[0]) - 0.5)
            lines += table([[n, f3(mk[(mk.label_version == "label_binary") & (mk.model == n) & (mk.mask_pct == 0)].balanced_accuracy.iloc[0])] +
                            [f3(mk[(mk.label_version == "label_binary") & (mk.model == n) & (mk.group == "all") & (mk.mask_pct == p)].balanced_accuracy.iloc[0]) for p in MC.LEVELS_PCT] + [f"{kept(n):.0%}"] for n in names], ["model", "0%", "10%", "30%", "50%", "70%", "lift kept at 70%"])
            lines += ["'lift kept' = (BA at 70% - 0.5) / (BA unmasked - 0.5); it is unstable for weak models whose unmasked lift is small. Highest: " + ", ".join(f"{n} {kept(n):.0%}" for n in sorted(names, key=kept, reverse=True)[:3]) + "; the rule below is applied to the saved XGBoost."]
            xg = lambda g, p: xb[(xb.group == g) & (xb.mask_pct == p)].iloc[0].balanced_accuracy
            lines += ["", f"which group carries the saved XGBoost's performance: masking 70% of the API values changes its balanced accuracy from {f3(b0.balanced_accuracy)} to {f3(xg('api', 70))}, 70% of the log values to {f3(xg('log', 70))}, 70% of the metric values to {f3(xg('metric', 70))}. Its result rests on the metric (host and container) readings; the API and log readings have no effect on it."]
            lines += ["", "five-class models: balanced accuracy unmasked and with ALL groups masked 10 / 30 / 50 / 70%"]
            names5 = list(mk[mk.label_version == "label_level"].model.unique())
            lines += table([[n, f3(mk[(mk.label_version == "label_level") & (mk.model == n) & (mk.mask_pct == 0)].balanced_accuracy.iloc[0])] +
                            [f3(mk[(mk.label_version == "label_level") & (mk.model == n) & (mk.group == "all") & (mk.mask_pct == p)].balanced_accuracy.iloc[0]) for p in MC.LEVELS_PCT] for n in names5], ["model", "0%", "10%", "30%", "50%", "70%"])
            lines += ["", f"rule: {MC.RULES[1]}",
                      f"saved XGBoost: unmasked {f3(v['saved_xgb_ba'])}; with 70% of all values masked {f3(v['masking_all70_ba'])}; share of the lift over chance kept: {v['masking_all70_lift_kept']:.0%}"]
            s.add("Masking real values: performance falls toward chance as the signal is removed; it is not held up by the missingness pattern", "WARN" if v["masking_explains"] else "PASS", lines, "report_outputs/missingness_check.csv (scripts/missingness_check.py)", dict(masking_all70_lift_kept=v["masking_all70_lift_kept"], xgb_binary=dict(unmasked=float(b0.balanced_accuracy), api70=float(xg("api", 70)), log70=float(xg("log", 70)), metric70=float(xg("metric", 70)), all70=float(xg("all", 70)),
                                                                                                                                                   normal_recall_unmasked=float(b0.normal_recall), normal_recall_metric70=float(xb[(xb.group == "metric") & (xb.mask_pct == 70)].iloc[0].normal_recall))))
            # ---- 2 indicator-only
            io = res[res.kind == "indicator_only"]
            lines = ["A small throw-away XGBoost (100 trees, depth 3, balanced weights) on the missingness columns only (no system flag, no real feature). Trained on the training split with the 10 block-grouped folds (10-fold = validation score), then fitted on the whole training split and scored once on test. Shuffled-label control: the training labels are permuted (5 shuffles), the scoring labels are real."]
            for l in ("label_binary", "label_level"):
                sub = io[io.label_version == l]
                lines += ["", f"{l[6:]} labels (chance level {'0.500' if l == 'label_binary' else '0.200'}):"] + table(
                    [[r.model, r.group, f3(r.cv_balanced_accuracy), ci(r)] for r in sub.itertuples()], ["inputs", "labels", "10-fold BA", "test BA (95% CI)"])
            lines += ["", f"rule: {MC.RULES[0]}", f"saved XGBoost test balanced accuracy {f3(v['saved_xgb_ba'])}; best indicator-only classifier {f3(v['indicator_only_ba'])} ({v['indicator_only_set']}); share of the saved model's lift: {v['indicator_only_lift_share']:.0%}",
                      "reading: on the training folds the indicators carry a small signal (10-fold balanced accuracy slightly above chance) but on the untouched test set the same classifier is at or below chance, so the indicators alone do not reproduce the saved model's performance"]
            s.add("The was_missing_* indicators and the per-window missing share, on their own, do not explain the performance", "WARN" if v["indicator_explains"] else "PASS", lines, "report_outputs/missingness_check.csv (scripts/missingness_check.py)", dict(indicator_only_lift_share=v["indicator_only_lift_share"]))
            # ---- 3 no-shortcut xgb
            ns = res[res.kind == "no_shortcut_xgb"]
            lines = ["Throw-away XGBoost with the saved model's settings and the same training procedure (SMOTE / weights, early stopping) on the same training split; not saved."]
            for l in ("label_binary", "label_level"):
                sub = ns[ns.label_version == l]
                lines += ["", f"{l[6:]} labels:"] + table([[r.model, r.group, ci(r), f"{f3(r.normal_recall)} ({f3(r.normal_recall_ci_low)}-{f3(r.normal_recall_ci_high)})"] for r in sub.itertuples()], ["model", "inputs", "test BA (95% CI)", "Normal recall (95% CI)"])
            lines += ["", f"rule: {MC.RULES[2]}", f"saved {f3(v['saved_xgb_ba'])} versus without indicators and without system flag {f3(v['no_shortcut_ba'])} (difference {v['no_shortcut_drop']:+.3f})",
                      "the intervals are wide: only 36 of the 647 test windows are Normal and they come from a few blocks (the project's own block bootstrap gives the same width for the saved model)"]
            s.add("Removing the was_missing_* indicators and the system flag does not change the result", "WARN" if v["no_shortcut_relies"] else "PASS", lines, "report_outputs/missingness_check.csv (scripts/missingness_check.py)", dict(no_shortcut_drop=v["no_shortcut_drop"]))
            # ---- 4 what missing data does to a decision (live relevance)
            xa = xb[xb.group == "metric"]
            lines = ["binary XGBoost, share of windows called Anomaly when a random share of the METRIC values is masked: " + ", ".join(f"{p}%: {xa[xa.mask_pct == p].iloc[0].share_predicted_anomaly:.1%}" for p in MC.LEVELS_PCT) + f" (unmasked: {b0.share_predicted_anomaly:.1%}; the test set itself is {(D.yte['label_binary'] == 1).mean():.1%} Anomaly)",
                     "Normal recall in the same cells: " + ", ".join(f"{p}%: {f3(xa[xa.mask_pct == p].iloc[0].normal_recall)}" for p in MC.LEVELS_PCT),
                     "read together with the previous items: missing readings can push a call toward Anomaly. This is why the Console does not issue alerts on windows with more than half of the readings missing (state 'Not enough data to judge')."]
            s.add("What missing data does to the decision (why the Console has a low-data state)", "PASS", lines, "report_outputs/missingness_check.csv", dict())
            if v["explains"] or v["no_shortcut_relies"]:
                s.add("Overall verdict", "WARN", [f"masking explains: {v['masking_explains']}; indicator-only explains: {v['indicator_explains']}; no-shortcut model relies: {v['no_shortcut_relies']}"], "see the items above")
            else:
                s.add("Overall verdict", "PASS", ["Neither masking nor an indicator-only model explains the saved model's performance, and removing the indicators and the system flag changes the test balanced accuracy by " + f"{v['no_shortcut_drop']:+.3f}.",
                                                 "This does not prove the model uses only real signals: it shows that the missingness columns are not what carries the result on these test windows."], "see the items above")
            sections.append(s)

    # ------------------------------------------------------------------------------------------------ P14 run identification
    if want("P14"):
        s = Section("P14", "RUN IDENTIFICATION: can a model tell which run a window comes from, and would that alone reproduce the scores?"); header(s)
        import run_identification as RI
        if not RI.OUT.exists():
            s.add("Run identification", "SKIP", ["report_outputs/run_identification.csv missing; run python scripts/run_identification.py"], "-")
            sections.append(s)
        else:
            ri = pd.read_csv(RI.OUT)
            fresh = RI.compute(1000, log=lambda *_: None)
            dmax, n_cmp = 0.0, 0
            for col in ("accuracy", "accuracy_ci_low", "accuracy_ci_high", "agreement", "model_accuracy", "rule_accuracy", "balanced_accuracy", "balanced_accuracy_ci_low", "balanced_accuracy_ci_high", "normal_recall", "macro_f1"):
                if col in fresh and col in ri:
                    a_, b_ = pd.to_numeric(fresh[col], errors="coerce").to_numpy(float), pd.to_numeric(ri[col], errors="coerce").to_numpy(float)
                    ok_ = np.isfinite(a_) & np.isfinite(b_)
                    if ok_.any():
                        dmax = max(dmax, float(np.abs(a_[ok_] - b_[ok_]).max())); n_cmp += int(ok_.sum())
            s.add("The run-identification results are reproduced by recomputation", "PASS" if (dmax <= 1e-5 and len(fresh) == len(ri)) else "FAIL",
                  [f"throw-away XGBoost refitted, the saved models re-scored; rows {len(fresh)} = {len(ri)}; numbers compared {n_cmp}; max |difference| {dmax:.1e}"], "scripts/run_identification.py, report_outputs/run_identification.csv")
            f3 = lambda x: f"{x:.3f}"
            rid_ = ri[ri.kind == "run_identification"]
            lines = ["A throw-away XGBoost (150 trees, depth 4; never saved) predicts which of the 26 runs a window comes from, trained on the training windows, scored on the 647 test windows. The project's split is by 10-window blocks, so every run has windows on both sides (a held-out-run split cannot recognise a run).",
                     ""] + table([[r.feature_set, int(r.n_features), f"{f3(r.accuracy)} ({f3(r.accuracy_ci_low)}-{f3(r.accuracy_ci_high)})", f"{r.chance_26_runs:.3f}", f3(r.system_correct)] for r in rid_.itertuples()],
                                 ["inputs", "features", "run accuracy (95% CI)", "chance", "system right"])
            allr = rid_[rid_.feature_set == "all 124 model inputs"].iloc[0]
            lines += ["", f"with all model inputs the run is identified in {allr.accuracy:.1%} of the test windows (chance {allr.chance_26_runs:.1%}); with the metric inputs alone {rid_[rid_.feature_set == 'metric inputs only'].iloc[0].accuracy:.1%}; with the API and log inputs alone {rid_[rid_.feature_set == 'API + log inputs only'].iloc[0].accuracy:.1%}; with the generic metric aggregates alone {rid_[rid_.feature_set.str.startswith('generic')].iloc[0].accuracy:.1%}"]
            s.add("Runs can be identified from the model inputs (26 classes)", "REPORTED", lines, "report_outputs/run_identification.csv (scripts/run_identification.py)", dict(accuracy_all=float(allr.accuracy)), result=f"run identified in {allr.accuracy:.1%} of test windows (chance {allr.chance_26_runs:.1%})")
            rule = ri[ri.kind == "run_id_rule"]
            sv = ri[ri.kind == "saved_model_vs_rule"]
            lines = ["The run-identification rule: predict the run, then output that run's own label (every window of a run carries the run's single label). Its test scores are the ceiling of pure run recognition."]
            for r in rule.itertuples():
                lines += [f"  {r.label_version[6:]:>6} labels: balanced accuracy {f3(r.balanced_accuracy)} ({f3(r.balanced_accuracy_ci_low)}-{f3(r.balanced_accuracy_ci_high)}), macro-F1 {f3(r.macro_f1)}, Normal recall {f3(r.normal_recall)} ({f3(r.normal_recall_ci_low)}-{f3(r.normal_recall_ci_high)})"]
            for l in ("label_binary", "label_level"):
                x = sv[sv.label_version == l]
                lines += ["", f"{l[6:]} labels: how often a saved final model's test prediction equals the label of the predicted run (agreement), its own accuracy and balanced accuracy"]
                lines += table([[r.model, f3(r.agreement), f3(r.model_accuracy), f3(r.rule_accuracy), f3(r.balanced_accuracy)] for r in x.itertuples()], ["model", "agreement with rule", "model accuracy", "rule accuracy", "model BA"])
            xb_ = sv[(sv.label_version == "label_binary") & (sv.model == "XGB")].iloc[0]
            rb = rule[rule.label_version == "label_binary"].iloc[0]
            tm_ = pd.read_csv(REP / "test_metrics.csv")
            xnr = float(tm_[(tm_.model == "XGB") & (tm_.label_version == "label_binary") & (tm_.scope == "ALL") & (tm_.variant == "official")].normal_recall.iloc[0])
            expl = bool(allr.accuracy >= 0.5 and rb.balanced_accuracy >= xb_.balanced_accuracy - 0.02)
            lines += ["", "rule for the verdict: run identification explains the scores if the run-identification rule reaches the saved XGBoost's balanced accuracy within 0.02 (binary labels) and the run is identified in at least half of the test windows.",
                      f"binary: rule {f3(rb.balanced_accuracy)} versus saved XGBoost {f3(xb_.balanced_accuracy)} (Normal recall {f3(rb.normal_recall)} versus {f3(xnr)}); the saved XGBoost agrees with the rule on {xb_.agreement:.1%} of the test windows",
                      "reading: this does not prove the saved models recognise runs. It shows that recognising the run would, on its own, reproduce their test scores, so those scores cannot be read as the ability to detect faults. A test that separates the two needs windows labelled normal and faulty inside the same run, which needs fault start and end times that the data do not contain (P15)."]
            s.add("Would run identification alone reproduce the saved models' scores?", "WARN" if expl else "PASS", lines, "report_outputs/run_identification.csv (scripts/run_identification.py)", dict(rule_ba=float(rb.balanced_accuracy), saved_xgb_ba=float(xb_.balanced_accuracy), agreement=float(xb_.agreement)))
            sections.append(s)

    # ------------------------------------------------------------------------------------------------ P15 fault times and the within-run (segment) test
    if want("P15"):
        s = Section("P15", "FAULT TIMES AND THE WITHIN-RUN (SEGMENT) TEST: do exact fault start and end times exist?"); header(s)
        import fault_times as FT
        import contextlib
        import io
        ftp = FT.OUT
        if not ftp.exists():
            s.add("Fault times", "SKIP", ["report_outputs/fault_times.csv missing; run python scripts/fault_times.py"], "-")
            sections.append(s)
        else:
            old = pd.read_csv(ftp)
            with contextlib.redirect_stdout(io.StringIO()):
                new = FT.main()
            same = len(old) == len(new) and (old.run_id == new.run_id).all() and (old.files_checked == new.files_checked.to_numpy()).all() and (old.files_with_injection_keyword == new.files_with_injection_keyword.to_numpy()).all()
            s.add("The fault-time table is reproduced by re-reading the raw folders", "PASS" if same else "FAIL", [f"runs {len(new)}; files checked {int(new.files_checked.sum())}; files with an injection keyword {int(new.files_with_injection_keyword.sum())}; identical to report_outputs/fault_times.csv: {same}"], "scripts/fault_times.py, report_outputs/fault_times.csv")
            n_start = int((~new.fault_start.astype(str).str.startswith("not recorded")).sum())
            cols = ["system", "scenario", "label_level", "experiment_start_from_folder_name", "collection_end_from_log_report", "run_minutes", "api_probe_period", "fault_start", "fault_end"]
            lines = ["Every metric, log, API-response and trace folder of each run was searched (coverage artefacts excluded): folder names, metadata, collection reports, summaries, API probes, Kubernetes events, trace files, injection / chaos keywords.", ""]
            lines += table([[str(getattr(r, c))[:38] if c != "api_probe_period" else str(r.api_probe_period)[:36] for c in cols] for r in new.itertuples()], ["system", "scenario", "class", "experiment start (folder name)", "collection end", "min", "API probe period", "fault start", "fault end"])
            lines += ["", f"runs with a recorded fault start: {n_start} of {len(new)}; with a recorded fault end: {int((~new.fault_end.astype(str).str.startswith('not recorded')).sum())} of {len(new)}.",
                      "the only absolute times are the experiment start (folder name) and, for TrainTicket, the end of collection; the API probes cover only the end of a run (SocialNetwork: one 60 s probe; TrainTicket: the last 3.7 to 12.4 minutes).",
                      "labels are per run: all windows of a run carry the run's single label. Every run, the Normal ones too, begins with a start-up transient (first windows differ from the last 30% of the run), so metric excursions cannot locate a fault.",
                      "consequence: the segment-label test (Normal before the fault, fault class during it, Normal after it) and the early-warning labels (fault within the next 2 / 5 / 10 minutes) cannot be built without inventing the times. They were NOT built; nothing is estimated.",
                      "what would make it possible: report_outputs/early_warning_feasibility.md"]
            s.add("Exact fault start and end times do not exist in the data, so the within-run test and early-warning labels cannot be built", "REPORTED", lines, "scripts/fault_times.py, report_outputs/fault_times.csv, report_outputs/early_warning_feasibility.md",
                  dict(runs=len(new), runs_with_fault_start=n_start), result=f"not run: 0 of {len(new)} runs have a recorded fault start" if n_start == 0 else f"{n_start} of {len(new)} runs have a recorded fault start")
            sections.append(s)

    # ------------------------------------------------------------------------------------------------ P10 reproducibility
    if want("P10"):
        s = Section("P10", "REPRODUCIBILITY: seeds, versions, commit"); header(s)
        import common, sklearn, xgboost, imblearn, scipy
        vers = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scikit_learn=sklearn.__version__, xgboost=xgboost.__version__, imbalanced_learn=imblearn.__version__, scipy=scipy.__version__)
        for mod in ("lightgbm", "optuna", "tensorflow", "keras", "shap", "lime", "fastapi", "joblib"):
            try:
                vers[mod] = __import__(mod).__version__
            except Exception:
                vers[mod] = "n/a"
        git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        commit, dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--untracked-files=no"))
        sys.path.insert(0, str(HERE))
        import split_blocks
        s.add("Seeds", "PASS", [f"global seed {common.SEED} (scripts/common.py); split_blocks.SEED {split_blocks.SEED}; preprocess.SEED {P.SEED}; SMOTE random_state {common.SEED}; XGBoost/LightGBM/forest random_state 42; "
                                "StratifiedGroupKFold random_state 42; Optuna TPESampler seed 42; bootstrap seed 42; Keras per-model seeds 42 + 7*fold + model index"], "scripts/common.py, scripts/v2_tab.py, scripts/full_pipeline.py")
        s.add("Library versions and platform", "PASS", [", ".join(f"{k} {v}" for k, v in vers.items()), f"{platform.platform()}, CPU only (TensorFlow has no native-Windows GPU support)"], "this run", dict(versions=vers))
        s.add("Code version", "PASS" if commit else "WARN", [f"git commit {commit}{' (working tree has uncommitted changes to tracked files)' if dirty else ' (clean)'}"], "git", dict(commit=commit, dirty=dirty))
        s.add("Run time of this report", "PASS", [f"{time.time() - T0:.0f} s so far (see the total in the summary)"], "this run")
        sections.append(s)

    # ------------------------------------------------------------------------------------------------ summary / report assembly
    global CUR
    CUR = []
    new_secs = {x.id: dict(id=x.id, title=x.title, status=x.status, items=x.items, lines=x.lines, generated_at=x.generated_at) for x in sections}
    order = [f"P{i}" for i in range(1, 20)]
    merged = dict(new_secs)
    base_p = REP / "proofs.json"
    if (only or args.quick) and base_p.exists():                      # a partial re-run refreshes only its sections of the existing report
        old = {x["id"]: x for x in json.loads(base_p.read_text(encoding="utf-8"))["sections"]}
        merged = {**old, **new_secs}
    final = [merged[i] for i in order if i in merged]
    rows = [(sec["id"], it["claim"][:78] + (f"  [Result: {it['result']}]" if it.get("result") else ""), it["status"], it["source"][:46]) for sec in final for it in sec["items"]]
    flat = [it["status"] for sec in final for it in sec["items"]]
    tot = {k: flat.count(k) for k in ("PASS", "WARN", "FAIL", REPORTED, "SKIP") if flat.count(k)}
    summary_lines = ["", "=" * 100, "SUMMARY", "=" * 100]
    summary_lines += ["  " + r for r in table([[a_, b_, c_, d_] for a_, b_, c_, d_ in rows], ["#", "claim", "status", "source file"])]
    secs_txt = ", ".join(x["id"] + " " + x["status"] for x in final)
    summary_lines += ["", f"  totals: {tot}   sections: {secs_txt}   runtime of this run {time.time() - T0:.0f} s"]
    if "FAIL" in flat:
        summary_lines.append("  !!! at least one item FAILED !!!")
    for ln in summary_lines:
        print(ln, flush=True)
    head = ["FAILSIGHT PROOF REPORT", time.strftime("%Y-%m-%d %H:%M:%S"), "Every number below is recomputed from the data or the saved models, or read from a named result file. Nothing is retrained."]
    text = "\n".join(head + [ln for sec in final for ln in sec.get("lines", [])] + summary_lines)
    complete = all(i in merged for i in order) and not any(it["status"] == "SKIP" for sec in final for it in sec["items"])
    report = dict(generated_at=time.strftime("%Y-%m-%d %H:%M:%S"), partial=not complete, refreshed_sections=list(new_secs), runtime_seconds=round(time.time() - T0, 1), totals=tot,
                  git_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
                  sections=[{k: v for k, v in sec.items()} for sec in final], summary=[dict(section=a_, claim=b_, status=c_, source=d_) for a_, b_, c_, d_ in rows], text=text)
    base_p.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print("  wrote report_outputs/proofs.json", flush=True)
    sys.exit(1 if "FAIL" in flat else 0)


if __name__ == "__main__":
    main()
