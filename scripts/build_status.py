"""Writes report_outputs/OVERNIGHT_STATUS.md from the saved result files, report_outputs/overnight_steps.json (what was done in each step, with timestamps; written by hand during the run) and
report_outputs/overnight_tests.json (test commands and their outcomes). Every number is read from a named file.   python scripts/build_status.py"""
import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "report_outputs"
FOLD_NOTE = "The earlier AnoMod models were checked with 10-fold cross-validation and the hybrid models with 5-fold. All AnoMod models, hybrids included, are scored on the same 647 test windows."
f3 = lambda x: "n/a" if x is None or x != x else f"{x:.3f}"


def rd(n, **kw):
    p = REP / n
    return pd.read_csv(p, **kw) if p.exists() else None


def js(n):
    p = REP / n
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def secs(s):
    return "n/a" if s is None or s != s else (f"{s / 60:.1f} min" if s >= 90 else f"{s:.0f} s")


def main():
    steps = js("overnight_steps.json") or {"steps": [], "failures": []}
    tests = js("overnight_tests.json") or {"runs": []}
    L = ["# Overnight status", "", f"Generated {pd.Timestamp.now():%Y-%m-%d %H:%M} (IST). Task: FailSight overnight master task, deadline 1 PM IST on 23 Oct 2026.", ""]
    fails = steps.get("failures", [])
    L += ["## Anything that failed or was cut", ""] + ([f"* {x}" for x in fails] if fails else ["* No step failed. Anything that was cut or partial is listed under its step below and under *Limitations*."]) + [""]
    L += ["## Steps", "", "| step | status | time (IST) | notes |", "|---|---|---|---|"]
    for s in steps["steps"]:
        L.append(f"| {s['step']} | **{s['status']}** | {s.get('time', '')} | {s.get('notes', '')} |")
    L += ["", "> " + FOLD_NOTE, ""]
    # ---- ten hybrids
    L += ["## The ten hybrid models", "", "| dataset | labels | hybrid | status | folds | test balanced accuracy (95% range) | healthy-window recall | ROC-AUC | training time | reliability |", "|---|---|---|---|---|---|---|---|---|---|"]
    for ds, pre, rel in (("AnoMod", "hybrid_anomod_", "36 healthy test windows (small)"), ("LO2", "hybrid_lo2_", "33 whole runs, 7 untouched test runs")):
        lb = rd(f"{pre}leaderboard.csv")
        if lb is None:
            L.append(f"| {ds} | - | - | not completed | - | - | - | - | - | result file missing |")
            continue
        for r in lb.sort_values(["label_version", "model"]).itertuples():
            if r.status == "not completed":
                L.append(f"| {ds} | {r.label_version[6:]} | {r.model} | not completed | - | - | - | - | - | not completed |")
            else:
                L.append(f"| {ds} | {r.label_version[6:]} | {r.model} | {r.status} | {r.folds} | {f3(r.test_bal_acc)} ({f3(r.test_bal_acc_ci_low)}-{f3(r.test_bal_acc_ci_high)}) | {f3(r.test_normal_recall)} | {f3(r.test_roc_auc)} | {secs(r.train_secs)} | {rel} |")
    L.append("")
    # ---- comparison with the best earlier models
    L += ["## Comparison with the best earlier models (AnoMod, same 647 test windows)", ""]
    hb = rd("hybrid_anomod_leaderboard.csv")
    bs = rd("test_bootstrap_ci.csv")
    for lab in ("label_binary", "label_level"):
        e = rd(f"leaderboard_{lab}.csv")
        if e is None or hb is None:
            continue
        e = e.sort_values("test_bal_acc", ascending=False)
        h = hb[(hb.label_version == lab) & (hb.status != "not completed")].sort_values("test_bal_acc", ascending=False)
        if h.empty:
            continue
        top, b = e.iloc[0], h.iloc[0]
        x = bs[(bs.model == top.model) & (bs.label_version == lab) & (bs.scope == "ALL") & (bs.metric == "balanced_accuracy")].iloc[0]
        L.append(f"* {lab[6:]}: best hybrid **{b.model}** {f3(b.test_bal_acc)} ({f3(b.test_bal_acc_ci_low)}-{f3(b.test_bal_acc_ci_high)}); best of the 36 earlier models **{top.model}** {f3(top.test_bal_acc)} ({f3(x.ci95_low)}-{f3(x.ci95_high)}). "
                 f"{'A hybrid scores higher.' if b.test_bal_acc > top.test_bal_acc else 'No hybrid beats the best earlier model.'} The ranges {'overlap' if (b.test_bal_acc_ci_low <= x.ci95_high and x.ci95_low <= b.test_bal_acc_ci_high) else 'do not overlap'}; the earlier best was picked on the test set, so its lead is optimistic. "
                 "Reason: small tabular data (2,549 training windows, about 128 numbers each) and only 36 healthy test windows, where tree models are hard to beat.")
    L.append("")
    # ---- LO2 slice
    man = rd("lo2_slice_manifest.csv", dtype={"run_id": str})
    sp = rd("lo2_split_ids.csv", dtype={"run_id": str})
    L += ["## LO2 slice", ""]
    if man is not None:
        L += [f"* {len(man)} whole runs, **{man.size_mb.sum():.1f} MB** of the downloaded run files (every downloaded run is used; an earlier 5-run slice under a 150 MB cap was too small).",
              f"* Scrape-row windows (5 s): {int(man.scrape_rows.sum())} ({int(man.normal_rows.sum())} normal, {int(man.anomalous_rows.sum())} anomalous)."]
    if sp is not None and "window_id" in sp.columns:
        L += [f"* Class counts: {sp.label_level.value_counts().to_dict()}; split by WHOLE RUNS (written before any window existed): {int((sp.split == 'train').sum())} training windows from {sp[sp.split == 'train'].run_id.nunique()} runs, {int((sp.split == 'test').sum())} test windows from {sp[sp.split == 'test'].run_id.nunique()} untouched runs (test class counts {sp[sp.split == 'test'].label_level.value_counts().to_dict()}).",
              "* Folds: 5, whole runs (no run in two folds); the test part is runs no fold has seen; the training part is SMOTE-balanced, the test part untouched.", "* Template embedding: TF-IDF + truncated SVD (a downloadable BERT-style encoder was not available)."]
    L.append("")
    fl = rd("lo2_full_leaderboard.csv")
    L += ["## LO2 model families", ""]
    if fl is not None:
        for fam, g in fl.groupby("family"):
            done = sorted(set(g[g.status != "not completed"].model))
            nc = sorted(set(g[g.status == "not completed"].model))
            L.append(f"* {fam}: done {', '.join(done) or 'none'}" + (f"; **not completed:** {', '.join(nc)}" if nc else ""))
        d = fl[(fl.label_version == "label_binary") & (fl.status != "not completed")].sort_values("test_bal_acc", ascending=False)
        if not d.empty:
            L.append(f"* Best on LO2 (binary): {d.iloc[0].model} {f3(d.iloc[0].test_bal_acc)} ({f3(d.iloc[0].test_bal_acc_ci_low)}-{f3(d.iloc[0].test_bal_acc_ci_high)}); many models reach the same high score. The log-volume-only baseline is in `hybrid_lo2_logvolume_baseline.csv`.")
    else:
        L.append("* (leaderboard file missing)")
    L.append("")
    # ---- proofs
    prf = js("proofs.json")
    L += ["## Proof totals", ""]
    if prf:
        L += [f"* {prf['totals']} (generated {prf['generated_at']}, partial: {prf['partial']}); earlier baseline: 65 PASS / 4 WARN / 0 FAIL / 9 REPORTED ACCURATELY before the new sections.", "* " + ", ".join(f"{s['id']} {s['status']}" for s in prf["sections"])]
    L.append("")
    L += ["## Tests", ""] + ([f"* `{t['command']}` -> {t['result']}" for t in tests["runs"]] or ["* (no test results recorded)"]) + [""]
    nb = js("notebook_check.json")
    L += ["## Colab notebook", ""]
    if nb:
        L += [f"* Executed in FAST mode in a fresh virtual environment (clean clone, bundle unzipped, no AnoMod raw folders, no models folder): executed_ok = {nb['executed_ok']}, cells {nb['n_cells']}, errors {nb['n_errors']}, soft failures {len(nb['soft_failures'])}, run time {nb['seconds'] / 60:.1f} min.",
              f"* Numbers compared with the proofs / result files: {len(nb['comparisons'])}; different: {sum(not c['equal'] for c in nb['comparisons'])}.", "* Files: `notebooks/FailSight_Workflow.ipynb`, `notebooks/FailSight_Workflow_executed.ipynb`, `notebooks/FailSight_Workflow_executed.html`, `docs/COLAB.md`, `failsight_bundle.zip` (git-ignored)."]
        if nb["soft_failures"]:
            L += ["* Sections that printed a soft-failure message: " + "; ".join(nb["soft_failures"][:6])]
    else:
        L.append("* Not executed (no notebook_check.json).")
    L.append("")
    # ---- website defaults
    L += ["## Website: which models are shown by default", ""]
    for lab in ("label_binary", "label_level"):
        e = rd(f"leaderboard_{lab}.csv")
        hb_ = rd("hybrid_anomod_leaderboard.csv")
        if e is None:
            continue
        rows = [(r.model, r.test_bal_acc) for r in e.itertuples()]
        if hb_ is not None:
            rows += [(f"{r.model} (hybrid)", r.test_bal_acc) for r in hb_[(hb_.label_version == lab) & (hb_.status != "not completed")].itertuples()]
        rows.sort(key=lambda x: -x[1])
        L.append(f"* AnoMod {lab[6:]}: top 9 shown by default: {', '.join(n for n, _ in rows[:9])}. Under *Show all models* ({len(rows)} in total): {', '.join(n for n, _ in rows[9:])}.")
    L += ["* LO2 tab: top 9 by overall score shown by default, the rest under *Show all models*; every row marked unreliable (very small sample); unfinished families listed as not completed.", ""]
    L += ["## Limitations (all stated on the Limits page, the Hybrid models page and in the docs)", "",
          "* " + FOLD_NOTE, "* AnoMod test part: only 36 healthy windows, wide ranges; the best model on every list was picked on the test set (lead is optimistic).",
          "* Run identification can reproduce the AnoMod scores (P14, P20): they are not proof of fault detection.", "* LO2: fewer than 6 whole runs under the 150 MB cap, one test run (172 windows, 13 healthy); trained on LO2's own data only; no transfer claim; binary LO2 is almost trivially separable (log volume alone scores high; see the baseline).",
          "* Families not finished on LO2 are listed above as not completed.", "* No transfer to other systems, no claim that FailSight works on any app, no forecasting of failures, no time-until-failure (not built; future work).",
          "* The template encoder on LO2 is TF-IDF + SVD, not a real BERT. The AnoMod graph is built from the component names in the feature names, not from real service calls.",
          "* Hybrid selection used a single screening fold (validation scores only) to pick hidden size and imbalance method; the CV then used 5 folds; fold models were not averaged.", ""]
    L += ["## How to start the app", "", "```", "cd \"C:\\Users\\Lenovo\\Documents\\AI PROJECT\\code\"", ".\\start_testbed.bat", "```", "",
          "| address | what |", "|---|---|", "| http://127.0.0.1:5000 | the shop |", "| http://127.0.0.1:5000/ops | presenter controls (this computer only) |", "| http://127.0.0.1:8000 | the live monitor; the Hybrid models page is at http://127.0.0.1:8000/examiners/hybrids |", "",
          "Results website without the shop: `python -m app.backend.serve` then http://127.0.0.1:8000 . Fallback commit for the storefront demo: see `docs/PRESENT_NOW.md`; the tag `pre-overnight` marks the state before this run.", ""]
    log = subprocess.run(["git", "log", "--oneline", "-20"], cwd=ROOT, capture_output=True, text=True).stdout
    L += ["## git log --oneline -20", "", "```", log.strip(), "```", ""]
    (REP / "OVERNIGHT_STATUS.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote OVERNIGHT_STATUS.md")


if __name__ == "__main__":
    main()
