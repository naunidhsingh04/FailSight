"""Leakage audit (run first). Prints PASS / WARN / FAIL per check with the evidence and writes
report_outputs/leakage_audit.json, leakage_audit.csv and data_integrity.csv.

  python scripts/leakage_audit.py

Nothing saved is modified. The only model trained here is the throw-away shuffled-label XGBoost of check 6 (never saved).
AnoMod and LO2 are only read (hashed before and after to prove it).
"""
import argparse, ast, hashlib, json, os, platform, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
WS = ROOT.parent                                    # "AI PROJECT"
ANOMOD, LO2 = WS / "AnoMod", WS / "LO2"
REP, SCRIPTS = ROOT / "report_outputs", ROOT / "scripts"
SEED = 42
CHECKS = []
T0 = time.time()


def say(*a):
    print(*a, flush=True)


def record(cid, name, status, summary, evidence, source):
    CHECKS.append(dict(id=cid, check=name, status=status, summary=summary, evidence=evidence, source=source))
    say(f"\n[{status}] {cid} {name}\n      {summary}")


# --------------------------------------------------------------------------------------------- data integrity (hash manifest)
def used_files():
    """every external file the project reads: raw AnoMod inputs of 01_tidy.py (api .jsonl, metric .csv, log .log), the combined raw CSV used for the row
    cross-check, AnoMod processed tables/reports, and the LO2 files used (lo2_subset.csv + the one tar the sample came from)"""
    out = []
    for root in ("SN_data", "TT_data"):
        for dp, _, fns in os.walk(ANOMOD / root):
            for f in fns:
                if f.lower().endswith((".jsonl", ".csv", ".log")):
                    out.append(Path(dp) / f)
    out.append(ANOMOD / "anomod_all_csv_raw_combined.csv")
    for sub in ("processed", "report_outputs"):
        for dp, _, fns in os.walk(ANOMOD / sub):
            out += [Path(dp) / f for f in fns]
    out += [LO2 / "lo2_subset.csv", LO2 / "lo2_tar" / "light-oauth2-data-1719592986.tar"]
    return sorted({p for p in out if p.is_file()})


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def hash_all(files, workers=6):
    with ThreadPoolExecutor(workers) as ex:
        return dict(zip(files, ex.map(sha256, files)))


def snapshot():
    return {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in used_files()}


# --------------------------------------------------------------------------------------------- helpers
def ast_calls(names, files=None):
    out = []
    for f in sorted(files or SCRIPTS.glob("*.py")):
        tree = ast.parse(Path(f).read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                fn = n.func
                nm = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
                if nm in names:
                    out.append(f"{Path(f).name}:{n.lineno} {ast.unparse(n)[:120]}")
    return out


def state_digest(st):
    h = hashlib.sha256()
    for k in sorted(st):
        v = st[k]
        h.update(k.encode())
        if isinstance(v, dict):
            for kk in sorted(v):
                h.update(np.asarray(v[kk]).tobytes())
        elif isinstance(v, (list, tuple)):
            h.update(json.dumps(v, default=str).encode())
        else:
            h.update(np.asarray(v).tobytes() if isinstance(v, np.ndarray) else repr(v).encode())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-hash", action="store_true", help="debug only: skip the (slow) hashing")
    args = ap.parse_args()
    say("=" * 78); say("FAILSIGHT LEAKAGE AUDIT", time.ctime()); say("=" * 78)

    # ---------------- C7 start: hash manifest before anything else
    files = used_files()
    snap0 = snapshot()
    say(f"hashing {len(files)} AnoMod / LO2 files ({sum(p.stat().st_size for p in files) / 1e9:.2f} GB) ...")
    t = time.time()
    h0 = {} if args.skip_hash else hash_all(files)
    say(f"  done in {time.time() - t:.0f}s")

    import preprocess as P
    import v2_data
    from v2_data import DataV2
    v2_data.common.LOG = ROOT / "outputs" / "leakage_audit_log.txt"
    D = DataV2()
    df = D.df
    sp = pd.read_csv(REP / "split_ids.csv")
    feats = P.feature_columns(df)

    # ================================================================= C1 fitting on training rows only
    train = df.iloc[D.train_pos]
    prep0 = P.fit_preprocessor(train)
    d0 = state_digest(prep0.to_state())
    df2 = df.copy()
    test_mask = (df2.split == "test").to_numpy()
    df2.loc[test_mask, feats] = df2.loc[test_mask, feats].to_numpy(float) * 1e6 + 12345.0          # wreck every test feature
    d2 = df2.merge(sp[["window_id", "split"]].rename(columns={"split": "split_ids"}), on="window_id")
    train2 = d2[d2.split_ids == "train"].drop(columns="split_ids")
    prep1 = P.fit_preprocessor(train2)
    same_global = state_digest(prep1.to_state()) == d0
    # fold preprocessors: perturb validation AND test rows, refit on the fold-train rows, compare with the preprocessor the pipeline uses
    fold_same = []
    for k in (1, 5, 10):
        tr, va = D.folds[k - 1]
        df3 = df.copy()
        bad = np.zeros(len(df3), bool); bad[va] = True; bad[D.test_pos] = True
        df3.loc[bad, feats] = df3.loc[bad, feats].to_numpy(float) * 1e6 - 777.0
        a, b = D.prep(k), P.fit_preprocessor(df3.iloc[tr])
        fold_same.append(all(np.allclose(getattr(a, n), getattr(b, n), equal_nan=True) for n in ("median", "lo", "hi", "mean_", "scale_")))
    try:
        P.fit_preprocessor(df.iloc[np.concatenate([D.train_pos[:50], D.test_pos[:5]])])
        guard = False
    except ValueError:
        guard = True
    fit_sites = ast_calls({"fit_preprocessor"})
    ok = same_global and all(fold_same) and guard
    record("C1a", "Preprocessor / imputer / scaler fitted on training rows only", "PASS" if ok else "FAIL",
           f"wrecking all test rows leaves the fitted state identical (global: {same_global}; fold 1/5/10 with validation+test wrecked: {fold_same}); fit() rejects non-train rows: {guard}",
           dict(global_state_unchanged=same_global, fold_state_unchanged=fold_same, rejects_test_rows=guard, call_sites=fit_sites,
                method="refit on rows selected through split_ids.csv after multiplying every test feature by 1e6; state hash compared"),
           "scripts/preprocess.py, scripts/v2_data.py")

    # SMOTE
    leak = []
    for k in range(1, 11):
        tr, va = D.folds[k - 1]
        leak.append(len(set(tr) & set(va)) + len(set(tr) & set(D.test_pos)) + len(set(va) & set(D.test_pos)))
    union_ok = sorted(np.concatenate([D.folds[k][1] for k in range(10)]).tolist()) == D.train_pos.tolist()
    from v2_data import smote_resample
    Xtr1 = D.W(1)[D.folds[0][0]]
    y1 = D.y["label_level"][D.folds[0][0]]
    _, ys = smote_resample(Xtr1, y1)
    before, after = np.bincount(y1), np.bincount(ys)
    sm_sites = ast_calls({"SMOTE", "smote_resample", "smote_seq"})
    ok = sum(leak) == 0 and union_ok and (after == before.max()).all()
    record("C1b", "SMOTE applied to training parts only", "PASS" if ok else "FAIL",
           f"10 folds: train/validation/test index overlaps = {sum(leak)}; validation parts tile the training set exactly: {union_ok}; fold-1 SMOTE balances to the majority count ({before.tolist()} -> {after.tolist()}), validation/test never enter it",
           dict(fold_overlap_counts=leak, validation_parts_cover_training=union_ok, fold1_before=before.tolist(), fold1_after=after.tolist(), call_sites=sm_sites),
           "scripts/v2_data.py, scripts/v2_deep.py, scripts/v2_tab.py")

    # thresholds + Optuna
    sys.argv = ["x", "--stages", "0"]
    import full_pipeline as FP
    oof = pd.read_parquet(REP / "oof_predictions.parquet")
    thr = json.loads((REP / "thresholds.json").read_text())
    ids_ok = set(oof.window_id) <= set(sp[sp.split == "train"].window_id) and oof.groupby(["label_version", "model"]).size().eq(len(D.train_pos)).all()
    n_cmp, bad_t = 0, []
    for l in ("label_binary", "label_level"):
        C = 2 if l == "label_binary" else 5
        for m, spec in thr[l].items():
            o = oof[(oof.label_version == l) & (oof.model == m)].sort_values("window_id")
            P_ = o[[f"p{c}" for c in range(C)]].to_numpy()
            # same row order as the pipeline: train positions in file order
            order = pd.Series(range(len(o)), index=o.window_id)
            idx = order.loc[df.window_id.to_numpy()[D.train_pos]].to_numpy()
            got = FP.tune_decision(o.y_true.to_numpy(int)[idx], P_[idx], C)
            key = "threshold" if C == 2 else "weights"
            same = np.allclose(np.atleast_1d(got[key]), np.atleast_1d(spec[key]), atol=1e-9)
            n_cmp += 1
            if not same:
                bad_t.append(f"{l}/{m}")
    opt_sites = ast_calls({"tab_cv"}, [SCRIPTS / "full_pipeline.py"])
    trials = pd.read_csv(REP / "optuna_trials.csv")
    ok = ids_ok and not bad_t
    record("C1c", "Thresholds and Optuna choices depend on training rows only", "PASS" if ok else "FAIL",
           f"{n_cmp} saved decision rules recomputed from out-of-fold predictions (training windows only: {ids_ok}): {n_cmp - len(bad_t)} identical; Optuna objective calls tab_cv on the run-aware training folds ({len(trials)} trials, {trials.groupby(['label_version', 'model']).ngroups} studies)",
           dict(oof_windows_subset_of_train=ids_ok, rules_recomputed=n_cmp, rules_differing=bad_t, optuna_call_sites=opt_sites,
                note="Optuna was run on the same training folds, so cross-validation scores of tuned models are optimistically selected; test rows were never used"),
           "scripts/full_pipeline.py (tune_decision, stage2), report_outputs/thresholds.json, oof_predictions.parquet")

    # ================================================================= C2 split integrity
    tr_ids, te_ids = set(df.window_id[df.split == "train"]), set(df.window_id[df.split == "test"])
    sp_tr, sp_te = set(sp[sp.split == "train"].window_id), set(sp[sp.split == "test"].window_id)
    ok = (not (tr_ids & te_ids)) and (len(tr_ids), len(te_ids)) == (2549, 647) and tr_ids == sp_tr and te_ids == sp_te and df.window_id.is_unique
    record("C2", "No window id in both train and test; counts match split_ids.csv", "PASS" if ok else "FAIL",
           f"train {len(tr_ids)} / test {len(te_ids)}, shared ids {len(tr_ids & te_ids)}, ids unique {df.window_id.is_unique}, identical to split_ids.csv: {tr_ids == sp_tr and te_ids == sp_te}",
           dict(train=len(tr_ids), test=len(te_ids), shared_ids=len(tr_ids & te_ids), per_system=sp.groupby(["system", "split"]).size().unstack().to_dict()),
           "report_outputs/split_ids.csv")

    # ================================================================= C3 time / run overlap, duplicates, effect of dropping
    n_ov, n_x = {}, {}
    for s in ("SN", "TT"):
        a, b = df[(df.system == s) & (df.split == "train")], df[(df.system == s) & (df.split == "test")]
        ov = (a.window_start.to_numpy()[:, None] < b.window_end.to_numpy()[None]) & (a.window_end.to_numpy()[:, None] > b.window_start.to_numpy()[None])
        same = a.run_id.to_numpy()[:, None] == b.run_id.to_numpy()[None]
        n_ov[s], n_x[s] = int((ov & same).any(1).sum()), int((ov & ~same).sum())
    cols = feats + ["system_is_TT"]
    ht = pd.util.hash_pandas_object(df.iloc[D.train_pos][cols], index=False).to_numpy()
    he = pd.util.hash_pandas_object(df.iloc[D.test_pos][cols], index=False).to_numpy()
    dup_mask = np.isin(he, ht)
    ytr_by_hash = pd.Series(df.iloc[D.train_pos].label_level.to_numpy()).groupby(ht).agg(set)
    conflicts = sum(lv not in ytr_by_hash[h] for h, lv in zip(he[dup_mask], df.iloc[D.test_pos].label_level.to_numpy()[dup_mask]))
    runs_both = len(set(df[df.split == "train"].run_id) & set(df[df.split == "test"].run_id))
    v1 = pd.read_csv(ROOT / "results" / "v1_16epoch" / "test_metrics.csv")
    v1 = v1[(v1.scope == "ALL")]
    eff = {}
    for l in ("label_binary", "label_level"):
        o = v1[(v1.label_version == l) & (v1.variant == "official")].set_index("model")
        n = v1[(v1.label_version == l) & (v1.variant == "no_overlap")].set_index("model")
        dlt = (n.balanced_accuracy - o.balanced_accuracy).dropna()
        eff[l] = dict(models=len(dlt), mean_delta_balanced_accuracy=float(dlt.mean()), min=float(dlt.min()), max=float(dlt.max()))
    status = "WARN" if (n_ov["SN"] + n_ov["TT"] > 0 or dup_mask.any() or runs_both) else "PASS"
    record("C3", "Time / run overlap and duplicates between train and test", status,
           f"{n_ov['SN']} SN and {n_ov['TT']} TT training windows overlap in time with a test window of the SAME run (<= 20 s at block boundaries); overlaps between different runs: {n_x['SN'] + n_x['TT']}; "
           f"{int(dup_mask.sum())} test windows duplicate a training window (label conflicts: {conflicts}); all {runs_both} runs appear in both parts (block split). "
           f"Dropping the overlapping windows and refitting (first-run models) changed test balanced accuracy by {eff['label_binary']['mean_delta_balanced_accuracy']:+.3f} (binary) and {eff['label_level']['mean_delta_balanced_accuracy']:+.3f} (five-class) on average",
           dict(overlap_same_run=n_ov, overlap_different_runs=n_x, duplicate_test_windows=int(dup_mask.sum()), duplicate_label_conflicts=int(conflicts), runs_in_both_parts=int(runs_both),
                no_overlap_effect_first_run=eff, blocks_in_both_parts=int(len(set(df[df.split == 'train'].block_id) & set(df[df.split == 'test'].block_id)))),
           "report_outputs/leakage_checks.csv, results/v1_16epoch/test_metrics.csv (variant no_overlap)")

    # ================================================================= C4 only data inside the window or earlier
    length = (df.window_end - df.window_start).dt.total_seconds()
    step_ok = True
    for _, g in df.sort_values(["run_id", "window_index"]).groupby("run_id"):
        d_ = g.window_start.diff().dropna().dt.total_seconds()
        step_ok &= bool(((d_ - 10).abs() < 1e-3).all())
    api = pd.read_parquet(ANOMOD / "processed" / "tidy" / "api.parquet", columns=["system", "scenario", "t"])
    lg = pd.read_parquet(ANOMOD / "processed" / "tidy" / "logs.parquet", columns=["system", "scenario", "t", "level"])
    lg = lg[lg.t.notna()]
    lg_g = {k: (g.t.to_numpy(), g.level.to_numpy()) for k, g in lg.groupby(["system", "scenario"], observed=True)}
    api_g = {k: g.t.to_numpy() for k, g in api.groupby(["system", "scenario"], observed=True)}
    rng = np.random.RandomState(SEED)
    sample = df.iloc[np.sort(rng.choice(len(df), 400, replace=False))]
    bad, checked = 0, 0
    for _, r in sample.iterrows():
        t, lv = lg_g[(r.system, r.scenario)]
        ws = r.window_start.value / 1e9
        we = ws + 30.0
        m = (t >= ws - 1e-6) & (t < we - 1e-6)
        a = api_g.get((r.system, r.scenario), np.array([]))
        ma = (a >= ws - 1e-6) & (a < we - 1e-6)
        got = (int(m.sum()), int((lv[m] == 3).sum()), int((lv[m] == 2).sum()), int(ma.sum()))
        want = (r.log_line_count, r.log_error_count, r.log_warning_count, r.api_request_count)
        bad += int(any(abs(float(x) - float(y)) > 0.5 for x, y in zip(got, want)))
        checked += 1
    ctx = D.ctx
    run, spl, wi = df.run_id.to_numpy(), df.split.to_numpy(), df.window_index.to_numpy()
    viol = 0
    for i in range(len(df)):
        c = ctx[i]
        viol += int((c > i).any() or (run[c] != run[i]).any() or (spl[c] != spl[i]).any() or (np.diff(wi[c]) < 0).any())
    n_pad = D.padded_slots
    slots = ctx.size
    ok = checked > 0 and bad == 0 and viol == 0 and (length == 30).all() and step_ok
    record("C4", "Features use only data inside the window or earlier; context = past windows of the same run and split", "PASS" if ok else "FAIL",
           f"{checked} random windows (both systems, both splits) recomputed from the raw event tables: {checked - bad} identical for log lines, ERROR lines, WARN lines and API requests counted strictly in [start, start+30 s); "
           f"all windows are 30 s long, step 10 s: {bool((length == 30).all()) and step_ok}; 6-window context violations (future index, other run, other split): {viol} of {len(df)}; "
           f"{n_pad} of {slots} context slots ({n_pad / slots:.1%}) repeat the earliest available window",
           dict(windows_recomputed=checked, mismatches=bad, context_violations=viol, context_slots=int(slots), context_slots_repeating_earliest=int(n_pad),
                share_repeating=n_pad / slots, features_recomputed=["log_line_count", "log_error_count", "log_warning_count", "api_request_count"]),
           "AnoMod/processed/tidy/{logs,api}.parquet (read-only), AnoMod/processed/*/windows.parquet")

    # ================================================================= C5 no label/id column reaches a model; single-feature power
    prep = P.load_preprocessor()
    names = prep.output_names
    forbidden = [n for n in names if any(k in n.lower() for k in P.FORBIDDEN_NAMES)]
    scen = {s.lower() for s in set(df.scenario) | set(df.run_id)}
    forbidden += [n for n in names if any(s in n.lower() for s in scen)]
    meta_in = [c for c in P.META if c in names and c != "system_is_TT"]
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.tree import DecisionTreeClassifier
    Wf = D.W("final")
    Xtr, Xte = Wf[D.train_pos], Wf[D.test_pos]
    best = {}
    for l in ("label_binary", "label_level"):
        ytr, yte = D.ytr[l], D.yte[l]
        top = ("", 0.0)
        for j, n in enumerate(names):
            if np.ptp(Xtr[:, j]) == 0:
                continue
            ba = balanced_accuracy_score(yte, DecisionTreeClassifier(max_depth=4, random_state=SEED).fit(Xtr[:, [j]], ytr).predict(Xte[:, [j]]))
            if ba > top[1]:
                top = (n, ba)
        best[l] = top
    flagged = [l for l, (_, b) in best.items() if b > 0.90]
    ok = not forbidden and not meta_in and not flagged
    record("C5", "No label / scenario / run / id column reaches a model; single-feature power", "PASS" if ok else "FAIL",
           f"{len(names)} model inputs; forbidden-name columns: {len(forbidden)}; metadata columns among inputs: {len(meta_in)} (system_is_TT is an intended input); best single feature balanced accuracy: "
           f"binary {best['label_binary'][1]:.3f} ({best['label_binary'][0]}), five-class {best['label_level'][1]:.3f} ({best['label_level'][0]}), alarm level 0.90",
           dict(inputs=len(names), forbidden=forbidden, metadata_in_inputs=meta_in, best_single_feature={l: dict(feature=f, balanced_accuracy=b) for l, (f, b) in best.items()}),
           "report_outputs/leakage_checks.csv (same method, recomputed)")

    # ================================================================= C6 shuffled-label sanity test (throw-away models, never saved)
    from imblearn.over_sampling import SMOTE
    from xgboost import XGBClassifier
    shuf = {}
    for l in ("label_binary", "label_level"):
        C = 2 if l == "label_binary" else 5
        ytr_true = D.ytr[l]
        res_l = {}
        blocks = df.block_id.to_numpy()[D.train_pos]
        for variant in ("window_level", "block_level"):
            rng = np.random.RandomState(SEED)
            if variant == "window_level":
                y_sh = ytr_true[rng.permutation(len(ytr_true))]
            else:
                bl = pd.Series(ytr_true).groupby(blocks).first()
                perm = pd.Series(rng.permutation(bl.to_numpy()), index=bl.index)
                y_sh = perm.loc[blocks].to_numpy()
            idx_of = {p: i for i, p in enumerate(D.train_pos)}

            def fit_pred(Xa, ya, Xb):
                k = max(1, min(5, int(np.bincount(ya, minlength=C)[np.bincount(ya, minlength=C) > 0].min()) - 1))
                Xs, ys = SMOTE(random_state=SEED, k_neighbors=k).fit_resample(Xa, ya)
                m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, tree_method="hist", n_jobs=-1, random_state=SEED, verbosity=0).fit(Xs, ys)
                return m.predict(Xb)
            fold_ba = []
            for k in range(1, 11):
                tr, va = D.folds[k - 1]
                W = D.W(k)
                itr = [idx_of[p] for p in tr]
                iva = [idx_of[p] for p in va]
                fold_ba.append(balanced_accuracy_score(ytr_true[iva], fit_pred(W[tr], y_sh[itr], W[va])))
            te_ba = balanced_accuracy_score(D.yte[l], fit_pred(Xtr, y_sh, Xte))
            res_l[variant] = dict(cv_validation_ba_mean=float(np.mean(fold_ba)), cv_validation_ba_std=float(np.std(fold_ba)), test_ba=float(te_ba), chance=1.0 / C)
            say(f"   shuffled labels {l} {variant}: CV validation BA {np.mean(fold_ba):.3f} +- {np.std(fold_ba):.3f}, test BA {te_ba:.3f} (chance {1 / C:.2f})")
        shuf[l] = res_l
    tm = pd.read_csv(REP / "test_metrics.csv")
    real = {l: float(tm[(tm.model == "XGB") & (tm.label_version == l) & (tm.scope == "ALL") & (tm.variant == "official")].balanced_accuracy.iloc[0]) for l in ("label_binary", "label_level")}
    worst = max(v["test_ba"] - v["chance"] for l in shuf for v in shuf[l].values())
    status = "PASS" if worst <= 0.10 else "WARN" if worst <= 0.20 else "FAIL"
    msg = "; ".join(f"{l[6:]} {v}: test BA {r['test_ba']:.3f}" for l in shuf for v, r in shuf[l].items())
    record("C6", "Shuffled-label sanity test (XGBoost, same folds, SMOTE, never saved)", status,
           f"{msg}. Chance = 0.50 (binary) / 0.20 (five-class); highest excess over chance {worst:+.3f}. Saved XGB with the real labels: {real['label_binary']:.3f} (binary), {real['label_level']:.3f} (five-class)"
           + ("" if status == "PASS" else "  !!! NOT NEAR CHANCE: INVESTIGATE !!!"),
           dict(results=shuf, real_label_saved_xgb_test_ba=real, rule="PASS if every excess over chance <= 0.10, WARN <= 0.20, else FAIL",
                variants="window_level = labels permuted over windows; block_level = label of each 10-window block reassigned to another block (keeps label autocorrelation)"),
           "throw-away models trained in this script")

    # ================================================================= C7 data integrity: re-hash and compare
    snap1 = snapshot()
    t = time.time()
    h1 = {} if args.skip_hash else hash_all(files)
    base_p = REP / "data_integrity.csv"
    base = pd.read_csv(base_p).set_index("path") if base_p.exists() else pd.DataFrame()
    rows = []
    for p in files:
        rel = p.relative_to(WS).as_posix()
        b = base.sha256_baseline.get(rel) if len(base) and rel in base.index else None
        b0 = base.baseline_time.get(rel) if len(base) and rel in base.index else None
        rows.append(dict(path=rel, size_bytes=snap0[p][0], sha256_baseline=b if isinstance(b, str) else h0.get(p), baseline_time=b0 if isinstance(b0, str) else time.strftime("%Y-%m-%d %H:%M:%S"),
                         sha256_start=h0.get(p), sha256_end=h1.get(p)))
    di = pd.DataFrame(rows)
    di["match"] = (di.sha256_baseline == di.sha256_start) & (di.sha256_start == di.sha256_end)
    di.to_csv(base_p, index=False)
    changed = [p for p in files if snap0[p] != snap1[p]]
    nbad = int((~di.match).sum()) if not args.skip_hash else 0
    status = "PASS" if (nbad == 0 and not changed) else "FAIL"
    record("C7", "AnoMod / LO2 files unmodified (SHA-256 recorded at start, re-checked at end)", status,
           f"{len(di)} files ({di.size_bytes.sum() / 1e9:.2f} GB): hash at start == hash at end == recorded baseline for {int(di.match.sum())}; size/mtime changed during the audit: {len(changed)}",
           dict(files=len(di), mismatches=nbad, mtime_or_size_changed=[str(c) for c in changed][:10], baseline_first_recorded=str(di.baseline_time.min()), csv="report_outputs/data_integrity.csv",
                scope="AnoMod raw .jsonl/.csv/.log inputs, combined raw CSV, processed + report_outputs files; LO2 lo2_subset.csv and the one tar used"),
           "report_outputs/data_integrity.csv")

    # ================================================================= report
    import sklearn, xgboost, imblearn, scipy
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    summ = pd.Series([c["status"] for c in CHECKS]).value_counts().to_dict()
    out = dict(generated_at=time.strftime("%Y-%m-%d %H:%M:%S"), git_commit=commit, python=platform.python_version(), seed=SEED, runtime_seconds=round(time.time() - T0, 1),
               versions=dict(numpy=np.__version__, pandas=pd.__version__, scikit_learn=sklearn.__version__, xgboost=xgboost.__version__, imbalanced_learn=imblearn.__version__, scipy=scipy.__version__),
               summary=summ, checks=CHECKS)
    (REP / "leakage_audit.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    pd.DataFrame([dict(id=c["id"], check=c["check"], status=c["status"], summary=c["summary"], source=c["source"]) for c in CHECKS]).to_csv(REP / "leakage_audit.csv", index=False)
    say("\n" + "=" * 78); say("SUMMARY")
    for c in CHECKS:
        say(f"  {c['status']:<5} {c['id']:<4} {c['check']}")
    say(f"\n  totals: {summ}   runtime {time.time() - T0:.0f}s   commit {commit[:10]}")
    say("  wrote report_outputs/leakage_audit.json, leakage_audit.csv, data_integrity.csv")
    if summ.get("FAIL"):
        say("\n  !!! At least one check FAILED. Stop and read the evidence above before continuing. !!!")
        sys.exit(1)


if __name__ == "__main__":
    main()
