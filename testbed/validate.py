"""Run-held-out validation of fault detection on the testbed dataset. Whole runs are split, never windows: held-out TEST runs (chosen with a fixed seed so that every fault
type appears at least once and at least 2 runs are fault-free) and TRAIN runs (run-aware folds for model selection). Nothing is tuned on the test runs.

  models      logistic regression, random forest, XGBoost, LightGBM (class weights instead of SMOTE: the imbalance is 3.7 : 1, see docs/TESTBED_DATASET.md) on raw features,
              baseline-relative features or both (chosen per model by run-grouped cross-validation on the TRAIN runs only), and two unsupervised per-run detectors that learn each
              run's normal from its first 150 s only: a robust z-score and an IsolationForest (their alert thresholds are chosen on the TRAIN runs)
  scored      windows after the learning period (window end > 150 s) that are not transition windows; every method is evaluated on exactly these windows
  metrics     balanced accuracy, macro-F1, Normal recall, ROC-AUC with run-bootstrap 95% CIs; recall by fault type and by severity level; detection delay (seconds from the recorded fault
              start to the first alert), false alarms per hour on fault-free runs, lead time of the warning before the failure of gradual faults; shuffled-label control; run identification

python -m testbed.validate [--boot 1000]"""
import argparse
import json
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from testbed import dataset as DS  # noqa: E402

REP = ROOT / "report_outputs"
SPLIT_SEED = 20261021
N_TEST_RUNS = 12
N_FOLDS = 7
MODELS = ["LR", "RF", "XGB", "LGBM"]
FEATURE_SETS = ["raw", "rel", "both"]
CORE = ["api_request_count", "api_5xx_count", "api_error_rate", "api_latency_p95_ms", "log_line_count", "log_error_count", "m_process_cpu_percent__mean", "m_process_rss_bytes__mean",
        "m_process_threads__mean", "m_event_loop_lag_ms__mean", "m_event_loop_lag_ms__max", "m_db_pool_wait_ms__mean", "m_inflight_requests__mean", "m_process_rss_bytes__slope"]
RELIABLE_MIN_INSTANCES = 5          # a per-type / per-severity number is called reliable only with at least this many fault instances in the evaluated runs


# ------------------------------------------------------------------------------------------------ data and split
def load():
    W = pd.read_parquet(DS.OUT)
    base, rel = DS.feature_columns(W)
    W["fault_free_run"] = W.groupby("run_id").label_binary.transform("max") == 0
    return W, base, rel


def run_faults(W):
    """fault instances per run from the recorded schedules"""
    rows = []
    for d in sorted(DS.RUNS.glob("r*")):
        if (d / "DONE").exists():
            s = pd.read_csv(d / "fault_schedule.csv", dtype={"fault_id": str})
            rows.append(s)
    return pd.concat(rows, ignore_index=True)


def assign_runs(W, S):
    """test runs: a fixed-seed draw of N_TEST_RUNS runs in which every fault type appears at least once and at least 2 runs are fault-free; folds for the rest"""
    runs = sorted(W.run_id.unique())
    types = sorted(S.fault_type.unique())
    free = set(W[W.fault_free_run].run_id)
    by_run = S.groupby("run_id").fault_type.apply(set).to_dict()
    rng = random.Random(SPLIT_SEED)
    for attempt in range(1, 200000):
        test = rng.sample(runs, N_TEST_RUNS)
        have = set().union(*[by_run.get(r, set()) for r in test])
        if have >= set(types) and sum(r in free for r in test) >= 2:
            break
    else:
        raise RuntimeError("no valid test-run draw found")
    train = [r for r in runs if r not in test]
    order = list(train)
    random.Random(SPLIT_SEED + 1).shuffle(order)
    # run-aware folds: greedy balance of anomalous windows per fold, whole runs only
    anom = W.groupby("run_id").label_binary.sum().to_dict()
    folds = {r: -1 for r in train}
    load_ = [0] * N_FOLDS
    count = [0] * N_FOLDS
    for r in sorted(order, key=lambda r: -anom[r]):
        k = min(range(N_FOLDS), key=lambda i: (count[i] >= -(-len(train) // N_FOLDS), load_[i], count[i]))
        folds[r] = k
        load_[k] += anom[r]
        count[k] += 1
    asg = pd.DataFrame([dict(run_id=r, split="test" if r in test else "train", fold=("" if r in test else folds[r]), n_faults=len(S[S.run_id == r]),
                             faults="; ".join(f"{x.fault_type}[{x.severity_level}]" for x in S[S.run_id == r].itertuples()) or "none (fault-free)") for r in runs])
    return asg, attempt


# ------------------------------------------------------------------------------------------------ supervised models
def make_X(W, base, rel, fs):
    if fs == "raw":
        return W[base].to_numpy(float)
    R = np.nan_to_num(W[rel].to_numpy(float), nan=0.0)               # no baseline yet (first windows) = no deviation
    return R if fs == "rel" else np.hstack([W[base].to_numpy(float), R])


def make_model(name, ytr):
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    if name == "LR":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced"))
    if name == "RF":
        return make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=42))
    if name == "XGB":
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw, tree_method="hist", n_jobs=-1, random_state=42, verbosity=0)
    if name == "LGBM":
        return LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, class_weight="balanced", n_jobs=-1, random_state=42, verbose=-1)
    raise ValueError(name)


def fit_proba(name, Xtr, ytr, Xte):
    m = make_model(name, ytr).fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def bal_acc(y, pred):
    r1 = (pred[y == 1] == 1).mean() if (y == 1).any() else np.nan
    r0 = (pred[y == 0] == 0).mean() if (y == 0).any() else np.nan
    return float(np.nanmean([r0, r1]))


def best_threshold(y, p):
    grid = np.unique(np.round(np.quantile(p, np.linspace(0.02, 0.98, 97)), 6))
    scores = [bal_acc(y, (p >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(scores))])


# ------------------------------------------------------------------------------------------------ unsupervised per-run detectors
def learn_stats(Xl):
    med = np.nanmedian(Xl, axis=0)
    mad = np.nanmedian(np.abs(Xl - med), axis=0)
    return med, np.maximum(1.4826 * mad, np.maximum(0.05 * np.abs(med), 1e-9))


def robust_z_scores(Xr, learn):
    med, sc = learn_stats(Xr[learn])
    z = np.clip(np.abs((Xr - med) / sc), 0, 10)
    z = np.nan_to_num(z, nan=0.0)
    s = np.sort(z, axis=1)
    return s[:, -2:].mean(axis=1)                                # mean of the two largest deviations: one noisy feature alone is not enough


def iforest_scores(Xr, learn):
    from sklearn.ensemble import IsolationForest
    med, sc = learn_stats(Xr[learn])
    Z = np.clip(np.nan_to_num((Xr - med) / sc, nan=0.0), -10, 10)
    m = IsolationForest(n_estimators=200, random_state=42, contamination="auto").fit(Z[learn])
    return -m.score_samples(Z)


def unsup_scores(W, core, kind):
    out = np.full(len(W), np.nan)
    for run, idx in W.groupby("run_id").indices.items():
        idx = np.sort(idx)
        Xr = W.loc[W.index[idx], core].to_numpy(float)
        learn = W.loc[W.index[idx], "in_learning"].to_numpy() == 1
        out[idx] = robust_z_scores(Xr, learn) if kind == "RobustZ" else iforest_scores(Xr, learn)
    return out


# ------------------------------------------------------------------------------------------------ alerts, delay, false alarms, lead time
def alert_series(flags, rule):
    f = np.asarray(flags, bool)
    if rule == "1 window":
        return f
    a = np.zeros(len(f), bool)
    for k in range(len(f)):
        a[k] = f[max(0, k - 2):k + 1].sum() >= 2
    return a


def episodes(alert, gap=3):
    """start indices of alert episodes (a new episode starts after `gap` quiet windows)"""
    starts, quiet = [], gap
    for k, a in enumerate(alert):
        if a and quiet >= gap:
            starts.append(k)
        quiet = 0 if a else quiet + 1
    return starts


def detection_table(D, flags, S, method, rule, runs):
    """D: scored windows (all columns), flags aligned with D. Returns per-instance rows (delay, lead) and a false-alarm summary for the given runs"""
    inst, fa_free, fa_other = [], [0, 0.0], [0, 0.0]
    for run in runs:
        idx = np.where(D.run_id.to_numpy() == run)[0]
        idx = idx[np.argsort(D.window_index.to_numpy()[idx])]
        we = D.window_end.to_numpy()[idx]
        al = alert_series(flags[idx], rule)
        sc = S[S.run_id == run]
        for r in sc.itertuples():
            ok = np.where(al & (we >= r.start_ts) & (we <= r.end_ts + DS.WIN))[0]
            first = we[ok[0]] if len(ok) else None
            row = dict(method=method, rule=rule, run_id=run, fault_id=r.fault_id, fault_type=r.fault_type, severity_level=r.severity_level, kind=r.kind,
                       detected=first is not None, delay_s=(first - r.start_ts) if first is not None else np.nan)
            if pd.notna(r.failure_ts):
                pre = np.where(al & (we >= r.start_ts) & (we < r.failure_ts))[0]
                row["warned_before_failure"] = len(pre) > 0
                row["lead_time_s"] = (r.failure_ts - we[pre[0]]) if len(pre) else np.nan
                row["failure_s_after_start"] = r.failure_ts - r.start_ts
            inst.append(row)
        starts = episodes(al)
        hours = len(idx) * DS.STEP / 3600.0
        if sc.empty:
            fa_free[0] += len(starts)
            fa_free[1] += hours
        else:
            bad = [k for k in starts if not any(we[k] >= r.start_ts and we[k] - DS.WIN <= r.end_ts + DS.WIN for r in sc.itertuples())]
            fa_other[0] += len(bad)
            fa_other[1] += sum(1 for k in range(len(idx)) if not any(we[k] >= r.start_ts and we[k] - DS.WIN <= r.end_ts + DS.WIN for r in sc.itertuples())) * DS.STEP / 3600.0
    return pd.DataFrame(inst), fa_free, fa_other


# ------------------------------------------------------------------------------------------------ bootstrap over runs
class RunBoot:
    def __init__(self, runs_of_rows, n_boot, seed=SPLIT_SEED):
        self.runs = np.asarray(runs_of_rows)
        self.ids = sorted(set(self.runs))
        self.pos = {r: np.where(self.runs == r)[0] for r in self.ids}
        rng = np.random.default_rng(seed)
        self.draw = rng.integers(0, len(self.ids), (n_boot, len(self.ids)))
        self.ids_arr = np.array(self.ids)

    def rows(self, b):
        return np.concatenate([self.pos[self.ids[i]] for i in self.draw[b]])

    def ci(self, fn):
        vals = np.array([fn(self.rows(b)) for b in range(len(self.draw))], float)
        if np.isfinite(vals).sum() < 20:
            return np.nan, np.nan
        return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def overall_metrics(y, p1, pred, boot):
    from sklearn.metrics import f1_score, roc_auc_score
    def one(rows):
        yy, pp, pr = y[rows], p1[rows], pred[rows]
        if len(np.unique(yy)) < 2:
            return dict(balanced_accuracy=np.nan, macro_f1=np.nan, normal_recall=np.nan, roc_auc=np.nan)
        return dict(balanced_accuracy=bal_acc(yy, pr), macro_f1=float(f1_score(yy, pr, average="macro")), normal_recall=float((pr[yy == 0] == 0).mean()), roc_auc=float(roc_auc_score(yy, pp)))
    pt = one(np.arange(len(y)))
    out = dict(pt)
    cache = [one(boot.rows(b)) for b in range(len(boot.draw))]
    for k in pt:
        v = np.array([c[k] for c in cache], float)
        out[f"{k}_ci_low"], out[f"{k}_ci_high"] = (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) if np.isfinite(v).sum() >= 20 else (np.nan, np.nan)
    return out


def recall_cell(mask, pred, boot):
    """recall (share of the windows in `mask` that are flagged) with a run-bootstrap CI"""
    if mask.sum() == 0:
        return np.nan, np.nan, np.nan
    point = float(pred[mask].mean())
    lo, hi = boot.ci(lambda rows: float(pred[rows][mask[rows]].mean()) if mask[rows].any() else np.nan)
    return point, lo, hi


# ------------------------------------------------------------------------------------------------ main experiment
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    t0 = time.time()
    W, base, rel = load()
    S = run_faults(W)
    asg, attempts = assign_runs(W, S)
    asg.to_csv(REP / "testbed_run_assignment.csv", index=False)
    print(f"run assignment (fixed seed {SPLIT_SEED}, draw {attempts}): {int((asg.split == 'test').sum())} test runs, {int((asg.split == 'train').sum())} train runs")
    print(asg.to_string(index=False))
    test_runs = list(asg[asg.split == "test"].run_id)
    train_runs = list(asg[asg.split == "train"].run_id)
    fold_of = dict(zip(asg.run_id, asg.fold))
    use = (W.transition == 0).to_numpy()
    y_all = W.label_binary.to_numpy(int)
    in_tr = W.run_id.isin(train_runs).to_numpy() & use
    in_te = W.run_id.isin(test_runs).to_numpy() & use
    scored = (W.in_learning == 0).to_numpy()
    fold_w = W.run_id.map(lambda r: fold_of[r] if r in fold_of and fold_of[r] != "" else -1).to_numpy()

    # ---- model selection by run-grouped CV on the train runs
    print("\nrun-grouped cross-validation on the train runs (balanced accuracy of out-of-fold predictions)")
    cv_rows, oof = [], {}
    for fs in FEATURE_SETS:
        X = make_X(W, base, rel, fs)
        for name in MODELS:
            p = np.full(len(W), np.nan)
            for k in range(N_FOLDS):
                tr = in_tr & (fold_w != k)
                te = in_tr & (fold_w == k)
                p[te] = fit_proba(name, X[tr], y_all[tr], X[te])
            m = in_tr & scored
            ba = bal_acc(y_all[m], (p[m] >= 0.5).astype(int))
            thr = best_threshold(y_all[in_tr], p[in_tr])
            cv_rows.append(dict(model=name, feature_set=fs, cv_balanced_accuracy_at_0_5=ba, tuned_threshold=thr, cv_balanced_accuracy_tuned=bal_acc(y_all[m], (p[m] >= thr).astype(int))))
            oof[(name, fs)] = p
            print(f"  {name:<5} {fs:<5} BA {ba:.3f}  tuned threshold {thr:.3f} -> {cv_rows[-1]['cv_balanced_accuracy_tuned']:.3f}")
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(REP / "testbed_train_cv.csv", index=False)
    chosen = {n: cv[cv.model == n].sort_values("cv_balanced_accuracy_tuned", ascending=False).iloc[0] for n in MODELS}

    # ---- held-out test
    ev = scored & in_te
    boot = RunBoot(W.run_id.to_numpy()[ev], a.boot)
    methods, probs, flags = {}, {}, {}
    for name in MODELS:
        c = chosen[name]
        X = make_X(W, base, rel, c.feature_set)
        p = fit_proba(name, X[in_tr], y_all[in_tr], X[in_te | ev])
        full = np.full(len(W), np.nan)
        full[in_te | ev] = p
        probs[name], flags[name] = full, (full >= c.tuned_threshold)
        methods[name] = f"{name} ({c.feature_set})"
    core = [c for c in CORE]
    for kind in ("RobustZ", "IsoForest"):
        sc_all = unsup_scores(W, core, kind)
        grid = np.unique(np.round(np.quantile(sc_all[in_tr & scored], np.linspace(0.3, 0.99, 70)), 6))
        best = max(grid, key=lambda t: bal_acc(y_all[in_tr & scored], (sc_all[in_tr & scored] >= t).astype(int)))
        probs[kind], flags[kind] = sc_all, sc_all >= best
        methods[kind] = f"{kind} (per-run, first 150 s)"
        print(f"  {kind}: alert threshold {best:.3f} chosen on the train runs")
    rows = []
    for key, label in methods.items():
        fl = flags[key]
        s = probs[key]
        m = overall_metrics(y_all[ev], s[ev], fl[ev].astype(int), boot)
        rows.append(dict(method=label, kind="supervised" if key in MODELS else "unsupervised", n_windows=int(ev.sum()), n_fault_windows=int(y_all[ev].sum()), n_test_runs=len(test_runs), **m))
    overall = pd.DataFrame(rows)
    overall.to_csv(REP / "testbed_heldout_overall.csv", index=False)
    print("\nheld-out test runs: overall (run-bootstrap 95% CI)")
    for r in overall.itertuples():
        print(f"  {r.method:<34} BA {r.balanced_accuracy:.3f} ({r.balanced_accuracy_ci_low:.3f}-{r.balanced_accuracy_ci_high:.3f})  macro-F1 {r.macro_f1:.3f}  Normal recall {r.normal_recall:.3f} ({r.normal_recall_ci_low:.3f}-{r.normal_recall_ci_high:.3f})  ROC-AUC {r.roc_auc:.3f}")

    # ---- recall by fault type and by severity level (test runs)
    cells = []
    Dev = W[ev].reset_index(drop=True)
    inst_test = S[S.run_id.isin(test_runs)]
    for key, label in methods.items():
        pr = flags[key][ev].astype(int)
        for group, col in (("fault type", "fault_type"), ("severity level", "severity_level"), ("fault kind", "fault_kind")):
            for val in sorted(v for v in Dev[col].unique() if v):
                mask = (Dev[col] == val).to_numpy()
                point, lo, hi = recall_cell(mask, pr, boot)
                n_inst = int(inst_test[{"fault type": "fault_type", "severity level": "severity_level", "fault kind": "kind"}[group]].eq(val).sum())
                cells.append(dict(method=label, group=group, value=val, windows=int(mask.sum()), instances=n_inst, recall=point, ci_low=lo, ci_high=hi, reliable=bool(n_inst >= RELIABLE_MIN_INSTANCES)))
        nm = (Dev.label_level == "Normal").to_numpy()
        cells.append(dict(method=label, group="Normal windows", value="Normal", windows=int(nm.sum()), instances=int(len(set(Dev[nm].run_id))), recall=float(1 - pr[nm].mean()), ci_low=np.nan, ci_high=np.nan, reliable=True))
    bycell = pd.DataFrame(cells)
    bycell.to_csv(REP / "testbed_heldout_by_fault.csv", index=False)

    # ---- detection delay, false alarms per hour, lead time (test runs, scored windows)
    D = W[ev].reset_index(drop=True)
    inst_rows, fa_rows = [], []
    for key, label in methods.items():
        fl = flags[key][ev]
        for rule in ("1 window", "2 of last 3"):
            it, fa_free, fa_other = detection_table(D, fl, S, label, rule, test_runs)
            inst_rows.append(it)
            fa_rows.append(dict(method=label, rule=rule, fault_free_runs=int(D[D.fault_free_run].run_id.nunique()), fault_free_hours=fa_free[1], alert_episodes_fault_free=fa_free[0],
                                false_alarms_per_hour_fault_free=(fa_free[0] / fa_free[1]) if fa_free[1] else np.nan, alert_episodes_outside_faults_in_fault_runs=fa_other[0],
                                hours_outside_faults_in_fault_runs=fa_other[1], false_alarms_per_hour_outside_faults=(fa_other[0] / fa_other[1]) if fa_other[1] else np.nan))
    inst = pd.concat(inst_rows, ignore_index=True)
    inst.to_csv(REP / "testbed_heldout_instances.csv", index=False)
    fa = pd.DataFrame(fa_rows)
    fa.to_csv(REP / "testbed_false_alarms.csv", index=False)
    print("\nfalse alarms per hour on the fault-free test runs, and detection summary")
    for r in fa.itertuples():
        sub = inst[(inst.method == r.method) & (inst.rule == r.rule)]
        print(f"  {r.method:<34} {r.rule:<12} fault-free: {r.alert_episodes_fault_free} episodes in {r.fault_free_hours:.2f} h = {r.false_alarms_per_hour_fault_free:.1f}/h; outside faults in fault runs {r.false_alarms_per_hour_outside_faults:.1f}/h; "
              f"detected {int(sub.detected.sum())}/{len(sub)}, median delay {sub.delay_s.median():.0f} s")

    # ---- pre-failure warning (gradual faults): dedicated classifier per horizon, XGBoost on the chosen feature set
    best_name = max(MODELS, key=lambda n: chosen[n].cv_balanced_accuracy_tuned)
    fsel = chosen["XGB"].feature_set
    X = make_X(W, base, rel, fsel)
    pre_rows, lead_rows = [], []
    normal = (W.label_level == "Normal").to_numpy()
    for nmin in (1, 2, 3):
        pos = (W[f"prefail_{nmin}m"] == 1).to_numpy()
        trm = in_tr & (pos | normal)
        yt = pos.astype(int)
        # threshold from run-grouped OOF on the train runs
        p_oof = np.full(len(W), np.nan)
        for k in range(N_FOLDS):
            tr = trm & (fold_w != k)
            te = trm & (fold_w == k)
            if yt[tr].sum() and yt[te].size and te.any():
                p_oof[te] = fit_proba("XGB", X[tr], yt[tr], X[te])
        ok = trm & np.isfinite(p_oof)
        grid = np.unique(np.round(np.quantile(p_oof[ok], np.linspace(0.5, 0.995, 100)), 6))
        def f1(t):
            pr = p_oof[ok] >= t
            tp = (pr & (yt[ok] == 1)).sum()
            return 2 * tp / max(pr.sum() + yt[ok].sum(), 1)
        thr = max(grid, key=f1)
        pfull = np.full(len(W), np.nan)
        tem = in_te & scored & (pos | normal)
        pfull[tem] = fit_proba("XGB", X[trm], yt[trm], X[tem])
        flag = pfull >= thr
        d = W[tem].reset_index(drop=True)
        yy, ff = yt[tem], flag[tem]
        tp, fp, fn = int((ff & (yy == 1)).sum()), int((ff & (yy == 0)).sum()), int((~ff & (yy == 1)).sum())
        bt = RunBoot(d.run_id.to_numpy(), a.boot)
        prec = lambda rows: (lambda t, f: t / (t + f) if t + f else np.nan)(int((ff[rows] & (yy[rows] == 1)).sum()), int((ff[rows] & (yy[rows] == 0)).sum()))
        rec = lambda rows: (ff[rows][yy[rows] == 1]).mean() if (yy[rows] == 1).any() else np.nan
        pl, ph = bt.ci(prec)
        rl, rh = bt.ci(rec)
        # lead time per gradual instance: first warning window before the failure
        leads = []
        for r in S[(S.run_id.isin(test_runs)) & S.failure_ts.notna()].itertuples():
            idx = np.where((d.run_id == r.run_id).to_numpy() & ff & (d.window_end.to_numpy() < r.failure_ts) & (d.window_end.to_numpy() >= r.failure_ts - nmin * 60 - DS.WIN))[0]
            idx = idx[np.argsort(d.window_end.to_numpy()[idx])]
            leads.append(dict(horizon_min=nmin, run_id=r.run_id, fault_type=r.fault_type, severity_level=r.severity_level, warned=bool(len(idx)), lead_time_s=(r.failure_ts - d.window_end.to_numpy()[idx[0]]) if len(idx) else np.nan))
        lead_rows += leads
        pre_rows.append(dict(horizon_min=nmin, features=fsel, threshold=thr, train_positive_windows=int(yt[trm].sum()), test_positive_windows=int(yy.sum()), test_negative_windows=int((yy == 0).sum()),
                             true_positive=tp, false_positive=fp, false_negative=fn, precision=tp / (tp + fp) if tp + fp else np.nan, precision_ci_low=pl, precision_ci_high=ph,
                             recall=tp / max(tp + fn, 1), recall_ci_low=rl, recall_ci_high=rh, gradual_instances=len(leads), instances_warned=int(sum(x["warned"] for x in leads)),
                             median_lead_time_s=float(np.nanmedian([x["lead_time_s"] for x in leads])) if any(x["warned"] for x in leads) else np.nan))
    pre = pd.DataFrame(pre_rows)
    pre.to_csv(REP / "testbed_prefailure.csv", index=False)
    pd.DataFrame(lead_rows).to_csv(REP / "testbed_prefailure_instances.csv", index=False)
    print("\npre-failure warning on the held-out test runs (XGBoost, gradual faults)")
    print(pre.round(3).to_string(index=False))

    # ---- shuffled-label control (training labels permuted, test labels real)
    shuf = []
    for name in MODELS:
        c = chosen[name]
        X = make_X(W, base, rel, c.feature_set)
        bas = []
        for r in range(5):
            ys = y_all.copy()
            ys[in_tr] = np.random.default_rng(SPLIT_SEED + r).permutation(y_all[in_tr])
            p = fit_proba(name, X[in_tr], ys[in_tr], X[ev])
            bas.append(bal_acc(y_all[ev], (p >= 0.5).astype(int)))
        shuf.append(dict(method=methods[name], shuffled_labels_mean_ba=float(np.mean(bas)), min_ba=float(np.min(bas)), max_ba=float(np.max(bas)), real_labels_ba=float(overall[overall.method == methods[name]].balanced_accuracy.iloc[0])))
    shuf = pd.DataFrame(shuf)
    shuf.to_csv(REP / "testbed_shuffled_control.csv", index=False)
    print("\nshuffled-label control (balanced accuracy on the test runs):")
    print(shuf.round(3).to_string(index=False))

    # ---- run identification: are runs identifiable, and could that explain the result?
    from xgboost import XGBClassifier
    rid = W.run_id.astype("category").cat.codes.to_numpy()
    blk = (W.window_index // 10).to_numpy()
    te_b = (blk % 5 == 3) & use                                    # within-run block split (as in AnoMod): every run on both sides
    tr_b = (blk % 5 != 3) & use
    Xb = W[base].to_numpy(float)
    clf = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=42, verbosity=0).fit(Xb[tr_b], rid[tr_b])
    pr_run = clf.predict(Xb[te_b])
    acc = float((pr_run == rid[te_b]).mean())
    majority = W.groupby("run_id").label_binary.agg(lambda s: int(s.mean() > 0.5)).reindex(W.run_id.astype("category").cat.categories).to_numpy()
    share = W.groupby("run_id").label_binary.mean().reindex(W.run_id.astype("category").cat.categories).to_numpy()
    rule_pred = majority[pr_run]
    rule_ba = bal_acc(y_all[te_b], rule_pred)
    rule_ba_share = bal_acc(y_all[te_b], (share[pr_run] > 0.0).astype(int))
    # and across runs: a model that only identified runs cannot transfer (test runs are unseen)
    fault_runs = W[~W.fault_free_run]
    rid_df = pd.DataFrame([dict(check="runs identifiable from the 39 window features (within-run block split, 40 classes)", value=acc, chance=1 / 40),
                           dict(check="balanced accuracy of the rule 'predict the run, output that run's majority label'", value=rule_ba, chance=0.5),
                           dict(check="balanced accuracy of the rule 'predict the run, call it a fault if the run has any fault'", value=rule_ba_share, chance=0.5),
                           dict(check="share of fault windows inside the runs that have faults", value=float(fault_runs.label_binary.mean()), chance=np.nan),
                           dict(check="runs in which every window has the same label (a run-identity shortcut would be perfect there)", value=float((W.groupby("run_id").label_binary.nunique() == 1).mean()), chance=np.nan),
                           dict(check="best held-out model balanced accuracy (unseen runs, so run identity cannot help)", value=float(overall.balanced_accuracy.max()), chance=0.5)])
    rid_df.to_csv(REP / "testbed_runid_check.csv", index=False)
    print("\nrun-identification check")
    print(rid_df.round(3).to_string(index=False))

    # ---- supplementary: every run held out once (8-fold run-grouped CV over all 40 runs) for per-type / per-severity numbers with more instances
    allruns = sorted(W.run_id.unique())
    order = list(allruns)
    random.Random(SPLIT_SEED + 2).shuffle(order)
    sup_fold = {r: i % 8 for i, r in enumerate(sorted(order, key=lambda r: -int(W[W.run_id == r].label_binary.sum())))}
    supp_methods = {m: chosen[m] for m in ("XGB",)}
    X = make_X(W, base, rel, chosen["XGB"].feature_set)
    p = np.full(len(W), np.nan)
    sf = W.run_id.map(sup_fold).to_numpy()
    for k in range(8):
        tr, te = use & (sf != k), use & (sf == k)
        p[te] = fit_proba("XGB", X[tr], y_all[tr], X[te])
    thr_s = float(chosen["XGB"].tuned_threshold)
    fl = (p >= thr_s)
    evs = scored & use
    Ds = W[evs].reset_index(drop=True)
    bs = RunBoot(Ds.run_id.to_numpy(), a.boot)
    prs = fl[evs].astype(int)
    sup_rows = []
    for group, col, icol in (("fault type", "fault_type", "fault_type"), ("severity level", "severity_level", "severity_level"), ("fault kind", "fault_kind", "kind")):
        for val in sorted(v for v in Ds[col].unique() if v):
            mask = (Ds[col] == val).to_numpy()
            pt, lo, hi = recall_cell(mask, prs, bs)
            n_inst = int(S[icol].eq(val).sum())
            sup_rows.append(dict(method="XGB (every run held out once, 8 folds)", group=group, value=val, windows=int(mask.sum()), instances=n_inst, recall=pt, ci_low=lo, ci_high=hi, reliable=bool(n_inst >= RELIABLE_MIN_INSTANCES)))
    it, fa_free, fa_other = detection_table(Ds, fl[evs], S, "XGB (CV)", "1 window", allruns)
    it2, fa_free2, _ = detection_table(Ds, fl[evs], S, "XGB (CV)", "2 of last 3", allruns)
    sup = pd.DataFrame(sup_rows)
    sup.to_csv(REP / "testbed_cv_by_fault.csv", index=False)
    pd.concat([it, it2]).to_csv(REP / "testbed_cv_instances.csv", index=False)
    pd.DataFrame([dict(rule="1 window", fault_free_hours=fa_free[1], episodes=fa_free[0], per_hour=fa_free[0] / fa_free[1]), dict(rule="2 of last 3", fault_free_hours=fa_free2[1], episodes=fa_free2[0], per_hour=fa_free2[0] / fa_free2[1])]).to_csv(REP / "testbed_cv_false_alarms.csv", index=False)
    print(f"\nsupplementary (every run held out once): {int(it.detected.sum())}/{len(it)} faults detected (1 window), false alarms on the {int(Ds[Ds.fault_free_run].run_id.nunique())} fault-free runs: {fa_free[0] / fa_free[1]:.1f}/h")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
