"""Universal (system-agnostic) features and their baseline-relative versions.

PART 1  shared features: which of the 155 raw window features exist (have values) in BOTH SocialNetwork (SN) and TrainTicket (TT)?   -> report_outputs/shared_features.csv
PART 1B generic metric aggregates (scripts/generic_metrics.py): 9 concepts x mean/max/std that exist in both systems.            -> report_outputs/generic_metric_features.csv
        feature sets:  (a) shared API + log (8 features)   (b) generic metric aggregates (27)   (c) both (35)

BASELINE-RELATIVE FEATURES (the exact rule)
  For a window k of a run, every feature x is replaced by a robust z-score against that run's own recent past:
        eligible past  = windows j of the SAME run with index j <= k - GUARD          (GUARD = 3: window j ends at 10j+30 s, window k starts at 10k s, so j <= k-3
                                                                                      means the past window ended before this window began: no overlap)
        baseline       = the most recent HISTORY (60) eligible values that are not missing; at least MIN_HISTORY (9) are needed, otherwise the window has no value
        median m, MAD  = median and median absolute deviation of the baseline;  scale s = max(1.4826 * MAD, 0.05 * |m|, 1e-12)
        z              = clip((x - m) / s, -10, 10)
  No label, no other run, no future window and no other system enters a window's baseline.  Test rows never enter the baseline of a TRAINING window:
  for a window in the training split the eligible past is restricted to windows of the training split; a test window may use any earlier window of its own
  run, as a live agent would (it needs no label).  The first GUARD + MIN_HISTORY - 1 = 11 windows of every run therefore have no value and are left out of
  training and testing alike.  `python scripts/universal_features.py --prove` runs the invariance proof: it overwrites every future window (and, for training
  windows, every test window) with random values and checks that the baseline-relative features of the windows under test do not change by a single bit.

python scripts/universal_features.py [--prove]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
REP = ROOT / "report_outputs"
OUT = ROOT / "processed" / "universal_windows.parquet"
GUARD, HISTORY, MIN_HISTORY, CLIP = 3, 60, 9, 10.0
SHARED_COLS_NOTE = "a feature is shared when it has values in both systems' windows"


# ---------------------------------------------------------------------------------------------------- feature sets
def shared_report(w):
    import preprocess as P
    feats = P.feature_columns(w)
    rows = []
    for c in feats:
        r = dict(feature=c, group=P.feature_group(c))
        for s in ("SN", "TT"):
            r[f"{s}_share_with_value"] = float(w[w.system == s][c].notna().mean())
        r["in_SN"], r["in_TT"] = r["SN_share_with_value"] > 0, r["TT_share_with_value"] > 0
        r["shared"] = r["in_SN"] and r["in_TT"]
        rows.append(r)
    return pd.DataFrame(rows)


def feature_sets(shared):
    sh = list(shared[shared.shared].feature)
    from generic_metrics import CONCEPTS
    gen = [f"g_{c}__{s}" for c in CONCEPTS for s in ("mean", "max", "std")]
    return dict(a_shared_api_log=sh, b_generic_metrics=gen, c_both=sh + gen)


# ---------------------------------------------------------------------------------------------------- baseline-relative transform
def baseline_stats(vals):
    m = float(np.median(vals))
    mad = float(np.median(np.abs(vals - m)))
    return m, max(1.4826 * mad, 0.05 * abs(m), 1e-12)


def relative_features(X, run_order, is_train, cols_with_nan_ok=True):
    """X [n, f] raw values of ONE run in time order; is_train [n] bool (the split of each window). Returns Z [n, f] (NaN where there is no value)."""
    n, f = X.shape
    Z = np.full((n, f), np.nan)
    for k in range(n):
        elig_all = np.arange(0, k - GUARD + 1)
        if len(elig_all) == 0:
            continue
        elig = elig_all[is_train[elig_all]] if is_train[k] else elig_all
        if len(elig) < MIN_HISTORY:
            continue
        for j in range(f):
            v = X[elig, j]
            v = v[np.isfinite(v)][-HISTORY:]
            if len(v) < MIN_HISTORY or not np.isfinite(X[k, j]):
                continue
            m, s = baseline_stats(v)
            Z[k, j] = np.clip((X[k, j] - m) / s, -CLIP, CLIP)
    return Z


def build_all(w, G, sets):
    """w: window table with split; G: generic windows. Returns DataFrame with raw columns (shared + generic) and rel__<col> columns, plus keys."""
    cols = sorted(set(sum(sets.values(), [])))
    d = w.merge(G, on="window_id", how="left").sort_values(["run_id", "window_index"]).reset_index(drop=True)
    rel = np.full((len(d), len(cols)), np.nan)
    for run, idx in d.groupby("run_id").indices.items():
        idx = np.sort(idx)
        X = d.loc[idx, cols].to_numpy(float)
        rel[idx] = relative_features(X, None, (d.loc[idx, "split"] == "train").to_numpy())
    R = pd.DataFrame(rel, columns=[f"rel__{c}" for c in cols])
    keep = ["window_id", "system", "run_id", "block_id", "scenario", "label_binary", "label_level", "window_index", "split"]
    return pd.concat([d[keep + cols].reset_index(drop=True), R], axis=1), cols


# ---------------------------------------------------------------------------------------------------- proof that no future / test data enters a baseline
def prove(d, cols, n_runs=6, seed=7):
    """Overwrite the data a window must not see with random garbage and verify that its relative features are bit-identical."""
    rng = np.random.default_rng(seed)
    runs = list(d.groupby("run_id").indices.items())
    rng.shuffle(runs)
    checked, bad = 0, 0
    for run, idx in runs[:n_runs]:
        idx = np.sort(idx)
        X = d.loc[idx, cols].to_numpy(float)
        tr = (d.loc[idx, "split"] == "train").to_numpy()
        ref = relative_features(X, None, tr)
        n = len(idx)
        for k in rng.choice(np.arange(GUARD + MIN_HISTORY, n), size=min(15, n - GUARD - MIN_HISTORY), replace=False):
            Xp = X.copy()
            future = np.array([j for j in range(k - GUARD + 1, n) if j != k])   # windows that overlap or follow window k (but not k itself): must not matter
            scale = np.nanmax(np.abs(X), axis=0) + 1.0
            Xp[future] = rng.standard_normal((len(future), X.shape[1])) * scale * 100
            if tr[k]:                                               # a training window must not depend on ANY test window either
                past_test = np.where(~tr[:k - GUARD + 1])[0]
                Xp[past_test] = rng.standard_normal((len(past_test), X.shape[1])) * scale * 100
            Zp = relative_features(Xp, None, tr)
            same = np.array_equal(ref[k], Zp[k], equal_nan=True)
            checked += 1
            bad += 0 if same else 1
    return checked, bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prove", action="store_true")
    a = ap.parse_args()
    import preprocess as P
    t0 = time.time()
    w = P.load_windows()
    sp = pd.read_csv(REP / "split_ids.csv")
    w = w.merge(sp[["window_id", "split"]], on="window_id")
    shared = shared_report(w)
    shared.to_csv(REP / "shared_features.csv", index=False)
    s = shared[shared.shared]
    print(f"shared features (values in both SN and TT): {len(s)} of {len(shared)}")
    print(s.groupby("group").feature.apply(list).to_string())
    print("only SN:", len(shared[shared.in_SN & ~shared.in_TT]), " only TT:", len(shared[shared.in_TT & ~shared.in_SN]), " none:", len(shared[~shared.in_SN & ~shared.in_TT]))
    print("SN-only API features:", list(shared[shared.in_SN & ~shared.in_TT & (shared.group == "api")].feature))
    G = pd.read_parquet(ROOT / "processed" / "generic_metric_windows.parquet")
    sets = feature_sets(shared)
    d, cols = build_all(w, G, sets)
    d.to_parquet(OUT, index=False)
    print(f"universal table: {len(d)} windows ({time.time() - t0:.0f} s); windows with at least one baseline-relative value per set, and share of values present:")
    for name, cl in sets.items():
        R = d[[f"rel__{c}" for c in cl]]
        t = d.assign(any_=R.notna().any(axis=1).to_numpy(), share=R.notna().mean(axis=1).to_numpy()).groupby(["system", "split"]).agg(windows=("any_", "size"), with_value=("any_", "sum"), mean_share=("share", "mean"))
        print(name); print(t.round(3).to_string())
    if a.prove:
        n, bad = prove(d, cols)
        print(f"no-future / no-test proof: {n} windows checked against overwritten future (and test) windows, {bad} differences")
        assert bad == 0


if __name__ == "__main__":
    main()
