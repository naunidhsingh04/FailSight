"""Single source of preprocessing for training, testing and the web app.

Reuses (does not rewrite) the existing logic:
  * windows / features : AnoMod/scripts/02_windows.py output (30 s windows, 10 s step, SN + TT window tables); counters
                         (*_total) are already converted to per-second rates there, `counter_to_rate()` below is the same rule
                         for raw samples at inference time.
  * split              : AnoMod/scripts/split_blocks.py (copied unchanged to scripts/split_blocks.py): API-aware blocks of
                         10 windows, RandomState(42), 80:20 per system.
  * combined layout    : AnoMod/scripts/04_combined.py (union of SN+TT features, feature absent for a system -> 0 after
                         scaling, system_is_TT as input).

Preprocessor (fitted on TRAIN ONLY, statistics per system):
  non-finite -> NaN -> log1p on heavy-tailed non-negative features -> was_missing indicator per group (api/logs/metrics)
  -> median imputation -> clip at train 0.5 / 99.5 percentiles -> StandardScaler -> absent features = 0 -> constant features dropped.

Usage: python scripts/preprocess.py        (builds split_ids.csv, fold_assignment.csv, preprocessor.joblib, feature_names.json)
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_blocks import block_split  # noqa: E402  (copied unchanged from AnoMod/scripts)

ROOT = Path(__file__).resolve().parent.parent
INPUTS, PROCESSED, REPORTS = ROOT / "inputs", ROOT / "processed", ROOT / "report_outputs"
ANOMOD = ROOT.parent / "AnoMod" / "processed"          # read-only source, used only if inputs/ has no copy
SEED, BLOCK, N_FOLDS = 42, 10, 10
SYSTEMS = ("SN", "TT")
GROUPS = ("api", "logs", "metrics")
HEAVY_SKEW, MIN_NONNULL, CLIP_Q = 2.0, 20, (0.5, 99.5)
MISSING_FRAC = 0.5          # was_missing_<group> = 1 when >= 50% of the group's features that exist for the system are NaN
META = ["window_id", "system", "scenario", "run_id", "block_id", "label_binary", "label_level", "window_index",
        "window_start", "window_end", "system_is_TT"]
FORBIDDEN_NAMES = ("label", "scenario", "run", "split", "window_id", "block", "target", "class", "fold")


def log(*a):
    print(*a, flush=True)


def feature_group(c):
    return "api" if c.startswith("api_") else "metrics" if c.startswith("m_") else "logs" if c.startswith("log_") else None


# --------------------------------------------------------------------------------------------- data
def load_windows():
    """Combined SN+TT window table (one row per 30 s window) + run_id, block_id, system_is_TT. No row dropped."""
    frames = []
    for s in SYSTEMS:
        p = INPUTS / f"{s}_windows.parquet"
        p = p if p.exists() else ANOMOD / s / "windows.parquet"
        w = pd.read_parquet(p).reset_index(drop=True)
        w["run_id"] = s + "|" + w.scenario
        frames.append(w)
    df = pd.concat(frames, ignore_index=True, sort=False)
    feats = [c for c in df.columns if feature_group(c)]
    order = [c for g in GROUPS for c in feats if feature_group(c) == g]
    df["system_is_TT"] = (df.system == "TT").astype(int)
    # block_id = the 10-window block of split_blocks.make_blocks (windows of a run in time order, chunks of 10)
    pos = df.groupby("run_id", sort=False).window_index.rank(method="first").astype(int) - 1
    assert (df.groupby("run_id").window_index.apply(lambda s: (s.diff().dropna() == 1).all())).all()
    df["block_id"] = df.run_id + "|b" + (pos // BLOCK).astype(str).str.zfill(2)
    assert df.window_id.is_unique
    return df[META + order]


def feature_columns(df):
    return [c for c in df.columns if feature_group(c)]


def make_splits(df):
    """Existing API-aware 80:20 block split (seed 42) per system -> DataFrame(window_id, system, run_id, block_id, split)."""
    out = []
    for s in SYSTEMS:
        W = df[df.system == s].reset_index(drop=True)
        mask, info, _ = block_split(W)
        t = W[["window_id", "system", "run_id", "block_id"]].copy()
        t["split"] = np.where(mask, "train", "test")
        out.append(t)
    sp = pd.concat(out, ignore_index=True)
    sp["api_window"] = sp.window_id.map(df.set_index("window_id").api_request_count.gt(0)).astype(int)
    return sp


def verify_split_against_existing(sp):
    """The re-generated split must equal the split already used by every earlier step (combined_windows.parquet)."""
    ex = pd.read_parquet(INPUTS / "combined_windows.parquet", columns=["window_id", "split"]).set_index("window_id").split
    same = (sp.set_index("window_id").split == ex.reindex(sp.window_id).to_numpy()).all()
    return bool(same), int((sp.set_index("window_id").split != ex.reindex(sp.window_id).to_numpy()).sum())


def counter_to_rate(t, v):
    """per-second rate between consecutive samples of ONE counter series (same rule as 02_windows.py): first sample,
    resets (negative delta) and non-positive dt -> NaN."""
    t, v = np.asarray(t, float), np.asarray(v, float)
    dv, dt = np.diff(v, prepend=np.nan), np.diff(t, prepend=np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = dv / dt
    r[np.isnan(r) | (dv < 0) | (dt <= 0)] = np.nan
    return r


# --------------------------------------------------------------------------------------------- preprocessor
class Preprocessor:
    """All statistics are per system (SN / TT); a feature with no value in a system's training rows is 'absent' there."""

    def fit(self, train_df):
        if "split" in train_df.columns and (train_df["split"] != "train").any():
            raise ValueError("fit_preprocessor received non-train rows")
        feats = feature_columns(train_df)
        self.raw_cols = feats
        self.groups = {c: feature_group(c) for c in feats}
        X = np.array(train_df[feats].to_numpy(float), copy=True)
        X[~np.isfinite(X)] = np.nan
        sysid = train_df.system_is_TT.to_numpy(int)
        self.n_fit_rows = len(train_df)
        self.fit_ids_hash = int(pd.util.hash_pandas_object(train_df.window_id, index=False).sum() % (2 ** 62)) if "window_id" in train_df else None
        # counters: windows already hold rates -> verify (non-negative) and record, never convert twice
        self.counter_cols = [c for c in feats if c.startswith("m_") and c[2:].rsplit("__", 1)[0].endswith("_total")]
        ci = [feats.index(c) for c in self.counter_cols]
        self.counter_min = float(np.nanmin(X[:, ci])) if ci else 0.0
        # log1p on heavy-tailed non-negative features (pooled train values)
        self.log_mask = np.zeros(len(feats), bool)
        for j in range(len(feats)):
            v = X[:, j][~np.isnan(X[:, j])]
            if len(v) >= MIN_NONNULL and v.min() >= 0 and np.ptp(v) > 0 and skew(v) > HEAVY_SKEW:
                self.log_mask[j] = True
        Xl = self._log(X)
        f = len(feats)
        self.absent = np.zeros((2, f), bool)
        self.median, self.lo, self.hi = (np.zeros((2, f)) for _ in range(3))
        self.std_clipped = np.zeros((2, f))
        self.mean_, self.scale_ = np.zeros((2, f)), np.ones((2, f))
        Xs_all = {}
        for s in (0, 1):
            r = Xl[sysid == s]
            if len(r) == 0:
                raise ValueError(f"no training rows for system {SYSTEMS[s]}")
            nn = ~np.isnan(r)
            self.absent[s] = ~nn.any(0)
            for j in range(f):
                if self.absent[s, j]:
                    continue
                v = r[nn[:, j], j]
                self.median[s, j] = np.median(v)
                self.lo[s, j], self.hi[s, j] = np.percentile(v, CLIP_Q)
            Xi = np.where(np.isnan(r), self.median[s], r)
            Xc = np.clip(Xi, self.lo[s], self.hi[s])
            pres = ~self.absent[s]
            sc = StandardScaler().fit(Xc[:, pres])
            self.mean_[s, pres], self.scale_[s, pres] = sc.mean_, sc.scale_
            Xs_all[s] = (Xc[:, pres] - sc.mean_) / sc.scale_
            self.std_clipped[s, pres] = Xc[:, pres].std(0)
        # constant features: constant (or absent) in every system
        const = np.ones(f, bool)
        for s in (0, 1):
            const &= self.absent[s] | (self.std_clipped[s] == 0)
        self.const_cols = [c for c, k in zip(feats, const) if k]
        self.keep = ~const
        self.kept_cols = [c for c, k in zip(feats, self.keep) if k]
        # group masks per system for was_missing indicators (features that exist for the system)
        self.group_idx = {g: np.array([j for j, c in enumerate(feats) if self.groups[c] == g]) for g in GROUPS}
        self.indicator_names = [f"was_missing_{g}" for g in GROUPS]
        self.output_names = self.kept_cols + self.indicator_names + ["system_is_TT"]
        self.n_out = len(self.output_names)
        return self

    def _log(self, X):
        X = X.copy()
        X[:, self.log_mask] = np.log1p(np.maximum(X[:, self.log_mask], 0))
        return X

    def transform_array(self, X, sysid):
        X = np.array(X, float, copy=True)
        X[~np.isfinite(X)] = np.nan
        sysid = np.asarray(sysid, int)
        n, f = X.shape
        assert f == len(self.raw_cols)
        Xl = self._log(X)
        out = np.zeros((n, self.n_out))
        k = len(self.kept_cols)
        for s in (0, 1):
            m = sysid == s
            if not m.any():
                continue
            r = Xl[m]
            nan = np.isnan(r)
            for gi, g in enumerate(GROUPS):
                idx = self.group_idx[g]
                ex = ~self.absent[s, idx]
                frac = nan[:, idx][:, ex].mean(1) if ex.any() else np.ones(m.sum())
                out[np.where(m)[0], k + gi] = (frac >= MISSING_FRAC).astype(float)
            Xi = np.where(nan, self.median[s], r)
            Z = (np.clip(Xi, self.lo[s], self.hi[s]) - self.mean_[s]) / self.scale_[s]
            Z[:, self.absent[s]] = 0.0
            out[np.where(m)[0], :k] = Z[:, self.keep]
        out[:, -1] = sysid
        assert np.isfinite(out).all()
        return out

    def transform(self, df):
        missing = [c for c in self.raw_cols if c not in df.columns]
        X = df.reindex(columns=self.raw_cols).to_numpy(float) if missing else df[self.raw_cols].to_numpy(float)
        return self.transform_array(X, df.system_is_TT.to_numpy(int) if "system_is_TT" in df else (df.system == "TT").to_numpy(int))

    def transform_raw_window(self, w):
        """One window as a dict {'system': 'SN'|'TT', '<feature>': value, ...}; absent feature keys count as missing.
        Optional w['_counter_samples'] = {metric_name: (timestamps, cumulative_values)} for *_total counters: converted to
        per-second rates (counter_to_rate) and summarised as m_<name>__mean/max/std for the window."""
        w = dict(w)
        for name, (t, v) in (w.pop("_counter_samples", None) or {}).items():
            r = counter_to_rate(t, v)
            r = r[~np.isnan(r)]
            if len(r):
                w[f"m_{name}__mean"], w[f"m_{name}__max"], w[f"m_{name}__std"] = r.mean(), r.max(), r.std()
        if w.get("system") not in SYSTEMS:
            raise ValueError("window dict needs 'system' in ('SN','TT')")
        x = np.array([[w.get(c, np.nan) if w.get(c, np.nan) is not None else np.nan for c in self.raw_cols]], float)
        return self.transform_array(x, [int(w["system"] == "TT")])[0]

    # ------------------------------------------------------------------ persistence (state dict -> no pickled class)
    _STATE = ("raw_cols", "groups", "n_fit_rows", "fit_ids_hash", "counter_cols", "counter_min", "log_mask", "absent", "median",
              "lo", "hi", "mean_", "scale_", "std_clipped", "const_cols", "keep", "kept_cols", "group_idx", "indicator_names",
              "output_names", "n_out")

    def to_state(self):
        return {k: getattr(self, k) for k in self._STATE}

    @classmethod
    def from_state(cls, st):
        p = cls()
        for k, v in st.items():
            setattr(p, k, v)
        return p

    def feature_names(self):
        grp = {"api": [], "log": [], "metric": [], "system": ["system_is_TT"], "missing_indicator": list(self.indicator_names)}
        for c in self.kept_cols:
            grp[{"api": "api", "logs": "log", "metrics": "metric"}[self.groups[c]]].append(c)
        return dict(groups=grp, all=self.output_names, n_features=self.n_out, dropped_constant=self.const_cols,
                    log1p_features=[c for c, m in zip(self.raw_cols, self.log_mask) if m],
                    absent_in_SN=[c for c, a in zip(self.raw_cols, self.absent[0]) if a],
                    absent_in_TT=[c for c, a in zip(self.raw_cols, self.absent[1]) if a])


def fit_preprocessor(train_df):
    return Preprocessor().fit(train_df)


def transform(df, prep=None):
    return (prep or load_preprocessor()).transform(df)


def transform_raw_window(w, prep=None):
    return (prep or load_preprocessor()).transform_raw_window(w)


def save_preprocessor(prep, path=None):
    path = Path(path or PROCESSED / "preprocessor.joblib")
    path.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(dict(version=1, state=prep.to_state()), path)
    (path.parent / "feature_names.json").write_text(json.dumps(prep.feature_names(), indent=1), encoding="utf-8")
    return path


def load_preprocessor(path=None):
    return Preprocessor.from_state(joblib.load(Path(path or PROCESSED / "preprocessor.joblib"))["state"])


# --------------------------------------------------------------------------------------------- folds
def make_folds(df, split_ids):
    """StratifiedGroupKFold(10) on the train rows, stratified on label_level, groups = run_id. Falls back to block_id when
    any class has fewer runs than folds. Returns DataFrame(window_id, fold (1..10, 0 = test row), group_type)."""
    tr = df.drop(columns=["split"], errors="ignore").merge(split_ids[["window_id", "split"]], on="window_id")
    tr = tr[tr.split == "train"].reset_index(drop=True)
    runs_per_class = tr.groupby("label_level").run_id.nunique()
    blocks_per_class = tr.groupby("label_level").block_id.nunique()
    use_blocks = bool((runs_per_class < N_FOLDS).any())
    if use_blocks:
        log(f"FOLD GROUPS: runs per class in train {runs_per_class.to_dict()} -> fewer than {N_FOLDS} for some class -> fallback groups = block_id "
            f"(blocks per class {blocks_per_class.to_dict()})")
    groups = tr.block_id if use_blocks else tr.run_id
    assert (blocks_per_class >= N_FOLDS).all() or not use_blocks, "fewer blocks than folds"
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold = np.zeros(len(tr), int)
    for k, (_, va) in enumerate(sgkf.split(tr, tr.label_level, groups), 1):
        fold[va] = k
    assert (fold > 0).all()
    fa = pd.DataFrame({"window_id": tr.window_id, "fold": fold, "group_type": "block_id" if use_blocks else "run_id"})
    test = split_ids[split_ids.split == "test"][["window_id"]].assign(fold=0, group_type="")
    out = pd.concat([fa, test], ignore_index=True)
    return out


# --------------------------------------------------------------------------------------------- main
def main():
    df = load_windows()
    log(f"windows: {len(df)} (SN {int((df.system=='SN').sum())}, TT {int((df.system=='TT').sum())}), feature columns {len(feature_columns(df))}, "
        f"runs {df.run_id.nunique()}, blocks {df.block_id.nunique()}")
    sp = make_splits(df)
    same, ndiff = verify_split_against_existing(sp)
    log(f"split regenerated with split_blocks.block_split (seed {SEED}); identical to existing combined split: {same} (differing rows: {ndiff})")
    if not same:
        raise SystemExit("regenerated split differs from the existing one -> stop")
    REPORTS.mkdir(exist_ok=True)
    sp.to_csv(REPORTS / "split_ids.csv", index=False)
    log(f"split_ids.csv: train {int((sp.split=='train').sum())}, test {int((sp.split=='test').sum())}; per system:\n" +
        sp.groupby(["system", "split"]).size().unstack().assign(train_pct=lambda t: (100 * t.train / (t.train + t.test)).round(2)).to_string())
    d = df.merge(sp[["window_id", "split"]], on="window_id")
    train = d[d.split == "train"]
    prep = fit_preprocessor(train)
    save_preprocessor(prep)
    log(f"preprocessor fitted on {prep.n_fit_rows} train rows; raw features {len(prep.raw_cols)}; log1p on {int(prep.log_mask.sum())}; "
        f"dropped constant {len(prep.const_cols)}; absent in SN {int(prep.absent[0].sum())}, in TT {int(prep.absent[1].sum())}; output width {prep.n_out}")
    log(f"counter features (*_total) in windows: {len(prep.counter_cols)}; min value {prep.counter_min} (rates already, no second conversion)")
    Z = prep.transform(d)
    for g in prep.indicator_names:
        j = prep.output_names.index(g)
        log(f"  {g}: share of windows =1  " + ", ".join(f"{s}/{sp_}: {Z[(d.system==s).to_numpy() & (d.split==sp_).to_numpy(), j].mean():.3f}" for s in SYSTEMS for sp_ in ("train", "test")))
    folds = make_folds(d, sp)
    folds.to_csv(REPORTS / "fold_assignment.csv", index=False)
    f = folds.merge(d[["window_id", "system", "label_level", "block_id", "run_id"]], on="window_id")
    f = f[f.fold > 0]
    log("rows per fold (validation) by class:\n" + f.groupby(["fold", "label_level"]).size().unstack(fill_value=0).assign(total=lambda t: t.sum(axis=1)).to_string())
    log("blocks per fold:", f.groupby("fold").block_id.nunique().to_dict(), "| same block in >1 fold:", int((f.groupby("block_id").fold.nunique() > 1).sum()))
    log("saved:", PROCESSED / "preprocessor.joblib", PROCESSED / "feature_names.json", REPORTS / "split_ids.csv", REPORTS / "fold_assignment.csv")


if __name__ == "__main__":
    main()
