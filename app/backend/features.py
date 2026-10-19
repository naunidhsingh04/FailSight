"""Input validation for raw windows (JSON or CSV): columns, system flag, missing features by group, reliability, sequence context."""
from typing import Any
import numpy as np
import pandas as pd

from . import config

SYSTEM_COLS = ("system", "system_is_TT")
OPTIONAL_COLS = ("run_id", "window_id", "window_index", "label_binary", "label_level", "scenario", "block_id", "split",
                 "window_start", "window_end")


class InputError(Exception):
    def __init__(self, status: int, detail: Any):
        self.status, self.detail = status, detail
        super().__init__(str(detail))


def to_frame(records: list, prep) -> tuple[pd.DataFrame, dict]:
    """records -> DataFrame with every raw feature column (NaN where absent) + system_is_TT. Raises InputError on unusable input."""
    if not records:
        raise InputError(400, "No windows in the request.")
    if len(records) > config.MAX_ROWS:
        raise InputError(413, f"Too many windows: {len(records)} (limit {config.MAX_ROWS}).")
    df = pd.DataFrame.from_records(records)
    known = set(prep.raw_cols) | set(SYSTEM_COLS) | set(OPTIONAL_COLS)
    unknown = [c for c in df.columns if c not in known]
    present = [c for c in prep.raw_cols if c in df.columns]
    if not present:
        raise InputError(422, dict(message="None of the columns match the 155 raw window features. Check the column names (e.g. log_line_count, api_error_rate, m_node_load5__mean).",
                                   example_columns=list(prep.raw_cols[:5]), unknown_columns=unknown[:20]))
    if "system_is_TT" in df.columns:
        s = pd.to_numeric(df["system_is_TT"], errors="coerce")
        if s.isna().any() or not s.isin([0, 1]).all():
            raise InputError(422, "system_is_TT must be 0 or 1 in every row.")
        sysid = s.astype(int).to_numpy()
    elif "system" in df.columns:
        s = df["system"].astype(str).str.strip().str.upper()
        if not s.isin(["SN", "TT"]).all():
            raise InputError(422, dict(message="'system' must be 'SN' or 'TT' in every row (the models were trained on SocialNetwork and TrainTicket).",
                                       bad_values=sorted(set(s[~s.isin(['SN', 'TT'])]))[:5]))
        sysid = (s == "TT").astype(int).to_numpy()
    else:
        raise InputError(422, "Missing system flag: provide a 'system' column ('SN' or 'TT') or 'system_is_TT' (0/1). The preprocessing is fitted per system.")
    cols, bad_cells = {}, 0
    for c in prep.raw_cols:
        if c in df.columns:
            raw = df[c]
            num = pd.to_numeric(raw, errors="coerce")
            bad_cells += int((num.isna() & raw.notna() & (raw.astype(str).str.strip() != "")).sum())
            cols[c] = num.replace([np.inf, -np.inf], np.nan).to_numpy(float)
        else:
            cols[c] = np.full(len(df), np.nan)
    cols["system_is_TT"] = sysid
    for c in ("run_id", "window_id", "window_index"):
        if c in df.columns:
            cols[c] = df[c].to_numpy()
    X = pd.DataFrame(cols, index=df.index)
    info = dict(n_windows=len(df), unknown_columns=unknown[:50], n_unknown_columns=len(unknown), feature_columns_present=len(present),
                feature_columns_missing=len(prep.raw_cols) - len(present), non_numeric_cells=bad_cells)
    return X, info


def missing_report(prep, X: pd.DataFrame) -> list:
    """per window: missing share of the features the system is expected to have and the model actually uses (constant features were dropped)"""
    arr = X[prep.raw_cols].to_numpy(float)
    sysid = X["system_is_TT"].to_numpy(int)
    groups = np.array([prep.groups[c] for c in prep.raw_cols])
    used = prep.keep
    out = []
    for i in range(len(arr)):
        expected = (~prep.absent[sysid[i]]) & used
        miss = np.isnan(arr[i]) & expected
        by = {}
        for g in ("api", "logs", "metrics"):
            m = groups == g
            by[config.GROUP_LABEL[g]] = dict(expected=int((expected & m).sum()), missing=int((miss & m).sum()))
        n_exp, n_miss = int(expected.sum()), int(miss.sum())
        frac = n_miss / n_exp if n_exp else 1.0
        tier = "unreliable" if frac > config.RELIABILITY_BAD else "degraded" if frac > config.RELIABILITY_WARN else "normal"
        out.append(dict(expected_features=n_exp, missing_features=n_miss, missing_fraction=frac, by_group=by, reliability=tier,
                        structurally_absent_for_system=int((prep.absent[sysid[i]] & used).sum())))
    return out


def sequence_context(X: pd.DataFrame, T: int = 6) -> np.ndarray:
    """index array (n, T): the last T consecutive rows of the same run_id (file order); first row repeated where history is missing.
    Without a run_id column every window is its own run (its context is the window repeated T times)."""
    n = len(X)
    run = X["run_id"].astype(str).to_numpy() if "run_id" in X.columns else np.arange(n).astype(str)
    ctx = np.zeros((n, T), dtype=int)
    for i in range(n):
        chain = [i]
        for j in range(1, T):
            p = i - j
            if p >= 0 and run[p] == run[i]:
                chain.append(p)
            else:
                break
        chain += [chain[-1]] * (T - len(chain))
        ctx[i] = chain[::-1]
    return ctx
