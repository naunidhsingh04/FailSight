"""Evaluation of one live demo from the monitor's per-window predictions and the app's own fault record (exact start / end / failure timestamps).
Used by testbed/live_demo.py and recomputed by the proofs (P12), so the numbers in testbed_live_validation.csv can be checked from the stored per-window file."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from testbed import dataset as DS  # noqa: E402

GUARD_S = DS.WIN                     # an alert up to 30 s after the end of a fault still belongs to it (windows are 30 s long)


def label_windows(windows, sched, anchor):
    df = windows.copy().sort_values("window_index").reset_index(drop=True)
    df = DS.label_run(df, {"start_ts": anchor}, sched)
    return df


def evaluate(df, sched):
    """df: labelled windows (monitor output + labels); sched: the recorded fault schedule of this run. Returns one result row (dict)."""
    scored = (~df.learning) & (df.transition == 0)
    S = df[scored]
    fault_w = S[S.label_binary == 1]
    normal_w = S[S.label_binary == 0]
    row = dict(windows=int(len(df)), learning_windows=int(df.learning.sum()), scored_windows=int(len(S)), transition_windows_excluded=int(((~df.learning) & (df.transition == 1)).sum()),
               fault_windows=int(len(fault_w)), fault_windows_flagged=int(fault_w.flagged.sum()), fault_windows_alerted=int(fault_w.alert.sum()),
               missed_fault_windows=int((~fault_w.flagged).sum()), normal_windows=int(len(normal_w)), normal_windows_flagged=int(normal_w.flagged.sum()),
               normal_window_false_flag_share=float(normal_w.flagged.mean()) if len(normal_w) else np.nan, fault_window_recall=float(fault_w.flagged.mean()) if len(fault_w) else np.nan)
    if len(fault_w) and len(normal_w):
        row["balanced_accuracy"] = float((fault_w.flagged.mean() + (1 - normal_w.flagged.mean())) / 2)
    else:
        row["balanced_accuracy"] = np.nan
    spans = [(r.start_ts, (r.end_ts if pd.notna(r.end_ts) else r.start_ts) + GUARD_S) for r in sched.itertuples()]
    new = df[df.new_alert & (~df.learning)]
    false_alerts = [a for a in new.itertuples() if not any(a.window_end >= lo and a.window_end - DS.WIN <= hi for lo, hi in spans)]
    quiet_s = float(sum(DS.STEP for w in S.itertuples() if not any(w.window_end >= lo and w.window_end - DS.WIN <= hi for lo, hi in spans)))
    row.update(alerts_total=int(len(new)), false_alarms=int(len(false_alerts)), scored_hours_outside_faults=quiet_s / 3600.0,
               false_alarms_per_hour=(len(false_alerts) / (quiet_s / 3600.0)) if quiet_s else np.nan, false_alarm_window_end_s=[round(a.window_end - df.window_start.iloc[0], 1) for a in false_alerts])
    inst = []
    for r in sched.itertuples():
        end = r.end_ts if pd.notna(r.end_ts) else r.start_ts
        hit = df[df.alert & (~df.learning) & (df.window_end >= r.start_ts) & (df.window_end <= end + GUARD_S)]
        first = hit.window_end.min() if len(hit) else np.nan
        d = dict(fault_id=r.fault_id, fault_type=r.fault_type, severity_level=r.severity_level, kind=r.kind, start_ts=r.start_ts, end_ts=end, failure_ts=r.failure_ts,
                 detected=bool(len(hit)), delay_s=float(first - r.start_ts) if len(hit) else np.nan)
        if pd.notna(r.failure_ts):
            pre = df[df.alert & (~df.learning) & (df.window_end >= r.start_ts) & (df.window_end < r.failure_ts)]
            d["warned_before_failure"] = bool(len(pre))
            d["lead_time_s"] = float(r.failure_ts - pre.window_end.min()) if len(pre) else np.nan
        inst.append(d)
    row["instances"] = inst
    return row
