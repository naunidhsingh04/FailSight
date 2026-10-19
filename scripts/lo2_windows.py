"""LO2 step 2: 30 s windows (10 s step) from the sampled LO2 scenarios, mapped onto the AnoMod feature names used by preprocess.py.

Each LO2 scenario is its own short run (error scenarios last 10-18 s of log activity, `correct` 60 s), so the AnoMod window rule of
02_windows.py is applied per scenario: run_start = first log line, windows k = 0 .. K-2 of 10 s bins (K = bin of the last line), at least one
window per scenario. An error scenario therefore gives ONE window that covers all of its (shorter than 30 s) activity, `correct` gives 5.

Variant A "logs_only" (the requested mapping): log_line_count / log_error_count / log_warning_count from the 7 service logs + MySQL log;
  all api_* and all m_* features are missing (NaN) and the preprocessor imputes them and sets was_missing_api / was_missing_metrics.
Variant B "logs_plus_node_metrics" (extra, clearly separated): additionally the 19 node_exporter / process metric names that also exist among the
  AnoMod metric features, built with the same statistics (mean / max / population std over all samples of that metric name inside the window,
  *_total counters converted to per-second rates per series first, as in 02_windows.py).
"""
import json, re, sys, os
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preprocess as P

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "processed" / "lo2_sample"
REP = ROOT / "report_outputs"
STEP, WIN = 10.0, 30.0
NB = int(WIN // STEP)
SVC = re.compile(r"^light-oauth2-oauth2-(client|code|key|refresh-token|service|token|user)-1\.log$")
TS = re.compile(r"^(\d\d):(\d\d):(\d\d)\.(\d{3}) \[[^\]]*\]\s+(?:\S+\s+)?(TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL)\b")
TS_ANY = re.compile(r"^(\d\d):(\d\d):(\d\d)\.(\d{3}) \[")
MYSQL = re.compile(r"^(\d{4}-\d\d-\d\d)T(\d\d):(\d\d):(\d\d)\.(\d+)Z \S+ \[(\w+)\]")


def read_log(path, day0, start_epoch, mysql=False):
    """-> arrays (t_abs, level) with one entry per PHYSICAL line; continuation lines inherit the time of the previous timestamped line.
    level: 0 other, 2 WARN, 3 ERROR (continuation lines count as lines but not as records)"""
    t, lv, rec = [], [], []
    last, pending = None, 0
    for line in open(path, encoding="utf-8", errors="replace"):
        tod, level = None, 0
        if mysql:
            m = MYSQL.match(line)
            if m:
                tod = int(m[2]) * 3600 + int(m[3]) * 60 + int(m[4]) + float("0." + m[5])
                level = {"Warning": 2, "ERROR": 3}.get(m[6], 0)
        else:
            m = TS.match(line) or TS_ANY.match(line)
            if m:
                tod = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + int(m[4]) / 1000
                lg = m[5] if m.re is TS else None
                level = 3 if lg in ("ERROR", "FATAL") else 2 if lg in ("WARN", "WARNING") else 0
        if tod is not None:
            a = day0 + tod
            if a < start_epoch - 3600:
                a += 86400                                   # midnight wrap
            last = a
            t += [a] * (pending + 1); lv += [0] * pending + [level]    # leading continuation lines (before the first timestamp) get the first time
            pending = 0
        elif last is not None:
            t.append(last); lv.append(0)
        else:
            pending += 1
    return np.array(t), np.array(lv)


def win_sum(a, nwin):
    c = np.concatenate([[0.0], np.cumsum(a)])
    return (c[NB:] - c[:-NB])[:nwin] if len(a) >= NB else np.array([a.sum()])


def build(variant):
    """variant 'A' (logs only) or 'B' (logs + overlapping node metrics) -> DataFrame of windows with all 155 AnoMod feature columns"""
    man = pd.read_csv(ROOT / "processed" / "lo2_sample_manifest.csv")
    man = man[man.copied].sort_values("scenario_order_in_run")
    fl = pd.read_csv(ROOT / "inputs" / "combined_feature_list.csv")
    feats = list(fl.feature)
    metric_feats = [f for f in feats if f.startswith("m_")]
    rows = []
    for _, sc in man.iterrows():
        d = SAMPLE / str(sc.run_id) / sc.scenario
        day0 = (int(sc.start_epoch) // 86400) * 86400
        evs = []
        for f in sorted(d.glob("light-oauth2-oauth2-*-1.log")):
            if SVC.match(f.name):
                evs.append(read_log(f, day0, int(sc.start_epoch)))
        my = d / "light-oauth2-mysqldb-1.log"
        if my.exists() and my.stat().st_size:
            evs.append(read_log(my, day0, int(sc.start_epoch), mysql=True))
        t = np.concatenate([e[0] for e in evs if len(e[0])])
        lv = np.concatenate([e[1] for e in evs if len(e[0])])
        run_start, run_end = t.min(), t.max()
        K = int((run_end - run_start) // STEP)
        nbins = max(K + 1, NB)
        nwin = max(K + 1 - NB + 1, 1)
        b = ((t - run_start) // STEP).astype(int)
        cnt = np.bincount(b, minlength=nbins).astype(float)
        err = np.bincount(b[lv == 3], minlength=nbins).astype(float)
        wrn = np.bincount(b[lv == 2], minlength=nbins).astype(float)
        w = pd.DataFrame({"window_id": [f"LO2|{sc.scenario}|{k}" for k in range(nwin)], "scenario": sc.scenario, "run_id": f"LO2|{sc.scenario}",
                          "label_binary": int(sc.is_error), "label_level": "Anomaly" if sc.is_error else "Normal", "window_index": np.arange(nwin),
                          "scenario_order_in_run": int(sc.scenario_order_in_run), "window_start_epoch": run_start + STEP * np.arange(nwin),
                          "n_log_lines_total": cnt.sum()})
        w = pd.concat([w, pd.DataFrame(np.nan, index=w.index, columns=feats)], axis=1)
        w["log_line_count"] = win_sum(cnt, nwin)
        w["log_error_count"] = win_sum(err, nwin)
        w["log_warning_count"] = win_sum(wrn, nwin)
        if variant == "B":
            ws = run_start + STEP * np.arange(nwin)
            wanted = {f[2:].rsplit("__", 1)[0] for f in metric_feats}
            for name in sorted(wanted):
                p = d / "metrics" / f"metric_{name}.json"
                if not p.exists():
                    continue
                series = json.loads(p.read_text())
                ts, vs = [], []
                for s in series:
                    arr = np.array([(a, float(b_) if b_ not in ("NaN", "nan") else np.nan) for a, b_ in s["values"]], float)
                    if not len(arr):
                        continue
                    v = arr[:, 1]
                    if name.endswith("_total"):
                        v = P.counter_to_rate(arr[:, 0], v)
                    ts.append(arr[:, 0]); vs.append(v)
                if not ts:
                    continue
                ts, vs = np.concatenate(ts), np.concatenate(vs)
                ok = ~np.isnan(vs)
                ts, vs = ts[ok], vs[ok]
                for k in range(nwin):
                    m = (ts >= ws[k]) & (ts < ws[k] + WIN)
                    if m.any():
                        x = vs[m]
                        for stat, val in (("mean", x.mean()), ("max", x.max()), ("std", x.std())):
                            col = f"m_{name}__{stat}"
                            if col in w.columns:
                                w.loc[k, col] = val
        rows.append(w)
    df = pd.concat(rows, ignore_index=True)
    return df


def mapping_table(A, B):
    fl = pd.read_csv(ROOT / "inputs" / "combined_feature_list.csv")
    prep = P.load_preprocessor()
    kept = set(prep.kept_cols)
    rows = []
    for _, r in fl.iterrows():
        f = r.feature
        grp = r.group
        if f in ("log_line_count", "log_error_count", "log_warning_count"):
            how = {"log_line_count": "physical lines (incl. stack-trace continuation lines) of the 7 oauth2 service logs + MySQL log in the window",
                   "log_error_count": "records with level ERROR/FATAL in those logs (MySQL: [ERROR])",
                   "log_warning_count": "records with level WARN in those logs (MySQL: [Warning])"}[f]
            sa = sb = "built"
            src = "light-oauth2-oauth2-{client,code,key,refresh-token,service,token,user}-1.log, light-oauth2-mysqldb-1.log"
            note = "LO2 logs are DEBUG-level Java logs: line counts are orders of magnitude above AnoMod log counts (distribution shift); only 10-18 s of activity in error scenarios"
        elif grp == "api":
            sa = sb = "cannot be built (missing)"
            src = "-"
            how = "-"
            note = ("LO2 has no per-request status code / latency / content length. Locust log lines are task messages without latency; trace spans (not copied) are not HTTP responses. "
                    "Left missing, the preprocessor imputes and sets was_missing_api = 1")
        else:
            has = B[f].notna().any()
            sa = "cannot be built (missing)"
            sb = "built (node_exporter metric of the same name)" if has else "cannot be built (missing)"
            if has:
                base = f[2:].rsplit("__", 1)[0]
                src = f"metrics/metric_{base}.json"
                how = "mean / max / population std over all samples of that metric name in the window" + ("; per-second rate of the counter per series first" if base.endswith("_total") else "")
                note = "same exporter and metric name as AnoMod TT; LO2 samples are 5 s scrapes, only 2-3 per error window"
            else:
                src = "-"
                how = "-"
                note = "no LO2 metric with this name" if (r.present_in_TT or r.present_in_SN) else "-"
                if not has and f[2:].rsplit("__", 1)[0] in {x[2:].rsplit("__", 1)[0] for x in B.columns if x.startswith("m_") and B[x].notna().any()}:
                    note = "metric exists but gave no value in this window"
        rows.append(dict(feature=f, group=grp, present_in_AnoMod_SN=r.present_in_SN, present_in_AnoMod_TT=r.present_in_TT, variant_A_logs_only=sa, variant_B_logs_plus_node_metrics=sb,
                         lo2_source=src, rule=how, kept_by_preprocessor=f in kept, notes=note))
    t = pd.DataFrame(rows)
    t.to_csv(REP / "lo2_feature_mapping.csv", index=False)
    return t


def main():
    A, B = build("A"), build("B")
    for name, d in (("A", A), ("B", B)):
        d.to_parquet(ROOT / "processed" / f"lo2_windows_{name}.parquet", index=False)
    t = mapping_table(A, B)
    kept = t[t.kept_by_preprocessor]
    print(f"LO2 windows: {len(A)} from {A.scenario.nunique()} scenarios; label_binary counts {A.label_binary.value_counts().to_dict()} (Normal windows come from the single `correct` scenario)")
    print(f"windows per scenario: correct {int((A.scenario == 'correct').sum())}, error scenarios 1 each")
    for lab, col in (("A logs_only", "variant_A_logs_only"), ("B logs_plus_node_metrics", "variant_B_logs_plus_node_metrics")):
        nb = (t[col].str.startswith("built")).sum()
        nk = (kept[col].str.startswith("built")).sum()
        print(f"variant {lab}: {nb} of {len(t)} AnoMod features built ({nb / len(t):.1%}); of the {len(kept)} features the models actually use, {nk} ({nk / len(kept):.1%}) are built, the rest are missing/imputed")
    print(t.groupby(["group", "variant_A_logs_only", "variant_B_logs_plus_node_metrics"]).size().to_string())
    print(A[["window_id", "label_binary", "log_line_count", "log_error_count", "log_warning_count"]].head(8).to_string())
    print(A.groupby("label_binary")[["log_line_count", "log_error_count", "log_warning_count"]].describe().T.round(1).to_string())


if __name__ == "__main__":
    main()
