"""EXTERNAL TEST on RCAEval RE1 (metrics only; Online Boutique, Sock Shop, Train Ticket). TESTING ONLY: nothing is tuned or trained on this data.

Cases: for every system and fault type (cpu, mem, disk, delay, loss) the 10 cases <service>/runs 1 and 2 of the five services (deterministic choice). Every case has an explicit injection time
(inject_time.txt), so no assumption about a 10-minute normal period is needed. Features: 30 s windows with a 10 s step over generic aggregates that exist in the case (CPU, memory,
network, disk, request rate, latency, errors; scripts documented in report_outputs/rcaeval_feature_mapping.csv). Detectors: the two unsupervised per-app detectors of the testbed
(robust z-score and IsolationForest, learned from each case's own first 150 s) with the thresholds FROZEN from the testbed TRAIN runs and the frozen alert rule (2 of the last 3 windows).
Placebo control: the same detection test at random fake injection times inside the normal period.

python scripts/rcaeval_eval.py [--boot 1000]
Writes report_outputs/rcaeval_external_results.csv, rcaeval_case_results.csv, rcaeval_feature_mapping.csv, rcaeval_summary.md."""
import argparse
import json
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
DATA = ROOT / "external" / "rcaeval"
REP = ROOT / "report_outputs"
SYSTEMS = {"RE1-OB": "Online Boutique", "RE1-SS": "Sock Shop", "RE1-TT": "Train Ticket"}
FAULTS = ["cpu", "mem", "disk", "delay", "loss"]
RUNS = (1, 2)
WIN, STEP, LEARN_S, HORIZON_S, PLACEBOS = 30.0, 10.0, 150.0, 120.0, 5
AGG = ["cpu_total", "mem_total", "net_rx", "net_tx", "disk_read", "disk_write", "latency_p95", "request_rate", "error_rate"]


def select_cases():
    cases = []
    for sysd in SYSTEMS:
        for fault in FAULTS:
            for folder in sorted((DATA / sysd).glob(f"*_{fault}")):
                for run in RUNS:
                    d = folder / str(run)
                    if (d / "data.csv").exists():
                        cases.append(dict(system=sysd, fault=fault, service=folder.name.rsplit("_", 1)[0], run=run, dir=d))
    return cases


def rate(df, cols):
    """per-second rate of cumulative counters (resets and gaps -> 0), summed over the columns"""
    if not cols:
        return None
    t = df["time"].to_numpy(float)
    X = df[cols].to_numpy(float)
    d = np.diff(X, axis=0, prepend=np.nan) / np.maximum(np.diff(t, prepend=np.nan), 1e-9)[:, None]
    d[(d < 0) | ~np.isfinite(d)] = 0.0
    return d.sum(axis=1)


def aggregates(df, system):
    """generic aggregates present in this case (name -> per-second series); the exact source columns per system are listed in rcaeval_feature_mapping.csv"""
    cols = list(df.columns[1:])
    pick = lambda pat: [c for c in cols if re.search(pat, c)]
    out = {}
    if system == "RE1-OB":
        for name, pat in (("cpu_total", r"_cpu$"), ("mem_total", r"_mem$"), ("request_rate", r"_load$"), ("error_rate", r"_error$")):
            c = pick(pat)
            if c:
                out[name] = df[c].sum(axis=1, min_count=1).to_numpy(float)
        c = pick(r"_latency$")
        if c:
            out["latency_p95"] = df[c].mean(axis=1).to_numpy(float)
    else:
        r = rate(df, pick(r"_container-cpu-usage-seconds-total$"))
        if r is not None:
            out["cpu_total"] = r
        c = pick(r"_container-memory-working-set-bytes$")
        if c:
            out["mem_total"] = df[c].sum(axis=1).to_numpy(float)
        for name, pat in (("net_rx", r"_container-network-receive-bytes-total$"), ("net_tx", r"_container-network-transmit-bytes-total$"), ("disk_read", r"_container-fs-reads-bytes-total$"),
                          ("disk_write", r"_container-fs-writes-bytes-total$"), ("request_rate", r"_istio-request-total$"), ("error_rate", r"_istio-error-total$")):
            r = rate(df, pick(pat))
            if r is not None:
                out[name] = r
        c = pick(r"_istio-latency-95$")
        if c:
            out["latency_p95"] = df[c].mean(axis=1).to_numpy(float)
    return out


def case_windows(case, feats):
    df = pd.read_csv(case["dir"] / "data.csv")
    df = df[df["time"].notna()].reset_index(drop=True)
    inject = float(open(case["dir"] / "inject_time.txt").read().strip())
    t = df["time"].to_numpy(float)
    ag = aggregates(df, case["system"])
    n_win = int((t[-1] - t[0] - WIN) // STEP) + 1
    rows = []
    for k in range(n_win):
        ws = t[0] + STEP * k
        m = (t >= ws) & (t < ws + WIN)
        r = dict(window_index=k, window_start=ws, window_end=ws + WIN)
        for f in feats:
            v = ag[f][m] if f in ag else np.array([np.nan])
            r[f] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
        rows.append(r)
    W = pd.DataFrame(rows)
    return W, inject, t[0], t[-1], sorted(ag)


def frozen_thresholds():
    """alert thresholds of the two unsupervised detectors, chosen on the testbed TRAIN runs only (the same procedure as testbed.validate): nothing here looks at RCAEval"""
    import testbed.validate as V
    W, base, rel = V.load()
    asg = pd.read_csv(ROOT / "report_outputs" / "testbed_run_assignment.csv")      # committed testbed result, independent of where this run writes its output
    train = set(asg[asg.split == "train"].run_id)
    use = (W.transition == 0).to_numpy() & W.run_id.isin(train).to_numpy() & (W.in_learning == 0).to_numpy()
    y = W.label_binary.to_numpy(int)
    out = {}
    for kind in ("RobustZ", "IsoForest"):
        sc = V.unsup_scores(W, V.CORE, kind)
        grid = np.unique(np.round(np.quantile(sc[use], np.linspace(0.3, 0.99, 70)), 6))
        out[kind] = float(max(grid, key=lambda t: V.bal_acc(y[use], (sc[use] >= t).astype(int))))
    return out


def alerts_2of3(flags):
    f = np.asarray(flags, bool)
    return np.array([f[max(0, k - 2):k + 1].sum() >= 2 for k in range(len(f))])


def boot_ci(values_fn, n_cases, n_boot, seed=7):
    rng = np.random.default_rng(seed)
    v = np.array([values_fn(rng.integers(0, n_cases, n_cases)) for _ in range(n_boot)], float)
    return (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) if np.isfinite(v).sum() > 20 else (np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    import testbed.validate as V
    t0 = time.time()
    cases = select_cases()
    size = sum((c["dir"] / "data.csv").stat().st_size for c in cases)
    per_sys = ", ".join(f"{s_}: {sum(1 for c_ in cases if c_['system'] == s_)}" for s_ in SYSTEMS)
    print(f"cases taken: {len(cases)} ({per_sys}), per fault type per system {len(cases) // (len(SYSTEMS) * len(FAULTS))}; selected data.csv files {size / 1e6:.0f} MB uncompressed (downloaded: RE1-OB.zip 31 MB, RE1-SS.zip 79 MB, RE1-TT.zip 280 MB = 390 MB in total)")
    th = frozen_thresholds()
    print("frozen thresholds (testbed train runs):", th)
    # feature availability per system
    avail = {s: {f: 0 for f in AGG} for s in SYSTEMS}
    n_case = {s: 0 for s in SYSTEMS}
    for c in cases:
        df = pd.read_csv(c["dir"] / "data.csv")
        n_case[c["system"]] += 1
        for f in aggregates(df, c["system"]):
            avail[c["system"]][f] += 1
    feats = {s: [f for f in AGG if avail[s][f] == n_case[s]] for s in SYSTEMS}
    src = {"RE1-OB": dict(cpu_total="sum of <service>_cpu", mem_total="sum of <service>_mem", request_rate="sum of <service>_load (workload as given)", error_rate="sum of <service>_error", latency_p95="mean of <service>_latency (as given)"),
           "other": dict(cpu_total="sum of the per-second rate of container-cpu-usage-seconds-total", mem_total="sum of container-memory-working-set-bytes", net_rx="rate of container-network-receive-bytes-total", net_tx="rate of container-network-transmit-bytes-total",
                         disk_read="rate of container-fs-reads-bytes-total", disk_write="rate of container-fs-writes-bytes-total", request_rate="rate of istio-request-total", error_rate="rate of istio-error-total", latency_p95="mean of istio-latency-95")}
    mapping = pd.DataFrame([dict(system=SYSTEMS[s], aggregate=f, cases_with_aggregate=avail[s][f], cases=n_case[s], in_every_case=avail[s][f] == n_case[s], used_by_detectors=f in feats[s],
                                 source=(src["RE1-OB"] if s == "RE1-OB" else src["other"]).get(f, "not present")) for s in SYSTEMS for f in AGG])
    mapping.to_csv(REP / "rcaeval_feature_mapping.csv", index=False)
    print("aggregates used per system:", mapping.groupby("system").used_by_detectors.sum().to_dict())
    rows, placebo_rng = [], np.random.default_rng(11)
    for c in cases:
        W, inject, ts, te, have = case_windows(c, feats[c["system"]])
        X = W[feats[c["system"]]].to_numpy(float)
        learn = (W.window_end.to_numpy() - ts) <= LEARN_S
        we = W.window_end.to_numpy()
        ov = np.clip(np.minimum(we, te) - np.maximum(W.window_start.to_numpy(), inject), 0, None)
        scored = (~learn) & ~((ov > 0) & (ov < WIN / 2))
        normal = scored & (we <= inject)
        for det, scores, thr in (("RobustZ", V.robust_z_scores(X, learn), th["RobustZ"]), ("IsoForest", V.iforest_scores(X, learn), th["IsoForest"])):
            flags = scores >= thr
            alert = alerts_2of3(np.where(scored, flags, False))
            hit = np.where(alert & scored & (we >= inject) & (we <= inject + HORIZON_S))[0]
            starts = [k for k in np.where(alert & normal)[0] if k == 0 or not alert[k - 1]]
            lo, hi = ts + LEARN_S + WIN, inject - HORIZON_S
            fakes = np.sort(placebo_rng.uniform(lo, hi, PLACEBOS)) if hi > lo else np.array([])
            pl = [bool(((alert & normal) & (we >= f) & (we <= f + HORIZON_S)).any()) for f in fakes]
            rows.append(dict(system=c["system"], fault=c["fault"], service=c["service"], run=c["run"], detector=det, threshold=thr, normal_s=float(inject - (ts + LEARN_S)), detected=bool(len(hit)),
                             delay_s=float(we[hit[0]] - inject) if len(hit) else np.nan, false_alarm_episodes=len(starts), placebo_detected=float(np.mean(pl)) if pl else np.nan,
                             fault_window_flag_share=float(flags[scored & (ov >= WIN / 2)].mean()) if (scored & (ov >= WIN / 2)).any() else np.nan,
                             normal_window_flag_share=float(flags[normal].mean()) if normal.any() else np.nan))
    cr = pd.DataFrame(rows)
    cr.to_csv(REP / "rcaeval_case_results.csv", index=False)
    out = []
    groups = [("system", s, f) for s in SYSTEMS for f in FAULTS] + [("system all faults", s, "all") for s in SYSTEMS]
    for det in ("RobustZ", "IsoForest"):
        for kind, s, f in groups:
            g = cr[(cr.detector == det) & (cr.system == s) & ((cr.fault == f) if f != "all" else True)].reset_index(drop=True)
            n = len(g)
            det_rate = lambda idx: float(g.detected.to_numpy()[idx].mean())
            far = lambda idx: float(g.false_alarm_episodes.to_numpy()[idx].sum() / (g.normal_s.to_numpy()[idx].sum() / 3600.0))
            dl = lambda idx: float(np.nanmedian(g.delay_s.to_numpy()[idx])) if np.isfinite(g.delay_s.to_numpy()[idx]).any() else np.nan
            pl = lambda idx: float(np.nanmean(g.placebo_detected.to_numpy()[idx]))
            r = dict(detector=det, system=SYSTEMS[s], fault_type=f, cases=n, normal_hours_per_case_mean=float(g.normal_s.mean() / 3600.0), detection_rate=det_rate(np.arange(n)), median_delay_s=dl(np.arange(n)),
                     false_alarms_per_hour=far(np.arange(n)), false_alarm_episodes=int(g.false_alarm_episodes.sum()), placebo_detection_rate=pl(np.arange(n)), threshold=float(g.threshold.iloc[0]), horizon_s=HORIZON_S,
                     reliable=bool(n >= 10))
            for name, fn in (("detection_rate", det_rate), ("median_delay_s", dl), ("false_alarms_per_hour", far), ("placebo_detection_rate", pl)):
                r[f"{name}_ci_low"], r[f"{name}_ci_high"] = boot_ci(fn, n, a.boot)
            out.append(r)
    res = pd.DataFrame(out)
    res.insert(0, "generated", time.strftime("%Y-%m-%d %H:%M:%S"))
    res.to_csv(REP / "rcaeval_external_results.csv", index=False, float_format="%.6f")
    print(res[res.fault_type == "all"][["detector", "system", "cases", "detection_rate", "median_delay_s", "false_alarms_per_hour", "placebo_detection_rate"]].round(3).to_string(index=False))
    write_summary()
    print(f"done in {time.time() - t0:.0f} s")




def write_summary(rep=None):
    """rcaeval_summary.md: every number is read from the CSVs written above"""
    rep = rep or REP
    res = pd.read_csv(rep / "rcaeval_external_results.csv")
    cr = pd.read_csv(rep / "rcaeval_case_results.csv")
    mp = pd.read_csv(rep / "rcaeval_feature_mapping.csv")
    f2 = lambda x: "-" if pd.isna(x) else f"{x:.2f}"
    f1 = lambda x: "-" if pd.isna(x) else f"{x:.1f}"
    cell = lambda r, k: f"{f2(r[k])} ({f2(r[k + '_ci_low'])}-{f2(r[k + '_ci_high'])})"
    L = ["# RCAEval external test (testing only)", "",
         "Nothing was trained or tuned on this data. The two unsupervised per-app detectors of the testbed (robust z-score and IsolationForest, each learned from the first 150 s of the case itself) were applied with their alert thresholds FROZEN from the testbed train runs "
         f"(robust z-score {res[res.detector == 'RobustZ'].threshold.iloc[0]:g}, IsolationForest {res[res.detector == 'IsoForest'].threshold.iloc[0]:.4f}) and the frozen alert rule (2 of the last 3 windows). Generated by `python scripts/rcaeval_eval.py`.", "",
         "## Data", "",
         f"RCAEval RE1 (metrics only), Zenodo records 14504480 / 14504481: RE1-OB (Online Boutique), RE1-SS (Sock Shop), RE1-TT (Train Ticket), 390 MB downloaded in total. {int(cr.groupby(['system', 'fault', 'service', 'run']).ngroups)} cases: for each system and fault type (cpu, mem, disk, delay, loss) the runs 1 and 2 of the five services = 10 cases per cell. "
         "Every case has an explicit injection time (`inject_time.txt`), so no assumption about a normal period was needed. Normal data before the injection: Online Boutique " + f"{cr[cr.system == 'RE1-OB'].normal_s.iloc[0] / 60:.0f}" + " min, Sock Shop " + f"{cr[cr.system == 'RE1-SS'].normal_s.iloc[0] / 60:.0f}" + " min, Train Ticket " + f"{cr[cr.system == 'RE1-TT'].normal_s.iloc[0] / 60:.0f}" + " min after the 150 s learning period; windows are 30 s with a 10 s step.", "",
         "## Features available (generic aggregates present in every selected case of a system; `rcaeval_feature_mapping.csv`)", ""]
    for s in mp.system.unique():
        u = mp[(mp.system == s) & mp.used_by_detectors]["aggregate"].tolist()
        n = mp[(mp.system == s) & ~mp.used_by_detectors & (mp.cases_with_aggregate > 0)]["aggregate"].tolist()
        absent = mp[(mp.system == s) & (mp.cases_with_aggregate == 0)]["aggregate"].tolist()
        some = ("; present in only some cases (not used): " + ", ".join(n)) if n else ""
        L.append("- " + s + ": used " + ", ".join(u) + some + "; not present: " + (", ".join(absent) or "none"))
    L += ["", "## Results (detection = an alert within 120 s after the injection; 95% CIs resample cases)", ""]
    for det in ("RobustZ", "IsoForest"):
        L += [f"### {det}", "", "| system | fault | cases | detection rate (CI) | median delay s | false alarms per hour (CI) | placebo detection (CI) |", "|---|---|---|---|---|---|---|"]
        for r in res[res.detector == det].itertuples():
            d = r._asdict()
            L.append(f"| {r.system} | {r.fault_type} | {r.cases} | {cell(d, 'detection_rate')} | {f1(r.median_delay_s)} | {cell(d, 'false_alarms_per_hour')} | {cell(d, 'placebo_detection_rate')} |")
        L.append("")
    rz, iso = res[(res.detector == "RobustZ") & (res.fault_type == "all")], res[(res.detector == "IsoForest") & (res.fault_type == "all")]
    L += ["## Reading the numbers", "",
          f"- Robust z-score: detects {rz.detection_rate.min():.0%} to {rz.detection_rate.max():.0%} of the faults per system (placebo {rz.placebo_detection_rate.min():.0%} to {rz.placebo_detection_rate.max():.0%}) with {rz.false_alarms_per_hour.min():.1f} to {rz.false_alarms_per_hour.max():.1f} false alarms per hour in the normal period.",
          f"- IsolationForest: detects {iso.detection_rate.min():.0%} to {iso.detection_rate.max():.0%} but also alerts at the placebo times in {iso.placebo_detection_rate.min():.0%} to {iso.placebo_detection_rate.max():.0%} of the cases and raises {iso.false_alarms_per_hour.min():.1f} to {iso.false_alarms_per_hour.max():.1f} false alarms per hour, so most of its detections are not specific to the fault.",
          "- Cells have 10 cases each; false-alarm rates in Sock Shop and Train Ticket come from only a few minutes of normal data per case and are coarse. Delays are on a 10 s grid.",
          "- The testbed's alert thresholds were chosen on 14 testbed-specific signals; here the signals are different aggregates, so this is a test of how far the same unsupervised recipe carries, not a fair benchmark of root-cause analysis.", "",
          "## Supervised testbed models", "",
          f"Not applied. The frozen random forest takes 78 inputs (39 testbed features and their baseline-relative forms: per-request statistics, log counts, thread counts, event-loop lag, database-pool wait, in-flight requests). RCAEval has at most {int(mp.groupby('system').used_by_detectors.sum().max())} generic aggregates per system (CPU, memory, request rate, latency, errors, network and disk where present) with different meaning and units, and none of the log, thread, loop-lag or pool signals. Mapping them onto the model's inputs would mean inventing equivalences, so no supervised result is reported; retraining on this data was not done because nothing may be tuned on external data.", "",
          "## Limits", "", "Public benchmark, 150 cases from three systems, RE1 metrics only (no traces or logs), unsupervised detectors with frozen thresholds from another application: a first look at transfer, not evidence that FailSight works on other systems."]
    (rep / "rcaeval_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if "--summary-only" in sys.argv:
        write_summary()
    else:
        main()
