"""Part 3: what does the raw AnoMod data say about WHEN each fault starts and ends?

Looks (read-only) in every place that could hold a time: scenario folder names, run metadata files, log-collection reports, summary files, API-probe files, Kubernetes
events, trace files, and the injection / chaos keywords. Writes report_outputs/fault_times.csv (one row per run) and prints what exists and how reliable it is.
The result decides whether an early-warning model or a within-run (segment) test can be built honestly: see report_outputs/early_warning_feasibility.md.

python scripts/fault_times.py
"""
import glob
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
ANOMOD = ROOT.parent / "AnoMod"
REP = ROOT / "report_outputs"
OUT = REP / "fault_times.csv"
KEYWORDS = re.compile(r"inject|chaos|stress-ng|fault_start|fault start|experiment_start|iptables|tc qdisc|chaosblade", re.I)


def folder_start(name):
    """experiment start time encoded in the run folder name: SN '<scenario>_YYYYMMDD_HHMMSS', TT '<scenario>_YYYYMMDDTHHMMSSZ_em' (UTC, as the metrics)"""
    m = re.search(r"_(\d{8})_(\d{6})(?:_|$)", name)
    if m:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    m = re.search(r"_(\d{8}T\d{6})Z", name)
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S") if m else None


def scenario_of(name):
    return re.sub(r"(_em)?_\d{8}[T_]\d{6}Z?(_.*)?$", "", name)


def sniff(paths, limit_bytes=3_000_000):
    """number of files (of the text-like ones, at most limit_bytes read from each) that mention an injection / chaos keyword"""
    hits = 0
    for p in paths:
        try:
            if p.suffix.lower() not in (".txt", ".json", ".jsonl", ".log", ".csv", ".md", ".yaml", ".yml"):
                continue
            with open(p, "rb") as f:
                head = f.read(limit_bytes).decode("utf-8", "ignore")
            if KEYWORDS.search(head):
                hits += 1
        except OSError:
            pass
    return hits


def excursion_share(w_run, cols):
    """DESCRIPTIVE ONLY (not a fault time): share of a run's windows whose generic metrics sit far from the run's own last 30% of windows (robust z > 5 in any aggregate)"""
    n = len(w_run)
    ref = w_run.iloc[int(n * 0.7):]
    z = []
    for c in cols:
        r = ref[c].to_numpy(float)
        r = r[np.isfinite(r)]
        if len(r) < 5:
            continue
        m = np.median(r)
        s = max(1.4826 * np.median(np.abs(r - m)), 0.05 * abs(m), 1e-12)
        z.append(np.abs((w_run[c].to_numpy(float) - m) / s))
    if not z:
        return np.nan, np.nan
    zz = np.nanmax(np.vstack(z), axis=0)
    ex = zz > 5
    return float(ex.mean()), (float(np.where(ex)[0][0] / n) if ex.any() else np.nan)


def main():
    import preprocess as P
    w = P.load_windows().sort_values(["run_id", "window_index"])
    g = pd.read_parquet(ROOT / "processed" / "generic_metric_windows.parquet")
    w = w.merge(g, on="window_id")
    gcols = [c for c in g.columns if c.endswith("__mean")]
    rows = []
    for system, base in (("SN", ANOMOD / "SN_data"), ("TT", ANOMOD / "TT_data")):
        folders = {}
        for sub in ("metric_data", "log_data", "api_responses", "trace_data"):
            for d in sorted((base / sub).glob("*")):
                if d.is_dir():
                    folders.setdefault(scenario_of(d.name), {})[sub] = d
        for run, grp in w[w.system == system].groupby("run_id"):
            scen = grp.scenario.iloc[0]
            f = folders.get(scen, {})
            row = dict(system=system, run_id=run, scenario=scen, label_level=grp.label_level.iloc[0], n_windows=len(grp))
            st = folder_start(next(iter(f.values())).name) if f else None
            row["experiment_start_from_folder_name"] = st.isoformat(sep=" ") if st else ""
            row["first_window_start_utc"] = str(grp.window_start.min())
            row["last_window_end_utc"] = str(grp.window_end.max())
            row["run_minutes"] = round((grp.window_end.max() - grp.window_start.min()).total_seconds() / 60, 1)
            files = [p for d in f.values() for p in d.rglob("*") if p.is_file()]
            row["files_checked"] = len(files)
            row["files_with_injection_keyword"] = sniff(files)
            end = ""
            for r in glob.glob(str(f.get("log_data", Path("x")) / "log_collection_report_*.json")):
                end = json.load(open(r, encoding="utf-8")).get("collection_time", "")
            row["collection_end_from_log_report"] = end
            probe = ""
            if system == "SN" and "api_responses" in f:
                for r in glob.glob(str(f["api_responses"] / "response_summary.json")):
                    ci = json.load(open(r, encoding="utf-8")).get("collection_info", {})
                    probe = f"{ci.get('start_time', '')} for {ci.get('duration_seconds', '')} s"
            elif system == "TT" and "api_responses" in f:
                for r in glob.glob(str(f["api_responses"] / "*" / "api_responses.jsonl")):
                    ts = [json.loads(line)["timestamp"] for line in open(r, encoding="utf-8")]
                    probe = f"{min(ts)} to {max(ts)}"
            row["api_probe_period"] = probe
            row["fault_start"] = "not recorded"
            row["fault_end"] = "not recorded"
            row["reliability"] = "none: no fault start/end time exists in the data"
            sh, first = excursion_share(grp, gcols)
            row["descriptive_metric_excursion_share"] = sh
            row["descriptive_first_excursion_position"] = first
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, float_format="%.4f")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 38)
    print(df[["system", "scenario", "label_level", "experiment_start_from_folder_name", "collection_end_from_log_report", "run_minutes", "files_checked", "files_with_injection_keyword", "api_probe_period", "fault_start", "fault_end"]].to_string(index=False))
    print("\ndescriptive only (NOT fault times): share of each run's windows far from the run's last 30% (generic metrics), and where the first such window sits in the run")
    print(df[["system", "scenario", "label_level", "descriptive_metric_excursion_share", "descriptive_first_excursion_position"]].to_string(index=False))
    return df


if __name__ == "__main__":
    main()
