"""LO2 step 1: inspect the structure, print the first lines of every file type, and copy a contiguous ~100 MB sample into processed/lo2_sample/.
LO2 (..\\LO2) is read-only: it is only read here (tar members are streamed out, nothing under LO2 is written or modified).

Sampling rule (also written to processed/lo2_sample_manifest.csv and report_outputs/lo2_sample_manifest.csv):
  * one whole run (the earliest run_id, i.e. the smallest tar epoch);
  * inside it the first K scenarios in start-time order (scenario_order_in_run = 1..K): a contiguous block of whole scenarios starting
    with `correct` (always order 1), K = the largest value for which the copied files stay <= 100 MB;
  * per scenario only the files the feature mapping uses are copied: the 7 service logs, the MySQL log and the node_exporter metric
    series whose names also exist among the AnoMod metric features. Locust, traces, jaeger/prometheus logs and the other ~466 metric files
    are not copied (not used by the mapping, see lo2_feature_mapping.csv).
"""
import sys, re, tarfile, csv, io
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LO2 = ROOT.parent / "LO2"
SAMPLE = ROOT / "processed" / "lo2_sample"
REP = ROOT / "report_outputs"
BUDGET = 100e6
SVC = re.compile(r"^light-oauth2-oauth2-(client|code|key|refresh-token|service|token|user)-1\.log$")


def overlapping_metric_names():
    fl = pd.read_csv(ROOT / "inputs" / "combined_feature_list.csv")
    m = fl[fl.group == "metrics"].feature.str.replace(r"^m_", "", regex=True).str.replace(r"__(mean|max|std)$", "", regex=True)
    return sorted(set(m))


def head_lines(tf, member, n=3, width=230):
    f = tf.extractfile(member)
    out = []
    for _ in range(n):
        line = f.readline()
        if not line:
            break
        out.append(line.decode("utf-8", "replace").rstrip("\r\n")[:width])
    return out


def main():
    SAMPLE.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(LO2 / "lo2_subset.csv", usecols=["run_id", "scenario", "is_error", "scenario_order_in_run", "scenario_start_epoch", "sec_since_scenario_start", "scenario_n_points"],
                       dtype={"run_id": str})
    runs = sorted(meta.run_id.unique())
    run = runs[0]
    sc = meta[meta.run_id == run].groupby("scenario").agg(order=("scenario_order_in_run", "first"), start_epoch=("scenario_start_epoch", "first"), is_error=("is_error", "first"),
                                                        n_points=("scenario_n_points", "first"), duration_s=("sec_since_scenario_start", "max")).reset_index().sort_values("order")
    tar_path = LO2 / "lo2_tar" / f"light-oauth2-data-{run}.tar"
    names = overlapping_metric_names()
    want_metric = {f"metrics/metric_{n}.json" for n in names}
    lines = [f"LO2 inspection (read-only). runs in lo2_subset.csv: {len(runs)} ({runs[0]} .. {runs[-1]}); raw tar per run in LO2/lo2_tar ({tar_path.stat().st_size/1e6:.0f} MB each)",
             f"lo2_subset.csv: {len(meta)} rows (5 s scrape rows), 54 scenarios per run; scenario `correct` = order 1 in every run ({int((meta.groupby('run_id').scenario_n_points.first()>0).sum())} runs)",
             f"selected run: {run} (rule: earliest run_id)", ""]
    rows, info = [], []
    with tarfile.open(tar_path) as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        by_sc = {}
        for m in members:
            p = m.name.split("/")
            by_sc.setdefault(p[1], []).append((("/".join(p[2:])), m))
        # ---- 1. first lines of each file type (one `correct` and one error scenario)
        shown = set()
        for scen in ("correct", "get_token_page_400_no_page"):
            for rel, m in sorted(by_sc[scen]):
                kind = ("service log" if SVC.match(rel) else "mysql log" if "mysqldb" in rel else "locust.log" if rel == "locust.log" else "metric json" if rel.startswith("metrics/metric_")
                        else "traces csv" if rel.startswith("metrics/traces_") else "last_fetch_time" if rel.endswith("last_fetch_time.txt") else "other log (jaeger/prometheus/node_exporter)")
                if (kind, scen) in shown:
                    continue
                shown.add((kind, scen))
                hl = head_lines(tf, m, 6 if kind == "metric json" else 3)
                lines.append(f"--- {kind}  [{scen}/{rel}, {m.size} bytes]")
                lines += ["    " + h for h in hl] or ["    (empty file)"]
        # ---- 2. choose K scenarios
        cum, stopped = 0.0, False
        for _, r in sc.iterrows():
            sel = [(rel, m) for rel, m in by_sc[r.scenario] if SVC.match(rel) or "mysqldb" in rel or rel in want_metric]
            size = sum(m.size for _, m in sel)
            take = (not stopped) and cum + size <= BUDGET
            if not take:
                stopped = True                       # the block must stay contiguous: nothing after the first scenario that does not fit
            rows.append(dict(run_id=run, scenario=r.scenario, scenario_order_in_run=int(r.order), start_epoch=int(r.start_epoch), is_error=int(r.is_error), n_scrape_points=int(r.n_points),
                             duration_s=int(r.duration_s), files_selected=len(sel), bytes_selected=size, cumulative_bytes_if_taken=cum + size, copied=bool(take),
                             reason=("contiguous block from the run start, within the 100 MB budget" if take else "not copied: 100 MB budget reached (block stays contiguous)")))
            if not take:
                continue
            cum += size
            for rel, m in sel:
                dest = SAMPLE / run / r.scenario / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(m) as src, open(dest, "wb") as out:
                    out.write(src.read())
    man = pd.DataFrame(rows)
    man.to_csv(ROOT / "processed" / "lo2_sample_manifest.csv", index=False)
    man.to_csv(REP / "lo2_sample_manifest.csv", index=False)
    taken = man[man.copied]
    lines += ["", f"SAMPLE: run {run}, scenarios order 1..{int(taken.scenario_order_in_run.max())} ({len(taken)} scenarios, {int(taken.is_error.sum())} error + {int((taken.is_error == 0).sum())} correct), "
                  f"{taken.bytes_selected.sum()/1e6:.1f} MB copied to processed/lo2_sample/{run}/ ({taken.files_selected.sum()} files). "
                  f"Time span: epoch {int(taken.start_epoch.min())} .. {int(taken.start_epoch.max() + taken.duration_s.iloc[-1])}",
              f"AnoMod metric names also present as LO2 metric files (copied): {', '.join(sorted(Path(p).name[7:-5] for p in {str(r) for r in (SAMPLE / run / 'correct' / 'metrics').glob('metric_*.json')}))}"]
    (REP / "lo2_inspection.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
