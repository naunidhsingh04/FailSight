"""Writes docs/TESTBED_DATASET.md (dataset description: generation, schema, labels, class counts, imbalance table, sample rows, figures) and docs/figures/testbed_*.png from
processed/testbed_windows.parquet and the run folders. Every number in the document is computed here.   python -m testbed.dataset_doc"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from testbed import dataset as DS  # noqa: E402

DOCS = ROOT / "docs"
FIG = DOCS / "figures"
LEVELS = ["Normal", "Performance", "Service", "Database", "Code"]
plt.rcParams.update({"font.family": "serif", "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})
GREY = ["#6b6b6b", "#2b5d8a", "#9b3d2e", "#5a7d3a", "#7d5200"]


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def figures(W):
    FIG.mkdir(parents=True, exist_ok=True)
    # 1 class balance
    c = W[W.transition == 0].label_level.value_counts().reindex(LEVELS)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(c.index, c.values, color=GREY)
    for i, v in enumerate(c.values):
        ax.text(i, v + 15, str(v), ha="center", fontsize=9)
    ax.set_ylabel("windows")
    ax.set_title("Windows per class (transition windows excluded)")
    fig.tight_layout()
    fig.savefig(FIG / "testbed_class_balance.png")
    plt.close(fig)
    # 2 an example run with its faults
    run = W[W.fault_type != ""].groupby("run_id").fault_type.nunique().sort_values(ascending=False).index[0]
    d = W[W.run_id == run].sort_values("window_index")
    sched = pd.read_csv(ROOT / "testbed" / "runs" / run / "fault_schedule.csv")
    t0 = json.loads((ROOT / "testbed" / "runs" / run / "meta.json").read_text(encoding="utf-8"))["start_ts"]
    fig, axs = plt.subplots(3, 1, figsize=(7, 5.2), sharex=True)
    for ax, col, lab in zip(axs, ["api_latency_p95_ms", "m_process_cpu_percent__mean", "api_error_rate"], ["p95 latency (ms)", "process CPU (%)", "error rate"]):
        ax.plot(d.t_offset, d[col], color="#333", lw=1.2)
        ax.set_ylabel(lab, fontsize=8)
        for r in sched.itertuples():
            ax.axvspan(r.start_ts - t0, r.end_ts - t0, color=GREY[LEVELS.index(r.klass)], alpha=0.25)
            if pd.notna(r.failure_ts):
                ax.axvline(r.failure_ts - t0, color="#9b3d2e", ls="--", lw=1)
    axs[0].set_title(f"{run}: shaded = recorded fault (dashed = failure time of a gradual fault)", fontsize=9)
    axs[-1].set_xlabel("seconds since the start of the run")
    for r in sched.itertuples():
        axs[0].text((r.start_ts - t0), axs[0].get_ylim()[1] * 0.92, r.fault_type, fontsize=7, color="#222")
    fig.tight_layout()
    fig.savefig(FIG / "testbed_example_run.png")
    plt.close(fig)
    # 3 features by class
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    D = W[(W.transition == 0)]
    for ax, col, lab in zip(axs, ["api_latency_p95_ms", "m_process_cpu_percent__mean", "api_error_rate"], ["p95 latency (ms, log scale)", "process CPU (%)", "error rate"]):
        data = [D[D.label_level == k][col].dropna().to_numpy() for k in LEVELS]
        ax.boxplot(data, tick_labels=[k[:4] for k in LEVELS], showfliers=False)
        ax.set_title(lab, fontsize=9)
        if "latency" in col:
            ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(FIG / "testbed_features_by_class.png")
    plt.close(fig)
    # 4 fault durations and severity
    S = pd.concat([pd.read_csv(f) for f in sorted((ROOT / "testbed" / "runs").glob("r*/fault_schedule.csv"))], ignore_index=True)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    order = list(S.fault_type.value_counts().index)
    bottom = np.zeros(len(order))
    for lv, col in zip(["mild", "medium", "severe"], ["#9aa9b8", "#5a7d9a", "#2b3d4f"]):
        v = np.array([((S.fault_type == t) & (S.severity_level == lv)).sum() for t in order])
        axs[0].bar(order, v, bottom=bottom, color=col, label=lv)
        bottom += v
    axs[0].set_xticks(range(len(order)))
    axs[0].set_xticklabels(order, rotation=60, ha="right", fontsize=7)
    axs[0].legend(fontsize=7)
    axs[0].set_title("Fault instances by type and severity", fontsize=9)
    axs[1].hist(S.duration_s, bins=14, color="#6b6b6b")
    axs[1].set_title("Fault duration (s)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "testbed_faults.png")
    plt.close(fig)
    return run, S


def write():
    W = pd.read_parquet(DS.OUT)
    base, rel = DS.feature_columns(W)
    run, S = figures(W)
    meta = [json.loads((d / "meta.json").read_text(encoding="utf-8")) for d in sorted(DS.RUNS.glob("r*")) if (d / "DONE").exists()]
    runs = pd.DataFrame([dict(run_id=m["run_id"], profile=m["profile"], rate=m["rate"], faults=len(m["faults"])) for m in meta])
    used = W[W.transition == 0]
    cnt = used.label_level.value_counts().reindex(LEVELS)
    tab = pd.DataFrame({"class": LEVELS, "windows": cnt.values, "share": [f"{v / cnt.sum():.1%}" for v in cnt.values],
                        "imbalance vs Normal": [f"{cnt['Normal'] / v:.1f} : 1" if k != "Normal" else "1 : 1" for k, v in zip(LEVELS, cnt.values)],
                        "runs with this class": [used[used.label_level == k].run_id.nunique() for k in LEVELS]})
    bin_ = used.label_binary.value_counts().sort_index()
    by_type = used[used.fault_type != ""].groupby(["fault_type", "fault_kind"]).agg(windows=("window_id", "size"), instances=("fault_id", "nunique")).reset_index()
    by_sev = used[used.severity_level != ""].groupby("severity_level").agg(windows=("window_id", "size"), instances=("fault_id", "nunique")).reindex(["mild", "medium", "severe"]).reset_index()
    pre = pd.DataFrame([{"target": f"pre-failure within {n} min", "windows": int(W[f"prefail_{n}m"].sum()), "runs": int(W[W[f"prefail_{n}m"] == 1].run_id.nunique())} for n in (1, 2, 3)])
    show = ["window_id", "label_level", "fault_type", "severity_level", "api_request_count", "api_error_rate", "api_latency_p95_ms", "log_error_count", "m_process_cpu_percent__mean", "m_process_rss_bytes__mean", "m_event_loop_lag_ms__mean", "m_db_pool_wait_ms__mean"]
    sample = pd.concat([used[used.label_level == k].sample(2, random_state=1) for k in LEVELS])[show].round(3)
    groups = {"api": [c for c in base if c.startswith("api_")], "log": [c for c in base if c.startswith("log_")], "process metrics": [c for c in base if c.startswith("m_")], "changes (against the previous non-overlapping window)": [c for c in base if c.startswith("d_")]}
    schema = pd.DataFrame([{"group": g, "features": len(v), "names": ", ".join(f"`{x}`" for x in v[:6]) + (" ..." if len(v) > 6 else "")} for g, v in groups.items()])
    n_runs, n_free = len(meta), int((runs.faults == 0).sum())
    n_chk, n_bad = DS.prove_no_future(W, base)
    text = f"""# Testbed dataset

Generated by `python -m testbed.dataset_doc` from the raw events of {n_runs} recorded runs. Every number below is computed from the data.

## What it is

A small real ticket-shop application (FastAPI + SQLite, 5 endpoints: search, book, pay, cancel, health) run on one laptop, monitored only through the `failsight_agent` package, driven by an asyncio load generator, with **real faults injected into the running app at recorded times**. Unlike AnoMod (one label per run, no fault times), every window here can be labelled from an exact fault schedule.

- **Runs:** {n_runs} runs of 10 minutes, one at a time. {n_free} runs have no fault; the others have 1-3 faults ({len(S)} in total). The first 150 s of every run are fault-free.
- **Traffic:** a random profile per run (steady, bursty, ramp, daily-like; {', '.join(f'{p} {int((runs.profile == p).sum())}' for p in ['steady', 'bursty', 'ramp', 'daily'])} runs), a random request rate (3 to 12 requests per second), and a random mix of user behaviour (browse / purchase / abandon / refund), so that load changes alone are not faults.
- **Faults (10 types, 4 classes):** Performance: `latency`, `cpu_burn`, `latency_ramp` (gradual). Service: `route_500`, `crash`, `pool_exhaustion` (gradual). Database: `db_slow`, `db_lock`. Code: `exception_route`, `leak` (gradual). Each instance has a random severity (mild / medium / severe) drawn from the run's seed and mapped to its parameters (for example latency +30 ms to +800 ms). Gradual faults grow until the app fails at a recorded failure time.
- **Windows:** 30 s long, a new one every 10 s, only windows that lie fully inside the run: {len(W)} windows ({W.groupby('run_id').size().iloc[0]} per run).

![class balance](figures/testbed_class_balance.png)

![an example run](figures/testbed_example_run.png)

![faults](figures/testbed_faults.png)

## Features ({len(base)} per window, plus {len(rel)} baseline-relative versions)

{md_table(schema)}

The baseline-relative version `rel__<feature>` is a robust z-score against the run's own past only (the last 60 windows that ended before the window began, at least 9; clipped to +-10). No label, no other run and no future window enters it; `python -m testbed.dataset` checks this by overwriting all later windows with garbage ({n_chk} windows checked, {n_bad} differences).

## Labels

- `label_level`: **Normal**, or the class of the fault (Performance / Service / Database / Code) that overlaps the window for at least half of it (15 s). `label_binary`: 0 Normal, 1 any fault.
- `fault_type`, `severity_level`, `fault_kind` (abrupt or gradual) and `fault_id` of that fault.
- `transition` = 1 when a fault overlaps the window for less than half: the label is ambiguous. These {int(W.transition.sum())} windows keep the label Normal as specified, are flagged, and are **left out of training and evaluation** ({len(used)} windows remain).
- Gradual faults: `time_to_failure_s` = failure time minus the window end, and `prefail_1m`, `prefail_2m`, `prefail_3m` = the window ends within 1, 2 or 3 minutes before the failure.
- `in_learning` = the window ends within the first 150 s of its run (fault-free by construction).

## Class counts

{md_table(tab)}

Binary: Normal {int(bin_.get(0, 0))}, fault {int(bin_.get(1, 0))} (imbalance {bin_.get(0, 0) / max(bin_.get(1, 0), 1):.1f} : 1).

By fault type (windows labelled with that type):

{md_table(by_type)}

By severity level:

{md_table(by_sev)}

Pre-failure targets (gradual faults only):

{md_table(pre)}

## Sample rows (2 per class)

{md_table(sample.reset_index(drop=True))}

![features by class](figures/testbed_features_by_class.png)

## Limits

One application, one machine, synthetic traffic and faults injected by us: the data show what the monitor can see in this app, not in production systems. Windows overlap by 20 s, so neighbouring windows are not independent; all validation therefore splits by whole run. The fault effects are real (the app sleeps, burns CPU, fails, locks its database, leaks memory) but they are the effects we chose to implement.
"""
    DOCS.mkdir(exist_ok=True)
    (DOCS / "TESTBED_DATASET.md").write_text(text, encoding="utf-8")
    sample.to_csv(ROOT / "report_outputs" / "testbed_dataset_sample.csv", index=False)
    print("wrote docs/TESTBED_DATASET.md,", len(list(FIG.glob("testbed_*.png"))), "figures")
    print(tab.to_string(index=False))


if __name__ == "__main__":
    write()
