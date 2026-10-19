# Early warning and within-run (segment) labels: feasibility

Status: **not feasible with the data in AnoMod. No early-warning model and no segment-label test was built, because the exact fault start and end times do not exist in the data.** Nothing here is estimated or guessed. The evidence below is produced by `python scripts/fault_times.py` (table: `report_outputs/fault_times.csv`) and is shown in the proofs report.

## What was searched (read-only)

For each of the 26 runs (13 SocialNetwork, 13 TrainTicket) every file in the run's metric, log, API-response and trace folders was checked: names, metadata files, summaries, collection reports, API-probe files, Kubernetes event files, trace files. The coverage folders (code-coverage artefacts, tens of thousands of files) hold no timestamps of faults and were not searched. Text-like files were scanned for injection / chaos keywords (`inject`, `chaos`, `stress-ng`, `iptables`, `tc qdisc`, `chaosblade`, `fault start`, `experiment_start`).

## What exists, run by run

| Item | SocialNetwork (13 runs) | TrainTicket (13 runs) | Use for fault times |
|---|---|---|---|
| Experiment start time | in the folder name (`..._20251104_024819`) | in the folder name (`..._20251103T185917Z_em`) | the start of the experiment, **not** of the fault |
| Collection end time | not recorded | `log_collection_report_*.json`, `collection_time` | end of the experiment, **not** of the fault |
| Injection log, manifest or chaos-tool record | none | none | - |
| Files mentioning an injection keyword | 0 of 35-36 per run | 0 of 58-59 per run, except 2 files of one run whose scenario is named `exception_injection` (the keyword is that scenario name inside a path or file name) | no fault time in them |
| API responses | one 60 s probe near the **end** of the run (`response_summary.json`) | requests during the **last 3.7 to 12.4 minutes** of the run | cover only the end of a run |
| Kubernetes events | - | the same cluster events (1,469-1,481 inside each run's period, including pods stopped and started at the start of every run, the Normal run too) | not fault events |
| Fault start, fault end | **not recorded** | **not recorded** | - |

Reliability of fault start and end times: **none**. The only absolute times are the experiment start (folder name) and the end of collection.

## What the data do show (descriptive, not fault times)

- **Labels are per run, not per window.** All windows of a run carry the run's single label (one Normal run per system, 126 and 80 windows).
- **Every run begins with a transient.** In every run, including the Normal runs, the first windows of the generic metric aggregates differ from the last 30% of the same run. The share of windows that differ is 0.34 for the SocialNetwork Normal run and 0.75 for the TrainTicket Normal run (`descriptive_metric_excursion_share` in `fault_times.csv`), and the first such window is the first window of the run in all 26 runs. Metric excursions therefore cannot be used to locate where a fault starts: they would mostly mark the start-up of the run.
- **API data are almost absent in the recordings.** The API probe runs only at the end of a run (see the table above), so the windows before it have no request data.

## Why nothing was fabricated

- A "fault window" derived from the metrics themselves (a change point, a threshold on CPU) would make the labels a function of the features. Any model trained on those labels would look good by construction and say nothing about detecting real faults.
- Early-warning labels ("a failure starts within the next N minutes", N = 2, 5, 10) need the fault start time. Without it there is no "before" and no "lead time" to measure.
- A segment-label test (Normal before the fault, fault class during it, Normal after it) needs both times. Without them it cannot be built.

## What would make it possible

1. A timestamped record from the fault-injection tool (start and stop of every injection) stored with each run.
2. A normal period before each injection, and a recovery period after it, inside the same run.
3. Probes and request logs that cover the whole run, not only its end.
4. Labels per window or per segment instead of one label per run.

With these, the early-warning model and the within-run test described in the project plan could be built and measured with run-aware blocks. Until then, the honest answer to "can FailSight give early warning?" is: **not shown, and not testable on AnoMod**.
