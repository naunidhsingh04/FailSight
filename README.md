# FailSight

**An explainable early-warning monitor for microservice failures, built on the AnoMod dataset.**

FailSight looks at the last 30 seconds of an application's activity (API calls, log lines and system measurements), every 10 seconds, and says how likely it is that something is going wrong *right now*. When it raises an alert it shows, in plain words, which signals pushed the risk up.

> **Replay of recorded AnoMod data.** The Console in this repository does not watch a live system. It replays recorded AnoMod test data through a real ingestion pipeline (the same code a live collector would call). It is a research prototype and a demonstration, **not a live production deployment**. The app says this on every page.

## What it can and cannot do

| It can | It cannot |
|---|---|
| Turn 30-second windows of API, log and metric data into a failure risk and a Healthy / Watch / Failing status | Forecast a failure or say when one will happen: it judges what is in the current window |
| Explain an alert with the signals that moved most (SHAP, shown as plain sentences) | Prove a root cause: the reasons are what the model reacted to |
| Show every result with its 95% interval and re-check every number (Proofs page) | Work on a different system: on the external LO2 test every saved model scored balanced accuracy 0.500 (see Limits) |

Known weak spot, stated plainly: on healthy recordings the best binary model's recall is only 0.556, so it raises many false alarms. All limits are on the app's **Limits** page and in `report_outputs/`.

## Run it (Windows first, no Docker needed)

You need Python 3.11+ and Node 20+.

**What a fresh clone contains and what it does not.** Git holds the code, the result files in `report_outputs/` (including the proof report), the testbed run schedules and the documents. It does **not** hold data or trained models: the AnoMod / LO2 data, `models/`, `processed/*.joblib` and `processed/*.parquet`, the raw events of the testbed runs, `external/`. Without the AnoMod models the backend still starts, in **testbed-only mode** (`/api/health` says `testbed-only`): the Proofs, Testbed, Limits and Connect pages and the live testbed monitor work, while the AnoMod replay console and the Results page answer with a clear 503 message. To get the AnoMod models you need the AnoMod data and `scripts/full_pipeline.py` (see [app/README.md](app/README.md)); to get the testbed detector you generate the testbed runs yourself (see below).

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r app\backend\requirements.txt -r app\backend\requirements-deep.txt
cd app\frontend; npm install; npm run build; cd ..\..
python -m app.backend.serve          # http://127.0.0.1:8000 (loads the models: 30-60 s)
```

The first install pulls TensorFlow (large); for the testbed and the dashboard alone `pip install -r app\backend\requirements.txt` is enough (set `FAILSIGHT_LOAD_DEEP=0` to skip the deep models when you have the AnoMod models). Run the tests with `pip install -r app\backend\requirements-dev.txt` and `python -m pytest tests app\backend\tests` (tests that need data or models are skipped with a reason).

**Reproduce the testbed from a clean clone (no data needed, about 7 hours of laptop time):** `python -m testbed.run_generator --runs 40 --minutes 10`, then `python -m testbed.dataset`, `python -m testbed.validate`, `python -m testbed.freeze`, then `start_testbed.bat`. The generated runs will differ from ours in timing details, so numbers will not match `report_outputs/` exactly.

Open the address, choose a scenario, pick a speed (1x, 10x, 60x) and press **Play**. On macOS or Linux use `python3` and `python -m uvicorn app.backend.main:app --port 8000`.

Use the **Simple / Detailed** switch in the top bar: Simple uses plain language; Detailed adds model names, thresholds, tables and raw values. Every technical word has a "?" help button and an entry in the Glossary.

The replay can also be run from a terminal, without the browser:

```powershell
python scripts\replay_scenario.py --scenario Service --speed 60 --server http://127.0.0.1:8000
python scripts\replay_scenario.py --scenario Normal --dry-run     # show the plan only
```

## Proofs: check every number yourself

```powershell
python scripts\proofs.py              # all sections P1-P10 and P13 (about 10 minutes; loads every saved model)
python scripts\proofs.py --quick      # data, split, folds, versions only
python scripts\proofs.py --only P5    # one section; it is merged into report_outputs\proofs.json
python scripts\leakage_audit.py       # the leakage audit (C1a-C7) that P7 reports
python scripts\missingness_check.py  # masking / indicator-only / no-shortcut checks (about 9 minutes) -> report_outputs\missingness_check.csv, reported as P13
```

The script prints a PASS / WARN / FAIL table and exits non-zero if any check fails. The app's **Proofs** page shows the same report (`report_outputs/proofs.json`), can re-run the checks and stream the terminal output, and downloads the report as JSON or a printable page.

Last run: 48 PASS, 3 WARN, 0 FAIL, plus 2 items marked **REPORTED ACCURATELY** (the LO2 external test, where the result is a failure and the check is only that it is reported faithfully; the recomputation of the LO2 numbers stays PASS). P13 (missingness and shortcuts, section numbers 11 and 12 are not used) found that masking, an indicator-only classifier and removing the was_missing_* indicators and the system flag do not explain the saved model's result; it also found that the saved XGBoost rests on the metric readings (masking API or log values changes nothing) and that missing metric readings push its call toward Anomaly.  The warnings are reported, not hidden:

- **P5, ranking metrics.** Decision metrics of all 36 saved models are recomputed and equal `test_metrics.csv` within 1e-6. ROC-AUC / PR-AUC of the forest models differ in the 3rd decimal because tied scores are ordered by floating-point noise; the probabilities themselves match to 1e-16.
- **C3, time overlap.** Some training windows overlap in time (up to 20 s) with a test window of the same run. This is real; it stays a WARN. Measured effect of removing them: -0.036 balanced accuracy (binary) and +0.010 (five-class), on average.

**What the file-integrity check proves.** The audit's check C7 hashes the AnoMod and LO2 files (SHA-256). That baseline was **first recorded during the audit run itself**, so it proves that no file has changed *since then*, not before.

## The testbed: our own shop, faults at recorded times, whole runs held out

AnoMod has one label per run and no fault times, so it cannot show whether failures are really detected (proofs P14, P15). The **testbed** answers that for one app: a small real ticket shop (`testbed/shop`, FastAPI + SQLite) monitored only through the `failsight_agent` package (`agent/`), driven by a load generator (steady, bursty, ramp and daily profiles, random user mixes and rates) and broken with real faults injected at exact, recorded times (10 fault types in 4 classes, random severity from mild to severe, some gradual with a known failure time).

```powershell
python -m testbed.run_generator --runs 40 --minutes 10   # 40 labelled runs, resumable (about 7 hours); invalid runs (laptop asleep, stalls) are set aside and redone
python -m testbed.dataset ; python -m testbed.dataset_doc # windows, labels, docs\TESTBED_DATASET.md
python -m testbed.validate                               # run-held-out validation -> report_outputs\testbed_*.csv
python -m testbed.freeze                                 # freeze the detector and the alert rule BEFORE the live demos
python -m testbed.live_demo --scenario all               # three 12-minute live demos on new seeds (9001-9003), schedule written first
start_testbed.bat                                        # one command: shop + storefront + monitor + load generator, opens the shop and the monitor
```

**The shop, its storefront and the monitor (three addresses).** `start_testbed.bat` (or `python -m testbed.launcher`) starts everything and opens two browser tabs. Anything that is already running on its port is reused.

| Address | What it is |
|---|---|
| http://127.0.0.1:5000 | **The shop**: *Marigold Coaches*, a fictional coach-ticket site (home, search, journey page, cart, checkout, confirmation). It only uses the shop's own endpoints; the shop's code, timings and telemetry are unchanged. The only mention of FailSight is the footer line "Fictional shop used to test FailSight". |
| http://127.0.0.1:5000/ops | **Presenter controls**, only reachable from this computer: plain buttons that really break the shop (slow pages, slow database, page crash, server crash, surprise me) and **Fix it**. The only place where the fault names appear. |
| http://127.0.0.1:8000 | **The live monitor** (FailSight): shop status in plain words, a link "Open the shop", the answer and a response-time chart. Also *Start the shop* if the shop is not running. |

Presenter flow (5 steps): 1. open the shop; 2. open the monitor; 3. wait about 2 minutes while the monitor learns what normal looks like; 4. break something on `/ops`; 5. watch the monitor raise an alert and say why. The full script is in [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md); the decisions behind the storefront are in [report_outputs/storefront_log.md](report_outputs/storefront_log.md), and `python scripts\smoke_storefront.py` repeats the 3-minute check that the storefront left the shop's telemetry unchanged ([report_outputs/storefront_smoke_comparison.md](report_outputs/storefront_smoke_comparison.md)).

- **Validation by whole runs.** 12 test runs (fixed seed; every fault type appears, 2 are fault-free), the rest train/validation folds by run. Best supervised model, chosen on the train runs only: RF (both), balanced accuracy 0.960 (95% run-bootstrap CI 0.893-1.000) on windows after each run's 150 s learning period. Shuffled labels give 0.50; recognising the run explains nothing here (P17).
- **Live demos (P12).** The detector (RF on raw + baseline-relative features, alert when 2 of the last 3 windows are flagged) was frozen first. In three new 12-minute runs the schedule was written before each run: detected 2 of 2 injected faults, 1 false alarm(s) in total, 0 of 24 fault windows missed. This is a demonstration, not a statistic.
- **Reliability.** The held-out test has 12 runs and 15 faults; most fault types have one to three test instances, so per-type recall is marked unreliable (fewer than 5 instances). The 2 fault-free test runs cover only 0.25 hours, so false-alarm rates are very coarse.
- **What is not shown.** One app, one machine, synthetic traffic and injected faults, a small held-out test, unreliable per-type results, and **no transfer to other apps** (the cross-system test, P11, failed). A claim that FailSight works on any app is not supported.

Dataset description: [docs/TESTBED_DATASET.md](docs/TESTBED_DATASET.md). Presentation script: [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md).

## External test on RCAEval (testing only)

`python scripts/rcaeval_eval.py` downloads nothing itself; the RE1 metrics-only cases (RCAEval, Zenodo 14504480/14504481, 390 MB: Online Boutique, Sock Shop, Train Ticket) are read from `external/rcaeval/` (git-ignored). 150 cases (10 per fault type per system, explicit injection times). The testbed's two unsupervised per-app detectors, with their thresholds frozen from the testbed train runs, were applied to generic aggregates (CPU, memory, network, disk, request rate, latency, errors where present): the robust z-score detected 20% to 36% of the faults per system (placebo detections at most 10%), IsolationForest 68% to 86% but also at placebo times in 31% to 79% of the cases. The supervised testbed models could not be applied (testbed-specific inputs). Nothing was tuned on this data; it is a first look, not evidence of transfer. Details: `report_outputs/rcaeval_summary.md`, proofs P18.

## Connect your app

[docs/INTEGRATION.md](docs/INTEGRATION.md) ("Connect your app in 15 minutes") has copy-paste snippets for a Python ASGI middleware (`agent/failsight_agent`), a Node/Express middleware (`agent/node/failsight-express.js`) and tailing an nginx or application access log (`python -m failsight_agent.tail`); the web app has a **Connect** page with the same snippets, the connection state (events received, last event, learning or live), an optional webhook form, and Simple/Detailed modes. Optional and off by default: a per-app API key on the ingestion endpoints (`FAILSIGHT_API_KEYS="app:key,..."`) and a Slack-compatible alert webhook (`FAILSIGHT_WEBHOOK_URL`). Flow: collect, send, learning period (150 s), scoring every 10 s, alerts with reasons. **The detector was validated only on the testbed shop; every other app is labelled unvalidated, and transfer to other apps has not been shown.** Checks: proofs P19.

## Architecture

```
 Collector            Ingestion API             Window builder            Model                 Alerts
 (replay script  ->   POST /ingest/api     ->   30 s window every   ->    saved model +    ->   status, risk,
  or real agent)      POST /ingest/logs         10 s per system,          saved decision        top-10 SHAP,
                      POST /ingest/metrics      same feature code         rule (never           plain-language
                      validates, buffers        as training               retrained here)       reasons -> SSE
```

- **Collector.** `scripts/replay_scenario.py` reads recorded AnoMod events (read-only) from *test* blocks only (never training data) and posts them at 1x / 10x / 60x.
- **Ingestion API.** `app/backend/monitor.py`, `main.py`: validates events, rejects bad ones with a reason, and keeps per-system buffers. Late events are accepted up to 1 s after a window closes.
- **Window builder.** `scripts/window_features.py` is the exact feature arithmetic used for training, shared with the live path; a test checks that windows built from replayed events equal the training windows. Counter metrics are converted to rates; the first two windows are warm-up and are not scored. The replay sends recorded events in slices of at most 0.5 s of recorded time (all three kinds per slice), because the ingestion API drops events older than its 1 s lateness; at high speeds the replay can therefore run a little slower than requested.
- **Model.** The saved final model and its saved decision rule (`report_outputs/thresholds.json`), explained with SHAP.
- **Alerts.** Browsers receive each result over Server-Sent Events. If more than half of the features a system normally provides are missing in a window, the window is shown as **Not enough data to judge**: no alert is raised and it is counted separately (the saved model and threshold are unchanged). Status: *Failing* if at least 2 of the last 3 scored windows are flagged; *Watch* if one is flagged or the risk is close to the alert line; otherwise *Healthy*. "Correct alert / false alarm / missed" counts are known only because the recording carries the true labels.

## How a real deployment connects

The ingestion API is the integration point; nothing in the Window builder or the Model knows about replay.

- **Access logs / traces.** An OpenTelemetry Collector (or the web server's access log, shipped as JSON) posts request records to `POST /ingest/api`: `{"system": "SN"|"TT", "events": [{"timestamp", "status_code", "latency_ms", "content_length"}, ...]}`.
- **Application logs.** Fluent Bit (or Fluentd) tails container logs and posts `{"system", "events": [{"timestamp", "level"}, ...]}` to `POST /ingest/logs` (level: INFO, WARN, ERROR, DEBUG, CONT or OTHER).
- **Metrics.** Prometheus (remote-write through a small adapter, or a scrape job) posts `{"system", "events": [{"timestamp", "metric", "series", "value"}, ...]}` to `POST /ingest/metrics`. Counter metrics are handled by the rate rule.
- The models were trained on AnoMod's benchmark systems; on a different system they did not work (LO2), so a deployment would need its own recorded data and a new training run. Nothing here is validated for production use.

The exact payloads are in `app/backend/monitor.py` (`clean_api`, `clean_log`, `clean_metric`) and in [app/README.md](app/README.md).

## Repository map

| Path | What |
|---|---|
| `scripts/` | training and evaluation pipeline, `leakage_audit.py`, `proofs.py`, `window_features.py`, `replay_scenario.py` |
| `app/backend/` | FastAPI app: ingestion, monitor, replay control, proofs API, model registry; `tests/` |
| `app/frontend/` | React + Vite + TypeScript console; `tests/` (Playwright, Lighthouse) |
| `report_outputs/` | every result file the app reads: metrics, bootstrap intervals, thresholds, audit, `proofs.json` |

AnoMod and LO2 are read-only inputs and are never modified. Data, trained models and `node_modules` are not committed.
