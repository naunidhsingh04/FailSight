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

You need Python 3.11+ and Node 20+. The trained models are not in git (see [app/README.md](app/README.md)); they must be in `models/` and `processed/preprocessor.joblib`.

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r app\backend\requirements.txt -r app\backend\requirements-deep.txt
cd app\frontend; npm install; npm run build; cd ..\..
python -m app.backend.serve          # http://127.0.0.1:8000 (loads the models: 30-60 s)
```

Open the address, choose a scenario, pick a speed (1x, 10x, 60x) and press **Play**. On macOS or Linux use `python3` and `python -m uvicorn app.backend.main:app --port 8000`.

Use the **Simple / Detailed** switch in the top bar: Simple uses plain language; Detailed adds model names, thresholds, tables and raw values. Every technical word has a "?" help button and an entry in the Glossary.

The replay can also be run from a terminal, without the browser:

```powershell
python scripts\replay_scenario.py --scenario Service --speed 60 --server http://127.0.0.1:8000
python scripts\replay_scenario.py --scenario Normal --dry-run     # show the plan only
```

## Proofs: check every number yourself

```powershell
python scripts\proofs.py              # all ten sections P1-P10 (about 7 minutes; loads every saved model)
python scripts\proofs.py --quick      # data, split, folds, versions only
python scripts\proofs.py --only P5    # one section; it is merged into report_outputs\proofs.json
python scripts\leakage_audit.py       # the leakage audit (C1a-C7) that P7 reports
```

The script prints a PASS / WARN / FAIL table and exits non-zero if any check fails. The app's **Proofs** page shows the same report (`report_outputs/proofs.json`), can re-run the checks and stream the terminal output, and downloads the report as JSON or a printable page.

Last full run: 44 PASS, 3 WARN, 0 FAIL. The warnings are reported, not hidden:

- **P5, ranking metrics.** Decision metrics of all 36 saved models are recomputed and equal `test_metrics.csv` within 1e-6. ROC-AUC / PR-AUC of the forest models differ in the 3rd decimal because tied scores are ordered by floating-point noise; the probabilities themselves match to 1e-16.
- **C3, time overlap.** Some training windows overlap in time (up to 20 s) with a test window of the same run. This is real; it stays a WARN. Measured effect of removing them: -0.036 balanced accuracy (binary) and +0.010 (five-class), on average.

**What the file-integrity check proves.** The audit's check C7 hashes the AnoMod and LO2 files (SHA-256). That baseline was **first recorded during the audit run itself**, so it proves that no file has changed *since then*, not before.

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
- **Window builder.** `scripts/window_features.py` is the exact feature arithmetic used for training, shared with the live path; a test checks that windows built from replayed events equal the training windows. Counter metrics are converted to rates; the first two windows are warm-up and are not scored.
- **Model.** The saved final model and its saved decision rule (`report_outputs/thresholds.json`), explained with SHAP.
- **Alerts.** Browsers receive each result over Server-Sent Events. Status: *Failing* if at least 2 of the last 3 scored windows are flagged; *Watch* if one is flagged or the risk is close to the alert line; otherwise *Healthy*. "Correct alert / false alarm / missed" counts are known only because the recording carries the true labels.

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
