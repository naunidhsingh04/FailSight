# FailSight web app

A FastAPI backend and a React + Vite + TypeScript frontend for the FailSight models. Pages: **Console** (replay of recorded AnoMod data through the real ingestion pipeline, with status, risk, alerts and a plain-language "Why this alert?"), **How it works**, **Proofs** (the P1-P10 checklist with a live re-run), **Results**, **Limits** (including the failed LO2 external test) and **Glossary**, with a Simple / Detailed switch. The demo source is a *replay of recorded AnoMod data*, not a live system. See the [root README](../README.md) for the architecture and how a real deployment would connect.

Everything shown comes from result files in `report_outputs/` and `results/v1_16epoch/` and from the saved final models. **Nothing is retrained or tuned by the app.** AnoMod and LO2 are never touched.

## What you need first (not in git)

| Needed | Where | How to get it |
|---|---|---|
| Final models | `models/` (`label_binary__XGB.joblib`, `label_level__Fusion.keras`, ...) | produced by the training run (`scripts/full_pipeline.py`, stage 8) |
| Preprocessor | `processed/preprocessor.joblib` | `python scripts/preprocess.py` |

The backend refuses to start, with a message that lists every missing file, if a model file is absent. Soft voting needs 40 fold models and per-fold preprocessors, so it is shown in the leaderboard as *report only* and cannot be served live.

Python 3.11+ and Node 20+ are required. TensorFlow is needed only for the deep models, the XGBoost heads and stacking; without it those models appear as unavailable and the tree/linear models still work (`FAILSIGHT_LOAD_DEEP=0` forces this).

## Run it (development)

Two terminals, from the repository root (`code/`).

**Windows (PowerShell)**

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r app\backend\requirements.txt -r app\backend\requirements-deep.txt
python -m app.backend.serve                               # terminal 1 (port 8000; loads the models, about 30-60 s with TensorFlow)

cd app\frontend; npm install; npm run dev                  # terminal 2 -> http://localhost:5173
```

**macOS / Linux**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r app/backend/requirements.txt -r app/backend/requirements-deep.txt
python -m uvicorn app.backend.main:app --port 8000          # terminal 1

cd app/frontend && npm install && npm run dev               # terminal 2 -> http://localhost:5173
```

The Vite dev server proxies `/api` to port 8000.

## Run it (single process, like production)

```bash
cd app/frontend && npm install && npm run build && cd ../..
python -m uvicorn app.backend.main:app --port 8000          # serves the API and the built app on http://localhost:8000
```

## Docker

```bash
docker compose up --build          # http://localhost:8000
WITH_DEEP=0 docker compose up --build   # no TensorFlow: smaller image, deep models unavailable
```

`docker-compose.yml` mounts `./models` and `./processed/preprocessor.joblib` read-only. (The Dockerfile was written but could not be built on the machine used for development because Docker is not installed there.)

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | status, number of models, TensorFlow availability |
| `GET /api/models[?label_version=]` | every model: family, corrected variant, saved decision rule, test metrics with 95% CIs |
| `POST /api/predict` | JSON `{model, label_version, window / windows / sample_id}` or multipart `file` (CSV, max 5 MB, 2000 rows) + `model` |
| `GET /api/samples` | 20 real test windows (4 per class, 2 SN + 2 TT) with true labels |
| `POST /api/explain` | SHAP (top 15, grouped api/log/metric) and LIME for one window and a tree model |
| `GET /api/results` | leaderboards, paired bootstrap, correction table, first run vs retrain |
| `GET /api/results/model/{label}/{name}` | confusion matrices and ROC/PR points |
| `GET /api/data` | dataset numbers, split, folds, SMOTE counts, leakage checks, SHAP summary, error analysis |
| `GET /api/lo2` | LO2 external-test results |
| `POST /ingest/api`, `/ingest/logs`, `/ingest/metrics` | `{system: SN\|TT, events: [...]}` (or one bare event); API event `{timestamp, status_code, latency_ms, content_length}`, log event `{timestamp, level}`, metric event `{timestamp, metric, series, value}`; invalid events are rejected with the reason |
| `POST /ingest/reset`, `/ingest/flush` | start a fresh session for a system / close the windows that are ready |
| `GET /api/monitor/state`, `PUT /api/monitor/config` | current status, counters, recent alerts; choose the model |
| `GET /api/stream` | Server-Sent Events: one `window` event per scored window |
| `POST /api/replay/start`, `GET/POST /api/replay/control`, `GET /api/replay/status` | start a replay (scenario, speed 1/10/60), pause / resume / change speed / stop |
| `GET /api/proofs`, `/api/proofs/download.json`, `/api/proofs/report.html` | the proof report (`report_outputs/proofs.json`), JSON download, printable page |
| `POST /api/proofs/run` | re-run `scripts/proofs.py` (optionally `{sections: [...]}`) and stream its output; 409 if a run is in progress |

A prediction window is one row with the 155 raw feature names (see `report_outputs/` and `processed/feature_names.json`) plus `system` = `SN` or `TT`. Missing columns are treated as missing features and reported by group (api / log / metric). An optional `run_id` column lets the sequence models use the previous windows of the same run; otherwise the window is repeated six times.

The decision rule is the one saved in `report_outputs/thresholds.json` (binary: a threshold on P(Anomaly) tuned on out-of-fold predictions; five-class: per-class weights), not plain argmax; the response also gives the argmax class.

## Tests

```bash
pip install -r app/backend/requirements-dev.txt
python -m pytest app/backend/tests -q                       # 71 backend tests, use the real models (skipped if they are absent)

cd app/frontend
npx playwright install chromium firefox webkit              # once
npm run build && npm run test:e2e                           # Chromium, Firefox, WebKit at 375 / 768 / 1280 / 1920 px, axe accessibility checks
npm run test:replay                                         # replay flows (start, pause, alert, explanation, proofs re-run); one worker because the server owns one replay
node tests/lighthouse.mjs                                   # Lighthouse on every page (needs the app on port 8765)
```

`tests/lighthouse-summary.json` holds the last Lighthouse scores (accessibility 100, best practices 100 and SEO 100 on every page; performance 79-100).

The proofs re-run test executes `scripts/proofs.py` for real and refreshes its sections in `report_outputs/proofs.json`.

Test notes: on Windows start the server with `python -m app.backend.serve` (it selects the selector event loop; uvicorn's default Proactor loop can stop accepting connections when WebKit resets speculative connections). WebKit/Safari do not Tab onto links by default, so the skip-link test focuses the link directly there. Headless Firefox on Windows occasionally loses its software compositor under parallel load, so the Firefox project allows one retry.

## Design notes

Editorial, restrained: one serif (Source Serif 4) for headings, one sans (Inter) for the interface, both self-hosted; neutral palette plus a single accent; grayscale-first charts that stay readable when printed; tables become stacked cards on narrow screens; skeleton loaders, empty and error states; visible focus rings, ARIA labels, a skip link; light and dark colour schemes; animations are disabled under `prefers-reduced-motion`. No UI kit, template, gradient or glass effect is used.

## Regenerating the small assets

`app/backend/build_assets.py` (needs the local training state, not the app) rewrites `model_manifest.json`, `samples.json`, `curves.json` and the git-ignored `lime_background.npy`. Without the LIME background the Explain page shows SHAP only and says why.
