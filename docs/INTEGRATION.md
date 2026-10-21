# Connect your app

FailSight watches an application from the outside: a small **agent** records each request (time, route, status, latency), optionally forwards log lines and samples the process (CPU, memory, threads, event-loop lag, database-pool wait), and sends them to a **monitor**. The monitor learns what normal looks like for your app, then scores it every 10 seconds and raises alerts with plain-language reasons.

```
your app  --(agent)-->  POST /ingest/api|logs|metrics  -->  window builder  -->  detector  -->  alerts
   collect                 send (batched, every 1 s)         30 s window,        learning 150 s,    dashboard feed,
                                                             every 10 s          then scoring       optional webhook
```

> **Read this first.** The detector was validated **only on our testbed app** (one small ticket shop, one machine, synthetic traffic, injected faults, a small held-out test: 12 runs and 15 faults). **Transfer to other apps has not been shown** (a cross-system test on AnoMod failed, and the external LO2 test failed). Alerts for any app other than the testbed shop are labelled *unvalidated* by the monitor. Another app needs its **own labelled data** (inject faults at recorded times, as `testbed/run_generator.py` does for the shop, and validate on whole runs) **or at least the learning period with the unsupervised per-app detector** (a robust z-score against the app's own first minutes; it is shown as a second opinion and never raises alerts on its own). A claim that FailSight works on any app is not supported.

## 15 minutes to a first score

1. **Start the monitor** (Windows, no Docker): `python -m app.backend.serve` from the repository root (the dashboard is at `http://127.0.0.1:8000`; build it once with `cd app\frontend && npm install && npm run build`). The frozen testbed detector must exist: `python -m testbed.freeze` creates it from the testbed runs.
2. **Choose an app id**, for example `my-app`. Every event you send names it in a `system` field.
3. **Connect the app** with one of the snippets below.
4. **Open the dashboard**, Connect page: it shows the events received, the time of the last event and the state *Learning normal behaviour: NN% done* (the first 150 seconds), then *Live*.
5. **Break something on purpose in a test environment** and check that an alert appears. Do not rely on it before you have done this for your own app.

## (a) Python: ASGI middleware (FastAPI, Starlette, any ASGI app)

```python
# copy the failsight_agent folder next to your app (or: pip install ./agent); pip install psutil for the process metrics
from failsight_agent import FailSight

fs = FailSight.from_config(dict(app_id="my-app", monitor_url="http://127.0.0.1:8000"))   # add api_key="..." if the monitor requires one
fs.instrument(app)                                   # app = your FastAPI / Starlette application
# with a database connection pool whose acquire() returns a context manager:  fs.instrument(app, pool=my_pool)
```

What it sends: one event per request (`timestamp`, `method`, `route`, `status_code`, `latency_ms`, `content_length`), every log record at INFO or above, and six process readings every 2 seconds. It never blocks your app: events are queued in memory and sent in the background; if the monitor is down they are dropped.

WSGI apps (Flask, Django): the middleware is ASGI only. Use (c) below, or wrap your app with a WSGI-to-ASGI adapter (not tested here).

## (b) Node: Express middleware

Copy `agent/node/failsight-express.js` (no dependencies, Node 18 or newer):

```js
const failsight = require('./failsight-express');
app.use(failsight({ monitorUrl: 'http://127.0.0.1:8000', appId: 'my-app', apiKey: process.env.FAILSIGHT_KEY }));
```

It posts the same fields to `/ingest/api` in batches of up to 500 every second, and ignores errors when the monitor is unreachable. It sends request data only (no process metrics).

## (c) Any server: tail an nginx or application access log

```
python -m failsight_agent.tail --file /var/log/nginx/access.log --monitor http://127.0.0.1:8000 --app-id my-app [--api-key KEY]
```

Understands the nginx/Apache *combined* format. If the line ends with the request time in seconds (nginx `log_format ... '$request_time'`) it is used as the latency; otherwise latency is sent as 0 and only counts and error rates are available. Add `--from-start` to send the lines already in the file.

Sending your own events (any language): `POST /ingest/api` with `{"system": "my-app", "events": [{"timestamp": 1760000000.5, "status_code": 200, "latency_ms": 12.3, "content_length": 512}]}`; logs go to `/ingest/logs` (`timestamp`, `level`), process readings to `/ingest/metrics` (`timestamp`, `metric`, `series`, `value`; the monitor uses `process_cpu_percent`, `process_rss_bytes`, `process_threads`, `event_loop_lag_ms`, `db_pool_wait_ms`, `inflight_requests`). Invalid events are rejected one by one and counted.

## Optional: API keys

Off by default. Set `FAILSIGHT_API_KEYS="my-app:secretkey,other-app:otherkey"` before starting the monitor; then `/ingest/*` for those apps needs the header `X-API-Key` (the agent sends it when you pass `api_key`). A missing or wrong key returns 401. The AnoMod replay ids (`SN`, `TT`) are not affected.

## Optional: alerts to Slack (or any Slack-compatible webhook)

Set `FAILSIGHT_WEBHOOK_URL` before starting the monitor, or from the dashboard (Connect page) / `PUT /api/testbed/webhook {"url": "https://hooks.slack.com/services/..."}`. One message `{"text": "FailSight alert for my-app (failure risk 87%): ..."}` is sent for each alert episode, in the background; a failing webhook never disturbs the monitor. Apps other than the validated testbed shop are tagged *unvalidated app* in the message.

## What to expect

- **Learning period:** the first 150 seconds after the first event (`reset` anchors it), no alerts. Send realistic *normal* traffic during this time; faults during learning become part of "normal".
- **Delay:** a window is 30 s and a new one is scored every 10 s; an alert needs two of the last three windows flagged. In the testbed the first alert came about 12 to 17 s after a fault started in the three live demos, with one false alarm in 0.37 hours outside faults. That is three runs on one app, not a statistic.
- **Data it needs:** request data is required; process metrics and logs improve it. The model was trained with all three. If an app sends only requests, the missing readings are treated as missing and results are less trustworthy.
- **Limits:** one validated app, one machine, synthetic traffic and injected faults, a small held-out test, per-fault-type results that are unreliable (1 to 3 test instances for most types), and no demonstrated transfer to other apps. See the **Limits** and **Testbed** pages and `docs/TESTBED_DATASET.md`.
