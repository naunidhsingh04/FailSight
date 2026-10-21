# Demo script (5 minutes)

What you show: a small real shop, connected to FailSight with three lines, learns its own normal, gets a real fault, and the monitor raises an alert with a plain-language reason; then the proofs and the honest limits. Everything runs on one Windows laptop, no Docker.

## Before you start (2 minutes, not part of the 5)

1. Plug the laptop in, close heavy programs.
2. Build the dashboard once: `cd app\frontend && npm install && npm run build`.
3. Check that `models\testbed\detector.joblib` exists (`python -m testbed.freeze --check`). It is created by `python -m testbed.freeze` and is never changed after the demos.
4. Start everything with one command: double-click `start_testbed.bat` (or `python -m testbed.launcher`). The browser opens on **Console > Live demo app**. The monitor needs about 40 seconds to start.

## The 5 minutes

**0:00 - 0:30 What this is.** "FailSight watches an application and says when it looks like it is failing. AnoMod, our public dataset, has no fault times, so we cannot tell from it whether failures are really detected. So we built our own small ticket shop and break it on purpose at recorded times." Point at the banner on the Live demo tab: *live demo of our own test shop, not a production system*.

**0:30 - 1:15 Connecting an app takes three lines.** Open `testbed\shop\app.py` and show:

```python
from failsight_agent import FailSight
fs = FailSight.from_config(dict(app_id="shop", monitor_url="http://127.0.0.1:8000"))
fs.instrument(app, pool=pool)
```

"That is all the shop contains: no reporting code. The agent records every request (time, route, status, latency), forwards the log lines, and samples the process (CPU, memory, threads, event-loop lag, database-pool wait) every two seconds."

**1:15 - 2:00 Learning mode.** On the Live demo tab the status says **Learning** and a bar reads *Learning normal behaviour: NN% done*. "For the first 150 seconds the monitor only learns what normal looks like for this app: no alerts. If you started the launcher a few minutes before, this is already finished: the status shows Healthy and the risk gauge sits low."

**2:00 - 3:00 Break something.** Choose **How bad: Medium** and press **Slow database queries** (or **Out of database connections** for a gradual fault). Say: "The switch changes what the running shop really does. The monitor is not told which fault it was." Watch the risk line rise; after two of three windows (about 30 seconds) the status turns **Failing**.

**3:00 - 3:45 The alert and why.** Open the alert in the feed: "Why this alert?" shows plain sentences such as *the slowest 5% of requests rose to 850 ms (normally 24 ms)*. Switch to **Detailed** to show the SHAP table, the frozen model and the alert rule (2 of the last 3 windows). Press **Stop all faults** and watch it recover.

**3:45 - 4:45 The proofs and the honest limits.** Open **Testbed**: the three scripted live demos on new seeds (the fault schedule was written down before each run; results are shown exactly as they came out, misses included). Open **Proofs**: P16 (whole runs the models never saw, with run-bootstrap intervals), P17 (recognising the run cannot explain the result here), P12 (the live demos, recomputed from the stored predictions). Then **Limits**: one app, one machine, synthetic traffic and injected faults, a small held-out test, per-type results unreliable, transfer to other apps not shown.

**4:45 - 5:00 Close.** "We can show that on our own shop, with recorded fault times and whole recordings held out, faults are detected. We cannot claim it works on any other application: that needs its own learning period and its own check."

## If something goes wrong

- The Live demo tab says *the live monitor is not available*: the frozen detector is missing (`python -m testbed.freeze`) or the backend did not start (see `outputs\launcher_backend.log`).
- *The demo shop is not reachable*: the shop is not running; restart the launcher.
- Nothing becomes an alert after a fault: say so. The monitor needs about 30 seconds of clearly abnormal windows, and mild faults can be missed (that is in the results). Do not tune anything during the demo.
- Fall back to the scripted results on the **Testbed** page.
