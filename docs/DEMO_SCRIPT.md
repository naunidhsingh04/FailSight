# Presenter script (5 minutes)

What you show: a small real shop with its own website, connected to FailSight with three lines of code, learns its own normal, gets a real problem, and the live monitor raises an alert with a plain-language reason; then the proof and the honest limits. Everything runs on one Windows laptop, no Docker.

## The three addresses

| Address | What it is |
|---|---|
| http://127.0.0.1:5000 | **The shop** (Marigold Coaches, fictional). What a customer sees. |
| http://127.0.0.1:5000/ops | **Presenter controls**: the buttons that break the shop. Only works on this computer. |
| http://127.0.0.1:8000 | **The live monitor** (FailSight). What the operator sees. |

## Before you start (2 minutes, not part of the 5)

1. Plug the laptop in, close heavy programs.
2. Build the monitor page once: `cd app\frontend && npm install && npm run build`.
3. Check that `models\testbed\detector.joblib` exists. It is created by `python -m testbed.freeze` and is never changed after the live runs.
4. Double-click `start_testbed.bat` (or run `python -m testbed.launcher`). It starts the shop, the storefront, the monitor and the load generator, then opens two tabs: the shop and the monitor. The monitor needs about 40 seconds to start. Anything that was already running is reused, so you can run it again safely.

## The 5 steps

**1. Open the shop** (tab 1, http://127.0.0.1:5000). "This is a coach-ticket shop. It is made up: the brand, the journeys, the customers. Search for a journey, put seats in the cart, check out: every click is a real request to a real shop with a real database." Do one booking.

**2. Open the monitor** (tab 2, http://127.0.0.1:8000). "This is FailSight. It does not see the shop's pages or customers, only numbers sent by three lines of code in the shop":

```python
from failsight_agent import FailSight
fs = FailSight.from_config(dict(app_id="shop", monitor_url="http://127.0.0.1:8000"))
fs.instrument(app, pool=pool)
```

The status strip says *The shop is running* and has a link **Open the shop**. If it says *not running*, press **Start the shop** in step 1 of the page.

**3. Wait while it learns** (about 2 minutes, less if you started early). Step 1 on the monitor shows *FailSight is learning what normal looks like: NN% done*. "For its first 150 seconds the monitor only watches. No alerts. Anything it sees later is compared with this." Use the time to show the shop or the Proof page.

**4. Break something** on the presenter page (http://127.0.0.1:5000/ops, or the button *Open the presenter controls* on the monitor). Press **Make pages slow** (or **Make the database slow**, **Make a page crash**, **Make the server crash**, **Surprise me**). "These really change what the running shop does. The monitor is not told which one I pressed." Reload a page in the shop to feel it.

**5. Watch the monitor alert.** Back on the monitor: step 3 turns from *Everything looks normal* to *Something is wrong: pages are getting slow* (colour, icon and words), with a timeline *A problem started at 14:02:10 / FailSight noticed at 14:02:26 (16 seconds later)*, a plain reason (*Response time went up 14 times*) and the chart with the moment marked. Press **Fix it** on `/ops` and watch it recover (about a minute). Switch to **Detailed** for the numbers.

## Then, if there is time

- **Proof** (menu): one sentence and the three numbers an examiner asks for, read from the result files: score on test runs it never saw, problems caught, false alarms per hour, and the honest-limits line. The full list of checks is one click below.
- **For examiners > Limits**: one app, one machine, synthetic traffic and injected faults, a small held-out test, per-type results unreliable, transfer to other apps not shown.

Close with: "We can show that on our own shop, with recorded fault times and whole recordings held out, problems are noticed. We cannot claim it works on any other application: that needs its own learning period and its own check."

## If something goes wrong

- The monitor says *FailSight itself is not answering*: the backend did not start; see `outputs\launcher_backend.log`.
- The shop tab shows *We cannot load journeys right now*: that is what a customer sees while the shop is broken or restarting (a crash lasts about 30 seconds). Press **Try again** or **Fix it**.
- `/ops` says *Not available*: it only works from this computer (127.0.0.1).
- Port 5000 is used by another program: the launcher says so and carries on without the storefront; close the other program and run it again.
- Nothing becomes an alert after a problem: say so. The monitor needs about 30 seconds of clearly abnormal checks, and mild problems can be missed (that is in the results). Do not tune anything during the demonstration.
- Fall back to the scripted results on **For examiners > Testbed**.
