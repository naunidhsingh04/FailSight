# Present now: the exact steps

Written for: the person who has to give the presentation, with no time to read anything else.

## The three addresses

| Address | What it is |
|---|---|
| http://127.0.0.1:5000 | The shop (Marigold Coaches, fictional). What a customer sees. |
| http://127.0.0.1:5000/ops | Presenter controls: the buttons that break the shop. Only works on this computer. |
| http://127.0.0.1:8000 | The live monitor (FailSight). |

## The 5 steps

1. **Start everything.** In PowerShell: `cd "C:\Users\Lenovo\Documents\AI PROJECT\code"` then `.\start_testbed.bat`. Two browser tabs open by themselves: the shop and the monitor. Leave the black window open. The shop comes up in about 10 seconds, the monitor in about 40 seconds (the monitor tab may need a reload once).
2. **Show the shop** (tab 1, port 5000). Pick a journey, add seats to the cart, check out. It is made up and nothing is charged.
3. **Show the monitor** (tab 2, port 8000). It says *The shop is running* with a link *Open the shop*. Step 1 shows *FailSight is learning what normal looks like*. **Wait about 2.5 minutes** until it says *FailSight knows what normal looks like* and step 3 says *Everything looks normal*.
4. **Break something.** Open http://127.0.0.1:5000/ops (or press *Open the presenter controls* on the monitor) and press **Make pages slow**. Reload a page in the shop to feel it.
5. **Watch the monitor.** About 20 seconds later step 3 turns red: *Something is wrong: pages are getting slow*, with the time the problem started, the time FailSight noticed, and a plain reason. Press **Fix it** on `/ops`; the monitor needs about a minute to calm down again.

To stop: press Ctrl+C in the black window.

## What was checked today (2026-10-22, no browser test suites)

Started with `start_testbed.bat`: the shop loaded (home page and the footer line), `/ops` loaded with its buttons, the monitor page loaded ("Live monitor", the shop strip with *Open the shop*, steps 1 to 3). Learning finished after 158 seconds. *Make pages slow* was pressed once; the monitor raised an alert 20 seconds later ("the slowest 5% of requests rose to 515 ms (normally 5.4 ms)"). *Fix it* was pressed; the fault list was empty afterwards and the monitor was still recovering 70 seconds later (status "watch"; leave it a bit longer before the next break).

## What is NOT finished

- The long browser test that presses **Start the shop** on a stopped shop in all three browsers did not finish (the first run failed because the test looked for the shop site a moment too early; the test was corrected, but its re-run was stopped). The Start button itself works in the backend test with mocks and starts the shop and the shop site when used; it was not proven end to end in all three browsers.
- The earlier browser suites (storefront, 30 tests in three browsers) passed; the full older suite, the replay spec, the backend tests and the proofs were **not re-run after the storefront was added**.
- The monitor page may keep saying "Something is wrong" for a minute or two after a problem is fixed.

## Last known-good commit (the redesign, before the storefront)

`3a74616f3e0b21fef46265d822f6dc374c74e191` ("docs: redesign log and screenshots"). That version has the guided demo page on the monitor, all browser suites, the replay spec and the proofs passing (65 PASS, 4 WARN, 0 FAIL, 9 REPORTED ACCURATELY), and no storefront.

To look at it without losing anything: `git switch --detach 3a74616` (go back with `git switch main`).
To return `main` to it (this throws away the storefront work on this branch; do not do this during the presentation): `git reset --hard 3a74616`. Never force-push.

## If something goes wrong

- Nothing on port 5000: run `python -m storefront.server` in a second window, or run `.\start_testbed.bat` again (it reuses whatever already runs).
- The monitor says *FailSight itself is not answering*: wait 40 seconds and reload; if it stays, see `outputs\launcher_backend.log`.
- A port is busy: `Get-NetTCPConnection -LocalPort 8000,8802,5000 -State Listen` shows who; stop that process and start again.

## Cold-start re-check (overnight run, 22 Oct 2026, 23:45 IST)

From a cold start (`python -m testbed.launcher --no-browser`, same as start_testbed.bat): the shop, /ops and the monitor all answered with HTTP 200 within 60 s; learning finished in about 150 s; *Make pages slow* was detected as an alert 17 s later ('failing'); *Fix it* worked. No code changed. The monitor may stay in 'watch' for about 2.5 minutes after a fix: wait before the next break.

## What to say about limits

- The detector was trained on one made-up shop; we do not claim it works on any other application, and we do not claim it can forecast a failure or tell how long until one. It notices a problem that has started.
- The test sets are small (647 AnoMod windows, only 36 healthy ones), so every score comes with a range.

