# Front-end redesign: decision log

Date: 2026-10-21. Goal: a person who has never heard of FailSight understands the app in 30 seconds.
Not touched (as instructed): models, `testbed_frozen_config.json`, saved results, proof numbers, demo seeds 9001-9003. `python -m testbed.live_demo` was not run.

## What was wrong (from the screenshots) and what was found

| # | Problem | Cause | Fix |
|---|---|---|---|
| 1 | Home opened on the AnoMod replay | `/` rendered the replay console; our own shop was a second tab (`?tab=live`) | `/` is now the Demo of our own shop. The replay moved to `/examiners/replay`. `/?tab=live` still opens the Demo; `/?tab=replay` and `/console` open the replay. |
| 2 | Empty gauges and charts that said "Waiting" | Gauge and chart were drawn even with no data | The gauge is gone. Empty states say what to do ("Ready. Press Start.", "Start the shop in step 1 first", "No line yet. Press Start in step 1 and it appears here."). The empty chart is not drawn at all. |
| 3 | "Not playing. Live updates connected. The system is overloaded or slow." | `Console.tsx` always appended the hint of the *selected* scenario ("The system is overloaded or slow." for "Slow performance") to the status line, even when nothing was playing, so a description of a choice read like a live fact | Idle now reads "Ready. Press Start." The scenario hint is only shown as "You picked ...: ..." and never as a status. The button is called **Start** (was Play). A test checks the idle line never contains "Not playing" or "Connecting". |
| 4 | "alert line" label overlapping "failure risk" under the gauge | SVG text placed at fixed offsets around the arc | The gauge is replaced by a three-box status bar (OK / Watch / Problem), also in the recorded-data page. The new response-time chart puts its two marker labels on separate rows and flips them to the left near the right edge. |
| 5 | Jargon | Many pages written for specialists | Demo, How it works and Proof rewritten in everyday words. A test fails if SHAP, LIME, baseline, window(s), threshold, ROC, F1, XGBoost, Isolation or z-score appear on the Demo page in Simple mode. "Learning what normal looks like" has a hover/tap explanation (`Term`). |
| 6 | Eight top-level tabs | History | Four: Demo, How it works, Proof, For examiners. Results, Connect, Testbed, Limits, Glossary and the recorded-data replay are sub-pages of For examiners (`/examiners/...`). Every old URL still works. |

## Decisions (made without asking, as instructed)

1. **Step 1 "Start the demo shop" really starts it.** The monitor cannot start a shop by itself, so I added `POST /api/testbed/start` (spawns the same `testbed.launcher --no-browser` that `start_testbed.bat` runs, pointed at this monitor through `FAILSIGHT_MONITOR_URL`, never a second shop) and `POST /api/testbed/stop` (only for a shop started this way). If it fails, the page says so and names `start_testbed.bat`. The launcher change is one line (the monitor address can come from the environment; default unchanged).
2. **Step 2 stays locked until learning is done.** Breaking the shop during the first 150 s would spoil what FailSight learns as "normal". The locked state says why and what to do.
3. **The four choices map to real faults** (nothing simulated in the browser): "Make pages slow" = `latency`, "Make the database slow" = `db_slow`, "Make a page crash" = `exception_route`, "Make the server crash" = `crash`; "Surprise me" picks one of the four at random. Strength is always "medium", 90 s for the first three (the crash fault keeps its own duration). "Fix it now" clears all faults.
4. **"FailSight noticed"** = the first alert (the rule from the frozen config: 2 of the last 3 checks) that was emitted after the moment of breaking. If the fault ends and no alert came, the page says "FailSight did not raise an alert this time" instead of hiding it.
5. **The plain reason** is derived from the strongest reading behind the alert (`plain.ts`): e.g. 4.4 ms to 63 ms becomes "Response time went up 14 times". It always carries the caveat that FailSight is not told what was broken, so this is a clue, not a proven cause.
6. **Proof page numbers are read, never typed.** Held-out score: balanced accuracy of the frozen model on the 12 held-out runs. Problems caught: rows of `testbed_heldout_instances.csv` for the frozen model and alert rule. False alarms per hour: `testbed_false_alarms.csv` (0 per hour, but over only 15 minutes of quiet running; the sentence says so, and also gives the rate in the calm parts of the runs with problems). To read the per-fault file, `/api/testbed/results` gained an `instances` field (a read of an existing file).
7. **Simple vs Detailed.** Simple is the default. Detailed only adds numbers (risk, alert level, counts, model, the real fault injected, the strongest readings).
8. **Light is the default, even if the computer prefers dark.** Dark is a button in the header, remembered in `localStorage`. One font (Inter) everywhere; the serif font was removed from the bundle. Body text 18 px, buttons at least 48 px high (the big ones 56 px).
9. **Examiner pages keep their technical wording** (Results, Testbed, Limits, Glossary, Connect). They are the detail for people who asked for it, and every technical word there already has a tooltip or the Glossary. Only the three pages a stranger sees (Demo, How it works, Proof top) were rewritten for a 14-year-old.
10. **Tooltips now work by tap.** Found while testing: a click or tap on the "?" first fired the hover handler and then the click handler, which closed it again, so it could not be opened by touch. It now stays open once tapped and closes on a second tap, Escape or a tap elsewhere.
11. **Old live-demo page deleted** (`LiveDemo.tsx`); the Demo replaces it and nothing links to it.

## Things not done / limits of this redesign

- The long end-to-end test (`demo.spec.ts`: start the shop, learn for about 2.5 minutes, break four ways) runs alone with one worker (`npm run test:demo`) and only in the three browsers at 1280 px; the 375/768/1920 layouts of the Demo are covered by the fast tests (three steps, buttons, no jargon, no overflow, axe).
- The Demo cannot prove anything about other apps; the limits box and the Proof page say so.
