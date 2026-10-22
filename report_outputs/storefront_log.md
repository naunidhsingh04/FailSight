# Storefront: decision log

Date: 2026-10-22. Goal: give the test shop its own realistic website, separate from the FailSight monitor, with no word "demo" on it.
Not touched (as instructed): the shop's code (`testbed/shop/*`), its routes, database logic, timings, fault injection and telemetry; the models; `testbed_frozen_config.json`; saved results; proof numbers; seeds 9001-9003. `python -m testbed.live_demo` was not run. Nothing was force-pushed.

## What was built

| Piece | Where | Decision |
|---|---|---|
| Storefront | `storefront/` (`python -m storefront.server`, http://127.0.0.1:5000) | A separate FastAPI + Jinja2 app, its own process and port. It calls only the shop's existing endpoints (`GET /search`, `POST /book`, `POST /pay`) over HTTP, exactly as any customer-facing site in front of an API would. |
| Presenter page | `storefront/ops.py`, `/ops` | Reachable only from this computer (the server binds 127.0.0.1 and each request is also checked). The only place where fault names appear. |
| Monitor page | `app/frontend/src/pages/LiveMonitor.tsx` | "Demo" is now "Live monitor" everywhere on the monitor, in the navigation, the page title, the footer and the texts. |
| Launcher | `testbed/launcher.py`, `start_testbed.bat` | Starts shop, storefront, monitor and load generator, reuses what is already running, opens two tabs. |

## Decisions (made without asking, as instructed)

1. **The brand is "Marigold Coaches", not "Kirana Corner".** The shop behind the storefront is a ticket shop (routes between twelve cities, seats, bookings, payments). The endpoints cannot sell groceries, so a grocery name would have been a lie about what the shop does. The brand, the journeys' wording and the customers are made up; the city names are only place names.
2. **Products are coach journeys.** The shop has no product catalogue, only `/search?origin=&dest=` and bookings, so the "product grid" on the home page is a set of six popular city pairs filled from six real searches, the "product page" is one departure, and the "cart" is a cookie that holds the chosen departures (their details are copied into the cookie because the shop has no "get one route" endpoint).
3. **Checkout** calls `POST /book` then `POST /pay` for each item, which is the same sequence the load generator uses. No card is asked for: "No card is needed and nothing is charged". The only personal data is a name and an e-mail address typed by the visitor; nothing is stored apart from an in-memory order list that is lost when the storefront stops.
4. **When the shop is slow or broken the storefront shows it.** A slow shop makes slow pages (that is the point of the demonstration); a 5xx or an unreachable shop gives a plain "We cannot load journeys right now" page with a Try again button instead of a stack trace.
5. **The words demo, test and FailSight do not appear anywhere a visitor can read or hear them** (text, title, meta description, labels, placeholders, alt text), except the footer line "Fictional shop used to test FailSight". A Playwright test checks every page at four widths in three browsers.
6. **The monitor finds problems by watching the shop, not by being told.** Breaking moved to `/ops`, so the monitor page can no longer know when a problem was caused. It notices a switched-on fault from the shop's fault list (and uses the fault's own start time), or the shop stopping to answer after it had answered. It never shows a fault name.
7. **A crash is shown as "The shop has stopped. It restarts by itself" for 60 seconds** (a crash lasts about 30 seconds); after that it is simply "The shop is not running" with the Start button.
8. **The presenter page repeats the fault parameters exactly as the monitor did before** (`testbed.run_generator.severity_params`, medium strength, 90 s for the three timed faults), so a break from `/ops` is the same break as before.
9. **The Start button now starts the storefront too** (it runs the same launcher), and Stop kills what the launcher started. A shop started with `start_testbed.bat` is stopped in its own window.
10. **The launcher reuses a running shop and does not reset the monitor's learning in that case**, and it starts a load generator only if no shop traffic reached the monitor in the last 10 seconds. If port 5000 or 8802 is taken by something that is not ours it says so instead of crashing (shop: stop with a message; storefront: warning and carry on).
11. **Jinja2 was added to `app/backend/requirements.txt`** (the storefront's only new dependency).

## Smoke run (3 minutes) against a saved normal run

`python scripts/smoke_storefront.py` runs the recorder and the shop as the run generator does, the storefront, the steady load for 180 s and 14 real storefront sessions on top (64 page and form requests, all answered without an error), and compares the raw telemetry with `testbed/runs/r000_s1000` (fault-free). The full comparison is in `report_outputs/storefront_smoke_comparison.md`. Summary: the fields of every event type, the metric series, the log events per API event (1.000 against 0.999), the metric events per second (2.98 against 3.01), the 5xx and 4xx shares (0%), the route mix and the median latency per route are the same as before. API events per second differ (7.85 against 5.81) only because the saved run used the "daily" profile at 9 sessions/s and the smoke run the steady profile at 6.

## Limits

- The storefront is one more client of the shop. It is not part of what FailSight was validated on; the validation is still about the shop's telemetry only.
- The cart cookie is not signed: it holds only made-up journeys and is read back through the same validation as a form post.
