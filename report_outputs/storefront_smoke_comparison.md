# Storefront smoke run compared with a saved normal run

Smoke run: 180 s, steady load 6 requests/s from the load generator plus 64 storefront page and form requests (browse, cart, checkout; 64 of 64 answered without an error). Baseline: `testbed/runs/r000_s1000` (fault-free, 600 s, daily profile, 9 requests/s).
Rates differ because the traffic profiles differ, so the comparison is made on ratios, field names and per-route behaviour.

| What | Saved normal run | Smoke run | |
|---|---|---|---|
| run length (s) | 603 | 183 |  |
| API events per second | 7.85 | 5.81 |  |
| log events per API event | 1.000 | 0.999 | same |
| metric events per second | 2.98 | 3.01 | same |
| fields of an API event | content_length, latency_ms, method, route, status_code, timestamp | content_length, latency_ms, method, route, status_code, timestamp | same |
| fields of a log event | level, logger, message, timestamp | level, logger, message, timestamp | same |
| fields of a metric event | metric, series, timestamp, value | metric, series, timestamp, value | same |
| metric series | db_pool_wait_ms, event_loop_lag_ms, inflight_requests, process_cpu_percent, process_rss_bytes, process_threads | db_pool_wait_ms, event_loop_lag_ms, inflight_requests, process_cpu_percent, process_rss_bytes, process_threads | same |
| server errors (5xx) share | 0.00% | 0.00% | same |
| client errors (4xx) share | 0.0% | 0.0% | same |
| status codes seen | 200: 4736 | 200: 1065 | same |

## Per route (share of requests, median and 95th percentile latency in ms)

| Route | Saved share | Smoke share | Saved p50 / p95 | Smoke p50 / p95 | |
|---|---|---|---|---|---|
| GET /health | 0.0% | 0.1% | 110 / 110 | 23 / 23 | same order |
| GET /search | 45.8% | 49.4% | 3 / 8 | 3 / 12 | same order |
| POST /book | 24.0% | 23.0% | 3 / 11 | 3 / 8 | same order |
| POST /cancel | 10.7% | 9.3% | 3 / 11 | 3 / 8 | same order |
| POST /pay | 19.4% | 18.2% | 19 / 29 | 18 / 28 | same order |

## Metric series (mean value)

| Series | Saved | Smoke |
|---|---|---|
| db_pool_wait_ms | 0.0209 | 0.0195 |
| event_loop_lag_ms | 19.1 | 19.1 |
| inflight_requests | 0.0267 | 0.0435 |
| process_cpu_percent | 4.74 | 3.65 |
| process_rss_bytes | 3.72e+07 | 5.3e+07 |
| process_threads | 7.12 | 9.11 |

Log levels, saved: {'INFO': 4735}; smoke: {'INFO': 1064}.

**Result: telemetry fields, request mix, error rates and latencies look the same as before.**
