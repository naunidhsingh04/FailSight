"""Live monitor for the FailSight testbed (the shop instrumented with failsight_agent).

Events arrive through the same ingestion protocol as every other app (POST /ingest/api|logs|metrics with {"system": "<app id>", "events": [...]}). Per app the monitor
 * learns the app's normal for the first 150 s ("Learning normal behaviour: NN% done"): no alerts, the baseline and the unsupervised detector learn only from these windows,
 * builds a 30 s window every 10 s with the SAME feature code as the dataset (testbed.dataset.compute_window) and the same baseline-relative rule,
 * scores it with the FROZEN model (report_outputs/testbed_frozen_config.json: model, feature set, threshold, alert rule) and raises an alert when 2 of the last 3 windows are flagged,
 * explains an alert in plain language from the tree-SHAP contributions of the model compared with the app's own normal,
 * runs a per-app robust z-score detector as a second opinion (it never raises alerts).
Nothing here is tuned: the configuration is read from the frozen file."""
import hmac
import json
import math
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from testbed import dataset as DS  # noqa: E402
import universal_features as U  # noqa: E402
from .monitor import Hub  # noqa: E402

CONFIG = ROOT / "report_outputs" / "testbed_frozen_config.json"
MODEL = ROOT / "models" / "testbed" / "detector.joblib"
LIVE_DIR = ROOT / "outputs" / "testbed_live"
LATENESS = 3.0                       # seconds an event may arrive after its window closed (the agent flushes every 1 s)
IDLE_AFTER = 90.0                    # no event for this long: the app is idle and no more windows are built
UNSUP_THRESHOLD = 10.0               # robust z-score alert level chosen on the train runs in the held-out validation (second opinion only)
KEEP_SECONDS = 70.0
VALIDATED_APPS = {"shop"}            # the only app the detector was validated on (the testbed shop); every other app id is reported as unvalidated

PLAIN = {
    "api_request_count": ("requests in the last 30 s", ""), "api_4xx_count": ("client-error responses (4xx)", ""), "api_5xx_count": ("failed requests (5xx)", ""),
    "api_error_rate": ("share of failed requests", "%"), "api_latency_mean_ms": ("average request time", " ms"), "api_latency_p95_ms": ("the slowest 5% of requests", " ms"),
    "api_latency_p99_ms": ("the slowest 1% of requests", " ms"), "api_content_length_mean": ("average response size", " bytes"),
    "log_line_count": ("log lines", ""), "log_error_count": ("error lines in the log", ""), "log_warning_count": ("warning lines in the log", ""), "log_error_share": ("share of error lines in the log", "%"),
    "m_process_cpu_percent__mean": ("CPU used by the app", " %"), "m_process_cpu_percent__max": ("peak CPU used by the app", " %"), "m_process_cpu_percent__std": ("swings in CPU use", " %"),
    "m_process_rss_bytes__mean": ("memory used by the app", " MB"), "m_process_rss_bytes__max": ("peak memory used by the app", " MB"), "m_process_rss_bytes__slope": ("memory growth", " MB/s"),
    "m_process_threads__mean": ("threads in the app", ""), "m_event_loop_lag_ms__mean": ("delay in the app's event loop", " ms"), "m_event_loop_lag_ms__max": ("worst delay in the app's event loop", " ms"),
    "m_db_pool_wait_ms__mean": ("wait for a database connection", " ms"), "m_db_pool_wait_ms__max": ("longest wait for a database connection", " ms"),
    "m_inflight_requests__mean": ("requests being handled at the same time", ""),
}


def plain_name(base):
    if base in PLAIN:
        return PLAIN[base]
    if base.startswith("d_"):
        n, u = plain_name(base[2:])
        return "change in " + n, u
    return base.replace("m_", "").replace("__", " ").replace("_", " "), ""


def fmt_value(base, v):
    name, unit = plain_name(base)
    if unit == "%" and base.endswith(("rate", "share")):
        return f"{v * 100:.0f}%"
    return f"{v:.0f}{unit}" if abs(v) >= 100 else f"{v:.1f}{unit}" if abs(v) >= 1 else f"{v:.2f}{unit}"


# ------------------------------------------------------------------------------------------------ optional per-app API keys and alert webhook
def parse_keys(spec):
    """FAILSIGHT_API_KEYS="shop:key1,other:key2" -> {"shop": "key1", "other": "key2"}; empty or unset = the check is OFF (default)"""
    keys = {}
    for part in (spec or "").split(","):
        if ":" in part:
            a, k = part.split(":", 1)
            if a.strip() and k.strip():
                keys[a.strip()] = k.strip()
    return keys


def check_api_key(app_id, provided, keys):
    """True when no keys are configured (off), or when `provided` equals the key configured for this app"""
    if not keys:
        return True
    want = keys.get(app_id)
    return want is not None and provided is not None and hmac.compare_digest(str(provided), want)


def slack_payload(ev):
    """Slack-compatible incoming-webhook body ({"text": ...}); other chat tools that accept the same shape work as well"""
    pct = int(round(ev["risk"] * 100))
    why = ev.get("summary") or "The detector flagged this window."
    tag = "" if ev.get("validated") else " (unvalidated app: the detector was validated only on the testbed shop)"
    return {"text": f":rotating_light: FailSight alert for {ev['app']}{tag} (failure risk {pct}%): {why}"}


class AppStream:
    def __init__(self, app_id):
        self.app_id = app_id
        self.reset(None)

    def reset(self, anchor):
        self.anchor = anchor
        self.api, self.logs, self.mets = [], [], {m: [] for m in DS.METRICS}
        self.k = 0
        self.base_hist = []                  # base feature rows (39 values) of all windows so far
        self.windows = []
        self.alerts = []
        self.flags = deque(maxlen=3)
        self.alert_on = False
        self.quiet = 3
        self.last_event_at = None
        self.counters = dict(windows=0, flagged=0, alerts=0, rejected=0, events_api=0, events_logs=0, events_metrics=0, late_dropped=0)
        self.learn_median = None
        self.learn_scale = None
        self.t_started = time.time()


class TestbedMonitor:
    def __init__(self, config_path=CONFIG, model_path=MODEL):
        self.cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        blob = joblib.load(model_path)
        self.pipeline = blob["model"]
        self.columns = blob["columns"]
        self.base_cols = blob["base"]
        self.raw_used, self.rel_used = blob["raw_used"], blob["rel_used"]
        self.threshold = float(self.cfg["threshold"])
        self.k_rule, self.n_rule = self.cfg["alert_rule"]["k"], self.cfg["alert_rule"]["n"]
        self.learning_s = float(self.cfg["learning_period_s"])
        self.hub = Hub()
        self.lock = threading.RLock()
        self.apps = {}
        self.explainer = None
        self.webhook_url = os.environ.get("FAILSIGHT_WEBHOOK_URL") or None
        self.webhook_log = []
        self._stop = threading.Event()
        self._thread = None

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._tick, name="testbed-windows", daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()

    def _tick(self):
        while not self._stop.wait(1.0):
            try:
                self.close_ready(time.time())
            except Exception:                                  # a bad window must never stop the monitor
                import logging
                logging.getLogger("testbed").exception("window processing failed")

    def app(self, app_id):
        if app_id not in self.apps:
            self.apps[app_id] = AppStream(app_id)
        return self.apps[app_id]

    def reset(self, app_id, start_ts=None):
        with self.lock:
            st = self.app(app_id)
            st.reset(float(start_ts) if start_ts is not None else None)
            LIVE_DIR.mkdir(parents=True, exist_ok=True)
            (LIVE_DIR / f"{app_id}_windows.jsonl").write_text("", encoding="utf-8")
        self.hub.publish(dict(type="session", app=app_id, anchor=st.anchor))
        return dict(app=app_id, anchor=st.anchor)

    # ------------------------------------------------------------------ ingestion
    def ingest(self, kind, app_id, events):
        if not isinstance(app_id, str) or not app_id or len(app_id) > 64:
            raise ValueError("system must be a non-empty app id")
        if not isinstance(events, list):
            raise ValueError("events must be a list")
        with self.lock:
            st = self.app(app_id)
            bad = 0
            for e in events:
                try:
                    if not isinstance(e, dict):
                        raise ValueError
                    t = float(e["timestamp"])
                    if not math.isfinite(t):
                        raise ValueError
                    if kind == "api":
                        sc = int(e.get("status_code", e.get("status")))
                        st.api.append((t, sc, float(e.get("latency_ms", 0.0)), float(e.get("content_length", 0.0))))
                    elif kind == "logs":
                        st.logs.append((t, str(e.get("level", "INFO")).upper()))
                    elif kind == "metrics":
                        name = e["metric"]
                        if name in st.mets:
                            st.mets[name].append((t, float(e["value"])))
                    else:
                        raise ValueError
                    st.last_event_at = max(st.last_event_at or 0.0, t)
                    if st.anchor is None:
                        st.anchor = t
                except (KeyError, TypeError, ValueError):
                    bad += 1
            ok = len(events) - bad
            st.counters["rejected"] += bad
            st.counters[f"events_{kind}"] += ok
        return dict(app=app_id, accepted=len(events) - bad, rejected=bad)

    # ------------------------------------------------------------------ window building
    def close_ready(self, now):
        built = []
        with self.lock:
            for st in self.apps.values():
                if st.anchor is None or st.last_event_at is None:
                    continue
                while True:
                    ws = st.anchor + DS.STEP * st.k
                    we = ws + DS.WIN
                    if now < we + LATENESS or (st.last_event_at is not None and now - st.last_event_at > IDLE_AFTER and we > st.last_event_at + 1):
                        break
                    built.append(self._process(st, st.k, ws))
                    st.k += 1
        return built

    def _arrays(self, st):
        api = sorted(st.api)
        at = np.array([a[0] for a in api], float)
        sc = np.array([a[1] for a in api], int)
        lat = np.array([a[2] for a in api], float)
        cl = np.array([a[3] for a in api], float)
        lg = sorted(st.logs)
        lt = np.array([x[0] for x in lg], float)
        lv = np.array([x[1] for x in lg], str)
        series = {}
        for m, rows in st.mets.items():
            rows = sorted(rows)
            series[m] = (np.array([r[0] for r in rows], float), np.array([r[1] for r in rows], float))
        return at, sc, lat, cl, lt, lv, series

    def base_row(self, st, k, ws):
        f = DS.compute_window(ws, *self._arrays(st))
        for c in DS.DELTA_BASE:
            prev = st.base_hist[k - 3][c] if k >= 3 else np.nan
            f[f"d_{c}"] = f[c] - prev if np.isfinite(prev) else np.nan
        row = {c: f.get(c, np.nan) for c in self.base_cols}
        return {c: (np.nan if (isinstance(v, float) and not math.isfinite(v)) else v) for c, v in row.items()}

    def rel_row(self, st, k, row):
        """the baseline-relative features of window k: exactly universal_features.relative_features for the last row (past windows of this app only)"""
        out = np.full(len(self.base_cols), np.nan)
        elig = np.arange(0, k - U.GUARD + 1)
        if len(elig) < U.MIN_HISTORY:
            return out
        H = np.array([[st.base_hist[j][c] for c in self.base_cols] for j in elig], float)
        for j, c in enumerate(self.base_cols):
            v = H[:, j]
            v = v[np.isfinite(v)][-U.HISTORY:]
            x = row[c]
            if len(v) < U.MIN_HISTORY or not np.isfinite(x):
                continue
            m, s = U.baseline_stats(v)
            out[j] = float(np.clip((x - m) / s, -U.CLIP, U.CLIP))
        return out

    def vector(self, row, rel):
        parts = []
        if self.raw_used:
            parts.append(np.array([row[c] for c in self.base_cols], float))
        if self.rel_used:
            parts.append(np.nan_to_num(rel, nan=0.0))
        return np.concatenate(parts)

    def _process(self, st, k, ws):
        we = ws + DS.WIN
        row = self.base_row(st, k, ws)
        rel = self.rel_row(st, k, row)
        st.base_hist.append(row)
        x = self.vector(row, rel)[None, :]
        risk = float(self.pipeline.predict_proba(x)[0, 1])
        learning = (we - st.anchor) <= self.learning_s
        if not learning and st.learn_median is None:
            self._finish_learning(st)
        flagged = (not learning) and risk >= self.threshold
        if not learning:
            st.flags.append(flagged)
        alert = (not learning) and sum(st.flags) >= self.k_rule
        new_alert = alert and not st.alert_on and st.quiet >= 3
        st.quiet = 0 if alert else st.quiet + 1
        st.alert_on = alert
        z_score = self._z_score(st, row) if st.learn_median is not None else None
        reasons, summary = [], None
        if flagged or alert:
            reasons = self.reasons(st, x)
            summary = ("Flagged because " + "; ".join(r["sentence"] for r in reasons[:3]) + ".") if reasons else "Flagged: the combination of many small changes looks like a failure."
        status = "learning" if learning else "failing" if alert else "watch" if flagged or risk >= 0.5 * self.threshold else "healthy"
        done = min(1.0, (we - st.anchor) / self.learning_s)
        ev = dict(type="window", app=st.app_id, window_index=k, window_start=ws, window_end=we, emitted_at=time.time(), learning=bool(learning), learning_progress=done,
                  risk=risk, threshold=self.threshold, flagged=bool(flagged), alert=bool(alert), new_alert=bool(new_alert), status=status, reasons=reasons, summary=summary,
                  validated=st.app_id in VALIDATED_APPS, sources=dict(api=bool(row["api_request_count"] > 0), logs=bool(row["log_line_count"] > 0), process_metrics=bool(np.isfinite(row["m_process_cpu_percent__mean"]))),
                  robust_z_score=z_score, robust_z_flag=bool(z_score is not None and z_score >= UNSUP_THRESHOLD), has_baseline=bool(np.isfinite(rel).any()),
                  features={c: (None if not np.isfinite(row[c]) else row[c]) for c in DS.DELTA_BASE if c in row},
                  model=dict(name=self.cfg["model"], feature_set=self.cfg["feature_set"], alert_rule=self.cfg["alert_rule"]["name"], frozen_at=self.cfg["frozen_at"]))
        st.windows.append(ev)
        st.counters["windows"] += 1
        st.counters["flagged"] += int(flagged)
        if new_alert:
            st.counters["alerts"] += 1
            st.alerts.append(ev)
            self._notify(ev)
        with open(LIVE_DIR / f"{st.app_id}_windows.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(ev, default=float) + "\n")
        self.hub.publish(ev)
        # forget raw events that no later window can need
        cut = ws + DS.STEP - 5
        st.api = [a for a in st.api if a[0] >= cut]
        st.logs = [a for a in st.logs if a[0] >= cut]
        for m in st.mets:
            st.mets[m] = [a for a in st.mets[m] if a[0] >= cut]
        return ev


    def set_webhook(self, url):
        self.webhook_url = url or None

    def _notify(self, ev):
        """send the alert to the configured webhook in the background; a failing webhook never disturbs the monitor"""
        url = self.webhook_url
        if not url:
            return

        def post():
            try:
                req = urllib.request.Request(url, data=json.dumps(slack_payload(ev)).encode(), headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=5) as r:
                    status = r.status
            except Exception as e:                                # noqa: BLE001
                status = repr(e)
            self.webhook_log.append(dict(app=ev["app"], window_end=ev["window_end"], status=status))
            del self.webhook_log[:-50]
        threading.Thread(target=post, daemon=True).start()

    # ------------------------------------------------------------------ learning mode, second opinion, explanation
    def _finish_learning(self, st):
        core = [c for c in self._core() if c in self.base_cols]
        rows = [st.base_hist[i] for i in range(len(st.base_hist)) if (st.anchor + DS.STEP * i + DS.WIN - st.anchor) <= self.learning_s]
        if len(rows) < 3:
            rows = st.base_hist[:3]
        X = np.array([[r[c] for c in core] for r in rows], float)
        med = np.nanmedian(X, axis=0)
        mad = np.nanmedian(np.abs(X - med), axis=0)
        st.learn_core = core
        st.learn_median = med
        st.learn_scale = np.maximum(1.4826 * mad, np.maximum(0.05 * np.abs(med), 1e-9))
        st.normal = {c: float(m) for c, m in zip(core, med)}
        st.normal.update({c: float(np.nanmedian([r[c] for r in rows])) for c in self.base_cols if c not in st.normal})

    @staticmethod
    def _core():
        from testbed.validate import CORE
        return CORE

    def _z_score(self, st, row):
        x = np.array([row[c] for c in st.learn_core], float)
        z = np.clip(np.abs((x - st.learn_median) / st.learn_scale), 0, 10)
        z = np.nan_to_num(z, nan=0.0)
        return float(np.sort(z)[-2:].mean())

    def reasons(self, st, x, top=3):
        import shap
        if self.explainer is None:
            self.explainer = shap.TreeExplainer(self.pipeline.steps[-1][1])
        Xi = self.pipeline.steps[0][1].transform(x) if len(self.pipeline.steps) > 1 else x
        sv = self.explainer.shap_values(Xi)
        sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if getattr(sv, "ndim", 2) == 3 else sv)
        contrib = np.asarray(sv).reshape(-1)
        names = self.columns
        agg = {}
        for n, c in zip(names, contrib):
            b = n[5:] if n.startswith("rel__") else n
            agg[b] = agg.get(b, 0.0) + float(c)
        out = []
        last = st.base_hist[-1]
        for b, c in sorted(agg.items(), key=lambda t: -t[1])[:top]:
            if c <= 0:
                continue
            v, normal = last.get(b), getattr(st, "normal", {}).get(b)
            name, unit = plain_name(b)
            if v is None or not np.isfinite(v):
                continue
            if normal is not None and np.isfinite(normal):
                if abs(v - normal) < 0.05 * max(abs(normal), 1e-9) and abs(v - normal) < 1e-6:
                    continue
                verb = "rose" if v > normal else "fell"
                sentence = f"{name} {verb} to {fmt_value(b, v)} (normally {fmt_value(b, normal)})"
            else:
                sentence = f"{name} is {fmt_value(b, v)}"
            out.append(dict(feature=b, plain_name=name, value=float(v), normal=None if normal is None else float(normal), shap=c, sentence=sentence))
        return out

    # ------------------------------------------------------------------ state for the UI
    def state(self):
        with self.lock:
            apps = {}
            for a, st in self.apps.items():
                last = st.windows[-1] if st.windows else None
                idle = st.last_event_at is None or (time.time() - st.last_event_at) > IDLE_AFTER
                apps[a] = dict(app=a, anchor=st.anchor, status=(last["status"] if last else "waiting"),
                               idle=bool(idle), last_event_at=st.last_event_at, learning=dict(done=bool(st.learn_median is not None), progress=(last["learning_progress"] if last else 0.0)),
                               last=last, counters=dict(st.counters),
                               history=[dict(window_index=w["window_index"], window_end=w["window_end"], risk=w["risk"], flagged=w["flagged"], alert=w["alert"], learning=w["learning"], status=w["status"]) for w in st.windows[-120:]],
                               alerts=[dict(w) for w in st.alerts[-30:]])
            cfg = {k: self.cfg[k] for k in ("frozen_at", "model", "feature_set", "n_features", "threshold", "learning_period_s")}
            cfg["alert_rule"] = self.cfg["alert_rule"]["name"]
            return dict(config=cfg, apps=apps)

    def windows(self, app_id):
        with self.lock:
            return [dict(w) for w in self.apps[app_id].windows] if app_id in self.apps else []
