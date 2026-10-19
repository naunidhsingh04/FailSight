"""Live-monitoring core: ingestion with validation, rolling per-system buffers, 30 s windows every 10 s (event time), the saved model with its saved
decision rule, SHAP + plain-language reasons, status/counters and a Server-Sent-Events hub.

The feature arithmetic lives in scripts/window_features.py (shared with the offline equivalence tests); preprocessing is scripts/preprocess.py.
The demo data source is a REPLAY of recorded AnoMod test-split blocks (scripts/replay_scenario.py); this module cannot tell and does not care.
"""
import asyncio
import json
import math
import threading
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np

from . import config, explain_text
from .decision import apply_decision
from .features import missing_report

import sys

sys.path.insert(0, str(config.SCRIPTS))
import window_features as WF   # noqa: E402  (the shared training / live feature code, scripts/window_features.py)

SYSTEMS = ("SN", "TT")
LATENESS = 1.0                 # seconds of event time a window waits for stragglers before it is built
HISTORY = 6                    # sequence length of the deep models (past windows of the same session)
WARMUP_WINDOWS = 2             # counter rates need a previous sample: the first windows of a session have incomplete counter features (not scored, no alerts)
MAX_BATCH = 20000
KEEP_WINDOWS = 600


class IngestError(Exception):
    def __init__(self, status, detail):
        self.status, self.detail = status, detail
        super().__init__(str(detail))


# ------------------------------------------------------------------------------------------------ event validation
def parse_ts(x):
    if isinstance(x, bool):
        raise ValueError("timestamp must be a number (epoch seconds) or an ISO-8601 string")
    if isinstance(x, (int, float)):
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            raise ValueError("timestamp is not finite")
        return v / 1000.0 if v > 1e11 else v
    if isinstance(x, str):
        try:
            d = datetime.fromisoformat(x.replace("Z", "+00:00"))
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            return d.timestamp()
        except ValueError:
            raise ValueError(f"timestamp '{x[:40]}' is not ISO-8601")
    raise ValueError("timestamp is required (epoch seconds or ISO-8601)")


def num(x, name, lo=None, hi=None, allow_none=True):
    if x is None:
        if allow_none:
            return math.nan
        raise ValueError(f"{name} is required")
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{name} must be a number")
    v = float(x)
    if math.isinf(v):
        raise ValueError(f"{name} is not finite")
    if not math.isnan(v) and ((lo is not None and v < lo) or (hi is not None and v > hi)):
        raise ValueError(f"{name}={v:g} is outside [{lo}, {hi}]")
    return v


def clean_api(e):
    if not isinstance(e, dict):
        raise ValueError("event must be a JSON object")
    t = parse_ts(e.get("timestamp"))
    st = e.get("status_code", e.get("status"))
    if st is None or isinstance(st, bool) or not isinstance(st, (int, float)) or int(st) != st or not (100 <= st <= 599):
        raise ValueError("status_code must be an integer between 100 and 599")
    return (t, float(st), num(e.get("latency_ms"), "latency_ms", 0), num(e.get("content_length"), "content_length", 0))


def clean_log(e):
    if not isinstance(e, dict):
        raise ValueError("event must be a JSON object")
    t = parse_ts(e.get("timestamp"))
    lv = e.get("level", "INFO")
    if isinstance(lv, bool) or not isinstance(lv, (str, int)):
        raise ValueError("level must be a string (INFO, WARN, ERROR, DEBUG, CONT, OTHER) or a code 0-5")
    code = WF.level_code(lv)
    if isinstance(lv, int) and not (0 <= lv <= 5):
        raise ValueError("numeric level must be 0-5")
    return (t, code)


def clean_metric(e):
    if not isinstance(e, dict):
        raise ValueError("event must be a JSON object")
    t = parse_ts(e.get("timestamp"))
    name = e.get("metric")
    if not isinstance(name, str) or not name.strip() or len(name) > 300:
        raise ValueError("metric must be a non-empty string")
    v = e.get("value")
    if isinstance(v, str) and v.strip().lower() == "nan":
        v = None
    return (t, name.strip(), str(e.get("series", "0")), num(v, "value"))


CLEANERS = {"api": clean_api, "logs": clean_log, "metrics": clean_metric}


# ------------------------------------------------------------------------------------------------ SSE hub
class Hub:
    def __init__(self):
        self.loop = None
        self.subs = set()
        self.recent = deque(maxlen=400)
        self.lock = threading.Lock()

    def publish(self, event: dict):
        with self.lock:
            self.recent.append(event)
            subs = list(self.subs)
        if self.loop is not None:
            for q in subs:
                self.loop.call_soon_threadsafe(q.put_nowait, event)

    def subscribe(self, replay: int = 60):
        q = asyncio.Queue()
        with self.lock:
            snap = list(self.recent)[-replay:] if replay > 0 else []
            self.subs.add(q)
        return q, snap

    def unsubscribe(self, q):
        with self.lock:
            self.subs.discard(q)


# ------------------------------------------------------------------------------------------------ per-system stream
class Stream:
    def __init__(self, system):
        self.system = system
        self.reset(None)

    def reset(self, anchor, session=None, meta=None):
        self.anchor = anchor
        self.session = session
        self.meta = meta or {}
        self.next_k = 0
        self.api, self.logs = [], []
        self.metrics = {}                 # name -> list of (t, value)
        self.rate_state = {}              # (metric, series) -> (t, v)
        self.watermark = -math.inf
        self.history = []                 # preprocessed vectors of the windows of this session
        self.flags = deque(maxlen=3)
        self.windows = deque(maxlen=KEEP_WINDOWS)
        self.counters = dict(windows=0, alerts=0, correct_alerts=0, false_alarms=0, missed=0, correct_quiet=0, labelled=0, late_dropped=0, before_start=0,
                             events_api=0, events_logs=0, events_metrics=0, rejected=0)
        self.last = None

    # ---- ingestion of already validated rows (shared by the live API and the offline equivalence tests)
    def add_rows(self, kind, rows):
        for row in rows:
            t = row[0]
            if self.anchor is None:
                self.anchor = t
            if t < self.anchor:
                self.counters["before_start"] += 1
                continue
            if t < self.anchor + WF.STEP * self.next_k:
                self.counters["late_dropped"] += 1                  # its window was already built
                continue
            if kind == "api":
                self.api.append(row)
            elif kind == "logs":
                self.logs.append(row)
            else:
                _, name, series, v = row
                if WF.is_counter(name):
                    key = (name, series)
                    r = WF.counter_rate(self.rate_state.get(key), t, v)
                    if not math.isnan(v):
                        self.rate_state[key] = (t, v)
                    v = r
                self.metrics.setdefault(name, []).append((t, v))
            self.watermark = max(self.watermark, t)

    def ready(self):
        return self.anchor is not None and self.anchor + WF.STEP * self.next_k + WF.WIN + LATENESS <= self.watermark

    def window_features(self, k):
        ws, we = WF.window_bounds(self.anchor, k)
        a = [r for r in self.api if ws <= r[0] < we]
        lg = [r for r in self.logs if ws <= r[0] < we]
        mm = {n: [v for (t, v) in rows if ws <= t < we] for n, rows in self.metrics.items()}
        mm = {n: v for n, v in mm.items() if v}
        return WF.build_window_features(api=([r[1] for r in a], [r[2] for r in a], [r[3] for r in a]), logs=[r[1] for r in lg], metrics=mm), (ws, we)

    def advance(self):
        """drop buffered events no later window needs (called after a window was built)"""
        self.next_k += 1
        ws = self.anchor + WF.STEP * self.next_k
        self.api = [r for r in self.api if r[0] >= ws]
        self.logs = [r for r in self.logs if r[0] >= ws]
        for n in list(self.metrics):
            self.metrics[n] = [r for r in self.metrics[n] if r[0] >= ws]


class Monitor:
    def __init__(self, registry, explainer):
        self.reg, self.explainer = registry, explainer
        self.lock = threading.RLock()
        self.hub = Hub()
        self.streams = {s: Stream(s) for s in SYSTEMS}
        self.config = dict(model="XGB", label_version="label_binary")
        self.alerts = deque(maxlen=200)
        self.normal = explain_text.baselines()

    # ---------------------------------------------------------------- configuration
    def monitor_models(self):
        out = []
        for (label, name), e in self.reg.entries.items():
            if label == "label_binary" and e["available"] and e["family"] == "tree":
                out.append(name)
        return out

    def set_config(self, model=None, label_version=None):
        label = label_version or self.config["label_version"]
        name = model or self.config["model"]
        if label != "label_binary":
            raise IngestError(422, "the monitor supports binary (Normal / Anomaly) models: label_version must be 'label_binary'")
        if name not in self.monitor_models():
            raise IngestError(422, f"model '{name}' cannot drive the monitor (needs a servable tree model with SHAP): {self.monitor_models()}")
        with self.lock:
            self.config = dict(model=name, label_version=label)
            for s in self.streams.values():       # a new model starts from a clean history (sequence models need consistent context)
                s.history, s.flags = [], deque(maxlen=3)
        self.hub.publish(dict(type="config", config=self.config))
        return self.config

    # ---------------------------------------------------------------- session control
    def start_session(self, system, anchor, session=None, meta=None):
        with self.lock:
            st = self.streams[system]
            st.reset(parse_ts(anchor) if anchor is not None else None, session or f"{system}-{int(time.time())}", meta)
        self.hub.publish(dict(type="session", system=system, session=self.streams[system].session, meta=meta or {}, anchor=self.streams[system].anchor))
        return dict(system=system, session=self.streams[system].session, anchor=self.streams[system].anchor)

    # ---------------------------------------------------------------- ingestion
    def ingest(self, kind, system, events):
        if system not in SYSTEMS:
            raise IngestError(422, f"system must be one of {SYSTEMS}")
        if not isinstance(events, list):
            raise IngestError(422, "events must be a list")
        if len(events) > MAX_BATCH:
            raise IngestError(413, f"batch too large: {len(events)} events (limit {MAX_BATCH})")
        clean, errors = [], []
        for i, e in enumerate(events):
            try:
                clean.append(CLEANERS[kind](e))
            except ValueError as ex:
                errors.append(dict(index=i, error=str(ex)))
        built = []
        with self.lock:
            st = self.streams[system]
            st.add_rows(kind, clean)
            st.counters[f"events_{kind}"] += len(clean)
            st.counters["rejected"] += len(errors)
            built = self._build_ready(st)
        return dict(system=system, accepted=len(clean), rejected=len(errors), errors=errors[:10], windows_built=len(built), watermark=None if st.watermark == -math.inf else st.watermark)

    def flush(self, system, upto=None):
        if system not in SYSTEMS:
            raise IngestError(422, f"system must be one of {SYSTEMS}")
        with self.lock:
            st = self.streams[system]
            if st.anchor is None:
                return dict(system=system, windows_built=0)
            if upto is not None:
                st.watermark = max(st.watermark, parse_ts(upto))
            st.watermark += LATENESS + 1e-6
            built = self._build_ready(st)
        return dict(system=system, windows_built=len(built))

    # ---------------------------------------------------------------- windows
    def _build_ready(self, st):
        out = []
        while st.ready():
            out.append(self._process_window(st, st.next_k))
            st.advance()
        return out

    def _process_window(self, st, k):
        import pandas as pd
        prep = self.reg.prep
        feats, (ws, we) = st.window_features(k)
        row = {c: feats.get(c, math.nan) for c in prep.raw_cols}
        sysid = 1 if st.system == "TT" else 0
        X = pd.DataFrame([dict(row, system_is_TT=sysid)])
        miss = missing_report(prep, X)[0]
        Xp = prep.transform_array(X[prep.raw_cols].to_numpy(float), [sysid])
        st.history.append(Xp[0])
        hist = st.history[-HISTORY:]
        hist = [hist[0]] * (HISTORY - len(hist)) + hist
        seq = np.stack(hist)[None]
        label, model = self.config["label_version"], self.config["model"]
        P = self.reg.predict_proba(label, model, Xp, seq)
        pred, rule = apply_decision(label, model, P)
        risk = float(P[0, 1])
        warmup = k < WARMUP_WINDOWS
        flagged = bool(pred[0] == 1) and not warmup
        sh = self.explainer.shap_values(label, model, Xp, 1, top=10)
        top = [dict(feature=t["feature"], group=t["group"], shap=t["shap"], scaled_value=t["preprocessed_value"], raw_value=(None if (row.get(t["feature"]) is None or (isinstance(row.get(t["feature"]), float) and math.isnan(row.get(t["feature"])))) else row.get(t["feature"])),
                   normal_value=self.normal.get(st.system, {}).get(t["feature"])) for t in sh["top"]]
        reasons = explain_text.reasons(st.system, top, row, 3)
        if not warmup:
            st.flags.append(flagged)
        f3 = sum(st.flags)
        thr = rule.get("value", 0.5)
        status = "failing" if f3 >= 2 else "watch" if (f3 == 1 or risk >= max(0.5, 0.9 * thr)) else "healthy"
        truth = None
        c = st.counters
        c["windows"] += 1
        if st.meta.get("true_label_binary") is not None and not warmup:
            tb = int(st.meta["true_label_binary"])
            c["labelled"] += 1
            if flagged and tb:
                c["correct_alerts"] += 1
            elif flagged and not tb:
                c["false_alarms"] += 1
            elif (not flagged) and tb:
                c["missed"] += 1
            else:
                c["correct_quiet"] += 1
        if st.meta.get("true_label_binary") is not None:
            truth = dict(label_binary="Anomaly" if int(st.meta["true_label_binary"]) else "Normal", label_level=st.meta.get("true_label_level"), scenario=st.meta.get("scenario"), known_from="recording")
        if flagged:
            c["alerts"] += 1
        ev = dict(type="window", system=st.system, session=st.session, window_index=k, window_start=ws, window_end=we, emitted_at=time.time(), model=model, label_version=label,
                  risk=risk, verdict="Anomaly" if flagged else "Normal", flagged=flagged, status=status, decision_rule=dict(type=rule["type"], value=rule.get("value")),
                  top_features=top, reasons=reasons, summary=(explain_text.summary_sentence(reasons) if flagged else ("Warming up: rates need two samples, so no alert is raised for the first windows." if warmup else "No alert: the signals look like normal operation.")),
                  missing=dict(fraction=miss["missing_fraction"], tier=miss["reliability"], expected=miss["expected_features"], missing=miss["missing_features"], by_group=miss["by_group"]),
                  warmup=warmup, scored=(not warmup), truth=truth, counters=dict(c), n_features_built=int(sum(1 for v in feats.values() if not (isinstance(v, float) and math.isnan(v)))))
        st.last = ev
        st.windows.append(ev)
        if flagged:
            self.alerts.append(ev)
        self.hub.publish(ev)
        return ev

    # ---------------------------------------------------------------- state for page loads
    def state(self):
        with self.lock:
            systems = {}
            for s, st in self.streams.items():
                last = st.last
                systems[s] = dict(session=st.session, meta=st.meta, anchor=st.anchor, status=(last or {}).get("status", "idle"), risk=(last or {}).get("risk"),
                                  counters=dict(st.counters), last=last, history=[dict(window_index=w["window_index"], window_start=w["window_start"], risk=w["risk"], flagged=w["flagged"],
                                                                                         truth=(w["truth"] or {}).get("label_binary")) for w in list(st.windows)[-120:]])
            return dict(config=self.config, models=self.monitor_models(), systems=systems, alerts=list(self.alerts)[-30:])
