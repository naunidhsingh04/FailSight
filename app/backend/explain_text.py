"""Plain-language sentences for an alert, built from the REAL top SHAP features, their actual values and the typical value of Normal training windows
(assets/normal_baselines.json). Nothing is invented: a sentence is only produced for a feature with a measured value."""
import json
import math
import re
from functools import lru_cache

from . import config

# feature -> (plain phrase, unit)
BASE = {
    "api_error_rate": ("the share of failed requests", "ratio"), "api_latency_mean_ms": ("the average response time", "ms"),
    "api_latency_p95_ms": ("the response time of the slowest 5% of requests", "ms"), "api_latency_p99_ms": ("the response time of the slowest 1% of requests", "ms"),
    "api_content_length_mean": ("the average response size", "bytes"), "api_request_count": ("the number of requests", "count"),
    "api_4xx_count": ("requests rejected as invalid (4xx)", "count"), "api_5xx_count": ("requests that failed on the server (5xx)", "count"),
    "log_line_count": ("log lines", "count"), "log_error_count": ("error log lines", "count"), "log_warning_count": ("warning log lines", "count"),
    "was_missing_api": ("whether API data was missing", "flag"), "was_missing_logs": ("whether log data was missing", "flag"), "was_missing_metrics": ("whether metric data was missing", "flag"),
    "system_is_TT": ("which system this is", "flag"),
}
METRIC = {
    "process_resident_memory_bytes": ("memory used by a process", "bytes"), "system_disk_usage_percent": ("disk usage", "percent"),
    "node_filesystem_avail_bytes": ("free disk space", "bytes"), "node_filesystem_size_bytes": ("disk size", "bytes"), "node_cpu_seconds_total": ("CPU activity", "rate"),
    "container_memory_usage_bytes": ("container memory", "bytes"), "container_memory_working_set_bytes": ("container working memory", "bytes"), "process_open_fds": ("open file handles", "count"),
    "container_processes": ("processes in a container", "count"), "node_load5": ("system load (5 min)", "num"), "node_memory_MemAvailable_bytes": ("available memory", "bytes"),
    "node_memory_MemFree_bytes": ("free memory", "bytes"), "container_memory_failcnt": ("container memory-limit hits", "count"), "socialnet_container_memory": ("memory of the application containers", "bytes"),
    "system_network_transmit_bytes": ("network data sent", "bytes"), "system_disk_io_time": ("time the disk was busy", "seconds"), "node_disk_io_time_seconds_total": ("time the disk was busy", "rate"),
    "up": ("services reporting as up", "num"),
}
STAT = {"mean": "the average of", "max": "the peak of", "std": "the variation in"}


@lru_cache(maxsize=1)
def baselines() -> dict:
    p = config.ASSETS / "normal_baselines.json"
    return json.loads(p.read_text()) if p.exists() else {}


def split_feature(feature: str):
    if feature.startswith("m_"):
        base, _, stat = feature[2:].rpartition("__")
        return base, stat
    return feature, None


def phrase_unit(feature: str):
    base, stat = split_feature(feature)
    if feature in BASE:
        return BASE[feature]
    if feature.startswith("m_"):
        b = re.sub(r"\[\d+m\]", "", base)
        for k, v in METRIC.items():
            if b == k or b.endswith(k) or k in b:
                return v
        unit = "bytes" if b.endswith("_bytes") else "percent" if b.endswith("_percent") else "seconds" if "seconds" in b else "num"
        return (b.replace("_", " ").replace("rate(", "rate of ").replace(")", ""), unit)
    return (feature.replace("_", " "), "num")


def fmt_value(v: float, unit: str) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "missing"
    if unit == "ratio":
        return f"{v * 100:.1f}%"
    if unit == "percent":
        return f"{v:.1f}%"
    if unit == "ms":
        return f"{v:.0f} ms" if v >= 10 else f"{v:.1f} ms"
    if unit == "bytes":
        for lim, name in ((1e9, "GB"), (1e6, "MB"), (1e3, "kB")):
            if abs(v) >= lim:
                return f"{v / lim:.1f} {name}"
        return f"{v:.0f} bytes"
    if unit == "count":
        return f"{v:.0f}" if abs(v) >= 10 or float(v).is_integer() else f"{v:.1f}"
    if unit == "seconds":
        return f"{v:.2f} s"
    if unit == "flag":
        return "yes" if v >= 0.5 else "no"
    return f"{v:.3g}"


def describe(feature: str, value, normal, shap: float) -> dict:
    """one feature -> plain name, a value phrase and a full sentence fragment ('the error rate rose to 18% (normally 2%)')"""
    phrase, unit = phrase_unit(feature)
    base, stat = split_feature(feature)
    name = f"{STAT[stat]} {phrase}" if stat in STAT else phrase
    vs, ns = fmt_value(value, unit), fmt_value(normal, unit)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        frag = f"{phrase} was missing in this window"
    elif normal is None or (isinstance(normal, float) and math.isnan(normal)):
        frag = f"{name} was {vs}"
    elif unit == "flag":
        frag = f"{phrase}: {vs}"
    else:
        up = value > normal
        rel = abs(value - normal) / max(abs(normal), 1e-9)
        same = abs(value - normal) <= 1e-9 * max(1.0, abs(normal))
        slight = rel < 0.05 and not same
        verb = ('rose slightly' if up else 'fell slightly') if slight else ('rose' if up else 'fell')
        frag = f"{name} {'was ' + vs + ' (normally ' + ns + ')' if same else verb + ' to ' + vs + ' (normally ' + ns + ')'}"
    return dict(feature=feature, plain_name=name, value=value, normal=normal, value_text=vs, normal_text=ns, direction=("towards failure" if shap > 0 else "towards normal"), sentence=frag)


def reasons(system: str, top: list, raw: dict, k: int = 3) -> list:
    """top = SHAP list (feature, shap); only features pushing towards failure, with a real measured value, make a reason"""
    base = baselines().get(system, {})
    out, seen = [], set()
    for t in top:
        if t["shap"] <= 0 or len(out) >= k:
            continue
        f = t["feature"]
        v = raw.get(f)
        if f.startswith("was_missing") or f == "system_is_TT" or v is None or (isinstance(v, float) and math.isnan(v)):
            continue                                   # a reading that was not measured cannot be described as "rose" or "fell"
        key = split_feature(f)[0]
        if key in seen:                                # one sentence per underlying signal (not mean + max + std of the same one)
            continue
        seen.add(key)
        out.append(describe(f, v, base.get(f), t["shap"]))
    return out


def summary_sentence(rs: list) -> str:
    if not rs:
        return "No single signal stands out; the combination of many small changes pushed the risk up."
    frags = [r["sentence"] for r in rs]
    return "Flagged because " + (frags[0] if len(frags) == 1 else ", ".join(frags[:-1]) + " and " + frags[-1]) + "."
