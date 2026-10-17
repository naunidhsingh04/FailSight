"""Step 0: read the raw api / metrics / log files (READ-ONLY) into tidy parquet tables with epoch-second timestamps.

Nothing is dropped: every raw row/line becomes one row here. Timestamps are UTC epoch seconds (float).
Log continuation lines (stack traces etc.) inherit the timestamp of the previous record of the same file;
lines before the first record have no timestamp (t = NaN) and are counted as unassignable.
"""
import calendar
import json
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

TIDY = os.path.join(PROC, "tidy")
os.makedirs(TIDY, exist_ok=True)
MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
SN_RE = re.compile(r"^\[(\d{4})-(\w{3})-(\d{2}) (\d{2}):(\d{2}):(\d{2})\.(\d+)\] <(\w+)>")
TT_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})(?:[.,](\d+))?(Z|[+-]\d{2}:\d{2})?\s*(.*)")
LVL_RE = re.compile(r"\b(ERROR|FATAL|CRITICAL|WARN|WARNING|INFO|NOTE|DEBUG|TRACE)\b", re.I)
# level codes: 0 = continuation line (no own timestamp), 1 INFO, 2 WARN, 3 ERROR, 4 DEBUG, 5 OTHER (record without a recognised level)
LVL = {"ERROR": 3, "FATAL": 3, "CRITICAL": 3, "WARN": 2, "WARNING": 2, "INFO": 1, "NOTE": 1, "DEBUG": 4, "TRACE": 4}
ID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|(?<=/)\d+(?=/|$)|(?<=/)[A-Z]{1,3}\d+(?=/|$)")
RAW_ROWS = []  # (system, datatype, scenario, raw_rows_in_files)


def epoch(ts):
    return (ts - pd.Timestamp("1970-01-01")).dt.total_seconds()


# ----------------------------------------------------------------------------------------------- API
def tidy_api():
    parts = []
    for system in SYSTEMS:
        for rec in discover(system, "api"):
            rows = []
            with open(rec["path"], encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if system == "SN":
                        url, cl, lat = r["endpoint"], r.get("content_length"), r.get("latency_ms")
                    else:
                        url, cl, lat = r["url"], (r.get("response_headers") or {}).get("Content-Length"), np.nan
                    rows.append((r["timestamp"], r["method"], re.sub(r"^https?://[^/?]+", "", url).split("?")[0],
                                 r["status_code"], lat, cl))
            d = pd.DataFrame(rows, columns=["ts", "method", "endpoint", "status_code", "latency_ms", "content_length"])
            d["t"] = epoch(pd.to_datetime(d.ts))
            d["latency_ms"] = pd.to_numeric(d.latency_ms, errors="coerce")
            d["content_length"] = pd.to_numeric(d.content_length, errors="coerce")
            d = d.drop(columns="ts")
            for k in ("scenario", "system"):
                d.insert(0, k, rec[k])
            RAW_ROWS.append((system, "api", rec["scenario"], len(d)))
            parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(os.path.join(TIDY, "api.parquet"), index=False)
    return out


# ----------------------------------------------------------------------------------------------- METRICS
def tidy_metrics():
    parts = []
    for system in SYSTEMS:
        for rec in discover(system, "metrics"):
            d = pd.read_csv(rec["path"], low_memory=False)
            n_raw = len(d)
            if system == "SN":
                metric = os.path.splitext(os.path.basename(rec["path"]))[0]
                t = epoch(pd.to_datetime(d.timestamp))
                lab = d["metric"].fillna("") if "metric" in d else pd.Series([""] * n_raw)
                sid = pd.factorize(lab)[0]
                o = pd.DataFrame({"metric": metric, "t": t, "series_id": sid.astype("int32"),
                                  "value": pd.to_numeric(d.value, errors="coerce")})
            else:
                labs = [c for c in d.columns if c not in ("metric_name", "timestamp", "datetime", "value")]
                sid = d.groupby(labs, dropna=False, sort=False).ngroup()
                o = pd.DataFrame({"metric": d.metric_name, "t": d.timestamp.astype("float64"), "series_id": sid.astype("int32"),
                                  "value": pd.to_numeric(d.value, errors="coerce")})
            for k in ("scenario", "system"):
                o.insert(0, k, rec[k])
            o["src"] = rec["rel"].rsplit("/", 1)[1] if system == "SN" else ""
            RAW_ROWS.append((system, "metrics", rec["scenario"], n_raw))
            parts.append(o)
    out = pd.concat(parts, ignore_index=True).drop(columns="src")
    for c in ("system", "scenario", "metric"):
        out[c] = out[c].astype("category")
    out.to_parquet(os.path.join(TIDY, "metrics.parquet"), index=False)
    return out


# ----------------------------------------------------------------------------------------------- LOGS
def parse_log(rec):
    """-> arrays (n_lines): local_epoch (NaN for continuation), explicit_offset_h (NaN if none), level code."""
    system = rec["system"]
    loc, off, lvl = [], [], []
    with open(rec["path"], encoding="utf-8", errors="replace") as f:
        for line in f:
            if system == "SN":
                m = SN_RE.match(line)
                if m:
                    y, mo, d, h, mi, s, frac, lv = m.groups()
                    loc.append(calendar.timegm((int(y), MONTHS[mo], int(d), int(h), int(mi), int(s))) + int((frac + "000000")[:6]) / 1e6)
                    off.append(np.nan); lvl.append(LVL.get(lv.upper(), 5))
                    continue
            else:
                m = TT_RE.match(line)
                if m:
                    y, mo, d, h, mi, s, frac, tz, rest = m.groups()
                    loc.append(calendar.timegm((int(y), int(mo), int(d), int(h), int(mi), int(s))) + int(((frac or "0") + "000000")[:6]) / 1e6)
                    off.append(np.nan if not tz else (0.0 if tz == "Z" else (1 if tz[0] == "+" else -1) * (int(tz[1:3]) + int(tz[4:6]) / 60)))
                    lm = LVL_RE.search(rest[:70])
                    lvl.append(LVL[lm.group(1).upper()] if lm else 5)
                    continue
            loc.append(np.nan); off.append(np.nan); lvl.append(0)
    return np.array(loc), np.array(off), np.array(lvl, dtype=np.int8)


def pod_kind(rel):
    pod = rel.split("/")[3]
    return "ts-services" if pod.startswith("ts-") else re.sub(r"-\d+$|-[0-9a-f]{8,10}-[a-z0-9]{5}$", "", pod)


def tidy_logs():
    files = []
    for system in SYSTEMS:
        for rec in discover(system, "logs"):
            files.append((rec,) + parse_log(rec))
    # TT: timezone offset per log kind, derived from the data (last record time - file collection time)
    obs = {}
    for rec, loc, off, lvl in files:
        if rec["system"] != "TT":
            continue
        rec_mask = ~np.isnan(loc)
        if not rec_mask.any() or (~np.isnan(off[rec_mask])).any():
            continue
        c = re.search(r"_(\d{8})_(\d{6})\.log$", rec["path"])
        coll = calendar.timegm(datetime.strptime(c.group(1) + c.group(2), "%Y%m%d%H%M%S").timetuple())
        obs.setdefault(pod_kind(rec["rel"]), []).append(int(round((np.nanmax(loc) - coll) / 3600)))
    mode = {k: int(pd.Series(v).mode().iloc[0]) for k, v in obs.items()}
    log("TT log timezone offsets (hours) per log kind - derived: round((last record - file collection time)/1h)")
    for k, v in sorted(obs.items()):
        log(f"   {k:<22} files={len(v):>3} offsets seen={pd.Series(v).value_counts().to_dict()} -> used {mode[k]}")
    parts = []
    for rec, loc, off, lvl in files:
        n = len(loc)
        t = loc.copy()
        if rec["system"] == "TT":
            k = pod_kind(rec["rel"])
            o = np.where(np.isnan(off), mode.get(k, 0), off)
            t = t - o * 3600
        s = pd.Series(t).ffill().to_numpy()  # continuation lines inherit the previous record's UTC time
        service = (re.sub(r"_?\.log$", "", os.path.basename(rec["path"])) if rec["system"] == "SN"
                   else re.sub(r"-[0-9a-f]{8,10}-[a-z0-9]{5}$", "", rec["rel"].split("/")[3]))
        d = pd.DataFrame({"t": s, "level": lvl})
        d.insert(0, "service", service)
        d.insert(0, "scenario", rec["scenario"])
        d.insert(0, "system", rec["system"])
        RAW_ROWS.append((rec["system"], "logs", rec["scenario"], n))
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    for c in ("system", "scenario", "service"):
        out[c] = out[c].astype("category")
    out.to_parquet(os.path.join(TIDY, "logs.parquet"), index=False)
    return out


def main():
    snap = raw_snapshot()
    json.dump(snap, open(os.path.join(PROC, "raw_snapshot_before.json"), "w"), indent=1)
    log("raw snapshot (before):", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "newest_mtime"} for k, v in snap.items()})[:400])
    api = tidy_api(); log(f"api rows {len(api):,}")
    met = tidy_metrics(); log(f"metric rows {len(met):,}  (non-numeric values: {int(met.value.isna().sum())})")
    lg = tidy_logs(); log(f"log lines {len(lg):,}  (no timestamp: {int(lg.t.isna().sum())})")
    pd.DataFrame(RAW_ROWS, columns=["system", "datatype", "scenario", "raw_rows"]).to_csv(os.path.join(PROC, "raw_rows_by_scenario.csv"), index=False)

    log("\nper-run time spans (UTC):  [api] [metrics] [logs]   (hh:mm:ss, date)")
    fmt = lambda x: pd.to_datetime(x, unit="s").strftime("%m-%d %H:%M:%S")
    for (s, sc), a in api.groupby(["system", "scenario"], sort=False):
        m = met[(met.system == s) & (met.scenario == sc)]
        l = lg[(lg.system == s) & (lg.scenario == sc)]
        log(f"{s} {sc:<34} api {fmt(a.t.min())}..{fmt(a.t.max())} | met {fmt(m.t.min())}..{fmt(m.t.max())} | log {fmt(l.t.min())}..{fmt(l.t.max())}")


if __name__ == "__main__":
    main()
