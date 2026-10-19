"""Generic (system-agnostic) metric aggregates that exist in any Prometheus / cAdvisor / node-exporter deployment.

SocialNetwork (SN) and TrainTicket (TT) share NOT ONE raw metric name (SN: 'system_cpu_usage', 'socialnet_container_memory'; TT: 'node_cpu_seconds_total',
'container_memory_working_set_bytes'). Both measure the same things though: host CPU busy, container CPU, host memory used, container memory, network
receive/transmit, disk busy / read / write. This script builds one aggregate per concept from each system's own raw names (SN: AnoMod/processed/tidy, TT: the raw
Prometheus CSVs, which keep the labels that tell CPU modes, devices and containers apart) and the same 30 s windows as the window table.
Read-only on AnoMod; writes processed/generic_metric_windows.parquet (git-ignored) and report_outputs/generic_metric_features.csv.

python scripts/generic_metrics.py
"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
ANOMOD = ROOT.parent / "AnoMod"
REP = ROOT / "report_outputs"
OUT_PARQUET = ROOT / "processed" / "generic_metric_windows.parquet"
OUT_CSV = REP / "generic_metric_features.csv"
SCRAPE = 15.0

# concept -> (description, unit note)
CONCEPTS = {
    "host_cpu_busy": "host CPU busy share (0-1)",
    "container_cpu_total": "CPU used by all containers together (cores)",
    "host_mem_used": "host memory used share (0-1)",
    "container_mem_total": "memory used by all containers together (bytes)",
    "net_rx": "host network receive (bytes/s)",
    "net_tx": "host network transmit (bytes/s)",
    "disk_busy": "disk busy share (0-1)",
    "disk_read": "disk read (bytes/s)",
    "disk_write": "disk write (bytes/s)",
}
# what each system offers (None = cannot be built); filled with the exact raw names used
SOURCES = {
    "host_cpu_busy": dict(SN="system_cpu_usage (percent / 100)", TT="node_cpu_seconds_total: rate of all modes except idle and iowait, summed over CPUs, divided by the number of CPUs"),
    "container_cpu_total": dict(SN="socialnet_container_cpu summed over its 31 containers", TT="container_cpu_usage_seconds_total: rate, summed over all named containers"),
    "host_mem_used": dict(SN="system_memory_usage_percent / 100", TT="1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes"),
    "container_mem_total": dict(SN="socialnet_container_memory summed over containers", TT="container_memory_working_set_bytes summed over all named containers"),
    "net_rx": dict(SN="system_network_receive_bytes summed over its 2 series", TT="node_network_receive_bytes_total: rate, physical device only (ens3; loopback, veth, cni, flannel, docker excluded)"),
    "net_tx": dict(SN="system_network_transmit_bytes summed over its 2 series", TT="node_network_transmit_bytes_total: rate, physical device only"),
    "disk_busy": dict(SN="system_disk_io_time", TT="node_disk_io_time_seconds_total: rate (share of time the disk was busy), mean over devices"),
    "disk_read": dict(SN="system_disk_read_bytes", TT="rate(node_disk_read_bytes_total[5m]) summed over devices (already a 5-minute rate)"),
    "disk_write": dict(SN="system_disk_write_bytes", TT="rate(node_disk_written_bytes_total[5m]) summed over devices (already a 5-minute rate)"),
}
NOT_BUILDABLE = [
    ("container_restarts", "no restart counter exists in either system's metrics (no kube_pod_container_status_restarts_total, no container start time)"),
    ("host_load_average", "SN exports system_load1 (1-minute) and TT node_load5 (5-minute): different quantities, so not treated as one aggregate"),
    ("container_network_bytes", "SN has container network bytes; TT only has container network ERROR counters (all zero), so it cannot be built for both"),
    ("trace_span_rate", "only SN has jaeger_spans_rate (constant 0 in every run)"),
]


def window_bounds(w):
    t0 = w.window_start.astype("int64").to_numpy() / 1e9
    t1 = w.window_end.astype("int64").to_numpy() / 1e9
    return t0, t1


def series_stats(ts, vs, t0, t1):
    """mean / max / std (population, ddof=0) of the samples of one aggregate time series inside every window [t0, t1)"""
    order = np.argsort(ts)
    ts, vs = ts[order], vs[order]
    ok = np.isfinite(vs)
    ts, vs = ts[ok], vs[ok]
    lo, hi = np.searchsorted(ts, t0, "left"), np.searchsorted(ts, t1, "left")
    c1 = np.concatenate([[0.0], np.cumsum(vs)])
    c2 = np.concatenate([[0.0], np.cumsum(vs * vs)])
    n = (hi - lo).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (c1[hi] - c1[lo]) / n
        var = np.maximum((c2[hi] - c2[lo]) / n - mean ** 2, 0.0)
    mx = np.full(len(t0), np.nan)
    for i in range(len(t0)):
        if hi[i] > lo[i]:
            mx[i] = vs[lo[i]:hi[i]].max()
    mean[n == 0] = np.nan
    std = np.sqrt(var)
    std[n == 0] = np.nan
    return mean, mx, std


def per_scrape_sum(df, how="sum"):
    """df: columns t, key, value (already per-series values at one time); aggregate over series inside one 15 s scrape bucket -> (t, value)"""
    b = np.floor(df.t.to_numpy() / SCRAPE)
    g = df.assign(b=b).groupby("b")
    t = g.t.median()
    v = g.value.sum() if how == "sum" else g.value.mean()
    return t.to_numpy(), v.to_numpy()


def rates(df, keycols):
    """counter -> per-second rate between consecutive samples of the same series; resets (negative) dropped"""
    df = df.sort_values(keycols + ["t"])
    key = df[keycols].astype(str).agg("|".join, axis=1) if len(keycols) > 1 else df[keycols[0]].astype(str)
    d = df.assign(k=key.to_numpy())
    dt = d.groupby("k").t.diff()
    dv = d.groupby("k").value.diff()
    r = dv / dt
    r[(dv < 0) | (dt <= 0)] = np.nan
    return d.assign(value=r).dropna(subset=["value"])


# ---------------------------------------------------------------------------------------------------- SN (tidy)
def sn_series(tidy, scenario):
    m = tidy[(tidy.system == "SN") & (tidy.scenario == scenario)]
    out = {}

    def one(metric, f=lambda v: v, how="sum"):
        x = m[m.metric == metric]
        if x.empty:
            return None
        t, v = per_scrape_sum(pd.DataFrame(dict(t=x.t.to_numpy(float), value=x.value.to_numpy(float))), how)
        return t, f(v)

    out["host_cpu_busy"] = one("system_cpu_usage", lambda v: v / 100.0)
    out["container_cpu_total"] = one("socialnet_container_cpu")
    out["host_mem_used"] = one("system_memory_usage_percent", lambda v: v / 100.0)
    out["container_mem_total"] = one("socialnet_container_memory")
    out["net_rx"] = one("system_network_receive_bytes")
    out["net_tx"] = one("system_network_transmit_bytes")
    out["disk_busy"] = one("system_disk_io_time")
    out["disk_read"] = one("system_disk_read_bytes")
    out["disk_write"] = one("system_disk_write_bytes")
    return out


# ---------------------------------------------------------------------------------------------------- TT (raw Prometheus CSVs)
TT_COLS = ["metric_name", "timestamp", "value", "instance", "cpu", "mode", "device", "container", "name", "pod", "id"]
TT_METRICS = ["node_cpu_seconds_total", "container_cpu_usage_seconds_total", "node_memory_MemAvailable_bytes", "node_memory_MemTotal_bytes", "container_memory_working_set_bytes",
              "node_network_receive_bytes_total", "node_network_transmit_bytes_total", "node_disk_io_time_seconds_total", "rate(node_disk_read_bytes_total[5m])", "rate(node_disk_written_bytes_total[5m])"]
VIRTUAL = re.compile(r"^(lo|veth|cni|flannel|docker|cali|br-|tunl|kube|vxlan|virbr)")


def tt_series(csv_path):
    d = pd.read_csv(csv_path, usecols=TT_COLS, low_memory=False)
    d = d[d.metric_name.isin(TT_METRICS)].rename(columns={"timestamp": "t"})
    d["t"] = d.t.astype(float)
    d["value"] = d.value.astype(float)
    key = ["metric_name", "instance", "cpu", "mode", "device", "container", "name", "pod", "id"]
    d[key] = d[key].fillna("")
    d = d.groupby(key + ["t"], as_index=False).value.mean()           # node-exporter is scraped by 3 jobs: identical samples are averaged, not summed
    out = {}
    x = d[d.metric_name == "node_cpu_seconds_total"]
    r = rates(x, ["instance", "cpu", "mode"])
    ncpu = max(x.cpu.nunique(), 1) * max(x.instance.nunique(), 1)
    r = r[~r["mode"].isin(["idle", "iowait"])]
    out["host_cpu_busy"] = per_scrape_sum(r[["t", "value"]]) if len(r) else None
    if out["host_cpu_busy"]:
        out["host_cpu_busy"] = (out["host_cpu_busy"][0], out["host_cpu_busy"][1] / ncpu)
    x = d[(d.metric_name == "container_cpu_usage_seconds_total") & (d.container != "")]
    r = rates(x, ["instance", "id", "container", "pod", "name"])
    out["container_cpu_total"] = per_scrape_sum(r[["t", "value"]]) if len(r) else None
    a = d[d.metric_name == "node_memory_MemAvailable_bytes"].groupby("t").value.mean()
    tot = d[d.metric_name == "node_memory_MemTotal_bytes"].groupby("t").value.mean()
    j = pd.concat([a, tot], axis=1, keys=["a", "t"]).dropna()
    mu = pd.DataFrame(dict(t=j.index.to_numpy(float), value=(1 - j.a / j.t).to_numpy()))
    out["host_mem_used"] = per_scrape_sum(mu, "mean") if len(mu) else None
    x = d[(d.metric_name == "container_memory_working_set_bytes") & (d.container != "")]
    out["container_mem_total"] = per_scrape_sum(x[["t", "value"]]) if len(x) else None
    for name, mn in (("net_rx", "node_network_receive_bytes_total"), ("net_tx", "node_network_transmit_bytes_total")):
        x = d[(d.metric_name == mn) & (~d.device.str.match(VIRTUAL))]
        r = rates(x, ["instance", "device"])
        out[name] = per_scrape_sum(r[["t", "value"]]) if len(r) else None
    x = d[d.metric_name == "node_disk_io_time_seconds_total"]
    r = rates(x, ["instance", "device"])
    out["disk_busy"] = per_scrape_sum(r[["t", "value"]], "mean") if len(r) else None
    for name, mn in (("disk_read", "rate(node_disk_read_bytes_total[5m])"), ("disk_write", "rate(node_disk_written_bytes_total[5m])")):
        x = d[d.metric_name == mn]
        out[name] = per_scrape_sum(x[["t", "value"]]) if len(x) else None
    return out


def build():
    import preprocess as P
    t0 = time.time()
    w = P.load_windows()
    tidy = pd.read_parquet(ANOMOD / "processed" / "tidy" / "metrics.parquet")
    tidy["system"] = tidy.system.astype(str); tidy["scenario"] = tidy.scenario.astype(str); tidy["metric"] = tidy.metric.astype(str)
    tt_files = {}
    for p in sorted((ANOMOD / "TT_data" / "metric_data").glob("*/*.csv")):
        tt_files[re.sub(r"(_em)?_\d{8}T\d{6}Z(_em)?$", "", p.parent.name)] = p
    cols = {}
    parts = []
    for run, g in w.groupby("run_id", sort=False):
        system, scenario = run.split("|", 1)
        ser = sn_series(tidy, scenario) if system == "SN" else tt_series(tt_files[scenario])
        a, b = window_bounds(g)
        feat = {"window_id": g.window_id.to_numpy()}
        for c in CONCEPTS:
            s = ser.get(c)
            if s is None:
                for st in ("mean", "max", "std"):
                    feat[f"g_{c}__{st}"] = np.full(len(g), np.nan)
                continue
            mean, mx, sd = series_stats(np.asarray(s[0], float), np.asarray(s[1], float), a, b)
            feat[f"g_{c}__mean"], feat[f"g_{c}__max"], feat[f"g_{c}__std"] = mean, mx, sd
        parts.append(pd.DataFrame(feat))
        print(f"  {run:<40} {len(g):>4} windows  ({time.time() - t0:.0f} s)", flush=True)
    G = pd.concat(parts, ignore_index=True)
    OUT_PARQUET.parent.mkdir(exist_ok=True)
    G.to_parquet(OUT_PARQUET, index=False)
    # report
    full = w[["window_id", "system"]].merge(G, on="window_id")
    rows = []
    for c, desc in CONCEPTS.items():
        row = dict(aggregate=c, description=desc)
        for s in ("SN", "TT"):
            v = full[full.system == s][f"g_{c}__mean"]
            row[f"{s}_buildable"] = bool(v.notna().any())
            row[f"{s}_windows_with_value"] = f"{v.notna().mean():.1%}"
            row[f"{s}_median"] = float(v.median()) if v.notna().any() else np.nan
            row[f"{s}_source"] = SOURCES[c][s]
        row["both_systems"] = row["SN_buildable"] and row["TT_buildable"]
        rows.append(row)
    for c, why in NOT_BUILDABLE:
        rows.append(dict(aggregate=c, description="", SN_buildable=False, TT_buildable=False, both_systems=False, SN_source=why, TT_source=why))
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(pd.DataFrame(rows)[["aggregate", "SN_buildable", "TT_buildable", "SN_windows_with_value", "TT_windows_with_value", "SN_median", "TT_median"]].to_string())
    return G


if __name__ == "__main__":
    build()
