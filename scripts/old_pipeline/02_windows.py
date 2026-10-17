"""Step 1: 30 s windows, 10 s step, api + metrics + logs aligned by timestamp inside each run.

Window k covers [run_start + 10k, run_start + 10k + 30). run_start/run_end = earliest/latest timestamp of any
api / metric / log row of the run. Windows k = 0 .. K-2 (K = bin of the last row) are built, so every row falls in
at least one window and every window spans a full 30 s of the run timeline.

Metric features: mean / max / std (population, ddof=0) over all samples of that metric name in the window.
Counter metrics (TT names ending in _total) are first converted to a per-second rate between consecutive samples
of the same series (a cumulative counter's absolute value encodes machine uptime, not the scenario).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

TIDY = os.path.join(PROC, "tidy")
STEP, WIN = 10.0, 30.0
NB = int(WIN // STEP)  # bins per window


def win_sum(a):
    """sum of NB consecutive bins -> array of len(a)-NB+1"""
    c = np.concatenate([[0.0], np.cumsum(a)])
    return c[NB:] - c[:-NB]


def win_max(a):
    out = a[: len(a) - NB + 1].copy()
    for j in range(1, NB):
        out = np.maximum(out, a[j: len(a) - NB + 1 + j])
    return out


def bin_stats(bins, vals, nb):
    n = np.bincount(bins, minlength=nb).astype(float)
    s = np.bincount(bins, weights=vals, minlength=nb)
    q = np.bincount(bins, weights=vals * vals, minlength=nb)
    mx = np.full(nb, -np.inf)
    np.maximum.at(mx, bins, vals)
    return n, s, q, mx


def main():
    api = pd.read_parquet(os.path.join(TIDY, "api.parquet"))
    met = pd.read_parquet(os.path.join(TIDY, "metrics.parquet"))
    lg = pd.read_parquet(os.path.join(TIDY, "logs.parquet"))
    met["system"] = met.system.astype(str); met["scenario"] = met.scenario.astype(str); met["metric"] = met.metric.astype(str)
    lg["system"] = lg.system.astype(str); lg["scenario"] = lg.scenario.astype(str)
    raw_rows = pd.read_csv(os.path.join(PROC, "raw_rows_by_scenario.csv"))

    all_windows, feat_info, membership = {}, {}, []
    counter_note = []
    for system in SYSTEMS:
        runs = []
        for sc in sorted(set(api[api.system == system].scenario)):
            runs.append(sc)
        frames = []
        for sc in runs:
            a = api[(api.system == system) & (api.scenario == sc)].sort_values("t")
            m = met[(met.system == system) & (met.scenario == sc)].copy()
            l = lg[(lg.system == system) & (lg.scenario == sc)]
            lb, ll = labels_of(system, sc)
            # ---- counter -> rate (per series, consecutive samples); rows keep their place, first sample / resets have no rate
            is_counter = m.metric.str.endswith("_total") & ~m.metric.str.startswith("rate(")
            n_nr = 0
            if is_counter.any():
                c = m[is_counter].sort_values(["metric", "series_id", "t"])
                g = c.groupby(["metric", "series_id"], sort=False)
                dv, dt = g.value.diff(), g.t.diff()
                rate = dv / dt
                bad = rate.isna() | (dv < 0) | (dt <= 0)
                n_nr = int(bad.sum())
                m.loc[c.index, "value"] = np.where(bad, np.nan, rate)
            counter_note.append((system, sc, int(is_counter.sum()), n_nr))
            t_all = [a.t.to_numpy(), m.t.to_numpy(), l.t.dropna().to_numpy()]
            run_start = min(x.min() for x in t_all if len(x))
            run_end = max(x.max() for x in t_all if len(x))
            K = int((run_end - run_start) // STEP)           # bin index of the last row
            nbins = K + 1
            nwin = max(nbins - NB + 1, 1)
            if nbins < NB:
                nbins = NB
            w = pd.DataFrame({"system": system, "scenario": sc, "label_binary": np.int8(lb), "label_level": ll,
                              "window_index": np.arange(nwin)})
            w["window_start"] = pd.to_datetime(run_start + STEP * w.window_index, unit="s")
            w["window_end"] = pd.to_datetime(run_start + STEP * w.window_index + WIN, unit="s")
            feats = {}
            # ---------------------------------------------------------------- API (needs raw rows for percentiles)
            ws = run_start + STEP * np.arange(nwin)
            ta = a.t.to_numpy()
            lo, hi = np.searchsorted(ta, ws, "left"), np.searchsorted(ta, ws + WIN, "left")
            sc_a, lat_a, cl_a = a.status_code.to_numpy(), a.latency_ms.to_numpy(), a.content_length.to_numpy()
            f = {k: np.full(nwin, np.nan) for k in ("api_error_rate", "api_latency_mean_ms", "api_latency_p95_ms", "api_latency_p99_ms", "api_content_length_mean")}
            f["api_request_count"] = np.zeros(nwin); f["api_4xx_count"] = np.zeros(nwin); f["api_5xx_count"] = np.zeros(nwin)
            for i in range(nwin):
                if hi[i] == lo[i]:
                    continue
                s = sc_a[lo[i]:hi[i]]
                n4, n5 = int(((s >= 400) & (s < 500)).sum()), int((s >= 500).sum())
                f["api_request_count"][i], f["api_4xx_count"][i], f["api_5xx_count"][i] = len(s), n4, n5
                f["api_error_rate"][i] = (n4 + n5) / len(s)
                la = lat_a[lo[i]:hi[i]]; la = la[~np.isnan(la)]
                if len(la):
                    f["api_latency_mean_ms"][i], f["api_latency_p95_ms"][i], f["api_latency_p99_ms"][i] = la.mean(), np.percentile(la, 95), np.percentile(la, 99)
                ca = cl_a[lo[i]:hi[i]]; ca = ca[~np.isnan(ca)]
                if len(ca):
                    f["api_content_length_mean"][i] = ca.mean()
            feats.update(f)
            api_assigned = int(((ta >= run_start) & (ta < run_start + STEP * (nwin - 1) + WIN)).sum())
            # ---------------------------------------------------------------- LOGS
            lt = l.t.to_numpy(); lv = l.level.to_numpy()
            has_t = ~np.isnan(lt)
            lb_bins = ((lt[has_t] - run_start) // STEP).astype(int)
            lvh = lv[has_t]
            cnt = np.bincount(lb_bins, minlength=nbins).astype(float)
            err = np.bincount(lb_bins[lvh == 3], minlength=nbins).astype(float)
            wrn = np.bincount(lb_bins[lvh == 2], minlength=nbins).astype(float)
            feats["log_line_count"] = win_sum(cnt)[:nwin]
            feats["log_error_count"] = win_sum(err)[:nwin]
            feats["log_warning_count"] = win_sum(wrn)[:nwin]
            # ---------------------------------------------------------------- METRICS
            met_used = 0
            m_assigned = int(len(m))
            for name, g in m.groupby("metric", sort=True):
                v = g.value.to_numpy(); tt = g.t.to_numpy()
                ok = ~np.isnan(v)
                met_used += int(ok.sum())
                if not ok.any():
                    for stat in ("mean", "max", "std"):
                        feats[f"m_{name}__{stat}"] = np.full(nwin, np.nan)
                    continue
                v, tt = v[ok], tt[ok]
                shift = np.median(v)                              # shift for numerical stability of the variance
                b = ((tt - run_start) // STEP).astype(int)
                n, s1, s2, mx = bin_stats(b, v - shift, nbins)
                N, S1, S2, MX = win_sum(n)[:nwin], win_sum(s1)[:nwin], win_sum(s2)[:nwin], win_max(mx)[:nwin]
                with np.errstate(invalid="ignore", divide="ignore"):
                    mean = S1 / N
                    var = np.maximum(S2 / N - mean ** 2, 0)
                has = N > 0
                feats[f"m_{name}__mean"] = np.where(has, mean + shift, np.nan)
                feats[f"m_{name}__max"] = np.where(has, MX + shift, np.nan)
                feats[f"m_{name}__std"] = np.where(has, np.sqrt(var), np.nan)
            w = pd.concat([w, pd.DataFrame(feats)], axis=1)
            frames.append(w)
            # ---------------------------------------------------------------- row -> window membership
            covered_end = run_start + STEP * (nwin - 1) + WIN
            nm = int(((m.t >= run_start) & (m.t < covered_end)).sum())
            nl = int((has_t & (lt < covered_end)).sum())
            membership.append(dict(system=system, scenario=sc, run_start=pd.to_datetime(run_start, unit="s"), run_end=pd.to_datetime(run_end, unit="s"),
                                   windows=nwin,
                                   api_rows=len(a), api_rows_in_windows=api_assigned,
                                   metric_rows=len(m), metric_rows_in_windows=nm, metric_rows_with_value=met_used,
                                   log_lines=len(l), log_lines_with_time=int(has_t.sum()), log_lines_in_windows=nl))
        W = pd.concat(frames, ignore_index=True)
        W.insert(0, "window_id", [f"{system}|{s}|{i}" for s, i in zip(W.scenario, W.window_index)])
        d = os.path.join(PROC, system)
        os.makedirs(d, exist_ok=True)
        W.to_parquet(os.path.join(d, "windows.parquet"), index=False)
        fl = []
        for c in W.columns:
            if c.startswith("api_"):
                grp = "api"
            elif c.startswith("m_"):
                grp = "metrics"
            elif c.startswith("log_"):
                grp = "logs"
            else:
                continue
            fl.append(dict(feature=c, group=grp, description=DESC(c)))
        pd.DataFrame(fl).to_csv(os.path.join(d, "feature_list.csv"), index=False)
        all_windows[system] = W
        log(f"{system}: {len(W):,} windows, {len(fl)} features")

    mem = pd.DataFrame(membership)
    mem.to_csv(os.path.join(PROC, "window_membership_by_run.csv"), index=False)
    # ---- rows -> windows, per system and data type
    rows = []
    for system in SYSTEMS:
        s = mem[mem.system == system]
        W = all_windows[system]
        for dt, raw, inw in (("api", "api_rows", "api_rows_in_windows"), ("metrics", "metric_rows", "metric_rows_in_windows"),
                             ("logs", "log_lines_with_time", "log_lines_in_windows")):
            rr = raw_rows[(raw_rows.system == system) & (raw_rows.datatype == dt)].raw_rows.sum()
            rows.append(dict(system=system, datatype=dt, raw_rows_in_files=int(rr),
                             rows_with_timestamp=int(s[raw].sum()), rows_in_at_least_one_window=int(s[inw].sum()),
                             rows_not_in_any_window=int(s[raw].sum() - s[inw].sum()),
                             rows_without_timestamp=int(rr - s[raw].sum()), windows_built=len(W),
                             windows_containing_rows=int((W["api_request_count"] > 0).sum()) if dt == "api" else None))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "rows_to_windows.csv"), index=False)
    pd.DataFrame(counter_note, columns=["system", "scenario", "counter_rows", "counter_rows_without_rate"]).to_csv(
        os.path.join(OUT, "counter_rate_conversion.csv"), index=False)
    log("\nrows -> windows"); log(pd.DataFrame(rows).to_string())


def DESC(c):
    if c.startswith("m_"):
        name, stat = c[2:].rsplit("__", 1)
        return f"{ {'mean':'mean','max':'maximum','std':'standard deviation (population)'}[stat] } of metric '{name}' over all samples in the 30 s window" + (
            " (per-second rate for *_total counters)" if name.endswith("_total") else "")
    return {
        "api_request_count": "number of API responses in the window",
        "api_error_rate": "(4xx+5xx responses)/requests; NaN when no requests",
        "api_4xx_count": "responses with status 400-499", "api_5xx_count": "responses with status 500-599",
        "api_latency_mean_ms": "mean request latency (ms); only SN logs latency",
        "api_latency_p95_ms": "95th percentile latency (ms)", "api_latency_p99_ms": "99th percentile latency (ms)",
        "api_content_length_mean": "mean response content length (bytes)",
        "log_line_count": "physical log lines (incl. stack-trace continuation lines) attributed to the window, all services",
        "log_error_count": "log records with level ERROR", "log_warning_count": "log records with level WARN/WARNING"}[c]


if __name__ == "__main__":
    main()
