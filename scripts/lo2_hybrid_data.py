"""LO2 slice for the hybrid models: choose whole runs (cap 150 MB of the downloaded files), split FIRST by whole runs, then build 5-second windows from LO2's OWN signals.

  select : lo2_slice_manifest.csv + lo2_split_ids.csv (run level, written before anything else) + docs/LO2_SLICE.md (sizes)
  parse  : reads each chosen run's logs from the (read-only) LO2 tar, Drain-style log templates, per-window counts, plus the metric series of the same scrape rows from
           LO2/lo2_subset.csv (stable series only) -> processed/lo2_hybrid/windows.parquet, templates.csv, template_counts.npz
Windows = the scrape rows of the LO2 data set (5 s apart; logs aggregated over (t-5 s, t]), the finest window LO2 offers, so no longer window is needed.
Raw LO2 data are only ever READ (SHA-256 before and after: outputs/lo2_hash_before.json / lo2_hash_after.json).

python scripts/lo2_hybrid_data.py select
python scripts/lo2_hybrid_data.py parse
"""
import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REP = ROOT / "report_outputs"
OUT = ROOT / "processed" / "lo2_hybrid"
LO2 = ROOT.parent / "LO2"
CAP_MB = 150.0
SEED = 42
SERVICES = ["client", "code", "key", "refresh-token", "service", "token", "user"]
SOURCES = SERVICES + ["mysql", "locust"]
LEVELS = ["DEBUG", "INFO", "WARN", "ERROR"]
SKIP_FIRST = 100                      # same rule as the LO2 data set's own derived table: the first 100 lines of each file are start-up noise


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(REP / "overnight_log.md", "a", encoding="utf-8") as f:
        f.write("- " + s + "\n")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hash_lo2():
    d = {}
    for p in sorted((LO2 / "lo2_raw").glob("*")) + [LO2 / "lo2_subset.csv", LO2 / "stable_columns.txt"]:
        d[str(p.relative_to(LO2))] = sha(p)
    return d


def select():
    raw = sorted((LO2 / "lo2_raw").glob("light-oauth2-data-*.tar.deflate"))
    runs = [(p.name.split("-")[-1].split(".")[0], p.stat().st_size / 1e6) for p in raw]
    log(f"LO2 folder found: {LO2} ({len(runs)} runs, {sum(s for _, s in runs):.1f} MB of downloaded run files)")
    h0 = hash_lo2()
    (ROOT / "outputs").mkdir(exist_ok=True)
    json.dump(h0, open(ROOT / "outputs" / "lo2_hash_before.json", "w"), indent=0)
    # whole runs, evenly spread over the 33 collection dates (so the slice covers drift), as many as fit under the cap
    n_fit = 0
    for n in range(len(runs), 0, -1):
        pick = [runs[int(round(i * (len(runs) - 1) / max(1, n - 1)))] for i in range(n)] if n > 1 else [runs[0]]
        if sum(s for _, s in pick) <= CAP_MB:
            n_fit = n
            break
    chosen = pick
    rng = np.random.RandomState(SEED)
    test_run = chosen[int(rng.randint(len(chosen)))][0]            # 20% of whole runs = 1 of 5 runs
    sub = pd.read_csv(LO2 / "lo2_subset.csv", usecols=["run_id", "scenario", "is_error", "scenario_http_code_from_name"], dtype={"run_id": str})
    rows = []
    for rid, mb in chosen:
        s = sub[sub.run_id == rid]
        rows.append(dict(run_id=rid, raw_file=f"lo2_raw/light-oauth2-data-{rid}.tar.deflate", size_mb=round(mb, 2), tar_read_from=f"lo2_tar/light-oauth2-data-{rid}.tar",
                         scenarios=s.scenario.nunique(), scrape_rows=len(s), normal_rows=int((s.is_error == 0).sum()), anomalous_rows=int((s.is_error == 1).sum()),
                         role="test" if rid == test_run else "train",
                         reason=f"whole run; one of {len(chosen)} evenly spread over the 33 runs; both normal ('correct') and error scenarios; "
                                f"{'held out as the 20% test part (whole run, chosen by seed 42)' if rid == test_run else 'training part'}"))
    man = pd.DataFrame(rows)
    man.to_csv(REP / "lo2_slice_manifest.csv", index=False)
    man[["run_id", "role"]].assign(level="run", note="split made before any window existed; whole runs, no run in both parts").to_csv(REP / "lo2_split_ids.csv", index=False)
    tot = man.size_mb.sum()
    log(f"slice: {len(chosen)} whole runs, {tot:.1f} MB (cap {CAP_MB:.0f} MB, {100 * tot / sum(s for _, s in runs):.1f}% of the downloaded run files); test run {test_run}; "
        f"6 runs would need {sorted(s for _, s in runs)[:6] and sum(sorted(s for _, s in runs)[:6]):.1f} MB, over the cap, so fewer than 6 runs is forced")
    return man


# ----------------------------------------------------------------------------------------------- Drain-style templates
RX = [(re.compile(r"\{.*\}|\[.*\]"), "<J>"), (re.compile(r"\"[^\"]*\""), "<S>"), (re.compile(r"'[^']*'"), "<S>"),
      (re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F-]{4,}"), "<U>"), (re.compile(r"\b[0-9a-fA-F]{12,}\b"), "<H>"),
      (re.compile(r"\b[A-Za-z0-9_\-]{20,}\b"), "<ID>"), (re.compile(r"\d+(\.\d+)*"), "<N>")]
LINE = re.compile(r"^(\d\d):(\d\d):(\d\d)\.(\d{3}) \[[^\]]*\]\s+(?:\S+\s+)?(TRACE|DEBUG|INFO|WARN|ERROR|FATAL)\s+(\S+)\s*(?:\S+\s+-\s+)?(.*)$")
LOCUST = re.compile(r"^\[(\d{4})-(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d),\d{3}\] \S+/(\w+)/[\w.]+: (.*)$")
MYSQL = re.compile(r"^(\d{4})-(\d\d)-(\d\d)T(\d\d):(\d\d):(\d\d)\.\d+Z \d+ \[(\w+)\] (.*)$")


def mask(msg):
    for rx, rep in RX:
        msg = rx.sub(rep, msg)
    return msg.split()[:14]


class Drain:
    """Drain-style template miner: lines are grouped by (token count, first token); inside a group a line joins the most similar template if at least half of the positions
    agree, and differing positions become wildcards. With learn=False (the test run) the mined templates are frozen: an unseen line gets id -1 ('new template')."""

    def __init__(self, sim=0.5):
        self.groups, self.sim, self.cache, self.n = defaultdict(list), sim, {}, 0

    def match(self, toks, learn):
        key = (len(toks), toks[0] if toks else "")
        ck = (key, tuple(toks))
        if ck in self.cache and (not learn or True):
            return self.cache[ck]
        best, bs = None, -1.0
        for c in self.groups[key]:
            s = sum(1 for a, b in zip(c[0], toks) if a == b) / max(1, len(toks))
            if s > bs:
                best, bs = c, s
        if best is not None and bs >= self.sim:
            if learn:
                best[0] = [a if a == b else "<*>" for a, b in zip(best[0], toks)]
            self.cache[ck] = best[1]
            return best[1]
        if not learn:
            return -1
        c = [list(toks), self.n]
        self.groups[key].append(c)
        self.n += 1
        self.cache[ck] = c[1]
        return c[1]

    def templates(self):
        return {c[1]: " ".join(c[0]) for g in self.groups.values() for c in g}


def parse_sources(tf, members, scrape_t, drains, learn):
    """members: dict source -> TarInfo for one scenario. Returns per-window feature dict list and per-window Counter of (source, template id)."""
    n = len(scrape_t)
    t0 = scrape_t[0]
    day0 = int(t0 // 86400) * 86400
    feats = [dict() for _ in range(n)]
    tcnt = [Counter() for _ in range(n)]
    for src in SOURCES:
        m = members.get(src)
        cnt = np.zeros((n, 7))                          # n, cont, DEBUG, INFO, WARN, ERROR, FATAL+TRACE
        last_w = None
        if m is not None and m.size > 0:
            data = tf.extractfile(m).read().decode("utf-8", "replace").split("\n")[SKIP_FIRST:]
            for ln in data:
                if not ln:
                    continue
                ep, lvl, msg = None, None, None
                if src == "locust":
                    mm = LOCUST.match(ln)
                    if mm:
                        y, mo, d, h, mi, s, lv, msg = mm.groups()
                        ep = dt.datetime(int(y), int(mo), int(d), int(h), int(mi), int(s), tzinfo=dt.timezone.utc).timestamp() - 3 * 3600      # locust local time = UTC+3 (as the data set's own table)
                        lvl = {"WARNING": "WARN", "CRITICAL": "FATAL"}.get(lv, lv)
                elif src == "mysql":
                    mm = MYSQL.match(ln)
                    if mm:
                        y, mo, d, h, mi, s, lv, msg = mm.groups()
                        ep = dt.datetime(int(y), int(mo), int(d), int(h), int(mi), int(s), tzinfo=dt.timezone.utc).timestamp()
                        lvl = "INFO"
                else:
                    if ln[0] == "\t":
                        # continuation (stack trace) line: counted in the window of the previous timestamped line
                        if last_w is not None:
                            cnt[last_w, 1] += 1
                        continue
                    mm = LINE.match(ln)
                    if mm:
                        h, mi, s, ms, lvl, logger, msg = mm.groups()
                        ep = day0 + int(h) * 3600 + int(mi) * 60 + int(s) + int(ms) / 1000
                        if ep - t0 > 12 * 3600:
                            ep -= 86400
                        elif t0 - ep > 12 * 3600:
                            ep += 86400
                        msg = logger + " " + msg
                if ep is None:
                    continue
                w = int(np.searchsorted(scrape_t, ep, side="left"))
                if w == n and ep <= scrape_t[-1] + 5:
                    w = n - 1
                if w >= n or ep <= scrape_t[0] - 5:
                    last_w = None
                    continue
                last_w = w
                cnt[w, 0] += 1
                cnt[w, {"DEBUG": 2, "INFO": 3, "WARN": 4, "ERROR": 5}.get(lvl, 6)] += 1
                tid = drains[src].match(mask(msg), learn)
                tcnt[w][(src, tid)] += 1
        last_w = None
        for w in range(n):
            f = feats[w]
            f[f"log_{src}_n"], f[f"log_{src}_cont"] = cnt[w, 0], cnt[w, 1]
            for j, lv in enumerate(["DEBUG", "INFO", "WARN", "ERROR", "OTHER"]):
                f[f"log_{src}_{lv}"] = cnt[w, 2 + j]
            f[f"log_{src}_error_rate"] = cnt[w, 5] / (cnt[w, 0] + 1.0)
            f[f"log_{src}_warn_rate"] = cnt[w, 4] / (cnt[w, 0] + 1.0)
    for f in feats:
        f["log_total_volume"] = sum(f[f"log_{s}_n"] for s in SOURCES)
    return feats, tcnt


def parse():
    man = pd.read_csv(REP / "lo2_slice_manifest.csv", dtype={"run_id": str})
    split = pd.read_csv(REP / "lo2_split_ids.csv", dtype={"run_id": str})
    role = dict(zip(split.run_id, split.role))
    head = pd.read_csv(LO2 / "lo2_subset.csv", nrows=1)
    stable = set(open(LO2 / "stable_columns.txt").read().split("\n"))
    meta = ["run_id", "scenario", "is_error", "scenario_http_code_from_name", "scenario_order_in_run", "timestamp_epoch", "sec_since_scenario_start"]
    mcols = [c for c in head.columns if c in stable and not c.startswith(("log_", "locust_", "trace_"))]
    sub = pd.read_csv(LO2 / "lo2_subset.csv", usecols=meta + mcols, dtype={"run_id": str})
    sub = sub[sub.run_id.isin(man.run_id)].sort_values(["run_id", "scenario_order_in_run", "timestamp_epoch"]).reset_index(drop=True)
    log(f"metric series read: {len(mcols)} stable series x {len(sub)} scrape rows of the {len(man)} chosen runs")
    drains = {s: Drain() for s in SOURCES}
    rows, tmpl_rows = [], []
    order = list(man[man.role == "train"].run_id) + list(man[man.role == "test"].run_id)      # train runs first: the templates are mined on training runs only
    for rid in order:
        learn = role[rid] == "train"
        t0 = time.time()
        tf = tarfile.open(LO2 / "lo2_tar" / f"light-oauth2-data-{rid}.tar")
        scen = defaultdict(dict)
        for m in tf:
            parts = m.name.split("/")
            if len(parts) == 3 and m.isfile():
                fn = parts[2]
                src = "locust" if fn == "locust.log" else "mysql" if "mysqldb" in fn and fn.endswith(".log") else next((s for s in SERVICES if fn == f"light-oauth2-oauth2-{s}-1.log"), None)
                if src:
                    scen[parts[1]][src] = m
        sr = sub[sub.run_id == rid]
        for sc, g in sr.groupby("scenario", sort=False):
            g = g.sort_values("timestamp_epoch")
            feats, tcnt = parse_sources(tf, scen[sc], g.timestamp_epoch.to_numpy(float), drains, learn)
            for i, (f, c) in enumerate(zip(feats, tcnt)):
                row = g.iloc[i]
                rows.append(dict(window_id=f"LO2|{rid}|{sc}|{i}", run_id=rid, scenario=sc, split=role[rid], window_index=i, scenario_order_in_run=int(row.scenario_order_in_run),
                                 timestamp_epoch=int(row.timestamp_epoch), is_error=int(row.is_error), http_code=row.scenario_http_code_from_name, tcounts=dict(c), **f))
        tf.close()
        log(f"parsed run {rid} ({role[rid]}): {len(sr)} windows in {time.time() - t0:.0f} s; templates so far {sum(len(d.templates()) for d in drains.values())}")
    df = pd.DataFrame(rows)
    # metric columns: counters (names ending in _total/_count/_sum/_bucket) -> per-second rate between consecutive scrapes of the same scenario (first scrape of a scenario = missing)
    M = sub.set_index(["run_id", "scenario", "timestamp_epoch"])[mcols].apply(pd.to_numeric, errors="coerce")
    ctr = [c for c in mcols if c.split("&")[0].endswith(("_total", "_count", "_sum", "_bucket"))]
    key = pd.MultiIndex.from_frame(df[["run_id", "scenario", "timestamp_epoch"]])
    Mw = M.reindex(key)
    Mw.index = df.index
    prev = M.groupby(level=["run_id", "scenario"]).shift(1)
    dtp = sub.assign(_t=sub.timestamp_epoch).set_index(["run_id", "scenario", "timestamp_epoch"])["_t"].groupby(level=["run_id", "scenario"]).diff()
    dtp = dtp.reindex(key).to_numpy()
    prevw = prev.reindex(key)
    for c in ctr:
        r = (Mw[c].to_numpy() - prevw[c].to_numpy()) / dtp
        r[(Mw[c].to_numpy() - prevw[c].to_numpy()) < 0] = np.nan
        Mw[c] = r
    fam = defaultdict(list)
    for c in mcols:
        fam[c.split("&")[0]].append(c)
    F = pd.DataFrame({f"m_{k}": Mw[v].mean(axis=1, skipna=True) for k, v in fam.items()})       # one value per metric family: mean over its series
    df = pd.concat([df.drop(columns=["tcounts"]), F.reset_index(drop=True)], axis=1)
    df["label_binary"] = df.is_error.astype(int)
    df["label_level"] = np.where(df.is_error == 0, "Normal", "HTTP" + df.http_code.fillna(0).astype(int).astype(str))
    # template table + sparse counts
    tmpl = [(s, t, txt) for s, d in drains.items() for t, txt in d.templates().items()]
    index = {(s, t): i for i, (s, t, _) in enumerate(tmpl)}
    unk = len(tmpl)
    C = np.zeros((len(df), len(tmpl) + 1), dtype=np.float32)
    for r, row in enumerate(rows):
        for (s, t), v in row["tcounts"].items():
            C[r, index.get((s, t), unk)] += v
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "windows.parquet")
    np.savez_compressed(OUT / "template_counts.npz", counts=C)
    pd.DataFrame([dict(template_index=i, source=s, template_id=t, template=txt) for i, (s, t, txt) in enumerate(tmpl)]).to_csv(OUT / "templates.csv", index=False)
    log(f"windows {len(df)} ({df.run_id.nunique()} runs), metric families {F.shape[1]}, templates {len(tmpl)}; class counts {df.label_level.value_counts().to_dict()}")
    # window-level split file: the run assignment was fixed earlier; assert nothing changed
    ws = df[["window_id", "run_id", "scenario", "split", "label_binary", "label_level"]]
    assert (ws.groupby("run_id").split.first().to_dict() == role)
    ws.to_csv(REP / "lo2_split_ids.csv", index=False)
    h1 = hash_lo2()
    h0 = json.load(open(ROOT / "outputs" / "lo2_hash_before.json"))
    json.dump(h1, open(ROOT / "outputs" / "lo2_hash_after.json", "w"))
    log("LO2 raw files unchanged (SHA-256 before == after): " + str(h0 == h1))
    if h0 != h1:
        sys.exit("LO2 raw data changed: STOP")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "parse"])
    a = ap.parse_args()
    {"select": select, "parse": parse}[a.stage]()
