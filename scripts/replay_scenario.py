"""Replay of recorded AnoMod data through the ingestion API, as a real client.

  python scripts/replay_scenario.py --scenario Performance --speed 10 --server http://127.0.0.1:8000

This is a REPLAY of RECORDED AnoMod data, not a live production system. It reads record-level events (one API response / log line / metric sample
per row) from AnoMod/processed/tidy (read-only), restricted to the time span of TEST-SPLIT blocks (split_ids.csv, split == 'test'): a block is 10
consecutive 30 s windows of one run (window k covers [start + 10 k, start + 10 k + 30) so a block spans 120 s). Training blocks are never selected, and
the windows the server builds from a replayed block are exactly the block's 10 test windows (verified by tests/test_replay.py).
Note: neighbouring blocks share 20 s of events at their borders (the known overlap of the block split); the replay still never emits a training BLOCK.

Scenario picker: Normal | Performance | Service | Database | Code (the five classes of label_level). Speeds: 1, 10, 60 (event seconds per wall second).
"""
import argparse, json, math, os, random, sys, threading, time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
ANOMOD = ROOT.parent / "AnoMod" / "processed" / "tidy"
REP = ROOT / "report_outputs"
CLASSES = ["Normal", "Performance", "Service", "Database", "Code"]
SOURCE = "replay of recorded AnoMod test-split blocks (read-only)"


def test_blocks(df=None, split_ids=None):
    """one row per TEST block: system, scenario, run_id, block_id, label_level, start, end (epoch seconds), n_windows, window_ids"""
    import preprocess as P
    df = P.load_windows() if df is None else df
    sp = pd.read_csv(REP / "split_ids.csv") if split_ids is None else split_ids
    d = df.merge(sp[["window_id", "split"]], on="window_id", suffixes=("", "_ids"))
    d = d[d["split_ids" if "split_ids" in d.columns else "split"] == "test"]
    rows = []
    for bid, g in d.groupby("block_id", sort=False):
        rows.append(dict(system=g.system.iloc[0], scenario=g.scenario.iloc[0], run_id=g.run_id.iloc[0], block_id=bid, label_level=g.label_level.iloc[0], label_binary=int(g.label_binary.iloc[0]),
                         start=g.window_start.min().value / 1e9, end=g.window_end.max().value / 1e9, n_windows=len(g), window_ids=g.window_id.tolist(),
                         window_starts=(g.window_start.astype("int64") / 1e9).tolist()))
    return pd.DataFrame(rows)


def plan_segments(scenario_class, system="both", blocks=2, seed=42, tb=None):
    """segments to replay: per system, up to `blocks` test blocks of that class from distinct scenarios where possible, ordered by time"""
    tb = test_blocks() if tb is None else tb
    rng = random.Random(seed)
    plan = {}
    for s in (["SN", "TT"] if system == "both" else [system]):
        cand = tb[(tb.system == s) & (tb.label_level == scenario_class) & (tb.n_windows == 10)]
        if cand.empty:
            continue
        by = {sc: g for sc, g in cand.groupby("scenario")}
        names = sorted(by)
        rng.shuffle(names)
        chosen = []
        i = 0
        while len(chosen) < blocks and any(len(g) for g in by.values()):
            sc = names[i % len(names)]
            g = by[sc]
            if len(g):
                j = rng.randrange(len(g))
                chosen.append(g.iloc[j])
                by[sc] = g.drop(g.index[j])
            i += 1
        plan[s] = sorted(chosen, key=lambda r: r["start"])
    return plan


def assert_test_only(plan, split=None):
    """safety: every planned block must consist of TEST windows only (checked against split_ids.csv independently of how the plan was made)"""
    sp = (pd.read_csv(REP / "split_ids.csv").set_index("window_id").split) if split is None else split
    for s, rows in plan.items():
        for r in rows:
            bad = [w for w in r["window_ids"] if sp[w] != "test"]
            if bad:
                raise AssertionError(f"refusing to replay a block with training windows: {r['block_id']} ({len(bad)} training windows)")
    return True


def load_events(system, scenario, t0, t1):
    """record-level events of one scenario run inside [t0, t1) from the tidy tables (read-only)"""
    filt = [("system", "=", system), ("scenario", "=", scenario)]
    api = pd.read_parquet(ANOMOD / "api.parquet", filters=filt)
    lg = pd.read_parquet(ANOMOD / "logs.parquet", filters=filt, columns=["system", "scenario", "service", "t", "level"])
    mt = pd.read_parquet(ANOMOD / "metrics.parquet", filters=filt)
    api = api[(api.t >= t0) & (api.t < t1)].sort_values("t")
    lg = lg[lg.t.notna() & (lg.t >= t0) & (lg.t < t1)].sort_values("t")
    mt = mt[(mt.t >= t0) & (mt.t < t1)].sort_values("t")
    return api, lg, mt


def _f(x):
    return None if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x)


def to_events(api, lg, mt):
    """DataFrames -> lists of JSON events sorted by time per type"""
    a = [dict(timestamp=float(r.t), status_code=int(r.status_code), latency_ms=_f(r.latency_ms), content_length=_f(r.content_length), endpoint=str(r.endpoint), method=str(r.method)) for r in api.itertuples()]
    l = [dict(timestamp=float(t), level=int(v), service=str(sv)) for t, v, sv in zip(lg.t.to_numpy(), lg.level.to_numpy(), lg.service.astype(str).to_numpy())]
    m = [dict(timestamp=float(t), metric=str(n), series=str(s), value=_f(v)) for t, n, s, v in zip(mt.t.to_numpy(), mt.metric.astype(str).to_numpy(), mt.series_id.to_numpy(), mt["value"].to_numpy())]
    return a, l, m


class Control:
    """optional remote control (pause / speed / stop) polled once per tick when --control URL is given (used by the web UI)"""
    def __init__(self, client, url):
        self.client, self.url, self.state = client, url, dict(paused=False, speed=None, stop=False)

    def poll(self):
        if not self.url:
            return self.state
        try:
            self.state = self.client.get(self.url, timeout=3).json()
        except Exception:
            pass
        return self.state


def send(client, base, path, payload):
    for attempt in range(4):
        try:
            r = client.post(base + path, json=payload, timeout=60)
            if r.status_code >= 500:
                raise RuntimeError(f"HTTP {r.status_code}")
            return r
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"could not POST {path}")


def replay_segment(client, base, system, row, speed, ctrl, log, chunk_event_seconds=1.0):
    """send one test block: reset (anchor = block start, truth from the recording), events paced at `speed`, then flush"""
    tick = lambda: ctrl.poll()
    t0, t1 = float(row["start"]), float(row["end"])
    api, lg, mt = load_events(system, row["scenario"], t0, t1)
    a, l, m = to_events(api, lg, mt)
    send(client, base, "/ingest/reset", dict(system=system, start_ts=t0, scenario=row["scenario"], true_label_level=row["label_level"], true_label_binary=int(row["label_binary"]),
                                              block_ids=[row["block_id"]], source=SOURCE, session=f"{system}-{row['block_id']}"))
    log(f"[{system}] segment {row['block_id']} ({row['label_level']}, {row['scenario']}): {len(a):,} api, {len(l):,} log, {len(m):,} metric events over {t1 - t0:.0f} s of recorded time")
    ia = il = im = 0
    wall0 = time.monotonic()
    paused_total = 0.0
    sim = t0
    while sim < t1:
        st = tick()
        if st.get("stop"):
            return False
        if st.get("paused"):
            time.sleep(0.2)
            paused_total += 0.2
            continue
        sp = float(st.get("speed") or speed)
        step = chunk_event_seconds * sp
        nxt = min(sim + step, t1)
        for kind, evs, idx in (("api", a, ia), ("logs", l, il), ("metrics", m, im)):
            j = idx
            while j < len(evs) and evs[j]["timestamp"] < nxt:
                j += 1
            for k in range(idx, j, 20000):
                send(client, base, f"/ingest/{kind}", dict(system=system, events=evs[k:min(k + 20000, j)]))
            if kind == "api":
                ia = j
            elif kind == "logs":
                il = j
            else:
                im = j
        sim = nxt
        wall_target = (sim - t0) / sp
        slack = wall_target - (time.monotonic() - wall0 - paused_total)
        if slack > 0:
            time.sleep(slack)
    send(client, base, "/ingest/flush", dict(system=system, upto_ts=t1))
    return True


def run(args):
    import httpx
    log = (lambda *a: print(*a, flush=True))
    plan = plan_segments(args.scenario, args.system, args.blocks, args.seed)
    if not plan:
        log(f"no test blocks of class {args.scenario}"); return 1
    assert_test_only(plan)
    log("REPLAY of recorded AnoMod data (test-split blocks only) -> " + args.server)
    for s, rows in plan.items():
        for r in rows:
            log(f"  plan {s}: {r['block_id']}  {r['label_level']:<11} {r['scenario']}  {r['n_windows']} test windows, {r['end'] - r['start']:.0f} s")
    if args.dry_run:
        return 0
    client = httpx.Client()
    ctrl = Control(client, args.control)
    if args.model:
        client.put(args.server + "/api/monitor/config", json=dict(model=args.model, label_version="label_binary"), timeout=30).raise_for_status()
    threads, results = [], {}

    def worker(system, rows):
        for r in rows:
            ok = replay_segment(client, args.server, system, r, args.speed, ctrl, log)
            if not ok:
                break
        results[system] = True

    for s, rows in plan.items():
        th = threading.Thread(target=worker, args=(s, rows), daemon=True)
        th.start(); threads.append(th)
    for th in threads:
        th.join()
    log("replay finished")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=CLASSES, default="Performance")
    ap.add_argument("--system", choices=["SN", "TT", "both"], default="both")
    ap.add_argument("--speed", type=float, choices=[1, 10, 60], default=10)
    ap.add_argument("--blocks", type=int, default=2, help="test blocks per system (each block = 120 s of recorded data)")
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=None, help="monitor model (a tree model, e.g. XGB)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--control", default=None, help="URL polled for {paused, speed, stop} (used by the web UI)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan (blocks that would be replayed) and exit")
    sys.exit(run(ap.parse_args()))


if __name__ == "__main__":
    main()
