"""Shared helpers: paths, run discovery, labels, read-only snapshot of the raw data."""
import hashlib
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "report_outputs")      # CSV / PNG reports (new folder)
PROC = os.path.join(ROOT, "processed")          # parquet / json (new folder)
SYSTEMS = ("SN", "TT")
DATATYPES = {"api_responses": "api", "metric_data": "metrics", "log_data": "logs"}
LEVELS = ["Normal", "Performance", "Service", "Database", "Code"]
LEVEL_NAMES = {"Perf": "Performance", "Svc": "Service", "DB": "Database", "Code": "Code",
               "P": "Performance", "S": "Service", "D": "Database", "C": "Code"}


def log(*a):
    print(*a, flush=True)


def scenario_of(system, run_dir):
    if system == "SN":
        return re.match(r"^(.*?)_\d{8}_\d{6}_", run_dir).group(1)
    return re.sub(r"_\d{8}T\d{6}Z(_em)?$", "", run_dir).replace("_em", "")


def labels_of(system, scenario):
    if scenario.startswith("Normal"):
        return 0, "Normal"
    if system == "SN":
        return 1, LEVEL_NAMES[scenario.split("_")[0]]
    return 1, LEVEL_NAMES[re.match(r"Lv_([CDPS])_", scenario).group(1)]


def discover(system, datatype):
    """Raw input files of one datatype, as dicts (read-only use)."""
    droot = os.path.join(ROOT, f"{system}_data", {v: k for k, v in DATATYPES.items()}[datatype])
    want = {"api": (".jsonl",), "metrics": (".csv",), "logs": (".log",)}[datatype]
    out = []
    for dp, dns, fns in os.walk(droot):
        dns.sort()
        for fn in sorted(fns):
            if fn.lower().endswith(want):
                p = os.path.join(dp, fn)
                run_dir = os.path.relpath(p, droot).split(os.sep)[0]
                sc = scenario_of(system, run_dir)
                lb, ll = labels_of(system, sc)
                out.append(dict(system=system, datatype=datatype, scenario=sc, label_binary=lb, label_level=ll,
                                path=p, rel=os.path.relpath(p, ROOT).replace(os.sep, "/")))
    return out


def raw_snapshot():
    """Fingerprint of the read-only inputs: file count, total bytes, newest mtime (+ sha256 of the combined CSV)."""
    snap = {}
    for d in ("SN_data", "TT_data"):
        n = tot = 0
        newest = 0.0
        for dp, _, fns in os.walk(os.path.join(ROOT, d)):
            for fn in fns:
                st = os.stat(os.path.join(dp, fn))
                n += 1; tot += st.st_size; newest = max(newest, st.st_mtime)
        snap[d] = dict(files=n, bytes=tot, newest_mtime=newest)
    p = os.path.join(ROOT, "anomod_all_csv_raw_combined.csv")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    st = os.stat(p)
    snap["anomod_all_csv_raw_combined.csv"] = dict(bytes=st.st_size, mtime=st.st_mtime, sha256=h.hexdigest())
    return snap
