"""Stack every CSV in SN_data and TT_data into one CSV, rows and columns exactly as in the raw files.

Added columns (first 5): system, source_file, class_id, class_name, data_type.
class_id: 0 = Normal, 1..12 = the 12 fault types (per system).
"""
import csv
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "anomod_all_csv_raw_combined.csv")
EXTRA = ["class_id"]


def exp_name(system, folder):
    if system == "SN":
        return re.match(r"^(.*?)_\d{8}_\d{6}_", folder).group(1)
    return re.sub(r"_\d{8}T\d{6}Z(_em)?$", "", folder).replace("_em", "")


files = []  # (system, data_type, class_name, path)
for system in ("SN", "TT"):
    base = os.path.join(ROOT, f"{system}_data")
    for dp, _, fn in os.walk(base):
        for f in fn:
            if f.lower().endswith(".csv"):
                rel = os.path.relpath(os.path.join(dp, f), base).split(os.sep)
                files.append((system, rel[0], exp_name(system, rel[1]), os.path.join(dp, f)))

ids = {}
for s in ("SN", "TT"):
    names = sorted({c for sy, _, c, _ in files if sy == s}, key=lambda n: (not n.startswith("Normal"), n))
    assert len(names) == 13, (s, names)
    ids.update({(s, n): i for i, n in enumerate(names)})

cols = []
for *_, p in files:
    for c in pd.read_csv(p, nrows=0, dtype=str).columns:
        if c not in cols:
            cols.append(c)
header = EXTRA + cols
print(len(files), "csv files,", len(cols), "raw columns", flush=True)

total = 0
with open(OUT, "w", newline="", encoding="utf-8") as out:
    w = csv.writer(out)
    w.writerow(header)
    for system, dtype, cname, p in files:
        n = 0
        for ch in pd.read_csv(p, dtype=str, keep_default_na=False, chunksize=200_000):
            ch = ch.reindex(columns=cols, fill_value="")
            ch.insert(0, "class_id", ids[(system, cname)])
            w.writerows(ch.itertuples(index=False, name=None))
            n += len(ch)
        total += n
        print(f"{n:>9}  {os.path.relpath(p, ROOT)}", flush=True)
print("total rows:", total, "->", OUT)
