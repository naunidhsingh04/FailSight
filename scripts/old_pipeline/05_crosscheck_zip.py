"""Final step: raw-data row cross-check (expected 234 CSV files, 4,190,698 rows), read-only proof, then zip the small result files."""
import json
import os
import sys
import zipfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

ZIP = os.path.join(ROOT, "report_outputs_for_report.zip")
LIMIT = 20 * 1024 * 1024


def crosscheck():
    files, rows, cols = [], 0, set()
    for system in SYSTEMS:
        for dp, _, fns in os.walk(os.path.join(ROOT, f"{system}_data")):
            for fn in fns:
                if fn.lower().endswith(".csv"):
                    files.append(os.path.join(dp, fn))
    for p in files:
        cols.update(pd.read_csv(p, nrows=0).columns)
        for ch in pd.read_csv(p, usecols=[0], chunksize=500_000, dtype=str, keep_default_na=False):
            rows += len(ch)
    comb = os.path.join(ROOT, "anomod_all_csv_raw_combined.csv")
    crow = 0
    for ch in pd.read_csv(comb, usecols=["class_id"], chunksize=1_000_000):
        crow += len(ch)
    ccols = len(pd.read_csv(comb, nrows=0).columns)
    before = json.load(open(os.path.join(PROC, "raw_snapshot_before.json")))
    after = raw_snapshot()
    same = {k: after[k] == before[k] for k in after}
    lines = [f"raw CSV files found in SN_data + TT_data : {len(files)} (expected 234) -> {'OK' if len(files) == 234 else 'MISMATCH'}",
             f"raw CSV data rows (quoting-aware count)   : {rows:,} (expected 4,190,698) -> {'OK' if rows == 4190698 else 'MISMATCH'}",
             f"distinct raw CSV column names             : {len(cols)}",
             f"anomod_all_csv_raw_combined.csv rows      : {crow:,} -> {'matches raw rows' if crow == rows else 'MISMATCH with raw rows'}",
             f"anomod_all_csv_raw_combined.csv columns   : {ccols} (= {len(cols)} raw + class_id) -> {'OK' if ccols == len(cols) + 1 else 'MISMATCH'}",
             f"raw folders / combined CSV unchanged since the start of the pipeline (file count, bytes, newest mtime, sha256 of the combined CSV): {same}"]
    open(os.path.join(OUT, "row_crosscheck.txt"), "w").write("\n".join(lines) + "\n")
    pd.DataFrame([dict(check="csv_files", found=len(files), expected=234), dict(check="csv_rows", found=rows, expected=4190698),
                  dict(check="combined_csv_rows", found=crow, expected=rows)]).to_csv(os.path.join(OUT, "row_crosscheck.csv"), index=False)
    # append to preservation log
    pl = pd.read_csv(os.path.join(OUT, "data_preservation_log.csv"))
    pl = pl[pl.step != "final raw data check"]
    add = pd.DataFrame([dict(step="final raw data check", system="-", object=k, note="IDENTICAL to the snapshot taken before the pipeline started" if v else "CHANGED!") for k, v in same.items()]
                       + [dict(step="final raw data check", system="-", object="CSV row cross-check", rows_in=rows, rows_out=crow, rows_lost=rows - crow,
                               note=f"{len(files)} CSV files, {rows:,} rows vs combined CSV {crow:,} rows")])
    pd.concat([pl, add], ignore_index=True).to_csv(os.path.join(OUT, "data_preservation_log.csv"), index=False)
    return lines


def make_zip():
    if os.path.exists(ZIP):
        os.remove(ZIP)
    names = []
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in sorted(os.listdir(OUT)):
            p = os.path.join(OUT, fn)
            if os.path.isfile(p) and fn.lower().endswith((".csv", ".png", ".json", ".txt", ".html")) and os.path.getsize(p) <= LIMIT:
                z.write(p, f"report_outputs/{fn}"); names.append(f"report_outputs/{fn}")
        for fn in sorted(os.listdir(os.path.join(ROOT, "scripts"))):
            p = os.path.join(ROOT, "scripts", fn)
            if fn.endswith(".py") and os.path.getsize(p) <= LIMIT:
                z.write(p, f"scripts/{fn}"); names.append(f"scripts/{fn}")
    return names


if __name__ == "__main__":
    lines = crosscheck()
    log("\n".join(lines))
    names = make_zip()
    log(f"\nzip: {ZIP}  size {os.path.getsize(ZIP) / 1024:.1f} KB, {len(names)} files")
    for n in names:
        log("  " + n)
