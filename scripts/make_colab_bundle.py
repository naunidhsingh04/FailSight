"""Zips the small files the Colab notebook needs into failsight_bundle.zip (repository root, git-ignored).

python scripts/make_colab_bundle.py

Contents: AnoMod window tables (inputs/SN_windows.parquet, TT_windows.parquet), the split and fold files, the testbed window table, the frozen testbed detector, the LO2 slice windows /
templates, every result file (report_outputs/*.csv|json|md|txt, hybrid and LO2 results, proofs.json) and the figures. Raw AnoMod / LO2 data and trained models (except the 1 MB frozen testbed
detector) are NOT included. The zip stays well under 250 MB.
Upload the CONTENTS of the zip to Google Drive: MyDrive/FailSight_bundle/ (see docs/COLAB.md).
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "failsight_bundle.zip"
LIMIT_MB = 250

FILES = ["inputs/SN_windows.parquet", "inputs/TT_windows.parquet", "inputs/combined_windows.parquet", "processed/feature_names.json", "processed/testbed_windows.parquet",
         "processed/lo2_hybrid/windows.parquet", "processed/lo2_hybrid/template_counts.npz", "processed/lo2_hybrid/templates.csv", "models/testbed/detector.joblib"]
RESULT_SUFFIX = {".csv", ".json", ".md", ".txt"}


def main():
    items = [(ROOT / f, f) for f in FILES]
    for p in sorted((ROOT / "report_outputs").iterdir()):
        if p.is_file() and p.suffix in RESULT_SUFFIX:
            items.append((p, f"report_outputs/{p.name}"))
    fig = ROOT / "report_outputs" / "figures"
    if fig.exists():
        for p in sorted(fig.rglob("*")):
            if p.is_file():
                items.append((p, f"report_outputs/figures/{p.relative_to(fig).as_posix()}"))
    missing = [rel for p, rel in items if not p.exists()]
    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p, rel in items:
            if p.exists():
                z.write(p, rel)
    mb = OUT.stat().st_size / 1e6
    print(f"wrote {OUT} ({mb:.1f} MB, limit {LIMIT_MB} MB); {len(items) - len(missing)} files")
    if missing:
        print("not found (skipped):", ", ".join(missing))
    with zipfile.ZipFile(OUT) as z:
        tot = sum(i.file_size for i in z.infolist())
        by_dir = {}
        for i in z.infolist():
            d = i.filename.rsplit("/", 1)[0] if "/" in i.filename else "."
            by_dir.setdefault(d, [0, 0])
            by_dir[d][0] += 1
            by_dir[d][1] += i.file_size
        print(f"uncompressed {tot / 1e6:.1f} MB; contents by folder:")
        for d, (n, s) in sorted(by_dir.items()):
            print(f"  {d:<32} {n:>4} files {s / 1e6:>8.2f} MB")
    if mb > LIMIT_MB:
        sys.exit("bundle too large")


if __name__ == "__main__":
    main()
