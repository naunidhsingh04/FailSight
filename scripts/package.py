import sys, os, zipfile, re
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
from common import OUT, LOG
# append every warning captured from the raw console output (TF/absl/Keras print theirs to stderr)
con = OUT / "console_train.txt"
if con.exists():
    seen = set(LOG.read_text(encoding="utf-8").splitlines())
    extra = [l for l in con.read_text(encoding="utf-8", errors="ignore").splitlines() if re.search(r"warn", l, re.I) and l not in seen]
    with open(LOG, "a", encoding="utf-8") as f:
        f.write("\n--- additional warning lines from console_train.txt ---\n" + "\n".join(dict.fromkeys(extra)) + "\n")
zp = OUT / "report_outputs_training.zip"
if zp.exists(): zp.unlink()
files = [p for p in OUT.rglob("*") if p.is_file() and p.suffix.lower() in (".csv", ".png", ".json", ".txt", ".html")
         and p.stat().st_size < 20e6 and "_state" not in p.parts and "models" not in p.parts]
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for p in files: z.write(p, p.relative_to(OUT))
print("zip", zp, round(zp.stat().st_size / 1e6, 2), "MB,", len(files), "files")
for p in sorted(files): print("  ", p.relative_to(OUT))
t = pd.read_csv(OUT / "test_metrics.csv")
t = t[t.variant == "official"]  # no_overlap rows are in the same CSV
cols = ["accuracy", "balanced_accuracy", "macro_f1", "normal_recall", "roc_auc", "pr_auc"]
pd.set_option("display.width", 250)
for l in ("label_binary", "label_level"):
    for sc in ("ALL", "SN", "TT"):
        print(f"\n== TEST {l} scope={sc}"); print(t[(t.label_version == l) & (t.scope == sc)].set_index("model")[cols].round(3).to_string())
