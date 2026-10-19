"""LO2 vs AnoMod training distribution of the three log features that the mapping can build -> report_outputs/lo2_distribution_shift.csv"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import preprocess as P

df = P.load_windows()
sp = pd.read_csv(P.REPORTS / "split_ids.csv")
tr = df.merge(sp[["window_id", "split"]], on="window_id").query("split == 'train'")
A = pd.read_parquet(P.ROOT / "processed" / "lo2_windows_A.parquet")
rows = []
for f in ("log_line_count", "log_error_count", "log_warning_count"):
    for s in ("SN", "TT"):
        v = tr[tr.system == s][f].dropna()
        for lab, g in (("Normal", A[A.label_binary == 0]), ("Anomaly", A[A.label_binary == 1])):
            x = g[f]
            rows.append(dict(feature=f, anomod_system=s, anomod_train_median=v.median(), anomod_train_p995=v.quantile(.995), anomod_train_max=v.max(), lo2_class=lab,
                             lo2_median=x.median(), lo2_min=x.min(), lo2_max=x.max(), share_lo2_above_anomod_p995=float((x > v.quantile(.995)).mean()),
                             ratio_lo2_median_to_anomod_median=x.median() / max(v.median(), 1e-9)))
pd.DataFrame(rows).to_csv(P.REPORTS / "lo2_distribution_shift.csv", index=False)
print("saved lo2_distribution_shift.csv")
