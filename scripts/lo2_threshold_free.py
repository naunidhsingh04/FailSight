"""LO2 threshold-free metrics from the saved window-level probabilities (lo2_window_predictions.csv). No model is run, trained or fitted here.
ROC-AUC and PR-AUC with 1000-resample bootstrap 95% CIs (windows resampled with replacement, seed 42), plus mean predicted P(Anomaly) for Normal
and Anomaly windows. PR-AUC is reported with Anomaly as positive class and also with Normal as positive class (Normal = the minority)."""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

REP = Path(__file__).resolve().parent.parent / "report_outputs"
p = pd.read_csv(REP / "lo2_window_predictions.csv")
N, SEED = 1000, 42
rows = []
for (fs, fl, m), g in p.groupby(["feature_set", "system_flag", "model"], sort=False):
    y, s = g.true_label_binary.to_numpy(int), g.p_anomaly.to_numpy(float)
    rng = np.random.RandomState(SEED)
    idxs = [rng.randint(0, len(y), len(y)) for _ in range(N)]
    def stats(yy, ss):
        if yy.min() == yy.max():
            return (np.nan,) * 3
        return roc_auc_score(yy, ss), average_precision_score(yy, ss), average_precision_score(1 - yy, 1 - ss)
    pt = stats(y, s)
    bs = np.array([stats(y[i], s[i]) for i in idxs])
    row = dict(model=m, feature_set=fs, system_flag=fl, n_windows=len(y), n_normal=int((y == 0).sum()), n_anomaly=int((y == 1).sum()),
               mean_p_anomaly_normal_windows=float(s[y == 0].mean()), mean_p_anomaly_anomaly_windows=float(s[y == 1].mean()),
               mean_p_gap_anomaly_minus_normal=float(s[y == 1].mean() - s[y == 0].mean()), n_distinct_scores=int(len(np.unique(np.round(s, 6)))))
    for j, k in enumerate(("roc_auc", "pr_auc_anomaly_positive", "pr_auc_normal_positive")):
        v = bs[:, j]
        row.update({k: pt[j], f"{k}_boot_mean": np.nanmean(v), f"{k}_ci95_low": np.nanpercentile(v, 2.5), f"{k}_ci95_high": np.nanpercentile(v, 97.5)})
    row["n_valid_resamples"] = int(np.isfinite(bs[:, 0]).sum())
    row["pr_auc_anomaly_chance_level"] = float((y == 1).mean())
    row["pr_auc_normal_chance_level"] = float((y == 0).mean())
    rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(REP / "lo2_threshold_free.csv", index=False)
print(f"saved lo2_threshold_free.csv: {len(out)} rows")

# ---- is the ranking just log volume / scenario length? (raw log features and Spearman correlation of each model's score with log_line_count)
from scipy.stats import spearmanr
W = pd.read_parquet(REP.parent / "processed" / "lo2_windows_A.parquet").set_index("window_id")
yy = W.label_binary.to_numpy(int)
lc = []
for f in ("log_line_count", "log_error_count"):
    lc.append(dict(kind="raw_feature_roc_auc_anomaly_positive", name=f, value=roc_auc_score(yy, -W[f].to_numpy()), note="negated: more log lines/errors = Normal (`correct` runs 60 s)"))
for (fs, fl, m), g in p.groupby(["feature_set", "system_flag", "model"], sort=False):
    x = W.loc[g.window_id, "log_line_count"].to_numpy()
    lc.append(dict(kind="spearman_score_vs_log_line_count", name=f"{m}|{fs}|{fl}", value=spearmanr(g.p_anomaly.to_numpy(), x).statistic, note=""))
pd.DataFrame(lc).to_csv(REP / "lo2_threshold_free_logvolume_check.csv", index=False)
print("saved lo2_threshold_free_logvolume_check.csv")
