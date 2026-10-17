import shutil, json, pandas as pd, numpy as np
from pathlib import Path
A=Path("../AnoMod"); P=A/"processed"; R=A/"report_outputs"; I=Path("inputs"); I.mkdir(exist_ok=True)
files={}
for s in ["SN","TT"]:
    for f in ["windows.parquet","train_raw.parquet","test_raw.parquet","train_smote_binary.parquet","train_smote_level.parquet","feature_list.csv"]:
        files[f"{s}_{f}"]=P/s/f
for f in ["combined_windows.parquet","combined_train_model.parquet","combined_test_model.parquet","train_smote_binary.parquet","train_smote_level.parquet"]:
    files["combined_"+f.replace("combined_","")]=P/"combined"/f
files["fold_indices.json"]=P/"fold_indices.json"
files["combined_fold_indices.json"]=R/"combined_fold_indices.json"
files["combined_feature_list.csv"]=R/"combined_feature_list.csv"
files["combined_split_counts.csv"]=R/"combined_split_counts.csv"
rows=[]; missing=[]
for n,p in files.items():
    if not p.exists(): missing.append(str(p)); continue
    shutil.copy2(p,I/n)
    if n.endswith(".parquet"): d=pd.read_parquet(I/n); sh=d.shape
    elif n.endswith(".csv"): d=pd.read_csv(I/n); sh=d.shape
    else: sh=(len(json.load(open(I/n))),"keys")
    rows.append((n,sh[0],sh[1]))
print(pd.DataFrame(rows,columns=["file","rows","cols"]).to_string(index=False))
print("MISSING:",missing)
w=pd.read_parquet(I/"combined_windows.parquet")
c=w.split.value_counts(); print(c.to_dict()); tr,te=c["train"],c["test"]; print("train pct",round(100*tr/(tr+te),2))
assert (tr,te)==(2549,647), "split counts differ -> STOP"
for s,g in w.groupby("system"):
    t=(g.split=="train").sum(); print(s,"train",t,"test",len(g)-t,"pct",round(100*t/len(g),2))
