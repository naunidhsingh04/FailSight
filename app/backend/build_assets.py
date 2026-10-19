"""One-off asset builder (needs the local training state in outputs/v2_state and the local data; NOT needed to run the app).

Writes into app/backend/assets/:
  model_manifest.json   which saved final model file implements which registry entry (names, family, corrected tag, members, files)
  samples.json          20 real windows from the TEST split (split_ids.csv), balanced over the five classes, so the app works without user data
  curves.json           ROC / PR curve points of every model on the test set (downsampled), computed from the saved test probabilities
  lime_background.npy   (git-ignored) 400 preprocessed TRAIN windows used only as LIME background statistics
Nothing is trained or tuned here.
"""
import json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).parent / "assets"
V2 = ROOT / "outputs" / "v2_state"
LABELS = ["label_binary", "label_level"]
LEVELS = ["Normal", "Performance", "Service", "Database", "Code"]
sys.path.insert(0, str(ROOT / "scripts"))


def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def manifest():
    out = []
    for l in LABELS:
        reg = json.loads((V2 / f"registry__{l}.json").read_text())
        for n, s in reg.items():
            fin = load(V2 / f"s8__{l}__{n}.pkl")
            e = dict(name=n, label_version=l, kind=s["kind"], base=s.get("base"), imbalance_method=s.get("variant"), tag=s["tag"], corrected=s["tag"] == "corr",
                     members=s.get("members", []), files={}, servable=True, reason="")
            fam = {"tab": "linear" if s.get("base") == "LR" else "tree", "deep": "deep", "head": "deep", "ens": "ensemble"}[s["kind"]]
            e["family"] = fam
            if s["kind"] == "ens" and s.get("ens_kind") == "vote":
                e.update(servable=False, reason="Soft voting averages 40 fold models that each need their own fold preprocessor; the fold preprocessors are not shipped, so it can only be reported, not served.")
            else:
                e["files"]["model"] = Path(fin["file"]).name
            if s["kind"] == "head":
                e["files"]["base_model"] = f"{l}__{s['base']}.keras"
                e["head_tag"] = s.get("head_tag")
            if s["kind"] == "ens":
                e["ensemble_kind"] = s.get("ens_kind")
            out.append(e)
    (ASSETS / "model_manifest.json").write_text(json.dumps(out, indent=1))
    print("manifest:", len(out), "entries")


def samples():
    import preprocess as P
    df = P.load_windows()
    sp = pd.read_csv(ROOT / "report_outputs" / "split_ids.csv")
    te = df.merge(sp[["window_id", "split"]], on="window_id")
    te = te[te.split == "test"].reset_index(drop=True)
    rng = np.random.RandomState(42)
    picks = []
    for cls in LEVELS:                                      # 4 per class: 2 SN + 2 TT
        for s in ("SN", "TT"):
            idx = np.where((te.label_level == cls) & (te.system == s))[0]
            picks += list(rng.choice(idx, 2, replace=False))
    feats = P.feature_columns(te)
    rows = []
    for i in picks:
        r = te.iloc[i]
        rows.append(dict(id=f"sample-{len(rows) + 1:02d}", window_id=r.window_id, system=r.system, scenario=r.scenario, label_level=r.label_level,
                         label_binary="Anomaly" if r.label_binary == 1 else "Normal", features={f: (None if pd.isna(r[f]) else float(r[f])) for f in feats}))
    (ASSETS / "samples.json").write_text(json.dumps(dict(source="report_outputs/split_ids.csv, split == test; seed 42; 2 SN + 2 TT windows per class", windows=rows)))
    print("samples:", len(rows))


def curves():
    out = {}
    import preprocess as P
    df = P.load_windows()
    sp = pd.read_csv(ROOT / "report_outputs" / "split_ids.csv")
    te = df.merge(sp[["window_id", "split"]], on="window_id").query("split == 'test'")
    ylev = te.label_level.map({n: i for i, n in enumerate(LEVELS)}).to_numpy()
    ybin = te.label_binary.to_numpy()
    for l in LABELS:
        y = ybin if l == "label_binary" else ylev
        C = 2 if l == "label_binary" else 5
        out[l] = {}
        reg = json.loads((V2 / f"registry__{l}.json").read_text())
        for n in reg:
            P_ = load(V2 / f"s9__{l}__{n}.pkl")["test_proba"]
            P_ = P_ / P_.sum(1, keepdims=True)
            yy, pp = (y, P_[:, 1]) if C == 2 else (np.eye(C)[y].ravel(), P_.ravel())
            fpr, tpr, _ = roc_curve(yy, pp)
            pr, rc, _ = precision_recall_curve(yy, pp)
            def thin(a, b, k=80):
                i = np.unique(np.linspace(0, len(a) - 1, min(k, len(a))).astype(int))
                return [[round(float(a[j]), 4), round(float(b[j]), 4)] for j in i]
            out[l][n] = dict(roc=thin(fpr, tpr), pr=thin(rc, pr))
    (ASSETS / "curves.json").write_text(json.dumps(out))
    print("curves done")


def lime_bg():
    from v2_data import DataV2
    D = DataV2()
    W = D.W("final")[D.train_pos]
    rng = np.random.RandomState(42)
    np.save(ASSETS / "lime_background.npy", W[rng.choice(len(W), 400, replace=False)].astype(np.float32))
    print("lime background saved")


def baselines():
    """median raw value of every feature over the NORMAL TRAINING windows of each system (used for 'normally X' in plain-language alerts)"""
    import preprocess as P
    df = P.load_windows()
    sp = pd.read_csv(ROOT / "report_outputs" / "split_ids.csv")
    d = df.merge(sp[["window_id", "split"]], on="window_id")
    d = d[(d.split == "train") & (d.label_level == "Normal")]
    feats = P.feature_columns(d)
    out = {s: {f: float(g[f].median()) for f in feats if g[f].notna().any()} for s, g in d.groupby("system")}
    (ASSETS / "normal_baselines.json").write_text(json.dumps(dict(source="median over Normal windows of the training split (split_ids.csv)", **out)))
    print("baselines:", {s: len(v) for s, v in out.items()}, {s: int((d.system == s).sum()) for s in out})


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == "baselines":
        baselines()
    else:
        manifest(); samples(); curves(); lime_bg(); baselines()
