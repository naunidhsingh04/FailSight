"""LO2 step 3: evaluate every saved FINAL model (binary labels) unchanged on the LO2 windows. Nothing is trained, tuned or refitted; LO2 is never
used for fitting (the preprocessor and models are the ones saved from the AnoMod training part).

Feature sets: A = logs only (the requested mapping), B = logs + the 19 node_exporter metric names shared with AnoMod (extra).
System flag: LO2 is a Java micro-service stack; it is passed as system_is_TT = 1 (TrainTicket is the Java/Spring system) as the primary choice and
as system_is_TT = 0 (SocialNetwork) as a sensitivity check, because the models use this flag.
"""
import json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_data import *
import v2_deep as DP
import models as M
import joblib, keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

common.LOG = REP / "lo2_log.txt"
L = "label_binary"
N_BOOT = 1000
plt.rcParams.update({"font.family": "serif", "axes.grid": False})
FSETS = {"A_logs_only": "A", "B_logs_plus_node_metrics": "B"}
FLAGS = {"TT": 1, "SN": 0}


def say(*a):
    log(*a)


def lo2_ctx(df):
    run, wi = df.run_id.to_numpy(), df.window_index.to_numpy()
    ctx = np.zeros((len(df), T), dtype=int)
    for i in range(len(df)):
        chain = [i]
        for j in range(1, T):
            p = i - j
            if p >= 0 and run[p] == run[i] and wi[p] == wi[chain[-1]] - 1:
                chain.append(p)
            else:
                break
        chain += [chain[-1]] * (T - len(chain))
        ctx[i] = chain[::-1]
    return ctx


def point(y, P):
    C = 2
    pred = P.argmax(1)
    cm = np.bincount(C * y + pred, minlength=4).reshape(2, 2)
    sup = cm.sum(1)
    rec = np.where(sup > 0, np.diag(cm) / np.maximum(sup, 1), np.nan)
    pre = np.where(cm.sum(0) > 0, np.diag(cm) / np.maximum(cm.sum(0), 1), 0.0)
    f1 = np.where(pre + np.nan_to_num(rec) > 0, 2 * pre * np.nan_to_num(rec) / np.maximum(pre + np.nan_to_num(rec), 1e-12), 0.0)
    Pn = P / P.sum(1, keepdims=True)
    try:
        auc = roc_auc_score(y, Pn[:, 1])
    except Exception:
        auc = np.nan
    return dict(balanced_accuracy=float(np.nanmean(rec)), macro_f1=float(f1[sup > 0].mean()), normal_recall=float(rec[0]), anomaly_recall=float(rec[1]),
                roc_auc=float(auc), accuracy=float(np.diag(cm).sum() / cm.sum()), share_predicted_anomaly=float((pred == 1).mean()))


def resamples(df, seed=SEED):
    """scenario = block; blocks are resampled with replacement within each class (stratified), so both classes are always present.
    NB: the Normal class has ONE block (the `correct` scenario), so its variability is not represented."""
    blocks = {c: [np.where(df.scenario.to_numpy() == s)[0] for s in df[df.label_binary == c].scenario.unique()] for c in (0, 1)}
    rng = np.random.RandomState(seed)
    return [np.concatenate([blocks[c][j] for c in (0, 1) for j in rng.randint(0, len(blocks[c]), len(blocks[c]))]) for _ in range(N_BOOT)], {c: len(b) for c, b in blocks.items()}


def main():
    D = DataV2()
    reg = json.loads((V2 / f"registry__{L}.json").read_text())
    say("=" * 60); say("LO2 external test: every final binary model, unchanged")
    base = {v: pd.read_parquet(ROOT / "processed" / f"lo2_windows_{v}.parquet") for v in ("A", "B")}
    y = base["A"].label_binary.to_numpy(int)
    say(f"LO2 windows {len(y)}: Normal {int((y == 0).sum())} (scenario `correct` only), Anomaly {int((y == 1).sum())} (23 error scenarios, 1 window each)")
    rows, probs_out = [], []
    for fname, v in FSETS.items():
        for flag_name, flag in FLAGS.items():
            fr = base[v].copy()
            fr["system_is_TT"] = flag
            fr["system"] = flag_name
            Wg = D.gprep.transform(fr)
            ctx = lo2_ctx(fr)
            Xseq = Wg[ctx].astype(np.float32)
            P = {}
            order = [n for n in reg if reg[n]["kind"] != "ens"] + [n for n in reg if reg[n]["kind"] == "ens"]
            for n in order:
                s = reg[n]
                fin = load_pkl(V2 / f"s8__{L}__{n}.pkl")
                if s["kind"] == "tab":
                    P[n] = joblib.load(fin["file"]).predict_proba(Wg)
                elif s["kind"] == "deep":
                    P[n] = keras.models.load_model(fin["file"]).predict(Xseq, batch_size=256, verbose=0)
                elif s["kind"] == "head":
                    em = M.embed_model(keras.models.load_model(load_pkl(V2 / f"s8__{L}__{s['base']}.pkl")["file"]))
                    P[n] = joblib.load(fin["file"]).predict_proba(em.predict(Xseq, batch_size=256, verbose=0))
                elif s["ens_kind"] == "stack":
                    P[n] = joblib.load(fin["file"]).predict_proba(np.hstack([P[m] for m in s["members"]]))
                else:
                    ps = []
                    for m in s["members"]:
                        for k in range(1, 11):
                            Wk = D.prep(k).transform(fr)
                            fm = keras.models.load_model(DP.fold_model_path(L, m, reg[m]["tag"], k))
                            ps.append(fm.predict(Wk[ctx].astype(np.float32), batch_size=256, verbose=0))
                    P[n] = np.mean(ps, axis=0)
            rs, nb = resamples(base[v])
            for n in order:
                Pn = P[n] / P[n].sum(1, keepdims=True)
                pt = point(y, Pn)
                bm = {k: np.empty(N_BOOT) for k in ("balanced_accuracy", "macro_f1", "normal_recall", "roc_auc")}
                for b, idx in enumerate(rs):
                    m = point(y[idx], Pn[idx])
                    for k in bm:
                        bm[k][b] = m[k]
                row = dict(model=n, feature_set=fname, system_flag=flag_name, n_windows=len(y), n_normal=int((y == 0).sum()), n_anomaly=int((y == 1).sum()), **pt)
                for k, vv in bm.items():
                    row[f"{k}_ci95_low"], row[f"{k}_ci95_high"], row[f"{k}_boot_mean"] = np.nanpercentile(vv, 2.5), np.nanpercentile(vv, 97.5), np.nanmean(vv)
                rows.append(row)
                if True:                      # keep the window-level probabilities of every variant (used by lo2_threshold_free.py)
                    probs_out.append(pd.DataFrame({"model": n, "feature_set": fname, "system_flag": flag_name, "window_id": fr.window_id, "scenario": fr.scenario, "true_label_binary": y, "p_anomaly": Pn[:, 1]}))
            say(f"  evaluated {len(order)} models: {fname}, system flag {flag_name}; bootstrap blocks per class {nb}")
    res = pd.DataFrame(rows)
    res.to_csv(REP / "lo2_test_metrics.csv", index=False)
    allp = pd.concat(probs_out)
    allp.to_csv(REP / "lo2_window_predictions.csv", index=False)
    allp[(allp.feature_set == "A_logs_only") & (allp.system_flag == "TT")].drop(columns=["feature_set", "system_flag"]).to_csv(REP / "lo2_window_predictions_A_TT.csv", index=False)
    # ---- drop vs the AnoMod test set (official argmax rows of test_metrics.csv, binary)
    an = pd.read_csv(REP / "test_metrics.csv")
    an = an[(an.label_version == "label_binary") & (an.variant == "official")]
    out = []
    for _, r in res.iterrows():
        for scope in ("ALL", "TT"):
            a = an[(an.model == r.model) & (an.scope == scope)].iloc[0]
            for k, ak in (("balanced_accuracy", "balanced_accuracy"), ("macro_f1", "macro_f1"), ("normal_recall", "normal_recall"), ("roc_auc", "roc_auc")):
                out.append(dict(model=r.model, feature_set=r.feature_set, system_flag=r.system_flag, metric=k, lo2=r[k], lo2_ci95_low=r[f"{k}_ci95_low"], lo2_ci95_high=r[f"{k}_ci95_high"],
                                anomod_scope=scope, anomod_test=a[ak], drop_lo2_minus_anomod=r[k] - a[ak]))
    pd.DataFrame(out).to_csv(REP / "lo2_vs_anomod.csv", index=False)
    # ---- figure: balanced accuracy, AnoMod test vs LO2 (A/TT and B/TT)
    prim = res[(res.system_flag == "TT")]
    names = list(res.model.unique())
    fig, ax = plt.subplots(figsize=(12, 4.6))
    x = np.arange(len(names))
    w = 0.27
    a_vals = [an[(an.model == n) & (an.scope == "ALL")].balanced_accuracy.iloc[0] for n in names]
    ax.bar(x - w, a_vals, w, color="#bbbbbb", edgecolor="black", label="AnoMod test (all)")
    for j, (fs, hat, col) in enumerate((("A_logs_only", "//", "#666666"), ("B_logs_plus_node_metrics", "..", "#222222"))):
        d = prim[prim.feature_set == fs].set_index("model").loc[names]
        ax.bar(x + j * w, d.balanced_accuracy, w, color=col, edgecolor="black", hatch=hat, label=f"LO2, {fs} (system flag TT)")
        ax.errorbar(x + j * w, d.balanced_accuracy, yerr=[d.balanced_accuracy - d.balanced_accuracy_ci95_low, d.balanced_accuracy_ci95_high - d.balanced_accuracy], fmt="none", ecolor="black", capsize=2, lw=0.8)
    ax.axhline(0.5, color="black", ls=":", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("balanced accuracy (binary)"); ax.set_ylim(0, 1.05); ax.legend(fontsize=8)
    ax.set_title("AnoMod test vs LO2 external test (28 windows: 5 Normal, 23 Anomaly); dotted = chance", fontsize=10)
    fig.tight_layout(); (FIG / "step4").mkdir(exist_ok=True); fig.savefig(FIG / "step4" / "lo2_vs_anomod_balanced_accuracy.png", dpi=130); plt.close(fig)
    say("saved lo2_test_metrics.csv, lo2_vs_anomod.csv, lo2_window_predictions_A_TT.csv, figures/step4/lo2_vs_anomod_balanced_accuracy.png")


if __name__ == "__main__":
    t0 = time.time()
    main()
    say(f"done in {(time.time() - t0) / 60:.1f} min")
