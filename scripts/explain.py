"""SHAP (gradient explainer, kernel fallback) + LIME on Fusion and CNN_BiGRU_Att; attention weights for 3 test windows."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import keras
from scipy.stats import spearmanr
import models as M
from common import *

TARGETS = ["Fusion", "CNN_BiGRU_Att"]
N_BG, N_EXP, N_LIME = 50, 40, 20


def agg_by_feature(arr, nfeat):
    """arr: (..., nfeat) -> mean |value| per feature"""
    return np.abs(arr).reshape(-1, nfeat).mean(0)


def shap_values(model, bg, X, label):
    import shap
    C = len(CLASS_NAMES[label])
    try:
        ex = shap.GradientExplainer(model, bg)
        sv = ex.shap_values(X)
        sv = np.stack(sv, -1) if isinstance(sv, list) else sv
        return np.abs(sv).reshape(len(X), T, X.shape[2], -1).mean((0, 1, 3)), "GradientExplainer"
    except Exception as e:
        log(f"WARNING SHAP GradientExplainer failed for {model.name}/{label}: {type(e).__name__}: {str(e)[:200]} -> KernelExplainer fallback (small sample)")
        nf = X.shape[2]
        f = lambda Z: model.predict(Z.reshape(-1, T, nf).astype("float32"), verbose=0)
        ex = shap.KernelExplainer(f, bg[:10].reshape(10, -1))
        sv = ex.shap_values(X[:10].reshape(10, -1), nsamples=300)
        sv = np.stack(sv, -1) if isinstance(sv, list) else sv
        return np.abs(sv).reshape(10, T, nf, -1).mean((0, 1, 3)), "KernelExplainer(10 bg, 10 samples, 300 nsamples)"


def lime_values(model, Xtr, X, label, names):
    from lime.lime_tabular import LimeTabularExplainer
    nf = X.shape[2]
    flat_names = [f"{n}@t{t}" for t in range(T) for n in names]
    ex = LimeTabularExplainer(Xtr.reshape(len(Xtr), -1), feature_names=flat_names, class_names=CLASS_NAMES[label],
                              mode="classification", discretize_continuous=False, random_state=SEED)
    f = lambda Z: model.predict(Z.reshape(-1, T, nf).astype("float32"), batch_size=1024, verbose=0)
    imp = np.zeros(nf)
    for i in range(len(X)):
        pc = int(f(X[i:i + 1].reshape(1, -1)).argmax())
        e = ex.explain_instance(X[i].reshape(-1), f, num_features=60, num_samples=500, labels=(pc,))
        for fn, wgt in e.as_list(label=pc):
            base = fn.split("@t")[0] if "@t" in fn else None
            if base is None:  # lime may rewrite names with comparison text when discretised; not the case here
                continue
            imp[names.index(base)] += abs(wgt)
    return imp / len(X)


def main(D):
    names = D.feat_names
    Xtr, Xte = D.fold_arrays("final")
    rng = np.random.RandomState(SEED)
    bg = Xtr[rng.choice(len(Xtr), N_BG, replace=False)]
    rows, att_rows = [], []
    for label in LABELS:
        exp_idx = rng.choice(len(Xte), N_EXP, replace=False)
        for n in TARGETS:
            p = OUT / "models" / f"{label}__{n}.keras"
            if not p.exists():
                log(f"explain: missing {p.name}, skipped"); continue
            model = keras.models.load_model(p)
            log(f"explain: {label} {n}")
            sh, how = shap_values(model, bg, Xte[exp_idx], label)
            li = lime_values(model, Xtr[rng.choice(len(Xtr), 300, replace=False)], Xte[exp_idx[:N_LIME]], label, names)
            ts, tl = np.argsort(-sh)[:10], np.argsort(-li)[:10]
            overlap = len(set(ts) & set(tl))
            rho = spearmanr(sh, li).statistic
            for r in range(10):
                rows.append(dict(label_version=label, model=n, rank=r + 1, shap_feature=names[ts[r]], shap_mean_abs=sh[ts[r]],
                                 lime_feature=names[tl[r]], lime_mean_abs_weight=li[tl[r]], top10_overlap=overlap,
                                 spearman_all_features=rho, shap_method=how, n_shap_samples=N_EXP if "Gradient" in how else 10,
                                 n_lime_instances=N_LIME))
            log(f"  {label} {n}: SHAP via {how}; top-10 overlap SHAP vs LIME = {overlap}/10, Spearman over all features = {rho:.3f}")
            # attention weights for 3 test windows (one Normal, one Service/other class, one TT anomaly)
            am = M.attn_model(model)
            if am is not None:
                yl = D.y["label_level"][D.test_pos]
                pick = [int(np.where(yl == 0)[0][0]), int(np.where(yl == 2)[0][0]), int(np.where((yl == 4) & (D.sys_te == 1))[0][0])]
                w = am.predict(Xte[pick], verbose=0); pr = model.predict(Xte[pick], verbose=0).argmax(1)
                for j, i in enumerate(pick):
                    r = D.w.iloc[D.test_pos[i]]
                    att_rows.append(dict(label_version=label, model=n, window_id=r.window_id, run=r.run, system=r.system,
                                         true=CLASS_NAMES[label][D.y[label][D.test_pos[i]]], pred=CLASS_NAMES[label][pr[j]],
                                         **{f"w_t-{T-1-t}": w[j, t] for t in range(T)}))
    pd.DataFrame(rows).to_csv(OUT / "shap_lime_agreement.csv", index=False)
    pd.DataFrame(att_rows).to_csv(OUT / "attention_weights.csv", index=False)
    if att_rows:
        a = pd.DataFrame(att_rows)
        wc = [c for c in a.columns if c.startswith("w_t")]
        fig, axs = plt.subplots(len(LABELS), len(TARGETS), figsize=(11, 6), squeeze=False)
        for i, l in enumerate(LABELS):
            for j, n in enumerate(TARGETS):
                g = a[(a.label_version == l) & (a.model == n)]
                for _, r in g.iterrows():
                    axs[i, j].plot(range(T), r[wc].astype(float).values, marker="o", label=f"{r.run} ({r.true})")
                axs[i, j].set_title(f"{n} - {l[6:]}", fontsize=9); axs[i, j].set_xticks(range(T)); axs[i, j].set_xticklabels(["t-5", "t-4", "t-3", "t-2", "t-1", "t"])
                axs[i, j].legend(fontsize=6)
        axs[0, 0].set_ylabel("attention weight")
        fig.tight_layout(); fig.savefig(OUT / "attention_weights.png", dpi=110); plt.close(fig)
    log("explain: done")
