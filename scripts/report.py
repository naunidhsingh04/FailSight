"""Builds all CSV / PNG / JSON outputs from the saved per-model state, then packages the zip."""
import json, pickle, platform, zipfile, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from common import *

ORDER = ["CNN_BiGRU_Att", "CNN_BiLSTM_Att", "Fusion", "Transformer", "Fusion_XGBhead", "CNN_BiGRU_Att_XGBhead",
         "Stacking", "SoftVote"]
PRETTY = {"CNN_BiGRU_Att": "CNN+BiGRU+Att", "CNN_BiLSTM_Att": "CNN+BiLSTM+Att", "Fusion": "Fusion", "Transformer": "Transformer",
          "Fusion_XGBhead": "Fusion+XGB head", "CNN_BiGRU_Att_XGBhead": "CNN-BiGRU-Att+XGB head",
          "Stacking": "Stacking (deep+XGB meta)", "SoftVote": "Soft-voting ensemble"}
SC = ["ALL", "SN", "TT"]


def load_all():
    st = {}
    for l in LABELS:
        for n in ORDER:
            p = STATE / f"{l}__{n}.pkl"
            if p.exists():
                st[(l, n)] = pickle.load(open(p, "rb"))
    return st


def masks(D):
    return {"ALL": np.ones(len(D.test_pos), bool), "SN": D.sys_te == 0, "TT": D.sys_te == 1}


def cv_tables(D, st):
    rows = []
    for (l, n), s in st.items():
        for k, f in s["folds"].items():
            yt, yv = D.ytr[l][f["train_rows"]], D.ytr[l][f["val_rows"]]
            tr = scope_scores(yt, f["train_proba"], D.sys_tr[f["train_rows"]])
            va = scope_scores(yv, f["val_proba"], D.sys_tr[f["val_rows"]])
            for sc in SC:
                rows.append(dict(model=n, label_version=l, scope=sc, fold=k, train_macro_f1=tr[sc][0], train_bal_acc=tr[sc][1],
                                 val_macro_f1=va[sc][0], val_bal_acc=va[sc][1]))
    cv = pd.DataFrame(rows)
    num = ["train_macro_f1", "train_bal_acc", "val_macro_f1", "val_bal_acc"]
    summ = []
    for (n, l, sc), g in cv.groupby(["model", "label_version", "scope"]):
        for stat, fn in (("mean", g[num].mean()), ("std", g[num].std())):
            summ.append(dict(model=n, label_version=l, scope=sc, fold=stat, **fn.to_dict()))
    out = pd.concat([cv.astype({"fold": object}), pd.DataFrame(summ)], ignore_index=True)
    out["_o"] = out.model.map({n: i for i, n in enumerate(ORDER)})
    out = out.sort_values(["label_version", "_o", "scope"], kind="stable").drop(columns="_o")
    out.to_csv(OUT / "cv_scores.csv", index=False)
    return cv


def test_table(D, st):
    mk = masks(D)
    rows = []
    for (l, n), s in st.items():
        for sc in SC:
            m = mk[sc]
            rows.append(dict(model=n, label_version=l, scope=sc, **full_metrics(D.yte[l][m], s["test_proba"][m], CLASS_NAMES[l])))
    t = pd.DataFrame(rows)
    t["_o"] = t.model.map({n: i for i, n in enumerate(ORDER)})
    t = t.sort_values(["label_version", "scope", "_o"], kind="stable").drop(columns="_o")
    t.to_csv(OUT / "test_metrics.csv", index=False)
    return t


def overfit_table(D, st, cv):
    mk = masks(D)
    rows = []
    for (l, n), s in st.items():
        g = cv[(cv.model == n) & (cv.label_version == l)]
        for sc in SC:
            a = g[g.scope == sc]
            ytr_all = D.ytr[l]
            fm = quick(ytr_all[mk_tr(D, sc)], s["final_train_proba"][mk_tr(D, sc)])
            tm = quick(D.yte[l][mk[sc]], s["test_proba"][mk[sc]])
            gap = a.train_macro_f1.mean() - a.val_macro_f1.mean()
            fh = s.get("final_hist")
            rows.append(dict(model=n, label_version=l, scope=sc,
                             cv_train_macro_f1=a.train_macro_f1.mean(), cv_val_macro_f1=a.val_macro_f1.mean(), cv_gap_macro_f1=gap,
                             cv_train_bal_acc=a.train_bal_acc.mean(), cv_val_bal_acc=a.val_bal_acc.mean(),
                             cv_gap_bal_acc=a.train_bal_acc.mean() - a.val_bal_acc.mean(),
                             final_train_macro_f1=fm[0], test_macro_f1=tm[0], train_test_gap_macro_f1=fm[0] - tm[0],
                             final_train_bal_acc=fm[1], test_bal_acc=tm[1], train_test_gap_bal_acc=fm[1] - tm[1],
                             final_last_train_loss=(fh["loss"][-1] if fh else np.nan),
                             diagnosis=("overfit (train-test gap > 0.10)" if fm[0] - tm[0] > 0.10 else
                                        "underfit (train macro-F1 < 0.80)" if fm[0] < 0.80 else "ok")))
    o = pd.DataFrame(rows)
    o.to_csv(OUT / "underfit_overfit.csv", index=False)
    return o


def mk_tr(D, sc):
    return {"ALL": np.ones(len(D.train_pos), bool), "SN": D.sys_tr == 0, "TT": D.sys_tr == 1}[sc]


def ablation(D, st, cv, tt):
    rows = []
    for l in LABELS:
        for base, head in (("Fusion", "Fusion_XGBhead"), ("CNN_BiGRU_Att", "CNN_BiGRU_Att_XGBhead")):
            for variant, n in (("own output layer", base), ("XGBoost head", head)):
                if (l, n) not in st:
                    continue
                for sc in SC:
                    a = cv[(cv.model == n) & (cv.label_version == l) & (cv.scope == sc)]
                    t = tt[(tt.model == n) & (tt.label_version == l) & (tt.scope == sc)].iloc[0]
                    rows.append(dict(base_model=base, label_version=l, scope=sc, variant=variant,
                                     cv_train_macro_f1=a.train_macro_f1.mean(), cv_val_macro_f1=a.val_macro_f1.mean(),
                                     cv_val_bal_acc=a.val_bal_acc.mean(), test_macro_f1=t.macro_f1,
                                     test_balanced_accuracy=t.balanced_accuracy, test_normal_recall=t.normal_recall))
    pd.DataFrame(rows).to_csv(OUT / "ablation_xgboost_head.csv", index=False)


def per_scenario(D, st):
    rows = []
    run = D.w.run.to_numpy()[D.test_pos]
    for (l, n), s in st.items():
        pred = s["test_proba"].argmax(1)
        ok = pred == D.yte[l]
        for r in pd.unique(run):
            m = run == r
            rows.append(dict(model=n, label_version=l, system=("TT" if D.sys_te[m][0] else "SN"), scenario=r,
                             true_class=CLASS_NAMES[l][D.yte[l][m][0]], n_test_windows=int(m.sum()), recall=float(ok[m].mean())))
    pd.DataFrame(rows).to_csv(OUT / "test_recall_per_scenario.csv", index=False)


def arch_time(st):
    a, t = [], []
    for (l, n), s in st.items():
        ud = s.get("units_drop")
        a.append(dict(model=n, label_version=l, config=s.get("cfg"), units=(ud[0] if ud else None), dropout=(ud[1] if ud else None),
                      layers=s["layers"], parameters=s["params"], size_mb=round(s["params"] * 4 / 1e6, 3),
                      note=("deep part shared with base model; + XGBoost 300 rounds depth 6 lr 0.1 on 64-d embedding" if s.get("kind") == "xgb_head"
                            else "XGBoost meta-learner (200 rounds, depth 3) on OOF probs of CNN_BiGRU_Att, Fusion, Transformer" if n == "Stacking"
                            else "average of 40 fold models' probabilities" if n == "SoftVote" else "")))
        eps = [f["epochs"] for f in s["folds"].values()]
        secs = [f["secs"] for f in s["folds"].values()]
        t.append(dict(model=n, label_version=l, cv_total_seconds_incl_cfg_selection=s.get("cv_secs"), fold_seconds_mean=np.mean(secs),
                      epochs_mean=np.mean(eps), seconds_per_epoch=(sum(secs) / max(1, sum(eps))), final_model_seconds=s.get("final_secs"),
                      final_model_epochs=s.get("final_epochs"), xgb_head_seconds=s.get("head_secs")))
    pd.DataFrame(a).to_csv(OUT / "model_architectures.csv", index=False)
    pd.DataFrame(t).to_csv(OUT / "training_time.csv", index=False)


# ------------------------------------------------------------------ plots
def cm_plots(D, st):
    mk = masks(D)
    (OUT / "confusion").mkdir(exist_ok=True)
    for (l, n), s in st.items():
        names = CLASS_NAMES[l]
        pred = s["test_proba"].argmax(1)
        for sc in SC:
            m = mk[sc]
            cm = confusion_matrix(D.yte[l][m], pred[m], labels=range(len(names)))
            pd.DataFrame(cm, index=[f"true_{x}" for x in names], columns=[f"pred_{x}" for x in names]).to_csv(
                OUT / "confusion" / f"cm_{l}_{n}_{sc}.csv")
            if sc != "ALL":
                continue
            fig, ax = plt.subplots(figsize=(4.4, 3.8))
            cn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
            ax.imshow(cn, cmap="Blues", vmin=0, vmax=1)
            for i in range(len(names)):
                for j in range(len(names)):
                    ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cn[i, j] > .5 else "black", fontsize=8)
            ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
            ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{PRETTY[n]} - {l[6:]} (test)", fontsize=9)
            fig.tight_layout(); fig.savefig(OUT / "confusion" / f"cm_{l}_{n}.png", dpi=110); plt.close(fig)


def curve_plots(D, st):
    for l in LABELS:
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
        names = CLASS_NAMES[l]; C = len(names); y = D.yte[l]
        for n in ORDER:
            if (l, n) not in st:
                continue
            P = st[(l, n)]["test_proba"]; P = P / P.sum(1, keepdims=True)
            if C == 2:
                yy, pp = y, P[:, 1]
            else:
                yy, pp = np.eye(C)[y].ravel(), P.ravel()  # micro-average one-vs-rest
            fpr, tpr, _ = roc_curve(yy, pp); pr, rc, _ = precision_recall_curve(yy, pp)
            axs[0].plot(fpr, tpr, label=PRETTY[n], lw=1.2); axs[1].plot(rc, pr, lw=1.2)
        axs[0].plot([0, 1], [0, 1], "k:", lw=.8)
        t = "Anomaly vs Normal" if C == 2 else "micro-average one-vs-rest"
        axs[0].set_title(f"ROC - {l[6:]} ({t}, test)", fontsize=10); axs[0].set_xlabel("FPR"); axs[0].set_ylabel("TPR")
        axs[1].set_title(f"Precision-Recall - {l[6:]} ({t}, test)", fontsize=10); axs[1].set_xlabel("recall"); axs[1].set_ylabel("precision")
        axs[0].legend(fontsize=7, loc="lower right"); fig.tight_layout(); fig.savefig(OUT / f"roc_pr_{l}.png", dpi=120); plt.close(fig)


def bar_plots(tt):
    for l in LABELS:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
        for ax, sc in zip(axs, SC):
            g = tt[(tt.label_version == l) & (tt.scope == sc)].set_index("model").reindex([n for n in ORDER if n in set(tt.model)])
            x = np.arange(len(g)); ax.bar(x - .2, g.balanced_accuracy, .4, label="balanced accuracy", color="#2a6fbb")
            ax.bar(x + .2, g.macro_f1, .4, label="macro-F1", color="#e08a1e")
            ax.set_xticks(x); ax.set_xticklabels([PRETTY[i] for i in g.index], rotation=60, ha="right", fontsize=7)
            ax.set_title(f"{l[6:]} - test, {sc}", fontsize=10); ax.set_ylim(0, 1.02)
        axs[0].legend(fontsize=8, loc="lower left"); fig.tight_layout(); fig.savefig(OUT / f"bar_bal_acc_macro_f1_{l}.png", dpi=120); plt.close(fig)


def loss_plots(st):
    (OUT / "loss").mkdir(exist_ok=True)
    for l in LABELS:
        fig, axs = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True)
        for ax, n in zip(axs, ["CNN_BiGRU_Att", "CNN_BiLSTM_Att", "Fusion", "Transformer"]):
            if (l, n) not in st:
                continue
            fo = st[(l, n)]["folds"]
            for key, col in (("loss", "#2a6fbb"), ("val_loss", "#d1481f")):
                L = max(len(f["hist"][key]) for f in fo.values())
                A = np.full((len(fo), L), np.nan)
                for i, f in enumerate(fo.values()):
                    A[i, :len(f["hist"][key])] = f["hist"][key]
                mu = np.nanmean(A, 0)
                ax.plot(range(1, L + 1), mu, color=col, label=("train (SMOTE)" if key == "loss" else "validation") + " mean of folds")
                ax.fill_between(range(1, L + 1), np.nanmin(A, 0), np.nanmax(A, 0), color=col, alpha=.15)
            fh = st[(l, n)]["final_hist"]["loss"]
            ax.plot(range(1, len(fh) + 1), fh, "k--", lw=1, label="final model (full 80%)")
            ax.set_title(PRETTY[n], fontsize=10); ax.set_xlabel("epoch")
        axs[0].set_ylabel("loss"); axs[0].legend(fontsize=7)
        fig.suptitle(f"Loss per epoch - {l}", fontsize=11); fig.tight_layout()
        fig.savefig(OUT / "loss" / f"loss_{l}.png", dpi=110); plt.close(fig)


def fold_graph():
    p = OUT / "fold_class_counts_for_graph.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.4))
    for ax, l in zip(axs, LABELS):
        g = d[d.label_version == l]
        cl = CLASS_NAMES[l]; w = .8 / len(cl)
        for i, c in enumerate(cl):
            h = g[g["class"] == c].sort_values("fold")
            x = h.fold.to_numpy()
            ax.bar(x + (i - len(cl) / 2 + .5) * w, h.before_smote, w * .9, color=f"C{i}", alpha=.45, label=f"{c} before")
            ax.bar(x + (i - len(cl) / 2 + .5) * w, h.after_smote, w * .9, color="none", edgecolor=f"C{i}", lw=1.4, label=f"{c} after")
        ax.set_xticks(range(1, 11)); ax.set_xlabel("fold (training part)"); ax.set_ylabel("sequences"); ax.set_title(f"{l}: before (filled) vs after SMOTE (outline)", fontsize=10)
        ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "fold_before_after_smote.png", dpi=120); plt.close(fig)


def environment(D):
    import tensorflow as tf, sklearn, imblearn, xgboost, pandas
    lines = [f"Python {platform.python_version()} ({platform.platform()})", f"TensorFlow {tf.__version__}", f"Keras {tf.keras.__version__}",
             f"scikit-learn {sklearn.__version__}", f"imbalanced-learn {imblearn.__version__}", f"xgboost {xgboost.__version__}",
             f"pandas {pandas.__version__}", f"numpy {np.__version__}"]
    for m in ("shap", "lime"):
        try:
            lines.append(f"{m} {__import__(m).__version__}")
        except Exception:
            lines.append(f"{m} NOT INSTALLED")
    lines += [f"Devices: {[d.device_type + ':' + d.name for d in tf.config.list_physical_devices()]}",
              "GPU: none usable (TensorFlow >= 2.11 has no native-Windows GPU support; machine has GeForce MX350) -> trained on CPU",
              f"Seeds: global {SEED}; SMOTE random_state {SEED}; StratifiedKFold(10, shuffle, {SEED}) via combined_fold_indices.json; "
              f"per-model seed = {SEED} + 7*fold + model index; XGBoost random_state {SEED}",
              "Op-level GPU/CPU nondeterminism not forced off, so reruns can differ slightly."]
    (OUT / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(D):
    st = load_all()
    log(f"report: {len(st)} model/label states found")
    cv = cv_tables(D, st); tt = test_table(D, st); over = overfit_table(D, st, cv)
    ablation(D, st, cv, tt); per_scenario(D, st); arch_time(st); environment(D)
    cm_plots(D, st); curve_plots(D, st); bar_plots(tt); loss_plots(st); fold_graph()
    res = {"split": {"train": int(len(D.train_pos)), "test": int(len(D.test_pos))},
           "test_all": {l: tt[(tt.label_version == l) & (tt.scope == "ALL")].set_index("model")[["accuracy", "balanced_accuracy", "macro_f1", "normal_recall", "roc_auc", "pr_auc"]].round(4).to_dict("index") for l in LABELS},
           "config_selected": {f"{l}/{n}": s.get("cfg") for (l, n), s in st.items() if s.get("kind") == "own"},
           "epoch_cap": "see log.txt (reduced from 30 to the value used for time box)"}
    (OUT / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    log("report: CSV/PNG/JSON written")
