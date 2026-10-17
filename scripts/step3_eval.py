"""Step 3: test-set robustness. NO retraining, NO tuning: uses the saved test probabilities (outputs/v2_state/s9__*), the final models in models/
and the out-of-fold predictions (report_outputs/oof_predictions.parquet).

python scripts/step3_eval.py [--parts bootstrap,calibration,explain,errors,summary]
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_data import *                      # DataV2, state loaders, REP, V2 ...
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression

ap = argparse.ArgumentParser()
ap.add_argument("--parts", default="bootstrap,calibration,explain,errors,summary")
ARGS = ap.parse_args()
PARTS = ARGS.parts.split(",")
S3 = FIG / "step3"
S3.mkdir(exist_ok=True)
N_BOOT = 1000
plt.rcParams.update({"font.family": "serif", "axes.grid": False, "figure.dpi": 110})
GRAYS = ["#000000", "#555555", "#999999", "#c8c8c8"]
LAB = ["label_binary", "label_level"]
STEP3_LOG = REP / "step3_log.txt"


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(STEP3_LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M:%S ") + s + "\n")


D = DataV2()
df = D.df
te = D.test_pos
blocks_te = df.block_id.to_numpy()[te]
sys_te = D.sys_te
reg = {l: json.loads((V2 / f"registry__{l}.json").read_text()) for l in LAB}
TESTP = {l: {n: load_pkl(V2 / f"s9__{l}__{n}.pkl")["test_proba"] for n in reg[l]} for l in LAB}
OOF = pd.read_parquet(REP / "oof_predictions.parquet")
SCOPES = {"ALL": np.ones(len(te), bool), "SN": sys_te == 0, "TT": sys_te == 1}


def classes(l):
    return CLASS_NAMES[l]


# ------------------------------------------------------------------------------------------------ metric kernels (numpy)
def conf(y, p, C):
    return np.bincount(C * y + p, minlength=C * C).reshape(C, C)


def point_metrics(y, P, C):
    pred = P.argmax(1)
    cm = conf(y, pred, C)
    sup = cm.sum(1)
    rec = np.where(sup > 0, np.diag(cm) / np.maximum(sup, 1), np.nan)
    pre = np.where(cm.sum(0) > 0, np.diag(cm) / np.maximum(cm.sum(0), 1), 0.0)
    f1 = np.where(pre + np.nan_to_num(rec) > 0, 2 * pre * np.nan_to_num(rec) / np.maximum(pre + np.nan_to_num(rec), 1e-12), 0.0)
    present = sup > 0
    try:
        Pn = P / P.sum(1, keepdims=True)
        auc = roc_auc_score(y, Pn[:, 1]) if C == 2 else roc_auc_score(y, Pn, multi_class="ovr", average="macro", labels=list(range(C)))
    except Exception:
        auc = np.nan
    return dict(balanced_accuracy=float(np.nanmean(rec)), macro_f1=float(f1[present].mean()), normal_recall=float(rec[0]), roc_auc=float(auc),
                accuracy=float(np.diag(cm).sum() / cm.sum())), rec, pre, f1


# ------------------------------------------------------------------------------------------------ 1+2: block bootstrap, per-system, per-class, paired
def resamples(scope, seed=SEED):
    """index arrays into the test rows. Whole blocks (10 consecutive windows of one run) are drawn with replacement, STRATIFIED BY RUN: each run
    keeps its number of test blocks, so every class stays present in every resample (a plain block bootstrap drops all Normal blocks in many
    resamples because the 36 Normal test windows sit in only a few blocks)."""
    m = SCOPES[scope]
    runs_te = df.run_id.to_numpy()[te]
    per_run = {}
    for i in np.where(m)[0]:
        per_run.setdefault(runs_te[i], {}).setdefault(blocks_te[i], []).append(i)
    groups = [[np.array(v) for v in blk.values()] for blk in per_run.values()]
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(N_BOOT):
        parts = []
        for arrs in groups:
            for j in rng.randint(0, len(arrs), len(arrs)):
                parts.append(arrs[j])
        out.append(np.concatenate(parts))
    return out, sum(len(g) for g in groups)


def part_bootstrap():
    say("=" * 60); say(f"BOOTSTRAP: {N_BOOT} resamples of the test windows by block, stratified by run (seed {SEED}); same resamples for every model")
    rows_ci, rows_cls, pair_rows = [], [], []
    boot = {}
    for sc in SCOPES:
        rs, nb = resamples(sc)
        say(f"  scope {sc}: {int(SCOPES[sc].sum())} test windows in {nb} blocks")
        for l in LAB:
            C = len(classes(l))
            y = D.yte[l]
            for n in reg[l]:
                P = TESTP[l][n]
                pt, rec, pre, f1 = point_metrics(y[SCOPES[sc]], P[SCOPES[sc]], C)
                M = {k: np.empty(N_BOOT) for k in ("balanced_accuracy", "macro_f1", "normal_recall", "roc_auc")}
                R = np.empty((N_BOOT, C))
                for b, idx in enumerate(rs):
                    m, rc, _, _ = point_metrics(y[idx], P[idx], C)
                    for k in M:
                        M[k][b] = m[k]
                    R[b] = rc
                boot[(sc, l, n)] = M
                for k, v in M.items():
                    rows_ci.append(dict(model=n, label_version=l, scope=sc, metric=k, point_estimate=pt[k], boot_mean=np.nanmean(v),
                                        ci95_low=np.nanpercentile(v, 2.5), ci95_high=np.nanpercentile(v, 97.5), n_resamples=N_BOOT,
                                        n_valid_resamples=int(np.isfinite(v).sum()), n_blocks=nb))
                for ci, cn in enumerate(classes(l)):
                    rows_cls.append(dict(model=n, label_version=l, scope=sc, **{"class": cn}, support=int((y[SCOPES[sc]] == ci).sum()),
                                         precision=pre[ci], recall=rec[ci], f1=f1[ci], recall_boot_mean=np.nanmean(R[:, ci]),
                                         recall_ci95_low=np.nanpercentile(R[:, ci], 2.5), recall_ci95_high=np.nanpercentile(R[:, ci], 97.5)))
    ci = pd.DataFrame(rows_ci)
    ci.to_csv(REP / "test_bootstrap_ci.csv", index=False)
    pd.DataFrame(rows_cls).to_csv(REP / "test_per_class.csv", index=False)
    say("saved test_bootstrap_ci.csv, test_per_class.csv (per-system scopes SN/TT and per-class recall/precision/F1 with bootstrap CI for recall)")
    # ---- paired bootstrap: best model (by test balanced accuracy, ALL scope) vs every other model
    best = {}
    for l in LAB:
        s = ci[(ci.label_version == l) & (ci.scope == "ALL") & (ci.metric == "balanced_accuracy")].set_index("model").point_estimate
        best[l] = s.idxmax()
        say(f"  best model {l}: {best[l]} (test balanced accuracy {s.max():.4f}; chosen on the test set itself, so the comparison favours it)")
    for sc in SCOPES:
        for l in LAB:
            b0 = best[l]
            for metric in ("balanced_accuracy", "macro_f1", "normal_recall"):
                prow = []
                for n in reg[l]:
                    if n == b0:
                        continue
                    d = boot[(sc, l, b0)][metric] - boot[(sc, l, n)][metric]
                    d = d[np.isfinite(d)]
                    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
                    prow.append(dict(label_version=l, scope=sc, metric=metric, best_model=b0, other_model=n, mean_diff=d.mean(),
                                     ci95_low=np.percentile(d, 2.5), ci95_high=np.percentile(d, 97.5), p_two_sided=min(1.0, max(p, 1.0 / N_BOOT)),
                                     best_better_in_share_of_resamples=(d > 0).mean()))
                # Holm correction within (label, scope, metric)
                order = np.argsort([r["p_two_sided"] for r in prow])
                m_ = len(prow)
                run = 0.0
                for rank, i in enumerate(order):
                    run = max(run, min(1.0, (m_ - rank) * prow[i]["p_two_sided"]))
                    prow[i]["p_holm"] = run
                    prow[i]["significant_at_0.05_holm"] = run < 0.05
                pair_rows += prow
    pd.DataFrame(pair_rows).to_csv(REP / "paired_bootstrap.csv", index=False)
    say("saved paired_bootstrap.csv")
    json.dump(best, open(V2 / "step3_best.json", "w"))


# ------------------------------------------------------------------------------------------------ 3: calibration
def logits(P):
    return np.log(np.clip(P / P.sum(1, keepdims=True), 1e-7, 1))


def softmax_T(L, T):
    z = L / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(P, y):
    L = logits(P)
    nll = lambda T: -np.log(np.clip(softmax_T(L, T)[np.arange(len(y)), y], 1e-12, 1)).mean()
    return float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)


def fit_isotonic(P, y):
    C = P.shape[1]
    if C == 2:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(P[:, 1], y)
        return [iso]
    return [IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(P[:, c], (y == c).astype(int)) for c in range(C)]


def apply_isotonic(isos, P):
    if len(isos) == 1:
        p1 = isos[0].predict(P[:, 1])
        return np.stack([1 - p1, p1], 1)
    Q = np.stack([isos[c].predict(P[:, c]) for c in range(P.shape[1])], 1)
    s = Q.sum(1, keepdims=True)
    return np.where(s > 0, Q / np.where(s > 0, s, 1), 1.0 / P.shape[1])


def brier(P, y):
    C = P.shape[1]
    if C == 2:
        return float(((P[:, 1] - y) ** 2).mean())
    return float(((P - np.eye(C)[y]) ** 2).sum(1).mean())


def ece(P, y, bins=15):
    conf_, pred = P.max(1), P.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf_ > a) & (conf_ <= b)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf_[m].mean())
    return float(e)


def reliability(P, y, bins=10):
    """confidence reliability: mean confidence vs accuracy per bin"""
    conf_, pred = P.max(1), P.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    xs, ys, ns = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf_ > a) & (conf_ <= b)
        if m.sum() >= 5:
            xs.append(conf_[m].mean()); ys.append((pred[m] == y[m]).mean()); ns.append(int(m.sum()))
    return xs, ys, ns


def part_calibration():
    say("=" * 60); say("CALIBRATION: temperature scaling and isotonic regression fitted on OUT-OF-FOLD predictions only, evaluated on the test set")
    rows, curves = [], {}
    for l in LAB:
        C = len(classes(l))
        for n in reg[l]:
            o = OOF[(OOF.label_version == l) & (OOF.model == n)]
            Po, yo = o[[f"p{c}" for c in range(C)]].to_numpy(), o.y_true.to_numpy(int)
            Pt, yt = TESTP[l][n], D.yte[l]
            Pt = Pt / Pt.sum(1, keepdims=True)
            T = fit_temperature(Po, yo)
            iso = fit_isotonic(Po, yo)
            variants = {"raw": Pt, "temperature": softmax_T(logits(Pt), T), "isotonic": apply_isotonic(iso, Pt)}
            for v, Q in variants.items():
                m, _, _, _ = point_metrics(yt, Q, C)
                ll = float(-np.log(np.clip(Q[np.arange(len(yt)), yt], 1e-12, 1)).mean())
                rows.append(dict(model=n, label_version=l, calibration=v, temperature=T if v == "temperature" else np.nan, brier=brier(Q, yt), ece_15bins=ece(Q, yt),
                                 log_loss=ll, test_balanced_accuracy=m["balanced_accuracy"], test_macro_f1=m["macro_f1"], test_normal_recall=m["normal_recall"],
                                 oof_brier_raw=brier(Po / Po.sum(1, keepdims=True), yo)))
            curves[(l, n)] = (variants, yt)
    cal = pd.DataFrame(rows)
    cal.to_csv(REP / "calibration.csv", index=False)
    best = json.loads((V2 / "step3_best.json").read_text()) if (V2 / "step3_best.json").exists() else {}
    for l in LAB:
        s = cal[(cal.label_version == l) & (cal.calibration == "raw")].set_index("model").test_balanced_accuracy
        top = list(s.sort_values(ascending=False).index[:3])
        fig, axs = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
        for ax, n in zip(axs, top):
            variants, yt = curves[(l, n)]
            for (v, Q), ls, col in zip(variants.items(), ["-", "--", ":"], ["#000000", "#555555", "#888888"]):
                xs, ys, ns = reliability(Q, yt)
                ax.plot(xs, ys, ls, color=col, marker="o", ms=3, label=f"{v} (Brier {brier(Q, yt):.3f})")
            ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=1)
            ax.set_title(f"{n}\n{l[6:]}", fontsize=10); ax.set_xlabel("mean confidence"); ax.legend(fontsize=7, loc="upper left")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        axs[0].set_ylabel("accuracy in bin")
        fig.suptitle("Reliability curves on the test set (calibrators fitted on out-of-fold predictions)", fontsize=10)
        fig.tight_layout(); fig.savefig(S3 / f"calibration_reliability_{l}.png", dpi=130); plt.close(fig)
    say("saved calibration.csv and calibration_reliability_*.png")


# ------------------------------------------------------------------------------------------------ 4: explainability (best tree-based single model per label)
def group_of(f):
    return "API" if f.startswith("api_") else "log" if f.startswith("log_") else "metric" if f.startswith("m_") else "system/indicator"


def short(f, k=38):
    f = f.replace("m_", "", 1) if f.startswith("m_") else f
    return f if len(f) <= k else f[:k - 1] + "~"


def shap_to_3d(sv, C):
    if isinstance(sv, list):
        sv = np.stack(sv, -1)
    sv = np.asarray(sv)
    if sv.ndim == 2:                       # binary XGBoost: contribution to the positive class
        sv = sv[:, :, None]
    return sv


def part_explain():
    import shap, joblib
    from lime.lime_tabular import LimeTabularExplainer
    say("=" * 60); say("EXPLAINABILITY: SHAP (TreeExplainer) and LIME on the best single (non-ensemble, non-deep) model per label")
    Wf = D.W("final")
    names = D.gprep.output_names
    Xtr, Xte = Wf[D.train_pos], Wf[te]
    rng = np.random.RandomState(SEED)
    glob_rows, local_rows, agree_rows, summ_rows = [], [], [], []
    for l in LAB:
        C = len(classes(l))
        tab = {n: s for n, s in reg[l].items() if s["kind"] == "tab"}
        t_bal = {n: point_metrics(D.yte[l], TESTP[l][n], C)[0]["balanced_accuracy"] for n in tab}
        bn = max(t_bal, key=t_bal.get)
        say(f"[{l}] explained model: {bn} (test balanced accuracy {t_bal[bn]:.4f}; best among the tree/linear baselines, chosen on the test set)")
        full = joblib.load(MODELS / f"{l}__{bn}.joblib")
        clf = full.named_steps["clf"] if hasattr(full, "named_steps") else full
        P = TESTP[l][bn]
        pred, y = P.argmax(1), D.yte[l]
        ex = shap.TreeExplainer(clf)
        sv = shap_to_3d(ex.shap_values(Xte), C)
        sv_class = (lambda c: sv[:, :, c if sv.shape[2] > 1 else 0])
        # ---------- global
        imp = np.abs(sv).mean(axis=(0, 2))
        order = np.argsort(-imp)
        g = pd.DataFrame({"feature": np.array(names)[order], "group": [group_of(names[i]) for i in order], "mean_abs_shap": imp[order], "rank": np.arange(1, len(order) + 1)})
        g.insert(0, "model", bn); g.insert(0, "label_version", l)
        g.to_csv(REP / f"shap_global_{l}.csv", index=False)
        grp = g.groupby("group").mean_abs_shap.sum()
        grp = (grp / grp.sum()).rename("share_of_total_mean_abs_shap").reset_index().assign(label_version=l, model=bn)
        grp.to_csv(REP / f"shap_group_share_{l}.csv", index=False)
        top = g.head(20)
        shade = {"API": "#d0d0d0", "log": "#909090", "metric": "#404040", "system/indicator": "#ffffff"}
        hatch = {"API": "//", "log": "..", "metric": "", "system/indicator": "xx"}
        fig, ax = plt.subplots(figsize=(8, 6.2))
        yy = np.arange(len(top))[::-1]
        for yi, (_, r) in zip(yy, top.iterrows()):
            ax.barh(yi, r.mean_abs_shap, color=shade[r.group], edgecolor="black", hatch=hatch[r.group], linewidth=0.8)
        ax.set_yticks(yy); ax.set_yticklabels([short(f) for f in top.feature], fontsize=8)
        ax.set_xlabel("mean |SHAP| over test windows" + (" and classes" if C > 2 else ""))
        ax.set_title(f"Global SHAP, top 20 features: {bn} ({l[6:]})", fontsize=10)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor=shade[k], edgecolor="black", hatch=hatch[k], label=k) for k in shade], fontsize=8, loc="lower right")
        fig.tight_layout(); fig.savefig(S3 / f"shap_global_bar_{l}.png", dpi=130); plt.close(fig)
        # beeswarm for one class: Anomaly (binary) / Normal (five-class, the class of interest)
        cb = 1 if C == 2 else 0
        fig = plt.figure(figsize=(8.5, 6.5))
        shap.summary_plot(sv_class(cb), Xte, feature_names=[f"[{group_of(f)}] {short(f, 30)}" for f in names], max_display=20, show=False, cmap=plt.cm.Greys, plot_size=None)
        plt.title(f"SHAP beeswarm, class {classes(l)[cb]}: {bn} ({l[6:]})", fontsize=10, family="serif")
        plt.tight_layout(); plt.savefig(S3 / f"shap_beeswarm_{l}.png", dpi=130); plt.close("all")
        glob_rows.append(g.assign(label_version=l))
        # ---------- 5 correct + 5 wrong windows
        ok, bad = np.where(pred == y)[0], np.where(pred != y)[0]
        pick_ok = rng.choice(ok, 5, replace=False)
        pick_bad = rng.choice(bad, min(5, len(bad)), replace=False)
        picks = [(i, "correct") for i in pick_ok] + [(i, "wrong") for i in pick_bad]
        lime_ex = LimeTabularExplainer(Xtr, feature_names=names, class_names=classes(l), mode="classification", discretize_continuous=True, random_state=SEED)
        predict_fn = full.predict_proba
        fig, axs = plt.subplots(2, 5, figsize=(20, 7.5))
        win_ids = df.window_id.to_numpy()[te]
        for k, (i, kind) in enumerate(picks):
            c = 1 if C == 2 else int(pred[i])
            s_c = sv_class(c)[i]
            s_top = np.argsort(-np.abs(s_c))[:10]
            le = lime_ex.explain_instance(Xte[i], predict_fn, num_features=20, num_samples=3000, labels=(c,))
            lm = dict(le.as_map()[c])
            l_top = sorted(lm, key=lambda j: -abs(lm[j]))[:10]
            for r, j in enumerate(s_top):
                local_rows.append(dict(label_version=l, model=bn, window_id=win_ids[i], outcome=kind, true=classes(l)[y[i]], predicted=classes(l)[pred[i]], method="SHAP",
                                       explained_class=classes(l)[c], rank=r + 1, feature=names[j], group=group_of(names[j]), contribution=s_c[j], feature_value=Xte[i, j]))
            for r, j in enumerate(l_top):
                local_rows.append(dict(label_version=l, model=bn, window_id=win_ids[i], outcome=kind, true=classes(l)[y[i]], predicted=classes(l)[pred[i]], method="LIME",
                                       explained_class=classes(l)[c], rank=r + 1, feature=names[j], group=group_of(names[j]), contribution=lm[j], feature_value=Xte[i, j]))
            s5, l5 = set(s_top[:5]), set(l_top[:5])
            agree_rows.append(dict(label_version=l, model=bn, window_id=win_ids[i], outcome=kind, true=classes(l)[y[i]], predicted=classes(l)[pred[i]],
                                   shap_top5="; ".join(names[j] for j in s_top[:5]), lime_top5="; ".join(names[j] for j in l_top[:5]),
                                   n_shared_of_5=len(s5 & l5), jaccard=len(s5 & l5) / len(s5 | l5), set="5 correct + 5 wrong"))
            ax = axs[k // 5, k % 5]
            vals = s_c[s_top][::-1]
            ax.barh(range(10), vals, color=["#333333" if v > 0 else "#bbbbbb" for v in vals], edgecolor="black", linewidth=0.6)
            ax.set_yticks(range(10)); ax.set_yticklabels([short(names[j], 26) for j in s_top[::-1]], fontsize=6.5)
            ax.set_title(f"{kind}: true {classes(l)[y[i]]} / pred {classes(l)[pred[i]]}\n{win_ids[i][-34:]}", fontsize=7.5)
            ax.axvline(0, color="black", lw=0.6)
        fig.suptitle(f"Local SHAP (dark = pushes towards {'Anomaly' if C == 2 else 'the predicted class'}, light = against), top 10 features: {bn} ({l[6:]})", fontsize=10)
        fig.tight_layout(); fig.savefig(S3 / f"shap_local_{l}.png", dpi=120); plt.close(fig)
        # ---------- larger agreement sample: 60 random test windows
        for i in rng.choice(len(te), 60, replace=False):
            c = 1 if C == 2 else int(pred[i])
            s_top = np.argsort(-np.abs(sv_class(c)[i]))[:5]
            le = lime_ex.explain_instance(Xte[i], predict_fn, num_features=20, num_samples=3000, labels=(c,))
            lm = dict(le.as_map()[c])
            l_top = sorted(lm, key=lambda j: -abs(lm[j]))[:5]
            agree_rows.append(dict(label_version=l, model=bn, window_id=win_ids[i], outcome="correct" if pred[i] == y[i] else "wrong", true=classes(l)[y[i]],
                                   predicted=classes(l)[pred[i]], shap_top5="; ".join(names[j] for j in s_top), lime_top5="; ".join(names[j] for j in l_top),
                                   n_shared_of_5=len(set(s_top) & set(l_top)), jaccard=len(set(s_top) & set(l_top)) / len(set(s_top) | set(l_top)), set="60 random"))
        say(f"  [{l}] SHAP/LIME done for {bn}")
    pd.DataFrame(local_rows).to_csv(REP / "shap_lime_local_windows.csv", index=False)
    ag = pd.DataFrame(agree_rows)
    ag.to_csv(REP / "shap_lime_agreement_windows.csv", index=False)
    # summary of agreement
    for (l, s), g in ag.groupby(["label_version", "set"]):
        for oc, gg in [("all", g)] + list(g.groupby("outcome")):
            summ_rows.append(dict(label_version=l, model=g.model.iloc[0], window_set=s, outcome=oc, n_windows=len(gg), mean_shared_top5=gg.n_shared_of_5.mean(),
                                  share_with_ge3_shared=(gg.n_shared_of_5 >= 3).mean(), share_with_ge1_shared=(gg.n_shared_of_5 >= 1).mean(), mean_jaccard=gg.jaccard.mean(),
                                  share_top1_shap_in_lime_top5=np.nan))
    pd.DataFrame(summ_rows).to_csv(REP / "shap_lime_agreement_summary.csv", index=False)
    say("saved shap_global_*.csv, shap_group_share_*.csv, shap_lime_local_windows.csv, shap_lime_agreement_windows.csv, shap_lime_agreement_summary.csv and PNGs")


# ------------------------------------------------------------------------------------------------ 5: error analysis of Normal windows
def part_errors():
    import joblib, shap
    say("=" * 60); say("ERROR ANALYSIS: Normal test windows that the best model misclassifies")
    Wf = D.W("final")
    names = D.gprep.output_names
    Xtr, Xte = Wf[D.train_pos], Wf[te]
    run_tr = df.run_id.to_numpy()[D.train_pos]
    run_te = df.run_id.to_numpy()[te]
    win_te = df.window_id.to_numpy()[te]
    wi_te = df.window_index.to_numpy()[te]
    split_all = df.split.to_numpy()
    rows, summ, shift_rows, shap_rows = [], [], [], []
    for l in LAB:
        C = len(classes(l))
        tab = {n: s for n, s in reg[l].items() if s["kind"] == "tab"}
        bn = max(tab, key=lambda n: point_metrics(D.yte[l], TESTP[l][n], C)[0]["balanced_accuracy"])
        P = TESTP[l][bn]
        pred, y = P.argmax(1), D.yte[l]
        normal = np.where(y == 0)[0]
        wrong = normal[pred[normal] != 0]
        right = normal[pred[normal] == 0]
        say(f"[{l}] {bn}: {len(normal)} Normal test windows, {len(wrong)} misclassified ({len(wrong) / len(normal):.1%}), {len(right)} correct")
        # nearest training scenario (k=5 majority, same system) in the preprocessed feature space
        nn_run = {}
        for i in wrong:
            m = np.where(D.sys_tr == D.sys_te[i])[0]
            d = np.linalg.norm(Xtr[m] - Xte[i], axis=1)
            o = np.argsort(d)[:5]
            votes = pd.Series(run_tr[m][o]).value_counts()
            nn_run[i] = (votes.index[0], d[o][0], run_tr[m][o][0])
        # train-Normal reference per system for z-scores
        ref = {}
        for s in (0, 1):
            mm = (D.sys_tr == s) & (D.ytr["label_binary"] == 0)
            ref[s] = (Xtr[mm].mean(0), Xtr[mm].std(0) + 1e-6)
        clf = joblib.load(MODELS / f"{l}__{bn}.joblib")
        clf = clf.named_steps["clf"] if hasattr(clf, "named_steps") else clf
        sv = shap_to_3d(shap.TreeExplainer(clf).shap_values(Xte[wrong]), C) if len(wrong) else None
        for k, i in enumerate(wrong):
            mu, sd = ref[D.sys_te[i]]
            z = (Xte[i] - mu) / sd
            top = np.argsort(-np.abs(z))[:5]
            c_pred = pred[i]
            sc = sv[k][:, c_pred if sv.shape[2] > 1 else 0]
            tops = np.argsort(-np.abs(sc))[:3]
            rows.append(dict(label_version=l, model=bn, window_id=win_te[i], system="TT" if D.sys_te[i] else "SN", run=run_te[i], window_index=int(wi_te[i]),
                             predicted=classes(l)[c_pred], prob_predicted=P[i, c_pred] / P[i].sum(), prob_normal=P[i, 0] / P[i].sum(),
                             nearest_train_scenario_k5=nn_run[i][0], nearest_train_scenario_k1=nn_run[i][2], nearest_distance=nn_run[i][1],
                             top5_deviating_features_vs_train_normal="; ".join(f"{names[j]} (z={z[j]:+.1f})" for j in top),
                             top3_shap_drivers_of_prediction="; ".join(f"{names[j]} ({sc[j]:+.2f})" for j in tops)))
            for j in tops:
                shap_rows.append(dict(label_version=l, model=bn, feature=names[j], group=group_of(names[j]), shap=sc[j]))
        # feature shift: mistaken vs correct Normal windows, in train-Normal z units
        if len(wrong) and len(right):
            Z = lambda idxs: np.stack([(Xte[i] - ref[D.sys_te[i]][0]) / ref[D.sys_te[i]][1] for i in idxs])
            Zw, Zr = Z(wrong), Z(right)
            dz = Zw.mean(0) - Zr.mean(0)
            for j in np.argsort(-np.abs(dz))[:25]:
                shift_rows.append(dict(label_version=l, model=bn, feature=names[j], group=group_of(names[j]), mean_z_mistaken=Zw[:, j].mean(), mean_z_correct=Zr[:, j].mean(),
                                       difference=dz[j], mean_raw_scaled_mistaken=Xte[wrong][:, j].mean(), mean_raw_scaled_correct=Xte[right][:, j].mean()))
        # summary tables
        w = pd.DataFrame([r for r in rows if r["label_version"] == l])
        def add(kind, series):
            for key, cnt in series.value_counts().items():
                summ.append(dict(label_version=l, model=bn, breakdown=kind, value=key, n_misclassified_normal=int(cnt), share_of_misclassified=cnt / len(wrong)))
        if len(w):
            add("predicted_class", w.predicted); add("system", w.system); add("run", w.run); add("nearest_train_scenario_k5", w.nearest_train_scenario_k5)
            pos = []
            for i in wrong:   # is the window next to a train block (20 s overlap / boundary)?
                p = np.where(win_te == win_te[i])[0][0]
                gpos = np.where(df.window_id.to_numpy() == win_te[i])[0][0]
                nb = [q for q in (gpos - 1, gpos + 1) if 0 <= q < len(df) and df.run_id.iloc[q] == df.run_id.iloc[gpos] and split_all[q] == "train"]
                pos.append("adjacent to a train window (20 s overlap)" if nb else "inside a test block")
            add("position", pd.Series(pos))
        summ.append(dict(label_version=l, model=bn, breakdown="counts", value="normal_test_windows", n_misclassified_normal=len(normal), share_of_misclassified=np.nan))
        summ.append(dict(label_version=l, model=bn, breakdown="counts", value="misclassified", n_misclassified_normal=len(wrong), share_of_misclassified=np.nan))
    pd.DataFrame(rows).to_csv(REP / "error_normal_misclassified.csv", index=False)
    pd.DataFrame(summ).to_csv(REP / "error_normal_summary.csv", index=False)
    pd.DataFrame(shift_rows).to_csv(REP / "error_normal_feature_shift.csv", index=False)
    pd.DataFrame(shap_rows).groupby(["label_version", "model", "feature", "group"]).shap.agg(["count", "mean"]).reset_index().sort_values(["label_version", "count"], ascending=[True, False]).to_csv(
        REP / "error_normal_shap_drivers.csv", index=False)
    # figure: predicted class of Normal windows per label
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, l in zip(axs, LAB):
        s = pd.DataFrame(summ)
        s = s[(s.label_version == l) & (s.breakdown == "predicted_class")]
        ax.bar(s.value, s.n_misclassified_normal, color="#777777", edgecolor="black")
        ax.set_title(f"Normal windows misclassified, predicted as ({l[6:]})", fontsize=9); ax.set_ylabel("windows")
        for t in ax.get_xticklabels():
            t.set_rotation(20)
    fig.tight_layout(); fig.savefig(S3 / "error_normal_predicted_as.png", dpi=130); plt.close(fig)
    say("saved error_normal_*.csv")


# ------------------------------------------------------------------------------------------------ summary markdown (all numbers read from the CSVs)
def part_summary():
    say("=" * 60); say("SUMMARY: results_summary.md (numbers read from the CSVs written above)")
    ci = pd.read_csv(REP / "test_bootstrap_ci.csv")
    pb = pd.read_csv(REP / "paired_bootstrap.csv")
    cal = pd.read_csv(REP / "calibration.csv")
    pc = pd.read_csv(REP / "test_per_class.csv")
    es = pd.read_csv(REP / "error_normal_summary.csv")
    em = pd.read_csv(REP / "error_normal_misclassified.csv")
    ag = pd.read_csv(REP / "shap_lime_agreement_summary.csv")
    f = lambda x: f"{x:.3f}"
    out = ["# Step 3 results summary", "",
           "All numbers below are read from the CSVs in `report_outputs/` by `scripts/step3_eval.py`. The test set (647 windows) was not used for training or tuning; "
           "the best model was nevertheless *chosen* on it, so differences in its favour are optimistic. Blocks (10 consecutive windows of one run) were resampled with replacement within each run (stratified by run, so every class is present in every resample), 1000 times, seed 42. Runs themselves are not resampled, so the intervals do not cover run-to-run variation.", ""]
    for l in LAB:
        c = ci[(ci.label_version == l) & (ci.scope == "ALL")]
        ba = c[c.metric == "balanced_accuracy"].set_index("model")
        b = ba.point_estimate.idxmax()
        out += [f"## {l[6:].capitalize()} labels", ""]
        r = lambda m, mod=b, sc="ALL": ci[(ci.label_version == l) & (ci.scope == sc) & (ci.metric == m) & (ci.model == mod)].iloc[0]
        out.append(f"- Best model on test: **{b}**. Balanced accuracy {f(r('balanced_accuracy').point_estimate)} (95% CI {f(r('balanced_accuracy').ci95_low)} to {f(r('balanced_accuracy').ci95_high)}), "
                   f"macro-F1 {f(r('macro_f1').point_estimate)} ({f(r('macro_f1').ci95_low)} to {f(r('macro_f1').ci95_high)}), Normal recall {f(r('normal_recall').point_estimate)} "
                   f"({f(r('normal_recall').ci95_low)} to {f(r('normal_recall').ci95_high)}), ROC-AUC {f(r('roc_auc').point_estimate)} ({f(r('roc_auc').ci95_low)} to {f(r('roc_auc').ci95_high)}).")
        p = pb[(pb.label_version == l) & (pb.scope == "ALL") & (pb.metric == "balanced_accuracy")]
        out.append(f"- Paired bootstrap on balanced accuracy, {b} vs the other {len(p)} models: the best model is significantly better after Holm correction (p < 0.05) than "
                   f"{int(p['significant_at_0.05_holm'].sum())} of them; not distinguishable from {', '.join(p[~p['significant_at_0.05_holm']].other_model) or 'none'}.")
        for sc in ("SN", "TT"):
            out.append(f"- {sc} only, {b}: balanced accuracy {f(r('balanced_accuracy', sc=sc).point_estimate)} ({f(r('balanced_accuracy', sc=sc).ci95_low)} to {f(r('balanced_accuracy', sc=sc).ci95_high)}), "
                       f"Normal recall {f(r('normal_recall', sc=sc).point_estimate)} ({f(r('normal_recall', sc=sc).ci95_low)} to {f(r('normal_recall', sc=sc).ci95_high)}).")
        q = pc[(pc.label_version == l) & (pc.scope == "ALL") & (pc.model == b)]
        out.append("- Per-class recall of the best model (support): " + ", ".join(f"{x['class']} {f(x.recall)} ({int(x.support)})" for _, x in q.iterrows()) + ".")
        out.append("")
        cc = cal[(cal.label_version == l) & (cal.model == b)].set_index("calibration")
        out.append(f"- Calibration of {b} on test (fitted on out-of-fold predictions): Brier raw {f(cc.loc['raw','brier'])}, temperature {f(cc.loc['temperature','brier'])} (T = {f(cc.loc['temperature','temperature'])}), "
                   f"isotonic {f(cc.loc['isotonic','brier'])}; ECE raw {f(cc.loc['raw','ece_15bins'])}, temperature {f(cc.loc['temperature','ece_15bins'])}, isotonic {f(cc.loc['isotonic','ece_15bins'])}.")
        allc = cal[cal.label_version == l].pivot(index="model", columns="calibration", values="brier")
        out.append(f"- Across all {len(allc)} models, temperature scaling lowered the test Brier score for {int((allc.temperature < allc.raw).sum())} and isotonic for {int((allc.isotonic < allc.raw).sum())}.")
        gs = pd.read_csv(REP / f"shap_global_{l}.csv")
        gsh = pd.read_csv(REP / f"shap_group_share_{l}.csv")
        out.append(f"- SHAP ({gs.model.iloc[0]}): top-5 features " + ", ".join(f"{x.feature} ({x.group}, {x.mean_abs_shap:.3f})" for _, x in gs.head(5).iterrows()) +
                   "; share of total mean |SHAP| by group: " + ", ".join(f"{x.group} {x.share_of_total_mean_abs_shap:.1%}" for _, x in gsh.iterrows()) + ".")
        a = ag[(ag.label_version == l) & (ag.outcome == "all")]
        out.append("- SHAP vs LIME top-5 agreement: " + "; ".join(f"{x.window_set}: mean {x.mean_shared_top5:.2f} of 5 shared features, >=3 shared in {x.share_with_ge3_shared:.0%} of {int(x.n_windows)} windows" for _, x in a.iterrows()) + ".")
        e = es[(es.label_version == l)]
        cnt = e[e.breakdown == "counts"].set_index("value").n_misclassified_normal
        out.append(f"- Error analysis ({e.model.iloc[0]}): {int(cnt['misclassified'])} of {int(cnt['normal_test_windows'])} Normal test windows are misclassified.")
        for kind, label in (("predicted_class", "predicted as"), ("nearest_train_scenario_k5", "feature-space nearest training scenario"), ("system", "system"), ("position", "position")):
            ee = e[e.breakdown == kind].sort_values("n_misclassified_normal", ascending=False)
            if len(ee):
                out.append(f"  - {label}: " + ", ".join(f"{x.value} {int(x.n_misclassified_normal)}" for _, x in ee.head(5).iterrows()))
        out.append("")
    out += ["## Caveats (interpretation, not numbers)", "",
            "- Each class comes from a small number of scenario runs, and most top SHAP features are host-level gauges (disk usage, filesystem space, process memory). "
            "They can identify *which run* a window belongs to rather than *what failed*, so high importance is not evidence of a causal failure indicator.",
            "- The test set holds 36 Normal windows (SN 16, TT 20) from two runs, so Normal recall and its confidence intervals rest on very few blocks.",
            "- The best model per label was identified on the test set; the paired-bootstrap p-values therefore favour it. SHAP/LIME explain the best single tree-based model, not the ensemble.", "",
            "## Files", "", "test_bootstrap_ci.csv, test_per_class.csv, paired_bootstrap.csv, calibration.csv, shap_global_*.csv, shap_group_share_*.csv, shap_lime_local_windows.csv, "
            "shap_lime_agreement_windows.csv, shap_lime_agreement_summary.csv, error_normal_*.csv; figures in `report_outputs/figures/step3/`.", ""]
    (REP / "results_summary.md").write_text("\n".join(out), encoding="utf-8")
    say("saved results_summary.md")


if __name__ == "__main__":
    say("=" * 60); say("step3_eval start", time.ctime(), "parts", PARTS)
    for p in PARTS:
        t0 = time.time()
        {"bootstrap": part_bootstrap, "calibration": part_calibration, "explain": part_explain, "errors": part_errors, "summary": part_summary}[p]()
        say(f"part {p} finished in {(time.time() - t0) / 60:.1f} min")
