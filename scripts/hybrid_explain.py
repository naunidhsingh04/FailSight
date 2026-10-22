"""Explanations of the best hybrid on each data set (and of the best tree model on LO2). The 'best' model is chosen by its CROSS-VALIDATION validation score (never by test scores).

hybrid: gradient x input attribution (how much each input feature moved the model's probability of its predicted class), averaged over real test windows and the windows of the
sequence, shown per feature and per feature group; BERT + GNN also reports its attention over the log templates. This explains the model, not the real cause of a fault.
tree (LO2): SHAP (TreeExplainer) on the real test windows and LIME for one window.

python scripts/hybrid_explain.py anomod | lo2
-> report_outputs/hybrid_<ds>_explain_<labels>.csv, hybrid_<ds>_explain_groups.csv, lo2_full_shap_global.csv, lo2_full_lime_example.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import hybrid_engine as E  # noqa: E402

REP = E.REP


def group_of(name):
    if name.startswith(("api_",)):
        return "api"
    if name.startswith("m_"):
        return "metric"
    if name.startswith("tmpl_"):
        return "log template"
    if name.startswith("log_"):
        return "log"
    if name.startswith("was_missing") or name == "system_is_TT":
        return "flag"
    return "other"


def explain_hybrid(D, registry, prefix, mdir, names, n_windows=200):
    import tensorflow as tf
    lb = pd.read_csv(REP / f"{prefix}leaderboard.csv")
    W = D.W("final")
    X = D.seq(W, D.test_pos[:n_windows]).astype("float32")
    feat_names = getattr(D, "out_names", None) or list(D.gprep.output_names)
    group_rows = []
    for label in E.LABELS:
        d = lb[(lb.label_version == label) & (lb.status != "not completed")]
        if d.empty:
            continue
        best = d.sort_values("cv_val_bal_acc", ascending=False).iloc[0].model            # chosen on cross-validation, not on the test scores
        frozen = json.load(open(REP / f"{prefix}frozen_{label}__{best}.json"))
        m = registry[best](D.spec(label), frozen["config"])
        m(np.zeros((1,) + X.shape[1:], "float32"))
        m.load_weights(Path(mdir) / f"{label}__{best}.weights.h5")
        xt = tf.convert_to_tensor(X)
        with tf.GradientTape() as tape:
            tape.watch(xt)
            p = m(xt, training=False)
            cls = tf.argmax(p, axis=1)
            sel = tf.reduce_sum(p * tf.one_hot(cls, p.shape[1]), axis=1)
        g = tape.gradient(sel, xt).numpy()
        attr = np.abs(g * X).mean(axis=(0, 1))
        df = pd.DataFrame(dict(feature=feat_names, group=[group_of(n) for n in feat_names], mean_abs_gradient_x_input=attr)).sort_values("mean_abs_gradient_x_input", ascending=False)
        df["rank"] = np.arange(1, len(df) + 1)
        df.insert(0, "model", best)
        df.insert(1, "label_version", label)
        df.to_csv(REP / f"{prefix}explain_{label}.csv", index=False, float_format="%.6g")
        share = df.groupby("group").mean_abs_gradient_x_input.sum()
        for gname, v in (share / share.sum()).items():
            group_rows.append(dict(model=best, label_version=label, group=gname, share_of_total=float(v)))
        if hasattr(m, "second") and hasattr(m.second, "last_attention") and hasattr(D, "tm_top"):
            att = np.asarray(m.second.last_attention).reshape(-1, np.asarray(m.second.last_attention).shape[1]).mean(0)
            tm = D.tm_top.iloc[D.tmpl_order][:len(att)]
            pd.DataFrame(dict(template_source=tm.source.to_numpy(), template=tm.template.to_numpy(), mean_attention=att[:len(tm)])).sort_values("mean_attention", ascending=False).head(15).to_csv(
                REP / f"{prefix}explain_attention_{label}.csv", index=False, float_format="%.6g")
        print(f"explained {best} ({label}); top feature {df.iloc[0].feature}")
    pd.DataFrame(group_rows).to_csv(REP / f"{prefix}explain_groups.csv", index=False, float_format="%.6g")


def explain_lo2_trees(D):
    """SHAP + LIME for the best classical tree model on LO2 (binary labels), chosen by cross-validation validation balanced accuracy"""
    import shap
    import lo2_full as F
    best = None
    for n in ("RF", "ET", "XGB", "LGBM", "RF_tuned", "XGB_tuned", "LGBM_tuned"):
        p = F.STATE / f"result__label_binary__{n}.json"
        if p.exists():
            r = json.load(open(p))
            v = r["cv_before"]["val_ba"] if r["chosen_tag"] == "cv" else r["cv_after"]["val_ba"]
            if best is None or v > best[0]:
                best = (v, n, r)
    if best is None:
        return
    _, name, r = best
    label = "label_binary"
    W = D.W("final")
    ytr = D.y[label][D.train_pos]
    m, *_ = F.fit_tab(name, r["chosen_cfg"]["params"], r["chosen_method"], W[D.train_pos], ytr, D.binary_cols)
    Xte = W[D.test_pos]
    sv = shap.TreeExplainer(m).shap_values(Xte)
    sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)
    names = D.out_names
    df = pd.DataFrame(dict(model=name, feature=names, group=[group_of(n) for n in names], mean_abs_shap=np.abs(sv).mean(0))).sort_values("mean_abs_shap", ascending=False)
    df["rank"] = np.arange(1, len(df) + 1)
    df.to_csv(REP / "lo2_full_shap_global.csv", index=False, float_format="%.6g")
    from lime.lime_tabular import LimeTabularExplainer
    Xs, ys, *_ = E.smote_safe(W[D.train_pos], ytr)
    lt = LimeTabularExplainer(np.asarray(Xs), feature_names=names, class_names=D.class_names[label], discretize_continuous=True, random_state=E.SEED)
    i = 0
    ex = lt.explain_instance(Xte[i], m.predict_proba, num_features=10, num_samples=1000)
    pd.DataFrame(ex.as_list(), columns=["feature_rule", "weight"]).assign(model=name, window_id=D.df.window_id.iloc[D.test_pos[i]], true_class=D.class_names[label][int(D.y[label][D.test_pos[i]])]).to_csv(
        REP / "lo2_full_lime_example.csv", index=False)
    print(f"LO2 SHAP / LIME for {name}; top feature {df.iloc[0].feature}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "anomod"
    if which == "anomod":
        import hybrid_anomod as H
        D = H.AnoModData()
        explain_hybrid(D, H.ANOMOD_HYBRIDS, H.PREFIX, H.MODELS, H.NAMES)
    else:
        import hybrid_lo2 as H
        from lo2_data import LO2Data
        D = LO2Data()
        explain_hybrid(D, H.LO2_HYBRIDS, H.PREFIX, H.MODELS, H.NAMES)
        explain_lo2_trees(D)
