"""SHAP (TreeExplainer) and LIME for the tree models, grouped by api / log / metric / system. No model is refitted; LIME only needs a small background
sample of preprocessed training windows (assets/lime_background.npy, generated locally and not stored in git)."""
from functools import lru_cache
import numpy as np

from . import config


def group_of(feature: str) -> str:
    return "api" if feature.startswith("api_") else "log" if feature.startswith("log_") else "metric" if feature.startswith("m_") else "system"


class Explainer:
    def __init__(self, registry):
        self.reg = registry
        self._shap = {}
        self._lime = {}
        p = config.ASSETS / "lime_background.npy"
        self.background = np.load(p) if p.exists() else None

    def supported(self, label, name):
        return self.reg.tree_estimator(label, name) is not None and self.reg.get(label, name)["family"] == "tree"

    def _tree_explainer(self, label, name):
        import shap
        key = (label, name)
        if key not in self._shap:
            self._shap[key] = shap.TreeExplainer(self.reg.tree_estimator(label, name))
        return self._shap[key]

    def shap_values(self, label, name, x_pre, class_idx, top=15):
        names = self.reg.prep.output_names
        sv = self._tree_explainer(label, name).shap_values(x_pre)
        if isinstance(sv, list):
            sv = np.stack(sv, -1)
        sv = np.asarray(sv)
        if sv.ndim == 2:                         # binary boosters return the contribution to the positive class only
            sv = sv[:, :, None]
        c = class_idx if sv.shape[2] > 1 else 0
        s = sv[0, :, c]
        ev = self._tree_explainer(label, name).expected_value
        base = float(np.atleast_1d(ev)[c if np.size(ev) > 1 else 0])
        order = np.argsort(-np.abs(s))
        total = float(np.abs(s).sum()) or 1.0
        groups = {}
        for i, n in enumerate(names):
            g = group_of(n)
            groups.setdefault(g, 0.0)
            groups[g] += abs(float(s[i]))
        return dict(
            method="SHAP TreeExplainer", base_value=base, explained_class_index=int(c),
            top=[dict(feature=names[i], group=group_of(names[i]), shap=float(s[i]), preprocessed_value=float(x_pre[0, i])) for i in order[:top]],
            groups=[dict(group=g, total_abs_shap=v, share=v / total) for g, v in sorted(groups.items(), key=lambda kv: -kv[1])],
            n_features=len(names))

    def lime_values(self, label, name, x_pre, class_idx, top=15, num_samples=2000):
        if self.background is None:
            return dict(available=False, reason="LIME needs a background sample of preprocessed training windows (app/backend/assets/lime_background.npy). "
                                                "It is generated locally by build_assets.py and is not stored in git, so LIME is unavailable here.")
        from lime.lime_tabular import LimeTabularExplainer
        names = self.reg.prep.output_names
        if label not in self._lime:
            self._lime[label] = LimeTabularExplainer(self.background, feature_names=names, class_names=config.CLASS_NAMES[label], mode="classification",
                                                     discretize_continuous=True, random_state=42)
        model = self.reg.full_tab_model(label, name)
        exp = self._lime[label].explain_instance(x_pre[0], model.predict_proba, num_features=top, num_samples=num_samples, labels=(class_idx,))
        m = dict(exp.as_map()[class_idx])
        order = sorted(m, key=lambda j: -abs(m[j]))
        return dict(available=True, method="LIME (tabular, 2000 samples, background = 400 preprocessed training windows)", explained_class_index=int(class_idx),
                    top=[dict(feature=names[j], group=group_of(names[j]), weight=float(m[j]), preprocessed_value=float(x_pre[0, j])) for j in order[:top]])
