"""Model registry: loads the saved FINAL models once at startup (models/ + processed/preprocessor.joblib) and predicts probabilities.
Nothing is trained or fitted here."""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
import joblib

from . import config

sys.path.insert(0, str(config.SCRIPTS))
import preprocess as P            # the single source of preprocessing (training, testing and the app)


class RegistryError(RuntimeError):
    """Raised at startup with a message that says exactly what is missing and where it was looked for."""


class Registry:
    def __init__(self, manifest_path=None, models_dir=None, preprocessor_path=None, load_deep=None):
        self.manifest_path = Path(manifest_path or config.ASSETS / "model_manifest.json")
        self.models_dir = Path(models_dir or config.MODELS_DIR)
        self.preprocessor_path = Path(preprocessor_path or config.PREPROCESSOR)
        self.load_deep = config.LOAD_DEEP if load_deep is None else load_deep
        self.entries = {}
        self.objects = {}
        self.prep = None
        self.tf_available = importlib.util.find_spec("tensorflow") is not None
        self._ds = None

    # ------------------------------------------------------------------ startup
    def load(self):
        if not self.manifest_path.exists():
            raise RegistryError(f"Model manifest not found: {self.manifest_path}. Run `python app/backend/build_assets.py` once (needs the training state).")
        if not self.preprocessor_path.exists():
            raise RegistryError(f"Preprocessor not found: {self.preprocessor_path}. Create it with `python scripts/preprocess.py` (it is not stored in git).")
        manifest = json.loads(self.manifest_path.read_text())
        missing = []
        for e in manifest:
            if not e["servable"]:
                continue
            for fname in e["files"].values():
                if not (self.models_dir / fname).exists():
                    missing.append(fname)
        if missing:
            raise RegistryError(f"{len(set(missing))} model file(s) listed in {self.manifest_path.name} are missing from {self.models_dir}: "
                                f"{', '.join(sorted(set(missing)))}. The model files are not stored in git; copy the `models/` folder produced by the training run "
                                f"or set FAILSIGHT_MODELS_DIR.")
        self.prep = P.load_preprocessor(self.preprocessor_path)
        for e in manifest:
            self.entries[(e["label_version"], e["name"])] = e
        if self._deep_ok():
            from . import deep_support
            self._ds = deep_support
        for key, e in self.entries.items():
            self._load_entry(key, e)
        return self

    def _deep_ok(self):
        return self.load_deep and self.tf_available

    def _load_entry(self, key, e):
        label, name = key
        e["available"], e["unavailable_reason"] = False, e["reason"]
        if not e["servable"]:
            return
        if e["kind"] in ("deep", "head") and not self._deep_ok():
            e["unavailable_reason"] = "needs TensorFlow (not installed or disabled with FAILSIGHT_LOAD_DEEP=0)"
            return
        if e["kind"] == "ens" and any(not self._member_ok(label, m) for m in e["members"]):
            e["unavailable_reason"] = "a member model is unavailable (stacking needs the deep models and TensorFlow)"
            return
        try:
            f = e["files"]
            if e["kind"] == "tab":
                self.objects[key] = joblib.load(self.models_dir / f["model"])
            elif e["kind"] == "deep":
                self.objects[key] = self._ds.load_keras(self.models_dir / f["model"])
            elif e["kind"] == "head":
                base = self._ds.load_keras(self.models_dir / f["base_model"])
                self.objects[key] = (self._ds.embedding_model(base), joblib.load(self.models_dir / f["model"]))
            elif e["kind"] == "ens":
                self.objects[key] = joblib.load(self.models_dir / f["model"])
            e["available"], e["unavailable_reason"] = True, ""
        except Exception as ex:                                     # a corrupt file must be reported, not hidden
            raise RegistryError(f"Could not load model file for {name} ({label}): {type(ex).__name__}: {ex}") from ex

    def _member_ok(self, label, m):
        e = self.entries.get((label, m))
        return bool(e and e["servable"] and (e["kind"] == "tab" or self._deep_ok()))

    # ------------------------------------------------------------------ access
    def get(self, label, name):
        e = self.entries.get((label, name))
        if e is None:
            raise KeyError(f"unknown model '{name}' for {label}")
        return e

    # ------------------------------------------------------------------ prediction (probabilities of the model's own output layer)
    def predict_proba(self, label, name, X, seq, _cache=None):
        """X: (n, 128) preprocessed windows; seq: (n, 6, 128) last-6-window sequences. Returns (n, C) probabilities."""
        cache = _cache if _cache is not None else {}
        if (label, name) in cache:
            return cache[(label, name)]
        e = self.get(label, name)
        if not e["available"]:
            raise RegistryError(f"{name} is not available: {e['unavailable_reason']}")
        obj = self.objects[(label, name)]
        if e["kind"] == "tab":
            p = obj.predict_proba(X)
        elif e["kind"] == "deep":
            p = obj.predict(seq.astype(np.float32), batch_size=256, verbose=0)
        elif e["kind"] == "head":
            emb, head = obj
            p = head.predict_proba(emb.predict(seq.astype(np.float32), batch_size=256, verbose=0))
        else:
            Z = np.hstack([self.predict_proba(label, m, X, seq, cache) for m in e["members"]])
            p = obj.predict_proba(Z)
        p = np.asarray(p, dtype=float)
        p = p / p.sum(1, keepdims=True)
        cache[(label, name)] = p
        return p

    def tree_estimator(self, label, name):
        """the fitted tree/linear estimator (unwrapping an imblearn Pipeline) for SHAP; None for deep / ensemble models"""
        e = self.get(label, name)
        if e["kind"] != "tab" or not e["available"]:
            return None
        obj = self.objects[(label, name)]
        return obj.named_steps["clf"] if hasattr(obj, "named_steps") else obj

    def full_tab_model(self, label, name):
        return self.objects[(label, name)]
