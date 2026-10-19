"""Paths and constants of the FailSight backend. Everything can be overridden with environment variables (used by Docker and the tests)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = Path(__file__).resolve().parent
ASSETS = Path(os.environ.get("FAILSIGHT_ASSETS", BACKEND / "assets"))
MODELS_DIR = Path(os.environ.get("FAILSIGHT_MODELS_DIR", ROOT / "models"))
PREPROCESSOR = Path(os.environ.get("FAILSIGHT_PREPROCESSOR", ROOT / "processed" / "preprocessor.joblib"))
REPORTS = Path(os.environ.get("FAILSIGHT_REPORTS", ROOT / "report_outputs"))
RESULTS_V1 = Path(os.environ.get("FAILSIGHT_RESULTS_V1", ROOT / "results" / "v1_16epoch"))
SCRIPTS = ROOT / "scripts"
FRONTEND_DIST = Path(os.environ.get("FAILSIGHT_FRONTEND_DIST", ROOT / "app" / "frontend" / "dist"))
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_ROWS = 2000
LOAD_DEEP = os.environ.get("FAILSIGHT_LOAD_DEEP", "1") != "0"      # set to 0 to serve only tree / linear models (no TensorFlow needed)
CLASS_NAMES = {"label_binary": ["Normal", "Anomaly"], "label_level": ["Normal", "Performance", "Service", "Database", "Code"]}
GROUP_LABEL = {"api": "api", "logs": "log", "metrics": "metric"}
# heuristic bands for the share of expected input features that are missing (NOT measured on held-out data; see /api/predict notes)
RELIABILITY_WARN, RELIABILITY_BAD = 0.10, 0.50
