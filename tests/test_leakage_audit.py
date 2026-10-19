"""Unit tests behind check C1 of the leakage audit: fitted state must not depend on test (or validation) rows."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import preprocess as P  # noqa: E402

pytestmark = pytest.mark.skipif(not (ROOT / "inputs" / "SN_windows.parquet").exists() or not (ROOT / "report_outputs" / "split_ids.csv").exists(),
                                reason="window tables (inputs/) or split_ids.csv not present")
ARRAYS = ("median", "lo", "hi", "mean_", "scale_", "absent", "keep", "log_mask")


@pytest.fixture(scope="module")
def data():
    df = P.load_windows()
    sp = pd.read_csv(ROOT / "report_outputs" / "split_ids.csv")
    fa = pd.read_csv(ROOT / "report_outputs" / "fold_assignment.csv")
    d = df.merge(sp[["window_id", "split"]], on="window_id").merge(fa[["window_id", "fold"]], on="window_id")
    return d


def test_split_counts_and_disjoint_ids(data):
    tr, te = set(data.window_id[data.split == "train"]), set(data.window_id[data.split == "test"])
    assert (len(tr), len(te)) == (2549, 647) and not (tr & te)


def test_global_preprocessor_ignores_test_rows(data):
    feats = P.feature_columns(data)
    a = P.fit_preprocessor(data[data.split == "train"])
    wrecked = data.copy()
    m = (wrecked.split == "test").to_numpy()
    wrecked.loc[m, feats] = wrecked.loc[m, feats].to_numpy(float) * 1e6 + 99.0
    b = P.fit_preprocessor(wrecked[wrecked.split == "train"])
    for n in ARRAYS:
        assert np.array_equal(getattr(a, n), getattr(b, n), equal_nan=True), n
    assert a.kept_cols == b.kept_cols


def test_fold_preprocessor_ignores_validation_and_test_rows(data):
    feats = P.feature_columns(data)
    tr = data[(data.split == "train") & (data.fold != 3)]
    wrecked = data.copy()
    m = ((wrecked.split == "test") | (wrecked.fold == 3)).to_numpy()
    wrecked.loc[m, feats] = wrecked.loc[m, feats].to_numpy(float) * -1e5 + 7.0
    a = P.fit_preprocessor(tr)
    b = P.fit_preprocessor(wrecked[(wrecked.split == "train") & (wrecked.fold != 3)])
    for n in ("median", "lo", "hi", "mean_", "scale_"):
        assert np.array_equal(getattr(a, n), getattr(b, n), equal_nan=True), n


def test_fit_refuses_test_rows(data):
    with pytest.raises(ValueError):
        P.fit_preprocessor(pd.concat([data[data.split == "train"].head(40), data[data.split == "test"].head(3)]))


def test_folds_partition_training_rows(data):
    tr = data[data.split == "train"]
    assert (tr.fold >= 1).all() and (tr.fold <= 10).all()
    assert (data[data.split == "test"].fold == 0).all()
    # no block is split across folds (run-aware / block-grouped folds)
    assert (tr.groupby("block_id").fold.nunique() == 1).all()
