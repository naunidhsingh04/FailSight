"""Baseline-relative features: a window's baseline never contains future windows, other runs, or (for training windows) test windows."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import universal_features as U  # noqa: E402


def make(n=80, f=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(10, 2, (n, f))
    X[:, 1] = np.abs(X[:, 1])
    is_train = (np.arange(n) // 10) % 5 != 3           # blocks of 10; every 5th block is a test block
    return X, is_train


def test_future_windows_do_not_change_a_window():
    X, tr = make()
    ref = U.relative_features(X, None, tr)
    for k in (20, 33, 47, 70):
        Xp = X.copy()
        Xp[[j for j in range(k - U.GUARD + 1, len(X)) if j != k]] = 1e9
        assert np.array_equal(U.relative_features(Xp, None, tr)[k], ref[k], equal_nan=True)


def test_training_windows_do_not_see_test_windows_but_test_windows_may_see_the_past():
    X, tr = make()
    ref = U.relative_features(X, None, tr)
    Xp = X.copy()
    Xp[~tr] = 1e9                                         # destroy every test window
    out = U.relative_features(Xp, None, tr)
    train_idx = np.where(tr & ~np.isnan(ref).all(axis=1))[0]
    assert len(train_idx) > 10
    for k in train_idx:                                    # train windows: identical although all test windows were overwritten
        assert np.array_equal(out[k], ref[k], equal_nan=True)


def test_first_windows_have_no_value_and_guard_prevents_overlap():
    X, tr = make()
    Z = U.relative_features(X, None, np.ones(len(X), bool))
    assert np.isnan(Z[:U.GUARD + U.MIN_HISTORY - 1]).all()
    assert not np.isnan(Z[U.GUARD + U.MIN_HISTORY]).any()
    # the baseline of window k ends at window k - GUARD: a spike in window k - GUARD + 1 .. k - 1 is invisible to it
    Xs = X.copy()
    Xs[30] = 1e6
    assert np.array_equal(U.relative_features(Xs, None, np.ones(len(X), bool))[32], Z[32], equal_nan=True)


def test_robust_z_value():
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    m, s = U.baseline_stats(vals)
    assert m == 5.0 and s == pytest.approx(1.4826 * 2.0)


def test_constant_zero_baseline_gives_a_large_but_clipped_deviation():
    X = np.zeros((30, 1))
    X[25, 0] = 3.0
    Z = U.relative_features(X, None, np.ones(30, bool))
    assert Z[25, 0] == U.CLIP and Z[24, 0] == 0.0
