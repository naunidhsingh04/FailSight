import sys
from pathlib import Path
import numpy as np, pandas as pd, pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import preprocess as P


@pytest.fixture(scope="module")
def data():
    try:
        df = P.load_windows()
    except FileNotFoundError:
        pytest.skip("the AnoMod window tables (inputs/*.parquet) are not present: they are data and are not stored in git")
    sp = P.make_splits(df)
    d = df.merge(sp[["window_id", "split"]], on="window_id")
    return d, d[d.split == "train"].copy(), d[d.split == "test"].copy()


@pytest.fixture(scope="module")
def prep(data):
    return P.fit_preprocessor(data[1])


def test_split_counts(data):
    d, tr, te = data
    assert (len(tr), len(te)) == (2549, 647)


def test_fit_rejects_test_rows(data):
    d, tr, te = data
    with pytest.raises(ValueError):
        P.fit_preprocessor(pd.concat([tr.head(50), te.head(5)]))


def test_no_test_row_used_in_fit(data, prep):
    d, tr, te = data
    assert prep.n_fit_rows == len(tr)
    # wildly changing the test rows must not change anything the preprocessor learned or produces for train rows
    te2 = te.copy()
    fc = P.feature_columns(te2)
    te2[fc] = te2[fc] * 1e6 + 12345.0
    p2 = P.fit_preprocessor(pd.concat([tr, te2]).query("split == 'train'"))      # test rows are not part of the fit input
    assert np.array_equal(prep.median, p2.median) and np.array_equal(prep.mean_, p2.mean_) and np.array_equal(prep.scale_, p2.scale_)
    assert np.array_equal(prep.transform(tr), p2.transform(tr))
    # fitted statistics are exactly those of the train rows: recompute one median by hand
    j = prep.raw_cols.index("api_latency_mean_ms") if not prep.log_mask[prep.raw_cols.index("api_latency_mean_ms")] else None
    if j is not None:
        v = tr[tr.system == "SN"].api_latency_mean_ms.dropna()
        assert np.isclose(prep.median[0, j], v.median())
    # test rows can be outside the train range: the output is clipped to the train 0.5/99.5 percentile band
    Z = prep.transform(te2)
    assert np.isfinite(Z).all() and np.abs(Z).max() < 50


def test_fixed_width(data, prep):
    d, tr, te = data
    assert prep.transform(tr).shape == (len(tr), prep.n_out)
    assert prep.transform(te).shape == (len(te), prep.n_out)
    assert prep.transform(te.head(1)).shape == (1, prep.n_out)
    assert prep.transform_raw_window({"system": "SN"}).shape == (prep.n_out,)          # empty window: everything missing
    assert prep.transform_raw_window({"system": "TT", "api_error_rate": 0.2, "unknown_key": 1}).shape == (prep.n_out,)
    assert len(prep.output_names) == prep.n_out == len(set(prep.output_names))
    assert not any(c in prep.output_names for c in prep.const_cols)


def test_inference_equals_batch(data, prep):
    d, tr, te = data
    rows = pd.concat([d[d.system == "SN"].sample(25, random_state=1), d[d.system == "TT"].sample(25, random_state=1)])
    batch = prep.transform(rows)
    for i, (_, r) in enumerate(rows.iterrows()):
        w = {k: (None if pd.isna(v) else v) for k, v in r[prep.raw_cols].items()}
        w["system"] = r.system
        assert np.allclose(prep.transform_raw_window(w), batch[i], atol=1e-12), r.window_id


def test_joblib_roundtrip(data, prep, tmp_path):
    p = P.save_preprocessor(prep, tmp_path / "pp.joblib")
    q = P.load_preprocessor(p)
    assert np.array_equal(q.transform(data[2]), prep.transform(data[2]))


def test_counter_to_rate():
    r = P.counter_to_rate([0, 10, 20, 30, 40], [0, 100, 300, 5, 55])      # reset at index 3
    assert np.isnan(r[0]) and r[1] == 10 and r[2] == 20 and np.isnan(r[3]) and r[4] == 5


def test_absent_features_are_zero(data, prep):
    d, tr, te = data
    Z = prep.transform(d)
    k = len(prep.kept_cols)
    for s, name in ((0, "SN"), (1, "TT")):
        rows = (d.system == name).to_numpy()
        cols = [i for i, c in enumerate(prep.kept_cols) if prep.absent[s, prep.raw_cols.index(c)]]
        assert len(cols) > 0 and (Z[rows][:, cols] == 0).all()


def test_raw_counter_samples_path(prep):
    name = prep.counter_cols[0][2:].rsplit("__", 1)[0]
    sysname = "TT" if not prep.absent[1, prep.raw_cols.index(prep.counter_cols[0])] else "SN"
    via_counters = prep.transform_raw_window({"system": sysname, "_counter_samples": {name: ([0, 10, 20, 30], [0, 50, 100, 150])}})
    direct = prep.transform_raw_window({"system": sysname, f"m_{name}__mean": 5.0, f"m_{name}__max": 5.0, f"m_{name}__std": 0.0})
    assert np.array_equal(via_counters, direct)
