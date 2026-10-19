"""Replay of recorded AnoMod data: it must never emit a training block, and the windows built live from replayed events must equal the training windows."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from app.backend import config, monitor as M  # noqa: E402

TIDY = ROOT.parent / "AnoMod" / "processed" / "tidy" / "api.parquet"
needs_data = pytest.mark.skipif(not TIDY.exists() or not (ROOT / "inputs" / "SN_windows.parquet").exists(), reason="AnoMod tidy tables / window tables not present")
needs_models = pytest.mark.skipif(not (config.MODELS_DIR / "label_binary__XGB.joblib").exists() or not config.PREPROCESSOR.exists(), reason="models not present")
STD_ATOL = 1e-5      # the training code computes std as sqrt(E[x^2] - E[x]^2) around a shift: it returns ~4e-6 where the exact std is 0


@pytest.fixture(scope="module")
def R():
    import replay_scenario
    return replay_scenario


@pytest.fixture(scope="module")
def tables():
    import preprocess as P
    df = P.load_windows()
    sp = pd.read_csv(ROOT / "report_outputs" / "split_ids.csv")
    return df, sp


@needs_data
def test_every_planned_block_is_a_test_block_for_every_scenario_and_seed(R, tables):
    df, sp = tables
    split = sp.set_index("window_id").split
    tb = R.test_blocks(df, sp)
    assert len(tb) > 0 and tb.n_windows.sum() == 647
    for scen in R.CLASSES:
        for system in ("SN", "TT", "both"):
            for seed in (1, 42, 7):
                plan = R.plan_segments(scen, system, blocks=4, seed=seed, tb=tb)
                assert plan, (scen, system)
                assert R.assert_test_only(plan, split)
                for s, rows in plan.items():
                    for r in rows:
                        assert r["label_level"] == scen and r["system"] == s
                        assert all(split[w] == "test" for w in r["window_ids"]) and r["n_windows"] == 10 and r["end"] - r["start"] == 120


@needs_data
def test_the_guard_refuses_a_block_with_training_windows(R, tables):
    df, sp = tables
    split = sp.set_index("window_id").split
    train_block = df.merge(sp[["window_id", "split"]], on="window_id", suffixes=("", "_s"))
    train_block = train_block[train_block.split == "train"].groupby("block_id").head(10).block_id.iloc[0]
    g = df[df.block_id == train_block]
    fake = {"SN": [dict(block_id=train_block, window_ids=g.window_id.tolist())]}
    with pytest.raises(AssertionError, match="refusing to replay"):
        R.assert_test_only(fake, split)


@needs_data
def test_events_replayed_for_a_block_build_exactly_that_blocks_test_windows(R, tables):
    """offline through the real Stream code: emitted window starts == the block's test windows, none of them is a training window"""
    df, sp = tables
    split = sp.set_index("window_id").split
    tb = R.test_blocks(df, sp)
    rng = np.random.RandomState(3)
    pick = tb[tb.n_windows == 10].sample(8, random_state=3)
    train_starts = set(((df.window_start.astype("int64") / 1e9).round(3).astype(str) + "|" + df.run_id)[df.window_id.map(split) == "train"])
    for _, r in pick.iterrows():
        api, lg, mt = R.load_events(r.system, r.scenario, r.start, r.end)
        assert api.t.min() >= r.start - 1e-6 if len(api) else True
        a, l, m = R.to_events(api, lg, mt)
        st = M.Stream(r.system)
        st.reset(r.start)
        st.add_rows("api", [M.clean_api(e) for e in a]); st.add_rows("logs", [M.clean_log(e) for e in l]); st.add_rows("metrics", [M.clean_metric(e) for e in m])
        st.watermark = max(st.watermark, r.end) + M.LATENESS + 1e-6          # what /ingest/flush does
        starts = []
        while st.ready():
            ws, _ = M.WF.window_bounds(st.anchor, st.next_k)
            starts.append(round(ws, 3))
            st.advance()
        assert starts == [round(x, 3) for x in r.window_starts]
        assert not (set(f"{s}|{r.run_id}" for s in starts) & train_starts)


@needs_data
def test_live_window_features_equal_the_training_tables(R, tables):
    """the shared feature code reproduces the training windows: API and log features exactly, metrics too except the counter rates of the first 2 windows"""
    df, sp = tables
    tb = R.test_blocks(df, sp)
    import preprocess as P
    feats = P.feature_columns(df)
    pick = tb[tb.n_windows == 10].groupby(["system", "label_level"]).head(1)
    compared, bad_noncounter, bad_counter_late, bad_counter_early = 0, [], [], 0
    for _, r in pick.iterrows():
        api, lg, mt = R.load_events(r.system, r.scenario, r.start, r.end)
        a, l, m = R.to_events(api, lg, mt)
        st = M.Stream(r.system)
        st.reset(r.start)
        st.add_rows("api", [M.clean_api(e) for e in a]); st.add_rows("logs", [M.clean_log(e) for e in l]); st.add_rows("metrics", [M.clean_metric(e) for e in m])
        tbl = df[df.window_id.isin(r.window_ids)].set_index("window_id")
        for k in range(10):
            f, _ = st.window_features(k)
            ref = tbl.loc[r.window_ids[k]]
            for c in feats:
                x, y = f.get(c, math.nan), float(ref[c])
                if math.isnan(x) and math.isnan(y):
                    ok = True
                elif math.isnan(x) or math.isnan(y):
                    ok = False
                else:
                    ok = abs(x - y) <= 1e-6 * max(1.0, abs(y)) + (STD_ATOL if c.endswith("__std") else 0.0)
                compared += 1
                if not ok:
                    counter = c.startswith("m_") and c[2:].rsplit("__", 1)[0].endswith("_total") and not c.startswith("m_rate(")
                    if counter and k < M.WARMUP_WINDOWS:
                        bad_counter_early += 1
                    elif counter:
                        bad_counter_late.append((r.block_id, k, c))
                    else:
                        bad_noncounter.append((r.block_id, k, c))
    assert compared == len(pick) * 10 * len(feats)
    assert not bad_noncounter, bad_noncounter[:5]
    assert not bad_counter_late, bad_counter_late[:5]
    assert bad_counter_early > 0          # the warm-up exception is real, not hypothetical


@needs_data
@needs_models
def test_replay_client_end_to_end_through_the_ingestion_api(R, tables, client):
    df, sp = tables
    tb = R.test_blocks(df, sp)
    row = tb[(tb.system == "SN") & (tb.label_level == "Performance") & (tb.n_windows == 10)].iloc[0]
    client.put("/api/monitor/config", json=dict(model="XGB", label_version="label_binary"))
    logs = []
    ok = R.replay_segment(client, "", "SN", row, 60, R.Control(client, None), lambda *a: logs.append(a))
    assert ok
    s = client.get("/api/monitor/state").json()["systems"]["SN"]
    assert s["counters"]["windows"] == 10 and s["meta"]["scenario"] == row["scenario"] and s["meta"]["source"].startswith("replay of recorded AnoMod")
    starts = [round(h["window_start"], 3) for h in s["history"]]
    assert starts == [round(x, 3) for x in row["window_starts"]]
    c = s["counters"]
    assert c["labelled"] == 8 and c["correct_alerts"] + c["missed"] == 8 and c["false_alarms"] == 0     # a Performance block: every scored window is an anomaly window
    assert s["last"]["truth"]["known_from"] == "recording" and s["last"]["truth"]["label_level"] == "Performance"
    assert any(("segment" in str(x)) for x in logs)


def test_replay_control_endpoints_validate_input(client):
    assert client.get("/api/replay/control").json()["paused"] in (True, False)
    assert client.post("/api/replay/control", json=dict(speed=5)).status_code == 422
    assert client.post("/api/replay/control", json=dict(paused=True, speed=10)).json()["paused"] is True
    assert client.post("/api/replay/control", json=dict(paused=False)).json()["paused"] is False
    r = client.post("/api/replay/start", json=dict(scenario="Banana"))
    assert r.status_code == 422 and "scenario" in r.json()["error"]
    assert client.post("/api/replay/start", json=dict(scenario="Normal", speed=7)).status_code == 422
    assert client.get("/api/replay/status").json()["running"] is False
