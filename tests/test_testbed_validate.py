"""Run-held-out validation: the split is by whole runs and meets its constraints, alert rules, detection delay, false alarms and lead time are computed as documented."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from testbed import dataset as DS, validate as V  # noqa: E402

needs_data = pytest.mark.skipif(not DS.OUT.exists(), reason="processed/testbed_windows.parquet not built (python -m testbed.dataset)")


@needs_data
def test_run_assignment_is_by_whole_runs_and_meets_the_constraints():
    W, base, rel = V.load()
    S = V.run_faults(W)
    asg, _ = V.assign_runs(W, S)
    test = set(asg[asg.split == "test"].run_id)
    train = set(asg[asg.split == "train"].run_id)
    assert not (test & train) and len(test) == V.N_TEST_RUNS and test | train == set(W.run_id)
    assert set(S[S.run_id.isin(test)].fault_type) == set(S.fault_type)                   # every fault type appears in the test runs
    assert len(test & set(W[W.fault_free_run].run_id)) >= 2                              # at least two fault-free test runs
    folds = asg[asg.split == "train"].set_index("run_id").fold.astype(int)
    assert set(folds.index) == train and folds.nunique() == V.N_FOLDS                          # every train run is in exactly one fold, runs are never split
    # the seeds of the live demo (9000 and above) are never used
    assert W.run_seed.max() < 9000
    assert V.assign_runs(W, S)[0].equals(asg)                                            # fixed seed: the same assignment every time


def test_alert_rules_and_episodes():
    f = np.array([0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1], bool)
    assert V.alert_series(f, "1 window").tolist() == f.tolist()
    a = V.alert_series(f, "2 of last 3")
    assert a.tolist() == [False, False, False, False, False, True, True, False, False, False, False]
    assert V.episodes(np.array([1, 1, 0, 1, 0, 0, 0, 1, 1], bool)) == [0, 7]              # a new episode starts after 3 quiet windows
    assert V.episodes(np.array([1, 0, 1], bool)) == [0]


def test_detection_delay_false_alarms_and_lead_time():
    n = 20
    D = pd.DataFrame(dict(run_id=["rA"] * n + ["rB"] * n, window_index=list(range(n)) * 2, window_end=[1000 + 30 + 10 * k for k in range(n)] * 2, fault_free_run=[False] * n + [True] * n))
    S = pd.DataFrame([dict(run_id="rA", fault_id="f", fault_type="leak", severity_level="mild", kind="gradual", start_ts=1100.0, end_ts=1180.0, failure_ts=1150.0)])
    fl = np.zeros(2 * n, bool)
    fl[[10, 11, 12]] = True                  # run A: windows ending 1130, 1140, 1150
    fl[n + 3] = True                         # run B (fault-free): one false alarm
    inst, free, other = V.detection_table(D, fl, S, "m", "1 window", ["rA", "rB"])
    r = inst.iloc[0]
    assert r.detected and r.delay_s == 30.0                                                # first alert window ends at 1130, the fault started at 1100
    assert r.warned_before_failure and r.lead_time_s == 20.0                               # failure at 1150
    assert free[0] == 1 and abs(free[1] - n * 10 / 3600) < 1e-9
    assert other[0] == 0


def test_run_bootstrap_resamples_whole_runs():
    runs = np.array(["a"] * 5 + ["b"] * 5 + ["c"] * 5)
    b = V.RunBoot(runs, 50)
    for i in range(50):
        rows = b.rows(i)
        assert len(rows) % 5 == 0 and set(runs[rows]) <= {"a", "b", "c"}
        for r in set(runs[rows]):
            assert (runs[rows] == r).sum() % 5 == 0                                        # a run is always taken whole
