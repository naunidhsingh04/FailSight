"""Train and test every model family on BOTH data sets together (see both_data.py): the nine distinct hybrid classes (the ten hybrids = five of AnoMod + five of LO2, Autoencoder + LSTM is in both lists),
the classical algorithms, their Optuna-tuned versions, the deep models with XGBoost heads, stacking and soft voting. Same protocol as before (80:20 split first, preprocessing and SMOTE fitted on
training parts only, 5-fold, out-of-fold thresholds, config frozen before the single test scoring, shuffled-label control, bootstrap ranges). Binary labels only.
Results: report_outputs/both_leaderboard.csv (+ hybrid_both_* and both_full_* files).   python scripts/run_both.py [--stage hybrids|families|assemble]"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import hybrid_engine as E                       # noqa: E402
E.LABELS = ["label_binary"]
from both_data import BothData                  # noqa: E402
from models_src.hybrid import ANOMOD_HYBRIDS, LO2_HYBRIDS   # noqa: E402
import lo2_full as F                            # noqa: E402
import lo2_deep as DP                           # noqa: E402

REP, ROOT = E.REP, E.ROOT
REG = {**ANOMOD_HYBRIDS, **LO2_HYBRIDS}
STATE_H = ROOT / "outputs" / "hybrid_state" / "both_hybrid"
STATE_F = ROOT / "outputs" / "hybrid_state" / "both_full"
MODELS = ROOT / "models" / "hybrid" / "both"
F.STATE, F.PREFIX = STATE_F, "both_full_"
DP.STATE, DP.PREFIX, DP.MDIR = STATE_F, "both_full_", MODELS
LABEL = "label_binary"


def by_dataset(D, state, names):
    rows = []
    yte = D.y[LABEL][D.test_pos]
    ds = D.dataset[D.test_pos]
    for n in names:
        p = Path(state) / f"test_proba__{LABEL}__{n}.npy"
        if not p.exists():
            continue
        P = np.load(p)
        r = dict(model=n)
        for d in ("AnoMod", "LO2"):
            m = ds == d
            r[f"ba_{d}"] = E.bal_f1(yte[m], P[m])[0]
            r[f"healthy_recall_{d}"] = E.normal_recall(yte[m], P[m])
        rows.append(r)
    return pd.DataFrame(rows)


def assemble(D):
    hn = list(REG)
    fam = F.TAB + F.TUNED + F.DEEP + F.DEEP_HEAD + ["Stacking", "SoftVote"]
    lh = E.assemble(D, hn, "hybrid_both_", STATE_H, runid=E.run_id_rule(D, None))
    lf = E.assemble(D, fam, "both_full_", STATE_F, runid=None, n_boot=1000)
    lh["family"] = "Hybrid model"
    names = {**{n: "Decision-tree / linear model" for n in F.TAB + F.TUNED}, **{n: "Neural network" for n in F.DEEP + F.DEEP_HEAD}, "Stacking": "Combination of models", "SoftVote": "Combination of models"}
    lf["family"] = lf.model.map(names)
    lead = pd.concat([lh, lf], ignore_index=True)
    bd = pd.concat([by_dataset(D, STATE_H, hn), by_dataset(D, STATE_F, fam)])
    lead = lead.merge(bd, on="model", how="left")
    lead.sort_values("test_bal_acc", ascending=False).to_csv(REP / "both_leaderboard.csv", index=False, float_format="%.4f")
    E.log("both leaderboard written: " + str(len(lead)) + " rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--until", default="", help="HH:MM: models not started by then are listed as not completed")
    a = ap.parse_args()
    dead = None
    if a.until:
        import datetime as dt
        h, m = map(int, a.until.split(":"))
        now = dt.datetime.now()
        t = now.replace(hour=h, minute=m, second=0, microsecond=0)
        dead = (t + dt.timedelta(days=1) if t < now - dt.timedelta(hours=1) else t).timestamp()
    D = BothData()
    E.log(f"BOTH: train {len(D.train_pos)} (AnoMod {len(D.A.train_pos)} + LO2 {len(D.L.train_pos)}), test {len(D.test_pos)} (AnoMod {len(D.A.test_pos)} + LO2 {len(D.L.test_pos)}), inputs {D.F}")
    if a.stage in ("all", "hybrids"):
        R = E.Runner(D, REG, STATE_H, MODELS, "hybrid_both_", deadline=dead)
        E.imbalance_report(D, "hybrid_both_")
        for n in REG:
            R.run(n, LABEL)
    if a.stage in ("all", "families"):
        T = F.TabRunner(D)
        for n in F.TAB:
            T.run(n, LABEL)
        for n in F.TUNED:
            info = T.tune(n, LABEL, 40)
            T.run(n, LABEL, params=info["best_params"], tuned_info=info)
        DP.run_all(D, dead)
        if all((STATE_F / f"result__{LABEL}__{n}.json").exists() for n in F.TAB):
            F.record_derived(D, LABEL, "Stacking", F.TAB)
            F.record_derived(D, LABEL, "SoftVote", F.TAB)
    assemble(D)


if __name__ == "__main__":
    main()
