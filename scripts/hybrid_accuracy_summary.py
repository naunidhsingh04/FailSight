"""One table of the balanced accuracy of every hybrid on the untouched 20% test part: trained and tested on AnoMod alone, on LO2 alone, and trained on both data sets together
(pooled test part, and that same pooled test part split by data set).
python scripts/hybrid_accuracy_summary.py   ->  report_outputs/hybrid_accuracy_by_dataset.csv"""
from pathlib import Path

import pandas as pd

REP = Path(__file__).resolve().parent.parent / "report_outputs"


def main():
    a = pd.read_csv(REP / "hybrid_anomod_leaderboard.csv")
    l = pd.read_csv(REP / "hybrid_lo2_leaderboard.csv")
    b = pd.read_csv(REP / "both_leaderboard.csv")
    b = b[b.family == "Hybrid model"]
    names = list(dict.fromkeys(list(a.model) + list(l.model) + list(b.model)))
    pick = lambda d, lab: d[d.label_version == lab].set_index("model").test_bal_acc
    rows = []
    for n in names:
        r = dict(model=n)
        r["AnoMod_alone_binary"] = pick(a, "label_binary").get(n)
        r["AnoMod_alone_level"] = pick(a, "label_level").get(n)
        r["LO2_alone_binary"] = pick(l, "label_binary").get(n)
        r["LO2_alone_level"] = pick(l, "label_level").get(n)
        bb = b[b.model == n]
        r["both_trained_together_binary"] = bb.test_bal_acc.iloc[0] if len(bb) else None
        r["both_tested_on_AnoMod_part"] = bb.ba_AnoMod.iloc[0] if len(bb) else None
        r["both_tested_on_LO2_part"] = bb.ba_LO2.iloc[0] if len(bb) else None
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(REP / "hybrid_accuracy_by_dataset.csv", index=False, float_format="%.4f")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
