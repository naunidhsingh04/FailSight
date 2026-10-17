"""Part A split: API-aware block split, as close to 80:20 as possible.

1. each run's windows (time order) -> contiguous blocks of 10 windows (last block of a run may be shorter);
   a block is "API" if any of its windows has API requests
2. per system, pool the blocks of all runs, shuffle API blocks and non-API blocks separately (RandomState(42)),
   first 80% (rounded to whole blocks) of each group -> train, rest -> test
3. fix-ups: every run keeps >= MIN_TRAIN train blocks and >= min_test test blocks (swaps with other runs, same group when
   possible); then single blocks are moved train<->test until the system's train share of windows is within 79-81%.
   min_test is 3 when 3 test blocks per run still leaves room for a train share of 80%; otherwise it is lowered to the
   largest value that does (the two rules cannot both hold when runs only have ~8-20 blocks).
No window is dropped or trimmed. API windows stay on both sides.
"""
import numpy as np
import pandas as pd

BLOCK, SEED, TRAIN_FRAC, WANT_BLOCKS, LO, HI = 10, 42, 0.8, 3, 0.79, 0.81
MIN_TRAIN = WANT_BLOCKS


def make_blocks(W):
    blocks = []
    for sc, g in W.groupby("scenario", sort=False):
        g = g.sort_values("window_index")
        idx, api = g.index.to_numpy(), (g.api_request_count.to_numpy() > 0)
        for i in range(0, len(idx), BLOCK):
            blocks.append(dict(bid=len(blocks), run=sc, idx=idx[i:i + BLOCK], n=len(idx[i:i + BLOCK]), api=bool(api[i:i + BLOCK].any()),
                               api_windows=int(api[i:i + BLOCK].sum()), train=False))
    return blocks


def share(blocks):
    return sum(b["n"] for b in blocks if b["train"]) / sum(b["n"] for b in blocks)


def run_counts(blocks):
    c = {}
    for b in blocks:
        c.setdefault(b["run"], [0, 0])[0 if b["train"] else 1] += 1   # [train blocks, test blocks]
    return c


def feasibility(blocks):
    """smallest total test windows when every run has m test blocks, for m = 3, 2, 1; and the largest usable m"""
    tot = sum(b["n"] for b in blocks)
    runs = sorted({b["run"] for b in blocks})
    out = {}
    for m in (3, 2, 1):
        out[m] = sum(sum(sorted(b["n"] for b in blocks if b["run"] == r)[:m]) for r in runs)
    best = next(m for m in (3, 2, 1) if out[m] <= (1 - TRAIN_FRAC) * tot)
    return best, out, tot


def block_split(W):
    """-> (train mask aligned with W rows, info dict, per-run DataFrame)"""
    blocks = make_blocks(W)
    min_test, minwin, tot = feasibility(blocks)
    rs = np.random.RandomState(SEED)
    groups = {True: [b["bid"] for b in blocks if b["api"]], False: [b["bid"] for b in blocks if not b["api"]]}
    for g in (True, False):                                  # API blocks first, then non-API (one shared RandomState)
        perm = rs.permutation(groups[g])
        for bid in perm[:int(np.floor(TRAIN_FRAC * len(perm) + 0.5))]:
            blocks[bid]["train"] = True
    info = dict(blocks=len(blocks), api_blocks=len(groups[True]), nonapi_blocks=len(groups[False]), share_initial=share(blocks),
                min_test_blocks_used=min_test, swaps=0, moves=0,
                min_test_windows_if_3_test_blocks_per_run=minwin[3], total_windows=tot,
                max_train_share_if_3_test_blocks_per_run=1 - minwin[3] / tot)
    need = {"train": MIN_TRAIN, "test": min_test}

    # ---- fix-up 1: per-run minimum of train / test blocks, by swapping with a block of another run
    for _ in range(2000):
        rc = run_counts(blocks)
        bad = [(r, side) for r, (a, b) in rc.items() for side, v in (("train", a), ("test", b)) if v < need[side]]
        if not bad:
            break
        r, side = bad[0]
        want_train = side == "train"
        other = "test" if want_train else "train"
        if rc[r][0 if other == "train" else 1] <= need[other]:
            raise RuntimeError(f"run {r} cannot give a block away")
        done = False
        for same_group in (True, False):
            for gb in sorted((b for b in blocks if b["run"] == r and b["train"] != want_train), key=lambda b: (not b["api"], b["bid"])):
                cand = [b for b in blocks if b["run"] != r and b["train"] == want_train
                        and rc[b["run"]][0 if want_train else 1] > need[side]
                        and (not same_group or b["api"] == gb["api"])]
                if cand:
                    db = min(cand, key=lambda b: (abs(b["n"] - gb["n"]), -rc[b["run"]][0 if want_train else 1], b["bid"]))
                    gb["train"], db["train"] = want_train, not want_train
                    info["swaps"] += 1
                    done = True
                    break
            if done:
                break
        if not done:
            raise RuntimeError(f"cannot satisfy minimum {side} blocks for run {r}")
    info["share_after_run_fix"] = share(blocks)

    # ---- fix-up 2: move single blocks until the train share is inside 79-81%
    tot = sum(b["n"] for b in blocks)

    def api_test_frac(g):
        gb = [b for b in blocks if b["api"] == g]
        return sum(not b["train"] for b in gb) / len(gb)

    for _ in range(2000):
        p = share(blocks)
        if LO <= p <= HI:
            break
        to_train = p < LO
        rc = run_counts(blocks)
        cand = []
        for b in blocks:
            if b["train"] == to_train:
                continue
            if rc[b["run"]][1 if to_train else 0] <= (need["test"] if to_train else need["train"]):
                continue
            left = sum((not x["train"]) if to_train else x["train"] for x in blocks if x["api"] == b["api"]) - 1
            if b["api"] and left < 1:
                continue                         # API windows must stay on both sides
            cand.append(b)
        if not cand:
            raise RuntimeError("no movable block left")
        tr = sum(b["n"] for b in blocks if b["train"])

        def key(b):
            newp = (tr + (b["n"] if to_train else -b["n"])) / tot
            dev = api_test_frac(b["api"]) - (1 - TRAIN_FRAC)      # > 0: that group has too many test blocks
            return (abs(newp - TRAIN_FRAC), -dev if to_train else dev, b["bid"])
        min(cand, key=key)["train"] = to_train
        info["moves"] += 1
    else:
        raise RuntimeError("share fix-up did not converge")

    pos = {ix: i for i, ix in enumerate(W.index)}
    mask = np.zeros(len(W), dtype=bool)
    for b in blocks:
        if b["train"]:
            mask[[pos[ix] for ix in b["idx"]]] = True
    rows = []
    for sc in W.scenario.drop_duplicates():
        bb = [b for b in blocks if b["run"] == sc]
        n, tw = sum(b["n"] for b in bb), sum(b["n"] for b in bb if b["train"])
        rows.append(dict(scenario=sc, windows=n, blocks=len(bb), train_blocks=sum(b["train"] for b in bb),
                         test_blocks=sum(not b["train"] for b in bb), train=tw, test=n - tw, train_pct=round(100 * tw / n, 1),
                         api_windows_train=sum(b["api_windows"] for b in bb if b["train"]),
                         api_windows_test=sum(b["api_windows"] for b in bb if not b["train"])))
    info.update(share_final=share(blocks),
                api_blocks_train=sum(b["train"] for b in blocks if b["api"]), api_blocks_test=sum(not b["train"] for b in blocks if b["api"]),
                nonapi_blocks_train=sum(b["train"] for b in blocks if not b["api"]), nonapi_blocks_test=sum(not b["train"] for b in blocks if not b["api"]))
    return mask, info, pd.DataFrame(rows)
