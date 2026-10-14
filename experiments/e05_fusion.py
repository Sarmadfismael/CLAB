"""Table 13: the fusion rule. The two branches are fixed; only the rule combining them
changes. Every rule uses the same weight alpha = 0.30 and the same 50-epoch head.

    z-score      CLAB: Z[.] with location and scale from SOURCE scores (inductive)
    min-max      per-class min-max range from SOURCE scores (inductive)
    rank         per-class rank among the target images (transductive by nature)
    sigmoid      sigmoid of each branch (inductive)
    raw          no standardization (inductive)

    python experiments/e05_fusion.py   -> results/fusion.csv   (~1 min)
"""
import _common  # noqa: F401
import numpy as np
import pandas as pd

from clab import config as C
from clab.core import EPS, F32, mAP, sigmoid, views
from clab.data import out_path, seed_split, setting_arrays


def minmax(Mt, Ms):
    lo, hi = Ms.min(0, keepdims=True), Ms.max(0, keepdims=True)
    return (Mt - lo) / np.maximum(hi - lo, EPS)


def rank(M):
    return np.argsort(np.argsort(M, axis=0), axis=0).astype(F32) / max(len(M) - 1, 1)


def main():
    a = C.ALPHA
    rows = []
    for sn in C.ORDER:
        Fs_a, Ys_a, Ft_a, Yt_a, A, Ac = setting_arrays(sn, C.HEADLINE)
        for seed in C.SEEDS:
            Fs, Ys, Ft, Yt = seed_split(Fs_a, Ys_a, Ft_a, Yt_a, seed)
            v = views(Fs, Ys, Ft, A, Ac, seed, C.EPOCHS)
            u, us, h, hs = v["Ut"], v["Us"], v["Ht"], v["Hs"]
            S = {"zscore": a * v["zU"] + (1 - a) * v["zH"],
                 "minmax": a * minmax(u, us) + (1 - a) * minmax(h, hs),
                 "rank": a * rank(u) + (1 - a) * rank(h),
                 "sigmoid": a * sigmoid(u) + (1 - a) * sigmoid(h),
                 "raw": a * u + (1 - a) * h}
            for m, sc in S.items():
                rows.append(dict(setting=sn, seed=seed, rule=m, mAP=mAP(Yt, sc)))
        print(f"  {sn:12s} done", flush=True)
    pd.DataFrame(rows).to_csv(out_path("fusion.csv"), index=False)
    print("wrote results/fusion.csv")


if __name__ == "__main__":
    main()
