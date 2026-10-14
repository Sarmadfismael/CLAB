"""Section 6.6: source-head budget sweep. Source-train mAP and target mAP of the head as
it is trained for 20 ... 800 epochs, everything else fixed, split seeds 0-4, both
backbones. (This sweep is run in float64, as in the original measurement.)

    python experiments/e07_source_budget.py   -> results/source_budget.csv   (~3 min)
"""
import _common  # noqa: F401
import numpy as np
import pandas as pd

from clab import config as C
from clab.core import logistic, mAP
from clab.data import out_path, seed_split, setting_arrays

EPOCH_GRID = [20, 50, 100, 200, 400, 800]
SEEDS = [0, 1, 2, 3, 4]


def main():
    rows = []
    for sn in C.ORDER:
        for bb in [C.HEADLINE, C.DIAGNOSTIC]:
            Fs_a, Ys_a, Ft_a, Yt_a, _, _ = setting_arrays(sn, bb)
            for seed in SEEDS:
                Fs, Ys, Ft, Yt = seed_split(Fs_a, Ys_a, Ft_a, Yt_a, seed)
                mu, sd = Fs.mean(0, keepdims=True), Fs.std(0, keepdims=True) + 1e-6
                Zs, Zt = (Fs - mu) / sd, (Ft - mu) / sd
                for ep in EPOCH_GRID:
                    W, b = logistic(Zs, Ys, ep, seed=seed, dtype=np.float64)
                    rows.append(dict(setting=sn, backbone=bb, seed=seed, epochs=ep,
                                     src_train_mAP=mAP(Ys, Zs @ W + b),
                                     tgt_linear_mAP=mAP(Yt, Zt @ W + b)))
            print(f"  {sn:12s} {bb:22s} done", flush=True)
    pd.DataFrame(rows).to_csv(out_path("source_budget.csv"), index=False)
    print("wrote results/source_budget.csv")


if __name__ == "__main__":
    main()
