"""Tables 6 and 7: leave-one-transfer-out selection of alpha and the head budget, and
the sensitivity grid it is selected from (CLIP ViT-B/32, 20 seeds).

For each held-out transfer, (alpha, budget) is chosen by the mean mAP over the OTHER
three transfers, then evaluated on the held-out one against the probe trained with the
same budget.

    python experiments/e02_loto.py   -> results/loto_raw.csv, results/loto_selected.csv
"""
import _common  # noqa: F401
import numpy as np
import pandas as pd

from clab import config as C
from clab.core import mAP, views
from clab.data import out_path, seed_split, setting_arrays
from clab.stats import paired_ci

ALPHA_GRID = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
BUDGET_GRID = [20, 50, 100, 200]


def main():
    rows = []
    for sn in C.ORDER:
        Fs_a, Ys_a, Ft_a, Yt_a, A, Ac = setting_arrays(sn, C.HEADLINE)
        for seed in C.SEEDS:
            Fs, Ys, Ft, Yt = seed_split(Fs_a, Ys_a, Ft_a, Yt_a, seed)
            for ep in BUDGET_GRID:
                v = views(Fs, Ys, Ft, A, Ac, seed, ep)
                lin = mAP(Yt, v["Ht"])
                for al in ALPHA_GRID:
                    rows.append(dict(setting=sn, seed=seed, epochs=ep, alpha=al,
                                     mAP=mAP(Yt, al * v["zU"] + (1 - al) * v["zH"]),
                                     linear=lin))
        print(f"  {sn:12s} done", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(out_path("loto_raw.csv"), index=False)

    g = d.groupby(["setting", "epochs", "alpha"]).mAP.mean().reset_index()
    sel = []
    for held in C.ORDER:
        ep, al = g[g.setting != held].groupby(["epochs", "alpha"]).mAP.mean().idxmax()
        q = d[(d.setting == held) & (d.epochs == ep)
              & np.isclose(d.alpha, al)].sort_values("seed")
        m, lo, hi = paired_ci(q.mAP.values - q.linear.values)
        sel.append(dict(held_out=held, alpha=al, epochs=int(ep), clab=q.mAP.mean(),
                        probe=q.linear.mean(), delta=m, lo=lo, hi=hi))
        print(f"  held out {held:12s}: alpha {al}, {int(ep)} epochs, "
              f"{q.mAP.mean():.2f} vs {q.linear.mean():.2f}, d {m:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    pd.DataFrame(sel).to_csv(out_path("loto_selected.csv"), index=False)


if __name__ == "__main__":
    main()
