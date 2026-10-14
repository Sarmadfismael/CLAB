"""Table 14 (per-class AP on the DFC15 transfers) and the Proposition 1 measurements of
Section 6.4 (source vs target standardization constants, dispersion ratio rho).

    python experiments/e06_perclass_invariance.py
        -> results/perclass.csv, results/invariance.csv   (~1 min)
"""
import _common  # noqa: F401
import numpy as np
import pandas as pd

from clab import config as C
from clab.core import EPS, ap, mAP, views
from clab.data import out_path, seed_split, setting_arrays


def main():
    a = C.ALPHA
    out = []
    for sn in ["UCM->DFC15", "AID->DFC15"]:
        Fs_a, Ys_a, Ft_a, Yt_a, A, Ac = setting_arrays(sn, C.HEADLINE)
        acc = {m: {c: [] for c in range(6)} for m in ["zeroshot", "probe", "clab"]}
        for seed in C.SEEDS:
            Fs, Ys, Ft, Yt = seed_split(Fs_a, Ys_a, Ft_a, Yt_a, seed)
            v = views(Fs, Ys, Ft, A, Ac, seed, C.EPOCHS)
            S = {"zeroshot": v["zU"], "probe": v["Ht"], "clab": a * v["zU"] + (1 - a) * v["zH"]}
            for m in acc:
                for c in range(6):
                    acc[m][c].append(ap(Yt[:, c], S[m][:, c]) * 100)
        for c, name in enumerate(C.SHARED_6):
            z_, p_, k_ = (np.mean(acc[m][c]) for m in ["zeroshot", "probe", "clab"])
            out.append(dict(setting=sn, label=name, prevalence=100 * Yt_a[:, c].mean(),
                            zeroshot=z_, probe=p_, clab=k_, d=k_ - p_))
    pd.DataFrame(out).to_csv(out_path("perclass.csv"), index=False)

    rows = []
    for sn in C.ORDER:
        Fs_a, Ys_a, Ft_a, Yt_a, A, Ac = setting_arrays(sn, C.HEADLINE)
        for seed in C.SEEDS:
            Fs, Ys, Ft, Yt = seed_split(Fs_a, Ys_a, Ft_a, Yt_a, seed)
            v = views(Fs, Ys, Ft, A, Ac, seed, C.EPOCHS)
            Us, Ut, Hs, Ht = v["Us"], v["Ut"], v["Hs"], v["Ht"]
            f = lambda mu_u, sd_u, mu_h, sd_h: (a * (Ut - mu_u) / (sd_u + EPS)
                                                + (1 - a) * (Ht - mu_h) / (sd_h + EPS))
            rows.append(dict(setting=sn, seed=seed,
                             source=mAP(Yt, f(Us.mean(), Us.std(), Hs.mean(), Hs.std())),
                             target=mAP(Yt, f(Ut.mean(), Ut.std(), Ht.mean(), Ht.std())),
                             zero_means=mAP(Yt, f(0., Us.std(), 0., Hs.std())),
                             rho_src=float(Hs.std() / Us.std()),
                             rho_tgt=float(Ht.std() / Ut.std())))
    pd.DataFrame(rows).to_csv(out_path("invariance.csv"), index=False)
    print("wrote results/perclass.csv, results/invariance.csv")


if __name__ == "__main__":
    main()
