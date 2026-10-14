"""Section 6.7 and Table 15: can the blending weight be set per category without
target labels?

Part 1 (oracle, CLIP ViT-B/32, 20 seeds): the gain available to an oracle that picks a
single global alpha, or one alpha per category, using TARGET labels, over the fixed
alpha = 0.30 on the standard CLAB pipeline.

Part 2 (label-free estimators, 20 seeds, both backbones, 8 arms): the source split is
divided again into a fitting part (80%) and a source-validation part (20%, seed
1000 + split seed). One head is fitted on the fitting part and shared by every row, so
the fixed weight and each estimator see identical branches. Both branches are
standardized with constants from the fitting-part source scores, as in CLAB.

    rank_agreement            alpha_c = clip(0.5 * (1 - Spearman(u_c, h_c)), 0, 1) on target
    source_validation         alpha_c maximising AP on the source-validation part
    shrunk_source_validation  halfway between that and the fixed weight
    self_training             alpha_c maximising AP against pseudo-labels from the fixed
                              blend thresholded at its per-class median

    python experiments/e08_alpha_estimators.py
        -> results/alpha_oracle.csv, results/alpha_estimators.csv   (~10 min)
"""
import _common  # noqa: F401
import numpy as np
import pandas as pd

from clab import config as C
from clab.core import EPS, ap, logistic, mAP, nrm, split_idx, standardize_features, views
from clab.data import out_path, seed_split, setting_arrays

AGRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def best_alpha(y, T, S):
    best, ba = -1.0, C.ALPHA
    for al in AGRID:
        v = ap(y, al * T + (1 - al) * S)
        if not np.isnan(v) and v > best:
            best, ba = v, al
    return ba


def blend(T, S, av):
    return T * av[None, :] + S * (1.0 - av[None, :])


def oracle():
    rows = []
    for sn in C.ORDER:
        Fs_a, Ys_a, Ft_a, Yt_a, A, Ac = setting_arrays(sn, C.HEADLINE)
        for seed in C.SEEDS:
            Fs, Ys, Ft, Yt = seed_split(Fs_a, Ys_a, Ft_a, Yt_a, seed)
            v = views(Fs, Ys, Ft, A, Ac, seed, C.EPOCHS)
            T, S = v["zU"], v["zH"]
            K = Yt.shape[1]
            fixed = mAP(Yt, blend(T, S, np.full(K, C.ALPHA)))
            g = max(AGRID, key=lambda a: mAP(Yt, a * T + (1 - a) * S))
            per = np.array([best_alpha(Yt[:, c], T[:, c], S[:, c]) for c in range(K)])
            rows.append(dict(setting=sn, seed=seed, fixed=fixed,
                             global_oracle=mAP(Yt, g * T + (1 - g) * S), global_alpha=g,
                             perclass_oracle=mAP(Yt, blend(T, S, per)),
                             alpha_spread=float(per.max() - per.min())))
        print(f"  oracle {sn:12s} done", flush=True)
    pd.DataFrame(rows).to_csv(out_path("alpha_oracle.csv"), index=False)


def estimators():
    rows = []
    for sn in C.ORDER:
        for bb in [C.HEADLINE, C.DIAGNOSTIC]:
            Fs_a, Ys_a, Ft_a, Yt_a, A, Ac = setting_arrays(sn, bb)
            for seed in C.SEEDS:
                Xs, Ys, Xt, Yt = seed_split(Fs_a, Ys_a, Ft_a, Yt_a, seed)
                K = Ys.shape[1]
                fi, vi = split_idx(len(Xs), 0.8, 1000 + seed)
                Zf, Zt = standardize_features(Xs[fi], Xt)
                _, Zv = standardize_features(Xs[fi], Xs[vi])
                W, b = logistic(Zf, Ys[fi], C.EPOCHS, seed=seed)
                Uf, Hf = nrm(Xs[fi]) @ Ac.T, Zf @ W + b
                z = lambda M, R: (M - R.mean()) / (R.std() + EPS)
                txt_t, src_t = z(nrm(Xt) @ Ac.T, Uf), z(Zt @ W + b, Hf)
                txt_v, src_v = z(nrm(Xs[vi]) @ Ac.T, Uf), z(Zv @ W + b, Hf)
                Yv = Ys[vi]
                fixed_v = np.full(K, C.ALPHA)
                base = mAP(Yt, blend(txt_t, src_t, fixed_v))
                a1 = np.array([np.clip(0.5 * (1 - spearman(txt_t[:, c], src_t[:, c])), 0, 1)
                               for c in range(K)])
                a2 = np.array([best_alpha(Yv[:, c], txt_v[:, c], src_v[:, c]) for c in range(K)])
                a3 = 0.5 * a2 + 0.5 * C.ALPHA
                S0 = blend(txt_t, src_t, fixed_v)
                P = (S0 >= np.median(S0, axis=0, keepdims=True)).astype(float)
                a4 = np.array([best_alpha(P[:, c], txt_t[:, c], src_t[:, c]) for c in range(K)])
                rec = dict(setting=sn, backbone=bb, seed=seed, fixed=base)
                for nm, av in [("rank_agreement", a1), ("source_validation", a2),
                               ("shrunk_source_validation", a3), ("self_training", a4)]:
                    rec[nm] = mAP(Yt, blend(txt_t, src_t, av))
                rows.append(rec)
            print(f"  estimators {sn:12s} {bb:22s} done", flush=True)
    pd.DataFrame(rows).to_csv(out_path("alpha_estimators.csv"), index=False)


if __name__ == "__main__":
    oracle()
    estimators()
    print("wrote results/alpha_oracle.csv, results/alpha_estimators.csv")
