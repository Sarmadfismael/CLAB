"""R2.5: a label co-occurrence prior estimated from SOURCE labels only, applied to CLAB scores.
Refinement: S' = (1 - g) S + g S P, with P the row-normalized conditional co-occurrence matrix
P[i, j] = p(j | i) on the source fitting set (diagonal removed), applied to the standardized
CLAB scores. gamma is selected by leave-one-transfer-out."""
import _paths  # noqa: F401  (repository paths; must come before clab)
from common import *
import pandas as pd

GAMMAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
rows = []
a = C.ALPHA
for sn in ORDER:
    Fs_a, Ys_a, Ft_a, Yt_a, A, Ac = setting_arrays(sn, CLIP)
    for seed in C.SEEDS:
        si, sh, tr, ti = seed_parts(Fs_a, Ys_a, Ft_a, Yt_a, seed)
        Fs, Ys, Ft, Yt = Fs_a[si], Ys_a[si], Ft_a[ti], Yt_a[ti]
        v = views(Fs, Ys, Ft, A, Ac, seed)
        S = a * v['zU'] + (1 - a) * v['zH']
        cnt = Ys.T @ Ys
        P = cnt / np.maximum(np.diag(cnt)[:, None], 1)
        np.fill_diagonal(P, 0)
        P = P / np.maximum(P.sum(1, keepdims=True), EPS)
        for gm in GAMMAS:
            rows.append(dict(setting=sn, seed=seed, gamma=gm, clab=mAP(Yt, (1 - gm) * S + gm * S @ P.T),
                             probe=mAP(Yt, v['Ht'])))
    print(sn, 'done', flush=True)
d = pd.DataFrame(rows); d.to_csv(OUT / 'a8_cooc.csv', index=False)
g = d.groupby(['gamma', 'setting']).clab.mean().unstack(); g['mean'] = g.mean(1); print(g.round(2))
for held in ORDER:
    gs = d[d.setting != held].groupby('gamma').clab.mean().idxmax()
    q = d[(d.setting == held) & np.isclose(d.gamma, gs)].sort_values('seed').clab.values
    base = d[(d.setting == held) & (d.gamma == 0)].sort_values('seed').clab.values
    m_, lo, hi = paired_ci(q - base)
    print('LOTO', held, 'gamma', gs, f"{m_:+.2f} [{lo:+.2f},{hi:+.2f}]")
