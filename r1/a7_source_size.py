"""R2.4: does the 50-epoch / alpha = 0.30 recipe depend on source size?
The source fitting set is subsampled to 25%, 50% and 100% of its usual size."""
import _paths  # noqa: F401  (repository paths; must come before clab)
from common import *
import pandas as pd

FRACS = [0.25, 0.5, 1.0]
BUDGETS = [20, 50, 100, 200]
ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5]
rows = []
for sn in ORDER:
    Fs_a, Ys_a, Ft_a, Yt_a, A, Ac = setting_arrays(sn, CLIP)
    for seed in C.SEEDS:
        si, sh, tr, ti = seed_parts(Fs_a, Ys_a, Ft_a, Yt_a, seed)
        Ft, Yt = Ft_a[ti], Yt_a[ti]
        perm = np.random.RandomState(500 + seed).permutation(len(si))
        for fr in FRACS:
            sub = si[perm[:int(round(fr * len(si)))]]
            Fs, Ys = Fs_a[sub], Ys_a[sub]
            for ep in BUDGETS:
                v = views(Fs, Ys, Ft, A, Ac, seed, ep)
                rec = dict(setting=sn, seed=seed, frac=fr, n_source=len(sub), epochs=ep, probe=mAP(Yt, v['Ht']))
                for al in ALPHAS:
                    rec[f'a{al}'] = mAP(Yt, al * v['zU'] + (1 - al) * v['zH'])
                rows.append(rec)
    print(sn, 'done', flush=True)
d = pd.DataFrame(rows); d.to_csv(OUT / 'a7_source_size.csv', index=False)
cols = ['probe'] + [f'a{al}' for al in ALPHAS]
g = d.groupby(['frac', 'epochs'])[cols].mean()
print(g.round(2).to_string())
print(d.groupby(['setting', 'frac', 'epochs'])[['probe', 'a0.3']].mean().unstack('epochs').round(2).to_string())
for fr in FRACS:
    q = d[(d.frac == fr) & (d.epochs == 50)]
    for sn in ORDER:
        x = q[q.setting == sn].sort_values('seed')
        m_, lo, hi = paired_ci(x['a0.3'].values - x.probe.values)
        print(fr, sn, f"clab-probe {m_:+.2f} [{lo:+.2f},{hi:+.2f}]")
