import _paths  # noqa: F401  (repository paths; must come before clab)
import os
"""R2.2: (a) intervals that resample target images as well as split realizations;
(b) per-class AP for all transfers; (c) Holm over the Table 4 family."""
from common import *
import pandas as pd

B = 1000
rng = np.random.default_rng(2026)
a = C.ALPHA
boot_rows, pc_rows = [], []
for bb in [CLIP, RC]:
    for sn in ORDER:
        Fs_a, Ys_a, Ft_a, Yt_a, A, Ac = setting_arrays(sn, bb)
        S_clab, S_probe, S_zs = [], [], []
        for seed in C.SEEDS:
            si, sh, tr, ti = seed_parts(Fs_a, Ys_a, Ft_a, Yt_a, seed)
            b = branch_scores(Fs_a[si], Ys_a[si], Ft_a, A, Ac, seed)   # score the WHOLE target
            S_probe.append(b['h'])
            S_clab.append(a * zscore(b['u'], b['us']) + (1 - a) * zscore(b['h'], b['hs']))
            S_zs.append(b['u'])
            if bb == CLIP:   # per-class AP on the standard 20% evaluation split
                for c in range(Yt_a.shape[1]):
                    y = Yt_a[ti, c]
                    pc_rows.append(dict(setting=sn, seed=seed, label=(C.LABELS_17 if Yt_a.shape[1] == 17 else C.SHARED_6)[c],
                                        prevalence=100 * Yt_a[:, c].mean(),
                                        zeroshot=ap(y, b['u'][ti, c]) * 100, probe=ap(y, b['h'][ti, c]) * 100,
                                        clab=ap(y, S_clab[-1][ti, c]) * 100))
        n = len(Yt_a)
        full_diff = np.mean([mAP(Yt_a, S_clab[k]) - mAP(Yt_a, S_probe[k]) for k in range(len(C.SEEDS))])
        dist = []
        for _ in range(B):
            ks = rng.integers(0, len(C.SEEDS), len(C.SEEDS))
            idx = rng.integers(0, n, n)
            Y = Yt_a[idx]
            dist.append(np.mean([mAP(Y, S_clab[k][idx]) - mAP(Y, S_probe[k][idx]) for k in ks]))
        dist = np.array(dist)
        boot_rows.append(dict(backbone=bb, setting=sn, n_images=n, delta=full_diff,
                              lo=np.percentile(dist, 2.5), hi=np.percentile(dist, 97.5),
                              p_le0=float((dist <= 0).mean())))
        print(bb, sn, boot_rows[-1], flush=True)
bt = pd.DataFrame(boot_rows); bt.to_csv(OUT / 'a5_bootstrap.csv', index=False)
pc = pd.DataFrame(pc_rows); pc.to_csv(OUT / 'a5_perclass_raw.csv', index=False)
s = pc.groupby(['setting', 'label']).mean(numeric_only=True).drop(columns='seed')
cis = []
for (sn, lab), q in pc.groupby(['setting', 'label']):
    m_, lo, hi = paired_ci(q.sort_values('seed').clab.values - q.sort_values('seed').probe.values)
    cis.append(dict(setting=sn, label=lab, d=m_, lo=lo, hi=hi))
s = s.reset_index().merge(pd.DataFrame(cis), on=['setting', 'label'])
s.to_csv(OUT / 'a5_perclass.csv', index=False)
print(s.round(2).to_string())

# Holm over the Table 4 family (4 CLAB-vs-best-adapter contrasts), and over all 12 contrasts
G = pd.read_csv(os.environ.get('CLAB_RESULTS', str(_paths.RESULTS)) + '/main_grid.csv')
Ad = pd.read_csv(os.environ.get('CLAB_RESULTS', str(_paths.RESULTS)) + '/adapters20.csv')
v = lambda s_, m, b_=CLIP: G[(G.setting == s_) & (G.backbone == b_) & (G.method == m)].sort_values('seed').mAP.values
g = Ad.groupby(['setting', 'config']).mAP.mean()
d4 = []
for s_ in ORDER:
    best = g.loc[s_].idxmax(); q = Ad[(Ad.setting == s_) & (Ad.config == best)].sort_values('seed').mAP.values
    d4.append(v(s_, 'clab') - q)
d8 = [v(s_, 'clab', b_) - v(s_, 'probe', b_) for b_ in (CLIP, RC) for s_ in ORDER]
for name, fam in [('table4_only', d4), ('all12', d8 + d4)]:
    res = holm_intervals(fam)
    for i, r in enumerate(res):
        print(name, i, [round(x, 3) for x in r])
pd.DataFrame([dict(family=n, idx=i, mean=r[0], lo=r[1], hi=r[2], holm_lo=r[3], holm_hi=r[4], p=r[5])
              for n, fam in [('table4_only', d4), ('all12', d8 + d4)] for i, r in enumerate(holm_intervals(fam))]
             ).to_csv(OUT / 'a5_holm_table4.csv', index=False)
