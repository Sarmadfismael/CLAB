"""R2.4: leave-one-transfer-out over six transfers. The two extra transfers use DFC15 as the
source (DFC15->AID, DFC15->UCM, shared-six space, empty images removed in both domains as in the
main protocol). They played no part in choosing alpha or the budget."""
import _paths  # noqa: F401  (repository paths; must come before clab)
from common import *
import pandas as pd

EXTRA = {'DFC15->AID': ('dfc15', 'aid', 'shared6'), 'DFC15->UCM': ('dfc15', 'ucm', 'shared6')}
AG = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
BG = [20, 50, 100, 200]
rows = []
for sn in ORDER + list(EXTRA):
    if sn in EXTRA:
        s, t, sp = EXTRA[sn]
        Fs_a, Ys_a, Ft_a, Yt_a, A, Ac = setting_arrays((s, t), CLIP, space=sp)
    else:
        Fs_a, Ys_a, Ft_a, Yt_a, A, Ac = setting_arrays(sn, CLIP)
    for seed in C.SEEDS:
        Fs, Ys, Ft, Yt = seed_split(Fs_a, Ys_a, Ft_a, Yt_a, seed)
        for ep in BG:
            v = views(Fs, Ys, Ft, A, Ac, seed, ep)
            lin = mAP(Yt, v['Ht'])
            for al in AG:
                rows.append(dict(setting=sn, seed=seed, epochs=ep, alpha=al, linear=lin,
                                 mAP=mAP(Yt, al * v['zU'] + (1 - al) * v['zH']),
                                 zeroshot=mAP(Yt, v['zU'])))
    print(sn, 'done', flush=True)
d = pd.DataFrame(rows); d.to_csv(OUT / 'a9_loto6_raw.csv', index=False)
g = d.groupby(['setting', 'epochs', 'alpha']).mAP.mean().reset_index()
S = list(d.setting.unique()); sel = []
for held in S:
    ep, al = g[g.setting != held].groupby(['epochs', 'alpha']).mAP.mean().idxmax()
    q = d[(d.setting == held) & (d.epochs == ep) & np.isclose(d.alpha, al)].sort_values('seed')
    m_, lo, hi = paired_ci(q.mAP.values - q.linear.values)
    sel.append(dict(held_out=held, alpha=al, epochs=int(ep), clab=q.mAP.mean(), probe=q.linear.mean(),
                    zeroshot=q.zeroshot.mean(), delta=m_, lo=lo, hi=hi))
s = pd.DataFrame(sel); s.to_csv(OUT / 'a9_loto6_selected.csv', index=False); print(s.round(2).to_string())
q = d[(d.epochs == 50) & np.isclose(d.alpha, 0.3)]
for sn in EXTRA:
    x = q[q.setting == sn].sort_values('seed'); m_, lo, hi = paired_ci(x.mAP.values - x.linear.values)
    print(sn, 'fixed recipe', round(x.mAP.mean(), 2), round(x.linear.mean(), 2), round(x.zeroshot.mean(), 2), f"{m_:+.2f} [{lo:+.2f},{hi:+.2f}]")
