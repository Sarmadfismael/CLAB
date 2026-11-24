"""R1.1: hard label assignment with thresholds set from source data only."""
import _paths  # noqa: F401  (repository paths; must come before clab)
from common import *
import pandas as pd


def per_class_thr(S, Y):
    """Per-class threshold maximizing F1 on (S, Y); candidates are the observed scores."""
    th = np.zeros(S.shape[1])
    for c in range(S.shape[1]):
        s, y = S[:, c], Y[:, c]
        o = np.argsort(-s); ss, yy = s[o], y[o]
        tp = np.cumsum(yy); k = np.arange(1, len(s) + 1)
        f1 = 2 * tp / (k + yy.sum())
        if yy.sum() == 0:
            th[c] = np.inf; continue
        i = int(np.argmax(f1)); th[c] = ss[i]
    return th


def card_thr(S, Y):
    """Single global threshold so that mean predicted cardinality equals mean true cardinality."""
    k = Y.sum(1).mean()
    flat = np.sort(S.ravel())[::-1]
    n = int(round(k * len(S)))
    return flat[max(n - 1, 0)]


rows = []
for sn in ORDER:
    Fs_a, Ys_a, Ft_a, Yt_a, A, Ac = setting_arrays(sn, CLIP)
    for seed in C.SEEDS:
        si, sh, tr, ti = seed_parts(Fs_a, Ys_a, Ft_a, Yt_a, seed)
        Fs, Ys = Fs_a[si], Ys_a[si]
        b = branch_scores(Fs, Ys, np.concatenate([Ft_a[ti], Fs_a[sh]]), A, Ac, seed)
        nt = len(ti)
        zu = lambda M: (M - b['us'].mean()) / (b['us'].std() + EPS)
        zh = lambda M: (M - b['hs'].mean()) / (b['hs'].std() + EPS)
        a = C.ALPHA
        meth = {
            'zeroshot': (zu(b['u']), zu(b['us'])),
            'probe': (b['h'], b['hs']),
            'clab': (a * zu(b['u']) + (1 - a) * zh(b['h']), a * zu(b['us']) + (1 - a) * zh(b['hs'])),
        }
        Yt, Yh = Yt_a[ti], Ys_a[sh]
        for m, (Sx, Ssrc) in meth.items():
            St, Sh = Sx[:nt], Sx[nt:]
            schemes = {
                'perclass_srcval': St >= per_class_thr(Sh, Yh)[None, :],
                'cardinality_src': St >= card_thr(Ssrc, Ys),
                'median_target(transductive)': St >= np.median(St),
            }
            if m == 'probe':
                schemes['sigmoid_0.5'] = sigmoid(St) >= 0.5
            for sc, P in schemes.items():
                cf1, of1, cp, cr = micro_macro_f1(Yt, P.astype(float))
                rows.append(dict(setting=sn, seed=seed, method=m, scheme=sc, CF1=cf1, OF1=of1,
                                 CP=cp, CR=cr, card=float(P.sum(1).mean()), true_card=float(Yt.sum(1).mean())))
    print(sn, 'done', flush=True)
d = pd.DataFrame(rows); d.to_csv(OUT / 'a1_thresholds.csv', index=False)
g = d.groupby(['scheme', 'method', 'setting'])[['CF1', 'OF1']].mean().unstack('setting')
print(g.round(2).to_string())
for sc in ['perclass_srcval', 'cardinality_src']:
    for met in ['CF1', 'OF1']:
        for sn in ORDER:
            q = d[(d.scheme == sc) & (d.setting == sn)].pivot(index='seed', columns='method', values=met)
            m_, lo, hi = paired_ci(q.clab - q.probe)
            print(sc, met, sn, f"{m_:+.2f} [{lo:+.2f},{hi:+.2f}]")
