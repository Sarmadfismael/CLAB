import _paths  # noqa: F401  (repository paths; must come before clab)
import os
"""R3: per-image predictions (seed 0 evaluation split) for stock CLIP zero-shot (raw anchors),
the linear probe and CLAB, with per-class thresholds chosen on held-out SOURCE images.
Also per-class confusion counts summed over the 20 seeds."""
from common import *
from a1_thresholds import per_class_thr  # noqa
import pandas as pd, json

a = C.ALPHA
sel, conf = [], []
for sn in ORDER:
    Fs_a, Ys_a, Ft_a, Yt_a, A, Ac = setting_arrays(sn, CLIP)
    ids_t = np.load(f"{os.environ.get('CLAB_FEATURES', str(_paths.FEAT))}/{CLIP}__{C.SETTINGS[sn][1]}__{C.SETTINGS[sn][2]}.npz", allow_pickle=True)['ids']
    names = C.LABELS_17 if Yt_a.shape[1] == 17 else C.SHARED_6
    for seed in C.SEEDS:
        si, sh, tr, ti = seed_parts(Fs_a, Ys_a, Ft_a, Yt_a, seed)
        b = branch_scores(Fs_a[si], Ys_a[si], np.concatenate([Ft_a[ti], Fs_a[sh]]), A, Ac, seed)
        nt = len(ti)
        zu = lambda M: (M - b['us'].mean()) / (b['us'].std() + EPS)
        zh = lambda M: (M - b['hs'].mean()) / (b['hs'].std() + EPS)
        S = {'clip_zeroshot': b['r'], 'probe': b['h'], 'clab': a * zu(b['u']) + (1 - a) * zh(b['h'])}
        P = {}
        for m, X in S.items():
            th = per_class_thr(X[nt:], Ys_a[sh]); P[m] = (X[:nt] >= th[None, :]).astype(int)
            Yt = Yt_a[ti].astype(int)
            for c, nm in enumerate(names):
                conf.append(dict(setting=sn, seed=seed, method=m, label=nm,
                                 TP=int((P[m][:, c] & Yt[:, c]).sum()), FP=int((P[m][:, c] & (1 - Yt[:, c])).sum()),
                                 FN=int(((1 - P[m][:, c]) & Yt[:, c]).sum()), TN=int(((1 - P[m][:, c]) & (1 - Yt[:, c])).sum())))
        if seed == 0:
            Yt = Yt_a[ti].astype(int)
            f1 = lambda p, y: (2 * (p & y).sum() / max(p.sum() + y.sum(), 1))
            for k, j in enumerate(ti):
                sel.append(dict(setting=sn, image_id=str(ids_t[j]),
                                truth=[n for n, v in zip(names, Yt[k]) if v],
                                clip=[n for n, v in zip(names, P['clip_zeroshot'][k]) if v],
                                probe=[n for n, v in zip(names, P['probe'][k]) if v],
                                clab=[n for n, v in zip(names, P['clab'][k]) if v],
                                f1_clip=f1(P['clip_zeroshot'][k], Yt[k]), f1_probe=f1(P['probe'][k], Yt[k]),
                                f1_clab=f1(P['clab'][k], Yt[k])))
    print(sn, 'done', flush=True)
cf = pd.DataFrame(conf); cf.to_csv(OUT / 'a11_confusion.csv', index=False)
ex = pd.DataFrame(sel); ex.to_json(OUT / 'a11_examples.json', orient='records', indent=1)
for sn in ORDER:
    q = ex[ex.setting == sn]
    print(sn, 'images', len(q), 'CLAB>CLIP', int((q.f1_clab > q.f1_clip).sum()), '=', int((q.f1_clab == q.f1_clip).sum()),
          '<', int((q.f1_clab < q.f1_clip).sum()), '| CLAB>probe', int((q.f1_clab > q.f1_probe).sum()),
          '<', int((q.f1_clab < q.f1_probe).sum()))
