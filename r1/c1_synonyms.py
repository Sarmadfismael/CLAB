"""R1-5: source-side synonym expansion of the anchors. Each anchor is the renormalized mean over the
three templates x a fixed synonym list written from general English usage before any evaluation
(no list was changed after seeing results). Reports zero-shot (centered) on whole datasets and CLAB
(alpha 0.30, 50 epochs, twenty seeds) against the standard anchors, paired."""
import os, sys, numpy as np, pandas as pd
import _paths  # noqa: F401
from clab import config as C
from clab.core import center, mAP, nrm, views
from clab.data import setting_arrays, seed_split
from clab.stats import paired_ci
import cliptext

SYN = {"airplane": ["airplane", "aeroplane", "aircraft"],
       "bare-soil": ["bare soil", "bare ground", "barren land"],
       "buildings": ["buildings", "houses"],
       "cars": ["cars", "vehicles", "automobiles"],
       "chaparral": ["chaparral", "shrubland", "scrub"],
       "court": ["court", "sports court", "tennis court"],
       "dock": ["dock", "pier", "wharf"],
       "field": ["field", "farmland", "cropland"],
       "grass": ["grass", "lawn", "grassland"],
       "mobile-home": ["mobile home", "trailer home", "caravan"],
       "pavement": ["pavement", "paved road", "asphalt"],
       "sand": ["sand", "sandy ground"],
       "sea": ["sea", "ocean"],
       "ship": ["ship", "boat", "vessel"],
       "tanks": ["tanks", "storage tanks", "oil tanks"],
       "trees": ["trees", "forest", "woodland"],
       "water": ["water", "lake", "river"]}
OUT = sys.argv[1]
tt = cliptext.TextTower('quick')


def syn_anchors(names):
    import torch, torch.nn.functional as F
    out = []
    with torch.no_grad():
        for c in names:
            e = F.normalize(tt.encode([t.format(s) for s in SYN[c] for t in C.PROMPTS]), dim=-1).mean(0)
            out.append(F.normalize(e, dim=-1).numpy())
    return np.stack(out).astype(np.float32)


rows, zrows = [], []
for sn in C.ORDER:
    Fs_a, Ys_a, Ft_a, Yt_a, A, Ac = setting_arrays(sn, C.HEADLINE)
    names = C.LABELS_17 if C.SETTINGS[sn][2] == 'native' else C.SHARED_6
    S = syn_anchors(names); Sc = center(S)
    Xt = nrm(Ft_a)
    zrows.append(dict(setting=sn, zs_std=mAP(Yt_a, Xt @ Ac.T), zs_syn=mAP(Yt_a, Xt @ Sc.T)))
    for seed in C.SEEDS:
        Fs, Ys, Ft, Yt = seed_split(Fs_a, Ys_a, Ft_a, Yt_a, seed)
        v = views(Fs, Ys, Ft, A, Ac, seed, C.EPOCHS)
        w = views(Fs, Ys, Ft, S, Sc, seed, C.EPOCHS)
        a = C.ALPHA
        rows.append(dict(setting=sn, seed=seed,
                         clab=mAP(Yt, a * v['zU'] + (1 - a) * v['zH']),
                         clab_syn=mAP(Yt, a * w['zU'] + (1 - a) * w['zH']),
                         zs=mAP(Yt, v['zU']), zs_syn=mAP(Yt, w['zU']), probe=mAP(Yt, v['Ht'])))
        if C.SETTINGS[sn][2] != 'native':  # per-class AP on DFC15
            from clab.core import ap
            for nm, sc in [('std', a * v['zU'] + (1 - a) * v['zH']), ('syn', a * w['zU'] + (1 - a) * w['zH'])]:
                for c in range(6):
                    zrows.append(dict(setting=sn, seed=seed, kind=nm, cat=names[c], ap=ap(Yt[:, c], sc[:, c]) * 100))
d = pd.DataFrame(rows); d.to_csv(OUT + 'c1_synonyms.csv', index=False)
pd.DataFrame(zrows).to_csv(OUT + 'c1_synonyms_extra.csv', index=False)
for sn in C.ORDER:
    x = d[d.setting == sn]
    print(sn, 'zs', round(x.zs.mean(), 2), round(x.zs_syn.mean(), 2), ' clab', round(x.clab.mean(), 2),
          round(x.clab_syn.mean(), 2), 'diff', np.round(paired_ci((x.clab_syn - x.clab).values), 2))
print(pd.DataFrame([r for r in zrows if 'zs_std' in r]).round(2))
print('mean clab', d.clab.mean().round(2), 'syn', d.clab_syn.mean().round(2))
