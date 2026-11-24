import _paths  # noqa: F401  (repository paths; must come before clab)
import os
"""R2.3: is mean removal of the anchors the right correction? Alternatives on anchors and images."""
from common import *
import pandas as pd


def lowdin(M):
    """Symmetric orthonormalization of the rows of M (whitening within the anchor span)."""
    U, s, Vt = np.linalg.svd(M.astype(np.float64), full_matrices=False)
    keep = s > 1e-6 * s.max()
    return (U[:, keep] @ Vt[keep]).astype(F32)


def remove_pcs(M, k, about_mean=True):
    X = M.astype(np.float64)
    mu = X.mean(0, keepdims=True) if about_mean else 0 * X[:1]
    Xc = X - mu
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Vt[:k]
    Y = Xc - (Xc @ P.T) @ P
    return (Y / np.linalg.norm(Y, axis=1, keepdims=True)).astype(F32)


def variants(A):
    return {
        'raw': A,
        'centered (CLAB)': center(A),
        'raw, remove PC1 about origin': remove_pcs(A, 1, about_mean=False),
        'centered + remove PC1 (all-but-the-top, D=1)': remove_pcs(A, 1),
        'centered + remove PC1-2 (D=2)': remove_pcs(A, 2),
        'centered + remove PC1-3 (D=3)': remove_pcs(A, 3),
        'centered + whitening (Lowdin)': lowdin(center(A)),
        'raw + whitening (Lowdin)': lowdin(A),
    }


rows = []
a = C.ALPHA
for sn in ORDER:
    Fs_a, Ys_a, Ft_a, Yt_a, A, Ac = setting_arrays(sn, CLIP)
    V = variants(A)
    for seed in C.SEEDS:
        si, sh, tr, ti = seed_parts(Fs_a, Ys_a, Ft_a, Yt_a, seed)
        Fs, Ys, Ft, Yt = Fs_a[si], Ys_a[si], Ft_a[ti], Yt_a[ti]
        b = branch_scores(Fs, Ys, Ft, A, Ac, seed)
        zH = zscore(b['h'], b['hs'])
        ms = nrm(Fs).mean(0, keepdims=True)             # SOURCE image mean (inductive)
        img_c = lambda X: nrm(nrm(X) - ms)
        cases = {k: (nrm(Fs) @ T.T, nrm(Ft) @ T.T) for k, T in V.items()}
        cases['image centering (source mean), raw anchors'] = (img_c(Fs) @ A.T, img_c(Ft) @ A.T)
        cases['image centering + centered anchors'] = (img_c(Fs) @ Ac.T, img_c(Ft) @ Ac.T)
        for k, (Us, Ut) in cases.items():
            rows.append(dict(setting=sn, seed=seed, variant=k, zeroshot=mAP(Yt, Ut),
                             clab=mAP(Yt, a * zscore(Ut, Us) + (1 - a) * zH)))
    print(sn, 'done', flush=True)
d = pd.DataFrame(rows); d.to_csv(OUT / 'a6_centering.csv', index=False)
g = d.groupby(['variant', 'setting'])[['zeroshot', 'clab']].mean().unstack('setting')
g[('zeroshot', 'mean')] = d.groupby('variant').zeroshot.mean(); g[('clab', 'mean')] = d.groupby('variant').clab.mean()
print(g.round(2).to_string())
for k in d.variant.unique():
    if k == 'centered (CLAB)': continue
    for col in ['zeroshot', 'clab']:
        diffs = [d[(d.setting == s) & (d.variant == k)].sort_values('seed')[col].values -
                 d[(d.setting == s) & (d.variant == 'centered (CLAB)')].sort_values('seed')[col].values for s in ORDER]
        print(f"{k:48s} {col:8s} vs centered: " + " ".join(f"{np.mean(x):+.2f}" for x in diffs))

# Table 11: explained variance of the raw anchors about their mean as well as about the origin
A = np.load(os.environ.get('CLAB_FEATURES', str(_paths.FEAT)) + '/clip_vitb32_control__TEXT__native.npy').astype(np.float64)
for lab, X in [('raw about origin', A), ('raw about mean', A - A.mean(0)), ('centered about mean', center(A) - center(A).mean(0))]:
    s = np.linalg.svd(X, compute_uv=False); ev = s ** 2 / (s ** 2).sum()
    print(lab, np.round(100 * ev[:3], 1), round(100 * ev[:3].sum(), 1))
