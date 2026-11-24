"""R1.2 (ridge / strong L2 at convergence), R1.3 (per-class standardization),
R1.4 (evaluation on 100% of the target dataset)."""
import _paths  # noqa: F401  (repository paths; must come before clab)
from common import *
import pandas as pd

LAMS = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
WDS = [1e-4, 1e-2, 1e-1, 1.0]
LONG = 3000


def ridge(Z, Y, lam):
    ym = Y.mean(0, keepdims=True)
    G = Z.T @ Z + lam * np.eye(Z.shape[1], dtype=np.float64)
    W = np.linalg.solve(G, Z.T.astype(np.float64) @ (Y - ym))
    return W.astype(F32), ym.ravel().astype(F32)


rows, rows_pc, rows_full = [], [], []
a = C.ALPHA
for sn in ORDER:
    Fs_a, Ys_a, Ft_a, Yt_a, A, Ac = setting_arrays(sn, CLIP)
    for seed in C.SEEDS:
        si, sh, tr, ti = seed_parts(Fs_a, Ys_a, Ft_a, Yt_a, seed)
        Fs, Ys, Ft, Yt = Fs_a[si], Ys_a[si], Ft_a[ti], Yt_a[ti]
        mu = Fs.mean(0, keepdims=True); sd = Fs.std(0, keepdims=True) + 1e-6
        Zs = ((Fs - mu) / sd).astype(F32); Zt = ((Ft - mu) / sd).astype(F32)
        Us, Ut = nrm(Fs) @ Ac.T, nrm(Ft) @ Ac.T
        zU = zscore(Ut, Us)

        def rec(kind, par, Hs, Ht):
            rows.append(dict(setting=sn, seed=seed, head=kind, param=par, probe=mAP(Yt, Ht),
                             clab=mAP(Yt, a * zU + (1 - a) * zscore(Ht, Hs))))
        W, b = logistic(Zs, Ys, 50, seed=seed); rec('gd50', 1e-4, Zs @ W + b, Zt @ W + b)
        for wd in WDS:
            W, b = logistic(Zs, Ys, LONG, wd=wd, seed=seed); rec(f'gd{LONG}', wd, Zs @ W + b, Zt @ W + b)
        for lam in LAMS:
            W, b = ridge(Zs, Ys, lam); rec('ridge', lam, Zs @ W + b, Zt @ W + b)

        # R1.3 per-class standardization (constants per category, estimated on source)
        W, b = logistic(Zs, Ys, 50, seed=seed); Hs, Ht = Zs @ W + b, Zt @ W + b
        pz = lambda M, R: (M - R.mean(0, keepdims=True)) / (R.std(0, keepdims=True) + EPS)
        rows_pc.append(dict(setting=sn, seed=seed, probe=mAP(Yt, Ht),
                            clab_global=mAP(Yt, a * zU + (1 - a) * zscore(Ht, Hs)),
                            clab_perclass=mAP(Yt, a * pz(Ut, Us) + (1 - a) * pz(Ht, Hs)),
                            rho_spread=float((Hs.std(0) / Us.std(0)).std() / (Hs.std(0) / Us.std(0)).mean())))

        # R1.4 evaluation on the whole target dataset (all methods are source-only)
        Zf = ((Ft_a - mu) / sd).astype(F32); Uf = nrm(Ft_a) @ Ac.T
        Hf = Zf @ W + b
        rows_full.append(dict(setting=sn, seed=seed, probe_20=mAP(Yt, Ht),
                              clab_20=mAP(Yt, a * zU + (1 - a) * zscore(Ht, Hs)),
                              probe_100=mAP(Yt_a, Hf),
                              clab_100=mAP(Yt_a, a * zscore(Uf, Us) + (1 - a) * zscore(Hf, Hs)),
                              zeroshot_100=mAP(Yt_a, Uf)))
    print(sn, 'done', flush=True)

d = pd.DataFrame(rows); d.to_csv(OUT / 'a2_heads.csv', index=False)
print(d.groupby(['head', 'param', 'setting'])[['probe', 'clab']].mean().unstack('setting').round(2).to_string())
g = d.groupby(['head', 'param'])[['probe', 'clab']].mean(); print(g.round(2))
p = pd.DataFrame(rows_pc); p.to_csv(OUT / 'a3_perclass_z.csv', index=False)
print(p.groupby('setting').mean(numeric_only=True).round(3))
for sn in ORDER:
    q = p[p.setting == sn]; print(sn, 'perclass-global', [round(x, 2) for x in paired_ci(q.clab_perclass - q.clab_global)])
f = pd.DataFrame(rows_full); f.to_csv(OUT / 'a4_full_target.csv', index=False)
print(f.groupby('setting').agg(['mean', 'std']).round(2).to_string())
for sn in ORDER:
    q = f[f.setting == sn]; print(sn, 'full: clab-probe', [round(x, 2) for x in paired_ci(q.clab_100 - q.probe_100)])
