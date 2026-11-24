"""R2-5a: multi-label prompt learning on the frozen CLIP ViT-B/32 features.

CoOp-sigmoid: one context of M learnable token vectors shared by all categories,
  "[ctx] <name>.", logits 100*cos(z, t_c) + b_c, per-category sigmoid (binary cross-entropy).
DualCoOp-global: a positive and a negative context; p_c = sigmoid(100*(cos(z,t+_c) - cos(z,t-_c))),
  the two-prompt formulation of DualCoOp applied to the global image feature (no region features
  are available on cached features).
Both are trained full-batch with Adam on the source fit set; the text encoder is frozen and gradients
flow through it to the context vectors. Like the adapters (e03), the hyperparameters are swept on
seeds 0-2 (two context designs x two learning rates x 50/100/200 Adam steps, raw or centered text features)
and the configuration with the best TARGET mAP is kept (oracle selection), then rerun on
all twenty seeds; evaluation uses the learned text features raw and mean-centered.

    python c2_promptlearn.py sweep|final OUTDIR [settings]
"""
import os, sys, time, json, numpy as np, pandas as pd, torch, torch.nn.functional as F
import _paths  # noqa: F401
from clab import config as C
from clab.core import mAP, nrm
from clab.data import setting_arrays, seed_split
import cliptext

torch.set_num_threads(int(os.environ.get('NT', '2')))
tt = cliptext.TextTower('quick')
CHECK = [int(x) for x in os.environ.get('CHECK', '50,100,200').split(',')]
GRID = [dict(M=M, init=i, lr=lr) for (M, i) in ((4, 'an aerial photograph containing'), (16, None)) for lr in (2e-3, 1e-2)]


def ctx_param(M, init, seed):
    g = torch.Generator().manual_seed(seed)
    if init:
        c = tt.ctx_init(init)
        assert c.shape[0] == M, (c.shape, M)
        return c.clone().requires_grad_(True)
    return (torch.randn(M, 512, generator=g) * 0.02).requires_grad_(True)


def text_feats(ctx, names):
    return F.normalize(cliptext.encode_ctx_fast(tt, ctx, [n.replace('-', ' ') for n in names]), dim=-1)


def evaluate(T, Xt, Yt):
    T = T.detach().numpy().astype(np.float64)
    Tc = T - T.mean(0, keepdims=True); Tc /= np.linalg.norm(Tc, axis=1, keepdims=True)
    return mAP(Yt, Xt @ T.T), mAP(Yt, Xt @ Tc.T)


def run(kind, cfg, Fs, Ys, Ft, Yt, names, seed, checks):
    Xs = torch.from_numpy(nrm(Fs).astype(np.float32)); ys = torch.from_numpy(Ys.astype(np.float32))
    Xt = nrm(Ft).astype(np.float64)
    torch.manual_seed(seed)
    if kind == 'coop':
        ctx = ctx_param(cfg['M'], cfg['init'], seed); b = torch.zeros(len(names), requires_grad=True)
        params = [ctx, b]
    else:
        cp = ctx_param(cfg['M'], cfg['init'], seed); cn = ctx_param(cfg['M'], None, seed + 1000)
        params = [cp, cn]
    opt = torch.optim.Adam(params, lr=cfg['lr'])
    out = {}
    for step in range(1, max(checks) + 1):
        opt.zero_grad()
        if kind == 'coop':
            T = text_feats(ctx, names); logit = 100 * Xs @ T.T + b
        else:
            Tp, Tn = text_feats(cp, names), text_feats(cn, names); logit = 100 * (Xs @ Tp.T - Xs @ Tn.T)
        F.binary_cross_entropy_with_logits(logit, ys).backward(); opt.step()
        if step in checks:
            with torch.no_grad():
                if kind == 'coop':
                    out[step] = evaluate(text_feats(ctx, names), Xt, Yt)
                else:  # rank by cos+ - cos-; the 'centered' variant centers both text sets
                    Tp, Tn = text_feats(cp, names).numpy().astype(np.float64), text_feats(cn, names).numpy().astype(np.float64)
                    cen = lambda T: (lambda D: D / np.linalg.norm(D, axis=1, keepdims=True))(T - T.mean(0, keepdims=True))
                    out[step] = (mAP(Yt, Xt @ Tp.T - Xt @ Tn.T), mAP(Yt, Xt @ cen(Tp).T - Xt @ cen(Tn).T))
    return out


def names_for(sn):
    return C.LABELS_17 if C.SETTINGS[sn][2] == 'native' else C.SHARED_6


if __name__ == '__main__':
    mode, OUT = sys.argv[1], sys.argv[2]
    sets = sys.argv[3].split(',') if len(sys.argv) > 3 else C.ORDER
    kinds = os.environ.get('KINDS', 'coop,dual').split(',')
    if mode == 'sweep':
        rows = []
        for sn in sets:
            Fs_a, Ys_a, Ft_a, Yt_a, A, Ac = setting_arrays(sn, C.HEADLINE)
            for kind in kinds:
                for gi, cfg in enumerate(GRID):
                    if os.environ.get('GRIDSEL') and str(gi) not in os.environ['GRIDSEL'].split(','): continue
                    for seed in (0, 1, 2):
                        Fs, Ys, Ft, Yt = seed_split(Fs_a, Ys_a, Ft_a, Yt_a, seed)
                        t = time.time(); r = run(kind, cfg, Fs, Ys, Ft, Yt, names_for(sn), seed, CHECK)
                        for st, (raw, cen) in r.items():
                            rows.append(dict(setting=sn, kind=kind, grid=gi, **cfg, steps=st, seed=seed, mAP_raw=raw, mAP_centered=cen))
                        print(sn, kind, gi, seed, f'{time.time()-t:.0f}s', {k: np.round(v, 2) for k, v in r.items()}, flush=True)
                        pd.DataFrame(rows).to_csv(f'{OUT}/c2_sweep_{"_".join(sets)}_{"_".join(kinds)}{os.environ.get("TAG", "")}.csv', index=False)
    else:
        sw = pd.concat([pd.read_csv(f'{OUT}/{f}') for f in os.listdir(OUT) if f.startswith('c2_sweep_')])
        L = sw.melt(id_vars=['setting', 'kind', 'grid', 'steps', 'seed'], value_vars=['mAP_raw', 'mAP_centered'],
                    var_name='anchors', value_name='mAP')
        best = L.groupby(['setting', 'kind', 'grid', 'steps', 'anchors'])['mAP'].mean().reset_index()
        best = best.loc[best.groupby(['setting', 'kind'])['mAP'].idxmax()]
        print(best)
        rows = []
        for sn in sets:
            Fs_a, Ys_a, Ft_a, Yt_a, A, Ac = setting_arrays(sn, C.HEADLINE)
            for kind in kinds:
                bb = best[(best.setting == sn) & (best.kind == kind)].iloc[0]
                cfg = GRID[int(bb.grid)]
                for seed in [int(x) for x in os.environ.get('SEEDS', ','.join(map(str, C.SEEDS))).split(',')]:
                    Fs, Ys, Ft, Yt = seed_split(Fs_a, Ys_a, Ft_a, Yt_a, seed)
                    r = run(kind, cfg, Fs, Ys, Ft, Yt, names_for(sn), seed, [int(bb.steps)])[int(bb.steps)]
                    rows.append(dict(setting=sn, kind=kind, seed=seed, grid=int(bb.grid), steps=int(bb.steps),
                                     anchors=bb.anchors, mAP=r[0] if bb.anchors == 'mAP_raw' else r[1]))
                    print(sn, kind, seed, round(rows[-1]['mAP'], 2), flush=True)
                    pd.DataFrame(rows).to_csv(f'{OUT}/c2_final_{"_".join(sets)}_{"_".join(kinds)}.csv', index=False)
