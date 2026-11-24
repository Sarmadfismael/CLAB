"""R2-1 (option 1): published unsupervised domain adaptation methods re-run on the frozen CLIP
ViT-B/32 features, under the paper's splits. Unlike CLAB they consume the unlabeled target images:
the unused 80% of the target (Section 4.2) is their unlabeled target set; evaluation is on the same
20% as every other method.

DDA-MLIC (WACV 2024), ported from the released code (github.com/cvi2snt/DDA-MLIC, main.py and
src/loss_functions/losses.py): linear head, asymmetric loss (gamma_neg 3, gamma_pos 0, clip 0.05),
Adam with OneCycle (max lr 1e-3, 25 epochs, pct_start 0.2), weight decay 1e-4 on weights only,
batch 64 per domain, GMM-based Frechet discrepancy with trade-off 1. The released code keeps the
epoch with the best target mAP; we report that (oracle) and the last epoch. Image augmentation does
not apply to cached features.
DANN (JMLR 2016) with a multi-label head: bottleneck 512-256 (BN, ReLU), linear classifier, domain
discriminator 256-1024-1024-1 behind a warm-start gradient reversal layer, SGD (lr 0.01, momentum
0.9, wd 1e-3, inverse decay), batch 32 per domain, 2000 iterations, trade-off 1; last iterate and
the oracle best checkpoint (every 100 iterations) are reported.

    python c3_uda.py OUTDIR [ddamlic,dann] [check]
"""
import os, sys, math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import _paths  # noqa: F401
from clab import config as C
from clab.core import mAP, split_idx
from clab.data import setting_arrays

torch.set_num_threads(int(os.environ.get('NT', '2')))


def asl(x, y, gn=3, gp=0, clip=0.05, eps=1e-6):
    xp = torch.sigmoid(x); xn = 1 - xp
    xn = (xn + clip).clamp(max=1)
    loss = y * torch.log(xp.clamp(min=eps)) + (1 - y) * torch.log(xn.clamp(min=eps))
    with torch.no_grad():
        pt = xp * y + xn * (1 - y); g = gp * y + gn * (1 - y); w = torch.pow(1 - pt, g)
    return -(loss * w).sum()


def gmm_discrepancy(ys, yt):
    """As released: the GMMs are fitted to detached sigmoid outputs and the Frechet distance is
    computed on tensors built from NumPy, so the term carries no gradient."""
    from sklearn.mixture import GaussianMixture
    from scipy import stats

    def pdf(d):
        gm = GaussianMixture(n_components=2, reg_covar=0.001).fit(d.reshape(-1, 1))
        i0, i1 = np.argmin(gm.means_), np.argmax(gm.means_)
        y1 = stats.norm.pdf(np.linspace(0, .5, 300), gm.means_[i0][0], gm.covariances_[i0][0][0])
        y2 = stats.norm.pdf(np.linspace(.6, 1, 300), gm.means_[i1][0], gm.covariances_[i1][0][0])
        return torch.tensor(y1), torch.tensor(y2)

    def fd(X, Y):
        vx, mx = torch.var_mean(X); vy, my = torch.var_mean(Y)
        return ((mx - my) ** 2 + (vx.sqrt() - vy.sqrt()) ** 2).float()
    s1, s2 = pdf(torch.sigmoid(ys).detach().numpy()); t1, t2 = pdf(torch.sigmoid(yt).detach().numpy())
    return fd(s1, t1) + fd(s2, t2)


def ddamlic(Xs, Ys, Xu, Xe, Ye, seed, check=False):
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    head = nn.Linear(Xs.shape[1], Ys.shape[1])
    params = [{'params': [head.bias], 'weight_decay': 0.}, {'params': [head.weight], 'weight_decay': 1e-4}]
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=0)
    B, E = 64, 25
    ipe = max(math.ceil(len(Xs) / B), math.ceil(len(Xu) / B))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, steps_per_epoch=ipe, epochs=E, pct_start=0.2)
    ps, pt = rng.permutation(len(Xs)), rng.permutation(len(Xu)); i_s = i_t = 0
    hist = []
    for ep in range(E):
        for it in range(ipe):
            if i_s + B > len(ps): ps, i_s = rng.permutation(len(Xs)), 0
            if i_t + B > len(pt): pt, i_t = rng.permutation(len(Xu)), 0
            bs, bt = ps[i_s:i_s + B], pt[i_t:i_t + B]; i_s += B; i_t += B
            y = head(torch.cat([Xs[bs], Xu[bt]]))
            loss = asl(y[:B], Ys[bs])
            if check:
                d = gmm_discrepancy(y[:B], y[B:])
                assert not d.requires_grad, 'discrepancy unexpectedly differentiable'
                loss = loss + d
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        with torch.no_grad():
            hist.append(mAP(Ye, head(Xe).double().numpy()))
    return hist[-1], max(hist), int(np.argmax(hist)) + 1


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, c): ctx.c = c; return x.view_as(x)

    @staticmethod
    def backward(ctx, g): return -ctx.c * g, None


def dann(Xs, Ys, Xu, Xe, Ye, seed, iters=2000, B=32, tradeoff=1.0):
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    C_ = Ys.shape[1]
    bott = nn.Sequential(nn.Linear(Xs.shape[1], 256), nn.BatchNorm1d(256), nn.ReLU())
    cls = nn.Linear(256, C_)
    disc = nn.Sequential(nn.Linear(256, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
                         nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Linear(1024, 1))
    groups = [{'params': list(bott.parameters()) + list(cls.parameters()) + list(disc.parameters())}]
    opt = torch.optim.SGD(groups, lr=0.01, momentum=0.9, weight_decay=1e-3, nesterov=True)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: (1 + 0.001 * i) ** -0.75)
    hist = []
    for i in range(1, iters + 1):
        bs, bt = rng.randint(0, len(Xs), B), rng.randint(0, len(Xu), B)
        bott.train(); cls.train(); disc.train()
        f = bott(torch.cat([Xs[bs], Xu[bt]]))
        y = cls(f[:B])
        coeff = 2.0 / (1 + math.exp(-1.0 * i / 1000)) - 1  # warm start 0 -> 1
        d = disc(GRL.apply(f, coeff)).squeeze(1)
        dl = F.binary_cross_entropy_with_logits(d, torch.cat([torch.ones(B), torch.zeros(B)]))
        loss = F.binary_cross_entropy_with_logits(y, Ys[bs]) + tradeoff * dl
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if i % 100 == 0:
            bott.eval(); cls.eval()
            with torch.no_grad():
                hist.append(mAP(Ye, cls(bott(Xe)).double().numpy()))
    return hist[-1], max(hist), (int(np.argmax(hist)) + 1) * 100


if __name__ == '__main__':
    OUT = sys.argv[1]; kinds = sys.argv[2].split(',') if len(sys.argv) > 2 else ['ddamlic', 'dann']
    check = len(sys.argv) > 3 and sys.argv[3] == 'check'
    seeds = [0] if check else C.SEEDS
    rows = []
    for sn in C.ORDER:
        Fs_a, Ys_a, Ft_a, Yt_a, A, Ac = setting_arrays(sn, C.HEADLINE)
        for seed in seeds:
            si, _ = split_idx(len(Fs_a), C.SOURCE_FRAC, seed)
            tr, ti = split_idx(len(Ft_a), C.TARGET_FRAC, seed)
            mu, sd = Fs_a[si].mean(0, keepdims=True), Fs_a[si].std(0, keepdims=True) + 1e-6
            T = lambda X: torch.from_numpy(((X - mu) / sd).astype(np.float32))
            Xs, Xu, Xe = T(Fs_a[si]), T(Ft_a[tr]), T(Ft_a[ti])
            Ys = torch.from_numpy(Ys_a[si].astype(np.float32)); Ye = Yt_a[ti]
            for k in kinds:
                last, best, at = (ddamlic(Xs, Ys, Xu, Xe, Ye, seed, check) if k == 'ddamlic'
                                  else dann(Xs, Ys, Xu, Xe, Ye, seed))
                rows.append(dict(setting=sn, seed=seed, method=k, last=last, oracle_best=best, best_at=at))
                print(sn, seed, k, round(last, 2), round(best, 2), at, flush=True)
            pd.DataFrame(rows).to_csv(f'{OUT}/c3_uda{"_check" if check else ""}_{"_".join(kinds)}.csv', index=False)
