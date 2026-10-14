"""Table 4: CLAB against oracle-tuned CLIP-Adapter and Tip-Adapter on identical frozen
CLIP ViT-B/32 features (and the Tip-Adapter centering gains quoted in Section 5.3).

Stage 1 sweeps each baseline's hyperparameters on split seeds 0-2 and keeps, for every
(transfer, family, anchor choice), the configuration with the highest TARGET mAP. That
selection uses target labels and is impossible in deployment; it is made deliberately so
that the baselines get the most favourable treatment available.
Stage 2 re-runs the selected configurations on all 20 seeds for paired comparison.

    python experiments/e03_adapters.py
        -> results/adapters_sweep.csv, results/adapters20.csv          (~20-40 min)

Our adaptation for multi-label data: the CLIP-Adapter classifier is trained with a
per-class sigmoid loss, and the Tip-Adapter cache values are the multi-hot label
vectors of the cached source images.
"""
import _common  # noqa: F401
import numpy as np
import pandas as pd

from clab import config as C
from clab.core import F32, EPS, mAP, nrm, sigmoid
from clab.data import out_path, seed_split, setting_arrays

SWEEP_SEEDS = [0, 1, 2]
zsc = lambda M: (M - M.mean()) / (M.std() + EPS)


def _adam_step(params, grads, state, lr, wd=1e-4, b1=0.9, b2=0.999, eps=1e-8):
    state["t"] += 1
    t = state["t"]
    for i, (p, g) in enumerate(zip(params, grads)):
        g = g + wd * p
        state["m"][i] = b1 * state["m"][i] + (1 - b1) * g
        state["v"][i] = b2 * state["v"][i] + (1 - b2) * g * g
        mh = state["m"][i] / (1 - b1 ** t)
        vh = state["v"][i] / (1 - b2 ** t)
        p -= lr * mh / (np.sqrt(vh) + eps)


def clip_adapter_scores(Fs, Ys, Ft, Anc, seed, ratio, lr, epochs, hidden=4):
    """CLIP-Adapter: f' = ratio * MLP(f) + (1 - ratio) * f, frozen text classifier,
    Adam as in the original, sigmoid loss on logit scale 100."""
    rng = np.random.RandomState(seed)
    d = Fs.shape[1]; h = max(d // hidden, 8)
    W1 = rng.normal(0, (2.0 / d) ** .5, (d, h)).astype(F32); b1_ = np.zeros(h, F32)
    W2 = rng.normal(0, (2.0 / h) ** .5, (h, d)).astype(F32); b2_ = np.zeros(d, F32)
    params = [W1, b1_, W2, b2_]
    state = {"t": 0, "m": [np.zeros_like(p) for p in params],
             "v": [np.zeros_like(p) for p in params]}
    Xs, Xt, At = nrm(Fs), nrm(Ft), Anc.T.astype(F32)
    Ancf = Anc.astype(F32)
    for _ in range(epochs):
        H = Xs @ W1 + b1_; Hr = np.maximum(H, 0)
        O = Hr @ W2 + b2_; Or_ = np.maximum(O, 0)
        R = ratio * Or_ + (1 - ratio) * Xs
        Fn_ = nrm(R)
        G = ((sigmoid(100.0 * (Fn_ @ At)) - Ys) / len(Xs) * 100.0).astype(F32)
        dF = G @ Ancf
        nrmR = np.maximum(np.linalg.norm(R, axis=1, keepdims=True), EPS)
        dR = (dF - (dF * Fn_).sum(1, keepdims=True) * Fn_) / nrmR
        dO = dR * ratio * (O > 0)
        dH = (dO @ W2.T) * (H > 0)
        _adam_step(params, [Xs.T @ dH, dH.sum(0), Hr.T @ dO, dO.sum(0)], state, lr)
        if not np.isfinite(W1).all():
            return None
    H = np.maximum(Xt @ W1 + b1_, 0); O = np.maximum(H @ W2 + b2_, 0)
    return 100.0 * (nrm(ratio * O + (1 - ratio) * Xt) @ At)


def tip_scores(Fs, Ys, Ft, Anc, alpha, beta, fuse, shots=None, seed=0):
    """Training-free Tip-Adapter with a multi-hot cache."""
    Xs, Ys_ = nrm(Fs), Ys
    if shots:                                   # few-shot cache, as originally designed
        rng = np.random.RandomState(seed); keep = []
        for c in range(Ys.shape[1]):
            pos = np.where(Ys[:, c] > 0)[0]
            if len(pos):
                keep += list(rng.choice(pos, min(shots, len(pos)), replace=False))
        keep = np.unique(keep)
        Xs, Ys_ = Xs[keep], Ys[keep]
    Q = nrm(Ft)
    cache = np.exp(-beta * (1.0 - Q @ Xs.T)) @ Ys_
    clip_logits = 100.0 * (Q @ Anc.T)
    if fuse == "z":
        return alpha * zsc(clip_logits) + (1 - alpha) * zsc(cache)
    return clip_logits + alpha * cache


CA = [("clip_adapter", anch, ratio, lr, 200)
      for anch in ("raw", "centred") for ratio in (0.2, 0.5) for lr in (0.001, 0.003, 0.01)]
TP = [("tip_adapter", anch, al, be, fu, sh)
      for anch in ("raw", "centred")
      for fu, al in [("default", 0.5), ("default", 1.0), ("default", 2.0),
                     ("z", 0.2), ("z", 0.4), ("z", 0.6), ("z", 0.8)]
      for be in (1.0, 5.5) for sh in (None, 16)]


def run_cfg(cfg, Fs, Ys, Ft, A, Ac, seed):
    anc = Ac if cfg[1] == "centred" else A
    if cfg[0] == "clip_adapter":
        _, _, ratio, lr, ep = cfg
        return clip_adapter_scores(Fs, Ys, Ft, anc, seed, float(ratio), float(lr), int(ep))
    _, _, al, be, fu, sh = cfg
    sh = None if sh in (None, "None") else int(sh)
    return tip_scores(Fs, Ys, Ft, anc, float(al), float(be), fu, sh, seed)


def main():
    data = {sn: setting_arrays(sn, C.HEADLINE) for sn in C.ORDER}
    sweep = []
    for sn in C.ORDER:
        Fs_a, Ys_a, Ft_a, Yt_a, A, Ac = data[sn]
        for cfg in CA + TP:
            vals = []
            for seed in SWEEP_SEEDS:
                Fs, Ys, Ft, Yt = seed_split(Fs_a, Ys_a, Ft_a, Yt_a, seed)
                S = run_cfg(cfg, Fs, Ys, Ft, A, Ac, seed)
                vals.append(np.nan if S is None else mAP(Yt, S))
            sweep.append(dict(setting=sn, family=cfg[0], anchors=cfg[1],
                              config="|".join(str(x) for x in cfg),
                              mAP=float(np.nanmean(vals))))
        print(f"  sweep {sn:12s} done", flush=True)
    sw = pd.DataFrame(sweep)
    sw.to_csv(out_path("adapters_sweep.csv"), index=False)

    rows = []
    for (sn, fam, anch), q in sw.groupby(["setting", "family", "anchors"]):
        cfg = q.loc[q.mAP.idxmax()].config
        parts = cfg.split("|")
        Fs_a, Ys_a, Ft_a, Yt_a, A, Ac = data[sn]
        for seed in C.SEEDS:
            Fs, Ys, Ft, Yt = seed_split(Fs_a, Ys_a, Ft_a, Yt_a, seed)
            S = run_cfg(parts, Fs, Ys, Ft, A, Ac, seed)
            rows.append(dict(setting=sn, family=fam, anchors=anch, config=cfg, seed=seed,
                             mAP=np.nan if S is None else mAP(Yt, S)))
        print(f"  20 seeds {sn:12s} {fam:13s} {anch:8s} done", flush=True)
    pd.DataFrame(rows).to_csv(out_path("adapters20.csv"), index=False)
    print("wrote results/adapters_sweep.csv, results/adapters20.csv")


if __name__ == "__main__":
    main()
