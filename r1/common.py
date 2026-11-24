import sys
from pathlib import Path
import numpy as np
import _paths  # noqa: F401
from clab import config as C  # noqa
from clab.core import (F32, EPS, nrm, center, ap, mAP, split_idx, logistic,  # noqa
                       standardize_features, zscore, views, sigmoid)
from clab.data import setting_arrays, seed_split  # noqa
from clab.stats import paired_ci, paired_p, holm_intervals  # noqa

import os
OUT = Path(os.environ.get('R1_OUT', str(_paths.OUT))); OUT.mkdir(parents=True, exist_ok=True)
ORDER = C.ORDER
CLIP, RC = C.HEADLINE, C.DIAGNOSTIC


def seed_parts(Fs_a, Ys_a, Ft_a, Yt_a, seed):
    """Source fit (80%), source held-out (20%, unused by CLAB), target eval (20%), target rest."""
    si, sh = split_idx(len(Fs_a), C.SOURCE_FRAC, seed)
    tr, ti = split_idx(len(Ft_a), C.TARGET_FRAC, seed)
    return si, sh, tr, ti


def branch_scores(Fs, Ys, X, A, Ac, seed, ep=C.EPOCHS, W=None, b=None):
    """Raw branch scores u, h for arbitrary images X with the head fitted on (Fs, Ys)."""
    mu = Fs.mean(0, keepdims=True); sd = Fs.std(0, keepdims=True) + 1e-6
    Zs = ((Fs - mu) / sd).astype(F32)
    if W is None:
        W, b = logistic(Zs, Ys, ep, seed=seed)
    Zx = ((X - mu) / sd).astype(F32)
    return dict(us=nrm(Fs) @ Ac.T, hs=Zs @ W + b, u=nrm(X) @ Ac.T, h=Zx @ W + b,
                rs=nrm(Fs) @ A.T, r=nrm(X) @ A.T, W=W, b=b)


def micro_macro_f1(Y, P):
    tp = (P * Y).sum(0); fp = (P * (1 - Y)).sum(0); fn = ((1 - P) * Y).sum(0)
    cf1 = np.nanmean(np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, EPS), np.nan)) * 100
    of1 = 2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), EPS) * 100
    cp = np.nanmean(np.where(tp + fp > 0, tp / np.maximum(tp + fp, EPS), np.nan)) * 100
    cr = np.nanmean(np.where(tp + fn > 0, tp / np.maximum(tp + fn, EPS), np.nan)) * 100
    return float(cf1), float(of1), float(cp), float(cr)
