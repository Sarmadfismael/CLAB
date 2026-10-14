"""CLAB: centered CLIP label anchors blended with a short-budget source head.

All computation is NumPy on cached frozen features. The functions are numerically
identical to the code that produced the paper's tables.
"""
from __future__ import annotations

import numpy as np

from . import config as C

F32 = np.float32
EPS = 1e-8


# ------------------------------------------------------------------ basic ops
def nrm(X):
    """Row-wise L2 normalization."""
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), EPS)


def sigmoid(M):
    return 1.0 / (1.0 + np.exp(-np.clip(M, -60, 60)))


def center(A):
    """Equation (center): subtract the anchor mean and renormalize each anchor."""
    Ac = A - A.mean(0, keepdims=True)
    return Ac / np.maximum(np.linalg.norm(Ac, axis=1, keepdims=True), EPS)


# ------------------------------------------------------------------ metrics
def ap(y, s):
    """Average precision of one category, computed on raw scores."""
    o = np.argsort(-s)
    y = y[o]
    c = np.cumsum(y)
    if y.sum() == 0:
        return np.nan
    return float(((c / np.arange(1, len(y) + 1)) * y).sum() / y.sum())


def mAP(Y, S):
    """Mean over categories of average precision, in percent."""
    return float(np.nanmean([ap(Y[:, c], S[:, c]) for c in range(Y.shape[1])]) * 100)


def f1_at_median(Y, S):
    """Secondary metrics: CF1 and OF1 at a fixed median-score threshold."""
    P = (S >= np.median(S)).astype(F32)
    tp = (P * Y).sum(0)
    cf1 = np.nanmean(2 * tp / np.maximum(P.sum(0) + Y.sum(0), EPS)) * 100
    of1 = 2 * tp.sum() / max(P.sum() + Y.sum(), EPS) * 100
    return float(cf1), float(of1)


# ------------------------------------------------------------------ splits
def split_idx(n, frac, seed):
    """Random permutation of range(n) drawn from RandomState(seed), cut at round(frac*n).

    The source head uses the first part of the source permutation; evaluation uses
    the second part of the target permutation.
    """
    i = np.random.RandomState(seed).permutation(n)
    k = int(round(frac * n))
    return i[:k], i[k:]


# ------------------------------------------------------------------ source head
def logistic(X, Y, epochs, lr=C.LR, wd=C.WEIGHT_DECAY, seed=0, dtype=F32):
    """Linear multi-label head, full-batch gradient descent on binary cross-entropy."""
    r = np.random.RandomState(seed)
    W = r.normal(0, C.INIT_SD, (X.shape[1], Y.shape[1])).astype(dtype)
    b = np.zeros(Y.shape[1], dtype)
    for _ in range(epochs):
        p = sigmoid(X @ W + b)
        g = ((p - Y) / len(X)).astype(dtype)
        W -= lr * (X.T @ g + wd * W)
        b -= lr * g.sum(0)
    return W, b


def standardize_features(Fs, Ft):
    """Per-dimension standardization with SOURCE statistics only."""
    mu = Fs.mean(0, keepdims=True)
    sd = Fs.std(0, keepdims=True) + 1e-6
    return ((Fs - mu) / sd).astype(F32), ((Ft - mu) / sd).astype(F32)


def zscore(M, ref):
    """The Z operator: one location and one scale, both estimated on `ref`."""
    return (M - ref.mean()) / (ref.std() + EPS)


# ------------------------------------------------------------------ the method
def views(Fs, Ys, Ft, A, Ac, seed, epochs=C.EPOCHS):
    """Both branches on source (s) and target (t), plus their source-standardized forms.

    Fs, Ys : source features / labels (the fitted 80%)
    Ft     : target features (labels are never passed in)
    A, Ac  : raw and centered anchors
    """
    Zs, Zt = standardize_features(Fs, Ft)
    W, b = logistic(Zs, Ys, epochs, seed=seed)
    Us, Ut = nrm(Fs) @ Ac.T, nrm(Ft) @ Ac.T          # semantic branch, centered anchors
    Rs, Rt = nrm(Fs) @ A.T, nrm(Ft) @ A.T            # semantic branch, raw anchors
    Hs, Ht = Zs @ W + b, Zt @ W + b                  # source head
    return dict(Us=Us, Ut=Ut, Rs=Rs, Rt=Rt, Hs=Hs, Ht=Ht,
                zU=zscore(Ut, Us), zR=zscore(Rt, Rs), zH=zscore(Ht, Hs))


def clab_scores(v, alpha=C.ALPHA):
    """Equation (blend): s = alpha * Z[u] + (1 - alpha) * Z[h], constants from source."""
    return alpha * v["zU"] + (1 - alpha) * v["zH"]


class CLAB:
    """Convenience wrapper for using the method outside the experiment scripts.

    >>> m = CLAB().fit(F_source, Y_source, anchors)
    >>> scores = m.score(F_target)      # one row per target tile, one column per class
    """

    def __init__(self, alpha=C.ALPHA, epochs=C.EPOCHS, seed=0):
        self.alpha, self.epochs, self.seed = alpha, epochs, seed

    def fit(self, Fs, Ys, A):
        Fs = Fs.astype(F32)
        self.Ac = center(A.astype(F32))
        self.mu = Fs.mean(0, keepdims=True)
        self.sd = Fs.std(0, keepdims=True) + 1e-6
        Zs = ((Fs - self.mu) / self.sd).astype(F32)
        self.W, self.b = logistic(Zs, Ys.astype(F32), self.epochs, seed=self.seed)
        Us, Hs = nrm(Fs) @ self.Ac.T, Zs @ self.W + self.b
        self.u_loc, self.u_scale = Us.mean(), Us.std() + EPS
        self.h_loc, self.h_scale = Hs.mean(), Hs.std() + EPS
        return self

    def score(self, Ft):
        Ft = Ft.astype(F32)
        Ut = nrm(Ft) @ self.Ac.T
        Ht = ((Ft - self.mu) / self.sd).astype(F32) @ self.W + self.b
        return (self.alpha * (Ut - self.u_loc) / self.u_scale
                + (1 - self.alpha) * (Ht - self.h_loc) / self.h_scale)
