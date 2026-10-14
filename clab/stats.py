"""Paired t-intervals and Holm-Bonferroni intervals, without SciPy."""
from __future__ import annotations

import math

import numpy as np


def _bcf(a, b, x):
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((a - 1.0 + m2) * (a + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1.0 + aa / (c if abs(c) > 1e-30 else 1e-30); h *= d * c
        aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + 1.0 + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1.0 + aa / (c if abs(c) > 1e-30 else 1e-30); de = d * c; h *= de
        if abs(de - 1.0) < 3e-12:
            break
    return h


def _betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lb = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
          + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return math.exp(lb) * _bcf(a, b, x) / a
    return 1.0 - math.exp(lb) * _bcf(b, a, 1 - x) / b


def t_sf(t, df):
    """One-sided upper tail of Student's t (t >= 0)."""
    return 0.5 * _betai(df / 2.0, 0.5, df / (df + t * t))


def t_ppf(q, df):
    lo, hi = 0.0, 200.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_sf(mid, df) > 1 - q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def paired_ci(d, level=0.95):
    """Mean of the per-seed differences and its two-sided paired-t interval."""
    d = np.asarray(d, float)
    n = len(d)
    se = d.std(ddof=1) / math.sqrt(n)
    t = t_ppf(1 - (1 - level) / 2, n - 1)
    return d.mean(), d.mean() - t * se, d.mean() + t * se


def paired_p(d):
    d = np.asarray(d, float)
    n = len(d)
    t = abs(d.mean() / (d.std(ddof=1) / math.sqrt(n)))
    return 2 * t_sf(t, n - 1)


def holm_intervals(diffs, family_alpha=0.05):
    """Holm step-down intervals for a family of paired contrasts.

    The contrast with the i-th smallest p-value (i = 1..m) gets a two-sided interval at
    level 1 - family_alpha / (m - i + 1). Returns a list of
    (mean, lo95, hi95, holm_lo, holm_hi, p, step_level) in the input order.
    """
    m = len(diffs)
    p = [paired_p(d) for d in diffs]
    rank = list(np.argsort(p))
    out = []
    for i, d in enumerate(diffs):
        a = family_alpha / (m - rank.index(i))
        mean, lo, hi = paired_ci(d)
        _, hl, hh = paired_ci(d, 1 - a)
        out.append((mean, lo, hi, hl, hh, p[i], a))
    return out
