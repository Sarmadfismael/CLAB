"""Loading cached features and building the per-seed source/target arrays."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import config as C
from .core import F32, center, split_idx

ROOT = Path(__file__).resolve().parents[1]
FEAT = Path(os.environ.get("CLAB_FEATURES", ROOT / "features"))
RESULTS = Path(os.environ.get("CLAB_RESULTS", ROOT / "results"))


def load_features(backbone, dataset, space):
    """Returns F (N, D) float32, Y (N, C) float32 and the image ids in row order."""
    z = np.load(FEAT / f"{backbone}__{dataset}__{space}.npz", allow_pickle=True)
    ids = z["ids"] if "ids" in z.files else None
    return z["F"].astype(F32), z["Y"].astype(F32), ids


def load_anchors(backbone, space):
    """Raw anchors A and centered anchors Ac. `shared6keep` uses the shared6 anchors."""
    sp = "shared6" if space.startswith("shared6") else space
    A = np.load(FEAT / f"{backbone}__TEXT__{sp}.npy").astype(F32)
    return A, center(A)


def setting_arrays(setting, backbone, space=None, keep_source_empty=True):
    """Full source and target arrays for one transfer.

    space overrides the label space (used for the out-of-vocabulary robustness check,
    where the target keeps images with no positive label and the source is filtered).
    """
    src, tgt, sp = C.SETTINGS[setting] if setting in C.SETTINGS else (*setting, space)
    sp = space or sp
    Fs, Ys, _ = load_features(backbone, src, sp)
    Ft, Yt, _ = load_features(backbone, tgt, sp)
    if not keep_source_empty:
        k = Ys.sum(1) > 0
        Fs, Ys = Fs[k], Ys[k]
    A, Ac = load_anchors(backbone, sp)
    return Fs, Ys, Ft, Yt, A, Ac


def seed_split(Fs, Ys, Ft, Yt, seed):
    """The source 80% used for fitting and the target 20% used for evaluation."""
    si, _ = split_idx(len(Fs), C.SOURCE_FRAC, seed)
    _, ti = split_idx(len(Ft), C.TARGET_FRAC, seed)
    return Fs[si], Ys[si], Ft[ti], Yt[ti]


def out_path(name):
    RESULTS.mkdir(parents=True, exist_ok=True)
    return RESULTS / name
