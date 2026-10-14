"""Label files, label spaces and image lookup for UC Merced, AID and DFC15 (multi-label).

Expected layout under the data root (paths can be changed in IMG_DIRS / LABEL_FILES):

    AID/multilabel.csv            AID/images_tr/..., AID/images_test/...
    ucmerced/LandUse_Multilabeled.csv  ucmerced/uc_merced/<class>/<name>.tif
    DFC15/multilabel.csv          DFC15/images_tr/..., DFC15/images_test/...
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clab.config import LABELS_17, SHARED_6, SHARED_6_TO_DFC15  # noqa: E402

LABEL_FILES = {"aid": "AID/multilabel.csv",
               "ucm": "ucmerced/LandUse_Multilabeled.csv",
               "dfc15": "DFC15/multilabel.csv"}
IMG_DIRS = {"aid": ["AID/images_tr", "AID/images_test"],
            "ucm": ["ucmerced/uc_merced"],
            "dfc15": ["DFC15/images_tr", "DFC15/images_test"]}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
SPACES = {"native": LABELS_17, "shared6": SHARED_6, "shared6keep": SHARED_6}


@dataclass
class Split:
    ids: List[str]
    y: np.ndarray
    label_names: List[str]


def _read_label_table(path):
    with open(path, "rb") as fh:
        magic = fh.read(2)
    if magic == b"PK":                      # UC Merced ships an XLSX named .csv
        return pd.read_excel(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def load_labels(root, dataset):
    df = _read_label_table(os.path.join(root, LABEL_FILES[dataset]))
    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
    df = df.rename(columns={df.columns[0]: "image"})
    df["image"] = df["image"].astype(str).str.strip()
    for c in df.columns[1:]:
        df[c] = df[c].astype(int)
    return df.set_index("image")


def build_split(root, dataset, space_name):
    """Rows in label-file order. shared6 drops images with no positive shared label;
    shared6keep keeps them; native is the 17-label space (AID and UC Merced only)."""
    space = SPACES[space_name]
    df = load_labels(root, dataset)
    cols = [SHARED_6_TO_DFC15[n] if dataset == "dfc15" else n for n in space]
    y = np.stack([df[c].to_numpy() for c in cols], 1).astype(np.float32)
    ids = list(df.index)
    if space_name == "shared6":
        keep = y.sum(1) > 0
        y, ids = y[keep], [i for i, k in zip(ids, keep) if k]
    return Split(ids, y, list(space))


def file_index(roots):
    index = {}
    for r in roots:
        for dirpath, _, files in os.walk(r):
            for f in files:
                stem, ext = os.path.splitext(f)
                if ext.lower() in IMG_EXTS:
                    index.setdefault(stem, os.path.join(dirpath, f))
                    index.setdefault(stem.lower(), os.path.join(dirpath, f))
    return index


class MultiLabelRS:
    """torch Dataset over one split; ids without an image file are dropped and reported."""

    def __init__(self, root, dataset, split: Split, transform):
        import torch  # noqa: F401
        idx = file_index([os.path.join(root, d) for d in IMG_DIRS[dataset]])
        keep = [i for i, k in enumerate(split.ids) if (idx.get(k) or idx.get(k.lower()))]
        if len(keep) != len(split.ids):
            print(f"[{dataset}] {len(split.ids) - len(keep)} ids had no image file; dropped")
        self.paths = [idx.get(split.ids[i]) or idx.get(split.ids[i].lower()) for i in keep]
        self.split = Split([split.ids[i] for i in keep], split.y[keep], split.label_names)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import torch
        from PIL import Image
        img = self.transform(Image.open(self.paths[i]).convert("RGB"))
        return img, torch.from_numpy(self.split.y[i])


def eval_transform(stats, size=224):
    from torchvision import transforms as T
    mean, std = stats
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(mean, std)])
