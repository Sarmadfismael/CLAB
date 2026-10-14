"""Table 12 (prompt families) and the DFC15 vocabulary variant of Section 6.5.
Needs the CLIP ViT-B/32 text encoder (open_clip) and the cached image features.

    python extraction/prompt_ablation.py
        -> results/prompt_ablation.csv, results/vocabulary_variant.csv

Zero-shot only: no source labels, no training. Each dataset is scored in full.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from backbones import build_full_clip  # noqa: E402
from extract_features import anchors  # noqa: E402
from clab.config import LABELS_17, SHARED_6  # noqa: E402
from clab.core import ap, center, mAP, nrm  # noqa: E402
from clab.data import FEAT, out_path  # noqa: E402

BB = "clip_vitb32_control"
PROMPT_SETS = {
    "an aerial photograph containing {}": ["an aerial photograph containing {}"],
    "a satellite image of {}": ["a satellite image of {}"],
    "a remote sensing image showing {}": ["a remote sensing image showing {}"],
    "three aerial templates (used)": ["an aerial photograph containing {}",
                                      "a satellite image of {}",
                                      "a remote sensing image showing {}"],
    "a photo of {}": ["a photo of {}"],
    "bare category name": ["{}"],
    "three aerial templates + a photo of": ["an aerial photograph containing {}",
                                            "a satellite image of {}",
                                            "a remote sensing image showing {}",
                                            "a photo of {}"],
}
DATASETS = [("ucm", "native", LABELS_17), ("aid", "native", LABELS_17),
            ("dfc15", "shared6", SHARED_6)]
DFC15_NAMES = ["water", "vegetation", "building", "tree", "boat", "car"]


def pairwise(M):
    return (nrm(M) @ nrm(M).T)[np.triu_indices(len(M), 1)]


def main():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = build_full_clip(BB)
    model = model.eval().to(dev)
    data = {}
    for ds, sp, _ in DATASETS:
        z = np.load(FEAT / f"{BB}__{ds}__{sp}.npz")
        data[ds] = (nrm(z["F"].astype(np.float64)), z["Y"])
    rows = []
    for pname, templates in PROMPT_SETS.items():
        for ds, sp, names in DATASETS:
            X, Y = data[ds]
            A = anchors(model, tok, names, templates, dev); Ac = center(A)
            rows.append(dict(prompt_set=pname, n_templates=len(templates), dataset=ds,
                             cos_raw_mean=pairwise(A).mean(), cos_centered_mean=pairwise(Ac).mean(),
                             mAP_raw=mAP(Y, X @ A.T), mAP_centered=mAP(Y, X @ Ac.T)))
    pd.DataFrame(rows).to_csv(out_path("prompt_ablation.csv"), index=False)

    X, Y = data["dfc15"]
    vr = []
    for lab, names in [("17-label vocabulary (used)", SHARED_6), ("DFC15 vocabulary", DFC15_NAMES)]:
        A = anchors(model, tok, names, PROMPT_SETS["three aerial templates (used)"], dev)
        Ac = center(A)
        per = {f"AP_{SHARED_6[i]}": ap(Y[:, i], (X @ Ac.T)[:, i]) * 100 for i in range(6)}
        vr.append(dict(vocabulary=lab, names=" / ".join(names), mAP_raw=mAP(Y, X @ A.T),
                       mAP_centered=mAP(Y, X @ Ac.T), **per))
    pd.DataFrame(vr).to_csv(out_path("vocabulary_variant.csv"), index=False)
    print("wrote results/prompt_ablation.csv, results/vocabulary_variant.csv")


if __name__ == "__main__":
    main()
