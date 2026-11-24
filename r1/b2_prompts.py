"""Table 16 (prompt families), the DFC15 vocabulary variant, and synonym anchors (R1-5).
Zero-shot, whole datasets; same definitions as extraction/prompt_ablation.py."""
import os, sys, numpy as np, pandas as pd, torch
import _paths  # noqa: F401
from clab.config import LABELS_17, SHARED_6, PROMPTS
from clab.core import ap, center, mAP, nrm
import cliptext
FEAT = os.environ['CLAB_FEATURES']; OUT = sys.argv[1]; act = sys.argv[2] if len(sys.argv) > 2 else 'quick'
BB = 'clip_vitb32_control'
PS = {"an aerial photograph containing {}": ["an aerial photograph containing {}"],
      "a satellite image of {}": ["a satellite image of {}"],
      "a remote sensing image showing {}": ["a remote sensing image showing {}"],
      "three aerial templates (used)": PROMPTS,
      "a photo of {}": ["a photo of {}"], "bare category name": ["{}"],
      "three aerial templates + a photo of": PROMPTS + ["a photo of {}"]}
DS = [("ucm", "native", LABELS_17), ("aid", "native", LABELS_17), ("dfc15", "shared6", SHARED_6)]
tt = cliptext.TextTower(act)
data = {}
for ds, sp, _ in DS:
    z = np.load(f'{FEAT}/{BB}__{ds}__{sp}.npz'); data[ds] = (nrm(z['F'].astype(np.float64)), z['Y'])
pw = lambda M: (nrm(M) @ nrm(M).T)[np.triu_indices(len(M), 1)]
rows = []
for pn, tp in PS.items():
    for ds, sp, names in DS:
        X, Y = data[ds]; A = cliptext.anchors(tt, names, tp).astype(np.float64); Ac = center(A)
        rows.append(dict(prompt_set=pn, dataset=ds, cos_raw_mean=pw(A).mean(), cos_centered_mean=pw(Ac).mean(),
                         mAP_raw=mAP(Y, X @ A.T), mAP_centered=mAP(Y, X @ Ac.T)))
d = pd.DataFrame(rows); d.to_csv(OUT + 'prompt_ablation.csv', index=False)
print(d.groupby('prompt_set', sort=False)[['cos_raw_mean', 'cos_centered_mean', 'mAP_raw', 'mAP_centered']].mean().round(3))
X, Y = data['dfc15']; vr = []
for lab, names in [("17-label vocabulary (used)", SHARED_6), ("DFC15 vocabulary", ["water", "vegetation", "building", "tree", "boat", "car"])]:
    A = cliptext.anchors(tt, names, PROMPTS).astype(np.float64); Ac = center(A)
    vr.append(dict(vocabulary=lab, mAP_raw=mAP(Y, X @ A.T), mAP_centered=mAP(Y, X @ Ac.T),
                   **{f'AP_{SHARED_6[i]}': ap(Y[:, i], (X @ Ac.T)[:, i]) * 100 for i in range(6)}))
v = pd.DataFrame(vr); v.to_csv(OUT + 'vocabulary_variant.csv', index=False); print(v.round(2).T)
