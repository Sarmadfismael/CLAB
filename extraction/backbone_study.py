"""Table 10: frozen-feature backbone comparison (eight backbones, four transfers).

One fixed split (seed 0: the source head is fitted on the first 80% of the source
permutation, evaluation on the last 20% of the target permutation), five head
initializations (torch seeds 0-4). Head: one linear layer on source-standardized
features, AdamW (lr 1e-3, weight decay 1e-4), 300 full-batch steps, BCE loss.
mAP uses scikit-learn's average_precision_score on logits.

    python extraction/backbone_study.py --data-root /path/to/datasets
        -> results/backbone_study.csv          (GPU recommended; downloads checkpoints)

SatMAE needs `pip install rshf`; RemoteCLIP checkpoints come from the Hugging Face
repository chendelong/RemoteCLIP.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from backbones import CANDIDATES, STATS, build_image_encoder  # noqa: E402
from datasets import MultiLabelRS, build_split, eval_transform  # noqa: E402
from clab.config import ORDER, SETTINGS  # noqa: E402
from clab.core import split_idx  # noqa: E402
from clab.data import out_path  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]


def main():
    import torch
    from sklearn.metrics import average_precision_score
    from torch.utils.data import DataLoader
    ag = argparse.ArgumentParser()
    ag.add_argument("--data-root", required=True)
    ag.add_argument("--only", default="")
    a = ag.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    names = [n for n in a.only.split(",") if n] or list(CANDIDATES)

    def extract(mod, feat, root, ds, sp, idx):
        split = build_split(root, ds, sp)
        split.ids = [split.ids[i] for i in idx]; split.y = split.y[idx]
        dl = DataLoader(MultiLabelRS(root, ds, split, eval_transform(STATS[n])),
                        batch_size=32, shuffle=False)
        F, Y = [], []
        with torch.no_grad():
            for x, y in dl:
                F.append(feat(mod, x.to(dev)).float().cpu().numpy()); Y.append(y.numpy())
        return np.concatenate(F), np.concatenate(Y)

    rows = []
    for sn in ORDER:
        src, tgt, sp = SETTINGS[sn]
        ns = len(build_split(a.data_root, src, sp).ids); nt = len(build_split(a.data_root, tgt, sp).ids)
        si, _ = split_idx(ns, 0.8, 0); _, ti = split_idx(nt, 0.8, 0)
        for n in names:
            mod, feat = build_image_encoder(n)
            mod = mod.eval().to(dev)
            Fs, Ys = extract(mod, feat, a.data_root, src, sp, si)
            Ft, Yt = extract(mod, feat, a.data_root, tgt, sp, ti)
            mu, sd = Fs.mean(0, keepdims=True), Fs.std(0, keepdims=True) + 1e-6
            xs = torch.from_numpy((Fs - mu) / sd).float(); xt = torch.from_numpy((Ft - mu) / sd).float()
            ys = torch.from_numpy(Ys).float()
            for s in SEEDS:
                torch.manual_seed(s)
                head = torch.nn.Linear(xs.shape[1], Ys.shape[1])
                opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
                bce = torch.nn.BCEWithLogitsLoss()
                for _ in range(300):
                    opt.zero_grad(); bce(head(xs), ys).backward(); opt.step()
                with torch.no_grad():
                    z = head(xt).numpy().astype(np.float64)
                aps = [average_precision_score(Yt[:, c], z[:, c]) for c in range(Yt.shape[1])
                       if Yt[:, c].sum() > 0]
                rows.append(dict(setting=sn, backbone=n, seed=s, mAP=100 * float(np.mean(aps))))
            print(f"  {sn:12s} {n:24s} {np.mean([r['mAP'] for r in rows[-5:]]):.2f}", flush=True)
            del mod
            if dev == "cuda":
                torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(out_path("backbone_study.csv"), index=False)


if __name__ == "__main__":
    main()
