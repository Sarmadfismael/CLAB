"""Table 13 row for CLIP ViT-B/32 under the protocol of extraction/backbone_study.py, from cached
features (feature dir from CLAB_FEATURES). Prints GELU check vs results/backbone_study.csv."""
import os, sys, numpy as np, pandas as pd, torch
import _paths  # noqa: F401
from clab.config import ORDER, SETTINGS
from clab.core import split_idx
from sklearn.metrics import average_precision_score
FEAT = os.environ['CLAB_FEATURES']; OUT = sys.argv[1]
rows = []
for sn in ORDER:
    src, tgt, sp = SETTINGS[sn]
    zs = np.load(f'{FEAT}/clip_vitb32_control__{src}__{sp}.npz'); zt = np.load(f'{FEAT}/clip_vitb32_control__{tgt}__{sp}.npz')
    si, _ = split_idx(len(zs['Y']), 0.8, 0); _, ti = split_idx(len(zt['Y']), 0.8, 0)
    Fs, Ys, Ft, Yt = zs['F'][si], zs['Y'][si], zt['F'][ti], zt['Y'][ti]
    mu, sd = Fs.mean(0, keepdims=True), Fs.std(0, keepdims=True) + 1e-6
    xs = torch.from_numpy((Fs - mu) / sd).float(); xt = torch.from_numpy((Ft - mu) / sd).float(); ys = torch.from_numpy(Ys).float()
    for s in range(5):
        torch.manual_seed(s); head = torch.nn.Linear(xs.shape[1], Ys.shape[1])
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4); bce = torch.nn.BCEWithLogitsLoss()
        for _ in range(300):
            opt.zero_grad(); bce(head(xs), ys).backward(); opt.step()
        with torch.no_grad(): z = head(xt).numpy().astype(np.float64)
        aps = [average_precision_score(Yt[:, c], z[:, c]) for c in range(Yt.shape[1]) if Yt[:, c].sum() > 0]
        rows.append(dict(setting=sn, backbone='clip_vitb32_control', seed=s, mAP=100 * float(np.mean(aps))))
d = pd.DataFrame(rows); d.to_csv(OUT, index=False)
print(d.groupby('setting', sort=False)['mAP'].agg(['mean', 'std']).round(2))
