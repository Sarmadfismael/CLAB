"""Figures for R3: qualitative examples and per-class confusion counts."""
import _paths  # noqa: F401  (repository paths; must come before clab)
import json, textwrap
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams.update({'font.size': 7, 'font.family': 'serif', 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})
import os
OUT = os.environ.get('FIG_OUT', str(_paths.OUT) + '/')
R1 = os.environ.get('R1_OUT', str(_paths.OUT))
pick = json.load(open(R1 + '/a11_picked.json'))
IMG = os.environ.get('EX_IMG', '') + '/'   # folder with the example tiles as PNG (from the datasets)
TGT = {'AID->UCM': 'UC Merced (AID$\\to$UCM)', 'UCM->AID': 'AID (UCM$\\to$AID)', 'AID->DFC15': 'DFC15 (AID$\\to$DFC15)'}


def fmt(pred, truth):
    pred, truth = list(pred), list(truth)
    ok = [p for p in pred if p in truth]; fp = [p for p in pred if p not in truth]
    miss = [t for t in truth if t not in pred]
    parts = ok + ['+' + p for p in fp] + ['−' + m for m in miss]
    return ', '.join(parts) if parts else '(none)'


fig, axes = plt.subplots(3, 3, figsize=(7.1, 9.6))
for k, r in enumerate(pick):
    ax = axes[k // 3][k % 3]
    ax.imshow(Image.open(IMG + r['image_id'] + '.png')); ax.set_xticks([]); ax.set_yticks([])
    tag = 'CLAB worse than probe' if k % 3 == 2 else 'largest gain over CLIP'
    ax.set_title(f"{TGT[r['setting']]}\n{r['image_id']} — {tag}", fontsize=6.6)
    w = lambda h, s: textwrap.fill(h + s, 46, subsequent_indent='   ')
    txt = '\n'.join([w('Truth: ', ', '.join(r['truth'])), w('CLIP zero-shot: ', fmt(r['clip'], r['truth'])),
                     w('Probe: ', fmt(r['probe'], r['truth'])), w('CLAB: ', fmt(r['clab'], r['truth']))])
    ax.text(0, -0.04, txt, transform=ax.transAxes, va='top', ha='left', fontsize=5.6, wrap=True)
plt.subplots_adjust(hspace=1.25, wspace=0.08)
fig.savefig(OUT + 'Figure5_examples.pdf'); plt.close(fig)

cf = pd.read_csv(R1 + '/a11_confusion.csv')
sets = ['AID->DFC15', 'UCM->DFC15']
meths = [('clip_zeroshot', 'CLIP zero-shot'), ('probe', 'Linear probe'), ('clab', 'CLAB')]
labels = ['water', 'grass', 'buildings', 'trees', 'ship', 'cars']
fig, axes = plt.subplots(len(sets) * len(meths), len(labels), figsize=(7.1, 7.0))
for i, sn in enumerate(sets):
    for j, (m, mname) in enumerate(meths):
        row = i * len(meths) + j
        for c, lab in enumerate(labels):
            q = cf[(cf.setting == sn) & (cf.method == m) & (cf.label == lab)][['TP', 'FN', 'FP', 'TN']].sum()
            M = np.array([[q.TP, q.FN], [q.FP, q.TN]], float)
            Mn = M / M.sum(1, keepdims=True)
            ax = axes[row][c]
            ax.imshow(Mn, cmap='Blues', vmin=0, vmax=1)
            for (y, x), v in np.ndenumerate(Mn):
                ax.text(x, y, f"{v:.2f}", ha='center', va='center', fontsize=5.5, color='white' if v > 0.6 else 'black')
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(['pred +', 'pred −'], fontsize=5); ax.set_yticklabels(['true +', 'true −'], fontsize=5)
            ax.tick_params(length=1, pad=1)
            if row == 0: ax.set_title(lab, fontsize=7)
            if c == 0: ax.set_ylabel(f"{sn.replace('->', chr(8594))}\n{mname}", fontsize=6)
plt.subplots_adjust(hspace=0.45, wspace=0.35)
fig.savefig(OUT + 'Figure4_confusion.pdf'); plt.close(fig)
print('ok')
