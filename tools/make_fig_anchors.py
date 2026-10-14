"""Figure 3 (anchor geometry). python tools/make_fig_anchors.py -> results/fig_anchors.pdf"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clab import config as C  # noqa: E402
from clab.core import center, nrm  # noqa: E402
from clab.data import FEAT, out_path  # noqa: E402

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.7,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
A = np.load(FEAT / f"{C.HEADLINE}__TEXT__native.npy"); Ac = center(A)
pairwise = lambda M: (nrm(M) @ nrm(M).T)[np.triu_indices(len(M), 1)]

fig = plt.figure(figsize=(7.1, 2.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1.14], wspace=0.30)
ax = fig.add_subplot(gs[0])
r, c = pairwise(A), pairwise(Ac); bins = np.linspace(-1, 1, 41)
ax.hist(r, bins=bins, alpha=.8, color="#c0504d", label=f"raw (mean {r.mean():+.2f})")
ax.hist(c, bins=bins, alpha=.8, color="#4472c4", label=f"centered (mean {c.mean():+.2f})")
ax.axvline(0, color="k", lw=.6, ls=":")
ax.set_xlabel("pairwise cosine similarity"); ax.set_ylabel("label pairs")
ax.legend(frameon=False, fontsize=6.8, loc="upper left"); ax.set_title("(a) distribution", fontsize=8)
for k, (M, ttl) in enumerate([(A, "(b) raw anchors"), (Ac, "(c) centered anchors")]):
    ax = fig.add_subplot(gs[k + 1])
    im = ax.imshow(nrm(M) @ nrm(M).T, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(17)); ax.set_yticks(range(17))
    ax.set_xticklabels(C.LABELS_17, rotation=90, fontsize=4.6)
    ax.set_yticklabels(C.LABELS_17, fontsize=4.6)
    ax.set_title(ttl, fontsize=8); ax.tick_params(length=1.2, pad=1)
    if k == 1:
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=6, length=1.5)
fig.savefig(out_path("fig_anchors.pdf")); print("wrote results/fig_anchors.pdf")
