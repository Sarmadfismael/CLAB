"""Table 11 and Section 6.2: geometry of the label anchors (CLIP ViT-B/32), and the
choice of the set over which the shared-six anchors are centered.

    python experiments/e09_geometry.py
        -> results/geometry_spectrum.csv, results/geometry_cosines.csv,
           results/geometry_centering_set.csv                     (seconds)
"""
import _common  # noqa: F401
import numpy as np
import pandas as pd

from clab import config as C
from clab.core import F32, center, mAP, nrm
from clab.data import FEAT, load_features, out_path


def pairwise(M):
    G = nrm(M) @ nrm(M).T
    return G[np.triu_indices(len(M), 1)]


def main():
    A = np.load(FEAT / f"{C.HEADLINE}__TEXT__native.npy").astype(np.float64)
    spec, cos = [], []
    for lab, M, about_origin in [("raw", A, True), ("centered", center(A), False)]:
        X = M if about_origin else M - M.mean(0, keepdims=True)
        s = np.linalg.svd(X, compute_uv=False)
        ev = s ** 2 / (s ** 2).sum()
        spec.append(dict(anchors=lab, PC1=100 * ev[0], PC2=100 * ev[1], PC3=100 * ev[2],
                         PC1_3=100 * ev[:3].sum()))
        v = pairwise(M)
        cos.append(dict(anchors=lab, n_pairs=len(v), mean=v.mean(), median=np.median(v),
                        min=v.min(), max=v.max(), sd=v.std()))
    pd.DataFrame(spec).to_csv(out_path("geometry_spectrum.csv"), index=False)
    pd.DataFrame(cos).to_csv(out_path("geometry_cosines.csv"), index=False)

    # centering over the six evaluated categories vs over all seventeen
    A17 = np.load(FEAT / f"{C.HEADLINE}__TEXT__native.npy").astype(F32)
    A6 = np.load(FEAT / f"{C.HEADLINE}__TEXT__shared6.npy").astype(F32)
    idx6 = [C.LABELS_17.index(c) for c in C.SHARED_6]
    F, Y, _ = load_features(C.HEADLINE, "dfc15", "shared6")
    X = nrm(F.astype(np.float64))
    rows = []
    for lab, Ause in [("six evaluated categories (used)", center(A6)),
                      ("all seventeen, six retained", center(A17)[idx6]),
                      ("none (raw anchors)", A6)]:
        rows.append(dict(anchor_set=lab,
                         dfc15_zero_shot_mAP_all_images=mAP(Y, X @ np.asarray(Ause, np.float64).T),
                         mean_pairwise_cos=float(pairwise(Ause).mean())))
    pd.DataFrame(rows).to_csv(out_path("geometry_centering_set.csv"), index=False)
    print("wrote results/geometry_*.csv")


if __name__ == "__main__":
    main()
