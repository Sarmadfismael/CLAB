"""Write the split membership of every image for every seed, so the protocol can be
reproduced without this code.

    python tools/export_splits.py      -> splits/{dataset}__{space}.csv

Each file lists the images in feature-row order (= label-file order after the stated
filtering). Column seed_k is 'fit' for the first 80% of the seed-k permutation and
'eval' for the remaining 20%. A dataset used as SOURCE contributes its 'fit' rows to the
head; a dataset used as TARGET is evaluated on its 'eval' rows. Nothing else is used.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clab import config as C  # noqa: E402
from clab.core import split_idx  # noqa: E402
from clab.data import load_features  # noqa: E402

JOBS = [("aid", "native"), ("ucm", "native"), ("aid", "shared6"), ("ucm", "shared6"),
        ("dfc15", "shared6"), ("aid", "shared6keep"), ("ucm", "shared6keep"),
        ("dfc15", "shared6keep")]


def main():
    out = ROOT / "splits"; out.mkdir(exist_ok=True)
    for ds, sp in JOBS:
        F, Y, ids = load_features(C.HEADLINE, ds, sp)
        names = C.LABELS_17 if sp == "native" else C.SHARED_6
        d = pd.DataFrame({"row": np.arange(len(F)), "image_id": ids})
        for n, col in zip(names, Y.T):
            d[f"label_{n}"] = col.astype(int)
        for s in C.SEEDS:
            fit, ev = split_idx(len(F), C.SOURCE_FRAC, s)
            role = np.empty(len(F), dtype=object); role[fit] = "fit"; role[ev] = "eval"
            d[f"seed_{s}"] = role
        d.to_csv(out / f"{ds}__{sp}.csv", index=False)
        print(f"  {ds}__{sp}.csv  {len(d)} images")


if __name__ == "__main__":
    main()
