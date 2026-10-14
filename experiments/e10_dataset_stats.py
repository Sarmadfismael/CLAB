"""Table 1: dataset statistics and the effect of the shared-six restriction.

Reads the original label files, so it needs the datasets (no GPU):

    python experiments/e10_dataset_stats.py --data-root /path/to/datasets
        -> results/datasets.csv

Expected layout under --data-root:
    AID/multilabel.csv
    ucmerced/LandUse_Multilabeled.csv     (an XLSX file with a .csv extension)
    DFC15/multilabel.csv
"""
import _common  # noqa: F401
import argparse
from pathlib import Path

import pandas as pd

from clab import config as C
from clab.data import out_path


def read_labels(root, ds):
    p = {"aid": root / "AID" / "multilabel.csv",
         "ucm": root / "ucmerced" / "LandUse_Multilabeled.csv",
         "dfc15": root / "DFC15" / "multilabel.csv"}[ds]
    with open(p, "rb") as fh:
        is_xlsx = fh.read(2) == b"PK"
    d = pd.read_excel(p) if is_xlsx else pd.read_csv(p, encoding="utf-8-sig")
    d.columns = [str(c).strip() for c in d.columns]
    names = C.LABELS_DFC15 if ds == "dfc15" else C.LABELS_17
    lower = {c.lower(): c for c in d.columns}
    return d[[lower[n.lower()] for n in names]].astype(int), names


def main():
    ag = argparse.ArgumentParser()
    ag.add_argument("--data-root", required=True)
    root = Path(ag.parse_args().data_root)
    rows = []
    for ds in ["aid", "ucm", "dfc15"]:
        D, names = read_labels(root, ds)
        cols = [C.SHARED_6_TO_DFC15[c] if ds == "dfc15" else c for c in C.SHARED_6]
        lower = {c.lower(): c for c in D.columns}
        S = D[[lower[c.lower()] for c in cols]]
        keep = S.sum(axis=1) > 0
        rows.append(dict(dataset=ds, images=len(D), categories=len(names),
                         labels_per_image=float(D.sum(axis=1).mean()),
                         shared6_images=int(keep.sum()), removed=int((~keep).sum()),
                         removed_pct=100 * float((~keep).mean()),
                         eval_images_native=len(D) - int(round(0.8 * len(D))),
                         eval_images_shared6=int(keep.sum()) - int(round(0.8 * keep.sum()))))
    d = pd.DataFrame(rows)
    d.to_csv(out_path("datasets.csv"), index=False)
    print(d.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
