"""Table 8 and Section 5.7: robustness when target imagery outside the label vocabulary
is restored, plus the alpha sweep on the standard transfers used in Table 7's text.

Mismatch arms: all six directions among AID, UC Merced and DFC15 in the shared-six
space. The SOURCE is filtered to images with at least one positive label (a head must
be fitted on labelled imagery); the TARGET keeps every image, including those with no
evaluated category. Standard arms: the four transfers of the main table.

    python experiments/e04_robustness.py
        -> results/robust_mismatch.csv, results/robust_standard.csv   (~3 min)
"""
import _common  # noqa: F401
import pandas as pd

from clab import config as C
from clab.core import mAP, views
from clab.data import out_path, seed_split, setting_arrays

ALPHAS = [round(0.025 * i, 3) for i in range(1, 29)]
DATASETS = ["aid", "ucm", "dfc15"]
PAIRS = [(a, b) for a in DATASETS for b in DATASETS if a != b]


def run(arrays):
    Fs_a, Ys_a, Ft_a, Yt_a, A, Ac = arrays
    out = []
    for seed in C.SEEDS:
        Fs, Ys, Ft, Yt = seed_split(Fs_a, Ys_a, Ft_a, Yt_a, seed)
        v = views(Fs, Ys, Ft, A, Ac, seed, C.EPOCHS)
        rec = dict(seed=seed, oov=float((Yt.sum(1) == 0).mean()), probe=mAP(Yt, v["Ht"]))
        for a in ALPHAS:
            rec[f"u{a}"] = mAP(Yt, a * v["zU"] + (1 - a) * v["zH"])
        out.append(rec)
    return pd.DataFrame(out)


def main():
    mism = []
    for s, t in PAIRS:
        name = f"{s.upper()}->{t.upper()}"
        arr = setting_arrays((s, t), C.HEADLINE, space="shared6keep", keep_source_empty=False)
        mism.append(run(arr).assign(setting=name))
        print(f"  mismatch {name:12s} done", flush=True)
    pd.concat(mism).to_csv(out_path("robust_mismatch.csv"), index=False)
    std = [run(setting_arrays(sn, C.HEADLINE)).assign(setting=sn) for sn in C.ORDER]
    pd.concat(std).to_csv(out_path("robust_standard.csv"), index=False)
    print("wrote results/robust_mismatch.csv, results/robust_standard.csv")


if __name__ == "__main__":
    main()
