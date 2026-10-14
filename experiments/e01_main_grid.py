"""Tables 2, 3 and 5 (main results, paired contrasts, ablation), the zero-shot
centering gains of Section 6.2 and the zero-shot column of Table 16.

For every transfer, backbone (CLIP ViT-B/32 and RemoteCLIP ViT-B/32) and split seed
0-19, scores eight arms on identical features, source indices and evaluation set:

    zeroshot_raw   semantic branch, raw anchors
    zeroshot       semantic branch, centered anchors
    probe          the linear source head alone (the matched baseline)
    clab           CLAB, alpha = 0.30, 50 epochs
    clab_raw       CLAB with raw anchors
    clab_a02/a04   CLAB with alpha = 0.20 / 0.40
    clab_long      CLAB with an 800-epoch head

    python experiments/e01_main_grid.py          -> results/main_grid.csv  (~2 min)
"""
import _common  # noqa: F401
import pandas as pd

from clab import config as C
from clab.core import f1_at_median, mAP, views
from clab.data import out_path, seed_split, setting_arrays

BACKBONES = [C.HEADLINE, C.DIAGNOSTIC]


def main():
    rows = []
    for sn in C.ORDER:
        for bb in BACKBONES:
            Fs_a, Ys_a, Ft_a, Yt_a, A, Ac = setting_arrays(sn, bb)
            for seed in C.SEEDS:
                Fs, Ys, Ft, Yt = seed_split(Fs_a, Ys_a, Ft_a, Yt_a, seed)
                v = views(Fs, Ys, Ft, A, Ac, seed, C.EPOCHS)
                v8 = views(Fs, Ys, Ft, A, Ac, seed, 800)
                a = C.ALPHA
                S = {"zeroshot_raw": v["zR"], "zeroshot": v["zU"], "probe": v["Ht"],
                     "clab": a * v["zU"] + (1 - a) * v["zH"],
                     "clab_raw": a * v["zR"] + (1 - a) * v["zH"],
                     "clab_a04": 0.40 * v["zU"] + 0.60 * v["zH"],
                     "clab_a02": 0.20 * v["zU"] + 0.80 * v["zH"],
                     "clab_long": a * v8["zU"] + (1 - a) * v8["zH"]}
                for m, sc in S.items():
                    cf1, of1 = f1_at_median(Yt, sc)
                    rows.append(dict(setting=sn, backbone=bb, method=m, seed=seed,
                                     mAP=mAP(Yt, sc), CF1=cf1, OF1=of1))
            print(f"  {sn:12s} {bb:22s} done", flush=True)
    pd.DataFrame(rows).to_csv(out_path("main_grid.csv"), index=False)
    print("wrote results/main_grid.csv")


if __name__ == "__main__":
    main()
