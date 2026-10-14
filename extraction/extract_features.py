"""Extract and cache frozen image features and class-name anchors (needs a GPU or patience).

    python extraction/extract_features.py --data-root /path/to/datasets
        [--backbones clip_vitb32_control,remoteclip_vitb32] [--out features]

Writes, per backbone:
    features/{bb}__{aid,ucm}__native.npz          17-label space
    features/{bb}__{aid,ucm,dfc15}__shared6.npz   shared-six space, empty images removed
    features/{bb}__{aid,ucm,dfc15}__shared6keep.npz  shared-six space, all images kept
    features/{bb}__TEXT__{native,shared6}.npy     anchors (rows follow the label order)
Each .npz holds F (N, D) float32 features, Y (N, C) labels and ids (image names, in row
order). Row order is the label-file order, which is what the split seeds index into.

Images: resized to 224x224, no crop, normalized with the backbone's pretraining
statistics. Features are the output of the image tower (CLIP: the projected embedding).
Anchors: for each class name (hyphens -> spaces), the three prompts of clab/config.py are
encoded, each embedding L2-normalized, averaged, and the mean renormalized.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

from backbones import STATS, TEXT_CAPABLE, build_full_clip, build_image_encoder  # noqa: E402
from datasets import MultiLabelRS, build_split, eval_transform  # noqa: E402
from clab.config import LABELS_17, PROMPTS, SHARED_6  # noqa: E402

JOBS = [("aid", "native"), ("ucm", "native"),
        ("aid", "shared6"), ("ucm", "shared6"), ("dfc15", "shared6"),
        ("aid", "shared6keep"), ("ucm", "shared6keep"), ("dfc15", "shared6keep")]


def anchors(model, tok, names, templates, device):
    import torch
    import torch.nn.functional as Fn
    out = []
    with torch.no_grad():
        for c in names:
            nm = c.replace("-", " ")
            e = model.encode_text(tok([t.format(nm) for t in templates]).to(device))
            e = Fn.normalize(e.float(), dim=-1).mean(0)
            out.append(Fn.normalize(e, dim=-1).cpu().numpy())
    return np.stack(out)


def main():
    import torch
    from torch.utils.data import DataLoader
    ag = argparse.ArgumentParser()
    ag.add_argument("--data-root", required=True)
    ag.add_argument("--backbones", default="clip_vitb32_control,remoteclip_vitb32")
    ag.add_argument("--out", default=str(HERE.parent / "features"))
    ag.add_argument("--batch-size", type=int, default=32)
    a = ag.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for bb in [b.strip() for b in a.backbones.split(",") if b.strip()]:
        mod, feat = build_image_encoder(bb)
        mod = mod.eval().to(dev)
        for ds, sp in JOBS:
            f = out / f"{bb}__{ds}__{sp}.npz"
            if f.exists():
                print(f"  cached {f.name}"); continue
            data = MultiLabelRS(a.data_root, ds, build_split(a.data_root, ds, sp),
                                eval_transform(STATS[bb]))
            dl = DataLoader(data, batch_size=a.batch_size, shuffle=False, num_workers=0)
            F, Y = [], []
            with torch.no_grad():
                for x, y in dl:
                    F.append(feat(mod, x.to(dev)).float().cpu().numpy()); Y.append(y.numpy())
            F, Y = np.concatenate(F), np.concatenate(Y)
            np.savez_compressed(f, F=F, Y=Y, ids=np.array(data.split.ids))
            print(f"  wrote {f.name} {F.shape}")
        del mod
        if bb in TEXT_CAPABLE:
            model, tok = build_full_clip(bb)
            model = model.eval().to(dev)
            for sp, names in [("native", LABELS_17), ("shared6", SHARED_6)]:
                f = out / f"{bb}__TEXT__{sp}.npy"
                if not f.exists():
                    np.save(f, anchors(model, tok, names, PROMPTS, dev))
                    print(f"  wrote {f.name}")
            del model
        if dev == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
