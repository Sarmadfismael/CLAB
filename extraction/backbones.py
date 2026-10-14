"""Frozen encoders used in the paper, with exact checkpoint identifiers."""
from __future__ import annotations

IMAGENET_STATS = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
CLIP_STATS = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

# name -> how to build it. "clip" in the name selects CLIP normalization statistics.
CANDIDATES = {
    "imagenet_tresnet_m":    dict(kind="timm", model="tresnet_m"),            # ImageNet-1K
    "imagenet21k_tresnet_m": dict(kind="timm", model="tresnet_m.miil_in21k"),  # ImageNet-21K
    "satmae_vitlarge":       dict(kind="rshf", module="satmae", cls="SatMAE",
                                  repo="MVRL/satmae-vitlarge-fmow-pretrain-800"),
    "clip_rn50_control":     dict(kind="open_clip", model="RN50", pretrained="openai"),
    "clip_vitb32_control":   dict(kind="open_clip", model="ViT-B-32", pretrained="openai"),
    "remoteclip_rn50":       dict(kind="open_clip_hf", model="RN50", repo="chendelong/RemoteCLIP",
                                  filename="RemoteCLIP-RN50.pt"),
    "remoteclip_vitb32":     dict(kind="open_clip_hf", model="ViT-B-32",
                                  repo="chendelong/RemoteCLIP",
                                  filename="RemoteCLIP-ViT-B-32.pt"),
    "remoteclip_vitl14":     dict(kind="open_clip_hf", model="ViT-L-14",
                                  repo="chendelong/RemoteCLIP",
                                  filename="RemoteCLIP-ViT-L-14.pt"),
}
STATS = {n: (CLIP_STATS if "clip" in n else IMAGENET_STATS) for n in CANDIDATES}
TEXT_CAPABLE = {n for n, s in CANDIDATES.items() if s["kind"].startswith("open_clip")}


def _sig(m):
    import torch
    with torch.no_grad():
        return float(sum(p.detach().float().abs().sum() for p in m.parameters()))


def build_full_clip(name):
    """The complete open_clip model (image and text towers) with the right weights."""
    import open_clip
    import torch
    spec = CANDIDATES[name]
    if spec["kind"] == "open_clip":
        m, _, _ = open_clip.create_model_and_transforms(spec["model"],
                                                        pretrained=spec["pretrained"])
    else:
        from huggingface_hub import hf_hub_download
        m, _, _ = open_clip.create_model_and_transforms(spec["model"])
        before = _sig(m.visual)
        sd = torch.load(hf_hub_download(spec["repo"], spec["filename"]), map_location="cpu")
        m.load_state_dict(sd.get("state_dict", sd), strict=False)
        if abs(_sig(m.visual) - before) < 1e-3:
            raise RuntimeError(f"{name}: checkpoint did not load into the visual tower")
    return m, open_clip.get_tokenizer(spec["model"])


def build_image_encoder(name):
    """Returns (module, feature_fn) with feature_fn(module, x) -> (B, D)."""
    spec = CANDIDATES[name]
    if spec["kind"] == "timm":
        import timm
        m = timm.create_model(spec["model"], pretrained=True, num_classes=0)
        return m, (lambda mdl, x: mdl(x))
    if spec["kind"].startswith("open_clip"):
        m, _ = build_full_clip(name)
        return m.visual, (lambda mdl, x: mdl(x))
    if spec["kind"] == "rshf":                         # SatMAE: mean of patch tokens
        import importlib
        mod = importlib.import_module(f"rshf.{spec['module']}")
        m = getattr(mod, spec["cls"]).from_pretrained(spec["repo"])

        def feats(mdl, x):
            if hasattr(mdl, "forward_encoder"):
                t = mdl.forward_encoder(x, mask_ratio=0.0)[0]
                return t[:, 1:, :].mean(dim=1) if t.ndim == 3 else t
            out = mdl(x)
            if isinstance(out, (list, tuple)):
                out = out[-1]
            if out.ndim == 4:
                return out.mean(dim=(2, 3))
            if out.ndim == 3:
                return out[:, 1:, :].mean(dim=1)
            return out
        return m, feats
    raise ValueError(spec["kind"])
