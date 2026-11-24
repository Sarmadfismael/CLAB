"""R3: computational cost. Analytic multiply-accumulate counts for the CLIP ViT-B/32 encoders and
the CLAB-specific operations, plus measured CPU time of the CLAB-specific steps on cached features."""
import _paths  # noqa: F401  (repository paths; must come before clab)
import time
from common import *

def vit_macs(img=224, patch=32, d=768, layers=12, mlp=4, out=512):
    n = (img // patch) ** 2 + 1
    patch_embed = (img // patch) ** 2 * d * 3 * patch * patch
    per_layer = 4 * n * d * d + 2 * n * n * d + 2 * n * d * mlp * d
    return patch_embed + layers * per_layer + d * out

def text_macs(n=77, d=512, layers=12, mlp=4, out=512):
    return layers * (4 * n * d * d + 2 * n * n * d + 2 * n * d * mlp * d) + d * out

C17, D = 17, 512
img = vit_macs(); txt = text_macs()
print(f"ViT-B/32 image encoder: {img/1e9:.2f} GMACs per image")
print(f"text encoder: {txt/1e9:.2f} GMACs per prompt; anchors once: 3 x 17 prompts = {51*txt/1e9:.1f} GMACs")
zs = D * C17; clab_extra = D * C17 + D * C17 + 2 * D + 4 * C17
print(f"zero-shot CLIP scoring after encoder: {zs} MACs; CLAB scoring after encoder: {clab_extra} MACs")
print(f"CLAB total per image {(img+clab_extra)/1e9:.4f} GMACs vs CLIP zero-shot {(img+zs)/1e9:.4f} GMACs; ratio {(img+clab_extra)/(img+zs):.6f}")

Fs_a, Ys_a, Ft_a, Yt_a, A, Ac = setting_arrays('AID->UCM', CLIP)
Fs, Ys, Ft, Yt = seed_split(Fs_a, Ys_a, Ft_a, Yt_a, 0)
t0 = time.perf_counter()
for _ in range(10):
    v = views(Fs, Ys, Ft, A, Ac, 0, 50)
t1 = time.perf_counter()
print(f"fit (50 epochs, {len(Fs)} source images) + score {len(Ft)} target images, CPU: {(t1-t0)/10*1000:.0f} ms")
from clab import CLAB
m = CLAB().fit(Fs, Ys, A)
x = Ft[:1]; t0 = time.perf_counter()
for _ in range(2000): m.score(x)
print(f"scoring one cached feature vector: {(time.perf_counter()-t0)/2000*1e6:.1f} microseconds")
