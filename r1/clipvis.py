"""Standalone CLIP ViT-B/32 image tower; act='quick' (correct for OpenAI weights) or 'gelu'
(what open_clip builds for create_model('ViT-B-32', pretrained='openai'))."""
import _paths  # noqa: F401  (repository paths; must come before clab)
import sys, time, numpy as np, torch, torch.nn.functional as F
from safetensors.torch import load_file

import os
W = os.environ.get('CLIP_VISUAL_WEIGHTS', '')   # safetensors with the image-tower weights
PRE = os.environ.get('CLIP_PIXELS', '') + '/'   # folder with <ds>_<part>.npy uint8 arrays (224x224, bilinear resize)
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


class Vis:
    def __init__(self, act):
        self.sd = load_file(W); self.act = act

    @torch.no_grad()
    def __call__(self, u8):  # (B,224,224,3) uint8
        g = lambda n: self.sd['visual.' + n]
        x = torch.from_numpy(u8).permute(0, 3, 1, 2).float() / 255.
        x = (x - MEAN) / STD
        x = F.conv2d(x, g('conv1.weight'), stride=32).flatten(2).transpose(1, 2)  # B,49,768
        B = x.shape[0]
        x = torch.cat([g('class_embedding').expand(B, 1, 768), x], 1) + g('positional_embedding')
        x = F.layer_norm(x, (768,), g('ln_pre.weight'), g('ln_pre.bias'))
        for i in range(12):
            p = f'transformer.resblocks.{i}.'
            h = F.layer_norm(x, (768,), g(p + 'ln_1.weight'), g(p + 'ln_1.bias'))
            q, k, v = F.linear(h, g(p + 'attn.in_proj_weight'), g(p + 'attn.in_proj_bias')).chunk(3, -1)
            sh = lambda t: t.view(B, 50, 12, 64).transpose(1, 2)
            a = F.scaled_dot_product_attention(sh(q), sh(k), sh(v)).transpose(1, 2).reshape(B, 50, 768)
            x = x + F.linear(a, g(p + 'attn.out_proj.weight'), g(p + 'attn.out_proj.bias'))
            h = F.layer_norm(x, (768,), g(p + 'ln_2.weight'), g(p + 'ln_2.bias'))
            h = F.linear(h, g(p + 'mlp.c_fc.weight'), g(p + 'mlp.c_fc.bias'))
            h = h * torch.sigmoid(1.702 * h) if self.act == 'quick' else F.gelu(h)
            x = x + F.linear(h, g(p + 'mlp.c_proj.weight'), g(p + 'mlp.c_proj.bias'))
        x = F.layer_norm(x[:, 0], (768,), g('ln_post.weight'), g('ln_post.bias'))
        return (x @ g('proj')).numpy()


def load_pixels(ds):
    parts = {'aid': 2, 'ucm': 1, 'dfc15': 2}[ds]
    return np.concatenate([np.load(PRE + f'{ds}_{i}.npy', mmap_mode='r') for i in range(parts)])


if __name__ == '__main__':
    act = sys.argv[1]; dss = sys.argv[2].split(',')
    m = Vis(act)
    for ds in dss:
        X = load_pixels(ds); out = []; t = time.time()
        for i in range(0, len(X), 64):
            out.append(m(np.ascontiguousarray(X[i:i + 64])))
        Fz = np.concatenate(out).astype(np.float32)
        np.save(_paths.OUT / f'feat_{act}_{ds}.npy', Fz)
        print(ds, act, Fz.shape, f'{time.time() - t:.0f}s', flush=True)
