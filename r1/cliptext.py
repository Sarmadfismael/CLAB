"""Standalone CLIP ViT-B/32 (OpenAI) text tower with QuickGELU, causal mask and EOT pooling, as in
OpenCLIP's ViT-B-32-quickgelu. Used for the prompt-ablation, synonym and prompt-learning analyses;
supports learnable context vectors (CoOp) and shares the class-independent prefix across classes.
Weights: CLIP_TEXT_WEIGHTS (safetensors) if set, otherwise OpenCLIP's 'openai' weights."""
import _paths  # noqa: F401  (repository paths; must come before clab)
import importlib.util, sys, torch, torch.nn as nn, torch.nn.functional as F
from safetensors.torch import load_file

import os
W = os.environ.get('CLIP_TEXT_WEIGHTS')   # optional: a safetensors file with the text-tower weights
_oc = importlib.util.find_spec('open_clip')
_spec = importlib.util.spec_from_file_location(
    'oc_tok', os.path.join(os.path.dirname(_oc.origin), 'tokenizer.py'))
_tokmod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_tokmod)
TOK = _tokmod.SimpleTokenizer()
SOT, EOT = TOK.encoder['<start_of_text>'], TOK.encoder['<end_of_text>']


class TextTower(nn.Module):
    def __init__(self, act='quick'):
        super().__init__()
        self.act = act
        if W:
            sd = load_file(W)
        else:   # take the OpenAI weights through OpenCLIP (downloads them once)
            import open_clip
            m = open_clip.create_model('ViT-B-32-quickgelu', pretrained='openai')
            sd = {k: v.detach().float() for k, v in m.state_dict().items() if not k.startswith('visual.')}
        self.sd = {k: v for k, v in sd.items()}
        for v in self.sd.values(): v.requires_grad_(False)
        self.L = 12; self.H = 8

    def tok(self, s):
        return [SOT] + TOK.encode(s) + [EOT]

    def blocks(self, x):  # x: (B, T, 512), causal
        T = x.shape[1]
        mask = torch.full((T, T), float('-inf')).triu(1)
        for i in range(self.L):
            p = f'transformer.resblocks.{i}.'
            g = lambda n: self.sd[p + n]
            h = F.layer_norm(x, (512,), g('ln_1.weight'), g('ln_1.bias'))
            qkv = F.linear(h, g('attn.in_proj_weight'), g('attn.in_proj_bias'))
            q, k, v = qkv.chunk(3, -1)
            B = x.shape[0]
            sh = lambda t: t.view(B, T, self.H, 64).transpose(1, 2)
            a = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), attn_mask=mask)
            a = a.transpose(1, 2).reshape(B, T, 512)
            x = x + F.linear(a, g('attn.out_proj.weight'), g('attn.out_proj.bias'))
            h = F.layer_norm(x, (512,), g('ln_2.weight'), g('ln_2.bias'))
            h = F.linear(h, g('mlp.c_fc.weight'), g('mlp.c_fc.bias'))
            h = h * torch.sigmoid(1.702 * h) if self.act == 'quick' else F.gelu(h)
            x = x + F.linear(h, g('mlp.c_proj.weight'), g('mlp.c_proj.bias'))
        return x

    def head(self, x, eot_idx):
        x = F.layer_norm(x, (512,), self.sd['ln_final.weight'], self.sd['ln_final.bias'])
        x = x[torch.arange(x.shape[0]), eot_idx]
        return x @ self.sd['text_projection']

    def encode(self, strings):
        """Encode a list of strings; sequences are trimmed to their own length (exact under the
        causal mask, since EOT pooling never sees later positions)."""
        out = []
        for s in strings:
            ids = torch.tensor(self.tok(s))
            x = self.sd['token_embedding.weight'][ids] + self.sd['positional_embedding'][:len(ids)]
            out.append(self.head(self.blocks(x[None]), torch.tensor([len(ids) - 1]))[0])
        return torch.stack(out)

    def encode_ctx(self, ctx, names):
        """CoOp: [SOT] ctx_1..ctx_M  name tokens . [EOT]; ctx is (M,512) learnable (shared)."""
        M = ctx.shape[0]
        seqs, eots, Tmax = [], [], 0
        for n in names:
            ids = TOK.encode(n + '.')
            seqs.append(ids); Tmax = max(Tmax, 1 + M + len(ids) + 1)
        E = self.sd['token_embedding.weight']
        xs, eot = [], []
        for ids in seqs:
            pre = E[torch.tensor([SOT])]
            post = E[torch.tensor(ids + [EOT])]
            x = torch.cat([pre, ctx, post], 0)
            eot.append(x.shape[0] - 1)
            if x.shape[0] < Tmax:
                x = torch.cat([x, E[torch.zeros(Tmax - x.shape[0], dtype=torch.long)]], 0)
            xs.append(x)
        x = torch.stack(xs) + self.sd['positional_embedding'][:Tmax]
        return self.head(self.blocks(x), torch.tensor(eot))

    def ctx_init(self, phrase):
        ids = TOK.encode(phrase)
        return self.sd['token_embedding.weight'][torch.tensor(ids)].clone()


def anchors(tt, names, templates):
    with torch.no_grad():
        out = []
        for c in names:
            nm = c.replace('-', ' ')
            e = F.normalize(tt.encode([t.format(nm) for t in templates]), dim=-1).mean(0)
            out.append(F.normalize(e, dim=-1))
        return torch.stack(out).numpy()


if __name__ == '__main__':
    import numpy as np
    import _paths  # noqa: F401
    from clab.config import LABELS_17, SHARED_6, PROMPTS
    tt = TextTower()
    A = anchors(tt, LABELS_17, PROMPTS)
    ref = np.load(_paths.FEAT / 'clip_vitb32_control__TEXT__native.npy')
    print('17: min cos to cached', (A * ref).sum(1).min())
    A6 = anchors(tt, SHARED_6, PROMPTS)
    ref6 = np.load(_paths.FEAT / 'clip_vitb32_control__TEXT__shared6.npy')
    print('6: min cos to cached', (A6 * ref6).sum(1).min())


def _layer_qkv(sd, i, x):
    p = f'transformer.resblocks.{i}.'
    h = F.layer_norm(x, (512,), sd[p + 'ln_1.weight'], sd[p + 'ln_1.bias'])
    return F.linear(h, sd[p + 'attn.in_proj_weight'], sd[p + 'attn.in_proj_bias']).chunk(3, -1)


def _layer_rest(sd, i, x, a, act):
    p = f'transformer.resblocks.{i}.'
    x = x + F.linear(a, sd[p + 'attn.out_proj.weight'], sd[p + 'attn.out_proj.bias'])
    h = F.layer_norm(x, (512,), sd[p + 'ln_2.weight'], sd[p + 'ln_2.bias'])
    h = F.linear(h, sd[p + 'mlp.c_fc.weight'], sd[p + 'mlp.c_fc.bias'])
    h = h * torch.sigmoid(1.702 * h) if act == 'quick' else F.gelu(h)
    return x + F.linear(h, sd[p + 'mlp.c_proj.weight'], sd[p + 'mlp.c_proj.bias'])


def encode_ctx_fast(tt, ctx, names):
    """Same output as TextTower.encode_ctx, but the class-independent prefix [SOT ctx] is computed
    once and shared: under the causal mask its states do not depend on the class name."""
    sd, E = tt.sd, tt.sd['token_embedding.weight']
    M = ctx.shape[0]; P = 1 + M
    suf = [TOK.encode(n + '.') + [EOT] for n in names]
    S = max(len(s) for s in suf); C_ = len(names)
    pos = sd['positional_embedding']
    xp = (torch.cat([E[torch.tensor([SOT])], ctx], 0) + pos[:P])[None]          # 1,P,512
    ids = torch.tensor([s + [0] * (S - len(s)) for s in suf])
    xs = E[ids] + pos[P:P + S]                                                    # C,S,512
    eot = torch.tensor([len(s) - 1 for s in suf])
    mp = torch.full((P, P), float('-inf')).triu(1)
    ms = torch.cat([torch.zeros(S, P), torch.full((S, S), float('-inf')).triu(1)], 1)
    sh = lambda t, B, T: t.view(B, T, 8, 64).transpose(1, 2)
    for i in range(tt.L):
        qp, kp, vp = _layer_qkv(sd, i, xp); qs, ks, vs = _layer_qkv(sd, i, xs)
        ap_ = F.scaled_dot_product_attention(sh(qp, 1, P), sh(kp, 1, P), sh(vp, 1, P), attn_mask=mp)
        ap_ = ap_.transpose(1, 2).reshape(1, P, 512)
        K = torch.cat([kp.expand(C_, P, 512), ks], 1); V = torch.cat([vp.expand(C_, P, 512), vs], 1)
        as_ = F.scaled_dot_product_attention(sh(qs, C_, S), sh(K, C_, P + S), sh(V, C_, P + S), attn_mask=ms)
        as_ = as_.transpose(1, 2).reshape(C_, S, 512)
        xp = _layer_rest(sd, i, xp, ap_, tt.act); xs = _layer_rest(sd, i, xs, as_, tt.act)
    x = F.layer_norm(xs, (512,), sd['ln_final.weight'], sd['ln_final.bias'])
    return x[torch.arange(C_), eot] @ sd['text_projection']
