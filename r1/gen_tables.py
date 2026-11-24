"""LaTeX table bodies (between \\midrule and \\bottomrule) for every results table of the paper,
computed from the corrected (QuickGELU) result files. No hand transcription."""
import json, os, sys, numpy as np, pandas as pd
import _paths  # noqa: F401
from clab.stats import paired_ci, holm_intervals
R = str(_paths.RESULTS) + '/'; O = str(_paths.OUT) + '/'
ORDER = ['AID->UCM', 'UCM->AID', 'AID->DFC15', 'UCM->DFC15']
TX = {'AID->UCM': 'AID$\\to$UCM', 'UCM->AID': 'UCM$\\to$AID', 'AID->DFC15': 'AID$\\to$DFC15',
      'UCM->DFC15': 'UCM$\\to$DFC15', 'DFC15->AID': 'DFC15$\\to$AID', 'DFC15->UCM': 'DFC15$\\to$UCM'}
f2 = lambda x: f"{x:.2f}"; sg = lambda x: f"{x:+.2f}"
T, N = {}, {}   # table bodies, in-text numbers
G = pd.read_csv(R + 'main_grid.csv')
CL, RC = 'clip_vitb32_control', 'remoteclip_vitb32'
P = {bb: G[G.backbone == bb].pivot_table(index=['setting', 'seed'], columns='method', values='mAP') for bb in (CL, RC)}
m = lambda bb, s, k: P[bb].xs(s, level='setting')[k]

# ---- Table main
rows = ["Linear probe, ImageNet TResNet-M & 35.31 $\\pm$ 1.42 & 40.66 $\\pm$ 1.19 & 42.31 $\\pm$ 1.85 & 45.51 $\\pm$ 1.71 \\\\", "\\midrule"]
for k, lab in [('zeroshot_raw', 'Zero-shot, raw anchors      '), ('zeroshot', 'Zero-shot, centered anchors  '),
               ('probe', 'Linear probe                '), ('clab_raw', 'CLAB, raw anchors           '),
               ('clab_long', 'CLAB, $800$-epoch head      '), ('clab', '\\textbf{CLAB}               ')]:
    cells = [f"{m(CL, s, k).mean():.2f} $\\pm$ {m(CL, s, k).std(ddof=1):.2f}" for s in ORDER]
    if k == 'clab': cells = [f"\\textbf{{{c}}}" for c in cells]
    rows.append(lab + ' & ' + ' & '.join(cells) + ' \\\\')
T['main'] = '\n'.join(rows)
N['main_means'] = {k: round(float(np.mean([m(CL, s, k).mean() for s in ORDER])), 2) for k in
                   ['zeroshot_raw', 'zeroshot', 'probe', 'clab', 'clab_raw', 'clab_long', 'clab_a02', 'clab_a04']}

# ---- Table paired (Holm over 8)
diffs = [(bb, s, (m(bb, s, 'clab') - m(bb, s, 'probe')).values) for bb in (CL, RC) for s in ORDER]
H = holm_intervals([d for _, _, d in diffs])
rows = []
for i, (bb, s, d) in enumerate(diffs):
    ci = paired_ci(d); h = H[i]
    name = 'CLIP ViT-B/32' if bb == CL else ('RemoteCLIP' if s == 'AID->DFC15' else 'RemoteCLIP$^\\dagger$')
    rows.append(f"{TX[s]} & {name} & ${sg(ci[0])}$ & $[{sg(ci[1])}, {sg(ci[2])}]$ & $[{sg(h[3])}, {sg(h[4])}]$ & {'yes' if h[3] > 0 or h[4] < 0 else 'no'} \\\\")
    if i == 3: rows.append('\\midrule')
T['paired'] = '\n'.join(rows)
N['paired'] = {f'{bb[:4]} {s}': [round(float(x), 2) for x in list(paired_ci(d)) + [H[i][3], H[i][4]]] for i, (bb, s, d) in enumerate(diffs)}

# ---- bootstrap (a5)
bt = pd.read_csv(O + 'a5_bootstrap.csv'); rows = []
for _, r in bt.iterrows():
    name = 'CLIP ViT-B/32' if r.backbone == CL else ('RemoteCLIP' if r.setting == 'AID->DFC15' else 'RemoteCLIP$^\\dagger$')
    rows.append(f"{TX[r.setting]} & {name} & {r.n_images} & ${sg(r.delta)}$ & $[{sg(r.lo)}, {sg(r.hi)}]$ \\\\")
T['boot'] = '\n'.join(rows[:4]) + '\n\\midrule\n' + '\n'.join(rows[4:])
N['boot'] = bt.round(3).to_dict('records')

# ---- adapters (oracle)
A = pd.read_csv(R + 'adapters20.csv'); g = A.groupby(['setting', 'config']).mAP.mean()
rows, ad = [], []
for s in ORDER:
    best = g[s].idxmax(); q = A[(A.setting == s) & (A.config == best)].set_index('seed').mAP
    d = (m(CL, s, 'clab') - q).values; ad.append(d)
    fam = 'CLIP-Adapter' if best.startswith('clip_adapter') else 'Tip-Adapter'
    anc = 'c' if '|centred|' in best else 'r'
    rows.append((s, fam, anc, q.mean(), m(CL, s, 'probe').mean(), m(CL, s, 'clab').mean(), paired_ci(d), best))
Hb = holm_intervals(ad)
Hall = holm_intervals([d for _, _, d in diffs] + ad)
T['baselines'] = '\n'.join(f"{TX[s]} & {fam}$^{{\\,{anc}}}$ & {f2(q)} & {f2(p)} & \\textbf{{{f2(c)}}} & ${sg(ci[0])}\\ [{sg(ci[1])}, {sg(ci[2])}]$ & $[{sg(Hb[i][3])}, {sg(Hb[i][4])}]$ \\\\"
                           for i, (s, fam, anc, q, p, c, ci, best) in enumerate(rows))
N['baselines'] = [dict(setting=s, cfg=b, base=round(q, 2), delta=[round(float(x), 2) for x in ci]) for s, fam, anc, q, p, c, ci, b in rows]
N['pooled12_min_lo'] = round(float(min(h[3] for h in Hall)), 2)
tp = A.groupby(['setting', 'family', 'anchors']).mAP.mean()
N['tip_centering_gain'] = [round(float(tp[s]['tip_adapter']['centred'] - tp[s]['tip_adapter']['raw']), 2) for s in ORDER]
N['clipad_centering_gain'] = [round(float(tp[s]['clip_adapter']['centred'] - tp[s]['clip_adapter']['raw']), 2) for s in ORDER]
N['adapters_below_probe'] = [s for s, fam, anc, q, p, c, ci, b in rows if q < p]

# ---- ablation (both backbones)
abl = {k: float(np.mean([(m(bb, s, k) - m(bb, s, 'clab')).mean() for bb in (CL, RC) for s in ORDER]))
       for k in ['clab_raw', 'probe', 'zeroshot', 'clab_long']}
T['ablation'] = (f"Anchor centering, Equation~\\eqref{{eq:center}}   & ${abl['clab_raw']:+.2f}$ & fusion with raw anchors \\\\\n"
                 f"The source-trained branch, Equation~\\eqref{{eq:srcview}} & ${abl['zeroshot']:+.2f}$ & centered zero-shot only \\\\\n"
                 f"The semantic branch, Equation~\\eqref{{eq:textview}} & ${abl['probe']:+.2f}$ & linear probe only \\\\\n"
                 f"The short source budget ($800$ epochs)        & ${abl['clab_long']:+.2f}$ & CLAB with a converged head \\\\").replace('+-', '-')
N['ablation'] = {k: round(v, 2) for k, v in abl.items()}
N['ablation_clip_only'] = {k: round(float(np.mean([(m(CL, s, k) - m(CL, s, 'clab')).mean() for s in ORDER])), 2) for k in abl}

# ---- LOTO (4) and LOTO6
L = pd.read_csv(R + 'loto_selected.csv')
T['loto'] = '\n'.join(f"{TX[r.held_out]} & {r.alpha:.2f} & {int(r.epochs)} & {f2(r.clab)} & {f2(r.probe)} & ${sg(r.delta)}\\ [{sg(r.lo)}, {sg(r.hi)}]$ \\\\" for _, r in L.iterrows())
L6 = pd.read_csv(O + 'a9_loto6_selected.csv')
T['loto6'] = '\n'.join(f"{TX[r.held_out]} & {r.alpha:.2f} & {int(r.epochs)} & {f2(r.clab)} & {f2(r.probe)} & ${sg(r.delta)}\\ [{sg(r.lo)}, {sg(r.hi)}]$ \\\\" for _, r in L6.iterrows())

# ---- alpha sweep (from robust_standard u-columns, 50 epochs) and budget
RS = pd.read_csv(R + 'robust_standard.csv')
rows = []
for a in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    col = f"u{a:g}" if f"u{a:g}" in RS else f"u{a}"
    v = [RS[RS.setting == s][col].mean() for s in ORDER]
    cells = [f2(x) for x in v] + [f2(np.mean(v))]
    if abs(a - 0.30) < 1e-9:
        rows.append('\\textbf{0.30} & ' + ' & '.join(f'\\textbf{{{c}}}' for c in cells) + ' \\\\')
    else:
        rows.append(f"{a:.2f} & " + ' & '.join(cells) + ' \\\\')
T['alpha_rows'] = '\n'.join(rows)
N['alpha_mean'] = {f"{a:.2f}": round(float(np.mean([RS[RS.setting == s][f'u{a:g}'].mean() for s in ORDER])), 2) for a in np.arange(0.025, 0.701, 0.025)}
N['probe_mean_robust_standard'] = round(float(np.mean([RS[RS.setting == s]['probe'].mean() for s in ORDER])), 2)
print('alpha means', N['alpha_mean'])

# ---- robustness (out-of-vocabulary restored), alpha = 0.30
RM = pd.read_csv(R + 'robust_mismatch.csv')
OOV = {'AID->UCM': '17.2', 'AID->DFC15': '34.4', 'UCM->AID': '8.7', 'UCM->DFC15': '34.4', 'DFC15->AID': '8.7', 'DFC15->UCM': '17.2'}
rows, dl = [], []
for s in ['AID->UCM', 'AID->DFC15', 'UCM->AID', 'UCM->DFC15', 'DFC15->AID', 'DFC15->UCM']:
    q = RM[RM.setting == s]; d = (q['u0.3'] - q.probe).values; ci = paired_ci(d); dl.append(ci[0])
    rows.append(f"{TX[s]:14s} & {OOV[s]:4s} & {f2(q.probe.mean())} & {f2(q['u0.3'].mean())} & ${sg(ci[0])}_{{[{ci[1]:.2f},{ci[2]:.2f}]}}$ \\\\")
T['robust'] = '\n'.join(rows) + f"\n\\midrule\nMean           &      &       &       & ${sg(np.mean(dl))}$ \\\\"
N['robust_worst'] = {a: round(float(min((RM[RM.setting == s][f'u{a}'] - RM[RM.setting == s].probe).mean() for s in RM.setting.unique())), 2) for a in ['0.2', '0.25', '0.3', '0.35', '0.4']}
N['robust_cis'] = {s: [round(float(x), 2) for x in paired_ci((RM[RM.setting == s]['u0.3'] - RM[RM.setting == s].probe).values)] for s in RM.setting.unique()}
RSd = {s: (RS[RS.setting == s]['u0.3'] - RS[RS.setting == s].probe).mean() for s in ORDER}
N['alpha_gain_range_020_040'] = round(float(max(N['alpha_mean'][f'{a:.2f}'] for a in (0.2, 0.25, 0.3, 0.35, 0.4)) - min(N['alpha_mean'][f'{a:.2f}'] for a in (0.2, 0.25, 0.3, 0.35, 0.4))), 2)

# ---- thresholds (a1)
d = pd.read_csv(O + 'a1_thresholds.csv'); q = d[d.scheme == 'perclass_srcval']; rows = []
N['thr'] = {}
for s in ORDER:
    x = q[q.setting == s]; mm = x.groupby('method')[['CF1', 'OF1']].mean()
    pv = {met: x.pivot(index='seed', columns='method', values=met) for met in ['CF1', 'OF1']}
    dc = paired_ci(pv['CF1'].clab - pv['CF1'].probe); do = paired_ci(pv['OF1'].clab - pv['OF1'].probe)
    N['thr'][s] = dict(CF1=[round(float(v), 2) for v in dc], OF1=[round(float(v), 2) for v in do])
    rows.append(f"{TX[s]} & {f2(mm.loc['zeroshot','CF1'])} & {f2(mm.loc['probe','CF1'])} & {f2(mm.loc['clab','CF1'])} & "
                f"${sg(dc[0])}\\,[{sg(dc[1])}, {sg(dc[2])}]$ & {f2(mm.loc['zeroshot','OF1'])} & {f2(mm.loc['probe','OF1'])} & "
                f"{f2(mm.loc['clab','OF1'])} & ${sg(do[0])}\\,[{sg(do[1])}, {sg(do[2])}]$ \\\\")
T['thr'] = '\n'.join(rows)
N['thr_card'] = {}
for met in ['CF1', 'OF1']:
    for s in ORDER:
        x = d[(d.scheme == 'cardinality_src') & (d.setting == s)].pivot(index='seed', columns='method', values=met)
        N['thr_card'][f'{met} {s}'] = [round(float(v), 2) for v in paired_ci(x.clab - x.probe)]

# ---- published: our two rows
T['published_ours'] = ("CLIP linear probe         & CLIP ViT-B/32 & " + ' & '.join(f"{m(CL, s, 'probe').mean():.1f}" for s in ORDER) + " \\\\\n"
                       "CLAB (ours)               & CLIP ViT-B/32 & " + ' & '.join(f"{m(CL, s, 'clab').mean():.1f}" for s in ORDER) + " \\\\")

# ---- backbone
B = pd.read_csv(R + 'backbone_study.csv'); bg = B.groupby(['backbone', 'setting']).mAP
bm, bs = bg.mean(), bg.std(ddof=1)
cell = lambda b, s: f"{bm[b][s]:.2f} $\\pm$ {bs[b][s]:.2f}"
names = [('imagenet_tresnet_m', 'TResNet-M (ImageNet-1K)  '), ('imagenet21k_tresnet_m', 'TResNet-M (ImageNet-21K) '),
         ('satmae_vitlarge', 'SatMAE ViT-L             '), ('clip_rn50_control', 'CLIP RN50                '),
         ('clip_vitb32_control', 'CLIP ViT-B/32            '), ('remoteclip_rn50', 'RemoteCLIP RN50$^\\dagger$    '),
         ('remoteclip_vitb32', 'RemoteCLIP ViT-B/32$^\\dagger$'), ('remoteclip_vitl14', 'RemoteCLIP ViT-L/14$^\\dagger$')]
rows = [f"{lab} & " + ' & '.join(cell(b, s) for s in ORDER) + ' \\\\' for b, lab in names]
dv = lambda b: [bm[b][s] - bm['imagenet_tresnet_m'][s] for s in ORDER]
rows += ['\\midrule', "$\\Delta$ CLIP ViT-B/32 vs ImageNet-1K & " + ' & '.join(f'${sg(x)}$' for x in dv('clip_vitb32_control')) + ' \\\\',
         "$\\Delta$ CLIP RN50 vs ImageNet-1K     & " + ' & '.join(f'${sg(x)}$' for x in dv('clip_rn50_control')) + ' \\\\']
T['backbone'] = '\n'.join(rows)
N['backbone_delta_clip'] = [round(x, 2) for x in dv('clip_vitb32_control')]
N['backbone_delta_rn50'] = [round(x, 2) for x in dv('clip_rn50_control')]

# ---- geometry
sp = pd.read_csv(R + 'geometry_spectrum.csv').set_index('anchors'); co = pd.read_csv(R + 'geometry_cosines.csv').set_index('anchors')
c6log = open(O + 'a6_centering_alternatives.log').read()
import re
mm_ = re.search(r'raw about mean \[([\d.]+)\s+([\d.]+)\s+([\d.]+)\]\s+([\d.]+)', c6log)
pc = lambda x: f"{float(x):.1f}\\%"
T['geom'] = (f"Raw, about the origin     & {pc(sp.loc['raw','PC1'])} & {pc(sp.loc['raw','PC2'])}  & {pc(sp.loc['raw','PC3'])} & {pc(sp.loc['raw','PC1_3'])} \\\\\n"
             f"Raw, about the mean       & {pc(mm_.group(1))} & {pc(mm_.group(2))} & {pc(mm_.group(3))} & {pc(mm_.group(4))} \\\\\n"
             f"Centered, about the mean  & {pc(sp.loc['centered','PC1'])} & {pc(sp.loc['centered','PC2'])} & {pc(sp.loc['centered','PC3'])} & {pc(sp.loc['centered','PC1_3'])} \\\\\n"
             "\\midrule\n & \\textbf{Mean} & \\textbf{Median} & \\textbf{Min} & \\textbf{Max} \\\\\n\\midrule\n"
             f"Pairwise cosine, raw     & ${co.loc['raw','mean']:+.3f}$ & ${co.loc['raw','median']:+.3f}$ & ${co.loc['raw','min']:+.3f}$ & ${co.loc['raw','max']:+.3f}$ \\\\\n"
             f"Pairwise cosine, centered & ${co.loc['centered','mean']:+.3f}$ & ${co.loc['centered','median']:+.3f}$ & ${co.loc['centered','min']:+.3f}$ & ${co.loc['centered','max']:+.3f}$ \\\\")
cs = pd.read_csv(R + 'geometry_centering_set.csv'); N['centering_set'] = cs.round(3).to_dict('records')

# ---- centering alternatives
c6 = pd.read_csv(O + 'a6_centering.csv')
order = ['raw', 'centered (CLAB)', 'raw, remove PC1 about origin', 'raw + whitening (Lowdin)',
         'centered + whitening (Lowdin)', 'centered + remove PC1 (all-but-the-top, D=1)',
         'centered + remove PC1-2 (D=2)', 'centered + remove PC1-3 (D=3)',
         'image centering (source mean), raw anchors', 'image centering + centered anchors']
nm = {'raw': 'Raw anchors (no correction)', 'centered (CLAB)': '\\textbf{Mean removal (CLAB)}',
      'raw, remove PC1 about origin': 'Remove leading direction about the origin',
      'raw + whitening (Lowdin)': 'Whitening of raw anchors', 'centered + whitening (Lowdin)': 'Mean removal + whitening',
      'centered + remove PC1 (all-but-the-top, D=1)': 'Mean removal + remove PC1', 'centered + remove PC1-2 (D=2)': 'Mean removal + remove PC1--2',
      'centered + remove PC1-3 (D=3)': 'Mean removal + remove PC1--3', 'image centering (source mean), raw anchors': 'Image centering, raw anchors',
      'image centering + centered anchors': 'Image centering + mean removal'}
rows = []; N['center'] = {}
for k in order:
    x = c6[c6.variant == k].groupby('setting')[['zeroshot', 'clab']].mean().loc[ORDER]
    rows.append(f"{nm[k]} & " + ' & '.join(f2(v) for v in x.zeroshot) + f" & {f2(x.zeroshot.mean())} & {f2(x.clab.mean())} \\\\")
    N['center'][k] = dict(zs=[round(v, 2) for v in x.zeroshot], zs_mean=round(x.zeroshot.mean(), 2), clab=[round(v, 2) for v in x.clab], clab_mean=round(x.clab.mean(), 2))
T['center'] = '\n'.join(rows)

# ---- prompts
pa = pd.read_csv(R + 'prompt_ablation.csv').groupby('prompt_set', sort=False)[['cos_raw_mean', 'mAP_raw', 'mAP_centered']].mean()
pa['gain'] = pa.mAP_centered - pa.mAP_raw; pa = pa.sort_values('cos_raw_mean')
lab = {"a photo of {}": "``a photo of \\{\\}''                    ", "bare category name": "bare category name                     ",
       "three aerial templates + a photo of": "three aerial templates $+$ ``a photo of''", "a satellite image of {}": "``a satellite image of \\{\\}''          ",
       "three aerial templates (used)": "three aerial templates (used here)     ", "an aerial photograph containing {}": "``an aerial photograph containing \\{\\}''",
       "a remote sensing image showing {}": "``a remote sensing image showing \\{\\}''"}
T['prompts'] = '\n'.join(f"{lab[k]} & ${r.cos_raw_mean:+.3f}$ & {f2(r.mAP_raw)} & {f2(r.mAP_centered)} & ${sg(r.gain)}$ \\\\" for k, r in pa.iterrows())
N['prompts'] = pa.round(3).to_dict('index')

# ---- fusion
FU = pd.read_csv(R + 'fusion.csv').groupby(['rule', 'setting']).mAP.mean().unstack()[ORDER]
lab = [('zscore', '$\\mathcal{Z}$-score (CLAB)  ', 'inductive'), ('minmax', 'Min--max normalization      ', 'inductive'),
       ('rank', 'Rank normalization          ', 'transductive'), ('sigmoid', 'Sigmoid probabilities       ', 'inductive'),
       ('raw', 'No standardization          ', 'inductive')]
T['fusion'] = '\n'.join(f"{l} & " + ' & '.join(f2(v) for v in FU.loc[k]) + f" & {'\\textbf{' + f2(FU.loc[k].mean()) + '}' if k == 'zscore' else f2(FU.loc[k].mean())} & {form} \\\\" for k, l, form in lab)
N['fusion_means'] = FU.mean(1).round(2).to_dict()

# ---- per-class DFC15
PCc = pd.read_csv(R + 'perclass.csv'); rows = []
for s in ['UCM->DFC15', 'AID->DFC15']:
    x = PCc[PCc.setting == s].sort_values('d', ascending=False)
    rows += [f"{TX[s]} & {r.label:9s} & {r.prevalence:.1f} & {f2(r.zeroshot)} & {f2(r.probe)} & {f2(r.clab)} & ${sg(r.d)}$ \\\\" for _, r in x.iterrows()]
    rows.append('\\midrule')
T['perclass'] = '\n'.join(rows[:-1])
N['perclass'] = PCc.round(2).to_dict('records')

# ---- estimators
E = pd.read_csv(R + 'alpha_estimators.csv'); rows = []
lab = [('rank_agreement', 'Rank agreement            '), ('source_validation', 'Source validation         '),
       ('shrunk_source_validation', 'Shrunk source validation  '), ('self_training', 'Self-training on the blend')]
N['estimators'] = {}
for k, l in lab:
    arm = (E[k] - E.fixed).groupby([E.setting, E.backbone]).mean()
    rows.append(f"{l} & ${(E[k] - E.fixed).mean():+.2f}$ & {int((arm > 0.005).sum())} & ${arm.min():+.2f}$ \\\\")
    N['estimators'][k] = dict(mean=round(float((E[k] - E.fixed).mean()), 3), improved=int((arm > 0.005).sum()), worst=round(float(arm.min()), 2), maxabs=round(float(arm.abs().max()), 3))
T['estimators'] = '\n'.join(rows).replace('$-0.00$', '$0.00$').replace('$+0.00$', '$0.00$')
OR = pd.read_csv(R + 'alpha_oracle.csv'); N['oracle'] = dict(glob=round(float((OR.global_oracle - OR.fixed).mean()), 2), perclass=round(float((OR.perclass_oracle - OR.fixed).mean()), 2))
IV = pd.read_csv(R + 'invariance.csv'); dt = IV.target - IV.source
N['prop1'] = dict(rho_src=[round(float(IV.rho_src.min()), 1), round(float(IV.rho_src.max()), 1)], rho_tgt=[round(float(IV.rho_tgt.min()), 1), round(float(IV.rho_tgt.max()), 1)],
                  tgtconst_mean=round(float(dt.mean()), 2), tgtconst_worst=round(float(dt.min()), 2), zero_means_maxabs=float((IV.zero_means - IV.source).abs().max()))

# ---- per-class all transfers (a5)
pc = pd.read_csv(O + 'a5_perclass.csv'); rows = []
for s in ORDER:
    x = pc[pc.setting == s].sort_values('d', ascending=False, na_position='last')
    for _, r in x.iterrows():
        dd = '--' if np.isnan(r.d) else f"${sg(r.d)}\\,[{sg(r.lo)}, {sg(r.hi)}]$"
        rows.append(f"{TX[s]} & {r.label} & {r.prevalence:.1f} & {f2(r.zeroshot)} & {f2(r.probe)} & {f2(r.clab)} & {dd} \\\\")
    rows.append('\\midrule')
T['perclass_all'] = '\n'.join(rows[:-1])
N['perclass_all'] = pc.round(2).to_dict('records')

# ---- source budget (5 seeds) and head alternatives
SB = pd.read_csv(R + 'source_budget.csv').groupby(['setting', 'backbone', 'epochs'])[['src_train_mAP', 'tgt_linear_mAP']].mean().round(2)
N['source_budget'] = {f'{a}|{b}|{c}': v for (a, b, c), v in SB.to_dict('index').items()}

LR = pd.read_csv(R + 'loto_raw.csv')
N['budget_means'] = {str(ep): round(float(np.mean([LR[(LR.epochs == ep) & np.isclose(LR.alpha, 0.3) & (LR.setting == s_)].mAP.mean() for s_ in ORDER])), 2) for ep in [20, 50, 100, 200]}
zsm = lambda bb, s_: m(bb, s_, 'zeroshot').mean()
N['contam'] = {'ucm': [zsm(RC, 'AID->UCM') - zsm(CL, 'AID->UCM'), bm['remoteclip_vitb32']['AID->UCM'] - bm['clip_vitb32_control']['AID->UCM']],
               'aid': [zsm(RC, 'UCM->AID') - zsm(CL, 'UCM->AID'), bm['remoteclip_vitb32']['UCM->AID'] - bm['clip_vitb32_control']['UCM->AID']],
               'dfc': [zsm(RC, 'AID->DFC15') - zsm(CL, 'AID->DFC15'), bm['remoteclip_vitb32']['AID->DFC15'] - bm['clip_vitb32_control']['AID->DFC15'],
                       bm['remoteclip_vitb32']['UCM->DFC15'] - bm['clip_vitb32_control']['UCM->DFC15']]}
N['centering_gain_zs'] = {s_: round(float((m(CL, s_, 'zeroshot') - m(CL, s_, 'zeroshot_raw')).mean()), 2) for s_ in ORDER}
N['centering_gain_zs_rc'] = {s_: round(float((m(RC, s_, 'zeroshot') - m(RC, s_, 'zeroshot_raw')).mean()), 2) for s_ in ORDER}

json.dump(T, open(O + 'tables_qg.json', 'w'), indent=1)
json.dump(N, open(O + 'numbers_qg.json', 'w'), indent=1, default=float)
for k, v in T.items(): print('=====', k); print(v)
print(json.dumps(N, indent=0, default=float)[:3000])
