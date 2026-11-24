"""Rows and numbers for the prompt-learning baselines (Table 5) and the UDA re-runs (Table 13)."""
import json, sys, glob, numpy as np, pandas as pd
import _paths  # noqa: F401
from clab.stats import paired_ci, holm_intervals
O = str(_paths.OUT) + '/'; R = str(_paths.RESULTS) + '/'
ORDER = ['AID->UCM', 'UCM->AID', 'AID->DFC15', 'UCM->DFC15']
TX = {'AID->UCM': 'AID$\\to$UCM', 'UCM->AID': 'UCM$\\to$AID', 'AID->DFC15': 'AID$\\to$DFC15', 'UCM->DFC15': 'UCM$\\to$DFC15'}
f2 = lambda x: f"{x:.2f}"; sg = lambda x: f"{x:+.2f}"
G = pd.read_csv(R + 'main_grid.csv'); G = G[G.backbone == 'clip_vitb32_control'].pivot_table(index=['setting', 'seed'], columns='method', values='mAP')
T, N = {}, {}
# ---- prompt learning (10 seeds)
F = pd.concat([pd.read_csv(f) for f in glob.glob(O + 'c2_final_*.csv')])
rows, diffs = [], []
for kind, lab in [('coop', 'CoOp-sigmoid'), ('dual', 'DualCoOp-global')]:
    for s in ORDER:
        x = F[(F.kind == kind) & (F.setting == s)].set_index('seed').mAP.sort_index()
        c = G.xs(s, level='setting').loc[x.index]
        d = (c.clab - x).values; diffs.append(d)
        cfg = F[(F.kind == kind) & (F.setting == s)].iloc[0]
        anc = 'c' if cfg.anchors == 'mAP_centered' else 'r'
        rows.append(dict(kind=kind, lab=lab, s=s, base=x.mean(), probe=c.probe.mean(), clab=c.clab.mean(), ci=paired_ci(d), anc=anc,
                         steps=int(cfg.steps), grid=int(cfg.grid), n=len(x)))
H = holm_intervals(diffs[:4]) + holm_intervals(diffs[4:])  # Holm within each family of four
T['pl_rows'] = '\n'.join(f"{TX[r['s']]} & {r['lab']}$^{{\\,{r['anc']}}}$ & {f2(r['base'])} & {f2(r['probe'])} & \\textbf{{{f2(r['clab'])}}} & "
                         f"${sg(r['ci'][0])}\\ [{sg(r['ci'][1])}, {sg(r['ci'][2])}]$ & $[{sg(H[i][3])}, {sg(H[i][4])}]$ \\\\"
                         for i, r in enumerate(rows))
N['pl'] = [dict(kind=r['kind'], setting=r['s'], base=round(r['base'], 2), probe=round(r['probe'], 2), clab=round(r['clab'], 2),
                delta=[round(float(v), 2) for v in r['ci']], holm=[round(float(H[i][3]), 2), round(float(H[i][4]), 2)],
                steps=r['steps'], grid=r['grid'], anc=r['anc'], n=r['n']) for i, r in enumerate(rows)]
# ---- UDA re-runs (20 seeds)
U = pd.read_csv(O + 'c3_uda_ddamlic_dann.csv')
rows = []; N['uda'] = []
for s in ORDER:
    c = G.xs(s, level='setting')
    cells = [f2(c.probe.mean())]
    rec = dict(setting=s, probe=round(c.probe.mean(), 2), clab=round(c.clab.mean(), 2))
    for k in ['ddamlic', 'dann']:
        x = U[(U.method == k) & (U.setting == s)].set_index('seed').sort_index()
        cells += [f2(x['last'].mean()), f2(x.oracle_best.mean())]
        dl = paired_ci((c.clab.loc[x.index] - x['last']).values); db = paired_ci((c.clab.loc[x.index] - x.oracle_best).values)
        rec[k] = dict(last=round(x['last'].mean(), 2), best=round(x.oracle_best.mean(), 2), d_last=[round(float(v), 2) for v in dl],
                      d_best=[round(float(v), 2) for v in db], best_at=float(x.best_at.mean()))
    cells.append('\\textbf{' + f2(c.clab.mean()) + '}')
    rows.append(f"{TX[s]} & " + ' & '.join(cells) + ' \\\\')
    N['uda'].append(rec)
T['uda'] = '\n'.join(rows)
json.dump(T, open(O + 'tables_extra.json', 'w'), indent=1); json.dump(N, open(O + 'numbers_extra.json', 'w'), indent=1, default=float)
print(T['pl_rows']); print(T['uda']); print(json.dumps(N, indent=0, default=float)[:4000])
