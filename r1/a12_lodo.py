"""R2-4: leave-one-DATASET-out selection, stricter than leave-one-transfer-out: for each held-out
transfer, (alpha, budget) is chosen only on the transfers that do not involve its target dataset
at all (the two transfers between the other two datasets), then applied to it."""
import _paths  # noqa: F401  (repository paths; must come before clab)
import os, sys, numpy as np, pandas as pd
import _paths  # noqa: F401
from clab.stats import paired_ci
O = os.environ.get('R1_OUT', str(_paths.OUT)) + '/'
d = pd.read_csv(O + 'a9_loto6_raw.csv')
g = d.groupby(['setting', 'epochs', 'alpha']).mAP.mean().reset_index()
ds = lambda s: s.split('->')
rows = []
for held in d.setting.unique():
    tgt = ds(held)[1]
    pool = [s for s in d.setting.unique() if tgt not in ds(s)]
    ep, al = g[g.setting.isin(pool)].groupby(['epochs', 'alpha']).mAP.mean().idxmax()
    q = d[(d.setting == held) & (d.epochs == ep) & np.isclose(d.alpha, al)].sort_values('seed')
    m_, lo, hi = paired_ci(q.mAP.values - q.linear.values)
    rows.append(dict(held_out=held, selected_on=' + '.join(pool), alpha=al, epochs=int(ep), clab=q.mAP.mean(), probe=q.linear.mean(), delta=m_, lo=lo, hi=hi))
r = pd.DataFrame(rows); r.to_csv(O + 'a12_lodo.csv', index=False); print(r.round(2).to_string())
