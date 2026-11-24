"""Fixed-rule tile selection for the example figure: for each target dataset (AID->UCM, UCM->AID,
AID->DFC15; seed 0), the two tiles with the largest gain in per-image F1 of CLAB over stock CLIP
and the tile with the largest loss of CLAB against the probe (ties: first in evaluation order)."""
import _paths  # noqa: F401  (repository paths; must come before clab)
import json, os, sys, pandas as pd
R1 = os.environ.get('R1_OUT', str(_paths.OUT))
ex = pd.read_json(R1 + '/a11_examples.json')
ex['g'] = ex.f1_clab - ex.f1_clip; ex['l'] = ex.f1_probe - ex.f1_clab
out = []
for sn in ['AID->UCM', 'UCM->AID', 'AID->DFC15']:
    q = ex[ex.setting == sn]
    out += q.sort_values('g', ascending=False, kind='stable').head(2).to_dict('records')
    out += q.sort_values('l', ascending=False, kind='stable').head(1).to_dict('records')
json.dump(out, open(R1 + '/a11_picked.json', 'w'), indent=1, default=str)
print([(r['setting'], r['image_id'], round(r['g'], 2), round(r['l'], 2)) for r in out])
