import pandas as pd, numpy as np, sys, json
import _paths; from clab.stats import paired_ci
O=str(_paths.OUT) + '/'; ORDER=['AID->UCM','UCM->AID','AID->DFC15','UCM->DFC15']
X={}
h=pd.read_csv(O+'a2_heads.csv'); g=h.groupby(['head','param','setting'])[['probe','clab']].mean()
for fam in ['gd3000','ridge']:
    res=[]
    for s in ORDER:
        others=[o for o in ORDER if o!=s]
        cand=g.loc[fam]
        best=max(cand.index.get_level_values(0).unique(), key=lambda p: np.mean([cand.loc[(p,o),'clab'] for o in others]))
        res.append((s,best,cand.loc[(best,s),'clab']))
    X[f'loto_{fam}']=dict(params=[r[1] for r in res], mean_clab=round(float(np.mean([r[2] for r in res])),2))
wd=h[(h['head']=='gd3000')&(np.isclose(h.param,0.1))].set_index(['setting','seed']); gd=h[h['head']=='gd50'].set_index(['setting','seed'])
X['wd01_minus_gd50_probe']={s: round(float((wd.loc[s].probe-gd.loc[s].probe).mean()),2) for s in ORDER}
a3=pd.read_csv(O+'a3_perclass_z.csv')
X['perclassZ']={s:[round(float(v),2) for v in paired_ci((a3[a3.setting==s].clab_perclass-a3[a3.setting==s].clab_global).values)] for s in ORDER}
X['rho_cv']=[round(float(a3.rho_spread.min()),2), round(float(a3.rho_spread.max()),2)]
X['rho_cv_by_setting']=a3.groupby('setting').rho_spread.mean().round(2).to_dict()
cf=pd.read_csv(O+'a11_confusion.csv'); q=cf[cf.setting.isin(['AID->DFC15','UCM->DFC15'])]
fpr=q.groupby(['setting','method'])[['FP','TN']].sum(); fpr=(fpr.FP/(fpr.FP+fpr.TN)).round(3)
X['fpr_dfc']=fpr.to_dict()
fa=cf.groupby(['setting','method'])[['FP','TN']].sum(); X['fpr_all']=((fa.FP/(fa.FP+fa.TN)).round(3)).to_dict()
json.dump({k:(v if not isinstance(v,dict) else {str(a):b for a,b in v.items()}) for k,v in X.items()},open(O+'extra_numbers.json','w'),indent=1,default=float)
print(json.dumps({k:(v if not isinstance(v,dict) else {str(a):b for a,b in v.items()}) for k,v in X.items()},indent=0,default=float))
