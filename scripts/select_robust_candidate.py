import json,csv,numpy as np
from develop_phgeofuse_regression import OUT,metrics
from phgeofuse.cache import atomic_json
D=np.load(OUT/'development_features.npz');y=D['yv'];low=D['lowv'];keys=D['keys'][7124:];search=json.loads((OUT/'compact/fusion_search.json').read_text())
# Use no extra validation-fitted affine correction in the first confirmatory candidate.
search=[r for r in search if r['calibration']=='none']
bs={}
for s in [0,1,2,3,42]:
 f=OUT/('baseline_validation.csv' if s==42 else f'baseline_seed{s}_validation.csv');rows={r['key']:r for r in csv.DictReader(f.open())};bs[s]=np.array([float(rows[k]['prediction']) for k in keys])
baselines=[metrics(y,p,low) for p in bs.values()]
def summary(ms):
 return {**{k:float(np.mean([m[k] for m in ms])) for k in ['rmse','mae','r2','spearman']},'low_rmse':float(np.mean([m['low_homology']['rmse'] for m in ms])),'acid_abs_bias':float(np.mean([abs(m['acidic']['bias']) for m in ms])),'alk_abs_bias':float(np.mean([abs(m['alkaline']['bias']) for m in ms]))}
b=summary(baselines);allrows=[]
for c in search:
 seq=np.load(OUT/f"compact/{c['name']}.validation.npy");res=np.load(OUT/f"retrieval_residual/val_weight{c['residual_power']}_i150.npy");w=c['weights']
 ms=[metrics(y,w['baseline']*base+w['sequence']*seq+w['residual']*res,low) for base in bs.values()];su=summary(ms)
 passing=all(su[k]<b[k] for k in ['rmse','mae','low_rmse','acid_abs_bias','alk_abs_bias']) and su['spearman']>=b['spearman']
 allrows.append({'candidate':c,'mean_validation':su,'passes':passing})
allrows.sort(key=lambda r:r['mean_validation']['rmse']);passing=[r for r in allrows if r['passes']]
atomic_json(OUT/'compact/five_seed_validation.json',{'baseline':b,'candidates':allrows})
if passing:
 selected=passing[0];atomic_json(OUT/'compact/frozen_candidate.json',selected);print('FROZEN',json.dumps(selected))
else:print('NO_PASSED',json.dumps(allrows[:2]))
