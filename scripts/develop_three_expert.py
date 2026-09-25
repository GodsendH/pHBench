import csv,json,numpy as np
from develop_phgeofuse_regression import OUT,metrics
D=np.load(OUT/'development_features.npz');y=D['yv'];low=D['lowv'];r={r['key']:r for r in csv.DictReader((OUT/'baseline_validation.csv').open())};b=np.array([float(r[k]['prediction']) for k in D['keys'][7124:]])
k=np.load(OUT/'esm2_exploratory/val_g0_a0.01.npy');records=[]
for power in [.25,.5]:
 residual=np.load(OUT/f'retrieval_residual/val_weight{power}_i150.npy')
 for wk,wr in [(.25,.25),(.25,.5),(.5,.25),(0,.25),(0,.5)]:
  p=(1-wk-wr)*b+wk*k+wr*residual
  records.append({'esm2_weight':wk,'residual_weight':wr,'residual_power':power,'validation':metrics(y,p,low)})
records.sort(key=lambda r:r['validation']['rmse']);(OUT/'three_expert_exploratory.json').write_text(json.dumps(records,indent=2));print(json.dumps(records[:3],indent=2))
