import csv,json,numpy as np
from sklearn.model_selection import KFold
from develop_phgeofuse_regression import OUT,metrics
D=np.load(OUT/'development_features.npz');y=D['yv'];low=D['lowv'];r={r['key']:r for r in csv.DictReader((OUT/'baseline_validation.csv').open())};b=np.array([float(r[k]['prediction']) for k in D['keys'][7124:]])
records=[]
for label,p in [('baseline',b),('esm2_blend',.75*b+.25*np.load(OUT/'esm2_exploratory/val_g0.25_a0.1.npy'))]:
 a=np.column_stack([p,np.ones(len(p))]);oof=np.zeros(len(p))
 for tr,te in KFold(5,shuffle=True,random_state=42).split(a):
  coef=np.linalg.lstsq(a[tr],y[tr],rcond=None)[0];oof[te]=a[te]@coef
 coef=np.linalg.lstsq(a,y,rcond=None)[0]
 records.append({'expert':label,'coef':coef.tolist(),'uncalibrated':metrics(y,p,low),'calibrated_oof':metrics(y,oof,low)})
print(json.dumps(records,indent=2));(OUT/'calibration_development.json').write_text(json.dumps(records,indent=2))
