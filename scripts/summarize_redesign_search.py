import json,csv
from pathlib import Path
import numpy as np
from develop_phgeofuse_regression import OUT,metrics
D=np.load(OUT/'development_features.npz');yv=D['yv'];low=D['lowv'];r={r['key']:r for r in csv.DictReader((OUT/'baseline_validation.csv').open())}
b=np.array([float(r[k]['prediction']) for k in D['keys'][7124:]])
files=list(OUT.glob('val_g*_a*.npy'))+list((OUT/'esm2_exploratory').glob('val_g*_a*.npy'))
a=[]
for p in files:
 pred=np.load(p)
 for w in [.25,.5,.75,1.]:
  m=metrics(yv,w*pred+(1-w)*b,low)
  a.append({'source':str(p.relative_to(OUT)),'weight':w,'metrics':m})
a.sort(key=lambda x:x['metrics']['rmse']);(OUT/'blends_corrected.json').write_text(json.dumps(a,indent=2))
print(json.dumps(a[:3],indent=2))
# Repair group metric reporting from the initial search, whose boundary differed at pH=8.
search=json.loads((OUT/'search.json').read_text())
for v in search:
 v['validation']=metrics(yv,np.load(OUT/f"val_g{v['gamma']}_a{v['alpha']}.npy"),low)
(OUT/'search.json').write_text(json.dumps(search,indent=2))
(OUT/'best_validation.json').write_text(json.dumps(min(search,key=lambda r:r['validation']['rmse']),indent=2))
