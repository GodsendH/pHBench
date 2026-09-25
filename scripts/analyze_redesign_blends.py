import sys,csv,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from develop_phgeofuse_regression import OUT,metrics
D=np.load(OUT/'development_features.npz');y=D['yv'];low=D['lowv'];keys=D['keys'][len(D['y']):]
rows={r['key']:r for r in csv.DictReader((OUT/'baseline_validation.csv').open())};base=np.array([float(rows[k]['prediction']) for k in keys]);print('BASE',metrics(y,base,low))
results=[]
for file in sorted(OUT.glob('val_g*_a*.npy')):
 p=np.load(file)
 for w in [0.25,.5,.75,1.]:
  pred=w*p+(1-w)*base
  results.append({'source':file.name,'weight':w,'metrics':metrics(y,pred,low)})
results.sort(key=lambda x:x['metrics']['rmse']);(OUT/'blend_validation.json').write_text(json.dumps(results,indent=2));print(json.dumps(results[:5],indent=2))
