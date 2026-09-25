"""Fit compact experts with freshly encoded features; select only on validation."""
import os
os.environ['OMP_NUM_THREADS']='8';os.environ['OPENBLAS_NUM_THREADS']='8'
import sys,json,csv
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import joblib
sys.path.insert(0,str(Path(__file__).resolve().parent))
from develop_phgeofuse_regression import ROOT,OUT,metrics
from phgeofuse.cache import atomic_json
D=np.load(OUT/'development_features.npz');F=np.load(OUT/'esm2_masked/features.npz');assert np.array_equal(D['keys'],F['keys'])
y=D['y'];yv=D['yv'];low=D['lowv'];folder=OUT/'compact';folder.mkdir(exist_ok=True)
rows={r['key']:r for r in csv.DictReader((OUT/'baseline_validation.csv').open())};base=np.array([float(rows[k]['prediction']) for k in D['keys'][7124:]])
bm=metrics(yv,base,low);mean=F['mean'].astype('float64');std=F['std'].astype('float64')
mean/=np.linalg.norm(mean,axis=1,keepdims=True);std/=np.linalg.norm(std,axis=1,keepdims=True)
results=[];candidates=[]
for pooling,xall in [('mean',mean),('mean_std',np.column_stack([mean,std]))]:
 x=xall[:7124];xv=xall[7124:]
 for power in [0,.25]:
  bins=np.clip(np.floor(y).astype(int),0,14);counts=np.bincount(bins,minlength=15);weights=(len(y)/np.maximum(counts[bins],1))**power;weights/=weights.mean();weights=np.clip(weights,.5,2.5)
  for alpha in [.01,.1,1,10]:
   name=f'{pooling}_p{power}_a{alpha}';model=Ridge(alpha=alpha,solver='cholesky');model.fit(x,y,sample_weight=weights)
   pred=model.predict(xv);train=model.predict(x);joblib.dump(model,folder/f'{name}.joblib');np.save(folder/f'{name}.validation.npy',pred);np.save(folder/f'{name}.train.npy',train)
   row={'name':name,'pooling':pooling,'power':power,'alpha':alpha,'train_rmse':float(np.sqrt(np.mean((train-y)**2))),'validation':metrics(yv,pred,low)};results.append(row);print(json.dumps(row),flush=True)
   for residual_power in [.25,.5]:
    residual=np.load(OUT/f'retrieval_residual/val_weight{residual_power}_i150.npy')
    for wk,wr in [(.25,.25),(.25,.5),(.5,.25)]:
     blended=(1-wk-wr)*base+wk*pred+wr*residual
     # Calibration is cross-fitted within validation for development diagnostics.
     a=np.column_stack([blended,np.ones(len(yv))]);oof=np.empty(len(yv))
     for tr,te in KFold(5,shuffle=True,random_state=42).split(a):
      c=np.linalg.lstsq(a[tr],yv[tr],rcond=None)[0];oof[te]=a[te]@c
     coeff=np.linalg.lstsq(a,yv,rcond=None)[0]
     for calibration,pr in [('none',blended),('affine_oof',oof)]:
      m=metrics(yv,pr,low)
      acceptable=(m['rmse']<bm['rmse'] and m['mae']<bm['mae'] and m['low_homology']['rmse']<bm['low_homology']['rmse'] and abs(m['acidic']['bias'])<abs(bm['acidic']['bias']) and abs(m['alkaline']['bias'])<abs(bm['alkaline']['bias']))
      candidates.append({'name':name,'pooling':pooling,'power':power,'alpha':alpha,'residual_power':residual_power,'weights':{'baseline':1-wk-wr,'sequence':wk,'residual':wr},'calibration':calibration,'affine_coefficients':coeff.tolist(),'validation':m,'passes_all_validation_constraints':acceptable})
atomic_json(folder/'regression_search.json',results);candidates.sort(key=lambda r:r['validation']['rmse']);atomic_json(folder/'fusion_search.json',candidates)
passing=[r for r in candidates if r['passes_all_validation_constraints']]
atomic_json(folder/'selected.json',passing[0] if passing else {'status':'no_candidate_meets_all_constraints','best':candidates[0]})
print('COMPACT_SEARCH_COMPLETE',json.dumps(passing[0] if passing else candidates[0]),flush=True)
