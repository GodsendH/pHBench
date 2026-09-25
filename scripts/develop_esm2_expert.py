"""Exploratory ESM2 mean-feature kernel expert; legacy features require provenance validation."""
import os
os.environ['OPENBLAS_NUM_THREADS']='8'
os.environ['OMP_NUM_THREADS']='8'
import pickle,sys,json,time
from pathlib import Path
import numpy as np
import scipy.linalg as la
sys.path.insert(0,str(Path(__file__).resolve().parent))
from develop_phgeofuse_regression import ROOT,OUT,metrics
from phgeofuse.io import read_manifest
from phgeofuse.cache import atomic_json
D=np.load(OUT/'development_features.npz')
rec=[r for r in read_manifest(ROOT/'artifacts/phgeofuse/manifest.csv') if r.split in ('train','validation')]
raw={s:pickle.loads((ROOT/f'data/features/opt_{suffix}_features.pkl').read_bytes()) for s,suffix in [('train','train'),('validation','valid')]}
xx=np.array([raw[r.split][r.protein_id] for r in rec],dtype='float64');assert xx.shape==(7884,1280)
xx/=np.linalg.norm(xx,axis=1,keepdims=True);x=xx[:7124];xv=xx[7124:];y=D['y'];yv=D['yv'];mu=y.mean();yc=y-mu
folder=OUT/'esm2_exploratory';folder.mkdir(exist_ok=True)
np.savez(folder/'features.npz',x=x,xv=xv)
distance=np.maximum(2-2*x@x.T,0);distv=np.maximum(2-2*xv@x.T,0);scale=np.median(distance);results=[];best=float('inf')
for gamma in [0,.25,1,4]:
 print('KERNEL',gamma,flush=True)
 k=x@x.T if gamma==0 else np.exp(-gamma*distance/scale);kv=xv@x.T if gamma==0 else np.exp(-gamma*distv/scale)
 ev,u=la.eigh(k,check_finite=False,driver='evd');ev=np.maximum(ev,0);uy=u.T@yc;vu=kv@u
 for alpha in [.01,.1,1,10]:
  coeff=uy/(ev+alpha);pred=mu+vu@coeff
  row={'gamma':gamma,'alpha':alpha,'validation':metrics(yv,pred,D['lowv']),'train_rmse':float(np.sqrt(np.mean((mu+u@(ev*coeff)-y)**2)))}
  np.save(folder/f'val_g{gamma}_a{alpha}.npy',pred);results.append(row);print(json.dumps(row),flush=True)
  if row['validation']['rmse']<best:
   best=row['validation']['rmse'];np.savez(folder/'best_kernel.npz',x=x,dual=u@coeff,mu=mu,gamma=gamma,scale=scale,alpha=alpha)
 atomic_json(folder/'search.json',results)
print('ESM2_SEARCH_COMPLETE',flush=True)
