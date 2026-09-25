"""Train/validation-only development of regularized PHGeoFuse regressors."""
import os
os.environ.setdefault('OMP_NUM_THREADS','8')
os.environ.setdefault('OPENBLAS_NUM_THREADS','8')
import sys,json,time,csv,hashlib
from pathlib import Path
import numpy as np
import scipy.linalg as la
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from phgeofuse.io import read_manifest
from phgeofuse.retrieval import RetrievalStore,record_key,_load_mean_embedding
from phgeofuse.cache import atomic_json
OUT=ROOT/'experiments/phgeofuse_redesign_20260914';OUT.mkdir(exist_ok=True)
SOURCE=ROOT/'experiments/phgeofuse_phopt_full_20260913/inputs'

def metrics(y,p,low):
    out={'rmse':float(np.sqrt(np.mean((p-y)**2))),'mae':float(np.mean(abs(p-y))),'r2':float(1-np.sum((p-y)**2)/np.sum((y-y.mean())**2)),'spearman':float(spearmanr(y,p).statistic)}
    for name,mask in [('acidic',y<6),('neutral',(y>=6)&(y<8)),('alkaline',y>=8),('low_homology',low)]:
        if mask.any():out[name]={'count':int(mask.sum()),'rmse':float(np.sqrt(np.mean((p[mask]-y[mask])**2))),'bias':float(np.mean(p[mask]-y[mask]))}
    return out

def prepare():
    torch.set_num_threads(4)
    rec=[r for r in read_manifest(SOURCE/'manifest.csv') if r.split in ('train','validation')]
    store=RetrievalStore.load(SOURCE/'retrieval.pt');train=[r for r in rec if r.split=='train'];val=[r for r in rec if r.split=='validation']
    assert store.payload['training_keys']==[record_key(r) for r in train]
    fp=OUT/'development_features.npz'
    if fp.exists():return np.load(fp)
    x=store.payload['training_vectors'].numpy().astype('float64')
    xv=[]
    for i,r in enumerate(val):
        xv.append(_load_mean_embedding(r).numpy())
        if i%100==0:print('VAL_FEATURE',i,flush=True)
    feat=np.array([store.features(record_key(r)).numpy() for r in rec])
    low=~((feat[:,4]>=.2)&(feat[:,9]>=.8)&(feat[:,10]>=.8))
    np.savez(fp,x=x,xv=np.array(xv,dtype='float64'),y=np.array([r.ph_opt for r in train]),yv=np.array([r.ph_opt for r in val]),low=low[:len(train)],lowv=low[len(train):],retrieval=feat,keys=np.array([record_key(r) for r in rec]))
    atomic_json(OUT/'protocol.json',{'train':len(train),'validation':len(val),'test_access_for_selection':False,'manifest_sha256':hashlib.sha256((SOURCE/'manifest.csv').read_bytes()).hexdigest(),'primary':'validation RMSE','secondary':['low-homology RMSE','acidic and alkaline absolute bias'],'families':['linear ridge','RBF kernel ridge'],'regularization':[.001,.01,.1,1,10,100]})
    return np.load(fp)

def main():
    data=prepare();x=data['x'];xv=data['xv'];y=data['y'];yv=data['yv'];mu=y.mean();yc=y-mu
    x=x/np.linalg.norm(x,axis=1,keepdims=True);xv=xv/np.linalg.norm(xv,axis=1,keepdims=True)
    distance=cdist(x,x,'sqeuclidean');distv=cdist(xv,x,'sqeuclidean');scale=np.median(distance)
    results=[];best=float('inf')
    for gamma in [0, .25,1,4,16]:
        print('KERNEL',gamma,'start',time.time(),flush=True)
        kernel=x@x.T if gamma==0 else np.exp(-gamma*distance/scale)
        kv=xv@x.T if gamma==0 else np.exp(-gamma*distv/scale)
        ev,u=la.eigh(kernel,check_finite=False,driver='evd');ev=np.maximum(ev,0);uy=u.T@yc;vu=kv@u
        for alpha in [.001,.01,.1,1,10,100]:
            coeff=uy/(ev+alpha);pred=mu+vu@coeff;train_pred=mu+u@(ev*coeff)
            inv_diag=(u*u)@(1/(ev+alpha));loo=y-(u@coeff)/inv_diag
            row={'gamma':gamma,'alpha':alpha,'validation':metrics(yv,pred,data['lowv']),'training_rmse':float(np.sqrt(np.mean((train_pred-y)**2))),'loo_rmse':float(np.sqrt(np.mean((loo-y)**2)))}
            results.append(row);print(json.dumps(row),flush=True)
            np.save(OUT/f'val_g{gamma}_a{alpha}.npy',pred)
            if row['validation']['rmse']<best:
                best=row['validation']['rmse'];np.savez(OUT/'best_kernel.npz',x=x,dual=u@coeff,mu=mu,gamma=gamma,scale=scale,alpha=alpha)
                atomic_json(OUT/'best_validation.json',row)
        atomic_json(OUT/'search.json',results)
    print('SEARCH_COMPLETE',flush=True)
if __name__=='__main__':main()

