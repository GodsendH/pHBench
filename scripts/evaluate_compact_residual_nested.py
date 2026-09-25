"""Capacity and mild-tail-weight ablation with fixed nested reliability gates.

Only PHOPT development records are read. Outer labels cannot enter the
sequence, retrieval, gate, or residual fitted for that outer fold.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
import sys
import json
import time
import hashlib
from pathlib import Path
import numpy as np
import joblib
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from develop_phgeofuse_regression import ROOT, OUT, metrics
from phgeofuse.cache import atomic_json
from phgeofuse.robust_train import frequency_weights
from phgeofuse.robust_fusion import chemistry_features, pool_features
from phgeofuse.reliability_fusion import reliability_inputs
from phgeofuse.retrieval import RetrievalStore, record_key
from phgeofuse.io import read_manifest

RECIPES=[{'name':f'l{leaf}_i{iterations}_p{power}', 'leaves':leaf,
          'iterations':iterations,'power':power}
         for leaf,iterations in [(7,50),(7,25),(3,50),(2,100)]
         for power in [0.,.1]]


def fit(recipe,x,y,anchor):
    return HistGradientBoostingRegressor(max_leaf_nodes=recipe['leaves'],
        max_iter=recipe['iterations'],min_samples_leaf=80,l2_regularization=30,
        learning_rate=.05,early_stopping=False,random_state=42).fit(x,y-anchor,
            sample_weight=frequency_weights(y,recipe['power']))


def main():
    out=OUT/'compact_reliability_nested_20260915'
    out.mkdir(exist_ok=False)
    source=OUT/'multiview_nested_20260915'
    gates=OUT/'reliability_residual_nested_20260915'
    baseline=np.load(gates/'quality_residual_unweighted_predictions.npz')
    z=np.load(source/'dual_ridge_control_predictions.npz')
    keys,y,folds,groups=z['keys'],z['y'],z['fold'],z['groups']
    assert np.array_equal(baseline['keys'],keys) and np.array_equal(baseline['y'],y)
    records=[r for r in read_manifest(ROOT/'artifacts/phgeofuse/manifest.csv')
             if r.split in ('train','validation')]
    training=np.array([r.split=='train' for r in records])
    allkeys=np.array([record_key(r) for r in records])
    assert training.sum()==7124 and (~training).sum()==760
    assert np.array_equal(keys,allkeys[training])
    chemall=chemistry_features([r.sequence for r in records])
    chem,chemv=chemall[training],chemall[~training]
    atomic_json(out/'protocol.json',{'dataset':'PHOPT','test_access':False,
        'train':7124,'validation':760,'recipes':RECIPES,'seed':42,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'python':sys.executable,'sequence_cache':str(source),'gate_cache':str(gates),
        'primary':'strict outer pooled RMSE',
        'secondary':['low-homology RMSE','MAE','acidic/alkaline bias','neutral RMSE','training-heldout gap'],
        'caveat':'Fixed exploratory comparisons; repeated development is not independent confirmation.',
        'exclusion':'Meta training base predictions exclude both outer and inner; gate and residual '
                    'fit only on outer training rows. Training gap is a diagnostic, not a causal overfit estimate.'})
    cache={}
    def load(excluded):
        excluded=tuple(sorted(excluded))
        if excluded in cache:return cache[excluded]
        tag='_'.join(map(str,excluded))
        q=np.flatnonzero(np.isin(folds,excluded));ref=np.flatnonzero(~np.isin(folds,excluded))
        b=np.load(source/f'dual_ridge_control_excluded_{tag}.npz')
        r=torch.load(OUT/f'nested_homology_strict/excluded_{tag}.pt',map_location='cpu')
        assert np.array_equal(b['keys'],keys[q]) and np.array_equal(b['reference_keys'],keys[ref])
        assert r['metadata']['query_keys']==keys[q].tolist() and r['metadata']['reference_keys']==keys[ref].tolist()
        assert not set(groups[q]) & set(groups[ref])
        cache[excluded]=q,np.asarray(r['retrieval'],float),b['prediction']
        return cache[excluded]
    predictions={r['name']:np.full(len(y),np.nan) for r in RECIPES}
    foldresults={r['name']:[] for r in RECIPES}
    train_predictions={r['name']:[] for r in RECIPES}
    for outer in range(5):
        tr=np.flatnonzero(folds!=outer);te,r,s=load([outer])
        ir=np.full((len(y),15),np.nan);iseq=np.full(len(y),np.nan)
        for inner in range(5):
            if inner==outer:continue
            q,rr,ss=load([outer,inner]);keep=folds[q]==inner
            ir[q[keep]],iseq[q[keep]]=rr[keep],ss[keep]
        assert np.isnan(iseq[te]).all() and np.isfinite(iseq[tr]).all()
        gate=joblib.load(gates/f'quality_residual_unweighted_outer{outer}.joblib')
        a=gate.predict(*reliability_inputs(ir[tr],iseq[tr]))
        av=gate.predict(*reliability_inputs(r,s))
        meta=np.column_stack([ir[tr],iseq[tr],chem[tr]])
        held=np.column_stack([r,s,chem[te]])
        low=~((r[:,4]>=.2)&(r[:,9]>=.8)&(r[:,10]>=.8))
        np.savez(out/f'outer{outer}_meta.npz',training_keys=keys[tr],heldout_keys=keys[te],
            x=meta,anchor=a,y=y[tr],xheld=held,anchorheld=av,yheld=y[te],low=low)
        for recipe in RECIPES:
            name=recipe['name']
            atomic_json(out/'status.json',{'status':'running','pid':os.getpid(),
                'outer':outer,'recipe':name,'updated':time.time()})
            model=fit(recipe,meta,y[tr],a)
            p=av+model.predict(held);pt=a+model.predict(meta)
            predictions[name][te]=p
            train_predictions[name].append(pt)
            row={'outer':outer,'heldout':metrics(y[te],p,low),
                 'train_rmse':float(np.sqrt(np.mean((pt-y[tr])**2)))}
            row['gap']=row['heldout']['rmse']-row['train_rmse']
            foldresults[name].append(row)
            joblib.dump(model,out/f'{name}_outer{outer}.joblib')
        print('COMPACT_OUTER_COMPLETE',outer,flush=True)
    store=RetrievalStore.load(ROOT/'artifacts/phgeofuse/retrieval.pt')
    assert store.payload['training_keys']==keys.tolist()
    rv=np.array([store.features(k).numpy() for k in allkeys[~training]],float)
    xv=[]
    for encoder in ('esm1v','esm2'):
        with np.load(OUT/f'{encoder}_masked/features.npz') as f:
            mapping={str(k):i for i,k in enumerate(f['keys'])}
            idx=[mapping[k] for k in allkeys[~training]]
            xv.append(pool_features(f['mean'][idx],f['std'][idx],'mean_std'))
    sequence=joblib.load(source/'dual_ridge_control_sequence.joblib')
    sv=sequence.predict(np.column_stack(xv));r=z['retrieval'];s=z['sequence']
    gate=joblib.load(gates/'quality_residual_unweighted_gate.joblib')
    a=gate.predict(*reliability_inputs(r,s));av=gate.predict(*reliability_inputs(rv,sv))
    meta=np.column_stack([r,s,chem]);held=np.column_stack([rv,sv,chemv])
    low=~((r[:,4]>=.2)&(r[:,9]>=.8)&(r[:,10]>=.8))
    lowv=~((rv[:,4]>=.2)&(rv[:,9]>=.8)&(rv[:,10]>=.8))
    _,inv,counts=np.unique(groups,return_inverse=True,return_counts=True)
    draws=np.random.default_rng(42).integers(len(counts),size=(1000,len(counts)))
    denom=counts[draws].sum(1)
    ref_sse=np.bincount(inv,weights=(baseline['prediction']-y)**2)
    results={}
    for recipe in RECIPES:
        name=recipe['name'];p=predictions[name]
        assert np.isfinite(p).all()
        model=fit(recipe,meta,y,a);pv=av+model.predict(held)
        joblib.dump(model,out/f'{name}_residual.joblib')
        np.savez(out/f'{name}_predictions.npz',keys=keys,y=y,fold=folds,groups=groups,
            prediction=p,validation=pv,yv=z['yv'],validation_keys=allkeys[~training],
            **{f'training_prediction_outer{k}':v for k,v in enumerate(train_predictions[name])})
        sse=np.bincount(inv,weights=(p-y)**2)
        delta=np.sqrt(sse[draws].sum(1)/denom)-np.sqrt(ref_sse[draws].sum(1)/denom)
        row={'recipe':recipe,'strict_nested':metrics(y,p,low),'validation':metrics(z['yv'],pv,lowv),
             'fold_results':foldresults[name],
             'gap_mean':float(np.mean([f['gap'] for f in foldresults[name]])),
             'family_bootstrap_delta95_vs_reliability':np.quantile(delta,[.025,.975]).tolist()}
        if name=='l7_i50_p0.0':
            row['control_max_abs_diff']=float(np.max(np.abs(p-baseline['prediction'])))
            assert row['control_max_abs_diff']<1e-7
        results[name]=row
        print(name,json.dumps(row['strict_nested']),flush=True)
    atomic_json(out/'results.json',results)
    atomic_json(out/'status.json',{'status':'complete','pid':os.getpid(),'updated':time.time()})
    print('COMPACT_RELIABILITY_COMPLETE',flush=True)


if __name__=='__main__':main()
