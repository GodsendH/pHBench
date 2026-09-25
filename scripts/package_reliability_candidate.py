"""Package and independently verify the development-only reliability candidate."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
import json
import sys
import shutil
import hashlib
import argparse
from pathlib import Path
import numpy as np
import joblib
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from develop_phgeofuse_regression import ROOT, OUT, metrics
from phgeofuse.cache import atomic_json
from phgeofuse.io import read_manifest
from phgeofuse.retrieval import RetrievalStore, record_key
from phgeofuse.reliability_fusion import ReliabilityFusion, reliability_inputs
from phgeofuse.robust_fusion import chemistry_features


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compact',action='store_true')
    args=parser.parse_args()
    gates=OUT/'reliability_residual_nested_20260915'
    source=OUT/('compact_reliability_nested_20260915' if args.compact else 'reliability_residual_nested_20260915')
    multi=OUT/'multiview_nested_20260915'
    name='l3_i50_p0.0' if args.compact else 'quality_residual_unweighted'
    assert json.loads((source/'status.json').read_text())['status']=='complete'
    out=OUT/('compact_reliability_candidate_20260915' if args.compact else 'reliability_candidate_20260915')
    out.mkdir(exist_ok=False)
    mapping={'sequence.joblib':multi/'dual_ridge_control_sequence.joblib',
        'gate.joblib':gates/'quality_residual_unweighted_gate.joblib',
        'residual.joblib':source/f'{name}_residual.joblib'}
    for filename,path in mapping.items():
        shutil.copyfile(path,out/filename)
    protocol={'kind':'PHOPT sequence/retrieval residual expert', 'dataset':'PHOPT',
        'train':7124,'validation':760,'test_access':False,'seed':42,
        'status':'development candidate; overfitting and tail-bias objectives remain open',
        'source':str(source),'recipe':name,
        'file_sha256':{n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in mapping},
        'code_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
            ['phgeofuse/reliability_fusion.py','phgeofuse/reliability_gate.py',
             'scripts/evaluate_reliability_gate_nested.py']}}
    atomic_json(out/'model.json',protocol)
    model=ReliabilityFusion(out)
    records=[r for r in read_manifest(ROOT/'artifacts/phgeofuse/manifest.csv')
             if r.split in ('train','validation')]
    train=[r for r in records if r.split=='train']
    val=[r for r in records if r.split=='validation']
    keys=np.array([record_key(r) for r in train])
    valkeys=np.array([record_key(r) for r in val])
    y=np.array([r.ph_opt for r in train]);yv=np.array([r.ph_opt for r in val])
    pred=np.load(source/f'{name}_predictions.npz')
    assert np.array_equal(pred['keys'],keys) and np.array_equal(pred['validation_keys'],valkeys)
    arrays=[]
    for encoder in ('esm1v','esm2'):
        with np.load(OUT/f'{encoder}_masked/features.npz') as f:
            mapping={str(k):i for i,k in enumerate(f['keys'])}
            idx=[mapping[k] for k in valkeys]
            arrays.extend([f['mean'][idx],f['std'][idx]])
    store=RetrievalStore.load(ROOT/'artifacts/phgeofuse/retrieval.pt')
    assert store.payload['training_keys']==keys.tolist()
    rv=np.array([store.features(k).numpy() for k in valkeys],float)
    pv=model.predict(*arrays,rv,[r.sequence for r in val])
    valdiff=float(np.max(np.abs(pv-pred['validation'])))
    assert valdiff<1e-7
    # Independent reconstruction from saved fold models and excluded cache keys.
    poof=np.full(len(y),np.nan); rr=np.full((len(y),15),np.nan)
    chem=chemistry_features([r.sequence for r in train])
    for k in range(5):
        te=np.flatnonzero(pred['fold']==k); ref=np.flatnonzero(pred['fold']!=k)
        base=np.load(multi/f'dual_ridge_control_excluded_{k}.npz')
        assert np.array_equal(base['keys'],keys[te]) and np.array_equal(base['reference_keys'],keys[ref])
        payload=torch.load(OUT/f'nested_homology_strict/excluded_{k}.pt',map_location='cpu')
        assert payload['metadata']['reference_keys']==keys[ref].tolist()
        rr[te]=np.asarray(payload['retrieval'],float)
        gate=joblib.load(gates/f'quality_residual_unweighted_outer{k}.joblib')
        res=joblib.load(source/(f'{name}_outer{k}.joblib' if args.compact else f'{name}_outer{k}_residual.joblib'))
        s=base['prediction']
        poof[te]=gate.predict(*reliability_inputs(rr[te],s))+res.predict(np.column_stack([rr[te],s,chem[te]]))
    oofdiff=float(np.max(np.abs(poof-pred['prediction'])))
    assert oofdiff<1e-7 and np.isfinite(pv).all() and np.isfinite(poof).all()
    low=~((rr[:,4]>=.2)&(rr[:,9]>=.8)&(rr[:,10]>=.8))
    lowv=~((rv[:,4]>=.2)&(rv[:,9]>=.8)&(rv[:,10]>=.8))
    result={'status':'verified','test_access':False,'validation_max_abs_diff':valdiff,
        'outer_max_abs_diff':oofdiff,'strict_nested':metrics(y,poof,low),
        'validation':metrics(yv,pv,lowv)}
    atomic_json(out/'verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
