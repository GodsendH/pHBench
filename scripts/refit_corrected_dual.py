"""Refit the frozen full-Dual recipe using versioned corrected retrieval features."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import torch
from sklearn.base import clone
from sklearn.linear_model import Ridge

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from phgeofuse.cache import atomic_json,atomic_torch_save,sha256_file
from phgeofuse.config import load_config
from phgeofuse.io import read_manifest
from phgeofuse.retrieval import RetrievalStore,record_key,_build_retrieval_rows
from phgeofuse.robust_fusion import pool_features,chemistry_features
from phgeofuse.robust_train import fit as fit_robust,frequency_weights
from phgeofuse.dual_fusion import DualFusion,retrieval_sequence_anchor


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment',type=Path,required=True)
    ap.add_argument('--predictions-only',action='store_true')
    args=ap.parse_args();out=args.experiment.resolve();torch.set_num_threads(4)
    legacy=ROOT/'experiments/phgeofuse_redesign_20260914'
    work=out/'dual_refit';work.mkdir(exist_ok=True)
    bundle=work/'bundle';config=load_config(out/'baseline.yaml')
    records=read_manifest(out/'manifest.csv');train=[r for r in records if r.split=='train']
    keys=[record_key(r) for r in train];y=np.array([r.ph_opt for r in train])
    store=RetrievalStore.load(out/'retrieval.pt')
    if store.payload['training_keys']!=keys:raise ValueError('retrieval training keys differ')
    parts=[];raw_features={}
    for name in ['esm1v','esm2']:
        with np.load(legacy/f'{name}_masked/features.npz',allow_pickle=False) as f:
            mapping={str(k):i for i,k in enumerate(f['keys'])};ix=[mapping[k] for k in keys]
            parts.append(pool_features(f['mean'][ix],f['std'][ix],'mean_std'))
    x=np.column_stack(parts)
    foldfile=legacy/'homology_oof/strict_folds.json'
    fr=json.loads(foldfile.read_text())['rows']
    if [r['key'] for r in fr]!=[r.protein_id for r in train]:raise ValueError('fold keys differ')
    folds=np.array([r['fold'] for r in fr]);groups=np.array([r['group'] for r in fr])
    if not args.predictions_only and not (bundle/'model.json').exists():
        if not (bundle/'robust_v1/model.json').exists():
            fit_robust(out/'manifest.csv',out/'retrieval.pt',legacy/'esm2_masked/features.npz',bundle/'robust_v1')
        retrieval=np.zeros((len(train),15));seq=np.zeros(len(train))
        for fold in sorted(set(folds)):
            te=np.flatnonzero(folds==fold);tr=np.flatnonzero(folds!=fold)
            if set(groups[te])&set(groups[tr]):raise ValueError('homology groups cross folds')
            atomic_json(work/'status.json',dict(status='running',phase='oof_retrieval',fold=int(fold),pid=os.getpid(),updated=time.time()))
            cache=work/f'fold{fold}.pt'
            expected={'query_keys':[keys[i] for i in te],'reference_keys':[keys[i] for i in tr],
                      'retrieval_sha256':sha256_file(out/'retrieval.pt')}
            if cache.exists():
                payload=torch.load(cache,map_location='cpu')
                if payload['metadata']!=expected:raise ValueError('OOF cache provenance drift')
            else:
                rows=_build_retrieval_rows([train[i] for i in te],[train[i] for i in tr],
                                           store.payload['training_vectors'][tr].float(),torch.tensor(y[tr],dtype=torch.float32),config)
                payload=dict(rows=rows,metadata=expected);atomic_torch_save(cache,payload)
            view=RetrievalStore(payload)
            retrieval[te]=np.array([view.features(keys[i]).numpy() for i in te])
            seq[te]=Ridge(alpha=.2,solver='cholesky').fit(x[tr],y[tr]).predict(x[te])
            print('DUAL_OOF_FOLD',fold,flush=True)
        chemistry=chemistry_features([r.sequence for r in train])
        anchor=retrieval_sequence_anchor(retrieval,seq)
        model=clone(joblib.load(legacy/'dual_candidate_float64/residual.joblib'))
        weights=frequency_weights(y,.25);weights=np.clip(weights/weights.mean(),.25,4.)
        model.fit(np.column_stack([retrieval,seq,chemistry]),y-anchor,sample_weight=weights)
        joblib.dump(model,bundle/'residual.joblib')
        joblib.dump(Ridge(alpha=.2,solver='cholesky').fit(x,y),bundle/'sequence.joblib')
        np.savez(work/'expert_oof_inputs.npz',keys=np.array(keys),retrieval=retrieval,sequence=seq,fold=folds)
        metadata=dict(architecture='full Dual fixed recipe, corrected confidence v1',dual_weight=.5,
                      recipe='phweighted50',training_count=len(train),labels_used='PHOPT train only',
                      manifest_sha256=sha256_file(out/'manifest.csv'),retrieval_sha256=sha256_file(out/'retrieval.pt'),
                      fold_sha256=sha256_file(foldfile),historical_bundle_sha256=sha256_file(legacy/'dual_candidate_float64/model.json'),
                      scope='OOF inputs train the dual residual; this is not full-Dual outer-fold evaluation',
                      file_hashes={str(p.relative_to(bundle)):sha256_file(p) for p in bundle.rglob('*') if p.is_file()})
        atomic_json(bundle/'model.json',metadata)
        atomic_json(work/'status.json',dict(status='fit_complete',pid=os.getpid(),updated=time.time()))
    # Prediction is separately callable after corrected PHGeoFuse training/calibration finishes.
    model=DualFusion(bundle)
    for split in ['validation','test']:
        baseline=out/'dual_baseline'/f'{split}.csv'
        if not baseline.exists():continue
        subset=[r for r in records if r.split==split];skeys=[record_key(r) for r in subset]
        mapping={r['key']:float(r['prediction']) for r in csv.DictReader(baseline.open())}
        if set(mapping)!=set(skeys):raise ValueError('baseline prediction coverage mismatch')
        features=[]
        for name in ['esm1v','esm2']:
            suffix='features_test.npz' if split=='test' else 'features.npz'
            with np.load(legacy/f'{name}_masked'/suffix,allow_pickle=False) as f:
                order={str(k):i for i,k in enumerate(f['keys'])};ix=[order[k] for k in skeys]
                features.extend([f['mean'][ix],f['std'][ix]])
        retrieval=np.array([store.features(k).numpy() for k in skeys])
        prediction=model.predict([mapping[k] for k in skeys],*features,retrieval,[r.sequence for r in subset])
        rows=[dict(key=k,label=r.ph_opt,sequence_sha256=r.sequence_sha256,
                   model_hash=sha256_file(bundle/'model.json'),
                   low_homology=not bool(retrieval[i,4]>=.2 and retrieval[i,9]>=.8 and retrieval[i,10]>=.8),
                   **{name:float(values[i]) for name,values in prediction.items()}) for i,(k,r) in enumerate(zip(skeys,subset))]
        with (work/f'{split}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        print('DUAL_PREDICT',split,len(rows),flush=True)


if __name__=='__main__':main()
