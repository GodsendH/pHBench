"""Fit the frozen robust-fusion recipe on PHOPT train only."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from .cache import atomic_json
from .io import read_manifest, read_fasta
from .retrieval import RetrievalStore, record_key
from .robust_fusion import chemistry_features, pool_features


def frequency_weights(y,power):
    bins=np.clip(np.floor(y).astype(int),0,14)
    counts=np.bincount(bins,minlength=15)
    weights=(len(y)/np.maximum(counts[bins],1))**power
    return np.clip(weights/weights.mean(),.5,2.5)


def fit(manifest,retrieval,sequence_features,output,seed=42):
    root=Path(__file__).resolve().parents[1]
    records=[r for r in read_manifest(manifest) if r.split=='train']
    official=read_fasta(root/'data/phopt_training.fasta','train')
    signature=lambda rs:sorted((r.protein_id,r.sequence,r.ph_opt,r.sample_weight) for r in rs)
    if signature(records)!=signature(official):raise ValueError('training manifest differs from PHOPT train')
    if len(records)!=7124 or not all(r.status=='ready' for r in records):raise ValueError('PHOPT train not complete')
    keys=[record_key(r) for r in records];y=np.array([r.ph_opt for r in records])
    store=RetrievalStore.load(retrieval)
    if store.payload['training_keys']!=keys or store.payload['training_sequences']!=[r.sequence for r in records]:raise ValueError('retrieval reference set differs from PHOPT train')
    if not np.allclose(store.payload['training_labels'].numpy(),y,atol=1e-6):raise ValueError('retrieval reference labels differ')
    with np.load(sequence_features,allow_pickle=False) as f:
        indices={str(k):i for i,k in enumerate(f['keys'])};order=[indices[k] for k in keys]
        x=pool_features(f['mean'][order],f['std'][order],'mean_std')
    sequence_model=Ridge(alpha=.1,solver='cholesky').fit(x,y,sample_weight=frequency_weights(y,.25))
    chem=chemistry_features([r.sequence for r in records]);views=[];anchors=[]
    for view in ['normal','low_homology']:
        values=np.array([store.features(k,view=view).numpy() for k in keys])
        available=values[:,7:9]
        anchors.append(np.where(available.sum(1)>0,(values[:,:2]*available).sum(1)/np.maximum(available.sum(1),1),y.mean()))
        views.append(np.column_stack([values,chem]))
    weights=frequency_weights(y,.5)
    residual_model=HistGradientBoostingRegressor(max_leaf_nodes=7,max_iter=150,learning_rate=.05,min_samples_leaf=60,l2_regularization=20,early_stopping=False,random_state=seed)
    residual_model.fit(np.concatenate(views),np.tile(y,2)-np.concatenate(anchors),sample_weight=np.concatenate([weights,weights*.25]))
    output=Path(output)
    if (output/'model.json').exists():raise FileExistsError('refusing to overwrite a fitted model')
    output.mkdir(parents=True,exist_ok=True)
    joblib.dump(sequence_model,output/'sequence.joblib');joblib.dump(residual_model,output/'residual.joblib')
    config={'architecture':'PHGeoFuse robust fusion v1','dataset':'phopt','seed':seed,'training_count':len(y),'pooling':'mean_std','training_mean_label':float(y.mean()),'weights':{'baseline':.5,'sequence':.25,'residual':.25},'calibration':'none','recipe_selection':'fixed using validation in phgeofuse_redesign_20260914','training_signature_sha256':hashlib.sha256(json.dumps(signature(records)).encode()).hexdigest(),'inputs':{str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [manifest,retrieval,sequence_features]},'file_hashes':{name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in ['sequence.joblib','residual.joblib']}}
    atomic_json(output/'model.json',config)
    return config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['manifest','retrieval','sequence-features','output']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    print(json.dumps(fit(args.manifest,args.retrieval,args.sequence_features,args.output,args.seed),indent=2))

if __name__=='__main__':main()
