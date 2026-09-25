"""Regularized sequence/chemistry correction for PHGeoFuse predictions.

Task labels are never inputs to predict(). The baseline and retrieval database
must be trained exclusively on the corresponding training split.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis


def chemistry_features(sequences):
    rows=[]
    alphabet='ACDEFGHIKLMNPQRSTVWY'
    for sequence in sequences:
        clean=''.join(a for a in sequence if a in alphabet)
        if not clean:
            raise ValueError('sequence has no canonical residues')
        analysis=ProteinAnalysis(clean)
        rows.append([sequence.count(a)/len(sequence) for a in alphabet+'X']+[
            np.log1p(len(sequence)),analysis.isoelectric_point(),analysis.gravy(),analysis.aromaticity()])
    return np.asarray(rows,dtype=np.float64)


def pool_features(mean,std,pooling):
    parts=[]
    for value in ([mean,std] if pooling=='mean_std' else [mean]):
        value=np.asarray(value,dtype=np.float64)
        if value.ndim!=2 or not np.isfinite(value).all():raise ValueError('invalid pooled features')
        norm=np.linalg.norm(value,axis=1,keepdims=True)
        if (norm==0).any():raise ValueError('zero feature vector')
        parts.append(value/norm)
    return np.column_stack(parts)


class RobustFusion:
    def __init__(self,bundle):
        bundle=Path(bundle)
        self.config=json.loads((bundle/'model.json').read_text())
        for name,expected in self.config['file_hashes'].items():
            if hashlib.sha256((bundle/name).read_bytes()).hexdigest()!=expected:
                raise ValueError(f'model hash mismatch: {name}')
        self.sequence=joblib.load(bundle/'sequence.joblib')
        self.residual=joblib.load(bundle/'residual.joblib')
        self.weights=self.config['weights']
        if abs(sum(self.weights.values())-1)>1e-8 or min(self.weights.values())<0:
            raise ValueError('fusion weights must be a convex combination')

    def predict(self,baseline,mean,std,retrieval,sequences):
        baseline=np.asarray(baseline,dtype=np.float64)
        retrieval=np.asarray(retrieval,dtype=np.float64)
        n=len(sequences)
        if baseline.shape!=(n,) or retrieval.shape!=(n,15):raise ValueError('sample dimensions differ')
        if not np.isfinite(baseline).all() or not np.isfinite(retrieval).all():raise ValueError('nonfinite input')
        sequence=self.sequence.predict(pool_features(mean,std,self.config['pooling']))
        if sequence.shape!=(n,):raise ValueError('pooled feature sample count differs')
        features=np.column_stack([retrieval,chemistry_features(sequences)])
        available=retrieval[:,7:9]
        anchor=np.where(available.sum(1)>0,(retrieval[:,:2]*available).sum(1)/np.maximum(available.sum(1),1),self.config['training_mean_label'])
        residual=anchor+self.residual.predict(features)
        result=self.weights['baseline']*baseline+self.weights['sequence']*sequence+self.weights['residual']*residual
        return {'prediction':result,'baseline_prediction':baseline,'sequence_prediction':sequence,'residual_prediction':residual}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','manifest','retrieval','baseline-predictions','sequence-features','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--split',choices=['train','validation','test'],required=True)
    args=parser.parse_args()
    from .io import read_manifest
    from .retrieval import RetrievalStore,record_key
    records=[r for r in read_manifest(args.manifest) if r.split==args.split]
    if not records:raise ValueError('empty split')
    keys=[record_key(r) for r in records]
    with open(args.baseline_predictions) as f:baseline_rows=list(csv.DictReader(f))
    baseline={r['key']:float(r['prediction']) for r in baseline_rows}
    if len(baseline)!=len(baseline_rows) or set(baseline)!=set(keys):raise ValueError('baseline sample set differs')
    with np.load(args.sequence_features,allow_pickle=False) as data:
        order={str(k):i for i,k in enumerate(data['keys'])}
        indices=[order[k] for k in keys];mean=data['mean'][indices];std=data['std'][indices]
    store=RetrievalStore.load(args.retrieval)
    retrieval=np.array([store.features(k).numpy() for k in keys])
    predictions=RobustFusion(args.bundle).predict([baseline[k] for k in keys],mean,std,retrieval,[r.sequence for r in records])
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['key','label',*predictions]);writer.writeheader()
        for i,r in enumerate(records):writer.writerow({'key':keys[i],'label':r.ph_opt,**{k:float(v[i]) for k,v in predictions.items()}})

if __name__=='__main__':main()
