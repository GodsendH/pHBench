"""Reuse per-sequence features, rebuild baseline supports, validate all training inputs."""
import argparse
import copy
import hashlib
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dataset_registry import dataset_fasta_paths
from phgeofuse.config import load_config
from phgeofuse.datasets import apply_dataset, validate_fixed_test_inputs
from phgeofuse.io import read_fasta,read_manifest,write_manifest
from models.data_loader import ProteinpHDataset

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def make_entries(records,train,features,strategy):
    ids=[r.protein_id for r in train]; positions={k:i for i,k in enumerate(ids)}
    target=np.array([features['train'][k] for k in ids],dtype=np.float32)
    target/=np.linalg.norm(target,axis=1,keepdims=True)
    rng=np.random.RandomState(42);out=[]
    for start in range(0,len(records),128):
        batch=records[start:start+128]
        queries=np.array([features[r.split][r.protein_id] for r in batch],dtype=np.float32)
        queries/=np.linalg.norm(queries,axis=1,keepdims=True)
        scores=queries@target.T
        for i,r in enumerate(batch):
            own=positions.get(r.protein_id) if r.split=='train' else None
            if strategy=='opt_retrieval':
                if own is not None:scores[i,own]=-np.inf
                selected=scores[i].argsort()[-5:][::-1]
            else:
                pool=np.arange(len(train))
                if own is not None:pool=np.delete(pool,own)
                selected=rng.choice(pool,size=5,replace=False)
            support=[train[int(j)] for j in selected]
            assert all(s.split=='train' for s in support)
            assert own is None or all(s.protein_id!=r.protein_id for s in support)
            out.append(dict(opt_sequence=r.sequence,opt_id=r.protein_id,opt_pH=str(r.ph_opt),
                env_sequences=[s.sequence for s in support],env_ids=[s.protein_id for s in support],env_pHs=[str(s.ph_opt) for s in support]))
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--work',required=True);args=ap.parse_args()
    work=Path(args.work);summary=json.loads((work/'summary.json').read_text())
    original=read_manifest(ROOT/'artifacts/phgeofuse/manifest.csv')
    prepared={(r.split,r.protein_id,r.sequence_sha256):r for r in original}
    features={};feature_sources={}
    for split,suffix in [('train','train'),('validation','valid'),('test','test')]:
        source=ROOT/f'data/features/opt_{suffix}_features.pkl'
        features[split]=pickle.loads(source.read_bytes());feature_sources[split]=dict(path=str(source),sha256=sha(source))
        official=read_fasta(dataset_fasta_paths(ROOT,'phopt')[split],split)
        assert set(features[split])=={r.protein_id for r in official}
        assert all(np.isfinite(v).all() and np.linalg.norm(v)>0 for v in features[split].values())
    # Fresh task base training; v3 remains a separate same-dataset fine-tuning stage.
    for seed in range(5):
        config=yaml.safe_load((ROOT/'configs/phgeofuse_phopt_tuned_v1.yaml').read_text())
        config['training'].update(seed=seed,trainable_scope='all',run_name='phgeofuse_fixedtest_base_v3')
        (ROOT/f'configs/phgeofuse_fixedtest_base_seed{seed}.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    reports={}
    for name in summary:
        print(f'Preparing {name}',flush=True)
        paths=dataset_fasta_paths(ROOT,name);splits={s:read_fasta(p,s) for s,p in paths.items()}
        records=[]
        for rs in splits.values():
            for r in rs:
                old=prepared[(r.split,r.protein_id,r.sequence_sha256)]
                assert old.ph_opt==r.ph_opt and old.sample_weight==r.sample_weight
                assert old.status=='ready'
                for p in (old.structure_path,old.graph_path,old.embedding_path):assert Path(p).is_file(),p
                records.append(copy.deepcopy(old))
        artifact=ROOT/'artifacts/phgeofuse/datasets'/name;artifact.mkdir(parents=True,exist_ok=True)
        write_manifest(artifact/'manifest.csv',records)
        config=apply_dataset(load_config(ROOT/'configs/phgeofuse_fixedtest_base_seed0.yaml'),name)
        validate_fixed_test_inputs(read_manifest(artifact/'manifest.csv'),config)
        reports[name]=dict(counts=dict(Counter(r.split for r in records)),ready=len(records),
            dataset_fingerprint=config['data']['dataset_fingerprint'],manifest_sha256=sha(artifact/'manifest.csv'),strategies={})
        for strategy in ['opt_retrieval','opt_random']:
            dest=ROOT/'data/processed'/name/'top5'/f'esm2_{strategy}';dest.mkdir(parents=True,exist_ok=True)
            hashes={}
            for split,rs in splits.items():
                suffix='valid' if split=='validation' else split
                output=dest/f'retrieval_{suffix}.json'
                entries=make_entries(rs,splits['train'],features,strategy)
                output.write_text(json.dumps(entries))
                ds=ProteinpHDataset(output,verbose=False)
                assert len(ds)==len(rs)
                sample=ds[0];assert len(sample['env_ids'])==5
                hashes[split]=sha(output)
            reports[name]['strategies'][strategy]=hashes
            (dest/'provenance.json').write_text(json.dumps(dict(dataset_fingerprint=config['data']['dataset_fingerprint'],
                feature_sources=feature_sources,feature_policy='Original repository ESM2 pooled features reused by official split and ID; no re-encoding',
                support_policy='current training set only; exclude training query self',random_support_seed=42,top_k=5,hashes=hashes),indent=2))
        print(json.dumps(reports[name]['counts']),flush=True)
    (work/'training_readiness.json').write_text(json.dumps(reports,indent=2))

if __name__=='__main__':main()
