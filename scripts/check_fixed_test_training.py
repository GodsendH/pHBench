"""Readiness smoke check on real identity20 data; no experiment checkpoint is written."""
import json
import argparse
import os
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from phgeofuse.config import load_config,path
from phgeofuse.datasets import apply_dataset,validate_fixed_test_inputs
from phgeofuse.io import read_manifest
from phgeofuse.retrieval import ensure_retrieval_store,record_key
from phgeofuse.dataset import ProteinGraphDataset,collate_graphs,move_batch
from phgeofuse.model import PHGeoFuse,compute_loss

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',default='identity20')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--work',default=str(ROOT/'data/dataset_audits/fixed_test_removal_v3'))
    args=parser.parse_args()
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    torch.set_num_threads(2)
    config=apply_dataset(load_config(ROOT/'configs/phgeofuse_fixedtest_base_seed0.yaml'),args.dataset)
    records=read_manifest(path(config,'paths.manifest'));validate_fixed_test_inputs(records,config)
    print(f'Building {args.dataset} retrieval exclusively from new training records',flush=True)
    store=ensure_retrieval_store(records,config)
    expected={record_key(r) for r in records if r.split=='train'}
    assert set(store.payload['training_keys'])==expected
    assert set(store.rows)=={record_key(r) for r in records}
    assert store.payload['dataset_fingerprint']==config['data']['dataset_fingerprint']
    device=torch.device(args.device)
    model=PHGeoFuse(config,device).to(device);report={'dataset':args.dataset,'dataset_fingerprint':config['data']['dataset_fingerprint'],'device':str(device),'counts':{},'finite_outputs':{},'backward':False}
    for split in ('train','validation','test'):
        ds=ProteinGraphDataset(records,split,store)
        report['counts'][split]=len(ds)
        # Keep this check small while reading actual cached geometry/embedding/retrieval.
        i=min(range(len(ds)),key=lambda i:len(ds.records[i].sequence))
        batch=move_batch(collate_graphs([ds[i]]),device)
        model.train(split=='train')
        with torch.set_grad_enabled(split=='train'):
            out=model(batch);assert torch.isfinite(out['mean']).all()
            if split=='train':
                loss,_=compute_loss(out,batch,config);assert torch.isfinite(loss)
                loss.backward();assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                report['backward']=True
        report['finite_outputs'][split]=True
    target=Path(args.work)/'model_smoke.json'
    target.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
