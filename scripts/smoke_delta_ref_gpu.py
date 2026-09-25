"""Bounded real-model checks; outputs are never used for recipe selection."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def fasta_smoke(data, output):
    import numpy as np
    from phgeofuse.delta_ref.inference import attach_baseline,predict_fasta
    from phgeofuse.delta_ref.baseline import FullBaseline
    from phgeofuse.delta_ref.training import load_bundle
    from phgeofuse.cache import atomic_json
    # Use original validation sequences strictly for numerical parity, no tuning.
    indices=sorted(data.validation,key=lambda i:len(data.records[i].sequence))[:2]
    source=ROOT/'experiments/delta_ref_phopt_20260916/smoke/real_features'
    model=output/'bundle';model.mkdir(parents=True,exist_ok=True)
    for name in ('weights.pt','model.json'):
        shutil.copy2(source/name,model/name)
    attach_baseline(model,data.config,42)
    fasta=output/'queries.fasta'
    fasta.write_text(''.join(f'>{data.records[i].protein_id}\n{data.records[i].sequence}\n' for i in indices))
    result=predict_fasta(fasta,model,data.config['_config_path'],output/'predictions.csv')
    baseline=FullBaseline(data,ROOT/'experiments/delta_ref_phopt_20260916/baseline/seed42').frozen_validation(42)
    order={i:j for j,i in enumerate(data.validation)}
    expected=baseline[[order[i] for i in indices]]
    with np.load(output/'predictions.inputs/features.npz') as z:
        difference=float(np.max(abs(z['features']-data.x[indices])))
    # Predicting a subset separately can change BF16 graph reduction roundoff.
    max_error=float(np.max(abs(result['baseline_prediction']-expected)))
    if difference>1e-6 or max_error>1e-3:
        raise ValueError(f'FASTA baseline parity failed: features={difference}, baseline={max_error}')
    saved={'stage':'fasta_parity','queries':data.keys[indices].tolist(),
           'feature_max_abs_error':difference,'complete_baseline_max_abs_error':max_error,
           'baseline_expected':expected.tolist(),'baseline_actual':result['baseline_prediction'].tolist(),
           'performance_evidence':False,'test_access':False}
    atomic_json(output/'fasta_result.json',saved);print(json.dumps(saved),flush=True)


def lora_smoke(data, output):
    import numpy as np
    import torch
    from phgeofuse.delta_ref.lora import PrefixEncoder
    from phgeofuse.delta_ref.training import seed_all,sample_pairs
    from phgeofuse.delta_ref.model import DeltaNetwork,Standardizer
    from phgeofuse.cache import atomic_json
    seed_all(42);torch.set_num_threads(4)
    encoder=PrefixEncoder(data.config['lora'],output/'prefix','cuda')
    # Check a >1022-residue input without introducing any training label.
    sequence=(data.records[data.train[0]].sequence*40)[:1031]
    encoder.prepare(sequence)
    encoder.set_adapter_training(False)
    with torch.no_grad():
        pooled=encoder.encode(sequence)
    payload=encoder._load_prefix(sequence)
    assert sum(payload['lengths'])==1031 and len(payload['chunks'])==2 and torch.isfinite(pooled).all()
    train=data.train
    scaler=Standardizer.fit(data.x[train])
    center=torch.tensor(scaler.mean,dtype=torch.float32,device='cuda')
    scale=torch.tensor(scaler.scale,dtype=torch.float32,device='cuda')
    head=DeltaNetwork(5145,64,.1).cuda()
    params=[p for p in encoder.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW([{'params':head.parameters(),'lr':3e-4},{'params':params,'lr':1e-5}],weight_decay=1e-3)
    q,r=sample_pairs(data.labels[train],data.groups[train],np.random.default_rng(42))
    pair_indices=np.column_stack([train[q[:64]],train[r[:64]]])
    prefix_started=time.monotonic()
    for j,i in enumerate(np.unique(pair_indices)):
        encoder.prepare(data.records[i].sequence)
        if j%20==0:print('SMOKE_PREFIX',j,flush=True)
    prefix_seconds=time.monotonic()-prefix_started
    def feature(i):
        fixed=torch.tensor(data.x[i],dtype=torch.float32,device='cuda')
        encoded=encoder.encode(data.records[i].sequence,gradient=True)
        return (torch.cat([fixed[:2560],encoded,fixed[5120:]])-center)/scale
    torch.cuda.reset_peak_memory_stats()
    encoder.set_adapter_training(True);head.train();optimizer.zero_grad(set_to_none=True)
    started=time.monotonic();losses=[]
    for step,(i,j) in enumerate(pair_indices):
        pred=head(feature(i)[None],feature(j)[None])[0]
        loss=(pred-float(data.labels[i]-data.labels[j])).square()
        (loss/16).backward();losses.append(float(loss))
        if (step+1)%16==0:
            optimizer.step();optimizer.zero_grad(set_to_none=True)
        if (step+1)%16==0:print('SMOKE_LORA_PAIRS',step+1,flush=True)
    torch.cuda.synchronize();seconds=time.monotonic()-started
    assert all(p.grad is None for p in encoder.parameters() if not p.requires_grad)
    saved={'stage':'lora_resource_profile','device':torch.cuda.get_device_name(),
           'pairs':64,'pair_seconds':seconds,'prefix_seconds':prefix_seconds,
           'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated(),
           'median_residues':float(np.median([len(data.records[i].sequence) for i in pair_indices.flat])),
           'estimated_4274_query_epoch_seconds':seconds/64*4274*8,
           'adapter_parameters':sum(p.numel() for p in params),'long_residue_coverage':1031,
           'mean_smoke_loss':float(np.mean(losses)),'performance_evidence':False,'test_access':False}
    atomic_json(output/'lora_result.json',saved);print(json.dumps(saved),flush=True)
    encoder._load_prefix.cache_clear()


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['fasta','lora'],required=True)
    args=p.parse_args()
    from phgeofuse.delta_ref.data import DevelopmentData
    data=DevelopmentData.load(ROOT/'configs/delta_ref_phopt.yaml')
    output=ROOT/'experiments/delta_ref_phopt_20260916/smoke'/args.stage
    output.mkdir(parents=True,exist_ok=True)
    (fasta_smoke if args.stage=='fasta' else lora_smoke)(data,output)


if __name__=='__main__':main()
