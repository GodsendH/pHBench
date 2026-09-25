"""Label-free ESM2 residue pooling with complete sequence coverage."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .cache import atomic_json
from .io import read_manifest


def encode_residue_pools(sequence,tokenizer,model,device,chunk_length=1022):
    if not sequence or chunk_length<1:raise ValueError('nonempty sequence and positive chunk_length required')
    chunks=[]
    for start in range(0,len(sequence),chunk_length):
        chunk=sequence[start:start+chunk_length]
        tokens=tokenizer(chunk,return_tensors='pt',add_special_tokens=True)
        tokens={k:v.to(device) for k,v in tokens.items()}
        with torch.inference_mode():
            h=model(**tokens).last_hidden_state[0,1:len(chunk)+1].float().cpu()
        if len(h)!=len(chunk):raise ValueError('token/residue correspondence failed')
        chunks.append(h)
    values=torch.cat(chunks)
    if len(values)!=len(sequence) or not torch.isfinite(values).all():raise ValueError('invalid residue embeddings')
    return values.mean(0).numpy(),values.std(0,unbiased=False).numpy()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--splits',nargs='+',choices=['train','validation','test'],required=True)
    parser.add_argument('--model',default='facebook/esm2_t33_650M_UR50D')
    parser.add_argument('--revision',default=None)
    args=parser.parse_args()
    from transformers import AutoTokenizer,AutoModel
    tokenizer=AutoTokenizer.from_pretrained(args.model,revision=args.revision,local_files_only=True)
    model=AutoModel.from_pretrained(args.model,revision=args.revision,local_files_only=True).cuda().eval().half()
    torch.set_num_threads(4)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    commit=getattr(model.config,'_commit_hash',None)
    identity=json.dumps({'model':args.model,'commit':commit,'schema':'residue_mean_population_std_chunk1022_v1'},sort_keys=True)
    cache=output.parent/'sequence_pool_cache'/hashlib.sha256(identity.encode()).hexdigest();cache.mkdir(parents=True,exist_ok=True)
    records=[r for r in read_manifest(args.manifest) if r.split in args.splits]
    if not records:raise ValueError('no requested samples')
    for i,r in enumerate(sorted(records,key=lambda r:len(r.sequence))):
        target=cache/f'{r.sequence_sha256}.npz'
        if not target.exists():
            mean,std=encode_residue_pools(r.sequence,tokenizer,model,'cuda')
            tmp=target.with_suffix('.tmp')
            with tmp.open('wb') as f:np.savez(f,mean=mean,std=std)
            tmp.replace(target)
        if i%100==0:print(f'POOLED {i+1}/{len(records)}',flush=True)
    means=[];stds=[]
    for r in records:
        with np.load(cache/f'{r.sequence_sha256}.npz',allow_pickle=False) as f:means.append(f['mean']);stds.append(f['std'])
    with output.with_suffix('.tmp').open('wb') as f:np.savez(f,mean=np.array(means),std=np.array(stds),keys=np.array([r.split+'::'+r.protein_id for r in records]))
    output.with_suffix('.tmp').replace(output)
    atomic_json(output.with_suffix('.provenance.json'),{'feature_identity':json.loads(identity),'splits':args.splits,'count':len(records),'labels_used':False,'manifest_sha256':hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()})
    print('POOLING_COMPLETE',flush=True)

if __name__=='__main__':main()
