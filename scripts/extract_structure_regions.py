"""Read PHOPT development SaProt/graph caches; no pretrained model inference."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
import sys
import time
import hashlib
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from develop_phgeofuse_regression import ROOT,OUT
from phgeofuse.cache import atomic_json
from phgeofuse.io import read_manifest
from phgeofuse.retrieval import record_key
from phgeofuse.structure_pooling import regional_pool


def main():
    torch.set_num_threads(4)
    out=OUT/'saprot_regions_20260915'
    out.mkdir(exist_ok=False)
    records=[r for r in read_manifest(ROOT/'artifacts/phgeofuse/manifest.csv')
             if r.split in ('train','validation')]
    assert len(records)==7884
    data={};means=[];stds=[];locals_=[];sources=[];projection=None
    for i,r in enumerate(records):
        if r.sequence_sha256 not in data:
            embedding=torch.load(r.embedding_path,map_location='cpu')
            graph=torch.load(r.graph_path,map_location='cpu')
            assert embedding['metadata']['sequence_sha256']==r.sequence_sha256
            assert graph['metadata']['sequence_sha256']==r.sequence_sha256
            h=embedding['embedding'].float().numpy()
            if projection is None:
                projection=np.random.default_rng(42).normal(size=(h.shape[1],64))/8.
                np.save(out/'projection.npy',projection)
            data[r.sequence_sha256]=regional_pool(r.sequence,h,graph['rsa'].numpy(),
                graph['plddt'].numpy(),projection)
            sources.append({'sha256':r.sequence_sha256,'embedding_path':r.embedding_path,
                'embedding_key':embedding['metadata']['embedding_key'],
                'graph_path':r.graph_path,'graph_metadata':graph['metadata']})
        mean,std,local=data[r.sequence_sha256]
        means.append(mean);stds.append(std);locals_.append(local)
        if i%200==0 or i==len(records)-1:
            status={'status':'running','pid':os.getpid(),'complete':i+1,'count':len(records),'updated':time.time()}
            atomic_json(out/'status.json',status)
            print('REGIONS_COMPLETE',i+1,len(records),flush=True)
    np.savez(out/'features.npz',mean=np.asarray(means),std=np.asarray(stds),local=np.asarray(locals_),
        keys=np.asarray([record_key(r) for r in records]))
    atomic_json(out/'provenance.json',{'labels_used':False,'splits':['train','validation'],
        'model':'existing general-pretrained SaProt residue embeddings','count':len(records),
        'projection':'fixed Gaussian 1280 to 64, seed42, no dataset fitting',
        'regions':['rsa>=0.25','rsa<0.25','DE','HKR','H','C'],
        'features':'region-minus-global mean divided by global mean norm, fixed projection; '
                   'region fractions and missing flags; structure confidence and RSA summaries',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'pooling_sha256':hashlib.sha256((ROOT/'phgeofuse/structure_pooling.py').read_bytes()).hexdigest(),
        'sources':sources})
    atomic_json(out/'status.json',{'status':'complete','pid':os.getpid(),'count':len(records),'updated':time.time()})
    print('STRUCTURE_REGIONS_COMPLETE',flush=True)


if __name__=='__main__':main()
