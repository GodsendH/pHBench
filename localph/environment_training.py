"""Training utilities for audited frozen-residue transfer experiments."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .phenv_data import sha_file


class TokenRows(Dataset):
    def __init__(self, packed, indices, labels):
        self.packed,self.indices,self.labels=packed,np.asarray(indices),np.asarray(labels)
    def __len__(self): return len(self.indices)
    def __getitem__(self,i):
        idx=int(self.indices[i]); a,b=self.packed['offsets'][idx:idx+2]
        return self.packed['tokens'][a:b],float(self.labels[idx]),idx


class ShardedTokens:
    """Read-only virtual concatenation; a protein never spans cache shards."""
    def __init__(self, arrays):
        self.arrays=arrays
        self.bounds=np.r_[0,np.cumsum([len(a) for a in arrays])]
        self.shape=(int(self.bounds[-1]),arrays[0].shape[1])
        if any(a.shape[1]!=self.shape[1] for a in arrays): raise ValueError('shard widths differ')
    def __len__(self): return self.shape[0]
    def __getitem__(self,index):
        if not isinstance(index,slice) or index.step not in (None,1): raise ValueError('use contiguous protein slices')
        start,stop=index.start,index.stop
        shard=int(np.searchsorted(self.bounds,start,side='right')-1)
        if not 0<=shard<len(self.arrays) or stop>self.bounds[shard+1] or stop<=start:
            raise ValueError('protein slice crosses a cache boundary')
        return self.arrays[shard][start-self.bounds[shard]:stop-self.bounds[shard]]


def collate_tokens(rows):
    x=torch.zeros(len(rows),max(len(r[0]) for r in rows),rows[0][0].shape[1])
    mask=torch.zeros(x.shape[:2],dtype=torch.bool)
    for i,(tokens,_,_) in enumerate(rows):
        x[i,:len(tokens)]=torch.from_numpy(np.array(tokens,dtype=np.float32))
        mask[i,:len(tokens)]=True
    return x,mask,torch.tensor([r[1] for r in rows]),torch.tensor([r[2] for r in rows])


def load_cache(directory, verify=True):
    directory=Path(directory)
    cert=json.loads((directory/'complete.json').read_text())
    if cert['state']!='complete': raise ValueError('cache incomplete')
    if 'protocol_sha256' in cert and sha_file(directory/'protocol.json')!=cert['protocol_sha256']:
        raise ValueError('cache protocol differs')
    if 'shards' in cert:
        arrays=[]; key_parts=[]; offset_parts=[np.array([0])]; offset=0
        for row in cert['shards']:
            path=directory/row['directory']
            if path.resolve().parent!=directory.resolve(): raise ValueError('shards must be direct child directories')
            if sha_file(path/'complete.json')!=row['complete_sha256']: raise ValueError('shard certificate changed')
            packed,_=load_cache(path,verify)
            arrays.append(packed['tokens']); key_parts.append(packed['keys'])
            offset_parts.append(packed['offsets'][1:]+offset); offset+=len(packed['tokens'])
        keys=np.concatenate(key_parts)
        if len(set(keys))!=len(keys): raise ValueError('duplicate keys across shards')
        return {'keys':keys,'offsets':np.concatenate(offset_parts),'tokens':ShardedTokens(arrays)},cert
    if verify:
        for name,digest in cert['hashes'].items():
            if sha_file(directory/name)!=digest: raise ValueError('cache checksum differs')
    with np.load(directory/'index.npz',allow_pickle=False) as z:
        keys=z['keys'].copy(); offsets=z['offsets'].copy()
    tokens=np.load(directory/'tokens.npy',mmap_mode='r')
    if len(set(keys))!=len(keys) or len(offsets)!=len(keys)+1 or offsets[0]!=0 or offsets[-1]!=len(tokens) or not (np.diff(offsets)>0).all():
        raise ValueError('invalid cache coverage')
    return {'keys':keys,'offsets':offsets,'tokens':tokens},cert


@torch.inference_mode()
def predict(model,packed,indices,baseline=None,device='cuda'):
    model.eval()
    loader=DataLoader(TokenRows(packed,indices,np.zeros(len(packed['keys']))),batch_size=32,
                      shuffle=False,collate_fn=collate_tokens,num_workers=0)
    result=[]
    for x,mask,_,idx in loader:
        if baseline is None: p=model.predict_environment(x.to(device),mask.to(device))
        else: p=model(x.to(device),mask.to(device),torch.as_tensor(baseline[idx.numpy()],dtype=torch.float32,device=device))
        result.append(p.cpu().numpy())
    return np.concatenate(result)


def organism_metrics(y,p,organisms):
    y,p,organisms=np.asarray(y),np.asarray(p),np.asarray(organisms)
    result={}
    for name,mask in {'all':np.ones(len(y),bool),'acid':y<=4,'alkaline':y>=10,'core':(y>4)&(y<10)}.items():
        error=p[mask]-y[mask]
        orgs=organisms[mask]
        means=[float(np.mean(error[orgs==g]**2)) for g in np.unique(orgs)]
        result[name]={'sequences':int(mask.sum()),'organisms':len(means),
                      'rmse':float(np.sqrt(np.mean(error**2))) if len(error) else None,
                      'organism_macro_rmse':float(np.sqrt(np.mean(means))) if means else None}
    return result
