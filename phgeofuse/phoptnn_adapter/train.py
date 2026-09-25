"""Manifest-driven upstream pHoptNN training; portable checkpoints and predictions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .constants import ALL_ATOM_LABELS
from .graphs import atomic_json, digest, GRAPH_VERSION
from .model import EGNN


class AtomDataset(Dataset):
    def __init__(self, rows):
        self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        r=self.rows[index]
        with np.load(r['graph_path'],allow_pickle=False) as f:
            return dict(key=r['key'],label=float(r['label']),
                        **{k:f[k].copy() for k in ['positions','charges','atom_type','edge_index','edge_attr']})


def collate_graphs(samples):
    b=len(samples);n=max(len(s['charges']) for s in samples)
    pos=torch.zeros(b,n,3);charge=torch.zeros(b,n);types=torch.zeros(b,n,37)
    mask=torch.zeros(b,n,1);edges=[];attrs=[]
    for i,s in enumerate(samples):
        length=len(s['charges'])
        pos[i,:length]=torch.from_numpy(s['positions'])
        charge[i,:length]=torch.from_numpy(s['charges'])
        types[i,:length]=torch.nn.functional.one_hot(torch.from_numpy(s['atom_type']).long(),37).float()
        mask[i,:length]=1
        edges.append(torch.from_numpy(s['edge_index']).long()+i*n)
        attrs.append(torch.from_numpy(s['edge_attr']))
    edge=torch.cat(edges,dim=1)
    return dict(positions=pos,charges=charge,one_hot=types,node_mask=mask,edges=edge,
                edge_attr=torch.cat(attrs),edge_mask=torch.ones(edge.shape[1],1),
                labels=torch.tensor([s['label'] for s in samples]),keys=[s['key'] for s in samples])


def predict_batch(model,batch,device,charge_scale=1.):
    b,n,_=batch['positions'].shape
    charges=batch['charges'].to(device)
    onehot=batch['one_hot'].to(device)
    powers=torch.arange(3,device=device,dtype=torch.float32)
    features=onehot.unsqueeze(-1)*(charges[:,:,None,None]/charge_scale).pow(powers)
    return model(h0=features.reshape(b*n,111),x=batch['positions'].to(device).reshape(b*n,3),
                 edges=batch['edges'].to(device),edge_attr=batch['edge_attr'].to(device),
                 node_mask=batch['node_mask'].to(device).reshape(b*n,1),
                 edge_mask=batch['edge_mask'].to(device),n_nodes=n)


def create_model(config,device):
    return EGNN(in_node_nf=111,in_edge_nf=5,hidden_nf=config['hidden_dim'],
                device=device,n_layers=config['layers'],node_attr=0,attention=config['attention'])


def label_weights(labels,power=.99501715,bins=65):
    # Best_hp row 6 uses ks=1: LDS convolution is identity.
    ids=np.clip((np.asarray(labels)/14*bins).astype(int),0,bins-1)
    counts=np.bincount(ids,minlength=bins)
    values=1/np.maximum(counts,1).astype(float)**power
    values/=np.mean(values[ids])
    return values.astype('float32')


@torch.no_grad()
def evaluate(model,loader,device,mean,std):
    model.eval();rows=[]
    for batch in loader:
        prediction=predict_batch(model,batch,device).float().cpu().numpy()*std+mean
        for k,y,p in zip(batch['keys'],batch['labels'].numpy(),prediction):
            rows.append(dict(key=k,label=float(y),prediction=float(p)))
    y=np.array([r['label'] for r in rows]);p=np.array([r['prediction'] for r in rows])
    if not len(rows) or not np.isfinite(p).all():raise ValueError('empty/nonfinite evaluation')
    return rows,dict(count=len(rows),rmse=float(np.sqrt(np.mean((y-p)**2))),mae=float(np.mean(abs(y-p))))


def save_checkpoint(path,payload):
    tmp=path.with_suffix('.tmp.pt');torch.save(payload,tmp);os.replace(tmp,path)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--epochs',type=int,default=1000)
    ap.add_argument('--patience',type=int,default=30)
    ap.add_argument('--batch-size',type=int,default=1)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--hidden-dim',type=int,default=111)
    ap.add_argument('--layers',type=int,default=16)
    ap.add_argument('--lr',type=float,default=.00006752)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--predict-only',action='store_true')
    ap.add_argument('--allow-failures',action='store_true')
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    allrows=list(csv.DictReader(args.manifest.open()))
    if len({r['key'] for r in allrows})!=len(allrows):raise ValueError('duplicate sample keys')
    failed=[r for r in allrows if r['status']!='ready']
    if failed and not args.allow_failures:raise ValueError(f'{len(failed)} graph failures; review coverage before training')
    rows=[r for r in allrows if r['status']=='ready']
    parts={s:[r for r in rows if r['split']==s] for s in ['train','validation','test']}
    if any(not v for v in parts.values()):raise ValueError('missing required split')
    train_y=np.array([float(r['label']) for r in parts['train']])
    mean,std=float(train_y.mean()),float(train_y.std())
    config=dict(graph_version=GRAPH_VERSION,manifest_sha256=digest(args.manifest),seed=args.seed,
                hidden_dim=args.hidden_dim,layers=args.layers,attention=True,charge_power=2,charge_scale=1.,
                charge_scale_policy='fixed physical partial-charge unit; trained from scratch',
                y_mean=mean,y_std=std,lr=args.lr,weight_decay=.00000031,optimizer='Adam',
                atom_labels=ALL_ATOM_LABELS,batch_size=args.batch_size,epochs=args.epochs,patience=args.patience,
                lds_power=.99501715,lds_bins=65,lds_ks=1,lds_weight_normalization='training mean=1',
                gradient_clip=10.,precision='float32',
                split_counts={k:len(v) for k,v in parts.items()},failed_keys=[r['key'] for r in failed],
                source='pHoptNN Best_hp row 6; fresh supervised parameters; fixed PHOPT splits')
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=config:
        raise ValueError('run config drift')
    atomic_json(out/'config.json',config)
    if (out/'last.pt').exists() and not(args.resume or args.predict_only):raise FileExistsError('use --resume')
    generator=torch.Generator().manual_seed(args.seed)
    loaders={s:DataLoader(AtomDataset(rs),batch_size=args.batch_size,shuffle=s=='train',
                         num_workers=args.workers,collate_fn=collate_graphs,pin_memory=device.type=='cuda',
                         persistent_workers=args.workers>0,generator=generator if s=='train' else None)
             for s,rs in parts.items()}
    model=create_model(config,device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=config['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=10)
    weight=torch.tensor(label_weights(train_y),device=device)
    start=0;best=math.inf;stale=0
    if args.resume and (out/'last.pt').exists():
        cp=torch.load(out/'last.pt',map_location=device,weights_only=False)
        if cp['config']!=config:raise ValueError('checkpoint config differs')
        model.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer']);scheduler.load_state_dict(cp['scheduler'])
        start=cp['epoch']+1;best=cp['best'];stale=cp['stale']
        torch.set_rng_state(cp['torch_rng'].cpu());generator.set_state(cp['loader_rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state_all([v.cpu() for v in cp['cuda_rng']])
    started=time.time()
    if not args.predict_only:
        for epoch in range(start,args.epochs):
            if stale>=args.patience:break
            model.train();loss_sum=0.;count=0;epoch_start=time.time()
            for step,batch in enumerate(loaders['train']):
                optimizer.zero_grad(set_to_none=True)
                prediction=predict_batch(model,batch,device)
                y=batch['labels'].to(device);target=(y-mean)/std
                bins=(y/14*65).long().clamp(0,64)
                loss=((prediction-target).square()*weight[bins]).mean()
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
                loss_sum+=float(loss.detach())*len(y);count+=len(y)
                if step%100==0:
                    atomic_json(out/'status.json',dict(status='training',pid=os.getpid(),epoch=epoch,step=step,
                                samples=count,total=len(parts['train']),updated=time.time(),seconds=time.time()-started))
            _,metric=evaluate(model,loaders['validation'],device,mean,std)
            scheduler.step(metric['rmse']**2)
            improved=metric['rmse']<best-1e-4
            if improved:best=metric['rmse'];stale=0
            else:stale+=1
            cp=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                    config=config,epoch=epoch,best=best,stale=stale,torch_rng=torch.get_rng_state(),
                    loader_rng=generator.get_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [])
            save_checkpoint(out/'last.pt',cp)
            if improved:save_checkpoint(out/'best.pt',cp)
            row=dict(epoch=epoch,train_weighted_mse=loss_sum/count,validation=metric,best=best,stale=stale,
                     seconds=time.time()-epoch_start,lr=optimizer.param_groups[0]['lr'])
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
    cp=torch.load(out/'best.pt',map_location=device,weights_only=False);model.load_state_dict(cp['model'])
    results={}
    for split in ['validation','test']:
        predictions,metric=evaluate(model,loaders[split],device,mean,std)
        mapping={r['key']:r for r in parts[split]}
        model_hash=digest(out/'best.pt')
        for r in predictions:
            r.update(sequence_sha256=mapping[r['key']]['sequence_sha256'],model_hash=model_hash,status='ready')
        with (out/f'{split}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
        results[split]=metric
    atomic_json(out/'results.json',dict(best_epoch=cp['epoch'],metrics=results,config=config))
    atomic_json(out/'status.json',dict(status='complete',pid=os.getpid(),updated=time.time(),best_epoch=cp['epoch'],metrics=results))


if __name__=='__main__':main()
