"""Single-GPU ESM2 adaptation with a reusable, label-free frozen-prefix cache.

Only query/value projections in the last four encoder layers get rank-8
adapters. The first 29 layers are immutable and are evaluated once per sequence
chunk. Prefix caches preserve float32 activations, special tokens and all
residues; no long sequence is silently truncated.
"""
from __future__ import annotations

import copy
from functools import lru_cache
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from phgeofuse.cache import atomic_json, atomic_torch_save, sha256_file
from .data import freeze_json, stable_hash, atomic_npz
from .model import DeltaNetwork, Standardizer, ReferencePredictor, select_panel, blend_predictions
from .training import seed_all, sample_pairs, frequency_weights, save_bundle, load_bundle
from .metrics import select_strength, metrics


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16, dropout=.05):
        super().__init__()
        self.base=base
        self.base.requires_grad_(False)
        self.a=nn.Linear(base.in_features,rank,bias=False)
        self.b=nn.Linear(rank,base.out_features,bias=False)
        nn.init.kaiming_uniform_(self.a.weight,a=5**.5)
        nn.init.zeros_(self.b.weight)
        self.dropout=nn.Dropout(dropout)
        self.scale=alpha/rank

    def forward(self,x):
        return self.base(x)+self.b(self.a(self.dropout(x)))*self.scale


class _PrefixCaptured(Exception):
    pass


def pool_chunk_residues(chunks, lengths):
    """Differentiable global moments excluding CLS/EOS; equal residue weight."""
    if len(chunks)!=len(lengths) or not chunks or any(n<1 for n in lengths):
        raise ValueError('invalid residue chunks')
    total=sum(lengths)
    first=None;second=None
    for h,length in zip(chunks,lengths):
        if h.shape[0]!=length+2 or not torch.isfinite(h).all():
            raise ValueError('one token per residue plus CLS/EOS required')
        v=h[1:1+length].float()
        first=v.sum(0) if first is None else first+v.sum(0)
        second=v.square().sum(0) if second is None else second+v.square().sum(0)
    mean=first/total
    variance=(second/total-mean.square()).clamp_min(0)
    # Clamp solely for finite derivatives near numerical zero variance.
    std=variance.clamp_min(1e-12).sqrt()
    return mean,std


class PrefixEncoder(nn.Module):
    def __init__(self,config,cache,device='cuda'):
        super().__init__()
        from transformers import AutoModel,AutoTokenizer
        from transformers.utils.hub import cached_file
        self.settings=dict(config)
        self.device=torch.device(device)
        model_source=Path(cached_file(config['model'],'config.json',local_files_only=True)).parent
        self.tokenizer=AutoTokenizer.from_pretrained(str(model_source),local_files_only=True)
        self.model=AutoModel.from_pretrained(str(model_source),local_files_only=True).to(self.device).eval()
        self.model.requires_grad_(False)
        self.start=len(self.model.encoder.layer)-config['last_layers']
        if self.start<0:raise ValueError('too many adapter layers')
        identity={'model':config['model'],'revision':getattr(self.model.config,'_commit_hash',None) or model_source.name,
                  'prefix_layers':self.start,'schema':'esm_prefix_float32_full_residue_chunks_v1',
                  'chunk_length':config['chunk_length'],'compute_dtype':'bfloat16' if self.device.type=='cuda' else 'float32'}
        if not identity['revision']:raise ValueError('ESM2 revision could not be pinned')
        self.identity=identity
        self.cache=Path(cache)/stable_hash(identity)
        self.cache.mkdir(parents=True,exist_ok=True)
        freeze_json(self.cache/'provenance.json',identity)
        for layer in self.model.encoder.layer[self.start:]:
            for name in ('query','value'):
                original=getattr(layer.attention.self,name)
                setattr(layer.attention.self,name,LoRALinear(original,config['rank'],config['alpha'],config['dropout']).to(self.device))
        self.set_adapter_training(False)

    def set_adapter_training(self,training):
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module,LoRALinear):module.train(training)
        self.adapters_training=training

    def adapter_state(self):
        return {n:p.detach().cpu().clone() for n,p in self.model.named_parameters() if p.requires_grad}

    def load_adapter_state(self,state):
        params={n:p for n,p in self.model.named_parameters() if p.requires_grad}
        if set(params)!=set(state):raise ValueError('adapter parameter schema mismatch')
        with torch.no_grad():
            for name,p in params.items():p.copy_(state[name].to(p))

    def _autocast(self):
        return torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.device.type=='cuda')

    def prefix_file(self,sequence):
        return self.cache/(stable_hash(sequence)+'.pt')

    def prepare(self,sequence):
        if not sequence:raise ValueError('empty sequence')
        destination=self.prefix_file(sequence)
        if destination.exists():return
        chunks=[];sizes=[]
        for start in range(0,len(sequence),self.settings['chunk_length']):
            part=sequence[start:start+self.settings['chunk_length']]
            tokens=self.tokenizer(part,return_tensors='pt',add_special_tokens=True)
            if tokens['input_ids'].shape[1]!=len(part)+2:raise ValueError('token/residue mismatch')
            tokens={k:v.to(self.device) for k,v in tokens.items()}
            captured=[]
            def capture(module,args):
                captured.append(args[0].detach().float().cpu())
                raise _PrefixCaptured()
            hook=self.model.encoder.layer[self.start].register_forward_pre_hook(capture)
            try:
                with torch.no_grad(),self._autocast():
                    try:self.model(**tokens)
                    except _PrefixCaptured:pass
            finally:hook.remove()
            if len(captured)!=1:raise RuntimeError('frozen prefix capture failed')
            chunks.append(captured[0]);sizes.append(len(part))
        atomic_torch_save(destination,{'identity':self.identity,'sequence_sha256':stable_hash(sequence),
                                     'chunks':chunks,'lengths':sizes,'length':len(sequence)})

    @lru_cache(maxsize=192)
    def _load_prefix(self,sequence):
        self.prepare(sequence)
        p=torch.load(self.prefix_file(sequence),map_location='cpu')
        if p['identity']!=self.identity or p['sequence_sha256']!=stable_hash(sequence) or sum(p['lengths'])!=len(sequence):
            raise ValueError('prefix cache provenance/coverage differs')
        return p

    def encode(self,sequence,gradient=False):
        p=self._load_prefix(sequence);outputs=[]
        with self._autocast():
            for prefix in p['chunks']:
                hidden=prefix.to(self.device)
                if gradient:hidden.requires_grad_(True)
                attention_mask=torch.zeros((1,1,1,hidden.shape[1]),dtype=hidden.dtype,device=self.device)
                for layer in self.model.encoder.layer[self.start:]:
                    def forward(x,module=layer,mask=attention_mask):
                        return module(x,attention_mask=mask)[0]
                    hidden=checkpoint(forward,hidden,use_reentrant=False) if gradient else forward(hidden)
                if self.model.encoder.emb_layer_norm_after is not None:
                    hidden=self.model.encoder.emb_layer_norm_after(hidden)
                outputs.append(hidden[0])
        mean,std=pool_chunk_residues(outputs,p['lengths'])
        return torch.cat((mean/mean.norm().clamp_min(1e-12),std/std.norm().clamp_min(1e-12)))

    @torch.no_grad()
    def encode_rows(self,data,indices):
        self.set_adapter_training(False)
        result=np.asarray(data.x[indices],dtype=np.float64).copy()
        for j,i in enumerate(indices):
            result[j,2560:5120]=self.encode(data.records[i].sequence).float().cpu().numpy()
        return result


class LoRAPredictor:
    def __init__(self,predictor,features,keys,adapter_directory=None):
        self.predictor=predictor
        self.features=np.asarray(features)
        self.order={str(k):i for i,k in enumerate(keys)}
        self.adapter_directory=adapter_directory

    def transfer(self,features,query_keys=None,query_groups=None,**kwargs):
        if query_keys is None:raise ValueError('adapted prediction requires explicit sequence keys')
        if not set(map(str,query_keys))<=set(self.order):raise ValueError('encode new sequences with the fitted adapter before prediction')
        x=self.features[[self.order[str(k)] for k in query_keys]]
        return self.predictor.transfer(x,query_keys,query_groups,**kwargs)

    def predict(self,features,baseline,strength,query_keys=None,query_groups=None,consistency=True):
        t,d,v=self.transfer(features,query_keys,query_groups)
        return blend_predictions(baseline,t,d,strength,v,consistency)

    def predict_records(self,data,indices,baseline,strength):
        return self.predict(data.x[indices],baseline,strength,data.keys[indices])


def fit_lora_subset(data,fit,validation,baseline_validation,recipe,output,seed=42,
                    fixed_epochs=None,excluded=(),device=None,progress=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    config=data.config;settings=config['lora'];device=device or config['training']['device']
    metadata={**data.certificate(fit,validation,excluded),'recipe':recipe,'seed':seed,
              'fixed_epochs':fixed_epochs,'lora_settings':settings,
              'head_settings':config['training'],'panel_settings':config['model'],
              'epoch_selection_label_sha256':None if fixed_epochs else stable_hash(data.labels[validation].tolist())}
    freeze_json(output/'fit.json',metadata)
    if (output/'complete.json').exists():
        base,saved=load_bundle(output,device)
        if saved['metadata']['fit_hash']!=stable_hash(metadata):raise ValueError('LoRA fit provenance differs')
        with np.load(output/'adapted_features.npz',allow_pickle=False) as z:
            if sha256_file(output/'adapter.pt')!=saved['adapter_sha256']:raise ValueError('adapter checksum differs')
            if sha256_file(output/'adapted_features.npz')!=saved['adapted_features_sha256']:raise ValueError('adapted feature checksum differs')
            return LoRAPredictor(base,z['features'],z['keys'],output),json.loads((output/'complete.json').read_text())
    seed_all(seed);torch.set_num_threads(config['training'].get('threads',4))
    root=Path(config['_root'])/'artifacts/delta_ref/esm2_prefix'
    encoder=PrefixEncoder(settings,root,device)
    started=time.monotonic()
    # Preparing only this fit and its legitimate epoch-selection queries keeps
    # supervised adaptation isolated; the cached prefix itself is label-free.
    indices=np.r_[fit,validation]
    for j,i in enumerate(indices):
        encoder.prepare(data.records[i].sequence)
        if j%100==0:print(json.dumps({'event':'lora_prefix','output':str(output),'complete':j,'total':len(indices)}),flush=True)
    scaler=Standardizer.fit(data.x[fit])
    center=torch.as_tensor(scaler.mean,dtype=torch.float32,device=device)
    scale=torch.as_tensor(scaler.scale,dtype=torch.float32,device=device)
    head=DeltaNetwork(data.x.shape[1],recipe['width'],config['model']['dropout'],recipe['kind']).to(device)
    adapter_params=[p for p in encoder.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW([{'params':head.parameters(),'lr':config['training']['learning_rate']},
                                {'params':adapter_params,'lr':settings['learning_rate']}],weight_decay=config['training']['weight_decay'])
    refs,refweights=select_panel(data.embeddings[fit],data.labels[fit],data.groups[fit],data.keys[fit],config['model']['references_per_bin'])
    refs=fit[refs]
    weights=frequency_weights(data.labels[fit],recipe['power'])
    best_state=None;best_adapter=None;best_rank=(float('inf'),)*4;stale=0;history=[];best_epoch=0;strength=0.
    rng=np.random.default_rng(seed)
    accumulation=settings['accumulation_steps']
    def differentiable_feature(index):
        adapted=encoder.encode(data.records[index].sequence,gradient=True)
        fixed=torch.as_tensor(data.x[index],dtype=torch.float32,device=device)
        raw=torch.cat((fixed[:2560],adapted,fixed[5120:]))
        return (raw-center)/scale
    if torch.device(device).type=='cuda':torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,(fixed_epochs or settings['max_epochs'])+1):
        encoder.set_adapter_training(True);head.train();optimizer.zero_grad(set_to_none=True)
        q,r=sample_pairs(data.labels[fit],data.groups[fit],rng)
        total_loss=0.
        epoch_started=time.monotonic()
        for step,(qi,ri) in enumerate(zip(q,r)):
            i,j=fit[qi],fit[ri]
            fq=differentiable_feature(i);fr=differentiable_feature(j)
            prediction=head(fq[None],fr[None])[0]
            loss=(weights[qi]*weights[ri])**.5*(prediction-float(data.labels[i]-data.labels[j])).square()
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite LoRA pair loss')
            # The last, partial accumulation block gets its actual denominator.
            block_start=(step//accumulation)*accumulation
            denominator=min(accumulation,len(q)-block_start)
            (loss/denominator).backward();total_loss+=float(loss.detach())
            if (step+1)%accumulation==0 or step+1==len(q):
                torch.nn.utils.clip_grad_norm_([*head.parameters(),*adapter_params],1.)
                optimizer.step();optimizer.zero_grad(set_to_none=True)
            if step==63:
                profile={'pairs':64,'seconds':time.monotonic()-epoch_started,
                         'estimated_epoch_seconds':(time.monotonic()-epoch_started)*len(q)/64,
                         'peak_gpu_bytes':torch.cuda.max_memory_allocated() if torch.device(device).type=='cuda' else 0,
                         'device':device,'automatic_multigpu':False}
                atomic_json(output/'resource_profile.json',profile)
                print(json.dumps({'event':'lora_resource_profile',**profile}),flush=True)
        row={'epoch':epoch,'training_loss':total_loss/len(q),'pairs':len(q),'seconds':time.monotonic()-started}
        if fixed_epochs is None:
            refx=encoder.encode_rows(data,refs);valx=encoder.encode_rows(data,validation)
            p=ReferencePredictor(head,scaler,refx,data.labels[refs],data.groups[refs],data.keys[refs],refweights,device)
            t,d,v=p.transfer(valx,data.keys[validation])
            choice,_=select_strength(data.labels[validation],baseline_validation,t,d,v,data.groups[validation])
            raw=metrics(data.labels[validation],t,data.groups[validation]);rank=(*choice['rank'],raw['all']['rmse'])
            row.update(validation=choice['metrics'],strength=choice['strength'],standalone=raw)
            if best_state is None or rank<best_rank:
                best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
                best_adapter=encoder.adapter_state();best_rank=rank;best_epoch=epoch;strength=choice['strength'];stale=0
            else:stale+=1
        else:best_epoch=epoch
        history.append(row);atomic_json(output/'history.json',history)
        print(json.dumps({'event':'lora_epoch','output':str(output),**row}),flush=True)
        if fixed_epochs is None and stale>=settings['patience']:break
    if fixed_epochs is None:
        head.load_state_dict(best_state);encoder.load_adapter_state(best_adapter)
    encoded=encoder.encode_rows(data,indices)
    order={int(i):j for j,i in enumerate(indices)}
    refx=encoded[[order[int(i)] for i in refs]]
    predictor=ReferencePredictor(head,scaler,refx,data.labels[refs],data.groups[refs],data.keys[refs],refweights,device)
    report={'best_epoch':best_epoch,'strength':strength,'wall_seconds':time.monotonic()-started,
            'epochs_run':len(history),'fit_hash':stable_hash(metadata),
            'head_parameters':sum(p.numel() for p in head.parameters()),'adapter_parameters':sum(p.numel() for p in adapter_params)}
    atomic_torch_save(output/'adapter.pt',{'state':encoder.adapter_state(),'identity':encoder.identity,'settings':settings})
    atomic_npz(output/'adapted_features.npz',keys=data.keys[indices],features=encoded)
    save_bundle(output,predictor,recipe,strength,{**report,'provenance':metadata})
    saved=json.loads((output/'model.json').read_text())
    saved.update(adapter_sha256=sha256_file(output/'adapter.pt'),adapted_features_sha256=sha256_file(output/'adapted_features.npz'))
    atomic_json(output/'model.json',saved);atomic_json(output/'complete.json',report)
    encoder._load_prefix.cache_clear()
    return LoRAPredictor(predictor,encoded,data.keys[indices],output),report


def nested_lora(data,output,frozen,seed=42):
    from .experiment import nested
    recipe={**frozen['final_selection']['recipe'],'name':'lora_rank8_last4','representation':'lora'}
    # Each outer fold inherits only its own INNER-selected frozen head recipe.
    # The final deployment recipe can use all development folds after evaluation.
    source=Path(output).parent/'frozen'/'outer_results.json'
    rows=json.loads(source.read_text())
    outer_recipes={row['outer']:[{**row['winner']['recipe'],'name':recipe['name'],'representation':'lora'}] for row in rows}
    return nested(data,output,[recipe],seed,fit_function=fit_lora_subset,outer_recipes=outer_recipes)
