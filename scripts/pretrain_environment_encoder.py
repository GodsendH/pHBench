"""Pretrain a compact residue encoder on pHenv, with organism-heldout selection."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from localph.environment_transfer import EnvironmentTransfer
from localph.environment_training import TokenRows,collate_tokens,load_cache,predict,organism_metrics
from localph.phenv_data import sha_file,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-epochs',type=int,default=30)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    cert=json.loads((a.data/'complete.json').read_text())
    if sha_file(a.data/'records.csv')!=cert['records_sha256']: raise ValueError('data differs')
    packed,cache_cert=load_cache(a.cache)
    cache_protocol=json.loads((a.cache/'protocol.json').read_text())
    if cache_protocol['certificate_sha256']!=sha_file(a.data/'complete.json'):
        raise ValueError('cache prepared from different data')
    with (a.data/'records.csv').open(newline='') as f: rows={r['key']:r for r in csv.DictReader(f)}
    if set(rows)!=set(packed['keys']): raise ValueError('all input records must be encoded')
    ordered=[rows[k] for k in packed['keys']]
    labels=np.array([float(r['phenv']) for r in ordered]); org=np.array([r['organism'] for r in ordered])
    split=np.array([r['split'] for r in ordered])
    tr=np.flatnonzero(split=='train'); va=np.flatnonzero(split=='validation')
    if not len(tr) or not len(va) or set(org[tr])&set(org[va]): raise ValueError('organism split failed')
    weights=np.array([float(r['train_weight']) if r['split']=='train' else 0. for r in ordered],dtype=np.float32)
    if not np.isfinite(weights).all() or (weights[tr]<=0).any() or not np.isclose(weights[tr].mean(),1.):
        raise ValueError('invalid training weights')
    protocol={'task':'pHenv','seed':a.seed,'input_dim':1280,'width':32,'dropout':.1,'batch_size':32,
      'optimizer':{'name':'AdamW','lr':.001,'weight_decay':.05},'max_epochs':a.max_epochs,'patience':5,
      'selection':'organism macro MSE + 0.05*(acid organism macro MSE + alkaline organism macro MSE)',
      'weights':'equal total weight per training organism; globally mean normalized; batch loss is mean(w*error^2)',
      'data_certificate_sha256':sha_file(a.data/'complete.json'),'cache_complete_sha256':sha_file(a.cache/'complete.json'),
      'encoder_checkpoint_sha256':cache_protocol.get('checkpoint_sha256'),
      'source_sha256':{str(path.relative_to(ROOT)):sha_file(path) for path in [Path(__file__),ROOT/'localph/environment_transfer.py',ROOT/'localph/environment_training.py']},
      'train_keys':packed['keys'][tr].tolist(),'validation_keys':packed['keys'][va].tolist(),'phopt_labels_consumed':False}
    write_json(a.output/'protocol.json',protocol)
    torch.manual_seed(a.seed); torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False
    model=EnvironmentTransfer().to(a.device)
    optimizer=torch.optim.AdamW(list(model.encoder.parameters())+list(model.environment_head.parameters()),lr=.001,weight_decay=.05)
    loader=DataLoader(TokenRows(packed,tr,labels),batch_size=32,shuffle=True,
        generator=torch.Generator().manual_seed(a.seed),collate_fn=collate_tokens,num_workers=0)
    all_weights=torch.as_tensor(weights,device=a.device)
    best=float('inf'); best_state=None; stable=float('inf'); stale=0; history=[]; start=time.monotonic()
    if a.max_epochs<1: raise ValueError('max epochs must be positive')
    for epoch in range(1,a.max_epochs+1):
        model.train(); total=0.; n=0
        for x,mask,y,idx in loader:
            optimizer.zero_grad(set_to_none=True)
            pred=model.predict_environment(x.to(a.device),mask.to(a.device))
            loss=((pred-y.to(a.device)).square()*all_weights[idx.to(a.device)]).mean()
            if not torch.isfinite(loss): raise ValueError('nonfinite environment loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
            total+=float(loss.detach())*len(y); n+=len(y)
        vp=predict(model,packed,va,device=a.device)
        metrics=organism_metrics(labels[va],vp,org[va])
        if any(metrics[g]['organism_macro_rmse'] is None for g in ('all','acid','alkaline')):
            raise ValueError('validation needs all three endpoints')
        objective=metrics['all']['organism_macro_rmse']**2+.05*sum(metrics[g]['organism_macro_rmse']**2 for g in ('acid','alkaline'))
        row={'epoch':epoch,'training_loss':total/n,'validation':metrics,'objective':objective,'seconds':time.monotonic()-start}
        history.append(row); write_json(a.output/'history.json',history); print(json.dumps(row),flush=True)
        if objective<best:
            best=objective; best_epoch=epoch; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if objective<stable-.002: stable=objective; stale=0
        else: stale+=1
        if stale>=5: break
    model.load_state_dict(best_state)
    tp=predict(model,packed,tr,device=a.device); vp=predict(model,packed,va,device=a.device)
    payload={'task':'pHenv','architecture':{'input_dim':1280,'width':32,'dropout':.1,'affine':True},
        'state_dict':best_state,'epoch':best_epoch,'protocol_sha256':sha_file(a.output/'protocol.json')}
    torch.save(payload,a.output/'weights.pt')
    np.savez(a.output/'predictions.npz',train_keys=packed['keys'][tr],train_prediction=tp,
        validation_keys=packed['keys'][va],validation_prediction=vp)
    verification_model=EnvironmentTransfer().cpu()
    reloaded=torch.load(a.output/'weights.pt',map_location='cpu',weights_only=True)
    verification_model.load_state_dict(reloaded['state_dict'])
    check=va[:min(32,len(va))]
    cp=predict(verification_model,packed,check,device='cpu')
    difference=float(np.max(abs(cp-vp[:len(check)])))
    if difference>1e-4: raise ValueError('CPU reload failed')
    complete={'state':'complete','task':'pHenv','selected_epoch':best_epoch,'epochs_run':len(history),
        'train':organism_metrics(labels[tr],tp,org[tr]),'validation':organism_metrics(labels[va],vp,org[va]),
        'CPU_reload_max_error':difference,'weights_sha256':sha_file(a.output/'weights.pt'),
        'protocol_sha256':sha_file(a.output/'protocol.json'),'seconds':time.monotonic()-start,
        'enzyme_improvement_established':False,'default_model_replaced':False}
    write_json(a.output/'complete.json',complete); print(json.dumps(complete),flush=True)


if __name__=='__main__': main()
