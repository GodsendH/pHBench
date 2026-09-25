"""Fixed-recipe family CV of pHenv transfer using excluded complete baselines.

No epoch/recipe/strength is selected using outer labels. Report every recipe;
these reused development folds are not independent confirmation evidence.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from localph.environment_transfer import EnvironmentTransfer,training_bin_weights,enzyme_objective
from localph.environment_training import TokenRows,collate_tokens,load_cache,predict
from localph.phenv_data import sha_file,write_json
from phgeofuse.delta_ref.data import DevelopmentData,stable_hash
from phgeofuse.delta_ref.metrics import metrics,acceptance


@torch.inference_mode()
def encode(model,packed,device):
    loader=DataLoader(TokenRows(packed,np.arange(len(packed['keys'])),np.zeros(len(packed['keys']))),
                      batch_size=32,shuffle=False,collate_fn=collate_tokens,num_workers=0)
    model.eval(); result=[]
    for x,mask,_,_ in loader: result.append(model.encoder(x.to(device),mask.to(device)).cpu())
    return torch.cat(result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--environment',type=Path,required=True)
    p.add_argument('--phopt-cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    ec=json.loads((a.environment/'complete.json').read_text())
    if ec['task']!='pHenv' or sha_file(a.environment/'weights.pt')!=ec['weights_sha256']:
        raise ValueError('environment checkpoint not certified')
    env=torch.load(a.environment/'weights.pt',map_location='cpu',weights_only=True)
    if env['task']!='pHenv': raise ValueError('checkpoint is not environment pretraining')
    if sha_file(a.environment/'protocol.json')!=env['protocol_sha256']:
        raise ValueError('environment protocol differs')
    env_protocol=json.loads((a.environment/'protocol.json').read_text())
    source=ROOT/'experiments/field_comparisons_phopt_20260917/esm1v_full_precision'
    phopt_source=json.loads((source/'protocol.json').read_text())
    packed_protocol=json.loads((a.phopt_cache/'protocol.json').read_text())
    if (env_protocol.get('encoder_checkpoint_sha256')!=phopt_source['checkpoint_sha256']
        or packed_protocol['source_complete_sha256']!=sha_file(source/'complete.json')):
        raise ValueError('environment and PHOPT PLM sources differ')
    data=DevelopmentData.load(ROOT/'configs/delta_ref_phopt.yaml')
    t=data.train
    keys,y,folds,groups=data.keys[t],data.labels[t],data.folds[t],data.groups[t]
    # Existing full PHOPT cache uses the same shape and checksum certificate format.
    packed,cert=load_cache(a.phopt_cache)
    if not np.array_equal(keys,packed['keys']): raise ValueError('PHOPT cache keys differ')
    recipes=[
      {'name':'phopt_only_weighted','pretrained':False,'freeze':False,'affine':True,'mix':.5,'epochs':10},
      {'name':'env_finetune_weighted','pretrained':True,'freeze':False,'affine':True,'mix':.5,'epochs':10},
      {'name':'env_frozen_weighted','pretrained':True,'freeze':True,'affine':True,'mix':.5,'epochs':200},
      {'name':'env_frozen_natural','pretrained':True,'freeze':True,'affine':True,'mix':0.,'epochs':200},
      {'name':'env_frozen_additive_weighted','pretrained':True,'freeze':True,'affine':False,'mix':.5,'epochs':200}]
    baseline_root=ROOT/'experiments/delta_ref_phopt_20260916/baseline/seed42'
    baseline_hashes={}
    for d in sorted(baseline_root.glob('excluded_*')):
        for name in ('fit.json','predictions.npz'): baseline_hashes[str((d/name).relative_to(ROOT))]=sha_file(d/name)
    protocol={'scope':'fixed-recipe exploratory five-fold family CV, seed42; no outer selection',
      'recipes':recipes,'core_strength':1.,'encoder_width':32,'dropout':.1,
      'end_to_end_optimizer':'AdamW lr0.001 weight_decay0.05 batch32 gradient_clip1',
      'frozen_head_optimizer':'AdamW lr0.01 weight_decay0.05 full training batch gradient_clip1',
      'frozen_optimization_note':'More epochs for cheap head fitting; do not attribute its difference solely to freezing.',
      'baseline_training':'Each training residual uses complete baseline excluding outer and its own fold; query excludes outer.',
      'baseline_hashes':baseline_hashes,'environment_weights_sha256':ec['weights_sha256'],
      'environment_complete_sha256':sha_file(a.environment/'complete.json'),'phopt_cache_sha256':sha_file(a.phopt_cache/'complete.json'),
      'source_sha256':{str(f.relative_to(ROOT)):sha_file(f) for f in [Path(__file__),ROOT/'localph/environment_transfer.py',ROOT/'localph/environment_training.py',ROOT/'phgeofuse/delta_ref/metrics.py']},
      'test_access':False,'original_validation_used':False,'default_model_replaced':False}
    write_json(a.output/'protocol.json',protocol)
    def load_baseline(excluded):
        fit=np.flatnonzero(~np.isin(folds,excluded)); q=np.flatnonzero(np.isin(folds,excluded))
        directory=baseline_root/('excluded_'+'_'.join(map(str,sorted(excluded))))
        certificate=json.loads((directory/'fit.json').read_text())
        if certificate['fit_keys']!=keys[fit].tolist() or certificate['query_keys']!=keys[q].tolist() or certificate['fit_label_sha256']!=stable_hash(y[fit].tolist()) or set(groups[fit])&set(groups[q]):
            raise ValueError('excluded baseline isolation failed')
        with np.load(directory/'predictions.npz',allow_pickle=False) as z:
            if not np.array_equal(z['keys'],keys[q]): raise ValueError('baseline rows differ')
            return q,z['prediction'].copy()
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False
    template=EnvironmentTransfer().to(a.device); template.load_state_dict(env['state_dict']); template.freeze_environment()
    z_all=encode(template,packed,a.device).to(a.device)
    baseline=np.full(len(y),np.nan)
    predictions={r['name']:np.full(len(y),np.nan) for r in recipes}
    reports=[]; start=time.monotonic()
    for outer in range(5):
        fit=np.flatnonzero(folds!=outer); query,base_query=load_baseline([outer])
        base_train=np.full(len(y),np.nan)
        for inner in sorted(set(range(5))-{outer}):
            q,b=load_baseline([outer,inner]); keep=folds[q]==inner; base_train[q[keep]]=b[keep]
        if not np.isfinite(base_train[fit]).all() or not np.isnan(base_train[query]).all():
            raise ValueError('training baseline leaked or incomplete')
        baseline[query]=base_query
        base=base_train.copy(); base[query]=base_query
        bt=torch.as_tensor(base,dtype=torch.float32,device=a.device)
        yt=torch.as_tensor(y,dtype=torch.float32,device=a.device)
        fi=torch.as_tensor(fit,device=a.device); qi=torch.as_tensor(query,device=a.device)
        for recipe in recipes:
            torch.manual_seed(42); np.random.seed(42)
            model=EnvironmentTransfer(affine=recipe['affine']).to(a.device)
            if recipe['pretrained']:
                model.encoder.load_state_dict(template.encoder.state_dict())
                model.environment_head.load_state_dict(template.environment_head.state_dict())
            model.reset_enzyme_head()
            if recipe['freeze']: model.freeze_environment()
            w,weight_cert=training_bin_weights(y[fit],power=1.,mix=recipe['mix'])
            wf=np.zeros(len(y),dtype=np.float32); wf[fit]=w; wt=torch.as_tensor(wf,device=a.device)
            params=list(model.enzyme_head.parameters()) if recipe['freeze'] else list(model.encoder.parameters())+list(model.enzyme_head.parameters())
            optimizer=torch.optim.AdamW(params,lr=.01 if recipe['freeze'] else .001,weight_decay=.05)
            loader=DataLoader(TokenRows(packed,fit,y),batch_size=32,shuffle=True,
                generator=torch.Generator().manual_seed(42),collate_fn=collate_tokens,num_workers=0)
            run=a.output/f'outer{outer}_{recipe["name"]}'; run.mkdir()
            history=[]; begin=time.monotonic()
            def frozen_prediction(idx):
                c=model.enzyme_head(z_all[idx]); delta=c[:,0]
                if recipe['affine']: delta=delta+c[:,1]*(bt[idx]-7)
                return bt[idx]+delta
            for epoch in range(1,recipe['epochs']+1):
                model.train(); total=0.; n=0
                if recipe['freeze']:
                    optimizer.zero_grad(set_to_none=True)
                    loss,_=enzyme_objective(frozen_prediction(fi),bt[fi],yt[fi],wt[fi],1.)
                    if not torch.isfinite(loss): raise ValueError('nonfinite loss')
                    loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.); optimizer.step()
                    total=float(loss.detach())*len(fit); n=len(fit)
                else:
                    for x,mask,labels,idx in loader:
                        idx=idx.to(a.device); optimizer.zero_grad(set_to_none=True)
                        pred=model(x.to(a.device),mask.to(a.device),bt[idx])
                        loss,_=enzyme_objective(pred,bt[idx],labels.to(a.device),wt[idx],1.)
                        if not torch.isfinite(loss): raise ValueError('nonfinite loss')
                        loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.); optimizer.step()
                        total+=float(loss.detach())*len(idx); n+=len(idx)
                history.append({'epoch':epoch,'training_objective':total/n,'seconds':time.monotonic()-begin})
                if not recipe['freeze'] or epoch%50==0:
                    print(json.dumps({'outer':outer,'recipe':recipe['name'],**history[-1]}),flush=True)
                    write_json(run/'history.json',history)
            model.eval()
            if recipe['freeze']:
                with torch.inference_mode(): tp=frozen_prediction(fi).cpu().numpy(); qp=frozen_prediction(qi).cpu().numpy()
            else:
                tp=predict(model,packed,fit,base,a.device); qp=predict(model,packed,query,base,a.device)
            predictions[recipe['name']][query]=qp
            torch.save({'architecture':{'affine':recipe['affine']},'state_dict':model.cpu().state_dict()},run/'weights.pt')
            np.savez(run/'predictions.npz',train_keys=keys[fit],train_prediction=tp,query_keys=keys[query],query_prediction=qp)
            cpu=EnvironmentTransfer(affine=recipe['affine']).eval()
            cpu.load_state_dict(torch.load(run/'weights.pt',map_location='cpu',weights_only=True)['state_dict'])
            check=predict(cpu,packed,query[:8],base,device='cpu')
            err=float(np.max(abs(check-qp[:8])))
            if err>1e-4: raise ValueError('CPU checkpoint verification failed')
            row={'outer':outer,'recipe':recipe,'seconds':time.monotonic()-begin,'weight_fit':weight_cert,
                 'train_metrics':metrics(y[fit],tp,groups[fit]),'query_metrics':metrics(y[query],qp,groups[query]),
                 'train_keys':keys[fit].tolist(),'query_keys':keys[query].tolist(),'train_label_sha256':stable_hash(y[fit].tolist()),
                 'CPU_reload_max_error':err,'weights_sha256':sha_file(run/'weights.pt')}
            write_json(run/'complete.json',row); reports.append(row)
    bm=metrics(y,baseline,groups)
    summary={name:{'metrics':metrics(y,pred,groups),'acceptance':acceptance(metrics(y,pred,groups),bm)} for name,pred in predictions.items()}
    np.savez(a.output/'predictions.npz',keys=keys,labels=y,groups=groups,folds=folds,baseline=baseline,**predictions)
    result={'baseline':bm,'candidates':summary,'seconds':time.monotonic()-start,'test_access':False,
            'default_model_replaced':False,'fit_count':len(reports),'predictions_sha256':sha_file(a.output/'predictions.npz')}
    write_json(a.output/'results.json',result); print(json.dumps(result),flush=True)


if __name__=='__main__': main()
