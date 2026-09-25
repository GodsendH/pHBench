"""Bounded tail-priority development with one frozen historical test follow-up."""
from __future__ import annotations
import os
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
import argparse
import csv
import itertools
import json
from pathlib import Path
import shutil
import sys
import time
import joblib
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVR

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_dual_tail_weighting import Experiment, HGB, metrics, bootstrap, SOURCE, OLD_BUNDLE
from train_ephod_svr_control import load_features, weight_function
from phgeofuse.cache import atomic_json,sha256_file
from phgeofuse.delta_ref.data import DevelopmentData,atomic_npz,freeze_json,read_predictions,write_predictions
from phgeofuse.dual_fusion import DualFusion
from phgeofuse.io import read_fasta
from phgeofuse.tail_priority import priority_weights,meta_inputs,fit_priority_residual,kernel_predict,TailPriorityFusion

KERNEL_ROOT=ROOT/'experiments/field_comparisons_phopt_20260917'
KERNEL_SOURCE=KERNEL_ROOT/'ephod_svr_control'
WEIGHT_SOURCE=ROOT/'docs/extreme_ph_review_20260916/ephod_official_ephod_training_trainutils.py'
MIXES=[.5,.75,1.]
RECIPES=[dict(name=n,acid_mass=a,alkaline_mass=b,hard_acid=h,under_multiplier=u,kernel=k)
 for n,a,b,h,u,k in [
 ('M0',.05,.05,False,1.,False),('M1',.05,.10,False,1.,False),
 ('M2',.075,.10,False,1.,False),('M3',.05,.15,False,1.,False),
 ('M4',.075,.15,False,1.,False),('M5',.10,.20,False,1.,False),
 ('H1',.05,.10,True,1.,False),('H2',.075,.15,True,1.,False),
 ('A1',.05,.10,True,2.,False),('A2',.05,.15,True,2.,False),
 ('K1',.05,.10,False,1.,True),('K2',.05,.10,True,1.,True),
 ('K3',.075,.15,True,1.,True)]]


def expanded_metrics(y,p,low,groups=None):
    m=metrics(y,p,low,groups)
    for name,mask in [('extreme_acid',y<=4),('extreme_alkaline',y>=10)]:
        e=np.abs(p[mask]-y[mask]);count=max(1,int(np.ceil(.2*len(e))))
        m[name].update(abs_error_p90=float(np.quantile(e,.9)),worst20_mse=float(np.mean(np.sort(e**2)[-count:])),
                       underestimated_fraction=float(np.mean(p[mask]<y[mask])))
    return m


def rank_row(candidate,original,svr):
    budget=candidate['all']['rmse']<=1.05*original['all']['rmse'] and candidate['all']['mae']<=1.045*original['all']['mae']
    ratios={g+'_'+m:candidate[g][m]/svr[g][m] for g in ['extreme_acid','extreme_alkaline'] for m in ['rmse','mae']}
    return dict(proxy_overall_budget=budget,tail_ratios_to_matched_svr=ratios,
                worst_tail_ratio=max(ratios.values()),mean_tail_ratio=float(np.mean(list(ratios.values()))),
                beats_matched_svr_all_tail_metrics=all(v<1 for v in ratios.values()))


def score_key(row):
    x=row['selection']
    return (not x['proxy_overall_budget'],x['worst_tail_ratio'],x['mean_tail_ratio'],row['metrics']['all']['rmse'])


class PriorityExperiment(Experiment):
    def __init__(self,out):
        out.mkdir(parents=True,exist_ok=False)
        self.out=out;self.start=time.monotonic()
        self.data=DevelopmentData.load(ROOT/'configs/delta_ref_phopt.yaml');d=self.data
        self.n=len(d.train);self.y=d.labels[d.train];self.fold=d.folds[d.train];self.chem=d.x[:,-25:]
        if not np.array_equal(d.train,np.arange(self.n)):raise ValueError('noncontiguous training rows')
        official=read_fasta(ROOT/'data/phopt_training.fasta','train')
        sig=lambda rs:sorted((r.protein_id,r.sequence,r.ph_opt) for r in rs)
        if sig(official)!=sig([d.records[i] for i in d.train]):raise ValueError('official training differs')
        self.cache={};self.parity=[];self.sources=dict(d.provenance['files']);self.kernel_cache={}
        self.protocol=dict(seed=42,recipes=RECIPES,mixes=MIXES,hgb=HGB,data=d.provenance,
            source_files={str(p):sha256_file(p) for p in [Path(__file__),ROOT/'phgeofuse/tail_priority.py',
                ROOT/'scripts/evaluate_dual_tail_weighting.py',ROOT/'docs/dual_tail_priority_20260919/PLAN_ZH.md',ROOT/'pdf_text.txt']},
            paper_venus=dict(rmse=.809,mae=.578,doi='10.1021/acs.jcim.4c02291',
                            status='Published Reptile mean; local controlled reproduction incomplete.'),
            official_ephod=dict(acid_rmse=1.5463121962893993,acid_mae=1.2145561258813882,
                                alkaline_rmse=1.8918528990474008,alkaline_mae=1.5532525299350353),
            local_svr=dict(acid_rmse=1.543022441500071,acid_mae=1.1792340019518375,
                           alkaline_rmse=1.8408190361936687,alkaline_mae=1.4920996518323983),
            selection='Train OOF: budget RMSE ratio <=1.05 and MAE ratio <=1.045 vs original; then minimize worst of four tail ratios to matched fold SVR. Top three distinct recipes refit; validation selects exactly one with same rule.',
            final_acceptance='Actual historical test RMSE<.809 and MAE<.578 plus all four strict tail metrics below official EpHod Ensemble; local SVR is separately reported stronger control.',
            kernel='Fixed prior-validation-selected RBF SVR C1 gamma auto LDS_inv, training-subset normalization. All single/double exclusion models refit.',
            epistemic_scope='Exploratory follow-up after prior test inspection. No selected candidate test is read until model release; no test-driven revision inside this run.',
            frozen='Encoders/retrieval/robust branch; changed residual losses and optional kernel sequence expert.')
        freeze_json(out/'protocol.json',self.protocol);self.sources.update(self.protocol['source_files'])
        self.emit('protocol_frozen',recipes=len(RECIPES),mixes=MIXES)

    def prepare_kernel(self):
        keys,split,labels,features=load_features(KERNEL_ROOT/'esm1v_full_precision',ROOT/'artifacts/phgeofuse/manifest.csv')
        self.kernel_keys=keys;self.kernel_features=features
        self.kernel_index={str(k):i for i,k in enumerate(keys)}
        ix=np.array([self.kernel_index[k] for k in self.data.keys])
        self.kx=features[ix]
        if not np.array_equal(labels[ix[:self.n]],self.y):raise ValueError('kernel training labels differ')
        self.get_weights=weight_function(WEIGHT_SOURCE)
        release=json.loads((KERNEL_SOURCE/'release.json').read_text())
        if sha256_file(KERNEL_SOURCE/'model.joblib')!=release['checkpoint_sha256']:raise ValueError('kernel model changed')
        if sha256_file(KERNEL_SOURCE/'protocol.json')!=release['protocol_sha256']:raise ValueError('kernel release changed')
        kp=json.loads((KERNEL_SOURCE/'protocol.json').read_text())
        for name,digest in kp['source_sha256'].items():
            if sha256_file(name)!=digest:raise ValueError('kernel source provenance changed')
            self.sources[name]=digest
        if kp['training_keys']!=self.data.keys[:self.n].tolist():raise ValueError('full kernel training set differs')
        self.full_kernel=joblib.load(KERNEL_SOURCE/'model.joblib')
        for file in [WEIGHT_SOURCE,KERNEL_SOURCE/'release.json',KERNEL_SOURCE/'model.joblib',KERNEL_SOURCE/'protocol.json',KERNEL_ROOT/'esm1v_full_precision/complete.json']:
            self.record(file)
        self.emit('kernel_source_verified',pooling='Official all-token ESM1v float32; training-subset standardization',
                  mixed_feature_archive='Includes label-free test embeddings; no test labels/predictions used for fitting or selection.')

    def kernel_fold(self,excluded):
        excluded=tuple(sorted(excluded))
        if excluded in self.kernel_cache:return self.kernel_cache[excluded]
        fit,q=self.data.partition(excluded);mean=self.kx[fit].mean(0);std=self.kx[fit].std(0)
        x=(self.kx[fit]-mean)/(std+1e-8);xq=(self.kx[q]-mean)/(std+1e-8);gamma=1/x.shape[1]
        w=self.get_weights(self.data.labels[fit],'LDS_inv')
        model=SVR(kernel='precomputed',C=1.,epsilon=.1,tol=.001,shrinking=True,cache_size=512,max_iter=-1)
        model.fit(rbf_kernel(x,x,gamma=gamma),self.data.labels[fit],sample_weight=w)
        if model.fit_status_!=0:raise ValueError('kernel fit failed to converge')
        pred=model.predict(rbf_kernel(xq,x,gamma=gamma))
        folder=self.out/'kernel_folds'/('_'.join(map(str,excluded)));folder.mkdir(parents=True)
        joblib.dump(dict(model=model,mean=mean,std=std,gamma=gamma,fit_indices=fit),folder/'model.joblib')
        atomic_json(folder/'fit.json',dict(certificate=self.data.certificate(fit,q,excluded),
            weight_type='LDS_inv',recipe={'C':1.,'kernel':'rbf','gamma':'auto'},model_sha256=sha256_file(folder/'model.joblib')))
        write_predictions(folder/'training_weights.csv',self.data.keys[fit],dict(label=self.data.labels[fit],weight=w))
        atomic_npz(folder/'predictions.npz',keys=self.data.keys[q],prediction=pred)
        self.kernel_cache[excluded]=(q,pred)
        self.emit('kernel_fold_complete',excluded=list(excluded),training_count=len(fit))
        return q,pred

    def upstream(self,excluded):
        q,v=self.cached(excluded);qk,k=self.kernel_fold(excluded)
        if not np.array_equal(q,qk):raise ValueError('kernel/upstream rows differ')
        return q,{**v,'kernel':k}

    def assemble(self,outer=None):
        fit=np.arange(self.n) if outer is None else np.flatnonzero(self.fold!=outer)
        values={name:np.full((self.n,15) if name=='retrieval' else self.n,np.nan)
                for name in ['retrieval','ridge','robust','prediction','kernel']}
        for inner in sorted(set(self.fold[fit])):
            excluded=[int(inner)] if outer is None else sorted([int(outer),int(inner)])
            q,v=self.upstream(excluded);mask=self.fold[q]==inner;rows=q[mask]
            for key in values:values[key][rows]=v[key][mask]
        if not all(np.isfinite(z[fit]).all() for z in values.values()):raise ValueError('meta coverage differs')
        if outer is not None and not all(np.isnan(z[self.fold==outer]).all() for z in values.values()):
            raise ValueError('outer rows entered training assembly')
        return fit,{k:z[fit] for k,z in values.items()}

    def fit_one(self,recipe,fit,values,folder):
        folder.mkdir(parents=True,exist_ok=True)
        x,a=meta_inputs(values['retrieval'],values['ridge'],values['robust'],self.chem[fit],
                        values['kernel'] if recipe['kernel'] else None)
        model,arrays,info=fit_priority_residual(x,self.y[fit],a,values['robust'],values['prediction'],recipe,HGB)
        joblib.dump(model,folder/'residual.joblib')
        write_predictions(folder/'training_objective.csv',self.data.keys[fit],
            dict(label=self.y[fit],guide_cf=values['prediction'],anchor_cf=a,robust_cf=values['robust'],**arrays))
        atomic_json(folder/'fit.json',dict(recipe=recipe,weight_metadata=info,training_count=len(fit),feature_count=x.shape[1],
                                         fit_keys=self.data.keys[fit].tolist()))
        return model

    def development(self):
        self.prepare_kernel()
        for size in [1,2]:
            for excluded in itertools.combinations(range(5),size):self.upstream(excluded)
        self.emit('all_upstream_ready',cache_replay_max=max(x['max_difference'] for x in self.parity))
        self.baseline=np.full(self.n,np.nan);self.svr_oof=np.full(self.n,np.nan);self.low=np.zeros(self.n,bool)
        self.oof={r['name']:np.full(self.n,np.nan) for r in RECIPES}
        for outer in range(5):
            fit,inner=self.assemble(outer);q,held=self.upstream([outer])
            self.baseline[q]=held['prediction'];self.svr_oof[q]=held['kernel'];self.low[q]=held['low_homology']
            for recipe in RECIPES:
                folder=self.out/'folds'/recipe['name']/f'outer{outer}'
                model=self.fit_one(recipe,fit,inner,folder)
                x,a=meta_inputs(held['retrieval'],held['ridge'],held['robust'],self.chem[q],held['kernel'] if recipe['kernel'] else None)
                p=.5*held['robust']+.5*(a+model.predict(x));self.oof[recipe['name']][q]=p
                write_predictions(folder/'predictions.csv',self.data.keys[q],dict(label=self.y[q],prediction=p,reference=held['prediction'],kernel=held['kernel']))
                atomic_json(folder/'isolation.json',dict(outer=outer,inner_exclusions=[sorted([outer,int(j)]) for j in range(5) if j!=outer],
                    fit_keys=self.data.keys[fit].tolist(),query_keys=self.data.keys[q].tolist()))
            self.emit('outer_complete',outer=outer,recipes=len(RECIPES))
        # Previous R4b must be reproduced before evaluating any new selection.
        previous=read_predictions(ROOT/'experiments/dual_tail_weighting_20260919/oof/R4b.csv',self.data.keys[:self.n])
        difference=float(np.max(abs(previous-self.oof['M0'])))
        if difference>1e-7:raise ValueError(f'R4b reproduction differs {difference}')
        original=expanded_metrics(self.y,self.baseline,self.low,self.data.groups[:self.n])
        svr=expanded_metrics(self.y,self.svr_oof,self.low,self.data.groups[:self.n]);rows=[]
        for recipe in RECIPES:
            for mix in MIXES:
                name=f'{recipe["name"]}_s{mix}'
                p=(1-mix)*self.baseline+mix*self.oof[recipe['name']]
                m=expanded_metrics(self.y,p,self.low,self.data.groups[:self.n])
                rows.append(dict(name=name,recipe=recipe,mix=mix,metrics=m,selection=rank_row(m,original,svr)))
                write_predictions(self.out/'oof'/f'{name}.csv',self.data.keys[:self.n],dict(label=self.y,fold=self.fold,
                    group=self.data.groups[:self.n],low_homology=self.low,reference=self.baseline,kernel=self.svr_oof,prediction=p))
        rows.sort(key=score_key)
        selected=[];seen=set()
        for row in rows:
            if row['recipe']['name'] not in seen:
                selected.append(row);seen.add(row['recipe']['name'])
            if len(selected)==3:break
        freeze_json(self.out/'validation_plan.json',dict(candidates=[r['name'] for r in selected],
                    rule=self.protocol['selection'],test_used=False))
        atomic_json(self.out/'development.json',dict(original=original,matched_fold_svr=svr,ranking=rows,previous_R4b_max_difference=difference))
        atomic_json(self.out/'cache_audit.json',dict(replay=self.parity,files=self.sources))
        self.ranking=rows;self.selected_development=selected
        self.emit('development_complete',shortlist=[dict(name=r['name'],selection=r['selection'],all=r['metrics']['all']) for r in selected])

    def full_training(self):
        fit,values=self.assemble()
        seq=Ridge(alpha=.2,solver='cholesky').fit(self.data.embeddings[fit],self.y)
        for row in self.selected_development:
            name=row['name'];recipe=row['recipe'];folder=self.out/'full'/name
            model=self.fit_one(recipe,fit,values,folder)
            bundle=folder/'bundle';bundle.mkdir()
            shutil.copytree(OLD_BUNDLE,bundle/'original_dual')
            joblib.dump(seq,bundle/'sequence.joblib');joblib.dump(model,bundle/'residual.joblib')
            if recipe['kernel']:shutil.copy2(KERNEL_SOURCE/'model.joblib',bundle/'kernel.joblib')
            config=dict(recipe=recipe,mix=row['mix'],seed=42,training_count=self.n,
                        architecture='Frozen original DualFusion plus loss-trained residual and optional sequence kernel expert',
                        prediction_uses_labels=False,selected_on='training OOF, pending validation',
                        file_hashes={str(p.relative_to(bundle)):sha256_file(p) for p in bundle.rglob('*') if p.is_file()})
            freeze_json(bundle/'model.json',config)
            self.emit('full_fit_complete',candidate=name,training_count=self.n)

    def eval_split(self,split,names):
        test=split=='test'
        d=DevelopmentData.load(ROOT/'configs/delta_ref_phopt.yaml',test=True) if test else self.data
        ix=np.arange(len(d.keys)) if test else d.validation
        self.sources.update(d.provenance['files'])
        official=read_fasta(ROOT/f'data/phopt_{"testing" if test else "validation"}.fasta',split)
        sig=lambda rs:sorted((r.protein_id,r.sequence,r.ph_opt) for r in rs)
        if sig(official)!=sig([d.records[i] for i in ix]):raise ValueError('official split differs')
        raw,features,retrieval,low=self.prediction_inputs(d,ix,split)
        kx=self.kernel_features[[self.kernel_index[k] for k in d.keys[ix]]]
        kp=kernel_predict(self.full_kernel,kx)
        y=d.labels[ix];seqs=[d.records[i].sequence for i in ix]
        old=DualFusion(OLD_BUNDLE).predict(raw,*features,retrieval,seqs)
        original=expanded_metrics(y,old['prediction'],low);svr=expanded_metrics(y,kp,low)
        if not test:
            historic=json.loads((KERNEL_SOURCE/'release.json').read_text())['selected']['validation']
            if abs(historic['all']['rmse']-svr['all']['rmse'])>1e-10:raise ValueError('full kernel validation replay differs')
        rows=[]
        for name in names:
            folder=self.out/'full'/name/'bundle'
            pred=TailPriorityFusion(folder).predict(raw,*features,retrieval,seqs,kernel_features=kx)
            m=expanded_metrics(y,pred['prediction'],low)
            row=dict(name=name,metrics=m,selection=rank_row(m,original,svr));rows.append(row)
            write_predictions(self.out/split/f'{name}.csv',d.keys[ix],dict(label=y,low_homology=low,
                raw_baseline=raw,reference=old['prediction'],kernel_reference=kp,**pred))
        result=dict(split=split,count=len(y),original=original,local_svr=svr,candidates=rows)
        atomic_json(self.out/f'{split}.json',result)
        return result

    def release_and_test(self):
        validation=self.eval_split('validation',[r['name'] for r in self.selected_development])
        ranked=sorted(validation['candidates'],key=score_key);winner=ranked[0]['name']
        bundle=self.out/'full'/winner/'bundle'
        freeze_json(self.out/'release.json',dict(candidate=winner,model_sha256=sha256_file(bundle/'model.json'),
            validation_ranking=ranked,selection=self.protocol['selection'],test_scored=False,
            protocol_sha256=sha256_file(self.out/'protocol.json')))
        self.emit('winner_frozen_before_test',winner=winner,validation=ranked[0])
        if sha256_file(bundle/'model.json')!=json.loads((self.out/'release.json').read_text())['model_sha256']:
            raise ValueError('frozen model changed')
        test=self.eval_split('test',[winner]);candidate=test['candidates'][0]['metrics']
        # Official comparison is opened only after the single candidate is frozen.
        pub=ROOT/'experiments/delta_ref_phopt_20260916/analysis/ephod_official_strict_tail_20260916/predictions.csv'
        rows=list(csv.DictReader((self.out/'test'/f'{winner}.csv').open()));keys=[r['key'] for r in rows]
        official={r['key']:r for r in csv.DictReader(pub.open())}
        local={r['key']:r for r in csv.DictReader((KERNEL_SOURCE/'followup_test/predictions.csv').open())}
        y=np.array([float(r['label']) for r in rows]);p=np.array([float(r['prediction']) for r in rows]);low=np.array([r['low_homology']=='True' for r in rows])
        if set(keys)!=set(official) or set(keys)!=set(local):raise ValueError('comparison key sets differ')
        if not np.array_equal(y,[float(official[k]['label']) for k in keys]) or not np.array_equal(y,[float(local[k]['label']) for k in keys]):
            raise ValueError('comparison labels differ')
        ep=np.array([float(official[k]['Ensemble']) for k in keys]);lp=np.array([float(local[k]['prediction']) for k in keys])
        if np.max(abs(lp-np.array([float(r['kernel_reference']) for r in rows])))>1e-7:raise ValueError('local SVR test replay differs')
        epm=expanded_metrics(y,ep,low)
        checks=dict(venus_rmse=candidate['all']['rmse']<.809,venus_mae=candidate['all']['mae']<.578)
        for g in ['extreme_acid','extreme_alkaline']:
            for metric in ['rmse','mae']:checks[g+'_'+metric]=candidate[g][metric]<epm[g][metric]
        strong={g+'_'+metric:candidate[g][metric]<test['local_svr'][g][metric]
                for g in ['extreme_acid','extreme_alkaline'] for metric in ['rmse','mae']}
        paired=bootstrap(y,ep,p,low,np.arange(len(y)))
        atomic_json(self.out/'test_comparators.json',dict(official_ensemble=epm,local_svr=test['local_svr'],
                    checks=checks,strong_local_svr_checks=strong,paired_sample_bootstrap_vs_ensemble=paired,
                    caveat='Sample bootstrap, not independent-family inference; historical test reused across research.'))
        selected=next(r for r in self.ranking if r['name']==winner)
        oof=self.oof[selected['recipe']['name']]*selected['mix']+self.baseline*(1-selected['mix'])
        atomic_json(self.out/'selected_oof_group_bootstrap.json',dict(unit='homology_group',
            results=bootstrap(self.y,self.svr_oof,oof,self.low,self.data.groups[:self.n]),
            caveat='Conditional on exploratory model selection; not selection-adjusted.'))
        self.record(pub);self.record(KERNEL_SOURCE/'followup_test/predictions.csv')
        decision=dict(selected=winner,seed=42,user_target_reached=all(checks.values()),checks=checks,
            beats_stronger_local_svr=all(strong.values()),strong_local_svr_checks=strong,
            test_candidate_count=1,test_used_for_selection=False,production_replaced=False,
            train_count=self.n,validation_count=760,test_count=1971,elapsed_seconds=time.monotonic()-self.start)
        atomic_json(self.out/'decision.json',decision)
        atomic_json(self.out/'source_hashes.json',self.sources)
        self.report(validation,test,epm,decision)
        atomic_json(self.out/'artifact_hashes.json',{str(p.relative_to(self.out)):sha256_file(p) for p in sorted(self.out.rglob('*'))
                    if p.is_file() and p.name not in ['artifact_hashes.json','status.json']})
        self.emit('complete',decision=decision,test_metrics=candidate)

    def report(self,validation,test,ep,decision):
        m=test['candidates'][0]['metrics'];svr=test['local_svr'];old=test['original']
        lines=['# 极端 pH 优先实验：放宽整体代价','',
            f'单次冻结测试候选：{decision["selected"]}，seed42。用户目标达成：{decision["user_target_reached"]}；更强本地SVR双端四项均超过：{decision["beats_stronger_local_svr"]}。',
            '整体门槛采用Venus-DREAM Reptile论文均值 RMSE<0.809、MAE<0.578。本地同协议Venus尚未完成，本文不将论文点估计比较解释为受控胜出。','',
            '| 区间 | n | 原dual RMSE | 新模型 RMSE | EpHod集成 RMSE | 本地SVR RMSE | 新MAE | EpHod集成 MAE | 本地SVR MAE |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for g in ['all','extreme_acid','core','extreme_alkaline','low_homology']:
            lines.append(f'| {g} | {m[g]["count"]} | {old[g]["rmse"]:.6f} | {m[g]["rmse"]:.6f} | {ep[g]["rmse"]:.6f} | {svr[g]["rmse"]:.6f} | {m[g]["mae"]:.6f} | {ep[g]["mae"]:.6f} | {svr[g]["mae"]:.6f} |')
        lines+=['',f'整体 Pearson={m["all"]["pearson"]:.6f}，Spearman={m["all"]["spearman"]:.6f}，R²={m["all"]["r2"]:.6f}。',
            f'酸端偏差={m["extreme_acid"]["bias"]:.6f}、绝对误差P90={m["extreme_acid"]["abs_error_p90"]:.6f}；碱端偏差={m["extreme_alkaline"]["bias"]:.6f}、低估比例={m["extreme_alkaline"]["underestimated_fraction"]:.4f}。',
            f'中心误判极端比例：{old["false_extreme_rate"]:.6f} → {m["false_extreme_rate"]:.6f}。','',
            '## 训练内开发排序（前12项；完整39项见development.json）','',
            '| 配方/混合强度 | 整体RMSE | MAE | 极酸RMSE | 极碱RMSE | 最弱尾部比值 | 代理预算 |','|---|---:|---:|---:|---:|---:|---|']
        for r in self.ranking[:12]:
            z=r['metrics'];s=r['selection']
            lines.append(f'| {r["name"]} | {z["all"]["rmse"]:.6f} | {z["all"]["mae"]:.6f} | {z["extreme_acid"]["rmse"]:.6f} | {z["extreme_alkaline"]["rmse"]:.6f} | {s["worst_tail_ratio"]:.6f} | {s["proxy_overall_budget"]} |')
        lines+=['','代理预算仅用于开发：整体RMSE比原dual≤1.05、MAE比≤1.045。尾部比值取酸/碱RMSE/MAE相对同折训练SVR四个比值的最大值。不能将开发预算等同于最终达到论文门槛。','',
            '## 验证选择','',
            '| 候选 | 整体RMSE | MAE | 极酸RMSE | 极碱RMSE | 最弱尾部比值 | 代理预算 |','|---|---:|---:|---:|---:|---:|---|']
        for r in sorted(validation['candidates'],key=score_key):
            z=r['metrics'];s=r['selection']
            lines.append(f'| {r["name"]} | {z["all"]["rmse"]:.6f} | {z["all"]["mae"]:.6f} | {z["extreme_acid"]["rmse"]:.6f} | {z["extreme_alkaline"]["rmse"]:.6f} | {s["worst_tail_ratio"]:.6f} | {s["proxy_overall_budget"]} |')
        lines+=['','## 验收逐项','']
        for k,v in decision['checks'].items():lines.append(f'- {k}: {v}')
        lines+=['','本轮13个固定配方×5折，3个混合强度；15套SVR单/双排除重训；前三个不同配方在全7124条训练样本拟合、760条验证选择，冻结后仅一个候选进入1971条测试。',
            '完整上游预测和检索排除外层及当前行所属折；SVR标准化、标签权重仅用各自训练子集。酸端困难度来自交叉拟合基线，测试标签从未进入预测函数。',
            '只有seed42，固定表征/robust分支。完整EpHod官方集成含pHenv预训练，本地SVR为PHOPT-only组件。原始测试历史已被访问，当前研究属于探索性后续比较。',
            '没有按单个测试样本做手动修正，也没有在新测试结果后再调整本轮配方。模型及逐样本结果见full/、oof/、validation/、test/；来源、权重、训练目标、迭代日志和哈希均保留。',
            f'计算用时 {decision["elapsed_seconds"]:.1f} 秒，不含实现和事后复核。生产模型未替换。']
        (self.out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

    def run(self):
        self.development();self.full_training();self.release_and_test()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/dual_tail_priority_20260919')
    args=parser.parse_args();torch.set_num_threads(4)
    PriorityExperiment(args.output).run()
