"""Bounded train-only development search for complete DualFusion.

Selection on grouped OOF scores is exploratory, not nested HPO evaluation.
Cached robust experts are fold-scoped; all new dual residual inputs are cross-fit.
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phgeofuse.cache import atomic_json, sha256_file
from phgeofuse.delta_ref.data import (DevelopmentData, atomic_npz, freeze_json,
    read_predictions, stable_hash, write_predictions)
from phgeofuse.dual_fusion import retrieval_sequence_anchor as anchor
from phgeofuse.retrieval import RetrievalStore
from phgeofuse.robust_train import frequency_weights

SOURCE = ROOT / 'experiments/phgeofuse_redesign_20260914'
CACHE = ROOT / 'experiments/delta_ref_phopt_20260916/baseline/seed42'
DEFAULT = dict(max_leaf_nodes=7, max_iter=50, min_samples_leaf=80,
               l2_regularization=30, ph_power=.25)
SEEDS = [0, 1, 2, 3, 42]


def metrics(y, p, low):
    assert len(y) == len(p) and np.isfinite(p).all()
    e = p-y
    out = dict(count=len(y), rmse=float(np.sqrt(np.mean(e**2))),
               mae=float(np.mean(abs(e))), pearson=float(pearsonr(y,p).statistic),
               spearman=float(spearmanr(y,p).statistic), bias=float(e.mean()))
    for name, mask in [('acidic', y<6), ('neutral', (y>=6)&(y<8)),
                       ('alkaline',y>=8), ('extreme_acidic', y<=4),
                       ('extreme_alkaline',y>=10), ('low_homology',low)]:
        if mask.any():
            out[name] = dict(count=int(mask.sum()), rmse=float(np.sqrt(np.mean(e[mask]**2))),
                             mae=float(np.mean(abs(e[mask]))), bias=float(e[mask].mean()))
    return out


def fit_residual(recipe, x, y, labels):
    params = {k:v for k,v in recipe.items() if k != 'ph_power'}
    return HistGradientBoostingRegressor(**params, learning_rate=.05,
        early_stopping=False, random_state=42).fit(x,y,
            sample_weight=frequency_weights(labels,recipe['ph_power']))


class Search:
    def __init__(self, out):
        self.out=out
        out.mkdir(parents=True, exist_ok=True)
        self.started=time.monotonic()
        self.data=DevelopmentData.load(ROOT/'configs/delta_ref_phopt.yaml')
        d=self.data
        self.n=len(d.train)
        assert np.array_equal(d.train,np.arange(self.n))
        self.y=d.labels[d.train]
        self.fold=d.folds[d.train]
        self.chem=d.x[:,-25:]
        self.base={}
        self.features={}
        self.results={}
        self.provenance={}
        self.robust=np.full(self.n,np.nan)
        self.baseline=np.full(self.n,np.nan)
        self.retrieval=np.full((self.n,15),np.nan)
        for outer in range(5):
            fit,q=d.partition([outer])
            folder=CACHE/f'excluded_{outer}'
            cert=json.loads((folder/'fit.json').read_text())
            self.check_certificate(cert,fit,q,[outer])
            protocol=cert['baseline_protocol']
            assert protocol['refit_all_supervised_weights'] and protocol['seed']==42
            assert protocol['neural_fixed_epochs']==5 and protocol['homology_gate_scale']==0
            checkpoint=json.loads((folder/'graph/training.complete.json').read_text())
            assert checkpoint['fit_keys']==d.keys[fit].tolist()
            assert sha256_file(Path(checkpoint['checkpoint_path']))==checkpoint['checkpoint_sha256']
            with np.load(folder/'predictions.npz',allow_pickle=False) as z:
                assert np.array_equal(z['keys'],d.keys[q])
                self.base[outer]={k:z[k] for k in z.files}
            b=self.base[outer]
            self.robust[q]=b['robust'];self.baseline[q]=b['prediction']
            self.retrieval[q]=b['retrieval']
            assert np.allclose(b['prediction'],.5*b['robust']+.5*b['dual'],atol=1e-12,rtol=0)
            self.provenance[str(folder/'predictions.npz')]=sha256_file(folder/'predictions.npz')
            self.provenance[str(folder/'fit.json')]=sha256_file(folder/'fit.json')
        self.low=~((self.retrieval[:,4]>=.2)&(self.retrieval[:,9]>=.8)&(self.retrieval[:,10]>=.8))
        rng=np.random.default_rng(20260918)
        space=[dict(zip(DEFAULT, v)) for v in itertools.product([3,5,7],[25,50,100],
                    [40,80,160],[10,30,100],[0.,.25,.5])]
        space=[p for p in space if p!=DEFAULT]
        self.recipes=[DEFAULT]+[space[i] for i in rng.choice(len(space),23,replace=False)]
        protocol=dict(data=d.provenance, baseline_protocol=protocol, recipes=self.recipes,
            weights=np.linspace(0,1,11).tolist(), gammas=[0,.25,.5,.75,1],
            alphas=[.05,.1,.2,.5,1.], screening_folds=[0,1], survivors=8,
            ridge_recipe_survivors=2, seeds=SEEDS, seed_scope='shared deterministic experts; baseline seeds vary',
            selection='Exploratory grouped OOF search, not unbiased nested HPO evaluation',
            gates=dict(rmse_improvement=.003,min_improved_folds=3,mae_tolerance=0.,low_rmse_tolerance=.005),
            validation='At most three train-selected candidates; same gates except fold count',
            test='Only after a candidate passes train and validation gates and is frozen; historical follow-up',
            script_sha256=sha256_file(Path(__file__)))
        freeze_json(out/'protocol.json',protocol)
        self.emit('audit_baseline_complete',baseline=metrics(self.y,self.baseline,self.low))

    def emit(self,event,**kwargs):
        row=dict(event=event,seconds=time.monotonic()-self.started,pid=os.getpid(),**kwargs)
        atomic_json(self.out/'status.json',row)
        print(json.dumps(row),flush=True)

    def check_certificate(self,cert,fit,q,excluded):
        expected=self.data.certificate(fit,q,excluded)
        for k,v in expected.items():
            assert cert[k]==v, f'certificate mismatch: {k}, excluded={excluded}'

    def sequence(self,alpha,excluded):
        d=self.data;excluded=tuple(sorted(excluded));key=(alpha,excluded)
        if key in self.features:return self.features[key]
        fit,q=d.partition(excluded)
        tag='_'.join(map(str,excluded))
        if len(excluded)==1:
            b=self.base[excluded[0]]
            r=b['retrieval'];s=b['ridge']
        else:
            file=CACHE/'base_features'/f'excluded_{tag}.npz'
            self.check_certificate(json.loads(file.with_suffix('.json').read_text()),fit,q,excluded)
            rc=json.loads(file.with_suffix('.retrieval.json').read_text())
            assert rc['fit_keys']==d.keys[fit].tolist() and rc['query_keys']==d.keys[q].tolist()
            assert np.allclose(rc['fit_labels'],d.labels[fit],rtol=0,atol=1e-12)
            with np.load(file,allow_pickle=False) as z:
                assert np.array_equal(z['keys'],d.keys[q])
                r=z['retrieval'];s=z['sequence']
            self.provenance[str(file)]=sha256_file(file)
        if alpha!=.2:
            file=self.out/'ridge_cache'/f'a{alpha}_excluded_{tag}.npz'
            if file.exists():
                with np.load(file,allow_pickle=False) as z:
                    assert np.array_equal(z['keys'],d.keys[q]);s=z['sequence']
            else:
                start=time.monotonic()
                model=Ridge(alpha=alpha,solver='cholesky').fit(d.embeddings[fit],d.labels[fit])
                s=model.predict(d.embeddings[q])
                atomic_npz(file,keys=d.keys[q],sequence=s)
                self.emit('ridge_fit',alpha=alpha,excluded=excluded,fit_seconds=time.monotonic()-start)
        assert np.isfinite(r).all() and np.isfinite(s).all()
        self.features[key]=(q,r,s)
        return q,r,s

    def inputs(self,alpha,outer):
        fit,q=self.data.partition([outer])
        _,r,s=self.sequence(alpha,[outer])
        ir=np.full((self.n,15),np.nan);ss=np.full(self.n,np.nan)
        for inner in range(5):
            if inner==outer:continue
            qi,ri,si=self.sequence(alpha,[outer,inner])
            keep=self.fold[qi]==inner
            ir[qi[keep]]=ri[keep];ss[qi[keep]]=si[keep]
        assert np.isnan(ss[q]).all() and np.isfinite(ss[fit]).all()
        return (fit,q,np.column_stack([ir[fit],ss[fit],self.chem[fit]]),
                self.y[fit]-anchor(ir[fit],ss[fit]),
                np.column_stack([r,s,self.chem[q]]),anchor(r,s))

    def fit_fold(self,recipe_id,alpha,outer,inputs):
        file=self.out/'fold_predictions'/f'r{recipe_id}_a{alpha}_f{outer}.npz'
        fit,q,x,target,xq,a=inputs
        if file.exists():
            with np.load(file,allow_pickle=False) as z:
                assert np.array_equal(z['keys'],self.data.keys[q]);return z['anchor'],z['residual']
        model=fit_residual(self.recipes[recipe_id],x,target,self.y[fit])
        residual=model.predict(xq)
        atomic_npz(file,keys=self.data.keys[q],anchor=a,residual=residual)
        joblib.dump(model,file.with_suffix('.joblib'))
        if recipe_id==0 and alpha==.2:
            diff=float(np.max(abs(a+residual-self.base[outer]['dual'])))
            assert diff<1e-7, f'baseline parity failure {diff}'
        return a,residual

    def evaluate(self,recipe_id,alpha,folds):
        q=np.flatnonzero(np.isin(self.fold,folds))
        a=np.full(self.n,np.nan);r=a.copy()
        for outer in folds:
            file=self.out/'fold_predictions'/f'r{recipe_id}_a{alpha}_f{outer}.npz'
            ix=np.flatnonzero(self.fold==outer)
            with np.load(file,allow_pickle=False) as z:
                a[ix]=z['anchor'];r[ix]=z['residual']
        rows=[]
        for w,g in itertools.product(np.linspace(0,1,11),[0,.25,.5,.75,1]):
            p=(1-w)*self.robust[q]+w*(a[q]+g*r[q])
            rows.append(dict(recipe_id=recipe_id,alpha=alpha,w=float(w),gamma=g,
                             rmse=float(np.sqrt(np.mean((p-self.y[q])**2)))))
        best=min(rows,key=lambda v:v['rmse'])
        p=(1-best['w'])*self.robust[q]+best['w']*(a[q]+best['gamma']*r[q])
        best['metrics']=metrics(self.y[q],p,self.low[q])
        best['folds']=list(folds)
        atomic_json(self.out/'scores'/f'r{recipe_id}_a{alpha}_{len(folds)}fold.json',dict(grid=rows,best=best))
        if len(folds)==5:
            bm=metrics(self.y,self.baseline,self.low)
            improvements=[float(np.sqrt(np.mean((self.baseline[self.fold==f]-self.y[self.fold==f])**2))-
                np.sqrt(np.mean((p[self.fold==f]-self.y[self.fold==f])**2))) for f in range(5)]
            best['fold_rmse_improvements']=improvements
            best['passes_train_gate']=(bm['rmse']-best['rmse']>=.003 and
                sum(v>0 for v in improvements)>=3 and best['metrics']['mae']<=bm['mae'] and
                best['metrics']['low_homology']['rmse']<=bm['low_homology']['rmse']+.005)
            self.results[(recipe_id,alpha)]=(best,p)
        return best

    def run(self):
        # A: reproduce all baseline folds, scan output-only combinations.
        for f in range(5):
            start=time.monotonic()
            self.fit_fold(0,.2,f,self.inputs(.2,f))
            self.emit('baseline_fold_verified',fold=f,fit_seconds=time.monotonic()-start)
        a=self.evaluate(0,.2,range(5))
        atomic_json(self.out/'stage_a.json',a)
        atomic_json(self.out/'cache_audit.json',dict(files=self.provenance,baseline_parity_tolerance=1e-7))
        self.emit('stage_a_complete',best=a)
        # B: bounded random search, same two folds for all configurations.
        for f in [0,1]:
            inputs=self.inputs(.2,f)
            joblib.Parallel(n_jobs=2,prefer='threads')(
                joblib.delayed(self.fit_fold)(r,.2,f,inputs) for r in range(1,24))
            self.emit('screening_fold_complete',fold=f)
        screening=sorted([self.evaluate(r,.2,[0,1]) for r in range(24)],key=lambda z:z['rmse'])
        keep=[r['recipe_id'] for r in screening[:8]]
        atomic_json(self.out/'stage_b_screening.json',dict(ranking=screening,survivors=keep))
        for f in [2,3,4]:
            inputs=self.inputs(.2,f)
            joblib.Parallel(n_jobs=2,prefer='threads')(
                joblib.delayed(self.fit_fold)(r,.2,f,inputs) for r in keep)
            self.emit('confirmation_fold_complete',fold=f)
        ranked=sorted([self.evaluate(r,.2,range(5)) for r in set(keep+[0])],key=lambda z:z['rmse'])
        atomic_json(self.out/'stage_b.json',ranked)
        self.emit('stage_b_complete',best=ranked[0])
        # C: five alpha values for the two leading residual recipes.
        top=[v['recipe_id'] for v in ranked[:2]]
        for alpha in [.05,.1,.5,1.]:
            for f in range(5):
                inputs=self.inputs(alpha,f)
                for recipe_id in top:self.fit_fold(recipe_id,alpha,f,inputs)
            for recipe_id in top:self.evaluate(recipe_id,alpha,range(5))
            self.emit('stage_c_alpha_complete',alpha=alpha)
        ranking=sorted([v[0] for v in self.results.values()],key=lambda z:z['rmse'])
        atomic_json(self.out/'train_ranking.json',ranking)
        best,p=self.results[(ranking[0]['recipe_id'],ranking[0]['alpha'])]
        write_predictions(self.out/'best_oof.csv',self.data.keys[:self.n],
            dict(label=self.y,fold=self.fold,group=self.data.groups[:self.n],
                 baseline=self.baseline,prediction=p,low_homology=self.low))
        _,inv,counts=np.unique(self.data.groups[:self.n],return_inverse=True,return_counts=True)
        base_sse=np.bincount(inv,weights=(self.baseline-self.y)**2)
        candidate_sse=np.bincount(inv,weights=(p-self.y)**2)
        rng=np.random.default_rng(42);delta=[]
        for _ in range(2000):
            ix=rng.integers(len(counts),size=len(counts));n=counts[ix].sum()
            delta.append(np.sqrt(candidate_sse[ix].sum()/n)-np.sqrt(base_sse[ix].sum()/n))
        atomic_json(self.out/'bootstrap.json',dict(delta_rmse_95ci=np.quantile(delta,[.025,.975]).tolist(),
            caveat='Conditional family bootstrap after model selection; not selection-adjusted'))
        self.emit('search_complete',best=best)
        self.validation(ranking)

    def validation(self,ranking):
        from phgeofuse.dual_fusion import DualFusion
        d=self.data;v=d.validation
        store=RetrievalStore.load(ROOT/'artifacts/phgeofuse/retrieval.pt')
        assert store.payload['training_keys']==d.keys[d.train].tolist()
        assert np.allclose(store.payload['training_labels'].numpy(),self.y,rtol=0,atol=1e-6)
        rv=np.array([store.features(k).numpy() for k in d.keys[v]],dtype=float)
        low=~((rv[:,4]>=.2)&(rv[:,9]>=.8)&(rv[:,10]>=.8))
        old=DualFusion(SOURCE/'dual_candidate_float64')
        oldseq=old.sequence.predict(d.embeddings[v])
        olddual=anchor(rv,oldseq)+old.residual.predict(np.column_stack([rv,oldseq,self.chem[v]]))
        robust={};reference={}
        for seed in SEEDS:
            source=SOURCE/('baseline_validation.csv' if seed==42 else f'baseline_seed{seed}_validation.csv')
            raw=read_predictions(source,d.keys[v])
            # RobustFusion takes original mean/std, but its ridge inputs are already normalized here.
            rs=old.robust.sequence.predict(d.embeddings[v,2560:])
            av=rv[:,7:9];den=av.sum(1)
            ra=np.where(den>0,(rv[:,:2]*av).sum(1)/np.maximum(den,1),old.robust.config['training_mean_label'])
            rr=ra+old.robust.residual.predict(np.column_stack([rv,self.chem[v]]))
            ww=old.robust.weights
            robust[seed]=ww['baseline']*raw+ww['sequence']*rs+ww['residual']*rr
            reference[seed]=(1-old.dual_weight)*robust[seed]+old.dual_weight*olddual
            saved=read_predictions(SOURCE/'dual_candidate_float64'/f'seed{seed}_validation.csv',d.keys[v])
            assert np.max(abs(saved-reference[seed]))<1e-7
        summaries=[]
        for rank,candidate in enumerate(ranking[:3]):
            rid,alpha=candidate['recipe_id'],candidate['alpha']
            folder=self.out/f'validation_candidate_{rank}';folder.mkdir(exist_ok=True)
            seq=Ridge(alpha=alpha,solver='cholesky').fit(d.embeddings[d.train],self.y)
            sv=seq.predict(d.embeddings[v]);oof=np.empty(self.n)
            for f in range(5):
                q,_,s=self.sequence(alpha,[f]);oof[q]=s
            model=fit_residual(self.recipes[rid],np.column_stack([self.retrieval,oof,self.chem[:self.n]]),
                               self.y-anchor(self.retrieval,oof),self.y)
            dual=anchor(rv,sv)+candidate['gamma']*model.predict(np.column_stack([rv,sv,self.chem[v]]))
            joblib.dump(seq,folder/'sequence.joblib');joblib.dump(model,folder/'residual.joblib')
            perseed=[]
            for seed in SEEDS:
                p=(1-candidate['w'])*robust[seed]+candidate['w']*dual
                perseed.append(dict(seed=seed,candidate=metrics(d.labels[v],p,low),reference=metrics(d.labels[v],reference[seed],low)))
                write_predictions(folder/f'seed{seed}.csv',d.keys[v],dict(label=d.labels[v],prediction=p,baseline=reference[seed]))
            mean=lambda which,key:float(np.mean([r[which][key] for r in perseed]))
            gain=mean('reference','rmse')-mean('candidate','rmse')
            lowdelta=float(np.mean([r['candidate']['low_homology']['rmse']-r['reference']['low_homology']['rmse'] for r in perseed]))
            passed=(candidate['passes_train_gate'] and gain>=.003 and
                    mean('candidate','mae')<=mean('reference','mae') and lowdelta<=.005)
            row=dict(rank=rank,configuration=candidate,recipe=self.recipes[rid],per_seed=perseed,
                     mean_rmse=mean('candidate','rmse'),rmse_gain=gain,passes=passed)
            atomic_json(folder/'result.json',row);summaries.append(row)
        atomic_json(self.out/'validation.json',summaries)
        passing=sorted([r for r in summaries if r['passes']],key=lambda r:r['mean_rmse'])
        decision=dict(accepted=bool(passing),selected=passing[0] if passing else None,
                      test_accessed=False,reason='Train and validation gates' if passing else 'No candidate passed both predeclared gates')
        freeze_json(self.out/'decision.json',decision)
        self.emit('validation_complete',accepted=decision['accepted'])
        self.report(ranking,summaries,decision)
        if passing:
            self.emit('accepted_pending_test',selected=passing[0])
        else:self.emit('complete',accepted=False,test_accessed=False)

    def report(self,ranking,summaries,decision):
        base=metrics(self.y,self.baseline,self.low);best=ranking[0]
        lines=['# Dual 完整融合轻量调参', '',
            '分组 OOF 用于开发选参；不是无偏的嵌套调参评价。测试集未参与搜索。', '',
            f'- 当前完整基线 OOF RMSE：{base["rmse"]:.9f}',
            f'- 最佳开发候选 OOF RMSE：{best["rmse"]:.9f}',
            f'- 开发改善：{base["rmse"]-best["rmse"]:.9f}',
            f'- 最佳参数：`{json.dumps(best,ensure_ascii=False)}`', '',
            '| 候选 | 五种子验证 RMSE 均值 | 相对旧模型改善 | 通过门槛 |',
            '|---|---:|---:|---|']
        for r in summaries:lines.append(f'| {r["rank"]} | {r["mean_rmse"]:.9f} | {r["rmse_gain"]:.9f} | {r["passes"]} |')
        lines+=['',f'接受新模型：{decision["accepted"]}。',
            '五种子复用神经基线，Ridge/HGB 共享；不是五次完整独立重训。',
            '未通过时保留旧模型，所有候选与预测仍留档。']
        (self.out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/dual_lightweight_20260918')
    args=parser.parse_args()
    torch.set_num_threads(4)
    Search(args.output).run()
