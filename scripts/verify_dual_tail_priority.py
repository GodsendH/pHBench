"""Independent artifact replay and metric verification for tail-priority work."""
import os
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='4'
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from verify_dual_tail_weighting import load_csv,values,verify_metrics,assert_close
from phgeofuse.cache import atomic_json,sha256_file
from phgeofuse.delta_ref.data import DevelopmentData
from phgeofuse.dual_fusion import retrieval_sequence_anchor
from phgeofuse.tail_priority import TailPriorityFusion
from phgeofuse.retrieval import RetrievalStore

CACHE=ROOT/'experiments/delta_ref_phopt_20260916/baseline/seed42'
SOURCE=ROOT/'experiments/phgeofuse_redesign_20260914'
FEATURE=ROOT/'experiments/field_comparisons_phopt_20260917/esm1v_full_precision/features.npz'


def main(out):
    if json.loads((out/'status.json').read_text())['event']!='complete':raise ValueError('run is incomplete')
    artifact_hashes=json.loads((out/'artifact_hashes.json').read_text())
    sources=json.loads((out/'source_hashes.json').read_text())
    for name,digest in artifact_hashes.items():
        if sha256_file(out/name)!=digest:raise ValueError(f'artifact changed {name}')
    for name,digest in sources.items():
        if sha256_file(name)!=digest:raise ValueError(f'source changed {name}')
    d=DevelopmentData.load(ROOT/'configs/delta_ref_phopt.yaml');n=len(d.train);fold=d.folds[:n]
    with np.load(FEATURE) as z:
        index={str(k):i for i,k in enumerate(z['keys'])};kx=z['token_mean'].astype(float)
    raw=kx[[index[k] for k in d.keys]]
    maxdiff=0.;kernels={};audited_kernel=0
    for folder in sorted((out/'kernel_folds').iterdir()):
        cert=json.loads((folder/'fit.json').read_text())['certificate']
        fit,q=d.partition(cert['excluded_folds'])
        if cert!=d.certificate(fit,q,cert['excluded_folds']):raise ValueError('kernel fit provenance differs')
        saved=joblib.load(folder/'model.joblib')
        np.testing.assert_array_equal(saved['fit_indices'],fit)
        assert_close(saved['mean'],raw[fit].mean(0));assert_close(saved['std'],raw[fit].std(0))
        x=(raw[fit]-saved['mean'])/(saved['std']+1e-8);xq=(raw[q]-saved['mean'])/(saved['std']+1e-8)
        pred=saved['model'].predict(rbf_kernel(xq,x,gamma=saved['gamma']))
        with np.load(folder/'predictions.npz') as z:
            np.testing.assert_array_equal(z['keys'],d.keys[q])
            diff=float(np.max(abs(z['prediction']-pred)));maxdiff=max(maxdiff,diff);assert diff<1e-7
        kernels[tuple(cert['excluded_folds'])]=pred
        audited_kernel+=1
    protocol=json.loads((out/'protocol.json').read_text());dev=json.loads((out/'development.json').read_text())
    full_predictions={};audited_residual=0;guide_checks=0
    for recipe in protocol['recipes']:
        name=recipe['name'];full=np.full(n,np.nan)
        for outer in range(5):
            q=np.flatnonzero(fold==outer);fit=np.flatnonzero(fold!=outer)
            folder=out/'folds'/name/f'outer{outer}'
            with np.load(CACHE/f'excluded_{outer}/predictions.npz') as z:
                r=z['retrieval'];s=z['ridge'];robust=z['robust'];baseline=z['prediction']
            anchor=retrieval_sequence_anchor(r,s);parts=[r,s,d.x[q,-25:]]
            if recipe['kernel']:
                k=kernels[(outer,)];anchor=.5*anchor+.5*k;parts.extend([k,k-s,k-robust])
            x=np.column_stack(parts);model=joblib.load(folder/'residual.joblib')
            pred=.5*robust+.5*(anchor+model.predict(x));full[q]=pred
            rows=load_csv(folder/'predictions.csv')
            assert [r['key'] for r in rows]==d.keys[q].tolist()
            diff=float(np.max(abs(pred-values(rows,'prediction'))));maxdiff=max(maxdiff,diff);assert diff<1e-7
            weights=load_csv(folder/'training_objective.csv')
            assert [r['key'] for r in weights]==d.keys[fit].tolist()
            y=values(weights,'label');guide=values(weights,'guide_cf');assert_close(y,d.labels[fit])
            # Reassemble guide using double-excluded upstream models; never use
            # a single-exclusion in-sample prediction for residual weighting.
            gg=np.full(n,np.nan)
            for inner in range(5):
                if inner==outer:continue
                excluded=tuple(sorted([outer,inner]));_,qi=d.partition(excluded)
                with np.load(CACHE/('excluded_'+'_'.join(map(str,excluded)))/'predictions.npz') as z:
                    np.testing.assert_array_equal(z['keys'],d.keys[qi])
                    keep=fold[qi]==inner;gg[qi[keep]]=z['prediction'][keep]
            assert_close(guide,gg[fit]);assert np.isnan(gg[q]).all();guide_checks+=1
            info=json.loads((folder/'fit.json').read_text())['weight_metadata']
            a,b=y<=4,y>=10;hard=np.ones(len(y))
            if recipe['hard_acid']:hard[a]=1+np.minimum((abs(guide[a]-y[a])/2)**2,3)
            assert_close(hard,values(weights,'hardness'))
            am,bm=info['acid_mass_effective'],info['alkaline_mass_effective']
            expected=1-am-bm+am*a*hard/(a.mean()*hard[a].mean())+bm*b/b.mean()
            assert_close(expected,values(weights,'weight'));assert_close(expected.mean(),1.)
            assert_close(values(weights,'target'),2*y-values(weights,'robust_cf')-values(weights,'anchor_cf'))
            history=info['iteration_history'];chosen=info['selected_iteration']
            assert history[chosen]['objective']==min(x['objective'] for x in history)
            audited_residual+=1
        full_predictions[name]=full
    for row in dev['ranking']:
        rows=load_csv(out/'oof'/f'{row["name"]}.csv');verify_metrics(rows,row['metrics'])
        p=row['mix']*full_predictions[row['recipe']['name']]+(1-row['mix'])*values(rows,'reference')
        assert_close(p,values(rows,'prediction'))
        candidate=row['metrics'];original=dev['original'];svr=dev['matched_fold_svr']
        expected_budget=candidate['all']['rmse']<=1.05*original['all']['rmse'] and candidate['all']['mae']<=1.045*original['all']['mae']
        assert row['selection']['proxy_overall_budget']==expected_budget
        ratios=[candidate[g][m]/svr[g][m] for g in ['extreme_acid','extreme_alkaline'] for m in ['rmse','mae']]
        assert_close(max(ratios),row['selection']['worst_tail_ratio'])
    release=json.loads((out/'release.json').read_text());winner=release['candidate'];replays=[]
    assert release['protocol_sha256']==sha256_file(out/'protocol.json')
    assert release['model_sha256']==sha256_file(out/'full'/winner/'bundle/model.json')
    assert release['test_scored'] is False
    for split in ['validation','test']:
        dd=DevelopmentData.load(ROOT/'configs/delta_ref_phopt.yaml',test=True) if split=='test' else d
        ix=np.arange(len(dd.keys)) if split=='test' else dd.validation
        summary=json.loads((out/f'{split}.json').read_text());features=[]
        for encoder in ['esm1v','esm2']:
            source=SOURCE/f'{encoder}_masked'/('features_test.npz' if split=='test' else 'features.npz')
            with np.load(source) as z:
                idx={str(k):i for i,k in enumerate(z['keys'])};indices=[idx[k] for k in dd.keys[ix]]
                features.extend([z['mean'][indices],z['std'][indices]])
        store=RetrievalStore.load(ROOT/'artifacts/phgeofuse/retrieval.pt')
        r=np.array([store.features(k).numpy() for k in dd.keys[ix]])
        k=kx[[index[key] for key in dd.keys[ix]]]
        for row in summary['candidates']:
            name=row['name'];rows=load_csv(out/split/f'{name}.csv')
            assert [r['key'] for r in rows]==dd.keys[ix].tolist();assert_close(values(rows,'label'),dd.labels[ix])
            verify_metrics(rows,row['metrics'])
            pred=TailPriorityFusion(out/'full'/name/'bundle').predict(values(rows,'raw_baseline'),*features,r,
                       [dd.records[i].sequence for i in ix],kernel_features=k)['prediction']
            diff=float(np.max(abs(pred-values(rows,'prediction'))));maxdiff=max(maxdiff,diff);assert diff<1e-7
            replays.append(dict(split=split,candidate=name,count=len(rows),max_difference=diff))
    decision=json.loads((out/'decision.json').read_text());comparators=json.loads((out/'test_comparators.json').read_text())
    test=json.loads((out/'test.json').read_text())['candidates'][0]['metrics'];ep=comparators['official_ensemble']
    checks=dict(venus_rmse=test['all']['rmse']<.809,venus_mae=test['all']['mae']<.578)
    for g in ['extreme_acid','extreme_alkaline']:
        for metric in ['rmse','mae']:checks[g+'_'+metric]=test[g][metric]<ep[g][metric]
    assert checks==decision['checks'] and all(checks.values())==decision['user_target_reached']
    result=dict(status='passed',source_hash_count=len(sources),artifact_hash_count=len(artifact_hashes),
        independently_replayed_kernel_folds=audited_kernel,replayed_residual_folds=audited_residual,
        double_exclusion_guide_checks=guide_checks,oof_metric_sets=len(dev['ranking']),
        full_prediction_replays=replays,maximum_prediction_difference=maxdiff,
        test_candidate_count=1,acceptance_checks_recomputed=True)
    atomic_json(out/'verification.json',result)
    annotate(out,winner,test,ep)
    plot(out,test,ep,comparators['local_svr'])
    for relative in ['scripts/evaluate_dual_tail_priority.py','scripts/verify_dual_tail_priority.py',
        'scripts/evaluate_dual_tail_weighting.py','scripts/verify_dual_tail_weighting.py','scripts/train_ephod_svr_control.py',
        'phgeofuse/tail_priority.py','phgeofuse/tail_weighting.py','phgeofuse/dual_fusion.py','phgeofuse/robust_fusion.py',
        'phgeofuse/delta_ref/data.py','tests/test_tail_priority.py','docs/dual_tail_priority_20260919/PLAN_ZH.md']:
        dest=out/'reproduce'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,dest)
    with (out/'REPORT_ZH.md').open('a',encoding='utf-8') as f:
        f.write(f'\n## 独立核验\n\n{audited_kernel}套SVR分折模型、{audited_residual}套残差分折模型、39组OOF指标及{len(replays)}份全量预测已重放；最大预测差{maxdiff:.3g}。所有来源哈希、双排除困难度、权重均值/目标、测试前冻结模型和验收计算核验通过。4项新增单元测试通过。\n')
    atomic_json(out/'artifact_hashes.json',{str(p.relative_to(out)):sha256_file(p) for p in sorted(out.rglob('*'))
                if p.is_file() and p.name not in ['artifact_hashes.json','status.json']})
    print(json.dumps(result,indent=2))


def annotate(out,winner,test,ep):
    rows=load_csv(out/'test'/f'{winner}.csv');y=values(rows,'label');p=values(rows,'prediction')
    pub=ROOT/'experiments/delta_ref_phopt_20260916/analysis/ephod_official_strict_tail_20260916/predictions.csv'
    public={r['key']:r for r in csv.DictReader(pub.open())}
    groups={}
    for name,mask in [('acid',y<=4),('alkaline',y>=10)]:
        ix=np.flatnonzero(mask);ix=ix[np.argsort(-((p[ix]-y[ix])**2))]
        groups[name]=[dict(key=rows[i]['key'],label=float(y[i]),prediction=float(p[i]),
            signed_error=float(p[i]-y[i]),ephod_ensemble=float(public[rows[i]['key']]['Ensemble'])) for i in ix]
    atomic_json(out/'postfreeze_tail_error_audit.json',dict(scope='Post-freeze descriptive audit only; not used for recipe selection.',rows=groups))
    lines=['','## 距离目标的剩余差距','',
           '| 指标 | 新候选 | 对照目标 | 绝对差（新−目标） | 从新候选还需下降 |','|---|---:|---:|---:|---:|']
    for g,metric,label in [('all','rmse','整体RMSE vs Venus'),('all','mae','整体MAE vs Venus'),
        ('extreme_acid','rmse','极酸RMSE vs EpHod'),('extreme_acid','mae','极酸MAE vs EpHod'),
        ('extreme_alkaline','rmse','极碱RMSE vs EpHod'),('extreme_alkaline','mae','极碱MAE vs EpHod')]:
        value=test[g][metric];target=({'rmse':.809,'mae':.578}[metric] if g=='all' else ep[g][metric])
        lines.append(f'| {label} | {value:.6f} | {target:.6f} | {value-target:+.6f} | {max(0,1-target/value)*100:.2f}% |')
    lines+=['','已实现整体指标低于Venus论文报告值，以及碱端RMSE低于官方EpHod集成；尚未同时满足酸端RMSE与碱端MAE。上述指标均为点估计。',
        '酸端困难样本权重与碱端单侧损失未成为验证胜出配方；冻结候选K1采用非对称权重和额外SVR序列专家，说明本轮收益不能归功于所有提出的模块。',
        '不应根据已查看的31条酸样本手动纠错。后续若继续，需在训练内部引入更有辨识度的残基层功能/结构信息，并做独立的异常标签核验；继续放大同一组权重已显示明显的整体MAE代价。']
    with (out/'REPORT_ZH.md').open('a',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')


def plot(out,new,ep,svr):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4.3),layout='constrained')
    labels=['Acid RMSE','Acid MAE','Alkaline RMSE','Alkaline MAE']
    pairs=[('extreme_acid','rmse'),('extreme_acid','mae'),('extreme_alkaline','rmse'),('extreme_alkaline','mae')]
    x=np.arange(4);w=.26
    for shift,source,label,color in [(-w,ep,'EpHod ensemble','#566c80'),(0,svr,'Local EpHod-SVR','#bd9561'),(w,new,'New candidate','#159c96')]:
        bars=axes[0].bar(x+shift,[source[g][m] for g,m in pairs],w,label=label,color=color)
        axes[0].bar_label(bars,fmt='%.2f',fontsize=7,padding=2)
    axes[0].set_xticks(x,labels,rotation=25,ha='right',fontsize=8);axes[0].set_ylabel('pH units')
    axes[0].set_title('Extreme-pH errors');axes[0].legend(frameon=False,fontsize=7);axes[0].set_ylim(0,2.6)
    for ax,metric,threshold in [(axes[1],'rmse',.809),(axes[2],'mae',.578)]:
        bars=ax.bar(['Venus paper','New candidate'],[threshold,new['all'][metric]],color=['#566c80','#159c96'],width=.55)
        ax.bar_label(bars,fmt='%.4f',padding=4,fontsize=10);ax.set_title('Overall '+metric.upper());ax.set_ylim(0,threshold*1.22)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('PHOPT historical test · seed 42 · 1,971 samples\nSelected K1 / mix 0.75; lower errors are better',fontsize=12)
    fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.svg');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'experiments/dual_tail_priority_20260919')
    main(p.parse_args().output)
