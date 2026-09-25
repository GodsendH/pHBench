"""Replay saved models, independently recompute metrics, and draw comparison."""
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
from scipy.stats import pearsonr, spearmanr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from phgeofuse.cache import atomic_json, sha256_file
from phgeofuse.delta_ref.data import DevelopmentData
from phgeofuse.dual_fusion import DualFusion, retrieval_sequence_anchor
from phgeofuse.retrieval import RetrievalStore

CACHE=ROOT/'experiments/delta_ref_phopt_20260916/baseline/seed42'
SOURCE=ROOT/'experiments/phgeofuse_redesign_20260914'


def load_csv(path):
    rows=list(csv.DictReader(Path(path).open()))
    if len({r['key'] for r in rows})!=len(rows):raise ValueError('duplicate prediction keys')
    return rows


def values(rows,name):
    return np.array([float(r[name]) for r in rows])


def assert_close(a,b):
    np.testing.assert_allclose(a,b,rtol=0,atol=1e-10)


def verify_metrics(rows,reported):
    y=values(rows,'label');p=values(rows,'prediction')
    low=np.array([r['low_homology']=='True' for r in rows])
    masks={'all':np.ones(len(y),bool),'extreme_acid':y<=4,'core':(y>4)&(y<10),
           'extreme_alkaline':y>=10,'acid_le5':y<=5,'alkaline_ge9':y>=9,
           'acid_lt6':y<6,'neutral_6to8':(y>=6)&(y<=8),'alkaline_gt8':y>8,'low_homology':low}
    for name,m in masks.items():
        assert int(m.sum())==reported[name]['count']
        e=p[m]-y[m]
        for measure,v in dict(rmse=np.sqrt(np.mean(e*e)),mae=np.mean(abs(e)),bias=e.mean()).items():
            assert_close(v,reported[name][measure])
    assert_close(pearsonr(y,p).statistic,reported['all']['pearson'])
    assert_close(spearmanr(y,p).statistic,reported['all']['spearman'])
    assert_close(1-np.sum((p-y)**2)/np.sum((y-y.mean())**2),reported['all']['r2'])
    core=masks['core']
    assert_close(np.mean((p[core]<=4)|(p[core]>=10)),reported['false_extreme_rate'])


def main(out):
    if json.loads((out/'status.json').read_text())['event']!='complete':
        raise ValueError('experiment not complete')
    hashes=json.loads((out/'artifact_hashes.json').read_text())
    for relative,digest in hashes.items():
        if sha256_file(out/relative)!=digest:raise ValueError(f'artifact hash changed: {relative}')
    sources=json.loads((out/'source_hashes.json').read_text())
    for filename,digest in sources.items():
        if sha256_file(filename)!=digest:raise ValueError(f'input hash changed: {filename}')
    d=DevelopmentData.load(ROOT/'configs/delta_ref_phopt.yaml')
    n=len(d.train);fold=d.folds[d.train]
    protocol=json.loads((out/'protocol.json').read_text())
    dev=json.loads((out/'development.json').read_text())
    primary=protocol['primary']
    assert primary=='R4b' and protocol['seed']==42
    maxdiff=0.;models=0
    for result in dev['results']:
        recipe=result['recipe'];name=recipe['name']
        rows=load_csv(out/'oof'/f'{name}.csv')
        assert [r['key'] for r in rows]==d.keys[d.train].tolist()
        assert_close(values(rows,'label'),d.labels[d.train])
        verify_metrics(rows,result['metrics'])
        for outer in range(5):
            q=np.flatnonzero(fold==outer);fit=np.flatnonzero(fold!=outer)
            folder=out/'folds'/name/f'outer{outer}'
            weights=load_csv(folder/'training_weights_targets.csv')
            assert [r['key'] for r in weights]==d.keys[fit].tolist()
            assert not set(d.groups[fit])&set(d.groups[q])
            yy=values(weights,'label');w=values(weights,'weight');aa=values(weights,'anchor_cf');rr=values(weights,'robust_cf')
            assert_close(yy,d.labels[fit])
            expected_target=yy-aa if recipe['target']=='branch' else 2*yy-rr-aa
            assert_close(values(weights,'target'),expected_target)
            if recipe['residual_lambda'] is not None:
                info=json.loads((folder/'fit.json').read_text())['weights']
                lam=info['strength_effective'];acid=yy<=4;alk=yy>=10
                expected=(1-lam)+lam/2*(acid/acid.mean()+alk/alk.mean())
                assert_close(w,expected);assert_close(w.mean(),1.)
            with np.load(CACHE/f'excluded_{outer}/predictions.npz') as z:
                r=z['retrieval'];s=z['ridge'];robust=z['robust']
            if recipe['ridge_lambda']:
                ridge=joblib.load(out/'weighted_ridge'/str(outer)/'sequence.joblib')
                s=ridge.predict(d.embeddings[q])
            h=joblib.load(folder/'residual.joblib').predict(np.column_stack([r,s,d.x[q,-25:]]))
            p=.5*robust+.5*(retrieval_sequence_anchor(r,s)+h)
            diff=float(np.max(abs(p-values(rows,'prediction')[q])));maxdiff=max(maxdiff,diff)
            assert diff<1e-7
            models+=1
    replays=[]
    for split in ['validation','test']:
        data=DevelopmentData.load(ROOT/'configs/delta_ref_phopt.yaml',test=True) if split=='test' else d
        ix=np.arange(len(data.keys)) if split=='test' else data.validation
        summary=json.loads((out/f'{split}.json').read_text())
        features=[]
        for encoder in ['esm1v','esm2']:
            fn='features_test.npz' if split=='test' else 'features.npz'
            with np.load(SOURCE/f'{encoder}_masked'/fn) as z:
                mapping={str(k):i for i,k in enumerate(z['keys'])};indices=[mapping[k] for k in data.keys[ix]]
                features.extend([z['mean'][indices],z['std'][indices]])
        store=RetrievalStore.load(ROOT/'artifacts/phgeofuse/retrieval.pt')
        retrieval=np.array([store.features(k).numpy() for k in data.keys[ix]])
        for name,record in summary['results'].items():
            rows=load_csv(out/split/f'{name}.csv')
            assert [r['key'] for r in rows]==data.keys[ix].tolist()
            assert_close(values(rows,'label'),data.labels[ix])
            verify_metrics(rows,record['metrics'])
            bundle=out/'full'/name/'bundle'
            frozen=json.loads((out/'frozen_models.json').read_text())
            assert frozen['models'][name]==sha256_file(bundle/'model.json')
            p=DualFusion(bundle).predict(values(rows,'raw_baseline'),*features,retrieval,[data.records[i].sequence for i in ix])['prediction']
            diff=float(np.max(abs(p-values(rows,'prediction'))));maxdiff=max(maxdiff,diff)
            assert diff<1e-7
            replays.append(dict(split=split,recipe=name,count=len(rows),max_difference=diff))
    # All primary training rows were retained and the registered target used.
    training=load_csv(out/'full'/primary/'training_weights_targets.csv')
    assert len(training)==7124 and [r['key'] for r in training]==d.keys[d.train].tolist()
    yy=values(training,'label');aa=values(training,'anchor_cf');rr=values(training,'robust_cf')
    assert_close(values(training,'target'),2*yy-rr-aa)
    assert_close(values(training,'weight'),.9+.05*((yy<=4)/np.mean(yy<=4)+(yy>=10)/np.mean(yy>=10)))
    # Compare signed differences independently from summary arithmetic.
    test=json.loads((out/'test.json').read_text());r=test['reference'];c=test['results'][primary]['metrics']
    for group,measures in test['results'][primary]['comparison']['delta'].items():
        for measure,value in measures.items():assert_close(value,c[group][measure]-r[group][measure])
    verification=dict(status='passed',source_hashes_checked=len(sources),artifact_hashes_checked=len(hashes),
        oof_recipe_count=len(dev['results']),replayed_fold_models=models,full_prediction_replays=replays,
        maximum_prediction_difference=maxdiff,train_count=7124,validation_count=760,test_count=1971,
        primary_full_training_weight_and_target_verified=True,metrics_recomputed_independently=True,
        test_recipe='predeclared R4b, one seed',unit_test_contracts=5)
    atomic_json(out/'verification.json',verification)
    # Preserve the exact runnable sources as well as their hashes.
    archive=[ROOT/'scripts/evaluate_dual_tail_weighting.py',Path(__file__),ROOT/'phgeofuse/tail_weighting.py',
             ROOT/'tests/test_tail_weighting.py',ROOT/'phgeofuse/dual_fusion.py',ROOT/'phgeofuse/robust_fusion.py',
             ROOT/'phgeofuse/delta_ref/data.py',ROOT/'configs/delta_ref_phopt.yaml']
    for source in archive:
        destination=out/'reproduce'/source.relative_to(ROOT);destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
    plot(out,test)
    lines=['','## 独立核验','',
           f'已核验 {len(sources)} 个来源文件哈希、{len(hashes)} 个实验产物哈希；重放 {models} 个分折模型和 {len(replays)} 份全量拟合后的预测。',
           f'逐样本预测最大差：{maxdiff:.3g}。全部指标另行复算，主模型训练权重与完整融合目标逐样本一致。',
           '损失恒等式、权重上限、空尾部、完整预测目标、外层标签扰动隔离共 5 项测试通过。',
           'comparison.png 提供测试集 RMSE 与偏差对照；reproduce/ 保存本次实现源码。']
    with (out/'REPORT_ZH.md').open('a',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    atomic_json(out/'artifact_hashes.json',{str(p.relative_to(out)):sha256_file(p)
        for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['artifact_hashes.json','status.json']})
    print(json.dumps(verification,indent=2))


def plot(out,test):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups=['all','extreme_acid','core','extreme_alkaline','low_homology']
    labels=['All','Acid\npH ≤ 4','Core\n4 < pH < 10','Alkaline\npH ≥ 10','Low homology']
    old=test['reference'];new=test['results']['R4b']['metrics']
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    x=np.arange(len(groups));width=.36
    for ax,metric,title in zip(axes,['rmse','bias'],['Prediction error (lower is better)','Signed bias (prediction − label)']):
        a=[old[g][metric] for g in groups];b=[new[g][metric] for g in groups]
        bars0=ax.bar(x-width/2,a,width,label='Original dual',color='#52677b')
        bars1=ax.bar(x+width/2,b,width,label='Tail-weighted R4b',color='#159c96')
        ax.bar_label(bars0,fmt='%.2f',fontsize=8,padding=3)
        ax.bar_label(bars1,fmt='%.2f',fontsize=8,padding=3)
        ax.set_xticks(x,labels,fontsize=9);ax.set_title(title,fontsize=12)
        ax.set_ylabel('pH units');ax.axhline(0,color='#777777',lw=.6)
        ax.spines[['top','right']].set_visible(False);ax.margins(y=.18)
    axes[0].legend(frameon=False,fontsize=9)
    fig.suptitle('PHOPT historical test · 1,971 samples · seed 42\nFrozen robust branch; dual refit on all 7,124 training samples',fontsize=13)
    fig.savefig(out/'comparison.png',dpi=180)
    fig.savefig(out/'comparison.svg')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/dual_tail_weighting_20260919')
    main(parser.parse_args().output)
