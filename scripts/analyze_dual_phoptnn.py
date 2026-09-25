"""Matched-split complementarity, with blend weight selected on validation only."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import pearsonr,spearmanr

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from phgeofuse.phoptnn_adapter.graphs import atomic_json,digest


def read_rows(path):
    rows=list(csv.DictReader(Path(path).open()))
    mapping={r['key']:r for r in rows}
    if len(mapping)!=len(rows):raise ValueError(f'duplicate keys: {path}')
    return mapping


def read_rows_manifest(path):
    rows=list(csv.DictReader(Path(path).open()))
    mapping={r['split']+'::'+r['protein_id']:r for r in rows}
    if len(mapping)!=len(rows):raise ValueError('duplicate manifest keys')
    return mapping


def metrics(y,p):
    return dict(n=len(y),rmse=float(np.sqrt(np.mean((y-p)**2))),mae=float(np.mean(abs(y-p))),
                bias=float(np.mean(p-y)),pearson=float(pearsonr(y,p).statistic) if len(y)>2 and np.std(p)>0 and np.std(y)>0 else None,
                spearman=float(spearmanr(y,p).statistic) if len(y)>2 and np.std(p)>0 and np.std(y)>0 else None)


def cluster_bootstrap(y,d,p,groups,repetitions=2000):
    _,inverse=np.unique(groups,return_inverse=True);k=int(inverse.max()+1)
    n=np.bincount(inverse);a=np.bincount(inverse,weights=(d-y)**2);b=np.bincount(inverse,weights=(p-y)**2)
    rng=np.random.default_rng(42);delta=[]
    for _ in range(repetitions):
        ix=rng.integers(k,size=k);delta.append(float(np.sqrt(b[ix].sum()/n[ix].sum())-np.sqrt(a[ix].sum()/n[ix].sum())))
    return dict(unit='MMseqs sequence cluster (30% identity, 80% bidirectional coverage)',clusters=k,
                repetitions=repetitions,rmse_delta_95ci=np.quantile(delta,[.025,.975]).tolist())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment',type=Path,required=True)
    ap.add_argument('--seeds',type=int,nargs='+',default=[42,0,1])
    ap.add_argument('--dual-version',choices=['historical','corrected'],default='historical')
    args=ap.parse_args();out=args.experiment.resolve();report=out/('complementarity_'+args.dual_version);report.mkdir(exist_ok=True)
    legacy_root=ROOT/'experiments/phgeofuse_redesign_20260914'
    source_manifest=read_rows_manifest(out/'manifest.csv')
    quality=read_rows(out/'quality.csv')
    provenance_paths=[out/'manifest.csv',out/'quality.csv']
    cluster_file=out/'evaluation_clusters_cluster.tsv'
    if not cluster_file.exists():raise FileNotFoundError('family bootstrap requires evaluation clusters')
    clusters={member:representative for representative,member in csv.reader(cluster_file.open(),delimiter='\t')}
    results=[];predictions=[]
    for seed in args.seeds:
        arrays={}
        for split in ['validation','test']:
            dual=read_rows(out/'dual_refit'/f'{split}.csv')
            if args.dual_version=='historical':
                history=read_rows(legacy_root/('dual_test/seed42.csv' if split=='test' else 'dual_candidate_float64/seed42_validation.csv'))
                if set(history)!=set(dual):raise ValueError('historical baseline coverage differs')
                for k in dual:
                    if abs(float(history[k]['label'])-float(source_manifest[k]['ph_opt']))>1e-5:
                        raise ValueError('historical labels differ from frozen manifest')
                    dual[k]['prediction']=history[k]['prediction']
            gnn=read_rows(out/'phoptnn'/f'seed{seed}'/f'{split}.csv')
            provenance_paths.extend([out/'phoptnn'/f'seed{seed}'/f'{split}.csv',out/'phoptnn'/f'seed{seed}'/'best.pt'])
            provenance_paths.append(legacy_root/('dual_test/seed42.csv' if split=='test' else 'dual_candidate_float64/seed42_validation.csv')
                                    if args.dual_version=='historical' else out/'dual_refit'/f'{split}.csv')
            if not set(gnn)<=set(dual):raise ValueError('unexpected GNN keys')
            keys=list(dual);y=np.array([float(dual[k]['label']) for k in keys])
            d=np.array([float(dual[k]['prediction']) for k in keys]);available=np.array([k in gnn for k in keys])
            for k in gnn:
                if abs(float(gnn[k]['label'])-float(dual[k]['label']))>1e-5 or gnn[k]['sequence_sha256']!=dual[k]['sequence_sha256']:
                    raise ValueError('expert label or sequence mismatch')
            g=np.array([float(gnn[k]['prediction']) if k in gnn else d[i] for i,k in enumerate(keys)])
            if not all(np.isfinite(v).all() for v in [y,d,g]):raise ValueError('nonfinite predictions')
            low=np.array([dual[k]['low_homology'].lower()=='true' for k in keys])
            arrays[split]=(keys,y,d,g,available,low)
        _,yv,dv,gv,av,_=arrays['validation']
        grid=np.linspace(0,1,21);losses=np.array([np.mean(((1-a)*dv+a*gv-yv)**2) for a in grid])
        alpha=float(grid[np.argmin(losses)])
        row=dict(seed=seed,validation_selected_weight=alpha,weight_selection='fixed grid 0:0.05:1 on validation only',splits={})
        for split,(keys,y,d,g,available,low) in arrays.items():
            blend=(1-alpha)*d+alpha*g
            q=np.array([float(quality[k]['mean_plddt']) for k in keys])
            source=np.array([quality[k]['source'] for k in keys])
            masks={'all':np.ones(len(y),bool),'low_homology':low,'pH_le4':y<=4,'pH_center':(y>4)&(y<10),'pH_ge10':y>=10,
                   'plddt_lt50':q<50,'plddt_50_70':(q>=50)&(q<70),'plddt_70_90':(q>=70)&(q<90),
                   'plddt_ge90':q>=90,'alphafold_db':source=='alphafold_db','esmfold':source=='esmfold'}
            groups={}
            for name,mask in masks.items():
                if not mask.any():continue
                eligible=mask&available
                groups[name]=dict(dual=metrics(y[mask],d[mask]),blend_with_fallback=metrics(y[mask],blend[mask]),
                                  graph_coverage=float(available[mask].mean()))
                if eligible.any():
                    groups[name]['phoptnn_on_available']=metrics(y[eligible],g[eligible])
                    groups[name]['dual_on_available']=metrics(y[eligible],d[eligible])
                    groups[name]['gnn_win_fraction_on_available']=float((abs(g[eligible]-y[eligible])<abs(d[eligible]-y[eligible])).mean())
            errors=(d[available]-y[available],g[available]-y[available])
            summary=dict(groups=groups,error_correlation=float(np.corrcoef(*errors)[0,1]),
                         confidence_vs_gnn_squared_error_advantage_spearman=float(spearmanr(q[available],errors[0]**2-errors[1]**2).statistic),
                         oracle_rmse_diagnostic_only=float(np.sqrt(np.minimum((d-y)**2,(g-y)**2).mean())),
                         missing_graph_keys=[k for k,a in zip(keys,available) if not a])
            if split=='test':
                if not set(keys)<=set(clusters):raise ValueError('incomplete cluster coverage')
                summary['paired_cluster_bootstrap']=cluster_bootstrap(y,d,blend,[clusters[k] for k in keys])
            legacy_root=ROOT/'experiments/phgeofuse_redesign_20260914'
            legacy_path=(legacy_root/'dual_test/seed42.csv' if split=='test'
                         else legacy_root/'dual_candidate_float64/seed42_validation.csv')
            historical=read_rows(legacy_path)
            if set(historical)!=set(keys):raise ValueError('historical baseline keys differ')
            hp=np.array([float(historical[k]['prediction']) for k in keys])
            summary['historical_dual_same_seed']=metrics(y,hp)
            summary['confidence_correction_dual_rmse_delta']=metrics(y,d)['rmse']-metrics(y,hp)['rmse']
            row['splits'][split]=summary
            if split=='test':predictions.append(blend)
            export=[dict(key=k,label=float(y[i]),dual=float(d[i]),phoptnn=float(g[i]) if available[i] else '',
                         blend=float(blend[i]),weight=alpha if available[i] else 0.,mean_plddt=float(q[i])) for i,k in enumerate(keys)]
            with (report/f'seed{seed}_{split}.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(export[0]));w.writeheader();w.writerows(export)
        results.append(row)
    rmse=np.array([r['splits']['test']['groups']['all']['blend_with_fallback']['rmse'] for r in results])
    result=dict(status='complete',dual_version=args.dual_version,seeds=args.seeds,results=results,test_rmse_mean=float(rmse.mean()),
                test_rmse_sd=float(rmse.std(ddof=1)) if len(rmse)>1 else None,
                protocol='Historical PHOPT test already viewed; follow-up comparison, not untouched confirmation. Dual seed42 fixed; GNN seeds vary. No learned gate fitted.',
                low_homology_definition='identity/coverage flags from corrected retrieval; common evaluation stratification',
                inputs={str(p):digest(p) for p in set(provenance_paths+[cluster_file])})
    atomic_json(report/'results.json',result)
    lines=['# Dual 与 pHoptNN 互补性实测','',result['protocol'],'',
           '| GNN seed | 验证选择的 GNN 权重 | Dual 测试 RMSE | GNN 测试 RMSE（可用结构） | 融合测试 RMSE |',
           '|---|---:|---:|---:|---:|']
    for r in results:
        g=r['splits']['test']['groups']['all']
        lines.append(f"| {r['seed']} | {r['validation_selected_weight']:.2f} | {g['dual']['rmse']:.5f} | {g['phoptnn_on_available']['rmse']:.5f} | {g['blend_with_fallback']['rmse']:.5f} |")
    lines+=['',f"主对照：{args.dual_version} Dual seed42。融合权重在测试前由验证集选择。",'']
    for r in results:
        summary=r['splits']['test'];groups=summary['groups'];ci=summary['paired_cluster_bootstrap']['rmse_delta_95ci']
        lines.extend([f"## GNN seed {r['seed']}",'',
          f"测试误差相关系数：{summary['error_correlation']:.4f}；逐样本 GNN 绝对误差较小的比例：{groups['all']['gnn_win_fraction_on_available']:.2%}。",
          f"融合相对 Dual 的 RMSE 差的簇级 bootstrap 95% 区间：[{ci[0]:.5f}, {ci[1]:.5f}]，负值表示融合更好。",
          f"pLDDT 与 GNN 平方误差优势的 Spearman：{summary['confidence_vs_gnn_squared_error_advantage_spearman']:.4f}；正值表示较高置信度对应更大的相对优势，不能单独作因果结论。",'',
          '| 分组 | N | Dual RMSE | pHoptNN RMSE | 融合 RMSE | GNN 胜出比例 |',
          '|---|---:|---:|---:|---:|---:|'])
        for name,g in groups.items():
            lines.append(f"| {name} | {g['dual']['n']} | {g['dual']['rmse']:.5f} | {g['phoptnn_on_available']['rmse']:.5f} | {g['blend_with_fallback']['rmse']:.5f} | {g['gnn_win_fraction_on_available']:.2%} |")
        delta=groups['all']['blend_with_fallback']['rmse']-groups['all']['dual']['rmse']
        if r['validation_selected_weight']==0:
            conclusion='验证集选择关闭 GNN 分支；当前配方未显示可用于固定融合的验证收益。'
        elif delta<0 and ci[1]<0:
            conclusion='该种子显示测试融合收益，且簇级区间低于零；仍需结合其他种子和独立数据判断稳定性。'
        elif delta<0:
            conclusion='测试点估计有改善，但区间未完全低于零，证据不足以确认稳定收益。'
        else:
            conclusion='验证选择的固定融合未改善该次测试结果；不能据此宣称已实现有效互补。'
        lines.extend(['',conclusion,''])
    lines+=['','完整分层指标、覆盖率、误差相关性和簇级 bootstrap 区间见 results.json。',
            '固定融合权重仅由验证集选择；oracle 仅用于互补潜力诊断。不能将其作为可部署成绩。']
    (report/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='complete',report=str(report),test_rmse_mean=result['test_rmse_mean'])),flush=True)


if __name__=='__main__':main()
