"""Reconstruct fixed-recipe pHenv transfer results; no fitting or selection."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from localph.phenv_data import sha_file,write_json
from phgeofuse.delta_ref.data import stable_hash
from phgeofuse.delta_ref.metrics import metrics,acceptance
from localph.family_comparison import family_comparisons


def shrinkage_diagnostics(labels, prediction, baseline):
    result={'minimum_prediction':float(prediction.min()),'maximum_prediction':float(prediction.max())}
    error=prediction-labels
    for region,truth,called in [('acid',labels<=4,prediction<=4),('alkaline',labels>=10,prediction>=10)]:
        e=error[truth]
        mse=float(np.mean(e**2)) if len(e) else None
        bias=float(e.mean()) if len(e) else None
        variance=float(e.var()) if len(e) else None
        if len(e) and not np.isclose(mse,bias*bias+variance,atol=1e-12,rtol=1e-12):
            raise ValueError('tail bias/variance decomposition failed')
        result[region]={'true_count':int(truth.sum()),'predicted_count':int(called.sum()),
            'true_positive_count':int((truth&called).sum()),
            'recall':float(called[truth].mean()) if truth.any() else None,
            'precision':float(truth[called].mean()) if called.any() else None,
            'mean_label':float(labels[truth].mean()) if truth.any() else None,
            'mean_prediction':float(prediction[truth].mean()) if truth.any() else None,
            'bias':bias,'error_variance':variance,'mse':mse,
            'squared_bias_fraction_of_mse':bias*bias/mse if mse else None}
    core=(labels>4)&(labels<10)
    correction=prediction[core]-baseline[core]
    result['center_correction']={'rms':float(np.sqrt(np.mean(correction**2))),
        'mean_absolute':float(np.mean(abs(correction))),
        'absolute_95th_percentile':float(np.quantile(abs(correction),.95)),
        'mse_difference':float(np.mean(error[core]**2)-np.mean((baseline[core]-labels[core])**2))}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    cert=json.loads((a.experiment/'results.json').read_text())
    protocol=json.loads((a.experiment/'protocol.json').read_text())
    for relative,digest in protocol['source_sha256'].items():
        if sha_file(ROOT/relative)!=digest: raise ValueError('training source changed')
    for relative,digest in protocol['baseline_hashes'].items():
        if sha_file(ROOT/relative)!=digest: raise ValueError('baseline artifact changed')
    if sha_file(a.experiment/'predictions.npz')!=cert['predictions_sha256']:
        raise ValueError('predictions changed')
    with np.load(a.experiment/'predictions.npz',allow_pickle=False) as z: values={k:z[k] for k in z.files}
    y,keys,groups,folds=values['labels'],values['keys'],values['groups'],values['folds']
    base=metrics(y,values['baseline'],groups)
    rows={}; fit_summary=[]
    for recipe in protocol['recipes']:
        name=recipe['name']; reconstructed=np.full(len(y),np.nan)
        for outer in range(5):
            directory=a.experiment/f'outer{outer}_{name}'
            fit=json.loads((directory/'complete.json').read_text())
            tr=np.flatnonzero(folds!=outer); qu=np.flatnonzero(folds==outer)
            if fit['train_keys']!=keys[tr].tolist() or fit['query_keys']!=keys[qu].tolist() or fit['train_label_sha256']!=stable_hash(y[tr].tolist()) or set(groups[tr])&set(groups[qu]):
                raise ValueError('fold isolation certificate failed')
            if sha_file(directory/'weights.pt')!=fit['weights_sha256']:
                raise ValueError('fit checkpoint changed')
            with np.load(directory/'predictions.npz',allow_pickle=False) as z:
                if not np.array_equal(z['query_keys'],keys[qu]) or not np.array_equal(z['train_keys'],keys[tr]):
                    raise ValueError('fit prediction keys differ')
                reconstructed[qu]=z['query_prediction']
                tm=metrics(y[tr],z['train_prediction'],groups[tr]); qm=metrics(y[qu],z['query_prediction'],groups[qu])
            fit_summary.append({'outer':outer,'recipe':name,'train_rmse':tm['all']['rmse'],
                'query_rmse':qm['all']['rmse'],'gap':qm['all']['rmse']-tm['all']['rmse'],
                'CPU_reload_max_error':fit['CPU_reload_max_error']})
        np.testing.assert_array_equal(reconstructed,values[name])
        m=metrics(y,reconstructed,groups)
        for region in ('all','acid','alkaline','core'):
            for metric in ('rmse','mae','bias'):
                if not np.isclose(m[region][metric],cert['candidates'][name]['metrics'][region][metric],atol=1e-12,rtol=0):
                    raise ValueError('aggregate metrics differ')
        rows[name]={'metrics':m,'acceptance':acceptance(m,base)}
    comparisons=[(name,'complete_baseline') for name in rows]
    comparisons += [(name,'phopt_only_weighted') for name in rows if name!='phopt_only_weighted']
    uncertainty=family_comparisons(y,{'complete_baseline':values['baseline'][None],
        **{name:values[name][None] for name in rows}},groups,comparisons,draws=200000,seed=42)
    overfit={}
    for name in rows:
        if name=='phopt_only_weighted': continue
        pairs=[]
        for outer in range(5):
            candidate=next(r for r in fit_summary if r['outer']==outer and r['recipe']==name)
            control=next(r for r in fit_summary if r['outer']==outer and r['recipe']=='phopt_only_weighted')
            pairs.append({'outer':outer,'training_rmse_difference':candidate['train_rmse']-control['train_rmse'],
                'query_rmse_difference':candidate['query_rmse']-control['query_rmse'],
                'gap_difference':candidate['gap']-control['gap']})
        overfit[name]={'per_fold_vs_phopt_only':pairs,
            'mean_query_rmse_difference':float(np.mean([r['query_rmse_difference'] for r in pairs])),
            'mean_gap_difference':float(np.mean([r['gap_difference'] for r in pairs])),
            'same_optimization_control':name=='env_finetune_weighted',
            'interpretation':'Descriptive overlapping training folds, not five independent seeds. A smaller gap alone is not evidence of better generalization.'}
    report={'source_verified':True,'all_25_fits_reconstructed':True,'fold_isolation_verified':True,
        'baseline':base,'candidates':rows,'fit_summary':fit_summary,'seed':42,'test_access':False,
        'default_model_replaced':False,'selection':'No candidate selected by this report',
        'uncertainty_scope':'Reused development folds, one seed; intervals are exploratory, not independent confirmation.',
        'paired_family_comparisons':uncertainty,'uncertainty_source_sha256':sha_file(ROOT/'localph/family_comparison.py'),
        'overfitting_diagnostics':overfit,
        'shrinkage_diagnostics':{name:shrinkage_diagnostics(y,p,values['baseline']) for name,p in
            {'complete_baseline':values['baseline'],**{name:values[name] for name in rows}}.items()},
        'input_predictions_sha256':cert['predictions_sha256'],'report_source_sha256':sha_file(Path(__file__))}
    write_json(a.output/'verification.json',report)
    lines=['# pHenv 迁移：固定配方家族开发结果','',
      '以下是原PHOPT训练集7124条的五外层家族预测，seed42。五折不等于五种子；不是1971条原始测试集。所有配方预先固定轮数并全部报告，没有按外层分数选模型。','',
      '| 方法 | 整体RMSE | 中心RMSE | 酸端RMSE | 碱端RMSE | 开发门槛 |',
      '|---|---:|---:|---:|---:|---|',
      f"| Complete baseline | {base['all']['rmse']:.6f} | {base['core']['rmse']:.6f} | {base['acid']['rmse']:.6f} | {base['alkaline']['rmse']:.6f} | — |"]
    for name,row in rows.items():
        m=row['metrics']
        lines.append(f"| {name} | {m['all']['rmse']:.6f} | {m['core']['rmse']:.6f} | {m['acid']['rmse']:.6f} | {m['alkaline']['rmse']:.6f} | {'通过' if row['acceptance']['passed'] else '未通过'} |")
    lines+=['','酸端≤4，碱端≥10。完整MAE、偏差、假极端率、每折训练/留出差距及配对家族区间见verification.json。整体和中心保持、两端改善均比较配对差值；同时区间覆盖本报告全部9组比较的10个端点。旧开发容差不等于中心/整体不劣的统计证明。','',
      '训练内基线残差经过双重折排除；外层查询基线排除外层家族。权重仅由各训练子集拟合。25个拟合的记录、权重哈希、CPU重载误差及拼合预测已重新核对。','',
      'PHOPT沿用3186组开发折：原始分组与观测到的MMseqs≥30% identity、双向coverage≥80%链接取并集。pHenv对PHOPT及辅助验证集的隔离另使用≥20% identity和双向coverage≥80%。这些是观测搜索条件，不是绝对无远同源的证明。','',
      '预训练初始化与PHOPT-only端到端控制使用相同配方；冻结头使用更多廉价全批次步骤，不能把其差异只归因于冻结。辅助数据为来源生物限额小样本，不能代表190万条完整数据的迁移效果。','',
      '区间条件于保存的预测，没有在每次重采样中重新训练编码器/任务头；辅助预训练、优化器与划分不确定性需要另做重复实验。', '',
      '本报告不替换默认模型，不声称领域领先。即使通过开发门槛，仍需固定方案五种子、原PHOPT与独立极端确认；未通过则不能宣称解决极端收缩或缓解过拟合。']
    lines+=['','## 输出收缩与中心扰动','',
        '| 方法 | 预测最大值 | 预测≥10条数 | 真碱端召回 | 碱端偏差²/MSE | 中心修正RMS |','|---|---:|---:|---:|---:|---:|']
    for name,d in report['shrinkage_diagnostics'].items():
        alk=d['alkaline']
        lines.append(f"| {name} | {d['maximum_prediction']:.4f} | {alk['predicted_count']} | {alk['recall']:.2%} | {alk['squared_bias_fraction_of_mse']:.2%} | {d['center_correction']['rms']:.4f} |")
    lines+=['','输出超过10、极端召回增加或偏差占比下降均不能单独证明成功；需结合上表整体/中心与两端误差，以及家族配对区间。酸端同样的分解、precision和中心扰动分位数均保存在verification.json。']
    (a.output/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'verified':True,'candidates':{k:v['acceptance'] for k,v in rows.items()}}),flush=True)


if __name__=='__main__': main()
