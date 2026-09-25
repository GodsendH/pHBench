"""Recompute auxiliary generalization and actual tail loss contributions."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from localph.phenv_data import sha_file,write_json
from localph.environment_training import organism_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    data_cert=json.loads((a.data/'complete.json').read_text())
    fit=json.loads((a.experiment/'complete.json').read_text())
    protocol=json.loads((a.experiment/'protocol.json').read_text())
    if sha_file(a.data/'records.csv')!=data_cert['records_sha256'] or sha_file(a.data/'complete.json')!=protocol['data_certificate_sha256']:
        raise ValueError('data source differs from fitted model')
    if sha_file(a.experiment/'weights.pt')!=fit['weights_sha256'] or sha_file(a.experiment/'protocol.json')!=fit['protocol_sha256']:
        raise ValueError('checkpoint provenance differs')
    with (a.data/'records.csv').open(newline='') as f: rows={r['key']:r for r in csv.DictReader(f)}
    with np.load(a.experiment/'predictions.npz',allow_pickle=False) as z: saved={k:z[k] for k in z.files}
    arrays={}
    for split in ('train','validation'):
        keys=saved[split+'_keys']; pred=saved[split+'_prediction']
        expected=protocol['train_keys' if split=='train' else 'validation_keys']
        if keys.tolist()!=expected or len(set(keys))!=len(keys): raise ValueError('saved prediction keys differ')
        if any(rows[k]['split']!=split for k in keys): raise ValueError('split membership differs')
        y=np.array([float(rows[k]['phenv']) for k in keys]); org=np.array([rows[k]['organism'] for k in keys])
        actual=organism_metrics(y,pred,org)
        for region in actual:
            for metric in ('rmse','organism_macro_rmse'):
                if not np.isclose(actual[region][metric],fit[split][region][metric],atol=1e-12,rtol=0):
                    raise ValueError('recomputed metric differs')
        arrays[split]=(keys,y,pred,org)
    keys,y,pred,org=arrays['train']
    w=np.array([float(rows[k]['train_weight']) for k in keys])
    constant=float(np.average(y,weights=w))
    baseline={s:organism_metrics(ys,np.full(len(ys),constant),orgs) for s,(_,ys,_,orgs) in arrays.items()}
    loss=(pred-y)**2*w; mass={}
    for region,mask in {'acid':y<=4,'alkaline':y>=10,'core':(y>4)&(y<10)}.items():
        mass[region]={'sample_count':int(mask.sum()),'organisms':len(set(org[mask])),
            'nominal_weight_fraction':float(w[mask].sum()/w.sum()),
            'weighted_square_error_fraction':float(loss[mask].sum()/loss.sum()) if loss.sum()>0 else 0.,
            'weighted_mse_contribution':float(loss[mask].sum()/w.sum())}
    gaps={r:fit['validation'][r]['organism_macro_rmse']-fit['train'][r]['organism_macro_rmse'] for r in fit['train']}
    result={'verified':True,'task':'pHenv','training_organism_mean_baseline':constant,
      'baseline':baseline,'candidate':{'train':fit['train'],'validation':fit['validation']},
      'organism_macro_validation_minus_train':gaps,'selected_checkpoint_training_loss_mass':mass,
      'loss_mass_note':'Eval-mode squared errors at the selected checkpoint, not accumulated stochastic training loss or gradient mass.',
      'labels_used':'pHenv only','phopt_improvement_established':False,
      'source_code_sha256':sha_file(Path(__file__)),'fit_complete_sha256':sha_file(a.experiment/'complete.json'),
      'predictions_sha256':sha_file(a.experiment/'predictions.npz')}
    write_json(a.output/'verification.json',result)
    lines=['# pHenv 辅助预训练核验','',
      '所有数值属于生长pH辅助任务，不能作为酶最适pH性能改善证据。','',
      '| 区间 | 训练来源宏RMSE | 验证来源宏RMSE | 训练均值基线：验证来源宏RMSE | 验证来源数 |',
      '|---|---:|---:|---:|---:|']
    for region in ('all','core','acid','alkaline'):
        lines.append(f"| {region} | {fit['train'][region]['organism_macro_rmse']:.6f} | {fit['validation'][region]['organism_macro_rmse']:.6f} | {baseline['validation'][region]['organism_macro_rmse']:.6f} | {fit['validation'][region]['organisms']} |")
    lines += ['',f'常数基线只使用训练来源生物平均pHenv（{constant:.6f}），未使用验证标签拟合。宏RMSE为先在来源生物内求MSE、再在来源间平均并开根；不是平均来源RMSE，也不是把同源序列当独立标签。','',
      '实际训练误差贡献与名义权重质量分别保存在verification.json；没有用名义样本比例冒充梯度贡献。极碱验证只有少量来源，选模不稳定性需要在酶任务的家族留出结果中检验。']
    (a.output/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
