"""Fixed simple pHenv controls on the same organism/homology-heldout pilot.

No tuning or PHOPT labels: length-only and amino-acid-composition Ridge,
training-weighted standardization, fixed alpha=100. These are auxiliary-task
sanity controls, not claims about the best possible linear baseline.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
from sklearn.linear_model import Ridge
from localph.phenv_data import sha_file,write_json
from localph.environment_training import organism_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--environment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    dc=json.loads((a.data/'complete.json').read_text())
    verification=json.loads((a.data/'verification.json').read_text())
    ec=json.loads((a.environment/'complete.json').read_text())
    ep=json.loads((a.environment/'protocol.json').read_text())
    if (dc['records_sha256']!=sha_file(a.data/'records.csv')
        or verification['complete_sha256']!=sha_file(a.data/'complete.json')
        or not verification['verified'] or ep['data_certificate_sha256']!=sha_file(a.data/'complete.json')
        or ec['protocol_sha256']!=sha_file(a.environment/'protocol.json')):
        raise ValueError('auxiliary split/source differs')
    with (a.data/'records.csv').open(newline='') as f:
        rows={r['key']:r for r in csv.DictReader(f)}
    keys=np.array(ep['train_keys']+ep['validation_keys'])
    if set(keys)!=set(rows) or len(keys)!=len(rows):
        raise ValueError('environment fit coverage differs')
    ordered=[rows[k] for k in keys]
    y=np.array([float(r['phenv']) for r in ordered])
    org=np.array([r['organism'] for r in ordered])
    tr=np.array([r['split']=='train' for r in ordered])
    weights=np.array([float(r['train_weight']) for r in ordered if r['split']=='train'])
    if not tr[:sum(tr)].all() or tr[sum(tr):].any() or set(org[tr])&set(org[~tr]):
        raise ValueError('training/validation membership differs')
    alphabet='ACDEFGHIKLMNPQRSTVWYX'
    features=np.array([[np.log1p(len(r['sequence']))]+
        [r['sequence'].count(aa)/len(r['sequence']) for aa in alphabet] for r in ordered])
    a.output.mkdir(parents=True,exist_ok=False)
    protocol={'task':'pHenv simple diagnostic controls','recipes':['length_only','composition_length'],
        'alpha':100.,'scaling':'weighted mean and variance fitted on training rows only',
        'selection':'Fixed alpha; no validation tuning or PHOPT predictions',
        'feature_names':['log1p_length',*list(alphabet)],'training_keys':keys[tr].tolist(),
        'validation_keys':keys[~tr].tolist(),'data_complete_sha256':sha_file(a.data/'complete.json'),
        'environment_complete_sha256':sha_file(a.environment/'complete.json'),
        'code_sha256':sha_file(Path(__file__)),'phopt_labels_consumed':False}
    write_json(a.output/'protocol.json',protocol)
    reports={}
    saved={'keys':keys,'labels':y,'organisms':org}
    for name,columns in [('length_only',np.array([0])),('composition_length',np.arange(features.shape[1]))]:
        x=features[:,columns]
        mean=np.average(x[tr],axis=0,weights=weights)
        scale=np.sqrt(np.average((x[tr]-mean)**2,axis=0,weights=weights)).clip(1e-6)
        normalized=(x-mean)/scale
        model=Ridge(alpha=100.).fit(normalized[tr],y[tr],sample_weight=weights)
        pred=model.predict(normalized)
        bundle={'model':model,'mean':mean,'scale':scale,'columns':columns}
        file=a.output/(name+'.joblib')
        joblib.dump(bundle,file)
        reloaded=joblib.load(file)
        check=reloaded['model'].predict((features[:,reloaded['columns']]-reloaded['mean'])/reloaded['scale'])
        if not np.array_equal(check,pred):
            raise ValueError('simple control reload changed predictions')
        reports[name]={'train':organism_metrics(y[tr],pred[tr],org[tr]),
            'validation':organism_metrics(y[~tr],pred[~tr],org[~tr]),'model_sha256':sha_file(file)}
        saved[name]=pred
    np.savez(a.output/'predictions.npz',**saved)
    result={'controls':reports,'environment':{'train':ec['train'],'validation':ec['validation']},
        'protocol_sha256':sha_file(a.output/'protocol.json'),'predictions_sha256':sha_file(a.output/'predictions.npz'),
        'phopt_improvement_established':False,'notes':'Fixed simple controls; the pHenv validation set selected the neural checkpoint and is not independent confirmation.'}
    write_json(a.output/'complete.json',result)
    lines=['# pHenv 简单序列对照','',
        '训练和验证来源与pilot完全相同。只读取序列和pHenv，未使用PHOPT标签或拟合PHOPT预测。Ridge alpha=100在得分前固定，特征缩放只拟合训练数据。','',
        '| 模型 | 训练来源宏RMSE | 验证全部 | 验证中心 | 验证酸端 | 验证碱端 |',
        '|---|---:|---:|---:|---:|---:|']
    for name,m in {**reports,'environment_encoder':result['environment']}.items():
        values=[m['train']['all']['organism_macro_rmse']]+[m['validation'][g]['organism_macro_rmse'] for g in ('all','core','acid','alkaline')]
        lines.append('| '+name+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    lines+=['','这两个线性对照仅检查是否超越简单序列统计；没有调参，不能称为最优线性基线。',
        '来源验证参与神经检查点选择，极碱仅2个来源。它不是外部测试，也不代替PHOPT的家族配对比较。']
    (a.output/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'controls':reports,'phopt_improvement_established':False}),flush=True)


if __name__=='__main__':
    main()
