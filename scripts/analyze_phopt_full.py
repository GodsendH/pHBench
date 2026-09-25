import csv,json,hashlib,statistics,sys
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy.stats import spearmanr
root=Path('/home/hetianci/projects/Venus-DREAM');sys.path.insert(0,str(root))
from phgeofuse.io import read_manifest
p=root/'experiments/phgeofuse_phopt_full_20260913'
manifest=read_manifest(p/'inputs/manifest.csv')
expected={'test::'+r.protein_id:r.ph_opt for r in manifest if r.split=='test'}
report=[]
for seed in [0,1,2,3,42]:
 d=p/f'seed{seed}'; stages={s:json.loads((d/f'{s}.done.json').read_text()) for s in ['baseline_train','v3_train','calibrate','test']}
 assert all(s['returncode']==0 for s in stages.values())
 assert '--init-checkpoint' not in stages['baseline_train']['command'] and '--resume' not in stages['baseline_train']['command']
 init=json.loads((d/'initialization.json').read_text()); assert init['seed']==seed and init['dataset']=='phopt'
 assert hashlib.sha256(Path(init['checkpoint']).read_bytes()).hexdigest()==init['sha256']
 assert stages['v3_train']['command'][-1]==init['checkpoint']
 rows=list(csv.DictReader((d/'test_predictions.csv').open()));assert len(rows)==len(expected) and {r['key'] for r in rows}==set(expected); assert all(abs(float(r['label'])-expected[r['key']])<1e-6 for r in rows)
 y=np.array([float(r['label']) for r in rows]);pred=np.array([float(r['prediction']) for r in rows]);assert np.isfinite(pred).all()
 recomputed={'rmse':float(np.sqrt(np.mean((pred-y)**2))),'mae':float(np.mean(abs(pred-y))),'r2':float(1-np.sum((pred-y)**2)/np.sum((y-y.mean())**2)),'spearman':float(spearmanr(y,pred).statistic)}
 m=json.loads((d/'test_predictions.metrics.json').read_text());assert all(abs(m[k]-v)<1e-8 for k,v in recomputed.items())
 epochs={}
 for name in ['baseline','v3']:
  files=list((d/'runs').glob('*tuned_mse_v1*' if name=='baseline' else '*homology_gate_v3*'))
  lines=[json.loads(x) for x in (files[0]/'metrics.jsonl').read_text().splitlines()];epochs[name]=len(lines)
 report.append({'seed':seed,**recomputed,'epochs':epochs,'scale':m['homology_residual_scale'],'rmse_acidic':m['rmse_acidic'],'rmse_neutral':m['rmse_neutral'],'rmse_alkaline':m['rmse_alkaline'],'low_homology_rmse':m['low_homology_rmse'],'start':stages['baseline_train']['started'],'end':stages['test']['updated']})
summary={'verified':True,'test_count':len(expected),'runs':report,'completed_local':datetime.fromtimestamp(max(r['end'] for r in report)).isoformat(),'hours':(max(r['end'] for r in report)-min(r['start'] for r in report))/3600,'group_means':{k:statistics.mean(r[k] for r in report) for k in ['rmse_acidic','rmse_neutral','rmse_alkaline','low_homology_rmse']}}
(p/'verification_analysis.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

