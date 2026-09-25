"""One-shot confirmatory evaluation of the frozen robust-fusion candidate."""
import sys,json,csv,hashlib,statistics
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from develop_phgeofuse_regression import ROOT,OUT,metrics
from phgeofuse.io import read_manifest
from phgeofuse.retrieval import RetrievalStore,record_key
from phgeofuse.robust_fusion import RobustFusion
from phgeofuse.cache import atomic_json
model=RobustFusion(OUT/'frozen_model');records=[r for r in read_manifest(ROOT/'artifacts/phgeofuse/manifest.csv') if r.split=='test'];keys=[record_key(r) for r in records];y=np.array([r.ph_opt for r in records]);store=RetrievalStore.load(ROOT/'artifacts/phgeofuse/retrieval.pt');retrieval=np.array([store.features(k).numpy() for k in keys]);low=~((retrieval[:,4]>=.2)&(retrieval[:,9]>=.8)&(retrieval[:,10]>=.8))
F=np.load(OUT/'esm2_masked/features_test.npz');assert list(F['keys'])==keys
# Verify the production inference path reproduces the selected validation experiment.
val=list(csv.DictReader((OUT/'production_validation.csv').open()));D=np.load(OUT/'development_features.npz');vm=metrics(np.array([float(r['label']) for r in val]),np.array([float(r['prediction']) for r in val]),D['lowv'])
assert abs(vm['rmse']-model.config['validation']['rmse'])<1e-9
result=[];bases=[];new=[]
for seed in [0,1,2,3,42]:
 source=ROOT/f'experiments/phgeofuse_phopt_full_20260913/seed{seed}/test_predictions.csv';rows={r['key']:r for r in csv.DictReader(source.open())};assert set(rows)==set(keys)
 baseline=np.array([float(rows[k]['prediction']) for k in keys]);pred=model.predict(baseline,F['mean'],F['std'],retrieval,[r.sequence for r in records]);assert np.isfinite(pred['prediction']).all()
 row={'seed':seed,'baseline':metrics(y,baseline,low),'candidate':metrics(y,pred['prediction'],low)};result.append(row);bases.append(baseline);new.append(pred['prediction'])
 with (OUT/f'confirmatory_seed{seed}.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['key','label',*pred]);writer.writeheader()
  for i,k in enumerate(keys):writer.writerow({'key':k,'label':float(y[i]),**{name:float(v[i]) for name,v in pred.items()}})
 print(json.dumps(row),flush=True)
summary={}
for modelname in ['baseline','candidate']:
 summary[modelname]={k:{'mean':statistics.mean(r[modelname][k] for r in result),'std':statistics.stdev(r[modelname][k] for r in result)} for k in ['rmse','mae','r2','spearman']}
 summary[modelname]['groups']={g:{k:statistics.mean(r[modelname][g][k] for r in result) for k in ['rmse','bias']} for g in ['acidic','neutral','alkaline','low_homology']}
# Paired sample bootstrap, keeping every seed's pair together; does not establish external generalization.
rng=np.random.default_rng(42);deltas=[];base=np.array(bases);candidate=np.array(new)
for _ in range(2000):
 ix=rng.integers(0,len(y),len(y));deltas.append(float(np.sqrt(np.mean((candidate[:,ix]-y[ix])**2,axis=1)).mean()-np.sqrt(np.mean((base[:,ix]-y[ix])**2,axis=1)).mean()))
summary['rmse_delta_bootstrap_95ci']=np.quantile(deltas,[.025,.975]).tolist()
summary['test_count']=len(y);summary['low_count']=int(low.sum());summary['frozen_model_sha256']=hashlib.sha256((OUT/'frozen_model/model.json').read_bytes()).hexdigest()
atomic_json(OUT/'confirmatory_results.json',result);atomic_json(OUT/'confirmatory_summary.json',summary);print('CONFIRMATORY_COMPLETE',json.dumps(summary),flush=True)
