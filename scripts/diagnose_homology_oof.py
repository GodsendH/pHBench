"""Training-only, homology-grouped stress test; never reads test or validation labels."""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='8'
import sys,json,subprocess,csv,time
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
sys.path.insert(0,str(Path(__file__).resolve().parent))
from develop_phgeofuse_regression import ROOT,OUT,metrics
from phgeofuse.io import read_fasta
from phgeofuse.robust_fusion import pool_features
from phgeofuse.robust_train import frequency_weights
from phgeofuse.cache import atomic_json
work=OUT/'homology_oof';work.mkdir(exist_ok=True)
records=read_fasta(ROOT/'data/phopt_training.fasta','train');fasta=work/'train_sequences.fasta'
fasta.write_text(''.join(f'>{r.protein_id}\n{r.sequence}\n' for r in records))
cluster=work/'clusters_cluster.tsv'
if not cluster.exists():
 command=['mmseqs','easy-cluster',str(fasta),str(work/'clusters'),str(work/'tmp'),'--min-seq-id','0.3','-c','0.8','--cov-mode','0','--threads','8','--cluster-mode','1','-v','1']
 atomic_json(work/'protocol.json',{'source':'PHOPT train only','identity':.3,'coverage':.8,'cov_mode':0,'folds':5,'model_choices':{'alpha':[.1,1,10],'frequency_power':[0,.25]},'test_access':False,'command':command})
 with (work/'mmseqs.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
group={member:representative for representative,member in csv.reader(cluster.open(),delimiter='\t')};groups=np.array([group[r.protein_id] for r in records]);y=np.array([r.ph_opt for r in records])
F=np.load(OUT/'esm2_masked/features.npz');mapping={str(k):i for i,k in enumerate(F['keys'])};indices=[mapping['train::'+r.protein_id] for r in records];x=pool_features(F['mean'][indices],F['std'][indices],'mean_std');folds=list(GroupKFold(5).split(x,y,groups));assign=np.empty(len(y),int)
for f,(tr,te) in enumerate(folds):assert not set(groups[tr])&set(groups[te]);assign[te]=f
atomic_json(work/'folds.json',{'groups':len(set(groups)),'rows':[{'key':r.protein_id,'group':str(groups[i]),'fold':int(assign[i])} for i,r in enumerate(records)]})
results=[]
for alpha in [.1,1,10]:
 for power in [0,.25]:
  predictions=np.empty(len(y));train_errors=[]
  for fold,(tr,te) in enumerate(folds):
   model=Ridge(alpha=alpha,solver='cholesky');model.fit(x[tr],y[tr],sample_weight=frequency_weights(y[tr],power));predictions[te]=model.predict(x[te]);train_errors.append(float(np.sqrt(np.mean((model.predict(x[tr])-y[tr])**2))))
   atomic_json(work/'status.json',{'status':'running','alpha':alpha,'power':power,'fold':fold,'updated':time.time(),'pid':os.getpid()})
  m=metrics(y,predictions,np.ones(len(y),bool));row={'alpha':alpha,'power':power,'oof':m,'training_rmse_mean':float(np.mean(train_errors))};results.append(row);np.save(work/f'predictions_a{alpha}_p{power}.npy',predictions);print(json.dumps(row),flush=True)
 atomic_json(work/'results.json',results)
atomic_json(work/'status.json',{'status':'complete','updated':time.time(),'groups':len(set(groups))});print('HOMOLOGY_OOF_COMPLETE',flush=True)
