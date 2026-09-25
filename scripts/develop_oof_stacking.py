"""Build leakage-controlled OOF retrieval and train a compact stacking candidate."""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='8'
import sys,json,time
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib
sys.path.insert(0,str(Path(__file__).resolve().parent))
from develop_phgeofuse_regression import ROOT,OUT,metrics
from phgeofuse.io import read_manifest
from phgeofuse.retrieval import RetrievalStore,record_key,_build_retrieval_rows
from phgeofuse.config import load_config
from phgeofuse.cache import atomic_json,atomic_torch_save
from phgeofuse.robust_fusion import pool_features,chemistry_features
from phgeofuse.robust_train import frequency_weights
work=OUT/'oof_stacking';work.mkdir(exist_ok=True);torch.set_num_threads(4)
records=[r for r in read_manifest(ROOT/'artifacts/phgeofuse/manifest.csv') if r.split=='train']
foldrows=json.loads((OUT/'homology_oof/folds.json').read_text())['rows'];assert [r['key'] for r in foldrows]==[r.protein_id for r in records]
fold=np.array([r['fold'] for r in foldrows]);groups=np.array([r['group'] for r in foldrows]);y=np.array([r.ph_opt for r in records]);D=np.load(OUT/'development_features.npz');F=np.load(OUT/'esm2_masked/features.npz');xall=pool_features(F['mean'],F['std'],'mean_std');x=xall[:7124];xv=xall[7124:]
store=RetrievalStore.load(ROOT/'artifacts/phgeofuse/retrieval.pt');assert store.payload['training_keys']==[record_key(r) for r in records]
config=load_config(ROOT/'configs/phgeofuse_phopt_homology_gate_v3.yaml');oof_retrieval=np.empty((7124,15));oof_sequence=np.empty(7124)
for k in range(5):
 tr=np.where(fold!=k)[0];te=np.where(fold==k)[0];assert not set(groups[tr])&set(groups[te])
 atomic_json(work/'status.json',{'status':'running','phase':'retrieval','fold':k,'pid':os.getpid(),'updated':time.time()})
 output=work/f'fold{k}.pt'
 if output.exists():rows=torch.load(output,map_location='cpu')['rows']
 else:
  rows=_build_retrieval_rows([records[i] for i in te],[records[i] for i in tr],store.payload['training_vectors'][tr].float(),torch.tensor(y[tr],dtype=torch.float32),config)
  atomic_torch_save(output,{'rows':rows,'reference_keys':[record_key(records[i]) for i in tr],'heldout_keys':[record_key(records[i]) for i in te]})
 view=RetrievalStore({'rows':rows});oof_retrieval[te]=np.array([view.features(record_key(records[i])).numpy() for i in te])
 sequence=Ridge(alpha=.1,solver='cholesky').fit(x[tr],y[tr]);oof_sequence[te]=sequence.predict(x[te]);print('OOF_FOLD_COMPLETE',k,flush=True)
np.savez(work/'oof_features.npz',retrieval=oof_retrieval,sequence=oof_sequence,fold=fold)
sequence=Ridge(alpha=.1,solver='cholesky').fit(x,y);seqv=sequence.predict(xv);joblib.dump(sequence,work/'sequence.joblib')
valrecords=[r for r in read_manifest(ROOT/'artifacts/phgeofuse/manifest.csv') if r.split=='validation'];rv=np.array([store.features(record_key(r)).numpy() for r in valrecords]);chem=chemistry_features([r.sequence for r in records]);chemv=chemistry_features([r.sequence for r in valrecords])
trainmeta=np.column_stack([oof_retrieval,oof_sequence,chem]);valmeta=np.column_stack([rv,seqv,chemv]);target=D['yv']
def anchor(r,s):
 a=r[:,7:9];return ((r[:,:2]*a).sum(1)+s)/(a.sum(1)+1)
base=anchor(oof_retrieval,oof_sequence);bv=anchor(rv,seqv);results=[]
for leaves in [3,7]:
 for power in [0,.25]:
  for iterations in [50,150]:
   model=HistGradientBoostingRegressor(max_leaf_nodes=leaves,max_iter=iterations,min_samples_leaf=80,l2_regularization=30,learning_rate=.05,early_stopping=False,random_state=42)
   model.fit(trainmeta,y-base,sample_weight=frequency_weights(y,power));pred=bv+model.predict(valmeta)
   name=f'l{leaves}_p{power}_i{iterations}';joblib.dump(model,work/f'{name}.joblib');np.save(work/f'{name}.validation.npy',pred);row={'name':name,'validation':metrics(target,pred,D['lowv'])};results.append(row);print(json.dumps(row),flush=True)
atomic_json(work/'results.json',results);atomic_json(work/'status.json',{'status':'complete','pid':os.getpid(),'updated':time.time()});print('OOF_STACKING_COMPLETE',flush=True)
