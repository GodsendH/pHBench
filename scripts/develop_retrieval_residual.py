"""Low-capacity residual learner on retrieval reliability and sequence chemistry."""
import os
os.environ['OMP_NUM_THREADS']='4'
import sys,json
from pathlib import Path
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib
sys.path.insert(0,str(Path(__file__).resolve().parent))
from develop_phgeofuse_regression import ROOT,OUT,metrics
from phgeofuse.io import read_manifest
from phgeofuse.retrieval import RetrievalStore,record_key
D=np.load(OUT/'development_features.npz');y=D['y'];yv=D['yv'];rec=[r for r in read_manifest(ROOT/'artifacts/phgeofuse/manifest.csv') if r.split in ('train','validation')]
store=RetrievalStore.load(ROOT/'artifacts/phgeofuse/retrieval.pt');torch.set_num_threads(2)
chem=[]
for r in rec:
 seq=r.sequence;clean=''.join(a for a in seq if a in 'ACDEFGHIKLMNPQRSTVWY');a=ProteinAnalysis(clean)
 chem.append([seq.count(letter)/len(seq) for letter in 'ACDEFGHIKLMNPQRSTVWYX']+[np.log1p(len(seq)),a.isoelectric_point(),a.gravy(),a.aromaticity()])
chem=np.array(chem);normal=np.array([store.features(record_key(r)).numpy() for r in rec]);lowview=np.array([store.features(record_key(r),view='low_homology').numpy() for r in rec])
x=np.column_stack([normal,chem]);xl=np.column_stack([lowview,chem]);xt=np.concatenate([x[:7124],xl[:7124]]);yt=np.tile(y,2)
# Missing retrieval experts use the available branch, or the training label mean.
def anchor(z):
 a=z[:,7:9];v=z[:,:2];return np.where(a.sum(1)>0,(v*a).sum(1)/np.maximum(a.sum(1),1),y.mean())
b=anchor(xt);bv=anchor(x[7124:]);bn=anchor(x[:7124]);weights=np.concatenate([np.ones(len(y)),np.full(len(y),.25)])
folder=OUT/'retrieval_residual';folder.mkdir(exist_ok=True);np.savez(folder/'features.npz',x=x,xlow=xl)
results=[];best=1e9
for leaves in [3,7,15]:
 for iterations in [50,150,300]:
  model=HistGradientBoostingRegressor(max_leaf_nodes=leaves,max_iter=iterations,learning_rate=.05,min_samples_leaf=60,l2_regularization=20,early_stopping=False,random_state=42)
  model.fit(xt,yt-b,sample_weight=weights)
  pred=bv+model.predict(x[7124:]);tp=bn+model.predict(x[:7124]);m=metrics(yv,pred,D['lowv'])
  row={'leaves':leaves,'iterations':iterations,'validation':m,'train_rmse':float(np.sqrt(np.mean((tp-y)**2)))};results.append(row);print(json.dumps(row),flush=True)
  np.save(folder/f'val_l{leaves}_i{iterations}.npy',pred)
  if m['rmse']<best:best=m['rmse'];joblib.dump(model,folder/'best.joblib')
(folder/'search.json').write_text(json.dumps(results,indent=2));print('RESIDUAL_COMPLETE',flush=True)
