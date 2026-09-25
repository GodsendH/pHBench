import os
os.environ['OMP_NUM_THREADS']='4'
import json,numpy as np,joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from develop_phgeofuse_regression import OUT,metrics
D=np.load(OUT/'development_features.npz');y=D['y'];yv=D['yv'];F=np.load(OUT/'retrieval_residual/features.npz');x=F['x'];xl=F['xlow'];xt=np.concatenate([x[:7124],xl[:7124]]);yt=np.tile(y,2)
def anchor(z):
 a=z[:,7:9];return np.where(a.sum(1)>0,(z[:,:2]*a).sum(1)/np.maximum(a.sum(1),1),y.mean())
b=anchor(xt);bv=anchor(x[7124:]);bn=anchor(x[:7124]);results=[]
folder=OUT/'retrieval_residual'
for power in [.25,.5]:
 bins=np.clip(np.floor(y).astype(int),0,14);counts=np.bincount(bins,minlength=15);w=(len(y)/np.maximum(counts[bins],1))**power;w/=w.mean();w=np.clip(w,.5,2.5);weights=np.concatenate([w,w*.25])
 for iterations in [50,150]:
  model=HistGradientBoostingRegressor(max_leaf_nodes=7,max_iter=iterations,learning_rate=.05,min_samples_leaf=60,l2_regularization=20,early_stopping=False,random_state=42)
  model.fit(xt,yt-b,sample_weight=weights)
  pred=bv+model.predict(x[7124:]);tp=bn+model.predict(x[:7124]);m=metrics(yv,pred,D['lowv'])
  row={'power':power,'iterations':iterations,'validation':m,'train_rmse':float(np.sqrt(np.mean((tp-y)**2)))};results.append(row);print(json.dumps(row),flush=True)
  np.save(folder/f'val_weight{power}_i{iterations}.npy',pred);joblib.dump(model,folder/f'model_weight{power}_i{iterations}.joblib')
(folder/'weighted_search.json').write_text(json.dumps(results,indent=2))
