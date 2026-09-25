"""Asymmetric tail objectives and label-free complete-fusion inference."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics.pairwise import rbf_kernel
from .dual_fusion import DualFusion, retrieval_sequence_anchor
from .robust_fusion import pool_features, chemistry_features


def priority_weights(labels, guide, acid_mass=.05, alkaline_mass=.10, hard_acid=False, cap=40.):
    y,p=np.asarray(labels,float),np.asarray(guide,float)
    if y.ndim!=1 or len(y)==0 or y.shape!=p.shape or not np.isfinite([y,p]).all():
        raise ValueError('finite aligned training labels/guide required')
    if not np.isfinite([acid_mass,alkaline_mass,cap]).all() or min(acid_mass,alkaline_mass)<0 or acid_mass+alkaline_mass>=1 or cap<1:
        raise ValueError('invalid tail masses or cap')
    a,b=y<=4,y>=10
    if not a.any() or not b.any():raise ValueError('empty training tail')
    hardness=np.ones(len(y))
    if hard_acid:hardness[a]=1+np.minimum((np.abs(p[a]-y[a])/2)**2,3.)
    acid_term=a*hardness/(hardness[a].mean()*a.mean())
    alkaline_term=b/b.mean()
    extra=acid_mass*(acid_term-1)+alkaline_mass*(alkaline_term-1)
    scale=min(1.,(cap-1)/extra.max()) if extra.max()>0 else 1.
    w=1+scale*extra
    if not np.isclose(w.mean(),1,rtol=0,atol=1e-12):raise ValueError('non-unit weight mean')
    metadata=dict(acid_mass_requested=acid_mass,alkaline_mass_requested=alkaline_mass,
        acid_mass_effective=acid_mass*scale,alkaline_mass_effective=alkaline_mass*scale,
        hard_acid=hard_acid,cap=cap,minimum=float(w.min()),maximum=float(w.max()),mean=float(w.mean()),
        effective_sample_size=float(w.sum()**2/np.dot(w,w)),
        coefficient_mass=dict(acid=float(w[a].sum()/w.sum()),alkaline=float(w[b].sum()/w.sum()),core=float(w[~(a|b)].sum()/w.sum())))
    return w,hardness,metadata


def meta_inputs(retrieval,ridge,robust,chemistry,svr=None):
    r,s,q,c=[np.asarray(v,float) for v in [retrieval,ridge,robust,chemistry]]
    n=len(s)
    if r.shape!=(n,15) or q.shape!=(n,) or c.shape!=(n,25):raise ValueError('invalid meta shapes')
    a=retrieval_sequence_anchor(r,s)
    parts=[r,s,c]
    if svr is not None:
        v=np.asarray(svr,float)
        if v.shape!=(n,):raise ValueError('SVR vector shape differs')
        a=.5*a+.5*v
        parts.extend([v,v-s,v-q])
    x=np.column_stack(parts)
    if not np.isfinite(x).all():raise ValueError('nonfinite meta features')
    return x,a


def fit_priority_residual(x,y,anchor,robust,guide,recipe,hgb_params):
    y=np.asarray(y,float);a=np.asarray(anchor,float);r=np.asarray(robust,float)
    if y.shape!=a.shape or y.shape!=r.shape:raise ValueError('target alignment differs')
    base,hardness,wm=priority_weights(y,guide,recipe['acid_mass'],recipe['alkaline_mass'],recipe['hard_acid'])
    target=2*y-r-a
    multiplier=recipe['under_multiplier'];count=3 if multiplier>1 else 1
    weights=base.copy();best=None;history=[]
    for iteration in range(count):
        model=HistGradientBoostingRegressor(**hgb_params).fit(x,target,sample_weight=weights)
        p=.5*r+.5*(a+model.predict(x))
        factor=np.where((y>=10)&(p<y),multiplier,1.)
        objective=float(np.mean(base*factor*(p-y)**2))
        history.append(dict(iteration=iteration,objective=objective,mean_fit_weight=float(weights.mean()),
                            max_fit_weight=float(weights.max()),alkaline_under_fraction=float(np.mean(p[y>=10]<y[y>=10]))))
        if best is None or objective<best[0]:best=(objective,model,iteration,weights.copy())
        weights=base*factor;weights/=weights.mean()
    wm.update(iteration_history=history,selected_iteration=best[2],under_multiplier=multiplier)
    return best[1],dict(weight=base,fit_weight=best[3],hardness=hardness,target=target),wm


def kernel_predict(package,features):
    x=(np.asarray(features,float)-package['mean'])/(package['std']+1e-8)
    return package['model'].predict(rbf_kernel(x,package['train_features'],gamma=package['gamma']))


class TailPriorityFusion:
    """Inference consumes only sequence/retrieval/model outputs, never pH labels."""
    def __init__(self,bundle):
        bundle=Path(bundle)
        self.config=json.loads((bundle/'model.json').read_text())
        for name,digest in self.config['file_hashes'].items():
            if hashlib.sha256((bundle/name).read_bytes()).hexdigest()!=digest:
                raise ValueError(f'model hash mismatch: {name}')
        self.original=DualFusion(bundle/'original_dual')
        self.sequence=joblib.load(bundle/'sequence.joblib')
        self.residual=joblib.load(bundle/'residual.joblib')
        self.kernel=joblib.load(bundle/'kernel.joblib') if self.config['recipe']['kernel'] else None
        self.mix=float(self.config['mix'])
        if not 0<=self.mix<=1:raise ValueError('invalid final mixing strength')

    def predict(self,baseline,esm1v_mean,esm1v_std,esm2_mean,esm2_std,retrieval,sequences,*,kernel_features=None):
        old=self.original.predict(baseline,esm1v_mean,esm1v_std,esm2_mean,esm2_std,retrieval,sequences)
        embeddings=np.column_stack([pool_features(esm1v_mean,esm1v_std,'mean_std'),pool_features(esm2_mean,esm2_std,'mean_std')])
        s=self.sequence.predict(embeddings)
        k=kernel_predict(self.kernel,kernel_features) if self.kernel is not None else None
        x,a=meta_inputs(retrieval,s,old['robust_v1_prediction'],chemistry_features(sequences),k)
        h=self.residual.predict(x)
        candidate=.5*old['robust_v1_prediction']+.5*(a+h)
        final=(1-self.mix)*old['prediction']+self.mix*candidate
        return dict(prediction=final,original=old['prediction'],unmixed=candidate,
                    robust=old['robust_v1_prediction'],ridge=s,anchor=a,residual=h,
                    **({'kernel':k} if k is not None else {}))
