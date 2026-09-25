"""Nested, label-free calibration of strict outer predictions.

Each calibrator is fitted on predictions from the four non-held-out folds and
then applied to the fifth fold.  Features used by the calibrator are only the
prediction itself and retrieval reliability variables available at inference.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler, PolynomialFeatures
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, str(Path(__file__).resolve().parent))
from develop_phgeofuse_regression import OUT, ROOT, metrics
from phgeofuse.io import read_manifest
from phgeofuse.robust_fusion import chemistry_features
from phgeofuse.cache import atomic_json


def feature_matrix(p, r, mode):
    p = np.asarray(p, float); r = np.asarray(r, float)
    q = np.column_stack([p, p*p, p*p*p, p-7., np.abs(p-7.),
                         r[:, 2], r[:, 3], r[:, 4], r[:, 5], r[:, 6],
                         r[:, 7], r[:, 8], r[:, 9], r[:, 10], r[:, 13], r[:, 14]])
    if mode == 'p_only': return q[:, :5]
    if mode == 'reliability': return q
    if mode == 'interactions':
        return np.column_stack([q, (p-7.)[:,None] * q[:,5:]])
    raise ValueError(mode)


def fit_apply(name, x, y, xt, p, pt):
    # All candidates have low effective capacity and fixed hyperparameters.
    if name.startswith('affine'):
        model = make_pipeline(StandardScaler(), Ridge(alpha=float(name.split('_')[-1])))
    elif name.startswith('poly'):
        degree = int(name.split('_')[1]); model = make_pipeline(StandardScaler(), PolynomialFeatures(degree, include_bias=False), Ridge(alpha=30.))
    elif name.startswith('spline'):
        knots = int(name.split('_')[1]); model = make_pipeline(SplineTransformer(n_knots=knots, degree=2), Ridge(alpha=30.))
    elif name.startswith('hgb'):
        model = HistGradientBoostingRegressor(max_leaf_nodes=int(name.split('_')[1]), max_iter=40,
            min_samples_leaf=100, l2_regularization=50, learning_rate=.05,
            early_stopping=False, random_state=42)
    elif name == 'isotonic':
        model = IsotonicRegression(out_of_bounds='clip').fit(p, y)
        return model.predict(pt), model
    elif name == 'huber':
        model = HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=300)
    else: raise ValueError(name)
    model.fit(x, y)
    return model.predict(xt), model


def main():
    z = np.load(OUT/'nested_homology_strict/predictions.npz', allow_pickle=False)
    y, fold, groups, r = z['y'].astype(float), z['fold'].astype(int), z['groups'], z['retrieval'].astype(float)
    low = ~((r[:,4]>=.2)&(r[:,9]>=.8)&(r[:,10]>=.8))
    source = z['unweighted50'].astype(float)
    base_names = ['unweighted50','unweighted150','phweighted50','family_phweighted50','anchor','sequence']
    out = OUT/'prediction_calibration_nested'; out.mkdir(exist_ok=True)
    candidates=[]
    for base in base_names:
      for mode in ['p_only','reliability','interactions']:
       for cal in ['affine_1','affine_10','affine_100','poly_2','spline_4','spline_6','hgb_3','hgb_5','isotonic','huber']:
        candidates.append((base,mode,cal))
    results={f'{b}_{m}_{c}':np.full(len(y),np.nan) for b,m,c in candidates}
    fold_rows=[]
    for outer in range(5):
      tr=np.flatnonzero(fold!=outer); te=np.flatnonzero(fold==outer)
      for b,mode,cal in candidates:
       p=z[b].astype(float)
       x=feature_matrix(p[tr],r[tr],mode); xt=feature_matrix(p[te],r[te],mode)
       q, model=fit_apply(cal,x,y[tr],xt,p[tr],p[te])
       results[f'{b}_{mode}_{cal}'][te]=q
      print('CALIBRATION_OUTER_COMPLETE',outer,flush=True)
    summary={}
    for name,p in results.items():
      m=metrics(y,p,low)
      # Penalize both tail bias and low-homology error modestly; this is a
      # reporting score, while RMSE remains the primary metric.
      composite=m['rmse']+.04*abs(m['acidic']['bias'])+.04*abs(m['alkaline']['bias'])+.02*m['low_homology']['rmse']
      summary[name]={'metrics':m,'composite':float(composite)}
    np.savez(out/'predictions.npz',**results,y=y,fold=fold,groups=groups,retrieval=r)
    atomic_json(out/'results.json',{'protocol':{'dataset':'PHOPT train only','source':'strict nested outer predictions','test_access':False,'outer_folds':5},'results':summary,'best_by_rmse':sorted(summary.items(),key=lambda kv:kv[1]['metrics']['rmse'])[:30],'best_by_composite':sorted(summary.items(),key=lambda kv:kv[1]['composite'])[:30]})
    for key, val in sorted(summary.items(),key=lambda kv:kv[1]['metrics']['rmse'])[:12]:
      m=val['metrics']; print(key, round(m['rmse'],6), round(m['low_homology']['rmse'],6), round(m['acidic']['bias'],4), round(m['alkaline']['bias'],4), flush=True)
    print('CALIBRATION_COMPLETE',flush=True)

if __name__=='__main__': main()
