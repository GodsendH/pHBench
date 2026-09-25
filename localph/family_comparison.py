"""Paired family uncertainty for overall preservation and extreme-pH gains.

All models, seeds, endpoints and requested comparisons share each family
resample. Point metrics retain the original sample distribution; uncertainty
uses families as the independent resampling unit.
"""
import numpy as np

ENDPOINTS = [(g,m) for g in ('all','core','acid','alkaline') for m in ('rmse','mae')]
ENDPOINTS += [('acid','abs_bias'),('alkaline','abs_bias')]


def family_comparisons(labels, predictions, groups, comparisons, draws=10000, seed=42):
    y=np.asarray(labels,dtype=float); g=np.asarray(groups).astype(str)
    models={name:np.asarray(p,dtype=float) for name,p in predictions.items()}
    if y.ndim!=1 or g.shape!=y.shape or not len(y) or not np.isfinite(y).all() or draws<100:
        raise ValueError('invalid labels, groups or draws')
    if not models or not comparisons or len(set(comparisons))!=len(comparisons):
        raise ValueError('unique model comparisons are required')
    shape=next(iter(models.values())).shape
    if len(shape)!=2 or shape[0]<1 or shape[1]!=len(y) or any(p.shape!=shape or not np.isfinite(p).all() for p in models.values()):
        raise ValueError('predictions require matched finite [seed,sample] axes')
    if any(a not in models or b not in models for a,b in comparisons):
        raise ValueError('comparison refers to an absent model')
    unique,inverse=np.unique(g,return_inverse=True)
    nf=len(unique)
    if nf<2: raise ValueError('at least two families are required')
    masks={'all':np.ones(len(y),bool),'core':(y>4)&(y<10),'acid':y<=4,'alkaline':y>=10}
    counts={r:np.bincount(inverse,weights=mask,minlength=nf) for r,mask in masks.items()}
    sums={}
    for name,p in models.items():
        e=p-y
        for region,mask in masks.items():
            sums[(name,region)]={metric:np.array([np.bincount(inverse,weights=row*mask,minlength=nf) for row in values])
                for metric,values in (('mse',e**2),('mae',abs(e)),('bias',e))}
    def aggregate(multiplicity):
        score={}
        for (name,region),vals in sums.items():
            denominator=multiplicity@counts[region]
            denominator=np.where(denominator>0,denominator,np.nan)
            for metric,totals in vals.items():
                v=(multiplicity@totals.T)/denominator[:,None]
                if metric=='mse': v=np.sqrt(v); output='rmse'
                elif metric=='bias': v=abs(v); output='abs_bias'
                else: output=metric
                score[(name,region,output)]=v.mean(1)
        return score
    points=aggregate(np.ones((1,nf)))
    values={(a,b,r,m):[] for a,b in comparisons for r,m in ENDPOINTS}
    rng=np.random.default_rng(seed)
    for begin in range(0,draws,128):
        size=min(128,draws-begin)
        sampled=rng.integers(nf,size=(size,nf))
        multiplicity=np.array([np.bincount(row,minlength=nf) for row in sampled])
        scores=aggregate(multiplicity)
        for a,b,r,m in values:
            values[(a,b,r,m)].extend(scores[(a,r,m)]-scores[(b,r,m)])
    correction_count=len(comparisons)*len(ENDPOINTS)
    alpha=.05/correction_count
    result={'unit':'family','family_count':nf,'draws':draws,'seed':seed,
        'seed_aggregation':'mean of per-seed metrics, not metrics of averaged predictions',
        'point_estimand':'sample-distribution metric difference candidate minus baseline',
        'simultaneous_method':'Bonferroni percentile bootstrap over every requested comparison and endpoint',
        'simultaneous_one_sided_quantile':alpha/2,
        'expected_draws_below_simultaneous_quantile':draws*alpha/2,
        'monte_carlo_note':'Very small tail order statistics are unstable; interpret interval endpoints as approximate.',
        'conditional_on_saved_predictions':True,
        'training_uncertainty_note':'Resampling does not refit the encoder or heads; auxiliary pretraining, fold and optimizer uncertainty need separate replication.',
        'correction_count':correction_count,'comparisons':{},
        'limitations':'Bootstrap intervals are approximate; reuse of development data is not independent confirmation.'}
    for (a,b,r,m),samples in values.items():
        samples=np.asarray(samples); valid=samples[np.isfinite(samples)]
        difference=points[(a,r,m)][0]-points[(b,r,m)][0]
        row={'point_difference':float(difference) if np.isfinite(difference) else None,
             'sample_count':int(masks[r].sum()),'contributing_families':int((counts[r]>0).sum()),
             'valid_draws':len(valid),'empty_endpoint_draws':draws-len(valid),'ci95':None,
             'simultaneous_ci95':None,'ci_upper_at_most_zero':False,'simultaneous_improvement':False}
        if len(valid)>=.95*draws:
            row['ci95']=np.quantile(valid,[.025,.975]).tolist()
            row['simultaneous_ci95']=np.quantile(valid,[alpha/2,1-alpha/2]).tolist()
            row['ci_upper_at_most_zero']=row['simultaneous_ci95'][1]<=0
            row['simultaneous_improvement']=row['simultaneous_ci95'][1]<0
        result['comparisons'].setdefault(a,{}).setdefault(b,{}).setdefault(r,{})[m]=row
    return result
