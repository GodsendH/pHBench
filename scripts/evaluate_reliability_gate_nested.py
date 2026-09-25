"""Strict nested gate using outer/inner-excluded dual sequence caches."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
import sys
import time
import json
import argparse
from pathlib import Path
import joblib
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from develop_phgeofuse_regression import ROOT, OUT, metrics
from phgeofuse.cache import atomic_json
from phgeofuse.reliability_gate import ReliabilityGate
from phgeofuse.io import read_manifest
from phgeofuse.retrieval import RetrievalStore, record_key
from phgeofuse.robust_fusion import pool_features
from phgeofuse.robust_fusion import chemistry_features
from phgeofuse.robust_train import frequency_weights
from sklearn.ensemble import HistGradientBoostingRegressor


def inputs(r, s, constant=False):
    # No expert predicted pH enters the quality features.
    f = r[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14]].astype(float)
    if constant:
        f = np.empty((len(r), 0))
    experts = np.column_stack([s, r[:, 0], r[:, 1]])
    available = np.column_stack([np.ones(len(r), bool), r[:, 7:9] > 0])
    return f, experts, available


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--residual', action='store_true')
    args = parser.parse_args()
    source = OUT / 'multiview_nested_20260915'
    control = json.loads((source/'results.json').read_text())['dual_ridge_control']
    assert control['control_max_abs_diff'] < 1e-7
    out = OUT / ('reliability_residual_nested_20260915' if args.residual else 'reliability_gate_nested_20260915')
    out.mkdir(exist_ok=False)
    z = np.load(source / 'dual_ridge_control_predictions.npz')
    y, folds, groups, keys = z['y'], z['fold'], z['groups'], z['keys']
    retrieval = OUT / 'nested_homology_strict'
    recipes = [('constant', 0., True), ('quality_l2_001', .01, False), ('quality_l2_01', .1, False)]
    if args.residual:
        recipes = [('quality_residual_unweighted', .01, False), ('quality_residual_weighted', .01, False)]
    records = [r for r in read_manifest(ROOT/'artifacts/phgeofuse/manifest.csv')
               if r.split in ('train', 'validation')]
    train = np.array([r.split == 'train' for r in records])
    allkeys = np.array([record_key(r) for r in records])
    assert np.array_equal(keys, allkeys[train])
    assert np.array_equal(z['validation_keys'], allkeys[~train])
    chemall = chemistry_features([r.sequence for r in records])
    chem, chemv = chemall[train], chemall[~train]
    store = RetrievalStore.load(ROOT/'artifacts/phgeofuse/retrieval.pt')
    assert store.payload['training_keys'] == keys.tolist()
    rv = np.array([store.features(k).numpy() for k in allkeys[~train]], float)
    parts=[]
    for encoder in ('esm1v', 'esm2'):
        with np.load(OUT/f'{encoder}_masked/features.npz') as f:
            mapping = {str(k): i for i,k in enumerate(f['keys'])}
            idx = [mapping[k] for k in allkeys[~train]]
            parts.append(pool_features(f['mean'][idx],f['std'][idx],'mean_std'))
    sequence_model = joblib.load(source/'dual_ridge_control_sequence.joblib')
    sv = sequence_model.predict(np.column_stack(parts))
    lowv = ~((rv[:,4]>=.2)&(rv[:,9]>=.8)&(rv[:,10]>=.8))
    def correction(name, r, s, c, target):
        power = .25 if name.endswith('_weighted') else 0.
        return HistGradientBoostingRegressor(max_leaf_nodes=7,max_iter=50,
            min_samples_leaf=80,l2_regularization=30,learning_rate=.05,
            early_stopping=False,random_state=42).fit(np.column_stack([r,s,c]),
                target, sample_weight=frequency_weights(current_y, power))
    atomic_json(out/'protocol.json', {'dataset': 'PHOPT', 'test_access': False,
        'sequence_source': str(source), 'retrieval_source': str(retrieval),
        'recipes': recipes, 'exclusion': 'inner training features exclude both outer and inner folds',
        'scope': 'convex sequence/SaProt/Foldseek gate; optional residual HGB; no neural blend',
        'residual': args.residual,
        'residual_training': 'gate fit on inner OOF features; residual on same training rows; '
            'outer labels excluded from both fits, training error is in-sample',
        'selection': 'fixed exploratory recipes, not post-selection confirmation'})
    def load(excluded):
        tag = '_'.join(map(str, sorted(excluded)))
        q = np.flatnonzero(np.isin(folds, excluded))
        ref = np.flatnonzero(~np.isin(folds, excluded))
        b = np.load(source / f'dual_ridge_control_excluded_{tag}.npz')
        r = torch.load(retrieval / f'excluded_{tag}.pt', map_location='cpu')
        assert np.array_equal(b['keys'], keys[q]) and np.array_equal(b['reference_keys'], keys[ref])
        assert r['metadata']['query_keys'] == keys[q].tolist()
        assert r['metadata']['reference_keys'] == keys[ref].tolist()
        assert not set(groups[q]) & set(groups[ref])
        return q, np.asarray(r['retrieval'], float), b['prediction']
    results = {}
    for name, penalty, constant in recipes:
        prediction = np.full(len(y), np.nan)
        rows = []
        for outer in range(5):
            atomic_json(out/'status.json', {'status': 'running', 'pid': os.getpid(),
                'recipe': name, 'outer': outer, 'updated': time.time()})
            tr = np.flatnonzero(folds != outer)
            te, r, s = load([outer])
            ir, iseq = np.full((len(y), 15), np.nan), np.full(len(y), np.nan)
            for inner in range(5):
                if inner == outer:
                    continue
                q, rr, ss = load([outer, inner])
                keep = folds[q] == inner
                ir[q[keep]], iseq[q[keep]] = rr[keep], ss[keep]
            assert np.isnan(iseq[te]).all() and np.isfinite(iseq[tr]).all()
            fit_inputs = inputs(ir[tr], iseq[tr], constant)
            model = ReliabilityGate(penalty).fit(*fit_inputs, y[tr])
            prediction[te] = model.predict(*inputs(r, s, constant))
            train_prediction = model.predict(*fit_inputs)
            if args.residual:
                current_y = y[tr]
                residual = correction(name, ir[tr], iseq[tr], chem[tr], y[tr]-train_prediction)
                prediction[te] += residual.predict(np.column_stack([r,s,chem[te]]))
                train_prediction += residual.predict(np.column_stack([ir[tr],iseq[tr],chem[tr]]))
                joblib.dump(residual,out/f'{name}_outer{outer}_residual.joblib')
            low = ~((r[:,4]>=.2)&(r[:,9]>=.8)&(r[:,10]>=.8))
            rows.append({'outer': outer, 'heldout': metrics(y[te], prediction[te], low),
                'train_rmse': float(np.sqrt(np.mean((train_prediction-y[tr])**2))),
                'iterations': model.iterations_})
            joblib.dump(model, out/f'{name}_outer{outer}.joblib')
        r = z['retrieval']
        low = ~((r[:,4]>=.2)&(r[:,9]>=.8)&(r[:,10]>=.8))
        _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
        draws = np.random.default_rng(42).integers(len(counts), size=(1000,len(counts)))
        denom = counts[draws].sum(1)
        sse = np.bincount(inv, weights=(prediction-y)**2)
        base_sse = np.bincount(inv, weights=(z['prediction']-y)**2)
        delta = np.sqrt(sse[draws].sum(1)/denom)-np.sqrt(base_sse[draws].sum(1)/denom)
        result = {'strict_nested': metrics(y,prediction,low), 'fold_results': rows,
            'family_bootstrap_rmse_delta95_vs_dual': np.quantile(delta,[.025,.975]).tolist()}
        model = ReliabilityGate(penalty).fit(*inputs(r,z['sequence'],constant), y)
        val = model.predict(*inputs(rv,sv,constant))
        if args.residual:
            current_y = y
            residual = correction(name,r,z['sequence'],chem,y-model.predict(*inputs(r,z['sequence'],constant)))
            val += residual.predict(np.column_stack([rv,sv,chemv]))
            joblib.dump(residual,out/f'{name}_residual.joblib')
        joblib.dump(model,out/f'{name}_gate.joblib')
        result['validation'] = metrics(z['yv'],val,lowv)
        np.savez(out/f'{name}_predictions.npz', prediction=prediction, y=y, keys=keys, fold=folds,
            groups=groups, validation=val, yv=z['yv'], validation_keys=allkeys[~train])
        results[name] = result
        atomic_json(out/'results.json', results)
        print('GATE_RECIPE_COMPLETE', name, json.dumps(result['strict_nested']), flush=True)
    atomic_json(out/'status.json', {'status': 'complete', 'pid': os.getpid(), 'updated': time.time()})
    print('RELIABILITY_GATE_NESTED_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
