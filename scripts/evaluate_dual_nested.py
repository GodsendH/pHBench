"""Nested evaluation of dual sequence representations using audited retrieval caches."""
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '8'
import json
import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_nested_homology import RECIPES, weights
from phgeofuse.dual_fusion import retrieval_sequence_anchor as anchor
from develop_phgeofuse_regression import ROOT, OUT, metrics
from phgeofuse.cache import atomic_json
from phgeofuse.robust_fusion import chemistry_features, pool_features
from phgeofuse.io import read_manifest
from phgeofuse.retrieval import RetrievalStore, record_key
from phgeofuse.bias_residual import BiasConstrainedResidual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bias-constrained', action='store_true')
    parser.add_argument('--extra-ankh', action='store_true')
    args = parser.parse_args()
    if args.bias_constrained and args.extra_ankh:
        parser.error('evaluate the representation and bias experiments separately')
    encoders = ['esm1v', 'esm2'] + (['ankh'] if args.extra_ankh else [])
    sequence_alpha = .3 if args.extra_ankh else .2
    out = OUT / ('bias_nested' if args.bias_constrained else
                 'triple_nested' if args.extra_ankh else 'dual_nested')
    recipes = ([{'name': f'a{a}_b{b}', 'alpha': a, 'bias_penalty': b,
                 'ph_power': 0., 'family_power': 0.}
                for a in [100., 1000.] for b in [0., .1, .5]]
               if args.bias_constrained else RECIPES)

    def fit_model(recipe, meta, residual, labels, family):
        if args.bias_constrained:
            return BiasConstrainedResidual(recipe['alpha'], recipe['bias_penalty']).fit(
                meta, residual, labels=labels)
        model = HistGradientBoostingRegressor(max_leaf_nodes=7, max_iter=recipe['iterations'],
            min_samples_leaf=80, l2_regularization=30, learning_rate=.05,
            early_stopping=False, random_state=42)
        return model.fit(meta, residual, sample_weight=weights(labels, family, recipe))
    out.mkdir(exist_ok=True)
    source = OUT / 'nested_homology_strict'
    assert json.loads((source / 'status.json').read_text())['status'] == 'complete'
    records = [r for r in read_manifest(ROOT / 'artifacts/phgeofuse/manifest.csv')
               if r.split in ('train', 'validation')]
    keys = np.array([record_key(r) for r in records])
    training = np.array([r.split == 'train' for r in records])
    assert training.sum() == 7124 and (~training).sum() == 760
    trainkeys = keys[training]
    y = np.array([r.ph_opt for r in records])[training]
    yv = np.array([r.ph_opt for r in records])[~training]
    foldrows = json.loads((OUT / 'homology_oof/strict_folds.json').read_text())['rows']
    assert [r['key'] for r in foldrows] == [r.protein_id for r in records if r.split == 'train']
    fold = np.array([r['fold'] for r in foldrows])
    groups = np.array([r['group'] for r in foldrows])
    parts = []
    for encoder in encoders:
        with np.load(OUT / f'{encoder}_masked/features.npz') as f:
            mapping = {str(k): i for i, k in enumerate(f['keys'])}
            indices = [mapping[k] for k in keys]
            parts.append(pool_features(f['mean'][indices], f['std'][indices], 'mean_std'))
    xall = np.column_stack(parts)
    x, xv = xall[training], xall[~training]
    chemall = chemistry_features([r.sequence for r in records])
    chem, chemv = chemall[training], chemall[~training]
    atomic_json(out / 'protocol.json', {'representation': ' + '.join(e + ' mean/std' for e in encoders),
        'alpha': sequence_alpha, 'sequence_weighting': 'none', 'recipes': recipes,
        'folds': str(OUT / 'homology_oof/strict_folds.json'), 'retrieval_cache': str(source),
        'test_access': False, 'caveat': 'Exploratory development following representation comparisons. '
        'Outer-fold labels remain excluded from that fold\'s fitting; comparisons are not a new untouched test.'})
    cache = {}

    def base(excluded):
        excluded = tuple(sorted(excluded))
        if excluded in cache:
            return cache[excluded]
        path = source / ('excluded_' + '_'.join(map(str, excluded)) + '.pt')
        payload = torch.load(path, map_location='cpu')
        query = np.flatnonzero(np.isin(fold, excluded))
        ref = np.flatnonzero(~np.isin(fold, excluded))
        assert payload['metadata']['query_keys'] == trainkeys[query].tolist()
        assert payload['metadata']['reference_keys'] == trainkeys[ref].tolist()
        assert not set(groups[query]) & set(groups[ref])
        model = Ridge(alpha=sequence_alpha, solver='cholesky').fit(x[ref], y[ref])
        pred = model.predict(x[query])
        cache[excluded] = query, payload['retrieval'], pred
        print('DUAL_BASE_COMPLETE', excluded, flush=True)
        return cache[excluded]

    predictions = {r['name']: np.full(len(y), np.nan) for r in recipes}
    predictions.update(sequence=np.full(len(y), np.nan), anchor=np.full(len(y), np.nan))
    retrieval = np.full((len(y), 15), np.nan)
    foldresults = []
    for outer in range(5):
        tr = np.flatnonzero(fold != outer)
        te, r, s = base([outer])
        retrieval[te] = r
        innerr = np.full((len(y), 15), np.nan)
        inners = np.full(len(y), np.nan)
        for inner in range(5):
            if inner == outer:
                continue
            query, rr, ss = base([outer, inner])
            keep = fold[query] == inner
            innerr[query[keep]] = rr[keep]
            inners[query[keep]] = ss[keep]
        assert np.isnan(inners[te]).all() and np.isfinite(inners[tr]).all()
        meta = np.column_stack([innerr[tr], inners[tr], chem[tr]])
        heldout = np.column_stack([r, s, chem[te]])
        a, av = anchor(innerr[tr], inners[tr]), anchor(r, s)
        predictions['sequence'][te], predictions['anchor'][te] = s, av
        low = ~((r[:, 4] >= .2) & (r[:, 9] >= .8) & (r[:, 10] >= .8))
        for recipe in recipes:
            model = fit_model(recipe, meta, y[tr] - a, y[tr], groups[tr])
            p = av + model.predict(heldout)
            predictions[recipe['name']][te] = p
            row = {'outer': outer, 'recipe': recipe, 'heldout': metrics(y[te], p, low),
                   'meta_training_rmse': float(np.sqrt(np.mean((a + model.predict(meta) - y[tr]) ** 2)))}
            foldresults.append(row)
            joblib.dump(model, out / f'outer{outer}_{recipe["name"]}.joblib')
            print(json.dumps(row), flush=True)
        atomic_json(out / 'status.json', {'status': 'running', 'pid': os.getpid(), 'outer_complete': outer,
                                          'updated': time.time()})
    assert all(np.isfinite(p).all() for p in predictions.values())
    low = ~((retrieval[:, 4] >= .2) & (retrieval[:, 9] >= .8) & (retrieval[:, 10] >= .8))
    result = {'strict_nested': {n: metrics(y, p, low) for n, p in predictions.items()},
              'fold_results': foldresults, 'validation': {}}
    np.savez(out / 'predictions.npz', **predictions, keys=trainkeys, fold=fold, groups=groups, y=y)

    # Refit each fixed recipe on all training OOF features and evaluate validation.
    sequence = Ridge(alpha=sequence_alpha, solver='cholesky').fit(x, y)
    joblib.dump(sequence, out / 'sequence.joblib')
    seqv = sequence.predict(xv)
    store = RetrievalStore.load(ROOT / 'artifacts/phgeofuse/retrieval.pt')
    assert store.payload['training_keys'] == trainkeys.tolist()
    rv = np.array([store.features(k).numpy() for k in keys[~training]])
    lowv = ~((rv[:, 4] >= .2) & (rv[:, 9] >= .8) & (rv[:, 10] >= .8))
    valmeta = np.column_stack([rv, seqv, chemv])
    meta = np.column_stack([retrieval, predictions['sequence'], chem])
    a, av = predictions['anchor'], anchor(rv, seqv)
    for recipe in recipes:
        model = fit_model(recipe, meta, y - a, y, groups)
        pred = av + model.predict(valmeta)
        np.save(out / (recipe['name'] + '.validation.npy'), pred)
        joblib.dump(model, out / (recipe['name'] + '.joblib'))
        result['validation'][recipe['name']] = metrics(yv, pred, lowv)
    # Conditional cluster bootstrap for paired strict-OOF comparison.
    baseline = np.load(source / 'predictions.npz')
    assert np.array_equal(baseline['keys'], trainkeys)
    _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    rng = np.random.default_rng(42)
    draws = rng.integers(len(counts), size=(1000, len(counts)))
    denom = counts[draws].sum(1)
    result['cluster_bootstrap_delta_vs_esm2_unweighted50'] = {}
    ref = np.bincount(inv, weights=(baseline['unweighted50'] - y) ** 2)
    for name, p in predictions.items():
        sse = np.bincount(inv, weights=(p - y) ** 2)
        delta = np.sqrt(sse[draws].sum(1) / denom) - np.sqrt(ref[draws].sum(1) / denom)
        result['cluster_bootstrap_delta_vs_esm2_unweighted50'][name] = np.quantile(delta, [.025, .975]).tolist()
    atomic_json(out / 'results.json', result)
    atomic_json(out / 'status.json', {'status': 'complete', 'pid': os.getpid(), 'updated': time.time()})
    print('DUAL_NESTED_COMPLETE', json.dumps(result['strict_nested']), flush=True)


if __name__ == '__main__':
    main()
