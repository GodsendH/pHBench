"""Bounded train/validation-only nonlinear sequence representation experiment."""
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '8'
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from develop_phgeofuse_regression import ROOT, OUT, metrics
from phgeofuse.cache import atomic_json
from phgeofuse.io import read_manifest
from phgeofuse.robust_fusion import pool_features
from phgeofuse.robust_train import frequency_weights


def main():
    out = OUT / 'nonlinear_sequence'
    out.mkdir(exist_ok=True)
    records = [r for r in read_manifest(ROOT / 'artifacts/phgeofuse/manifest.csv')
               if r.split in ('train', 'validation')]
    keys = [r.split + '::' + r.protein_id for r in records]
    train = np.array([r.split == 'train' for r in records])
    assert train.sum() == 7124 and (~train).sum() == 760
    y = np.array([r.ph_opt for r in records])[train]
    yv = np.array([r.ph_opt for r in records])[~train]
    parts = []
    for encoder in ['esm1v', 'esm2']:
        with np.load(OUT / f'{encoder}_masked/features.npz') as f:
            order = {str(k): i for i, k in enumerate(f['keys'])}
            idx = [order[k] for k in keys]
            parts.append(pool_features(f['mean'][idx], f['std'][idx], 'mean_std'))
    xall = np.column_stack(parts)
    x, xv = xall[train], xall[~train]
    rows = json.loads((OUT / 'homology_oof/strict_folds.json').read_text())['rows']
    assert [r['key'] for r in rows] == [r.protein_id for r in records if r.split == 'train']
    fold = np.array([r['fold'] for r in rows])
    groups = np.array([r['group'] for r in rows])
    recipes = [{'gamma_multiplier': g, 'alpha': a, 'power': p}
               for g, a, p in [(1., .1, 0.), (1., 1., 0.), (4., .1, 0.),
                               (4., 1., 0.), (1., .1, .25), (4., .1, .25)]]
    atomic_json(out / 'protocol.json', {'recipes': recipes, 'components': 1024, 'seed': 42,
        'feature_order': ['esm1v mean/std', 'esm2 mean/std'], 'folds': 'strict_folds.json',
        'kernel_scale': 'median pair distance in 512 training-only rows per fold',
        'note': 'Nystroem landmarks and scale fit inside training fold. Six development '
                'recipes, no test access. Best observed score remains selection-biased.'})
    predictions = [np.full(len(y), np.nan) for _ in recipes]
    training_rmse = [[] for _ in recipes]
    validation = [None for _ in recipes]
    for k in range(6):
        tr = np.flatnonzero(fold != k) if k < 5 else np.arange(len(y))
        te = np.flatnonzero(fold == k) if k < 5 else None
        if k < 5:
            assert not set(groups[tr]) & set(groups[te])
        rng = np.random.default_rng(42)
        subset = rng.choice(tr, size=min(512, len(tr)), replace=False)
        distances = pairwise_distances(x[subset], metric='sqeuclidean')
        scale = float(np.median(distances[np.triu_indices(len(subset), 1)]))
        assert scale > 0
        for g in [1., 4.]:
            atomic_json(out / 'status.json', {'status': 'running', 'pid': os.getpid(),
                'fold': k, 'gamma_multiplier': g, 'updated': time.time()})
            transform = Nystroem(kernel='rbf', gamma=g / scale, n_components=1024, random_state=42)
            z = transform.fit_transform(x[tr])
            zt = transform.transform(x[te] if k < 5 else xv)
            for i, recipe in enumerate(recipes):
                if recipe['gamma_multiplier'] != g:
                    continue
                model = Ridge(alpha=recipe['alpha'], solver='cholesky')
                model.fit(z, y[tr], sample_weight=frequency_weights(y[tr], recipe['power']))
                p = model.predict(zt)
                if k < 5:
                    predictions[i][te] = p
                    training_rmse[i].append(float(np.sqrt(np.mean((model.predict(z) - y[tr]) ** 2))))
                else:
                    validation[i] = p
                    joblib.dump(make_pipeline(transform, model), out / f'recipe{i}.joblib')
            print('NONLINEAR_FOLD', k, g, flush=True)
    lowv = np.load(OUT / 'development_features.npz')['lowv']
    strict = np.load(OUT / 'nested_homology_strict/predictions.npz')
    assert np.array_equal(strict['keys'], np.array(keys)[train])
    retrieval = strict['retrieval']
    low = ~((retrieval[:, 4] >= .2) & (retrieval[:, 9] >= .8) & (retrieval[:, 10] >= .8))
    result = []
    for i, recipe in enumerate(recipes):
        assert np.isfinite(predictions[i]).all()
        row = {'recipe': recipe, 'index': i, 'strict_oof': metrics(y, predictions[i], low),
               'training_rmse_mean': float(np.mean(training_rmse[i])),
               'validation': metrics(yv, validation[i], lowv)}
        result.append(row)
        np.savez(out / f'recipe{i}.predictions.npz', oof=predictions[i], validation=validation[i])
        print(json.dumps(row), flush=True)
    atomic_json(out / 'results.json', result)
    atomic_json(out / 'status.json', {'status': 'complete', 'pid': os.getpid(), 'updated': time.time()})
    print('NONLINEAR_SEQUENCE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
