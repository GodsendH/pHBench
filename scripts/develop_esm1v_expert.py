"""Six fixed sequence recipes, strict grouped OOF and PHOPT validation only."""
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '8'
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from develop_phgeofuse_regression import ROOT, OUT, metrics
from phgeofuse.io import read_manifest
from phgeofuse.robust_train import frequency_weights
from phgeofuse.robust_fusion import pool_features
from phgeofuse.cache import atomic_json


def main():
    out = OUT / 'esm1v_experts'
    out.mkdir(exist_ok=True)
    records = [r for r in read_manifest(ROOT / 'artifacts/phgeofuse/manifest.csv')
               if r.split in ('train', 'validation')]
    train = np.array([r.split == 'train' for r in records])
    keys = [r.split + '::' + r.protein_id for r in records]
    y = np.array([r.ph_opt for r in records])[train]
    yv = np.array([r.ph_opt for r in records])[~train]
    assert train.sum() == 7124 and (~train).sum() == 760
    folds_data = json.loads((OUT / 'homology_oof/strict_folds.json').read_text())
    assert [r['key'] for r in folds_data['rows']] == [r.protein_id for r in records if r.split == 'train']
    folds = np.array([r['fold'] for r in folds_data['rows']])
    groups = np.array([r['group'] for r in folds_data['rows']])
    features = {}
    for model, path in [('esm1v', 'esm1v_masked/features.npz'), ('esm2', 'esm2_masked/features.npz')]:
        with np.load(OUT / path) as f:
            mapping = {str(k): i for i, k in enumerate(f['keys'])}
            idx = [mapping[k] for k in keys]
            features[model] = pool_features(f['mean'][idx], f['std'][idx], 'mean_std')
            if model == 'esm1v':
                mean = f['mean'][idx].astype(np.float64)
                norm = np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12)
                charge = []
                for k in ['acid', 'basic']:
                    values = f[k][idx].astype(np.float64)
                    missing = np.linalg.norm(values, axis=1) == 0
                    values = (values - mean) / norm
                    values[missing] = 0
                    charge.append(values)
                features['esm1v_charge'] = np.column_stack([features[model], *charge])
    features['esm1v_esm2'] = np.column_stack([features['esm1v'], features['esm2']])
    recipes = [{'name': name, 'alpha': .2 if name == 'esm1v_esm2' else .1, 'power': power}
               for name in ['esm1v', 'esm1v_charge', 'esm1v_esm2'] for power in [0., .25]]
    atomic_json(out / 'protocol.json', {'recipes': recipes, 'selection': 'Fixed recipes evaluated '
        'with train-only strict grouped OOF and original validation. No test access. Comparing '
        'recipes remains development; best observed OOF is not an unbiased post-selection estimate.',
        'strict_group_count': folds_data['groups'], 'seed': 42})
    development = np.load(OUT / 'development_features.npz')
    lowv = development['lowv']
    assert np.array_equal(development['keys'], keys)
    results = []
    for recipe in recipes:
        x = features[recipe['name']][train]
        xv = features[recipe['name']][~train]
        oof = np.full(len(y), np.nan)
        fit_rmse = []
        for fold in range(5):
            tr = folds != fold
            te = ~tr
            assert not set(groups[tr]) & set(groups[te])
            atomic_json(out / 'status.json', {'status': 'running', 'pid': os.getpid(),
                'updated': time.time(), 'recipe': recipe, 'fold': fold})
            model = Ridge(alpha=recipe['alpha'], solver='cholesky')
            model.fit(x[tr], y[tr], sample_weight=frequency_weights(y[tr], recipe['power']))
            oof[te] = model.predict(x[te])
            fit_rmse.append(float(np.sqrt(np.mean((model.predict(x[tr]) - y[tr]) ** 2))))
            print('SEQUENCE_OOF_FOLD', json.dumps(recipe), fold, flush=True)
        model = Ridge(alpha=recipe['alpha'], solver='cholesky')
        model.fit(x, y, sample_weight=frequency_weights(y, recipe['power']))
        val = model.predict(xv)
        name = recipe['name'] + '_p' + str(recipe['power'])
        joblib.dump(model, out / (name + '.joblib'))
        np.savez(out / (name + '.predictions.npz'), oof=oof, validation=val)
        row = {'recipe': recipe, 'train_rmse_mean': float(np.mean(fit_rmse)),
               'strict_oof': metrics(y, oof, np.zeros(len(y), dtype=bool)),
               'validation': metrics(yv, val, lowv)}
        results.append(row)
        atomic_json(out / 'results.json', results)
        print(json.dumps(row), flush=True)
    atomic_json(out / 'status.json', {'status': 'complete', 'pid': os.getpid(), 'updated': time.time()})
    print('ESM1V_EXPERTS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
