"""Four fixed Ankh representation comparisons; strict grouped train OOF and validation."""
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
from phgeofuse.robust_fusion import pool_features
from phgeofuse.robust_train import frequency_weights
from phgeofuse.cache import atomic_json


def main():
    out = OUT / 'ankh_experts'
    out.mkdir(exist_ok=True)
    records = [r for r in read_manifest(ROOT / 'artifacts/phgeofuse/manifest.csv')
               if r.split in ('train', 'validation')]
    keys = [r.split + '::' + r.protein_id for r in records]
    train = np.array([r.split == 'train' for r in records])
    assert train.sum() == 7124 and (~train).sum() == 760
    labels = np.array([r.ph_opt for r in records])
    y, yv = labels[train], labels[~train]
    data = {}
    for encoder in ['ankh', 'esm1v', 'esm2']:
        with np.load(OUT / f'{encoder}_masked/features.npz') as f:
            order = {str(k): i for i, k in enumerate(f['keys'])}
            assert len(order) == len(f['keys'])
            idx = [order[k] for k in keys]
            data[encoder] = pool_features(f['mean'][idx], f['std'][idx], 'mean_std')
    data['triple'] = np.column_stack([data['esm1v'], data['esm2'], data['ankh']])
    foldrows = json.loads((OUT / 'homology_oof/strict_folds.json').read_text())['rows']
    assert [r['key'] for r in foldrows] == [r.protein_id for r in records if r.split == 'train']
    folds = np.array([r['fold'] for r in foldrows])
    groups = np.array([r['group'] for r in foldrows])
    recipes = [{'name': name, 'alpha': .1 if name == 'ankh' else .3, 'power': p}
               for name in ['ankh', 'triple'] for p in [0., .25]]
    atomic_json(out / 'protocol.json', {'recipes': recipes, 'test_access': False,
        'feature_order_triple': ['esm1v mean/std', 'esm2 mean/std', 'ankh mean/std'],
        'selection_note': 'Four development comparisons in established strict grouped folds; '
                          'best observed OOF is not independent confirmation.'})
    reference = np.load(OUT / 'nested_homology_strict/predictions.npz')
    assert np.array_equal(reference['keys'], np.array(keys)[train])
    retrieval = reference['retrieval']
    low = ~((retrieval[:, 4] >= .2) & (retrieval[:, 9] >= .8) & (retrieval[:, 10] >= .8))
    lowv = np.load(OUT / 'development_features.npz')['lowv']
    results = []
    for recipe in recipes:
        x, xv = data[recipe['name']][train], data[recipe['name']][~train]
        pred = np.full(len(y), np.nan)
        train_error = []
        for fold in range(5):
            tr = folds != fold
            te = ~tr
            assert not set(groups[tr]) & set(groups[te])
            atomic_json(out / 'status.json', {'status': 'running', 'pid': os.getpid(),
                'updated': time.time(), 'recipe': recipe, 'fold': fold})
            model = Ridge(alpha=recipe['alpha'], solver='cholesky')
            model.fit(x[tr], y[tr], sample_weight=frequency_weights(y[tr], recipe['power']))
            pred[te] = model.predict(x[te])
            train_error.append(float(np.sqrt(np.mean((model.predict(x[tr]) - y[tr])**2))))
            print('ANKH_OOF_FOLD', json.dumps(recipe), fold, flush=True)
        model = Ridge(alpha=recipe['alpha'], solver='cholesky')
        model.fit(x, y, sample_weight=frequency_weights(y, recipe['power']))
        pv = model.predict(xv)
        name = recipe['name'] + '_p' + str(recipe['power'])
        joblib.dump(model, out / (name + '.joblib'))
        np.savez(out / (name + '.predictions.npz'), oof=pred, validation=pv)
        row = {'recipe': recipe, 'train_rmse_mean': float(np.mean(train_error)),
               'strict_oof': metrics(y, pred, low), 'validation': metrics(yv, pv, lowv)}
        results.append(row)
        atomic_json(out / 'results.json', results)
        print(json.dumps(row), flush=True)
    atomic_json(out / 'status.json', {'status': 'complete', 'pid': os.getpid(), 'updated': time.time()})
    print('ANKH_EXPERTS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
