"""Extra-annotation diagnostic: EC features are not a sequence-only benchmark."""
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '8'
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from develop_phgeofuse_regression import ROOT, OUT, metrics
from phgeofuse.io import read_manifest
from phgeofuse.robust_fusion import pool_features
from phgeofuse.cache import atomic_json


def main():
    out = OUT / 'ec_information'
    out.mkdir(exist_ok=True)
    records = [r for r in read_manifest(ROOT / 'artifacts/phgeofuse/manifest.csv')
               if r.split in ('train', 'validation')]
    keys = [r.split + '::' + r.protein_id for r in records]
    train = np.array([r.split == 'train' for r in records])
    yall = np.array([r.ph_opt for r in records])
    y, yv = yall[train], yall[~train]
    parts = []
    for name in ['esm1v', 'esm2']:
        with np.load(OUT / f'{name}_masked/features.npz') as f:
            order = {str(k): i for i, k in enumerate(f['keys'])}
            idx = [order[k] for k in keys]
            parts.append(pool_features(f['mean'][idx], f['std'][idx], 'mean_std'))
    features = np.column_stack(parts)
    x, xv = features[train], features[~train]
    categories = []
    for r in records:
        ec = r.ec.strip()
        if not re.fullmatch(r'[1-7](?:\.(?:\d+|-)){3}', ec):
            categories.append(['unknown']*3)
        else:
            bits = ec.split('.')
            categories.append(['.'.join(bits[:i]) for i in [1, 2, 3]])
    categories = np.array(categories)
    ec, ecv = categories[train], categories[~train]
    foldrows = json.loads((OUT / 'homology_oof/strict_folds.json').read_text())['rows']
    assert [r['key'] for r in foldrows] == [r.protein_id for r in records if r.split == 'train']
    fold = np.array([r['fold'] for r in foldrows])
    groups = np.array([r['group'] for r in foldrows])
    reference = np.load(OUT / 'nested_homology_strict/predictions.npz')
    assert np.array_equal(reference['keys'], np.array(keys)[train])
    r = reference['retrieval']
    low = ~((r[:, 4] >= .2) & (r[:, 9] >= .8) & (r[:, 10] >= .8))
    lowv = np.load(OUT / 'development_features.npz')['lowv']
    atomic_json(out / 'protocol.json', {'extra_annotation': 'Provided EC hierarchy levels 1-3',
        'unknown_train': int((ec[:, 0] == 'unknown').sum()), 'unknown_validation': int((ecv[:, 0] == 'unknown').sum()),
        'feature_scales': [.1, .3], 'ridge_alpha': .2, 'test_access': False,
        'caveat': 'EC is an additional input. Results do not establish sequence-only improvement. '
                  'Categories are fit in each training fold; unseen categories encode as all zeros.'})
    results = []
    for scale in [.1, .3]:
        oof = np.full(len(y), np.nan)
        train_errors = []
        unseen_counts = []
        for k in range(6):
            tr = np.flatnonzero(fold != k) if k < 5 else np.arange(len(y))
            te = np.flatnonzero(fold == k) if k < 5 else None
            if k < 5:
                assert not set(groups[tr]) & set(groups[te])
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float64)
            etrain = encoder.fit_transform(ec[tr])
            eheld = encoder.transform(ec[te] if k < 5 else ecv)
            z = sparse.hstack([sparse.csr_matrix(x[tr]), scale*etrain], format='csr')
            zv = sparse.hstack([sparse.csr_matrix(x[te] if k < 5 else xv), scale*eheld], format='csr')
            atomic_json(out / 'status.json', {'status': 'running', 'pid': os.getpid(),
                'updated': time.time(), 'scale': scale, 'fold': k})
            model = Ridge(alpha=.2, solver='lsqr', tol=1e-7, max_iter=1000).fit(z, y[tr])
            pred = model.predict(zv)
            if k < 5:
                oof[te] = pred
                train_errors.append(float(np.sqrt(np.mean((model.predict(z)-y[tr])**2))))
                unseen_counts.append(int((eheld.getnnz(axis=1) < 3).sum()))
            else:
                validation = pred
            print('EC_DIAGNOSTIC_FOLD', scale, k, 'iterations', model.n_iter_.tolist(), flush=True)
        row = {'ec_feature_scale': scale, 'strict_oof': metrics(y, oof, low),
            'validation': metrics(yv, validation, lowv), 'training_rmse_mean': float(np.mean(train_errors)),
            'outer_samples_with_unseen_ec_level': sum(unseen_counts)}
        results.append(row)
        np.savez(out / f'scale{scale}.predictions.npz', oof=oof, validation=validation)
        atomic_json(out / 'results.json', results)
        print(json.dumps(row), flush=True)
    atomic_json(out / 'status.json', {'status': 'complete', 'pid': os.getpid(), 'updated': time.time()})
    print('EC_INFORMATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
