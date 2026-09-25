"""Strict grouped OOF annotation expert using PHOPT header metadata.

EC and organism are available in the PHOPT manifest. This is an annotation-assisted
branch and is never mixed into the sequence-only benchmark silently.
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'
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
from phgeofuse.cache import atomic_json
from phgeofuse.io import read_manifest
from phgeofuse.robust_fusion import pool_features


def ec_levels(value):
    value = value.strip()
    if not re.fullmatch(r'[1-7](?:\.(?:\d+|-)){3}', value):
        return ['unknown', 'unknown', 'unknown']
    parts = value.split('.')
    return ['.'.join(parts[:i]) for i in [1, 2, 3]]


def organism_levels(value):
    value = re.sub(r'\s+', ' ', value.strip())
    if not value:
        return ['unknown', 'unknown']
    parts = value.split(' ')
    genus = parts[0]
    species = ' '.join(parts[:2]) if len(parts) > 1 else 'unknown'
    return [genus, species]


def main():
    out = OUT / 'annotation_expert'
    out.mkdir(exist_ok=True)
    records = [r for r in read_manifest(ROOT / 'artifacts/phgeofuse/manifest.csv')
               if r.split in ('train', 'validation')]
    keys = np.array([r.split + '::' + r.protein_id for r in records])
    train = np.array([r.split == 'train' for r in records])
    labels = np.array([r.ph_opt for r in records])
    y, yv = labels[train], labels[~train]
    parts = []
    for encoder in ['esm1v', 'esm2']:
        with np.load(OUT / f'{encoder}_masked/features.npz') as f:
            mapping = {str(k): i for i, k in enumerate(f['keys'])}
            idx = [mapping[k] for k in keys]
            parts.append(pool_features(f['mean'][idx], f['std'][idx], 'mean_std'))
    sequence = np.column_stack(parts)
    ec = np.array([ec_levels(r.ec) for r in records])
    org = np.array([organism_levels(r.organism) for r in records])
    annotations = {
        'ec1': ec[:, 0], 'ec2': ec[:, 1], 'ec3': ec[:, 2],
        'genus': org[:, 0], 'species': org[:, 1],
        'ec12': np.array([a + '|' + b for a, b in ec[:, :2]]),
    }
    foldrows = json.loads((OUT / 'homology_oof/strict_folds.json').read_text())['rows']
    assert [r['key'] for r in foldrows] == [r.protein_id for r in records if r.split == 'train']
    folds = np.array([r['fold'] for r in foldrows])
    groups = np.array([r['group'] for r in foldrows])
    reference = np.load(OUT / 'nested_homology_strict/predictions.npz')
    assert np.array_equal(reference['keys'], keys[train])
    retrieval = reference['retrieval']
    low = ~((retrieval[:, 4] >= .2) & (retrieval[:, 9] >= .8) & (retrieval[:, 10] >= .8))
    lowv = np.load(OUT / 'development_features.npz')['lowv']
    combinations = [
        {'name': 'ec', 'columns': ['ec1', 'ec2', 'ec3'], 'scale': .10},
        {'name': 'ec_org', 'columns': ['ec1', 'ec2', 'ec3', 'genus'], 'scale': .10},
        {'name': 'ec_species', 'columns': ['ec1', 'ec2', 'ec3', 'species'], 'scale': .10},
        {'name': 'ec12_org', 'columns': ['ec12', 'genus'], 'scale': .10},
        {'name': 'all', 'columns': list(annotations), 'scale': .05},
    ]
    atomic_json(out / 'protocol.json', {'combinations': combinations, 'base': 'ESM1v+ESM2 mean/std',
        'ridge_alpha': .2, 'one_hot': 'fit within each outer training fold; unknown categories zero',
        'strict_groups': str(OUT / 'homology_oof/strict_folds.json'), 'test_access': False,
        'annotation_assisted': True, 'note': 'These metadata are present in PHOPT headers. Results '
        'are not sequence-only and cannot be compared as a fair replacement without an annotation availability protocol.'})
    results = []
    for combo in combinations:
        oof = np.full(len(y), np.nan)
        train_errors = []
        unseen = []
        for k in range(5):
            tr, te = folds != k, folds == k
            assert not set(groups[tr]) & set(groups[te])
            enc = OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float64)
            atr = enc.fit_transform(np.column_stack([annotations[c][train] for c in combo['columns']])[tr])
            ate = enc.transform(np.column_stack([annotations[c][train] for c in combo['columns']])[te])
            z = sparse.hstack([sparse.csr_matrix(sequence[train][tr]), combo['scale'] * atr], format='csr')
            ze = sparse.hstack([sparse.csr_matrix(sequence[train][te]), combo['scale'] * ate], format='csr')
            model = Ridge(alpha=.2, solver='lsqr', tol=1e-7, max_iter=1000).fit(z, y[tr])
            oof[te] = model.predict(ze)
            train_errors.append(float(np.sqrt(np.mean((model.predict(z) - y[tr]) ** 2))))
            unseen.append(int((ate.getnnz(axis=1) == 0).sum()))
            atomic_json(out / 'status.json', {'status': 'running', 'pid': os.getpid(),
                'updated': time.time(), 'combination': combo['name'], 'fold': k})
        enc = OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float64)
        atr = enc.fit_transform(np.column_stack([annotations[c][train] for c in combo['columns']]))
        av = enc.transform(np.column_stack([annotations[c][~train] for c in combo['columns']]))
        z = sparse.hstack([sparse.csr_matrix(sequence[train]), combo['scale'] * atr], format='csr')
        zv = sparse.hstack([sparse.csr_matrix(sequence[~train]), combo['scale'] * av], format='csr')
        model = Ridge(alpha=.2, solver='lsqr', tol=1e-7, max_iter=1000).fit(z, y)
        val = model.predict(zv)
        row = {'combination': combo, 'strict_oof': metrics(y, oof, low),
               'validation': metrics(yv, val, lowv),
               'training_rmse_mean': float(np.mean(train_errors)),
               'outer_unseen_annotation_rows': int(sum(unseen))}
        results.append(row)
        np.savez(out / (combo['name'] + '.predictions.npz'), oof=oof, validation=val)
        atomic_json(out / 'results.json', results)
        print(json.dumps(row), flush=True)
    atomic_json(out / 'status.json', {'status': 'complete', 'pid': os.getpid(), 'updated': time.time()})
    print('ANNOTATION_EXPERT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
