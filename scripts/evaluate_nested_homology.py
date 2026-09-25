"""Train-only nested grouped evaluation of a redesigned sequence/retrieval stacker.

Outer-fold labels are excluded from all experts, retrieval references and meta
training. Fixed recipes are compared; outer scores are not used to choose a
recipe inside a fold. This is a research comparison, not a new final test.
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '8'

import csv
import argparse
import fcntl
import hashlib
import itertools
import json
import time
from dataclasses import replace
from pathlib import Path
import sys

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from develop_phgeofuse_regression import ROOT, OUT, metrics
from phgeofuse.cache import atomic_json, atomic_torch_save
from phgeofuse.config import load_config
from phgeofuse.io import read_manifest, read_fasta
from phgeofuse.retrieval import RetrievalStore, record_key, _build_retrieval_rows
from phgeofuse.robust_fusion import chemistry_features, pool_features
from phgeofuse.robust_train import frequency_weights


RECIPES = [
    {'name': 'unweighted50', 'iterations': 50, 'ph_power': 0., 'family_power': 0.},
    {'name': 'unweighted150', 'iterations': 150, 'ph_power': 0., 'family_power': 0.},
    {'name': 'phweighted50', 'iterations': 50, 'ph_power': .25, 'family_power': 0.},
    {'name': 'family_phweighted50', 'iterations': 50, 'ph_power': .25, 'family_power': .5},
]


def anchor(retrieval, sequence):
    available = retrieval[:, 7:9]
    return ((retrieval[:, :2] * available).sum(1) + sequence) / (available.sum(1) + 1)


def weights(labels, groups, recipe):
    value = frequency_weights(labels, recipe['ph_power'])
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    value *= counts[inverse].astype(float) ** -recipe['family_power']
    return np.clip(value / value.mean(), .25, 4.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folds', type=Path, default=OUT / 'homology_oof/folds.json')
    parser.add_argument('--output', type=Path, default=OUT / 'nested_homology')
    args = parser.parse_args()
    torch.set_num_threads(4)
    work = args.output
    work.mkdir(parents=True, exist_ok=True)
    lock = (work / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def status(phase, **extra):
        row = {'status': 'running', 'phase': phase, 'pid': os.getpid(),
               'updated': time.time(), **extra}
        atomic_json(work / 'status.json', row)
        print(json.dumps(row), flush=True)

    manifest = ROOT / 'artifacts/phgeofuse/manifest.csv'
    records = [r for r in read_manifest(manifest) if r.split == 'train']
    official = read_fasta(ROOT / 'data/phopt_training.fasta', 'train')
    signature = lambda rs: sorted((r.protein_id, r.sequence, r.ph_opt) for r in rs)
    assert signature(records) == signature(official) and len(records) == 7124
    assert all(r.status == 'ready' for r in records)
    keys = [record_key(r) for r in records]
    y = np.array([r.ph_opt for r in records])
    foldrows = json.loads(args.folds.read_text())['rows']
    assert [r['key'] for r in foldrows] == [r.protein_id for r in records]
    folds = np.array([r['fold'] for r in foldrows])
    groups = np.array([r['group'] for r in foldrows])
    store = RetrievalStore.load(ROOT / 'artifacts/phgeofuse/retrieval.pt')
    assert store.payload['training_keys'] == keys
    assert store.payload['training_sequences'] == [r.sequence for r in records]
    assert np.allclose(store.payload['training_labels'].numpy(), y, atol=1e-6)
    with np.load(OUT / 'esm2_masked/features.npz') as features:
        order = {str(k): i for i, k in enumerate(features['keys'])}
        indices = [order[k] for k in keys]
        x = pool_features(features['mean'][indices], features['std'][indices], 'mean_std')
    chem = chemistry_features([r.sequence for r in records])
    config = load_config(ROOT / 'configs/phgeofuse_phopt_homology_gate_v3.yaml')
    config['retrieval'].update(require_mmseqs=True, require_foldseek=True, search_threads=8)
    protocol = {
        'dataset': 'original PHOPT train only', 'count': len(y), 'groups': len(set(groups)),
        'outer_folds': 5, 'inner_folds': 4,
        'sequence': {'alpha': .1, 'pooling': 'mean_std', 'weighting': 'none'},
        'meta': {'leaves': 7, 'min_leaf': 80, 'l2': 30, 'learning_rate': .05, 'seed': 42},
        'recipes': RECIPES, 'retrieval': config['retrieval'],
        'protocol_note': 'For outer k and inner j, every base expert and retrieval reference '
                         'excludes both k and j. Meta training uses only rows outside k. '
                         'No validation/test fitting or evaluation. Cluster representative '
                         'thresholds do not guarantee pairwise cross-fold identity bounds.',
        'train_sha256': hashlib.sha256(json.dumps(signature(records)).encode()).hexdigest(),
        'fold_sha256': hashlib.sha256(json.dumps(foldrows).encode()).hexdigest(),
        'pooled_features_sha256': hashlib.sha256(x.tobytes()).hexdigest(),
        'retrieval_code_sha256': hashlib.sha256((ROOT / 'phgeofuse/retrieval.py').read_bytes()).hexdigest(),
    }
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    existing = work / 'protocol.json'
    if existing.exists():
        assert json.loads(existing.read_text()) == protocol, 'existing experiment protocol differs'
    atomic_json(existing, protocol)

    def base_features(excluded):
        excluded = tuple(sorted(excluded))
        tag = '_'.join(map(str, excluded))
        query = np.flatnonzero(np.isin(folds, excluded))
        reference = np.flatnonzero(~np.isin(folds, excluded))
        assert not set(groups[query]) & set(groups[reference])
        path = work / f'excluded_{tag}.pt'
        metadata = {'protocol_hash': protocol_hash, 'excluded_folds': list(excluded),
                    'query_keys': [keys[i] for i in query],
                    'reference_keys': [keys[i] for i in reference]}
        if path.exists():
            payload = torch.load(path, map_location='cpu')
            assert payload['metadata'] == metadata
            return query, payload['retrieval'], payload['sequence']
        status('build_retrieval', excluded=list(excluded), query=len(query), reference=len(reference))
        queries = [replace(records[i], ph_opt=float('nan'), ec='', organism='', sample_weight=1.)
                   for i in query]
        rows = _build_retrieval_rows(queries, [records[i] for i in reference],
                                    store.payload['training_vectors'][reference].float(),
                                    torch.tensor(y[reference], dtype=torch.float32), config)
        assert set(rows) == set(metadata['query_keys'])
        view = RetrievalStore({'rows': rows})
        retrieval = np.array([view.features(keys[i]).numpy() for i in query])
        sequence = Ridge(alpha=.1, solver='cholesky').fit(x[reference], y[reference])
        prediction = sequence.predict(x[query])
        assert np.isfinite(retrieval).all() and np.isfinite(prediction).all()
        atomic_torch_save(path, {'metadata': metadata, 'retrieval': retrieval,
                                 'sequence': prediction})
        status('base_features_complete', excluded=list(excluded))
        return query, retrieval, prediction

    predictions = {r['name']: np.full(len(y), np.nan) for r in RECIPES}
    predictions.update(sequence=np.full(len(y), np.nan), anchor=np.full(len(y), np.nan))
    outer_retrieval = np.full((len(y), 15), np.nan)
    fold_results = []
    for outer in range(5):
        training = np.flatnonzero(folds != outer)
        test, retr, seq = base_features([outer])
        outer_retrieval[test] = retr
        inner_retr = np.full((len(y), 15), np.nan)
        inner_seq = np.full(len(y), np.nan)
        for inner in range(5):
            if inner == outer:
                continue
            query, inner_r, inner_s = base_features([outer, inner])
            keep = folds[query] == inner
            inner_retr[query[keep]] = inner_r[keep]
            inner_seq[query[keep]] = inner_s[keep]
        assert np.isnan(inner_seq[test]).all()
        assert np.isfinite(inner_seq[training]).all()
        features = np.column_stack([inner_retr[training], inner_seq[training], chem[training]])
        heldout = np.column_stack([retr, seq, chem[test]])
        train_anchor = anchor(inner_retr[training], inner_seq[training])
        test_anchor = anchor(retr, seq)
        predictions['sequence'][test] = seq
        predictions['anchor'][test] = test_anchor
        low = ~((retr[:, 4] >= .2) & (retr[:, 9] >= .8) & (retr[:, 10] >= .8))
        audit = {'outer': outer, 'train_keys': [keys[i] for i in training],
                 'heldout_keys': [keys[i] for i in test], 'models': []}
        for recipe in RECIPES:
            status('fit_meta', outer=outer, recipe=recipe['name'])
            model = HistGradientBoostingRegressor(max_leaf_nodes=7, max_iter=recipe['iterations'],
                min_samples_leaf=80, l2_regularization=30, learning_rate=.05,
                early_stopping=False, random_state=42)
            model.fit(features, y[training] - train_anchor,
                      sample_weight=weights(y[training], groups[training], recipe))
            pred = test_anchor + model.predict(heldout)
            predictions[recipe['name']][test] = pred
            fitted = train_anchor + model.predict(features)
            row = {'outer': outer, 'name': recipe['name'], 'heldout': metrics(y[test], pred, low),
                   'meta_training_rmse': float(np.sqrt(np.mean((y[training] - fitted) ** 2)))}
            fold_results.append(row)
            audit['models'].append(row)
            joblib.dump(model, work / f'outer{outer}_{recipe["name"]}.joblib')
            print(json.dumps(row), flush=True)
        atomic_json(work / f'outer{outer}_audit.json', audit)
        np.savez(work / 'partial_predictions.npz', **predictions, fold=folds)
        atomic_json(work / 'fold_results.json', fold_results)

    low = ~((outer_retrieval[:, 4] >= .2) & (outer_retrieval[:, 9] >= .8) & (outer_retrieval[:, 10] >= .8))
    assert all(np.isfinite(p).all() for p in predictions.values())
    result = {'protocol_hash': protocol_hash, 'metrics': {n: metrics(y, p, low) for n, p in predictions.items()},
              'fold_results': fold_results, 'note': 'All final predictions are outer-held-out; '
              'meta_training_rmse is optimistic and is not the primary generalization estimate.'}
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    rng = np.random.default_rng(42)
    draws = rng.integers(len(unique), size=(1000, len(unique)))
    reference_sse = np.bincount(inverse, weights=(predictions['anchor'] - y) ** 2)
    bootstrap = {}
    for name, pred in predictions.items():
        sse = np.bincount(inverse, weights=(pred - y) ** 2)
        denominator = counts[draws].sum(1)
        delta = np.sqrt(sse[draws].sum(1) / denominator) - np.sqrt(reference_sse[draws].sum(1) / denominator)
        bootstrap[name] = np.quantile(delta, [.025, .975]).tolist()
    result['cluster_bootstrap_rmse_delta_vs_anchor_95ci'] = bootstrap
    np.savez(work / 'predictions.npz', **predictions, y=y, fold=folds, groups=groups,
             keys=np.array(keys), retrieval=outer_retrieval)
    atomic_json(work / 'results.json', result)
    status('complete')
    atomic_json(work / 'status.json', {'status': 'complete', 'pid': os.getpid(), 'updated': time.time()})
    print('NESTED_HOMOLOGY_COMPLETE', json.dumps(result['metrics']), flush=True)


if __name__ == '__main__':
    main()
