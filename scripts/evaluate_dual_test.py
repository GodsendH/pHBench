"""Post-freeze PHOPT test evaluation; the test was previously evaluated for v1."""
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from develop_phgeofuse_regression import ROOT, OUT, metrics
from phgeofuse.cache import atomic_json
from phgeofuse.dual_fusion import DualFusion
from phgeofuse.io import read_manifest, read_fasta
from phgeofuse.retrieval import RetrievalStore, record_key


def main():
    bundle = OUT / 'dual_candidate_float64'
    out = OUT / 'dual_test'
    if (out / 'results.json').exists():
        raise FileExistsError('test results already recorded')
    out.mkdir(exist_ok=True)
    frozen_hash = hashlib.sha256((bundle / 'model.json').read_bytes()).hexdigest()
    assert json.loads((bundle / 'verification.json').read_text())['model_sha256'] == frozen_hash
    model = DualFusion(bundle)
    records = [r for r in read_manifest(ROOT / 'artifacts/phgeofuse/manifest.csv') if r.split == 'test']
    official = read_fasta(ROOT / 'data/phopt_testing.fasta', 'test')
    signature = lambda rs: sorted((r.protein_id, r.sequence, r.ph_opt) for r in rs)
    assert len(records) == 1971 and signature(records) == signature(official)
    keys = [record_key(r) for r in records]
    features = []
    for source in ['esm1v_masked/features_test.npz', 'esm2_masked/features_test.npz']:
        with np.load(OUT / source) as f:
            order = {str(k): i for i, k in enumerate(f['keys'])}
            assert set(order) == set(keys) and len(order) == len(f['keys'])
            idx = [order[k] for k in keys]
            features.extend([f['mean'][idx], f['std'][idx]])
    store = RetrievalStore.load(ROOT / 'artifacts/phgeofuse/retrieval.pt')
    train = read_fasta(ROOT / 'data/phopt_training.fasta', 'train')
    assert store.payload['training_keys'] == [record_key(r) for r in train]
    r = np.array([store.features(k).numpy() for k in keys])
    low = ~((r[:, 4] >= .2) & (r[:, 9] >= .8) & (r[:, 10] >= .8))
    y = np.array([r.ph_opt for r in records])
    results = []
    predictions = {'baseline': [], 'robust_v1': [], 'dual_candidate': []}
    for seed in [0, 1, 2, 3, 42]:
        source = ROOT / f'experiments/phgeofuse_phopt_full_20260913/seed{seed}/test_predictions.csv'
        with source.open() as handle:
            mapping = {row['key']: float(row['prediction']) for row in csv.DictReader(handle)}
        assert set(mapping) == set(keys)
        baseline = np.array([mapping[k] for k in keys])
        pred = model.predict(baseline, *features, r, [rec.sequence for rec in records])
        values = {'baseline': baseline, 'robust_v1': pred['robust_v1_prediction'],
                  'dual_candidate': pred['prediction']}
        row = {'seed': seed, **{n: metrics(y, p, low) for n, p in values.items()}}
        results.append(row)
        for n, p in values.items():
            predictions[n].append(p)
        with (out / f'seed{seed}.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['key', 'label', 'baseline', *pred])
            writer.writeheader()
            for i, key in enumerate(keys):
                writer.writerow({'key': key, 'label': y[i], 'baseline': baseline[i],
                                 **{n: p[i] for n, p in pred.items()}})
        print(json.dumps(row), flush=True)
    summary = {}
    for name in predictions:
        summary[name] = {k: {'mean': statistics.mean(row[name][k] for row in results),
                             'std': statistics.stdev(row[name][k] for row in results)}
                         for k in ['rmse', 'mae', 'r2', 'spearman']}
        summary[name]['groups'] = {g: {k: statistics.mean(row[name][g][k] for row in results)
                                      for k in ['rmse', 'bias']}
                                   for g in ['acidic', 'neutral', 'alkaline', 'low_homology']}
    rng = np.random.default_rng(42)
    candidate = np.array(predictions['dual_candidate'])
    bootstrap = {}
    for ref in ['baseline', 'robust_v1']:
        base = np.array(predictions[ref])
        deltas = []
        for _ in range(2000):
            ix = rng.integers(len(y), size=len(y))
            deltas.append(np.sqrt(np.mean((candidate[:, ix] - y[ix])**2, axis=1)).mean()
                          - np.sqrt(np.mean((base[:, ix] - y[ix])**2, axis=1)).mean())
        bootstrap[ref] = np.quantile(deltas, [.025, .975]).tolist()
    atomic_json(out / 'results.json', results)
    atomic_json(out / 'summary.json', {'metrics': summary, 'sample_bootstrap_rmse_delta_95ci': bootstrap,
        'test_count': len(y), 'low_count': int(low.sum()), 'model_sha256': frozen_hash,
        'protocol': 'One evaluation after this candidate freeze. Original PHOPT test was already '
                    'viewed for v1, so this is a follow-up test, not a new untouched confirmation. '
                    'Shared deterministic experts; seed SD reflects baseline seed variation. '
                    'Bootstrap uses samples, not families.'})
    print('DUAL_TEST_COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
