"""Separate in-sample full-model diagnostics from nested expert diagnostics."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from develop_phgeofuse_regression import ROOT, OUT, metrics
from phgeofuse.cache import atomic_json
from phgeofuse.dual_fusion import DualFusion
from phgeofuse.io import read_manifest
from phgeofuse.retrieval import RetrievalStore, record_key


def main():
    bundle = OUT / 'dual_candidate_float64'
    records = [r for r in read_manifest(ROOT / 'artifacts/phgeofuse/manifest.csv') if r.split == 'train']
    keys = [record_key(r) for r in records]
    with (OUT / 'baseline_train.csv').open() as f:
        rows = {r['key']: float(r['prediction']) for r in csv.DictReader(f)}
    assert set(rows) == set(keys)
    features = []
    for source in ['esm1v_masked/features.npz', 'esm2_masked/features.npz']:
        with np.load(OUT / source) as f:
            mapping = {str(k): i for i, k in enumerate(f['keys'])}
            idx = [mapping[k] for k in keys]
            features.extend([f['mean'][idx], f['std'][idx]])
    store = RetrievalStore.load(ROOT / 'artifacts/phgeofuse/retrieval.pt')
    retrieval = np.array([store.features(k).numpy() for k in keys])
    low = ~((retrieval[:, 4] >= .2) & (retrieval[:, 9] >= .8) & (retrieval[:, 10] >= .8))
    y = np.array([r.ph_opt for r in records])
    pred = DualFusion(bundle).predict([rows[k] for k in keys], *features, retrieval, [r.sequence for r in records])
    train = metrics(y, pred['prediction'], low)
    val = next(r['validation'] for r in json.loads((bundle / 'verification.json').read_text())['runs']
               if r['seed'] == 42)
    nested = json.loads((OUT / 'dual_nested/results.json').read_text())
    result = {'seed': 42, 'in_sample_train': train, 'validation': val,
              'in_sample_gap': val['rmse'] - train['rmse'],
              'nested_dual_expert': nested['strict_nested']['phweighted50'],
              'warning': 'Full blend is not nested-crossvalidated. Train predictions use in-sample '
              'sequence experts and normal retrieval. The dual meta-model was fitted with OOF base '
              'features, so this in-sample gap also reflects feature distribution differences and '
              'must not be used alone to claim overfitting resolved.'}
    atomic_json(bundle / 'overfit_diagnostic.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
