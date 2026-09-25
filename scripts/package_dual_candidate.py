"""Freeze an exploratory candidate and verify production prediction parity on validation."""
import csv
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from develop_phgeofuse_regression import ROOT, OUT, metrics
from phgeofuse.cache import atomic_json
from phgeofuse.io import read_manifest
from phgeofuse.retrieval import RetrievalStore, record_key
from phgeofuse.dual_fusion import DualFusion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT / 'dual_candidate_float64')
    args = parser.parse_args()
    out = args.output
    results = json.loads((OUT / 'dual_nested/fusion_validation.json').read_text())['results']
    selected = next(r for r in results if r['passes_v1_constraints'])
    if (out / 'model.json').exists():
        raise FileExistsError('refusing to overwrite frozen research candidate')
    out.mkdir(exist_ok=True)
    shutil.copytree(OUT / 'frozen_model', out / 'robust_v1')
    shutil.copy2(OUT / 'dual_nested/sequence.joblib', out / 'sequence.joblib')
    shutil.copy2(OUT / 'dual_nested' / (selected['dual_recipe'] + '.joblib'), out / 'residual.joblib')
    hashfile = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    config = {'architecture': 'robust v1 plus OOF-trained ESM1v/ESM2 retrieval-chemistry expert',
        'status': 'frozen research candidate; not confirmatory test validated',
        'dataset': 'PHOPT', 'train': 7124, 'validation': 760, 'test_used_for_selection': False,
        'dual_recipe': selected['dual_recipe'], 'dual_weight': selected['dual_weight'],
        'selection': selected, 'encoder_order': ['ESM1v', 'ESM2'],
        'pooling': 'each mean/std vector L2-normalized separately',
        'anchor_arithmetic': 'float64 via phgeofuse.dual_fusion.retrieval_sequence_anchor',
        'sequence': {'alpha': .2, 'sample_weighting': 'none'},
        'meta_training': 'strict-group out-of-fold base predictions; 7 leaves, 50 iterations, '
                         'min leaf 80, l2 30, lr .05, seed42, pH frequency power .25',
        'esm1v_provenance': json.loads((OUT / 'esm1v_masked/provenance.json').read_text()),
        'esm2_provenance': json.loads((OUT / 'frozen_model/model.json').read_text())['feature_provenance'],
        'reproduce_fit': 'python scripts/evaluate_dual_nested.py',
        'nested_scope': 'Nested evaluation covers the dual expert alone. The blend with the '
                        'original PHGeoFuse-derived v1 has not been retrained in nested outer folds.',
        'file_hashes': {str(p.relative_to(out)): hashfile(p) for p in out.rglob('*') if p.is_file()}}
    atomic_json(out / 'model.json', config)
    records = [r for r in read_manifest(ROOT / 'artifacts/phgeofuse/manifest.csv') if r.split == 'validation']
    keys = [record_key(r) for r in records]
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
    model = DualFusion(out)
    expected_dual = np.load(OUT / 'dual_nested' / (selected['dual_recipe'] + '.validation.npy'))
    audit = []
    for seed in [0, 1, 2, 3, 42]:
        filename = 'baseline_validation.csv' if seed == 42 else f'baseline_seed{seed}_validation.csv'
        with (OUT / filename).open() as f:
            rows = {r['key']: float(r['prediction']) for r in csv.DictReader(f)}
        assert set(rows) == set(keys)
        pred = model.predict([rows[k] for k in keys], *features, retrieval, [r.sequence for r in records])
        maxdiff = float(np.max(abs(pred['dual_prediction'] - expected_dual)))
        assert maxdiff < 1e-7, maxdiff
        with (out / f'seed{seed}_validation.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['key', 'label', *pred])
            writer.writeheader()
            for i, k in enumerate(keys):
                writer.writerow({'key': k, 'label': float(y[i]), **{n: float(p[i]) for n, p in pred.items()}})
        audit.append({'seed': seed, 'max_prediction_difference': maxdiff,
                      'validation': metrics(y, pred['prediction'], low)})
    atomic_json(out / 'verification.json', {'model_sha256': hashfile(out / 'model.json'),
        'tolerance': 1e-7, 'runs': audit})
    print(json.dumps({'bundle': str(out), 'selected': selected, 'verified_seeds': len(audit),
        'max_prediction_difference': max(r['max_prediction_difference'] for r in audit)}, indent=2))


if __name__ == '__main__':
    main()
