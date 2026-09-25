"""Inference for the PHOPT dual-encoder research candidate; no labels in predict."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from .robust_fusion import RobustFusion, chemistry_features, pool_features


def retrieval_sequence_anchor(retrieval, sequence):
    retrieval = np.asarray(retrieval, dtype=np.float64)
    sequence = np.asarray(sequence, dtype=np.float64)
    if retrieval.shape != (len(sequence), 15) or sequence.ndim != 1:
        raise ValueError('invalid anchor dimensions')
    if not np.isfinite(retrieval).all() or not np.isfinite(sequence).all():
        raise ValueError('nonfinite anchor inputs')
    available = retrieval[:, 7:9]
    if not np.isin(available, [0., 1.]).all():
        raise ValueError('retrieval availability must be binary')
    return ((retrieval[:, :2] * available).sum(1) + sequence) / (available.sum(1) + 1)


class DualFusion:
    def __init__(self, bundle):
        bundle = Path(bundle)
        self.config = json.loads((bundle / 'model.json').read_text())
        for name, expected in self.config['file_hashes'].items():
            if hashlib.sha256((bundle / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'model hash mismatch: {name}')
        self.robust = RobustFusion(bundle / 'robust_v1')
        self.sequence = joblib.load(bundle / 'sequence.joblib')
        self.residual = joblib.load(bundle / 'residual.joblib')
        self.dual_weight = float(self.config['dual_weight'])
        if not 0 <= self.dual_weight <= 1:
            raise ValueError('invalid dual weight')

    def predict(self, baseline, esm1v_mean, esm1v_std, esm2_mean, esm2_std,
                retrieval, sequences):
        retrieval = np.asarray(retrieval, dtype=np.float64)
        if retrieval.shape != (len(sequences), 15) or not np.isfinite(retrieval).all():
            raise ValueError('invalid retrieval features')
        x = np.column_stack([pool_features(esm1v_mean, esm1v_std, 'mean_std'),
                             pool_features(esm2_mean, esm2_std, 'mean_std')])
        if len(x) != len(sequences):
            raise ValueError('sequence feature count differs')
        seq = self.sequence.predict(x)
        anchor = retrieval_sequence_anchor(retrieval, seq)
        meta = np.column_stack([retrieval, seq, chemistry_features(sequences)])
        dual = anchor + self.residual.predict(meta)
        robust = self.robust.predict(baseline, esm2_mean, esm2_std, retrieval, sequences)['prediction']
        return {'prediction': (1-self.dual_weight)*robust + self.dual_weight*dual,
                'robust_v1_prediction': robust, 'dual_prediction': dual,
                'dual_sequence_prediction': seq}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['bundle', 'manifest', 'retrieval', 'baseline-predictions',
                 'esm1v-features', 'esm2-features', 'output']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--split', choices=['train', 'validation', 'test'], required=True)
    args = parser.parse_args()
    from .io import read_manifest
    from .retrieval import RetrievalStore, record_key
    records = [r for r in read_manifest(args.manifest) if r.split == args.split]
    keys = [record_key(r) for r in records]
    if not keys:
        raise ValueError('empty split')
    with open(args.baseline_predictions) as handle:
        rows = list(csv.DictReader(handle))
    mapping = {r['key']: float(r['prediction']) for r in rows}
    if len(mapping) != len(rows) or set(mapping) != set(keys):
        raise ValueError('baseline keys differ')
    features = []
    for source in [args.esm1v_features, args.esm2_features]:
        with np.load(source, allow_pickle=False) as f:
            order = {str(k): i for i, k in enumerate(f['keys'])}
            if len(order) != len(f['keys']):
                raise ValueError('duplicate feature keys')
            indices = [order[k] for k in keys]
            features.extend([f['mean'][indices], f['std'][indices]])
    store = RetrievalStore.load(args.retrieval)
    retrieval = np.array([store.features(k).numpy() for k in keys])
    output = DualFusion(args.bundle).predict([mapping[k] for k in keys], *features,
                                            retrieval, [r.sequence for r in records])
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['key', *output])
        writer.writeheader()
        for i, k in enumerate(keys):
            writer.writerow({'key': k, **{n: float(p[i]) for n, p in output.items()}})


if __name__ == '__main__':
    main()
