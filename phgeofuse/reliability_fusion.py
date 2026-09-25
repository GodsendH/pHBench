"""Inference for frozen PHOPT sequence/retrieval reliability residual experts."""
import json
import hashlib
from pathlib import Path
import numpy as np
import joblib
from phgeofuse.robust_fusion import pool_features, chemistry_features


def reliability_inputs(retrieval, sequence):
    r = np.asarray(retrieval, dtype=float)
    s = np.asarray(sequence, dtype=float)
    if r.shape != (len(s), 15) or not np.isfinite(r).all() or not np.isfinite(s).all():
        raise ValueError('invalid sequence/retrieval shape or nonfinite values')
    features = r[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14]]
    experts = np.column_stack([s, r[:, 0], r[:, 1]])
    available = np.column_stack([np.ones(len(r), bool), r[:, 7:9] > 0])
    return features, experts, available


class ReliabilityFusion:
    def __init__(self, bundle):
        bundle = Path(bundle)
        self.config = json.loads((bundle/'model.json').read_text())
        for name, digest in self.config['file_sha256'].items():
            if hashlib.sha256((bundle/name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'checksum mismatch: {name}')
        self.sequence = joblib.load(bundle/'sequence.joblib')
        self.gate = joblib.load(bundle/'gate.joblib')
        self.residual = joblib.load(bundle/'residual.joblib')

    def predict(self, mean1, std1, mean2, std2, retrieval, sequences):
        x = np.column_stack([pool_features(mean1,std1,'mean_std'),
                             pool_features(mean2,std2,'mean_std')])
        if len(x) != len(sequences):
            raise ValueError('sequence and embedding counts differ')
        sequence = self.sequence.predict(x)
        r = np.asarray(retrieval,float)
        gate = self.gate.predict(*reliability_inputs(r,sequence))
        return gate + self.residual.predict(np.column_stack([r,sequence,chemistry_features(sequences)]))
