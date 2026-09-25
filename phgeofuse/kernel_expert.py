"""Inference for a frozen-embedding, regularized PHGeoFuse regression expert."""
from pathlib import Path
import numpy as np


class KernelExpert:
    def __init__(self, training_vectors, dual, mean_label, gamma, distance_scale):
        self.training_vectors = np.asarray(training_vectors, dtype=np.float64)
        self.dual = np.asarray(dual, dtype=np.float64)
        self.mean_label = float(mean_label)
        self.gamma = float(gamma)
        self.distance_scale = float(distance_scale)
        if self.training_vectors.ndim != 2 or self.dual.shape != (len(self.training_vectors),):
            raise ValueError('invalid kernel checkpoint dimensions')
        if not np.isfinite(self.training_vectors).all() or not np.isfinite(self.dual).all():
            raise ValueError('nonfinite kernel checkpoint')
        if self.gamma < 0 or self.distance_scale <= 0:
            raise ValueError('invalid kernel parameters')

    @classmethod
    def load(cls, source):
        with np.load(Path(source), allow_pickle=False) as data:
            return cls(data['x'], data['dual'], data['mu'], data['gamma'], data['scale'])

    def predict(self, embeddings, batch_size=256):
        x = np.asarray(embeddings, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.training_vectors.shape[1] or not np.isfinite(x).all():
            raise ValueError('embeddings must be a finite matrix matching checkpoint dimensions')
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        if (norm == 0).any():
            raise ValueError('zero embedding cannot be normalized')
        x = x / norm
        result = []
        for start in range(0, len(x), batch_size):
            dot = x[start:start + batch_size] @ self.training_vectors.T
            kernel = dot if self.gamma == 0 else np.exp(-self.gamma * np.maximum(2 - 2 * dot, 0) / self.distance_scale)
            result.append(self.mean_label + kernel @ self.dual)
        return np.concatenate(result) if result else np.empty(0)
