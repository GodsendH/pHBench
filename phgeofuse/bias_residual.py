"""Low-capacity residual regression with explicit training-group bias penalties."""
import numpy as np
from sklearn.preprocessing import StandardScaler


class BiasConstrainedResidual:
    def __init__(self, alpha=100., bias_penalty=0.):
        if alpha <= 0 or bias_penalty < 0:
            raise ValueError('positive regularization and nonnegative bias penalty required')
        self.alpha = alpha
        self.bias_penalty = bias_penalty

    @staticmethod
    def features(meta):
        meta = np.asarray(meta, dtype=np.float64)
        if meta.ndim != 2 or meta.shape[1] != 41 or not np.isfinite(meta).all():
            raise ValueError('expected 15 retrieval + sequence + 25 chemistry features')
        # Missing retrieval predictions use the independent sequence estimate.
        p = np.column_stack([np.where(meta[:, 7] > 0, meta[:, 0], meta[:, 15]),
                             np.where(meta[:, 8] > 0, meta[:, 1], meta[:, 15]), meta[:, 15]])
        centered = p - 7.
        return np.column_stack([meta, centered**2,
            (p[:, 0]-p[:, 2])**2, (p[:, 1]-p[:, 2])**2,
            centered[:, 2:3] * meta[:, 16:]])

    def fit(self, meta, target, sample_weight=None, *, labels):
        target = np.asarray(target, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)
        if target.shape != labels.shape or target.shape != (len(meta),):
            raise ValueError('label/target dimensions differ')
        if not np.isfinite(labels).all() or not np.isfinite(target).all():
            raise ValueError('nonfinite training labels')
        self.scaler = StandardScaler().fit(self.features(meta))
        z = self.scaler.transform(self.features(meta))
        w = np.ones(len(z)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if w.shape != target.shape or not np.isfinite(w).all() or (w <= 0).any():
            raise ValueError('invalid weights')
        # Explicit intercept avoids Ridge centering away the added group constraints.
        design = np.column_stack([np.ones(len(z)), z])
        design_rows = [design * np.sqrt(w)[:, None]]
        targets = [target * np.sqrt(w)]
        masks = [labels < 6, (labels >= 6) & (labels < 8), labels >= 8]
        for mask in masks:
            if mask.any() and self.bias_penalty > 0:
                scale = np.sqrt(w.sum() * self.bias_penalty / 3.)
                design_rows.append(scale * design[mask].mean(0, keepdims=True))
                targets.append(np.array([scale * target[mask].mean()]))
        # Leave intercept unpenalized; all standardized feature coefficients shrink.
        penalty = np.column_stack([np.zeros(z.shape[1]), np.eye(z.shape[1])])
        design_rows.append(np.sqrt(self.alpha) * penalty)
        targets.append(np.zeros(z.shape[1]))
        self.coefficients = np.linalg.lstsq(np.vstack(design_rows), np.concatenate(targets), rcond=None)[0]
        self.training_counts = [int(mask.sum()) for mask in masks]
        return self

    def predict(self, meta):
        z = self.scaler.transform(self.features(meta))
        return self.coefficients[0] + z @ self.coefficients[1:]
