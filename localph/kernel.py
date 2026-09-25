"""Small, regularized two-view predictor with weighted centering.

Dual ridge is solved by Cholesky; no task-trained token attention or reference
label attention is used. The two kernels are inner products of global sequence
features and training-standardized ionizable context sketches respectively.
"""
from dataclasses import dataclass
import numpy as np
import torch
from .features import SiteScaler


def training_weights(y, power):
    y = np.asarray(y, dtype=float)
    bins = np.clip(np.floor(y).astype(int), 0, 13)
    counts = np.bincount(bins, minlength=14)
    weights = np.maximum(counts[bins], 1).astype(float) ** (-power)
    weights /= weights.mean()
    return np.minimum(weights, 3.)


@dataclass
class ContextRidge:
    site_weight: float
    alpha: float
    power: float
    device: str = "cpu"

    def features(self, global_x, site_x):
        return np.column_stack([global_x, self.site_weight ** .5 * self.scaler.transform(site_x)])

    def fit(self, global_x, site_x, y):
        if self.site_weight < 0 or self.alpha <= 0 or self.power < 0:
            raise ValueError("invalid regularization/weight")
        y = np.asarray(y, dtype=np.float64)
        self.scaler = SiteScaler().fit(site_x)
        x = self.features(global_x, site_x)
        if y.shape != (len(x),) or not np.isfinite(y).all():
            raise ValueError("aligned finite labels required")
        w = training_weights(y, self.power)
        self.x_mean = np.average(x, weights=w, axis=0)
        self.y_mean = np.average(y, weights=w)
        tx = torch.as_tensor((x - self.x_mean) * w[:, None] ** .5,
                             dtype=torch.float64, device=self.device)
        ty = torch.as_tensor((y - self.y_mean) * w ** .5,
                             dtype=torch.float64, device=self.device)
        # Primal solve is smaller for toy tests; normal data use the dual.
        dual = len(x) < x.shape[1]
        gram = tx @ tx.T if dual else tx.T @ tx
        gram.diagonal().add_(self.alpha)
        rhs = ty if dual else tx.T @ ty
        solved = torch.cholesky_solve(rhs[:, None], torch.linalg.cholesky(gram))[:, 0]
        self.coef = (tx.T @ solved if dual else solved).cpu().numpy()
        return self

    def predict(self, global_x, site_x):
        return (self.features(global_x, site_x) - self.x_mean) @ self.coef + self.y_mean
