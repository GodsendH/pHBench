"""Source-aware ridge with a regularized external-source nuisance offset.

Predictions target the PHOPT measurement distribution (source indicator zero).
External labels inform shared sequence/context coefficients, while an external
intercept absorbs a constant source offset. No external labels are consulted at
inference and no query label chooses a source or pH branch.
"""
import numpy as np
import torch
from .features import SiteScaler
from .kernel import training_weights


class SourceContextRidge:
    def __init__(self, site_weight, alpha, power, external_weight, device="cpu"):
        self.site_weight, self.alpha, self.power = site_weight, alpha, power
        self.external_weight, self.device = external_weight, device

    def features(self, global_x, site_x, source=0):
        return np.column_stack([global_x, np.sqrt(self.site_weight) * self.scaler.transform(site_x),
                                np.full(len(global_x), source)])

    def fit(self, global_x, site_x, y, external_x, external_site, external_y):
        if self.alpha <= 0 or min(self.site_weight, self.power, self.external_weight) < 0:
            raise ValueError("invalid recipe")
        self.scaler = SiteScaler().fit(site_x)
        x = self.features(global_x, site_x)
        y = np.asarray(y, dtype=float)
        w = training_weights(y, self.power)
        if self.external_weight:
            ex = self.features(external_x, external_site, 1)
            ey = np.asarray(external_y, dtype=float)
            ew = training_weights(ey, self.power) * self.external_weight
            x, y, w = np.concatenate([x, ex]), np.concatenate([y, ey]), np.concatenate([w, ew])
        if y.shape != (len(x),) or not np.isfinite(y).all():
            raise ValueError("invalid labels")
        self.x_mean, self.y_mean = np.average(x, weights=w, axis=0), np.average(y, weights=w)
        tx = torch.as_tensor((x - self.x_mean) * np.sqrt(w[:, None]), dtype=torch.float64, device=self.device)
        ty = torch.as_tensor((y - self.y_mean) * np.sqrt(w), dtype=torch.float64, device=self.device)
        dual = len(x) < x.shape[1]
        gram = tx @ tx.T if dual else tx.T @ tx
        gram.diagonal().add_(self.alpha)
        rhs = ty if dual else tx.T @ ty
        sol = torch.cholesky_solve(rhs[:, None], torch.linalg.cholesky(gram))[:, 0]
        self.coef = (tx.T @ sol if dual else sol).cpu().numpy()
        return self

    def predict(self, global_x, site_x):
        return (self.features(global_x, site_x) - self.x_mean) @ self.coef + self.y_mean
