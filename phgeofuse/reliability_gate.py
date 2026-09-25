"""Low-capacity convex gate fitted on independently cross-fitted experts."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax


class ReliabilityGate:
    def __init__(self, penalty=.01):
        self.penalty = penalty

    @staticmethod
    def objective(flat, x, experts, available, y, penalty):
        matrix = flat.reshape(x.shape[1], experts.shape[1])
        logits = x @ matrix
        weights = softmax(np.where(available, logits, -np.inf), axis=1)
        pred = np.sum(weights * experts, axis=1)
        residual = pred - y
        loss = np.mean(residual**2) + penalty * np.sum(matrix[1:]**2)
        dlogits = (2. / len(y)) * residual[:, None] * weights * (experts - pred[:, None])
        gradient = x.T @ dlogits
        gradient[1:] += 2 * penalty * matrix[1:]
        return float(loss), gradient.ravel()

    def _design(self, features):
        z = np.clip((np.asarray(features, float) - self.mean_) / self.std_, -6., 6.)
        return np.column_stack([np.ones(len(z)), z])

    @staticmethod
    def _check(experts, available):
        if experts.shape != available.shape or experts.ndim != 2:
            raise ValueError('expert/availability shape mismatch')
        if not np.isfinite(experts).all() or not available.any(axis=1).all():
            raise ValueError('nonfinite expert or no available prediction')

    def fit(self, features, experts, available, y):
        features = np.asarray(features, float)
        experts, available = np.asarray(experts, float), np.asarray(available, bool)
        self._check(experts, available)
        if not np.isfinite(features).all() or not np.isfinite(y).all():
            raise ValueError('nonfinite training values')
        self.mean_ = features.mean(axis=0)
        self.std_ = np.maximum(features.std(axis=0), 1e-8)
        x = self._design(features)
        result = minimize(self.objective, np.zeros(x.shape[1] * experts.shape[1]),
            args=(x, experts, available, np.asarray(y, float), self.penalty),
            jac=True, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-11})
        if not result.success:
            raise RuntimeError(f'gate optimization failed: {result.message}')
        self.matrix_ = result.x.reshape(x.shape[1], experts.shape[1])
        self.iterations_ = int(result.nit)
        return self

    def weights(self, features, available):
        return softmax(np.where(available, self._design(features) @ self.matrix_, -np.inf), axis=1)

    def predict(self, features, experts, available):
        experts, available = np.asarray(experts, float), np.asarray(available, bool)
        self._check(experts, available)
        return np.sum(self.weights(features, available) * experts, axis=1)
