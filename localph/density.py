"""Smooth conditional pH density via regularized cosine moments.

This estimates a response distribution, not a calibrated confidence interval
or a mechanistic activity curve. Twenty-four output harmonics share the same
ridge projection. Gaussian spectral damping smooths the continuous label axis.
"""
import numpy as np
import torch


def cosine_targets(y, harmonics=24):
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all() or ((y < 0) | (y > 14)).any():
        raise ValueError("finite labels within pH 0..14 required")
    return np.cos(y[:, None] * np.arange(1, harmonics + 1)[None] * np.pi / 14)


class DensityRidge:
    def __init__(self, alpha, harmonics=24, device="cpu"):
        if alpha <= 0 or harmonics < 1:
            raise ValueError("positive regularization and harmonics required")
        self.alpha, self.harmonics, self.device = alpha, harmonics, device

    def fit(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        t = cosine_targets(y, self.harmonics)
        if x.ndim != 2 or len(x) != len(t) or not np.isfinite(x).all():
            raise ValueError("finite aligned feature matrix required")
        self.x_mean, self.target_mean = x.mean(0), t.mean(0)
        tx = torch.as_tensor(x - self.x_mean, device=self.device, dtype=torch.float64)
        tt = torch.as_tensor(t - self.target_mean, device=self.device, dtype=torch.float64)
        dual = len(x) < x.shape[1]
        gram = tx @ tx.T if dual else tx.T @ tx
        gram.diagonal().add_(self.alpha)
        rhs = tt if dual else tx.T @ tt
        solved = torch.cholesky_solve(rhs, torch.linalg.cholesky(gram))
        self.coef = (tx.T @ solved if dual else solved).cpu().numpy()
        return self

    def moments(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.x_mean) or not np.isfinite(x).all():
            raise ValueError("invalid query features")
        return (x - self.x_mean) @ self.coef + self.target_mean


def decode_density(moments, prior_moments, bandwidth, prior_power, decision):
    c = np.asarray(moments, dtype=float)
    prior = np.asarray(prior_moments, dtype=float)
    if (c.ndim != 2 or prior.shape != c.shape[1:] or not np.isfinite(c).all()
            or not np.isfinite(prior).all() or bandwidth <= 0 or prior_power < 0
            or decision not in ("mean", "mode")):
        raise ValueError("invalid continuous density inputs")
    grid = np.linspace(0, 14, 281)
    freq = np.arange(1, c.shape[1] + 1) * np.pi / 14
    basis = np.cos(freq[:, None] * grid[None]) * np.exp(-.5 * (bandwidth * freq[:, None]) ** 2)
    density = (1 + 2 * c @ basis) / 14
    marginal = (1 + 2 * prior @ basis) / 14
    # Truncation prevents an unconstrained least-squares density being negative.
    negative_mass = np.maximum(-density, 0).sum(1) * .05
    density = np.maximum(density, 0)
    marginal = np.maximum(marginal, .001)
    density /= marginal[None] ** prior_power
    integration = np.ones(len(grid))
    integration[[0, -1]] = .5
    mass = density * integration
    total = mass.sum(1)
    if (total <= 0).any():
        raise ValueError("no positive density mass")
    if decision == "mean":
        p = mass @ grid / total
    else:
        # Resolve exact maximum ties using proximity to the conditional mean.
        mean = mass @ grid / total
        maxima = np.isclose(density, density.max(1, keepdims=True), rtol=0, atol=1e-12)
        distance = np.where(maxima, abs(grid[None] - mean[:, None]), np.inf)
        p = grid[distance.argmin(1)]
    return p, negative_mass
