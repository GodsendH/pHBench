"""Training-only smooth rarity weights and consistent weighted label priors.

The weighted objective allocates gradients to scarce labels. Prediction still
uses the natural training prior, with the same predeclared decoding choices.
No query labels or external labels enter either training prior.
"""
import numpy as np
import torch
import torch.nn.functional as F


def smooth_rarity_weights(labels, sigma=.5, cap=8.):
    y = np.asarray(labels, dtype=np.float64)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all() or ((y < 0) | (y > 14)).any():
        raise ValueError("finite training pH labels in [0, 14] required")
    # Blocked leave-in kernel density: the self term stabilizes isolated labels.
    density = np.empty(len(y))
    for start in range(0, len(y), 256):
        delta = (y[start:start + 256, None] - y[None, :]) / sigma
        density[start:start + 256] = np.exp(-.5 * delta ** 2).mean(1)
    weights = density ** -.5
    weights /= weights.mean()
    weights = np.minimum(weights, cap)
    weights /= weights.mean()
    return weights.astype(np.float32)


def weighted_label_prior(labels, weights, grid, sigma=.5):
    y = torch.as_tensor(labels, device=grid.device, dtype=torch.float32)
    w = torch.as_tensor(weights, device=grid.device, dtype=torch.float32)
    if y.ndim != 1 or w.shape != y.shape or not len(y) or not torch.isfinite(w).all() or (w <= 0).any():
        raise ValueError("positive aligned sample weights required")
    soft = torch.exp(-.5 * ((grid[None] - y[:, None]) / sigma) ** 2)
    soft /= soft.sum(1, keepdim=True)
    prior = (soft * w[:, None]).sum(0) / w.sum()
    prior = prior.clamp_min(1e-4)
    return prior / prior.sum()


def rarity_loss(output, labels, weights, grid, training_prior, kind):
    if kind == "direct":
        return (weights * (output["prediction"] - labels).square()).mean()
    soft = torch.exp(-.5 * ((grid[None] - labels[:, None]) / .5) ** 2)
    soft /= soft.sum(1, keepdim=True)
    logits = output["logits"] + training_prior.log()[None]
    nll = -(soft * logits.log_softmax(-1)).sum(1)
    mean = logits.softmax(-1) @ grid
    curvature = output["logits"][:, 2:] - 2 * output["logits"][:, 1:-1] + output["logits"][:, :-2]
    return (weights * (nll + .25 * (mean - labels).square())).mean() + .01 * curvature.square().mean()
