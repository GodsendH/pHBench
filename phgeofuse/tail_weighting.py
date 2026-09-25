"""Training-subset-only weights and residual targets for complete DualFusion."""
from __future__ import annotations

import numpy as np


def tail_weights(labels, strength=0.10, *, acid_max=4., alkaline_min=10., max_weight=10.):
    """Return mean-one weights for natural MSE plus equally weighted tail MSEs.

    Fit this function separately on each actual training subset. A weight cap
    reduces the common mixing strength instead of clipping individual weights.
    """
    y = np.asarray(labels, dtype=np.float64)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
        raise ValueError('labels must be a nonempty finite vector')
    if not np.isfinite(strength) or not 0 <= strength < 1:
        raise ValueError('strength must lie in [0, 1)')
    if not np.isfinite([acid_max, alkaline_min, max_weight]).all():
        raise ValueError('thresholds and maximum weight must be finite')
    if acid_max >= alkaline_min or max_weight < 1:
        raise ValueError('invalid tail thresholds or maximum weight')
    acid, alkaline = y <= acid_max, y >= alkaline_min
    effective = float(strength)
    if strength:
        if not acid.any() or not alkaline.any():
            raise ValueError('both tails must occur in the actual training subset')
        extra = .5 * (acid / acid.mean() + alkaline / alkaline.mean()) - 1.
        if extra.max() > 0:
            effective = min(effective, (max_weight - 1.) / float(extra.max()))
        weights = 1. + effective * extra
    else:
        weights = np.ones(len(y))
    if not np.isclose(weights.mean(), 1., rtol=0, atol=1e-12):
        raise ValueError('tail weights must have mean one')
    masks = {'extreme_acid': acid, 'core': ~(acid | alkaline), 'extreme_alkaline': alkaline}
    metadata = dict(strength_requested=float(strength), strength_effective=effective,
                    acid_max=float(acid_max), alkaline_min=float(alkaline_min),
                    maximum_allowed=float(max_weight), count=len(y), mean=float(weights.mean()),
                    minimum=float(weights.min()), maximum=float(weights.max()),
                    effective_sample_size=float(weights.sum() ** 2 / np.dot(weights, weights)),
                    regions={name: dict(count=int(m.sum()), coefficient_mass=float(weights[m].sum()/weights.sum()))
                             for name, m in masks.items()})
    return weights, metadata


def residual_target(labels, anchor, robust=None, *, target='branch', dual_weight=.5):
    y, a = np.asarray(labels, dtype=float), np.asarray(anchor, dtype=float)
    if y.ndim != 1 or y.shape != a.shape or not np.isfinite(y).all() or not np.isfinite(a).all():
        raise ValueError('label and anchor vectors must be finite and aligned')
    if target == 'branch':
        return y - a
    if target != 'complete' or not np.isfinite(dual_weight) or not 0 < dual_weight <= 1:
        raise ValueError('invalid residual target or dual weight')
    r = np.asarray(robust, dtype=float)
    if r.shape != y.shape or not np.isfinite(r).all():
        raise ValueError('complete target requires aligned cross-fit robust predictions')
    return (y - (1. - dual_weight) * r) / dual_weight - a


def crossfit_rows(folds, outer, fetch):
    """Assemble training inputs using only models excluding outer and row fold.

    fetch(excluded) returns global row indices and aligned prediction arrays.
    No labels are read here. This function makes the two-exclusion contract
    explicit and keeps excluded rows unavailable to downstream training.
    """
    folds = np.asarray(folds)
    if folds.ndim != 1 or outer not in set(folds):
        raise ValueError('invalid outer fold')
    fit = np.flatnonzero(folds != outer)
    fields = {}
    written = np.zeros(len(folds), dtype=bool)
    for inner in sorted(set(folds) - {outer}):
        query, values = fetch(tuple(sorted((int(outer), int(inner)))))
        query = np.asarray(query)
        if not np.array_equal(query, np.flatnonzero(np.isin(folds, [outer, inner]))):
            raise ValueError('cache query rows differ from excluded folds')
        keep = folds[query] == inner
        rows = query[keep]
        if written[rows].any():
            raise ValueError('duplicate cross-fit rows')
        for name in ('retrieval', 'ridge', 'robust'):
            value = np.asarray(values[name], dtype=float)
            if len(value) != len(query) or not np.isfinite(value).all():
                raise ValueError('invalid cross-fit prediction values')
            if name not in fields:
                fields[name] = np.full((len(folds), *value.shape[1:]), np.nan)
            fields[name][rows] = value[keep]
        written[rows] = True
    if not written[fit].all() or written[folds == outer].any():
        raise ValueError('cross-fit training coverage differs')
    return fit, {k: v[fit] for k, v in fields.items()}
