"""Label-free pair network and reference-label transfer.

Labels belong to the training reference panel, never to query inputs. All
aggregation is float64; the small neural network uses float32.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn


def finite_array(value, ndim=None):
    value = np.asarray(value, dtype=np.float64)
    if (ndim is not None and value.ndim != ndim) or not np.isfinite(value).all():
        raise ValueError("invalid dimensions or nonfinite input")
    return value


def ph_bins(y):
    y = finite_array(y, 1)
    if ((y < 0) | (y > 14)).any():
        raise ValueError("this PHOPT protocol supports labels in [0, 14]")
    return np.minimum(np.floor(y).astype(int), 13)


class DeltaNetwork(nn.Module):
    def __init__(self, input_dim=5145, width=64, dropout=0.1, kind="pair"):
        super().__init__()
        if kind not in {"pair", "additive", "absolute"}:
            raise ValueError("unknown network kind")
        self.spec = dict(input_dim=input_dim, width=width, dropout=dropout, kind=kind)
        self.kind = kind
        self.project = nn.Sequential(nn.Linear(input_dim, width), nn.LayerNorm(width), nn.GELU())
        self.pair_hidden = nn.Sequential(nn.Linear(4 * width, width), nn.GELU()) if kind == "pair" else None
        self.scalar_hidden = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU()) if kind != "pair" else None
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(width if kind == "pair" else 4 * width, 1, bias=kind == "absolute")

    def encode(self, x):
        return self.project(x)

    def difference(self, q, r):
        if self.kind == "absolute":
            raise ValueError("absolute-regression control has no pair prediction")
        if self.kind == "pair":
            a = torch.cat((q, r, q - r, q * r), -1)
            b = torch.cat((r, q, r - q, q * r), -1)
            # Share the dropout mask between directions. In particular d(x,x)
            # is exactly zero even while training. Antisymmetry is deterministic
            # at inference (or for the same stochastic mask during training).
            h = (self.pair_hidden(a) - self.pair_hidden(b)) * 0.5
        else:
            h = self.scalar_hidden(q) - self.scalar_hidden(r)
        return self.output(self.dropout(h)).squeeze(-1)

    def forward(self, query, reference=None):
        q = self.encode(query)
        if self.kind == "absolute":
            return self.output(self.dropout(self.scalar_hidden(q))).squeeze(-1)
        if reference is None:
            raise ValueError("reference features are required")
        return self.difference(q, self.encode(reference))


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x):
        x = finite_array(x, 2)
        if not len(x):
            raise ValueError("empty standardizer training set")
        std = x.std(0)
        return cls(x.mean(0), np.where(std > 1e-8, std, 1.0))

    def transform(self, x):
        x = finite_array(x, 2)
        if x.shape[1:] != self.mean.shape:
            raise ValueError("feature schema mismatch")
        return ((x - self.mean) / self.scale).astype(np.float32)


def select_panel(embedding, labels, groups, keys, per_bin=8, balanced=True):
    """Deterministic farthest-point representatives, one family per bin.

    Distances use frozen embeddings, not trained projections or chemistry.
    Lexical keys break every tie, including the first representative.
    """
    x = finite_array(embedding, 2)
    labels = finite_array(labels, 1)
    groups, keys = np.asarray(groups).astype(str), np.asarray(keys).astype(str)
    if not (len(x) == len(labels) == len(groups) == len(keys)) or len(set(keys)) != len(keys):
        raise ValueError("panel rows must have aligned, unique keys")
    if per_bin < 1:
        raise ValueError("per_bin must be positive")
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    bins = ph_bins(labels)
    bin_values = np.unique(bins)
    capacity = {b: len(set(groups[bins == b])) for b in bin_values}
    quotas = {b: min(per_bin, capacity[b]) for b in bin_values}
    if not balanced and len(labels):
        total = sum(quotas.values())
        desired = {b: total * float((bins == b).mean()) for b in bin_values}
        quotas = {b: min(capacity[b], int(np.floor(desired[b]))) for b in bin_values}
        while sum(quotas.values()) < total:
            eligible = [b for b in bin_values if quotas[b] < capacity[b]]
            b = max(eligible, key=lambda k: desired[k] - quotas[k])
            quotas[b] += 1
    selected = []
    for b in bin_values:
        candidates = sorted(np.flatnonzero(bins == b), key=lambda i: keys[i])
        chosen, used = [], set()
        distance = np.full(len(x), np.inf)
        while len(chosen) < quotas[b]:
            available = [i for i in candidates if groups[i] not in used]
            if not available:
                break
            j = available[0] if not chosen else max(available, key=lambda i: distance[i])
            chosen.append(j)
            used.add(groups[j])
            distance = np.minimum(distance, np.maximum(0., 1. - x @ x[j]))
        selected.extend(chosen)
    selected = np.asarray(selected, dtype=int)
    weights = np.zeros(len(selected), dtype=float)
    for b in np.unique(bins):
        m = bins[selected] == b
        mass = 1. if balanced else float(m.sum())
        weights[m] = mass / max(1, m.sum())
    if len(weights):
        weights /= weights.sum()
    return selected, weights


def weighted_median(values, weights):
    values = finite_array(values, 2)
    w = finite_array(weights)
    w = np.broadcast_to(w, values.shape)
    if (w < 0).any() or (w.sum(1) <= 0).any():
        raise ValueError("positive reference weight required for each query")
    order = np.argsort(values, axis=1, kind="stable")
    v, w = np.take_along_axis(values, order, 1), np.take_along_axis(w, order, 1)
    positions = (np.cumsum(w, 1) >= w.sum(1, keepdims=True) * .5).argmax(1)
    return v[np.arange(len(v)), positions]


def huber_location(values, weights, delta=1.0):
    values = finite_array(values, 2)
    weights = finite_array(weights)
    weights = np.broadcast_to(weights, values.shape)
    if not values.shape[1] or delta <= 0 or (weights < 0).any() or (weights.sum(1) <= 0).any():
        raise ValueError("invalid Huber references")
    low = np.min(np.where(weights > 0, values, np.inf), 1)
    high = np.max(np.where(weights > 0, values, -np.inf), 1)
    for _ in range(56):
        middle = (low + high) * .5
        score = (weights * np.clip(values - middle[:, None], -delta, delta)).sum(1)
        low = np.where(score > 1e-14, middle, low)
        high = np.where(score < -1e-14, middle, high)
        zero = np.abs(score) <= 1e-14
        low, high = np.where(zero, middle, low), np.where(zero, middle, high)
    return (low + high) * .5


def blend_predictions(baseline, transfer, dispersion, strength, valid=None, consistency=True):
    b, t, d = [finite_array(v, 1) for v in (baseline, transfer, dispersion)]
    if not (b.shape == t.shape == d.shape) or not 0 <= strength <= 1 or (d < 0).any():
        raise ValueError("invalid blend inputs")
    if ((b < 0) | (b > 14)).any():
        raise ValueError("baseline outside [0,14]; cannot guarantee exact zero-strength parity")
    valid = np.ones(len(b), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if valid.shape != b.shape:
        raise ValueError("valid-reference mask shape mismatch")
    agreement = 1. / (1. + d ** 2) if consistency else np.ones(len(b))
    correction = strength * agreement * np.clip(t - b, -4., 4.)
    correction = np.where(valid, correction, 0.)
    result = np.clip(b + correction, 0., 14.)
    return {"prediction": result, "baseline_prediction": b.copy(), "transfer_prediction": t,
            "dispersion": d, "agreement": agreement, "correction": result - b,
            "fallback": ~valid,
            "fallback_reason": np.where(valid, "", "fewer_than_3_distinct_reference_families")}


class ReferencePredictor:
    def __init__(self, network, scaler, reference_features, labels, groups, keys, weights, device="cpu", balanced=True):
        self.network = network.to(device).eval()
        self.scaler, self.device = scaler, torch.device(device)
        self.labels = finite_array(labels, 1)
        self.groups = np.asarray(groups).astype(str)
        self.keys = np.asarray(keys).astype(str)
        self.weights = finite_array(weights, 1)
        self.balanced = bool(balanced)
        self.features = finite_array(reference_features, 2)
        n = len(self.labels)
        if any(len(v) != n for v in (self.features, self.groups, self.keys, self.weights)):
            raise ValueError("reference panel shape mismatch")
        if len(set(self.keys)) != n or (self.weights < 0).any() or (n and self.weights.sum() <= 0):
            raise ValueError("invalid panel keys or weights")

    @torch.inference_mode()
    def transfer(self, features, query_keys=None, query_groups=None, batch_size=128):
        x = self.scaler.transform(features)
        n, nr = len(x), len(self.labels)
        keys = np.full(n, "") if query_keys is None else np.asarray(query_keys).astype(str)
        groups = np.full(n, "") if query_groups is None else np.asarray(query_groups).astype(str)
        if len(keys) != n or len(groups) != n or batch_size < 1:
            raise ValueError("invalid query metadata")
        if self.network.kind == "absolute":
            p = [self.network(torch.as_tensor(x[i:i+batch_size], device=self.device)).cpu().numpy()
                 for i in range(0, n, batch_size)]
            return np.concatenate(p) if p else np.empty(0), np.zeros(n), np.ones(n, bool)
        result, dispersion, valid = np.zeros(n), np.zeros(n), np.zeros(n, bool)
        if nr == 0:
            return result, dispersion, valid
        h_reference = self.network.encode(torch.as_tensor(self.scaler.transform(self.features), device=self.device))
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            h = self.network.encode(torch.as_tensor(x[start:end], device=self.device))
            q = h[:, None, :].expand(-1, nr, -1)
            r = h_reference[None, :, :].expand(len(h), -1, -1)
            estimates = self.network.difference(q, r).cpu().numpy().astype(float) + self.labels
            weights = np.broadcast_to(self.weights, estimates.shape).copy()
            for i, (key, group) in enumerate(zip(keys[start:end], groups[start:end])):
                weights[i, self.keys == key] = 0.
                if group:
                    weights[i, self.groups == group] = 0.
                if self.balanced:
                    bins = ph_bins(self.labels)
                    for b in np.unique(bins):
                        mask = bins == b
                        remaining = weights[i, mask].sum()
                        if remaining > 0:
                            weights[i, mask] *= self.weights[mask].sum() / remaining
                valid[start+i] = len(set(self.groups[weights[i] > 0])) >= 3
            keep = valid[start:end]
            if keep.any():
                v, w = estimates[keep], weights[keep]
                result[start:end][keep] = huber_location(v, w)
                median = weighted_median(v, w)
                dispersion[start:end][keep] = weighted_median(np.abs(v - median[:, None]), w)
        return result, dispersion, valid

    def predict(self, features, baseline, strength, query_keys=None, query_groups=None, consistency=True):
        t, d, valid = self.transfer(features, query_keys, query_groups)
        return blend_predictions(baseline, t, d, strength, valid, consistency)
