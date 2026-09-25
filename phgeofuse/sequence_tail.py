"""Nonlinear sequence experts for tail-priority complete fusion.

Only ``fit_expert`` accepts labels. All transforms and prediction paths are
label-free; full-model bundles carry the training feature reference explicitly.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVR

from .robust_fusion import pool_features
from .tail_priority import TailPriorityFusion, priority_weights


def fit_transform(features):
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('finite nonempty training feature matrix required')
    state = dict(mean=x.mean(0), std=x.std(0), gamma=1. / x.shape[1])
    return transform(x, state), state


def transform(features, state):
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[1:] != state['mean'].shape or not np.isfinite(x).all():
        raise ValueError('feature dimensions or finite values differ')
    return (x-state['mean'])/(state['std']+1e-8)


def fit_expert(kernel, labels, guide, recipe):
    y, guide = np.asarray(labels, float), np.asarray(guide, float)
    if kernel.shape != (len(y), len(y)) or y.shape != guide.shape:
        raise ValueError('kernel/label/guide alignment differs')
    weights, _, info = priority_weights(y, guide, recipe['acid_mass'], recipe['alkaline_mass'])
    target = y-guide if recipe['target'] == 'residual' else y
    if recipe['method'] == 'svr':
        model = SVR(kernel='precomputed', C=recipe['regularization'], epsilon=.1,
                    tol=.001, cache_size=1024, max_iter=-1).fit(kernel, target, sample_weight=weights)
        if model.fit_status_ != 0:
            raise ValueError('SVR did not converge')
    elif recipe['method'] == 'krr':
        model = KernelRidge(kernel='precomputed', alpha=recipe['regularization']).fit(
            kernel.copy(), target, sample_weight=weights)
    else:
        raise ValueError('unknown expert method')
    return model, weights, target, info


def complete_expert(model_prediction, original, target):
    if target not in ['direct', 'residual']:
        raise ValueError('unknown target')
    p, old = np.asarray(model_prediction, float), np.asarray(original, float)
    if p.shape != old.shape or not np.isfinite([p, old]).all():
        raise ValueError('prediction alignment differs')
    return p+old if target == 'residual' else p


class KernelReadout:
    """Compact, exact fitted kernel decision function without an n-by-n cache."""
    def __init__(self, fitted, n_train):
        self.n_train = int(n_train)
        if isinstance(fitted, SVR):
            self.support = fitted.support_.copy()
            self.coefficients = fitted.dual_coef_.reshape(-1).copy()
            self.intercept = float(fitted.intercept_[0])
        elif isinstance(fitted, KernelRidge):
            self.support = np.arange(n_train)
            self.coefficients = fitted.dual_coef_.copy()
            self.intercept = 0.
        else:
            raise ValueError('unsupported fitted kernel regressor')

    def predict(self, kernel):
        k = np.asarray(kernel, float)
        if k.ndim != 2 or k.shape[1] != self.n_train or not np.isfinite(k).all():
            raise ValueError('invalid query kernel')
        return k[:, self.support] @ self.coefficients + self.intercept


class SequenceTailFusion:
    def __init__(self, bundle):
        bundle = Path(bundle)
        self.config = json.loads((bundle/'model.json').read_text())
        for name, expected in self.config['file_hashes'].items():
            if hashlib.sha256((bundle/name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'model hash mismatch: {name}')
        self.previous = TailPriorityFusion(bundle/'previous')
        self.package = joblib.load(bundle/'sequence_expert.joblib')
        self.mix = float(self.config['mix'])
        if not 0 <= self.mix <= 1:
            raise ValueError('invalid mixing strength')

    def predict(self, baseline, esm1v_mean, esm1v_std, esm2_mean, esm2_std,
                retrieval, sequences, *, kernel_features):
        prior = self.previous.predict(baseline, esm1v_mean, esm1v_std, esm2_mean, esm2_std,
                                      retrieval, sequences, kernel_features=kernel_features)
        recipe = self.config['recipe']
        if recipe['features'] == 'dual':
            x = np.column_stack([pool_features(esm1v_mean, esm1v_std, 'mean_std'),
                                 pool_features(esm2_mean, esm2_std, 'mean_std')])
        else:
            x = kernel_features
        z = transform(x, self.package['transform'])
        prediction = []
        # Keep the full predictor usable with arbitrarily large query batches.
        for start in range(0, len(z), 256):
            kernel = rbf_kernel(z[start:start+256], self.package['train_features'],
                                gamma=self.package['transform']['gamma'])
            prediction.append(self.package['model'].predict(kernel))
        raw = np.concatenate(prediction) if prediction else np.empty(0)
        expert = complete_expert(raw, prior['original'], recipe['target'])
        base = prior['original'] if self.config['base'] == 'original' else prior['prediction']
        return dict(prediction=(1-self.mix)*base+self.mix*expert, expert=expert,
                    original=prior['original'], previous=prior['prediction'], base=base)
