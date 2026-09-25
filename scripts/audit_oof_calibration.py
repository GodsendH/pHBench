"""Distinguish true-label tail bias from predicted-value calibration on grouped OOF."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from develop_phgeofuse_regression import OUT, metrics
from phgeofuse.cache import atomic_json


def main():
    source = OUT / 'dual_nested/predictions.npz'
    data = np.load(source)
    y = data['y']
    groups = data['groups']
    report = {'source': str(source), 'test_access': False, 'models': {}}
    report['note'] = ('Descriptive strict-group outer-held-out predictions only. Fitting a line '
        'to these predictions estimates calibration, not a new independent model score. '
        'True-label-conditioned tail bias and prediction-conditioned calibration differ.')
    names = ['sequence', 'anchor', 'unweighted50', 'phweighted50', 'family_phweighted50']
    for name in names:
        p = data[name]
        slope = LinearRegression().fit(p[:, None], y)
        shrink = LinearRegression().fit(y[:, None], p)
        boundaries = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
        which = np.searchsorted(boundaries[1:-1], p, side='right')
        bins = []
        for b in range(len(boundaries)-1):
            mask = which == b
            bins.append({'count': int(mask.sum()), 'prediction_min': float(p[mask].min()),
                'prediction_max': float(p[mask].max()), 'prediction_mean': float(p[mask].mean()),
                'label_mean': float(y[mask].mean()), 'bias_prediction_minus_label': float((p[mask]-y[mask]).mean())})
        # Cluster-bootstrap uncertainty for the descriptive slope.
        _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
        sums = [np.bincount(inverse, weights=v) for v in [p, y, p*p, p*y]]
        rng = np.random.default_rng(42)
        slopes = []
        for _ in range(1000):
            draw = rng.integers(len(counts), size=len(counts))
            n = counts[draw].sum()
            sx, sy, sxx, sxy = [v[draw].sum() for v in sums]
            slopes.append((sxy - sx*sy/n) / (sxx - sx*sx/n))
        report['models'][name] = {'observed': metrics(y, p, np.zeros(len(y), dtype=bool)),
            'label_on_prediction_slope': float(slope.coef_[0]),
            'label_on_prediction_intercept': float(slope.intercept_),
            'slope_cluster_bootstrap_95ci': np.quantile(slopes, [.025, .975]).tolist(),
            'prediction_on_label_slope': float(shrink.coef_[0]),
            'prediction_sd': float(p.std()), 'label_sd': float(y.std()), 'predicted_deciles': bins}
    out = OUT / 'oof_calibration'
    out.mkdir(exist_ok=True)
    atomic_json(out / 'diagnostic.json', report)
    for name, row in report['models'].items():
        print(name, json.dumps({k: row[k] for k in ['label_on_prediction_slope',
            'label_on_prediction_intercept', 'slope_cluster_bootstrap_95ci', 'prediction_on_label_slope']}))


if __name__ == '__main__':
    main()
