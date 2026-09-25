"""Validation-only comparison against frozen robust v1 across original seeds."""
import csv
import json
import numpy as np
from develop_phgeofuse_regression import OUT, metrics
from phgeofuse.cache import atomic_json


def main():
    d = np.load(OUT / 'development_features.npz')
    keys = d['keys'][7124:]
    y, low = d['yv'], d['lowv']
    seq = np.load(OUT / 'compact/mean_std_p0.25_a0.1.validation.npy')
    res = np.load(OUT / 'retrieval_residual/val_weight0.5_i150.npy')
    robust = {}
    for seed in [0, 1, 2, 3, 42]:
        name = 'baseline_validation.csv' if seed == 42 else f'baseline_seed{seed}_validation.csv'
        with (OUT / name).open() as handle:
            rows = {r['key']: float(r['prediction']) for r in csv.DictReader(handle)}
        assert set(rows) == set(keys)
        robust[seed] = .5 * np.array([rows[k] for k in keys]) + .25 * seq + .25 * res

    def summary(pred):
        ms = [metrics(y, p, low) for p in pred.values()]
        out = {k: float(np.mean([m[k] for m in ms])) for k in ['rmse', 'mae', 'r2', 'spearman']}
        for group in ['acidic', 'neutral', 'alkaline', 'low_homology']:
            out[group] = {k: float(np.mean([m[group][k] for m in ms])) for k in ['rmse', 'bias']}
        return out

    reference = summary(robust)
    results = []
    for name in ['unweighted50', 'phweighted50', 'family_phweighted50']:
        p = np.load(OUT / 'dual_nested' / (name + '.validation.npy'))
        for weight in [.25, .5, .75]:
            preds = {s: (1-weight)*b + weight*p for s, b in robust.items()}
            m = summary(preds)
            passing = (all(m[k] < reference[k] for k in ['rmse', 'mae'])
                and m['spearman'] >= reference['spearman']
                and m['low_homology']['rmse'] < reference['low_homology']['rmse']
                and all(abs(m[k]['bias']) < abs(reference[k]['bias']) for k in ['acidic', 'alkaline']))
            row = {'dual_recipe': name, 'dual_weight': weight, 'mean_validation': m,
                   'passes_v1_constraints': passing}
            results.append(row)
    results.sort(key=lambda r: r['mean_validation']['rmse'])
    atomic_json(OUT / 'dual_nested/fusion_validation.json', {'reference_v1': reference,
        'results': results, 'protocol': 'Nine exploratory validation combinations; no new test evaluation.'})
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
