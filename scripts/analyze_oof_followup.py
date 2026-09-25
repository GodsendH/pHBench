"""Validation-only comparison of completed OOF stacking; never reads test rows."""
import csv
import json
from pathlib import Path

import numpy as np

from develop_phgeofuse_regression import OUT, metrics


def main():
    data = np.load(OUT / 'development_features.npz')
    y, low = data['yv'], data['lowv']
    keys = data['keys'][len(data['y']):]
    baseline = {}
    for seed in [0, 1, 2, 3, 42]:
        filename = 'baseline_validation.csv' if seed == 42 else f'baseline_seed{seed}_validation.csv'
        with (OUT / filename).open() as handle:
            rows = {r['key']: float(r['prediction']) for r in csv.DictReader(handle)}
        assert set(rows) == set(keys)
        baseline[seed] = np.array([rows[k] for k in keys])
    sequence = np.load(OUT / 'compact/mean_std_p0.25_a0.1.validation.npy')
    residual = np.load(OUT / 'retrieval_residual/val_weight0.5_i150.npy')
    robust = {s: .5 * b + .25 * sequence + .25 * residual for s, b in baseline.items()}

    def summarize(predictions):
        per_seed = {s: metrics(y, p, low) for s, p in predictions.items()}
        summary = {k: float(np.mean([m[k] for m in per_seed.values()]))
                   for k in ['rmse', 'mae', 'r2', 'spearman']}
        for group in ['acidic', 'neutral', 'alkaline', 'low_homology']:
            summary[group] = {k: float(np.mean([m[group][k] for m in per_seed.values()]))
                              for k in ['rmse', 'bias']}
        return {'mean_validation': summary, 'per_seed': per_seed}

    reference = summarize(robust)
    candidates = []
    for item in json.loads((OUT / 'oof_stacking/results.json').read_text()):
        p = np.load(OUT / 'oof_stacking' / (item['name'] + '.validation.npy'))
        for weight in [.25, .5, .75, 1.]:
            result = summarize({s: (1 - weight) * b + weight * p for s, b in robust.items()})
            result.update(name=item['name'], oof_weight=weight)
            m, ref = result['mean_validation'], reference['mean_validation']
            result['improves_all_v1_metrics'] = (
                m['rmse'] < ref['rmse'] and m['mae'] < ref['mae']
                and m['spearman'] >= ref['spearman']
                and m['low_homology']['rmse'] < ref['low_homology']['rmse']
                and abs(m['acidic']['bias']) < abs(ref['acidic']['bias'])
                and abs(m['alkaline']['bias']) < abs(ref['alkaline']['bias']))
            candidates.append(result)
    candidates.sort(key=lambda r: r['mean_validation']['rmse'])
    oof = np.load(OUT / 'oof_stacking/oof_features.npz')
    r, s = oof['retrieval'], oof['sequence']
    available = r[:, 7:9]
    anchor = ((r[:, :2] * available).sum(1) + s) / (available.sum(1) + 1)
    low_oof = ~((r[:, 4] >= .2) & (r[:, 9] >= .8) & (r[:, 10] >= .8))
    counts = {}
    for split, labels in [('train', data['y']), ('validation', y)]:
        counts[split] = {'count': len(labels), 'acidic': int((labels < 6).sum()),
                         'neutral': int(((labels >= 6) & (labels < 8)).sum()),
                         'alkaline': int((labels >= 8).sum())}
    result = {'protocol': 'Exploratory validation-only search after v1 freeze; no new test evaluation. '
                         'OOF base predictions are held out, but the fitted meta-model has not been '
                         'evaluated with a nested outer cross-validation loop.',
              'counts': counts, 'baseline': summarize(baseline), 'robust_v1': reference,
              'oof_sequence': metrics(data['y'], s, low_oof),
              'oof_anchor': metrics(data['y'], anchor, low_oof),
              'candidates': candidates}
    output = OUT / 'oof_stacking/validation_followup.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'counts': counts, 'baseline': result['baseline']['mean_validation'],
                      'robust_v1': reference['mean_validation'], 'best': candidates[0],
                      'passes': sum(c['improves_all_v1_metrics'] for c in candidates),
                      'oof_sequence': result['oof_sequence'], 'oof_anchor': result['oof_anchor']}, indent=2))


if __name__ == '__main__':
    main()
