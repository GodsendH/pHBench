"""Strict outer-fold confidence gate over already generated OOF experts."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from develop_phgeofuse_regression import OUT, metrics
from phgeofuse.cache import atomic_json


def main():
    out = OUT / 'confidence_gate'
    out.mkdir(exist_ok=True)
    data = np.load(OUT / 'dual_nested/predictions.npz')
    retrieval = np.load(OUT / 'nested_homology_strict/predictions.npz')['retrieval']
    y, folds, groups = data['y'], data['fold'], data['groups']
    names = ['anchor', 'unweighted50', 'phweighted50', 'family_phweighted50']
    p = np.column_stack([data[n] for n in names])
    # Confidence features are available at inference and do not use labels.
    r = retrieval
    f = np.column_stack([p, r[:, 5:7], r[:, 13:15], r[:, 4:5],
                         np.abs(p[:, 1:2] - p[:, 0:1]), np.abs(p[:, 2:3] - p[:, 0:1]),
                         np.abs(p[:, 3:4] - p[:, 0:1]), np.abs(p[:, 0:1] - 7.)])
    recipes = [{'alpha': a, 'target': t, 'name': f'a{a}_t{t}'}
               for a in [.1, 1., 10.] for t in ['direct', 'unweighted', 'phweighted']]
    atomic_json(out / 'protocol.json', {'inputs': names + ['retrieval variance/margins/identity/extremeness'],
        'recipes': recipes, 'inner_gate': 'fit on the other four outer-fold predictions; '
        'evaluate on the held-out fold', 'test_access': False,
        'caveat': 'Fixed exploratory gate comparison; no test evaluation.'})
    results = []
    for recipe in recipes:
        final = np.full(len(y), np.nan)
        fold_rows = []
        for k in range(5):
            tr, te = folds != k, folds == k
            assert not set(groups[tr]) & set(groups[te])
            if recipe['target'] == 'direct':
                target = y[tr]
            elif recipe['target'] == 'unweighted':
                target = y[tr] - p[tr, 1]
            else:
                target = y[tr] - p[tr, 2]
            if recipe['target'] == 'direct':
                model = make_pipeline(StandardScaler(), Ridge(alpha=recipe['alpha']))
                model.fit(f[tr], target)
                pred = model.predict(f[te])
            else:
                model = make_pipeline(StandardScaler(), Ridge(alpha=recipe['alpha']))
                model.fit(f[tr], target)
                base = p[te, 1] if recipe['target'] == 'unweighted' else p[te, 2]
                pred = base + model.predict(f[te])
            final[te] = pred
            fold_rows.append({'fold': k, 'metrics': metrics(y[te], pred, np.zeros(te.sum(), dtype=bool))})
        row = {'recipe': recipe, 'strict_oof': metrics(y, final, np.zeros(len(y), dtype=bool)),
               'folds': fold_rows}
        results.append(row)
        np.save(out / (recipe['name'] + '.npy'), final)
        print(json.dumps({'recipe': recipe, 'metrics': row['strict_oof']}), flush=True)
    atomic_json(out / 'results.json', results)
    atomic_json(out / 'status.json', {'status': 'complete'})
    print('CONFIDENCE_GATE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
