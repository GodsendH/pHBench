"""PHOPT-only, outer/inner family-excluded sequence expert experiment.

Fixed recipes are exploratory comparisons, not post-selection confirmation.
Every outer fold is excluded from sequence fitting, retrieval and meta fitting.
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
import sys
import json
import time
import hashlib
import argparse
from pathlib import Path
import numpy as np
import torch
import joblib
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from develop_phgeofuse_regression import OUT, ROOT, metrics
from phgeofuse.cache import atomic_json
from phgeofuse.io import read_manifest
from phgeofuse.robust_fusion import pool_features, chemistry_features
from phgeofuse.dual_fusion import retrieval_sequence_anchor as anchor
from phgeofuse.retrieval import RetrievalStore, record_key
from phgeofuse.reliability_gate import ReliabilityGate
from phgeofuse.reliability_fusion import reliability_inputs

RECIPES = [
    {'name': 'dual_ridge_control', 'kind': 'ridge', 'alpha': .2, 'charge_scale': 0.},
    {'name': 'dual_charge_ridge', 'kind': 'ridge', 'alpha': .2, 'charge_scale': .5},
    {'name': 'dual_pls8', 'kind': 'pls', 'components': 8, 'charge_scale': 0.},
    {'name': 'dual_pls24', 'kind': 'pls', 'components': 24, 'charge_scale': 0.},
]


def fit_sequence(recipe, x, y):
    if recipe['kind'] == 'ridge':
        model = Ridge(alpha=recipe['alpha'], solver='cholesky')
    else:
        model = PLSRegression(n_components=recipe['components'], scale=False)
    return model.fit(x, y)


def predict(model, x):
    return np.asarray(model.predict(x), dtype=float).reshape(-1)


def fit_residual(x, residual):
    return HistGradientBoostingRegressor(max_leaf_nodes=7, max_iter=50,
        min_samples_leaf=80, l2_regularization=30, learning_rate=.05,
        early_stopping=False, random_state=42).fit(x, residual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--structural-reliability', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = OUT / ('structural_reliability_nested_20260915' if args.structural_reliability
                 else 'multiview_nested_20260915')
    out.mkdir(exist_ok=False)
    started = time.time()
    records = [r for r in read_manifest(ROOT / 'artifacts/phgeofuse/manifest.csv')
               if r.split in ('train', 'validation')]
    keys = np.array([record_key(r) for r in records])
    training = np.array([r.split == 'train' for r in records])
    assert training.sum() == 7124 and (~training).sum() == 760
    trainkeys = keys[training]
    labels = np.array([r.ph_opt for r in records], dtype=float)
    y, yv = labels[training], labels[~training]
    foldpath = OUT / 'homology_oof/strict_folds.json'
    rows = json.loads(foldpath.read_text())['rows']
    assert [r['key'] for r in rows] == [r.protein_id for r in records if r.split == 'train']
    folds = np.array([r['fold'] for r in rows])
    groups = np.array([r['group'] for r in rows])
    source = OUT / 'nested_homology_strict'
    old = np.load(OUT / ('reliability_residual_nested_20260915/quality_residual_unweighted_predictions.npz'
                        if args.structural_reliability else 'dual_nested/predictions.npz'))
    old_prediction = old['prediction'] if args.structural_reliability else old['unweighted50']
    assert np.array_equal(old['keys'], trainkeys) and np.array_equal(old['y'], y)
    parts = []
    hashes = {}
    for encoder in ('esm1v', 'esm2'):
        path = OUT / f'{encoder}_masked/features.npz'
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as f:
            mapping = {str(k): i for i, k in enumerate(f['keys'])}
            idx = [mapping[k] for k in keys]
            parts.append(pool_features(f['mean'][idx], f['std'][idx], 'mean_std'))
            if encoder == 'esm1v':
                mean = f['mean'][idx].astype(float)
                norm = np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12)
                charge = []
                for name in ('acid', 'basic'):
                    values = f[name][idx].astype(float)
                    missing = np.linalg.norm(values, axis=1) == 0
                    delta = (values - mean) / norm
                    delta[missing] = 0
                    charge.append(delta)
    dual = np.column_stack(parts)
    charge = np.column_stack(charge)
    recipes = RECIPES
    if args.structural_reliability:
        path = OUT / 'saprot_regions_20260915/features.npz'
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as f:
            mapping = {str(k): i for i,k in enumerate(f['keys'])}
            idx = [mapping[k] for k in keys]
            structural_global = pool_features(f['mean'][idx], f['std'][idx], 'mean_std')
            structural_local = f['local'][idx].astype(float)
        recipes = [{'name': name, 'kind': 'ridge', 'alpha': .2, 'charge_scale': 0.,
                    'structural_global': global_scale, 'structural_local': local_scale}
                   for name,global_scale,local_scale in [('dual_ridge_control',0.,0.),
                    ('structure_global',.5,0.),('structure_local',0.,.5),
                    ('structure_global_local',.5,.5)]]
    chemall = chemistry_features([r.sequence for r in records])
    chem, chemv = chemall[training], chemall[~training]
    store = RetrievalStore.load(ROOT / 'artifacts/phgeofuse/retrieval.pt')
    assert store.payload['training_keys'] == trainkeys.tolist()
    rv = np.array([store.features(k).numpy() for k in keys[~training]], dtype=float)
    lowv = ~((rv[:, 4] >= .2) & (rv[:, 9] >= .8) & (rv[:, 10] >= .8))
    protocol = {'dataset': 'PHOPT', 'train': len(y), 'validation': len(yv),
        'test_access': False, 'recipes': recipes, 'seed': 42,
        'reliability_gate': args.structural_reliability,
        'python': sys.executable, 'feature_sha256': hashes,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'fold_sha256': hashlib.sha256(foldpath.read_bytes()).hexdigest(),
        'exclusion': 'outer fold excluded from every sequence/retrieval/meta fit; '
                     'inner fold additionally excluded from meta-training base predictions',
        'selection': 'fixed exploratory recipes; original validation not used to tune these recipes',
        'scope': 'sequence plus retrieval residual expert; not nested evaluation of full PHGeoFuse blend'}
    atomic_json(out / 'protocol.json', protocol)
    results = {}
    payloads = {}
    for recipe in recipes:
        name = recipe['name']
        xall = (np.column_stack([dual, recipe['charge_scale'] * charge])
                if recipe['charge_scale'] else dual)
        if args.structural_reliability:
            extra = [xall]
            if recipe['structural_global']:
                extra.append(recipe['structural_global'] * structural_global)
            if recipe['structural_local']:
                extra.append(recipe['structural_local'] * structural_local)
            xall = np.column_stack(extra)
        x, xv = xall[training], xall[~training]
        cache = {}

        def base(excluded):
            excluded = tuple(sorted(excluded))
            if excluded in cache:
                return cache[excluded]
            query = np.flatnonzero(np.isin(folds, excluded))
            ref = np.flatnonzero(~np.isin(folds, excluded))
            assert not set(groups[query]) & set(groups[ref])
            if excluded not in payloads:
                path = source / ('excluded_' + '_'.join(map(str, excluded)) + '.pt')
                payload = torch.load(path, map_location='cpu')
                assert payload['metadata']['query_keys'] == trainkeys[query].tolist()
                assert payload['metadata']['reference_keys'] == trainkeys[ref].tolist()
                payloads[excluded] = np.asarray(payload['retrieval'], dtype=float)
            atomic_json(out / 'status.json', {'status': 'running', 'pid': os.getpid(),
                'recipe': name, 'excluded': excluded, 'updated': time.time()})
            model = fit_sequence(recipe, x[ref], y[ref])
            prediction = predict(model, x[query])
            train_rmse = float(np.sqrt(np.mean((predict(model, x[ref]) - y[ref]) ** 2)))
            cache[excluded] = query, payloads[excluded], prediction, train_rmse
            np.savez(out / (name + '_excluded_' + '_'.join(map(str, excluded)) + '.npz'),
                keys=trainkeys[query], reference_keys=trainkeys[ref], prediction=prediction)
            print('BASE_COMPLETE', name, excluded, round(time.time() - started, 1), flush=True)
            return cache[excluded]

        p = np.full(len(y), np.nan)
        seq = np.full(len(y), np.nan)
        r_all = np.full((len(y), 15), np.nan)
        fold_results = []
        for outer in range(5):
            tr = np.flatnonzero(folds != outer)
            te, r, s, train_rmse = base([outer])
            ir, iseq = np.full_like(r_all, np.nan), np.full(len(y), np.nan)
            for inner in range(5):
                if inner == outer:
                    continue
                q, rr, ss, _ = base([outer, inner])
                keep = folds[q] == inner
                ir[q[keep]], iseq[q[keep]] = rr[keep], ss[keep]
            assert np.isnan(iseq[te]).all() and np.isfinite(iseq[tr]).all()
            a, av = anchor(ir[tr], iseq[tr]), anchor(r, s)
            if args.structural_reliability:
                gate = ReliabilityGate(.01).fit(*reliability_inputs(ir[tr],iseq[tr]),y[tr])
                a = gate.predict(*reliability_inputs(ir[tr],iseq[tr]))
                av = gate.predict(*reliability_inputs(r,s))
                joblib.dump(gate,out/f'{name}_outer{outer}_gate.joblib')
            meta = np.column_stack([ir[tr], iseq[tr], chem[tr]])
            model = fit_residual(meta, y[tr] - a)
            p[te] = av + model.predict(np.column_stack([r, s, chem[te]]))
            seq[te], r_all[te] = s, r
            low = ~((r[:, 4] >= .2) & (r[:, 9] >= .8) & (r[:, 10] >= .8))
            fold_results.append({'outer': outer, 'heldout': metrics(y[te], p[te], low),
                'sequence_heldout': metrics(y[te], s, low),
                'sequence_train_rmse': train_rmse,
                'meta_train_rmse': float(np.sqrt(np.mean((a + model.predict(meta) - y[tr]) ** 2)))})
            joblib.dump(model, out / f'{name}_outer{outer}_residual.joblib')
            print('OUTER_COMPLETE', name, outer, fold_results[-1]['heldout']['rmse'], flush=True)
        assert np.isfinite(p).all() and np.isfinite(seq).all()
        low = ~((r_all[:, 4] >= .2) & (r_all[:, 9] >= .8) & (r_all[:, 10] >= .8))
        sequence_model = fit_sequence(recipe, x, y)
        sv = predict(sequence_model, xv)
        a, av = anchor(r_all,seq), anchor(rv,sv)
        if args.structural_reliability:
            gate = ReliabilityGate(.01).fit(*reliability_inputs(r_all,seq),y)
            a = gate.predict(*reliability_inputs(r_all,seq))
            av = gate.predict(*reliability_inputs(rv,sv))
            joblib.dump(gate,out/f'{name}_gate.joblib')
        residual = fit_residual(np.column_stack([r_all, seq, chem]), y - a)
        val = av + residual.predict(np.column_stack([rv, sv, chemv]))
        joblib.dump(sequence_model, out / f'{name}_sequence.joblib')
        joblib.dump(residual, out / f'{name}_residual.joblib')
        np.savez(out / f'{name}_predictions.npz', keys=trainkeys, y=y, fold=folds, groups=groups,
            prediction=p, sequence=seq, retrieval=r_all, validation=val, yv=yv,
            validation_keys=keys[~training])
        result = {'recipe': recipe, 'strict_nested': metrics(y, p, low),
            'sequence_oof': metrics(y, seq, low), 'validation': metrics(yv, val, lowv),
            'fold_results': fold_results}
        # Fixed family bootstrap conditional on these folds and fitted recipes.
        _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
        draws = np.random.default_rng(42).integers(len(counts), size=(1000, len(counts)))
        denom = counts[draws].sum(axis=1)
        sse = np.bincount(inv, weights=(p - y)**2)
        ref_sse = np.bincount(inv, weights=(old_prediction - y)**2)
        delta = np.sqrt(sse[draws].sum(1)/denom) - np.sqrt(ref_sse[draws].sum(1)/denom)
        result['family_bootstrap_rmse_delta95_vs_dual'] = np.quantile(delta, [.025, .975]).tolist()
        result['comparison_reference'] = ('reliability_residual_unweighted' if args.structural_reliability
                                           else 'dual_unweighted50')
        if name == 'dual_ridge_control':
            result['control_max_abs_diff'] = float(np.max(np.abs(p - old_prediction)))
            if result['control_max_abs_diff'] > 1e-7:
                raise RuntimeError('same-protocol control failed to reproduce')
        results[name] = result
        atomic_json(out / 'results.json', results)
        print('RECIPE_COMPLETE', name, json.dumps(result['strict_nested']), flush=True)
    atomic_json(out / 'status.json', {'status': 'complete', 'pid': os.getpid(),
        'updated': time.time(), 'elapsed_seconds': time.time() - started})
    print('MULTIVIEW_NESTED_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
