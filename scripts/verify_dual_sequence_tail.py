"""Independently audit fitted subsets, exports, metrics and the test gate."""
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
from phgeofuse.cache import atomic_json, sha256_file
from phgeofuse.delta_ref.data import DevelopmentData
from phgeofuse.retrieval import RetrievalStore
from phgeofuse.sequence_tail import SequenceTailFusion
from verify_dual_tail_weighting import verify_metrics

CACHE = ROOT/'experiments/delta_ref_phopt_20260916/baseline/seed42'
PREVIOUS = ROOT/'experiments/dual_tail_priority_20260919'
SOURCE = ROOT/'experiments/phgeofuse_redesign_20260914'
TOKEN = ROOT/'experiments/field_comparisons_phopt_20260917/esm1v_full_precision/features.npz'


def rows(path):
    return list(csv.DictReader(path.open()))


def column(rr, name):
    return np.array([float(r[name]) for r in rr])


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=0., atol=1e-7)


def check_metrics(rr, m):
    verify_metrics(rr, m)
    y, p = column(rr, 'label'), column(rr, 'prediction')
    for group, mask in [('extreme_acid', y <= 4), ('extreme_alkaline', y >= 10)]:
        err = abs(p[mask]-y[mask]); top = max(1, int(np.ceil(.2*len(err))))
        close(m[group]['abs_error_p90'], np.quantile(err, .9))
        close(m[group]['worst20_mse'], np.mean(np.sort(err**2)[-top:]))
        close(m[group]['underestimated_fraction'], np.mean(p[mask] < y[mask]))


def check_selection(row, old, svr):
    m, s = row['metrics'], row['selection']
    ratios = [m[g][v]/svr[g][v] for g in ['extreme_acid', 'extreme_alkaline'] for v in ['rmse', 'mae']]
    expected = m['all']['rmse'] <= 1.05*old['all']['rmse'] and m['all']['mae'] <= 1.035*old['all']['mae']
    assert expected == s['proxy_overall_budget']
    close(max(ratios), s['worst_tail_ratio']); close(np.mean(ratios), s['mean_tail_ratio'])


def ranking_key(row):
    s = row['selection']
    return (not s['proxy_overall_budget'], s['worst_tail_ratio'], s['mean_tail_ratio'], row['metrics']['all']['rmse'])


def main(out):
    status = json.loads((out/'status.json').read_text())
    assert status['event'] == 'complete'
    sources = json.loads((out/'source_hashes.json').read_text())
    hashes = json.loads((out/'artifact_hashes.json').read_text())
    for name, value in sources.items():
        assert sha256_file(name) == value, name
    for name, value in hashes.items():
        assert sha256_file(out/name) == value, name
    protocol = json.loads((out/'protocol.json').read_text())
    development = json.loads((out/'development.json').read_text())
    d = DevelopmentData.load(ROOT/'configs/delta_ref_phopt.yaml'); n = len(d.train)
    assert n == 7124 and len(d.validation) == 760
    with np.load(TOKEN) as archive:
        token = archive['token_mean'].astype(float)
        token_index = {str(k): i for i, k in enumerate(archive['keys'])}
    features = dict(dual=d.embeddings, token=token[[token_index[k] for k in d.keys]])
    maxdiff = 0.; fold_replays = 0; oof = {r['name']: np.full(n, np.nan) for r in protocol['recipes']}
    original = np.full(n, np.nan)
    for outer in range(5):
        fit, query = d.partition([outer]); guide = np.full(n, np.nan)
        for inner in range(5):
            if inner == outer:
                continue
            excluded = sorted([outer, inner]); _, q = d.partition(excluded)
            with np.load(CACHE/('excluded_'+'_'.join(map(str, excluded)))/'predictions.npz') as archive:
                np.testing.assert_array_equal(archive['keys'], d.keys[q])
                use = d.folds[q] == inner
                guide[q[use]] = archive['prediction'][use]
        assert np.isnan(guide[query]).all()
        with np.load(CACHE/f'excluded_{outer}/predictions.npz') as archive:
            np.testing.assert_array_equal(archive['keys'], d.keys[query])
            original[query] = archive['prediction']
        for feature, raw in features.items():
            mean, std = raw[fit].mean(0), raw[fit].std(0)
            z, qz = (raw[fit]-mean)/(std+1e-8), (raw[query]-mean)/(std+1e-8)
            kernel = rbf_kernel(qz, z, gamma=1/raw.shape[1])
            for recipe in [r for r in protocol['recipes'] if r['features'] == feature]:
                folder = out/'folds'/recipe['name']/f'outer{outer}'
                isolation = json.loads((folder/'isolation.json').read_text())
                assert isolation['certificate'] == d.certificate(fit, query, [outer])
                assert isolation['guide_exclusions'] == [sorted([outer, j]) for j in range(5) if j != outer]
                package = joblib.load(folder/'model.joblib')
                np.testing.assert_array_equal(package['fit_indices'], fit)
                close(package['transform']['mean'], mean); close(package['transform']['std'], std)
                close(package['transform']['gamma'], 1/raw.shape[1])
                model = package['model']
                # Evaluate the saved coefficients directly, not its predict method.
                pred = kernel[:, model.support]@model.coefficients+model.intercept
                if recipe['target'] == 'residual':
                    pred += original[query]
                rr = rows(folder/'predictions.csv')
                assert [r['key'] for r in rr] == d.keys[query].tolist()
                close(column(rr, 'label'), d.labels[query]); close(column(rr, 'prediction'), pred)
                maxdiff = max(maxdiff, float(np.max(abs(pred-column(rr, 'prediction')))))
                oof[recipe['name']][query] = pred
                train = rows(folder/'training_objective.csv')
                assert [r['key'] for r in train] == d.keys[fit].tolist()
                y = d.labels[fit]; close(column(train, 'label'), y)
                close(column(train, 'guide_cf'), guide[fit])
                a, b = y <= 4, y >= 10; am, bm = recipe['acid_mass'], recipe['alkaline_mass']
                weights = 1-am-bm+am*a/a.mean()+bm*b/b.mean()
                close(column(train, 'weight'), weights); close(weights.mean(), 1.)
                close(column(train, 'target'), y-guide[fit] if recipe['target'] == 'residual' else y)
                info = json.loads((folder/'fit.json').read_text())
                assert info['recipe'] == recipe and info['fit_keys'] == d.keys[fit].tolist()
                assert info['compact_native_max_difference'] < 1e-7
                fold_replays += 1
    previous_rows = rows(PREVIOUS/'oof/K1_s0.75.csv')
    assert [r['key'] for r in previous_rows] == d.keys[:n].tolist()
    previous = column(previous_rows, 'prediction')
    assert len(development['ranking']) == 120
    for row in development['ranking']:
        rr = rows(out/'oof'/f'{row["name"]}.csv')
        assert [r['key'] for r in rr] == d.keys[:n].tolist()
        base = original if row['base'] == 'original' else previous
        expected = (1-row['mix'])*base+row['mix']*oof[row['recipe']['name']]
        close(column(rr, 'prediction'), expected)
        check_metrics(rr, row['metrics'])
        check_selection(row, development['original'], development['local_svr'])
    assert development['ranking'] == sorted(development['ranking'], key=ranking_key)
    expected_shortlist = []; seen = set()
    for row in development['ranking']:
        if row['recipe']['name'] not in seen:
            expected_shortlist.append(row); seen.add(row['recipe']['name'])
        if len(expected_shortlist) == 4:
            break
    plan = json.loads((out/'validation_plan.json').read_text())
    assert plan['candidates'] == expected_shortlist and not plan['test_used']
    release = json.loads((out/'release.json').read_text()); name = release['candidate']
    assert release['protocol_sha256'] == sha256_file(out/'protocol.json')
    assert release['model_sha256'] == sha256_file(out/'full'/name/'bundle/model.json')
    assert release['test_scored'] is False
    # Full fits: no validation/test row may enter normalization or residual target.
    for row in expected_shortlist:
        folder = out/'full'/row['name']; package = joblib.load(folder/'bundle/sequence_expert.joblib')
        raw = features[row['recipe']['features']][:n]
        close(package['transform']['mean'], raw.mean(0)); close(package['transform']['std'], raw.std(0))
        close(package['train_features'], (raw-raw.mean(0))/(raw.std(0)+1e-8))
        np.testing.assert_array_equal(package['fit_indices'], np.arange(n))
        train = rows(folder/'training_objective.csv')
        assert [r['key'] for r in train] == d.keys[:n].tolist()
        close(column(train, 'guide_cf'), original); close(column(train, 'label'), d.labels[:n])
        recipe = row['recipe']; y = d.labels[:n]; a, b = y <= 4, y >= 10
        close(column(train, 'weight'), 1-recipe['acid_mass']-recipe['alkaline_mass']+
              recipe['acid_mass']*a/a.mean()+recipe['alkaline_mass']*b/b.mean())
        close(column(train, 'target'), y-original if recipe['target'] == 'residual' else y)
    replays = []
    for split in (['validation', 'test'] if release['permitted_test'] else ['validation']):
        dd = DevelopmentData.load(ROOT/'configs/delta_ref_phopt.yaml', test=True) if split == 'test' else d
        ix = np.arange(len(dd.keys)) if split == 'test' else dd.validation
        summary = json.loads((out/f'{split}.json').read_text())
        feature_list = []
        for encoder in ['esm1v', 'esm2']:
            with np.load(SOURCE/f'{encoder}_masked'/('features_test.npz' if split == 'test' else 'features.npz')) as z:
                index = {str(k): i for i, k in enumerate(z['keys'])}; order = [index[k] for k in dd.keys[ix]]
                feature_list.extend([z['mean'][order], z['std'][order]])
        store = RetrievalStore.load(ROOT/'artifacts/phgeofuse/retrieval.pt')
        retrieval = np.array([store.features(k).numpy() for k in dd.keys[ix]])
        raw_path = (ROOT/'experiments/phgeofuse_phopt_full_20260913/seed42/test_predictions.csv'
                    if split == 'test' else SOURCE/'baseline_validation.csv')
        raw_map = {r['key']: float(r['prediction']) for r in rows(raw_path)}
        raw = np.array([raw_map[k] for k in dd.keys[ix]])
        token_query = token[[token_index[k] for k in dd.keys[ix]]]
        for row in summary['candidates']:
            rr = rows(out/split/f'{row["name"]}.csv')
            assert [r['key'] for r in rr] == dd.keys[ix].tolist()
            close(column(rr, 'label'), dd.labels[ix]); close(column(rr, 'raw_baseline'), raw)
            predictor = SequenceTailFusion(out/'full'/row['name']/'bundle')
            prediction = predictor.predict(raw, *feature_list, retrieval, [dd.records[i].sequence for i in ix],
                                           kernel_features=token_query)['prediction']
            diff = float(np.max(abs(prediction-column(rr, 'prediction')))); maxdiff = max(maxdiff, diff)
            close(prediction, column(rr, 'prediction')); check_metrics(rr, row['metrics'])
            if split == 'validation':
                check_selection(row, summary['original'], summary['local_svr'])
            replays.append(dict(split=split, candidate=row['name'], count=len(rr), max_difference=diff))
    validation = json.loads((out/'validation.json').read_text())
    ranked = sorted(validation['candidates'], key=ranking_key)
    assert ranked == release['validation_ranking'] and ranked[0]['name'] == name
    chosen = next(r for r in expected_shortlist if r['name'] == name)
    old_val_worst = max(validation['previous'][g][m]/validation['local_svr'][g][m]
                        for g in ['extreme_acid', 'extreme_alkaline'] for m in ['rmse', 'mae'])
    gates = dict(oof_budget=chosen['selection']['proxy_overall_budget'],
                 validation_budget=ranked[0]['selection']['proxy_overall_budget'],
                 oof_tail_improvement=chosen['selection']['worst_tail_ratio'] < development['previous_selection']['worst_tail_ratio'],
                 validation_tail_improvement=ranked[0]['selection']['worst_tail_ratio'] < old_val_worst)
    assert gates == release['test_gate'] and all(gates.values()) == release['permitted_test']
    decision = json.loads((out/'decision.json').read_text())
    assert decision['test_candidate_count'] == int(release['permitted_test'])
    if not release['permitted_test']:
        assert not (out/'test').exists() and not decision['user_target_reached']
    else:
        test = json.loads((out/'test.json').read_text())
        assert len(test['candidates']) == 1 and test['count'] == 1971
        m = test['candidates'][0]['metrics']; rr = rows(out/'test'/f'{name}.csv')
        ep = {r['key']: r for r in rows(ROOT/'experiments/delta_ref_phopt_20260916/analysis/ephod_official_strict_tail_20260916/predictions.csv')}
        assert set(ep) == {r['key'] for r in rr}
        y = column(rr, 'label'); p = np.array([float(ep[r['key']]['Ensemble']) for r in rr])
        close(y, [float(ep[r['key']]['label']) for r in rr])
        checks = dict(venus_rmse=m['all']['rmse'] < .809, venus_mae=m['all']['mae'] < .578)
        for g, mask in [('extreme_acid', y <= 4), ('extreme_alkaline', y >= 10)]:
            e = p[mask]-y[mask]
            checks[g+'_rmse'] = m[g]['rmse'] < np.sqrt(np.mean(e**2))
            checks[g+'_mae'] = m[g]['mae'] < np.mean(abs(e))
        assert checks == decision['checks'] and all(checks.values()) == decision['user_target_reached']
    result = dict(status='passed', fold_models_replayed=fold_replays, double_exclusion_target_audits=fold_replays,
        oof_combinations_verified=120, full_training_audits=4, full_prediction_replays=replays,
        maximum_prediction_difference=maxdiff, source_hashes=len(sources), artifact_hashes=len(hashes),
        test_gate_recomputed=True, test_candidates=decision['test_candidate_count'], acceptance_recomputed=bool(release['permitted_test']))
    atomic_json(out/'verification.json', result)
    target = out/'reproduce/scripts/verify_dual_sequence_tail.py'; shutil.copy2(Path(__file__), target)
    with (out/'REPORT_ZH.md').open('a', encoding='utf-8') as f:
        f.write(f'\n独立核验通过：100个分折模型和双排除目标、120个OOF组合、4个全量拟合及{len(replays)}份完整预测；最大重放差{maxdiff:.3g}。测试准入及实际验收逐项复算。\n')
    atomic_json(out/'artifact_hashes.json', {str(p.relative_to(out)): sha256_file(p)
        for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['artifact_hashes.json', 'status.json']})
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'experiments/dual_sequence_tail_20260919')
    main(parser.parse_args().output)
