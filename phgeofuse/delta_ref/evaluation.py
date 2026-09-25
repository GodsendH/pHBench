"""Frozen, keyed, five-seed follow-up evaluation and family comparisons."""
from __future__ import annotations

import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import numpy as np

from phgeofuse.cache import atomic_json, sha256_file
from phgeofuse.config import load_config, path
from .data import DevelopmentData, atomic_npz, freeze_json, stable_hash, write_predictions
from .metrics import acceptance, metrics, paired_family_bootstrap, seed_summary
from .model import finite_array
from .training import load_bundle

SEEDS = (0, 1, 2, 3, 42)
TEST_LIMITS = {
    'all.rmse': .78271, 'all.mae': .56420,
    'acid.rmse': 1.50566, 'acid.mae': 1.17960, 'acid.abs_bias': 1.00610,
    'alkaline.rmse': 1.99925, 'alkaline.mae': 1.86109, 'alkaline.abs_bias': 1.65431,
}


def _rows(source):
    with Path(source).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any('key' not in row for row in rows):
        raise ValueError(f'empty or unkeyed CSV: {source}')
    result = {row['key']: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f'duplicate prediction keys: {source}')
    return result


def load_five_predictions(sources, labels=None):
    if len(sources) != 5:
        raise ValueError('exactly five CSVs in seed order 0,1,2,3,42 are required')
    tables = [_rows(p) for p in sources]
    keys = sorted(tables[0])
    if any(set(t) != set(keys) for t in tables):
        raise ValueError('seed files must have identical sample coverage')
    for seed, table in zip(SEEDS, tables):
        if any('seed' in row and int(row['seed']) != seed for row in table.values()):
            raise ValueError('prediction seed order differs from the locked protocol')
    if labels is None:
        if any('label' not in row for row in tables[0].values()):
            raise ValueError('candidate evaluation requires labels')
        labels = {k: float(tables[0][k]['label']) for k in keys}
    if not set(keys) <= set(labels):
        raise ValueError('prediction contains unknown evaluation samples')
    for table in tables:
        for key, row in table.items():
            if 'label' in row and not np.isclose(float(row['label']), labels[key], rtol=0, atol=1e-8):
                raise ValueError(f'evaluation label mismatch: {key}')
    values = finite_array([[float(t[k]['prediction']) for k in keys] for t in tables], 2)
    return np.asarray(keys), values, {k: labels[k] for k in keys}


def compare_csvs(candidate_files, baseline_files, families_file, draws=10000):
    keys, candidate, labels = load_five_predictions(candidate_files)
    families = _rows(families_file)
    if set(families) != set(keys) or any(not row.get('group') for row in families.values()):
        raise ValueError('family file must cover every candidate sample exactly')
    y = np.array([labels[k] for k in keys])
    groups = np.array([families[k]['group'] for k in keys])
    loaded, common = {}, set(keys)
    coverage, full_metrics = {}, {}
    for name, files in baseline_files.items():
        bkeys, prediction, _ = load_five_predictions(files, labels)
        loaded[name] = (bkeys, prediction)
        common &= set(bkeys)
        missing = sorted(set(keys)-set(bkeys))
        coverage[name] = {'count': len(bkeys), 'total': len(keys), 'fraction': len(bkeys)/len(keys),
                          'missing_keys': missing, 'missing_acid': sum(labels[k] <= 4 for k in missing),
                          'missing_alkaline': sum(labels[k] >= 10 for k in missing)}
        full_metrics[name] = seed_summary([labels[k] for k in bkeys], prediction, [families[k]['group'] for k in bkeys])
    if not common:
        raise ValueError('no shared evaluation samples')
    ckeys = sorted(common)
    order = {k: i for i, k in enumerate(keys)}
    c = candidate[:, [order[k] for k in ckeys]]
    comparisons = {}
    for name, (bkeys, prediction) in loaded.items():
        order = {k: i for i, k in enumerate(bkeys)}
        comparisons[name] = prediction[:, [order[k] for k in ckeys]]
    cy = np.array([labels[k] for k in ckeys]); cg = np.array([families[k]['group'] for k in ckeys])
    candidate_metrics = seed_summary(y, candidate, groups)
    primary = 'baseline' if 'baseline' in loaded else next(iter(loaded))
    gate = acceptance(candidate_metrics['mean'], full_metrics[primary]['mean']) if coverage[primary]['count'] == len(keys) else {
        'passed': False, 'failures': ['primary_baseline_incomplete_coverage']}
    return {'seeds': list(SEEDS), 'seed_aggregation': 'mean of per-seed metrics',
            'candidate': candidate_metrics, 'baselines_on_available_samples': full_metrics,
            'acceptance': gate, 'coverage': coverage,
            'common_samples': {'count': len(ckeys), 'keys_sha256': stable_hash(ckeys),
                'candidate': seed_summary(cy, c, cg),
                'baselines': {k: seed_summary(cy, v, cg) for k, v in comparisons.items()}},
            'bootstrap': paired_family_bootstrap(cy, c, comparisons, cg, draws),
            'provenance': {'candidate_files': {str(p): sha256_file(p) for p in candidate_files},
                'baseline_files': {k: {str(p): sha256_file(p) for p in v} for k, v in baseline_files.items()},
                'families_sha256': sha256_file(families_file)},
            'field_leadership_established': False, 'default_model_replaced': False}


def sequence_families(records, output):
    """Label-free test-family units, with no singleton-ID fallback on search failure."""
    from phgeofuse.retrieval import _mmseqs_hits, _safe_key, record_key
    import shutil
    import subprocess
    output = Path(output)
    os.environ['PATH'] = str(Path(os.sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    binary = shutil.which('mmseqs')
    if binary is None:
        raise FileNotFoundError('MMseqs is required for paired family evaluation')
    version = subprocess.run([binary, 'version'], capture_output=True, text=True, check=True).stdout.strip()
    keys = [record_key(r) for r in records]
    metadata = {'sequence_keys': {record_key(r): r.sequence_sha256 for r in records},
                'mmseqs_version': version, 'identity': .3, 'qcov': .8, 'tcov': .8,
                'search_sensitivity': 7.5, 'max_seqs': len(records),
                'grouping': 'connected components of observed test-test links plus identical sequences',
                'labels_used': False, 'search_is_heuristic': True}
    freeze_json(output.with_suffix('.provenance.json'), metadata)
    if output.exists():
        table = _rows(output)
        if set(table) != set(keys):
            raise ValueError('cached family coverage differs')
        return np.array([table[k]['group'] for k in keys])
    anonymous = [replace(r, ph_opt=float('nan'), ec='', organism='', sample_weight=1.) for r in records]
    hits = _mmseqs_hits(anonymous, anonymous, {'retrieval': {'require_mmseqs': True,
        'candidate_k': len(records), 'search_threads': 8, 'search_sensitivity': 7.5}})
    safe = {_safe_key(r): record_key(r) for r in records}
    parent = {k: k for k in keys}
    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]; k = parent[k]
        return k
    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)
    sequences = {}
    for record in records:
        key = record_key(record)
        if record.sequence_sha256 in sequences:
            union(key, sequences[record.sequence_sha256])
        sequences[record.sequence_sha256] = key
    links = []
    for (a, b), h in hits.items():
        if h.identity >= .3 and min(h.query_coverage, h.target_coverage) >= .8:
            union(safe[a], safe[b])
            links.append([safe[a], safe[b], h.identity, h.query_coverage, h.target_coverage])
    groups = np.array([find(k) for k in keys])
    write_predictions(output, keys, {'group': groups})
    atomic_json(output.with_suffix('.links.json'), {'links': links, 'family_count': len(set(groups))})
    return groups


def verify_release(config_path, output):
    """Verify gates and dependencies BEFORE opening any test feature/prediction file."""
    from .inference import verify_dependencies
    output = Path(output)
    release = json.loads((output / 'frozen_release.json').read_text())
    if not release.get('ready_for_followup_test') or not release.get('internal_acceptance', {}).get('passed'):
        raise ValueError('no accepted, validation-checked frozen release; test access denied')
    protocol = json.loads((output / 'protocol.json').read_text())
    if release.get('protocol_sha256') != sha256_file(output / 'protocol.json'):
        raise ValueError('release protocol checksum differs')
    if sha256_file(config_path) != protocol['inputs']['files'][str(Path(config_path).resolve())]:
        raise ValueError('configuration changed after protocol freeze')
    for filename,digest in protocol['inputs']['files'].items():
        if sha256_file(filename)!=digest:
            raise ValueError(f'input changed since protocol freeze: {filename}')
    if sha256_file(output/'feature_assets.json')!=protocol['feature_assets_sha256']:
        raise ValueError('feature asset inventory changed since protocol freeze')
    for filename,digest in json.loads((output/'feature_assets.json').read_text()).items():
        if sha256_file(filename)!=digest:raise ValueError(f'feature asset changed: {filename}')
    if release.get('controls_release_sha256')!=sha256_file(output/'comparisons/frozen_release.json'):
        raise ValueError('control release changed since freeze')
    for filename, digest in protocol['source_hashes'].items():
        source = Path(protocol['source_root']) / filename
        if sha256_file(source) != digest:
            raise ValueError(f'implementation changed since freeze: {filename}')
    if [r['seed'] for r in release['models']] != list(SEEDS):
        raise ValueError('frozen release does not contain the five predeclared seeds')
    for row in release['models']:
        directory = Path(row['path'])
        if sha256_file(directory / 'model.json') != row['model_sha256']:
            raise ValueError('frozen model metadata changed')
        load_bundle(directory, 'cpu')
        saved = json.loads((directory / 'model.json').read_text())
        if sha256_file(directory / 'baseline.json') != saved['baseline_manifest_sha256']:
            raise ValueError('frozen baseline manifest changed')
        verify_dependencies(json.loads((directory / 'baseline.json').read_text()))
    return release


def followup_test(config_path, output):
    from .inference import adapt_features
    from .experiment import status
    from phgeofuse.retrieval import RetrievalStore
    output = Path(output)
    release = verify_release(config_path, output)
    config = load_config(config_path)
    data = DevelopmentData.load(config_path, test=True)
    directory = output / 'followup_test'
    directory.mkdir(parents=True, exist_ok=True)
    groups = sequence_families(data.records, directory / 'families.csv')
    baseline_files, candidate_files, arrays = [], [], []
    baseline_by_seed={}
    research = path(config, 'paths.source_experiment')
    store = RetrievalStore.load(Path(config['_root']) / 'artifacts/phgeofuse/retrieval.pt')
    retrieval = np.array([store.features(k).numpy() for k in data.keys])
    low = ~((retrieval[:, 4] >= .2) & (retrieval[:, 9] >= .8) & (retrieval[:, 10] >= .8))
    for row in release['models']:
        seed, bundle = row['seed'], Path(row['path'])
        source = research / f'dual_test/seed{seed}.csv'
        bkeys, bmat, labels = load_five_predictions([source]*5)
        order = {k: i for i, k in enumerate(bkeys)}
        if set(order) != set(data.keys) or any(abs(labels[k]-data.labels[i]) > 1e-8 for i, k in enumerate(data.keys)):
            raise ValueError('historical baseline test labels/coverage changed')
        baseline = bmat[0, [order[k] for k in data.keys]]
        baseline_by_seed[seed]=baseline
        # Recomputed historical files are fixed by the pre-test protocol hash.
        expected = release['baseline_test_files'][str(source)]
        if sha256_file(source) != expected:
            raise ValueError('frozen complete-baseline test predictions changed')
        predictor, saved = load_bundle(bundle, config['training']['device'])
        x = data.x
        if saved['recipe'].get('representation') == 'lora':
            x = adapt_features(bundle, data.records, x, directory / 'esm2_prefix', config['training']['device'])
        prediction = predictor.predict(x, baseline, saved['strength'], data.keys)
        candidate = directory / f'seed{seed}.csv'
        base = directory / f'baseline_seed{seed}.csv'
        shared = {'label': data.labels, 'seed': np.full(len(data.keys), seed), 'group': groups, 'low_homology': low}
        write_predictions(candidate, data.keys, {**shared, **prediction})
        write_predictions(base, data.keys, {**shared, 'prediction': baseline})
        candidate_files.append(candidate); baseline_files.append(base); arrays.append(prediction['prediction'])
        del predictor
    from .comparisons import predict_controls,field_predictions,FIELD_METHODS
    controlled,extra=field_predictions(release.get('field_manifests',{}))
    controls=predict_controls(data,output,retrieval,baseline_by_seed,groups)
    result = compare_csvs(candidate_files, {'baseline': baseline_files,**controls,**controlled}, directory / 'families.csv', config['protocol']['bootstrap_draws'])
    if extra:
        result['extra_task_pretraining_separate']=compare_csvs(candidate_files,extra,directory/'families.csv',config['protocol']['bootstrap_draws'])
    fixed_failures = []
    for name, limit in TEST_LIMITS.items():
        group, metric = name.split('.')
        if result['candidate']['mean'][group][metric] > limit:
            fixed_failures.append(name)
    result['fixed_test_limits'] = TEST_LIMITS
    result['acceptance']['failures'] = sorted(set(result['acceptance']['failures']+fixed_failures))
    result['acceptance']['passed'] = not result['acceptance']['failures']
    result['missing_field_comparisons']=sorted(set(FIELD_METHODS)-set(controlled))
    result['field_leadership_established']=not result['missing_field_comparisons'] and result['acceptance']['passed'] and all(
        result['coverage'][name]['fraction']==1 for name in FIELD_METHODS) and all(
        result['bootstrap']['comparisons'][name][tail][metric]['supported_improvement']
        for name in FIELD_METHODS for tail in ('acid','alkaline') for metric in ('rmse','mae','abs_bias'))
    result['promotion_eligible']=result['acceptance']['passed'] and result['field_leadership_established']
    result['low_homology'] = seed_summary(data.labels[low], np.asarray(arrays)[:, low], groups[low]) if low.any() else None
    result.update(evaluation_role='follow-up: PHOPT test was inspected historically',
                  test_used_for_selection=False, frozen_release_sha256=sha256_file(output / 'frozen_release.json'))
    atomic_npz(directory / 'predictions.npz', keys=data.keys, y=data.labels, groups=groups,
               low_homology=low, candidate=np.asarray(arrays))
    atomic_json(directory / 'results.json', result)
    status(output, 'followup_test_complete', acceptance=result['acceptance'],
           field_leadership_established=result['field_leadership_established'],default_model_replaced=False)
    return result
