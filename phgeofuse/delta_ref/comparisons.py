"""Controlled compact/Ridge baselines and a provenance-checked field registry.

Official task-pretrained weights are a different information condition. Missing
external reproductions remain explicit blockers for a field-leadership claim.
"""
from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from phgeofuse.cache import atomic_json, sha256_file
from phgeofuse.config import path
from phgeofuse.reliability_gate import ReliabilityGate
from phgeofuse.reliability_fusion import reliability_inputs
from .baseline import FullBaseline
from .data import DevelopmentData, atomic_npz, freeze_json, write_predictions
from .metrics import metrics
from .training import fit_subset, load_bundle

FIELD_METHODS = ('EpHod', 'Venus-DREAM', 'OpHReda')


def registry(root):
    root = Path(root)
    return {
        'full_baseline': {'status': 'implemented', 'information': 'historical PHOPT-only supervised fit'},
        'dual_ridge': {'status': 'implemented', 'information': 'PHOPT-only', 'alpha': .2,
                       'stochastic': False},
        'compact_residual': {'status': 'implemented', 'information': 'PHOPT-only',
                             'recipe': 'historical l3_i50_p0.0; subset cross-fitted inputs'},
        'direct_regression': {'status': 'implemented', 'information': 'PHOPT-only',
                              'selection': 'four head recipes; inner folds only'},
        'EpHod': {'status': 'not_reproduced', 'information': 'PHOPT-only reproduction required',
                  'reason': 'Local official RLAT weights include pHenv task supervision; the full controlled RLATtr+ESM1v-SVR five-seed experiment has not been run.',
                  'code': str(root/'baseline/EpHod'), 'source': 'https://github.com/jafetgado/EpHod'},
        'Venus-DREAM': {'status': 'not_reproduced', 'information': 'PHOPT-only reproduction required',
                        'reason': 'Code is present; an aligned original-split five-seed Reptile run with training-only support sets remains pending.',
                        'code': str(root/'reptile.py'), 'source': 'https://doi.org/10.1021/acs.jcim.4c02291'},
        'OpHReda': {'status': 'not_reproduced', 'information': 'PHOPT-only reproduction required',
                    'reason': 'Audited public inference code and three-stage training instructions; no verified local three-stage PHOPT-only checkpoint and reference database.',
                    'source': 'https://github.com/RIA-lab/OpHReda'},
        'EnzOracle': {'status': 'design_audit_only', 'source': 'https://doi.org/10.64898/2026.06.02.729708',
                      'evidence': 'abstract; classification-guided mixture of experts for extreme shrinkage'},
        'pHoptNN': {'status': 'design_audit_only', 'source': 'https://doi.org/10.1021/acs.jcim.6c01482',
                    'evidence': 'formal full text; atomic equivariant graph and charge features'},
        'DeepPH': {'status': 'design_audit_only', 'source': 'https://doi.org/10.1109/JBHI.2026.3729250',
                   'evidence': 'abstract and author README; sequence/structure/interval prediction'},
    }


def _crossfit_features(data, provider, excluded):
    fit = data.train[~np.isin(data.folds[data.train], excluded)]
    r = np.full((len(data.keys), 15), np.nan); s = np.full(len(data.keys), np.nan)
    for inner in sorted(set(data.folds[fit])):
        q, f = provider.base_features(sorted(set(excluded) | {int(inner)}))
        use = data.folds[q] == inner
        r[q[use]], s[q[use]] = f['retrieval'][use], f['sequence'][use]
    if not np.isfinite(r[fit]).all() or not np.isfinite(s[fit]).all():
        raise ValueError('incomplete compact-baseline cross-fit features')
    return fit, r[fit], s[fit]


def _fit_compact(r, s, chemistry, labels, seed):
    gate = ReliabilityGate(.01).fit(*reliability_inputs(r, s), labels)
    anchor = gate.predict(*reliability_inputs(r, s))
    residual = HistGradientBoostingRegressor(max_leaf_nodes=3, max_iter=50,
        min_samples_leaf=80, l2_regularization=30, learning_rate=.05,
        early_stopping=False, random_state=seed).fit(np.column_stack([r, s, chemistry]), labels-anchor)
    return gate, residual


def _compact_predict(gate, residual, r, s, chemistry):
    return gate.predict(*reliability_inputs(r, s)) + residual.predict(np.column_stack([r, s, chemistry]))


def nested_classical(data, output, seed=42):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    provider = FullBaseline(data, output.parent/'baseline'/f'seed{seed}', seed, data.config['training']['device'])
    predictions = {k: np.full(len(data.keys), np.nan) for k in ('dual_ridge', 'compact_residual')}
    for outer in range(5):
        fit, r, s = _crossfit_features(data, provider, [outer])
        q, held = provider.base_features([outer])
        gate, residual = _fit_compact(r, s, data.x[fit, -25:], data.labels[fit], seed)
        predictions['dual_ridge'][q] = held['sequence']
        predictions['compact_residual'][q] = _compact_predict(gate, residual, held['retrieval'], held['sequence'], data.x[q, -25:])
        joblib.dump({'gate': gate, 'residual': residual}, output/f'compact_outer{outer}.joblib')
        freeze_json(output/f'compact_outer{outer}.provenance.json', data.certificate(fit, q, [outer]))
    train = data.train
    result = {name: metrics(data.labels[train], p[train], data.groups[train]) for name, p in predictions.items()}
    atomic_npz(output/'predictions.npz', keys=data.keys[train], y=data.labels[train], groups=data.groups[train],
               **{k: v[train] for k, v in predictions.items()})
    atomic_json(output/'results.json', result)
    atomic_json(output/'field_registry.json', registry(data.config['_root']))
    return result


def finalize_controls(data, output):
    """Train and freeze control packages before allowing test access."""
    output = Path(output)
    if (output/'comparisons/frozen_release.json').exists():
        release=json.loads((output/'comparisons/frozen_release.json').read_text())
        for package in release['models']:
            for filename,digest in package['files'].items():
                if sha256_file(filename)!=digest:raise ValueError('frozen control package changed')
        return release
    provider = FullBaseline(data, output/'baseline/seed42', 42, data.config['training']['device'])
    fit, r, s = _crossfit_features(data, provider, [])
    if not np.array_equal(fit, data.train):
        raise ValueError('control training keys differ')
    direct = json.loads((output/'ablations/absolute/results.json').read_text())['final_selection']
    additive = json.loads((output/'ablations/additive/results.json').read_text())['final_selection']
    packages = []
    for seed in data.config['protocol']['seeds']:
        directory = output/'comparisons/final'/f'seed{seed}'
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {**data.certificate(data.train, data.validation, []), 'seed': seed,
                    'compact_recipe': 'l3_i50_p0.0', 'ridge_alpha': .2}
        freeze_json(directory/'fit.json', metadata)
        sequence = Ridge(alpha=.2, solver='cholesky').fit(data.embeddings[fit], data.labels[fit])
        gate, residual = _fit_compact(r, s, data.x[fit, -25:], data.labels[fit], seed)
        for name, model in [('sequence', sequence), ('gate', gate), ('residual', residual)]:
            joblib.dump(model, directory/f'{name}.joblib')
        for name, selected in [('direct', direct), ('additive', additive)]:
            fit_subset(data, data.train, data.validation, None, selected['recipe'], directory/name,
                       seed=seed, fixed_epochs=selected['epochs'], excluded=[])
        files = [*directory.glob('*.joblib'), directory/'fit.json', directory/'direct/model.json', directory/'direct/weights.pt',
                 directory/'additive/model.json', directory/'additive/weights.pt']
        packages.append({'seed': seed, 'path': str(directory), 'direct_strength': direct['strength'],
                         'additive_strength': additive['strength'], 'files': {str(p): sha256_file(p) for p in files}})
    release = {'models': packages, 'field_registry': registry(data.config['_root']),
               'test_access': False, 'labels': 'original PHOPT training only',
               'deterministic_ridge_repeated_across_seeds': True}
    freeze_json(output/'comparisons/frozen_release.json', release)
    return release


def predict_controls(data, output, retrieval, baseline_by_seed, groups):
    output = Path(output)
    release = json.loads((output/'comparisons/frozen_release.json').read_text())
    paths = {name: [] for name in ('dual_ridge', 'compact_residual', 'direct_regression', 'direct_with_anchor', 'additive_difference')}
    for package in release['models']:
        for p, digest in package['files'].items():
            if sha256_file(p) != digest:
                raise ValueError('control changed after pre-test freeze')
        seed = package['seed']; directory = Path(package['path'])
        sequence = joblib.load(directory/'sequence.joblib').predict(data.embeddings)
        gate, residual = [joblib.load(directory/f'{n}.joblib') for n in ('gate', 'residual')]
        predictions = {'dual_ridge': sequence, 'compact_residual': _compact_predict(gate, residual, retrieval, sequence, data.x[:, -25:])}
        direct, _ = load_bundle(directory/'direct', data.config['training']['device'])
        predictions['direct_regression'] = np.clip(direct.transfer(data.x, data.keys)[0], 0, 14)
        predictions['direct_with_anchor'] = direct.predict(data.x, baseline_by_seed[seed], package['direct_strength'], data.keys)['prediction']
        additive, _ = load_bundle(directory/'additive', data.config['training']['device'])
        predictions['additive_difference'] = additive.predict(data.x, baseline_by_seed[seed], package['additive_strength'], data.keys)['prediction']
        for name, prediction in predictions.items():
            destination = output/'followup_test'/f'{name}_seed{seed}.csv'
            write_predictions(destination, data.keys, {'label': data.labels, 'seed': np.full(len(data.keys), seed),
                                                       'group': groups, 'prediction': prediction})
            paths[name].append(destination)
    return paths


def import_field_manifest(source, output, config_path):
    """Register independently reproduced outputs with explicit supervision scope.

    The manifest records source/checkpoint hashes, original split hashes, seeds,
    retrieval-fit keys, exclusions and paths. Import never infers these facts
    from a model name or from paper scores.
    """
    from phgeofuse.config import load_config
    config = load_config(config_path)
    source = Path(source).resolve()
    payload = json.loads(source.read_text())
    if payload.get('method') not in FIELD_METHODS:
        raise ValueError('unknown required field comparator')
    required = ('information_condition', 'training_keys_sha256', 'manifest_sha256', 'source_hashes',
                'checkpoint_hashes', 'selection_protocol', 'prediction_files', 'seeds', 'retrieval_training_only')
    if any(k not in payload for k in required) or payload['seeds'] != [0, 1, 2, 3, 42]:
        raise ValueError('incomplete field reproduction provenance')
    if payload['manifest_sha256'] != sha256_file(path(config, 'paths.manifest')):
        raise ValueError('field comparator used a different data manifest')
    data = DevelopmentData.load(config_path)
    from .data import stable_hash
    if payload['training_keys_sha256'] != stable_hash(data.keys[data.train].tolist()):
        raise ValueError('field comparator used a different training set')
    if payload['information_condition'] not in ('PHOPT-only', 'extra-task-pretraining'):
        raise ValueError('supervision information condition must be explicit')
    if not payload['retrieval_training_only'] or len(payload['prediction_files']) != 5:
        raise ValueError('training-only reference provenance and five seed outputs required')
    for name in ('source_hashes', 'checkpoint_hashes'):
        if not payload[name]:
            raise ValueError('empty comparator hashes')
        for filename, digest in payload[name].items():
            if sha256_file(filename) != digest:
                raise ValueError('field comparator source/checkpoint checksum differs')
    payload['prediction_hashes'] = {p: sha256_file(p) for p in payload['prediction_files']}
    payload['manifest_source_sha256'] = sha256_file(source)
    target = Path(output)/'comparisons/field'/f"{payload['method']}_{payload['information_condition']}.json"
    freeze_json(target, payload)
    return target


def frozen_field_manifests(output):
    directory = Path(output)/'comparisons/field'
    return {str(p): sha256_file(p) for p in sorted(directory.glob('*.json'))}


def field_predictions(frozen_manifests):
    controlled, extra = {}, {}
    for filename, digest in frozen_manifests.items():
        if sha256_file(filename) != digest:
            raise ValueError('field manifest changed after freeze')
        payload = json.loads(Path(filename).read_text())
        for p, expected in payload['prediction_hashes'].items():
            if sha256_file(p) != expected:
                raise ValueError('field predictions changed after freeze')
        target = controlled if payload['information_condition'] == 'PHOPT-only' else extra
        target[payload['method']] = payload['prediction_files']
    return controlled, extra
