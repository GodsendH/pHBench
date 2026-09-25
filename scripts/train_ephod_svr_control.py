"""PHOPT-only EpHod-SVR component, with official search/weights and frozen test.

This is the deterministic SVR component, not the complete RLATtr+SVR ensemble.
Training consumes only train/validation labels. Test evaluation is a separate
phase after a checksum-bound model release. Precomputed kernels reuse identical
pairwise calculations across the official 200-candidate search.
"""
import argparse
import csv
import fcntl
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import sklearn
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phgeofuse.delta_ref.metrics import metrics

PUBLIC_COMMIT = 'e823cd2f1172258dc1e81cc00326e6975f22d10a'
PUBLIC_WEIGHTS_BLOB = '8274d90991d155437a416138f7cbf4f29a242089'
WEIGHTS = ['bin_inv', 'bin_inv_sqrt', 'LDS_inv', 'LDS_inv_sqrt', 'LDS_extreme']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def frozen_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Frozen comparator state changed: {path}')
    else:
        write_json(path, value)


def weight_function(source):
    content = source.read_bytes()
    blob = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
    if blob != PUBLIC_WEIGHTS_BLOB:
        raise ValueError('Official EpHod weight-function source differs from audited commit.')
    spec = importlib.util.spec_from_file_location('audited_ephod_weights', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_sample_weights


def kernel_values(query, train, kind, gamma):
    if kind == 'rbf':
        return rbf_kernel(query, train, gamma=gamma)
    if kind == 'poly':
        return polynomial_kernel(query, train, degree=3, gamma=gamma, coef0=0)
    raise ValueError('Unexpected kernel')


def grid():
    return [dict(kernel=k, gamma=g, C=float(c), weight_type=w)
            for k, g, c, w in itertools.product(
                ['poly', 'rbf'], ['scale', 'auto'], 10. ** np.arange(-5, 5), WEIGHTS)]


def load_features(cache, manifest, *, include_test_labels=False):
    complete = json.loads((cache / 'complete.json').read_text())
    if sha(cache / 'protocol.json') != complete['protocol_sha256']:
        raise ValueError('Embedding protocol changed after completion.')
    if sha(cache / 'features.npz') != complete['features_sha256']:
        raise ValueError('Embedding features changed after completion.')
    protocol = json.loads((cache / 'protocol.json').read_text())
    if sha(manifest) != protocol['manifest_sha256'] or protocol['labels_consumed']:
        raise ValueError('Embedding provenance does not match PHOPT.')
    with manifest.open() as stream:
        records = list(csv.DictReader(stream))
    keys = np.array([r['split'] + '::' + r['protein_id'] for r in records])
    split = np.array([r['split'] for r in records])
    labels = np.array([
        float(r['ph_opt']) if r['split'] != 'test' or include_test_labels else np.nan
        for r in records
    ])
    if {s: int(sum(split == s)) for s in ['train', 'validation', 'test']} != {
        'train': 7124, 'validation': 760, 'test': 1971,
    }:
        raise ValueError('Expected original PHOPT partition sizes.')
    with np.load(cache / 'features.npz', allow_pickle=False) as data:
        if not np.array_equal(data['keys'], keys):
            raise ValueError('Embedding sample order differs from manifest.')
        features = data['token_mean'].astype(np.float64)
    if features.shape != (len(keys), 1280) or not np.isfinite(features).all():
        raise ValueError('Invalid ESM1v token means.')
    return keys, split, labels, features


def wait_for_cache(cache):
    while not (cache / 'complete.json').exists():
        pidfile = cache / 'runner.pid.json'
        if not pidfile.exists():
            raise FileNotFoundError('Encoding has not been launched.')
        process = json.loads(pidfile.read_text())
        cmdline = Path('/proc') / str(process['pid']) / 'cmdline'
        if not cmdline.exists() or 'cache_phopt_field_embeddings.py' not in cmdline.read_bytes().decode():
            raise RuntimeError('Encoder is no longer running and has no completion artifact.')
        print(json.dumps({'state': 'waiting_for_encoder', 'encoder_pid': process['pid'],
                          'pid': os.getpid(), 'updated': time.time()}), flush=True)
        time.sleep(30)


def train(args):
    output, cache = args.output.resolve(), args.cache.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.wait_for_cache:
            wait_for_cache(cache)
        keys, split, y, x = load_features(cache, args.manifest)
        training, validation = split == 'train', split == 'validation'
        mean, std = x[training].mean(0), x[training].std(0)
        xtrain = np.ascontiguousarray((x[training] - mean) / (std + 1e-8))
        xval = np.ascontiguousarray((x[validation] - mean) / (std + 1e-8))
        get_weights = weight_function(args.weight_source)
        weights = {w: get_weights(y[training], w) for w in WEIGHTS}
        valweights = get_weights(y[validation], 'bin_inv')
        if any(not np.isfinite(w).all() or not (w > 0).all() for w in [*weights.values(), valweights]):
            raise ValueError('Invalid official sample weights.')
        protocol = {
            'method': 'EpHod-SVR component', 'information_condition': 'PHOPT-only',
            'cache_path': str(cache), 'manifest_path': str(args.manifest.resolve()),
            'stochastic': False, 'public_commit': PUBLIC_COMMIT,
            'selection': 'minimum official bin_inv weighted validation RMSE; full 200-recipe grid',
            'tie_break': 'unweighted validation RMSE, then deterministic recipe index',
            'reported_metrics': 'unweighted; same evaluator as DeltaRef',
            'test_used_for_selection': False, 'train_count': int(training.sum()),
            'training_keys': keys[training].tolist(), 'validation_keys': keys[validation].tolist(),
            'recipes': grid(), 'SVR': {'epsilon': 0.1, 'tol': 0.001, 'shrinking': True, 'max_iter': -1},
            'normalization': 'training population mean/std + 1e-8',
            'pooling': 'all sequence tokens including BOS/EOS; batch size one',
            'sklearn': sklearn.__version__, 'numpy': np.__version__,
            'source_sha256': {str(p.resolve()): sha(p) for p in [
                Path(__file__), args.weight_source, ROOT / 'phgeofuse/delta_ref/metrics.py',
                args.manifest, cache / 'features.npz', cache / 'protocol.json',
            ]},
        }
        frozen_json(output / 'protocol.json', protocol)
        if (output / 'release.json').exists():
            print('SVR_RELEASE_ALREADY_FROZEN', flush=True)
            return
        rank = None
        winner = None
        rows = []
        old_kernel = None
        started = time.time()
        for i, recipe in enumerate(grid()):
            kernel_id = (recipe['kernel'], recipe['gamma'])
            gamma = 1 / (xtrain.shape[1] * xtrain.var()) if recipe['gamma'] == 'scale' else 1 / xtrain.shape[1]
            if kernel_id != old_kernel:
                ktrain = kernel_values(xtrain, xtrain, recipe['kernel'], gamma)
                kval = kernel_values(xval, xtrain, recipe['kernel'], gamma)
                old_kernel = kernel_id
            record_file = output / f'recipe{i:03d}.json'
            model_file = output / f'recipe{i:03d}.joblib'
            if record_file.exists() and model_file.exists():
                row = json.loads(record_file.read_text())
                if row['recipe'] != recipe or sha(model_file) != row['checkpoint_sha256']:
                    raise ValueError('Completed comparator recipe changed.')
            else:
                write_json(output / 'status.json', {'state': 'fit', 'recipe_index': i,
                    'total_recipes': 200, 'recipe': recipe, 'pid': os.getpid(), 'updated': time.time()})
                tick = time.time()
                model = SVR(kernel='precomputed', C=recipe['C'], epsilon=0.1,
                            tol=0.001, shrinking=True, cache_size=512, max_iter=-1)
                model.fit(ktrain, y[training], sample_weight=weights[recipe['weight_type']])
                if model.fit_status_ != 0:
                    raise RuntimeError('SVR did not converge.')
                prediction = model.predict(kval)
                error = prediction - y[validation]
                score = float(np.sqrt(np.average(error ** 2, weights=valweights)))
                joblib.dump(model, model_file)
                row = {'recipe_index': i, 'recipe': recipe, 'gamma_value': float(gamma),
                       'selection_rmse': score, 'validation': metrics(y[validation], prediction),
                       'seconds': time.time() - tick, 'checkpoint_sha256': sha(model_file)}
                write_json(record_file, row)
                print(json.dumps({'recipe_complete': i + 1, 'weighted_validation_rmse': score,
                                  'seconds': row['seconds']}), flush=True)
            rows.append(row)
            candidate = (row['selection_rmse'], row['validation']['all']['rmse'], i)
            if rank is None or candidate < rank:
                rank, winner = candidate, row
        selected = joblib.load(output / f"recipe{winner['recipe_index']:03d}.joblib")
        package = output / 'model.joblib'
        joblib.dump({'model': selected, 'train_features': xtrain, 'mean': mean, 'std': std,
                     'recipe': winner['recipe'], 'gamma': winner['gamma_value']}, package)
        release = {'method': 'EpHod-SVR component', 'stochastic': False,
                   'independent_fits': 1, 'selected': winner, 'all_recipes': rows,
                   'protocol_sha256': sha(output / 'protocol.json'),
                   'checkpoint_sha256': sha(package), 'test_used_for_selection': False,
                   'elapsed_seconds_this_invocation': time.time() - started}
        frozen_json(output / 'release.json', release)
        write_json(output / 'status.json', {'state': 'frozen_before_test', 'pid': os.getpid(), 'updated': time.time()})
        print('SVR_CONTROL_FROZEN_BEFORE_TEST', flush=True)


def evaluate(args):
    output = args.output.resolve()
    release = json.loads((output / 'release.json').read_text())
    if (sha(output / 'protocol.json') != release['protocol_sha256'] or
            sha(output / 'model.joblib') != release['checkpoint_sha256']):
        raise ValueError('Comparator release changed before evaluation.')
    protocol = json.loads((output / 'protocol.json').read_text())
    if (str(args.cache.resolve()) != protocol['cache_path'] or
            str(args.manifest.resolve()) != protocol['manifest_path']):
        raise ValueError('Evaluation inputs must match the frozen comparator paths.')
    for filename, digest in protocol['source_sha256'].items():
        if sha(filename) != digest:
            raise ValueError(f'Comparator provenance changed: {filename}')
    keys, split, y, x = load_features(args.cache, args.manifest, include_test_labels=True)
    package = joblib.load(output / 'model.joblib')
    test = split == 'test'
    x = (x[test] - package['mean']) / (package['std'] + 1e-8)
    k = kernel_values(x, package['train_features'], package['recipe']['kernel'], package['gamma'])
    prediction = package['model'].predict(k)
    destination = output / 'followup_test'
    destination.mkdir(exist_ok=False)
    with (destination / 'predictions.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['key', 'label', 'prediction'])
        writer.writerows(zip(keys[test], y[test], prediction))
    write_json(destination / 'results.json', {'method': 'EpHod-SVR component',
        'information_condition': 'PHOPT-only', 'independent_fits': 1, 'stochastic': False,
        'test_role': 'follow-up; historically inspected', 'metrics': metrics(y[test], prediction),
        'release_sha256': sha(output / 'release.json'), 'predictions_sha256': sha(destination / 'predictions.csv')})
    print(json.dumps(metrics(y[test], prediction)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'artifacts/phgeofuse/manifest.csv')
    parser.add_argument('--weight-source', type=Path, default=ROOT / 'docs/extreme_ph_review_20260916/ephod_official_ephod_training_trainutils.py')
    parser.add_argument('--phase', choices=['train', 'evaluate'], default='train')
    parser.add_argument('--wait-for-cache', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        (train if args.phase == 'train' else evaluate)(args)


if __name__ == '__main__':
    main()
