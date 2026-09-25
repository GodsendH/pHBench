"""Recompute published predictions and document metric definitions, without fitting."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/metric_definition_audit_20260919'
SOURCE = ROOT/'docs/extreme_ph_review_20260916'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def regression(y, p):
    e = p-y
    return dict(n=len(y), rmse=float(np.sqrt(np.mean(e**2))), mae=float(np.mean(abs(e))),
                bias=float(np.mean(e)), p90_absolute_error=float(np.quantile(abs(e), .9)),
                large_error_ge2_count=int(np.sum(abs(e) >= 2)),
                large_error_ge2_rate=float(np.mean(abs(e) >= 2)))


def classification(y, p, weights=None, *, paper_literal=False):
    if paper_literal:
        classes = lambda z: np.where(z < 5, 0, np.where(z > 9, 2, 1))
    else:
        classes = lambda z: np.digitize(z, [5, 9])
    yt, yp = classes(y), classes(p)
    return dict(true_counts=np.bincount(yt, minlength=3).tolist(),
                macro_f1=float(f1_score(yt, yp, average='macro', sample_weight=weights)),
                per_class_f1=f1_score(yt, yp, average=None, sample_weight=weights).tolist(),
                per_class_precision=precision_score(yt, yp, average=None, sample_weight=weights, zero_division=0).tolist(),
                per_class_recall=recall_score(yt, yp, average=None, sample_weight=weights).tolist(),
                confusion_matrix=confusion_matrix(yt, yp, sample_weight=weights).tolist())


def main():
    fresh = OUT/'primary_sources/prediction.csv'
    archived = SOURCE/'ephod_official_example_prediction.csv'
    assert fresh.read_bytes() == archived.read_bytes()
    assert (OUT/'primary_sources/trainutils.py').read_bytes() == (SOURCE/'ephod_official_ephod_training_trainutils.py').read_bytes()
    public = pd.read_csv(fresh, index_col=0)
    aligned = pd.read_csv(OUT/'ephod_recheck/predictions.csv').set_index('key')
    keys = aligned.index.tolist(); y = aligned.label.to_numpy()
    predictions = {'EpHod_official_ensemble': public.loc[[k.split('::', 1)[1] for k in keys], 'Ensemble'].to_numpy()}
    np.testing.assert_allclose(predictions['EpHod_official_ensemble'], aligned.Ensemble, rtol=0, atol=1e-12)
    paths = [fresh, archived, OUT/'primary_sources/trainutils.py', OUT/'ephod_recheck/results.json',
             OUT/'ephod_recheck/predictions.csv']
    for name, path in [('Original_dual_seed42', ROOT/'experiments/phgeofuse_redesign_20260914/dual_test/seed42.csv'),
                       ('K1_s0.75_seed42', ROOT/'experiments/dual_tail_priority_20260919/test/K1_s0.75.csv')]:
        df = pd.read_csv(path).set_index('key')
        assert set(df.index) == set(keys) and df.index.is_unique
        df = df.loc[keys]; np.testing.assert_allclose(df.label, y, rtol=0, atol=1e-12)
        predictions[name] = df.prediction.to_numpy(); paths.append(path)
    yc = np.digitize(y, [5, 9]); counts = np.bincount(yc, minlength=3)
    w = 1/counts[yc]; w /= w.mean()
    masks = dict(all=np.ones(len(y), bool), strict_acid_le4=y <= 4,
                 strict_alkaline_ge10=y >= 10, official_acid_lt5=y < 5,
                 official_middle_5to9=(y >= 5)&(y < 9), official_alkaline_ge9=y >= 9,
                 paper_literal_middle=(y >= 5)&(y <= 9), paper_literal_alkaline_gt9=y > 9)
    result = dict(scope='Metric source audit during paused model iteration. No training, selection, or new prediction inference.',
        primary_source_commit='e823cd2f1172258dc1e81cc00326e6975f22d10a',
        fresh_official_download_matches_archived_bytes=True,
        official_code_boundaries=['pH < 5', '5 <= pH < 9', 'pH >= 9'],
        boundary_counts=dict(y_equal_5=int(np.sum(y == 5)), y_equal_9=int(np.sum(y == 9))),
        interval_counts={k: int(v.sum()) for k, v in masks.items()},
        models={})
    for name, p in predictions.items():
        result['models'][name] = dict(
            unweighted_regression={k: regression(y[mask], p[mask]) for k, mask in masks.items()},
            unweighted_classification_official_boundaries=classification(y, p),
            bin_inverse_sample_weighted_classification=classification(y, p, w),
            bin_inverse_rmse=float(np.sqrt(np.average((p-y)**2, weights=w))),
            unweighted_classification_paper_literal_boundaries=classification(y, p, paper_literal=True))
    manifest = pd.read_csv(ROOT/'artifacts/phgeofuse/manifest.csv')
    result['split_counts'] = {}
    for split in ['train', 'validation', 'test']:
        yy = manifest.loc[manifest.split == split, 'ph_opt'].to_numpy()
        result['split_counts'][split] = dict(n=len(yy), strict_acid=int(np.sum(yy <= 4)), strict_alkaline=int(np.sum(yy >= 10)),
            official_classes=np.bincount(np.digitize(yy, [5, 9]), minlength=3).tolist())
    paths += [Path(__file__), ROOT/'artifacts/phgeofuse/manifest.csv']
    result['source_sha256'] = {str(p): sha(p) for p in paths}
    (OUT/'metric_audit.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(interval_counts=result['interval_counts'], boundary_counts=result['boundary_counts'],
                         split_counts=result['split_counts'], metrics={k: dict(
                             strict_acid=v['unweighted_regression']['strict_acid_le4'],
                             strict_alkaline=v['unweighted_regression']['strict_alkaline_ge10'],
                             f1_unweighted=v['unweighted_classification_official_boundaries']['macro_f1'],
                             f1_weighted=v['bin_inverse_sample_weighted_classification']['macro_f1'],
                             weighted_rmse=v['bin_inverse_rmse']) for k, v in result['models'].items()}), indent=2))


if __name__ == '__main__':
    main()
