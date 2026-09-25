"""Fixed-recipe, grouped development plus one frozen PHOPT follow-up test.

The primary R4b recipe is fixed before results. Eight OOF comparisons do not
select a new test recipe. All 7,124 training records fit the final dual expert;
the encoders and historical robust branch stay frozen, as specified in PLAN_ZH.
"""
from __future__ import annotations

import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import argparse
import itertools
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phgeofuse.cache import atomic_json, sha256_file
from phgeofuse.delta_ref.data import DevelopmentData, atomic_npz, freeze_json, read_predictions, write_predictions
from phgeofuse.dual_fusion import DualFusion, retrieval_sequence_anchor as anchor
from phgeofuse.io import read_fasta
from phgeofuse.retrieval import RetrievalStore
from phgeofuse.robust_train import frequency_weights
from phgeofuse.tail_weighting import tail_weights, residual_target, crossfit_rows

SOURCE = ROOT / 'experiments/phgeofuse_redesign_20260914'
CACHE = ROOT / 'experiments/delta_ref_phopt_20260916/baseline/seed42'
OLD_BUNDLE = SOURCE / 'dual_candidate_float64'
PRIMARY = 'R4b'
RECIPES = [
    dict(name='R0', ridge_lambda=0., residual_lambda=None, target='branch'),
    dict(name='R1', ridge_lambda=0., residual_lambda=0., target='branch'),
    dict(name='R2a', ridge_lambda=0., residual_lambda=.05, target='branch'),
    dict(name='R2b', ridge_lambda=0., residual_lambda=.10, target='branch'),
    dict(name='R3', ridge_lambda=0., residual_lambda=0., target='complete'),
    dict(name='R4a', ridge_lambda=0., residual_lambda=.05, target='complete'),
    dict(name='R4b', ridge_lambda=0., residual_lambda=.10, target='complete'),
    dict(name='R5', ridge_lambda=.05, residual_lambda=.10, target='complete'),
]
HGB = dict(max_leaf_nodes=7, max_iter=50, min_samples_leaf=80,
           l2_regularization=30, learning_rate=.05, early_stopping=False, random_state=42)


def regions(y, low):
    return {'all': np.ones(len(y), bool), 'extreme_acid': y <= 4,
            'core': (y > 4) & (y < 10), 'extreme_alkaline': y >= 10,
            'acid_le5': y <= 5, 'alkaline_ge9': y >= 9,
            'acid_lt6': y < 6, 'neutral_6to8': (y >= 6) & (y <= 8),
            'alkaline_gt8': y > 8, 'low_homology': low}


def metrics(y, p, low, groups=None):
    if y.shape != p.shape or not np.isfinite(p).all():
        raise ValueError('nonfinite or misaligned predictions')
    out = {}
    for name, mask in regions(y, low).items():
        if not mask.any():
            out[name] = {'count': 0}
            continue
        e = p[mask] - y[mask]
        out[name] = dict(count=int(mask.sum()), rmse=float(np.sqrt(np.mean(e**2))),
                         mae=float(np.mean(abs(e))), bias=float(e.mean()))
        if groups is not None:
            out[name]['groups'] = len(set(groups[mask]))
    yy = y-y.mean()
    out['all'].update(pearson=float(pearsonr(y, p).statistic),
                      spearman=float(spearmanr(y, p).statistic),
                      r2=float(1.-np.sum((p-y)**2)/np.sum(yy**2)))
    core = (y > 4) & (y < 10)
    out['false_extreme_rate'] = float(np.mean((p[core] <= 4) | (p[core] >= 10)))
    return out


def comparison(base, candidate):
    checks = {f'{g}_rmse_tolerance': candidate[g]['rmse'] <= base[g]['rmse']+.01
              for g in ['all', 'core', 'low_homology']}
    checks['all_mae_tolerance'] = candidate['all']['mae'] <= base['all']['mae']+.01
    checks['false_extreme_tolerance'] = candidate['false_extreme_rate'] <= base['false_extreme_rate']+.005
    guardrails = all(checks.values())
    for g in ['extreme_acid', 'extreme_alkaline']:
        checks[g+'_5pct_rmse_gain'] = candidate[g]['rmse'] <= base[g]['rmse']*.95
        checks[g+'_mae'] = candidate[g]['mae'] <= base[g]['mae']
        checks[g+'_abs_bias'] = abs(candidate[g]['bias']) < abs(base[g]['bias'])
    score = float(np.mean([(candidate[g]['rmse']/base[g]['rmse'])**2
                          for g in ['extreme_acid', 'extreme_alkaline']]))
    return dict(checks=checks, guardrails_passed=guardrails, metrics_passed=all(checks.values()),
                tail_relative_mse=score,
                delta={g: {k: candidate[g][k]-base[g][k] for k in ['rmse', 'mae', 'bias']}
                       for g in regions(np.array([3., 7., 11.]), np.ones(3, bool))
                       if 'rmse' in candidate.get(g, {}) and 'rmse' in base.get(g, {})})


def bootstrap(y, p0, p1, low, groups, draws=3000):
    """Paired resampling of actual groups, with empty-tail draws omitted."""
    _, inv = np.unique(groups, return_inverse=True)
    ng = int(inv.max())+1
    sums = {}
    selected = ['all', 'extreme_acid', 'core', 'extreme_alkaline', 'low_homology']
    for name, mask in regions(y, low).items():
        if name not in selected:
            continue
        sums[name] = [np.bincount(inv, weights=a*mask, minlength=ng) for a in
                      [np.ones(len(y)), (p0-y)**2, (p1-y)**2, abs(p0-y), abs(p1-y), p0-y, p1-y]]
    rng = np.random.default_rng(42)
    values = {g: {m: [] for m in ['rmse', 'mae', 'bias']} for g in sums}
    for _ in range(draws):
        ix = rng.integers(ng, size=ng)
        for g, arrays in sums.items():
            n, se0, se1, ae0, ae1, e0, e1 = [a[ix].sum() for a in arrays]
            if n:
                values[g]['rmse'].append(np.sqrt(se1/n)-np.sqrt(se0/n))
                values[g]['mae'].append((ae1-ae0)/n)
                values[g]['bias'].append((e1-e0)/n)
    return {g: {m: dict(delta_95ci=np.quantile(v, [.025, .975]).tolist(), valid_draws=len(v))
                for m, v in measures.items()} for g, measures in values.items()}


class Experiment:
    def __init__(self, output):
        self.out = output
        output.mkdir(parents=True, exist_ok=False)
        self.start = time.monotonic()
        self.data = DevelopmentData.load(ROOT/'configs/delta_ref_phopt.yaml')
        d = self.data
        self.n = len(d.train)
        if not np.array_equal(d.train, np.arange(self.n)):
            raise ValueError('expected training-first feature order')
        official = read_fasta(ROOT/'data/phopt_training.fasta', 'train')
        signature = lambda rs: sorted((r.protein_id, r.sequence, r.ph_opt) for r in rs)
        if signature(official) != signature([d.records[i] for i in d.train]):
            raise ValueError('training manifest differs from official PHOPT')
        self.y, self.fold = d.labels[d.train], d.folds[d.train]
        self.chem = d.x[:, -25:]
        self.cache, self.weighted = {}, {}
        self.sources = dict(d.provenance['files'])
        self.parity = []
        self.protocol = dict(
            primary=PRIMARY, recipes=RECIPES, hgb=HGB, ridge_alpha=.2, dual_weight=.5,
            seed=42, data=d.provenance, training_count=self.n,
            planned_validation_count=760, planned_test_count=1971,
            scope='All PHOPT train rows refit dual Ridge/HGB; encoders and robust branch frozen. One seed.',
            primary_test_rule='R4b fixed a priori; evaluate once after freeze even if development guards fail, for user-requested diagnostic comparison. No deployment replacement.',
            secondary_rule='At most one non-primary candidate: rank train guardrails first, then tail relative MSE. Validation only; never test it in this run.',
            folds='Fixed five homology groups folds; upstream excluded outer and row fold. Fixed recipes; OOF selection is exploratory, not nested HPO.',
            thresholds=dict(acid_max=4, alkaline_min=10, rmse_tolerance=.01,
                            mae_tolerance=.01, false_extreme_tolerance=.005, tail_rmse_gain=.05),
            test_history='Previously accessed PHOPT test; this is a follow-up, not untouched confirmation.',
            runtime=dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
                         sklearn=sklearn.__version__, torch=torch.__version__),
            source_files={str(p): sha256_file(p) for p in [Path(__file__), ROOT/'phgeofuse/tail_weighting.py',
                ROOT/'phgeofuse/dual_fusion.py', ROOT/'phgeofuse/robust_fusion.py',
                ROOT/'docs/dual_loss_reweighting_20260919/PLAN_ZH.md']})
        freeze_json(output/'protocol.json', self.protocol)
        self.sources.update(self.protocol['source_files'])
        self.emit('protocol_frozen', primary=PRIMARY, recipes=len(RECIPES))

    def emit(self, event, **details):
        payload = dict(event=event, elapsed_seconds=time.monotonic()-self.start, pid=os.getpid(), **details)
        atomic_json(self.out/'status.json', payload)
        print(json.dumps(payload), flush=True)

    def record(self, p):
        self.sources[str(p)] = sha256_file(p)

    def cached(self, excluded):
        excluded = tuple(sorted(excluded))
        if excluded in self.cache:
            return self.cache[excluded]
        d = self.data
        fit, q = d.partition(excluded)
        folder = CACHE/('excluded_'+'_'.join(map(str, excluded)))
        cert = json.loads((folder/'fit.json').read_text())
        for k, v in d.certificate(fit, q, excluded).items():
            if cert[k] != v:
                raise ValueError(f'cache certificate differs: {folder} {k}')
        protocol = cert['baseline_protocol']
        if not protocol['refit_all_supervised_weights'] or protocol['seed'] != 42 or protocol['homology_gate_scale'] != 0:
            raise ValueError('unexpected baseline training protocol')
        graph = json.loads((folder/'graph/training.complete.json').read_text())
        cp = Path(graph.get('checkpoint_path', folder/'graph/runs/delta_baseline_subset_frozen_seed42/last.pt'))
        if graph['fit_keys'] != d.keys[fit].tolist() or sha256_file(cp) != graph['checkpoint_sha256']:
            raise ValueError('baseline checkpoint provenance mismatch')
        self.sources[str(cp)] = graph['checkpoint_sha256']
        retrieval = json.loads((folder/'all.retrieval.json').read_text())
        if retrieval['fit_keys'] != d.keys[fit].tolist() or retrieval['fit_labels'] != d.labels[fit].tolist():
            raise ValueError('retrieval reference provenance mismatch')
        if retrieval['query_keys'] != d.keys[np.r_[fit, q]].tolist():
            raise ValueError('retrieval query provenance mismatch')
        audits = json.loads((folder/'inner_audit.json').read_text())
        expected = [{'excluded_folds': sorted(set(excluded)|{int(j)}),
                     'used_query_keys': d.keys[fit[self.fold[fit]==j]].tolist()}
                    for j in sorted(set(self.fold[fit]))]
        if audits != expected:
            raise ValueError('nested baseline cross-fit provenance mismatch')
        with np.load(folder/'predictions.npz', allow_pickle=False) as z:
            if not np.array_equal(z['keys'], d.keys[q]):
                raise ValueError('cached query keys differ')
            values = {k: z[k] for k in z.files if k != 'keys'}
        if not all(np.isfinite(a).all() for a in values.values()):
            raise ValueError('nonfinite baseline cache')
        seq = joblib.load(folder/'dual_sequence.joblib').predict(d.embeddings[q])
        dh = joblib.load(folder/'dual_residual.joblib').predict(np.column_stack([values['retrieval'], seq, self.chem[q]]))
        dual = anchor(values['retrieval'], seq)+dh
        rs = joblib.load(folder/'robust_sequence.joblib').predict(d.embeddings[q, 2560:])
        r = values['retrieval']; available = r[:, 7:9]; den = available.sum(1)
        ra = np.where(den>0, (r[:, :2]*available).sum(1)/np.maximum(den, 1), d.labels[fit].mean())
        rr = ra+joblib.load(folder/'robust_residual.joblib').predict(np.column_stack([r, self.chem[q]]))
        robust = .5*values['graph']+.25*rs+.25*rr
        differences = [float(np.max(abs(dual-values['dual']))), float(np.max(abs(seq-values['ridge']))),
                       float(np.max(abs(robust-values['robust']))),
                       float(np.max(abs(.5*robust+.5*dual-values['prediction'])))]
        if max(differences) > 1e-7:
            raise ValueError(f'cache model replay differs: {excluded} {differences}')
        for name in ['fit.json', 'all.retrieval.json', 'all.retrieval.pt', 'inner_audit.json',
                     'graph/training.complete.json', 'predictions.npz', 'dual_sequence.joblib',
                     'dual_residual.joblib', 'robust_sequence.joblib', 'robust_residual.joblib']:
            self.record(folder/name)
        self.parity.append(dict(excluded=list(excluded), max_difference=max(differences)))
        self.cache[excluded] = q, values
        return q, values

    def upstream(self, excluded, ridge_lambda):
        q, cached = self.cached(excluded)
        if ridge_lambda == 0:
            return q, cached
        tag = tuple(sorted(excluded))
        if tag not in self.weighted:
            fit, _ = self.data.partition(tag)
            w, meta = tail_weights(self.data.labels[fit], ridge_lambda)
            model = Ridge(alpha=.2, solver='cholesky').fit(self.data.embeddings[fit], self.data.labels[fit], sample_weight=w)
            sequence = model.predict(self.data.embeddings[q])
            folder = self.out/'weighted_ridge'/('_'.join(map(str, tag)))
            folder.mkdir(parents=True)
            joblib.dump(model, folder/'sequence.joblib')
            atomic_npz(folder/'predictions.npz', keys=self.data.keys[q], prediction=sequence)
            write_predictions(folder/'weights.csv', self.data.keys[fit], dict(label=self.data.labels[fit], weight=w))
            atomic_json(folder/'fit.json', dict(certificate=self.data.certificate(fit, q, tag), weights=meta))
            self.weighted[tag] = {**cached, 'ridge': sequence}
        return q, self.weighted[tag]

    def fit_residual(self, recipe, x, labels, a, r, keys, folder, *, historical_full_normalization=False):
        folder.mkdir(parents=True, exist_ok=True)
        if recipe['residual_lambda'] is None:
            w = frequency_weights(labels, .25)
            # The historical packaged model used evaluate_nested_homology.weights,
            # which normalizes again after frequency_weights clips. The later
            # FullBaseline fold cache used frequency_weights directly. Reproduce
            # each reference's actual training semantics, recording the difference.
            if historical_full_normalization:
                w = np.clip(w/w.mean(), .25, 4.)
            wm = dict(kind='historical_frequency_power_0.25', mean=float(w.mean()),
                      historical_full_normalization=historical_full_normalization)
        else:
            w, wm = tail_weights(labels, recipe['residual_lambda'])
        target = residual_target(labels, a, r, target=recipe['target'])
        model = HistGradientBoostingRegressor(**HGB).fit(x, target, sample_weight=w)
        joblib.dump(model, folder/'residual.joblib')
        write_predictions(folder/'training_weights_targets.csv', keys,
                          dict(label=labels, weight=w, anchor_cf=a, robust_cf=r, target=target))
        atomic_json(folder/'fit.json', dict(recipe=recipe, weights=wm, count=len(labels),
                                            feature_shape=list(x.shape), hgb=HGB))
        return model

    def development(self):
        for size in (1, 2):
            for excluded in itertools.combinations(range(5), size):
                self.cached(excluded)
        atomic_json(self.out/'cache_audit.json', dict(replay=self.parity, files=self.sources))
        self.emit('upstream_audit_complete', caches=len(self.cache), max_difference=max(r['max_difference'] for r in self.parity))
        self.baseline = np.full(self.n, np.nan)
        self.low = np.zeros(self.n, bool)
        self.oof = {r['name']: np.full(self.n, np.nan) for r in RECIPES}
        self.fold_metrics = {r['name']: [] for r in RECIPES}
        parity = []
        for outer in range(5):
            q, test = self.cached([outer])
            self.baseline[q], self.low[q] = test['prediction'], test['low_homology']
            for ridge_lambda in [0., .05]:
                fit, inner = crossfit_rows(self.fold, outer, lambda e: self.upstream(e, ridge_lambda))
                _, held = self.upstream([outer], ridge_lambda)
                x = np.column_stack([inner['retrieval'], inner['ridge'], self.chem[fit]])
                xq = np.column_stack([held['retrieval'], held['ridge'], self.chem[q]])
                a, aq = anchor(inner['retrieval'], inner['ridge']), anchor(held['retrieval'], held['ridge'])
                for recipe in [r for r in RECIPES if r['ridge_lambda']==ridge_lambda]:
                    folder = self.out/'folds'/recipe['name']/f'outer{outer}'
                    model = self.fit_residual(recipe, x, self.y[fit], a, inner['robust'], self.data.keys[fit], folder)
                    h = model.predict(xq)
                    p = .5*held['robust']+.5*(aq+h)
                    self.oof[recipe['name']][q] = p
                    write_predictions(folder/'predictions.csv', self.data.keys[q],
                        dict(label=self.y[q], prediction=p, reference=test['prediction'],
                             robust=held['robust'], ridge=held['ridge'], anchor=aq, residual=h))
                    atomic_json(folder/'isolation.json', dict(outer=outer, excluded_training_fold=outer,
                        fit_keys=self.data.keys[fit].tolist(), query_keys=self.data.keys[q].tolist(),
                        inner_exclusions=[sorted([outer,j]) for j in range(5) if j!=outer]))
                    self.fold_metrics[recipe['name']].append(metrics(self.y[q], p, self.low[q], self.data.groups[q]))
                    if recipe['name']=='R0':
                        difference=float(np.max(abs(p-test['prediction'])))
                        if difference>1e-7:
                            raise ValueError(f'R0 refit parity failed: {difference}')
                        parity.append(difference)
            self.emit('outer_complete', outer=outer, fitted_recipes=len(RECIPES))
        reference = metrics(self.y, self.baseline, self.low, self.data.groups[:self.n])
        results = []
        for recipe in RECIPES:
            name=recipe['name']; p=self.oof[name]
            m=metrics(self.y,p,self.low,self.data.groups[:self.n]); c=comparison(reference,m)
            scores=[comparison(self.fold_metrics['R0'][f],self.fold_metrics[name][f])['tail_relative_mse'] for f in range(5)]
            c['improved_tail_folds']=sum(s<1 for s in scores)
            c['fold_tail_relative_mse']=scores
            c['accepted_development']=c['metrics_passed'] and c['improved_tail_folds']>=3
            results.append(dict(recipe=recipe,metrics=m,comparison=c,fold_metrics=self.fold_metrics[name]))
            write_predictions(self.out/'oof'/f'{name}.csv',self.data.keys[:self.n],
                dict(label=self.y,fold=self.fold,group=self.data.groups[:self.n],low_homology=self.low,
                     reference=self.baseline,prediction=p))
        ranked=sorted([r for r in results if r['recipe']['name'] not in ['R0',PRIMARY]],
                      key=lambda r:(not r['comparison']['guardrails_passed'],r['comparison']['tail_relative_mse']))
        secondary=ranked[0]['recipe']['name']
        freeze_json(self.out/'full_fit_plan.json',dict(primary=PRIMARY,secondary_validation_only=secondary,
                    seed=42,test_recipe_fixed_before_development=True,
                    reason='Primary predeclared. Secondary selected only from training OOF; no secondary test.'))
        atomic_json(self.out/'development.json',dict(reference=reference,results=results,
                    baseline_refit_max_difference=max(parity)))
        atomic_json(self.out/'oof_primary_bootstrap.json',dict(unit='homology_group',draws=3000,
                    results=bootstrap(self.y,self.baseline,self.oof[PRIMARY],self.low,self.data.groups[:self.n])))
        self.dev_results=results
        self.emit('development_complete',primary_metrics=next(r['metrics'] for r in results if r['recipe']['name']==PRIMARY),secondary=secondary)
        return secondary

    def full_fit(self, names):
        d=self.data
        original=DualFusion(OLD_BUNDLE)
        for p in OLD_BUNDLE.rglob('*'):
            if p.is_file(): self.record(p)
        fitted={}
        sequence_models={}
        for name in names:
            recipe=next(r for r in RECIPES if r['name']==name)
            lam=recipe['ridge_lambda']
            if lam not in sequence_models:
                w,wm=tail_weights(self.y,lam)
                seq=Ridge(alpha=.2,solver='cholesky').fit(d.embeddings[d.train],self.y,sample_weight=w)
                sequence_models[lam]=seq,wm,w
            seq,wm,w=sequence_models[lam]
            inner={k:np.full((self.n,15) if k=='retrieval' else self.n,np.nan) for k in ['retrieval','ridge','robust']}
            for f in range(5):
                q,values=self.upstream([f],lam)
                for k in inner:inner[k][q]=values[k]
            a=anchor(inner['retrieval'],inner['ridge'])
            x=np.column_stack([inner['retrieval'],inner['ridge'],self.chem[:self.n]])
            folder=self.out/'full'/name
            model=self.fit_residual(recipe,x,self.y,a,inner['robust'],d.keys[:self.n],folder,
                                    historical_full_normalization=name=='R0')
            bundle=folder/'bundle';bundle.mkdir()
            shutil.copytree(OLD_BUNDLE/'robust_v1',bundle/'robust_v1')
            joblib.dump(seq,bundle/'sequence.joblib');joblib.dump(model,bundle/'residual.joblib')
            write_predictions(folder/'ridge_weights.csv',d.keys[:self.n],dict(label=self.y,weight=w))
            config=dict(architecture='DualFusion with fixed robust and tail-weighted cross-fit residual',
                        dual_weight=.5,recipe=recipe,seed=42,training_count=self.n,
                        ridge_alpha=.2,ridge_weight_metadata=wm,feature_schema=d.provenance['feature_schema'],
                        train_manifest_sha256=d.provenance['files'][str(ROOT/'artifacts/phgeofuse/manifest.csv')],
                        primary= name==PRIMARY,test_used_for_selection=False,
                        residual_training='OOF meta inputs; complete targets include OOF robust outputs',
                        historical_baseline_full_weight_renormalization=name=='R0',
                        frozen_components='ESM1v/ESM2 encoders and historical robust_v1 including neural baseline',
                        file_hashes={str(p.relative_to(bundle)):sha256_file(p) for p in bundle.rglob('*') if p.is_file()})
            freeze_json(bundle/'model.json',config)
            fitted[name]=DualFusion(bundle)
            if name=='R0':
                before=original.sequence.predict(d.embeddings[d.validation])
                after=seq.predict(d.embeddings[d.validation])
                diff=float(np.max(abs(before-after)))
                if diff>1e-7:raise ValueError(f'full Ridge reproduction failed {diff}')
            self.emit('full_training_complete',recipe=name,training_count=self.n)
        self.models=fitted
        freeze_json(self.out/'frozen_models.json',dict(primary=PRIMARY,
            models={name:sha256_file(self.out/'full'/name/'bundle/model.json') for name in names},
            validation_scored=False,test_scored=False,primary_unchanged_by_development=True))

    def prediction_inputs(self, data, indices, split):
        features=[]
        for name in ['esm1v','esm2']:
            source=SOURCE/f'{name}_masked'/('features_test.npz' if split=='test' else 'features.npz')
            with np.load(source,allow_pickle=False) as z:
                mapping={str(k):i for i,k in enumerate(z['keys'])}
                if len(mapping)!=len(z['keys']):raise ValueError('duplicate feature keys')
                ix=[mapping[k] for k in data.keys[indices]]
                features.extend([z['mean'][ix],z['std'][ix]])
            self.record(source)
        store=RetrievalStore.load(ROOT/'artifacts/phgeofuse/retrieval.pt')
        if store.payload['training_keys']!=self.data.keys[self.data.train].tolist():
            raise ValueError('full-data retrieval references differ')
        if store.payload['training_sequences']!=[self.data.records[i].sequence for i in self.data.train]:
            raise ValueError('full retrieval reference sequences differ')
        if not np.allclose(store.payload['training_labels'].numpy(),self.y,rtol=0,atol=1e-6):
            raise ValueError('full retrieval reference labels differ')
        r=np.array([store.features(k).numpy() for k in data.keys[indices]],dtype=float)
        self.record(ROOT/'artifacts/phgeofuse/retrieval.pt')
        source=(ROOT/'experiments/phgeofuse_phopt_full_20260913/seed42/test_predictions.csv'
                if split=='test' else SOURCE/'baseline_validation.csv')
        raw=read_predictions(source,data.keys[indices]);self.record(source)
        low=~((r[:,4]>=.2)&(r[:,9]>=.8)&(r[:,10]>=.8))
        return raw,features,r,low

    def evaluate(self,split,names):
        if split=='test':
            frozen=json.loads((self.out/'frozen_models.json').read_text())
            for name in names:
                if sha256_file(self.out/'full'/name/'bundle/model.json')!=frozen['models'][name]:
                    raise ValueError('model changed after freeze')
            self.emit('frozen_test_evaluation_started',primary=PRIMARY)
            d=DevelopmentData.load(ROOT/'configs/delta_ref_phopt.yaml',test=True)
            indices=np.arange(len(d.keys))
            self.sources.update(d.provenance['files'])
        else:
            d=self.data;indices=d.validation
        official=read_fasta(ROOT/f'data/phopt_{"testing" if split=="test" else "validation"}.fasta',split)
        signature=lambda rs:sorted((r.protein_id,r.sequence,r.ph_opt) for r in rs)
        if signature(official)!=signature([d.records[i] for i in indices]):
            raise ValueError(f'{split} manifest differs from official PHOPT')
        raw,features,r,low=self.prediction_inputs(d,indices,split)
        sequences=[d.records[i].sequence for i in indices]
        y=d.labels[indices]
        old=DualFusion(OLD_BUNDLE).predict(raw,*features,r,sequences)
        saved_source=(SOURCE/'dual_test/seed42.csv' if split=='test' else OLD_BUNDLE/'seed42_validation.csv')
        saved=read_predictions(saved_source,d.keys[indices]);self.record(saved_source)
        olddiff=float(np.max(abs(old['prediction']-saved)))
        if olddiff>1e-7:raise ValueError('historical reference predictions differ')
        reference=metrics(y,old['prediction'],low)
        results={}
        for name in names:
            model=DualFusion(self.out/'full'/name/'bundle')
            pred=model.predict(raw,*features,r,sequences)
            seq=model.sequence.predict(d.embeddings[indices])
            a=anchor(r,seq)
            h=model.residual.predict(np.column_stack([r,seq,d.x[indices,-25:]]))
            manual=.5*old['robust_v1_prediction']+.5*(a+h)
            diff=float(np.max(abs(manual-pred['prediction'])))
            if diff>1e-7:raise ValueError('production inference replay mismatch')
            if name=='R0' and np.max(abs(pred['prediction']-old['prediction']))>1e-7:
                raise ValueError('full baseline residual reproduction failed')
            m=metrics(y,pred['prediction'],low)
            results[name]=dict(metrics=m,comparison=comparison(reference,m),prediction_replay_difference=diff)
            write_predictions(self.out/split/f'{name}.csv',d.keys[indices],
                dict(label=y,low_homology=low,reference=old['prediction'],prediction=pred['prediction'],
                     raw_baseline=raw,robust=old['robust_v1_prediction'],ridge=seq,anchor=a,residual=h))
            if split=='test' and name==PRIMARY:
                boot=bootstrap(y,old['prediction'],pred['prediction'],low,np.arange(len(y)))
                atomic_json(self.out/'test_primary_bootstrap.json',dict(unit='sample',draws=3000,
                    caveat='Test homology groups unavailable here; sample bootstrap can understate family dependence. Historical follow-up, not new independent confirmation.',results=boot))
        result=dict(split=split,count=len(y),reference=reference,results=results,
                    historical_reference_max_difference=olddiff,seed=42)
        atomic_json(self.out/f'{split}.json',result)
        self.emit(split+'_evaluation_complete',count=len(y),primary=results[PRIMARY])
        return result

    def run(self):
        secondary=self.development()
        self.full_fit(['R0',PRIMARY,secondary])
        validation=self.evaluate('validation',['R0',PRIMARY,secondary])
        test=self.evaluate('test',['R0',PRIMARY])
        development=next(r for r in self.dev_results if r['recipe']['name']==PRIMARY)['comparison']
        decision=dict(primary=PRIMARY,seed=42,training_count=self.n,validation_count=760,test_count=1971,
            development_passed=development['accepted_development'],
            validation_passed=validation['results'][PRIMARY]['comparison']['metrics_passed'],
            test_descriptive_passed=test['results'][PRIMARY]['comparison']['metrics_passed'],
            production_replaced=False,elapsed_seconds=time.monotonic()-self.start,
            scope='One full-training-data refit of the predeclared primary dual expert; frozen robust/encoders.',
            test_not_used_to_select_or_revise_recipe=True)
        decision['candidate_passed_all']=all(decision[k] for k in ['development_passed','validation_passed','test_descriptive_passed'])
        atomic_json(self.out/'decision.json',decision)
        atomic_json(self.out/'source_hashes.json',self.sources)
        self.report(validation,test,decision)
        atomic_json(self.out/'artifact_hashes.json',{str(p.relative_to(self.out)):sha256_file(p)
            for p in sorted(self.out.rglob('*')) if p.is_file() and p.name not in ['artifact_hashes.json','status.json']})
        self.emit('complete',decision=decision)

    def report(self,validation,test,decision):
        b=test['reference'];c=test['results'][PRIMARY]['metrics']
        lines=['# Dual 完整融合重加权实验：PHOPT，seed42','',
            '已完成 8 个固定配方的五折开发比较、主方案 R4b 的 7124 条全训练集拟合，以及 760 条验证 / 1971 条测试评估。',
            '主方案在结果产生前锁定；原编码器与 robust 分支冻结，dual Ridge 与残差重拟合。不是全网络从头重训，也不是五种子实验。','',
            '## 测试集：原模型与预先锁定主方案','',
            '| 区间 | n | 原 RMSE | 新 RMSE | ΔRMSE | 原 MAE | 新 MAE | 原偏差 | 新偏差 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for g,label in [('all','整体'),('extreme_acid','极酸 pH≤4'),('core','中心 4<pH<10'),('extreme_alkaline','极碱 pH≥10'),('low_homology','低同源')]:
            x,z=b[g],c[g]
            lines.append(f'| {label} | {x["count"]} | {x["rmse"]:.6f} | {z["rmse"]:.6f} | {z["rmse"]-x["rmse"]:+.6f} | {x["mae"]:.6f} | {z["mae"]:.6f} | {x["bias"]:.6f} | {z["bias"]:.6f} |')
        lines+=['',f'整体 Pearson：{b["all"]["pearson"]:.6f} → {c["all"]["pearson"]:.6f}；Spearman：{b["all"]["spearman"]:.6f} → {c["all"]["spearman"]:.6f}；R²：{b["all"]["r2"]:.6f} → {c["all"]["r2"]:.6f}。',
            f'中心误判极端比例：{b["false_extreme_rate"]:.6f} → {c["false_extreme_rate"]:.6f}。','',
            '## 训练内部五折开发对照','',
            '| 配方 | 整体 RMSE | 极酸 RMSE | 极碱 RMSE | 中心 RMSE | 双端相对 MSE | 接受 |','|---|---:|---:|---:|---:|---:|---|']
        for r in self.dev_results:
            m=r['metrics'];co=r['comparison']
            lines.append(f'| {r["recipe"]["name"]} | {m["all"]["rmse"]:.6f} | {m["extreme_acid"]["rmse"]:.6f} | {m["extreme_alkaline"]["rmse"]:.6f} | {m["core"]["rmse"]:.6f} | {co["tail_relative_mse"]:.6f} | {co["accepted_development"]} |')
        lines+=['','R0：历史频率加权；R1：均匀；R2a/b：仅双端权重 λ=.05/.10；R3：仅完整融合残差目标；R4a/b：完整目标 + λ=.05/.10；R5：R4b + Ridge λ=.05。','',
            '## 验证集（未用于修订主配方）','',
            '| 配方 | 整体 RMSE | 极酸 RMSE | 极碱 RMSE | 中心 RMSE | 接受 |','|---|---:|---:|---:|---:|---|']
        for name,row in validation['results'].items():
            m=row['metrics']
            lines.append(f'| {name} | {m["all"]["rmse"]:.6f} | {m["extreme_acid"]["rmse"]:.6f} | {m["extreme_alkaline"]["rmse"]:.6f} | {m["core"]["rmse"]:.6f} | {row["comparison"]["metrics_passed"]} |')
        lines+=['','## 接受条件与结论','',
            '两端 RMSE 各下降至少 5%，MAE 不上升、绝对偏差下降；整体/中心/低同源 RMSE 增量≤0.01，整体 MAE 增量≤0.01，中心误判极端比例增量≤0.005；开发至少 3/5 折的双端相对 MSE 改善。',
            f'开发通过：{decision["development_passed"]}；验证通过：{decision["validation_passed"]}；测试描述性通过：{decision["test_descriptive_passed"]}。',
            f'全部门槛通过：{decision["candidate_passed_all"]}。生产模型未替换。',
            '测试未用于选参；按用户要求，即使开发门槛失败仍完成预先锁定主方案的测试以观察变化。该历史测试集此前已被访问，结果是后续比较，不是新的独立确认。','',
            '## 复算与来源','',
            '- protocol.json / full_fit_plan.json / frozen_models.json：结果产生前的方案、全量拟合计划、测试前模型冻结。',
            '- folds/、weighted_ridge/、full/：分折模型、逐样本训练权重/目标、可加载完整预测 bundle。',
            '- oof/、validation/、test/：逐样本预测，所有汇总均可复算。',
            '- cache_audit.json / source_hashes.json：上游排除范围、标签/样本/特征/模型哈希，以及保存模型的预测重放。',
            '- 历史基线复现细节：分折 FullBaseline 使用裁剪后频率权重；原完整模型包的训练另做一次均值归一化。本轮 R0 在各自场景严格复现原逻辑，新混合损失权重始终均值为 1。',
            '- oof_primary_bootstrap.json：同源组配对 bootstrap；test_primary_bootstrap.json：样本配对 bootstrap（没有测试同源组，不能当作家族独立区间）。',
            f'- 计算部分用时：{decision["elapsed_seconds"]:.2f} 秒；不包括实现与事后核验。']
        (self.out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/dual_tail_weighting_20260919')
    args=parser.parse_args()
    torch.set_num_threads(4)
    Experiment(args.output).run()
