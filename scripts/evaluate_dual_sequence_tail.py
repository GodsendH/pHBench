"""Fixed nonlinear sequence-tail experiment with a gated, frozen test phase."""
from __future__ import annotations

import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
import argparse
import csv
import itertools
import json
from pathlib import Path
import shutil
import sys
import time

import joblib
import numpy as np
import torch
from sklearn.metrics.pairwise import rbf_kernel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
from evaluate_dual_tail_weighting import Experiment, bootstrap
from evaluate_dual_tail_priority import expanded_metrics, KERNEL_ROOT, KERNEL_SOURCE
from train_ephod_svr_control import load_features
from phgeofuse.cache import atomic_json, sha256_file
from phgeofuse.delta_ref.data import DevelopmentData, freeze_json, read_predictions, write_predictions
from phgeofuse.sequence_tail import (KernelReadout, SequenceTailFusion, complete_expert,
                                    fit_expert, fit_transform, transform)

PREVIOUS = ROOT/'experiments/dual_tail_priority_20260919'
PREVIOUS_BUNDLE = PREVIOUS/'full/K1_s0.75/bundle'
MIXES = [.25, .5, .75]
BASES = ['original', 'previous']
RECIPES = []
for feature, target, c, masses in itertools.product(['token', 'dual'], ['direct', 'residual'],
                                                   [1., 4.], [(.025, .05), (.05, .10)]):
    name = f'S{len(RECIPES):02d}'
    RECIPES.append(dict(name=name, features=feature, target=target, method='svr',
                        regularization=c, acid_mass=masses[0], alkaline_mass=masses[1]))
for feature, alpha in itertools.product(['token', 'dual'], [.1, 1.]):
    name = f'R{len(RECIPES)-16:02d}'
    RECIPES.append(dict(name=name, features=feature, target='residual', method='krr',
                        regularization=alpha, acid_mass=.05, alkaline_mass=.10))


def selection(candidate, original, svr):
    ratios = [candidate[g][m]/svr[g][m]
              for g in ['extreme_acid', 'extreme_alkaline'] for m in ['rmse', 'mae']]
    return dict(proxy_overall_budget=(candidate['all']['rmse'] <= 1.05*original['all']['rmse']
                                     and candidate['all']['mae'] <= 1.035*original['all']['mae']),
                worst_tail_ratio=max(ratios), mean_tail_ratio=float(np.mean(ratios)))


def score_key(row):
    s = row['selection']
    return (not s['proxy_overall_budget'], s['worst_tail_ratio'], s['mean_tail_ratio'],
            row['metrics']['all']['rmse'])


class SequenceExperiment(Experiment):
    def __init__(self, out):
        out.mkdir(parents=True, exist_ok=False)
        self.out, self.start = out, time.monotonic()
        self.data = DevelopmentData.load(ROOT/'configs/delta_ref_phopt.yaml')
        d = self.data; self.n = len(d.train)
        if not np.array_equal(d.train, np.arange(self.n)):
            raise ValueError('training rows not contiguous')
        self.y, self.fold, self.chem = d.labels[:self.n], d.folds[:self.n], d.x[:, -25:]
        self.cache, self.parity, self.sources = {}, [], dict(d.provenance['files'])
        sources = [Path(__file__), ROOT/'phgeofuse/sequence_tail.py',
                   ROOT/'docs/dual_sequence_tail_20260919/PLAN_ZH.md',
                   ROOT/'scripts/evaluate_dual_tail_weighting.py',
                   ROOT/'scripts/evaluate_dual_tail_priority.py', ROOT/'phgeofuse/tail_priority.py']
        for p in sources:
            self.record(p)
        self.protocol = dict(seed=42, recipes=RECIPES, mixes=MIXES, bases=BASES,
            data=d.provenance, source_files=dict(self.sources),
            selection='OOF top four distinct recipes; validation ranks by budget, worst tail ratio to matched SVR, mean tail ratio, overall RMSE. Budget: RMSE<=1.05*original and MAE<=1.035*original.',
            test_gate='Both OOF and validation must pass proxy budgets and have lower worst tail ratio than previous K1_s0.75. Only one frozen candidate is tested.',
            final_acceptance=dict(venus_paper_rmse=.809, venus_paper_mae=.578,
                acid_rmse=1.5463121962893993, acid_mae=1.2145561258813882,
                alkaline_rmse=1.8918528990474008, alkaline_mae=1.5532525299350353),
            scope='Exploratory adaptive historical follow-up; fixed train homology folds; OOF model selection is not unbiased nested HPO. Encoders and upstream complete baseline frozen.',
            feature_archive='Token archive contains all splits without consumed test labels; only train/validation rows are used before release. Dual test feature file opens only after release.',
            previous_goal_turn='progress: completed loss/SVR experiment and verification; numerical target remains unmet')
        freeze_json(out/'protocol.json', self.protocol)
        self.emit('protocol_frozen', recipes=len(RECIPES), combinations=len(RECIPES)*len(BASES)*len(MIXES))

    def previous_file(self, relative):
        p = PREVIOUS/relative
        expected = self.previous_hashes[relative]
        if sha256_file(p) != expected:
            raise ValueError(f'previous artifact changed: {relative}')
        self.record(p)
        return p

    def prepare(self):
        self.previous_hashes = json.loads((PREVIOUS/'artifact_hashes.json').read_text())
        self.record(PREVIOUS/'artifact_hashes.json')
        verify = json.loads(self.previous_file('verification.json').read_text())
        if verify['status'] != 'passed':
            raise ValueError('previous experiment is not verified')
        release = json.loads(self.previous_file('release.json').read_text())
        if release['candidate'] != 'K1_s0.75' or release['model_sha256'] != sha256_file(PREVIOUS_BUNDLE/'model.json'):
            raise ValueError('previous release differs')
        self.previous_oof = read_predictions(self.previous_file('oof/K1_s0.75.csv'), self.data.keys[:self.n])
        self.svr_oof = read_predictions(self.previous_file('oof/K1_s0.75.csv'), self.data.keys[:self.n], 'kernel')
        keys, split, labels, features = load_features(KERNEL_ROOT/'esm1v_full_precision', ROOT/'artifacts/phgeofuse/manifest.csv')
        self.token_source, self.token_index = features, {str(k): i for i, k in enumerate(keys)}
        ix = [self.token_index[k] for k in self.data.keys]
        np.testing.assert_array_equal(labels[ix], self.data.labels)
        self.features = dict(token=features[ix], dual=self.data.embeddings)
        for name in ['features.npz', 'complete.json', 'protocol.json']:
            self.record(KERNEL_ROOT/'esm1v_full_precision'/name)
        for size in [1, 2]:
            for excluded in itertools.combinations(range(5), size):
                self.cached(excluded)
        atomic_json(self.out/'cache_audit.json', dict(replays=self.parity, sources=self.sources))
        self.emit('upstream_verified', max_difference=max(r['max_difference'] for r in self.parity))

    def guide(self, outer=None):
        fit = np.arange(self.n) if outer is None else np.flatnonzero(self.fold != outer)
        guide = np.full(self.n, np.nan)
        for inner in sorted(set(self.fold[fit])):
            excluded = [int(inner)] if outer is None else sorted([int(outer), int(inner)])
            q, values = self.cached(excluded)
            use = self.fold[q] == inner
            guide[q[use]] = values['prediction'][use]
        if not np.isfinite(guide[fit]).all():
            raise ValueError('guide coverage differs')
        if outer is not None and not np.isnan(guide[self.fold == outer]).all():
            raise ValueError('outer rows entered guide')
        return fit, guide[fit]

    def fit_one(self, recipe, fit, guide, state, kernel, folder):
        folder.mkdir(parents=True, exist_ok=True)
        native, weights, target, info = fit_expert(kernel, self.y[fit], guide, recipe)
        model = KernelReadout(native, len(fit))
        # Exact compact export is compared before discarding the native estimator.
        diff = float(np.max(abs(model.predict(kernel[:16])-native.predict(kernel[:16]))))
        if diff > 1e-7:
            raise ValueError('compact estimator export differs')
        package = dict(model=model, transform=state, fit_indices=fit)
        joblib.dump(package, folder/'model.joblib')
        write_predictions(folder/'training_objective.csv', self.data.keys[fit],
            dict(label=self.y[fit], guide_cf=guide, weight=weights, target=target))
        atomic_json(folder/'fit.json', dict(recipe=recipe, weight_metadata=info,
            fit_keys=self.data.keys[fit].tolist(), feature_count=len(state['mean']),
            compact_native_max_difference=diff, n_support=len(model.support)))
        return package

    def development(self):
        self.prepare()
        self.baseline, self.low = np.full(self.n, np.nan), np.zeros(self.n, bool)
        self.oof = {r['name']: np.full(self.n, np.nan) for r in RECIPES}
        for outer in range(5):
            fit, guide = self.guide(outer)
            q, held = self.cached([outer])
            self.baseline[q], self.low[q] = held['prediction'], held['low_homology']
            for feature in ['token', 'dual']:
                x = self.features[feature]
                z, state = fit_transform(x[fit]); zq = transform(x[q], state)
                k = rbf_kernel(z, gamma=state['gamma']); kq = rbf_kernel(zq, z, gamma=state['gamma'])
                for recipe in [r for r in RECIPES if r['features'] == feature]:
                    folder = self.out/'folds'/recipe['name']/f'outer{outer}'
                    package = self.fit_one(recipe, fit, guide, state, k, folder)
                    raw = package['model'].predict(kq)
                    p = complete_expert(raw, held['prediction'], recipe['target'])
                    self.oof[recipe['name']][q] = p
                    write_predictions(folder/'predictions.csv', self.data.keys[q],
                        dict(label=self.y[q], prediction=p, original=held['prediction']))
                    atomic_json(folder/'isolation.json', dict(certificate=self.data.certificate(fit, q, [outer]),
                        guide_exclusions=[sorted([outer, j]) for j in range(5) if j != outer]))
                    self.emit('fold_expert_complete', outer=outer, recipe=recipe['name'])
                del k, kq, z, zq
        old = expanded_metrics(self.y, self.baseline, self.low, self.data.groups[:self.n])
        svr = expanded_metrics(self.y, self.svr_oof, self.low, self.data.groups[:self.n])
        previous = expanded_metrics(self.y, self.previous_oof, self.low, self.data.groups[:self.n])
        rows = []
        for recipe, base, mix in itertools.product(RECIPES, BASES, MIXES):
            base_prediction = self.baseline if base == 'original' else self.previous_oof
            p = (1-mix)*base_prediction+mix*self.oof[recipe['name']]
            m = expanded_metrics(self.y, p, self.low, self.data.groups[:self.n])
            name = f'{recipe["name"]}_{base}_s{mix}'
            rows.append(dict(name=name, recipe=recipe, base=base, mix=mix,
                             metrics=m, selection=selection(m, old, svr)))
            write_predictions(self.out/'oof'/f'{name}.csv', self.data.keys[:self.n],
                dict(label=self.y, prediction=p, original=self.baseline, previous=self.previous_oof,
                     low_homology=self.low, fold=self.fold, group=self.data.groups[:self.n]))
        self.ranking = sorted(rows, key=score_key)
        self.shortlist = []; seen = set()
        for row in self.ranking:
            if row['recipe']['name'] not in seen:
                self.shortlist.append(row); seen.add(row['recipe']['name'])
            if len(self.shortlist) == 4:
                break
        self.development_result = dict(original=old, previous=previous, local_svr=svr,
            previous_selection=selection(previous, old, svr), ranking=self.ranking)
        atomic_json(self.out/'development.json', self.development_result)
        freeze_json(self.out/'validation_plan.json', dict(candidates=self.shortlist, test_used=False))
        self.emit('development_complete', shortlist=[dict(name=r['name'], selection=r['selection'],
                  overall=r['metrics']['all']) for r in self.shortlist])

    def full_training(self):
        fit, guide = self.guide()
        for row in self.shortlist:
            recipe, name = row['recipe'], row['name']
            z, state = fit_transform(self.features[recipe['features']][fit])
            kernel = rbf_kernel(z, gamma=state['gamma'])
            folder = self.out/'full'/name
            package = self.fit_one(recipe, fit, guide, state, kernel, folder)
            del kernel
            bundle = folder/'bundle'; bundle.mkdir()
            shutil.copytree(PREVIOUS_BUNDLE, bundle/'previous')
            package['train_features'] = z
            joblib.dump(package, bundle/'sequence_expert.joblib')
            config = dict(recipe=recipe, mix=row['mix'], base=row['base'], seed=42,
                training_count=self.n, prediction_uses_labels=False,
                file_hashes={str(p.relative_to(bundle)): sha256_file(p)
                             for p in bundle.rglob('*') if p.is_file()})
            freeze_json(bundle/'model.json', config)
            self.emit('full_fit_complete', candidate=name, count=self.n)

    def eval_split(self, split, names):
        is_test = split == 'test'
        d = DevelopmentData.load(ROOT/'configs/delta_ref_phopt.yaml', test=True) if is_test else self.data
        ix = np.arange(len(d.keys)) if is_test else d.validation
        self.sources.update(d.provenance['files'])
        raw, features, retrieval, low = self.prediction_inputs(d, ix, split)
        token = self.token_source[[self.token_index[k] for k in d.keys[ix]]]
        sequences = [d.records[i].sequence for i in ix]
        rows = []
        for name in names:
            model = SequenceTailFusion(self.out/'full'/name/'bundle')
            predictions = model.predict(raw, *features, retrieval, sequences, kernel_features=token)
            write_predictions(self.out/split/f'{name}.csv', d.keys[ix],
                dict(label=d.labels[ix], low_homology=low, raw_baseline=raw, **predictions))
            m = expanded_metrics(d.labels[ix], predictions['prediction'], low)
            rows.append(dict(name=name, metrics=m))
        old = expanded_metrics(d.labels[ix], predictions['original'], low)
        previous = expanded_metrics(d.labels[ix], predictions['previous'], low)
        if not is_test:
            old_previous = read_predictions(self.previous_file('validation/K1_s0.75.csv'), d.keys[ix])
            if np.max(abs(old_previous-predictions['previous'])) > 1e-7:
                raise ValueError('previous validation replay differs')
            svr_pred = read_predictions(self.previous_file('validation/K1_s0.75.csv'), d.keys[ix], 'kernel_reference')
            svr = expanded_metrics(d.labels[ix], svr_pred, low)
            for row in rows:
                row['selection'] = selection(row['metrics'], old, svr)
        else:
            svr = None
        result = dict(split=split, count=len(ix), original=old, previous=previous, local_svr=svr, candidates=rows)
        atomic_json(self.out/f'{split}.json', result)
        return result

    def release_and_test(self):
        validation = self.eval_split('validation', [r['name'] for r in self.shortlist])
        ranked = sorted(validation['candidates'], key=score_key)
        candidate = ranked[0]; name = candidate['name']
        development = next(r for r in self.shortlist if r['name'] == name)
        previous_val = selection(validation['previous'], validation['original'], validation['local_svr'])
        gate = dict(oof_budget=development['selection']['proxy_overall_budget'],
                    validation_budget=candidate['selection']['proxy_overall_budget'],
                    oof_tail_improvement=development['selection']['worst_tail_ratio'] < self.development_result['previous_selection']['worst_tail_ratio'],
                    validation_tail_improvement=candidate['selection']['worst_tail_ratio'] < previous_val['worst_tail_ratio'])
        passed = all(gate.values())
        freeze_json(self.out/'release.json', dict(candidate=name,
            model_sha256=sha256_file(self.out/'full'/name/'bundle/model.json'),
            protocol_sha256=sha256_file(self.out/'protocol.json'), validation_ranking=ranked,
            test_gate=gate, permitted_test=passed, test_scored=False))
        self.emit('selection_frozen', candidate=name, test_gate=gate)
        if not passed:
            decision = dict(selected=name, test_candidate_count=0, test_used_for_selection=False,
                production_replaced=False, user_target_reached=False, gate=gate,
                reason='Predeclared development/validation gate failed; no new test evaluation.',
                elapsed_seconds=time.monotonic()-self.start)
            atomic_json(self.out/'decision.json', decision)
            self.report(validation, decision)
            self.finish(decision)
            return
        test = self.eval_split('test', [name])
        rows = list(csv.DictReader((self.out/'test'/f'{name}.csv').open()))
        keys = [r['key'] for r in rows]
        y = np.array([float(r['label']) for r in rows]); p = np.array([float(r['prediction']) for r in rows])
        low = np.array([r['low_homology'] == 'True' for r in rows])
        comparisons = {}
        comparator_paths = [('official_ensemble', ROOT/'experiments/delta_ref_phopt_20260916/analysis/ephod_official_strict_tail_20260916/predictions.csv', 'Ensemble'),
                            ('local_svr', KERNEL_SOURCE/'followup_test/predictions.csv', 'prediction')]
        for label, source, column in comparator_paths:
            other = {r['key']: r for r in csv.DictReader(source.open())}
            if set(keys) != set(other):
                raise ValueError('comparison keys differ')
            np.testing.assert_array_equal(y, [float(other[k]['label']) for k in keys])
            ep = np.array([float(other[k][column]) for k in keys])
            comparisons[label] = expanded_metrics(y, ep, low)
            if label == 'official_ensemble':
                comparisons['paired_sample_bootstrap'] = bootstrap(y, ep, p, low, np.arange(len(y)))
            self.record(source)
        m = test['candidates'][0]['metrics']; official = comparisons['official_ensemble']
        checks = dict(venus_rmse=m['all']['rmse'] < .809, venus_mae=m['all']['mae'] < .578)
        strong = {}
        for region in ['extreme_acid', 'extreme_alkaline']:
            for metric in ['rmse', 'mae']:
                checks[region+'_'+metric] = m[region][metric] < official[region][metric]
                strong[region+'_'+metric] = m[region][metric] < comparisons['local_svr'][region][metric]
        atomic_json(self.out/'test_comparators.json', comparisons)
        decision = dict(selected=name, seed=42, user_target_reached=all(checks.values()), checks=checks,
            strong_local_svr_checks=strong, beats_stronger_local_svr=all(strong.values()),
            test_candidate_count=1, test_used_for_selection=False, production_replaced=False,
            train_count=self.n, validation_count=760, test_count=1971,
            elapsed_seconds=time.monotonic()-self.start)
        atomic_json(self.out/'decision.json', decision)
        self.report(validation, decision, test, comparisons)
        self.finish(decision)

    def report(self, validation, decision, test=None, comparisons=None):
        lines = ['# 非线性序列尾部专家实验', '',
                 f'本轮选择：{decision["selected"]}；目标达成：{decision["user_target_reached"]}；新测试候选数：{decision["test_candidate_count"]}。', '',
                 '20个配方×5个同源折，120个融合组合；前四个不同配方在7124条训练样本全量拟合，760条验证选择。表征及上游完整dual固定。',
                 '整体门槛是Venus-DREAM论文均值0.809/0.578，不等于本地受控重训。历史测试反复访问背景下，结果只作探索性研究。', '',
                 '| OOF候选（前12） | RMSE | MAE | 酸RMSE | 酸MAE | 碱RMSE | 碱MAE | 最弱尾部比值 | 预算 |',
                 '|---|---:|---:|---:|---:|---:|---:|---:|---|']
        for row in self.ranking[:12]:
            m, s = row['metrics'], row['selection']
            lines.append(f'| {row["name"]} | {m["all"]["rmse"]:.6f} | {m["all"]["mae"]:.6f} | {m["extreme_acid"]["rmse"]:.6f} | {m["extreme_acid"]["mae"]:.6f} | {m["extreme_alkaline"]["rmse"]:.6f} | {m["extreme_alkaline"]["mae"]:.6f} | {s["worst_tail_ratio"]:.6f} | {s["proxy_overall_budget"]} |')
        lines += ['', '| 验证候选 | RMSE | MAE | 酸RMSE | 碱RMSE | 最弱尾部比值 | 预算 |',
                  '|---|---:|---:|---:|---:|---:|---|']
        for row in sorted(validation['candidates'], key=score_key):
            m, s = row['metrics'], row['selection']
            lines.append(f'| {row["name"]} | {m["all"]["rmse"]:.6f} | {m["all"]["mae"]:.6f} | {m["extreme_acid"]["rmse"]:.6f} | {m["extreme_alkaline"]["rmse"]:.6f} | {s["worst_tail_ratio"]:.6f} | {s["proxy_overall_budget"]} |')
        lines += ['', '开发预算是总体RMSE≤原dual×1.05、MAE≤原dual×1.035。最终接受必须逐项检查实际测试绝对门槛。', '']
        if test is None:
            lines += ['预先规定的测试准入未通过，因此没有新增测试结果，也未达成原目标。',
                      json.dumps(decision['gate'], ensure_ascii=False), '']
        else:
            lines += ['| 测试指标 | 原dual | 上一轮K1 | 本轮 | EpHod集成 | 本地SVR |',
                      '|---|---:|---:|---:|---:|---:|']
            for region in ['all', 'extreme_acid', 'extreme_alkaline']:
                for metric in ['rmse', 'mae']:
                    values = [test['original'], test['previous'], test['candidates'][0]['metrics'],
                              comparisons['official_ensemble'], comparisons['local_svr']]
                    lines.append('| '+region+' '+metric+' | '+' | '.join(f'{m[region][metric]:.6f}' for m in values)+' |')
            lines += ['', '验收：'+json.dumps(decision['checks'], ensure_ascii=False), '']
        lines += ['所有分折标准化和损失权重仅用拟合子集；残差目标来自双排除上游。预测入口不接收标签；不按测试ID修改样本或输出。',
                  '完整预测/权重/目标/证书/模型保留；OOF选择是探索性搜索，不能解读为无偏嵌套调参估计。生产模型未替换。',
                  f'本轮计算用时 {decision["elapsed_seconds"]:.1f} 秒，不含实现和独立核验。', '',
                  '重现：`python scripts/evaluate_dual_sequence_tail.py --output experiments/dual_sequence_tail_reproduction`（目录需不存在）。']
        (self.out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')

    def finish(self, decision):
        atomic_json(self.out/'source_hashes.json', self.sources)
        for relative in ['scripts/evaluate_dual_sequence_tail.py', 'phgeofuse/sequence_tail.py',
                         'tests/test_sequence_tail.py', 'docs/dual_sequence_tail_20260919/PLAN_ZH.md']:
            target = self.out/'reproduce'/relative; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/relative, target)
        atomic_json(self.out/'artifact_hashes.json', {str(p.relative_to(self.out)): sha256_file(p)
            for p in sorted(self.out.rglob('*')) if p.is_file() and p.name not in ['artifact_hashes.json', 'status.json']})
        self.emit('complete', decision=decision)

    def run(self):
        self.development(); self.full_training(); self.release_and_test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'experiments/dual_sequence_tail_20260919')
    args = parser.parse_args(); torch.set_num_threads(4)
    SequenceExperiment(args.output).run()
