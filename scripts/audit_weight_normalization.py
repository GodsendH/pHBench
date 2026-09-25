"""Quantify stochastic batch-normalization effects, without fitting a model.

The reported masses are expected coefficients of per-sample squared errors,
not observed losses or gradient magnitudes. Only PHOPT training rows are used.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from localph.phenv_data import sha_file, write_json


def coefficient_mass(counts, weights, batch_size, seed=42, draws=200000):
    counts = np.asarray(counts, dtype=int)
    weights = np.asarray(weights, dtype=float)
    n = int(counts.sum())
    if batch_size > n or min(counts) < 1:
        raise ValueError('batch exceeds population or an empty bin is present')
    rng = np.random.default_rng(seed)
    fractions = counts / n
    if batch_size == 1:
        return fractions, 'exact singleton cancellation'
    if batch_size == n:
        return counts * weights / (counts @ weights), 'exact full training batch'
    coefficients = []
    for i, w in enumerate(weights):
        others = counts.copy()
        others[i] -= 1
        if batch_size == 2:
            expected = float((others / (n - 1)) @ (w / (w + weights)))
        else:
            samples = rng.multivariate_hypergeometric(others, batch_size - 1, size=draws)
            expected = float(np.mean(w / (w + samples @ weights)))
        coefficients.append(batch_size * fractions[i] * expected)
    return np.asarray(coefficients), ('exact pair expectation' if batch_size == 2
        else f'Monte Carlo without replacement, {draws} draws per bin')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=ROOT/'artifacts/phgeofuse/manifest.csv')
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    with a.manifest.open(newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['split']=='train' and r['status']=='ready']
    y = np.asarray([float(r['ph_opt']) for r in rows])
    published = np.asarray([float(r['sample_weight']) for r in rows])
    bins = np.digitize(y, [5., 9.])
    counts = np.bincount(bins, minlength=3)
    official = np.asarray([published[bins==i][0] for i in range(3)])
    if any(not np.all(published[bins==i]==official[i]) for i in range(3)):
        raise ValueError('published weights are not constant within these bins')
    folded = len(y)/(3*counts)
    recipes = {'published_unclipped':official, 'published_distribution_clipped':np.clip(official,.5,3.),
               'training_inverse_frequency':folded, 'training_half_mixed':.5+.5*folded}
    result = {'training_rows':len(y),'bin_counts':counts.tolist(),'bins':['<5','[5,9)','>=9'],
        'source_sha256':sha_file(a.manifest),'code_sha256':sha_file(Path(__file__)),
        'seed':42,'models_fitted':0,'validation_test_labels_used':False,
        'interpretation':'Expected squared-error coefficients, not realized loss or gradient shares.',
        'recipes':{}}
    lines=['# 小批次权重归一化审计','',
        '只使用 PHOPT 训练记录。表中为各桶期望平方误差系数质量，不是实际误差或梯度贡献。', '',
        '| 权重 | batch=1 酸/中/碱 | batch=2 酸/中/碱 | batch=32 酸/中/碱 | 固定训练均值归一化 酸/中/碱 |',
        '|---|---|---|---|---|']
    for name, weights in recipes.items():
        fixed = counts*weights/(counts@weights)
        batches = {}
        for batch in (1,2,32):
            mass, method = coefficient_mass(counts, weights, batch)
            batches[str(batch)]={'coefficient_mass':mass.tolist(),'method':method,'sum':float(mass.sum())}
        result['recipes'][name]={'weights':weights.tolist(),'fixed_training_mean_mass':fixed.tolist(),
            'batch_weight_sum_normalization':batches}
        fmt=lambda values:'/'.join(f'{v:.2%}' for v in values)
        lines.append('| '+name+' | '+' | '.join(fmt(batches[str(b)]['coefficient_mass']) for b in (1,2,32))+' | '+fmt(fixed)+' |')
    lines += ['', '固定训练均值指 mean(w*error²)/mean_train(w)，分母只来自当前训练子集及其裁剪后的权重。',
        'batch=1、2结果精确；batch=32为随机打乱无放回小批次的蒙特卡洛估计。梯度累积不能撤销各微批次内部已发生的权重抵消。',
        '本轮pHenv预训练已使用固定均值；PHOPT端到端pilot使用batch=32的批内权重和归一化。冻结头使用全训练批次，两种归一化相同。',
        '尚未通过重训证明新选项带来性能改善；结果不用于更改正在运行的配方或挑选其外层模型。']
    write_json(a.output/'verification.json', result)
    (a.output/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result), flush=True)


if __name__=='__main__':
    main()
