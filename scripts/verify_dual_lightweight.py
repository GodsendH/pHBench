"""Independently replay saved lightweight-search validation models and metrics."""
import os
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='4'
import csv
import json
import sys
from pathlib import Path
import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from phgeofuse.cache import atomic_json, sha256_file
from phgeofuse.delta_ref.data import DevelopmentData, read_predictions
from phgeofuse.dual_fusion import DualFusion, retrieval_sequence_anchor
from phgeofuse.retrieval import RetrievalStore

OUT=ROOT/'experiments/dual_lightweight_20260918'
SOURCE=ROOT/'experiments/phgeofuse_redesign_20260914'
rmse=lambda y,p:float(np.sqrt(np.mean((p-y)**2)))
protocol=json.loads((OUT/'protocol.json').read_text())
assert protocol['script_sha256']==sha256_file(ROOT/'scripts/tune_dual_lightweight.py')
for name,digest in json.loads((OUT/'cache_audit.json').read_text())['files'].items():
    assert sha256_file(Path(name))==digest
d=DevelopmentData.load(ROOT/'configs/delta_ref_phopt.yaml')
assert protocol['data']==d.provenance
with (OUT/'best_oof.csv').open() as f:rows=list(csv.DictReader(f))
assert [r['key'] for r in rows]==d.keys[d.train].tolist()
y=np.array([float(r['label']) for r in rows]);p=np.array([float(r['prediction']) for r in rows])
base=np.array([float(r['baseline']) for r in rows])
assert np.array_equal(y,d.labels[d.train]) and np.isfinite(p).all()
ranking=json.loads((OUT/'train_ranking.json').read_text())
best=ranking[0]
assert abs(rmse(y,p)-best['rmse'])<1e-12
for outer in range(5):
    mask=d.folds[d.train]==outer
    path=OUT/'fold_predictions'/f'r{best["recipe_id"]}_a{best["alpha"]}_f{outer}.npz'
    with np.load(path) as z, np.load(ROOT/f'experiments/delta_ref_phopt_20260916/baseline/seed42/excluded_{outer}/predictions.npz') as b:
        assert np.array_equal(z['keys'],d.keys[d.train][mask])
        reconstructed=(1-best['w'])*b['robust']+best['w']*(z['anchor']+best['gamma']*z['residual'])
        np.testing.assert_allclose(reconstructed,p[mask],atol=1e-12,rtol=0)
        assert abs(rmse(y[mask],base[mask])-rmse(y[mask],p[mask])-best['fold_rmse_improvements'][outer])<1e-12

v=d.validation;features=[]
for name in ['esm1v','esm2']:
    with np.load(SOURCE/f'{name}_masked/features.npz') as z:
        index={k:i for i,k in enumerate(z['keys'])};ix=[index[k] for k in d.keys[v]]
        features.extend([z['mean'][ix],z['std'][ix]])
store=RetrievalStore.load(ROOT/'artifacts/phgeofuse/retrieval.pt')
r=np.array([store.features(k).numpy() for k in d.keys[v]],dtype=float)
old=DualFusion(SOURCE/'dual_candidate_float64')
validation=json.loads((OUT/'validation.json').read_text())
maxdiff=0.
for candidate in validation:
    folder=OUT/f'validation_candidate_{candidate["rank"]}'
    config=candidate['configuration']
    seq=joblib.load(folder/'sequence.joblib').predict(d.embeddings[v])
    residual=joblib.load(folder/'residual.joblib').predict(np.column_stack([r,seq,d.x[v,-25:]]))
    dual=retrieval_sequence_anchor(r,seq)+config['gamma']*residual
    for row in candidate['per_seed']:
        seed=row['seed']
        raw=read_predictions(SOURCE/('baseline_validation.csv' if seed==42 else f'baseline_seed{seed}_validation.csv'),d.keys[v])
        reference=old.predict(raw,*features,r,[d.records[i].sequence for i in v])
        pred=(1-config['w'])*reference['robust_v1_prediction']+config['w']*dual
        saved=read_predictions(folder/f'seed{seed}.csv',d.keys[v])
        diff=float(np.max(abs(saved-pred)));maxdiff=max(diff,maxdiff)
        assert diff<1e-7
        assert abs(rmse(d.labels[v],saved)-row['candidate']['rmse'])<1e-12
        assert abs(rmse(d.labels[v],reference['prediction'])-row['reference']['rmse'])<1e-12
decision=json.loads((OUT/'decision.json').read_text())
assert not decision['accepted'] and not decision['test_accessed']
summary=dict(status='passed',train_samples=len(y),validation_samples=len(v),
    baseline_oof_rmse=rmse(y,base),candidate_oof_rmse=rmse(y,p),
    replayed_validation_files=15,max_prediction_difference=maxdiff,
    residual_fold_fits=len(list((OUT/'fold_predictions').glob('*.joblib'))),
    new_ridge_fold_fits=len(list((OUT/'ridge_cache').glob('*.npz'))),
    validation_refits=3,training_script_hash_verified=True,source_cache_hashes_verified=True,
    independent_replay='Saved candidate models and production DualFusion.predict',test_accessed=False)
atomic_json(OUT/'verification.json',summary)
atomic_json(OUT/'artifact_hashes.json',{str(f.relative_to(OUT)):sha256_file(f)
    for f in sorted(OUT.rglob('*')) if f.is_file() and f.name not in ['artifact_hashes.json','REPORT_ZH.md']})
boot=json.loads((OUT/'bootstrap.json').read_text())['delta_rmse_95ci']
c=validation[0];seedrows=c['per_seed']
ref=np.array([s['reference']['rmse'] for s in seedrows]);new=np.array([s['candidate']['rmse'] for s in seedrows])
recipe=c['recipe']
lines=['# Dual 完整融合轻量调参结果','',
    '**结论：搜索与验证完成，没有候选达到预设接受门槛，保留原模型。**','',
    '## 执行范围','',
    '- 冻结编码器、检索规则和 robust 内部权重；复用经过训练范围核验的完整基线分折缓存。',
    '- A：当前残差模型的 11×5=55 组融合权重/残差缩放扫描；最优仍是原来的 w=0.5、γ=1。',
    '- B：24 个残差配置在固定的两个折筛选，前 8 个补齐五折。',
    '- C：前两个残差配置比较 5 个 Ridge alpha，复用 alpha=0.2 的已有预测。',
    '- 共保存 112 个分折残差模型、60 次新增 Ridge 分折拟合、3 组全训练集验证重拟合。',
    '- 搜索与验证用时 138.34 秒（不含此前缓存生成、脚本编写和事后核验）；未重新训练 GPU 神经基线。','',
    '## 最佳开发候选','',
    '| 参数 | 原模型 | 候选 |','|---|---:|---:|',
    '| dual 权重 | 0.5 | 0.6 |','| 残差缩放 | 1 | 1 |','| Ridge alpha | 0.2 | 0.2 |',
    '| pH 频率加权指数 | 0.25 | 0 |','| HGB 叶数 / 轮数 / 最小叶样本 / L2 / 学习率 | 7 / 50 / 80 / 30 / 0.05 | 同原模型 |','',
    '## 开发与验证结果','',
    '| 指标 | 原模型 | 最佳开发候选 |','|---|---:|---:|',
    f'| 7124 条分组 OOF RMSE | {rmse(y,base):.9f} | {rmse(y,p):.9f} |',
    f'| 7124 条分组 OOF MAE | {np.mean(abs(base-y)):.9f} | {np.mean(abs(p-y)):.9f} |',
    f'| 760 条验证集 RMSE，五种子均值 ± 样本标准差 | {ref.mean():.9f} ± {ref.std(ddof=1):.9f} | {new.mean():.9f} ± {new.std(ddof=1):.9f} |','',
    f'开发 RMSE 仅改善 {rmse(y,base)-rmse(y,p):.9f}，低于 0.003 门槛；只有 2/5 个折改善，低于至少 3 折门槛。',
    f'按同源簇配对 bootstrap 的 ΔRMSE 95% 区间为 [{boot[0]:.6f}, {boot[1]:.6f}]，跨越零。该区间是选参后的条件区间，未校正选择偏差。',
    '最佳开发候选的酸端/碱端误差也没有改善；原验证集五个基线种子全部变差。','',
    '| 种子 | 原验证 RMSE | 候选验证 RMSE |','|---|---:|---:|']
for s in seedrows:lines.append(f'| {s["seed"]} | {s["reference"]["rmse"]:.9f} | {s["candidate"]["rmse"]:.9f} |')
lines+=['','| 开发排名 | 验证 RMSE 均值 | 候选减原模型 | 接受 |','|---|---:|---:|---|']
for c in validation:lines.append(f'| {c["rank"]+1} | {c["mean_rmse"]:.9f} | {-c["rmse_gain"]:+.9f} | 否 |')
lines+=['','## 核验与解释边界','',
    '- 训练样本、分组、排除折、标签哈希、输入特征哈希和基线 checkpoint 哈希经过核验。',
    '- 原参数在五折上重拟合，dual 预测与旧缓存一致，容差 1e-7。',
    f'- 独立调用生产 DualFusion.predict 并重载候选模型，复算全部 15 份验证预测；最大差 {maxdiff:.3g}。',
    '- 五个种子为 0、1、2、3、42；共享 Ridge/HGB，仅神经基线种子变化，不是五次完整独立重训。',
    '- 分组 OOF 被用于选参，属于探索性开发结果，不是嵌套调参的无偏泛化估计。',
    '- 测试集未打开；没有合格候选，因此没有进行新测试或替换部署模型。',
    '- 结论仅适用于这次有限搜索，不能据此宣称原模型已达到全局最优。','',
    '## 复核入口','',
    '- scripts/tune_dual_lightweight.py：完整搜索逻辑；protocol.json 固定搜索空间和门槛。',
    '- scripts/verify_dual_lightweight.py：独立预测重放及指标核验。',
    '- best_oof.csv、validation_candidate_*/seed*.csv：逐样本结果。',
    '- fold_predictions/、ridge_cache/：分折模型与预测。',
    '- train_ranking.json、validation.json、bootstrap.json、decision.json：结果与决策。',
    '- verification.json、cache_audit.json、artifact_hashes.json：核验与来源哈希。']
(OUT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2))
