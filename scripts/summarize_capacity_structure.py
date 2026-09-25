"""Verify saved prediction counts and summarize capacity/structure ablations."""
import sys
import json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from develop_phgeofuse_regression import OUT
from phgeofuse.cache import atomic_json


def main():
    names=['compact_reliability_nested_20260915','structural_reliability_nested_20260915']
    base=np.load(OUT/'multiview_nested_20260915/dual_ridge_control_predictions.npz')
    records={}
    for name in names:
        folder=OUT/name
        assert json.loads((folder/'status.json').read_text())['status']=='complete'
        results=json.loads((folder/'results.json').read_text())
        for recipe,row in results.items():
            p=np.load(folder/f'{recipe}_predictions.npz')
            assert np.array_equal(p['keys'],base['keys']) and np.array_equal(p['y'],base['y'])
            assert np.array_equal(p['validation_keys'],base['validation_keys'])
            assert np.array_equal(p['yv'],base['yv'])
            assert p['prediction'].shape==(7124,) and p['validation'].shape==(760,)
            for field,expected,y in [('prediction',row['strict_nested'],p['y']),
                                    ('validation',row['validation'],p['yv'])]:
                q=p[field]
                assert np.isfinite(q).all()
                assert abs(float(np.sqrt(np.mean((q-y)**2)))-expected['rmse'])<1e-10
                assert abs(float(np.mean(abs(q-y)))-expected['mae'])<1e-10
        records[name]=results
    compact=records[names[0]]
    baseline=compact['l7_i50_p0.0']
    candidate=compact['l3_i50_p0.0']
    def gaps(row):
        d=np.array([f['heldout']['rmse']-f['train_rmse'] for f in row['fold_results']])
        return {'mean_signed':float(d.mean()),'mean_absolute':float(np.abs(d).mean()),
                'mean_positive':float(np.maximum(d,0).mean()),'folds':d.tolist()}
    checks={'all_predictions_recomputed':True,'train_count':7124,'validation_count':760,
        'test_access':False,'baseline_gap':gaps(baseline),'compact_candidate_gap':gaps(candidate),
        'scope':'expert-level development; no full-model five-seed final comparison',
        'objective_complete':False}
    atomic_json(OUT/'CAPACITY_STRUCTURE_20260915.verification.json',checks)
    lines=['# 2026-09-15 容量控制与局部结构实验','',
        '本轮完成 8 个容量/加权配方及 4 个结构配方的严格嵌套评估。仅使用 PHOPT 7124 条训练记录和 760 条原验证记录；未访问测试特征或测试预测，未使用 identity20 任务权重。', '',
        '## 主要结果','',
        '3 叶、50 轮、未加权的紧凑残差专家整体 RMSE 略优于上一轮候选，但配对家族区间包含零；没有获得可靠的整体增益证据。它可以作为较低容量研究候选，不能宣称完成原始目标。','',
        '| 配方 | 严格 RMSE | 低同源 RMSE | MAE | 酸性 bias | 碱性 bias | 中性 RMSE | 验证 RMSE |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for folder,results in records.items():
        for recipe,row in results.items():
            m=row['strict_nested']
            lines.append(f"| {recipe} | {m['rmse']:.6f} | {m['low_homology']['rmse']:.6f} | {m['mae']:.6f} | {m['acidic']['bias']:+.6f} | {m['alkaline']['bias']:+.6f} | {m['neutral']['rmse']:.6f} | {row['validation']['rmse']:.6f} |")
    ci=candidate['family_bootstrap_delta95_vs_reliability']
    lines += ['',f'紧凑候选相对上一轮候选的 RMSE 差值 95% 家族 bootstrap 区间：[{ci[0]:.6f}, {ci[1]:.6f}]。比较条件于当前分组和模型；重复开发的选择后偏差仍存在。', '',
        '## 不能只用平均有符号差距判断过拟合','',
        '| 诊断 | 原 7 叶残差 | 紧凑 3 叶残差 |','|---|---:|---:|']
    for key,label in [('mean_signed','平均有符号差距'),('mean_positive','平均正向差距'),('mean_absolute','平均绝对差距')]:
        lines.append(f"| {label} | {checks['baseline_gap'][key]:.6f} | {checks['compact_candidate_gap'][key]:.6f} |")
    lines += ['',
        '训练误差上升、正向差距缩小，同时留出误差没有恶化，支持减少容量。但均值接近零有折间正负抵消因素，平均绝对差距没有下降，不能声称整体过拟合消失。该诊断也混合了内层/外层基础模型训练量和输入分布变化。', '',
        '## 区域结构表征','',
        '读取原有 SaProt 每残基缓存及图缓存，校验序列哈希和长度；按表面/核心、DE、HKR、H、C 六种区域计算相对全局均值的表征，再用固定随机投影降到 64 维。附带区域比例、缺失标记和结构置信度。全部 7884 条开发样本完成，无标签参与提取，不需要重新跑大模型。', '',
        '每种新表示的 Ridge、检索、可靠性门控及残差均遵守内/外层排除。结构实验对照组重新拟合并要求重现上一轮 0.883278 结果。不能用对照未匹配的增益来归因。', '',
        '## 产物与验证','',
        '- `compact_reliability_candidate_20260915/`：较低容量候选权重、配置和验证；独立重新加载后，全部外层与验证预测差异为零。',
        '- `compact_reliability_nested_20260915/`：8 组配方、每折训练/留出预测、元模型输入缓存与权重。',
        '- `saprot_regions_20260915/`：区域特征、固定投影和序列/结构来源。',
        '- `structural_reliability_nested_20260915/`：4 组结构对照与每个排除集合的基础预测。',
        '- `CAPACITY_STRUCTURE_20260915.verification.json`：所有新结果的样本键、标签、有限性、RMSE 和 MAE 核对。',
        '- 两个区域聚合测试验证残基同步置换不变性、缺失区域和长度不匹配；可靠性门控两个测试覆盖梯度和留出行为，共 4 项通过。', '',
        '## 未完成事项与下一步','',
        '酸碱两端偏差仍然明显。本轮固定区域统计只检验了一种局部表征，不排除可学习的残基级/域级聚合。下一步应检验能够从 PHOPT 训练监督中选择关键残基的低容量注意力聚合，并用内层家族留出控制训练轮数；不要继续依赖整体拉伸或密集尾部权重网格。', '',
        '完整模型的五种子重训、同协议完整嵌套比较和最终冻结仍未完成；测试集已被查看，后续测试仍只能标注 follow-up test。']
    (OUT/'CAPACITY_STRUCTURE_20260915_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
