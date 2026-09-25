# Venus-DREAM 评估状态审计

2026-09-17。改动限于 Venus-DREAM 对照的 `reptile.py`、`maml.py` 和新增 `models/meta_evaluation.py`。DeltaRef 冻结实验的 46 个源码哈希仍完全一致；其模型、参考选择、超参数和验收门槛未改变。

## 问题与修复

- 旧验证和测试没有切换 eval，支持适配和查询预测都启用 dropout。现在两阶段关闭 dropout，保留支持适配所需梯度，支持集采用固定遍历顺序。
- 旧评估会消耗训练的全局 PyTorch RNG 和支持采样 generator；Reptile 在异常退出时还会残留适配后的参数。新增共享上下文在正常或异常退出时恢复模型模式、buffers、梯度引用和随机状态，Reptile 另恢复所有可训练参数。MAML 使用独立克隆进行适配，不复制整个冻结编码器的参数。
- MAML 验证原先对逐样本损失求和后除以 batch 数，结果随 batch 大小改变；现按查询样本数平均。默认单样本验证时该计数问题不改变原数值。
- MAML 在达到最大 epoch 时原先返回最后模型；现与提前停止路径一致，恢复已记录的最佳验证检查点，避免后期更差的权重直接进入测试。

训练期 dropout、支持采样、优化器及学习率未变。评估噪声消除后，检查点选择和训练随机轨迹可能不同，必须重新训练再报告新成绩。已有历史预测文件保持原样。

## 行为测试

`tests/test_meta_evaluation.py` 的六项测试同时检查 Reptile 和 MAML：

1. 同一检查点评估可重复，外部随机种子变化不影响输出，训练状态得到恢复。
2. 改变查询标签不会改变预测，改变支持标签确实会影响适配后的预测。
3. 查询顺序和分批变化不改变对应预测，并精确保留原来各子模块的 train/eval 状态。
4. 支持适配后查询发生异常，参数、模式和随机状态仍得到恢复。
5. MAML 验证结果对不同 batch 大小保持一致。
6. MAML 达到 epoch 上限时恢复最佳检查点。

六项新增测试和六项原训练优化测试全部通过。初始化专项的六项测试在前次修订已通过，初始化代码本次未改。未将单元测试通过作为预测性能证据。

```bash
CUDA_VISIBLE_DEVICES= python -B -m unittest discover -s tests -p test_meta_evaluation.py -v
CUDA_VISIBLE_DEVICES= python -B -m unittest discover -s tests -p test_training_optimizations.py -v
```

## 真实 RLAT 复核

使用本地真实 EpHod RLAT 检查点、缓存 ESM1v 表征及原始 PHOPT 训练任务 `train::P36639`（156 残基），其五条支持为 `M4I1C6 / P0AFC0 / P77788 / Q4V6M1 / Q55928`，均来自训练集。以 CPU、一次 inner step 作实现检查，三次都恢复同一初始权重，只改变运行随机种子：

| 随机种子 | 旧 train-mode 评估预测 | 修复后 eval-mode 预测 |
|---|---:|---:|
| 0 | 8.131711 | 8.677034 |
| 1 | 8.426938 | 8.677034 |
| 42 | 7.694289 | 8.677034 |

旧输出跨度 0.732649 pH；新输出跨度为零。三次修复后评估均精确恢复参数、全局 RNG、支持 generator 和各模块模式。此次仅核验一条训练任务的随机性，不用于判断准确率、总体方差、极端性能或五种子优势；也不能据此声称统计过拟合已解决。

原始结果与全部源码、检查点及表征缓存 SHA256：`experiments/delta_ref_phopt_20260916/analysis/venus_eval_state_audit_20260917/results.json`。

```bash
CUDA_VISIBLE_DEVICES= python -B scripts/audit_meta_evaluation.py --output experiments/venus_eval_recheck
```

## 剩余工作

继续补齐原始 PHOPT 残基缓存并进行固定五种子的 Venus-DREAM 受控训练。新比较应注明采用修复后实现；官方已发布 EpHod 单次输出仍单独保留其信息条件。DeltaRef 仍须通过完整分组验证、双端及中心护栏、触发时的 LoRA、最终五种子和领域强对照验收；本次没有替换默认模型。
