# Venus-DREAM PHOPT 受控比较：初始化与输入审计

核查日期：2026-09-16。此次工作修复未来 Venus-DREAM 对照实验的初始化和检查点问题，没有改变当前完整 PH-GeoFuse 或冻结中的 DeltaRef 模型、超参数及验收门槛。以下检查不能作为极端预测性能改善的证据。

## 已确认的问题和修复

1. `pHPredictionModel(pretrained=False)` 原先仅重置可训练参数，继承了旧监督 EpHod 检查点的五层 BatchNorm 统计量。实际 `num_batches_tracked` 均为 535536；主层均值绝对值的均值为 1.02360，方差均值为 0.20962。训练模式又固定 BatchNorm 为 eval，因此统计量一直参与后续预测。新训练现在将 RLAT 的运行均值归零、方差置一、计数归零；ESM 冻结状态和 `pretrained=True` 保持原行为。
2. RLAT 构造函数会多次调用检查点配置中的 `torch.manual_seed(10)`，旧 fresh 初始化因此忽略调用者种子。现在在加载前记录调用者 seed，初始化时通过隔离的随机状态使用该 seed。两入口原有的训练期 reseed 保持不变。
3. 旧轻量检查点只保存参数，未保存影响推理的 BatchNorm buffers。新文件保存完整任务头参数和 buffers，仍不保存 ESM。读取旧参数文件会恢复原 EpHod buffers 并发出说明；这要求本地原 EpHod 检查点与历史训练一致。部分缺失 buffers 的文件会在修改模型前被拒绝。

旧 `pretrained=False` 实验仍依赖监督任务统计量，不能因开关名称把它们重新归为 PHOPT-only。已有历史分数和默认模型未替换。这里核查的是 Venus-DREAM 对照路径；不能由此推断 DeltaRef 差值支路的过拟合来源。

## 验证证据

新增 6 项行为测试覆盖新初始化不依赖旧统计量、种子作用、预训练兼容、一步优化后磁盘重载、旧格式预测兼容和损坏缓冲状态拒绝。原 6 项训练优化测试也通过，包括 MAML 梯度、Reptile 参数恢复和分布式采样。

真实本地 RLAT 检查点的 CPU 核验使用缓存训练序列 `train::Q938E9`（108 残基），仅替换 ESM 加载以复用已有表征：

- `pretrained=True` 的所有权重、buffers 与原检查点逐项完全一致。
- fresh 的五层 BatchNorm 状态全部重置；seed 0 重复初始化全部任务头状态哈希相同，seed 1 不同。
- 旧参数格式重载和新任务头格式重载均产生完全一致的预测。
- 运行前后冻结主实验的 46 个源码哈希全部保持一致。

原始 JSON：`experiments/delta_ref_phopt_20260916/analysis/venus_fresh_init_audit_20260916/results.json`。保存了模型、代码、输入缓存和主协议的 SHA256。

```bash
CUDA_VISIBLE_DEVICES= python -B -m unittest discover -s tests -p test_task_head_initialization.py -v
CUDA_VISIBLE_DEVICES= python -B -m unittest discover -s tests -p test_training_optimizations.py -v
CUDA_VISIBLE_DEVICES= python -B scripts/audit_task_head_initialization.py --output experiments/venus_init_recheck
```

## 原始 PHOPT 输入审计

逐项核对 `data/processed/top5/esm2_opt_retrieval/retrieval_{train,valid,test}.json` 与 PHOPT manifest 的 ID、序列、标签及划分。每条查询具有五个不同的训练支持 ID，无查询自身；全部支持序列与标签均和训练记录一致。此处只验证原始划分，不能把这些支持文件直接用于需要排除同源家族的嵌套验证。

| 项目 | 训练 | 验证 | 测试 |
|---|---:|---:|---:|
| 查询数 | 7124 | 760 | 1971 |
| 强酸 pH<=4 | 123 | 13 | 31 |
| 强碱 pH>=10 | 63 | 11 | 22 |
| 支持引用次数 | 35620 | 3800 | 9855 |
| 不同支持 ID 数 | 6559 | 2168 | 4462 |
| 最长查询序列 | 1021 | 1017 | 1012 |
| 已有查询残基缓存 | 373 | 0 | 4 |
| 查询及支持所需不同缓存键 | 7073 | 2914 | 6386 |
| 缺少的缓存键 | 6708 | 2773 | 6106 |

本划分所有序列均不超过 loader 的 1022 残基限制，样本覆盖率为 100%，未因长度删掉任何极端样本。缓存计数仅检查文件存在，不表示已完整验证其内容；三个划分有共享训练支持，不可简单相加估算总编码量。

原始 JSON：`experiments/delta_ref_phopt_20260916/analysis/venus_input_audit_20260916/results.json`，含输入、loader、模型、缓存代码与 Reptile 的哈希。

```bash
CUDA_VISIBLE_DEVICES= python -B scripts/audit_venus_phopt_inputs.py --output experiments/venus_inputs_recheck
```

## 尚未完成

完整强基线还需补齐残基编码，随后按预先固定的五种子重新训练并归档完整预测。后续已完成 meta-validation 和测试阶段模型模式及随机状态的修复，详见 [评估审计](venus_phopt_evaluation_audit_20260917.md)。修复初始化和评估是受控比较的前提，并不证明新训练已优于旧模型。上述历史 JSON 保留审计时的源码哈希；后续代码修订记录于新审计，不覆盖旧证据。

23:48 主 DeltaRef 进程 PID 8440 存活，完成 3/5 外层，排除折 `[3]` 的完整基线刚完成，耗时 1773.88 秒。主实验仍按原冻结协议继续；尚无最终五种子新模型性能。
