# PHOPT 受控领域对照

本轮使用原始 PHOPT 7124/760/1971 划分，不改变 DeltaRef 冻结模型和选择规则。所有测试指标使用未加权评价代码。官方已发布 EpHod 输出作为额外证据保留；下面的本地重训与它分开归档。

## 表征缓存

入口：`scripts/cache_phopt_field_embeddings.py`。输出：`experiments/field_comparisons_phopt_20260917/esm1v_full_precision/`。

- 使用本地通用 ESM1v checkpoint，float32，关闭 autocast 和 TF32，单序列编码。
- 只消费 ID、序列和划分，不消费 pH 标签；9774 条不同标准化序列覆盖 9855 个样本。
- 保存每个完整 token 的表示，包括 BOS/EOS；没有 padding、截断或样本过滤。原始 PHOPT 最长序列 1021 残基。
- 为官方 EpHod-SVR 保存所有 token 的均值，保持官方 batch-size-one 推理约定。不能用本项目只含残基 token 的 mean/std 表征冒充它。
- `protocol.json` 记录 checkpoint、代码、manifest 的哈希和计算精度；完成文件记录全部 token 文件及汇总表征的哈希。缓存不与来源未认证的旧目录混用。
- `runner.log`、`runner.pid.json`、`status.json` 和 `encoding_events.jsonl` 记录进度与成本。

实测短/中/长序列为 33/396/1021 残基。首次调用含 warmup 为 1.316 秒，中位长度 0.091 秒，最长 0.218 秒；最大 PyTorch 分配显存为 2891226112 字节。这是资源探针，完整吞吐以实际编码日志为准。

## EpHod-SVR 组件

入口：`scripts/train_ephod_svr_control.py`。输出：`experiments/field_comparisons_phopt_20260917/ephod_svr_control/`。

明确标为 EpHod-SVR 组件，不能替代完整 EpHod 集成。监督来源仅为 PHOPT 训练集，通用 PLM 预训练保留。SVR 是确定性算法，报告一次独立拟合；不把重复 seed 名称当成独立随机实验。

官方提交 `e823cd2f1172258dc1e81cc00326e6975f22d10a` 的训练代码规定：kernel={poly,rbf}、gamma={scale,auto}、C=10^-5 到 10^4、五种标签权重，共 200 个不同配方。作者的无重复抽样数量也为 200，因此覆盖整个网格。本地采用固定枚举顺序，仅在验证集上最小化官方 bin_inv 加权 RMSE；并列时依次使用未加权验证 RMSE和配方序号。报告成绩使用未加权 RMSE/MAE/偏差。

样本权重直接调用已审阅且 Git blob 校验通过的官方 `trainutils.py`。标准化均值和总体标准差仅由训练表征计算，保持 `std+1e-8`。预计算核只复用不同 C/权重配方之间相同的计算，单元测试确认与原生 sklearn SVR 在 poly/rbf、scale/auto 四种设置下的输出一致。epsilon=0.1、tol=0.001、shrinking=True、max_iter=-1 保持默认，未增加输出拉伸或裁剪。

训练阶段没有测试分数。完成全部配方后写 `release.json`，绑定选择结果、模型及来源；单独 `--phase evaluate` 才计算测试输出，且拒绝改动后的源码、输入路径、表征或检查点。该测试历史上已被查看，归类为后续验证。

```bash
python -B scripts/train_ephod_svr_control.py \
  --cache experiments/field_comparisons_phopt_20260917/esm1v_full_precision \
  --output experiments/field_comparisons_phopt_20260917/ephod_svr_control --wait-for-cache
```

## Venus-DREAM Reptile

入口：`scripts/train_venus_phopt_control.py`；固定配方：`configs/venus_phopt_control.json`。

沿用仓库现有 Reptile 超参数：50 epoch 上限、meta_lr=1、inner_lr=0.001、5 次内层更新、5 个训练支持、global meta batch=5、每 200 步验证、patience=5、min_delta=0.0001。运行种子为 0/1/2/3/42，每个种子单独训练和保存最佳验证检查点。使用修复后的 fresh RLAT 初始化和评估状态，监督信息为 PHOPT-only；未来若单列任务预训练条件，须使用独立配置与目录。

启动前重新审计查询、支持集和全部 token 内容哈希。已验证旧原始划分支持全部来自训练集，不包含查询自身；这不能使该支持库自动适用于排除家族的嵌套验证。

模型初始化后卸载冻结 ESM 编码器，使用已认证的相同 token 缓存；若缓存意外缺失则报错，不自动切换到其他精度的编码。任务头、掩码与支持适配计算保持现有实现。所有未完成训练尝试保留，恢复时在新 attempt 目录从固定 seed 重训，不能宣称严格恢复优化器状态。

`--phase profile` 对训练集中短、中、长三项任务测资源，不产生可参赛检查点；`--phase train` 完成五种子后冻结 release；`--phase evaluate` 在冻结来源通过核验后才逐种子测试。逐样本标签使用 manifest 原始精度，指标为各种子先计算后取均值。

```bash
python -B scripts/train_venus_phopt_control.py \
  --cache experiments/field_comparisons_phopt_20260917/esm1v_full_precision \
  --output experiments/field_comparisons_phopt_20260917/venus_control --phase profile
```

完整 EpHod 的 RLATtr 分支、完整 Venus 五种子和最终家族 bootstrap 仍需完成。组件实现、来源审计及资源测试不构成领域领先证据。

## 2026-09-17 启动记录

ESM1v 全部9774条不同序列已完成编码，覆盖9855个样本。SVR 进程 PID 32840 已在 CPU 上执行200配方搜索；Venus 进程 PID 34015 已启动固定五种子训练。各目录 `runner.pid.json` 保存完整命令，`runner.log` 保存运行日志。PID 为此时快照，恢复前应重新核对实际 cmdline。

Venus 的三任务资源测试通过：常规任务0.329秒、长任务0.403秒，峰值分配显存1601871360字节；短任务首次调用2.488秒包含warmup。测试采用完整五次内层更新。按常规吞吐，7124训练查询约39分钟，另有每200步的760条验证；五种子总时长取决于实际早停，不能从三条任务精确外推。

WSL 下 cuDNN 需要在启动进程前将 `/usr/lib/wsl/lib` 加入 `LD_LIBRARY_PATH`；缺少该项会出现 `libcudnn_cnn_infer.so.8` 无法加载 `libcuda.so`。首次资源探针因此退出，确认终止后按 README 的路径设置重试并成功。当前训练启动环境已经包含该路径。

后续结果：SVR的200配方训练和冻结后测试已完成。原始测试整体RMSE=0.90625、强酸RMSE=1.54302、强碱RMSE=1.84082。两端优于当前完整模型，整体明显较差，不能替换默认模型。详细报告位于 `experiments/field_comparisons_phopt_20260917/ephod_svr_control/REPORT_ZH.md`。这是一次确定性组件重训；完整EpHod集成仍未完成。
