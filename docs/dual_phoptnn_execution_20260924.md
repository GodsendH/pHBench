# Dual / pHoptNN 前三阶段执行记录

实验目录：`experiments/dual_phoptnn_20260924/`。状态：Dual 基线已完成；pHoptNN 在 CUDA 异常后恢复训练，尚无正式互补性结论。

## 后续恢复与主对照调整

2026-09-24 晚核查：修正后的完整 Dual 已完成并导出 760 条验证预测、1,971 条测试预测；验证 RMSE 0.80949、测试 RMSE 0.77370。历史完整 Dual seed42 对应为 0.80827、0.77087。后续互补性主对照改为冻结的历史 Dual，修正后的 Dual 仅作独立对照。

pHoptNN seed42 完成 7 轮后，在第 8 轮发生 CUDA illegal memory access 并退出。原堆栈为异步报错，尚不能确定出错算子或根因。已从第 7 轮末 checkpoint 恢复，保留模型、优化器、调度器与原早停规则；启用 `CUDA_LAUNCH_BLOCKING=1`，数据加载 workers 改为 0。原有训练配置未变，但不声称恢复后与旧多进程加载逐步数值完全一致。

恢复后的全量图检查覆盖 9,774 个图：无越界边、非法原子类型、非有限值或形状错误，见 `cuda_recovery_graph_audit.json`。运行时选择和源文件哈希见 `phoptnn/recovery_protocol.json`。恢复日志为 `phoptnn/resume_seed42.log`。

完成第一个种子的正式训练后即生成 `complementarity_historical/REPORT_ZH.md`，后续加入 seeds 0、1 更新三种子分析。测试仅在该种子达到原定训练结束条件后导出，固定融合权重只在验证集选择。低 pLDDT 组较小：验证集 <70 共 6 条；测试集 <70 共 24 条，其中 <50 为 3 条，需谨慎解释。

## 已完成并验证

- 冻结原始 PHOPT 划分：训练 7,124、验证 760、测试 1,971，共 9,855 条。
- 在独立目录修正 273 个 ESMFold 结构的 pLDDT 量纲，对应 280 条样本记录；原始坐标与历史文件保留。
- 迁移 273 份受影响残基图；276 条记录的 3Di 改变，实际生成 269 份唯一的新 SaProt 特征。
- 重建全部样本相对训练库的检索特征。
- 独立校验源/目标 PDB 的非置信度字段、图中非置信度张量、标签、划分、特征键和检索训练来源，见 `confidence_verification.json`。
- 完成 9,774 个唯一原子图，覆盖全部 9,855 条，无转换失败，无孤立节点；34 个图因 RDKit 分子处理失败使用明确记录的距离特征回退。见 `atom_coverage_verification.json`。
- 将上游 pHoptNN 的模型核心放入 `phgeofuse/phoptnn_adapter/`，保留 MIT LICENSE、上游 commit 与源文件哈希；仅模型导入改为相对命名空间。
- 使用 manifest 驱动的数据集与训练器，修正 padding 边偏移、不足一批的形状和 PQR 固定列解析。原子图重建 RDKit 重原子顺序并验证序号，避免直接把 CIF 索引当作 PQR 索引。
- 完成 7 个针对性单元测试、真实结构的 GPU 前向/反向与单独/批量一致性检查，以及独立小样本两轮训练、checkpoint 恢复与预测导出检查。小样本结果仅证明接口可运行，不用于模型性能报告。
- 建立不使用标签的 30% identity / 80% coverage 序列簇，供互补性报告的配对 bootstrap 使用。

## 正在运行

修正后的 PHGeoFuse seed42 从头训练；随后执行历史配方的 v3 门控、验证校准、验证/测试预测。Dual 的 robust v1 及双编码器残差专家也在修正后的检索特征上重拟合，保留原先固定配方和 0.5/0.5 融合。

CPU 上的 Dual 专家重拟合可以与 GPU 基线训练并行。pHoptNN 的正式三种子训练（42、0、1）在 Dual GPU 阶段完成后串行执行，原子图已全部就绪。每次训练固定使用原始 PHOPT train/validation/test 划分，不再对预组 batch 做随机切分。

实时证据：

- `dual_baseline/pipeline_status.json`、`baseline_train.log`：基线及其后续阶段。
- `dual_refit/status.json`、`dual_refit_early.log`：Dual 专家重拟合。
- `phoptnn/pipeline_status.json`、`phoptnn/seed*/status.json`：pHoptNN 排队/训练。
- `execution_protocol.json`：冻结的实验选择规则。
- `implementation_snapshot.json`、`source_snapshot/`、两个 `*_pip_freeze.txt`：代码和依赖快照。

状态文件必须结合对应 PID、命令行与 checkpoint 核查，不能只凭旧状态文件重启或判断任务已经完成。

## 训练结束后的分析

使用 `scripts/analyze_dual_phoptnn.py` 生成 `complementarity/results.json` 与 `REPORT_ZH.md`。比较同一批样本上的历史 Dual、修正后 Dual、pHoptNN 和固定权重融合；融合权重仅由验证集上的 0:0.05:1 网格选择。

报告全量与低 pLDDT、低同源、结构来源、极端 pH 分组的误差、覆盖率、误差相关性和同源簇级 bootstrap。三种子变化来自 pHoptNN，Dual 固定 seed42，应明确这一点。当前 PHOPT 测试集历史上已经被查看，结果属于后续比较；oracle 仅为诊断上界，不是可部署结果。本阶段不训练新的学习式融合门控。

## 恢复入口

先确认已有进程是否仍活跃，避免重复启动。已退出的训练可通过其 `last.pt` 恢复：PHGeoFuse 使用原 `phgeofuse.train --resume`；pHoptNN 使用 `phgeofuse.phoptnn_adapter.train --resume` 并保持原配置。两者的批次、归一化、种子和检查点配置必须一致。

本轮 pHoptNN 配方采用上游 Best_hp 第 6 行的模型规模与学习参数，从头拟合，固定电荷缩放为 1，并记录 float32、梯度裁剪、训练集均值归一化的 LDS 权重等适配细节；它是同数据协议重训，不能在完成论文配方核查与对照之前称为论文成绩的严格复现。
