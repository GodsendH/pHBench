# 稀疏残基 pH 响应场：候选与冻结协议

本轮目标是在原 PHOPT 性能保护下改善真正的两端误差，并验证是否减少跨家族过拟合。正式实验已完成，三个分支均未通过既定双端门槛，默认模型没有替换。ΔRef LoRA继续按用户要求暂停。

2026-09-17 完成状态：75次拟合共108.12分钟，来源、家族隔离、选择规则、预测及三个分支的CPU重载均已复核。完整结果见[完成报告](../experiments/residue_field_phopt_20260917/analysis/REPORT_ZH.md)，后续方向见[环境监督与输出残差设计修订](extreme_ph_phenv_revision_20260917.md)。以下保留本次实验的原始候选设计和冻结协议，不应理解为仍在训练或已证明有效。

## 为什么换到残基层

之前的整体均值/标准差、局部可电离残基统计、来源校正及线性密度模型均未通过双端门槛。这排除了本轮固定统计特征上的几条廉价路径，不证明序列中没有可用信息。本候选保留每个残基，通过小型共享网络学习在不同pH下哪些局部表征可能有用。

## 模型

输入为认证的ESM1v float32全残基表征，经固定、无标签128维正交随机投影后缓存为float16。移除BOS/EOS，不截断、不抽样、不删除任何残基；全部7124条训练序列、所有残基都保留。推理不需要真实pH、EC或已知活性位点。

逐残基LayerNorm、128→32共享映射、GELU、dropout 0.25，产生残基隐变量。全局均值/标准差进入一个32维小头，在13个固定pH径向基上产生全局响应。稀疏分支对D/E/H/C/K/R/Y残基产生同一组基系数，在每个候选pH上取最多8个有效残基的最大响应并平均，然后加到全局响应。全流程在0–14、步长0.25网格输出响应。

**这不是物理酶活曲线、pKa估计或校准置信区间。** 化学残基类型只是先验筛选；被模型选中的残基不能直接叫催化位点。没有任务训练的Transformer注意力、等变图、近邻标签注意力、类别多专家、测试时梯度适配或PLM LoRA。稀疏版本7309个可训练参数，仍可能过拟合，不能以小参数量代替实证。

三个预先固定对照：普通单值回归（同残基隐变量的全局均值/标准差）、全局pH响应场、带稀疏局部贡献的响应场。所有对照使用相同缓存、隐藏宽度、优化器、dropout和家族划分；报告参数量差异，不声称严格逐参数相同。

## 训练和选择

- 原PHOPT训练集7124条、3186个现有同源组；五外层、四内层，每个外层完整排除其标签、训练样本及完整基线参考库。原验证/测试成绩不参与本轮选择。
- 每个训练子集重新初始化小网络，不更新ESM；seed42、batch32、AdamW、lr0.001、weight decay0.05、gradient clip1、40轮上限、patience5。
- 直接回归用MSE。响应场以sigma0.5的软标签NLL训练；log训练标签先验显式加入响应，同时加0.25自然先验均值MSE和0.01响应曲率惩罚。先验只在当前训练子集拟合。
- 内层比较推理先验指数1/0.5/0及均值/峰值决策，融合强度0/0.25/0.5/0.75/1。使用相同整体/中心RMSE、MAE+0.01和中心错误极端率+0.005保护条件，然后最小化两端相对RMSE较大值。
- 选检查点先比较上述融合效用，完全并列时比较自然先验支路的`全体MSE + 0.05*(酸端MSE+碱端MSE)`。早停使用后者、最小改善0.002。如此允许融合强度暂时为零时仍追踪支路本身是否改善。
- 外层重训轮数为该结构四个内层最佳轮数中位数（取较小整数）；决策及融合强度来自内层拼合结果。三种结构也仅按内层效用选择，完整外层同时报告各结构和嵌套选择器。
- 原内部双端改善门槛保持不变。开发保护容差不等于严格整体性能不变；通过开发后仍须原始PHOPT复核、五种子完整系统和同信息条件领域强基线。最终还需要确认性数据，不能把反复查看的开发折当独立外部证据。

## 本机资源与恢复

真实4274条训练、1425条查询的稀疏模型探针：2轮训练加查询/训练预测共约10.14秒，峰值PyTorch分配82333696字节。按75次拟合、每次10轮粗估约1.06小时，实际早停/验证吞吐会影响时长。监督器对整组进程设置8小时硬上限，小于用户10小时边界。没有自动拆成无限重启的长任务，也没有自动集群或LoRA切换。

正式输出为`experiments/residue_field_phopt_20260917/nested_v2/`，日志`nested_v2.log`，监督状态`nested_v2_budget.json`。源码、缓存和已有完整基线排除预测均冻结哈希。每个子拟合保留学习曲线、训练/查询键、训练标签哈希、模型权重、训练先验、预测及其哈希。完整子拟合只在来源完全匹配时复用；中断子拟合保留原文件后从固定seed重训，不能把它叫严格优化器状态恢复。

早期`nested/`试跑约116秒后主动停止，原因是零修正的融合分数并列会把检查点锁在第1轮。其源码副本保存在`nested/interrupted_source/`；它不是有效完整实验，不与`nested_v2`合并。该修复在任何完整外层结果形成前实施，未使用外层指标选择改变。

## 验证入口

```bash
cd /home/hetianci/projects/Venus-DREAM
conda activate phbench
python -B -m unittest discover -s tests -p test_residue_field.py -v
python -B scripts/check_residue_field_artifacts.py \
  --experiment experiments/residue_field_phopt_20260917/nested_v2 \
  --features experiments/residue_field_phopt_20260917/features/tokens.npz \
  --fit experiments/residue_field_phopt_20260917/nested_v2/outer0/direct/inner1 \
  --output experiments/residue_field_phopt_20260917/reload_verification.json
```

5项行为测试已通过，覆盖padding不变性、短序列/无可电离残基的有限梯度、训练先验不受查询改变、零修正并列时可选择改善的后续轮次，以及修改外层查询标签不改变固定轮数拟合权重/先验/预测。最终direct、global和sparse检查点均独立在CPU以虚拟查询标签重载通过：与GPU保存结果的最大差异分别为4.77e-7、1.91e-6和1.91e-6；各峰值决策完全一致。最终结果见reload_verification_final_direct.json、reload_verification_final_global.json、reload_verification_final_sparse.json及analysis/verification.json，75次拟合的最终审计已完成。

完成后运行的报告入口（不会训练或重新选择方案）：

```bash
python -B scripts/report_residue_field.py \
  --experiment experiments/residue_field_phopt_20260917/nested_v2 \
  --budget experiments/residue_field_phopt_20260917/nested_v2_budget.json \
  --output experiments/residue_field_phopt_20260917/analysis
```

该报告从75个完整拟合重建检查点轮次、内层决策及外层融合，复算最终指标，并输出分支训练/跨家族留出差距、偏差分段图、学习曲线和来源核验。单种子的开发结果不替代五种子确认。

## 设计差异的证据边界

仓库已核查的EpHod（RLAT/SVR）、Venus-DREAM（支持集适配）、OpHReda（检索标签注意力）、EnzOracle（类别引导MoE）、CatOpt（多尺度CNN/注意力）和pHoptNN（原子电荷等变图）与本分支核心运算不同。共享PLM、池化、连续密度、top-k等均是通用已有技术，当前并不能证明领域首次。最终系统若仍融合旧PHGeoFuse，必须披露结构/检索基线的使用，不能把新小头说成完全独立的纯序列系统。
