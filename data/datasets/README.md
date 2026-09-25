# PHOPT 固定测试集近源移除实验 v3（保留官方原始记录）

本目录已原位覆盖旧版。取消基础去重及标签冲突删除，仅按相似度删减训练集和验证集，测试集固定。
本次没有新增备份。旧版本的结果不能与本版本混用。
官方 `data/phopt_*.fasta` 保持原样；所有新条件逐字节复制官方 1971 条测试记录。

| --dataset | 训练 | 验证 | 固定测试 |
|---|---:|---:|---:|
| fixedtest_control | 7124 | 760 | 1971 |
| identity100 | 7099 | 756 | 1971 |
| identity50 | 5421 | 574 | 1971 |
| identity30 | 3800 | 411 | 1971 |
| identity20 | 3242 | 344 | 1971 |

`fixedtest_random100_seed0` 至 `seed4`、random50、random30、random20 共 20 个随机对照。
从官方原训练池（7124 条）和原验证池（760 条）分别随机删减，不从已经近源过滤的集合抽样。
训练集、验证集分别与相应 identity 数据集严格匹配样本数及 pH×长度分箱分布。
每个随机对照的训练、验证数量与上表中对应的 identity 条件一致。
随机组不要求满足近源隔离：其作用正是与定向删除作对照，近源序列可能随机保留。
seed 后缀是数据抽样种子，与模型训练种子不同。训练 RNG 使用 seed，验证 RNG 使用 100000+seed。
fixedtest_control 使用完整官方记录，不施加近源删除，三份 FASTA 均与官方文件逐字节一致；
`phopt` 仍指官方原始划分。两者记录相同，control 采用本实验的独立输入目录及版本指纹。

## 输入记录与相似度判定

- 保留同序列同标签不同 ID 的全部记录，也保留同序列不同标签的全部记录；不平均标签、不重新分配权重。
- 只进行格式、序列字符、ID 唯一性和训练/验证标签及权重有限性检查；异常报错，不静默删除记录。
- 重复和标签分歧仅统计于 input_audit.tsv；统计单位是记录，唯一序列数另见 metadata.json。
- 随机对照按记录 ID 抽样，不按唯一序列抽样；不同 ID 的同序列记录可能分别入选。
- 训练与测试之间 identity >= 名称阈值且双方覆盖度 >= 80% 的命中使训练样本被移除。
- 验证集与训练集使用相同阈值分别对测试集过滤：identity50 用 50%，identity30 用 30%，依此类推。
  训练集与验证集之间并未另加低相似度隔离。
- identity100 表示完整比对一致性为 100% 且满足覆盖条件，不只是全序列哈希去重。
- 使用 MMseqs2 双向高灵敏度搜索与正向无预筛选比对的并集，额外用序列哈希检查完全一致。
- 实际参数、版本和源哈希见审计目录 protocol.json。E-value 上限为 100；由 nident/alnlen 计算一致性，
  由比对坐标/序列长度计算双方覆盖度，避免仅以三位小数输出决定边界。保留 backtrace 以取得有效 nident。
- 这是限定算法/比对条件下的低序列相似度实验，不证明绝对不存在远缘同源关系。
- 官方 sample_weight 原值保留，没有重新估计。原始权重来源须在论文中披露；无权重主指标单独报告。
- 通用结构、SaProt 嵌入及原仓库 ESM2 特征按原始序列/ID复用，未使用测试标签构造检索支持集。

## 文件

每个子目录含兼容原程序的三份 `phopt_*.fasta`、`records.tsv`、`metadata.json`、
`test_nearest_neighbors.tsv`、`selection.tsv`。最近邻表记录每条测试序列对该训练条件的最近邻；
空白表示未检出满足条件的命中，不能解释为零相似度。selection 表逐条记录训练/验证是否保留及参考近源移除条件的选择。

本版本审计目录为 `data/dataset_audits/fixed_test_removal_v3/`，与模型实验结果目录分开保存。
本次针对完整官方训练/验证记录重新执行全部比对；缓存签名包含实际搜索 FASTA 哈希，不能误用旧清洗池的结果。
包含原始比对、只记录不删减的 input_audit.tsv、相似度移除日志、协议、数据统计、训练输入检查和生成日志。
旧 cleaning.tsv 已移除，避免将上一版的标签删除记录误认为当前规则。
本次原位覆盖数据集、manifest 和 baseline 预处理输入，并使旧 PHGeoFuse 检索缓存失效；不创建备份。

## PHGeoFuse 直接训练

在 WSL 项目根目录及 phbench 环境中运行：

```bash
python -m phgeofuse.train --config configs/phgeofuse_fixedtest_base_seed0.yaml --dataset identity20
```

seed0 至 seed4 的基础训练配置已经生成。它们使用原 tuned_v1 基础架构和训练配方，
完整训练任务网络（通用 SaProt 编码器仍为 frozen），不加载 PHOPT 的旧任务检查点。
全部条件已有完整的 manifest 和结构/图/嵌入。首次训练会自动为当前训练集生成独立检索缓存。
新基础运行名为 phgeofuse_fixedtest_base_v3，指纹包含协议版本，避免复用旧版任务检查点。
这里的 v3 表示数据协议版本，与 PHGeoFuse v3 模型名称是不同概念。

基础阶段完成后，如需 v3 同源残差门控：使用原 v3 配置加同一个 `--dataset`，
`--init-checkpoint` 必须指向该条件刚训练的基础模型，再在同条件验证集校准。
v3 的训练种子还需与对应基础阶段一致设置。数据集指纹会拒绝旧划分或其他条件的任务检查点。
当前固定测试集数据仍可能包含通用预训练资源的序列重叠，未进行全量预训练库审计。

## Venus-DREAM MAML/Reptile

全部新条件已经重新生成 `data/processed/<dataset>/top5/esm2_opt_retrieval/` 和
`esm2_opt_random/`，含 train/valid/test JSON。支持序列只能来自当前条件训练集，训练查询排除自身。
可直接沿用原训练命令并设置 `--dataset identity20 --topk 5 --retrieval_strategy opt_retrieval`。
仅排除训练查询自身的 ID，不排除其他 ID 的相同序列；同序列记录可能占据多个支持位置，与保留原始记录的主协议一致。
`--random_test` 所需 top5 随机支持文件也已提供，支持抽样固定 seed42。
其他 topk 或检索策略应使用原 retrieval.py 在当前数据集上重新预处理。
ESM2 原始特征的哈希与生成来源见每组 processed 文件的 provenance.json。

EpHod 可读取相同 FASTA；其 server11 环境和任务预训练数据本次未修改。远端使用时需传送这些文件并独立隔离任务检查点。

## 重现

```bash
python scripts/build_fixed_test_datasets.py --work data/dataset_audits/fixed_test_removal_v3 --mmseqs mmseqs --threads 8
```

先在审计目录生成、检查，再追加 `--install` 原位覆盖数据及旧预处理缓存，不保存旧版副本。
随后运行 `python scripts/prepare_fixed_test_training.py --work data/dataset_audits/fixed_test_removal_v3`，
再运行 `python scripts/audit_fixed_test_datasets.py` 和 `python scripts/check_fixed_test_training.py --dataset identity20 --device cpu`。
协议确定后不要根据测试成绩改变阈值或挑选随机对照。

工具依据：[MMseqs2 官方说明](https://github.com/soedinglab/MMseqs2)。
