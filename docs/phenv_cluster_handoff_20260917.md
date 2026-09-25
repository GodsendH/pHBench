# pHenv 集群计算交接

状态：首轮19473条pilot完成，五种PHOPT迁移配方均未达标，尚无性能证据支持直接放大全量。已有可运行的扩展数据准备、分片编码、聚合校验和辅助头预训练入口；尚未连接、提交或验证集群任务。调度器/GPU/存储/路径信息仍待用户提供。扩展数据尚未完成全部同源认证。

## 资源依据

原始全量候选约189.6万条，筛查前热身后的ESM1v探针估计28.59单卡3090 Ti GPU小时、fp16残基缓存1.753 TB。扩展计划实际包含151.9万条训练/固定验证候选，对应22.88 GPU小时、1.401 TB。均为float32推理、batch1的小规模长度分层估计，不含全量磁盘写入、同源清理、辅助头训练或集群排队。

最新记录见 `experiments/phenv_transfer_20260917/warm_profile/result.json`；旧 `profile/result.json` 的36.66小时估计把首次CUDA冷启动计入短序列平均并放大，已由显式热身复测修正，原文件保留。新探针对已完成pilot预测1027.82秒，实际1075.47秒，可作量级交叉检查。应先根据最终保留记录重算规模，再在目标GPU上测吞吐。

不能以“冻结ESM”为由忽略编码成本；不能在本机把全量任务拆成连续重启来规避10小时限制。当前本机任务分别有4小时和2.5小时进程组硬上限。

## 数据先决条件

编码输入必须含 `records.csv`、`complete.json` 和独立 `verification.json`，并且SHA256相互匹配。每条记录至少有 `key,sequence,phenv,organism,split,train_weight`，其标签是pHenv；PHOPT留出标签不进入准备或预训练。验证记录不用于拟合训练权重。

目前全量PHOPT正向同源筛查仍在本机执行，来源生物/辅助验证同源清理的扩展版本尚未认证。**本交接入口不把全量原始CSV直接提升为可训练清单。** 小规模数据可用于集群环境冒烟，但不能据此声称完成全量预训练。全部最终筛查条件和每条排除原因必须随全量数据交付。

## 扩展数据的CPU准备任务

已生成 `experiments/phenv_transfer_20260917/expansion_plan/plan.json`：1513391条训练候选、固定5640条pilot验证蛋白；验证来源生物的其余377195条序列全部留在训练之外。训练候选有545472554个残基。这里的“全量”指纳入所有合格训练来源序列，验证仍是固定来源抽样，不是发布方原始划分。

计划生成实测46.21秒，15分钟硬上限内完成。训练候选按50000条分成31片，共93项CPU搜索，任务号0至92。每片有PHOPT→训练、训练→辅助验证、辅助验证→训练三个方向；已经运行的全量pHenv→PHOPT正向结果另作必需输入。分片改变启发式搜索的信息范围，因此完整报告将保留每片命令/版本/命中；不宣称穷举隔离。

索引0已做dry-run，并在本机以4线程、15分钟硬上限完成真实规模探针：监督器总耗时274.38秒、exit0。其终态见 `expansion_search0_budget.json` 和 `expansion_plan/searches/task_00000/complete.json`。其他92项未批量启动；不同方向与分片的耗时可能不同。已有完成结果应带着哈希一同迁移；脚本遇到同名输出目录会拒绝覆盖，不能把部分结果当成完成。

集群需要复制同一版本的准备脚本、`localph/phenv_data.py`、完整原始audit目录（包括SQLite与PHOPT FASTA）、已完成的全量正向筛查目录，以及整个expansion_plan目录。计划绑定源代码和输入SHA256。若改算法或输入，应建立新的计划目录，不修改已冻结的plan.json来绕过检查。以下变量均为集群实际路径；不包含调度资源配置或提交动作。

```bash
cd "$PHENV_REPO"
"$PHENV_PYTHON" -B scripts/prepare_phenv_expansion.py search \
  --audit "$PHENV_AUDIT" --plan "$PHENV_PLAN" \
  --task-index "$PHENV_TASK_INDEX" --mmseqs "$PHENV_MMSEQS" \
  --threads "$PHENV_CPU_THREADS" --memory 8G --dry-run
```

在已有CPU分配内移除 `--dry-run` 执行单个任务。`8G` 是MMseqs拆分提示，不是进程峰值内存的硬保证；根据探针和集群资源设置整体作业内存。可按调度器数组任务提供索引，但应先跳过已验证完成的任务，不能启动整套连续本机重试来规避10小时限制。

所有93项搜索以及全量正向筛查完成后：

```bash
"$PHENV_PYTHON" -B scripts/prepare_phenv_expansion.py finalize \
  --audit "$PHENV_AUDIT" --plan "$PHENV_PLAN" \
  --forward "$PHENV_FORWARD_SCREEN" --output "$PHENV_PREPARED"

"$PHENV_PYTHON" -B scripts/verify_phenv_expansion.py \
  --audit "$PHENV_AUDIT" --plan "$PHENV_PLAN" \
  --forward "$PHENV_FORWARD_SCREEN" --data "$PHENV_PREPARED"
```

独立验证器重建每个命中的排除原因、核查全部保留行的原始序列/标签、训练候选完整覆盖、固定验证和来源权重；缺少任一搜索都不能签发verification。全量正向搜索若对固定验证发现新PHOPT命中，流程停止，需显式修订数据协议。之后根据最终保留残基数重算GPU编码与缓存需求，再进入下方GPU阶段。

## 分片工作方式

每个编码任务占用一张已分配GPU，明确指定 `shard-count` 和零起始 `shard-index`。编码器按长度/键排序后交错分片，保留所有残基，移除BOS/EOS，float32推理后存fp16；不截断。不同任务写不同的 `shard_00000` 等目录。每100条刷新缓存并记录已写前缀；复用时要求输入、源码、权重、Torch版本与编码配方一致。

所有分片完成后，`finalize` 校验每个文件哈希、键无重复且完整覆盖输入清单。聚合使用虚拟拼接，不复制一份1.75 TB缓存。预训练当前为单GPU小头训练，支持读取这些分片；它不是多GPU分布式训练器。多个编码GPU也不保证墙钟时间按卡数线性缩短。

把以下命令作为已有资源脚本的任务主体。大写变量均需在集群中填为真实路径；示例不自选节点、分区、时限或GPU数。

```bash
cd "$PHENV_REPO"
"$PHENV_PYTHON" -B scripts/cluster_environment_stage.py encode \
  --data "$PHENV_PREPARED" --cache "$PHENV_CACHE" \
  --checkpoint "$PHENV_ESM1V_CHECKPOINT" \
  --shard-count "$PHENV_SHARDS" --shard-index "$PHENV_SHARD_INDEX" --dry-run
```

检查打印的路径和命令后，在已分配GPU的作业中移除 `--dry-run` 执行同一命令。指定检查点为 `esm1v_t33_650M_UR90S_1.pt`，当前对照权重SHA256为 `9519ee60f1cddad3c101afb1f42612499e188534969c3f682e94850870f70433`。若更换检查点或推理精度，PHOPT表征必须重新生成并做匹配对照。

```bash
"$PHENV_PYTHON" -B scripts/cluster_environment_stage.py finalize \
  --data "$PHENV_PREPARED" --cache "$PHENV_CACHE" --shard-count "$PHENV_SHARDS"

"$PHENV_PYTHON" -B scripts/cluster_environment_stage.py pretrain \
  --data "$PHENV_PREPARED" --cache "$PHENV_CACHE" --shard-count "$PHENV_SHARDS" \
  --output "$PHENV_PRETRAIN_OUTPUT"
```

预训练检查点用作酶任务参数初始化，不是恢复同一个优化器训练过程；酶头重置为零，不继承pHenv输出标签含义。结束需保留协议、学习曲线、模型权重、预测、完成标记与CPU重载核验。断点后的编码前缀恢复不等于辅助模型优化器状态恢复。

用户历史EpHod工作流约定是保留 `sub.sh` 资源参数并使用 `yhbatch sub.sh`；这条历史约定没有被当作当前集群已核实状态。本仓库及相邻镜像目前未找到可复用的 `sub.sh`。新入口不生成或覆盖资源脚本，也不擅自替换提交命令。
