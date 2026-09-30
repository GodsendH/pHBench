# PHOPT 图神经网络架构调研与主架构选择

## 结论

PHOPT 的主架构采用 **Hybrid EGNN + edge-aware sparse graph attention**：

```text
SaProt 残基表示（冻结） + 结构/化学节点特征
        -> EGNN（更新几何与节点状态）
        -> 接触边上的多头图注意力（按目标节点稀疏 softmax）
        -> Jumping-Knowledge 多尺度融合
        -> pH 条件化残基注意力解码器
        -> pHopt 分布和均值
```

在本仓库中对应 `model.graph_encoder: hybrid`，推荐配置为
`configs/phgeofuse_phopt_hybrid.yaml`。原 `egnn` 配置继续保留，用于历史
checkpoint 兼容和严格的同预算消融。

## 任务约束

当前 PHOPT FASTA 的规模是 train/validation/test = **7,124/760/1,971**。
序列长度范围约为 33--1,021，三份数据的中位长度约为 399--410。标签是连续
pHopt 标量，模型还要输出 pH 网格上的分布；因此图编码器的目标是提取局部
结构和残基相互作用，最终预测仍是图级回归，而不是节点分类或生成序列。

当前图构建使用 C-alpha 坐标、序列相邻边和 16 Å 内最多 16 个空间邻居，边
特征包含 RBF 距离、归一化序列方向和是否为相邻残基。这个规模适合稀疏边注意力，
不适合在每一层构造全连接的 O(N²) 注意力。

## 候选架构比较

| 架构 | 适配点 | 主要问题 | PHOPT 决策 |
| --- | --- | --- | --- |
| GCN/GAT/GraphSAGE | 简单、便宜，适合一般拓扑图 | 不处理 3D 刚体变换；普通 GAT 对距离和方向表达弱 | 作为轻量基线 |
| EGNN | E(3) 等变，参数和显存开销低；直接使用坐标和距离 | 均匀聚合容易受节点度影响，深层会过平滑 | 保留为基线，并作为主架构的几何层 |
| GVP-GNN | 标量/向量通道清晰，蛋白结构任务已有 ATOM3D 证据 | 需要向量特征和更多实现约束；与当前缓存接口不直接兼容 | 有价值的第二主线候选 |
| SE(3)-Transformer | 等变注意力，表达力强 | 球谐/irreps 和全注意力成本高；1k 残基图训练不经济 | 不作为当前主线 |
| Equiformer | 将等变特征与图注意力结合，适合原子图 | 实现和显存复杂度明显高于当前任务需要；需重新设计特征类型 | 后续升级/小图实验 |
| GearNet | 面向蛋白结构预训练，适合迁移学习 | 重点是预训练 encoder；本项目已有冻结 SaProt，直接替换会改变实验协议 | 作为预训练对照，不作为主干 |
| GraphGPS/Graph Transformer | 局部消息传递加全局注意力，能做多尺度建模 | 全局注意力在长蛋白上增加 O(N²) 成本，且需要额外位置编码 | 只借鉴局部+全局的设计，不使用全局层 |
| Hybrid EGNN + 稀疏边注意力 | 同时保留几何等变、边条件化路由和多尺度残差；复杂度按边数 | 比 EGNN 多一组注意力参数，需要防止过拟合 | **主架构** |

## 文献依据

- EGNN：Satorras et al., *E(n) Equivariant Graph Neural Networks*，说明无需高阶
  球谐表示也可获得 E(n) 等变性，适合当前 C-alpha 图。论文：
  <https://arxiv.org/abs/2102.09844>。
- GVP-GNN：Jing et al., *Learning from Protein Structure with Geometric Vector
  Perceptrons*，在 ATOM3D 多个结构任务上优于或持平参考架构，说明蛋白结构图
  中显式几何通道有价值。论文：<https://arxiv.org/abs/2106.03843>。
- SE(3)-Transformer：Fuchs et al.，使用 SE(3) 等变注意力，但计算和表示复杂度
  更高。论文：<https://arxiv.org/abs/2006.10503>。
- Equiformer：Liao and Smidt，使用 irreps 和等变图注意力，在 3D 原子图上取得
  强结果，但其特征和算子栈明显重于当前 PHOPT 需求。论文：
  <https://arxiv.org/abs/2206.11990>。
- GearNet：Zhang et al., *Protein Representation Learning by Geometric Structure
  Pretraining*，通过结构预训练提升蛋白功能/折叠表示。论文：
  <https://arxiv.org/abs/2203.06125>。
- GraphGPS：Rampasek et al., *Recipe for a General, Powerful, Scalable Graph
  Transformer*，将局部消息传递、结构编码和全局注意力解耦；本项目采用其
  “局部消息 + 多尺度”思想，但把全局注意力限制为真实接触边。论文：
  <https://arxiv.org/abs/2205.12454>。
- 近期结构对齐工作仍采用预训练 pGNN 作为序列模型的结构适配器，而不是用全局
  Transformer 取代局部几何图。例如 *Structure-Aligned Protein Language Model*
  (2025) 将 pGNN 表征与 pLM 表征做对比对齐，这支持本项目“冻结 SaProt + 轻量
  结构图适配器”的组合。论文：<https://arxiv.org/abs/2505.16896>。

## 为什么选择 Hybrid

1. **几何归纳偏置保留。** EGNN 层继续用相对坐标和距离更新节点/坐标，输入整体
   平移、旋转或反射时预测不应发生非物理变化。
2. **边路由更适合 pH 机制。** 酸碱性由局部离子化残基、溶剂暴露和空间邻居共同
   决定。注意力 bias 使用 RBF 距离、序列方向和当前距离，让模型在同一图中区分
   近邻、长程接触和序列邻接，而不是只按度数求和。
3. **控制过平滑。** 每个 block 采用残差门；输入层和各中间层通过可学习权重做
   Jumping-Knowledge 融合，避免把口袋局部信号压成单一的深层均值。
4. **复杂度可控。** 注意力只在现有边上计算，边数约为 O(16N)，不引入全连接
   O(N²) 的长蛋白开销。
5. **与现有实验协议兼容。** 冻结 SaProt、结构缓存、retrieval gate、pH 网格
   decoder 和现有 checkpoint 都不需要改变；只替换图编码器，并且 `egnn` 仍可
   用作基线。

## 推荐配置

```yaml
model:
  mode: frozen
  graph_encoder: hybrid
  hidden_dim: 256
  egnn_layers: 4
  graph_attention_heads: 4
  local_ph_conditioning: true
  dropout: 0.10
graph:
  spatial_k: 16
  cutoff: 16.0
  rbf_bins: 16
```

先使用 `seed=42` 做同预算比较；若验证集变差，按顺序尝试 3 层、dropout 0.15、
关闭 `local_ph_conditioning`，不要先扩大模型。SaProt 和 retrieval 设置必须与
EGNN 基线完全一致。

## 验证门槛

主指标不应只看总体 RMSE。至少同时报告：

- validation/test 的总体 RMSE、MAE 和 pH 分布校准误差；
- pH <= 5、5 < pH < 9、pH >= 9 三个区间的 RMSE 和 bias；
- 低同源子集（`test_low_identity`）RMSE；
- 结构图不可用或低 pLDDT 样本上的性能；
- 3 个随机种子的均值和标准差；
- `egnn`、`hybrid` 去掉 Jumping-Knowledge、去掉距离 bias、只替换聚合器四个消融。

只有当 hybrid 在验证集总体 RMSE 不恶化，并且低同源或 pH 两端至少一项有稳定
改善（3 个 seed 的方向一致）时，才将它作为默认主架构。当前工作区没有可用的
PyTorch 运行环境，因此本文档记录的是架构选择和验证协议；不能把尚未执行的
hybrid 训练结果表述为已验证收益。
