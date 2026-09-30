# pH-GeoFuse 图预测改进

## 现有瓶颈

当前图分支把冻结的 SaProt 残基表示与结构特征拼接后送入 6 层 EGNN。每条边的消息用 `scatter_sum` 聚合，聚合权重不随节点度归一化；因此高度连接的残基会产生更大的更新量，深层还容易出现过平滑。距离 RBF 只作为 MLP 输入，没有显式决定邻居之间的信息路由。最终只用节点均值池化，浅层的局部口袋信息和深层的全局拓扑信息被压成同一个尺度。

已有多 seed 测试也支持这个判断：图分支的 `global` expert RMSE 约为 0.95--1.01，而融合后约为 0.785--0.790；酸性样本偏差约为 +0.84--+0.97，碱性样本偏差约为 -0.68-- -0.77。也就是说，结构分支对 pH 两端的条件表征明显不足，单独继续调 gate 很难解决。

此外，pH 解码器先对整张图按电荷加权，再让每个 pH 网格查询共享同一份上下文。它能利用 pKa，但没有让局部残基在不同 pH 下重新选择邻居或改变通道权重。图预测部分因此更像“结构编码 + 全局回归”，而不是条件化的残基相互作用建模。

## 已实现的结构

`model.graph_encoder=hybrid` 启用交替的 EGNN/边感知图注意力：

1. EGNN 继续更新坐标和节点状态，保持距离和刚体变换的归纳偏置。
2. 多头注意力只在接触图边上计算，注意力 bias 由 RBF、序列方向、是否相邻和当前距离共同生成，并按目标节点做稀疏 softmax，消除 degree-dependent 的 `scatter_sum` 尺度。
3. 残差门控制注意力更新幅度，避免注意力分支破坏预训练 SaProt 表示。
4. 对输入投影和每个混合块保留 Jumping-Knowledge 状态，使用可学习层权重融合局部和全局尺度，降低 6 层深度带来的过平滑风险。
5. `local_ph_conditioning=true` 时，把每个 pH 网格下的残基电荷作为 token 调制后再做图级注意力；同一个残基在酸性和碱性 pH 下可以贡献不同的局部证据，减少两端回归被拉向中性均值。

基线 `egnn` 仍是默认值，已有 checkpoint 可以原样加载。可直接使用 `configs/phgeofuse_phopt_hybrid.yaml` 训练独立实验。

## 建议的验证顺序

先固定数据划分、SaProt cache、retrieval cache、seed 和训练步数，对 `egnn` 与 `hybrid` 做同预算比较；同时记录整体 RMSE、低同源子集 RMSE、pH 极端区间 RMSE 和校准误差。若 hybrid 的训练误差下降而验证误差变差，优先把 `egnn_layers` 从 4 降到 3、提高 dropout 到 0.15，或冻结 SaProt/图投影前两层；若低同源提升但高同源下降，保留 hybrid 图分支并单独重调 retrieval gate，而不要把两种误差混入一个平均指标。

推荐的消融是 `egnn`、`hybrid` 去掉 Jumping-Knowledge、`hybrid` 去掉距离 bias、以及只替换 scatter 聚合为 attention。这样能区分收益来自归一化、边条件化还是多尺度融合。
