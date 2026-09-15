# V27d-r10：低维因果 Doppler profile-shape descriptors

## 实验身份

- 实验：`V27d-r10`
- 隶属：`V27d`
- 直接基线：冻结 `V27a-r3 Main` epoch 20；使用 V27b-r5 中冻结的 `frozen_l2_logits`
- 唯一变量：将 r9 的高维 181-bin raw profile trajectory 替换为低维因果 profile-shape descriptors
- seed：`42、44`
- 数据：CQ08，同一 V27a Main train/validation known DBB parent population
- Liquid V2：冻结，不重新训练；不引入 GRU
- L1、knownness、quality、maturity、L3、Unknown、test：不在本实验范围

## 为什么进行 r10

r9 两个 seed 的 train accuracy 均为 100%，但 validation accuracy 只有 92.3767%/90.58296%，且 pairwise residual 大量饱和到 ±6；这证明 r9 的高维 raw profile branch 过拟合，不能直接证明 Doppler shape 没有可迁移信息。独立诊断显示 spread、peak drift、intra-event cosine 和 top4 concentration 在 train/validation 对 Drone/Bird 具有一致方向，因此 r10 只改变 shape representation 的维度与统计表达，保持基线、损失、训练预算和 residual layout 固定。

## 输入特征

每个可靠事件由当前事件可靠帧的 residual Doppler profiles 构造 8 维 descriptor：

1. profile centroid / 90 m/s；
2. profile spread / 90 m/s；
3. profile entropy / log(181)；
4. signed positive-negative energy balance；
5. top-4 profile mass；
6. scaled profile RMS；
7. current-event frame-profile cosine mean；
8. current-event frame peak drift / 180 bins。

每帧 profile 先做 L1 shape normalization，再对当前事件可靠帧求均值并计算上述统计量。所有 descriptor 均不读取标签、L2 logits 或 softmax。

## 因果分支

对每个事件按因果顺序维护 descriptor running count、running mean、running second moment 和 previous-valid descriptor，构造：

- current event descriptor；
- causal running mean；
- causal running standard deviation；
- 与 previous-valid event descriptor 的逐维绝对差。

四组 descriptor 经共享 `8 → 16` 投影后进入两层 SiLU MLP，输出一个有界 Drone/Bird pairwise delta：

`DroneTarget:+delta/2`、`BirdTarget:-delta/2`、`BalloonTarget:0`。

最后一层零初始化；无效 descriptor 不更新状态且输出零 residual；padding 不更新状态。

## 训练与评估

- loss：`parent_final_fixed_parent_count_ce`；继续使用 V27 L2 固定 class weights 和固定 parent-count denominator。
- primary checkpoint：固定 epoch 120；validation 只记录诊断，不参与 checkpoint 选择。
- 晋级标准：Bird Recall ≥96%，correct→wrong=0，至少纠正一个 Drone→Bird 错误，总正确数至少增加 1；两个 seed 均满足才通过。
- 预期：若 r9 失败主要来自高维过拟合，r10 应显著减少 residual 饱和和 validation 退化；该结果仍不能外推为 open-set/Unknown 能力。

## 运行与产物

```powershell
Set-Location F:\Project\RadarInsight\src-v3
..\.venv\Scripts\python.exe -m v27.V27d.r10.cli --project-root F:\Project\RadarInsight prepare
..\.venv\Scripts\python.exe -m v27.V27d.r10.cli --project-root F:\Project\RadarInsight materialize
..\.venv\Scripts\python.exe -m v27.V27d.r10.cli --project-root F:\Project\RadarInsight smoke
..\.venv\Scripts\python.exe -m v27.V27d.r10.cli --project-root F:\Project\RadarInsight train --seed 42
..\.venv\Scripts\python.exe -m v27.V27d.r10.cli --project-root F:\Project\RadarInsight train --seed 44
..\.venv\Scripts\python.exe -m v27.V27d.r10.cli --project-root F:\Project\RadarInsight evaluate --seed 42
..\.venv\Scripts\python.exe -m v27.V27d.r10.cli --project-root F:\Project\RadarInsight evaluate --seed 44
..\.venv\Scripts\python.exe -m v27.V27d.r10.cli --project-root F:\Project\RadarInsight summarize
```

## 实际结果

训练和评估已完成，两个 seed 都使用固定 epoch 120。

### seed 42

- treatment confusion：`[[109,4,0],[1,54,1],[1,1,52]]`
- accuracy：`96.4126%`，macro-F1：`96.1436%`
- Drone P/R/F：`98.1982% / 96.4602% / 97.3214%`
- Bird P/R/F：`91.5254% / 96.4286% / 93.9130%`
- Balloon P/R/F：`98.1132% / 96.2963% / 97.1963%`
- 相对冻结基线：总正确数 `0`，correct→wrong `0`，Drone→Bird 修正 `0`。

### seed 44

- treatment confusion：`[[108,5,0],[1,54,1],[1,1,52]]`
- accuracy：`95.9641%`，macro-F1：`95.7202%`
- Drone P/R/F：`98.1818% / 95.5752% / 96.8610%`
- Bird P/R/F：`90.0000% / 96.4286% / 93.1034%`
- Balloon P/R/F：`98.1132% / 96.2963% / 97.1963%`
- 相对冻结基线：总正确数 `-1`，correct→wrong `1`，Drone→Bird 错误 `+1`，修正 `0`。

### 结论

`V27d-r10` 未通过晋级标准。相比 r9，低维 descriptors 消除了高维 raw profile branch 的严重泛化退化，但没有产生有效的 Drone/Bird 纠错；seed42 完全保持基线，seed44 引入一个新错误。`correction_absolute_max` 仍约为 `2.92`，说明分支虽然不再大面积饱和，但训练后仍能产生非零位移，位移方向没有转化为目标纠错。

结合 r9/r10 的配对结果，当前结论是：raw Doppler profile 确实包含稳定的 Drone/Bird 区分统计，但 V27a Main 的四个 `Drone→Bird` 错例中至少两个在 profile descriptor 轨迹上本身已经 Bird-like；仅靠无 Unknown 监督的 L2 residual 无法可靠纠正它们。继续增加 L2 profile 分支的容量或重复调参缺乏收益证据。

下一步应优先回到 Bird-route knownness：利用独立的质量/轨迹一致性表征，把“真正 Bird”和“Bird-like Unknown”分开；L2 仍保持三类闭集分类，r10 不进入 L3。Unknown/L3 结果不应在本实验内宣称已验证。
