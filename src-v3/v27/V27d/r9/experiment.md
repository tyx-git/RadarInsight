# V27d-r9：因果 Doppler profile shape branch

## 实验身份

- 实验：`V27d-r9`
- 隶属：`V27d`
- 直接基线：冻结的 `V27a-r3 Main` epoch 20；其 V27b-r5 `frozen_l2_logits` 作为基线序列
- 唯一变量：因果 residual Doppler profile shape branch
- seed：`42、44`
- 数据：CQ08，V27a Main 的同一 train/validation known DBB parent population
- Liquid 主干：Liquid V2 保持冻结，不重新训练；不引入 GRU
- L1、knownness、quality、maturity、L3、Unknown 和 test：不在本实验范围

## 实验问题与假设

V27a 已确认完整 raw Doppler profile 尚未进入 L2，且 V27a validation 的主要 Drone/Bird 边界错误包含 4 个 `DroneTarget → BirdTarget`。本实验只验证：将已经物化、经过原始 evidence 合同保护的 181-bin residual Doppler profile 以因果 shape 特征接入一个独立的 Drone/Bird pair residual，是否能在不破坏 Bird 安全线的情况下修正至少一个 Drone/Bird 错误。

这不是把 profile 直接替换进 Liquid V2，也不是把 181 维向量作为第四类或 Unknown 信号；新分支只对三类冻结 logits 做零和的 Drone/Bird 边界修正，Balloon correction 恒为零。

## 输入与因果构造

- 原始输入：V18 `residual_doppler_profiles`，每帧 181 维，轴为 `[-90, 90] m/s`。
- 每个 reliable frame 的 profile 已由 V18 evidence 生成逻辑做峰值归一化；每个 event 对可靠 frame profile 求均值，再做 event-level L1 shape 归一化。
- 对每个有效事件按时间顺序构造四组 181 维量：
  - 当前有效事件 shape；
  - 截止当前事件的 running mean shape；
  - 截止当前事件的 running standard deviation shape；
  - 与上一个有效事件 shape 的逐 bin absolute delta。
- 无可靠 profile 的 event：不更新 running state，residual、residual increment 和 raw residual 均为零。
- padding：不更新任何 profile state，所有输出为零。
- profile event 与 V27b-r5 parent/event order 通过 sample id、parent length 和 event offset 严格校验。

## 模型与训练

- 共享 `181 → 32` profile projection 分别作用于四组 profile，再拼接为 128 维。
- 两层 SiLU MLP 输出一个 scalar pairwise delta，使用 `6*tanh(raw)` 有界化。
- correction layout：`DroneTarget:+delta/2`、`BirdTarget:-delta/2`、`BalloonTarget:0`。
- 最后一层零初始化，训练初始时应逐值复现冻结 V27a Main。
- loss：继续使用 `parent_final_fixed_parent_count_ce` 的语义：每个 parent 取 final state，使用 V27 L2 固定 class weights，除以 batch parent 数量；不按 batch 中出现的类别重新归一化。
- checkpoint：固定 primary epoch 120；validation 只用于每 epoch 诊断，不用于选择最佳 epoch。

## 晋级标准

每个 seed 都必须满足：

1. Bird Recall ≥ 96%；
2. 相对冻结 V27a Main 不新增错误（correct→wrong = 0）；
3. 至少减少一个 `DroneTarget → BirdTarget` 错误，并且属于 Drone/Bird 边界的可观察修正；
4. 总正确数至少增加 1 个。

两个 seed 都满足才可判为通过。否则只保留为 diagnostic，不进入后续正式 L3 或部署判断。

## 已核验的输入证据

- V27a Main checkpoint file SHA-256：`5c416261810bdaf6f8e6e751c51612c98d1f77c8abc23b6060d7ac2898078305`。
- V27b-r5 train direct-frame cache file SHA-256：`551f97521bc46846ad00806f78bbd4f80ee7e72b2fc4fde04a7199d8ee5ce5c9`。
- V27b-r5 validation direct-frame cache file SHA-256：`ceab68e04bca8fcc1b05229466c3491954e2b15cdcb277d7fea109d8d2848800`。
- 原始 train/validation evidence 的 SHA-256 已写入 r9 config 与 manifest。
- V27a Main validation confusion：`[[109,4,0],[1,54,1],[1,1,52]]`，总正确 `215/223`，Bird Recall `54/56=96.4286%`。

## 运行命令

```powershell
Set-Location F:\Project\RadarInsight\src-v3
..\.venv\Scripts\python.exe -m v27.V27d.r9.cli --project-root F:\Project\RadarInsight prepare
..\.venv\Scripts\python.exe -m v27.V27d.r9.cli --project-root F:\Project\RadarInsight materialize
..\.venv\Scripts\python.exe -m v27.V27d.r9.cli --project-root F:\Project\RadarInsight smoke
..\.venv\Scripts\python.exe -m v27.V27d.r9.cli --project-root F:\Project\RadarInsight train --seed 42
..\.venv\Scripts\python.exe -m v27.V27d.r9.cli --project-root F:\Project\RadarInsight train --seed 44
..\.venv\Scripts\python.exe -m v27.V27d.r9.cli --project-root F:\Project\RadarInsight evaluate --seed 42
..\.venv\Scripts\python.exe -m v27.V27d.r9.cli --project-root F:\Project\RadarInsight evaluate --seed 44
..\.venv\Scripts\python.exe -m v27.V27d.r9.cli --project-root F:\Project\RadarInsight summarize
```

## 结果

训练和评估已完成。两个 seed 均使用固定 primary epoch 120；validation 只用于记录诊断，未用于 checkpoint 选择。

### seed 42

- treatment confusion：`[[106,7,0],[3,51,2],[1,4,49]]`
- accuracy：`92.3767%`，macro-F1：`91.6138%`
- Drone P/R/F：`96.3636% / 93.8053% / 95.0673%`
- Bird P/R/F：`82.2581% / 91.0714% / 86.4407%`
- Balloon P/R/F：`96.0784% / 90.7407% / 93.3333%`
- Bird Recall 相对基线：`-5.3571pp`
- 总正确数相对基线：`-9`；correct→wrong：`11`；wrong→correct：`2`
- `Drone→Bird` 相对基线增加 `3`

### seed 44

- treatment confusion：`[[105,8,0],[5,48,3],[1,4,49]]`
- accuracy：`90.58296%`，macro-F1：`89.6538%`
- Drone P/R/F：`94.5946% / 92.9204% / 93.7500%`
- Bird P/R/F：`80.0000% / 85.7143% / 82.7586%`
- Balloon P/R/F：`94.2308% / 90.7407% / 92.4528%`
- Bird Recall 相对基线：`-10.7143pp`
- 总正确数相对基线：`-13`；correct→wrong：`13`；wrong→correct：`0`
- `Drone→Bird` 相对基线增加 `4`

### 失败归因

- r9 train accuracy 两个 seed 均为 `100%`，而 validation accuracy 分别为 `92.3767%` 和 `90.58296%`，属于明显的 train/validation 泛化断裂。
- validation 的 final pairwise delta 在大量 parent 上接近 `±6` 饱和值；seed42/44 的 branch 都把多个原本正确的 Drone、Bird、Balloon 推过了分类边界。
- 失败不是 raw Doppler evidence 缺失：对同一 raw evidence 独立计算的低维 shape statistics 在 train/validation 方向一致；验证集 Drone/Bird 的单变量方向性 AUROC 约为 spread `0.955`、peak drift `0.913`、intra-event cosine `0.873`、top4 concentration `0.837`。
- r9 的主要问题是 181 维当前 shape、running mean、running std、相邻差分共 `724` 维高容量输入，在 953 个 parent 上使用无额外幅度约束的 120 epoch pair residual，学习到了不可迁移的 shape 细节。
- Balloon correction 虽然 logit correction 的第三维恒为零，但 Drone/Bird 相对位移仍会改变 Balloon 的 argmax；validation 中出现 `Balloon→Bird` 新错误。因此“第三类 logit 不变”不等于“第三类预测不受影响”。

### 晋级判断

`V27d-r9` 未通过晋级标准，两个 seed 均失败；不得进入 L3、Unknown 拒识或部署判断。r9 仅保留为“raw profile shape 有可迁移低维信号，但高维 residual branch 过拟合”的诊断证据。下一步应验证低维、低容量的因果 Doppler shape descriptor branch，并继续使用配对错误转移和 Bird Recall 安全线判定。

Unknown/L3 结果不应在本实验中填写为已验证能力；本实验的 `unknown_evaluation_exposure` 为 `0`。
