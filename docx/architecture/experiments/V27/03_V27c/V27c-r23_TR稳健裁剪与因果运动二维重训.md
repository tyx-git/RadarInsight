# V27c-r23：TR robust clipping × causal motion 的 2×2 L2 重训

## 问题

此前 r22 只比较了 motion treatment 和 zero-motion control，且两者共用未裁剪的
legacy TR 输入。当前已核实训练 known DBB 中存在未裁剪的极端 `range_m` 和
`speed_mps`。r23 用正交 2×2 设计分离两个因素，避免把 clipping 的效果误归因于
causal motion。

R23 因此检验更强、但仍可归因的问题：motion 数值若在每个事件进入 Liquid V2
更新之前，能否在重训的 L2 表征中改善 Drone/Bird/Balloon 分类？

## 固定边界

- L1 固定为 V27 epoch-14：只绑定其 hash，不加载、不前向、不优化；
- L2 从 V27 epoch-11 分类器和 V27a Main epoch-20 stable8 证据分支初始化；
- 保留 V27a 已验证的可靠 stable8 全帧路径，不能回退为只使用压缩 summary；
- 只读取已知 DBB 的 train/validation；Unknown、Clutter、knownness、quality、
  calibration、L3 与 test 均不参与；
- 训练损失继续是固定 parent 分母的 class-balanced final CE；不使用验证集选择
  checkpoint，固定训练 20 epoch。

- robust clipping 只使用 known train 的 `range_m/speed_mps` 99.9% 上分位；
  val 只做审计，不能参与拟合。当前拟合边界由 prepare 阶段写入 manifest：
  `range_m <= 4843.660789968004 m`、`speed_mps <= 51.11839758400121 m/s`。

## 四个匹配 arm

| Arm | 训练项 | 唯一输入差异 |
|---|---|---|
| `legacy_control` | legacy TR、L2 base、V27a Main stable8、同容量 adapter | adapter 接收全零数值 |
| `legacy_motion` | legacy TR、L2 base、V27a Main stable8、同容量 adapter | adapter 接收 legacy 原始 TR 重算的 9 维因果 motion |
| `robust_control` | robust TR、L2 base、V27a Main stable8、同容量 adapter | adapter 接收全零数值 |
| `robust_motion` | robust TR、L2 base、V27a Main stable8、同容量 adapter | adapter 接收裁剪后原始 TR 重算的 9 维因果 motion |

motion adapter 是无 bias 的 `9 -> 16 -> 16 -> 48` SiLU 残差。其输出层零初始化，
故每个 clipping 水平内的 control/motion 两臂在训练前精确同态；adapter 输出被加到
对应 TR fusion event 后、Liquid V2 递归之前。所有 arm 的 reference L2 输入固定为
未裁剪 legacy TR，确保 clipping 主效应不会在差分 reference 中被抵消。
因此它能影响整条后续 raw-memory 因果轨迹，而不是仅修改 parent-final logits。

`tr_motion_valid_mask` 仅在 dataset sidecar materialization 时用于把无效数值置零；
它不会进入模型、checkpoint 或学习特征。可靠 stable8 的 frame mask 是已有微多普勒
分支的必要帧 padding 合同，与 motion validity mask 不是同一种输入。

## 评价

每个 seed 的报告同时包含：

- V27a Main 冻结参考、四个 arm，以及每个 arm 的 zero-motion 反事实；
- parent-final 与 parent-balanced eligible-prefix 指标、逐 parent paired transition、
  四个既有 `Drone -> Bird` 高置信错误审计；
- 分层 paired bootstrap；
- length、活动 token support、分量 support 三种不读取 label 的 motion 置换；
- clipping 主效应、motion 主效应和 clipping×motion 交互分别判断。任何结果均不
  构成 Unknown 拒识、L3 或部署结论；seed 仅为 42、43，不得写成三 seed 结论。

## 结果

正式训练与评估已完成：4 个 arm × 2 个 seed，共 8 个 L2 训练 run。

### 结果概览

| Seed | Arm | Accuracy | Macro F1 | Drone P/R/F | Bird P/R/F | Balloon P/R/F |
|---:|---|---:|---:|---|---|---|
| 42 | 四个 arm 均相同 | 0.964126 | 0.963050 | 0.973214 / 0.964602 / 0.968889 | 0.915254 / 0.964286 / 0.939130 | 1.000000 / 0.962963 / 0.981132 |
| 43 | 四个 arm 均相同 | 0.968610 | 0.967541 | 0.981982 / 0.964602 / 0.973214 | 0.916667 / 0.982143 / 0.948276 | 1.000000 / 0.962963 / 0.981132 |

两 seed 汇总：Macro F1 为 `0.965296 ± 0.002245`，Bird recall 为
`0.973214 ± 0.008929`。seed 42 的四个 arm 均为 215/223 个 parent 正确，
seed 43 均为 216/223 个 parent 正确；四个 arm 都满足“不低于 V27a Main 参考的
各类别正确数”安全条件。Bird 仍是 Macro F1 最低的类别，但 seed 43 相比冻结
V27a Main 参考多纠正了 1 个 Bird；该变化同时出现在四个 arm，不能归因于
clipping 或 causal motion。

### 因素效应

- `legacy_control`、`legacy_motion`、`robust_control`、`robust_motion` 的
  parent-final 离散预测在同一个 seed 内完全一致。
- motion 主效应、clipping 主效应以及 clipping × motion 交互的 Accuracy、
  Macro F1、Balanced Accuracy 等离散指标均为 `0`。
- 这不等价于 motion 没有进入模型或没有影响连续输出。逐 parent 终态核验显示：
  - seed 42，motion/control 的最大绝对 logit 差约 `0.0923`，最大绝对概率差约
    `0.01885`；
  - seed 43，最大绝对 logit 差约 `0.1134`，最大绝对概率差约 `0.02191`；
  - 两个 seed 中 motion 都影响了所有 223 个 parent 的连续输出，但没有任何
    parent 跨过 argmax 边界。
- clipping 也造成了连续变化，但更小：两个 seed 的最大绝对 logit 差约为
  `0.1264` 和 `0.1352`，最大绝对概率差约为 `0.000151` 和 `0.000217`，仍未
  改变任何 parent 的 argmax。验证集实际受 speed clipping 影响 3 个 event，
  range clipping 影响 0 个 event；训练集两项各影响 19 个 event。
- zero-motion 反事实与各 control arm 的离散结果一致；motion arm 在置零后连续
  输出发生变化，但离散结果不变。因此 r23 证明了 motion 分支有连续层面的作用，
  但没有证明它能单独改善当前 known-only L2 的离散分类。

### 实验结论

1. r23 未获得 clipping 或 causal TR motion 对 known-only L2 离散分类的可观测
   增益；两 seed 的 factorial 交互为零，当前不应将任一因素晋级为“分类收益
   已验证”的默认配置。
2. r23 没有否定 motion 的潜在价值。motion 已改变连续 logits，后续若要继续，
   更适合检验它在独立 knownness、质量/成熟度或时序一致性任务中的排序和校准
   价值，或设计真正的 task-specific motion 表征；不能只用当前 parent argmax
   作为“特征完全无效”的证据。
3. clipping 在本数据划分中的实际覆盖率很低，且未带来离散改善。它可以作为
   数值安全选项继续保留，但不能声称已解决 TR 极端值污染问题；若要验证该风险，
   需要把连续稳定性、质量头或 knownness 排序作为预先登记的终点。
4. 本实验严格是 known-only L2：Unknown、Clutter、knownness、quality、
   calibration、L3 和 test 均未暴露，不能从本实验推导拒识效果或产品动作结论。

### 复现与审计说明

- 8 个正式 checkpoint 均在固定 epoch 20 完成，训练 summary 状态为
  `completed_fixed_epoch_matched_l2_retraining`；训练与验证使用 train-only
  `99.9%` clipping 拟合边界，manifest SHA256 为
  `6d4be963291f46be3e079278461e37cc7be2eb0374c957e22411f32571358374`。
- 初次直接执行 evaluation CLI 时发现 `evaluation.py` 缺少
  `build_initialized_model` 导入。为不改变已完成实验的 manifest/code hash，正式
  报告使用运行时 shim 将该符号绑定到 `training.build_initialized_model`；shim
  运行成功并生成封存报告。该源码问题应在新代码版本/新 manifest 中修复后再用于
  后续实验，不能直接修改当前 r23 的封存源文件。
- 运行目录中保留了早期失败与重试日志（包括 CUBLAS 环境变量启动问题）；最终
  training-summary、checkpoint、evaluation report 均已完成并通过对应 hash/契约
  校验。阅读日志时应以最终成功段和 summary 为准，同时保留早期失败记录作为审计
  历史。

详细指标见 `reports/V27c-r23-seed42-factorial-validation.json`、
`reports/V27c-r23-seed43-factorial-validation.json` 和
`reports/V27c-r23-two-seed-summary.json`；训练日志位于
`runs/V27c-r23-{arm}-seed{seed}/training.log`。
