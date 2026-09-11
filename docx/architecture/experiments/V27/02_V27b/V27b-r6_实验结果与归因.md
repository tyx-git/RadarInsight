# V27b-r6 结果：可靠 stable8 因果稳定性用于冻结 L2 knownness

## 状态

完成。该实验是 `representation_development` 上的三种子表征诊断；没有
拟合 calibration 或 L3，也没有读取 test。分类端点保持为封存的 V27a Main
epoch-20，分类 logits 不作为 knownness 输入。

## V27a 的分类基线与真实瓶颈

封存 V27a Main 在 223 个 DBB parent 上的指标为：

| 类别 | Precision | Recall | F1 |
|---|---:|---:|---:|
| Drone | 98.20% | 96.46% | 97.32% |
| Bird | 91.53% | 96.43% | 93.91% |
| Balloon | 98.11% | 96.30% | 97.20% |

- Accuracy: 96.41%
- Macro-F1: 96.14%
- 混淆矩阵（真实行、预测列）: `[[109, 4, 0], [1, 54, 1], [1, 1, 52]]`

瓶颈是 Bird precision：四个 Drone 被高置信预测为 Bird：`1535`、`1908`、
`2083`、`2103`。它们的 Bird softmax 为 0.981–0.992，且都具有完整的
可靠 stable8 事件覆盖。因此它们不是“缺微多普勒”或“低质量 Pending”的案例，
单纯 softmax、margin、单点距离或质量门都不应被当作安全修复方案。

## R6 对照结果

比较两条同容量、同训练池、同固定 30 epoch 的 independent knownness head：

- `control`: 冻结 Liquid V2 原始轨迹、innovation、gate/alpha/tau、时间与质量事实；
- `stable8_treatment`: 在 control 上加入可靠 stable8 的因果事件摘要、均值、标准差、
  末次偏差及覆盖事实。

| 种子 | Control AUROC | Treatment AUROC | Δ AUROC | Control Unknown rejection @99% known recall | Treatment | Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.936835 | 0.937330 | +0.000495 | 30.6% | 41.7% | +11.1pp |
| 43 | 0.945434 | 0.942403 | -0.003031 | 25.0% | 47.2% | +22.2pp |
| 44 | 0.941475 | 0.943950 | +0.002475 | 30.6% | 36.1% | +5.6pp |
| mean | 0.941248 | 0.941227 | -0.000021 | 28.7% | 41.7% | +13.0pp |

在 97% 和 95% known-recall 点，Unknown rejection 没有稳定净收益：

- @97%: 三种子平均 `-0.9pp`；
- @95%: 三种子平均 `-0.9pp`；
- AUROC 近似持平，且每种子 bootstrap AUROC 区间均跨 0。

每个种子在 99% overall known-recall 点的 Bird coverage 都是 96.5%。这说明
R6 的 99% 结果没有通过把 Bird 整体变成低 knownness 来换取 Unknown rejection；
但它也不能证明对 Bird 类内错误的定向修复。

## 结论与下一步

R6 形成的正向反馈是受限的：可靠 stable8 的因果稳定性对**极保守**（99% known
recall）工作点存在三种子一致的 Unknown rejection 正增益，而整体排序没有提升。
这足以保留该特征作为下一轮的候选输入，但不满足自动进入 calibration/L3 的条件。

特别地，四个 `Drone -> Bird` 错例在 calibration 诊断中的 knownness 都高于各自
99% operating threshold；R6 没有安全拒识它们。分类瓶颈仍是具有完整可靠微多普勒
证据的类内 Drone/Bird 表征重叠，而不是 knownness 缺失。

建议的 V27b 后续方向不是继续堆叠 stable8 summary，也不是启用固定门：

1. 保留 R6 stable8 stability 特征；
2. 为 Drone/Bird 建立一个**独立、冻结端点之外**的类条件冲突/非一致性诊断，输入应是
   全轨迹的 raw-state innovation、stable8 变化、物理 RD/质量事实，不能读取 target
   label 或把分类 logit 直接作为 knownness 训练特征；
3. 将 “类条件冲突” 与 Unknown knownness 分开报告。只有已知 coverage 与 Unknown
   排序均达到独立 development 条件，才在独立 calibration pool 拟合 L3；
4. 任何将四个高置信错误变为拒识的方案，必须同时报告它对正确 Bird 的 coverage，不能将
   错误已知和 Unknown 合并成拒识成功。

## 产物

- 封存 V27a 主报告：
  `artifacts/archive/v27/v27a-history/20260903/reports/V27a-r3-paired-validation.json`
- R6 smoke：`reports/V27b-r6-contract-smoke.json`
- 三种子 development 配对报告：
  `reports/V27b-r6-seed{42,43,44}-representation_development-paired-knownness.json`
- 训练日志：`runs/V27b-r6-{control,stable8_treatment}-seed{42,43,44}/training.log`
