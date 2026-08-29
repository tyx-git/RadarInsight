# V26h：L1 train-only Mahalanobis OOD 可分性诊断

## 目的

V26f 的冻结 mixed audit 已确认，当前主要安全缺口是 L1 将绝大多数严格
open-set `UnknownTarget` 高置信接受为已知目标或已知背景；V26g 的质量可用性
门控只能把证据缺失的样本转为 `PENDING_ABSTAIN`，不能拒识这些高置信 Unknown。

V26a 已经做过基于 `1 - max softmax probability`（MSP unknownness）的阈值扫描，
冻结结论为 `no_useful_threshold`：它不能在保护 Known 路由的同时拒识 Unknown。
因此 V26h 不重试 MSP 阈值，也不直接训练新的拒识头。它先验证一个更基本且可
证伪的问题：**当前 L1 最终决策表征中，Unknown 是否比 Known 更偏离该折训练
Known 类分布？**

## 唯一变量

在每个冻结 V26e L1 epoch-32 checkpoint 上，新增一个只读的
class-conditional pooled diagonal Mahalanobis novelty score：

\[
d(x) = \min_{c \in \{\mathrm{KnownTarget},\mathrm{KnownClutter}\}}
\sqrt{\sum_j \frac{(z_j(x)-\mu_{c,j})^2}{\sigma^2_{j,\mathrm{pooled}}}}
\]

其中 \(z(x)\) 是 `classifier[0](liquid_final_state)`，即 L1 最终
`Linear(48, 2)` 实际消费的 LayerNorm 后状态。每个 fold 的 \(\mu_c\) 与
\(\sigma^2\) 只由该 fold 的 train Known parent-final 状态拟合。对方差使用
train-only 的正中位数乘 \(10^{-6}\) 与 \(10^{-10}\) 的较大者作数值下限。

该变量不修改模型参数、输入特征、液态更新、loss、batch、采样、checkpoint、
L2、L3、动作或阈值。高分仅表示“更不像该折 train Known”，不是本轮的
`UNKNOWN_REJECTED` 动作。

## 严格隔离

- V26e 的五个 L1 固定 epoch-32 checkpoint 分别重载；禁止跨 fold 合并状态、
  中心、方差、分数或阈值。
- 每折 reference 只用该折 V26e train parent：KnownTarget（三个目标类合并）与
  KnownClutter 两个已知 L1 类。
- `UnknownTarget` 不参与模型训练、归一化、reference fit、checkpoint 选择、
  阈值拟合或任何 score fusion；只在 checkpoint 和分数冻结后进入 mixed val/test
  的报告标签。
- 只报告 parent-final 的 threshold-free AUROC / Average Precision 和 parent
  bootstrap；不生成阈值、不输出最终动作、不修改 L3 registry。
- V26a 仅作为 MSP 的历史失败对照。其 mixed-validation oracle threshold 绝不
  被 V26h 使用。

## 为什么不等同于 V12/V13

V12/V13 的 knownness 诊断作用于旧 DBB specialist 的 Drone/Bird/Balloon
subclass 状态，研究的是 DBB 子类空间中的 Unknown-vs-Bird 排序。V26h 作用于
当前 V26e L1 的 KnownTarget/Clutter 二分类 readout state，且当前错误主要在 L1
将 Unknown 接受为 KnownTarget。因此它是新的架构层与新的故障位置；但它只验证
可分性，若无排序信号，后续不应继续在相同 L1 表征上搜索距离函数。

## 方法依据与限制

- Lee, K. et al. (2018), *A Simple Unified Framework for Detecting
  Out-of-Distribution Samples and Adversarial Attacks*,
  <https://arxiv.org/abs/1807.03888>：在冻结深度表征上利用 class-conditional
  Mahalanobis 距离；V26h 采用其不需要 Unknown 训练样本的核心思想，但使用
  diagonal pooled covariance 以适应每折仅约 26--27 个 Clutter train parent 的
  小样本条件。
- Hendrycks, D. & Gimpel, K. (2016), *A Baseline for Detecting
  Misclassified and Out-of-Distribution Examples in Neural Networks*,
  <https://arxiv.org/abs/1610.02136>：MSP 是必要对照；V26a 已证实该数据/模型上
  MSP 无可用阈值。
- Liu, W. et al. (2020), *Energy-based Out-of-distribution Detection*,
  <https://arxiv.org/abs/2010.03759>：energy 依赖 logits 标度；当前仅有 binary
  L1 logits、且没有能量训练目标，故本轮不把 energy 与 Mahalanobis 叠加，以免
  同时改变多个 score 变量。

V26h 不是部署实验：Unknown support 仅 val=15、test=16，且 Clutter train
support 很小。任何 AUROC/AP 信号都只能证明该冻结 CQ08 审计中的排序可能性；
只有 V26h 通过后，才值得另立实验用 Known-only OOF 选择保守阈值并审计
Known false reject、Unknown false accept 与 Pending 的全链路动作影响。

## 通过/停止规则（预冻结）

V26h 本身没有“部署通过”结论。

- 值得继续：两套 mixed split 的五折平均 Mahalanobis AUROC 均高于重新计算的
  MSP comparator，且 test 平均 AUROC 至少 `0.65`、五折最小值至少 `0.55`；同时
  test AUROC 的 parent bootstrap 下界不低于 `0.50`。这些是“有证据但未充分
  验证”的排序信号，不是阈值或产品收益。
- 停止：test 平均 AUROC 不高于 MSP，或任何 split 的五折结果出现明显反向且
  test bootstrap 下界不高于随机水平。此时当前 L1 readout state 没有足够的
  Unknown novelty 信息，不继续堆叠 cosine/energy/distance 变体。
- 不确定：总 AUROC 有小幅改善但最小 fold 或 bootstrap 约束失败；报告为证据
  不足，优先检查 Unknown 语义/来源覆盖或收集数据，而不是拟合阈值。
