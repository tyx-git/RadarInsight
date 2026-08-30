# V26k：L2 train-only strict-open-set knownness 诊断

## 目的

V26k 是只读诊断，不训练模型，也不把 L2 gate 接入 L3。它回答一个
可证伪的问题：在当前冻结的 L2 DBB specialist 表征中，严格 open-set 的
`UnknownTarget` 是否能够被一个不看 Unknown 的 knownness 分数排序出来，且
不会先牺牲已知目标召回。

## 直接来源与冻结条件

- L2 模型、fold 划分、训练 parent 和 epoch-15 checkpoint：V26c。
- mixed validation/test parent、L1 观察路由和 L2 输入 cache：V26f。
- 数据源：CQ08，V16 index；不引入其他 source、index 或类别定义。
- L2 训练类别：`DroneTarget`、`BirdTarget`、`BalloonTarget`。
- 每个 fold 单独拟合参考分布；不合并 fold 的状态、距离或阈值。
- Unknown 不参与模型、归一化、参考中心、方差、阈值或任何选择。
- Unknown 只在 score 与 train-only 诊断操作点冻结后进入审计统计。

## 唯一诊断变量

从冻结 L2 的每个 parent 最终状态取得分类器真实输入：

```text
liquid_final_state[48]
  -> classifier[0] LayerNorm(48)
  -> classifier[1] Linear(48, 3)
```

报告代码用 `LayerNorm(liquid_final_state)` 作为 48 维 embedding，并重建
`Linear` logits 与模型输出进行断言；因此不会误把 raw liquid state 或其他
中间层当作分类器输入。

对每个 fold 的 train-only DBB parent，按类别计算中心，并使用 pooled diagonal
within-class variance：

\[
d(x)=\min_{c\in\{D,B,Ba\}}
\sqrt{\sum_j\frac{(z_j(x)-\mu_{c,j})^2}{\sigma^2_{j,\mathrm{pooled}}}}
\]

距离越大表示越不像该 fold 的 train-known DBB。方差只使用 train-known
残差估计；数值 floor 为正方差中位数乘 `1e-6` 与绝对值 `1e-10` 的较大者。

训练集距离的经验 order statistic 仅作为诊断操作点，目标为保留至少 95% 的
train-known parent。这个阈值不是产品 calibration，也不使用 validation/test
或 Unknown 选择。

## 评估层级

主分析集是冻结 V26f 路由为 `KNOWN_TARGET_EVIDENCE` 的 DBB 与 Unknown parent。
报告同时提供：

- parent-final threshold-free Mahalanobis AUROC / Average Precision；
- MSP（`1 - max softmax`）只读 comparator；
- L2 条件 DBB 保留率；
- 现有冻结 L1 与 L2 诊断 gate 串联后的 KnownTarget Recall；
- Unknown rejection / false-accept，尤其是已被 L1 接受为 target 的 Unknown；
- 每 fold 的分数、阈值和 bootstrap 区间；
- capture 分层、parent length 分层和每标签分布；
- Clutter 作为 secondary audit，绝不进入 DBB reference fit。

`KnownTarget Recall` 分为两项：L1 在 L2 前的冻结召回，以及应用 L2 诊断阈值
后的组合召回。若前者低于 95%，V26k 不能声称距离诊断修复了 L1 问题。

## 预注册解释边界

V26k 只产生排序证据，不产生产品动作。只有同时满足以下条件，才值得另立
后续实验，把独立 L2 strict-open-set gate 接入 L3：

1. KnownTarget Recall 达到预先声明的 `>=95%` 要求；必须明确是 L2 条件保留率
   还是 L1→L2 组合召回，不能用二者混淆。
2. 主 L1-target-route Unknown 排序在 validation 和 test 都有有效方向，且不
   依赖单一 fold 或单一 capture。
3. fold 间没有明显反向结果，parent bootstrap 下界和支持数被如实报告。
4. 该信号不能仅由扩大拒识、Pending/Abstain 或改变 L3 动作得到。

如果任一条件失败，结论止于“当前冻结 L2 embedding 的 train-only knownness
诊断不足以接入 L3”，不得继续拟合更多阈值或把 Unknown 当作训练类。

## 方法依据与限制

- Lee et al., 2018, *A Simple Unified Framework for Detecting Out-of-Distribution
  Samples and Adversarial Attacks*, https://arxiv.org/abs/1807.03888：冻结深度
  表征上的 class-conditional Mahalanobis 思路。V26k 使用其核心思想，但因
  每折样本条件采用 pooled diagonal covariance。
- Hendrycks & Gimpel, 2016, *A Baseline for Detecting Misclassified and
  Out-of-Distribution Examples in Neural Networks*, https://arxiv.org/abs/1610.02136：
  MSP comparator；它不作为 V26k 的训练或阈值来源。
- V26k 的 Unknown support 很小（由 V26f mixed contract 固定），且 CQ08
  Clutter parent support 也有限。因此一次五折诊断只能提供“有证据但未充分
  验证”的排序信号，不能替代跨 source、更多 parent 或产品级验证。

## 执行范围

流水线的 `prepare` 只生成 hash-bound contract/readiness，`smoke` 只做合同和
数学边界检查，`diagnose` 才按 fold 在短生命周期 CPU worker 中重载 checkpoint、
提取 train embedding、拟合 train-only reference，并审计冻结 mixed val/test。
训练、OOF 重训、L1 forward、L3 registry、产品阈值、语义动作、full cascade
和 deployment 均明确为未执行。
