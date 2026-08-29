# V26i：L1 train-only 固定双模态 Clutter Mahalanobis 几何诊断

## 目的

V26h 已确认：在冻结 V26e L1 epoch-32 readout state 上，Unknown 相对
Known 的 Mahalanobis 排序总体优于 MSP；但合法 Clutter 常常比 Unknown 更偏离
单一 KnownClutter 中心。尤其在 L1 `KNOWN_TARGET_EVIDENCE` 区域，测试
Unknown-vs-Clutter AUROC 方向反转，故单一全局距离阈值没有安全依据。

V26i 只检验一个可证伪解释：**合法 Clutter 的高 novelty 是否主要由单中心
覆盖不足造成？**

## 唯一变量

保留 V26h 的每折 train-only `KNOWN_TARGET_EVIDENCE` 单一中心；仅把
`KNOWN_CLUTTER_EVIDENCE` 从单中心改为固定两中心：

\[
d_{C}^{(2)}(z)=
\min_{k\in\{1,2\}}
\sqrt{\sum_j\frac{(z_j-\mu_{C,k,j})^2}{\sigma^2_{j,\mathrm{pooled}}}}.
\]

最终 novelty 保持为 KnownTarget 单中心距离与上述 Clutter 双中心距离的最小值。
KnownTarget 与两个 Clutter 成分共用一个 pooled diagonal within-reference
variance；这避免在每折约 26--27 个 Clutter train parent 的小样本下额外估计
不稳定的 component covariance。

固定规则：

- `K=2` 在评分前固定，不搜索 K；
- 初始化为 train Clutter state 中全局最远 parent 对，parent id 用作稳定 tie-break；
- assignment 使用未加权 readout-state squared Euclidean distance；
- 每个成分至少 5 个 train Clutter parent；否则该 fold 标记为 `infeasible`，不改
  K、不合并成分、不从验证/测试/Unknown 调参；
- 距离评分仍使用 pooled diagonal Mahalanobis，且高分表示更不像 train Known。

## 严格隔离

- 五个 V26e L1 fixed epoch-32 checkpoint、其 train cache、V26f mixed cache 与
  parent 顺序逐 fold hash 绑定；
- representation 固定为 `classifier[0](liquid_final_state)`，即实际 L1 Linear
  classifier 消费的 LayerNorm 后 48 维 state；
- Unknown 不参与模型训练、归一化、Clutter clustering、reference fit、组件数选择、
  checkpoint 选择、阈值拟合或 score fusion；只在所有分数冻结后作为 mixed val/test
  排序报告的 truth label；
- 不产生阈值、不重放 semantic action、不改 L1/L2/L3、不输出部署结论；
- 每折在短生命周期 CPU process 中运行，不跨 fold pool state、centre、variance 或 score。

## KnownTarget Recall 门控

任何后续把该分数转成 `UNKNOWN_REJECTED` 动作的实验，必须预注册并满足：

\[
\mathrm{KnownTargetRecall}_{\mathrm{parent-final,mixed}}\ge0.95.
\]

V26i 是无阈值的排序诊断，故不改变 action，**不能也不得宣称已经满足该动作门控**。
它只记录此门控，供后续独立阈值/动作实验使用。

## 通过、停止与后续边界

V26i 不是部署实验。验证集是主诊断集，测试集只是冻结复核；两者均已在 V26h/V26f
中被诊断访问，不能再被当作全新外部 test。

值得继续到独立 threshold/action 设计的必要证据：

1. 每折 Clutter 的 V26i novelty 中位数低于 V26h 单中心分数；
2. validation Unknown-vs-Clutter AUROC 相对 V26h 提高，且五折多数同向；
3. validation Unknown-vs-Drone 与 Unknown-vs-Balloon AUROC 不出现实质退化；
4. Target-like route / closest-KnownTarget 区域的 Unknown-vs-Clutter 排序至少不再
   明显反向；
5. 明确报告每折 Clutter 与 Unknown parent support、bootstrap interval、以及是否有
   component support infeasibility。

停止：任一 fold 因 component support 不足而不可估，或 Clutter novelty 没有下降且
Unknown-vs-Clutter 排序无改善/继续反向。此时不应继续堆叠距离函数或直接设阈值；应转向
Clutter 机制覆盖、Unknown 语义审计，或另立有明确训练边界的 knownness 目标。

## 方法依据

- Lee, K. et al. (2018), *A Simple Unified Framework for Detecting
  Out-of-Distribution Samples and Adversarial Attacks*,
  <https://arxiv.org/abs/1807.03888>：类条件 Mahalanobis 表征距离的基础；
  V26i 不复制其输入扰动或阈值流程。
- Hendrycks, D. & Gimpel, K. (2016), *A Baseline for Detecting
  Misclassified and Out-of-Distribution Examples in Neural Networks*,
  <https://arxiv.org/abs/1610.02136>：MSP 对照已由 V26a/V26h 覆盖；V26i 不重新
  搜索 MSP 阈值。
- Liu, W. et al. (2020), *Energy-based Out-of-distribution Detection*,
  <https://arxiv.org/abs/2010.03759>：energy 为后续可独立比较的候选，但本轮不与
  Clutter 支持几何同时叠加。
