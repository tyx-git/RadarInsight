# RADAR-LN3 V25f：causal-prefix checkpoint utility selection

## 实验版本

V25f。只扫描已经完成的 V25a 20 个 epoch checkpoint，不重新训练。

## 变量

只替换 checkpoint 选择规则。V25a 的模型、loss、prefix auxiliary weight、
causal-course 特征、dt 语义、优化器、初始化、DataLoader 顺序和 20 个保存状态
全部冻结。

训练期间的历史选择目标是 parent-final `parent_min_precision_recall`，而产品动作
实际还要求 causal-prefix 的已知目标/杂波证据。V25f 将两者分开报告：

1. 严格产品安全筛选：Clutter parent 至少 8/9、Precision 至少 0.90，KnownTarget
   parent-final 和 eligible-prefix Recall 均不得低于 V25a legacy checkpoint。
2. 诊断前沿筛选：允许最多损失 1 个 KnownTarget parent，仅用于定位
   parent-final 与 prefix utility 的冲突，绝不作为采纳门槛。

通过严格筛选后的 epoch 以 parent-balanced eligible-prefix Clutter F1 排序，随后以
Clutter Precision/Recall 联合值、KnownTarget Recall 和 parent Macro-F1 作为 tie-break。
不使用固定 2pp 采纳阈值；收益使用 paired parent bootstrap 和逐 parent 变化报告。

## 前缀动作语义

事件 1–2 仍为 `PENDING_ABSTAIN`，不计作 KnownTarget/Clutter 产品动作；从第 3 个
事件开始，eligible-prefix 排除最终事件。当前 prefix loss 仍复用完整 parent-final
KnownGate 标签，因此 V25f 只诊断 checkpoint 目标错配，不修改监督语义。

## 通过条件

严格产品安全筛选存在可行 epoch，且其 parent-balanced causal-prefix Clutter F1、
Recall/Precision 联合结果优于 V25a legacy，同时不牺牲 KnownTarget 安全指标。否则
证明 checkpoint 选择单独不能解决冲突，下一实验应转向前缀监督语义。

V25f 只属于 KnownGate 开发诊断，不执行 Unknown、OOF、test、L3、完整级联、拒识、
stable 或 P95，也不产生 deployment 结论。
