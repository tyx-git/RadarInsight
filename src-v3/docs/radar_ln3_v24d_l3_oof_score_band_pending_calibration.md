# V24d L3 OOF score-band Pending calibration

实验版本：V24d。

变量：V24d 不训练或修改 KnownGate L1。它仅消费已完成的 V21e
capture/content provenance-grouped 五折 OOF treatment 预测中的
`known_clutter_probability`，并用一个 L3 语义决策规则替代二元局部
argmax 的产品解释：

```text
p_clutter <= t_target  -> KNOWN_TARGET_EVIDENCE
p_clutter >= t_clutter -> KNOWN_CLUTTER_EVIDENCE
otherwise              -> PENDING_ABSTAIN
```

严格 score-band 候选满足 `0 <= t_target < t_clutter <= 1`；唯一允许的
零宽候选是 `(t_target, t_clutter) = (0.5, 0.5)`，它严格复现冻结 L1 的
binary raw argmax（0.5 平局归 KnownTarget），仅作为基线而不是可任意移动的
无 Pending 阈值。首两个事件仍由既有
`minimum_evidence_events=3` 规则直接输出 `PENDING_ABSTAIN`；它们不被
阈值带重新解释。Unknown 不在本实验的训练、分数或动作空间中，Pending
也绝不计作 Unknown 拒识。

为防止阈值泄漏，外折 `f` 的阈值只能由其他四个外折的 **parent-final
OOF** 记录拟合。拟合时枚举所有会改变硬动作的实际 OOF 分数断点（再加
0 与 1），而非用近似步长网格，并使用以下 parent-final
目标：Clutter Precision >= 0.90、Clutter acceptance Recall >= 0.88、
KnownTarget acceptance Recall >= 0.98、总 action coverage >= 0.95。若
不存在满足全部目标的候选，报告最小违反量的候选，但该候选不得被表述为
通过或可采纳。拟合出的阈值随后原样应用于该折的 parent、eligible-prefix
和全 window 记录；绝不使用 prefix/window 标签调阈值。

每个候选还必须相对同一拟合集的 raw argmax 同时满足
`KnownTarget -> Clutter` 与 `Clutter -> KnownTarget` 错误接受率均不增加。
该约束同时参与可行性判断和最小违反量回退排序，避免用一种方向的“修正”
掩盖另一种方向的错误转移。

生成的 protocol、contract、readiness 与 report 同时绑定本 V24d build、report
和 smoke 实现文件的 SHA-256；任一实现变化都必须重新生成可审计产物，不能将
旧结果误标为新逻辑的结果。

验证：V24d 测试的是冻结 L1 分数是否有足够稳定的跨来源置信区间，能让
证据不足的窗口变为真实的 `PENDING_ABSTAIN`，而不是把错误改写为
KnownTarget 或 Clutter。报告必须包含：

- 五折 cross-fitted parent-final 的 raw-argmax 与 score-band 指标；
- parent-balanced eligible-prefix 的 coverage、selective risk 和两类错误
  接受方向；
- Clutter parent Precision/Recall 的 Wilson 区间（OOF n=33）与 5000 次
  capture/content component bootstrap；
- 每折阈值、每折拟合可行性、capture/component 分层和所有 parent-final
  动作变化；
- 一个仅作诊断、不可采纳的 pooled-OOF oracle：它用全部 OOF parent 标签
  检查该分数是否连乐观的全量阈值带也无法满足目标，从而与 cross-fitted
  不稳定性区分；
- raw/test/validation/Unknown 访问均为零，以及不作 deployment、完整级联、
  stable 或 open-set 放行声明。

边界：这是 L3 规则的 OOF 诊断，不是新 L1 表征实验，也不是五类产品性能。
只有当 cross-fitted parent-final 目标和 prefix coverage/safety 同时通过后，
才可讨论将该规则实现为独立的 L3 组件；仍须在冻结后执行后续开发/测试协议。
