# V21g 模态一致性取消训练决策

## 结论

V21g 不进入训练。该决定不是因为特征物理上完全无效，而是因为在当前 V21e KnownGate 的实际错误边界上，没有足够的增量证据支持再做一次完整嵌套 OOF。

审计报告：

```text
src-v3/processed/report/RADAR-LN3-V21g-modal-consistency-cancellation-audit.json
```

## 审计范围

只读取：

- V21e 五折 train parent-grouped OOF Treatment parent-final prediction；
- V21 KnownGate train-only canonical feature cache；
- 已完成的 V21b 正式 paired report。

不读取 validation/test payload，不训练 Unknown，不拟合阈值。

V21e 的当前 OOF 错误支持为：986 个 parent 中 21 个错误，其中 6 个 Clutter→KnownTarget、15 个 KnownTarget→Clutter、14 个困难 Bird；Balloon 错误数为 0。

## 关键证据

- modal median：Clutter/Drone 方向无关 parent AUC=`0.568`；V21e 困难 Bird/正确 Bird=`0.567`。
- modal max：Clutter/Drone 方向无关 parent AUC=`0.509`；V21e 困难 Bird/正确 Bird=`0.566`。
- V21b 既有正式 paired 训练的 parent 硬预测变化数为 `0`，`positive_signal=false`。
- Clutter/Balloon 的单变量分离较高，但 V21e 当前没有 Balloon 错误，无法证明它能修复产品边界；该方向若未来重新研究必须使用新的独立错误覆盖。
- 当前 Clutter 错误只有 6 个、困难 Bird 只有 14 个，条件错误组太小，不能据此再拟合一个模型并宣称收益。

因此 V21g 的 `training_required=false`，后续编号直接保留给 V21h iso；不生成 V21g checkpoint，也不提供 V21g 训练命令。

## 边界

这是“当前 V21e 错误边界下取消训练”的工程决策，不是对模态一致性在所有数据域中无效的普遍结论。未来若新增独立采集、Balloon/Clutter边界错误重新出现，仍可用新的版本重新评估。
