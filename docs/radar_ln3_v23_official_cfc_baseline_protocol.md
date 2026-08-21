# RADAR-LN3 V23 纯官方 CfC 基座协议

## 目标与阶段边界

V23 是一个单模型的 KnownGate 开发基座，用于验证完整官方 Neural Circuit Policies `ncps.torch.CfC` 在当前 Radar-LN3 输入前端上的可训练性和开发集表现。它不是 V23a/V23b，不带 course 特征，也不使用 V22 fixed-parent loss。

当前阶段只执行一份 seed43、完整 20 epoch 的 formal training，以及开发 validation 评估。五折 OOF、多 seed、test、Unknown 开集校准、L3 语义决策、完整级联、stable、P95 和部署均延后，不能由 V23 基座结果替代。

## 冻结来源与单因素范围

V23 以冻结的 V21 seed43 formal baseline 为来源：

- train / validation parent 分别为 986 / 232；validation Clutter 为 9 parent；
- 复用 V16 index、V21 known-only train normalization、TR/RD 前端、fusion、classifier、二元 KnownGate 标签、batch size=2、AdamW、学习率、20 epoch、gradient clipping、parent-final 加 prefix auxiliary loss 和 development checkpoint selection；
- Unknown 不进入模型类别、训练、normalization、checkpoint selection 或本阶段评估；
- 唯一变化是用官方 dense CfC 替换项目自定义 `CfCLiquidCausalEncoder`。

V23 不声称与历史 V21e/V22 的 OOF artifact 成对；目前 V21e Control OOF prediction 并不存在。若后续需要严格架构差分，必须在独立阶段补齐匹配 Control 或重新预注册 OOF 设计。

## 官方实现与固定规格

- distribution：`ncps==1.0.1`
- 官方源码：`https://github.com/mlech26l/ncps.git`
- 固定 commit：`695ec22d7d1831f2ced67346ef7d2f08c525bd47`
- 状态单元：未改写的 `ncps.torch.CfC` / `ncps.torch.cfc.CfCCell`
- `input_size=48`、`units=48`、`backbone_units=48`、`backbone_layers=1`
- `mode=default`、`activation=lecun_tanh`、`backbone_dropout=0`
- `mixed_memory=false`、`proj_size=None`、无 wiring、`return_sequences=true`、`batch_first=true`

官方递归模块为 14,064 参数；历史自定义 48 维模块为 14,112 参数，仅作为容量背景记录，不构成 logit 或状态逐值对齐宣称。

## 时间与适配器契约

适配器仅做右侧 padding 排除、有限非负 dt 校验、first/zero/positive dt reason 审计和 offline/online 接口对齐。

- first event 与 same-second observation 向官方 CfC 传入真实 `dt=0`；
- positive delta 传入真实测量秒数，不截断，训练集中的 140s 和 290s 事件必须原值通过；
- 每个 parent 单独调用未修改的官方 `CfC.forward()`，以绕过上游 `ncps==1.0.1` 的 batch timespan 广播限制；
- padding 从不进入官方 recurrence；loss 与评估只读真实 causal length。

V23 明确不复刻下列历史机制：可学习 tau、`alpha=1-exp(-dt/tau)`、0.001s zero-dt floor、`bounded_observation` 增益、recurrent LayerNorm、项目自定义 candidate/gate 更新，以及 120s 正 dt 上限。官方 CfC default 自身的 `ff1/ff2/time_a/time_b` 时间插值仍是官方状态方程的一部分，不应误写为上述项目机制。

## 本阶段验收

训练前 smoke 必须验证：官方版本/commit/类模块、48×1 dense 无 memory/projection/wiring、14,064 recurrent 参数、无项目旧时间机制、train 内 140s/290s 正 dt 不截断、first/zero/positive dt 合同、全参数梯度、padding 排除与 offline/online 一致。

训练完成后只报告 development validation 的 window、eligible-prefix、parent-final 指标及 Clutter support=9。它是诊断性开发结果，不是跨 source 泛化、产品 `CLUTTER_ACCEPTED`、Unknown 拒识、最终五类、正式拒识或部署收益。
