# RADAR-LN3 V26f frozen mixed Known/Unknown audit

实验版本：V26f。

实验类型：只读推理审计，不是训练实验。V26f 将已经封存的 V26e L1
epoch-32 五折 checkpoint、V26c L2 epoch-15 五折 checkpoint，以及 V26e
基于 Known-only OOF 得到的五折阈值，接入同一批 CQ08 混合
Known/Unknown validation/test parent，执行真实的 L1 → L2 → semantic
decision 路径。

## 固定来源

- L1：V26e，每折固定 epoch 32；不在 V26f 重新训练或重新 forward 生成
  L1 分数。
- L2：V26c，每折固定 epoch 15；对混合 parent 做一次真实 L2 forward，
  包括 Clutter 和 Unknown，以复现物理调用路径。
- L3：直接复用 V26e 的外层交叉拟合阈值；V26f 不重新拟合阈值，也不
  把五个 fold checkpoint 合并成一个部署模型。
- 数据：CQ08 的 V16 index。validation parent 为
  Drone 113、Bird 56、Balloon 54、Clutter 9、Unknown 15；test parent
  为 Drone 109、Bird 53、Balloon 58、Clutter 8、Unknown 16。

## 路由语义

L2 是否参与动作证据，只由模型产生的 L1 target route 决定；真实标签
不能屏蔽 L2 或改变路由。L2 的输出只作为已接受 KnownTarget 路由后的
子类证据。

| 条件 | 动作 |
| --- | --- |
| KnownTarget 与子类证据均充分 | Drone/Bird/Balloon |
| KnownClutter 证据充分 | ClutterTarget |
| 目标或背景证据均不足 | UNKNOWN_REJECTED |
| 目标证据充分但子类证据不足 | UNKNOWN_REJECTED |
| 观测少于 3 个事件或 target/clutter 同时接受 | PENDING_ABSTAIN |

`UnknownTarget` 不是 L1/L2 学习类别。Unknown 的拒识率只统计
`UNKNOWN_REJECTED`；`PENDING_ABSTAIN` 不计为拒识成功。

## 隔离与解释边界

Unknown 不进入训练、归一化、checkpoint 选择或阈值拟合。validation/test
仅在模型与阈值均冻结后读取，因此结果是 CQ08 混合审计证据，不是新的
训练结果。五个 fold 的均值、标准差和范围是 fold-level diagnostic
summary，不代表一个 pooled deployment endpoint。跨源、正式拒识、线上
stable、P95 和部署资格仍未执行。

## 命令

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  F:\Project\RadarInsight\src-v3\scripts\v26f_known_gate_frozen_mixed_open_set_audit_pipeline.ps1 `
  -Stage all -ProjectRoot F:\Project\RadarInsight
```

分阶段时可使用 `prepare`、`smoke` 或 `audit`。`audit` 只在 contract 和
readiness 已通过后运行。
