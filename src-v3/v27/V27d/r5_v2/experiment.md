# V27d-r5-bird-route-v2

## Material Passport

- 状态：`ANALYZED`；seed 42；单 seed 诊断实验。
- 实验身份：`V27d-r5-bird-route-v2`。
- 隶属实验：V27d。
- 直接基线：`V27d-r5-control`，即 r5 的 raw causal control head。
- 数据：CQ08；`representation_fit`、`representation_development`、`calibration`。
- 正式 L3、conformal deployment、test payload 均未启用。

## 实验问题与唯一变量

本实验回答：Bird-route Unknown 被 L2 判为 Bird 后，是否可以通过因果前缀一致性目标，降低仅凭不稳定事件轨迹获得高 knownness 的风险。

冻结条件：

- L1、L2 checkpoint 和所有分类端点冻结；Unknown 仍不是 L2 第四类。
- r5 control 的 population、parent 顺序、质量规则、padding 和 raw causal 输入保持不变。
- knownness head 结构保持不变：causal GRU + 32 维 projection。
- knownness head 禁止读取 L2 logits、softmax 和 label identity；分类端点只用于审计报告。
- seed 42、batch size 32、30 epochs、学习率、weight decay 和 gradient clipping 保持不变。

唯一变量是训练目标：

```text
L_v2 = L_parent_balanced_BCE
     + 0.35 * L_Bird_prefix_consistency
     + 0.35 * L_Unknown_terminal_uplift
```

其中 Bird 正例约束有效成熟前缀的 knownness 接近末端分数；Bird-route Unknown 约束末端分数不能相对成熟前缀均值产生超过 `0.05` 的 uplift。该目标不改变状态方程、输入特征或 L2 分类端点。

## Bird-route 错误样本审计

审计对象是 calibration 中的 12 个质量合格 Bird-route Unknown：r5 control 接受为 Bird 的 11 个，以及拒识的 1 个。质量合格定义仍由确定性质量规则给出，低质量样本不计入 Unknown 拒识。

| 审计项 | 结果 |
| --- | ---: |
| Bird-route Unknown 总数 | 12 |
| r5 control 接受为 Bird | 11 |
| r5 control 拒识 | 1 |
| 最终质量不合格 | 0/12 |
| 单次状态更新占主导 | 0/12 |
| 前缀 knownness 不一致 | 6/12 |
| RD/微多普勒事件稳定性异常 | 10/12 |

逐样本审计包含有效事件数、quality/dt facts、innovation norms、事件间 Bird 预测一致性、Bird 分类 margin、raw state/update 贡献、stable8 事件变化和 r5 knownness 分数轨迹。审计因此支持“事件级 RD/微多普勒证据不稳定及前缀差异”是值得诊断的现象，但不能证明它们在独立池上具有稳定可分性。

一个重要排除结论是：这 12 个样本并非因为最终质量不足而被错误接受；所有样本最终质量均合格，且 r5 control 的 L2 Bird 路由在多数样本上持续为正 margin。另一个排除结论是：失败通常不是最后一个事件的单点突刺，因而 v2 使用的是末端相对成熟前缀均值的 uplift 约束，而不是单一末事件 jump 门。

## 人口与训练产物

| pool | Bird positive | Bird-route Unknown negative | quality eligible |
| --- | ---: | ---: | ---: |
| representation_fit | 123 | 26 | 546 |
| representation_development | 113 | 32 | 501 |
| calibration | 55 | 12 | 244 |

- selected epoch：`26`。
- 训练无非有限 loss 或 gradient；最终日志显示 epoch 30 完成。
- 阈值选择使用 CPU canonical replay，避免 CUDA/CPU 浮点候选阈值差异。
- checkpoint SHA256：`789803e6f607cd7dfb1f2a881c60fe1fd5a672f8e29ce354c789c00250ed6fdc`。
- manifest SHA256：`b9ec912edb97d3219f5888f2dbc2f60535196c7a0244897e62ce4a08666c57df`。

## 结果

| pool | AUROC | Bird recall | Unknown rejection |
| --- | ---: | ---: | ---: |
| development | 0.803927 | 111/113 = 98.23% | 4/32 = 12.50% |
| calibration threshold transfer | 0.736364 | 55/55 = 100.00% | 1/12 = 8.33% |

与直接基线 r5 control 的配对变化：

| 指标 | development | calibration |
| --- | ---: | ---: |
| Unknown rejection delta | 0.00pp | 0.00pp |
| Bird recall delta | 0.00pp | 0.00pp |

r5 control 的对应 AUROC 为 development `0.813053`、calibration `0.748485`。因此 v2 的 AUROC 没有提升，calibration AUROC 反而低于 r5 control；Bird recall 安全线满足，但 Unknown 拒识没有任何增益。

## 结论与晋级判定

结论：**v2 失败，不晋级 L3，也不具备部署资格。**

已满足：

- calibration Bird recall `>=98%`；
- 没有新增 calibration 正确 Bird 拒绝；
- L1/L2 分类端点未改变；
- Unknown 未进入 L2 classifier；
- 低质量样本未被计作 Unknown 拒识成功。

未满足：

- calibration Unknown rejection 相对 r5 control 提升至少 `15pp`；
- development 与 calibration 同方向提升。

由于 calibration 只有 12 个质量合格 Bird-route Unknown，每个样本代表 `8.33pp`，本实验不能证明稳定的开放集泛化能力。正式 L3 要求至少 100 个质量合格 Unknown；当前来源无法扩充，因此 L3 保持关闭。

工程判断：r5 已经具备因果序列输入，单纯增加前缀一致性损失没有产生跨池收益。继续堆叠同类 knownness 正则、stable8 或固定阈值没有当前证据支持。下一阶段若继续，必须先解决 Bird-like Unknown 的来源覆盖/异质性问题，或重新获得足够独立的校准支持；在当前 Unknown 无法扩充的限制下，应停止把该池的局部 rejection 当作模型能力提升。

## 产物

- 配置：[V27d-r5-bird-route-v2.json](config/V27d-r5-bird-route-v2.json)
- Manifest：[V27d-r5-bird-route-v2-manifest.json](manifests/V27d-r5-bird-route-v2-manifest.json)
- 错误审计：[V27d-r5-bird-route-v2-seed42-error-audit.json](reports/V27d-r5-bird-route-v2-seed42-error-audit.json)
- 评估：[V27d-r5-bird-route-v2-seed42-evaluation.json](reports/V27d-r5-bird-route-v2-seed42-evaluation.json)
- Contract smoke：[V27d-r5-bird-route-v2-contract-smoke.json](reports/V27d-r5-bird-route-v2-contract-smoke.json)
- Checkpoint：[V27d-r5-bird-route-v2-control-seed42.pt](checkpoints/V27d-r5-bird-route-v2-control-seed42.pt)
- 训练摘要：[V27d-r5-bird-route-v2-control-seed42-training-summary.json](runs/V27d-r5-bird-route-v2-control-seed42-training-summary.json)
- 训练日志：[V27d-r5-bird-route-v2-control-seed42.log](runs/V27d-r5-bird-route-v2-control-seed42.log)

历史修复前产物保存在 `archive/` 下，未与最终 provenance 混用。

## 未验证事项

- 仅 seed 42；未执行多 seed 稳定性验证。
- Unknown support 只有 12；未达到正式 L3 的 100 个支持要求。
- 未执行正式 conformal coverage、产品 L3 action、跨源验证和 test payload 评估。
- `sol_advisor` 深度分析调用因外部服务返回 `403 Forbidden: insufficient balance` 未完成；本文件结论仅依据本地真实训练、审计、smoke 和评估产物。
