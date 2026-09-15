# V27d-r12：causal prefix lower-quartile aggregation

## 实验身份与唯一变量

- 直接基线：同 seed 的 `V27d-r6-liquid_treatment` knownness head。
- 唯一变量：不训练、不修改 Liquid V2 或 knownness head，把最终分数从最后有效事件分数改为 eligible causal prefix 分数的 P25。
- seed：`42、44`；数据：r6 固定的 CQ08 `representation_fit`、`representation_development`、`calibration` 三池。
- L1/L2 分类 endpoint、Unknown/L2 语义、质量 eligibility、padding 和 prefix mask 全部冻结。
- L2 logits、softmax、label identity 和 test payload 均不进入聚合；正式 L3/conformal/deployment 关闭。

P25 只对 `prefix_route_mask & candidate_population` 中的有效前缀计算，padding 不参与；candidate 若没有合格 prefix 直接 fail-closed。当前配置的 exploratory Bird recall floor 为 `96%`，但产品/正式校准安全线仍按历史协议要求 `98%`，因此即使满足 exploratory floor 也不能晋级产品 L3。

## 支持度

| pool | parent | Bird positive | route negative | quality-eligible route Unknown |
| --- | ---: | ---: | ---: | ---: |
| representation_fit | 550 | 118 | 31 | n/a |
| representation_development | 505 | 108 | 32 | n/a |
| calibration | 247 | 53 | 18 | 12 |

calibration route Unknown 只有 `12` 个，正式 L3 要求至少 `100` 个；每个样本约代表 `8.33pp`，所以本实验的 Unknown 变化只具诊断意义。

## 最终结果

| seed | pool | Bird P/R treatment | Bird P/R r6 | Unknown rejection treatment | Unknown rejection r6 | AUROC treatment | AUROC r6 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | development | 83.20% / 96.30% | 80.77% / 97.22% | 11/32 = 34.375% | 7/32 = 21.875% | 0.8223 | 0.8356 |
| 42 | calibration | 79.69% / 96.23% | 77.61% / 98.11% | 3/12 = 25.00% | 2/12 = 16.67% | 0.7463 | 0.8071 |
| 44 | development | 83.20% / 96.30% | 83.33% / 97.22% | 11/32 = 34.375% | 11/32 = 34.375% | 0.8359 | 0.8553 |
| 44 | calibration | 79.69% / 96.23% | 77.61% / 98.11% | 3/12 = 25.00% | 2/12 = 16.67% | 0.7568 | 0.8249 |

两 seed 汇总：

- calibration Unknown rejection 从 `16.67%` 提升到 `25.00%`，即稳定多拒识 `1/12` 个 Unknown；
- calibration Bird recall 从 `98.11%` 降到 `96.23%`，两 seed 都多拒识 1 个 Bird；低于产品 `98%` 安全线；
- calibration AUROC 平均从 `0.8160` 降到 `0.7516`，下降约 `6.45pp`；
- development 的 FP rejection 仅 seed42 提升，seed44 不变，不能与 calibration 形成稳定同方向证据；
- 新增拒识中包含 calibration 的 `DroneTarget` route negative，说明该聚合也会压低其他已知/错误路由，不是纯粹的 Unknown 分离器。

## 结论与停止决定

`V27d-r12` 判定为 **`diagnostic_only_no_promotion`**。

P25 验证了“前缀低分可以暴露部分 Bird-like Unknown”的现象，但它同时压低了真实 Bird 和其他已知路由的分数，且造成 ranking AUROC 明显下降。因此不能把 `P25` 作为独立硬 gate，也不能把 `3/12` 计作可部署 Unknown 能力。当前证据不支持继续在同一 12 个 Unknown 上枚举 quantile、blend 或固定阈值；下一步应优先获得更多质量合格的 Bird-like Unknown，或者建立带已知类安全约束的独立校准/学习型组合器。

## 工程审查修复

首次评估后发现阈值选择函数读取了 `minimum_bird_recall` 配置但未执行该约束，已补上检查；随后又补充了 candidate 无 eligible prefix 的真实合成负例 smoke。两次修复前产物分别保存在 `archive/pre-recall-selection-fix-20260910/` 和 `archive/pre-prefix-boundary-smoke-fix-20260910/`，活动结果绑定 manifest SHA-256：

`4084a065e2841bae3aa8e600334edecbfe515ce6b9e72ab3b1f6dfd5ce7cd90c`

## 产物

- manifest：[V27d-r12-manifest.json](manifests/V27d-r12-manifest.json)
- contract smoke：[V27d-r12-contract-smoke.json](reports/V27d-r12-contract-smoke.json)
- seed42 evaluation：[V27d-r12-prefix_p25-seed42-evaluation.json](reports/V27d-r12-prefix_p25-seed42-evaluation.json)
- seed44 evaluation：[V27d-r12-prefix_p25-seed44-evaluation.json](reports/V27d-r12-prefix_p25-seed44-evaluation.json)
- two-seed summary：[V27d-r12-two-seed-summary.json](reports/V27d-r12-two-seed-summary.json)
- 日志：`runs/V27d-r12-prefix_p25-seed42.log`、`runs/V27d-r12-prefix_p25-seed44.log`

## 未执行

- calibration Unknown 扩充至 100：未执行，当前来源无法扩充；
- 正式 L3/conformal/deployment：未执行；
- OOF、跨源验证和 test payload 评估：未执行；
- 将 P25 聚合接入产品默认 knownness：未执行。
