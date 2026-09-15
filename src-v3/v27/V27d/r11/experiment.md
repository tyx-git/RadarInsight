# V27d-r11：full stable8 causal vector shape projection

## 实验身份

- 直接基线：同 seed 的 `V27d-r6-liquid_treatment` checkpoint。
- 唯一变量：把每个事件的 stable8 frame-vector 因果形状统计接入独立 knownness readout。
- Liquid V2 core、r6 base readout、r6 verifier head、L1/L2 endpoint 全部冻结。
- seed：`42、44`；数据：CQ08 的 r6 `representation_fit`、`representation_development`、`calibration` 三池。
- Unknown 不是 L2 类别；本实验只做 Bird-route knownness 诊断，正式 L3、conformal 和 deployment 关闭。

## 变量与因果边界

每个 reliable event 的 8 维 stable8 frame vector 先做事件级方向归一化，再构造四组 8 维特征：当前事件、截至当前事件的 causal running mean、causal running std、与上一有效事件的绝对差，共 32 维。无效事件和 padding 不更新 shape 状态，valid mask 不作为数值输入。

32 维特征只经过一个零初始化、无 bias 的 `shape_projection`，与冻结的 r6 base projection 相加后进入原 verifier head。训练期间唯一允许更新的参数是 `shape_projection.weight`；不读取 L2 logits、softmax、label identity 或质量事实。

训练使用 r6 同语义的 `parent_balanced_bce_on_eligible_prefixes`，development 上选择阈值并冻结迁移至 calibration。每池最多允许比直接 r6 基线多拒绝 1 个 Bird。

## 人口支持

| pool | parent | Bird positive | route false-positive negative | calibration Unknown |
| --- | ---: | ---: | ---: | ---: |
| representation_fit | 550 | 118 | 31 | n/a |
| representation_development | 505 | 108 | 32 | n/a |
| calibration | 247 | 53 | 18 | route candidate `12`（全部 Unknown `15`） |

正式 L3 要求至少 100 个质量合格 Unknown；当前 route-candidate Unknown 仍只有 `12`，因此不能把本实验数值解释为正式开放集泛化。

## 实际结果

下表为 r11 treatment 相对同 seed r6 Liquid baseline 的 paired 结果。`FP rejection` 是 route false-positive rejection；当前 route candidate 中的 Unknown rejection 与该负例 rejection 相同，但报告另列 Unknown support。

| seed | pool | Bird P/R treatment | Bird P/R baseline | FP rejection treatment | FP rejection baseline | AUROC treatment | AUROC baseline |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | development | 83.20% / 96.30% | 80.77% / 97.22% | 34.375% (11/32) | 21.875% (7/32) | 0.8411 | 0.8356 |
| 42 | calibration | 77.61% / 98.11% | 77.61% / 98.11% | 16.667% (3/18) | 16.667% (3/18) | 0.8166 | 0.8071 |
| 44 | development | 84.43% / 95.37% | 83.33% / 97.22% | 40.625% (13/32) | 34.375% (11/32) | 0.8556 | 0.8553 |
| 44 | calibration | 77.27% / 96.23% | 77.61% / 98.11% | 16.667% (3/18) | 16.667% (3/18) | 0.8208 | 0.8249 |

Unknown-only calibration result is `2/12 = 16.667%` for both r11 seeds and also for both direct r6 baselines. r11 did not correct any of the two accepted Unknowns in the direct calibration action.

两 seed aggregate：

- development Bird precision `83.81% ± 0.87pp`，recall `95.83% ± 0.65pp`；FP rejection `37.50% ± 4.42pp`；
- calibration Bird precision `77.44% ± 0.24pp`，recall `97.17% ± 1.33pp`；Unknown rejection `16.667% ± 0pp`；
- development FP rejection 相对 r6 平均 `+9.375pp`，但 recall 安全约束未在两 seed 都满足；calibration FP/Unknown rejection 平均 `0pp`。

## 结论

`V27d-r11` 判定为 **`diagnostic_only_no_promotion`**，不晋级、不接入 L3、不具备部署资格。

它证明 full stable8 causal vector shape branch 可以在 development 上改变排序并在 seed42/44 都提高 development 的 route false-positive rejection，但该方向不能迁移到 calibration 的动作边界。seed44 calibration 还额外拒绝了一个 Bird，calibration AUROC 下降约 `0.42pp`。因此继续增加同类 stable8 shape residual 或仅调整其训练容量缺乏依据。

随后对冻结 r6 的逐前缀 knownness 序列进行只读诊断时，发现较低前缀分位数聚合比末端分数更贴合“轨迹不一致”假设；该方向作为下一独立实验候选，但不能回写为 r11 已验证收益。

## 产物

- manifest：[V27d-r11-manifest.json](manifests/V27d-r11-manifest.json)
- contract smoke：[V27d-r11-contract-smoke.json](reports/V27d-r11-contract-smoke.json)
- seed42 evaluation：[V27d-r11-shape_treatment-seed42-evaluation.json](reports/V27d-r11-shape_treatment-seed42-evaluation.json)
- seed44 evaluation：[V27d-r11-shape_treatment-seed44-evaluation.json](reports/V27d-r11-shape_treatment-seed44-evaluation.json)
- two-seed summary：[V27d-r11-two-seed-summary.json](reports/V27d-r11-two-seed-summary.json)
- training logs：`runs/V27d-r11-shape_treatment-seed42.log`、`runs/V27d-r11-shape_treatment-seed44.log`

首次运行因汇总代码修复而归档在 `archive/pre-summary-fix-20260910/`，活动产物已重新绑定当前 manifest；旧产物未与活动结果混用。

## 未执行

- calibration Unknown 扩充至 100：未执行，当前来源无法扩充；
- 正式 L3/conformal/deployment：未执行；
- OOF 或跨源验证：未执行；
- 将 r11 shape branch 替换为默认 knownness：未执行。
