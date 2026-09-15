# V27d-r6 Bird-route Liquid V2 Verifier

## 实验身份

- 实验：`V27d-r6-bird-liquid-verifier`
- 直接基线：`V27d-r5-control`
- 实验内 control：64 维 GRU Bird verifier
- treatment：64 维 Liquid V2 Bird verifier
- seed：42、43、44
- 数据：CQ08
- L1/L2 分类 endpoint：全部冻结
- Unknown：不进入 L2 三分类
- 正式 L3、conformal、deployment：关闭

## 实验目的

在保持 Bird 动作召回基本稳定的前提下，提高最终 Bird 动作 precision。当前实际部署候选中，Bird-route false positives 不仅包含 Unknown，也包含被 L2 误判为 Bird 的 Drone、Balloon 和 Clutter，因此 verifier 采用 route-matched population。

本实验唯一因果变量是递归核心：

```text
GRU(362 -> 64)  versus  LiquidV2(362 -> 64)
```

两臂递归参数量均为 `82,176`。公共输入、readout、损失、数据人口、训练预算和阈值协议完全相同。

## Population

候选定义：

```text
quality_eligible
AND L1 predicted Target
AND L2 predicted Bird
```

| Pool | Bird | route false positives | candidate total |
| --- | ---: | ---: | ---: |
| fit | 118 | 31 | 149 |
| development | 108 | 32 | 140 |
| calibration | 53 | 18 | 71 |

负例包含 Unknown、Drone、Balloon 和 Clutter；Unknown 仍不是 L2 类别。

## 训练方法

每个 parent 先对有效成熟 prefix 求平均 BCE，再做严格二组平衡：

```text
L = 0.5 * mean(Bird parent loss)
  + 0.5 * mean(route false-positive parent loss)
```

固定条件：batch 32、30 epochs、AdamW、learning rate `5e-4`、weight decay `1e-3`、gradient clipping `5.0`。Liquid 使用项目正式 raw-memory/pre-norm Liquid V2，tau `[0.05, 30.0]`、`bounded_observation`、invalid dt gain `0.25`。

## 三 seed 结果

### Development

| Seed | Arm | Bird precision | Bird recall | false-positive rejection | AUROC |
|---:|---|---:|---:|---:|---:|
| 42 | GRU | 81.40% | 97.22% | 25.00% | 0.824653 |
| 42 | Liquid | 80.77% | 97.22% | 21.875% | 0.835648 |
| 43 | GRU | 79.55% | 97.22% | 15.625% | 0.731192 |
| 43 | Liquid | 80.30% | 98.15% | 18.75% | 0.837963 |
| 44 | GRU | 80.15% | 97.22% | 18.75% | 0.843171 |
| 44 | Liquid | 83.33% | 97.22% | 34.375% | 0.855324 |

Liquid mean：Bird precision `81.47% ± 1.63pp`，Bird recall `97.53% ± 0.53pp`，false-positive rejection `25.00% ± 8.27pp`。

### Calibration threshold transfer

阈值只在 development 选择，冻结迁移到 calibration。

| Seed | Arm | Bird precision | Bird recall | false-positive rejection | Unknown rejection | AUROC |
|---:|---|---:|---:|---:|---:|---:|
| 42 | GRU | 77.61% | 98.11% | 16.67% | 16.67% | 0.807128 |
| 42 | Liquid | 77.61% | 98.11% | 16.67% | 16.67% | 0.807128 |
| 43 | GRU | 75.36% | 98.11% | 5.56% | 8.33% | 0.692872 |
| 43 | Liquid | 77.94% | 100.00% | 16.67% | 16.67% | 0.809224 |
| 44 | GRU | 77.94% | 100.00% | 16.67% | 16.67% | 0.811321 |
| 44 | Liquid | 77.61% | 98.11% | 16.67% | 16.67% | 0.824948 |

Liquid mean：Bird precision `77.72% ± 0.19pp`，Bird recall `98.74% ± 1.09pp`，false-positive rejection `16.67% ± 0pp`。

## Paired Liquid minus GRU

| Seed | Development precision | Calibration precision | Development FP rejection | Calibration FP rejection | Calibration Bird recall |
|---:|---:|---:|---:|---:|---:|
| 42 | -0.63pp | 0.00pp | -3.125pp | 0.00pp | 0.00pp |
| 43 | +0.76pp | +2.58pp | +3.125pp | +11.11pp | +1.89pp |
| 44 | +3.33pp | -0.33pp | +15.625pp | 0.00pp | -1.89pp |

均值：

- development precision delta：`+1.15pp`
- calibration precision delta：`+0.75pp`
- development false-positive rejection delta：`+5.21pp`
- calibration false-positive rejection delta：`+3.70pp`
- calibration Bird recall delta：`0pp`

但 Liquid 在 calibration 的 precision 增益并非三个 seed 同方向，false-positive rejection 也只有一个 seed 明显优于 GRU。因此不能晋级为默认 verifier。

## 成功标准判定

| 条件 | 结果 |
| --- | --- |
| 每个 pool 最多新增损失 1 个 Bird | 通过 |
| 每个 seed、每个 pool precision 至少 +5pp | 失败 |
| 每个 seed calibration FP rejection 同方向提升 | 失败 |
| calibration FP rejection 达到 25% | 失败，三个 Liquid seed均16.67% |
| development 与 calibration 同方向 | 不稳定 |
| formal L3 support >=100 Unknown | 失败，当前仅12 |

最终判定：**Liquid treatment 为诊断性结果，不晋级 L3 或部署。**

## 工程结论

扩张为独立 64 维 Liquid verifier 可以提高 development 的部分排序指标，且 seed43 calibration 出现正向配对结果；但收益没有在三个 seed、两个池中稳定复现。当前最强的稳定信号是 calibration false-positive rejection 在所有 arm/seed 组合中最多达到 `16.67%`，说明 12 个 Unknown 和有限 false-positive population 仍是主要限制。

因此不能继续通过增大 Liquid hidden size 追逐单点结果，也不能把 seed44 development 的 `83.33%` precision 写成泛化收益。当前应保留 r6 作为容量方向的负/混合证据，停止继续扩容，除非先扩大或重构独立 Unknown/calibration 来源。

## 产物

- 配置：[V27d-r6-bird-liquid-verifier.json](config/V27d-r6-bird-liquid-verifier.json)
- Manifest：[V27d-r6-manifest.json](manifests/V27d-r6-manifest.json)
- Contract smoke：[V27d-r6-contract-smoke.json](reports/V27d-r6-contract-smoke.json)
- Seed42 评估：[V27d-r6-paired-seed42-evaluation.json](reports/V27d-r6-paired-seed42-evaluation.json)
- Seed43 评估：[V27d-r6-paired-seed43-evaluation.json](reports/V27d-r6-paired-seed43-evaluation.json)
- Seed44 评估：[V27d-r6-paired-seed44-evaluation.json](reports/V27d-r6-paired-seed44-evaluation.json)
- 三 seed 汇总：[V27d-r6-three-seed-summary.json](reports/V27d-r6-three-seed-summary.json)
- GRU 日志：`runs/V27d-r6-gru_control-seed42.log`、`seed43.log`、`seed44.log`
- Liquid 日志：`runs/V27d-r6-liquid_treatment-seed42.log`、`seed43.log`、`seed44.log`

## 未执行与限制

- 正式 L3、conformal coverage、deployment、test payload：未执行。
- calibration Unknown support 仅 `12`，无法满足正式 `100` support 要求。
- 未进行 OOF、跨源验证。
- 未调用实验顾问，后续判断完全基于当前项目代码与本地实验产物。
