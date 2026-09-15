# V27d-r7 causal stable8 dispersion

## 实验身份

- 直接基线：V27d-r6 Liquid V2 treatment
- 唯一变量：`pulse_envelope_rms_clipped_0_2_over_2` 的 causal stable8 dispersion readout branch
- control：Liquid V2 + branch 全零
- treatment：Liquid V2 + causal dispersion 三维分支
- seed：42、43、44
- L1/L2 endpoint：冻结
- calibration Unknown support：12；正式 L3 关闭

## 因果特征

每个 event prefix 只使用当前及历史事件的 stable8 原始帧值：

1. 当前事件可靠帧标准差；
2. 从首个有效事件到当前事件的事件级标准差；
3. 当前事件与上一有效事件的绝对变化量。

`valid_fraction` 写入 cache 供质量诊断，但不进入 knownness readout，避免质量 proxy 被当成 Unknown 表征。

## 固定条件

- Liquid V2 raw-memory/pre-norm 递归方程不变；
- r6 Liquid checkpoint 初始化；Liquid core 冻结，只训练 readout/head；
- r6 route-matched population、parent-balanced loss、batch、epoch、学习率、阈值选择协议固定；
- treatment 与 zero control 使用相同 Liquid 初始化和相同 seed；
- development 选择阈值，冻结迁移 calibration；正式 L3/conformal/deployment 关闭。

## 结果

训练、评估和 summary 已完成。所有训练使用 CUDA，评估也固定使用 `cuda:0`，避免阈值边界在 CPU/CUDA 间漂移。

### Development：stable8 treatment vs zero control

| Seed | Control Bird P/R | Treatment Bird P/R | Control FP rejection | Treatment FP rejection | Treatment - Control |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.8387 / 0.9630 | 0.8387 / 0.9630 | 37.50% | 37.50% | 0.00pp |
| 43 | 0.8346 / 0.9815 | 0.8400 / 0.9722 | 34.375% | 37.50% | +3.125pp |
| 44 | 0.8400 / 0.9722 | 0.8468 / 0.9722 | 37.50% | 40.625% | +3.125pp |

### Calibration：stable8 treatment vs zero control

| Seed | Control Bird P/R | Treatment Bird P/R | Control Unknown rejection | Treatment Unknown rejection |
|---:|---:|---:|---:|---:|
| 42 | 0.7727 / 0.9623 | 0.7727 / 0.9623 | 16.67% (2/12) | 16.67% (2/12) |
| 43 | 0.7761 / 0.9811 | 0.7761 / 0.9811 | 16.67% (2/12) | 16.67% (2/12) |
| 44 | 0.7761 / 0.9811 | 0.7727 / 0.9623 | 16.67% (2/12) | 16.67% (2/12) |

### 三 seed汇总

- Treatment development Bird precision：`0.8418 ± 0.0043`；false-positive rejection：`38.54% ± 1.80pp`。
- Treatment calibration Bird precision：`0.7739 ± 0.20pp`；Unknown rejection：`16.67% ± 0pp`。
- 相对 zero control，development rejection 在 `2/3` 个 seed 提升，平均 `+2.083pp`；calibration `0/3` 个 seed 提升。
- 相对 r6 Liquid baseline，development rejection 提升 `+15.625pp / +18.75pp / +6.25pp`；但该对比包含 r7 readout/head 重训，不能单独归因于 stable8 branch。
- 相对 r6 Liquid baseline，三个 seed 每个池最多新增 `1` 个 Bird rejection，满足 recall budget；Unknown calibration rejection 仍为 `2/12`。

### 结论

本实验为 **`diagnostic_only_no_promotion`**。

stable8 dispersion 在 development 上出现可重复但幅度有限的正向排序/动作信号；然而 calibration 三个 seed 的 Unknown rejection 完全没有提升，且 seed44 calibration 额外拒绝了 1 个 Bird。由于 calibration Unknown support 只有 12，且未达到正式 L3 所需的 100，stable8 branch 不接入产品 L3，也不替换 r6 默认 verifier。

这次实验支持的工程结论是：stable8 离散度可能是 Bird-route knownness 的辅助信号，但目前只证明了 development 方向，尚未证明跨池开放集收益。下一步应优先扩大或重建独立 Bird-like Unknown calibration pool，而不是继续微调该分支。

产物：

- `reports/V27d-r7-contract-smoke.json`
- `reports/V27d-r7-paired-seed42-evaluation.json`
- `reports/V27d-r7-paired-seed43-evaluation.json`
- `reports/V27d-r7-paired-seed44-evaluation.json`
- `reports/V27d-r7-three-seed-summary.json`
- `runs/*.log`

归档的早期未最终确认产物位于 `archive/`，包括 readout exact-replay 修复前、训练入口修复前和 CPU/CUDA evaluation 修复前的版本。

## 晋级标准

- 每个池新增 Bird rejection 不超过 1 个；
- calibration 至少两个 seed 相对 zero control 提升 false-positive rejection；
- development 与 calibration 方向一致；
- 不把低质量样本计为 Unknown 拒识成功；
- calibration support 不足时不接入正式 L3。

## 未执行

- 正式 L3、conformal、deployment：未执行；
- calibration Unknown support 扩充至 100：未执行；
- OOF、跨源验证：未执行；
- stable8 分支晋级为默认 knownness verifier：未执行。
