# V27d-r13：原始包络调制微多普勒形状分支

## 实验身份

- 直接基线：冻结 `V27a-r3 Main` epoch 20，checkpoint SHA-256 为 `5c416261810bdaf6f8e6e751c51612c98d1f77c8abc23b6060d7ac2898078305`。
- 唯一变量：接入 V18 evidence 中此前未被 V27a/r9/r10 模型使用的 `envelope_modulation_profiles`。
- seed：`42、44`；数据：CQ08，V27a Main 的同一 train/validation parent 集合。
- Liquid V2、V27a Main L2 logits、L1、Knownness、Quality、L3 均未重训或改变。
- Unknown 不进入 L2；本实验只验证已知三类 L2 分类收益，Unknown/L3 不在范围内。

## 原始特征核实

原始 evidence 的 `envelope_modulation_profiles` 为每帧 64 维频率形状，频率支持约为 `300–2000 Hz`。它由去趋势原始脉冲包络、邻近非目标距离单元噪声谱和带内 RMS 计算得到，数值有限且在 train/validation 中均有大量非零样本。

此前链路只把 `frame_values` 中的 8 个 stable8 标量摘要输入 V27a；r9/r10 使用的是另一组 181-bin `residual_doppler_profiles`，并没有验证 envelope modulation profile。r13 因此不是重复 r9/r10，而是首次把该 64-bin 原始包络形状作为独立 causal profile branch 接入。

每个有效事件对可靠帧的 envelope profile 做均值并进行事件级 L1 shape normalization，再因果构造当前 shape、running mean、running std 和上一有效事件绝对差。分支输出受界约束的 Drone/Bird pair residual：`Drone:+delta/2`、`Bird:-delta/2`、`Balloon:0`。padding 和无效 profile 不更新状态，也不产生 residual；不读取 L2 logits、softmax 或标签身份。

## 数据与训练

- train：`953` parents，`17,955` 个有效 envelope-profile events；
- validation：`223` parents，`4,184` 个有效 envelope-profile events；
- 固定 parent-final class-balanced CE、batch size `32`、120 epochs、learning rate `0.001`、weight decay `0.001`；
- 两个 seed 均使用 CUDA，训练无 NaN、无非有限梯度和设备错误；
- 固定 primary epoch：`120`；validation 只记录诊断，未用于选择最佳 epoch；
- formal L3、Unknown exposure、quality/maturity 和 deployment：关闭/未执行。

## 各类别指标

指标为 validation parent-final，`delta` 为 r13 treatment 减去冻结 V27a Main control，单位为百分点（pp）。

### Seed 42

| 类别 | Support | Control P/R/F1 | r13 P/R/F1 | P delta | R delta | F1 delta | 是否提升 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| DroneTarget | 113 | 98.20% / 96.46% / 97.32% | 97.30% / 95.58% / 96.43% | -0.90pp | -0.88pp | -0.89pp | 否 |
| BirdTarget | 56 | 91.53% / 96.43% / 93.91% | 88.33% / 94.64% / 91.38% | -3.19pp | -1.79pp | -2.53pp | 否 |
| BalloonTarget | 54 | 98.11% / 96.30% / 97.20% | 98.08% / 94.44% / 96.23% | -0.04pp | -1.85pp | -0.97pp | 否 |

总体 accuracy：`96.41% → 95.07%`，下降 `1.35pp`；总正确数 `215 → 212`，下降 `3`。

### Seed 44

| 类别 | Support | Control P/R/F1 | r13 P/R/F1 | P delta | R delta | F1 delta | 是否提升 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| DroneTarget | 113 | 98.20% / 96.46% / 97.32% | 97.30% / 95.58% / 96.43% | -0.90pp | -0.88pp | -0.89pp | 否 |
| BirdTarget | 56 | 91.53% / 96.43% / 93.91% | 86.89% / 94.64% / 90.60% | -4.64pp | -1.79pp | -3.31pp | 否 |
| BalloonTarget | 54 | 98.11% / 96.30% / 97.20% | 98.04% / 92.59% / 95.24% | -0.07pp | -3.70pp | -1.96pp | 否 |

总体 accuracy：`96.41% → 94.62%`，下降 `1.79pp`；总正确数 `215 → 211`，下降 `4`。

### 两 seed 平均

| 类别 | Control P/R/F1 | r13 P/R/F1 | P delta | R delta | F1 delta | 是否提升 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| DroneTarget | 98.20% / 96.46% / 97.32% | 97.30% / 95.58% / 96.43% | -0.90pp | -0.88pp | -0.89pp | 否 |
| BirdTarget | 91.53% / 96.43% / 93.91% | 87.61% / 94.64% / 90.99% | -3.92pp | -1.79pp | -2.92pp | 否 |
| BalloonTarget | 98.11% / 96.30% / 97.20% | 98.06% / 93.52% / 95.73% | -0.06pp | -2.78pp | -1.46pp | 否 |

两 seed 平均 accuracy：`96.41% → 94.84%`，下降约 `1.57pp`；没有任何类别的 P、R、F1 在两个 seed 平均上提升。

## 混淆与诊断

冻结 control 混淆矩阵（两 seed 相同）：

```text
[[109, 4, 0],
 [  1,54, 1],
 [  1, 1,52]]
```

r13 seed42：

```text
[[108, 5, 0],
 [  2,53, 1],
 [  1, 2,51]]
```

r13 seed44：

```text
[[108, 5, 0],
 [  2,53, 1],
 [  1, 3,50]]
```

- 两个 seed 都把一个原本正确的 Drone 推成 Bird，`Drone→Bird` 增加 `1`；
- 两个 seed 都把一个原本正确的 Bird 推成 Drone，`Bird→Drone` 增加 `1`；
- Balloon→Bird 分别增加 `1` 和 `2`，说明即使 Balloon 的 correction logit 恒为零，Drone/Bird 相对位移仍会改变 Balloon 的 argmax；
- wrong→correct 为 `0`，新增错误分别为 `3` 和 `4`；
- residual 最大绝对值约为 `3.0`，已接近配置上限，提示该分支在当前训练目标下出现较强边界推动；
- 无效 profile 的 correction 和 increment 均为精确零，padding/因果/chunk smoke 均通过。

## 结论

`V27d-r13` 判定为 **`diagnostic_only_no_promotion`**，不晋级、不替换 V27a Main、不接入 L3。

这次实验确认原始 envelope modulation profile 确实可以被严格接入模型链路，但在当前“无 Unknown、无质量头、仅 parent-final 闭集 CE”的 residual 训练目标下，它没有转化为分类收益，反而稳定损害 Bird、Drone 和 Balloon 的 validation 指标。当前证据不能说明原始形状没有信息，只能说明该信息不能通过当前 residual 结构和闭集监督被可靠利用。

因此下一步不应继续扩大同一个 raw profile residual 的容量或继续调阈值。若继续研究，应先改变任务匹配方式：例如在冻结分类端点之外建立独立的 profile representation/knownness 诊断，或用明确的 Bird/Drone 错误样本审计验证 profile 是否能区分目标类边界；不能把 r13 的闭集负收益解释成原始微多普勒证据失效。

## 产物

- 配置：[V27d-r13-raw-envelope-profile-shape.json](config/V27d-r13-raw-envelope-profile-shape.json)
- Manifest：[V27d-r13-manifest.json](manifests/V27d-r13-manifest.json)
- Contract smoke：[V27d-r13-contract-smoke.json](reports/V27d-r13-contract-smoke.json)
- Seed42 evaluation：[V27d-r13-seed42-evaluation.json](reports/V27d-r13-seed42-evaluation.json)
- Seed44 evaluation：[V27d-r13-seed44-evaluation.json](reports/V27d-r13-seed44-evaluation.json)
- Two-seed summary：[V27d-r13-two-seed-summary.json](reports/V27d-r13-two-seed-summary.json)
- Seed42 training log：[training.log](runs/V27d-r13-seed42/training.log)
- Seed44 training log：[training.log](runs/V27d-r13-seed44/training.log)

Manifest SHA-256：`202cdace05f910d3253265c7756a74d2efc02a993fac9656d6e5f1119bb47f01`。

## 未执行

- Unknown/L3/open-set 评估：未执行；
- 独立 calibration pool：未执行；
- OOF、跨源验证和 deployment：未执行；
- 将原始 envelope profile 接入完整 V27 三层产品链路：未执行。
