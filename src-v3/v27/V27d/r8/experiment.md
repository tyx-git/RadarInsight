# V27d-r8：V21e causal course × causal stable8 paired attribution

## 实验身份

- 实验：`V27d-r8-joint-course-stable8`
- 直接基线：`V27d-r6-liquid_treatment`
- 主干：冻结的 Liquid V2 raw-memory/pre-norm Bird-route verifier
- L1/L2 分类 endpoint：冻结，不重新训练
- seed：42、43、44
- 数据：CQ08；fit/development/calibration 三池固定
- Unknown：不进入 L2 分类；仅作为 route-matched knownness 负例诊断
- 正式 L3、conformal、deployment：关闭；calibration Unknown support 仍为 12，要求 100

## 唯一问题与四个配对臂

本实验只回答一个问题：在同一冻结 Liquid V2 和同一 readout 初始化下，V21e 因果 course 与 r7 causal stable8 dispersion 是否分别或联合改善 Bird-route knownness 排序。

| arm | V21e course | causal stable8 | 目的 |
|---|---:|---:|---|
| `zero_control` | 关 | 关 | 同一 r6 Liquid readout control |
| `stable8_treatment` | 关 | 开 | 复现 r7 stable8 变量 |
| `course_treatment` | 开 | 关 | 测试 V21e causal course 单独收益 |
| `joint_treatment` | 开 | 开 | 测试联合收益及交互 |

四臂的新增 projection 权重全部零初始化，训练前必须与 r6 verifier bitwise 相同；训练后只允许 readout 参数变化，Liquid core、L1/L2 endpoint 和分类 logits 不变。

## 特征与边界

- course 来源为已核验的 V21e `causal_prefix_course_circular_incoherence`，按 CQ08 index 的 `parent_sample_id + causal_event_index` 严格对齐到每个事件。
- stable8 来源为 r6 `event_frame_values` 的 channel 6 `pulse_envelope_rms_clipped_0_2_over_2`，每个事件只使用当前及历史前缀计算 `current_frame_std`、`causal_event_std`、`causal_adjacent_delta`。
- course valid mask 与 stable8 valid fraction 只用于缺失/质量契约，不作为 knownness 数值输入。
- 不加入 r5-v2 prefix-consistency loss、V27c motion、L2 logits、softmax、label identity 或质量头输出。
- padding 不更新 Liquid 状态，所有 padding feature 必须为零。

## 选择与停止标准

每个 seed 在 development pool 选择满足 Bird recall 安全预算的 threshold，并冻结迁移到 calibration。正式 L3 不因 development 单点收益启用。

候选 arm 必须同时满足：calibration Bird recall ≥98%，development/calibration Unknown 或 route false-positive rejection 同方向，且不能超过每池 1 个额外正确 Bird 拒识。任一 arm 的 calibration support 不足或跨池方向不一致，都只保留为 diagnostic。

## 实际结果

三 seed summary：`reports/V27d-r8-three-seed-summary.json`；manifest：`ebb3d465995b609fdcc7b1477d4531a2f94e876348179d57c786f4f01811e5e7`。

三池 route-matched support 固定为：fit `Bird=118 / negative=31`，development `Bird=108 / negative=32`，calibration `Bird=53 / negative=18`；其中 calibration Unknown support 仍为 `12`，不能支持正式 L3/conformal。

三 seed 均值（mean ± sample std）：

| arm | development Bird P/R | development FP rejection | calibration Bird P/R | calibration FP/Unknown rejection |
|---|---:|---:|---:|---:|
| `zero_control` | `83.78% ± 0.28 / 97.22% ± 0.93` | `36.46% ± 1.80pp` | `77.50% ± 0.20 / 97.48% ± 1.09` | `16.67% / 16.67%` |
| `stable8_treatment` | `84.18% ± 0.43 / 96.91% ± 0.53` | `38.54% ± 1.80pp` | `77.39% ± 0.20 / 96.86% ± 1.09` | `16.67% / 16.67%` |
| `course_treatment` | `83.78% ± 0.28 / 97.22% ± 0.93` | `36.46% ± 1.80pp` | `77.50% ± 0.20 / 97.48% ± 1.09` | `16.67% / 16.67%` |
| `joint_treatment` | `84.18% ± 0.43 / 96.91% ± 0.53` | `38.54% ± 1.80pp` | `77.39% ± 0.20 / 96.86% ± 1.09` | `16.67% / 16.67%` |

相对 `zero_control`：

- `stable8_treatment` 与 `joint_treatment` 在 development 的 FP rejection 平均 `+2.08pp`，development precision 平均 `+0.40pp`，但 development Bird recall 平均 `-0.31pp`。
- `course_treatment` 在 development 的动作指标与 control 相同；它没有在当前 threshold 选择下产生可观测动作收益。
- 三个 treatment 在 calibration 的 FP/Unknown rejection 都没有相对 control 提升，全部为 `16.67%`；`stable8` 和 `joint` 的 calibration Bird recall 平均 `96.86%`，低于 `98%` 安全线。`course` 平均 `97.48%`，仍不能满足所有 seed 的安全要求。
- 所有 treatment 的 calibration support 都只有 12 个 Unknown，不能把上述数值解释为正式开放集泛化。

## 工程核验

- r8 smoke 通过：四臂未训练输出与 r6 Liquid verifier bitwise 一致；course/stable8 padding 为零；sealed hash 与路径边界检查通过。
- treatment checkpoint 的分支权重已实际非零：course-only 仅 course projection 非零，stable8-only 仅 dispersion projection 非零，joint 两者均非零；四臂 Liquid core state 均与对应 r6 checkpoint 相同。
- L1/L2 classification logits 未被修改，也没有向 knownness 输入传入 logits、softmax、label identity、quality output 或 V27c motion。
- 当前 r7 活动包也已重新 materialize、重训三 seed、评估并生成 summary；r5/r5_v2/r6/r7/r8 的活动 provenance 均指向 `V27d/r*`，旧迁移前产物保留在 archive。

## 结论与停止决定

本实验验证了工程链路和两个特征分支的因果接入，但没有验证它们在当前 calibration 池上带来稳定开放集收益。`joint_treatment` 不能晋级，也不能触发动态 L3、conformal 或部署；三臂统一判定为 `diagnostic_only_no_promotion`。

当前可保留的正向证据是：stable8/joint 在 development 上相对 zero control 有小幅、两/三 seed方向一致的排序/拒识改善，且 course 与 stable8 都确实进入了独立 readout。当前不能把 development gain 转写成 Unknown 拒识能力。下一步若继续，应先扩大质量合格的独立 Bird-route Unknown calibration pool，再重新校准和审计，而不是在 12 个 Unknown 上继续调阈值。

## 产物

运行后填写：

- `manifests/V27d-r8-manifest.json`
- `cache/V27d-r8-*-population.pt`
- `reports/V27d-r8-contract-smoke.json`
- `reports/V27d-r8-all-seed{seed}-evaluation.json`
- `reports/V27d-r8-three-seed-summary.json`
- `runs/V27d-r8-{arm}-seed{seed}.log`

当前正式 calibration、conformal 和 deployment 均未启用，最终结论必须以三 seed summary、support 和跨池风险为准。
