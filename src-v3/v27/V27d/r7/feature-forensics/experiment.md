# V27d-r7 特征取证与下一实验闸门

## 身份与边界

- 当前产物：`V27d-r7-feature-forensics`
- 直接基线：`V27d-r6-liquid_treatment-seed42`；r6 的完整比较仍以三 seed 结果为准
- 数据：CQ08，沿用 r6 的 `representation_fit`、`representation_development`、`calibration` route-matched population
- 主干：Liquid V2；本阶段没有训练、没有修改 L1/L2 endpoint、没有拟合 L3
- Unknown：仍不是 L2 第四类；Unknown 只作为 route false-positive/严格 open-set 诊断标签
- calibration Unknown support：12；正式 L3 需要至少 100，当前仍关闭

报告：[V27d-r7-feature-forensics.json](V27d-r7-feature-forensics.json)

## 已核实的数据链路

1. V21e `causal_prefix_course_circular_incoherence` 的混合 course cache 在三个 r6 role 均按 CQ08 `sample_id` 严格完整对齐，事件顺序、事件数量和文件 hash 均通过。
2. r6 cache 的 `event_frame_values` / `event_frame_mask` 在三个 role 均存在，stable8 的原始逐事件帧证据没有在 r6 cache 中丢失。
3. V27c motion cache 不是当前 r6 route population 的完整 sidecar：
   - fit 覆盖 `500/550` 个 parent，candidate Unknown 覆盖 `0/26`；
   - development 覆盖 `453/505` 个 parent，candidate Unknown 覆盖 `0/32`；
   - calibration 覆盖 `223/247` 个 parent，candidate Unknown 覆盖 `0/12`。
   因此 motion 不能作为本轮 Bird-route knownness 的实验变量，也不能通过只保留交集来冒充完整 open-set 结果。

## r6 参照结果

| Pool | route support | Bird | false positive | r6 Liquid AUROC | r6 Liquid AUPRC |
|---|---:|---:|---:|---:|---:|
| fit | 149 | 118 | 31 | 0.8890 | 0.9674 |
| development | 140 | 108 | 32 | 0.8356 | 0.9440 |
| calibration | 71 | 53 | 18 | 0.8071 | 0.9328 |

r6 的三 seed calibration rejection 仍稳定在 `2/12 = 16.67%`，所以本报告的 feature ranking 只能作为候选生成，不能作为 L3 成功结论。

## 特征取证结论

### 1. stable8 是当前最值得继续验证的特征族，但尚未直接进入训练

在 final-parent 诊断的单变量增量探针中，最稳定的候选集中在：

`pulse_envelope_rms_clipped_0_2_over_2` 的 frame-level / event-level dispersion，以及部分 modulation-SNR 的事件间变化量。

其中 `stable_frame_std_pulse_envelope_rms_clipped_0_2_over_2` 的方向为“值越低越像已知”，相对 r6 Liquid score 的 exploratory AUROC 增量为：

- development：`+0.0133`
- calibration：`+0.0283`

`stable_frame_absmax_residual_top4_resolution_cell_concentration` 也在 development/calibration 呈同方向增量，但 fit 上较弱，不能单独视为确定性收益。

这些数值来自 final-parent diagnostic aggregate；它们不是已训练 verifier 的结果，也没有接入动作阈值。

### 2. V21e circular course 是真实可用的辅助候选，但收益较弱

course 的 `mean/p90/absmax/std` 在 development 与 calibration 方向大体一致，单变量增量很小，约为 `0.002--0.007 AUROC`。它已具备因果和对齐条件，但优先级低于 stable8 dispersion。

### 3. 不能把质量覆盖 proxy 当作 Unknown 表征

`course_valid_fraction`/`stable_valid_fraction` 有一定排序能力，但它们主要反映观测质量或可用性。产品语义上必须先进入 quality/maturity/PENDING_ABSTAIN；不能仅凭这类覆盖率把低质量样本计为 UNKNOWN_REJECTED。

## 对 final-parent 诊断的限制

本报告的 stable8/circular 派生量是针对完整已观测 prefix 的 final-parent diagnostic。它证明了“原始证据存在、某些统计量有候选增量”，但没有证明这些量已经可以安全地复制到每一个训练 prefix。

正式实验前必须完成：

1. 每个 event `t` 只使用 `1..t` 的 stable8 frame 值和 mask；
2. padding、无可靠帧、RD 缺失不更新递归状态；
3. prefix 统计不读取 L1/L2 logits、softmax、label identity 或未来事件；
4. 记录 valid frame count/fraction，并将其作为质量事实或 gating-only 信息；
5. 对 final event 的 causal materialization 与本报告 aggregate 做数值一致性检查。

## 下一次唯一主变量建议

下一次正式实验建议命名为 `V27d-r7-causal-stable8-dispersion`，唯一主变量为：

> 在冻结 L1/L2 endpoint 和 Liquid V2 递归核心的前提下，向 Bird-route knownness readout 增加一个由 stable8 原始帧证据派生的 causal dispersion branch。

建议首个 branch 只使用一个原始 stable8 分量：

`pulse_envelope_rms_clipped_0_2_over_2`

并在每个 eligible prefix 输出有限的 causal 统计量：当前事件 frame std、累计 valid-event std、相邻事件变化量和 valid fraction。它们属于同一个 stable8 dispersion feature family；不要同时加入 course、motion、TR residual 或新的 L2 分类 residual。

实施约束：

- Liquid V2 raw-memory 主干、tau/dt/strict-skip 语义不变；stable8 branch 在 readout 侧接入，不改变递归方程；
- 使用 r6 route-matched population；L1/L2 endpoint、parent IDs、split、cache、loss 和训练预算固定；
- treatment 与 zero-feature control 使用同一 Liquid V2 verifier 结构和 seed `42/43/44`，避免把“新增分支容量”误当作特征收益；
- 质量不足继续 `PENDING_ABSTAIN`，不得转为 Unknown 拒识成功；
- development 选择阈值，冻结迁移 calibration；calibration 仍只做 exploratory transfer，正式 L3 关闭。

## 晋级、失败和停止标准

晋级候选条件：

- 每个 pool 相对直接 r6 route-matched baseline 新增 Bird rejection 不超过 1 个；
- development 与 calibration 的 AUROC/固定 recall rejection 方向一致，至少两个 seed 在 calibration 同方向；
- calibration 在已知 Bird recall 安全线内至少多拒识 `2/12` 个 Unknown，或在支持度不足时明确记录为“排序正向但未达到动作晋级”；
- 不新增低质量样本的 UNKNOWN_REJECTED，quality/PENDING 口径保持不变；
- paired zero-feature control 能证明收益来自 stable8 branch，而非重新初始化或容量变化。

失败条件：

- causal materialization 与 final-parent diagnostic 方向不一致；
- development 正向但 calibration 反向；
- Bird recall 超出预算；
- stable8 branch 主要依赖 valid fraction/帧数 proxy，无法与质量头语义分离。

停止条件：

- 若三 seed 中至少两次 calibration 不改善且没有新增 Unknown support，则停止继续微调 stable8 统计，转向重建独立 Unknown/calibration pool；
- motion 在完整 r6 route population 重新物化并完成 parent/hash 对齐前，不得重新尝试 motion treatment；
- 不因为单个 calibration 样本、单个 seed 或 final-parent AUROC 峰值接入 L3。

## 未执行事项

- causal stable8 prefix materialization：待下一实验实施
- `V27d-r7-causal-stable8-dispersion` 训练：未执行
- 多 seed 评估：未执行
- calibration/L3/conformal/deployment：未执行
