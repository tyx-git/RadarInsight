# V27 系列实验台账与结果归因

## Material Passport

- Origin Skill: `academic-research-suite / experiment-agent`
- Origin Mode: `validate`
- Origin Date: `2026-09-06`
- Verification Status: `ANALYZED`
- Version Label: `v27_experiment_ledger_v1`
- Evidence Scope: 当前工作区内可读取的 config、manifest、checkpoint、training summary、paired report，以及 V18/V21e 历史正式报告；本次未重新训练，也未读取 test。

## 1. 结论摘要

V27 系列目前存在三种不同意义的“最佳”，必须分开解释：

1. **人工审核标记的最佳局部绝对结果**是 V27c-r21 seed42 报告中 `motion_sparse_lt_0_5` 子群。该子群 Accuracy 为 `98.1481%`、Macro-F1 为 `97.8139%`；用户标记的第 309–336 行给出了三类 P/R/F。但是 control 与 motion treatment 对全部 223 个 parent 的离散预测完全一致，三个 seed 的 treatment-control 增益都为 `0`。因此它是有价值的**难度/质量分层信号**，不是 causal motion 带来的分类提升。
2. **全量 223-parent 验证集上的最高绝对端点**是 V27c-r22/r23 seed43：Accuracy `96.8610%`、Macro-F1 `96.7541%`，比冻结 V27a 多正确 1 个 Bird。但同 seed 的 control、motion、clipping 各 arm 完全相同，所以只能归因于重训/seed，不能归因于 motion 或 clipping。
3. **当前最明确、可归因的模型级正收益**仍是 V27a-r3 Main：相对冻结 V27 baseline L2，Accuracy `+0.8969pp`、Macro-F1 `+0.9130pp`、Bird Recall `+3.5714pp`，净纠正 2 个 Bird。

开放集方面，V27b-r6 给出了一个受限但值得保留的 knownness 候选信号：在 `99%` overall known-recall 工作点，stable8 因果稳定性使 Unknown rejection 三 seed 平均增加 `12.9630pp`；但 AUROC 平均变化为 `-0.0021pp`，`97%/95%` known-recall 工作点平均各下降 `0.9259pp`，且 bootstrap 区间均跨 0。它尚不足以进入正式 calibration/L3。

当前活动 V27 baseline 并未完成开放集闭环：knownness 未拟合、quality/maturity 未拟合、L3 registry 为 `unfitted`，Unknown 拒识未启用，系统 fail-closed 为 Pending。根目录旧文档中的历史 open-set 数字没有对应的当前活动报告可复核，本台账不将其作为正式 V27 正收益。

## 2. 统一判定口径

- `pp` 表示百分点，例如 `0.9641 - 0.9552 = +0.8969pp`。
- “收益”优先指同数据、同冻结基座、同评估口径下的 treatment-control 配对净差；不同 seed、不同训练过程或不同子群之间的差异不能直接算变量收益。
- parent-final、eligible-prefix、knownness 排序、L3 产品动作分别记账，不能用其中一项替代另一项。
- 子群绝对高分只说明该子群更容易或具有不同数据结构；除非同一子群内的配对 treatment-control 也提升，否则不能写成特征的因果增益。
- 单 seed、bootstrap 95% 区间跨 0、或以 prefix 安全退化换取终态小幅变化的结果，最多记为弱/混合信号，不晋级默认配置。
- Unknown 不是 L2 第四类。known-only L2 实验不能产生 Unknown rejection、L3 或部署结论。

本台账使用以下标签：

- **明确正向**：同口径净收益清楚，关键安全方向未退化。
- **条件正向**：只在特定工作点或分层中出现，尚不能晋级。
- **混合/不采纳**：局部指标上升，但净正确数、prefix 或不确定性不支持。
- **无正向**：变量的离散指标净收益为 0 或为负。
- **工程结果**：只验证合同、路径、hash 或 evaluator，不是性能实验。

## 3. 实验总览

| 实验序号 | 问题 | 主要变量 | 结果 | 正向反馈与收益 |
|---|---|---|---|---|
| V27 baseline | 组合 V21e L1 特征、V18 stable8 L2 特征与 raw-memory Liquid V2 后，能否建立完整三层基线并保持分类能力？ | 组合基线，不是严格单因素：L1 因果前缀航向圆形不一致度；L2 RD physical4 + stable8；Liquid V2 raw memory + pre-norm computation | L1：231/232，Macro-F1 `96.9470%`；L2：213/223，Macro-F1 `95.2306%`。相对 V18 stable8 历史端点，L2 Accuracy 持平、Macro-F1 `-0.1623pp`、Bird Recall `-1.7857pp`。当前 open-set heads/L3 未拟合 | **分类无新增可归因正向；open-set 尚未完成。** L1 绝对端点较好，但组合设计不能把收益单独归因给 Liquid V2 |
| V27a-r3 Main | 冻结 V27 L2 后，独立的可靠逐帧 stable8 因果 late residual 能否恢复 Bird 证据？ | 新增 event-local frame GRU + causal parent aggregation + 有界零和 late residual；V27 基座仍冻结且仍含原 shared-Fusion stable8 | Accuracy `95.5157% → 96.4126%`；Macro-F1 `95.2306% → 96.1436%`；Bird Recall `92.8571% → 96.4286%`；213→215 正确 | **明确正向。** Accuracy `+0.8969pp`，Macro-F1 `+0.9130pp`，Bird Recall `+3.5714pp`，Bird F1 `+1.8776pp`，净纠正 2 个 Bird |
| V27b-r1/r2 | 建立冻结 V27a 后的 Drone/Bird 条件残差合同 | 路径、provenance、zero-sum/bound、padding/causality smoke | 只有合同 smoke，没有独立性能结论 | **工程结果**，收益不记账 |
| V27b-r3 | 决策状态条件化的 stable8 残差能否松动 Drone/Bird 边界？ | 在冻结 V27a 上训练 Drone-vs-Bird 条件残差 | parent-final Accuracy 不变；Macro-F1 `+0.1615pp`；1 个 correct→wrong 与 1 个 wrong→correct；eligible-prefix Macro-F1 `-0.8796pp` | **混合/不采纳。** 没有净正确样本收益，且 prefix 明显退化 |
| V27b-r4 | 修复 provenance/路径/paired-report 后，r3 信号能否复现？ | 与 r3 同模型，修复实验合同 | 精确复现 r3：parent-final `+0.1615pp`、Accuracy `0pp`；prefix `-0.8796pp`；Macro-F1 paired bootstrap 95% CI 约 `[-1.3521pp, +1.8268pp]` | **混合/不采纳。** CI 跨 0，且 intended Drone/Bird 瓶颈没有净修复 |
| V27b-r5 | 直接读取完整 stable8 frame sequence 是否优于 r4 的载体？ | 新增 event-local mask-aware frame GRU | parent-final 与 r4 相同；相对 V27a Macro-F1 `+0.1615pp`、Accuracy `0pp`；prefix `-0.9053pp`；95% CI 约 `[-1.3503pp, +1.8313pp]` | **无可采纳正向。** 对 r4 的 parent-final 增量为 `0`，prefix 还略差 |
| V27b-r6 | stable8 事件间因果稳定性是否改善冻结 L2 的 knownness？ | 在 raw trajectory knownness control 上加入可靠 stable8 的均值、波动、末次偏差与覆盖事实；分类 logits 不输入 knownness | 三 seed平均 AUROC 几乎不变；@99% known recall 的 Unknown rejection `+12.9630pp`；@97%/@95% 均为 `-0.9259pp`；所有 bootstrap AUROC CI 跨 0 | **条件正向。** 只在极保守 99% 工作点有一致方向，保留候选特征，但不能直接晋级 L3 |
| V27c-r1～r20 | 建立 `causal-tr-motion-v1` late residual 的可审计训练链路 | manifest/cache/hash、mask 不入模、causality、padding、counterfactual 等工程修订 | 多数序号仅生成 manifest/smoke；当前活动树未保留 r11，也没有 r1～r20 的正式训练结果 | **工程结果**，不能包装成 20 次性能实验 |
| V27c-r21 | TR motion 数值作为冻结 V27a 外的因果 late residual，能否带来独立 L2 增益？ | 只训练 9 维 causal motion residual；seed 42/43/44 | 三 seed 的 223 个 parent 预测均与 control 完全一致；Accuracy/Macro-F1 净差均为 `0`。但 `motion_sparse_lt_0_5` 子群 Macro-F1 `97.8139%`，比高覆盖子群高 `3.6223pp` | **分类变量无正向；分层条件正向。** causal motion 分类收益 `0`；motion coverage 显示出强难度/质量关联 |
| V27c-r22 | late residual 可能太弱；motion 在 Liquid V2 递归前进入并完整重训 L2 是否有效？ | matched control vs pre-Liquid motion adapter，完整 L2 + V27a stable8 分支重训 | seed42/43 内 treatment 与 control 预测完全一致，motion 净收益均为 `0`。seed43 比冻结 V27a 多正确 1 个 Bird，但 control 同样出现 | **motion 无正向。** seed43 的 `+0.4484pp` Accuracy、`+0.6105pp` Macro-F1 属于重训/seed 信号，不能归因 motion |
| V27c-r22e | NumPy 2 evaluator 兼容修复后，能否正确读取 r22 封存 checkpoint？ | 只修复派生评估，不改模型、数据、checkpoint 或训练 | 生成 seed42/43 paired report，并确认 r22 treatment-control 均为 0 | **工程结果**，模型收益为 `0` |
| V27c-r23 | TR 极端值是否污染主干；robust clipping 与 causal motion 是否存在主效应或交互？ | `legacy/robust clipping × zero/causal motion` 2×2，seed42/43 | 同 seed 四个 arm 的预测完全相同；motion、clipping、交互的 Accuracy/Macro-F1 均为 `0`。连续 logits 有变化，但无 argmax 翻转 | **离散分类无正向。** clipping 可保留为数值安全候选，不能宣称分类改善 |

## 4. V27 baseline：当前可复核状态

### 4.1 L1 与 L2 分类端点

| 层 | 冻结 epoch | Accuracy | Macro-F1 | 关键类别结果 | 混淆矩阵 |
|---|---:|---:|---:|---|---|
| L1 known/clutter | 14 | `99.5690%` | `96.9470%` | KnownTarget Recall `100%`；Clutter P/R/F `100% / 88.8889% / 94.1176%` | `[[223,0],[1,8]]` |
| L2 DBB | 11 | `95.5157%` | `95.2306%` | Drone P/R/F `96.4602/96.4602/96.4602%`；Bird `91.2281/92.8571/92.0354%`；Balloon `98.1132/96.2963/97.1963%` | `[[109,4,0],[3,52,1],[1,1,52]]` |

V27 L2 与 V18 stable8 历史 own-best treatment 都是 `213/223` 正确，Accuracy 完全相同。V27 的 Drone Recall 高 `0.8850pp`，但 Bird Recall 低 `1.7857pp`，Macro-F1 低 `0.1623pp`。因此当前数据只支持“错误边界重新分配”，不支持“Liquid V2 已使 L2 优于 V18”。而且两者训练超参数和模型合同并非严格 matched pair，不能进一步推导 Liquid V2 本身更差。

V27 L1 的绝对结果很好，并保留了 V21e 已验证的因果前缀航向圆形不一致度。与同为 seed42 的 V21e 历史端点作描述性比较，V27 从 `230/232` 提升到 `231/232`，Accuracy `+0.4310pp`、Macro-F1 `+2.7267pp`、KnownTarget Recall `+0.4484pp`，Clutter Recall 同为 `88.8889%`。但 V27 baseline 是组合设计，两者也不是 matched 单因素对照，故这只能记作正向端点信号，不能把收益单独归因给 raw-memory Liquid V2。

### 4.2 当前开放集状态

当前活动产物给出一致结论：

- `V27-frozen-classification-bases-seed42.json`：L1/L2 knownness 均为 `not_yet_fitted`，quality 为 `not_yet_fitted`，calibration 为 `not_yet_fitted`。
- `V27-open-set-audit.json`：`not_ready_pending_head_training`，final heads 与 calibration score cache 均不存在。
- `V27-l3-registry.json`：`status=unfitted`，固定 softmax 阈值和单距离 gate 均被禁止；未拟合时 fail-closed。

因此当前 V27 baseline 只有可冻结的 L1/L2 分类端点和开放集代码合同，不是已完成的严格三层开放集模型。根目录 `V27 baseline.md` 中曾记录 `424/451` KnownTarget route recall、`13/38` Unknown rejection 等历史数字，但对应旧 archive 当前不存在，且与活动 audit/registry 状态冲突；除非恢复并校验原报告与 registry hash，否则这些数字只能视为历史叙述，不能进入正式收益表。

## 5. V27a：当前最明确的分类正收益

V27a-r3 Main 在冻结 V27 L2 epoch11 的基础上增加独立、可靠帧级 stable8 因果证据路径。冻结基座原本已经包含 stable8 经共享 Fusion 的路径，因此本实验能证明的是“增加独立逐帧证据 residual 有用”，不能证明原共享 Fusion 已被移除或一定有害。

| 指标 | V27 L2 control | V27a-r3 Main | 增益 |
|---|---:|---:|---:|
| Accuracy | `95.5157%` | `96.4126%` | `+0.8969pp` |
| Macro-F1 | `95.2306%` | `96.1436%` | `+0.9130pp` |
| Drone Precision | `96.4602%` | `98.1982%` | `+1.7380pp` |
| Drone Recall | `96.4602%` | `96.4602%` | `0pp` |
| Drone F1 | `96.4602%` | `97.3214%` | `+0.8613pp` |
| Bird Precision | `91.2281%` | `91.5254%` | `+0.2974pp` |
| Bird Recall | `92.8571%` | `96.4286%` | `+3.5714pp` |
| Bird F1 | `92.0354%` | `93.9130%` | `+1.8776pp` |
| Balloon P/R/F | `98.1132/96.2963/97.1963%` | 相同 | `0pp` |

混淆矩阵从 `[[109,4,0],[3,52,1],[1,1,52]]` 变为 `[[109,4,0],[1,54,1],[1,1,52]]`，即净纠正 2 个 `Bird→Drone`，没有增加其他终态错误。这正面支持“Bird 困难部分来自微多普勒证据载体/融合方式，而不是 stable8 完全无效”。

以 V18 stable8 own-best 历史端点作描述性锚定，V27a Main 同样多正确 2 个 parent：Accuracy `+0.8969pp`、Macro-F1 `+0.7507pp`、Bird Recall `+1.7857pp`。这说明 V27 系列的最佳冻结分类端点相对该历史锚点确有前进；但两者不是严格 matched 重训，不能把全部差值归因给逐帧 residual 这一单项。

证据限制：原始三臂 `V27a-r3-paired-validation.json` 在当前工作区缺失，只剩 manifest 中的路径/hash 绑定。上述 Main 端点由封存 checkpoint 自带 assessment 与 V27b/V27c 后续冻结 control 交叉确认；因此可确认 Main 相对 V27 baseline 的收益，但不能在当前状态下补写 Main 相对 summary-only arm 的精确增益。

## 6. V27b：边界残差与 knownness

### 6.1 r3/r4/r5 分类残差

| 实验 | Parent-final Macro-F1 | 相对 V27a | Accuracy 增益 | Eligible-prefix Macro-F1 | 相对 V27a prefix | 裁决 |
|---|---:|---:|---:|---:|---:|---|
| V27a control | `96.1436%` | — | — | `96.5688%` | — | 冻结参考 |
| V27b-r3 | `96.3050%` | `+0.1615pp` | `0pp` | `95.6892%` | `-0.8796pp` | 混合，不采纳 |
| V27b-r4 | `96.3050%` | `+0.1615pp` | `0pp` | `95.6892%` | `-0.8796pp` | 复现 r3，但 CI 跨 0 |
| V27b-r5 | `96.3050%` | `+0.1615pp` | `0pp` | `95.6635%` | `-0.9053pp` | 对 r4 无终态增量，prefix 更差 |

r3/r4/r5 均保持 `215/223` 正确，只发生 1 个 correct→wrong 与 1 个 wrong→correct。终态 Macro-F1 上升主要来自错误去向改变后 Balloon precision 变为 `100%`，并未净纠正 Drone/Bird 瓶颈；同时 prefix 少正确 2 个 parent。r4/r5 的 paired bootstrap Macro-F1 区间均跨 0。因此这组实验告诉我们：在已经很强的冻结端点上继续堆叠条件 residual，容易移动错误而不是消除错误；终态 `+0.1615pp` 不能作为采纳证据。

### 6.2 r6 stable8 因果稳定性 knownness

| Seed | Control AUROC | Treatment AUROC | Δ AUROC | Unknown rejection Δ @99% known recall | @97% | @95% |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | `0.936835` | `0.937330` | `+0.0495pp` | `+11.1111pp` | `-13.8889pp` | `-5.5556pp` |
| 43 | `0.945434` | `0.942403` | `-0.3031pp` | `+22.2222pp` | `+5.5556pp` | `-5.5556pp` |
| 44 | `0.941475` | `0.943950` | `+0.2475pp` | `+5.5556pp` | `+5.5556pp` | `+8.3333pp` |
| Mean | `0.941248` | `0.941227` | `-0.0021pp` | `+12.9630pp` | `-0.9259pp` | `-0.9259pp` |

这是 V27 系列目前唯一直接训练并评估的独立 L2 knownness 分支实验，但它仍只是 `representation_development` 排序诊断：Unknown support 为 36，eligible known support 为 449，calibration/L3 未拟合。@99% overall known recall 时三个 seed 的 Bird coverage 都是 `96.4912%`，说明收益不是通过拒掉大多数 Bird 获得；但也说明 overall 99% 并不等于每类 99%。所有 AUROC bootstrap CI 以及 @99% rejection 的单 seed CI 都跨 0，故只能记为条件正向。

## 7. V27c：motion 与 clipping

### 7.1 r21 人工标记的最优局部区间

人工审核文件：`v27c/reports/(最优309-336)V27c-r21-motion-seed42-paired-validation.json`。第 309–336 行对应 `motion_sparse_lt_0_5` 子群的逐类结果：

| 类别 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Drone | `98.0769%` | `100.0000%` | `99.0291%` | 51 |
| Bird | `100.0000%` | `96.8750%` | `98.4127%` | 32 |
| Balloon | `96.0000%` | `96.0000%` | `96.0000%` | 25 |

该子群共 108 个 parent，Accuracy `98.1481%`、Macro-F1 `97.8139%`。与 `motion_coverage_ge_0_5` 子群相比：

| 指标 | Sparse `<0.5` | Coverage `≥0.5` | 关联差值 |
|---|---:|---:|---:|
| Accuracy | `98.1481%` | `94.7826%` | `+3.3655pp` |
| Macro-F1 | `97.8139%` | `94.1916%` | `+3.6223pp` |
| Bird Precision | `100.0000%` | `82.1429%` | `+17.8571pp` |
| Bird Recall | `96.8750%` | `95.8333%` | `+1.0417pp` |
| Bird F1 | `98.4127%` | `88.4615%` | `+9.9512pp` |

这确实是显著的分层差异，并且由于 r21 三个 seed 的离散预测都等于同一个冻结 V27a control，该分层结果在三个 seed 中完全复现。然而，它不是 treatment-control 差值：

- seed42/43/44 的 parent-final Accuracy、Macro-F1、各类 P/R/F 和混淆矩阵，motion treatment 与 frozen control 全部相同；
- 三 seed 均为 `215/223` 正确，Macro-F1 增益均为 `0`；
- motion residual 非零，置零可 bitwise 回放 control，motion 置换也会改变连续 logits，说明链路确实生效，不是特征没接入；
- 但没有任何 parent 跨越 argmax 边界，4 个既有 `Drone→Bird` 错例也均未修复。

正确解释是：motion coverage 或其相关变量识别出一个更难的轨迹群体。高 coverage 可能与轨迹长度、飞行状态、类别构成、可观测性或 Drone/Bird 重叠程度共同变化；不能解释为“motion 越少越有利”，也不能把 `+3.6223pp` 写成 causal motion 的分类收益。它的正向价值在于提示后续把 coverage/一致性作为 quality、maturity、knownness 条件或分层校准变量进行预注册验证。

### 7.2 r22/r22e：motion 前置并完整重训

| Seed | Control = Motion Accuracy | Control = Motion Macro-F1 | Bird Recall | 相对冻结 V27a | Motion 净收益 |
|---:|---:|---:|---:|---|---:|
| 42 | `96.4126%` | `96.3050%` | `96.4286%` | 正确数相同；Macro-F1 `+0.1615pp`，属于错误重排 | `0` |
| 43 | `96.8610%` | `96.7541%` | `98.2143%` | 多正确 1 个 Bird；Accuracy `+0.4484pp`、Macro-F1 `+0.6105pp` | `0` |

r22 排除了“motion 只有放在 late residual 才无效”这一解释：即便 motion 在 Liquid V2 递归前进入并重训完整 L2，同 seed 的 matched control 仍得到完全相同的预测。seed43 是当前全量验证集绝对分数最高的端点，但属于优化随机性/重训收益候选，至少还需预注册的多 seed baseline-only 复现才能决定是否替换冻结端点。

r22e 只是 NumPy 2 evaluator 兼容修订，不是新模型实验，也不能重复记一份收益。

### 7.3 r23：robust clipping × causal motion

r23 使用四个 matched arm：`legacy_control`、`legacy_motion`、`robust_control`、`robust_motion`。两个 seed 内四臂的离散预测完全一致：

- seed42：Accuracy `96.4126%`、Macro-F1 `96.3050%`，215/223 正确；
- seed43：Accuracy `96.8610%`、Macro-F1 `96.7541%`，216/223 正确；
- 两 seed Macro-F1 均值 `96.5296% ± 0.2245pp`，Bird Recall 均值 `97.3214% ± 0.8929pp`；
- motion 主效应、clipping 主效应和 clipping×motion 交互的离散收益全部为 `0`。

链路并非失效：motion/control 最大绝对 logit 差约为 seed42 `0.0923`、seed43 `0.1134`；clipping 也改变连续输出，但均无 argmax 翻转。99.9% train-only clipping 在 train 中分别修改 19 个 range event 和 19 个 speed event，在 validation 中修改 0 个 range event、3 个 speed event。如此低的 intervention 覆盖显著限制了检出离散收益的能力，但不能单独证明零收益完全由覆盖不足造成。

因此，r23 支持的结论是“clipping 可作为数值安全措施继续研究”，而不是“clipping 已改善分类”。此前发现的 train 极端值风险仍真实存在，但本次数据没有证明它影响 parent-final argmax；若继续，应预注册连续稳定性、knownness 或质量头指标，而不是再次只看离散分类。

## 8. V27 系列已经显示出的正向反馈

### 8.1 已确认的模型级正反馈

1. **可靠逐帧 stable8 的独立因果证据路径有效。** V27a-r3 相对 V27 L2 净纠正 2 个 Bird，Macro-F1 `+0.9130pp`、Bird Recall `+3.5714pp`。这是当前最强、最直接的 V27 增量证据。
2. **Bird 瓶颈可被证据载体设计松动。** V27a 在 Drone、Balloon 正确数不下降的同时把 Bird 正确数从 52 提升到 54，说明问题不是“微多普勒完全无信息”，而是共享 Fusion/事件级压缩后的利用效率不足。

### 8.2 条件正向、值得进入下一轮验证的信号

1. **stable8 的跨事件稳定性可能帮助极高 known-recall 下的 Unknown 排序。** V27b-r6 在 @99% known recall 下三 seed Unknown rejection 均上升，平均 `+12.9630pp`。但 AUROC和其他工作点不支持整体提升，必须在独立 calibration pool 上复核。
2. **motion coverage 是强难度/质量分层变量。** r21 sparse 子群比高 coverage 子群 Macro-F1 高 `3.6223pp`，Bird F1 高 `9.9512pp`。其用途更可能是 quality/knownness 条件化、分层校准或困难样本采样，而不是直接作为分类残差。
3. **完整 L2 重训存在 seed43 的单点正向端点。** r22/r23 seed43 比冻结 V27a 多纠正 1 个 Bird，Macro-F1 `+0.6105pp`；但所有 feature arm 同步出现，当前只能作为“重训策略/随机种子值得复现”的候选信号。

### 8.3 历史正向配置在 V27 中的正确定位

- V21e 的 canonical 特征是**因果前缀航向圆形不一致度/圆形离散度**，不是两套应重复拼接的特征。历史三 seed 相对各自 baseline 的 parent Accuracy 平均提升 `1.8678pp`、Macro-F1 平均提升 `13.1906pp`，combined parent net rescue 平均 `4.33` 个；这支持它继续留在 V27 L1。三个 seed 是同一组 232 个 development parent 上的优化重复，不是三个独立数据样本；它也是 V27 继承的历史证据，不应重复记为 V27 的新增收益。
- V18 stable8 的 own-best 对比曾出现 Macro-F1 `+2.1549pp`，但没有通过预注册的 same-epoch 严格门和弱类安全门。V27a 的新结果支持“stable8 在独立逐帧路径中有价值”，不能反向把 V18 的失败门改写为成功。

## 9. 已被当前证据排除或尚未证明的说法

- **“Liquid V2 已优于 V18 stable L2”**：未证明。Accuracy 相同，Macro-F1 反而低 `0.1623pp`；且不是严格 matched pair。
- **“V27b 条件残差进一步解决了 Drone/Bird 瓶颈”**：未证明。正确数不变、bootstrap 跨 0、prefix 退化。
- **“TR motion 对当前 L2 分类有正收益”**：r21 late residual、r22 pre-Liquid 重训、r23 2×2 两种强度下均未证明，离散净收益为 `0`。
- **“robust clipping 改善了分类”**：未证明。分支产生连续变化，但覆盖很低且 argmax 零翻转。
- **“V27 baseline 已完成 L1/L2 knownness、quality、动态 calibration 和 Unknown 拒识”**：与当前活动 audit/registry 不符；当前仍为未拟合、fail-closed。
- **“r21 第 309–336 行证明 motion treatment 提升”**：错误。它证明的是一个高分子群存在，不是 treatment-control 增益。

## 10. 统计与解释风险扫描

本次已检查 11/11 类统计谬误：

| 风险 | 本系列中的状态 |
|---|---|
| Simpson's paradox | r21 的总体零增益与 coverage 子群巨大差异并存；子群差异不得替代总体配对结果 |
| Ecological fallacy | 未从子群均值推断每个 parent；保留逐 parent paired 口径 |
| Berkson/selection bias | `motion_sparse_lt_0_5` 是按观测后 coverage 筛选的子集，存在选择偏差风险 |
| Collider bias | coverage 可能同时受类别、轨迹状态、长度和观测质量影响；对其条件化可能制造关联 |
| Base-rate neglect | V27b-r6 Unknown 仅 36 个，V27/V21e Clutter 仅 9 个；拒识率和 Clutter recall 不应脱离 support 解读 |
| Regression to the mean | 未把困难错例下一轮自然波动当作修复；r22/r23 seed43 只记为 seed/重训候选 |
| Survivorship bias | r6 将 eligible、Pending、Clutter 分开；不能只在 eligible 样本上宣称完整产品收益 |
| Look-elsewhere effect | V27c 有大量 manifest 修订与分层检查；r21 子群结果按探索性信号处理 |
| Garden of forking paths | 未把 r1～r20 工程迭代算成独立成功实验；后续须预注册指标与阈值 |
| Correlation ≠ causation | r21 coverage 差异明确标记为关联；只有 matched treatment-control 才用于变量归因 |
| Reverse causality | 高 motion coverage 很可能是困难轨迹状态的结果，而非导致分类错误的原因 |

综合置信度为 **CAUTION**：V27a 的方向清楚，但只有一个训练 seed；r6 的 Unknown support 小且区间跨 0；r21 是 post-hoc 分层；r22/r23 只有两个已完成 seed。当前结论足以筛选下一步变量，不足以形成部署声明。

## 11. 当前实验决策

1. 分类基座优先保留 V27a-r3 的可靠逐帧 stable8 路径；在没有三 seed matched 复现前，不因 r22/r23 seed43 的绝对高分替换正式冻结端点。
2. 不再沿用同一种 9 维 motion 表征继续堆叠分类 residual。若继续研究 motion，优先将其用于预注册的 quality/maturity、knownness 排序或时序一致性任务。
3. 将 V27b-r6 stable8 stability 与 r21 motion coverage 分别作为 knownness 候选和质量分层候选；二者不能未经消融直接合并。
4. 恢复或重新生成独立 calibration pool、L1/L2 final knownness/quality heads 和动态 L3 registry 后，再评估 KnownTarget/Clutter 安全、Pending、Unknown rejection 与 DBB 唯一集合。当前不得报告完整三层 open-set 指标。
5. 下一轮所有晋级判断至少同时报告 parent-final、eligible-prefix、逐类 coverage、Unknown support、paired transition 和置信区间，避免重复出现“绝对高分但净收益为 0”的误判。

## 12. 证据索引与缺口

主要证据：

- V27 baseline 分类冻结：`v27_baseline/runs/V27-frozen-classification-bases-seed42.json`
- V27 baseline 当前开放集审计：`v27_baseline/reports/V27-open-set-audit.json`
- V27 baseline 当前 L3 registry：`v27_baseline/calibration/V27-l3-registry.json`
- V18 historical anchor：`../processed/report/RADAR-LN3-V18-broadband-multiscale-micro-doppler8-stable-paired-validation.json`
- V21e historical anchor：`../processed/report/RADAR-LN3-V21e-known-gate-causal-course-consistency-multiseed-paired-validation.json`
- V27a Main checkpoint：`v27b/inputs/V27a-r3-main-seed42-epoch-020.pt`
- V27b r4/r5 paired reports：`v27b/reports/V27b-r4-paired-validation-seed42.json`、`v27b/reports/(最优72-107)V27b-r5-paired-validation-seed42.json`
- V27b-r6 结果汇总：`v27b/r6/results.md`
- V27c-r21 三 seed 汇总：`v27c/reports/V27c-r21-motion-three-seed-summary.json`
- V27c-r21 人工标记局部最佳：`v27c/reports/(最优309-336)V27c-r21-motion-seed42-paired-validation.json`
- V27c-r22 修复评估：`v27c/r22e/reports/V27c-r22e-seed42-paired-validation.json`、`V27c-r22e-seed43-paired-validation.json`
- V27c-r23 两 seed 汇总：`v27c/r23/reports/V27c-r23-two-seed-summary.json`

当前证据缺口：

- manifest 绑定的 `artifacts/archive/v27/v27a-history/20260903/reports/V27a-r3-paired-validation.json` 当前不存在；需恢复后才能审计 V27a summary arm 与 Main arm 的完整配对差异。
- V27 baseline 历史 open-set 报告/registry 当前不存在或已被未拟合活动产物替代；需恢复原 hash 链或按当前代码重训、重校准。
- V27c-r22 只存在 seed42/43 派生报告，未形成配置中声明的三 seed 结论。
- r23 正式报告由 runtime shim 补入缺失的 `build_initialized_model` 符号后生成；后续新 revision 应在源码中修复导入并重新绑定 manifest，不能静默改写封存 r23。
