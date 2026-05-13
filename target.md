# 阶段性目标：Expert1 V2 与 System 收敛路线

更新时间：2026-05-13

## 当前主线

下一阶段围绕一个核心目标推进：

```text
先重构 Expert1 为“轨迹保底 + TR/RD 联合”架构，
让有 RD 样本的 joint 分支产生稳定正收益，
再把合格 Expert1 接回 System，继续提升系统 macro F1、Aircraft 和 Ground。
```

短期目标：

```text
Expert1 trajectory-only macro F1 不低于当前稳定基线
Expert1 joint-on-RD macro F1 > trajectory-on-RD macro F1
Expert1 final macro F1 >= trajectory macro F1
Expert1 final net gain on RD >= 0
Expert1 without-RD accuracy >= 0.78
Expert1 excel_track macro F1 >= 0.60
System macro F1 >= 0.75
Aircraft recall >= 0.80
Aircraft precision >= 0.80
Ground group F1 >= 0.60
```

## 当前基线与候选

最新 Expert1 V2 stable full 候选：

`artifacts/20260512-204012/expert1_v2_tr_rd_stable_full`

性能快照：

`performance_records/20260512-204012/expert1_v2_tr_rd_stable_full`

| 指标 | 值 |
|---|---:|
| best selection reason | `stage2_v2_joint_composite` |
| best validation epoch | 14 |
| final accuracy | 0.8645 |
| final macro F1 | 0.7526 |
| trajectory accuracy / macro F1 | 0.8581 / 0.7358 |
| trajectory macro F1 on RD | 0.7459 |
| joint macro F1 on RD | 0.7618 |
| joint - trajectory macro F1 on RD | +0.0158 |
| RD-only macro F1 on RD | 0.5677 |
| final rescue / harm / net on RD | 7 / 5 / +2 |
| RD-only rescue / harm / net on RD | 9 / 52 / -43 |
| with RD / without RD accuracy | 0.8734 / 0.8356 |
| cq08 / excel / raw macro F1 | 0.7659 / 0.5991 / 0.2000 |
| source-balanced macro F1 | 0.5217 |

结论：

- stable full 达成了保护轨迹分支和无 RD 路径的目标，without-RD / Excel 明显恢复。
- 该模型仍不能接入 System 主力：final macro F1 低于 20260512-124930，RD/joint 正收益过小。
- 下一轮不能继续单向加强 trajectory protection，需要在保留 stage2 ramp 的前提下恢复 RD/joint 学习强度，并增加 loss component logging。

对照 Expert1 V2 full 候选：

`artifacts/20260512-124930/expert1_v2_tr_rd_full`

性能快照：

`performance_records/20260512-124930/expert1_v2_tr_rd_full`

| 指标 | 值 |
|---|---:|
| best selection reason | `stage2_v2_joint_composite` |
| best validation epoch | 11 |
| final accuracy | 0.8613 |
| final macro F1 | 0.7706 |
| trajectory accuracy / macro F1 | 0.7774 / 0.6993 |
| trajectory macro F1 on RD | 0.7101 |
| joint macro F1 on RD | 0.7964 |
| joint - trajectory macro F1 on RD | +0.0863 |
| RD-only macro F1 on RD | 0.6627 |
| final rescue / harm / net on RD | 31 / 5 / +26 |
| RD-only rescue / harm / net on RD | 21 / 41 / -20 |
| with RD / without RD accuracy | 0.8987 / 0.7397 |
| cq08 / excel / raw macro F1 | 0.8007 / 0.5677 / 0.2000 |
| source-balanced macro F1 | 0.5228 |

结论：

- 当前 TR+RD joint 架构方向成立：有 RD 样本上 joint 明确超过 trajectory，final 在 RD 样本上产生正净收益。
- 该模型不能直接接入 System 主力：overall macro F1 低于 20260511 候选，trajectory 分支和 without-RD/excel 表现被 stage2 强融合训练压低。
- 主要风险从 raw 弱源扩展为三项：stage2 打开过猛、trajectory anchor 保护不足、无 RD 样本只能依赖退化后的 trajectory 分支。
- 下一轮目标不是改回 RD 选择性激活，而是保留 TR+RD joint，同时做 stable-stage 训练：更长 trajectory 预热、stage2 ramp 覆盖所有强融合 loss、持续 trajectory supervision。

2026-05-12 全量审查后的约束：

- 不能只看 overall accuracy；必须同时看 macro F1、without-RD、excel_track、Unknown/Clutter recall。
- raw_tr_rd 当前 test 只有 4 条，保留为弱源诊断，不作为唯一淘汰依据。
- RD-only evidence 当前仍是负净收益，不允许回退为 RD 接管主路径；RD 只作为 joint evidence 和 auxiliary teacher。
- 如果 stable full 后 trajectory 仍退化，下一步优先冻结/降学习率保护 track encoder，而不是削弱 joint 架构。

历史稳定 fallback：

`artifacts/20260510-105622/expert1_config_smoke`

该模型只能作为历史 fallback，不进入 System 主力：

| 指标 | 值 |
|---|---:|
| historical best selection reason | `fallback_no_rd_quality_checkpoint` |
| final accuracy | 0.8548 |
| final macro F1 | 0.7327 |
| trajectory accuracy | 0.8581 |
| trajectory macro F1 | 0.7486 |
| RD macro F1 on RD | 0.6174 |
| RD rescue / harm / net | 7 / 47 / -40 |
| final rescue / harm / net | 2 / 3 / -1 |
| `raw_tr_rd` accuracy / macro F1 | 0.2500 / 0.0800 |

结论：

- 当前轨迹分支可作为强 anchor 保留。
- RD 表征有进步，但当前 class-wise RD logit correction 会伤害 trajectory。
- 下一阶段不继续沿用“有 RD 就 RD logits correction”的旧主路径语义。

历史失败样本：

`artifacts/20260511-152913/expert1_v2_fast`

该模型不能作为 Expert1 候选，也不能接入 System：

| 指标 | 值 |
|---|---:|
| historical best selection reason | `fallback_no_rd_quality_checkpoint` |
| final accuracy | 0.7903 |
| final macro F1 | 0.6922 |
| trajectory accuracy | 0.7903 |
| trajectory macro F1 | 0.6922 |
| RD macro F1 on RD | 0.4481 |
| joint macro F1 on RD | 0.0434 |
| joint - trajectory macro F1 on RD | -0.6958 |
| excel_track macro F1 | 0.4594 |
| `raw_tr_rd` accuracy / macro F1 | 0.2500 / 0.0800 |

失败原因：

- `joint_logit_delta_scale = 0.0` 使 final 安全回退到 trajectory，避免了 joint 误伤。
- 但 fast 配方同时把 epochs、deep reroute auxiliary、RD forced fusion 等关键监督信号明显削弱。
- 因此本次退化主要来自 trajectory anchor 与 RD evidence 训练不足，不是 joint 直接拖坏 final。
- 后续不能继续用“削弱关键 loss 换速度”的配置做候选训练。

## P0：基线保护

目标：在改 Expert1 架构前，明确哪些能力不能丢。

### 任务

- 固定当前 Expert1 轨迹分支基线：
  - trajectory accuracy
  - trajectory macro F1
  - trajectory per-class recall / F1
  - trajectory by-source metrics
- 固定当前报告中的对照指标：
  - final-vs-trajectory macro F1
  - RD rescue / harm / net gain
  - hard-pair final net gain
  - source-aware final-vs-trajectory
- 保留 `performance_records/20260510-105622/expert1_config_smoke` 作为本轮性能快照。

### 验收

```text
所有后续 Expert1 V2 训练都能和 20260510-105622 对齐比较
trajectory-only 指标不会因为 V2 joint 改造被覆盖或丢失
```

## P0.5：真实部署窗口化切分

目标：把 Expert1 训练 / 验证从“整段离线 track”扩展到“真实部署数据流窗口”，解决当前整段样本与在线识别场景不一致的问题。

### 背景判断

当前整段样本切分适合做架构离线对比，但不适合作为最终部署指标：

- CQ08 样本通常是几十秒轨迹。
- Excel 样本可能是几十到数百秒轨迹。
- raw TR/RD 是 0.1 秒量级短片段。
- 三类来源的时间尺度差异会混入模型学习，导致 source 泛化、低空 Drone/Bird、Bird/Unknown、Unknown/Clutter 召回不稳定。

部署场景应按“数据流 -> 滑动窗口 -> 实时判断 -> 轨迹级聚合”评估，而不是只看整段 track 一次性分类。

### 切分原则

- 保留当前 `expert1_track_index.csv` 作为整段基线，不破坏历史可比性。
- 新增窗口化索引，不直接覆盖旧索引：
  - `data/processed/expert1_window_index.csv`
  - `data/manifests/expert1_window_split_manifest.json`
  - `data/processed/expert1_window_summary.json`
- 切分必须按原始 track / batch group 做 train / val / test，禁止同一条原始轨迹切出的窗口同时进入训练和测试。
- 真实部署主口径先采用：

```text
window_seconds = 10
stride_seconds = 2
min_points_per_window = 5
short_sample_policy = keep_one_full_window_for_raw
split_group_key = source_name + original sample_id
```

### 数据格式约束

- 窗口样本必须写成普通 CSV，复用现有 `TorchTrackDataset` 读取链路。
- 窗口 CSV 保留现有 `TRACK_SCHEMA_COLUMNS`。
- 窗口索引必须兼容当前 index 字段，并新增部署字段：
  - `parent_sample_id`
  - `window_id`
  - `window_start_seconds`
  - `window_end_seconds`
  - `window_duration_seconds`
  - `window_point_count`
  - `window_mode`
  - `split_group_id`
- 有 RD 的 raw / CQ08 样本暂时继承父样本 RD tensor；后续再做 RD 时间对齐裁切。
- 无 RD 的 Excel 窗口必须走 trajectory-only path。

### 评估口径

窗口级指标：

```text
window_accuracy
window_macro_f1
window_macro_recall
Drone/Bird/Unknown/Clutter per-class recall
without-RD window accuracy
source-balanced window macro F1
```

轨迹级聚合指标：

```text
parent_track_majority_vote_accuracy
parent_track_mean_logit_accuracy
parent_track_macro_f1
early_window_accuracy@first_10s
stable_decision_latency_seconds
```

### 验收

```text
expert1_window_index.csv 可生成
window split 无 parent leakage
window label/source/RD 覆盖统计进入 summary
窗口级报告能同时输出 precision、recall、macro F1
Bird recall、Unknown recall、Clutter recall 不被 overall accuracy 掩盖
raw 样本因时长不足保留为短窗诊断，不作为唯一淘汰依据
```

### 当前落地状态

已完成：

- 新增 `src/process/build_expert1_window_index.py`。
- 已生成部署窗口索引：
  - `data/processed/expert1_window_index.csv`
  - `data/manifests/expert1_window_split_manifest.json`
  - `data/processed/expert1_window_summary.json`
- 当前默认切分：

```text
window_seconds = 10
stride_seconds = 2
min_points = 5
split_group_key = source_name|parent_sample_id
```

- 当前窗口统计：

| 指标 | 值 |
|---|---:|
| total windows | 16271 |
| trainable windows | 15793 |
| diagnostic windows | 478 |
| parent tracks | 2048 |
| parent leakage count | 0 |
| train / val / test trainable windows | 11426 / 2300 / 2067 |
| with RD / without RD windows | 6154 / 10117 |
| sliding / short_full / diagnostic_sparse_full / short_full_below_min_points | 15118 / 675 / 446 / 32 |

校验结果：

```text
trainable duration > 10s = 0
trainable point_count < 5 = 0
non-trainable but duration<=10s and point_count>=5 = 0
```

训练链路已完成：

- `Expert1TrainConfig.index_name` 支持显式选择索引。
- 默认仍使用 `expert1_track_index.csv`，不破坏历史整轨迹基线。
- 新增窗口训练配置 `tmp/expert1_v2_window_stable_full_config.json`，使用 `expert1_window_index.csv`。
- `build_expert1_dataloaders` 支持窗口索引内置 `split_name`，并过滤 `is_trainable=false` 诊断窗。
- 训练采样加入 `parent_window_balance_power`，避免长轨迹被大量滑窗重复放大。
- Expert1 测试报告新增 `window_evaluation.parent_aggregation`：
  - parent mean-logit
  - parent majority-vote
  - first-window
  - stable-decision latency

仍需保留为下一步：

- 有 RD 的 CQ08 / raw 窗口当前继承父样本 RD tensor，尚未做 RD 时间对齐裁切。
- 当前 10 秒窗在 CQ08 / Excel 上常见只有 5 个轨迹点，训练后必须重点观察 Bird / Unknown / Clutter recall；若召回不足，再对比 14 秒窗或多尺度窗口。
- 窗口级训练完成后，必须同时比较窗口级指标和父轨迹聚合指标，不能只看窗口 accuracy。

## P1：Expert1 V2 架构落地

目标：把 Expert1 改成二路径结构。

当前状态：V2 已从“embedding 拼接 joint head”升级为“TR token + RD token cross-modal fusion”；20260511-224549 全量训练已证明 joint-on-RD 高于 trajectory-on-RD。

```text
无 RD：trajectory-only branch -> final logits
有 RD：trajectory tokens + RD tokens -> TR+RD 混合判断 -> final logits
```

### 任务

- 保留当前轨迹分支：
  - `traj_backbone`
  - `traj_logits`
  - `traj_residual_head`
  - 新增稳定 `traj_embedding` 输出给 joint branch
- 改造 RD 分支为 evidence branch：
  - 输出 `rd_embedding`
  - 保留 `rd_logits`、Drone/Bird、Unknown、Clutter、Unknown/Clutter、morphology heads 作为 auxiliary evidence
  - RD auxiliary 不再直接作为主路径全类 correction
- 新增 TR+RD joint branch：
  - 输入 `trajectory sequence tokens`
  - 输入 `RD sequence tokens`
  - cross-attention 后生成 `traj_fused_embedding` 与 `rd_fused_embedding`
  - 输入原始 `traj_embedding` / `rd_embedding`
  - 输入 `abs(fused_traj_embedding - fused_rd_embedding)`
  - 输入 `fused_traj_embedding * fused_rd_embedding`
  - 输入 RD stats / morphology / source quality
  - 输出 `joint_logits`
  - 输出 `track_weight` / `rd_weight`
- 替换旧主路径语义，正式推理模式为 `v2_joint`：
  - 无 RD 样本绕过 joint，直接输出 `traj_logits`
  - 有 RD 样本默认走 joint branch
  - `rd_weight` 必须 bounded、source-aware、quality-aware
- 删除或降级当前主路径中的 class-wise `traj_logits + w * (rd_logits - traj_logits)`。

### 验收

```text
无 RD 样本 final_logits == trajectory path 输出
有 RD 样本产生 joint_logits、rd_weight、track_weight
rd_weight 在无 RD 时为 0
rd_weight 有上限，初始建议 <= 0.45
RD auxiliary logits 不再直接全类接管 final logits
```

已完成：

- `LowAltitudeSuspectExpert` 新增 `joint_head` 与 `joint_reliability_head`。
- `LowAltitudeSuspectExpert` 新增 `CrossModalFusionBlock`，在 joint head 前做 TR/RD token-level cross-attention。
- `LowAltitudeFusionOutput` 新增 `joint_logits`、`joint_rd_weight`、`joint_track_weight`。
- `v2_joint` 主路径为：无 RD 输出 `traj_logits`，有 RD 输出 `joint_logits + bounded residual`。
- 无 RD / 非多模态 RD 样本 `rd_weight = 0`，final 回退到 trajectory。
- RD hard-pair 只保留小 residual，不再强接管全类 logits。
- Expert1 trainer 新增 `joint-on-RD`、`trajectory-on-RD`、`joint_minus_traj_macro_f1_on_rd` 与 joint RD weight 指标。

已完成全量验证：

- `20260511-224549/expert1_v2_tr_rd_full`：joint-on-RD macro F1 = 0.8216，trajectory-on-RD macro F1 = 0.7756。
- final RD rescue / harm / net = 22 / 5 / +17。
- 已完成 selection gate 清理：当前代码不再用 RD-only 净收益作为硬门槛，而是按 joint/final 指标选择 V2 checkpoint。

## P2：Expert1 V2 分阶段训练

目标：让 joint 分支学习 TR+RD 联合证据，而不是过早压过轨迹 anchor。

训练约束：

- 默认关闭逐 batch 参数 CPU 回滚，只保留 non-finite loss 检查和梯度裁剪。
- 验证 / 测试默认不计算 RD/deep auxiliary loss，只计算主前向和指标。
- Expert1 候选训练不再使用轻量 quick val，必须每轮 full val。
- RD specialist residual 只作为小证据修正，当前建议 `rd_specialist_residual_scale = 0.35`。
- 全量候选训练使用 `joint_logit_delta_scale = 1.0`：
  - 有 RD 样本的主 loss 直接监督 TR+RD 混合分支。
  - 无 RD 样本仍走纯轨迹分支。
  - 不再使用轻量快速训练作为候选依据。
- 速度优化只能保留工程侧加速，不再下调关键监督信号：
  - epochs 恢复到 10 到 12。
  - `low_altitude_deep_reroute_aux_weight` 恢复到 0.35。
  - `low_altitude_rd_forced_fusion_weight` 恢复到 0.9。
  - `low_altitude_joint_aux_weight = 0.35`，显式监督 `joint_logits`。
  - `low_altitude_joint_reliability_aux_weight = 0.12`，只监督 bounded residual 权重。
  - `low_altitude_joint_agreement_prior_scale = 0.25`，轨迹/RD 一致时给 joint agreed class 小 prior。
  - 旧 `rd_gate` / class gate / pair gate auxiliary 已删除，不再恢复；`v2_selection_gate` 必须按 joint/final 指标约束。
  - 每轮执行 full val，不再使用 quick val。

### Stage A：Trajectory Anchor

- 复用当前强轨迹分支 checkpoint 或低学习率微调。
- 无 RD / 有 RD 样本都要保持 trajectory-only 可评估。
- 不允许 joint 分支训练破坏 trajectory-only 基线。
- 下一轮不得再从 epoch 1 开始全量 Stage2。
- Stage A 至少保留纯 trajectory 预热，并记录 anchor checkpoint。

验收：

```text
trajectory-only macro F1 >= 0.7486 - 可配置容忍值
trajectory-only accuracy >= 0.8581 - 可配置容忍值
```

### Stage B：RD Evidence

- 训练 RD embedding 和 RD auxiliary heads。
- 训练 Drone/Bird、Unknown/Clutter、Bird/Clutter hard-pair evidence。
- 训练 Unknown residual 和 Clutter morphology evidence。
- RD auxiliary 指标用于表征质量诊断，不直接决定 final。
- 在 RD macro F1 on RD 回到 0.60 附近前，不允许 joint final 强融合。

验收：

```text
RD macro F1 on RD >= 当前 0.6174 附近稳定基线
RD Unknown/Clutter recall mean 不下降
RD Drone/Bird recall mean 不下降
```

### Stage C：TR+RD Joint

- 有 RD 样本训练 joint head。
- joint head 输入必须包含 trajectory/RD 置信度、熵、一致性、RD cosine confidence、residual score、RD stats/patch norm 等质量特征。
- 无 RD 样本回放 trajectory-only path，防止纯轨迹路径退化。
- `rd_weight` ramp-up，不允许开局强融合。
- RD specialist residual 必须乘 bounded `rd_weight`，且默认 scale 降到 0.35，避免 raw 弱源被 RD-only logits 二次误伤。
- reliability head 初始必须强 trajectory-biased，且 RD cap 从 0.05/0.10 慢慢升到 0.45。
- 若 `joint_rd_weight_on_rd_mean` 贴近上限，同时 `joint_minus_traj_macro_f1_on_rd < 0`，该 checkpoint 直接判定失败。
- 全量候选训练中，有 RD 样本 final 直接采用 TR+RD 混合分支；若 `joint-on-RD macro F1 <= trajectory-on-RD macro F1`，则判定该架构/训练配方失败，而不是回退成轻量诊断训练。

验收：

```text
joint-on-RD macro F1 > trajectory-on-RD macro F1
final macro F1 >= trajectory macro F1
final net gain on RD >= 0
raw_tr_rd final 不低于 trajectory；若 raw support 仍只有 4 条，则作为强诊断而非唯一硬否决
```

### Stage D：Calibration / Selection

- 选模不只看 final macro F1。
- 必须同时看：
  - final-vs-trajectory macro F1
  - joint-on-RD minus trajectory-on-RD macro F1
  - final net gain on RD
  - final harm rate on RD
  - hard-pair final net gain
  - source-balanced macro F1
  - raw / excel 弱源指标
- RD-only 分支只保留 evidence 质量下限，不再要求 RD-only rescue / harm / net 为正。

验收：

```text
checkpoint 使用 v2_selection_gate，不再因为 RD-only net gain 保存 fallback
joint_minus_traj_macro_f1_on_rd >= 0
final_net_gain_rate_on_rd >= 0
final_harm_rate_on_rd <= 配置阈值
Drone/Bird final net gain >= 0
Unknown/Clutter final net gain >= 0
Bird/Clutter final net gain >= 0
```

## P3：跨源与弱源评估

目标：避免 CQ08 高分掩盖 raw / excel 弱源问题。

### 任务

- `raw_tr_rd` 当前 test 只有 4 条，必须增加 bootstrap / leave-source / raw holdout 诊断。
- 弱源报告必须增加：
  - source x trajectory metrics
  - source x joint metrics
  - source x final-vs-traj
  - source x RD rescue / harm / net
- 不再只通过提高 source weight 解决弱源不稳。
- 如果 raw 仍不稳，优先做 source-specific calibration 或 raw replay。

### 验收

```text
source_balanced_macro_f1 随整体 macro F1 同步提升
raw_tr_rd final net gain >= 0
excel_track final macro F1 不低于当前稳定基线
min_source_accuracy 不低于当前稳定基线
```

## P4：System 接回

目标：只有 Expert1 V2 通过质量门槛后，才更新 System。

### 任务

- 在 Expert1 V2 过 `v2_selection_gate` 前，不把旧 fallback 作为 System 主力 checkpoint。
- 不用带 missing deep keys 的旧 Expert1 checkpoint 验证 deep reroute 收益。
- Expert1 V2 合格后，更新 `config/config.json` 的 `expert1_checkpoint`。
- 重新训练 System，并重新比较 fusion candidates。
- System 报告必须包含 Expert1 V2 的 trajectory-only / joint-on-RD / final 诊断。

### 验收

```text
System macro F1 >= 0.75
Expert1 low-altitude macro F1 不低于 Expert1 standalone 可解释水平
with_reroute macro F1 >= without_reroute macro F1
reroute_harm_count 不明显上升
Aircraft recall / precision 不因 Expert1 V2 接入下滑
Ground group F1 >= 0.60
```

## P5：Expert2 Open-Set 校准

目标：保证 System 接回后 Aircraft 不被低空或 Ground 仲裁拖垮。

### 任务

- 继续 Expert2 独立阈值校准：
  - aircraft precision
  - aircraft recall
  - false aircraft rate
  - raw source false aircraft rate
- 检查 System 中：
  - Expert2 positive but final not Aircraft
  - final Aircraft but Expert2 not positive
  - Aircraft FP by source

### 验收

```text
Expert2 independent Aircraft precision >= 0.50
Expert2 independent Aircraft recall >= 0.60
System Aircraft precision >= 0.80
System Aircraft recall >= 0.80
```

## P6：Ground / Expert4 仲裁

目标：维持或提升 Ground group F1，避免低空或 Aircraft 抢 Ground。

### 任务

- 基于新增 Ground diagnostics 重跑 System 训练和 `tests/test_ground_system_alignment.py`。
- 验证高置信 Ground candidate protection 不引入明显非 Ground FP。
- 若 `system_representation_plus_expert4_head` 好于 `system_final_global`，继续调 calibrated candidate arbitration。
- 若两者都差，回到 Expert4 head / representation 漂移治理。

### 验收

```text
ground_group_f1 >= 0.60
Ship recall >= 0.60
Car recall >= 0.70
pred_ground_on_non_ground_count 不明显增加
```

## P7：性能与工程约束

### 性能测试

- 推理延迟：
  - end-to-end
  - trajectory-only branch
  - RD evidence branch
  - TR+RD joint branch
  - deep_reroute path
  - Expert2 / Expert3 / Expert4
  - RD 预处理
- 记录显存：
  - batch size
  - RD sequence 长度
  - patch / deep patch 开关
  - joint branch 开关

### 工程约束

- 不写冗余代码和冗余函数。
- 不新增没有训练意义的标签字段。
- 不复制训练器逻辑。
- 不让细类体系破坏当前大类输出。
- 不把 RD 退回弱辅助分支。
- 不在没有可靠标注时引入新细类。
- `.history` 只读保留，不删除。
- 每次训练后的完整性能参数必须进入可被 git 跟踪的性能快照目录。

## 已落地内容归档

以下内容已经进入代码，不再作为待办项保留：

- 两层标签接口：`coarse_label` / `fine_label`。
- CQ-08 旋翼无人机细类映射与旧索引兼容。
- Expert1 正式模式：`v2_joint` / `deep_reroute`。
- Expert1 第一轮 Unknown 作为 reject signal 的基础逻辑。
- System 推理期有方向 reroute graph。
- 测试报告中的 without / with reroute 基础对比。
- Expert1 routed / true-low-altitude 双口径报告。
- 回退路径、reject reason、deep 使用率、逐类别回退前后 recall 报告。
- 回退 helped / harmed 逐样本审计。
- Expert1 low reject score 组成项进入 prediction record。
- source-balanced macro F1、min source macro F1、source x label precision / recall / F1。
- System Aircraft / Expert2 仲裁诊断。
- 回退阈值从 `config` 进入 JSON / Markdown 报告。
- `calibrated_candidates` 作为 eval-only candidate probe 接入融合模式对比。
- 专家内部性能报告。
- RD 分支 CosFace 原型余弦相似度矩阵。
- 安全关键误判对回退前后变化。
- 训练损失组件曲线进入 `training_history.json`。
- Expert1 `deep_reroute` 独立增强路径。
- Expert1 RD-only hard pair 固定报告。
- Expert1 跨源泛化选模强化。
- Ground / Expert4 仲裁基础优化。
