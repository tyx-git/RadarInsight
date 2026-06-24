# 阶段性目标：Expert1 V2 与 System 收敛路线

更新时间：2026-05-16

## 当前主线

下一阶段围绕一个核心目标推进：

```text
Expert1 当前优先级从“训练/推理速度修复”切回“窗口化精度恢复”。
已落地的 RD segment-store、窗口聚合、pin-memory / prefetch 和 stage1 快路径继续保留。
无 RD 继续走强 trajectory-only 分支；有 RD 必须让 TR+RD joint 分支稳定超过 trajectory-only。
当前阶段不继续堆模型复杂度，先修正选模、验证、采样、RD 监督强度和窗口级弱类泛化。
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

工程性能目标：

```text
TR-only branch 保持毫秒级响应
TR+RD branch 支持 micro-batch 推理
RD 滑窗重复计算显著降低
GPU 利用率不再长期被 CPU/RD IO 阻塞
端到端 RD 数据流吞吐优先按 100 Mb/s 场景验证，并保留 100 MB/s 压力评估口径
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

## P0：Expert1 窗口化精度恢复闭环（当前最高优先级）

### 2026-05-16 审查依据

最新窗口化训练报告：

`reports/real/20260516-011202/expert1_v2_window_stable_full.json`

关键指标：

| 指标 | 值 |
|---|---:|
| final accuracy | 0.8665 |
| final macro F1 | 0.7633 |
| trajectory macro F1 | 0.7859 |
| trajectory macro F1 on RD | 0.7807 |
| joint macro F1 on RD | 0.7664 |
| joint - trajectory macro F1 on RD | -0.0143 |
| RD evidence macro F1 on RD | 0.6139 |
| final net gain rate on RD | +0.0092 |
| final harm rate on RD | 0.0389 |
| source-balanced macro F1 | 0.4340 |
| excel_track macro F1 | 0.4524 |
| raw_tr_rd macro F1 | 0.0800 |
| parent mean-logit macro F1 | 0.8115 |

数据口径：

```text
expert1_window_index.csv trainable windows = 15793
train / val / test = 11426 / 2300 / 2067
with RD / without RD windows = 6154 / 10117
test Clutter / Unknown windows = 61 / 87
RD segment store parent_count = 1228
RD segment_count = 9994
RD time sync status = proportional_fallback 9994
```

判断：

- 本轮训练没有减少主训练窗口，也没有关闭 TR+RD 主路径；速度提升主要来自 RD segment-store、DataLoader 加速、stage1 快路径和 deep auxiliary 抽样。
- 当前模型不能作为精度候选：窗口级 final macro F1 低于 trajectory macro F1，且 RD 样本上 joint 分支低于 trajectory 分支。
- 父轨迹聚合 macro F1 达到 0.8115，说明窗口预测存在抖动和弱类窗口召回问题，不是模型完全不可用。
- RD evidence on RD 达到 0.6139，但 RD-only harm 仍高，说明 RD 特征有信息量，joint/reliability 还没有学会稳定采用正确 RD evidence。
- Clutter / Unknown 的窗口支持少，Drone/Bird 仍是主要混淆；后续必须以 macro F1、Unknown/Clutter recall、joint gain 和 parent aggregation 同时选模。

### 明确问题

1. Stage1 验证语义不一致。

当前训练阶段 `stage2_enabled=false` 时，训练前向关闭 TR+RD fusion；但验证前向未同步关闭 fusion，导致 epoch 1 到 epoch 3 的 final 验证指标混入未训练 joint 路径。后续必须让 stage1 train / val 都只评估 trajectory final，同时单独输出诊断性的 joint probe。

2. Checkpoint 保存策略不足。

本轮所有 full validation 都没有通过 `selection_eligible`，最终使用 `fallback_candidate_model.pt`。但训练历史显示 epoch 12 的 validation final macro F1 高于 epoch 9，而 fallback 只保存综合分最高模型，缺少 `best_final_macro_f1`、`best_joint_gain`、`last_full_val` 对照 checkpoint，导致无法精确判断哪一轮在测试集上更优。

3. 质量相关监督被压得偏弱。

当前速度版配置中：

```text
low_altitude_deep_reroute_aux_weight = 0.2
low_altitude_deep_reroute_aux_interval = 4
low_altitude_deep_reroute_max_frames = 32
low_altitude_rd_forced_fusion_weight = 0.45
epochs = 12
full_val_interval_epochs = 3
```

这些设置适合验证训练速度，但不适合作为精度冲刺候选。尤其 deep reroute auxiliary 和 forced fusion 偏弱，会让 RD/joint 深层证据不足。

4. 窗口化训练存在源/模态/弱类竞争。

Excel 无 RD 窗口数量明显多于 RD 窗口；CQ08 / raw 才提供 RD。若普通 batch 随机混合，stage2 中 RD joint 的有效监督比例不足，且 Clutter / Unknown 的窗口支持远低于 Drone / Bird。

5. Source macro 指标需要补充 supported-class 版本。

当前按来源 macro F1 会把某个 source 中 support=0 的类别记为 0，从而压低 excel / raw 的 source-balanced macro F1。该指标仍可作为弱源风险提示，但不能单独作为硬门槛；需要新增 supported-class source macro F1，避免选模被无支持类别误导。

6. RD 时间同步仍是比例回退。

当前所有 RD segment 的 time sync status 都是 `proportional_fallback`。这比完整父 RD 继承更贴近部署，但仍不是严格时间戳对齐。若 RD joint 在质量版训练后仍不稳定，下一步必须优先补齐显式 RD 时间轴同步。

### 优化方案

#### P0-A：先修评估与 checkpoint

必须先完成：

- stage1 validation 与 stage1 training 一致：`stage2_enabled=false` 时 final 强制等于 trajectory-only。
- 报告同时保留 trajectory final metrics、diagnostic joint probe metrics、final-vs-trajectory delta、parent mean-logit / majority-vote metrics。
- 每个 full validation 至少保存四类 checkpoint：
  - `best_selection_model.pt`
  - `best_final_macro_f1_model.pt`
  - `best_joint_gain_model.pt`
  - `last_full_val_model.pt`
- 测试报告必须标明最终测试使用的是哪一种 checkpoint，不允许只有 fallback 名称。
- 当没有 checkpoint 通过 `selection_eligible` 时，训练报告必须明确标记 `candidate_status=invalid_for_release`，但仍测试上述诊断 checkpoint。

验收：

```text
epoch 1-3 stage1 final == trajectory
训练报告显示 best_final_epoch / best_joint_gain_epoch / last_full_val_epoch
fallback 不再覆盖唯一候选判断
```

#### P0-B：恢复质量版训练强度

保留工程加速项：

```text
use_rd_segment_store = true
pin_memory = true
persistent_workers = true
prefetch_factor = 2
non_blocking_h2d = true
stage1 disable_tr_rd_fusion fast path = true
```

恢复精度相关配置：

```text
epochs = 18
stage1_traj_only_epochs = 3
stage2_ramp_epochs = 5
full_val_interval_epochs = 1
quick_val_max_batches = 0
low_altitude_deep_reroute_aux_weight = 0.35
low_altitude_deep_reroute_aux_interval = 2
low_altitude_deep_reroute_max_frames = 64
low_altitude_rd_forced_fusion_weight = 0.60 到 0.75
low_altitude_rd_aux_ratio = 0.50
low_altitude_traj_aux_ratio = 0.50
low_altitude_traj_supervision_weight = 0.55
low_altitude_traj_anchor_weight = 0.45 到 0.55
low_altitude_rd_teacher_warmup_epochs = 2
```

若质量版仍不能让 joint-on-RD 超过 trajectory-on-RD，再启动一轮更重配置：

```text
low_altitude_deep_reroute_aux_interval = 1
low_altitude_deep_reroute_max_frames = 96
```

约束：

- 不回退到旧 RD logits correction 主路径。
- 不关闭 segment-store。
- 不为了速度继续降低 RD/joint 关键监督。

#### P0-C：Stage2 RD-aware batch 采样

新增 stage2 专用采样策略，避免 RD joint 监督被无 RD Excel 窗口稀释：

```text
stage1:
  使用全量 parent-balanced / class-balanced window sampler，专注 trajectory anchor。

stage2:
  每个 epoch 保留无 RD replay，防止 trajectory-only 退化。
  同时强制每个训练 batch 或 micro-batch 包含足量 RD 窗口。
```

建议策略：

```text
rd_batch_fraction = 0.50 到 0.60
without_rd_replay_fraction = 0.40 到 0.50
rare_class_boost 对 Unknown / Clutter 继续生效
parent_window_balance_power 保留，避免长轨迹窗口重复支配训练
```

实现优先级：

1. 先实现最小可控版本：训练 epoch 内分别维护 RD index pool 与 no-RD index pool，组 batch 时按比例混合。
2. 再升级为 source-label-parent aware sampler，保证 RD / no-RD、source、label、parent 四个维度都不过度倾斜。

验收：

```text
stage2 每轮报告 rd_window_fraction_in_train_batches
stage2 每轮报告 rare_class_fraction_in_train_batches
joint-on-RD macro F1 不再低于 trajectory-on-RD macro F1
without-RD accuracy 不低于 0.84
```

#### P0-D：弱类与窗口抖动治理

针对 Clutter / Unknown / Drone-Bird 的窗口级问题，下一轮必须新增或强化以下诊断：

- per-class window recall / precision / F1。
- parent-level per-class recall / precision / F1。
- prediction flip count per parent。
- first-window、mean-logit、stable-decision 三种聚合的混淆矩阵。
- Unknown / Clutter hard-pair 的 window-level 和 parent-level rescue / harm。

训练侧优先做：

- Unknown / Clutter 继续使用 class weight，但同步限制过强 Unknown bias，避免 Unknown 被 Bird / Drone 大类吸走。
- Drone / Bird pairwise loss 继续保留，但观察 Bird 召回和 Drone precision 的 tradeoff。
- 对 Clutter / Unknown parent 做 parent-level oversampling，而不是只放大单个窗口，避免同一父轨迹重复窗口造成过拟合。

暂不做：

- 不直接引入新的细类标签。
- 不把 raw_tr_rd 4 条 test 样本作为硬淘汰门槛。
- 不以单轮 window accuracy 作为唯一选模依据。

#### P0-E：RD 时间同步作为下一层精度风险

质量版训练完成后，如果仍出现：

```text
RD evidence macro F1 on RD >= 0.60
但 joint-on-RD <= trajectory-on-RD
```

则优先怀疑 RD segment 与轨迹窗口的时间证据不够精确。下一步需要：

- 在 RD manifest 中写入可追溯的原始帧时间戳或帧序号。
- `WindowAssembler` 从比例裁切升级为时间戳裁切。
- 训练报告区分 exact_time_sync、proportional_fallback、low_coverage、short_segment。
- 分别报告不同 time_sync_status 下的 joint gain。

该项不抢在 P0-A 到 P0-D 前做，因为当前首先要排除选模、监督强度和采样导致的 joint 退化。

### 下一轮验收门槛

下一轮质量候选必须同时满足：

```text
window final macro F1 >= 0.7800
window final macro F1 >= trajectory macro F1 - 0.0030
joint-on-RD macro F1 >= trajectory-on-RD macro F1 + 0.0150
RD evidence macro F1 on RD >= 0.6200
final net gain rate on RD >= 0.0150
final harm rate on RD <= 0.0500
parent mean-logit macro F1 >= 0.8200
Clutter recall >= 0.5000
Unknown recall >= 0.5500
without-RD accuracy >= 0.8400
```

若只满足 parent-level 指标但窗口级 joint gain 不达标，该模型只能作为 track-level aggregation 诊断候选，不能作为 Expert1 V2 主候选。

## P0.1：Expert1 RD 流式分片缓存与窗口聚合（已落地支撑项）

目标：把 Expert1 的 RD 处理从“每个 10 秒窗口重新读取、裁切、构造大 patch、送入模型”改成“2 秒 RD segment 只编码一次，缓存紧凑 embedding，10 秒窗口复用最近 5 个 segment 聚合后再做 TR+RD 融合判断”。

该项已经从最高优先级转为当前精度恢复工作的工程支撑项；后续只继续补齐未完成的时间同步、消融和异步调度。

### 背景判断

当前窗口配置为：

```text
window_seconds = 10
stride_seconds = 2
```

相邻窗口存在约 80% 时间重叠。若每个窗口都重新处理完整 RD，RD 读取、预处理、deep patch 构造和 CPU->GPU 搬运会被重复放大，导致 GPU 周期性等待数据，吞吐和时延都被 RD 数据管线限制。

当前优化判断：

- TR-only 路径已经足够快，暂不作为主要瓶颈。
- TR+RD 路径的瓶颈优先怀疑在 RD IO、CPU 预处理、大张量搬运和重复窗口计算。
- 继续堆叠 cross-modal 模型复杂度会进一步加重 RD 路径负担，当前阶段不优先。
- RD 优化必须同时服务精度和部署，不允许只为速度牺牲 RD 与窗口时间范围的一致性。

### 目标架构

Expert1 保持二路径结构：

```text
无 RD：
trajectory window -> trajectory encoder -> TR-only classifier

有 RD：
trajectory window -> trajectory encoder
RD stream -> 2s segment -> RD segment encoder -> segment embedding cache
最近 5 个 RD segment embedding -> RD window aggregation
trajectory embedding + RD window embedding -> TR+RD joint classifier
```

RD 分支核心变化：

```text
旧逻辑：10s window -> 读取完整 RD -> 构造 rd_features / rd_patch / rd_deep_patch -> joint
新逻辑：2s RD segment -> 编码一次 -> 缓存 compact embedding -> 10s window 聚合最近 5 个 embedding -> joint
```

### 数据与缓存设计

缓存粒度从完整窗口下沉到 RD 时间片：

```text
cache_key = source_name + parent_sample_id/track_id + segment_start + segment_end + rd_preprocess_version
```

缓存内容优先保存紧凑表征，不优先保存巨大 raw patch：

```text
rd_segment_embedding
rd_quality_flags
rd_time_range
rd_valid_mask
rd_source_stats
```

部署侧每个目标维护环形缓存：

```text
TrackRingBuffer:
  trajectory_points: 最近 10s 到 20s
  rd_segment_embeddings: 最近 5 到 10 个 segment
  rd_quality_state
```

每次 stride 到达时只处理新增 2 秒 RD segment，旧的 8 秒 RD evidence 直接复用缓存。

### RD 时间对齐要求

RD 不允许继续无条件继承父样本完整 tensor 作为最终方案。窗口化训练和部署必须逐步切到严格时间对齐：

```text
trajectory window = [window_start, window_end]
RD evidence 只能来自 [window_start, window_end] 内的 segment
```

若某个窗口内 RD 不完整，需要显式输出：

```text
rd_valid_mask
rd_coverage_ratio
rd_quality_flags
```

不允许让 joint 分支隐式看到窗口外 RD 信息，否则线下指标会高估，线上部署会回落。

### deep patch 搬运优化

`rd_deep_patch` 不再作为每个 10 秒窗口反复搬运的常规输入。目标做法：

- deep patch 只进入 RD segment encoder。
- 每个 2 秒 segment 编码一次。
- 编码后只保留 compact segment embedding。
- joint window 阶段只聚合 embedding，不再搬运完整 deep patch。

必须做消融验证：

```text
full deep patch
降采样 deep patch
无 deep patch，仅 rd_features / rd_patch
segment embedding cache
```

如果 deep patch 对 joint-on-RD macro F1、Drone/Bird recall、Unknown/Clutter recall 的收益不足以抵消吞吐和时延成本，则默认降采样或移除。

### 异步预取与批量化

训练和部署都要按流水线组织：

```text
数据读取 / RD 解码
  -> RD segment 预处理
  -> segment encoder / cache
  -> window aggregation
  -> TR-only batch 或 TR+RD micro-batch
  -> Expert1 输出
  -> parent / track 聚合
```

批量策略：

- TR-only 和 TR+RD 分开组 batch。
- TR-only 允许较大 batch，保持低时延。
- TR+RD 使用 micro-batch，优先验证 batch size 4 / 8 / 16。
- 部署时允许 20ms 到 50ms 短暂聚合窗口，兼顾吞吐和 P95/P99 延迟。
- 预取队列必须避免 GPU 长时间等待 CPU/RD IO。

### 训练策略

训练口径必须贴近部署：

```text
trajectory window = 10s
stride = 2s
RD segment = 2s
RD window evidence = 最近 5 个 segment embedding
```

无 RD 样本：

```text
只训练 trajectory-only path
final_logits = traj_logits
```

有 RD 样本：

```text
训练 TR+RD joint path
final_logits = joint_logits + bounded residual
```

训练中必须继续保留：

- trajectory-only head，防止无 RD 路径退化。
- TR+RD joint head，验证 RD 是否产生正收益。
- RD auxiliary diagnostics，只作为 evidence 质量诊断，不重新变成 RD 接管主路径。

### 评估指标

新增性能和质量联动指标：

```text
rd_segment_cache_hit_rate
rd_window_recompute_ratio
rd_preprocess_ms_p50 / p95 / p99
rd_h2d_transfer_ms_p50 / p95 / p99
tr_only_latency_ms_p50 / p95 / p99
tr_rd_latency_ms_p50 / p95 / p99
tr_only_throughput_samples_s
tr_rd_throughput_samples_s
gpu_utilization_mean
gpu_idle_wait_ratio
```

模型指标仍必须同时保留：

```text
trajectory macro F1
joint-on-RD macro F1
joint - trajectory macro F1 on RD
final net gain on RD
Drone/Bird recall
Unknown/Clutter recall
without-RD accuracy
source-balanced macro F1
parent_track_mean_logit_macro_f1
stable_decision_latency_seconds
```

### 验收

```text
RD segment cache 可用，重复窗口不再完整重算 RD
10s 窗口由最近 5 个 2s RD segment embedding 聚合产生
RD evidence 严格限制在 window_start 到 window_end 内
TR-only 与 TR+RD 分离 batch，互不拖慢
TR+RD branch 在 GPU 上能稳定 micro-batch 推理
joint-on-RD macro F1 不低于当前稳定 full 候选
final net gain on RD >= 0
without-RD accuracy 不低于当前稳定 full 候选
端到端报告同时输出吞吐、P50/P95/P99 延迟、缓存命中率和窗口/父轨迹指标
```

### 当前落地状态

已完成首轮可训练改造：

- `TorchTrackDataset` 新增父级 RD 编码 LRU cache：
  - `rd_encoded_cache_size`
  - 同一 `rd_tensor_path` / `rd_summary_csv_path` 只做一次 `np.load + encode_*`，窗口样本复用编码结果。
- `TorchTrackDataset` 新增窗口 RD 时间裁切：
  - `enable_window_rd_time_crop`
  - `rd_window_crop_min_frames`
  - 对 `expert1_window_index.csv` 中的 trainable 窗口，按 `parent_sample_id` 的窗口时间范围估计父轨迹比例，只取当前窗口对应 RD 片段。
- `LowAltitudeSuspectExpert` 新增 joint RD segment token 聚合：
  - `rd_joint_segment_count = 5`
  - `enable_rd_segment_joint_tokens = true`
  - RD backbone / RD auxiliary 仍使用当前窗口 RD；TR+RD cross-attention 前将 RD token 压成最多 5 个 segment token，对齐 `10s = 5 x 2s` 的部署目标。
- `tmp/expert1_v2_window_stable_full_config.json` 已打开上述开关。
- Expert1 训练报告新增 `rd_pipeline` 字段，记录本轮 RD 管线配置。
- System 构造和 System 配置已接入 RD segment joint token 参数，避免 Expert1 单训与 System 接回语义不一致。
- 新增 `src/process/build_expert1_rd_segment_store.py`：
  - 支持按 parent track 去重构建 2 秒 RD segment store。
  - 正式输出路径为 `data/processed/expert1_rd_segment_store/`、`data/processed/expert1_rd_segment_manifest.csv`、`data/processed/expert1_rd_segment_summary.json`。
  - segment 保存 `rd_features`、`rd_patch_features`、`rd_deep_patch_features`、valid length、time range、coverage、quality flags、time sync status。
  - 当前 time sync status 先显式标记为 `proportional_fallback`，不伪装成精确同步。
- 新增 `src/core/data/rd_segment_store.py`：
  - `WindowAssembler` 可按窗口时间范围从 RD segment manifest 选择窗口内 segment。
  - 支持读取 window-to-segment map，按 parent-level `.pt` 组装当前模型需要的 `EncodedRDSequence`。
  - 输出窗口 coverage、segment ids、quality flags、time sync status counts，为后续报告指标提供来源。
- 新增 `src/process/build_expert1_window_segment_map.py`：
  - 可生成 `expert1_window_segment_map.csv`，缓存 window -> segment 映射，不重复保存窗口 RD 大 tensor。
- Expert1 DataLoader 新增 segment-store 可选模式：
  - `use_rd_segment_store`
  - `rd_segment_manifest_name`
  - `window_segment_map_name`
  - `rd_segment_store_cache_size`
  - 正式训练默认不允许静默回退；segment store 或映射缺失时直接报错，避免误以为启用了预处理但实际走慢路径。
- 已全量构建正式 RD segment store：
  - `data/processed/expert1_rd_segment_store/`
  - `data/processed/expert1_rd_segment_manifest.csv`
  - `data/processed/expert1_rd_segment_summary.json`
  - parent_count = 1228
  - segment_count = 9994
  - failed_parent_count = 0
  - store size 约 18.68GB
- 已全量生成正式 window-to-segment map：
  - `data/processed/expert1_window_segment_map.csv`
  - mapped_window_count = 5798
  - missing_window_count = 0
  - segment_count_mean = 4.6342
  - rd_coverage_ratio_mean = 1.0
- `tmp/expert1_v2_window_stable_full_config.json` 已切换到 `use_rd_segment_store=true`。

首轮验证：

```text
compileall 通过
窗口 dataloader 可正常 collate
裁切后部分 CQ08 RD 窗口从完整 128 / 160 帧降至几十帧
Expert1 小 batch 前向通过，logits / joint_logits 输出正常
RDSegmentStore smoke 通过：3 个 CQ08 parent 生成 103 个 2s segment，manifest / parent-level .pt 可正常读取
WindowAssembler smoke 通过：基于 window smoke segment store 生成 window-segment map，segment-store DataLoader 和 collate 可正常输出 RD batch
新增 unittest smoke 通过：覆盖 RD segment store 构建、overwrite 清理过期 shard、window-segment map、segment-store Dataset、collate，以及 CSV 字符串 False 不误激活多模态 RD
```

训练加速已落地：

- class weights、RD class weights、RD coverage 已改为 index 级统计，不再触发真实 DataLoader 扫描和 RD 加载。
- source normalization stats 已改为只读取轨迹 CSV，并缓存到 `data/processed/<index_name>_source_feature_stats.json`；后续同一 index 直接复用缓存。
- DataLoader 新增 `pin_memory`、`persistent_workers`、`prefetch_factor` 配置；CUDA tensor 搬运使用 `non_blocking_h2d`。
- full validation 新增 `full_val_warmup_epochs`，窗口训练配置改为 warmup 期 full val，之后每 3 epoch full val，其余走 quick val。
- `LowAltitudeSuspectExpert` 新增整 batch 无 RD 快路径：当 batch 内全部无 RD 时，直接走 trajectory-only 输出，不再运行 RD encoder / patch CNN / joint fusion。
- `LowAltitudeSuspectExpert` 在 `disable_tr_rd_fusion=true` 时直接走 trajectory-only 快路径，stage1 不再运行无意义 RD backbone / patch CNN / joint fusion。
- deep reroute auxiliary 新增抽样控制：
  - `low_altitude_deep_reroute_aux_interval`
  - `low_altitude_deep_reroute_max_frames`
  - 当前窗口训练配置为每 4 个 batch 运行一次 deep auxiliary，每次最多 32 帧。
- `TorchTrackDataset(use_rd_segment_store=true)` 默认不再静默回退；正式 manifest/map 缺失时直接报错，避免误以为启用了预处理但实际走慢路径。
- `tmp/expert1_v2_window_stable_full_config.json` 已打开 `num_workers=2`、`pin_memory=true`、`persistent_workers=true`、`prefetch_factor=2`、`non_blocking_h2d=true`，并将验证节奏改为 `full_val_interval_epochs=3`、`quick_val_max_batches=80`。

训练加速验证：

```text
compileall 通过
tests.test_expert1_rd_segment_store_smoke 通过：6 tests
1-batch speed smoke 可完整写出 summary/report
正式 segment-store 前向 smoke 通过：同一 batch 中 RD 样本 `tr_rd_active=True`，纯轨迹样本 `tr_rd_active=False`
stage2 降速 profiling 通过：修改后 stage2 batch 约 0.2s 到 0.35s，DataLoader get_batch 约 0.07s 到 0.20s
```

仍未完成：

- 还没有完整 BatchScheduler / micro-batch 调度器；DataLoader pinned memory 已完成，异步预取仍停留在 PyTorch DataLoader worker/prefetch 级别。
- 当前 RD 时间裁切是基于父轨迹窗口范围的比例映射，raw / CQ08 的 RD 原始时间轴仍需进一步做显式时间同步。
- deep patch 仍在 RD backbone / deep_reroute 路径中使用；后续必须继续做 full / downsample / remove 消融。

### 数据处理优化方案

下一阶段不采用“把所有窗口全部预处理成固定样本文件”的方案。最终数据层应迁移为：

```text
Segment 是存储单位
Window 是策略生成单位
Track 是聚合单位
Batch 是 GPU 执行单位
```

核心原则：

- 只预处理模型无关、窗口无关、可复用的 RD 基础 segment。
- 不预处理死固定 `10s window` / `20s window` 大样本，避免多尺度窗口和自适应 stride 后样本爆炸。
- 训练和部署复用同一个 `WindowAssembler` 逻辑，保证离线训练尽可能贴近真实数据流。
- RD encoder 输出 embedding 不作为首轮离线固定特征，除非后续明确冻结 RD encoder；否则会削弱 RD 表征学习能力。

目标数据流：

```text
离线训练：
raw TR/RD files
  -> TimeSynchronizer
  -> RDSegmentStore
  -> WindowStrategy
  -> WindowAssembler
  -> BatchScheduler
  -> GPU

真实部署：
TR/RD stream
  -> TrackRingBuffer
  -> TimeSynchronizer
  -> RDSegmentStore / MemorySegmentCache
  -> WindowStrategy
  -> WindowAssembler
  -> BatchScheduler
  -> GPU
```

#### RDSegmentStore

`RDSegmentStore` 负责把原始 RD 转成模型无关的 2 秒基础 segment。该层可以预处理到 `data/processed`，训练时直接加载。

建议输出：

```text
data/processed/expert1_rd_segment_store/
data/processed/expert1_rd_segment_manifest.csv
data/processed/expert1_rd_segment_summary.json
```

manifest 字段：

```text
segment_id
parent_sample_id
track_id
source_name
label_name
segment_start_seconds
segment_end_seconds
segment_duration_seconds
segment_frame_count
rd_feature_path / shard_key
rd_patch_path / shard_key
rd_deep_patch_path / shard_key
rd_coverage_ratio
rd_quality_flags
rd_time_sync_status
rd_preprocess_version
```

存储策略：

- 不建议每个窗口一个 RD 文件。
- 优先按 parent track 或 shard 存储 segment，manifest 记录 key。
- 对小数据可用 parent-level `.pt` 文件；对大数据改为 shard：

```text
expert1_rd_segments_shard_000.pt
expert1_rd_segments_shard_001.pt
...
```

每个 segment 保存模型无关张量：

```text
rd_features
rd_patch_features
rd_deep_patch_features 或 downsampled_deep_patch_features
valid_length
time_range
quality_flags
```

#### WindowAssembler

`WindowAssembler` 不提前落死窗口 RD 文件，而是根据窗口策略动态组装。

输入：

```text
expert1_window_index / multiscale_window_index
rd_segment_manifest
track csv / track ring buffer
window_strategy
```

输出：

```text
window trajectory tensor
window rd segment tensor list
rd_valid_mask
rd_coverage_ratio
window_metadata
```

窗口到 segment 的映射：

```text
4s / 6s window  -> 最近 2 到 3 个 2s RD segment
10s window      -> 最近 5 个 2s RD segment
20s window      -> 最近 10 个 2s RD segment
30s window      -> 最近 15 个 2s RD segment
```

该设计允许后续无成本迁移到 target 中的多尺度窗口策略：

```text
4s / 6s early window
10s main window
20s / 30s stable window
1s / 2s / 4s adaptive stride
```

#### TimeSynchronizer

当前首轮实现采用父轨迹时间比例裁切 RD，这是过渡方案。下一阶段必须补充显式时间同步。

同步规则：

```text
track time_seconds -> normalized parent time
rd time_raw -> normalized RD time
window_start/end -> RD segment_start/end
```

若 RD 原始时间轴无法可靠映射，必须显式标记：

```text
rd_time_sync_status = exact / normalized / proportional_fallback / unavailable
```

不同状态进入报告和采样权重，不能静默混用。

#### BatchScheduler

训练和部署都应从“DataLoader 单线程准备完整 batch”迁移为更接近部署的调度：

```text
TR-only queue
TR+RD queue
RD segment preload queue
GPU micro-batch queue
```

调度原则：

- TR-only 与 TR+RD 分离 batch，避免无 RD 快路径被 RD 拖慢。
- TR+RD 使用 micro-batch，优先验证 batch size 4 / 8 / 16。
- 训练阶段先支持普通 DataLoader 兼容模式；部署阶段再接入异步队列、pinned memory 和 non_blocking H2D。
- 高风险 / 低置信目标允许更短 stride，提高召回；稳定高置信目标放宽 stride，降低算力。

#### 不采用的方案

不建议：

```text
预处理每个 10s window 为独立 RD tensor
预处理每个多尺度 window 为独立 RD tensor
预处理固定 RD encoder embedding 后继续训练 RD encoder
为每种 stride 单独生成一套重复 RD 文件
```

原因：

- 多尺度窗口会造成窗口样本和磁盘文件指数式膨胀。
- 重叠窗口会重复保存 80% 以上 RD 内容。
- 自适应 stride 在部署时是动态决策，不能完全靠静态窗口文件表达。
- 固定 RD embedding 会让 RD encoder 失去端到端学习空间。

#### 分阶段实施

1. 全量构建正式 `RDSegmentStore` 产物，并确认失败 parent、segment 数量和 `rd_time_sync_status` 统计。
2. 保持 `expert1_window_index.csv` 不变，全量生成 window-to-segment 映射缓存：

```text
data/processed/expert1_window_segment_map.csv
```

3. 打开 Expert1 DataLoader 的 segment-store 读取模式，与当前动态 RD 读取模式对照验证。
4. 报告增加数据管线指标：

```text
segment_cache_hit_rate
window_segment_count_mean
rd_coverage_ratio_mean
rd_time_sync_status_counts
rd_load_ms_p50 / p95 / p99
collate_ms_p50 / p95 / p99
h2d_ms_p50 / p95 / p99
```

5. 用 `10s/2s` 主基线先验证精度不下降、吞吐提升。
6. 再新增 `expert1_multiscale_window_index.csv`，复用同一个 segment store，不重做 RD 基础预处理。
7. 最后接入异步预取和部署 RingBuffer。

验收：

```text
训练时不再从原始 RD 文件重复 encode 完整父 RD
10s/2s 主基线可从 RDSegmentStore 动态组窗
多尺度窗口可复用同一套 RD segment store
无 parent leakage
RD evidence 不越过 window_start/window_end
segment cache hit rate 可报告
RD load / collate / H2D 延迟可报告
joint-on-RD macro F1 不低于当前窗口基线
without-RD accuracy 不退化
```

### 实施顺序

1. 建立 RD segment 数据结构和 cache key，不破坏现有 `expert1_window_index.csv`。（已完成首轮内存缓存）
2. 为 CQ08 / raw RD 增加按窗口时间范围裁切或 segment 对齐能力。（已完成首轮比例裁切，仍需显式时间同步）
3. 全量构建正式 `RDSegmentStore` 产物，输出正式 manifest / summary。（已完成：1228 parent / 9994 segment / 约 18.68GB）
4. 全量生成 `expert1_window_segment_map.csv`，缓存 window -> segment 映射，但不重复保存窗口 RD 大 tensor。（已完成：5798/5798 mapped）
5. 打开 `use_rd_segment_store`，用 10s/2s 主基线验证 segment-store 模式精度不下降、吞吐提升。（已完成 smoke；待完整训练验证精度与吞吐）
6. 增加训练报告中的 cache / latency / recompute 指标。（部分完成：记录 rd_pipeline 配置，未输出 cache hit / latency）
7. 做 deep patch 消融，决定 full / downsample / remove。
8. 再启动多尺度窗口索引，复用同一套 RDSegmentStore。
9. 做 CUDA micro-batch / 异步预取验证，再进入 System 接回。

## P0.2：基线保护

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

### 下一轮窗口策略备选方案

当前 `10s window / 2s stride` 继续作为真实部署主基线，不直接替换。下一轮若窗口训练后出现 Bird / Unknown / Clutter 召回不足、早期识别慢、或窗口级抖动明显，再启动多尺度窗口备选方案。

备选目标：

```text
短窗负责早期预警
主窗负责主要分类
长窗负责轨迹级稳定确认
stride 根据目标不确定性和 RD 到达节奏自适应
```

建议实验矩阵：

| 窗口 | 用途 | 建议 stride | 主要观察指标 |
|---|---|---:|---|
| 4s / 6s | 早期预警、低时延召回 | 1s / 2s | early_window_accuracy、Drone/Bird recall、Unknown recall |
| 10s | 主分类基线 | 2s | window macro F1、joint-on-RD macro F1、without-RD accuracy |
| 20s / 30s | 稳定确认、降低误报 | 4s / 5s | parent macro F1、stable decision latency、false alarm rate |

推荐部署式流程：

```text
track_id 数据流
  -> 轨迹/RD 环形缓存
  -> 4s/6s 短窗早期判断
  -> 10s 主窗分类
  -> 20s/30s 长窗稳定确认
  -> window logits 连续聚合
  -> track-level stable decision
```

自适应 stride 规则：

```text
普通目标：2s stride
高置信稳定目标：4s 或更长 stride，降低算力
低置信 / 高风险 pair：1s stride，提高召回
无 RD 目标：2s 到 4s stride
有 RD 目标：优先按 RD segment 到达节奏触发
```

评估必须同时输出：

```text
window-level macro F1 / macro recall
track-level mean-logit macro F1
first-stable-decision latency
early decision accuracy
prediction flip count per track
Drone/Bird recall
Unknown/Clutter recall
false alarm rate
RD compute cost per track
```

启动条件：

```text
10s/2s 主基线训练后，若 Bird / Unknown / Clutter recall 不足
或 first-window / stable-decision latency 不能满足部署需求
或窗口预测抖动导致 track-level 聚合收益不足
则启动多尺度窗口实验。
```

约束：

- 不直接删除当前 `expert1_window_index.csv`。
- 多尺度窗口必须新增索引文件，例如：

```text
data/processed/expert1_multiscale_window_index.csv
data/manifests/expert1_multiscale_window_split_manifest.json
data/processed/expert1_multiscale_window_summary.json
```

- 多尺度切分仍必须按 `source_name|parent_sample_id` 分组，禁止 parent leakage。
- RD 仍必须按窗口时间范围裁切，不能回退到完整父 RD 继承。
- 多尺度报告必须分开统计 4s/6s、10s、20s/30s，不允许只汇总一个 overall accuracy。

仍需保留为下一步：

- 有 RD 的 CQ08 / raw 窗口已完成首轮比例裁切，但尚未做显式 RD 原始时间轴同步。
- 当前 10 秒窗在 CQ08 / Excel 上常见只有 5 个轨迹点，训练后必须重点观察 Bird / Unknown / Clutter recall；若召回不足，再启动上述多尺度窗口备选方案。
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
