# RadarInsight 当前问题核实清单

当前阶段：训练期。目标优先级是最高精度、强鲁棒性、强泛化、低空安全。部署吞吐和边缘推理成本暂不作为主要约束。

## 0. 最新结论

- System 当前最高可见结果：accuracy `0.8645`，macro F1 `0.6026`，route macro recall `0.7460`。
- Expert1 已明显增强，但 RD 仍不是强判别分支：`rd_macro_f1_on_rd=0.5308`。
- Expert2 已从漏检转为偏激进：Aircraft recall `0.8000`，precision `0.2105`。
- System 中 Aircraft 保护能把 recall 拉到 `1.0000`，但 precision 仍只有 `0.4545`，non-Aircraft as Aircraft rate `0.0183`。
- source 泛化是当前最大短板之一：System by_source accuracy 为 raw_tr_rd `0.4545`、excel_track `0.7159`、cq08_track `0.9399`。

## 1. P0 必须先止血

### P0.1 reroute 仍是后处理式路径

状态：部分已修，核心仍成立。

- 已修复：
  - reroute 选中专家后不再复制 `soft_log_weighted` 的 gated base logits。
  - Markdown 报告已显示 reroute 指标。
  - prediction records 已拆分 Router 初始路由和最终回流路由。
- 仍存在：
  - `RadarClassificationSystem.forward()` 仍一次性计算所有专家。
  - `_project_to_global_logits_reroute()` 仍是在已算好的专家输出上改 route 和投影，不是真正的“Router -> 主专家 -> 低置信 -> 回流 -> 重新推理”。
  - `high_accept` 计算后未用于 high_altitude_filter 输出裁决，high route 有 Aircraft 假阳性风险。
  - `eval_fusion_candidate_modes` 默认仍包含 `reroute`，虽然 configured 模式不会选它，但 probe 仍会运行。

修复方向：

1. 训练期和默认评估候选中移除 `reroute`。
2. high_altitude_filter 只有在规则/高空专家明确 positive 时才允许写 Aircraft。
3. 真正的 reroute 后续作为独立推理路径重写，不再混入训练期 fusion probe。

### P0.2 System 全局 logits 仍混用不可比分数

状态：成立。

- 低空专家、Expert2、Expert4、高空规则专家输出的分数语义不同。
- 当前 `_project_to_global_logits()` 仍将这些分数拼进同一全局 logits 空间竞争。
- 这会同时造成两类相反问题：
  - Expert2 正确识别 Aircraft 时可能被低空/地面压制。
  - Expert2 偏激进时又可能通过保护逻辑放大 Aircraft 假阳性。

修复方向：

1. 短期：加强 selection score 硬约束，避免选到 Aircraft FP 或 source 崩坏模型。
2. 中期：改为“专家候选 + calibrated confidence 仲裁”，不再直接比较 raw logits。

### P0.3 Expert2 Aircraft 假阳性偏高

状态：成立。

- 最新 Expert2：Aircraft precision `0.2105`，recall `0.8000`。
- raw_tr_rd 源域上 NonAircraft / Aircraft 相似度几乎不可分。
- 当前 Expert2 threshold selection 仍偏向 macro F1 和 recall，没有 precision / FP rate 硬门槛。

修复方向：

1. Expert2 选模加入 Aircraft precision 下限和 FP rate 上限。
2. 增加 source-aware threshold / calibration，尤其 raw_tr_rd。
3. System Aircraft 保护增加 route 不强烈反对、低空/地面未强命中等二次条件。

### P0.4 System 选模对 FP 和 source 泛化约束过弱

状态：成立。

- `_system_selection_score()` 当前仍奖励 Aircraft recall/F1。
- `non_aircraft_as_aircraft_rate` 惩罚系数太小。
- System by_source 目前只报告 accuracy，没有 source balanced macro F1、min source accuracy、per-source Aircraft FP。

修复方向：

1. selection score 纳入：
  - `aircraft_system_precision`
  - `non_aircraft_as_aircraft_rate`
  - `min_source_accuracy`
  - `source_balanced_accuracy`
2. 对 Aircraft precision 过低、FP rate 过高、min source accuracy 过低的模型施加强惩罚。

## 2. P1 下一批问题

### P1.1 Expert1 RD 分支仍弱

- final macro F1 `0.7518`
- RD on-RD macro F1 `0.5308`
- RD fusion weight on-RD mean `0.2451`

方向：继续 RD branch 单训、Drone/Bird 与 Unknown/Clutter hard-negative、融合门控监督。

### P1.2 Expert4 groundness 在 System 中没有稳定发挥

- Expert4 groundness 有监督，但 System 配置中 `groundness_protection_enabled=false`。
- 最新 System groundness true/non-ground 均值差异仍弱。

方向：先做 threshold sweep，再开启 System groundness protection。

### P1.3 Router 跨源不稳

- cq08_track route 很好，raw_tr_rd 和 excel_track 明显弱。
- 说明 Router 问题不是单纯容量，而是 source feature distribution 不一致。

方向：source-balanced route recall、source normalization 校准、增加时序统计特征。

## 3. 当前执行优先级

1. 止血：
  - 默认候选移除 `reroute`
  - high_altitude_filter 使用 high_accept gate
  - selection score 加强 Aircraft FP / source 约束
  - source report 增加 min/balanced 指标
2. 下一步：
  - System 仲裁从 raw logits 拼接改成专家候选 + calibrated confidence
  - Expert2 source-aware threshold
  - Expert1 RD 继续增强
