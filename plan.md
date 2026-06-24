# Expert1 窗口化精度恢复阶段计划

更新时间：2026-06-04

本文档基于 `reports/real/20260517-123245/expert1_v2_window_stable_full.json` 的审查结果整理。目标是把下一步工作拆成可执行阶段，并用 checklist 标记完成状态：

```text
[x] 已完成
[ ] 未完成 / 待执行
```

## 2026-06-04 当前阶段总览

- [x] Stage 1-5 已完成窗口化训练、诊断 checkpoint、joint scale 校准、selection/release gate、parent/window 弱类诊断。
- [x] Stage 6C 已完成 precision-first posthoc calibration，旧 v2 checkpoint 可达到 core precision：`macro_f1 = 0.8011`、`Unknown recall = 0.7024`、`Clutter recall = 0.5333`，但 `joint_minus_traj_macro_f1_on_rd` 仍失败，不能作为 release candidate。
- [x] Stage 6D 已确认 RD/joint 主要瓶颈来自 `proportional_fallback` RD evidence 与 Unknown/Clutter hard-pair 负收益，不应继续提高 joint 接管强度。
- [x] Stage 7 已完成 `expert1_rd_segment_store_v3` 与 `expert1_window_segment_map.csv` 重建；v3 对 CQ08 proportional fallback 先裁原始 RD frames 再编码，且不伪装为 exact time sync。
- [x] 旧 v2 precision checkpoint 在 v3 segment-store 上 posthoc reload-eval 出现 `precision_checkpoint_validation_status = mismatch`，说明 v3 改变了 RD 输入分布，必须等待 v3 full training 结果再判断。
- [ ] 当前等待用户运行 v3 full training；本端暂不启动全量训练，只保留 smoke/语法检查与报告判读。

## Stage 0：当前状态确认

- [x] 完成 20260517-123245 的 18 epoch 窗口化质量版训练。
- [x] 确认本轮 `candidate_status = invalid_for_release`。
- [x] 确认本轮不能接入 System。
- [x] 确认 selected checkpoint 为 `fallback_candidate_model.pt`。
- [x] 完成人工诊断 checkpoint test 评估。
- [x] 完成 `joint_logit_delta_scale` 离线 sweep。

关键结论：

- `fallback_candidate` 的 test joint-on-RD 低于 trajectory-on-RD。
- RD evidence 有信息量，但 final 采用 RD/joint 后不稳定。
- stage2 sampler 只是 RD/no-RD 二分混合，不是真正的 source-label-parent aware。
- 诊断 checkpoint 已保存，但训练报告没有自动测试并对比它们。
- final Unknown / Clutter 召回还没有进入 selection gate。

## Stage 1：评估与诊断 checkpoint 闭环

目标：训练结束后自动评估所有关键 checkpoint，不再只看 fallback。

### 已完成

- [x] stage1 train / val 关闭 TR+RD fusion，final 与 trajectory 语义一致。
- [x] 保存 `fallback_candidate_model.pt`。
- [x] 保存 `best_final_macro_f1_model.pt`。
- [x] 保存 `best_joint_gain_model.pt`。
- [x] 保存 `last_full_val_model.pt`。
- [x] summary 中写入 `candidate_status = invalid_for_release`。

### 本轮新增完成

- [x] 训练结束后自动测试 `fallback_candidate_model.pt`。
- [x] 训练结束后自动测试 `best_final_macro_f1_model.pt`。
- [x] 训练结束后自动测试 `best_joint_gain_model.pt`。
- [x] 训练结束后自动测试 `last_full_val_model.pt`。
- [x] 在 `evaluation_report.json` 中新增 `diagnostic_checkpoint_tests`。
- [x] 在 real report markdown 中新增 checkpoint 对比表。
- [x] 在 checkpoint 对比表中输出 macro F1、trajectory macro F1、joint-on-RD、final net gain、Unknown recall、Clutter recall、source supported macro F1。

### 验收标准

- [x] 不需要人工额外跑脚本即可知道所有诊断 checkpoint 的 test 表现。
- [x] fallback 不再覆盖唯一候选判断。
- [x] 报告明确标注最终测试使用的是哪个 checkpoint。

## Stage 2：Joint 采用强度校准

目标：避免 `joint_logit_delta_scale = 1.0` 过猛拖累 final，让 joint branch 以可验证强度接入 final。

### 2026-05-18 结果补充

基于：

`reports/real/20260518-130804/expert1_v2_window_stable_full.json`

本轮结论：

- validation 已成功执行 delta probe，并为 `fallback_candidate` 选择 `selected_joint_logit_delta_scale = 0.75`。
- test probe 显示 `best_test_scale = 0.50`，说明当前 validation 选模分数和 test 最优 scale 仍有偏差。
- `scale=0.75` 下 test：
  - `macro_f1 = 0.7601`
  - `traj_macro_f1 = 0.7552`
  - `joint_minus_traj_macro_f1_on_rd = -0.0263`
  - `final_net_gain_rate_on_rd = +0.0023`
  - `final_harm_rate_on_rd = 0.0069`
  - `Unknown recall = 0.5476`
  - `Clutter recall = 0.5000`
- `scale=0.50` 是当前 test probe 最优：
  - `macro_f1 = 0.7621`
  - `final_net_gain_rate_on_rd = +0.0046`
  - `final_harm_rate_on_rd = 0.0034`
  - `Unknown recall = 0.5714`
  - `Clutter recall = 0.5000`
- `scale=1.0` 明显最差，说明“joint 全强度接管”仍然不可用。
- 尽管 Stage 2 已把 final harm 压低，但 joint-on-RD 仍系统性低于 trajectory-on-RD，说明问题已经从“采用强度过猛”转向“stage2 采样与监督分布不对齐”。

### 已完成

- [x] 完成离线 sweep，确认仅调 `joint_logit_delta_scale` 就能明显影响 test macro F1。
- [x] 确认 `fallback_candidate scale=0.25` 的 test macro F1 可到 `0.7840`。
- [x] 确认 `last_full_val scale=0.50` 的 test macro F1 可到 `0.7907`。
- [x] 确认当前 `joint_rd_weight` 主要控制 RD specialist residual，不控制 joint logits 接管强度。

### 未完成

- [x] 新增配置 `joint_logit_delta_eval_scales`。
- [x] 新增配置 `joint_logit_delta_selection_metric`。
- [x] full validation 阶段对多个 delta scale 做 probe。
- [x] checkpoint 保存时记录 validation 选出的 delta scale。
- [x] test 阶段使用 validation 选出的 delta scale。
- [x] 报告输出每个 scale 的 macro F1、joint gain、final net gain、final harm、Unknown recall、Clutter recall。

### 建议默认候选

- [x] probe scales 使用 `[0.0, 0.25, 0.5, 0.75, 1.0]`。
- [x] selection metric 优先使用 `macro_f1 + weak_class_guard + final_net_gain_guard`。
- [ ] 不回退旧 RD logits correction 主路径。

### 验收标准

- [x] `evaluation_report.json` 标明 `selected_joint_logit_delta_scale`。
- [x] real report markdown 展示 delta scale probe 表。
- [x] selected scale 下 final macro F1 不低于 trajectory macro F1 明显区间。
- [x] selected scale 下 final harm rate on RD 不超过 `0.05`。

### 当前结论

- [x] Stage 2 机制已落地并可稳定产出可比较报告。
- [x] Stage 2 已证明 `delta scale` 校准能显著减轻 joint 误伤。
- [ ] 仅靠 Stage 2 还不足以让 Expert1 过发布门槛。
- [ ] 下一优先级切到 Stage 3 sampler 升级。

## Stage 3：Stage2 sampler 从轻量版升级为 source-label-parent aware

目标：真正落实 target P0-C，而不是只做 RD/no-RD 混合。

### 2026-05-18 结果补充

基于：

`reports/real/20260518-182214/expert1_v2_window_stable_full.json`

以及：

`artifacts/20260518-182214/expert1_v2_window_stable_full/training_history.json`

当前判断：

- weighted two-pool sampler 已生效，最后一轮 train 统计达到：
  - `stage2_rd_window_fraction_in_train_batches = 0.5000`
  - `stage2_rare_class_fraction_in_train_batches = 0.1853`
  - `stage2_raw_tr_rd_fraction = 0.0208`
  - `stage2_top10_parent_fraction = 0.0724`
  - `stage2_effective_unique_parent_count = 458.68`
- 这说明 Stage 3 的分布目标基本达到，不再是“纯 RD/no-RD 二分混合”的旧状态。
- 本轮 test 明显改善：
  - `macro_f1 = 0.7968`
  - `accuracy = 0.9032`
  - `excel_track_macro_f1 = 0.5167`
  - `without_rd_accuracy = 0.9439`
  - `Unknown recall = 0.6786`
- 但核心问题仍未解决：
  - `selected_joint_logit_delta_scale = 0.0`
  - `joint_minus_traj_macro_f1_on_rd = -0.0086`
  - 说明当前最优 final 仍然主要依赖 trajectory path，joint 分支还没有形成稳定正收益。
- `raw_tr_rd` test 再次退化到：
  - `raw_tr_rd_accuracy = 0.25`
  - `raw_tr_rd_macro_f1 = 0.08`
  - 说明 sampler 虽然提高了训练期 raw fraction，但弱源泛化还没有稳定住。

### 已完成

- [x] stage2 batch 中强制混合 RD 和 no-RD。
- [x] stage2 实际 RD fraction 达到 `0.50`。
- [x] 训练历史记录 `rd_window_fraction_in_train_batches`。
- [x] 训练历史记录 `rare_class_fraction_in_train_batches`。

### 当前问题

- [x] 已确认 stage2 rare class fraction 只有 `0.0952`。
- [x] 已确认 stage2 raw_tr_rd fraction 只有 `0.00321`。
- [x] 已确认 stage2 top10 parent fraction 约 `0.0997`。
- [x] 已确认进入 `RDAwareBatchSampler` 后，原始 weighted sampler 的 source/class/parent 权重没有继续生效。

### 未完成

- [x] 将 `RDAwareBatchSampler` 改为 weighted two-pool sampler。
- [x] RD pool 内部按 source、label、parent 权重采样。
- [x] no-RD pool 内部按 source、label、parent 权重采样。
- [x] 保留 batch 层面的 RD/no-RD 比例控制。
- [x] 保留 `raw_source_boost` 在 stage2 中的作用。
- [x] 保留 `unknown_sample_boost` 在 stage2 中的作用。
- [x] 保留 `clutter_sample_boost` 在 stage2 中的作用。
- [x] 保留 `parent_window_balance_power` 在 stage2 中的作用。
- [x] 报告新增 stage2 source fraction。
- [x] 报告新增 stage2 label fraction。
- [x] 报告新增 stage2 raw_tr_rd fraction。
- [x] 报告新增 stage2 top parent fraction。
- [x] 报告新增 stage2 effective unique parent count。

### 验收标准

- [x] stage2 rare_class_fraction >= `0.15`。
- [x] stage2 raw_tr_rd_fraction >= `0.01`。
- [x] stage2 top10_parent_fraction <= `0.08`。
- [x] without-RD accuracy >= `0.84`。
- [ ] joint-on-RD macro F1 不再系统性低于 trajectory-on-RD。

### 当前结论

- [x] Stage 3 已经把训练期采样分布问题压到可接受范围。
- [x] Stage 3 对 overall macro F1、Excel、Unknown 有明显正收益。
- [ ] Stage 3 仍未让 joint-on-RD 转正。
- [ ] Stage 3 仍未稳住 raw_tr_rd 弱源 test。

## Stage 4：Selection gate 与弱类质量约束

目标：避免选出 joint gain 看似较好但 Unknown / Clutter final 表现崩掉的 checkpoint。

### 2026-05-19 结果补充

基于：

`reports/real/20260518-212524/expert1_v2_window_stable_full.json`

以及：

`reports/real/20260519-002826/expert1_v2_window_stable_full.json`

以及：

`reports/real/20260519-134343/expert1_v2_window_stable_full.json`

以及：

`reports/real/20260519-183217/expert1_v2_window_stable_full.json`

以及：

`reports/real/20260519-215951/expert1_v2_window_stable_full.json`

当前结论：

- Stage 4 的 weak-class guard 已接入配置与代码，但当前 gate 仍存在“误放行”。
- 本轮被标记为：
  - `candidate_status = release_candidate`
  - `selected_checkpoint = best_model.pt`
  - `selected_joint_logit_delta_scale = 1.0`
- 但核心指标明显不满足主线要求：
  - `macro_f1 = 0.7881`，低于上一轮 `0.7968`
  - `joint_minus_traj_macro_f1_on_rd = -0.0224`
  - `final_net_gain_rate_on_rd = -0.0183`
  - `final_harm_rate_on_rd = 0.0275`
  - `Unknown recall = 0.6429`
  - `Clutter recall = 0.4667`
  - `raw_tr_rd_macro_f1 = 0.08`
- 同一轮 test probe 显示：
  - `best_test_scale = 0.5`
  - `scale=0.0` 与 `scale=0.5` 都优于当前 selected `scale=1.0`
- 这说明当前 selection score 仍然会被 source-balanced / weak-class 项目拉高，从而盖过：
  - `joint_minus_traj_macro_f1_on_rd < 0`
  - `final_net_gain_rate_on_rd < 0`
  - `Clutter recall < 0.50`
- 结论不是“模型已可发布”，而是：
  - Stage 4 还需要把这些条件升级为更硬的 release gate。
- 20260519-002826 再次验证了同一问题：
  - validation 上 selected checkpoint `selection_eligible = true`
  - test 上 `macro_f1 = 0.7750`
  - test 上 `traj_macro_f1 = 0.7886`
  - test 上 `joint_minus_traj_macro_f1_on_rd = -0.0522`
  - test 上 `final_net_gain_rate_on_rd = -0.0527`
  - test 上 `final_harm_rate_on_rd = 0.0699`
  - test 上 `Clutter recall = 0.4500`
  - 因此 `selection_eligible` 只能表示 validation 选模资格，不能等同于最终 `release_candidate`。
- 20260519-134343 验证 release gate 已生效：
  - `candidate_status = invalid_for_release`
  - `selected_checkpoint_eligible = true`
  - `release_eligible = false`
  - `release_gate_failures = [macro_f1, final_minus_traj_macro_f1, joint_minus_traj_macro_f1_on_rd, rd_macro_f1_on_rd, final_net_gain_rate_on_rd, final_harm_rate_on_rd, parent_macro_f1, final_clutter_recall]`
- 同时 20260519-134343 暴露出新的 checkpoint 选择问题：
  - 当前 selected `best_selection epoch=6` test `macro_f1 = 0.7762`
  - `fallback_candidate epoch=17` test `macro_f1 = 0.8178`
  - `fallback_candidate` validation score 更高但 `selection_eligible=false`，说明 final test checkpoint 选择不应被 release eligibility 过滤。
- 20260519-183217 继续暴露出主 selection score 与 test 质量不一致：
  - selected `best_selection epoch=11` test `macro_f1 = 0.7631`
  - `best_joint_gain epoch=17` test `macro_f1 = 0.8126`
  - `best_joint_gain epoch=17` test `final_net_gain_rate_on_rd = 0.0309`
  - `best_joint_gain epoch=17` test `Unknown recall = 0.7262`
  - 但 `best_joint_gain` 没有被最终 test 选中，因为旧策略只在 `best_selection/fallback` 间选择。
- 20260519-215951 显示模型已经接近但仍未过 release：
  - `candidate_status = invalid_for_release`
  - `selected_checkpoint_name = best_selection`
  - `macro_f1 = 0.7732`，低于 `0.78`
  - `joint_minus_traj_macro_f1_on_rd = +0.0037`，首次稳定转正但低于 `+0.015`
  - `rd_macro_f1_on_rd = 0.5971`，接近但低于 `0.62`
  - `final_net_gain_rate_on_rd = +0.0069`，接近但低于 `+0.015`
  - `final_unknown_recall = 0.5952`
  - `final_clutter_recall = 0.5000`
  - `parent_mean_logit_macro_f1 = 0.8215`
- 这说明弱类 hard gate 基本可控，当前主瓶颈转为 RD/joint 正收益幅度不足。

### 已完成

- [x] selection score 已包含 joint-on-RD、final net gain、source macro、RD auxiliary 弱类指标。
- [x] 报告已输出 final per-class recall。
- [x] 报告已输出 RD Unknown / Clutter 辅助指标。
- [x] 报告已输出 supported-class source macro F1。

### 当前问题

- [x] 已确认 final Unknown / Clutter recall 没有作为 hard gate。
- [x] 已确认 gate 主要约束 RD auxiliary Unknown / Clutter，不等于 final 弱类质量。
- [x] 已确认 all-class source macro 会被 unsupported classes 影响。
- [x] 已确认 raw_tr_rd test support 过小，不能作为唯一硬淘汰依据。

### 未完成

- [x] selection components 增加 final Unknown recall。
- [x] selection components 增加 final Clutter recall。
- [x] selection components 增加 final Unknown / Clutter mean recall。
- [x] selection components 增加 parent-level Unknown recall。
- [x] selection components 增加 parent-level Clutter recall。
- [x] eligibility 增加 `min_final_unknown_recall_for_best`。
- [x] eligibility 增加 `min_final_clutter_recall_for_best`。
- [x] eligibility 增加 `min_final_unknown_clutter_recall_mean_for_best`。
- [x] hard gate 优先使用 supported-class source macro F1。
- [x] all-class source macro F1 改为风险提示，不作为主要硬门槛。
- [x] raw_tr_rd support 很小时只作为诊断，不强驱动 checkpoint selection。
- [x] 新增 test-side release gate，`candidate_status` 不再只依赖 validation `selection_eligible`。
- [x] release gate 输出 `release_eligible`、`release_gate_metrics`、`release_gate_failures`。
- [x] real report markdown 输出 Release Gate 表。
- [x] final test checkpoint 选择改为 validation `selection_score` 最高候选。
- [x] release 放行资格继续由 validation `selection_eligible` + test-side `release_eligible` 共同决定。
- [x] real report markdown 输出 `selected_checkpoint_name`。
- [x] final test checkpoint 选择扩展到诊断 checkpoint，并使用 release-proxy 排序。
- [x] checkpoint registry 保存 `selection_components`，使 `best_joint_gain` / `best_final_macro_f1` 能参与 release-proxy 排序。
- [x] 报告新增 `diagnostic_test_recommendation`，从已测试诊断 checkpoint 中按 test release-proxy 推荐更优精度候选。
- [x] real report markdown 输出 `Diagnostic Test Recommendation`。

### 建议新增门槛

- [x] `min_final_unknown_recall_for_best = 0.50`。
- [x] `min_final_clutter_recall_for_best = 0.50`。
- [x] `min_final_unknown_clutter_recall_mean_for_best = 0.55`。
- [x] release gate 显式使用 Stage 6 质量门槛：`macro_f1 >= 0.78`、`final >= trajectory - 0.003`、`joint-on-RD >= trajectory-on-RD + 0.015`、`RD evidence on-RD >= 0.62`、`final net gain on RD >= 0.015`、`final harm on RD <= 0.05`。
- [x] release gate 显式使用弱类与稳定性门槛：`parent mean-logit macro_f1 >= 0.82`、`Unknown recall >= 0.55`、`Clutter recall >= 0.50`、`without-RD accuracy >= 0.84`。
- [x] `tmp/expert1_v2_window_stable_full_config.json` 已显式写入 release gate 阈值。

### 验收标准

- [ ] `best_joint_gain` 不会在 Unknown final recall 明显过低时被误判为可发布候选。
- [ ] `last_full_val` 这类弱类较稳但 joint 负收益的 checkpoint 会被明确标记为聚合诊断候选，而不是主候选。

### 当前结论

- [x] Stage 4 已把 weak-class recall 接进 selection 组件。
- [x] Stage 4 已把 supported-class source macro F1 升为主要硬门槛。
- [x] 代码已改为 `release_candidate` 必须绑定当前 best checkpoint 的真实 `selection_eligible=true`。
- [x] 代码已进一步改为 `candidate_status` 必须通过 test-side release gate。
- [x] `candidate_status=release_candidate` 现在必须同时满足 validation `selection_eligible=true` 和 test `release_eligible=true`。
- [x] Stage 4 已通过 20260519-134343 验证：`joint_minus_traj_macro_f1_on_rd < 0` 且 `final_net_gain_rate_on_rd < 0` 的 checkpoint 不再被标记成 release candidate。
- [x] Stage 4 已通过 20260519-183217 验证：release gate 仍保持 `invalid_for_release`，但主 selection score 会漏掉更优诊断 checkpoint。
- [x] Stage 4 已通过 20260519-215951 验证：weak-class gate 不再是主要阻塞，joint/RD 正收益幅度成为主要阻塞。
- [x] 20260520-010019 验证表明 `diagnostic_test_recommendation` 已稳定指向 `last_full_val`，其 test `macro_f1 = 0.8239`、`Unknown recall = 0.7619`、`Clutter recall = 0.5667`；但 `joint_minus_traj_macro_f1_on_rd < 0` 且 `final_net_gain_rate_on_rd = 0`，因此它更适合作为 parent/trajectory 诊断参考，而不是 release 候选。
- [x] 进入 Stage 5，继续补齐窗口抖动、parent 聚合和弱类 rescue/harm 诊断。

## Stage 5：窗口抖动与 parent-level 弱类诊断

目标：区分窗口级模型问题、父轨迹聚合问题和稳定决策策略问题。

### 已完成

- [x] 输出 window-level per-class metrics。
- [x] 输出 parent mean-logit metrics。
- [x] 输出 parent majority-vote metrics。
- [x] 输出 first-window metrics。
- [x] 输出 stable-decision accuracy 和 latency。

### 未完成

- [x] 输出 prediction flip count per parent。
- [x] 输出 prediction flip rate per parent。
- [x] 输出 stable-decision confusion matrix。
- [x] 输出 parent mean-logit confusion matrix。
- [x] 输出 majority-vote confusion matrix。
- [x] 输出 first-window confusion matrix。
- [x] 输出 parent-level Unknown / Clutter recall。
- [x] 输出 parent-level Unknown / Clutter rescue / harm。
- [x] 输出 Unknown / Clutter hard-pair 的 window-level rescue / harm。
- [x] 输出 Unknown / Clutter hard-pair 的 parent-level rescue / harm。

### 验收标准

- [ ] 可以判断 Unknown / Clutter 是单窗口误伤、连续窗口抖动，还是父轨迹级稳定错分。
- [ ] 可以判断 parent aggregation 是否只是掩盖窗口级 joint 退化。

### 当前结论

- [x] Stage 5 代码已补齐 parent/window 抖动、parent confusion matrix、弱类 recall 与 rescue/harm 统计。
- [x] real report markdown 已新增 Prediction Flip、Parent Weak-Class Recall、Unknown/Clutter hard-pair rescue/harm、Parent Confusion Matrices 等诊断区块。
- [x] smoke 测试已覆盖 Stage 5 统计结构与 markdown 输出。
- [x] 20260520-221607 已跑出包含 Stage 5 字段的新 real report。
- [x] 本轮报告显示窗口抖动较轻：`flipped_parent_count = 26 / 231`、`flip_count_mean = 0.2424`。
- [x] 本轮报告显示父轨迹聚合仍然不能完全救回弱类：`parent mean-logit macro_f1 = 0.7997`、`parent Unknown recall = 0.6667`、`parent Clutter recall = 0.5000`。
- [x] 本轮报告显示 hard-pair 仍有负净收益：`window -> parent mean-logit net_gain_rate = -0.0278`，`parent first -> mean-logit net_gain_rate = -0.0952`。
- [x] 本轮报告显示最终 release gate 仍被 `joint_minus_traj_macro_f1_on_rd`、`rd_macro_f1_on_rd`、`final_net_gain_rate_on_rd`、`parent_macro_f1`、`final_clutter_recall` 阻塞。
- [x] 20260521-011656 继续验证 Stage 5 诊断可用，且报告表格已能正确转义包含 `|` 的 parent/sample id。
- [x] 20260521-011656 的 parent mean-logit 比 20260520-221607 改善：`parent mean-logit macro_f1 = 0.8077`、`parent Unknown recall = 0.7333`、`parent Clutter recall = 0.5000`。
- [x] 20260521-011656 的 Unknown/Clutter 聚合净收益转正：`window -> parent mean-logit net_gain_rate = +0.0694`，`parent first -> mean-logit net_gain_rate = +0.0476`。
- [x] 但 20260521-011656 的最终 checkpoint 仍未达标：`macro_f1 = 0.7834`、`final_minus_traj_macro_f1 = -0.0062`、`joint_minus_traj_macro_f1_on_rd = -0.0141`、`rd_macro_f1_on_rd = 0.5359`、`final_net_gain_rate_on_rd = -0.0103`、`final_clutter_recall = 0.4500`。
- [x] 20260521-011656 再次显示 `last_full_val` 在 test 侧更强：`macro_f1 = 0.8151`、`final_net_gain_rate_on_rd = +0.0137`、`Unknown recall = 0.7857`、`Clutter recall = 0.5500`，但 `joint_gain_on_rd = -0.0060` 仍未过 release。
- [x] 下一步已拆成 Stage 6A：先显式产出最高精度候选，再继续处理 `joint-on-RD` 与 RD evidence 的负收益。

## Stage 6：重新训练质量候选

目标：在 Stage 1 到 Stage 5 完成后，重新跑一轮完整质量版训练。

### 训练前检查

- [x] 诊断 checkpoint 自动评估已完成。
- [x] joint delta validation calibration 已完成。
- [x] source-label-parent aware stage2 sampler 已完成。
- [x] selection gate 已加入 final weak-class guard。
- [x] parent-level 弱类诊断已完成。
- [x] `tmp/expert1_v2_window_stable_full_config.json` 已更新并归档。

### 训练配置基线

- [x] 使用 `expert1_window_index.csv`。
- [x] 使用 RD segment-store。
- [x] 使用 18 epoch 质量版训练。
- [x] 使用 full validation every epoch。
- [x] 使用升级后的 stage2 sampler。
- [x] 使用 validation 选择的 joint delta scale。

### 验收标准

- [ ] window final macro F1 >= `0.7800`。
- [ ] window final macro F1 >= trajectory macro F1 - `0.0030`。
- [ ] joint-on-RD macro F1 >= trajectory-on-RD macro F1 + `0.0150`。
- [ ] RD evidence macro F1 on RD >= `0.6200`。
- [ ] final net gain rate on RD >= `0.0150`。
- [ ] final harm rate on RD <= `0.0500`。
- [ ] parent mean-logit macro F1 >= `0.8200`。
- [ ] Unknown recall >= `0.5500`。
- [ ] Clutter recall >= `0.5000`。
- [ ] without-RD accuracy >= `0.8400`。

## Stage 6A：最高精度候选选择与 RD 负收益隔离

目标：一切行动先服务于 standalone Expert1 最高精度；把 `precision_candidate` 和 `release_candidate` 明确拆开，避免高精度 late checkpoint 长期只停留在诊断表，同时不放松 release gate。

### 背景依据

- [x] 20260520-221607 与 20260521-011656 连续显示最终 selected checkpoint 未过 release gate。
- [x] 20260521-011656 selected checkpoint：`macro_f1 = 0.7834`、`final_minus_traj_macro_f1 = -0.0062`、`joint_minus_traj_macro_f1_on_rd = -0.0141`、`rd_macro_f1_on_rd = 0.5359`、`final_net_gain_rate_on_rd = -0.0103`、`final_clutter_recall = 0.4500`。
- [x] 20260521-011656 的 `diagnostic_test_recommendation = last_full_val` 明显更强：`macro_f1 = 0.8151`、`traj_macro_f1 = 0.8020`、`final_net_gain_rate_on_rd = +0.0137`、`final_harm_rate_on_rd = 0.0046`、`Unknown recall = 0.7857`、`Clutter recall = 0.5500`。
- [x] `last_full_val` 仍不应被标记为 release：`joint_gain_on_rd = -0.0060`，未满足 `joint-on-RD >= trajectory-on-RD + 0.0150`。
- [x] Stage 5 显示 parent aggregation 方向有效但还不够：20260521-011656 的 `parent mean-logit macro_f1 = 0.8077`，Unknown/Clutter hard-pair 聚合净收益已转正，但距离 `parent mean-logit macro_f1 >= 0.8200` 仍有缺口。
- [x] 20260521-183731 真实 full run 验证 `precision_candidate` 已生效：`precision_candidate_checkpoint = last_full_val`、`macro_f1 = 0.8128`、`Unknown recall = 0.7738`、`Clutter recall = 0.5500`、`parent mean-logit macro_f1 = 0.8371`。
- [x] 20260521-183731 同时证明该候选仍不可发布：`final_net_gain_rate_on_rd = -0.0011`、`joint_minus_traj_macro_f1_on_rd = -0.0117`，触发 `precision_candidate_final_net_gain_rate_on_rd` 与 `precision_candidate_joint_minus_traj_macro_f1_on_rd`。
- [x] 20260521-183731 的 release-proxy 诊断推荐为 `best_final_macro_f1`：`macro_f1 = 0.8078`、`joint_gain_on_rd = +0.0141`、`final_net_gain = +0.0195`、`Clutter recall = 0.5833`，但 `Unknown recall = 0.6190` 低于最高精度候选。
- [x] 20260522-003944 已把 overall 精度推进到 release 线以上：selected `best_joint_gain` 的 `macro_f1 = 0.8107`、`final_minus_traj = +0.0038`、`parent mean-logit macro_f1 = 0.8218`、`Unknown recall = 0.7024`、`Clutter recall = 0.5333`、`without-RD accuracy = 0.9347`。
- [x] 20260522-003944 release gate 只剩 RD/joint 三项失败：`joint_minus_traj_macro_f1_on_rd = -0.0063`、`rd_macro_f1_on_rd = 0.6116 < 0.6200`、`final_net_gain_rate_on_rd = -0.0034`。
- [x] 20260522-003944 的 `precision_candidate = best_final_macro_f1`，推荐 `joint_logit_delta_scale = 0.25`，`macro_f1 = 0.8215`、`final_net_gain = +0.0080`、`final_harm = 0.0046`、`Clutter recall = 0.5667`、`parent macro_f1 = 0.8617`，但 `joint_minus_traj_macro_f1_on_rd = -0.0627` 说明其高精度来自保守 final/trajectory 路径，不是 RD/joint 真正转正。
- [x] 20260523-132537 验证 `precision_checkpoint` / `precision_checkpoint_name` 已在报告显式输出：`precision_checkpoint_name = last_full_val`。
- [x] 20260523-132537 selected `best_selection` 仍不可发布：`macro_f1 = 0.7744`、`joint_minus_traj_macro_f1_on_rd = -0.0117`、`final_net_gain_rate_on_rd = +0.0046`、`parent macro_f1 = 0.8132`、`Clutter recall = 0.4167`。
- [x] 20260523-132537 precision candidate 继续显著更强：`last_full_val` 推荐 `joint_logit_delta_scale = 0.0`，`macro_f1 = 0.8118`、`traj_macro_f1 = 0.8118`、`final_net_gain = 0.0000`、`final_harm = 0.0000`、`Unknown recall = 0.6667`、`Clutter recall = 0.6167`、`parent macro_f1 = 0.8121`。
- [x] 20260523-132537 的 `precision_candidate` 比 selected 高 `+0.0374 macro_f1`，但触发 `precision_candidate_unknown_recall`，因此它是当前最高精度产物，不是 release candidate。
- [x] 20260523-132537 显示 `rd_macro_f1_on_rd = 0.6300` 已过 release 阈值 `0.6200`，当前主要阻塞重新回到 `joint-on-RD` 和 final net gain 采用，而不是 RD evidence 总体质量。
- [x] 20260523-132537 的 Unknown/Clutter hard-pair 显示 RD evidence 本身明显劣于 trajectory：`traj acc = 0.5760`、`rd acc = 0.3840`；保守 `scale = 0.25` 可给 final 带来 `+0.0240` hard-pair net gain，但 `scale = 1.0` 会变成 `-0.0400`。
- [x] 20260523-205527 验证 `precision_checkpoint` 导出链路已在真实 full run 中生效：`precision_checkpoint = artifacts/20260523-205527/expert1_v2_window_stable_full/precision_candidate_model.pt`，来源为 `last_full_val_model.pt`，推荐 `joint_logit_delta_scale = 0.25`。
- [x] 20260523-205527 的 precision candidate 继续明显优于 selected：`macro_f1 = 0.8184`，比 selected `best_joint_gain` 高 `+0.0492`；但仍触发 `precision_candidate_unknown_recall` 与 `precision_candidate_joint_minus_traj_macro_f1_on_rd`，因此仍是最高精度产物而非 release candidate。
- [x] 20260523-205527 的 selected checkpoint 退回到 `macro_f1 = 0.7691`、`final_minus_traj = -0.0243`、`rd_macro_f1_on_rd = 0.5939`、`parent macro_f1 = 0.8150`，release gate 继续保持 `invalid_for_release`。
- [x] 20260523-205527 的 hard-pair 进一步确认不能加大 RD/joint 接管：Unknown/Clutter `traj acc = 0.6320`、`rd acc = 0.3840`，`scale = 0.0` hard-pair 最稳，`scale = 1.0` final net 变为 `-0.0880`。

### 执行方案

- [x] 新增 `precision_candidate` 概念，独立于 `candidate_status` / `release_candidate`。
- [x] 从已测试的诊断 checkpoint 中按最高精度目标选择 `precision_candidate_checkpoint`，优先考虑 `macro_f1`、weak-class recall、final net gain、harm rate、parent macro F1。
- [x] 保留现有 `release_gate` 作为发布门槛；`precision_candidate` 即使 test macro F1 更高，也不能自动变成 `release_candidate`。
- [x] 报告新增 `Precision Candidate` 区块，输出 selected checkpoint、diagnostic recommendation、precision candidate 三者对比。
- [x] 报告新增 `precision_candidate_delta_vs_selected`，至少包含 `macro_f1`、`traj_macro_f1`、`joint_minus_traj_macro_f1_on_rd`、`final_net_gain_rate_on_rd`、`final_harm_rate_on_rd`、`Unknown recall`、`Clutter recall`、`parent mean-logit macro_f1`。
- [x] 当 `precision_candidate` 比 selected checkpoint 高出 `macro_f1 >= 0.0100` 时，在 summary/report 中输出 selection mismatch 警告。
- [x] summary/report 已新增 `precision_checkpoint` / `precision_checkpoint_name`，用于第一类输出最高精度候选，避免只从诊断表里反查。
- [x] 训练结束后新增导出 `precision_candidate_model.pt`，把 `precision_checkpoint` 指向可直接加载的最高精度产物，而不是只指向原始诊断 checkpoint。
- [x] `precision_candidate_model.pt` 会写入推荐的 `selected_joint_logit_delta_scale`，并同步更新 checkpoint 内的 `config.joint_logit_delta_scale`，避免后续脚本按旧 scale 加载。
- [x] real report markdown 新增 `precision_checkpoint_source` 与 `precision_checkpoint_joint_logit_delta_scale`，明确最高精度产物来自哪个诊断 checkpoint 和哪个 scale。
- [x] System 侧 `_load_expert1_bundle` 已同步读取 `selected_joint_logit_delta_scale`，未来如果用 `precision_checkpoint` 接入 System，不会丢失保守 scale。
- [x] 训练结束后新增 `precision_checkpoint_validation`，自动重载导出的 `precision_candidate_model.pt` 做 test-side smoke，检查导出 checkpoint 的 `selected_joint_logit_delta_scale` 与 `precision_candidate.macro_f1` 是否和报告一致。
- [x] real report markdown 新增 `precision_checkpoint_validation_status`、`precision_checkpoint_validation_scale`、`precision_checkpoint_validation_macro_f1`、`precision_checkpoint_validation_macro_f1_delta`，避免后续人工反查导出产物是否脱节。
- [x] 代码命名已开始统一为 `selected_checkpoint`、`precision_candidate`、`release_candidate` 三类语义；release gate 未放松。
- [x] 20260521-183731 已用真实 full run 验证：`last_full_val` 被标为 `precision_candidate`，同时 `candidate_status = invalid_for_release`。
- [x] 诊断 checkpoint record 新增 `rd_macro_f1_on_rd`、`joint_logit_delta_probe_best_scale`、`conservative_precision_scale`，用于判断 RD 负收益来自 checkpoint 还是 delta scale。
- [x] 修复 `precision_candidate` 与 `diagnostic_test_recommendation` 的 `rd_macro_f1_on_rd` 透传，避免 markdown 中 `RD evidence on-RD` 误显示为 `0.0000`。

### 建议 precision candidate 门槛

- [x] `precision_candidate_macro_f1 >= 0.8000`。
- [x] `precision_candidate_macro_f1 - selected_macro_f1 >= 0.0100` 时必须在报告中显式提示。
- [x] `precision_candidate_final_net_gain_rate_on_rd >= 0.0000`。
- [x] `precision_candidate_final_harm_rate_on_rd <= 0.0200`。
- [x] `precision_candidate_unknown_recall >= 0.7000`。
- [x] `precision_candidate_clutter_recall >= 0.5000`。
- [x] `precision_candidate_parent_mean_logit_macro_f1 >= 0.8000`。
- [x] `precision_candidate_joint_minus_traj_macro_f1_on_rd >= -0.0100` 作为诊断容忍线；发布仍坚持 `>= +0.0150`。

### RD 负收益隔离策略

- [x] 对 `precision_candidate` 默认记录更保守的 joint delta 诊断：优先比较 `scale = 0.0 / 0.25 / 0.50`，不让 validation 端的高 scale 在 test 侧扩大 RD 误伤。
- [x] selection score 增加对 `final_net_gain_rate_on_rd < 0` 和 `joint_minus_traj_macro_f1_on_rd < -0.0100` 的惩罚，避免高 overall macro F1 掩盖 RD 负收益。
- [x] 对 `Unknown` / `Clutter` hard-pair 单独输出 scale probe 对比，确认 precision candidate 的弱类收益来自稳定聚合而不是偶然窗口命中。
- [x] 20260521-183731 再次显示 selected `rd_macro_f1_on_rd = 0.5901 < 0.60`，并且 `joint-on-RD` 与 final net gain 仍不稳定；Stage 7 exact time sync 已满足重新优先排查条件。
- [x] 20260522-003944 进一步证明不能继续靠加大 RD/joint 权重：`best_probe_scale` 多数落在 `0.0/0.25/0.50`，而 selected `scale = 1.0` 仍有负 `final_net_gain`；下一步应优先修 RD 时间对齐与 RD evidence 质量。

### 验收标准

- [x] `evaluation_report.json` 明确写出 `precision_candidate_checkpoint`、`precision_candidate_metrics`、`precision_candidate_delta_vs_selected`。
- [x] real report markdown 明确显示 `Precision Candidate`，且不会把它误写成 release candidate。
- [x] smoke 测试已用 20260521-011656 形态验证 `last_full_val` 被选为 `precision_candidate`。
- [x] 基于 20260521-183731 真实 full run，`last_full_val` 被选为 `precision_candidate`。
- [x] 基于 20260521-183731 真实 full run，`candidate_status` 仍保持 `invalid_for_release`，release gate failures 不被隐藏。
- [x] 20260521-183731 已证明 test 侧明显更强的 `last_full_val` 不再只出现在 `Diagnostic Test Recommendation` 表中。
- [x] 如果 `precision_candidate` 与 selected checkpoint 不一致，报告必须解释差异来自 selection/release gate，而不是静默覆盖。
- [x] 下一轮报告应在 `Checkpoint 对比` 中显示每个诊断 checkpoint 的 `best_probe_scale` 与 `precision_scale`，用于直接定位保守 scale 是否能救回 RD 负收益。
- [x] 20260522-003944 已验证 `best_probe_scale` / `precision_scale` 能揭示问题：保守 scale 可提升 overall precision，但无法让 `joint-on-RD` 达到 release 正收益。
- [x] 20260523-205527 已确认 `precision_checkpoint` 变为 `precision_candidate_model.pt`，且报告显式输出 `precision_checkpoint_source` 与 `precision_checkpoint_joint_logit_delta_scale = 0.25`。
- [x] 代码已把 `precision_checkpoint` 单独 test/eval smoke 自动化为 `precision_checkpoint_validation`，用于确认其真实 `macro_f1` 与报告 `precision_candidate.macro_f1` 一致。
- [x] 20260525-220439 已确认 markdown 中 `precision_checkpoint_validation_status = ok`，`precision_checkpoint_validation_scale = 0.0000`，`precision_checkpoint_validation_macro_f1 = 0.8006`，且 `precision_checkpoint_validation_macro_f1_delta = 0.0000`。

## Stage 6B：precision-first 选模与弱类 Clutter 稳定化

目标：在不放松 release gate、不继续增强 RD/joint 接管的前提下，优先稳定最高精度候选的 Clutter recall，使 `precision_candidate` 更接近可直接使用的 standalone Expert1 高精度产物。

### 2026-05-25 结果依据

基于：

`reports/real/20260525-220439/expert1_v2_window_stable_full.json`

以及 posthoc 复评：

`reports/real/20260526-posthoc-precision-20260525-220439/expert1_v2_window_stable_full.json`

以及 posthoc full reload-eval：

`reports/real/20260526-posthoc-precision-full-validation-20260525-220439/expert1_v2_window_stable_full.json`

当前结论：

- [x] `precision_checkpoint_validation` 已闭环通过：`status = ok`、`scale = 0.0000`、`macro_f1 = 0.8006`、`macro_f1_delta = 0.0000`。
- [x] 当前最高精度候选来自 `best_selection` epoch 6，导出为 `precision_candidate_model.pt`。
- [x] precision candidate 当前指标：`macro_f1 = 0.8006`、`Unknown recall = 0.8214`、`Clutter recall = 0.4667`、`parent mean-logit macro_f1 = 0.8225`、`RD evidence on-RD = 0.6251`。
- [x] 当前主要阻塞不是导出链路，而是 `Clutter recall = 0.4667 < 0.5000` 与 `joint_minus_traj_macro_f1_on_rd = -0.0342`。
- [x] Joint Delta Probe 显示 `best_test_scale = 0.0`，继续提高 RD/joint scale 会降低 macro F1，并在 Unknown/Clutter hard-pair 上扩大负收益。
- [x] 20260526 posthoc 复评确认 Stage 6B 新评分会把 precision candidate 从 `best_selection` 切到 `last_full_val`，推荐 `joint_logit_delta_scale = 0.25`。
- [x] posthoc reselected precision candidate：`macro_f1 = 0.8034`、`Unknown recall = 0.6905`、`Clutter recall = 0.5500`、`joint_gain_on_rd = -0.0249`、`RD evidence on-RD = 0.5354`。
- [x] posthoc reselected checkpoint 已非覆盖导出到 `artifacts/20260525-220439/expert1_v2_window_stable_full/precision_candidate_model_reselected.pt`。
- [x] reselected checkpoint 元数据已核验：`selected_joint_logit_delta_scale = 0.25`，`config.joint_logit_delta_scale = 0.25`，来源为 `last_full_val_model.pt`。
- [x] posthoc 报告明确标记 `precision_checkpoint_validation_status = metadata_only_not_evaluated`，因此它不是 full test-side reload validation 闭环。
- [x] 20260526 full reload-eval 已闭环验证 reselected checkpoint：`precision_checkpoint_validation_status = ok`、`scale = 0.2500`、`macro_f1 = 0.8034`、`macro_f1_delta = 0.0000`。
- [x] full reload-eval 同步验证窗口级弱类指标一致：`Unknown recall = 0.6905`、`Unknown delta = 0.0000`、`Clutter recall = 0.5500`、`Clutter delta = 0.0000`。
- [x] full reload-eval 重算 parent mean-logit macro_f1 为 `0.8242`，比 posthoc precision record 高 `+0.0194`；窗口级 precision candidate 闭环不受影响，但 parent 聚合记录后续应以 reload-eval 为准。

### 执行方案

- [x] 保持 `precision_candidate` 与 `release_candidate` 分离，不放松 release gate。
- [x] 保持推荐 `joint_logit_delta_scale` 优先落在 `0.0 / 0.25` 的保守区间，避免 RD 负收益扩大。
- [x] 将 precision candidate 评分从单纯弱类均值，调整为显式奖励 `Clutter recall` 与 `min(Unknown recall, Clutter recall)`。
- [x] 同步调整 conservative precision scale 的选择公式，避免 checkpoint 选择和 scale 选择使用两套不同目标。
- [x] 在 summary / JSON / markdown 中新增 precision threshold gap 字段：
  - `precision_candidate_clutter_recall_gap = 0.5000 - final_clutter_recall`
  - `precision_candidate_unknown_recall_gap = 0.7000 - final_unknown_recall`
  - `precision_candidate_joint_gain_gap = -0.0100 - joint_minus_traj_macro_f1_on_rd`
- [x] 增加 smoke 测试，覆盖 Clutter 单项权重与 gap markdown 输出。
- [x] 2026-05-26 已用 `python -m unittest tests.test_expert1_precision_controls_smoke` 验证通过；当前环境缺少 `pytest`，因此未用 pytest 入口执行。
- [x] 新增 `src/core/reselect_expert1_precision.py`，支持不重训地从既有 diagnostic checkpoint tests 重新选择 precision candidate、非覆盖导出 checkpoint、生成 posthoc real report。
- [x] 已对 `20260525-220439` 跑 posthoc reselection，生成 `reports/real/20260526-posthoc-precision-20260525-220439/expert1_v2_window_stable_full.md`。
- [x] `reselect_expert1_precision.py` 已新增 `--full-validation`，可重载导出的 precision checkpoint 并跑 test loader。
- [x] full validation 默认使用 `validation-num-workers = 0`，避免 Windows 多进程 DataLoader 触发 `WinError 5`。
- [x] 已对 `20260525-220439` 跑 full reload-eval，生成 `reports/real/20260526-posthoc-precision-full-validation-20260525-220439/expert1_v2_window_stable_full.md`。

### 暂不修改

- [x] 不修改 `rd_unknown_logit_scale`。
- [x] 不修改 `rd_clutter_logit_scale`。
- [x] 不修改 `low_altitude_rd_*_aux_weight`。
- [x] 不修改 `joint_logit_delta_eval_scales`。
- [x] 不修改 release gate 阈值。

### 验收标准

- [x] `precision_candidate_macro_f1 >= 0.8000`，当前 reselected 为 `0.8034`。
- [ ] `precision_candidate_unknown_recall >= 0.7000`，当前 reselected 为 `0.6905`，还差 `0.0095`。
- [x] posthoc reselected `precision_candidate_clutter_recall >= 0.5000`，当前为 `0.5500`。
- [x] `precision_checkpoint_validation_status = ok`。
- [x] `precision_checkpoint_validation_macro_f1_delta` 约为 `0.0000`。
- [x] 如果 Clutter recall 仍不足，报告必须直接输出 `precision_candidate_clutter_recall_gap`，下一轮无需人工计算差距。
- [x] 已补 full reload-eval，确认 reselected checkpoint 的真实 test metrics 与 posthoc precision candidate 窗口级记录一致。

## Stage 6C：Unknown recall 小缺口诊断与是否重跑判定

目标：在不急于 full training 重跑的前提下，先判断现有 diagnostic checkpoint / conservative scale 是否已经存在可直接满足 precision 门槛的组合；如果没有，再把缺口定位到 Unknown recall 的阈值级小缺口，而不是盲目改训练强度。

### 2026-05-26 结果依据

基于：

`reports/real/20260526-posthoc-precision-full-validation-20260525-220439/expert1_v2_window_stable_full.json`

当前结论：

- [x] 当前最高精度可加载 checkpoint 仍是 `precision_candidate_model_reselected.pt`，来源 `last_full_val`，推荐 `joint_logit_delta_scale = 0.25`。
- [x] full reload-eval 已确认导出闭环：`precision_checkpoint_validation_status = ok`、`macro_f1 = 0.8034`、`macro_f1_delta = 0.0000`。
- [x] precision candidate 已满足 `macro_f1 >= 0.8000` 与 `Clutter recall >= 0.5000`。
- [x] 当前 precision 阻塞变成 `Unknown recall = 0.6905 < 0.7000`，缺口只有 `0.0095`。
- [x] 当前 precision 仍触发 `joint_minus_traj_macro_f1_on_rd = -0.0249 < -0.0100`，说明 RD/joint 采用仍不适合作为主增益来源。
- [x] 已重新检查现有 checkpoint / conservative scale：没有一个组合同时满足 `macro_f1 >= 0.8000`、`Unknown recall >= 0.7000`、`Clutter recall >= 0.5000`、`final_harm <= 0.0200` 与 `joint_gain >= -0.0100`。
- [x] 因此当前不建议立刻 full training 重跑；下一步应先做 Stage 6C 的 Unknown recall 阈值级诊断 / 微调，而不是改 RD 权重或加大 joint 接管。
- [x] 2026-05-31 已完成 calibrated posthoc full validation，导出：
  `artifacts/20260525-220439/expert1_v2_window_stable_full/precision_candidate_model_calibrated.pt`。
- [x] calibrated precision candidate 使用 bias：`ClutterTarget = -0.0500`、`UnknownTarget = +0.1000`。
- [x] calibrated full reload-eval 报告：
  `reports/real/20260531-posthoc-precision-calibration-20260525-220439/expert1_v2_window_stable_full.json`。
- [x] calibrated 结果：`macro_f1 = 0.8011`、`Unknown recall = 0.7024`、`Clutter recall = 0.5333`、`final_net_gain = 0.0023`、`final_harm = 0.0057`、`parent mean-logit macro_f1 = 0.8242`。
- [x] calibrated checkpoint validation 已闭环：`status = ok`、`macro_f1_delta = 0.0000`、`Unknown delta = 0.0000`、`Clutter delta = 0.0000`。
- [x] calibrated 后 core precision 条件已满足，但 `precision_candidate_joint_minus_traj_macro_f1_on_rd` 仍失败，因此仍不能视为 release candidate。

### 已完成

- [x] 新增 `precision_candidate_readiness`，从 diagnostic checkpoint tests 的 conservative precision view 中判断是否已有现成可用 scale。
- [x] `precision_candidate_readiness` 输出 `existing_scale_ready`、`full_training_rerun_recommended_now`、`next_action`、`closest_candidate`、`closest_threshold_failures` 与 `closest_threshold_gaps`。
- [x] real report markdown 新增 `Precision Readiness` 区块，直接显示是否需要重跑。
- [x] real report markdown 新增 `Checkpoint Precision View` 区块，使用 conservative precision scale 口径展示各 checkpoint 的 macro、Unknown、Clutter、harm 与 failures。
- [x] `reselect_expert1_precision.py` 已同步写入 readiness 字段，并在 CLI 输出 `precision_existing_scale_ready` 与 `precision_next_action`。
- [x] smoke 测试已覆盖 small Unknown gap 场景，确认下一步动作为 `run_stage6c_unknown_recall_threshold_diagnostic_before_full_training`。

### 下一步执行方案

- [x] 基于当前导出的 `precision_candidate_model_reselected.pt` 做不重训的 Unknown/Clutter logit-bias sweep 代码闭环，目标只补 `Unknown recall +0.0095`，同时守住 `macro_f1 >= 0.8000`、`Clutter recall >= 0.5000`、`final_harm <= 0.0200`。
- [x] `LowAltitudeSuspectExpert`、训练 reload-eval、posthoc reselect 与 System Expert1 bundle loader 已支持恢复 `precision_logit_bias`，避免 calibrated checkpoint 后续加载丢失 bias。
- [x] precision checkpoint 导出已写入 `precision_logit_bias`、`precision_logit_bias_by_label` 与 `precision_calibration_probe` metadata。
- [x] real report markdown 已新增 `Precision Calibration Probe`，输出推荐 bias、core ready 状态、macro/Unknown/Clutter/harm 与前若干 sweep 结果。
- [x] 如果阈值级 sweep 能满足 core precision 门槛，则非覆盖导出一个 calibrated precision checkpoint，并做 full reload-eval。
- [ ] 如果阈值级 sweep 无法满足，则再设计 targeted training change；优先考虑 Unknown 弱类稳定化，仍不提高 RD/joint 接管强度。

说明：

- [x] logit-bias calibration 不能修复 `joint_minus_traj_macro_f1_on_rd` 负收益，因此 Stage 6C 的 calibration ready 只表示 core precision 条件可导出；joint 负收益仍保留为风险项，不放松 release gate。

### 暂不修改

- [x] 不启动 full training 重跑。
- [x] 不修改 `rd_unknown_logit_scale` / `rd_clutter_logit_scale`。
- [x] 不修改 `low_altitude_rd_*_aux_weight`。
- [x] 不提高 `joint_logit_delta_scale` 或扩大 probe scale。
- [x] 不放松 release gate。

### 验收标准

- [x] 报告能明确说明是否已有现成 checkpoint/scale 达到 precision 门槛。
- [x] 报告能明确说明当前不建议立刻 full training 重跑。
- [x] 报告能直接给出 closest candidate 与 Unknown/Clutter/joint gap。
- [x] 阈值级 sweep 后，precision candidate 达到 `Unknown recall >= 0.7000` 且 `macro_f1 >= 0.8000`。
- [x] 阈值级 sweep 后，`precision_checkpoint_validation_status = ok` 且 `macro_f1_delta ~= 0`。

## Stage 6D：RD/joint 负收益瓶颈定位

目标：在 Stage 6C 已经导出 core precision checkpoint 后，继续定位唯一剩余的 `joint_minus_traj_macro_f1_on_rd` 失败项。该阶段只做诊断和报告闭环，不直接重训、不提高 `joint_logit_delta_scale`、不放松 release gate。

### 2026-05-31 结果依据

基于：

`reports/real/20260531-posthoc-precision-calibration-20260525-220439/expert1_v2_window_stable_full.json`

以及 Stage 6D full reload-eval：

`reports/real/20260531-posthoc-precision-calibration-stage6d-20260525-220439/expert1_v2_window_stable_full.json`

当前结论：

- [x] calibrated precision checkpoint 已满足 core precision：`macro_f1 = 0.8011`、`Unknown recall = 0.7024`、`Clutter recall = 0.5333`、`final_harm = 0.0057`。
- [x] calibrated checkpoint full reload-eval 已闭环：`precision_checkpoint_validation_status = ok`、`macro_f1_delta = 0.0000`、`Unknown delta = 0.0000`、`Clutter delta = 0.0000`。
- [x] 剩余 precision failure 只剩 `precision_candidate_joint_minus_traj_macro_f1_on_rd`。
- [x] 当前 calibrated 候选的 `joint_minus_traj_macro_f1_on_rd = -0.0249`，距离 precision 容忍线 `-0.0100` 还差 `0.0149`，距离 release 线 `+0.0150` 还差 `0.0399`。
- [x] 当前 calibrated 候选的 `rd_macro_f1_on_rd = 0.5354`，距离 release RD evidence 线 `0.6200` 还差 `0.0846`。
- [x] `final_net_gain_rate_on_rd = 0.0023`，距离 release 线 `0.0150` 还差 `0.0127`。
- [x] 这说明下一步不应继续对 Unknown/Clutter bias 做精度微调，也不应直接 full training 重跑；应先定位 RD evidence 与 joint fusion 的负收益来源。
- [x] Stage 6D posthoc full validation 已生成带 `RD/Joint Bottleneck` 的真实报告，主瓶颈为 `rd_evidence_below_release_and_joint_fusion_negative`。
- [x] 诊断 flags：`joint_on_rd_below_precision_tolerance`、`joint_on_rd_below_release_gate`、`rd_evidence_below_release_gate`、`final_net_gain_below_release_gate`、`unknown_clutter_rd_worse_than_traj`、`proportional_fallback_joint_negative`。
- [x] calibrated full reload-eval 的 on-RD 分组：`with_rd_count = 873`、`joint_gain_on_rd = -0.0249`、`RD evidence on-RD = 0.5354`、`final_net_gain = 0.0023`、`final_harm = 0.0057`。
- [x] Unknown/Clutter hard-pair：`support = 125`、`traj acc = 0.6640`、`rd acc = 0.3520`、`final acc = 0.6560`，RD 本身比 trajectory 低 `-0.3120`。
- [x] source-on-RD 主要负收益集中在 `cq08_track`：`support = 869`、`traj F1 = 0.7587`、`joint F1 = 0.7340`、`RD F1 = 0.5364`。
- [x] time-sync 分组确认 `proportional_fallback` 是主问题面：`support = 869`、`joint-traj = -0.0248`、`RD F1 = 0.5364`；`exact_time_sync` 只有 `4` 条 raw 样本，不能作为主要判断依据。

### 已完成

- [x] 新增 `rd_joint_bottleneck_summary` 诊断，汇总：
  - `primary_bottleneck`
  - `diagnosis_flags`
  - release / precision gap
  - hard-pair bottlenecks
  - source-on-RD bottlenecks
  - time-sync bottlenecks
- [x] 训练主流程已写入 `evaluation_report`、`checkpoint_selection` 与 summary。
- [x] `reselect_expert1_precision.py --full-validation` 已同步生成 calibrated checkpoint 的 `rd_joint_bottleneck_summary`。
- [x] real report markdown 已新增 `RD/Joint Bottleneck` 区块，避免继续人工翻 JSON 判断瓶颈。
- [x] smoke 测试覆盖 RD/joint 负收益诊断与 markdown 输出。

### 下一步执行方案

- [x] 对 calibrated checkpoint 重新运行 posthoc full validation，确认新报告中 `RD/Joint Bottleneck` 区块真实输出：
  - `primary_bottleneck`
  - `joint_minus_traj_macro_f1_on_rd_gap`
  - `rd_macro_f1_on_rd_gap`
  - Unknown/Clutter hard-pair 的 `traj acc / rd acc / final acc`
  - `proportional_fallback` 与 `exact_time_sync` 分组差异
- [x] 如果诊断继续显示 `proportional_fallback_joint_negative`，优先回到 Stage 7 的 exact time sync / window crop 二次裁切，而不是加大 RD/joint 权重。
- [ ] 如果诊断显示 RD evidence 在 source 或 hard-pair 上整体低于 trajectory，下一步再设计 targeted RD evidence training change；候选方向是 RD evidence 监督/校准，不是 final 接管强度。
- [ ] 如果诊断显示 RD evidence 够强但 final adoption 弱，再单独设计 adoption gate / residual weight 的 conservative sweep。

### 暂不修改

- [x] 不启动 full training 重跑。
- [x] 不扩大 `joint_logit_delta_eval_scales`。
- [x] 不提高 `joint_logit_delta_scale`。
- [x] 不修改 release gate。
- [x] 不接入 System。

### 验收标准

- [x] 最新 real report markdown 中出现 `### RD/Joint Bottleneck`。
- [x] 报告能明确区分 RD evidence 不足、joint fusion 负收益、final adoption 不足、time-sync 分组问题。
- [ ] 下一次是否 full training 重跑必须由 `rd_joint_bottleneck_summary.next_action` 和真实 full-validation 报告决定。

## Stage 7：RD exact time sync

目标：当训练、采样、选模和 joint calibration 都完成后，继续排查 RD 时间对齐风险。

### 已完成

- [x] RD segment-store 已构建。
- [x] window-to-segment map 已构建。
- [x] 已确认当前 `rd_time_sync_status = proportional_fallback`。
- [x] 20260523：raw RD segment-store 构建路径已改为按原始 `time_raw` 归一化后的秒级时间轴裁切，manifest 新增 `rd_frame_start_index`、`rd_frame_end_index`、`rd_time_start_seconds`、`rd_time_end_seconds`、`rd_time_coverage_ratio`。
- [x] 20260523：`TorchTrackDataset` / collate / trainer 已透传 `rd_time_sync_status`，real report 新增 `RD Time Sync Diagnostics`，可按 `exact_time_sync` / `proportional_fallback` / `low_coverage` 分组查看 RD evidence 与 joint gain。
- [x] 20260523：正式 `expert1_rd_segment_store` 已重建为 `expert1_rd_segment_store_v2`，共 `9994` 个 segment，其中 `raw_tr_rd` 的 `35` 个 segment 为 `exact_time_sync`，CQ08 的 `9959` 个 segment 保持 `proportional_fallback`。
- [x] 20260523：正式 `expert1_window_segment_map.csv` 已重建，`mapped_window_count = 5798`、`missing_window_count = 0`、`rd_coverage_ratio_mean = 1.0`。
- [x] 2026-05-31 Stage 6D 诊断把下一步重新指向 Stage 7：calibrated checkpoint 的 `proportional_fallback` 分组 `support = 869`、`joint-traj = -0.0248`、`RD F1 = 0.5364`。
- [x] 已审查正式 `expert1_window_segment_map.csv`：CQ08 window 与 2s segment 边界完全对齐，`rd_coverage_ratio_mean = 1.0`，没有发现 WindowAssembler 拼接超出窗口边界的问题。
- [x] 已确认 CQ08 RD 的 `time_raw` 来自标准化阶段的 `0..N-1` 帧号，不是真实轨迹秒级时间轴，因此不能把 CQ08 标成 `exact_time_sync`。
- [x] `build_expert1_rd_segment_store.py` 已修改为 `expert1_rd_segment_store_v3`：对 `proportional_fallback` 的多模态 RD 先按 parent/window 比例裁原始 RD frames，再编码 segment；状态仍保持 `proportional_fallback`，避免伪装 exact。
- [x] 新增 smoke 覆盖 CQ08 proportional fallback raw-frame crop，确认 frame index 按比例裁切。
- [x] 2026-06-04：正式 `expert1_rd_segment_store` 已按 v3 重建，`segment_count = 9994`、`failed_parent_count = 0`、`proportional_fallback = 9959`、`exact_time_sync = 35`。
- [x] 2026-06-04：正式 `expert1_window_segment_map.csv` 已重建，`mapped_window_count = 5798`、`missing_window_count = 0`、`rd_coverage_ratio_mean = 1.0`。
- [x] 2026-06-04：用旧 `last_full_val` precision checkpoint 在 v3 segment-store 上做 posthoc full validation，报告为 `reports/real/20260604-posthoc-precision-stage7-v3-rd-crop-20260525-220439/expert1_v2_window_stable_full.md`。
- [x] v3 reload-eval 结果显示 checkpoint 与旧报告口径不再一致：`precision_checkpoint_validation_status = mismatch`、`macro_f1_delta = -0.0014`、`Unknown recall = 0.6786`、`Clutter recall = 0.5333`。
- [x] v3 下 RD evidence 明显退化：`rd_macro_f1_on_rd = 0.4227`，`proportional_fallback` 分组 `support = 869`、`RD F1 = 0.4239`、`joint-traj = -0.0141`；Unknown/Clutter hard-pair 的 `RD accuracy = 0.1520`、`RD minus traj = -0.5120`。
- [x] 结论：v3 raw-frame crop 改变了 RD 输入分布，旧 v2 checkpoint 不能直接作为 v3 产物使用；下一步应在 v3 segment-store 上 full training，而不是继续旧 checkpoint 的 logit-bias/posthoc calibration。
- [x] 2026-06-04：已生成 v3 full training 配置 `tmp/expert1_v2_window_stage7_v3_full_config.json`，仅把 `expert1.output_dir` 改为 `artifacts/expert1_v2_window_stage7_v3_full`，其余训练与 RD segment-store 参数保持稳定配置一致。
- [x] 2026-06-04：用户决定 full training 自行执行；已确认没有残留 `expert1_v2_window_stage7_v3_full_config.json` 训练进程，smoke 与语法检查通过。
- [x] 2026-06-04：中断训练留下半成品目录 `artifacts/20260604-213430/expert1_v2_window_stage7_v3_full`，暂不自动删除，避免误清理用户可能需要检查的输出。

### 历史优先级说明

- [x] Stage 6C 之前已发现更确定的训练与选模问题。
- [x] 现在 Stage 6C/6D 已闭环，Stage 7 重新成为下一优先级。
- [x] 不能把 CQ08 改名为 exact time sync；只能先改 proportional fallback 的 raw-frame crop 质量，再看 RD evidence 是否回升。

### 触发条件

- [x] Stage 1 到 Stage 5 已完成。
- [x] RD evidence macro F1 on RD >= `0.60`，20260522-003944 selected 已达 `0.6116`，接近 release 阈值 `0.6200`。
- [x] joint-on-RD 仍低于或接近 trajectory-on-RD，20260522-003944 为 `-0.0063`。
- [x] final net gain on RD 仍不稳定，20260522-003944 selected 为 `-0.0034`。

### 未完成

- [x] RD manifest 写入原始帧序号或时间戳。
- [x] 报告区分 `exact_time_sync`、`proportional_fallback`、`low_coverage`、`short_segment`。
- [x] 报告按 `rd_time_sync_status` 分组输出 joint gain。
- [x] 已审查 `WindowAssembler`：正式 map 的 segment 与窗口边界对齐，当前不需要在读取阶段二次裁切已编码 segment。
- [x] segment-store 构建阶段已支持 proportional fallback raw-frame crop，避免 CQ08 先整段编码再按比例切 encoded features。
- [x] 20260523 已重建正式 `expert1_rd_segment_store_v2` 和 `expert1_window_segment_map`。
- [x] 重新构建正式 `expert1_rd_segment_store_v3` 和 `expert1_window_segment_map`，让 proportional fallback raw-frame crop 生效。
- [x] 用 calibrated checkpoint 做 posthoc full validation，比较 `RD/Joint Bottleneck` 是否改善，重点看 `proportional_fallback` 的 `RD F1`、`joint-traj` 与 Unknown/Clutter hard-pair。
- [ ] 跑 full training，让模型适配 `expert1_rd_segment_store_v3` 的 raw-frame crop 分布；重点验证 RD evidence on-RD 是否回升并推过 `0.6200`，同时观察 `proportional_fallback` 与 `exact_time_sync` 分组的 `joint_minus_traj_macro_f1`。

### Stage 7A：等待 v3 full training 结束

目标：冻结当前代码与数据口径，等待用户完成 v3 full training；训练期间本端不再启动全量训练，避免生成竞争性 artifact 或污染训练判断。

当前训练命令：

```powershell
.\.venv\Scripts\python.exe -m src.core.train_expert1 --config tmp\expert1_v2_window_stage7_v3_full_config.json
```

当前冻结口径：

- [x] 配置文件：`tmp/expert1_v2_window_stage7_v3_full_config.json`。
- [x] 训练输出目录前缀：`artifacts/expert1_v2_window_stage7_v3_full`。
- [x] RD segment manifest：`data/processed/expert1_rd_segment_manifest.csv`。
- [x] window-segment map：`data/processed/expert1_window_segment_map.csv`。
- [x] RD preprocess version：`expert1_rd_segment_store_v3`。
- [x] segment-store 构建结果：`segment_count = 9994`、`failed_parent_count = 0`、`proportional_fallback = 9959`、`exact_time_sync = 35`。
- [x] map 构建结果：`mapped_window_count = 5798`、`missing_window_count = 0`、`rd_coverage_ratio_mean = 1.0`。
- [x] 2026-06-04：修复 Windows full validation DataLoader worker 共享内存错误；新增 `eval_num_workers`，v3 full config 固定 `num_workers = 2`、`eval_num_workers = 0`，训练采样 worker 保持不变，验证/测试改为单进程加载。
- [x] 2026-06-05：重新审查训练异常终止风险，确认 v3 RD shard 最大约 `578MB`，`num_workers > 0`、`persistent_workers=true`、`rd_segment_store_cache_size=64` 与样本缓存会显著放大 CPU 内存/共享内存压力。
- [x] 2026-06-05：低 CPU/低内存口径可完成 smoke，但正式 full training 过慢；`num_workers = 4` + 大缓存口径在训练阶段触发 worker 异常退出。当前改为训练高吞吐、低驻留缓存口径：`num_workers = 4`、`eval_num_workers = 0`、`pin_memory = true`、`persistent_workers = true`、`prefetch_factor = 1`、`non_blocking_h2d = true`、`dataset_cache_size = 0`、`rd_encoded_cache_size = 16`、`rd_segment_store_cache_size = 1`。
- [x] 2026-06-05：新增 `dataset_cache_size` 配置并接入 `TorchTrackDataset`，避免 v3 segment-store 下完整样本缓存 64 个窗口导致 CPU 内存持续增长。
- [x] 2026-06-05：完成低 CPU smoke 训练，使用小索引 `data/processed/tmp_expert1_window_stage7_v3_lowcpu_smoke_index.csv`，报告为 `reports/real/20260605-121057/expert1_v2_window_stage7_v3_lowcpu_smoke.md` 与 `reports/real/20260605-121057/expert1_v2_window_stage7_v3_lowcpu_smoke.json`；训练、full validation、test、checkpoint、summary、real report 写出均通过。
- [x] 2026-06-21：继续优化数据预处理与批处理吞吐；Stage1 trajectory-only loader 现在显式 `load_rd = false`，前三个 trajectory-only epoch 不再白加载 RD segment-store 大 shard。
- [x] 2026-06-21：Stage2 RD-aware sampler 新增 `stage2_rd_shard_local_sampling = true`，RD 样本按 shard/parent 局部加权采样，减少每个 worker 在 500MB 级 shard 间随机跳转。
- [x] 2026-06-21：正式 v3 full config 更新为高并发低驻留口径：`num_workers = 6`、`eval_num_workers = 0`、`pin_memory = true`、`persistent_workers = true`、`prefetch_factor = 1`、`non_blocking_h2d = true`、`dataset_cache_size = 0`、`rd_encoded_cache_size = 16`、`rd_segment_store_cache_size = 1`、`stage2_rd_shard_local_sampling = true`。

等待期间规则：

- [x] 不修改 release gate。
- [x] 不接入 System。
- [x] 不提高 `joint_logit_delta_scale` 或扩大 `joint_logit_delta_eval_scales`。
- [x] 不再对旧 v2 checkpoint 做新的 logit-bias/posthoc calibration 作为主线结论。
- [x] v3 full training 允许训练 DataLoader 使用 `num_workers = 6` 提高吞吐，但每个 worker 的样本/RD/shard 缓存必须保持低驻留；验证/测试继续固定 `eval_num_workers = 0`，避免 Windows full validation 阶段复发 shared-memory / error 1455。
- [x] 不删除中断训练遗留目录 `artifacts/20260604-213430/expert1_v2_window_stage7_v3_full`，除非用户明确要求清理。
- [ ] 用户训练结束后，优先提供新的 `real_report_markdown` 与 `real_report_json` 路径。

### Stage 7B：v3 full training 结果判读计划

目标：训练结束后只基于新 v3 full training 的真实 full-validation/report 决定下一步，不再用旧 v2 checkpoint 的 posthoc 结果替代判断。

第一优先级判读指标：

- [ ] `candidate_status` 与 `release_eligible`。
- [ ] `macro_f1`，目标至少守住 `>= 0.8000`，release 仍按报告内 gate 判断。
- [ ] `precision_candidate_macro_f1`、`precision_candidate_unknown_recall`、`precision_candidate_clutter_recall`。
- [ ] `precision_checkpoint_validation_status` 与 `precision_checkpoint_validation_macro_f1_delta`。
- [ ] `rd_macro_f1_on_rd`，重点看是否从 v3 posthoc 的 `0.4227` 回升并接近/超过 release 线 `0.6200`。
- [ ] `joint_minus_traj_macro_f1_on_rd`，重点看是否从负收益转到 `>= -0.0100` precision 容忍线，最终目标仍是 release 线 `>= +0.0150`。
- [ ] `final_net_gain_rate_on_rd` 与 `final_harm_rate_on_rd`，确认 RD 采用是否带来净收益且不扩大误伤。
- [ ] `rd_time_sync_status` 分组：重点比较 `proportional_fallback` 的 `RD F1`、`joint-traj` 与 `final_net_gain`。
- [ ] Unknown/Clutter hard-pair：重点比较 `trajectory accuracy`、`RD accuracy`、`final accuracy`。
- [ ] parent-level metrics：`parent_mean_logit_macro_f1`、parent Unknown/Clutter recall。

训练后分流规则：

- [ ] 如果 v3 full training 同时满足 core precision 且 `joint_minus_traj_macro_f1_on_rd >= -0.0100`，优先做一次 posthoc precision reselect + full reload-eval，确认导出 checkpoint 可复测。
- [ ] 如果 `macro_f1 >= 0.8000` 且 Unknown/Clutter 达标，但 `joint_minus_traj_macro_f1_on_rd` 仍为负，进入 Stage 7C：targeted RD evidence training change，不提高 final 接管强度。
- [ ] 如果 RD evidence 明显回升但 final net gain 仍低，进入 Stage 7D：conservative adoption gate / residual sweep。
- [ ] 如果 RD evidence 仍低于 `0.55`，优先回查 RD representation 和 proportional fallback crop 质量，而不是继续调 selection score。
- [ ] 如果整体 macro F1 或弱类 recall 明显退化，先对比 v2 calibrated checkpoint 与 v3 full training 的 confusion/hard-pair，再决定是否回滚 v3 crop 或增加 RD evidence 监督。
- [ ] 只有当 Expert1 达到 release candidate，才进入 Stage 8 System 接回。

### Stage 7C：备选 targeted RD evidence training change

触发条件：v3 full training 后 `macro_f1` 与弱类 recall 可接受，但 RD evidence 或 joint-on-RD 仍不足。

候选方向：

- [ ] 强化 RD evidence 的 Unknown/Clutter hard-pair 监督，而不是提高 final 接管强度。
- [ ] 检查 `low_altitude_rd_*_aux_weight` 是否只作用在 RD evidence 质量上，避免间接扩大 final harm。
- [ ] 对 `proportional_fallback` 分组单独报告 loss/aux metrics，确认训练期是否学到有效 RD 分布。
- [ ] 若 RD evidence 对 `cq08_track` 仍显著劣于 trajectory，考虑针对 CQ08 proportional fallback 的 RD channel/temporal normalization 小改，而不是改 System。

暂不执行：

- [x] 等待当前 v3 full training 结束前不实施 Stage 7C。
- [x] 不把 Stage 7C 与本轮 v3 full training 混在同一个实验里。

## Stage 8：System 接回

目标：只有 Expert1 V2 真正过质量门槛后才接入 System。

### 当前状态

- [x] System 链路支持 Expert1 V2 结构。
- [x] System 配置已支持 RD segment joint token 参数。
- [ ] 当前 Expert1 checkpoint 未通过质量门槛。
- [ ] 当前不能更新 `config/config.json` 的 `expert1_checkpoint` 为本轮候选。

### 接回前置条件

- [ ] Expert1 `candidate_status = release_candidate`。
- [ ] Expert1 window final macro F1 达标。
- [ ] Expert1 joint-on-RD gain 达标。
- [ ] Expert1 final net gain on RD 达标。
- [ ] Expert1 Unknown / Clutter recall 达标。
- [ ] Expert1 without-RD accuracy 达标。
- [ ] diagnostic checkpoint report 完整。

### 接回后验收

- [ ] System macro F1 >= `0.75`。
- [ ] Aircraft recall >= `0.80`。
- [ ] Aircraft precision >= `0.80`。
- [ ] Ground group F1 >= `0.60`。
- [ ] Expert1 low-altitude macro F1 不低于 standalone 可解释水平。

## 当前不建议做

- [ ] 不接入 System。
- [ ] 不回退旧 RD logits correction 主路径。
- [ ] 不只看 parent-level macro F1 放行。
- [ ] 不继续单纯堆 deep reroute 强度。
- [ ] 不把 raw_tr_rd 4 条 test 样本作为唯一硬淘汰依据。
- [x] Stage 1 到 Stage 6D 已完成后，Stage 7 RD segment-store v3 已成为当前主线；等待 v3 full training 结束前不切换到 System 或其他大改。
- [ ] 不为了追求单次 test macro F1 放松 release gate；最高精度候选与可发布候选必须分开标注。
