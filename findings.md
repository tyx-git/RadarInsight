# RD-only 全链路审查结论

更新时间：2026-08-06（Asia/Shanghai）

本轮更新状态：Stage 013–039 历史审查/修复与 full validation 均保留；Stage 040 ParentGate CU-weight=5 full 已由用户完成并经 2710-window validation 验收，因牺牲 Bird 路由 Recall、CU Precision、总体/parent Macro-F1 而拒绝替换旧基线。Stage 041 已修复 weighted-CE 分母、reference provenance、五类路由非劣化守卫和 candidate/qualified checkpoint 语义，smoke/strict reload/105 项 RD 回归通过，当前待用户手动执行 Stage 041 full。五分类 flat、soft 组合和 hard routing 仍未达到总体及五类 90.00% 门槛；oracle-parent 仅是真值父组上限诊断。所有当前模型 `supports_rejection=false`；本轮未读取 test、未接线上 stable/fusion、未修改 TR-only 代码或跟踪文件。

## 1. 核心结论

1. 历史 active RD-only 是严格 DroneTarget/BirdTarget closed-set 二分类，不是五分类，也不是多分支融合：`RD_ONLY_LABELS=("DroneTarget","BirdTarget")`、`PhysicalRDDroneBirdClassifier.label_names=("DroneTarget","BirdTarget")`、`supports_rejection=False`。独立五分类 RD-only 链路已经实现，并已完成 Stage 023 validation；当前指标未达交付门槛。
2. active forward 只接收 RD tensor 及其合法派生量：6-channel RD、raw RD dB、RD time、padding、range/velocity axes、8 项 RD quality。没有 TR input、TR trajectory、TR match、path、source、ID、split 或 label 参数。
3. 没有发现 active 模型输入层面的 TR 泄漏或 test 调参：trainer 只构建 train/val loader，test 被 config/trainer/evaluator 硬拒绝；normalization/feature stats 都是 train-only。
4. parent/content-group 字符串不编码进模型；它们只按相等关系用于 RD-derived grouped split、等权 loss 和 parent 聚合。`parent_split_group_id` 来源是 RD tensor content fingerprint（少数 fallback 为 source+parent），不是 TR fingerprint。
5. active index 虽包含 BalloonTarget/ClutterTarget/UnknownTarget，但 Physical Dataset 只选择 Drone/Bird；这些类不属于本分支 support、loss、阈值或预测。
6. 当前支路没有 Unknown/OOD/reject/abstain 输出，结论必须是：**当前支路不具备已验证的拒识能力**。
7. 用户手动完成的 Stage 011 v9 全量训练，其完整 validation（非 test）满足全部 90.00% 门槛：window Accuracy=97.99%、Macro-P/R/F1/Balanced Accuracy=97.57%，Drone P/R/F1=98.58%，Bird P/R/F1=96.56%；parent Accuracy=98.65%、Macro-P/R/F1/Balanced Accuracy=98.36%。
8. 历史 Stage 011 状态是 **二分类全量训练已完成、best checkpoint 已通过当前代码 strict reload 验收**；训练未由本代理启动，test 未运行。独立全量 evaluator/analyzer 仍待用户执行，因此 source 细表、完整 confidence/calibration 与 retrospective stability 尚无正式全量报告。
9. 新冻结设计是五分类 RD-only：标签严格沿用 `LOW_ALTITUDE_CLASSES = ["DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget", "UnknownTarget"]`；采用共享 RD encoder、软级联概率头和直接五分类 flat 旁路。`UnknownTarget` 是分类标签，不自动等于拒识；reject/abstain 必须独立验证。Stage 023 已有完整 validation，但指标不能声称达到90%。
10. 新冻结融合安全契约是 availability-aware late fusion：RD+TR 融合；仅 RD 严格透传 RD-only；仅 TR 严格透传 TR-only；双缺失输出 `abstain/unavailable`。训练 RD-only 时保留 TR 入口但不读取轨迹、TR 标签、TR 轨迹、TR 匹配结果或 pairing 字段。

## 1.1 Stage 023 五分类全量 validation 结论

- 用户手动完成 Stage 022 full training；checkpoint=`artifacts/20260803-112945/rd-only-five-stage022-full-seed42/best_rd_physical_five_model.pt`，epoch=16，SHA-256=`168068014FEC472D31477FC1167C9A927FC5C88EACEBB2AB14F0B85A8AB3AC43`。
- validation report=`tmp/20260803-rd-five-stage022-full-validation.json`，predictions=`tmp/20260803-rd-five-stage022-full-validation.predictions.csv`；2710/2710 windows，excluded=0，`test_evaluation_executed=false`、`tr_inputs_loaded=false`、`supports_rejection=false`。
- Flat window：Accuracy=93.17%、Macro-P=86.00%、Macro-R=82.96%、Macro-F1=84.15%、Balanced Accuracy=82.96%。未达到 90.00%。
- Flat per-class：Drone 97.99/99.76/98.87%，Bird 91.68/82.22/86.69%，Balloon 92.66/99.84/96.12%，Clutter 77.01/58.77/66.67%，Unknown 70.66/74.21/72.39%（顺序均为 P/R/F1）。
- Parent=221：Drone 98.13/100.00/99.06%，Bird 90.00/83.72/86.75%，Balloon 94.34/100.00/97.09%，Clutter 66.67/50.00/57.14%（support=8，low stability），Unknown 80.00/80.00/80.00%（support=15，low stability）。
- Flat confusion 行真值、列预测：`[[1266,3,0,0,0],[24,430,34,10,25],[1,0,644,0,0],[1,16,6,67,24],[0,20,11,10,118]]`。Bird→Drone=24、Bird→Balloon=34、Bird→Unknown=25、Bird→Clutter=10；Clutter↔Unknown 双向=24/10。
- Cascade window Macro-F1=83.71%、Bird Recall=81.26%、Clutter Recall=58.77%、Unknown Recall=74.21%，低于 flat；不切换 cascade。
- 与 Stage 011 二分类 validation 的 Bird Recall 96.56% 比较，五分类下降14.34个百分点；pairwise 仅在 Drone/Bird 两列比较得到 Bird recall=95.41%，不能作为五分类能力结论。
- 错误置信度与校准诊断：Bird→Drone/Balloon/Unknown 平均最大概率约95.57%/96.22%/96.78%；Clutter→Bird/Unknown约98.28%/96.84%；Unknown→Clutter/Balloon约99.84%/99.94%。overall ECE约0.0630，Bird约0.1652，Clutter约0.3850，Unknown约0.2512；未做 post-hoc calibration。
- 错误 parent 集中：Bird 93 个错误来自13/43 parents；Clutter 47来自5/8（`cq08|track|4198` 为19/19）；Unknown 41来自7/15。该事实支持 parent/场景异质性假设，但不证明具体物理根因。

## 1.2 已证实问题与待核实假设

1. **已证实：五分类候选未达交付门槛。** 2710 validation windows 的 Macro-F1=84.15%，Bird Recall=82.22%、Clutter Recall=58.77%、Unknown Recall=74.21%；best selector 的 `eligible=false`。这是指标事实，不是代码缺陷。
2. **已证实：五分类训练没有保留旧二分类的 Bird 专项权重。** `config/rd_only/rd_physical_five_full_seed42.json:23` 是五个 class weight 全1，旧二分类配置使用 Bird weight=1.35；该差异可复现，但是否造成全部下降需单因素实验。
3. **已证实：多头目标同时优化。** `src/core/rd_physical_five_trainer.py:297` 同时优化 flat、cascade、parent、prefix、micro 五个目标；当前没有证据证明这些辅助目标对 Bird 或 Clutter/Unknown 有益。它是可执行消融点，不应写成已确认损坏。
4. **已修正：旧权重审计不能代表当前代码。** Stage 023 历史复算曾得到 Clutter=1.0597、Unknown=0.7016、Bird=0.1246，但 Stage 033 已移除 parent/content ID 权重，当前 evidence-only train mean 为 Drone=0.8783、Bird=0.8996、Balloon=0.8987、Clutter=0.9095、Unknown=0.8631；该历史数字不再作为当前根因证据。
5. **已证实：错误不是简单低置信度。** 多个错误方向的平均最大概率在90%–99.94%，因此仅改 reject/confidence threshold 无法恢复五分类 recall。
6. **已证实：UnknownTarget 不是拒识。** `cq08_manifest.py` 的 label_id=6 是原始分类映射；evaluator `supports_rejection=false`、`unknown_target_is_class=true`、`abstain_output=false`。当前支路无已验证拒识能力。
7. **已审计：Clutter/Unknown label semantics 与 parent 一致性。** `tmp/20260805-rd-five-label-semantics-audit.json` 显示 Clutter 的 MAT 文本统一为 `Clutter`、Unknown 统一为 `Other`，train/val frame-manifest label mismatch=0、parent 多标签/跨 source/track 冲突=0、train/val group/content-group intersection=0；`Other` 仍不是 OOD/reject 证据。
8. **待核实：parent/evidence weighting 的长尾放大是否损伤共享特征。** 需只改变该因素的独立训练和 grouped validation。
9. **待核实：Bird 下降主要来自五类竞争还是共享编码退化。** pairwise Bird recall=95.41% 表明竞争有贡献，但不能排除编码/辅助 loss 影响。

## 1.3 Stage 032 全链路审查新增结论

### Stage 032 已确认问题（修复前记录；修复状态见 Stage 033–034）

1. **严格泄漏边界问题：parent/content-group ID 参与了 loss 权重。**
   - 证据：`src/core/rd_physical_five_trainer.py:58-105` 和 `src/core/rd_physical_specialist_trainer.py:43-65` 从 `parent_sample_id`、`parent_split_group_id` 建立 `parent_balance`/`evidence_sum`，再在 `_weights`/`_run_epoch` 中进入 CE 权重；训练 summary 只披露 `tr_inputs_loaded=false`，但这不等于 ID 未进入 loss。
   - 影响：若按用户当前“目标 ID、配对信息不得进入损失、采样或后处理”的严格定义，该实现不满足契约；它可能改变各 parent/group 对梯度的贡献，不能仅因原始字符串未拼进 tensor 就判定安全。
   - 边界：这些 ID 当前没有进入 forward 数值，也不是 TR pairing 字段；是否允许“仅用于等权 parent/content group”需要用户确认，不能擅自判作可接受或泄漏。

2. **旧二分类模型仍消费 pairing-derived `canonical_time` quality。**
   - 证据：`src/core/data/rd_physical_window_dataset.py:188-203` 将 `rd_time_sync_status == canonical_parent_association` 编为第8个 quality；`PhysicalRDDroneBirdClassifier.quality_projection` 消费 8 维 quality；旧 feature stats/artifact 明确包含 `canonical_time`。
   - 影响：旧二分类虽然没有 TR tensor，但把 RD/TR 关联语义的派生字段带入模型输入，违反本轮 RD-only 严格边界；旧 checkpoint 不能在静默改 schema 后继续使用，修复需独立 schema/artifact 和重训或明确只读历史兼容。

3. **旧二分类 physical axis runtime contract 弱于五分类。**
   - 证据：`src/core/data/rd_physical_window_dataset.py:365-376` 只检查 range/velocity shape、finite；`src/core/rd_physical_drone_bird_model.py:214-228` 检查 time 但不强制 range/velocity 严格递增/端点/单位；五分类 `src/core/rd_physical_five_model.py:151-156` 已有严格递增检查。
   - 影响：旧链可在错误转置、翻转或尺度漂移被数据层接受时继续前向，导致 range/Doppler 物理语义错位和 Drone/Bird 泛化风险。当前 v9 artifact 数据轴审计正常，因此是契约缺口，不是已证实当前 tensor 错位。

4. **五分类/模型允许 interior padding，时间位置和末帧语义可能与压缩序列不同。**
   - 证据：`PhysicalRDFiveClassifier.forward` 只对 `valid[:,i] & valid[:,i+1]` 检查时间，GRU 使用 packed length（隐式取前 `lengths` 帧），而 `collate_physical_rd_five_windows` 生成 suffix padding；手工 forward 对 mask `[False,True,False]` 可执行，且与压缩为两个有效帧的输出最大 logit 差异约0.359（随机契约输入），说明 interior mask 未被拒绝且语义不等价。
   - 影响：未来数据若出现中间缺帧/错误 mask，GRU 可能把 padding 帧当有效 prefix，attention/statistics 与 frame/micro 分支语义不一致，影响时间因果和线上稳定；当前生成 index 未发现该结构，故需作为输入契约修复。

5. **specialist checkpoint 选择没有 Drone/Bird 或局部双类非劣化门槛。**
   - 证据：`src/core/rd_physical_specialist_trainer.py:148-149` 只比较 `val_metrics["macro_f1"]`；配置没有 reference metrics/non-degradation 字段。ParentGate/DBB/CU smoke 均未完整评估，无法证明组合后不牺牲 Drone/Bird 或 Clutter/Unknown。
   - 影响：specialist 可能以局部 Macro-F1 提升换取某一类别 recall/precision 下降；soft 组合后会把该退化传播到五分类，违反“不牺牲 Drone/Bird”和主要指标门槛。

6. **specialist base encoder 包含未用于对应主损失的共享 head，增加未验证容量和梯度路径。**
   - 证据：`PhysicalRDSpecialistClassifier` 为每个 specialist 新建完整 `PhysicalRDFiveClassifier`；DBB/CU 只取 `flat_logits` 的局部列，ParentGate 只取 `parent_logits`，但 base 同时计算 flat/parent/child/frame/micro heads；specialist trainer 只对 `output.logits`、末帧 frame logits 和 micro logits 计算 loss，child/flat 等其他参数仍存在但不被 specialist 目标直接监督。
   - 影响：未使用 head 占用参数和显存，并可能产生未验证的共享 trunk/头部行为；不是当前精度下降的已证实根因，但应在修复阶段单独消融/精简，避免复杂度掩盖归因。

7. **physical RD 五分类和 specialist 适配器尚未接入真实生产线上调用链。**
   - 证据：`rg` 结果显示 `route_rd_tr_five` 和 `compose_rd_specialists` 只有定义/测试引用；`System._forward_experts` 仍调用 `LowAltitudeSuspectExpert` 的 TR/TR+RD v2 路径，physical evaluator 输出 `online_stable_decision.available=false`。
   - 影响：不能把当前 specialist/flat validation 当作线上 stable 或 TR+RD 融合能力；线上缺失模态安全状态机目前只存在独立适配器，未完成接入。

### Stage 033 已执行修复

1. **已修复 ID 进入 loss 权重边界。** 五分类 `src/core/rd_physical_five_trainer.py` 与 specialist `src/core/rd_physical_specialist_trainer.py` 已移除按 parent/content ID 查表的训练权重；新增 `_evidence_weights` 仅使用窗口自身的 prefix、有效帧数和 `rd_present`。ID 仍只用于 split 审计、排序和 parent 聚合报告。对应 ID 改写不变性测试已通过。
2. **已修复 padding/轴运行时契约。** 旧二分类与五分类 forward 均增加 range/velocity shape、finite、严格递增及 suffix-only padding 检查；interior padding 显式拒绝。相关 53 项 RD/fusion/entry/artifact/window/contract 测试通过。
3. **已建立旧二分类 quality schema 隔离。** `rd_physical_window_dataset.py` 新增 `legacy_v4` 与 `rd_only_v5`；历史 v4 artifact/checkpoint 默认显式走 legacy 兼容，clean schema 只包含七项 RD quality，不包含 pairing-derived `canonical_time`。`rd_physical_artifacts.py` 按 schema 严格匹配 feature-stats、generator hash 和 quality_dim，禁止静默复用旧 checkpoint。
4. **clean train-only feature-stats artifact 已生成并已核验。** 其 schema、index/normalization/generator provenance 和 `test_evaluation_executed=false` 均通过；文件 SHA-256 记录见“Stage 033–034 修复后证据”。

### Stage 033–034 修复后证据

1. **clean artifact 已生成并通过 provenance 核验。** `data/processed/expert1_rd_physical_feature_stats_v5_clean.json` 的 SHA-256=`27A555B96F91FE3EAA837EDF36597A438F0D24FA0571A6A50DA62DD7B9E6F1CA`；schema=`rd_physical_feature_stats_v5_clean`、split=`train`、labels=`DroneTarget,BirdTarget`、quality 为7项且不含 `canonical_time`，index/normalization/generator hash 与实际文件一致，`test_evaluation_executed=false`。
2. **旧二分类 clean loss/选择已去除 ID 依赖。** `train_physical_rd_drone_bird` 和 evaluator 使用 `compute_window_evidence_weights`，输入只依赖当前窗口 `is_prefix_window/window_frame_count/rd_present`；clean `_is_better(..., use_parent_metrics=False)` 不要求 parent metrics；`compute_parent_evidence_weights` 仅保留 legacy/报告测试用途。ID 改写不变性新增测试通过。
3. **specialist 已改为局部 head。** `PhysicalRDFiveClassifier(head_mode=parent_gate|dbb|cu)` 只构建本地 parent/flat/frame/micro 输出；specialist v2 manifest 记录 `head_mode` 和 `local_class_count`，三套 strict reload 分别为2/3/2类，未再通过 global five-class logits 切片。
4. **specialist checkpoint 选择已改为逐类非劣化。** 配置固定来自已有 validation 的 reference metrics；`_selection_record` 对 DBB 的 Drone/Bird/Balloon、CU 的 Clutter/Unknown、ParentGate 的 DBBGroup/CUGroup 逐类检查，任一类失败则 `eligible=false`。本次受限 smoke 分别为 `eligible=false`，因为每个 smoke 仅4个 val windows，不能作性能结论。
5. **回归证据。** `.venv\\Scripts\\python.exe -m pytest -q tests/test_rd*.py` = `93 passed, 1 warning`；`compileall -q src/core src/process config` 通过。clean 二分类、五分类、ParentGate、DBB、CU smoke 均完成，summary 明确 test=false、TR inputs=false；三 specialist v2 evaluator reload 输出 `tmp/stage034-parent-reload.json`、`tmp/stage034-dbb-reload.json`、`tmp/stage034-cu-reload.json`，均 `full_split_evaluated=false`。
6. **历史 checkpoint 边界。** Stage011 旧二分类 checkpoint 仍含 `legacy_v4`/`canonical_time` 且旧 inference hashes；当前 strict loader 明确报 `physical RD checkpoint inference contract does not match active artifacts or code`，未静默兼容。需要 clean 二分类能力时必须使用新配置重新训练。

### 已确认但属于性能/能力阻塞（不是代码泄漏缺陷）

8. **五分类 full validation 未达到目标。** `tmp/20260803-rd-five-stage022-full-validation.json` 与 `training_summary.json` 一致：window N=2710，Accuracy=93.17%、Macro-F1=84.15%、Macro-Recall=82.96%；Bird P/R/F1=91.68%/82.22%/86.69%，Clutter=77.01%/58.77%/66.67%，Unknown=70.66%/74.21%/72.39%。best selection `eligible=false`。

9. **级联头没有改善当前 flat 基线。** 同一 validation split 上 cascade Accuracy=92.99%、Macro-F1=83.71%、Bird Recall=81.26%，均不高于 flat；配置 `deployment_head=flat`，因此当前部署头保留 flat。

10. **当前 RD-only 不具备已验证拒识能力。** 五分类 artifact/evaluator 均声明 `supports_rejection=false`、`unknown_label_is_rejection=false`、`abstain_output=false`；不能输出 Coverage/FAR/FRR/Selective Risk 的有效开放集结论。

### 待核实假设（不作为已确认缺陷）

1. ID/group 等权策略是否应作为“只用于 split/聚合”的合法例外；若不允许，需重新设计不依赖 ID 的 loss/sampler。
2. evidence-only train weight 的小幅类别差异（Drone=0.8783、Bird=0.8996、Balloon=0.8987、Clutter=0.9095、Unknown=0.8631）是否仍影响共享表示，需单因素训练，不得凭机制直接宣称因果。
3. dB-domain resampling、常量 quality 特征、Bird class weight 和多头 loss 各自对 Bird/Clutter/Unknown 的影响；均需独立配置、grouped validation 和停止条件。
4. `UnknownTarget`/`ClutterTarget` 的物理语义是否足够稳定仍有限；label_id、MAT 文本与 train/val parent 一致性已审计通过，但 Unknown 原始文本为 `Other`，不把标签名当作开放集真值。

## 1.4 修复前阻塞（历史快照，已由 Stage 033–034 覆盖）

- 用户随后批准并要求“开始修复”；Stage 033–034 已完成 clean artifact 核验、三套 specialist v2 smoke/strict reload、全 RD 回归和 compileall。
- 该条为历史快照：Stage 035 full training 与 Stage 036 完整 validation/组合诊断已由用户/本轮完成；仍未执行 test evaluation、融合权重拟合或线上 physical RD stable/fusion caller 接入。

## 2. Active 调用链与分支边界

### 2.1 Active Physical v9

```text
src/core/train_rd_physical_drone_bird.py
  -> config/configmanager.py::load_config_file
  -> PhysicalRDDroneBirdConfig
  -> PhysicalRDWindowDataset(split=train|val)
  -> collate_physical_rd_windows
  -> load_physical_rd_artifacts
  -> PhysicalRDDroneBirdClassifier
  -> train_physical_rd_drone_bird
  -> strict v9 checkpoint/evaluator/analyzer
```

实际输入契约：

| 输入 | shape/语义 | 是否进模型 | 泄漏结论 |
|---|---|---:|---|
| `features` | `[B,T,6,21,384]`，RD 派生通道 | 是 | 合法 RD |
| `raw_amplitude_db` | `[B,T,21,384]` | 是，物理统计与 micro profile | 合法 RD |
| `time_seconds` | `[B,T]`，累计 CPI acquisition time | 是 | 合法 RD；非 TR wall-clock |
| `padding_mask` | `[B,T]`，True=padding | 是 | 合法结构量 |
| `range_offset_m` | 21 bins，`[-75,75] m` | 是 | 合法 RD 物理轴 |
| `velocity_mps` | 384 bins，`[-85,85] m/s` | 是 | 合法 RD 物理轴 |
| `quality_features` | `[B,8]`，均由 RD present/phase/coverage/frame/time/dynamic range/canonical metadata 派生 | 是 | 合法 RD；多项为常量 |
| `label_name` | Drone/Bird | 否；只用于 target/loss/metrics | 无输入泄漏 |
| `sample_id/parent_id/group_id` | 字符串关联键 | 否；仅分组权重/聚合/报告 | 不编码进 forward |
| `source_name` | `cq08_track` | 否；仅审计/报告 | 不进模型 |
| `split_name` | train/val/test | 否；只选 loader | test 未加载 |
| path/file name | tensor 定位 | 否；只做 IO | 不进模型 |
| TR/trajectory/match | 不存在于 active forward | 否 | 未发现 TR 泄漏 |

### 2.2 其他 RD 分支必须分开

- legacy `rd_drone_bird_*`：Drone/Bird 二分类，但依赖旧 v7/TR-window anchor 和 Expert1 数据链；不能证明 active Physical 能力。
- `rd_patch_residual_*`：旧 sequence checkpoint + patch residual/gate；按 window F1 选 checkpoint；不能与 active v9 混算。
- `rd_novel_parent_*`：旧 Physical-v2 novel-evidence parent state；两类、独立 artifact。
- `rd_novel_anchor_residual_*`：trainer 已 retired，入口会抛错。
- Expert1/System：五分类包含 Balloon/Clutter/Unknown，且有可选 reject/stable 逻辑；它不是 active Physical 二分类，任何结果都不能移植。

## 3. 数据、split 与统计事实

### 3.1 数据版本与哈希

| Artifact | 版本/数量 | SHA-256 |
|---|---|---|
| active window index | 22085 rows | `c9cbf3ff9a354766583df3a896594420b65f566206b65b80c2c7522e572d19c1` |
| normalization | v4，train-only | `2f28dc25f0c3454beadb258c25b5a3f096edc4cebbb842060b5130642aba0e5b` |
| feature stats | v4，train-only | `ecaf684947467fb76672dd0b77ed380f41a9d6fc034c8ee0a59cb2a72fd428e9` |
| physical frame index | 106026 frames | `0a4176660c85ecd6298cd505070f681b7ee880b1b9ed9e73b05ff2606d656c59` |
| split ancestor | v8 RD-content | `d4b1b8a2f481ac30ef288cd5e14ebc9e293b5636407db86665fb3d2e2d560ca2` |
| canonical tensors | 1549 files / 2.77 GB | `596b9df5764665c6f91635dd4af113939fb486ca6122841be728f927485ef0b6` |

### 3.2 Active Drone/Bird 支持数

| Split | Drone windows | Bird windows | 合计 | trainer 是否加载 |
|---|---:|---:|---:|---:|
| train | 5542 | 2665 | 8207 | 是 |
| val | 1269 | 523 | 1792 | 是 |
| test | 1188 | 632 | 1820 | 否 |

- 全 index trainable 还包含 Balloon/Clutter/Unknown；active Dataset 的 label filter 将其排除。
- parent cross-split=0；content-group cross-split=0；content-group label conflict=0。
- active source 只有 `cq08_track`；index 没有 `session_id` 或 `session_name`，因此没有合法 session-level 指标。
- `rd_fusion_eligible=0`；active time semantics 全为 `canonical_sensor_relative_elapsed`。
- 22085 总窗口中 17797 trainable、4288 因少于 12 个 RD frames 排除；排除规则在建索引时固定，不是按预测难度删除。

### 3.3 Canonical/时间/缺失

- 106026 frame merge audit 中 missing reference、unexpected、duplicate、tensor index mismatch、missing tensor、invalid shape、partial range coverage、partial velocity coverage 全为 0。
- amplitude 来自 `data_proc_MTD_result_db`；phase 必须来自 complex `data_proc_MTD_result`，缺 phase 会失败，不能补零。
- range/velocity 均严格递增，dataset 对 shape、finite、time strictly-increasing 做运行时拒绝。
- 当前 build 直接在 dB 表示上做插值/anti-alias，随后统计与 micro 分支按 `10^(dB/10)` 处理功率。是否改为 linear-power-domain resampling 尚无能量基准和独立 validation 证据，因此是后续假设，不列为已确认缺陷。
- 原 builder 在缺 `detected_range_m` 时 fallback 到 `track_range_m`；虽然现有 106026 帧从未触发，这会给未来数据引入轨迹参考风险。本轮已删除 fallback 并测试。

### 3.4 Train-only statistics 与质量特征

- normalization：713 train parents、712 RD content groups、labels 顺序 Drone/Bird、test=false。
- feature stats：8207 train windows、713 parents、712 groups、14 statistics、test=false。
- normalization/feature stats 均绑定 active index SHA；feature stats 还绑定 normalization SHA 与 generator code SHA，strict loader 会拒绝漂移。
- 8 项 quality 都进入模型，但 active complete-present 数据使 `rd_present/phase_complete/multimodal/coverage/canonical_time` 等为常量；`dynamic_range_proxy` 也因 clip 后 `unique_count=1`。它们主要等价于 bias，不能被描述为已验证性能来源；变化量主要来自 frame count 与 duration。

## 4. 模型、mask、loss 与 checkpoint

### 4.1 模型前向

- 6-channel anisotropic `(3x7)` CNN；range 维不下采样，Doppler 逐层 stride 2。
- spatial avg/max pool 到 `(6,16)` 后投影到 token_dim=256。
- time projection + causal GRU；使用 final hidden、valid mean、valid max、masked attention。
- 14 项物理统计和 physical velocity-time micro profile 接入最终 classifier。
- frame head 产生 `frame_logits`；末个有效 frame 用于 prefix aux CE。
- micro head 产生 `micro_logits`；进入最终 classifier，同时有 micro aux CE。
- padding frame 在 token、GRU packing、mean/max/attention、frame logits、statistics、micro profile 中均被屏蔽；all-padding 路径保持 finite。

### 4.2 Loss 与权重

```text
total = main_focal_CE
      + 0.15 * final_valid_frame_prefix_focal_CE
      + 0.25 * micro_focal_CE
```

- `bird_loss_weight=1.35`、`focal_gamma=1.0`。
- content-group/parent/evidence weighting 先避免长 parent 和重复 RD content 主导，再额外施加 Bird class weight；这是已披露的训练策略，不改变评估分母。
- 每个 head 的 CE 只计算一次；没有对同一 logits 重复 main classification loss。
- missing RD mask 使 loss/metrics/aggregation weight 为 0；full config `include_missing_rd=false`。
- optimizer=AdamW；无 scheduler、AMP、early stopping。
- trainer 新增 non-finite loss/gradient 硬失败；Stage 011 的 18 个 epoch train/val loss 与 gradient audit 全部有限，最大 batch pre-clip gradient norm=674.6139（epoch 8），现有 clip max_norm=1.0。该峰值是剩余稳定性风险，不等于训练失败。

### 4.3 Checkpoint

- schema=`physical_rd_drone_bird_checkpoint_v9`。
- 保存 epoch、完整 config、artifact manifest、model state 与 validation。
- loader 对 task/model/data/inference code hash 和 state dict 做 strict 校验。
- Stage 011 best checkpoint（epoch 13，SHA-256=`07A14677016D9E0A39208277F63ED5E7BB6F1311D12131A98C78953636443BF2`）在当前代码下 strict reload + 受限 evaluator 成功。
- 未保存 optimizer/RNG/latest training state，不能从中断 epoch 无缝恢复；这是可靠性限制，不影响一次性挂机训练入口。

## 5. 完整 validation 性能

以下均为 validation，不是 test；百分比由真实 `training_summary.json` 计算，未使用 smoke 替换。

### 5.1 当前 Stage 011 v9 总体指标

| 层级 | N | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Balanced Accuracy | >=90% |
|---|---:|---:|---:|---:|---:|---:|---:|
| window | 1792 | 97.99% | 97.57% | 97.57% | 97.57% | 97.57% | 是 |
| parent weighted-mean-logit | 148 | 98.65% | 98.36% | 98.36% | 98.36% | 98.36% | 是 |
| parent unweighted-mean-logit | 148 | 98.65% | 98.36% | 98.36% | 98.36% | 98.36% | 是 |

### 5.2 当前 Stage 011 v9 分类别指标

| 层级 | 类别 | Support | Precision | Recall | F1 | 三项均 >=90% |
|---|---|---:|---:|---:|---:|---:|
| window | DroneTarget | 1269 | 98.58% | 98.58% | 98.58% | 是 |
| window | BirdTarget | 523 | 96.56% | 96.56% | 96.56% | 是 |
| parent weighted/unweighted | DroneTarget | 105 | 99.05% | 99.05% | 99.05% | 是 |
| parent weighted/unweighted | BirdTarget | 43 | 97.67% | 97.67% | 97.67% | 是 |

### 5.3 当前 Stage 011 v9 混淆

window，row=true、column=pred：

| true/pred | DroneTarget | BirdTarget |
|---|---:|---:|
| DroneTarget | 1251 | 18 |
| BirdTarget | 18 | 505 |

- DroneTarget→BirdTarget=18。
- BirdTarget→DroneTarget=18；双向错误对称，没有以显著牺牲其中一类换取另一类召回。

parent：

| true/pred | DroneTarget | BirdTarget |
|---|---:|---:|
| DroneTarget | 104 | 1 |
| BirdTarget | 1 | 42 |

### 5.4 source/session/stable

- source：训练 summary 证明 active train/validation 的唯一 source 是 `cq08_track`；独立新格式 evaluator 尚未全量运行，因此没有单独生成的 source report，且单一 source 不能证明跨源泛化。
- session：未评估且不可用，原因是 index 无 session 字段；不得以 148 parents 冒充 sessions。
- online stable：未评估且不可用，active Physical 未接生产 stable state machine。
- analyzer 的 cumulative-prefix prediction 是 causal；其“从何时起不再改变”必须看完序列后回溯，只能报告 retrospective stability，不能当线上 latency。
- confidence distribution/完整 ECE：当前 Stage 011 training summary 没有逐样本 full prediction report；本轮只用 8-window reload smoke 验证入口，不能得出完整校准结论。

### 5.5 v8 历史对照

| 版本 | 层级 | Accuracy | Macro-P | Macro-R | Macro-F1 | Bird Recall |
|---|---|---:|---:|---:|---:|---:|
| v8 seed-42 | window | 99.16% | 99.18% | 98.79% | 98.98% | 97.90% |
| 历史 v9 seed-42 | window | 97.54% | 97.80% | 96.24% | 96.98% | 93.12% |
| Stage 011 v9 seed-42 | window | 97.99% | 97.57% | 97.57% | 97.57% | 96.56% |
| v8 seed-42 | parent | 99.32% | 99.53% | 98.84% | 99.17% | 97.67% |
| 历史 v9 seed-42 | parent | 99.32% | 99.53% | 98.84% | 99.17% | 97.67% |
| Stage 011 v9 seed-42 | parent | 98.65% | 98.36% | 98.36% | 98.36% | 97.67% |

- Stage 011 相比历史 v9：window Macro-F1 提高约 0.59 个百分点，Bird Recall 提高约 3.44 个百分点；Drone Recall 从 99.37% 降到 98.58%，仍高于目标。相比 v8，Stage 011 window Macro-F1 仍低约 1.41 个百分点，Bird Recall 低约 1.34 个百分点，parent Macro-F1 低约 0.81 个百分点。
- v8→v9 曾同时改变 pool/attention/token/dropout/prefix/micro/Bird weight/focal，且不同 run 还包含随机训练波动；不能把上述差异归因于任何单一改动。
- 当前代码是 v9 schema，v8 checkpoint 不能被 active strict loader 复载；所以 full 交付保持已验证 v9 配置，不擅自回滚到不可执行的旧 schema。

## 6. 拒识与校准

### 6.1 实现结论

- active logits 只有 Drone/Bird 两维。
- 没有 UnknownTarget、ClutterTarget 输出。
- 没有 reject threshold、energy/OOD score、prototype distance、margin gate 或 abstain action。
- `supports_rejection=False` 同时出现在 model 和 checkpoint task contract。
- Expert1/System 中的 reject/Unknown 配置属于另一条五分类链，不能作为 active Physical 能力。

### 6.2 可报告指标

| 指标 | 结果 | 原因 |
|---|---|---|
| threshold | 未定义 | active 无拒识阈值 |
| Coverage | 未评估 | 无 accept/reject 动作 |
| Accepted Support/Accuracy/Macro-F1 | 未评估 | 无 accepted subset |
| Drone/Bird accepted Recall | 未评估 | 无拒识动作 |
| Unknown/Clutter rejection Recall | 未评估 | active 无这些标签/输出 |
| False Acceptance Rate | 未评估 | 无 unknown denominator 与 accept 动作 |
| False Rejection Rate | 未评估 | 无 reject 动作 |
| Selective Risk | 未评估 | 无 coverage-risk curve |

结论：**当前支路不具备已验证的拒识能力**。没有真实 open-set/reject label 和动作，不能得出用户要求的 90%/10% 拒识目标结论。

calibration analyzer 只计算未校准 softmax 的 validation 诊断；没有应用 temperature/isotonic 等 post-hoc calibration，也没有选择阈值。8-window smoke ECE 不能代表完整 validation。

## 7. 已确认问题、修复和剩余限制

### 7.1 已修复

1. evaluator 缺 Macro-P/Macro-R/Balanced Accuracy、分类别 confusion direction、source/session/completion/rejection 声明；已补齐并测试。
2. analyzer 把完整序列后回溯的最后翻转时间命名为 stable decision，可能被误认为 online；已改为 retrospective 并显式声明 online unavailable。
3. future canonical builder 允许 `track_range_m` fallback；已强制 `detected_range_m` 并测试，现有 store 无需重建。
4. trainer 对 non-finite loss/gradient 无硬失败与日志；已增加 guard/gradient audit/progress artifact。
5. 缺独立可执行 full/smoke 配置；已创建并实载验证。
6. 四态融合双缺失/混合 batch 原先存在零 logits 哨兵可能被误用的契约风险；现在不可用或混合 batch 的 `RDFusionDecision.logits` 为 `None`，每行通过 `logits_available` 与全零 abstain probabilities 明确标识，`require_all_logits()` 会拒绝不可用行，新增回归测试覆盖。

### 7.2 已确认但未阻塞的限制

1. quality 多个维度在 active complete-present train 上为常量，`dynamic_range_proxy` 饱和；它们没有已证明的判别增益。
2. Stage 011 window/parent 指标仍低于历史 v8，且 v8→v9 改动多因素耦合；需要后续 one-factor 实验，不能跨 run 归因。
3. 无 session 字段、无独立 production stable 路径、无 open-set/rejection；这些能力不能报告为已具备。
4. 无 optimizer/RNG resume、scheduler、AMP、early stopping；挂机中断需要从头训练。
5. seed=42 已固定，但代码未强制 bitwise deterministic CUDA algorithms；多次训练可能有小幅非确定性。
6. Stage 011 全量 history 的最大 pre-clip gradient norm=674.6139，全部 finite 且现有 clipping=1.0；后续实验需继续监控峰值及其与 val loss 波动的关系。
7. 单一 source=`cq08_track`，不能证明跨 source 泛化；source-level 等于总体不是多域证据。

### 7.3 待核实假设，不作为已确认缺陷

1. dB-domain 重采样是否应改为 linear-power-domain：需要新 store/schema、能量守恒测试和 grouped validation；本轮禁止全量重建，现有表示又有完整一致契约，所以不阻塞。
2. dynamic-range scaling 重设计是否提升 Bird Recall：必须 train-only 拟合、独立配置、一次只改这一项，且 Drone/Bird 任一指标下降即停止。

## 8. 本轮验证与 readiness

- 最终 RD 专项回归：29 passed、0 failed（含五类 contract、融合四态、TR-only 只读入口契约、causal/pairing/split 隔离、旧 RD dataset/artifact 回归）。
- compileall/py_compile/git diff check：通过。
- 最终训练 smoke：`artifacts/20260802-003631/rd-only-stage010-v9-smoke-seed42`；1 epoch、2 train、2 val、test=false、status=completed。
- train smoke 两类 support：Drone=5、Bird=3；loss/gradient finite；checkpoint SHA-256=`69965159c42d0911306ea943ce2e3ac5ef76a4fb78436b38494da3f1f4c56c30`。
- strict evaluator：8 个 balanced val windows、Drone=4/Bird=4、excluded=1784、明确 incomplete/no performance claim。
- analyzer：retrospective/online、calibration/threshold、rejection 边界通过。
- latency：1 warmup+2 samples，入口通过；样本量不足，不作 SLA 声明。
- Stage 011 full artifact：`artifacts/20260802-010756/rd-only-stage011-v9-full-seed42-18epochs`；progress=completed、18/18 epochs、best epoch=13、test=false、supports_rejection=false。
- Stage 011 best checkpoint SHA-256=`07A14677016D9E0A39208277F63ED5E7BB6F1311D12131A98C78953636443BF2`，在当前代码下 strict reload 通过；受限 report SHA-256=`37939F5CB56BBE469356D1EA222F63430F195D74A83F6CDBA0681FA15866F5B2`。
- full config：`config/rd_only/rd_physical_drone_bird_v9_full_seed42.json`。
- 当前状态（历史快照）：**该记录产生时五分类全量训练尚未启动；随后用户已手动完成 Stage 022，best checkpoint 已在 Stage 023 验收；本代理未启动任何全量训练。**
- 五分类 full 训练交付目录及命令已在 Stage 022 准备；随后用户已手动生成 full checkpoint，当前不再输出新的全量训练命令。
- 独立完整 evaluator 已运行并覆盖 2710/2710 validation；test 继续关闭，session、online stable、校准拟合和拒识仍不可声称已具备。

## 9. 五分类设计冻结（实施前）

### 9.1 标签和输出契约

- 规范标签集合及顺序固定为 `LOW_ALTITUDE_CLASSES`：`DroneTarget`、`BirdTarget`、`BalloonTarget`、`ClutterTarget`、`UnknownTarget`。
- 用户消息中的 `UnknowTarget` 是拼写变体；代码和所有新 artifact 必须使用 `UnknownTarget`，遇到变体应在数据契约阶段失败而不是静默映射。
- 五分类 RD-only 的 forward 输入边界仍只允许 RD tensor、RD 物理轴、RD 时间、padding/mask、RD quality/statistics/micro-Doppler 等合法派生量。label 只进入 target/loss/metrics；ID、路径、文件名、source、split、TR 字段和 pairing 状态不进入 forward。
- `UnknownTarget` 为普通五分类类标签；低置信度 reject、OOD、abstain 是独立动作，必须有独立标签/阈值来源/验证报告，不能从存在 Unknown 类推导“具备拒识”。

### 9.2 为什么冻结“软级联 + flat 旁路”

- 硬级联先做“目标/非目标”再做子类会把上游错误不可逆地传给下游，且容易把 Unknown/Clutter 的边界混入人工阈值；因此训练中不使用 argmax 产生中间标签。
- 软级联保留父分布和条件子分布的概率不确定性，可检查 Drone/Bird 关键边界；直接 flat head 保留五类全局判别旁路，便于比较是否级联约束造成 Balloon/Clutter/Unknown 或 Drone/Bird 性能下降。
- 在未有五分类验证结果前，不能宣称级联优于 flat；两者必须独立输出、独立记录、按同一 split 和分母比较。

### 9.3 融合安全状态机

| RD 可用 | TR 可用 | 输出来源 | 约束 |
|---|---|---|---|
| 是 | 是 | 校准后的 late fusion | 只用两模型五类概率/logits和合法质量；权重由 train/OOF validation 确定 |
| 是 | 否 | RD-only 五分类 | 严格透传 RD 输出，不填零伪造 TR |
| 否 | 是 | TR-only 五分类 | 严格透传 TR 输出，不把结果称为 RD 能力 |
| 否 | 否 | `abstain/unavailable` | 不计入 accepted 分类指标；记录原因和分母 |

### 9.4 实施前威胁模型和阻断条件

- 标签泄露：五类 label、TR label、后验人工标注、Unknown/reject 结果不得成为 RD 输入或采样/阈值字段。
- 关联泄露：parent/content/session/path/file fingerprint 可用于 split 和审计，但原始字符串、TR/RD pairing、匹配质量不得编码进模型或后处理。
- 时间泄露：causal window 末端不得读取未来 RD 帧；padding、重复时间和缺失帧必须显式 mask/拒绝。
- 统计泄露：normalization、feature stats、class/sampler weight、校准、融合权重、checkpoint 选择不得读取 test。
- 语义泄露：`UnknownTarget` 分类不得被当成 reject 指标；低置信度预测不得被当成 Unknown 真值。
- 可用性泄露：缺失模态不以全零张量伪造有效模态；双缺失不得强行输出五类预测。
- 任何字段 provenance 未闭环、split 交叉非零、五类顺序漂移、checkpoint schema 不匹配、smoke 未通过，均阻断 Stage 022。

## 10. 待实施阶段发现登记

以下是设计阶段的待核实事项，不是已确认代码缺陷：

1. 五类标签 provenance、Balloon/Clutter/Unknown 的有效帧支持和 split 隔离已由 Stage 013 核实；Stage 023 完整 validation 已显示长尾类指标不足。
2. 五类 stats 与现有 physical canonical store 的兼容性已由 Stage 014 strict artifact/hash 核验；不复用旧二分类 stats。
3. `rd_time_sync_status` 在 physical index 中为 `canonical_parent_association`，被视为 parent 关联语义；五分类 quality 明确排除 `canonical_time`。
4. TR-only 五分类 checkpoint 已发现并读取 pure-TR `best_model.pt` 的 labels/taxonomy/checkpoint metadata；其完整输出契约和融合路由仍待 Stage 018–020 专项验证。
5. 软级联 partition 已在同一 full validation 实测（cascade window Macro-F1=83.71%，低于 flat=84.15%）；late-fusion 权重仍无实测，不能承诺融合提升。

## 11. Stage 013–014 实施进展（已核实）

### 11.1 五类 physical RD 数据链已存在，无需 canonical 重建

- `data/processed/cq08_rd_physical_canonical_frame_index_v3_1.csv`：106026 frames、1549 parents、21×384 canonical shape；五类标签集合与 `LOW_ALTITUDE_CLASSES` 完全一致，source 仅 `cq08_track`。
- `data/processed/expert1_rd_only_causal_window_index_v1_physical_v3_1.csv`：22085 windows；train/val/test trainable support 分别为 12396/2710/2691，诊断窗口 4288（全部 `too_few_rd_frames`）。
- trainable 五类 support：

| split | Drone | Bird | Balloon | Clutter | Unknown | total |
|---|---:|---:|---:|---:|---:|---:|
| train | 5542 | 2665 | 3165 | 370 | 654 | 12396 |
| val | 1269 | 523 | 645 | 114 | 159 | 2710 |
| test | 1188 | 632 | 684 | 54 | 133 | 2691 |

- parent cross-split=0、content-group cross-split=0、future window=0、`rd_fusion_eligible=False` 22085/22085；因此没有发现必须重建 canonical tensors 的数据契约阻塞。

### 11.2 pairing 和字段 provenance

- `src/process/audit_expert1_rd_pairing.py` 对 `expert1_rd_segment_manifest.csv` 的 25021 segments 审计结果：fusion_eligible=0；24986 rejected（`proportional_fallback`/coverage 原因），35 diagnostic_only。
- physical window index 的 `rd_time_sync_status` 全为 `canonical_parent_association`，但它只是 RD parent 关联语义；不能把该字段或其派生值作为五分类 RD 特征。旧二分类 dataset 的 `canonical_time` quality 已确认属于该风险边界。
- 五分类设计排除 `canonical_time`，保留 RD present/phase/coverage/frame-count/duration/dynamic-range 等合法 RD 派生量；旧二分类 v9 代码、stats 和 checkpoint 不修改。

### 11.3 已实施的独立五类链路

- [x] 新增独立 `src/core/data/rd_physical_five_dataset.py`、`src/core/rd_physical_five_model.py`、`src/core/rd_physical_five_artifacts.py` 和五类配置/训练/评估入口；旧 `rd_physical_window_dataset.py`、旧 schema 和旧 checkpoint 未修改。
- [x] 五类使用独立 `rd_physical_train_normalization_v4_five` schema；旧 `v4` 不变。
- [x] 新增 `src/process/build_rd_physical_five_normalization_stats.py` 和 `src/process/build_rd_physical_five_feature_stats.py`；输出文件不覆盖旧二分类 artifact，五类 feature stats 不含 `canonical_time`。
- [x] 旧 RD physical dataset/artifact 回归：15 passed；py_compile 通过。此前发现的旧 checkpoint provenance 失败已通过“独立 builder/独立 schema”方式消除，未修改旧 generator。

### 11.4 已完成 stats 核验与当前阻塞

- 用户已完成 stats 生成和历史清理；清理报告为 `tmp/20260728-rd-only-cleanup-execution-report.json`，DeletedCount=30、FailedCount=0。五类 stats 已核验 train-only、finite、五类顺序和 provenance hash；不再重复执行生成或删除。
- 五类 stats 已核验：normalization schema=`rd_physical_train_normalization_v4_five`、feature schema=`rd_physical_five_feature_stats_v1`、index SHA=`c9cbf3ff9a354766583df3a896594420b65f566206b65b80c2c7522e572d19c1`、quality names 为7维且不含 `canonical_time`，test_evaluation_executed=false。
- 当前不再阻塞 RD-only 五分类全量训练前置条件；剩余后置工作是完整五类 validation、TR-only 真实模型完成后的四态融合接入/评估、线上 stable 和拒识验证。最终五类 smoke 仅4/2710 val windows，且类别分布偏斜，不作性能结论。

本段原先的人工生成命令已完成，仅作为历史过程保留，不再重复执行。

## 12. Stage 024–031 独立 ParentGate、DBB/CU 专科实施发现（已达全量训练前 readiness）

### 12.1 已确认的代码事实

- 当前 `PhysicalRDFiveClassifier` 在同一个 RD embedding 上同时创建 `flat_head`、`parent_head` 和两个 `child_heads`；`parent_head` 不是独立模型，当前配置 `deployment_head="flat"`，因此现有第一层只参与辅助训练和 cascade 对照，不承担独立线上路由。
- 当前 trainer 在同一 batch 对 flat/cascade/parent/prefix/micro 输出计算联合 loss；没有 ParentGate 专属 optimizer、checkpoint、class balance 或独立校准。
- 当前 `cascade_logits` 使用 `log_softmax(parent_logits)+log_softmax(child_logits)` 形成软概率乘积；它不使用 parent argmax，但其 parent 和 child 仍共享 encoder，且 Stage 023 flat 优于 cascade（Macro-F1 84.15% vs 83.71%）。
- Stage 023 flat validation 混淆折叠成 DBB/CU 为 `[[2402,35],[54,219]]`，折叠准确率约96.72%；这是父组边界的代理诊断，不是独立 ParentGate 能力证据。

### 12.2 设计判断

- 独立 ParentGate 可行且有明确实验价值：它能单独优化 DBB/CU 先验、类别权重、校准和 checkpoint；但它不能解决 Bird↔Drone/Balloon 或 Clutter↔Unknown 的组内混淆。
- DBB 专科应为严格三分类，CU 专科应为严格二分类；各自的局部指标必须与完整五分类组合指标分开，不能用 oracle 父组结果冒充部署结果。
- 默认采用 soft parent probability，而非 hard routing。硬路由的上游父组错误会不可逆地屏蔽正确专科类别；soft product 允许父组不确定性传播。
- 为保持当前 Drone/Bird 能力，暂不删除 Stage 023 flat 路径。候选组合须同时满足完整五类指标非劣化和 Drone/Bird Precision/Recall/F1 不低于 Stage 011 参考；否则保留 flat。
- CU 组 validation support 只有 Clutter=114、Unknown=159 windows，parent support 分别为 8、15，故 CU 改善结论必须标记统计稳定性，不能用少量 parent 的偶然提升宣称泛化。

### 12.3 统一概率契约

```text
q = softmax(ParentGate(x))
r = softmax(DBBSpecialist(x))
s = softmax(CUSpecialist(x))

P(Drone)   = q(DBB) * r(Drone)
P(Bird)    = q(DBB) * r(Bird)
P(Balloon) = q(DBB) * r(Balloon)
P(Clutter) = q(CU)  * s(Clutter)
P(Unknown) = q(CU)  * s(Unknown)
```

每行五类概率应和为 1；父组、专科或 RD 输入不可用时不能以全零输入伪造预测，必须显式 `unavailable/abstain` 或回退到已验证 flat RD-only 输出。UnknownTarget 仍是普通分类标签，不能转译成拒识能力。

### 12.4 已实施证据

- 已新增独立 RD-only specialist 链路：`src/core/data/rd_physical_specialist_dataset.py`、`src/core/rd_physical_specialist_config.py`、`src/core/rd_physical_specialist_model.py`、`src/core/rd_physical_specialist_artifacts.py`、`src/core/rd_physical_specialist_trainer.py`、`src/core/train_rd_physical_specialist.py`、`src/core/evaluate_rd_physical_specialist.py`、`src/core/rd_physical_hierarchical_composer.py`。
- 三视图实际支持（仅 train/val，来自现有五类 index）：ParentGate train/val=12396/2710 windows、DBB=11372/2437、CU=1024/273；parent 数分别为 1058/221、953/198、105/23。CU 的 Clutter/Unknown parent 支持低，后续完整指标需标记统计不稳定。
- 三份 smoke config 已固定：`config/rd_only/rd_specialist_parent_gate_smoke_seed42.json`、`rd_specialist_dbb_smoke_seed42.json`、`rd_specialist_cu_smoke_seed42.json`；三份 full config 已固定：对应 `*_full_seed42.json`。入口实际 `python -m src.core.train_rd_physical_specialist --config ...`，evaluator 实际 `python -m src.core.evaluate_rd_physical_specialist --checkpoint ... --output ... --split val --device cuda --max-batches ...`。四个 full checkpoint 已生成并核验：flat epoch17 SHA=`4CD37A4D30167E1D6D96A5E13A31CAD852A772A31C4AC4BB31B3FA9B0DDE899E`，ParentGate epoch2 SHA=`C0CA0D9C2D9C90AA88644A5B1FCD7C9C90E8F7FC4D78D26EFFE68652AE293932`，DBB epoch4 SHA=`C5CF1B72C2E1C44F4C085B88609F93CEF39ACA7C13C4D59FCA653EEC70C6AB04`，CU epoch9 SHA=`7D4CDE566367944A3D07F9450FF6AAF45B756118A741DFB48C7CFFE98D855A73`。
- 最新 smoke 结果（最终 schema）：ParentGate checkpoint=`artifacts/20260803-232631/rd-specialist-parent-gate-smoke-seed42/best_rd_physical_parent_gate_specialist.pt`；DBB=`artifacts/20260803-232700/rd-specialist-dbb-smoke-seed42/best_rd_physical_dbb_specialist.pt`；CU=`artifacts/20260803-232709/rd-specialist-cu-smoke-seed42/best_rd_physical_cu_specialist.pt`。每个均 1 epoch、2 train batches、2 val batches、loss/gradient finite、test=false、TR 输入未加载、strict reload 通过。受限 evaluator 每个仅评估 4 个 validation windows，不能作为性能指标。
- smoke evaluator 已明确输出 source=`cq08_track`、session unavailable（index 无 session 字段）、online stable unavailable、supports_rejection=false；UnknownTarget 仍是普通分类类，不是拒识。
- 组合器已通过 soft/oracle/hard/fallback 契约测试：`P5=P_parent×P_specialist`，每行和为 1±1e-6；缺 specialist 不用零 logits，flat fallback 仅在显式提供已验证 flat logits 时启用；hard routing 仅诊断。
- 回归依据：本轮 `pytest` 逐文件收集 `tests/test_rd*.py` 为94 passed、1 warning；`compileall -q src/core src/process config` 通过；新增同窗 evaluator py_compile 通过。未执行 test、OOF 校准、融合权重选择或线上接入。

### 12.5 当前未验证事项和后续阻塞

- 独立 ParentGate、DBB 专科、CU 专科已具备 full checkpoint 和完整局部 validation；局部指标不能替代完整五类组合能力。
- 实现复用了 audited RD encoder 的代码结构，但每个 specialist 都新建独立参数、optimizer、config、artifact 和 checkpoint；没有加载 Stage 023 权重，且已通过无 TR 输入测试。
- 三种 train/val 专科数据视图 provenance 已完成；完整 validation 已覆盖 val，但 CU 仅273 windows/23 parents，parent 层两类 support=8/15，统计稳定性有限。
- 尚未执行 OOF calibration、blend 权重选择、TR late-fusion 接入或线上 stable 评估；同窗完整组合验证已完成；test 继续禁止读取。
- 已完成 flat/oracle/predicted-soft/hard/fallback 同窗比较；predicted-soft window Macro-F1=83.80%，低于90%；oracle-parent window Macro-F1=89.79% 仅为真值父组诊断，不能部署；flat fallback 仍是当前安全基线。

## 13. Stage 035–036 full 训练后验证结论（2026-08-05）

### 13.1 Full artifact 与边界

用户完成的四个 full checkpoint 均通过 strict loader，且各自 `training_summary.json` 明确 `test_evaluation_executed=false`、`tr_inputs_loaded=false`、`supports_rejection=false`：

| 模型 | checkpoint | best epoch | SHA-256 | validation support |
|---|---|---:|---|---:|
| flat five | `artifacts/20260804-114842/rd-only-five-stage034-full-seed42/best_rd_physical_five_model.pt` | 17 | `4CD37A4D30167E1D6D96A5E13A31CAD852A772A31C4AC4BB31B3FA9B0DDE899E` | 2710 |
| ParentGate | `artifacts/20260804-184925/rd-specialist-parent-gate-stage034-full-seed42/best_rd_physical_parent_gate_specialist.pt` | 2 | `C0CA0D9C2D9C90AA88644A5B1FCD7C9C90E8F7FC4D78D26EFFE68652AE293932` | 2710 |
| DBB | `artifacts/20260804-231455/rd-specialist-dbb-stage034-full-seed42/best_rd_physical_dbb_specialist.pt` | 4 | `C5CF1B72C2E1C44F4C085B88609F93CEF39ACA7C13C4D59FCA653EEC70C6AB04` | 2437（仅三类视图） |
| CU | `artifacts/20260805-170322/rd-specialist-cu-stage034-full-seed42/best_rd_physical_cu_specialist.pt` | 9 | `7D4CDE566367944A3D07F9450FF6AAF45B756118A741DFB48C7CFFE98D855A73` | 273（仅两类视图） |

五分类同窗组合报告为 `tmp/20260805-rd-hierarchy-stage034-full-val.json`，覆盖同一 2710 个 validation windows；没有读取 test、没有用 validation 选择新 checkpoint/阈值/融合权重，oracle-parent 只用于误差上限诊断。

### 13.2 五分类 flat 与层级组合指标

以下均为 window 层 validation；百分比是实际计算值，不代表 test 或线上能力。

| 模式 | N | Accuracy | Macro-P | Macro-R | Macro-F1 | Balanced Accuracy | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| flat | 2710 | 93.62% | 87.07% | 81.86% | 83.55% | 81.86% | 当前 RD 安全基线 |
| cascade（flat 模型内部辅助头） | 2710 | 93.28% | 86.05% | 81.38% | 82.90% | 81.38% | 低于 flat |
| predicted-soft（ParentGate×DBB/CU） | 2710 | 92.88% | 86.08% | 82.05% | 83.80% | 82.05% | 不达标，不接线上 |
| hard-routing（仅诊断） | 2710 | 92.84% | 85.93% | 82.01% | 83.72% | 82.01% | 仅诊断 |
| flat fallback | 2710 | 93.62% | 87.07% | 81.86% | 83.55% | 81.86% | 与 flat 相同 |
| oracle-parent（真值父组，仅诊断） | 2710 | 95.68% | 90.72% | 89.26% | 89.79% | 89.26% | 不可部署 |

flat window 分类别：Drone `P/R/F1=98.52%/100.00%/99.26%`，Bird `90.38%/86.23%/88.26%`，Balloon `94.29%/99.84%/96.99%`，Clutter `85.29%/50.88%/63.74%`，Unknown `66.86%/72.33%/69.49%`。flat parent=221，Accuracy=94.57%、Macro-F1=85.31%；Clutter parent support=8、Unknown=15，均低 support。

predicted-soft 分类别：Drone `97.75%/99.21%/98.47%`，Bird `89.63%/84.32%/86.90%`，Balloon `91.58%/99.53%/95.39%`，Clutter `71.13%/60.53%/65.40%`，Unknown `80.30%/66.67%/72.85%`。其 parent=221，Accuracy=95.48%、Macro-F1=88.76%，但 window 主要指标和 Drone/Bird 均未满足 Stage 011 参考非劣化。

### 13.3 混淆和根因证据

- flat confusion（行真值、列预测）为 `[[1269,0,0,0,0],[18,451,29,0,25],[0,1,644,0,0],[0,21,3,58,32],[1,26,7,10,115]]`。主要错误为 Bird→Drone/Balloon/Unknown=`18/29/25`，Clutter→Bird/Unknown=`21/32`，Unknown→Bird/Clutter=`26/10`。
- predicted-soft confusion 为 `[[1259,3,1,1,5],[27,441,30,14,11],[0,3,642,0,0],[2,18,15,69,10],[0,27,13,13,106]]`。组合减少部分 Clutter→Unknown（32→10），但增加跨父组和 Bird→Drone/Balloon，整体 Macro-F1 只由83.55%升至83.80%，不足以部署。
- oracle-parent 将父组错误去除后，Bird F1=93.67%、Clutter F1=74.88%、Unknown F1=84.18%，但 Clutter→Unknown=35、Unknown→Clutter=18 仍存在；这证明 CU 组内判别而非单纯 ParentGate 是主要瓶颈，同时 oracle 不可作为真实系统结果。
- DBB 专科局部 window Macro-F1=96.63%、Bird Recall=89.10%；CU 专科局部 window Macro-F1=79.53%、Clutter Recall=69.30%、Unknown Recall=88.68%。局部视图 support 与完整五类分母不同，不能拼成五类能力。
- flat ECE=6.04%，predicted-soft ECE=4.61%，oracle=3.27%；这些是未校准 validation 诊断，不能转译为拒识或阈值效果。
- 训练/验证泛化差距是可复现的风险证据：flat best epoch17 train Macro-F1=99.31% 而 val Macro-F1=83.55%；CU best epoch9 train Macro-F1=90.97% 而 val Macro-F1=79.53%（train/val 支持分别1024/273）。这证明存在明显泛化差距，但不能仅凭该现象断言唯一根因；后续需单因素正则化、采样或标签审计实验。
- 报告口径已统一：`src/core/rd_physical_parent_aggregation.py` 将 flat/five、specialist trainer/evaluator 的 parent 聚合固定为“每窗口 softmax 概率均值后 argmax”。Stage 037 full val 回归显示 flat window Accuracy/Macro-F1 仍为93.62%/83.55%，parent（221）Accuracy/Macro-F1=94.57%/85.31%；window 指标不变。旧 Stage 022/Stage 034 mean-logit parent 数字不与新口径横向比较，也不用于新 checkpoint/融合选择。

### 13.4 能力边界与后续阻塞

四个模型和组合器的 `supports_rejection=false`、`abstain_output=false`、`threshold=null`；`UnknownTarget` 是普通五分类标签，不是 OOD/reject。由于没有经过验证的 reject 标签、动作和阈值来源，Coverage、Accepted Accuracy、FAR、FRR、Selective Risk 均为未评估，不能伪造。

当前只能保留 flat 作为 RD-only 五分类安全基线，不能标记“90%达标”或接入线上。后续应先做 Clutter/Unknown label semantics 与 parent 内部一致性审计，再分别开展 CU 单因素目标/采样实验、Bird 单因素权重实验和 parent-grouped OOF 校准；任何实验必须新 artifact、只用 train/val、保持完整分母并同时检查 Drone/Bird 不退化。
- 尚未执行 calibration、blend 权重选择、TR late-fusion 接入或线上 stable 评估；test 继续禁止读取。

## 15. Stage 037 聚合修复后的追加证据（2026-08-05）

- `tmp/20260805-rd-five-stage037-parent-prob-val.json`、`tmp/20260805-rd-parent-stage037-parent-prob-val.json`、`tmp/20260805-rd-dbb-stage037-parent-prob-val.json`、`tmp/20260805-rd-cu-stage037-parent-prob-val.json` 均为 validation-only full split，分别覆盖 2710、2710、2437、273 windows，`test_evaluation_executed=false`、`tr_inputs_loaded=false`。
- 统一 parent 口径后 flat window 仍为 Accuracy=93.62%、Macro-F1=83.55%；parent 221 Accuracy=94.57%、Macro-F1=85.31%。ParentGate window/parent Macro-F1=88.28%/91.34%，DBB=96.63%/98.72%，CU=79.53%/80.83%。CU parent support=8/15，仍标记 low stability。
- 对 flat 预测按窗口长度做只读诊断：`is_prefix_window=True` 的 485 windows Macro-F1=79.12%，非 prefix 2225 windows Macro-F1=84.21%；长度 `>=24` 的 702 windows Macro-F1=68.21%，`<24` 的 2008 windows Macro-F1=84.85%。长窗口子集 Bird Recall=64.47%、Clutter Recall=0.00%，但该切片没有改变总体分母，不能作为调参依据；它只说明时间长度/parent 场景是后续单因素分析候选。
- CU 错误集中于少数 parent：flat Clutter 错误主要来自 6 个 val parent（其中 `cq08|track|4198` 为 19/19、`3183` 为 11/14、`2251` 为 9/9）；Unknown 错误主要来自 8 个 parent（`2709` 10/14、`3222` 9/14、`1548` 7/7）。这些 parent 的 manifest label 和 MAT 文本与真值一致，故是异质性/泛化线索，不是标签泄露证据。
- 当前 full train/val 泛化差距：flat train/val Macro-F1=99.31%/83.55%；DBB=97.85%/96.63%；CU=90.97%/79.53%。这是可复现风险证据，不能单独断言唯一根因。

- Stage 037 smoke A（Bird class weight=1.35）和 smoke B（parent_loss_weight=0）均只评估4个 val windows，`full_split_evaluated=false`，因此其 0.00% 等受限数字全部不作性能结论；它们仅证明配置、RD-only forward、loss/gradient、checkpoint 保存/加载和 evaluator 可执行。两项 full experiment 尚未启动。
- 两份 full config 已创建但未执行：A `config/rd_only/rd_physical_five_bird_weight_stage037_full_seed42.json`，B `config/rd_only/rd_physical_five_parent_loss0_stage037_full_seed42.json`。实载检查确认 18 epochs、batch=4、train/val full、固定 seed=42/CUDA、test-off、TR input-off，且输出目录当前不存在；是否启动由用户决定。
- 四 full checkpoint 的 hierarchy smoke (`tmp/20260805-rd-hierarchy-stage037-smoke.json`) 已通过 strict reload、概率归一化和缺失/回退契约；只评估4个 val windows，`full_split_evaluated=false`，不作性能结论。

## 16. 当前问题清单（按严重性）

1. [文件] `artifacts/20260804-114842/rd-only-five-stage034-full-seed42/best_rd_physical_five_model.pt` 及对应 validation；[位置] 五分类 flat full validation；[问题] 五分类总体及多个类别未达到 90.00% 目标；[证据] val 2710 windows：Accuracy=93.62%、Macro-F1=83.55%；Bird P/R/F1=90.38%/86.23%/88.26%，Clutter=85.29%/50.88%/63.74%，Unknown=66.86%/72.33%/69.49%；[影响] Bird、Clutter、Unknown 召回/宏平均不足，不能标记五分类达标或上线。
2. [文件] `tmp/20260805-rd-hierarchy-stage034-full-val.json`；[位置] predicted-soft ParentGate×DBB/CU；[问题] 独立级联组合未改善到部署门槛且牺牲 Drone/Bird 参考；[证据] val Macro-F1=83.80%，Bird R/F1=84.32%/86.90%，Drone P/R/F1=97.75%/99.21%/98.47%，低于 Stage 011 二分类参考；[影响] 级联误差传播，当前只能保留 flat RD-only 基线。
3. [文件] `src/core/evaluate_rd_physical_five.py`、`src/core/evaluate_rd_physical_specialist.py`；[位置] parent 聚合及已生成旧 Stage 034 报告；[问题] 历史报告使用 mean-logit、hierarchy 使用 mean-probability，曾造成 parent 指标不可直接比较；[证据] Stage 037 新增 `rd_physical_parent_aggregation.py`、统一代码路径、96 项 RD 测试通过；旧报告仍是历史口径；[影响] 若混用会误选 checkpoint/融合策略。已修复新路径，旧 artifact 不追溯改写。
4. [文件] `src/process/cq08_manifest.py`、`tmp/20260805-rd-five-label-semantics-audit.json`；[位置] CQ-08 label_id=5/6；[问题] Unknown 原始 MAT 文本为 `Other`，语义不是独立 OOD/reject 标注；[证据] Unknown 100 parents/6170 frame texts 全为 `Other`，Clutter 全为 `Clutter`；train/val frame-manifest mismatch=0、parent 混标=0；[影响] 不能将 UnknownTarget 当拒识或据此计算 FAR/FRR；当前无已验证拒识能力。
5. [文件] `src/core/experts.py`、`src/core/rd_tr_fusion.py`、`src/core/rd_physical_hierarchical_composer.py`；[位置] physical RD 五分类生产调用链；[问题] physical RD-only flat/specialist/four-state fusion 尚未接入实际 System/online stable caller；[证据] `route_rd_tr_five`/`compose_rd_specialists` 仅定义/测试引用，evaluator `online_stable_decision.available=false`；[影响] 不能宣称线上 stable、TR late-fusion 或缺失模态生产行为已验证。
6. [文件] `artifacts/20260804-231455/...`、`artifacts/20260805-170322/...`；[位置] DBB/CU full specialist 泛化；[问题] 专科 validation 与 train 存在差距，CU parent support 极低；[证据] DBB train/val Macro-F1=97.85%/96.63%，CU=90.97%/79.53%；CU val parent Clutter/Unknown support=8/15；[影响] CU/Unknown 泛化与 parent 指标统计不稳定，不能把局部专科分数拼成五类能力。
7. [文件] `src/core/rd_physical_five_trainer.py`、`src/core/rd_physical_specialist_trainer.py`；[位置] 训练优化/辅助目标；[问题] flat、cascade、parent、prefix、micro 多目标和 class/evidence 组合的单因素收益尚未验证；[证据] full config 权重为 1/0.25/0.15/0.10/0.10，Stage 037 A/B 仅 smoke 4 val windows；[影响] Bird 与 CU 瓶颈根因尚未可归因，不能承诺某个权重修改提升。

## 17. Stage 037 Clutter/Unknown 标签语义与元数据审计（2026-08-05）

本阶段只读取 CQ-08 manifest、grouped split，以及 physical canonical frame/window index 的 train/val 元数据；没有读取 test tensor、test prediction、test metric，也没有用审计结果选择 checkpoint 或调参。可复现实行入口为 `src/process/audit_rd_five_label_semantics.py`，报告为 `tmp/20260805-rd-five-label-semantics-audit.json`。

### 14.1 原始标签来源

- manifest 共 1550 条：DroneTarget 700（label_id=1/2，其中 699 usable）、BirdTarget 350（3）、BalloonTarget 350（4）、ClutterTarget 50（5）、UnknownTarget 100（6）。
- Clutter 的所有 RD MAT 文件文本均为 `Clutter`（2852 frame records）；Unknown 的所有 RD MAT 文件文本均为 `Other`（6170 frame records）。manifest label、label_id、track/point/raw 文件名 ID 校验无不一致（0）；train/val frame label 与 manifest label 不一致为 0。
- `Other` 是 CQ-08 原始文件命名文本，不能单独证明“开放集”“未知真值”或拒识标签；UnknownTarget 仍是普通五分类标签，不能据此宣称 reject/OOD 能力。

### 14.2 parent、split 和窗口一致性

- train/val physical frame metadata 89997 行、1316 个 parent；parent 内多 label、跨 track、跨 source 冲突均为 0。
- train/val window metadata 的 parent label/split/content-group 冲突为 0；train/val parent group intersection=0，content-group intersection=0。
- trainable support：train 12396 windows（Clutter 370、Unknown 654），val 2710 windows（Clutter 114、Unknown 159）；排除原因均为 `too_few_rd_frames`，未删除困难样本或改变评估分母。
- train/val 的 Clutter parent=43、Unknown parent=85；val trainable 只有 Clutter 8、Unknown 15 个 parent，parent 指标统计稳定性有限。

### 14.3 RD 物理元数据分布（仅 train/val）

- 所有五类 `source_range_coverage_ratio` 与 `source_velocity_coverage_ratio` 的 train/val frame 值均为 1.0，未发现 coverage 缺失造成的 Clutter/Unknown 标签差异证据。
- Clutter `target_range_reference_m` 中位数 1717.5 m、q01/q99=592.5/4125.0 m；Unknown 中位数 1132.5 m、q01/q99=277.5/3937.5 m。两类范围重叠，不能用 range reference 直接解释为可分离标签或把该字段当作标签。
- Clutter observation duration 中位数 0.02736 s，Unknown 0.01728 s；frame time 中位数分别 0.58752/0.496056 s。这些是 RD canonical metadata 的审计统计，不进入模型能力结论；仍需单因素模型实验才能判断是否有泛化收益。

### 14.4 结论与阻塞

当前没有证据支持“Clutter/Unknown 标签映射错误、parent 混标或 train/val 泄露”是已确认根因；因此不重标、不重建 canonical/index/stats。已确认的性能瓶颈仍是 validation 中 CU 组内混淆和低 parent support：flat Clutter/Unknown F1=63.74%/69.49%，oracle-parent 仍有 Clutter→Unknown=35、Unknown→Clutter=18。CU 单因素训练、Bird 单因素训练、OOF calibration 与线上 four-state fusion 仍未执行，继续禁止读取 test。

## 18. Stage 038 V3/V4 full validation 与 Bird/Clutter 低召回归因（2026-08-06）

### 18.1 严格评估边界

- V3 checkpoint：`artifacts/20260805-193521/rd-only-five-stage037-bird-weight-full-seed42/best_rd_physical_five_model.pt`；报告：`tmp/20260806-rd-five-stage037-v3-full-val.json`及 predictions CSV；best epoch=9。
- V4 checkpoint：`artifacts/20260806-142225/rd-only-five-stage037-parent-loss0-full-seed42/best_rd_physical_five_model.pt`；报告：`tmp/20260806-rd-five-stage037-v4-full-val.json`及 predictions CSV；best epoch=12。
- 两者均由 `evaluate_rd_physical_five --split val --device cuda --max-batches 0` 严格 reload，2710/2710 windows、excluded=0、`test_evaluation_executed=false`、`tr_inputs_loaded=false`、`supports_rejection=false`、source唯一为`cq08_track`；未读取 test、未调阈值、未启动新训练。

### 18.2 实际指标

| 版本 | Accuracy | Macro-P | Macro-R | Macro-F1 | Bird R | Clutter R | Unknown R |
|---|---:|---:|---:|---:|---:|---:|---:|
| V3 Bird weight=1.35 | 93.28% | 86.12% | 82.85% | 84.01% | 83.94% | 57.02% | 74.21% |
| V4 parent loss=0 | 92.47% | 83.94% | 80.39% | 81.83% | 82.60% | 57.02% | 62.89% |
| Stage 034 flat参照 | 93.62% | 87.07% | 81.86% | 83.55% | 86.23% | 50.88% | 72.33% |

分类别完整 P/R/F1：V3 Drone=98.60/99.68/99.14%、Bird=90.89/83.94/87.28%、Balloon=93.03/99.38/96.10%、Clutter=80.25/57.02/66.67%、Unknown=67.82/74.21/70.87%；V4 Drone=98.37/99.76/99.06%、Bird=89.07/82.60/85.71%、Balloon=92.52/99.69/95.97%、Clutter=76.47/57.02/65.33%、Unknown=63.29/62.89/63.09%。

### 18.3 因果可归因性

- V3相对 Stage 034 只把 Bird class weight 从1.0改为1.35；Bird Recall反而下降2.29个百分点，Clutter Recall上升6.14个百分点。该实验不支持“仅提高 Bird weight 就恢复 Bird recall”。
- V4相对 Stage 034 只把 parent loss 从0.15改为0；Bird/Unknown下降，Clutter Recall升至57.02%。V3与V4同时改变了 class weight 和 parent loss，不能用V3↔V4差异作单因素结论。
- V4并非完全无父组约束：trainer仍保留0.25 cascade CE，而 model 的 cascade logits由 parent log-prob+child log-prob构造；因此 parent loss=0不能解释为级联完全移除。

### 18.4 低召回的直接证据

- 混淆矩阵：V3 Bird→Drone/Balloon/Clutter/Unknown=15/29/9/31，Clutter→Bird/Unknown=18/25，Unknown→Bird/Clutter=21/6；V4分别为19/36/6/30、16/27、34/14。Bird并非只被Drone吞掉，Clutter也同时流向Bird和Unknown。
- val support仅 Bird=523/43 parents、Clutter=114/8、Unknown=159/15；V3 Clutter 49个错误集中在5个parent，前5个覆盖100%，`cq08|track|4198`为19/19；V4 Clutter 49个仍集中，4198/2251/2901全错。Bird V3/V4错误集中在13/15个parent，Unknown集中在7/9个parent。
- 长窗口证据：V3 Bird Recall `<24`/`>=24`=86.58%/68.42%，Clutter=64.36%/0.00%；V4 Bird=85.46%/65.79%，Clutter=64.36%/0.00%。prefix窗口也低于non-prefix。该切片未改评估分母，仅说明时间长度/场景异质性候选。
- 训练/验证差距：V3/V4 train Macro-F1约98–99%，val Macro-F1=84.01%/81.83%；这是过拟合/场景异质性风险证据，不是唯一根因证明。
- train evidence有效权重总量：Drone=4867.40、Balloon=2844.35、Bird=2397.46、Unknown=564.44、Clutter=336.50；份额44.21%/25.83%/21.78%/5.13%/3.06%。CU在共享模型中的监督量极低，是机制证据；不能据此承诺提高CU权重必然提升。
- 高置信度误判：V3错误最大概率均值 Bird/Clutter/Unknown≈95.84%/94.35%/94.74%；V4≈96.06%/89.20%/95.43%。因此单一低置信度阈值不能直接修复召回。

### 18.5 结论、问题和后续

1. 已确认：Bird低召回是五分类竞争（Balloon/Unknown/Drone等方向均有）与高置信度泛化错误共同表现，不能归结为Bird weight单一缺失。
2. 已确认：Clutter低召回的主要可观测瓶颈是CU有效样本/parent极少、Clutter↔Unknown/Bird边界和少数parent集中；V3/V4共同57.02%说明两个开关都不是充分解释。
3. 未确认：具体RD物理表征是否缺少可泛化的Clutter/Unknown特征；需要仅改变CU采样/目标或输入特征的独立 train/val 实验，不能读取test或删除困难样本。
4. 当前不能把 UnknownTarget 当拒识，不能输出FAR/FRR/Selective Risk；不能接线上stable/fusion。后续实验需新配置、新artifact、一次一个主要因素，同时检查五类总体及Drone/Bird非劣化。

## 19. Stage 039 ParentGate 第一层路由审查结论（2026-08-06）

### 19.1 先纠正路径认知

- V3/V4 的实际 config `deployment_head="flat"`，模型 forward 虽然同时计算 `parent_logits`、`cascade_logits`、`flat_logits`，但 evaluator 的 selected head 是 flat；因此 V3/V4 的 Bird→Clutter/Unknown 是 flat 五分类 head 直接产生，不能作为 ParentGate 第一层性能证据。
- 真正级联只在独立层级 evaluator 中使用 ParentGate→DBB/CU。其输出顺序固定为 `DBBGroup, CUGroup`，再映射到 `Drone/Bird/Balloon` 或 `Clutter/Unknown`。

### 19.2 ParentGate 实际 validation

| 项目 | 数值 |
|---|---:|
| validation windows | 2710 |
| DBBGroup support | 2437 |
| CUGroup support | 273 |
| Accuracy | 96.05% |
| Macro-F1 | 88.28% |
| DBBGroup Recall | 98.69% |
| CUGroup Recall | 72.53% |

混淆矩阵（行真值、列预测）为：`[[2405,32],[75,198]]`。即 DBB→CU=32，CU→DBB=75。

类别路由错误：

- Bird→CU：26/523，ParentGate Bird 路由 Recall=95.03%。
- Clutter→DBB：35/114，ParentGate Clutter 路由 Recall=69.30%。
- Unknown→DBB：40/159，ParentGate Unknown 路由 Recall=74.84%。

### 19.3 级联错误拆分

在 ParentGate 路由正确的样本上：

- DBB specialist：Drone Recall=99.68%，Bird Recall=88.53%，Balloon Recall=99.53%。
- CU specialist：Clutter Recall=87.34%，Unknown Recall=89.08%。

因此级联存在两个独立瓶颈：

1. ParentGate 第一层 CU Recall 只有72.53%，造成 Bird 被错误送入 CU、Clutter/Unknown 被错误送入 DBB；
2. 即使 ParentGate 路由正确，DBB 内部 Bird Recall 仍只有88.53%，CU 内部仍存在 Clutter↔Unknown 混淆。

不能只修 ParentGate 就宣称 Bird/Clutter 已解决。

### 19.4 代码正确性审查

- `src/core/data/rd_physical_specialist_dataset.py` 的 remap 明确为：Drone/Bird/Balloon→DBBGroup，Clutter/Unknown→CUGroup；没有发现标签反转。
- `src/core/rd_physical_specialist_model.py` 的 ParentGate 使用原生2类 `parent_logits`；DBB/CU 使用原生3/2类 local heads；checkpoint metadata shape 分别为2、3、2，没有发现全局五类切片错位。
- `src/core/rd_physical_hierarchical_composer.py` 的 predicted-soft 概率为 `P(parent)×P(local|parent)`，hard-routing 只在 ParentGate argmax 后选择对应局部 head；概率和测试通过。
- `tests/test_rd_specialist_contract.py` 等专科/融合测试共22项通过，相关模块 compileall 通过。

### 19.5 已确认配置问题

- ParentGate full config 使用 `class_weights=[1,1]`，没有 sampler；trainer 使用普通 weighted CE 加 prefix/micro auxiliary loss。
- train effective evidence 权重 DBB=10109.21、CU=900.94，约11.22:1；validation support DBB/CU=2437/273。
- 当前 best epoch=2，`best_selection.eligible=false`。epoch 2 CU Recall=72.53%，epoch 18 也只有74.36%；现有配置参考 CU Recall=80.22%，18个 epoch 均未达到参考门槛。

### 19.6 修复边界

已确认需要调试的是 ParentGate 第一层的类别不平衡、checkpoint selection 和泛化，而不是 composer 标签顺序。下一步应只选一个主要因素进行实验：

- ParentGate CU class weight 单因素；或
- ParentGate balanced sampler 单因素；或
- ParentGate 选择规则加入最小 DBB/CU Recall 约束。

每个实验必须独立 config/checkpoint/validation 报告，只使用 train/val，不调测试集，不用 flat 混淆结果替代 ParentGate 指标。**本段是 Stage 039 历史快照**；Stage 040 full 与 Stage 041 修复的后续结论见下文。

## 20. Stage 040 ParentGate CU 权重单因素 full 验收结论（2026-08-07）

### 20.1 实际产物与边界

- 用户完成 checkpoint=`artifacts/20260806-235753/rd-specialist-parent-gate-stage040-cu-weight5-full-seed42/best_rd_physical_parent_gate_specialist.pt`，epoch=3，SHA-256=`0A8F89EED1EEF408B33BBC7B352C9D9C9E0E49349CA8C9963B60589C536F9806`。
- 实际 config 为 ParentGate、`class_weights=[1,5]`、18 epochs、batch=4、lr=1e-4、seed=42、CUDA、同 index/stats、`run_test_evaluation=false`、`allow_tr_inputs=false`；summary 明确 test split 未加载、TR inputs 未加载。
- 独立复算报告=`tmp/20260807-rd-parent-stage040-cu-weight5-full-val.json`，predictions CSV 同 basename；2710/2710 validation windows、excluded=0、test=false。没有读取 test、没有阈值/校准、没有拒识或线上 stable。

### 20.2 ParentGate 指标与旧基线差异

| 指标 | Stage 034旧 gate | Stage 040 weight=5 | 差值 |
|---|---:|---:|---:|
| Accuracy | 96.05% | 95.68% | -0.37个百分点 |
| Macro-F1 | 88.28% | 88.03% | -0.25个百分点 |
| DBB Recall | 98.69% | 97.66% | -1.03个百分点 |
| CU Precision | 86.09% | 78.89% | -7.20个百分点 |
| CU Recall | 72.53% | 78.02% | +5.49个百分点 |
| CU F1 | 78.73% | 78.45% | -0.28个百分点 |
| parent Macro-F1 | 91.34% | 89.90% | -1.44个百分点 |

- 旧/新 window confusion（行真值、列预测 DBB/CU）分别为 `[[2405,32],[75,198]]` 与 `[[2380,57],[60,213]]`。权重5减少15个 CU→DBB，却增加25个 DBB→CU。
- 五类路由变化：Bird→CU 26/523→40/523，Balloon→CU 0/645→12/645，Drone→CU 6/1269→5/1269；Clutter→DBB 35/114→29/114，Unknown→DBB 40/159→31/159。
- 因而 Bird 路由 Recall 95.03%→92.35%，Balloon 100.00%→98.14%；Clutter 69.30%→74.56%，Unknown 74.84%→80.50%。这是明确的决策边界整体向 CU 移动，不是“免费提升 CU”。
- Stage 040 的 `best_selection.eligible=false`；配置中的原 reference 数字只在配置及派生 summary 中找到，未找到对应完整 validation 报告，provenance 不可验证，不能把该 eligibility 当作可靠部署门槛。

### 20.3 同窗五分类影响与停止条件

- 使用 `tmp/20260806-rd-hierarchy-v4-full-val.predictions.csv` 恢复固定 DBB/CU 局部条件概率，只把 ParentGate 概率替换为 Stage 040 predictions，再按已审查公式 `P(parent)×P(local)` 重组；样本 ID 2710/2710 对齐、每行概率和断言通过。该离线同窗重组不读取 RD/test，不改变分母或局部 specialist。
- 旧→新 predicted-soft：Accuracy 92.88%→92.29%、Macro-F1 83.80%→82.89%、Bird Recall 84.32%→81.45%、Clutter Recall 60.53%→63.16%、Unknown Recall 66.67%→71.07%。parent Macro-F1 88.76%→86.76%。
- Stage 040 改善 Clutter/Unknown，但牺牲 Bird、Balloon、CU Precision、总体和 parent 指标，违反用户明确的 Bird 非劣化要求。结论：不替换 Stage 034 ParentGate，不接线上，不进入 test。
- `UnknownTarget` 仍是普通五分类标签，不是 reject；所有 specialist `supports_rejection=false`。

## 21. Stage 041 weighted-CE 与安全选择修复结论（2026-08-07）

### 21.1 已确认代码问题

1. `_masked_ce` 旧实现调用 `cross_entropy(..., weight=class_weights, reduction="none")` 后，分子包含真值类别权重，分母却只有 evidence weight。PyTorch weighted mean 的正确分母应包含真值类别权重；旧实现使 CU=5 同时改变相对类别代价和整个小 batch 的梯度尺度。
2. Stage 040 pre-clip gradient norm 的 epoch mean=2.73–16.56、max最高2303.91；Stage 034 equal-weight mean=0.89–3.67、max最高581.33。该证据支持“梯度尺度受权重放大且波动”的机制风险，但不宣称它是所有泛化问题的唯一根因。
3. 旧 `eligible_weak_class_guard` 在所有 epoch 均不合格时优先最小 Recall，可能选择“CU Recall较高但 CU Precision/DBB/Bird明显退化”的候选；同时旧 summary 仍把不合格回退候选命名为 `best_*`，容易被误用。
4. ParentGate 只看 DBB/CU 聚合 P/R/F1不足以保护 Bird；Stage 040 DBB Recall仅下降1.03个百分点，而 Bird路由 Recall下降2.68个百分点、Balloon下降1.86个百分点。

### 21.2 已实施修复

- `_masked_ce` 分母改为 `sum(valid × evidence × target_class_weight)`；当 class weights 全为1时与旧数值相同，既有 checkpoint 推理不受影响。测试覆盖手工 weighted mean、全局 class-weight 缩放不变性和 invalid mask。
- 新 `eligible_reference_guard` 强制 `enforce_non_degradation=true`，并在训练启动前验证 full-val report、对应 checkpoint、predictions CSV 三份 SHA-256、报告指标、样本数、test=false 与五类路由 Recall；任何不一致直接失败。
- Stage 041 reference 固定为 Stage 034 独立 ParentGate 同入口报告：组级 DBB P/R/F1=96.98%/98.69%/97.82%，CU=86.09%/72.53%/78.73%；五类正确路由数 Drone=1263/1269、Bird=497/523、Balloon=645/645、Clutter=79/114、Unknown=119/159。
- 五类原始 label 只用于 validation guard/报告；模型 forward 仍只接收 RD tensor、时间、mask、quality、range/velocity 轴和 raw RD amplitude。原始 label、ID、路径、split、TR/pairing 不进入 input/loss/sampler/阈值/后处理。
- 新增默认关闭的 `require_qualified_checkpoint` 保持历史 specialist 配置兼容；Stage 041 显式开启。未达全部 guard 的 epoch只保存 `candidate_*`；只有 `qualified=true` 才保存 `best_*`。summary 明确 `training_outcome`、candidate 与 best，避免把不合格候选当成可部署 best。
- specialist evaluator 同步输出五类路由 Recall，训练 summary 与独立 full validation 可交叉验证。

### 21.3 配置、smoke 与回归

- full config=`config/rd_only/rd_specialist_parent_gate_weighted_ce_fix_stage041_full_seed42.json`，SHA-256=`249475B8ED7CCC5BA654DC0FCD384944E4B468763B3B2885B1448B95AF4A46CB`；18 epochs、batch=4、lr=1e-4、seed=42、CUDA、CU weight=5、`require_qualified_checkpoint=true`、test=false、TR=false、max train/val batches=0。
- 最终 smoke artifact=`artifacts/20260807-073648/rd-specialist-parent-gate-stage041-weighted-ce-fix-smoke-seed42`；candidate SHA-256=`607B665D621CF3ED98148C54FBEB55A823530221FB75F91A7EEAD146F05D091F`。
- smoke：1 epoch、2 train+2 val batches，train loss=0.45546、val loss=0.12190、gradient norm min/mean/max=4.2994/16.8170/29.3347，全部 finite；test split未加载、TR inputs未加载、test evaluation=false、reference provenance verified=true。
- 受限 val 只有8个 DBB样本、CU support=0；系统正确输出 `no_qualified_checkpoint`、`qualified=false`、`best_checkpoint=""`。`tmp/20260807-rd-parent-stage041-weighted-ce-fix-smoke-eval.json` 为8/2710、不完整，不能作为性能证据。
- 定向测试16 passed；全 `tests/test_rd*.py` 105 passed、1个既有 PyTorch nested-tensor warning；compileall、入口 `--help`、`git diff --check` 通过；Stage 034/040 checkpoint CPU strict reload通过。

### 21.4 当前判断

- Stage 041 已达到代码/配置/数据契约/smoke层面的“待全量训练”，尚无 full 性能，不能承诺恢复 Bird或使CU达到90%。
- 整个五分类系统仍不是“待部署”；历史 best hierarchy 的 Bird/Clutter/Unknown仍低于目标，拒识、session、online stable和TR/RD融合均未验证。
- full完成后必须先看 `qualified`：false则保留旧基线并只诊断 candidate；true才允许完整 ParentGate与hierarchy val。任何 Bird/Drone/其他旧安全指标退化立即停止，不读取test调参。
