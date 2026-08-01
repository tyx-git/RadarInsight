# RD-only 全链路审查结论

更新时间：2026-08-02（Asia/Shanghai）

## 1. 核心结论

1. 当前 active RD-only 是严格 DroneTarget/BirdTarget closed-set 二分类，不是五分类，也不是多分支融合：`RD_ONLY_LABELS=("DroneTarget","BirdTarget")`、`PhysicalRDDroneBirdClassifier.label_names=("DroneTarget","BirdTarget")`、`supports_rejection=False`。
2. active forward 只接收 RD tensor 及其合法派生量：6-channel RD、raw RD dB、RD time、padding、range/velocity axes、8 项 RD quality。没有 TR input、TR trajectory、TR match、path、source、ID、split 或 label 参数。
3. 没有发现 active 模型输入层面的 TR 泄漏或 test 调参：trainer 只构建 train/val loader，test 被 config/trainer/evaluator 硬拒绝；normalization/feature stats 都是 train-only。
4. parent/content-group 字符串不编码进模型；它们只按相等关系用于 RD-derived grouped split、等权 loss 和 parent 聚合。`parent_split_group_id` 来源是 RD tensor content fingerprint（少数 fallback 为 source+parent），不是 TR fingerprint。
5. active index 虽包含 BalloonTarget/ClutterTarget/UnknownTarget，但 Physical Dataset 只选择 Drone/Bird；这些类不属于本分支 support、loss、阈值或预测。
6. 当前支路没有 Unknown/OOD/reject/abstain 输出，结论必须是：**当前支路不具备已验证的拒识能力**。
7. 现有完整 v9 validation（非 test）满足用户 90% 门槛；但 window 指标低于历史 v8，且 v9 同时改变多个因素，不能宣称任何单项改动带来提升。
8. 代码、配置、数据 artifact、80 项专项测试和受限 smoke 已闭环，状态为 **待全量训练：是**；全量训练未由本次审查启动。

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
- trainer 新增 non-finite loss/gradient 硬失败；最终 smoke 的 pre-clip gradient norm 最大 92.8206，有限但偏大，现有 clip max_norm=1.0。

### 4.3 Checkpoint

- schema=`physical_rd_drone_bird_checkpoint_v9`。
- 保存 epoch、完整 config、artifact manifest、model state 与 validation。
- loader 对 task/model/data/inference code hash 和 state dict 做 strict 校验。
- 最终 smoke checkpoint与既有 v9 full checkpoint均在当前代码下复载成功。
- 未保存 optimizer/RNG/latest training state，不能从中断 epoch 无缝恢复；这是可靠性限制，不影响一次性挂机训练入口。

## 5. 完整 validation 性能

以下均为 validation，不是 test；百分比由真实 `training_summary.json` 计算，未使用 smoke 替换。

### 5.1 当前 v9 总体指标

| 层级 | N | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Balanced Accuracy | >=90% |
|---|---:|---:|---:|---:|---:|---:|---:|
| window | 1792 | 97.54% | 97.80% | 96.24% | 96.98% | 96.24% | 是 |
| parent weighted-mean-logit | 148 | 99.32% | 99.53% | 98.84% | 99.17% | 98.84% | 是 |

### 5.2 当前 v9 分类别指标

| 层级 | 类别 | Support | Precision | Recall | F1 | 三项均 >=90% |
|---|---|---:|---:|---:|---:|---:|
| window | DroneTarget | 1269 | 97.22% | 99.37% | 98.29% | 是 |
| window | BirdTarget | 523 | 98.38% | 93.12% | 95.68% | 是 |
| parent | DroneTarget | 105 | 99.06% | 100.00% | 99.53% | 是 |
| parent | BirdTarget | 43 | 100.00% | 97.67% | 98.82% | 是 |

### 5.3 当前 v9 混淆

window，row=true、column=pred：

| true/pred | DroneTarget | BirdTarget |
|---|---:|---:|
| DroneTarget | 1261 | 8 |
| BirdTarget | 36 | 487 |

- DroneTarget→BirdTarget=8。
- BirdTarget→DroneTarget=36，是当前主要混淆方向。

parent：

| true/pred | DroneTarget | BirdTarget |
|---|---:|---:|
| DroneTarget | 105 | 0 |
| BirdTarget | 1 | 42 |

### 5.4 source/session/stable

- source：active validation 的 source 全为 `cq08_track`，因此 source-level 数学上等于总体；当前历史 v9 summary 没有独立 source 报告文件，新 evaluator 已补齐，需在用户完成新 full run 后用 `--max-batches 0` 生成正式 source report。
- session：未评估且不可用，原因是 index 无 session 字段；不得以 148 parents 冒充 sessions。
- online stable：未评估且不可用，active Physical 未接生产 stable state machine。
- analyzer 的 cumulative-prefix prediction 是 causal；其“从何时起不再改变”必须看完序列后回溯，只能报告 retrospective stability，不能当线上 latency。
- confidence distribution/完整 ECE：当前 v9 历史 summary 没有 full prediction report；本轮只对 8-window smoke 验证 analyzer 入口，不能得出完整校准结论。

### 5.5 v8 历史对照

| 版本 | 层级 | Accuracy | Macro-P | Macro-R | Macro-F1 | Bird Recall |
|---|---|---:|---:|---:|---:|---:|
| v8 seed-42 | window | 99.16% | 99.18% | 98.79% | 98.98% | 97.90% |
| v9 seed-42 | window | 97.54% | 97.80% | 96.24% | 96.98% | 93.12% |
| v8 seed-42 | parent | 99.32% | 99.53% | 98.84% | 99.17% | 97.67% |
| v9 seed-42 | parent | 99.32% | 99.53% | 98.84% | 99.17% | 97.67% |

- v9 window Macro-F1 比 v8 低约 2.00 个百分点，Bird Recall 低约 4.78 个百分点；这是真实 validation 回归。
- v9 同时改变 pool/attention/token/dropout/prefix/micro/Bird weight/focal，违反单主要因素归因条件；不能断言某个改动是根因。
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

### 7.2 已确认但未阻塞的限制

1. quality 多个维度在 active complete-present train 上为常量，`dynamic_range_proxy` 饱和；它们没有已证明的判别增益。
2. v9 window 指标低于 v8，且改动多因素耦合；需要后续 one-factor 实验，不能在本轮短 smoke 归因。
3. 无 session 字段、无独立 production stable 路径、无 open-set/rejection；这些能力不能报告为已具备。
4. 无 optimizer/RNG resume、scheduler、AMP、early stopping；挂机中断需要从头训练。
5. seed=42 已固定，但代码未强制 bitwise deterministic CUDA algorithms；多次训练可能有小幅非确定性。
6. 最终 smoke pre-clip gradient norm 较大但 finite，需在 full history 持续监控；现有 clipping=1.0。
7. 单一 source=`cq08_track`，不能证明跨 source 泛化；source-level 等于总体不是多域证据。

### 7.3 待核实假设，不作为已确认缺陷

1. dB-domain 重采样是否应改为 linear-power-domain：需要新 store/schema、能量守恒测试和 grouped validation；本轮禁止全量重建，现有表示又有完整一致契约，所以不阻塞。
2. dynamic-range scaling 重设计是否提升 Bird Recall：必须 train-only 拟合、独立配置、一次只改这一项，且 Drone/Bird 任一指标下降即停止。

## 8. 本轮验证与 readiness

- 最终 RD 专项回归：80 passed、1 个 PyTorch warning、0 failed。
- compileall/py_compile/git diff check：通过。
- 最终训练 smoke：`artifacts/20260802-003631/rd-only-stage010-v9-smoke-seed42`；1 epoch、2 train、2 val、test=false、status=completed。
- train smoke 两类 support：Drone=5、Bird=3；loss/gradient finite；checkpoint SHA-256=`69965159c42d0911306ea943ce2e3ac5ef76a4fb78436b38494da3f1f4c56c30`。
- strict evaluator：8 个 balanced val windows、Drone=4/Bird=4、excluded=1784、明确 incomplete/no performance claim。
- analyzer：retrospective/online、calibration/threshold、rejection 边界通过。
- latency：1 warmup+2 samples，入口通过；样本量不足，不作 SLA 声明。
- 既有 v9 full checkpoint epoch 8 在当前代码下 strict reload 通过。
- full config：`config/rd_only/rd_physical_drone_bird_v9_full_seed42.json`。
- 当前状态：**待全量训练：是**。
- **全量训练未由本次审查启动，请用户手动执行。**
