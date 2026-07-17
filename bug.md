# RadarInsight 项目问题审查与修复

> 生成日期：2026-07-17  
> 审查范围：第一阶段 Expert1 数据完整性、RD 采样、TR+RD 融合、评估/选模控制面、系统级路由与长尾类别风险。  
> 约束：忽略 `.history/`；不得将零填充/伪造 RD 输入融合路径；不得用测试集选择 checkpoint、校准超参数、决定发布或导出；未经完整训练不得将 smoke 结果视为模型性能结论。

## 一、使用方式：后续审查与修复提示词

将以下提示词连同本文件交给后续执行者。执行者必须逐阶段更新本文件中的复选框、证据、涉及文件和验证命令；没有证据的事项一律保持 `[ ]`，不得写成已完成。

```text
你正在维护 RadarInsight（低空雷达 MoE 分类系统）。请严格按 bug.md 的阶段顺序执行审查与修复。

硬约束：
1. 忽略 .history/；不要读取、修改、暂存或提交其中内容。
2. 不运行全量 Expert1/System/MoE 训练；只允许单元测试、smoke、import、compile 和静态检查。完整训练由用户执行。
3. 不得用测试集做 checkpoint 选择、阈值/偏置校准、candidate/release/readiness/deployment/export 决策。测试数据只能输出 diagnostics，且必须有：
   {"report_scope": "test", "test_used_for_selection": false}。
4. 不得将 synthetic/zero RD 送入 TR+RD fusion。无真实 RD 样本必须使用 trajectory-only 路径。
5. 外部 checkpoint、checkpoint.config、registry record、调用方传入 calibration lock 和导出源 payload 都是不可信边界；转换为 float 不等于被授权。
6. 不覆盖、不回退、不混入用户已有的无关工作区改动；只暂存本阶段明确列出的文件；不推送。
7. 每个阶段先补失败回归，再最小修复，再运行该阶段测试；完成后在 bug.md 将对应项改为 [x]，并写入“证据”。

完成某项的最低标准：代码实现、反向回归测试、目标 smoke/静态检查通过、bug.md 状态与证据同步。若任一项缺失，保持 [ ] 并注明阻塞原因。
```

---

## 二、结论总览

| 编号 | 阶段 | 状态 | 风险等级 | 结论 |
|---|---|---|---|---|
| P0 | 基线与数据泄漏防护 | 部分完成 | 高 | grouped split 与 source-qualified parent identity 已有保护；RD 窗口与父 RD 的时间同步仍未闭环。 |
| P1 | RD shard-local 采样 | 已完成（需回归保持） | 中 | 已修正 sampler 欠产出/局部性问题并有 smoke 覆盖。 |
| P2 | 数据分布和长尾可辨识性 | 未解决 | 高 | Clutter/Unknown 支持量过低，raw TR+RD 极小，任何单 split 指标均可能高度波动。 |
| P3 | Expert1 TR+RD 联合建模 | 部分完成 | 高 | joint path、质量特征、辅助监督已加入；跨源弱 RD、RD 时间对齐和训练稳定性尚未证明。 |
| P4 | Expert1 校准与验证控制面 | 已完成（focused regression） | 严重 | checkpoint、registry、export、system loader 均执行严格授权净化；release 控制状态在唯一 test 诊断前由 validation 冻结。 |
| P5 | System 路由和跨专家仲裁 | 未解决 | 高 | Aircraft、Ground、低空类别仍会被跨专家竞争压制；样本量使结论不稳定。 |
| P6 | 可复现评估与发布准入 | 未解决 | 高 | 需要验证集唯一所有权、测试诊断隔离、跨 split 验证和受限导出链路。 |

---

## 三、阶段 P0：数据基线、分组切分与 RD 时间语义

### P0-1：父轨迹窗口跨 train/val/test 泄漏

- [x] **已完成：使用 source-qualified parent/window group identity 防止窗口同源泄漏。**
  - 问题：仅按 `parent_sample_id` 或 `sample_id` 前缀分组时，跨数据源同名 parent 会被错误当作同一组，或同一父轨迹的不同窗口会跨 split。
  - 修复位置：`src/core/data/split.py:71-176`。
  - 当前机制：`canonical_group_id()` 优先 parent id，退化为 `|win` 前缀，并使用 `source_name::identity` 隔离跨源同名记录；`validate_grouped_split()` 校验样本唯一、分区不重叠、全覆盖、组不跨 split、组内标签一致。
  - 证据：`tests/test_expert1_split_integrity.py` 覆盖 source-qualified identity、重复 sample、父窗口跨 split、确定性 grouped split、组内标签冲突；此前目标测试报告为 `14 passed`。
  - 后续要求：每次改 index/schema/split 逻辑后重新执行该测试。

### P0-2：CQ08 原始 RD 被父样本复用于窗口，时间同步语义不充分

- [ ] **未解决：RD 片段与 trajectory window 的时间同步裁切尚未闭环。**
  - 影响：当前 CQ08/raw 窗口可能继承父样本完整 RD tensor。即便不会造成 split 泄漏，也会让一个短 trajectory window 看到超出自身时间范围的 RD 信息，导致训练/评估语义偏差及过高的窗口级估计。
  - 风险类别：数据-模态对齐；影响 Drone/Bird、Unknown/Clutter 等依赖微多普勒与局部形态的边界。
  - 相关文件：`src/process/build_expert1_rd_segment_store.py`、`src/core/data/rd_segment_store.py`、`src/core/data/torch_dataset.py`、窗口 index 构建流程。
  - 已知证据：`debug.md` 记录“CQ08/raw 窗口仍继承父样本 RD tensor，尚未做 RD 时间同步裁切”。`src/core/data/rd_segment_store.py:16-25` 已有 `WindowSegmentAssembly` 数据结构与 `rd_time_sync_status_counts`，但应确认完整训练/推理路径是否实际按窗口选择 segment。
  - 修复要求：
    1. 建立 window 起止时间到 RD segment 的显式映射；
    2. 无可对齐真实 RD 时标记为无 RD，走 trajectory-only；
    3. 报告 coverage、同步状态、拒绝/降级原因；
    4. 增加“窗口只访问自身时间范围内 RD segment”的测试，以及“无有效 segment 不进入 fusion”的测试。
  - 完成证据：构建数据的 smoke + dataset/segment-store 回归 + 报告抽样核验。

### P0-3：标签/来源口径的持续一致性

- [x] **已完成（历史修复，需保持）：CQ08 标签与 Unknown 闭环已纳入统一低空五分类。**
  - 历史问题：CQ08 label 6 曾映射为 `OtherTarget`，Unknown 曾被训练 index 默认排除或被 reject 临时折叠到 Clutter。
  - 已落地策略：低空类别固定为 `DroneTarget/BirdTarget/BalloonTarget/ClutterTarget/UnknownTarget`；Unknown 按 reject-style residual 语义处理。
  - 相关文件：标签映射/数据构建链、`src/core/data/labels.py`、`src/core/modeling/experts.py`、`src/core/expert1_trainer.py`。
  - 注意：这不是 Unknown 已学好的证明；P2/P3/P4 仍可能使其召回或精度不足。

---

## 四、阶段 P1：RD shard-local 采样与训练吞吐

### P1-1：RDShardLocalBatchSampler 欠产出与 shard locality 退化

- [x] **已完成：修复 epoch 实际产出少于 `__len__`、以及 RD batch 缺少连续 shard 局部性的风险。**
  - 问题：采样器若按 shard 轮次不严谨地提前结束，会使实际 batch 数和 scheduler/进度统计不一致；若 RD 索引完全散列，RD segment shard cache 命中率下降，训练吞吐不稳定。
  - 修复位置：`src/core/expert1_data.py:122-211`。
  - 当前机制：`num_batches` 依据 RD/no-RD 组成计算；`yield_count < num_batches` 强制精确产出；每个 shard 连续 `shard_group_batches` 次以保持局部性；RD 不足时优先当前 shard，再从其他 RD 补齐，最后才带权重重复当前 shard；no-RD 比例保持。
  - 证据：此前新增 `tests/test_stage2_dataloader_throughput.py` 回归，报告为 `4 passed`；覆盖 sampler 长度、组局部性、随机种子确定性和 RD/no-RD 组成。
  - 回归命令：`python -m pytest tests/test_stage2_dataloader_throughput.py`。

### P1-2：采样器性能/分布风险

- [ ] **待复核：跨 shard 补齐会降低极小 shard 的纯局部性，且每 batch 重算权重可能带来 CPU 开销。**
  - 位置：`src/core/expert1_data.py:186-203`。
  - 影响：这是一项正确性优先的折中，不是已确认 bug；当 shard 很小且 RD 类别极不均衡时，batch 内 RD 将可能来自多个 shard。`rd_weights` 在循环中重建也会影响极大数据集的 CPU 侧开销。
  - 处理原则：只有 profiling 显示 dataloader 是主要瓶颈才优化；优化后必须保持 P1-1 的四项不变量。
  - 验证：记录每 epoch 实际 batch 数、同-shard RD 占比、缓存命中率和 dataloader wall time；不能仅凭训练总耗时判断。

---

## 五、阶段 P2：数据分布、长尾类别和指标可信度

### P2-1：Clutter/Unknown 的极低测试支持使 recall/precision 不稳定

- [ ] **未解决：长尾支持量不足，单一 split 上的精度/召回无法作为发布结论。**
  - 已知 test 分布：Drone=151、Bird=80、Balloon=54、Clutter=7、Unknown=18；raw TR+RD test=4。
  - 影响：Clutter 的一次样本变化约等于 14.3 个百分点 recall 波动；raw 的一次样本变化等于 25 个百分点。用 hard gate 或 checkpoint selection 直接依赖这些小样本指标会严重过拟合。
  - 相关文件：`src/core/expert1_trainer.py` 的 selection/release metrics，`src/core/expert1_config.py` 的最小 support 配置，索引与数据构建流程。
  - 修复要求：
    1. 将小样本 source/label 指标显式标记为 diagnostic 或不确定性指标；
    2. 用 repeated grouped split / bootstrap confidence interval 验证长尾类；
    3. 扩充经过复核的 Clutter、Unknown、raw TR+RD 数据；
    4. 报告每类 support、PR/F1、混淆去向和 source×label 统计，不只报告 accuracy。
  - 验收：不存在任何“仅因 4/7 个样本表现好就发布”的控制规则。

### P2-2：总体 accuracy 被 CQ08 主导

- [ ] **未解决：整体 accuracy/macro 汇总仍可能掩盖弱源与少数类退化。**
  - 影响：CQ08 占主要样本量；Excel 无 RD，raw TR+RD 极少。高总体 accuracy 不代表跨源可用，更不代表 Drone/Bird/Unknown/Clutter 均达到目标。
  - 已知现象：某次 V2 评估中 cq08 final macro F1=0.8287、excel=0.5881、raw=0.0800；joint 在 raw 上存在 harm。
  - 修复要求：把 source-balanced supported macro F1、per-source/per-label recall、RD-on-RD rescue/harm、窗口到父级聚合指标写入候选比较；raw 仅作风险诊断，除非支持量达到预先定义门槛。

---

## 六、阶段 P3：Expert1 TR+RD 融合、训练稳定性与难类边界

### P3-1：RD 直接全类 logit 插值会误伤 trajectory 判断

- [x] **已完成（架构方向已切换，仍需训练验证）：RD 被限制为联合证据/有界 residual，不再作为无限制全类覆盖。**
  - 历史根因：将 RD 作为独立 logits 的后验插值会在弱 source 或低质量 RD 下扭转正确 trajectory；典型现象为 raw Bird trajectory 与 RD evidence 均正确，final 却变为 Drone。
  - 已实施内容：trajectory-only 无 RD 路径、TR+RD joint 路径、joint quality feature、bounded RD residual、RD specialist auxiliary 监督。
  - 相关文件：`src/core/modeling/experts.py`、`src/core/expert1_trainer.py`、`src/core/expert1_config.py`。
  - 不变量：无真实 RD 时 final 必须等于 trajectory branch，不得将零 tensor 当作真实 RD。

### P3-2：RD 对 Bird/Clutter/Unknown 与 Drone/Bird hard pair 的区分仍不足

- [ ] **未解决：RD 质量提升尚未在可靠验证集上证明覆盖所有低空类别目标。**
  - 已知问题：早期 RD-only 质量对 Unknown/Clutter 很弱；Bird/Clutter/Unknown 的 RD-on-RD recall 是持续瓶颈。用户目标是所有类别 recall≥0.90，但目前无法证明达到。
  - 风险：只增加 auxiliary head/权重可能让 RD 抢占 trajectory，而不是提升联合判别；Micro-Doppler 或 RD specialist 必须只做可靠、受限的补充证据。
  - 修复要求：
    1. 在 validation 上分别比较 trajectory、RD-only、joint/final 的 hard-pair confusion；
    2. 记录 rescue/harm 而不是仅看 final F1；
    3. 对 Drone-Bird、Unknown-Clutter 使用受支持量约束的 pairwise 指标；
    4. 检查 RD quality/同步状态、source、窗口长度对错误的条件分布；
    5. 优先改善数据对齐和标注，再调高 RD 权重。
  - 验收：每个被采纳的配置都要证明 final 相对 trajectory 的净收益为正、伤害受控，并对弱源单列风险。

### P3-3：Stage2 过早/过强介入会冲击 trajectory backbone

- [x] **已完成第一轮修复，尚待完整训练验证。**
  - 历史现象：stage2 在短时间内升至 scale=1.0 时，train loss 曾从约 2.43 跳至约 13.31。
  - 已实施：保留 trajectory supervision；trajectory anchor 实际受 start epoch 控制；forced fusion、joint auxiliary、joint reliability 与 deep reroute direct loss 乘 `stage2_scale`。
  - 相关文件：`src/core/expert1_trainer.py`、`src/core/expert1_config.py`。
  - [ ] **待完成：记录细粒度 loss component history。**
    - 当前训练历史不足以区分 RD auxiliary、forced fusion、joint auxiliary、deep reroute 哪项导致震荡。
    - 修复要求：向 `training_history.json` 写入主损失、trajectory、RD、joint、anchor、deep-reroute、residual 等分量，并按 stage 记录 scale。

### P3-4：训练与评估路径存在可避免性能开销

- [x] **已完成：提供训练/评估性能控制开关。**
  - 已实施：避免默认逐 batch CPU 全模型 rollback；评估可默认不计算 auxiliary loss；支持 quick validation/周期 full validation。
  - 相关文件：`src/core/expert1_config.py`、`src/core/expert1_trainer.py`。
  - [ ] **待验证：性能优化不得跳过最终 checkpoint 所需的完整 validation。**
    - quick val 只能辅助训练观察；checkpoint 候选与 release 控制必须使用完整 validation。

---

## 七、阶段 P4：Expert1 校准、checkpoint 和评估控制面（当前最高优先级）

### P4-1：有限但未经授权的 precision bias 可穿透不可信边界

- [x] **已完成：checkpoint/config/registry/caller/system loader 提供的 bias 均经严格授权验证。**
  - 严重性：严重。恶意、损坏或过期 checkpoint 可改变 Drone/Bird/Balloon 预测，或持久化非法偏置；这会破坏验证集唯一所有权。
  - 当前缺陷位置：
    - `src/core/expert1_trainer.py`：`_precision_logit_bias_from_checkpoint()` 目前采用转换型 `_normalize_precision_logit_bias()`，不保证政策授权；
    - `src/core/expert1_trainer.py`：`_locked_precision_parameters()` 当前只检查长度、有限性与 scale 范围；
    - `src/core/expert1_trainer.py`：`_build_expert1_checkpoint_payload()`、`_save_checkpoint()` 仍可能信任调用方 lock；
    - `src/core/reselect_expert1_precision.py`：registry recovery/export 需拒绝非法 validation record，并重写所有陈旧校准位点。
  - 授权策略（必须精确遵守，顺序为 `LOW_ALTITUDE_CLASSES`）：
    - DroneTarget = `0.0`；BirdTarget = `0.0`；BalloonTarget = `0.0`；
    - ClutterTarget ∈ `{-0.05, 0.0, 0.05}`；
    - UnknownTarget ∈ `{-0.05, 0.0, 0.05, 0.10, 0.15}`；
    - 必须恰好五个有限值；任何 malformed/non-finite/未授权值都回退 `[0.0]*5`。
  - 修复要求：
    1. 保留内部受控转换 helper，但新增严格“授权验证” helper；
    2. 对顶层 checkpoint、嵌套 config、caller lock、registry record、export source payload 全部经严格 helper；
    3. 非法 validation provenance 必须排除 candidate，不能静默降级成有效 validation；
    4. config-default fallback 必须是合法 scale + 全零 bias；
    5. 导出时改写 top-level scale/source/bias、`locked_precision_parameters`、nested config scale/bias 与按标签字段，禁止遗留旧值。
  - 必测反例：`[9.0]*5`、非零 Drone/Bird/Balloon、非法有限 Clutter/Unknown、NaN/Inf、长度错误、不可迭代/字典异常、顶层与 nested config 注入、registry 注入、caller lock 注入。
  - 证据：`_authorized_precision_logit_bias()` 仅允许指定五维策略；`test_precision_bias_boundaries_reject_unauthorized_values`、`test_checkpoint_payload_rebuilds_caller_lock_and_nested_config` 与 `test_system_loader_rejects_unauthorized_checkpoint_bias` 通过。

### P4-2：测试集可能参与 release/candidate/readiness/bottleneck 决策

- [x] **已完成：`train_expert1_v1()` 先由完整 validation 冻结控制面，test 后仅写入诊断报告。**
  - 严重性：严重。测试集被用于 release gate、candidate status、next action 或 bottleneck 归因时，会形成评估泄漏并让后续调参污染 test。
  - 当前问题位置：`src/core/expert1_trainer.py` 的最终 checkpoint 加载、`loaders.test_loader` `_run_epoch`、release gate、candidate status、报告组装区域。
  - 修复顺序（必须保持）：
    1. 读取并严格净化 selected checkpoint 的 validation probe 与 calibration lock；
    2. 在唯一一次 test `_run_epoch` 前，从 validation 构建全部控制状态：选择资格、release gate、candidate/release 状态、readiness、bottleneck summary、export/deployment eligibility、next action；
    3. 精确执行一次 test loader；
    4. 把原始 test 观测写入 `test_reporting`，强制添加 `report_scope="test"` 与 `test_used_for_selection=false`；
    5. 静态和运行时保证 test-derived variable 不能回流到任一控制状态。
  - 允许 test reporting 包含：混淆矩阵、预测、weak-class 指标、per-source、窗口聚合、诊断统计。
  - 禁止 test reporting 决定：checkpoint selection、calibration、candidate/release、release eligibility/gate、readiness、deployment/export、bottleneck summary、next action。
  - 必测要求：AST/static 约束“test metrics 定义后不得流入上述控制变量”；精确一个 test loader `_run_epoch` 调用。
  - 证据：`validation_control_metrics` 在唯一 `loaders.test_loader` 调用前构建 release gate、candidate status、readiness、bottleneck 和 next action；`test_test_reporting_path_cannot_export_or_select_calibration` 通过。

### P4-3：checkpoint registry/reselector 的 provenance 与持久化不一致

- [x] **已完成：registry record 仅接受严格 validation lock，export 从净化 lock 重建全部 calibration 投影。**
  - 影响：训练保存、手工 reselect 和导出若各自解释字段，会造成同一 checkpoint 在不同入口下使用不同 scale/bias/provenance。
  - 修复要求：
    1. `_save_checkpoint()` 从重建后的安全 lock 序列化，包含 `locked_precision_parameters`；
    2. reselector 仅接受合法 validation lock，非法记录从 candidate 集排除；
    3. 允许 export fallback 使用 config-default 全零校准，但必须显式重写 provenance，不能伪称 validation；
  4. 所有 report/registry 都保留 `test_used_for_selection=False`。
  - 证据：`_locked_calibration_from_record()` 拒绝无 lock、非 validation 或未授权记录；`_write_precision_checkpoint()` 重写 top-level、nested config、按标签字段、lock 和 provenance；三个 reselector 回归测试通过。

### P4 当前 TDD 状态

- [x] 已完成：在 `tests/test_expert1_precision_controls_smoke.py` 添加 `_precision_logit_bias_from_checkpoint` import，为红测准备。
- [x] 已完成：添加 P4-1/P4-2/P4-3 失败回归，并确认旧实现拒绝测试失败。
- [x] 已完成：实现严格授权 helper 并接入 checkpoint、config、caller lock、registry、export 与 system loader。
- [x] 已完成：重构 final train/test/report 控制流，validation 先拥有控制面。
- [x] 已完成：P4 focused smoke 16 项、`py_compile` 与 AST 控制面检查通过。
- [x] 已完成：完成 specification、correctness/security/performance 五轴复核；未运行全量训练，完整 smoke 中两项既有 P3 指标断言仍漂移。

---

## 八、阶段 P5：System 路由、跨专家竞争与开放集风险

### P5-1：System 跨专家 logits/仲裁压制单专家正确结果

- [ ] **未解决：Expert2/Expert4/Expert1 的单专家能力不能直接推导 System 最终输出。**
  - 已知现象：Expert2 单独 Aircraft recall 曾恢复，但 System 仍会将真实 Aircraft 判为低空类别；Ground 路线也会受 system-level arbitration 压制。
  - 影响：route accuracy 可虚高；最终 9 类性能受路由、reroute、logit 比较和规则覆盖共同决定。
  - 相关文件：`src/core/modeling/systems.py`、`src/core/modeling/routing.py`、`src/core/modeling/experts.py`、`src/core/training/system.py`。
  - 修复要求：
    1. 分开报告 expert-only、routed-expert、final-system 三层混淆矩阵；
    2. 对每条错例记录初始 route、reroute 次数、专家 logits、规则 override、最终来源；
    3. 用 validation 选择 arbitration 策略，禁止从 test 选择；
    4. 各 route 使用最小 support 和 source 约束，避免 CQ08 主导总分。

### P5-2：Aircraft/ground 长尾及阈值可行域不足

- [ ] **未解决：Aircraft 的 precision/recall 与 raw false-positive 之间尚未证明存在稳定可行阈值；People/Ship 数据不足。**
  - 已知风险：Aircraft test support 极小；People 总量曾只有 2；raw source 非 Aircraft 与 Aircraft 相似度重叠；Ship/raw source recall 不稳定。
  - 修复方向：新增/清洗样本、repeated grouped split、按 source×label 审核 false positive；不要仅降低阈值换 recall 或仅升阈值换 precision。

---

## 九、阶段 P6：验证、发布与实验治理

### P6-1：候选资格与报告口径未完全统一

- [ ] **未解决：所有候选必须以完整 validation 的固定指标、固定数据版本和已净化 calibration 决策。**
  - 最低报告字段：数据/索引版本、split seed、每类 support/precision/recall/F1、每 source 与 source×label、trajectory vs joint/final、RD rescue/harm、校准 provenance、候选资格原因、test diagnostics 隔离字段。
  - 禁止：把 quick val 当最终决策；把 smoke 训练结果标为正式；将 raw 4 条结果当总体准入/否决唯一依据。

### P6-2：当前工作区含大量无关未提交改动

- [ ] **待处理：Task P4 的提交必须保持三文件范围，不污染已有改动。**
  - 当前允许提交范围：
    - `src/core/expert1_trainer.py`
    - `src/core/reselect_expert1_precision.py`
    - `tests/test_expert1_precision_controls_smoke.py`
  - 注意：工作区已有配置、兼容层、tmp 文件等无关变更；不得使用 `git add .`、`git add -A`、reset/restore/clean 覆盖用户工作。
  - 完成条件：focused tests 通过后，仅 stage 以上三文件并建立本地 commit；不 push。

---

## 十、推荐执行顺序

1. [ ] **先完成 P4 控制面。**这是唯一会直接让评估结论失真的严重工程问题；未完成前，禁止根据测试表现调 precision bias 或宣布 release candidate。
2. [ ] **完成 P0-2 RD 时间同步。**在模态证据时间范围不可信时，不应继续增加 RD 权重或 specialist head。
3. [ ] **完成 P2 数据支持与重复 grouped validation。**先量化不确定性与数据缺口，再设类别目标与 gate。
4. [ ] **完成 P3 联合模型 ablation。**仅用 validation 比较 trajectory、joint、RD quality、pairwise rescue/harm，保留能够稳定净增益的最小设计。
5. [ ] **完成 P5 system-level trace 与仲裁验证。**确认单专家进步能够传递到最终九分类输出。
6. [ ] **完成 P6 准入规则与本地提交。**由 validation 选择、test 只报告，训练结果才可被可复现地比较。

---

## 十一、每阶段落地记录模板

后续执行者必须将以下模板复制到对应条目下，不得只在对话中报告。

```markdown
### 执行记录：<P编号/子项>
- 状态：`[ ] 未完成` / `[x] 已完成` / `[ ] 阻塞`（阻塞时说明原因）
- 修改文件：`path:line-line`
- 测试文件：`path::test_name`
- 失败回归证据：<旧实现失败的命令与关键输出>
- 修复说明：<最小、可验证的行为变化>
- 通过证据：<命令、日期、结果>
- 未覆盖风险：<仍不能由本次验证证明的边界>
- 后续动作：<明确下一步；若已完成则写“无”>
```

## 十二、当前验证命令清单

```bash
# P1：RD shard-local sampler
python -m pytest tests/test_stage2_dataloader_throughput.py

# P0：分组切分完整性
python -m pytest tests/test_expert1_split_integrity.py

# P4：校准与测试隔离控制面（实施后使用）
python -m pytest tests/test_expert1_precision_controls_smoke.py

# 仅静态/导入检查；不运行全量训练
python -m py_compile src/core/expert1_trainer.py src/core/reselect_expert1_precision.py
python -c "import src.core.expert1_trainer; import src.core.reselect_expert1_precision"
```

> 说明：上述命令通过只证明代码路径与回归约束；不证明模型达到 Drone/Bird/Balloon/Clutter/Unknown 全类别 recall≥0.90。该目标必须在用户执行的完整、版本固定、grouped validation/repeated split 训练评估中验证。
