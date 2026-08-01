# RD-only 全链路审查与待全量训练计划

更新时间：2026-08-02（Asia/Shanghai）
当前交付状态：**待全量训练：是**。代码、配置、数据契约、80 项 RD 专项测试、受限训练 smoke、checkpoint strict reload、受限 evaluator/analyzer 与 2-sample latency smoke 均通过；全量训练与完整新格式 validation 未由本次审查启动。
任务边界：active 分支是 `PhysicalRDDroneBirdClassifier` 的 DroneTarget/BirdTarget closed-set 二分类；`RD_ONLY_LABELS=("DroneTarget","BirdTarget")`，`label_names=("DroneTarget","BirdTarget")`，`supports_rejection=False`。禁止把五分类 Expert1、TR-only、legacy patch/novel 分支或 smoke 指标混入 active 能力结论。

## Stage 001 冻结当前 RD-only 基线

依赖：项目指令、工作区状态、active 入口、配置、数据 artifact、checkpoint 元数据。
证据：`src/core/train_rd_physical_drone_bird.py`、`src/core/rd_physical_drone_bird_*.py`、v8/v9 `training_summary.json`、两份独立配置。
产物：冻结调用链、标签/shape/轴/seed/epoch/checkpoint 规则与完整 validation 基线。
验收：入口和实际调用一致；test 未加载；二分类与其他分支分开。

- [x] 确认 active 训练链：`train_rd_physical_drone_bird -> PhysicalRDWindowDataset -> PhysicalRDDroneBirdClassifier -> rd_physical_drone_bird_trainer`；依赖为实际 import；证据为源码引用；产物为唯一 active 链；验收为线上/legacy 扫描无隐式替代入口。
- [x] 冻结标签与能力：仅 DroneTarget/BirdTarget，输出顺序 Drone=0、Bird=1，`supports_rejection=False`；证据为模型类、dataset encoder、artifact task contract；验收为无五分类扩展。
- [x] 冻结输入：`[batch,time,6,21,384]`，range offset `[-75,75] m`，velocity `[-85,85] m/s`；证据为 dataset/model/schema 与 canonical summary；验收为 shape/axis 单测通过。
- [x] 冻结数据版本：window index v3.1、normalization v4、feature stats v4、canonical frame index v3.1；证据为 SHA-256；验收为 strict artifact loader 通过。
- [x] 冻结训练预算：seed=42、CUDA、batch=4、18 epochs、AdamW lr=1e-4、weight_decay=1e-4；证据为 `rd_physical_drone_bird_v9_full_seed42.json`；验收为配置类实载通过。
- [x] 冻结 checkpoint 选择：依次比较 parent Macro-F1、最低分类别 parent Recall、最低分类别 parent Precision、window Macro-F1；证据为 `_is_better` 与 progress/summary；验收为单测通过。
- [x] 冻结现有完整 validation：v8 best epoch 16、v9 best epoch 8，二者均 `test_evaluation_executed=false`；证据为各自 `training_summary.json`；验收为不得称为 test 指标。

## Stage 002 审查数据、标签、split 和泄漏

依赖：Stage 001、window index、causal manifest、v8 RD-content split ancestor、normalization/feature stats。
证据：CSV/JSON 只读复核、Stage 025/026 audit、dataset 前向签名单测。
产物：split/内容组/未来帧/统计拟合/无 TR 边界结论。
验收：parent/content group 跨 split 为 0；test 不参与训练、选择、校准或阈值。

- [x] 逐 split 核对 active trainable Drone/Bird：train 8207 windows、val 1792、test 1820；依赖为 index；证据为 `Import-Csv` 分组；产物为支持数；验收为 trainer 只构建 train/val loader。
- [x] 核对 parent 隔离：跨 split parent=0；证据为当前 index 全量分组和 Stage 026 audit；验收为 0。
- [x] 核对 RD content group 隔离：跨 split group=0、group-label conflict=0；证据为 `parent_split_group_id` 分组与 v8 RD tensor fingerprint audit；验收为 0。
- [x] 核对重复内容：已知重复组由相同 RD tensor 指纹形成，不是 TR 指纹；证据为 Stage 025 manifest/audit；验收为只用于 split 与等权训练，不作为 forward 数值。
- [x] 核对 causal window：noncausal window=0、最大 index/tensor 时间边界误差 `2.3803711e-07 s`、future frame=0；证据为 Stage 026 structural audit；验收为窗口结束不晚于请求时点。
- [x] 核对统计防泄漏：normalization 与 feature stats 均声明 split=train、Drone/Bird、713 parents、712 groups、test=false；证据为 v4 JSON；验收为 index/hash/generator hash strict 校验通过。
- [x] 核对字段边界：路径、文件名、sample/parent ID、source、split、label 未进入 model forward；证据为 forward signature 单测；验收为仅 RD tensor/轴/时间/质量派生量进入模型。
- [x] 核对 parent/group 的合法用途：ID 仅按相等关系用于每 parent/content-group 等权 loss 与 parent 聚合，原始字符串不编码进模型；证据为 `_parent_balance_metadata`/`compute_parent_evidence_weights`；验收为 disclosure 完整且无 TR pairing 字段。
- [x] 核对 RD/TR pairing：active index `rd_fusion_eligible=0`，dataset 遇 fusion eligible 会拒绝；`rd_pairing_contract.py` 未被 active 调用；验收为配对状态不进 input/loss/threshold/postprocess。

## Stage 003 审查 RD 物理数据和特征提取

依赖：canonical frame index/tensors、normalization/feature stats、builder 与 statistics 源码。
证据：106026-frame merge summary、坐标审计、单元测试、feature quality audit。
产物：轴、单位、相位、dB、缺失、统计特征与质量特征结论。
验收：现有 store 结构完整；未来构建不得 fallback 到 track range。

- [x] 核对 canonical store：106026 frames、1549 parents、1549 tensor files、2,773,411,553 bytes；missing/extra/duplicate/tensor mismatch/invalid shape/partial coverage 均 0；证据为 merge summary 与 artifact inventory；验收为 strict loader 通过。
- [x] 核对 Doppler/velocity：384 bins、`[-85,85] m/s`、递增方向，无转置/翻转；证据为 builder/dataset shape checks；验收为 axis 单测通过。
- [x] 核对 range：21 bins、target-relative `[-75,75] m`、递增方向；证据为 canonical summary/contract test；验收为目标峰居中测试通过。
- [x] 核对相位：complex RD 必须存在，按 sin/cos 插值后 atan2；缺相位不再补零；证据为 builder 与 missing-phase test；验收为圆周相位测试通过。
- [x] 修复 range reference：未来构建强制 `debug_info.detected_range_m`，删除 `track_range_m` fallback，并记录 `target_range_reference_source`；证据为源码与拒绝 fallback 测试；验收为现有 106026/106026 metadata 均有 detected range，故本轮无需重建。
- [x] 核对时间：累计 CPI acquisition duration，不是 wall-clock/TR 同步；现有 tensor 严格递增；证据为 builder、Stage 026 time audit、dataset guard；验收为 nonmonotonic parent=0。
- [x] 核对特征接线：6 通道、14 项物理统计、8 项质量量、raw dB micro profile 都进入最终或辅助分支；证据为 forward 逐层调用；验收为梯度/forward smoke 有限。
- [x] 标记质量特征饱和：train artifact 中 `dynamic_range_proxy.unique_count=1`，其余多个完整性量也为常量；证据为 feature stats v4；产物为非阻塞限制；验收为不得宣称这些常量提供判别增益。
- [ ] 后续单因素实验：比较 dB-domain resampling 与 linear-power-domain resampling；依赖为新 store/stats 独立版本和用户允许全量重建；证据要求为能量守恒测试与 grouped validation；停止条件为任一 Drone/Bird 指标下降或无法证明物理定义。当前仅为合理假设，不是已确认缺陷，不阻塞本轮训练。
- [ ] 后续单因素实验：重新设计饱和的 dynamic-range quality scaling；依赖为新 feature-stats schema；验收为 train-only 拟合、无 test、质量分布非退化且两类均不下降。当前不阻塞。

## Stage 004 审查 dataset、窗口、padding 和时间处理

依赖：Stage 002/003、dataset/collate/model。
证据：dataset guards、shape/mask/padding/time 单测与 smoke。
产物：端到端 shape/mask 契约。
验收：padding/未来帧不参与 GRU、attention、pooling、frame/prefix/micro 分支。

- [x] 验证 item `[time,6,21,384]`、batch `[batch,time,6,21,384]`、raw dB `[batch,time,21,384]`；验收为 dataset/collate tests 通过。
- [x] 验证 padding mask：True 表示 padding；CNN token 清零、packed GRU 用真实 length、mean/max/attention 排除 padding；验收为 padding invariance/all-padding tests 通过。
- [x] 验证 prefix final index：`lengths-1` 只取末个有效 frame logit；验收为无 padding frame loss。
- [x] 验证 micro profile：使用 raw dB 转相对功率、真实 time 与 mask；验收为时间顺序/时间间隔敏感测试通过。
- [x] 验证边界样本：空 batch 拒绝、missing RD 走显式 all-padding 且 active config 禁用、time duplicate/倒序拒绝、短于 12 帧排除；验收为相关单测与 index summary 一致。

## Stage 005 审查 RD 编码和模型前向链路

依赖：Stage 004、model/statistics/artifacts。
证据：源码逐层审查、model contract、real smoke forward。
产物：分支接线和梯度结论。
验收：输出顺序、mask、softmax/logit、辅助头和最终头一致。

- [x] 审查 anisotropic `(3x7)` CNN、avg/max spatial pool `(6,16)`、token projection；验收为 6 通道固定且无轴错位。
- [x] 审查 time projection、causal GRU、hidden/mean/max/attention；验收为无 future token、mask 维度正确。
- [x] 审查 14 维 statistics 与 8 维 quality fusion；验收为 shape/finite guards 和最终 classifier 接线存在。
- [x] 审查 micro-Doppler 分支与 micro logits；验收为进入 classifier 且有独立 aux loss。
- [x] 审查 frame/prefix logits；验收为 frame head 有 prefix aux loss，padding frame logits 归零。
- [x] 审查标签与 checkpoint 输出顺序；验收为 dataset/model/artifact 都是 Drone=0、Bird=1。
- [x] 审查 legacy residual/gate：patch residual 是旧 v7/TR-window anchor 二分类诊断，不进入 active；novel parent 是旧 Physical-v2；anchor residual trainer 明确 retired；验收为分支结果不得混算。

## Stage 006 审查训练目标、loss、采样和 checkpoint

依赖：Stage 002/005、trainer/config。
证据：loss 源码、weight audit、checkpoint round-trip、最终 smoke history。
产物：loss/权重/优化/选择/恢复能力结论。
验收：无重复主 CE、无 padding loss、两类均反传、best checkpoint 可复载。

- [x] 拆分 loss：main focal CE + `0.15*prefix CE + 0.25*micro CE`；证据为 full config/trainer；验收为仅各自 logits 计算一次。
- [x] 核对 class/content/parent/evidence 权重：content/group 与 parent 等权、prefix 按 frame evidence 限幅、Bird class weight=1.35；验收为权重单测与 audit 通过。
- [x] 核对 missing/padding：`rd_present` mask 使缺失窗口 loss/metrics/aggregation weight 为 0；active `include_missing_rd=false`；验收为回归测试通过。
- [x] 增加 finite loss/gradient guard 与 gradient norm 记录；最终 smoke 两次 pre-clip norm=21.1628/92.8206，均有限，现有 clip max_norm=1.0；验收为 status=completed。
- [x] 核对 optimizer：AdamW，无 AMP/scheduler/early stopping；证据为 trainer；验收为配置与风险披露。
- [x] 核对 checkpoint：保存 model/config/artifact manifest/validation，strict inference reload 通过；验收为最终 smoke 与已有 v9 full checkpoint 都成功复载。
- [x] 核对 test：`run_test_evaluation=true` 在 config 和 trainer 均拒绝；验收为 test loader 未构建。
- [ ] 后续增强：保存 optimizer/scheduler/RNG latest checkpoint 以支持中断续训；依赖为新 checkpoint schema 和 round-trip 测试；当前仅支持 best inference checkpoint 恢复，不阻塞一次性全量训练。

## Stage 007 审查模型性能评估

依赖：已有完整 validation summary、修复后的 evaluator/analyzer。
证据：v8/v9 training summaries、80 项测试、restricted reports。
产物：window/parent/分类别/混淆/source/session/stable 口径。
验收：不使用 test，不把 incomplete smoke 当性能。

- [x] 完整 v9 validation（1792 windows/148 parents）主指标与两类 P/R/F1 均 >=90%；证据为用户此前手动完成的 v9 summary；验收为 best epoch=8、test=false。
- [x] 完整 v8 validation 作为历史对照；证据为用户此前手动完成的 v8 summary；验收为不通过 legacy checkpoint 冒充当前 v9 复载结果。
- [x] 补齐 Macro-Precision、Macro-Recall、Balanced Accuracy、confusion 与双向错误；证据为 `_class_metrics` 和单测。
- [x] 补齐 source 分层；active source 唯一为 `cq08_track`；证据为 index 与 restricted evaluator；验收为 source 仅用于报告不进 forward。
- [x] 明确 session 不可用：index 无 `session_id/session_name`，不得以 parent 代替；证据为 header/evaluator；验收为 `available=false`。
- [x] 修正 stable 语义：累计 prefix 是 causal，但“最后一次翻转后稳定时间”只能完整序列后回溯；证据为 analyzer 字段；验收为线上 stable 明确 `available=false`。
- [x] 增加 evaluator `--max-batches` 与确定性两类均衡 smoke 子集；验收为 incomplete/excluded count 明确且单测通过。
- [ ] 用户完成全量训练后运行完整新格式 validation（`--max-batches 0`）并生成 predictions/source/analysis；依赖为 Stage 011 checkpoint；验收为 1792 个 val window 全覆盖、excluded=0、仍不运行 test。

## Stage 008 审查拒识、开放集和校准

依赖：active model/config/evaluator、项目 reject/OOD 全局搜索。
证据：`supports_rejection=False`、task contract、无 Unknown head/threshold/abstain 输出。
产物：拒识能力边界。
验收：不得生成虚假 Coverage/FAR/FRR/Selective Risk。

- [x] 搜索 reject/enable_reject/Unknown/OOD/energy/margin/confidence/abstain；结果位于 Expert1/System 五分类链，不属于 active Physical 二分类。
- [x] 确认 active 无 reject threshold、UnknownTarget logit、OOD score、abstain action；验收结论为“当前支路不具备已验证的拒识能力”。
- [x] 确认 calibration analyzer 只报告未校准 softmax 的 validation 诊断，`posthoc_calibration_applied=false`、`threshold_selection_performed=false`；验收为不使用 test 调参。
- [x] 确认 Balloon/Clutter/Unknown 仅存在于全 index/其他分支，active Dataset 严格过滤；验收为不混入二分类 support/分母。

## Stage 009 执行手术式优化

依赖：Stage 001–008 的已证实问题。
证据：代码 diff、单测、smoke artifact。
产物：最小修改集与独立配置。
验收：不改标签、不删困难样本、不改 test 分母、不引入 TR。

- [x] 修复 evaluator 报告契约；文件为 `rd_physical_drone_bird_trainer.py`、`evaluate_rd_physical_drone_bird.py`；验收为指标/source/session/completion/rejection 字段测试通过。
- [x] 修复 analyzer 对 retrospective stable 的命名与线上声明；文件为 `analyze_rd_physical_validation.py`；验收为在线 stable unavailable 测试通过。
- [x] 修复未来 canonical builder 的 track-range fallback；文件为 `build_cq08_rd_physical_canonical_store.py`；验收为 missing detected range 明确失败。
- [x] 增加 loss/gradient finite guard、progress artifact 与 checkpoint 选择说明；验收为最终 smoke history/progress 完整。
- [x] 新建 smoke/full 独立配置；验收为 config loader 和 `--help` 通过，full 与已验证 v9 超参数逐字段一致，只有输出目录独立。
- [ ] 后续 one-factor 性能实验：v9 相对 v8 同时改变 pool/attention/token/dropout/prefix/micro/Bird weight/focal，无法归因；依赖为每次只改一个主要因素、独立 artifact；停止条件为任一主要指标或 Drone/Bird P/R/F1 <90%。

## Stage 010 Smoke 验证和待训练准备

依赖：Stage 009 代码/配置。
证据：80 passed、compileall、最终 smoke/evaluator/analyzer/latency JSON。
产物：可执行入口、checkpoint、日志/progress 与 readiness 结论。
验收：不启动全量训练/重建/完整长评估/test。

- [x] 静态编译：`python -m compileall -q src/core src/process config` 通过；验收为 exit 0。
- [x] RD 专项回归：80 passed、1 个 PyTorch nested-tensor warning；验收为 0 failed。
- [x] 最终训练 smoke：1 epoch、2 train batches、2 val batches、batch=4、train/val only；artifact=`artifacts/20260802-003631/rd-only-stage010-v9-smoke-seed42`；验收为 status=completed。
- [x] 两类反传：train support Drone=5、Bird=3，loss finite，gradient finite；验收为 checkpoint 已保存。
- [x] checkpoint strict reload + evaluator：8 个 balanced val windows，Drone=4/Bird=4，excluded=1784，明确 no performance claim；验收为 exit 0。
- [x] analyzer：校准仅诊断、无 threshold、retrospective/online 边界明确；验收为 exit 0。
- [x] latency smoke：1 warmup+2 measured，batch=1，finite logits；验收为 exit 0；不作为生产 stable 证据。
- [x] 已有 v9 full checkpoint 在当前代码下 strict reload smoke 通过；验收为 schema v9、epoch 8。
- [x] 待训练条件：入口/config/data hashes/seed/device/batch/epochs/lr/checkpoint/test-off/rejection-off/artifact 规则均固定；验收为“待全量训练：是”。

## Stage 011 输出人工挂机训练命令

依赖：Stage 010 readiness、实际 `--help`。
证据：解释器 `.venv\Scripts\python.exe`、入口只支持 `--config`。
产物：前台/后台/日志/状态/完整 validation 命令。
验收：本次审查不执行这些全量命令。

- [x] 生成并复核前台 PowerShell 命令；配置为 `config/rd_only/rd_physical_drone_bird_v9_full_seed42.json`，日志路径固定。
- [x] 生成并复核后台 `Start-Process -WindowStyle Hidden` 命令，stdout/stderr 分离并保存 PID。
- [x] 生成日志 tail、PID 状态和训练后 validation/analyzer 命令；validation 强制 `--split val --max-batches 0`。
- [ ] 用户手动启动全量训练；依赖为本次最终答复命令；本次审查明确不代为启动。
- [ ] 用户训练完成后运行完整 validation/analyzer 并审查两类是否仍 >=90%；若任一低于 90%，停止进入 test/线上部署并回到 Stage 009 单因素实验。
