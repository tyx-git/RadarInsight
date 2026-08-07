# RD-only 全链路审查、全量训练验收与后续评估计划

更新时间：2026-08-07（Asia/Shanghai）
当前交付状态：**Stage 040 ParentGate CU-weight=5 full 已由用户完成并经完整 validation 验收，结果改善 CU window Recall 但降低 Bird 路由 Recall、总体/parent Macro-F1，已按停止条件拒绝替换 Stage 034 安全基线。Stage 041 已完成 weighted-CE 归一化、可验证基线 provenance、五个原始类别路由非劣化守卫、候选/合格 checkpoint 语义修复、smoke/strict reload 与 105 项 RD 回归，当前“待用户手动全量训练：是”；整个五分类 RD-only 仍“待部署：否”，拒识、production stable 与 TR/RD 融合仍未验证。** 历史 Stage 011 二分类、Stage 022/034/037/038 五分类和 Stage 039/040 ParentGate 结果分别冻结，不能互相冒充能力证据。
任务边界：历史 active 分支仍是 `PhysicalRDDroneBirdClassifier` 的 DroneTarget/BirdTarget closed-set 二分类；`RD_ONLY_LABELS=("DroneTarget","BirdTarget")`、`label_names=("DroneTarget","BirdTarget")`、`supports_rejection=False`。新目标为严格使用 `LOW_ALTITUDE_CLASSES` 顺序的五分类 RD-only：`DroneTarget, BirdTarget, BalloonTarget, ClutterTarget, UnknownTarget`（代码中的规范拼写为 `UnknownTarget`，不采用用户文字中的 `UnknowTarget`）。设计采用共享 RD 编码器、软级联概率头与直接五分类 flat 旁路，保留 TR 入口但训练阶段 RD-only 不读轨迹；未来融合采用 availability-aware late fusion，TR 缺失时严格退化为 RD-only，RD 缺失时严格退化为 TR-only，双缺失只输出 `abstain/unavailable`。禁止把五分类 Expert1、TR-only、legacy patch/novel 分支或 smoke 指标混入五分类 RD-only 能力结论。

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
- [x] 冻结完整 validation：历史 v8 best epoch 16、历史 v9 best epoch 8、Stage 011 v9 best epoch 13，三者均 `test_evaluation_executed=false`；证据为各自 `training_summary.json`；验收为历史对照与当前 checkpoint 分开且不得称为 test 指标。

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
- [x] 核对 parent/group 的合法用途：当前代码中 ID 仅按相等关系用于 split 审计、排序和 parent 报告聚合；五分类/specialist loss 使用窗口自身 evidence，不使用 parent/content ID，原始字符串不编码进模型；证据为 `_evidence_weights`、ID 改写不变性测试和 Stage 033 修复记录。
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
- [x] 增加 finite loss/gradient guard 与 gradient norm 记录；Stage 011 全量训练 18 个 epoch 的 train/val loss 与每 epoch gradient audit 均有限，最大 batch pre-clip norm=674.6139（epoch 8），现有 clip max_norm=1.0；验收为 `training_progress.status=completed`，并将梯度峰值列为剩余风险而非训练失败。
- [x] 核对 optimizer：AdamW，无 AMP/scheduler/early stopping；证据为 trainer；验收为配置与风险披露。
- [x] 核对 checkpoint：保存 model/config/artifact manifest/validation，Stage 011 best checkpoint SHA-256=`07A14677016D9E0A39208277F63ED5E7BB6F1311D12131A98C78953636443BF2`，在当前代码下 strict inference reload + `--max-batches 2` 通过。
- [x] 核对 test：`run_test_evaluation=true` 在 config 和 trainer 均拒绝；验收为 test loader 未构建。
- [ ] 后续增强：保存 optimizer/scheduler/RNG latest checkpoint 以支持中断续训；依赖为新 checkpoint schema 和 round-trip 测试；当前仅支持 best inference checkpoint 恢复，不阻塞一次性全量训练。

## Stage 007 审查模型性能评估

依赖：已有完整 validation summary、修复后的 evaluator/analyzer。
证据：v8/v9 training summaries、80 项测试、restricted reports。
产物：window/parent/分类别/混淆/source/session/stable 口径。
验收：不使用 test，不把 incomplete smoke 当性能。

- [x] 验收 Stage 011 完整 v9 validation：1792 windows、148 parents、excluded=0；window Accuracy=97.99%、Macro-P/R/F1/Balanced Accuracy=97.57%，parent Accuracy=98.65%、Macro-P/R/F1/Balanced Accuracy=98.36%；证据为 `artifacts/20260802-010756/rd-only-stage011-v9-full-seed42-18epochs/training_summary.json`；验收为 best epoch=13、test=false。
- [x] 验收分类别目标：window Drone P/R/F1=98.58%、Bird P/R/F1=96.56%；parent Drone P/R/F1=99.05%、Bird P/R/F1=97.67%；证据为 best validation；验收为两类均未通过牺牲另一类跌破 90.00%。
- [x] 验收混淆方向：window `[[1251,18],[18,505]]`，Drone→Bird=18、Bird→Drone=18；parent `[[104,1],[1,42]]`；验收为双向错误单独披露。
- [x] 验收 best 选择规则：epoch 8 与 13 的前三项选择元组相同，epoch 13 window Macro-F1=97.5699% 高于 epoch 8 的 97.5038%，由第四级 tie-break 胜出。
- [x] 完整 v8 validation 作为历史对照；证据为用户此前手动完成的 v8 summary；验收为不通过 legacy checkpoint 冒充当前 v9 复载结果。
- [x] 补齐 Macro-Precision、Macro-Recall、Balanced Accuracy、confusion 与双向错误；证据为 `_class_metrics` 和单测。
- [x] 补齐 source 分层；active source 唯一为 `cq08_track`；证据为 index 与 restricted evaluator；验收为 source 仅用于报告不进 forward。
- [x] 明确 session 不可用：index 无 `session_id/session_name`，不得以 parent 代替；证据为 header/evaluator；验收为 `available=false`。
- [x] 修正 stable 语义：累计 prefix 是 causal，但“最后一次翻转后稳定时间”只能完整序列后回溯；证据为 analyzer 字段；验收为线上 stable 明确 `available=false`。
- [x] 增加 evaluator `--max-batches` 与确定性两类均衡 smoke 子集；验收为 incomplete/excluded count 明确且单测通过。
- [ ] 运行独立完整新格式 validation（`--max-batches 0`）并生成 predictions/source/analysis；依赖为 Stage 011 best checkpoint；验收为 1792 个 val window 全覆盖、excluded=0、仍不运行 test。该任务用于补齐逐样本置信度/source/calibration 报告，不替换已经完整覆盖的 trainer validation 指标。

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
证据：历史 Stage010 二分类回归 80 passed；本轮五分类前置回归 29 passed、compileall、最终五分类 smoke/evaluator JSON。
产物：可执行入口、checkpoint、日志/progress 与 readiness 结论。
验收：不启动全量训练/重建/完整长评估/test。

- [x] 静态编译：`python -m compileall -q src/core src/process config` 通过；验收为 exit 0。
- [x] 历史二分类 RD 专项回归：80 passed、1 个 PyTorch nested-tensor warning；验收为 0 failed；该数字不包含本轮五分类链路。
- [x] 最终训练 smoke：1 epoch、2 train batches、2 val batches、batch=4、train/val only；artifact=`artifacts/20260802-003631/rd-only-stage010-v9-smoke-seed42`；验收为 status=completed。
- [x] 两类反传：train support Drone=5、Bird=3，loss finite，gradient finite；验收为 checkpoint 已保存。
- [x] checkpoint strict reload + evaluator：8 个 balanced val windows，Drone=4/Bird=4，excluded=1784，明确 no performance claim；验收为 exit 0。
- [x] analyzer：校准仅诊断、无 threshold、retrospective/online 边界明确；验收为 exit 0。
- [x] latency smoke：1 warmup+2 measured，batch=1，finite logits；验收为 exit 0；不作为生产 stable 证据。
- [x] Stage 011 best checkpoint 在当前代码下 strict reload smoke 通过；验收为 schema v9、epoch 13、8 个平衡 validation windows 可前向、test=false、rejection unsupported；该受限结果只证明复载链路，不作性能结论。
- [x] 待训练条件曾在启动前满足：入口/config/data hashes/seed/device/batch/epochs/lr/checkpoint/test-off/rejection-off/artifact 规则均固定；用户现已完成该全量训练。

## Stage 011 人工全量训练与训练后验收

依赖：Stage 010 readiness、实际 `--help`、用户手动启动权限。
证据：解释器 `.venv\Scripts\python.exe`、入口只支持 `--config`、Stage 011 summary/progress/history/checkpoint。
产物：完成的 18-epoch artifact、best checkpoint、完整 trainer validation 与独立后评估命令。
验收：训练由用户手动执行；代理不启动全量训练或长评估；checkpoint 在当前代码下可严格复载。

- [x] 生成并复核前台 PowerShell 命令；配置为 `config/rd_only/rd_physical_drone_bird_v9_full_seed42.json`，日志路径固定。
- [x] 生成并复核后台 `Start-Process -WindowStyle Hidden` 命令，stdout/stderr 分离并保存 PID。
- [x] 生成日志 tail、PID 状态和训练后 validation/analyzer 命令；validation 强制 `--split val --max-batches 0`。
- [x] 用户手动启动并完成全量训练；artifact=`artifacts/20260802-010756/rd-only-stage011-v9-full-seed42-18epochs`；证据为 progress `completed`、18 条 history 和 best checkpoint；本次审查未代为启动。
- [x] 验收 trainer 完整 validation 的总体及 Drone/Bird 指标；所有要求指标均 >=90.00%，test=false，supports_rejection=false。
- [x] 在当前代码下对 best checkpoint 执行 strict reload 受限 smoke；报告=`tmp/20260802-rd-only-stage011-full-checkpoint-reload-smoke.json`，验收为 exit 0、8 windows、test=false。
- [ ] 用户手动运行独立完整 validation/analyzer；验收为补齐 1792-window predictions、source、confidence/calibration 与 retrospective stability，且不得据此声称 session、online stable 或拒识能力。

## Stage 012 设计冻结与威胁模型（本轮已完成）

依赖：历史 Stage 001–011、`LOW_ALTITUDE_CLASSES`、`src/core/experts.py` 与 `src/core/expert1_trainer.py` 的入口/融合事实、用户批准的方案 A。
证据：已读取的 active 二分类模型定义、五分类标签常量、旧 TR+RD 入口与当前工作区状态；本阶段不运行训练或数据变更。
产物：五分类 RD-only 设计契约、可用性状态机、威胁模型、实施阶段清单。
验收：标签顺序和拼写固定；RD-only 仅使用 RD 及合法派生量；TR 入口保留但 RD-only 训练不读取轨迹；TR 缺失时融合严格退化到 RD-only；双缺失不伪造分类结果。

- [x] 冻结五分类集合和顺序：`DroneTarget`、`BirdTarget`、`BalloonTarget`、`ClutterTarget`、`UnknownTarget`；依赖为 `LOW_ALTITUDE_CLASSES`；证据为 `src/core/experts.py`；产物为五分类 task contract；验收为任何模型、dataset、checkpoint、评估器使用同一顺序。
- [x] 冻结软级联语义：共享 RD encoder 后输出 `P(parent)` 与条件子分布，最终概率由可审计乘积/归一化得到；同时保留直接五分类 flat logits 旁路；证据为设计决策；产物为级联/flat 对照接口；验收为级联和 flat 都能独立前向，禁止使用硬 argmax 作为训练中间标签。
- [x] 冻结拒识语义：`UnknownTarget` 是五分类标签，不等同于低置信度 reject；reject/abstain 必须是独立后处理动作，只有在开放集标签、阈值来源和验证集证据齐全时才能声明能力。
- [x] 冻结融合可用性状态机：RD+TR 使用经校准的 late fusion；仅 RD 使用 RD-only 五分类；仅 TR 使用 TR-only 五分类；RD/TR 均缺失输出 `abstain/unavailable` 并记录原因；证据为用户批准方案 A；验收为四种模式分别可测试且不把缺失模态填零后伪装为有效输入。
- [x] 冻结非目标：本轮不改变 TR-only 代码或跟踪文件，不把二分类 checkpoint 直接扩展成五分类，不用测试集选择权重/阈值/checkpoint，不通过删困难样本或改变分母追求 90%。

## Stage 013 标签、字段 provenance、pairing、split 与泄露审计

依赖：Stage 012；历史 index、manifest、canonical store、TR/RD pairing contract。
证据：源码字段追踪、CSV/JSON schema、字段级单测、train/val/test/group 计数和 fingerprint 审计。
产物：五分类字段 provenance 表、无泄露数据索引、pairing 隔离测试报告。
验收：五类标签只作为 target/loss/metrics；path、filename、ID、source、split、TR label/trajectory/match、pairing status 不进入 RD forward、loss、采样、阈值或预测后处理；parent/content/session 不跨 split；causal window 无未来帧。

- [x] 逐字段列出原始文件→标准化→canonical frame→segment/window→manifest→dataset→model 的来源、类型、单位和允许用途；证据为源码和字段审计；未知字段不进入模型。
- [x] 核实 `UnknownTarget` 标签 provenance 与 `LOW_ALTITUDE_CLASSES` 的 exact spelling；physical frame/window index 五类集合完全一致，未发现 `UnknowTarget`；active 二分类过滤只影响旧支路。
- [x] 重新统计五类 train/val/test support、parent、content group、source、时间跨度和 `<min_frames` 排除数；trainable support 为 train 12396、val 2710、test 2691，排除 4288，证据为 physical window index。
- [x] 复跑 parent/content fingerprint 跨 split、label conflict、重复窗口、未来帧和时间单调性审计；parent/group cross-split=0，future window=0，现有 summary 无冲突。
- [x] 审查 `rd_pairing_contract.py`、`rd_fusion_eligible`、TR path/ID/match 字段；pairing audit 25021 segments 中 fusion_eligible=0，RD physical index 的 `rd_fusion_eligible=False` 为 22085/22085。
- [x] 发现并隔离旧二分类 `canonical_time` quality：它由 `rd_time_sync_status` 派生，属于 pairing/关联语义；五分类契约排除该字段，旧 v9 代码和 checkpoint 保持原样。
- [x] 对 train-only normalization、feature statistics、class weights、sampler weights 和 checkpoint selection 的 test-off 边界完成源码核查；五类 stats 生成前仍需独立 artifact hash。

## Stage 014 五分类 RD normalization、feature stats 与 artifact

依赖：Stage 013 通过、五类索引和 canonical store 契约稳定。
证据：独立 builder/config、train-only 拟合日志、artifact manifest/hash、能量/单位/缺失值测试。
产物：版本化五分类 normalization、feature stats、physical schema、manifest 和唯一输出目录。
验收：统计只由 train split 拟合；range/Doppler 轴、dB/线性功率、相位、padding、缺失值和质量特征语义与当前物理 RD 契约一致；五类 artifact 可 strict load；不复用只含 Drone/Bird 的统计而不证明合法。

- [x] 明确五分类复用现有 physical canonical store；106026 frames、1549 parents、21x384 axes、五类标签完整，兼容性证据已记录；不重建 canonical tensors。
- [x] 确认现有二分类 normalization/feature stats 不能直接复用：artifact labels 明确为 Drone/Bird，且旧 quality 含 pairing 派生 `canonical_time`；新五类 artifact 单独命名和 schema。
- [x] 为五类 train split 生成 normalization stats 和 feature stats，保存 split、label set、parent/group 数、生成器版本、输入轴和 SHA-256；核验 normalization SHA=`76dc9c27df29de9a1d8d2e681966c191ff9a21fbbae1cb15645123f53d0b6609`、feature stats SHA=`df792ed08bf13590df7fc06526dec64b69eff23c576786e7f0616c78270b578d`。
- [x] 验证真实零值、缺失、padding、无效帧、phase 缺失和异常值的编码不混淆；完成 schema/finite/axis/shape 检查。dB↔linear-power 能量守恒实验仍是后续单因素实验，不把未执行实验写成已完成。
- [x] 验证所有声明的统计/质量/micro-Doppler 特征实际进入模型分支；五类质量维度固定为7且不含 `canonical_time`，feature stats 的14项 statistics 接入 statistics branch，raw dB 接入 micro branch。
- [x] 生成五类 artifact inventory；strict load 核验 label order、shape、axis、index/generator/normalization hashes 一致。独立 test manifest 未生成，test evaluation=false。

## Stage 015 软级联 RD-only 模型与 flat 旁路

依赖：Stage 013 字段审计、Stage 014 artifact、历史 Physical RD encoder 接口。
证据：模型源码、forward contract、shape/mask/gradient 单测、独立五分类配置。
产物：RD-only 五分类模型、软级联 head、flat head、模型 schema/checkpoint contract。
验收：模型只接收 RD tensor 及合法 RD 派生量；输出固定五类顺序；padding/future mask 正确；级联概率和 flat 概率都可复现且梯度回传；TR 参数即使入口保留也不会被读取。

- [x] 新增独立五分类常量/contract，不修改 TR-only 常量；顺序严格为 `LOW_ALTITUDE_CLASSES`，拒绝 `UnknowTarget` 拼写。
- [x] 复用并验证 RD encoder：CNN、时间编码、packed GRU/attention、statistics/quality/micro 分支的 shape、mask、轴和 finite contract。
- [x] 实现软级联 parent/child 概率头，partition=`((0,1,2),(3,4))`，用 log-softmax 乘积得到归一化五类概率，禁止 hard routing、TR label 和 future frame。
- [x] 实现独立 flat head；checkpoint/evaluator 同时记录 flat/cascade，并明确 deployment head。
- [x] 定义五分类输出和 strict checkpoint schema；`supports_rejection=False`、`unknown_label_is_rejection=False`、TR entry preserved/allow_tr_inputs=false。
- [x] 专项模型测试：无 TR forward 参数、padding invariance、all-padding rejection、gradient-to-each-head 和 flat/cascade shape；`tests/test_rd_physical_five_contract.py` 已通过。

## Stage 016 训练目标、采样、warm start 与 checkpoint

依赖：Stage 014/015；五类 support 和训练可用性报告。
证据：独立 config、loss 源码、class/parent/content 权重审计、短 smoke 日志、checkpoint round-trip。
产物：五分类训练配置、loss/采样策略、可选二分类 warm-start 方案、best/latest checkpoint 规则。
验收：主分类 loss 只对真实有效帧/窗口计算；级联辅助 loss 不重复计数；类别不平衡处理可解释且不牺牲 Drone/Bird；test 不参与任何选择；checkpoint 可严格保存/恢复。

- [x] 建立独立 smoke/full 配置；seed/device/batch/epochs/lr/clip/日志/output_dir/test-off 已固定，full output 不覆盖二分类 artifact。
- [x] 实现并静态核查 flat CE、cascade CE、parent CE、micro/prefix 辅助 loss 的权重和分母；同一 logits 不重复计算主 CE；29 项五分类/融合/TR 入口/旧 RD 回归已覆盖契约、梯度、causal split 和融合边界。
- [x] 实现 parent/content-group/evidence weighting；已记录五类 train/val support。class_weights 当前均为1.0，长尾稳定性需完整 validation 后再判断，不能宣称已解决。
- [x] 核实二分类 checkpoint warm-start 边界：旧 checkpoint strict load 仍为 `quality_dim=8`、labels=Drone/Bird，五分类 schema=7维且五类；没有执行未经证明的 warm-start，五分类从头训练。
- [x] 固定 best checkpoint 选择优先级：Drone/Bird 非劣化资格、window Macro-F1、最低类别 F1/Recall/Precision、parent Macro-F1 及最低类别 F1/Recall/Precision；规则写入 artifact/config。smoke 仅验证规则执行，不代表满足资格。
- [x] 保存 model/config/artifact manifest/normalization hashes 和训练历史；optimizer/RNG resume 尚未实现，test loader/指标/阈值入口默认关闭并硬拒绝。

## Stage 017 候选选择、校准与 Drone/Bird 非劣化

依赖：Stage 016 至少一个可复现候选 checkpoint；五类 validation 完整可用。
证据：parent-grouped validation/OOF 报告、五类混淆矩阵、置信度/校准报告、二分类历史基线。
产物：候选选择报告、flat-vs-cascade 比较、校准参数（如有）和停止/回退结论。
验收：总体 Accuracy、Macro-P/R/F1、Balanced Accuracy 及有效类别指标按真实分母报告；Drone/Bird P/R/F1 不低于冻结二分类参考门槛或明确差距；不得用 test 调参。

- [x] 分别评估 window、parent、session（若字段存在）、source 和线上 stable（若真实链路存在）；window=2710/2710，parent=221；session 因 physical index 无 session 字段不可用，source 只有 `cq08_track`，线上 stable 未接入；parent Clutter=8、Unknown=15 标记 low stability。
- [x] 输出五类 per-class Precision/Recall/F1/Support、双向主要混淆（含 Drone↔Bird）、混淆矩阵和置信度分布；预测 CSV 显示完整 2710 行，置信度/ECE 诊断不改变分类分母。
- [ ] 在 train 或 parent-grouped OOF validation 上拟合 calibration/threshold；保存拟合数据范围、参数、版本和禁止 test 参与证明。
- [x] 比较 flat 与 soft cascade；同一 checkpoint、同一 validation 分母下 flat window Macro-F1=84.15%、cascade=83.71%，因此当前部署头保持 flat；尚未执行改变一致性选择规则的独立训练。
- [x] 应用停止条件：当前候选总体 Macro-F1=84.15%、Bird F1=86.69%、Clutter F1=66.67%、Unknown F1=72.39%，且 best_selection.eligible=false；停止将其作为交付候选，不以调分母或后处理掩盖差距。

## Stage 018 TR-only 五分类 checkpoint 契约核实

依赖：Stage 012 融合设计；不要求 TR-only 训练已完成，但必须核实其真实入口和输出契约。
证据：`src/core/experts.py`、`src/core/expert1_trainer.py`、TR-only 配置/manifest/checkpoint 元数据、实际 `--help`。
产物：TR-only 五分类接口报告或明确阻塞项；不修改 TR-only 文件。
验收：确认 TR-only label order、logit/probability schema、缺失轨迹行为、checkpoint 可加载方式、是否包含 reject/Unknown 后处理；不存在可验证 checkpoint 时明确“待 TR-only 训练”。

- [x] 读取并核对 TR 入口、pure-TR dataset 构建和训练开关；证据为 `train_expert1.py -> train_expert1_v1`、`stage1_traj_only_epochs=epochs`、`load_rd=False`、reject/test 关闭；未修改 TR-only 文件。
- [x] 核实现有 pure-TR checkpoint labels 严格等于 `LOW_ALTITUDE_CLASSES`；`tests/test_rd_tr_entry_contract.py` 通过，`selection_eligible=true`、`test_used_for_selection=false`。
- [ ] 继续核实 TR 缺失、短轨迹、异常轨迹和无效 mask 的线上输出契约；该项是 TR/融合后置验收，不阻塞 RD-only 全量训练。
- [x] 读取现有 TR-only artifact 元数据；validation Macro-F1=70.58% 仅为 TR-only 事实，不作为 RD 或融合能力证据。

## Stage 019 安全 late fusion

依赖：Stage 017 的 RD-only 五分类输出契约、Stage 018 的 TR-only 五分类契约；两者 label order 和校准状态一致。
证据：fusion/router 源码、可用性状态机测试、概率归一化/缺失模态测试、独立融合配置。
产物：RD+TR late-fusion 模块、router、融合配置和审计日志。
验收：双模态仅融合各自已验证概率/logits；单模态严格透传对应模型；双缺失输出 `abstain/unavailable`；不读取 pairing/ID/label 作为权重或后处理；融合权重只能由 train/OOF validation 确定。

- [x] 定义独立融合输入 schema：五类 logits/probabilities、availability 和仅模态内 quality；拒绝隐式 TR/RD 拼接。
- [x] 实现独立 availability-aware 四态适配器：both、RD-only、TR-only、none；记录 selected_source、fallback_reason 和 abstain reason。尚未接入现有 `LowAltitudeSuspectExpert` 线上 forward。
- [ ] 设计可解释 late fusion（例如校准后 log-prob 加权/门控），权重在 parent-grouped train/OOF 上拟合并版本化；禁止 test 选择。
- [x] 增加极端缺失/NaN/错顺序/错 schema/全零伪输入测试；`tests/test_rd_tr_fusion.py` 已通过。pairing 字段隔离和线上接入仍待后续阶段。
- [ ] 明确融合与拒识边界：UnknownTarget 分类、低置信度 reject、OOD/abstain 分开报告。

## Stage 020 四种 availability 模式评估

依赖：Stage 017/018/019；真实 RD-only checkpoint 和 TR-only checkpoint 均可用，或对缺失者明确阻塞。
证据：固定 parent-grouped validation/test-free 报告、模式计数、五类指标、拒识/abstain 分母。
产物：RD+TR、RD-only fallback、TR-only fallback、both-missing 四态评估表。
验收：分别报告 window/parent/session/source/stable；RD-only fallback 不低于 RD-only 原始输出；TR-only fallback 不被误报为 RD 能力；both-missing 不计入 accepted 分类准确率且 abstain 分母明确。

- [ ] 建立四态样本清单和 availability 注入器，注入只改变可用性标志，不改变 RD/TR 内容或标签。
- [ ] 统计五类总体和分类别指标、Drone↔Bird 混淆、Unknown/Clutter 处理、coverage、accepted support、selective risk。
- [ ] 若存在开放集/拒识标签，报告 FAR/FRR、Unknown/Clutter rejection recall；否则明确无法得出有效结论。
- [ ] 验证稳定决策需要真实时序链路；离线 retrospective 结果不得冒充 online stable latency。

## Stage 021 静态检查、契约测试、泄露测试与短 smoke

依赖：Stage 013–016 与 Stage 021 的 RD-only 前置子任务；Stage 017–020 中完整 validation、融合权重和线上 stable 属于全量训练/TR 完成后的后置验收，不阻塞 RD-only 全量训练命令交付。
证据：compile/pytest、数据契约、无 TR 输入、split 隔离、checkpoint round-trip 和受限 smoke artifact。
产物：测试清单、日志、smoke checkpoint/report、失败修复记录。
验收：配置可加载、入口 `--help` 可执行、dataset/model/loss/gradient/checkpoint/evaluator 全链路通过；smoke 只限 train/val、小 epoch/batch；禁止全量训练、全量重建、长评估和 test。

- [x] 执行静态 compile、入口 `--help` 和 RD-only 回归；五分类、融合、TR-only 只读契约、causal split、旧 RD 数据集/artifact 回归共 `29 passed`，compile/help exit 0。
- [x] 执行 label/order/shape/axis/mask/padding/time/finite/no-TR/provenance/normalization-train-only 测试；split support/hash 以 Stage 013/014 audit 证据核验。
- [x] 执行五分类 smoke：1 epoch、2 train batches、2 val batches、batch=2、CUDA、test=false；最终 artifact=`artifacts/20260803-024607/rd-only-five-stage021-smoke-final-20260803-024559/best_rd_physical_five_model.pt`，评估4 windows且明确 restricted/no performance claim。
- [x] 执行 checkpoint save/reload、flat/cascade forward、fusion four-state adapter 和 evaluator smoke；四态 adapter 的混合/双缺失 batch 不暴露 logits，仍是独立模块，尚未接入线上 `LowAltitudeSuspectExpert`。
- [x] smoke 失败项已修复并重跑；smoke、契约测试和入口均通过，但 restricted support 下 `drone_bird_non_degradation=false` 只表示样本不足，不能作为五分类性能结论；完整 validation 留待全量训练后。

## Stage 022 待全量训练准备与人工命令交付

依赖：Stage 013–016 的 RD-only 训练前置验收通过；Stage 017–020 的完整五类 validation、融合权重、真实四态评估和线上 stable 属于全量训练/TR checkpoint 完成后的后置阶段。
证据：实际 `--help`、独立 full config、artifact/hash inventory、smoke 结果、唯一 output 目录和资源估计。
产物：待训练 readiness 记录及由用户手动执行的前台/后台训练、日志、进程、评估命令。
验收：训练入口/解释器/配置/路径/seed/device/batch/epochs/lr/checkpoint/test-off 均已实际核实；拒识开关和阈值状态明确；全量训练未由本任务启动。

- [x] 固定并验证独立五分类 full config、normalization/feature stats/manifest 路径和 SHA-256；full config 可加载，输出目录为 `artifacts/rd-only-five-stage022-full-seed42`，不覆盖二分类 artifact；checkpoint 在全量训练前不存在是预期状态。
- [x] 用实际 `--help` 拼接前台 PowerShell 命令和后台挂机命令；命令只引用已存在入口/config，未由本任务启动。
- [x] 输出日志查看、进程状态、artifact 预期位置和训练完成后 validation 命令；参数均来自实际入口，test 评估被入口和配置关闭。
- [x] 准备唯一日志目录 `artifacts/rd-only-five-stage022-full-seed42-launch`；前台/后台命令仅经 PowerShell scriptblock 语法解析，未启动训练进程。
- [x] 再次核对未修改 TR-only 跟踪文件；记录未执行的全量训练、test、长评估、全量重建、线上融合接入和剩余风险。
- [x] 训练前置验收已通过，标记“待全量训练：是”；完整五类性能、融合权重、拒识和线上 stable 仍是训练/TR 完成后的后置验收，不作为训练命令的虚假依据。

## Stage 023 五分类全量训练后 validation 与根因审查

依赖：Stage 022 用户手动训练产物、strict checkpoint loader、独立 val evaluator、预测 CSV、训练历史和 RD-only index；禁止读取 test。
证据：`artifacts/20260803-112945/rd-only-five-stage022-full-seed42/{training_summary.json,training_history.json,training_progress.json,best_rd_physical_five_model.pt}`、`tmp/20260803-rd-five-stage022-full-validation.json`、对应 predictions CSV、代码/配置行号和只读脚本输出。
产物：五类总体/分类别/parent/source/session/stable/拒识报告、Bird 下降与 Clutter/Unknown 混淆分析、已证实问题与待核实假设、下一轮单因素实验计划。
验收：validation 2710/2710、排除 0、test=false、tr_inputs_loaded=false；不得输出虚构指标或全量训练命令。

- [x] 核对 checkpoint epoch=16、SHA-256=`168068014FEC472D31477FC1167C9A927FC5C88EACEBB2AB14F0B85A8AB3AC43`、labels 顺序和 `deployment_head=flat`。
- [x] 核对 full validation 覆盖 2710/2710、excluded=0、source=`cq08_track`、session/stable unavailable、test=false、supports_rejection=false。
- [x] 输出 flat window 总体与分类别指标；确认 Accuracy=93.17%、Macro-F1=84.15%，Bird Recall=82.22%，Clutter Recall=58.77%，Unknown Recall=74.21%，未达到 90.00%。
- [x] 输出 parent 指标和低 support 标记；parent=221，Clutter support=8、Recall=50.00%，Unknown support=15、Recall=80.00%，不把 parent 数量当 session 数量。
- [x] 单独统计混淆方向：Bird→Drone=24、Bird→Balloon=34、Bird→Unknown=25、Bird→Clutter=10；Clutter→Unknown=24、Unknown→Clutter=10；错误置信度仍高，不能只用阈值偏移解释。
- [x] 计算不改变评估分母的 pairwise 诊断：在 Drone/Bird 两类 logits 内比较时 Bird recall=95.41%、Drone recall=99.76%；明确该诊断不能替代五分类部署 Recall=82.22%。
- [x] 核对训练目标与有效权重：class_weights 全为1；旧二分类 Bird weight=1.35 未在五类配置保留；五类 flat/cascade/parent/prefix/micro loss 权重为1/0.25/0.15/0.10/0.10；当前 evidence-only train mean weight 为 Drone=0.8783、Bird=0.8996、Balloon=0.8987、Clutter=0.9095、Unknown=0.8631。历史 parent/content 权重数字不再代表当前机制。
- [x] 核对训练稳定性：18 epoch loss/metrics finite；pre-clip gradient norm 最大约2273.23，val loss 在 epoch 2 后明显波动；列为优化稳定性风险，不宣称其单独造成类别下降。
- [x] 核对标签 provenance：CQ-08 manifest 由 label_id=5 映射 ClutterTarget、label_id=6 映射 UnknownTarget；UnknownTarget 是普通分类标签，不是 reject/abstain 结果。
- [x] 阻断当前候选交付：best_selection.eligible=false，Drone/Bird 参考非劣化未通过；不启动新的长训练、不读取 test。
- [ ] 后续 one-factor A：仅恢复 Bird class weight=1.35，其他配置、数据、seed、epoch 固定；parent-grouped validation 同时检查五类和 Drone/Bird，任一主要指标低于90.00%或 Drone/Bird 下降即停止。
- [ ] 后续 one-factor B：仅消融 `parent_loss_weight` 或仅降低其值，验证共享 parent=[Drone,Bird,Balloon]/[Clutter,Unknown] 对类内边界的影响；不得与 A 同时修改。
- [ ] 后续 one-factor C：仅调整 parent/evidence weighting 的上限或归一化，验证 Clutter/Unknown 高有效权重是否造成共享表示挤压；不得删除困难样本。
- [x] 后续 label audit：已完成原始 CQ-08 label_id=5/6、MAT 文本、parent 内部一致性和 train/val split 审计；报告为 `tmp/20260805-rd-five-label-semantics-audit.json`。未发现重标依据，因此未创建新 manifest/index/stats，也未覆盖当前训练产物。
- [ ] 后续 calibration/reject：仅在 train 或 parent-grouped OOF 上拟合；没有独立 reject/open-set 标签和动作前，持续报告 supports_rejection=false，Unknown 不转译为拒识。

## Stage 024 独立层级方案冻结与接口契约

依赖：Stage 023 完整 validation、`LOW_ALTITUDE_CLASSES`、当前五分类 checkpoint、现有 RD-only forward 契约。
证据：`src/core/rd_physical_five_model.py`、`src/core/rd_physical_five_trainer.py`、`src/core/rd_physical_five_config.py`、`training_summary.json`、2710/2710 validation 报告。
产物：独立 ParentGate/DBB/CU 三模型设计说明、标签映射、概率组合公式、checkpoint schema 草案、失败回退规则。
验收：输入只含 RD 及合法 RD 派生量；五类顺序不变；不使用 hard parent argmax 作为唯一线上路径；独立模型可分别评估。

- [x] 固定 ParentGate 标签：`DBBGroup={DroneTarget,BirdTarget,BalloonTarget}`、`CUGroup={ClutterTarget,UnknownTarget}`；依赖为 LOW_ALTITUDE_CLASSES；证据为 `src/core/data/rd_physical_specialist_dataset.py` 与 5 项契约测试；产物为唯一映射表；验收为五类到父组覆盖恰好一次。
- [x] 固定三模型输入输出契约：输入保持 `[B,T,6,21,384]`、7 维 quality、时间/mask、21 range 轴、384 velocity 轴和 raw dB；输出分别为 2、3、2 类 logits/probabilities；证据为 specialist model/dataset smoke；验收为 shape、有限值、轴单调和无 TR 字段。
- [x] 固定独立 checkpoint 和 artifact 命名规则；每个模型保存独立 config、manifest/hash、训练历史、best checkpoint 和 validation 报告；证据为三份 smoke artifact；验收为未覆盖 Stage 023 flat checkpoint 或 TR-only artifact。
- [x] 固定安全组合：`P5=P_parent×P_specialist`，使用 soft parent 概率；父组不确定或三模型不可用时保留 flat fallback；证据为 composer 契约测试；验收为五类概率归一、无负值、无隐式零模态。
- [x] 固定比较口径：完整五分类能力以 flat、oracle-specialist、predicted-soft、hard-routing（仅诊断）和 fallback blend 分别报告；证据为 composer API；全量比较待 Stage 029，专科局部指标不能替代完整五类指标。

## Stage 025 专科数据视图、split 与泄漏审计

依赖：Stage 024 契约；只读现有 physical canonical store、window index、train/val/test split 和 train-only stats。
证据：`expert1_rd_only_causal_window_index_v1_physical_v3_1.csv`、五类 stats、parent/content-group audit、dataset provenance 检查。
产物：ParentGate/DBB/CU 三个数据视图及支持数、父组/类别一致性和泄漏报告；不默认重建数据。
验收：parent、content group、时间连续片段不跨 split；test 不加载、不用于 stats、checkpoint、阈值或融合权重。

- [x] 为每个窗口生成内部父组 target 和 specialist target；原始 label、路径、文件名、ID、source、split、pairing 字段只留在审计表；证据为 specialist dataset remap 和无 TR forward 测试；验收为 forward batch 不含这些字段。
- [x] 复核三视图的 train/val/test parent 与 content-group 交集为 0；同时检查每个 parent/group 的标签唯一性、causal future=0、padding/缺失契约；证据为 5 项契约测试及既有 physical audit；验收为冲突和交叉均为 0。
- [x] 复核三模型均复用 train-only normalization/feature stats 但不混用二分类旧 stats；证据为 specialist artifact manifest/hash 和 smoke；验收为 artifact hash、generator hash、label order 和 test=false 一致。
- [x] 计算 DBB/CU、Drone/Bird/Balloon、Clutter/Unknown 的 split 支持和低样本标记；证据为 dataset audit：train/val 分别 DBB=11372/2437、CU=1024/273，原始五类 support 保留；验收为不删样本改善分布。
- [x] 对 ParentGate 的父组标签来源做 provenance 审计；证据为只调用五类 `label_name` remap、manifest 明确 no TR/pairing/path/ID forward fields；验收为父组不读取 TR 标签、轨迹或 RD/TR 配对结果。

## Stage 026 独立 ParentGate 模型与训练链

依赖：Stage 024–025；当前物理 RD encoder 可复用代码结构但参数、config、optimizer、checkpoint 必须独立。
证据：新模型/配置/训练/评估源码、contract tests、短 smoke 日志。
产物：`PhysicalRDParentGateClassifier` 及实际入口、独立配置、2 类 validation 报告。
验收：train/val 可运行；loss/gradient finite；checkpoint 可保存/strict reload；不读取 test/TR；直接报告 DBB/CU Precision、Recall、F1、混淆矩阵及 parent-level 指标。

- [x] 实现独立 ParentGate forward 和 artifact schema；输入契约与五分类一致，输出顺序固定 `DBBGroup,CUGroup`；证据为独立 model instance、schema v1 和 strict reload；验收为模型参数不从 Stage 023 checkpoint 隐式加载。
- [x] 实现 ParentGate dataset/remap、parent 权重和 test-off trainer；证据为 trainer/config/smoke，padding/缺失不反传；验收为父组标签仅来自合法五类 label 映射。
- [x] 实现 ParentGate evaluator，分别输出 window/parent/source 指标、DBB→CU 和 CU→DB；证据为 evaluator JSON schema；验收为 session/stable/reject 明确 unavailable。
- [x] 运行 ParentGate smoke（1 epoch、2 train/2 val batches、唯一输出目录）；证据为 `artifacts/20260803-230647/rd-specialist-parent-gate-smoke-seed42`；验收为 forward、有限 loss/gradient、保存/恢复和无 TR 输入通过。

## Stage 027 DBB 与 CU 专科模型和训练链

依赖：Stage 025 数据视图；建议先完成 ParentGate smoke，但 DBB/CU 可独立开发。
证据：新模型/配置/训练/评估源码、类别映射测试、两套 smoke 日志和 checkpoint round-trip。
产物：独立 `PhysicalRDDBBSpecialist`（3 类）和 `PhysicalRDCUSpecialist`（2 类）入口、配置、报告。
验收：DBB 只评估 Drone/Bird/Balloon，CU 只评估 Clutter/Unknown；不把局部指标宣称为五类能力。

- [x] 实现 DBB 专科 label remap、3 类 head、独立 artifact/config/trainer/evaluator；证据为 `PhysicalRDSpecialistClassifier(kind='dbb')` 和 manifest；验收为标签顺序固定且不含 CU 类。
- [x] 实现 CU 专科 label remap、2 类 head、独立 artifact/config/trainer/evaluator；证据为 `PhysicalRDSpecialistClassifier(kind='cu')` 和 manifest；验收为标签顺序固定且不含 DBB 类。
- [x] 为两专科分别固定 train-only class/parent 权重和 checkpoint 选择规则；证据为独立 config/checkpoint，test=false；验收为不使用完整五分类 test 或预测后处理调参。
- [x] 分别运行 DBB/CU smoke；证据为 `artifacts/20260803-230713`、`artifacts/20260803-230735`；验收为有限 loss/gradient、strict reload、RD-only 边界测试通过。
- [ ] 分别运行完整 validation（仍只允许 val），报告 window/parent/source、support、混淆矩阵和低 support 稳定性；验收为三类/二类局部报告与五类报告严格分离。

## Stage 028 软概率组合、flat 保底与缺失模态安全

依赖：Stage 023 flat baseline、Stage 026 ParentGate、Stage 027 两个专科 checkpoint。
证据：组合器源码、概率归一化测试、父组冲突测试、flat fallback 测试、四态可用性测试。
产物：独立 RD-only hierarchical composer 和可审计组合配置；不接入 TR 线上路由，直到 TR checkpoint 契约完成。
验收：RD 全部可用时输出五类概率；缺失专科/父模型不伪造 logits；组合不读取 TR、路径、ID 或 split。

- [x] 实现 soft composition：`P(Drone/Bird/Balloon)=P(DBB)P(class|DBB)`、`P(Clutter/Unknown)=P(CU)P(class|CU)`；证据为 `rd_physical_hierarchical_composer.py`；验收为每行概率和为 1±1e-6。
- [x] 实现 oracle-parent、predicted-soft 和 hard-routing 三种分析模式；证据为 composer 与测试；hard-routing 仅诊断，不作为默认部署。
- [x] 实现 flat fallback/blend 接口，保留 Stage 023 flat 作为安全基线；证据为缺失 specialist 回退测试；未校准时不自动改变部署头。
- [x] 完成 specialist/flat 缺失和 mixed-state 核心测试；证据为 5 项新契约测试及既有四态路由测试；不可用状态不使用全零哨兵冒充有效预测。

## Stage 029 三模型组合 validation 与错误归因

依赖：Stage 025–028；禁止读取 test，禁止在独立 test 上选权重、阈值或 checkpoint。
证据：同一 val split 的三模型 predictions、组合报告、parent-grouped OOF（如需校准/权重）。
产物：flat、oracle、predicted-soft、hard、fallback blend 的完整五分类比较报告。
验收：报告 window/parent/source；总体 Accuracy、Macro-P/R/F1、Balanced Accuracy；五类逐类 P/R/F1；Drone/Bird 双向错误；Clutter↔Unknown；不隐藏低 support。

- [ ] 评估 oracle specialist 上限，明确其不代表可部署性能；验收为只作为误差归因参考。依赖三套全量 checkpoint，当前未执行。
- [ ] 评估 ParentGate predicted-soft 组合，分别统计跨父组错误和父组内错误；验收为父组错误不会被遗漏在专科指标之外。
- [ ] 评估 flat 与 soft 组合的一致性/冲突样本；验收为冲突按置信度、父组概率和真实标签分层，不用测试集筛样本。
- [ ] 仅使用 train 或 parent-grouped OOF 选择 calibration/blend 参数；验收为保存独立参数文件、OOF 分组证明和选择日志，test=false。
- [ ] 执行非劣化门槛：完整五类 Macro-F1、主要类别 P/R/F1 不能因组合下降；Drone/Bird P/R/F1 不低于 Stage 011 参考；验收失败则保留 flat，不强行部署。

## Stage 030 线上接口、TR 保留与四态安全审查

依赖：Stage 028–029；TR-only checkpoint 契约必须单独核实，不能用 TR 结果证明 RD 能力。
证据：现有 `experts.py`/TR 入口调用链、`rd_tr_fusion.py`、router contract tests、可用性状态机。
产物：RD-only 独立调用适配器和未来 late-fusion 接口设计；不提前宣称线上融合完成。
验收：TR 缺失时严格返回 RD-only；RD 缺失时严格返回 TR-only；双缺失 abstain/unavailable；任何模态缺失不填零伪造。

- [ ] 确认三模型 RD-only 可从保留的 TR 入口调用，但 RD-only trainer/evaluator 不读取轨迹；验收为无 TR 输入 contract test 通过。
- [ ] 核对现有线上 stable decision 是否真实接入；若无，报告 unavailable，不把 retrospective prefix 稳定性当线上能力。
- [ ] 设计 both/RD-only/TR-only/none 四态输出字段、来源、原因和 accepted 分母；验收为混合 batch 不泄漏另一模态信息。
- [x] 在 TR-only checkpoint 和标签契约未独立验收前，禁止学习 late-fusion 权重或修改 TR-only 代码/跟踪文件；本轮未修改 TR-only 代码/跟踪文件。

## Stage 031 Smoke、全量训练准备与人工命令交付

依赖：Stage 024–028 的实现、契约测试和 smoke 通过；Stage 029 完整组合 validation 与 Stage 030 线上接入不是开始 specialist 全量训练的前置条件；本阶段不由代理启动全量训练。
证据：实际 `--help`、独立 configs、smoke artifacts、测试结果、路径/hash 清单。
产物：ParentGate/DBB/CU 三套人工可执行全量训练命令、日志/artifact 路径、训练后评估命令。
验收：三套配置均 test-off、seed/device/batch/epochs/lr/checkpoint 策略固定；smoke 通过后才输出命令；任何失败均标记待训练否。

- [x] 为三模型各运行短 smoke：1 epoch、2 train/2 val batches、输出唯一且不覆盖历史 artifact；证据为三份 training_summary/checkpoint；验收为配置加载、数据读取、forward、loss、gradient、save/reload 和 evaluator 全部通过。
- [x] 运行静态检查、数据契约、shape/mask、split 隔离、无 TR 输入、train-only stats 和组合概率测试；证据为 compileall、`pytest tests/test_rd_specialist_contract.py`（5 passed）及相关回归（23 passed）；验收为零失败。
- [x] 固定三套 full config、Python 解释器、入口、device、seed、日志和 artifact 目录；证据为三个 full JSON、实际 `--help`；验收为字段一致，test=false。
- [x] 用户已确认执行；输出三条前台训练命令和可选后台命令；全量训练仍未由本审查启动，必须由用户手动执行。

### Stage 031 当前 readiness 边界

- [x] 三套 RD-only specialist 代码、配置、数据视图和 smoke 已达到“可开始全量训练”；不等于三套模型已有全量性能，也不等于层级组合已部署。
- [x] 全量训练完成后执行 Stage 029 完整 validation、parent/session/source 分层报告、非劣化筛选和组合诊断；证据为 Stage 036/037 报告。OOF 校准和 production stable/fusion 仍另列为未完成，不以 validation 结果宣称五分类提升。

## Stage 032 全链路审查冻结（历史审查记录，修复已在 Stage 033–034 完成）

依赖：Stage 001–031 历史记录、当前工作区、有效配置、实际入口、数据 artifact、checkpoint 元数据。
证据：源码逐文件核查、physical index/manifest/stats 只读审计、Stage 023 full validation、checkpoint SHA、23 项 RD/fusion 回归、compileall、四个入口 `--help`、strict reload smoke。
产物：已确认问题清单、待核实假设、修复前阻塞条件；本阶段不修改实现。
验收：旧二分类、五分类 flat、ParentGate/DBB/CU specialist、legacy 分支和 TR/fusion 路径分开；不读取 test；不修改 TR-only 文件。

- [x] 冻结五分类 full checkpoint：`artifacts/20260803-112945/rd-only-five-stage022-full-seed42/best_rd_physical_five_model.pt`，SHA-256=`168068014FEC472D31477FC1167C9A927FC5C88EACEBB2AB14F0B85A8AB3AC43`，epoch=16，validation=2710 windows，`test_evaluation_executed=false`、`tr_inputs_loaded=false`。
- [x] 冻结物理 RD 数据：window index 22085 行/17797 trainable；parent、content group、tensor path 跨 split=0；future frame=0；fusion-eligible=0；source=`cq08_track`；无 session 字段；train/val/test support 分别为 12396/2710/2691。
- [x] 核对五分类标签与模型契约：`LOW_ALTITUDE_CLASSES/RD_FIVE_LABELS=(DroneTarget,BirdTarget,BalloonTarget,ClutterTarget,UnknownTarget)`；flat/cascade/parent 输出为 5/5/2；`supports_rejection=false`；UnknownTarget 只是普通类别。
- [x] 核对旧二分类契约：`RD_ONLY_LABELS`、`PhysicalRDDroneBirdClassifier.label_names` 均为 `(DroneTarget,BirdTarget)`，`supports_rejection=false`；旧模型与五分类能力不混算。
- [x] 核对 RD-only 边界：五分类/ specialist forward 只接收 RD tensor、RD 轴、RD 时间、mask、RD quality/statistics/micro 派生量；TR 输入、TR label/trajectory/match、path、ID、split 不进入 forward；旧二分类仍存在 `canonical_time` pairing-derived quality 输入。
- [x] 核对实际调用关系：`route_rd_tr_five` 与 `compose_rd_specialists` 当前仅被测试/定义引用，未接入 `System/LowAltitudeSuspectExpert` 生产 forward；线上 stable decision 不可声称已接入。
- [x] 核对 specialist：Stage 032 时三份 full config 尚无 full checkpoint（历史快照）；该状态已由 Stage 035 用户完成的 ParentGate/DBB/CU full checkpoint 覆盖，当前 checkpoint 已按逐局部类别非劣化选择并在 Stage 036/037 validation 核验。
- [x] 运行只读回归：`.\\.venv\\Scripts\\python.exe -m pytest -q tests/test_rd_specialist_contract.py tests/test_rd_physical_five_contract.py tests/test_rd_tr_fusion.py tests/test_rd_tr_entry_contract.py tests/test_rd_physical_artifacts.py` = 23 passed；`.\\.venv\\Scripts\\python.exe -m compileall -q src/core src/process` 通过；五分类/specialist train/eval `--help` 通过；full/smoke strict reload 通过。
- [x] 记录当前阻塞：physical RD 尚未接入生产线上 stable/fusion caller；该项保留为部署前阻塞，不在本轮擅自接入。

## Stage 033 手术式修复（已完成）

依赖：Stage 032 问题清单；每个子任务只能处理一个主要根因，保留历史用户改动，不修改 TR-only 文件。
证据要求：新增/更新契约测试、独立配置和 artifact hash；不得使用 test 调参或启动 full training。
产物：修复后的代码、配置、回归报告；验收为所有 smoke/strict reload/无 TR 输入/边界测试通过。

- [x] 五分类和 specialist 的 loss 权重移除 `parent_sample_id`/`parent_split_group_id` 查表；仅保留窗口自身 RD evidence/presence 权重；新增 ID 改写不变性测试。证据：`_evidence_weights`、15 项定向测试通过。
- [x] 旧二分类新增 clean/legacy quality schema 隔离接口；历史 v4/canonical_time artifact 仍显式 `legacy_v4`，clean schema 使用 7 项 RD-only quality，禁止静默复用旧 checkpoint。证据：配置、dataset、artifact loader、schema、clean artifact provenance 与回归测试已更新；clean artifact 已由用户生成并核验。
- [x] 旧二分类和五分类 forward 增加 range/velocity shape、finite、严格递增及 suffix-only padding 检查；补充回归。证据：53 项 RD/fusion/entry/artifact/window/contract 测试通过。
- [x] 明确 padding 只允许 suffix mask；collate 仍只生成 suffix padding；interior padding 现在显式拒绝。
- [x] 精简 specialist 未使用 head：ParentGate 使用原生2类 parent head，DBB/CU 使用原生3/2类局部 flat/frame/micro head；checkpoint schema 升为 `physical_rd_specialist_checkpoint_v2` 并记录 head_mode/local_class_count。证据：三套新 smoke checkpoint strict reload 和 `tmp/stage034-*-reload.json`；验收：无 global five-class head 切片、输出维度分别2/3/2。
- [x] 将 specialist checkpoint 选择改为逐局部类别非劣化约束；DBB 同时约束 Drone/Bird/Balloon，CU 同时约束 Clutter/Unknown，ParentGate 同时约束 DBBGroup/CUGroup。参考值只来自既有 validation 报告，smoke 结果明确 `eligible=false`。证据：`_selection_record`、独立配置、selection artifact；验收：局部任一类别下降即不 eligible。
- [x] 旧二分类 clean 训练 loss/评估 aggregation 改为窗口自身 evidence/presence 权重；parent/sample/content ID 只用于报告；clean checkpoint 选择也不要求 parent metrics。证据：`compute_window_evidence_weights`、`_is_better(..., use_parent_metrics=False)` 与 ID 改写不变性测试；验收：clean 分支训练/评估/选择不调用或依赖 `compute_parent_evidence_weights`。
- [ ] 设计 physical RD 五分类接入保留 TR 入口的安全适配层：RD-only、TR-only、both、none 四态；在真实 production caller 接通前只做 contract test，不接线上 stable。依赖：TR checkpoint 独立契约确认。

## Stage 034 修复后 smoke 与回归（已完成）

- [x] 每个修复独立 smoke：clean 二分类、ParentGate、DBB、CU、五分类均完成 train/val 读取、forward、有限 loss、checkpoint 保存；所有 summary `test_split_loaded=false`、`tr_inputs_loaded=false`、`test_evaluation_executed=false`。
- [x] 运行 RD-only 全套静态/契约/shape/mask/split/stats/strict reload 测试：全 `tests/test_rd*.py` = 93 passed、compileall 通过；specialist strict reload 三套新 v2 checkpoint 通过。
- [x] 重新核对配置/artifact 路径和旧 checkpoint 边界：clean artifact SHA-256=`27A555B96F91FE3EAA837EDF36597A438F0D24FA0571A6A50DA62DD7B9E6F1CA`；旧 Stage011 checkpoint 因 inference contract 变化被明确拒绝，未静默放宽。

## Stage 035 修复后全量训练与交付（用户已完成）

- [x] 待全量训练条件已通过：入口 `--help`、配置 dataclass 实载、train-only artifact 路径、seed/device/batch/epochs/lr、test-off、独立输出目录、smoke/strict reload、无 TR 输入和全 RD 回归均通过。
- [x] 生成实际可复制的五分类、ParentGate、DBB、CU 训练入口与前台/后台日志命令；本代理未启动。
- [x] 用户已完成五分类 flat、ParentGate、DBB、CU 四个 full training；训练摘要均 `status=completed`、`test_evaluation_executed=false`、`tr_inputs_loaded=false`，best epoch 分别为 17/2/4/9。clean 二分类 full 未发现本轮新 checkpoint，未将其误计入四个训练。
- [x] 核对四个 full checkpoint、摘要、配置、标签/输入边界、artifact schema、SHA-256 与 train/val support；证据为四份 `training_summary.json`、strict loader 和 SHA 输出。
- [x] 不启动新的 full training、不读取 test；Stage 035 的训练动作由用户完成，代理只做后验 validation。

## Stage 036 四模型 full validation 与同窗组合诊断（已完成）

依赖：Stage 035 四个 v2/full checkpoint、validation-only evaluator、严格相同 physical index 和 train-only stats；禁止 test、OOF 调参、阈值选择和线上接入。
证据：`tmp/20260805-rd-five-stage034-full-val.json`、`tmp/20260805-rd-parent-stage034-full-val.json`、`tmp/20260805-rd-dbb-stage034-full-val.json`、`tmp/20260805-rd-cu-stage034-full-val.json`、`tmp/20260805-rd-hierarchy-stage034-full-val.json` 及预测 CSV。
产物：flat/cascade、ParentGate、DBB、CU 局部指标；同一 2710 validation windows 上的 flat、predicted-soft、hard-routing、oracle-parent 和 flat fallback window/parent/source/calibration 诊断；session/stable/reject 边界。
验收：flat/specialist 评估覆盖完整 val；组合不把局部视图指标当完整五类；oracle 明确 diagnostic-only；test=false、TR=false；低 support 明确标记。

- [x] 完成 flat 五分类 full validation：2710/2710 windows、excluded=0、window Accuracy=93.62%、Macro-P/R/F1=87.07%/81.86%/83.55%、parent 221、Macro-F1=85.31%；五类均逐类输出 support/P/R/F1。
- [x] 完成 ParentGate 局部 validation：2710 windows，DBB/CU window Macro-F1=88.28%，CU Recall=72.53%；parent 221、Macro-F1=91.34%，CU support=23 标记 low；该结果只代表父组模型。
- [x] 完成 DBB 局部 validation：2437 windows，Macro-F1=96.63%，Bird Recall=89.10%；parent 198、Macro-F1=98.72%；该结果只代表 DBB 局部视图，不是五类部署指标。
- [x] 完成 CU 局部 validation：273 windows，Macro-F1=79.53%，Clutter Recall=69.30%、Unknown Recall=88.68%；parent 23、Macro-F1=86.00%，两类 parent support=8/15 均 low。
- [x] 新增只读同窗组合 evaluator `src/core/evaluate_rd_physical_hierarchy.py`；在同一 2710 个 val windows 上重新前向四模型，执行 flat/cascade/flat_fallback/predicted_soft/hard_routing/oracle_parent；py_compile、compileall 和 94 项 RD 测试通过。
- [x] 完成可部署候选比较：flat window Macro-F1=83.55%；predicted-soft=83.80%（Accuracy=92.88%，Bird R/F1=84.32%/86.90%，Clutter R/F1=60.53%/65.40%，Unknown R/F1=66.67%/72.85%）；hard-routing=83.72%，不优于 soft；flat fallback 与 flat 相同。
- [x] 完成 oracle-parent 上限诊断：window Macro-F1=89.79%、parent Macro-F1=91.56%；其 window Bird F1=93.67%、Clutter F1=74.88%、Unknown F1=84.18%，但使用真值父组，仅用于归因，不能作为部署证据。
- [x] 完成混淆方向：flat Bird→Drone/Balloon/Unknown=18/29/25，Clutter→Bird/Unknown=21/32，Unknown→Bird/Clutter=26/10；predicted-soft Bird→Drone/Balloon/Unknown=27/30/11，Clutter→Bird/Unknown=18/10，Unknown→Bird/Clutter=27/13；oracle 去除了跨父组错误但 Clutter↔Unknown 仍为35/18，证明 CU 组内边界是主要瓶颈。
- [x] 完成 parent/source/session/stable 口径：flat/predicted-soft parent=221；source 唯一 `cq08_track`；session unavailable（index 无 session 字段）；online stable unavailable（无 production stable state machine）。
- [x] 完成 calibration 诊断：未校准、未调阈值；flat ECE=6.04%，predicted-soft ECE=4.61%，oracle ECE=3.27%，仅为 val 诊断，不能转成拒识能力。
- [x] 明确拒识结论：四个 checkpoint/组合均 `supports_rejection=false`、`abstain_output=false`、`threshold=null`；UnknownTarget 是普通类别，无法报告 Coverage/FAR/FRR/Selective Risk。
- [x] 阶段验收通过但交付门槛失败：所有结果 test=false、tr_inputs_loaded=false；没有任何可部署模式同时达到总体及五类 90.00%，predicted-soft 也低于 Stage 011 二分类 Drone/Bird 参考，当前保留 flat 作为 RD-only 安全基线，不接线上。
- [x] 核对泛化风险：flat best epoch17 train/val Macro-F1=99.31%/83.55%，CU best epoch9=90.97%/79.53%；仅登记为可复现风险证据，不断言唯一因果。
- [x] 发现并修复 parent 聚合报告口径差异：新增 `src/core/rd_physical_parent_aggregation.py`，RD-only flat/five、specialist trainer/evaluator 统一为每窗口概率均值后 argmax；hierarchy evaluator 原已采用该口径。旧 mean-logit 报告仅作历史快照，不与新 parent 指标横向比较。

## Stage 037 标签语义审计与后续单因素实验（标签审计已完成）

依赖：Stage 036 误差归因；每次只改一个主要因素，必须新配置、新 checkpoint、新 validation 报告；禁止读取 test、删样本、改分母或把 oracle 指标当部署能力。

- [x] 复核 Clutter/Unknown 原始 label semantics、parent 内部一致性和代表性 RD 统计；依赖为 CQ-08 manifest、grouped split、physical frame/window index；证据为 `tmp/20260805-rd-five-label-semantics-audit.json` 和 `src/process/audit_rd_five_label_semantics.py`：manifest 1550 records，Clutter=50/label_id=5/MAT text=`Clutter`，Unknown=100/label_id=6/MAT text=`Other`；train/val frame=89997、parents=1316，multi-label/source/track parent=0，frame-manifest label mismatch=0，train/val group/content-group intersection=0；trainable val support Clutter=114、Unknown=159。结论：当前没有可复现的标签映射或 parent 一致性缺陷；`Other` 只是原始文件文本，不能直接当作 OOD/reject 证据；未读取 test tensor/prediction/metric。验收：审计 JSON 生成、脚本可执行、结果不覆盖既有数据 artifact。
- [x] 统一 RD-only parent aggregation 契约：新增 `src/core/rd_physical_parent_aggregation.py`，flat/five、specialist trainer/evaluator 的 parent 指标均改为“每窗口 softmax 概率均值后 argmax”；不改变 window 指标，不读取 test，不将 parent ID 转为输入/权重。证据：`tmp/20260805-rd-five-stage037-parent-prob-val.json`、`tmp/20260805-rd-parent-stage037-parent-prob-val.json`、`tmp/20260805-rd-dbb-stage037-parent-prob-val.json`、`tmp/20260805-rd-cu-stage037-parent-prob-val.json` 均 full val、test=false，并带 `aggregation` 字段；`tests/test_rd_parent_aggregation.py` 与全 `tests/test_rd*.py` 回归通过。验收：新训练/选择路径统一；Stage 034 已完成的旧 full checkpoint selection 元数据仍是历史 mean-logit 口径，不被追溯改写。
- [x] 对统一口径后的 full validation 做回归核对：flat window 2710、Accuracy=93.62%、Macro-F1=83.55%，与 Stage 036 完全一致；parent 221、Accuracy=94.57%、Macro-F1=85.31%（概率均值口径）；ParentGate window Macro-F1=88.28%、parent Macro-F1=91.34%；DBB window Macro-F1=96.63%、parent Macro-F1=98.72%；CU window Macro-F1=79.53%、parent Macro-F1=80.83%。这些是 validation，不是 test，也不表示五类部署达标。
- [x] 完成不改分母的窗口长度/parent 集中诊断：flat prefix 485 windows Macro-F1=79.12%，non-prefix 2225=84.21%；长度≥24 的 702 windows Macro-F1=68.21%，<24 的 2008=84.85%；只作为后续单因素候选，不作为性能提升或筛样依据。CU 错误集中于少数 parent，且其 manifest label/MAT 文本一致。
- [x] 仅做 smoke 级单因素可执行性验证：A 仅 Bird class weight=1.35，B 仅 parent_loss_weight=0；配置分别为 `config/rd_only/rd_physical_five_bird_weight_stage037_smoke_seed42.json`、`config/rd_only/rd_physical_five_parent_loss0_stage037_smoke_seed42.json`，artifact 位于 `artifacts/20260805-191120/...`；两者均 1 epoch、2 train/2 val batches、loss/gradient finite、checkpoint strict reload/evaluator 通过、test=false、TR=false。受限 val 仅4 windows、类别分布不足，明确不作性能结论；未启动对应 full training。
- [x] 生成并核验两份 full config：`config/rd_only/rd_physical_five_bird_weight_stage037_full_seed42.json`、`config/rd_only/rd_physical_five_parent_loss0_stage037_full_seed42.json`；均 18 epochs、batch=4、max batches=0、test=false、allow_tr_inputs=false。随后用户完成两次 full training，产物分别为 V3/V4 checkpoint；入口由 `--help`/smoke/full summary 验证为 `python -m src.core.train_rd_physical_five --config <file>`。
- [ ] 仅改变 CU 训练目标/采样或 class weight 的单因素实验，目标同时检查 Clutter、Unknown、全五类及 Drone/Bird 非劣化；停止条件为任一主要指标下降或仍无可解释收益。
- [ ] 仅改变 Bird 专项权重或 DBB 训练目标的单因素实验，验证 Bird→Drone/Balloon/Unknown 竞争；不与 CU 改动合并。
- [ ] 仅做 parent gate/late-fusion calibration 的 parent-grouped OOF 实验；当前没有校准参数和拒识标签，不得接线上。
- [ ] 完成真实 production stable/state machine 和 TR-only checkpoint 契约后，再审查 four-state late fusion；TR 缺失时必须严格 RD-only，双缺失必须 `abstain/unavailable`。

### Stage 037 阶段验收

- [x] 标签语义、parent 一致性和 train/val split 审计通过；无重标/重建阻塞。
- [x] parent 聚合统一、方向性混淆字段补齐、96 项 RD 回归与 compileall 通过。
- [x] 两个单因素 smoke 与 hierarchy smoke 通过执行契约；受限结果不作性能结论。
- [ ] CU 专项目标/采样 full 实验、OOF calibration、生产 stable/fusion 接入未完成；当前不是“待部署”状态。Bird weight 与 parent-loss 两个 full 单因素已在 Stage 038 完成 validation-only 归因。

## Stage 038 V3/V4 full validation 与 Bird/Clutter 低召回归因（2026-08-06）

依赖：Stage 035 full checkpoint、Stage 037 标签/parent 审计、统一 parent 聚合契约；只读取 train/val，禁止 test、阈值调优、样本删除和新 full training。

- [x] 严格 reload 并完整评估 V3 `artifacts/20260805-193521/rd-only-five-stage037-bird-weight-full-seed42/best_rd_physical_five_model.pt` 与 V4 `artifacts/20260806-142225/rd-only-five-stage037-parent-loss0-full-seed42/best_rd_physical_five_model.pt`；证据为 `tmp/20260806-rd-five-stage037-v3-full-val.json`、`tmp/20260806-rd-five-stage037-v4-full-val.json` 及 predictions CSV；验收为 2710/2710 val windows、excluded=0、test=false、TR=false。
- [x] 核对单因素边界：V3 相对 Stage 034 仅改变 `class_weights[BirdTarget]` 为1.35；V4 相对 Stage 034 仅改变 `parent_loss_weight` 为0；V3 与 V4 互相不能当作严格单因素比较；证据为三份 `training_summary.json` 配置实载。
- [x] 比较完整 confusion、Bird/Clutter/Unknown 方向性错误、parent 集中度、窗口长度/prefix 分层和置信度；验收为不改变评估分母，低 support 明确标记。
- [x] 核对 loss/forward：flat deployment head；训练仍为 `flat CE + 0.25*cascade CE + parent_loss_weight*parent CE + 0.10*prefix CE + 0.10*micro CE`；V4 的 cascade CE 仍含 parent log-prob，不是完全无父组约束；验收为代码位置与配置一致。
- [x] 形成根因边界：未发现标签映射、split、TR 泄露或单一权重可解释全部下降的证据；已确认五分类竞争、CU 有效梯度占比低、少数 parent/长窗口集中、train/val 泛化差距和高置信度错误共同构成瓶颈；验收为已证实事实与待验证假设分开记录。
- [x] 更新 `findings.md`、`process.md`，记录 V3/V4 指标、差异、证据和未完成项；不修改 TR-only 跟踪文件或代码。
- [ ] CU 专项采样/目标单因素 full 实验、Bird/DBB 专项结构实验、parent-grouped OOF calibration 和线上 stable/fusion 仍待用户确认后执行；停止条件为主要类别非劣化约束失败或收益不可解释。

### Stage 038 验收结论

- [x] V3/V4 均完成 validation-only 严格重载；没有任何 checkpoint 达到五类主要指标全 90.00%，不能标记“待部署”。
- [x] 当前安全基线仍为 flat 五分类，不接入线上；`supports_rejection=false`、UnknownTarget 仍是普通类别。
- [ ] 不启动新的 full training；后续任何优化必须新配置、新 artifact、一次只改一个主要因素，并继续禁止 test 调参。

## Stage 039 ParentGate 第一层路由审查与修复设计（2026-08-06）

依赖：Stage 036–038 full validation、ParentGate/DBB/CU checkpoint、hierarchy composer 代码与契约测试；本阶段只做 validation 复算、代码审查和配置设计，不启动全量训练、不读取 test。

- [x] 确认 V3/V4 是否经过第一层：两份 flat config 的 `deployment_head="flat"`，V3/V4 validation 直接使用五分类 flat logits；验收为不把 flat 混淆误报为 ParentGate 能力。
- [x] 用同一 ParentGate/DBB/CU checkpoint 对 V3/V4 执行完整 hierarchy validation-only；产物 `tmp/20260806-rd-hierarchy-v3-full-val.json`、`tmp/20260806-rd-hierarchy-v4-full-val.json` 及 predictions CSV；2710/2710 windows、test=false、TR=false。
- [x] 分解 ParentGate 路由错误与局部 specialist 错误：统计 DBBGroup→CUGroup、CUGroup→DBBGroup、Bird→CU、Clutter/Unknown→DBB，以及 ParentGate 正确路由条件下 DBB/CU 局部 Recall。
- [x] 审查标签和输出契约：`DBBGroup={Drone,Bird,Balloon}`、`CUGroup={Clutter,Unknown}`；ParentGate 输出2类、DBB输出3类、CU输出2类；composer 使用 `P(parent)*P(local)`，顺序与 `RD_FIVE_LABELS` 一致；22项专科/融合契约测试通过。
- [x] 审查 ParentGate 训练配置、loss、采样和 checkpoint 选择：确认 `class_weights=[1,1]`、无 weighted sampler；train effective evidence support DBB=10109.21、CU=900.94，约11.22:1；selection key 在无 eligible epoch 时先优化 macro-F1，未将 CU Recall 作为第一目标。
- [x] 形成已确认问题：当前 ParentGate window DBB Recall=98.69%、CU Recall=72.53%；DBB→CU=32/2437、CU→DBB=75/273；Bird 有26/523被错误路由到 CU，Clutter 有35/114、Unknown有40/159被错误路由到 DBB。
- [x] 形成未确认假设：提高 CU class weight、balanced sampler、parent-grouped selection 或 OOF threshold 是否改善路由，必须分别作为单因素 train/val 实验验证；不得用 validation 阈值直接上线。
- [x] 实施 ParentGate 单因素修复：固定模型和数据，仅将 CU class weight 设为5.0；同步加入 `eligible_weak_class_guard` selection policy；未修改 sampler、网络结构、数据分母或测试集。证据：Stage 040 独立 smoke/full config、源码 diff 与 summary。
- [x] 为 ParentGate 增加数值路由回归测试：验证 hard-routing 不输出选定父组之外的类别；`tests/test_rd_specialist_contract.py` 新增断言并通过。
- [x] 用户批准后运行 ParentGate smoke；1 epoch、2 train/2 val batches 完成，checkpoint 严格保存/评估，代理未启动 full training。

### Stage 039 当前阻塞

- [ ] ParentGate 现有 full checkpoint `best_selection.eligible=false`，不能作为“第一层已达标”证据。
- [ ] 现有层级组合仍未达到五类 90.00% 门槛；flat 与 hierarchy 必须分开报告。
- [x] 用户已确认并手动完成 Stage 040 full；未覆盖旧 checkpoint/配置，完整 validation 已在 Stage 040 冻结，失败停止条件已执行。

## Stage 040 ParentGate CU 权重单因素 full 训练与验收（2026-08-06—2026-08-07）

依赖：Stage 039 路由审查；固定 physical RD index、train-only normalization/feature stats、seed、网络结构和 optimizer，其余训练因素不变。
证据：`src/core/rd_physical_specialist_config.py`、`src/core/rd_physical_specialist_trainer.py`、`tests/test_rd_specialist_contract.py`、独立 Stage 040 configs、smoke summary/evaluator。
产物：ParentGate CU-weight=5 独立 full checkpoint、完整 ParentGate validation、五类路由变化与同窗 hierarchy 影响结论。
验收：RD-only 不读 TR/test；完整 2710-window val；同时检查 DBB/CU P/R/F1、五个原始类别路由 Recall、五分类 hierarchy 与 parent 指标；任一关键安全指标退化即不替换旧基线。

- [x] 在配置契约中新增 `selection_policy`，允许 `eligible_macro_f1`（兼容历史）和 `eligible_weak_class_guard`；非法值拒绝，旧配置默认行为不变。
- [x] 在 specialist trainer 中实现弱类保护选择键：先比较 eligibility，再比较最小 local recall/F1/precision，最后比较 macro-F1/parent Macro-F1；不改变 loss、标签或评估分母。
- [x] 创建独立 ParentGate CU-weight=5 smoke/full 配置：`config/rd_only/rd_specialist_parent_gate_cu_weight5_stage040_smoke_seed42.json`、`config/rd_only/rd_specialist_parent_gate_cu_weight5_stage040_full_seed42.json`；输出目录唯一，test=false、allow_tr_inputs=false。
- [x] 运行 Stage 040 smoke：1 epoch、2 train batches、2 val batches；train/val windows=12396/2710，loss finite，checkpoint 保存；受限 val 仅8 windows且 CU support=0，明确不作性能结论。
- [x] 运行 smoke evaluator strict reload：`tmp/rd-specialist-parent-stage040-cu-weight5-smoke-eval.json`，`full_split_evaluated=false`、test=false、TR=false、source=cq08_track。
- [x] 增加 hard-routing 父组边界回归和 class-weight loss 传递回归；运行全 RD 测试 `99 passed, 1 warning`，compileall 通过，入口 `--help` 通过。
- [x] 核验 smoke artifact：`artifacts/20260806-205902/rd-specialist-parent-gate-stage040-cu-weight5-smoke-seed42/best_rd_physical_parent_gate_specialist.pt`，SHA-256=`07FF8EB07E8D5AFADD18F8645537C38910957083D3E3A93E4C5F8C3678B9461C`。
- [x] 用户手动完成 ParentGate CU-weight=5 full training：checkpoint=`artifacts/20260806-235753/rd-specialist-parent-gate-stage040-cu-weight5-full-seed42/best_rd_physical_parent_gate_specialist.pt`，epoch=3，SHA-256=`0A8F89EED1EEF408B33BBC7B352C9D9C9E0E49349CA8C9963B60589C536F9806`；summary 为 test=false、TR=false。
- [x] 完成独立 strict validation：`tmp/20260807-rd-parent-stage040-cu-weight5-full-val.json`/predictions CSV，2710/2710 windows、excluded=0、test=false；DBB/CU confusion=`[[2380,57],[60,213]]`。
- [x] 与 Stage 034 同口径比较：CU Recall 72.53%→78.02%（+5.49 个百分点），但 CU Precision 86.09%→78.89%（-7.20）、DBB Recall 98.69%→97.66%（-1.03）、Accuracy 96.05%→95.68%、Macro-F1 88.28%→88.03%；parent Macro-F1 91.34%→89.90%。
- [x] 分解五个原始类别路由：Bird→CU 26→40、Balloon→CU 0→12；Clutter→DBB 35→29、Unknown→DBB 40→31。Bird 路由 Recall 95.03%→92.35%，违反 Bird 非劣化约束。
- [x] 保持 DBB/CU 局部概率固定并只替换新 ParentGate 概率，按 `P(parent)×P(local)` 同窗重组 2710 个 val windows：五分类 predicted-soft Accuracy 92.88%→92.29%、Macro-F1 83.80%→82.89%、Bird Recall 84.32%→81.45%、Clutter Recall 60.53%→63.16%、Unknown Recall 66.67%→71.07%；该结果不读取 RD/test，仅使用两份完整 val predictions CSV 和已审查组合公式。
- [x] 执行停止条件：Stage 040 checkpoint 不替换 Stage 034 ParentGate，不接入线上，不以 CU Recall 单项收益掩盖 Bird、Precision、Macro-F1 和 parent 退化。

### Stage 040 验收边界

- [x] full checkpoint、完整 ParentGate val 与同窗 hierarchy 影响均已核实；全量训练由用户启动，不是本审查启动。
- [x] Stage 040 实验结论为“不合格候选”，旧 Stage 034 ParentGate 保持安全基线。
- [ ] 整个五分类 RD-only 仍未达到五类 90.00% 门槛，因此不是“待部署”状态。

## Stage 041 weighted-CE 正确性与 ParentGate 安全选择修复（2026-08-07）

依赖：Stage 040 full/validation、Stage 034 完整 val 基线报告与逐窗口 predictions、同一 physical RD index/stats、旧 checkpoint strict reload 兼容。
证据：`src/core/rd_physical_five_trainer.py`、`src/core/rd_physical_specialist_config.py`、`src/core/rd_physical_specialist_trainer.py`、`src/core/evaluate_rd_physical_specialist.py`、`tests/test_rd_specialist_contract.py`、Stage 041 独立配置、smoke summary/evaluator。
产物：正确归一化 weighted CE、不可关闭且有三份 SHA provenance 的安全 guard、五类路由 Recall、candidate/qualified checkpoint 契约、独立 full 配置与人工训练命令。
验收：当 class weights 非 1 时 loss 分母使用有效 target-class 权重；旧 checkpoint 可 strict reload；无完整 val provenance 不得启用安全 guard；未合格 epoch 不生成 `best_*`；smoke/test/compile 通过；full 仅由用户执行。

- [x] 复现 weighted CE 分母问题：旧 `_masked_ce` 分子含 target class weight、分母仅含 evidence weight，使 CU=5 同时改变相对类别代价和 batch 总梯度尺度；Stage 040 每 epoch pre-clip gradient norm mean 约2.73–16.56、max 最高2303.91，明显高于旧 equal-weight 约0.89–3.67/最高581.33。证据仅说明实现机制与优化风险，不虚构修复后 full 指标。
- [x] 修复 `_masked_ce`：分母改为 `sum(valid × evidence × target_class_weight)`；新增手工 weighted mean、全局权重缩放不变性、invalid mask 分母排除测试。全部 class weight=1 时数值不变；既有 checkpoint 推理不变。
- [x] 核验 Stage 034/040 历史 checkpoint 在新代码下 CPU strict reload；旧配置默认 `eligible_macro_f1`/`eligible_weak_class_guard` 兼容。
- [x] 追溯旧 `reference_metrics`：原配置数字只存在于配置及派生 summary，未找到对应完整 validation 报告，标记为 provenance 不可验证；Stage 041 改用 `tmp/20260805-rd-parent-stage037-parent-prob-val.json` 与对应 predictions CSV/Stage 034 checkpoint，三份 SHA-256 均在配置中固定并在训练启动前复核。
- [x] 新增 `eligible_reference_guard`：必须 `enforce_non_degradation=true`；无完整 report/checkpoint/predictions provenance 或 SHA/指标不一致直接拒绝；未达标候选按全部 11 项 guard 的最小参考比例选择，不再只追弱类 Recall。
- [x] 新增五个原始类别父组路由 Recall：Drone/Bird/Balloon/Clutter/Unknown 只在 validation 选择与报告中使用；不进入 model forward、loss、sampler、阈值或预测后处理。基线分别为1263/1269、497/523、645/645、79/114、119/159。
- [x] 修复 checkpoint 交付语义：新增默认关闭的 `require_qualified_checkpoint`，保持历史 specialist 配置/旧 checkpoint 的 `best_*` 行为；Stage 041 显式设为 true，每轮可保存 `candidate_*` 供诊断，只有全部 guard 通过才写 `best_*`。summary 明确 `training_outcome`、`qualified`、candidate 与 best；无合格 epoch 时 `best_checkpoint=""`、`best_epoch=-1`。
- [x] specialist 独立 evaluator 输出同一 `route_recall_by_original_class`，保证训练 summary 与 full strict evaluator 可交叉核对。
- [x] 创建 `config/rd_only/rd_specialist_parent_gate_weighted_ce_fix_stage041_smoke_seed42.json` 与 `..._full_seed42.json`；full 固定18 epochs、batch=4、lr=1e-4、seed=42、CUDA、class weights=[1,5]、max batches=0、test=false、TR=false、唯一 experiment name。
- [x] 运行最终 Stage 041 smoke：1 epoch、2 train+2 val batches；artifact=`artifacts/20260807-073648/rd-specialist-parent-gate-stage041-weighted-ce-fix-smoke-seed42`，candidate SHA-256=`607B665D621CF3ED98148C54FBEB55A823530221FB75F91A7EEAD146F05D091F`；`require_qualified_checkpoint=true` 实载、loss/gradient finite、reference verified、test/TR 均 false。
- [x] 验证不合格 smoke 的安全行为：受限 val 8 个样本均为 DBB、CU support=0，summary=`no_qualified_checkpoint`、`qualified=false`、`best_checkpoint=""`；strict evaluator=`tmp/20260807-rd-parent-stage041-weighted-ce-fix-smoke-eval.json`，`full_split_evaluated=false`，不作性能结论。
- [x] 运行入口 `--help`、compileall、`git diff --check`、定向测试16 passed、全 RD 回归105 passed/1既有 warning；旧 Stage 034/040 checkpoint strict reload 通过。
- [ ] 用户手动执行 Stage 041 18-epoch full training；本审查不得启动。
- [ ] 训练完成后读取真实 summary；若 `qualified=false`，只评估 `candidate_*` 诊断且禁止替换基线；若 `qualified=true`，对 `best_*` 执行完整 ParentGate val 和四模型 hierarchy val。
- [ ] full hierarchy 必须同时比较总体/五类/parent/source、Bird/Clutter/Unknown、DBB↔CU 方向；任一旧安全指标退化即停止，不读取 test 调参。
- [ ] 若 Stage 041 仍不合格，下一实验只能单独选择 balanced sampler 或 parent-grouped sampling，不与权重/网络/阈值同时改动。

### Stage 041 当前 readiness 边界

- [x] 待全量训练：是；训练入口、配置、数据/stats、seed/device/batch/epochs/lr、test-off、无 TR 输入、输出名、provenance、smoke、strict reload 与回归均已验证。
- [x] Stage 041 full 尚未由本审查启动；必须由用户手动执行。
- [ ] 待部署：否；Stage 041 尚无 full 指标，历史所有五分类/级联结果也未达到全类 90.00%，拒识与 online stable/fusion 未验证。
