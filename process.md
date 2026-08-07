# RD-only 全链路审查过程记录

更新时间：2026-08-06（Asia/Shanghai）
范围：仅 RD-only；未修改 `TRonlyTask_plan.md`、`TRonlyProcess.md`、`TRonlyFindings.md`，也未修改 TR-only 代码。Stage 013–039 历史实施/审查保留；Stage 040 ParentGate CU-weight=5 full 已由用户完成并完成 validation/停止验收；Stage 041 weighted-CE/provenance/五类路由 guard/checkpoint 语义修复、smoke与回归已完成，full 尚未由本审查启动。OOF calibration、production stable/fusion 仍未执行。未读取 test、未重建 canonical/store、未接线上 stable/fusion。

## 13. Stage 033–034 修复执行记录

- 用户回传 `expert1_rd_physical_feature_stats_v5_clean.json` 后只读核验：SHA-256=`27A555B96F91FE3EAA837EDF36597A438F0D24FA0571A6A50DA62DD7B9E6F1CA`；schema/train-only/index hash/normalization hash/generator hash 均一致；quality 7项无 `canonical_time`。
- 修改 `src/core/rd_physical_drone_bird_trainer.py`：新增 `compute_window_evidence_weights`；clean 训练、evaluator 和 checkpoint selection 不再调用/依赖 parent/content ID 权重，loss 分母改为当前 batch 有效 evidence 权重；legacy parent weighting helper 仅保留历史测试/报告用途。
- 修改 `src/core/rd_physical_five_model.py`、`rd_physical_five_artifacts.py`、`rd_physical_specialist_model.py`、`rd_physical_specialist_artifacts.py`：新增 local head mode；specialist checkpoint schema v2，manifest 显式记录 kind/head_mode/local_class_count。
- 修改 `src/core/rd_physical_specialist_config.py`、`rd_physical_specialist_trainer.py`：新增 `reference_metrics`、`non_degradation_tolerance`、`enforce_non_degradation`，逐局部类别选择 checkpoint；更新三套 full/smoke config。
- 新增 clean 二分类配置：`config/rd_only/rd_physical_drone_bird_v10_clean_full_seed42.json` 与对应 smoke；quality_schema=`rd_only_v5`、quality_dim=7、feature_stats=v5_clean。
- 修复后定向回归 `31 passed`；最终全 RD 测试 `93 passed, 1 warning`；compileall（src/core、src/process、config）通过。
- Smoke：clean 二分类输出 `artifacts/20260804-023612/rd-only-stage034-clean-smoke-seed42`；ParentGate `artifacts/20260804-023644/...`；DBB `artifacts/20260804-023718/...`；CU `artifacts/20260804-023743/...`；五分类新 smoke `artifacts/20260804-025249/rd-only-five-stage021-smoke-seed42`。均仅 train/val 限量 batches，summary/test=false，TR inputs=false；受限指标不作能力结论。
- Strict reload/evaluator：三 specialist v2 输出 `tmp/stage034-parent-reload.json`、`tmp/stage034-dbb-reload.json`、`tmp/stage034-cu-reload.json`，每份4个 val windows、`full_split_evaluated=false`；clean 二分类 CPU strict reload 通过；旧 Stage011 checkpoint 被 inference contract 明确拒绝。
- 本轮未执行：OOF calibration、test evaluation、线上 physical RD stable/fusion caller 接入；四个 full training 已由用户在本轮前完成，完整 validation/组合诊断已在 Stage 036 执行。
- full 配置实载验收：clean 二分类、五分类、ParentGate、DBB、CU 均 dataclass load 成功，`run_test_evaluation=false`，输出目录均不存在且互不复用；full 逻辑目录已固定为 Stage034 独立目录。
- 交付命令已按真实 `--help` 生成：训练入口仅支持 `--config`；full 训练命令包含 `.venv\\Scripts\\python.exe`、Stage034 独立配置、日志目录和 test-off 配置。命令只输出给用户，本代理未调用 `Start-Process` 或启动 full training。

## 1. 审查起点与工作区保护

- 审查前完整读取 `task_plan.md`（966 行）、`process.md`（736 行）、`findings.md`（907 行）。
- 审查前三文件 SHA-256：
  - `task_plan.md`: `78E99A74E6C647E9B4B635BDA7769B9EC95A051BE4130712566667C92D1B4925`
  - `process.md`: `B1C17556638313578DF6E715435908196D1C84C3FFED4ED299C29FAB0B9890EB`
  - `findings.md`: `98A62799D810F884F47E6D711E879CCE15A5CA7CD02CD85563F0D7A9607C2DA5`
- 审查前 TR-only 文件 SHA-256：
  - `TRonlyTask_plan.md`: `ECFA1F4DA1B800176F9BF94A6730CB1BDC40A4DCC3E7A53DC5A16E212724D0D4`
  - `TRonlyProcess.md`: `02BBB4D381240B6D2D2A70C991C7D7E2563FEF8EC58B88BC06D6A0E2634A1D87`
  - `TRonlyFindings.md`: `0DE2DBCEFC38357632ECF1C4E4674324035DBC21203F37BDDBC065284F66EA2C`
- 工作区在审查前已有大量 modified/untracked 用户资产；本轮没有 reset/checkout/清理，也没有修改无关 TR-only 代码。
- 未发现额外 `AGENTS.md`/`CLAUDE.md` 项目级指令。

## 2. 仍有效的历史数据链事实

### 2.1 Stage 025/026 内容安全 split 与 causal window

- `expert1_realtime_causal_window_index_v8_rd_content.csv` 是 active RD-only split ancestor，SHA-256=`d4b1b8a2f481ac30ef288cd5e14ebc9e293b5636407db86665fb3d2e2d560ca2`。
- Stage 025 structural audit：parent cross-split=0、RD fingerprint cross-split=0、3 个重复 fingerprint 被合并为 2045 个 split groups；fingerprint algorithm=`sha256-npz-arrays-v1`。
- Stage 026 structural audit：22085 windows、1549 parents、cross-split parent/group=0、noncausal windows=0、fusion eligible=0、最大 tensor 时间边界误差 `2.3803711002301498e-07 s`。
- window contract：0.5 s window、0.1 s stride、min frames=12、包含 prefix；4288 个 `<12 frames` 窗口标为 `too_few_rd_frames`，不是为性能而删除。

### 2.2 Physical v3.1 canonical store

- merge summary：106026 frames、1549 parents；reference/missing/unexpected/duplicate/tensor-index mismatch/missing tensor/invalid shape/partial range/partial velocity 全为 0。
- canonical shape=`21x384`，range offset=`[-75,75] m`，velocity=`[-85,85] m/s`，anti-alias frames=33584。
- artifact inventory：1549 tensor files、2,773,411,553 bytes、content SHA-256=`596b9df5764665c6f91635dd4af113939fb486ca6122841be728f927485ef0b6`。
- 时间是累计 CPI acquisition duration，不是 wall-clock/TR 同步。
- 既有 metadata audit 已确认 106026/106026 帧存在 `debug_info.detected_range_m`；所以删除 `track_range_m` fallback 不要求重建现有 store。

### 2.3 Train-only statistics

- active index SHA-256=`c9cbf3ff9a354766583df3a896594420b65f566206b65b80c2c7522e572d19c1`。
- normalization v4 SHA-256=`2f28dc25f0c3454beadb258c25b5a3f096edc4cebbb842060b5130642aba0e5b`；split=train、labels=Drone/Bird、713 parents、712 content groups、test=false。
- feature stats v4 SHA-256=`ecaf684947467fb76672dd0b77ed380f41a9d6fc034c8ee0a59cb2a72fd428e9`；8207 windows、713 parents、712 groups、test=false。
- feature quality audit：`log_frame_count.unique_count=82`、`log_duration.unique_count=1369`，但 `dynamic_range_proxy.unique_count=1`；该常量特征不能作为判别能力证据。

## 3. Active 与其他 RD 分支的调用链审查

### 3.1 Active Physical v9

```text
src/core/train_rd_physical_drone_bird.py
  -> PhysicalRDDroneBirdConfig
  -> PhysicalRDWindowDataset / collate_physical_rd_windows
  -> load_physical_rd_artifacts / build_physical_rd_model
  -> PhysicalRDDroneBirdClassifier
  -> train_physical_rd_drone_bird
  -> best_rd_physical_drone_bird_model.pt
```

- `RD_ONLY_LABELS=("DroneTarget","BirdTarget")`。
- `PhysicalRDDroneBirdClassifier.label_names=("DroneTarget","BirdTarget")`。
- `PhysicalRDDroneBirdClassifier.supports_rejection=False`。
- forward 参数只包括 6-channel RD、RD time、padding mask、8 项 RD quality、range axis、velocity axis、raw RD dB；无 TR、trajectory、path、source、ID、split 或 label 参数。
- active Dataset 严格过滤 Drone/Bird；全 index 中 Balloon/Clutter/Unknown 不进入 active loader。
- `rd_fusion_eligible=True` 行会被 Dataset 拒绝；当前 active 行为 0。

### 3.2 Legacy/独立分支

- `rd_drone_bird_*`：旧 v7/TR-window anchor 的 Drone/Bird 诊断，不是 active Physical。
- `rd_patch_residual_*`：冻结旧 sequence trunk 的 patch residual，仍使用旧 anchor/index 语义，checkpoint 按 window F1；不得混入 active 指标。
- `rd_novel_parent_*`：旧 Physical-v2 novel-evidence parent state，两类，不进入 active v9。
- `rd_novel_anchor_residual_*`：trainer 直接抛出 retired/incompatible 错误，不可训练。
- `rd_segment_store`、`build_expert1_rd_segment_store`、`forwardlayer/rd.py`：legacy Expert1 链，active Physical 不引用。
- 项目 `system.py`/`expert1_streaming.py` 的 Unknown/reject/stable 逻辑属于五分类 Expert1/System，不连接 active Physical v9。

## 4. 完整 validation 基线与 Stage 011 全量训练验收（均由用户手动完成）

### 4.1 v8 seed-42

- artifact：`artifacts/20260801-173510/20260801-stage059-rd-physical-v8-fullbudget-v1-18epochs`
- best epoch=16，test=false。
- window 1792：Accuracy=99.1629%、Macro-Precision=99.1814%、Macro-Recall/Balanced Accuracy=98.7908%、Macro-F1=98.9835%。
- window Drone：support=1269，P/R/F1=99.1379%/99.6848%/99.4106%。
- window Bird：support=523，P/R/F1=99.2248%/97.8967%/98.5563%。
- window confusion=`[[1265,4],[11,512]]`。
- parent 148：Accuracy=99.3243%、Macro-Precision=99.5283%、Macro-Recall/Balanced Accuracy=98.8372%、Macro-F1=99.1748%。
- parent confusion=`[[105,0],[1,42]]`。
- 当前 schema v9 不支持把 v8 checkpoint 当作 active checkpoint 复载；只保留为历史性能对照。

### 4.2 v9 seed-42

- artifact：`artifacts/20260801-204123/20260801-stage061-rd-physical-v9-performance-fullbudget-v1-18epochs`
- 该训练在本轮审查开始前由用户启动，本轮没有代为启动；best epoch=8，test=false。
- window 1792：Accuracy=97.5446%、Macro-Precision=97.8041%、Macro-Recall/Balanced Accuracy=96.2431%、Macro-F1=96.9815%。
- window Drone：support=1269，P/R/F1=97.2244%/99.3696%/98.2853%。
- window Bird：support=523，P/R/F1=98.3838%/93.1166%/95.6778%。
- window confusion=`[[1261,8],[36,487]]`；Drone→Bird=8，Bird→Drone=36。
- parent 148：Accuracy=99.3243%、Macro-Precision=99.5283%、Macro-Recall/Balanced Accuracy=98.8372%、Macro-F1=99.1748%。
- parent Drone：support=105，P/R/F1=99.0566%/100.0000%/99.5261%；Bird：support=43，P/R/F1=100.0000%/97.6744%/98.8235%。
- parent confusion=`[[105,0],[1,42]]`。
- v9 所有要求的 window/parent/两类 P/R/F1 均 >90%，但 window 指标低于 v8；v9 同时改变多个因素，不能归因于任何单一改动。

### 4.3 Stage 011 v9 full seed-42（当前交付 checkpoint）

- artifact：`artifacts/20260802-010756/rd-only-stage011-v9-full-seed42-18epochs`；训练由用户手动执行，本代理未启动。
- `training_progress.json`：status=`completed`、epoch=18、best epoch=13；`training_history.json` 有 18 条 epoch 记录；`test_evaluation_executed=false`、`supports_rejection=false`。
- best checkpoint：`best_rd_physical_drone_bird_model.pt`，SHA-256=`07A14677016D9E0A39208277F63ED5E7BB6F1311D12131A98C78953636443BF2`。
- window 1792、excluded=0：Accuracy=97.9911%，Macro-Precision/Recall/F1/Balanced Accuracy=97.5699%。
- window Drone：support=1269，P/R/F1=98.5816%；Bird：support=523，P/R/F1=96.5583%。
- window confusion=`[[1251,18],[18,505]]`；Drone→Bird=18、Bird→Drone=18。
- parent weighted/unweighted mean-logit 148：Accuracy=98.6486%，Macro-Precision/Recall/F1/Balanced Accuracy=98.3610%。
- parent Drone：support=105，P/R/F1=99.0476%；Bird：support=43，P/R/F1=97.6744%；confusion=`[[104,1],[1,42]]`。
- 所有 18 个 epoch 的 train/val loss 有限；每个 epoch 的 2052 个训练 batch gradient audit 均 `all_finite=true`。最大 batch pre-clip gradient norm=674.6139（epoch 8），现有 `max_norm=1.0` clipping；这是需披露的优化稳定性风险，不是 non-finite 或训练失败。
- checkpoint 规则复核：epoch 8 与 13 的 parent Macro-F1、最低分类别 parent Recall、最低分类别 parent Precision完全相同；epoch 13 window Macro-F1=97.5699% 高于 epoch 8 的 97.5038%，故按第四级 tie-break 正确当选。
- artifact 哈希：summary=`4A37815A6B37ACB5EDA0AC1ADD334BC349AF1389922537C488C32A5F2E6FC3ED`；progress=`006AA26C9E420B766672822EEC8F8A2E02D5EBFF0EBFBD3558323697E1B841CF`；history=`99D9D0E85C2973C8F34220A091565F267FD77B72E84815ACE7FA19940B736ABE`。

## 5. 本轮手术式代码与配置变更

### 5.1 `src/core/rd_physical_drone_bird_trainer.py`

- `_class_metrics` 新增 Macro-Precision、Macro-Recall、Balanced Accuracy、sample count、confusion matrix、Drone→Bird/Bird→Drone。
- `_run_epoch` 新增 non-finite loss/gradient 拒绝与 gradient norm audit。
- 训练新增 `training_progress.json` 的 started/running/completed 状态、数据支持、source/session 字段、checkpoint 选择规则。
- 不改变 forward 输入、模型结构、标签、数据分母或 checkpoint 选择优先级。
- 最终 SHA-256=`4c77e2a7629c7b3da43ac93c11e77b1dda98f0097a4f0775bb76363b59fee6e6`。

### 5.2 `src/core/evaluate_rd_physical_drone_bird.py`

- 新增 source/session/online stable/rejection capability/evaluation completion 报告。
- 新增 `--max-batches`；0=完整 validation，正数=restricted smoke，报告强制标记 incomplete/no performance claim。
- restricted smoke 使用确定性 Drone/Bird 均衡子集，不改变默认完整 validation 分母。
- evaluator 仍硬拒绝非 `val` split。
- 最终 SHA-256=`1375241ca9f567175352bb9a26c17aeee3275893efa59c7277dfcd784266ece7`。

### 5.3 `src/core/analyze_rd_physical_validation.py`

- label mapping 改为严格 Drone/Bird，Unknown 等标签会失败。
- calibration 明确 `posthoc_calibration_applied=false`、`threshold_selection_performed=false`。
- cumulative prefix 保留 causal 口径，但稳定时间重命名为 retrospective，并明确不得作为 online latency。
- 新增最终 parent 分类别指标、source stratification、online stable/rejection unavailable 声明。
- 最终 SHA-256=`f2d53cff4e3ce7a6c8f474592410abb48e7ed54e235d7c5cd6fcca21fe885d35`。

### 5.4 `src/process/build_cq08_rd_physical_canonical_store.py`

- 删除 `debug_info.track_range_m` fallback；未来构建强制 `debug_info.detected_range_m`。
- metadata 新增 `target_range_reference_source=debug_info.detected_range_m`，summary 新增 reference policy。
- 未重写现有 canonical tensors。
- 最终 SHA-256=`c10375593a1e5ad7237547e6691f488de296b9be66dc1c21753f13bb2f6fcc56`。

### 5.5 独立配置

- smoke：`config/rd_only/rd_physical_drone_bird_v9_smoke_seed42.json`，SHA-256=`a4db9471d094462fe526a8e5a156871317d75f473256c0931d7fbe112d819d49`。
- full：`config/rd_only/rd_physical_drone_bird_v9_full_seed42.json`，SHA-256=`e5b5de8a89cc5ec06a8f7531de6a3afe99fd294593c13886f44f43b36050cae4`。
- full 与已有 v9 已验证超参数逐字段一致，只有独立 output name；smoke 只把 epochs/train batches/val batches 限制为 1/2/2，并把 log interval 设为 1。

### 5.6 测试变更

- `tests/test_rd_physical_trainer.py`：新增宏指标、Balanced Accuracy、混淆方向测试。
- `tests/test_analyze_rd_physical_validation.py`：新增 retrospective/online/rejection/source 测试。
- `tests/test_rd_physical_contract.py`：新增 detected-range-only 与无 TR/provenance forward signature 测试。
- `tests/test_evaluate_rd_physical_reporting.py`：新增 source/session/strict label/balanced smoke tests。
- `tests/` 被项目 `.gitignore` 忽略，但本地测试文件存在并已实际执行。

## 6. 本轮实际命令与结果

### 6.1 CLI/config 验证

- `\.venv\Scripts\python.exe -m src.core.train_rd_physical_drone_bird --help`：exit 0；实际参数仅 `--config`。
- `\.venv\Scripts\python.exe -m src.core.evaluate_rd_physical_drone_bird --help`：exit 0；包含 `--max-batches`。
- `\.venv\Scripts\python.exe -m src.core.analyze_rd_physical_validation --help`：exit 0。
- full config 实载为 `PhysicalRDDroneBirdConfig`：exit 0；CUDA available=true、device=`NVIDIA GeForce RTX 4060 Laptop GPU`。
- 前台、后台 `Start-Process`、训练后完整 validation/analyzer 三条 PowerShell 命令均用 `[scriptblock]::Create` 做了仅语法解析：全部 `ok`；没有执行 full train 或后台进程。

### 6.2 静态检查

- `python -m py_compile`（修改源码与测试，pycache 重定向到本轮 tmp）：exit 0。
- `python -m compileall -q src/core src/process config`：exit 0。
- `git diff --check`：exit 0；只有既有 LF→CRLF warning，无 whitespace error。

### 6.3 单测

- 首组 active/contract：31 passed，1 warning。
- 第二组 dataset/artifact/window/pairing：29 passed。
- evaluator/analyzer 增量：6 passed。
- trainer 增量：9 passed。
- 历史二分类 RD 专项集合：**80 passed，1 warning，0 failed，30.06 s**；本轮五分类前置回归另计 29 passed。
- warning：PyTorch nested tensor prototype API；不是本轮错误。

### 6.4 训练 smoke

- 第一次尝试因工具命令超时设为 10 s，在 10.7 s 被包装层终止；不是代码失败，未形成可验收结果。
- 中间 smoke 成功：`artifacts/20260802-002511/rd-only-stage010-v9-smoke-seed42`；后续由最终 smoke 替代。
- 最终命令：

```powershell
.\.venv\Scripts\python.exe -m src.core.train_rd_physical_drone_bird --config config/rd_only/rd_physical_drone_bird_v9_smoke_seed42.json
```

- 最终 artifact：`artifacts/20260802-003631/rd-only-stage010-v9-smoke-seed42`。
- 限制：epochs=1、max train batches=2、max val batches=2、batch=4、test=false。
- train support：Drone=5、Bird=3；train loss=2.6712619、val loss=0.2394688，均有限。
- pre-clip gradient norms：min=21.1628、mean=56.9917、max=92.8206、all_finite=true；trainer 使用 max_norm=1.0 clipping。
- progress status=completed；best checkpoint SHA-256=`69965159c42d0911306ea943ce2e3ac5ef76a4fb78436b38494da3f1f4c56c30`。
- smoke validation 前 2 个顺序 batch 恰好全为 Drone，故其 Macro-F1=50% 仅是 smoke 子集现象，禁止作为性能结论。

### 6.5 最终 checkpoint reload/evaluator/analyzer

- evaluator 命令使用最终 smoke checkpoint、`--split val --max-batches 2`，exit 0。
- report：`tmp/20260802-rd-only-stage010-final-smoke-validation.json`，SHA-256=`d1d52a59bbcdf8283dc521c1c3b68281f99d214f4115b6f42d927b431e02b062`。
- 8 个均衡窗口：Drone=4、Bird=4；full=false；excluded_by_batch_limit=1784；source available、session unavailable、online stable unavailable、rejection unsupported、test=false。
- analyzer：`tmp/20260802-rd-only-stage010-final-smoke-analysis.json`，SHA-256=`9771cbd4962df5198893172a82f5907b4ef45884b068ac46ddbd0b59db9c0413`，exit 0。
- analyzer 明确 calibration 未应用、threshold 未选择、retrospective stable 不得作为 online latency。
- 已有 v9 full checkpoint 也在当前代码下 strict reload：`tmp/20260802-rd-only-v9-full-checkpoint-reload-smoke.json`，SHA-256=`a6e9677e2884c7747f928383bd4a4f1b81e65f258d8207201d81c7423466f366`。
- Stage 011 best checkpoint 在当前代码下 strict reload + evaluator `--split val --max-batches 2` 通过；报告=`tmp/20260802-rd-only-stage011-full-checkpoint-reload-smoke.json`，SHA-256=`37939F5CB56BBE469356D1EA222F63430F195D74A83F6CDBA0681FA15866F5B2`。
- 该 reload report 仅覆盖 8 个确定性平衡 windows（Drone=4、Bird=4），用于证明 schema/artifact/model state/forward/evaluator 兼容；其 50% Accuracy 不是全量 checkpoint 性能证据。报告明确 test=false、fusion=false、supports_rejection=false、session/online stable unavailable。

### 6.6 latency smoke

- 命令限制 `warmup=1`、`sample-count=2`、batch=1，exit 0。
- report：`tmp/20260802-rd-only-stage010-latency-smoke.json`，SHA-256=`cf2b72baa1b434b7a862162c5f809382e9493c9575ccb1eff3b29af2cea79511`。
- 仅 2 个样本：preprocess mean=16.2685 ms、H2D mean=2.2573 ms、forward mean=31.5638 ms、total mean=50.1445 ms。
- 样本太少，不能推断生产 SLA；该工具是 validation dataset 顺序重放，不是 production ring buffer/stable decision。

## 7. 未执行事项、当前状态与资源证据

- 用户已执行 full config 并完成 18 epochs；本代理没有启动该训练。
- 本次仍没有执行独立完整新格式 evaluator/analyzer、test、阈值搜索、校准拟合、全量 canonical rebuild 或 feature stats rebuild。trainer 的完整 validation 已覆盖 1792 windows/148 parents、excluded=0，但不含逐样本 predictions/source/calibration 分析文件。
- v9 best checkpoint 约 15.2 MB；canonical tensors 约 2.77 GB。
- 当前 Stage 011 相比历史旧 v9：window Macro-F1 96.98%→97.57%，Bird Recall 93.12%→96.56%，混淆从 Drone→Bird=8/Bird→Drone=36 变为 18/18；相比历史 v8，window Macro-F1 仍低约 1.41 个百分点，Bird Recall 低约 1.34 个百分点。不同 run 间差异不能归因于单一因素。
- 全量训练最大 pre-clip gradient norm=674.6139，虽全部有限且被 clip 到 max_norm=1.0，仍需在后续单因素实验监控。
- 当前不支持 optimizer/RNG 训练状态 resume，只支持 strict best inference checkpoint reload。
- 当前状态（历史快照）：该记录产生时五分类全量训练尚未启动；随后用户已手动完成 Stage 022，产物和 Stage 023 validation 见文末新增记录；历史二分类 full checkpoint 已验收，test 继续关闭。
- 本次训练后验收开始时只读记录的 TR-only 跟踪文件 SHA-256 为：`TRonlyTask_plan.md=F6AE41B4F250A1035E59ACEFA8493D09E402A9E5866CAD0B5543D260E144E081`、`TRonlyProcess.md=5C998C5815405789DA256862AF3CB7C90A35642805D4689C6ED009A88DB18EF5`、`TRonlyFindings.md=091AD4C5FAEB2DD6A6015B25BCE8D1FE89492A22108A1F3BD7D35566CD0EE167`；它们与最初审查快照不同，本次不回滚、不编辑，结束时复核哈希保持一致。

## 8. 本轮设计冻结与等待确认

### 8.1 已读取和保留的依据

- 已重新核对三个 RD-only 跟踪文件的当前内容和工作区状态；保留历史 Stage 001–011 的代码、配置、数据、日志和测试事实，不将历史二分类结果改写成五分类结果。
- 已核对 `src/core/experts.py` 中 `LOW_ALTITUDE_CLASSES` 的实际顺序：`DroneTarget`、`BirdTarget`、`BalloonTarget`、`ClutterTarget`、`UnknownTarget`；新设计固定沿用该顺序。
- 已核对当前工作区存在用户对 TR-only 相关代码、配置和测试的修改；本轮未触碰这些文件。TR-only 入口将来保留，但 RD-only 训练阶段不传入或读取轨迹。
- 现有二分类 checkpoint `artifacts/20260802-010756/rd-only-stage011-v9-full-seed42-18epochs/best_rd_physical_drone_bird_model.pt` 仅作 Drone/Bird 非劣化参考，不能直接作为五分类 head 或五分类性能证据。

### 8.2 设计决定

- RD-only 五分类采用共享 RD encoder + 软级联概率头 + 直接五分类 flat 旁路。软级联不使用 hard argmax 中间标签；flat 与 cascade 独立输出，先比较再决定部署输出。
- TR 入口保留为系统接口兼容性要求，但训练 RD-only 时只构建 RD dataset/loader，任何 TR tensor、trajectory、TR label、match、path、ID、split 或 pairing status 都不得进入 forward、loss、采样、阈值、校准或后处理。
- 未来 late fusion 采用四态 availability router：both→校准 late fusion，RD-only→RD-only 透传，TR-only→TR-only 透传，none→`abstain/unavailable`。UnknownTarget 分类和 reject/abstain 分开评估。

### 8.3 分阶段执行顺序

1. Stage 013 先完成字段 provenance、五类标签、split、pairing 和泄露阻断；任何非零交叉或未解释字段都停止。
2. Stage 014 再建立独立五类 normalization/feature stats/artifact；不覆盖历史二分类 artifact。
3. Stage 015–016 实现模型、loss、采样和 checkpoint；先 flat 基线，再软级联/单因素候选。
4. Stage 017 做五类评估和 Drone/Bird 非劣化门槛；Stage 018 单独核实 TR-only checkpoint 契约。
5. Stage 019–020 最后实现和评估 late fusion 四种 availability 模式；缺失模态安全优先。
6. Stage 021 只做静态、契约、泄露测试和短 smoke；Stage 022 在全部通过后才准备人工全量训练命令。

### 8.4 历史暂停点（已被批准实施覆盖）

- 该段记录的是批准实施前的暂停状态；随后用户已批准，本轮已执行 Stage 013 代码/数据审计并进入 Stage 014。
- 当前仍未执行五分类训练、canonical tensor 全量重建、测试集评估、阈值/校准/融合权重选择或全量训练命令。

## 9. Stage 013–014 执行记录与人工暂停点

### 9.1 Stage 013 只读审计结果

- physical canonical frame index 和 RD-only causal window index 已读取并按字段、标签、split、parent/group、时间和 pairing 复核；五类标签 exact match，`UnknownTarget` 为规范拼写。
- physical window index SHA-256=`c9cbf3ff9a354766583df3a896594420b65f566206b65b80c2c7522e572d19c1`；physical frame index SHA-256=`0a4176660c85ecd6298cd505070f681b7ee880b1b9ed9e73b05ff2606d656c59`；split ancestor SHA-256=`d4b1b8a2f481ac30ef288cd5e14ebc9e293b5636407db86665fb3d2e2d560ca2`。
- Stage 013 实际核验命令使用 Python/pandas：统计 22085 windows、五类 trainable support、parent/group cross-split 和 future count；结果为 parent/group cross-split=0、future=0、fusion eligible=0。
- pairing 命令：

```powershell
.\.venv\Scripts\python.exe -m src.process.audit_expert1_rd_pairing --manifest-name expert1_rd_segment_manifest.csv --output tmp/20260802-stage013-pairing-audit.json --records-output tmp/20260802-stage013-pairing-audit-records.csv
```

- pairing 输出：25021 segments；fusion_eligible=0；rejected=24986；diagnostic_only=35。该报告只作字段边界证据，不进入模型训练。

### 9.2 发现的旧链路问题与隔离策略

- 旧 `PhysicalRDWindowDataset` 的 quality metadata 将 `rd_time_sync_status == canonical_parent_association` 映射为 `canonical_time`；该字段携带 pairing/关联语义，不满足新五分类 RD-only 输入边界。
- 直接改旧 builder 会改变 generator hash，导致历史二分类 checkpoint strict provenance reload 失败；已实测该回归并修复为独立五类 builder，不修改旧 v4 builder、旧 config 或旧 checkpoint contract。
- `rd_physical_window_dataset.py` 现在支持显式 labels/quality names；默认二分类仍使用原 8 维 quality，新五分类将使用不含 `canonical_time` 的 7 维 quality。

### 9.3 五类 stats 生成器和用户执行结果

- 新增：`src/process/build_rd_physical_five_normalization_stats.py`、`src/process/build_rd_physical_five_feature_stats.py`；schema=`rd_physical_train_normalization_v4_five`（normalization）和 `rd_physical_five_feature_stats_v1`（feature stats）。
- 旧二分类回归命令及结果：`tests/test_rd_physical_window_dataset.py tests/test_rd_physical_artifacts.py`，15 passed；新文件 py_compile 通过。
- stats 生成需扫描训练窗口和 canonical tensors，按用户授权由用户手动完成；本代理未后台启动。

```powershell
.\.venv\Scripts\python.exe -m src.process.build_rd_physical_five_normalization_stats --index-name expert1_rd_only_causal_window_index_v1_physical_v3_1.csv --output-name expert1_rd_physical_train_db_normalization_five_v1.json --sample-stride 257
.\.venv\Scripts\python.exe -m src.process.build_rd_physical_five_feature_stats --index-name expert1_rd_only_causal_window_index_v1_physical_v3_1.csv --normalization-stats-name expert1_rd_physical_train_db_normalization_five_v1.json --output-name expert1_rd_physical_feature_stats_five_v1.json --batch-size 8
```

- 已核验两个输出存在且 provenance/finite/train-only/label order 正确；normalization SHA=`76dc9c27df29de9a1d8d2e681966c191ff9a21fbbae1cb15645123f53d0b6609`，feature stats SHA=`df792ed08bf13590df7fc06526dec64b69eff23c576786e7f0616c78270b578d`。

### 9.4 历史遗留清理（仅供用户手动执行）

- 已核对 `tmp/20260728-rd-only-cleanup-verified-plan.json` 和 `tmp/20260728-execute-rd-only-cleanup.ps1`：目标均在项目目录内，且不包含当前 `artifacts/20260802-010756`、`artifacts/20260802-003631`、physical v3.1 canonical store、五类 window index 或五类 stats 输出。
- 先执行只读 dry-run，确认目标清单和回收大小：

```powershell
& .\tmp\20260728-execute-rd-only-cleanup.ps1
```

- 只有确认 dry-run 列出的目标均为可删除历史遗留后，才由用户手动执行删除：

```powershell
& .\tmp\20260728-execute-rd-only-cleanup.ps1 -Execute
```

- 脚本会在 `tmp/20260728-rd-only-cleanup-execution-report.json` 写删除报告；该删除不可由本代理回收，执行前请自行保留必要备份。禁止将 `-Execute` 改为 workspace 根目录、旧清单外的 glob 或递归目标。

### 9.5 Stage 015–021 当前执行记录

- 五分类新增链路：`rd_physical_five_dataset.py`、`rd_physical_five_model.py`、`rd_physical_five_config.py`、`rd_physical_five_artifacts.py`、`rd_physical_five_trainer.py`、`train_rd_physical_five.py`、`evaluate_rd_physical_five.py`；旧二分类 `rd_physical_window_dataset.py`/model/schema 已保持 strict 回归通过。注意：五类链路使用独立 dataset，不向旧二分类 dataset 增加五类常量。
- 五分类配置固定 `quality_dim=7`、`input_channels=6`、`parent_partition=[[0,1,2],[3,4]]`、`allow_tr_inputs=false`、`run_test_evaluation=false`。新增 best selector 会先检查 Drone/Bird 相对于 Stage 011 v9 validation 的 P/R/F1 非劣化，再比较 window/parent 指标；smoke 只能验证选择逻辑执行，不能证明资格满足。
- 五分类 smoke 命令：

```powershell
.\.venv\Scripts\python.exe -m src.core.train_rd_physical_five --config config/rd_only/rd_physical_five_smoke_seed42.json
```

- 初轮 smoke artifact=`artifacts/20260803-020321/rd-only-five-stage021-smoke-seed42`；1 epoch、2 train batches、2 val batches；该产物已被后续代码 hash 变化后的 smoke 替代。
- 当前代码修正：五分类 forward 明确固定 `[B,T,6,21,384]`、finite axes/raw dB、拒绝 all-padding；evaluator 明确 `deployment_head` 和 selected metrics；trainer 只将 `rd_present=true` 样本纳入 classification metrics，避免缺失窗口进入分母。
- 已完成用户清理核验：`tmp/20260728-rd-only-cleanup-execution-report.json` DeletedCount=30、FailedCount=0；未重复执行删除。
- 后置未完成：完整五分类 validation、TR 缺失/异常轨迹线上输出契约、availability-aware 路由接入现有线上调用链、四态融合评估；这些不阻塞 RD-only 全量训练命令交付，全量训练尚未启动。

### 9.6 本轮修复后核验

- `tests/test_rd_physical_five_contract.py tests/test_rd_tr_fusion.py tests/test_rd_tr_entry_contract.py tests/test_rd_physical_window_dataset.py tests/test_rd_physical_artifacts.py`：最终 `29 passed`，新增窗口 causal、pairing-ineligible、parent/group split 隔离断言。
- `python -m compileall -q`（五分类配置、模型、artifact、trainer、dataset、evaluator、测试）通过；训练/evaluator `--help` 均显示真实参数，仅支持 `--config` 或 `--checkpoint/--output/--split/--device/--max-batches`。
- 中间 smoke artifact=`artifacts/20260803-022025/rd-only-five-stage021-smoke-seed42`；最终 finite/axis 修复后的 smoke artifact=`artifacts/20260803-024607/rd-only-five-stage021-smoke-final-20260803-024559/best_rd_physical_five_model.pt`；训练1 epoch、2 train batches、2 val batches；`test_split_loaded=false`、`tr_inputs_loaded=false`、`test_evaluation_executed=false`。
- 最终 evaluator=`tmp/20260803-rd-five-stage021-smoke-final-final.json`；仅4/2710 val windows，`full_split_evaluated=false`，排除2706，`deployment_head=flat`；flat/cascade 均有输出，结果仅作链路证据，不作性能结论。
- smoke best selection 显示 `drone_bird_non_degradation=false`（样本不足且指标为0）；该 restricted smoke 不作性能结论，不阻塞已通过的训练前置 readiness。
- TR-only 事实核验：入口为 `src/core/train_expert1.py -> train_expert1_v1`；pure-TR 配置 `stage1_traj_only_epochs=epochs`、`enable_reject_in_train/eval=false`、`run_test_evaluation=false`；dataset 以 `load_rd=False` 构建 train/val/test，且权重函数在 `pure_tr_only` 时不读取 RD availability。pure-TR checkpoint `artifacts/20260802-232731/20260802-tronly-v10r1-expert1-pure-tr-seed42/best_model.pt` labels 与 `LOW_ALTITUDE_CLASSES` 顺序一致，`selection_eligible=true`、`test_used_for_selection=false`；其 validation summary Macro-F1=70.58%，不是 RD 能力证据，也不是融合达标证据。新增只读契约测试 `tests/test_rd_tr_entry_contract.py`，未修改 TR-only 文件。
- 旧二分类 RD checkpoint strict load：labels=`DroneTarget,BirdTarget`、quality_dim=8、supports_rejection=false；五分类 quality_dim=7/labels=5，未执行未经证明的 warm-start。
- 新增 `src/core/rd_tr_fusion.py` 仅提供独立 availability-aware 四态适配器；双缺失或混合 batch 的 `logits=None`，每行由 `logits_available` 标识，消费者必须先处理 abstain，`require_all_logits()` 会拒绝不可用行。`tests/test_rd_tr_fusion.py`、TR-only 只读契约、causal/split 断言和五分类/旧 RD 回归合计 `29 passed`。该适配器尚未接入 `LowAltitudeSuspectExpert` 的现有线上 forward，因此不能宣称线上融合已完成。

### 9.7 Stage 022 前置 readiness 核验

- full config：`config/rd_only/rd_physical_five_full_seed42.json`，SHA-256=`36552bb71eabfe6a0f77d93c4ff1b56ad6e328967317f68872011d64039a3355`；`PhysicalRDFiveConfig` 实载通过，`epochs=18`、`batch_size=4`、`lr=1e-4`、`seed=42`、`device=cuda`、`run_test_evaluation=false`、`output_dir=artifacts/rd-only-five-stage022-full-seed42`。
- 训练入口 `python -m src.core.train_rd_physical_five --config ...` 和 validation 入口 `python -m src.core.evaluate_rd_physical_five --checkpoint ... --output ... --split val` 的实际 `--help` 通过；训练只加载 train/val，test 被配置和代码双重拒绝。
- 最终 smoke artifact=`artifacts/20260803-024607/rd-only-five-stage021-smoke-final-20260803-024559/best_rd_physical_five_model.pt`；最终 smoke report=`tmp/20260803-rd-five-stage021-smoke-final-final.json`；1 epoch、2 train batches、2 val batches、4/2710 validation windows，`test_split_loaded=false`、`tr_inputs_loaded=false`、`test_evaluation_executed=false`。不把该 smoke 当五类性能指标。
- 训练前置状态：**待全量训练：是**。这是代码/配置/数据契约/smoke 层面的 readiness，不代表五类指标已达到90%，也不代表融合或拒识已验证。
- 用户手动前台命令（本轮仅做 scriptblock 语法解析，未启动）：

```powershell
Set-Location F:\Project\RadarInsight
$logDir = 'artifacts/rd-only-five-stage022-full-seed42-launch'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
& .\.venv\Scripts\python.exe -m src.core.train_rd_physical_five --config .\config\rd_only\rd_physical_five_full_seed42.json 2>&1 | Tee-Object -FilePath (Join-Path $logDir 'train.log')
```

- 用户手动后台命令（本轮仅做 scriptblock 语法解析，未启动）：

```powershell
Set-Location F:\Project\RadarInsight
$logDir = 'artifacts/rd-only-five-stage022-full-seed42-launch'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$pythonPath = (Resolve-Path '.\.venv\Scripts\python.exe').Path
$projectPath = (Get-Location).Path
$trainProcess = Start-Process -WindowStyle Hidden -FilePath $pythonPath -WorkingDirectory $projectPath -ArgumentList @('-m','src.core.train_rd_physical_five','--config','config/rd_only/rd_physical_five_full_seed42.json') -RedirectStandardOutput (Join-Path $logDir 'stdout.log') -RedirectStandardError (Join-Path $logDir 'stderr.log') -PassThru
$trainProcess.Id | Set-Content (Join-Path $logDir 'pid.txt')
```

- 日志查看：`Get-Content .\artifacts\rd-only-five-stage022-full-seed42-launch\stdout.log -Wait`；进程检查：`$trainPid = [int](Get-Content .\artifacts\rd-only-five-stage022-full-seed42-launch\pid.txt); Get-Process -Id $trainPid -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,StartTime,CPU`。
- 训练完成后的 validation 命令会按真实 timestamp 目录自动定位唯一 full checkpoint，不使用未定义占位符：

```powershell
$checkpointItem = Get-ChildItem .\artifacts -Directory | ForEach-Object { $candidate = Join-Path $_.FullName 'rd-only-five-stage022-full-seed42\best_rd_physical_five_model.pt'; if (Test-Path -LiteralPath $candidate) { Get-Item -LiteralPath $candidate } } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $checkpointItem) { throw '未找到五分类 full checkpoint' }
& .\.venv\Scripts\python.exe -m src.core.evaluate_rd_physical_five --checkpoint $checkpointItem.FullName --output .\tmp\rd-five-stage022-full-validation.json --split val --device cuda --max-batches 0
```

- 后台 stdout/stderr 写入 `artifacts/rd-only-five-stage022-full-seed42-launch/{stdout.log,stderr.log}`，PID 写入 `pid.txt`；训练完成后只允许 `--split val --max-batches 0`，不执行 test。
- 未执行（历史快照）：该记录产生时五分类全量训练、完整 validation 尚未执行；随后用户手动完成训练和独立完整 validation。至今仍未执行 test、融合权重拟合、四态真实样本评估、线上 stable 接入；全量训练由用户手动执行。

## 11. Stage 024–031 独立 ParentGate、DBB/CU 专科执行记录（已完成 smoke，待用户全量训练）

### 11.1 本次只读核对

- 已重新读取 `task_plan.md`、`process.md`、`findings.md` 全部内容，并检查工作区状态；现有用户修改保留，不对无关文件做回滚或重构。
- 已核对当前五分类模型源码：`parent_head` 与 `child_heads` 和 flat head 共享同一 embedding；`deployment_head="flat"`，所以现有第一层不是独立父组模型，也不是当前部署路由器。
- 已核对当前训练目标为 flat CE + 0.25 cascade CE + 0.15 parent CE + 0.10 prefix CE + 0.10 micro CE；父组头是辅助任务，未有独立 optimizer/checkpoint/evaluator。
- 已核对 Stage 023 validation：flat 五分类 2710 windows，Accuracy=93.17%、Macro-F1=84.15%；Bird Recall=82.22%、Clutter Recall=58.77%、Unknown Recall=74.21%；当前候选不满足交付门槛。
- 已由当前混淆矩阵折叠得到 DBB/CU 代理边界：`[[2402,35],[54,219]]`，但该数值只是 flat 结果折叠，不能当作独立 ParentGate 指标。

### 11.2 已冻结的设计原则

- 三个独立 RD-only 模型：ParentGate（DBB/CU）、DBB 专科（Drone/Bird/Balloon）、CU 专科（Clutter/Unknown）。独立指参数、配置、优化器、checkpoint、报告独立；可复用相同 RD 数据读取和编码代码，但禁止隐式加载当前五分类权重。
- 组合默认采用 soft probability product，不采用不可逆的 hard argmax 路由；完整五类概率由父组概率乘以条件专科概率得到，最后保留 flat 模型作为安全回退。
- 三模型都只接收 RD tensor、RD 物理轴、RD 时间、mask、RD quality/statistics/micro-Doppler；父组 target 仅由五类标签在 loss/evaluation 侧映射，不进入模型输入。
- 专科模型必须分别报告局部指标，同时重新组合后报告完整五类指标；任何局部高分不能代替五类部署结果。
- TR 入口继续保留，但 RD-only 训练/评估不读 TR；在 TR checkpoint 契约和四态真实评估完成前，不接入线上 late-fusion，也不修改 TR-only 代码或跟踪文件。

### 11.3 实际执行顺序与结果

1. Stage 024：读取旧五分类代码和 full summary，冻结五类/父组/专科标签、`[B,T,6,21,384]` 输入、7 维 quality、2/3/2 输出、独立 checkpoint schema。
2. Stage 025：只读现有 physical index 和 train-only 五类 stats；未发现 parent/content-group 跨 split、future window、fusion-eligible 或 TR 字段进入 forward，因此不需要重建。
3. Stage 026：实现 ParentGate dataset remap、独立模型/artifact/trainer/evaluator；运行 1 epoch、2 train/2 val batches smoke，生成独立 checkpoint 并 strict reload。
4. Stage 027：实现 DBB 三分类与 CU 二分类专科；分别运行同等 smoke，生成独立 checkpoint；局部 smoke 指标仅 4 windows，不作性能结论。
5. Stage 028：实现 soft probability product、oracle/hard 诊断和 flat fallback；测试概率归一、缺失 specialist 和不使用全零哨兵。
6. Stage 029：该条为历史快照；完整组合 validation 已在 Stage 036 完成，OOF calibration 仍未执行；不读取 test。
7. Stage 030：确认保留 TR 入口但 specialist trainer/evaluator 的 `tr_inputs_loaded=false`；不修改 TR-only 代码/跟踪文件，不提前接线上 late fusion。
8. Stage 031：运行 compileall、入口 `--help`、契约测试和相关回归；固定三套 full config，输出人工训练命令。本代理未启动 full training，随后由用户完成 Stage 035。

### 11.4 当前交付点

- 三套代码、配置、数据视图、smoke checkpoint 和组合器已创建；没有执行数据重建、删除、test 评估、OOF 校准或全量训练。
- readiness 仅表示“可开始三套全量训练”：五类 flat baseline 仍为 Stage 023 validation Accuracy=93.17%、Macro-F1=84.15%、Bird Recall=82.22%、Clutter Recall=58.77%、Unknown Recall=74.21%；新 specialist 未有全量指标。
- 本轮对 TR-only 代码和 `TRonly*.md` 零修改；保留 TR 入口由配置的 `tr_entry_preserved=true/allow_tr_inputs=false` 和无 TR contract test 证明。

### 11.5 最终 smoke 复核（权重契约修复后）

- specialist trainer 已补齐局部 class/group/parent/evidence 权重，来源仅为各自 train split；配置新增 `prefix_max_loss_weight=0.5`、`evidence_target_frames=24`，没有读取 test 或改变评估分母。
- 最终 schema 变更后重新运行三套 smoke：ParentGate 输出目录 `artifacts/20260803-232631/rd-specialist-parent-gate-smoke-seed42`，DBB=`artifacts/20260803-232700/rd-specialist-dbb-smoke-seed42`，CU=`artifacts/20260803-232709/rd-specialist-cu-smoke-seed42`；三份 `training_progress.json` 均 `status=completed`，均 test=false、tr_inputs_loaded=false。
- 三次 evaluator strict reload 均通过；每次 `--max-batches 2` 仅评估 4 个 val windows，JSON 明确 `full_split_evaluated=false`、`scope=restricted ... no performance claim`，因此 smoke 的 0/50/75% 等受限数字不进入能力结论。
- 修复后回归仍为 23 passed；compileall、配置 dataclass 实载、训练/evaluator `--help` 均通过。

## 14. Stage 035–036 full training 后验证执行记录（2026-08-05）

1. 只读定位四个用户 full artifact：flat `artifacts/20260804-114842/rd-only-five-stage034-full-seed42`，ParentGate `artifacts/20260804-184925/rd-specialist-parent-gate-stage034-full-seed42`，DBB `artifacts/20260804-231455/rd-specialist-dbb-stage034-full-seed42`，CU `artifacts/20260805-170322/rd-specialist-cu-stage034-full-seed42`。摘要均为 completed，test=false、TR inputs=false；记录 best epoch 和 SHA-256 后再开始 evaluator。
2. 分别运行 validation-only evaluator：四个入口均固定 `--split val --max-batches 0 --device cuda`；flat 覆盖2710窗口，ParentGate覆盖2710，DBB覆盖2437，CU覆盖273；生成 `tmp/20260805-rd-*-stage034-full-val.json` 及预测 CSV。
3. 发现局部 specialist 数据视图不同，未直接拼接局部指标。新增只读 `src/core/evaluate_rd_physical_hierarchy.py`，在同一2710个 physical RD validation windows 上重新前向 flat/ParentGate/DBB/CU，调用 `compose_rd_specialists` 计算 predicted-soft、hard-routing，调用 `compose_with_oracle_parent` 计算 oracle-parent；flat/cascade/fallback 也同窗输出。
4. hierarchy smoke 使用 `--max-batches 1` 先验证输出 schema、概率和为1、缺失状态不伪造；随后 `--max-batches 0` 完整执行，报告 `full_split_evaluated=true`、excluded=0、test=false、tr_inputs_loaded=false。结果写入 `tmp/20260805-rd-hierarchy-stage034-full-val.json` 和预测 CSV。
5. 评估结论：flat window Macro-F1=83.55%；predicted-soft=83.80%；hard=83.72%；oracle-parent=89.79%（诊断-only）；flat fallback=83.55%。CU 组内 Clutter/Unknown 仍为主要瓶颈，未达到90%，不接线上。
6. 运行 `pytest`（逐文件收集 `tests/test_rd*.py`）得到94 passed、1 warning；`compileall -q src/core src/process config` 和新增 evaluator `py_compile` 通过。未运行 test、OOF 校准、阈值/融合权重选择或全量训练。
7. 覆盖式更新 `task_plan.md`、`findings.md`、`process.md`，将 Stage 035 标为用户已完成、Stage 036 标为 validation 已完成，新增 Stage 037 后续单因素实验阻塞。未修改 TR-only 跟踪文件和代码。
8. 读取四份 `training_history.json` 的 best epoch 训练/验证摘要：flat train/val Macro-F1=99.31%/83.55%，CU=90.97%/79.53%；仅记录为泛化风险证据，不把它直接写成唯一根因。

## 15. Stage 037 标签语义、parent 一致性与聚合契约执行记录（2026-08-05）

1. 新增只读审计脚本 `src/process/audit_rd_five_label_semantics.py`。脚本只读取 `cq08_unified_manifest.json`、`cq08_grouped_split.json`、physical canonical frame index 和 causal window index 的 train/val 元数据，不读取 RD tensor、test tensor、模型 prediction 或 test metric。
2. 实际执行命令：`.\\.venv\\Scripts\\python.exe -m src.process.audit_rd_five_label_semantics --project-root F:\\Project\\RadarInsight --output tmp/20260805-rd-five-label-semantics-audit.json`，退出码 0，输出 schema=`rd_five_label_semantics_audit_v1`。
3. manifest 语义核验：1550 records；Clutter=50、label_id=5、MAT 文本统一 `Clutter`；Unknown=100、label_id=6、MAT 文本统一 `Other`；路径/文件名 ID mismatch=0；train/val frame label 与 manifest label mismatch=0。
4. parent/split 核验：train/val frame metadata=89997、parents=1316，multi-label/track/source parent=0；window parent label/split/content-group conflict=0；train/val group intersection=0、content-group intersection=0。
5. train/val 物理 metadata 统计：range/velocity coverage 两类及五类均为 1.0；Clutter range reference median=1717.5m、Unknown=1132.5m，存在重叠；Clutter observation duration median=0.02736s、Unknown=0.01728s。该统计仅用于审计，不用于调参或能力结论。
6. 结果写入 `tmp/20260805-rd-five-label-semantics-audit.json`，并覆盖更新 `task_plan.md`、`findings.md`；未修改 TR-only 代码或 TR-only 跟踪文件，未重标、未重建数据、未启动训练。
7. 阶段结论：没有已证实的 Clutter/Unknown 标签映射、parent 混标或 train/val 泄露缺陷；`Other` 不能直接解释为 OOD/reject 标签。Stage 037 剩余 CU/Bird 单因素实验、OOF calibration、生产 stable/fusion 接入仍未执行。
8. 发现旧 RD-only flat/specialist trainer/evaluator 使用 mean-logit parent 聚合，而 hierarchy evaluator 使用 mean-probability 聚合；这只影响 parent 报告/选择口径，不影响 window 指标。新增 `src/core/rd_physical_parent_aggregation.py`，并将五分类与 specialist trainer/evaluator 统一为 arithmetic mean of per-window probabilities, then argmax；新增两项契约测试。
9. 聚合修复后的 full validation-only 回归：flat 2710 windows Accuracy=93.62%、Macro-F1=83.55%，parent 221 Accuracy=94.57%、Macro-F1=85.31%；ParentGate 2710/221 window/parent Macro-F1=88.28%/91.34%；DBB 2437/198=96.63%/98.72%；CU 273/23=79.53%/80.83%。window 指标与 Stage 036 一致，test=false、TR=false；旧 mean-logit parent 数字不再与新口径横向比较。
10. 回归命令：`.\\.venv\\Scripts\\python.exe -m pytest -q (Get-ChildItem tests -Filter 'test_rd*.py' | ForEach-Object { $_.FullName })`，结果 `96 passed, 1 warning`；`.\\.venv\\Scripts\\python.exe -m compileall -q src/core src/process config`，exit 0。
11. 在同一 full validation predictions CSV 上做只读分层：prefix 485 windows Macro-F1=79.12%，non-prefix 2225=84.21%；frame_count>=24 的702 windows Macro-F1=68.21%，<24 的2008=84.85%。未删除/排除任何样本，未修改总体分母；该结果仅作为时间长度/parent 异质性候选。
12. 为后续单因素实验创建两个独立 smoke config：Bird weight=1.35（唯一主变因）和 parent_loss_weight=0（唯一主变因），其余 seed/device/index/stats/test-off 固定；实际运行两条 `train_rd_physical_five --config` 命令，均 exit 0，artifact `artifacts/20260805-191120/rd-only-five-stage037-bird-weight-smoke-seed42` 与 `...parent-loss0-smoke-seed42`，summary 明确 train/val dataset loaded、test_split_loaded=false、tr_inputs_loaded=false、loss/gradient finite。
13. 对两个 smoke checkpoint 运行 validation evaluator `--max-batches 2`，均报告 `full_split_evaluated=false`、evaluated=4、excluded=2706；受限类别分布不用于性能比较。未启动 A/B full training、未读取 test、未调阈值。
14. 创建两份人工可执行 full config：`config/rd_only/rd_physical_five_bird_weight_stage037_full_seed42.json` 与 `config/rd_only/rd_physical_five_parent_loss0_stage037_full_seed42.json`。实载命令通过，输出目录 `artifacts/rd-only-five-stage037-bird-weight-full-seed42` 和 `artifacts/rd-only-five-stage037-parent-loss0-full-seed42` 尚不存在；本轮没有启动它们。
15. hierarchy smoke 回归：使用四个已存在 full checkpoint、`--split val --max-batches 1`，exit 0，strict reload、概率和、flat/predicted-soft/hard/oracle/fallback 输出均通过；报告 `tmp/20260805-rd-hierarchy-stage037-smoke.json` 明确 evaluated=4、full_split_evaluated=false、test=false、tr_inputs_loaded=false，仅作契约 smoke 不作性能结论。
16. 更新 `findings.md` 为覆盖式 Stage 037 问题清单：已确认性能未达标、层级组合不达标、历史 parent 口径差异（已修复新路径）、Unknown 非拒识、生产 caller 未接入、CU 低 parent support/泛化风险及未归因的多目标训练；没有把推测写成已确认根因。

## 12. Stage 032 全链路只读审查（历史记录，后续已批准修复）

- 读取范围覆盖项目指令、工作区状态、RD-only 三跟踪文件、有效配置、实际 train/eval 入口、物理 canonical/frame/window index、manifest、normalization/feature stats、旧二分类/五分类/specialist/legacy RD 分支、`experts.py`、`system.py` 和 `rd_tr_fusion.py`；未修改 TR-only 代码或 `TRonly*.md`。
- 当前 workspace 已有用户修改和大量 untracked RD/TR 资产；本轮没有 reset/checkout/删除/重构。
- 物理 index 只读审计结果：22085 rows、17797 trainable；parent/content-group/tensor path cross-split=0；future frame=0；fusion-eligible=0；source=`cq08_track`；session columns absent；time sync 全为 `canonical_parent_association`，time semantics 全为 `canonical_sensor_relative_elapsed`。
- 五类 full checkpoint strict reload smoke：epoch=16、SHA-256=`168068014FEC472D31477FC1167C9A927FC5C88EACEBB2AB14F0B85A8AB3AC43`；`--max-batches 1` 只评估4个 val windows，报告明确 incomplete/no performance claim。三 specialist smoke checkpoint strict reload 同样只做受限 windows，未作能力结论。
- 只读测试：23 项 RD/fusion/entry/artifact tests passed；`compileall -q src/core src/process` passed；训练和评估入口 `--help` 真实输出通过。系统环境的全局 Python 无 torch/pytest，本轮统一使用项目 `\.venv\Scripts\python.exe`。
- 代码接线结论：`route_rd_tr_five`、`compose_rd_specialists` 当前只有定义和测试引用，未被 `System/LowAltitudeSuspectExpert` 生产 forward 调用；现有 online stable/reject 路径属于 Expert1/TR+RD，不是 physical RD-only 能力。
- Stage 032 记录的“等待修复确认”已由用户后续“开始修复”指令覆盖；修复已按 Stage 033–034 完成。Stage 035 full training 和 Stage 036 validation/组合已完成；当前仅保留 OOF 校准、线上 physical RD stable/fusion 接入和后续单因素实验，未输出未经验证的性能或拒识结论。

## 10. Stage 023 五分类全量训练后审查记录（历史）

### 10.1 产物和边界

- 用户手动完成训练，checkpoint=`artifacts/20260803-112945/rd-only-five-stage022-full-seed42/best_rd_physical_five_model.pt`，best epoch=16/18；checkpoint SHA-256=`168068014FEC472D31477FC1167C9A927FC5C88EACEBB2AB14F0B85A8AB3AC43`。
- `training_summary.json` 明确 `test_evaluation_executed=false`、`tr_inputs_loaded=false`、`supports_rejection=false`；配置标签顺序严格为 `DroneTarget,BirdTarget,BalloonTarget,ClutterTarget,UnknownTarget`，`deployment_head=flat`。
- 独立 evaluator 报告=`tmp/20260803-rd-five-stage022-full-validation.json`，预测=`tmp/20260803-rd-five-stage022-full-validation.predictions.csv`；`full_split_evaluated=true`、2710/2710 windows、excluded=0，test 未读取。

### 10.2 Flat validation 结果（均为 validation，不是 test）

| 层级 | N | Accuracy | Macro-P | Macro-R | Macro-F1 | Balanced Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| window | 2710 | 93.17% | 86.00% | 82.96% | 84.15% | 82.96% |
| parent mean-logit | 221 | 93.67% | 85.83% | 82.74% | 84.01% | 82.74% |

| window 类别 | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| DroneTarget | 1269 | 97.99% | 99.76% | 98.87% |
| BirdTarget | 523 | 91.68% | 82.22% | 86.69% |
| BalloonTarget | 645 | 92.66% | 99.84% | 96.12% |
| ClutterTarget | 114 | 77.01% | 58.77% | 66.67% |
| UnknownTarget | 159 | 70.66% | 74.21% | 72.39% |

- flat confusion（行真值、列预测）：`[[1266,3,0,0,0],[24,430,34,10,25],[1,0,644,0,0],[1,16,6,67,24],[0,20,11,10,118]]`。
- 主要方向：Bird→Drone=24、Bird→Balloon=34、Bird→Clutter=10、Bird→Unknown=25；Clutter→Bird=16、Clutter→Unknown=24；Unknown→Bird=20、Unknown→Clutter=10、Unknown→Balloon=11。
- source 只有 `cq08_track`；session unavailable（physical index 无 session 字段）；online stable unavailable（没有生产 stable state machine）。parent Clutter support=8、Unknown support=15，均标记 low stability。

### 10.3 深度诊断

- Bird 旧二分类 validation recall=96.56%，五类 flat recall=82.22%，下降14.34个百分点；Bird F1=86.69%，下降9.87个百分点。五分类中 Drone recall=99.76%，因此不是 Drone 牺牲换 Bird。
- 仅对预测概率中的 Drone/Bird 两列做 pairwise 比较，Bird recall=95.41%（499/523），Drone recall=99.76%（1266/1269）；该数字只说明其他三类竞争造成的分流，不能替代五分类部署指标。
- 预测错误高度自信：Bird→Drone 平均最大概率95.57%，Bird→Balloon 96.22%，Bird→Unknown 96.78%；Clutter→Bird 98.28%，Clutter→Unknown 96.84%；Unknown→Clutter 99.84%，Unknown→Balloon 99.94%。因此不能用单一 confidence threshold 解释。
- 15-bin validation ECE（诊断，不做校准）：overall约0.0630，Bird约0.1652，Clutter约0.3850，Unknown约0.2512；当前没有应用 calibration 或 threshold。
- 错误集中在少数 parent：Bird 93 个错误来自13/43 parents；Clutter 47 个错误来自5/8 parents（其中 parent `cq08|track|4198` 为19/19）；Unknown 41 个错误来自7/15 parents。说明至少存在 parent/场景异质性，不能只按窗口数量推断。
- 当前五类 loss 在 `src/core/rd_physical_five_trainer.py:297` 为 flat CE +0.25 cascade CE +0.15 parent CE +0.10 prefix CE +0.10 micro CE；`config/rd_only/rd_physical_five_full_seed42.json:23` 的 class_weights 全为1。旧二分类的 Bird weight=1.35 未保留。
- 训练权重审计：代码 `src/core/rd_physical_five_trainer.py:58-85` 先按 class group/parent/evidence 平衡。按 train index 计算平均有效窗口权重：Drone=0.0625、Balloon=0.0993、Bird=0.1246、Unknown=0.7016、Clutter=1.0597；这是高可信机制证据，但尚未由单因素实验证明为唯一根因。
- 18 epoch 的 train/val loss 和梯度有限，但 pre-clip gradient norm 最大约2273.23，val loss 在 epoch 2 后波动，属于优化稳定性风险，不单独定性为类别退化根因。

### 10.4 标签语义与拒识边界

- `src/process/cq08_manifest.py:19-26` 将 CQ-08 `label_id=5` 映射为 `ClutterTarget`，`label_id=6` 映射为 `UnknownTarget`；这是原始数据标签，不是模型后处理生成的 reject。
- `src/process/standardize_tracks.py:28-47` 的中文映射也把“杂波”映射为 ClutterTarget、“待定”映射为 UnknownTarget；当前 physical 五类链路使用 CQ-08 manifest 的 label_name，不读取 TR label。
- evaluator 明确 `supports_rejection=false`、`unknown_target_is_class=true`、`abstain_output=false`、threshold=null；不能把 UnknownTarget 召回率当作拒识 Recall。

### 10.5 训练选择和后续阻塞

- flat 比 cascade 好：window Macro-F1 84.15% vs 83.71%，Bird Recall 82.22% vs 81.26%；继续保持 flat deployment head。
- `best_selection.eligible=false`，Drone/Bird 相对 Stage 011 二分类参考的 P/R/F1 非劣化检查均未通过；当前 checkpoint 不应标记为五分类达标或待部署。
- 后续实验必须一次只改一个主要因素：A 仅恢复 Bird class weight=1.35；B 仅消融/降低 parent loss；C 仅调整 parent/evidence 权重；另行做 label provenance/parent 内部一致性审计。所有实验只用 train/val，禁止 test 调参、删除困难样本或改变分母。

## 16. Stage 038 V3/V4 full validation 与 Bird/Clutter 低召回归因（2026-08-06）

1. 核实 evaluator：`.\.venv\Scripts\python.exe -m src.core.evaluate_rd_physical_five --help`；本阶段只执行 `--split val --max-batches 0`，未读取 test。
2. V3 严格 reload：`artifacts/20260805-193521/rd-only-five-stage037-bird-weight-full-seed42/best_rd_physical_five_model.pt`，输出 `tmp/20260806-rd-five-stage037-v3-full-val.json`/predictions CSV；2710/2710 windows、excluded=0、test=false、TR=false。
3. V4 严格 reload：`artifacts/20260806-142225/rd-only-five-stage037-parent-loss0-full-seed42/best_rd_physical_five_model.pt`，输出 `tmp/20260806-rd-five-stage037-v4-full-val.json`/predictions CSV；2710/2710 windows、excluded=0、test=false、TR=false。
4. 配置实载：V3 Bird class weight=1.35、parent loss=0.15、best epoch=9；V4 全类权重=1、parent loss=0、best epoch=12；其余 18 epochs、batch=4、lr=1e-4、seed=42、CUDA、同 index/stats、flat head 固定。V3 与 V4 互相不能作单因素因果比较，Stage 034 才是两者共同参照。
5. V3：Accuracy=93.28%、Macro-F1=84.01%；Bird P/R/F1=90.89%/83.94%/87.28%，Clutter=80.25%/57.02%/66.67%，Unknown=67.82%/74.21%/70.87%。V4：Accuracy=92.47%、Macro-F1=81.83%；Bird=89.07%/82.60%/85.71%，Clutter=76.47%/57.02%/65.33%，Unknown=63.29%/62.89%/63.09%。
6. Stage 034 参照 Bird/Clutter/Unknown Recall=86.23%/50.88%/72.33%；故 V3 Bird weight 未提升 Bird Recall（下降2.29个百分点），Clutter 上升6.14个百分点；V4 去 parent loss 后 Bird/Unknown 下降，Clutter 恰为同一 65/114=57.02%，不能把共同值归因于一个开关。
7. 混淆方向：V3 Bird→Drone/Balloon/Clutter/Unknown=15/29/9/31，Clutter→Bird/Unknown=18/25，Unknown→Bird/Clutter=21/6；V4 为19/36/6/30、16/27、34/14。错误是多类竞争，不是单一 Bird↔Drone。
8. 错误集中：val Bird=523/43 parents、Clutter=114/8、Unknown=159/15；V3 Clutter 49个错误来自5个 parent且前5个覆盖100%，`cq08|track|4198` 为19/19；V4 Clutter 49个仍集中，4198/2251/2901 全错。Bird V3/V4 错误来自13/15个 parent，Unknown来自7/9个 parent。
9. 时间分层：V3 Bird Recall `<24`/`>=24`=86.58%/68.42%，Clutter=64.36%/0.00%；V4 Bird=85.46%/65.79%，Clutter=64.36%/0.00%；prefix Recall 低于 non-prefix。保持完整分母，仅作为时间/场景异质性证据。
10. 代码核对：`src/core/rd_physical_five_trainer.py:58-82,258-274` loss 为 `flat CE +0.25*cascade CE + parent_loss_weight*parent CE +0.10*prefix CE +0.10*micro CE`，evidence 只用 prefix/frame_count/presence；V4 cascade CE 仍通过 `src/core/rd_physical_five_model.py:225-235` 使用 parent log-prob，因此不是完全去除父组约束。Bird class weight 会同时作用 flat/cascade/prefix/micro。
11. train 有效权重（evidence/presence 后）：Drone=4867.40、Balloon=2844.35、Bird=2397.46、Unknown=564.44、Clutter=336.50；有效份额=44.21%/25.83%/21.78%/5.13%/3.06%。CU 在共享模型中获得的有效梯度很少，这是机制证据而非“提高权重必然有效”的证明。
12. V3/V4 错误最大概率均很高（Bird 错误约95.84%/96.06%，Clutter约94.35%/89.20%），不是单纯低置信度问题；当前 `supports_rejection=false`、UnknownTarget为普通类别，不能用拒识阈值掩盖。
13. 训练/验证泛化：V3/V4 train Macro-F1 接近98–99%，val仅84.01%/81.83%，与少数 parent 集中共同支持过拟合/场景异质性风险；不能据此断言唯一根因。标签审计仍为 Clutter=`Clutter`、Unknown=`Other`、frame mismatch=0、parent/split/content冲突=0，不重标、不删样本、不重建数据。
14. 阶段结论：Bird 低召回由五分类竞争（尤其 Balloon/Unknown/Drone）与高置信度泛化错误叠加；Clutter 低召回主要由 CU 有效样本/parent 极少、Clutter↔Unknown/Bird 边界和少数场景集中造成。V3/V4 不能证明某一损失开关是唯一原因；CU 专项目标/采样、Bird/DBB 专项与 OOF 校准必须另做单因素 train/val 实验，本阶段不启动新 full training。

## 17. Stage 039 ParentGate 第一层路由审查（2026-08-06）

1. 先检查 V3/V4 的实际部署头：两份配置均为 `deployment_head="flat"`。`src/core/evaluate_rd_physical_five.py` 评估的 selected logits 是 flat logits，因此 V3/V4 中 Bird→Clutter/Unknown 不是第一层 ParentGate 造成，而是五分类 flat head 直接输出。
2. 使用实际入口 `.\.venv\Scripts\python.exe -m src.core.evaluate_rd_physical_hierarchy`，固定同一 validation index/stats 和已完成的 ParentGate、DBB、CU checkpoint，对 V3/V4 分别完整复算；报告为 `tmp/20260806-rd-hierarchy-v3-full-val.json`、`tmp/20260806-rd-hierarchy-v4-full-val.json`，均 2710/2710、excluded=0、`test_evaluation_executed=false`、`tr_inputs_loaded=false`。
3. 代码契约核验：`src/core/data/rd_physical_specialist_dataset.py:17-42` 的 remap 为 Drone/Bird/Balloon→0(DBBGroup)、Clutter/Unknown→1(CUGroup)；`src/core/rd_physical_specialist_model.py:41-56` ParentGate 只暴露 `parent_logits`；checkpoint metadata 显示 ParentGate head=(2,1664)、DBB=(3,1664)、CU=(2,1664)，无全局五类切片错误。
4. 组合器核验：`src/core/rd_physical_hierarchical_composer.py:53-76` predicted-soft 使用 `P(DBB)*P(local_DBB)`、`P(CU)*P(local_CU)`；hard-routing 使用 ParentGate argmax 选择对应局部 head；`src/core/evaluate_rd_physical_hierarchy.py:185-188` 的 parent labels 仅用于 oracle 诊断。22 项专科/TR-fusion/ParentGate 契约测试通过，compileall 通过；未发现 Bird→CU 的标签索引或概率拼接代码缺陷。
5. ParentGate 实际 full val：window Accuracy=96.05%、Macro-F1=88.28%；DBBGroup P/R/F1=96.98%/98.69%/97.82%，CUGroup=86.09%/72.53%/78.73%；混淆矩阵行真值列预测 `[[2405,32],[75,198]]`。因此 DBB→CU=32/2437，CU→DBB=75/273，第一层 CU Recall 明显不足。
6. 类别级路由：Bird 被错路由到 CU=26/523（Bird ParentGate Recall=95.03%）；Clutter 被错路由到 DBB=35/114（Recall=69.30%）；Unknown 被错路由到 DBB=40/159（Recall=74.84%）。这正是级联中出现 Bird→Clutter/Unknown、Clutter/Unknown→Bird/Balloon/Drone 的来源。
7. 条件局部性能：ParentGate 路由正确时，DBB specialist Bird Recall=440/497=88.53%，Drone=1259/1263=99.68%，Balloon=642/645=99.53%；CU specialist Clutter=69/79=87.34%，Unknown=106/119=89.08%。因此层级问题同时包含 ParentGate 路由错误和 DBB Bird 局部模型不足；不能只修第一层。
8. hierarchy 结果（V3/V4 与 flat 输入无关，专科 checkpoint 相同）：predicted-soft window Bird Recall=84.32%、Clutter=60.53%、Unknown=66.67%；oracle-parent（真值父组，仅诊断）Bird=89.10%、Clutter=69.30%、Unknown=88.68%。oracle 与 predicted-soft 的差距量化了路由错误影响，但 oracle 不可部署。
9. ParentGate 配置/训练审查：`config/rd_only/rd_specialist_parent_gate_full_seed42.json` 使用 `class_weights=[1,1]`，无 sampler；trainer `src/core/rd_physical_specialist_trainer.py:158-172` 仅做普通 weighted CE+prefix/micro auxiliary loss；有效 evidence 权重 train DBB=10109.21、CU=900.94（约11.22:1）。validation support DBB=2437、CU=273，训练/验证组别不平衡会使普通 CE 更偏向 DBB。
10. checkpoint 选择审查：现有 ParentGate best epoch=2、`best_selection.eligible=false`；epoch 2 window CU Recall=72.53%，epoch 18 CU Recall=74.36%，但 epoch 2 被 macro-F1 选择。配置的 reference CU Recall/F1=80.22%/83.11%，现有 18 个 epoch 均未满足该非劣化门槛；没有证据显示当前 best checkpoint 已达到第一层目标。
11. 安全边界：不能通过把 flat 五分类结果称为级联结果来掩盖第一层错误；也不能用 validation 阈值直接改 ParentGate 决策后上线。应先做 ParentGate 单因素 class weight 或 sampler 实验，并把 DBB/CU 最小 Recall 和五类组合非劣化纳入 selection；每次只改一个主要因素，使用新配置、新 artifact、train/val，test=false。

## Stage 040 执行记录：ParentGate CU 权重单因素（2026-08-06）

1. 读取并核对 Stage 039 ParentGate 证据：旧配置 `class_weights=[1,1]`、train effective evidence DBB/CU≈11.22:1、旧 best epoch=2 且 `best_selection.eligible=false`；未修改旧 checkpoint。
2. 代码修复：在 `PhysicalRDSpecialistConfig` 增加 `selection_policy` 校验；在 specialist trainer 选择键增加 `eligible_weak_class_guard`，不改变 loss/forward/数据分母。旧默认 `eligible_macro_f1` 保持兼容。
3. 新建配置：CU weight=5 的 Stage 040 smoke/full 两份 JSON，输出目录分别为 `artifacts/rd-specialist-parent-gate-stage040-cu-weight5-smoke-seed42` 与 `artifacts/rd-specialist-parent-gate-stage040-cu-weight5-full-seed42`；full 目录尚未产生，避免覆盖历史 artifact。
4. 执行 smoke：`.\\.venv\\Scripts\\python.exe -m src.core.train_rd_physical_specialist --config config/rd_only/rd_specialist_parent_gate_cu_weight5_stage040_smoke_seed42.json`。结果 1 epoch、2 train/2 val batches，数据读取与 loss/保存完成；8 个受限 val 样本无 CU support，明确不作性能结论。
5. 执行 strict evaluator：`.\\.venv\\Scripts\\python.exe -m src.core.evaluate_rd_physical_specialist --checkpoint artifacts/20260806-205902/rd-specialist-parent-gate-stage040-cu-weight5-smoke-seed42/best_rd_physical_parent_gate_specialist.pt --output tmp/rd-specialist-parent-stage040-cu-weight5-smoke-eval.json --split val --device cuda --max-batches 2`。结果 `full_split_evaluated=false`、test=false、TR=false。
6. 新增/运行回归：hard-routing 不输出选定父组之外的类别，CU weight=5 确实进入 `_masked_ce`；全部 RD 测试 `99 passed, 1 warning`；compileall 与训练入口 `--help` 通过。
7. 本段前六项是 Stage 040 smoke 时的历史记录；其“full 尚未训练”状态已由用户完成的 `artifacts/20260806-235753/rd-specialist-parent-gate-stage040-cu-weight5-full-seed42` 覆盖，实际 full 验收见下文。

## Stage 040 full 后验收与 Stage 041 修复执行记录（2026-08-07）

1. 读取用户提供的 Stage 040 summary，并核验实际 checkpoint、配置、training_history/progress：epoch=3、SHA-256=`0A8F89EED1EEF408B33BBC7B352C9D9C9E0E49349CA8C9963B60589C536F9806`；train/val windows=12396/2710、test split未加载、TR inputs未加载、test=false。
2. 首次调用完整 evaluator 时工具1秒超时，进程未被工具清理；第二次10分钟工具超时也遗留重复子进程。用 `Win32_Process` 精确确认两组命令均是本次只读 RD validation；仅终止后启动的重复 PID 60012/59600，保留先启动的62056/61260自然完成。未触碰同机用户的 TR-only或`train.tasks`进程。
3. 完整 evaluator最终自然完成，产物=`tmp/20260807-rd-parent-stage040-cu-weight5-full-val.json`及predictions CSV，2710/2710、excluded=0、test=false。Stage 040 DBB/CU confusion=`[[2380,57],[60,213]]`；旧Stage 034=`[[2405,32],[75,198]]`。
4. 训练历史审计：Stage 040 val loss epoch3=0.80405、epoch18=2.88739；CU Recall epoch3达78.02%后反复下降，继续增加epoch不能解释为可靠改进。Stage 040每epoch train gradient norm mean=2.73–16.56、max最高2303.91；旧equal-weight mean=0.89–3.67、max最高581.33。
5. 逐窗口路由比较：Bird→CU 26→40、Balloon→CU 0→12、Clutter→DBB 35→29、Unknown→DBB 40→31。Stage 040通过将边界向CU移动换取CU Recall，明显牺牲Bird/Balloon与CU Precision。
6. 用已有完整 hierarchy predictions 恢复DBB/CU局部条件概率，只替换Stage 040 ParentGate概率并按`P(parent)×P(local)`离线重组；两次PowerShell脚本先后因JSON数组转换与数组除法优先级停止，未写文件、未输出指标。第三次改为逐元素解析并加入row count/概率和断言，成功得到新同窗指标：Accuracy=92.29%、Macro-F1=82.89%、Bird Recall=81.45%、Clutter=63.16%、Unknown=71.07%。旧值为92.88%/83.80%/84.32%/60.53%/66.67%。
7. 执行停止条件：不替换Stage 034 ParentGate，不接线上，不读取test，不用Stage 040 CU Recall单项上升宣称整体改进。
8. 审查`_masked_ce`，确认旧实现的weighted CE分母没有包含target class weight；class weight=5会同时放大含CU batch的整体梯度尺度。修改`src/core/rd_physical_five_trainer.py`，分母改为有效evidence×target-class weight之和；全权重=1时不变。
9. 审查旧reference provenance：`0.978013.../0.802197...`等数字只存在于specialist配置及派生summary，未找到对应完整validation报告。Stage 041改用同入口Stage 034完整val报告、predictions CSV与checkpoint，并固定/启动前校验三份SHA、test=false、全split和指标完全一致。
10. 新增`eligible_reference_guard`：强制non-degradation开启；组级DBB/CU P/R/F1与五个原始类父组路由Recall共11项同时保护；无provenance拒绝。原始五类label仅在validation guard/report使用，不进入模型输入、loss、sampler、阈值或预测后处理。
11. 修复checkpoint语义：新增默认关闭的`require_qualified_checkpoint`保持历史配置的`best_*`行为；Stage 041显式设为true。其所有epoch不合格时只写`candidate_*`，summary=`no_qualified_checkpoint`；只有全部guard通过才写`best_*`与`qualified=true`。旧checkpoint schema保持v2，Stage 034/040 CPU strict reload通过。
12. 新建Stage 041 smoke/full配置；最终full config SHA-256=`249475B8ED7CCC5BA654DC0FCD384944E4B468763B3B2885B1448B95AF4A46CB`，输出experiment name唯一，18epochs、batch4、lr1e-4、seed42、CUDA、`require_qualified_checkpoint=true`、test/TR关闭、max batches=0。
13. 最终Stage 041 smoke命令：`.\.venv\Scripts\python.exe -m src.core.train_rd_physical_specialist --config config\rd_only\rd_specialist_parent_gate_weighted_ce_fix_stage041_smoke_seed42.json`。1epoch、2train/2val batches，loss/gradient finite；artifact=`artifacts/20260807-073648/rd-specialist-parent-gate-stage041-weighted-ce-fix-smoke-seed42`。
14. smoke受限val只有8个DBB windows、CU=0，系统输出`qualified=false`、`best_checkpoint=""`、candidate SHA=`607B665D621CF3ED98148C54FBEB55A823530221FB75F91A7EEAD146F05D091F`。strict evaluator命令使用candidate与`--max-batches 2`，输出`tmp/20260807-rd-parent-stage041-weighted-ce-fix-smoke-eval.json`，full_split=false，不作性能结论。
15. 测试过程中一次`pytest tests\test_rd*.py`因PowerShell不展开通配符而0 tests退出；改为`Get-ChildItem`枚举真实文件后运行。新增强制provenance后，一个旧测试夹具因无provenance按预期被配置拒绝；补齐虚拟格式完整provenance后通过。最终定向16 passed，全RD 105 passed/1既有PyTorch warning，compileall、入口`--help`、`git diff --check`通过。
16. 当前状态：Stage 041代码/配置/数据契约/smoke达到“待全量训练”；本审查未启动18epoch full。整个五分类仍非“待部署”，test/reject/stable/fusion均未执行。
