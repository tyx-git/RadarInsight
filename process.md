# RD-only 全链路审查过程记录

更新时间：2026-08-02（Asia/Shanghai）
范围：仅 RD-only；未修改 `TRonlyTask_plan.md`、`TRonlyProcess.md`、`TRonlyFindings.md`，未启动全量训练、全量数据重建、完整长评估或 test 评估。

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

## 4. 完整 validation 基线（均由用户此前手动完成）

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
- 最终统一 RD 专项集合：**80 passed，1 warning，0 failed，30.06 s**。
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

### 6.6 latency smoke

- 命令限制 `warmup=1`、`sample-count=2`、batch=1，exit 0。
- report：`tmp/20260802-rd-only-stage010-latency-smoke.json`，SHA-256=`cf2b72baa1b434b7a862162c5f809382e9493c9575ccb1eff3b29af2cea79511`。
- 仅 2 个样本：preprocess mean=16.2685 ms、H2D mean=2.2573 ms、forward mean=31.5638 ms、total mean=50.1445 ms。
- 样本太少，不能推断生产 SLA；该工具是 validation dataset 顺序重放，不是 production ring buffer/stable decision。

## 7. 未执行事项与资源证据

- 本次没有执行 full config、完整新格式 validation、test、阈值搜索、校准拟合、全量 canonical rebuild 或 feature stats rebuild。
- 已有 v9 18-epoch run 在同机文件时间上约为 20:41:23–23:08:16，约 2 h 27 min；这是历史观测，不是保证。
- v9 best checkpoint 约 15.2 MB；canonical tensors 约 2.77 GB。
- 最终 full 训练需要监控 pre-clip gradient norm、Bird recall、Bird→Drone 错误和 v8→v9 window 回归。
- 当前不支持 optimizer/RNG 训练状态 resume，只支持 strict best inference checkpoint reload。
- 全量训练必须由用户手动执行；本次审查未启动。
