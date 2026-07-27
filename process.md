# Process - RD 修复执行记录

> 状态：CQ08 帧索引和 RD-only 日志系统均已修复；修复版 Stage 009A 全预算训练待重新启动，融合与 test 继续冻结。  
> 原则：保留用户与并行修改；不提交、不暂存、不重置；Python 仅使用 `.venv`；test 冻结直到 validation 门禁通过。

## Stage 001 - 2026-07-23 冻结基线与评估协议

Status: Completed

- 固定 seed-42 trajectory-only checkpoint、v7 index、父轨迹 split 和 validation-only 复评口径。
- 记录窗口级、逐来源、父轨迹聚合、稳定决策和流式性能基线。
- independent test 未运行。

## Stage 002 - 2026-07-23 RD 主干、窗口和训练调度审计

Status: Completed

- 审计 `src/core/experts.py`、`src/core/expert1_trainer.py`、`src/core/data/torch_dataset.py`、`src/core/data/rd_segment_store.py`、`src/process/build_expert1_rd_segment_store.py` 与 v7 manifest/map。
- 确认 RD 主干包含 Transformer、统计特征、可选 patch CNN、micro-Doppler 与多种辅助头，但 Stage 019 设定 `stage1_traj_only_epochs=12`，因此 12 epoch 都未训练/启用 RD。
- 统计到 `25021` RD 段中 `24986` 是比例映射，严格安全 RD 窗口仅 train=`25`、val=`6`、test=`4`。
- 确认 exact 段中位时长仅 `0.1049s`，所有 exact 段均为 `short_segment`；比例映射保持安全阻断。
- 未改动模型、数据、配置或测试；未执行 independent test。

## Stage 003 - 2026-07-23 精确配对契约规划

Status: Planned; waiting for approval

- 将 RD 修复拆成 Stage 003-011：数据契约、原始资产审计、精确 segment、严格窗口 map、最小表征消融、RD-only 训练、残差融合、实时回退和发布门禁。
- 规定先验证数据合法性和 RD-only 增益，再允许任何 Joint 融合；不得放开 `proportional_fallback`。
- 覆盖更新 `task_plan.md`、`findings.md`、`process.md`；未实施任何未完成阶段。
- 等待用户确认后从 Stage 003 开始。

## Stage 003 - 2026-07-23 精确 RD/TR 配对契约与审计

Status: Completed

- 新增 `src/core/data/rd_pairing_contract.py` 和 `src/process/audit_expert1_rd_pairing.py`，将 RD 配对划分为 fusion-eligible、diagnostic-only、rejected；比例映射显式禁止。
- 写入 `config/contracts/expert1_rd_tr_pairing_v1.json`，并运行 `pytest tests\test_rd_pairing_contract.py -q --basetemp=tmp\pytest-stage003-pairing-contract`，结果 `6 passed`。
- 审计既有 manifest：`24986` 段 rejected、`35` 段 diagnostic-only、零段 fusion-eligible；未运行训练或 test。

## Stage 004 - 2026-07-23 原始 RD/TR 时间轴与目标关联审计

Status: Completed

- 扫描 CQ08 `106026` 个 MAT 帧的 `index_trackPointNo`，起始时间 `2026-07-23 01:05:05`，耗时 `526.9s`，无解析错误；未对原始数据写入。
- 关联确认：`1505/1549` 轨迹覆盖全部轨迹点，其余 `44` 条为部分覆盖但索引仍在界；CQ08 RD 可与同一目标的轨迹点时间锚定。
- 限制确认：CQ08 标准化 `time_raw` 是帧序号、GPS 为空；MAT 仅给出单帧 CPI 参数，不给出帧间 wall-clock。因此后续索引必须标记为 `track_point_anchor`，不能标称 exact-time fusion。
- 进入 Stage 005：构建新版本、不覆盖 v7 的 anchor manifest/store；重建前先完成实现和 smoke 校验。

## Stage 005 - 2026-07-23 CQ08 轨迹点锚定 RD 索引重建

Status: Completed (diagnostic-only)

- 2026-07-23 01:22:01 启动首版全量 anchor shard 重建；因每个锚点重复保存编码张量，至 01:47:35（`1533.6s`）耗尽 F 盘并失败。
- 确认失败输出仅为本次生成的 `1002` 个不完整 shard、`267.23GiB`，安全删除后恢复存储；未删除 v7、原始数据或用户文件。
- 改为紧凑 manifest 设计：引用既有 CQ08 `.npz` 和源帧索引，不复制 RD 张量。全量构建于 01:50:38–02:05:39 完成（`901.8s`），输出 `29413` anchor 段、`1549` 父轨迹、`106026` 帧引用。
- 所有 CQ08 anchor 标记为 `track_point_anchor`、`diagnostic-only`；不改变 fusion eligibility。

## Stage 006 - 2026-07-23 Anchor 因果窗口映射与安全读取

Status: Completed

- 构建 `expert1_realtime_causal_window_cq08_anchor_map_v1.csv`：`8240/9236` v7 窗口有同目标点锚定 RD 关联，`rd_fusion_eligible_window_count=0`。
- 扩展 `WindowAssembler` 读取紧凑 manifest 的源张量/帧索引；诊断路径可加载 RD，严格路径仍返回 `unavailable` 并强制 trajectory fallback。
- 修正多 anchor 窗口重复编码：聚合窗口内所有源帧后编码一次，RD 诊断长度从错误的 `512` 回归为实际 `128`。

## Stage 007 - 2026-07-23 RD-only Drone/Bird 诊断链路

Status: Completed; full-budget training ready

- 新增 `rd_drone_bird_diagnostic`、`train_rd_drone_bird` 与专用 config；只训练 RD Transformer/统计/micro-Doppler/Bird head，不构造 Joint 或最终融合 logits。
- 定义的 anchor 数据支持 train=`2718` windows / `510` parents、validation=`641` / `110` parents；test 保持未加载。
- 两次 smoke 验证数据、forward、反向、checkpoint 与 validation 报告；均衡 smoke 是 `32/32` 窗口且 Macro-F1=`0.3333`，不用于选模。
- 执行 `pytest tests\test_rd_pairing_contract.py tests\test_cq08_anchor_segment_store.py tests\test_rd_drone_bird_diagnostic.py tests\test_expert1_rd_segment_store_smoke.py -q --basetemp=tmp\pytest-stage008-final`，结果 `31 passed, 1 warning`；nested tensor warning 为 PyTorch prototype 提示。

## Stage 008 - 2026-07-23 RD-only 全预算训练准备

Status: Ready for user-run training

- 冻结 `tmp/20260723-stage008-rd-only-anchor-drone-bird-fullbudget-v1.json`：8 epochs、batch=4、680 train batches/epoch、161 full validation batches、test=false、patch CNN=false。
- CUDA 预检确认诊断 batch 的 RD 为 multimodal，Bird/micro-Doppler logits shape=`[4,2]`；同步状态仍为 `unavailable`，不能进入融合。
- 未启动全预算训练，等待用户执行命令后回传 `best_checkpoint`；之后仅做 validation 评估和 RD rescue/harm 分析。

## Stage 008 - 2026-07-23 RD-only 全预算 validation 复核

Status: Completed (negative diagnostic; fusion blocked)

- 用户完成 `artifacts/20260724-122359/20260723-stage008-rd-only-anchor-drone-bird-fullbudget-v1/best_rd_only_drone_bird_model.pt` 的 8 epoch RD-only 训练；最佳为 epoch=`3`，训练/选模仅使用 validation，test=false。
- 新增 validation-only evaluator `src/core/evaluate_rd_drone_bird.py`，输出 `tmp/20260724-stage008-rd-only-anchor-drone-bird-validation-recheck.json`；复评完全复现训练记录的 window Macro-F1=`0.5250`、parent Macro-F1=`0.5317`。
- 新增可复现对比 `src/core/compare_rd_drone_bird_trajectory.py`，输出 `tmp/20260724-stage008-rd-only-vs-trajectory-rescue-harm.json`；确认 RD-only 与 frozen trajectory 的 `641` 个 CQ08 Drone/Bird validation windows 在样本、父轨迹、真值和来源上完全一致。
- 窗口比较为 rescue=`66`、harm=`169`、net=`-103`；Bird 的 `142` 个 trajectory 正确窗口被 RD-only 错改为 Drone。父轨迹为 rescue=`4`、harm=`34`、net=`-30`，RD accuracy=`63.64%`，trajectory=`90.91%`。
- 结论：point-anchor RD 的 RD-only 专家为总体负增益，Stage 009 质量门控残差融合不启动；independent test、融合阈值与任何 RD 权重调优均未运行。

## Stage 009A - 2026-07-24 CQ08 压缩 tensor 帧索引完整性修复

Status: Completed repair; fixed full-budget RD-only training ready

- 2026-07-24：复核 v1 RD-only 负增益后发现 `standardize_cq08.py` 会按 target shape 跳过 MAT 帧，而 v1 anchor 使用原始 MAT 枚举序号访问压缩 tensor，导致对齐错配。
- 2026-07-24 14:33:16–14:50:42：全量扫描 `106026` 个 MAT 帧，耗时 `1045.2s`；生成 `data/processed/cq08_rd_frame_index_v1.csv`，其中 `88505` 帧被保留，`17521` 帧因 shape 不同被跳过，受影响父轨迹=`535`。
- 2026-07-24：修复 `build_expert1_cq08_anchor_segment_store.py`，将 `raw_frame_index` 与 `tensor_frame_index` 分列；重建 v2 manifest/map 耗时 `20.6s`，不复制 RD 张量、不覆盖 v1/v7。
- 2026-07-24：运行逐帧验证，`88505` 个保留 tensor 帧零漏引、零越界、零额外引用；严格加载仍 `unavailable` fallback，诊断加载可读。
- 2026-07-24：运行 v2 smoke（64 train / 64 validation windows）和相关回归，结果 `35 passed, 1 warning`；Macro-F1=`0.3333` 仅为短训练链路证据。
- 2026-07-24：冻结 `tmp/20260724-stage009-rd-only-anchor-v2-frame-index-fullbudget-v1.json`：8 epochs、batch=4、train=`2620` windows、validation=`618` windows、test=false。等待用户执行全预算 RD-only 训练后回传 checkpoint。

## Stage 009A - 2026-07-25 RD-only 训练日志与过程持久化修复

Status: Completed; full-budget restart ready

- 用户报告 v2 训练只显示 PyTorch nested-tensor warning 后无输出。检查确认训练器本身没有 batch/epoch print，CLI 仅在全部训练结束后打印 summary；这不是 nested-tensor warning 造成的死锁。
- 检查用户中断 artifact：只存在 `best_rd_only_drone_bird_model.pt`（epoch 3 时间戳），没有 history/summary，说明旧实现无法在中断时保留进度。
- 新增 READY、epoch START/END、first batch、每 `log_interval_batches` 的 `flush=True` 日志，并在每 epoch 立即落盘 `training_progress.json`/`training_history.json`。
- 运行 `tmp/20260725-stage009-rd-only-logging-smoke-v1.json`：64 train / 32 validation batch 完成，首训练 batch 0.80s、进度日志和 ETA 均正常；测试/融合仍未启用。
- 重新启动应使用带日志版本的 v2 frame-index 配置 `tmp/20260725-stage009-rd-only-anchor-v2-frame-index-logging-fullbudget-v1.json`；旧不完整 artifact 不作为候选或恢复点。

## Stage 009A - 2026-07-25 修复版 RD-only 训练预算调整

Status: Ready for user-run training

- 将修复版 RD-only 预算从 8 提升至 18 epoch，并保留原 8 epoch 配置以维持实验可追溯性。
- 新配置为 `tmp/20260725-stage009-rd-only-anchor-v2-frame-index-logging-fullbudget-v2-18epochs.json`：每 epoch 655 train batch、155 validation batch，合计最多 11790 train updates。
- 训练器每 epoch 使用 validation Macro-F1 更新 best checkpoint，同时立即落盘 progress/history；不以最终 epoch 替代 best checkpoint。
- test 与融合仍未启用。

## Stage 009A - 2026-07-25 v2 18-epoch RD-only validation 结论

Status: Completed (negative diagnostic; fusion remains blocked)

- 用户完成 `artifacts/20260725-010547/20260725-stage009-rd-only-anchor-v2-frame-index-logging-fullbudget-v2-18epochs/best_rd_only_drone_bird_model.pt`；最佳为 epoch `4`，未使用 independent test。
- validation 复评：window Macro-F1=`0.6198`，Drone P/R/F1=`73.47%/69.77%/71.58%`，Bird P/R/F1=`50.21%/54.75%/52.38%`，parent mean-logit Macro-F1=`0.6596`。
- 与冻结 seed-42 trajectory 的 `618` 个同集合 Drone/Bird validation windows 比较：RD rescue=`54`、harm=`134`、net=`-80`；parent rescue=`2`、harm=`26`、net=`-24`。因此 v2 不能参与 residual 或 Joint 融合。

## Stage 009B - 2026-07-25 全帧 Canonical RD store

Status: Full rebuild in progress

- 剩余数据根因：旧 `standardize_cq08.py` 对每个父轨迹固定首帧 shape，跳过 `17521/106026` MAT 帧；v2 虽索引正确，仍缺少部分 RD 序列。
- 新增 `src/process/build_cq08_rd_canonical_store.py`：每帧变换到 `31x64`，存储 float16 amplitude/phase 与 raw/tensor frame index；phase 经 sin/cos 重采样，规避圆周相位平均错误。
- 更新 anchor builder 读取 frame index 声明的 `rd_tensor_path`，可生成不覆盖 v2 的 v3 manifest；并修复 max-parent 的字符串排序问题。
- trainable smoke（tracks `276,277,281`）：`310/310` 帧、`52` anchors、`36` map windows；RD loader/encoder 可读，所有窗口 realtime status=`unavailable`，fusion eligible=`0`。
- 2026-07-25：全量 canonical 生成在工具时限内分段完成；最终 `1549` 父轨迹、`106026/106026` 帧、零 shape-drop、零错误。中途增加 resume 支持，已完成 tensor 的元数据直接复用 `cq08_rd_frame_index_v1.csv`，避免重复编码。
- 2026-07-25：生成 `expert1_cq08_rd_anchor_manifest_v3.csv`（`29413` segments）和 `expert1_realtime_causal_window_cq08_anchor_map_v3.csv`（`8240` windows）；所有 realtime status 为 `unavailable`，fusion eligible=`0`。
- 2026-07-25：运行 v3 RD-only smoke 配置：`2718` train / `641` validation windows，64 train / 32 validation batches 完整输出并保存 best validation checkpoint；test=false。冻结 18-epoch v3 full-budget 配置，等待用户训练和回传 checkpoint。
- 2026-07-25：相关回归完成：`tests/test_cq08_rd_canonical_store.py` 与 `tests/test_expert1_rd_segment_store_smoke.py` 共 `20 passed, 1 warning`；warning 为 PyTorch nested tensor prototype 提示。`py_compile`、配置 JSON 校验和 `git diff --check` 均通过。

## Stage 009B - Canonical v3 18-epoch RD-only validation 结论

Status: Completed (substantially improved; fusion remains blocked)

- 用户完成 checkpoint `artifacts/20260725-180452/20260725-stage009b-rd-only-anchor-v3-canonical-fullbudget-v1-18epochs/best_rd_only_drone_bird_model.pt`；best validation checkpoint 来自 epoch `3`，test=false。
- validation-only evaluator 输出 `tmp/20260725-stage009b-v3-18epochs-rd-only-validation-recheck.json`：window Macro-F1=`0.7346`，Drone P/R/F1=`80.15%/80.94%/80.54%`，Bird P/R/F1=`66.95%/65.82%/66.38%`，parent mean-logit Macro-F1=`0.7709`。
- 与 frozen trajectory 在完整 `641` 个对齐 Drone/Bird validation windows 比较，报告为 `tmp/20260725-stage009b-v3-18epochs-rd-only-vs-trajectory-rescue-harm.json`：RD rescue=`99`、harm=`106`、net=`-7`；父轨迹 rescue=`7`、harm=`21`、net=`-14`。trajectory window accuracy=`76.44%`，RD-only=`75.35%`。
- 结论：canonical v3 将 v2 的显著负增益收窄（v2 window net=`-80`），但尚未达到 rescue>harm、parent net>=0 或 Drone/Bird P/R>=95% 的门禁；融合与 independent test 继续冻结。

## Stage 009C - 2026-07-25 二维 RD patch 主干单因素验证

Status: Smoke in progress

- 新增 validation-only provenance 诊断 `src/core/diagnose_rd_drone_bird_residuals.py`，并输出 `tmp/20260725-stage009b-v3-residual-provenance-diagnostic.json`；对齐整体结果保持 rescue=`99`、harm=`106`、net=`-7`。
- 分组证据：Bird 总 net=`-36`，其中原始 shape `31x360` 的 Bird 窗口为 rescue=`1`、harm=`30`、net=`-29`；所有窗口选帧索引密度均高于 `0.95`，并非稀疏锚点造成。
- 主干复核：v3 最优配置 `use_rd_patch_cnn=false`，二维 `64x64` patch CNN 未被训练；准备仅把该开关置为 true 的 RD-only CUDA smoke `tmp/20260725-stage009c-rd-only-anchor-v3-patchcnn-smoke-v1.json`。
- 2026-07-25：patch-CNN smoke 先以 batch=`2`（16 train/16 validation batch）再以 batch=`4`（4 train/4 validation batch）运行，均完成且无 CUDA OOM、日志或 checkpoint 异常；test=false。batch=`4` 的后续训练 batch 约 `1.3–1.8s`，明显高于非 patch v3。
- 冻结全预算单因素配置 `tmp/20260725-stage009c-rd-only-anchor-v3-patchcnn-fullbudget-v1-18epochs.json`：与 v3 canonical 18-epoch 训练唯一算法开关差异为 `use_rd_patch_cnn=true`；等待用户训练并回传 best checkpoint。

## Stage 009C - 2026-07-26 Patch-CNN 18-epoch validation 结论

Status: Completed (shape subgroup improves; overall candidate rejected)

- 用户完成 checkpoint `artifacts/20260726-003011/20260725-stage009c-rd-only-anchor-v3-patchcnn-fullbudget-v1-18epochs/best_rd_only_drone_bird_model.pt`；best epoch=`12`，test=false。
- 初次 evaluator 复评暴露 patch 参数转发缺失：训练器传入 `rd_patch_features` 与 `rd_patch_padding_mask`，而 evaluator 未传入，导致 128 步 mask 和 4 步 patch 的 shape mismatch。修复 `src/core/evaluate_rd_drone_bird.py` 后，validation-only evaluator 复现 checkpoint：window Macro-F1=`0.7325`，Drone P/R=`89.53%/65.59%`，Bird P/R=`59.71%/86.92%`，parent Macro-F1=`0.8261`。
- 对齐 frozen trajectory 的 `641` validation windows：rescue=`73`、harm=`92`、net=`-19`；parent rescue=`5`、harm=`14`、net=`-9`。Bird `31x360` 由 net=`-29` 改善到 `-12`，但 Drone harm 大幅增加，整体比非 patch v3 的 window net=`-7` 更差。
- 结论：patch CNN 不能作为替代主干或融合候选；下一步仅考虑 sequence 主干保持不变、patch 作为受控 residual correction 的 RD-only 消融，independent test 和融合继续冻结。

## Stage 009D–009H - 2026-07-26 RD `0.90+` 残差路线规划

Status: Planned; awaiting user approval before implementation

- 目标明确为同一 validation 协议下 Drone 与 Bird 的 precision、recall 均达到 `0.90`；该目标不是当前 `0.73` Macro-F1 的近似表达，必须同时通过四项类别指标、window rescue/harm 和 parent net 门禁。
- 基于 v3 与 patch-CNN 对照，冻结下一条路线：non-patch v3 checkpoint 保持 sequence 主 logits；patch 支路只输出受限 residual delta，不能直接替换主 logits，也不能使用标签、split、trajectory 预测或人工阈值作为 gate 输入。
- 已在 `task_plan.md` 拆分为最小可执行阶段：残差契约、独立 wrapper、batch=4 smoke、18-epoch 对照、分组门禁。每阶段都保持 validation-only、test=false、fusion=false。
- 后备路线已记录：若 residual 对照仍未通过，才审计原始 RD 物理 Doppler 坐标并考虑 high-resolution canonical v2；本次未修改模型、数据、训练配置或启动任何新实验。

## Stage 009D–009G - 2026-07-26 Frozen Sequence Patch Residual 已就绪

Status: Implementation and smoke completed; full-budget user-run training ready

- 新增隔离模块：`rd_patch_residual_config.py`、`rd_patch_residual_model.py`、`rd_patch_residual_trainer.py`、`train_rd_patch_residual.py`、`evaluate_rd_patch_residual.py`。base 为 non-patch v3 checkpoint 且所有参数冻结；仅优化 patch CNN、residual head 与 gate。
- residual 输出为 `base_logits + tanh(raw_delta) * gate * residual_scale`；residual head 末层零初始化，gate 初始偏向关闭。patch 缺失、全无效或 `force_zero_residual` 时，final logits 逐元素回退 base logits。
- 真实 batch 契约验证通过：zero residual exact、invalid patch exact、base frozen 均为 true。batch=`4` CUDA smoke 完成 `8` train/`8` validation batch，无 OOM；normal/force-zero evaluator 均可运行，test=false、fusion=false。
- 冻结 18-epoch 单因素配置 `tmp/20260726-stage009g-frozen-sequence-patch-residual-fullbudget-v1-18epochs.json`；保持 canonical v3 数据、split、seed、batch=4、optimizer 和 validation-only 协议，等待用户训练并回传 best checkpoint。

## Stage 009H - 2026-07-27 Frozen Sequence Patch Residual 门禁结论

Status: Completed (candidate rejected; physical-resolution audit next)

- 用户完成 checkpoint `artifacts/20260726-235348/20260726-stage009g-frozen-sequence-patch-residual-fullbudget-v1-18epochs/best_rd_patch_residual_model.pt`；best epoch=`9`，test=false。
- validation-only evaluator 输出 `tmp/20260727-stage009g-residual-18epochs-validation-recheck.json`：window Macro-F1=`0.7504`，Drone P/R=`83.83%/76.98%`，Bird P/R=`65.56%/74.68%`，parent Macro-F1=`0.7869`。
- 与 frozen trajectory 的完整 `641` 窗口对齐：rescue=`100`、harm=`102`、net=`-2`；parent rescue=`8`、harm=`21`、net=`-13`。Bird `31x360` net=`-18` 优于 base `-29`，但 Drone `31x570` net=`-17` 劣于 base `-10`。
- residual gate mean=`0.866`、P90=`0.9999`，delta P90 sum=`2.00`；说明有界形式仍被训练为近乎全局改写。该候选未通过所有 Stage 009H 门禁，fusion/test 继续冻结。
- 下一步仅做原始 MAT `debug_info` 的只读 Doppler 物理坐标/采样率审计，判断 high-resolution canonical v2 是否有物理依据；本阶段不重建数据或启动新训练。

## Stage 009I - 2026-07-27 CQ08 物理 Doppler 坐标审计

Status: Completed; high-resolution canonical v2 rebuild awaits user confirmation

- 新增 `src/process/audit_cq08_rd_physical_coordinates.py`，只读取原始 MAT 的 `debug_info` 并生成审计报告；未读取 label 参与任何判定，未重建数据、训练、融合、调阈值或运行 test。
- 初次单进程扫描在工具 `30` 分钟时限前完成 `98000/106026` 帧且零错误；改为三个可复现分段并完成全量：`tmp/20260727-stage009i-cq08-rd-physical-coordinate-audit-part1.json`、`...part2.json`、`...part3.json`，合并报告为 `tmp/20260727-stage009i-cq08-rd-physical-coordinate-audit.json`。
- 全量结果：`106026/106026` 帧必需字段覆盖、一个一致字段集合、零 MAT 读取/解析错误、零 metadata shape mismatch、零 `prt_num` 与 Doppler width 不一致。
- 采用设备使用的工程常数 `c=3.0e8` 后，物理 grid 公式逐帧通过：最大 bin-width 相对误差=`2.124e-16`。原始 bin 宽度范围由 `31x142` 的 `2.0905–2.4184 m/s` 到 `31x900` 的 `0.2055–0.2226 m/s`，证明旧 `31x64` index-space resize 混合了不同物理尺度。
- 全部帧公共无外推速度区间为 `[-90.2201, 89.4217] m/s`；可据此设计保守公共物理速度轴，但 v2 的分辨率、抗混叠、裁剪 mask 仍需先获得用户确认。RD 的时序同步证据未变化，fusion 与 test 继续冻结。

## Stage 009I - 2026-07-27 物理 Canonical v2 重建执行

Status: In progress; last partition delegated to user to avoid CLI timeout

- 用户确认后，固定 schema：只重采样 Doppler 物理轴，公共范围=`[-85,85] m/s`、`384` bins、bin width=`0.443864 m/s`；range 保留 `31` 个 target-local bins。source bin 更细时才应用 Gaussian prefilter；相位固定为 sin/cos 插值后 `atan2` 还原。
- 新增 `src/process/build_cq08_rd_physical_canonical_store.py`。每 parent 生成独立 `.npz`，包含 amplitude、phase、time_raw、velocity_mps；每帧 index 保存 source bounds、source bin width、公共轴覆盖率、anti-alias sigma。新增 `src/process/merge_cq08_rd_physical_canonical_index.py`，合并前会逐帧比较原始 parent/raw-frame 集合、检查 tensor index 连续、tensor 文件存在、canonical shape 和覆盖率。
- 更新 `src/core/data/rd.py`：仅当 `.npz` 含 `velocity_mps` 且长度匹配时，主 RD 序列在 encode 前保留物理 canonical 的 `384` Doppler bins；旧 v1/其他数据仍使用既有 `64` bins 兼容路径。
- `3` parent smoke：`308/308` 帧、`31x384`、零错误、覆盖率=100%。额外原始 `31x570/900` 验证输出有限，anti-alias sigma=`0.1315/0.4971`；v2 tensor 载入后 `velocity_mps=(384,)` 并能成功经过 RD 编码。
- 全量因单一运行超过 CLI `30` 分钟上限，在约 `50000/106026` 帧后被工具终止；未采用不安全的覆盖重跑。改按 parent 分区重建，前三段 `[0,320)`、`[320,762)`、`[762,1149)` 分别完成 `26009/26025/26025` 帧，均零错误，累计 `78059` 帧、anti-alias=`21149`。
- 最终 `[1149,1549)` 分区预期 `27967` 帧，需用户手动运行；之后先合并与完整性校验，再创建 v4 anchor/map。期间不允许训练、fusion、阈值调优或 test。

## Stage 009I - 2026-07-27 物理 v2 合并、v4 映射与训练就绪

Status: Training-ready; waiting for user-run full-budget RD-only experiment

- 用户手动完成最后 `[1149,1549)` 分区并运行合并校验。最终物理 v2 frame index：`106026` frames、`1549` parents，对原始 reference 集合零缺失/零额外、零重复、零连续 tensor-index 错误、零缺 tensor、全部 `31x384`、公共速度轴覆盖率 100%；anti-alias=`33584`。
- 生成 `expert1_cq08_rd_physical_anchor_manifest_v4.csv`：`29413` segments、`1505` full/`44` partial anchor parents、零错误、零 shape-drop。生成 `expert1_realtime_causal_window_cq08_physical_anchor_map_v4.csv`：`8240/9236` mapped windows、平均 segment count=`7.67`、全部 realtime=`unavailable`、fusion eligible=`0`。
- 真实 v4 dataloader 初检发现 RD placeholder：map/manifest/tensor 路径均正确，根因是 `WindowAssembler._slice_anchor_tensor` 将静态 `velocity_mps` 误按时间 index 切片并返回 None。修复后相同窗口 batch 为 `[4,128,512,4]`，RD length 均=`128`、multimodal=true、数值 finite；sync 仍为 `unavailable`。
- 创建并执行 CUDA smoke `tmp/20260727-stage009i-rd-physical-v2-smoke.json`：batch=`4`、`4` train/`4` validation batch，首个 train batch 完成 `2.47s`，首个 validation batch `0.87s`；日志、validation checkpoint、progress/history 均正常，test=false。
- 冻结 full-budget config `tmp/20260727-stage009i-rd-physical-v2-fullbudget-v1-18epochs.json`：18 epochs、batch=4、`2718` train / `641` validation windows、validation-only selection、patch=false、test=false、fusion=false。等待用户回传 best checkpoint 后再做评估。

## Stage 009I - 2026-07-27 物理 v2 Full-budget Validation 结论

Status: Completed as RD-only baseline improvement; rejected for fusion and `0.90+` gate

- 用户完成 checkpoint `artifacts/20260727-145406/20260727-stage009i-rd-physical-v2-fullbudget-v1-18epochs/best_rd_only_drone_bird_model.pt`；best epoch=`7`，仅 validation 选模，test=false。
- validation-only evaluator 输出 `tmp/20260727-stage009i-physical-v2-18epochs-validation-recheck.json`：window accuracy=`0.7816`、Macro-F1=`0.7746`；Drone P/R=`87.71%/75.99%`，Bird P/R=`66.67%/81.86%`；parent mean-logit Macro-F1=`0.8594`。这是所有既有 RD-only 对照中的最高 window 与 parent Macro-F1。
- `tmp/20260727-stage009i-physical-v2-18epochs-vs-trajectory-rescue-harm.json` 对齐冻结 trajectory 的相同 `641` 个 windows：rescue=`94`、harm=`83`、net=`+11`。但 `110` parents 的 mean-logit 对比为 rescue=`7`、harm=`12`、net=`-5`，仍不满足 parent 安全门禁。
- provenance 输出 `tmp/20260727-stage009i-physical-v2-residual-provenance-diagnostic.json`：`31x142` Drone net=`+28`、Bird `31x226`=`+16`、Bird `31x570`=`+7`；Bird `31x360`=`-17`、Drone `31x226+31x570`=`-8` 和 Drone `31x570`=`-10` 仍是主要负组。
- 结论：物理 Doppler 对齐有效，v2 保留为唯一 RD-only baseline；但不可启动 fusion/threshold/test，也不可声明达到 Drone/Bird `0.90+`。后续只允许围绕 parent-stability 的单因素 RD-only 设计，先使 parent net 非负。
