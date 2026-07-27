# Task Plan - RD 驱动 Drone/Bird 判别修复

> 状态：v2 帧索引修复后 RD-only 仍为负增益；Stage 009B 正在建立全帧 canonical RD v3 资产，融合继续阻断。  
> 约束：训练仅用 train；模型选择、阈值与消融只用 validation；independent test 冻结。比例时间映射 RD 永不进入融合训练或推理。Python 与依赖仅使用 `.venv`。

## Stage 001 - 冻结基线与评估协议

状态：已完成

- [x] 固定 seed-42 trajectory-only 基线、v7 index、父轨迹 split、标签顺序与 validation-only 命令。
- [x] 固定窗口、source-balanced、父轨迹 mean-logit/stable、翻转率、稳定决策时延与流式 P95/P99 指标。
- [x] 明确 test 不参与模型选择、损失权重、阈值或架构决策。
- [x] 验收：基线契约和回滚检查点可复现。

## Stage 002 - RD 现状与阻断证据冻结

状态：已完成

- [x] 审计 RD manifest、v7 window map、数据加载、主干、辅助头、融合门控和训练阶段调度。
- [x] 量化精确同步 RD 覆盖、类别支持、RD 时长及窗口尺度差异。
- [x] 确认 Stage 019 全程 trajectory-only，RD 路径未训练、未参与最终判决。
- [x] 验收：形成可审计的“不得放开 proportional fallback”证据。

## Stage 003 - 定义精确 RD/TR 配对数据契约

状态：已完成

- [x] 定义同目标关联、共享/可校正时间轴、覆盖率、连续时长、帧间隔与质量元数据的契约：`config/contracts/expert1_rd_tr_pairing_v1.json`。
- [x] 定义拒绝原因与三态决策：`fusion_eligible`、`diagnostic_only`、`rejected`；比例映射明确禁止。
- [x] 固定不泄漏字段边界：来源、路径、样本/父 ID、split、标签与 timestamp 均不得输入模型或门控。
- [x] 新增 `rd_pairing_contract` 单元测试，`6 passed`；manifest-only 审计输出 `tmp/20260723-stage003-rd-pairing-audit.json`。
- [x] 验收：既有 `25021` 段中 `24986` 被拒绝、`35` 为短段 diagnostic-only，零段可融合。

## Stage 004 - 审计原始 RD 时间轴与目标关联资产

状态：已完成

- [x] 审计 raw 与 CQ08 原始容器、时间字段、MAT 元数据和目标关联字段；输出 `tmp/20260723-stage004-cq08-mat-trackpoint-scan.json` 与 `tmp/20260723-stage004-cq08-trackpoint-coverage.csv`。
- [x] `raw_tr_rd` 有配对 ID 与相对时间轴，但仅 `35` 个短轨迹；不满足连续 RD 融合样本量。
- [x] CQ08 标准化的 `time_raw` 是人为帧序号、GPS 字段为空，不能称为 absolute/exact time。
- [x] 发现原始 CQ08 MAT 帧含 `index_trackPointNo`：扫描 `106026` 帧耗时 `526.9s`，零解析错误；`1505/1549` 轨迹覆盖全部轨迹点、全部索引均在边界内。
- [x] 验收：CQ08 可重建为“同目标、轨迹点时间锚定、带时间不确定度”的 RD 索引；不得将其误标为 exact time sync。

## Stage 005 - 重建精确 RD 段与质量元数据

状态：已完成（CQ08 为诊断锚定索引，不是 exact fusion RD）

- [x] 对 raw 数据保留真实相对时间轴；对 CQ08 按 `index_trackPointNo -> trajectory point time` 重建，未使用父轨迹比例切片。
- [x] 生成紧凑 manifest：`data/processed/expert1_cq08_rd_anchor_manifest_v1.csv`，写入目标关联、轨迹点索引/时间、源 RD 帧索引、锚定不确定度、单帧 CPI 时长与质量标记。
- [x] 全量重建引用 `106026` 个既有 RD 源帧，生成 `29413` 个 anchor 段、`1549` 个父轨迹（完整点覆盖=`1505`、部分=`44`）；耗时 `901.8s`（`15分02秒`）。
- [x] 修正一次失败的存储设计：首次把重复编码张量写入 shard，在 `25分34秒` 后耗尽 F 盘（`1002` 文件、`267.23GiB`）；已安全删除本次不完整输出，并改为仅引用既有 `.npz` 源张量的紧凑 manifest。
- [x] 验收：anchor 状态固定为 `track_point_anchor`/`diagnostic-only`，不合格融合证据保持不可融合；v7 资产未覆盖。

## Stage 006 - 构建严格因果 RD 窗口映射

状态：已完成（诊断读取可用，融合资格保持为零）

- [x] 以 v7 因果轨迹窗口为基准，构建 `data/processed/expert1_realtime_causal_window_cq08_anchor_map_v1.csv`；映射 `8240/9236` 个窗口。
- [x] 每个 anchor 仅表达同目标轨迹点关联；覆盖率固定为 `0`，短 RD 不伪装为 10 秒连续覆盖。
- [x] map 中全部 `track_point_anchor` 窗口解析为 realtime `unavailable`，`rd_fusion_eligible_window_count=0`；生产融合仍强制 trajectory fallback。
- [x] Drone/Bird 支持量：train=`2718` windows / `510` parents，val=`641` / `110`，test=`572` / `114`；test 未加载进训练或选模。
- [x] 验收：读取器可按 manifest 引用的源帧索引加载诊断 RD，并在 `load_only_fusion_eligible_rd=true` 时明确拒绝。

## Stage 007 - RD 表征与辅助头最小化消融

状态：实现与 smoke 已完成；全预算 RD-only 诊断待训练

- [x] 新增 `rd_drone_bird_diagnostic`：复用 RD Transformer、统计与 6 维 micro-Doppler 头，但 API 不构造 trajectory/Joint/final fusion logits。
- [x] 新增 RD-only Drone/Bird 训练入口；仅加载 train/validation anchor windows，`run_test_evaluation=false`，且训练代码不调用任何 Joint 融合。
- [x] 默认第一轮仅启用 Transformer token + 统计 + micro-Doppler 辅助头，关闭 patch CNN，禁用 Unknown/Clutter 辅助头。
- [x] 修复单帧 RD 统计的 `std` warning（使用 `unbiased=false`）；新增 anchor 诊断读取与 RD-only forward 测试。
- [x] 验收：相关回归 `31 passed`；均衡 smoke 的 `32` train / `32` validation windows 完成（Macro-F1=`0.3333`，仅证明链路，不能选模）。

## Stage 008 - RD-only Drone/Bird 专家训练与评估

状态：已完成（diagnostic-only，未通过融合门禁）

- [x] 全预算训练完成：`8` epochs、batch=`4`、train=`2718` windows、validation=`641` windows、test=`false`；最佳 checkpoint 为 `artifacts/20260724-122359/20260723-stage008-rd-only-anchor-drone-bird-fullbudget-v1/best_rd_only_drone_bird_model.pt`（epoch=`3`）。
- [x] 独立 validation 复评一致：window Macro-F1=`0.5250`，Drone P/R=`65.06%/80.20%`、Bird P/R=`44.06%/26.58%`；parent mean-logit Macro-F1=`0.5317`。
- [x] 对齐冻结 trajectory 基线的全部 `641` 个 CQ08 Drone/Bird validation windows（样本 ID、父轨迹、真值、来源均一致）：RD-only rescue=`66`、harm=`169`、净=`-103`。
- [x] 父轨迹层比较：RD mean-logit rescue=`4`、harm=`34`、净=`-30`；accuracy=`63.64%`，低于 trajectory majority=`90.91%`。
- [x] 结论：真实 Bird 的 RD margin 仍整体偏向 Drone，RD-only 不能作为 residual 或替换 trajectory；`track_point_anchor` 的融合阻断保持不变，未运行 test。
- [x] 验收：诊断报告与可复现对比脚本已输出；Stage 009 不启动，除非取得独立帧级时钟/连续时间证据并重新通过 RD-only 门禁。

## Stage 009A - CQ08 压缩 RD 帧索引完整性修复

状态：已修复；修复版 RD-only 全预算诊断待训练

- [x] 定位根因：CQ08 标准化会跳过父轨迹中 shape 不同的 MAT 帧，但 v1 anchor 将原始 MAT 枚举序号误作压缩 `.npz` tensor 序号；`17521` 个被跳过帧使 `5886` 个 v1 anchor 段含失效引用。
- [x] 全量构建 `data/processed/cq08_rd_frame_index_v1.csv`：`106026` 原始 MAT 帧映射至 `88505` 保留 tensor 帧，`535` 父轨迹存在 shape-drop，耗时 `1045.2s`（`17分25秒`）。
- [x] 生成 v2 anchor manifest/map：`expert1_cq08_rd_anchor_manifest_v2.csv`、`expert1_realtime_causal_window_cq08_anchor_map_v2.csv`；记录原始 MAT 索引与压缩 tensor 索引两个独立字段。
- [x] 逐帧验收：全部 `88505` 个保留 tensor 帧零漏引、零越界、零额外引用；`rd_fusion_eligible_window_count=0` 保持不变。
- [x] v2 诊断读取与均衡 smoke 验证通过：严格路径仍回退，诊断路径可加载；相关测试 `35 passed`。
- [x] 修复 RD-only 训练日志：启动、首 batch、每 `log_interval_batches`、每 epoch 均 `flush=True` 输出；`training_progress.json` 和 `training_history.json` 每 epoch 立即落盘。
- [x] 日志 smoke 验证：`64` train / `32` validation batch，首 batch=`0.80s`，每 `8` batch 有进度、ETA 和 loss；中断不会再只留下 checkpoint。
- [ ] 重新运行修复版 Stage 009A RD-only full-budget：`tmp/20260725-stage009-rd-only-anchor-v2-frame-index-logging-fullbudget-v2-18epochs.json`：epochs=`18`、每 epoch=`655` batch；逐 epoch 保存最佳 validation checkpoint，test/fusion 禁止。
- [ ] 验收：修复版 RD-only 必须在同一 validation 对齐集合上使 rescue>harm、父轨迹净增益非负；否则 Stage 009 融合永久保持阻断，等待新的独立时间证据。

## Stage 009 - 质量门控 RD 残差融合

状态：阻断（Stage 008 RD-only 净增益为负）

- [ ] 使用 trajectory logits 作为主判决，RD 只输出 Drone↔Bird 残差；不让 RD 直接重写其他类别。
- [ ] 门控只使用可观测质量：同步、覆盖、连续时长、帧间隔、SNR/饱和、RD 自置信与 TR/RD 冲突，不使用 provenance 字段。
- [ ] 先实现单一可解释残差门控，再与现有复杂 Joint head 隔离对比；禁止同时新增多个辅助头。
- [ ] 训练时加入 RD dropout、迟到/缺失/乱序模拟，确保无 RD 时 logits 与 trajectory 基线一致。
- [ ] 验收：validation 的 Drone/Bird rescue 大于 harm，source-balanced、父轨迹 stable 指标不低于基线。

## Stage 009B - CQ08 全帧 Canonical RD 表征修复

状态：已完成（全帧 canonical v3 已验证；融合继续阻断）

- [x] 诊断 v2：帧索引已正确但旧标准化仍丢弃 `17521/106026`（`16.52%`）异形 MAT 帧，可能移除 Drone/Bird 的关键微多普勒片段。
- [x] 实现 `build_cq08_rd_canonical_store.py`：每个 MAT 帧保存为统一 `31x64` 幅度/相位张量；相位以 sin/cos 重采样后还原，避免 `±π` 环绕伪影；索引分列 raw/tensor frame index。
- [x] 修复 anchor builder 的父轨迹选择：先和 frame index 求交集后按数值 track ID 截断，避免 `1/10/100` 字符串排序 smoke 错配。
- [x] trainable smoke：`276/277/281` 共 `310/310` 帧、`52` anchor、`36` 实时窗口，零 shape-drop、零索引越界；`rd_fusion_eligible_window_count=0`。
- [x] 全量生成 `cq08_rd_canonical_v1/`、`cq08_rd_canonical_frame_index_v1.csv`，验收所有成功读取帧被保留且每父轨迹 tensor 索引连续。
- [x] 生成 v3 canonical anchor manifest/map，保持 `track_point_anchor`、`unavailable` 和融合资格零；不覆盖 v2。
- [x] 全量生成 `cq08_rd_canonical_v1/`、`cq08_rd_canonical_frame_index_v1.csv`：`1549` 父轨迹、`106026/106026` 帧、零 shape-drop、零错误、每父轨迹 tensor index 连续。
- [x] 生成 v3 canonical anchor manifest/map：`29413` segments、`8240` mapped windows；全部为 `unavailable`，`rd_fusion_eligible_window_count=0`，未覆盖 v2。
- [x] v3 RD-only smoke：`2718` train / `641` validation windows，CUDA batch/epoch 日志与 best validation checkpoint 正常，未运行 test。
- [x] 冻结 `tmp/20260725-stage009b-rd-only-anchor-v3-canonical-fullbudget-v1-18epochs.json`：18 epoch、batch=4、仅 validation 选模；test 与融合禁止。
- [x] 完成 v3 18-epoch validation 复评：Macro-F1=`0.7346`，但冻结 trajectory 对齐比较 window net=`-7`、parent net=`-14`；Stage 009 融合继续阻断。
- [x] 按父轨迹、原始 RD shape、有效连续帧数和类别分组诊断 v3 的 rescue/harm：所有选帧基本连续；Bird `31x360` 为主要 residual harm 组（net=`-29`）。

## Stage 009C - 二维 RD Patch 主干单因素消融

状态：已完成（局部 Bird 子组改善，但整体候选拒绝）

- [x] 复核主干：v3 最优 RD-only 使用 `use_rd_patch_cnn=false`，二维 `64x64` RD patch CNN 不参与分类；这与 Bird `31x360` 残余伤害的表征缺口相符。
- [x] 运行 patch-CNN CUDA smoke（batch=`2` 与正式 batch=`4`）：无 OOM、batch/epoch 日志正常、validation checkpoint 写入；test=false。batch=`4` 的训练 batch 约 `1.3–1.8s`，后续全预算训练应预留数小时。
- [x] 冻结 `tmp/20260725-stage009c-rd-only-anchor-v3-patchcnn-fullbudget-v1-18epochs.json`：仅将 `use_rd_patch_cnn` 从 false 切换为 true，保持 v3 数据、seed、batch、损失、优化器与 validation-only 选模不变。
- [x] 完成 18-epoch patch-CNN 单因素对照：Bird `31x360` harm 降低，但整体 window net=`-19`、parent net=`-9`；候选拒绝，融合继续阻断。
- [ ] 验收：若 window rescue<=harm 或 parent net<0，融合继续阻断；不得用阈值补偿或 test 调优。

## Stage 009D - Sequence 主干与 Patch 残差契约

状态：已完成（冻结 sequence 与零残差契约已验证）

- [x] 固定 non-patch v3 checkpoint 为回滚主干：window Macro-F1=`0.7346`、Drone P/R=`80.15%/80.94%`、Bird P/R=`66.95%/65.82%`。
- [x] 固定 patch-CNN 对照结论：Bird `31x360` net=`-12` 优于 non-patch 的 `-29`，但全窗口 net=`-19`，不得作为直接主分类器。
- [x] 定义残差输出：`final_logits = frozen_sequence_logits + bounded_patch_delta`；patch 不得直接替换或重训练 sequence 主 logits。
- [x] 定义零残差契约：patch 缺失、全无效或 force-zero 时，final logits 必须逐元素等于 frozen sequence logits。
- [x] 定义审计字段：evaluator 输出 base logits、delta logits、gate；原始 RD shape 与连续帧数继续由 provenance diagnostic 关联，禁止使用标签、split、轨迹预测或人工阈值作为 gate 输入。
- [x] 验收：接口可加载 frozen checkpoint；真实 batch 零残差回归通过；strict `unavailable` 未进入 Joint/final fusion。

## Stage 009E - 受控 Patch Residual 实现

状态：已完成（独立 wrapper、trainer 与 evaluator 已实现）

- [x] 新建独立 RD-only residual wrapper，冻结 sequence 主干；只训练 patch encoder、residual head 与可解释 gate。
- [ ] 将 patch delta 通过有界函数和残差正则限制，记录平均/分位 residual 幅度，防止重复 Stage 009C 的 Bird 偏置。
- [ ] 保持训练数据、parent split、seed、optimizer、batch size、loss 主项和 validation-only 选模与 v3 一致；不改 Unknown 规则。
- [ ] 为 checkpoint 加入 base checkpoint provenance、残差尺度和 gate 配置，拒绝不匹配的 base 权重。
- [x] 验收：真实 frozen checkpoint + real RD batch 覆盖零残差、patch mask 和 checkpoint 契约；测试目录补丁受限时不绕过写入。

## Stage 009F - Residual Smoke 与性能预算

状态：已完成（batch=4 CUDA smoke 与零残差回归通过）

- [x] 运行 batch=`4` CUDA smoke，验证显存、首 batch、日志、checkpoint 与训练器/evaluator 参数一致。
- [ ] 测量 sequence-only、patch-only、residual 三种 RD-only forward 的 batch 时延与显存；不以 smoke Macro-F1 选模。
- [ ] 验证强制零残差模式逐元素复现 non-patch v3 logits；RD 缺失/无效时仍保持 trajectory fallback。
- [x] 验收：无 OOM、evaluator/train path 一致、force-zero 逐元素精确复现 base logits。

## Stage 009G - Full-budget Residual 对照训练

状态：待训练（18-epoch 单因素配置已冻结）

- [x] 冻结 18-epoch 单因素配置；仅增加 residual branch，不扩大原始 RD 数据、window、label 或 test 范围。
- [ ] 每 epoch 写入整体、Bird `31x360`、Drone `31x570` 的 validation P/R/F1 与 rescue/harm；best checkpoint 只按预先声明的 validation Macro-F1 选择。
- [ ] 训练过程中不调整阈值、类别权重、学习率或残差尺度；若需要新超参数，另建独立实验卡。
- [x] 验收：训练完成且 artifact 记录完整；independent test 与融合仍未执行。

## Stage 009H - 分组门禁与下一方向决策

状态：待执行（依赖 Stage 009G）

- [x] 对齐冻结 trajectory 的同一 `641` 个 validation windows，报告全窗口和 parent rescue/harm/net。
- [x] 复核 Bird `31x360` 改善是否保持 Drone recall 不低于 non-patch v3 的 `80.94%`：未保持，residual Drone recall=`76.98%`。
- [x] 拒绝 residual 方向：window net=`-2`、parent net=`-13`；转入物理坐标高分辨率 canonical v2 的只读可行性审计。
- [x] `0.90+` 门禁未通过：Drone/Bird precision、recall 均未同时达到 `0.90`；fusion 与 independent test 继续冻结。

## Stage 009I - CQ08 物理 Doppler 坐标可行性审计

状态：full-budget 已评估（窗口级正增益；父轨迹与 `0.90+` 门禁未通过，fusion/test 继续冻结）

- [x] 新增只读审计器 `src/process/audit_cq08_rd_physical_coordinates.py`；不加载标签进判定逻辑，不重建、训练、融合或评估 test。
- [x] 全量读取 `106026/106026` 原始 CQ08 MAT 的 `debug_info`；每帧均具备 `freq_hz`、`prt_num`、`prt_s`、`rd_result_size`、`vr_start`、`vr_end`，零解析错误、零 shape/PRT 宽度不一致。
- [x] 验证 `vr` 轴与 `c=3.0e8` 的 `c/(4*freq_hz*prt_s)`、`2*v_unambiguous/prt_num` 完全一致；最大相对 bin-width 误差=`2.124e-16`。
- [x] 确认所有帧共同无外推速度支撑约为 `[-90.22, 89.42] m/s`；原始 shape 不能再作为分类输入或切分捷径。
- [x] 固定 v2 契约：只物理重采样 Doppler 轴，range 保留 `31` 个 target-local bins；公共速度范围=`[-85,85] m/s`、`384` Doppler bins（`0.443864 m/s/bin`），只对 source bin 更细的帧进行 Gaussian 抗混叠；相位以 sin/cos 插值后 `atan2` 还原。
- [x] 实现 `build_cq08_rd_physical_canonical_store.py` 与 `merge_cq08_rd_physical_canonical_index.py`；每帧 frame index 记录 source velocity bounds、bin width、公共轴覆盖率、anti-alias sigma，合并时与原始 `106026` parent/raw-frame 集合逐一比对。
- [x] smoke：`3` parents、`308/308` 帧、`31x384`、零错误；`31x570/900` 帧输出有限且 anti-alias sigma 分别为 `0.1315/0.4971`；新 `velocity_mps` 元数据在 `encode_multimodal_rd` 中被识别，主序列 Doppler 投影不再预先压回 `64` bins。
- [x] 完成全量分区 1–3：父轨迹 `[0,320)`、`[320,762)`、`[762,1149)`，合计 `78059/106026` 帧，零错误；已对 `21149` 帧应用 anti-alias。
- [x] 用户完成分区 4 并合并验收：`106026` 帧集合零缺失/零额外、每父轨迹 tensor index 连续、`31x384`、公共速度覆盖率 100%、tensor 文件零缺失；anti-alias 总帧数=`33584`。
- [x] 全量生成 v4 physical anchor manifest/window map：`29413` segments、`8240` mapped windows、零 shape-drop、`rd_fusion_eligible_window_count=0`。
- [x] 修复 v4 RD 读取：`velocity_mps` 是静态 Doppler 坐标，`WindowAssembler` 不得按帧 index 切片；真实 batch 现为 `[4,128,512,4]`、multimodal=`true`、sync=`unavailable`。
- [x] CUDA smoke：`4` train + `4` validation batches、batch=`4`、真实 RD length=`128`、日志/validation checkpoint 正常、test=false；冻结 `tmp/20260727-stage009i-rd-physical-v2-fullbudget-v1-18epochs.json`。
- [x] 用户完成 physical v2 18-epoch RD-only：best epoch=`7`，checkpoint=`artifacts/20260727-145406/20260727-stage009i-rd-physical-v2-fullbudget-v1-18epochs/best_rd_only_drone_bird_model.pt`；仅 validation 选模，test=false。
- [x] validation 复评与 provenance 诊断：window Macro-F1=`0.7746`、parent Macro-F1=`0.8594`；相对 v3 的 `0.7346/0.7709` 明显提升。对齐 trajectory：window rescue=`94`、harm=`83`、net=`+11`，但 parent rescue=`7`、harm=`12`、net=`-5`。
- [x] `0.90+` 与融合门禁未通过：Drone P/R=`87.71%/75.99%`，Bird P/R=`66.67%/81.86%`；不运行 test、不启用 fusion、不做阈值调优或把原始 shape 作为部署 gate。
- [ ] 下一阶段先设计父轨迹稳定化的 RD-only 表征/训练单因素实验；必须针对 `31x360` Bird 和混合 `31x226+31x570` Drone 的负增益，且先证明 parent net 非负后再讨论任何融合。

## Stage 010 - 实时安全回退与部署压力验证

状态：待执行（依赖 Stage 009）

- [ ] 测量目标硬件上的 RD 预处理、模型、后处理、队列等待和端到端 P50/P95/P99。
- [ ] 注入 RD 缺失、迟到、乱序、错误关联、低覆盖和质量突变，验证强制 trajectory fallback。
- [ ] 验证多目标持续流下 buffer 上限、丢弃策略、显存峰值和稳定判决时延。
- [ ] 验收：安全回退零泄漏、P95/P99 满足部署 SLA，且 RD 路径不降低无 RD 场景能力。

## Stage 011 - 联合候选门禁与独立测试准备

状态：待执行（依赖 Stage 010）

- [ ] 在 validation 上冻结最佳配置、门控和阈值，报告 Drone/Bird P/R、全类指标、source-balanced、父轨迹 stable、Unknown/Clutter 与实时指标。
- [ ] 只有 Drone/Bird precision 和 recall 均达到 `95%`，且所有安全/稳定门禁通过，才批准一次 independent test。
- [ ] independent test 仅做最终报告；不得回流修改模型、数据、阈值或训练策略。
- [ ] 更新配置、报告、任务文档和复现命令；不追踪数据、artifact 或 `.anthor/`。
- [ ] 验收：形成可发布或明确不发布的可复现结论。

## 当前执行事项

- [x] Stage 001-002 已完成，RD 阻断原因已证实为数据对齐/覆盖/时长与训练调度，不是“RD 物理上无区分力”。
- [x] Stage 003-007 已完成：契约、原始资产审计、紧凑 anchor manifest、窗口 map 与 RD-only 链路均已验证。
- [x] 用户已批准；Stage 003-009A 的索引修复已完成。修复版 RD-only full-budget 已冻结，融合与 independent test 继续冻结。
- [x] Stage 009B 已达到待训练状态：全帧 canonical v3 资产、v3 anchor/map、训练 smoke 与相关回归均完成；下一步仅运行冻结的 validation-only 18-epoch RD-only 训练。
