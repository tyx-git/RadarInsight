# Findings - RD 修复计划基线

> 更新日期：2026-07-23  
> 范围：DroneTarget/BirdTarget 的 RD 数据、表征、训练与融合。所有数值来自 frozen v7 资产与 validation-only 审计；未运行 independent test。

## F001 - Trajectory 基线仍是唯一可用回滚点

Severity: High  
Evidence: `artifacts/20260722-222612/20260722-stage009-v7-trajectory-only-baseline-v1/best_final_macro_f1_model.pt`；validation stable Macro-F1=`0.7763`。  
Impact: 任何 RD 候选必须在同一父轨迹 split 上不损害稳定聚合和无 RD 回退。  
Action: Stage 001 已冻结比较协议。

## F002 - Stage 019 不能评价 RD 能力

Severity: High  
Evidence: `tmp/20260723-stage019-v8-sequence-parent-pair-fullbudget-v1.json` 设定 `epochs=12`、`stage1_traj_only_epochs=12`；训练历史 epoch 1-12 的 `stage2_enabled=false`、`stage2_scale=0`。  
Impact: RD Transformer、patch CNN、RD-only 分类、micro-Doppler、pairwise、residual、Joint 与门控损失均未获得训练梯度，最终 RD 融合权重为零。  
Action: Stage 007-008 在合法数据到位后独立训练 RD-only，不将 Stage 019 的 RD 指标解释为架构结论。

## F003 - 安全可用 RD 覆盖不足

Severity: Critical  
Evidence: `data/processed/expert1_rd_segment_manifest.csv` 共 `25021` 段，其中 `24986`（`99.86%`）为 `proportional_fallback`；v7 安全窗为 train=`25`、val=`6`、test=`4`。  
Impact: 当前没有足够独立的 Drone/Bird 父轨迹支持 RD 训练、校准或融合评估。  
Action: Stages 003-006 必须先恢复共享真实时间轴与严格窗口映射。

## F004 - 精确同步 RD 仍存在观测尺度缺口

Severity: High  
Evidence: exact RD 时长中位数=`0.1049s`；Drone=`0.0413s`、Bird=`0.1565s`；所有 exact 段均标记 `short_segment`；轨迹窗口为 `10s`。  
Impact: 单个短 RD burst 可以是局部散射证据，但不能被伪装成覆盖整段轨迹窗的连续微多普勒。  
Action: Stage 006 显式记录最长连续时长与覆盖率；Stage 009 的门控只能把短段作为受限残差证据。

## F005 - 比例时间映射不可安全用于监督或融合

Severity: Critical  
Evidence: CQ08 RD 以父轨迹相对时长比例截取；当前 `load_only_fusion_eligible_rd=true`、`rd_proportional_fallback_quality=0.0` 强制将其排除。  
Impact: 直接放开会把错误时间片的散射特征与轨迹标签绑定，可能制造来源/时序伪相关和不可重复的虚高指标。  
Action: 保持阻断；Stage 003 定义时间轴、偏移和目标关联契约。

## F006 - 当前 RD 主干具备候选能力，但辅助头过多且无数据支撑

Severity: Medium  
Evidence: `src/core/experts.py` 有 4 通道 RD token Transformer、16 维 RD 统计、可选 shallow/deep patch CNN、6 维 micro-Doppler、RD Bird/Unknown/Clutter/residual heads、Joint head 与可靠性门控。  
Impact: 合法样本稀少时同时训练多头将显著放大过拟合风险，且无法定位真正有效的 RD 特征。  
Action: Stage 007 做最小表征消融，先仅保留 Drone/Bird RD-only 目标，再根据证据引入必要分支。

## F007 - RD 的合理价值应是受质量约束的 Drone/Bird 残差

Severity: Medium  
Evidence: 物理上旋翼调制、翼拍时变谱、Doppler 扩展与散射闪烁可为 Drone/Bird 提供补充；当前 trajectory-only Drone P/R=`90.11%/83.15%`、Bird P/R=`74.33%/85.56%`，存在 hard-pair 空间。  
Impact: RD 应用于纠正有 RD 且 TR 不确定的 Drone↔Bird 样本，而非无条件覆盖高置信 trajectory 分类。  
Action: Stage 008 先量化 RD rescue/harm；Stage 009 使用 trajectory anchor + gated residual。

## F008 - 95% 是发布门槛，不是当前结果

Severity: High  
Evidence: 当前任何验证候选均未同时达到 Drone/Bird precision>=`95%` 与 recall>=`95%`。  
Impact: 不能因 RD 设计预期而宣称达到或接近部署要求。  
Action: Stage 011 以 frozen validation 门禁判定是否允许一次独立 test。

## 阶段发现状态

| 阶段 | 发现状态 | 结论 |
|---|---|---|
| Stage 001 | 已完成 | 轨迹基线与评估协议已冻结。 |
| Stage 002 | 已完成 | RD 阻断根因已量化。 |
| Stage 003 | 已完成 | 已落地 strict pairing contract 与拒绝审计。 |
| Stage 004 | 已完成 | CQ08 MAT 的 point association 已全量审计。 |
| Stage 005 | 已完成 | 生成紧凑 point-anchor manifest，不复制 RD 张量。 |
| Stage 006 | 已完成 | anchor 因果窗口 map 可读但融合资格保持为零。 |
| Stage 007 | 已完成 | RD-only Transformer+micro-Doppler 诊断链路已 smoke 验证。 |
| Stage 008 | 已完成（失败） | RD-only 在对齐 validation 上净损，融合继续阻断。 |
| Stage 009 | 阻断 | Stage 008 未通过 RD-only rescue/harm 门禁；先完成 009A 索引修复复训。 |
| Stage 010 | 待执行 | 需要部署级安全回退和压力验证。 |
| Stage 011 | 待执行 | 需要最终 validation 门禁后才可准备 test。 |

## F009 - CQ08 MAT 帧具备同轨迹点关联，但没有独立绝对时钟

Severity: High  
Evidence: 全量扫描 `OriData/23/CQ-08/MAT` 的 `106026` 个 MAT 帧（耗时 `526.9s`）均可读取 `index_trackPointNo`；与轨迹点数核对后，全部索引在合法边界内，`1505/1549` 轨迹覆盖全部轨迹点。扫描产物：`tmp/20260723-stage004-cq08-mat-trackpoint-scan.json`、`tmp/20260723-stage004-cq08-trackpoint-coverage.csv`。  
Impact: 现有 CQ08 `time_raw=0..N-1` 确实只是标准化占位帧序号，不能用于比例切分；但每个 RD MAT 帧可关联到同一目标的轨迹点及该点相对时间，可重建为带明确时间不确定度的 point-anchored RD 索引。  
Action: Stage 005 输出独立 v2 store/index，记录 `track_point_index`、`track_point_time_seconds`、锚定不确定度和 `track_point_anchor` 同步状态；此状态仅可先用于 RD-only 诊断，不能直接视为 exact fusion evidence。

## F010 - CQ08 单帧 CPI 时长可由 MAT 元数据估计，但帧间采集时刻仍不可证实

Severity: Medium  
Evidence: 抽样 MAT 的 `CPIcount=2`、`PRTnum=226`、`PRT=20us`，单帧 CPI 约 `9.04ms`；同一轨迹点可对应多个 MAT 帧。MAT 文件名和 `index_trackPointNo` 不提供帧级 wall-clock。  
Impact: 可将单 RD 帧视作短时局部微多普勒观测，并以轨迹点时间作为关联锚点；不能虚构多帧间连续绝对时间或把锚点覆盖率当作物理连续覆盖率。  
Action: v2 manifest 使用 point-anchor coverage 与最大锚点间隔，Stage 007 先进行 RD-only 表征/类别诊断；融合仍需额外时钟证据或保守门控。

## F011 - 全量 shard 重建因重复编码膨胀失败，紧凑引用 manifest 已替代

Severity: High  
Evidence: 首次 CQ08 anchor store 将每个锚定点重新编码并保存到父轨迹 shard；运行 `2026-07-23 01:22:01–01:47:35`（`1533.6s`）后 F 盘写入失败。已确认其产生 `1002` 个不完整文件、`267.23GiB`，并仅安全删除这个本次生成且未完成的目录。  
Impact: 对 `29413` 个 anchor 重复持久化 512×4 RD 特征会超出本机存储预算，且没有增加原始证据。  
Action: Stage 005 改为紧凑 manifest，仅保存 `rd_tensor_path`、源帧索引、轨迹点锚定和质量元数据；全量重建成功，未复制张量。

## F012 - CQ08 point-anchor 为 RD-only 提供足够父轨迹覆盖，但仍不是融合许可

Severity: High  
Evidence: `expert1_cq08_rd_anchor_manifest_v1.csv` 含 `29413` anchor 段、`106026` 源帧引用、`1549` 父轨迹；anchor v7 map 的 Drone/Bird 为 train=`2718` windows / `510` parents、val=`641` / `110` parents、test=`572` / `114` parents。  
Impact: 覆盖已足以评估 RD 表征对 Drone/Bird 的诊断判别能力；但帧间 absolute time 仍未知，窗口 realtime status 被刻意解析为 `unavailable`，因此不能把诊断 RD 用于 Joint 融合。  
Action: 完成 Stage 008 RD-only validation 后仅依据救援/伤害与不确定性决定是否研究更强时间证据；Stage 009 继续阻断。

## F013 - RD-only smoke 已验证训练链路，不构成模型能力证据

Severity: Medium  
Evidence: `artifacts/20260724-022003/20260723-stage007-rd-only-anchor-drone-bird-smoke-v2` 使用 `32` train / `32` balanced validation windows，window Macro-F1=`0.3333`。相关测试 `31 passed`。  
Impact: 随机初始化和极小训练预算不足以判断 RD 判别上限；该指标不能替换任何 trajectory 基线或触发融合。  
Action: 使用已冻结的 8 epoch、2718/641 window 全预算 RD-only 诊断配置，选择与判定仅使用 validation。

## F014 - Point-anchor RD-only 在 Drone/Bird 验证上显著伤害 Bird，不能进入残差融合

Severity: Critical  
Evidence: 对 `artifacts/20260724-122359/20260723-stage008-rd-only-anchor-drone-bird-fullbudget-v1/best_rd_only_drone_bird_model.pt` 的独立 validation 复评（`641` CQ08 Drone/Bird windows、`110` parents）为 window Macro-F1=`0.5250`、parent mean-logit Macro-F1=`0.5317`。Drone P/R=`65.06%/80.20%`，Bird P/R=`44.06%/26.58%`。  
Impact: RD-only 预测分布偏向 Drone（`498/641`）；它无法可靠恢复 Bird，而不是可安全叠加在 trajectory 上的补充证据。  
Action: Stage 009 保持阻断；不调高 RD 权重、不校准融合阈值、不运行 test。

## F015 - 对齐比较证明 RD-only 的救援远小于伤害，且父轨迹退化更严重

Severity: Critical  
Evidence: `tmp/20260724-stage008-rd-only-vs-trajectory-rescue-harm.json` 将 RD-only 与 frozen seed-42 trajectory report 的同 `641` 个样本精确对齐（ID/parent/truth/source 全一致）。窗口 rescue=`66`、harm=`169`、net=`-103`；其中 `142` 个原本 trajectory 正确的 Bird 被 RD-only 改为 Drone。父轨迹 majority-vs-RD-mean rescue=`4`、harm=`34`、net=`-30`，trajectory accuracy=`90.91%` 对 RD=`63.64%`。  
Impact: 即使只在有 point-anchor RD 的目标上，RD 也不能作为 residual gate 的候选；门控无法从一个总体负增益专家中安全获得发布增益。  
Action: 只有获得独立帧级时间轴/连续 RD 证据并重新运行 RD-only 训练后，才重新考虑 Stage 009；当前回退 trajectory-only baseline。

## F016 - RD Bird margin 未形成正确的 Bird 方向分离

Severity: High  
Evidence: 在真实 Bird 上，RD bird-minus-drone margin 均值为：trajectory 正确 Bird=`-0.0911`、trajectory Unknown Bird=`-0.0743`，即 RD 仍偏向 Drone；救援/伤害样本的 RD margin 中位分别为 `-0.0936/-0.1415`，无可用的置信边际分离。  
Impact: 不能通过简单阈值或低置信门控修复 Stage 008；现有表示/时间锚定下的 Bird 方向证据本身不可靠。  
Action: 后续仅研究新的独立时间证据、连续帧组织或更强物理校准；不在当前 anchor 数据上继续堆叠辅助头。

## F017 - v1 anchor 将原始 MAT 帧号误用为压缩 tensor 帧号，造成 RD/TR 对齐错配

Severity: Critical  
Evidence: `standardize_cq08.py` 为同一父轨迹保留首个 RD shape，并跳过后续 shape 不同的 MAT 帧；v1 anchor 却以 `enumerate(rd_mat_files)` 的原始序号访问压缩 `.npz`。全量帧索引审计发现原始 MAT=`106026`，保留 tensor 帧=`88505`，跳过=`17521`，涉及 `535` 父轨迹；旧 `29413` anchor 段中 `5886` 段引用了被跳过的原始帧位置。  
Impact: v1 RD-only 并非稳定地读取其 `index_trackPointNo` 所属的同一 RD 帧，Stage 008 的负增益不能用于评价 RD 物理判别能力或制定融合策略。  
Action: 生成 `cq08_rd_frame_index_v1.csv`，并使用 v2 anchor 清单将原始 MAT 序号和压缩 tensor 序号分离；重新进行 RD-only full-budget validation。

## F018 - v2 anchor 逐帧引用完整，安全融合资格仍为零

Severity: High  
Evidence: `tmp/20260724-stage009-v2-anchor-frame-reference-validation.json` 验证 v2 的 `88505` 保留 tensor 帧：raw/tensor 引用均为零漏引、零越界、零额外引用。v2 map 共映射 `7843/9236` 窗口且 `rd_fusion_eligible_window_count=0`。  
Impact: 修复了 RD-only 诊断输入的内容一致性，但没有凭空补出独立帧级 wall-clock；`track_point_anchor` 仍只能诊断，不能融合。  
Action: 使用 v2 config 复训 RD-only；仅当 validation rescue/harm 与父轨迹门禁通过后才重新讨论融合。

## F019 - v2 smoke 只证明修复后的读取与优化闭环

Severity: Medium  
Evidence: v2 balanced smoke 使用 `64` train / `64` validation windows，Macro-F1=`0.3333`；`35 passed`。  
Impact: 随机初始化和单 epoch 小预算没有能力结论，既不能宣布 RD 有效也不能否定 v2 RD-only。  
Action: 冻结 8 epoch、v2 map、train=`2620` / val=`618` window 的全预算诊断配置；test/fusion 继续关闭。

## F020 - RD-only 训练无输出源于日志缺失，不是已证实的数据加载死锁

Severity: High  
Evidence: 原 `rd_drone_bird_trainer.py` 只在全部 epoch 完成后由 CLI 输出 JSON，训练循环内没有 batch/epoch print，也只在最后写入 history/summary。用户中断的 v2 artifact 仅含 epoch=`3` 的 best checkpoint、没有 history/summary，符合“训练继续/中断但无可见进度”的表现。  
Impact: 无法区分首批 RD 读取、GPU 前向、反向或验证阶段的长耗时，且中断会丢失已完成 epoch 的过程指标。  
Action: 增加 READY、首 batch、固定 batch 间隔、epoch START/END 的强制刷新日志；每 epoch 立即写入 `training_progress.json` 与 `training_history.json`。

## F021 - 新日志 smoke 证明训练过程可观测，未改变 RD 融合安全边界

Severity: Medium  
Evidence: `tmp/20260725-stage009-rd-only-logging-smoke-v1.json` 运行 `64` train / `32` validation batch，首训练 batch=`0.80s`，每 `8` batch 输出 loss/elapsed/ETA，完整 epoch 记录写入 `artifacts/20260725-010028/20260725-stage009-rd-only-logging-smoke-v1`。  
Impact: 重启全预算训练后可立即观察实际进度并保留中断前 epoch 结果；这不是模型性能改善实验。  
Action: 使用 v2 帧索引修复配置重新启动 Stage 009A 全预算 RD-only；继续禁止 test 和融合。

## F022 - 修复版 RD-only 需要更长训练预算，但必须按 validation checkpoint 而非末轮选模

Severity: Medium  
Evidence: v2 RD-only 每 epoch=`655` train batch；原 `8` epoch 仅=`5240` optimizer updates。此前 trajectory 全预算的最佳 epoch 约为 `18`，因此修复帧索引后用 8 epoch 直接否定 RD 表征预算不足。  
Impact: epoch 数无法跨任务直接比较，但在修复输入错配后，8 epoch 只适合作为保守预算，不足以得出最终 RD-only 能力结论。  
Action: 冻结独立 18 epoch 配置；每 epoch 在 validation 上保存最佳 checkpoint，最终只评估 best checkpoint，不以第 18 轮末模型替代最佳模型；test/fusion 继续禁止。

## F023 - v2 帧索引正确后，旧 CQ08 标准化仍丢弃异形 RD 帧

Severity: High  
Evidence: `data/processed/cq08_rd_frame_index_v1.summary.json` 显示 `106026` 个原始 MAT 帧仅 `88505` 个进入旧 tensor，`17521`（`16.52%`）因父轨迹首帧 shape 不同被 `standardize_cq08.py:_load_rd_stack` 跳过。v2 修复了 raw/tensor frame index 引用，但不能恢复已删除的物理 RD 证据。  
Impact: Drone/Bird 窗口可能失去不同 Doppler 宽度或距离门配置下的关键片段；RD-only 序列仍不完整，不能据此解除融合阻断。  
Action: Stage 009B 建立新的全帧 canonical `31x64` RD store；幅度双线性重采样，相位使用 sin/cos 圆周重采样后还原，再由 v3 anchor 引用连续 tensor index。

## F024 - Canonical v3 trainable smoke 保留全部帧且不改变融合安全边界

Severity: Medium  
Evidence: tracks `276/277/281` smoke 生成 `310/310` canonical 帧、`52` anchors、`36` realtime windows，覆盖 `31x226`、`31x360` 与 `31x570` 源 shape。所有窗口 `rd_realtime_sync_status=unavailable`，`rd_fusion_eligible_window_count=0`。  
Impact: 全帧保存、锚点索引、RD loader/encoder 与实时窗口映射均可用，但 point-anchor 没有独立连续时钟，不能作为 Joint/final fusion 依据。  
Action: 完成全量 v3 后仅执行 RD-only validation 诊断与对齐 rescue/harm；没有独立时序证据和净增益门禁，不解除 trajectory fallback。

## F025 - 全量 Canonical v3 已消除 shape-drop，但尚未证明分类增益

Severity: High  
Evidence: `cq08_rd_canonical_frame_index_v1.summary.json`：`1549` 父轨迹、`106026/106026` 帧、`0` shape-drop、`0` error。v3 manifest 具有 `29413` anchors，realtime map 映射 `8240` 个窗口；v3 RD-only smoke 成功加载 `2718` train 和 `641` validation windows。  
Impact: 数据完整性、索引连续性和 RD-only 可训练性已建立；但 v2 在同类 validation 对齐比较中仍是 rescue=`54`、harm=`134`，因此该修复不能被视为已达 95% 精度/召回或可融合的证据。  
Action: 仅运行冻结的 v3 18-epoch RD-only 训练；使用 validation best checkpoint 复评并与冻结 trajectory 做 rescue/harm。independent test、Joint/final fusion 和阈值调优继续禁止。

## F026 - Canonical v3 显著改善 RD-only，但仍未通过 trajectory 净增益门禁

Severity: High  
Evidence: checkpoint `artifacts/20260725-180452/20260725-stage009b-rd-only-anchor-v3-canonical-fullbudget-v1-18epochs/best_rd_only_drone_bird_model.pt` 的 validation best epoch=`3`。RD-only window Macro-F1=`0.7346`，Drone P/R=`80.15%/80.94%`，Bird P/R=`66.95%/65.82%`，parent mean-logit Macro-F1=`0.7709`，均优于 v2 的 `0.6198/73.47%/69.77%/50.21%/54.75%/0.6596`。在冻结 trajectory 的同一 `641` 个 validation windows 上，RD rescue=`99`、harm=`106`、net=`-7`；父轨迹 rescue=`7`、harm=`21`、net=`-14`。  
Impact: 全帧 canonical 表征确实修复了大部分 v2 的 RD 信息损失（v2 window net=`-80`，v3=`-7`），但 RD 单独替换 trajectory 仍略差，且 Bird 精度/召回远低于 `95%`。v2/v3 validation 窗口数不同（`618` vs `641`），因此仅将其视为方向一致的表征改善，不把跨集合差值当作严格显著性结论。  
Action: 继续阻断 Joint/final fusion、independent test 与阈值调优。下一阶段应仅在 validation 上做受控的 RD 表征误差分解：按父轨迹、源 RD shape、连续帧长度、Bird/Drone 分组定位 residual harm；任何融合候选必须先达到 window rescue>harm 且 parent net gain 非负。

## F027 - v3 残余 harm 集中在 Bird 的 `31x360`，而二维 RD patch 主干未参与最优训练

Severity: High  
Evidence: `tmp/20260725-stage009b-v3-residual-provenance-diagnostic.json` 对齐 `641` 个 validation windows，Bird 总 net=`-36`，其中 `BirdTarget|31x360` 为 `86` 窗口、rescue=`1`、harm=`30`、net=`-29`，trajectory accuracy=`97.67%` 而 RD=`63.95%`。所有 map 的选帧索引密度均 `>0.95`，因此不是 anchor 帧稀疏主因。训练配置 `tmp/20260725-stage009b-rd-only-anchor-v3-canonical-fullbudget-v1-18epochs.json` 设定 `use_rd_patch_cnn=false`；此时 `_encode_rd_patch` 返回零维向量，分类主干只接收经 range/Doppler 投影和池化后的序列特征，未训练二维 `64x64` RD patch 纹理路径。  
Impact: canonical v3 修复了帧丢失，但主干仍可能把不同原始 Doppler 宽度归一化后的局部微多普勒结构过度压缩；这与 Bird `31x360` 残余混淆一致，但尚不是因果证明。  
Action: Stage 009C 先仅切换 `use_rd_patch_cnn=true` 做 RD-only CUDA smoke，验证显存、日志、validation loader 和 patch 路径；通过后冻结与 v3 相同数据/seed/优化器的单因素 full-budget ablation。继续禁止 fusion、阈值调优和 independent test。

## F028 - Patch CNN 在正式 batch size 下可训练，但显著提高训练成本

Severity: Medium  
Evidence: `tmp/20260725-stage009c-rd-only-anchor-v3-patchcnn-batch4-smoke-v1.json` 在 RTX 4060 Laptop GPU 上以 batch=`4` 完成训练/validation smoke，无 OOM；首 train batch=`3.53s`，后续 train batch 约 `1.32–1.81s`，validation batch 约 `0.58–0.78s`。这仅验证执行链路，smoke 的 Macro-F1=`0.3333` 不构成能力比较。  
Impact: 18 epoch 全预算约 `680` train 和 `161` validation batch/epoch，训练耗时将明显高于非 patch v3；但二维空间纹理是当前 Bird `31x360` 残余误判的唯一已证实未启用的主干能力，值得作为单因素对照。  
Action: 使用冻结的 batch=`4`、18 epoch patch-CNN 配置训练；只在 validation 以 best checkpoint 复评。若没有同时实现 window rescue>harm 与 parent net>=0，则不进入融合。

## F029 - Patch CNN 改善 Bird `31x360`，但以 Drone 召回和整体净增益为代价

Severity: High  
Evidence: patch-CNN checkpoint `artifacts/20260726-003011/20260725-stage009c-rd-only-anchor-v3-patchcnn-fullbudget-v1-18epochs/best_rd_only_drone_bird_model.pt` 的 best epoch=`12`。window Macro-F1=`0.7325`（非 patch v3=`0.7346`），Drone P/R=`89.53%/65.59%`（v3=`80.15%/80.94%`），Bird P/R=`59.71%/86.92%`（v3=`66.95%/65.82%`）。同一 `641` 窗口的 rescue=`73`、harm=`92`、net=`-19`；parent rescue=`5`、harm=`14`、net=`-9`。Bird `31x360` 改善为 rescue=`0`、harm=`12`、net=`-12`（非 patch 为 `1/30/-29`），但 Drone harm 从 `35` 增至 `68`，且 Drone `31x570` net 从 `-10` 变为 `-14`。  
Impact: 二维 patch 包含有用的 Bird `31x360` 局部信息，但当前端到端拼接使主分类器偏向 Bird，不能作为整体 RD-only 替代，更不能进入联合融合。  
Action: 不再单独扩大 patch CNN 或调阈值。下一阶段必须保留已验证的 sequence RD 主干，并将 patch 支路限制为可审计的 residual correction；仅接受 validation 上同时改善 Bird `31x360` 且不降低 Drone recall、window rescue>harm、parent net>=0 的候选。

## F030 - Patch checkpoint 的 evaluator 已修复并与训练路径一致

Severity: Medium  
Evidence: 初次复评失败，因为 `evaluate_rd_drone_bird.py` 未传递 `rd_patch_features` 与 `rd_patch_padding_mask`，将 128 步 RD mask 错用于 4 步 patch。修复后真实 checkpoint 的 validation-only 复评精确复现训练 checkpoint 指标，输出 `tmp/20260726-stage009c-patchcnn-18epochs-rd-only-validation-recheck.json`。  
Impact: 之后任何 patch-CNN checkpoint 均可按训练等价路径复评，避免将 evaluator 形状错误误判为模型失效。  
Action: 保留 patch 参数转发；相关单元回归与真实 checkpoint validation 均已通过，test/fusion 继续禁止。

## F031 - 达成 RD `0.90+` 不能靠 patch 直接拼接或阈值补偿

Severity: High  
Evidence: non-patch v3 的 Drone P/R=`80.15%/80.94%`、Bird P/R=`66.95%/65.82%`；patch-CNN 将 Bird recall 提升至 `86.92%`，但 Drone recall 降至 `65.59%`，整体 window net 从 `-7` 退化为 `-19`。两个模型均未让 Drone 与 Bird 的 precision、recall 同时达到 `0.90`。  
Impact: 直接把 patch 特征拼入主分类器会产生类别偏置迁移，单纯调阈值只会在 precision/recall 间重新分配错误，无法满足严格的四项 `0.90` 门禁。  
Action: Stage 009D–009H 冻结为 sequence 主干保持、patch 仅做有界 residual correction 的单因素路线；每个候选必须同时报告全窗口、父轨迹、Bird `31x360` 与 Drone `31x570`，且不允许降低 Drone recall。

## F032 - 高分辨率物理坐标 canonical v2 是 residual 路线失败后的后备方向

Severity: Medium  
Evidence: 原始 CQ08 RD 宽度存在 `142/226/360/570/900` 五种 Doppler shape，而当前 canonical v1 全部压缩到 `31x64`；patch-CNN 即使启用，也只接收已压缩后的 `64x64` patch。Bird `31x360` 受益于二维纹理但仍未达门禁，表明更高分辨率或物理坐标对齐可能必要。  
Impact: 在未确认 debug metadata 能否恢复统一 Doppler 物理坐标前，直接扩大网络或把 shape 当作分类捷径会增加采集域泄漏风险。  
Action: 不立即重建数据。只有 Stage 009H residual 对照仍未通过净增益门禁时，才审计 `debug_info` 的 Doppler 采样间隔/速度范围，并决定是否构建带抗混叠、保留 resolution metadata 的 high-resolution canonical v2。

## F033 - Frozen-sequence patch residual 已满足零改写与无效 patch 安全契约

Severity: High  
Evidence: `FrozenSequencePatchResidual` 从 non-patch v3 checkpoint 加载并冻结全部 base 参数；真实 CQ08 RD batch 验证 `force_zero_residual` 时 final logits 与 base logits 逐元素相等，全 patch padding 时 residual delta 为零且 final logits 同样精确复现 base。batch=`4` smoke 完成 `8` train/`8` validation batch，无 CUDA OOM，训练器/evaluator 均使用相同 patch features/mask。  
Impact: 新支路无法在零 residual、patch 缺失或无效时破坏已验证的 sequence RD 主干；这消除了 Stage 009C 直接拼接 patch 时主干被整体偏置重写的结构性风险。  
Action: 冻结 `tmp/20260726-stage009g-frozen-sequence-patch-residual-fullbudget-v1-18epochs.json`，仅训练 residual branch；完成后只在 validation 比较全局、Bird `31x360`、Drone `31x570` 与 trajectory rescue/harm。

## F034 - Frozen-sequence residual 进一步收窄 window 误差，但 gate 饱和且未通过 Stage 009H

Severity: High  
Evidence: checkpoint `artifacts/20260726-235348/20260726-stage009g-frozen-sequence-patch-residual-fullbudget-v1-18epochs/best_rd_patch_residual_model.pt` 的 best epoch=`9`，window Macro-F1=`0.7504`，Drone P/R=`83.83%/76.98%`，Bird P/R=`65.56%/74.68%`。与 frozen trajectory 对齐 `641` validation windows：rescue=`100`、harm=`102`、net=`-2`；parent rescue=`8`、harm=`21`、net=`-13`。Bird `31x360` net 从 non-patch 的 `-29` 改善到 `-18`，但 Drone `31x570` 从 `-10` 退化到 `-17`。residual mean gate=`0.866`、P90=`0.9999`，mean absolute delta sum=`1.52`，P90=`2.00`，说明当前 gate 大多数时间近饱和，残差已不再保守。  
Impact: residual 契约和回退安全性成立，但训练目标仍使 patch 对 base 进行大范围重写；它没有达到 window rescue>harm、parent net>=0、Drone recall>=`80.94%` 或 Drone/Bird P/R>=`0.90` 的任何发布/融合门禁。  
Action: 拒绝 Stage 009G 作为融合候选，不扩大 residual scale 或以阈值补救。按 F032 转入只读物理 Doppler 坐标和 high-resolution canonical v2 可行性审计；在审计完成并更新计划前，不重建数据或训练新模型。

## F035 - CQ08 `debug_info` 可精确恢复每帧物理 Doppler 坐标

Severity: High  
Evidence: `src/process/audit_cq08_rd_physical_coordinates.py` 分三段全量审计 `106026/106026` 个原始 MAT，报告 `tmp/20260727-stage009i-cq08-rd-physical-coordinate-audit.json`。所有帧具有同一组 `freq_hz`、`prt_num`、`prt_s`、`rd_result_size`、`vr_start`、`vr_end` 字段，零解析错误、零 metadata shape mismatch、零 `prt_num`/Doppler-width mismatch。按设备采用的 `c=3.0e8`，`(vr_end-vr_start)/(N-1)` 与 `2*(c/(4*freq_hz*prt_s))/prt_num` 最大相对误差为 `2.124e-16`。五种宽度的速度 bin 分辨率分别为：`31x142=2.0905–2.4184`、`31x226=0.7984–2.1273`、`31x360=0.5138–0.5564`、`31x570=0.3245–0.3514`、`31x900=0.2055–0.2226 m/s`；全部帧的共同无外推支撑为约 `[-90.2201, 89.4217] m/s`。  
Impact: canonical v1 的 index-space `31x64` 缩放混合了不同物理速度分辨率和范围，确有表征失真风险；现可按每帧真实速度重采样，而非把原始 shape 作为分类捷径。该证据仅证明空间物理坐标可恢复，不能解决帧间绝对时间未知，也不能解除 RD→fusion 阻断。  
Action: high-resolution canonical v2 具有数据依据，但须在用户确认后先固定公共速度范围、Doppler bins、抗混叠规则、裁剪/out-of-support mask 和 v4 manifest 契约；保持 train/validation-only、test=false、fusion=false。

## F036 - 物理坐标 canonical v2 前三分区零丢失，最终分区应在用户端完成

Severity: High  
Evidence: 已实现 `src/process/build_cq08_rd_physical_canonical_store.py`：Doppler 只按每帧 `vr_start/vr_end` 重采样到 `[-85,85] m/s` 的 `384` bins（`0.443864 m/s/bin`），range 保持 31 bins；phase 采用 sin/cos 圆周插值，source bin 更细时才 Gaussian anti-alias。`3` parent smoke 写入 `308/308` 帧、零错误；`31x570/900` 的 anti-alias sigma 分别为 `0.1315/0.4971`。全量前三父轨迹分区共完成 `78059/106026` 帧，三段均零错误，抗混叠帧数合计 `21149`。`src/core/data/rd.py` 对带 `velocity_mps` 的 v2 tensor 保留原始 `384` Doppler bins 供主序列投影，而旧数据仍维持 `64` bins 行为。  
Impact: 新表征不会将不同物理速度范围压到相同 index space，也避免载入时再次丢失 high-resolution 信息；但最后 `27967` 帧若因 CLI 超时未完整重建，不能创建 v4 anchor/map 或启动任何训练。  
Action: 用户手动完成父轨迹 `[1149,1549)` 的第 4 分区，随后用 `merge_cq08_rd_physical_canonical_index.py` 对原始 `106026` 个 `(parent_sample_id, raw_frame_index)` 做集合级零缺失/零额外校验，并确认每父轨迹 tensor index 连续、`31x384`、覆盖率 100%；通过后才生成 v4 anchor/map。 

## F037 - 物理 v2 RD 已真实进入训练链路，但仍维持诊断隔离

Severity: High  
Evidence: 用户完成第 4 分区和合并：`106026` 帧、`1549` parents、reference 集合零缺失/零额外、零重复、零 tensor-index mismatch、零缺 tensor、全部 `31x384`、零 partial velocity coverage，anti-alias=`33584`。v4 生成 `29413` anchor segments 与 `8240` causal windows，`rd_fusion_eligible_window_count=0`。初次真实 batch 回退为占位 RD，定位到 `WindowAssembler._slice_anchor_tensor` 将静态 `velocity_mps` 当作时间序列切片；修复后真实 batch=`[4,128,512,4]`、`rd_is_multimodal=true`、全部 finite、sync=`unavailable`。CUDA smoke 的 `4` train/`4` validation batch 均为 RD length=`128`，训练、validation、日志与 checkpoint 正常，test=false。  
Impact: physical coordinate v2 的判别价值现可在与 v3 相同的 RD-only validation 协议下被公平检验，且不会再发生高分辨率 store 在窗口切片阶段静默失效。物理坐标仍不能证明 RD/TR frame-level wall-clock 同步，因此不能进入 fusion。  
Action: 仅运行冻结的 `tmp/20260727-stage009i-rd-physical-v2-fullbudget-v1-18epochs.json`，由 validation 选模；训练结束后先做 checkpoint validation 复评、对齐 trajectory rescue/harm 与原始 shape 分组。只有满足原有严格门禁才讨论后续表征，test、threshold 与 fusion 继续禁止。

## F038 - 物理速度对齐首次使 RD-only 窗口净增益转正，但父轨迹稳定性仍不足

Severity: High  
Evidence: checkpoint `artifacts/20260727-145406/20260727-stage009i-rd-physical-v2-fullbudget-v1-18epochs/best_rd_only_drone_bird_model.pt`（best epoch=`7`）的 validation-only 复评在 `641` windows / `110` parents 上达到 window Macro-F1=`0.7746`、parent Macro-F1=`0.8594`，优于 canonical v3 的 `0.7346/0.7709` 和 frozen patch residual 的 `0.7504/0.7869`。Drone P/R=`87.71%/75.99%`，Bird P/R=`66.67%/81.86%`。与冻结 trajectory 对齐，window rescue=`94`、harm=`83`、net=`+11`，但 parent rescue=`7`、harm=`12`、net=`-5`。按 provenance：`31x142` Drone 为 net=`+28`，Bird `31x226` 为 `+16`，Bird `31x570` 为 `+7`；主要负组为 Bird `31x360`=`-17`、Drone `31x226+31x570`=`-8`、Drone `31x570`=`-10`。  
Impact: 物理坐标重采样是此前 RD-only 退步的重要根因之一，现已证明其对窗口判别有效；但父轨迹层持续翻转说明逐窗口训练/均值聚合没有约束跨窗口一致性。任何直接 fusion 会把 `83` 个 window harm 和 `12` 个 parent harm 带入部署，仍不安全。  
Action: 保留 physical v2 作为唯一 RD-only baseline；不回退到 v3、不调阈值、不使用原始 shape 作为部署特征。下一实验必须是仅针对 parent-stability 的单因素 RD-only 训练/聚合设计，并预先要求 parent net>=`0`、不降低 Drone recall、且继续满足全窗口 rescue>harm，之后才考虑更强二维表征或融合。
