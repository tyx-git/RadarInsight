# V27d 实验记录

## V27d-r1：冻结端点与校准人口修正 control

状态：训练、动态 calibration 和完整 L3 evaluation 已完成。r1 是内部 control，不具备正式 conformal coverage 或 deployment 资格。

### 实验身份

- 直接基线：L1 使用 V27 baseline `known_gate` epoch 14，L2 使用 V27a-r3 Main epoch 20，seed 42。
- 唯一主要变量：在冻结 L1/L2 分类端点上，接入独立 knownness/quality heads、修正 knownness population、确定性质量 eligibility 和动态 L3 校准流程。
- 数据范围：CQ08；`representation_fit`、`representation_development` 和 `calibration` 三个角色；未执行 OOF、跨源验证或 post-selection audit pool。
- 主干：Liquid V2 raw memory + pre-norm computation；r1 的 raw-scale 和 stable8 causal 增强关闭，相关 feature cache 字段置零。
- 分类端点未被训练改变：所有训练摘要和评估报告均为 `classification_logits_changed=false`；Unknown 不是 L2 分类类别。
- Manifest：`manifests/V27d-r1-manifest.json`，SHA-256 `6f09c077e204af8e577a91194e3ab0327cbff0a46e20e8d0878a76cfe4d70a3e`。

### 训练结果

#### L1 `known_gate`

- knownness 选中 epoch 2；development AUROC `0.896356`，AUPRC `0.991532`，support `501`（positive `465`，Unknown negative `36`）。
- development knownness accuracy `0.922156`，precision `0.927711`，recall `0.993548`，F1 `0.959502`。
- development checkpoint 安全选择点：Drone/Bird/Balloon/Clutter known recall 均 `1.000000`，Unknown rejection `0.000000`。
- quality 选中 epoch 1；development AUROC `0.707086`，support `505`（positive `501`，negative `4`）。

#### L2 `dbb_specialist`

- knownness 选中 epoch 16；development AUROC `0.933173`，AUPRC `0.994296`，support `487`（DBB positive `451`，Unknown negative `36`）。
- development knownness accuracy `0.749487`，precision `0.996979`，recall `0.731707`，F1 `0.843990`。
- development checkpoint 安全选择点：Drone recall `1.000000`、Bird recall `0.982301`、Balloon recall `1.000000`，Unknown rejection `0.277778`。
- quality 选中 epoch 29；development AUROC `0.995509`，support `505`（positive `501`，negative `4`）。

### 分类端点指标

以下是冻结端点在两个实际评估池上的分类指标。L1 的 `Target` 合并 Drone/Bird/Balloon；L2 只统计 DBB，Unknown 和 Clutter 不进入 L2 普通分类支持集。

#### Representation development

| 层/类别 | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| L1 Target | 453 | 0.9259259 | 0.9933775 | 0.9584665 |
| L1 Clutter | 14 | 0.7368421 | 1.0000000 | 0.8484848 |
| L2 DroneTarget | 228 | 0.9913043 | 1.0000000 | 0.9956332 |
| L2 BirdTarget | 114 | 0.7000000 | 0.9824561 | 0.8175182 |
| L2 BalloonTarget | 111 | 0.9652174 | 1.0000000 | 0.9823009 |

L2 在 development 的 DBB endpoint evaluated support 为 `453/505`；Bird 的 precision `0.700000` 是当前已知类与 Unknown/其他路由混合时的主要风险信号，不能只看 Bird recall。

#### Calibration pool

| 层/类别 | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| L1 Target | 223 | 0.9409283 | 1.0000000 | 0.9695652 |
| L1 Clutter | 9 | 0.8000000 | 0.8888889 | 0.8421053 |
| L2 DroneTarget | 113 | 0.9819820 | 0.9646018 | 0.9732143 |
| L2 BirdTarget | 56 | 0.6585366 | 0.9642857 | 0.7826087 |
| L2 BalloonTarget | 54 | 0.9629630 | 0.9629630 | 0.9629630 |

L2 calibration endpoint evaluated support 为 `223/247`。三类 DBB exact-class recall 均约 `0.963`；它不是 knownness route recall 的同义词，但反映了已知子类边界的风险，不能仅凭阈值修复。

### Knownness 与 quality 指标

| 池 | 任务 | Knownness support (正/负) | Knownness Precision/Recall/F1 | AUROC | AUPRC | Quality AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Development | L1 | 501 (465/36) | 0.927711/0.993548/0.959502 | 0.896356 | 0.991532 | 0.707086 |
| Development | L2 | 487 (451/36) | 0.996979/0.731707/0.843990 | 0.933173 | 0.994296 | 0.995509 |
| Calibration | L1 | 244 (229/15) | 0.942386/1.000000/0.970339 | 0.884425 | 0.991498 | 0.711749 |
| Calibration | L2 | 235 (220/15) | 1.000000/0.713636/0.832891 | 0.902727 | 0.993133 | 1.000000 |

质量头的 L1 负例只有 development `4`、calibration `3`，因此 L1 quality AUROC 不能被当作稳定的成熟度泛化证明。L2 quality 的高 AUROC 也不能替代 Unknown 排序验证。

### Dynamic calibration 与 L3

校准命令完成并生成 registry，但结果为：

- registry status：`unavailable`。
- calibration scope：`internal_exploratory`。
- Unknown support：`15`，正式要求：至少 `100`；当前每个 Unknown 约代表 `6.67pp`。
- L1 risk-constrained selection：`feasible=false`；在最低 conformal p 阈值 `0.1` 且保持 Clutter recall `8/9` 时，联合候选中已知目标 route recall 的最高最小值约为 `0.904545`，低于 `0.98` 安全线。
- L2 risk-constrained selection：`feasible=false`；在最低 conformal p 阈值 `0.1` 时，联合候选中三类已知 route recall 的最高最小值约为 `0.909091`，低于 `0.98` 安全线；该限制同时反映 knownness 排序和 conformal 单例集合，不等同于 L2 exact-class recall。
- 没有写入活动 L1/L2 阈值；没有启用固定 softmax 阈值或单一全局距离 gate。
- `deployment_eligible=false`，`formal_conformal_support=false`，`test_payload_access_count=0`。

L3 按设计 fail-closed：

| 池 | Parent | L3 action | Calibration not fitted | Low quality / immature |
| --- | ---: | ---: | ---: | ---: |
| Representation development | 505 | `PENDING_ABSTAIN=505` | 501 | 4 |
| Calibration | 247 | `PENDING_ABSTAIN=247` | 244 | 3 |

#### Representation development open-set action metrics

- Known eligible：`451`；DBB accepted `0`，Pending `451`，known route recall `0.000000`，因为 calibration 未拟合。
- Unknown eligible：`36`；strictly rejected `0`，error accepted `0`，Pending `36`。Unknown strict rejection `0.000000`，不能记作拒识失败或成功，只能记作未进入可用 L3。
- Clutter eligible：`14`；accepted `0`，Pending/non-Clutter action `14`。此处不能把 Clutter recall 写成模型分类 recall，L1 endpoint 的 Clutter recall 仍为 `1.000000`。
- 按标签的 L3 Pending：Drone `228`、Bird `114`、Balloon `111`、Clutter `14`、Unknown `38`。

#### Calibration pool open-set action metrics

- Known eligible：`220`；DBB accepted `0`，Pending `220`，known route recall `0.000000`，原因是 calibration 未拟合。
- Unknown eligible：`15`；strictly rejected `0`，error accepted `0`，Pending `15`。Unknown strict rejection `0.000000`，不计作拒识成功。
- Clutter eligible：`9`；accepted `0`，Pending/non-Clutter action `9`；L1 endpoint Clutter recall 是 `8/9=0.888889`。
- 按标签的 L3 Pending：Drone `113`、Bird `56`、Balloon `54`、Clutter `9`、Unknown `15`。

### 结论

1. r1 成功建立了可复核的三层 control：L1/L2 分类端点、独立 knownness/quality heads、质量优先和未校准 fail-closed 的 L3 语义均实际运行，Unknown 没有进入 L2 类别。
2. r1 没有产生可计入的 Unknown 拒识提升。L3 没有活动阈值，这是正确的安全结果，不是训练或评估流程完成的证明。
3. 当前首要瓶颈有两个且必须分开处理：calibration pool 的 Unknown support 只有 15，无法支持正式校准；同时 knownness/conformal 组合在 calibration 上无法满足 98% route-recall safety floor（L1 最大约 90.45%，L2 最大最小类约 90.91%）。L2 exact-class recall 约 96.3% 是相关风险背景，但不是该不可行性的唯一原因。
4. L2 knownness 的 development AUROC `0.933173` 与 Unknown rejection `27.78%` 说明有排序信号，但 calibration knownness recall `71.36%` 和 Bird precision `65.85%` 表明不能直接接入 L3。不能用 development 的单点指标替代跨池稳定性。
5. r1 未验证 raw-scale、stable8 causal 或 32 维 scorer 变量；这些仍属于后续实验设计变量。下一轮若继续，应先解决独立 Unknown/audit support 与风险约束可行性，并保持 V27a endpoint、数据分割和动作语义固定。

### 产物位置

- L1 训练日志：[V27d-r1-known_gate-seed42.log](runs/V27d-r1-known_gate-seed42.log)
- L2 训练日志：[V27d-r1-dbb_specialist-seed42.log](runs/V27d-r1-dbb_specialist-seed42.log)
- L1 head：[V27d-r1-known_gate-seed42-heads.pt](checkpoints/V27d-r1-known_gate-seed42-heads.pt)
- L2 head：[V27d-r1-dbb_specialist-seed42-heads.pt](checkpoints/V27d-r1-dbb_specialist-seed42-heads.pt)
- Dynamic registry：[V27d-r1-l3-registry.json](calibration/V27d-r1-l3-registry.json)
- Full evaluation：[V27d-r1-cascade-seed42-all-evaluation.json](reports/V27d-r1-cascade-seed42-all-evaluation.json)
- Contract smoke：[V27d-r1-contract-smoke.json](reports/V27d-r1-contract-smoke.json)

所有路径均相对于 `src-v3/v27/V27d`。

### 核查修正（2026-09-12）

- 本记录原先登记的 r1 manifest `6f09c077e204af8e577a91194e3ab0327cbff0a46e20e8d0878a76cfe4d70a3e` 是历史运行上下文的文件哈希；当前活动 `manifests/V27d-r1-manifest.json` 的实际 SHA-256 为 `37b0af2ccb57684027bf50ee5a9efa4a7a154c5dea97ca08622b040a52db11a5`，由 `load_context` 和当前 manifest 文件复核。
- 当前 `calibration/V27d-r1-l3-registry.json` 的 sealed payload 可独立验证，但其 `v27d_manifest_sha256` 仍为历史值 `6f09...`；当前代码加载时实际报 `V27d calibration registry context differs`。两个 r1 head checkpoint 也绑定历史 manifest `6f09...`，当前 `load_head_checkpoint` 实际报 `V27d head checkpoint provenance differs`。因此上文 head/L3 action 数字仅保留为历史 exploratory run 记录，不视为当前上下文下可重放的正式 L3 评估。
- 修正依据：`src-v3/v27/V27d/manifests/V27d-r1-manifest.json`、`src-v3/v27/V27d/calibration/V27d-r1-l3-registry.json`、`src-v3/v27/V27d/protocol.py` 与 `src-v3/v27/V27d/calibration.py`；未修改原始 JSON、summary、manifest、checkpoint 或 registry。

## V27d-r11：full stable8 causal vector shape projection

- 直接基线：同 seed 的 `V27d-r6-liquid_treatment`；Liquid V2、r6 base readout、verifier head 与 L1/L2 endpoint 冻结。
- 唯一变量：stable8 frame-vector 的当前事件、causal running mean/std、上一有效事件绝对差，共 32 维 shape projection；仅 `shape_projection.weight` 可训练。
- seed：42、44；Unknown 不进入 L2，正式 L3/conformal/deployment 关闭；calibration route-candidate Unknown 仍为 12，未达到 100。
- 结果：development FP rejection 相对 r6 为 `+12.50pp`（seed42）、`+6.25pp`（seed44），但各多拒绝 1–2 个 Bird；calibration Unknown rejection 两 seed 均保持 `2/12=16.67%`，没有动作收益。
- calibration Bird recall：seed42 `52/53=98.11%`，seed44 `51/53=96.23%`；seed44 额外拒绝 1 个 Bird，calibration AUROC 下降约 `0.42pp`。
- 判定：`diagnostic_only_no_promotion`；不接入默认 knownness 或 L3。冻结 r6 的逐前缀序列诊断显示 prefix lower-quantile aggregation 值得作为下一独立变量验证，但不能当作 r11 已验证收益。
- 详细记录：[r11 experiment.md](r11/experiment.md)，汇总：[V27d-r11-two-seed-summary.json](r11/reports/V27d-r11-two-seed-summary.json)。

## V27d-r12：causal prefix lower-quartile aggregation

- 直接基线：同 seed `V27d-r6-liquid_treatment`；唯一变量是对 frozen knownness 的 eligible causal prefix 分数取 P25。
- seed42/44 均使 calibration Unknown rejection 从 `2/12` 提升到 `3/12`，但 Bird recall 均降到 `51/53=96.23%`，低于产品 `98%` 安全线；calibration AUROC 平均下降约 `6.45pp`。
- development 仅 seed42 有拒识提升，seed44 不变；新增拒识还包含一个 DroneTarget route negative，说明不是纯 Unknown 分离。
- 判定：`diagnostic_only_no_promotion`；不接入 L3。当前同一 12 个 Unknown 上继续枚举 quantile/blend 缺乏证据，优先解决独立 Bird-like Unknown 来源/校准支持。
- 详细记录：[r12 experiment.md](r12/experiment.md)，汇总：[V27d-r12-two-seed-summary.json](r12/reports/V27d-r12-two-seed-summary.json)。活动 manifest SHA-256：`4084a065e2841bae3aa8e600334edecbfe515ce6b9e72ab3b1f6dfd5ce7cd90c`。

## V27d-r13：原始包络调制微多普勒形状分支

- 直接基线：冻结 `V27a-r3 Main` epoch 20；唯一变量是此前未被模型使用的 V18 `envelope_modulation_profiles` 64-bin profile shape。
- 两个 seed 均完成 120 epoch 训练；Liquid V2、L1/L2 基础端点和 Unknown/L3 语义未改变。
- Seed42：Drone/Bird/Balloon P/R/F1 分别为 `97.30/95.58/96.43%`、`88.33/94.64/91.38%`、`98.08/94.44/96.23%`；相对 baseline 总正确数 `-3`，accuracy `-1.35pp`。
- Seed44：Drone/Bird/Balloon P/R/F1 分别为 `97.30/95.58/96.43%`、`86.89/94.64/90.60%`、`98.04/92.59/95.24%`；相对 baseline 总正确数 `-4`，accuracy `-1.79pp`。
- 两 seed 都新增 `Drone→Bird` 和 `Bird→Drone`，没有 wrong→correct；没有类别的平均 P/R/F1 提升。平均 accuracy 从 `96.41%` 降至约 `94.84%`，下降约 `1.57pp`。
- 判定：`diagnostic_only_no_promotion`；原始 envelope profile 可被正确接入，但当前闭集 parent-final residual 目标无法可靠利用它，不能据此判定原始微多普勒形状无信息，也不能接入默认 L2/L3。
- 详细记录：[r13 experiment.md](r13/experiment.md)，汇总：[V27d-r13-two-seed-summary.json](r13/reports/V27d-r13-two-seed-summary.json)。
