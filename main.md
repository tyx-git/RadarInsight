# RadarInsight 当前架构说明

本文档记录当前代码已经落地的架构，而不是早期设想稿。旧版方案已备份为 `main.bak001`。

当前主线目标不是直接追求口号式 95%，而是把系统稳定推进到可解释、可复现、可调优的多专家雷达分类系统：先确保数据切分无泄漏、训练链路一致、专家接线正确，再逐步提升 Expert1 RD、Ground、Aircraft 和跨源泛化能力。

## 1. 项目定位

RadarInsight 面向低空安防和雷达目标识别，输入来自统一标准化后的航迹、点迹和 RD 数据，输出全局大类标签。

当前全局标签体系：

| 标签 | 说明 |
|---|---|
| `AircraftTarget` | 常规空中目标 |
| `BalloonTarget` | 气球 |
| `BirdTarget` | 鸟 |
| `CarTarget` | 车辆 |
| `ClutterTarget` | 杂波/干扰 |
| `DroneTarget` | 无人机 |
| `PeopleTarget` | 人 |
| `ShipTarget` | 船 |
| `UnknownTarget` | 低空内部未知/拒识残差 |

当前路线是：

1. 统一 manifest 和 grouped split，禁止 RD 帧级随机切分。
2. 共享轨迹表征作为系统底座。
3. 物理特征路由器负责粗分流。
4. 专家一负责低空疑似目标，是当前主攻方向。
5. 专家二、三、四作为常规空中、高空规则、地面目标支路。
6. System 训练阶段并行计算专家，推理/评估阶段支持 route-reject-reroute。

## 2. 目录职责

| 目录/文件 | 职责 |
|---|---|
| `src/process` | 原始数据标准化、manifest 构建、训练索引生成 |
| `src/core/data` | 标准数据读取、RD 读取、DataLoader、grouped split |
| `src/core/modeling` | 兼容导出层，转发到当前核心模型实现 |
| `src/core/training` | 兼容训练入口，转发到当前 trainer |
| `src/core/trajectory.py` | 共享轨迹表征编码器 |
| `src/core/router.py` | 物理特征路由器和规则覆盖 |
| `src/core/experts.py` | Expert1/2/3/4 模型 |
| `src/core/system.py` | RadarClassificationSystem 和推理端 reroute |
| `src/core/*_trainer.py` | 各阶段训练与评估 |
| `config/config.json` | 当前训练参数主入口 |
| `config/configmanager.py` | 配置读取和 dataclass 映射 |
| `reports/real` | 时间戳分类的真实训练报告 |
| `artifacts` | 时间戳分类的模型、history、manifest、summary |
| `debug.md` | 架构问题、解决方案、训练结果追踪 |

## 3. 数据流

当前使用处理后的 `data` 目录，不直接在训练阶段读取原始 `OriData`。

已接入数据来源：

| 来源 | 当前用途 |
|---|---|
| `raw_tr_rd` | 原始航迹 + RD，样本少但风险高，是弱源重点 |
| `excel_track` | Excel 航迹数据，无 RD |
| `cq08_track` | 23/CQ-08 航迹、点迹、RD mat 标准化结果 |

核心索引：

| 索引 | 用途 |
|---|---|
| `trainable_track_index.csv` | 通用旧入口，逐步降级为兼容索引 |
| `expert1_track_index.csv` | Expert1 低空五类训练 |
| `system_trainable_track_index.csv` | System 全局九类训练，包含 CQ08 |
| `cq08_track_index.csv` | CQ08 单源索引 |
| `cq08_grouped_split.json` | CQ08 grouped split 依据 |

重要数据约束：

- CQ08 的 RD 来自 `cq08/mat` 标准化后的 mat/npz 结果。
- CQ08 点迹中的速度、幅度、SNR 已进入标准化轨迹。
- CQ08 的高度和相位缺失必须以 missing 标记处理，不能当作真实 0。
- 切分必须按航迹编号、采集批次或等价 group 执行，禁止把同一航迹的 RD 帧随机拆入 train/test。
- System、Expert1、Router、Pretrain 应尽量使用统一分布索引，避免预训练和最终 System 分布错位。

## 4. 共享轨迹表征

共享轨迹底座由 `TrackRepresentationBuilder` 实现，输入标准化序列特征和物理特征，输出：

| 输出 | 用途 |
|---|---|
| `sequence_features` | 专家二等时序支路 |
| `deep_features` | 专家四和 System 共享深特征 |
| `physical_features` | 归一化物理特征 |
| `raw_physical_features` | 路由规则和物理规则覆盖 |
| `fused_features` | Expert1 轨迹分支输入 |

训练稳定性策略：

- 支持按 `source_name` 拟合 source normalization。
- System 可从 Expert1 或轨迹预训练 checkpoint 回灌 shared encoder。
- Expert2/Expert4 不允许再用私有 representation 覆盖 System shared encoder。
- System 训练有 sequence encoder 低学习率、独立梯度裁剪和全局梯度裁剪。

## 5. 路由器

路由器是 `DeepGatingRouter`，输入物理特征，输出四路 route。

当前 route：

| Route | 专家 |
|---|---|
| `low_altitude_suspect` | Expert1 |
| `conventional_air` | Expert2 |
| `high_altitude_filter` | Expert3 |
| `ground_target` | Expert4 |

当前实现要点：

- Router 网络输入加 `LayerNorm`，避免物理特征量纲爆炸。
- 规则覆盖仍使用 raw physical，保证高度、速度等物理语义不被归一化破坏。
- 高空/地面极端规则可直接覆盖 route。
- 训练指标必须看 `route_macro_recall` 和 per-route recall，不能只看 `route_accuracy`。
- System 默认不使用 hard route 训练，避免 router 不稳时压死正确专家。

## 6. Expert1 低空专家

Expert1 是当前核心专家，类别为：

`DroneTarget`、`BirdTarget`、`BalloonTarget`、`ClutterTarget`、`UnknownTarget`

### 6.1 轨迹分支

轨迹分支只吃 `representation.fused_features`，不拼接 RD summary。这个约束很重要：未来推理时，轨迹分支必须能作为纯轨迹快速分支独立工作。

结构：

- `traj_backbone`
- `CosineMarginHead`
- `traj_residual_head`

当前轨迹分支职责：

- 快速低空粗判。
- 作为 RD 是否需要采纳的 base evidence。
- 给 Unknown residual 提供轨迹侧残差信号。

### 6.2 RD 分支

RD 分支输入包含：

- range amplitude projection
- range phase projection
- Doppler amplitude projection
- Doppler phase projection
- 轻量 RD patch CNN
- RD morphology/statistics
- RD context projection

当前 RD 统计包括：

- skew
- kurtosis
- phase abs mean
- phase std
- range entropy
- Doppler entropy
- peak-to-mean
- temporal roughness

当前 RD 分支不是简单五分类辅助头，而是第二证据源：

- `rd_head` 做五类 CosFace。
- `rd_bird_head` 强化 Drone/Bird。
- `rd_unknown_head` 强化 Unknown one-vs-rest。
- `rd_clutter_head` 强化 Clutter one-vs-rest。
- `rd_residual_head` 建模 known-vs-residual。
- `rd_unknown_clutter_head` 建模 Unknown-vs-Clutter。
- `rd_clutter_morphology_head` 用 RD morphology/patch 建模 Clutter 局部形态。

### 6.3 Unknown/Clutter 三层建模

Unknown 不再只按普通闭集类理解，而是低空内部 residual/reject bucket。

当前三层结构：

1. `known-vs-residual`：判断是否不像 Drone/Bird/Balloon。
2. `Unknown-vs-Clutter`：在 residual 内部区分未知与杂波。
3. `final correction adoption`：通过 class gate 和 pair gate 决定 RD 是否接管局部类别组。

Clutter 依赖局部形态，因此除了全局 RD embedding，还使用 entropy、peak-to-mean、temporal roughness 和 patch CNN 输出。

### 6.4 融合和接管

Expert1 当前不是单一 `(B, 2)` 全局 RD/trajectory 权重。

融合分三层：

- class-wise correction：每个低空类一组 RD correction gate。
- pair-wise adoption：Drone/Bird、Unknown/Clutter、Balloon 三个局部 pair gate。
- reject-style residual：当已知类 cosine 低于阈值时，回收到 Unknown 或 Clutter。

训练阶段可以全量计算 RD。推理阶段可按置信度和模式控制是否启用 RD。

## 7. Expert2 常规空中专家

Expert2 已从普通二分类向 open-set Aircraft 原型识别靠拢。

当前职责：

- 只学习 Aircraft 的紧凑原型和相似度边界。
- NonAircraft 作为默认拒识结果，不再把八个不相关类强行学成一个紧凑负类。
- 支持 source-specific threshold，尤其压制 `raw_tr_rd` false Aircraft。
- 训练时不再训练私有 shared encoder，必须使用共享 representation checkpoint。

当前风险：

- Aircraft 样本极少，独立阈值仍容易过激。
- System 中 Aircraft 会受到跨专家仲裁影响，需要继续校准 candidate arbitration。

## 8. Expert3 高空规则专家

Expert3 是高空/高速规则过滤器。

当前职责：

- 对明确高空高速目标直接判定或过滤。
- 在 reroute 中属于 terminal route，规则命中后不进入多轮回流。

当前限制：

- 数据中高空样本不足，`high_altitude_filter` 的 recall 指标经常为 0 或无支持。
- 若后续引入真实高空类别，需要补全标签、训练、评估和全局 logits 映射。

## 9. Expert4 地面专家

Expert4 负责 `PeopleTarget`、`CarTarget`、`ShipTarget`。

当前结构：

- 输入 System shared deep features 和 raw physical features。
- 使用 ground classifier 输出三类 ground logits。
- 使用 `groundness_head` 输出是否为 ground 的 membership confidence。

System 训练已接入：

- `system_groundness_loss_weight`
- `system_groundness_separation_loss_weight`
- true ground 推向高 groundness。
- non-ground 推向低 groundness。
- ground expert 低学习率和前期特征冻结策略。

当前限制：

- People/Ship/Aircraft 样本极少，System macro F1 受长尾明显制约。
- Groundness protection 默认未开启，当前先报告诊断，避免不稳定 groundness 放大 FP。

## 10. System 级融合

训练期和推理期当前分开处理。

训练期：

- 默认并行计算所有专家。
- 使用 `_project_to_global_logits()` 做全局 logits 投影。
- 支持 `soft_log_weighted`、`ungated`、`soft_top2`、`calibrated_candidates` 等 fusion probe。
- 不把 `reroute` 作为可微训练路径。

评估/推理期：

- 支持 `RadarClassificationSystem.forward(..., inference_mode="reroute")`。
- Reroute 由 System 内部完成，不再只是 trainer 后处理。
- 共享表征和 RD projection 只计算一次。
- 每个专家暴露统一 `acceptance()` 接口。

Reroute 流程：

1. Router 输出初始 route logits。
2. 选择当前最高 route。
3. 调用对应专家并读取 membership confidence。
4. 置信度足够则接受。
5. 置信度不足则惩罚该 route logits，并进入下一轮。
6. 达到最大轮数或惩罚上限仍不能接受，则 fallback 到 `UnknownTarget`。

当前 reroute 指标：

- `reroute_rate`
- `reroute_avg_count`
- `reroute_recovery_rate`
- `reroute_fallback_rate`
- `reroute_worsen_rate`
- `reroute_count_distribution`

## 11. 训练入口

所有训练参数优先从 `config/config.json` 读取，命令行只保留少量入口参数。

常用命令：

```powershell
.\.venv\Scripts\python.exe -m src.core.train_pretrain_track --config config/config.json
.\.venv\Scripts\python.exe -m src.core.train_router --config config/config.json
.\.venv\Scripts\python.exe -m src.core.train_expert1 --config config/config.json
.\.venv\Scripts\python.exe -m src.core.train_expert2 --config config/config.json
.\.venv\Scripts\python.exe -m src.core.train_expert4 --config config/config.json
.\.venv\Scripts\python.exe -m src.core.train_system --config config/config.json
```

设备策略：

- 默认 `device=cuda`。
- 当前训练不使用 CPU 作为性能结论。
- 只有超过约 30 分钟的长训才需要人工确认。

## 12. 评估报告

每次训练按时间戳输出：

- `artifacts/<timestamp>/<experiment>/best_model.pt`
- `artifacts/<timestamp>/<experiment>/training_history.json`
- `artifacts/<timestamp>/<experiment>/training_split_manifest.json`
- `artifacts/<timestamp>/<experiment>/evaluation_report.json`
- `artifacts/<timestamp>/<experiment>/training_summary.json`
- `reports/real/<timestamp>/<experiment>.md`
- `reports/real/<timestamp>/<experiment>.json`

System 报告必须重点看：

- global accuracy / macro F1
- route macro recall
- per-route recall
- Expert1 low-altitude macro F1
- Unknown / Clutter recall
- Drone-vs-Bird 混淆指标
- Ground group F1、Ground FP、Ground recovery
- Aircraft precision/recall/F1、Aircraft FP by source
- source-balanced accuracy
- min source accuracy
- reroute metrics

Expert1 报告必须重点看：

- final macro F1
- trajectory macro F1
- RD macro F1 on RD
- RD rescue/harm/net gain
- final rescue/harm/net gain
- RD per-class recall on RD
- Drone/Bird recall mean
- Unknown/Clutter recall mean
- residual/detail/morphology head accuracy
- by-source accuracy，尤其 `raw_tr_rd`

## 13. 最近一次 CUDA 训练结果

### 13.1 Expert1

训练命令：

```powershell
.\.venv\Scripts\python.exe -m src.core.train_expert1 --config config/config.json
```

输出目录：

`artifacts/20260508-124356/expert1_config_smoke`

核心指标：

| 指标 | 值 |
|---|---:|
| accuracy | 0.8290 |
| macro F1 | 0.7118 |
| trajectory macro F1 | 0.7246 |
| RD macro F1 on RD | 0.5742 |
| RD Drone/Bird recall mean | 0.7015 |
| RD Unknown/Clutter recall mean | 0.4190 |
| RD harm rate on RD | 0.2110 |
| RD rescue rate on RD | 0.0338 |
| final harm rate on RD | 0.0084 |
| final rescue rate on RD | 0.0127 |
| RD residual accuracy on RD | 0.9072 |
| RD Unknown/Clutter detail accuracy | 0.7273 |
| RD clutter morphology accuracy | 0.8182 |
| raw_tr_rd accuracy | 0.2500 |
| source balanced accuracy | 0.6064 |

结论：

- residual 和 Clutter morphology 头已经学到有效信号。
- RD on-RD 宏 F1 有明显改善，但 RD 纯分支净收益仍为负。
- final 融合能显著减伤，但还没有稳定放大 RD rescue。
- `raw_tr_rd` 样本过少且最弱，当前是 Expert1 的主要风险入口。

### 13.2 System

训练命令：

```powershell
.\.venv\Scripts\python.exe -m src.core.train_system --config config/config.json
```

输出目录：

`artifacts/20260508-130610/system_config_smoke`

核心指标：

| 指标 | 值 |
|---|---:|
| selected fusion mode | `calibrated_candidates` |
| selected accuracy | 0.8163 |
| selected macro F1 | 0.5518 |
| best test probe mode | `soft_log_weighted` |
| best test probe accuracy | 0.8283 |
| best test probe macro F1 | 0.5560 |
| route macro recall | 0.7460 |
| low-altitude Expert1 macro F1 | 0.7452 |
| Drone/Bird recall mean | 0.8636 |
| Unknown recall | 0.6111 |
| Clutter recall | 0.4286 |
| Ground group F1 | 0.5641 |
| Aircraft recall | 0.6000 |
| Aircraft precision | 0.3750 |
| source balanced accuracy | 0.6352 |
| min source accuracy | 0.3636 |
| skipped non-finite batches | 0 |

结论：

- 新 Expert1 checkpoint 已完整加载到 System，无 missing key。
- System 的低空能力可用，但全局 macro F1 仍被 Aircraft、Ground 和弱源拖低。
- `calibrated_candidates` 在 validation 上被选中，但 test probe 上 `soft_log_weighted` 略高，说明仲裁器仍需更稳的校准策略。
- Reroute 链路可用，但当前默认阈值下 test selected mode 没触发 reroute；后续应单独用 reroute fusion probe 调阈值。

## 14. 当前已实现

- 统一 `config/config.json` 配置入口。
- UTF-8 BOM 兼容读取。
- 统一大类/子类标签接口，当前无细分子类时子类等于大类。
- CQ08 manifest、标准化和 grouped split。
- System 使用 `system_trainable_track_index.csv`。
- Expert1 纯轨迹分支和 RD 分支解耦。
- Expert1 RD patch CNN、RD stats、pair gate、residual/reject-style Unknown、Clutter morphology head。
- Expert1 选模和报告输出 RD rescue/harm、source-balanced、Unknown/Clutter、Drone/Bird 指标。
- Expert2 open-set Aircraft 原型和 source threshold。
- Expert4 groundness head 和 System groundness 监督。
- System calibrated candidate probe。
- System forward 层 route-reject-reroute。
- 时间戳 artifacts 和 reports。

## 15. 当前未达标项

1. Expert1 RD 仍不是稳定正收益分支。
2. `raw_tr_rd` 样本极少，min source accuracy 不稳。
3. Unknown/Clutter 有改善，但 Clutter recall 仍偏低。
4. Aircraft precision 仍不足，Expert2 positive 过多但 System 最终被仲裁压制。
5. Ground group F1 尚未稳定到 0.60+。
6. Reroute 已上移到 System forward，但阈值和候选策略还没有带来稳定收益。
7. 高空 route 数据支持不足，暂时只能保留规则路径。
8. 长尾样本 People/Ship/Aircraft 太少，宏 F1 的上限受数据约束明显。

## 16. 下一阶段优先级

1. 继续稳 Expert1 RD，目标是先降低 RD harm，再提升 rescue。
2. 对 Unknown/Clutter 做更强 residual calibration 和 Clutter morphology 约束。
3. 用 source-aware sampling 和 raw source 回放提升 `raw_tr_rd`。
4. 修 Expert2 的 precision/recall 阈值平衡，降低 false Aircraft。
5. 继续 Ground 仲裁器，先验证高置信 Ground protection，再考虑默认开启。
6. 单独调 reroute threshold，让回流机制产生可测 recovery，而不是只作为可用链路。
7. 当 Expert1 checkpoint 真正过 RD quality gate 后，再把它作为 System 主力长期训练入口。
