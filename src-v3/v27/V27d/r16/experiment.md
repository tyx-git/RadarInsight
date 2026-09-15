# V27d-r16：RD 帧真实局部时间连续递推

## 实验身份

- 实验：`V27d-r16`
- 隶属：`V27d`
- 直接基线：同 seed 的 `V27d-r14` control，固定 epoch 20；其外层仍是 V27 Liquid V2 L2 与 V27a Main stable8 组合。
- 研究问题：canonical RD tensor 的帧级 `time_seconds` 与相邻帧 `Delta t` 是否能在当前 RD 空间编码之后提供可迁移的连续时间证据。
- 唯一变量：当前 TR 事件内部的 RD 帧聚合从 frozen masked mean 扩展为“masked mean + zero-initialized Liquid V2 frame residual”。
- seeds：`42、44`。

## 设计

### 数据与时间语义

canonical RD tensor 已实际包含：`amplitude`、`phase`、`time_seconds`、`frame_event_indices`。r16 从与 index 中 frame indices 对齐的 tensor 读取 `time_seconds`；只使用相邻选中帧的差值，时间原点只影响被消去的常数，不输入绝对 capture 时间。

- RD 时间语义：`cpi_relative_elapsed_not_tr_wall_clock`；不是 TR 事件间 wall-clock `dt`。
- 帧顺序：保持 canonical index 顺序；不排序、不取绝对值、不修复负时间。
- 每个 TR 事件独立重置 RD frame Liquid V2 状态；不会把前一 TR 事件的帧状态带入下一事件。
- 第一帧使用 first-event 语义；后续帧的相邻正 Δt 驱动 frame Liquid V2 的 alpha。
- 单帧或缺失 RD：保留 masked mean，temporal residual 为零。
- 非单调 RD：保留 masked mean，temporal residual 为零，并记录 fallback 数量。当前已知训练/验证池的非单调 present RD 事件数为 `0`；全 V16 主范围的非单调标记主要对应缺失 RD 行。
- padding：不进入外层 TR Liquid、不进入 RD frame Liquid、不产生 residual。

### 模型边界

- control：冻结 `V27d-r14` bounded-observation control；RD 维持 masked mean。
- treatment：复制 control 的空间 RD encoder 权重，加入独立 `radar_ln3.liquid_v2.LiquidV2Cell` 处理帧序列；输出为 `masked_mean + residual_scale * tanh(residual_projection(continuous - masked_mean))`。
- treatment 仅训练 `frame_cell` 与 `residual_projection`，共 `4176` 个参数；外层 V27 Liquid V2、空间 CNN/projection、V27a stable8 分支和分类端点均冻结。
- treatment residual projection 零初始化，因此初始化时逐值复现 control。
- L2 仍只有 `DroneTarget/BirdTarget/BalloonTarget` 三类；Unknown 不进入 L2 类别、训练或本实验评估。
- knownness、quality/maturity、L3、conformal、test、OOF 和跨源验证均不在范围内。
- 损失继续使用 `parent_final_fixed_parent_count_ce` 与固定 parent-count reduction。

## 时间输入实际覆盖

validation 每个 seed 的统计完全一致：

| 项目 | 数量 |
| --- | ---: |
| active RD event | 4,233 |
| frame-time eligible event | 4,219 |
| single/missing fallback event | 14 |
| non-monotonic present fallback event | 0 |
| positive frame Δt | 11,090 |
| zero frame Δt | 0 |
| total active validation events | 4,288 |

这证明 treatment 不是只按帧序号递推：`time_seconds` 已参与相邻帧的连续时间参数计算；同时也显示当前数据的帧时间模式高度规则，所有实际相邻 Δt 均为正，r16 没有获得 zero/irregular frame-gap 的有效对照。

## 固定 epoch 20 结果

指标为 validation parent-final。百分比和差值保留四位小数；差值为 treatment 减 control，单位为百分点（pp）。

### Seed 42

| 类别 | Control P/R/F1 | r16 treatment P/R/F1 | P Δ | R Δ | F1 Δ |
| --- | --- | --- | ---: | ---: | ---: |
| DroneTarget | 98.1818% / 95.5752% / 96.8610% | 98.1481% / 93.8053% / 95.9276% | -0.0337 | -1.7699 | -0.9334 |
| BirdTarget | 91.5254% / 96.4286% / 93.9130% | 88.5246% / 96.4286% / 92.3077% | -3.0008 | 0.0000 | -1.6054 |
| BalloonTarget | 98.1481% / 98.1481% / 98.1481% | 98.1481% / 98.1481% / 98.1481% | 0.0000 | 0.0000 | 0.0000 |

- Accuracy：`96.4126% -> 95.5157%`，`-0.8969pp`；总正确数 `215 -> 213`。
- Macro-F1：`96.3074% -> 95.4611%`，`-0.8462pp`。
- 配对：`2` 个 `correct -> wrong`，`0` 个 `wrong -> correct`；两个原本正确的 Drone 被推为 Bird。

### Seed 44

| 类别 | Control P/R/F1 | r16 treatment P/R/F1 | P Δ | R Δ | F1 Δ |
| --- | --- | --- | ---: | ---: | ---: |
| DroneTarget | 98.1818% / 95.5752% / 96.8610% | 98.1818% / 95.5752% / 96.8610% | 0.0000 | 0.0000 | 0.0000 |
| BirdTarget | 91.5254% / 96.4286% / 93.9130% | 93.1034% / 96.4286% / 94.7368% | +1.5780 | 0.0000 | +0.8238 |
| BalloonTarget | 100.0000% / 100.0000% / 100.0000% | 98.1818% / 100.0000% / 99.0826% | -1.8182 | 0.0000 | -0.9174 |

- Accuracy：`96.86099% -> 96.86099%`，`0.0000pp`；总正确数持平 `216/223`。
- Macro-F1：`96.9247% -> 96.8935%`，`-0.0312pp`。
- 配对：`0` 个 `correct -> wrong`，`0` 个 `wrong -> correct`；1 个原本错误的 Drone 变为 Balloon，仍然错误。

### 两 seed 汇总

| 指标 | Control mean | Treatment mean | Treatment - Control |
| --- | ---: | ---: | ---: |
| Accuracy | 96.6368% | 96.1883% | -0.4484pp |
| Macro-F1 | 96.6160% | 96.0393% | -0.4387pp |
| Bird Recall | 96.4286% | 96.4286% | 0.0000pp |

## 结论

`V27d-r16` 判定为 **`diagnostic_only_no_promotion`**。

实验成功证明数据和工程链路具备做这项研究的条件：原始 canonical RD tensor 的真实局部时间已被读取，帧间 Δt 已进入独立 Liquid V2 frame cell，状态按 TR 事件隔离，padding、单帧、缺失和非单调边界均有明确处理。它没有证明该变量能改善当前分类：两 seed 的 accuracy 平均下降 `0.4484pp`，macro-F1 平均下降 `0.4387pp`；seed42 新增 2 个错误，seed44 没有净纠错。

因此不把 r16 treatment 接入 V27 默认 L2，也不接入 knownness/L3。该结果更可能说明：在当前每事件 RD 帧数量较少、帧间时间模式规则、空间 encoder 已压缩为单事件 embedding 的情况下，闭集 parent-final CE 无法稳定利用新增的帧时间动态；它不能推出 RD 帧时间信息本身无效。后续若继续，应优先做时间分支的独立轨迹/质量/knownness 诊断或受约束的表示学习，而不是把该闭集 residual 直接晋级为产品主干。

## 产物

- 配置：`config/V27d-r16-rd-frame-liquid.json`
- manifest：`manifests/V27d-r16-manifest.json`
- manifest SHA-256：`c4df36f5698831d5de800d342668991a4b82b2ec540a26b49c97fa71fc53ec8d`
- contract smoke：`reports/V27d-r16-contract-smoke.json`
- 两 seed summary：`reports/V27d-r16-two-seed-summary.json`
- seed42 paired：`reports/V27d-r16-seed42-paired-validation.json`
- seed44 paired：`reports/V27d-r16-seed44-paired-validation.json`
- seed42 training log：`runs/V27d-r16-treatment-seed42.log`
- seed44 training log：`runs/V27d-r16-treatment-seed44.log`
- control training records：`runs/V27d-r16-control-seed42.log`、`runs/V27d-r16-control-seed44.log`
- 早期 provenance 产物：`archive/pre-paired-path-fix/`、`archive/pre-provenance-migration/`、`archive/pre-contract-wording-fix/`

## 执行范围

- 已执行：来源 hash 校验、manifest prepare、contract smoke、control provenance reuse、treatment seed42/44 固定 20 epoch 训练、evaluation、paired validation、two-seed summary。
- 未执行：Unknown 训练或扩充、knownness/quality 训练、动态 L3/conformal、test、OOF、跨源验证和 deployment。
