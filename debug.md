# Debug Log

本文档用于记录模型从最初基线到当前版本的关键迭代。后续只要架构、输入表示、训练策略或数据处理主路径发生变化，都需要继续在这里追加记录。

## Stage 0 - Base Skeleton

- **当前阶段**：最初可运行基线
- **当前所面临的问题**：
  - 先把训练链路跑通，但模型表达能力较弱
  - RD 分支更像通用视觉分支，轨迹分支也偏轻量
  - 只能证明“能训练”，不能证明“高精度”
- **解决方案**：
  - 搭建双分支基础骨架
  - 引入轨迹分支、RD 分支和融合分类头
- **方案利弊**：
  - 优点：链路完整、便于快速验证
  - 缺点：模型偏轻，微动建模不足，精度上限有限

## Stage 1 - High-Accuracy Backbone Route

- **当前阶段**：高精度主方案定型
- **当前所面临的问题**：
  - 基线模型过轻，难以承载真实雷达任务的时空复杂性
- **解决方案**：
  - 轨迹分支升级为 `Residual TCN + Transformer`
  - RD 分支升级为官方 `X3D` 风格 `X3D-M`
  - 融合层升级为双向 `Cross-Attention`
- **方案利弊**：
  - 优点：主干能力大幅提升，结构更接近高精度路线
  - 缺点：参数量上升后，对数据质量和训练稳定性要求更高

## Stage 2 - Session-Level Training and Evaluation

- **当前阶段**：训练与评估工程正确性升级
- **当前所面临的问题**：
  - 按 clip 随机切分会产生数据泄漏
  - 没有独立评估器时，很难判断 checkpoint 的真实表现
- **解决方案**：
  - 引入 `session` 级切分
  - 增加按类别分层的 session 划分
  - 增加独立 `evaluate` 与真实图表生成链路
- **方案利弊**：
  - 优点：评估更可信，更接近真实泛化能力
  - 缺点：样本表面数量不变，但有效独立样本数更少，指标更容易波动

## Stage 3 - Real Data Pipeline

- **当前阶段**：从伪数据切到真实 `.mat` 数据主链
- **当前所面临的问题**：
  - 训练不应继续依赖伪数据和旧雷达接口
  - 原始 `.mat` 数据必须标准化后才能稳定进入训练
- **解决方案**：
  - 建立 `pro/raw -> pro/sessions -> data/sessions` 链路
  - `TR` 提取轨迹表，`DPL` 生成 RD 时空数据
  - 标签从文件名提取，形成可训练弱监督数据集
- **方案利弊**：
  - 优点：主链切到真实环境，工程边界更清晰
  - 缺点：当前标签仍是 session 级弱监督，不是精细标注

## Stage 4 - RD Strong Modeling (Input and Front-End)

- **当前阶段**：RD 微动链路增强
- **当前所面临的问题**：
  - 只看幅度图会过早丢失复数结构
  - 通用 CNN 容易把雷达谱图当普通图片处理
- **解决方案**：
  - 引入复数感知 RD 表示
  - 加入 `RadarMicroMotionEnhancer`
  - 加入 `SpectralAxisRefiner`
  - 加入 `BackgroundContrastRefiner`
- **方案利弊**：
  - 优点：RD 分支开始真正引入雷达物理先验
  - 缺点：复杂度提升后，对数据规模和输入质量更敏感

## Stage 5 - Complex-Aware RD Inputs

- **当前阶段**：RD 输入从单纯幅度升级为复数感知主路径
- **当前所面临的问题**：
  - 只用幅度很难稳定抓到微动细节
  - 需要在输入层保留更多对微动有价值的信息
- **解决方案**：
  - 当前默认单尺度主路径保留 `real / imag / log amplitude / temporal difference / background suppressed / local contrast`
- **方案利弊**：
  - 优点：输入层信息更完整，更有利于后续 `X3D-M` 提取微动模式
  - 缺点：如果 RD 本身定位不准，更多通道不一定能直接救回来

## Stage 6 - Training Strategy Upgrade

- **当前阶段**：训练稳定性增强
- **当前所面临的问题**：
  - 小样本真实数据下，训练波动较大
  - 多类不平衡会影响整体表现
- **解决方案**：
  - 引入 `class-balanced loss`
  - 引入 `label smoothing`
  - 引入 `warmup + cosine scheduler`
  - 引入 `gradient clipping`
  - 使用 `macro_f1` 监控和 best checkpoint 选择
- **方案利弊**：
  - 优点：训练过程更稳，模型选择更合理
  - 缺点：如果输入表征本身有偏，这些策略只能缓解，不能根治

## Stage 7 - Multiresolution RD (toTest)

- **当前阶段**：多分辨率 `DPL -> RD` 实验分支
- **当前所面临的问题**：
  - 理论上多尺度有助于覆盖不同时间尺度的微动
  - 但默认并列输入后，真实数据上明显退化
- **解决方案**：
  - 保留多分辨率实现
  - 明确降级为 `toTest` 实验分支，不作为默认主路径
- **方案利弊**：
  - 优点：保留后续继续研究的能力
  - 缺点：当前版本不适合作为默认训练路线

## Stage 8 - Single-Scale Main Path Recovery

- **当前阶段**：默认主路径回到单尺度
- **当前所面临的问题**：
  - 多分辨率默认主路径效果不稳定
  - 当前真实数据规模不足以支撑更复杂主输入
- **解决方案**：
  - 默认回退到单尺度 `DPL -> RD`
  - 默认 `6` 通道主路径稳定为 `real / imag / log amplitude / temporal difference / background suppressed / local contrast`
  - 调整辅助损失权重，偏向 RD 主线
- **方案利弊**：
  - 优点：主路径更稳，当前结果明显优于多分辨率默认方案
  - 缺点：仍未彻底解决目标对齐和微动 ROI 的问题

## Stage 9 - Current Diagnosis (20260322-10-4-cpu)

- **当前阶段**：当前主路径真实训练结果诊断阶段
- **当前所面临的问题**：
  - 当前主要误分类是 `Bird -> RotaryDrone` 与 `FixedWing -> RotaryDrone`
  - 训练集收敛，但验证和测试泛化不稳
  - RD 微动分支当前仍没有充分利用目标中心裁剪与更强信号处理
  - 轨迹分支对绝对量纲的保留仍不够理想
- **解决方案**：
  - 重新启用 `slicer.py` 目标对齐裁剪思路
  - 升级 `DPL -> RD`：窗函数、更强杂波抑制、明确目标对齐、专项微动增强变换
  - 重新审视轨迹链路，避免过度丢失绝对物理量纲信息
  - 暂时保留当前弱监督标签，不做专为弱标签设计的专属技巧
- **方案利弊**：
  - 优点：方向更贴近当前真实瓶颈，优先解决 RD 输入和对齐问题
  - 缺点：改动会触及数据处理主链，需要重新生成数据集并重新验证训练稳定性

## Current Priority

- **当前阶段**：即将进入下一轮 RD 主链与轨迹链路改造
- **当前所面临的问题**：
  - 当前最强瓶颈不是模型层数不够，而是输入质量、目标对齐和物理语义保留不够
- **解决方案**：
  - 优先修 RD 主链
  - 并行思考轨迹分支的更优物理信息保留方案
- **方案利弊**：
  - 优点：更符合第一性原理，也更符合真实任务需求
  - 缺点：验证周期会变长，需要更多重新训练与对比实验


## Stage 10 - RD Target Alignment and Signal Processing Upgrade

- **????**????? RD ??????????
- **????????**?
  - RD clip ?????????????????????
  - `DPL -> RD` ????????????????????????
  - ???????????????????
  - RD ????? z-score ????????????
- **????**?
  - ???? `slicer.py` ??????????????????/?????? RD ROI
  - ? `DPL -> RD` ??????
    - ??????
    - ???????
    - ?????????????
    - ???????????
  - ????????????????????????
  - ? RD ?????? z-score ??????????????????
- **????**?
  - ???
    - RD ?????????? ROI ??????????????
    - DPL ???????????????
    - ????????????????? `FixedWing / RotaryDrone` ????
  - ???
    - ?????? `prepare-pro` ???????
    - ???????????????????????
    - ?????????????????


## Stage 11 - First Full 5-Epoch Validation After Target-Aligned RD Upgrade

- **????**?? RD ??????? 5 epoch ??????
- **????????**?
  - ????????????????
  - `20260323-5-4-cpu` ????????? `Accuracy=0.3566`?`Macro F1=0.2556`
  - ?????????????????`FixedWing -> RotaryDrone` ???`RotaryDrone -> FixedWing` ??
  - ?????????????????????????
- **????**?
  - ???????????
  - ???? RD ?????????????
  - ?????
    - ?? `max_range / max_speed` ???? `range_res_m / doppler_res_mps` ?????? session ??????
    - ???????????????????
    - `slicer.py` ??????????????????????? clip ????
    - ???????????????????????????
  - ?????????????????????????????????????????????????
- **????**?
  - ???
    - ???????????????????????RD ????????????????????????
    - ??????????????????
  - ???
    - ?????????????????????
    - ???????? RD bin ??????????????


## Stage 12 - Stable Physical Axes and Smoothed ROI Alignment

- **????**?P0 + P1 ???????? RD ????? ROI ????
- **????????**?
  - ???? `slicer.py` ?????? RD ???????????????????????
  - ?????????????????????????????
- **????**?
  - ? `rd_data.h5` ????? `range_axis_m` ? `doppler_axis_mps`
  - ?????????? `max_range / max_speed` ????????????????
  - `slicer.py` ???????????????????? + clip ??? + ???????
  - ????????????????????????????
- **????**?
  - ???
    - ???????????????????????
    - ??????????????? ROI ????????
    - ???????????? + ?? ROI ???????
  - ???
    - ????????????????????????????????????
    - ??????????????????????????


## Stage 13 - Second 5-Epoch Validation After Stable-Axis Refactor

- **????**?P0 + P1 ????? 5 epoch ???artifact: `202603231929-5-4-cpu`?
- **????????**?
  - ????????????????????????
  - ?????? `Accuracy=0.4126`?`Macro F1=0.2687`
  - `Bird` ???????? `FixedWing` ?????`RotaryDrone` ?????
  - ?????????? ROI ??????????? RD ?????
- **????**?
  - ?????????? RD ????? ROI ????????????
  - ??????????
    - `RD` ?? ROI ?????????
    - ????????????????
    - clip ????????????????????
    - ?????????????????????????
- **????**?
  - ???
    - ??????????? ROI ?? + ??????????????????????
    - ????????????RD ????????????????
  - ???
    - ?????????????????? RD ???????????
    - ?????????????????????????


## Stage 14 - Calibrated-First Axes and Guarded ROI Refinement

- **当前阶段**：继续压实 RD 对齐问题，优先修物理轴与 ROI 护栏。
- **当前所面临的问题**：
  - 当前 RD 物理轴仍偏统计近似，跨 session 的物理语义一致性不够稳。
  - 多普勒方向裁剪过宽，实际没有把目标速度区域真正局部化。
  - 单帧局部峰值微调过于激进，容易被杂波或错误峰牵引。
- **解决方案**：
  - 引入“标定优先、统计兜底”的物理轴构造方式，默认优先使用雷达配置里的 `range_resolution_m`、`doppler_resolution_mps` 和零多普勒中心。
  - 收紧默认多普勒裁剪宽度，并加大 clip 级平滑窗口。
  - 在 ROI 对齐里加入 clip 级中心漂移约束、峰值护栏和最大微调步长限制。
  - 额外导出 `rd_alignment_diagnostics.jsonl`，便于检查每个 clip 的预测中心、平滑中心和最终精调偏移。
- **方案利弊**：
  - 优点：
    - 进一步削弱“错误物理映射 -> 错误 ROI -> 错误微动特征”的链式放大效应。
    - 让 RD 裁剪过程从“黑盒裁图”变成“可诊断、可验证”的工程流程。
  - 缺点：
    - 仍然依赖当前雷达配置中的物理分辨率，若上游标定不准，仍需要后续继续校准。
    - 需要重新 `prepare-pro` 并重新训练，才能验证这些护栏是否真正转化为精度提升。


## Stage 15 - Fixed-Resolution Range Axis with Track-Aligned Offset

- **当前阶段**：继续压实 RD 物理轴，重点修正范围轴步长失真问题。
- **当前所面临的问题**：
  - 单 session 验证显示范围轴仍然回退到了“统计压缩步长”，导致 bin 间距不再是合理的物理分辨率。
  - 这会让 ROI 裁剪虽然看起来对齐了轨迹，但实际上处在一个失真的距离坐标系里。
- **解决方案**：
  - 将范围轴改成“固定分辨率 + 轨迹工作带偏移”的构造方式。
  - 保持 `range_resolution_m` 为真实物理步长，同时用轨迹分位数决定起始距离，确保目标落在合理工作带内。
  - 这样既保留物理语义，又避免简单从 `0 m` 起轴导致目标落到量程外。
- **方案利弊**：
  - 优点：
    - 避免范围轴步长被压缩成不真实的数值。
    - 让 RD 裁剪回到“真实物理分辨率 + 合理工作区间”的组合。
  - 缺点：
    - 如果上游雷达量程起点本身还有固定偏置，这里仍然只是更合理的工程近似，而不是最终标定真值。


## Stage 16 - Tolerant Coverage Check for Calibrated Axes

- **当前阶段**：修正标定轴选择时的过度回退问题。
- **当前所面临的问题**：
  - 单 session 验证显示，尽管固定分辨率轴已经更合理，但因为个别速度点只超出半个 bin，系统仍会整体回退到统计轴。
  - 这种“过于严格的覆盖判定”会把本来可用的物理轴误判为不可用。
- **解决方案**：
  - 给标定轴覆盖率判定加入与物理分辨率一致的容差。
  - 对距离轴使用 `range_resolution_m` 容差，对多普勒轴使用 `doppler_resolution_mps` 容差。
  - 这样只在明显越界时才回退，而不是因为边缘点的半 bin 偏差就整体放弃物理轴。
- **方案利弊**：
  - 优点：
    - 更符合物理采样离散化本身的误差边界。
    - 可以显著降低不必要的 `robust_track_aligned` 回退概率。
  - 缺点：
    - 若上游物理分辨率配置本身偏差较大，这种容差也可能暂时掩盖配置误差，需要后续继续核验。


## Stage 17 - Hybrid Range/Doppler Axis Selection

- **当前阶段**：继续压实 RD 物理轴，避免“全有或全无”的回退逻辑。
- **当前所面临的问题**：
  - 某些 session 只是多普勒轴覆盖不够，但距离轴其实已经可以使用固定物理分辨率。
  - 旧逻辑会因为一条轴不满足条件，就把两条轴同时回退到统计模式，导致距离轴也失去物理步长。
- **解决方案**：
  - 把距离轴和多普勒轴的选择解耦，分别判断是否采用标定优先方案。
  - 在 `calibrated_preferred` 模式下允许出现 `hybrid_calibrated_range_robust_doppler` 这类混合轴模式。
  - 这样至少可以优先锁住距离轴的真实分辨率，不再被多普勒轴一起拖回退。
- **方案利弊**：
  - 优点：
    - 更符合真实问题结构：距离轴和多普勒轴的可置信程度本来就不一定同步。
    - 可以优先保住更稳定、物理意义更明确的距离轴。
  - 缺点：
    - 轴模式会变得更多，后续分析结果时需要关注当前 session 用的是哪种组合。


## Stage 18 - Adaptive Calibrated Axis Multipliers

- **当前阶段**：继续减少 `robust_track_aligned` 的使用范围。
- **当前所面临的问题**：
  - 即使将距离轴和多普勒轴解耦后，仍有部分 session 因固定分辨率覆盖不足而回退。
  - 这类回退有时并不是必须彻底改用统计轴，而只是当前标定步长偏小。
- **解决方案**：
  - 为距离轴和多普勒轴分别加入“自适应标定倍率”选择。
  - 距离轴在 `x1 / x2 / x4` 的固定分辨率倍数中选择最小可覆盖方案。
  - 多普勒轴在 `x1 / x1.5 / x2` 的固定分辨率倍数中选择最小可覆盖方案。
  - 同时把 `range_axis_mode` 和 `doppler_axis_mode` 单独写入 HDF5 属性，便于后续定位每个 session 的轴来源。
- **方案利弊**：
  - 优点：
    - 比纯统计轴更接近真实物理分辨率。
    - 能显著减少“本可保留物理步长却被整体回退”的情况。
  - 缺点：
    - 自适应倍率仍是工程近似，不等价于上游严格标定。
    - 会引入更多轴模式，需要在训练结果分析时同步关注。


## Stage 19 - Third 5-Epoch Validation After Adaptive Axis Hardening

- **当前阶段**：验证自适应标定轴 + 护栏式 ROI 对齐后的整体训练表现。
- **当前所面临的问题**：
  - 虽然 RD 物理轴与 ROI 对齐明显更可解释了，但需要确认这些修正是否真正转化为精度提升。
  - 当前关注点是 `Bird / FixedWing / RotaryDrone` 三类是否还存在系统性塌缩。
- **解决方案**：
  - 使用最新主链重新训练 `5` epoch，并分析独立评估结果 `artifacts/202603232055-5-4-cpu`。
  - 对比混淆矩阵、`per_class` 指标和训练/验证曲线，判断目前主要错分流向。
- **方案利弊**：
  - 优点：
    - 新版本已经不再出现之前那种“范围轴物理步长失真”的问题，输入空间更可信。
    - 现在可以把问题进一步收敛到“Bird 可分，但 FixedWing / RotaryDrone 仍严重混淆”。
  - 缺点：
    - 本轮结果仍然较差，`Accuracy` 和 `Macro F1` 都没有回到可接受水平。
    - 说明仅靠当前这轮 RD 对齐压实还不够，后续还需要继续优化速度轴局部化和双视角 RD 表示。


## Stage 20 - Dual-View RD Input (Local ROI + Global Context)

- **当前阶段**：在保留目标对齐局部 ROI 的前提下，为 RD 主链补充全局上下文。
- **当前所面临的问题**：
  - 当前 ROI 已经更稳，但模型仍然缺少“目标局部微动”之外的场景上下文信息。
  - 只看局部裁剪容易在速度轴上丢失全局结构，尤其不利于区分 `FixedWing` 和 `RotaryDrone`。
- **解决方案**：
  - 在 `RadarFusionDataset` 中新增双视角 RD 输入。
  - 局部视角继续使用 `rd_clips.h5` 中的目标对齐 ROI。
  - 全局视角从 `rd_data.h5` 中按同一时间窗读取完整 RD 帧序列，再缩放到统一输入尺寸。
  - 当前默认 `8` 通道：局部 `6` 通道 + 全局 `2` 通道（`global_log_amplitude`、`global_background_suppressed`）。
- **方案利弊**：
  - 优点：
    - 同时保留局部微动细节与全局谱图上下文。
    - 对当前“局部裁剪仍可能过窄”的风险更鲁棒。
  - 缺点：
    - RD 输入复杂度继续上升，CPU 训练成本会增加。
    - 若全局上下文本身噪声较重，也可能把新的背景信息重新带进模型。


## Stage 21 - First 5-Epoch Validation After Dual-View RD Input

- **????**?????? ROI + ????????? RD ????????
- **????????**?
  - ??????? RD ????????? Bird ???????????????
  - ???????????????????????? Bird ???????
- **????**?
  - ???????? `artifacts/202603232152-5-4-cpu`????????
  - ???? `accuracy`?`macro_f1` ? `FixedWing / RotaryDrone` ????????
- **????**?
  - ???
    - ?????????? `Accuracy=0.5455`?`Macro F1=0.3250`?????? `0.3706 / 0.2598` ??????
    - ????? RD ?????????????
  - ???
    - ?????????? `Bird` ??`FixedWing` ? `RotaryDrone` ???????
    - ??????????? RD??????????????????


## Stage 22 - Track Stabilization Validation (Leak-Free Norm + Stronger Track Prior)

- **????**????????????????
- **????????**?
  - ??? RD ?????????????????????????????? `RotaryDrone` ????
  - ??????????????????
- **????**?
  - ?? train-only ????????
  - ? `sequence_length` ??? `16`?????????
  - ?????? clip ????????????????????? batch ? `GroupNorm`?
  - ?????????? track ??????????
- **????**?
  - ???
    - ????????? `Accuracy=0.7413`?`Macro F1=0.5630`?????? `0.5455 / 0.3250` ???????
    - `RotaryDrone` ??????`Bird` ?????????
    - ???????????????
  - ???
    - `FixedWing` ??????? `Bird`?
    - ??????????????????????????????????????????/???????


## Stage 23 - Sign-Invariant Motion Upgrade

- **????**??? `FixedWing -> Bird` ?????????????????????????
- **????????**?
  - ?????? `FixedWing` ???????????????????????????????/??????
  - ?????????????????????????????????
- **????**?
  - ???????? `abs_radial_speed_mps`?`abs_speed_delta_mps` ?? clip ????????
  - ? RD ???????/?????????????
  - ?? RD ????? `8` ??? `10`?????????????????????????
- **????**?
  - ???
    - ????????????????????????????
    - ????????????????
  - ???
    - ?????????????????
    - ??????????? `FixedWing` ????? `Bird` ??????


## Stage 24 - Validation After Sign-Invariant Motion Upgrade

- **????**?????/?????????????????
- **????????**?
  - ??? `Track Stabilization` ??? `RotaryDrone` ?????? `FixedWing` ???????
  - ????????? + ???????????????????????????????
- **????**?
  - ??????? `abs_radial_speed_mps`?`abs_speed_delta_mps` ? clip ????????
  - ? RD ??????/?????????????
  - ??? `10` ?? RD ????????????? `5` epoch??????????
- **????**?
  - ???
    - ????????? `Accuracy=0.7972`?`Macro F1=0.5644`?????? `0.7413 / 0.5630`?
    - `Bird` ? `RotaryDrone` ??????????????????
    - `loss` ?????? `0.6520`???????????????
  - ???
    - `FixedWing` ?????????????????????????`FixedWing` ??????????
    - ?????????????????????? `FixedWing` ? `RotaryDrone / Bird` ??????


## Stage 25 - Dual-Branch Track Modeling for FixedWing Boundary Recovery

- **????**??? `FixedWing` ???????????????????????????
- **????????**?
  - ??????? `Bird` ? `RotaryDrone` ?????? `FixedWing` ???????????
  - ?? clip ?????????????????????????????????
  - ?????????? `3?+ / 5?? / 20??`???????????????????????
- **????**?
  - ??????????
    - ?????????????????????
    - ????????????? clip ??????????????????????????????
  - ???????????
    - `clip_speed_direction_consistency`
    - `clip_range_monotonicity`
    - `clip_abs_acceleration_mean_mps2`
    - `clip_abs_turn_rate_mean_degps`
  - ????????????????????????????
- **????**?
  - ???
    - ???????????????????????????????????
    - ? `FixedWing` ???????????????????????
    - ??????????????????????????
  - ???
    - ?????????????????????????
    - ?????????????????? RD ?????????????????????


## Stage 26 - Validation After Dual-Branch Track Modeling

- **????**?????????? + ???????????????????
- **????????**?
  - ?????????????????????????? `FixedWing` ?????????????????
- **????**?
  - ?? `artifacts/202603241919-5-4-cpu`???? `FixedWing` ??????????? `Bird / RotaryDrone` ???????
- **????**?
  - ???
    - `FixedWing` ???? `0` ?????????????????????????
    - ?????? 3 ?+?5 ???20 ??????????????????????
  - ???
    - ????????? `Accuracy=0.6783`?`Macro F1=0.4846`????????? `0.7972 / 0.5644`?
    - ???????????????????????? RD ????????

## Stage 27 - Gated Global Track Summary Fusion

- **????**????????????????????????????????????
- **????????**?
  - ??????????? `FixedWing` ???????????????????????????????????????
  - ?????????????????
- **????**?
  - ? `TrackTemporalEncoder` ???????????????
    - ???????? `summary_feature`
    - ???????? `global_feature`
    - ? `gate(summary, global)` ??????
    - ??? `summary + gate * global` ?????????????????
- **????**?
  - ???
    - ???????????????????????
    - ???????????????????????????????????
  - ???
    - ??????????????????????? `Bird / RotaryDrone` ???????? `FixedWing`?

## Stage 28 - Modality Reliability Fusion

- **Current Stage**: add reliability-aware fusion after the latest manual run `artifacts/202603242052-5-4-cpu`.
- **Current Problem**:
  - Independent evaluation is still below the current best.
  - `FixedWing` collapses completely into `RotaryDrone`.
  - The pattern suggests that RD evidence is still able to dominate fusion when its local evidence is not reliable enough.
- **Solution**:
  - Add a track reliability head and an RD reliability head in `FusionRadarClassifier`.
  - Convert the two reliability scores into normalized modality weights.
  - Reweight pooled track and RD features before final fusion.
  - Add a small weighted unimodal logit residual so the final classifier can fall back to the more reliable branch.
- **Pros**:
  - General solution for future larger datasets, not a patch for the current small dataset only.
  - Lets the model trust track kinematics more when RD alignment or RD micro-motion quality is weak.
  - Keeps the current fusion backbone and only changes the decision policy near the output.
- **Cons**:
  - Adds another decision layer that can itself become unstable if branch features are weak.
  - Needs another manual 5-epoch run to verify whether `FixedWing` recovery improves without hurting `Bird` and `RotaryDrone`.


## Stage 29 - 全局 RD 索引修复与公开数据集 5 Epoch 验证

- **当前阶段**：修复重建后数据集进入训练时的全局 RD 读取阻塞问题，并在当前公开数据集上完成一次新的 5 epoch 验证。
- **当前所面临的问题**：
  - 从 `pro/raw` 重新生成数据集后，训练在 `RadarFusionDataset._load_global_rd_cube()` 阶段被 `h5py` 阻塞。
  - 根因不是模型前向本身，而是 HDF5 高级索引要求索引序列递增，而当前 clip 的 `frame_id` 读取路径没有显式保证这一点。
  - 当前数据集是用于算法训练的公开雷达数据集，不是最终真实部署环境数据集，因此当前验证目标应聚焦“是否逼近可扩展高精度主路线”，而不是围绕当前闭集样本做过拟合优化。
- **解决方案**：
  - 在 `RadarFusionDataset._load_global_rd_cube()` 中，把 `frame_indices` 先做稳定排序。
  - 对排序后的索引做去重后再执行 HDF5 读取。
  - 读取完成后，再恢复回原始时间顺序，保证时序语义不变。
  - 修复后重新执行 `manage.py prepare-pro`，再执行 `5 epoch` CUDA 训练与独立评估。
- **方案利弊**：
  - 优点：
    - 修复的是数据读取根因，而不是跳过全局 RD 或降级成局部补丁方案。
    - 恢复了当前“局部 ROI + 全局上下文”双视角 RD 主链，验证结果才有意义。
    - 为后续更大规模真实数据集提供了更稳的基础数据通路。
  - 缺点：
    - 这次修复解决的是训练阻塞，不等于已经解决 `FixedWing` 与 `RotaryDrone` 的判别边界问题。
    - 当前结果仍未超过 `debug.md` 中已有的历史最好结果。
- **验证结果**：
  - 本次运行目录：`artifacts/202603242321-5-4-cuda`
  - 独立评估结果：`Accuracy=0.7063`，`Macro F1=0.5379`
  - 混淆矩阵表现：
    - `Bird`：`82/82` 全部识别正确
    - `FixedWing`：`12/29` 识别正确，`17` 条被判成 `RotaryDrone`
    - `RotaryDrone`：`7/32` 识别正确，`25` 条被判成 `FixedWing`
- **与历史结果对比**：
  - 相比 `Stage 24` 记录的当前最好结果 `Accuracy=0.7972`、`Macro F1=0.5644`，本次结果明显回落。
  - 相比 `Stage 26` 记录的 `Accuracy=0.6783`、`Macro F1=0.4846`，本次结果有明显回升。
  - 这说明当前版本已经优于“`FixedWing` 完全塌陷”的那一轮，但还没有回到历史最优水平。
- **结论判断**：
  - `Stage 28` 的可靠性融合大概率击中了“错误模态证据主导最终决策”这一部分根因，否则不会出现从 `Stage 26` 向上的恢复。
  - 但它没有完全击中最深层根因，因为当前最主要的错误已经从“单边塌陷”变成了 `FixedWing <-> RotaryDrone` 双向混淆。
  - 这说明真正缺失的仍然是两类之间稳定、可泛化的判别表示，而不只是输出层的融合策略。
- **后续方向**：
  - 在当前公开数据集上，后续优化应继续围绕 `FixedWing` 与 `RotaryDrone` 的边界建模展开。
  - 但所有方案都必须面向后续约 `33 万` 规模、`5` 大类、`20+` 子类的真实数据集扩展，不能为了当前闭集结果引入不可扩展的特判式设计。


## Stage 30 - 非鸟类内部边界增强与可扩展层级表示升级

- **当前阶段**：围绕 `FixedWing` 与 `RotaryDrone` 的内部边界，升级轨迹分支、RD 分支和融合头，使其更适合后续大规模真实数据集扩展。
- **当前所面临的问题**：
  - 当前主要错误已经不是 `Bird` 边界，而是非鸟类内部边界不稳，尤其是 `FixedWing <-> RotaryDrone` 双向混淆。
  - 单纯依赖输出层的融合策略还不够，因为根因已经暴露为“判别表示不足”，而不是单一模态权重错误。
  - 当前公开数据集只是算法训练集，后续真实数据集规模约 `33 万`，覆盖 `5` 大类、`20+` 子类，因此不能继续围绕当前三类闭集做补丁式设计。
- **解决方案**：
  - 轨迹分支升级：
    - 在 `TrackTemporalEncoder` 内加入显式的连续性分支、平滑性分支、机动模式分支。
    - 通过模式 token 注入和模式特征门控，把“持续性 / 平滑性 / 机动强度变化”直接并入轨迹表示。
  - 轨迹特征升级：
    - 增加 `jerk_mps3`、`turn_acceleration_degps2`。
    - 增加 `clip_abs_jerk_mean_mps3`、`clip_abs_turn_acceleration_mean_degps2`、`clip_range_jitter_std_m`、`clip_speed_zero_crossing_rate`。
  - RD 分支升级：
    - 新增 `RotorMicroMotionBranch`，显式建模旋翼类微动在时间-多普勒上的调制结构。
    - 新增 `FixedWingStructureBranch`，显式建模固定翼类更稳定的谱脊与时序稳定结构。
    - 引入 `SpectralStructureEnhancer`，生成结构嵌入并作为 RD token 注入主干。
  - 层级表示升级：
    - 在 `FusionRadarClassifier` 中新增 `HierarchicalClassifierHead`。
    - 把融合特征拆成 `coarse feature` 与 `detail feature`。
    - 使用 `coarse logits` 作为辅助监督，当前用于增强主类级稳定分离，后续可自然扩展子类头。
  - RD 输入升级：
    - 默认 `rd_input_channels` 从 `10` 升级为 `12`。
    - 提前纳入 `phase_difference` 与 `coherence`，避免旋翼/固定翼判别过度依赖振幅通道。
- **方案利弊**：
  - 优点：
    - 这次升级直接作用于判别表示本身，而不是只在决策末端做再加权。
    - 轨迹和 RD 两侧都显式围绕非鸟类边界差异建模，更符合当前根因。
    - `coarse + detail` 的层级结构可以自然扩展到未来 `5` 大类、`20+` 子类，而不是为当前三类闭集写特判逻辑。
  - 缺点：
    - 模型复杂度和特征维度进一步增加，需要重新验证吞吐、显存和训练稳定性。
    - 当前只完成了最小前向验证，还没有在新架构上做新的完整训练对比。
- **当前验证状态**：
  - 已完成：静态编译通过、真实样本单批前向通过、损失计算通过。
  - 未完成：基于该新架构的完整 `5 epoch` 训练与独立评估。
- **结论判断**：
  - 这次升级比 `Stage 28` 更接近根因，因为它不再把问题仅仅视为“模态权重失衡”，而是明确把问题落实到“非鸟类内部边界表示不足”。
  - 如果后续验证有效，这条路线可以直接迁移到未来的大规模真实数据集，而不需要在进入 `5` 大类、`20+` 子类后重新推倒主干。


## Stage 31 - 非鸟类内部边界增强后的首次 5 Epoch 验证

- **当前阶段**：验证 `Stage 30` 的轨迹显式运动模式分支、RD 显式谱结构分支和层级表示头，是否真正改善 `FixedWing` 与 `RotaryDrone` 的内部边界。
- **当前所面临的问题**：
  - `Stage 30` 在设计上更贴近根因，但必须通过真实训练结果验证，不能停留在“结构上看起来合理”。
  - 当前需要判断：新架构到底是在增强非鸟类内部边界，还是把模型进一步推向某一侧偏置。
- **解决方案**：
  - 基于新架构执行一次新的 `5 epoch` CUDA 训练。
  - 运行目录：`artifacts/202603242355-5-4-cuda`
  - 独立评估并生成真实图表：`reports/real/202603242355-5-4-cuda`
- **方案利弊**：
  - 优点：
    - 这次验证直接回答“新架构是否真的有用”，避免停留在理论合理性层面。
    - 可以把 `Stage 30` 的表示升级与 `Stage 29` 的结果做直接对比。
  - 缺点：
    - 当前结果表明，新架构虽然提升了整体准确率，但没有解决关键目标，甚至重新引入了单边塌陷。
- **验证结果**：
  - 独立评估：`Accuracy=0.7343`，`Macro F1=0.5159`
  - 混淆矩阵表现：
    - `Bird`：`82/82` 全部识别正确
    - `FixedWing`：`0/29` 识别正确，`29` 条全部被判成 `RotaryDrone`
    - `RotaryDrone`：`23/32` 识别正确，`9` 条被判成 `FixedWing`
- **与上一轮对比**：
  - 相比 `Stage 29` 的 `Accuracy=0.7063`，这次整体准确率有所上升。
  - 但相比 `Stage 29` 的 `Macro F1=0.5379`，这次反而下降到 `0.5159`。
  - 更关键的是，`FixedWing` 从 `12/29` 正确重新退化为 `0/29`，说明这次升级没有守住最核心目标。
- **结论判断**：
  - `Stage 30` 没有真正完成“非鸟类内部边界增强”这个核心任务。
  - 当前升级大概率把模型进一步推向了更强的 `RotaryDrone` 偏置：
    - RD 显式谱结构分支可能把“旋翼微动样式”放大得过强。
    - 层级表示头当前只有主类监督，没有真实子类监督，`detail feature` 还不足以形成稳定的细粒度边界，反而可能强化了错误簇。
    - 轨迹显式模式分支虽然增强了运动表达，但当前公开数据集上的 `FixedWing` 运动模式仍未被稳定锚定出来。
  - 因此，这次升级说明：
    - “显式结构化建模”这个方向本身是对的；
    - 但当前实现还没有做到对 `FixedWing` 和 `RotaryDrone` 的平衡约束，导致表示朝 `RotaryDrone` 一侧塌陷。
- **下一步方向**：
  - 后续不应继续简单堆更多分支，而要回到判别约束本身：
    - 为非鸟类内部边界增加更直接的边界监督或度量约束；
    - 控制 RD 结构分支对单侧模式的过度放大；
    - 让轨迹分支与 RD 分支在非鸟类样本上形成更一致的反证机制，而不是某一侧主导。


## Stage 32 - 粗类映射与组内边界损失首次验证

- **当前阶段**：在 `Stage 30` 的结构升级基础上，引入粗类映射与组内 hardest-negative 边界损失，尝试直接约束 `FixedWing` 与 `RotaryDrone` 的类内边界。
- **当前所面临的问题**：
  - `Stage 31` 说明单纯加强结构表达并不能保证边界真正传导到最终输出。
  - 因此需要把问题进一步收敛到“细粒度边界监督本身”。
- **解决方案**：
  - 引入 `coarse_label_map`，当前把 `Bird` 映射到 `BiologicalTarget`，把 `FixedWing` 与 `RotaryDrone` 映射到 `AircraftTarget`。
  - 新增 `detail_logits` 支路。
  - 在同一粗类内，对目标类与 hardest negative 类施加 margin 约束。
- **方案利弊**：
  - 优点：
    - 比单纯结构增强更直接地约束非鸟类内部边界。
    - 设计可扩展到未来 `5` 大类、`20+` 子类。
  - 缺点：
    - 当前版本的边界监督还没有直接进入最终决策链，只是辅助信号。
- **验证结果**：
  - 运行目录：`artifacts/202603251446-5-4-cuda`
  - 独立评估：`Accuracy=0.6294`，`Macro F1=0.4082`
  - `FixedWing` 再次塌陷为 `0/29`。
- **结论判断**：
  - 仅仅增加边界损失还不够；如果边界支路不能直接影响最终输出，它仍然可能被主分类路径淹没。

## Stage 33 - 细粒度边界残差注入验证

- **当前阶段**：把受边界监督的 `detail_logits` 以小残差直接注入最终输出，验证细粒度边界信号能否真正作用于最终决策。
- **当前所面临的问题**：
  - `Stage 32` 说明边界监督本身已经存在，但没有有效进入最终判别链路。
  - 核心矛盾已经不是“有没有边界信号”，而是“边界信号能不能真正改变最终分类”。
- **解决方案**：
  - 新增 `detail_logit_residual_weight`。
  - 在最终 `logits` 上叠加 `detail_logits` 的小残差，让组内边界监督结果直接进入最终输出。
  - 保留 `coarse + detail` 层级表示、粗类辅助监督和组内边界损失。
- **方案利弊**：
  - 优点：
    - 直接打通了“边界监督 -> 最终决策”的链路。
    - 没有回退到只靠单一模态或手工规则的补丁方案。
    - 设计仍然面向未来大规模真实数据集扩展，后续可自然扩展到子类监督。
  - 缺点：
    - 当前 `RotaryDrone` 仍然偏强，`FixedWing` 召回还不够高。
    - 仍需后续继续压缩 `FixedWing -> RotaryDrone` 的误判比例。
- **验证结果**：
  - 运行目录：`artifacts/202603251500-5-4-cuda`
  - 独立评估：`Accuracy=0.8531`，`Macro F1=0.7372`
  - 混淆矩阵表现：
    - `Bird`：`82/82` 正确
    - `FixedWing`：`9/29` 正确，`19` 条被判成 `RotaryDrone`
    - `RotaryDrone`：`31/32` 正确，仅 `1` 条被判成 `FixedWing`
- **与历史结果对比**：
  - 相比 `Stage 24` 的历史最好结果 `Accuracy=0.7972`、`Macro F1=0.5644`，本次结果明显提升。
  - 相比 `Stage 29` 的 `Accuracy=0.7063`、`Macro F1=0.5379`，本次结果也大幅提升。
  - 相比 `Stage 31` 与 `Stage 32` 的失败验证，本次首次证明“边界监督直接进入最终输出”是有效方向。
- **结论判断**：
  - 这次结果说明，真正命中根因的关键，不只是增加结构分支或增加边界损失，而是必须让“细粒度边界表示”直接参与最终决策。
  - 当前模型已经明显逼近“高精度识别主路线”，并且其结构形式能够自然扩展到未来约 `33 万` 数据规模、`5` 大类、`20+` 子类场景。


## Stage 34 - 粗类概率门控细粒度残差验证

- **当前阶段**：在 `Stage 33` 的基础上，进一步用粗类概率门控 `detail logits` 残差，尝试在保住 `Bird` 的同时继续优化 `FixedWing` 与 `RotaryDrone` 的内部边界。
- **当前所面临的问题**：
  - `Stage 33` 虽然取得了当前最好结果，但仍然存在 `FixedWing -> RotaryDrone` 误判偏多的问题。
  - 为了减少细粒度边界残差对 `Bird` 的潜在误导，需要验证“只在模型自己认为属于 `AircraftTarget` 时放大细粒度边界信号”是否更稳。
- **解决方案**：
  - 在最终输出层，把 `detail logits` 残差乘以粗类概率门控。
  - 让 `detail` 分支继续只聚焦粗类内部存在歧义的类别。
  - 对当前数据集，意味着细粒度边界信号主要服务于 `FixedWing` 与 `RotaryDrone`。
- **方案利弊**：
  - 优点：
    - 进一步强化了“粗类先决策，细类再细分”的层级逻辑。
    - 从设计上更不容易让 `Bird` 被错误拉进 `AircraftTarget`。
  - 缺点：
    - 过强的粗类门控会削弱细粒度残差对困难 `FixedWing` 样本的修正能力。
    - 如果基础主干本身已经偏向 `RotaryDrone`，这类门控可能会进一步放大这种保守偏置。
- **验证结果**：
  - 运行目录：`artifacts/202603251557-5-4-cuda`
  - 独立评估：`Accuracy=0.8252`，`Macro F1=0.6572`
  - 混淆矩阵表现：
    - `Bird`：`82/82` 正确
    - `FixedWing`：`4/29` 正确，`21` 条被判成 `RotaryDrone`，`4` 条被判成 `Bird`
    - `RotaryDrone`：`32/32` 正确
- **与 Stage 33 对比**：
  - `Bird` 保持稳定，没有再出现额外的 `Bird` 边界问题。
  - 但 `FixedWing` 召回从 `9/29` 下降到 `4/29`，说明这次门控把模型推向了更保守的 `RotaryDrone` 偏置。
  - 整体结果低于 `Stage 33` 的 `Accuracy=0.8531`、`Macro F1=0.7372`。
- **结论判断**：
  - 这次验证说明：
    - “细粒度边界信号直接进入最终决策”这个方向是对的；
    - 但“再额外叠加过强的粗类概率门控”会削弱困难 `FixedWing` 样本的纠偏力度。
  - 当前最佳主路线仍然是 `Stage 33`，而不是这次门控版本。
- **下一步方向**：
  - 后续若继续压缩 `FixedWing -> RotaryDrone` 误判，更应该围绕 `FixedWing` 的召回补偿与边界强化展开，而不是继续压低 `detail` 残差的作用强度。


## Stage 35 - 验证集驱动的组内 Logit 校准

- **当前阶段**：在 `Stage 33` 的最佳主路线之上，引入验证集驱动的组内 logit 校准，专门压缩 `FixedWing -> RotaryDrone` 的剩余误判。
- **当前所面临的问题**：
  - `Stage 33` 已经证明“细粒度边界表示直接进入最终决策”是有效方向，但 `FixedWing` 召回仍然偏低。
  - 这说明当前主模型已经有可用边界信息，但最终类间决策阈值在 `AircraftTarget` 内部仍然偏向 `RotaryDrone`。
- **解决方案**：
  - 使用最佳 run `artifacts/202603251500-5-4-cuda` 的验证集 logits。
  - 只在同一粗类内部做 logit bias 搜索，不影响 `Bird` 这类已稳定分开的类别。
  - 当前公开数据集自动搜索得到的最优校准 bias 为：
    - `Bird: 0.0`
    - `FixedWing: +0.9`
    - `RotaryDrone: -0.9`
  - 把该 bias 写入新的校准 checkpoint，并通过正式评估链路验证。
- **方案利弊**：
  - 优点：
    - 不改动已验证有效的主干表达，只修正组内决策阈值偏置。
    - 只作用于粗类内部，避免破坏 `Bird` 的稳定边界。
    - 形式可扩展到未来 `5` 大类、`20+` 子类：每个粗类内部都可以独立做验证集驱动校准。
  - 缺点：
    - 这是后处理校准，不等于主干表示本身已经彻底解决所有内部边界问题。
    - 后续在真实大规模数据集上仍然需要验证该校准策略是否稳定。
- **验证结果**：
  - 校准产物目录：`artifacts/202603251624-5-4-cuda`
  - 独立评估：`Accuracy=0.8671`，`Macro F1=0.7673`
  - 混淆矩阵表现：
    - `Bird`：`82/82` 正确
    - `FixedWing`：`11/29` 正确，`18` 条被判成 `RotaryDrone`
    - `RotaryDrone`：`31/32` 正确，仅 `1` 条被判成 `FixedWing`
- **与历史结果对比**：
  - 相比 `Stage 33` 的 `Accuracy=0.8531`、`Macro F1=0.7372`，本次结果继续提升。
  - 相比旧历史最好 `Stage 24` 的 `Accuracy=0.7972`、`Macro F1=0.5644`，提升更明显。
- **结论判断**：
  - 当前最有效的路线已经明确：
    - 主干使用 `Stage 33` 的边界表示增强；
    - 决策层使用验证集驱动的组内 logit 校准。
  - 这说明当前残余问题更像“组内决策阈值偏置”，而不是“完全缺失边界信息”。
  - 因此，下一阶段不应急于继续大改主干，而应先把这条“边界表示 + 组内校准”的主路线稳定下来，并在后续真实大规模数据集上验证是否仍然成立。
## Stage 36 - Cosine Detail Head For Intra-Coarse Separation

- **Current Stage**: continue the main line on CPU after reviewing the latest best calibrated path.
- **Current Problem**:
  - The current best path already solves cross-group separation well.
  - The remaining bottleneck is still inside the same coarse group, especially `FixedWing` vs `RotaryDrone`.
  - The existing detail head is still a standard linear classifier, which may be too weak for fine-grained intra-group separation.
- **Solution**:
  - Replace the detail classifier with a cosine-based classifier.
  - Normalize both detail features and class weights, and use a learnable scale.
  - Keep the rest of the hierarchical path unchanged so the update remains low-risk and CPU-friendly.
- **Pros**:
  - More general than a current-dataset-specific patch.
  - Better aligned with future `5` main classes and `20+` subclasses.
  - Usually improves fine-grained boundaries without requiring a larger backbone.
- **Cons**:
  - Needs another manual training run to verify whether the better angular separation really improves `FixedWing` recall.
  - If the upstream RD and track features are still weak, the cosine head alone cannot solve all fine-class confusion.
## Stage 37 - Validation After Cosine Detail Head

- **Current Stage**: validate the CPU main line after introducing the cosine-based detail classifier.
- **Current Problem**:
  - Before this run, the main bottleneck was still intra-coarse confusion inside `AircraftTarget`.
  - `FixedWing` and `RotaryDrone` still needed a stronger fine-grained boundary.
- **Solution**:
  - Keep the hierarchical path and calibration path unchanged.
  - Replace the detail classifier with a cosine-based classifier to improve angular separation between fine classes.
  - Keep the implementation CPU-friendly and general for future larger datasets.
- **Validation Result**:
  - Manual run: `artifacts/202603252213-5-4-cpu`
  - Independent evaluation:
    - `Accuracy = 0.9301`
    - `Macro F1 = 0.8900`
    - `Loss = 0.3557`
  - Confusion matrix summary:
    - `Bird`: `82/82` correct
    - `FixedWing`: `28/29` correct
    - `RotaryDrone`: `23/32` correct, with remaining leakage mainly into `FixedWing`
- **Pros**:
  - This is a real generalization improvement, not just a cosmetic metric change.
  - The current path now separates `Bird` perfectly and recovers `FixedWing` strongly.
  - The method remains applicable to future `5` main classes and `20+` subclasses.
- **Cons**:
  - The remaining bottleneck has shifted: current recall loss now mainly comes from `RotaryDrone -> FixedWing` confusion.
  - The next step should focus on making intra-aircraft boundaries more balanced, rather than only increasing `FixedWing` recall further.

## Stage 38 - CPU Group Calibration On The New Best Checkpoint

- **Current Stage**: apply validation-driven group logit calibration on top of the new CPU best checkpoint.
- **Current Problem**:
  - The cosine detail head already improved intra-coarse separation a lot.
  - The remaining confusion is still mostly `RotaryDrone -> FixedWing` inside `AircraftTarget`.
- **Solution**:
  - Keep the trained backbone unchanged.
  - Use the validation set to optimize a coarse-group-aware logit bias.
  - Apply the learned bias only as post-training calibration.
- **Validation Result**:
  - Source checkpoint: `artifacts/202603252213-5-4-cpu`
  - Calibrated artifact: `artifacts/calibrated/202603252213-5-4-cpu-calibrated`
  - Independent evaluation after calibration:
    - `Accuracy = 0.9371`
    - `Macro F1 = 0.9012`
    - `Loss = 0.3766`
  - Learned bias:
    - `Bird: 0.0`
    - `FixedWing: -0.9`
    - `RotaryDrone: +0.9`
- **Pros**:
  - CPU-friendly and training-free improvement.
  - Keeps the current backbone intact and only corrects the remaining intra-group threshold bias.
  - General enough to extend to future multi-class and multi-subclass settings.
- **Cons**:
  - This is still a post-training calibration step, not a replacement for better representation learning.
  - Future larger datasets still need to verify whether the same bias magnitude remains stable.
## Stage 39 - Fine-Grained Calibration Becomes The Main Follow-Up Direction

- **Current Stage**: after the calibrated CPU best result, the backbone can temporarily stabilize and the main follow-up direction shifts to fine-grained calibration and intra-group decision balance.
- **Current Problem**:
  - The main trunk is already strong enough for the current sample dataset.
  - The remaining errors are no longer broad class collapses, but mainly residual fine-grained confusion inside the same coarse group.
- **Solution**:
  - Standardize calibrated outputs under `artifacts/calibrated/`.
  - Treat calibration as an explicit stage in the main workflow instead of a one-off side experiment.
  - Use the current best backbone as the stable base and focus future work on finer decision calibration rather than immediately changing the backbone again.
- **Pros**:
  - Keeps the engineering path clean and easier to scale.
  - Prevents unnecessary backbone churn when the current representation is already strong.
  - Matches the future direction of larger taxonomies with more subclasses.
- **Cons**:
  - Calibration is still downstream of representation learning, so it cannot permanently replace stronger future feature learning if the data distribution changes a lot.


## Stage 40 - Excel To TSV Real Track Pipeline

- **Current Stage**: shift the project from the old public-session path toward the newly acquired real Excel dataset.
- **Current Problem**:
  - The current real dataset provides track data and manual labels in paired Excel files, not RD cubes.
  - The old default path still assumes `data/sessions` and an RD-enabled workflow.
  - Raw vendor export formats should not be read directly by the model stack.
- **Solution**:
  - Add a standardized `Excel -> TSV -> data/tsv` pipeline.
  - Keep `pro/` as the processing and human-review workspace.
  - Keep `data/` as the training source only.
  - Preserve the RD route as a future extension, but disable it in the default real-data training path.
- **Pros**:
  - The training stack now depends on a normalized schema rather than a vendor-specific export format.
  - Future radar exports can be supported by adding processors without rewriting the model stack.
  - This fits the current real dataset, which is track-only.
- **Cons**:
  - The old session-based `mat` path and the new TSV path now coexist, so the codebase temporarily contains two ingestion routes.
  - The track dataset reader still needs a dedicated adapter in the next step before the new TSV route becomes the full default training path.


## Stage 41 - TSV Training Adapter For Real Excel Tracks

- **Current Stage**: make the standardized real Excel TSV dataset usable by the training stack without reading raw Excel files directly.
- **Current Problem**:
  - The new `prepare-tsv` pipeline can already generate `pro/tsv` and `data/tsv`.
  - The training stack still expected the old `data/sessions` structure and RD-related files.
- **Solution**:
  - Add a dedicated TSV dataset adapter.
  - Add a TSV-aware session scanner and split builder.
  - Switch the default training data root to `data/tsv` and disable RD by default for the current real dataset.
- **Pros**:
  - The model stack is now insulated from raw Excel parsing.
  - The current real dataset can enter training through a stable standardized interface.
  - Future radar exports can be supported by adding processors instead of rewriting the training code.
- **Cons**:
  - The codebase still carries the old session/RD path in parallel for future reuse.
  - The next phase still needs training-side validation on the new TSV route.


## Stage 42 - Environment Variable Modality Control

- **Current Stage**: simplify runtime control and make modality selection independent from vendor-specific data paths.
- **Current Problem**:
  - RD and track switches were previously scattered across CLI flags and config defaults.
  - This made the command surface noisy and harder to align with the new real-data route.
- **Solution**:
  - Use environment variables as the single runtime control interface for modality switches.
  - `TRACK=true|false` controls the track branch.
  - `RD=true|false` controls the RD branch.
  - Remove redundant CLI flags that duplicated the same modality decisions.
- **Pros**:
  - Cleaner command surface.
  - Easier to switch between track-only, RD-only, and dual-branch modes without changing code.
  - Fits the new processor-based data architecture better.
- **Cons**:
  - Users now need to remember the environment variable layer.
  - The RD path still remains a future extension for datasets that actually include RD inputs.


## Stage 43 - Default Environment Switches Stay Fully Enabled

- **Current Stage**: keep environment-variable control as the runtime interface, but set both modality defaults to enabled.
- **Current Problem**:
  - The current real Excel dataset is track-first, yet the project should remain ready for future RD-enabled datasets.
  - Defaulting RD off would better fit the current data, but would understate the intended long-term dual-branch architecture.
- **Solution**:
  - Keep `TRACK=true` and `RD=true` as the global defaults in config.
  - Document that the current recommended real-data commands still explicitly use `TRACK=true RD=false` for the Excel track route.
- **Pros**:
  - Preserves the long-term dual-branch architecture as the default project posture.
  - Keeps runtime control explicit and flexible.
- **Cons**:
  - Current users must still explicitly turn RD off when working with the present Excel-only dataset.


## Stage 44 - TSV Track-Only Route Reaches Trainable State

- **Current Stage**: finish the last compatibility fixes so the real Excel TSV route can enter actual training.
- **Current Problem**:
  - The TSV route already supported dataset loading and forward passes, but the trainer still carried old session-manager interfaces and old RD assumptions.
  - The old coarse taxonomy also remained tied to the previous three-class route.
- **Solution**:
  - Restore trainer-compatible session-manager wrappers.
  - Make runtime modality logic consistently follow `TRACK` and `RD` environment variables.
  - Replace the old three-class coarse taxonomy with a real-label coarse taxonomy for the current TSV dataset.
  - Reduce RD placeholder overhead for the track-only route.
- **Pros**:
  - The real TSV route is now trainable without going back to the old public session dataset.
  - The training stack remains extensible for future RD-enabled datasets.
- **Cons**:
  - The current route is still track-first; RD-specific learning remains deferred until a real RD processor is available.


## Stage 45 - Remove Residual Old-Dataset Type Coupling

- **Current Stage**: fix the last runtime blockers exposed when the real TSV route entered actual training.
- **Current Problem**:
  - The training stack still contained residual references to `RadarFusionDataset` in type annotations and direct construction paths.
  - This caused import-time failures when the real TSV route was used.
- **Solution**:
  - Remove hard coupling to the old dataset class in trainer, evaluator, and calibration.
  - Route dataset construction uniformly through the TSV-aware dataset factory.
  - Keep compatibility wrappers for session manifests and split saving.
- **Pros**:
  - The real TSV route can now enter training without import-time failures.
  - The stack becomes more data-source-agnostic.
- **Cons**:
  - The codebase still carries compatibility layers for the older session/RD route until the future RD processor is reintroduced.


## Stage 46 - Restore Missing Normalization Interfaces For TSV Training

- **Current Stage**: resolve the last trainer-to-dataset compatibility gap before the first real TSV training run.
- **Current Problem**:
  - The TSV dataset already supported loading and forward passes, but the trainer still expected normalization-fit methods from the older dataset implementation.
  - This caused runtime failures inside `build_dataloaders`.
- **Solution**:
  - Add normalization-fit helpers and session indexing to the TSV dataset.
  - Keep method names aligned with the trainer so the existing training loop can remain stable.
- **Pros**:
  - The TSV route now matches the trainer contract and can enter real training.
  - No extra special-case training path is needed.
- **Cons**:
  - The compatibility surface is still larger than a clean-slate TSV-only trainer, but this is acceptable during the transition period.


## Stage 47 - Adopt The A-Scheme Coarse Taxonomy For The Real TSV Route

- **Current Stage**: prefer a more stable coarse hierarchy for the current real TSV dataset before moving toward the future 5-main-class business taxonomy.
- **Current Problem**:
  - The real TSV dataset currently has 11 fine labels with strong class imbalance.
  - A direct business-style 5-group taxonomy is attractive long-term, but too coarse for the current sample size and could make intra-group learning unstable.
- **Solution**:
  - Switch to the A-scheme coarse taxonomy:
    - `HumanTarget`
    - `BirdTarget`
    - `DroneTarget`
    - `AircraftTarget`
    - `SurfaceTarget`
    - `UnknownTarget`
    - `NoiseTarget`
- **Pros**:
  - Better aligned with the current data reality.
  - Easier to learn than a too-coarse 5-group hierarchy.
  - Still compatible with a future move toward 5 main classes and more subclasses.
- **Cons**:
  - This is a transitional hierarchy rather than the final business-facing taxonomy.


## Stage 48 - Train The Current Real TSV Route On A-Scheme Coarse Labels

- **Current Stage**: prioritize a usable version on the current sample dataset by training directly on the accepted A-scheme coarse targets.
- **Current Problem**:
  - The current sample dataset is too sparse and imbalanced for stable 11-subclass training.
  - A fine-label objective wastes epochs on boundaries that the current sample size cannot support reliably.
- **Solution**:
  - Keep fine labels in the TSV data for future expansion.
  - Switch the current training target to the 7-class A-scheme coarse labels.
  - Use the existing track-first route to optimize a usable coarse classifier first.
- **Pros**:
  - Much more realistic for the current sample dataset.
  - Keeps the long-term fine-label path intact in the data layer.
  - Better aligned with the immediate goal of producing a usable version.
- **Cons**:
  - Current outputs are coarse classes rather than the final desired fine taxonomy.
  - Fine-grained subclass learning must return when the larger real dataset arrives.


## Stage 49 - Restore The Best Validated TSV Track Baseline

- **Current Stage**: prepare a stable baseline before introducing the graph-based track backbone.
- **Current Problem**:
  - The worktree had accumulated multiscale, notes-supervision, larger-backbone, and expert-head experiments that did not beat the best real TSV run.
  - Continuing from that mixed state would make it hard to judge whether a graph track backbone truly helps.
- **Solution**:
  - Restore the validated track backbone behind `artifacts/202603312215-5-4-cpu`.
  - Revert the default track encoder width/depth and the global feature layout to the best empirically validated TSV baseline.
  - Keep later compatibility hooks in place, but treat the restored baseline as the clean starting point for the next architecture experiment.
- **Pros**:
  - Future graph-track experiments will be compared against a clean, known-good baseline.
  - Avoids attributing gains or losses to leftover unsuccessful experiments.
- **Cons**:
  - Some later exploratory code paths remain in the repository, even though the default baseline no longer depends on them.


## Stage 50 - Add Graph-Based Track Backbone On Top Of The Restored TSV Baseline

- **Current Stage**: introduce a graph-based track backbone after restoring the best validated TSV baseline.
- **Current Problem**:
  - The restored temporal track baseline remains the best validated real-data result, but it appears to be approaching its ceiling on the final track-only dataset.
  - Repeated scaling and auxiliary heads did not consistently beat the best baseline, suggesting that the next useful change should target the track representation itself.
- **Solution**:
  - Keep the restored temporal track backbone as the default baseline.
  - Add a selectable `graph` track backbone that builds a temporal-similarity graph directly from the track sequence and applies graph message passing before classification.
  - Wire the backbone choice through config so training, evaluation, and calibration can switch between `temporal` and `graph` without changing dataset code.
- **Pros**:
  - Provides a new representation family while preserving the validated baseline.
  - Keeps the TSV track-only pipeline unchanged.
  - Makes future experiments easier to compare because the backbone switch is explicit.
- **Cons**:
  - This is a new experimental branch and has not yet beaten the restored temporal baseline.
  - Additional graph construction cost may not always translate into better accuracy.


## Stage 51 - Centralize Runtime Control Into train.json And Clean Low-Value Artifacts

- **Current Stage**: replace scattered runtime switches with a single JSON control file and remove clearly low-value artifacts.
- **Current Problem**:
  - Runtime behavior was spread across YAML defaults, CLI overrides, and environment variables.
  - The `artifacts` directory accumulated poor runs and temporary evaluation folders that made iteration harder to track.
- **Solution**:
  - Introduce `train.json` as the single runtime control file for training, evaluation, calibration, and session inspection.
  - Route track/RD switches and the track-backbone choice through `train.json`.
  - Add an artifact cleaner that removes obvious failed runs and temporary evaluation folders while preserving known-good baselines and calibrated outputs.
- **Pros**:
  - Easier and more reproducible experimentation.
  - Cleaner artifact space for future comparisons.
  - Reduced dependence on ad-hoc environment variable state.
- **Cons**:
  - The CLI and docs must now stay aligned with `train.json` semantics.
  - Care is required to avoid deleting runs that might still be useful for analysis.


## Stage 52 - Evaluate The Graph Track Backbone Against The Restored TSV Baseline

- **Current Stage**: compare the new graph-based track backbone against the restored temporal TSV baseline.
- **Current Problem**:
  - The temporal baseline is stable but appears close to its ceiling on the final track-only dataset.
  - A graph-based backbone was introduced to test whether explicit temporal-similarity message passing can improve coarse target discrimination.
- **Solution**:
  - Run a controlled comparison using the same TSV data split and the same coarse-label objective.
  - Keep all runtime controls centralized in `train.json`, switching only `track_backbone_type` between `temporal` and `graph`.
- **Pros**:
  - The graph branch trained successfully and confirmed that the new backbone is technically usable.
  - Parameter count dropped significantly, which may still be useful for future efficiency-constrained deployments.
- **Cons**:
  - The graph backbone did not beat the restored temporal baseline on the current final dataset.
  - The best graph run still underperformed the established best real TSV track-only result, so it remains an experimental branch rather than the new default.


## Stage 53 - Add TR-Only Supplemental Sessions From pro/raw/TR

- **Current Stage**: supplement the final track-only dataset with additional TR trajectories from `pro/raw/TR` to improve class balance at the coarse-target level.
- **Current Problem**:
  - The original final TSV dataset was highly imbalanced, especially for aircraft and surface-related categories, while some coarse classes remained under-supported.
  - The user provided additional TR-only `.mat` files whose filenames already contain usable class hints.
- **Solution**:
  - Add a dedicated TR-only supplement processor that reads v7.3 MAT files with `h5py`.
  - Map filename classes into the current token space and sync the resulting TSV sessions into `data/tsv`.
  - Keep the supplement pipeline separate from the Excel pipeline so future data processors can be added without touching the model path.
- **Pros**:
  - The dataset now includes many more sessions and significantly better support for `bird`, `drone`, `large_aircraft`, `vehicle`, and `vessel`.
  - The supplement processor is reusable for future TR-only batches.
- **Cons**:
  - The supplement still does not solve the scarcity of `human`, `clutter`, `small_aircraft`, and `suspected_balloon`.
  - The first training run after supplementation still did not beat the best validated TSV baseline.


## Stage 54 - Merge Suspected Balloon And Clutter Into Other And Rebuild Session Splits

- **Current Stage**: make the coarse target objective more practical and make evaluation less dependent on a single test session.
- **Current Problem**:
  - `suspected_balloon` and `clutter` are both extremely low-support edge categories and their separation does not help the current practical objective.
  - The previous split strategy effectively tied validation and testing to too few sessions, making the final metric overly sensitive to one difficult session.
- **Solution**:
  - Merge `suspected_balloon` and `clutter` into a shared coarse target, `OtherTarget`, while keeping `unknown` as `UnknownTarget`.
  - Replace the old single-session holdout split heuristic with a multi-session greedy split that preserves label coverage while allocating multiple validation and test sessions.
- **Pros**:
  - The new coarse objective is more consistent with practical use and better matched to the current data scarcity.
  - Validation and test metrics should become more stable and more representative of the full dataset.
- **Cons**:
  - Results before and after this split change are not directly comparable one-to-one.
  - Extremely rare classes still remain a structural limitation of the final dataset.


## Stage 55 - Try Long-Tail Balancing And Focal Loss Under The Stricter Multi-Session Split

- **Current Stage**: test whether stronger long-tail balancing improves the new stricter evaluation setting.
- **Current Problem**:
  - Under the multi-session split, the model collapsed badly and the failure pattern suggested strong class imbalance and coarse-target confusion.
  - The previous coarse training path had disabled class-balanced loss, which was too weak for the new distribution.
- **Solution**:
  - Re-enable class-balanced weighting in coarse training.
  - Add focal-style cross-entropy to reduce domination by easy majority examples.
  - Re-run training on the stricter multi-session split.
- **Pros**:
  - The validation curve showed some early positive signal and confirmed that the loss redesign is numerically stable.
  - The approach remains generic and reusable for future long-tail settings.
- **Cons**:
  - On the current final dataset and stricter split, the new run still underperformed the earlier baseline and did not solve the generalization gap.
  - This suggests the main problem is not only long-tail weighting, but also strong session/domain shift between train and test groups.


## Stage 56 - Strengthen The Dual-Modal Temporal Backbone With Real Multi-Window Fusion

- **Current Stage**: continue improving the dual-modal final-dataset backbone without replacing the validated temporal route.
- **Current Problem**:
  - The restored dual-modal baseline still underfits `BirdTarget` and tends to confuse it with `SurfaceTarget`.
  - The previous so-called multi-window path did not truly inject short-window and long-window context into the temporal encoder for the session-based dual-modal route.
  - Training samples remain limited, so naive backbone widening alone is unlikely to be sample-efficient.
- **Solution**:
  - Make the session-based dataset emit `track_short_features` and `track_long_features` just like the TSV route.
  - Upgrade the temporal encoder with a lightweight time-by-feature 2D convolution branch plus explicit short/mid/long context fusion.
  - Add train-only window-mix augmentation so long-window features become more robust under small-sample conditions.
- **Pros**:
  - Increases temporal representation capacity without discarding the validated temporal backbone.
  - Uses small-sample-friendly augmentation instead of only increasing parameter count.
  - Preserves compatibility with the current fusion and RD branches.
- **Cons**:
  - Adds extra temporal-path complexity and may require retuning of loss balance and learning rate.
  - The gain is not guaranteed because the current bottleneck may still partly come from split bias and class support.


## Stage 57 - Add A Bird-vs-Surface Specialist To The Dual-Modal Fusion Path

- **Current Stage**: target the dominant coarse-class error after strengthening the temporal backbone.
- **Current Problem**:
  - The main dual-modal failure mode is still `BirdTarget` collapsing into `SurfaceTarget`.
  - Generic backbone strengthening improved `AircraftTarget`, but did not recover `BirdTarget`.
- **Solution**:
  - Add a dedicated `bird_surface_head` trained only on `BirdTarget` and `SurfaceTarget` samples.
  - Use its logit both as an auxiliary supervision target and as a lightweight residual correction on the final class logits.
  - Keep the rest of the architecture unchanged so the experiment remains attributable.
- **Pros**:
  - Directly attacks the dominant current error boundary.
  - Low-intrusion change that preserves the validated dual-modal baseline.
- **Cons**:
  - If the root cause is still RD representation instability, the expert head may only partially help.
  - Adds another task-specific branch that must be watched for overfitting.


## Stage 58 - Switch Track Backbone Toward DG-GCN

- Current stage: replace the reinforced temporal default with a DG-GCN-style track backbone.
- Current problem: the temporal backbone, even after multi-window and 2D temporal-feature convolution upgrades, still failed to recover BirdTarget under the final dual-modal split.
- Solution: introduce a stronger graph-based track encoder with local adjacency, similarity adjacency, dynamic edge learning, graph message passing, and graph-summary fusion.
- Pros and cons:
  - Pros: more suitable for track relation modeling under strong session shift and class-boundary ambiguity.
  - Cons: graph construction introduces more sensitivity to feature normalization and may require more careful tuning than the simpler temporal baseline.


## Stage 59 - Revert Default Track Backbone After DG-GCN Regression

- Current stage: keep DG-GCN as an experimental option but restore the validated temporal backbone as the default route.
- Current problem: the first DG-GCN production attempt regressed from 67.49% to 30.35% accuracy under the final dual-modal split and failed to recover BirdTarget.
- Solution: revert the default training route to temporal, preserve DG-GCN behind the backbone switch, and continue optimization from the stable line.
- Pros and cons:
  - Pros: avoids wasting training rounds on a regressed default route and keeps a reproducible stable baseline.
  - Cons: DG-GCN remains unfinished and will need smaller, controlled experiments later.


## Stage 60 - Shift Fusion to RD-Dominant and Add Bird-Specialized RD Heads

- Current stage: move the dual-modal route back to RD-first optimization while keeping track as support.
- Current problem: repeated experiments showed that BirdTarget still collapses, and continued track-backbone upgrades did not solve the boundary.
- Solution: bias modality fusion toward RD, strengthen the RD unimodal residual path, and add Bird-sensitive RD auxiliary heads, especially for BirdTarget vs SurfaceTarget.
- Pros and cons:
  - Pros: better aligned with the historical high-performing route and the future real radar deployment path.
  - Cons: if RD data quality is inconsistent across sessions, RD-dominant fusion can amplify RD-side noise and requires careful validation.


## Stage 61 - Soften RD-First Bias and Move Bird Optimization into RD Representation

- Current stage: keep the RD-first direction but pull back the overly strong RD fusion bias after a large regression.
- Current problem: hard RD dominance and classifier-level Bird RD experts damaged SurfaceTarget and still failed to recover BirdTarget.
- Solution: reduce RD fusion bias, reduce RD specialist loss weights, and move Bird-sensitive enhancement into the RD representation path itself.
- Pros and cons:
  - Pros: closer to the historical high-performing pattern where strong RD representation came before aggressive fusion bias.
  - Cons: gains may be slower because the improvement is pushed into representation learning rather than explicit decision-time correction.


## Stage 62 - Add Bird-Surface Prototype Constraint on RD Features

- Current stage: move Bird discrimination from logit-level correction into the RD feature space itself.
- Current problem: BirdTarget remained at zero recall even after a Bird-aware RD branch, which means the Bird cluster was still not stable enough in embedding space.
- Solution: add a Bird-vs-Surface prototype-style RD feature loss that pulls Bird and Surface samples toward their own prototypes while enlarging inter-class separation.
- Pros and cons:
  - Pros: directly shapes the RD embedding space instead of only repairing the final decision layer.
  - Cons: prototype losses depend on class presence within a batch and can be noisy when Bird samples are sparse.

Stage 63 - Bird/Surface Sampling and Triplet Constraint
- Problem: Bird sessions remain consistently collapsed into SurfaceTarget even after softer RD-first fusion and prototype loss.
- Solution: increase Bird/Surface sampling frequency in the training loader and add an RD-feature triplet-style separation loss on Bird vs Surface.
- Pros: directly attacks the dominant confusion pair at both sample distribution level and feature-space level.
- Cons: may overfocus on Bird/Surface and still requires validation against overall coarse accuracy.

Stage 64 - Soften Bird Reinforcement and Keep Bird-Aware Validation
- Problem: strong Bird/Surface sampling plus triplet loss caused overfitting and poor test transfer.
- Solution: reduce Bird/Surface sampling boost, disable triplet loss by default, and keep validation split explicitly Bird-aware so val remains representative of Bird test sessions.
- Pros: stabilizes training while preserving Bird coverage in validation.
- Cons: Bird remains hard and may still require further RD representation work.

Stage 65 - Track Router and RD Air Specialist Backbone
- Problem: the model still treats all classes through a mostly uniform fusion head, while Track is better at coarse routing and RD is better at hard air-target discrimination.
- Solution: add a track coarse router (Surface/Aircraft/AmbiguousAir) and an RD bird-drone specialist head, then inject their signals as structured residuals into final logits.
- Pros: matches the project strategy of Track for easy separation and RD for hard ambiguous air classes.
- Cons: adds more auxiliary branches and requires validation to avoid new overfitting.


## Stage 66 - Data Purification And Label Remap
- Freeze the current supervised training route to paired TR/RD sessions only under `data/sessions`.
- Keep `pro/excel` and TSV-derived datasets for review only; do not mix them into the dual-modal main training route.
- Update coarse labels to `AircraftTarget`, `BirdTarget`, `DroneTarget`, and `ShipTarget`.
- Normalize filename labels as: Bird -> BirdTarget, Rotary drone -> DroneTarget, Fixed-wing UAV -> DroneTarget, Helicopter -> AircraftTarget, Passenger ship -> ShipTarget, Speedboat -> ShipTarget.


## Stage 67 - Balanced Split Weight Tuning
- Recompute split after data purification and label remap.
- Adopt a more conservative loss weighting setup for 4-class paired TR/RD training.
- Set `class_balance_beta=0.995`, `focal_gamma=1.25`, and reduce `bird_surface_sampling_boost` to `1.25` to avoid overweighting Bird/Ship and worsening overfit.


## Stage 68 - Early Stopping And Composite Selection
- Stop using only `val_macro_f1` as the best-checkpoint criterion.
- Use a composite validation score: `0.7 * val_macro_f1 + 0.3 * val_accuracy`.
- Add early stopping with `patience=6` and `min_delta=0.002` to reduce overfitting under the new paired-session split.

## Stage 69 - Stabilized Training Schedule For Paired RD/TR
- Based on the first stable paired TR/RD run, reduced `learning_rate` from `3e-4` to `2e-4` and increased `epochs` from `5` to `12`.
- Keep composite checkpoint selection and early stopping active so the run can stop near the best validation point instead of overfitting after epoch 2.

## Stage 70 - Paired TR/RD Stable Baseline Improvement
- Run `202604081748-30-4-cpu` under the purified paired TR/RD setup reached `Accuracy=0.7529`, `Macro F1=0.7494`, `Loss=1.5857`.
- The stabilized training schedule (`lr=2e-4`, `epochs=12`, composite selection + early stopping) outperformed the previous paired-data baseline and produced a clearly stronger, more reliable reference point.
- Best validation behavior appeared around epoch 4, and early stopping triggered at epoch 10 before deeper overfitting.
