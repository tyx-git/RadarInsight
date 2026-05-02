# debug

## 2026-04-25
- 初始化新的架构变更跟踪文件。
- 当前项目根目录已重构为：
  - `OriData/`：原始数据集
  - `data/`：处理后的数据集
  - `artifacts/`：模型与训练产物
  - `reports/`：模型报告
  - `src/core/`：核心代码
  - `src/process/`：原始数据处理代码
  - `src/visual/`：可视化代码
- 后续所有涉及目录结构、模块职责、主流程入口调整的变更，都记录在此文件中。
- 方案进一步收敛：新增根目录 `target.md`，当前项目只使用：
  - `OriData/raw/TR`
  - `OriData/raw/RD`
  - `OriData/excel` 中的轨迹数据
- 暂停此前多源统一、扩类、复杂联合训练相关路线。
- 在 `src/process/` 下新增第一版处理脚手架：
  - `raw_excel_manifest.py`
  - `build_manifests.py`
- 第一阶段输出写入 `data/manifests/`，用于沉淀：
  - `raw/TR + raw/RD` 配对清单
  - `excel/tck + excel/bz` 配对清单
- 读取 `OriData/excel` 下 MATLAB 脚本后，已确认 Excel 轨迹字段语义主要对应：
  - 方位、距离、径向速度、仰角、高度、幅度、航向、速度、噪声、批次号等
- 已将这些字段含义写入 `excel_track_manifest.json`。
- 新增第一版统一轨迹 schema：
  - `src/process/schema.py`
- 新增第一版标准化导出器：
  - `src/process/standardize_tracks.py`
- 目标是先把：
  - `OriData/raw/TR`
  - `OriData/excel/tck`
 统一导出到 `data/processed/`，形成可继续使用的轨迹表。
- 将 `excel` 标准化导出的粒度从“整份时间戳文件”修正为“`timestamp + batch_id` 单元”。
- 当前 `data/processed/` 已不再只是分散 CSV，而是形成了稳定数据集结构：
  - `raw_tracks/`
  - `excel_tracks/`
  - `raw_track_index.csv`
  - `excel_track_index.csv`
  - `track_index.csv`
  - `summary.json`
- 当前处理后样本总量：
  - `raw_tr_rd = 77`
  - `excel_track = 588`
  - `total = 665`
- 在 `src/core/` 下新增第一版读取接口：
  - `processed_dataset.py`
  - 负责从 `data/processed/track_index.csv` 读取样本索引并按 `sample_id` 加载标准化轨迹表
- 当前标签层已升级为“双字段”：
  - `label_name_raw`：保留原始标签
  - `label_name`：统一后的标准标签
- 当前统一标签已经覆盖：
  - `AircraftTarget`
  - `BirdTarget`
  - `DroneTarget`
  - `ShipTarget`
  - `CarTarget`
  - `PeopleTarget`
  - `BalloonTarget`
  - `ClutterTarget`
  - `UnknownTarget`
- 当前仍保留少量需要后续单独处理的样本：
  - `MixedLabel`
- 当前已新增训练可用性筛选层：
  - `track_index_with_flags.csv`
  - `trainable_track_index.csv`
  - `trainable_summary.json`
- 当前默认训练可用规则：
  - 排除 `MixedLabel`
  - 排除 `UnknownTarget`
  - 排除 `record_count < 2`
  - 排除 `time_seconds_max - time_seconds_min <= 0`
- 当前训练可用样本数：
  - `626 / 665`
- 已补齐 `raw/RD` 处理支路：
  - `src/process/standardize_rd.py`
  - 输出 `data/processed/raw_rd/` 与 `raw_rd_index.csv`
- 已新增处理总入口：
  - `src/process/build_all_processed.py`
  - 当前可以一键重建：
    - `data/manifests/`
    - `data/processed/raw_tracks`
    - `data/processed/excel_tracks`
    - `data/processed/raw_rd`
    - 统一索引与训练可用索引
- 当前 `data/processed/pipeline_summary.json` 已能汇总：
  - manifest
  - tracks
  - raw rd
  - trainable
- 已开始搭建 `src/core` 第一版骨架：
  - `labels.py`
  - `features.py`
  - `samples.py`
  - `dataset.py`
  - `split.py`
- 当前 `src/core` 已支持：
  - 从 `data/processed/trainable_track_index.csv` 读取训练可用样本
  - 输出统一样本对象
  - 进行基础分层 train/val/test 切分
- 已继续补齐训练数据加载骨架：
  - `encoding.py`
  - `torch_dataset.py`
  - `collate.py`
  - `dataloaders.py`
- 当前已验证：
  - 可从 `626` 个训练可用样本构建 PyTorch 数据集
  - 可完成基础分层切分
  - 可输出带 padding 和 mask 的 batch
- 已继续补齐第一版可训练 baseline：
  - `models.py`
  - `metrics.py`
  - `train_config.py`
  - `trainer.py`
  - `train_baseline.py`
- 当前 baseline 采用：
  - GRU 序列分类器
  - 10维轨迹特征输入
  - 基础交叉熵训练
  - checkpoint / history / split manifest 输出到 `artifacts/`
## 2026-04-25 架构补齐：对照 main.md 完成 core 分层骨架
- 对照 `main.md` 审查当前架构后，确认此前 `src/core` 只有“处理后数据读取 + GRU baseline”，缺少总架构里明确要求的以下层次：
  - 轨迹建模层
  - 物理特征层
  - 深度门控路由层
  - 多专家层
  - 系统级推理编排层
- 本次补齐的文件：
  - `src/core/physical.py`
  - `src/core/trajectory.py`
  - `src/core/router.py`
  - `src/core/experts.py`
  - `src/core/system.py`
  - `src/core/architecture.py`
- 当前这些新增模块的定位：
  - 先补齐与 `main.md` 对齐的代码骨架与可导入接口
  - 不直接替换现有 `GRUSequenceClassifier` baseline
  - 让后续可以在现有 `data/processed` 与 dataloader 基础上继续演进到：
    - 双通道轨迹表征
    - 任务导向门控路由
    - 多专家并行分类
- 当前已完成的对齐：
  - `physical.py`：补了显式物理特征提取
  - `trajectory.py`：补了 Transformer 轨迹编码器和深度+物理融合表征构建器
  - `router.py`：补了门控路由 MLP 骨架
  - `experts.py`：补了低空可疑目标 / 常规空中目标 / 高空规则过滤 / 地面目标专家骨架
  - `system.py`：补了统一系统前向编排入口
  - `architecture.py`：把总架构中的 route 规格显式化
- 当前仍然只是“架构骨架”，尚未完成的部分：
  - 物理特征与 `main.md` 15维定义还不是一一完全等价，只是第一版可运行近似
  - 路由监督、专家监督、联合损失、按需激活逻辑尚未接入训练循环
  - RD 精分分支尚未真正实现，仅保留了未来对齐位置
- 轻量验证结果：
  - 新增文件已通过 `py_compile`
  - 新增模块可正常导入，不与现有 baseline 冲突
## 2026-04-25 第一版系统训练入口落地
- 在不推翻现有 baseline 的前提下，已把“第一版可训练系统”接起来：
  - `src/core/system_config.py`
  - `src/core/system_metrics.py`
  - `src/core/system_trainer.py`
  - `src/core/train_system.py`
- 同时为了让系统训练真正吃到 `main.md` 所要求的“轨迹深度特征 + 物理特征 + 路由标签”，补了：
  - `torch_dataset.py`：新增 `physical_features`、`route_name`、`route_index`
  - `collate.py`：新增 `physical_features` 和 `route_labels` batch 组织
- 当前 route 规则是第一版近似映射：
  - `Drone/Bird/Balloon/Clutter -> low_altitude_suspect`
  - `Aircraft -> conventional_air`
  - `People/Car/Ship -> ground_target`
  - 其余 -> high_altitude_filter`
- 当前系统训练做的是：
  - 共享轨迹表征
  - 路由监督
  - 专家分支输出
  - 映射回当前 8 类统一标签空间做分类监督
- 还没做的部分：
  - RD 精分分支
  - 任务导向按需激活
  - 更细的专家标签体系
  - 联合损失的完整版本
- smoke 结果：
  - 系统前向已通过
  - `artifacts/system_v1_smoke` 已成功产出 checkpoint / history / split manifest / summary
  - 1 epoch CPU smoke test metrics:
    - `accuracy ≈ 0.4842`
    - `macro_f1 ≈ 0.2237`
    - `route_accuracy ≈ 0.7684`
## 2026-04-25 系统级详细评估报告补齐
- 为了支撑后续反复调参与向 95% 准确率收敛，已在 `system_trainer.py` 中补齐详细测试报告输出。
- 当前系统训练完成后，除原有：
  - `best_model.pt`
  - `training_history.json`
  - `training_split_manifest.json`
  - `training_summary.json`
  之外，还会新增：
  - `evaluation_report.json`
- `evaluation_report.json` 当前包含：
  - 总体 summary
  - 分类 confusion matrix
  - per-class precision / recall / f1 / support
  - route confusion matrix
  - per-route support / recall
  - 按 source 的分类准确率
  - 每个测试样本的预测明细（真实类、预测类、真实路由、预测路由、是否分类正确、是否路由正确）
- 这一步的目标是把后续训练从“只看 accuracy”升级成“能定位具体哪类、哪路由、哪来源在拖后腿”。
## 2026-04-25 中文实验报告链打通
- 在保留 `artifacts/.../evaluation_report.json` 原始结构化评估结果的同时，新增：
  - `src/core/reporting.py`
- 当前系统训练结束后会自动把评估结果同步写到：
  - `reports/real/<experiment_name>.json`
  - `reports/real/<experiment_name>.md`
- markdown 报告当前包含：
  - 总体指标
  - 训练产物路径
  - 各类别指标表
  - 主要混淆对
  - 路由表现
  - 按来源统计
  - 典型错分样本前20条
- 这样后续每轮实验既保留机器可读 JSON，也保留人工可读中文报告，便于长期对比与逼近 95% 准确率目标。
## 2026-04-25 系统训练稳定性与 RD 接入修复
- 针对 `system_v1` 后期出现 `loss=NaN` 的问题，已先做第一轮稳定性修复：
  - 在 `system_trainer.py` 中加入 `clip_grad_norm_`
  - 把非有限 `loss` 检查前移到 `backward()` 之前
- 同时查明并修复了“RD 没进训练链”的问题：
  - 之前 `src/process/standardize_rd.py` 只生成了 `raw_rd` 摘要索引，没有真正进入 `src/core` 训练前向
  - 现在新增 `src/core/rd.py`
  - `TorchTrackDataset` / `collate.py` 已支持 `rd_features`
  - `RadarClassificationSystem` 现在会把 RD 摘要特征映射到一个轻量 RD 支路，并与轨迹融合特征拼接后送入专家
- 当前 RD 接入是第一版轻量实现：
  - 使用的是 `raw_rd` 的摘要特征（`frame_energy_mean/max/min`）
  - 不是完整 RD 图时序编码器
  - 其作用是先把 RD 真正接进训练闭环，而不是直接实现 `main.md` 里的最终精分支
- smoke 验证结果：
  - batch 中已存在 `rd_features`
  - 模型前向已实际消费 `rd_features`
  - `artifacts/system_v1_rd_smoke` 已可完成 1 epoch smoke 训练
## 2026-04-25 训练产物与报告按时间戳归档
- 新增：
  - `src/core/run_paths.py`
- baseline 与 system 训练链现在统一按时间戳创建输出目录：
  - `artifacts/<timestamp>/<experiment_name>/`
- 中文报告同步按时间戳写入：
  - `reports/real/<timestamp>/<experiment_name>.md`
  - `reports/real/<timestamp>/<experiment_name>.json`
- 这样每轮模型、checkpoint、history、summary、中文报告都不会互相覆盖，便于长期对照和回溯。
## 2026-04-25 按 `main.md` 继续补齐系统架构
- 在不改 `main.md` 的前提下，继续把当前 `src/core` 往目标系统推进，已补上三块关键结构：
  - 路由器加入物理规则兜底修正：
    - 高空高速样本可被强制路由到 `high_altitude_filter`
    - 低高度低速且垂直变化极小的样本可被强制路由到 `ground_target`
  - 低空专家从单一 MLP 升级为双流结构：
    - 轨迹常开分支
    - RD Transformer 轻分支
    - 基于置信度的按需激活与融合
  - 常规空中专家改为轻量 1D-CNN；地面专家改为复用深度轨迹特征+选定物理特征
- 这一步的目标不是宣称已经达到 `main.md` 完整形态，而是把当前系统从“有专家名义”推进到“专家内部结构也开始贴近设计目标”。
## 2026-04-25 专家一扩展到千万级预算方向
- 根据当前新的参数预算约束（1000万-2000万），已把主要容量集中投向专家一。
- 当前低空专家不再是轻量 RD 分支，而是：
  - 轨迹常开分类头
  - 幅度模态投影分支
  - 相位模态投影分支
  - 3层、512维的 RD Transformer
  - 偏度/峰度/相位统计特征分支
  - 双分支融合门
- 同时把低空专家内部类别对齐到当前项目主标签空间：
  - `DroneTarget / BirdTarget / BalloonTarget / ClutterTarget`
- 目标是让有限的大参数预算优先服务于系统主战场：低空可疑目标鉴别器。
## 2026-04-25 专家一 reject 从临时折叠改为正式 Unknown 闭环
- 当前全局标签空间已重新纳入 `UnknownTarget`。
- 低空专家内部类别现为：
  - `DroneTarget / BirdTarget / BalloonTarget / ClutterTarget / UnknownTarget`
- CosFace 低置信 reject 不再临时折叠到 `ClutterTarget`，而是正式投到 `UnknownTarget`。
- 数据处理侧同步放开：
  - `UnknownTarget` 不再被 `build_training_index.py` 默认排除
- 这样专家一的“识别 + 拒识”才形成真正闭环，而不是临时折中实现。
## 2026-04-25 专家一单独训练链打通
- 为避免继续被其他专家和全局路由耦合干扰，新增了专家一单独训练入口：
  - `src/core/expert1_config.py`
  - `src/core/expert1_data.py`
  - `src/core/expert1_trainer.py`
  - `src/core/train_expert1.py`
- 当前专家一单独训练只使用低空五类：
  - `DroneTarget / BirdTarget / BalloonTarget / ClutterTarget / UnknownTarget`
- 并复用：
  - 多模态幅度/相位 RD
  - CosFace 头
  - 双头辅助监督
  - 互学习
  - 监督对比学习
- 1 epoch CUDA smoke 已通过：
  - `artifacts/20260425-205946/expert1_v1_smoke3`
  - `reports/real/20260425-205946/expert1_v1_smoke3.md`
- 这一步的目标是先把专家一完全跑通、稳定下来，再回头继续推进其他专家层。
## 2026-04-25 其余专家层结构增强
- 在不打断 expert1 主线的前提下，已把其余专家层从“轻量占位头”推进到更接近 `main.md` 的可训练结构：
  - 常规空中专家：
    - 时序卷积
    - 注意力池化
    - 物理分支投影
    - 融合后分类
  - 地面专家：
    - 深轨迹特征分支
    - 选定物理特征分支
    - 门控融合后分类
- 这样后续在 expert1 稳定后，系统级训练不需要再从非常弱的其他专家头重新起步。
## 2026-04-25 expert1 默认训练策略继续向稳定收敛调整
- expert1 默认采样与两阶段配置已继续前推：
  - `rd_sample_boost: 4.0 -> 6.0`
  - `raw_source_boost: 2.0 -> 3.0`
  - `stage1_traj_only_epochs: 5 -> 8`
- 目标是：
  - 进一步提高真实 RD 样本在训练中的出现频率
  - 延长轨迹边界预热阶段
  - 在专家一尚未完全收敛前，避免过早让 RD 互学习/对比学习扰动主干
## 2026-04-26 expert1 继续延后 RD 介入
- 根据正式 `expert1_v1` 报告：
  - `Drone/Bird` 轨迹边界已经开始形成
  - 但 `raw_tr_rd / with_rd` 仍然几乎没学起来
- 因此把：
  - `stage1_traj_only_epochs: 8 -> 12`
- 目标是让轨迹主干先更充分收敛，再在后半程引入 RD 分支、互学习和对比学习。
## 2026-04-26 expert1 将 UnknownTarget 调整为弱监督类
- 根据正式 `expert1_v1` 报告：
  - `UnknownTarget` 仍未学起
  - 但会持续参与低空主边界竞争
- 当前调整为：
  - `unknown_loss_weight = 0.35`
  - 主分类损失与辅助头损失中，对 `UnknownTarget` 样本降权
  - 监督对比学习中暂时排除 `UnknownTarget`
- 目标是先稳定 `Drone/Bird/Balloon/Clutter` 主边界，再让 `UnknownTarget` 作为弱监督类存在。
## 2026-04-25 专家二与专家四单独训练闭环打通
- 新增：
  - `src/core/expert_other_config.py`
  - `src/core/expert_other_data.py`
  - `src/core/expert_other_trainer.py`
  - `src/core/train_expert2.py`
  - `src/core/train_expert4.py`
- 当前专家二可单独训练 `AircraftTarget` 子任务。
- 当前专家四可单独训练 `PeopleTarget / CarTarget / ShipTarget` 子任务。
- 1 epoch CUDA smoke 已通过：
  - `artifacts/20260425-235908/expert2_air_smoke`
  - `artifacts/20260425-235908/expert4_ground_smoke`
- 这样“其余专家层”已经从结构存在推进到“可单独训练、可出报告、可继续优化”的状态。
## 2026-04-26 路由器单独训练闭环打通
- 新增：
  - `src/core/router_train_config.py`
  - `src/core/router_trainer.py`
  - `src/core/train_router.py`
- 当前路由器采用：
  - 15维物理特征输入
  - `low_altitude_suspect / conventional_air / ground_target` 三路监督
  - 高空路径仍由规则兜底，不纳入当前监督闭环
- 1 epoch CUDA smoke 已通过：
  - `artifacts/20260426-003112/router_v1_smoke`
  - `reports/real/20260426-003112/router_v1_smoke.md`
- 这样 `main.md` 中“深度门控路由网络”的独立训练链也已经从无到有。
## 2026-04-26 主线训练代码接入预训练与阶段调度
- `system` 训练现在支持：
  - `pretrained_track_checkpoint`
  - `freeze_track_encoder_epochs`
  - `stage_warmup_epochs`
  - `stage_joint_epochs`
- 当前已经实现：
  - 轨迹预训练权重加载到 `representation_builder.sequence_encoder`
  - 前若干轮冻结 encoder
  - warmup / joint / finetune 三阶段骨架
- smoke 已验证：
  - 预训练权重可完整加载（`missing_keys=[]`, `unexpected_keys=[]`）
  - 三阶段参数能正常进入 `training_summary.json`
- 这样主线训练代码已经开始从“单阶段 system 训练”推进到更接近 `main.md` 的训练编排。
## 2026-04-26 主线训练代码补充三阶段 loss 调度
- `system` 训练进一步补上了显式阶段调度：
  - `warmup`
  - `joint`
  - `finetune`
- 当前实现：
  - `warmup` 阶段关闭低空额外联合损失，只保留 coarse 分类和 route
  - `joint` 阶段打开低空联合损失
  - `finetune` 阶段对 route / low-altitude aux / mutual / contrastive 使用可单独配置的缩放系数
- 对应新配置：
  - `stage_finetune_route_loss_scale`
  - `stage_finetune_low_altitude_aux_scale`
  - `stage_finetune_low_altitude_mutual_scale`
  - `stage_finetune_low_altitude_contrastive_scale`
- smoke 已通过，主线训练器现在已经不再是“所有 epoch 一把梭”的单阶段结构。
## 2026-04-26 主线训练代码接入系统级一致性损失
- 已按 `main.md` 接入第一版 `L_consistency`：
  - 仅对低空且有真实 RD 的样本生效
  - 当前实现为轨迹分支特征与 RD 分支特征之间的 MSE 对齐
- 新配置：
  - `consistency_loss_weight`
- smoke 已通过：
  - `artifacts/20260426-004941/system_v1_consistency_smoke`
- 这样主线训练代码已经具备：
  - 预训练接入
  - 冻结 encoder
  - warmup/joint/finetune 三阶段骨架
  - 系统级一致性损失接口
## 2026-04-26 主线训练代码补 route-only warmup
- 新增：
  - `freeze_non_router_in_warmup`
- 当前 warmup 阶段支持：
  - 仅训练 router
  - 其他 system 参数冻结
- 这一步的目标是把 route warmup 从“trainer 内部弱化部分损失”推进到“真正的 route-only 预热阶段”。
- 1 epoch smoke 已通过，虽然分类指标会塌，但这是预期行为，因为该阶段本就不以分类性能为目标。
## 2026-04-26 主线训练代码补 expert warmup / curriculum
- 新增：
  - `stage_expert_warmup_epochs`
  - `freeze_router_in_expert_warmup`
  - `freeze_track_encoder_in_expert_warmup`
  - `stage_expert_warmup_route_loss_scale`
  - `stage_expert_warmup_low_altitude_aux_scale`
  - `stage_expert_warmup_low_altitude_mutual_scale`
  - `stage_expert_warmup_low_altitude_contrastive_scale`
  - `stage_expert_warmup_consistency_scale`
- 当前 `system` 训练已具备四段编排骨架：
  - `route warmup`
  - `expert warmup`
  - `joint`
  - `finetune`
- 1 epoch `expert_warmup` smoke 已通过，说明主线训练器已经不再只是简单阶段标签，而是能按阶段冻结不同模块并切换损失权重。
## 2026-04-26 主线 pipeline orchestrator 打通
- 新增：
  - `src/core/pipeline_config.py`
  - `src/core/pipeline_trainer.py`
  - `src/core/train_main_pipeline.py`
- 当前已能按统一入口顺序执行：
  - `pretrain_track`
  - `router_v1`
  - `system_v1`
- 并自动把：
  - 预训练轨迹 encoder 权重
  - route warmup / expert warmup / joint / finetune 阶段参数
  串到 system 训练里。
- 1-1-1 的 smoke pipeline 已通过：
  - `artifacts/main_pipeline_smoke/...`
- 这样主线已经不再只是“有若干独立命令”，而是具备了一个可统一触发的阶段式训练编排入口。
## 2026-04-26 主线 pipeline 串上 expert2 / expert4
- `pipeline` 当前已扩展为：
  - `pretrain_track`
  - `router_v1`
  - `expert2_air`
  - `expert4_ground`
  - `system_v1`
- smoke 已验证完整链路可跑：
  - `artifacts/main_pipeline_smoke2/...`
- 当前仍未做的是：
  - expert2 / expert4 checkpoint 自动回灌 `system`
  - 但主线训练编排已经能把各阶段与各专家训练产物统一归档和汇总。
## 2026-04-26 src/core 按职责分层，降低耦合
- 新增子包：
  - `src/core/data`
  - `src/core/modeling`
  - `src/core/training`
  - `src/core/evaluation`
  - `src/core/utils`
- 目的：
  - 把原来堆平在 `src/core` 下的数据层、模型层、训练层、评估层和工具层拆开
  - 保留现有入口文件不动，先通过“子包导出 + 兼容壳”方式降低耦合
- 这样后续继续拆分内部模块时，不需要一次性打断现有训练命令。
## 2026-04-26 src/core 第二阶段解耦：数据访问链真实迁移
- 已完成：
  - `labels`
  - `processed_dataset`
  - `samples`
  - `track_dataset`
  - `split`
  - `rd`
  - `torch_dataset`
  - `collate`
  - `dataloaders`
- 这些模块的真实实现已迁入：
  - `src/core/data/`
- 原 `src/core/*.py` 同名文件当前只保留薄兼容壳，负责 re-export。
- 这样后续继续拆 `modeling/` 和 `training/` 时，风险会更低，也不会打断现有入口。
## 2026-04-26 主线补全：expert1 回灌 system，预训练回流专家，warmup 强约束
- `expert1` 独立训练已改成复用：
  - `TrackRepresentationBuilder`
  - `rd_proj`
  - `LowAltitudeSuspectExpert`
- 这样 `expert1` 的 checkpoint 现在包含：
  - `representation_builder_state_dict`
  - `rd_proj_state_dict`
  - `model_state_dict`
- `system` 已新增 `expert1_checkpoint` 装配能力，可直接回灌低空专家。
- `expert2` / `expert4` 训练入口已支持加载预训练轨迹 encoder，并可冻结前若干轮 encoder。
- `main pipeline` 已扩展为：
  - `pretrain_track`
  - `router_v1`
  - `expert1_v1`
  - `expert2_air`
  - `expert4_ground`
  - `system_v1`
- `system` 的 `expert warmup` 现在支持更强约束：
  - 只训 low-altitude expert 相关模块
  - 其余模块保持冻结
## 2026-04-26 CQ-08 接入前置审查与统一 manifest
- 已对 `OriData/23/CQ-08` 做可用性审查：
  - `航迹`、`点迹`、`原始回波` 三类文件可按 `(track_id, label_id)` 一一对应
  - `MAT` 可按 `track_id` 聚合成 RD 帧序列
  - `1550` 个航迹组中，当前 `1549` 个可用，`1` 个缺失 RD MAT（`track_id=1008`）
- 新增：
  - `src/process/cq08_manifest.py`
- 输出：
  - `data/manifests/cq08_unified_manifest.json`
  - `data/manifests/cq08_availability_summary.json`
  - `data/manifests/cq08_split_policy.json`
- 当前固定切分规则：
  - 按 `track_id` 分组切分
  - 同一 `track_id` 下的：
    - 航迹
    - 点迹
    - 原始回波
    - 全部 RD MAT 帧
    必须留在同一个 split
  - 禁止 RD 帧级随机切分
- 已新增：
  - `src/process/cq08_split.py`
- 当前 `CQ-08` grouped split 已独立落地到：
  - `data/manifests/cq08_grouped_split.json`
## 2026-04-26 CQ-08 标准化导出落地
- 已新增：
  - `src/process/standardize_cq08.py`
- 当前会独立导出：
  - `data/processed/cq08_tracks/`
  - `data/processed/cq08_rd/`
  - `data/processed/cq08_track_index.csv`
  - `data/processed/cq08_rd_index.csv`
- 当前策略：
  - 先把 `CQ-08` 做成独立可审查的 processed 数据源
  - 暂不直接并入现有 `raw + excel` 训练主线
  - 继续保持按 `track_id` 分组，禁止 RD 帧级随机切分
## 2026-04-26 expert1 切换到五类单独任务索引
- 已新增：
  - `src/process/build_expert1_index.py`
- 当前 expert1 专用索引会合并：
  - `raw_tr_rd` 中的
    - `DroneTarget`
    - `BirdTarget`
    - `BalloonTarget`
    - `ClutterTarget`
    - `UnknownTarget`
  - `CQ-08` 中同五类样本
- 输出：
  - `data/processed/expert1_track_index.csv`
  - `data/processed/expert1_summary.json`
- `src/core/expert1_data.py` 已切到读取该专用索引，不再依赖通用 `trainable_track_index.csv`
## 2026-04-26 expert1 五类任务采样再平衡
- 当前发现：
  - `CQ-08 + with_rd` 已经成为 expert1 主体
  - 但 `excel_track` 子集准确率仍明显落后
- 因此把 expert1 sampler 从“强拉高 with_rd”改成“来源更平衡”的版本：
  - `rd_sample_boost = 1.2`
  - `raw_source_boost = 2.0`
  - `excel_source_boost = 2.0`
  - `cq08_source_boost = 1.0`
- 目标：
  - 保留 `CQ-08 / RD` 的主导收益
  - 但不再让 `excel_track` 被过度边缘化
## 2026-04-26 expert1 stage2 渐进减弱
- 训练曲线显示：
  - `stage1` 到 `epoch 12` 左右时验证表现最好
  - `stage2` 一次性打开后，`val_f1` 会明显塌掉
- 当前改成：
  - `stage2_ramp_epochs = 6`
- 也就是：
  - `stage2` 相关辅助损失不再一步从 `0 -> 1`
  - 而是在进入 stage2 后按线性 ramp 缓慢放大
- 当前仍保持：
  - 主分类损失全程开启
  - 只对 `aux / mutual / contrastive` 做渐进减弱与缓启动
## 2026-04-26 修复审查问题：评估泄漏、CQ-08 标签/点迹/RD、system 索引
- 已修：
  - CosFace 评估阶段不再传入真实标签
  - `CQ-08 label 6` 从 `OtherTarget` 改为 `UnknownTarget`
  - `CQ-08` 标准化轨迹已接入点迹：
    - 多普勒速度
    - 和幅度
    - 信噪比
  - system low-alt 辅助损失已按 `rd_is_multimodal` 过滤
- 已接入：
  - `CQ-08 grouped split` 到 `expert1` 训练
  - `CQ-08 grouped split` 到 `system` 数据入口
- 已新增：
  - `src/process/build_system_index.py`
  - `data/processed/system_trainable_track_index.csv`
- 当前仍在继续优化：
  - `CQ-08` 3D RD token 规模控制
## 2026-04-26 架构审查问题修复（处理链/索引统一/CQ-08 RD/Expert2）

- `build_all_processed.py` 已接入 `build_system_index()`，现在一键重建会生成：
  - `data/processed/system_trainable_track_index.csv`
  - `data/processed/system_trainable_summary.json`
  - `pipeline_summary.json` 里也会写入 `system`
- 预训练、路由器、expert2/4 已统一切到 `system_trainable_track_index.csv`
  - 不再继续吃旧 `trainable_track_index.csv`
- `CQ-08` RD 编码不再直接平均掉 Doppler 维
  - 当前保持训练接口稳定的前提下，改成保留两套投影：
    - `range(amplitude/phase)`
    - `doppler(amplitude/phase)`
  - 每帧 RD 现编码为 `[T, 512, 4]`
- `CQ-08` 缺失高度/缺失相位已显式进入数据链
  - `height_missing`
  - `rd_phase_missing`
  - 物理特征追加：
    - `height_missing_flag`
    - `phase_missing_flag`
- `ConventionalAirExpert` 已从“单类伪训练”改成显式二分类语义：
  - `NonAircraftTarget`
  - `AircraftTarget`
- `ConventionalAirExpert` 的时间注意力池化已支持 `padding_mask`
  - 避免 padding token 参与池化

## 2026-04-27 system GPU 稳定性复核与 router 输入修正

- 根据最新 system 稳定性修补继续复核：
  - `source_normalization`
  - sequence encoder 低学习率参数组
  - joint/finetune loss ramp
  - sequence/global grad clip
  - loss component / logit / grad norm 记录
- 发现并修正一处路由输入语义问题：
  - `DeepGatingRouter` 已支持 `rule_features`
  - 但 `system.py` 仍把 raw physical 同时作为网络输入和规则输入
  - 已改成：
    - router network 输入 `representation.physical_features`
    - rule override 输入 `representation.raw_physical_features`
- 验证结果：
  - `python -m compileall src` 通过
  - `system_router_fix_smoke` 1 epoch CUDA 通过
  - `system_router_fix_3epoch` 3 epoch CUDA 通过
  - 两轮均 `skipped_non_finite_batches = 0`
  - route logits 回到小量级，3 epoch 测试 `route_logit_min/max ≈ -1.12 / 2.20`
- 当前状态：
  - system GPU 首轮 non-finite / all epochs unstable 已经压住
  - system 分类性能仍弱，主要偏向 `UnknownTarget`
  - route 仍几乎把所有 conventional/ground 样本预测为 `low_altitude_suspect`
  - 下一步应转向 system 路由/专家全局协调，而不是继续查 GPU 数值崩溃

## 2026-04-27 system 准确率骤降审查：低空 reject 过度吸收

- 针对 `system_router_fix_3epoch` 准确率偏低做审查：
  - 该实验只是 3 epoch 稳定性 smoke，不能和长训练直接横向比较
  - 但混淆矩阵显示大量 `Drone/Bird/Balloon/Clutter` 被预测为 `UnknownTarget`
  - `CQ-08` 准确率只有 `0.0644`
- 定位到真实代码问题：
  - `LowAltitudeSuspectExpert` 的 reject 评估路径默认开启
  - reject 条件使用 `has_rd`，导致所有有 RD 的低余弦样本都可能被强制改成 `UnknownTarget`
  - 全量 `CQ-08` 大部分都有 RD，因此被过度吸收到 `UnknownTarget`
- 已修：
  - `SystemTrainConfig` 新增：
    - `enable_low_altitude_reject_in_train = False`
    - `enable_low_altitude_reject_in_eval = False`
  - `train_system.py` 新增显式开关：
    - `--low-altitude-reject`
    - `--low-altitude-reject-in-train`
  - `LowAltitudeSuspectExpert` 的 reject 现在只作用于 `rd_activated` 样本，不再对所有 `has_rd` 样本生效
- 对照验证：
  - `system_router_fix_3epoch`：
    - `accuracy = 0.2229`
    - `macro_f1 = 0.1913`
    - `cq08_track accuracy = 0.0644`
  - `system_no_reject_3epoch`：
    - `accuracy = 0.5030`
    - `macro_f1 = 0.2330`
    - `cq08_track accuracy = 0.4506`
- 结论：
  - 准确率大幅降低的主因不是 GPU 稳定性修复，也不是 CQ-08 数据链重新断掉
  - 主因是 system 评估阶段未校准 reject 过度把有 RD 的低空样本改成 `UnknownTarget`
  - 后续 reject 必须作为单独校准/阈值实验，不应默认参与 system 主评估
## 2026-04-27 架构/代码审查：system 准确率下降原因定位

- 现象：早期只跑专家一时，低空任务可到 80% 左右；system 架构完善后短训一度掉到 20%~50%。
- 已确认不是“模型容量不足”导致。修复接线后，同样 3 epoch、无预训练、无专家 checkpoint 的 system smoke 达到：
  - `artifacts/20260427-185539/system_arch_review_raw_router_3epoch`
  - accuracy `0.7831`
  - macro F1 `0.5558`
  - CQ08 source accuracy `0.8627`
  - Drone F1 `0.8627`, Balloon F1 `0.9455`, Bird F1 `0.6933`, Unknown F1 `0.5161`
- 根因 1：低空 reject 在评估阶段过度吸收 Unknown。修复前大量 Drone/Bird/Balloon 被硬改成 Unknown，指标不代表真实分类能力。现在 system/expert1 默认评估关闭硬 reject，后续需要单独做阈值校准再打开。
- 根因 2：system 全局 logits 投影没有路由门控，所有专家 logits 对所有样本自由竞争；同时专家二 Aircraft 使用了错误列。已改为按 route weight 给对应专家 logits 加 log gate，并将 Aircraft 映射到 ConventionalAirExpert 的正类列。
- 根因 3：route loss 与 low-altitude aux loss 共用同一个 criterion。加入 route class weight 后会维度冲突，也说明 loss 耦合不清。已拆分 `class_criterion` / `route_criterion` / `low_criterion`。
- 根因 4：router standalone 训练曾按 `output.network_logits[:, :3]` 取列，第三列实际是 `high_altitude_filter`，不是 `ground_target`。已改为按 `ROUTE_NAMES.index()` 精确索引。
- 根因 5：轨迹预训练 checkpoint 原来保存整模型 key，system/expert loader 期望 `encoder.*` key，导致预训练回灌实际可能为空。已改为同时保存 `model_state_dict=encoder.*` 和 `pretrain_model_state_dict`。
- 根因 6：router standalone 与 system router 输入维度一度不一致。已按 main.md 回到 17 维显式物理特征输入，Router 内部 LayerNorm 控制尺度，规则兜底继续使用 raw physical。验证：
  - `artifacts/20260427-190301/system_router_load_check`
  - `pretrained_router_info.loaded_keys=8`
  - `missing_keys=[]`
  - `skipped_shape_mismatch={}`
- 仍未解决的问题：
  - route 总 accuracy 仍虚高，因为 low-altitude 占绝大多数；conventional_air 只有 33 条，test 仅 5 条，当前召回仍为 0。必须用 macro route recall 和 per-route recall 作为主指标。
  - PeopleTarget 总量只有 2 条，不能期待有效训练。
  - full pipeline 1 epoch smoke 可以跑通，但 system 1 epoch 低，是预期现象；需要分别把 expert1/router/expert2/expert4 训练到可用，再回灌 system。
  - `git diff --check` 仍有既有空白问题，主要在 README 和兼容 shim 文件，不影响训练。

## 2026-04-29 Expert2 Open-Set 改造与 System 接入复盘

### 问题 1：Expert2 二分类几乎全判 NonAircraft

- 现象：
  - Expert2 原本做 `[NonAircraftTarget, AircraftTarget]` 二分类。
  - 测试时容易退化成几乎全判 `NonAircraftTarget`。
  - macro F1 接近 `0.50`，Aircraft 正类召回趋近于 `0`。
- 数据证据：
  - `system_trainable_track_index.csv` 中 `AircraftTarget` 只有 `33` 条。
  - 负类由 `DroneTarget / BirdTarget / BalloonTarget / ClutterTarget / UnknownTarget / PeopleTarget / CarTarget / ShipTarget` 合并而来。
  - 负类不是一个语义紧凑类别，而是一个高方差“其它类集合”。
- 根因：
  - 原始 `nn.CrossEntropyLoss()` 没有类别权重，也没有均衡采样。
  - 在极端不均衡下，全判负类能得到很高 accuracy，但正类召回失效。
  - 更深层的问题是：`NonAircraftTarget` 本身不是可学习的紧凑类，强行二分类会让模型学习一个边界极不稳定的垃圾桶负类。

### 解决方案：将 Expert2 改为 Aircraft Open-Set 原型识别

- 落地思路：
  - 不再让 Expert2 学“什么是 NonAircraft”。
  - 只学习一个 `AircraftTarget` 原型向量。
  - 样本特征与 Aircraft 原型计算 cosine similarity。
  - similarity 高于阈值则输出 `AircraftTarget`，否则输出 `NonAircraftTarget`。
- 代码落地：
  - `src/core/experts.py`
    - `ConventionalAirExpert` 新增 `aircraft_prototype`。
    - 新增 `decision_threshold` buffer，保证阈值能随 checkpoint 保存/加载。
    - 默认 `open_set=True`。
    - 仍输出二分类 logits，保持 System 侧 `[NonAircraftTarget, AircraftTarget]` 接口兼容。
  - `src/core/expert_other_trainer.py`
    - 新增 `FocalCrossEntropyLoss`。
    - 新增 validation threshold calibration。
    - 把最佳阈值写入 checkpoint。
    - 训练/报告记录：
      - `aircraft_recall`
      - `non_aircraft_recall`
      - `balanced_accuracy`
      - `prediction_distribution`
      - `decision_threshold`
  - `src/core/expert_other_data.py`
    - Expert2 训练加入 `WeightedRandomSampler`。
    - 采用短 epoch 均衡采样，不再每轮全量扫过所有 NonAircraft。
  - `src/core/expert_other_config.py`
    - 新增 Expert2 配置：
      - `balanced_sampling`
      - `balanced_positive_multiplier`
      - `focal_loss_gamma`
      - `positive_class_weight`
      - `aircraft_positive_noise_std`
      - `open_set_aircraft`
      - `open_set_logit_scale`
      - `threshold_min/max/steps`
  - `config/config.json`
    - Expert2 训练轮数改为 `8`。
    - System 的 `expert2_checkpoint` 更新到新 Open-Set checkpoint。

### 问题 2：Expert2 训练一度看起来“卡住”

- 现象：
  - 第一次 Expert2 CUDA 训练被中断。
  - 后台仍残留 python 训练进程。
- 定位：
  - 不是 CPU 训练，也不是 deadlock。
  - 主要原因是均衡采样策略初版仍按大规模负类 epoch 运行，33 条 Aircraft 被大量 NonAircraft 拖慢。
- 解决：
  - 停止残留训练进程。
  - 将 Expert2 sampler 改成短 epoch：
    - `num_samples = 2 * positive_count * balanced_positive_multiplier`
    - 当前 `balanced_positive_multiplier = 8`
  - 保留正负均衡训练信号，同时减少无意义负类重复。
- 结果：
  - 后续 Expert2 CUDA 训练正常完成。
  - 训练时间约数分钟量级，没有再出现长时间无反馈。

### Expert2 最终训练结果

- 训练命令：
  - `.\.venv\Scripts\python.exe -m src.core.train_expert2 --config config/config.json`
- 运行目录：
  - `artifacts/20260428-201240/expert2_config_smoke`
- 设备：
  - `cuda`
  - `NVIDIA GeForce RTX 4060 Laptop GPU`
- 测试指标：
  - accuracy：`0.9697`
  - macro F1：`0.7144`
  - balanced accuracy：`0.8862`
  - Aircraft precision：`0.3077`
  - Aircraft recall：`0.8000`
  - Aircraft F1：`0.4444`
  - NonAircraft recall：`0.9723`
  - decision threshold：`0.63`
- 混淆矩阵：
  - `NonAircraftTarget`: `316` 判对，`9` 判成 Aircraft
  - `AircraftTarget`: `4` 判对，`1` 判成 NonAircraft
- 结论：
  - Expert2 “全判 NonAircraft”问题已被修复。
  - 由于 Aircraft 测试集只有 `5` 条，单次 recall 波动会很大，后续需要更多 Aircraft 样本或 grouped repeated split 来确认稳定性。

### 问题 3：System 接入后 Aircraft 最终分类仍被跨专家竞争压制

- 现象：
  - 新 Expert2 单独测试 Aircraft recall 已到 `0.80`。
  - 但 System 最终全局分类里，Aircraft 仍然会被低空专家/其它专家 logits 抢走。
- 验证运行：
  - `artifacts/20260428-232117/system_config_smoke`
- System 测试指标：
  - accuracy：`0.8434`
  - macro F1：`0.5651`
  - route accuracy：`0.7319`
  - route macro recall：`0.7460`
  - low-altitude macro F1：`0.8265`
  - low-altitude accuracy：`0.9032`
  - Drone/Bird recall mean：`0.9044`
  - Unknown recall：`0.6111`
  - Clutter recall：`0.7143`
  - ground route recall：`0.7059`
- System 接入验证：
  - 新 Expert2 checkpoint 已完整加载。
  - `expert2_info.loaded_keys = 30`
  - `missing_keys = []`
  - `unexpected_keys = []`
- 根因判断：
  - Expert2 的基础识别能力已经恢复。
  - 当前瓶颈转移到 System 全局融合层：
    - 各专家 logits 标尺不统一。
    - Expert1 CosFace logits、Expert2 Open-Set similarity logits、Expert4 MLP logits 直接进入全局竞争。
    - Router 仍不是稳定强约束，不能直接 hard gate。
  - 因此 Aircraft 在 Expert2 中可被识别，但在 System 最终 argmax 中仍可能被低空类别覆盖。

### 本轮最终落地

- 已提交 git：
  - `73a020d Improve expert2 open-set aircraft training`
- 提交中记录了模型性能：
  - Expert2：
    - accuracy `0.9697`
    - macro F1 `0.7144`
    - balanced accuracy `0.8862`
    - Aircraft recall `0.8000`
  - System：
    - accuracy `0.8434`
    - macro F1 `0.5651`
    - route macro recall `0.7460`
    - low-altitude macro F1 `0.8265`
    - Drone/Bird recall mean `0.9044`
- 未提交内容：
  - 其它手动重构留下的 dirty 文件未纳入本次 commit。
  - 只提交了本轮 Expert2/Open-Set、Expert1 指标补充、config 接入相关文件。

### 后续方向

- 最高优先级：
  - 做 System 级专家置信度标定与回退融合。
  - 目标是让 Expert2 的 Open-Set Aircraft 置信度真正影响最终决策，而不是只在单专家报告中好看。
- 可选方案：
  - 为各专家 logits 增加 per-expert temperature calibration。
  - 对 Expert2 Aircraft 使用独立 open-set 置信度覆盖规则：
    - 当 Aircraft similarity 高于高置信阈值时，优先保护 Aircraft 输出。
  - 加入 cross-expert fallback：
    - 当主路由专家低置信时，允许相邻专家二次判断。
  - 在训练报告中增加：
    - System 全局 Aircraft recall
    - Expert2-in-System recall
    - Expert2 high-confidence override 触发率
    - override 后的 false positive rate
- 注意：
  - 当前不要恢复 hard route 作为默认训练路径。
  - Router 未稳定前，hard route 会把错误路由放大成不可恢复的分类错误。

## 2026-04-29 System 级跨专家 Aircraft 仲裁修复

### 问题：Expert2 在 System 里仍被跨专家 logits 压制

- 现象：
  - Expert2 单独训练后已经不再全判 `NonAircraftTarget`，单专家测试中 `AircraftTarget recall = 0.8000`。
  - 但进入 System 后，最终全局 9 类输出仍会把真实 Aircraft 判成 `DroneTarget`、`BirdTarget` 等低空类别。
  - 根因不是 Expert2 完全不可用，而是 System 最终投影仍偏向 raw logits 拼接。
- 旧接线问题：
  - `RadarSystemOutput` 只暴露 `conventional_air_logits`，没有把 Expert2 的 `aircraft_similarity` / `aircraft_threshold` 透传给仲裁层。
  - `_project_to_global_logits()` 对 Aircraft 的处理近似为：
    - `conventional_air_logits[:, 1] + conventional_gate`
    - 再与 high-altitude 规则 logit 做 `max`
  - 这会把 Expert2 的 Open-Set 语义退化成普通二分类 raw logit。
  - Expert1 是 CosFace scale logits，Expert2 是 Open-Set margin logits，Expert4 是 MLP logits，三者数值标尺不一致，直接 argmax 竞争不可靠。
  - Router 在低空/常规空中边界上并不稳定，因此不应该作为 veto，只能作为 prior。

### 解决方案：把 System 从 raw-logits 拼接器改成带置信度的专家仲裁器

- `src/core/system.py`
  - `RadarSystemOutput` 新增：
    - `conventional_air_detail`
    - `high_altitude_detail`
    - `ground_detail`
  - 保留旧字段：
    - `conventional_air_logits`
    - `high_altitude_logits`
    - `ground_logits`
  - 这样旧训练/报告入口不被破坏，同时 System 仲裁层可以访问 Expert2 的 auxiliary 信号。
- `src/core/system_config.py`
  - 新增 Aircraft 仲裁配置：
    - `aircraft_protection_enabled = true`
    - `aircraft_confidence_tau_high = 0.10`
    - `aircraft_confidence_tau_low = -0.05`
    - `aircraft_conventional_gate_floor = 0.10`
    - `aircraft_temperature = 1.0`
- `src/core/system_trainer.py`
  - `_project_to_global_logits()` 改为接收完整 `SystemTrainConfig`。
  - 计算 Expert2 置信度：
    - `aircraft_margin = conventional_air_logits[:, 1] - conventional_air_logits[:, 0]`
    - `aircraft_confidence = aircraft_similarity - aircraft_threshold`
  - 三段式保护：
    - `aircraft_confidence >= tau_high`：Aircraft 分数基本绕过 Router 惩罚，Router 不再一票否决。
    - `tau_low <= aircraft_confidence < tau_high`：给 conventional gate 一个 floor，防止弱路由把 Expert2 直接压死。
    - `aircraft_confidence < tau_low`：仍按 Router prior 抑制，避免大量非 Aircraft 被抢成 Aircraft。
  - HighAltitude 规则只在规则正命中时参与 Aircraft，不再让未命中的 high-altitude negative logit 与 Expert2 无意义竞争。

### 报告和诊断落地

- `training_summary.json` / `evaluation_report.json` 新增 System 级 Aircraft 诊断：
  - `aircraft_system_recall`
  - `aircraft_system_precision`
  - `aircraft_system_f1`
  - `aircraft_recall_without_aircraft_protection`
  - `expert2_positive_system_not_aircraft_count`
  - `expert2_positive_system_not_aircraft_rate`
  - `non_aircraft_as_aircraft_rate`
  - `aircraft_overridden_by_low_altitude_count`
  - `aircraft_overridden_by_ground_count`
  - `true_aircraft_prediction_counts`
  - `true_aircraft_prediction_counts_without_protection`
- `predictions` 样本级记录新增：
  - `label_name_pred_without_aircraft_protection`
  - `aircraft_protection_changed_prediction`
  - `aircraft_similarity`
  - `aircraft_threshold`
  - `aircraft_margin`
  - `aircraft_confidence`
  - `conventional_route_weight`
  - `conventional_gate`
  - `adjusted_conventional_gate`
  - `expert2_positive`
  - `aircraft_high_confidence`
  - `aircraft_mid_confidence`
- `selection_score` 也纳入 Aircraft：
  - 正向：`aircraft_system_recall`、`aircraft_system_f1`
  - 惩罚：`non_aircraft_as_aircraft_rate`
  - 目的：避免只靠 overall macro F1 选中 Aircraft 已经崩坏的 checkpoint。

### 验证过程

- 编译检查：
  - `.\.venv\Scripts\python.exe -m compileall src config`
  - 通过。
- 格式检查：
  - `git diff --check`
  - 通过，仅有 Windows 换行提示。
- CUDA 单 batch 诊断：
  - model/checkpoint 全部加载成功。
  - Expert1 / Expert2 / Expert4 均无 shape mismatch。
  - `_project_to_global_logits(..., return_diagnostics=True)` 可输出完整 Aircraft 诊断字段。
- CUDA System 训练：
  - 命令：`.\.venv\Scripts\python.exe -X faulthandler -m src.core.train_system --config config/config.json`
  - 运行目录：`artifacts/20260429-021051/system_config_smoke`
  - 设备：`cuda`
  - GPU：`NVIDIA GeForce RTX 4060 Laptop GPU`
  - 训练过程无 non-finite batch，`skipped_non_finite_batches = 0`。

### 最终结果

- System 测试集：
  - accuracy：`0.8343`
  - macro precision：`0.6464`
  - macro recall：`0.6087`
  - macro F1：`0.5801`
  - route accuracy：`0.7319`
  - route macro recall：`0.7460`
- 低空核心：
  - expert1 low-altitude macro F1：`0.8279`
  - expert1 low-altitude accuracy：`0.9065`
  - Drone recall：`0.9470`
  - Bird recall：`0.8625`
  - Drone/Bird recall mean：`0.9048`
  - Unknown recall：`0.6111`
  - Clutter recall：`0.7143`
- Aircraft 仲裁：
  - 保护前 Aircraft recall：`0.2000`
  - 保护后 Aircraft recall：`0.8000`
  - Aircraft precision：`0.2353`
  - Aircraft F1：`0.3636`
  - Expert2 判正但 System 未判 Aircraft：`10`
  - Aircraft 被低空类别压制：
    - 保护前：`4`
    - 保护后：`1`
  - 保护机制恢复真实 Aircraft：`3`
  - 新增 false Aircraft：`10`
  - non-Aircraft as Aircraft rate：`0.0398`

### 判断

- 已修复：
  - System 现在能使用 Expert2 Open-Set 的 `similarity/threshold/margin`。
  - Aircraft 不再被 Router gate 轻易 veto。
  - 报告能明确比较保护前/保护后 Aircraft 表现。
  - 报告能定位 Aircraft 是被低空专家还是地面专家压制。
- 仍未完全解决：
  - Aircraft precision 偏低，当前保护机制用 recall 换来了 false positive。
  - 测试集 Aircraft 只有 `5` 条，单次 recall 波动很大。
  - Expert2 positive false Aircraft 仍有 `22` 条，说明 Expert2 Open-Set 阈值和 System 仲裁阈值还需要继续校准。
  - System accuracy 相比 20260428 的 `0.8434` 略降，但 macro F1 从 `0.5651` 提升到 `0.5801`，并且 Aircraft recall 从 System 保护前 `0.20` 拉回到 `0.80`。

### 下一步

- 优先继续做 System 级校准，而不是回退本次架构：
  - 增加 Expert2 高置信 override 的 precision 约束。
  - 对 Expert2 false positive 样本做按类别统计，区分是 Bird/Drone/Ship 还是 Unknown 被抢成 Aircraft。
  - 增加 per-expert temperature/bias calibration，避免 Expert1/Expert2/Expert4 raw logits 尺度继续互相污染。
  - 在更多 split 或 repeated grouped split 上验证 Aircraft recall，不要只依据 5 条测试样本下结论。

## 2026-05-01 Expert4 Groundness 拒识与 System 地面门控

### 问题：Expert4 之前只会在 People/Car/Ship 内硬分类

- 旧设计问题：
  - `GroundTargetExpert` 只有三分类 head。
  - 任何被误路由进 Expert4 的样本，都会被强制判成 `PeopleTarget / CarTarget / ShipTarget` 之一。
  - 这和 `target.md` 里的设计不一致：地面专家至少要先判断“是否属于地面目标族”，再做子类细分。
- 直接风险：
  - 低空样本、未知样本、甚至 Aircraft，只要地面 route 或全局专家竞争稍有偏差，就可能被地面专家硬拉走。
  - 这会污染 System 的鲁棒性，尤其是 route 本身不稳定时更明显。

### 落地方案

- `src/core/experts.py`
  - 给 `GroundTargetExpert` 增加 `groundness_head`。
  - 输出 auxiliary：
    - `groundness_logits`
    - `groundness_probability`
    - `groundness_threshold`
    - `groundness_margin`
  - 保留 `People/Car/Ship` 三分类 head，不破坏现有接口。
- `src/core/expert_other_data.py`
  - `expert4` 训练时允许引入 hard negatives：
    - `DroneTarget`
    - `BirdTarget`
    - `BalloonTarget`
    - `ClutterTarget`
    - `UnknownTarget`
    - `AircraftTarget`
- `src/core/expert_other_trainer.py`
  - 地面三分类 CE 只在真实地面样本上计算。
  - `groundness` 二分类 loss 在全部样本上计算。
  - 加入 groundness threshold calibration。
  - 输出：
    - `groundness_precision`
    - `groundness_recall`
    - `groundness_specificity`
    - `groundness_f1`
- `src/core/system_trainer.py`
  - 在 `_project_to_global_logits()` 中读取 `ground_detail.auxiliary`。
  - 用 `groundness_probability - groundness_threshold` 构造 `groundness_confidence`。
  - 地面专家的 route gate 改为三段式：
    - 高置信地面：弱化 route 惩罚
    - 中置信地面：给 route floor
    - 低置信地面：额外下压地面 logits，减少误抢占
  - 新增 System 级地面诊断：
    - `ground_group_precision`
    - `ground_group_recall`
    - `ground_group_f1`
    - `pred_ground_on_non_ground_count`
    - `groundness_probability_mean_true_ground`
    - `groundness_probability_mean_non_ground`

### Expert4 CUDA 训练结果

- 训练命令：
  - `.\.venv\Scripts\python.exe -m src.core.train_expert4 --config config/config.json`
- 运行目录：
  - `artifacts/20260501-211707/expert4_config_smoke`
- 测试指标：
  - accuracy：`0.8824`
  - macro F1：`0.6061`
  - groundness precision：`0.4667`
  - groundness recall：`0.4118`
  - groundness specificity：`0.9744`
  - groundness F1：`0.4375`
  - calibrated groundness threshold：`0.22`
- 解释：
  - 当前 groundness head 已能较强压制非地面误报，specificity 很高。
  - 但 recall 仍偏低，说明它对“真地面样本”还偏保守。

### System CUDA 训练结果

- 训练命令：
  - `.\.venv\Scripts\python.exe -X faulthandler -m src.core.train_system --config config/config.json`
- 运行目录：
  - `artifacts/20260501-212307/system_config_smoke`
- 全局指标：
  - accuracy：`0.8434`
  - macro F1：`0.5607`
  - route macro recall：`0.7460`
- 低空指标：
  - low-altitude macro F1：`0.8047`
  - Drone/Bird recall mean：`0.8893`
  - Unknown recall：`0.7778`
  - Clutter recall：`0.5714`
- 地面组指标：
  - ground group precision：`0.4286`
  - ground group recall：`0.3529`
  - ground group F1：`0.3871`
  - non-ground 被判为地面目标：`8`
- Aircraft 指标：
  - Aircraft recall：`0.6000`
  - Aircraft precision：`0.3000`
  - Aircraft F1：`0.4000`
  - protection 前 recall：`0.0000`
  - non-Aircraft as Aircraft rate：`0.0214`

### 当前判断

- 已经完成：
  - Expert4 不再是纯三分类硬头，已经具备 groundness 拒识能力。
  - System 已能使用地面专家的 groundness 置信度，而不是无条件让地面 logits 参与竞争。
  - 报告里已经能看到 ground group 指标和非地面误判计数。
- 当前问题：
  - 新 Expert4 让 Aircraft false positive 下降了，但同时把 Aircraft recall 从上轮 `0.8000` 压到了 `0.6000`。
  - groundness 概率在真地面/非地面之间均值差异过小：
    - true ground `0.2806`
    - non-ground `0.2708`
  - 说明 groundness head 虽然通过阈值把误报压下来了，但特征分离还不够强。

### 下一步

- 优先做两件事：
  - 重新校准 `groundness` 和 `aircraft` 两条保护线的相互作用，避免 Expert4 把 System 的 Aircraft 主线重新压低。
  - 针对 Expert4 增加更强的非地面 hard negatives 采样和更明确的 group-level selection metric，不只盯三分类 macro F1。

## 2026-05-01 S2 推理端回流重路由机制

### 目标

- 按 `target.md` 增加仅推理启用的回流重路由：
  - 训练阶段不启用，不引入不可微链路。
  - 推理阶段允许样本在“当前专家低置信”时惩罚当前 route，再尝试下一专家。
  - 多轮后仍无结论时回落到 `UnknownTarget`。

### 代码落地

- `src/core/system_config.py`
  - 新增 reroute 配置：
    - `reroute_max_passes = 3`
    - `reroute_penalty = 3.0`
    - `reroute_max_total_penalty = 6.0`
    - `reroute_low_altitude_conf_threshold = 0.35`
    - `reroute_conventional_air_conf_threshold = 0.35`
    - `reroute_ground_conf_threshold = 0.50`
  - `eval_fusion_candidate_modes` 新增 `reroute`。
- `src/core/system_trainer.py`
  - 新增 `_project_to_global_logits_reroute()`。
  - 回流置信度信号：
    - Expert1：`traj_cosine.max()`
    - Expert2：`aircraft_similarity`
    - Expert3：一旦选中直接接受
    - Expert4：`groundness_probability`
  - 规则覆盖样本：
    - `high_altitude_filter`
    - `ground_target`
    - 在 pass 0 直接完成，不进入回流循环。
  - 诊断字段：
    - `reroute_count`
    - `reroute_triggered`
    - `reroute_fallback`
    - `reroute_selected_route`
    - `reroute_total_penalty`
  - 报告指标：
    - `reroute_rate`
    - `reroute_avg_count`
    - `reroute_recovery_rate`
    - `reroute_fallback_rate`
    - `reroute_worsen_rate`
    - `reroute_count_distribution`
  - `predictions` 已增加 reroute 样本级记录，便于回查具体误判。

### 当前实现策略

- 先用现有 `soft_log_weighted` 仲裁得到 base global logits。
- 回流机制只决定“最终采用哪个专家族”：
  - 低空族：`Drone/Bird/Balloon/Clutter/Unknown`
  - 常规空中族：`Aircraft`
  - 地面族：`People/Car/Ship`
  - 高空族：`Aircraft`
- fallback 样本强制输出 `UnknownTarget`。

### CUDA 验证

- 独立 reroute probe：
  - 使用 `artifacts/20260501-212307/system_config_smoke/best_model.pt`
  - 在测试集直接跑 `fusion_mode='reroute'`
- 正式主线验证：
  - `.\.venv\Scripts\python.exe -X faulthandler -m src.core.train_system --config config/config.json`
  - 运行目录：`artifacts/20260501-222747/system_config_smoke`

### 验证结果

- 当前主线最佳模式仍是：
  - `soft_log_weighted`
- `fusion_validation_probe` 中 `reroute` 表现：
  - accuracy：`0.6391`
  - macro F1：`0.4290`
  - Aircraft recall：`0.6000`
  - Aircraft precision：`0.1200`
  - Aircraft F1：`0.2000`
  - ground group F1：`0.3158`
  - non-Aircraft as Aircraft rate：`0.0683`
- 说明：
  - `reroute` 已经实现成功并能稳定运行。
  - 但当前阈值/惩罚配置下，它明显弱于 `soft_log_weighted`，还不适合作为默认推理策略。

### 当前判断

- 已完成：
  - 推理端回流机制已落地。
  - 训练链路保持不变。
  - 报告链路已能直接比较 `reroute` 与原融合模式。
- 当前问题：
  - `reroute` 会显著伤害低空主任务。
  - 当前阈值配置过于激进，fallback 和跨专家串扰仍偏高。
  - 它适合继续做推理实验，但现在不应取代主线融合。

### 下一步

- 若继续推进 S2，应优先做：
  - 对 `reroute_*_conf_threshold` 做系统性 sweep。
  - 给 Expert1/Expert2/Expert4 增加更保守的“低置信定义”，减少无谓回流。
  - 将 reroute 约束在相邻专家子图，而不是开放所有专家自由跳转。

## 2026-05-01 全链路主线调试与 main 对齐复核

### 发现的问题

- `src/core/pipeline_trainer.py` 仍是旧编排版本：
  - 没有复用 `config/config.json` 中各阶段的真实配置。
  - 没有把最新的 `expert4 groundness`、`system reroute probe` 等设置带进主线。
  - checkpoint 回流链路虽然概念上存在，但参数来源与当前主线不一致。
- 结果是：
  - 单阶段训练可以跑通。
  - 但 `train_main_pipeline` 作为全链路入口，不能稳定代表当前真实架构。

### 修复内容

- `src/core/pipeline_config.py`
  - 改为真正的 orchestrator 配置：
    - 每个阶段单独开关
    - 每个阶段单独输出目录
    - 支持显式 checkpoint 恢复
- `src/core/pipeline_trainer.py`
  - 改为从 `config/config.json` 读取：
    - `pretrain_track`
    - `router`
    - `expert1`
    - `expert2`
    - `expert4`
    - `system`
  - 按顺序回流 checkpoint：
    - `pretrain -> expert1/system`
    - `router -> system`
    - `expert1 -> expert2/expert4/system`
    - `expert2/expert4 -> system`
  - 增加续跑逻辑：
    - 优先读取各阶段 `training_summary.json`
    - 若 summary 缺失，可直接用显式 `best_checkpoint` 恢复
- `src/core/train_main_pipeline.py`
  - 显式把 `--config` 路径传入 pipeline trainer。
- `config/config.json`
  - 新增 `pipeline` section，主线训练入口现在可直接由单一配置文件驱动。

### 全链路 smoke 结果

- 主线 smoke 实际顺序完成：
  - `pretrain_track`
  - `router_v1`
  - `expert1_v1`
  - `expert2_air`
  - `expert4_ground`
  - `system_v1`
- 产物根目录：
  - `artifacts/main_pipeline_config_smoke`
- 汇总文件：
  - `artifacts/main_pipeline_config_smoke/pipeline_summary.json`

### 各阶段结果

- `pretrain_track`
  - `artifacts/main_pipeline_config_smoke/20260501-225130/pretrain_track`
  - final loss：`0.1912`
- `router_v1`
  - `artifacts/main_pipeline_config_smoke/20260501-225235/router_v1`
  - accuracy：`0.7629`
  - macro F1：`0.7942`
- `expert1_v1`
  - `artifacts/main_pipeline_config_smoke/20260501-225426/expert1_v1`
  - final accuracy：`0.8258`
  - final macro F1：`0.7226`
  - RD on-RD macro F1：`0.5356`
- `expert2_air`
  - `artifacts/main_pipeline_config_smoke/20260501-231311/expert2_air`
  - accuracy：`0.9515`
  - macro F1：`0.6541`
  - Aircraft recall：`0.8000`
- `expert4_ground`
  - `artifacts/main_pipeline_config_smoke/20260501-231610/expert4_ground`
  - 本轮由于主命令超时，只保留了 `best_model.pt`
  - 后续已通过显式 checkpoint 恢复进 pipeline 汇总
- `system_v1`
  - `artifacts/main_pipeline_config_smoke/20260501-232217/system_v1`
  - accuracy：`0.8645`
  - macro F1：`0.5760`
  - selected fusion mode：`ungated`
  - Aircraft precision：`0.4286`
  - Aircraft recall：`0.6000`
  - Aircraft F1：`0.5000`
  - ground group F1：`0.3846`
  - low-altitude macro F1：`0.8019`
  - Drone/Bird recall mean：`0.9055`

### 与 main.md 的当前对齐结论

- 已对齐的部分：
  - 深度轨迹粗筛 + 物理特征路由 + 多专家结构
  - 专家一轨迹/RD 双流
  - 轨迹预训练回流
  - 路由器独立训练并回灌 system
  - 专家二 open-set 化
  - 专家四 groundness 拒识
  - 主线 pipeline 串联与 checkpoint 回流
- 仍未完全对齐的部分：
  - `system` 当前最佳融合仍是 `ungated`，尚未达到 `main.md` 中更理想的“稳定门控协同”
  - `reroute` 已实现但当前不是有效主线策略
  - 专家一 RD 分支仍未达到你要的强判别上限
  - Expert3 仍是规则路径，没有进一步的高空学习专家
  - 轨迹预训练仍是骨架级 self-supervised，而不是更强的跨域预训练体系

### 当前建议

- 下一主线应收敛到两件事：
  - 继续强化 `expert1` 的 RD 判别质量，尤其是 `Bird / Clutter / Unknown`
  - 继续压实 `system` 的融合策略，让 `ungated` 逐步让位给更稳定的门控协同，而不是继续扩展 `reroute`

## 2026-05-02 第一轮提准研究：先关掉 groundness_protection

### 研究动机

- 当前 `system_v1` 全链路最好结果是：
  - accuracy：`0.8645`
  - macro F1：`0.5760`
  - Aircraft F1：`0.5000`
- 但 `groundness_probability_mean_true_ground` 与 `groundness_probability_mean_non_ground` 非常接近：
  - true ground：`0.2585`
  - non-ground：`0.2714`
- 这说明当前 `groundness_head` 的分离度还不够，System 里继续启用 `groundness_protection` 很可能在“保护地面”之前，先错误抑制了其它专家的正常竞争。

### 快速校准实验

- 基于 `artifacts/main_pipeline_config_smoke/20260501-232217/system_v1/best_model.pt` 做纯推理评估，不重训：
  - 模式：`ungated`
  - 修改：`groundness_protection_enabled = false`
- 测试集结果：
  - accuracy：`0.8645`（保持不变）
  - macro F1：`0.6010`
  - Aircraft precision：`0.6000`
  - Aircraft recall：`0.6000`
  - Aircraft F1：`0.6000`
  - ground group F1：`0.4848`
  - false aircraft rate：`0.0061`

### 结论

- 这是当前最明确、最便宜的一步收益。
- 与开启 `groundness_protection` 相比：
  - macro F1：`0.5760 -> 0.6010`
  - Aircraft F1：`0.5000 -> 0.6000`
  - false aircraft rate：`0.0122 -> 0.0061`
- 因此当前主线配置先切换为：
  - `system.groundness_protection_enabled = false`

### 当前判断

- 这不是否定 Expert4/groundness 的方向，而是说明：
  - 当前 `groundness_head` 还没强到足以承担 System 级保护职责。
  - 在它真正学好之前，把它从全局裁决里退一步，更有利于整体精度。
- 下一步应该继续：
  - 提升 `Expert1` 的 `Bird / Clutter / Unknown` RD 判别
  - 再重新训练一版 `System`，验证这条配置修改是否能稳定把正式 best checkpoint 提到 `macro_f1 > 0.60`
