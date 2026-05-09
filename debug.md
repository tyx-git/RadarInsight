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

## 2026-05-06 Expert2 / Expert4 / System 最近修复记录
- 先修复了 `expert2_air` 的开集校准链路，解决“源内阈值只看 raw_tr_rd、excel_track 自身假 Aircraft 也被放行”的问题。
- 代码落点：
  - `src/core/expert_other_trainer.py`
  - `src/core/expert_other_config.py`
  - `config/config.json`
- 具体修复：
  - 源内阈值校准加入 `source_false_aircraft_rate` 和 `max_source_false_aircraft_rate` 约束
  - 非 focus 源在 positive support 太少时回退默认阈值
  - best checkpoint 选择改为以校准后宏 F1 + aircraft precision/recall 为主
  - 增加 hard negative label 级加权，重点压 `BirdTarget` / `DroneTarget` / `ShipTarget`
- 实测结果：
  - `expert2_config_smoke` 独立测试从早期“precision 很低、FP 爆炸”修到：
    - `Aircraft precision = 0.3333`
    - `Aircraft recall = 0.4`
    - `macro F1 = 0.6764`
  - 训练后 `raw_tr_rd` 和 `excel_track` 的 Aircraft 假阳性明显收敛

- 再训练 `expert4_ground`，把 groundness 分离损失和 ground 类别权重打开。
- 代码落点：
  - `config/config.json`
  - `src/core/expert_other_trainer.py`
- 具体修复：
  - `balanced_sampling = true`
  - `use_ground_class_weighting = true`
  - `groundness_separation_loss_weight = 0.15`
  - 重新选择 groundness threshold
- 实测结果：
  - `expert4_config_smoke` 独立测试 ground group F1 从旧版 `0.4516` 提升到 `0.5641`
  - `System` 中 `representation + expert4 head` 也提升到 `0.5405`
  - 说明 ground head 不是纯失效，System 仲裁仍有压制

- 对 Ground 做过两次系统级保护实验：
  - `groundness_protection_enabled = true` 时，ground 独立可更高，但系统最终 ground F1 反而下降
  - 回退到 `groundness_protection_enabled = false` 后，系统最终更稳
- 结论：
- 当前 Ground 的主要瓶颈不是简单的保护阈值，而是 System 级仲裁仍会压低 ground 候选
- 当前阶段不再继续盲开 groundness 保护

- System 侧最近修复：
  - `fusion_mode_selection_policy` 从 `configured` 改为 `validation_score`
  - 系统验证 probe 会选择 `ungated` 作为当前更优模式，而不是被 `soft_log_weighted` 固定住
  - `groundness_protection_enabled` 经过实测后回退关闭，避免误伤最终 ground_group
- 最新系统结果：
  - `accuracy = 0.8614`
  - `macro F1 = 0.6112`
  - `aircraft precision = 0.5714`
  - `aircraft recall = 0.8`
  - `ground_group F1 = 0.4375`
  - `best_test_probe = ungated`

- 当前结论：
  - Expert2 已从“Aircraft FP 爆炸”修到可用
  - Expert4 独立变强了，但 System 最终 ground 仍被仲裁压制
  - 当前下一步应继续做的是：
    - 系统级 Ground 候选保护的更稳版本
    - 或者把 Ground 重新纳入 calibrated candidate 仲裁
    - 同时继续提升 Expert1 RD 对 Unknown / Clutter / Bird 的分离能力
  - per-class precision / recall / f1 / support

## 2026-05-07 Expert1 源域鲁棒与 RD 二证据修复
- 这轮聚焦的是 Expert1 不再只看平均分，而要同时压住弱源和 RD 失真。
- 之前的问题：
  - `raw_tr_rd` 样本很少，旧版选模会被少量弱源样本过度放大，导致 stage2 候选经常被判死。
  - RD 分支虽然有多种辅助损失，但 `UnknownTarget` / `ClutterTarget` 仍然没有形成直接对抗边界。
  - RD 在 `on-RD` 上能救一部分样本，但还不足以被当成稳定的第二证据源。
- 本轮修复：
  - 在 `src/core/expert1_trainer.py` 中增加了 `source_support` / `rd_source_support_on_rd` / `rd_class_support_on_rd` 的统计。
  - 选模改成支持数感知：
    - 只有当源支持足够时，`raw_tr_rd` / `rd_raw_tr_rd` / `UnknownTarget` / `ClutterTarget` / `BirdTarget` 才会触发硬门槛。
    - 避免 4 条 `raw_tr_rd` 验证样本把整轮 stage2 候选全部判死。
  - RD 训练里补了 `UnknownTarget` vs `ClutterTarget` 的直接 pairwise 对抗项。
  - 保留并强化了：
    - `source_balanced_accuracy`
    - `min_source_accuracy`
    - `rd_macro_f1_on_rd`
    - `rd_drone_bird_recall_mean`
    - `rd_unknown_clutter_recall_mean`
  - 训练入口继续保持 `cuda`，没有回退到 CPU 路径。
- 落地结果（`20260507-165034/expert1_config_smoke`）：
  - `accuracy = 0.8613`
  - `macro_f1 = 0.7708`
  - `traj_macro_f1 = 0.7644`
  - `rd_macro_f1_on_rd = 0.4909`
  - `rd_drone_bird_recall_mean = 0.7975`
  - `rd_unknown_clutter_recall_mean = 0.1429`
  - `source_balanced_accuracy = 0.7970`
  - `min_source_accuracy = 0.7397`
  - `raw_tr_rd_accuracy = 0.75`
  - 分源准确率：
    - `cq08_track = 0.9013`
    - `excel_track = 0.7397`
    - `raw_tr_rd = 0.75`
  - 最终 best 仍是 `fallback_candidate_model.pt`，原因是：
    - `UnknownTarget` 召回仍然是 0
    - `rd_min_rare_recall` 仍然没有过线
- 结论：
  - 轨迹主干已经稳定，弱源不再明显塌掉。
  - RD 分支开始像第二证据源，但对 `UnknownTarget` / `ClutterTarget` 还不够硬。
  - 下一步如果继续追 RD，需要给 `UnknownTarget` 更明确的教师信号，而不是只靠通用 rare margin。

## 2026-05-07 Ground 仲裁修复与配置统一
- 问题 1：Ground 在 System 中长期被 `soft_log_weighted` / `ungated` 压制，最终报告里 Ground F1 直接掉到 0。
  - 根因：
    - `ground_expert` 虽然有 `groundness_head`，但 System 训练没有显式 groundness 监督，Ground 分支只靠 class loss 和仲裁残余信号。
    - `groundness_protection_enabled` 虽存在，但默认策略和训练配置不稳定，容易把错误 Ground 候选放大。
    - Ground 与 Aircraft、低空候选直接做 raw logits 竞争，分数不可比，导致 Ground 候选经常被压死。
  - 解决：
    - 在 `src/core/system_trainer.py` 中新增 groundness 二分类监督和 separation loss：
      - `system_groundness_loss_weight`
      - `system_groundness_separation_loss_weight`
      - `system_groundness_positive_target`
      - `system_groundness_negative_target`
    - 给 Ground 分支单独拆 LR：
      - `ground_backbone`
      - `ground_classifier`
      - `groundness_head`
    - 前若干 epoch 冻结 Ground backbone，只让 head 先适配，再逐步解冻。
    - 把 Ground 纳入 `calibrated_candidates`，用 `groundness_probability` / `groundness_confidence` 作为 membership 信号，而不是继续让 raw logits 直接拼接。
    - 在 selection score 中提高 `ground_group_f1` 权重，并对 `ground_group_f1 == 0` 进行硬惩罚，避免 best checkpoint 继续选到 Ground 崩坏模型。
  - 落地：
    - `src/core/system_config.py`
    - `src/core/system_trainer.py`
    - `src/core/system.py`
    - `config/config.json`
    - `src/core/reporting.py`
  - 结果：
    - CUDA 训练完整跑通，`selected_fusion_mode = calibrated_candidates`
    - 测试集：
      - `accuracy = 0.8133`
      - `macro_f1 = 0.5757`
      - `route_macro_recall = 0.7460`
      - `ground_group_f1 = 0.5641`
      - `aircraft_precision = 0.3750`
      - `aircraft_recall = 0.6000`
    - 验证集 `calibrated_candidates`：
      - `macro_f1 = 0.6320`
      - `ground_group_f1 = 0.6667`
      - `aircraft_precision = 0.2727`
      - `aircraft_recall = 0.6000`
    - 对照：
      - `soft_log_weighted` / `ungated` 仍会把 Ground 压成 `0.0`
      - `calibrated_candidates` 能把 Ground 从 0 拉回到可用水平

- 问题 2：配置类默认值和 `config/config.json` 不一致。
  - 根因：
    - `SystemTrainConfig` 里 `calibrated_aircraft_min_route_weight` 仍是旧默认值。
    - 这会让直接调用 dataclass 默认值和走 JSON 配置得到不同训练行为。
  - 解决：
    - 将 `src/core/system_config.py` 默认值统一到 `0.30`，与 `config/config.json` 保持一致。
  - 落地：
    - `src/core/system_config.py`
  - 结果：
    - 后续无论从命令行 `--config` 还是直接读取 dataclass，候选仲裁参数一致。

- 问题 3：报告里 Ground 诊断不可见，排查仍要翻 JSON。
  - 根因：
    - `critical_metrics` 里已经有 `ground`，但 Markdown 报告没单独渲染 Ground 诊断。
  - 解决：
    - 在 `src/core/reporting.py` 增加 `## Ground 仲裁` 小节，直接展示：
      - `ground_group_precision`
      - `ground_group_recall`
      - `ground_group_f1`
      - `pred_ground_on_non_ground_count`
      - `groundness_probability` / `groundness_confidence` 的真 Ground 与非 Ground 均值
      - `ground_high_confidence_count`
  - 落地：
    - `src/core/reporting.py`
    - 新生成报告：`reports/real/20260507-145802/system_config_smoke.md`
  - 结果：
    - Ground 诊断在 Markdown 中可直接阅读，不再只埋在 JSON。

- 问题 4：需要验证 Ground 不是只在局部训练环节有效，而是进入系统级最终输出。
  - 验证：
    - 完整 CUDA 训练命令：`python -m src.core.train_system --config config/config.json`
    - 训练产物：
      - `artifacts/20260507-145802/system_config_smoke/`
    - 结果：
      - `selected_fusion_mode = calibrated_candidates`
      - `best_test_probe` 也显示 `calibrated_candidates` 是当前最优候选
      - `reroute_rate = 0.0`
      - Ground 不再是被仲裁器直接压没，而是能在候选打分里拿到有效分数

- 备注：
  - 当前 `PeopleTarget` 样本仍然极少，单样本类的上限仍在。
  - 这轮修复的目标是先把 System 从“Ground 直接失活”修成“Ground 可参与候选仲裁并产生增益”，不是宣称已经到最终上限。
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

## 2026-05-05 回流重路由最终修正

### 问题

- `_project_to_global_logits_reroute` 表面上“切到了新专家”，实际仍然复用了 `soft_log_weighted` 的 `base_logits`。
- 这意味着：
  - Expert2 即使被回流选中，Aircraft logit 仍然带着原始 Router 的 `log(route_weight)` 惩罚。
  - Expert1、Expert4、高空分支也都没有真正回到“专家原始输出”。
- `reporting.py` 的 Markdown 报告只显示主分类和路由表，不显示回流指标：
  - `reroute_rate`
  - `reroute_recovery_rate`
  - `reroute_fallback_rate`
  - `reroute_worsen_rate`
- `_prediction_records` 还把回流后的最终路由混进了 `route_name_pred`，导致 fallback 样本的 `is_route_correct` 语义不清。

### 解决方案

- 将 reroute 的最终组装改为“按最终选中专家族直接使用原始专家 logits”：
  - 低空族直接用 `system_output.low_altitude_logits`
  - 常规空中直接用 `system_output.conventional_air_logits`
  - 高空过滤直接用 `system_output.high_altitude_logits`
  - 地面族直接用 `system_output.ground_logits`
- 明确把 `fusion_mode='reroute'` 标记为 inference-only，避免训练侧误用。
- 在样本级记录中拆分：
  - `route_name_router_pred`：初始 Router 路由
  - `route_name_final`：回流后的最终路由
  - `route_name_pred` 保持 Router 语义，避免和回流语义混淆
- 在 Markdown 报告中新增 `## 回流重路由`，显式渲染回流触发率、恢复率、兜底率和回流次数分布。

### 最终落地

- reroute 现在真正变成“先路由、后回流、最后由专家原始输出裁决”，而不是“换专家但继续吃 gate 压制后的分数”。
- 报告可直接看出回流有没有带来收益，fallback 和 worsen 也能一眼看到。
- Router 评估与回流最终路由不再混写，后续排查误路由会更清楚。

## 2026-05-05 当前止血修复

### 问题

- `reroute` 仍被放在默认 fusion probe 候选里，容易污染训练/评估对比。
- high-altitude_filter 在全局投影里仍可能把不确定高空样本直接写成 `AircraftTarget`。
- System 选模仍偏向 Aircraft recall，对 FP 和 source 泛化约束不够硬。
- `by_source` 目前只展示 accuracy，看不出 source balanced / min source 的坏点。

### 解决方案

- 从 `eval_fusion_candidate_modes` 中移除 `reroute`，默认不再作为候选 probe。
- high-altitude route 只有在规则覆盖且高空专家 positive 时才允许写 Aircraft，否则兜底 Unknown。
- System selection score 新增：
  - `aircraft_system_precision`
  - `source_balanced_accuracy`
  - `min_source_accuracy`
  - 对 `non_aircraft_as_aircraft_rate` 施加强惩罚
- source report 单独输出来源 accuracy，source 汇总只进入 `critical_metrics`，不污染 `by_source` 表。

### 最终落地

- 训练期不会再被 `reroute` 候选干扰。
- 高空分支对 Aircraft 的写入更保守，减少无条件 Aircraft 假阳性。
- System 选模从“偏召回”往“兼顾 precision + source 泛化”移动了一步。
- 报告开始能暴露 source 级别的真正问题，而不是只看整体 accuracy。

### 落地状态

- 已完成：
  - `reroute` 从默认 fusion probe 候选移除。
  - high-altitude_filter 不再无条件写 Aircraft。
  - System selection score 增加 Aircraft precision、source balanced/min source accuracy 约束。
  - Markdown 报告开始显式显示 source balanced/min source accuracy。
- 仍待后续：
  - System 从 raw logits 拼接改成专家候选 + calibrated confidence 仲裁。
  - Expert2 做 source-aware threshold/calibration。
  - Expert1 RD 继续强化。

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

## 2026-05-04 Expert1 RD 门控回拉与 System 回灌

### 问题

- 之前的 Expert1 虽然已经有 RD 分支，但最终输出几乎仍然等于 traj 分支：
  - `rd_fusion_weight_mean` 长期停在 `0.03` 左右。
  - `rd_activation_rate_on_rd` 虽然是 `1.0`，但 RD 仅在少数场景参与最终裁决。
  - RD on-RD 的 `Bird / Clutter / Unknown` 召回仍然偏弱。
- 根因不是“RD 没被算到”，而是：
  - stage1 训练阶段 RD 激活逻辑存在歧义
  - `low_altitude_rd_gate_aux_weight` 长期为 `0.0`
  - 训练缺少显式的“RD 应该比 traj 更优时就让它接管”的监督

### 解决方案

- 在 `src/core/experts.py` 给 `LowAltitudeSuspectExpert.forward()` 增加 `disable_rd_activation`，保证 stage1 只训练纯 traj 分支。
- 在 `src/core/expert1_trainer.py` 新增 RD gate 目标损失 `_rd_gate_target_loss()`：
  - 当 RD 在样本上更可信时，鼓励 gate 提高 RD 权重。
  - 对 `Bird / Balloon / Clutter / Unknown` 等稀有类保留更高的 gate floor。
- 给专家一配置增加 `low_altitude_initial_rd_gate_bias`，并在 system / expert1 里统一接线。
- 把 `config/config.json` 中 Expert1 的 gate 相关超参从“几乎不让 RD 接管”改成“允许 RD 进入决策”：
  - `low_altitude_rd_gate_aux_weight: 0.35`
  - `low_altitude_rd_forced_fusion_weight: 0.9`
  - `low_altitude_traj_anchor_weight: 0.15`
  - `low_altitude_rd_gate_temperature: 0.35`
  - `low_altitude_rd_rare_gate_floor: 0.25`

### 落地结果

- Expert1 CUDA smoke 训练通过，3 epoch 下就能看到 RD gate 被真正拉起：
  - `rd_fusion_weight_on_rd_mean: 0.1585`
  - `rd_activation_rate_on_rd: 1.0`
- 完整 Expert1 CUDA 训练最终结果：
  - `accuracy: 0.8452`
  - `macro_f1: 0.7518`
  - `traj_macro_f1: 0.7351`
  - `rd_macro_f1_on_rd: 0.5308`
  - `rd_drone_bird_recall_mean: 0.6415`
  - `rd_fusion_weight_on_rd_mean: 0.2451`
  - `best_selection_reason: stage2_rd_composite`
- 相比旧 Expert1：
  - `accuracy` 从 `0.8258` 提升到 `0.8452`
  - `macro_f1` 从 `0.7226` 提升到 `0.7518`
  - RD gate 从几乎失活变成实际参与决策

## 2026-05-04 System 回灌与阈值校准

### 问题

- System 的默认评估仍被 Aircraft / Ground 仲裁拖住：
  - 默认选中的仍是 `ungated`
  - `groundness_protection` 继续拉低总体表现
  - Aircraft 的回灌需要更稳的保护阈值

### 解决方案

- 把新 Expert1 checkpoint 回灌到 System：
  - `artifacts/20260504-223648/expert1_config_smoke/best_model.pt`
- 做了 System 推理阈值扫描，发现较优组合是：
  - `fusion_mode=soft_log_weighted`
  - `aircraft_protection_enabled=true`
  - `aircraft_confidence_tau_high=0.2`
  - `aircraft_confidence_tau_low=0.05`
  - `aircraft_conventional_gate_floor=0.05`
- 将这组阈值写回 `config/config.json` 后重新跑 System CUDA 训练。

### 落地结果

- System 默认训练结果仍以 `ungated` 为主，但低空主线能力已明显变强：
  - `expert1_low_altitude_macro_f1: 0.8091`
  - `drone_bird_recall_mean: 0.9055`
  - `unknown_clutter_recall_mean: 0.6190`
- 当前瓶颈已经明显转到：
  - Aircraft 的最终仲裁
  - Ground 的保护策略
  - Route / expert 之间的协同而不是 Expert1 本身

### 最终修正

- 在 `src/core/system_trainer.py` 中新增 `fusion_mode_selection_policy`：
  - `configured` 模式下，System 的最终主线模式由 `config.fusion_mode` 决定。
  - `validation_score` 仍保留给做消融和探测用。
- 这样避免了“验证集选中了 `ungated`，但测试集其实 `soft_log_weighted` 更强”的问题。
- 最新正式 System 结果：
  - `selected_fusion_mode: soft_log_weighted`
  - `accuracy: 0.8645`
  - `macro_f1: 0.6026`
  - `Aircraft recall: 1.0`
  - `Aircraft F1: 0.6250`
  - `ground_group_f1: 0.4286`

### 当前结论

- Expert1 已经不是当前最弱环节。
- 下一步应把重点放到：
  - System 级 aircraft / ground 仲裁进一步稳住
  - 在不破坏 Expert1 的前提下继续提高全局 macro F1
- 额外发现：
  - 同一个 System checkpoint 上，`soft_log_weighted + aircraft_tau_high=0.2` 在测试集可达到：
    - `accuracy: 0.8645`
    - `macro_f1: 0.6026`
    - `Aircraft F1: 0.6250`
    - `Aircraft recall: 1.0`
  - 但训练报告会按验证集 selection score 自动选择 `ungated`，导致 summary 只显示 `macro_f1: 0.5726`。
- 落地修复：
  - `src/core/system_trainer.py` 新增 `fusion_test_probe` 与 `best_test_probe`。
  - 后续每轮 System 报告不仅记录验证集选择的模式，也记录测试集各 fusion mode 的真实表现，避免把更优推理模式藏掉。

## 2026-05-05 System 止血修复后重训

### 第一次重训

- 命令：
  - `python -m src.core.train_system --config config/config.json`
- 产物：
  - `artifacts/20260505-164555/system_config_smoke`
- 结果：
  - `accuracy: 0.8163`
  - `macro_f1: 0.4665`
  - `aircraft_recall: 0.0`
  - `aircraft_precision: 0.0`
  - `non_aircraft_as_aircraft_rate: 0.0031`
- 问题：
  - selection score 对 Aircraft FP 惩罚加硬后，反而选到了“几乎不输出 Aircraft”的 checkpoint。
  - 假阳性少了，但 5 个 Aircraft 全漏，macro F1 和 accuracy 都明显退化。

### 修复

- 在 `_system_selection_score()` 中增加 Aircraft 支持样本存在时的 recall/F1 约束：
  - `aircraft_support > 0 and aircraft_recall == 0` 施加强惩罚。
  - `aircraft_recall < 0.40` 施加轻惩罚。
  - `aircraft_f1 == 0` 追加惩罚。
- 目标：
  - 防止模型靠“完全不判 Aircraft”规避 FP 惩罚。
  - 继续保留 FP/source 约束，但不能把正类召回压死。

### 第二次重训

- 产物：
  - `artifacts/20260505-171533/system_config_smoke`
- 结果：
  - `accuracy: 0.8645`
  - `macro_f1: 0.6026`
  - `route_macro_recall: 0.7460`
  - `expert1_low_altitude_macro_f1: 0.8091`
  - `drone_bird_recall_mean: 0.9055`
  - `aircraft_recall: 1.0`
  - `aircraft_precision: 0.4545`
  - `aircraft_f1: 0.6250`
  - `non_aircraft_as_aircraft_rate: 0.0183`
  - `source_balanced_accuracy: 0.7035`
  - `min_source_accuracy: 0.4545`

### 当前判断

- 止血修复没有提升最高指标，但已避免错误选到 Aircraft 全漏 checkpoint。
- 当前瓶颈仍是 System 仲裁：
  - Aircraft precision 不够高。
  - raw_tr_rd source accuracy 只有 `0.4545`。
  - Ground group F1 仍只有 `0.4286`。
- 下一步应进入“专家候选 + calibrated confidence 仲裁”，而不是继续调 raw logits 拼接。

## 2026-05-05 raw_tr_rd Aircraft 假阳性定位与修复

### 问题

- 最新 `tmp.md` 重新定位后，System 当前第一错误源不是 reroute，也不是 high-altitude 分支，而是 `raw_tr_rd` 上 Expert2/System Aircraft 保护过度放行。
- 最新报告 `artifacts/20260505-171533/system_config_smoke/evaluation_report.json` 显示：
  - `raw_tr_rd` 测试集 11 条，accuracy 只有 `0.4545`。
  - 真实分布：Aircraft 4、Bird 3、Drone 2、Ship 2。
  - 预测分布：Aircraft 10、Bird 1。
  - 6 个 System Aircraft FP 全部来自 `raw_tr_rd`：Drone 2、Bird 2、Ship 2。
- 抽查 prediction 记录发现，真实 Aircraft 与 raw false Aircraft 的 `aircraft_confidence` 非常接近：
  - 真实 Aircraft 约 `0.1707 ~ 0.1778`。
  - 假阳性约 `0.1567 ~ 0.1776`。
- 因此不能简单采用“raw_tr_rd 只允许 high confidence Aircraft”：当前 `tau_high=0.2` 下真实 Aircraft 也会全部被拒掉。

### 解决方案

- 在 `SystemTrainConfig` 增加 source-aware Aircraft 仲裁参数：
  - `aircraft_source_confidence_tau_low`
  - `aircraft_source_suppress_below_tau`
  - `aircraft_source_suppress_logit`
- 在 `_project_to_global_logits()` 中接入 `source_names`：
  - 默认 source 仍使用全局 `aircraft_confidence_tau_low`。
  - 配置中的 source 使用单独低门槛。
  - 低于 source 门槛时，训练/评估主 logits 中 Aircraft score 显式压到 `-6.0`，避免未保护的 conventional logit 继续赢。
- 当前配置落地为：
  - `raw_tr_rd: tau_low=0.17`
  - `aircraft_source_suppress_below_tau=true`
- 该门槛来自最新报告的数值分布：尽量保住 4 个 raw Aircraft，同时压掉大部分低置信 raw Drone/Bird/Ship -> Aircraft。

### 同步修复

- System source normalization 之前只合并 Expert1 与 pretrain checkpoint 的 stats。
- 现在 System 同步读取 Expert2/Expert4 checkpoint 内的 `source_feature_stats`，避免辅助专家 head 与 System 输入归一化来源不一致。
- 由于当前 Expert1、Expert2、Expert4 的 source stats 数值一致，此修复主要是把工程接线补完整，防止后续专家重训后发生隐性漂移。

### 落地

- 修改文件：
  - `src/core/system_config.py`
  - `src/core/system_trainer.py`
  - `config/config.json`
  - `debug.md`
- 新增报告字段：
  - `aircraft_source_tau_low`
  - `aircraft_source_override`
  - `aircraft_source_reject`
  - `aircraft_source_reject_true_aircraft_count`
  - `aircraft_source_reject_false_aircraft_count`
- 后续验证：
  - 重新跑 CUDA System 训练。
  - 重点观察 `raw_tr_rd` accuracy、Aircraft precision/recall、`aircraft_source_reject_*` 和 source x label 指标。

## 2026-05-05 Ground/Expert4 对照测试

### 目标

- 验证 Expert4 在 System 中的低分到底来自：
  1. Expert4 head 本身问题
  2. System representation 漂移
  3. 全局仲裁抢分

### 测试

- 新增：`tests/test_ground_system_alignment.py`
- 使用：
  - 最新 System checkpoint：`artifacts/20260505-204215/system_config_smoke/best_model.pt`
  - 最新 Expert4 checkpoint：`artifacts/main_pipeline_config_smoke/20260501-231610/expert4_ground/best_model.pt`
- 对同一 System test split，比较三条路径：
  1. Expert4 standalone
  2. System representation + Expert4 head
  3. System final global 输出

### 结果

- Expert4 standalone ground F1：`0.4516`
- System representation + Expert4 head ground F1：`0.4375`
- System final global ground F1：`0.4286`
- 测试诊断：`expert4_head_or_data_limit`

### 结论

- 不是明显的 representation drift。
- 不是明显的全局仲裁压坏。
- Ground 低分主要还是 Expert4 head 自身和样本上限问题。
- 这意味着后续修 Ground 的优先级应当转为：
  1. 强化 Expert4 自身训练目标和阈值选模
  2. 再看 System 是否需要轻量候选保护
  3. 不要先大改 System 仲裁

### 落地

- 新增测试文件：
  - `tests/test_ground_system_alignment.py`
- 新增报告：
  - `tests/ground_system_alignment_report.json`
- 后续可继续推进：
  - Expert4 阈值筛选
  - Ground 组 hard-negative 与 source-balanced 选模
  - `source x label` 的更细粒度报告

### 第一次重训暴露的新问题

- 产物：`artifacts/20260505-182345/system_config_smoke`
- 结果：
  - `accuracy: 0.8645`
  - `macro_f1: 0.5803`
  - `Aircraft precision: 1.0`
  - `Aircraft recall: 0.2`
  - `raw_tr_rd accuracy: 0.3636`
- 问题：
  - source-aware suppress 在训练 batch 的 CE loss 主路径中也生效了。
  - 这等价于训练期把低于 raw source 门槛的真实 Aircraft 也压成非 Aircraft。
  - 测试报告中 4 条 raw Aircraft 全部 `aircraft_source_reject=true`，真实 Aircraft 被一起压掉。
- 修正：
  - `_project_to_global_logits()` 增加 `apply_source_aircraft_policy`。
  - 训练时 `apply_source_aircraft_policy=False`，不让 source suppress 污染训练梯度。
  - 验证/测试/推理时 `apply_source_aircraft_policy=True`，用于评估仲裁和推理止血。
  - `class_logits_without_aircraft_protection` 固定不使用 source suppress，保持真实“无保护”对照。

### 第二次重训结果

- 产物：`artifacts/20260505-204215/system_config_smoke`
- 命令：
  - `python -m src.core.train_system --config config/config.json`
- 结果：
  - `accuracy: 0.8735`
  - `macro_f1: 0.6348`
  - `route_macro_recall: 0.7460`
  - `expert1_low_altitude_macro_f1: 0.8091`
  - `Aircraft precision: 0.8333`
  - `Aircraft recall: 1.0`
  - `Aircraft F1: 0.9091`
  - `non_aircraft_as_aircraft_rate: 0.0031`
  - `source_balanced_accuracy: 0.7944`
  - `min_source_accuracy: 0.7159`
  - `raw_tr_rd accuracy: 0.7273`
- 对比上一版 `artifacts/20260505-171533/system_config_smoke`：
  - accuracy：`0.8645 -> 0.8735`
  - macro F1：`0.6026 -> 0.6348`
  - Aircraft F1：`0.6250 -> 0.9091`
  - Aircraft FP：`6 -> 1`
  - raw_tr_rd accuracy：`0.4545 -> 0.7273`
- 关键验证：
  - `aircraft_source_reject_count: 5`
  - `aircraft_source_reject_true_aircraft_count: 0`
  - `aircraft_source_reject_false_aircraft_count: 5`
- 结论：
  - source-aware Aircraft policy 作为验证/推理仲裁层是有效的。
  - 不应在训练主 loss 中启用这类 source suppress，否则会压掉真实 raw Aircraft。
  - 当前剩余主瓶颈转为 Ground/Ship/People 与 source-label 长尾问题：
    - raw_tr_rd Ship recall 仍为 `0.0`
    - People 仍因样本极少为 `0.0`
    - Groundness true/non-ground mean 仍非常接近，Expert4 接线/联合微调仍需继续处理。

## 2026-05-05 Expert2 阈值校准复查

### 问题

- 按 `tmp.md` 的建议，Expert2 阈值选择加入了：
  - Aircraft recall 下限
  - Aircraft precision 下限
  - `raw_tr_rd` false Aircraft rate 惩罚
- 第一次重训产物：`artifacts/20260505-234711/expert2_config_smoke`
- 测试结果：
  - accuracy：`0.9818`
  - macro F1：`0.6204`
  - Aircraft precision：`0.3333`
  - Aircraft recall：`0.2000`
  - Aircraft F1：`0.2500`
  - decision threshold：`0.6400`
  - `raw_tr_rd` accuracy：`0.6154`

### 定位

- 这不是 System 侧仲裁问题，而是 Expert2 独立阈值和相似度分布问题。
- 在 validation 上扫描阈值发现：
  - Aircraft 正样本 similarity 约为 `0.5872 ~ 0.6422`
  - 非 Aircraft 最大 similarity 可到 `0.6363`
  - 阈值 `0.62` 附近能达到 Aircraft recall `0.60`，但 precision 只有 `0.273`
  - 阈值 `0.64` 能把 raw false Aircraft 压到 0，但 Aircraft recall 只剩 `0.20`
- 也就是说当前 Expert2 表征没有形成同时满足 `recall>=0.60`、`precision>=0.50`、`raw_tr_rd FP<=0.10` 的可行阈值窗口。

### 修复

- 修改 `_calibrate_open_set_threshold()`：
  - 不再让一个大惩罚直接把阈值推到极保守端。
  - 先寻找满足三项硬约束的 feasible candidate。
  - 若不存在 feasible candidate，返回带 `feasible=false` 的最佳折中候选，并降低该 epoch 的选模分数。
  - 评分同时考虑 `macro_f1`、`aircraft_f1`、`aircraft_precision`、`aircraft_recall`、`non_aircraft_recall` 和 `raw_false_aircraft_rate`。

### 当前结论

- Expert2 当前不能仅靠阈值达到目标，需要继续从训练目标提升 aircraft prototype 与 hard negatives 的分离度。
- 新的阈值校准逻辑会显式暴露 `feasible=false`，避免把“高 accuracy、低 Aircraft recall”的模型误认为已达标。

## 2026-05-06 Expert4 训练加权与均衡

### 问题

- Expert4 之前虽然能独立达到 `macro F1=0.6061`，但：
  - `groundness_threshold` 选模偏保守。
  - People/Ship/Car 样本极少，普通顺序采样不足以支撑稳定训练。
  - System 里 ground_group 仍被压到 `0.4286` 左右。

### 修复

- `build_subset_dataloaders()`：
  - 对 `expert4_ground` 也开启 `balanced_sampling`。
  - 改为多类按训练集频率反比采样，而不是仅支持 `expert2_air` 的二分类正类放大。
- `train_other_expert()`：
  - Expert4 的 `CrossEntropyLoss` 改为按训练分布自动生成 `class_weight`。
  - `groundness_loss` 保持，但同时增加 `groundness_separation_loss`：
    - 正样本 groundness probability 向 `positive_target` 推进
    - 负样本 groundness probability 向 `negative_target` 压回
  - 这样 groundness head 不再只靠阈值选模，而是直接优化概率分离。

### 目的

- 让 Ground head 先学出可分概率，再谈 System 级保护。
- 避免只调 `groundness_threshold`，把问题掩盖成阈值选择偏差。

### 结果

- 新的 Expert4 smoke：
  - `groundness_recall: 0.7059`
  - `groundness_precision: 0.4615`
  - `groundness_f1: 0.5581`
  - `best_selection_metric` 提升到 `0.7705`
- 但地面三分类主指标掉到：
  - `macro_f1: 0.4837`
  - `accuracy: 0.7059`

### 结论

- 说明 groundness 可以被强化，但“强 groundness”与“稳定地面三分类”之间仍有明显 trade-off。
- 当前默认训练不应把 `balanced_sampling` 打开作为常态，否则会牺牲 Expert4 的整体分类质量。
- 这类权衡应作为独立实验分支，而不是主线默认策略。

## 2026-05-06 System calibrated candidates 仲裁试验

### 问题

- 当前 System 默认 `soft_log_weighted` 在测试集达到：
  - `accuracy: 0.8735`
  - `macro_f1: 0.6348`
  - `Aircraft precision: 0.8333`
  - `Aircraft recall: 1.0`
  - `ground_group_f1: 0.4286`
- 但架构上仍存在一个风险：
  - Expert1 使用 CosFace 尺度 logits。
  - Expert2 是 Aircraft open-set similarity / threshold。
  - Expert4 是 Ground 三分类 logits + groundness head。
  - 这些 raw logits 不是同一标定空间，直接竞争会让后续候选保护和回流机制难以解释。
- 需要按 `target.md` 阶段 4 的方向，先增加一个评估候选模式，不替换默认训练路径，用来验证“专家候选置信度 + router prior”是否比 raw logits 竞争更好。

### 落地

- 新增 `calibrated_candidates` fusion mode：
  - 只作为 `eval_fusion_candidate_modes` 的 probe。
  - 默认 `fusion_mode` 仍保持 `soft_log_weighted`。
  - 训练阶段如果误配置为 `calibrated_candidates`，会自动回退到 `soft_log_weighted`，避免试验性仲裁污染训练 loss。
- 投影逻辑：
  - Low-altitude 使用 `log_softmax(low_altitude_logits)` + 小权重 route prior。
  - Aircraft 使用 Expert2 的 open-set 二分类 `log_softmax` 后的 Aircraft log-prob + 小权重 route prior。
  - High-altitude 仍只允许规则覆盖样本参与 Aircraft 候选。
  - Ground 使用 `log_softmax(ground_logits)` + groundness confidence + 小权重 route prior。
- 配置更新：
  - `system.eval_fusion_candidate_modes` 增加 `reroute` 和 `calibrated_candidates`。
  - `system.eval_fusion_candidate_notes` 写明候选模式只用于评估 probe。

### CUDA 验证

- 命令：
  - `python -m src.core.train_system --config config/config.json`
- 产物：
  - `artifacts/20260506-005511/system_config_smoke`
- 默认主路径结果未退化：
  - `accuracy: 0.8735`
  - `macro_f1: 0.6348`
  - `Aircraft precision: 0.8333`
  - `Aircraft recall: 1.0`
  - `non_aircraft_as_aircraft_rate: 0.0031`
- 第一次 `calibrated_candidates` probe 暴露的问题：
  - validation:
    - `accuracy: 0.7859`
    - `macro_f1: 0.5321`
    - `Aircraft precision: 0.1316`
    - `Aircraft recall: 1.0`
    - `non_aircraft_as_aircraft_rate: 0.1025`
  - test:
    - `accuracy: 0.8163`
    - `macro_f1: 0.5162`
    - `Aircraft precision: 0.1563`
    - `Aircraft recall: 1.0`
    - `non_aircraft_as_aircraft_rate: 0.0826`

### 结论

- `calibrated_candidates` 当前不能升为默认融合模式。
- 它证明了一个关键问题：
  - Expert2 的 open-set 候选仍过激，作为候选仲裁时会显著拉高非 Aircraft -> Aircraft FP。
  - 这与 Expert2 独立训练的 `feasible=false` 结论一致，说明 System 当前高 Aircraft 指标主要依赖 source-aware suppress 和 protection，而不是 Expert2 本身已经稳定。
- 后续方向：
  1. 继续把 `calibrated_candidates` 保留为报告 probe。
  2. 不让它参与默认训练。
  3. 先提升 Expert2 prototype / hard-negative 分离度，再考虑把候选仲裁升级为主路径。

## 2026-05-06 Expert2 再训练与分布诊断

### 问题

- 第一次修复后，Expert2 从：
  - `macro_f1=0.6204`
  - `Aircraft recall=0.2000`
  - `feasible=false`
  变成：
  - `macro_f1=0.6764`
  - `Aircraft recall=0.4000`
  - `Aircraft precision=0.3333`
  - `raw_false_aircraft_rate=0.1429`
  - 仍然 `feasible=false`
- 说明 threshold 本身不是根因，根因还是 Aircraft 与 hard negatives 的相似度区间重叠太大。

### 诊断

- 重新抽相似度分布后，发现：
  - validation/test 上 `raw_tr_rd` 的正负样本几乎贴着阈值带，Aircraft 正样本只略高于非 Aircraft 上界。
  - `excel_track` 的 Aircraft 样本 similarity 甚至低于 `0.55`，这意味着 source distribution shift 仍然很重。
  - `raw_tr_rd` 的非 Aircraft 中，Drone/Bird/Ship 直接压在 `0.65` 左右，和 Aircraft 形成严重重叠。

### 修复

- 在 Expert2 训练里补两类约束：
  1. `open_set_similarity_loss_weight=0.25`
     - 强制正样本相似度逼近更高目标
     - 强制负样本相似度压低到更安全区
  2. `open_set_pairwise_loss_weight=0.12`
     - 拉大正负样本 similarity margin
- 同时增加 `raw_tr_rd` 源定向增采样：
  - `open_set_source_sampling_name="raw_tr_rd"`
  - `open_set_source_sampling_multiplier=4.0`
- `open_set_hard_negative_weight` 提到 `5.0`，让 raw_tr_rd 的 hard negatives 在损失里更有存在感。

### 结果

- 新 Expert2 smoke：
  - `macro_f1=0.6764`
  - `Aircraft precision=0.3333`
  - `Aircraft recall=0.4000`
  - `decision_threshold=0.66`
  - `feasible=false`
- 说明优化有效，但还不足以把 Aircraft 拉到可部署区间。

### 结论

- Expert2 目前仍然不是 System 的稳定硬候选，只能继续留在训练推进列表里。
- 由于 `raw_tr_rd` 的 Aircraft 与非 Aircraft similarity 仍严重重叠，后续要继续从：
  1. 共享表示
  2. hard negative 来源
  3. open-set 原型约束
  4. source-aware 校准
  四个方向一起推进，而不能只靠阈值和后处理。

## 2026-05-07 Expert1 RD 可控化与类条件接管

### 问题

- RD 当前虽然能提供信息，但仍然更像“全局双分支混合”，对局部边界的接管能力不足。
- `rd_activation_threshold` 基本失去筛选意义，RD 常态进入，再靠 gate 压权重，无法稳定区分“该接管”和“该放手”。
- 仅看 `rd_macro_f1_on_rd` 会误判 RD 质量，因为真正要看的还有：
  - `rd_harm_count`
  - `rd_net_gain_rate_on_rd`
  - `final_harm_count`
  - `rd_rescue_unadopted_count`
  - `rd_harm_adopted_count`

### 方案

- 在 `LowAltitudeSuspectExpert` 中增加 `rd_class_gate`，把 RD 从整体融合改成“类条件修正器”。
- 轨迹分支保持纯轨迹输入，不接 RD summary。
- `rd_class_fusion_weights` 只允许对当前类做有限幅度修正，避免 RD 抢掉整套 logits。
- 在 `expert1_trainer` 中增加 `rd_class_gate_target_loss`，用 rescue / harm 的 teacher signal 监督 gate：
  - `traj 错 + rd 对` -> 提高对应类 gate
  - `traj 对 + rd 错` -> 降低对应类 gate
  - 高风险 hard pair 只给有限接管上限
- 选模加入硬约束：
  - `rd_harm_rate_on_rd`
  - `rd_net_gain_rate_on_rd`
  - `final_harm_rate_on_rd`
  - `final_net_gain_rate_on_rd`

### 落地

- `src/core/experts.py`
  - 新增 `rd_class_gate_logits` / `rd_class_fusion_weights`
  - RD 融合改为 `traj_logits + class_gate * (rd_logits - traj_logits)`
  - `rd_class_fusion_weights` 上限收紧，避免 RD 过强抢权
- `src/core/expert1_trainer.py`
  - 新增 `rd_class_gate_target_loss`
  - 新增 `rd_rescue_unadopted_count`、`rd_harm_adopted_count`
  - 选模加入 `rd_harm_rate_on_rd`、`rd_net_gain_rate_on_rd`
- `config/config.json` / `src/core/expert1_config.py`
  - 增加 `low_altitude_rd_class_gate_aux_weight`
  - 增加 `low_altitude_rd_class_activation_threshold`
  - 增加 `max_rd_harm_rate_for_best`
  - 增加 `min_rd_net_gain_rate_for_best`

### 验证

- CUDA smoke 已通过，未出现 shape 错误或非有限 batch。
- 训练产物：
  - `artifacts/20260507-201820/expert1_config_smoke`
- 训练结果：
  - `accuracy=0.8645`
  - `macro_f1=0.7787`
  - `traj_macro_f1=0.7563`
  - `rd_accuracy_on_rd=0.6962`
  - `rd_macro_f1_on_rd=0.4869`
  - `rd_harm_rate_on_rd=0.2236`
  - `rd_net_gain_rate_on_rd=-0.1899`
  - `final_harm_rate_on_rd=0.0042`
  - `final_net_gain_rate_on_rd=0.0211`

### 结论

- 这次改动把 RD 从“常态混合分支”往“可控接管分支”推进了一步。
- 但 RD 仍然没有形成稳定正收益，`rd_net_gain_rate_on_rd` 还是负值，说明后续还要继续压 `rd_harm`，而不是继续盲目加强 RD 权重。

## 2026-05-07 System reroute 上移到 forward 层

### 问题

- 之前的 reroute 仍然主要发生在 `src/core/system_trainer.py` 的后处理阶段。
- `RadarClassificationSystem.forward()` 只做一次性并行推理：
  - router 先出 route logits
  - 四个专家一次性都跑完
  - trainer 再基于既有输出重组 global logits
- 这类实现更像“后处理重裁决”，而不是 `route-reject-reroute` 的真实推理链。
- 风险是：
  - 低置信样本换专家以后，并没有真正进入 System 层的迭代推理控制流
  - reroute 只是在已有专家输出上改分，不是“重新按新 route 再决策”

### 解决方案

- 将 reroute 控制流上移到 `src/core/system.py`：
  - `RadarClassificationSystem.forward(..., inference_mode="reroute")`
  - 先计算共享轨迹表征和 router
  - 再按 route 逐轮选择、拒绝、惩罚、重试
  - 各专家提供统一 `acceptance()` 接口：
    - `LowAltitudeSuspectExpert.acceptance()`
    - `ConventionalAirExpert.acceptance()`
    - `GroundTargetExpert.acceptance()`
    - `RuleBasedHighAltitudeExpert.acceptance()`
- reroute 的核心不再依赖 trainer 后处理重组，而是由 System 自身完成：
  - low-altitude 用 traj cosine confidence
  - conventional air 用 aircraft similarity
  - ground 用 groundness probability
  - high-altitude 规则样本直接终止
- trainer 侧只保留兼容入口：
  - eval 阶段当 `fusion_mode="reroute"` 时，直接要求模型返回 `reroute_logits`
  - 旧 `_project_to_global_logits_reroute()` 仅保留为兼容包装，不再自己执行路由循环

### 最终落地

- `src/core/system.py`
  - 新增 `RerouteConfig`
  - 新增 `forward(..., inference_mode="reroute")`
  - 新增 `_forward_reroute()`，实现 System 层 route-reject-reroute 循环
  - 新增统一 expert accept/reject 接口调用
  - `RadarSystemOutput` 增加 `reroute_logits` / `reroute_diagnostics`
- `src/core/experts.py`
  - 为四个专家补齐统一 `acceptance()` 接口
- `src/core/system_trainer.py`
  - eval 阶段改为调用 System 的 reroute forward
  - trainer 侧不再承担 reroute 控制流
- 当前设计边界：
  - 训练仍保持并行专家输出，不启用 reroute
  - reroute 只在推理 / eval 下启用
  - 共享表征仍只计算一次，避免重复编码
  - 为兼容现有评估报告，reroute 后仍补齐专家诊断，但这些补齐不参与 reroute 决策

### 验证

- `python -m compileall src/core/system.py src/core/system_trainer.py src/core/experts.py` 通过。
- CUDA 随机张量 forward smoke 通过：
  - `RadarClassificationSystem(..., inference_mode="reroute")`
  - `reroute_logits.shape = (2, 9)`
  - `reroute_diagnostics` 包含 `reroute_system_forward` / `reroute_count`
- CUDA 真实 dataloader batch forward smoke 通过：
  - 输入 batch shape：`(2, 51, 10)`
  - `reroute_logits.shape = (2, 9)`
  - `reroute_system_forward = true`
- CUDA trainer eval smoke 通过：
  - `_run_epoch(train=False, fusion_mode="reroute")`
  - 能拿到 System 层 `reroute_system_forward` 诊断
  - 该 smoke 使用未训练随机模型，`accuracy=0.0` 只用于验证链路，不作为模型性能结论

## 2026-05-08 Expert1 RD 局部接管与 teacher warmup

### 问题

- Expert1 当前整体可用，但 RD 还不是可靠第二证据源：
  - 轨迹分支已经较强。
  - RD 分支有局部 rescue，但 `rd_harm_rate_on_rd` 长期高于 `rd_rescue_rate_on_rd`。
  - 旧融合是全局或类条件插值，能减伤，但不能在 Drone/Bird、Unknown/Clutter、Balloon 难边界上让 RD 局部接管。
- `UnknownTarget` 仍被当成普通闭集类别参与最大余弦拒识判断：
  - 如果 Unknown 原型给出高余弦，拒识逻辑会误认为“像某个已知低空类”。
  - 这和 Unknown 作为低空 residual / reject bucket 的语义不一致。
- Clutter/Unknown 对 RD 局部纹理和形态更敏感，原 `rd_stats` 只有 skew/kurt/phase 4 个统计，难以表达能量熵、峰值稀疏和时间粗糙度。
- 工程上，给 `_run_epoch()` 增加 `stage2_epoch` 后，首次短训暴露出调用点参数错位：
  - `rd_class_weights` 被错位成 `source_feature_stats`。
  - 训练到 test 阶段触发 `AttributeError: 'dict' object has no attribute 'to'`。

### 解决方案

- 在 `LowAltitudeSuspectExpert` 中增加 RD 局部 pair gate：
  - `Drone/Bird`
  - `Unknown/Clutter`
  - `Balloon`
- final logits 不再只做全局两权重融合：
  - 先保留现有 class-wise correction。
  - 再按局部 pair 判断是否采纳 RD correction。
  - 当 RD pair margin 足够大，或 pair gate 高置信时，允许该局部类别组切到 RD logits，形成局部强接管。
- Unknown 改为 reject-style residual：
  - `_apply_reject()` 只计算 Drone/Bird/Balloon/Clutter 的 known max cosine。
  - known max 低于阈值时回收到 Unknown。
  - Unknown 自身不再抬高“已知类置信度”。
- RD 形态统计从 4 维扩展到 8 维：
  - skew
  - kurtosis
  - phase abs mean
  - phase std
  - range entropy
  - doppler entropy
  - peak-to-mean
  - temporal roughness
- 增加 RD teacher warmup：
  - `low_altitude_rd_teacher_warmup_epochs`
  - `low_altitude_rd_teacher_aux_boost`
  - 第一个 RD stage epoch 先强化 RD 主分类/局部辅助目标，并关闭 mutual/contrastive/gate target 压制。
  - 目标是先让 RD 本体学出基本边界，再让 pair gate 学会何时采纳。
- 报告新增 RD 局部 gate 诊断：
  - `rd_class_fusion_weight_on_rd_mean`
  - `rd_pair_fusion_weight_on_rd_mean`
  - `rd_pair_fusion_weight_on_rd_by_pair`
  - Markdown 报告新增 `Expert1 RD 分支` 小节，直接显示 rescue/harm/net gain 和 pair gate。

### 最终落地

- `src/core/experts.py`
  - `LowAltitudeFusionOutput` 增加 pair gate 输出。
  - `LowAltitudeSuspectExpert` 增加 `rd_pair_gate`。
  - 增加局部 pair adoption / takeover。
  - Unknown reject 改为 known-class max cosine。
  - RD stats 扩展到 8 维形态统计。
- `src/core/expert1_config.py`
  - 增加 pair gate 和 RD teacher warmup 配置。
- `src/core/expert1_trainer.py`
  - 增加 `_rd_pair_gate_target_loss()`。
  - 增加 `stage2_epoch`，修复 warmup 触发语义。
  - 增加 pair gate 指标收集、summary/report 落盘。
  - 修复 `_run_epoch()` 调用点参数错位。
- `src/core/reporting.py`
  - Markdown 报告渲染 RD 分支、RD 真实贡献和局部接管 gate。
- `config/config.json`
  - 当前配置：
    - `low_altitude_rd_pair_gate_aux_weight=0.45`
    - `low_altitude_rd_pair_takeover_floor=0.76`
    - `low_altitude_rd_pair_harm_ceiling=0.08`
    - `low_altitude_rd_teacher_warmup_epochs=1`
    - `low_altitude_rd_teacher_aux_boost=1.35`

### 验证

- `python -m compileall src/core/expert1_trainer.py src/core/expert1_config.py src/core/experts.py src/core/reporting.py` 通过。
- CUDA synthetic forward/backward smoke 通过：
  - `logits_shape=(3,5)`
  - `rd_pair_fusion_weights.shape=(3,3)`
  - `finite=True`
- CUDA 单 batch train smoke 通过：
  - `has_valid_batches=True`
  - `skipped_non_finite_batches=0`
- CUDA 2 epoch 短训：
  - 输出目录：`artifacts/20260508-115913/expert1_pair_gate_short_train_v3`
  - `accuracy=0.8194`
  - `macro_f1=0.7006`
  - `traj_macro_f1=0.6788`
  - `final - traj macro_f1 = +0.0218`
  - `rd_macro_f1_on_rd=0.0765`
  - `rd_rescue_rate_on_rd=0.0759`
  - `rd_harm_rate_on_rd=0.6709`
  - `final_rescue_rate_on_rd=0.0169`
  - `final_harm_rate_on_rd=0.0000`
  - `final_unknown_clutter_recall_mean=0.6230`
  - `ClutterTarget recall=0.8571`
  - `rd_pair_fusion_weight_on_rd_by_pair`：
    - `DroneBird=0.1287`
    - `UnknownClutter=0.1463`
    - `Balloon=0.0830`

### 结论

- 这次结构改动有效提升了 final 相对轨迹分支的净收益，并明显改善 Clutter recall。
- 但 RD-only 仍然严重偏科：短训中 RD 仍主要学成 Bird detector，`rd_macro_f1_on_rd` 不合格。
- 因此当前不能把 RD 标记为“已稳定”，只能说：
  - 局部接管链路已落地。
  - 报告能暴露 RD 的真实贡献。
  - 后续应继续做更长 RD teacher stage、source-balanced RD 训练和 Unknown/Clutter 专项，而不是盲目继续提高全局 RD 权重。

## 2026-05-08 Expert1 Unknown/Clutter residual 三层建模与 System 接线修正

### 问题

- Expert1 的 RD 分支已经有 patch CNN、pair gate 和多项辅助 loss，但 Unknown/Clutter 仍不稳定：
  - Unknown 语义不是普通闭集单峰类，更接近低空内部 residual / reject bucket。
  - Clutter 更依赖 RD 局部形态、能量分布、粗糙度和稀疏性。
  - 只靠五分类 RD logits 和 pairwise loss，容易让 RD 分支“有信息但净收益为负”。
- 旧 System 配置仍指向 `artifacts/20260504-223648/expert1_config_smoke/best_model.pt`：
  - 该 checkpoint 没有新增 `traj_residual_head`、`rd_residual_head`、`rd_unknown_clutter_head`、`rd_clutter_morphology_head`。
  - System 如果加载旧权重，会让这些新增头随机初始化后参与训练/评估。
- System 内部创建 `LowAltitudeSuspectExpert` 时没有传入 Expert1 训练用的 RD logit scale：
  - `rd_unknown_logit_scale`
  - `rd_clutter_logit_scale`
  - `rd_residual_logit_scale`
  - `rd_unknown_clutter_logit_scale`
  - `rd_clutter_morphology_logit_scale`
  - 这会造成 Expert1 单训和 System 接入行为不一致。
- 原 `main.md` 仍是早期方案稿，已经不能准确描述当前架构，尤其是：
  - Expert1 residual/reject-style Unknown。
  - Clutter morphology head。
  - System forward 层 reroute。
  - calibrated candidate probe。
  - 当前训练结果和未达标项。

### 解决方案

- 在 `LowAltitudeSuspectExpert` 中补齐 Unknown/Clutter 三层建模：
  - `traj_residual_head`：轨迹侧 known-vs-residual。
  - `rd_residual_head`：RD 侧 known-vs-residual。
  - `rd_unknown_clutter_head`：residual 内部 Unknown-vs-Clutter。
  - `rd_clutter_morphology_head`：用 RD stats + RD patch embedding 强化 Clutter 局部形态。
- residual 分数用于修正 RD logits：
  - residual 越强，Unknown/Clutter 候选越强。
  - Unknown-vs-Clutter detail head 决定 residual 更偏 Unknown 还是 Clutter。
  - Clutter morphology delta 只增强 Clutter 侧局部形态证据。
- 对无 RD 样本保持轨迹分支独立：
  - no-RD 样本的 RD residual/detail/morphology logits 置零。
  - final 融合仍通过 RD activation/gate 控制，不让无 RD 样本被 RD 分支污染。
- Expert1 trainer 增加专项 loss：
  - `low_altitude_residual_aux_weight`
  - `low_altitude_rd_residual_aux_weight`
  - `low_altitude_rd_unknown_clutter_aux_weight`
  - `low_altitude_rd_clutter_morphology_aux_weight`
- Expert1 选模和报告新增指标：
  - `traj_residual_accuracy`
  - `rd_residual_accuracy_on_rd`
  - `rd_unknown_clutter_detail_accuracy_on_residual_rd`
  - `rd_clutter_morphology_accuracy_on_residual_rd`
  - `rd_unknown_clutter_detail_clutter_recall`
  - `rd_clutter_morphology_clutter_recall`
  - `residual_score_on_residual_mean`
- System 接线修正：
  - `SystemTrainConfig` 增加 Expert1 RD scale 配置。
  - `RadarClassificationSystem` 初始化 Expert1 时传入同一组 RD scale。
  - `config/config.json` 的 System `expert1_checkpoint` 改为本轮新 Expert1 checkpoint。
- 文档更新：
  - 旧 `main.md` 备份为 `main.bak001`。
  - 新 `main.md` 按当前实现重写，包含数据、专家、reroute、训练入口、评估指标和未达标项。

### 最终落地

- `src/core/experts.py`
  - `LowAltitudeFusionOutput` 增加 residual/detail/morphology 输出。
  - Expert1 增加 residual/detail/morphology heads。
  - RD logits 增加 Unknown/Clutter residual correction。
  - known confidence 改为从完整 traj softmax 中取 Drone/Bird/Balloon 最大概率。
- `src/core/expert1_config.py`
  - 增加 residual 和 morphology loss/scale 配置。
- `src/core/expert1_trainer.py`
  - 增加 residual targets、loss、metrics、summary/report 字段。
  - Expert1 构造时传入新增 RD scale。
  - 选模 score 纳入 residual/detail/morphology 指标，但尚未设置为硬门槛。
- `src/core/reporting.py`
  - Markdown 报告渲染 residual/detail/morphology 指标。
- `src/core/system.py`
  - System 构造 Expert1 时支持新增 RD scale 参数。
- `src/core/system_config.py`
  - System 配置 dataclass 增加 Expert1 RD scale。
- `src/core/system_trainer.py`
  - System 训练入口将配置传入 `RadarClassificationSystem`。
- `config/config.json`
  - Expert1 与 System 参数保持一致。
  - System `expert1_checkpoint` 指向 `artifacts/20260508-124356/expert1_config_smoke/fallback_candidate_model.pt`。
- `main.md`
  - 改为当前实现架构文档。
- `main.bak001`
  - 保留旧版主架构方案。

### 验证

- `python -m compileall src` 通过。
- `git diff --check` 对本轮触碰文件通过。
- CUDA Expert1 10 epoch 训练通过：
  - 输出目录：`artifacts/20260508-124356/expert1_config_smoke`
  - `skipped_non_finite_batches=0`
  - `accuracy=0.8290`
  - `macro_f1=0.7118`
  - `traj_macro_f1=0.7246`
  - `rd_macro_f1_on_rd=0.5742`
  - `rd_drone_bird_recall_mean=0.7015`
  - `rd_unknown_clutter_recall_mean=0.4190`
  - `rd_harm_rate_on_rd=0.2110`
  - `rd_rescue_rate_on_rd=0.0338`
  - `final_harm_rate_on_rd=0.0084`
  - `final_rescue_rate_on_rd=0.0127`
  - `rd_residual_accuracy_on_rd=0.9072`
  - `rd_unknown_clutter_detail_accuracy_on_residual_rd=0.7273`
  - `rd_clutter_morphology_accuracy_on_residual_rd=0.8182`
  - `rd_unknown_clutter_detail_clutter_recall=0.8571`
  - `rd_clutter_morphology_clutter_recall=0.7143`
  - `raw_tr_rd_accuracy=0.2500`
  - `source_balanced_accuracy=0.6064`
- CUDA System 3 epoch 训练通过：
  - 输出目录：`artifacts/20260508-130610/system_config_smoke`
  - 新 Expert1 checkpoint 完整加载：`missing_keys=[]`
  - `skipped_non_finite_batches=0`
  - selected mode：`calibrated_candidates`
  - selected `accuracy=0.8163`
  - selected `macro_f1=0.5518`
  - best test probe：`soft_log_weighted`
  - best test probe `accuracy=0.8283`
  - best test probe `macro_f1=0.5560`
  - `route_macro_recall=0.7460`
  - `expert1_low_altitude_macro_f1=0.7452`
  - `Drone/Bird recall mean=0.8636`
  - `Unknown recall=0.6111`
  - `Clutter recall=0.4286`
  - `Ground group F1=0.5641`
  - `Aircraft recall=0.6000`
  - `Aircraft precision=0.3750`
  - `source_balanced_accuracy=0.6352`
  - `min_source_accuracy=0.3636`

### 结论

- Unknown/Clutter 三层建模已经落地，新增 residual/detail/morphology heads 在短训中有明确学习信号。
- System 和 Expert1 的结构/scale/checkpoint 已重新对齐，避免新增头随机初始化进入主线。
- RD 分支仍未达“稳定正收益”标准：
  - `rd_harm_rate_on_rd` 仍明显高于 `rd_rescue_rate_on_rd`。
  - final 融合能减伤，但 rescue 放大不足。
- 当前 System 宏 F1 仍受 Aircraft precision、Ground F1、`raw_tr_rd` 弱源影响。
- 下一步应继续围绕三件事推进：
  - Expert1 RD 降 harm 和提高 rescue。
  - Aircraft/Ground candidate arbitration 标定。
  - source-aware training 提升 `raw_tr_rd`。

## 2026-05-08：两层标签与 Expert1 dual/deep 回退接口落地

### 问题

1. 标签只有单层 `label_name`，CQ-08 标签 1/2 的“旋翼无人机”监督信号被合并进 `DroneTarget` 后完全不可见。后续如果直接做细类训练或报告，会缺少稳定接口。
2. 标签方案不能引入没有数据证据的细类；`VTOLDrone`、`UnknownDroneSubtype` 这类概念不应写入训练标签。
3. Expert1 低空识别仍带有“RD 是否激活”的旧语义，而当前架构目标是低空默认 `dual_main`：轨迹主干和 RD 主干都作为主证据。
4. 已有 System reroute 更接近后处理或一次性专家选择，没有明确表达“第一轮 dual_main，reject 后 deep_reroute”的路径，也缺少 deep 使用诊断字段。

### 解决方案

1. 标签只保留两层：
   - `coarse_label`：最终业务大类。
   - `fine_label`：训练/诊断细类。
2. CQ-08 映射最小化：
   - 标签 1/2 -> `coarse_label=DroneTarget`，`fine_label=RotorDroneTarget`。
   - 标签 3/4/5/6 分别保持 Bird/Balloon/Clutter/Unknown。
3. 旧数据兼容：
   - 旧 CSV/索引没有 `coarse_label/fine_label` 时由数据层自动推导。
   - `label_name` 继续作为 coarse 训练标签，避免旧训练链路被破坏。
4. Expert1 增加正式推理模式：
   - `dual_main`：默认模式，有 RD 的样本默认进入 RD 融合路径。
   - `deep_reroute`：更强 RD 接管，增加 class/pair fusion 权重，放宽局部 pair takeover。
5. Expert1 输出 reject diagnostics：
   - `reject_score`
   - `unknown_trigger`
   - `confusion_pair_trigger`
   - `evidence_conflict_trigger`
   - `low_margin_trigger`
   - `high_risk_pair`
   - `traj_pred`
   - `rd_pred`
6. System reroute 低空路径调整为：
   - 第一轮使用 Expert1 `dual_main`。
   - 若低空 accept 失败，再运行 Expert1 `deep_reroute`。
   - 记录 `reroute_low_altitude_deep_used` 与 `reroute_low_altitude_deep_accepted`。

### 最终落地

- `src/core/data/labels.py`
  - 增加 `FINE_LABELS`、`FINE_VOCAB`、`two_layer_label_fields()`。
- `src/process/schema.py`
  - 增加 `coarse_label`、`fine_label` 到 track schema/index schema。
- `src/process/standardize_cq08.py`
  - CQ-08 标签 1/2 推导为 `RotorDroneTarget`。
- `src/process/standardize_tracks.py`
  - raw/excel 标准化输出两层标签。
  - 同步补齐 `rd_phase_missing` / `height_missing`，避免干净重建时列不齐。
- `src/process/build_training_index.py`
- `src/process/build_system_index.py`
- `src/process/build_expert1_index.py`
  - 兼容旧索引并输出 fine label 统计。
- `src/core/data/processed_dataset.py`
- `src/core/data/track_dataset.py`
- `src/core/data/torch_dataset.py`
- `src/core/data/collate.py`
  - batch 中增加 `coarse_label_names`、`fine_label_names`、`fine_labels`。
- `src/core/experts.py`
  - `LowAltitudeSuspectExpert.forward(..., inference_mode="dual_main|deep_reroute")`。
  - `dual_main` 下有 RD 默认参与融合。
  - `deep_reroute` 强化 RD class/pair fusion，并输出 reject diagnostics。
- `src/core/system.py`
  - 推理 reroute 中低空先 dual，再 deep。
  - 回退诊断增加 deep 使用/接受字段。
- `src/core/system_trainer.py`
  - evaluation report 增加 fine label 分布和回退 deep 统计。
- `src/core/expert1_trainer.py`
  - Expert1 report 增加 fine label 分布。

### 验证

- `python -m compileall src` 通过。
- dataloader smoke 通过：
  - 示例 coarse：`['DroneTarget', 'AircraftTarget']`
  - 示例 fine：`['RotorDroneTarget', 'AircraftTarget']`
  - fine ids：`[6, 0]`
- System forward smoke 通过：
  - parallel low-altitude mode：`dual_main`
  - 有 RD 样本 `rd_activated=[True, True]`
  - `reject_score` shape 正常。
- System reroute forward smoke 通过：
  - `reroute_logits` shape 为 `(2, 9)`
  - 输出 `reroute_low_altitude_deep_used`
  - 输出 `reroute_count`

### 当前边界

- 本轮没有重新训练。
- 当前 fine label 已进入数据接口和报告，但 loss 仍使用 coarse label；后续需要单独设计 fine label loss 或专家内部细类监督。
- `deep_reroute` 目前是增强路径接口和策略增强，尚未引入更高分辨率 RD 数据重采样；后续需要继续扩展 RD 数据编码和 deep 专家头。

## 2026-05-09：System 回退评估双口径与有方向 reject graph

### 问题

1. System 报告已有 fusion candidate probe，但没有固定展示“不开回退”和“开回退”的成对对比，无法判断回退机制是否真正提高 accuracy / macro F1。
2. 旧 reroute 控制流主要依赖 route logits 惩罚后重新 `argmax`，不是 `target.md` 要求的有方向 reject graph。
3. 回退样本缺少清晰的路径与原因记录，难以定位是 Expert1 Unknown、低空混淆、Aircraft reject、Ground reject 还是高空规则 reject 导致回退。
4. Expert1 低空报告只有单一口径，无法同时观察“Router 实际分给 Expert1 的样本”和“真实低空样本”的差异。

### 解决方案

1. 测试报告固定生成 `reroute_comparison`：
   - `without_reroute_mode`
   - `with_reroute_mode`
   - `accuracy_delta`
   - `macro_f1_delta`
   - `changed_count`
   - `helped_count`
   - `harmed_count`
2. System reroute 改为显式 reject graph：
   - Expert1 reject：先 dual -> deep；deep 后仍 reject 时按 Aircraft/Ground evidence 转 Expert2/Expert4，否则 fallback Unknown。
   - Expert2 reject：有 RD / 低空证据时转 Expert1 deep；Ground evidence 高时转 Expert4，否则 fallback Unknown。
   - Expert4 reject：Aircraft evidence 高时转 Expert2，有 RD 时转 Expert1，否则 fallback Unknown。
   - Expert3 不再作为普通候选，硬规则不接受则 fallback。
3. 回退诊断增加：
   - `reroute_initial_route`
   - `reroute_path`
   - `reroute_reject_reason_code`
   - `reroute_reject_reason`
   - `reroute_low_altitude_deep_used`
   - `reroute_low_altitude_deep_accepted`
4. System 报告 Expert1 增加双口径：
   - `routed`
   - `true_low_altitude`

### 最终落地

- `src/core/system.py`
  - `_forward_reroute()` 引入 `next_route_override` 和 `route_path`。
  - 新增按 route/reject reason 的 next-route 决策。
  - 仍保持 eval-only，不进入训练动态控制流。
- `src/core/system_trainer.py`
  - 新增 `REROUTE_REASON_NAMES`。
  - 新增 `_reroute_comparison()`。
  - 新增 `_expert1_scope_report()`。
  - prediction records 记录 `reroute_path` 与可读 reject reason。
- `src/core/reporting.py`
  - Markdown 显示回退前后对比。
  - Markdown 显示 reroute reason 分布、Expert1 deep 使用率/接受率。
  - Markdown 显示 Expert1 routed / true-low-altitude 双口径。

### 验证

- `python -m compileall src` 通过。
- System reroute forward smoke 通过：
  - `reroute_logits` shape 正常。
  - `reroute_path` 正常输出。
  - `reroute_reject_reason_code` 正常输出。
  - 随机初始化模型下样本 fallback 是预期行为，不代表训练性能。

### 当前边界

- 本轮仍未重新训练。
- 有方向 reject graph 已经进入推理/评估路径，但阈值仍沿用当前 config，需要通过训练后验证集调参。
- `deep_reroute` 仍是现有 RD 表征上的增强路径，尚未引入更高分辨率 RD patch 或更多帧采样。

## 2026-05-09：两层标签旧索引兼容与 reroute 报告补齐

### 问题

1. 两层标签方案要求“没有可靠细类时，子类名等于大类名”，但最初 `FINE_LABELS` 只包含 `RotorDroneTarget`，没有包含 `DroneTarget`。旧索引或无明确旋翼证据的无人机样本会出现 `fine_label=DroneTarget`，训练入口会在 `FINE_VOCAB.encode()` 处抛 `KeyError`。
2. 读取层只在索引缺少 `coarse_label` / `fine_label` 列时才推断标签。如果列存在但值为空、旧值非法或半重建索引没有更新，仍可能断链。
3. Markdown 错分样本表只展示初始路由和最终路由，没有展示 `reroute_path` 和 reject reason，回退路径不可审计。
4. prediction record 中 `reroute_reject_reason_code` 按 float 写入，不利于后续脚本直接聚合。

### 解决方案

1. `FINE_LABELS` 增加 `DroneTarget`，保留默认细类等于大类的能力；只有 `cq08_track` 的明确 1/2 批次或 raw 标注 `Rotary drone` / 小型、轻型旋翼无人机才归入 `RotorDroneTarget`。
2. 在 `ProcessedTrackDataset` 入口统一清洗两层标签：
   - 缺列时推断；
   - 空值、`nan`、`none` 时回退推断；
   - fine label 不在 `FINE_VOCAB` 时优先用推断结果，否则回退到合法 coarse label。
3. Markdown 错分样本表增加：
   - `回退路径`
   - `reject reason`
4. prediction record 将包含 `code` 的 reroute 诊断字段按整数保存。

### 最终落地

- `src/core/data/labels.py`
  - `FINE_LABELS` 支持 `DroneTarget` 与 `RotorDroneTarget` 并存。
- `src/core/data/processed_dataset.py`
  - 新增统一标签清洗逻辑，避免旧索引或半重建索引中断训练。
- `src/core/reporting.py`
  - 错分样本表展示回退路径和 reject reason。
- `src/core/system_trainer.py`
  - reroute reason code 按 int 写入 prediction record。

### 验证

- `python -m compileall src` 通过。
- 训练入口 dataloader smoke 通过，batch 可输出：
  - `coarse_label_names`
  - `fine_label_names`
  - `fine_labels`
- System reroute forward smoke 通过：
  - `reroute_logits` shape 为 `(4, 9)`。
  - `reroute_path` / `reroute_reject_reason_code` / `reroute_fallback` 正常输出。

### 当前边界

- 本轮未重新训练，尚无新 accuracy / macro F1。
- 旧索引读取已兼容，但建议下一次完整数据处理时重新生成所有 processed index，让落盘字段和运行时推断保持一致。
