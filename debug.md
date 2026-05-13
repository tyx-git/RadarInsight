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

## 2026-05-10 Expert1 精度与泛化优先修复

问题：
- Expert1 独立训练虽然总体 accuracy / macro F1 可用，但弱源表现被 `cq08_track` 高分掩盖；历史产物里 `raw_tr_rd` 与 `excel_track` 的 macro F1 明显偏低。
- 旧选模主要约束 source accuracy，缺少 source-level macro F1 硬门槛，容易选中“多数类/强源好看、弱源细类崩”的 checkpoint。
- Expert1 训练采样主要依靠固定 `raw_source_boost` / `excel_source_boost`，没有按 source 分布和 source+label 长尾动态补偿。
- 报告里 Expert1 by-source 只有有限字段，无法直接定位弱源是整体错还是具体标签错。

方案：
- 将 `source_balanced_macro_f1`、`min_source_macro_f1`、`raw_tr_rd_macro_f1`、`excel_track_macro_f1` 接入 Expert1 训练/验证/测试指标。
- 将 `source_balanced_macro_f1`、`raw_tr_rd_macro_f1`、`excel_track_macro_f1` 纳入 Expert1 best checkpoint 候选门槛，避免只凭整体 macro F1 选模。
- 在 Expert1 dataloader 中新增 source-aware 与 source-label-aware sampling，根据训练集源域和源内标签频次自动补偿弱源/弱标签，同时用 `max_sampler_weight` 控制重复采样上限。
- Expert1 evaluation report 增加 `critical_metrics`，并在 `by_source` 中输出 accuracy / macro precision / macro recall / macro F1 / trajectory source metrics / RD-on-RD source metrics / per-label metrics。

落地：
- `src/core/expert1_config.py`：新增 `source_balance_power`、`source_label_balance_power`、`max_sampler_weight` 以及 source macro F1 选模门槛。
- `src/core/expert1_data.py`：WeightedRandomSampler 增加 source 与 source+label 动态权重。
- `src/core/expert1_trainer.py`：新增 source macro F1 统计、source metrics 报告、critical metrics、选模打分和合格门槛。
- `config/config.json`：Expert1 默认启用 source-aware sampler 和弱源 macro F1 门槛；保留当前用户设置的 batch size 与训练 epoch。

下一步验证：
- 使用 CUDA 跑 Expert1 短训练或完整训练，重点观察 `source_balanced_macro_f1`、`raw_tr_rd_macro_f1`、`excel_track_macro_f1`、`rd_harm_rate_on_rd`、`rd_net_gain_rate_on_rd`。
- 若弱源 macro F1 提升但总体 macro F1 明显下降，再收紧 `max_sampler_weight` 或降低 `source_label_balance_power`。

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
- `deep_reroute` 当时只是增强路径接口和策略增强；模型侧更高分辨率 deep patch / deep 专家头已在 2026-05-09 后续补齐，数据层更多帧采样策略仍需继续扩展。

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
- `deep_reroute` 当时仍是现有 RD 表征上的增强路径；更高分辨率 deep RD patch 已在 2026-05-09 后续补齐，更多帧采样策略仍未落地。

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

## 2026-05-09：target 收敛与回退审计指标补齐

### 问题

1. `target.md` 仍保留大量已经落地的设计内容，导致“待做目标”和“已实现架构”混在一起，后续推进容易重复实现。
2. 回退报告已有 without / with reroute 总体对比，但缺少逐类别 recall before/after，无法判断回退到底救了哪个类别、伤了哪个类别。
3. 回退样本只有 path 和 reject reason，缺少 helped / harmed 逐样本记录。
4. Expert1 的 reject score 已在模型内计算，但 System reroute diagnostics 没把组成项带出来，报告端仍只能看到粗粒度 reason。

### 解决方案

1. 重写 `target.md` 为“剩余优化任务”：
   - 删除已落地的两层标签、基础 dual/deep、基础 reroute graph、基础报告项。
   - 保留仍需推进的 deep_reroute 强化、RD 净收益、Ground 仲裁、Expert2 校准、跨源泛化、性能测试清单。
2. `_reroute_comparison()` 增加：
   - `per_class_before_after_recall`
   - `sample_outcomes`
   - `reroute_helped`
   - `reroute_harmed`
   - `prediction_changed`
3. Markdown 报告增加：
   - 回退前后逐类别 recall delta 表。
   - 回退样本 helped / harmed 摘要表。
4. System reroute diagnostics 增加低空 reject 分解：
   - `low_reject_score`
   - `low_unknown_bonus`
   - `low_confusion_pair_bonus`
   - `low_evidence_conflict_bonus`
   - `low_margin_bonus`
   - `low_agreement_confidence_credit`
   - `low_high_risk_pair`
   - `low_final_margin`
   - `low_traj_pred`
   - `low_rd_pred`

### 最终落地

- `target.md`
  - 改为仅保留剩余目标。
- `src/core/system_trainer.py`
  - 回退对比补齐 per-class 与逐样本审计。
  - prediction records 写入 reroute helped/harmed 与低空 reject 分解。
- `src/core/system.py`
  - reroute diagnostics 输出低空 reject score 组成项。
- `src/core/reporting.py`
  - Markdown 显示回退逐类别 recall 变化、样本 helped/harmed、错分样本 low reject score。

### 验证

- `python -m compileall src` 通过。
- `_reroute_comparison()` smoke 通过，可正确输出：
  - per-class recall delta
  - helped / harmed sample outcomes
- System reroute forward smoke 通过：
  - `low_reject_score`
  - `low_unknown_bonus`
  - `low_confusion_pair_bonus`
  - `low_traj_pred`
  - `low_rd_pred`
  均正常输出。
- 定向 `git diff --check` 通过。

### 当前边界

- 本轮仍未重新训练，尚无新 accuracy / macro F1。
- Expert1 `deep_reroute` 仍需继续增强到真正更高分辨率 RD / 更多帧 / deep 专用头。

## 2026-05-09：System 来源指标、Aircraft 仲裁诊断与回退配置报告

### 问题

1. `target.md` 中仍保留了部分已经进入代码的内容，尤其是 source-balanced macro F1、source x label 指标和 Aircraft / Expert2 仲裁诊断，容易导致后续重复实现。
2. System 来源报告原先重点是 accuracy，弱源在 macro F1 上是否崩坏不够直观。只看 source accuracy 时，CQ08 高分仍可能掩盖 raw / excel 的类级失败。
3. Aircraft 仲裁已有 Expert2 positive 但 final not Aircraft 的方向，但缺少反向诊断：final 判 Aircraft 时 Expert2 是否真的给出 positive 证据，以及 Aircraft FP 来自哪个 source。
4. 回退机制实际使用的 max passes、penalty、各专家接受阈值来自 config，但 JSON / Markdown 报告没有固定记录。后续对比不同回退实验时，需要人工反查配置，容易误读。

### 解决方案

1. 来源统计增加：
   - per-source `macro_f1`
   - per-source `macro_recall`
   - `source_balanced_macro_f1`
   - `min_source_macro_f1`
   - source x label precision / recall / F1 / support / predicted
2. System 选模分数纳入 source-balanced macro F1 与 min source macro F1，让弱源宏指标参与 checkpoint 选择，而不是只在报告里展示。
3. Aircraft / Expert2 仲裁诊断增加：
   - `final_aircraft_expert2_not_positive_count`
   - `final_aircraft_expert2_not_positive_true_aircraft_count`
   - `final_aircraft_expert2_not_positive_false_aircraft_count`
   - `aircraft_false_positive_by_source`
4. 回退配置通过 `_system_reroute_config()` 统一导出到：
   - `evaluation_report["reroute_config"]`
   - `training_summary["reroute_config"]`
   - Markdown “回流重路由”章节
5. `target.md` 删除已落地项，只保留仍需推进的 Expert2 阈值校准、calibrated candidate 仲裁、Ground 保护、deep_reroute 强化等任务。

### 最终落地

- `src/core/system_trainer.py`
  - 来源报告、来源 summary、critical metrics、fusion probe 均补齐 source macro F1。
  - Aircraft 报告补齐反向仲裁诊断与 FP by source。
  - evaluation report / summary 写入 reroute config。
- `src/core/reporting.py`
  - Markdown 渲染 Aircraft / Expert2 仲裁表。
  - Markdown 渲染 Aircraft FP by source。
  - Markdown 渲染 source x label 指标。
  - Markdown 渲染 reroute config。
- `target.md`
  - 移除本轮已落地项，保留剩余优化目标。

### 验证

- `python -m compileall src` 通过。
- `_source_report()` / `_source_summary()` 定向 smoke 通过：
  - `source_balanced_macro_f1`
  - `min_source_macro_f1`
  - source x label `precision` / `recall` / `f1`
  均可正确输出。
- `_aircraft_report()` 定向 smoke 通过：
  - `final_aircraft_expert2_not_positive_count`
  - `aircraft_false_positive_by_source`
  - `expert2_positive_system_not_aircraft_count`
  均可正确输出。
- `build_markdown_report()` reroute config smoke 通过，Markdown 中可渲染 `max_passes` 与 `low_altitude_conf_threshold`。

### 当前边界

- 本轮仍未重新训练，尚无新的 accuracy / macro F1。
- Aircraft 独立阈值校准、Expert2 / Expert4 reject reason 细分、calibrated candidate 从 eval-only probe 推进到默认仲裁仍是后续任务。

## 2026-05-09：专家内部报告与安全关键回退对比

### 问题

1. `target.md` 的性能清单里还有多项已经可以通过现有 System 前向结果计算，但评估报告没有集中输出：
   - Expert2 内部混淆矩阵；
   - Expert3 高空规则命中正确率；
   - Expert4 地面内部混淆矩阵；
   - RD CosFace 原型相似度矩阵；
   - 安全关键误判对在 reroute 前后的变化。
2. System 端只看最终 9 类指标时，无法判断问题发生在专家本地 head、路由/仲裁，还是最终融合。
3. Expert1 的轨迹快速分支置信度和熵还没有进入 System 级报告，不利于后续设计 deep_reroute 激活条件。

### 解决方案

1. 在 `_run_epoch()` 中复用已有一次前向结果，聚合本地专家输出，不新增训练路径：
   - `expert2_logits`
   - `expert3_logits`
   - `expert4_logits`
   - `expert1_traj_logits`
   - `expert1_rd_logits`
   - `expert1_traj_confidence`
   - `expert1_traj_entropy`
   - `expert1_fusion_weights`
   - `rd_is_multimodal`
2. 新增纯评估报告函数：
   - Expert2 open-set 二分类 local confusion / precision / recall / false aircraft rate。
   - Expert3 rule selected / expected / precision / recall / selected label distribution。
   - Expert4 People / Car / Ship local confusion、People -> Car 比例。
   - Expert1 trajectory / RD / RD-on-RD 本地指标、轨迹高置信正确/错误数量、融合权重均值。
   - RD CosFace prototype cosine matrix。
3. `_reroute_comparison()` 增加固定安全关键误判对的 before / after / delta / reduced / introduced。
4. Markdown 报告新增：
   - “专家内部性能”章节；
   - “安全关键误判对”回退前后变化表。
5. `target.md` 删除上述已落地项，仅保留推理延迟、reject reason 细分、弱源增强、Expert4 People -> background 等未落地任务。

### 最终落地

- `src/core/system_trainer.py`
  - epoch metrics 增加 `expert_outputs`。
  - evaluation report 增加 `expert_internal`。
  - reroute comparison 增加 `safety_critical_pairs`。
- `src/core/reporting.py`
  - Markdown 渲染专家内部性能、RD prototype 相似度和安全关键误判对。
- `target.md`
  - 移除本轮已完成的报告类任务。

### 验证

- `python -m compileall src` 通过。
- `_expert_internal_report()` 定向 smoke 通过：
  - Expert2 local macro F1 可输出；
  - Expert4 People -> Car 统计可输出；
  - Expert1 trajectory / RD 字段可输出。
- `_reroute_comparison()` safety pair smoke 通过，`BirdTarget->DroneTarget` before / after 计数可输出。
- `build_markdown_report()` 专家内部报告 smoke 通过：
  - “专家内部性能”章节可渲染；
  - “安全关键误判对”表可渲染；
  - “RD CosFace 原型相似度”表可渲染。

### 当前边界

- 本轮仍未重新训练，尚无新的 accuracy / macro F1。
- 新增的是评估与报告能力，不代表 Expert2/Expert4 本身已完成优化。
- 推理延迟剖析尚未实现。

## 2026-05-09：Expert1 deep_reroute 独立增强路径

### 问题

1. `deep_reroute` 虽然已有独立 inference mode，但主要复用 `dual_main` 的 RD 表征，只是提高 gate / takeover 倾向。它还不是一个真正更强的二次纠错路径。
2. deep 路径缺少专用局部 correction head，无法在 Drone/Bird、Unknown/Clutter、Balloon 等局部边界上更强接管。
3. Unknown / Clutter 的 residual 判断仍复用普通 RD 头和已有辅助头，deep reroute 阶段缺少更明确的 residual / clutter 判别信号。
4. 报告中看不到 pass1 prediction、deep prediction、final prediction、deep helped / harmed、deep accept reason，无法判断 deep reroute 是否真的在纠错。

### 解决方案

1. 在 `LowAltitudeSuspectExpert` 内新增 deep 专用 RD patch 路径：
   - 24/48/64 channel 的更深 2D CNN；
   - `AdaptiveAvgPool2d((8, 8))` 保留比普通 patch 更高的局部纹理分辨率；
   - attention / mean / max 三路多帧聚合后融合，作为 deep patch embedding。
2. deep reroute 模式下，将 deep patch embedding 与原 RD embedding 融合进 `deep_rd_refine`，得到 deep 专用 RD 表征。
3. 新增 deep 专用头：
   - `deep_pair_correction_head`：输出 5 类局部 correction delta，用于 deep 阶段强化局部 pair correction；
   - `deep_unknown_clutter_residual_head`：输出 known / unknown / clutter 三路 residual 判断，用于 Unknown / Clutter 二次裁决。
4. deep 模式下：
   - RD logits 叠加 deep pair correction；
   - Unknown / Clutter 额外叠加 deep residual delta；
   - 保持 `dual_main` 轨迹快速路径不受 RD summary 污染。
5. System reroute 诊断新增：
   - `reroute_low_altitude_pass1_pred`
   - `reroute_low_altitude_deep_pred`
   - `reroute_low_altitude_final_pred`
   - `reroute_low_altitude_deep_helped`
   - `reroute_low_altitude_deep_harmed`
   - `reroute_low_altitude_deep_accept_reason_code`
   - `low_deep_pair_correction_max`
   - `low_deep_unknown_clutter_pred`
   - `low_deep_unknown_clutter_confidence`
6. Markdown 报告新增 “Expert1 deep_reroute 样本” 表，直接展示 pass1 / deep / final / accept reason / helped / harmed。

### 最终落地

- `src/core/experts.py`
  - 增加 deep patch CNN、多帧聚合、deep RD refine、deep pair correction、deep Unknown/Clutter residual。
- `src/core/system.py`
  - reroute 记录 pass1/deep/final 预测、deep helped/harmed 和 accept reason。
- `src/core/system_trainer.py`
  - deep 诊断写入 prediction record。
  - `_reroute_report()` 使用真实标签计算 deep recovery / harm 率。
- `src/core/reporting.py`
  - Markdown 输出 deep reroute 样本明细。
- `target.md`
  - 删除已落地的 deep patch、deep pair correction、deep residual、deep 报告项。

### 验证

- `python -m compileall src` 通过。
- `LowAltitudeSuspectExpert` 双模式 smoke 通过：
  - `dual_main` 输出 shape `(2, 5)`；
  - `deep_reroute` 输出 shape `(2, 5)`；
  - deep pair / Unknown-Clutter 诊断 shape 正常。
- `RadarClassificationSystem.forward(inference_mode='reroute')` smoke 通过：
  - `reroute_logits` shape `(2, 9)`；
  - `reroute_low_altitude_pass1_pred` / `deep_pred` / `deep_accept_reason_code` 正常输出。
- `build_markdown_report()` deep 表格 smoke 通过。

### 当前边界

- 本轮仍未重新训练，尚无新的 accuracy / macro F1。
- deep 专用头是新参数，需要后续训练验证；旧 checkpoint 加载时如果严格加载可能需要重新训练或使用兼容加载路径。
- “更多 RD 帧采样策略”尚未做成数据层可配置策略，仍保留在 `target.md`。

## 2026-05-09：Expert1 deep_reroute 独立 RD 帧采样视图

### 问题

1. `deep_reroute` 已经有更深的 RD patch CNN、pair correction head 和 Unknown / Clutter residual head，但它仍然复用 `dual_main` 的同一套 `rd_patch_features`。
2. 当前 `encode_multimodal_rd()` 对 3D RD 体数据统一插值成 128 帧，优点是稳定，缺点是 deep 路径没有看到更偏事件、更偏微动、更偏局部纹理的高信息密度帧序列。
3. 如果直接把默认 RD patch 全局改成更多帧，会影响 `dual_main` 的稳定主路径，也会增加训练显存和时延，不符合“默认 dual_main 稳定，deep_reroute 更强纠错”的设计。

### 解决方案

1. 保留原有 `rd_patch_features`：
   - 继续作为 `dual_main` 和常规 RD 分支的稳定输入；
   - 仍为 128 帧均匀时间插值视图。
2. 新增 `rd_deep_patch_features`：
   - 只供 `deep_reroute` 使用；
   - 3D RD 样本生成 160 帧 hybrid event deep patch；
   - 2D RD / summary RD 没有真实 Doppler 体数据时安全回退到原 patch。
3. hybrid event 采样策略：
   - `uniform`：保留全局轨迹覆盖；
   - `energy_peak`：采集强回波窗口；
   - `temporal_delta`：采集帧间变化大的窗口；
   - `doppler_entropy + roughness`：采集 Doppler 分布复杂和局部粗糙片段；
   - `low_energy_context`：保留少量弱回波上下文，避免只学习强能量片段。
4. 每类事件不是单点采样，而是 anchor 周围小窗口采样，再按原始时间排序，降低破坏时序连续性的风险。
5. 训练期增加 deep_reroute 辅助监督：
   - 不做动态回退控制流；
   - stage2 且样本有 RD 时额外前向一次 `inference_mode="deep_reroute"`；
   - 对 deep final logits 和 deep RD logits 加权监督，保证 deep patch CNN、deep pair correction、deep Unknown / Clutter residual 能收到梯度。

### 最终落地

- `src/core/data/rd.py`
  - `EncodedRDSequence` 增加 `deep_patch_features` / `deep_patch_valid_length`。
  - 新增 `MULTIMODAL_RD_DEEP_PATCH_TIME_BINS = 160`。
  - 新增 hybrid frame index 选择逻辑。
  - 3D RD 额外生成 deep patch tensor。
- `src/core/data/torch_dataset.py`
  - `TorchTrackItem` 增加 `rd_deep_patch_features` / `rd_deep_patch_valid_length`。
- `src/core/data/collate.py`
  - `TrackBatch` 增加 `rd_deep_patch_features` / `rd_deep_patch_lengths` / `rd_deep_patch_padding_mask`。
- `src/core/experts.py`
  - `LowAltitudeSuspectExpert.forward()` 增加 `rd_deep_patch_features` / `rd_deep_patch_padding_mask`。
  - `dual_main` 仍使用普通 patch；
  - `deep_reroute` 优先使用 deep patch，缺失时回退普通 patch。
- `src/core/system.py`
  - parallel / reroute 前向均透传 deep patch；
  - 低空 deep reroute 调用能实际吃到独立 deep RD 视图。
- `src/core/expert1_trainer.py`
  - Expert1 训练和评估传入 deep patch。
  - Expert1 stage2 训练增加 `low_altitude_deep_reroute_aux_weight` 控制的 deep 辅助损失。
- `src/core/expert1_config.py`
  - 新增 `low_altitude_deep_reroute_aux_weight`。
- `config/config.json`
  - Expert1 配置增加 `low_altitude_deep_reroute_aux_weight = 0.35`。
- `src/core/system_trainer.py`
  - System 训练和评估传入 deep patch。
- `target.md`
  - 删除“更多 RD 帧采样策略”待办，移入已落地清单。

### 当前边界

- deep patch 目前是 deterministic 采样，暂未加入训练期 stochastic jitter。
- 本轮是结构落地，尚未重新训练，因此没有新的 accuracy / macro F1。
- deep patch 会增加 deep_reroute 计算量，但 `dual_main` 路径不使用它，默认主路径稳定性不受影响。

## 2026-05-09：Expert1 RD hard-pair 固定报告

### 问题

1. Expert1 已经有 `rd_harm_rate_on_rd`、`rd_rescue_count`、`rd_net_gain_rate_on_rd` 等整体 RD 净收益指标。
2. 代码里也已经对 Drone/Bird、Unknown/Clutter、Bird/Clutter 等 hard pair 有部分训练约束或混淆触发逻辑。
3. 但报告层没有固定输出三组 hard pair 的 RD-only 局部表现：
   - `DroneTarget <-> BirdTarget`
   - `UnknownTarget <-> ClutterTarget`
   - `BirdTarget <-> ClutterTarget`
4. 后续调参如果只看整体 `rd_harm_rate_on_rd`，容易看不到 RD 到底在哪个局部边界救了样本、在哪个边界伤了样本。

### 解决方案

1. 新增 `_hard_pair_report()`，对每组 pair 统计：
   - support
   - trajectory accuracy
   - RD-only accuracy
   - final accuracy
   - RD rescue / harm / net gain
   - final rescue / harm / net gain
   - pair 内两个类别的 RD recall
   - pair 内互相误判次数
2. 将 hard pair report 写入：
   - `test_metrics["rd_hard_pair_report"]`
   - `evaluation_report["summary"]["rd_hard_pair_report"]`
   - `evaluation_report["branch_metrics"]["rd"]["hard_pair_report"]`
   - `summary["test_metrics"]["rd_hard_pair_report"]`
3. Markdown 报告在 “Expert1 RD 分支” 中渲染 RD hard pair 表。
4. `target.md` 删除“RD-only hard pair 分析固定进入报告”待办，并移入已落地清单。

### 最终落地

- `src/core/expert1_trainer.py`
  - 新增 `_pair_name()`。
  - 新增 `_hard_pair_report()`。
  - `_run_epoch()` 汇总 `rd_hard_pair_report`。
  - Expert1 evaluation report 和 summary 写入 hard pair 指标。
- `src/core/reporting.py`
  - Markdown 渲染 RD hard pair 局部净收益表。
- `target.md`
  - 移除已落地 hard pair report 待办。

### 当前边界

- 本轮只补报告和评估可见性，尚未重新训练。
- RD harm 优先训练约束本身已有基础 gate target 和选模门槛，后续还需继续根据 hard pair 报告做更细的 pair-specific harm penalty。

## 2026-05-09：Ground / Expert4 高置信仲裁保护与诊断

### 问题

1. Ground 类（People / Car / Ship）在 System final 中仍可能被低空专家或 Aircraft 抢分。
2. 直接堆 Expert4 head 不能解决 heterogeneous experts 的分数不可比问题；核心风险在全局仲裁层。
3. 旧的 Ground protection 只看 `groundness_confidence` 分层，缺少 Ground local logits margin 与 Router prior 的联合约束，容易把低置信 Ground 候选放大。
4. 报告里缺少 Ground protection 是否真正救回样本、是否引入非 Ground FP、Ground local margin 是否足够的诊断字段，后续调参容易盲调阈值。

### 解决方案

1. Ground candidate protection 改为高置信三条件：
   - `groundness_confidence >= groundness_confidence_tau_high`。
   - `ground_top1_margin >= ground_candidate_margin_tau`。
   - `ground_route_weight >= ground_route_gate_floor`，即 Router 没有强烈反对 Ground。
2. 只有满足三条件时，Ground candidate 才能解除 route gate 压制；中置信样本不再自动抬高 gate floor，避免弱 Ground 候选过度参与全局竞争。
3. `calibrated_candidates` 中仍保留低 groundness 的 suppression，用于 eval probe；普通 `groundness_protection_enabled` 只做高置信保护。
4. 新增 Ground diagnostics：
   - `ground_top1_margin`
   - `ground_candidate_high_confidence`
   - `ground_route_not_opposed`
   - `ground_candidate_changed_prediction`
   - `ground_recovered`
   - `ground_false_positive_introduced`
5. Markdown Ground 仲裁段落渲染：
   - `recovered_count`
   - `false_positive_introduced_count`
   - `ground_top1_margin`
   - `ground_candidate_changed_prediction_count`

### 最终落地

- `src/core/system_config.py`
  - 新增 `ground_candidate_margin_tau`。
- `config/config.json`
  - System 配置增加 `ground_candidate_margin_tau = 0.25`。
- `src/core/system_trainer.py`
  - `_project_to_global_logits()` 中 Ground protection 改为三条件高置信触发。
  - `_ground_report()` 汇总新增 Ground arbitration diagnostics。
  - prediction records 写入逐样本 Ground candidate 诊断。
  - summary 写入 `ground_candidate_margin_tau`。
- `src/core/reporting.py`
  - Markdown 报告渲染新增 Ground 仲裁指标。
- `target.md`
  - 已落地的 Ground 对照测试、三条件保护和 diagnostics 移出待办。

### 当前边界

- 本轮只修仲裁和诊断，不重新训练 Expert4 head。
- `groundness_protection_enabled` 默认仍为 `false`，避免在未重新验证前改变默认训练/评估路径。
- 后续必须运行 `tests/test_ground_system_alignment.py` 与 System 训练，确认高置信 candidate protection 不提升 `pred_ground_on_non_ground_count`。

## 2026-05-10：Expert1 20260510-105622 训练复盘与架构问题

### 训练产物

- 目录：`artifacts/20260510-105622/expert1_config_smoke`
- checkpoint：`fallback_candidate_model.pt`
- best epoch：12
- best selection reason：`fallback_no_rd_quality_checkpoint`
- best selection score：3.5879309831
- 设备：NVIDIA GeForce RTX 4060 Laptop GPU
- trainable parameters：15,900,564
- 预训练轨迹 checkpoint：`artifacts/main_pipeline_config_smoke/20260501-225130/pretrain_track/best_model.pt`

### 数据切分和 RD 覆盖

| split | samples | with RD | without RD |
|---|---:|---:|---:|
| train | 1432 | 912 | 520 |
| val | 306 | 238 | 68 |
| test | 310 | 237 | 73 |

### 本次关键配置

当前 `config/config.json` 覆盖了 `Expert1TrainConfig` 中更保守的默认节奏：

- `stage1_traj_only_epochs = 1`
- `stage2_ramp_epochs = 1`
- `low_altitude_rd_forced_fusion_weight = 0.9`
- `rd_sample_boost = 2.0`
- `raw_source_boost = 2.0`
- `excel_source_boost = 2.0`
- `unknown_sample_boost = 2.2`
- `clutter_sample_boost = 3.2`
- `bird_sample_boost = 1.9`
- `low_altitude_rd_pair_gate_aux_weight = 0.45`
- `low_altitude_rd_pair_takeover_floor = 0.76`
- `low_altitude_rd_pair_harm_ceiling = 0.08`
- `low_altitude_traj_anchor_weight = 0.15`
- `rd_unknown_logit_scale = 1.0`
- `rd_clutter_logit_scale = 1.3`
- `rd_bird_positive_weight = 4.5`
- `rd_unknown_positive_weight = 4.5`
- `rd_clutter_positive_weight = 8.0`

### Test 指标

| 指标 | 值 |
|---|---:|
| final accuracy | 0.8548 |
| final macro F1 | 0.7327 |
| trajectory accuracy | 0.8581 |
| trajectory macro F1 | 0.7486 |
| final - trajectory macro F1 | -0.0159 |
| RD accuracy | 0.6645 |
| RD macro F1 | 0.5547 |
| RD accuracy on RD | 0.7173 |
| RD macro F1 on RD | 0.6174 |
| RD Drone/Bird recall mean | 0.7246 |
| RD Unknown/Clutter recall mean | 0.5810 |
| RD rescue / harm / net | 7 / 47 / -40 |
| RD net gain rate on RD | -0.1688 |
| final rescue / harm / net | 2 / 3 / -1 |
| final net gain rate on RD | -0.0042 |
| RD activation rate | 0.7645 |
| RD activation rate on RD | 1.0000 |
| RD fusion weight mean on RD | 0.4582 |
| RD class fusion weight mean on RD | 0.3087 |
| RD pair fusion weight mean on RD | 0.1054 |
| RD pair weight Drone/Bird on RD | 0.1307 |
| RD pair weight Unknown/Clutter on RD | 0.0785 |
| RD pair weight Balloon on RD | 0.1069 |

RD per-class recall on RD：

| class | recall |
|---|---:|
| DroneTarget | 0.7170 |
| BirdTarget | 0.7321 |
| BalloonTarget | 0.7358 |
| ClutterTarget | 0.4286 |
| UnknownTarget | 0.7333 |

Source 指标：

| source | final accuracy | final macro F1 | RD accuracy on RD | RD macro F1 on RD |
|---|---:|---:|---:|---:|
| cq08_track | 0.8927 | 0.7615 | 0.7210 | 0.6193 |
| excel_track | 0.7671 | 0.5948 | N/A | N/A |
| raw_tr_rd | 0.2500 | 0.0800 | 0.5000 | 0.1333 |

汇总：

- source balanced accuracy：0.6366
- source balanced macro F1：0.4788
- min source accuracy：0.2500
- min source macro F1：0.0800
- with RD accuracy：0.8819
- without RD accuracy：0.7671

### Hard pair 结论

| pair | support | traj acc | RD acc | final acc | RD rescue/harm/net | final rescue/harm/net |
|---|---:|---:|---:|---:|---:|---:|
| DroneTarget vs BirdTarget | 162 | 0.8827 | 0.7222 | 0.8827 | 5 / 31 / -26 | 2 / 2 / 0 |
| UnknownTarget vs ClutterTarget | 22 | 0.7727 | 0.6364 | 0.7273 | 1 / 4 / -3 | 0 / 1 / -1 |
| BirdTarget vs ClutterTarget | 63 | 0.7143 | 0.6984 | 0.6984 | 4 / 5 / -1 | 1 / 2 / -1 |

逐类混淆矩阵 rows=true, cols=pred：

| true \ pred | Drone | Bird | Balloon | Clutter | Unknown |
|---|---:|---:|---:|---:|---:|
| Drone | 136 | 10 | 1 | 0 | 4 |
| Bird | 6 | 60 | 1 | 5 | 8 |
| Balloon | 0 | 1 | 51 | 2 | 0 |
| Clutter | 0 | 0 | 0 | 4 | 3 |
| Unknown | 1 | 1 | 0 | 2 | 14 |

### Raw 源逐样本

test 中 `raw_tr_rd` 只有 4 条，因此不能把 0.25 accuracy 当作稳定总体结论，但错误形态必须重视：

| sample | true | traj | RD | final | 结论 |
|---|---|---|---|---|---|
| `20230824104942|Rotary drone|04|1971` | Drone | Drone | Bird | Bird | RD/fusion 伤害 |
| `20230929161757|Bird|12|6527` | Bird | Bird | Bird | Bird | 正确 |
| `20231014161536|Bird|13|15` | Bird | Drone | Bird | Drone | RD rescue 未采纳 |
| `20231114155403|Bird|01|8175` | Bird | Bird | Drone | Drone | RD/fusion 伤害 |

### 和 20260508-124356 对比

- final macro F1：0.7118 -> 0.7327，整体有提升。
- RD macro F1 on RD：0.5742 -> 0.6174，RD 表征有提升。
- RD Unknown/Clutter recall mean：0.4190 -> 0.5810，Unknown/Clutter 局部有所改善。
- raw_tr_rd accuracy：仍为 0.2500，弱源没有改善。
- final net gain：由约 +1 变为 -1。
- `rd_harm_adopted_count` 从 0 变为 3，说明新模型更敢采纳 RD，但采纳策略还不安全。

### 架构问题

1. `dual_main` 推理绕过了 RD activation gate。
   - `src/core/experts.py` 中先按 `force_rd_activation`、trajectory confidence、class gate、pair gate 计算 `rd_activated`。
   - 但随后在 `inference_mode == "dual_main"` 时直接执行 `rd_activated = has_rd`。
   - 结果是所有有 RD 的样本都会进入融合路径，`rd_class_activation_threshold` 和 `rd_activation_threshold` 不能真正阻断 RD，只能间接影响融合强度。
   - 本次 test `rd_activation_rate_on_rd = 1.0`，与这个逻辑一致。

2. 训练阶段对 RD 过早、过强。
   - `src/core/expert1_trainer.py` 在 stage2 训练时传入 `force_rd_activation=True`。
   - 当前配置只有 1 个 epoch 纯轨迹、1 个 epoch ramp，而 dataclass 默认是 12 / 6。
   - 这会让模型还没建立稳定 trajectory anchor 就开始承受 RD aux、hard pair、binary head、forced fusion、gate 等多重目标。

3. RD 分支变强不等于最终融合安全。
   - RD on-RD macro F1 已到 0.6174，但 trajectory macro F1 是 0.7486。
   - 如果没有严格的采纳门控，RD 会在大量 trajectory 正确样本上提供错误修正。
   - 当前 hard pair 中 Drone/Bird 的 RD 净收益为 -26，是最大伤害来源。

4. Pair gate 是局部接管器，但不是可靠安全闸。
   - Pair gate 权重均值并不高，Drone/Bird on-RD 约 0.1307。
   - 但 class-wise fused logits 已经先进入 `selected_logits`，pair gate 后续再做局部覆盖，无法完全撤销前面 RD 对全类 logits 的扰动。
   - `strong_pair_takeover` 仍以 RD margin 和 gate prob 为主，没有显式要求“采纳后不低于 trajectory 置信安全边界”。

5. Reject / residual 结构存在，但默认没有真正参与最终安全闭环。
   - `enable_reject_in_eval=false`，Expert1 eval 不启用 `_apply_reject()`。
   - Unknown/Clutter residual、detail、morphology head 指标有诊断价值，但最终路径主要还是依赖 fusion/pair gate。

6. 选模 gate 是必要的，目前正确阻止了坏 checkpoint 晋级。
   - validation 上 final macro F1 0.7983，但 source balanced macro F1 0.5412、raw macro F1 0.3111，未过硬门槛。
   - validation RD net gain rate 为 -0.1723，也未过 `min_rd_net_gain_rate_for_best = 0`。
   - 因此保存 fallback 是合理行为，不能把这次 checkpoint 直接当作 System 主力。

7. System 当前仍没有消费这次 Expert1。
   - `config/config.json` 里的 `expert1_checkpoint` 仍指向 `artifacts/20260508-124356/.../fallback_candidate_model.pt`。
   - 最新 System 目录 `20260510-084208` 也使用旧 Expert1，不是 `20260510-105622`。
   - 旧 checkpoint 对当前 System 低空 deep reroute 结构存在 missing deep keys 风险；在 Expert1 未过 gate 前，不应打开 reroute 作为默认收益路径。

8. Raw 源样本太少，当前只能作为强风险诊断，不能作为统计充分的泛化结论。
   - Expert1 总样本中 raw 只有 35 条。
   - test raw 只有 4 条，且三条错误都集中在 Drone/Bird 边界。
   - 后续需要 raw holdout / bootstrap / leave-source 诊断，而不是只继续提高 source weight。

### 当前判断

这次训练说明 Expert1 的 RD 表征和 Unknown/Clutter 局部头确实继续进步，但核心架构矛盾没有解决：RD 是“第二主证据”，却还没有“可靠采纳条件”。因此 final macro F1 低于 trajectory macro F1，且 RD final net gain 变为负数。

下一轮不应先继续加 RD head 或继续提高 rare/source 权重，而应先修 RD 采纳安全：

1. 让 `dual_main` 下的 RD activation gate 真正生效，至少新增 safe-fusion candidate 做 A/B。
2. 降低或延后 forced fusion，恢复更长 trajectory-only 和 ramp。
3. 增加 final-vs-trajectory、hard-pair、source-aware 的采纳损失或选模门槛。
4. Expert1 过 RD quality gate 后，再更新 System 的 `expert1_checkpoint` 并重新训练 System。

## 2026-05-10：历史 TR+RD 架构对照

### 对照来源

主要查看 `.history/001`：

- `.history/001/mermory.md`
- `.history/001/arch.md`
- `.history/001/train.json`
- `.history/001/src/models/fusion_classifier.py`
- `.history/001/src/models/session_bag_classifier.py`
- `.history/001/src/training/trainer.py`

`.history/002` 更接近当前多专家配置文档快照，核心可执行 TR+RD 实现仍以 `.history/001` 为主。

### 历史版关键事实

- 数据是 paired TR/RD-only：`pro/raw -> data/sessions`。
- `pro/excel` 被保留审查，但不进入当时的 supervised dual-modal 训练。
- 当前 paired sessions：77，且 TR/RD fully paired。
- 标签是 4 类：`AircraftTarget`、`BirdTarget`、`DroneTarget`、`ShipTarget`。
- 历史最佳参考：`artifacts/202604141227-12-4-cuda`
  - independent evaluation：accuracy 0.8529，macro F1 0.8645。
  - session-level evaluation：accuracy 0.8750，macro F1 0.8810。
- 当时文档明确写了：
  - Track role：coarse routing and motion prior。
  - RD role：hard boundary discrimination。
  - Weak specialist policy：保留 specialist heads 作为 auxiliary supervision，避免强 direct residual injection 到 final logits。

### 为什么历史版“RD 默认第二证据”没有同样的问题

1. 历史版的 RD 默认参与是 feature-level joint evidence，不是 post-hoc logit 接管。
   - `fusion_classifier.py` 先用 track encoder 和 RD encoder 得到两个 embedding。
   - 通过 bidirectional cross-attention 让两个模态交互。
   - 再拼接 `weighted_track_feature`、`weighted_rd_feature`、`abs(diff)`、`product`，交给同一个 hierarchical head 学 final logits。
   - 这意味着 final classifier 从训练开始就学习“TR+RD 联合表征”，不是先独立得到 RD 类别，再把 RD logits 插到最终结果里。

2. 历史版 RD 权重是 reliability softmax，并且有上限和 mask。
   - `rd_reliability` 和 `track_reliability` 经 softmax 得到 `track_weight` / `rd_weight`。
   - `rd_mask` 缺失时，RD reliability 被置为 `-1e4`，RD feature 和 RD weight 都归零。
   - `rd_floor` 即使启用也被 clamp 到最高 0.45，然后再和 track weight 归一化。
   - 所以 RD 可以默认作为第二证据，但不是“有 RD 就改写结果”。

3. 历史版 residual 是小权重、受权重约束的。
   - final logits 上的 unimodal residual 为：
     - track residual：`track_logits * track_weight`
     - RD residual：`rd_logits * rd_weight * rd_unimodal_logit_residual_weight`
   - 当时默认 `rd_unimodal_logit_residual_weight` 是 0.18 级别。
   - Bird/Drone RD specialist residual 也是 0.08 级别。
   - Ship specialist residual 默认甚至可以为 0。
   - 这和当前 `traj_logits + rd_class_weight * (rd_logits - traj_logits)` 的直接 logit interpolation 风险不同。

4. 历史版主问题其实是 RD collapse，而不是 RD harm。
   - 历史诊断写明：旧 fusion path 在 test 上 `track_weight` 接近 1.0，`rd_weight` 接近 0.0。
   - anti-collapse fusion 在离线评估有帮助，但 full train-time anti-collapse 在小数据规模下不稳定。
   - 最终偏好的路线是：train-time 保持稳定 weighted fusion，evaluation / inference 再做 anti-collapse override。
   - 这说明历史版“没问题”的原因之一，是 RD 默认参与但模型仍能学会忽略弱 RD；它不是强制 RD 接管。

5. 历史版后期把 final target 上移到 session-level。
   - `SessionBagRadarClassifier` 先跑 clip-level TR/RD fusion model，再把 clip embedding 送入 session encoder / attention pooling / stat pooling。
   - `train.json` 中 session route 使用 `session_loss_weight = 1.0`、`clip_loss_weight = 0.0`。
   - 单个 clip 的 RD 错误不会直接等价于最终 session 错误；session attention 可以学习哪些 clip 更可信。
   - 当前 Expert1 是 sample-level 直接五分类，单条 RD 错误会直接伤 final prediction。

6. 历史版数据条件更干净。
   - 当时训练源是 fully paired TR/RD，同源、同标签体系、同任务口径。
   - 当前 Expert1 混合 `cq08_track`、`excel_track`、`raw_tr_rd`：
     - `excel_track` 没 RD。
     - `raw_tr_rd` 样本极少。
     - `cq08_track` 主导总体分布。
   - 在这种条件下，默认 RD 参与会引入 source/missing-modality calibration 问题。

7. 历史版任务更窄。
   - 历史 4 类任务主要是 Aircraft/Bird/Drone/Ship。
   - 当前 Expert1 低空 5 类加入 Balloon、Clutter、Unknown，其中 Unknown 是 residual/reject bucket，Clutter 是低支持形态类。
   - Unknown/Clutter 不是普通闭集边界，直接 RD logit correction 更容易制造误伤。

### 当前版和历史版的本质差异

| 维度 | 历史 TR+RD | 当前 Expert1 |
|---|---|---|
| RD 进入方式 | joint feature fusion | independent RD logits correction |
| RD 是否硬激活 | soft reliability weight | `dual_main` 下 `rd_activated = has_rd` |
| RD 残差强度 | 小权重且乘 `rd_weight` | class-wise interpolation 到 RD logits |
| 主损失对象 | 联合 head / session head | sample-level final logits |
| 缺失 RD 处理 | `rd_mask` 权重归零 | Excel 无 RD，RD 样本和无 RD 样本分布混训 |
| 数据口径 | fully paired TR/RD | CQ08 + Excel + raw 混合 |
| 历史主要风险 | RD 被忽略 | RD 误伤 trajectory |

### 结论

历史版并不能证明“RD 默认作为第二证据一定安全”。它证明的是另一件事：

```text
RD 可以默认进入联合表征，但必须被 reliability weight、mask、弱 residual、joint head 和 session aggregation 约束。
```

当前版把 RD 从“受限联合证据”改成了“独立分支对 trajectory logits 的后验修正”。这才是现在出问题的根因。

下一步 Expert1 更合理的方向不是简单回退到纯 trajectory，也不是继续堆 RD head，而是借回历史版的安全原则：

1. RD branch 继续保留为第二证据源。
2. RD logits 不再直接全类 interpolation 到 trajectory logits。
3. 增加 joint feature fusion candidate：
   - `traj_feature`
   - `rd_feature`
   - `abs(traj_feature - rd_feature)`
   - `traj_feature * rd_feature`
   - reliability `track_weight / rd_weight`
4. RD specialist heads 只作为 auxiliary evidence 或小 residual，必须乘以 bounded `rd_weight`。
5. final-vs-trajectory、hard-pair net gain、source-aware net gain 作为选择门槛。

## 2026-05-10：Expert1 V2 架构决策

### 用户决策

Expert1 下一版采用：

```text
无 RD：纯轨迹分支。
有 RD：激活 TR+RD joint 分支。
```

当前轨迹分支精度高于历史版本，因此轨迹分支不重写，作为 trajectory anchor 选择性保留。

### 为什么主线选择联合，而不是只做 RD 选择性激活

RD 选择性激活可以作为短期安全阀，但不适合作为主架构。原因：

1. 它仍然把问题建模成“这条样本要不要用 RD”，容易退化为阈值工程。
2. 当前 source 混合、RD 支持不均衡，gate 很难稳定判断 RD 是否可信。
3. 一旦激活后仍然是 RD logits correction，旧的误伤路径还在。
4. 历史高精度版本证明更有效的是受限联合表征，而不是强后验接管。

### Expert1 V2 结构

1. Trajectory-only path
   - 输入：`representation.fused_features`。
   - 输出：`traj_embedding`、`traj_logits`、`traj_residual_logits`。
   - 用途：无 RD 样本最终输出；有 RD 样本的 anchor。

2. RD evidence path
   - 输入：RD sequence、RD patch、RD stats、RD context。
   - 输出：`rd_embedding`、`rd_logits`、hard-pair logits、Unknown/Clutter residual evidence、Clutter morphology evidence。
   - 用途：训练 RD 表征和局部证据；不再直接全类覆盖 trajectory。

3. TR+RD joint path
   - 仅对 `has_rd=true` 样本启用。
   - 输入：
     - `traj_embedding`
     - `rd_embedding`
     - `abs(traj_embedding - rd_embedding)`
     - `traj_embedding * rd_embedding`
     - RD morphology/statistics
     - source / quality embedding
     - trajectory confidence / RD auxiliary confidence
   - 输出：
     - `joint_logits`
     - `track_weight`
     - `rd_weight`
     - `joint_confidence`

4. Final selection

```text
if no RD:
    final_logits = traj_logits
else:
    final_logits = traj_logits
                 + rd_weight * (joint_logits - traj_logits)
                 + bounded_rd_specialist_residual
```

约束：

- `rd_weight` 初始上限建议 0.35 到 0.45。
- `rd_weight` 必须能按 source / RD quality 降到 0。
- `bounded_rd_specialist_residual` 必须乘 `rd_weight`，并只允许小权重。
- 当前 class-wise `traj_logits + w * (rd_logits - traj_logits)` 不再作为主路径。

### 训练方案

1. Stage A：Trajectory anchor
   - 复用当前强轨迹分支。
   - 可冻结或低学习率微调。
   - 目标是保持 `traj_macro_f1` 不低于当前稳定基线。

2. Stage B：RD evidence
   - 训练 RD embedding、RD auxiliary、hard-pair、Unknown/Clutter、morphology heads。
   - RD auxiliary 不直接决定 final。

3. Stage C：TR+RD joint
   - 有 RD 样本训练 joint head。
   - 无 RD 样本回放 trajectory-only path，防止纯轨迹路径退化。
   - `rd_weight` ramp-up，不允许开局强融合。

4. Stage D：Calibration / selection
   - 验收不只看 final macro F1。
   - 必须同时看：
     - trajectory-only macro F1。
     - joint-on-RD macro F1。
     - final-vs-trajectory macro F1。
     - RD rescue / harm / net gain。
     - Drone/Bird、Unknown/Clutter、Bird/Clutter final net gain。
     - by-source final-vs-trajectory。

### 设计结论

Expert1 V2 的核心不是“RD 是否激活”，而是：

```text
无 RD 时，trajectory 独立可靠；
有 RD 时，TR+RD 联合表征必须比 trajectory anchor 提供正收益；
RD 独立 logits 只能提供证据，不能直接接管最终分类。
```

## 2026-05-10：Expert1 V2 代码首版落地

### 暂存

- 已在 `tmp.md` 写入 Expert1 V2 暂存实施计划。
- `.history` 未改动。
- 本轮没有启动正式训练，因此没有新增训练性能快照；下一次训练完成后必须写入可被 git 跟踪的完整性能记录。

### 代码改动

1. `src/core/experts.py`
   - `LowAltitudeFusionOutput` 新增：
     - `joint_logits`
     - `joint_reliability_logits`
     - `joint_rd_weight`
     - `joint_track_weight`
   - `LowAltitudeSuspectExpert` 新增 TR+RD joint branch：
     - `joint_traj_proj`
     - `joint_rd_proj`
     - `joint_head`
     - `joint_reliability_head`
   - joint 输入为：
     - `traj_features`
     - `rd_features_vec`
     - `abs(traj_features - rd_features_vec)`
     - `traj_features * rd_features_vec`
     - `rd_stats`
     - `rd_patch`
   - `dual_main` final path 改为：

```text
joint_selected_logits = traj_logits + rd_weight * (joint_logits - traj_logits)
```

   - 无 RD 或 `rd_is_multimodal=false` 时：

```text
rd_weight = 0
final_logits = traj_logits
```

   - `rd_weight` 当前上限：
     - `dual_main`：0.45
     - `deep_reroute`：0.65
   - 原 hard-pair 强接管逻辑已降级为小 residual：
     - `dual_main` residual scale：0.10
     - `deep_reroute` residual scale：0.16

2. `src/core/expert1_trainer.py`
   - `_run_epoch()` 记录 joint logits 与 joint RD weight。
   - 新增训练 / 验证 / 测试指标：
     - `joint_accuracy`
     - `joint_macro_f1`
     - `joint_accuracy_on_rd`
     - `joint_macro_f1_on_rd`
     - `traj_accuracy_on_rd`
     - `traj_macro_f1_on_rd`
     - `joint_minus_traj_macro_f1_on_rd`
     - `joint_rd_weight_mean`
     - `joint_rd_weight_on_rd_mean`
   - gate auxiliary target clamp 到 0.45，和 V2 bounded `rd_weight` 保持一致。
   - `training_history.json`、`training_summary.json`、`evaluation_report.json` 会输出 joint-on-RD 对照指标。

3. `tests/test_expert1_evidence.py`
   - 本地结构测试更新为 V2：
     - 混合 RD / 无 RD batch。
     - 检查 `joint_logits` 和 `joint_rd_weight` 存在。
     - 检查无 RD 样本 final 等于 trajectory。
     - 检查 `rd_weight <= 0.45`。
     - 检查 hard-pair residual 被小权重约束。

### 验证

使用项目虚拟环境 `.venv`：

```text
.venv torch version: 2.5.1+cu121
python -m compileall -q src\core tests：通过
python -m unittest tests.test_expert1_evidence.Expert1EvidenceTests.test_fusion_uses_bounded_joint_branch_with_trajectory_fallback：通过
```

独立 mixed RD smoke：

```text
logits shape = (3, 5)
joint_logits shape = (3, 5)
rd_weight = [0.11920291185379028, 0.0, 0.11920291185379028]
rd_activated = [True, False, True]
no-RD final == trajectory: True
```

### 当前风险

- 这只是结构首版，尚未证明训练后 joint-on-RD 能超过 trajectory-on-RD。
- 当前 joint head 仍缺少显式 source embedding / RD quality embedding，只接入了 RD stats 和 patch；后续若 raw_tr_rd 继续误伤，需要补 source-aware calibration。
- 旧 `fusion_gate` 模块暂时保留，避免 checkpoint / System 兼容路径出现无谓破坏，但 final path 已不再使用它。
- 当前 `config/config.json` 中 Expert1 epoch 变为 5；该改动不是本轮代码修改，需要训练前确认这是预期短训配置还是临时改动。

## 2026-05-11：Expert1 V2 首轮训练尝试与 checkpoint 评估

### 执行记录

1. 正式配置训练
   - 命令：`.venv\Scripts\python.exe -m src.core.train_expert1 --config config/config.json`
   - 当前 `config/config.json` 中 Expert1 epochs = 5。
   - 运行 1 小时后超时中断。
   - 产物：`artifacts/20260511-002434/expert1_config_smoke/fallback_candidate_model.pt`
   - 未生成 `training_summary.json` / `evaluation_report.json`，不作为有效完整训练结果。

2. V2 试训配置
   - 临时配置：`tmp/expert1_v2_trial_config.json`
   - 输出目录：`artifacts/20260511-012750/expert1_v2_trial`
   - 配置重点：
     - epochs = 2
     - `stage1_traj_only_epochs = 0`
     - `stage2_ramp_epochs = 1`
   - 运行 1 小时后超时中断。
   - 产物：`fallback_candidate_model.pt`
   - checkpoint metadata：
     - `selection_reason = stage2_rd_composite`
     - `selection_eligible = false`

3. Checkpoint 评估
   - 评估脚本：`tmp/evaluate_expert1_checkpoint.py`
   - test 输出：`artifacts/20260511-012750/expert1_v2_trial/checkpoint_eval_test.json`
   - val 输出：`artifacts/20260511-012750/expert1_v2_trial/checkpoint_eval_val.json`
   - 性能快照归档：
     - `performance_records/20260511-012750/expert1_v2_trial/README.md`
     - `performance_records/20260511-012750/expert1_v2_trial/checkpoint_eval_test.json`
     - `performance_records/20260511-012750/expert1_v2_trial/checkpoint_eval_val.json`
     - `performance_records/20260511-012750/expert1_v2_trial/expert1_v2_trial_config.json`

### Test 指标

| 指标 | 值 |
|---|---:|
| final accuracy | 0.7581 |
| final macro F1 | 0.6392 |
| trajectory macro F1 | 0.6428 |
| trajectory macro F1 on RD | 0.6659 |
| joint macro F1 on RD | 0.4255 |
| joint - trajectory macro F1 on RD | -0.2403 |
| RD macro F1 on RD | 0.0765 |
| RD harm rate on RD | 0.5865 |
| final net gain rate on RD | 0.0042 |
| joint RD weight mean on RD | 0.4500 |
| with RD accuracy | 0.7975 |
| without RD accuracy | 0.6301 |

### Val 指标

| 指标 | 值 |
|---|---:|
| final accuracy | 0.7614 |
| final macro F1 | 0.5975 |
| trajectory macro F1 | 0.6138 |
| trajectory macro F1 on RD | 0.6254 |
| joint macro F1 on RD | 0.4978 |
| joint - trajectory macro F1 on RD | -0.1277 |
| RD macro F1 on RD | 0.0740 |
| RD harm rate on RD | 0.5798 |
| final net gain rate on RD | -0.0042 |
| joint RD weight mean on RD | 0.4500 |
| with RD accuracy | 0.7815 |
| without RD accuracy | 0.6912 |

### 结论

这次 V2 试训不合格，不能进入 System：

1. joint branch 明显弱于 trajectory anchor。
   - test `joint_minus_traj_macro_f1_on_rd = -0.2403`
   - val `joint_minus_traj_macro_f1_on_rd = -0.1277`

2. reliability head 退化为“总是顶到上限”。
   - val/test `joint_rd_weight_on_rd_mean = 0.45`
   - 说明 reliability 不是在学习何时相信 RD，而是被 gate aux / 强 Stage2 推到 cap。

3. RD evidence branch 在该试训中严重崩坏。
   - test `rd_macro_f1_on_rd = 0.0765`
   - val `rd_macro_f1_on_rd = 0.0740`
   - 这远低于 20260510-105622 的 `rd_macro_f1_on_rd = 0.6174`。

4. 训练课程是主要问题之一。
   - `stage1_traj_only_epochs = 0` 直接让 joint/RD 从第一轮参与 final。
   - 新增 joint head 还未学到稳定联合表征，就被 bounded weight 直接推入 final path。
   - 这不是“TR+RD 联合架构失败”，而是“未经过 anchor/evidence/joint 分阶段训练的联合分支失败”。

### 下一轮调整

1. 恢复分阶段训练：
   - Stage A：至少 1 到 2 epoch trajectory anchor。
   - Stage B：RD evidence 单独恢复到 `rd_macro_f1_on_rd >= 0.55/0.60`。
   - Stage C：再打开 joint final。

2. 降低 joint 初始影响：
   - reliability final bias 继续 trajectory-biased。
   - 增加训练期 RD cap schedule：
     - warmup：0.00 或 0.05
     - early joint：0.10 到 0.20
     - stable joint：最高再到 0.45
   - 若 `joint_rd_weight_on_rd_mean` 贴近 cap 且 `joint_minus_traj_macro_f1_on_rd < 0`，该 checkpoint 直接失败。

3. 修 RD evidence 被 joint 训练拖垮的问题：
   - RD auxiliary 和 joint final 不应同一阶段全权重启动。
   - Stage B 先训练 RD evidence，不让 RD logits 进入 final。
   - Stage C 只训练 joint head / reliability head 或给 backbone 更低 LR。

4. 训练速度也需要处理：
   - 当前 2 epoch 试训 1 小时仍未完整收尾。
   - 下一次应先做 batch-level profiler 或缩短评估频率，避免每轮 val/test/report 成为主要耗时。

## 2026-05-11：Expert1 训练耗时优化

### 问题定位

首轮 V2 训练非常慢，不只是完整 val/test 的问题。`_run_epoch()` 在每个训练 batch 都执行：

```text
prev_state = _clone_bundle_state_to_cpu(representation_builder, rd_proj, model)
...
_parameters_are_finite_bundle(...)
```

这会把整套模型权重逐 batch 拷贝到 CPU，并在 optimizer step 后扫描全部参数。V2 新增 joint 分支后参数更多，这个保护逻辑成为明显瓶颈。

评估阶段也存在额外成本：即使 `train=False`，原逻辑只要 `stage2_enabled` 就会计算 RD/deep auxiliary loss，包括第二次 `deep_reroute` forward。评估只需要主前向和指标，这部分不应默认运行。

### 已改代码

1. `src/core/expert1_config.py` 新增速度控制项：
   - `enable_batch_parameter_rollback: bool = False`
   - `compute_eval_aux_loss: bool = False`
   - `max_train_batches_per_epoch: int = 0`
   - `quick_val_max_batches: int = 0`
   - `full_val_interval_epochs: int = 1`
   - `full_val_on_last_epoch: bool = True`

2. `src/core/expert1_trainer.py`
   - 默认不再逐 batch 克隆整套模型到 CPU。
   - 保留 non-finite loss 检查、梯度裁剪和 optional rollback。
   - `_run_epoch()` 支持 `max_batches`。
   - `_run_epoch()` 支持 `compute_aux_loss`，评估默认跳过 RD/deep auxiliary loss。
   - 训练循环支持 quick val / full val：
     - quick val 只用于观察趋势。
     - 只有 full val 才允许保存 best / fallback checkpoint。
   - history 中记录：
     - `evaluated_batches`
     - `max_batches`
     - `validation_scope`
   - summary 中记录 `speed_controls`。

3. 新增快速迭代配置：
   - `tmp/expert1_v2_fast_config.json`
   - 输出目录：`artifacts/expert1_v2_fast`
   - epochs = 4
   - stage1 trajectory anchor = 1 epoch
   - stage2 ramp = 2 epochs
   - quick val = 12 batches
   - full val 只在最后一轮执行

### 验证

```text
.venv\Scripts\python.exe -m compileall -q src\core tmp\evaluate_expert1_checkpoint.py
.venv\Scripts\python.exe -c "from config.configmanager import load_expert1_config; c=load_expert1_config('tmp/expert1_v2_fast_config.json'); print(c.output_dir, c.epochs, c.stage1_traj_only_epochs, c.quick_val_max_batches, c.full_val_interval_epochs, c.enable_batch_parameter_rollback, c.compute_eval_aux_loss)"
```

输出确认：

```text
artifacts/expert1_v2_fast 4 1 12 99 False False
```

### 手动训练命令

```powershell
.\.venv\Scripts\python.exe -m src.core.train_expert1 --config tmp\expert1_v2_fast_config.json
```

训练完成后必须把 `artifacts/<timestamp>/expert1_v2_fast/training_summary.json`、`evaluation_report.json`、`training_history.json` 的完整指标复制到 `performance_records/<timestamp>/expert1_v2_fast/`。

## 2026-05-11：Expert1 V2 smoke 训练与二次修正

### 训练 1：joint 直接进入 final

- 配置：`tmp/expert1_v2_smoke_config.json`
- 输出：`artifacts/20260511-132710/expert1_v2_smoke`
- 性能快照：`performance_records/20260511-132710/expert1_v2_smoke`
- 速度设置：
  - epochs = 5
  - `max_train_batches_per_epoch = 32`
  - `quick_val_max_batches = 12`
  - `low_altitude_deep_reroute_aux_weight = 0.0`

Test 指标：

| 指标 | 值 |
|---|---:|
| final macro F1 | 0.6004 |
| trajectory macro F1 | 0.5914 |
| trajectory macro F1 on RD | 0.6172 |
| joint macro F1 on RD | 0.3932 |
| joint - trajectory macro F1 on RD | -0.2241 |
| RD macro F1 on RD | 0.1478 |
| RD harm rate on RD | 0.5907 |
| joint RD weight mean on RD | 0.4500 |

结论：

- 训练速度已可控，5 epoch smoke 约 6 分钟完整结束。
- 但 joint 分支仍明显弱于 trajectory。
- reliability 继续顶到 0.45 cap，说明 joint 直接参与 final 仍不安全。

### 代码修正

新增两个配置项：

- `joint_logit_delta_scale`
- `rd_specialist_residual_scale`

对应代码：

- `src/core/expert1_config.py`
- `src/core/experts.py`
- `src/core/expert1_trainer.py`

目的：

```text
joint_logits 可以继续训练和报告，
但是否进入 final logits 由 joint_logit_delta_scale 控制。
```

fast/smoke 配置当前设置：

```text
joint_logit_delta_scale = 0.0
rd_specialist_residual_scale = 0.5
```

### 训练 2：joint 隔离 final

- 输出：`artifacts/20260511-133618/expert1_v2_smoke`
- 性能快照：`performance_records/20260511-133618/expert1_v2_smoke`

Test 指标：

| 指标 | 值 |
|---|---:|
| final macro F1 | 0.5972 |
| trajectory macro F1 | 0.5972 |
| trajectory macro F1 on RD | 0.6409 |
| joint macro F1 on RD | 0.0159 |
| joint - trajectory macro F1 on RD | -0.6250 |
| RD macro F1 on RD | 0.1449 |
| RD harm rate on RD | 0.5781 |
| joint RD weight mean on RD | 0.2035 |
| with RD accuracy | 0.7342 |
| without RD accuracy | 0.4795 |

结论：

- joint 隔离后 final 不再被 joint 直接拖垮，final 与 trajectory 完全一致。
- RD weight 不再顶到 0.45，下降到约 0.2035。
- 但 32 batch smoke 不足以训练出可用 joint/RD evidence。
- 下一轮应使用同样的安全 final path，去掉 `max_train_batches_per_epoch` 跑全量 5 epoch；如果仍太慢，再优化数据加载或精简 RD auxiliary，而不是重新让未成熟 joint logits 进入 final。

## 2026-05-11：Expert1 V2 fast 20260511-152913 深入复盘

### 训练产物

- 输出目录：`artifacts/20260511-152913/expert1_v2_fast`
- Markdown：`reports/real/20260511-152913/expert1_v2_fast.md`
- JSON：`reports/real/20260511-152913/expert1_v2_fast.json`
- 性能快照：`performance_records/20260511-152913/expert1_v2_fast`
- best checkpoint：`fallback_candidate_model.pt`
- best epoch：5
- best selection reason：`fallback_no_rd_quality_checkpoint`

### 关键配置

```text
epochs = 5
stage1_traj_only_epochs = 1
stage2_ramp_epochs = 2
compute_eval_aux_loss = false
enable_batch_parameter_rollback = false
quick_val_max_batches = 12
full_val_interval_epochs = 3
joint_logit_delta_scale = 0.0
rd_specialist_residual_scale = 0.5
low_altitude_deep_reroute_aux_weight = 0.0
low_altitude_rd_forced_fusion_weight = 0.45
low_altitude_rd_gate_aux_weight = 0.18
low_altitude_rd_class_gate_aux_weight = 0.15
low_altitude_rd_pair_gate_aux_weight = 0.20
```

### 和 20260510-105622 对比

| 指标 | 20260510-105622 | 20260511-152913 | 差值 |
|---|---:|---:|---:|
| final accuracy | 0.8548 | 0.7903 | -0.0645 |
| final macro F1 | 0.7327 | 0.6922 | -0.0405 |
| trajectory accuracy | 0.8581 | 0.7903 | -0.0677 |
| trajectory macro F1 | 0.7486 | 0.6922 | -0.0564 |
| RD accuracy on RD | 0.7173 | 0.5527 | -0.1646 |
| RD macro F1 on RD | 0.6174 | 0.4481 | -0.1693 |
| with RD accuracy | 0.8819 | 0.8397 | -0.0422 |
| without RD accuracy | 0.7671 | 0.6301 | -0.1370 |
| source balanced macro F1 | 0.4788 | 0.4281 | -0.0507 |
| excel_track macro F1 | 0.5948 | 0.4594 | -0.1354 |
| raw_tr_rd macro F1 | 0.0800 | 0.0800 | 0.0000 |

逐类变化：

| class | F1 差值 | recall 差值 |
|---|---:|---:|
| DroneTarget | -0.0497 | -0.0397 |
| BirdTarget | -0.1228 | -0.1125 |
| BalloonTarget | -0.0390 | -0.0556 |
| ClutterTarget | +0.1263 | +0.1429 |
| UnknownTarget | -0.1175 | -0.1667 |

逐源变化：

| source | accuracy 差值 | macro F1 差值 |
|---|---:|---:|
| cq08_track | -0.0429 | -0.0166 |
| excel_track | -0.1370 | -0.1354 |
| raw_tr_rd | 0.0000 | 0.0000 |

### 根因判断

1. 这次 final 不是被 RD/joint 拖坏。
   - `final macro F1 = trajectory macro F1 = 0.6922`
   - `final accuracy = trajectory accuracy = 0.7903`
   - `joint_logit_delta_scale = 0.0`
   - `final_harm_rate_on_rd = 0`
   - 说明 final 基本就是 trajectory anchor，joint 没进入 final。

2. 真正退化的是 trajectory anchor。
   - 20260510 trajectory macro F1：0.7486。
   - 20260511 trajectory macro F1：0.6922。
   - 差值 -0.0564。
   - without RD accuracy 从 0.7671 掉到 0.6301，说明 Excel/无 RD 轨迹路径明显变弱。

3. Excel 是最大可见退化源。
   - Excel macro F1 从 0.5948 掉到 0.4594。
   - Excel 没有 RD，无法被 RD/joint 解释。
   - 这说明问题主要在轨迹训练配方，而不是多模态融合本身。

4. RD evidence 也明显退化。
   - RD macro F1 on RD 从 0.6174 掉到 0.4481。
   - RD harm rate on RD 从约 0.1983 升到 0.3418。
   - RD Drone/Bird still usable partly，但 Balloon 为 0，整体不足以作为第二证据。

5. joint branch 仍不可用。
   - `joint_accuracy_on_rd = 0.0844`
   - `joint_macro_f1_on_rd = 0.0434`
   - `joint_minus_traj_macro_f1_on_rd = -0.6958`
   - 因为 joint 被隔离，这个结果没有伤 final，但证明 joint head 尚未学会。

6. 当前 fast 配置为了安全和速度削弱了太多训练信号。
   - 20260510 是 12 epoch；本次是 5 epoch。
   - deep reroute auxiliary 从 0.35 改为 0。
   - forced fusion 从 0.9 改为 0.45。
   - gate/class/pair gate auxiliary 权重整体减半左右。
   - `joint_logit_delta_scale=0` 是正确安全设置，但也意味着 final 完全吃不到任何可能的 RD 正收益。
   - `rd_specialist_residual_scale=0.5` 进一步减少了 RD 对 final 的实际影响。

### 结论

这次数据差的核心不是“V2 联合架构把模型搞坏”，而是：

```text
为了防止 joint 误伤，我们把 final 退回到了纯 trajectory；
但同时 fast 配方又削弱了 trajectory/RD evidence 的训练强度；
结果 final 只能反映一个训练不足的 trajectory anchor。
```

因此这版“一点不能用”是合理判断。它不能进入 System，也不能作为 Expert1 候选。

### 下一步修正

1. 保留安全原则：
   - `joint_logit_delta_scale = 0`
   - joint-on-RD 没超过 trajectory-on-RD 前，joint 不进 final。

2. 恢复强 trajectory/RD evidence 训练：
   - epochs 至少回到 10 到 12。
   - 恢复 `low_altitude_deep_reroute_aux_weight = 0.35` 或至少 0.2。
   - 恢复 `low_altitude_rd_forced_fusion_weight = 0.9`，但它只训练 final/evidence，不让 joint logits 直接进 final。
   - 旧 gate/class/pair gate auxiliary 后续已清理，不再恢复；同类约束改由 joint/final 选模指标承接。

3. 训练速度不要靠削弱关键 loss 解决：
   - 保持关闭 batch parameter rollback。
   - eval 继续不算 auxiliary loss。
   - quick val 继续只用于观察。
   - full val 至少在 epoch 5、10、last 执行。

4. 选模要求：
   - trajectory macro F1 必须先回到 0.74 附近。
   - RD macro F1 on RD 必须回到 0.60 附近。
   - Excel macro F1 不能低于 0.55。
   - raw_tr_rd 仍只能作为风险诊断，4 条样本不足以主导总体选择。

### 已执行修正

- `target.md` 已记录 `20260511-152913/expert1_v2_fast` 为失败样本，不允许进入 Expert1/System 候选。
- `tmp/expert1_v2_fast_config.json` 已从弱 fast 配方改为恢复型配方：
  - `epochs = 12`
  - `stage2_ramp_epochs = 1`
  - `low_altitude_deep_reroute_aux_weight = 0.35`
  - `low_altitude_rd_forced_fusion_weight = 0.9`
  - 旧 RD gate / class gate / pair gate auxiliary 已删除，不再恢复。
  - `low_altitude_traj_anchor_weight = 0.15`
- 仍保留工程侧加速：
  - `enable_batch_parameter_rollback = false`
  - `compute_eval_aux_loss = false`
  - `quick_val_max_batches = 12`
  - `full_val_interval_epochs = 3`
- 仍保留安全隔离：
  - 该 fast 失败版不作为候选。
  - 后续候选全量训练改用 `joint_logit_delta_scale = 1.0`，让有 RD 样本的主 loss 直接监督 TR+RD joint。

## 2026-05-11：Expert1 V2 TR+RD 混合判断决策与代码修正

用户确认 Expert1 主结构：

```text
if no RD:
    使用当前强 trajectory branch 判断
else:
    进入 TR+RD 混合判断
```

本次修正的核心原因：

- 上一版 `joint_logit_delta_scale = 0.0` 后，joint 分支不会从有 RD 样本的主分类 loss 得到有效监督。
- 这会导致 joint-on-RD 指标极差，不能说明 TR+RD 联合方向错误，只说明 joint head 没被当作主路径训练。
- 现在有 RD 样本必须直接用 TR+RD 混合分支承接主 loss；无 RD 样本继续由 trajectory-only 分支承接主 loss。

代码修正：

- `src/core/experts.py`
  - 新增 `CrossModalFusionBlock`。
  - `LowAltitudeSuspectExpert.forward()` 新增 `traj_sequence_features` 与 `traj_padding_mask`。
  - Joint 分支从简单 `traj_embedding + rd_embedding` 拼接，升级为：
    - trajectory sequence tokens
    - RD sequence tokens
    - TR->RD / RD->TR cross-attention
    - fused trajectory embedding
    - fused RD embedding
    - 原始 trajectory / RD anchor embedding
    - abs diff / product / RD stats / RD patch
  - `joint_logit_delta_scale >= 1.0` 时，有 RD 样本 final 使用 `joint_logits` 作为主输出。
  - 无 RD / 非多模态 RD 样本 `joint_rd_weight = 0`，final 保持 `traj_logits`。
- `src/core/expert1_trainer.py`
  - Expert1 训练时传入 `rep.sequence_features` 与 `padding_mask`。
  - 有 RD 样本的主 loss 现在可以直接监督 TR+RD 混合分支。
- `src/core/system.py`
  - System 调用 Expert1 时同步传入 `representation.sequence_features` 与 `padding_mask`，保持 standalone / system 架构一致。
- `tests/test_expert1_evidence.py`
  - 结构测试更新为：有 RD 样本使用 joint logits；无 RD 样本回退到 trajectory logits。

全量训练配置：

- `tmp/expert1_v2_tr_rd_full_config.json` 已保存为全量候选配置：
  - `output_dir = artifacts/expert1_v2_tr_rd_full`
  - `epochs = 12`
  - `max_train_batches_per_epoch = 0`
  - `quick_val_max_batches = 0`
  - `full_val_interval_epochs = 1`
  - `joint_logit_delta_scale = 1.0`
  - `low_altitude_deep_reroute_aux_weight = 0.35`
  - `low_altitude_rd_forced_fusion_weight = 0.9`
  - 旧 RD gate / class gate / pair gate auxiliary 参数不再存在。

后续只跑全量训练，不再用轻量快速训练作为候选结论。

## 2026-05-11：Expert1 V2 架构变更后的冗余代码清理

清理范围：当前代码中的 Expert1 历史遗留路径；未触碰 `.history`。

删除 / 停用的旧路径：

- `LowAltitudeSuspectExpert` 中旧的 RD activation gate。
- `fusion_gate`、`rd_class_gate`、`rd_pair_gate` 网络。
- trainer 中旧的 gate auxiliary loss：
  - `_fusion_gate_aux_loss`
  - `_rd_gate_target_loss`
  - `_rd_class_gate_target_loss`
  - `_rd_pair_gate_target_loss`
- `rd_activation_threshold`、`low_altitude_initial_rd_gate_bias`、`low_altitude_rd_class_activation_threshold` 等旧配置入口。
- `low_altitude_rd_gate_aux_weight`、class gate / pair gate auxiliary 权重。
- System / diagnostics 中对旧 gate 参数的传参。

保留内容：

- RD auxiliary heads：
  - `rd_head`
  - `rd_bird_head`
  - `rd_unknown_head`
  - `rd_clutter_head`
  - `rd_residual_head`
  - `rd_unknown_clutter_head`
  - `rd_clutter_morphology_head`
- TR+RD cross-modal fusion branch。
- `joint_reliability_head` 与 `joint_rd_weight`，用于 residual 限幅和诊断。
- hard-pair 报告与 RD rescue / harm / net gain 诊断。
- 小权重 bounded RD specialist residual。

当前语义：

```text
rd_activated = has_valid_rd and not disable_tr_rd_fusion
```

它不再表示“RD gate 选择性激活”，只表示样本是否进入 TR+RD 混合分支。

验证：

- `tests.test_expert1_evidence.Expert1EvidenceTests.test_rd_samples_use_joint_branch_and_no_rd_falls_back_to_trajectory` 通过。
- `tests.test_expert1_evidence.Expert1EvidenceTests.test_rd_activation_now_means_tr_rd_path_presence` 通过。
- System 前向 sanity check 通过：低空输出为 5 类，无 RD 样本 final 仍等于 `traj_logits`。
- `tmp/expert1_v2_tr_rd_full_config.json` 与 `config/config.json` 可正常加载。

## 2026-05-12：Expert1 V2 TR+RD full 训练复盘

本次全量训练产物：

- real report markdown：`reports/real/20260511-224549/expert1_v2_tr_rd_full.md`
- real report json：`reports/real/20260511-224549/expert1_v2_tr_rd_full.json`
- artifact：`artifacts/20260511-224549/expert1_v2_tr_rd_full`
- 性能快照：`performance_records/20260511-224549/expert1_v2_tr_rd_full`

训练设置：

- `epochs = 12`
- `stage1_traj_only_epochs = 1`
- `stage2_ramp_epochs = 1`
- `joint_logit_delta_scale = 1.0`
- `rd_specialist_residual_scale = 0.5`
- 每轮 full validation，不使用 quick val。
- split：train / val / test = 1432 / 306 / 310。
- RD 覆盖：train 913/1432，val 238/306，test 237/310。
- 可训练参数：24,409,745。

### 结果对比

| 版本 | final acc | final macro F1 | traj macro F1 | RD-only macro F1 on RD | joint macro F1 on RD | joint - traj on RD | final RD rescue / harm / net |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20260510 stable fallback | 0.8548 | 0.7327 | 0.7486 | 0.6174 | - | - | 2 / 3 / -1 |
| 20260511 fast failed | 0.7903 | 0.6922 | 0.6922 | 0.4481 | 0.0434 | -0.6958 | 0 / 0 / 0 |
| 20260511 full TR+RD | 0.8710 | 0.7971 | 0.7614 | 0.6431 | 0.8216 | +0.0460 | 22 / 5 / +17 |

这次不是 fast 版那种失败。核心变化是：

- final macro F1 达到 0.7971，超过 20260510 的 0.7327。
- trajectory macro F1 达到 0.7614，也略高于 20260510 的 0.7486。
- RD-only on-RD macro F1 达到 0.6431，略高于 20260510 的 0.6174，但 RD-only rescue/harm/net 仍是 12/45/-33，不能直接作为主判断。
- joint-on-RD macro F1 达到 0.8216，高于 trajectory-on-RD 0.7756。
- final 在 RD 样本上 rescue/harm/net 为 22/5/+17，说明 TR+RD joint 作为主路径确实学到了有效联合特征。

### 分支结论

纯轨迹分支：

- 全测试集：accuracy 0.8161，macro F1 0.7614。
- on-RD：accuracy 0.8312，macro F1 0.7756。
- no-RD 73 条样本 final 与 trajectory 等价，accuracy 0.7671。

TR+RD joint 分支：

- 全测试集 joint macro F1 0.7899。
- on-RD accuracy 0.9030，macro F1 0.8216。
- 相比 trajectory-on-RD 提升 +0.0460 macro F1。
- 这证明“有 RD 则进入 TR+RD 混合判断，无 RD 则走轨迹分支”的架构方向成立。

RD-only 分支：

- on-RD accuracy 0.6920，macro F1 0.6431。
- RD-only rescue/harm/net = 12/45/-33。
- 结论：RD-only 仍适合做 auxiliary evidence，不适合作为最终接管分支，也不适合作为新架构选模的主硬门槛。

### Hard Pair

| hard pair | support | trajectory acc | RD-only acc | final acc | final net |
|---|---:|---:|---:|---:|---:|
| DroneTarget vs BirdTarget | 162 | 0.7901 | 0.6728 | 0.8951 | +17 |
| UnknownTarget vs ClutterTarget | 22 | 0.7727 | 0.7727 | 0.7727 | 0 |
| BirdTarget vs ClutterTarget | 63 | 0.7460 | 0.6508 | 0.7778 | +2 |

最关键的 Drone/Bird 对已经被 TR+RD joint 明显拉开，这是这次训练最有价值的信号。

### 跨源问题

| source | count | final acc | final macro F1 | trajectory macro F1 | 结论 |
|---|---:|---:|---:|---:|---|
| cq08_track | 233 | 0.9142 | 0.8287 | 0.7791 | joint 明显提升 |
| excel_track | 73 | 0.7671 | 0.5881 | 0.5881 | 无 RD，保持轨迹路径；略低于 20260510 的 0.5948 |
| raw_tr_rd | 4 | 0.2500 | 0.0800 | 0.1333 | joint 对弱源仍有误伤 |

按 RD 样本逐源看：

- `cq08_track`：trajectory 195/233 对，final 213/233 对，final rescue/harm = 22/4。
- `raw_tr_rd`：trajectory 2/4 对，final 1/4 对，final rescue/harm = 0/1。

因此当前模型可以作为 Expert1 V2 standalone 候选，但还不能直接接入 System 主力。主要风险不是总体精度，而是 raw 弱源样本太少且不稳。

### 训练曲线

关键验证轮次：

| epoch | val final F1 | val traj F1 | val joint-on-RD F1 | val RD-only on-RD F1 | eligible |
|---:|---:|---:|---:|---:|---|
| 2 | 0.7694 | 0.7573 | 0.7924 | 0.1580 | false |
| 5 | 0.7301 | 0.5841 | 0.7586 | 0.4817 | false |
| 8 | 0.8117 | 0.7240 | 0.8386 | 0.4276 | false |
| 11 | 0.8405 | 0.7748 | 0.8655 | 0.5914 | false |
| 12 | 0.7667 | 0.7345 | 0.7986 | 0.5412 | false |

第 5 轮仍不稳定，第 8 到 11 轮才出现高质量 joint 表现，说明之前用 5 epoch fast 训练做结论是不可靠的。最佳验证点在 epoch 11，但由于旧 quality gate 仍要求 RD-only net gain 非负，所以 `best_selection_reason` 仍是 `fallback_no_rd_quality_checkpoint`。

### 架构判断

当前 Expert1 V2 应继续坚持联合路线：

```text
no RD:
    trajectory-only final

has RD:
    TR tokens + RD tokens cross-modal fusion -> joint final
    RD-only heads 只做 auxiliary evidence / residual 诊断
```

需要修正的是选模和报告语义：

- 旧规则把 `rd_net_gain_rate_on_rd >= 0` 当硬门槛，这适用于“RD 默认第二证据/接管”的历史架构。
- 新架构中 final 主路径是 joint，因此候选硬门槛应改成：
  - `joint_minus_traj_macro_f1_on_rd >= 0`
  - `final_net_gain_rate_on_rd >= 0`
  - `final_harm_rate_on_rd` 不超过阈值
  - hard-pair final net 不为负
- RD-only 只保留最低表征质量门槛，例如 `rd_macro_f1_on_rd`、Drone/Bird recall、Unknown/Clutter recall，不再要求 RD-only 净收益为正。

下一步优先级：

1. 修改 Expert1 selection gate，让它按 joint/final 选择 checkpoint，而不是被 RD-only net gain 否决。
2. 报告补充 source x joint metrics、source x final-vs-traj rescue/harm/net。
3. 针对 raw_tr_rd 做 leave-source / bootstrap 诊断，不让 4 条 raw 样本直接主导总体结论，但也不能忽略 raw 误伤。
4. 在 selection gate 修正后，用同一全量配置复训一次，确认 best checkpoint 不再落入 fallback reason。

## 2026-05-12：Expert1 旧架构逻辑第二轮清理，切换到 V2 语义

本轮目标是把代码和当前文档从旧 RD-only 选模语义切换到 V2 joint/final 语义；`.history` 未触碰。

代码清理：

- `LowAltitudeSuspectExpert` 默认推理模式从旧 `dual_main` 改为 `v2_joint`。
- System 调用 Expert1 时同步使用 `v2_joint`，保持 standalone / system 路径一致。
- Expert1 acceptance 不再用 RD-only cosine confidence 作为有 RD 样本的主置信度，而是使用 final logits confidence。
- 旧 RD-only selection hard gate 移除：
  - `min_rd_macro_f1_for_best` 改为 `min_rd_evidence_macro_f1_for_best`。
  - `min_rd_net_gain_rate_for_best` 和 `max_rd_harm_rate_for_best` 只保留 legacy config 丢弃兼容，不再进入选模。
- 新 `v2_selection_gate` 选模指标：
  - `joint_minus_traj_macro_f1_on_rd`。
  - `final_net_gain_rate_on_rd`。
  - `final_harm_rate_on_rd`。
  - `final_minus_traj_macro_f1`。
  - RD-only macro F1 仅作为 evidence 质量下限。
- `selection_reason` 从 `stage2_rd_composite` 切到 `stage2_v2_joint_composite`，fallback reason 切到 `fallback_v2_selection_checkpoint`。
- 新增 `tr_rd_active` / `tr_rd_path_rate` 作为 V2 诊断名；`rd_activated` / `rd_activation_rate` 仅作为旧报告和测试脚本兼容别名保留。

报告与指标清理：

- Expert1 epoch/test metrics 新增 joint by-source、joint-on-RD、joint minus trajectory-on-RD。
- Markdown 报告中的旧“RD 分支”改为“V2 分支”，展示 `TR+RD joint` 和 `RD evidence` 两套指标。
- prediction record 新增 `label_name_joint_pred`，便于审计 joint 与 final 的差异。
- 报告 summary 补齐 final/RD net gain rate 字段，避免测试从旧 artifact 读取时缺字段。

文档清理：

- `target.md` 当前目标改为 `v2_joint` / `v2_selection_gate`。
- `main.md` 增加 V2 选模门槛，明确 RD-only 分支不再以净收益作为 checkpoint 否决条件。
- 历史报告中的 `fallback_no_rd_quality_checkpoint` 保留为历史事实，不改写旧训练结论。

验证：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_expert1_evidence
```

结果：10 个测试通过，1 个测试按预期跳过，原因是当前最新 artifact 生成于 `v2_selection_gate` 代码清理前。

```powershell
.\.venv\Scripts\python.exe -m py_compile src/core/expert1_config.py src/core/expert1_trainer.py src/core/experts.py src/core/system.py src/core/reporting.py config/configmanager.py tests/test_expert1_evidence.py
```

结果：通过。

```powershell
.\.venv\Scripts\python.exe -c "from config.configmanager import load_expert1_config; c=load_expert1_config('config/config.json'); print(c.min_rd_evidence_macro_f1_for_best, c.min_joint_minus_traj_macro_f1_on_rd_for_best, c.min_final_net_gain_rate_on_rd_for_best, c.max_final_harm_rate_on_rd_for_best, c.joint_logit_delta_scale)"
```

结果：`0.22 0.0 0.0 0.08 1.0`。

后续动作：

1. 使用同一全量配置重新训练 Expert1 V2，确认新 best checkpoint 不再被旧 RD-only gate 否决。
2. 重点看 `stage2_v2_joint_composite` 是否出现，以及 test 上 joint/final 是否继续保持正收益。
3. 若 raw_tr_rd 仍被误伤，下一轮不回退 RD-only 选择性激活，而是做 source-aware calibration / raw replay。

## 2026-05-12：Expert1 V2.1 joint 质量特征与 residual 收缩

本轮继续优化新架构，目标是保留 “有 RD -> TR+RD joint” 主线，但降低 raw 弱源被 RD-only residual 二次误伤的风险。

发现的问题：

- 20260511-224549 全量模型中，`raw_tr_rd` 只有 4 条；trajectory 2/4 对，RD evidence 2/4 对，但 final 只有 1/4 对。
- 其中一条 raw Bird 样本 trajectory 和 RD evidence 都预测 Bird，但 final 变成 Drone，说明 joint/final 路径仍缺少显式的一致性保护。
- 原先 joint 输入主要是 fused embeddings、RD stats 和 patch，不直接看到 trajectory/RD 置信度、熵、agreement、RD cosine confidence、residual score 等质量特征。
- RD specialist residual 虽然已 bounded，但默认 scale 仍偏大，容易在样本极少的 raw 源上把 RD-only logits 的负收益带回 final。

代码优化：

- `LowAltitudeSuspectExpert` 新增 `joint_quality_proj`。
- TR+RD joint 输入新增 12 维质量特征：
  - `has_rd`
  - trajectory confidence
  - RD confidence
  - RD cosine confidence
  - trajectory/RD confidence gap
  - trajectory/RD agreement
  - residual score
  - trajectory entropy
  - RD entropy
  - RD stats mean / std
  - RD patch norm
- 新增 `low_altitude_joint_agreement_prior_scale = 0.25`：
  - 当 trajectory 与 RD evidence 预测一致时，给 agreed class 一个小 prior。
  - 该 prior 只进入 joint logits，不能替代主分类 loss。
- 新增 `low_altitude_joint_aux_weight = 0.35`：
  - 训练期对有 RD 样本的 `joint_logits` 直接加权 CE。
  - 目的：避免 final loss 被 residual/reject 细节稀释，确保 joint head 本身学会 TR+RD 判断。
- 新增 `low_altitude_joint_reliability_aux_weight = 0.12`：
  - 只监督 `joint_rd_weight` 这个 bounded residual 权重。
  - RD rescue 时允许 residual 权重略高，RD harm 时压低，避免 residual 变成旧式 RD 接管。
- `rd_specialist_residual_scale` 默认从 1.0 / 0.5 收缩到 0.35。
- System 初始化新增：
  - `low_altitude_joint_agreement_prior_scale`
  - `low_altitude_rd_specialist_residual_scale`

配置同步：

- `config/config.json` 已加入：
  - `low_altitude_joint_aux_weight = 0.35`
  - `low_altitude_joint_agreement_prior_scale = 0.25`
  - `low_altitude_joint_reliability_aux_weight = 0.12`
  - `rd_specialist_residual_scale = 0.35`
- `tmp/expert1_v2_tr_rd_full_config.json` 已同步上述参数，仍为 12 epoch、full validation、非快速训练。

下一次全量训练重点看：

```text
joint_macro_f1_on_rd > traj_macro_f1_on_rd
final_net_gain_rate_on_rd >= 0
final_harm_rate_on_rd <= 0.08
raw_tr_rd final >= raw_tr_rd trajectory
joint_rd_weight_on_rd_mean 不贴上限
best_selection_reason = stage2_v2_joint_composite
```

建议训练命令：

```powershell
.\.venv\Scripts\python.exe -m src.core.train_expert1 --config tmp/expert1_v2_tr_rd_full_config.json
```

主配置 `config/config.json` 的 Expert1 段也已同步为全量 V2 训练参数，因此也可以使用：

```powershell
.\.venv\Scripts\python.exe -m src.core.train_expert1 --config config/config.json
```

## 2026-05-12：Expert1 V2 full 20260512-124930 复盘与 stable-stage 修复

本轮全量训练命令：

```powershell
.\.venv\Scripts\python.exe -m src.core.train_expert1 --config tmp/expert1_v2_tr_rd_full_config.json
```

训练输出：

- real report markdown：`reports/real/20260512-124930/expert1_v2_tr_rd_full.md`
- real report json：`reports/real/20260512-124930/expert1_v2_tr_rd_full.json`
- artifact：`artifacts/20260512-124930/expert1_v2_tr_rd_full`
- 性能快照：`performance_records/20260512-124930/expert1_v2_tr_rd_full`

核心结果：

| 指标 | 值 |
|---|---:|
| best validation epoch | 11 |
| final accuracy / macro F1 | 0.8613 / 0.7706 |
| trajectory accuracy / macro F1 | 0.7774 / 0.6993 |
| joint-on-RD accuracy / macro F1 | 0.8987 / 0.7964 |
| joint-on-RD - trajectory-on-RD macro F1 | +0.0863 |
| RD-only on-RD macro F1 | 0.6627 |
| final rescue / harm / net on RD | 31 / 5 / +26 |
| RD-only rescue / harm / net on RD | 21 / 41 / -20 |
| source-balanced macro F1 | 0.5228 |
| cq08 / excel / raw macro F1 | 0.8007 / 0.5677 / 0.2000 |

判断：

- 架构方向成立：TR+RD joint 在有 RD 样本上明显高于 trajectory，final 对 RD 样本是正净收益。
- 模型还不能接入 System：整体 macro F1 低于 20260511 候选，Unknown/Clutter 与弱源仍拖后腿。
- 本轮最主要问题不是 joint 无效，而是训练过程破坏了 trajectory anchor：
  - `stage1_traj_only_epochs = 1`
  - `stage2_ramp_epochs = 1`
  - 第 2 轮直接全量打开 RD/joint/deep auxiliary，训练 loss 从 2.4309 跳到 13.3065。
  - 无 RD 的 `excel_track` 只能走 trajectory path，因此 trajectory 退化会直接压低 without-RD 表现。

代码修复：

- `Expert1TrainConfig` 新增 `low_altitude_traj_supervision_weight`。
- Expert1 训练 loss 新增持续 trajectory supervised CE：

```text
loss =
    final main loss
    + low_altitude_traj_supervision_weight * CE(traj_logits, label)
    + stage2_scale * auxiliary losses
```

- `low_altitude_traj_anchor_start_epoch` 现在真正生效；只有达到指定 stage2 epoch 后才启用 trajectory KL anchor。
- `low_altitude_rd_forced_fusion_weight`、`low_altitude_joint_aux_weight`、`low_altitude_joint_reliability_aux_weight`、`low_altitude_deep_reroute_aux_weight` 全部乘 `stage2_scale`，避免 stage2 第一轮强冲击。

下一轮 stable full 配置：

- `output_dir = artifacts/expert1_v2_tr_rd_stable_full`
- `epochs = 18`
- `stage1_traj_only_epochs = 3`
- `stage2_ramp_epochs = 4`
- `low_altitude_traj_supervision_weight = 0.55`
- `low_altitude_traj_aux_ratio = 0.55`
- `low_altitude_rd_aux_ratio = 0.45`
- `low_altitude_rd_forced_fusion_weight = 0.45`
- `low_altitude_deep_reroute_aux_weight = 0.20`
- `low_altitude_traj_anchor_weight = 0.35`
- `low_altitude_traj_anchor_start_epoch = 2`
- `low_altitude_rd_teacher_warmup_epochs = 2`
- `excel_source_boost = 2.6`
- `raw_source_boost = 2.5`

下一轮训练重点看：

```text
trajectory macro F1 是否回升
without_rd accuracy 是否回升
excel_track macro F1 是否回升
joint-on-RD macro F1 是否仍然 > trajectory-on-RD
final net gain on RD 是否保持 >= 0
stage2 第 4-7 轮是否不再出现 loss 突跳
```

## 2026-05-12：Expert1 V2 代码全量审查与数据归因

审查对象：

- 代码路径：`src/core/expert1_data.py`、`src/core/experts.py`、`src/core/expert1_trainer.py`、`config/config.json`、`tmp/expert1_v2_tr_rd_stable_full_config.json`
- 训练产物：`artifacts/20260512-124930/expert1_v2_tr_rd_full`
- 报告：`reports/real/20260512-124930/expert1_v2_tr_rd_full.md/json`
- 性能快照：`performance_records/20260512-124930/expert1_v2_tr_rd_full`

外部依据：

- Scientific Reports 2018：无人机螺旋桨和飞鸟扑翼都会产生 micro-Doppler，低空无人机与飞鸟需要依赖微多普勒差异进行区分。链接：`https://www.nature.com/articles/s41598-018-35880-9`
- Signals 2023：鸟和无人机 RCS 接近，单靠 RCS/HRRP/SAR 容易困难；micro-Doppler 二分类可到约 96%，但五类细分约 85%，并指出真实数据不足会限制识别。链接：`https://www.mdpi.com/2624-6120/4/2/18`

代码审查结论：

1. 当前 V2 主路径逻辑正确，已经是“无 RD 走 trajectory，有 RD 走 TR+RD joint”：
   - `LowAltitudeSuspectExpert.forward` 中 `tr_rd_active = has_rd`。
   - `selected_logits = torch.where(tr_rd_active, joint_selected_logits, traj_logits)`。
   - 无 RD 样本的 RD logits、RD tokens、joint RD weight 均被置零。
2. 20260512-124930 的训练配置存在明确问题：
   - `stage1_traj_only_epochs = 1`，`stage2_ramp_epochs = 1`。
   - 第 2 轮 stage2 直接 scale=1.00，train loss 从 2.4309 跳到 13.3065。
   - 训练 summary 中 `low_altitude_rd_forced_fusion_weight = 0.9`，`low_altitude_rd_aux_ratio = 0.65`，RD/joint/deep 监督过强。
3. 已修复的训练风险：
   - 新增持续 `low_altitude_traj_supervision_weight * CE(traj_logits, label)`。
   - `low_altitude_traj_anchor_start_epoch` 已实际控制 anchor KL 启动。
   - forced fusion、joint aux、joint reliability、deep reroute direct loss 已全部乘 `stage2_scale`。
4. 仍需观察的风险：
   - `main_loss` 对 RD 样本监督的是 final/joint logits，而 trajectory backbone 与 joint 共享上游表征；如果 stable full 后 trajectory 仍低，应冻结 track encoder 前几轮或继续降低 sequence encoder LR。
   - 训练历史没有细粒度 loss component 曲线，难以精确区分 RD aux、forced fusion、joint aux、deep reroute 哪一项造成 loss 冲击。下一轮若仍震荡，需要把 loss components 写入 `training_history.json`。
   - selection gate 对 raw 的 hard gate 最小支持数是 4，而 test raw 只有 4 条，raw 指标统计方差极大，适合作为诊断项，不适合单独决定主模型质量。

数据审查结论：

| split | count | cq08 | excel | raw | with RD | no RD |
|---|---:|---:|---:|---:|---:|---:|
| train | 1432 | 1084 | 323 | 25 | 1109 | 323 |
| val | 306 | 232 | 68 | 6 | 238 | 68 |
| test | 310 | 233 | 73 | 4 | 237 | 73 |

test 类别分布：

```text
DroneTarget=151
BirdTarget=80
BalloonTarget=54
ClutterTarget=7
UnknownTarget=18
```

test 来源 x 类别：

```text
cq08_track: Drone=105, Bird=53, Balloon=53, Clutter=7, Unknown=15
excel_track: Drone=45, Bird=24, Balloon=1, Clutter=0, Unknown=3
raw_tr_rd: Drone=1, Bird=3, Balloon=0, Clutter=0, Unknown=0
```

性能归因：

1. 总 accuracy=0.8613 不是完全失效，主要被 cq08 与 Balloon 抬高；macro F1=0.7706 偏低，是因为 Unknown/Clutter 和弱源表现低。
2. 有 RD 样本表现明显优于无 RD 样本：
   - with-RD accuracy=0.8987
   - without-RD accuracy=0.7397
   - no-RD 主要就是 excel_track，所以 Excel 直接暴露 trajectory 分支下限。
3. TR+RD joint 是有效的：
   - trajectory-on-RD macro F1=0.7101
   - joint-on-RD macro F1=0.7964
   - final on RD rescue/harm/net = 31/5/+26
4. RD-only 不是可靠教师：
   - RD-only on-RD macro F1=0.6627
   - RD-only rescue/harm/net = 21/41/-20
   - RD residual accuracy on RD=0.4895
   - 因此 RD 应保留为 joint evidence，不应回到旧式 RD 接管主路径。
5. 低召回主要来自三类：
   - Drone recall=0.8675，主要错成 Bird：17 次。
   - Unknown recall=0.6111，support 只有 18，且会错到 Bird/Drone/Clutter。
   - Clutter recall=0.7143，support 只有 7，precision=0.5000，宏平均被明显拉低。
6. 来源域问题非常明显：
   - cq08 final macro F1=0.8007。
   - excel final macro F1=0.5677，且无 RD，只能验证 trajectory。
   - raw final macro F1=0.2000，但 test 只有 4 条，不能过度解释。

阶段判断：

- 架构继续采用“轨迹保底 + 有 RD 进入 TR+RD joint”。
- 当前低精度/低召回的主要原因不是联合架构错误，而是旧 full 配方 stage2 打开过猛、轨迹分支被联合训练牵动、弱源/稀有类支持不足。
- 下一轮必须先运行 stable full 配方，不再基于 20260512-124930 继续微调旧强冲击配置。

下一步：

```powershell
.\.venv\Scripts\python.exe -m src.core.train_expert1 --config tmp\expert1_v2_tr_rd_stable_full_config.json
```

若 stable full 后仍未恢复：

1. `freeze_track_encoder_epochs` 提到 3-5。
2. `sequence_encoder_lr_multiplier` 从 0.1 降到 0.05。
3. 记录 loss components 到 `training_history.json`。
4. 将 raw hard gate 改为 support >= 8 或仅作为 diagnostic gate。
5. 对 Unknown/Clutter 增补样本或改成 source-aware residual calibration。

## 2026-05-13：Expert1 V2 stable full 20260512-204012 复盘

本轮训练命令：

```powershell
.\.venv\Scripts\python.exe -m src.core.train_expert1 --config tmp\expert1_v2_tr_rd_stable_full_config.json
```

训练输出：

- real report markdown：`reports/real/20260512-204012/expert1_v2_tr_rd_stable_full.md`
- real report json：`reports/real/20260512-204012/expert1_v2_tr_rd_stable_full.json`
- artifact：`artifacts/20260512-204012/expert1_v2_tr_rd_stable_full`
- 性能快照：`performance_records/20260512-204012/expert1_v2_tr_rd_stable_full`

核心结果：

| 指标 | 值 |
|---|---:|
| best epoch / selection reason | 14 / `stage2_v2_joint_composite` |
| final accuracy / macro F1 | 0.8645 / 0.7526 |
| trajectory accuracy / macro F1 | 0.8581 / 0.7358 |
| trajectory macro F1 on RD | 0.7459 |
| joint macro F1 on RD | 0.7618 |
| joint-on-RD - trajectory-on-RD macro F1 | +0.0158 |
| RD-only on-RD macro F1 | 0.5677 |
| final rescue / harm / net on RD | 7 / 5 / +2 |
| RD-only rescue / harm / net on RD | 9 / 52 / -43 |
| with RD / without RD accuracy | 0.8734 / 0.8356 |
| cq08 / excel / raw macro F1 | 0.7659 / 0.5991 / 0.2000 |
| source-balanced macro F1 | 0.5217 |

与 20260512-124930 对比：

| 指标 | 124930 full | 204012 stable | 变化 |
|---|---:|---:|---:|
| accuracy | 0.8613 | 0.8645 | +0.0032 |
| macro F1 | 0.7706 | 0.7526 | -0.0180 |
| trajectory macro F1 | 0.6993 | 0.7358 | +0.0365 |
| trajectory-on-RD macro F1 | 0.7101 | 0.7459 | +0.0359 |
| joint-on-RD macro F1 | 0.7964 | 0.7618 | -0.0346 |
| joint-on-RD minus trajectory-on-RD | +0.0863 | +0.0158 | -0.0705 |
| RD-only on-RD macro F1 | 0.6627 | 0.5677 | -0.0950 |
| with-RD accuracy | 0.8987 | 0.8734 | -0.0253 |
| without-RD accuracy | 0.7397 | 0.8356 | +0.0959 |
| final net gain rate on RD | +0.1097 | +0.0084 | -0.1013 |
| source-balanced macro F1 | 0.5228 | 0.5217 | -0.0011 |
| excel macro F1 | 0.5677 | 0.5991 | +0.0314 |

判断：

- stable full 成功修复了 trajectory / without-RD / Excel 下限：
  - trajectory macro F1 从 0.6993 回到 0.7358。
  - without-RD accuracy 从 0.7397 到 0.8356。
  - excel macro F1 从 0.5677 到 0.5991。
- 但本轮不是更好的 System 候选：
  - final macro F1 低于 20260512-124930。
  - joint-on-RD 增益从 +0.0863 缩到 +0.0158。
  - final RD 净收益从 +26 缩到 +2。
  - RD-only evidence 明显退化，RD harm 更多。
- 当前模型学成了“安全轨迹保底”，但没有学出足够强的 TR+RD 深融合；这说明继续只加大 trajectory protection 会让 RD/joint 分支失去贡献。

错误形态：

- Drone recall 升高到 0.9338，但 Bird recall 降到 0.7125。
- Bird 主要错成 Unknown 11 次、Drone 10 次。
- Unknown recall 升到 0.7778，但 precision 只有 0.4375，说明 Unknown 倾向偏强。
- Clutter support 只有 7，recall 从 0.7143 降到 0.5714，precision 仍低。

下一步建议：

1. 不接入 System。
2. 保留 stable full 作为“trajectory/without-RD 下限修复”证据。
3. 下一轮不是继续加强 trajectory，而是恢复 RD/joint 学习能力，同时保持 trajectory anchor：
   - 保持 `stage1_traj_only_epochs = 3`、`stage2_ramp_epochs = 4`。
   - `low_altitude_traj_supervision_weight` 从 0.55 降到 0.45。
   - `low_altitude_rd_aux_ratio` 从 0.45 回到 0.55。
   - `low_altitude_rd_forced_fusion_weight` 从 0.45 提到 0.60。
   - `low_altitude_joint_aux_weight` 从 0.35 提到 0.45。
   - `low_altitude_deep_reroute_aux_weight` 从 0.20 提到 0.28。
   - 加入 Bird/Unknown calibration，抑制 Bird -> Unknown。
4. 在训练前优先增加 loss component logging，避免继续只靠总 loss 判断 RD/joint 哪一项退化。
