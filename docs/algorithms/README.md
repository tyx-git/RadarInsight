# Algorithms Guide

这组文档面向第一次接触本项目的同学。目标不是做论文式推导，而是帮助你快速看懂三件事：

- 这个项目从原始 `.mat` 雷达数据到最终分类结果，中间经历了哪些核心算法步骤。
- 每个算法模块的输入、输出、作用和源码落点分别是什么。
- 当你要改代码、调模型、查问题时，应该优先看哪些文件。

## 建议阅读顺序

1. `docs/algorithms/01_real_data_preparation.md`
2. `docs/algorithms/02_tr_track_features.md`
3. `docs/algorithms/03_dpl_to_rd_fft.md`
4. `docs/algorithms/04_rd_micro_motion_representation.md`
5. `docs/algorithms/05_x3d_m_encoder.md`
6. `docs/algorithms/06_track_tcn_transformer.md`
7. `docs/algorithms/07_cross_attention_fusion.md`
8. `docs/algorithms/08_session_split_and_labels.md`
9. `docs/algorithms/09_training_loss_and_reporting.md`
10. `docs/algorithms/10_cnn_fundamentals.md`
11. `docs/algorithms/11_temporal_neural_network_fundamentals.md`
12. `docs/algorithms/12_transformer_backbone_in_project.md`
13. `docs/algorithms/13_rd_strong_modeling_stages.md`
14. `docs/algorithms/14_end_to_end_pipeline_walkthrough.md`
15. `docs/algorithms/15_radar_signal_processing_basics.md`
16. `docs/algorithms/16_fft_windowing_and_resolution.md`
17. `docs/algorithms/17_complex_signal_iq_and_real_imag.md`
18. `docs/algorithms/18_generalization_overfitting_and_session_split.md`
19. `docs/algorithms/19_optimizer_regularization_and_model_selection.md`
20. `docs/algorithms/20_project_data_files_and_formats.md`
21. `docs/algorithms/21_how_to_read_a_session.md`
22. `docs/algorithms/22_how_to_debug_training_regressions.md`
23. `docs/algorithms/23_how_to_read_confusion_matrix.md`
24. `docs/algorithms/24_how_to_read_artifacts_outputs.md`
25. `docs/algorithms/25_annotation_basics_and_current_project_state.md`
26. `docs/algorithms/26_high_quality_dataset_and_precise_annotation.md`
27. `docs/algorithms/27_cvat_usage_for_radar_annotation.md`
28. `docs/algorithms/28_project_labeling_workflow_and_best_practices.md`
29. `docs/algorithms/29_export_rd_clips_for_cvat.md`
30. `docs/algorithms/30_clip_level_annotation_specification.md`
31. `docs/algorithms/31_generalization_learning_in_this_project.md`
32. `docs/algorithms/32_how_to_judge_real_generalization_improvement.md`
33. `docs/algorithms/33_what_is_a_session_and_why_it_matters.md`
34. `docs/algorithms/34_session_clip_track_sample_label_relationships.md`
35. `docs/algorithms/35_how_to_read_training_history_and_choose_the_next_experiment.md`

## 建议同时打开的源码入口

- `src/prodata/mat_session_builder.py`
- `src/training/dataset.py`
- `src/models/cnn_encoder.py`
- `src/models/temporal_encoder.py`
- `src/models/fusion_classifier.py`
- `src/training/session_manager.py`
- `src/training/trainer.py`
- `src/training/evaluator.py`
- `manage.py`

## 你应该先建立的整体认知

先把项目想成一条串起来的流水线：

```text
TR.mat / RD.mat(DPL)
    -> pro/raw
    -> prepare-pro
    -> pro/sessions
    -> data/sessions
    -> RadarFusionDataset
    -> Track branch + RD branch
    -> Cross-Attention fusion
    -> Classifier
    -> train / evaluate / report
```

如果你只想快速上手，推荐先读完前 4 篇，再读第 14 篇。这样你会先知道数据是怎么来的、RD 是怎么变成可训练张量的，再去看模型结构就会容易很多。

如果你对深度学习或雷达信号处理基础还不熟，可以继续补读：

- 第 15 篇：雷达信号处理基础
- 第 16 篇：FFT、窗函数和分辨率
- 第 17 篇：复数信号、IQ 和 `real / imag`
- 第 18 篇：泛化、过拟合和 session 切分
- 第 19 篇：优化器、正则化和模型选择
- 第 20 篇：项目中的 CSV / JSONL / HDF5 文件格式

## 当前项目最值得注意的事实

- 真实数据训练主路径目前是**单尺度 `DPL -> RD`**。
- 当前默认 RD 输入是 `10` 通道，既保留局部 ROI，也保留全局上下文与符号不变视角。
- 轨迹分支已经比较成熟，当前主要优化重点仍然在 RD 微动特征链路。
- 标签来源于文件名，是 `session` 级弱监督，不是逐帧精标。
- 当前图表与报告系统只保留真实结果输出到 `reports/real/`。

## 读完这些文档后，你应当能回答的问题

- 为什么这个项目不能只靠轨迹，也不能只靠 RD。
- 为什么 RD 分支要保留复数信息，而不能只保留幅度图。
- 为什么轨迹分支用了 `Residual TCN + Transformer` 而不是单纯 `GRU`。
- 为什么融合层用了双向 `Cross-Attention` 而不是简单拼接。
- 为什么训练和评估一定要按 `session` 切分，而不是随机打散 clip。

如果你读完后还觉得某一段算法跳跃太快，优先回到第 14 篇，它会把整条链重新串一次。

## 标注专题补读

- 第 25 篇：当前项目标签从哪里来、为什么它是弱监督
- 第 26 篇：什么才算高质量数据集、如何做更精准标注
- 第 27 篇：`CVAT` 在本项目里的定位和使用方式
- 第 28 篇：项目标注工作流和最佳实践

## 标注落地补读

- 第 29 篇：如何把 RD clip 导出成适合 `CVAT` 的图片序列或视频
- 第 30 篇：如何设计一份正式的 clip 级标注规范

## 泛化专题补读

- 第 31 篇：当前项目如何进行泛化学习，以及为什么要从数据、表示、训练和评估四层一起约束泛化
- 第 32 篇：如何判断一次训练是不是**真的**提升了泛化能力，而不是只让某个数字更好看

## Session 专题补读

- 第 33 篇：`session` 主题的简短入口说明
- 第 34 篇：合并后的完整版，系统讲清 `session / track / clip / sample / label` 的关系，以及训练参数和常见专业名词

## 额外工具文档

- `docs/algorithms/how_to_use_cvat.md`：更细的 `CVAT` 入门操作说明
- `docs/algorithms/how_to_review_annotations.md`：标注完成后的复核方法
- `docs/algorithms/LinkAll.md`：从信号采集到 RD / 轨迹进入模型的全链路串联说明

## 实验分析补读

- 第 35 篇：训练完成后如何阅读 `training_history.json`、`evaluation_report.json`，以及如何决定下一轮实验该改哪里
