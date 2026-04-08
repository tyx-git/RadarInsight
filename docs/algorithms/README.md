# 算法文档索引
这组文档面向第一次接触本项目的读者，目标不是做论文式推导，而是帮助你快速回答三件事：第一，项目从真实雷达 `.mat` 数据到最终分类结果，中间经历了哪些关键算法步骤；第二，每个算法模块的输入、输出、作用和源码落点分别是什么；第三，当你要改代码、调模型、查问题时，应该先看哪些文件。
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
36. `docs/algorithms/36_current_best_model_full_explanation.md`
37. `docs/algorithms/37_training_parameters_and_metrics_explained.md`
38. `docs/algorithms/38_how_202603252213_5_4_cpu_calibrated_model_works.md`
## 建议同时打开的源码入口
- `src/prodata/mat_session_builder.py`
- `src/training/dataset.py`
- `src/models/cnn_encoder.py`
- `src/models/temporal_encoder.py`
- `src/models/fusion_classifier.py`
- `src/training/session_manager.py`
- `src/training/trainer.py`
- `src/training/evaluator.py`
- `src/training/calibration.py`
- `manage.py`
## 先建立的整体认识
建议先把项目想成一条连续的数据流水线：
```text
TR.mat / RD.mat(DPL)
-> pro/raw
-> prepare-pro
-> pro/sessions
-> data/sessions
-> RadarFusionDataset
-> Track branch + RD branch
-> Cross-Attention fusion
-> Hierarchical classifier
-> Calibration
-> train / evaluate / report
```
如果你只想快速上手，优先读前 4 篇，再读第 14 篇和第 36 篇。这样你会先知道数据是怎么来的、RD 是怎么变成可训练张量的、当前最优模型为什么这样设计，然后再回头看代码会轻松很多。
## 基础补读
如果你对深度学习或雷达信号处理基础还不熟，建议补读这些文档：
- 第 10 篇：CNN 基础
- 第 11 篇：时序神经网络基础
- 第 12 篇：项目里的 Transformer 骨干
- 第 15 篇：雷达信号处理基础
- 第 16 篇：FFT、窗函数和分辨率
- 第 17 篇：复数信号、IQ 与 `real / imag`
- 第 18 篇：泛化、过拟合与 `session` 切分
- 第 19 篇：优化器、正则化和模型选择
- 第 20 篇：CSV / JSONL / HDF5 文件格式
## 当前项目最值得注意的事
- 当前训练只面向真实环境，不再保留 fake 数据和 fake 报告链路。
- 当前最优模型已经不是简单双分支拼接，而是“轨迹分支 + RD 分支 + 模态可靠性融合 + 层级分类 + 组内校准”。
- 当前 RD 主链路既看目标对齐的局部 ROI，也看全局上下文，不再是单一幅度图输入。
- 当前标签仍然主要来自 `session` 级弱监督，因此看结果时一定要结合混淆矩阵和 `macro_f1`，不要只看 `accuracy`。
- 当前图表和正式输出只写入 `reports/real/`。
## 读完后你应该能回答的问题
- 为什么这个项目不能只靠轨迹，也不能只靠 RD？
- 为什么 RD 分支要保留复数信息和多通道表示，而不是只保留幅度图？
- 为什么轨迹分支使用 `Residual TCN + Transformer` 而不是单一 RNN？
- 为什么融合层要用 `Cross-Attention` 和模态可靠性门控，而不是简单拼接？
- 为什么分类头要分成粗类和细类？
- 为什么后面还要做组内校准？
- 为什么训练和评估必须按 `session` 切分，而不是随机打散 clip？
如果你读完后仍觉得某一段跳得太快，优先回到第 14 篇，它会重新把整条链从头串一次。
## 标注专题补读
- 第 25 篇：当前项目标签从哪里来，为什么属于弱监督
- 第 26 篇：什么才算高质量数据集，如何做更精准标注
- 第 27 篇：`CVAT` 在本项目里的定位与使用方式
- 第 28 篇：项目标注工作流与最佳实践
- 第 29 篇：如何把 RD clip 导出成适合 `CVAT` 的图片或视频
- 第 30 篇：如何设计正式的 clip 级标注规范
- `docs/algorithms/how_to_use_cvat.md`：更细的 `CVAT` 新手操作说明
- `docs/algorithms/how_to_review_annotations.md`：标注完成后的复核方法
## 泛化专题补读
- 第 31 篇：当前项目如何做泛化学习
- 第 32 篇：如何判断一次训练是否真的提升了泛化能力
## Session 专题补读
- 第 33 篇：什么是 `session`，为什么它如此重要
- 第 34 篇：`session / track / clip / sample / label` 的关系，以及这些概念在训练里的作用
## 实验分析补读
- 第 23 篇：如何阅读混淆矩阵
- 第 24 篇：如何阅读 `artifacts/` 下的各类产物
- 第 35 篇：训练结束后如何读 `training_history.json`、`evaluation_report.json`，并决定下一轮实验改哪里
- 第 37 篇：训练参数与指标到底是什么意思
## 推荐最后阅读的总串联文档
- `docs/algorithms/LinkAll.md`：从信号采集、数据处理、标注、筛选到 RD 和轨迹进入模型的全链路串联说明
- `docs/algorithms/36_current_best_model_full_explanation.md`：当前最优模型的完整讲解
