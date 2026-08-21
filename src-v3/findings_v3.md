# RadarInsight v3发现记录

## Finding F001

- Severity：High
- 文件位置：`data/processed/expert1_cq08_rd_physical_anchor_manifest_v6.summary.json`、`data/processed/expert1_realtime_causal_window_cq08_physical_anchor_map_v6.summary.json`
- 问题或结论：旧CQ08 RD anchor/window链路将RD标为`track_point_anchor`，窗口平均RD覆盖率为0，融合可用窗口为0；不能作为本次TR/RD融合契约。
- Evidence：实际摘要分别报告`rd_time_sync_status_counts.track_point_anchor=29413`、`rd_coverage_ratio_mean=0.0`和`rd_fusion_eligible_window_count=0`。
- Impact：直接复用旧window映射会退化为TR-only，无法验证RD是否改善分类或Clutter/Unknown。
- Action：以TR点为事件，按`track_point_index`聚合canonical RD帧；显式保存RD缺失、部分覆盖、顺序异常和局部CPI质量。
- 验证状态：已确认问题；V0点级契约已生成并通过纯度/shape校验，旧链路仍未复用。
- 关联版本：RADAR-LN3-V0。

## Finding F009

- Severity：High
- 文件位置：`data/processed/RADAR-LN3-V0-cq08-tr-rd-point-index.csv`
- 问题或结论：用户选择方案1；V0保持不变，主实验排除3个异常parent，异常样本进入独立审计集。
- Evidence：V0字段聚合实际得到`cq08|track|634`、`cq08|track|785`、`cq08|track|3338`；三者均在train，点数分别21、25、15；异常原因与标签已核实。
- Impact：主训练parent数从1549变为1546，训练parent从1084变为1081；Bird减少2个parent，Drone减少1个parent；正式指标分母和类别权重必须以V1主视图为准，审计集不得进入训练。
- Action：生成`RADAR-LN3-V1`主index、异常审计index、split、manifest、数据契约和配置；V0文件只读保留。
- 验证状态：策略已由用户确认；V1主/审计视图、配置、Dataset/DataLoader和审计forward-only均已验证。
- 关联版本：RADAR-LN3-V1。

## Finding F013

- Severity：Medium
- 文件位置：`src-v3/smoke/contract_tests.py`、`config/RADAR-LN3/RADAR-LN3-V1-cq08-tr-rd-anomaly-audit-config.json`
- 问题或结论：V1主配置的完整契约测试通过；审计配置的训练入口按设计硬拒绝参数更新；一次契约测试中的`forward_ms≈125.0s`不是有效部署延迟测量。
- Evidence：V1 contract test实际通过shape/loss/gradient/checkpoint/causal/online检查；audit `smoke.run_smoke`退出码1并报告`allow_parameter_updates=false`；CUDA环境中单次计时受初始化/同步/资源状态影响，不能代表P95。
- Impact：主/审计隔离可靠；但仍不能输出端到端或稳定模型P95。
- Action：正式部署验收需另建固定预热、真实RD摄入、事件聚合、GPU传输、模型推理和状态更新的独立benchmark；当前保持未执行。
- 验证状态：隔离和功能已验证；延迟待专门测量。
- 关联版本：RADAR-LN3-V1。

## Finding F010

- Severity：High
- 文件位置：`data/processed/RADAR-LN3-V1-cq08-tr-rd-main-index.csv`、`data/processed/RADAR-LN3-V1-cq08-tr-rd-anomaly-audit-index.csv`、`data/processed/RADAR-LN3-V1-cq08-tr-rd-data-contract.json`
- 问题或结论：V1主实验视图排除3个异常parent，独立审计视图保留全部异常parent；两者是V0的完整互斥分区。
- Evidence：main=1546 parent/29663点/105843 RD帧；audit=3 parent/61点/183 RD帧；V0 hash provenance一致；主/审计sample_id不重叠且并集覆盖V0。
- Impact：正式主指标分母固定为V1 main；异常审计指标只能报告在V1 anomaly-audit范围，不能并入主训练或主指标。
- Action：正式实验使用V1 main；审计配置只读forward；保留V0作为基线契约。
- 验证状态：已验证。
- 关联版本：RADAR-LN3-V1。

## Finding F011

- Severity：High
- 文件位置：`config/RADAR-LN3/RADAR-LN3-V1-cq08-tr-rd-anomaly-audit-config.json`、`src-v3/smoke/run_smoke.py`、`src-v3/smoke/audit_forward_test.py`
- 问题或结论：异常审计集已设置参数更新硬保护，不能被误用于训练。
- Evidence：审计配置实际加载时`allow_parameter_updates=false`；audit forward test在CUDA上对3个parent全部输出有限`[1,5]` logits；run_smoke遇到该配置会拒绝参数更新。
- Impact：审计结果与主训练权重隔离；审计集只能用于异常行为、数值稳定性和错误模式分析。
- Action：正式训练命令只允许引用V1 main配置；审计配置只用于forward/evaluation。
- 验证状态：已验证。
- 关联版本：RADAR-LN3-V1。

## Finding F012

- Severity：Medium
- 文件位置：`artifacts/RADAR-LN3-V1-cq08-tr-rd-main-smoke/checkpoint.pt`、Stage 017过程记录
- 问题或结论：V1 main短时smoke通过，但不能作为正式分类性能或训练放行结果。
- Evidence：CUDA 2-step smoke loss=`1.451975703239441→1.5055841207504272`，gradient有限，window logits shape `[2,5]`，checkpoint保存成功；没有运行正式训练和验证评估。
- Impact：当前只能证明V1主集链路可运行，不能证明90%目标、Clutter/Unknown改善或Drone/Bird混淆改善。
- Action：等待用户确认正式训练预算、seed、验证范围和端到端延迟测量后再启动。
- 验证状态：smoke已验证，正式性能待评估。
- 关联版本：RADAR-LN3-V1。

## Finding F002

- Severity：High
- 文件位置：`data/manifests/cq08_availability_summary.json`、`data/manifests/cq08_split_policy.json`
- 问题或结论：CQ08可用范围为1549个track group，标签为五分类；同track的轨迹、点迹、RD文件必须保持同一split。
- Evidence：实际摘要列出Drone 700、Bird 350、Balloon 350、Clutter 50、Unknown 100；split policy明确`track_id`分组。
- Impact：点级样本不能独立随机切分，否则会造成同目标泄漏；V0只保留parent/track级分组字段。
- Action：在点级契约中保留`parent_sample_id`、`track_id`、`split`和来源纯度字段；split校验按parent/track执行。
- 验证状态：已确认事实；V0 parent/track split校验已通过。
- 关联版本：RADAR-LN3-V0。

## Finding F003

- Severity：Medium
- 文件位置：`data/processed/cq08_rd_physical_canonical_frame_index_v3_1.csv`
- 问题或结论：现有canonical RD索引含`track_point_index`和物理canonical tensor引用，可作为只读RD输入；其`rd_frame_time_seconds`属于RD局部帧时间，不能替代TR墙钟时间。
- Evidence：CSV表头及首行实际包含`track_point_index`、`rd_frame_time_seconds`、`canonical_rd_shape=21x384`和tensor路径。
- Impact：液态模块必须沿TR点轴使用TR真实dt；RD局部时间只能作为点内质量/局部时间特征。
- Action：V0契约分离`tr_time`/`tr_dt`与`rd_frame_time_seconds`，并记录RD帧数、局部时长和缺失状态。
- 验证状态：已确认设计约束；V0字段校验和模型dt契约测试已通过。
- 关联版本：RADAR-LN3-V0。

## Finding F004

- Severity：Medium
- 文件位置：工作树状态
- 问题或结论：工作树已有大量用户修改和未跟踪实验文件。
- Evidence：`git status --short --branch`实际输出包含多个`M`、`D`和`??`条目。
- Impact：不能使用回滚、清理或覆盖式迁移；新实现必须隔离在`src-v3`，新数据/配置使用新路径和版本名。
- Action：保持既有内容不变；本次只新增v3范围文件。
- 验证状态：已确认，持续遵守。
- 关联版本：不适用。

## Finding F005

- Severity：High
- 文件位置：`data/manifests/cq08_availability_summary.json`、`data/processed/cq08_track_index.csv`、`data/processed/RADAR-LN3-V0-cq08-tr-rd-summary.json`
- 问题或结论：旧`cq08_availability_summary.json`的Drone=700与当前统一manifest、标准化track index和V0实际范围Drone=699不一致；不是V0漏掉parent。
- Evidence：交叉核对实际得到统一manifest usable=1549且标签为Drone 699；track index=1549且标签为Drone 699；两者track_id集合完全相同、missing/extra均为空；只有旧availability摘要写700。
- Impact：正式指标、类别权重和报告分母必须使用V0实际可读范围，不能引用旧摘要的700。
- Action：V0保持Drone 699，不递增版本；正式实验报告固定使用V0 hash和实际标签计数；如需修正旧摘要，另行授权且不能覆盖旧文件。
- 验证状态：已确认历史摘要不一致，V0数据范围已闭合。
- 关联版本：RADAR-LN3-V0。

## Finding F006

- Severity：High
- 文件位置：`OriData/23/CQ-08/航迹/Tracks_634_3_21.txt`、`data/processed/cq08_tracks/cq08__track__634.csv`、`data/processed/cq08_rd_physical_canonical_frame_index_v3_1.csv`
- 问题或结论：track 634存在真实TR时间回退及RD point index顺序异常；track 3338存在一次TR负dt；不能把这些异常当作正常连续时间。
- Evidence：634原始时间序列含`18:58:02→18:58:00`，标准化文件对应`tr_dt=-2.0`；RD索引含`12→20→13`。V0保留`tr_dt_is_negative`、`tr_time_order_status`、`rd_parent_order_status`等字段。
- Impact：液态状态更新若直接使用负dt会产生错误动力学；点级RD聚合若按物理排序会把源事件对应关系静默改写。
- Action：模型入口将负dt裁剪为安全值并暴露clamp mask；默认smoke/训练样本可排除异常parent，但V0索引不删除原始事件。
- 验证状态：输入异常、模型clamp、负dt可观察性和默认异常parent排除均已验证；V0仍保留原始事件。
- 关联版本：RADAR-LN3-V0。

## Finding F007

- Severity：Medium
- 文件位置：`src-v3/radar_ln3/model.py`、`src-v3/radar_ln3/streaming.py`、`src-v3/smoke/contract_tests.py`
- 问题或结论：最小在线因果接口已实现，液态状态和目标级稳定状态分离，能够输出窗口级临时结果与目标级稳定结果。
- Evidence：contract test实际通过`online point logits [1,5]`、DataLoader、negative dt clamp、stable state和checkpoint round-trip；online wrapper每次只消费一个TR点并保留hidden state。
- Impact：可以进入受控的后续模块化实验；不能由该接口测试推断分类性能提升或生产P95达标。
- Action：后续实验固定V0数据、split、特征和loss，只替换现有因果时序编码器；增加真实服务摄入/RD解码/聚合的端到端P95测量。
- 验证状态：已验证接口；正式实验待执行。
- 关联版本：RADAR-LN3-V0。

## Finding F008

- Severity：Medium
- 文件位置：`src-v3/smoke/latency_smoke.py`、Stage 011过程记录
- 问题或结论：模型-only CUDA focused forward在小样本batch上测得P50约25.27ms、P95约28.80ms，但完整30次重复在180秒预算内超时；不能将focused结果当成端到端P95≤2秒。
- Evidence：`--iterations 3`实际至少执行5次并输出上述数值；`--iterations 30`实际退出码124，命令超时。
- Impact：真实服务的RD读取、CPI聚合、GPU传输、模型前向、目标状态更新和调度开销仍未量化；部署验收未完成。
- Action：后续建立固定批次/单目标在线基准，覆盖端到端事件摄入和真实RD数据路径，再按P95≤2秒验收。
- 验证状态：focused模型-only已验证；端到端待验证。
- 关联版本：RADAR-LN3-V0。

## Finding F014

- Severity：Medium
- 文件位置：`src-v3/smoke/`、`src-v3/radar_ln3/`
- 问题或结论：Smoke、契约、审计和focused诊断代码已与正式`radar_ln3`链路隔离；Smoke入口不再直接放在`src-v3`根目录。
- Evidence：实际目录核验显示Smoke脚本均位于`src-v3/smoke/`；从`src-v3`目录运行`python -m smoke.static_isolation_check`通过，违规0；全量Python编译通过；V1主Smoke和异常审计forward均从新模块路径通过。
- Impact：正式链路目录不再混入有限步训练和诊断入口；Smoke checkpoint仍按artifact语义保存在`artifacts/`，不是正式模型产物。
- Action：后续所有Smoke/contract/latency/audit辅助代码继续写入`src-v3/smoke/`；正式Dataset、模型、trainer和evaluator只写入`src-v3/radar_ln3/`或正式入口目录。
- 验证状态：已确认。
- 关联版本：RADAR-LN3-V1；数据版本未变化。

## Finding F015

- Severity：High
- 文件位置：`src-v3/train/formal.py`、`src-v3/report/classification.py`、`config/RADAR-LN3/RADAR-LN3-V2-cq08-tr-rd-formal-supervised-config.json`
- 问题或结论：正式训练前置代码、V2配置和隔离入口契约已完成；当前可以进入“正式训练授权等待”，但不能把Smoke结果当作正式性能或自动启动全量训练。
- Evidence：V2配置固定V1 main index、train-only统计、seed=42、20 epochs、early stopping=5和`parent_macro_f1`选优；隔离`smoke.formal_entry_contract`实际输出`status=passed`、CUDA、1 step、checkpoint schema通过、focused validation parent=1；entry-contract artifact已生成。
- Impact：训练前置技术阻塞已解除；分类指标、Drone/Bird混淆、Clutter/Unknown能力、test和P95仍没有真实结果。
- Action：等待用户明确授权后执行正式监督训练和val；保持entry-contract checkpoint仅作契约证据，不作为正式初始化或性能模型。
- 验证状态：正式入口和隔离契约已验证；正式训练和性能未执行。
- 关联版本：RADAR-LN3-V2；数据复用RADAR-LN3-V1。

## Finding F016

- Severity：Medium
- 文件位置：`src-v3/radar_ln3/model.py`、`src-v3/radar_ln3/checkpoint.py`、`src-v3/radar_ln3/config.py`
- 问题或结论：当前监督训练技术上不要求预训练；模型支持随机初始化直接优化，配置没有预训练checkpoint字段，Smoke checkpoint只是有限步随机初始化产物，不能视为预训练权重。
- Evidence：`RadarLiquidClassifier`由构造函数直接创建参数；`smoke.run_smoke`直接构造模型、创建AdamW并执行参数更新；配置`TrainConfig`只有batch、smoke_steps、learning_rate、seed、device、checkpoint_path和更新许可，没有pretrain/load字段；checkpoint schema仅保存当前模型和可选optimizer状态。
- Impact：不需要先建立独立预训练阶段才能开始正式监督训练；但也不能复用旧`src`的TR-only或RD-only checkpoint替代当前TR/RD融合模型，旧checkpoint的输入/结构/数据契约不一致。
- Action：默认采用V1 main随机初始化监督训练；若后续要引入预训练，必须单独定义预训练任务、数据范围、权重迁移映射和新的配置/版本语义，不能隐式加载旧权重。
- 验证状态：已确认当前代码事实；预训练是否有收益属于待验证假设。
- 关联版本：RADAR-LN3-V1。

## Finding F017

- Severity：Medium
- 文件位置：`data/processed/RADAR-LN3-V1-cq08-tr-rd-main-index.csv`、`config/RADAR-LN3/RADAR-LN3-V2-cq08-tr-rd-formal-supervised-config.json`
- 问题或结论：V1主数据具备直接监督训练的序列范围和验证类别覆盖；正式监督训练不需要预训练，隔离契约已通过，当前仅等待用户授权。
- Evidence：实际读取V1主索引得到1546个parent、29663个TR点，train/val/test为1081/232/233个parent，序列长度12–29（中位18），五个类别均在val出现；V2配置没有预训练加载字段，训练入口直接随机初始化模型；隔离Smoke在CUDA上输出`status=passed`。
- Impact：不必增加未验证的预训练变量；可以在授权后直接进入V1 main监督训练，但当前不能把1 step边界产物当作性能结果。
- Action：以V1 main为训练与验证范围，审计集不参与训练；预训练保持未执行、非阻塞；等待用户正式训练授权。
- 验证状态：数据覆盖、不需要预训练和隔离契约已确认；正式训练效果待验证。
- 关联版本：RADAR-LN3-V1；训练配置关联RADAR-LN3-V2。

## Finding F018

- Severity：High
- 文件位置：`src-v3/radar_ln3/checkpoint.py`、`src-v3/radar_ln3/contracts.py`、`src-v3/smoke/formal_entry_contract.py`、`artifacts/RADAR-LN3-V2-cq08-tr-rd-formal/`
- 问题或结论：V2正式checkpoint schema、数据契约校验和隔离artifact入口契约均已通过；entry-contract checkpoint仍是有界契约产物，不是正式模型。
- Evidence：`FORMAL_CHECKPOINT_SCHEMA=RADAR-LN3-V2-formal-checkpoint-v1`；V2 config/data/stats hash匹配；本次Smoke输出`training_steps=1`、`window_logits_shape=[2,5]`、weighted loss=`1.3724843263626099`、focused validation parent=1；`artifacts/RADAR-LN3-V2-cq08-tr-rd-formal-entry-contract/`包含best/last checkpoint和summary。
- Impact：正式训练入口已具备随机初始化、更新、保存和恢复条件；但1 step和focused validation不能推断正式分类性能、稳定性、P95或test结果。
- Action：正式训练时使用V2正式配置和`artifacts/RADAR-LN3-V2-cq08-tr-rd-formal/`正式产物语义；entry-contract目录只保留为契约审计证据，不作为预训练权重。
- 验证状态：已验证；正式训练、完整val、test和P95未执行。
- 关联版本：RADAR-LN3-V2；数据复用RADAR-LN3-V1。

## Finding F019

- Severity：Medium
- 文件位置：`src-v3/smoke/formal_entry_contract.py`、GPU运行环境
- 问题或结论：此前GPU占用导致的Smoke超时已被后续GPU空闲运行覆盖；隔离正式入口Smoke现已通过。
- Evidence：此前命令曾以退出码124超时；后续同一契约命令在CUDA上输出`status=passed`，实际生成entry-contract artifact，checkpoint和summary均存在。
- Impact：资源占用不再阻塞Stage 018；仍不能由Smoke推断正式训练性能、吞吐或端到端P95。
- Action：正式训练前继续记录GPU资源窗口；正式训练和P95使用独立命令、独立产物和真实评估范围。
- 验证状态：历史超时已解决；正式性能和P95待验证。
- 关联版本：RADAR-LN3-V2。

## Finding F020

- Severity：Low
- 文件位置：`src-v3/build/`、`src-v3/train/`、`src-v3/report/`、`src-v3/validate/`、`src-v3/smoke/`
- 问题或结论：`src-v3`入口已按职责完成目录和命名规范化；旧入口文件已移除，Smoke保持独立目录，未改变运行语义。
- Evidence：新入口共7个；旧入口路径存在数为0；全量Python编译通过；`smoke.static_isolation_check`通过且`violations=[]`；7个新模块入口的`--help`均通过；移动后默认项目根目录已统一为`parents[2]`；Smoke训练契约改为`from train.formal import train_formal`。
- Impact：后续构建、训练、报告和校验命令不再混在`src-v3`根目录；历史旧命令不能直接继续使用，必须使用新的模块化入口。
- Action：后续命令统一使用`python -m build.*`、`python -m train.*`、`python -m report.*`和`python -m validate.*`；旧路径仅在过程记录中保留审计映射，不创建兼容副本。
- 验证状态：目录规范化已验证；本次重排未触发数据重建；后续Stage 018正式入口Smoke已通过；正式训练和完整评估未执行。
- 关联版本：不递增；RADAR-LN3数据、配置、模型和artifact语义未变化。

## Finding F021

- Severity：Medium
- 文件位置：`src-v3/radar_ln3/logging_utils.py`、`src-v3/train/formal.py`、`src-v3/smoke/logging_test.py`
- 问题或结论：训练日志模块已实现，支持artifact文件日志和stderr进度输出；正式训练已实际完成，但该次训练发生在本次输出契约修改之前；其best checkpoint随后已用新报告入口独立复核。
- Evidence：日志Smoke实际通过，验证`status=passed`、固定分隔格式、`format_duration(65)="01m05s"`、541 step时进度间隔为50、重复配置未产生重复handler；正式旧日志记录epoch 15 early stopping、best epoch=10、global step=8115；全量编译和src-v3隔离检查通过。
- Impact：新训练日志会输出总体/分类别指标和低置信度样本；旧正式日志没有本次新增的末尾指标段，不能冒充新格式；完整样本明细需要单独生成。
- Action：保留旧best checkpoint不变；使用新`report.classification`对旧best checkpoint运行val，生成新格式`evaluation-val.json`和`sample_detail_information.json`，无需重训。
- 验证状态：日志模块、独立Smoke和新格式对旧best checkpoint的GPU真实val报告均已验证；正式训练已执行。
- 关联版本：不递增；数据、配置、模型输入输出和checkpoint schema未变化。

## Finding F022

- Severity：High
- 文件位置：`artifacts/RADAR-LN3-V2-cq08-tr-rd-formal/training-summary.json`、`artifacts/RADAR-LN3-V2-cq08-tr-rd-formal/best_model.pt`
- 问题或结论：正式训练已完成，epoch 15触发early stopping，最佳epoch为10；旧训练输出格式已保留为legacy，当前正式summary和独立样本明细已迁移，并由GPU best checkpoint val复核。
- Evidence：实际读取summary得到`status=passed`、`history=15`、`global_step=8115`、`best_epoch=10`、parent Macro-F1=`0.703274980916419`、stop reason=`early_stopping`；旧summary约31 MB且嵌入完整validation records；best checkpoint SHA256=`CF22838CD723D6C77BDC9681C49D6CEDF9CCA53308B90D99DF704B4F3FF0F458`。
- Impact：训练结果可用于V1 main val分析；不能声称其为test、session或线上P95结果。
- Action：不重训、不覆盖best checkpoint；运行新报告入口对best checkpoint执行V1 val，生成不含records的`evaluation-val.json`和包含全部样本明细的`sample_detail_information.json`；必要时再生成新的summary兼容副本，避免直接覆盖历史训练记录。
- 验证状态：正式训练结果、输出契约迁移和GPU best checkpoint独立val均已验证；test和P95未执行。
- 关联版本：RADAR-LN3-V2；无数据/配置版本递增。

## Finding F023

- Severity：Medium
- 文件位置：`artifacts/RADAR-LN3-V2-cq08-tr-rd-formal/training-summary.json`、`sample_detail_information.json`、`training-summary-legacy.json`、`src-v3/report/migrate_training_summary.py`
- 问题或结论：既有正式训练结果已安全迁移到新输出契约；summary不再输出每个样本记录，所有样本明细单独保存，低置信度阈值为0.60。
- Evidence：迁移命令退出码0；新summary不含`best_checkpoint_path`和`records`，包含`best_checkpoint`；明细schema=`RADAR-LN3-V2-sample-detail-v1`，保存window=4474、parent=232、stable=232；低置信度window/parent/stable分别为641/34/34；legacy原始summary保留。
- Impact：现有best checkpoint可以直接按新路径和明细文件消费，无需重训；但迁移结果来自历史best epoch记录，不是重新forward计算。
- Action：如需最终独立核验，运行`python -m report.classification`对best checkpoint执行val；不读取test，除非另行授权。
- 验证状态：迁移、文件契约和独立GPU best checkpoint val复核均已验证；test/P95未执行。
- 关联版本：RADAR-LN3-V2；无数据/配置版本递增。

## Finding F024

- Severity：High
- 文件位置：`artifacts/RADAR-LN3-V2-cq08-tr-rd-formal/evaluation-val.json`、`sample_detail_information.json`、`best_model.pt`
- 问题或结论：best checkpoint已在GPU上完成独立V1 main validation复核，GPU结果与CPU batch=1复核一致；这是当前唯一可采纳的正式val性能证据，test和线上P95仍未执行。
- Evidence：评估命令退出码0，`device=cuda`，checkpoint epoch=10、global_step=5410、parent=232、window=4474、stable=232；parent Accuracy=`0.8448275862`、Macro F1=`0.7032749809`，window Macro F1=`0.6056502661`，stable Macro F1=`0.6638246874`；数据契约三项hash与训练一致。
- Impact：可以基于真实val分析模型瓶颈；不能把val结果写成test、session或生产在线P95结果。
- Action：冻结本次评估作为baseline；后续所有优化保持同一V1 index、split、评估逻辑和seed对照；test仍需单独授权。
- 验证状态：已验证。
- 关联版本：RADAR-LN3-V2；数据复用RADAR-LN3-V1。

## Finding F025

- Severity：Critical
- 文件位置：`src-v3/train/formal.py`、`src-v3/radar_ln3/evaluator.py`
- 问题或结论：当前训练loss只对每个parent最后一个`window_logits`计算一次，而评估将所有因果前缀`sequence_logits`作为window级结果统计，训练目标与在线窗口级验收目标不一致。
- Evidence：`formal.py`调用`five_class_loss(output["window_logits"], labels, class_weights)`；`evaluator.py`逐事件读取`sequence_logits[batch,event]`并生成4474个causal prefix记录；当前window Macro F1=`0.6056502661`显著低于parent Macro F1=`0.7032749809`。
- Impact：模型可能主要学习最终窗口而非早期窗口；这直接限制窗口级临时结果、短序列和在线前缀性能，不能先归因于液态单元容量不足。
- Action：下一实验优先只改loss为masked causal-prefix supervision或固定后缀采样，保持输入、模型、split、优化器预算和评估逻辑不变；先做短时smoke，再做单变量正式对照。
- 验证状态：已确认代码事实和val现象；改善幅度待验证。
- 关联版本：RADAR-LN3-V2；若改变训练语义，配置应在用户确认后递增新版本。

## Finding F026

- Severity：High
- 文件位置：`data/processed/RADAR-LN3-V1-cq08-tr-rd-main-index.csv`、`src-v3/radar_ln3/model.py`
- 问题或结论：验证集中1796/4474个TR事件的`dt=0`，其中非首点1564个；液态模块将zero-dt替换为0.001秒floor，真实有效时间推进被大量压缩为事件floor。
- Evidence：V1 val索引统计`dt_zero_event_count=1796`，模型评估同样输出`dt_zero_event_count=1796`且`dt_clamped_event_count=0`；`CfCLiquidCausalEncoder._step`在safe_dt小于等于阈值时使用`zero_dt_event_floor_seconds=0.001`。
- Impact：连续时间模型优势无法在这些事件上充分发挥；Bird/Drone短前缀混淆和不同类别zero-dt比例差异可能被时间语义影响。该问题可能是数据时间粒度/索引语义问题，也可能是合理的同一时刻多点事件，尚未完成根因判定。
- Action：先做只读时间审计（原始轨迹时间精度、同timestamp点语义、RD局部时间不能替代TR时间）；随后单变量比较zero-dt处理策略，禁止直接用索引位置伪造dt。
- 验证状态：已确认统计和代码处理；根因及最佳策略待验证。
- 关联版本：RADAR-LN3-V1/V2；若改变dt语义需递增数据契约和配置版本。

## Finding F027

- Severity：High
- 文件位置：`artifacts/RADAR-LN3-V2-cq08-tr-rd-formal/evaluation-val.json`、`sample_detail_information.json`
- 问题或结论：Clutter/Unknown是当前主要分类短板，错误大量流向BirdTarget；低置信度样本与错误高度相关，但固定0.60阈值尚未构成正式拒识能力。
- Evidence：window中Unknown→Bird=147、Clutter→Bird=84；parent Unknown Recall=`0.2667`、F1=`0.3077`，Clutter Precision=`0.4000`、Recall=`1.0000`、F1=`0.5714`，支持数仅Unknown=15、Clutter=8；parent低置信度34/232，其中accepted=198、accepted errors=18，Clutter/Unknown低置信度候选12/23、已知三类低置信度22/209。
- Impact：仅调整液态时间常数不能证明解决开放集/杂波识别；标签边界、类别覆盖、hard negative和拒识标定可能是更直接瓶颈。
- Action：先建立Unknown/Clutter错误审计与拒识基线：固定模型输出，使用验证集内部校准阈值并报告Coverage、Accepted Macro F1、Unknown/Clutter reject recall、FAR/FRR和Selective Risk；再单变量比较类别采样/损失与独立拒识头。不能把低置信度直接改写成Unknown。
- 验证状态：val诊断已验证；拒识指标和改进实验未执行。
- 关联版本：RADAR-LN3-V2；阈值校准若写入配置需新配置版本。

## Finding F028

- Severity：Medium
- 文件位置：`data/processed/RADAR-LN3-V1-cq08-tr-rd-main-index.csv`、`src-v3/radar_ln3/features.py`
- 问题或结论：TR缺失语义被显式保留，但val中height/latitude/longitude三字段全部缺失；这三类特征当前没有提供真实区分信息。
- Evidence：V1 val 4474行均`tr_missing_count=3`，缺失字段固定为`height_m,latitude,longitude`；`encode_tr_feature_values`将缺失值编码为0并返回11维missing mask，模型输入shape为14+11。
- Impact：当前特征契约中名义上的高度/经纬度维度实际由missing mask表达，不能期待其改善类别区分；应避免把缺失字段当作有效空间轨迹证据。
- Action：保留mask并将这些字段作为数据覆盖问题单独审计；优先验证可用range/径向速度/角度/course/speed/SNR与RD空间纹理，必要时设计不依赖未观测地理字段的TR子集对照。
- 验证状态：字段缺失统计和编码路径已验证；替换/删除特征的收益待验证。
- 关联版本：RADAR-LN3-V1/V2；改变输入schema需递增配置/契约版本。

## Finding F029

- Severity：Medium
- 文件位置：`src-v3/radar_ln3/stable.py`、`artifacts/RADAR-LN3-V2-cq08-tr-rd-formal/evaluation-val.json`
- 问题或结论：当前目标级稳定器降低了Drone/Bird双向混淆，但没有改善整体macro指标：window Macro F1=`0.6057`，parent=`0.7033`，stable=`0.6638`；stable不是简单的性能单调提升器。
- Evidence：window Drone→Bird/Bird→Drone=`54/74`，parent=`3/2`，stable=`2/2`；stable Accuracy=`0.8362`低于parent=`0.8448`，stable Macro F1=`0.6638`低于parent=`0.7033`；stable使用EMA decay=0.65、confirm_count=2。
- Impact：当前稳定策略对在线目标级输出的收益主要体现在减少Drone/Bird抖动，可能牺牲Clutter/Unknown或其他类别；不能直接把stable结果作为最终目标级性能提升。
- Action：把稳定器作为独立变量做按类别和延迟约束的校准实验；先固定模型，比较EMA、logit累计、最小证据数和类别条件阈值，报告稳定确认延迟与选择性指标。
- 验证状态：现状已验证；替代稳定策略待验证。
- 关联版本：RADAR-LN3-V2；若改变线上稳定语义需递增配置版本。

## Finding F030

- Severity：Critical
- 文件位置：`src-v3/train/formal.py:167`、`src-v3/radar_ln3/evaluator.py:98-126`
- 问题或结论：V2训练只对每个parent最后一个窗口计算CE，而评估和线上接口消费每个因果prefix的`sequence_logits`，窗口级训练目标与窗口级验收目标不一致。
- Evidence：训练调用`five_class_loss(output["window_logits"], labels, class_weights)`；评估逐事件读取`sequence_logits[batch,event]`形成4474个窗口；parent Macro F1=`0.7032749809`而causal-window Macro F1=`0.6056502661`。
- Impact：早期prefix没有直接监督，窗口级临时结果可能成为训练外推；长parent还可能贡献更多窗口而支配prefix loss。
- Action：V3采用对所有非padding prefix logits的CE，先在parent内平均再跨parent平均；保留V2 final-window分支和artifact，单因素比较。
- 验证状态：问题已确认；V3实现与性能收益待验证。
- 关联版本：RADAR-LN3-V3（待生成）；V2 baseline只读。

## Finding F031

- Severity：High
- 文件位置：`data/processed/RADAR-LN3-V1-cq08-tr-rd-main-index.csv`、`src-v3/radar_ln3/model.py`
- 问题或结论：非首点零dt主要来自CQ08轨迹时间字段的秒级量化，同一秒内多个点共享相同`time_seconds`；当前模型把这些未知间隔替换成0.001秒，未保留有效性语义。
- Evidence：V1主索引非首点zero-dt=9776/28117=34.77%；`cq08__track__1010.csv`时间序列包含`1.0,1.0,1.0...`；原始`Tracks_1010_2_18.txt`经GBK读取后`点时间`只有`HH:MM:SS`；RD时间轴在现有契约中为`cpi_relative_elapsed_not_tr_wall_clock`；`CfCLiquidCausalEncoder._step`对zero dt使用0.001秒floor。
- Impact：连续时间模块把“时间未知/同秒多点”伪装成极短有效时间，可能影响前缀证据累积和Drone/Bird混淆；简单epsilon不能恢复物理dt。
- Action：V3索引增加`tr_dt_valid`和`tr_dt_reason`；首帧和同秒量化间隔显式无效，保留数值floor用于稳定更新并将validity bit送入液态模块；RD CPI时间不替代TR dt。
- 验证状态：根因已确认；V3修复和收益实验未执行。
- 关联版本：RADAR-LN3-V3（待生成）；V1/V2保持只读。

## Finding F032

- Severity：High
- 文件位置：`artifacts/RADAR-LN3-V2-cq08-tr-rd-formal/evaluation-val.json`、`src-v3/radar_ln3/stable.py`
- 问题或结论：Clutter/Unknown是当前主要短板，但当前证据不足以归因于液态时序容量；现有链路没有正式拒识动作。
- Evidence：parent Clutter precision/recall/F1=`0.4000/1.0000/0.5714`、Unknown=`0.3636/0.2667/0.3077`，parent support分别为8和15；window Unknown→Bird=147、Clutter→Bird=84；稳定器只输出五分类EMA结果，没有abstain/reject分支；0.60只用于低置信度日志。
- Impact：增加液态单元复杂度不能修复标签语义、覆盖不足、类别不平衡或开放集拒识缺失；不能宣称Clutter/Unknown已解决。
- Action：V3先保持标签不变，完成原始语义、parent覆盖、质量分层和混淆审计；后续单独校准reject并报告Coverage、Accepted指标、Unknown/Clutter拒识Recall、FAR/FRR和Selective Risk。
- 验证状态：短板已验证；标签质量和拒识改进未执行。
- 关联版本：V2 baseline；拒识或标签语义变化需新版本。

## Finding F033

- Severity：High
- 文件位置：`data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-index.csv`、`data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-data-contract.json`、`src-v3/radar_ln3/model.py`
- 问题或结论：V3已将不可恢复的秒级量化时间显式建模为`tr_dt_valid=false`，但数值floor仍只是事件处理稳定性措施，不是物理时间恢复。
- Evidence：V3 index SHA256=`a5d054f044709047938c88ef5601caf7d4b482ef980a5a61f19db87b6e649f87`；有效dt=18341、无效=11322（首帧1546、zero-delta-at-second-precision=9776）；V3 contract声明`tr_dt_valid`/`tr_dt_reason`；`dt_prefix_test`和V3 formal-entry smoke通过。
- Impact：时间语义不再被静默当作正常dt输入，但V3收益仍需训练后验证；秒级源数据本身仍限制连续时间模型可利用的真实时间尺度。
- Action：先用V3做prefix-loss单变量训练对照；如业务必须利用亚秒时间，需回到采集/原始时间源补充证据，不能由RD CPI时间伪造。
- 验证状态：修复契约和shape/mask已验证；分类收益待验证。
- 关联版本：RADAR-LN3-V3。

## Finding F034

- Severity：High
- 文件位置：`data/manifests/cq08_unified_manifest.json`、`data/processed/RADAR-LN3-V1-cq08-tr-rd-main-index.csv`、`src-v3/radar_ln3/stable.py`
- 问题或结论：Clutter/Unknown标签映射一致且两类均有完整RD点覆盖，但样本覆盖与开放集拒识仍未解决；当前链路没有正式reject/abstain。
- Evidence：只读审计显示index与manifest标签无冲突；Clutter=50 parent、Unknown=100 parent，V1 train/val/test分别为35/8/7和70/15/15；两类验证点均无RD缺失；原始航迹文件只有运动字段、不含类别字段，类别来自manifest/index；session全部可映射且不跨split，capture有27个跨split记录；现有stable模块只有五分类EMA。
- Impact：不能把Clutter/Unknown短板归因于RD缺失或液态时序容量；capture不能直接当split grouping键；低support使弱类指标不稳定，固定0.60仍不是拒识机制。
- Action：保持标签不变，先做train/grouped-OOF拒识校准和parent级质量分层；再单变量比较采样/损失/拒识头；报告Coverage、Accepted Macro-F1、Unknown/Clutter拒识Recall、FAR/FRR、Selective Risk。test继续不读取。
- 验证状态：标签/覆盖只读审计已验证；拒识校准和专项训练未执行。
- 关联版本：V2 baseline/V3数据；拒识或标签规则变化需递增版本。

## Finding F035

- Severity：Medium
- 文件位置：`data/manifests/cq08_unified_manifest.json`、`src-v3/radar_ln3/evaluator.py`
- 问题或结论：session字段实际存在于统一manifest且可映射到全部V1 parent，但当前V1/V2 Dataset/evaluator没有消费session，因此session指标仍未评估；capture存在跨split，不能未经规则审计作为等价session键。
- Evidence：1546个V1 parent均映射session，session不跨train/val/test；capture有27个跨split记录；现有evaluator明确输出session `not_available`。
- Impact：当前parent结果不能替代session级泛化证据；后续session评估可加入报告而不必把session字段送入模型输入，但需先冻结分组契约。
- Action：先为session审计设计只读分组报告，确认是否需要按session重新切分；未经用户确认不改V3 split或训练范围。
- 验证状态：映射事实已验证；session级报告/重切分未执行。
- 关联版本：V1/V2/V3；改变split需新版本。

## Finding F036

- Severity：Medium
- 文件位置：`artifacts/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-formal-legacy-smoke/`、`src-v3/smoke/contract_tests.py`
- 问题或结论：早期普通contract smoke曾将V0 checkpoint schema的step=1临时产物写入V3正式artifact目标路径，存在把Smoke误认成正式训练产物的路径风险。
- Evidence：只读检查显示原文件`checkpoint_schema=RADAR-LN3-V0-checkpoint-v1`、`step=1`、无epoch/global_step；已移动到`artifacts/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-formal-legacy-smoke/`；V3正式artifact路径当前不存在。
- Impact：若不区分路径，会误判V3已训练或误加载错误checkpoint；不会改变V2权重或V3数据契约。
- Action：普通contract smoke固定写入`artifacts/*-contract-smoke/`，V3 formal-entry smoke固定写入`artifacts/*-entry-contract/`；正式训练才允许使用`artifacts/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-formal/`。
- 验证状态：路径边界已修正并复核；正式训练未执行。
- 关联版本：RADAR-LN3-V3。

## Finding F037

- Severity：Critical
- 文件位置：`src-v3/radar_ln3/losses.py`、`src-v3/train/formal.py`、`src-v3/radar_ln3/evaluator.py`、`config/RADAR-LN3/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-formal-supervised-config.json`
- 问题或结论：V3已将训练监督单位改为所有非padding causal prefix，并按parent内平均后跨parent平均；V2的窗口训练目标与窗口评估目标不一致已完成代码级修复，但prefix是否应共享完整parent标签仍未由业务标注契约证明。
- Evidence：V3配置`loss_mode=causal_prefix`；`causal_prefix_loss`消费`sequence_logits[B,L,5]`和`tr_padding_mask`，每个parent有效prefix平均后再跨parent平均；`evaluator`逐有效事件消费同一`sequence_logits`；V3 `contract_tests`和`formal_entry_contract` CUDA退出码0；当前没有V3正式分类指标。
- Impact：已消除“训练只优化最终窗口、线上消费全部前缀”的直接不一致；若早期prefix证据不足而仍使用最终类别标签，仍可能产生监督噪声，影响窗口临时结果、短序列和弱类Recall。
- Action：V3正式训练前冻结prefix标签语义；优先使用固定parent标签作为业务确认方案，或设计有明确起始证据的延迟监督/辅助任务，二者只能单独实验；不得用Liquid复杂度掩盖标签目标风险。
- 验证状态：训练/评估单位一致性已通过focused验证；prefix标签业务有效性和分类收益待验证。
- 关联版本：RADAR-LN3-V3。

## Finding F038

- Severity：High
- 文件位置：`data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-index.csv`、`data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-data-contract.json`、`src-v3/radar_ln3/model.py`
- 问题或结论：CQ08轨迹时间的秒级精度使大量相邻事件间隔不可辨识；V3已显式标记无效dt，但没有恢复真实亚秒时间。
- Evidence：V3 index实际统计zero-dt=11322/29663=`38.16%`；其中首帧1546、非首点秒级量化zero-dt=9776/28117=`34.77%`；负dt=0；contract声明`tr_dt_valid`、`tr_dt_reason`和“数值floor不是物理时间恢复”；模型将`dt_valid`送入fusion并输出`dt_invalid_mask`；`dt_prefix_test`和CUDA contract smoke通过。
- Impact：液态模块只能在18341个有效正间隔上使用可辨识时间推进；其余事件使用稳定性floor，不能证明连续时间建模收益；Bird/Drone及Clutter/Unknown的时间相关改善仍待验证。
- Action：保持V3显式mask；若业务需要亚秒优势，应补充原始采集时间，不得用RD CPI局部时间伪造TR wall-clock；正式实验按dt_valid、长度和类别分层报告。
- 验证状态：来源和契约修复已确认；分类收益、线上时延和原始亚秒时间补充均未验证。
- 关联版本：RADAR-LN3-V3。

## Finding F039

- Severity：High
- 文件位置：`src-v3/radar_ln3/stable.py`、`src-v3/radar_ln3/evaluator.py`、`artifacts/RADAR-LN3-V2-cq08-tr-rd-formal/evaluation-val.json`
- 问题或结论：Clutter/Unknown短板和拒识缺失仍未解决，不能通过增加液态单元复杂度替代标签、覆盖和开放集机制审查。
- Evidence：V2 val parent Clutter support=8、P/R/F1=`40.00/100.00/57.14%`；Unknown support=15、P/R/F1=`36.36/26.67/30.77%`；现有`TargetStableStateMachine`只做五类EMA和确认计数，没有reject/abstain输出；evaluator只报告普通分类和低置信度字段；V3未执行正式训练。
- Impact：当前无法报告Coverage、Accepted Macro-F1、Unknown/Clutter拒识Recall、FAR、FRR或Selective Risk；UnknownTarget不是拒识真值，低置信度也不能改写标签。
- Action：先在train或parent-grouped OOF上确定reject阈值来源，再单独实现/验证reject动作；保持五类标签不变，报告类别、质量、长度和source分层；拒识实验不得读取test或更换液态模块。
- 验证状态：短板和无reject代码事实已确认；标签边界、OOF阈值和专项训练未执行。
- 关联版本：V2 baseline、V3数据；若引入reject语义需新配置/契约版本。

## Finding F040

- Severity：Medium
- 文件位置：`data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-index.csv`、`src-v3/radar_ln3/model.py`、`src-v3/smoke/latency_smoke.py`
- 问题或结论：V3模型-only延迟远低于2秒，但尚无端到端在线P95证据。
- Evidence：CUDA latency smoke实际p50=`25.1672 ms`、p95=`29.4091 ms`、max=`30.9205 ms`，scope明确排除磁盘/RD解码、GPU传输、事件聚合和服务调度；用户验收目标为端到端P95≤2秒。
- Impact：不能据模型-only结果宣布在线部署达标；I/O、RD解码、批调度、目标状态管理和并发可能成为实际瓶颈。
- Action：正式训练和模型选择冻结后，单独执行包含原始RD读取/聚合、GPU传输、forward、stable state update和服务调度的端到端基准；记录p50/p95/p99、并发和显存。
- 验证状态：模型-only focused smoke已验证；端到端P95未执行。
- 关联版本：RADAR-LN3-V3；不触发数据版本递增。

## Finding F041

- Severity：Medium
- 文件位置：`config/RADAR-LN3/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-formal-supervised-config.json`、`src-v3/train/formal.py`、`src-v3/radar_ln3/evaluator.py`
- 问题或结论：V3训练loss与causal-window评估单位已经一致，但checkpoint仍按`parent_macro_f1`选优，stable层指标不参与选优；这是剩余的验收优先级风险，不是V2原有的窗口监督缺失错误。
- Evidence：V3配置`loss_mode=causal_prefix`、`checkpoint_selection_metric=parent_macro_f1`；trainer从`validation_full.parent_metrics`取选优值；evaluator分别输出window、parent和stable三层指标；V3正式训练尚未执行。
- Impact：选出的best checkpoint可能偏向最终parent结果，窗口临时结果或stable延迟/质量不一定最优；在90%硬门槛和在线输出双目标下，单一parent选优不能替代三层报告。
- Action：V3首轮保持当前选优指标以隔离主要变量；正式val后同时报告三层指标。若窗口或stable是主要验收对象，再单独设计选优指标实验，不与loss、dt或拒识同时修改。
- 验证状态：代码事实已确认；对best checkpoint三层指标的影响待V3正式训练验证。
- 关联版本：RADAR-LN3-V3。

## Finding F042

- Severity：Medium
- 文件位置：`data/processed/raw_track_index.csv`、`data/processed/raw_rd_index.csv`、`data/processed/raw_tracks/*.csv`、`data/processed/raw_rd/*_rd_tensor.npz`
- 问题或结论：raw是独立的77条TR/RD配对序列，时间精度明显高于CQ08，但标签集合、字段缺失、RD轴形状和split契约均不同；不能直接混入CQ08 V3五分类训练或指标。
- Evidence：本轮实际遍历得到77序列、11366点、11289相邻对，正dt=10395、zero=894、negative=0，正dt最小约35微秒；raw映射标签为BirdTarget=17、DroneTarget=18、AircraftTarget=21、ShipTarget=21；RD bin=512/1024两种，raw NPZ示例`[512,69]`，CQ08 canonical NPZ示例`[120,21,384]`；raw无split/session字段。
- Impact：raw可用于独立TR/RD连续时间研究，但若与CQ08混合会同时改变类别、时间、输入shape、split和归一化语义，污染当前CQ08主结论；raw也不能直接提供Clutter/Unknown证据。
- Action：保持raw只读、独立报告；如未来纳入训练，必须重新定义标签映射、数据契约、空间前端、split和RADAR-LN3新版本，不能复用V3配置。
- 验证状态：raw只读核查已确认；raw纳入V3训练未执行。
- 关联版本：V3数据未变化；raw不属于V3训练范围。

## Finding F043

- Severity：High
- 文件位置：OriData/23/CQ-08/原始回波/*.dat、OriData/23/CQ-08/batch_process_all.m、OriData/23/CQ-08/MAT/*.mat
- 问题或结论：CQ08原始回波包含可重新解析的复数I/Q和CPI/PRT元数据；12文件抽样已证明现有MAT的I/Q/RD和帧索引与原始解析链一致，但原始回波没有被确认包含显式wall-clock时间，且CPI/PRT结构存在多种取值，不能直接把raw帧序或CPI字段当作TR墙钟时间。
- Evidence：原始目录实际有1550个.dat；MAT目录实际有106026个.mat；MATLAB审计实际选择12个文件、解析1023个原始帧；12/12 sample状态为`ok`；raw帧数与manifest MAT数逐样本一致；36个首/中/末帧`data_out`和复数RD对照最大绝对/相对误差均为0；raw_track_point_index与MAT index_trackPointNo一致；未检测到wall-clock字段和未解析隐藏载荷。1023帧CPIcount=0/1/2/3为368/379/138/138，PRTnum=142/226/360/570/900为66/549/346/58/4；每个TR point对应多个raw/MAT帧。
- Impact：当前MAT/RD不存在由已审计证据支持的帧错配或RD重算错误，不需要为了修复错配而立即重解析或重建V3训练数据；但原始回波时间轴只能确认帧顺序+CPI/PRT局部参数，无法补回TR亚秒wall-clock。若将raw帧直接展开为液态事件，会引入事件粒度、CPI=0语义和多帧/TR点聚合风险；若研究未压缩I/Q相位/微多普勒增益，仍需独立输入契约和实验。
- Action：保持现有MAT/RD作为V3主输入；不覆盖MAT、不重建V3训练索引。将raw仅作为后续受控诊断/增益实验候选：先定义raw frame或CPI token的时间语义、点内聚合和数据版本，再与固定V3 baseline做单变量对照；如需恢复真实亚秒时间，回溯原始采集日志/设备时间源，不能由CPI/PRT或frame index伪造。
- 验证状态：12文件抽样审计已完成并通过；全量1550文件一致性未执行；未压缩I/Q相对canonical RD的信息增益、wall-clock字段恢复和正式训练均未执行。
- 关联版本：V3数据和配置不变；本诊断不递增RADAR-LN3版本。

## Finding F044

- Severity：Medium
- 文件位置：`src-v3/processed/report/RADAR-LN3-original-echo-audit-frames.csv`、`src-v3/processed/report/RADAR-LN3-original-echo-audit-samples.csv`、`data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-index.csv`
- 问题或结论：CQ08原始回波的帧级局部时序结构比当前TR时间字段丰富，但不能直接替代TR事件时间；抽样中存在多种慢时间长度和CPI状态，且多个回波帧映射到同一TR point。
- Evidence：抽样1023帧的I/Q shape为31x142、31x226、31x360、31x570、31x900；PRTnum对应142/226/360/570/900；CPIcount=0共368/1023=`35.973%`；track 1前10帧示例中同一track point 1连续对应多个raw帧，CPIcount在2和3之间交替，后续还出现CPIcount=0/1；raw frame count为MAT count的数倍而不是TR point count。
- Impact：液态模型若沿当前TR点轴运行，仍只能使用V3显式`tr_dt_valid`和TR时间语义；若沿raw帧轴运行，必须重新定义frame token、点内聚合、CPI=0、PRT变化和跨TR点边界，否则会把局部回波采样顺序误当成目标级墙钟时间。当前不能声称原始回波已提供可直接用于长时连续时间建模的真实时间轴。
- Action：主线保持TR+RD融合和现有V3契约；后续只设计“原始I/Q/RD帧级前端+受控时序聚合”的独立实验，不与V3训练同时改变标签、split、TR dt、loss或拒识机制。
- 验证状态：12文件抽样已确认；全量帧级分布、设备时间字段和raw帧级分类收益未执行。
- 关联版本：V3数据和配置不变；后续raw输入实验需新数据契约/配置版本。

## Finding F045

- Severity：Medium
- 文件位置：`src/process/build_cq08_rd_physical_canonical_store.py`第169-172行附近、`src-v3/radar_ln3/data.py`的`_load_rd_event`、`src-v3/processed/report/RADAR-LN3-original-echo-audit-frames.csv`
- 问题或结论：当原始帧`CPIcount<=0`时，现有物理RD构建链把观测时长回退为`PRTnum*PRT`；该回退可用于保持局部RD时间轴单调，但其是否等于真实采集时长尚未由原始设备协议或墙钟时间验证。
- Evidence：12文件抽样1023帧中`CPIcount=0`为368帧，占35.9736%；MATLAB审计对这些帧按原始字段计算的`cpi_count*prt_num*prt_s`为0；现有`build_cq08_rd_physical_canonical_store.py`在`observation_seconds<=0`时执行`observation_seconds=prt_num*prt_s`，随后用该值累加`time_seconds`；V3契约已明确RD局部时间不是TR wall-clock。
- Impact：当前V3模型的`rd_quality`局部持续时间可能包含工程回退值；它不能被解释为真实CPI持续时间，也不能用于修复TR的zero-dt或作为液态模块的物理dt。若不同类别/source的CPI=0比例不同，可能形成未经验证的质量/类别捷径。
- Action：本轮不修改V3数据、配置或训练链路；正式raw/RD时间实验前，增加`cpi_count`有效性和回退来源字段，按类别/source/长度报告回退比例，并以固定V3 baseline做单变量对照。不得用该回退值替代TR wall-clock。
- 验证状态：CPI=0分布和回退代码事实已确认；回退值的物理正确性、对分类性能的影响和是否形成类别捷径未验证。
- 关联版本：当前V3数据和配置不变；若改变RD局部时间或质量特征语义，必须递增RADAR-LN3版本。

## Finding F046

- Severity：Medium（待验证风险）
- 文件位置：`src-v3/radar_ln3/model.py`第18-55行附近、`src-v3/radar_ln3/data.py`第212行附近、`data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-index.csv`
- 问题或结论：当前V3将同一TR点的多个RD帧做空间embedding后的mask平均，RD帧局部顺序和`time_seconds`没有进入帧级时序建模；这可能丢失点内微多普勒或动态纹理，但尚无分类收益证据。
- Evidence：`RDSpatialEncoder.forward`实际执行`frame_embeddings.sum(dim=2) / denominator`；Dataset仅用选中RD时间计算duration，未返回`rd_frame_times`；V3每个TR点的`rd_frame_count`实际包含2/3/4/6/8等多帧，且原始回波审计确认同一TR point对应多个raw/MAT帧。
- Impact：可能限制Bird/Drone的微多普勒区分，也可能削弱Clutter/Unknown的局部动态质量判断；若直接改为RD帧时序，还会新增模型变量、局部时间语义和延迟成本，不能与V4 dt实验混合解释。
- Action：V4只验证无效dt更新，不修改RD帧聚合；V4基线稳定后再设计“CNN+点内因果聚合+外层Liquid”的单因素实验，并单独报告RD frame count、CPI回退和端到端P95分层结果。
- 验证状态：代码和数据结构风险已确认；对分类性能的影响、是否需要`rd_frame_times`和是否改善Drone/Bird或Clutter/Unknown均待验证。
- 关联版本：V3数据复用；RD帧时序输入或语义改变时使用新的配置/契约版本。

## Finding F047

- Severity：Medium（实验假设）
- 文件位置：`src-v3/radar_ln3/model.py`第115-120行附近、`config/RADAR-LN3/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-formal-supervised-config.json`
- 问题或结论：在显式`tr_dt_valid=false`的事件上，当前V3的0.001秒floor使液态状态主要保持旧状态；使用有界观测事件更新可能改善同秒多点的证据累积，但也可能放大标签噪声或短前缀监督风险。
- Evidence：V3无效dt=11322/29663=`38.16%`；V3配置使用`zero_dt_event_floor_seconds=0.001`；V4代码已增加`invalid_dt_update_mode=bounded_observation`和`liquid_invalid_dt_observation_gain=0.25`，但V3/V4正式分类结果尚未执行。
- Impact：可能影响短前缀、Bird/Drone双向混淆和Clutter/Unknown召回；若观察更新增益过大，可能破坏连续时间平滑和目标级稳定结果。不能将该假设写成性能提升。
- Action：V4只使用有界观测事件模式，固定数据、loss、split、RD前端和checkpoint选优；模式、梯度、有限值、真实batch和entry Smoke已通过，后续由用户确认后做V4与V3的训练对照。失败时回滚到V3 floor配置。
- 验证状态：代码和focused Smoke已通过；分类收益、稳定结果和端到端影响仍待正式实验。
- 关联版本：V3对照；V4配置语义变化。

## Finding F048

- Severity：Medium（已修复的Smoke入口问题）
- 文件位置：`src-v3/smoke/contract_tests.py`、`audit_forward_test.py`、`latency_smoke.py`、`missing_rd_mask_test.py`、`run_smoke.py`
- 问题或结论：V4配置复用V3数据行时，部分Smoke原先将配置`schema_version=RADAR-LN3-V4`误作为Dataset的索引行schema，导致合法V3索引被拒绝。
- Evidence：V4首次`contract_tests`和`missing_rd_mask_test`实际报错`dataset contains a row outside expected schema RADAR-LN3-V4`；`formal_entry_contract`已使用`config.data.index_schema_version=RADAR-LN3-V3`并通过。
- Impact：会阻塞V4 focused验证并可能误判数据契约不兼容；不影响模型、数据内容或训练结果。
- Action：所有相关Smoke改为使用`config.data.index_schema_version or config.schema_version`；V4真实batch contract、missing-RD、V3回归和entry contract随后全部通过。
- 验证状态：已修复并验证；V4正式训练未执行。
- 关联版本：V3数据复用、V4配置。

## Finding F049

- Severity：Info（实验授权已确认）
- 文件位置：`config/RADAR-LN3/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-formal-supervised-config.json`、`config/RADAR-LN3/RADAR-LN3-V4-cq08-tr-rd-invalid-dt-event-formal-supervised-config.json`、`src-v3/train/formal.py`
- 问题或结论：V3/V4 paired正式监督训练已获用户授权，但训练和性能结果尚不存在。
- Evidence：两套配置均为seed=42、batch=2、20 epoch、patience=5、causal-prefix loss、parent Macro-F1选优、`evaluate_test=false`；两套正式artifact目录在授权时均不存在；本代理仅完成入口帮助核对和配置只读核对。
- Impact：在用户手动运行前不能报告V3/V4性能、dt更新收益、Clutter/Unknown改善或90%达标情况。
- Action：用户按V3→V4顺序手动运行正式命令；训练结束后再使用同一validation评估入口独立复核best checkpoint；test、拒识校准和端到端P95继续单独授权。
- 验证状态：授权和命令准备已确认；正式训练未执行。
- 关联版本：RADAR-LN3-V3、RADAR-LN3-V4。

## Finding F050

- Severity：High
- 文件位置：`artifacts/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-formal/training-summary.json`、`artifacts/RADAR-LN3-V4-cq08-tr-rd-invalid-dt-event-formal/training-summary.json`
- 问题或结论：V3/V4五分类液态模型在当前验证集上整体和弱类表现均未达到目标；V4无效dt观测更新没有在五分类验证中形成整体收益，不能继续把增加液态复杂度当作Clutter/Unknown主要修复手段。
- Evidence：V3 parent Accuracy=`0.8620689655172413`、Macro-F1=`0.7530281349855817`；V4 parent Accuracy=`0.8362068965517241`、Macro-F1=`0.7140031743844241`。V3 Clutter/Unknown parent F1分别为`0.5714285714285714`/`0.5128205128205129`；V4分别为`0.6666666666666667`/`0.3225806451612903`。两套结果均来自真实val summary，test未执行。
- Impact：当前五分类头同时承担DBB类间边界和CU内部边界，弱类低支持与Bird/Unknown混淆会污染对液态时序模块的归因；不能把V3/V4后验合并结果写成独立DBB/CU模型收益。
- Action：建立独立DBB/CU gate、DBB specialist和CU specialist，保持同一CQ08 V3 split、TR+RD输入、液态骨干和因果监督，只改变任务标签映射与输出类别；先Smoke，再由用户手动训练。
- 验证状态：V3/V4五分类短板已确认；独立DBB/CU模型尚未训练，收益待验证。
- 关联版本：V3、V4；专项任务配置为V5。

## Finding F051

- Severity：High
- 文件位置：`data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-data-contract.json`、`artifacts/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-formal/training-summary.json`
- 问题或结论：CU专项在parent级验证集的有效支持很小，独立CU二分类即使指标升高也可能统计不稳定；DBB/CU gate的CU正类支持大于五分类中的单独Clutter或Unknown，但仍必须按parent、window、长度和source报告。
- Evidence：V3 parent label counts为DBB=1396、CU=150；val parent中Clutter=8、Unknown=15，故CU合并后val support=23。V3五分类parent混淆中Clutter→Unknown=2、Unknown→Clutter=2、Unknown→Bird=2；V4 Unknown→Bird=5。
- Impact：不能用少量CU parent的单次seed结果宣布90%达标或稳定泛化；硬路由可能放大gate错误，CU specialist还需区分标签语义与拒识动作。
- Action：使用概率软级联而非硬路由；分别输出gate、DBB specialist、CU specialist和组合五类结果，报告support、混淆矩阵和置信度；test不在本阶段使用。
- 验证状态：支持数和已有五分类混淆已确认；专项模型指标、置信度校准和跨seed稳定性未执行。
- 关联版本：V3数据复用；V5任务配置。

## Finding F052

- Severity：Medium（实验设计）
- 文件位置：`src-v3/tests/cu/`、`src-v3/tests/dbb-cu/`（本阶段新增范围）
- 问题或结论：DBB/CU三层软级联的合理最小结构是`P(DBB/CU|x)`乘以对应specialist条件概率，不应先采用硬gate；该结构能隔离gate错误与CU/DBB内部边界，但不能自动修复标签质量、覆盖不足或开放集拒识。
- Evidence：当前液态模型输出序列/窗口五分类logits，现有stable状态机只做五类EMA和确认计数；不存在正式reject/abstain输出。原五类标签可无损派生三组监督映射，不需要修改标签或使用RD-only信息。
- Impact：硬gate一旦将CU误判为DBB，CU specialist无法恢复；概率乘积保留不确定性，但若gate校准差仍会降低最终五分类性能。模块后续接入必须保持输入契约、因果边界和状态重置边界一致。
- Action：本轮实现独立任务和soft-composition接口；只在三项专项模型分别完成真实val实验且DBB/CU、Clutter/Unknown和整体指标不恶化后，才讨论接入其它模型。
- 验证状态：结构与输入映射已设计；代码Smoke和真实训练收益待执行。
- 关联版本：V5任务配置。

## Finding F053

- Severity：Medium
- 文件位置：`src-v3/radar_ln3/task_models.py`、`src-v3/tests/cu/`、`src-v3/tests/dbb-cu/`、`src-v3/smoke/task_dbb_cu.py`、`src-v3/smoke/task_cu.py`
- 问题或结论：DBB/CU专项的独立输入、标签、mask、因果液态更新、任务输出和checkpoint契约已通过有界Smoke，但真实分类性能尚未产生。
- Evidence：三项CUDA Smoke均返回`status=passed`；gate输出`[B,L,2]`并验证软组合`[B,L,5]`，DBB输出`[B,L,3]`，CU输出`[B,L,2]`；三项均完成forward、causal-prefix loss、梯度更新和checkpoint round-trip；数据hash均为V3 index=`a5d054f...e649f87`、V3 contract=`9aca5f7...e0487433`、V2 stats=`584eddc...8f826b6`。
- Impact：当前只能证明工程接口和数值链路可运行，不能证明DBB/CU gate、CU specialist或DBB specialist改善V3/V4五分类，也不能判断是否可模块化接入。
- Action：V5只保留历史审计；下一步创建V6数据语义契约和三份V6配置，完成有界Smoke后再由用户手动训练，并在同一CQ08 split上报告window/parent、类别、长度、dt_valid和RD缺失分层；test保持未执行。
- 验证状态：V5历史隔离、配置、Smoke和checkpoint契约已验证，但训练放行已撤销；V6正式训练与性能评估未执行。
- 关联版本：V5历史审计；V6待创建。

## Finding F055

- Severity：High（历史放行已撤销）
- 文件位置：`src-v3/train/tasks.py`、`src-v3/report/tasks.py`、`src-v3/smoke/task_dbb_cu.py`、`src-v3/smoke/task_cu.py`、`artifacts/`
- 问题或结论：V5 DBB/CU专项曾基于当时Smoke达到手动训练准备条件，但Stage 030确认其7维RD质量输入包含parent全局未来信息；该放行已撤销，V5现在只能作为历史审计版本。
- Evidence：F057追踪到`rd_quality_partial_parent`和`rd_quality_parent_order_anomaly`进入V5模型；修复后`LiquidTaskClassifier`要求`schema_version=RADAR-LN3-V6`，V5即使手工补上新字段也返回audit-only错误；`artifacts/`下仍不存在V5正式专项模型目录。
- Impact：历史V5训练命令不得执行；若执行会产生与修复后因果语义不兼容、且无法进入严格三checkpoint加载器的权重。V5 Smoke不能作为性能或训练放行证据。
- Action：创建V6数据语义契约和三份V6配置，先做有界forward/loss/gradient/checkpoint Smoke，再由用户手动训练Gate、DBB和CU；V5命令仅保留追溯。
- 验证状态：V5执行阻断已通过focused测试；V6契约、配置、训练和validation未执行。
- 关联版本：V5历史审计；V6待创建。

## Finding F056

- Severity：Info
- 文件位置：`src-v3/tests/cu/contract.py`、`src-v3/tests/dbb-cu/contract.py`
- 问题或结论：两个专项目录已具备明确的任务契约入口；共享模型实现集中于`src-v3/radar_ln3/task_models.py`，避免三项任务复制出三套不一致的液态骨干。
- Evidence：CU契约固定`cu_specialist`的两类映射；DBB/CU契约固定gate、DBB specialist、CU specialist和概率软组合；新增文件只导入`radar_ln3.task_models`，静态隔离检查未发现`src`运行时依赖。
- Impact：任务语义、输出类别和后续训练/评估入口可追溯；不会改变V3数据、V4模型或V5配置。
- Action：训练和报告统一使用`src-v3/train/tasks.py`与`src-v3/report/tasks.py`；Smoke统一使用`src-v3/smoke/`，避免正式链路和Smoke混放。
- 验证状态：目录契约已创建；新增文件编译、CU映射导入和隔离检查均通过，正式性能仍未执行。
- 关联版本：V5。

## Finding F054

- Severity：Medium（统计风险）
- 文件位置：`config/RADAR-LN3/RADAR-LN3-V5-cq08-liquid-cu-specialist-config.json`、`data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-summary.json`
- 问题或结论：V5 CU specialist保留了原始CU标签不平衡，train parent为Clutter=35、Unknown=70，val parent为Clutter=8、Unknown=15；class weight只能缓解损失比例，不能解决标注边界、覆盖不足或拒识语义。
- Evidence：V3 summary与索引分组统计确认五类parent counts train为Drone=488、Bird=243、Balloon=245、Clutter=35、Unknown=70，val为105/52/52/8/15；V5配置class weights=`[1.5,0.75]`，未删除任何parent、未改标签。
- Impact：CU val指标置信区间宽，单seed或单次训练不能宣称90%达标；Unknown分类仍不等同于abstain/reject。
- Action：训练后同时报告混淆矩阵、support、置信度和长度分组；拒识另行使用train/grouped OOF校准，不在V5专项Smoke或test上搜索阈值。
- 验证状态：支持数和配置已确认；标签质量、开放集边界、拒识收益未验证。
- 关联版本：V5。

## Finding F057

- Severity：Critical（代码已修复，待V6重训）
- 文件位置：`src-v3/build/cq08_point_contract.py`第310-316、408-409行附近；`src-v3/radar_ln3/data.py`的`RD_QUALITY_SCHEMA`、`encode_causal_rd_quality`和`CQ08PointSequenceDataset.__getitem__`
- 问题或结论：历史V3/V4/V5模型输入曾消费由完整parent计算的`rd_quality_partial_parent`和`rd_quality_parent_order_anomaly`，违反在线严格因果输入契约；修复后模型7维质量输入仅含当前事件可得信息。
- Evidence：构建代码仍保留两个字段作离线审计；`data.py`现使用`causal-current-event-v1`并只编码present、frame count、局部duration、missing、局部时间非单调、多帧和截断。真实partial parent `cq08|track|1010`上翻转两个全局字段，质量tensor最大差值=`0.0`。
- Impact：早期窗口可提前知道未来是否存在RD缺失/顺序异常，可能形成source或类别捷径并污染window、parent、stable和专项级联指标；液态模型复杂度不能修复该泄漏。
- Action：保留索引内容与旧字段只读；下一正式版本建立V6数据语义契约和配置并重训，禁止复用V3/V4/V5权重。
- 验证状态：代码修复、真实parent前缀不变性和旧配置阻断已通过；修复后分类性能、真实validation和线上P95未验证。
- 关联版本：V3/V4/V5为旧泄漏语义参考；V6待创建。

## Finding F058

- Severity：High（已修复）
- 文件位置：`src-v3/radar_ln3/metrics.py`的`summarize_records`；`src-v3/radar_ln3/task_models.py`的`summarize_task_records`
- 问题或结论：当某类别没有任何预测时，Precision被设为None并从Macro Precision分母移除，导致固定五类/专项类别宏指标被高估。
- Evidence：修复前二分类/五分类退化探针分别错误输出0.5/0.2；修复后实际输出0.25/0.04，与固定合同类别且zero-division=0一致。
- Impact：会掩盖Clutter/Unknown或其他类别完全未被预测的失败，90%验收门槛和checkpoint/报告解释不可信。
- Action：维持固定类别分母，并用退化探针防止回归；旧V3/V4报告须明确属于旧指标实现，不能与修复后数值直接混称。
- 验证状态：五类和专项指标focused探针已通过；真实V6 validation未执行。
- 关联版本：指标代码修复，不单独改变数据版本；旧报告需标记为旧指标实现。

## Finding F059

- Severity：High（已修复语义）
- 文件位置：`src-v3/radar_ln3/stable.py`的`TargetStableStateMachine`；`src-v3/radar_ln3/streaming.py`；`src-v3/radar_ln3/evaluator.py`的`evaluate_dataset`
- 问题或结论：stable状态尚未达到确认次数时被初始化并输出为`UnknownTarget`，在线结果没有`stable_confirmed`字段，评估还把未确认状态作为普通Unknown预测计入指标。
- Evidence：修复前`stable_label`初始化为4；修复后第一次更新返回`stable_label_id/name/confidence=None`且`stable_confirmed=false`，第二次达到确认数后才输出Drone。概率0.8入口保持`0.800000011920929`，未二次softmax；混用logits/probabilities被拒绝。
- Impact：把“暂未形成稳定判断”与真实Unknown类别混为一谈，污染短序列、Unknown Recall/Precision和拒识语义，无法可靠支持窗口临时结果加目标稳定结果的业务契约。
- Action：保持window临时结果和confirmed stable指标分离，所有报告同时输出coverage与unconfirmed；UnknownTarget继续只表示分类类别，不等于未确认或abstain。
- 验证状态：离线评估、五类在线包装器和软级联概率接口focused测试通过；真实线上流和拒识指标未执行。
- 关联版本：运行语义变化；正式配置/报告版本尚未创建。

## Finding F060

- Severity：High（工程缺口已闭环，性能待验证）
- 文件位置：`src-v3/radar_ln3/cascade.py`的`LiquidSoftCascade`、`OnlineLiquidSoftCascade`、`evaluate_cascade`和`load_cascade`；`src-v3/report/cascade.py`
- 问题或结论：三checkpoint联合加载、同一parent五分类评估、每目标三份液态状态和概率stable闭环现已实现；尚无三份V6正式checkpoint，因此真实性能链路未执行。
- Evidence：新增`radar_ln3/cascade.py`和`report/cascade.py`；持久化focused结果中联合概率有限且和为1，baseline和checkpoint round-trip的batch/逐点输出最大差=`7.450580596923828e-08`，reset恢复首点输出；checkpoint同时严格绑定模型契约和三份数据hash，混合V6/V7运行版本被拒绝。
- Impact：三个独立任务即使各自指标较好，也不能证明组合五分类性能、在线因果一致性、状态重置正确性或P95满足要求。
- Action：V6三模型训练后只在同一val split运行联合报告，输出window/parent/stable、组件、source、长度、置信度和dt_invalid；校准和test继续独立授权。
- 验证状态：接口、checkpoint round-trip、reset和batch/online一致性通过；联合真实validation、温度校准/ECE、P95和test未执行。
- 关联版本：V5历史组件；V6待创建。

## Finding F061

- Severity：Medium（有证据但未完成性能验证）
- 文件位置：`src-v3/radar_ln3/model.py`的`CfCLiquidCausalEncoder._step`和`readout`
- 问题或结论：液态状态在按dt指数插值后立即LayerNorm并将归一化结果作为下一递归状态，可能显著削弱时间常数对更新幅值的表达。
- Evidence：同一随机首事件、相同权重的纯内存探针中，`dt=0.1s`与`dt=10s`经过当前液态单元和同型分类头后的最大logit差约`1.6e-5`；tau参数仍有有限梯度，因此不能写成dt完全未使用。
- Impact：不规则采样和真实时间间隔可能没有按设计充分影响状态；直接修改归一化位置同时会改变递归动力学，必须作为单因素实验，不能和RD时序、loss或标签修复混合。
- Action：已保留`recurrent` baseline并增加默认关闭的`readout_only`候选；只允许它与`masked_mean` RD基线配对。分类收益、稳定性和P95必须由后续单因素paired实验验证。
- 验证状态：候选构造和batch/online一致性通过，持久化focused结果最大差=`2.9802322387695312e-08`；正式性能实验未执行。
- 关联版本：候选代码已存在；正式消融配置版本待创建。

## Finding F062

- Severity：Info（已解除，正式训练待执行）
- 文件位置：`src-v3/radar_ln3/task_models.py`、`src-v3/radar_ln3/cascade.py`、`data/processed/`、`config/RADAR-LN3/`
- 问题或结论：Stage 030时V6契约、配置和正式checkpoint缺失，曾阻止修复后训练；Stage 031已创建并验证V6契约和三份V6配置，训练阻塞已解除，但正式checkpoint尚不存在。
- Evidence：V6数据契约SHA-256=`2110938d1d94923edb44c2825e34fb595fb1b34c1353f3f3e512403f91ed8777`；三份V6配置加载、共享契约、hash、CQ08纯度和三项CUDA Smoke均通过；V5配置仍返回audit-only；`artifacts/`下未生成V6正式目录。
- Impact：可以进入用户手动正式训练准备；不能把Smoke当作性能结果，不能用V5权重执行修复后联合级联，也不能宣称90%指标、拒识收益或P95达标。
- Action：按V6 Gate→DBB specialist→CU specialist顺序手动训练；训练后分别运行val评估，再运行三checkpoint联合cascade报告；test、校准、拒识和P95继续保持未执行，除非另行授权。
- 验证状态：V6契约/配置/Smoke/compileall/隔离/入口帮助已通过；正式训练、validation、联合性能、test和消融训练未执行。
- 关联版本：V6。

## Finding F063

- Severity：Medium（待验证实验假设）
- 文件位置：`src-v3/radar_ln3/model.py`、`src-v3/radar_ln3/task_models.py`
- 问题或结论：RD点内帧候选`causal_gru`已替代`masked_mean`形成可运行单因素开关，但它仅依据现有frame顺序聚合，没有可靠物理frame dt，不能称为连续时间RD建模。
- Evidence：候选只按frame index顺序调用`GRUCell`；持久化focused结果中其batch/online最大差=`8.940696716308594e-08`，与液态`readout_only`同时开启会被拒绝。
- Impact：可能保留点内动态纹理，也可能增加参数、时延或过拟合；目前没有Drone/Bird、Clutter/Unknown或Macro-F1收益证据。
- Action：在V6修复baseline完成后单独创建RD帧时序配置，保持数据、loss、采样、液态归一化和评估不变；报告参数量、显存、P95和分类分层指标。
- 验证状态：shape/有限值/batch-online契约通过；正式消融未执行。
- 关联版本：正式消融配置版本待创建。

## Finding F064

- Severity：Medium（已解决，性能待验证）
- 文件位置：`data/processed/RADAR-LN3-V6-cq08-tr-rd-causal-quality-data-contract.json`、`config/RADAR-LN3/RADAR-LN3-V6-cq08-liquid-*.json`、`src-v3/smoke/RADAR-LN3-V6-*-contract-smoke/`
- 问题或结论：修复后模型需要V6数据契约和三份V6运行配置；V3 index和V2 train-only统计内容不变，按hash复用；V6契约、配置、有界Smoke以及后续用户手动正式训练均已完成。
- Evidence：V6 contract SHA-256=`2110938d1d94923edb44c2825e34fb595fb1b34c1353f3f3e512403f91ed8777`；三份V6配置分别通过加载和hash校验；Gate/DBB/CU三项CUDA Smoke均通过真实CQ08 Dataset、forward、causal-prefix loss、梯度和checkpoint round-trip。
- Impact：输入版本和artifact边界已明确，V5旧权重仍不能使用；Smoke只证明工程可运行，不证明分类收益、90%指标、拒识效果或P95。
- Action：V6作为只读性能基线；依据Stage 032结果实施V7/V8单因素修复，不覆盖V6。保持test、开放集校准、多seed和P95未执行。
- 验证状态：V6契约/配置/hash/纯度/Smoke/compileall/隔离/入口帮助、正式训练、组件validation和联合validation均已执行；test、真实开放集、正式消融、多seed和P95未执行。
- 关联版本：V6。

## Finding F065

- Severity：Medium（已修复的Smoke产物隔离问题）
- 文件位置：`src-v3/smoke/task_dbb_cu.py`、`src-v3/smoke/task_cu.py`、`src-v3/smoke/RADAR-LN3-V5-smoke-artifact-integrity-warning.md`
- 问题或结论：Smoke入口原先硬编码`RADAR-LN3-V5`输出目录，首次运行V6 Smoke时覆盖了旧V5 Smoke目录；未覆盖正式artifact，但旧V5 Smoke文件不能再作为历史证据。
- Evidence：首次V6 Smoke输出路径实际为`src-v3/smoke/RADAR-LN3-V5-*`；已将入口改为`{config.schema_version}-...`，重新运行后V6结果位于三个独立V6目录，均通过。
- Impact：若不修复，会造成Smoke版本混淆、错误复用旧结果和不可追溯的checkpoint语义。
- Action：保持动态版本目录；旧V5目录保留并写完整性警告，不删除用户文件；正式训练artifact从未被覆盖。
- 验证状态：路径修复和清洁V6 Smoke已验证；V5历史Smoke证据作废，V6 Smoke有效。
- 关联版本：V5历史Smoke、V6。

## Finding F066

- Severity：Critical（已确认性能问题）
- 文件位置：`artifacts/RADAR-LN3-V6-cq08-liquid-cu-specialist-formal/`、`config/RADAR-LN3/RADAR-LN3-V6-cq08-liquid-cu-specialist-config.json`
- 问题或结论：V6 CU specialist没有形成可用的Clutter/Unknown边界，远低于两类Precision/Recall均不低于0.90的目标；这不是当前0.5阈值的小偏差。
- Evidence：parent support为Clutter=8、Unknown=15，混淆矩阵=`[[7,1],[9,6]]`，两类P/R分别为`0.4375/0.875`和`0.857143/0.4`；窗口AUC=`0.551221`、parent AUC=`0.65`，同一validation乐观阈值下的双类最小P/R也只有`0.538462`。
- Impact：Unknown大量被并入Clutter；软级联无法达到CU门槛，也不能据此建立可靠开放集拒识。
- Action：先做标签/capture/原始RD困难样本审计与共享编码器/表征实验；class weight、阈值和更复杂液态单元均不得单独宣称解决。
- 验证状态：V6组件与阈值诊断、V7/V8 paired重训及组件validation已执行；标签复核、共享预训练、三checkpoint联合validation和真实开放集验证未执行。V7/V8 CU specialist结果相同，Clutter/Unknown最弱P/R仍为`0.466667`。
- 关联版本：V6 baseline；V7/V8正式结果。

## Finding F067

- Severity：High（已确认性能问题）
- 文件位置：`artifacts/RADAR-LN3-V6-cq08-liquid-dbb-cu-gate-formal/`、`config/RADAR-LN3/RADAR-LN3-V6-cq08-liquid-dbb-cu-gate-config.json`
- 问题或结论：Gate的CU Precision严重不足，且类别权重将决策明显推向CU；阈值调整不能同时满足DBB/CU P/R 0.95。
- Evidence：parent Gate矩阵=`[[192,17],[3,20]]`，CU Precision=`0.540541`；配置权重CU/DBB约9.29倍；同一validation最优双类最小P/R仅`0.608696`。
- Impact：DBB样本被错误送入CU分支并在概率乘积中污染Bird、Balloon、Clutter和Unknown最终结果。
- Action：保持V6权重基线，先做`dt_valid`捷径消融；后续类别权重/采样必须作为独立变量，并以弱类P/R而非平均F1选优。
- 验证状态：性能、权重和阈值诊断、V7/V8修复重训及组件validation已执行；V7与V8仍未同时满足Gate双类P/R 0.95，三checkpoint联合validation未执行。
- 关联版本：V6 baseline；V7/V8正式结果。

## Finding F068

- Severity：High（V6已确认；V7直接特征通路已修复且正式效果已验证为不足）
- 文件位置：`src-v3/radar_ln3/task_models.py`的`LiquidTaskClassifier.forward/forward_online_point`；`data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-index.csv`
- 问题或结论：V6中`tr_dt_valid`既作为融合分类特征，又控制液态状态更新；其类别分布高度不均。V7已移除直接融合通路，但按设计保留动力学控制，因此类别相关的更新节奏仍是待paired实验验证的残余风险。
- Evidence：V6 validation无效dt比例Drone/Bird/Balloon=`0.390725/0.495558/0.473320`，Clutter/Unknown=`0.053691/0.054945`。固定seed的V7 focused测试显示融合维度80→79，只翻转`dt_valid`时直接融合张量最大差=`0.0`，液态状态最大差=`0.054695725440979004`，batch/online最大差=`8.940696716308594e-08`。V7正式Gate仍为DBB P/R=`0.989189/0.875598`、CU P/R=`0.446809/0.913043`，说明移除直接通路没有解决Gate边界。
- Impact：离线CQ08 Gate可能获得不可泛化捷径，线上设备时间精度变化会造成域漂移；同时混淆对液态连续时间效果的归因。
- Action：继续禁止`dt_valid`直接进入fusion；下一特征实验应显式表达因果运动一致性，并按dt有效mask计算，不能再让采集有效率作为分类捷径。
- 验证状态：代码、契约、CPU/CUDA Smoke和V7正式重训/组件validation均已验证；跨采集泛化、联合级联和多seed未执行。
- 关联版本：V6 baseline；V7。

## Finding F069

- Severity：High（V6/V7已确认；V8正式训练证明选优修复不足以补足表征）
- 文件位置：`src-v3/train/tasks.py`的checkpoint选择；三份V6配置的`checkpoint_selection_metric`
- 问题或结论：V6/V7按`parent_macro_f1`选best，与用户要求每个类别Precision/Recall下限的验收目标不一致；V8已改为弱类下限优先并完成正式训练，但结果证明选优修复不能替代表征修复。
- Evidence：DBB第12轮parent最差P/R=`0.903846`，第13轮Macro-F1略高而被选中但最差P/R降为`0.838710`。V8正式训练后Gate best切换为第2轮，DBB P/R=`0.955224/0.918660`、CU P/R=`0.451613/0.608696`；DBB specialist与V7相同，Bird P/R=`0.905660/0.923077`；CU specialist也与V7相同，Clutter/Unknown最弱P/R仍为`0.466667`。
- Impact：best checkpoint可能牺牲弱类；但只改选优不能补足现有表征差距。
- Action：保留V8选优语义用于后续实验，但不得把选优修复当作表征修复；下一轮只改变一个因果特征因素。
- 验证状态：选择器、Smoke、V8正式训练和组件validation均已完成；三checkpoint联合validation、test和多seed未执行。
- 关联版本：V6/V7 baseline选择；V8。

## Finding F070

- Severity：High（已确认联合概率问题）
- 文件位置：`src-v3/radar_ln3/task_models.py`的`compose_soft_cascade`；`src-v3/radar_ln3/cascade.py`
- 问题或结论：三个使用不同class weight训练的softmax未经温度、先验或ECE校准即相乘，条件头正确结果会被Gate尺度覆盖。
- Evidence：DBB specialist parent上52个Bird全部正确，联合级联只保留40个，11个转Unknown、1个转Clutter；联合parent Macro-F1=`0.680308`，低于DBB条件模型；代码报告`probability_calibration=not_executed`。
- Impact：组件指标不能直接外推最终五分类；概率乘积造成额外误差传播。
- Action：在V7/V8 paired训练后，仅用train parent-grouped OOF拟合分支温度/先验校准；test不得参与。
- 验证状态：联合validation已执行；OOF校准未执行。
- 关联版本：V6 baseline；后续校准版本待定。

## Finding F071

- Severity：Critical（接口缺口已修复；拒识能力与校准仍未实现）
- 文件位置：`src-v3/report/audit.py`、`src-v3/radar_ln3/stable.py`、`src-v3/radar_ln3/cascade.py`
- 问题或结论：`UnknownTarget`仍是闭集类别，现已具备正交reject/abstain接口，但没有train parent-grouped OOF校准artifact，因此当前没有可启用、可报告性能的拒识能力；stable确认仍不等于拒识。
- Evidence：`RejectionPolicy`默认输出`decision_state=not_calibrated,rejected=null`，无校准强制启用会抛错，validation来源校准会被拒绝；接口测试中被拒识样本的`class_label_name`仍保持`UnknownTarget`。历史0.60诊断Coverage=`0.875`、Accepted Accuracy=`0.896552`、Accepted固定五类Macro-F1=`0.700625`，仍接受21个错误且5个错误置信度不低于0.90；该阈值未保存。
- Impact：无法报告可信FAR、FRR、Selective Risk，不能满足Unknown既为已知类别又承担开放集安全出口的业务意图。
- Action：保持五分类`class_label`与正交`decision_state`；下一独立版本只能用train parent-grouped OOF拟合并冻结阈值/温度，再报告Coverage、FAR、FRR和Selective Risk。test不得用于拟合。
- 验证状态：独立接口和未校准保护已通过focused测试；OOF校准、阈值artifact、真实开放集验证和拒识指标均未执行。
- 关联版本：V7/V8代码接口；正式阈值版本未分配。

## Finding F072

- Severity：High（表征信息损失已有物理统计证据，模型收益待验证）
- 文件位置：`src-v3/radar_ln3/model.py`的`RDSpatialEncoder`
- 问题或结论：RD前端采用大步长卷积、全局空间平均池化和点内`masked_mean`，会丢失距离/多普勒位置、峰形和帧间动态；相位仅除以π而非环状编码。
- Evidence：代码输出每帧24维全局池化向量，再投影到24维并对最多12帧均值；V6-V8配置均为`rd_frame_aggregation=masked_mean`。Stage 033完整validation统计显示V7正确Bird与Bird→CU的速度带宽中位数=`41.985/21.613 m/s`、零速±2 m/s能量=`0.204/0.650`、谱熵=`0.894/0.554`、top10集中度=`0.370/0.780`，但当前前端不显式保留物理速度坐标或这些统计。
- Impact：可能限制Drone/Bird微多普勒和Clutter/Unknown纹理分离，但尚无单因素性能证据。
- Action：先把“当前事件RD物理统计分支”和“点内帧时序聚合”拆成两个单因素实验；物理统计应包含速度带宽、零多普勒占比、谱熵、峰值集中度、相位相干性及TR/RD径向一致性。
- 验证状态：代码路径、物理轴和validation分布已确认；正式表征消融未执行。
- 关联版本：后续版本待定。

## Finding F073

- Severity：Medium（工程判断有实测支持）
- 文件位置：三份V6 `best_model.pt`中的`liquid.tau_unconstrained`
- 问题或结论：三个模型学到的液态时间常数仍集中在约15秒初始化附近，当前证据不支持把液态模块容量认定为主要瓶颈。
- Evidence：tau中位数Gate=`15.016s`、DBB=`14.923s`、CU=`15.022s`；CU仅371个优化step，性能远低于门槛。
- Impact：增加液态层数或隐藏维度可能只增加复杂度，不能修复CU标签/覆盖、Gate捷径、概率校准和拒识缺失。
- Action：液态模块继续作为候选；数据和训练契约修复后再与GRU/TCN做同数据单因素对照。
- 验证状态：checkpoint参数已读取；时序架构对照未执行。
- 关联版本：V6 baseline。

## Finding F074

- Severity：Medium（已修复并由V7/V8正式日志验证的可观测性问题）
- 文件位置：`src-v3/train/tasks.py`、`src-v3/radar_ln3/logging_utils.py`、`src-v3/smoke/logging_test.py`
- 问题或结论：V6专项任务训练结束日志只打印best checkpoint路径，未输出最终parent/window总体指标、逐类Precision/Recall/F1和低置信度样本，不满足现有训练审计约定；完整样本JSON虽存在，但操作人员必须额外打开summary才能掌握弱类结果。
- Evidence：三份V6 `training.log`结尾均只有epoch摘要和`run=complete`；代码搜索显示`train/tasks.py`未调用`log_metric_set`或`log_low_confidence_records`。修复后focused日志测试确认`per_class_metrics_logged=true`、`low_confidence_only=true`，0.59样本出现、0.95样本不出现。
- Impact：若不修复，V8即使按弱类下限选优，日志仍难以即时核对DBB/CU及各专项类别是否接近门槛，也容易误将高置信度样本明细淹没训练日志。
- Action：保留完整样本于`sample_detail_information.json`；训练结束只在日志输出总体/逐类指标及置信度低于0.60的样本。不得把日志格式变化作为性能变化。
- 验证状态：代码、focused日志测试和V7/V8六份正式training.log均已产生；日志格式变化不代表分类性能变化。
- 关联版本：V7/V8代码路径；不改变配置或数据版本语义。

## Finding F075

- Severity：High（已确认数据语义结论）
- 文件位置：`data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-index.csv`；`src-v3/processed/report/RADAR-LN3-V7-V8-dbb-cu-feature-audit.md`
- 问题或结论：CQ08中的ClutterTarget不是“无运动”紧凑类，而是具有较低路径一致性、较高转向和径向速度变化的伪运动航迹；使用简单速度阈值或纯DBB单类接受器无法可靠把Clutter全部排除。
- Evidence：train+validation共43个Clutter parent，速度中位数=`4.893 m/s`、几何推算速度=`6.577 m/s`；转向率中位数=`12.436 deg/s`、绝对径向加速度=`0.993 m/s²`、路径直线度=`0.285`、径向速度符号翻转率=`0.342`。相对地Drone/Bird/Balloon路径直线度分别=`0.876/0.842/0.774`，符号翻转率中位数均为0。
- Impact：如果Gate只学习“是否有运动”，Clutter会以运动目标形式进入DBB；应学习运动连贯性、物理一致性和RD谱结构。
- Action：后续TR特征候选优先使用因果增量、路径效率的在线累计量、环状转向变化、径向/距离变化一致性及显式valid mask；不得使用完整parent未来统计作为在线输入。
- 验证状态：train+validation描述统计已执行；单因素模型实验、test和多seed未执行。
- 关联版本：V3数据/V6契约只读证据；新特征版本未分配。

## Finding F076

- Severity：High（已确认类别边界问题）
- 文件位置：V3 CQ08索引；V7 Gate sample detail；Stage 033审计报告
- 问题或结论：UnknownTarget不是“无运动CU”类，其运动分布与Bird显著重叠；Unknown同时承担开放集意图时，不能由封闭的运动阈值或Clutter残差直接定义。
- Evidence：train+validation Unknown/Bird速度中位数=`8.467/9.545 m/s`、绝对径向速度=`6.140/5.625 m/s`、路径直线度=`0.902/0.842`。V7中有2个Unknown被Gate高置信接受为DBB；未校准拒识接口仍为`not_calibrated`。
- Impact：DBB knownness只能表示与三个已知DBB模式的匹配程度；未匹配结果不能直接等同UnknownTarget或abstain。
- Action：保持Unknown闭集标签、CU专项分类与正交开放集拒识三种语义分离；拒识阈值仍只能由train parent-grouped OOF校准。
- 验证状态：闭集分布和V7错误已核实；真实外部未知样本、OOF校准和开放集指标未执行。
- 关联版本：V7/V8；正式拒识版本未分配。

## Finding F077

- Severity：High（已确认V7 Bird→CU条件差异）
- 文件位置：`artifacts/RADAR-LN3-V7-cq08-liquid-dbb-cu-gate-formal/sample_detail_information.json`；V3索引与物理RD张量
- 问题或结论：Bird→CU不是因为Bird没有运动；错误Bird呈现更强转向/径向变化、完整有效dt以及低多普勒窄谱形态，正好与Clutter/Unknown边界竞争。
- Evidence：17个错误Bird与35个正确Bird的速度中位数=`8.398/10.634 m/s`，转向变化=`6.395/2.179°/event`，dt有效率=`1.0/0.214`；RD速度带宽=`21.613/41.985 m/s`、零速能量=`0.650/0.204`、谱熵=`0.554/0.894`、top10集中度=`0.780/0.370`。
- Impact：单纯提高Bird权重或降低DBB阈值会同时放入更多CU；需要显式建模运动一致性和RD谱形，而不是把所有DBB压成一个速度正类。
- Action：下一Gate实验先保留DBB三类条件证据；knownness分支使用因果运动一致性与当前事件RD物理统计，未接受只路由CU，不直接改五类标签。
- 验证状态：完整validation条件统计已执行；新Gate训练和联合级联未执行。
- 关联版本：V7/V8现有Gate；候选版本未分配。

## Finding F078

- Severity：High（已确认特征契约缺口）
- 文件位置：`src-v3/radar_ln3/features.py`的`TR_VALUE_NAMES`；`src-v3/radar_ln3/model.py`的`RDSpatialEncoder`
- 问题或结论：当前14维TR只编码点值，没有速度/径向速度增量、加速度、转向率、路径一致性或TR/RD一致性；RD张量虽带物理速度/距离轴，Dataset和模型却不消费轴值，并在空间全局池化后对点内帧求均值。
- Evidence：静态代码契约已核对；train→validation诊断中五点可解释TR组合AUC=`0.9530`但Bird路由仅`0.75`，运动组合完整parent AUC=`0.8883`；确定性分层RD train子集到完整validation的物理统计探针AUC=`0.8273`。这些仅证明信息存在且当前表达不充分，不是模型收益。
- Impact：液态模块只能累积输入中已经表达的事件；它不能自行恢复被全局池化/均值抹去的物理位置、点内帧序和显式运动一致性。
- Action：Stage 034已选择先做TR因果运动一致性单因素；V9只修改Gate输入schema，RD物理统计保持未实现。V9必须使用新checkpoint契约并拒绝V8 checkpoint。
- 验证状态：V9代码、契约、全量train/validation特征审计、shape/gradient/checkpoint、在线/离线、CPU/CUDA Smoke均通过；正式训练和性能归因未执行。
- 关联版本：V3/V6/V7/V8只读基线；V9。

## Finding F079

- Severity：Medium（有强证据但字段语义仍需上游确认）
- 文件位置：V3索引`tr_feature_values_json`首事件；`src-v3/radar_ln3/features.py`
- 问题或结论：首事件speed/course同时精确为0具有类别相关的初始化或采集占位特征，当前编码器把它当作真实零值且没有“初始化未知”mask。
- Evidence：train+validation首点双零比例Drone/Bird/Balloon=`24.79%/28.14%/20.20%`，Clutter/Unknown=`62.79%/80.00%`；首点绝对径向速度中位数仍分别为`9.97/5.78/2.84/3.71/7.51 m/s`。该单变量方向化validation AUC=`0.6993`。
- Impact：模型可能利用航迹初始化方式区分类别，线上跟踪器或字段填充策略变化会产生域漂移；也会把Unknown错误解释为静止。
- Action：V9保守地禁止首点speed/course作为第二点导数基线，变化量从后续已观察基线开始；原14维点值输入仍保留首点零值，因此上游字段规范和跨设备一致性仍需核实。
- 验证状态：V9导数放大风险已缓解并通过首点/在线测试；原点值捷径、上游语义和跨设备一致性仍未解决。
- 关联版本：V3数据契约；V9局部缓解。

## Finding F080

- Severity：High（实施与正式validation已验证，未达标）
- 文件位置：`src-v3/radar_ln3/features.py`、`data.py`、`task_models.py`及V9 Gate配置/数据契约
- 问题或结论：V9必须验证“显式运动一致性表征”而不是“DBB是否会动”。Clutter已有伪运动，Unknown与Bird运动重叠，因此速度阈值、四类knownness混用或把Gate未接受样本直接定义为Unknown/reject都不符合现有证据。
- Evidence：F075-F079的CQ08 train+validation统计；V8 Gate DBB P/R=`0.955224/0.918660`、CU P/R=`0.451613/0.608696`；V9正式best checkpoint parent矩阵=`[[198,11],[6,17]]`，DBB P/R=`0.970588/0.947368`、CU P/R=`0.607143/0.739130`，Macro-F1=`0.812752`。V9相对V8改善但仍未达到Gate双类P/R门槛。
- Impact：若同时改变RD前端、loss、标签或拒识，无法归因运动特征价值；若mask直接拼入分类输入，模型可能继续利用dt/采集完整度捷径。
- Action：V9已完成单seed正式训练和Stage 035归因；当前只保留为研究checkpoint，不进入正式级联。下一步必须先消除/等化时间精度可用性捷径，再以同数据、同split、同训练预算重训单因素对照。
- 验证状态：版本/hash、特征schema、代码、数据契约、配置、focused测试、CPU/CUDA Smoke、正式训练和validation已完成；test、多seed、正式级联和部署未执行。
- 关联版本：V8 baseline；V9。

## Finding F081

- Severity：Critical（validation干预已确认主要归因风险）
- 文件位置：`src-v3/processed/report/RADAR-LN3-V9-causal-tr-motion-feature-audit.json`；`src-v3/radar_ln3/task_models.py`的motion数值门控
- 问题或结论：V9不把valid mask直接输入分类器，但无效motion值必须为0，因此非零位置模式仍隐式携带类别相关采集节奏；V9正式收益不能自动归因于物理运动量。
- Evidence：train/validation各类几何motion有效率范围分别=`0.4356001/0.4418665`；validation DBB/CU有效率=`0.5624383/0.9454976`。Stage 035在同一V9 best checkpoint上将真实值替换为train-only逐特征中位数、同时保留valid位置，Parent Macro-F1=`0.782817`、CU F1=`0.608696`，分别保留V8→V9增益约64.86%/60.87%；3个跨parent扰动Macro-F1=`0.783231–0.798765`，均值保留约77.31%增益。
- Impact：V9大部分validation改善可在目标自身物理运动值被移除后保留，说明模型主要依赖motion是否可用/零位置模式；该模式随设备时间精度、采样节奏或tracker策略改变时可能失效，不能作为部署Gate证据。
- Action：V9不得进入正式级联。下一数据/特征单因素必须让motion数值表征与时间可用性解耦，并在不暴露类别相关valid模式的条件下重新训练；test不得用于选择方案。
- 验证状态：正式训练、直接validation、全零、train中位数模板和3个跨parent扰动归因已完成；重训式无捷径消融、跨设备、多seed和test未执行。
- 关联版本：V9。

## Finding F082

- Severity：Medium（已确认物理字段约定）
- 文件位置：V3 train index；`src-v3/radar_ln3/features.py`的径向/距离变化一致性
- 问题或结论：CQ08 `radial_speed_mps`与`delta(range_m)/dt`采用同号约定，V9可使用两者绝对误差作为物理一致性特征，无需符号反转。
- Evidence：仅train的12987个有效相邻对中，同号率=`0.9321629`；`abs(vr-dR/dt)`中位数=`0.8483202 m/s`，`abs(vr+dR/dt)`中位数=`10.2650375 m/s`。
- Impact：避免因符号约定错误把物理一致目标编码成高误差；该统计只冻结公式，不是分类阈值或性能证据。
- Action：保持V9同号绝对误差公式；若接入其他source或设备，必须分别重新核查坐标/符号约定，不能直接外推CQ08。
- 验证状态：CQ08 train-only约定已验证；raw和其他设备未验证。
- 关联版本：V9。

## Finding F083

- Severity：Critical（已确认数据采集/时间精度捷径）
- 文件位置：`data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-index.csv`的`tr_dt_valid/tr_dt_reason`；`src-v3/radar_ln3/features.py`的`CausalTRMotionState.update`；`src-v3/radar_ln3/task_models.py`的`motion_encoder`
- 问题或结论：V9的motion分支把CQ08秒级TR时间是否能形成正dt编码成强隐式分类开关。该可用性几乎按DBB/CU标签分离，不是纯物理运动信息。
- Evidence：validation非首帧`zero_delta_at_second_precision`为Drone=687、Bird=450、Balloon=427、Clutter=0、Unknown=0；负dt和非有限dt均为0。Clutter/Unknown parent的dt有效率均值均为1.00，而Drone/Bird/Balloon为0.63/0.54/0.58；低于0.75的105 parent全部为DBB，23个CU全部在高有效率组。motion encoder为无偏置Linear→无仿射LayerNorm→SiLU→无偏置Linear，全零严格输出0而非零输入被标准化放大。
- Impact：V9可以不理解运动方向、幅度或一致性，仅凭某事件有无可计算motion来路由CU；这会夸大validation表现并使线上换设备、时间戳精度或采样策略后退化。
- Action：不能把zero-dt简单加epsilon或用RD CPI时长冒充TR wall-clock。下一实验应先冻结一个不暴露可用位置的motion接口，例如只在合法、业务一致的事件子任务监督/评估，或用train-only、与标签无关的缺失机制进行重训式鲁棒化；具体schema需单独设计和Smoke后再授权训练。
- Action：用户已选择Stage 036方案A：以不读取`dt/dt_valid`的逐观测增量特征替换V9物理率特征。V10必须通过显式`dt/dt_valid`干预不变测试和按类valid率审计后，才可进入用户手动训练状态；不得用epsilon或RD CPI伪造TR时间。
- 验证状态：validation全量时间原因统计、代码机制和post-hoc归因已确认；无捷径重训和跨设备验证未执行。
- 关联版本：V3时间契约；V9正式checkpoint。

## Finding F084

- Severity：High（已确认无法从当前原始回波恢复TR亚秒wall-clock）
- 文件位置：`src-v3/processed/report/RADAR-LN3-original-echo-audit-summary.json`；`src-v3/tests/processed/original_echo_audit.m`
- 问题或结论：当前CQ08原始`.dat`/MAT可验证帧顺序、track point索引和CPI/PRT局部时长，但没有显式TR wall-clock字段；现有12文件证据不支持用原始回波恢复同秒TR点之间的真实间隔。
- Evidence：MATLAB抽样审计12个CQ08文件/1023帧，`explicit_raw_wall_clock_time_detected=false`、`explicit_mat_wall_clock_field_detected=false`；报告明确`raw_echo_has_cpi_prt_timing=true`但`raw_echo_has_explicit_wall_clock_time=false`，RD由原始IQ重算与MAT数值一致。
- Impact：RD CPI/PRT描述单帧观测局部时间，不是相邻TR track point的墙钟时间；错误替代会制造看似精细但物理语义错误的dt，进一步污染液态动力学和运动导数。
- Action：只有获取雷达/跟踪器上游高精度时间戳或已验证的帧到TR观测时钟映射后，才能重建真实亚秒TR dt；在此之前保持`dt_valid=false`语义并避免让其可用模式成为分类特征。
- 验证状态：12文件抽样已确认；全量原始回波wall-clock字段审计未执行，且抽样不能证明全部1550文件一致。
- 关联版本：原始回波审计；V3/V9。

## Finding F085

- Severity：High（正式validation已确认流式指标与parent-final方向不完全一致）
- 文件位置：`artifacts/RADAR-LN3-V8-cq08-liquid-dbb-cu-gate-formal/best_model.pt`；`artifacts/RADAR-LN3-V9-cq08-liquid-dbb-cu-gate-motion-formal/best_model.pt`；Stage 035归因报告
- 问题或结论：V9改善parent-final与整体window指标，但Clutter的因果prefix路由退化；不能只凭最终目标结果批准在线流式Gate。
- Evidence：V8→V9 window Macro-F1=`0.741029→0.801079`，DBB→CU总数`407→216`；但Clutter→DBB错误prefix从14增至25（149个Clutter prefix），对应Clutter Gate recall约`0.906→0.832`。Parent Clutter仅8个且错误`2→1`，掩盖了早期临时结果退化。
- Impact：业务需要“窗口级临时结果+目标级稳定结果”；早期Clutter被送入DBB会产生错误临时告警，即使最终纠正也可能影响线上稳定策略和P95内可用结论。
- Action：后续Gate必须分别以prefix长度、首次稳定时间、错误持续时长和parent-final验收；checkpoint不能只按parent最小P/R选优而忽略在线prefix弱类约束。
- 验证状态：V8/V9 validation window/parent已执行；正式stable时序、端到端P95和生产告警代价未执行。
- 关联版本：V8/V9。

## Finding F086

- Severity：High（正式结果未达标且统计不稳定）
- 文件位置：`artifacts/RADAR-LN3-V9-cq08-liquid-dbb-cu-gate-motion-formal/training-summary.json`与`best_model.pt`
- 问题或结论：V9是单seed validation改善，不是达到门槛或稳定泛化；CU support和epoch波动不足以批准正式模块。
- Evidence：best epoch 7的DBB Recall=`0.947368<0.95`，CU Precision/Recall=`0.607143/0.739130<0.90`；validation CU只有23 parent（Clutter=8、Unknown=15）。12个训练epoch中CU Precision约`0.337–0.607`、Recall约`0.739–0.957`，选中epoch依赖同一小validation集。
- Impact：单个CU parent可明显改变P/R，当前提升可能来自选优噪声、初始化或采样捷径；不能外推test、raw、线上或开放集Unknown。
- Action：先修捷径，再至少多seed训练并冻结选择规则；在train parent-grouped OOF之外不得校准reject。test保持最终一次性评估，不用于模型/阈值选择。
- 验证状态：单seed正式训练和validation已执行；多seed、test、raw域、拒识和P95未执行。
- 关联版本：V9。

## Finding F087

- Severity：High（已确认修复并完成单seed正式validation，性能未达标）
- 文件位置：`src-v3/radar_ln3/features.py`、`data.py`、`task_models.py`；V10 Gate数据契约/config；V10 focused与全量特征审计报告
- 问题或结论：V10已从Gate派生支路中切断`dt/dt_valid`和时间戳精度依赖，同时保持V9可复现和模型规模不变。正式paired validation确认改进真实存在于当前single-seed split，但CU仍未达标；新9维是“每次观测变化/一致性”而非SI物理率，不能写成真实时间问题已全部解决。
- Evidence：状态机时间干预不变；CQ08 train+validation 1313 parent/25158事件中8179个非首帧秒级零dt事件仍有合法增量，validation各类valid率最大差=`0.00722476`。V9→V10 parent Macro-F1=`0.812752→0.855211`，DBB P/R=`0.970588/0.947368→0.971292/0.971292`，CU P/R=`0.607143/0.739130→0.739130/0.739130`；window Macro-F1=`0.801079→0.832280`。V10 checkpoint SHA-256=`2973806b740094d6380468d3c95bff4b70986e6128764f12cb4c07316197dec1`。
- Impact：模型不再能通过motion支路的正dt可用模式直接识别DBB/CU，且单seed validation改善；但报告频率仍会改变逐观测增量幅度，CfC动力学仍消费既有`dt/dt_valid`，CU低support和未达标禁止发布。
- Action：V10只保留为研究Gate checkpoint，不进入同版本specialists、正式级联、test或部署。下一步先复核残差标签/配对语义，再决定是否以当前事件RD物理统计分支做单因素V11。
- 验证状态：代码、契约、focused测试、CPU/CUDA Smoke、正式训练、GPU独立validation、V9/V10 paired对照、compileall和隔离均通过；test、多seed、拒识、级联、stable和P95未执行。
- 关联版本：V9只读paired对照；V10。

## Finding F088

- Severity：Medium（已确认工程限制）
- 文件位置：`config/RADAR-LN3/RADAR-LN3-V10-cq08-liquid-dbb-cu-gate-observation-increment-config.json`；`src-v3/radar_ln3/cascade.py`
- 问题或结论：V10只版本化并训练了DBB/CU Gate，没有V10 DBB和CU specialist配置/checkpoint；现有级联契约要求三个任务具有同一配置版本和共享encoder/data契约，因此V10 Gate不能与V8 specialists组合后宣称为正式三checkpoint级联。
- Evidence：V10现在已有正式Gate artifact，但仍仅有一份Gate配置和一个数据契约；`validate_cascade_configs`会拒绝混版本或共享契约不一致的三模型组合。paired报告明确`formal_cascade=未执行`。
- Impact：即使V10 Gate单独validation改善，也只能证明路由候选价值，不能直接推断最终五分类、Clutter/Unknown specialist、Unknown开放集拒识或线上stable性能。
- Action：V10 Gate已完成paired validation但CU未达标，因此不创建V10 specialist、不执行混版本联合评估。只有后续单因素Gate达到门槛后才重新评估同版本三checkpoint方案。
- 验证状态：混版本防护、V10正式Gate训练和paired validation已确认；任何V10三checkpoint组合均未执行。
- 关联版本：V8/V9历史specialists；V10 Gate。

## Finding F089

- Severity：High（已确认正式性能未达标）
- 文件位置：`artifacts/RADAR-LN3-V10-cq08-liquid-dbb-cu-gate-observation-increment-formal/best_model.pt`；`src-v3/processed/report/RADAR-LN3-V9-V10-gate-paired-validation.json`
- 问题或结论：V10修复dt可用性捷径后显著改善Gate，但只通过DBB parent门槛；CU及流式window门槛失败，不能发布为软级联第一层。
- Evidence：V10 parent矩阵=`[[203,6],[6,17]]`，DBB P/R/F1=`0.971292/0.971292/0.971292`，CU=`0.739130/0.739130/0.739130`；window DBB P/R=`0.978393/0.949901`，CU=`0.624074/0.798578`。validation CU只有23 parent。20个epoch没有一个同时达到DBB P/R≥0.95和CU P/R≥0.90；epoch 16 Macro-F1最高但CU Precision仅0.714286。
- Impact：将V10作为正式Gate会把约26.1%的validation CU parent路由进DBB，并把约2.9%的DBB parent路由进CU；线上prefix的CU Precision更低，不能靠parent-final指标掩盖。
- Action：不延长同配置训练作为默认方案；冻结V10为paired baseline，后续只允许单因素特征实验并保持同split、loss、采样、预算和选优逻辑。
- 验证状态：单seed正式训练与独立validation已完成；test、多seed、raw域、stable和P95未执行。
- 关联版本：V9 paired baseline；V10。

## Finding F090

- Severity：High（已确认错误集中与拒识盲区）
- 文件位置：V10 `sample_detail_information.json`；V9/V10 paired报告；V10残差特征报告
- 问题或结论：V10已消除validation Clutter Gate错误，剩余parent错误全部集中于Bird/Unknown边界；这些错误多为持续且高置信，0.60置信度筛选不能充当修复或开放集拒识。
- Evidence：最终错误为6个Bird→CU和6个Unknown→DBB；Drone/Balloon/Clutter为0。6个错误Bird从首prefix到末prefix始终预测CU；12个最终错误置信度最小=`0.645257`、中位数=`0.868866`、5个≥0.90、0个<0.60。Unknown仍只是闭集标签，reject/abstain未校准。
- Impact：仅记录低置信样本会漏掉全部当前最终错误；简单调低/调高Gate阈值会在Bird与Unknown之间交换误差，不能建立开放集能力。
- Action：保留完整样本明细；人工复核残差标签语义；开放集阈值仅可在train parent-grouped OOF上校准，并与UnknownTarget闭集分类分开报告。
- 验证状态：validation错误和置信度已完整统计；人工语义复核、真实外部未知、OOF拒识和selective metrics未执行。
- 关联版本：V10。

## Finding F091

- Severity：Medium（已确认结构完整性，语义标签待人工复核）
- 文件位置：`src-v3/processed/report/RADAR-LN3-V10-gate-residual-integrity-audit.json`；CQ08 manifest、V3 index和物理RD帧索引
- 问题或结论：12个V10最终残差parent没有发现文件缺失、标签码冲突、事件断裂、RD帧重复/乱序或V3漏配；但结构一致不能证明Bird/Unknown人工语义标签正确。
- Evidence：12/12航迹、点迹、原始回波和全部声明MAT存在；文件名/manifest/index标签码一致；事件与track point连续，RD帧一一映射。11/12为全RD覆盖；track 333的22个源RD帧在物理帧索引中只属于point 16–22，V3精确映射这22帧，故其7/22事件覆盖是当前源索引中的晚启动/部分覆盖。
- Impact：当前没有证据把大部分残差归因于V3数据构建错误；直接重建全量索引不太可能解决Bird/Unknown边界。track 333的早期15个prefix没有RD，必须在流式评估中单独保留。
- Action：若有标注规范或原始采集日志，优先人工复核这12个parent，特别是6个Unknown；不得依据模型预测改标签或删除困难样本。
- 验证状态：自动结构审计已通过；人工语义标签、采集意图和开放集来源复核未执行。
- 关联版本：V3数据；V10报告。

## Finding F092

- Severity：Medium（有证据的单因素候选，性能待验证）
- 文件位置：`src-v3/processed/report/RADAR-LN3-V10-gate-residual-feature-audit.json`；`src-v3/radar_ln3/model.py`的现有RD前端
- 问题或结论：V10残差中存在当前模型未显式消费的RD物理谱统计差异，但错误组很小且与正确组重叠；这只支持把“当前事件RD物理统计分支”作为下一单因素候选，不支持承诺性能提升或修改液态单元。
- Evidence：Bird错误/正确的RD零速能量中位数=`0.451032/0.291077`、谱熵=`0.775495/0.806001`；Unknown错误/正确为零速能量=`0.560757/0.676832`、谱熵=`0.628636/0.519131`。6个错误Unknown内部同时存在窄带近零速和宽带高熵样本，单阈值不可行。当前RD空间前端不显式消费物理速度轴且点内帧采用masked mean。
- Impact：一个小型、因果、逐事件RD物理统计编码器可能补充被全局池化抹去的谱位置/形状信息；同时改RD帧时序、液态单元、loss或采样会破坏归因。
- Action：只有残差人工语义复核通过后，才设计V11：复用V3帧映射，每事件最多12帧，仅增加train-only归一化的物理RD统计分支；保持CfC、TR增量、loss、class weight、采样、seed/预算和checkpoint选择不变。V11需新契约/config、因果/shape/梯度/checkpoint Smoke后再由用户决定训练。
- 验证状态：validation描述统计和结构审计已完成；该Finding形成时V11尚未执行，后续完整实施准备已由Stage 041完成；正式训练、多seed结果和test仍未执行。
- 关联版本：V10 baseline；V11实施见F104–F108。

## Finding F093

- Severity：High（历史阻塞已由用户政策裁决解除；标签不变）
- 文件位置：V10 12个Bird/Unknown残差parent；Stage 037结构与特征报告
- 问题或结论：本地代码和数据只能证明标签码、文件和配对结构一致，不能自动裁决BirdTarget/UnknownTarget的人工语义是否正确；模型高置信错误也不能作为改标签证据。
- Evidence：12/12结构检查通过，但最终错误置信度全部≥0.645257且5个≥0.90；Unknown错误样本的TR/RD分布异质。项目当前没有在已读文件中提供足以自动重标这12个parent的人工标注规范或采集意图记录。
- Impact：若直接依据V10预测修正标签，会造成确认偏差和指标污染；若完全跳过语义复核，则V11可能针对标签噪声或采集例外过拟合。
- Action：Stage 038先生成证据包；用户随后确认默认保留现有标签及多方证据重开规则，Stage 040已将12条写为`KEEP_CURRENT_LABEL`。任何标签变化仍必须另行授权并版本化。
- 验证状态：证据包、独立验证、图像抽查、防覆盖测试和12条用户政策裁决均已完成；标签修改为0。
- 关联版本：V10；后续V11实施见Stage 041。

## Finding F094

- Severity：High（历史工程阻塞已解除；后续V11已单独授权）
- 文件位置：`src-v3/processed/report/RADAR-LN3-V10-gate-residual-manual-review/manual-review.csv`及`summary.json`
- 问题或结论：12条人工决定已完成并全部保留现标签；这只解除“待人工决定”的阻塞，当时不自动授权V11设计、配置、训练或test。后续V11完整实施准备已由Stage 041单独授权。
- Evidence：`review_decisions_completed=12`、`KEEP_CURRENT_LABEL=12`、`automatic_label_changes=0`；独立验证状态=`completed_keep_current_labels_under_user_policy`，V11=`ready_for_design_pending_explicit_authorization`。manual-review SHA-256=`5f8eff2749e5e32710d4bd3f9dc0397da4fe02c1b3a52a2bf1382cfe0b03195d`。
- Impact：后续实验按现标签评估且不得声称标签被重新证明；V11实施授权不包含改标、正式训练或test。
- Action：V11实施准备已按后续授权完成；任何改标仍需多方独立证据和单独申请。
- 验证状态：12条裁决、hash、字段约束、防覆盖和非法决定focused测试均通过；V11实施准备完成，正式训练/test未执行。
- 关联版本：V10；V11见Stage 041。

## Finding F095

- Severity：High（已确认残差异质性；根因尚未单因素验证）
- 文件位置：`src-v3/processed/report/RADAR-LN3-V10-gate-residual-manual-review/all-events.csv`、`manual-review.csv`；V10 validation sample detail
- 问题或结论：12个最终Gate残差至少包含三种不同的因果时序模式，不能统一解释为“液态状态后期遗忘”或用单一阈值修复。
- Evidence：333/845/852/2367/2956五个Bird从首prefix至末prefix始终为CU；3581的`P(CU)=0.522–0.766`也始终在CU侧。2862与4171从首prefix起始终为DBB，最终`P(CU)=0.009/0.103`。2908/3122/3154分别在前11/6/5个prefix为CU，随后各一次跨界并最终为DBB。1547起始`P(CU)=0.238`、prefix 6峰值0.733、四次跨界、最终0.316。用户对4171“早期正确”的视觉描述与机器记录不一致，已在备注中并列保留。
- Impact：持续型错误更接近当前事件表征/Gate边界问题；单次后期反转可能来自后段观测分布改变与状态累积共同作用；多次跨界需要校准/稳定性分析。盲目调大液态隐藏维、层数或统一stable阈值无法形成可靠归因。
- Action：先保持V10权重与标签不变；人工语义裁决完成后，分别对持续型、单次反转型和多次跨界型做受控输入分支/状态干预，不把三类合并为一个优化目标。
- 验证状态：逐prefix曲线和路由已全量复核；模型内部表征干预、counterfactual state reset、多seed和test未执行。
- 关联版本：V10；V11实施准备见Stage 041。

## Finding F096

- Severity：High（有证据的表征瓶颈；性能收益待验证）
- 文件位置：V10残差特征报告、人工复核`all-events.csv`、`src-v3/radar_ln3/task_models.py`的`LiquidTaskClassifier`
- 问题或结论：V10失败不能简化为“Bird/Unknown缺少运动”。错误Bird和错误Unknown往往具有明显、连续且笔直的TR运动；DBB/CU标签语义并不等于运动/非运动，现有融合边界可能把运动形态与已知DBB身份错误绑定。
- Evidence：错误/正确Bird的parent中位几何步长为`0.258385/0.036363`、路径直线度为`0.937252/0.876090`；错误/正确Unknown为几何步长`0.225203/0.238855`、直线度`0.938388/0.890592`。模型将TR、9维增量、RD embedding和7维质量直接拼接后压缩进单一CfC状态，再由二分类头判DBB/CU；没有显式的“运动存在性”与“known DBB身份/开放集证据”解耦头。
- Impact：仅增强motion分支可能继续把运动型Unknown推向DBB，也可能把某些Bird推向CU；运动显著并不足以证明属于DBB。Gate要同时承担闭集粗路由与Unknown语义，天然存在边界压力。
- Action：不改标签、不直接增加液态复杂度。人工裁决后优先评估当前事件RD物理统计分支及表征可分性；开放集Unknown另需train/OOF校准的reject/abstain机制，不能由DBB/CU闭集概率替代。
- 验证状态：validation描述统计和代码链路已确认；V11实施准备已完成，独立运动存在性辅助目标、开放集校准、V11正式训练和test未执行。
- 关联版本：V10；V11见Stage 041。

## Finding F097

- Severity：Medium（已确认个案机制，跨样本泛化待验证）
- 文件位置：`all-events.csv`中track 2908、3122、3154和1547的逐事件TR/RD物理统计
- 问题或结论：后期Unknown→DBB反转并非都由同一种输入变化驱动；2908有强烈后段RD形态迁移，而3122/3154方向不一致。
- Evidence：2908在CU前段与DBB后段间，RD带宽中位数`10.584→39.574 m/s`、零速能量`0.954→0.191`、谱熵`0.193→0.909`、profile change`0.025→0.138`；TR步长基本不变。3122在后段零速能量上升、谱熵下降；3154后段TR步长和方向一致性上升，但RD统计变化较弱。1547四次跨界均有RD且发生在不同RD形态下。
- Impact：2908更像当前事件RD证据真实改变并被液态状态累计，而不是纯遗忘；对3122/3154套用2908规则会过拟合3个validation parent。单一RD阈值或统一状态衰减不能被当前证据支持。
- Action：若人工标签保留，V11只允许加入因果当前事件RD物理统计分支作为单一变量；同时设计只读state-reset/last-N-event反事实评估区分“输入变化”与“状态惯性”，不得用validation拟合部署阈值。
- 验证状态：跨界前后描述统计已完成；V11当前事件四维RD分支实施准备已完成，反事实推理、OOF校准、正式训练和test未执行。
- 关联版本：V10；V11见Stage 041。

## Finding F098

- Severity：High（独立真值证据仍缺失；工程裁决已完成）
- 文件位置：`src-v3/processed/report/RADAR-LN3-V10-gate-residual-manual-review/manual-review.csv`、Stage 038审核包
- 问题或结论：12个V10残差parent目前只能进行证据对比，不能仅凭模型路由、TR运动强度、RD谱统计或训练分布百分位完成权威语义裁决；指标图应作为人工复核辅助，不应自动填入`KEEP_CURRENT_LABEL`、`SUSPECT_MISLABEL`或其他决定。
- Evidence：Stage 038/039确认没有足以自动重标的独立采集记录，错误Bird/Unknown与同类TR/RD分布重叠；用户据此确认当前以标签为准，12条现均为`KEEP_CURRENT_LABEL`且原语义标签不变。
- Impact：若把“超出P05-P95”或“模型高置信错误”当作误标证据，会将困难样本、开放集样本和真实标签冲突混为一谈，污染V11数据与性能评估；若把运动明显直接当作DBB，也会重复当前Gate的语义混淆。
- Action：保留现标签；只有多方独立证据共同冲突时才向用户重新申请，禁止依据模型/单指标自动改标。
- 验证状态：指标报告、图像查看和12条保留标签决定已完成；V11实施准备后续已完成，标签修改和拒识阈值均未执行。
- 关联版本：V10；V11见Stage 041。

## Finding F099

- Severity：High（已确认残差指标无单一安全分界；当前标签政策已裁决）
- 文件位置：`src-v3/processed/report/RADAR-LN3-V10-gate-residual-manual-review/metrics/`、`manual-view-notes.md`、V3 CQ08 train参考
- 问题或结论：12个V10残差的TR/RD parent级指标与同标签CQ08 train分布存在重叠；部分样本位于同类分布尾部，但没有证据支持用P05-P95、模型置信度或单一运动/RD阈值自动改标或实现Unknown拒识。
- Evidence：以CQ08 V3 train的1081 parent（Bird=243、Unknown=70）计算parent中位数参考，12个残差为232事件、217个RD-present和15个RD-missing；总体箱线图显示TR步长/直线度及RD带宽/零速能量/谱熵均有大量重叠。track 333为7/22 RD覆盖；2908出现后段RD变化并同步路由反转；2862与4171分别为近零速集中和宽谱高熵形态；Unknown内部并非单一中心。
- Impact：把尾部样本直接重标会污染训练和验证，把运动强度作为DBB证据会继续误伤运动型Unknown或Bird；把Unknown闭集概率当reject会产生开放集错误接受。
- Action：保留指标、图和人工查看记录；当前12条保留原标签。V11后续已单独授权完成实施准备，但没有由本报告自动放行正式训练或test。
- 验证状态：报告统计、PNG查看、编译/隔离和12条保留标签裁决已完成；V11实施准备已完成，标签改动、阈值拟合、正式训练和test未执行。
- 关联版本：V10；V11见Stage 041。

## Finding F100

- Severity：High（用户已确认语义边界；标签不变）
- 文件位置：Stage 038/039的12个V10 Bird/Unknown残差parent及人工审核表
- 问题或结论：UnknownTarget是包含风筝、昆虫等杂物的异质集合，Bird也可能处于静止/低速状态；运动强度、单一TR/RD指标或模型高置信错误均不能作为DBB归属或错标证据。原始采集记录已丢失时，当前工程裁决为保留现有标签，除非未来出现多方独立反证。
- Evidence：用户于2026-08-08明确给出四项语义规则；Stage 039同类train参考与残差指标也显示Bird/Unknown在TR运动和RD谱形上大量重叠，没有单一安全阈值。
- Impact：12个残差不得因运动强弱、V10路由或置信度重标；Unknown开放集拒识必须与闭集Gate分开设计。保留标签不等同于重新证明标签客观正确。
- Action：将12条审核决定记录为`KEEP_CURRENT_LABEL`并注明政策依据；未来只有源标签、TR/RD、独立模型/人工背景等多方证据共同冲突时，才向用户提交改标申请。
- 验证状态：语义规则已由用户确认；审核表12条正式写入，summary/hash和独立验证通过，标签修改=0。
- 关联版本：V10；标签与V3数据不变。

## Finding F101

- Severity：Medium（已确认源端部分RD覆盖；Python可恢复性审计通过）
- 文件位置：`OriData/23/CQ-08/原始回波/333_Label_3.dat`、Track 333的22个MAT、V3点级索引和物理RD帧索引
- 问题或结论：Track 333前15个TR点没有可从指定原始`.dat`恢复的RD。该文件从byte 0到EOF只有22个合法帧，point号仅16–22；现有MAT和V3没有漏掉raw中存在的帧。
- Evidence：`.dat`大小和解析字节均`2,694,512`，22帧header/footer有效、未解析payload=0；point帧数=`4,4,4,2,2,2,4`。22/22 raw/MAT target record、元数据和I/Q完全一致，I/Q最大绝对/相对误差=0；raw/MAT/物理/V3逐point映射完全一致。focused测试核对25个输入前后SHA-256不变。
- Impact：缺失是源端晚启动/部分采集，前15个prefix必须保留显式RD missing mask；重建索引或伪造RD不会恢复信息，反而会污染因果输入。
- Action：不重建数据。保留point 1–15的RD缺失；由用户手动运行MATLAB脚本完成独立`taylorwin` RD数值复核，若冲突则先定位解析差异。
- 验证状态：Python全文件解析、MAT/IQ/映射对照、不可变输入focused测试、MATLAB v2全帧I/Q/RD重算、独立结果验证和图像查看均通过；双路径收口完成。
- 关联版本：V3数据；V10残差；不创建新版本。

## Finding F102

- Severity：Medium（已确认审计脚本假阴性；数据无冲突）
- 文件位置：`src-v3/tests/processed/track_0333_original_echo_audit.m`的`buildCoverageRows`；首次MATLAB `point-coverage.csv`和`summary.json`
- 问题或结论：首次MATLAB审计把point 1–15的“raw与MAT均无帧”误判为coverage不一致。原因是`rawSequences=[]`为`0x0`，而空`find`结果为`1x0`，`isequal`比较形状后返回false；不是数据缺失语义冲突。
- Evidence：首次MATLAB帧级比较22/22均`matched`，I/Q和RD最大误差均为0；point 16–22 coverage逐行true。point 1–15两侧count均为0，CSV仅在空序列文本`[]`与`zeros(1,0)`及布尔比较上不同。summary唯一失败字段为`all_point_coverage_raw_matches_mat=false`。
- Impact：造成总体`status=needs_review`和`inconclusive_or_source_mapping_conflict`假阴性，可能错误阻塞已经由双路径帧级证据支持的结论；不影响原始数据、MAT、V3输入或模型。
- Action：将coverage比较改为`isequal(rawSequences(:), matSequences(:))`，统一空集合文本为`[]`，保留首次报告并请用户原命令重跑。
- 验证状态：根因已由首次实际CSV和MATLAB维度语义确认；方向无关比较、独立v2输出目录和结果验证器已实现；用户重跑后v2为passed，22点coverage全true，独立验证与图像检查通过。该假阴性已解决。
- 关联版本：不涉及RADAR-LN3数据/配置版本；仅审计脚本。

## Finding F103

- Severity：Medium（已确认最终数据完整性结论）
- 文件位置：Track 333 Python summary、MATLAB v2 summary、`cross-validation-summary.json`及RD/coverage图
- 问题或结论：Python与MATLAB两条独立路径一致证明Track 333原始`.dat`只保存point 16–22的22帧，point 1–15没有可恢复回波；现有22个MAT和V3映射完整且数值精确。
- Evidence：Python和MATLAB均解析22帧、point分布`4/4/4/2/2/2/4`；raw/MAT元数据与I/Q一致，MATLAB重算RD最大绝对/相对误差均为0；物理帧索引和V3逐点映射一致；coverage v2全部true；raw SHA-256在两路径均为`70bee1da3bf665d8436cd855d5d32ea97f9737fa05457388643b70ff60ef27a0`。
- Impact：重建数据无法恢复前15点信息；伪造、前向填充或把后段RD回填前缀会制造未来信息和因果泄漏。当前显式missing mask是正确输入语义。
- Action：保持V3数据和Track 333前15点RD missing不变；任何模型实验按真实coverage分层报告，不把缺失本身作为Bird/Unknown标签或拒识证据。
- 验证状态：Python、MATLAB v2、独立验证器和图像查看均通过；数据重建、训练、test未执行且本结论不要求执行。
- 关联版本：V3数据保持不变；V10残差审计；不创建新版本。

## Finding F104

- Severity：High（技术实施已确认；性能仍待正式训练）
- 文件位置：`src-v3/radar_ln3/rd_physical.py`、`data/processed/RADAR-LN3-V11-*`、六份V11配置和Stage 041 readiness
- 问题或结论：V11单因素Treatment已完整实现：只把当前事件四维RD物理谱值追加到V10 Gate Fusion，输入127→131；液态单元、TR增量、RD CNN、标签、split、loss、采样预算和评估逻辑保持不变。实现准备可验证，但没有证据表明分类性能已提升。
- Evidence：V11 train/val缓存分别20684/4474事件，valid=20482/4434；全部25158事件独立同源复算最大误差0。三seed共享参数/buffer逐值一致、新四列全零、CPU非零探针初始logits逐位一致；CPU/CUDA forward/loss/gradient/update/checkpoint Smoke通过。训练前readiness=`passed`且当时正式artifact为0；当前用户已手动启动paired正式训练，尚无完整联合性能结果。
- Impact：现在可以进行低维单因素paired归因；Smoke、链路复杂度或梯度非零都不能冒充DBB/CU性能改善。
- Action：由用户手动完成六次正式训练后运行联合报告；只有满足2/3 seed、中位数和离散Pareto才讨论第二阶段。
- 验证状态：代码、缓存、契约、配置、全量验证、CPU/CUDA Smoke、跨账户ACL回归和训练前readiness已完成；用户手动正式训练进行中。完整六run validation性能、test、OOF、拒识和P95尚未执行或未完成。
- 关联版本：V10 Control baseline；V11 Treatment/paired实施准备。

## Finding F105

- Severity：Medium（复现性边界已实测确认）
- 文件位置：Stage 041 paired训练协议
- 问题或结论：用户选择只固定seed和DataLoader顺序，不启用CUDA严格确定性算法，因此V11只能宣称“固定共享初始化与采样顺序的paired实验”，不能宣称CUDA位级确定性复现。
- Evidence：seed42/43/44的CPU参考初始logits均逐位一致；seed42 CUDA两次独立arm前向最大差=`3.427267074584961e-07`，在`5e-6`数值容差内但非逐位一致。独立同seed DataLoader generator在Smoke中产生相同parent顺序hash，正式入口会逐epoch记录。
- Impact：Control/Treatment单seed差异可获得较强归因，但极小数值差不能被解释为确定性模型收益；必须依赖3 seed、离散Pareto和中位数门槛判断。
- Action：训练摘要已编码确定性级别、CPU初值hash、loader顺序hash和运行环境；联合报告拒绝任一epoch顺序hash不匹配。
- 验证状态：CPU/CUDA及DataLoader focused验证通过；六次正式训练由用户手动进行中，尚未完成联合报告。
- 关联版本：V11。

## Finding F106

- Severity：High（数据缓存完整性已确认）
- 文件位置：V11 train/val NPZ、cache manifest、数据/输入契约、`src-v3/validate/rd_physical_paired.py`
- 问题或结论：V11缓存与V3 train/val事件集合、顺序、RD-present语义和在线共享公式完全一致；Track 333缺失语义未被回填。test没有进入物理特征构建或评估。
- Evidence：Train 20684事件、重复0、valid=20482；Val 4474事件、重复0、valid=4434；25158事件逐值复算最大误差0，`rd_present==valid`。Track 333 point 1–15全零/invalid、16–22 valid=7。共享V3 index扫描中4505个test行仅按split计数排除，test RD payload访问=0、test cache不存在。
- Impact：V11性能差异不会由cache错位、缺失回填、速度轴漂移或test RD特征污染解释；共享CSV元数据扫描不应被夸大为“test文件字节完全未读”。
- Action：正式训练前继续以manifest/hash为硬门；任何缓存或契约内容变化必须新版本，禁止覆盖V11。
- 验证状态：构建、独立全量验证和幂等重建均通过。
- 关联版本：V3 index/V2统计只读复用；V11缓存与契约。

## Finding F107

- Severity：High（训练恢复与artifact安全已确认）
- 文件位置：`src-v3/train/tasks.py`、`src-v3/radar_ln3/task_models.py`、V11六配置
- 问题或结论：原任务checkpoint只保存模型/优化器，不能恢复paired采样轨迹；V11已补齐epoch边界完整恢复状态并禁止有界调试参数进入正式artifact。
- Evidence：V11每epoch保存模型、优化器、Python/NumPy/Torch CPU/CUDA RNG、DataLoader generator、global step、best selection和history；CPU/CUDA round-trip全部精确恢复。`--max-steps 1`训练前测试在创建artifact前按设计拒绝；当时六个artifact不存在，当前artifact由用户手动正式训练按协议生成中。
- Impact：中断后可从同arm epoch checkpoint显式恢复，不会静默重头训练或改变后续shuffle；历史V6–V10 checkpoint读取契约保持兼容。
- Action：正式运行中断时仅使用同arm artifact内的`epochs/epoch-XXX.pt`并显式传`--resume-checkpoint`；不得删除/覆盖现有artifact后伪装同一次run。
- 验证状态：focused保存恢复、artifact路径和调试隔离通过；正式训练进行中，真实中断恢复是否发生未核实。
- 关联版本：V11。

## Finding F108

- Severity：High（待训练放行结论）
- 文件位置：`src-v3/smoke/RADAR-LN3-V11-rd-physical-training-readiness.json`
- 问题或结论：Stage 041 readiness遗漏的Windows跨账户缓存可读性问题已由Stage 042闭环；训练放行恢复。用户确认修复后正式训练正常进行，但这不构成V11性能结论。
- Evidence：原seed42/43均在`task_models.py:592`读取同一train cache时报`PermissionError`；修复后train/val均有9条父目录继承规则和通用用户读取权，固定SHA-256不变。seed42/43/44三份Treatment契约、真实Dataset读取和用户交互式正式入口均通过。
- Impact：Treatment三seed不再被共享缓存ACL阻断；数据内容、paired归因和V11版本语义未变化。完整性能仍必须等待六run和联合报告，任何进行中日志不能替代。
- Action：允许用户继续手动训练；完成六run后运行`python -m report.rd_physical_paired`。训练进行中不得重跑要求artifact全空的训练前readiness。
- 验证状态：ACL代码、现有缓存、focused回归和用户侧实际训练均验证通过；正式训练进行中，完整validation性能、test、OOF、阈值、正式级联、P95、部署尚未执行或未完成。
- 关联版本：V11。

## Finding F109

- Severity：Critical（已确认并修复的训练启动阻塞）
- 文件位置：`src-v3/build/rd_physical_paired.py::_write_npz_new_or_identical`；`data/processed/RADAR-LN3-V11-cq08-{train,val}-rd-physical-event-cache.npz`；`src-v3/radar_ln3/task_models.py:592-593`
- 问题或结论：V11 NPZ先在受保护的临时目录内创建，再以`rename`移动到`data/processed`。Windows保留源安全描述符，导致最终train/val缓存关闭父目录ACL继承，仅生成账户、SYSTEM和管理员可读；用户交互式训练账户无法完成训练前hash校验。
- Evidence：seed42与seed43错误堆栈的文件路径和代码行完全相同；两配置的train/val路径与hash完全相同。两个NPZ为普通非链接文件，但`AreAccessRulesProtected=True`，ACE仅含OWNER RIGHTS/SYSTEM/Administrators；父目录`AreAccessRulesProtected=False`且向Authenticated Users/Users授予访问。同构建器JSON直接创建文件均正常继承。
- Impact：三份Treatment配置均无法由用户账户启动；修复train后若不同时修复val，会在下一行继续失败。Control不消费四维缓存，因此不能以Control成功排除该问题。
- Action：已改为内存生成确定性NPZ、`xb`直接发布、Windows `icacls /reset`采用父目录DACL；内容相同复用路径同样规范化ACL。现有两份缓存已修复并增加focused回归。
- 验证状态：已解决。Train/val分别保持`346504/75717`字节与冻结SHA-256，均有9条继承规则和通用用户读取权；三seed Treatment契约、Dataset及用户侧实际训练入口通过。性能提升未由本修复验证。
- 关联版本：V11内容不变；仅代码与文件系统ACL元数据修复，不递增版本。

## Finding F110

- Severity：High（有正式结果证据，独立复核进行中）
- 文件位置：六个`artifacts/RADAR-LN3-V11-cq08-liquid-dbb-cu-gate-rd-physical-*-formal/`、`src-v3/processed/report/RADAR-LN3-V11-rd-physical-paired-validation.json`
- 问题或结论：V11三seed Control/Treatment六次正式训练均完整结束；四维当前事件RD物理谱Treatment未通过第一阶段Gate门槛，不能进入第二阶段OOF或正式级联。该结论已由当前代码独立重载六checkpoint复核定稿。
- Evidence：六个artifact各有20个epoch checkpoint、best/last、summary和完成日志；独立CUDA重算与summary逐值一致，联合报告状态=`completed_but_gate_failed`，通过seed=`0/3`。seed42 Treatment虽降低CU→DBB但增加DBB→CU，三seedCU Precision均低于0.90。
- Impact：四维分支可能改善CU召回但以更多DBB误拒为代价，尚未形成可泛化的Pareto收益；不能只看总体accuracy或单seed CU Recall放行。
- Action：保留V11为失败的受控消融；深层残差与knownness诊断见F111–F116，不进入第二阶段。
- 验证状态：训练、独立复算、同epoch/全epoch护栏、parent残差和统计边界均已确认；test/OOF/P95未执行。
- 关联版本：V11。

## Finding F111

- Severity：Critical（已确认正式性能门槛失败）
- 文件位置：`src-v3/processed/report/RADAR-LN3-V11-rd-physical-paired-validation.json`、六个V11正式artifact
- 问题或结论：V11当前事件四维RD物理谱Treatment在三seed paired正式validation中第一阶段失败，不能进入第二阶段OOF、正式级联或部署。
- Evidence：六个best checkpoint在CUDA上独立重算232 parent/4474 window并与training summary逐值一致；seed42/43/44均`gate.passed=false`，通过数`0/3`。三seed中位Treatment DBB P/R/F1=`0.980392/0.956938/0.968523`，CU=`0.678571/0.826087/0.745098`；Macro-F1=`0.856811`低于Control中位`0.861543`。中位误路由DBB→CU=`9`、CU→DBB=`4`。
- Impact：DBB已满足第一阶段门槛，但CU Precision远低于0.90；四维分支没有形成跨seed稳定Pareto收益。继续第二阶段会在未过预冻结门槛时消耗验证自由度，并可能把validation偶然变化当成模块收益。
- Action：保持V11为失败的受控实验记录；禁止提升为正式Gate、级联模块或部署版本。若启动新实验，必须新授权并维持单一变量。
- 验证状态：已确认。Checkpoint、summary、20 epoch顺序hash和初始CPU logits hash均核验；test/OOF/P95未执行。
- 关联版本：V11。

## Finding F112

- Severity：High（已确认不是checkpoint选优偶然）
- 文件位置：`src-v3/processed/report/RADAR-LN3-V11-rd-physical-residual-analysis.json::epoch_trajectory`
- 问题或结论：V11门槛失败不能由“best checkpoint选错”解释；六条20 epoch轨迹中不存在任何同时满足DBB floor≥0.95与CU floor≥0.90的epoch。
- Evidence：Control/Treatment共120个arm-epoch，双门槛通过数均为0。Treatment在DBB floor≥0.95约束下，每seed最佳CU floor仅`0.724138/0.678571/0.678571`；Control对应`0.730769/0.666667/0.642857`。训练按预冻结`parent_min_precision_recall`选优且各自best/同epoch护栏均保留。
- Impact：改best选择指标、事后挑epoch或延长对现有轨迹的观察不能制造第一阶段通过；这样做会产生Garden of Forking Paths和validation过拟合。
- Action：保留全轨迹作为否定证据；不事后切换window/parent/macro-F1/某弱类指标选checkpoint。新实验需先冻结唯一表征/目标变量。
- 验证状态：已确认；20 epoch summary全部读取并校验完整。
- 关联版本：V11。

## Finding F113

- Severity：High（已确认主要残差为Bird/Unknown而非Clutter或RD缺失）
- 文件位置：`RADAR-LN3-V11-rd-physical-residual-analysis.json::predictions`、parent transitions CSV
- 问题或结论：Treatment剩余共同误路由集中在Bird→CU和Unknown→DBB；Clutter路由已经稳定，RD缺失不是总体根因。
- Evidence：Treatment三seed共同错误包括Bird 2367/2956/333/3581/444/845/851/852及Unknown 2862/4171；Clutter 8/8在三seed均正确。除Track 333只有7/22 RD覆盖，其余共同错误RD完整。三seed合计Treatment仅修复6个、回退3个，Track 3154跨seed方向相反。
- Impact：继续增强“静态/低速Clutter”物理规则不能解决主要Unknown/Bird重叠；将RD missing视作Bird→CU主因会误导数据重建。高总体Accuracy被DBB基数掩盖，CU Precision仍受Bird误拒支配。
- Action：后续表征应显式保护Bird作为DBB已知亚类，并单独处理Unknown拒识；Track 333继续保留真实missing mask，不回填。
- 验证状态：已确认parent集合、类别、长度和RD coverage；人工标签仍按既有12条`KEEP_CURRENT_LABEL`政策保留，不等于独立真值证明。
- 关联版本：V11评估，V3数据不变。

## Finding F114

- Severity：High（已确认四维物理摘要信息不足以稳定解决Unknown/Bird）
- 文件位置：`src-v3/report/rd_physical_residuals.py`、`RADAR-LN3-V11-rd-physical-residual-analysis.json::physical_feature_description`
- 问题或结论：四维RD物理谱特征是现有RD CNN输入的确定性压缩，能够强化部分Clutter物理结构，但对Unknown/Bird分离弱，未形成正式收益。
- Evidence：Train final-event Unknown/Bird四特征directionless AUC约`0.5346–0.5785`，validation final-event约`0.5218–0.5513`；valid-event mean虽在小validation上最高约`0.6679`，仍有显著重叠。Clutter/Bird的zero-velocity/entropy等分离较强，与Clutter 8/8正确相符。四个新增Fusion列权重均非零，但列L2约`0.05–0.13`，基础列中位L2约`0.36`；正式paired指标未改善。
- Impact：不能把“权重学到非零”或单变量AUC当作分类性能；四维摘要没有新增传感信息，也无法覆盖Unknown内部风筝、昆虫等异质流型。
- Action：拒绝把当前四维分支提升为正式模块；保留实现和报告作为可复现的失败消融。不得通过增加液态隐藏维度/层数掩盖该信息与目标边界问题。
- 验证状态：有train/validation描述统计、三seed权重和正式paired实测；没有test或外部域验证。
- 关联版本：V11。

## Finding F115

- Severity：High（已确认当前静态拒识分数不可直接部署）
- 文件位置：`src-v3/report/liquid_knownness.py`、`RADAR-LN3-V11-liquid-dbb-knownness-diagnostic.json` schema v2
- 问题或结论：当前`P(CU)`排序能力较强，但Energy及两类train-only DBB中心距离没有提供能满足第一阶段门槛的独立静态拒识边界；直接“远离DBB中心→CU/Unknown”会继续误拒困难Bird。
- Evidence：三seed中位AUC中，`P(CU)`的CU/DBB=`0.97462`、Unknown/Bird=`0.88846`；余弦中心距离约`0.97400/0.87949`，没有明显增益。共同误拒Bird通常位于train DBB距离的96–99百分位。validation oracle穷举且不输出阈值：四种分数在三个seed均`stage_one_feasible=false`；CU Recall≥0.90时`P(CU)`最少DBB→CU仍为`8/18/13`，Energy为`29/18/62`，余弦距离为`9/20/15`。
- Impact：单纯调softmax阈值、Energy阈值或Mahalanobis/中心距离阈值不能达到DBB/CU门槛，并可能把Bird高置信误拒变得更坚定。validation oracle不能作为线上阈值来源。
- Action：不部署当前距离/Energy规则；Unknown仍不作为DBB已知原型。下一候选仅讨论train-only的DBB三亚类辅助表征/度量目标，使Bird/Drone/Balloon各自形成已知流型，再以独立OOF校准拒识；需要新授权和新版本实验。
- 验证状态：诊断完成；阈值拟合、OOF、test、线上reject和P95未执行。
- 关联版本：V11诊断，不改变数据/配置版本。

## Finding F116

- Severity：High（统计解释限制已确认）
- 文件位置：V11 paired、残差和knownness报告；CQ08 V3 validation split
- 问题或结论：V11结论只对固定CQ08 V3 validation和当前协议有效；CU低support、同一validation跨seed复用、多指标/epoch查看以及CUDA非位级确定性限制了外推强度。
- Evidence：Validation共232 parent但CU仅23（Clutter 8、Unknown 15）；三个seed复用同一parent集合，属于优化重复而非独立样本。报告查看6 run×20 epoch、多层指标及四种oracle分数；门槛和主比较预冻结，但oracle属于探索性可达性诊断。`cuda_deterministic_algorithms=false`，仅共享初值与DataLoader顺序固定；best checkpoint前向指标已精确复现，随机训练未重跑。
- Impact：不能给三seed中位数附加独立样本置信区间、不能用单个parent变化宣称泛化，也不能把validation oracle当部署阈值；不能把当前失败因果外推成“所有RD物理特征或所有液态网络均无效”。
- Action：11/11统计谬误扫描已记录；结论限定为“当前四维Treatment在冻结协议下失败”。未来拒识阈值必须来自train parent-grouped OOF，最终性能需独立test授权。
- 验证状态：统计扫描11/11完成；复现判定为范围内已验证、训练位级复现未验证。
- 关联版本：V11。

## Finding F117

- Severity：High（有正式残差证据支持、尚待V12单因素实验验证）
- 文件位置：`src-v3/processed/report/RADAR-LN3-V11-rd-physical-residual-analysis.json`、`RADAR-LN3-V11-liquid-dbb-knownness-diagnostic.json`、拟实现的V12辅助表征契约
- 问题或结论：V11 Gate主要共同残差是Bird→CU与Unknown→DBB；直接中心距离会同时把困难Bird判为异常。下一项可归因候选不是增加液态复杂度或部署距离阈值，而是仅用train中的Drone/Bird/Balloon三亚类标签约束液态共享表征，使困难Bird保留在DBB已知流型内。
- Evidence：V11三seedTreatment共同误拒8个Bird，共同误收2个Unknown；困难Bird通常也位于train DBB中心距离96–99百分位。CQ08 V3 train parent中Drone/Bird/Balloon为488/243/245，具备辅助监督覆盖；Clutter/Unknown为35/70且语义不应并入DBB knownness。该证据支持实验必要性，但尚不证明辅助头会改善Gate。
- Impact：若辅助表征有效，应优先降低Bird→CU且不增加Unknown→DBB；若只提高DBB Recall并进一步降低CU Precision，则不能通过Gate门槛。任何收益必须由同数据、同顺序、三seed Control/Treatment正式validation证明。
- Action：实施V12同构辅助头paired准备。Control权重0、Treatment固定0.20；辅助CE只作用于DBB parent最后一个合法因果状态，使用train-only固定逆频率权重且不搜索，CU严格零贡献；不叠加其他正则、度量损失或拒识分数。
- 验证状态：代码、六配置、CPU/CUDA Smoke、V11 Control等价、checkpoint/在线接口、训练后联合评估器、目标机制门槛、编译和隔离均已通过；正式训练、真实性能、test和开放集拒识未执行。
- 关联版本：V12配置语义；数据继续复用V3/V10/V2，不重建。

## Finding F118

- Severity：High（已确认并修复的窗口目标风险）
- 文件位置：`src-v3/radar_ln3/task_models.py::dbb_auxiliary_final_state_loss`、V12六配置`dbb_auxiliary_supervision_scope`
- 问题或结论：V12初始设计草案若对所有causal prefix使用完整parent的Drone/Bird/Balloon亚类标签，会重新引入早期窗口证据不足与完整轨迹标签冲突。该问题在配置创建和任何训练前被审查发现并修正为仅监督最后一个合法因果状态。
- Evidence：现有主Gate明确区分窗口预测与parent final结果；Stage 043残差含低速/静止Bird，不能证明每个早期prefix已有充分亚类证据。最终Smoke证明早期/padding辅助logit梯度精确为0，只有DBB final state非零；配置和checkpoint均记录`train_dbb_parents_final_causal_state_only`。
- Impact：避免辅助头以错误窗口目标污染液态状态，保持主Gate所有prefix监督与辅助final-state表征任务的语义边界；代价是辅助监督数量由所有prefix降为每个DBB parent一次。
- Action：冻结final-state-only，不在V12中搜索prefix权重或扩大到早期窗口；若未来研究早期亚类监督，必须建立独立合法窗口标签契约。
- 验证状态：已解决并由CPU/CUDA梯度focused测试、配置校验和checkpoint契约确认；正式性能待训练。
- 关联版本：V12。

## Finding F119

- Severity：High（已确认实施准备通过，性能未验证）
- 文件位置：`src-v3/smoke/RADAR-LN3-V12-dbb-auxiliary-training-readiness.json`、CPU/CUDA Smoke结果、六份V12配置
- 问题或结论：V12 DBB三亚类辅助表征实验曾达到用户手动paired正式训练状态；该训练前结论已由正式Stage 045结果取代。Smoke只证明契约、梯度、恢复和归因，不证明DBB/CU性能改善。
- Evidence：CPU/CUDA真实CQ08混合batch均通过；CU梯度0、Control辅助更新0、Treatment辅助梯度到达共享表征、offline/online一致、checkpoint严格恢复。V12 Control与V11 Control 42个共享state在初值和一次主更新后逐值一致。readiness状态=`ready_for_user_manual_paired_training`且六formal artifact为空。
- Impact：当时可进入受控三seed训练且无需预训练或数据重建；不得把Smoke loss、梯度或单步差异写成分类指标。Validation仍仅23个CU parent，三seed共享同一validation，不是独立样本。
- Action：六次训练和冻结报告后来已完成，正式失败结论见F120–F123；V12不得进入OOF、级联或部署。
- 验证状态：训练前实施准备已通过；正式validation已完成并失败，test、OOF、stable和P95仍未执行。
- 关联版本：V12。

## Finding F120

- Severity：Critical（已确认正式保留门槛失败）
- 文件位置：`src-v3/processed/report/RADAR-LN3-V12-dbb-auxiliary-paired-validation.json`、六个V12 formal artifact
- 问题或结论：V12仅对DBB parent final causal state施加权重0.20的Drone/Bird/Balloon辅助CE，未通过冻结DBB/CU Gate保留门槛；必须归档为失败受控消融，不能进入OOF、正式级联、test或部署。
- Evidence：CUDA独立重载六best checkpoint并与summary逐值一致。Treatment seed42/43/44均`gate.passed=false`，通过`0/3`；三seed中位DBB P/R/F1=`0.975610/0.966507/0.966184`满足0.95，但CU=`0.681818/0.782609/0.720000`远低于0.90。中位Accuracy=`0.939655`、Macro-F1=`0.843092`，均低于Control的`0.943966/0.861543`。
- Impact：辅助监督没有形成可接受的DBB/CU边界；以总体Accuracy或DBB Recall放行会掩盖CU Precision/Recall失败和危险Unknown误接收。
- Action：冻结`v12_retention_passed=false`；不执行第二阶段、OOF、阈值拟合、正式级联、stable、P95或test。禁止通过事后换checkpoint或只挑seed44宣称收益。
- 验证状态：已确认；六run完整20 epoch，初始化/采样配对和best checkpoint独立复算通过。test等后续阶段未执行。
- 关联版本：V12。

## Finding F121

- Severity：Critical（目标机制与安全Pareto失败）
- 文件位置：`RADAR-LN3-V12-dbb-auxiliary-paired-validation.json::targeted_mechanism_gate`
- 问题或结论：V12降低部分Bird误拒的同时增加Unknown被接收到DBB的风险，未达到“保护Bird且Unknown不恶化”的预冻结机制门槛。
- Evidence：Bird→CU Control→Treatment变化按seed为`7→7`、`10→7`、`8→8`，减少`0/3/0`；Unknown→DBB为`4→5`、`3→7`、`5→5`，减少`-1/-4/0`。机制通过seed=`0/3`，中位Bird减少0、Unknown减少-1。中位DBB→CU从10降7，但CU→DBB从4升5；共同Bird错误仅7降6，而共同Unknown错误2升3。
- Impact：这是用户明确认定最危险的Unknown错误接受方向；即使seed43救回3个Bird，也以新增4个Unknown→DBB及1个Clutter→DBB为代价，不能视为有效Pareto改进。
- Action：不得增加同一辅助loss权重追求更多DBB Recall；下一候选必须显式约束Unknown开放集负证据或拒识边界，同时保留Bird已知证据，且继续使用train-only/OOF阈值纪律。
- 验证状态：已确认parent级语义残差；人工标签继续按既有政策保留，没有修改标签。
- 关联版本：V12。

## Finding F122

- Severity：High（已确认辅助表征存在但未被Gate正确利用）
- 文件位置：`src-v3/report/dbb_auxiliary_diagnostics.py`、`RADAR-LN3-V12-dbb-auxiliary-diagnostic.json`
- 问题或结论：V12失败不能简单解释为“三亚类辅助任务没学会”。辅助头在validation上已形成明显DBB亚类判别，但Gate仍把辅助头明确识别为Bird的样本路由到CU；瓶颈位于共享表示到二类Gate边界的利用方式及开放集边界，而非继续增加相同辅助CE强度。
- Evidence：三seed辅助头validation DBB parent-final Accuracy=`0.913876/0.909091/0.904306`，Macro-F1=`0.913168/0.902351/0.896312`。Treatment分别有7/7/8个Bird→CU，三个辅助头对这些22个seed级错误全部仍预测Bird（22/22）。训练辅助loss显著下降，但Gate机制门槛仍失败。
- Impact：单纯提高0.20权重可能进一步强化DBB吸引而扩大Unknown→DBB，无法保证Gate使用“Bird已知证据”时保留开放集安全性；增加液态层数也不直接解决输出边界语义。
- Action：不搜索V12辅助权重、不将辅助头直接并入线上路由。若新实验获授权，应只改变一个决策耦合/开放集目标变量，并以Bird→CU和Unknown→DBB双向护栏验证。
- 验证状态：CUDA validation前向与checkpoint hash已确认；辅助头未在test或线上评估。
- 关联版本：V12诊断，不创建新数据/配置版本。

## Finding F123

- Severity：High（已确认Unknown不确定性不能由三亚类softmax表达）
- 文件位置：`RADAR-LN3-V12-dbb-auxiliary-diagnostic.json::by_semantic_label`、`gate_error_mechanism`
- 问题或结论：因为Clutter/Unknown严格排除于辅助loss，三亚类辅助softmax对它们只能强制选择Drone/Bird/Balloon之一；高softmax置信度不是knownness或拒识证据。
- Evidence：Treatment的Unknown→DBB按seed为5/7/5，强制辅助投影主要落到Bird；seed42/44这些错误的辅助最大概率中位数分别约`0.9603/0.9874`。Clutter也常被高置信投影为Bird，但主Gate仍可正确判CU，直接证明两头概率语义不同。
- Impact：将“辅助预测Bird且高置信”直接覆盖Gate会把异质Unknown和Clutter吸入DBB，破坏开放集拒识；这与用户定义“Unknown只承担拒识”冲突。
- Action：任何后续拒识必须使用显式开放集分数/目标和train parent-grouped OOF校准；不能复用三亚类closed-set softmax置信度作为Unknown阈值。
- 验证状态：已确认validation描述统计；阈值拟合、OOF、test、线上reject和P95均未执行。
- 关联版本：V12。

## Finding F124

- Severity：High（已确认失败不是checkpoint选优偶然，统计外推受限）
- 文件位置：`RADAR-LN3-V12-dbb-auxiliary-diagnostic.json::all_epoch_feasibility`、六份V12 training summary
- 问题或结论：六run共120个epoch无任何epoch同时满足DBB floor≥0.95与CU floor≥0.90；但结论仅适用于固定CQ08 V3 validation和当前协议，不能外推为所有辅助学习失败。
- Evidence：可行arm-epoch=`0/120`。DBB floor约束下最佳CU floor，Control为`0.730769/0.666667/0.642857`，Treatment为`0.720000/0.652174/0.666667`。Validation仅232 parent，其中CU 23；三个seed复用同一parent集合，CUDA严格确定性关闭。
- Impact：事后切换epoch、指标或单seed无法制造第一阶段通过；同时三seed不是独立数据样本，不能给中位数附加独立置信区间或声称生产泛化。
- Action：保留全轨迹为否定证据；11/11统计谬误扫描结论为High caution。复现仅声明best前向和契约范围内已验证，不声明随机训练位级复现。
- 验证状态：已确认；错误JSON层级的第一次临时PowerShell输出已丢弃，最终结果来自summary实际history结构与版本化诊断报告。
- 关联版本：V12。

## Finding F125

- Severity：High（已完成只读诊断并否定当前候选）
- 文件位置：`RADAR-LN3-V12-dbb-auxiliary-diagnostic.json`、拟实现的V12 paired knownness诊断
- 问题或结论：V12辅助头对Bird有closed-set亚类判别，但Treatment没有让最近DBB亚类距离获得超出Control或`P(CU)`的Unknown-vs-Bird排序信息；当前原型/距离耦合V13候选被否定。
- Evidence：Treatment余弦/对角Mahalanobis的Unknown-vs-Bird中位AUC=`0.823077/0.821795`，低于Treatment `P(CU)=0.876923`，也低于Control同距离`0.888462/0.896154`。CU-vs-DBB中位仅`0.942168/0.941752`，比Treatment `P(CU)=0.968379`下降约0.026。两个候选设计门槛均`0/3` seed通过，eligible为空。
- Impact：把这些距离接入Gate或拒识会增加验证自由度，并可能继续把高距离困难Bird拒到CU；不能用总体Unknown-vs-DBB AUC>0.90掩盖Unknown-vs-Bird和CU-vs-DBB退化。
- Action：停止当前原型/距离耦合方向，不拟合阈值、不实现V13。下一诊断回到Unknown子流型、困难Bird时序/RD覆盖和开放集目标，而不是搜索更多静态距离函数。
- 验证状态：已确认；1392行统一CSV独立复算120个AUC单元，focused test通过。训练、阈值、OOF和test未执行。
- 关联版本：V12诊断，不创建数据/配置版本。

## Finding F126

- Severity：Critical（当前V13设计方向不放行）
- 文件位置：`src-v3/processed/report/RADAR-LN3-V12-dbb-knownness-paired-diagnostic.json::v13_design_evidence_gate`
- 问题或结论：预冻结的V13设计证据门槛失败，不能进入原型/距离耦合实现、配置或训练。
- Evidence：余弦与对角Mahalanobis候选的passing seed均`0/3`，三seed中位Unknown-vs-Bird相对Control分别下降`0.065385/0.074359`，相对Treatment `P(CU)`下降`0.053846/0.055128`；CU-vs-DBB相对Treatment `P(CU)`下降`0.026212/0.026628`，超过允许0.01。仅Unknown-vs-DBB中位≥0.90通过，不足以放行。
- Impact：继续实现会违反单因素实验的事前门槛，并可能通过反复搜索距离/阈值过拟合固定validation；无法证明能降低危险Unknown→DBB且保护Bird。
- Action：`eligible_candidates=[]`作为硬阻塞；不创建V13配置、不训练、不做OOF或test。需要新的证据方向与用户授权。
- 验证状态：正式六checkpoint CUDA诊断、统一CSV及120 AUC独立复算通过；门槛结论已确认。
- 关联版本：V12诊断。

## Finding F127

- Severity：High（已确认困难Bird与误接收Unknown在绝对距离上重叠）
- 文件位置：`RADAR-LN3-V12-dbb-knownness-paired-diagnostic.json::runs.*.*.validation_gate_errors`
- 问题或结论：辅助头将困难Bird相对分类为Bird，并不表示其绝对接近典型train Bird/DBB流型；Bird→CU与Unknown→DBB均处于DBB参考的高距离尾部并明显重叠。
- Evidence：Treatment Bird→CU的余弦/对角距离在train DBB中的每seed中位约`0.980/0.981`、`0.985/0.983`、`0.988/0.985`分位；Unknown→DBB对应约`0.918/0.922`、`0.916/0.923`、`0.853/0.855`，且个体最大可达约0.97–0.997。两组无法由单一静态距离干净切分。
- Impact：简单规则“辅助头判Bird且距离近则DBB、距离远则Unknown”仍会拒绝困难Bird或接收部分Unknown；中心距离不是当前缺失的唯一边界信息。
- Action：保留这些分数仅作诊断，不接入线上。下一阶段应核查这些parent的时间证据、RD覆盖/形态和Unknown子流型，而不是继续静态中心距离搜索。
- 验证状态：已确认validation parent-final；没有window/stable/test结论。
- 关联版本：V12。

## Finding F128

- Severity：Critical（已确认设计实现偏差；V13代码级已修复）
- 文件位置：`src-v3/radar_ln3/cascade.py`、`src-v3/radar_ln3/rejection.py`、`src-v3/radar_ln3/stable.py`、`src-v3/report/cascade.py`
- 问题或结论：历史实现只是三个模型的五类soft posterior组合，不是产品定义的三层动作级联；V13现已将soft posterior与产品动作分离，实现Gate DBB接受/路由CU、DBB条件分类、Clutter接受/Unknown拒识，以及`DBB_PENDING/DBB_ACCEPTED/CU_PENDING/CLUTTER_ACCEPTED/UNKNOWN_REJECTED`状态。
- Evidence：V13 focused测试验证未校准策略不输出动作、最小证据期保持pending、验证集来源校准被拒绝；CPU/CUDA真实CQ08 Smoke均证明校准后offline/online逐事件`final_action/stable_state/accepted_label/rejected`完全一致，首事件pending、连续两次接受后才stable确认。V13在线历史`window_label_*`字段已置空，五类概率只从`soft_posterior_audit_*`读取；CU checkpoint输出名冻结为`CLUTTER_ACCEPT_EVIDENCE/CLUTTER_REJECT_EVIDENCE`。
- Impact：修复后不会再出现`rejected=true`同时确认具体stable类；五类soft posterior只用于审计，Coverage/FAR/FRR/Selective Risk由独立分层动作计算。V13监督性能已在Stage 048评估且未达标，正式OOF拒识仍未评估。
- Action：无calibration时报告保持`not_evaluated`，禁止线上放行。Stage 048四项基础护栏通过`0/4`，暂不进入OOF阈值校准；先完成训练集内错误覆盖与目标归因。
- 验证状态：代码、合成动作/指标、真实CQ08 CPU/CUDA Smoke和正式20 epoch监督训练均已完成；真实OOF calibration、validation动作性能、test和P95未执行。
- 关联版本：V13。

## Finding F143

- Severity：Critical（已确认V13不是严格open-set）
- 文件位置：`src-v3/radar_ln3/losses.py::hierarchical_final_state_nll`、V13三配置task mapping、`src-v3/train/cascade.py`
- 问题或结论：UnknownTarget仍参与V13参数学习与checkpoint选择，不符合“Unknown只承担拒识开放集、不作为普通闭集学习目标”。
- Evidence：V13 Gate mapping把Unknown设为CU；loss中`gate_targets=(labels>=3)`并对Unknown计算Gate CE，`cu_mask=labels>=3`又对Unknown计算CU第二类CE；每epoch selection还使用`unknown_to_dbb_count`。Train Unknown parent=70。
- Impact：V13学到的是“已知Unknown样本分布”，不是对未见开放空间的严格拒识；validation Unknown又参与模型选择，进一步形成development目标泄漏。
- Action：V14已把Unknown从normalization、train Dataset、loss、sampler、OOF、threshold与checkpoint ranking全部排除；Unknown只保留为冻结开放集审计真值和`UNKNOWN_REJECTED`动作分母。
- 验证状态：V13问题已确认；V14代码/数据契约修复、focused及CPU/CUDA Smoke、正式known-only联合训练、5/5 OOF、真实calibration和冻结development拒识/stable评估均已完成。严格open-set语义通过，性能未达标。
- 关联版本：V13问题；V14代码级已修复。

## Finding F144

- Severity：Critical（DBB分支缺少开放集拒识）
- 文件位置：`src-v3/radar_ln3/rejection.py::HierarchicalRejectionPolicy.decide`
- 问题或结论：V13只要Gate通过DBB阈值，就无条件接受DBB specialist argmax；没有DBB knownness阈值、margin或拒识动作。
- Evidence：`gate_accepts_dbb`分支直接构造`final_action=DBB_ACCEPTED`，只记录三分类top-two margin而不校验。V13 validation有6个Unknown→DBB，全部被强制映射为已知亚类。
- Impact：无法实现用户要求的“DBB中不认识的也拒识到Unknown”，Unknown误过Gate即产生危险假接受。
- Action：V14已新增DBB known acceptance threshold/margin；Gate通过但DBB不通过时直接`UNKNOWN_REJECTED`，且stable不携带具体类。
- 验证状态：V13问题已确认；V14合成动作、离线/在线一致性与stable冲突防护、真实OOF阈值及冻结development拒识/stable评估均已完成。DBB knownness阈值=`1e-7`且Unknown FAR=`0.733333`，真实性能未放行。
- 关联版本：V13问题；V14代码级已修复。

## Finding F145

- Severity：High（V2 normalization包含Unknown分布）
- 文件位置：`data/processed/RADAR-LN3-V2-cq08-tr-rd-train-normalization-stats.json`
- 问题或结论：V2统计由全部1081 train parent生成，包含70个Unknown；若V14直接复用，Unknown仍会影响RD输入中心和尺度。
- Evidence：V2记录`parent_count=1081`；V3 train标签计数为488/243/245/35/70。V2 center/scale=`55.875/6.911864353489518`，无法从现有文件证明known-only值相同。
- Impact：即使CE排除Unknown，预处理仍存在Unknown分布暴露，不能声明本项目定义下的严格零学习open-set。
- Action：已新建V14 known-only normalization，只使用1011个known train parent；记录排除70个Unknown、tensor provenance和hash，不覆盖V2。
- 验证状态：已解决。V14统计SHA-256=`1484f58521be677b0a5f992e40fd8af558660da2c8f764669f151d9d969194b0`，selected/excluded RD tensor路径交集为0，幂等复跑保持hash不变。
- 关联版本：V2历史统计；V14已替换正式训练输入统计。

## Finding F146

- Severity：High（严格open-set版本必须独立）
- 文件位置：`config/RADAR-LN3/`、`data/processed/`、V13 task/checkpoint/calibration schema
- 问题或结论：严格open-set同时改变训练population、normalization、任务标签、loss、动作、OOF校准和checkpoint选择，不能作为V13代码小修或加载V13 checkpoint。
- Evidence：当前全局最高版本V13；版本规则要求上述任一实质变化递增。V13三checkpoint已经绑定V13 config/data hash和selection，强加载器禁止语义漂移。
- Impact：复用V13文件名或checkpoint会破坏因果归因和校准绑定，并可能静默把已知Unknown监督带入新方案。
- Action：已使用全局V14，新建known-only stats、数据契约、三配置、task/joint checkpoint及calibration schema；V3 index/split/TR/RD张量只读复用。
- 验证状态：V14代码、配置、数据与OOF契约、Smoke已完成；V13 checkpoint不能加载V14，V13正式checkpoint仍能只读恢复。V14正式三checkpoint、5/5 OOF和绑定calibration均已生成并严格回读。
- 关联版本：V14。

## Finding F147

- Severity：Critical（严格open-set代码与数据边界已修复，真实性能已评估但未达标）
- 文件位置：`src-v3/radar_ln3/task_models.py`、`losses.py`、`rejection.py`、`cascade.py`、三份V14配置和V14数据契约
- 问题或结论：V14现符合本项目冻结的严格known-only open-set定义：Unknown不是模型类、训练target、normalization/optimizer/checkpoint/OOF/threshold population；Gate识别DBB或路由CU，DBB低knownness和CU未接受Clutter均形成`UNKNOWN_REJECTED`。
- Evidence：Known train=1011、Unknown train source=70但optimizer exposure=0；V14 loss收到label id 4立即失败；TaskSpec没有Unknown mapping；第五项只叫`UNRESOLVED_OPEN_SET_EVIDENCE`且不作为Unknown posterior；focused验证Unknown进入Dataset/loss/selection/calibration均fail-closed；CPU/CUDA真实CQ08 Smoke三模型梯度与更新非零、checkpoint恢复差0、离线/在线最大差`5.96e-8/4.47e-8`。
- Impact：消除了V13把有限Unknown样本当闭集学习目标及DBB无二次拒识的设计偏差；冻结development结果同时证明技术正确性没有转化为可用开放集泛化。
- Action：冻结V14为“协议正确、性能失败”基线；不得使用15个development Unknown反调模型、epoch或阈值。下一轮只允许在train/OOF范围改进knownness表征和Clutter覆盖。
- 验证状态：代码、数据、loss、checkpoint、动作、offline/online、Smoke、正式训练、5/5 OOF、真实calibration和冻结development拒识/stable均已执行。Unknown拒识Recall=`0.266667`、FAR=`0.733333`；test和P95未执行。
- 关联版本：V14。

## Finding F148

- Severity：High（OOF泄漏边界与完整长时执行已验证）
- 文件位置：`src-v3/build/strict_open_set_oof*.py`、`radar_ln3/strict_open_set_oof.py`、`train/strict_open_set_oof.py`、`report/strict_open_set_oof.py`及V14 OOF contracts
- 问题或结论：V14拒识阈值必须来自known train parent-grouped OOF；当前已完成5折assignment、fold-train-only normalization、fold-specific数据契约、固定最终epoch预测和正式checkpoint绑定入口，避免held-out normalization/epoch选择、Unknown、validation或test泄漏。
- Evidence：1011个known parent恰好一次held-out；fold 00–02为808/203，fold 03–04为810/201；每折Unknown/validation/test计数0；20 epoch、early stopping=false、held-out checkpoint selection=false、最终epoch后单次预测写入v2 fold contract。Focused为5折实际构造Dataset并验证白名单/hash；混入Unknown的synthetic fitter立即失败。
- Impact：后续阈值可控制known false rejection与coverage并保持来源可追溯；它仍不能用known-only OOF本身证明Unknown检测率。
- Action：已使用脚本恢复fold 02并串行完成fold 03/04；5折完成后生成calibration并绑定精确正式V14三checkpoint。后续保持缺折、重复parent、hash不匹配或Unknown/validation/test暴露一律拒绝。
- 验证状态：assignment、fold stats/contracts、入口、focused测试、5/5折训练/预测、1011-parent聚合和真实阈值拟合均完成并通过增强预检。
- 关联版本：V14 OOF calibration support。

## Finding F149

- Severity：High（报告语义问题已修复）
- 文件位置：`src-v3/radar_ln3/cascade.py::evaluate_strict_open_set_cascade/evaluate_hierarchical_cascade`
- 问题或结论：历史V13字段`dt_reason_counts_including_padding`把DataLoader batch对齐padding混入dt reason，导致相同数据在batch=2/8分别出现416/1334。V13/V14现拆成数据事件reason、collate padding位置和evaluation batch size。
- Evidence：代码输出`dt_reason_counts_data_events`、`collate_padding_positions`、`evaluation_batch_size`；分类指标与mask逻辑未改变。V14 CPU/CUDA Smoke和V13 focused通过。
- Impact：报告消费者不再把batch依赖padding计数误当数据固有dt质量；不会改变模型输入、loss或checkpoint。
- Action：历史报告保持只读；新报告仅使用新字段，跨batch比较数据质量时只比较数据事件reason。
- 验证状态：代码级已修复并通过回归；未重跑V13全validation，因为只读正式checkpoint恢复已通过且本修复不改变前向。
- 关联版本：代码修复，不改变V13/V14数据版本。

## Finding F150

- Severity：Critical（开放集性能已确认失败，部署被明确阻塞）
- 文件位置：`src-v3/smoke/RADAR-LN3-V14-strict-open-set-training-readiness.json`、V14 formal/OOF artifact预期目录
- 问题或结论：V14严格open-set协议和流水线均完成，但冻结development五动作、Unknown拒识与stable性能未达到产品门槛，不能进入test或部署放行。
- Evidence：232-parent五动作Accuracy/Macro-F1=`0.875000/0.733411`；Unknown拒识Recall=`4/15=0.266667`、FAR=`11/15=0.733333`；known FRR=`2/217=0.009217`。Stable仅确认228/232，保守全体Accuracy=`202/232=0.870690`、Unknown Recall=`3/15=0.200000`。
- Impact：模型以高coverage接受大多数Unknown，最危险的Unknown→known错误没有被严格open-set契约本身解决；任何部署会产生不可接受的假接受风险。
- Action：冻结V14不可部署；先实现校准放行状态，再在train/OOF范围进行显式可拒识DBB表征、Clutter数据覆盖和Drone/Bird边界单因素实验。
- 验证状态：正式训练、5/5 OOF、calibration、冻结development拒识和stable均已执行；test、P95和部署未执行。
- 关联版本：V14。

## Finding F151

- Severity：High（长时执行连续性风险；已修复为可恢复脚本）
- 文件位置：`artifacts/RADAR-LN3-V14-cq08-liquid-strict-open-set-parent-grouped-oof/`、`src-v3/scripts/v14_strict_open_set_oof_pipeline.ps1`
- 问题或结论：V14五折OOF是串行长时任务，fold 02曾在epoch 6期间被用户终止；若继续依赖多条手工命令，容易遗漏`--resume-checkpoint`、跳过折后契约核验，或在OOF未完整时误启动校准。
- Evidence：初始状态为fold 00/01完成、fold 02停在完整epoch 5、fold 03/04未启动；脚本后续成功恢复并完成全部五折、calibration与冻结development validation，证明断点方案有效。
- Impact：从头重跑fold 02会改变既有计算轨迹并可能触发防覆盖；漏折、重复parent或手工提前校准会破坏1011-parent OOF覆盖及阈值来源可追溯性。
- Action：已新增单入口脚本，顺序恢复fold 02、运行fold 03/04、验证5折1011个唯一known parent、拟合并绑定calibration、最后执行一次冻结development validation。已有完整折只读校验后跳过；非空不可恢复目录及任一不一致均fail-closed。
- 验证状态：用户默认运行已完成剩余长时OOF、calibration和冻结development validation；最终报告与sample detail已回读。
- 关联版本：V14 OOF执行阶段，不改变数据/config/checkpoint版本。

## Finding F152

- Severity：Medium（Windows脚本兼容性；已修复）
- 文件位置：`src-v3/scripts/v14_strict_open_set_oof_pipeline.ps1`
- 问题或结论：Windows PowerShell 5.1不会可靠地把无BOM UTF-8中文脚本识别为UTF-8，初次兼容测试出现中文乱码并继发ParserError。
- Evidence：同一脚本在当前PowerShell AST解析及预检通过，但`powershell.exe -File ... -PreflightOnly`最初在中文字符串处报`Try statement is missing its Catch or Finally block`等伪语法错误；文件加入UTF-8 BOM后，5.1同命令退出码0并正确显示中文。
- Impact：若不修复，用户在项目现有Windows PowerShell终端无法启动流水线，且错误表象容易被误判为脚本逻辑损坏。
- Action：脚本固定使用UTF-8 BOM并声明`#Requires -Version 5.1`；继续保留AST与5.1实际预检作为交付检查。
- 验证状态：已修复并在Windows PowerShell 5.1真实执行增强预检通过；默认完整模式最终完成五折、calibration和冻结validation。
- 关联版本：运行脚本，不改变RADAR-LN3版本。

## Finding F153

- Severity：High（已确认脚本后置断言Bug；已修复）
- 文件位置：`src-v3/scripts/v14_strict_open_set_oof_pipeline.ps1::Assert-Calibration`原第237/241行
- 问题或结论：校准范围错误消息把变量与紧随其后的中文直接连接，PowerShell将`$fieldName不在`解析成变量名；在StrictMode下即使数值条件本身为真，构造`-Message`实参时仍访问未设置变量并停止流水线。
- Evidence：用户实际运行在fold 04通过、calibration写盘后报“检索不到变量`$fieldName不在`”；代码两处均为`"calibration字段$fieldName不在..."`。五折prediction全部passed，calibration SHA-256=`af53913420ab9293885467812d3a1f0990a74eb8a7e7826edd1d5f44e40cf4e1`，校准后validation报告不存在。
- Impact：没有污染模型、OOF或阈值，但阻止了后置校验和最后一次冻结development validation；若误判为训练失败会浪费重跑五折时间。
- Action：两处统一改为`${fieldName}`明确变量边界；增强`-PreflightOnly`以在不启动任务时复核已有五折聚合、calibration和report，从而覆盖后置断言路径。
- 验证状态：Windows PowerShell 5.1增强预检通过；用户随后重新执行默认脚本并生成冻结development validation报告，错误未复现。
- 关联版本：V14运行脚本修复，不改变数据/config/checkpoint/calibration内容。

## Finding F154

- Severity：High（真实OOF、calibration与冻结审计均完成）
- 文件位置：五份`held-out-predictions.json`、`artifacts/RADAR-LN3-V14-cq08-liquid-strict-open-set-cascade-formal/strict-open-set-rejection-calibration.json`
- 问题或结论：V14五折known-only OOF与三道接受边界现已真实完成并可追溯，但这些结果只控制known接受/误拒识，不能单独证明Unknown拒识能力。
- Evidence：五折held-out=`203/203/203/201/201`，合计1011且每个parent唯一；Unknown OOF/threshold fit=`0`，validation/test拟合均为false。Calibration Gate阈值=`0.542160868644714`、DBB knownness阈值=`1e-7`、Clutter阈值=`0.566431522369385`，绑定正式epoch 12三checkpoint。
- Impact：DBB knownness概率阈值贴近数值下界，实际几乎不提供DBB开放集过滤；冻结development中进入DBB路由的8个Unknown全部被接受为Bird/Balloon。
- Action：停止在现有softmax分数上继续搜索validation阈值；下一实验须在known train/OOF中显式训练紧致可拒识表征，并保持Unknown零学习边界。
- 验证状态：OOF、calibration和冻结development评估均已执行并严格回读；test、P95和部署未执行。
- 关联版本：V14。

## Finding F155

- Severity：Critical（已确认开放集性能失败）
- 文件位置：`src-v3/processed/report/RADAR-LN3-V14-strict-open-set-calibrated-validation/evaluation-val.json`、`sample_detail_information.json`
- 问题或结论：V14虽然严格排除了Unknown学习暴露，但三道known acceptance边界无法识别大多数冻结development Unknown；Unknown高置信嵌入DBB/Bird是当前首要安全瓶颈。
- Evidence：Unknown仅4/15形成`UNKNOWN_REJECTED`，7个被接受为Bird、3个为Clutter、1个为Balloon；FAR=`0.733333`。15个Unknown的DBB top probability中位=`0.978122`、最大=`0.990667`，不是简单低置信边界样本。
- Impact：Unknown错误接受率远高于可放行水平，并把Bird Precision压到`0.714286`；提高普通softmax阈值会同时伤害known recall，不能直接解决开放空间高置信错误。
- Action：下一单因素实验采用显式训练的归一化紧致DBB embedding/角度间隔knownness；阈值只由known train parent-grouped OOF控制known误拒识，development Unknown继续封存，不得作为训练负类。
- 验证状态：冻结development parent-final和stable均确认；test、跨source泛化和多seed未执行。
- 关联版本：V14。

## Finding F156

- Severity：Critical（已确认校准放行语义缺失；Stage 053已修复）
- 文件位置：`src-v3/radar_ln3/strict_open_set_oof.py`第602/669行、`src-v3/report/strict_open_set_oof.py`第228/255行、`src-v3/scripts/v14_strict_open_set_oof_pipeline.ps1::Assert-Calibration`
- 问题或结论：校准产物记录三项`all_precision_recall_targets_passed=false`，但报告和流水线仍以结构性`status=passed`结束，且没有`deployment_eligible=false`或`audit_only`字段；`Assert-Calibration`只校验来源、hash和阈值数值范围，不校验性能门槛。
- Evidence：Gate最弱P/R=`0.487805`目标0.95，DBB knownness=`0.907336`目标0.95，Clutter=`0.542857`目标0.90，三项均失败；脚本第215–245行仍输出“calibration校验通过”，最终报告`status=passed`。
- Impact：自动化消费者可能把“产物结构与来源有效”误读为“拒识能力可部署”，造成未达标阈值被线上加载。
- Action：已新增v2质量字段和v1机械推导兼容；loader/报告默认deployment并fail-closed，显式audit只限冻结development评估，online接口拒绝任何audit-only策略。
- 验证状态：已修复。V14/V13 focused、真实v1 loader双用途探针、真实OOF只读汇总、默认report CLI负向测试、PowerShell 5.1后置预检、compileall与静态隔离均通过；当前calibration默认部署加载实际失败。CPU有界Smoke因正式artifact保护未执行且未绕过。
- 关联版本：V14报告/部署语义，数据版本不变。

## Finding F157

- Severity：High（已确认OOF协议与正式模型训练阶段不一致；影响待单因素验证）
- 文件位置：五份V14 OOF `held-out-predictions.json`、正式`training-summary.json`、calibration provenance
- 问题或结论：每折OOF固定使用epoch 20最终模型生成分数，而正式部署三checkpoint来自best epoch 12；calibration虽绑定正式checkpoint hash，却没有证明epoch 20折分数与epoch 12正式分数具有相同尺度。
- Evidence：五折prediction均记录`epoch=20`；正式training summary记录`best_epoch=12`，calibration deployment identity记录epoch 12/global step 6072。该差异不构成Unknown、validation或test泄漏。
- Impact：概率和margin阈值可能发生训练阶段尺度迁移，降低阈值在正式模型上的可迁移性；它不能解释全部Unknown高置信错误，但会削弱校准可信度。
- Action：下一OOF采用与正式模型一致的checkpoint选择协议，优先使用fold-train内部嵌套选择；held-out折只在选优完成后前向一次。
- 验证状态：协议差异已确认；同初始化epoch 12/20阈值尺度paired验证未执行。
- 关联版本：V14 OOF协议。

## Finding F158

- Severity：High（已确认Clutter覆盖与折间异质性瓶颈）
- 文件位置：V14五份OOF held-out prediction、calibration combined metrics、冻结development报告
- 问题或结论：Clutter support不足且不同折表现大幅波动，当前证据不支持通过继续增大class weight或液态容量达到CU门槛。
- Evidence：known train Clutter仅35、development仅8；OOF Clutter P/R/F1=`0.629630/0.485714/0.548387`。五折总体Accuracy从`0.852217`到`0.945813`，最弱P/R从`0.285714`到`0.796610`；冻结development Clutter P/R/F1=`0.500000/0.625000/0.555556`。
- Impact：少数parent和子流型分配即可显著改变门槛，单seed或单个固定validation点估计不具备稳定放行证据；CU边界继续限制总体Macro-F1。
- Action：先在合法CQ08/raw范围审计并分阶段补足Clutter子流型与parent覆盖，采样保持parent级；若没有额外合法样本，将数据不足列为正式阻塞。
- 验证状态：OOF与冻结development均确认；标签语义子流型全量人工审计和独立test未执行。
- 关联版本：V14数据覆盖，不修改标签。

## Finding F159

- Severity：Critical（已确认DBB Knownness没有受训练约束的开放集几何；待V15单因素修复验证）
- 文件位置：`src-v3/radar_ln3/cascade.py::evaluate_strict_open_set_cascade`、`src-v3/radar_ln3/strict_open_set_oof.py::predict_oof_parent_final`、`src-v3/radar_ln3/rejection.py::StrictOpenSetRejectionPolicy`
- 问题或结论：V14把DBB三分类softmax的最大概率与top-two概率差直接作为knownness；训练目标只有闭集DBB交叉熵，没有归一化embedding、类原型、类内紧致或已知边界约束。
- Evidence：代码对`dbb_sequence_logits`直接softmax并记录`dbb_known_confidence/dbb_known_margin`；真实OOF只能选出probability threshold=`1e-7`、margin=`0`，等价于近似全部放行；冻结development进入DBB路由的8个Unknown全部被接收，Unknown FAR=`0.733333`。
- Impact：闭集softmax可对流型外样本高置信，阈值不能从现有分数中形成有效拒识边界；继续在同一隐藏状态上事后搜索Energy/Mahalanobis或development阈值会增加选择偏差，却没有改变表征几何。
- Action：V15采用Control/Treatment配对，只在Treatment将DBB读出替换为归一化角度原型并以known DBB标签施加固定margin；推理使用无margin cosine分数。Unknown保持零学习暴露，收益只由known parent-grouped OOF与不可退化护栏决定。
- 验证状态：V14机制和失败结果已确认；V15代码、focused、正式训练、嵌套OOF与冻结development审计未执行。
- 关联版本：V14问题；V15候选修复。

## Finding F160

- Severity：High（已确认配对初始化会被可变头结构破坏；待V15代码修复）
- 文件位置：`src-v3/radar_ln3/cascade.py::build_cascade`、`src-v3/radar_ln3/task_models.py::LiquidTaskClassifier.__init__`
- 问题或结论：当前级联在一次全局seed后按Gate→DBB→CU顺序构造三个模型；若Treatment改变DBB头的参数构造或随机数消耗，后续CU初始化会变化，因此“同seed”不等于非干预模块逐位配对。
- Evidence：`build_cascade()`字典推导依序实例化三个`LiquidTaskClassifier`，每个encoder/head均使用PyTorch全局RNG；当前没有task-specific initialization seed、base-state hash或Control/Treatment非干预参数相等断言。
- Impact：Treatment与Control的Gate/CU或DBB encoder可能从不同初值出发，任何性能差异都混入初始化噪声，无法归因给角度knownness头。
- Action：V15引入固定逐task派生初始化并输出初始state hash；配对测试必须证明Gate、CU和DBB除干预头外参数逐位一致，DataLoader parent顺序hash一致。
- 验证状态：代码机制已确认；V15修复与paired focused测试未执行。
- 关联版本：V15实验契约。

## Finding F161

- Severity：Critical（已确认新版本可能静默退出严格open-set路径；实施前必须修复）
- 文件位置：`src-v3/radar_ln3/task_models.py`、`cascade.py`、`train/cascade.py`、`radar_ln3/strict_open_set_oof.py`中对`STRICT_OPEN_SET_SCHEMA_VERSION`的等值判断
- 问题或结论：V14严格open-set目前由多个`schema_version == RADAR-LN3-V14`分支激活；直接创建`RADAR-LN3-V15`配置会选回历史TaskSpec、非known-only Dataset或V13 loss/evaluator路径。
- Evidence：`task_spec()`、`build_cascade_dataset()`、`LiquidSoftCascade.forward()`、`cascade_training_objective()`和OOF predictor均存在V14单值判断；`HIERARCHICAL_SCHEMA_VERSIONS`也只列V13/V14。
- Impact：V15可能表面使用新配置，实际把Unknown重新当普通类或绕过strict-open-set拒识/评估边界，属于静默语义回退。
- Action：建立显式strict-open-set schema family并让V14/V15共享Unknown零暴露和动作语义；V15-specific readout另用独立字段区分。增加Unknown暴露和V14/V15 checkpoint混用负向测试。
- 验证状态：风险已由静态调用链确认；代码尚未修复，V15当前不能训练。
- 关联版本：V15实施阻塞。

## Finding F162

- Severity：High（已确认并已修复）
- 文件位置：`src-v3/build/angular_knownness_oof.py::_write_new_or_identical`、`data/processed/RADAR-LN3-V15-cq08-known-only-nested-oof-contract.json`
- 问题或结论：首次V15 nested OOF合同在每个stage记录了`write_status`，该字段依赖构建时文件是否刚创建；后续相同内容重建时stage contract转为`reused_identical`，使顶层contract误判为不同并阻断幂等preflight。
- Evidence：实际执行`python -m build.angular_knownness_oof`在无数据、split、配置或hash变化时抛出`FileExistsError`；逐字段比较显示唯一差异是嵌套stage的运行时`write_status`，既有顶层contract SHA-256仍为`bbed0a99a8623660b30b5d823e82eb5eed6165a9d84b1e2eb97ac8ca43b20326`。
- Impact：会阻塞V15人工训练前重复预检，容易诱导不安全的手动删除或重写合同；不影响已冻结样本集合本身，但影响可重复执行证据。
- Action：新增只读兼容比较，忽略既有contract中仅描述构建结果的`write_status`；新payload不再把它写入语义contract。既有contract不改写，hash保持不变。
- 验证状态：已执行`build.angular_knownness_oof`，返回`reused_legacy_runtime_status_only`；5折parent、source hash、stage contract/hash仍通过。
- 关联版本：V15 nested OOF执行契约。

## Finding F163

- Severity：Critical（已确认并已修复）
- 文件位置：`src-v3/train/cascade.py::train_hierarchical_cascade`
- 问题或结论：严格open-set训练入口将V14的训练结束development Unknown审计泛化到所有strict-open-set schema；若V15使用该入口，formal训练尚未冻结OOF/calibration前就会读取15个development Unknown，从而违反V15“development Unknown仅在两臂与阈值全部冻结后一次审计”的设计。
- Evidence：V14 branch在best checkpoint恢复后无条件构建`development_open_set_audit` Dataset；V15同属strict-open-set family，旧条件会进入该分支。该行为虽不进入optimizer或checkpoint selection，但仍会把冻结development样本提前暴露给实验过程。
- Impact：人为知晓的development Unknown结果可能影响后续margin、scale、epoch、阈值或路线选择，破坏V15 paired单因素的选择隔离。
- Action：V15 formal训练只构建known-only train/checkpoint-selection Dataset，明确写入`development_unknown_access_count_during_formal_training=0`；development Unknown留给OOF/calibration后独立一次报告入口。
- 验证状态：代码路径和focused preflight已确认；V15正式训练未执行，因此没有development Unknown实际访问。
- 关联版本：V15 strict-open-set formal training。

## Finding F164

- Severity：Critical（已确认并已修复）
- 文件位置：`src-v3/train/angular_knownness_oof.py`、`src-v3/radar_ln3/angular_knownness_oof.py`
- 问题或结论：最初outer held-out在最终预测写盘前没有持久化“已经开始唯一一次前向”的状态；若CUDA进程在prediction期间终止，重启无法证明held-out是否已部分被观察，继续运行会违反“only one held-out forward”的嵌套OOF协议。
- Evidence：原流程只在prediction JSON成功后写`outer_held_out_prediction_pass_count=1`；中间异常没有可审计状态。该问题可由静态流程复现，无需读取held-out数据。
- Impact：outer held-out可能被重复前向，造成epoch/阈值协议的不可审计重复观察；虽不直接泄漏标签到训练，但损害OOF独立性和结果可信度。
- Action：在outer final forward前写入contract/config/parent-hash绑定claim；存在claim但无completed prediction时fail-closed，禁止自动重试；completed claim必须绑定prediction SHA-256并由聚合器复核。
- 验证状态：focused临时路径验证重复claim被拒绝、completed claim与prediction hash可回读；真实OOF未执行。
- 关联版本：V15 nested OOF。

## Finding F165

- Severity：High（已确认V15实施准备完成，性能仍待验证）
- 文件位置：`src-v3/radar_ln3/task_models.py::AngularPrototypeReadout`、`losses.py::strict_open_set_final_state_nll`、`rejection.py`、`src-v3/smoke/angular_knownness_contract_test.py`
- 问题或结论：V15已实现一个可归因的DBB knownness单因素候选：Treatment以L2归一化48维embedding与3个可学习原型产生无标签raw cosine score，训练期仅对真实DBB target施加固定margin=`0.20`/scale=`16`；Unknown不作为class、prototype、loss、normalization、OOF或threshold样本。
- Evidence：focused实测margin使loss从`2.95758056640625`变为`6.071575164794922`；同一model forward前后inference score/evidence差异=0；Treatment offline/online knownness最大差`8.940696716308594e-08`。Control/Treatment Gate、CU和DBB同名同shape共享参数逐位一致；概率/余弦calibration或paired arm混用被实际拒绝。
- Impact：本轮修复了V14普通softmax knownness缺少训练几何约束的可验证候选机制，但不会自动证明能改善Unknown FAR、Bird、Clutter或总五类指标。
- Action：仅在用户手动完成两臂formal→5折nested OOF→各臂calibration后，依据known-only OOF DBB最弱P/R、known FRR和不可退化护栏判定；development Unknown只能在所有该等选择冻结后一次审计。
- 验证状态：代码/配置/focused/静态隔离/PowerShell preflight通过；正式训练、OOF、calibration、development audit、test、P95和部署未执行。
- 关联版本：V15。

## Finding F166

- Severity：High（已确认并已修复）
- 文件位置：`src-v3/scripts/v15_angular_knownness_pipeline.ps1::Invoke-Preflight`、`src-v3/build/angular_knownness.py::build`
- 问题或结论：V15流水线把可写的实验准备器无条件用于`formal/oof/calibrate/all`全部阶段；正式训练完成后，准备器按防覆盖规则拒绝非空formal artifact，导致合法的OOF阶段在开始前被错误阻断。
- Evidence：用户执行`-Stage oof -Arm both`时，`build.angular_knownness`抛出`FileExistsError: refusing to prepare over a non-empty V15 artifact`，目标为已完成的Control formal artifact。正式训练摘要同时确认两臂均为`status=passed`、20 epoch、Unknown训练及development Unknown访问均为0。
- Impact：流程错误会阻断nested OOF与已冻结known-only校准，而不应要求删除或覆盖正式checkpoint；若通过手工删除规避，会破坏正式模型与OOF校准的部署绑定。
- Action：新增只读`--verify-only`模式；`oof/calibrate`仅核验冻结contract/config、正式三checkpoint/joint checkpoint及零暴露摘要，`formal/all`仍保留可写准备与防覆盖保护。摘要读取显式固定UTF-8，以避免Windows PowerShell代码页破坏中文JSON。
- 验证状态：`-Stage oof -Arm both -PreflightOnly`退出码0；未启动fold训练、held-out预测、calibration、development Unknown审计、test、P95或部署。
- 关联版本：V15。

## Finding F167

- Severity：Medium（执行中断；协议恢复点已确认安全）
- 文件位置：`src-v3/scripts/v15_angular_knownness_pipeline.ps1::Invoke-NestedOOF/Get-StageResume`、`src-v3/train/angular_knownness_oof.py::_run_stage`、Control `outer-02/outer-retrain` artifact。
- 问题或结论：V15 Control nested OOF在outer fold 02的outer retrain第12/12 epoch中途以子进程exit code `-1`停止。它不是已记录的Python/CUDA训练异常；由于中断发生在epoch checkpoint之后、held-out forward之前，协议允许从上一个完整epoch恢复，而不需要重跑已完成fold或重复观察held-out parent。
- Evidence：Control outer fold 00/01各自具有completed prediction与forward claim；fold 02具有完成的inner selection summary和`joint_last_model.pt`（SHA-256=`f410fc161321f32ec15f7b267d27e0c35202804de5147f2f87cce6bb8c6b6b5b`），其run state为`next_epoch=12/global_step=4444/history=1..11`。训练日志止于epoch 12 step 0200/0404，没有traceback/OOM；fold 02的forward claim和prediction均不存在。
- Impact：若删除artifact或从头重训fold 02，会破坏既有训练可追溯性且无必要；若存在claim后仍自动重跑held-out forward则会违反唯一观察协议。当前状态不会污染Unknown、validation、test或校准，因为这些路径均未访问。
- Action：使用`-Stage oof -Arm control -OuterFold 2`恢复；脚本将只加载fold 02 outer retrain的`joint_last_model.pt`，重做epoch 12并在成功后生成唯一held-out prediction。随后独立完成剩余fold，5/5齐全后再运行calibration。不得运行`-Stage all`或删除任何现有artifact。
- 验证状态：脚本恢复分支和checkpoint payload已只读核验；实际恢复、剩余fold、5折聚合、校准、development Unknown审计、test、P95和部署均未执行。
- 关联版本：V15 nested OOF。

## Finding F168

- Severity：Critical（预注册V15主终点失败）
- 文件位置：两臂V15 `strict-open-set-rejection-calibration.json`、10份outer-held-out prediction
- 问题或结论：归一化角度原型Treatment没有改善known-only OOF的DBB knownness最弱Precision/Recall，且通过降低coverage换取部分accepted accuracy；按预注册规则不能保留为当前DBB knownness改进方案。
- Evidence：Control/Treatment各覆盖1011个known train parent且每个parent恰好一次。Control DBB Macro-F1=`0.955973`、最弱P/R=`0.930041`、coverage=`1.0`；Treatment为`0.950422/0.915323/0.983607`。known FRR从`0.016815`升至`0.032641`。Treatment拒绝16个DBB parent，其中10个raw错误、6个raw正确。Treatment低cosine-gap的raw错误ROC-AUC=`0.918093`，与Control低softmax-gap的`0.919322`基本相同；Treatment AP较高`0.418002>0.331366`，主要体现极低gap区域错误更浓。两臂DBB `all_precision_recall_targets_passed=false`。
- Impact：角度gap有一定错误富集能力，但当前边界仍牺牲正确Drone/Bird/Balloon，Drone Recall从`0.977459`降为`0.948770`；不能宣称达到DBB≥0.95，也不能据此进入Unknown正式拒识、test或部署。
- Action：冻结V15为失败但有诊断价值的候选；不得继续使用development Unknown搜索margin、scale或阈值。若重测角度knownness，应隔离DBB specialist优化或冻结Gate/CU，一次只改变一个主要因素。
- 验证状态：正式两臂训练、两臂5折nested OOF、known-only校准和parent配对诊断已完成；development Unknown、test、stable、P95和部署未执行。
- 关联版本：V15。

## Finding F169

- Severity：High（单折高分被误读风险；已完成语义裁决）
- 文件位置：Control `outer-04/inner-selection/stage-summary.json::history[epoch=20]`
- 问题或结论：用户观察到的DBB `Accuracy=0.969388/Macro-F1=0.963904/support=196`是单个inner-selection折、单个epoch的条件三分类快照，不是V15 1011-parent OOF聚合结果，也不是Treatment角度头收益。
- Evidence：该数值只定位到Control outer-04 epoch 20；Bird与Balloon Recall均为`0.938776`，未满足逐类P/R≥0.95。outer-04冻结epoch 14，因为epoch 20的Clutter acceptance Recall=`0.285714`，低于epoch 14的`0.714286`，known parent Macro-F1也更低`0.788536<0.798485`。
- Impact：若事后选择该快照，会产生look-elsewhere和局部头替代产品目标的问题，并以Clutter显著退化换取DBB局部提升；无法用于正式checkpoint、阈值或五类结论。
- Action：继续以外层OOF和事前产品护栏为主；单折快照只作为“DBB能力上限可能存在、跨折/跨目标不稳定”的诊断证据。
- 验证状态：文件路径、arm、fold、epoch、support、逐类矩阵和实际selected epoch均已核对。
- 关联版本：V15 Control nested OOF。

## Finding F170

- Severity：High（配对实验存在已确认的训练动力学间接干预）
- 文件位置：`src-v3/train/cascade.py::train_hierarchical_cascade/cascade_training_objective`、V15两臂训练日志与相同epoch checkpoint
- 问题或结论：V15表面只替换DBB读出头，但三模型由一个optimizer联合更新，并对全模型参数统一做梯度裁剪；角度Treatment更大的DBB损失/梯度会改变Gate和CU的梯度缩放，因此非干预模块在相同epoch后不再保持配对。
- Evidence：代码每step执行`objective["loss"].backward()`后对`model.parameters()`调用`clip_grad_norm_(...,100)`。正式日志抽样220个进度点中Control/Treatment `grad>100`分别为`4/220`和`39/220`，最大值`197.469/600.513`。同epoch checkpoint比较显示Gate/CU各40个state tensor全部非零差异；两臂5个outer fold只有fold 04选择相同epoch。
- Impact：Treatment的Gate CU Recall、Clutter F1和known-action Macro-F1改善不能直接归因于角度几何；它混入了全局裁剪和不同epoch选择产生的训练路径变化。当前实验仍可报告“系统级Treatment结果”，但不是纯DBB头因果效应。
- Action：下一次若验证DBB角度knownness，应采用冻结Gate/CU或分任务optimizer/分任务梯度裁剪，并保持同epoch护栏；不要同时改变数据覆盖。现有V15结果按`CAUTION`解释。
- 验证状态：代码、配置、日志与checkpoint均已只读核验；隔离optimizer的paired实验未执行。
- 关联版本：V15实验协议。

## Finding F171

- Severity：Critical（Gate/Clutter仍是known-only放行瓶颈）
- 文件位置：两臂V15 calibration `selection_diagnostics.gate/clutter_acceptance/combined_known_action`
- 问题或结论：无论Control还是Treatment，35个Clutter parent上的Gate路由和Clutter接受均远低于目标；角度DBB头没有解除CU数据覆盖与边界瓶颈。
- Evidence：Control Gate CU P/R=`0.435897/0.485714`，Clutter accept P/R=`0.520000/0.371429`，最终Clutter P/R/F1=`0.500000/0.314286/0.385965`。Treatment对应为`0.536585/0.628571`、`0.531250/0.485714`和`0.600000/0.428571/0.500000`；均显著低于0.90，support仅35。两份校准三组件均失败并`deployment_eligible=false`。
- Impact：总体Accuracy较高会被976个DBB known parent的base rate掩盖；Clutter/CU低support和低Recall继续阻塞严格open-set五动作、Unknown拒识安全与部署。
- Action：优先执行独立的Clutter parent/子流型覆盖和证据归因审计；不得把SMOTE window复制、增加液态容量或角度头作为数据覆盖替代。若合法CQ08/raw没有新增parent，明确记录数据不足。
- 验证状态：known-only OOF已确认；development Unknown、标签子流型全量人工审计、test和跨source验证未执行。
- 关联版本：V15，数据仍复用V3/V14 known-only范围。


## Finding F129

- Severity：Critical（已确认训练/选优目标不一致；V13代码级已修复）
- 文件位置：`src-v3/radar_ln3/losses.py::hierarchical_final_state_nll`、`src-v3/train/cascade.py`、`src-v3/train/tasks.py`
- 问题或结论：历史主loss由完整parent标签监督全部prefix，三个task又独立选best；V13改为同一全parent batch上的parent-final hierarchical NLL，prefix仅以0.10权重且从第3事件到最终事件之前作为辅助，三个独立编码器由一个optimizer step联合更新，三checkpoint只在同一产品候选胜出时原子更新。
- Evidence：focused梯度证明主目标早期prefix梯度为0、Unknown→DBB specialist梯度为0、DBB→CU specialist梯度为0；CPU/CUDA真实Bird+Unknown batch中Gate/DBB/CU三模型梯度与参数变化均非零。旧`train.tasks`明确拒绝V13，唯一正式入口为`train.cascade`；task/joint checkpoint恢复输出最大差异均0。Best ranking只使用产品guardrail、方向错误和diagnostic hierarchy，soft posterior明确不参与selection。
- Impact：训练loss、checkpoint和parent-final三层产品目标已对齐，避免局部best拼装；Stage 048证明当前0.10 prefix权重/最小3事件/联合目标组合未达到产品护栏，但因V13同时改变多个契约，不能把失败单独归因给prefix权重或液态模块。
- Action：不得分别运行三条`train.tasks`、事后改选局部epoch或仅根据loss继续加epoch。下一实验必须在train/parent-grouped OOF范围冻结一个主要变量并保持产品选优协议不变。
- 验证状态：代码、梯度、单步更新、checkpoint、入口保护及正式20 epoch训练已完成；Stage 048四项产品护栏为`0/4`。
- 关联版本：V13。

## Finding F130

- Severity：High（已确认时间语义丢失；V13代码级已修复）
- 文件位置：`src-v3/radar_ln3/data.py`、`src-v3/radar_ln3/model.py::CfCLiquidCausalEncoder`、V13数据契约
- 问题或结论：V3 index已有三种真实原因，但历史Dataset/模型丢弃原因。V13现将`first_event_no_predecessor=1`、`zero_delta_at_second_precision=2`、`positive_delta_at_second_precision=3`作为dynamics-only枚举传递，padding=0单独处理；首点初始化、同秒有界观测、正dt连续时间更新互相分离。
- Evidence：全索引真实计数为1546/9776/18341；1546个parent事件索引均从0连续且首点reason恰好一次；train 1081 parent Dataset强校验`reason↔valid↔raw dt↔padding↔event boundary`一致。focused测试中reason-aware batch/online最大差异0，首点与同秒状态最大差异`0.011045694351196289`，非法reason/valid、重复首点、fresh online非首reason和continued online再次首reason均被拒绝；最终CPU/CUDA联合Smoke在线/离线概率最大差异分别`4.470348358154297e-08`和`8.940696716308594e-08`。
- Impact：约38.2%无正dt事件不再共享一个无法区分的语义；没有伪造高精度时间，也未把reason拼入分类Fusion。V13正式结果未达标，但没有同初始化、只改dt reason的paired control，因此仍不能单独判断reason-aware dynamics的收益或损害。
- Action：冻结V13 reason-aware dynamics；若后续发现原始高精度时间，必须另建新数据版本而不是修改V13 index。
- 验证状态：数据传递、shape、输入一致性、离线/在线、CUDA Smoke和V13正式训练均已执行；dt单因素分类收益未评估。
- 关联版本：V13复用V3 index/V2 normalization并新增V13数据语义契约。

## Finding F131

- Severity：High（已确认版本/产物不可拼接；V13实施准备已修复）
- 文件位置：三份`config/RADAR-LN3/RADAR-LN3-V13-*`、`src-v3/radar_ln3/cascade.py::validate_cascade_configs/load_cascade`、`src-v3/train/cascade.py`
- 问题或结论：V12只有Gate配置，不能与V8 specialists混拼；V13已建立同版本Gate/DBB/CU三配置、共同V13数据/dynamics/训练契约、独立task checkpoint schema和原子joint checkpoint schema。
- Evidence：三配置共享index/contract/stats hash、seed、20 epoch、batch、loss、门槛和joint artifact；强校验train parent=1081、类别488/243/245/35/70。正式训练生成的三task best与joint best均为epoch 13/global step 7033，并共享`checkpoint_set_id=611c0817ef0f42afb9ccf9c8658e9602`；独立`report.cascade`严格恢复并复算通过。
- Impact：已得到可归因到同一次联合运行的三checkpoint产品基线，不会静默覆盖V8/V12或拼装不同局部best；该基线性能未达门槛，不能替代历史模型进入部署。
- Action：保留正式artifact只读；禁止手动拼task checkpoint、编辑配置后复用阈值或根据局部task结果另选epoch。
- 验证状态：配置、数据hash、模型构造、joint forward/loss/gradient/checkpoint、CPU/CUDA Smoke、正式20 epoch和best checkpoint独立恢复均通过。
- 关联版本：V13。

## Finding F132

- Severity：Critical（剩余放行阻塞，非本轮可提前完成）
- 文件位置：`src-v3/radar_ln3/rejection.py::load_hierarchical_rejection_calibration`、`src-v3/report/cascade.py`、`artifacts/RADAR-LN3-V13-cq08-liquid-hierarchical-cascade-formal/`
- 问题或结论：V13已具备安全的分层拒识和评估接口，但真实train parent-grouped OOF calibration尚未生成；因此Gate DBB接受阈值、CU Clutter接受阈值、质量门槛、Coverage/FAR/FRR/Selective Risk和线上stable性能仍未验证。
- Evidence：loader强制`fit_split=train/grouping=parent/prediction_protocol=out_of_fold`且validation/test不得参与阈值拟合；校准必须同时绑定data hashes、三配置语义hash、三checkpoint文件hash、共同epoch/global step和product selection hash。Focused合成校准仅验证接口和分母，不同部署checkpoint集合被拒绝；真实CQ08 Smoke中的阈值明确标记synthetic，正式报告仍写`OOF/validated rejection=未执行`。
- Impact：V13监督训练已完成但基础产品护栏`0/4`；当前不能进入拒识部署、test或生产放行。Unknown仍受train仅70 parent、val仅15 parent及内部异质性约束。
- Action：暂缓train parent-grouped OOF calibration；先用train/OOF内部诊断确认困难Bird与Unknown子流型、错误覆盖和目标权重机制。任何未来阈值仍不得使用validation/test拟合。
- 验证状态：接口、泄漏防护、正式监督训练和冻结validation复算已验证；真实OOF、阈值、拒识指标、test和P95未执行。
- 关联版本：V13后续阶段。

## Finding F133

- Severity：Critical（本轮额外确认并已修复）
- 文件位置：`src-v3/radar_ln3/rejection.py::load_hierarchical_rejection_calibration`、`src-v3/report/cascade.py`
- 问题或结论：初版V13校准artifact只绑定数据hash，没有绑定产生分数的三配置和三checkpoint；同一数据版本下旧阈值理论上可被套到另一组三模型。
- Evidence：初版loader只检查`data_contract_hashes`。修复后focused test构造不同CU checkpoint hash并调用`validate_hierarchical_calibration_deployment`，实际抛出ValueError；正式report入口在启用校准前计算并核对三config semantic hash、三checkpoint SHA-256、共同epoch/global step和selection hash。
- Impact：避免旧模型阈值静默迁移到新模型，防止Coverage/FAR/FRR和Unknown拒识动作失去可追溯来源。
- Action：后续OOF生成器必须写完整`deployment_checkpoint_set`；缺字段、validation/test拟合或部署身份不一致一律拒绝。
- 验证状态：focused负向测试通过；真实OOF未执行。
- 关联版本：V13。

## Finding F134

- Severity：Critical（本轮额外确认并已修复）
- 文件位置：`src-v3/radar_ln3/cascade.py::CascadeOnlineStepResult/OnlineLiquidSoftCascade.step`、V13 CU配置
- 问题或结论：初版V13虽然新增hierarchical action，但在线结果仍通过历史`window_label_id/name/confidence`暴露五类soft posterior argmax；CU配置又把第二logit命名为`UnknownTarget`，调用方仍可能把审计概率误当产品五分类或已验证Unknown识别。
- Evidence：只读搜索确认`window_label_name=LABELS[soft_argmax]`。修复后V13历史window分类字段全部为null，soft posterior迁至显式`soft_posterior_audit_*`；CU checkpoint输出冻结为`CLUTTER_ACCEPT_EVIDENCE/CLUTTER_REJECT_EVIDENCE`，运行时只有校准策略可产生`UNKNOWN_REJECTED`。
- Impact：产品消费者无法再从旧字段误取一次性五分类；真实Unknown标签、Clutter-reject evidence、Unknown拒识动作和pending保持四种独立语义。
- Action：线上集成只读取`final_action/accepted_label/rejected_to_unknown/stable_state`；soft posterior仅作审计。
- 验证状态：CPU/CUDA Smoke验证旧字段禁用、audit字段存在、动作证据字段完整；Stage 048未校准diagnostic性能已评估，真实分层动作性能仍未评估。
- 关联版本：V13。

## Finding F135

- Severity：High（本轮额外确认并已修复）
- 文件位置：`src-v3/radar_ln3/cascade.py::summarize_calibrated_actions`
- 问题或结论：初版Accepted Macro-F1沿用五类矩阵，将产品不存在的`UNKNOWN_ACCEPTED`类以support=0/F1=0纳入宏平均，使四个known类全部正确时理论上限仅0.80；初版Drone/Bird专项Precision/Recall又只在真实D/B子集或已决样本内计算，漏掉其他类别误接收和pending/rejected假阴性。
- Evidence：focused合成探针在修复前可推导四known全对Macro-F1=0.8。修复后四known全对Accepted Macro-F1实际为1.0；Unknown→Bird使Bird Precision=0.5；一个Bird正确接受加一个Bird拒识使Bird Recall=0.5。Pending action confidence现在为null。
- Impact：旧指标会系统性压低Accepted Macro同时高估Drone/Bird安全表现，无法用于90%门槛和拒识放行。
- Action：Accepted Macro只平均Drone/Bird/Balloon/Clutter；Unknown误接收仍计入known Precision和Accepted Accuracy；D/B Recall以所有真实D/B parent为分母。
- 验证状态：focused分母测试及CPU/CUDA动作记录Smoke通过；真实校准指标未评估。
- 关联版本：V13。

## Finding F136

- Severity：High（本轮额外确认并已修复）
- 文件位置：`src-v3/train/cascade.py::_joint_payload/save_task_checkpoint_set`、`src-v3/radar_ln3/cascade.py::load_cascade`、`src-v3/radar_ln3/task_models.py::save/load_task_checkpoint`
- 问题或结论：仅比较三个task checkpoint的epoch/global step/selection不足以证明来自同一次原子保存；同名配置被原位编辑也不能仅靠config_name识别。
- Evidence：V13现把语义规范化配置SHA-256写入task/joint checkpoint，每个epoch生成共同32字符`checkpoint_set_id`并写入三task/joint/run state。加载器要求config identity、正epoch/global step、有效selection、集合ID全部一致；同名配置学习率被改变及缺selection checkpoint均在CPU/CUDA Smoke中被拒绝。
- Impact：防止跨run或半写入checkpoint被误拼为产品级联，保证三checkpoint联合评估和后续校准可归因。
- Action：正式训练只允许`train.cascade`原子写入；禁止手动拼三个task best或编辑配置后继续加载旧V13 checkpoint。
- 验证状态：task/joint恢复差异0，负向测试通过；正式训练和四份best checkpoint共同epoch/global step/set ID核验通过。
- 关联版本：V13。

## Finding F137

- Severity：High（本轮额外确认并已修复）
- 文件位置：`src-v3/radar_ln3/data.py::CQ08PointSequenceDataset`、`src-v3/radar_ln3/model.py::CfCLiquidCausalEncoder.forward/step`
- 问题或结论：初版reason-aware实现校验枚举与dt值，但没有强制首点reason只能出现在每个parent/在线状态的真实起点；脏索引或调用方可能在已有状态上再次触发首点初始化更新。
- Evidence：V3全量检查显示1546个parent均从event0连续、首点reason恰好一次，数据当前无异常。修复后Dataset对全split强制事件连续与首点唯一；focused测试中重复首点、fresh-state非首reason和continued-state再次首reason均被拒绝。
- Impact：防止液态状态被中途重初始化或错用首点gain，保持offline/online及parent reset边界一致。
- Action：任何真实状态重置必须伴随新的parent/session边界与first-event reason；不得跨parent续状态。
- 验证状态：focused及真实CQ08 CPU/CUDA通过；性能收益未评估。
- 关联版本：V13。

## Finding F138

- Severity：Critical（正式监督性能未达到产品门槛）
- 文件位置：`artifacts/RADAR-LN3-V13-cq08-liquid-hierarchical-cascade-formal/training-summary.json`、`src-v3/processed/report/RADAR-LN3-V13-hierarchical-cascade-validation-batch2/evaluation-val.json`
- 问题或结论：V13代码/三层契约修复已进入正式训练，但最佳checkpoint在冻结validation上四项产品护栏仍为`0/4`；不能进入OOF拒识校准、test或部署。
- Evidence：best epoch 13的Gate/DBB/CU/Drone-Bird floor=`0.517241/0.825397/0.500000/0.732143`，目标=`0.95/0.95/0.90/0.90`。20个epoch均`0/4`，各护栏全轨迹最高仅`0.517241/0.852459/0.583333/0.732143`。独立batch=2前向与training summary核心字段逐值一致，规范化SHA-256均为`04b2905edf1b02c71d7f7a44ea6bbd83cf7217cd1e48418a22e8f72f8fd190da`。
- Impact：当前模型仍会把关键Bird路由CU并把Unknown接收到DBB；任何后续阈值只能改变coverage/动作权衡，不能证明基础表征达到质量门槛。
- Action：冻结V13为未放行基线；下一阶段先做train/parent-grouped OOF内部错误覆盖、Unknown子流型和目标贡献归因，不增加液态容量、不查看test、不在validation搜索阈值。
- 验证状态：正式CUDA训练完成，best checkpoint独立恢复与冻结validation复算通过；训练随机过程未重跑，多seed未执行。
- 关联版本：V13。

## Finding F139

- Severity：Critical（Bird/Unknown危险方向错误持续）
- 文件位置：V13规范validation样本明细、V10 12条人工审计报告
- 问题或结论：三层目标、reason-aware dt和原子checkpoint修复没有消除大部分既有困难parent；12条人工审计残差仅2条纠正，10条仍错误。
- Evidence：Gate矩阵`[[195,14],[8,15]]`，其中Bird→CU=11、Unknown→DBB=6。Track 333/845/2367/2956仍以final confidence约`0.858/0.884/0.922/0.959`路由CU；Track 1547/2908/3122/3154/4171仍以约`0.968/0.811/0.828/0.898/0.779`被接收为DBB。Track 852与2862在V13纠正；所有12条标签继续按人工政策保留，未因模型结果改标。
- Impact：Bird误拒和Unknown误接收仍集中在稳定、可复现的parent上，直接阻塞Gate安全性；不能依靠总体Accuracy或Balloon高分掩盖。
- Action：下一诊断以这些持续残差为锚点，在train/OOF内比较TR运动形态、RD覆盖/谱形、dt reason、序列长度和Unknown子流型；不得把模型置信度当标签证据。
- 验证状态：已确认parent-final validation；window演化可从样本明细复核，calibrated action/stable/test未执行。
- 关联版本：V10人工审计、V13正式结果。

## Finding F140

- Severity：High（CU估计不稳定且显著未达标）
- 文件位置：V13 `component_metrics.cu_parent_conditional`和Gate CU指标
- 问题或结论：CU specialist和Gate CU不仅点估计低，而且validation只有Clutter=8、Unknown=15，统计不确定性很大；不能把Clutter Recall=0.875解释为接近90%放行。
- Evidence：Gate CU P/R/F1=`0.517241/0.652174/0.576923`；CU specialist Macro P/R/F1=`0.694444/0.704167/0.651515`。Gate CU Precision/Recall 95% Wilson区间约`[0.3443,0.6861]`/`[0.4489,0.8119]`；Clutter Recall区间约`[0.5291,0.9776]`，Unknown标签Recall区间约`[0.3012,0.7519]`。
- Impact：少数parent变化即可大幅改变CU点估计；固定validation已被V8–V13反复查看，进一步用它搜索threshold、loss weight或checkpoint会加重过拟合。
- Action：validation只保留冻结验收用途；模型设计和阈值依据必须来自train parent-grouped OOF。首先确认CQ08允许范围内CU子流型/覆盖是否足以支持目标，否则明确数据阻塞。
- 验证状态：描述统计和Wilson区间已复算；独立test、跨source和多seed均未执行。
- 关联版本：V13。

## Finding F141

- Severity：Medium（报告口径问题；Stage 049已修复）
- 文件位置：`src-v3/radar_ln3/cascade.py::evaluate_cascade`输出字段`dt_reason_counts_including_padding`
- 问题或结论：字段中的`padding`不是数据集固有padding总数，而是DataLoader按每个batch最长序列对齐产生的collate位置，因batch size/分组而变化。
- Evidence：同一三checkpoint、同一232 parent validation，batch=2报告padding=416，batch=8报告padding=1334；所有分类指标一致。batch=2复跑与training summary核心字段逐值一致。
- Impact：若把该数用于比较数据版本、dt缺失率或模型间padding比例，会产生虚假的batch依赖差异；当前mask后分类结果不受影响。
- Action：V13/V14现分别输出`dt_reason_counts_data_events`、`collate_padding_positions`和`evaluation_batch_size`；历史报告保持只读，不把旧padding字段用于数据质量比较。
- 验证状态：问题已复现并完成代码级修复；compileall、V13/V14 focused及历史级联回归通过，test未执行。
- 关联版本：V13报告语义，数据版本不变。

## Finding F142

- Severity：High（训练优化与产品选优分离已被真实结果确认）
- 文件位置：V13 `training-summary.json::history`、`src-v3/train/cascade.py::build_product_checkpoint_selection`
- 问题或结论：训练loss持续下降不保证产品级联改善；冻结产品选优正确避免了用更低loss或更高五类Macro-F1事后替换弱护栏更差的checkpoint。
- Evidence：总loss从epoch 1的`1.451318`降至epoch 20的`0.344242`，但epoch 20最小归一化护栏完成度仅`0.138889`；best epoch 13为`0.544465`。epoch 15 diagnostic Macro-F1=`0.717829`高于epoch 13的`0.685081`，但最弱护栏完成度`0.540541`更低，因此未被选中。
- Impact：继续增加epoch、按train loss选best或事后切换到Macro-F1更高epoch，均会违反产品安全排序且不能解决CU/Bird/Unknown短板。
- Action：保留epoch 13为冻结V13 best；下一实验仍以事前产品护栏排序，报告loss只作优化诊断。
- 验证状态：20 epoch全轨迹已审查；没有事后改写checkpoint。
- 关联版本：V13。

## Finding F172

- Severity：Critical（V15产品监督目标失配；已确认）
- 文件位置：`src-v3/radar_ln3/task_models.py::STRICT_OPEN_SET_TASK_SPECS`、`src-v3/radar_ln3/losses.py::strict_open_set_final_state_nll`、六份V15配置
- 问题或结论：V15 CU specialist学习的是Clutter接受证据与DBB-known非Clutter证据，不是ClutterTarget/UnknownTarget二分类；UnknownTarget被排除在loss、normalization、OOF和阈值拟合之外。
- Evidence：V15 CU class names为`CLUTTER_ACCEPT_EVIDENCE/NON_CLUTTER_KNOWN_EVIDENCE`，Drone/Bird/Balloon映射1；loss对label 4直接报错，且`cu_targets=dbb_mask`。V15两臂OOF各自只有Drone/Bird/Balloon/Clutter 1011 parent，Unknown count=0。
- Impact：V15 CU Precision/Recall只能描述Clutter versus DBB-known，不可解释为Clutter/Unknown性能；V15不能实现用户冻结的Gate→CU→Clutter/Unknown产品链路。
- Action：V16恢复Gate的DBB/CU五标签映射，DBB specialist只用DBB，CU specialist只用Clutter/Unknown；真实Unknown标签预测与运行时`UNKNOWN_REJECTED`动作使用不同字段和指标。
- 验证状态：代码、配置、loss和两臂OOF记录已核实；V16修复未执行。
- 关联版本：V15失效问题，V16阻塞。

## Finding F173

- Severity：Critical（分层动作路由失配；已确认）
- 文件位置：`src-v3/radar_ln3/rejection.py::StrictOpenSetRejectionPolicy.decide`第740–781行及对应focused断言
- 问题或结论：Gate接受DBB后，若DBB knownness失败，当前策略直接输出`UNKNOWN_REJECTED`，没有回退到CU specialist。
- Evidence：代码路径返回`gate_action=DBB_ROUTE_ACCEPTED`、`conditional_action=DBB_KNOWN_NOT_ACCEPTED`、`final_action=UNKNOWN_REJECTED`；现有strict-open-set focused把该结果作为通过条件。
- Impact：低置信Bird/Drone/Balloon失去CU复核机会；Unknown拒识原因混合了DBB knownness失败和CU Unknown判断，不能解释三层动作来源。
- Action：V16局部训练阶段先只输出三个task各自概率与混淆；后续动作实现必须让DBB未接受回退CU，CU判Clutter或Unknown，禁止在DBB分支直接终止为Unknown。
- 验证状态：代码路径已确认；V16动作修复和完整级联暂未执行。
- 关联版本：V15失效问题，V16动作候选。

## Finding F174

- Severity：Critical（V15评估范围不能验证真实CU/Unknown；已确认）
- 文件位置：`src-v3/radar_ln3/cascade.py::build_cascade_dataset/evaluate_strict_open_set_cascade`、`src-v3/radar_ln3/angular_knownness_oof.py`、V15 experiment/nested OOF contract
- 问题或结论：V15正式训练、checkpoint selection、两臂nested OOF和calibration均只使用known parent；development Unknown也未执行，因此继续分析V15不能补出真实Clutter/Unknown或五类开放集性能。
- Evidence：两臂各5折均覆盖1011个known parent且Unknown/validation/test暴露为0；两份calibration均绑定known-only分数且`deployment_eligible=false`。development Unknown、test、stable和P95产物不存在。
- Impact：V15 known FRR、Clutter acceptance和DBB knownness不能冒充Unknown拒识Recall、FAR、FRR、Coverage、Selective Risk或最终五类动作。
- Action：冻结V15为研究诊断；V16三任务先独立训练和评估，完整级联与拒识校准另设后续Stage。
- 验证状态：当前真实磁盘状态为Control/Treatment均5/5 OOF完成；早期Treatment 3/5状态已过期。
- 关联版本：V15冻结。

## Finding F175

- Severity：Critical（跨split同源内容暴露；已确认）
- 文件位置：`src/process/cq08_split.py`、`data/manifests/cq08_unified_manifest.json`、`data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-split.json`及四个指定parent
- 问题或结论：V3只按track_id划分，未按capture/raw/RD内容等价组件隔离；至少两组同capture、同raw、同RD内容跨train/val。
- Evidence：manifest为1550条、45个重复capture组。1339/train与1490/val共享capture `20250616122328`、raw SHA=`761860...c87cb`、RD SHA=`9cb42c...6c1a3`；1343/val与1494/train共享capture `20250616122337`、raw SHA=`3bb1a5...f15c`、RD SHA=`7e29ab...440b`。每组track/point文件hash不同。
- Impact：V15和历史validation可能包含同一采集/回波/RD内容的近重复暴露，不能宣称可靠跨采集泛化；局部高分可能被泄漏抬高。
- Action：V16建立capture/raw/RD/规范轨迹内容的provenance组件并以组件为split group；完全等价内容零跨split，跨标签完全等价组未经人工政策裁决不得正式训练。
- 验证状态：两组已独立hash复现；45个重复capture中10个跨标签组的raw/RD hash均不相同，当前无跨标签完全等价内容证据；全部内容组件和规范轨迹hash仍待V16工具审计。
- 关联版本：V3/V15数据风险，V16数据重建阻塞。

## Finding F176

- Severity：High（同秒合法观测被误计为gap；已确认）
- 文件位置：`src-v3/radar_ln3/cascade.py::OnlineLiquidSoftCascade.step/evaluate_strict_open_set_cascade`
- 问题或结论：offline/online都把`tr_dt_reason==2`累计为`zero_delta_count`后直接作为`gap_fraction`，将秒级精度下的合法同秒观测误当轨迹缺口。
- Evidence：代码分别以`zero_delta_count/max(1, valid_points-1)`传入policy；V3契约定义reason 2为`zero_delta_at_second_precision`并执行bounded observation update，真实全索引该reason为9776个事件。
- Impact：质量门控会产生source、类别和轨迹长度相关误拒偏置，并使正常同秒观测触发pending/gap guard。
- Action：V16分别记录`zero_dt_fraction`与真实`gap_fraction`；reason 2不得增加gap，首点和padding也不进入gap分母。
- 验证状态：代码与V3契约已核实；修复及offline/online回归未执行。
- 关联版本：V15动作语义，V16修复。

## Finding F177

- Severity：High（prefix训练与产品验收单位未统一；已确认）
- 文件位置：`src-v3/radar_ln3/losses.py::strict_open_set_final_state_nll`、`src-v3/train/cascade.py::build_product_checkpoint_selection`、online streaming路径
- 问题或结论：V15以parent-final为主损失、prefix 0.10辅助并只按parent-final护栏选checkpoint，但online每个prefix都会产生路由和stable输入；早期动作未成为同一产品验收目标。
- Evidence：loss取`length-1`计算final NLL，event 3至final前状态仅进入辅助项；checkpoint ranking只读取known parent metrics，在线step逐事件调用拒识/stable。
- Impact：parent-final改善不保证早期Bird不被路由CU、Unknown不被接受DBB或不过早拒识；window、parent和stable指标不能互换。
- Action：V16独立task先分别报告prefix与parent-final；最小证据量之前为PENDING。prefix loss形式作为后续单因素实验，不与split/标签修复同时调参。
- 验证状态：调用链已确认；V16 focused gradient与动作验收未执行。
- 关联版本：V15目标风险，V16实验设计。

## Finding F178

- Severity：High（流程完成与性能通过混淆；已确认）
- 文件位置：`src-v3/train/cascade.py`训练摘要生成及两臂V15 `training-summary.json`
- 问题或结论：训练摘要无条件写`status=passed`，即使产品护栏失败，消费者也可能误认为模型达标。
- Evidence：Control摘要`status=passed`但best guardrails=`1/4`，Gate/DBB/CU/Drone-Bird floor=`0.6364/0.9245/0.5833/0.9038`；Treatment为`0/4`和`0.7000/0.7538/0.7000/0.7460`。
- Impact：可能错误放行calibration、test或部署，也会把训练命令成功与模型性能验收混为一谈。
- Action：V16摘要使用`status=completed|failed`描述流程，另设`performance_status=passed|failed`、`guardrails`、`deployment_eligible=false`；独立task训练不声称完整产品达标。
- 验证状态：真实摘要已核实；V16实现未执行。
- 关联版本：V15报告缺陷，V16修复。

## Finding F179

- Severity：Critical（V15不再是当前产品候选；结论已冻结）
- 文件位置：V15全部实现、六配置、两臂正式artifact、10份nested OOF prediction和Stage 057复核记录
- 问题或结论：V15实验执行和known-only角度knownness数值是真实可复现的，但其CU标签、路由、split和评估语义不匹配当前产品设计；因此“实验执行真实”不等于“产品性能结论有效”。
- Evidence：两臂正式训练及5/5 OOF完整；同时Finding F172–F178均由代码、配置、hash、artifact或契约复核确认。
- Impact：不能使用V15高分、OOF或calibration选择V16模型、阈值或部署；V15只保留为历史研究诊断和回归输入。
- Action：按`.rules`保存V15快照后进入V16；V16重新版本化index/split/stats/contract/config/checkpoint，三任务先独立训练评估，不执行完整级联。
- 验证状态：V15冻结；V16未执行。
- 关联版本：V15冻结、V16起点。

## Finding F180

- Severity：Critical（跨split同源暴露已修复）
- 文件位置：`src-v3/build/product_local_v16.py`、V16 provenance/split/index三文件
- 问题或结论：V3按track_id切分产生的capture/content同源跨split已通过V16连通组件重切分消除。
- Evidence：1546个parent形成1501个组件，45个多parent组件中27个原跨split；保守迁移27个parent且旧val/test→train为0。capture/raw/RD/规范track/规范point/模型输入语义六类键的跨split违反数均为0；跨标签content-equivalent冲突0组。
- Impact：V16局部validation不再包含已识别的同capture/同raw/同RD跨split暴露；历史V3–V15泛化指标仍受旧split风险限制，不能被追溯性洗白。
- Action：V16正式训练只允许加载新index/contract/stats hash；后续版本若改变provenance键或split必须再次递增数据版本。
- 验证状态：首次全量构建和第二次幂等全量复核均通过；正式分类性能未执行。
- 关联版本：V16。

## Finding F181

- Severity：Critical（真实CU监督目标已修复）
- 文件位置：三份V16配置、`src-v3/radar_ln3/task_models.py::TASK_SPECS/build_task_dataset/verify_task_data_contract`
- 问题或结论：V16 CU specialist恢复为`ClutterTarget` versus `UnknownTarget`真实二分类，不再使用Drone/Bird/Balloon作为Unknown替代负类。
- Evidence：CU train/val parent=`102/24`，标签support为train Clutter/Unknown=`33/69`、val=`9/15`；底层Dataset和task view都只包含这两类。Gate train/val为DBB/CU=`953/102`与`223/24`。
- Impact：V16可以真实评估局部Clutter/Unknown标签边界；但真实Unknown标签预测仍不等同运行时`UNKNOWN_REJECTED`开放集动作，后者尚未OOF校准或正式评估。
- Action：正式局部报告分别输出CU两类P/R/F1；不得把Unknown分类Recall写成拒识Recall/FAR/FRR。
- 验证状态：focused数据/标签/contract测试通过；正式训练和性能未执行。
- 关联版本：V16。

## Finding F182

- Severity：High（窗口级训练目标与局部产品目标已统一）
- 文件位置：`src-v3/radar_ln3/losses.py::parent_final_task_ce_with_prefix_auxiliary`、`src-v3/radar_ln3/task_models.py::task_training_objective/evaluate_task`
- 问题或结论：V16主损失只监督每parent完整序列最后合法状态；event≥3且非final的causal prefix仅作为0.10辅助并按parent均衡。前两事件不作为可用产品动作。
- Evidence：V16最长parent=29且配置max=29，不存在截断伪final。focused探针得到padding gradient max=0、event<3非final gradient max=0、eligible prefix只来自声明范围；class-weight辅助分母与weighted CE一致。
- Impact：训练loss与独立checkpoint选择都以parent-final局部任务为主，不再让所有早期证据不足prefix无条件主导；同时保留早期时序表征辅助。该设计尚未证明比final-only或prefix-ramp性能更优。
- Action：先完成冻结V16三任务训练；若未达标，下一单因素实验才比较final-only/final+prefix/prefix-ramp，不同时改变split、输入或液态容量。
- 验证状态：数学/梯度/CPU/CUDA单步通过；正式性能未执行。
- 关联版本：V16。

## Finding F183

- Severity：High（有效dt来源与gap语义已修复）
- 文件位置：`src-v3/radar_ln3/data.py`、`src-v3/radar_ln3/cascade.py::hierarchical_time_quality_fractions`
- 问题或结论：V16保留原始0秒与reason-aware dynamics；同秒观测不再计为gap。有效时间为零的真实来源是1546个首点无前驱和9776个秒级重复时间戳。
- Evidence：29663事件中first/zero/positive=`1546/9776/18341`；raw zero比例=`38.1688%`，same-second占全部=`32.9569%`、占非首转换=`34.7690%`。负dt=0；同秒仍执行bounded observation。当前V16无显式真实gap reason，故gap_fraction=0。
- Impact：质量门控不再因合法同秒观测产生类别/source/长度相关误拒偏置；没有伪造更高时间精度。
- Action：未来若发现真实缺失/超阈值间隔，必须新增显式reason并版本化，不得重新复用same-second计数。
- 验证状态：全量index校验、focused和V13回归通过；正式动作/stable未执行。
- 关联版本：V16及共享未来动作接口。

## Finding F184

- Severity：Critical（DBB失败直达Unknown路由已修复）
- 文件位置：`src-v3/radar_ln3/rejection.py::StrictOpenSetRejectionPolicy.decide`、V14/V16 focused
- 问题或结论：Gate接受DBB但DBB knownness未接受时，现在显式回退CU；CU接受Clutter则输出Clutter，否则才输出Unknown拒识。
- Evidence：合成探针得到`DBB_ROUTE_ACCEPTED_THEN_FALLBACK_TO_CU→CLUTTER_ACCEPTED`和同路由→`UNKNOWN_REJECTED`两条路径。旧V14/V15 action contract仍声明DBB失败直达Unknown，三配置均被当前验证器fail-closed。
- Impact：未来产品动作可区分Gate原路由CU与DBB knownness失败后的CU复核，避免低置信Bird/Drone/Balloon无条件直达Unknown。V15旧calibration与CU目标仍不适配新路由，不能复用。
- Action：V16局部阶段不启用正式拒识；后续新版本必须用真实CU与train parent-grouped OOF重新校准和逐事件验收。
- 验证状态：合成路由与旧契约负向测试通过；正式拒识、stable、FAR/FRR未执行。
- 关联版本：共享路由修复；V16后续产品阶段。

## Finding F185

- Severity：High（流程状态与性能状态已分离）
- 文件位置：`src-v3/train/tasks.py`、`src-v3/report/tasks.py`、`src-v3/train/cascade.py`
- 问题或结论：V16成功训练流程写`status=completed`，异常中止写`status=failed`；局部性能独立写`performance_status`并同时检查逐类Precision/Recall/F1。未来联合训练摘要也不再无条件写passed。
- Evidence：CPU/CUDA一阶段Smoke三任务均为`completed/not_evaluated/deployment_eligible=false`；合成0.94弱指标在0.95门槛下得到`performance_status=failed`。正式CLI缩短参数被拒绝。
- Impact：artifact消费者不能再把命令运行完成冒充模型达标，也不能把局部三任务通过冒充完整级联/部署通过。
- Action：正式训练后先读取`performance_status`和逐类guardrails；OOF/test/stable/P95未执行时`deployment_eligible`保持false。
- 验证状态：focused与CPU/CUDA单步通过；运行中失败摘要代码已编译，未故意制造正式失败artifact。
- 关联版本：V16及未来联合入口。

## Finding F186

- Severity：High（V16局部checkpoint不可直接冒充完整级联）
- 文件位置：`src-v3/radar_ln3/cascade.py::validate_cascade_configs`、三份V16配置
- 问题或结论：V16被明确限定为三个独立局部run；通用cascade组装、报告和联合训练入口对V16 product-local schema fail-closed。
- Evidence：focused调用`validate_cascade_configs`得到“restricted to independent task training and evaluation”错误；三任务各有独立artifact、best epoch和局部门槛，不存在联合checkpoint set或产品组合选择。
- Impact：防止把三个局部best直接相乘后宣称五类/开放集性能，也避免在未定义联合checkpoint协议时污染归因。
- Action：本轮只运行三条`train.tasks`和三条`report.tasks`；完整级联必须创建后续版本并显式定义组装、联合选优、OOF拒识和stable验收。
- 验证状态：负向scope测试通过；完整级联未执行。
- 关联版本：V16 product-local。

## Finding F187

- Severity：High（CU低support仍是未解决性能风险）
- 文件位置：V16 grouped split/data contract
- 问题或结论：provenance重切分提高独立性，但CU support仍很低，且比旧V3 train Clutter 35降为33；validation仅Clutter=9、Unknown=15。
- Evidence：V16五类train/val/test为`476/239/238/33/69`、`113/56/54/9/15`、`109/53/58/8/16`。迁移中1个train Clutter进入val、1个train Clutter进入test、1个train Unknown进入test。
- Impact：单个CU parent会显著改变P/R/F1，0.90门槛的点估计方差大；即使一次validation达标，也不能单独证明稳定泛化或严格open-set安全。
- Action：先执行冻结局部基线；后续使用train parent-grouped OOF/parent bootstrap报告置信区间和子流型覆盖，不复制window、不从val/test回采、不因模型错误改标签。
- 验证状态：support/provenance已确认；正式CU性能、OOF区间、test均未执行。
- 关联版本：V16。

## Finding F188

- Severity：High（微多普勒历史探针证据失效；已纠正）
- 文件位置：`src-v3/radar_ln3/micro_doppler.py`、`src-v3/smoke/micro_doppler_v17_contract_test.py`
- 问题或结论：先前声称的`Drone CV=0.52 vs Bird CV=0.13`来自错误的Python I/Q reshape，不能证明微多普勒判别潜力或作为V17收益先验。
- Evidence：现有MATLAB解析把交错复数样本按列reshape为`[31,PRTnum]`；V17改为`order="F"`后，代表事件归一化CV为Drone Track 1=`0.127563`、Bird Track 333=`0.181750`、Balloon Track 3010=`0.060699`。三类各3帧的目标距离行与MATLAB逐值一致。
- Impact：若继续使用旧数值冻结阈值或预期方向，会把解析错误当成物理规律，并可能选择迎合错误先验的特征公式。
- Action：旧数值明确作废；V17特征公式不按三条代表事件调参，正式训练必须先通过train-only全量非退化、效应和采样模式审计。
- 验证状态：focused Smoke通过；全量train-only分布尚未执行。
- 关联版本：V17。

## Finding F189

- Severity：High（采样模式捷径风险；已建立训练前门槛）
- 文件位置：`src-v3/build/micro_doppler_v17.py`、`src-v3/report/micro_doppler_v17_readiness.py`
- 问题或结论：STFT速度映射、时间bin数量和调制周期都受PRTnum、PRT秒与载频影响；若这些采集模式与DBB标签强相关，三维摘要可能学习雷达模式而非旋翼/扑翼物理结构。
- Evidence：raw帧显式携带frequency、PRTnum、PRT和CPI count；V17 evidence cache现逐帧保存这些元数据。readiness按标签报告分布、计算PRTnum/PRT秒/载频的Cramér's V，并在相同三元采样模式内重新计算Drone/Bird robust effect。
- Impact：仅看全局Drone/Bird均值差可能得到来源捷径驱动的假收益，validation改善也未必能迁移到新采集模式。
- Action：train-only要求单字段Cramér's V≤0.50，且至少一个Drone/Bird各≥8有效事件的共享模式内存在绝对effect≥0.10；失败则流水线停止，不启动正式训练。采样元数据和CPI count不得进入模型。
- 验证状态：代码/报告契约已实现和编译；真实全量门槛尚未执行。
- 关联版本：V17。

## Finding F190

- Severity：High（STFT峰可能偏离目标多普勒；已建立训练前对齐门槛）
- 文件位置：`src-v3/radar_ln3/micro_doppler.py`、`src-v3/report/micro_doppler_v17_readiness.py`
- 问题或结论：仅在目标距离行取STFT全局积分峰，仍可能被该距离行的零速杂波或旁瓣主导；若目标速度落在冻结的±8 m/s峰频带外，CV与调制周期就不再描述目标微动。
- Evidence：V17现同时保存局部MTD目标Doppler index/物理速度和STFT积分峰速度。Drone/Bird/Balloon各3帧focused探针最大绝对偏差约`4.77 m/s`，9/9均在±8 m/s内。
- Impact：代表帧通过不能保证全量；若某类或某采集模式大量偏离，Treatment可能学习杂波闪烁而非旋翼/扑翼。
- Action：train-only readiness要求至少90%逐帧目标多普勒落入STFT峰±8 m/s频带，并报告中位、P95和最大偏差；失败即停止，不事后扩大频带迎合validation。
- 验证状态：focused 9帧通过；全量train对齐率尚未执行。
- 关联版本：V17。

## Finding F191

- Severity：High（V17单因素配对与模型输入契约；已实现）
- 文件位置：`src-v3/radar_ln3/data.py`、`src-v3/radar_ln3/task_models.py`、`src-v3/train/tasks.py`
- 问题或结论：V17已把实验变量限制为Treatment新增三维当前事件微多普勒摘要；两臂共同加入四维`rd_physical`，其余架构、液态状态、loss、seed、parent顺序和训练规则保持一致。
- Evidence：Control/Treatment Fusion输入维度=`83/86`；40个共享state entry逐位相同，Treatment新增3列严格为0，初始logits逐位一致，新列梯度最大值=`0.666845`。两臂固定seed42、20 epoch、无early stopping并记录每epoch parent hash。
- Impact：若Treatment改善，可归因于三维摘要相对“已有四维RD物理特征”的增量；不能归因于完整STFT网络、液态容量、Gate/CU或拒识机制。
- Action：paired报告必须核验全部20 epoch parent顺序、各自best与不同same-epoch护栏，并要求Drone↔Bird方向一致改善和三类P/R/F1≥0.95。
- 验证状态：focused模型、梯度、两臂checkpoint round-trip通过；正式训练未执行。
- 关联版本：V17。

## Finding F192

- Severity：Medium（完整STFT张量定位；合同已明确）
- 文件位置：`src-v3/radar_ln3/micro_doppler.py`、V17 data contract/manifest生成路径
- 问题或结论：本轮模型不消费完整Doppler-time STFT张量；它仅作为版本化证据保存，避免把“3维摘要实验”与“STFT深网新输入契约”混为一个变量。
- Evidence：每raw帧保存`64×16 float16`相对谱、目标距离行、raw frame index、逐帧摘要和采样元数据；checkpoint model contract固定`complete_stft_tensor_is_model_input=false`。模型只接收3维[0,1]摘要。
- Impact：本轮负结果只能否定当前摘要及聚合口径，不能否定完整Doppler-time谱；正结果也不能宣称STFT张量网络有效。
- Action：只有V17摘要paired通过且需要进一步保留时，才另开后续版本研究完整张量编码；不得在V17中增加CNN/attention或液态容量。
- 验证状态：合同、cache schema和负向输入检查已实现；正式缓存尚未构建。
- 关联版本：V17，完整张量模型属于后续版本。

## Finding F193

- Severity：High（V17训练状态边界；当前仅可prepare）
- 文件位置：`src-v3/scripts/v17_micro_doppler_paired_pipeline.ps1`、`src-v3/report/micro_doppler_v17_readiness.py`
- 问题或结论：V17 focused Smoke通过不等于已达到正式训练状态；旧探针失效后，必须以train-only全量分布和采样模式门槛作为训练前硬闸。
- Evidence：当前没有V17 cache、manifest、data contract、生成配置、正式artifact或paired validation报告。脚本`prepare`串行运行builder、focused、readiness和静态隔离；readiness非passed即退出，train不会启动。
- Impact：跳过prepare直接根据三条代表事件训练，会产生选择偏差，且无法判断调制周期不可估率或采样模式混杂。
- Action：用户先运行`-Stage prepare`；仅退出码0后再运行`-Stage train`和`-Stage evaluate`，或一次性`-Stage all`让失败自动停止。`train.tasks`本身也强制绑定readiness v3、两臂配置/data hashes及canonical frame-index hash，无法通过直接Python CLI绕过。
- 验证状态：脚本AST、compileall、focused、V16回归和静态隔离通过；全量数据处理与训练按用户约束未执行。
- 关联版本：V17。

## Finding F194

- Severity：Critical（V17原始回波帧错配；已修复并建立反例回归）
- 文件位置：`src-v3/build/micro_doppler_v17.py`、`src-v3/radar_ln3/micro_doppler.py`、`src-v3/report/micro_doppler_v17_readiness.py`、`src-v3/smoke/micro_doppler_v17_contract_test.py`、`src/process/build_cq08_rd_physical_canonical_store.py`
- 问题或结论：V17旧构建器把V16的`rd_raw_frame_indices_json`直接作为完整CQ08 `.dat`帧号，但该字段源自canonical构建时的`enumerate(rd_mat_files)`，真实语义是MAT列表零基序号。MAT无缺失时两者数值巧合一致；Track 1010缺少前8个MAT文件时必然错配并使全量prepare失败。
- Evidence：Track 1010 `.dat`解析得到54帧，manifest只有46个MAT文件且文件名帧号为9–54。V16 point 3记录tensor index=`0`、legacy ordinal=`0`；canonical frame 0的`rd_mat_path`为`..._0009_1010.mat`，因此真实raw-DAT index=`8`。raw index 0的target record为point 1且与该MAT `data_out`最大绝对差=`277.504744`；raw index 8的target record为point 3且与MAT I/Q最大绝对/相对误差均为0。全106026行canonical frame index均可解析MAT帧号、track ID全匹配；仅Track 1010的46行与legacy ordinal不同，偏移统一为`+8`。
- Impact：旧V17无法完成全量prepare；若代码仅按数组边界读取且未检查point，还可能把其他track point的I/Q错误配给当前事件，污染三维摘要、readiness和后续配对归因。该问题不改变V16模型现有canonical RD输入，因为V16继续按tensor index读取NPZ；影响限定为V17重新读取完整`.dat`的链路。
- Action：V17改为`parent_sample_id+rd_tensor_frame_indices_json→SHA-256绑定的canonical frame index→rd_mat_path文件名一基采集帧号→raw-DAT零基帧号`；逐帧核验track、track point、tensor path和raw target record。证据缓存分别保存legacy MAT ordinal、canonical tensor index、MAT采集帧号和raw-DAT index；summary/STFT schema升级v2、paired protocol升级v2、checkpoint升级v2、readiness升级v3。readiness冻结要求train检出Track 1010的46个偏移帧且val为0，旧schema/缓存/report/checkpoint均fail-closed。
- 验证状态：Track 1010全部46个保留MAT帧逐帧I/Q零误差和track-point匹配通过；真实`_build_split`内存Smoke完成18事件、16有效事件和46帧STFT，四套索引序列精确；V17 focused Smoke v2、V16 product-local回归、全`src-v3` compileall、静态隔离和PowerShell AST均通过。全量V17 prepare、正式训练、paired validation、OOF、test、正式拒识、stable和P95仍未执行。
- 关联版本：V17修订协议v2；V16 index/split/checkpoint不修改。

## Finding F195

- Severity：High（训练前物理门禁发现提取器目标漂移；已修复，修订版全量验证待执行）
- 文件位置：`src-v3/radar_ln3/micro_doppler.py`、`src-v3/report/micro_doppler_v17_readiness.py`、`src-v3/build/micro_doppler_v17.py`
- 问题或结论：旧V17 v2在目标距离行上以无约束全局STFT积分峰作为三维摘要频带中心；更强零速杂波可夺取全局峰，使CV、调制周期和全局`-10 dB`展宽不再描述局部MTD已定位目标。首次全量prepare因此被90%目标覆盖门禁正确阻断。
- Evidence：旧v2 train目标峰±8 m/s覆盖=`58038/65879=0.880979`，validation=`13486/15323=0.880115`；train目标Doppler位于固定±85 m/s轴内的比例=`0.999772`，仅15帧越界，不能解释7841帧失败。Python局部MTD目标Doppler与MATLAB代表帧`detected_doppler_mps`最大误差≤`0.01083 m/s`，目标锚点正确。合成目标+强零速杂波探针中，无约束全局峰距目标`20.8982 m/s`，目标邻域峰距目标`0.6891 m/s`。
- Impact：若降低readiness门槛或直接训练，Treatment可能学习同距离单元杂波闪烁而非Drone/Bird微动；旧v2分类结果即使提升也不可归因于目标微多普勒。
- Action：保持90%门槛不变；在局部MTD目标Doppler±8 m/s内选择目标局部STFT峰，CV/周期围绕该峰，`-10 dB`展宽限制为目标峰连续谱瓣；全局峰仅作诊断。另要求至少90% train帧的目标局部峰/全局峰功率比≥0.10（−10 dB），使训练门禁仍能独立发现目标被杂波淹没，而不是仅检查按定义成立的几何邻域。旧v2证据只读估算train/val通过比例约`0.961445/0.964498`，validation不参与阈值定义。新产物使用独立`target-anchored-v3`路径，summary/STFT、paired/checkpoint、readiness分别为v3/v3/v4，旧v2产物只读保留。
- 验证状态：target-anchored focused Smoke v3、MATLAB Doppler交叉核验、强杂波反例、Track 1010映射、V16回归、compileall、静态隔离和PowerShell AST通过；修订版全量prepare、正式训练、paired validation、OOF、test、正式拒识、stable和P95未执行。
- 关联版本：V17 target-anchored修订协议v3；V16数据和checkpoint不修改。

## Finding F196

- Severity：High（V17局部配对未通过；Bird Recall表观上升伴随更大的Precision与总体性能损失）
- 文件位置：`src-v3/processed/report/RADAR-LN3-V17-target-anchored-v3-micro-doppler-paired-validation.json`、两臂`sample_detail_information.json`与`training-summary.json`
- 问题或结论：own-best下Treatment并未提升DBB整体判别，只把决策边界向Bird扩张。Control epoch 15的Bird Precision/Recall/F1为`0.925926/0.892857/0.909091`，Treatment epoch 20为`0.879310/0.910714/0.894737`；Recall只多判对1个Bird parent，却因Drone→Bird由3增至6使Precision下降4.66个百分点、Bird F1下降1.44个百分点。总正确parent由209降为208，Accuracy由`0.937220`降为`0.932735`，Macro-F1由`0.932380`降为`0.928712`，Drone↔Bird双向错误由7增至9。
- Evidence：两臂parent-final逐ID对齐后只有10个parent改变类别，其中Control正确/Treatment错误为5个、Control错误/Treatment正确为4个、两者均错但错误类改变为1个。Treatment纠正Bird Track 444/665/816和Drone Track 1945，但破坏Drone Track 1535/4237/4238与Bird Track 531/2687，并把原本Drone→Balloon的Track 3960改成Drone→Bird。50,000次固定seed=`17042`的parent bootstrap仅作不确定性诊断：Bird Recall差的95%区间约`[-5.88,+9.80]`个百分点，Bird Precision约`[-12.03,+2.01]`个百分点，Macro-F1约`[-3.24,+2.53]`个百分点；5对4的discordant correctness McNemar exact `p=1.0`。这些区间不能消除不同best epoch带来的选择偏差。
- Impact：单看Bird Recall会把阈值/边界移动误写成新信息增益；当前Treatment三类逐类P/R/F1均未达到0.95，paired gate状态为`completed_but_gate_failed`，不能进入多seed、完整级联、OOF、test、正式拒识、stable、P95或部署。
- Action：保留V17为负向但可复核的DBB局部摘要实验；后续结论同时报告Bird Recall、Bird Precision、Drone→Bird、Bird→Drone、双向总错误和Macro-F1，禁止只优化或展示Recall。
- 验证状态：target-anchored v3两臂20 epoch正式训练与paired validation已执行；完整五类/级联、OOF、test、正式拒识、stable和P95未执行。
- 关联版本：V17 target-anchored修订协议v3。

## Finding F197

- Severity：High（own-best Bird Recall差异受checkpoint选择混杂，不能直接归因于微多普勒）
- 文件位置：`src-v3/train/tasks.py::build_checkpoint_selection`、V17两臂`training-summary.json`、V17 paired report的`same_epoch_attribution_guards`
- 问题或结论：Control best为epoch 15，Treatment best为epoch 20，二者都按`parent_min_precision_recall`独立选择。相同epoch 15时Treatment Bird Recall反而比Control低5.36个百分点，Macro-F1低2.26个百分点；相同epoch 20时Treatment Bird Recall低3.57个百分点，Macro-F1低1.31个百分点。因此own-best的Bird Recall `+1.79`个百分点不是受控的同epoch增益。
- Evidence：Treatment epoch 19的Macro-F1=`0.944209`、Bird P/R=`0.873016/0.982143`、Drone→Bird/Bird→Drone=`7/0`，但epoch 20因最弱P/R从`0.873016`微升到`0.879310`而胜出，尽管Macro-F1降到`0.928712`且双向错误从7增至9。全20个same-epoch轨迹中Treatment Bird F1较高13次、Macro-F1较高12次，但Bird Precision仅9次、Recall仅10次，方向不稳定；这些epoch高度相关，不能当作20次独立重复。冻结paired报告要求的不同same-epoch护栏没有通过。
- Impact：当前选优协议对“所有类最弱P/R”是合理的局部护栏，但没有直接编码Drone↔Bird方向错误；在该单因素实验中会把训练轨迹差异与特征效应混合。事后把epoch 18/19改称best又会形成validation回看挑选，不能作为V17收益。
- Action：V17不追认epoch 18/19。若未来重做该方向，必须在训练前冻结checkpoint的Pareto/层级规则，将逐类P/R/F1门槛置于首层，并在同层候选中再最小化Drone↔Bird双向错误、最后比较Macro-F1；Control/Treatment仍需same-epoch归因护栏和多seed确认。
- 验证状态：两臂20个epoch的parent顺序hash与初始CPU logits均一致；own-best与same-epoch只读归因已完成，未重选checkpoint。
- 关联版本：V17；未来规则变更必须使用新实验版本。

## Finding F198

- Severity：High（三维摘要存在真实类间信号，但受采样模式条件反转、类内重叠和液态累积放大限制）
- 文件位置：`src-v3/radar_ln3/micro_doppler.py`、V17 target-anchored readiness、summary/STFT cache与两臂parent/prefix明细
- 问题或结论：全局train/val上Bird的CV和谱展宽通常高于Drone，但这个方向并非采样模式不变。train事件级Drone−Bird robust effect在CV/展宽/周期上为`-0.517/-0.250/0.000`，validation为`-0.557/-0.375/+0.103`；然而PRTnum=142的两个共享模式中三项effect全部反向为正，约`+0.80至+1.10`。当前readiness只要求“至少一个共享模式存在绝对效应”，没有要求各主要模式方向一致，因此能通过非退化检查却不能保证全局可迁移边界。
- Evidence：新增Drone→Bird中的Track 4237/4238分别来自相邻capture `20250616103607/20250616103609`，均使用PRTnum=142；其parent事件均值CV/展宽分别为`0.2706/0.3564`和`0.2990/0.4347`，处于validation Drone分布约第99至100百分位。Treatment从首个可验收prefix起对两轨迹始终判Bird，并在最终达到`0.8051/0.9010`置信度。Track 1535同样具有偏高CV/展宽/周期并被以`0.9911`判Bird。另一方面Bird Track 665的极宽谱被正确恢复，说明信号不是纯噪声，而是缺少条件化后同时扩大真阳性和假阳性。代码对PRTnum<192使用32点STFT窗、其余使用64点窗，hop、物理观测时长和周期可估范围随PRT/载频变化；采样元数据又按防捷径设计不进入模型。调制周期仅`26.82%` raw frame可估，事件中不可估与真实低周期共同编码为0且没有feature-specific有效性语义。Treatment新三列最终Fusion列范数约`0.118/0.174/0.177`，低于全部输入列中位数`0.446`，但相同方向的小偏移可被20余事件的液态状态持续累积成高置信动作。
- Impact：这更像输入语义未做到跨采样模式物理同尺度，而不是末期液态状态漂移；直接加dropout、状态正则、class weight或继续放大Bird loss可能压回Recall却不能修复模式条件反转。Bird与Balloon的CV中位数也接近，三维摘要同时扰动Bird/Balloon边界，当前负结果不能外推为“完整STFT张量无效”。
- Action：不在V17继续调权重或阈值。若未来重开，只允许一个预注册变量：优先把STFT窗长/hop、可分辨周期和谱展宽改为物理时间/速度分辨率一致的mode-invariant摘要，并将“周期不可估”用于证据门控或中性插补而非类别输入；先用train-only检查主要共享模式中的效应方向一致性，再做Control/Treatment。完整STFT张量仍属于独立新数据/模型契约，不能以本轮摘要失败直接否定。
- 验证状态：帧映射、目标局部峰、目标/全局功率门禁、采样模式统计和10条切换parent均已只读复核；未改标签、未调阈值、未启动后续训练。
- 关联版本：V17 target-anchored修订协议v3；潜在后续必须递增版本。

## Finding F199

- Severity：High（V17传入链路无主导故障；物理摘要合同是主要瓶颈）
- 文件位置：`src-v3/radar_ln3/micro_doppler.py`、`src-v3/radar_ln3/data.py`、`src-v3/radar_ln3/task_models.py`、V17 readiness/paired报告
- 问题或结论：V17出现Bird Recall上升而Precision下降，不是cache错位、valid掩码、Fusion拼接或offline/online分叉造成；三维提取器把高CV/展宽映射为Bird，但该数值方向受采样模式影响且没有专门保留Drone宽带旋翼侧带。
- Evidence：V17 focused已证明sample/cache对齐、共享初始化、三列零初始化、非零梯度、offline/online首事件误差0和checkpoint round-trip；正式两臂parent顺序一致。事件级方向无关AUC在train/val为CV约`0.673/0.694`、展宽约`0.575/0.608`、period约`0.518/0.549`，新增信息弱且period接近无信息。PRT=142 train三类parent=`4/3/0`，validation为Drone 2 parent、Bird/Balloon 0。
- Impact：继续调class weight、dropout或液态状态正则只能移动边界，不能恢复被提取公式遗漏的侧带物理信息；同时会把PRT=142极端Drone继续推向Bird。
- Action：V17不重训；新版本采用宽带侧带、原始脉冲包络调制、物理可观测性门控和PRT=142零干预，并保持传入链路其余变量冻结。
- 验证状态：代码和历史结果已复核；V18全量prepare/训练未执行。
- 关联版本：V17结论，V18修复。

## Finding F200

- Severity：High（旧Nyquist/周期解释需要物理校正）
- 文件位置：`src-v3/radar_ln3/micro_doppler.py::_stft_linear_power`、V17 evidence cache、`src-v3/radar_ln3/micro_doppler_broadband.py`
- 问题或结论：PRT=142并非简单受325 Hz Nyquist限制；原始脉冲采样调制Nyquist远高于旧STFT hop Nyquist，但其单帧仅约4 ms，统计窗极短且三类支持不足。PRT≥360的旧STFT hop Nyquist约306–326 Hz，400–800 Hz旋翼闪烁确会混叠。Bird 5–15 Hz在最长约46 ms单帧内仍不足一个15 Hz周期，不能声称被完整保留。
- Evidence：V17 raw metadata显示PRT=142单帧时长约`3.976–4.402 ms`，PRT 226约`4.520–11.526 ms`，360/570/900约`17.28–45.90 ms`；旧STFT时间步为segment/2乘PRI。PRT=142的主要问题是7个左右原生STFT时间窗和`4/3/0` parent支持，而非原始脉冲采样本身低于400 Hz。
- Impact：如果按错误Nyquist解释只缩短STFT窗，会提高表面时间带宽但进一步损害频率/速度分辨率；如果继续输出Bird扑翼period，则会制造不可辨识数值。
- Action：V18绕开旧STFT hop，直接对raw pulse envelope提取300–2000 Hz调制；2 kHz上限保证全部候选PRT/PRI帧满足Nyquist。明确不估计5–15 Hz周期，并用每帧速度/调制Nyquist门控物理可观测性。
- 验证状态：focused合成900 Hz调制提取通过；全量分布待prepare。
- 关联版本：V18。

## Finding F201

- Severity：High（目标行自身估噪会吞掉宽带旋翼侧带；已在V18修复）
- 文件位置：`src-v3/radar_ln3/micro_doppler_broadband.py::_neighbor_range_noise_spectrum`
- 问题或结论：简单把V17的±8 m/s扩到±90 m/s仍不充分；若噪声底从目标行30–90 m/s区域自身中位数估计，真正宽带的Drone旋翼侧带会被当成噪声扣除，重复“保Bird削Drone”的物理偏置。
- Evidence：目标行是唯一包含待分类目标的距离行，V18局部MTD已确定其邻域10–20；邻域内除目标行外最多10个距离行可提供频率分辨的独立噪声/杂波样本。最终focused合成相干主体+宽带正负侧带在邻行噪声、循环频差与零速平移不变修复后得到总侧带fraction=`0.239978`并保留900 Hz包络调制。
- Impact：目标行自估噪会使侧带越宽、被扣除越多，系统性压低Drone证据；全量训练即使收敛也可能再次出现Bird边界扩张。
- Action：V18固定使用目标距离邻域非目标行的逐频率中位数除`ln(2)`估计指数噪声均值；噪声估计、主体消除、侧带统计都写入版本化合同，禁止prepare后改公式。
- 验证状态：focused synthetic和真实代表事件通过；全量readiness待执行。
- 关联版本：V18。

## Finding F202

- Severity：High（V18最初宽带39维候选已被冗余审查取代）
- 文件位置：`src-v3/radar_ln3/micro_doppler_broadband.py`、`src-v3/build/micro_doppler_v18.py`、`src-v3/report/micro_doppler_v18_readiness.py`、`src-v3/report/micro_doppler_v18_paired.py`、`src-v3/scripts/v18_broadband_micro_doppler_paired_pipeline.ps1`
- 问题或结论：V18最初把实验变量设计为Control `rd_physical4` versus Treatment `rd_physical4+micro39`，并将信息充分性门槛、epoch排序和最终2个百分点采纳门槛分离；后续冗余审查发现模型摘要包含确定性派生关系、强共线簇和median/q90重复，因此该39维合同只保留为修订前focused证据，不再作为正式prepare入口。
- Evidence：39维由23项帧中位数和16项闪烁敏感q90组成；完整181×/64×审计profile不进入模型。focused Smoke显示Fusion输入`83/122`、40个共享state entry一致、39个新增列为0、梯度最大值`0.982568`、offline/online误差0、Track 1010 46帧映射偏移`+8`保持；合成readiness覆盖12个观测模式、8个校准模式和4个零干预模式。PowerShell AST和Python compile通过。
- Impact：prepare通过只能证明特征可计算、非退化且train-only有最低判别潜力，不能宣称模型提升；只有paired validation同时提高Macro-F1与Drone/Bird弱项floor严格`>0.02`且方向错误下降，才允许采纳。
- Action：不得执行旧39维路径。该Finding当时转向Finding F203的21维`pruned21`候选，但其train-only readiness随后被Finding F204阻断；当前只允许执行F204冻结的8维`stable8`合同，门槛不因再次降维而降低。
- 补充物理边界：V18最终复核发现普通频差会在主体靠近Doppler Nyquist边缘时丢失跨边界侧带；现已改为PRF周期上的有符号循环频差。合成跨边界回归证明正负`30–60 m/s`侧带同时保留，且每个FFT bin仅映射一次。
- 补充调制语义：三个高频频带幅度改为由一侧Parseval带内能量直接换算RMS，调制噪声底改由邻近非目标距离行逐频估计；不再以目标自身宽带中位数估噪，也不再用全包络RMS近似频带幅度。
- 补充平移不变性：删除绝对零速`±2 m/s`截断，避免运动主体的真实微动侧带跨过零Doppler时被误删；合成跨零回归要求正负`30–60 m/s`侧带同时保留。
- 补充模式全集：train共有12个`脉冲数+PRI+载频`组合；8个满足三类支持并可校准，2个PRT=142与2个仅4个Drone且validation缺失的226脉冲模式统一零干预。合同外新模式或任一未校准模式产生非零模型证据都会阻断prepare。
- 验证状态：旧39维focused Smoke通过但已被取代；旧39维全量prepare、正式训练、paired validation、OOF、test、正式拒识、stable、P95均未执行。
- 关联版本：V18。

## Finding F203

- Severity：High（V18模型摘要存在冗余；已修订为21维且保持物理审计信息完整）
- 文件位置：`src-v3/radar_ln3/micro_doppler_broadband.py`、`src-v3/build/micro_doppler_v18.py`、`src-v3/report/micro_doppler_v18_readiness.py`、`src-v3/smoke/micro_doppler_v18_contract_test.py`
- 问题或结论：旧39维由23个逐帧量的event median和其中16项的event q90组成，存在60–90 m/s方向带强共线、entropy/concentration近镜像以及大量median/q90重复，因此曾缩减为18 median + 3 q90，共21维。该`pruned21`候选随后被train-only模式反转门禁阻断，已由Finding F204的`stable8`取代；完整23维逐帧evidence与181点残差Doppler/64点包络调制profile始终保留。
- Evidence：21个模型列精确冻结为正负8–30/30–60、median total、RMS/q90范围、top4 concentration、peak count、侧带SNR、包络RMS、三调制带、centroid、top3、调制熵/SNR，以及q90正负8–30和q90 total。相对旧39维恰好删除18列。逐帧`total=sum(6 directional bands)`的合成误差为0，但三帧反例得到`median(frame total)=1.0`、`sum(median(directional bands))=0.0`，证明不能把帧级线性恒等直接应用到event median；因此median total保留，symmetry删除，q90 total作为瞬态总侧带量保留。调制三带保留能量幅度、centroid保留频率位置，删除可由三带主导的600–2000 fraction。
- Impact：Fusion输入从`83/122`降至`83/104`，减少18个冗余自由度和小样本过拟合风险，同时不损失逐帧物理审计能力。旧39维cache、配置、checkpoint和报告会被独立`pruned21` schema/protocol/path拒绝，不能静默混用。
- Action：`micro-doppler21-pruned`只读保留为中间特征筛选证据，禁止继续训练；当前执行入口见Finding F204的`micro-doppler8-stable`。最终仍要求Macro-F1和Drone/Bird弱项floor严格提升超过2个百分点、双向错误下降及三类P/R/F1均≥0.95，不因降维放宽。
- 验证状态：`focused-contract-v2-pruned21`通过，用户已执行全量prepare，但train-only readiness因6项全局有效特征在采样模式内反转而正确阻断；正式`pruned21`训练、paired validation、OOF、test、正式拒识、stable和P95未执行。
- 关联版本：V18 `pruned21`修订协议v2；V16/V17与旧V18 focused artifact只读保留。

## Finding F204

- Severity：High（V18 pruned21 readiness发现真实采样模式条件反转；已修订为stable8）
- 文件位置：`src-v3/processed/report/RADAR-LN3-V18-broadband-multiscale-micro-doppler21-pruned-train-readiness.json`、`src-v3/radar_ln3/micro_doppler_broadband.py`、`src-v3/build/micro_doppler_v18.py`、`src-v3/report/micro_doppler_v18_readiness.py`
- 问题或结论：`pruned21 prepare`成功，但train-only readiness正确阻断训练。21项中16项达到全局Drone/Bird AUC/effect门槛，只有10项在全部8个校准采样模式中不反转；因此问题不是数据构建失败或门槛过严，而是全局判别方向混合了采样模式条件效应。当前正式候选进一步冻结为8项全局有效、模式零反转且去除强共线项的median摘要。
- Evidence：6个阻断特征为median total、600–1200 Hz调制幅度、调制centroid、top3 concentration、q90正8–30和q90 total。median/q90 total在两个PRT=900模式出现约`+0.46至+0.69`强反转；600–1200 Hz在两个PRT=226模式约`+0.29至+0.32`反转；centroid/top3在PRT=900反转可达约`+0.40至+0.80`。train-only派生探针显示模式稳定的合并带/q90候选与现有安全列Spearman多为`0.92–0.997`，不值得扩维。10个安全项中q90 abs offset与RMS相关约`0.946`、peak count与RMS约`0.901`，最终作为evidence-only移除。
- Impact：如果绕过旧门禁，液态状态会在整条轨迹上累积模式相关的错误方向，重现V17 Drone→Bird高置信错误。stable8把Fusion从`104`降至`91`，保留正8–30、负/正30–60、RMS offset、top4 concentration、sideband SNR、envelope RMS和modulation SNR；完整23维frame evidence仍可支持未来mode-aware研究。
- Action：只允许执行`stable8` v3流水线。新readiness要求8/8项全部非退化、全局AUC≥0.55、绝对effect≥0.10、8模式零反转且至少两个模式绝对effect≥0.10；不再使用“至少4项”宽松门槛。旧blocked report和manifest必须SHA绑定且唯一失败项匹配，随后只读重聚合，不访问raw/RD；任何不一致都fail-closed。
- 验证状态：focused v3通过；实际train-only复算8/8全局有效且8/8模式一致，train/val `18266/4288`事件从旧23维evidence重聚合后与旧summary同名列逐值相等，raw/RD访问0，Fusion `83/91`、新增8列零初始化、梯度最大值`0.930194`、offline/online误差0。正式stable8 prepare、训练、paired validation、OOF、test、拒识、stable和P95未执行。
- 关联版本：V18 `stable8`修订协议v3；V18 pruned21 blocked产物只读保留为特征选择来源。

## Finding F205

- Severity：High（V18 stable8信息有效，但共享Fusion联合训练产生决策耦合错误）
- 文件位置：`src-v3/processed/report/RADAR-LN3-V18-broadband-multiscale-micro-doppler8-stable-paired-validation.json`、`src-v3/processed/report/RADAR-LN3-V18-broadband-multiscale-micro-doppler8-stable-error-attribution.json`、`src-v3/report/micro_doppler_v18_error_attribution.py`
- 问题或结论：V18 Treatment own-best Macro-F1和Drone/Bird弱项分别提升`0.021549/0.020936`，证明stable8含有真实DBB信息；但Bird Precision仅`0.913793`、Drone→Bird增加1、双向总错未下降，且same-epoch最大正式锚点Macro-F1只提升`0.018576`，未通过冻结采纳合同。8个指定错误中仅2083和3960可解释为当前stable8下的类别重叠/异常流型，其余6个的特征时序明显支持真实类。
- Evidence：1908/2103的stable8逐事件最近中心分别有`22/23`和`23/24`指向Drone，正确近邻均为`14/15 Drone`，模型却全程高置信Bird；2589为`15/16 Bird`，3027为`14/17 Balloon`，531/731均为所有事件Bird。531在Control正确Bird概率`0.861274`，Treatment加入stable8后反向变为Drone概率`0.522541`；这不能由特征不可分、缺失或单一PRT模式解释。
- Impact：继续增加特征、直接放宽2个百分点门槛或在validation调Bird阈值都会掩盖真正瓶颈。当前共享Fusion允许微多普勒改变TR/RD液态主干参数和三类边界，使正确的物理证据被错误耦合；V18不得进入多seed、级联、开放集或test结论。
- Action：保留stable8提取器与完整23维证据，下一版本只做决策耦合单因素修复：V18 Control best主干冻结，stable8经train-only归一化和因果累计均值进入零初始化、有界三类零和late-residual；只训练残差权重，Control初始/最终主干hash必须不变。采纳门槛继续使用三类P/R/F1≥0.95、Macro-F1和弱项严格`>0.02`、双向错误下降及same-epoch/固定基线归因护栏。
- 验证状态：V18 stable8 prepare、seed42两臂20 epoch、paired validation和8-parent归因已执行；归因模块编译与静态隔离通过。完整级联、OOF、test、正式拒识、stable、P95和部署未执行。
- 关联版本：V18 frozen result；V19 late-residual候选方向。

## Finding F206

- Severity：High（V19必须证明stable8收益来自受控后置修正，而非再次重训主干或validation调参）
- 文件位置：`src-v3/radar_ln3/micro_doppler_residual.py`、`src-v3/build/micro_doppler_residual_v19.py`、`src-v3/train/micro_doppler_residual_v19.py`、`src-v3/report/micro_doppler_residual_v19.py`、`src-v3/smoke/micro_doppler_residual_v19_contract_test.py`、`src-v3/scripts/v19_micro_doppler_residual_pipeline.ps1`
- 问题或结论：V18已证明stable8含有可用DBB信息，但共享Fusion联合训练会把部分正确物理证据耦合成错误边界。V19因此不再把stable8输入液态主干，而以冻结V18 Control logits加24参数后置残差进行最小干预；该结构在初始化时与Control完全相同，任何性能变化都只能来自stable8到三类logit修正的受限映射。
- Evidence：focused真实probe使用Drone/Bird/Balloon各1个train parent，残差权重梯度最大绝对值=`0.167272`且非零；一次AdamW step前后冻结Control state SHA均为`5d61933484a0f01dac629a5b8168b92b6733be304897a3dfac72e76e54780b7e`，主干无gradient。20/20 parent顺序hash与V18 Control正式训练完全一致；合成未来事件扰动不改变先前prefix，残差绝对/类间上界为约`3/6`，临时端到端20 epoch、20 checkpoint、best复载和独立report均通过。
- Impact：V19能将“特征无效”和“耦合错误”分开检验，并显著降低正式训练计算量；但冻结logit cache是局部研究合同，不代表线上主干已接入该残差，也不能用于完整五类级联、拒识或stable性能宣称。
- Action：由用户手动执行V19 prepare、train和evaluate。只有三类P/R/F1≥0.95、Macro-F1与Drone/Bird弱项严格`>0.02`、双向错误下降时才讨论多seed；若失败，不增加残差层数/容量、不回看validation调阈值，接受当前stable8摘要的局部收益上限并转向新的预注册变量。
- 验证状态：代码编译、静态隔离、PowerShell AST、synthetic端到端和真实三parent focused Smoke已通过；全量prepare、正式训练、validation性能、完整级联、OOF、test、正式拒识、stable、P95和部署未执行。
- 关联版本：V19 frozen-Control stable8 late-residual候选；V18 Control best与stable8证据只读复用。

## Finding F207

- Severity：Medium（V20实验代码与合同已通过准备阶段，正式性能尚未产生）
- 文件位置：`src-v3/radar_ln3/model.py`、`src-v3/radar_ln3/task_models.py`、`src-v3/radar_ln3/streaming.py`、`src-v3/report/liquid_dynamics_v20_readiness.py`、`src-v3/scripts/v20_zero_delta_skip_pipeline.ps1`
- 问题或结论：V20 skip已完成正式训练并失败收口。Control使用现行`bounded_observation`固定gain=`0.25`；Treatment只对`zero_delta_at_second_precision`使用`alpha=0`。first event和positive delta不变；由于分类头只读取liquid state，当前事件虽被Fusion计算但在alpha=0时不进入state/logits，因此真实语义是“丢弃同秒当前观测”，不是“观测可见但不累积”，也不是状态张量恒等不变。
- Evidence：V20配置逐字节复用V16数据字段和contract哈希；train/val DBB parent=`953/223`；zero-delta事件统计排除first event，train/val event级标签Cramér's V=`0.1264218055/0.0916738183`，仅写入readiness诊断块，不参与训练门禁；test payload access=`0`。focused Smoke确认skip对同一状态的不同zero-delta事件内容不敏感、density ordinal按`first/positive→1、zero→k+1`递增且offline/online最大误差=`0`、配对初始logits逐值一致。
- Impact：own-best总体Accuracy不变、Macro-F1仅`+0.004047`，但Bird Precision/F1下降且Drone↔Bird错误由7增至11；聚合指标掩盖了安全方向恶化。该结果不能证明固定0.25可弃，也不能外推到五类开放集、Gate/CU路由或stable动作。
- Action：V20 skip不采纳，保留bounded Control；V20r作为独立正向实验启动，不把skip当作严格对称反向证据。不得使用test、OOF、正式拒识、stable或P95调参/宣称收益。
- 验证状态：V20 prepare、两臂正式20 epoch、paired validation均完成；状态`completed_but_gate_failed`。V20r prepare完成但正式Treatment训练未执行；完整级联、OOF、test、正式拒识、stable、P95和部署未执行。
- 关联版本：V20-skip / V20r-density；V16数据contract只读复用。

## Finding F208

- Severity：High（V20r密度干预存在真实可训练路径，同时具有类别相关暴露风险）
- 文件位置：`src-v3/report/liquid_dynamics_v20_readiness.py`、`src-v3/smoke/liquid_dynamics_v20_contract_test.py`、`src-v3/scripts/v20r_density_weighted_pipeline.ps1`
- 问题或结论：V20r对zero-delta事件按同秒因果序使用k2=`0.50`、k≥3=`0.75`，并复用V20 skip Control而不重复训练。大多数受干预事件已经进入k≥3饱和区，因此这是强正向干预，不是围绕0.25的小扰动；其收益必须同时满足Bird/Drone方向安全门槛。
- Evidence：train事件`k1/k2/k≥3=11733/1677/4856`，受干预zero事件共6533，其中约74.3%位于k≥3。Bird/Drone/Balloon parent均衡暴露率=`0.441576/0.357492/0.243624`，density ordinal标签Cramér's V=`0.091077`。Smoke确认k增益1/2/3倍与0.75 cap、padding/online一致；真实Track 1010暴露前logits逐值相同、暴露后不同，loss和梯度finite且非零。
- Impact：density确实能改变液态状态并反向传播，但Bird在train中接受Treatment的比例最高，可能将同秒观测密度变成类别相关捷径；若只提升Bird Recall并降低Precision，或高k parent主导收益，不得采纳。
- Action：只训练V20r Treatment并与SHA冻结的V20 Control比较；报告own-best、same-epoch、三类P/R/F1、Drone↔Bird方向错误和按density exposure归因。严格保留`>0.02`与方向安全门槛，不因V20 skip失败放宽。
- 验证状态：配置、readiness、focused Smoke、Python compile、PowerShell AST和静态隔离通过；正式20 epoch Treatment和paired validation待用户执行。test、OOF、级联、拒识、stable、P95和部署未执行。
- 关联版本：V20r-density，复用V20-skip Control与V16数据contract。

## Finding F209

- Severity：High（产品层级和Unknown语义已迁移，旧DBB/CU解释不得继续沿用）
- 文件位置：`docs/radar_ln3_v21_strict_open_set_architecture.md`、`src-v3/radar_ln3/open_set_architecture.py`、`src-v3/radar_ln3/known_gate.py`、`src-v3/radar_ln3/known_gate_features.py`、`src-v3/build/known_gate_v21.py`、`src-v3/train/known_gate_v21.py`、`src-v3/report/known_gate_v21.py`
- 问题或结论：V21当前架构为`known_gate -> dbb_specialist -> semantic_decision_layer`。KnownGate仅学习已知目标与已知Clutter，DBB specialist仅学习Drone/Bird/Balloon，L3仅执行评分、质量、OOF阈值和互斥动作规则。UnknownTarget不再是普通闭集输出；`UNKNOWN_REJECTED`与`PENDING_ABSTAIN`具有不同语义和分母。历史`dbb_cu_gate/cu_specialist`只能复现旧artifact。
- Evidence：V21映射只含Drone/Bird/Balloon→`KNOWN_TARGET_EVIDENCE`和Clutter→`KNOWN_CLUTTER_EVIDENCE`，Unknown被显式列入forbidden training labels；known-only normalization合同固定986个train parent并排除69个Unknown，validation局部任务固定223个目标与9个Clutter并排除15个Unknown。focused Smoke确认Unknown不在模型映射、新增Fusion列零初始化且可训练、Treatment与Baseline初始logits一致、全部参数参与训练、offline/online一致。动作解析覆盖目标+子类接受、Clutter接受、双Known不接受、子类不足、质量不足和路由冲突。
- Impact：V21局部KnownGate高分不能直接解释为Unknown拒识、完整五类性能或部署收益；二类softmax/argmax只作特征实验诊断，真正的“双方均不可接受”必须等train parent-grouped OOF评分阈值和L3规则冻结后评估。Clutter validation只有9个parent，单点指标方差大，必须携带support和parent bootstrap。
- Action：用户先串行完成V21 Baseline及V21a-d五个独立20 epoch全参数运行。各Treatment只改一组证据特征，Macro-F1相对Baseline严格提升`>0.02`且预期方向净改善时，才晋级multi-seed/OOF；不在本阶段组合特征、调Unknown阈值或运行test。后续L3校准artifact必须版本化绑定KnownGate、DBB checkpoint、评分器、质量规则和OOF来源。
- 验证状态：架构文档、代码、训练/报告入口、focused Smoke、真实CQ08小样本probe、编译、静态隔离和PowerShell AST已通过；正式prepare、五个训练、冻结validation、OOF、Unknown拒识、完整级联、test、stable、P95和部署未执行。
- 关联版本：RADAR-LN3-V21/V21a/V21b/V21c/V21d KnownGate局部单因素阶段；V18r继续作为DBB局部候选。

## Finding F210

- Severity：Medium（V21 prepare阻断，已修复且未改变样本或标签）
- 文件位置：`src-v3/build/known_gate_v21.py`、`src-v3/radar_ln3/known_gate_features.py`、`src-v3/smoke/known_gate_v21_contract_test.py`
- 问题或结论：V21构建器把所有TR radial/range字段无条件转换为float，导致合法缺失`range_m=null`的`cq08|track|234#point0012`阻断全量prepare。该字段只被V21d使用，不应使谱形、模态或iso证据失效。
- Evidence：KnownGate train/val RD-present事件只读审计得到train完整18769、range缺失1、radial缺失0；validation完整4397、两字段缺失均0。受影响事件仍有有效radial speed和8帧RD。修复后真实parent探针输出六维valid=`[true,true,true,true,true,false]`，只有d_range值为0。
- Impact：旧实现不是性能问题，而是缺失语义耦合错误；若通过删除样本或插值修复，会污染固定parent人口和物理证据。逐特征valid修复保持V21a/b/c输入与原始事件一致，并只降低V21d该单点的有效support。
- Action：重新执行V21 `prepare`。构建器将复用内容一致的normalization，重新完成train/val cache并在readiness中记录train radial/range缺失=`0/1`、validation=`0/0`。若计数变化则阻断，不静默接收新缺失模式。
- 验证状态：合成缺失range/radial回归、真实Track 234单parent构建、V21脚本Smoke、Python编译、PowerShell AST、静态隔离和diff检查已通过；正式全量prepare、五个20 epoch训练和validation评估未执行。
- 关联版本：RADAR-LN3-V21 KnownGate特征合同修订，输入维度和实验变量不变。

## Finding F211

- Severity：High（V21 Baseline总体Accuracy掩盖Clutter低support边界，跨seed必须类别均衡选优）
- 文件位置：`artifacts/RADAR-LN3-V21-cq08-liquid-known-gate-baseline-seed42-formal/training-summary.json`、`src-v3/build/known_gate_v21_multiseed.py`、`src-v3/report/known_gate_v21_multiseed.py`、`src-v3/scripts/v21_known_gate_baseline_multiseed_pipeline.ps1`
- 问题或结论：seed42 Baseline Accuracy达到`0.978448`，但Clutter Recall只有`6/9=0.666667`且Macro-F1为`0.847348`，不能把Accuracy最高的seed定义为最优KnownGate。三seed候选采用最弱类Precision/Recall优先的development排序，并以parent bootstrap量化Clutter n=9造成的不稳定性。
- Evidence：seed42 Known Target support=223且P/R约0.99，Clutter support=9且P/R/F1=`0.75/0.666667/0.705882`，best selection weak floor=`0.666667`。新增seed43/44配置经逐字段审计只改变seed/DataLoader seed和artifact身份；Smoke确认27482个参数全部训练、同seed初始化可复现、不同seed初始化不同，排序先看weak floor再看Macro-F1，50次合成paired bootstrap和完整selection artifact探针通过。
- Impact：选择validation最优seed会引入development selection bias，bootstrap胜出频率也不能将其变成无偏test估计；但它可用于确定后续局部实验的受控初始化。若选中43/44而Treatment仍使用seed42，所谓特征收益将混入初始化差异，单因素归因失效。
- Action：用户只需补训seed43/44并运行冻结evaluate。选中模型仅标记为development champion；后续V21a-d必须与champion同seed配对，或三seed全部运行并比较逐seedTreatment−Baseline增量。不得据此执行Unknown阈值、test或部署放行。
- 验证状态：multiseed prepare、配置/readiness、训练入口只读校验、Python编译、PowerShell AST、原V21回归Smoke、新multiseed Smoke、合成bootstrap/report和静态隔离已通过；seed43/44正式训练及正式选优未执行。
- 关联版本：RADAR-LN3-V21 KnownGate Baseline multiseed development selection。

## Finding F212

- Severity：High（seed43按冻结规则胜出，但优势未与seed42稳定分离）
- 文件位置：`src-v3/processed/report/RADAR-LN3-V21-known-gate-baseline-multiseed-validation.json`、`data/processed/RADAR-LN3-V21-known-gate-baseline-development-selection.json`、三个V21 Baseline正式artifact
- 问题或结论：三seed排序为`43 > 42 > 44`，正式development champion为seed43。其Clutter Recall由seed42的`6/9`提高到`7/9`，但Known目标误拒由2增至3；Accuracy保持`0.978448`，Macro-F1提高到`0.862803`。由于Clutter只有9个parent，seed43只是点估计规则下更优，不是稳定显著优于seed42。
- Evidence：seed43 Known Target P/R/F1=`0.990991/0.986547/0.988764`，Clutter=`0.70/0.777778/0.736842`，weak floor=`0.70`。5000次bootstrap胜出频率seed43/42/44=`56.04%/43.96%/0%`；seed43−seed42 Macro-F1、weak floor和Clutter F1的95%区间分别为`[-0.072465,0.129464]`、`[-0.194498,0.333333]`、`[-0.138158,0.25]`，均跨0。三seed共同错误包括Bird 3580和Clutter 2900/3761；seed43在修复Bird 2619与Clutter 3605的同时新增Balloon 3027与Bird 852错误。
- Impact：选中seed43有利于按既定类别均衡规则继续实验，但不能声称随机种子解决了Clutter边界，也不能把selection后的0.8628当作无偏泛化性能。直接拿seed42 V21a-d与seed43 Baseline比较会把初始化差异混入特征效应。
- Action：后续V21a-d至少生成seed43同seed配对Treatment；更严谨方案是四个Treatment都跑42/43/44并比较逐seedpaired delta。任何单因素正向信号仍需Macro-F1严格`>0.02`、预期类别净改善和bootstrap，并最终由train parent-grouped OOF确认。Clutter覆盖不足仍是KnownGate主要瓶颈。
- 验证状态：三seed正式20 epoch、5000次bootstrap、development selection及其SHA绑定已执行并由流水线重复验证；OOF、Unknown拒识、完整级联、test、stable、P95和部署未执行。
- 关联版本：RADAR-LN3-V21 KnownGate Baseline seed43 development champion。

## Finding F213

- Severity：Critical（旧V21a实现的物理量与原始诊断不一致，且配对种子错误）
- 文件位置：`src-v3/radar_ln3/known_gate_features.py`、`src-v3/build/known_gate_v21a.py`、`src-v3/train/known_gate_v21.py`、`docs/radar_ln3_v21a_interframe_spectral_experiment.md`
- 问题或结论：原始探索将谱形时间一致性定义为canonical RD相邻帧的速度谱余弦、峰速度漂移和零速能量变化；旧V21a却提取raw单帧内部相邻STFT窗affinity，并固定配对seed42。它既不是同一物理特征，也不能与seed43 champion做单因素归因。
- Evidence：原脚本`tmp/v3_probe_archive/tmp_spectral_consistency.py`明确写“帧间谱相似/漂移”并输出6项；旧实现只输出2项intra-frame affinity。旧minimum还与事件帧数Spearman约`-0.120`，同帧数后Clutter−Bird差明显收缩。修正版六维合成probe在相同谱帧输出cosine=`1/1`、drift=`0/0`，突变帧输出cosine约`4.2e-7`、drift mean/max=`0.392/0.522`；原V21、multiseed及新V21a Smoke均通过。
- Impact：直接训练旧配置可能把采样窗/帧数差异误当谱形稳定性，并把seed差异混入特征收益。旧结果即使变好也不能回答用户提出的V21a假设。
- Action：旧`v21a`训练入口fail-closed；正式实验改为`v21a_interframe`，以seed43同初态全参数20 epoch运行，六维证据按当前事件因果归属并在RD缺失处reset。报告own-best、same-epoch、类别净救回和5000次parent bootstrap。
- 验证状态：代码、Focused Smoke、旧V21回归、真实四parent只读probe、编译和PowerShell AST已通过；正式prepare、训练和validation未执行。
- 关联版本：RADAR-LN3-V21a inter-frame spectral consistency修订。

## Finding F214

- Severity：High（V16 test对V21a特征构思已暴露，不能再作为无偏最终holdout）
- 文件位置：`tmp/v3_probe_archive/tmp_spectral_consistency.py`、`src-v3/build/known_gate_v21a.py`、`docs/radar_ln3_v21a_interframe_spectral_experiment.md`
- 问题或结论：最初用于提出V21a方向的探索脚本从全V16 index按标签抽track，没有限定train，并在每类调用中独立重置`RandomState(3)`。逐字只读重建后，108个探索track中train/val/test为`77/17/14`。
- Evidence：暴露包含Clutter/Drone/Bird/Balloon test track=`6/1/3/4`。当前修正版prepare只读取train/val canonical payload，test payload访问仍为0；问题是历史特征构思已经观察过test来源，而不是当前训练泄漏。
- Impact：未来即使V21a在原V16 test表现良好，也不是严格盲测泛化估计，不能据此放行KnownGate或Unknown拒识。
- Action：V21a只做development配对和train parent-grouped OOF；多seed后仍需新的独立holdout或新数据版本。readiness/report强制披露该边界，禁止删除或淡化。
- 验证状态：metadata重建已执行；原V16 test模型评估、正式拒识、级联、stable和P95均未执行。
- 关联版本：RADAR-LN3-V21a holdout contamination disclosure。

## Finding F215

- Severity：High（V21a正式配对没有产生任何parent-final分类收益）
- 文件位置：`src-v3/processed/report/RADAR-LN3-V21a-known-gate-interframe-spectral-paired-validation.json`、V21 seed43 Baseline与V21a正式artifact
- 问题或结论：V21a帧间谱形一致性已完成seed43同初态、同parent顺序、全参数20 epoch正式配对，但own-best结果与Baseline严格相同，不能晋级multi-seed或train parent-grouped OOF。
- Evidence：两臂best epoch均为16，parent-final混淆矩阵均为`[[220,3],[2,7]]`，Macro-F1均为`0.862803`，Clutter Precision/Recall/F1均为`0.70/0.777778/0.736842`。232个validation parent硬预测逐个相同，Bird错误为3580/852、Balloon错误为3027、Clutter错误为2900/3761；5000次paired parent bootstrap的全部硬指标差值区间均为`[0,0]`。20个same-epoch中Treatment仅epoch 14优于Control，最大Macro-F1增量`0.013684`，未超过2pp，另有5个epoch更差。
- Impact：新增六维输入确实获得非零权重并改变概率，但未改变任何own-best parent动作；其信息大部分可能已被现有RD编码器/液态时序吸收，且两条困难Clutter虽概率略向正确方向移动仍未跨边界。不能用train-only AUC或单个epoch的Clutter Recall=100%替代冻结own-best结论。
- Action：V21a正式裁决为无正向信号，不组合进KnownGate、不执行多seed/OOF、不调阈值。继续按预注册顺序独立验证V21b/c/d；若未来研究谱一致性，必须作为新变量重新预注册，不能回看当前validation选子特征。
- 验证状态：prepare、20 epoch正式训练、development paired report、5000次bootstrap已执行；test、OOF、完整级联、正式拒识、stable、P95和部署未执行。
- 关联版本：RADAR-LN3-V21a inter-frame spectral consistency正式负结果。

## Finding F216

- Severity：High（V21b正式配对无动作增益，且历史高AUC探针与正式特征不等价）
- 文件位置：`src-v3/build/known_gate_v21b.py`、`src-v3/report/known_gate_v21b.py`、`src-v3/smoke/known_gate_v21b_contract_test.py`、`src-v3/scripts/v21b_modal_consistency_pipeline.ps1`、`docs/radar_ln3_v21b_modal_consistency_experiment.md`
- 问题或结论：V21b正式变量保持为当前事件canonical RD零偏移目标行逐帧能量峰速度与当前TR有符号径向速度绝对偏差的median/max，唯一主假设是Clutter/Drone。历史探索AUC并非该正式公式的可靠先验：脚本跨split抽样、稀疏抽帧，并错误地取各距离行argmax Doppler索引的最大值。旧seed42和旧seed43广义方向候选均已阻断。
- Evidence：历史脚本抽样93个track，train/val/test=`68/10/15`，实际产出92个parent；其Clutter/Drone direction-free AUC=`0.809/0.765`。正式train-only cache的Drone/Bird/Balloon/Clutter parent support=`476/239/238/33`，Clutter−Drone均值差=`-0.017145/-0.019844`，但direction-free AUC仅`0.567577/0.509008`；事件级、parent median/final/max审计最高也仅约`0.5881`。prepare仍直接SHA复用V21核心cache，当前路径raw/test访问0。
- Impact：正式Treatment与Baseline均在epoch 16选优，parent-final混淆均为`[[220,3],[2,7]]`，Macro-F1均为`0.862803`；Drone/Bird/Balloon/Clutter四类232个parent全部`changed=rescued=harmed=0`。逐事件对齐进一步确认4452个window与3756个eligible prefix也无一改变硬预测，不只是汇总混淆碰巧相同；5000次paired bootstrap全部硬指标差值区间为`[0,0]`。因此当前公式未改善Clutter/Drone动作边界。历史15个test track参与特征构思，原V16 test也不再是该特征的纯盲holdout。
- Action：V21b不进入multi-seed、train parent-grouped OOF或组合特征。不能用epoch 14的Macro-F1临时增量`+0.043613`宣称主假设成立：该差异来自少误拒3个Bird，Clutter/Drone不变；epoch 15虽多救回1个Clutter，却新增1个Bird误拒。若未来重做，必须使用新的物理量/公式与独立实验身份，不得回看当前validation选择聚合或符号。
- 验证状态：正式全参数20 epoch、冻结development paired report、四标签净救回、20个same-epoch和5000次bootstrap已执行。新增两列96/96权重均非零，own-best新列L2=`0.500995`，证明负结果不是传入或训练失效。事后软评分诊断中Clutter/Drone NLL差约`-0.00190`、Brier差约`-0.000592`，但paired bootstrap 95%区间分别为`[-0.00452,+0.00121]`和`[-0.00149,+0.000452]`，均跨0；总体NLL/Brier反而恶化。该诊断不改变冻结裁决；test、OOF、完整级联、正式拒识、stable、P95和部署未执行。
- 关联版本：RADAR-LN3-V21b seed43 RD/TR modal consistency Clutter/Drone正式候选v2。

## Finding F217

- Severity：High（V21c具有train-only方向信号，但历史数值与正式特征不等价）
- 文件位置：`src-v3/build/known_gate_v21c.py`、`src-v3/report/known_gate_v21c.py`、`src-v3/smoke/known_gate_v21c_contract_test.py`、`src-v3/scripts/v21c_target_row_iso_pipeline.ps1`、`docs/radar_ln3_v21c_target_row_iso_experiment.md`
- 问题或结论：正式V21c定义为当前事件canonical RD零偏移目标行相对线性功率占完整`21×384`功率的逐帧比例均值。该量在冻结train parent上确有Clutter/Drone方向信号，但早期`Drone≈0.125/Clutter≈0.053`来自另一提取口径，不能作为正式数值或性能证据。
- Evidence：正式train-only Drone/Clutter parent平均iso均值=`0.281369/0.130682`，中位=`0.277772/0.070544`，Clutter-vs-Drone direction-free AUC=`0.777120`；Clutter-vs-Bird仅`0.518321`，Balloon-vs-Clutter约`0.707156`。focused物理探针得到集中目标行iso=`0.9999998`、21行均匀能量iso=`0.04761905`。早期脚本93个track跨train/val/test=`60/13/20`，使用原生`31×PRT`绝对幅度、首次能量峰选行后跨事件固定和最多10个事件，明显不等价于正式canonical事件特征。
- Impact：V21c值得完成正式单因素训练，且唯一主假设应为Clutter/Drone；但train-only AUC不能证明对已训练主干有互补增益。原V16 test已有20个track参与特征构思，未来不能作为该概念的纯盲最终holdout。Bird与Balloon可能因边界移动受损，必须作为安全守卫。
- Action：固定seed43同初态全参数20 epoch；只新增一列Fusion并从0开始，要求逐epochparent顺序与Baseline一致。采纳必须满足own-best Macro-F1严格`>+0.02`、Clutter+Drone合计净救回为正且各不受损、Bird/Balloon不受损、Known Target与Clutter Recall不下降，并报告5000次paired parent bootstrap。概率微动、中间epoch或“辅助特征”定位不得降低门槛。
- 验证状态：专属prepare、config/contract/readiness、Python compile、PowerShell AST、focused Smoke、静态隔离及V21/V21 multiseed/V21a/V21b回归Smoke已通过；正式20 epoch训练和冻结development paired evaluation未执行。multi-seed、OOF、正式拒识、完整级联、test、stable、P95和部署未执行。
- 关联版本：RADAR-LN3-V21c seed43 target-row iso Clutter/Drone正式候选v1。

## Finding F218

- Severity：High（V21c正式配对没有提供可采纳的互补收益）
- 文件位置：`src-v3/processed/report/RADAR-LN3-V21c-known-gate-target-row-iso-clutter-drone-paired-validation.json`、V21 seed43 Baseline与V21c正式artifact
- 问题或结论：V21c已完成同seed、同初态、同parent顺序的全参数20 epoch配对训练。目标行iso被正确传入并获得非零权重，但冻结own-best没有改变任何parent、window或eligible prefix硬预测，且概率评分轻微恶化；因此train-only单变量AUC没有转化为现有KnownGate主干之外的残差信息。
- Evidence：两臂best epoch均为16，parent Accuracy/Macro-F1均为`0.978448/0.862803`，Known Target P/R/F1均为`0.990991/0.986547/0.988764`，Clutter P/R/F1均为`0.70/0.777778/0.736842`，support=9，混淆矩阵均为`[[220,3],[2,7]]`。新增Fusion列48/48权重非零、L2=`0.203988`；parent/window/eligible-prefix的改变数分别为`0/0/0`，5000次paired bootstrap全部硬指标差值为`[0,0,0]`。总体NLL/Brier变化为`+0.000636/+0.000197`，Clutter/Drone子集为`+0.000125/+0.0000135`。
- Impact：5个own-best错误仍为Bird 852/3580、Balloon 3027和Clutter 2900/3761。困难Clutter 2900/3761的parent平均iso为`0.072712/0.057788`，误拒Bird 852/3580为`0.062829/0.057842`，在真正错误边界上高度重叠；Balloon 3027虽iso=`0.241074`更像已知目标，Treatment却把其Clutter概率再提高约`0.011041`。这说明全局AUC主要描述总体分布，而不是这些残余难例的可修正性；现有RD主干吸收同源浓度信息也是合理但尚未被独立消融证明的解释。
- Action：V21c冻结为负结果，不进入multi-seed、train parent-grouped OOF、组合特征或阈值调优；继续执行预注册的V21d单因素。不得用epoch 15的事后`+0.058786`替代own-best，因为其绝对Macro-F1仅`0.802149`，且后续epoch出现反向变化。
- 验证状态：正式全参数20 epoch、冻结development paired evaluation、5000次bootstrap、概率归因及错误parent核对已执行；报告SHA-256=`e9ca50adf0427b82930d75eca873b3db8e181227371fabc7955c90a41b966348`。Clutter validation support仅9；multi-seed、OOF、正式拒识、完整级联、test、stable、P95和部署未执行。
- 关联版本：RADAR-LN3-V21c seed43 target-row iso Clutter/Drone正式负结果。

## Finding F219

- Severity：Critical（旧V21d公式不能表达历史d_range语义，且历史12.4/1.9不是类别统计）
- 文件位置：`src-v3/radar_ln3/known_gate_features.py`、`src-v3/build/known_gate_v21d.py`、`tmp/v3_probe_archive/tmp_mat_feature_mining.m`、`docs/radar_ln3_v21d_causal_prefix_d_range_experiment.md`
- 问题或结论：旧V21d在每个事件内计算`std(MTD detected range - 一个常数TR range)`，其标准差严格等于事件内MTD range标准差，丢失“相对航迹滤波距离跨事件不稳定”的物理量；记录中的Bird 12.4 m与Clutter 1.9 m则来自两个代表轨迹前9帧，而非类别总体。
- Evidence：逐字复现MATLAB脚本得到Bird track 333=`12.414398 m`、Clutter track 1868=`1.925407 m`；二者分别属于V16 val/test。旧核心第6列train事件均值Bird/Clutter约`5.433/6.510 m`，方向已经相反。正式`v21d_prefix`按每帧`detected_range_m-该帧事件TR range_m`累计样本std后，train-final截断后均值Bird/Clutter=`10.888977/14.007493 m`，Clutter/Bird方向无关AUC=`0.648282`，仍不支持“Bird总体更高”。
- Impact：若直接训练旧seed42配置，会同时混入跨seed差异、错误公式和由两个val/test代表轨迹产生的先验偏差。即使模型提升，也无法归因到用户提出的d_range机制；原V16 val/test也不再是该特征概念的纯盲holdout。
- Action：旧`v21d`训练入口fail-closed；正式实验使用独立`v21d_prefix`、seed43同初态全参数20 epoch和独立cache。主假设仍冻结为Clutter/Bird，但不预设高值属于哪类；采用必须满足Macro-F1严格`>+0.02`、两主标签净救回与Drone/Balloon安全守卫。
- 验证状态：正式prepare、train-only诊断、合同Smoke、旧V21/V21b/V21c回归Smoke、编译、PowerShell AST和静态隔离通过；正式训练/paired evaluation未执行。自动prepare test payload访问0，但人工字段复核读取了已暴露test parent 1868的54个MAT帧；模型test评估、OOF、拒识、级联、stable与P95未执行。
- 关联版本：RADAR-LN3-V21d causal-prefix d-range Clutter/Bird正式候选v1。

## Finding F220

- Severity：High（d_range源数据存在单parent极端跳变，已截断但不可删除）
- 文件位置：`data/processed/RADAR-LN3-V21d-cq08-known-gate-train-causal-prefix-d-range-cache.npz`、V16 track 234 index与对应MAT debug字段
- 问题或结论：train Drone track 234的TR/debug track range在后段从约2750 m跳到566/996/553/1272/10330 m，而MTD detected range平滑落在870至720 m，导致累计残差std最大约3565 m。该异常是源MAT与V16事件共同携带的真实字段，不是V21d映射错位。
- Evidence：逐MAT读取0001/0017/0023/0029/0037/0043帧确认`debug_info.track_range_m`分别与V16事件range完全相同；正式train cache只有该parent共6个prefix超过75 m并被截断为1，validation截断数0、最大原始std约59.205 m。
- Impact：不截断会让单一Drone parent主导数值尺度；静默删除或插值又会改变困难样本与数据合同。75 m截断把影响限制在6/18825个有效train prefix，但该parent仍可能提供类别捷径，正式评估需检查其是否影响Drone安全方向。
- Action：保留track 234并在readiness中单列`clipped_parent_ids`、原始最大值和截断事件数；训练后按Drone/Clutter/Bird净救回与概率变化审计，不根据validation结果修改clip或删除样本。
- 验证状态：源MAT字段抽查、cache极值审计及train/val截断计数已执行；V21d正式20 epoch训练与冻结配对评估已完成，Drone validation 113个parent全部仍被Known Target接受，未观察到该异常parent导致的validation Drone安全回归。test、OOF、完整级联、正式拒识、stable与P95未执行。
- 关联版本：RADAR-LN3-V21d causal-prefix d-range异常值审计。

## Finding F221

- Severity：High（V21d软评分略有改善，但未产生任何冻结动作收益）
- 文件位置：`src-v3/processed/report/RADAR-LN3-V21d-known-gate-causal-prefix-d-range-clutter-bird-paired-validation.json`、V21 seed43 Baseline与V21d正式artifact、`data/processed/RADAR-LN3-V21d-cq08-known-gate-val-causal-prefix-d-range-cache.npz`
- 问题或结论：V21d因果前缀d_range已完成seed43同初态、同parent顺序、全参数20 epoch正式配对。新增证据确实进入模型并改善平均NLL/Brier，但own-best对parent、全部window和eligible prefix均没有改变任何硬预测，不满足预注册的2个百分点与净救回门槛，不能晋级multi-seed、OOF或组合特征。
- Evidence：20/20 epoch checkpoint齐全，Treatment共27530个参数全部训练；共享参数/缓冲初态完全相同，新增Fusion列初始为0，初始logits相同，20个epoch的parent顺序哈希逐轮与Baseline一致。两臂best epoch均为16，parent Accuracy/Macro-F1均为`0.978448/0.862803`，Known Target P/R/F1均为`0.990991/0.986547/0.988764`，Clutter P/R/F1均为`0.70/0.777778/0.736842`，support=9，混淆矩阵均为`[[220,3],[2,7]]`。parent/window/eligible-prefix支持为`232/4452/3756`且改变数全为0；四标签rescued/harmed全为0，5000次分层paired bootstrap全部硬指标差值为`[0,0,0]`。新增列48/48权重非零，L1/L2=`1.279879/0.206340`，排除了未接线或未训练。
- Impact：软评分诊断总体NLL/Brier变化为`-0.004038/-0.000727`，Clutter/Bird子集为`-0.011716/-0.002176`，但这是事后校准信号而不是产品动作收益。5个错误parent仍为Clutter 2900/3761、Bird 852/3580和Balloon 3027；其final d_range约为`16.612/18.668/5.545/13.323/3.411 m`。Treatment相对Baseline时Clutter 3761的正确Clutter概率提高`+0.079573`，Clutter 2900却下降`-0.052853`；两只Bird的错误Clutter概率又提高`+0.030641/+0.009512`。同一Treatment内将d_range输入后验置零，232个parent动作仍不变，NLL由`0.049848`回退到`0.052550`，说明输入本身贡献部分软评分改善；但直接输入效应也同时抬高Clutter 2900/3761和误拒Bird 852/3580、Balloon 3027的Clutter概率，无法作为单调L3评分规则。该一维量在真正错误边界上方向不稳定，validation Clutter/Bird方向无关AUC也仅约`0.657`。
- Action：冻结V21d为无动作增益，不并入KnownGate，不因软评分改善而单独调validation阈值，也不事后选择epoch 9的`+0.041545`临时增量。V21a-d四项seed43单因素均未改变own-best硬预测，下一步应回到Clutter合法覆盖、标签/来源子流型与L3 train parent-grouped OOF评分设计，而不是组合四个负结果。若未来重新研究距离不稳定性，需提出新物理量、预注册方向并使用新的独立holdout。
- 验证状态：冻结evaluate已重复执行且报告SHA-256=`324dbd6c009cb2a665adc8d1752b7a35780fa03f64945a85458df23546db6f99`；Clutter support仅9，单臂Clutter Recall bootstrap 95%区间约`[0.444444,1.0]`，相同硬预测只说明当前development集合无增益，不证明总体模型等价。test、multi-seed、OOF、完整级联、正式拒识、stable、P95和部署均未执行。
- 关联版本：RADAR-LN3-V21d seed43 causal-prefix d-range Clutter/Bird正式负结果。

## Finding F222

- Severity：Critical（Clutter低support是源数据覆盖约束，不是漏索引）
- 文件位置：`data/manifests/cq08_unified_manifest.json`、`data/processed/RADAR-LN3-V16-cq08-capture-content-provenance-manifest.json`、`data/processed/RADAR-LN3-V21-cq08-clutter-parent-registry.csv`、`src-v3/processed/report/RADAR-LN3-V21-clutter-coverage-semantics-audit.json`
- 问题或结论：CQ08中源数字标签5只有50个合法parent，且50个全部已进入V16；split固定为train/val/test=`33/9/8`。仓库内不存在遗漏的合法Clutter parent，无法靠重扫manifest或复制prefix补support。
- Evidence：manifest共1550条、Clutter=`50`，V16共1546个parent且Clutter仍=`50`；两集合track id逐项相同。V16外4条是Bird 634/785与Drone 1008/3338，Clutter为0。50个Clutter都有独立capture token和独立Clutter-containing provenance component；其中5个component/capture token同时含Drone或Balloon，但这只证明来源分组共享，不能自动判错标。精确capture/raw/RD/normalized track/point/model-input hash跨split交集仍为0。
- Impact：V21 validation Clutter只有9个parent，四个特征单因素都无法改变冻结硬动作；继续仅调特征、loss或threshold不能增加独立Clutter机制覆盖。任何把overlapping prefix/window计作新support或把validation/test移入train的做法都会夸大样本量并污染评估。
- Action：保留50-parent和50-capture registry作为唯一现有合法登记；合法扩充只能是带权威采集时间、环境、传感器模式、物理机制与内容provenance的新CQ08兼容采集。若新增数据或确认session关系改变split，创建新index/contract并重算train normalization。
- 验证状态：focused smoke两次临时输出字节一致；parent/capture=`50/50`、split=`33/9/8`、自动标签变化=0。test 8个parent只从既有metadata登记，TR物理JSON、raw、MAT、RD tensor访问均为0；模型test评估、训练、OOF、拒识、级联、stable与P95未执行。
- 关联版本：RADAR-LN3-V21 Clutter coverage/semantics audit v1。

## Finding F223

- Severity：Critical（capture时间和Clutter物理语义尚无权威真值）
- 文件位置：`OriData/23/CQ-08/batch_process_all.m`、`src/process/cq08_manifest.py`、`data/processed/RADAR-LN3-V21-cq08-clutter-capture-registry.csv`、`data/processed/RADAR-LN3-V21-cq08-clutter-audit-cluster-profiles.csv`、`docs/radar_ln3_clutter_semantic_adjudication_protocol.md`
- 问题或结论：当前可证明的Clutter语义只有“原文件数字5→manifest ClutterTarget→MAT文件token Clutter”。仓库没有场景、环境、操作者备注或Clutter机制字段；`capture_id`日期又由点迹文件修改日期生成，不是权威采集日期。源数字6在脚本中名为`Other`，也不能据此证明其天然就是严格open-set Unknown。
- Evidence：`batch_process_all.m`先从点迹文件`dir(...).datenum`取得日期，再拼接第一行`HH:MM:SS`形成MAT文件名前缀；50个Clutter token共享`20250616`，但已验证独立采集日期实际为0。派生60秒候选burst共30组，8组多parent且8组跨split、最大10个parent；这些只能提示采集日志复核。train-only四个描述簇支持=`25/4/1/3`，validation最近簇=`8/1/0/0`，两个train描述层在validation无最近成员且三个层support<5；聚类强烈不均衡，不能命名为物理子型。
- Impact：将50个token称为50个独立环境、把共同日期称为真实采集日、把时间burst称为session，或把audit cluster命名为植被/多径等子型，都会把工程派生字段升级成不存在的真值。validation未覆盖两个train描述层也使Clutter泛化估计更脆弱。
- Action：使用42条train/validation人工裁决模板追溯权威采集日志、环境、模式和TR/RD证据；test继续sealed。规划上先确认或补充至少3个权威独立日期、3个环境并盘点全部部署模式；每个确认机制暂按至少10 train + 5 development、跨2日期收集，最终缺口在人工裁决后计算。审计簇不得进入训练/采样/阈值，标签冲突只能升级为独立授权与版本化复核。
- 验证状态：标签5/6映射、50条文件名血缘、derived capture token来源、burst登记、train-only簇与test零payload访问均已确定性复核；独立人工语义真值仍未执行。11/11方法学风险已扫描，整体置信度为CAUTION而非语义VERIFIED。
- 关联版本：RADAR-LN3-V21 Clutter semantic adjudication protocol v1。

## Finding F224

- Severity：High（V21e产生首个超过冻结门槛的KnownGate单因素信号，但仍需多seed与OOF确认）
- 文件位置：`src-v3/processed/report/RADAR-LN3-V21e-known-gate-causal-course-consistency-paired-validation.json`、`src-v3/radar_ln3/known_gate_features.py`、`src-v3/report/known_gate_v21e.py`
- 问题或结论：非首点course的因果前缀圆周不一致度在seed43同初态、同parent顺序、全参数20 epoch配对中，将parent-final KnownGate的5个固定错误全部救回且无新增parent错误，是V21a-e中首个跨过`>2pp`冻结门槛的单因素候选。
- Evidence：Baseline/Treatment own-best分别为epoch 16/14，parent-final Accuracy=`0.978448/1.0`、Macro-F1=`0.862803/1.0`，增量=`+0.137197`；Bird 852/3580、Balloon 3027、Clutter 2900/3761全部纠正，四标签net rescue=`0/2/1/2`。5000次四标签分层paired parent bootstrap的Macro-F1差值95%区间=`[0.030530,0.258929]`。新增Fusion列48/48权重非零；固定Treatment权重只置零该输入后Clutter Recall由`9/9`降至`2/9`、Macro-F1降至`0.674092`，确认直接输入贡献。
- Impact：parent-final强收益没有等量延伸到在线早期语义。window/eligible-prefix Macro-F1仅变化约`+0.001317/+0.003806`，Known Target Recall约下降`-0.003498/-0.002763`；eligible prefix按原始标签净变化为Drone 0、Bird -4、Balloon -6、Clutter +8。Clutter 3761直到event 15才转为正确，故stable与早期路由仍未被验证。Clutter仅9个，Treatment全正确使非参数bootstrap单臂区间退化为`[1,1]`，不代表未来总体无误差。
- Action：V21e晋级seed42/43/44严格配对复核，再决定是否进入train parent-grouped OOF；每个seed必须有独立同seed Baseline、共享初态、零新增列、同20轮parent顺序和own-best冻结选择。多seed报告必须同时检查parent-final、prefix方向错误、Clutter support及跨seedpaired delta，不得用三个epoch或三个seed事后选优。原development用于特征发现，后续仍需要新独立holdout。
- 验证状态：seed43 prepare、20 epoch训练、development paired report、直接输入置零诊断和5000次bootstrap已执行；multi-seed、OOF、正式拒识、完整级联、test、stable、P95和部署未执行。
- 关联版本：RADAR-LN3-V21e causal course consistency seed43 positive candidate v1。

## Finding F225

- Severity：Critical
- Status：当前仍有效；实验命名与真实评估范围不一致
- Evidence：`data/processed/RADAR-LN3-V21e-cq08-known-gate-capture-content-grouped-oof-assignment.json`记录986个known parent、975个`provenance_component_id`组件，所有索引记录的`source_name`均为`CQ08`；`src-v3/processed/report/RADAR-LN3-V21e-known-gate-capture-content-grouped-oof-treatment.json`明确scope为`known train capture/content-grouped OOF; no validation/test/Unknown`。因此该结果是CQ08内部组件留出，不是真实跨source OOF。
- Impact：V21e Clutter P/R/F1=`0.642857/0.818182/0.720000`仅能说明CQ08内部组件泛化不稳定，不能证明跨采集源或跨设备泛化；当前没有真实跨source性能证据。
- Action：报告和V22验收统一称“CQ08内部组件留出”；只有加入具有权威source/provenance的新数据并新建相应`RADAR-LN3-VN`索引/契约后，才可报告真实跨source OOF。

## Finding F226

- Severity：Critical
- Status：已确认代码/训练协议问题
- Evidence：`src-v3/radar_ln3/losses.py:177`和`199-204`分别用`F.cross_entropy(..., weight=weights)`计算final与prefix loss；`src-v3/train/known_gate_v21e_oof.py:230-235`以配置`batch_size=2`构建DataLoader，`341-347`将同一class weight传入loss。PyTorch加权mean按当前batch出现的目标类别重新归一化；按V21e五折固定parent顺序重算，93.1%-93.6%的batch是同类目标，Clutter有效加权质量仅约6.3%-6.5%。
- Impact：Clutter虽配置了约14.6-15.2倍class weight，但历史归约下仅约6.2%-6.6%的有效batch权重质量来自Clutter；该机制足以造成Clutter监督不足并构成fold波动的高可信候选，但Clutter每折仅6-7个且provenance难度不同，不能在V22训练前宣称它是首要因果来源。
- Action：V22只验证`fixed_parent_count` class-weight归约，固定V21e Control、采样、batch size、特征、dt、液态、OOF分组和评估协议，不叠加其他因素；以Clutter P/R/F1、KnownTarget安全、eligible-prefix、component bootstrap及描述性fold方差共同判定性能猜测。

## Finding F234

- Severity：Critical
- Status：loss机制已修复；性能猜测待正式五折验证
- Evidence：`src-v3/radar_ln3/losses.py`新增`class_weight_normalization=fixed_parent_count`，final严格计算`sum(w[y] * CE) / batch_parent_count`，prefix先在parent内平均再按class weight求和并除以有prefix的parent数；历史V21默认`batch_weight_sum`且配置identity hash保持不变。V22 focused Smoke逐式验证历史同类Clutter batch权重完全抵消、修复后保留Clutter权重，并重放五折20 epoch全部parent order hash：目标同类batch比例=`0.9322/0.9334/0.9346/0.9352/0.9359`，历史Clutter有效质量=`0.06555/0.06436/0.06322/0.06271/0.06201`，V22约=`0.50015/0.49967/0.50000/0.50017/0.50066`。
- Impact：代码层已不再把稀有类权重按batch组成重新归一化，V22在不改采样和模型的条件下恢复声明的全局parent/class-balanced目标；但更强Clutter梯度也可能把更多Bird/Drone推向Clutter，不能保证Precision与Recall同步上升。
- Action：用户运行五个V22 Treatment fold，Control复用冻结V21e OOF Treatment；只有Clutter parent P/R/F1均严格改善、KnownTarget和三目标标签安全、eligible-prefix安全、fold描述性方差均不增且至少一项严格下降、component bootstrap的Clutter P/R与Macro-F1下界均>0，才把性能猜测记为支持；parent Macro-F1还需严格`>+0.02`才记为可采纳候选。无论结果如何，L3、Unknown、test、完整级联、stable和P95仍未执行。

## Finding F227

- Severity：High
- Status：当前仍有效；窗口/parent/产品动作目标不完全统一
- Evidence：`src-v3/radar_ln3/losses.py:132-136`规定主loss只监督每个parent最终状态；`188-216`规定prefix从第3事件开始、使用parent最终标签、排除最终状态并在parent内平均。`src-v3/radar_ln3/known_gate.py:1408-1417`同时输出window、eligible-prefix、parent-final指标并将最小证据前事件定义为`PENDING_ABSTAIN`，但该loss没有`PENDING_ABSTAIN`、`CLUTTER_ACCEPTED`或`UNKNOWN_REJECTED`动作目标。
- Impact：parent-final指标可能优于早期prefix，而V21e正出现parent收益远大于eligible-prefix收益且早期KnownTarget Recall下降；训练选优不能被解释为已优化线上L1/L2/L3动作或stable可靠性。
- Action：V22先固定现有监督协议以隔离类别平衡因素，同时把parent、eligible-prefix、window和产品动作分层验收；若后续修复动作监督，必须另建训练目标/checkpoint schema并重新训练。

## Finding F228

- Severity：High
- Status：有证据但待验证为主要因果瓶颈
- Evidence：V16时间字段只有1秒精度；`tr_dt_seconds=0`由首点无前驱和同秒时间戳构成，未发现负值、NaN、Inf或单位转换错误；train中Clutter同秒事件比例为0而Drone/Bird/Balloon存在同秒事件。V21配置使用`liquid_zero_dt_event_floor_seconds=0.001`和`liquid_invalid_dt_update_mode=bounded_observation`，V20/V21相关液态合同Smoke通过。
- Impact：dt分布可能让液态路径学习到类别/来源相关捷径，但当前没有独立KnownGate dt消融或跨fold因果结果，不能把dt直接认定为Clutter主因，也不能以增大液态容量掩盖该问题。
- Action：在V22固定类别平衡因素后另行做单因素dt语义消融；按source、类别、prefix、轨迹质量和fold报告零/无效dt，并验证offline/online逐点输出一致。

## Finding F229

- Severity：Critical
- Status：已确认代码语义冲突；当前产品影响待运行验证
- Evidence：`src-v3/radar_ln3/rejection.py:779-807`在DBB knownness未接受时先计算CU Clutter阈值，满足条件即返回`final_action="CLUTTER_ACCEPTED"`，理由包含`dbb_known_not_accepted_fallback_to_cu`；同文件注释说明该路径是DBB knownness失败后fallback到CU。用户固定语义要求“目标证据充分但DBB子类证据不足→UNKNOWN_REJECTED”，不能转为Clutter。
- Impact：一旦该历史fallback被当前线上级联调用，目标子类证据不足可能被误接受为Clutter，直接损害Clutter Precision、Drone/Bird Recall和Unknown/Pending语义；当前V21完整三级调用链尚未执行，影响数为未执行。
- Action：在完整L1/L2/L3重接前明确互斥动作契约；目标证据充分但子类不足只能进入`UNKNOWN_REJECTED`，观测不足/质量低只能进入`Pending/Abstain`，不得以CU预测替代open-set拒识。

## Finding F230

- Severity：High
- Status：当前仍有效；特征发现与确认集重复使用风险
- Evidence：V21e特征来自同一development错误分析；V21h/V21i均复用V21e OOF Treatment作为冻结Control，V21f完整OOF训练未执行。V21e development单seed正向信号后虽完成seed42/43/44复现，但五折OOF报告最终Clutter绝对护栏失败；V21h/V21i不能作为独立确认。
- Impact：V21e course特征虽已真实接入且可能有信号，但其正向收益受到development后验选择、重复验证和多重比较影响；不能把V21e、V21h或V21i的结果当作已证明跨源泛化。
- Action：V22不加入新特征组合；先以固定V21e Control验证训练平衡协议。后续特征实验必须使用未参与特征发现的独立确认集或嵌套OOF，并预注册主要错误方向和停止条件。

## Finding F231

- Severity：Critical
- Status：当前仍有效；产品级证据缺失
- Evidence：`data/processed/RADAR-LN3-V21-cq08-known-gate-data-contract.json`将V21明确为`known_gate`，UnknownTarget不作为模型类，L2为`dbb_specialist`，L3为`semantic_decision_layer`且`cu_specialist_exists_in_v21=false`；`src-v3/radar_ln3/cascade.py:1-7`声明V21不使用旧cascade，但仍保留旧`dbb_cu_gate/dbb_specialist/cu_specialist`兼容路径。V21e报告将`test`、`formal_rejection`、`stable`均记为“未执行”，`deployment_eligible=false`。
- Impact：当前Clutter指标只是KnownGate证据层，不是最终`CLUTTER_ACCEPTED`率；Unknown拒识、Pending/Abstain、L2子类、L3校准、stable和线上一致性均未完成，不能放行正式产品或用总指标替代三级决策指标。
- Action：V22只允许在KnownGate训练层验证，训练前后必须保留产品语义不变；正式放行必须补齐L3 train-parent-grouped OOF校准、Unknown open-set、Pending/Abstain、完整cascade、stable和test证据。

## Finding F232

- Severity：High
- Status：已确认实验结论；V21h不采纳
- Evidence：`src-v3/processed/report/RADAR-LN3-V21h-cq08-known-gate-course-iso-paired-oof.json`对应V21h的Clutter parent Precision约提升`+0.163594`，但Recall约下降`-0.060606`；该报告为探索性、绝对安全门槛未通过，且没有形成Precision/Recall同步改善。
- Impact：目标行能量占比iso只能改变Clutter接受边界的取舍，不能作为当前Clutter Precision和Recall同时提升的可靠主方向；并可能损害目标类别路由。
- Action：V21h冻结为不采纳，不进入V22组合特征或L3阈值调优；如未来重新验证，必须独立预注册并同时设置目标类别安全门槛。

## Finding F233

- Severity：High
- Status：已确认实验结论；V21i不采纳
- Evidence：`src-v3/processed/report/RADAR-LN3-V21i-cq08-known-gate-course-d-range-paired-oof.json`为探索性OOF结果，Clutter parent Precision、Recall、F1均下降，报告状态为`completed_exploratory_not_confirmed`，未进入部署或阈值校准。
- Impact：d_range稳定性当前没有改善Clutter跨组件留出边界的证据，组合该特征会增加因果归因和源域偏移风险。
- Action：V21i冻结为不采纳；液态模块、dt和已否决特征均不与V22训练平衡实验联合改变。

## Finding F235

- Severity：Critical
- Status：已确认冻结V21e模型对RD内容存在显著条件依赖
- Evidence：五折epoch 20 checkpoint只读重放与正式OOF的986个parent、18923个window和15965个eligible-prefix硬预测逐项一致，概率最大绝对误差`1.1921e-7`。仅把RD embedding置零而保留原始因果质量向量后，parent Macro-F1从`0.854465`降至`0.587757`，KnownTarget Recall从`0.984260`降至`0.841553`，Clutter Precision从`0.642857`降至`0.156425`；142个原正确parent受损、7个被救回，exact McNemar `p=8.26e-34`，component bootstrap Macro-F1差值95%区间`[-0.319191,-0.212289]`。五折差值均为负，但范围为`-0.028307`至`-0.453447`。
- Impact：现有masked-mean RD内容是V21e KnownGate已实际使用的判别来源，V22a/V22b只否决首尾embedding差分，不能据此删除整个RD分支。与此同时，跨fold幅度异质说明消融结果不能直接转化为“新增某个RD特征必然提升”的结论。
- Action：保留现有RD内容主路径；停止首尾embedding residual方向。后续RD候选必须使用全帧分布且单因素预注册，并继续用组件OOF和类别安全门槛确认，不能只看development或权重绝对值。

## Finding F236

- Severity：Critical
- Status：已确认RD缺失质量语义存在类别支持缺口
- Evidence：V21e known train中，Drone/Bird/Balloon合计有153/18266个缺失RD事件，涉及25/953个目标parent；Clutter的33个parent、657个事件均有RD，缺失事件为0；所有类别都没有整条parent全事件缺RD支持。把五折冻结模型的每个事件改为合法missing质量向量`[0,0,0,1,0,0,0]`并移除RD内容后，986个parent全部输出KnownTarget，Clutter P/R/F1全部为0。
- Impact：`rd_present/rd_missing`及其相关质量字段可形成目标/Clutter覆盖捷径；全missing臂是明确的分布外反事实，不能被报告成无RD重训练模型性能。若产品质量不足时仍直接消费KnownGate argmax，可能系统性偏向KnownTarget；当前尚未运行完整Pending/Abstain和L3语义决策链。
- Action：下一优先项不是继续追加RD residual，而是冻结一个单因素质量鲁棒性方案：补齐或合成标签均衡的合法RD缺失训练支持，或由L3在质量不足时强制`PENDING_ABSTAIN`。方案必须预注册且分别验证正常RD性能、缺失RD安全、Bird/Drone误拒、Clutter安全和offline/online一致性。

## Finding F237

- Severity：High
- Status：已确认架构命名/实现边界；V23基座尚未训练
- Evidence：`src-v3/radar_ln3/model.py:187-291`的历史`CfCLiquidCausalEncoder`仅定义`input_projection/state_projection/gate_projection/candidate_projection/tau_unconstrained/output_norm`，以`alpha=1-exp(-dt/tau)`及线性插值更新state；项目V3代码未导入`ncps`、`torch_cfc`或LTC实现，审计时`.venv`中三者均不存在。V23新增`src-v3/radar_ln3/official_cfc.py`，runtime硬校验官方`ncps==1.0.1`、Git commit `695ec22d7d1831f2ced67346ef7d2f08c525bd47`、类模块`ncps.torch.cfc`；真实V21 train batch smoke确认官方CfC参数14,064、梯度有限非零、290s正dt未截断且offline/online最大logit误差`4.76837158203125e-07`低于`1e-6`容差；历史48维自定义单元参数为14,112。
- Impact：V16-V22的相对实验结论只适用于项目自定义、CfC-inspired连续时间门控主干，不能宣称已经证明标准CfC/LTC收益。V23只先提供一份无course、`batch_weight_sum`、seed43的开发基座，不是完整KnownGate产品、Unknown拒识或跨source泛化结论。
- Action：历史文档与后续报告统一称`CfC-inspired custom encoder`或“自定义连续时间门控编码器”；仅V23称官方`ncps.torch.CfC`。先完成单模型20 epoch基座与开发validation，再决定多seed；五折OOF在多seed后独立预注册。旧V23a/V23b OOF预备方案已取消且未训练，禁止作为现有性能或实验依据；test、正式拒识、L3、完整级联、stable和P95继续未执行。

## Finding F238

- Severity：High
- Status：训练前V23隔离缺口已修复并复核
- Evidence：初版`OfficialCfCKnownGateClassifier`在官方CfC构造后继续初始化fusion/classifier，因官方CfC与历史自定义单元消耗不同数量随机数，非liquid参数中fusion和classifier初态并不等于冻结V21e/V22 Control；同时初版仅V23对padding logits施加零填充，而历史`KnownGateClassifier`保留classifier bias。复审后新增冻结source非recurrent tensor复制和哈希审计；真实batch Smoke确认30个非recurrent tensor逐值相等、40个可训练参数均有有限非零梯度，官方recurrent state仍保持独立。V23已改为只阻止padding进入official recurrence，并让loss/评估继续以真实长度排除padding。
- Impact：修复后V23的可变化因素收窄为官方CfC recurrent结构及其内在原生timespan方程；TR/RD前端、fusion、classifier、数据、loss、parent顺序、epoch和评估张量契约均冻结。不能把这保证解读为跨架构初始logits相同，也不能将V23作为Unknown、L3、完整级联、stable、P95或test结论。
- Action：V23 checkpoint、summary和report必须绑定source初态hash、共享non-recurrent-state hash与官方runtime direct-url/class-module/owner；先运行单一基座全量20 epoch与开发评估，之后再决定多seed和五折全量脚本。

## Finding F239

- Severity：High
- Status：已确认官方CfC运行时兼容边界；V23基座尚未训练
- Evidence：`smoke/known_gate_v23_official_cfc_runtime_compatibility.py`只用固定随机张量运行官方`ncps==1.0.1`。`CfC(dt=0)`的output/state均有限、最大状态变化非零，故零间隔不是skip/identity；`CfC.forward(..., padding_mask=...)`报`TypeError`；batch=2的异步`[B,L]`timespans报broadcast `RuntimeError`；dense 48-unit CfC无wiring；同一输入下`LTC(dt=0)`产生96个NaN。审计报告的data/checkpoint/validation/test访问计数均为0。
- Impact：直接替换为LTC、将零dt改为floor/skip、把异步batch强行传给官方CfC或把padding传给上游API，都会改变V23的冻结定义或失败。逐parent调用官方CfC是当前上游API限制下的必要适配，预计相对假设的向量化批处理更慢；这不是P95测量。
- Action：运行时sidecar必须在prepare/smoke中重新生成并通过，checkpoint、summary和development report绑定其路径/SHA/schema/checks。开发报告只留三种评估范围的核心指标、2×2混淆矩阵和Clutter support=9警告；训练、多seed、OOF、test、完整L1/L2/L3、拒识、stable和P95仍未执行。
