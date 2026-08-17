# Radar-LN3 Clutter标签语义与子流型人工裁决协议

本协议只用于审计CQ08源数字标签`5`（当前工程名`ClutterTarget`）的来源覆盖与物理语义。它不修改标签，不把审计聚类当成真值，不改变V16 split，也不把`UnknownTarget`作为KnownGate闭集训练类别。

## 1. 当前可证明与不可证明的语义

当前代码和文件可复核的最强结论只有：原始文件名携带数字标签`5`，统一manifest将其映射为`ClutterTarget`，`batch_process_all.m`再把数字`5`写成MAT文件名中的`Clutter`。仓库没有独立的采集场景记录、操作者备注、环境类型、杂波机制或既有人工复核字段。

因此，在完成外部采集记录核对前，不得把某个parent自动称为静态地杂波、植被、天气、多径、旁瓣、移动地物或处理伪影。数字标签`6`在MAT生成脚本中名为`Other`；这同样不能证明它天然等于严格open-set Unknown。V21产品中的Unknown仍是严格开放集结果/评估真值，不是本协议的Clutter负类或聚类输入。

## 2. 审计产物的证据等级

- `clutter-parent-registry.csv`：50个合法Clutter parent的全量登记。parent、capture、V16 provenance、split和hash属于来源证据；train/validation描述统计只用于覆盖审计。
- `clutter-capture-registry.csv`：50个capture token的登记。`capture_id`来自MAT文件名；其日期部分由`batch_process_all.m`读取点迹文件修改日期，时分秒来自点迹第一行。因此不同token可用于现有provenance精确分组，但不自动证明真实采集日期、环境、session或传感器模式独立。
- `clutter-acquisition-burst-registry.csv`：按相邻派生capture token时间差形成的候选组。它只提示应查阅哪些采集记录；由于日期源是文件修改日期，它不是权威session时间轴，不能直接改split或命名子流型。
- `clutter-audit-cluster-profiles.csv`：只用33个train Clutter parent拟合的描述簇；9个validation parent只分配到最近的冻结train簇；8个test parent不读取物理描述、不分簇。`audit_cluster_id`不是物理类别，也不得输入训练、采样、权重或阈值。
- `clutter-semantic-adjudication-template.csv`：仅含train/validation parent的人工裁决入口。test保持sealed，不在模板中。

## 3. 建议语义枚举

只有在采集日志、现场说明或可定位的TR/RD证据支持时，人工审核者才可填写以下主/次语义。枚举是审核词表，不代表仓库已经拥有这些子型：

- `STATIC_GROUND_OR_INFRASTRUCTURE`
- `VEGETATION_OR_BIOLOGICAL_BACKGROUND`
- `WEATHER_OR_ATMOSPHERIC`
- `MULTIPATH_OR_SIDELOBE_INTERFERENCE`
- `MOVING_ENVIRONMENTAL_OBJECT`
- `SENSOR_OR_PROCESSING_ARTIFACT`
- `MIXED_CLUTTER_MECHANISMS`
- `OTHER_DOCUMENTED_CLUTTER`
- `AMBIGUOUS_INSUFFICIENT_EVIDENCE`

若证据不足，必须使用`AMBIGUOUS_INSUFFICIENT_EVIDENCE`或保持空白；禁止依据模型预测、单一特征阈值、审计簇编号或“看起来像”强行填写机制。

## 4. 逐parent审核步骤

1. 先确认`parent_sample_id`、`track_id`、`capture_id`、原始数字标签、文件名和provenance hash对应同一记录。
2. 查阅原始采集日志、操作者备注和场景上下文，填写`confirmed_acquisition_group_id`、`confirmed_environment_id`与`confirmed_sensor_mode_id`。时间burst只能作为检索提示。
3. 分别记录是否查看了TR轨迹证据和raw/RD证据；在`evidence_references`中写入可复核文件、帧或外部记录定位，不写模糊结论。
4. 在有证据时填写主/次语义、置信等级与`subtype_evidence_basis`；审核者、时间必须完整。
5. `source_label_decision`只允许：`KEEP_SOURCE_LABEL_5`、`ESCALATE_LABEL_CONFLICT`、`INSUFFICIENT_EVIDENCE`。`ESCALATE_LABEL_CONFLICT`只会提出独立版本化复核请求，不会自动改训练标签。

## 5. 覆盖验收

子流型覆盖只能按人工确认的`environment_id × sensor_mode_id × semantic_subtype × acquisition_date`统计。parent数、capture数、时间burst数和审计簇数不得互相替代。正式声称某物理机制可泛化前，暂定规划下限为：该机制至少10个独立train parent与5个独立development parent、跨至少两个采集日期；这只是保守采集规划下限，不是统计功效计算或性能保证。

当前50个Clutter的capture token共享`20250616`，但该日期来自点迹文件修改日期，不能算作一个已验证采集日期；环境与传感器模式字段同样缺失。因此现阶段只能证明源parent完整登记，不能证明日期、环境或物理子流型覆盖充分。增加support的合法方式是新增带权威采集时间、provenance与人工场景记录的CQ08兼容采集；禁止复制重叠prefix/window、挪用validation/test、删除困难样本或按模型结果改标签。

## 6. test与开放集边界

本审计器只从既有manifest/provenance登记test的8个Clutter parent，不解析其TR物理字段，不打开raw、MAT或RD tensor，不进行分簇或人工语义裁决。模型test评估、OOF、正式拒识、完整级联、stable与P95均不由本协议执行。
