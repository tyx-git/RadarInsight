# RadarInsight v3实施计划

## Stage 008 - src-v3最小重建方案设计

目标：在不修改`src/`的前提下，建立CQ08主数据的TR点/RD点token契约和完全位于`src-v3/`的可运行链路。
前置依赖：用户已授权最小实施；CQ08的TR/RD对应关系、五分类、因果时序契约和GPU P95约束已确认；旧链路只作为只读参考。
最小子任务：

- [x] 核实工作树、项目目录、旧版本文件和`src-v3`/`config/RADAR-LN3`现状。
- [x] 建立并维护本计划、过程和发现跟踪文件。
- [x] 冻结V0数据契约、字段schema、输入输出shape和artifact语义。
- [x] 完成`src-v3`内部导入关系与静态隔离检查。

验证命令：`git status --short --branch`；`rg -n "from src|import src|sys\.path|importlib|SourceFileLoader" src-v3`。
产物：`src-v3/`跟踪文件、V0契约设计、src-v3模块骨架。
验收条件：无`src`运行时依赖；契约包含真实TR dt、点级RD聚合、missing/mask和质量字段；旧数据不被覆盖。
是否需要数据重建：需要，限CQ08点级索引/manifest/契约。
是否需要smoke：需要。
是否需要正式训练：不需要，未执行。
阻塞条件：无法确认TR点和RD帧映射、数据源不纯、或隔离检查发现旧src依赖。

## Stage 009 - CQ08/raw数据集和索引重建

目标：只使用CQ08主数据建立一行一个TR点的`RADAR-LN3-V0`点级契约，RD帧按`track_point_index`聚合，保留部分覆盖和异常质量标记。
前置依赖：Stage 008契约冻结；旧canonical RD帧索引和CQ08轨迹文件字段已核实。
最小子任务：

- [x] 读取CQ08轨迹、点迹和只读canonical RD帧索引。
- [x] 生成V0点级manifest、索引、数据契约和统计摘要。
- [x] 校验标签、source、parent、track point、RD帧数量、顺序异常和缺失状态。
- [x] 校验CQ08纯度及按parent/track的split隔离；split重建如无法复用现有分组规则则记录未执行。

验证命令：在`src-v3`目录运行`..\\.venv\\Scripts\\python.exe -m build.cq08_point_contract`；再运行`..\\.venv\\Scripts\\python.exe -m validate.cq08_point_contract`。
产物：`data/processed/RADAR-LN3-V0-*`点级文件和摘要。
验收条件：仅含CQ08；点级TR/RD关系可由文件重新验证；没有把RD CPI时间当作TR墙钟时间；缺失和异常不是静默丢弃。
是否需要数据重建：需要，执行小型索引/manifest重建，不复制原始RD张量。
是否需要smoke：需要。
是否需要正式训练：不需要，未执行。
阻塞条件：轨迹字段或canonical RD frame index无法一致解析；纯度或group split校验失败。

## Stage 010 - src-v3特征提取和液态时序模块实现

目标：在src-v3内实现TR特征、RD局部CNN前端、点级融合、CfC风格液态因果模块、五分类头和目标级稳定状态接口。
前置依赖：Stage 009 V0契约和字段shape校验通过。
最小子任务：

- [x] 实现TR物理/环状/缺失特征提取和训练统计接口（V0 smoke统计为固定契约缩放，正式训练统计未生成）。
- [x] 实现Dataset、collate、padding/mask和真实dt传递。
- [x] 实现RD局部空间前端与同TR点内帧聚合。
- [x] 实现完全src-v3内部的TR/RD融合、液态时序、窗口级输出和目标级稳定状态机。
- [x] 实现配置加载、checkpoint schema和在线推理接口；正式训练/评估入口未实现。

验证命令：在`src-v3`目录运行`..\\.venv\\Scripts\\python.exe -m smoke.static_isolation_check --root F:\\Project\\RadarInsight\\src-v3`和`..\\.venv\\Scripts\\python.exe -m smoke.contract_tests --project-root F:\\Project\\RadarInsight --config F:\\Project\\RadarInsight\\config\\RADAR-LN3\\RADAR-LN3-V0-cq08-tr-rd-smoke-config.json`。
产物：`src-v3/radar_ln3/*.py`正式链路、`src-v3/smoke/*.py`验证入口、`config/RADAR-LN3/RADAR-LN3-V0-cq08-tr-rd-smoke-config.json`。
验收条件：输入输出shape、因果顺序、dt、mask、有限值和checkpoint接口明确；不引用`src`。
是否需要数据重建：使用Stage 009产物，不新增数据范围。
是否需要smoke：需要。
是否需要正式训练：不需要，未执行。
阻塞条件：缺少确认的第三方依赖、模型无法在可用设备forward、或checkpoint语义与现有产物冲突。

## Stage 011 - 预训练或短时smoke验证

目标：验证数据读取、forward、loss、梯度、参数更新、checkpoint保存恢复和短时有限步训练。
前置依赖：Stage 009数据纯度通过；Stage 010隔离和契约测试通过；配置实际由src-v3入口加载。
最小子任务：

- [x] 运行导入隔离检查。
- [x] 运行数据纯度、索引统计、shape/mask测试。
- [x] 运行forward、loss、梯度和checkpoint round-trip测试。
- [x] 运行有限步GPU smoke，记录设备、步数、耗时和输出。
- [x] 尝试focused GPU latency测量并明确不宣称端到端生产达标；最终重复测量超时，稳定P95保持未验收。

验证命令：在`src-v3`目录运行`..\\.venv\\Scripts\\python.exe -m smoke.run_smoke --project-root F:\\Project\\RadarInsight --config F:\\Project\\RadarInsight\\config\\RADAR-LN3\\RADAR-LN3-V0-cq08-tr-rd-smoke-config.json`；契约、缺失RD、审计和focused latency入口均位于`src-v3/smoke/`。
产物：`src-v3/smoke/*.py`验证内容、`data/processed/RADAR-LN3-V0-*`验证摘要、`artifacts/RADAR-LN3-V0-*`仅用于smoke的checkpoint/日志（若成功）。Smoke checkpoint仍按artifact约定保存于`artifacts/`，不属于正式模型产物。
验收条件：有限值、梯度、恢复后输出一致性和数据纯度通过；长时间训练、正式预训练、大规模评估和独立测试集评估保持未执行。
是否需要数据重建：不需要。
是否需要smoke：需要，授权范围内执行。
是否需要正式训练：不需要，未执行。
阻塞条件：任何隔离、纯度、shape、mask、forward、loss、梯度或checkpoint失败。

## Stage 013 - 进入后续训练的放行判断

目标：基于实际命令输出判断是否可进入下一阶段。
前置依赖：Stage 011所有必需smoke证据。
最小子任务：

- [x] 汇总实际文件、hash、测试和smoke结果。
- [x] 区分已验证结果、工程判断和未执行事项。
- [x] 输出“不能进入下一阶段”，并列出正式训练/评估/P95放行阻塞项。

验证命令：读取v3过程、发现和实际输出文件并交叉核对。
产物：最终交付摘要和下一步执行命令。
验收条件：不把smoke或旧实验写成正式性能结论；未授权事项仍明确为未执行。
是否需要数据重建：否。
是否需要smoke：否。
是否需要正式训练：否，需另行授权。
阻塞条件：证据不足或契约/资源/依赖未满足。

## Stage 014 - 异常parent排除策略冻结

目标：按用户选择方案1，保持V0不变，主实验排除3个异常parent，建立独立异常审计范围。
前置依赖：V0索引、V0摘要和异常标记已通过只读核实；用户已明确选择方案1。
最小子任务：

- [x] 冻结异常parent集合为`cq08|track|634`、`cq08|track|785`、`cq08|track|3338`。
- [x] 确认异常原因、原始split、标签和点/RD数量。
- [x] 确认V0文件只读不修改，V1只表达选择视图变化。

验证命令：读取V0索引并按`tr_time_order_status`/`rd_parent_order_status`聚合。
产物：V1主视图与异常审计视图设计记录。
验收条件：异常集合可由V0字段重现；主集和审计集parent不重叠。
是否需要数据重建：需要，只生成V1视图索引和契约，不复制RD张量。
是否需要smoke：需要focused Dataset/shape校验。
是否需要正式训练：不需要，未执行。
阻塞条件：异常集合不稳定、V0 hash改变或主/审计范围重叠。

## Stage 015 - RADAR-LN3-V1主集/审计集视图生成

目标：创建全局下一个版本V1的主实验index、异常审计index、split、manifest、数据契约和配置引用。
前置依赖：Stage 014冻结；生成前已读取`data/`、`config/RADAR-LN3/`和`src-v3/process_v3.md`确认最大版本为V0。
最小子任务：

- [x] 生成V1主index：去除3个异常parent，保留train/val/test原split语义。
- [x] 生成V1异常审计index：仅保留3个异常parent，设置`split_name=audit`并保留原始split字段。
- [x] 生成V1主/审计split、manifest、数据契约和summary。
- [x] 生成并加载V1主smoke配置和审计配置。

验证命令：在`src-v3`目录运行`..\\.venv\\Scripts\\python.exe -m build.v1_anomaly_views`；再运行`..\\.venv\\Scripts\\python.exe -m validate.v1_anomaly_views`。
产物：`data/processed/RADAR-LN3-V1-*`、`config/RADAR-LN3/RADAR-LN3-V1-*`。
验收条件：V0 hash不变；main=1546 parent、audit=3 parent；主/审计不重叠；不复制或覆盖旧RD张量。
是否需要数据重建：需要，限视图index/manifest/split/契约。
是否需要smoke：需要。
是否需要正式训练：不需要，未执行。
阻塞条件：V1版本冲突、文件未以V1前缀命名、主/审计集合不闭合或配置未实际加载。

## Stage 016 - 主集/审计集Dataset与DataLoader验证

目标：证明主配置只读取非异常parent，审计配置只读取异常parent，shape/mask/label契约不变。
前置依赖：Stage 015 V1文件和配置通过校验。
最小子任务：

- [x] 加载V1主配置并验证train/val/test Dataset parent集合。
- [x] 加载V1审计配置并验证audit Dataset只含3个异常parent。
- [x] 执行主集与审计集DataLoader、shape、finite、RD missing mask检查。

验证命令：在`src-v3`目录运行`..\\.venv\\Scripts\\python.exe -m validate.v1_anomaly_views --run-dataset-checks`。
产物：V1视图验证摘要。
验收条件：主集不含634/785/3338；审计集包含且只包含它们；底层输入shape仍为TR 14维、RD `[K,2,21,384]`。
是否需要数据重建：否。
是否需要smoke：需要，限focused验证。
是否需要正式训练：不需要，未执行。
阻塞条件：Dataset误把审计样本混入主集，或配置路径/版本不一致。

## Stage 017 - V1验证训练放行

目标：在V1视图验证完成后，决定是否进入正式监督训练；默认只提供命令和资源说明，不自动启动长时间训练。
前置依赖：Stage 016通过；用户另行确认正式训练范围、轮数、seed和评估契约。
最小子任务：

- [x] 汇总V1主集/审计集数量、hash、配置和focused测试。
- [x] 保留V0 baseline和V1异常审计对照契约。
- [x] 输出正式训练放行结论：当前不能自动进入正式训练，需用户另行确认训练范围和资源。

验证命令：读取V1过程、发现、summary和实际测试输出；Stage 018隔离契约已通过，正式训练命令仍需用户确认。
产物：V1放行记录；正式训练入口已在Stage 018实现，但未授权执行。
验收条件：不把主集smoke写成正式性能；审计集不用于训练；未授权训练保持未执行；明确区分“待训练准备”和“可启动正式训练”。
是否需要数据重建：否。
是否需要smoke：否。
是否需要正式训练：需要另行确认，当前未执行。
阻塞条件：V1视图、split、指标分母或资源约束未冻结。

## Stage 018 - 正式监督训练前置准备

目标：把V1主集从“链路Smoke已通过”推进到“正式训练可启动”，不自动启动长时间训练。
前置依赖：Stage 017通过；用户确认训练预算、seed、epoch/early stopping、验证范围、checkpoint选择指标和GPU资源。
状态：已完成；正式训练未启动。隔离artifact路径契约复测已通过，Stage 018技术前置闭合；正式训练仍需用户明确授权。
最小子任务：

- [x] 在`src-v3`内实现正式监督训练入口；Smoke入口保持在`src-v3/smoke/`，未混入正式链路。
- [x] 在`src-v3`内实现正式验证/评估入口，覆盖causal window、目标稳定结果、parent和CQ08 source指标；session字段在当前V1契约中不存在，评估结果明确输出`not_available`，未估算。
- [x] 创建正式配置前读取`data/`、`config/RADAR-LN3/`和本文件；最大版本为V1，本批次使用下一个全局版本V2。
- [x] 冻结训练期归一化统计、类别权重、seed、checkpoint选择指标和V1 main数据范围；审计集保持forward/evaluation-only。
- [x] 完成隔离artifact路径下的正式入口配置加载、V1 main纯度、checkpoint恢复、验证分母和短时训练契约复测；此前正式artifact目录中的1 step边界产物不作为本项证据，当前证据来自独立entry-contract目录。

验证命令：已执行`src-v3`全量编译、`smoke.static_isolation_check`、V1视图校验、V2配置/数据hash核验和隔离artifact路径下的`smoke.formal_entry_contract`；本次Smoke退出码0并输出`status=passed`。
产物：`src-v3/train/formal.py`、`src-v3/report/classification.py`、`src-v3/radar_ln3/contracts.py`、`factory.py`、`metrics.py`、`evaluator.py`、V2正式配置、V2正式数据契约和train-only归一化统计；正式全量模型产物未生成。
验收条件：正式入口不依赖`src`；V1 main训练与audit隔离；V2正式配置不设置`sample_limit`；验证指标分母和checkpoint选择规则已编码；隔离artifact契约通过；test、全量训练和稳定P95仍未执行，不能将本Stage Smoke当作性能结果。
是否需要数据重建：默认不需要；只有归一化、字段、窗口或数据范围发生实质变化时才递增RADAR-LN3版本并重建对应契约。
是否需要smoke：需要，且Smoke代码只位于`src-v3/smoke/`。
是否需要正式训练：需要用户明确确认后执行；当前未执行。
阻塞条件：正式训练授权未给出、test/全量评估和端到端P95尚未执行；隔离artifact契约和GPU CUDA入口不再是当前阻塞项。

## Stage 019 - 正式训练与验证授权等待

目标：在Stage 018隔离契约复测通过且用户明确授权后，执行V1 main的正式监督训练和验证；不自动读取test，不把smoke或边界检查写成性能结果。
前置依赖：Stage 018最后一项契约复测通过；用户确认训练命令、GPU独占或资源窗口、20 epoch/early stopping=5、seed=42、验证集选优和评估范围。
状态：正式训练已执行并于epoch 15因early stopping结束；best checkpoint已完成GPU验证集独立复核。test、session和端到端P95仍未执行。
最小子任务：

- [x] 在`src-v3/smoke/`完成隔离artifact路径的有界契约复测，并确认checkpoint目录为`artifacts/RADAR-LN3-V2-cq08-tr-rd-formal-entry-contract/`。
- [x] 用户授权后运行V1 main train split的正式监督训练；预训练保持不使用。
- [x] 从旧正式summary安全迁移best epoch验证记录，生成新字段`best_checkpoint`、不含records的summary和`sample_detail_information.json`；不修改best checkpoint。
- [x] 使用新报告入口在GPU上重新运行V1 main val，独立复核best checkpoint；session缺失时保持`未评估`。
- [ ] 在明确授权test后再运行test；否则记录“未执行”。
- [ ] 单独执行GPU在线端到端P95基准，覆盖RD读取/聚合、GPU传输、模型forward和目标级状态更新。

验证命令：正式训练命令已实际执行；GPU独立验证命令已实际完成并写入`evaluation-val.json`；当前test评估和P95未执行。
产物：`artifacts/RADAR-LN3-V2-cq08-tr-rd-formal/`正式checkpoint和训练摘要、val报告、经授权的test报告及独立端到端延迟报告。
验收条件：正式训练命令实际加载V2配置和V1 index hash；无test越权；指标使用真实parent/source范围；90%目标、Drone/Bird混淆、Clutter/Unknown和P95只以真实运行结果判定。
是否需要数据重建：否，除非发现数据契约或归一化语义发生实质变化。
是否需要smoke：需要，Stage 018复测必须先通过。
是否需要正式训练：已执行；不因日志/报告格式修改自动重训。
阻塞条件：test未授权、session字段缺失、GPU端到端P95未执行；Clutter/Unknown的有效泛化能力仍受验证集支持数限制。

## Stage 020 - src-v3目录与入口命名规范化

目标：按职责整理`src-v3`代码，使数据构建/处理、训练、报告评估、校验和Smoke入口分层清晰；不改变数据、模型、配置、版本和artifact语义。
前置依赖：Stage 018已有正式入口和Stage 019保持未授权；旧入口路径已完成只读盘点。
状态：已完成。
最小子任务：

- [x] 盘点旧入口、实际引用、默认项目根目录计算和目标目录冲突；确认旧目标目录不存在。
- [x] 建立`build/`、`train/`、`report/`、`validate/`包目录及`__init__.py`。
- [x] 将数据构建/处理、训练、报告评估和校验入口移动到职责目录，并去除文件名中重复的`build`、`train`、`evaluate`、`validate`字样。
- [x] 修正移动后CLI默认项目根目录层级和Smoke到训练入口的包导入；未产生`src`运行时依赖。
- [x] 通过全量编译、静态导入隔离和全部新入口`--help`检查；确认旧入口不存在、新入口共7个。
- [x] 在`process_v3.md`记录旧路径历史兼容映射和本次无数据/配置/版本变化事实。

验证命令：`python -m py_compile`覆盖全部`src-v3/*.py`；`python -m smoke.static_isolation_check --root F:\\Project\\RadarInsight\\src-v3`；新`build.*`、`validate.*`、`train.formal`和`report.classification`入口分别执行`--help`。
产物：`src-v3/build/`、`src-v3/train/`、`src-v3/report/`、`src-v3/validate/`及更新后的三个跟踪文件。
验收条件：职责目录清晰；旧入口路径为0；新入口路径为7；编译、隔离、帮助检查通过；数据、配置、模型和artifact未改变。
是否需要数据重建：否。
是否需要smoke：需要静态隔离/入口focused检查，已完成；不需要模型训练Smoke。
是否需要正式训练：否，未执行。
阻塞条件：移动后导入失败、默认根目录错误、旧路径残留、隔离检查失败或发现语义变化。

## Stage 021 - 训练日志模块与进度输出

目标：在不改变训练配置、数据契约、模型输入输出或checkpoint语义的前提下，增加可读、高效、可审计的训练日志。
前置依赖：Stage 018隔离入口契约已通过；正式训练尚未授权；日志代码必须位于`src-v3`内部。
状态：日志模块、独立Smoke和真实正式训练日志均已存在；新输出契约已通过GPU best checkpoint val复核，正式训练未因日志格式变化重跑。
最小子任务：

- [x] 在`src-v3/radar_ln3/logging_utils.py`实现标准库日志模块，避免handler重复，支持文件和控制台输出。
- [x] 将日志写入当前artifact目录的`training.log`；控制台日志走stderr，stdout保留机器可解析的训练summary JSON。
- [x] 输出run启动、数据/契约、训练计划、周期进度、epoch摘要、checkpoint、early stopping、失败和最终summary日志。
- [x] 设计固定格式：`时间 | 级别 | event | key=value`，进度包含epoch、step、global step、loss、grad、lr、elapsed和ETA；默认每epoch约10次且最多每50 step输出一次。
- [x] 新增`src-v3/smoke/logging_test.py`，验证日志格式、文件写入、duration格式和handler去重。
- [x] 为既有best epoch结果增加迁移入口，写入指标日志和完整`sample_detail_information.json`；旧summary保留为`training-summary-legacy.json`。
- [x] 在可借用GPU资源窗口运行GPU best checkpoint val，确认新评估summary字段和样本明细输出；不因格式修改重训。

验证命令：全量`src-v3`编译；`python -m smoke.logging_test`；`python -m smoke.static_isolation_check --root F:\\Project\\RadarInsight\\src-v3`；`python -m train.formal --help`和`python -m report.classification --help`。
产物：`src-v3/radar_ln3/logging_utils.py`、`src-v3/smoke/logging_test.py`；正式训练运行时生成artifact目录下的`training.log`。
验收条件：日志不依赖第三方库；不重复注册handler；训练日志不污染stdout JSON；日志不逐batch刷屏；正式训练、数据重建和配置版本不因本次代码增加而改变。
是否需要数据重建：否。
是否需要smoke：需要，日志单元Smoke已通过；真实trainer集成Smoke待GPU空闲。
是否需要正式训练：否，已有正式训练结果；不因日志/报告格式修改自动重训。
阻塞条件：test和P95仍未执行；训练日志新格式未通过一次新的全量训练运行验证，但不影响本次best checkpoint评估。

## Stage 022 - 正式best checkpoint性能评估与瓶颈分析

目标：基于真实V1 main validation结果，分别报告窗口级、parent级和目标级稳定结果，定位五分类混淆、低置信度和TR/RD质量瓶颈，并形成下一阶段单变量优化路线。
前置依赖：Stage 019正式训练完成；best checkpoint、V1 index、V2数据契约和train-only统计hash一致；test未授权时只评估val。
最小子任务：

- [x] 在GPU上独立加载`best_model.pt`并运行V1 main `val`。
- [x] 核对window/parent/stable三层指标、五类指标、混淆矩阵和Drone/Bird双向混淆。
- [x] 统计低置信度记录、窗口长度、RD缺失、TR missing、真实dt和拒识候选；不把阈值诊断当作拒识验收。
- [x] 区分已实测结论、工程判断和待验证优化假设，更新瓶颈与放行状态。

验证命令：`Set-Location F:\Project\RadarInsight\src-v3; & ..\\.venv\\Scripts\\python.exe -m report.classification --project-root F:\Project\RadarInsight --config F:\Project\RadarInsight\\config\\RADAR-LN3\\RADAR-LN3-V2-cq08-tr-rd-formal-supervised-config.json --checkpoint F:\Project\RadarInsight\\artifacts\\RADAR-LN3-V2-cq08-tr-rd-formal\\best_model.pt --split val --batch-size 1`。
产物：`artifacts/RADAR-LN3-V2-cq08-tr-rd-formal/evaluation-val.json`、`sample_detail_information.json`；本阶段不创建新数据、索引、manifest或配置版本。
验收条件：指标来自实际GPU forward；数据契约hash一致；test、session和端到端P95未执行项明确记录；优化方案保持单一主要变量和可回滚。
是否需要数据重建：否。
是否需要smoke：否，Stage 011/018契约Smoke已通过；仅进行了独立评估。
是否需要正式训练：否，本阶段不重训。
阻塞条件：验证集的Clutter=8、Unknown=15个parent支持数很小；session字段不在当前契约；端到端P95尚未测量；任何优化收益均待新实验验证。

## Stage 023 - 窗口监督与有效dt契约手术式修复

目标：在V2 baseline之外建立V3单变量实验链路，修复因果前缀监督目标不一致，并显式区分真实有效时间间隔、首帧无前驱和秒级时间量化导致的未知间隔。
前置依赖：Stage 022 GPU validation基线已冻结；当前最大RADAR-LN3版本为V2；V3不得覆盖V0/V1/V2文件或正式artifact。
最小子任务：

- [x] 读取并覆盖式更新三个跟踪文件，记录V3修复范围、版本、回滚边界和未执行正式训练。
- [x] 由V1 CQ08主索引生成V3点级索引，增加`tr_dt_valid`、`tr_dt_reason`和时间精度证据字段；仅使用CQ08，保留旧索引只读。
- [x] 在src-v3内部加入V3数据契约读取、Dataset/collate有效dt字段和模型dt_valid输入；V2模型/旧checkpoint兼容路径保持可加载。
- [x] 在src-v3内部加入按parent归一化的masked causal-prefix supervision；保留V2 final-window loss作为baseline分支。
- [x] 创建V3数据契约、manifest和正式配置；V2训练归一化统计内容未变时复用并记录hash。
- [x] 增加dt、prefix loss、V3 checkpoint schema和配置加载的focused测试。
- [x] 运行静态导入隔离、数据纯度/数量/split、shape/mask/finite、forward、loss、梯度、checkpoint恢复和V3短时smoke。

验证命令：`python -m build.causal_contract`；`python -m validate.causal_contract`；`python -m smoke.static_isolation_check`；V3 focused smoke与`python -m smoke.formal_entry_contract`的V3配置运行。
产物：`data/processed/RADAR-LN3-V3-*`契约文件、`config/RADAR-LN3/RADAR-LN3-V3-*`配置、src-v3内部dt/prefix实现和独立entry-contract/smoke artifact；V3正式训练artifact未生成。
验收条件：V3只使用CQ08；V3索引与V1 parent/split/label范围一致；非首点零dt均标记为无效而非伪造正间隔；模型显式消费dt_valid；V3 prefix loss只聚合非padding因果logits且每个parent等权；V2 checkpoint仍可按V2配置加载；全部focused/smoke通过。已满足。
是否需要数据重建：需要，限V3索引、manifest、契约和摘要；不重建RD tensor。
是否需要smoke：需要，短时、有界、独立artifact目录。
是否需要正式训练：不需要，本Stage正式训练未执行。
阻塞条件：V1 hash变化、CQ08纯度失败、V3导入依赖src、V2兼容性破坏、dt_valid未进入模型、prefix loss无法证明mask/parent归一化、checkpoint契约失败。当前均未触发。

## Stage 024 - Clutter/Unknown与拒识专项审查设计

目标：在窗口/dt修复后，建立不修改标签、不删除困难样本的Clutter/Unknown标签质量、覆盖、开放集和拒识可验证实验设计。
前置依赖：Stage 023 V3契约和smoke通过；V3模型尚无正式性能结论。
最小子任务：

- [x] 审计原始标签语义、parent/session可用性、类别覆盖、短序列、RD缺失和主要混淆方向。
- [x] 明确Unknown分类标签与abstain/reject动作的区别，确认当前链路没有正式reject输出。
- [ ] 设计仅使用train或grouped OOF校准的拒识指标和阈值来源；不读取test。
- [ ] 设计单变量的采样/损失/拒识头对照，保持V3数据、split、时间和时序模块不变。

验证命令：已执行CQ08 manifest/index/原始轨迹只读审计；正式训练、test、全量阈值搜索和生产切换未执行。
产物：审查报告、拒识基线设计和后续单变量配置草案（未创建正式配置）。
验收条件：标签问题、覆盖问题、类别不平衡、拒识语义和模型问题分开；不把液态复杂度作为Clutter/Unknown默认修复。
是否需要数据重建：待验证；若标签/过滤规则变化，必须递增版本并保留原始视图。
是否需要smoke：需要。
是否需要正式训练：不需要，本Stage不自动训练。
阻塞条件：标签来源无法核实、验证集支持不足且无法设计parent级评估、拒识阈值需要test信息。当前拒识校准仍未执行，Clutter/Unknown支持较小。

## Stage 025 - RADAR-LN3全链路审查收口

目标：复核V3窗口监督、有效dt、CQ08纯度、split隔离、Clutter/Unknown覆盖和拒识边界，归档只读审计入口，形成可执行的单变量实验设计；不自动启动正式训练或test评估。
前置依赖：Stage 023技术验收通过；Stage 024已完成标签和拒识只读审计部分。
状态：审查与focused验证已完成；V3正式训练、全量V3 validation、test、拒识校准和Liquid对照未执行。
最小子任务：

- [x] 复跑V3全量编译、src-v3静态隔离、CQ08纯度/split、dt/prefix、缺失RD、forward/loss/梯度/checkpoint和CUDA entry-contract smoke。
- [x] 统计V3序列长度、跨split重复、标签冲突、按split/类别的dt有效性与RD缺失覆盖；未读取模型test结果。
- [x] 核对V3训练的causal-prefix loss与评估的sequence_logits是否使用同一窗口单位；保留prefix完整标签语义作为待验证业务风险。
- [x] 记录checkpoint parent选优与window/stable验收之间的残余目标风险；首轮不与主要修复同时改变选优指标。
- [x] 核对当前拒识边界：UnknownTarget是普通类别，现有stable状态机没有reject/abstain输出；拒识指标和阈值未执行。
- [x] 将只读数据审计和拒识设计入口归档到src-v3/report并运行，输出不得覆盖数据、配置或正式artifact。
- [ ] 用户确认prefix标签业务语义、训练预算和评估范围后，才允许V3正式训练。

验证命令：V3全量编译；`smoke.static_isolation_check`；`validate.causal_contract`；`smoke.dt_prefix_test`；`smoke.missing_rd_mask_test`；`smoke.contract_tests`；`smoke.formal_entry_contract`；`smoke.latency_smoke`；归档后的`report.audit`。
产物：审查命令输出、V3审查记录、拒识实验设计；V3正式checkpoint和正式性能报告未生成。
验收条件：V3输入仍为CQ08-only；无src运行时依赖；窗口监督和dt契约通过；不存在未记录的Critical泄漏；90%指标、拒识指标和端到端P95均不以Smoke替代。
是否需要数据重建：否；V3索引、manifest、split、数据契约、配置和V2训练统计均复用。
是否需要smoke：需要，已完成focused smoke；归档审计入口运行待完成。
是否需要正式训练：需要用户确认后手动运行，当前未执行。
阻塞条件：prefix标签业务语义尚未由项目证据确认；Clutter/Unknown验证parent支持仅8/15；正式拒识动作、test评估和端到端P95尚未执行。

## Stage 026 - CQ08原始回波抽样审计

目标：只读验证CQ08原始.dat回波、现有MAT/RD中间产物与点迹/航迹的帧序、时间和配对关系，为是否重解析原始回波提供证据；不覆盖现有MAT、RD、索引、配置或正式artifact。
前置依赖：Stage 025审查记录已读取；CQ08原始回波、点迹、航迹、MAT和统一manifest路径已确认；MATLAB可执行文件已确认存在。
最小子任务：

- [x] 读取现有V3跟踪文件、V3数据契约、统一manifest、batch_process_all.m和现有MAT读取链路。
- [x] 固定12个CQ08样本选择规则，覆盖首尾track边界、五类标签和不同序列长度。
- [x] 创建src-v3/tests/processed/original_echo_audit.m，只读解析.dat帧并对照MAT/RD/TR映射；最终脚本SHA-256为C2B48E4FE56328CDAF62358CD0CC7FEEF8B3E27AD8B427774F79492EB2DD626F。
- [x] 运行MATLAB抽样审计；实际审计12个CQ08文件，报告写入src-v3/processed/report/，未写入data/或正式artifact。
- [x] 核对原始I/Q、CPI/PRT、帧数、MAT帧数、index_trackPointNo、点迹/航迹时间和首/中/末帧RD重算误差；12/12样本状态为ok，1023个原始帧与MAT数量一致，36个抽查对照全部匹配。
- [x] 依据真实报告判断是否需要全量原始回波审计或独立重解析实验；当前不需要为修复MAT/RD错配而重解析或重建V3训练数据；全量一致性审计未执行，若研究未压缩I/Q或原始采集时钟，再另行授权独立实验。

验证命令：用户在MATLAB中执行`cd('F:\Project\RadarInsight'); addpath(fullfile(pwd,'src-v3','tests','processed')); outputs=original_echo_audit(pwd);`；PowerShell核对src-v3/processed/report/报告文件和JSON/CSV内容。
产物：src-v3/tests/processed/original_echo_audit.m、src-v3/processed/report/RADAR-LN3-original-echo-audit-summary.json、抽样/帧级CSV报告（实际运行后生成）。
验收条件：正好12个CQ08文件被审计；样本选择、原始帧解析、MAT对照、TR点索引和失败原因有可追溯记录；原始帧数/MAT数、I/Q、RD、索引和时间覆盖结果已记录；未执行的全量结论明确标为待验证；没有修改既有数据集、MAT、RD、配置或训练代码。
是否需要数据重建：不需要；仅生成诊断脚本和报告。
是否需要smoke：需要，限MATLAB脚本语法、抽样审计和报告结构检查。
是否需要正式训练：不需要，未执行。
阻塞条件：MATLAB不可执行、原始帧格式无法稳定解析、MAT与原始帧无法建立可复核映射或报告输出路径不可写。

## Stage 027 - V4无效dt观测更新单因素实验

目标：在V3数据、TR/RD输入、标签、split、causal-prefix loss和RD空间编码完全不变的条件下，仅替换无效TR dt事件的液态状态更新策略，验证显式`tr_dt_valid`是否可以转化为有效的观测更新；V3 floor行为保留为可回滚对照。
前置依赖：V3数据契约和配置hash已核对；V3静态隔离、dt/prefix Smoke和CQ08契约校验通过；V3正式训练artifact不存在；GPU可用但本阶段不自动启动正式训练。
状态：Stage 027已完成代码和focused Smoke；正式训练与性能验证未执行。
最小子任务：

- [x] 读取`data/`、`config/RADAR-LN3/`和`src-v3/process_v3.md`，确认当前最高版本为V3、复用文件hash和V3正式artifact不存在。
- [x] 在`src-v3`内增加可配置的无效dt更新模式；V3默认保持`floor`，V4仅启用有界观测事件增益。
- [x] 创建V4正式配置；复用V3 index/contract/split和V2 train-only归一化统计，使用独立V4 artifact和checkpoint schema。
- [x] 增加V4 invalid-dt focused Smoke，覆盖模式、增益、mask、有限值、有效dt行为和checkpoint契约。
- [x] 运行V4配置加载、静态导入隔离、全量编译、shape/mask/forward/loss/梯度、checkpoint恢复和有界entry Smoke。
- [x] 记录V4达到短时Smoke放行；正式V4训练、全量validation、test、拒识校准、Liquid对照和端到端P95保持未执行。

验证命令：`python -m smoke.static_isolation_check --root F:\\Project\\RadarInsight\\src-v3`；`python -m smoke.invalid_dt_update_test --config F:\\Project\\RadarInsight\\config\\RADAR-LN3\\RADAR-LN3-V4-cq08-tr-rd-invalid-dt-event-formal-supervised-config.json`；`python -m smoke.formal_entry_contract --project-root F:\\Project\\RadarInsight --config F:\\Project\\RadarInsight\\config\\RADAR-LN3\\RADAR-LN3-V4-cq08-tr-rd-invalid-dt-event-formal-supervised-config.json`；`python -m validate.causal_contract --project-root F:\\Project\\RadarInsight`。
产物：V4配置、V4无效dt focused Smoke和隔离entry-contract artifact；不生成新的index、manifest、split、RD tensor或正式checkpoint。
验收条件：V3数据hash全部复用且未被覆盖；V4无效dt模式显式、有界、因果并保留`dt_invalid_mask`；有效dt路径与V3一致；静态隔离和focused Smoke通过；没有正式训练或性能结论。
是否需要数据重建：否；数据范围、字段、shape、标签、split和归一化统计均不变。
是否需要smoke：需要；只允许focused和有界entry Smoke，产物与正式artifact隔离。
是否需要正式训练：本Stage不需要；只有Smoke通过且用户再次确认后才输出V4正式训练命令。
阻塞条件：V4无法复用V3数据契约、模式改变有效dt行为、增益无界或导致非有限输出、src隔离失败、Smoke写入正式artifact或正式训练被误启动。

## Stage 028 - V3/V4 paired正式监督训练

目标：在相同CQ08 V3数据、split、seed、batch、训练轮数、early stopping、causal-prefix loss和validation parent Macro-F1选优规则下，分别训练V3 floor对照与V4 bounded-observation候选，获得可归因的无效dt更新对照。
前置依赖：Stage 027代码隔离、配置加载、V3/V4真实batch contract、loss/梯度、checkpoint、entry Smoke已通过；V3和V4正式artifact目录为空；用户已明确授权两套正式训练。
状态：V3/V4均已由用户手动完成正式训练，paired validation summary已读取并形成F050结论；独立best-checkpoint复核与test仍未执行。
最小子任务：

- [x] 核对V3/V4配置路径、数据路径、seed、batch、epoch、patience、loss、选优指标、`evaluate_test=false`和artifact路径。
- [x] 核对V3/V4正式artifact目录不存在，避免覆盖已有正式checkpoint。
- [x] 向用户输出V3和V4串行正式训练命令；命令不含`--max-steps`、`--epochs`覆盖或`--skip-validation`。
- [x] 用户手动完成V3正式训练并保存best/last checkpoint、training-summary和training.log。
- [x] 用户手动完成V4正式训练并保存best/last checkpoint、training-summary和training.log。
- [x] 读取两套validation summary并比较parent总体与Clutter/Unknown类别结果；完整长度、dt_valid和RD缺失分层仍待独立评估，test保持未执行。
- [ ] 运行相同评估命令独立复核两套best checkpoint，并形成paired结论。

验证命令：V3和V4正式训练命令由用户手动执行；训练后使用`report.classification`分别评估`val`，不传`--split test`。
产物：`artifacts/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-formal/`和`artifacts/RADAR-LN3-V4-cq08-tr-rd-invalid-dt-event-formal/`中的正式checkpoint、summary、training.log和validation report。
验收条件：两套训练均实际完成或明确失败；两套使用同一V3数据hash、seed、训练预算和validation逻辑；V4只有在Macro-F1/弱类/Drone-Bird/稳定结果不恶化且资源可接受时才可保留。
是否需要数据重建：否；复用V3 index/contract/split和V2 train-only统计。
是否需要smoke：训练前Stage 027 Smoke已通过；训练后需要checkpoint加载和validation复核。
是否需要正式训练：需要，已获用户授权；由用户手动执行，本代理不自动启动。
阻塞条件：artifact目录非空且可能覆盖、数据hash不一致、训练入口或配置加载失败、GPU资源不足、训练产生非有限值、V3/V4无法使用同一评估契约。

## Stage 029 - 液态模型DBB/CU分层专项

目标：在不修改现有V3/V4数据、checkpoint和五分类主链路的前提下，围绕当前TR+RD液态模型建立独立的DBB/CU二分类gate、DBB三分类specialist和CU二分类specialist，验证其是否值得作为后续软级联模块候选。
前置依赖：V3/V4正式best checkpoint和validation结果已存在；V3索引、V3数据契约、V2 train-only归一化统计和V4液态无效dt更新行为已核实；用户明确要求优先构建DBB/CU并暂不依赖RD-only模型细节。
状态：独立代码、V5任务配置和有界Smoke已完成；Stage 030确认V5输入含parent全局RD质量泄漏后，原“可手动训练准备”结论已撤销。V5仅保留为历史审计基线，正式训练与真实性能评估未执行。
最小子任务：

- [x] 读取V3/V4训练摘要、任务输入契约、当前液态模型结构和已有跟踪文件；确认V3/V4效果不能作为DBB/CU独立模型结果。
- [x] 冻结任务标签映射：DBB/CU=`Drone/Bird/Balloon`对`Clutter/Unknown`，DBB specialist=`Drone/Bird/Balloon`，CU specialist=`Clutter/Unknown`；保留原五类标签，不改标签、不删困难parent。
- [x] 在`src-v3/tests/cu/`和`src-v3/tests/dbb-cu/`建立独立任务入口，复用当前TR+RD输入契约和CfC风格液态时序骨干；未引用`src/`和既有五分类checkpoint。
- [x] 创建以`RADAR-LN3-V5`为前缀的三份任务配置；仅配置/模型输出契约发生变化，V3 index、V3 data contract、V2 train-only normalization stats按hash复用，未创建新的数据索引。
- [x] 完成DBB/CU gate、DBB specialist、CU specialist的shape、mask、有限值、forward、causal prefix loss、梯度和checkpoint round-trip Smoke；Smoke结果写入`src-v3/smoke/`。
- [x] 运行`src-v3`静态导入隔离检查和三项任务Smoke；已记录真实命令、hash、设备、batch shape、loss、梯度和checkpoint路径。
- [x] Smoke通过后曾输出手动正式训练命令；Stage 030已将这些命令标记为历史且禁止执行，正式训练、validation、test、软级联收益和90%指标均未执行。

验证命令：在`src-v3`工作目录执行`..\\.venv\\Scripts\\python.exe -m smoke.task_dbb_cu --project-root F:\\Project\\RadarInsight --config F:\\Project\\RadarInsight\\config\\RADAR-LN3\\RADAR-LN3-V5-cq08-liquid-dbb-cu-gate-config.json --parent-limit 6`、`..\\.venv\\Scripts\\python.exe -m smoke.task_cu --project-root F:\\Project\\RadarInsight --config F:\\Project\\RadarInsight\\config\\RADAR-LN3\\RADAR-LN3-V5-cq08-liquid-dbb-specialist-config.json --parent-limit 6`、`..\\.venv\\Scripts\\python.exe -m smoke.task_cu --project-root F:\\Project\\RadarInsight --config F:\\Project\\RadarInsight\\config\\RADAR-LN3\\RADAR-LN3-V5-cq08-liquid-cu-specialist-config.json --parent-limit 6`，并执行`..\\.venv\\Scripts\\python.exe -m smoke.static_isolation_check --root F:\\Project\\RadarInsight\\src-v3`。
产物：`src-v3/tests/liquid_tasks/`共享任务实现、`src-v3/tests/cu/`和`src-v3/tests/dbb-cu/`任务包、`src-v3/train/tasks.py`、`src-v3/report/tasks.py`；`config/RADAR-LN3/RADAR-LN3-V5-*.json`任务配置；`src-v3/smoke/RADAR-LN3-V5-*-contract-smoke/`有界Smoke结果；正式模型artifact不在本阶段自动生成。
验收条件：三项任务均只消费当时的液态TR+RD张量、真实dt/dt_valid、missing和padding mask；原五类标签和CQ08 split不变；无`src`运行时依赖；Smoke checkpoint可保存恢复；未把Smoke或V3/V4五分类后验聚合写成独立DBB/CU性能。历史工程验收已满足，但因Stage 030发现F057因果泄漏，V5不再具备训练放行资格。
是否需要数据重建：否；数据范围、字段、RD张量、split和归一化统计不变，复用V3数据契约。
是否需要smoke：需要，结果和临时checkpoint限`src-v3/smoke/`。
是否需要正式训练：本阶段不自动执行；需Smoke通过后由用户手动确认并运行。
阻塞条件：V3数据hash变化、任务标签映射无法由原五类标签确定、模型输入越权、隔离检查失败、任何shape/mask/梯度/checkpoint Smoke失败，或正式artifact路径与已有V3/V4产物冲突。

历史手动训练命令（本轮未执行；Stage 030后禁止执行，仅保留追溯）：

1. `Set-Location F:\Project\RadarInsight\src-v3; & ..\\.venv\\Scripts\\python.exe -m train.tasks --project-root F:\Project\RadarInsight --config F:\Project\RadarInsight\\config\\RADAR-LN3\\RADAR-LN3-V5-cq08-liquid-dbb-cu-gate-config.json`
2. `Set-Location F:\Project\RadarInsight\src-v3; & ..\\.venv\\Scripts\\python.exe -m train.tasks --project-root F:\Project\RadarInsight --config F:\Project\RadarInsight\\config\\RADAR-LN3\\RADAR-LN3-V5-cq08-liquid-dbb-specialist-config.json`
3. `Set-Location F:\Project\RadarInsight\src-v3; & ..\\.venv\\Scripts\\python.exe -m train.tasks --project-root F:\Project\RadarInsight --config F:\Project\RadarInsight\\config\\RADAR-LN3\\RADAR-LN3-V5-cq08-liquid-cu-specialist-config.json`

历史val命令格式：`Set-Location F:\Project\RadarInsight\src-v3; & ..\\.venv\\Scripts\\python.exe -m report.tasks --project-root F:\Project\RadarInsight --config <对应V5配置> --checkpoint <对应artifact>\\best_model.pt --split val`。该命令不得用于修复后结论；正式训练、val评估、软级联组合评估和test均未执行。

## Stage 030 - 因果质量、指标语义与液态软级联闭环

目标：先移除模型输入中的parent全局RD质量泄漏并修正Macro/stable语义，再实现三个独立液态任务checkpoint的联合五分类评估和逐目标在线软级联，最后只增加液态状态归一化与RD帧时序聚合的单因素消融开关；不启动正式训练。
前置依赖：Stage 029三项任务shape/loss/gradient/checkpoint Smoke已通过；V3/V4真实validation证明当前弱类未达标；用户明确授权按上述顺序实施；本轮代码范围严格限制在`src-v3/`。
状态：代码修复、联合离线/在线接口、互斥消融开关和focused契约测试已完成；V6数据契约、三份V6配置、正式checkpoint、真实联合validation、test、校准和P95均未执行，当前不能训练。
最小子任务：

- [x] 读取三个跟踪文件、当前Dataset/模型/训练/评估/online代码和工作树，确认用户已有改动并冻结本轮仅修改`src-v3/`。
- [x] 复现parent全局RD质量位进入每个prefix、缺失预测类别被Macro Precision移出分母、stable未确认状态被编码为Unknown，以及V5仅有组合函数而无联合运行链路。
- [x] 将模型消费的7维RD质量向量限定为当前事件可得信息，移除`partial_parent`和`parent_order_anomaly`两个未来派生输入；保留原索引字段用于离线审计，未重建或覆盖数据。
- [x] 修正五分类和专项任务的固定类别Macro指标，未预测类别按zero-division=0进入宏平均。
- [x] 将stable未确认状态表示为显式`None`/`stable_confirmed=false`，从稳定分类指标分母中分离并报告coverage；为概率软级联增加无二次softmax的稳定更新接口和输入模式锁。
- [x] 实现三配置/三checkpoint兼容性校验、联合五分类train/validation评估能力、组件输出诊断和独立报告入口；真实联合validation未执行。
- [x] 为专项任务模型实现单点online forward，并实现每目标三份液态状态、软概率组合、窗口结果和可空stable结果的在线包装器及reset边界。
- [x] 增加checkpoint输入/动力学/数据hash契约校验，防止旧非因果质量语义、错误index/contract/stats或不同液态/RD聚合模式被静默加载；V5模型执行被明确阻断，V6为下一正式版本。
- [x] 在前述修复通过后增加两个独立默认关闭的消融开关：液态归一化仅改变递归状态/读出边界；RD帧时序仅将点内masked mean替换为因果顺序聚合；代码拒绝同时开启。
- [x] 运行全量编译、src-v3静态隔离、因果质量、指标、stable、联合级联、在线逐点、checkpoint和两个消融开关的focused测试；正式训练、test和端到端P95保持未执行。
- [x] 依据真实命令结果覆盖式更新三个跟踪文件，未通过或未执行项未勾选为实验完成。

验证命令：`..\\.venv\\Scripts\\python.exe -m smoke.logging_test`；`..\\.venv\\Scripts\\python.exe -m smoke.report_detail_test`；`..\\.venv\\Scripts\\python.exe -m smoke.causal_cascade_contract_test --project-root F:\\Project\\RadarInsight`；`..\\.venv\\Scripts\\python.exe -m compileall -q .`；`..\\.venv\\Scripts\\python.exe -m smoke.static_isolation_check --root F:\\Project\\RadarInsight\\src-v3`；`..\\.venv\\Scripts\\python.exe -m report.cascade --help`。未运行`train.formal`或`train.tasks`正式训练。
产物：`src-v3/radar_ln3/`内因果质量、指标、stable、cascade和可切换消融实现；`src-v3/report/`联合评估入口；`src-v3/smoke/`契约测试；三个覆盖式跟踪文件。数据、配置和正式artifact本Stage不创建或覆盖。
验收条件：任何prefix不消费由未来parent范围计算的质量位；Macro固定类别分母正确；未确认stable不再等于Unknown；三模型联合概率有限且和为1；batch与逐点联合输出一致；状态可按target重置；旧不兼容checkpoint被明确拒绝；两个消融变量可分别开启且默认保持baseline；无`src`运行时依赖。
是否需要数据重建：本Stage不需要，旧索引全局质量字段保留为离线审计字段但不进入模型；若后续冻结正式输入schema，必须在`data/`和`config/RADAR-LN3/`创建下一全局版本，本轮因目录范围限制未执行。
是否需要smoke：需要，限focused、短时、无正式artifact写入。
是否需要正式训练：本Stage不需要；后续必须先创建V6数据语义契约和三份V6配置并通过有界Smoke，再由用户手动重训三个专项模型。当前未执行。
阻塞条件：无法在不读取未来事件的情况下构成RD质量向量；三个task配置/数据hash不一致；checkpoint缺少可验证输入契约；联合离线与逐点输出不一致；任何隔离、有限值、mask、概率和reset测试失败。

## Stage 031 - V6因果数据契约、专项配置与有界Smoke

目标：在保留V3 index和V2 train-only归一化统计的前提下，创建V6修复后因果RD质量数据契约、Gate/DBB/CU三份V6任务配置，完成配置/hash/数据纯度校验和三项有界Smoke，为后续手动正式训练建立可验证入口。
前置依赖：Stage 030代码修复、checkpoint绑定、联合级联和focused测试通过；用户以“继续”确认推进V6契约和配置创建；已核对data/config/process中的最大版本和旧文件hash。
状态：V6契约、配置、校验和三项有界Smoke已完成；其后三个V6专项模型已由用户手动完成正式训练，并于2026-08-06使用相同CQ08 validation split完成组件和三checkpoint联合复核。test、开放集校准、正式消融、多seed和端到端P95仍未执行。
最小子任务：

- [x] 覆盖式读取并记录现有RADAR-LN3最大版本：data最高V3、config最高V5；确认V6未占用。
- [x] 创建V6-cq08-tr-rd因果质量数据契约，复用V3 index和V2 train-only统计，并记录三份输入hash。
- [x] 创建V6 Gate、DBB specialist、CU specialist配置，声明`contract_schema_version=RADAR-LN3-V6`、`rd_quality_schema=causal-current-event-v1`和V6 checkpoint/artifact路径。
- [x] 加载三份V6配置并运行CQ08纯度、index/contract/stats hash、字段、归一化统计和数据契约校验。
- [x] 运行三项V6有界Smoke，验证真实CQ08 Dataset、forward、causal-prefix loss、梯度、checkpoint round-trip和Smoke结果持久化。
- [x] 运行compileall、静态src-v3隔离和V6配置入口帮助检查。
- [x] 覆盖式更新三个跟踪文件；Stage 031当时未启动训练，后续V6正式训练和validation的实际结果已在Stage 032覆盖修正；test和正式消融仍未执行。

验证命令：`..\\.venv\\Scripts\\python.exe -m smoke.task_dbb_cu --project-root F:\\Project\\RadarInsight --config F:\\Project\\RadarInsight\\config\\RADAR-LN3\\RADAR-LN3-V6-cq08-liquid-dbb-cu-gate-config.json --parent-limit 6`；两项`smoke.task_cu`分别使用V6 DBB/CU配置；`..\\.venv\\Scripts\\python.exe -m compileall -q .`；`..\\.venv\\Scripts\\python.exe -m smoke.static_isolation_check --root F:\\Project\\RadarInsight\\src-v3`。
产物：`data/processed/RADAR-LN3-V6-cq08-tr-rd-causal-quality-data-contract.json`；`config/RADAR-LN3/RADAR-LN3-V6-cq08-liquid-*.json`；`src-v3/smoke/RADAR-LN3-V6-*-contract-smoke/`结果和临时checkpoint；三份跟踪文件。
验收条件：V6配置实际加载；配置声明的数据契约hash与实际文件一致；数据契约为CQ08-only且质量字段只含当前事件信息；三项Smoke通过；旧正式V5 artifact未覆盖，旧V5 Smoke目录曾被硬编码路径覆盖且已写完整性警告；正式训练仍未执行。
是否需要数据重建：不需要；只创建V6数据契约引用，V3 index内容和V2统计按hash复用。
是否需要smoke：需要，限三项有界Smoke和静态检查。
是否需要正式训练：Stage 031本身未自动启动；后续已由用户手动完成V6三项正式训练。
阻塞条件：V6版本冲突、契约hash不一致、数据纯度/字段校验失败、Smoke无法加载真实CQ08数据、checkpoint契约失败或发现旧文件被覆盖。

## Stage 032 - V6性能归因与V7/V8手术式修复

目标：冻结V6为只读性能基线，先解除`tr_dt_valid`作为分类特征造成的采集域捷径，再将checkpoint选优从平均指标改为弱类Precision/Recall下限优先；建立与`UnknownTarget`闭集类别正交的reject/abstain接口，但在train grouped-OOF校准完成前禁止启用正式拒识。
前置依赖：V6 Gate、DBB specialist、CU specialist正式checkpoint和training-summary存在；同一CQ08 validation split组件评估与三checkpoint联合评估已完成；test未读取。
状态：Stage 032技术实现与Smoke已完成；其后V7/V8六个专项模型已由用户手动完成正式训练，训练内validation summary和sample detail已生成。V7/V8三checkpoint联合validation、test、train parent-grouped OOF校准、多seed和P95仍未执行。
最小子任务：

- [x] 读取三份V6 checkpoint、summary、sample detail、配置、V6契约和训练历史，核实组件与联合五分类性能。
- [x] 证明阈值不能单独达到门槛：同一validation上的Gate/CU乐观阈值诊断均未达到目标；诊断阈值不保存、不部署。
- [x] 统计`tr_dt_valid`类别相关性、RD缺失、序列长度、置信度、稳定结果和液态时间常数，区分采集域捷径、概率校准、表征和液态动力学问题。
- [x] 核对全局版本：正式data/config最高为V6，V7只存在于旧focused测试的未来版本占位且没有正式文件；首个修复配置使用V7。
- [x] 实现`dt_valid`仅控制液态状态更新、不再直接拼入分类融合特征的显式模型契约，并保持batch/online一致。
- [x] 创建V7三任务配置，仅改变`dt_valid`分类特征暴露语义；复用V3 index、V6数据契约和V2 train-only统计，使用独立V7 artifact。
- [x] 实现弱类`min(Precision, Recall)`主指标、Macro-F1并列判据和可审计checkpoint选择记录。
- [x] 创建V8三任务配置，仅在V7模型契约上改变checkpoint选优语义；使用独立V8 artifact。
- [x] 实现独立reject/abstain输出契约和未校准保护；`UnknownTarget`分类标签不得被覆盖为未确认或reject动作。
- [x] 运行编译、静态隔离、配置/hash、shape/mask、batch/online、loss、梯度、checkpoint恢复、选择器和拒识保护focused测试。
- [x] 修复专项训练结束日志，输出parent/window总体与逐类Precision、Recall、F1，只在日志打印置信度低于0.60的样本，完整样本仍写入`sample_detail_information.json`。
- [x] 运行V7和V8三任务CPU有界Smoke，验证真实CQ08读取、有限loss/梯度和checkpoint round-trip；CPU结果不能代替CUDA放行证据。
- [x] 运行V7和V8三任务短时CUDA Smoke；不得启动正式训练、test、OOF阈值搜索或生产切换。
- [x] 依据当前真实输出覆盖更新三份跟踪文件；CUDA Smoke通过后允许用户手动paired正式训练。
- [x] 用户手动完成V7/V8 Gate、DBB specialist和CU specialist共六项正式训练；六个正式artifact均包含best/last checkpoint、training-summary、training.log和sample detail。
- [x] 读取六份training-summary和best checkpoint对应的parent/window组件validation结果，核对V7/V8选优差异和弱类结果。
- [ ] 分别运行V7与V8三checkpoint联合validation，冻结同口径五分类parent/window/stable结果；当前未执行。

验证命令：`..\\.venv\\Scripts\\python.exe -m compileall -q .`；`..\\.venv\\Scripts\\python.exe -m smoke.static_isolation_check --root F:\\Project\\RadarInsight\\src-v3`；`..\\.venv\\Scripts\\python.exe -m smoke.v7_v8_repair_contract_test --project-root F:\\Project\\RadarInsight`；`..\\.venv\\Scripts\\python.exe -m smoke.logging_test`；CPU Smoke通过设置`CUDA_VISIBLE_DEVICES=-1`分别运行`smoke.task_dbb_cu`和`smoke.task_cu`的六份V7/V8配置；CUDA Smoke命令`..\\.venv\\Scripts\\python.exe -m smoke.v7_v8_cuda_smoke --project-root F:\\Project\\RadarInsight --parent-limit 6`已执行并通过。
产物：`src-v3/radar_ln3/`内修复实现、`src-v3/smoke/`内focused测试及Smoke产物、`config/RADAR-LN3/RADAR-LN3-V7-*.json`与`RADAR-LN3-V8-*.json`；不修改V6配置/checkpoint，不重建V3 index或RD张量。
验收条件：V7只改变`dt_valid`分类特征暴露；V8只改变checkpoint选择；V6/V7/V8 checkpoint不能混用；拒识未校准时不能输出已启用的reject动作；所有focused测试和六项CUDA Smoke通过。技术验收已满足，后续六项正式训练也已完成；联合validation与校准仍未完成。
是否需要数据重建：否；数据范围、标签、split、TR/RD张量和归一化统计不变。
是否需要smoke：需要，限`src-v3/smoke/`。
是否需要正式训练：需要，已由用户手动完成；本代理未自动启动训练。
阻塞条件：V7/V8联合validation、train parent-grouped OOF校准、test、多seed和P95仍未执行；正式拒识仍不可启用。

## Stage 033 - DBB/CU因果运动与RD物理特征审计

目标：验证“DBB具有明显运动特征、Clutter缺少运动特征”是否符合CQ08实际数据，并判断V7/V8 Gate弱项来自信息不存在、现有特征未显式表达、采集质量捷径还是RD物理结构被压缩；本阶段只做只读统计和轻量探针，不修改模型、数据契约、索引或配置。
前置依赖：V7/V8六个正式checkpoint及其training-summary/sample detail存在；V3 CQ08点级索引、V6因果质量契约和V2 train-only统计hash保持不变；test模型评估未授权。
状态：Stage 033只读特征审计已完成；报告已生成。没有创建新数据版本、配置或checkpoint，没有启动训练。
最小子任务：

- [x] 核对当前14维TR点值、11维missing mask、7维当前事件RD质量、RD空间编码和点内`masked_mean`聚合的实际契约。
- [x] 在CQ08 train+validation的1313个parent上统计速度、径向速度、几何位移、加速度、转向率、路径直线度、符号翻转、dt有效率和首点零值；test不进入最终统计。
- [x] 用train拟合、完整validation评估的轻量线性探针对比点值池化、运动水平、运动变化、几何/SNR和采集质量信息；探针仅判断可分信息，不作为模型性能或阈值。
- [x] 对V7 Gate的Bird正确/错误路由parent进行TR运动条件分析，确认Bird→CU是否等同于“Bird没有运动”。
- [x] 对确定性分层train parent和完整232个validation parent读取现有物理RD张量，统计速度带宽、零多普勒能量、谱熵、峰值集中度、相位相干性、TR/RD一致性和事件间变化；每个TR事件仍遵守最多12帧映射。
- [x] 记录两次失败的临时RD探针：一次因60秒执行上限终止，一次因首个事件RD路径为空；两次均未写文件、未修改数据，随后修正为按首个`rd_present=true`事件读取。
- [x] 形成结论：Clutter不是静止紧凑类，而是低路径一致性、高转向/径向变化的伪运动航迹；Unknown与Bird存在真实运动重叠；现有特征未显式表达运动一致性，RD前端未保留物理坐标和点内帧序。
- [x] 生成`src-v3/processed/report/RADAR-LN3-V7-V8-dbb-cu-feature-audit.md`并覆盖式更新三份跟踪文件。
- [x] 用户随后确认继续，TR动态特征已在Stage 034作为单因素实现；RD物理统计分支和DBB knownness头仍未实现。

验证命令：使用`.venv\\Scripts\\python.exe -`执行只读NumPy/Pandas/PyTorch统计脚本；`Get-FileHash -Algorithm SHA256`复核V3 index、V6 contract和V2 stats；未运行`train.tasks`、`report.* --split test`或正式阈值搜索。
产物：`src-v3/processed/report/RADAR-LN3-V7-V8-dbb-cu-feature-audit.md`以及覆盖更新后的三个跟踪文件；没有新增或修改`data/`、`config/RADAR-LN3/`、模型代码或正式artifact。
验收条件：运动与采集质量证据分离；所有探针使用train拟合、validation评估且不读取test性能；parent全局统计明确只作诊断，不作为在线因果输入；未把线性探针写成正式模型结果。已满足。
是否需要数据重建：否，本阶段未改变数据范围、字段、标签、split、特征schema或归一化统计。
是否需要smoke：否，本阶段没有运行时代码修改；只读统计已完成有限值和样本范围检查。
是否需要正式训练：否，未执行。
阻塞条件：validation的Clutter/Unknown support仅8/15，RD探针train只使用确定性分层子集，结论需要后续单因素模型实验和多seed验证；Unknown开放集语义仍没有真实外部未知样本验证。

## Stage 034 - V9因果TR运动一致性Gate单因素验证

目标：冻结V8 DBB/CU Gate为paired baseline，只增加由当前及历史TR点计算的因果运动一致性特征与逐特征valid mask，验证显式运动表征能否改善DBB/CU路由；不把“有运动”直接等同DBB，不改变Unknown闭集类别或未校准拒识语义。
前置依赖：Stage 033已确认Clutter是伪运动而非无运动类、Unknown与Bird运动分布重叠、当前14维TR缺少显式动态量；用户已确认继续验证特征问题；V3 index、V6因果RD质量契约和V2 train-only统计保持只读。
状态：技术实现、全量train/validation特征审计、focused测试、历史回归和CPU/CUDA有界Smoke均已完成；用户已于2026-08-07手动完成V9 Gate正式训练，best checkpoint为epoch 7。V8/V9同一validation split的paired复核和Stage 035 post-hoc motion归因均已完成。test、多seed、OOF拒识校准、P95和正式三checkpoint级联仍未执行。
最小子任务：

- [x] 核对全局最大版本为V8，确认V9配置、数据契约和正式artifact均不存在；复核V3 index SHA-256=`a5d054f044709047938c88ef5601caf7d4b482ef980a5a61f19db87b6e649f87`、V2 stats SHA-256=`584eddcaeb8d78677f0f4902d622c2aa313d9374b767a47982c12dc058f826b6`。
- [x] 冻结单因素边界：仅V9 `dbb_cu_gate`增加因果TR运动一致性输入；RD前端、液态单元、标签、split、loss、class weight、训练预算、checkpoint选择和V8 DBB/CU specialists不变。
- [x] 冻结9维逐特征定义、物理单位、固定缩放、valid条件、首事件语义、环状角差和在线累计状态；所有依赖时间的量在`dt_valid=false`或`dt<=0`时均无效且值严格为0。
- [x] 创建V9数据/输入契约与Gate配置；复用V3 index和V2统计，不重建索引或RD张量；契约SHA-256=`a7cdee8696983006bdfa7dd8d1c1d0a69684bd59a6ca88cf742e2d7d9f30c074`，配置SHA-256=`ac2722b810a4557a6e72ddfbbd956393783b0ca1c4154281d01f78af93fda7a3`。
- [x] 在`src-v3/radar_ln3/`实现因果TR运动特征、显式valid mask、Dataset/collate、离线forward和在线单点状态接口；valid mask仅用于数值门控，不直接拼入分类特征。
- [x] 将运动特征schema、维度、名称、mask语义写入checkpoint契约；真实V8 best checkpoint回归加载通过，V8→V9及伪造schema后的V8→V9均被拒绝。
- [x] 在`src-v3/smoke/`增加单点、双点、359°→0°、重复/负/无效dt、missing、padding、极短/长间隔、未来点扰动、状态重置、shape/finite/mask、在线/离线、forward/loss/gradient/checkpoint round-trip测试。
- [x] 对CQ08 train+validation共1313 parent、25158事件执行全量运动特征审计；test未读，未拟合阈值或分类器；有限值、无效零值、首点和裁剪率契约通过。
- [x] 运行compileall、静态导入隔离、V6-V8历史回归、V9 CPU和CUDA有界Smoke；所有Smoke产物只写入`src-v3/smoke/`。
- [x] 冻结并交付用户手动执行的V9正式训练和validation命令；本代理未自动启动正式训练、test、OOF阈值搜索、多seed或P95。
- [x] 用户手动完成V9 Gate正式训练，并使用同一validation split与V8 Gate进行parent/window/逐类paired对照；V8/V9 parent矩阵分别为`[[192,17],[9,14]]`和`[[198,11],[6,17]]`，V9 validation Parent Macro-F1=`0.812752`。
- [x] V9优于V8后，在同一V9 best checkpoint上执行全零、保留valid位置的train-only中位数模板和3个跨parent扰动归因；确认真实motion数值有贡献，但类别相关有效率/零模式保留了大部分V8→V9增益，不能把V9收益主要归因于物理运动量。

验证命令：已执行`..\\.venv\\Scripts\\python.exe -m build.motion_gate_contract --project-root F:\\Project\\RadarInsight`、`..\\.venv\\Scripts\\python.exe -m report.motion_features --project-root F:\\Project\\RadarInsight`、`..\\.venv\\Scripts\\python.exe -m smoke.motion_feature_contract_test --project-root F:\\Project\\RadarInsight`、`..\\.venv\\Scripts\\python.exe -m smoke.v7_v8_repair_contract_test --project-root F:\\Project\\RadarInsight`、`..\\.venv\\Scripts\\python.exe -m smoke.causal_cascade_contract_test --project-root F:\\Project\\RadarInsight`、compileall、静态隔离，以及`smoke.task_dbb_cu`的`--device cpu`和`--device cuda`有界Smoke。
产物：V9数据契约、V9 Gate配置、`src-v3/radar_ln3/`最小实现、`src-v3/report/motion_features.py`、`src-v3/smoke/motion_feature_contract_test.py`、训练前readiness快照、CPU/CUDA Smoke、全量特征审计，以及用户手动训练生成的`artifacts/RADAR-LN3-V9-cq08-liquid-dbb-cu-gate-motion-formal/`正式产物；没有创建V9 specialist配置或修改V8配置/checkpoint。
验收条件：未来点变化不影响历史前缀；无效dt不会生成伪物理导数；mask、padding、离线/在线状态和checkpoint契约一致；真实V8/V9 best checkpoint复核与归因检查通过。技术与validation归因验收已满足，但V9没有达到DBB/CU双类P/R门槛，且存在已确认的采样有效率捷径，不能进入正式级联或部署。
是否需要数据重建：不重建V3 index或RD张量；输入schema实质变化需要创建V9数据/输入契约。
是否需要smoke：需要，限`src-v3/smoke/`。
是否需要正式训练：V9单seed正式训练已由用户手动执行；预训练不需要且未使用。Stage 034收口时新的V10或其他模型训练尚未授权、未执行；V10后续状态见Stage 036/037。
阻塞条件：进入正式三层级联、性能达标或部署状态仍阻塞于有效率捷径修复、V9与specialists的版本/概率校准兼容、test、多seed、拒识OOF校准、流式stable复核和P95。

## Stage 035 - V9 Gate正式结果与motion归因审计

目标：核验用户提供的V9 best checkpoint，严格复现V8/V9同split validation，使用同一V9 checkpoint的受控输入干预区分真实运动数值贡献与`dt_valid`/motion零位置捷径，并明确其是否可进入软级联。
前置依赖：V8/V9 Gate正式best checkpoint、对应配置和V3 CQ08 validation均存在；Stage 034已冻结单因素训练契约；test保持未授权。
状态：已完成validation归因审计；结论为V9相对V8有真实单seed validation改善，但主要收益不能归因于目标自身物理运动数值，当前不能进入正式级联、test或部署。
最小子任务：

- [x] 核验V9正式artifact完整性、配置、checkpoint schema、模型/输入/data hash和best checkpoint SHA-256=`e6242ef9aced9b75e3cef80fa4f0a078fcf08b01e81755dfd1874ff07b5f2211`。
- [x] 在CUDA上独立复现V8/V9 best checkpoint的232 validation parent和4474 causal prefix window；两者parent混淆矩阵及总体指标与checkpoint内嵌记录逐值一致。
- [x] 建立基础五类路由、窗口/parent、序列长度、dt有效率、motion有效率和置信度分组；完整记录写入`src-v3/processed/report/`。
- [x] 对同一V9 checkpoint执行motion全零干预；parent最终预测退化为全部DBB，证明motion分支对CU路由必要，但该结果不能区分物理值与可用性模式。
- [x] 保留每个接收parent原valid位置，以现有train-only特征审计的逐特征中位数替换真实motion值；Parent Macro-F1=`0.782817`、CU F1=`0.608696`，分别保留V8→V9增益约64.86%和60.87%。
- [x] 保留valid位置和validation逐特征边际分布，使用3个固定seed将每个有效值分配自不同parent；Parent Macro-F1=`0.783231–0.798765`，均值保留V8→V9 Macro-F1增益约77.31%。
- [x] 追溯validation `tr_dt_reason`：非首帧无效dt全部来自`zero_delta_at_second_precision`；Drone/Bird/Balloon分别有687/450/427个，Clutter/Unknown均为0；不是padding、负时间或非有限时间。
- [x] 对照12文件原始回波MATLAB审计：原始`.dat`/现有MAT只有帧序和CPI/PRT局部时长，没有可恢复TR亚秒wall-clock；不得用RD CPI时间替代TR点间dt。
- [x] 运行新增报告入口编译、全量`src-v3` compileall和静态导入隔离；`violations=[]`，未修改数据、配置、checkpoint或正式训练代码。
- [x] 覆盖更新三个跟踪文件并保留训练前readiness JSON为历史快照；未把validation干预、单seed或低support结果写成test/部署性能。
- [ ] test、多seed重训、train parent-grouped OOF拒识校准、正式三checkpoint级联、线上stable和端到端P95未执行。

验证命令：在`src-v3`执行`..\\.venv\\Scripts\\python.exe -m report.motion_gate_attribution --project-root F:\\Project\\RadarInsight --device cuda --batch-size 2`、`..\\.venv\\Scripts\\python.exe -m compileall -q .`和`..\\.venv\\Scripts\\python.exe -m smoke.static_isolation_check --root F:\\Project\\RadarInsight\\src-v3`。
产物：`src-v3/report/motion_gate_attribution.py`；`src-v3/processed/report/RADAR-LN3-V9-motion-gate-attribution.json`与`.md`；覆盖更新后的三个跟踪文件。V8/V9正式artifact只读，数据与配置未创建或修改。
验收条件：直接评估精确复现两份best checkpoint；所有干预保持同一validation parent/split和同一V9权重；跨parent扰动不使用标签且不从同parent取值；invalid/padding值严格为0；test未读；归因边界和低support限制明确。已满足。
是否需要数据重建：否；本阶段只读复用V3 index、V9 contract和V2 train-only统计。
是否需要smoke：无需训练Smoke；报告入口通过真实全量validation、compileall和隔离检查。
是否需要正式训练：本阶段不训练；V9正式训练已由用户手动完成。任何新schema训练必须另行确认。
阻塞条件：V9 DBB Recall=`0.947368<0.95`，CU Precision/Recall=`0.607143/0.739130<0.90`；validation CU仅23 parent；motion可用性与标签强相关；窗口Clutter Recall相对V8下降；test、多seed、拒识、正式级联、stable和P95未执行。

## Stage 036 - V10无dt逐观测增量Gate单因素修复

目标：按用户选择的方案A，将V9 Gate中依赖`dt/dt_valid`的9维物理率特征替换为只依赖当前点和历史观测点的因果逐观测增量特征，消除秒级时间戳可用性对motion分支的直接控制；保持DBB/CU任务、CQ08数据行与split、基础TR/RD输入、液态动力学、loss、采样、训练预算和checkpoint选择规则不变。
前置依赖：Stage 035已确认V9大部分增益受motion有效位置/零模式支配，且当前原始回波不能恢复TR亚秒wall-clock；用户于2026-08-07明确选择方案A；V3 index、V6因果RD质量契约、V2 train-only统计和V9正式checkpoint保持只读。
状态：V10无dt逐观测增量schema、按schema选择的Dataset/模型/checkpoint链路、版本化契约/config、全量train+validation特征审计、focused测试、历史回归、CPU/CUDA有界Smoke、compileall和静态隔离均已完成；用户随后于2026-08-07手动完成V10 Gate正式训练，正式checkpoint复核和V9/V10 paired validation见Stage 037。V10未达到CU门槛，test、多seed、拒识校准、正式级联、stable与P95仍未执行。
最小子任务：

- [x] 核对全局最大版本为V9、V9 Gate配置/数据契约/正式checkpoint存在，并确认工作区含大量既有用户改动，本Stage不修改`src/`或其他实验线。
- [x] 冻结单因素边界：V10仅替换Gate派生motion schema及其输入/checkpoint身份；V3 index/split、V2 RD统计、14维TR、11维missing、7维因果RD质量、RD前端、CfC液态单元、loss、class weight、seed、batch、epoch、early stopping和选优规则保持不变。
- [x] 冻结无dt逐观测增量特征的名称、单位、固定缩放、valid规则、首点语义、环状角差、路径状态和在线reset边界；状态机函数签名只接收当前TR字段，不接收`dt`、`dt_valid`或时间戳，也不读取标签、ID、路径、split、RD或未来点。
- [x] 在`src-v3/radar_ln3/`实现按schema选择V9/V10状态机，保持V9配置与checkpoint可复现，并将V10 Dataset、离线forward和在线单点输入接入。
- [x] 创建下一个全局版本V10数据/输入契约与Gate配置；复用V3 index和V2统计，不重建索引、split或RD张量；生成器重复运行返回`reused_identical`，内容不同时拒绝覆盖。
- [x] 增加单点、359°→0°、missing、极值clip、未来前缀、状态reset、真实parent时间字段干预、padding、shape/finite/mask、在线/离线、梯度和checkpoint隔离focused测试。
- [x] 对CQ08 train+validation的1313 parent/25158事件执行V10特征审计；test未读取。8179个非首帧秒级零dt事件全部仍有合法增量，validation各类valid率最大差降至`0.00722476`，最大裁剪率=`0.00110072`。
- [x] 运行compileall、静态导入隔离、V9 focused回归、V7/V8修复回归、基础级联回归以及V10 CPU/CUDA有界Smoke；所有V10 Smoke产物只写入`src-v3/smoke/`，`violations=[]`。
- [x] 根据真实测试结果判定可进入用户手动V10 Gate paired正式训练；readiness明确禁止正式级联、test、拒识和部署，本代理未启动正式训练。
- [x] 用户手动完成V10单seed正式训练；Stage 037已核验正式artifact、独立复现validation并完成V9/V10 paired对照。该结果不改变Stage 036的单因素实现边界。

验证命令：已在`src-v3`执行`..\\.venv\\Scripts\\python.exe -m build.observation_increment_gate --project-root F:\\Project\\RadarInsight`（连续两次）、`-m smoke.observation_increment_contract_test`、`-m report.observation_increments`、`-m smoke.motion_feature_contract_test`、`-m smoke.v7_v8_repair_contract_test`、`-m smoke.causal_cascade_contract_test`、`-m smoke.task_dbb_cu`的CPU/CUDA命令、`-m compileall -q .`及`-m smoke.static_isolation_check --root F:\\Project\\RadarInsight\\src-v3`；均在最终代码上通过。
产物：V10数据/输入契约、V10 Gate配置、`src-v3/radar_ln3/`最小实现、生成器、focused测试、全量审计、CPU/CUDA Smoke及`RADAR-LN3-V10-gate-training-readiness.json`；用户后续手动训练已创建正式artifact，训练后产物由Stage 037核验。
验收条件：同一TR点序列在任意`dt/dt_valid`输入下产生完全相同的V10 motion值与valid mask；无未来信息；V9行为不变；真实batch shape/mask/forward/loss/gradient/checkpoint round-trip和src-v3隔离通过。技术验收已满足；Stage 037确认性能门槛未满足。
是否需要数据重建：不重建V3 index、split或RD张量；特征与checkpoint语义实质变化，需要V10数据/输入契约和配置。
是否需要smoke：需要，限focused与有界Smoke，写入`src-v3/smoke/`。
是否需要正式训练：V10单seed正式训练已由用户手动完成；预训练不需要且未使用。本Stage没有由代理自动启动训练。
阻塞条件：V10 Gate的CU parent Precision/Recall仅`0.739130/0.739130`，未达到0.90；进入正式软级联还缺同版本specialists、概率兼容和联合评估；进入test/拒识/部署仍阻塞于多seed、train parent-grouped OOF校准、stable时序与P95。逐观测量并非SI物理率，线上报告节奏域漂移仍需验证。

## Stage 037 - V10 Gate正式结果、paired复核与残差完整性审计

目标：核验用户提供的V10正式best checkpoint，独立复现同一CQ08 V3 validation，与V9进行逐window/parent paired对照，并检查最终Bird/Unknown残差parent的标签码、源文件、事件序和RD映射完整性；不读取test、不拟合阈值、不启动训练、不拼接异版本specialist。
前置依赖：V9/V10正式Gate artifact、V3 CQ08 index/split、V10数据契约和V2 train-only统计存在；Stage 036已冻结V10单因素边界；用户允许继续validation审计。
状态：正式checkpoint/hash与契约已核验；GPU独立validation、V9/V10 paired对照、Bird/Unknown残差特征描述、12个最终错误parent结构完整性审计、compileall和静态隔离均已完成。V10改善单seed validation但未达到CU门槛，不能进入正式Gate模块、三checkpoint级联、reject或部署。
最小子任务：

- [x] 核验V10正式artifact、配置、checkpoint schema及数据hash；best checkpoint SHA-256=`2973806b740094d6380468d3c95bff4b70986e6128764f12cb4c07316197dec1`。
- [x] 在GPU上独立复现V10的232个validation parent和4474个causal-prefix window；parent/window指标与训练内嵌best记录逐值一致。
- [x] 对V9/V10相同记录键执行paired检查；window/parent key、truth和不可变元数据完全一致，V9/V10评估报告精确复现。
- [x] 统计基础五类路由、prefix长度、训练epoch和验收门槛；20个epoch没有任何一个同时达到DBB P/R≥0.95和CU P/R≥0.90。
- [x] 对validation的BirdTarget与UnknownTarget执行TR逐观测增量和当前事件RD物理统计描述；不拟合阈值、分类器、归一化或模型参数。
- [x] 对12个最终错误parent核对manifest、文件名标签码、航迹/点迹长度、原始回波、全部声明MAT、事件连续性、RD帧唯一性和V3映射；全部结构检查通过。
- [x] 确认`cq08|track|333`的22个源RD帧只覆盖track point 16–22，V3逐帧映射与物理帧索引完全一致；这是当前源索引中的晚启动/部分覆盖，不是V3漏配。人工语义标签复核仍未执行。
- [x] 修正残差特征报告中`formal_training`元数据，区分“V10正式训练已由用户完成”和“本描述性审计未启动训练”。
- [x] 在最终代码上运行新增报告入口`py_compile`、全量`src-v3` compileall和静态导入隔离；均通过，`violations=[]`。
- [ ] test、多seed、train parent-grouped OOF拒识校准、V10同版本specialists、正式三checkpoint联合评估、线上stable和端到端P95未执行。

验证命令：在`src-v3`执行`..\\.venv\\Scripts\\python.exe -m report.tasks --project-root F:\\Project\\RadarInsight --config ..\\config\\RADAR-LN3\\RADAR-LN3-V10-cq08-liquid-dbb-cu-gate-observation-increment-config.json --checkpoint ..\\artifacts\\RADAR-LN3-V10-cq08-liquid-dbb-cu-gate-observation-increment-formal\\best_model.pt --split val --batch-size 2`、`-m report.gate_paired_evaluation --project-root F:\\Project\\RadarInsight`、`-m report.gate_residual_features --project-root F:\\Project\\RadarInsight`、`-m report.gate_residual_integrity --project-root F:\\Project\\RadarInsight`、`-m py_compile report\\gate_paired_evaluation.py report\\gate_residual_features.py report\\gate_residual_integrity.py`、`-m compileall -q .`及`-m smoke.static_isolation_check --root F:\\Project\\RadarInsight\\src-v3`。
产物：`src-v3/report/gate_paired_evaluation.py`、`gate_residual_features.py`、`gate_residual_integrity.py`；`src-v3/processed/report/RADAR-LN3-V9-V10-gate-paired-validation.json/.md`、`RADAR-LN3-V10-gate-residual-feature-audit.json`和`RADAR-LN3-V10-gate-residual-integrity-audit.json`；覆盖更新后的三份跟踪文件。没有创建或修改data/config/checkpoint。
验收条件：V10独立validation精确复现；paired不可变记录一致；残差结构审计通过；指标、低support、Unknown闭集/开集边界和未执行项明确；最终代码编译与隔离通过。已满足。Gate性能门槛未满足。
是否需要数据重建：否；本阶段只读复用V3 index、V10 contract、V2统计、物理RD帧索引和原始CQ08文件。
是否需要smoke：不需要训练Smoke；报告入口通过真实validation、结构审计、compileall和隔离检查。
是否需要正式训练：V10训练已由用户完成；本阶段不训练。任何V11 schema或多seed训练必须另行确认。
阻塞条件：CU parent P/R=`0.739130/0.739130<0.90`；window CU P/R=`0.624074/0.798578`；validation CU仅23 parent；Unknown开放集拒识、test、多seed、同版本specialists、stable和P95均未执行。V10只能保留为研究Gate checkpoint。

## Stage 038 - V10 Bird/Unknown残差人工语义复核证据包

目标：为V10最终误路由的6个BirdTarget和6个UnknownTarget parent生成离线、可追溯的人工复核证据包，集中展示源文件、标签码、TR轨迹、逐事件增量、RD物理统计、RD覆盖和V10因果prefix概率演化；程序不得自动改标签、删除样本或把模型预测当作标注结论。
前置依赖：Stage 037已确认12个残差parent的结构完整性并冻结validation范围；V3 index、CQ08 manifest、物理RD张量、V10 sample detail和正式checkpoint保持只读。
状态：自动证据准备、指标查看和用户语义裁决均已完成。用户确认Unknown异质性、静止/低速Bird及“无多方独立反证时保留当前标签”的规则后，12条均记录为`KEEP_CURRENT_LABEL`，原标签零改动；这表示当前证据不足以推翻标签，不等于独立证明标签正确。审核包、独立验证器、防覆盖和非法决定防护focused测试均通过。Stage 038收口时V11为`ready_for_design_pending_explicit_authorization`；其后完整实施准备已由Stage 041完成，正式训练仍未执行。
最小子任务：

- [x] 冻结范围为12个V10 validation最终错误parent，不读取test，不修改标签、data/config/checkpoint，不拟合阈值或分类器。
- [x] 设计人工复核字段、结构事实与自动风险提示边界；审核决定字段留空，由具备标签规范/采集上下文的人员填写。
- [x] 实现独立报告入口，复用V3映射计算逐事件TR增量和当前事件RD物理统计，并对齐V10每个prefix的CU概率。
- [x] 生成离线HTML总览、parent级索引、逐事件CSV、人工review模板和hash清单；每个parent可追溯到航迹、点迹、原始回波、MAT和RD tensor。
- [x] 每个parent生成TR/前缀演化图、prefix×物理速度RD谱图和聚合range-Doppler图；图只用于描述和人工复核，不参与训练或阈值。
- [x] 验证12个parent、232个事件、概率有限范围、217个RD-present/15个RD-missing事件和源路径；初始review人工字段为空，后续12条用户曲线观察仅写入`review_notes`，`review_decision/reviewed_semantic_label/reviewer/review_date`保持为空；运行py_compile、compileall和src-v3静态隔离，`violations=[]`。
- [x] focused契约测试证明复核包有效，且`manual-review.csv`一旦含人工输入，生成器会以`FileExistsError`拒绝覆盖。
- [x] 将用户对12个残差parent的观察与统一`P(CU)`逐prefix数据交叉核对并记录；确认5个Bird全程CU、2862/4171全程DBB、2908/3122/3154由CU单次转DBB、1547四次跨界、3581始终位于CU侧。
- [x] 调整review校验契约：允许非裁决备注，合法决定必须含审核人/日期，拒绝非法决定、孤立审核字段及`KEEP_CURRENT_LABEL`下的类别变更。
- [x] 对跨界前后做描述性TR/RD对照；2908后段RD带宽/零速能量/谱熵发生强变化，3122/3154没有相同统一方向，因此未把全部后期下降归因为液态遗忘或单一RD规则。
- [x] 根据用户确认的标签政策完成12条裁决并运行独立校验：12/12为`KEEP_CURRENT_LABEL`、0次标签修改；该时点仅解除“待人工裁决”阻塞，后续V11授权和实施见Stage 041。

验证命令：已在`src-v3`执行`..\\.venv\\Scripts\\python.exe -m report.gate_residual_review --project-root F:\\Project\\RadarInsight`、`-m validate.gate_residual_review`、`-m smoke.gate_residual_review_contract_test`、`-m py_compile report\\gate_residual_review.py validate\\gate_residual_review.py smoke\\gate_residual_review_contract_test.py`、`-m compileall -q .`和`-m smoke.static_isolation_check --root F:\\Project\\RadarInsight\\src-v3`；均通过。
产物：`src-v3/report/gate_residual_review.py`、`finalize_residual_review.py`、`src-v3/validate/gate_residual_review.py`、`src-v3/smoke/gate_residual_review_contract_test.py`、focused结果JSON，以及`src-v3/processed/report/RADAR-LN3-V10-gate-residual-manual-review/`审核包；裁决后`manual-review.csv` SHA-256=`5f8eff2749e5e32710d4bd3f9dc0397da4fe02c1b3a52a2bf1382cfe0b03195d`，summary SHA-256=`29dd794898e24b3df881caef14eab265f7d02dc879dbbd62d6f690a501596d19`；不写data/config/artifacts。
验收条件：12个残差parent均有完整、可离线查看和机器读取的证据；用户观察、机器复核、标签政策和人工决定明确分离；12条决定均保留原标签且通过hash/字段/防覆盖校验；不读取test；编译和隔离通过。已满足。
是否需要数据重建：否；只读复用现有V3 index和RD张量。
是否需要smoke：报告输出focused契约已执行并通过；未运行训练Smoke。
是否需要正式训练：否，未执行。
阻塞条件：人工裁决阻塞已解除；但`KEEP_CURRENT_LABEL`不是独立真值证明。该Stage不自动触发V11、开放集校准、训练或test；后续V11完整实施准备已由Stage 041单独授权完成，正式训练/test仍未执行。

## Stage 039 - V10残差TR/RD指标对比与人工查看准备

目标：在不修改标签、数据、配置和checkpoint的前提下，使用CQ08 V3 train parent分布作为描述性参考，为12个V10 Bird/Unknown残差parent生成可比的TR/RD物理指标表、类别参考范围、百分位热图和逐parent指标图，辅助人工语义复核。
前置依赖：Stage 038证据包已生成；本Stage执行时12个parent决定仍为空，后续已在Stage 040按用户政策完成；V3 index、物理RD tensor和V10残差事件CSV保持只读。
最小子任务：

- [x] 读取并核对现有Stage 038审核包、V3 train索引和现有残差特征定义；确认本阶段不读取test、不拟合阈值、不改标签。
- [x] 实现`src-v3/report/review_metrics.py`，只依赖src-v3内部模块和已确认第三方库。
- [x] 生成CQ08 train parent中位数参考表、12个残差parent指标长表、P05/P25/median/P75/P95参考范围和SHA-256清单。
- [x] 生成类别箱线图、同类参考百分位热图、质量指标图和12个逐parent指标图；图中明确标注诊断用途。
- [x] 使用`view_image`完成输出图人工查看，逐parent记录可核对事实、异常范围和仍需外部证据的事项；没有把模型分数或统计异常直接转成标签裁决。
- [x] 更新人工审核指南，给出指标、单位、CQ08 train参考范围、推荐比较顺序和合理解释边界。
- [x] 运行报告入口、compileall和静态导入隔离检查；训练、数据重建、配置生成、阈值搜索和test评估未执行。

验证命令：`cd F:\\Project\\RadarInsight\\src-v3; F:\\Project\\RadarInsight\\.venv\\Scripts\\python.exe -m report.review_metrics --project-root F:\\Project\\RadarInsight`；随后运行`F:\\Project\\RadarInsight\\.venv\\Scripts\\python.exe -m validate.gate_residual_review --project-root F:\\Project\\RadarInsight`、`F:\\Project\\RadarInsight\\.venv\\Scripts\\python.exe -m smoke.gate_residual_review_contract_test --project-root F:\\Project\\RadarInsight`、`F:\\Project\\RadarInsight\\.venv\\Scripts\\python.exe -m compileall -q .`和`F:\\Project\\RadarInsight\\.venv\\Scripts\\python.exe -m smoke.static_isolation_check --root F:\\Project\\RadarInsight\\src-v3`。
产物：`src-v3/report/review_metrics.py`；`src-v3/processed/report/RADAR-LN3-V10-gate-residual-manual-review/metrics/`下的CSV、PNG、Markdown和JSON报告；不创建RADAR-LN3数据/配置版本。
验收条件：12个parent和CQ08 train参考均有实际统计；P05-P95只作为同类描述性筛查范围，不作为硬标签阈值；图表可离线查看；报告、编译和隔离检查通过。指标图与人工查看记录已完成；正式语义裁决随后由Stage 040完成。
是否需要数据重建：否。
是否需要smoke：需要报告/编译/隔离focused检查，不需要模型训练Smoke。
是否需要正式训练：否，未执行。
阻塞条件：无法从V3 index重算同口径参考分布、RD映射不完整、图表生成失败或发现审核包被人工输入后禁止覆盖。

## Stage 040 - 残差标签裁决收口与Track 333原始回波聚焦审计

目标：落实用户对12个V10残差parent的标签裁决边界，在不改标签的前提下记录“保留当前标签”；同时以Python `struct`和独立MATLAB脚本双路径只读解析Track 333原始`.dat`，确认前15个TR点的RD是源端未采集、现有MAT遗漏还是索引解析问题。
前置依赖：Stage 038/039证据包、V3点级索引、CQ08 manifest、Track 333原始`.dat`和22个现有MAT均存在；用户确认Unknown内部包含风筝/昆虫等异质目标、Bird可静止、原始采集记录已丢失且当前默认以标签为准。
状态：已完成。12条正式裁决、Python `struct`全文件解析、不可变输入focused契约、MATLAB v2独立解析/RD重算、Python结果验证器和图像人工查看均通过。双路径一致确认该`.dat`只有point 16–22的22帧，前15点不可恢复；不重建数据，保留显式RD-missing mask和BirdTarget标签。
最小子任务：

- [x] 记录用户语义约束：运动强度不能判定Unknown属于DBB，低峰值速度不能否定Bird；无多方独立反证时保留当前标签。
- [x] 核对Track 333的22个TR点、22个现有MAT、原始`.dat`路径/大小以及V3仅在point 16–22映射22帧的事实。
- [x] 将12条人工决定以`KEEP_CURRENT_LABEL`写入审核表，保留“默认沿用标签、并非独立证明标签正确”的审计说明，并同步校验summary/hash。
- [x] 在`src-v3/report/`实现Track 333只读Python `struct`解析器，遍历全部帧和全部目标记录，输出帧偏移、长度、raw target identifier、原始point号、CPI/PRT、I/Q shape及与MAT/物理帧索引的对照。
- [x] 实际运行Python解析器并将CSV/JSON报告写入`src-v3/processed/report/`独立目录；未写入`data/`或覆盖现有MAT。
- [x] 在`src-v3/tests/processed/`实现Track 333独立MATLAB复核脚本，逐帧比较raw point号、MAT `index_trackPointNo`、I/Q和重算RD误差并输出对照图；用户已完成v1和修复后的v2手动运行。
- [x] 运行Python编译、报告focused契约和src-v3静态导入隔离；最终focused结果同时校验并记录MATLAB v2为passed。
- [x] 用户首次手动执行MATLAB独立复核：22/22帧I/Q和RD最大绝对/相对误差均为0，point 16–22 coverage一致；识别出空向量方向导致point 1–15 coverage假阴性。
- [x] 修复MATLAB空coverage比较并由用户原命令重跑；v2实际`status=passed`、`all_point_coverage_raw_matches_mat=true`，I/Q和RD最大误差均为0，结论与Python一致。
- [x] 运行`validate.track_0333_matlab_audit`并查看RD/coverage图；独立验证和人工图像核对均通过，生成Python+MATLAB交叉验证摘要。

验证命令：已在`src-v3`运行`F:\Project\RadarInsight\.venv\Scripts\python.exe -m report.track_0333_original_echo_audit --project-root F:\Project\RadarInsight`、`-m validate.track_0333_matlab_audit --project-root F:\Project\RadarInsight`、`-m smoke.track_0333_original_echo_audit_test --project-root F:\Project\RadarInsight`、`-m validate.gate_residual_review --project-root F:\Project\RadarInsight`、`-m smoke.gate_residual_review_contract_test --project-root F:\Project\RadarInsight`、`-m compileall -q .`和`-m smoke.static_isolation_check --root F:\Project\RadarInsight\src-v3`；均通过。用户在MATLAB中实际执行`track_0333_original_echo_audit(pwd)`，v2结果为passed。
产物：`src-v3/report/track_0333_original_echo_audit.py`、`finalize_residual_review.py`、`src-v3/smoke/track_0333_original_echo_audit_test.py`、`src-v3/tests/processed/track_0333_original_echo_audit.m`、Track 333独立报告目录、更新后的审核表/summary和三份跟踪文件；未创建RADAR-LN3数据/配置版本。
验收条件：12条裁决可追溯且未改标签；Python解析覆盖原始文件全部合法帧/目标记录；MATLAB脚本不覆盖现有MAT；raw/MAT I/Q、RD及point coverage双路径一致；原始point号与MAT/物理索引/V3映射一致；未执行事项如实记录。已满足。
是否需要数据重建：不需要且未执行；双路径已证明raw不含前15点，重建无法恢复这些帧，因此不创建新版本。
是否需要smoke：只需要报告、解析、编译和隔离focused检查，不需要训练Smoke。
是否需要正式训练：否，未执行。
阻塞条件：Stage 040无剩余阻塞。V11完整实施准备已于2026-08-08另行获用户授权，执行边界见Stage 041；正式训练、OOF拒识校准和test仍未授权执行。

## Stage 041 - V11当前事件RD物理谱paired实施准备

目标：在保持V10液态DBB/CU Gate、V3 CQ08 causal-prefix数据范围、split、标签、loss、采样语义和评估逻辑不变的前提下，建立V10重训Control与仅新增当前事件四维RD物理谱输入的V11 Treatment paired链路；完成版本化train/val缓存、在线同源计算、共享初始化、采样顺序审计、配置和有界Smoke，不启动正式训练。
前置依赖：Stage 040已完成12条残差标签裁决及Track 333原始回波双路径审计；用户已冻结四维特征、物理公式、缓存形式、三seed paired协议、20 epoch、选优规则、第一/第二阶段门槛，并授权完整实施准备但禁止代理启动正式训练。
状态：模型、数据内容和paired协议实施准备已完成；用户手动启动Treatment seed42/43时暴露的V11共享NPZ跨账户读取ACL缺陷已由Stage 042修复。三份Treatment契约、真实Dataset读取和用户侧正式入口均已越过原阻塞点，训练放行已恢复；用户手动paired正式训练当前进行中。六run联合validation性能、test、第二阶段OOF、阈值拟合、正式级联、P95和部署尚未完成或未执行。
最小子任务：

- [x] 核对`data/processed/`、`config/RADAR-LN3/`和三个跟踪文件；确认全局最大版本为V10、V11文件不存在，V3 index与V10/V2契约hash保持既有记录。
- [x] 冻结唯一实验变量：Treatment仅增加`velocity_bandwidth_mps/85`、`zero_velocity_energy_fraction(|v|<=2 m/s)`、`velocity_spectral_entropy/log(384)`和`top10_velocity_energy_fraction`；逐帧峰值归一化后跨当前事件帧/距离聚合，原始规范化前RD幅度作为输入。
- [x] 冻结缺失与因果语义：RD缺失事件四维为0且`valid=false`，valid仅门控不进入分类；Track 333 point 1–15保持缺失，不回填；在线只消费当前事件并调用与缓存相同的提取函数。
- [x] 冻结缓存与数据边界：仅V3 train=20684事件和val=4474事件分别生成NPZ及统一manifest；共享V3索引中的test=4505行仅扫描`split_name`后排除，不打开test RD payload、不生成test缓存或指标；`sample_ids`只用于严格对齐，不进入模型。
- [x] 冻结paired协议：Control重训V10架构，Treatment将Fusion输入127→131且新增四列零初始化；seed=42/43/44，共6次正式训练；共享参数逐值一致、初始logits一致、独立同seed DataLoader generator、每epoch记录parent顺序hash；只固定初始化和DataLoader顺序，不宣称CUDA位级确定性。
- [x] 冻结训练/评估协议：固定20 epoch且关闭early stopping；各自best为主并保留同epoch轨迹、epoch 20和双方best epoch护栏；每epoch保存可恢复checkpoint，同时保留`best_model.pt`/`last_model.pt`语义；不需要预训练。
- [x] 冻结第一阶段门槛：至少2/3 Treatment seed独立满足DBB P/R/F1均≥0.95、CU P/R/F1均≥0.90，并满足paired离散Pareto；三seed中位数亦需达标和改善。第二阶段仅在第一阶段通过后讨论train+val五折grouped OOF，目标DBB≥0.99、CU≥0.95，本Stage不实现、不运行。
- [x] 审查并最小扩展V10配置schema、Dataset/collate、模型、在线状态、checkpoint和训练入口；V7/V8/V9/V10 focused历史回归通过，旧配置不加载V11字段。
- [x] 在`src-v3/radar_ln3/rd_physical.py`实现唯一共享的当前事件RD四维物理谱提取函数、384点速度轴hash校验、0–1物理定界、缺失全零/invalid和因果输入契约。
- [x] 在`src-v3/build/rd_physical_paired.py`实现train/val分文件确定性NPZ缓存、统一manifest、V11数据/输入契约和六配置生成器；首次创建后重复运行全部返回`reused_identical`。
- [x] 在`src-v3/validate/rd_physical_paired.py`实现缓存集合、顺序、唯一性、split纯度、hash、数值范围、速度轴、缺失语义及全部25158个train/val事件在线同源逐值复算。
- [x] 将V11缓存接入Dataset/collate、离线batch和在线单事件接口；Control batch不含四维tensor，Treatment只在Fusion末尾追加四维，valid仅门控且不拼入分类特征。
- [x] 实现paired共享初始化：三个seed的共享参数/buffer逐值一致、Treatment新增Fusion四列严格为零、CPU非零四维探针初始logits逐位一致；CUDA按非严格确定性协议数值一致并记录实际差值。
- [x] 实现独立同seed DataLoader generator、每epoch parent顺序SHA-256、固定20 epoch/关闭early stopping、每epoch完整恢复checkpoint、`best_model.pt`和`last_model.pt`，以及显式epoch边界resume。
- [x] 创建V11 Control/Treatment各3份seed配置，配置只位于`config/RADAR-LN3/`且已被配置加载器、数据契约校验器、readiness和paired模型构造器实际加载。
- [x] 在`src-v3/report/rd_physical_paired.py`实现训练后六artifact联合审计：独立重算六个best checkpoint validation、核对20 epoch parent顺序hash、双方best epoch、epoch20、同epoch护栏、2/3 seed、中位数和离散Pareto。
- [x] 运行全量编译、src-v3静态导入隔离、V7–V10历史回归、缓存纯度、shape/mask、因果性、在线/缓存一致性、paired初始函数、forward/loss/gradient/update、checkpoint完整恢复和CPU/CUDA有界Smoke；最终readiness为passed。
- [x] 生成只含技术放行证据和六条用户手动训练命令的readiness快照；六个正式artifact目录均不存在，代理未启动正式训练。
- [ ] 用户手动完成六次正式训练并运行paired联合报告；当前正式训练进行中，尚未取得完整六run联合报告。
- [ ] 第一阶段通过后再单独冻结/授权第二阶段五折grouped OOF；当前未实现、未执行，test继续封存。

验证命令：已实际执行`..\.venv\Scripts\python.exe -m build.rd_physical_paired --project-root F:\Project\RadarInsight`、`-m validate.rd_physical_paired`、`-m smoke.rd_physical_paired --device cpu`、`-m smoke.rd_physical_paired --device cuda`、`-m smoke.rd_physical_paired_report_test`、`-m smoke.rd_physical_readiness`、V10/V9/V7-V8 focused回归、`-m compileall -q .`和`-m smoke.static_isolation_check`；均退出码0。正式入口带`--max-steps 1`的调试调用按设计在创建artifact前被拒绝，证明V11禁止把有界调试混入正式链路。
产物：`data/processed/RADAR-LN3-V11-cq08-{train,val}-rd-physical-event-cache.npz`、V11 cache manifest、V11数据/输入契约；六份V11配置；`src-v3/radar_ln3/rd_physical.py`及Dataset/模型/checkpoint扩展；`build/`、`validate/`、`smoke/`和`report/`入口；训练前readiness。Stage 041 readiness生成时六个正式artifact为空；当前正式artifact由用户手动训练生成中，完整性能结果尚未形成。
验收条件：train/val缓存sample集合和顺序严格等于V3相应split，逐值复算最大误差0；共享V3索引中的4505个test行只用于按split计数排除，没有打开test RD payload、生成test缓存或计算test指标；四维公式、尺度、缺失、速度轴、12帧上限和在线/缓存语义一致；三个seed共享初值通过；采样/恢复可审计；CPU/CUDA Smoke和历史回归通过；训练前六个artifact路径互不冲突且均为空。Stage 041准备验收已满足，后续ACL缺陷由Stage 042闭环。
是否需要数据重建：需要生成V11派生特征缓存、manifest和输入契约，不重建或覆盖V3 index/split、原始回波、MAT或canonical RD张量。
是否需要smoke：需要，已完成；Smoke代码、checkpoint和结果仅位于`src-v3/smoke/`。
是否需要正式训练：用户已手动启动paired正式训练；不需要预训练。尚未完成六run和联合报告。
阻塞条件：Stage 042跨账户ACL阻塞已解除。V11真实性能、第一阶段门槛和是否进入第二阶段仍阻塞于六次正式训练完成及联合报告；Unknown开放集拒识、test、正式级联和P95继续保持未执行。

## Stage 042 - V11共享物理缓存跨账户读取修复

目标：修复V11 train/val RD物理NPZ缓存因临时目录内生成后`rename`保留受保护Windows ACL，导致用户交互式账户无法读取的问题；保持缓存字节、SHA-256、数据/配置语义和版本号不变，并防止后续新缓存复发。
前置依赖：用户手动执行Treatment seed42和seed43正式命令时均在`verify_task_data_contract`的同一train cache哈希读取处获得`PermissionError [Errno 13]`；只读ACL核验已确认两配置复用同一train/val缓存。
状态：已完成。代码发布路径、现有train/val缓存ACL、固定hash、三份Treatment契约和真实Dataset读取均通过；用户确认修复后训练正常进行。
最小子任务：

- [x] 核对seed42/43配置中的train/val缓存路径和预期SHA-256完全一致，确认seed无关且属于同一共享缓存问题。
- [x] 核对train/val缓存均为普通文件，内容长度正常，但`AreAccessRulesProtected=True`且仅OWNER RIGHTS/SYSTEM/Administrators拥有权限；同目录正常文件继承父目录ACL。
- [x] 手术式修改V11缓存发布逻辑：确定性NPZ先在内存形成，以`xb`直接创建目标文件，再在Windows上以`icacls /reset`采用`data/processed`父目录DACL；内容相同的幂等复用路径也会修复ACL。
- [x] 只修复现有train/val两份缓存的ACL，不改文件内容；长度保持`346504/75717`字节，修复前后SHA-256分别保持`726143...a5`和`91776d...7fc`。
- [x] 运行focused权限回归、NPZ读取、seed42/43/44三份Treatment数据契约、单parent Dataset缓存加载、compileall和src-v3静态隔离检查；均通过。
- [x] 确认两次失败调用发生在artifact创建之前且未留下Treatment半成品目录；恢复手动训练放行，用户随后确认训练正常进行。

验证命令：已执行`F:\Project\RadarInsight\.venv\Scripts\python.exe -m smoke.rd_physical_cache_access_test --project-root F:\Project\RadarInsight`、`-m compileall -q .`和`-m smoke.static_isolation_check --root F:\Project\RadarInsight\src-v3`；focused结果为passed、隔离`violations=[]`。用户实际重试正式入口后确认训练正常进行。未由代理启动20 epoch训练。
产物：修改后的`src-v3/build/rd_physical_paired.py`、位于`src-v3/smoke/`的权限focused测试、ACL已恢复继承但内容不变的两份V11 NPZ缓存，以及覆盖更新后的三个跟踪文件。
验收条件：train/val缓存均有9条父目录继承规则并向通用用户组授予读取权限；两份SHA-256与V11配置/manifest保持一致；seed42/43/44 Treatment契约及Dataset可加载；失败调用没有artifact副作用；用户账户可正常进入训练。已满足。
是否需要数据重建：否；只修复文件系统ACL元数据。缓存内容、索引、manifest、契约和配置均不得改变，因此保持V11。
是否需要smoke：需要，限权限、读取、契约和静态focused检查，代码必须位于`src-v3/smoke/`。
是否需要正式训练：本修复本身不需要训练；用户已在放行后手动执行，当前进行中。
阻塞条件：本Stage无剩余阻塞。完整paired性能仍等待六run结束和联合报告；训练过程中不得重跑训练前readiness，因为其安全护栏要求正式artifact为空。

## Stage 043 - V11三seed paired正式结果复核与优化判断

目标：独立加载V11 Control/Treatment六个正式best checkpoint，重算同一CQ08 V3 validation，核验20 epoch配对顺序、共享初始化、各自best与同epoch护栏，判断当前事件四维RD物理谱分支是否通过第一阶段门槛，并定位剩余DBB/CU误路由。
前置依赖：用户确认六次训练已完成并授权自行评估；六个artifact均存在20个epoch checkpoint、best/last、training-summary、sample detail和完成日志；test继续封存。
状态：已完成。六个best checkpoint已在CUDA上独立重载并精确复现训练summary；Control/Treatment 20个epoch采样顺序逐seed完全一致。第一阶段`0/3` seed通过且三seed中位数失败，V11四维当前事件RD物理谱Treatment不得进入第二阶段、正式级联或部署。
最小子任务：

- [x] 核对六个正式artifact结构完整：每个20个epoch checkpoint、best/last、summary、sample detail和`run=complete`日志均存在。
- [x] 运行`report.rd_physical_paired`，独立加载六个best checkpoint在配置设备上重算validation，并核对训练summary、checkpoint hash和20 epoch parent顺序。
- [x] 汇总每seed和三seed中位数的DBB/CU Precision、Recall、F1、双向误路由、总体指标及第一阶段离散Pareto门槛；Treatment中位DBB P/R/F1=`0.980392/0.956938/0.968523`，CU=`0.678571/0.826087/0.745098`，第一阶段失败。
- [x] 审查六条20 epoch轨迹：120个arm-epoch中无一同时达到DBB floor≥0.95与CU floor≥0.90；Treatment在DBB达标约束下每seed最佳CU floor仅`0.724138/0.678571/0.678571`。
- [x] 对Control/Treatment残差parent做paired集合分析；三seed合计修复6个、新增3个，变化跨seed不稳定；Treatment共同错误为8个Bird与2个Unknown，除Track 333外RD均完整。
- [x] 描述四维RD特征的train/validation分离与已学习权重；确认其是现有RD的确定性压缩且Unknown/Bird单特征分离弱，不能把Clutter较好分离外推到Unknown。
- [x] 使用train-only Drone/Bird/Balloon液态中心分析`P(CU)`、Energy和两类原型距离，并计算不输出阈值的validation oracle frontier；四种分数在三个seed均不存在满足第一阶段双类floor的切点。
- [x] 完成11/11统计谬误扫描和复现边界审查：同一232-parent validation在三个seed重复使用，CU support仅23，seed不是独立数据样本；CUDA只固定初始化和顺序，不宣称位级确定性。
- [x] 明确第二阶段不放行；下一候选仅保留“DBB三亚类辅助表征/度量目标”单因素设计讨论，不增加液态单元复杂度，不在当前隐藏层直接增加距离阈值。
- [x] 覆盖更新三个跟踪文件，记录真实结果、未执行项和结论边界。

验证命令：`F:\Project\RadarInsight\.venv\Scripts\python.exe -m report.rd_physical_paired --project-root F:\Project\RadarInsight`，以及只读解析六个`training-summary.json`、`sample_detail_information.json`和联合报告；最终运行compile/隔离/报告hash核验。
产物：`src-v3/processed/report/RADAR-LN3-V11-rd-physical-paired-validation.json`、`RADAR-LN3-V11-rd-physical-residual-analysis.json`、`RADAR-LN3-V11-rd-physical-parent-transitions.csv`、`RADAR-LN3-V11-liquid-dbb-knownness-diagnostic.json`，新增/更新的`src-v3/report/rd_physical_residuals.py`与`liquid_knownness.py`，以及覆盖更新的三个跟踪文件。
验收条件：六个best checkpoint独立validation与训练summary逐值一致；20 epoch parent顺序配对一致；第一阶段结论由2/3 seed和三seed中位数共同决定；残差、全epoch和oracle可达性边界均有证据；不把validation结果外推为test、开放集拒识或部署性能。已满足，性能门槛未满足。
是否需要数据重建：否。
是否需要smoke：不需要训练Smoke；需要真实checkpoint评估与静态检查。
是否需要正式训练：六次已由用户完成；本Stage不训练。
阻塞条件：V11第一阶段失败：`passing_seed_count=0<2`且中位CU P/R/F1远低于0.90；第二阶段被硬阻塞。test、OOF、阈值拟合、specialist/cascade、线上stable和P95均未执行。

## Stage 044 - V12 DBB三亚类辅助表征paired实施准备

目标：以V11 Control的CQ08 causal TR+RD液态Gate主链为基线，不启用V11四维RD物理Treatment，仅在液态序列状态上增加同构、仅训练期可请求的Drone/Bird/Balloon三亚类辅助头；通过Control辅助权重0与Treatment固定辅助权重0.20的三seed paired实验，检验DBB亚类监督能否保护困难Bird并降低Bird→CU，同时不恶化Unknown→DBB。当前授权边界为完整实施准备、focused测试和短时Smoke，不启动正式20 epoch训练。
前置依赖：Stage 043已完成并确认V11失败不是阈值或checkpoint选优问题；V3 index、V10 observation-increment数据契约、V2 train-only归一化统计、CQ08 train/val split和V11 Control主模型接口保持冻结；test继续封存。
状态：实施准备已完成；其后的六次正式20 epoch训练和validation评估已由Stage 045闭环。Stage 044的训练前readiness历史结论仍有效，但不再代表当前状态；test、OOF、级联、stable和P95仍未执行。
最小子任务：

- [x] 核对Stage 043真实结果、V11主Gate接口、loss/checkpoint/在线状态契约、现有全局最大版本V11及工作树状态。
- [x] 冻结数据复用：继续使用V3 CQ08 index、V10 observation-increment数据契约和V2 train-only统计；不生成缓存、不重建数据、不读取test。
- [x] 冻结唯一变量：Control与Treatment模型结构、初始化、主Gate loss、class weight、采样顺序、20 epoch、checkpoint选择和评估完全一致；仅`dbb_auxiliary_loss_weight`为0与0.20。
- [x] 冻结辅助监督单位：仅Drone/Bird/Balloon parent的最后一个合法因果状态参与三分类CE；复用train-only parent计数得到的固定逆频率权重且不搜索；Clutter/Unknown严格排除且不得建立DBB原型。不得对早期prefix静默使用完整轨迹亚类标签。
- [x] 冻结线上/评估语义：主Gate logits和`forward_online_point`接口不变；辅助头不参与validation Gate指标、checkpoint选择、级联、拒识或在线输出。
- [x] 在`src-v3/radar_ln3/task_models.py`和`src-v3/train/tasks.py`手术式实现V12配置、模型、loss、paired初始化、checkpoint和日志契约。
- [x] 创建V12 seed42/43/44 Control/Treatment六份正式配置，全部写入`config/RADAR-LN3/`；不创建数据契约新文件。
- [x] 在`src-v3/smoke/`实现辅助标签隔离、shape/finite、CU零贡献、梯度归因、主logits一致、在线接口、checkpoint恢复和paired顺序focused测试。
- [x] 冻结训练后联合评估器：独立重载六个best checkpoint，验证20 epoch parent顺序、2/3 seed与中位主门槛，并要求至少2/3 seed及中位Bird→CU减少且Unknown→DBB不增加。
- [x] 运行CPU和CUDA短时Smoke、V11 Control等价回归、报告门槛测试、compileall、静态导入隔离和限定文件`git diff --check`；均通过。
- [x] 覆盖更新三份跟踪文件并判断达到“可由用户手动启动V12 paired正式训练”的状态；readiness失败关闭且六个正式artifact目标均为空。

验证命令：已在`src-v3`执行`F:\Project\RadarInsight\.venv\Scripts\python.exe -m smoke.dbb_auxiliary_paired --project-root F:\Project\RadarInsight --device cpu`、同命令`--device cuda`、`-m smoke.dbb_auxiliary_paired_report_test`、`-m smoke.rd_physical_paired_report_test`、`-m smoke.logging_test`、`-m smoke.dbb_auxiliary_readiness`、`-m compileall -q .`、`-m smoke.static_isolation_check --root F:\Project\RadarInsight\src-v3`及限定范围`git diff --check`；最终均退出码0。
产物：修改`src-v3/radar_ln3/task_models.py`、`src-v3/train/tasks.py`和通用paired报告helper；新增`src-v3/report/dbb_auxiliary_paired.py`、三个V12 Smoke/readiness入口、CPU/CUDA结果、报告门槛结果、readiness和六份V12配置；不修改`src/`、V3/V10/V11数据文件或既有artifact。
验收条件：Control/Treatment初始主Gate与辅助logits逐值一致；真实batch及DataLoader顺序一致；辅助监督只命中DBB final state且CU标签对辅助loss和梯度零贡献；Treatment仅通过辅助loss改变液态共享表征；主Gate评估/在线接口不消费辅助输出；checkpoint严格保存恢复；CPU/CUDA Smoke、编译和隔离通过。正式训练不属于本Stage自动执行范围。
是否需要数据重建：否，未执行；输入字段、split、窗口、特征schema和归一化均不变。
是否需要smoke：需要，限`src-v3/smoke/`内的短时CPU/CUDA验证。
是否需要正式训练：本Stage不由代理启动；六次训练后来已由用户手动完成，结果见Stage 045。
阻塞条件：实施准备无技术阻塞；Stage 045确认V12未通过性能保留门槛，因而OOF、级联和部署继续被阻塞。

## Stage 045 - V12 DBB三亚类辅助表征paired正式评估

目标：对用户已完成的V12三seed Control/Treatment六次正式训练执行冻结validation联合评估、best checkpoint独立复算、全20 epoch可达性审计和辅助头机制诊断；判断V12是否可保留并进入OOF、级联或部署，不访问test、不事后改选checkpoint或拟合阈值。
前置依赖：Stage 044的V12六配置、paired初始化/采样契约、20 epoch且关闭early stopping、冻结Gate与机制门槛均保持不变；六个formal artifact由用户手动训练生成。
状态：已完成。V12状态为`completed_but_gate_failed`，正式保留门槛`v12_retention_passed=false`；不能进入OOF、正式级联、stable、P95或部署。
最小子任务：

- [x] 核对六个formal artifact：六份`training-summary.json`均`status=passed`、`stop_reason=epoch_limit`、history均20 epoch，best/last checkpoint和20份epoch checkpoint完整，test均未执行。
- [x] 在CUDA上重跑冻结`report.dbb_auxiliary_paired`；独立重载六个best checkpoint，重算232 parent/4474 causal-prefix window validation，并验证与训练summary逐值一致。
- [x] 验证paired归因：三个seed的Control/Treatment初始CPU logits hash相同、20 epoch parent顺序SHA-256逐epoch一致；未把三个seed误当三个独立validation样本。
- [x] 复算主Gate门槛：三个Treatment seed均满足DBB三指标不低于0.95，但均未满足CU三指标不低于0.90；通过seed为`0/3`，三seed中位门槛也失败。
- [x] 审查离散Pareto与目标机制：中位DBB→CU由10降至7，但CU→DBB由4升至5；Bird→CU中位减少0、Unknown→DBB中位增加1，目标机制通过seed为`0/3`。
- [x] 审查全部120个arm-epoch；不存在任何同时满足DBB floor≥0.95与CU floor≥0.90的epoch，排除best checkpoint选择偶然解释，且不据此事后改选epoch。
- [x] 新增并运行只读`report.dbb_auxiliary_diagnostics`：三份Treatment best checkpoint的validation DBB辅助三分类中位Accuracy=`0.909091`、Macro-F1=`0.902351`；所有22个seed级Bird→CU错误均仍被辅助头预测为Bird。
- [x] 解释Unknown边界：Treatment每seed有5/7/5个Unknown→DBB，强制三亚类投影主要为Bird且置信度可很高；因Unknown未参与辅助训练，该softmax不是knownness或拒识分数，禁止直接用于路由。
- [x] 完成11/11统计谬误扫描、复现边界、编译、静态隔离和限定差异检查；覆盖更新三个跟踪文件。

验证命令：在`src-v3`执行`F:\Project\RadarInsight\.venv\Scripts\python.exe -m report.dbb_auxiliary_paired --project-root F:\Project\RadarInsight`、`-m report.dbb_auxiliary_diagnostics --project-root F:\Project\RadarInsight`、`-m compileall -q .`、`-m smoke.static_isolation_check --root F:\Project\RadarInsight\src-v3`及限定文件`git diff --check`。
产物：`src-v3/processed/report/RADAR-LN3-V12-dbb-auxiliary-paired-validation.json`、`RADAR-LN3-V12-dbb-auxiliary-diagnostic.json`、`src-v3/report/dbb_auxiliary_diagnostics.py`和覆盖更新的三个跟踪文件；未修改data、config或checkpoint。
验收条件：六best checkpoint独立validation与summary一致；paired顺序和初始化契约通过；主门槛、机制门槛、全epoch可达性与统计边界有真实证据；未把辅助三分类、validation或单seed结果冒充开放集拒识、test或部署性能。已满足，性能门槛未满足。
是否需要数据重建：否，未执行；继续复用V3/V10/V2数据契约。
是否需要smoke：不需要训练Smoke；执行正式checkpoint只读评估和focused诊断。
是否需要正式训练：六次已由用户完成；本Stage未启动训练或重训。
阻塞条件：V12保留门槛失败；test、OOF、阈值拟合、正式级联、online stable和P95均未执行，当前禁止进入这些阶段。下一实验必须另行冻结单一变量，不能通过提高辅助loss权重或增加液态复杂度直接延续V12。

## Stage 046 - V12隐藏状态三亚类knownness只读诊断

目标：在V12正式保留失败后，使用六个Control/Treatment best checkpoint的液态parent-final隐藏状态，分别以各run的train-only Drone/Bird/Balloon样本拟合三亚类中心和固定尺度，比较`P(CU)`、Gate Energy、最近类中心余弦距离和最近类中心对角Mahalanobis距离；判断V12辅助监督是否产生了可用于后续开放集边界实验的增量排序信息。不得拟合阈值、改选checkpoint、访问test或启动训练。
前置依赖：Stage 045已确认V12 `v12_retention_passed=false`，但辅助头能识别被Gate误拒的Bird；六run、V3 train/val split、V10/V2数据契约、best checkpoint及标签政策冻结。
状态：已完成。六个best checkpoint的train-only三亚类knownness paired诊断和统一分数表已生成并独立复算；V13设计证据门槛失败，当前不支持原型/距离耦合V13，也不放行阈值、OOF、训练或部署。
最小子任务：

- [x] 冻结数据边界：每个run只用自身train parent拟合DBB三亚类参考；validation只评分；Clutter/Unknown不创建中心；test不读取。
- [x] 冻结分数：`p_cu`、`gate_energy=-logsumexp(logits)`、`nearest_subclass_cosine_distance`、使用train DBB类内pooled per-dimension方差的`nearest_subclass_diagonal_mahalanobis`；所有分数预定义为越高越偏CU/未知。
- [x] 冻结比较：分别报告CU-vs-DBB、Unknown-vs-DBB、Unknown-vs-Bird、Clutter-vs-DBB和Clutter-vs-Bird AUC；Control/Treatment必须同seed配对，并报告三seed中位，不把seed当独立数据样本。
- [x] 冻结残差诊断：分别查看Bird→CU与Unknown→DBB在train DBB参考中的百分位；不得输出或选择线上阈值。
- [x] 冻结V13设计证据门槛：某个Treatment距离分数须在至少2/3 seed及三seed中位上使Unknown-vs-Bird AUC相对同seed Control和Treatment `p_cu`均提高至少0.02；Treatment中位Unknown-vs-DBB AUC≥0.90；CU-vs-DBB AUC相对Treatment `p_cu`下降不超过0.01。该门槛仅放行后续设计讨论，不放行训练、阈值或部署。
- [x] 实现独立只读诊断入口及focused契约测试，不修改模型、训练、数据或配置。
- [x] 在CUDA上提取六run train/validation final state，生成诊断JSON和1392行统一validation分数CSV；核验hash、数量、标签、checkpoint及每run train-only参考契约。
- [x] 从统一CSV独立复算全部120个AUC单元，与JSON逐值一致；focused test验证CU不进入中心/方差、分数方向和门槛失败关闭。
- [x] 完成11/11统计谬误扫描，明确小CU support、多重分数查看和固定validation的解释边界。
- [x] 覆盖更新三个跟踪文件，运行compileall、静态隔离和限定差异检查，结论为不值得设计当前原型/距离耦合V13。

验证命令：用户在`src-v3`手动执行`F:\Project\RadarInsight\.venv\Scripts\python.exe -m report.dbb_knownness_paired --project-root F:\Project\RadarInsight --batch-size 4`；随后实际执行`-m smoke.dbb_knownness_paired_test --project-root=F:\Project\RadarInsight`。最终另运行compileall、静态隔离和限定差异检查。
产物：`src-v3/report/dbb_knownness_paired.py`、`src-v3/smoke/dbb_knownness_paired_test.py`、`src-v3/processed/report/RADAR-LN3-V12-dbb-knownness-paired-diagnostic.json`、`RADAR-LN3-V12-dbb-knownness-paired-validation-scores.csv`及focused test JSON。
验收条件：六run均为train=1081、val=232 parent，标签分布与冻结split一致；三类参考只使用train DBB=976并排除CU=105；1392行分数唯一且120个AUC可复算；无阈值、test或训练副作用。已满足，V13设计证据门槛未满足。
是否需要数据重建：否。
是否需要smoke：只需要focused只读契约测试，不需要训练Smoke。
是否需要正式训练：否，未执行。
阻塞条件：V13证据门槛已失败：两个距离候选通过seed均为`0/3`且中位Unknown-vs-Bird与CU-vs-DBB排序均弱于Treatment `P(CU)`；不得继续原型/距离耦合、拟合阈值或进入训练。下一方向必须回到输入/目标覆盖证据，不得在同一隐藏状态上继续搜索更多距离函数。

## Stage 047 - 三层软级联产品语义与关键Bug手术式修复

目标：暂停V13原型/距离knownness路线，严格实现产品定义的三层液态软级联：第一层只执行`DBB_ACCEPTED`或路由至CU；DBB specialist仅在DBB路由内输出Drone/Bird/Balloon；CU specialist只接受Clutter，未接受者形成`UNKNOWN_REJECTED`动作。软五类后验仅用于概率审计，不再等同产品动作。
前置依赖：Stage 046已否定当前静态距离候选；用户已明确三层语义并授权先核实后修复，修复完成后必须停下请求确认，由用户手动重新训练；V3/V10/V12历史数据、配置、checkpoint和报告均只读保留。
状态：V13三层产品语义、联合训练目标、reason-aware dynamics、分层动作/stable、checkpoint原子性、指标分母和评估接口均已完成手术式修复；focused及真实CQ08有界CPU/CUDA Smoke通过。本Stage未自行启动正式训练；用户随后已在Stage 048手动完成20 epoch监督训练和冻结validation复核。train parent-grouped OOF校准、test、P95和部署仍未执行；原型距离方向继续保持不放行。
最小子任务：

- [x] 对照实际代码逐项核验用户列出的10项问题，并区分已确认问题、非代码瓶颈和待验证假设。
- [x] 冻结产品动作与标签语义：真实`UnknownTarget`标签、CU closed-set条件预测、`UNKNOWN_REJECTED`动作和stable pending分别记录；禁止使用五类argmax或全局max-softmax拒识冒充分层动作。
- [x] 冻结版本策略：loss、dt输入、checkpoint与线上动作契约使用全局V13；复用V3 index和V2 train-only normalization，新建V13语义契约并以V6 causal-quality contract作为只读上游，不覆盖V8/V12 artifact。
- [x] 在`losses.py`/`task_models.py`实现parent-final主损失与显式低权重prefix辅助契约；事件1–2及final之前不满足最小证据量的prefix不进入辅助监督，final始终是主目标。
- [x] 实现route-aware三层final-state联合目标；Gate、DBB specialist和CU specialist保持条件标签边界，Unknown不向DBB specialist反传，DBB不向CU specialist反传。
- [x] 在`rejection.py`实现强类型分层校准和动作决策；Gate低DBB接受度路由CU，CU低Clutter接受度形成`UNKNOWN_REJECTED`；校准必须来自train parent-grouped OOF并绑定精确三checkpoint部署身份。
- [x] 在`stable.py`/`cascade.py`实现分层动作状态机，覆盖`DBB_PENDING`、`DBB_ACCEPTED`、`CU_PENDING`、`CLUTTER_ACCEPTED`、`UNKNOWN_REJECTED`；pending/rejected不能同时确认具体类别。
- [x] 扩展级联评估，分别输出soft posterior审计、Gate route、条件specialist、hierarchical parent-final、stable动作、Coverage/Accepted/Selective指标；无校准时动作指标保持未评估。
- [x] 修复指标分母：Accepted Macro只平均四个可接受known类，Unknown误接收仍进入known Precision假阳性；Drone/Bird Precision/Recall使用全体parent分母；pending confidence为空。
- [x] 将`tr_dt_reason`由Dataset/collate传至液态模块，区分首点、秒级重复时间和正间隔；Dataset和online均强制首事件边界，不伪造高精度dt，不把reason作为分类Fusion特征。
- [x] 为V13准备同版本Gate/DBB/CU三份配置和统一联合训练/epoch-checkpoint入口；三任务共享split、normalization、encoder/dynamics、seed与parent顺序，禁止与V8/V12混拼。
- [x] 强化checkpoint原子性：同epoch/global step/product selection/config semantic identity/checkpoint set ID才允许组合；soft posterior不参与best排名；旧`train.tasks`拒绝V13。
- [x] 运行focused单元测试、离线/在线逐事件一致性、shape/mask/finite、loss梯度边界、三checkpoint恢复、历史V7/V8回归、六个V12 best只读恢复、compileall和src-v3静态隔离检查。
- [x] 创建`src-v3/smoke/RADAR-LN3-V13-hierarchical-training-readiness.json`并覆盖更新三个跟踪文件；正式训练、OOF、test、P95与部署均明确未执行，随后请求用户确认。

验证命令：已在`src-v3`运行`python -m smoke.hierarchical_contract_test`、`python -m smoke.hierarchical_cascade_smoke --device cpu/cuda`、V7/V8历史核心回归、六个V12 best checkpoint只读恢复；最终还运行`python -m compileall -q .`、`python -m smoke.static_isolation_check --root F:\Project\RadarInsight\src-v3`和限定`git diff --check`。正式训练命令仅在用户再次确认后提供。
产物：V13三层动作/训练/评估实现，`data/processed/RADAR-LN3-V13-cq08-hierarchical-cascade-data-contract.json`，三份`config/RADAR-LN3/RADAR-LN3-V13-*`配置，V13 focused/CPU/CUDA Smoke结果及training-readiness；未创建新CQ08样本、index、split、RD tensor或normalization。
验收条件：三层动作与soft posterior语义分离；主训练目标、checkpoint候选和parent-final产品动作对齐；拒识与stable不会产生矛盾状态；dt原因在dynamics链路可见；三任务同版本checkpoint可严格恢复；所有focused检查通过。代码契约验收已满足；后续Stage 048真实性能已评估但未达到产品门槛，不能因Smoke或训练完成宣称达标。
是否需要数据重建：不需要；V3 index/V2 normalization及现有RD tensors原样复用，V13仅新增数据与动作语义契约。
是否需要smoke：需要，限定CPU/CUDA有界契约Smoke已通过；不是性能实验。
是否需要正式训练：本Stage未启动；用户随后已在Stage 048手动完成唯一授权的V13联合监督训练。真实拒识、test和部署仍未达到执行状态。
阻塞条件：Stage 048已确认V13监督基线四项产品护栏为`0/4`，暂不放行OOF校准。跨split RD/capture provenance未裁决、validation CU support仅8/15、test/P95未执行，继续阻塞泛化与部署结论。

## Stage 048 - V13正式训练后冻结Validation评估

目标：核验用户手动完成的V13三checkpoint联合训练产物，使用冻结CQ08 V3 validation独立恢复三份best checkpoint，复算三层诊断输出、Gate/DBB/CU条件指标、全epoch产品护栏和既有人工审计残差；不得访问test、拟合阈值、运行OOF或把五类soft posterior冒充产品动作。
前置依赖：Stage 047的V13代码、三配置、V3 index、V2 train-only normalization、V13数据语义契约和产品选优规则全部冻结；正式artifact由用户手动训练生成。
状态：已完成。训练与checkpoint契约通过，冻结validation独立复算通过；四项产品护栏通过`0/4`，V13状态为“代码/契约修复成功、监督性能未放行”。当前不进入OOF校准、stable、test、P95或部署。
最小子任务：

- [x] 核验正式artifact完整性：`status=passed`、CUDA、20/20 epoch、10820 global steps、训练日志正常结束且无Traceback/OOM/non-finite记录。
- [x] 核验三task best与joint best均来自epoch 13/global step 7033，共享`checkpoint_set_id=611c0817ef0f42afb9ccf9c8658e9602`，不存在局部best错配。
- [x] 使用正式三best checkpoint独立执行validation-only级联评估；batch=2复跑的全部核心报告字段与training summary逐值一致，核心规范化SHA-256均为`04b2905edf1b02c71d7f7a44ea6bbd83cf7217cd1e48418a22e8f72f8fd190da`。
- [x] 复核parent-final诊断五类结果：Accuracy=`0.840517`、Macro Precision/Recall/F1=`0.670816/0.709286/0.685081`；明确它没有OOF阈值，不是已验证拒识动作性能。
- [x] 复核组件结果：Gate DBB P/R/F1=`0.960591/0.933014/0.946602`、CU=`0.517241/0.652174/0.576923`；DBB specialist Macro P/R/F1=`0.938327/0.955433/0.943338`；CU specialist=`0.694444/0.704167/0.651515`。
- [x] 审查20个epoch：产品护栏每epoch均为`0/4`；各护栏全轨迹最高值为Gate=`0.517241`、DBB=`0.852459`、CU=`0.583333`、Drone/Bird=`0.732143`，不存在可事后切换的达标epoch。
- [x] 核对固定产品选优：epoch 13的最小归一化完成度`0.544465`高于epoch 15的`0.540541`；epoch 15虽诊断Macro-F1更高，但不得越过冻结护栏排序事后换best。
- [x] 审查方向错误：Gate有14个DBB→CU和8个CU→DBB，其中Bird→CU=11、Unknown→DBB=6；12条既有人工审计残差仅Track 852与2862在V13中纠正，其余10条持续错误。
- [x] 对小support计算Wilson区间并完成11/11统计谬误扫描；Clutter仅8 parent、Unknown仅15 parent，固定validation被历代实验反复使用，任何泛化或门槛外推均为高风险。
- [x] 验证batch=8复跑分类指标不变；确认报告字段`dt_reason_counts_including_padding.padding`随collate batch组成从416变为1334，属于报告口径问题，不是数据固有计数或预测差异，记录待修复。
- [x] 覆盖更新三个跟踪文件并保留所有未执行项；未修改模型、数据、配置、标签或checkpoint。
- [ ] train parent-grouped OOF分层拒识校准；因监督护栏`0/4`和固定validation过度使用暂不放行。
- [ ] calibrated hierarchical action、online stable、test、session、P95与部署评估；均未执行。

验证命令：在`src-v3`使用`python -m report.cascade`加载三份V13 best checkpoint，分别以batch=8作批次不变性检查、以冻结batch=2作训练摘要精确复核；其余检查为只读JSON/checkpoint/hash/日志解析。
产物：`artifacts/RADAR-LN3-V13-cq08-liquid-hierarchical-cascade-formal/`正式训练产物；规范复核报告`src-v3/processed/report/RADAR-LN3-V13-hierarchical-cascade-validation-batch2/evaluation-val.json`，SHA-256=`9b8aae57050c3d1a6283a96a0f0e301758b67b58f553fc78c1a6a68835f51d77`；对应样本明细SHA-256=`e93b3062ad5b763ab50757729fece8b1bbddf29990501fa0039b0b0c0db30fd5`。
验收条件：训练产物、三checkpoint原子身份、冻结validation复算、全epoch可达性、错误parent和统计边界均可追溯；test/OOF/stable/P95保持未执行。技术验收已满足，性能验收未满足。
是否需要数据重建：否；本Stage只读复用V3 index、V2 normalization和V13语义契约。
是否需要smoke：不需要新训练Smoke；正式checkpoint恢复和validation-only复算已执行。
是否需要正式训练：用户已完成本轮V13单seed正式训练；不自动重训或启动多seed。
阻塞条件：Gate、DBB、CU和Drone/Bird四项产品护栏全未通过；CU validation仅23 parent且标签内部异质；OOF拒识、test、stable、P95未执行；跨split RD/capture provenance未裁决。下一阶段必须先做训练集内错误覆盖/子流型与目标权重归因，不能用validation阈值或增加液态容量制造表面提升。

## Stage 049 - V14严格Open-set契约与Unknown零学习边界

目标：把V13的“已知Unknown监督二分类＋未校准拒识接口”升级为严格known-only open-set训练：`UnknownTarget`不得进入参数梯度、归一化统计、训练采样、checkpoint选择或阈值拟合；运行时只能作为`UNKNOWN_REJECTED`动作的开放集评估标签。第一层接受DBB或路由CU；DBB内部分Drone/Bird/Balloon且低knownness拒识Unknown；CU只接受Clutter，未接受者拒识Unknown。
前置依赖：Stage 048已确认V13不是严格open-set，Gate会用Unknown作为CU监督、CU CE会把Unknown作为第二类学习、DBB分支无二次拒识；用户要求先按`.rules`保存Git记录再实施。开发分支`codex/v13-strict-open-set`已创建，基线提交为`d4d39e2`和`627950a`。
状态：严格open-set代码、known-only数据边界、三层动作、正式known-only训练、5折parent-grouped OOF、known-only校准以及一次冻结development validation与stable评估均已完成。Unknown学习/归一化/OOF/阈值暴露为0，协议契约通过；冻结开放集性能未达标，V14冻结为“协议有效、性能失败、不可部署”基线。Test、P95和部署均未执行。
最小子任务：

- [x] 读取`.rules`，确认`main`禁止直接提交；创建普通开发分支并按关联工作拆分保存V12诊断与V13三层基线。
- [x] 明确严格open-set判据：Unknown不得参与任何可学习或模型选择步骤；validation/test Unknown只允许冻结评估，不能用于threshold、loss、采样或checkpoint选择。
- [x] 冻结V14版本策略：复用V3 index/split和原TR/RD张量；因训练population、normalization、loss、输出动作、校准和checkpoint schema发生实质变化，新建V14 known-only normalization、数据契约和三配置。
- [x] 新建known-only normalization构建入口，只使用train的Drone/Bird/Balloon/Clutter 1011 parent；排除70个train Unknown，known/Unknown RD tensor路径交集为0；修复基础统计复跑幂等性。
- [x] 新建V14数据契约，冻结known train=`488/243/245/35`、Unknown optimizer exposure=`0`；validation Unknown=15仅允许best checkpoint冻结后的development审计，test继续封存。
- [x] 修改Dataset/loader：V14 train和known checkpoint validation只包含四个known类；Unknown进入train Dataset、parent白名单、loss或checkpoint selection均fail-closed。
- [x] 修改Gate目标：只以DBB和Clutter训练`DBB_ACCEPT_EVIDENCE/ROUTE_TO_CU_EVIDENCE`，Unknown不映射为CU训练标签。
- [x] 修改CU目标：仅学习`CLUTTER_ACCEPT_EVIDENCE/NON_CLUTTER_KNOWN_EVIDENCE`，负样本只来自DBB known，不建立Unknown类、prototype或概率输出。
- [x] 保留DBB三分类known-only CE，并增加独立DBB knownness probability/margin接受边界；Gate接受DBB但knownness未通过时输出`UNKNOWN_REJECTED`。
- [x] 将分层校准扩展为Gate、DBB knownness和Clutter三道known-only接受边界；loader要求train parent-grouped OOF、Unknown/validation/test零拟合并绑定精确正式三checkpoint身份。
- [x] 实现5折OOF assignment、fold-train-only normalization、fold-specific数据契约、固定20 epoch折训练/held-out单次预测入口和独立阈值拟合入口；本Stage未执行任何折训练或真实拟合。
- [x] 修改checkpoint选择：正式训练每epoch只使用217个known validation parent产品护栏；Unknown发展审计在best冻结后才运行且不能返回selection inputs。
- [x] 修改offline/online动作和stable接口：`DBB_ACCEPTED`前同时通过Gate与DBB knownness；CU只接受Clutter；两条失败路径均汇合`UNKNOWN_REJECTED`且不携带accepted class。
- [x] 修改评估与报告：分离known closed-set、known acceptance、open-set rejection和development Unknown审计；无真实OOF calibration时Coverage/FAR/FRR/Selective Risk/stable均保持未评估。
- [x] 修复batch依赖padding报告语义：V13/V14均输出`dt_reason_counts_data_events`、`collate_padding_positions`和`evaluation_batch_size`，不再把collate padding冒充数据固有dt计数。
- [x] 增加负向测试：Unknown进入normalization/train batch/loss/checkpoint selection/calibration任一环节均被拒绝；V13 checkpoint不能加载V14，V14 ranking key长度固定为8且V13正式checkpoint仍可只读恢复。
- [x] 运行focused loss/gradient/shape/mask/checkpoint/offline-online、5折白名单与synthetic known-only拟合测试；有界CPU/CUDA Smoke、V13 focused、历史级联focused、compileall、静态隔离和限定差异检查通过。
- [x] 创建`src-v3/smoke/RADAR-LN3-V14-strict-open-set-training-readiness.json`并覆盖更新三个跟踪文件；所有性能与部署未执行项保持原样。

验证命令：已在`src-v3`运行`python -m build.strict_open_set`幂等复核、`python -m build.strict_open_set_oof_folds`、`python -m smoke.strict_open_set_contract_test`、`python -m smoke.strict_open_set_cascade_smoke --device cpu/cuda`、`python -m smoke.hierarchical_contract_test`和历史`smoke.causal_cascade_contract_test`；另执行全`src-v3` compileall、静态导入隔离、限定`git diff --check`及V13正式三checkpoint只读恢复。
产物：V14 known-only normalization、数据契约、三配置、正式三checkpoint、5折assignment/fold contract、5份fold-train统计、5份fold数据契约和held-out预测、严格open-set calibration、冻结development validation报告与样本明细；严格open-set训练/动作/评估实现；focused、CPU/CUDA Smoke及readiness证据。
验收条件：代码/数据/严格open-set语义与OOF来源契约已满足。Unknown只作为冻结开放集评估真值及`UNKNOWN_REJECTED`动作语义；无校准时不输出产品动作；V13/V14产物不可混用。性能验收未满足：五动作Macro-F1=`0.733411`、Unknown拒识Recall=`0.266667`、FAR=`0.733333`。
是否需要数据重建：不重建V3 index、split或RD张量；已新建V14 known-only normalization与语义契约，因为V2统计包含70个train Unknown。
是否需要smoke：focused及有界CPU/CUDA Smoke均已通过，产物全部位于`src-v3/smoke/`；Smoke不代表分类或拒识性能。
是否需要正式训练：V14正式联合训练及5折OOF均已由用户完成；本轮无需重训。
阻塞条件：Unknown高置信误接收、Clutter覆盖不足、Drone→Bird错误和三项OOF校准目标未通过；test、P95和部署继续阻塞。不得使用这15个development Unknown反调阈值或模型。

## Stage 050 - V14 Known-only训练与Parent-grouped OOF准备

目标：在Stage 049契约通过后，冻结V14三模型known-only训练与parent-grouped OOF执行协议，保证OOF用于阈值拟合而Unknown/validation/test不进入任何学习路径。
前置依赖：Stage 049全部focused/Smoke通过，known-only normalization和V14数据契约hash冻结。
状态：V14正式known-only联合训练已由用户手动完成，best epoch=`12`；5折OOF、known-only calibration和一次冻结development validation均已完成。1011个唯一known train parent完整覆盖，Unknown/validation/test拟合暴露为0；正式三checkpoint绑定通过。冻结开放集评估显示性能未达部署门槛。
最小子任务：

- [x] 冻结fold数=`5`、assignment seed=`42`、group单位=`parent`；1011个known train parent恰好一次held-out，Unknown OOF=0，validation/test access=0。
- [x] 冻结每折20 epoch、关闭early stopping、held-out不选epoch、只在最终epoch后单次预测；每折使用独立fold-train normalization和数据契约。
- [x] 实现正式known-only联合训练入口、单折OOF入口和真实OOF汇总/阈值拟合入口；正式artifact和部分OOF artifact现已由用户执行生成，既有产物由hash与拒绝覆盖契约保护。
- [x] 冻结阈值目标：仅用known OOF拟合Gate route、DBB knownness和Clutter acceptance边界，以known Precision/Recall和known false-rejection/coverage为约束；不得使用Unknown优化检测率。
- [x] 冻结calibration部署绑定：正式三配置语义hash、三checkpoint hash、共同epoch/global step、product selection hash、基础数据hash、assignment和fold contract hash必须完全匹配。
- [x] 获得用户对手动执行V14正式训练与5折OOF的确认。
- [x] 用户手动完成V14正式known-only三checkpoint联合训练；正式best epoch=`12`，三checkpoint原子集合已用于冻结预校准validation。
- [x] 完成并核验OOF fold 00/01：各20 epoch、各203个held-out parent、held-out仅最终单次预测、Unknown/validation/test exposure均为0。
- [x] 核验OOF fold 02恢复点：`joint_last_model.pt`为完整epoch 5、global step=`2020`、history count=`5`；未完成epoch 6不计入。
- [x] 创建`src-v3/scripts/v14_strict_open_set_oof_pipeline.ps1`：不使用环境变量，串行恢复/完成缺失fold、校准并执行一次冻结development validation；任一Python退出码、hash、范围或产物契约失败均停止。
- [x] 首次使用当前PowerShell和Windows PowerShell 5.1执行`-PreflightOnly`，验证当时fold 00/01通过、fold 02可恢复、fold 03/04未启动；完整运行后再次执行增强预检，验证5/5折、1011-parent聚合及calibration均通过且未启动validation。
- [x] 用户手动完成5折OOF训练与held-out预测；fold 00–02各203个held-out parent，fold 03–04各201个，均为20 epoch。
- [x] OOF完成后核验每个known train parent恰好一次out-of-fold预测、训练fold不含自身parent、Unknown exposure=0；五折合计1011个唯一known parent。
- [x] 仅用known OOF拟合已知接受率/误拒识约束下的三个阈值；Unknown threshold fit count=`0`，validation/test未参与。
- [x] 使用已冻结部署身份写出calibration artifact并通过严格loader/脚本回读；SHA-256=`af53913420ab9293885467812d3a1f0990a74eb8a7e7826edd1d5f44e40cf4e1`。
- [x] calibration冻结后单次执行development Unknown validation审计；报告覆盖217 known + 15 Unknown，未用于反调模型、epoch或阈值。
- [x] 执行冻结development层级拒识与stable评估；动作语义、分母和校准绑定均通过，但性能未达到放行门槛。
- [ ] test、P95和部署继续等待独立授权与验收。

验收条件：完整OOF provenance和Unknown零暴露可复核；校准只控制known false rejection，Unknown能力已在冻结development评估中测量且未达标。
是否需要正式训练：正式V14联合训练和五折OOF均已完成；无需重训，不需要预训练。
阻塞条件：Unknown FAR=`73.33%`、Unknown拒识Recall=`26.67%`，Clutter P/R/F1=`50.00%/62.50%/55.56%`，Drone Recall与Bird Precision/F1未达90%；test、P95和部署仍未授权、未执行。

## Stage 051 - V14 OOF断点续跑与冻结开放集评估脚本

目标：把剩余OOF、known-only阈值校准和一次冻结development validation五类开放集评估收敛为一个可从任意当前目录执行的PowerShell入口；禁止环境变量、并行fold、test、P95和development Unknown反向调参。
前置依赖：V14正式三checkpoint保持当前SHA-256；OOF assignment/fold contract保持当前SHA-256；fold 00/01完整，fold 02完整恢复点为epoch 5。
状态：脚本已串行完成全部五折、calibration和一次冻结development validation；首次运行的`$fieldName不在`插值问题修复后，最终报告及样本明细均已生成并通过范围、语义和部署绑定回读。流水线执行完成，性能验收失败。
最小子任务：

- [x] 从脚本自身位置解析项目根目录，直接使用`.venv/Scripts/python.exe`和固定V14路径，不读取或设置环境变量。
- [x] 串行处理fold 00–04：完成折只读复核；缺失预测且存在`joint_last_model.pt`时恢复；全新折从epoch 1开始；非空但不可恢复目录fail-closed。
- [x] 每折强制复核20 epoch、parent唯一性、train/held-out hash、fold normalization、held-out最终单次预测、Unknown/validation/test零暴露和三checkpoint文件hash。
- [x] 五折完成后强制复核1011个known train parent各恰好一次OOF，随后调用`report.strict_open_set_oof`并校验calibration仅来自known train parent-grouped OOF且绑定正式三checkpoint。
- [x] calibration冻结后仅调用`report.cascade --split val`一次，输出217 known + 15 development Unknown的五动作评估；Unknown语义固定为`UNKNOWN_REJECTED`动作而非学习类别。
- [x] 增加独占锁、外部进程退出码检查、JSON/hash/范围断言和PowerShell 5.1 UTF-8 BOM兼容；任一步失败退出码为1。
- [x] 当前PowerShell AST解析、Windows PowerShell 5.1 `-PreflightOnly`及`git diff --check`通过；预检未启动GPU训练或创建calibration/validation结果。
- [x] 用户执行默认完整模式并完成fold 02/03/04；五折均20 epoch且held-out prediction完整。
- [x] 脚本生成真实calibration；修复后增强预检已严格回读并验证其known-only OOF来源与部署绑定。
- [x] 修复PowerShell后置断言插值：`$fieldName不在`会被解析成未设置变量，现使用`${fieldName}不在`；同时让`-PreflightOnly`在产物存在时执行五折聚合、calibration和report后置断言。
- [x] Windows PowerShell 5.1增强预检通过：5/5折、1011个唯一known parent及calibration均通过，未运行validation。
- [x] 脚本生成并回读一次冻结development validation五类开放集报告。
- [x] 读取最终报告并复核五类性能、OOF结果、stable口径、深度瓶颈和下一步建议；未由development Unknown结果反调本轮阈值。

验收条件：脚本退出码0，5折总计1011个唯一known parent，Unknown/validation/test拟合暴露为0，calibration绑定正式三checkpoint，冻结validation为232 parent且含15个仅审计Unknown；test/P95/部署保持未执行。
是否需要正式训练：否；正式模型与五折OOF均已完成。重新执行脚本不会重训已完成fold。
阻塞条件：脚本无执行阻塞；性能阻塞为Unknown高置信误接收、Clutter低support/低Recall、Drone→Bird和校准子目标全部未通过。Test、P95和部署未执行。

## Stage 052 - V14冻结开放集结果审计与路线冻结

目标：只读核验V14最终validation、五折OOF和calibration产物，输出五动作性能、OOF结果、stable口径、统计限制、深层瓶颈与下一步单因素实验建议；不得重训、改阈值、访问test或把Unknown拒识动作写成学习类别。
前置依赖：Stage 051默认流水线退出并生成最终报告；正式三checkpoint、calibration和五份OOF预测保持冻结。
状态：已完成。V14严格open-set协议成立，性能未达到产品门槛；冻结为不可部署基线。
最小子任务：

- [x] 回读最终报告、sample detail、calibration和五份held-out prediction，核验SHA-256与父样本范围。
- [x] 核验development validation=`232` parent，其中known=`217`、仅审计Unknown=`15`；Unknown不是模型类别且没有参与阈值或checkpoint选择。
- [x] 复核五动作parent-final指标：Accuracy=`0.875000`、Macro Precision/Recall/F1=`0.770164/0.741996/0.733411`。
- [x] 复核Unknown安全指标：拒识Recall=`0.266667`、FAR=`0.733333`、known FRR=`0.009217`、Coverage=`0.974138`、Selective Risk=`0.119469`。
- [x] 复核stable：228/232确认；仅确认样本Macro-F1=`0.729919`，将4个未确认样本保守计为未命中后总体Accuracy=`0.870690`，stable没有解决Unknown拒识。
- [x] 从五份held-out records独立重算known OOF动作：Accuracy=`0.915925`、Macro-F1=`0.827326`、known coverage=`0.986152`；与calibration内置汇总一致。
- [x] 复核校准目标：Gate、DBB knownness、Clutter三项`all_precision_recall_targets_passed=false`；阈值来源有效但不具备部署放行资格。
- [x] 完成11/11统计谬误扫描，标记低support、基率失衡、固定development多版本反复查看、fold异质性、stable幸存者偏差和epoch协议差异。
- [x] 覆盖更新三个跟踪文件；不修改模型、数据、配置、checkpoint或calibration。
- [ ] test、P95、部署和新的正式训练；均未执行。

产物：`src-v3/processed/report/RADAR-LN3-V14-strict-open-set-calibrated-validation/evaluation-val.json`，SHA-256=`50a6387a18dc16d76a8cd1b700fac9e711cb427f08182b3cbaa7ba81e8367b18`；样本明细SHA-256=`cdbe8430f854c47d64a3c80f24ec483d9e413d753c6ca3cd556f362fb3343082`；calibration SHA-256=`af53913420ab9293885467812d3a1f0990a74eb8a7e7826edd1d5f44e40cf4e1`。
验收条件：结果与来源可追溯、Unknown语义无混淆、OOF与development分开解释、未执行项明确。技术验收满足，性能验收不满足。
是否需要正式训练：本Stage不需要且未执行。
阻塞条件：V14不可进入test或部署放行；下一轮必须先修复校准放行语义并在train/OOF范围内做DBB可拒识表征与Clutter覆盖实验，不能利用这15个development Unknown继续调参。

## Stage 053 - V14校准质量状态与部署Fail-closed修复

目标：修复V14校准“来源/结构通过但性能门槛失败仍显示passed”的安全缺口；把离线审计与部署加载分离，确保当前三项OOF门槛失败的校准只能用于冻结development审计，不能进入默认报告、在线状态机或部署动作。
前置依赖：Stage 052已确认Gate、DBB knownness和Clutter三项`all_precision_recall_targets_passed=false`，但历史v1 calibration与报告仍使用结构性`passed`。
状态：已完成。当前V14历史v1 calibration保持原文件和SHA-256不变，读取时从`selection_diagnostics`推导`audit_only/deployment_eligible=false`；未来新产物使用v2显式质量字段。默认deployment、策略构造和online注入均fail-closed；显式audit仅限V14冻结development validation。
最小子任务：

- [x] 审查calibration生成、loader、策略、offline evaluator、online cascade、报告CLI和PowerShell流水线调用链，确认原实现只校验来源/hash/数值范围。
- [x] 新增V14 calibration v2 schema，显式写入`calibration_quality_status`、`deployment_eligible`、三组件质量门槛和失败原因。
- [x] 兼容读取历史v1：必须具有完整`selection_diagnostics`，质量状态只可机械推导，禁止猜测或默认为可部署。
- [x] loader默认`purpose=deployment`；任一Gate/DBB knownness/Clutter P/R目标失败即抛出`PermissionError`。只有显式`purpose=audit`可读取audit-only校准。
- [x] `StrictOpenSetRejectionPolicy`默认拒绝audit-only calibration；即使调用方显式构造audit策略，`OnlineLiquidSoftCascade`仍拒绝注入。
- [x] audit-only策略只允许`development_open_set_audit` evaluator；禁止用于known checkpoint selection、train或test。
- [x] `report.cascade`新增`--calibration-purpose deployment|audit`，默认deployment；audit输出顶层`status=audit_only`、`execution_status=passed`、`deployment_eligible=false`和质量失败原因。
- [x] PowerShell流水线从真实诊断派生质量状态；历史报告兼容只读，新执行必须显式`--calibration-purpose audit`；流水线最终状态为`audit_only`而不是部署通过。
- [x] focused覆盖合格v2、失败v2默认拒绝、显式audit加载、历史v1推导、v2字段/诊断不一致拒绝、策略默认拒绝及online注入拒绝。
- [x] 使用真实V14 v1 calibration验证默认deployment拒绝、显式audit加载及三项失败原因；原calibration SHA-256保持`af53913420ab9293885467812d3a1f0990a74eb8a7e7826edd1d5f44e40cf4e1`。
- [x] 只读复算真实OOF汇总入口，返回`status=audit_only`、`deployment_eligible=false`且`write_status=reused_legacy_v1_with_derived_quality_status`。
- [x] 正式report CLI默认deployment用途在232-parent前向和写盘前拒绝当前calibration。
- [x] V14 focused、V13 hierarchical focused、compileall、静态隔离、PowerShell 5.1后置预检和限定diff检查通过。
- [ ] CPU有界V14 cascade Smoke：未执行；其既有安全前置检测到正式V14 artifact目录并按设计抛出`FileExistsError`，未绕过、未删除正式产物。
- [ ] 新模型训练、OOF重训、test、P95和部署：均未执行。

产物：代码级v2 schema和fail-closed路径；更新的focused结果与三份跟踪文件。历史calibration、正式checkpoint、配置、数据和冻结validation报告均未改写。
验收条件：当前V14失败calibration无法从默认loader、默认报告CLI或在线状态接口进入部署动作；显式audit仍可复现冻结development结果并清楚标记不可部署。已满足。
版本策略：模型、输入、split、阈值和checkpoint均未改变，继续冻结V14；calibration协议内部递增为v2。历史v1只读兼容，不覆盖、不迁移hash。
阻塞条件：部署安全语义缺口已修复，但模型性能瓶颈未改变；下一阶段仍需在train/OOF范围设计DBB可拒识表征和Clutter覆盖实验，当前V14不得部署。

## Stage 054 - V15 DBB角度Knownness单因素与嵌套OOF准备

目标：在冻结V14严格open-set语义、数据范围、液态容量和三层动作契约的前提下，只改变DBB specialist的读出几何，验证“显式归一化紧致DBB表征”能否改善known-only OOF上的DBB可接受性；同时修复V14折模型固定epoch 20而正式模型使用best epoch 12造成的校准尺度协议不一致。Unknown继续只作为冻结开放集评估真值，任何训练、归一化、checkpoint选择、OOF或阈值拟合暴露必须为0。

前置依赖：Stage 052确认V14 DBB knownness阈值=`1e-7`、冻结development Unknown FAR=`0.733333`；Stage 053已阻止该失败校准进入部署；Finding F157确认OOF固定epoch 20与正式best epoch 12不一致。V14正式checkpoint、OOF、calibration和development报告全部冻结只读。

状态：实施准备、正式Control/Treatment训练、两臂5折嵌套OOF和known-only校准均已完成。V15保持严格known-only open-set；Unknown在normalization、optimizer、checkpoint selection、OOF和阈值拟合中的暴露均为0。预注册主终点未通过：Control/Treatment的DBB knownness最弱Precision/Recall分别为`0.930041/0.915323`，两臂均低于`0.95`，Treatment还使known FRR由`0.016815`升至`0.032641`。两份校准均为`audit_only`且`deployment_eligible=false`。development Unknown一次冻结审计、test、stable、P95和部署均未执行。

最小子任务：

- [x] 冻结研究问题：V15只检验DBB specialist普通线性softmax读出与归一化角度原型读出的差异，不改变Gate、CU、液态隐藏维度、TR/RD输入、dt dynamics、采样、class weight、prefix权重或epoch预算。
- [x] 冻结严格open-set边界：Unknown不得成为类、prototype、loss target、hard negative、normalization来源、checkpoint selection样本、OOF样本或阈值拟合样本；15个development Unknown不得参与V15设计选择或调参。
- [x] 冻结配对设计：V15-Control保留普通线性DBB头，V15-Treatment仅使用L2归一化embedding/类原型与固定角度margin；两臂必须逐task独立派生初始化并核验非干预参数逐位一致。
- [x] 冻结归因指标：primary为known train parent-grouped OOF的DBB knownness最弱Precision/Recall及known FRR；secondary为DBB三类P/R/F1、Gate/Clutter护栏、parent-final五动作审计。Unknown development结果只能在模型和阈值全部冻结后执行一次，不参与胜负判定。
- [x] 冻结最小改进规则：Treatment必须在不降低Gate、Clutter、Drone/Bird护栏的前提下，提高DBB knownness最弱P/R；若Control/Treatment均未达到DBB P/R≥0.95，则只报告效应和失败，不把development Unknown结果用于继续搜索margin、scale或阈值。
- [x] 冻结V15版本理由：DBB输出头、训练logit几何、checkpoint/model contract及OOF选优协议发生实质变化，使用下一个全局版本V15；V3 index/split/TR/RD tensor、V14 known-only normalization和V14 parent assignment内容不变时只读复用，不伪造新数据版本。
- [x] 将严格open-set判定从单一`schema==V14`改为显式V14/V15 family，保证V15不会静默退回V13/闭集路径，且V13/V14历史checkpoint仍严格隔离。
- [x] 在配置/模型契约中加入DBB readout mode、embedding dim、固定scale、固定angular margin和knownness score语义；仅Treatment允许角度头，Control保持线性头。
- [x] 实现DBB归一化角度原型头；训练时只对真实DBB final/prefix状态施加固定target margin，推理时输出无margin cosine logits和显式top-cosine/top-gap knownness，不以训练margin污染部署分数。
- [x] 修改联合loss、离线评估、在线接口、OOF records与校准，使V15 DBB knownness消费显式角度分数；Gate/CU及soft evidence组合继续使用DBB三分类推理logits。
- [x] 建立逐task派生初始化协议，证明V15 Control/Treatment的Gate、CU和DBB非读出参数逐位一致；Treatment新增头参数不得改变其他模型的RNG初始化。
- [x] 创建V15 Control/Treatment三配置、V15模型/动作契约和隔离artifact路径；检查`data/`、配置与跟踪文件中的最高版本、内容和SHA-256，记录复用或递增原因。
- [x] 新建V15嵌套parent-grouped OOF协议：外层held-out不参与归一化、训练或选epoch；只在outer-train内确定epoch，随后以同一outer-train从冻结初始状态重训到该epoch，外层held-out仅前向一次。
- [x] 明确嵌套epoch selection不得使用Unknown、development validation或test；输出每折selection population、重训epoch、held-out forward count和全部parent/hash证据；新增outer-forward预写claim，崩溃后禁止自动重试该held-out前向。
- [x] 增加focused负向测试：Unknown暴露、V14/V15 checkpoint/calibration混用、角度margin用于推理、Treatment改变非干预初始化、held-out参与epoch selection、重复held-out前向和缺失cosine score均被拒绝。
- [x] 运行compileall、src-v3静态隔离、focused loss/gradient/finite/checkpoint/offline-online测试，以及不覆盖正式artifact的CPU有界forward；CUDA短Smoke未执行，避免在未创建的正式V15实验前扩大运行范围。
- [x] 覆盖更新三个跟踪文件并生成V15 training-readiness；只输出用户手动执行的Control/Treatment正式训练和后续OOF命令，不自动启动长时任务。
- [ ] Clutter parent/子流型覆盖审计：作为独立数据路线后续执行，不与本轮DBB头实验同时改变；只允许CQ08/raw且不得复制window伪造support。
- [x] 正式Control/Treatment训练：两臂各20 epoch完成，Control best epoch=`19`，Treatment best epoch=`6`；Unknown训练暴露和正式训练期间development Unknown访问均为0。
- [x] 完整嵌套OOF与known-only阈值拟合：两臂各5/5折，1011个known train parent各自恰好一次held-out前向；每折Unknown OOF、validation和test访问均为0，Control/Treatment parent集合完全一致。
- [x] 按预注册主终点完成Control/Treatment判定：Treatment没有提高DBB最弱P/R（`0.930041→0.915323`），known FRR恶化（`0.016815→0.032641`），因此拒绝“角度头改善DBB knownness”的主假设；不以Clutter次级改善覆盖该失败。
- [ ] 冻结development Unknown一次审计、test、stable、P95和部署：均未执行。

实验假设：若V14的失败主要来自普通softmax没有显式类内紧致/角度边界，则Treatment在known-only OOF上应提高DBB knownness最弱P/R并保持DBB三分类与Gate/CU护栏；若只提高训练分离度、未改善外层OOF，或通过增加known FRR换取表面Precision，则拒绝该Treatment。该假设不承诺改善Unknown，因为Unknown从未参与学习或选择。

版本与数据策略：全局模型/配置版本递增为V15；V3 index与split、V14 known-only normalization、原TR/RD tensor以及parent assignment在内容未变时复用并记录原SHA-256。因模型输出/loss/checkpoint与OOF execution contract变化，新建V15模型/动作契约、三配置、checkpoint schema、OOF execution contract和artifact目录；不得覆盖V14文件。

阻塞条件：在strict-open-set family、角度分数训练/推理分离、配对初始化、嵌套OOF隔离和focused Smoke全部通过前，V15不能进入正式训练；V15 OOF门槛失败时不能进入deployment，development Unknown也不得用于迭代调参。

## Stage 055 - V15角度Knownness实施准备与人工执行冻结

目标：完成V15 Control/Treatment训练、嵌套OOF和校准的可验证执行准备；严格保持Unknown零学习暴露、角度头单因素、冻结development Unknown和test封存。不自动执行正式训练、五折OOF、阈值拟合或development审计。

前置依赖：Stage 054设计冻结；V15 experiment contract SHA-256=`c86626be093e3d32b4fdfb268fdbd1ddc8c72f498e4802b6c4b05c5d2b71597`，nested OOF contract SHA-256=`bbed0a99a8623660b30b5d823e82eb5eed6165a9d84b1e2eb97ac8ca43b20326`；V14 artifact、OOF、calibration和development report只读冻结。

状态：正式Control/Treatment训练、两臂各5折嵌套OOF及known-only校准均已完成并复核。Control fold 02曾在outer retrain中断，后续从已确认安全的epoch边界恢复，最终所有10个outer held-out prediction/claim齐全。两臂校准均因Gate、DBB knownness和Clutter三组件目标未全部通过而标记`audit_only/deployment_eligible=false`。development Unknown一次冻结审计、test、stable、P95和部署仍未执行。

最小子任务：

- [x] 完成V15 strict-open-set family、DBB angular prototype readout、margin仅训练期应用、Control/Treatment独立calibration schema以及checkpoint/model binding。
- [x] 修复嵌套OOF合同二次构建误判：历史`write_status`仅视为运行时元数据，不改写既有合同或SHA-256。
- [x] 让`train.cascade`在V15正式训练结束时不读取development Unknown；训练摘要明确记录`development_unknown_access_count_during_formal_training=0`。
- [x] 实现`train.angular_knownness_oof`的inner selection→冻结初始化outer retrain→outer held-out单次前向；每次held-out前写入claim，若中断且不存在prediction则fail-closed，禁止自动重跑。
- [x] 实现`report.angular_knownness_oof`：聚合每臂5折、核验1011 known train parent恰好一次、校验fold派生config/hash/claim、只用known OOF拟合对应arm校准并绑定正式三checkpoint。
- [x] 新增`smoke.angular_knownness_contract_test`，验证margin生效且不污染推理、Unknown泄漏拒绝、paired初始化、offline/online cosine一致、calibration arm/score隔离、nested held-out/重复前向拒绝。
- [x] 新增`src-v3/scripts/v15_angular_knownness_pipeline.ps1`；`-PreflightOnly`完成幂等合同、focused和静态隔离检查，默认模式才依序执行人工授权的formal→OOF→calibration。
- [x] 运行全`src-v3` compileall、V15 focused、V14 strict-open-set回归、静态隔离和PowerShell 5.1 preflight；均通过。
- [x] 用户手动正式训练：Control与Treatment各20 epoch均完成；Control best epoch=`19`、Treatment best epoch=`6`，两臂`global_step=10120`，Unknown训练暴露与development Unknown访问均为`0`。
- [x] 用户手动完成两臂嵌套5折OOF：每臂fold parent数为`203/203/203/201/201`，合计1011且unique=1011；每折held-out前向次数为1，Unknown OOF、validation和test访问为0。
- [x] 完成每臂known-only校准并绑定正式三checkpoint：Control calibration SHA-256=`f95b08ae0f14abe4715be70b42f07eb237480c5039d924ccdfcea02883145c4b`；Treatment SHA-256=`d29b04e7a900e725329d1e29f22ecd381d3407ddaa7b5fadcc795458780fee06`。
- [x] 完成V15配对结果判定：Control DBB Macro-F1=`0.955973`、最弱P/R=`0.930041`、coverage=`1.0`；Treatment DBB Macro-F1=`0.950422`、最弱P/R=`0.915323`、coverage=`0.983607`。Treatment不满足预注册最小改进规则。
- [x] 核实用户观察到的`0.963904` DBB Macro-F1：它来自Control outer-04 inner-selection epoch 20、support=196的单折快照；该折冻结epoch 14，因为epoch 20的Clutter/CU产品护栏更差。该快照不是1011-parent OOF聚合结果，也未满足Bird/Balloon Recall均≥0.95。
- [ ] 冻结development Unknown一次审计、test、P95和部署，未执行。
- [x] 修复OOF阶段预检：`oof/calibrate`不再执行会拒绝非空formal artifact的可写`build.angular_knownness`；改用`--verify-only`核验冻结V15配置/契约，并以UTF-8读取正式训练摘要以兼容Windows PowerShell 5.1。
- [x] 只读核验V15 OOF中断恢复安全性：Control fold 02未生成held-out forward claim或prediction，inner selection已冻结；outer retrain的`joint_last_model.pt`保存第11个epoch完成状态，恢复时仅重做第12个训练epoch，随后才允许唯一held-out forward。

验证命令：`powershell -NoProfile -ExecutionPolicy Bypass -File F:\Project\RadarInsight\src-v3\scripts\v15_angular_knownness_pipeline.ps1 -PreflightOnly`。

产物：V15六份配置、两份数据/OOF契约、两臂正式三checkpoint及joint checkpoint、两臂各5折prediction/forward claim、两份正式checkpoint绑定的strict-open-set calibration、focused smoke和PowerShell执行入口。development Unknown审计报告、test、stable和P95产物仍不存在。

验收条件：Unknown不得进入模型、loss、normalization、checkpoint selection、OOF或threshold fitting；Control/Treatment不混用checkpoint或calibration；outer held-out不被用于normalization/epoch selection且只允许一次最终前向；均已在训练前focused检查通过。

是否需要数据重建：否；只读复用V3 index/split和V14 known-only normalization，V15只新增模型/OOF执行契约。

是否需要smoke：需要，已完成focused CPU contract；CUDA bounded smoke未执行。

是否需要正式训练：已由用户手动完成；Control摘要SHA-256=`9b1be7e2ac70014920ea6da13cf03ffd5a89498565aace16c4038d3fdfc28570`，Treatment摘要SHA-256=`ac4d2b94aa8fc8639909b69bd588566a610a5eb3c29770b0d4823cdfc0091803`。

阻塞条件：V15预注册DBB knownness主终点失败，Gate与Clutter目标也未通过；两份校准均不可部署。角度Treatment虽使known-only最终动作Macro-F1从`0.793345`升至`0.823156`、Clutter Recall从`0.314286`升至`0.428571`，但4/5折选择epoch不同，且联合全模型梯度裁剪使非干预Gate/CU训练状态分叉，不能把该次级收益直接归因于角度几何。development Unknown不得用于继续搜索margin、scale、epoch或阈值。

## Stage 056 - V15嵌套OOF结果审查与下一阶段冻结

目标：在不读取development Unknown和test的前提下，完成V15 Control/Treatment主终点判定、单折高分语义澄清、配对归因审查和下一阶段决策冻结。

前置依赖：Stage 055两臂formal、10个outer held-out prediction/claim及两份known-only calibration已完成；experiment contract与nested OOF contract SHA-256分别为`c86626be093e3d32b4fdfb268fdbd1ddc8c72f498e4802b6c4b05c5d2b71597`和`bbed0a99a8623660b30b5d823e82eb5eed6165a9d84b1e2eb97ac8ca43b20326`。

状态：已完成结果审查和主终点判定；性能未达到部署门槛，后续路线尚未实施。

最小子任务：

- [x] 独立展开两臂10份prediction JSON，确认每臂1011个known parent唯一覆盖、跨臂parent集合差异为0、每折held-out前向为1，Unknown/validation/test暴露均为0。
- [x] 生成并只读复核Control/Treatment校准；两者均为`audit_only`，Gate、DBB knownness、Clutter三组件均未全通过目标。
- [x] 报告预注册primary：Treatment DBB最弱P/R较Control下降`0.014719`，known FRR增加`0.015826`，角度knownness候选按预注册规则不保留为当前改进方案。
- [x] 报告secondary：Treatment accepted accuracy提高`0.008116`，known-action Macro-F1提高`0.029811`，Clutter F1提高`0.114035`，但overall known accuracy下降`0.006924`，Drone Recall下降`0.028689`。
- [x] 完成parent配对诊断：Treatment DBB knownness拒绝16个DBB parent，其中10个raw误分类、6个raw正确；说明gap具有一定错误富集能力，但代价是拒掉正确known，尚未形成满足0.95门槛的边界。
- [x] 完成known-only排序诊断：Treatment低cosine-gap识别raw DBB错误的ROC-AUC=`0.918093`、AP=`0.418002`，但Control低softmax-gap ROC-AUC=`0.919322`、AP=`0.331366`；角度头改善极低gap区域的错误浓度，没有改善总体ROC排序或最终P/R门槛。
- [x] 完成归因护栏：两臂只有outer fold 04选择相同epoch；同epoch下Gate/CU参数也已分叉。训练日志抽样中Control/Treatment `grad>100`分别4/220与39/220，而代码对全模型统一`clip_grad_norm_`，因此DBB角度损失会间接改变Gate/CU更新尺度。
- [x] 完成统计谬误扫描：禁止以单折inner快照替代外层OOF，禁止忽略Clutter低base rate，禁止从accepted subset推断全体性能，禁止把联合训练关联差异写成角度头直接因果收益；11/11类型已检查，主要风险为look-elsewhere、base-rate neglect、selection/survivorship和因果归因。
- [ ] 实施下一候选：未执行。优先候选为将DBB角度knownness实验改成真正隔离的DBB specialist单任务/冻结Gate-CU训练，或先进行Clutter parent/子流型覆盖审计；一次只能选一个主要变量。
- [ ] development Unknown一次冻结审计：未执行。由于两臂primary均失败，该审计若执行只能描述失败候选的开放集行为，不得用于选择或调参。
- [ ] test、stable、P95、部署：未执行。

验收条件：V15主终点、次终点、单折快照、OOF聚合、阈值来源和归因限制均已区分；没有把Unknown、test或未执行指标补写为结果。已满足。

是否达到训练/部署状态：代码与OOF执行链可复现，但当前模型不达到部署状态；无需重复V15训练。下一实验需先冻结单一问题定义和归因隔离方式。

阻塞条件：DBB最弱P/R、Gate CU和Clutter均未达门槛；联合优化使非干预模块分叉；Clutter仅35个known train parent。V15不得进入test、stable、P95或部署。

## Stage 057 - V15测试冻结与三层产品契约失效复核

目标：按`.rules`在进入V16修复前保存V15实现、配置、focused测试、正式训练和两臂nested OOF的可复现快照；同时明确V15回答的是known-only DBB knownness研究问题，不是当前冻结的“Gate识别DBB/CU、DBB三分类、CU真实Clutter/Unknown二分类”产品问题，禁止继续把V15指标解释为真实CU或五类产品性能。

前置依赖：Stage 056已完成两臂正式训练和两臂5/5 nested OOF；用户随后提供代码审查，要求暂停V15产品路线，修复真实CU标签、分层路由、split、dt质量、监督/评估单位和summary语义后，再分别训练Gate、DBB specialist与CU specialist，暂不进行完整级联联合训练。

状态：V15实验执行完整性已复核，产品契约失效已确认。V15冻结为“known-only角度knownness诊断、不可作为产品候选”；V16实现、数据重建、三任务独立训练和评估尚未执行。

最小子任务：

- [x] 复核V15 Control/Treatment正式训练摘要：两臂均完成20 epoch；Control/Treatment best epoch分别为`19/6`，摘要SHA-256分别为`9b1be7e2ac70014920ea6da13cf03ffd5a89498565aace16c4038d3fdfc28570`和`ac4d2b94aa8fc8639909b69bd588566a610a5eb3c29770b0d4823cdfc0091803`。
- [x] 复核两臂nested OOF当前真实状态：Control与Treatment均为5/5折，10份prediction和forward claim均存在；更早“Treatment仅3/5”的审查状态已经过期，不写入最终冻结结论。
- [x] 复核V15 CU训练目标：输出为`CLUTTER_ACCEPT_EVIDENCE/NON_CLUTTER_KNOWN_EVIDENCE`，负类来自Drone/Bird/Balloon，UnknownTarget被loss、normalization、OOF和阈值拟合排除；该指标不能解释为Clutter/Unknown二分类。
- [x] 复核V15动作路由：Gate接受DBB但DBB knownness未通过时直接产生`UNKNOWN_REJECTED`，没有进入CU specialist；与当前产品设计要求的“DBB不接受时回退CU，再由CU判Clutter/Unknown”不一致。
- [x] 复核V15训练、checkpoint选择、nested OOF和calibration均为known-only；development Unknown、test、stable、P95和部署均未执行，因此V15不能输出真实Unknown Recall、FAR、FRR或五类产品放行结论。
- [x] 复核V3 split仅按`track_id`隔离；CQ08 manifest含1550条记录和45个重复capture组。独立hash核验确认1339/train与1490/val、1343/val与1494/train分别共享capture、raw SHA-256和RD NPZ SHA-256，现有validation存在同源暴露风险。
- [x] 复核10个跨标签重复capture：每组raw SHA-256和RD NPZ SHA-256均不相同，当前没有“跨标签完全等价内容”证据，不申请改标签；V16仍按capture整体同split隔离。
- [x] 复核offline/online `gap_fraction`把`tr_dt_reason==2`的同秒观测累计为gap；该reason在V3契约中是秒级精度下的合法bounded observation，不是缺帧。
- [x] 复核V15主损失以parent-final为主、prefix仅0.10辅助且checkpoint只按parent产品护栏选优；prefix/window、parent-final和stable没有形成同一验收协议。
- [x] 复核训练摘要无条件写`status=passed`；Control/Treatment实际产品护栏仅`1/4`和`0/4`，流程完成与性能达标语义存在混淆。
- [x] 冻结V15为非产品诊断版本；保留实现、配置、实验/OOF契约、focused测试和结果来源，不修改checkpoint、calibration或历史指标。
- [x] 运行V15 angular focused、V14 strict-open-set focused、历史causal cascade回归、全`src-v3` compileall、静态导入隔离、V15 OOF只读preflight和`git diff --check`；最终均通过。静态隔离最初两次因工作目录和参数名调用错误未执行，按实际CLI `--root`重跑通过，不属于代码失败。
- [ ] 建立V16 capture/content provenance、无跨split等价组的新index/split、全train normalization和三任务数据契约；未执行。
- [ ] 修复真实Gate/DBB/CU独立训练、独立checkpoint和独立评估入口；未执行。
- [ ] V16三任务正式训练、完整级联、OOF拒识校准、development/test、stable和P95；均未执行。

验收条件：V15代码与实验结果可复现，但所有产品性能解释均明确失效；V16不得加载V15 normalization、checkpoint或calibration，不得沿用V3存在同源跨split暴露的split宣称泛化。

是否达到训练状态：V15无需重训且不可部署；V16尚未达到训练状态。

阻塞条件：V16必须先完成provenance分组、真实Clutter/Unknown监督、三任务独立loss/checkpoint、同秒非gap语义和流程/性能状态分离。任何跨标签完全等价内容若存在且未经人工政策裁决，必须阻止正式训练。

## Stage 058 - V16真实三任务局部训练链与Capture/Content隔离

目标：建立不依赖V15 known-only语义的V16产品三任务局部基线。第一任务为DBB/CU Gate，使用Drone/Bird/Balloon versus Clutter/Unknown；第二任务为DBB specialist，仅使用Drone/Bird/Balloon；第三任务为CU specialist，仅使用Clutter/Unknown。三个模型分别训练、分别选best、分别评估，本Stage禁止完整级联联合训练、拒识阈值拟合、test、stable、P95和部署。

前置依赖：Stage 057已把V15提交为非产品诊断快照`b6cd7e879fe0a78ac15022de781042b8a23aa140`；已确认V3按track_id切分且至少两组同capture/raw/RD内容跨train/val；已确认10个跨标签capture的raw/RD内容不同，当前无需标签修改申请。

状态：实现与训练准备已完成，等待用户确认后手动启动三任务正式20 epoch训练。当前没有V16正式分类性能；完整级联、OOF拒识校准、test、stable、P95和部署均未执行。

最小子任务：

- [x] 新建V16 provenance构建器，读取1546个V3 parent及CQ08 unified manifest，计算capture ID、原始raw/track/point SHA-256、RD NPZ SHA-256、去除轨迹ID后的规范track/point SHA-256和模型输入语义SHA-256。
- [x] 使用capture/raw/RD/规范track/规范point/模型输入语义任一相同建立同源连通组件；跨标签仅capture相同允许保留原标签，跨标签内容等价则fail-closed并请求人工裁决。实际得到1501个组件、45个多parent组件、跨标签内容等价冲突0组。
- [x] 采用保守split迁移：组件内存在旧test则整体test，否则存在旧val则整体val，否则train；27个parent迁移，旧val/test进入新train为0。
- [x] 创建`RADAR-LN3-V16-cq08-capture-content-provenance-manifest.json`、`...grouped-split.json`和`...grouped-index.csv`；capture/raw/RD/规范track/规范point/模型输入语义六类键跨split违反数均为0。
- [x] 只用V16 train全五类parent重算RD normalization；train/val/test parent=`1055/247/244`，normalization parent=1055、排除parent=0、selected/excluded RD路径交集=0。
- [x] 新增V16产品局部任务schema：Gate映射DBB/CU，DBB三分类，CU真实Clutter/Unknown；CU train/val support=`33/69`与`9/15`，Unknown已真实进入CU监督。V16 checkpoint绑定配置identity和V16数据hash，拒绝旧版本或原位改写配置。
- [x] 将V16局部任务主loss改为每parent最终合法状态CE，保留从第3事件开始、排除final的0.10 parent-balanced prefix辅助；class-weight分母与weighted CE一致，padding与event<3非final梯度均为0。
- [x] 三任务分别使用`parent_min_precision_recall`选best，门槛Gate≥0.95、DBB≥0.95、CU≥0.90；performance护栏同时检查逐类Precision/Recall/F1，prefix/window与parent-final独立报告。
- [x] 修复训练摘要：V16局部与未来联合训练均使用`status=completed|failed`描述流程，独立`performance_status`描述性能，`deployment_eligible=false`且未执行项逐项记录。
- [x] 修复同秒dt质量：全量first/zero/positive=`1546/9776/18341`；同秒只进入`zero_dt_fraction`/`same_second_fraction`，当前V16真实`gap_fraction=0`，offline/online共用同一计算函数。
- [x] 修复DBB knownness未接受后的动作拓扑：先显式回退CU，CU接受Clutter才输出Clutter，否则输出`UNKNOWN_REJECTED`；旧V14/V15直达Unknown动作契约被fail-closed，历史artifact未改。
- [x] 阻止V16 product-local三checkpoint进入通用cascade组装/报告入口；本Stage只允许三个局部模型独立训练和评估，完整级联必须使用后续显式版本。
- [x] 运行配置/数据纯度、split隔离、label、shape、finite、loss梯度、checkpoint保存恢复、prefix/parent评估、全`src-v3` compileall、静态import隔离、V13/V14回归和三任务CPU/CUDA各1步训练入口Smoke；最终均通过。
- [x] 覆盖更新三个跟踪文件并请求用户确认；确认后仅输出Gate、DBB、CU三条正式训练命令和三条独立validation评估命令。
- [ ] 三任务正式20 epoch训练与独立validation评估：待用户手动执行。
- [ ] 完整级联联合训练、三checkpoint组装选优、parent-grouped OOF、正式拒识、development/test、stable、session、P95和部署：均未执行。

实验门槛：Gate的DBB/CU两类Precision、Recall、F1均至少0.95；DBB的Drone/Bird/Balloon各自Precision、Recall、F1均至少0.95；CU的Clutter/Unknown各自Precision、Recall、F1均至少0.90。低support和单split结果只作为局部筛选，不替代最终OOF/test。

版本策略：split规则、index schema、train population、normalization、标签映射、loss、checkpoint和配置语义均实质改变，使用全局新版本V16。原V3/V13/V14/V15文件只读保留，不覆盖、不复用checkpoint或calibration。

是否达到训练状态：达到三任务独立正式训练准备状态；尚未达到性能通过、完整级联、严格开放集拒识或部署状态。正式20 epoch与独立validation必须由用户确认后手动执行。

阻塞条件：当前无代码或数据契约阻塞影响三任务独立训练。性能结论仍被正式训练未执行和CU validation低support（Clutter=9、Unknown=15）阻塞；完整级联、OOF拒识、test、stable、P95和部署继续禁止执行或宣称。

## Stage 059 - V17 DBB微多普勒摘要单因素配对

目标：验证当前事件原始复数I/Q中的帧内微多普勒时间结构能否补充已被现有MTD/RD平均弱化的信息，并改善DBB specialist中的Drone/Bird判别。Control为V16 DBB基础输入加既有四维`rd_physical`，Treatment仅再加入三维微多普勒摘要；不改变Gate、CU、拒识、stable、液态容量、loss、split或checkpoint选择协议。

前置依赖：V16 capture/content grouped index、provenance、全train normalization和DBB label/split保持冻结；原始CQ08解析已由12文件审计及Track 333专项审计验证，V17代表帧再次与现有MATLAB目标距离行交叉核验。

状态：target-anchored v3全量prepare、两臂20 epoch正式训练与paired validation均已完成；冻结结论为`completed_but_gate_failed`。Treatment own-best仅使Bird Recall增加1.79个百分点，同时Bird Precision下降4.66个百分点、Bird F1下降1.44个百分点、Macro-F1下降0.37个百分点，并把Drone→Bird由3增至6。该方向在V17失败收口，不进入多seed、完整级联、OOF、test、正式拒识、stable或P95。

最小子任务：

- [x] 冻结三维Treatment输入：峰频带能量时间CV（截断0–2后除2）、`-10 dB`物理速度谱展宽占64速度bin比例、自相关调制周期（截断0–0.1秒后除0.1；不可估计为0）。
- [x] 冻结当前事件因果范围：每个TR事件只消费V16 index映射的当前raw frame；禁止读取先前/未来事件、parent统计、标签、路径、sample ID或模型输出。每事件最多12帧，摘要按帧中位数聚合。
- [x] 使用MATLAB审计口径按Fortran列序解析`[31,PRTnum]`复数I/Q；以Taylor MTD和目标record多普勒邻域复现既有局部目标距离行，再对该距离行执行Hann 50%重叠、FFT=128的STFT。
- [x] 修复raw帧映射语义：确认V16字段`rd_raw_frame_indices_json`实际继承自`enumerate(rd_mat_files)`，属于MAT列表零基序号，不能直接索引完整`.dat`帧列表。V17现按`parent_sample_id+rd_tensor_frame_indices_json`回连SHA-256绑定的canonical frame index，从`rd_mat_path`文件名解析一基采集帧号，再减一得到真实raw-DAT帧号，并逐帧核验track、track point、tensor path和raw target record。
- [x] 建立完整STFT证据缓存：每帧`64×16 float16`，速度轴`[-85,85] m/s`；完整张量、raw frame index、逐帧原始摘要、目标距离行及采样元数据只用于证据审计，不进入本轮模型。
- [x] 将STFT证据缓存升级为v2并分别保存`legacy_mat_list_ordinals_zero_based`、`canonical_tensor_frame_indices_zero_based`、`mat_acquisition_frame_numbers_one_based`和`raw_dat_frame_indices_zero_based`；旧MAT ordinal永不再作为raw-DAT索引。
- [x] 修复全局STFT峰夺取目标频带：在已由局部MTD无标签定位的目标Doppler±8 m/s内选择最强积分STFT峰，CV与调制周期围绕该目标局部峰计算，`-10 dB`展宽改为围绕目标局部峰的连续谱瓣；全局STFT峰仅保留为证据诊断，不进入模型。为防止目标锚定后的几何覆盖检查按定义通过，另冻结独立能量门槛：至少90% train帧的目标局部峰/全局峰功率比≥0.10（即目标峰不低于全局峰−10 dB）；validation只报告。summary/STFT、paired protocol、checkpoint、readiness分别升级到v3/v3/v3/v4，并使用独立`target-anchored-v3`缓存、配置、artifact和报告路径，旧v2 blocked产物只读保留。
- [x] 建立版本化Control/Treatment共享缓存、manifest、data contract和配置生成器；只处理DBB train/val，test raw/RD payload访问固定为0；不同既有文件拒绝覆盖，Windows ACL继承后立即回读校验。
- [x] 扩展Dataset、Fusion、checkpoint和在线同源接口。两臂共同使用四维`rd_physical`；Treatment新增三列。微多普勒valid必须与RD-physical valid逐值相同，valid mask仅门控、不作为分类特征。
- [x] 建立配对初始化：共享40个参数/缓冲条目逐位相同，Control/Treatment Fusion输入维度`83/86`，Treatment新增3列严格置零，初始logits逐位相同，新增列梯度非零。
- [x] 冻结训练协议：seed/DataLoader seed=42，20 epoch，关闭early stopping，保存每epoch checkpoint，保持V16 parent-final主损失加0.10 prefix辅助，parent顺序逐epochhash一致。
- [x] 冻结train-only readiness：三特征非退化、调制周期可估帧≥5%、至少一项Drone/Bird robust effect绝对值≥0.10；同时审计PRTnum/PRT秒/载频与标签的Cramér's V≤0.50，要求至少一个共享采样模式内的Drone/Bird effect仍≥0.10，并要求至少90%逐帧STFT积分峰的±8 m/s频带覆盖局部MTD目标多普勒。validation仅报告，不参与是否启动训练。
- [x] 将readiness v4作为训练入口硬依赖：同时绑定Control/Treatment配置路径与SHA-256、数据contract hashes、canonical frame-index SHA-256和所有train-only checks；直接调用`train.tasks`也不能绕过prepare，训练摘要记录readiness SHA-256。冻结V16下train必须检出Track 1010共46个legacy/raw偏移帧且val为0，否则阻止训练。
- [x] 冻结paired验收：Treatment三类逐类Precision/Recall/F1均≥0.95；parent Macro-F1及Drone/Bird两类中较弱的P/R/F1 floor严格改善；Drone→Bird和Bird→Drone均不增加且双向错误至少减少1个parent；Balloon正确parent最多损失1个；至少两个不同same-epoch归因护栏改善（若三锚点实际只有一个不同epoch，则该唯一护栏必须改善）。
- [x] 修正旧可行性探针结论：`Drone CV=0.52 vs Bird CV=0.13`来自错误Python I/Q reshape，已失效，不能作为收益证据。MATLAB口径三条代表事件归一化CV为Drone Track 1=`0.127563`、Bird Track 333=`0.181750`、Balloon Track 3010=`0.060699`，仅作Smoke而非分布结论。
- [x] focused Smoke通过：Drone/Bird/Balloon各3帧共9帧目标距离行与MATLAB逐值一致；Control/Treatment初始logits一致；新增列梯度最大值=`0.666845`；offline/online首事件logits误差=0；两臂checkpoint round-trip通过。
- [x] target-anchored v3 focused Smoke通过：代表帧Python局部MTD Doppler与MATLAB `detected_doppler_mps`最大误差≤`0.01083 m/s`；合成强零速杂波反例中全局峰误差=`20.8982 m/s`，目标局部峰误差=`0.6891 m/s`且保持在±8 m/s内；缓存字段、readiness绑定和配对初始化通过。
- [x] Track 1010缺MAT反例Smoke通过：`.dat`共54帧、MAT仅保留采集帧9–54；canonical tensor 0/legacy ordinal 0对应MAT采集帧9和真实raw-DAT index 8。旧index 0落到track point 1且与MAT I/Q最大绝对差=`277.504744`，新index 8落到track point 3且MAT I/Q误差=`0`。46/46保留MAT帧的I/Q误差均为0且track point全部匹配；真实`_build_split`内存Smoke完整处理18事件（16有效、2缺RD），四套索引序列精确为legacy/tensor=`0–45`、MAT=`9–54`、raw-DAT=`8–53`。全canonical frame index共106026行，只有Track 1010的46行发生偏移，统一为`+8`。
- [x] V16 product-local回归、全`src-v3` compileall、静态隔离、PowerShell AST和`git diff --check`通过；未运行全量V17数据处理或正式训练。
- [x] 新建`src-v3/scripts/v17_micro_doppler_paired_pipeline.ps1`，支持`prepare/train/evaluate/all`，不使用环境变量；任一步失败停止，已完成arm可只读复核；`status=failed`且存在完整`last_model.pt`时可恢复，epoch20后只重建最终评估/摘要且不重复optimizer step，其他非空状态fail-closed。
- [x] 用户手动运行target-anchored v3 `prepare`：train/val缓存、data contract、manifest和配置构建完成；train-only readiness v4全部检查通过，validation不参与放行且test payload访问为0。
- [x] 用户手动运行Control/Treatment正式20 epoch训练及paired validation：两臂parent顺序和初始logits护栏通过；paired状态为`completed_but_gate_failed`，三类0.95门槛、own-best Macro-F1、Drone→Bird和双向错误门槛未通过。
- [x] 完成parent-final错误簇归因：10条轨迹改变预测，5条由对变错、4条由错变对、1条错误类改变；Bird正确数仅`50→51`，Drone→Bird为`3→6`，总正确数`209→208`。
- [x] 完成same-epoch/checkpoint归因：epoch 15和20同epoch下Treatment Bird Recall均不高于Control；own-best Recall上升受独立best epoch混杂，epoch 18/19仅作诊断且不事后追认best。
- [x] 完成采样模式/物理语义归因：全局Bird高CV/宽谱方向在PRTnum=142模式中反转，Track 4237/4238为该模式下极端Drone并从首个eligible prefix持续被判Bird；优先瓶颈为mode-invariant特征尺度与周期可估语义，不是简单液态容量或末期状态漂移。
- [ ] 多seed确认、完整级联、OOF、test、正式拒识、stable、P95和部署：均未执行；仅当seed42局部paired门槛通过后再讨论多seed。

实验假设：若现有MTD/RD聚合确实丢失对Drone旋翼与Bird扑翼有用的帧内幅度调制结构，则Treatment应在同初始化、同parent顺序、同RD-physical输入下减少Drone↔Bird错误；若train-only特征只反映采样模式、不可估计率过高，或Treatment只改善train loss/单个epoch而未通过不同same-epoch护栏，则拒绝该方向。

版本策略：完整STFT证据、三维输入、cache/manifest/data contract、Fusion维度和checkpoint contract均发生实质变化，使用新版本V17。V16 index/split/provenance/normalization内容只读复用并绑定原SHA-256；不覆盖V16 checkpoint或配置。

是否达到训练状态：正式局部配对已训练并完成validation，但性能门槛失败；V17不再处于“待训练”，而是“实验完成、不得晋级”状态。

阻塞条件：Treatment三类逐类P/R/F1未达到0.95，own-best Macro-F1下降、Drone→Bird和双向错误增加，same-epoch归因门槛失败；因此阻止V17多seed、完整五类、开放集、CU、stable和部署实验。若未来重开，必须使用新版本并先冻结mode-invariant物理摘要、feature-specific可估语义和checkpoint层级规则，不能在本V17结果上回看validation调参。

## Stage 060 - V18宽带多尺度微多普勒物理语义修复

目标：核实并修复V17三维摘要遗漏Drone关键宽带旋翼侧带、把不可估周期混入普通数值以及PRT=142采样模式方向反转的问题。在冻结V16 DBB局部任务、capture/content split、液态容量、loss、seed、20 epoch和四维`rd_physical`基线的前提下，Control保持`rd_physical4`；Treatment经历39维候选、21维去冗余候选后，最终只新增8维经train-only验证的模式稳定当前事件摘要。只有冻结validation相对Control产生严格超过2个百分点的实质增益时才采纳。

前置依赖：Stage 059已证明V17 Treatment未通过；Finding F196-F198确认Bird Recall表观上升伴随Precision/Macro-F1下降、own-best受epoch混杂、PRT=142方向反转。V16 index、provenance、normalization和DBB标签保持冻结，test payload继续封存。

状态：V18代码、focused Smoke、训练前readiness、配对训练/评估脚本和采纳门槛已实现；全量prepare、正式Control/Treatment训练和paired validation等待用户手动执行。完整级联、OOF、test、正式拒识、stable、P95和部署均未执行。

最小子任务：

- [x] 核实V17传入链路：cache/sample_id对齐、valid门控、配对初始化、Fusion拼接、offline/online和checkpoint round-trip没有发现会解释本轮负结果的硬错误；主问题属于提取器物理语义。
- [x] 修正时间可辨识结论：PRT=142原始脉冲采样Nyquist足以覆盖部分高频调制，但其单帧时长仅约4 ms且train仅Drone/Bird/Balloon=`4/3/0` parent，无法建立三类稳定校准；PRT≥360的旧STFT hop Nyquist约306–326 Hz，不能可靠描述400–800 Hz旋翼闪烁。Bird 5–15 Hz在所有单帧4–46 ms内也不足一个15 Hz周期，禁止输出虚假的扑翼周期数值。
- [x] 冻结V18新版本：输入维度、提取公式、证据cache、checkpoint排序和验收协议均改变，不覆盖V17；V16 index/split/normalization只读复用并绑定SHA-256。
- [x] 实现相干主体消除：在局部MTD目标Doppler±8 m/s内细化主体频率，拟合并消除一个复相干正弦，后续侧带统计只读取残差，避免主体回波淹没微动。
- [x] 将速度侧带扩展至相对主体`±90 m/s`，分别提取正负8–30、30–60、60–90 m/s能量、总侧带能量、RMS/q90范围、正负对称性、按Rayleigh分辨单元计算的熵/集中度/峰数和侧带SNR。
- [x] 修复Doppler Nyquist边界：主体搜索和相对侧带速度使用`[-PRF/2,PRF/2)`有符号循环频差；主体靠近无模糊速度边缘时，跨边界的真实旋翼侧带保留一次且不重复。focused合成回归要求跨边界的正负`30–60 m/s`侧带同时非零。
- [x] 保持相对侧带平移不变：移除对绝对零速`±2 m/s`的额外截断。主体相消和邻距离行噪声底承担杂波抑制；真实侧带即使从运动主体跨过绝对零Doppler也应保留。focused合成回归要求跨零后的正负`30–60 m/s`侧带同时非零。
- [x] 修复噪声底：不再用目标行30–90 m/s区域自身估计噪声，改用局部MTD目标距离邻域十个非目标行的逐频率中位数估计指数噪声均值，防止真实宽带旋翼侧带被当作噪声扣除。
- [x] 直接从原始脉冲包络提取300–600、600–1200、1200–2000 Hz调制幅度、高频占比、中心频率、top3集中度、熵和SNR，绕开旧STFT hop Nyquist；删除V17不可估period普通数值输入。上限冻结为2 kHz，确保所有候选PRT/PRI帧都满足Nyquist，而不是仅对部分PRT=226子模式生效。
- [x] 修正包络调制幅度/噪声语义：各频带幅度直接由一侧Parseval带内能量换算RMS，不再用全包络RMS乘归一化频带占比；调制噪声底也由邻近非目标距离行逐频估计，不从目标自身300–2000 Hz宽带中位数扣除，以免把宽带旋翼调制当作噪声。
- [x] 为每帧增加物理可观测性门控：不能只按`PRTnum`合并，冻结为8个`脉冲数+PRI+载频`完整采样模式；每个模式在train三类各至少8 parent、速度Nyquist≥90 m/s、包络调制Nyquist≥2000 Hz。现有V17只读证据核实最小模式支持仍为Drone/Bird/Balloon=`33/41/25`；未通过门控的帧不参与21维聚合。
- [x] 对PRT=142实施证据中性化：完整23维逐帧证据和审计谱保留，模型21维干预严格为0，valid仍与当前事件RD存在性一致；PRT编号、载频、CPI和可靠性标记均不进入分类器，避免模态存在性或采集模式捷径。
- [x] 枚举并冻结全部train采样模式：除8个校准模式与2个PRT=142模式外，还存在两个validation缺失的低支持模式`226×48µs×15.6GHz`和`226×51µs×16.3GHz`，各仅4个Drone、0个Bird/Balloon parent；两者也只保留审计证据并严格零干预。出现合同外新模式将阻断prepare。
- [x] 修订并冻结21维事件聚合：23项逐帧物理证据与完整181点残差Doppler/64点包络调制profile继续完整保存，仅模型摘要裁剪为18项median与3项q90。median删除正负60–90 m/s方向带、正负对称性、分辨单元熵和600–2000 Hz fraction；保留正负8–30/30–60、total、RMS/q90范围、top4 concentration、peak count、SNR、包络RMS、三调制带、centroid、top3、调制熵/SNR。q90仅保留正负8–30与total。
- [x] 裁决帧级恒等与事件聚合差异：每帧`total=sum(6个方向带)`误差为0，但`median(frame total)`一般不等于六个方向带median之和；构造反例得到`1.0`对`0.0`。因此保留median total作为典型总侧带能量，删除由方向带直接决定的symmetry；q90 total保留为瞬态总侧带闪烁量。
- [x] 创建V18全量构建器：复用V17已审计的canonical tensor→MAT采集帧→真实raw-DAT帧映射，保留Track 1010缺MAT `+8`偏移反例；只处理DBB train/val，test raw/RD payload访问为0。
- [x] 将cache/Dataset/Fusion/online/checkpoint合同修订为21维；Control/Treatment共享40个参数/缓冲条目逐位一致，Fusion输入维度`83/104`，Treatment新增21列严格置0且获得非零梯度。summary schema、paired protocol、manifest、checkpoint、readiness、配置、artifact和报告路径均升级为独立`pruned21` v2合同，旧39维产物不能混用。
- [x] 冻结checkpoint排序：先最大化三类全部Precision/Recall/F1中的最弱值，再最小化Drone↔Bird双向parent错误，最后比较Macro-F1；`+0.02`采纳门槛不参与epoch选择，防止事后按Treatment优势挑epoch。
- [x] 冻结paired采纳门槛：own-best parent Macro-F1绝对提升`>0.02`，Drone/Bird较弱P/R/F1 floor绝对提升`>0.02`，Drone→Bird和Bird→Drone均不增加且双向总错误至少减少1，Treatment三类各自P/R/F1均≥0.95，并要求至少一个预冻结same-epoch锚点也通过严格`>0.02`增益门槛。
- [x] 实现train-only readiness：21维逐项非退化；以parent均衡均值为统计单位，至少4项Drone/Bird特征同时满足方向无关AUC≥0.55和robust effect绝对值≥0.10，并要求其方向在全部8个校准完整采样模式中不反转、至少两个模式的绝对effect≥0.10；任何全局已达上述判别门槛但模式内反转/支持不足的特征都会阻断训练，不能让危险列混入21维；各模式三类train parent均≥8；PRT=142支持固定为`4/3/0`并保持零干预；summary必须从逐帧证据精确重建。validation只报告，不决定是否启动训练。
- [x] `pruned21` focused Smoke通过：模型21维名称与顺序精确固定，旧39维中恰好18列不再进入Fusion；23维逐帧证据不变；真实三类代表事件shape/finite；合成相干主体+宽带侧带与900 Hz调制、跨Nyquist/跨绝对零速侧带均保留；PRT=142证据保留而模型向量精确为0；21维cache→Dataset/Collate逐值一致；Fusion维度`83/104`，新增21列初始为0且梯度最大值=`0.921369`；offline/online误差0；checkpoint round-trip、readiness绑定和Track 1010映射通过。
- [x] 用户手动执行`pruned21 prepare`后，train-only readiness按设计阻断；唯一失败项为`no_globally_informative_drone_bird_feature_reverses_or_lacks_mode_support`。21维中16项达到全局AUC/effect门槛，仅10项跨8个校准模式不反转；6个危险全局特征为median total、600–1200 Hz调制幅度、调制centroid、top3 concentration、q90正8–30和q90 total。median/q90 total在两个PRT=900模式强反转，前四项也存在绝对effect≥0.10的真实反转，不能通过忽略近零符号或降低门槛放行。
- [x] 冻结`stable8` v3修订：从10项全局有效且模式一致的候选中，再将与RMS高度共线的q90 abs offset（Spearman约0.946）和量化peak count（与RMS约0.901）降为evidence-only；最终模型仅保留正8–30、负/正30–60、RMS offset、top4 concentration、sideband SNR、envelope RMS、modulation SNR共8项median。所有q90与其余23维帧量仍保留在evidence，不进入当前全局分类头。
- [x] 实现旧证据安全复用：新`stable8 prepare`要求旧`pruned21` manifest和blocked readiness同时存在且SHA-256、schema、唯一失败项、train-only/validation-report-only/test-sealed语义全部匹配；随后从23维frame evidence重新聚合8维，并要求与旧summary同名列逐值相等。focused只读回归覆盖train/val=`18266/4288`事件，raw/RD payload访问均为0。
- [x] `stable8` actual train-only复算通过：8/8项均满足全局AUC≥0.55和绝对effect≥0.10，8/8项在全部8个校准模式零反转且至少两个模式绝对effect≥0.10；其中7项在8个模式均有强同向效应，正8–30项有5个强同向模式、其余3个弱同向模式。Fusion维度`83/91`，新增8列零初始化、梯度最大值=`0.930194`、offline/online误差0。
- [x] 新建`src-v3/scripts/v18_broadband_micro_doppler_paired_pipeline.ps1`，支持`prepare/train/evaluate/all`，不使用环境变量，任一步失败停止，PowerShell AST解析通过。
- [x] 修复prepare恢复语义：中断后已有文件仅在全数组/全文逐值一致时复用，任何差异或损坏都fail-closed；任一正式V18 artifact非空后禁止重建数据/配置，避免训练产物与重建缓存混用。
- [x] 用户手动运行`stable8 prepare`；它从SHA绑定的旧frame evidence重聚合并生成独立summary/manifest/config/readiness，没有重复解析raw。train 8/8项非退化、全局有效且模式一致，validation仅5/8模式一致但仍保持8/8全局判别力，按冻结协议只作报告、不反调选择。
- [x] 用户手动完成Control/Treatment各20 epoch正式训练并执行paired validation。Treatment own-best parent Macro-F1=`0.953929`，相对Control提升`0.021549`；Drone/Bird弱项提升`0.020936`，但Bird Precision仅`0.913793`、Drone→Bird由3增至4、双向总错仍为7，且same-epoch 16 Macro-F1增益仅`0.018576`，最终状态为`completed_but_gate_failed`，禁止晋级。
- [x] 完成8个指定错误parent归因：4个Drone→Bird、3个Bird→Drone、1个Balloon→Bird全部stable8有效率100%。2083的8/8特征和14/23事件支持Bird，3960的23/25事件更接近Balloon且近邻混合，两者接受为当前stable8下的重叠/异常流型；其余6个在parent均值、逐事件中心和正确近邻上均明显支持真实类，判定为主干/Fusion/液态决策耦合错误，不得以“物理不可分”接受。
- [ ] 多seed、完整级联、OOF、test、正式拒识、stable、P95和部署：均未执行；只有seed42局部paired门槛通过后才讨论多seed。

实验假设：若V17失败主要由于主体回波/窄带截断/旧STFT时间分辨率和period缺失语义掩盖了旋翼宽带侧带，则V18 Treatment应同时提高Drone/Bird弱项floor与Macro-F1，并减少双向错误；若只移动Bird边界、只在own-best改善、或未达到2个百分点，则拒绝本8维`stable8`特征集。该实验只回答DBB specialist局部问题，不代表五类级联或开放集拒识收益。

版本策略：V18 `stable8` v3使用独立summary、manifest、data contract、配置、artifact、readiness、checkpoint和报告路径；V17、V18最初39维focused证据及`pruned21` blocked产物只读保留。完整Doppler-time张量模型仍是另一项模型合同，本轮仅8维模式稳定摘要进入Fusion。

当前执行边界：V18 stable8 prepare、seed42局部Control/Treatment正式训练、paired validation及8-parent归因已完成，V18冻结结论为“有真实信息增益但未通过采纳合同”。test、完整级联、OOF、正式拒识、stable、P95和部署必须继续记录为“未执行”。

## Stage 061 - V19 stable8后置残差决策耦合单因素修复

目标：在不新增微多普勒特征、不重新训练液态主干、不回看validation调阈值的前提下，验证V18失败是否主要来自stable8直接进入共享Fusion并联合更新全模型所产生的决策耦合。V19固定V18 Control best作为唯一基线，以train-only尺度和因果累计stable8驱动一个零初始化、有界、三类零和的后置logit残差，只训练24个权重。

前置依赖：Stage 060已冻结V18 stable8提取器、Control best checkpoint、20 epoch parent顺序和失败门槛；8-parent归因确认6条错误的物理特征时序支持真实类，允许将下一实验变量严格限定为决策耦合方式。V16 grouped split、V18缓存、标签、四维RD物理分支、loss与checkpoint排序保持不变。

状态：V19专用模块、prepare构建器、训练器、独立评估器、focused Smoke和PowerShell串行流水线均已完成；全量prepare、正式20 epoch训练和validation评估等待用户手动执行。test、完整级联、OOF、正式拒识、stable、P95和部署均未执行。

最小子任务：

- [x] 冻结单因素架构：V18 Control best主干和checkpoint文件只读；stable8不进入Fusion/CfC，只通过无bias的`8→3`线性层生成后置logit残差，共24个可训练参数。
- [x] 冻结残差约束：权重全零初始化，初始化combined logits与V18 Control逐值相同；逐事件先减三类均值，再按共同缩放限制每类绝对修正不超过3.0，保持三类零和且最大类间修正不超过6.0。
- [x] 冻结因果输入：尺度仅由953个V16 grouped train DBB parent的有效事件均值拟合median/IQR；每个prefix只聚合截至当前事件的有效stable8，缺失事件保持既有累计均值，首次有效证据前严格为0，不输入valid count或采样模式标识。
- [x] 实现`radar_ln3/micro_doppler_residual.py`，包含因果累计、24参数残差、版本化cache Dataset、parent-final+0.10 prefix辅助目标、独立评估和checkpoint hash/provenance合同。
- [x] 实现`build/micro_doppler_residual_v19.py`：只读加载V18 stable8 Treatment数据视图和V18 Control best，在`torch.inference_mode`下缓存train/val冻结baseline logits与因果stable8；主干state hash前后必须一致，validation只做缓存/基线重现完整性检查，不参与readiness选择，test payload访问固定为0。
- [x] 实现`train/micro_doppler_residual_v19.py`：沿用seed/DataLoader seed=42、batch=2、AdamW learning rate=`1e-4`、20 epoch、关闭early stopping、class weights、final主损失+0.10 prefix辅助和V18 checkpoint排序；每个epoch parent顺序SHA必须与V18 Control对应epoch完全一致，支持epoch边界resume。
- [x] 实现`report/micro_doppler_residual_v19.py`：独立复核20个epoch checkpoint和best checkpoint，固定比较V18 Control best，不重新训练Control；输出三类P/R/F1、Macro-F1、Drone/Bird弱项、双向错误、逐parent改变和8×3残差权重，不执行阈值搜索。
- [x] 新建`src-v3/scripts/v19_micro_doppler_residual_pipeline.ps1`，支持`prepare/train/evaluate/all`，不使用环境变量，任一步失败停止；已完成artifact时只读复核，失败/中断且存在last checkpoint时从最近完整epoch恢复。
- [x] focused Smoke通过：未来事件扰动不改变既有prefix，初始化logits逐值相等，参数数=24，残差绝对/类间上界=`3/6`，checkpoint/cache round-trip通过；真实三类parent残差梯度最大值=`0.167272`，V18 Control主干state SHA前后均为`5d61933484a0f01dac629a5b8168b92b6733be304897a3dfac72e76e54780b7e`，20/20 parent顺序hash与V18 Control一致。
- [x] synthetic端到端Smoke在临时目录完成20 epoch、20个epoch checkpoint、best复载和独立报告，训练/报告均完成且test保持“未执行”；该结果只验证流水线合同，不是性能证据。
- [ ] 用户手动运行V19 `prepare`，生成train-only normalization、train/val冻结logit+因果stable8缓存、contract、config和readiness；任一源hash、baseline指标、主干hash或初始logit检查失败即停止。
- [ ] 用户手动运行V19正式20 epoch训练和冻结validation评估；只有三类P/R/F1均≥0.95、Macro-F1与Drone/Bird弱项均严格提升`>0.02`、Drone→Bird/Bird→Drone均不增加且双向总错误至少下降1时才进入多seed讨论。
- [ ] 多seed、完整级联、OOF、test、正式拒识、stable、P95和部署：均未执行；seed42局部门槛通过前禁止启动。

验证命令：已在`src-v3`执行`F:\Project\RadarInsight\.venv\Scripts\python.exe -m smoke.micro_doppler_residual_v19_contract_test --project-root F:\Project\RadarInsight`、相关文件`compileall`、`smoke.static_isolation_check`、PowerShell AST解析和入口`--help`检查；均通过。全量prepare和正式训练未由代理执行。

实验假设：若V18 stable8本身包含有效信息而失败主要由共享Fusion/液态主干联合更新造成，则冻结Control并仅学习低容量后置残差应减少Drone↔Bird错误，同时避免531/731一类“物理证据支持Bird却被Treatment翻转”的错误；若24参数残差仍只移动单侧边界、未超过2个百分点或增加反向错误，则接受stable8在当前摘要空间的收益上限，不继续通过增大残差容量或validation阈值搜索追分。

版本策略：V19新增normalization、冻结logit/因果特征cache、contract、config、checkpoint和报告schema，但只读复用V16 split与V18 stable8证据/Control checkpoint；不覆盖V18文件，不修改通用`LiquidTaskClassifier`，避免将研究性late residual误接入完整级联。

当前执行边界：代码与focused Smoke已完成；全量数据处理、正式训练和validation性能评估由用户手动执行。test、完整级联、OOF、正式拒识、stable、P95和部署继续为“未执行”。
