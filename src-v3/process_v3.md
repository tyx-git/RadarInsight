# RadarInsight v3过程记录

## 2026-08-04 - Stage 008启动与只读核实

状态：实施已授权；只读核实完成；Stage 008正在进行。

实际命令与结果：

- `Get-ChildItem -Force`：确认工作目录为`F:\Project\RadarInsight`；存在`src`、`data`、`config`、`tests`、`OriData`和大量既有实验产物。
- `git status --short --branch`：确认工作树已有用户修改和未跟踪文件；本次不回滚、不覆盖、不清理。
- 检查`AGENTS.md`、`README.md`、`manage.py`、`src-v3`、`config/RADAR-LN3`：前三者未发现，后两者在本次建立前不存在。
- 搜索`data`和`config`中的`RADAR-LN3`文件：结果为`NO_RADAR_LN3_FILES`。
- 读取`data/processed/cq08_rd_physical_canonical_frame_index_v3_1.csv`：确认现有CQ08 canonical RD帧索引字段包含parent、label、track_point_index、rd_frame_time_seconds、tensor路径和物理RD shape；该文件作为只读输入，不复用其旧融合语义。
- 读取`data/processed/expert1_cq08_rd_physical_anchor_manifest_v6.summary.json`：确认1549个可用parent、29413个anchor segment、106026个RD帧、1505个完整覆盖parent、44个部分覆盖parent；旧状态全部为`track_point_anchor`，并明确禁止无独立时间证据进入融合。
- 读取`data/processed/expert1_realtime_causal_window_cq08_physical_anchor_map_v6.summary.json`：确认8240个窗口映射中`rd_coverage_ratio_mean=0.0`、融合可用窗口数为0；该结果证明旧window契约不能直接用于本次TR/RD点级实现。
- 读取`data/manifests/cq08_availability_summary.json`和`cq08_split_policy.json`：确认CQ08五分类及1549个可用track group；split单位为track_id，禁止同track跨split。
- 读取`src/core/rd_physical_schema.py`和`src/core/rd_tr_fusion.py`：仅作行为参考；本次src-v3不得导入这些模块。

已确认范围：CQ08主数据、TR点级因果事件、同`track_point_index`的RD帧聚合、真实TR dt、五分类、窗口级临时结果和目标级稳定结果、GPU在线部署目标、替换因果时序编码器。
本阶段未执行：代码重建、数据重建、配置生成、训练、预训练、正式评估、长时间运行。

## 2026-08-04 - Stage 009 V0点级契约生成与校验

状态：Stage 008的V0契约冻结完成；Stage 009数据契约生成完成；Stage 010尚未开始。

版本核对：生成前读取`data/`、`config/RADAR-LN3/`和本文件；生成前没有任何`RADAR-LN3`文件、没有历史v3版本，因此使用全局首个版本`RADAR-LN3-V0`。本批次文件内容分别为点级index、manifest、split、数据契约和统计摘要，同一V0批次关联记录在下方。

实际命令与结果：

- `python`依赖探测：系统解释器缺少`numpy`，未用于项目运行；`.\\venv\\Scripts\\python.exe -c "import numpy,pandas,torch"`成功，版本为NumPy 2.4.4、pandas 2.3.3、PyTorch 2.5.1+cu121。后续v3命令显式使用`.venv`解释器。
- 重排前首次运行`src-v3/build_cq08_point_contract.py`：因sample_id正则表达式错误失败，未写V0文件；已修复。该旧路径仅保留为历史记录。
- 第二次运行：因point集合变量名错误失败，未写V0文件；已修复。
- 第三次运行：因逐点重复打开RD NPZ，在120秒内超时，未写V0文件；已改为按tensor path缓存元数据。
- 第四次运行：发现真实输入异常`cq08__track__634.csv`时间回退；读取原始航迹/点迹确认第20点从`18:58:02`回退到`18:58:00`，并且RD point index出现`12→20→13`；未静默修复，契约保留原始dt和质量标记。
- 实际生成命令：`.\\venv\\Scripts\\python.exe src-v3\\build_cq08_point_contract.py --project-root F:\\Project\\RadarInsight`，成功，耗时约105秒。
- V0输出：`data/processed/RADAR-LN3-V0-cq08-tr-rd-point-index.csv`（29724行点级索引，SHA256=`bdbf0409e0c4dc14f1aedbfb5d32f7755e23fcd646f22f4ce6ff0eac5bbdd763`）；`RADAR-LN3-V0-cq08-tr-rd-point-manifest.json`；`RADAR-LN3-V0-cq08-tr-rd-split.json`；`RADAR-LN3-V0-cq08-tr-rd-data-contract.json`；`RADAR-LN3-V0-cq08-tr-rd-summary.json`。
- 统计：parent=1549，TR点=29724，kept RD帧=106026，RD present点=29413，RD missing点=311，完整覆盖parent=1505，部分覆盖parent=44，RD顺序异常parent=3，RD顺序逆序事件=4，TR时间异常parent=2（634、3338各1个负dt）。
- 纯度：index中`source_name`只有`CQ08`；parent split为train/val/test=1084/232/233；最大parent跨split数=1；未混入raw、Excel或RD-only。
- 标签：由当前标准化`cq08_track_index.csv`实际读取为Drone 699、Bird 350、Balloon 350、Clutter 50、Unknown 100；统一manifest摘要写为Drone 700。该一项差异已保留为Finding F005，未擅自补齐或改标签。

文件用途与版本原因：

- `RADAR-LN3-V0-cq08-tr-rd-point-index.csv`：一行一个TR事件，包含真实TR dt、TR缺失字段、RD帧tensor索引、RD局部时间和质量标记；实质变化为新增点级输入契约。
- `RADAR-LN3-V0-cq08-tr-rd-point-manifest.json`：记录CQ08数据范围、旧输入只读复用、数量、hash和版本理由；实质变化为新增manifest。
- `RADAR-LN3-V0-cq08-tr-rd-split.json`：从既有CQ08 track_id分组规则生成独立v3引用；实质变化为v3点级split引用，组规则未改变。
- `RADAR-LN3-V0-cq08-tr-rd-data-contract.json`：冻结模型输入输出、时间语义、missing/mask和泄漏策略；实质变化为新增契约。
- `RADAR-LN3-V0-cq08-tr-rd-summary.json`：记录实际校验摘要；实质变化为新增统计。

旧文件复用均为只读：`cq08_track_index.csv`、`cq08_rd_physical_canonical_frame_index_v3_1.csv`、`cq08_rd_physical_canonical_v3_1/*.npz`、`cq08_unified_manifest.json`、`cq08_grouped_split.json`；没有覆盖旧文件或复制旧张量。
本阶段未执行：src-v3模型实现、配置生成、训练、预训练、正式评估、独立测试集评估、生产部署和端到端P95验收。

## 2026-08-04 - Stage 010/011最小链路、测试与smoke

状态：Stage 010最小src-v3链路完成；Stage 011隔离、数据契约、shape/mask、forward、loss、梯度、checkpoint、DataLoader、在线单点和短时smoke均通过；正式训练和正式评估未执行。

实际新增代码与配置：

- `src-v3/radar_ln3/config.py`：V0配置类型校验和路径解析。
- `src-v3/radar_ln3/features.py`：TR物理缩放、角度sin/cos、signed-log幅度、missing mask。
- `src-v3/radar_ln3/data.py`：V0 parent序列Dataset、RD NPZ缓存读取、padding/mask、独立DataLoader/collate。
- `src-v3/radar_ln3/model.py`：RD 2D CNN空间前端、TR/RD点级融合、CfC风格闭式液态因果编码器、五分类头、单点online forward。
- `src-v3/radar_ln3/losses.py`：五分类交叉熵有限值校验。
- `src-v3/radar_ln3/checkpoint.py`：`RADAR-LN3-V0-checkpoint-v1`保存恢复契约。
- `src-v3/radar_ln3/stable.py`：窗口结果到目标稳定结果的因果EMA/确认状态机。
- `src-v3/radar_ln3/streaming.py`：每目标液态状态和稳定状态包装。
- 重排前的`src-v3/validate_cq08_point_contract.py`、`src-v3/validate_v1_anomaly_views.py`：正式数据契约和视图校验入口；当前入口已迁移至`src-v3/validate/`。
- `src-v3/smoke/static_isolation_check.py`、`src-v3/smoke/contract_tests.py`、`src-v3/smoke/missing_rd_mask_test.py`、`src-v3/smoke/run_smoke.py`、`src-v3/smoke/latency_smoke.py`、`src-v3/smoke/audit_forward_test.py`：Smoke、契约、缺失RD、审计和focused latency辅助入口；与正式`src-v3/radar_ln3/`链路分离。
- `config/RADAR-LN3/RADAR-LN3-V0-cq08-tr-rd-smoke-config.json`：唯一V0 smoke配置；实际由入口加载；正式配置未创建。

实际验证命令与结果：

- `.\\venv\\Scripts\\python.exe -m smoke.static_isolation_check --root F:\\Project\\RadarInsight\\src-v3`（工作目录`src-v3`）：通过，违规0；独立搜索未发现`from src`、`import src`、`sys.path`或动态加载绕过。
- `.\\venv\\Scripts\\python.exe -m py_compile <src-v3全部*.py>`：通过。
- `.\\venv\\Scripts\\python.exe src-v3\\validate_cq08_point_contract.py --project-root F:\\Project\\RadarInsight`：通过；parent=1549、point=29724、RD present点=29413、missing点=311、RD帧=106026、source=`CQ08`、split parent=1084/232/233；V0 index hash保持`bdbf0409e0c4dc14f1aedbfb5d32f7755e23fcd646f22f4ce6ff0eac5bbdd763`。
- `..\\.venv\\Scripts\\python.exe -m smoke.contract_tests --project-root F:\\Project\\RadarInsight --config F:\\Project\\RadarInsight\\config\\RADAR-LN3\\RADAR-LN3-V1-cq08-tr-rd-main-smoke-config.json`（工作目录`src-v3`）：通过；CUDA；batch shape为TR `[2,22,14]`、RD `[2,22,12,2,21,384]`、mask `[2,22,12]`、quality `[2,22,7]`；loss=1.423525094985962；gradient norm before clip=21.661792755126953；因果前缀不变性、negative dt clamp、DataLoader、online point logits `[1,5]`、stable state和checkpoint round-trip均通过；本次`forward_ms=292.9631000151858`仅为focused契约测试计时，不作为P95。
- `..\\.venv\\Scripts\\python.exe -m smoke.missing_rd_mask_test --project-root F:\\Project\\RadarInsight --config F:\\Project\\RadarInsight\\config\\RADAR-LN3\\RADAR-LN3-V1-cq08-tr-rd-main-smoke-config.json`：通过；真实parent=`cq08|track|1010`，2个missing-RD事件，缺失事件mask和=0，quality=`rd_present=0, rd_missing=1`，模型输出`[1,5]`有限。
- `..\\.venv\\Scripts\\python.exe -m smoke.run_smoke --project-root F:\\Project\\RadarInsight --config F:\\Project\\RadarInsight\\config\\RADAR-LN3\\RADAR-LN3-V1-cq08-tr-rd-main-smoke-config.json`：通过；CUDA；4个parent；2 steps；loss=1.451975703239441→1.5055841207504272；gradient norm=24.097423553466797→33.27187728881836；本次更新耗时约777.53ms→1546.94ms；checkpoint=`artifacts/RADAR-LN3-V1-cq08-tr-rd-main-smoke/checkpoint.pt`。
- `..\\.venv\\Scripts\\python.exe -m smoke.audit_forward_test --project-root F:\\Project\\RadarInsight --config F:\\Project\\RadarInsight\\config\\RADAR-LN3\\RADAR-LN3-V1-cq08-tr-rd-anomaly-audit-config.json`：通过；CUDA；3个异常parent均forward-only，参数更新许可为false。
- 全量Python编译：通过。Smoke脚本已迁移至`src-v3/smoke/`，正式链路仅保留`src-v3/radar_ln3/`、数据构建/视图校验入口和跟踪文件。

版本/配置记录：本次目录迁移不改变数据范围、字段、split、特征shape、模型输入输出或配置语义，因此复用`RADAR-LN3-V1`数据和配置版本，不创建V2。Smoke配置仍位于`config/RADAR-LN3/`；Smoke checkpoint是v3独立验证产物，不替换旧`src` artifact。
未执行：预训练、正式监督训练、正式trainer/evaluator实现、全量/多seed训练、val/test正式评估、parent/session分类指标、Drone/Bird/Clutter/Unknown正式指标、拒识指标、生产部署切换、真实端到端P95验收。

## 2026-08-04 - Stage 013放行判断

结论：不能进入正式训练或生产放行。

已具备证据：V0 CQ08纯度和split通过；src-v3无旧src运行时依赖；shape/mask/forward/loss/gradient/checkpoint/DataLoader/online point/smoke通过；model-only GPU latency曾有一次小样本观测，但最终重测超时，不能作为稳定性能结论。
阻塞项：V0仍保留3个RD顺序异常parent和2个TR负dt parent；旧摘要存在Drone=700与实际可读源清单Drone=699的历史差异，虽已定位为旧摘要不一致，仍需在正式实验报告中固定分母；没有正式训练权重、验证指标、parent/session指标、Clutter/Unknown拒识指标；端到端P95≤2秒没有实测。
正式训练、正式预训练、全量评估、稳定latency基准和生产切换均未执行。

## 2026-08-04 - Stage 014异常parent排除策略冻结

状态：用户确认方案1；Stage 014完成；Stage 015准备中。

用户选择：保持`RADAR-LN3-V0`不变；正式主实验排除异常parent；建立独立异常审计集；主集与审计集不混用。

实际核实命令与结果：

- 读取`data/processed/RADAR-LN3-V0-cq08-tr-rd-point-index.csv`并按`rd_parent_order_status=non_monotonic`或`tr_time_order_status=non_monotonic`聚合，异常parent精确为`cq08|track|634`、`cq08|track|785`、`cq08|track|3338`。
- 三个parent均属于原始`train`；634标签BirdTarget、21个TR点；785标签BirdTarget、25个TR点；3338标签DroneTarget、15个TR点。
- 634同时存在TR时间回退和RD point index异常；785存在RD point index异常；3338同时存在TR时间回退和RD point index异常。
- V0摘要和V0 index hash未修改；没有删除、重排或覆盖V0事件和RD张量。

版本计划：生成前已重新读取`data/`、`config/RADAR-LN3/`和本文件；当前最大版本为V0，V1是下一个全局版本。V1只因主/审计选择视图、split语义和配置输入发生实质变化而递增；不改变底层CQ08字段、RD张量、特征shape或V0 hash。
本阶段未执行：V1文件生成、V1配置生成、Dataset focused校验、正式训练、正式评估和生产P95验收。

## 2026-08-04 - Stage 015/016 V1视图生成与验证

状态：Stage 015和Stage 016完成；Stage 017完成放行判断；正式训练未执行。

版本核对：生成前重新读取`data/`、`config/RADAR-LN3/`和本文件；已有最大版本为`RADAR-LN3-V0`，因此本批次使用全局下一个版本`RADAR-LN3-V1`。V0文件未覆盖，V0 index SHA256仍为`bdbf0409e0c4dc14f1aedbfb5d32f7755e23fcd646f22f4ce6ff0eac5bbdd763`。

实际生成命令：`.\\venv\\Scripts\\python.exe src-v3\\build_v1_anomaly_views.py --project-root F:\\Project\\RadarInsight`，成功。

V1实际结果：

- main index：1546 parent、29663 TR点、105843 RD帧；split parent为train/val/test=1081/232/233；label parent为Balloon 350、Bird 348、Clutter 50、Drone 698、Unknown 100；index SHA256=`1e0e050cc9704d53f7f4e6228ed9dc8925809bf7ae4cdfd0d28cdcb186f6011d`；split SHA256=`8f0eeb9861caa0493f11d0046a3eb52144278897571ec74f93edefefb823f9f8`。
- anomaly-audit index：3 parent、61 TR点、183 RD帧；parent为`cq08|track|3338`、`cq08|track|634`、`cq08|track|785`；原始split均为train；index SHA256=`fa9bf8888a27e54c7a9d0b8040d82da2d9690588eb89e6c6605531a5c800d5b1`；split SHA256=`6ff91d3d1e7b6e54d119f626dd0c1edfd0e55ed2c46e0fa1a51e6ec23c7de1fc`。
- V1主/审计`sample_id`是V0的完整、互斥分区；主视图不含3个异常parent，审计视图只含3个异常parent；底层canonical RD tensor全部只读复用。

实际校验命令与结果：

- `.\\venv\\Scripts\\python.exe src-v3\\validate_v1_anomaly_views.py --project-root F:\\Project\\RadarInsight`：通过，V0 partition=true，主/审计parent集合互斥且闭合。
- `.\\venv\\Scripts\\python.exe src-v3\\validate_v1_anomaly_views.py --project-root F:\\Project\\RadarInsight --run-dataset-checks`：通过；V1主配置实际加载train=1081 parent，审计配置实际加载audit=3 parent；TR shape 14维、missing 11维、RD shape `[K,2,21,384]`、quality 7维，finite通过。
- 全量`src-v3` Python编译：通过。
- `static_isolation_check.py`：通过，违规0。

配置与审计保护：

- `config/RADAR-LN3/RADAR-LN3-V1-cq08-tr-rd-main-smoke-config.json`：V1 main，允许有限步参数更新，checkpoint=`artifacts/RADAR-LN3-V1-cq08-tr-rd-main-smoke/checkpoint.pt`。
- `config/RADAR-LN3/RADAR-LN3-V1-cq08-tr-rd-anomaly-audit-config.json`：V1 anomaly-audit，`allow_parameter_updates=false`，只允许forward，不允许训练更新。
- `src-v3/smoke/run_smoke.py`遇到审计配置会直接拒绝参数更新；`src-v3/smoke/audit_forward_test.py`验证审计集只读推理。

实际短时验证：

- V1 main smoke：`..\\.venv\\Scripts\\python.exe -m smoke.run_smoke --project-root F:\\Project\\RadarInsight --config F:\\Project\\RadarInsight\\config\\RADAR-LN3\\RADAR-LN3-V1-cq08-tr-rd-main-smoke-config.json`（工作目录`src-v3`），CUDA，4 parent，2 steps，loss=`1.451975703239441→1.5055841207504272`，gradient norm=`24.097423553466797→33.27187728881836`，window logits shape `[2,5]`，成功保存V1 smoke checkpoint。
- V1 anomaly audit：`..\\.venv\\Scripts\\python.exe -m smoke.audit_forward_test --project-root F:\\Project\\RadarInsight --config F:\\Project\\RadarInsight\\config\\RADAR-LN3\\RADAR-LN3-V1-cq08-tr-rd-anomaly-audit-config.json`（工作目录`src-v3`），CUDA，3个异常parent全部forward-only通过，每个window logits shape `[1,5]`，参数更新许可为false。
- 以上均为契约/数值smoke，不是分类性能结果；正式训练、预训练、全量评估、parent/session指标、拒识指标和端到端P95均未执行/未验收。

## 2026-08-04 - Stage 017 V1放行判断

结论：V1视图可以进入后续正式实验准备；当前不能自动进入正式训练。

放行证据：V0保持不变；V1主/审计视图闭合且互斥；V1配置实际加载；主集Dataset/DataLoader和审计forward-only通过；审计配置有参数更新硬保护；src-v3隔离和编译通过。
剩余阻塞：用户尚未授权正式训练的轮数、seed、资源和评估范围；尚无V1正式checkpoint、验证指标、混淆矩阵、parent/session指标、Clutter/Unknown拒识指标；端到端P95≤2秒仍未验收。
未执行：正式监督训练、预训练、全量/多seed评估、独立测试评估、生产切换和正式端到端延迟基准。

## 2026-08-04 - Stage 017最终focused核对

实际命令与结果：

- `..\\.venv\\Scripts\\python.exe -m smoke.contract_tests --project-root F:\\Project\\RadarInsight --config F:\\Project\\RadarInsight\\config\\RADAR-LN3\\RADAR-LN3-V1-cq08-tr-rd-main-smoke-config.json`（工作目录`src-v3`）：通过；CUDA；shape、DataLoader、因果前缀不变性、loss=1.423525094985962、梯度、negative dt clamp、online point和checkpoint round-trip均通过；本次`forward_ms=292.9631000151858`仅为focused契约测试计时，不作为延迟指标。
- 审计训练保护由`smoke.audit_forward_test`验证：通过，3个异常parent均forward-only，参数更新许可为false。
- `.\\venv\\Scripts\\python.exe -m py_compile <src-v3全部*.py>`：通过。
- `..\\.venv\\Scripts\\python.exe -m smoke.static_isolation_check --root F:\\Project\\RadarInsight\\src-v3`（工作目录`src-v3`）：通过，违规0。

最终放行：V1主视图可以进入“待训练准备”；当前不能进入“正式训练已就绪”，因为正式trainer/evaluator、正式配置和训练选择规则尚未实现，且本代理不自动启动长时间训练。V1主smoke和V1契约测试均不是分类性能结果；V1验证指标、混淆矩阵、parent/session指标、拒识指标和端到端P95均未执行/未验收。

## 2026-08-04 - Smoke目录隔离与训练前置判定

状态：Smoke内容已迁移至`src-v3/smoke/`；正式链路`src-v3/radar_ln3/`未修改；V1数据和配置版本复用；当前进入“待训练准备”，未进入“正式训练可启动”。

实际变更：

- 使用`apply_patch`将`audit_forward_test.py`、`contract_tests.py`、`latency_smoke.py`、`missing_rd_mask_test.py`、`run_smoke.py`、`static_isolation_check.py`迁移到`src-v3/smoke/`，新增`src-v3/smoke/__init__.py`；只修正新路径下的默认项目根目录层级。
- 未修改`src-v3/radar_ln3/`正式模型、Dataset、特征、loss、checkpoint实现；未修改`src/`；未创建数据或正式配置新版本。
- 新目录下执行全量Python编译、`smoke.static_isolation_check`、V1主`smoke.run_smoke`、V1主`smoke.contract_tests`、`smoke.missing_rd_mask_test`和异常`smoke.audit_forward_test`，均通过。

训练前置核验：

- 当前`src-v3`没有正式trainer、evaluator或预训练加载入口；只有Smoke有限步训练入口。
- `RadarLiquidClassifier`构造后可随机初始化；当前TrainConfig没有预训练checkpoint字段；Smoke直接构造随机模型、AdamW并更新参数。
- 因此当前监督训练不需要预训练前置；预训练不是已执行结果，也不是当前启动阻塞项。
- 旧`src`中的TR-only/RD-only预训练checkpoint不能直接复用，因为输入维度、RD/TR融合结构、时间契约、checkpoint schema和数据版本不一致。
- V1 Smoke checkpoint仅为2-step随机初始化验证产物，不是预训练权重，不作为正式初始化或性能证据。

当前放行边界：

- 可以进入：Stage 018正式训练前置实现、配置语义冻结、训练/验证入口设计和短时契约验证。
- 尚不能进入：正式全量监督训练、预训练、正式验证/测试、生产切换或P95放行。
- 进入正式训练还需要：实现正式trainer/evaluator；去除正式配置中的`sample_limit`/`smoke_steps`语义；冻结训练统计、类别权重、seed、epoch/early stopping、checkpoint选择指标和评估范围；创建/复用经过验证的正式配置；再由用户明确授权正式训练命令。
- 版本记录：本次仅迁移Smoke代码，数据范围、字段、split、模型契约和配置语义均未改变，复用V1，不创建V2。

## 2026-08-04 - Stage 018正式训练前置启动

状态：Stage 018进行中；已完成只读准备，尚未创建正式训练代码、正式配置或正式训练产物；本轮不启动全量/长时间训练。

本阶段开始前的只读核验：

- 重新读取`src-v3/task_plan_v3.md`、`src-v3/process_v3.md`、`src-v3/findings_v3.md`、`data/`和`config/RADAR-LN3/`。
- 当前全局最大RADAR-LN3版本为V1；V1主数据、V1配置和V1审计视图继续只读复用。
- V1主索引实际为1546个parent、29663个TR点；train/val/test parent为1081/232/233；序列长度最小12、最大29、中位18；五类均出现在val。
- V1主数据`source_name`实际只有`CQ08`；主索引已经排除634、785、3338，正式训练不再依赖旧V0异常过滤逻辑。
- 当前`src-v3`只有Smoke有限步更新入口，没有正式trainer/evaluator、正式配置字段或预训练加载入口；当前正式监督训练仍不要求预训练。

Stage 018设计边界：

- 正式配置将因新增训练/评估语义使用全局下一版本`RADAR-LN3-V2`；V1数据契约和索引不变，不重建数据。
- 正式训练使用完整V1 main train parent，验证使用V1 main val parent；test评估默认关闭，审计集只做forward/evaluation-only。
- 训练期采用配置声明的确定性parent级类别权重和固定物理缩放；权重、seed、epoch、early stopping和checkpoint选择规则写入正式配置/产物。
- 新增正式trainer/evaluator后，先执行配置、导入、数据纯度、shape、forward、loss、梯度和checkpoint契约检查；长时间训练仍需用户确认。

## 2026-08-04 - Stage 018技术实现与隔离契约首次复测核对（历史状态）

状态：这是隔离artifact路径契约的首次复测记录；当时正式trainer/evaluator、V2正式配置和训练期统计已经存在，但该次复测超时。后续GPU空闲后已重新执行并通过，最新状态见本文件末尾“Stage 018隔离正式入口契约通过”。

实际读取和校验结果：

- `src-v3/train/formal.py`、`src-v3/report/classification.py`和`src-v3/radar_ln3/{contracts,factory,metrics,evaluator}.py`实际存在；正式入口不导入`src`。
- V2正式配置：`config/RADAR-LN3/RADAR-LN3-V2-cq08-tr-rd-formal-supervised-config.json`，SHA256=`6e2eef5c052bbf5d62ec139f850b1d2fcb91668624a37f18405511cdd676ae61`。
- V2正式数据契约：`data/processed/RADAR-LN3-V2-cq08-tr-rd-formal-data-contract.json`，SHA256=`fe559b4c7cac7c161d001b804af42e27fb2a40d015645343841a40f93d634415`。
- V2 train-only归一化统计：`data/processed/RADAR-LN3-V2-cq08-tr-rd-train-normalization-stats.json`，SHA256=`584eddcaeb8d78677f0f4902d622c2aa313d9374b767a47982c12dc058f826b6`；统计parent=1081，amplitude center=55.875，scale=6.911864353489518。
- 配置引用的V1 main index SHA256=`1e0e050cc9704d53f7f4e6228ed9dc8925809bf7ae4cdfd0d28cdcb186f6011d`，未修改、未重建。
- 已有`artifacts/RADAR-LN3-V2-cq08-tr-rd-formal/training-summary.json`实际记录1个训练step、CUDA、train parent=1081、validation按参数跳过、test未执行；该目录中的checkpoint只能作为有界契约运行遗留产物，不能称为正式训练模型，也不能替代隔离artifact契约。

实际执行命令与结果：

- 在`src-v3`工作目录运行`..\\.venv\\Scripts\\python.exe -m smoke.formal_entry_contract --project-root F:\\Project\\RadarInsight --config F:\\Project\\RadarInsight\\config\\RADAR-LN3\\RADAR-LN3-V2-cq08-tr-rd-formal-supervised-config.json`；在`formal_entry_contract.py`将验证序列限制为1个parent的1个causal point后再次执行，命令仍以退出码124结束，运行约240.6秒超时；`artifacts/RADAR-LN3-V2-cq08-tr-rd-formal-entry-contract/`未生成。
- 同期`nvidia-smi`观察到GPU利用率55%、显存7557/8188 MiB，存在PID 44884的Python进程占用；未终止或干预该进程，资源争用是超时的工程判断因素，不是分类失败证据。
- 超时后重新检查进程和artifact：契约进程已退出；隔离artifact目录不存在；既有正式目录未被覆盖。

本阶段实际代码变更：

- 修改`src-v3/smoke/formal_entry_contract.py`，将验证Dataset的`max_sequence_length`限制为1，仅缩小Smoke范围，不改变正式Dataset、模型、配置、数据或checkpoint语义。
- 未修改`src/`、V1 index、V2 data contract、V2 normalization stats或V2正式配置；未删除既有artifact。

该次复测时的放行状态：

- 已具备：正式trainer/evaluator代码、V2配置、V1 main纯度/hash、train-only统计、随机初始化监督训练路径和预训练非必需结论。
- 未具备：隔离artifact契约复测通过、正式checkpoint、val真实指标、Drone/Bird混淆、Clutter/Unknown指标、session指标、test指标和端到端P95≤2秒证据。
- 当时结论：可以进入“待训练”，不能进入“正式训练已放行”。正式监督训练不需要预训练；预训练未执行、也不是当时阻塞项。该结论已由后续通过结果覆盖。

未执行：20 epoch正式训练、完整val、test、拒识评估、parent/session实测、端到端GPU P95和生产切换。

跟踪文件更新后的focused复核：

- `..\\.venv\\Scripts\\python.exe -m py_compile <rg --files src-v3 -g '*.py'>`：退出码0。
- `..\\.venv\\Scripts\\python.exe -m smoke.static_isolation_check --root F:\\Project\\RadarInsight\\src-v3`：通过，`violations=[]`。
- `rg -n "from src|import src|sys\\.path|SourceFileLoader|spec_from_file_location|import_module" src-v3 -g '*.py'`：仅命中隔离检查脚本自身用于检测的正则字符串；未发现src运行时依赖。

## 2026-08-04 - src-v3目录与入口命名规范化

状态：目录重排和入口命名完成；数据、配置、模型语义、版本号和artifact语义均未改变；未启动训练。

实际变更：

- 新增包目录：`src-v3/build/`、`src-v3/train/`、`src-v3/report/`、`src-v3/validate/`，各目录新增`__init__.py`。
- 数据构建/处理入口：`src-v3/build/cq08_point_contract.py`、`src-v3/build/radar_ln3_normalization.py`、`src-v3/build/v1_anomaly_views.py`。
- 训练入口：`src-v3/train/formal.py`。
- 报告/评估入口：`src-v3/report/classification.py`。
- 数据契约/视图校验入口：`src-v3/validate/cq08_point_contract.py`、`src-v3/validate/v1_anomaly_views.py`。
- Smoke入口仍全部位于`src-v3/smoke/`；`formal_entry_contract.py`改为从`train.formal`导入，未把Smoke代码放入正式链路。
- 所有移动后的CLI默认项目根目录从`parents[1]`修正为`parents[2]`；模块化入口统一使用`python -m build.*`、`python -m train.*`、`python -m report.*`和`python -m validate.*`。

历史记录说明：本文件前面的旧路径和旧命令是目录重排前实际执行过的命令，保留用于审计；当前有效路径以上述新路径为准。旧入口文件已不存在，不创建兼容副本，避免双入口和命名歧义。

实际验证：

- 全量`src-v3` Python编译：通过。
- `python -m smoke.static_isolation_check --root F:\\Project\\RadarInsight\\src-v3`：通过，`violations=[]`。
- `python -m build.cq08_point_contract --help`、`build.radar_ln3_normalization --help`、`build.v1_anomaly_views --help`：均通过。
- `python -m validate.cq08_point_contract --help`、`validate.v1_anomaly_views --help`：均通过。
- `python -m train.formal --help`、`python -m report.classification --help`：均通过。
- `python -m smoke.formal_entry_contract --help`：通过，确认Smoke到`train.formal`的新导入路径可解析。
- 旧入口路径存在数=0，新入口路径存在数=7；未修改`data/`、`config/RADAR-LN3/`和`src/`。

## 2026-08-04 - Stage 018隔离正式入口契约通过

状态：Stage 018完成；隔离artifact契约通过；正式训练、完整验证、test和端到端P95仍未执行。

实际命令：在`F:\\Project\\RadarInsight\\src-v3`工作目录运行`..\\.venv\\Scripts\\python.exe -m smoke.formal_entry_contract --project-root F:\\Project\\RadarInsight --config F:\\Project\\RadarInsight\\config\\RADAR-LN3\\RADAR-LN3-V2-cq08-tr-rd-formal-supervised-config.json`。

实际结果：

- 输出`status=passed`，`artifact_scope=bounded formal entry contract only; not a formal training artifact`，设备为`cuda`。
- 数据契约路径和hash均匹配：V1 main index=`1e0e050cc9704d53f7f4e6228ed9dc8925809bf7ae4cdfd0d28cdcb186f6011d`；V2 data contract=`fe559b4c7cac7c161d001b804af42e27fb2a40d015645343841a40f93d634415`；V2 train-only stats=`584eddcaeb8d78677f0f4902d622c2aa313d9374b767a47982c12dc058f826b6`。
- train-only统计parent=1081，RD amplitude center=`55.875`，scale=`6.911864353489518`。
- 仅执行1个训练step；正式checkpoint schema=`RADAR-LN3-V2-formal-checkpoint-v1`；checkpoint实际保存到`artifacts/RADAR-LN3-V2-cq08-tr-rd-formal-entry-contract/best_model.pt`。
- 训练batch parent=2；TR shape=`[2,22,14]`，TR missing=`[2,22,11]`，dt=`[2,22]`；RD shape=`[2,22,12,2,21,384]`，RD mask=`[2,22,12]`，RD quality=`[2,22,7]`，padding=`[2,22]`，labels=`[2]`，lengths=`[2]`。
- window logits shape=`[2,5]`；weighted loss=`1.3724843263626099`；focused validation parent=1；`validation_metrics_are_focused_only=true`。
- 本地entry-contract目录实际包含`best_model.pt`、`last_model.pt`和`training-summary.json`；未覆盖`artifacts/RADAR-LN3-V2-cq08-tr-rd-formal/`。

未执行：20 epoch正式监督训练、完整val、test、拒识评估、parent/session完整指标、Drone/Bird正式混淆统计、Clutter/Unknown正式指标、端到端P95和生产切换。

放行结论：可以进入“正式训练授权等待”；当前不需要预训练，正式训练仍需用户明确授权，Smoke结果不能作为分类性能结果。

## 2026-08-04 - Stage 021训练日志模块实现

状态：日志模块和独立日志Smoke完成；真实trainer集成Smoke未执行，原因是当前GPU仍被占用，避免与外部计算任务争用。

实际代码变更：

- 新增`src-v3/radar_ln3/logging_utils.py`，仅使用Python标准库`logging`，提供`configure_training_logger`、`close_training_logger`、`format_duration`、`format_metric`和`progress_interval`。
- 修改`src-v3/train/formal.py`：训练入口按artifact目录生成`training.log`；输出run启动、数据契约、schedule、模型契约、周期进度、epoch摘要、checkpoint、停止原因、summary和异常日志。
- 进度默认按每epoch约10次、最多每50 step打印一次；字段包括`epoch`、`step`、`global`、`loss`、`grad`、`lr`、`elapsed`和`eta`，减少逐batch I/O。
- 控制台Handler使用stderr，训练summary JSON继续写stdout；FileHandler使用UTF-8并避免重复handler；Smoke artifact与正式artifact均沿各自配置目录写日志。
- 新增`src-v3/smoke/logging_test.py`，验证日志文件、固定格式、duration格式和重复配置不产生重复handler。

实际验证命令与结果：

- 全量`src-v3` Python编译：通过。
- `python -m smoke.logging_test`：通过，输出`status=passed`、`format_duration_65=01m05s`、`progress_interval_541=50`、`handler_duplication=not_detected`。
- `python -m smoke.static_isolation_check --root F:\\Project\\RadarInsight\\src-v3`：通过，`violations=[]`。
- `python -m train.formal --help`、`python -m report.classification --help`：均通过。
- 本次代码修改未改变V1 index、V2 data contract、V2 normalization stats、配置内容、模型shape或checkpoint schema。

历史状态：当时未执行日志修改后的真实trainer 1-step Smoke；但此前正式训练已于2026-08-05 00:05:48实际完成，旧日志和旧summary已存在。新日志/明细输出契约在本次代码修改后尚未重新训练。

## 2026-08-05 - 正式训练结果与输出契约修正

实际读取正式artifact：

- `artifacts/RADAR-LN3-V2-cq08-tr-rd-formal/training-summary.json`记录`status=passed`、请求20 epochs、实际history=15 epochs、`global_step=8115`、`best_epoch=10`、`best_selection_metric_value=0.703274980916419`、`stop_reason=early_stopping`。
- epoch 10验证结果：parent accuracy=`0.844827586206897`、macro precision=`0.690616161616162`、macro recall=`0.772710622710623`、balanced accuracy=`0.772710622710623`、macro F1=`0.703274980916419`；stable macro F1=`0.663824687356736`；causal window macro F1=`0.605650266071029`。
- epoch 10类别结果：Drone precision/recall/F1=`0.9700/0.9238/0.9463`，support=105；Bird=`0.8444/0.7308/0.7835`，support=52；Balloon=`0.8750/0.9423/0.9074`，support=52；Clutter=`0.4000/1.0000/0.5714`，support=8；Unknown=`0.3636/0.2667/0.3077`，support=15。
- test为`not_executed`，生产P95为`not_evaluated`；以上仅为V1 main val的训练选择结果，不是test或线上验收结果。

本次代码变更：

- `src-v3/train/formal.py`将summary字段从`best_checkpoint_path`改为`best_checkpoint`。
- validation完整records不再嵌入`training-summary.json`或`evaluation-val.json`；完整样本明细写入artifact目录的`sample_detail_information.json`，包含window、parent、stable全部样本。
- 训练日志末尾新增总体指标、五类precision/recall/F1/support、主要混淆去向、Drone/Bird双向混淆和所有`confidence < 0.60`样本；高置信度样本不逐条写入日志，但仍保存在明细JSON。
- `src-v3/radar_ln3/evaluator.py`新增`write_sample_detail_information`和摘要去records逻辑；`src-v3/report/classification.py`的stdout也只输出摘要。
- 新增`src-v3/smoke/report_detail_test.py`验证全部records保存和summary移除records。

实际验证：

- 全量`src-v3` Python编译：通过。
- `python -m smoke.logging_test`：通过。
- `python -m smoke.report_detail_test`：通过。
- `python -m smoke.static_isolation_check --root F:\\Project\\RadarInsight\\src-v3`：通过，`violations=[]`。
- `src-v3`中`best_checkpoint_path`引用数为0；新训练入口使用`best_checkpoint`。

历史artifact边界：当前正式目录中的旧`training-summary.json`仍约31 MB，包含本次修改前训练生成的epoch详细records；未静默覆盖。新字段和`sample_detail_information.json`将在下一次正式训练或重新运行best checkpoint val报告时生成。旧`best_model.pt` SHA256=`CF22838CD723D6C77BDC9681C49D6CEDF9CCA53308B90D99DF704B4F3FF0F458`，未修改。

未执行：因日志/报告格式变更而重新训练、test评估、端到端P95和生产切换。

## 2026-08-05 - 既有正式结果输出契约迁移完成

实际操作：

- 先将旧`training-summary.json`完整复制为`training-summary-legacy.json`，保留旧训练记录；未删除旧文件，未覆盖best/last checkpoint。
- 新增`src-v3/report/migrate_training_summary.py`，从legacy summary的best epoch=10 validation records生成独立明细，并把summary迁移为新字段。
- 执行：`..\\.venv\\Scripts\\python.exe -m report.migrate_training_summary --summary F:\\Project\\RadarInsight\\artifacts\\RADAR-LN3-V2-cq08-tr-rd-formal\\training-summary-legacy.json --output F:\\Project\\RadarInsight\\artifacts\\RADAR-LN3-V2-cq08-tr-rd-formal\\training-summary.json`，退出码0。

迁移后实际结果：

- `training-summary.json`约495230字节，不含`best_checkpoint_path`，包含`best_checkpoint`=`F:\\Project\\RadarInsight\\artifacts\\RADAR-LN3-V2-cq08-tr-rd-formal\\best_model.pt`；summary和history中均不再嵌入`records`。
- `sample_detail_information.json`实际生成，schema=`RADAR-LN3-V2-sample-detail-v1`，split=`val`，threshold=`0.60`，保存window=4474、parent=232、stable=232条完整明细；选择epoch=10，checkpoint指向best_model.pt。
- 统计到低置信度记录共709条：window=641、parent=34、stable=34；高于或等于0.60的样本仍完整保存在明细JSON，但不写入日志逐条输出。
- 迁移工具重新写入`training.log`末尾：总体指标、五类指标、Drone/Bird双向混淆、低置信度样本和`final_metrics | best_checkpoint=...`均已输出。
- best checkpoint SHA256仍为`CF22838CD723D6C77BDC9681C49D6CEDF9CCA53308B90D99DF704B4F3FF0F458`；未重训、未改权重。

注意：本次迁移写入的是既有best epoch验证记录，不是重新运行模型forward；后续GPU独立评估已重新加载best checkpoint并完成真实forward，结果见下一条过程记录。

## 2026-08-05 - best checkpoint GPU validation独立复核与瓶颈统计

只读前置检查：

- 未结束、未修改其他GPU进程；评估使用可借用GPU资源，`batch-size=1`，只读取V1 main `val`，没有读取test。
- checkpoint：`F:\Project\RadarInsight\artifacts\RADAR-LN3-V2-cq08-tr-rd-formal\best_model.pt`，实际加载`epoch=10`、`global_step=5410`，参数量为27629。
- checkpoint SHA256=`CF22838CD723D6C77BDC9681C49D6CEDF9CCA53308B90D99DF704B4F3FF0F458`。
- V1 index SHA256=`1e0e050cc9704d53f7f4e6228ed9dc8925809bf7ae4cdfd0d28cdcb186f6011d`；V2 data contract SHA256=`fe559b4c7cac7c161d001b804af42e27fb2a40d015645343841a40f93d634415`；train-only stats SHA256=`584eddcaeb8d78677f0f4902d622c2aa313d9374b767a47982c12dc058f826b6`。

实际命令：在`F:\Project\RadarInsight\src-v3`运行：

`& ..\\.venv\\Scripts\\python.exe -m report.classification --project-root F:\\Project\\RadarInsight --config F:\\Project\\RadarInsight\\config\\RADAR-LN3\\RADAR-LN3-V2-cq08-tr-rd-formal-supervised-config.json --checkpoint F:\\Project\\RadarInsight\\artifacts\\RADAR-LN3-V2-cq08-tr-rd-formal\\best_model.pt --split val --batch-size 1`

实际结果：

- 命令退出码0，`device=cuda`，验证parent=232，window=4474，stable记录=232，stable confirmed=231；`dt_clamped_event_count=0`，`dt_zero_event_count=1796`。
- window：Accuracy=`0.8218596334`，Macro Precision=`0.6045705162`，Macro Recall/Balanced Accuracy=`0.6131413968`，Macro F1=`0.6056502661`。
- parent：Accuracy=`0.8448275862`，Macro Precision=`0.6906161616`，Macro Recall/Balanced Accuracy=`0.7727106227`，Macro F1=`0.7032749809`。
- stable：Accuracy=`0.8362068966`，Macro Precision=`0.6547742127`，Macro Recall/Balanced Accuracy=`0.7093406593`，Macro F1=`0.6638246874`。
- stable confirmed only：support=231，Macro F1=`0.6637398517`。
- GPU结果与此前CPU batch=1独立复核的指标逐项一致，仅置信度存在浮点末位差异；GPU结果作为本次正式评估来源。
- 报告实际写入`artifacts/RADAR-LN3-V2-cq08-tr-rd-formal/evaluation-val.json`；全量window/parent/stable明细实际写入`sample_detail_information.json`；未创建新数据、索引、manifest、split或配置版本。

错误与数据质量只读统计：

- parent混淆矩阵（行=true，列=pred）为：`[[97,3,2,1,2],[2,38,3,4,5],[1,1,49,1,0],[0,0,0,8,0],[0,3,2,6,4]]`。
- parent级Drone→Bird=3、Bird→Drone=2；stable级分别为2和2；window级分别为54和74。在线前缀阶段的Bird→Drone显著多于最终parent，说明短前缀/状态累积是实际瓶颈之一。
- parent错误数为Drone=8、Bird=14、Balloon=3、Clutter=0、Unknown=11；stable错误数为Drone=6、Bird=15、Balloon=3、Clutter=2、Unknown=12。
- window错误主要来自Bird=267、Unknown=247、Clutter=95；Unknown真值被预测为Bird=147，Clutter真值被预测为Bird=84。
- val parent支持数为Drone=105、Bird=52、Balloon=52、Clutter=8、Unknown=15；Clutter/Unknown的parent级结论受小样本影响，不能声称已达到稳定泛化。
- `confidence<0.60`：window=641/4474，parent=34/232，stable=34/232。parent阈值诊断下accepted=198、accepted errors=18、Clutter/Unknown拒识候选=12/23、已知三类拒识=22/209；这些数字只是候选拒识分析，当前没有正式拒识动作、阈值校准或Selective Risk验收。
- V1 val点级索引中三类TR字段`height_m/latitude/longitude`全部4474行缺失，且每行`tr_missing_count=3`；编码器用零值加missing mask保留语义，未静默伪造有效值。
- V1 val点级索引中`dt=0`共1796/4474，其中首点232，非首点1564；非首点占34.96%。按类别zero-dt比例为Balloon=47%、Bird=50%、Clutter=5%、Drone=39%、Unknown=5%。当前液态模块对zero-dt使用0.001秒floor，因此不能把重复墙钟时间等同于真实连续时间。
- val RD缺失点共40，其中Bird=22、Drone=17、Balloon=1；Clutter和Unknown没有RD缺失点。RD缺失不是Unknown/Clutter低召回的唯一解释，但Bird/Drone的多模态质量不均衡需要单独分层实验。
- 当前配置只对每个parent的`window_logits`最终窗口计算一次五分类loss；评估同时统计所有causal prefix windows。因此window指标不是当前训练目标的直接优化目标，这是已确认的训练-评估契约不一致。

未执行：test评估、session评估（当前V1契约没有session/capture字段）、端到端在线P95、拒识阈值校准、多seed/交叉验证、任何优化实验、重新训练、生产切换。

## 2026-08-05 - Stage 023启动：窗口监督与有效dt手术式修复

状态：只读核查完成，开始V3最小修复；V2索引、契约、配置、checkpoint和artifact保持只读；正式训练、test、全量阈值搜索和生产切换未执行。

已核实证据：

- V2训练入口只对`output["window_logits"]`计算五分类CE；V2评估逐事件统计`output["sequence_logits"]`，窗口级训练与验收目标不一致。
- V1主索引共29663点、1546个parent，非首点零dt为9776/28117=34.77%；CQ08轨迹CSV的`time_raw`为秒级`HH:MM:SS`，track 1010存在同一秒内多个点；RD局部时间是CPI累计时间，不是TR wall-clock，禁止替代TR dt。
- V2验证parent支持为Clutter=8、Unknown=15；现有EMA稳定器没有reject/abstain动作，0.60仅为低置信度诊断阈值。

版本决策：当前RADAR-LN3最大版本为V2。V3将新建索引、manifest、split、数据契约和配置；V2 train-only normalization stats内容和语义不变，读取hash后复用，不创建无意义的新统计版本。V3独立artifact路径为`artifacts/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-formal`，不覆盖V2正式artifact。

本阶段实际命令与结果：

- `..\\.venv\\Scripts\\python.exe -m build.causal_contract --project-root F:\\Project\\RadarInsight`：退出码0，生成V3 index/manifest/split/contract/summary。
- `..\\.venv\\Scripts\\python.exe -m validate.causal_contract --project-root F:\\Project\\RadarInsight`：退出码0；CQ08纯度、parent/split/label一致性和源字段对照通过。
- `..\\.venv\\Scripts\\python.exe -m build.causal_config --project-root F:\\Project\\RadarInsight`：退出码0；V3 config加载通过。
- `..\\.venv\\Scripts\\python.exe -m compileall -q .`和`python -m smoke.static_isolation_check --root F:\\Project\\RadarInsight\\src-v3`：通过，`violations=[]`。
- `python -m smoke.contract_tests --project-root F:\\Project\\RadarInsight --config F:\\Project\\RadarInsight\\config\\RADAR-LN3\\RADAR-LN3-V3-cq08-tr-rd-prefix-dt-formal-supervised-config.json`：CUDA退出码0；shape、forward、prefix loss、梯度、因果性、在线单点和checkpoint round-trip通过。
- `python -m smoke.missing_rd_mask_test ...V3...`：CUDA退出码0；真实CQ08缺失RD事件mask通过。
- `python -m smoke.latency_smoke ...V3... --iterations 8`：CUDA退出码0；模型-only P95=`26.86907507595606ms`，不作为端到端P95验收。
- `python -m smoke.dt_prefix_test --config ...V3...`：退出码0；invalid dt mask和padding梯度排除通过。
- `python -m smoke.formal_entry_contract --project-root F:\\Project\\RadarInsight --config ...V3...`：CUDA退出码0；1 step、V3 checkpoint schema、causal_prefix、dt_invalid=2通过；该artifact是entry-contract smoke，不是训练结果。
- V2兼容回归：CUDA加载`artifacts/RADAR-LN3-V2-cq08-tr-rd-formal/best_model.pt`退出码0，V2 schema、有限输出和V1 index hash通过。
- V1 anomaly view：`python -m validate.v1_anomaly_views --project-root F:\\Project\\RadarInsight --run-dataset-checks`退出码0。
- `python -m smoke.logging_test`、`python -m smoke.report_detail_test`：退出码0。

V3实际产物与hash：index=`data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-index.csv`, SHA256=`a5d054f044709047938c88ef5601caf7d4b482ef980a5a61f19db87b6e649f87`；contract=`data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-data-contract.json`, SHA256=`9aca5f7d12dc38bd9d624aa2f9f0906c785e056132beac00914db501e0487433`；split SHA256=`154d4efb1c9c34aa163e90f4b358e7a23a425b70505d468d532644a2a50c6e1d`；config=`config/RADAR-LN3/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-formal-supervised-config.json`, SHA256=`032ec97e36fba6bf35cc3ed385f081e99e73633bf75af706281e6e8dd29289e3`；V2 train-only stats复用，SHA256=`584eddcaeb8d78677f0f4902d622c2aa313d9374b767a47982c12dc058f826b6`。

V3数据统计：1546 parent、29663 point、CQ08-only；train/val/test parent=`1081/232/233`；有效dt=`18341`，无效dt=`11322`，其中首帧=`1546`、秒级量化zero-dt=`9776`；label parent=`Drone 698, Bird 348, Balloon 350, Clutter 50, Unknown 100`。V3不重建RD tensor。

Clutter/Unknown只读审计：manifest与index标签完全一致（Clutter 50 parent、Unknown 100 parent），原始航迹文件经GBK读取不含类别字段，类别来自CQ08统一manifest/index映射；session字段在manifest存在，V1 parent全部可映射且session不跨split，capture有27个跨split记录，不能直接作为分组键；两类验证parent支持为8/15，均无RD缺失点。当前未修改标签，拒识校准未执行。

本轮未执行：V3正式训练、全量V3 validation、test评估、端到端在线P95、拒识阈值校准、多seed、Clutter/Unknown专项训练、Liquid vs GRU对照和生产切换。

## 2026-08-05 - Stage 023收口与artifact边界修正

状态：Stage 023技术验收通过；Stage 024标签/覆盖只读审计部分完成；V3正式训练未执行，V3正式artifact目录保持未生成。

只读发现：早期`smoke.contract_tests`曾把普通`RADAR-LN3-V0-checkpoint-v1`、`step=1`的contract checkpoint写入`artifacts/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-formal/best_model.pt`。该文件不是V3 formal checkpoint，也不是训练结果。

已执行修正：在确认目标目录`artifacts/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-formal-legacy-smoke/`不存在后，将原目录原样移动到该legacy-smoke路径；未删除、未覆盖、未修改checkpoint内容。正式路径`artifacts/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-formal/`当前不存在，表示V3正式训练未执行。V3 formal-entry smoke的真实产物位于`artifacts/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-entry-contract/`，checkpoint schema=`RADAR-LN3-V3-formal-checkpoint-v1`，仅为有界契约证据。

最终验证：V3 compile、静态隔离、V3 contract、dt/prefix边界、V3 formal-entry CUDA smoke、V2 checkpoint兼容、V1 anomaly view、logging和report detail smoke均通过；V2 best checkpoint SHA256仍为`CF22838CD723D6C77BDC9681C49D6CEDF9CCA53308B90D99DF704B4F3FF0F458`。

## 2026-08-05 - Stage 025全链路审查收口启动

状态：基于交接状态继续执行；本轮不重建V3数据、不创建新配置、不启动V3正式训练、不读取test模型评估结果。V2旧artifact、V3数据契约和配置保持只读复用。

本轮实际只读核对：

- `git status --short`：工作树已有用户历史修改；`src-v3/`和V3配置为未跟踪新增内容，本轮未触碰`src/`及无关修改。
- V3配置实际为`config/RADAR-LN3/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-formal-supervised-config.json`；`loss_mode=causal_prefix`、`dt_valid_dim=1`、`batch_size=2`、`seed=42`、`evaluate_test=false`、预训练=`not_required_and_not_used`。
- V3正式artifact目录`artifacts/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-formal/`不存在；`artifacts/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-entry-contract/`只含1-step边界契约产物，不能作为正式模型。
- V3数据hash保持：index=`a5d054f044709047938c88ef5601caf7d4b482ef980a5a61f19db87b6e649f87`；contract=`9aca5f7d12dc38bd9d624aa2f9f0906c785e056132beac00914db501e0487433`；split=`154d4efb1c9c34aa163e90f4b358e7a23a425b70505d468d532644a2a50c6e1d`；V3 config=`032ec97e36fba6bf35cc3ed385f081e99e73633bf75af706281e6e8dd29289e3`；复用的V2 train-only stats=`584eddcaeb8d78677f0f4902d622c2aa313d9374b767a47982c12dc058f826b6`。无版本递增。

本轮实际验证命令与结果：

- `py_compile`覆盖全部`src-v3/*.py`：退出码0。
- `python -m smoke.static_isolation_check --root F:\\Project\\RadarInsight\\src-v3`：退出码0，`violations=[]`。
- `python -m validate.causal_contract --project-root F:\\Project\\RadarInsight`：退出码0；1546 parent、29663 point、train/val/test=1081/232/233、有效dt=18341、无效dt=11322、CQ08-only、split grouped。
- `python -m smoke.dt_prefix_test --config ...V3...`：退出码0；invalid dt mask、padding梯度排除和causal-prefix loss通过。
- `python -m smoke.missing_rd_mask_test --project-root ... --config ...V3...`：退出码0；真实CQ08缺失RD事件mask通过。
- `python -m smoke.contract_tests --project-root ... --config ...V3...`：CUDA退出码0；shape、forward、loss、梯度、因果前缀不变性、负dt处理、checkpoint round-trip和online point通过。
- `python -m smoke.formal_entry_contract --project-root ... --config ...V3...`：CUDA退出码0；`status=passed`、1 step、V3 checkpoint schema、`loss_mode=causal_prefix`、`dt_invalid_event_count=2`；结果仍是focused entry contract。
- `python -m smoke.latency_smoke --project-root ... --config ...V3... --iterations 8`：CUDA退出码0；模型-only p50=`25.1672 ms`、p95=`29.4091 ms`、max=`30.9205 ms`；不包含磁盘/RD解码/传输/服务调度，端到端P95仍未验收。

本轮V3数据只读统计：

- 29663 point、1546 parent；序列长度12–29，中位18，超过max_sequence_length=29的parent=0；sample_id重复=0；parent标签冲突=0；parent跨split=0；split_group/track/RD tensor跨split=0。
- `tr_dt_seconds==0`共11322/29663=`38.16%`，全部为`tr_dt_valid=false`；其中首帧无前驱1546，秒级量化zero-dt 9776/28117个非首点=`34.77%`；负dt=0。该零值来源已确认是秒级时间精度/首帧语义，不是padding或RD CPI时间替代。
- RD缺失311/29663=`1.05%`，且`rd_quality_missing`同步311；partial parent标记903行；无RD parent order anomaly。V3仍只消费合法TR+RD字段和显式mask。
- Clutter parent=50、Unknown parent=100；train/val/test分别为35/8/7和70/15/15；验证parent support仅8/15。当前未修改标签，未进行拒识阈值校准或专项训练。

审查结论：V3已修复V2的“只监督最终window、评估全部causal prefix”代码不一致，并显式保留无效dt语义；这两项仅有契约和Smoke证据，分类收益未评估。prefix使用parent完整标签的业务有效性仍是待验证风险。UnknownTarget仍是普通分类标签，stable模块没有reject/abstain动作；不能将低置信度诊断阈值当成拒识能力。液态模块继续作为受控时序候选，不作为Clutter/Unknown修复方案。

补充残余风险：V3 `loss_mode=causal_prefix`与window评估单位已一致，但`checkpoint_selection_metric=parent_macro_f1`，stable不参与选优；首轮保持该设置以隔离窗口/dt主变量，V3正式val后分别报告window、parent、stable，不把parent选优结果外推为线上stable最优。

本轮实际归档：

- 新增`src-v3/report/audit.py`，只读读取V3 index/contract/config/train-only stats，不读取checkpoint，不读取test，不修改data/config/formal artifact；执行`python -m report.audit --project-root F:\\Project\\RadarInsight --config F:\\Project\\RadarInsight\\config\\RADAR-LN3\\RADAR-LN3-V3-cq08-tr-rd-prefix-dt-formal-supervised-config.json --output F:\\Project\\RadarInsight\\artifacts\\RADAR-LN3-V3-audit.json`退出码0。
- 审计输出`artifacts/RADAR-LN3-V3-audit.json` SHA256=`678eef94db2b152152cc39e80414d29da33b37ee7ca1624373170bec8dd4776a`；入口SHA256=`d4305b1a457ed822772363353ab23fbcfbe2ad89cd99ff53a3c0b58d6b8b880`。
- 审计结果确认`source_purity_passed=true`、sample_id重复=0、parent标签/分割冲突=0、split_group/track/RD tensor跨split=0、序列长度12–29且超过配置上限=0；拒识字段明确为`reject_or_abstain_output=false`、校准和Coverage/Accepted/FAR/FRR/Selective Risk均`not_evaluated`。
- 新增`src-v3/report/rejection_design_v3.md`，记录parent-grouped OOF阈值来源、Coverage/Accepted指标、Unknown/Clutter拒识Recall、FAR/FRR、Selective Risk、单变量顺序和回滚条件；没有创建正式拒识配置。
- 归档后再次执行全量`py_compile`和`smoke.static_isolation_check`：均退出码0，`violations=[]`。

本轮未执行：V3正式训练、V3全量validation、test评估、拒识阈值/OOF校准、Clutter/Unknown专项采样或损失实验、多seed、Liquid vs GRU单因素对照、session级模型指标、端到端在线P95、生产切换。

补充只读统计：

- 遍历`data/processed/raw_tracks/*.csv`实际77条序列、11366点、11289个相邻对；正dt=10395、zero-dt=894、负dt=0；正dt最小约`3.5e-05s`、最大约`0.007438s`。raw标签映射为BirdTarget=17、DroneTarget=18、AircraftTarget=21、ShipTarget=21；raw不具备CQ08五分类、split或session契约，未进入V3。
- raw processed track audit：77个RD/TR样本均有RD tensor和MAT路径，`rd_phase_missing=false`为77条，height/latitude/longitude/amplitude_db/course_deg/speed_mps缺失为100%，SNR缺失为0；RD frame_count范围29–816，bin=512共58、1024共19。
- 实际读取CQ08 canonical NPZ示例`data/processed/cq08_rd_physical_canonical_v3_1/cq08__track__100_rd_tensor.npz`：amplitude/phase=`[120,21,384] float16`，time_seconds=`[120] float32`，velocity=`[384]`，range_offset=`[21]`，数组有限；raw NPZ示例为amplitude/phase=`[512,69] float32`、time_raw=`[69]`，两者输入契约不同。
- V3按序列长度统计命令未读checkpoint/test预测：12–15长度桶points=6727、invalid dt=2848；16–20 points=8960、invalid=2821；21–25 points=12968、invalid=5616；26–29 points=1008、invalid=37；所有负dt均为0。

跟踪记录修正后复核：再次执行全量`py_compile`和`smoke.static_isolation_check --root F:\\Project\\RadarInsight\\src-v3`，均退出码0，`violations=[]`；未改变代码、数据、配置和artifact。

## 2026-08-05 - Stage 027启动：V4无效dt观测更新单因素实验

状态：V4单因素实验准备开始；本次先完成代码和focused验证，不启动正式训练、全量validation、test、拒识校准或端到端P95。

版本与文件核对：

- 读取`data/`、`config/RADAR-LN3/`和本文件；当前已使用最高版本为`RADAR-LN3-V3`，下一个未使用全局版本为`RADAR-LN3-V4`。
- V3 index SHA-256=`a5d054f044709047938c88ef5601caf7d4b482ef980a5a61f19db87b6e649f87`。
- V3 data contract SHA-256=`9aca5f7d12dc38bd9d624aa2f9f0906c785e056132beac00914db501e0487433`。
- V3 split SHA-256=`154d4efb1c9c34aa163e90f4b358e7a23a425b70505d468d532644a2a50c6e1d`。
- V3 configuration SHA-256=`032ec97e36fba6bf35cc3ed385f081e99e73633bf75af706281e6e8dd29289e3`。
- V2 train-only normalization statistics SHA-256=`584eddcaeb8d78677f0f4902d622c2aa313d9374b767a47982c12dc058f826b6`，内容和语义不变，计划复用。
- `artifacts/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-formal/`不存在；现有V3目录仅为contract/entry-contract/legacy-smoke边界产物，不能作为正式模型。

只读证据：

- CUDA可用，设备为`NVIDIA GeForce RTX 4060 Laptop GPU`。
- `smoke.static_isolation_check`退出码0，`violations=[]`。
- `smoke.dt_prefix_test --config ...V3...`退出码0；无效dt mask、padding梯度排除和causal-prefix loss边界通过。
- `validate.causal_contract --project-root F:\\Project\\RadarInsight`退出码0；1546 parent、29663 point、CQ08-only、train/val/test parent=1081/232/233、有效dt=18341、无效dt=11322。
- `src-v3/radar_ln3/model.py`的`CfCLiquidCausalEncoder._step`当前对`~dt_valid`使用`zero_dt_event_floor_seconds=0.001`；该floor不恢复物理时间。V3非首点秒级量化zero-dt为9776/28117=`34.77%`。
- 同文件`RDSpatialEncoder.forward`对同一TR点的RD frame embedding执行mask平均；RD局部`time_seconds`目前只用于质量duration，未进入帧级时序编码。该项作为后续独立风险，不在V4主变量内修改。

V4实验边界：

- 只改变无效dt事件的液态状态更新系数；有效dt连续更新、TR/RD tensor、mask、label、split、loss、checkpoint选优指标和stable状态机保持不变。
- V3继续表示floor对照；V4启用有界观测事件增益，保留`tr_dt_valid`和`dt_invalid_mask`。
- 不创建新的数据index、manifest、split、data contract或RD tensor；V4配置语义改变，使用新的`RADAR-LN3-V4`配置和独立artifact路径。

本阶段尚未执行：V4代码修改、V4配置生成、V4 Smoke、正式训练、V4 validation、test、拒识校准、Liquid/GRU对照和端到端P95。

## 2026-08-05 - Stage 027完成：V4无效dt观测更新focused验证

状态：V4代码、配置和focused Smoke完成；正式V4训练、全量V4 validation、test、拒识校准、Liquid对照和端到端P95未执行。

实际修改文件：

- `src-v3/radar_ln3/model.py`：在`CfCLiquidCausalEncoder`增加`invalid_dt_update_mode`和`invalid_dt_observation_gain`；`floor`保持V3行为，`bounded_observation`仅对`tr_dt_valid=false`事件使用固定有界观测增益；有效dt路径不改变。`RadarLiquidClassifier`传递配置字段。
- `src-v3/radar_ln3/config.py`：支持RADAR-LN3-V4配置、V3数据契约复用字段和液态更新字段校验。
- `src-v3/radar_ln3/contracts.py`：允许V4配置校验V3数据契约，仍核对V3 index/contract/stats hash和CQ08纯度。
- `src-v3/radar_ln3/factory.py`、`src-v3/train/formal.py`：支持V4 formal supervised入口和模型构造。
- `src-v3/smoke/contract_tests.py`、`audit_forward_test.py`、`latency_smoke.py`、`missing_rd_mask_test.py`、`run_smoke.py`：模型字段传递和复用V3数据行schema修正；Smoke不再把配置版本误当成数据行版本。
- `src-v3/smoke/invalid_dt_update_test.py`：新增V4模式、mask、状态变化、有限值和梯度focused Smoke。
- `config/RADAR-LN3/RADAR-LN3-V4-cq08-tr-rd-invalid-dt-event-formal-supervised-config.json`：新增V4配置，配置SHA-256=`8aa89764a293a5f938f42be6eda5f44926a92149fec5e3a56be8c73a10951d35`。

数据与版本：

- 未创建或修改新的index、manifest、split、data contract、RD tensor或归一化统计；复用V3 index SHA-256=`a5d054f044709047938c88ef5601caf7d4b482ef980a5a61f19db87b6e649f87`、V3 contract SHA-256=`9aca5f7d12dc38bd9d624aa2f9f0906c785e056132beac00914db501e0487433`、V3 split SHA-256=`154d4efb1c9c34aa163e90f4b358e7a23a425b70505d468d532644a2a50c6e1d`和V2 train-only stats SHA-256=`584eddcaeb8d78677f0f4902d622c2aa313d9374b767a47982c12dc058f826b6`。
- V4版本递增仅因为模型输入/更新配置语义发生实质变化；数据版本保持V3，不生成V4数据文件。
- V4正式artifact目录`artifacts/RADAR-LN3-V4-cq08-tr-rd-invalid-dt-event-formal/`未生成；最终规范Smoke产物位于`artifacts/RADAR-LN3-V4-cq08-tr-rd-invalid-dt-event-contract-smoke/`和`artifacts/RADAR-LN3-V4-cq08-tr-rd-invalid-dt-event-entry-contract/`，均不是正式模型。
- 早期重跑在命名规范修正前生成的`artifacts/RADAR-LN3-V4-cq08-tr-rd-prefix-dt-contract-smoke/`和`artifacts/RADAR-LN3-V4-cq08-tr-rd-prefix-dt-entry-contract/`保留为历史边界产物；未删除、未覆盖，不作为V4正式训练或最终Smoke路径。

实际验证命令与结果：

- `..\\.venv\\Scripts\\python.exe -m compileall -q .`：通过。
- `..\\.venv\\Scripts\\python.exe -m smoke.static_isolation_check --root F:\\Project\\RadarInsight\\src-v3`：通过，`violations=[]`。
- `..\\.venv\\Scripts\\python.exe -m smoke.invalid_dt_update_test --config ...RADAR-LN3-V4...json`：通过；mode=`bounded_observation`、gain=`0.25`、invalid mask正确、状态有限、梯度有限。
- `..\\.venv\\Scripts\\python.exe -m validate.causal_contract --project-root F:\\Project\\RadarInsight`：通过；V3数据1546 parent/29663 point、CQ08-only、有效dt=18341、无效dt=11322、split grouped。
- `..\\.venv\\Scripts\\python.exe -m smoke.contract_tests --project-root F:\\Project\\RadarInsight --config ...RADAR-LN3-V4...json`：通过；CUDA，真实batch shape为TR `[2,22,14]`、missing `[2,22,11]`、dt `[2,22]`、RD `[2,22,12,2,21,384]`、quality `[2,22,7]`；forward/loss/梯度/因果性/负dt/checkpoint/online point均通过；formal training未执行。
- `..\\.venv\\Scripts\\python.exe -m smoke.missing_rd_mask_test --project-root F:\\Project\\RadarInsight --config ...RADAR-LN3-V4...json`：通过；真实parent `cq08|track|1010` 的2个missing-RD事件mask和quality通过。
- `..\\.venv\\Scripts\\python.exe -m smoke.formal_entry_contract --project-root F:\\Project\\RadarInsight --config ...RADAR-LN3-V4...json`：通过；CUDA、1 step、checkpoint schema=`RADAR-LN3-V4-formal-checkpoint-v1`、`dt_invalid_event_count=2`；仅为entry contract。
- `..\\.venv\\Scripts\\python.exe -m smoke.latency_smoke --project-root F:\\Project\\RadarInsight --config ...RADAR-LN3-V4...json --iterations 4`：通过；模型-only p50=`32.81730005983263ms`、p95=`41.60169998649508ms`、max=`42.226499994285405ms`；端到端P95仍未验收。
- `..\\.venv\\Scripts\\python.exe -m smoke.audit_forward_test --project-root F:\\Project\\RadarInsight --config ...RADAR-LN3-V1-cq08-tr-rd-anomaly-audit-config.json`：通过；3个异常parent forward-only；V4正式配置不适用于该`allow_parameter_updates=false`审计入口，未把该入口条件误记为V4模型失败。
- V3回归`smoke.contract_tests --config ...RADAR-LN3...V3...json`：通过；V3仍可加载并完成CUDA forward/loss/梯度/checkpoint/online contract。

本阶段结论：V4达到“代码和短时Smoke可继续实验”的状态，不代表V4分类性能改善。V3与V4的训练数据、label、split、loss和RD前端未变；尚无V4正式checkpoint、validation指标或Clutter/Unknown收益证据。

## 2026-08-05 - Stage 028启动：V3/V4 paired正式监督训练授权

状态：用户已明确选择“V3与V4都训练，获得可归因的paired baseline对照”；本轮只读核对并输出手动命令，本代理未启动任何训练进程。

只读核对结果：

- 训练入口：`src-v3/train/formal.py`，支持`--project-root`、`--config`、`--max-steps`、`--epochs`和`--skip-validation`；正式命令不使用后三个focused/覆盖参数。
- V3配置：`config/RADAR-LN3/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-formal-supervised-config.json`；数据为V3 CQ08 index；seed=42、batch=2、epochs=20、early_stopping_patience=5、loss=`causal_prefix`、checkpoint selection=`parent_macro_f1`、device=`cuda`、`evaluate_test=false`。
- V4配置：`config/RADAR-LN3/RADAR-LN3-V4-cq08-tr-rd-invalid-dt-event-formal-supervised-config.json`；复用同一V3 CQ08 index/contract/split和V2 train-only统计；seed=42、batch=2、epochs=20、early_stopping_patience=5、loss=`causal_prefix`、checkpoint selection=`parent_macro_f1`、device=`cuda`、`evaluate_test=false`；仅液态无效dt模式为`bounded_observation`、gain=0.25不同。
- V3正式artifact目录`artifacts/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-formal/`不存在；V4正式artifact目录`artifacts/RADAR-LN3-V4-cq08-tr-rd-invalid-dt-event-formal/`不存在；本轮无覆盖风险。
- V3 index SHA-256=`a5d054f044709047938c88ef5601caf7d4b482ef980a5a61f19db87b6e649f87`；V3 contract SHA-256=`9aca5f7d12dc38bd9d624aa2f9f0906c785e056132beac00914db501e0487433`；V3 split SHA-256=`154d4efb1c9c34aa163e90f4b358e7a23a425b70505d468d532644a2a50c6e1d`；V2 train-only stats SHA-256=`584eddcaeb8d78677f0f4902d622c2aa313d9374b767a47982c12dc058f826b6`。
- 预训练：两套配置均不需要且不使用；均为随机初始化端到端监督训练。

本阶段未执行：V3正式训练、V4正式训练、validation、test、paired指标比较、拒识校准和端到端P95。

## 2026-08-05 - Stage 026原始回波抽样审计完成

状态：用户授权抽样审计12个CQ08原始回波文件；MATLAB实际运行完成，报告已生成。原始.dat、现有MAT/RD、点迹、航迹、V3索引、配置和正式artifact保持只读。

本轮已读取：

- OriData/23/CQ-08/batch_process_all.m：确认原始.dat经帧头/帧长/帧尾解析，读取复数I/Q data_out，沿慢时间FFT生成复数data_proc_MTD_result，并保存Freq/CPIcount/PRTnum/PRT、index_trackPointNo和debug_info。
- OriData/23/CQ-08/原始回波：1550个.dat文件，约10.51 GB；当前未全量读取内容。
- OriData/23/CQ-08/MAT：106026个.mat文件，约41.89 GB；当前未全量重建或覆盖。
- data/manifests/cq08_unified_manifest.json：1550个CQ08 parent，包含raw echo、point、track、MAT列表、标签和track_length_hint。
- data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-index.csv和V3 contract：当前V3仍使用既有canonical RD tensor与显式dt契约，本Stage不改变版本或hash。
- MATLAB可执行文件：F:/Software/Matlab/bin/matlab.exe，只读检查确认存在。
- 创建：src-v3/tests/processed/original_echo_audit.m；功能包括12文件确定性抽样、原始帧头/帧长/帧尾解析、31路复数I/Q读取、CPI/PRT统计、现有MAT的index_trackPointNo/debug_info读取、首中末帧data_out与RD重算对照、点迹/航迹时间列覆盖检查、已知4字节尾部之外的未解析载荷检查和JSON/CSV报告写出。
- 脚本SHA-256：C2B48E4FE56328CDAF62358CD0CC7FEEF8B3E27AD8B427774F79492EB2DD626F。PowerShell静态核对确认无MATLAB save调用、无src-v3外部模块导入、无data写出或配置/artifact写出；报告只写入src-v3/processed/report/。
- MATLAB实际命令：`cd('F:\\Project\\RadarInsight'); addpath(fullfile(pwd,'src-v3','tests','processed')); outputs=original_echo_audit(pwd); disp(outputs);`。
- 实际输出：`status=completed`、`sample_count=12`、`raw_frame_count_match=1`、`raw_mat_data_out_comparable=1`、`raw_mat_rd_comparable=1`。
- 报告实际SHA-256：summary=`3A01009C5D7119742F764485E766D0C04D31C52C05F2751DF89EB685C7ACA0D6`；selection=`2F628AA93FFA31C0ED13F64229ED67F665FE4C3F4259C8474BEF17D2FA534659`；samples=`7171769D542EA21A186372DEB81546803996EDC01AC1C3491585D23630A45FF2`；frames=`FEADFA585400E0D10A761C8FC80ED5E8EE247C6C91B45427B99679749B5294F3`；RD comparison=`00CAC9E1C28DAA57EF86B62B3AA2D9DBEB2216C74CD10CB30BB27C912ED9A50B`。
- 实际抽样：track_id=`1,4260,66,335,3010,3757,2125,13,12,2,4,142`；标签覆盖DroneTarget=7、BirdTarget=2、BalloonTarget=1、ClutterTarget=1、UnknownTarget=1；track_length_hint覆盖12、15、18、21、22、23、29。
- 实际帧级结果：共1023个原始帧，逐样本raw帧数与manifest MAT数全部一致；36个首/中/末帧对照的`data_out` shape和数值均匹配，重算复数RD shape和数值均匹配，最大绝对/相对误差均为0；12个sample状态全部为`ok`；无MAT缺失、索引越界、帧/MAT序号异常、非有限I/Q或未解析隐藏载荷。
- 实际时间/形状结果：原始帧未检测到显式wall-clock字段，MAT debug_info也未检测到wall-clock字段；仅确认帧顺序、CPI/PRT和局部观测时长。1023帧中CPIcount=0/1/2/3分别为368/379/138/138；PRTnum=142/226/360/570/900分别为66/549/346/58/4；I/Q shape为31x142/31x226/31x360/31x570/31x900，全部有限；每个样本的point/track索引时间覆盖均为true。
- 解释边界：12个样本支持“现有MAT由当前原始解析链一致生成”，不支持1550个CQ08全量一致性结论；一个TR point对应多个raw/MAT帧，不能把raw frame index直接当作TR wall-clock时间；CPIcount=0代表该字段在部分帧不能提供正观测时长，不能直接作为真实零时间间隔。
- 后续只读代码核对：`src/process/build_cq08_rd_physical_canonical_store.py`在`observation_seconds<=0`时使用`prt_num*prt_s`回退并累加RD局部时间；这解释了当前canonical RD局部时间可以保持累积，但回退值不是已验证的真实wall-clock/CPI时长。该事实记录为Finding F045，本轮未修改。

审计设计：固定选择首尾track边界、五类标签代表、全局短/长序列和长度分位补充，去重后正好12个文件；每个文件扫描原始帧并统计I/Q、CPI/PRT、时间可用性和帧间结构；读取对应MAT的索引/元数据；比较点迹/航迹行数与index_trackPointNo；抽查首/中/末帧重算RD与MAT RD数值误差。报告只写入src-v3/processed/report/。

版本与范围：本Stage不创建索引、manifest、split、数据契约或配置，因此不递增RADAR-LN3版本；不使用Excel、raw独立数据或其他source；不启动训练、test评估、全量RD重建或生产切换。

未执行：1550个文件全量原始回波审计、原始回波训练数据重建、未压缩I/Q与现有canonical RD的信息增益实验、原始采集设备wall-clock字段恢复、正式训练和test评估。

## 2026-08-06 - Stage 029液态模型DBB/CU专项启动

用户范围已确认：不继续依赖或审查现有RD-only模型；当前只围绕已训练的TR+RD融合液态模型构建独立DBB/CU分层任务。目标是先验证DBB/CU二分类gate，再验证DBB三分类和CU二分类specialist；若三者均有稳定收益，后续才讨论软级联模块化接入。

本轮只读核实：

- V3 best checkpoint：`artifacts/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-formal/best_model.pt`；V3 validation parent support=232、Accuracy=`0.8620689655172413`、Macro-F1=`0.7530281349855817`；Clutter parent F1=`0.5714285714285714`、Unknown parent F1=`0.5128205128205129`。
- V4 best checkpoint：`artifacts/RADAR-LN3-V4-cq08-tr-rd-invalid-dt-event-formal/best_model.pt`；V4 validation parent support=232、Accuracy=`0.8362068965517241`、Macro-F1=`0.7140031743844241`；Clutter parent F1=`0.6666666666666667`、Unknown parent F1=`0.3225806451612903`。
- V3/V4均使用同一CQ08 V3 index、V3 data contract、V2 train-only RD归一化统计、五类原始标签和causal-prefix loss；V4只改变无效dt的液态更新模式。上述指标是五分类结果，不能当作独立DBB/CU模型结果。
- 当前液态模型真实输入为TR特征、TR missing mask、TR dt/dt_valid、每个TR点的RD帧、RD frame mask、RD quality和padding mask；RD空间embedding后进入CfC风格因果液态编码器，V4采用`bounded_observation`。
- V3数据实际label parent counts为Drone=698、Bird=348、Balloon=350、Clutter=50、Unknown=100；DBB/CU任务映射只由原五类`label_name`派生，不修改标签和样本集合。

版本核对：本轮已读取`data/`、`config/RADAR-LN3/`和本文件；现有最高版本为`RADAR-LN3-V4`。由于任务输出类别、标签映射、checkpoint schema和配置语义发生实质变化，配置批次使用全局下一个版本`RADAR-LN3-V5`。不创建新的index、manifest、split、data contract或RD tensor；V3 index、V3 contract和V2 normalization stats在配置中按已核实hash复用。V5同一批次关联DBB/CU gate、DBB specialist和CU specialist三份配置，分别表达不同输出类别契约。

实施边界：任务目录边界为`src-v3/tests/cu/`和`src-v3/tests/dbb-cu/`，共享实现位于`src-v3/radar_ln3/task_models.py`；不修改`src/`、V3/V4代码、旧checkpoint或正式artifact。Smoke代码入口位于`src-v3/smoke/`，Smoke JSON和Smoke checkpoint只写入`src-v3/smoke/`；正式训练、全量validation、test、拒识校准、软级联性能和端到端P95本轮未执行。

实际创建与验证：

- `src-v3/radar_ln3/task_models.py`：独立任务配置、CQ08-only hash校验、标签映射、液态TR+RD任务模型、causal-prefix loss、任务评估、checkpoint和`P(DBB/CU)×P(specialist)`软组合器；不导入`src`，不读取V3/V4五分类checkpoint。
- `src-v3/tests/cu/__init__.py`、`src-v3/tests/dbb-cu/__init__.py`、`src-v3/tests/liquid_tasks/__init__.py`：任务目录/命名边界。由于`dbb-cu`含连字符，不使用点式Python模块启动；共享运行实现位于`src-v3/radar_ln3/task_models.py`，正式入口位于`src-v3/train/tasks.py`和`src-v3/report/tasks.py`。
- `src-v3/tests/cu/contract.py`、`src-v3/tests/cu/README.md`：CU specialist任务契约和目录说明；`src-v3/tests/dbb-cu/contract.py`、`src-v3/tests/dbb-cu/README.md`：DBB/CU gate、DBB specialist、CU specialist和软组合契约及目录说明。新增文件只引用`src-v3`内部任务实现，不引入`src`。
- `src-v3/train/tasks.py`：V5任务正式训练入口，支持`--max-steps`/`--epochs`/`--skip-validation`，本轮未启动正式训练。
- `src-v3/report/tasks.py`：V5任务train/val评估入口，test显式拒绝，本轮未执行正式评估。
- `src-v3/smoke/task_dbb_cu.py`、`src-v3/smoke/task_cu.py`：Smoke入口与结果目录，正式链路不引用Smoke代码。
- `config/RADAR-LN3/RADAR-LN3-V5-cq08-liquid-dbb-cu-gate-config.json`、`...dbb-specialist-config.json`、`...cu-specialist-config.json`：三份V5任务配置；V3 index SHA256=`a5d054f044709047938c88ef5601caf7d4b482ef980a5a61f19db87b6e649f87`，V3 contract SHA256=`9aca5f7d12dc38bd9d624aa2f9f0906c785e056132beac00914db501e0487433`，V2 stats SHA256=`584eddcaeb8d78677f0f4902d622c2aa313d9374b767a47982c12dc058f826b6`；三份配置均复用上述hash，未创建新数据版本。
- 三个Smoke实际使用CUDA并通过：gate loss=`0.6565355062484741`、grad norm=`11.522035598754883`、batch `[2,22,14]`/`[2,22,12,2,21,384]`、输出`[2,22,2]`、软组合`[2,22,5]`；DBB specialist loss=`1.191115379333496`、grad norm=`13.28442668914795`、输出`[2,22,3]`；CU specialist loss=`0.40188688039779663`、grad norm=`6.550888538360596`、输出`[2,15,2]`。三者checkpoint round-trip均通过，结果文件和checkpoint均位于`src-v3/smoke/RADAR-LN3-V5-*-contract-smoke/`。
- 实际检查：`..\\.venv\\Scripts\\python.exe -m compileall -q .`通过；`..\\.venv\\Scripts\\python.exe -m smoke.static_isolation_check --root F:\\Project\\RadarInsight\\src-v3`通过，`violations=[]`。V5任务配置加载通过。
- 规范化入口复核：`..\\.venv\\Scripts\\python.exe -m train.tasks --help`和`..\\.venv\\Scripts\\python.exe -m report.tasks --help`均退出码0；`src-v3`内静态搜索仅命中隔离检查器自身的禁止模式正则，未发现运行时`from src`、`import src`、`sys.path`或动态加载依赖；`artifacts/`下不存在任何`RADAR-LN3-V5-*`正式目录。
- V5配置SHA256：gate=`DEC197B64AB5D88D404B9A16F7BDBC09F59D1844CF218174CF39C66F679FB83C`；DBB specialist=`3EEBCD93E76AF689C8261F0B8654A6B841AEF77AB1C90FFB8C87B8EC4D32FA9A`；CU specialist=`0D5AD77FF98096A832B7CEE7D438F386FEE59385CBF9BD455959500AA0B05F16`。

版本/文件记录：V5配置为任务输出类别、标签映射、checkpoint schema和配置语义实质变化后的全局下一个版本；V3 index、V3 data contract、V2 normalization stats内容未变，按hash复用。Smoke checkpoint不是正式模型artifact；V5正式训练、validation、test、软级联性能、拒识校准和P95均为“未执行”。

历史放行状态（已由Stage 030撤销）：V5三项任务当时基于代码编译、配置加载、CQ08数据hash、shape/mask/forward/loss/梯度/checkpoint和隔离证据进入“手动正式训练准备”；Stage 030发现V5模型输入包含parent全局RD质量泄漏后，该放行不再有效。V5只保留审计，三项正式训练、独立val、soft-cascade组合评估、90%门槛、跨seed、拒识、test和P95均未执行。

历史手动训练命令（未执行；Stage 030后禁止执行，仅保留追溯）：

- Gate：`Set-Location F:\Project\RadarInsight\src-v3; & ..\\.venv\\Scripts\\python.exe -m train.tasks --project-root F:\Project\RadarInsight --config F:\Project\RadarInsight\\config\\RADAR-LN3\\RADAR-LN3-V5-cq08-liquid-dbb-cu-gate-config.json`
- DBB specialist：`Set-Location F:\Project\RadarInsight\src-v3; & ..\\.venv\\Scripts\\python.exe -m train.tasks --project-root F:\Project\RadarInsight --config F:\Project\RadarInsight\\config\\RADAR-LN3\\RADAR-LN3-V5-cq08-liquid-dbb-specialist-config.json`
- CU specialist：`Set-Location F:\Project\RadarInsight\src-v3; & ..\\.venv\\Scripts\\python.exe -m train.tasks --project-root F:\Project\RadarInsight --config F:\Project\RadarInsight\\config\\RADAR-LN3\\RADAR-LN3-V5-cq08-liquid-cu-specialist-config.json`
- 评估模板：`Set-Location F:\Project\RadarInsight\src-v3; & ..\\.venv\\Scripts\\python.exe -m report.tasks --project-root F:\Project\RadarInsight --config <V5配置> --checkpoint <V5正式artifact>\\best_model.pt --split val`；test由入口拒绝，当前未授权。

## 2026-08-06 - Stage 030因果质量与软级联闭环完成

状态：已完成RD质量因果输入、Macro/stable语义、三checkpoint联合评估能力、在线三状态接口、checkpoint严格绑定和两个互斥消融开关的代码与focused验证。未修改`src/`、`data/`、`config/`或正式artifact；正式训练未执行。

只读检查与实际证据：

- `git status --short`确认工作区存在大量用户已有修改和未跟踪文件；本Stage严格避开`src/`、根目录报告、RD-only/TR-only、`data/`、`config/`和既有正式artifact，只修改`src-v3/`。
- 读取`src-v3/build/cq08_point_contract.py`第310-316、408-409行和`src-v3/radar_ln3/data.py`第240-249行：`rd_quality_partial_parent`由完整parent的期望点集合与全部RD点集合比较后生成，并被每个事件作为模型输入；`rd_quality_parent_order_anomaly`同样来自完整parent。V3记录中partial标记覆盖903行，后者当前主数据为0。
- 读取并以纯内存探针复现`metrics.py`与`task_models.py`：二分类全部预测第一类时当前Macro Precision输出0.5而固定两类zero-division=0应为0.25；五分类探针输出0.2而固定五类应为0.04。
- 读取并以纯内存探针复现`stable.py`：`confirm_count=2`时第一次明显Drone更新返回window=Drone、stable=Unknown、`candidate_count<confirm_count`；`streaming.OnlineStepResult`没有`stable_confirmed`字段。
- 全量搜索`compose_soft_cascade`和online入口：V5软组合只存在于`task_models.py`、契约文件和使用全零临时specialist logits的Smoke；没有三checkpoint联合加载/评估器，`OnlineTargetClassifier`只接受五分类`RadarLiquidClassifier`。
- 液态只读数值探针确认当前递归状态在指数插值后立即LayerNorm；相同首事件的`dt=0.1s`与`10s`通过同型分类头后最大logit差约`1.6e-5`。该结果证明时间幅值可能被显著压缩，但不是分类收益结论。
- 当前RD前端仍对同一TR点的frame embedding做masked mean；该事实已记录为F046，点内时序收益未评估。

版本与兼容边界：本Stage不创建索引、manifest、split、数据契约或配置，因此不分配新的RADAR-LN3版本。移除parent全局质量输入会改变特征语义，即使shape仍为7维；V3/V4及既有V5 Smoke checkpoint只能作为旧语义参考，不能静默用于修复后正式结果。下一次正式训练前必须在授权扩大到`data/`和`config/RADAR-LN3/`后创建下一全局版本并记录hash；本Stage未执行。

实际修改文件（全部位于`src-v3/`）：`radar_ln3/data.py`、`metrics.py`、`stable.py`、`streaming.py`、`evaluator.py`、`model.py`、`config.py`、`factory.py`、`contracts.py`、`checkpoint.py`、`task_models.py`、新增`cascade.py`；`train/formal.py`、`train/tasks.py`；新增`report/cascade.py`并修改`report/migrate_training_summary.py`；新增`smoke/causal_cascade_contract_test.py`并修改`smoke/report_detail_test.py`及旧Smoke构造器。未创建或修改数据、配置和正式artifact。

实现结果：

- RD质量模型输入schema冻结为`causal-current-event-v1`，字段为`rd_present`、`rd_frame_count_scaled`、`rd_local_duration_scaled`、`rd_missing`、`rd_local_time_non_monotonic`、`rd_multiple_frames`、`rd_frames_truncated`。旧索引中的`rd_quality_partial_parent`和`rd_quality_parent_order_anomaly`继续保留作离线审计，但Dataset不再消费。
- 真实partial parent `cq08|track|1010`上翻转两个parent全局字段后，模型质量tensor最大差值=`0.0`。
- 固定五类和专项类别均使用zero-division=0并保留全部合同类别Macro分母；退化探针五类Macro Precision=`0.04`、二类=`0.25`。
- 未确认stable输出`None`并显式`stable_confirmed=false`；stable指标只统计confirmed并报告coverage/unconfirmed。概率入口不执行二次softmax，0.8输入仍输出`0.800000011920929`；同一实例混用logits和probabilities会被拒绝。
- `radar_ln3/cascade.py`实现三模型严格组合、五类软概率、联合window/parent/stable评估、gate/DBB/CU组件指标、source/parent-source/长度/置信度/dt_invalid报告，以及每目标Gate/DBB/CU三份液态hidden state和reset接口。联合报告明确session字段不可用、概率温度校准/ECE未执行。
- 专项checkpoint现在绑定输入/动力学模型契约与index/contract/stats三个SHA-256；旧checkpoint缺字段或数据hash错误均拒绝。V5配置被标记为audit-only，模型执行要求`RADAR-LN3-V6`，防止通过补字段绕过版本边界。
- 两个默认baseline分别为液态`recurrent`归一化和RD`masked_mean`；候选为`readout_only`和`causal_gru`，同一配置同时开启会被拒绝。RD候选只使用已有帧顺序，不宣称使用可靠frame物理dt。
- 运行配置版本与数据契约版本已显式分离：修复基线配置至少为V6并声明`data.contract_schema_version`；后续V7/V8纯模型消融可以复用V6数据契约及其hash，但各自必须使用对应版本checkpoint schema。V5仍为audit-only。

实际命令与结果：

- `..\\.venv\\Scripts\\python.exe -m smoke.logging_test`：passed。
- `..\\.venv\\Scripts\\python.exe -m smoke.report_detail_test`：passed；sample detail包含`window/parent/stable/stable_unconfirmed`四类记录槽。
- `..\\.venv\\Scripts\\python.exe -m smoke.causal_cascade_contract_test --project-root F:\\Project\\RadarInsight`：passed，结果持久化到`src-v3/smoke/causal_cascade_contract_result.json`。最终结果包括V6修复配置/契约配对、V7模型配置复用V6数据契约、`legacy_config_rejected=true`、`parent_global_quality_max_difference=0.0`、`stable_mixed_mode_rejected=true`、`legacy_checkpoint_rejected=true`、`checkpoint_data_hash_rejected=true`、`mixed_cascade_versions_rejected=true`；baseline/checkpoint round-trip/readout-only/causal-RD-GRU/V7复用契约的batch-online最大差均不超过`8.940696716308594e-08`；联合概率和在float误差内为1；双消融同时开启被拒绝。
- `..\\.venv\\Scripts\\python.exe -m compileall -q .`：通过。
- `..\\.venv\\Scripts\\python.exe -m smoke.static_isolation_check --root F:\\Project\\RadarInsight\\src-v3`：passed，`violations=[]`。
- `..\\.venv\\Scripts\\python.exe -m report.cascade --help`：退出码0。

版本与文件状态：本Stage没有创建索引、manifest、split、数据契约、统计或配置，因此没有分配新文件版本。检查`data/`、`config/RADAR-LN3/`和`artifacts/`均不存在`RADAR-LN3-V6*`文件/目录。V6是下一次正式输入语义版本：可按hash复用V3索引内容与V2 train-only统计，但必须创建V6数据契约和三份V6配置，并让checkpoint使用V6 schema；该工作未获本轮目录授权且未执行。

未执行：V6数据契约/配置创建、V6三模型有界训练入口Smoke、三个正式checkpoint训练、真实联合validation、test、概率校准/ECE、拒识阈值、正式单因素消融训练、端到端P95和生产切换。

当前放行结论：不能进入正式训练。阻塞项是不存在V6数据契约、三份V6配置和对应有界Smoke证据；V5历史命令不得执行。

## 2026-08-06 - Stage 031 V6契约与配置创建启动

状态：用户回复“继续”，在上一轮已明确的下一步范围内推进V6数据契约和三份任务配置创建；只复用V3 index和V2 train-only统计，不覆盖旧文件。正式训练、validation、test和正式消融未执行。

创建前只读核验：

- data中最高RADAR-LN3数据版本为V3，config/RADAR-LN3中最高任务配置版本为V5；未发现V6文件或目录。
- 复用输入为`data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-index.csv`，SHA-256=`a5d054f044709047938c88ef5601caf7d4b482ef980a5a61f19db87b6e649f87`。
- 原V3数据契约只读参考为`data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-data-contract.json`，SHA-256=`9aca5f7d12dc38bd9d624aa2f9f0906c785e056132beac00914db501e0487433`。
- 复用训练统计为`data/processed/RADAR-LN3-V2-cq08-tr-rd-train-normalization-stats.json`，SHA-256=`584eddcaeb8d78677f0f4902d622c2aa313d9374b767a47982c12dc058f826b6`；scope明确为train且不含validation/test统计。
- V6数据契约将保留V3数据范围、split、时间字段、TR/RD shape和CQ08-only纯度，新增明确的`causal-current-event-v1` RD质量schema及字段名；V6配置将把`index_schema_version`保持为V3，把`contract_schema_version`设置为V6。

下一步实际操作：创建V6数据契约，计算新契约hash，再创建Gate/DBB/CU三份V6配置；随后只运行配置/契约校验和有界Smoke。未执行正式训练。

## 2026-08-06 - Stage 031 V6契约、配置与有界Smoke完成

实际创建文件：

- `data/processed/RADAR-LN3-V6-cq08-tr-rd-causal-quality-data-contract.json`，最终SHA-256=`2110938d1d94923edb44c2825e34fb595fb1b34c1353f3f3e512403f91ed8777`。
- `config/RADAR-LN3/RADAR-LN3-V6-cq08-liquid-dbb-cu-gate-config.json`，SHA-256=`5c2676b17469378c6967edb43a9690324f0f6d4e18489df8cebfdb137bdd63e0`。
- `config/RADAR-LN3/RADAR-LN3-V6-cq08-liquid-dbb-specialist-config.json`，SHA-256=`3f07df8196bc2d3cee16500838ac777bc4e83ca49c3f318e616194c6b6359aef`。
- `config/RADAR-LN3/RADAR-LN3-V6-cq08-liquid-cu-specialist-config.json`，SHA-256=`ab28daec036966d8554b86fe409bd43c63927e4233b99cf898b9db9191a87fd2`。
- V6契约引用的V3 index SHA-256=`a5d054f044709047938c88ef5601caf7d4b482ef980a5a61f19db87b6e649f87`，V2 train-only stats SHA-256=`584eddcaeb8d78677f0f4902d622c2aa313d9374b767a47982c12dc058f826b6`；二者未修改。

V6契约实际语义：schema=`RADAR-LN3-V6`；index schema保持V3；`rd_quality_schema=causal-current-event-v1`；模型质量字段为7维当前事件字段，排除`rd_quality_partial_parent`、`rd_quality_parent_order_anomaly`、`rd_parent_coverage_status`和`rd_parent_order_status`；source purity为CQ08-only；formal training/test/P95均明确未执行。

实际校验与Smoke：

- 三份V6配置加载、级联共享契约和数据校验：passed；三者均为V6配置/V6 contract schema，共享同一index/contract/stats hash。
- Gate Smoke：passed，CUDA，parent_limit=6，label counts DBB=3/CU=3，batch TR=`[2,22,14]`，RD=`[2,22,12,2,21,384]`，rd_quality=`[2,22,7]`，logits=`[2,22,2]`，loss=`0.7363489866256714`，gradient norm=`10.731317520141602`，checkpoint round-trip passed。
- DBB specialist Smoke：passed，CUDA，label counts Drone=2/Bird=2/Balloon=2，logits=`[2,22,3]`，loss=`1.278416395187378`，gradient norm=`12.225194931030273`，checkpoint round-trip passed。
- CU specialist Smoke：passed，CUDA，label counts Clutter=3/Unknown=3，logits=`[2,15,2]`，loss=`0.40960559248924255`，gradient norm=`6.805363178253174`，checkpoint round-trip passed。
- Smoke结果及临时checkpoint分别写入`src-v3/smoke/RADAR-LN3-V6-dbb-cu-gate-contract-smoke/`、`RADAR-LN3-V6-dbb_specialist-contract-smoke/`和`RADAR-LN3-V6-cu_specialist-contract-smoke/`，均标记为bounded smoke only，不是正式artifact。
- `compileall -q .`：passed；`smoke.static_isolation_check`：passed，`violations=[]`；`train.tasks --help`、`report.tasks --help`、`report.cascade --help`：均退出码0；Stage 030因果级联回归：passed。

异常与修复记录：首次V6 Smoke发现两个Smoke入口硬编码`RADAR-LN3-V5`输出目录，V6结果曾覆盖旧V5 Smoke目录；未覆盖正式artifact。已将Smoke输出改为动态使用配置schema，并写入`src-v3/smoke/RADAR-LN3-V5-smoke-artifact-integrity-warning.md`；旧V5 Smoke目录不再作为历史证据引用，V6清洁Smoke已重新运行并通过。

Stage 031当时放行结论：V6达到“可手动正式训练准备”状态且该Stage未启动训练。后续状态由Stage 032覆盖：用户已手动完成Gate、DBB、CU三模型正式训练，并已执行同一validation split的组件和联合级联复核；test、开放集校准、正式消融、多seed和P95仍未执行。

## 2026-08-06 - Stage 032 V6性能归因与修复启动

状态：用户要求“开始修复”。V6正式checkpoint和配置保持只读；V7/V8手术式代码、配置、focused测试、CPU有界Smoke和CUDA有界Smoke均已完成。该阶段结束时正式训练尚未执行；后续用户已手动完成V7/V8六项正式训练和训练内组件validation。三checkpoint联合validation、test和OOF校准仍未执行。

只读输入与实际命令范围：

- 读取三个V6 artifact目录中的`best_model.pt`、`training-summary.json`、`training.log`和`sample_detail_information.json`；checkpoint epoch为Gate=8、DBB=13、CU=7，三个checkpoint绑定相同index/contract/stats SHA-256。
- 使用`.venv`的PyTorch 2.5.1+cu121在CUDA上执行一次不落盘的V6 validation三checkpoint联合前向；设备为RTX 4060 Laptop GPU。没有创建或覆盖evaluation artifact。
- 读取V3 index并按split/类别统计parent、事件、序列长度、`tr_dt_valid`和RD缺失；读取V6配置、模型、训练、级联、stable和报告代码。
- 在同一validation上执行仅用于可分性判断的阈值/偏置诊断、ECE/NLL和0.60最大概率诊断；阈值未持久化、未写配置、未视为可部署校准。
- 读取`data/`、`config/RADAR-LN3/`和本文件确认正式数据输入最高仍为V6；V7/V8为本次新增模型/训练配置版本，复用V3索引、V6数据契约和V2统计，不创建新数据版本。

已确认V6结果：

- Gate parent：Accuracy=`0.913793`、Macro-F1=`0.808581`；DBB P/R=`0.984615/0.918660`，CU P/R=`0.540541/0.869565`。
- DBB specialist parent：Accuracy=`0.942584`、Macro-F1=`0.942809`；Drone P/R=`0.989583/0.904762`，Bird=`0.838710/1.0`，Balloon=`0.980392/0.961538`。
- CU specialist parent：Accuracy=`0.565217`、Macro-F1=`0.564394`；Clutter P/R=`0.4375/0.875`，Unknown=`0.857143/0.4`；support仅8/15。
- 联合五分类parent：Accuracy=`0.836207`、Macro Precision=`0.666740`、Macro Recall=`0.714927`、Macro-F1=`0.680308`；stable Macro-F1=`0.666504`且coverage=`1.0`，没有改善最终结果。
- test、真实开放集样本、OOF拒识校准、多seed和端到端P95：未执行。

关键归因：

- validation的无效dt事件比例在Drone/Bird/Balloon为`39.07%/49.56%/47.33%`，在Clutter/Unknown仅为`5.37%/5.49%`；当前`tr_dt_valid`既进入融合特征又控制液态更新，构成有证据的采集域捷径风险。
- Gate class weight为`[0.55379,5.14762]`，CU相对DBB约9.29倍；CU权重为`[1.5,0.75]`。同一validation乐观阈值仍不能达到Gate 0.95或CU 0.90的双类P/R下限，不能只靠阈值修复。
- DBB第12轮的parent最差P/R=`0.903846`，而按Macro-F1选中的第13轮最差P/R=`0.838710`；当前checkpoint选择与逐类下限验收不一致，但所有epoch仍未达到0.95。
- 三个不同class-weight训练的softmax未经温度/先验校准直接相乘；DBB specialist的Bird Recall从条件评估`1.0`降到联合级联`0.769231`。
- 液态tau中位数仍约15秒，接近初始化；不能把当前失败归因为液态容量不足。RD前端全局空间池化与点内masked mean是后续单因素表征候选，本Stage不与V7/V8首要修复混改。

版本与修改边界：V6配置hash保持Gate=`5c2676b...bdd63e0`、DBB=`3f07df8...aef`、CU=`ab28dae...87fd2`，V6 contract=`2110938...8777`。V7只改变`dt_valid`分类特征暴露语义；V8在V7基础上只改变checkpoint选择语义。V3 index和V2统计按原hash复用，V6配置/checkpoint/artifact均未修改。拒识代码只建立未校准保护接口，不生成正式阈值配置。

实际代码修改：

- `src-v3/radar_ln3/task_models.py`增加`dt_valid_feature_mode`。V6缺省归一化为`fusion_and_dynamics`并通过真实三份V6 checkpoint CPU加载；V7及以上必须显式为`dynamics_only`。V7融合输入从80维降至79维，batch和online路径均不再把`tr_dt_valid`拼入事件特征，但仍将其传给液态更新。
- `src-v3/radar_ln3/cascade.py`把`dt_valid_feature_mode`加入三模型共享契约；在线结果增加与五分类标签正交的`decision_state/rejected/rejection_score/rejection_calibration_source`；离线报告明确拒识接口存在但未校准、未启用且`supports_validated_rejection=false`。
- `src-v3/train/tasks.py`增加可审计选优器。V8按`(parent最弱类别min(Precision,Recall), parent Macro-F1)`字典序选优；零support类别直接拒绝；history、checkpoint metrics、summary和日志同时记录主指标与并列指标。V7仍使用`parent_macro_f1`。
- 新建`src-v3/radar_ln3/rejection.py`。`UnknownTarget`保持普通五分类标签；没有`train_parent_grouped_oof`来源的校准契约时不能启用reject，默认输出`decision_state=not_calibrated`和`rejected=null`。
- `src-v3/report/audit.py`修正报告语义，区分“接口存在”和“已验证拒识能力不存在”。
- 新建`src-v3/smoke/v7_v8_repair_contract_test.py`并修正旧级联测试的V7内存占位，使其声明`dynamics_only`。
- `src-v3/train/tasks.py`在best validation确定后复用统一日志模块，输出parent/window总体和逐类P/R/F1，只打印置信度低于0.60的样本；全量window/parent记录仍只保存到`sample_detail_information.json`。
- 新建`src-v3/smoke/v7_v8_cuda_smoke.py`作为唯一六配置CUDA入口，明确禁止CPU fallback；CUDA结果使用`*-contract-smoke-cuda/`目录，不覆盖现有CPU Smoke证据。

创建配置与SHA-256：

- `config/RADAR-LN3/RADAR-LN3-V7-cq08-liquid-dbb-cu-gate-config.json`：`1589926d8a288fa57ad27234e9dfe480510f96fe613c082eaaa8fe3c5d7251fd`。
- `config/RADAR-LN3/RADAR-LN3-V7-cq08-liquid-dbb-specialist-config.json`：`066453b9eb5f7b56875e58704380c8031764b923b137a0f113bc9914088a606c`。
- `config/RADAR-LN3/RADAR-LN3-V7-cq08-liquid-cu-specialist-config.json`：`8d56adb116e827fafa35bc07d0a0e2dae2350479bc7f68095fd9deed6f0cd322`。
- `config/RADAR-LN3/RADAR-LN3-V8-cq08-liquid-dbb-cu-gate-config.json`：`a7c415adc7deeed3b07797d52404e7c36d89e58538d1651dc3b318549a6c71b2`。
- `config/RADAR-LN3/RADAR-LN3-V8-cq08-liquid-dbb-specialist-config.json`：`e671bca3b0537c5b5138f5de515da37b09f21f89e4af618efb633a1857c47c06`。
- `config/RADAR-LN3/RADAR-LN3-V8-cq08-liquid-cu-specialist-config.json`：`17a5a9ca021191d5b4d768bcfb158bb45a01a4f62e7b8431e29ef0f80a7824ae`。
- 六份配置均复用V3 index SHA-256=`a5d054f044709047938c88ef5601caf7d4b482ef980a5a61f19db87b6e649f87`、V6 contract SHA-256=`2110938d1d94923edb44c2825e34fb595fb1b34c1353f3f3e512403f91ed8777`和V2 train-only stats SHA-256=`584eddcaeb8d78677f0f4902d622c2aa313d9374b767a47982c12dc058f826b6`。未创建或修改index、manifest、split、数据契约、统计或RD张量。
- V7/V8六个正式artifact目标在Stage 032修改前和Smoke结束时均不存在，未覆盖V6或其他正式产物；后续用户手动训练后六个正式artifact均已生成。

实际验证结果：

- V6三份真实best checkpoint在CPU上按旧配置加载通过；旧模型模式=`fusion_and_dynamics`、融合维度=80。
- `smoke.v7_v8_repair_contract_test`：passed；固定seed=`20260806`后连续两次结果一致。V7融合维度=79；只翻转`dt_valid`时直接融合张量最大差=`0.0`，同一事件液态状态最大差=`0.054695725440979004`，证明它只保留动力学作用；级联batch/online最大差=`8.940696716308594e-08`；V6 checkpoint→V7拒绝、V7/V8混合级联拒绝；弱类选优、Macro-F1并列、零support拒绝、未校准reject禁用和Unknown标签保留均通过。测试常量0.60仅用于接口单测，不是正式阈值。
- V6冻结兼容探针：三份真实V6 best checkpoint加载通过；以V6配置保存临时checkpoint时`model_contract`键集合与历史V6一致且不持久化V7字段`dt_valid_feature_mode`。V7及以上checkpoint才要求并持久化该字段。
- `smoke.causal_cascade_contract_test`回归：passed；V6因果质量、stable、checkpoint与旧消融契约未回退。
- `compileall -q`：passed；`smoke.static_isolation_check`：passed，`violations=[]`；训练、任务报告、级联报告和审计CLI导入检查均退出码0。
- `smoke.logging_test`：passed；`per_class_metrics_logged=true`、`low_confidence_only=true`，0.59置信度探针被打印，0.95探针未打印，logger handler无重复。
- `report.audit`仅支持V3五分类配置；尝试传入V7任务配置和V4配置分别按预期报错“仅支持V0-V4”和“requires V3”，均未生成目标文件。随后使用其唯一支持的V3配置执行只读审计通过，拒识元数据为`interface_available=true`、`reject_or_abstain_output=false`、`supports_validated_rejection=false`、`required_calibration_source=train_parent_grouped_oof`；报告写入`src-v3/smoke/RADAR-LN3-V7-rejection-metadata-audit.json`。该报告不代表V7模型评估。
- V7/V8 Gate、DBB、CU共六项真实CQ08 CPU有界Smoke均passed，parent_limit=6，完成forward、causal-prefix loss、梯度更新和checkpoint round-trip；Gate/DBB输入TR=`[2,22,14]`、RD=`[2,22,12,2,21,384]`，CU输入TR=`[2,15,14]`、RD=`[2,15,12,2,21,384]`。V7/V8对应loss分别相同：Gate=`0.1994821727`、DBB=`0.3635533154`、CU=`0.9379934072`；gradient norm分别=`3.5527384281/5.4473567009/9.3607339859`。结果和临时checkpoint只位于`src-v3/smoke/RADAR-LN3-V7-*`与`RADAR-LN3-V8-*`。
- CUDA Smoke实际结果：`src-v3/smoke/RADAR-LN3-V7-V8-cuda-smoke-result.json`，总状态=`passed`，设备=`NVIDIA GeForce RTX 4060 Laptop GPU`，parent_limit=`6`。V7/V8 Gate、DBB specialist、CU specialist六项均为`status=passed`；全部完成真实CQ08 Dataset读取、forward、causal-prefix loss、梯度更新和checkpoint round-trip。六项均复用index SHA-256=`a5d054f044709047938c88ef5601caf7d4b482ef980a5a61f19db87b6e649f87`、contract SHA-256=`2110938d1d94923edb44c2825e34fb595fb1b34c1353f3f3e512403f91ed8777`、stats SHA-256=`584eddcaeb8d78677f0f4902d622c2aa313d9374b767a47982c12dc058f826b6`。Gate loss=`0.1994822`、DBB loss=`0.3635533`、CU loss=`0.9379933`；对应梯度范数约为`3.55274/5.44736/9.36073`。这些是smoke数值，不是分类性能。
- CUDA smoke产物仅位于`src-v3/smoke/RADAR-LN3-V7-*contract-smoke-cuda/`、`src-v3/smoke/RADAR-LN3-V8-*contract-smoke-cuda/`和汇总JSON；该句描述Smoke结束时状态。后续正式训练artifact与Smoke目录保持隔离并已存在。

后续状态校正：V7/V8六项用户手动paired正式训练已完成，六份training-summary与sample detail已读取；V7/V8三checkpoint联合评估、train parent-grouped OOF校准、test、多seed、P95和生产切换仍未执行。本代理未自动启动正式训练。

## 2026-08-07 - Stage 033 DBB/CU因果运动与RD物理特征审计

状态：用户确认“先验证特征问题”。本阶段只读分析CQ08 V3索引、现有物理RD张量、V7/V8训练摘要与V7 Gate sample detail；未修改模型、数据、配置、split、checkpoint或正式artifact。

读取文件与契约：

- `data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-index.csv`，SHA-256=`a5d054f044709047938c88ef5601caf7d4b482ef980a5a61f19db87b6e649f87`。
- `data/processed/RADAR-LN3-V6-cq08-tr-rd-causal-quality-data-contract.json`，SHA-256=`2110938d1d94923edb44c2825e34fb595fb1b34c1353f3f3e512403f91ed8777`。
- `data/processed/RADAR-LN3-V2-cq08-tr-rd-train-normalization-stats.json`，SHA-256=`584eddcaeb8d78677f0f4902d622c2aa313d9374b767a47982c12dc058f826b6`。
- `src-v3/radar_ln3/features.py`、`data.py`、`model.py`、`task_models.py`；V7/V8 Gate、DBB、CU的六份training-summary与sample detail。
- `data/processed/cq08_rd_physical_canonical_v3_1/*.npz`：1549个文件、约2.77 GB；每个张量含amplitude、phase、time、`velocity_mps[-85,85]`和`range_offset_m[-75,75]`。

实际命令范围：

- 使用`Get-Content`、`rg`、`Get-ChildItem`和`Get-FileHash -Algorithm SHA256`核对代码、字段、物理坐标和文件hash。
- 使用`.venv\\Scripts\\python.exe -`运行不落盘的NumPy/Pandas/PyTorch统计脚本；train拟合、validation评估，最终报告不使用test样本或test模型输出。
- 首次RD统计因执行上限终止；第二次因部分parent首事件无RD导致路径为`nan`而失败。两次都未生成文件或修改状态。最终实现按首个`rd_present=true`事件打开同一parent张量并显式保留RD覆盖率，成功处理确定性分层train 195个parent和完整validation 232个parent。

TR数据实测：

- 最终运动分布只统计train+validation的1313个parent：Drone=593、Bird=295、Balloon=297、Clutter=43、Unknown=85；test排除。
- parent速度中位数：Drone=`13.047`、Bird=`9.545`、Balloon=`5.757`、Clutter=`4.893`、Unknown=`8.467 m/s`。Clutter不是静止类，Unknown与Bird在速度和径向速度上明显重叠。
- Clutter的转向率中位数=`12.436 deg/s`、绝对径向加速度中位数=`0.993 m/s²`，高于Drone=`1.142/0.109`、Bird=`2.637/0.360`、Balloon=`2.188/0.110`；其表现是高抖动伪运动，不是简单无运动。
- 路径直线度中位数：Drone=`0.876`、Bird=`0.842`、Balloon=`0.774`、Clutter=`0.285`、Unknown=`0.902`；径向速度符号翻转率中位数仅Clutter显著升高到`0.342`，其余四类为`0`。这支持以运动一致性而非速度阈值区分Clutter。
- 首事件speed/course同时精确为0的train+validation比例：Drone=`24.79%`、Bird=`28.14%`、Balloon=`20.20%`、Clutter=`62.79%`、Unknown=`80.00%`，同时首点径向速度中位数非零；该字段具有初始化/采集捷径风险，不能直接解释为真实静止。
- 单变量validation排序能力：完整parent的`dt_valid_fraction`方向化AUC=`0.7536`、duration AUC=`0.7287`、首点speed/course零值AUC=`0.6993`。组合采集质量探针AUC=`0.9276`，说明高分类信号中存在非物理捷径。

轻量TR探针：

- 仅用于信息存在性判断，使用train拟合、完整232个validation parent评估；不保存阈值、不作为模型性能。
- 五点前缀可解释TR组合AUC=`0.9530`，DBB P/R=`0.9894/0.8947`，但Bird正确路由仅`0.7500`。
- 完整parent可解释TR组合AUC=`0.9678`，DBB P/R=`0.9895/0.8995`，Bird正确路由仍仅`0.7500`；单纯运动水平/变化组合AUC=`0.8883`，DBB Recall=`0.8517`、CU specificity=`0.7391`。高AUC不是简单“DBB会动、CU不动”。
- V7 Gate中17个Bird→CU的速度中位数=`8.398 m/s`，不是无运动；其转向变化中位数=`6.395°/event`，高于35个正确Bird的`2.179°/event`。错误Bird的dt有效率中位数=`1.0`，正确Bird仅`0.214`，表明采集节奏仍与决策相关。

RD物理统计：

- 确定性分层train子集为Drone/Bird/Balloon各30、Clutter全部35、Unknown全部70，共195个parent；validation完整232个parent。每个TR事件只读取V3映射且最多12帧；test未用。
- 全部物理RD统计五点前缀validation AUC=`0.8332`，完整parentAUC=`0.8273`；该小样本线性探针不能证明正式模型收益。
- V7正确Bird与Bird→CU在RD上存在明显形态差异：速度带宽中位数=`41.985 vs 21.613 m/s`，零速±2 m/s能量占比=`0.204 vs 0.650`，速度谱熵=`0.894 vs 0.554`，top10速度能量集中度=`0.370 vs 0.780`。错误Bird更像低多普勒、窄谱、集中目标，这解释了其与CU竞争，但不等价于标签错误。
- validation Clutter/Unknown的RD峰值速度中位数均接近最小非零bin `0.222 m/s`，零速能量中位数=`0.640/0.649`；Bird总体为`0.350`。物理坐标和零多普勒结构有价值，但当前`RDSpatialEncoder`经过大步长卷积、全局池化和点内masked mean，不显式保留这些量或帧序。

本轮结论与边界：

- 已确认：Clutter在CQ08航迹中不是“无运动类”，而是路径不连贯、转向与径向变化大的伪运动类；简单DBB速度接受器不成立。
- 已确认：Bird→CU并非因Bird没有运动，而与更低多普勒、更窄谱、更高零速能量以及采集dt模式相关。
- 已确认：当前14维TR输入没有显式速度差、径向速度差、路径一致性或转向率；当前RD输入虽含完整物理图，但前端不显式消费速度/距离坐标并丢失点内帧序。
- 有证据但待模型验证：增加因果运动一致性特征或当前事件RD物理统计可能改善Gate；线性探针和validation分布不能写成性能提升。
- Unknown开放集样本、V7/V8联合级联、test、多seed、OOF校准和P95均未执行。

创建/修改文件：

- 创建`src-v3/processed/report/RADAR-LN3-V7-V8-dbb-cu-feature-audit.md`。
- 覆盖式更新`src-v3/task_plan_v3.md`、`src-v3/process_v3.md`和`src-v3/findings_v3.md`，并修正Stage 032的V7/V8正式训练过期状态。
- `data/`、`config/RADAR-LN3/`、模型代码、训练代码和正式artifact：未修改。

当前状态：特征问题已完成只读验证；尚未授权或实施新特征schema。下一阶段必须只选择一个主要变量，先通过shape/mask/因果与Smoke，再由用户手动训练。

## 2026-08-07 - Stage 034 V9因果TR运动一致性Gate实施准备

状态：用户回复“继续”，按Stage 033证据完成V9因果TR运动一致性Gate单因素实现、契约、配置、全量特征审计、focused测试、历史回归和CPU/CUDA有界Smoke。该段记录是正式训练前过程；用户随后于2026-08-07手动完成V9 Gate正式训练，真实性能validation与归因结果见Stage 035。test、多seed、OOF校准、P95和正式级联仍未执行。

只读检查与实际命令：

- 执行`git status --short`；确认工作区含大量用户既有改动，`src-v3/`和`config/RADAR-LN3/`仍整体未跟踪。本阶段只允许触碰V9所需文件和三个跟踪文件，不处理无关改动。
- 使用`rg`、`Get-ChildItem`和`Get-Content`读取Stage 033记录、`src-v3/radar_ln3/features.py`、`data.py`、`task_models.py`、V8 Gate配置、V3数据契约及现有测试入口。
- 核对`config/RADAR-LN3/`最大全局版本为V8；`config/RADAR-LN3/`、`data/processed/`和`artifacts/`均未发现V9目标文件或目录，无覆盖冲突。
- 复核V3 index SHA-256=`a5d054f044709047938c88ef5601caf7d4b482ef980a5a61f19db87b6e649f87`；V2 train-only stats SHA-256=`584eddcaeb8d78677f0f4902d622c2aa313d9374b767a47982c12dc058f826b6`。

冻结范围：

- baseline固定为V8 DBB/CU Gate；V9只增加因果TR运动一致性特征和逐特征valid mask。
- DBB/CU二分类标签、CQ08 V3 index/split、RD输入与前端、液态单元、loss、class weight、seed、epoch、early stopping和弱类checkpoint选择保持不变。
- V8 DBB与CU specialist不修改；V9 Gate在完成正式paired验证和兼容决策前不得与V8 specialists宣称为正式级联。
- valid mask采用`gating_only`语义：用于阻止无效运动值进入编码器，不直接作为分类特征；不得以epsilon伪造有效时间间隔。
- Unknown保持闭集类别语义；reject/abstain仍需train parent-grouped OOF校准，本阶段不实现或启用拒识阈值。

版本判断：输入特征schema和checkpoint模型契约发生实质变化，因此使用下一个全局版本V9；V3 index和V2统计内容/语义不变，复用原文件而未复制或无意义递增。

实际创建/修改：

- `src-v3/radar_ln3/features.py`：新增`causal-tr-motion-v1`、9维物理特征、固定缩放、环状角差和`CausalTRMotionState`；不读取标签、ID、路径、split、RD或未来点。
- `src-v3/radar_ln3/data.py`：V9按parent在线递推`tr_motion_features`和同形状bool mask；V8及更早配置默认motion维度为0，不增加计算成本。
- `src-v3/radar_ln3/task_models.py`：仅V9 Gate启用无偏置motion encoder；mask只把无效值归零，不拼入分类输入；训练、评估、在线单点和checkpoint契约均接入，旧V8 checkpoint严格拒绝。
- `src-v3/radar_ln3/cascade.py`与`src-v3/train/tasks.py`：扩展共享契约和正式训练张量传递；V9 Gate仍不能与V8 specialists构成声明为正式的混版本级联。
- `src-v3/build/motion_gate_contract.py`：生成器核对Stage 034、全局版本、V8 baseline配置hash、V3 index/V6 contract/V2 stats hash；目标已存在且内容不同会拒绝覆盖，重复运行结果为`reused_identical`。
- `src-v3/smoke/motion_feature_contract_test.py`：focused因果、mask、在线/离线、V8回归、checkpoint隔离测试。
- `src-v3/report/motion_features.py`：只读统计CQ08 train/validation运动特征分布，不读取test、不拟合阈值或分类器。
- 创建`data/processed/RADAR-LN3-V9-cq08-tr-rd-causal-motion-gate-data-contract.json`，SHA-256=`a7cdee8696983006bdfa7dd8d1c1d0a69684bd59a6ca88cf742e2d7d9f30c074`。
- 创建`config/RADAR-LN3/RADAR-LN3-V9-cq08-liquid-dbb-cu-gate-motion-config.json`，SHA-256=`ac2722b810a4557a6e72ddfbbd956393783b0ca1c4154281d01f78af93fda7a3`。
- 复用V3 index SHA-256=`a5d054f044709047938c88ef5601caf7d4b482ef980a5a61f19db87b6e649f87`、V6 causal-quality contract SHA-256=`2110938d1d94923edb44c2825e34fb595fb1b34c1353f3f3e512403f91ed8777`和V2 train-only stats SHA-256=`584eddcaeb8d78677f0f4902d622c2aa313d9374b767a47982c12dc058f826b6`；V3 index、split和RD张量未重建或修改。

运动特征冻结：

- 9维依次为报告速度加速度、径向加速度、环状航向转速、雷达球坐标三维几何速度、报告/几何速度绝对误差、距离变化率、径向/距离变化率绝对误差、径向符号翻转、在线累计路径直线度。
- 所有dt导数量要求`tr_dt_valid=true`且`dt>0`；无效值严格为0，不用epsilon或固定间隔伪造物理导数；padding mask全false。
- CQ08首点`speed/course`初始化语义未确认，V9不使用首点这两个字段作为第二点导数基线；报告速度和航向变化从后续已观察基线开始。
- 位置由range/azimuth/elevation转换为雷达局部三维坐标，不依赖普遍缺失的latitude/longitude/height；未来终点、完整parent时长、dt有效率和全局路径统计均禁止进入输入。
- train-only 12987个有效相邻对的径向符号审计：`radial_speed`与`delta_range/dt`同号率=`0.9321629`；同号绝对误差中位数=`0.8483202 m/s`，反号公式=`10.2650375 m/s`，因此冻结同号约定。

实际验证：

- 生成器重复运行：`contract_status=reused_identical`、`config_status=reused_identical`；没有无意义生成新版本。
- `report.motion_features`：passed；train+validation=`1313 parent/25158事件`，test排除。所有值有限、无效值严格为0、1313个首事件均无有效motion，跨scope最大裁剪率=`0.0008097166`。报告为`src-v3/processed/report/RADAR-LN3-V9-causal-tr-motion-feature-audit.json`，SHA-256=`ceb3f0b492ae570dfa9647f67456d1d30697578669a8f62dd5021a2a6b53de96`。
- 全量审计同时确认残余捷径：train/validation各类motion有效率范围差约`0.4356/0.4419`；validation DBB/CU几何运动有效率=`0.5624/0.9455`。mask虽不拼入模型，但零门控位置仍可能暴露采集模式，后续收益必须做运动值置零/扰动归因。
- `smoke.motion_feature_contract_test`：passed；单点、双点、359→0、重复/负/无效dt、missing、极短/长间隔、未来前缀不变、状态重置、真实CQ08 shape/mask、原始点在线状态与Dataset一致均通过；V9在线/离线prefix logits最大差=`4.1723251e-07`；V9前版本配置启用motion、缺失motion、越界motion、V8→V9和伪造schema的V8→V9均拒绝；真实V8 best checkpoint回归通过；结果SHA-256=`0dafa7acbf320d38a78c62411b515a9fbe233bde3b3a3ac5365aa77f9cbcd7ca`。
- `smoke.v7_v8_repair_contract_test`、`smoke.causal_cascade_contract_test`、compileall和静态隔离均passed；隔离`violations=[]`。
- CPU有界Smoke：passed；真实CQ08平衡6 parent，TR=`[2,22,14]`、motion=`[2,22,9]`、RD=`[2,22,12,2,21,384]`，loss=`0.1907315850`、总梯度范数=`4.1778927`、motion梯度范数=`0.09106856`、checkpoint round-trip通过；结果SHA-256=`1214498660c073819e4eac6b3f3872dcac1c5aa71acfbf82be88f427d0660bc0`。
- CUDA有界Smoke：passed，设备=`NVIDIA GeForce RTX 4060 Laptop GPU`；同一输入loss=`0.1907315999`、总梯度范数=`4.1778941`、motion梯度范数=`0.09106858`、checkpoint round-trip通过；结果SHA-256=`9747860bfdc025d6e2039be58905ef4aa08c8a6cb3fd6bd480d403cb5e13d39c`。Smoke数值不是分类性能。
- 一次CPU Smoke命令因`--output-suffix -cpu`被argparse解释为缺少值而在模型执行前退出；未生成产物。改为`--output-suffix=-cpu`后通过，失败不作为模型证据。

当前放行（历史训练前判断）：V9 Gate当时可以进入用户手动paired正式训练；预训练不需要且未使用。该放行已由后续正式训练完成状态取代；Stage 035确认V9仍不能与V8 specialists称为正式级联，不能启用拒识，也不能进入test/生产部署。

训练readiness：`src-v3/smoke/RADAR-LN3-V9-gate-training-readiness.json`，当前SHA-256=`f3840c0da52cbdd0499f97af63f418b6dbb124c35bedeb7ff4c30584aba45c32`；它是训练前历史快照，内部正式训练=`未执行`、validation性能=`未评估`不再代表当前状态，本轮保留原文件而不追写历史快照。

未执行（该阶段结束时的历史状态）：V9正式训练、真实性能validation、运动值置零/扰动归因当时未执行；前三项已在用户训练和Stage 035完成。test、多seed、train parent-grouped OOF校准、正式三checkpoint级联、P95和生产切换至今仍未执行。

## 2026-08-07 - Stage 035 V9 Gate正式结果与motion归因审计

状态：用户提供`artifacts/RADAR-LN3-V9-cq08-liquid-dbb-cu-gate-motion-formal/best_model.pt`并要求继续。完成正式产物核验、V8/V9 CUDA validation精确复现、基础五类路由与分层、全零/train中位数/跨parent三类post-hoc干预、时间来源追溯、报告入口验证和跟踪文件覆盖更新。未修改训练主链路、数据、配置或checkpoint。

只读检查：

- 执行`git status --short`；确认工作树仍有大量用户既有改动和未跟踪文件。本阶段仅创建`src-v3/report/motion_gate_attribution.py`、`src-v3/processed/report/RADAR-LN3-V9-motion-gate-attribution.{json,md}`并更新三个跟踪文件，不处理无关改动。
- V9正式artifact包含`best_model.pt`、`last_model.pt`、`training-summary.json`、`training.log`、`sample_detail_information.json`和`evaluation-val.json`；训练设备CUDA、seed=42、batch=2、最佳epoch=7、best global step=3787、总训练6492 step，因early stopping结束；预训练不需要且未使用。
- V9 best checkpoint SHA-256=`e6242ef9aced9b75e3cef80fa4f0a078fcf08b01e81755dfd1874ff07b5f2211`；V8 best checkpoint SHA-256=`7b6f7ca81e213d765e75aff40ebfc54a0532aaaa0653857f09869e1b5dc918bd`。
- V9配置SHA-256=`ac2722b810a4557a6e72ddfbbd956393783b0ca1c4154281d01f78af93fda7a3`；checkpoint严格绑定V3 index=`a5d054f044709047938c88ef5601caf7d4b482ef980a5a61f19db87b6e649f87`、V9 contract=`a7cdee8696983006bdfa7dd8d1c1d0a69684bd59a6ca88cf742e2d7d9f30c074`和V2 train-only stats=`584eddcaeb8d78677f0f4902d622c2aa313d9374b767a47982c12dc058f826b6`。
- 静态核对确认V8/V9的标签、CQ08 V3行/split、RD输入/前端、液态单元、hidden维度、loss、class weight、seed、batch、学习率、epoch、early stopping和checkpoint选择相同；V9唯一新增模型输入因素为9维因果TR motion及其gating-only valid mask。结构新增会改变新增层和下游融合层随机初始化，单seed比较仍不是统计因果证明。

实际创建：

- `src-v3/report/motion_gate_attribution.py`：只读validation报告入口；不读取test、不拟合阈值、不更新模型参数。入口输出window/parent指标、基础五类路由、长度/dt/motion有效率/置信度分层、逐parent变化和三类输入干预。
- `src-v3/processed/report/RADAR-LN3-V9-motion-gate-attribution.json`：完整机器可读报告，SHA-256=`5b4cb2e52cb3f45f5f9c5fde8a7598868764cd56d83cc97bad180fb4b75c01e8`。
- `src-v3/processed/report/RADAR-LN3-V9-motion-gate-attribution.md`：精简指标与解释边界，SHA-256=`4d67bd4218b9adb01efb78688ade52d23b4184d02e6e99e0eea99769569faa4a`。
- `src-v3/report/motion_gate_attribution.py`：最终SHA-256=`ba313f44efafeefd69f008a8e150a3b4942535ba0ed2d0b7d683b9bf3bf15ad9`；最终清理仅删除未使用import，不改变已执行诊断语义。
- V9训练前readiness JSON保留为历史快照，没有为了匹配训练后状态篡改；其SHA-256仍为`f3840c0da52cbdd0499f97af63f418b6dbb124c35bedeb7ff4c30584aba45c32`。

CUDA validation复现：

- 实际命令：在`src-v3`执行`..\\.venv\\Scripts\\python.exe -m report.motion_gate_attribution --project-root F:\\Project\\RadarInsight --device cuda --batch-size 2`。最终完整运行退出码0，232 parent/4474 causal prefix window，直接V8/V9矩阵和总体标量与各自best checkpoint内嵌validation逐值一致，绝对差均为0。
- V8 parent：矩阵=`[[192,17],[9,14]]`，Accuracy=`0.887931`，Macro-F1=`0.727552`；DBB P/R/F1=`0.955224/0.918660/0.936585`，CU=`0.451613/0.608696/0.518519`。
- V9 parent：矩阵=`[[198,11],[6,17]]`，Accuracy=`0.926724`，Macro-F1=`0.812752`；DBB P/R/F1=`0.970588/0.947368/0.958838`，CU=`0.607143/0.739130/0.666667`。相对V8 corrected=11、regressed=2、预测翻转13；但DBB Recall仍低于0.95，CU P/R远低于0.90。
- V8→V9 parent基础类路由：Drone→CU `4→0`、Bird→CU `11→9`、Balloon→CU `2→2`、Clutter→DBB `2→1`、Unknown→DBB `7→5`。
- V8/V9 window矩阵=`[[3645,407],[109,313]]→[[3836,216],[119,303]]`，Macro-F1=`0.741029→0.801079`；Drone/Bird/Balloon错误prefix分别`96→35/255→147/56→34`，但Clutter→DBB prefix `14→25`，Unknown→DBB `95→94`。因此目标最终结果改善不能掩盖在线临时Clutter Recall退化。
- validation CU只有23 parent（Clutter=8、Unknown=15），单个parent会显著改变CU指标；V9 epoch间CU Precision=`0.337–0.607`、Recall=`0.739–0.957`，波动明显。test和多seed均未执行。

motion归因：

- 全零：将9维motion所有值置零并保留原mask仅用于合法性检查；parent最终全部预测DBB，矩阵=`[[209,0],[23,0]]`、Macro-F1=`0.473923`、CU F1=0。这证明V9已依赖motion分支决定CU，但不能证明依赖的是物理数值。
- train中位数模板：使用既有train-only motion审计中每个特征有效值中位数；保留validation每个接收事件/特征的原valid位置，invalid/padding仍严格为0，完全移除目标自身motion数值。结果矩阵=`[[200,9],[9,14]]`、Macro-F1=`0.782817`、CU F1=`0.608696`；分别保留V8→V9 Macro-F1和CU F1增益约64.86%/60.87%。
- 跨parent扰动：3个固定seed（901/902/903），逐特征保持接收方valid位置和validation边际值集合，每个值强制来自不同parent，不使用标签。Parent Macro-F1=`0.783231/0.798261/0.798765`，CU F1=`0.612245/0.640000/0.638298`；均值保留V8→V9 Macro-F1增益约77.31%、CU F1增益约75.37%。
- 在唯一同时包含DBB/CU的motion有效率0.75–1.00子集（127 parent）中，V8/V9 Macro-F1=`0.715/0.787`；train中位数模板=`0.756`，跨parent扰动=`0.757–0.779`，同样保留大部分收益，不是低有效率纯DBB子集单独造成的统计假象。
- 解释边界：全零/模板/扰动都是同checkpoint的post-hoc输入干预，不等价于重训消融；跨parent扰动破坏时间连续性和跨特征物理一致性，属于OOD干预。尽管如此，真实值被模板完全移除后仍保留大部分增益，是“有效位置/零模式主导”的强证据；真实数值从模板/扰动到原始V9仍提供剩余改善，不能写成完全无效。

根因追溯：

- `features.py`中所有9维motion均要求`dt_valid=true and dt>0`；无效值为0。`task_models.py`的motion encoder为无偏置Linear→无仿射LayerNorm→SiLU→无偏置Linear；全零严格映射为零，任何非零向量经LayerNorm被标准化，因此构成强隐式“motion可用”开关。
- validation按基础类的非首帧`zero_delta_at_second_precision`计数为Drone=687、Bird=450、Balloon=427、Clutter=0、Unknown=0；负dt和非有限dt均为0。Clutter/Unknown全部非首帧dt有效，DBB则存在大量秒级重复时间。
- parent级dt有效率均值Drone/Bird/Balloon=`0.63/0.54/0.58`，Clutter/Unknown=`1.00/1.00`；motion有效率近似相同。低于0.75有效率的105 validation parent全部属于DBB；23个CU全部位于0.75–1.00组。
- 已有12文件MATLAB原始回波审计确认原始`.dat`和现有MAT有帧顺序与CPI/PRT局部时长，但没有显式TR wall-clock；RD CPI时间不是TR观测间隔，不能用于恢复同秒TR点的真实dt。因此当前本地数据证据无法从原始回波恢复CQ08亚秒TR wall-clock。

验证与状态：

- `py_compile report/motion_gate_attribution.py`、入口`--help`、`python -m compileall -q .`均退出码0。
- `python -m smoke.static_isolation_check --root F:\\Project\\RadarInsight\\src-v3`通过，`violations=[]`。
- 第一次显式1秒timeout仅发生在模型评估启动阶段，退出码124、没有报告产物；改用完整CUDA命令后通过。加入train中位数模板后重新完整执行并覆盖旧诊断报告，最终报告只对应后一次成功运行。
- 数据重建、配置创建/修改、checkpoint修改、正式训练代码修改、test、多seed训练、OOF拒识校准、正式三checkpoint级联、stable和P95：未执行。

当前判断：V9保留为研究checkpoint，不作为正式软级联Gate候选。下一单因素应先消除/等化时间精度可用性捷径，再测试物理motion；在这一点修复前，不进入更复杂液态单元、V9混版本级联或开放集阈值设计。

## 2026-08-07 - Stage 036 V10无dt逐观测增量Gate开始

用户选择：用户明确选择方案A。V10将用不依赖`dt/dt_valid`的逐观测增量替换V9的物理率特征；本轮先实施代码、版本化契约/config、focused测试和有界Smoke，不自动正式训练。

本轮开始前只读核对：

- 执行`git status --short`；确认工作区存在大量既有用户改动和未跟踪实验文件。本Stage严格不修改`src/`、RD-only旁线或既有artifact。
- 枚举`config/RADAR-LN3/`与`data/processed/RADAR-LN3-V*`；最大已使用全局版本为V9。V10文件在开始时不存在。
- 读取V9配置、V9数据契约、`features.py`、`data.py`、`task_models.py`、`cascade.py`、`train/tasks.py`、V9 focused测试及三份跟踪文件。
- 复核V9配置SHA-256=`ac2722b810a4557a6e72ddfbbd956393783b0ca1c4154281d01f78af93fda7a3`、V9数据契约SHA-256=`a7cdee8696983006bdfa7dd8d1c1d0a69684bd59a6ca88cf742e2d7d9f30c074`；V3 index、V6因果质量契约和V2 train-only stats继续按Stage 034冻结hash只读复用。

冻结实施边界：

- 对照为V9 Gate；V10只替换9维派生motion语义及对应数据/模型/checkpoint身份，维度和motion encoder结构保持不变。
- 特征只可读取同一parent当前及历史TR观测值；不得读取或用`dt`、`dt_valid`、时间戳精度决定值或valid位置；不得读取标签、ID、文件路径、split、RD或未来点。
- CQ08 V3 index/split、基础TR/RD张量、7维当前事件RD质量、RD空间前端、CfC液态动力学及其原有`dt_valid`动态语义、causal-prefix loss、class weight、seed、训练预算和parent最弱P/R选优规则保持不变。V10修复的是motion支路捷径，不在本单因素中改变液态单元对dt的既有处理。
- 首点无前驱，逐观测增量必须无效且值为0；字段缺失由同形状valid mask数值门控，mask不拼入分类器；在线状态在parent/session/target边界reset。

实施结果：

- `src-v3/radar_ln3/features.py`新增`causal-tr-observation-increment-v1`及独立`CausalTRObservationIncrementState`。9维依次为报告速度增量、径向速度增量、环状航向增量、三维几何步长、距离增量、径向/距离增量方向一致性、径向符号翻转、连续步方向余弦和在线累计路径直线度；均为“每次观测变化”，不是SI每秒率。状态机`update(self, values)`在函数签名上不接收时间信息。
- `src-v3/radar_ln3/data.py`由配置schema选择V9 dt-rate状态机或V10 no-dt状态机；旧`enable_tr_motion=True`仍映射V9，保持历史调用兼容。`task_models.py`和`cascade.py`按实际schema解析特征名称、缩放、clip、Dataset和checkpoint契约，维度相同也不能跨schema加载。
- 创建`src-v3/build/observation_increment_gate.py`、`src-v3/smoke/observation_increment_contract_test.py`和`src-v3/report/observation_increments.py`；`smoke/task_dbb_cu.py`仅将写死V9的梯度错误消息改为通用Gate语义。
- 创建`data/processed/RADAR-LN3-V10-cq08-tr-rd-observation-increment-gate-data-contract.json`，SHA-256=`b7d905ae0f373882c17dddab7fb13785c2c6fe86016b0a20aac5d7d0755880af`；这是特征/模型输入契约实质变化，使用下一个全局版本V10。
- 创建`config/RADAR-LN3/RADAR-LN3-V10-cq08-liquid-dbb-cu-gate-observation-increment-config.json`，SHA-256=`6d43b927e275354ff5089ed8355aa660c438173d6c341fc185fefeb40d0472b1`；正式artifact路径为`artifacts/RADAR-LN3-V10-cq08-liquid-dbb-cu-gate-observation-increment-formal`，收口时目录不存在，无覆盖冲突。
- 复用V3 index SHA-256=`a5d054f044709047938c88ef5601caf7d4b482ef980a5a61f19db87b6e649f87`、V6因果RD质量契约SHA-256=`2110938d1d94923edb44c2825e34fb595fb1b34c1353f3f3e512403f91ed8777`和V2 train-only RD统计SHA-256=`584eddcaeb8d78677f0f4902d622c2aa313d9374b767a47982c12dc058f826b6`。数据范围、标签、split、index行和RD张量未变，因此未重建或复制这些文件。
- V10生成器首次执行返回`created`，第二次执行返回`reused_identical`；没有创建V11或覆盖不同内容的V10文件。

实际验证：

- `smoke.observation_increment_contract_test`通过，结果`src-v3/smoke/RADAR-LN3-V10-observation-increment-contract-result.json`，SHA-256=`63e1b78380fd0c8e70ba40115d6775787455e235b5731a1d32ec875e4ce7415d`。真实parent `cq08|track|1`的时间戳、dt、dt_valid和reason被显式改写后，9维值与valid mask逐位完全相同；在线/离线prefix logits最大差=`1.9371509552e-07`；真实V9 checkpoint回归加载通过，伪造schema/hash的V9→V10被模型/input契约拒绝，V10 checkpoint round-trip通过。
- 全量特征审计覆盖CQ08 train+validation `1313 parent/25158事件`，test排除；报告`src-v3/processed/report/RADAR-LN3-V10-causal-tr-observation-increment-audit.json`，SHA-256=`74556819391fe0dc6340911fab10b9eeac48feb2cabc24dc69e580db61f08cd0`。8179个非首帧`zero_delta_at_second_precision`事件全部仍产生至少一个合法逐观测增量；train/validation各类逐特征valid率最大差=`0.00574827/0.00722476`，相对V9 validation参考约`0.4418665`已消除主要可用性捷径；最大裁剪率=`0.00110072`，有限值、首点无效和invalid严格零通过。
- V10 CPU有界Smoke通过：真实CQ08平衡6 parent，TR=`[2,22,14]`、增量=`[2,22,9]`、RD=`[2,22,12,2,21,384]`；loss=`0.1814372540`、总梯度=`4.05536318`、增量分支梯度=`0.09354136`，参数更新和checkpoint round-trip通过。结果SHA-256=`5a6bd10fbf55a2bd70d73eb3d02c8cd198d2cefc948348a51ccfcb339399a871`。
- V10 CUDA有界Smoke通过：loss=`0.1814373136`、总梯度=`4.05536318`、增量分支梯度=`0.09354140`，结果SHA-256=`11bc2a1798af4477b5340c522398bc5507f143a78d5d5156b02e9337623ef8b0`。两套checkpoint均只在`src-v3/smoke/`，不是正式模型产物。
- V9完整focused回归、V7/V8修复契约和基础因果级联回归均通过；`python -m compileall -q .`通过；静态隔离通过，`violations=[]`。
- 创建`src-v3/smoke/RADAR-LN3-V10-gate-training-readiness.json`，SHA-256=`cc7331b260ba6957edac935b33befb4d117f43c46e2c5672b9f701a2032b0941`。结论仅为“可以进入用户手动V10 Gate paired正式训练”；不能进入正式级联、test、拒识或部署。

实际失败与修复：

- 一次组合预检因搜索V10目标无匹配而返回码1，编译与artifact预检随后拆分执行并通过；未写版本文件。
- 第一次V9 focused回归在模型已正确拒绝缺失motion的情况下，因旧测试只匹配错误消息短语`requires causal TR motion`而报失败；恢复兼容短语并保留新语义后完整V9测试通过。没有修改V9公式、配置、数据或checkpoint。

该阶段收口时的历史放行：可以由用户手动训练V10 Gate，并用同一V3 validation与V9 Gate做paired比较；预训练不需要且未使用。V10参数量仍为32522，除特征schema/checkpoint身份外模型维度与训练超参数未变。用户随后已完成训练，见Stage 037。

该阶段收口时未执行：V10正式训练和真实性能validation当时未执行，后续已由用户训练并在Stage 037复核；test、多seed、train parent-grouped OOF拒识校准、V10同版本specialists、正式三checkpoint级联、stable流式评估、P95和生产切换至今仍未执行。Smoke与特征审计没有被作为分类性能提升证据。

## 2026-08-07 - Stage 037 V10 Gate正式结果、paired复核与残差审计

用户输入：用户提供正式best checkpoint `artifacts/RADAR-LN3-V10-cq08-liquid-dbb-cu-gate-observation-increment-formal/best_model.pt`并要求继续。范围冻结为CQ08 V3 validation；不读取test、不启用reject、不运行正式级联、不拼接V8 specialists。

只读核验与正式artifact：

- V10 best checkpoint SHA-256=`2973806b740094d6380468d3c95bff4b70986e6128764f12cb4c07316197dec1`；配置/V10 contract/V3 index/V2 stats hash与checkpoint契约一致。
- 正式训练使用CUDA、seed=42、batch=2、1081 train parent、232 validation parent、20 epoch、10820 global step；best epoch=20，停止原因为epoch limit，未使用预训练。
- `training-summary.json` SHA-256=`c3f90749f36b19775305173e716b919c59a2b1d0600bd9d59e8a1a9690e70fa1`；`evaluation-val.json`=`d3d5ded582582bed32a944b1496ef0f8d0c23385bc531c604407d2a6f9fe4734`；`sample_detail_information.json`=`e8a75f55174fc22931be0c835e44c5774a4a3dfe6410cdc61012ff08046aba4a`；`training.log`=`e0e265f911dddad9163a8ed97326c9eb70a8eb304d158071dff7c2a5aec18688`。
- GPU独立validation复现232 parent/4474 causal-prefix window，指标和矩阵与best checkpoint训练记录逐值一致。

V9/V10 paired结果：

- 创建`src-v3/report/gate_paired_evaluation.py`并运行同record paired审计；V9/V10的4474个window和232个parent key、truth和不可变元数据一致，两个evaluation报告均被精确复现。
- Parent：V9→V10 Accuracy=`0.926724→0.948276`、Macro-F1=`0.812752→0.855211`；DBB P/R/F1=`0.970588/0.947368/0.958838→0.971292/0.971292/0.971292`；CU=`0.607143/0.739130/0.666667→0.739130/0.739130/0.739130`；矩阵=`[[198,11],[6,17]]→[[203,6],[6,17]]`。
- Window：Macro-F1=`0.801079→0.832280`；V10 DBB P/R/F1=`0.978393/0.949901/0.963937`，CU=`0.624074/0.798578/0.700624`；矩阵=`[[3849,203],[85,337]]`。
- V10 parent路由错误为Bird→CU 6、Unknown→DBB 6；Drone、Balloon和Clutter最终parent均无Gate错误。Window错误相对V9为Drone `35→11`、Bird `147→162`、Balloon `34→30`、Clutter `25→0`、Unknown `94→85`。V10改善最终Bird parent，但Bird早期prefix总体退化。
- Paired parent纠正9、退化4；window纠正125、退化78。V10达到parent DBB P/R≥0.95，但CU P/R均仅0.739130，正式Gate放行失败。
- 20个epoch没有一个同时达到DBB P/R≥0.95和CU P/R≥0.90。epoch 16的Parent Macro-F1最高0.878840，但CU P/R=`0.714286/0.869565`；epoch 20的四项弱指标下限最高0.739130。当前没有证据支持仅延长训练。
- Paired JSON SHA-256=`718fcf4ab4fccc48ae3c6c738b443f98ebf9984e9bd2cdfd476bea9460476e6d`；Markdown=`268bd82af2e85257796c683c882919499b0fe237385bf74dd092046be5e6093d`。

Bird/Unknown残差特征与结构完整性：

- 创建并运行`src-v3/report/gate_residual_features.py`。范围只含validation Bird/Unknown，组数为Bird正确46/错误6、Unknown正确9/错误6；每个事件只使用V3已映射的最多12帧RD，不拟合阈值、分类器或参数。
- Bird错误组相对正确组的parent中位数：几何步长=`0.258385/0.036363`、距离增量=`-0.151517/-0.005776`、路径直线度=`0.937252/0.876090`、RD零速能量=`0.451032/0.291077`、谱熵=`0.775495/0.806001`。仅6个错误parent且分布重叠，不能作为收益结论。
- Unknown错误/正确中位数：几何步长=`0.225203/0.238855`、路径直线度=`0.938388/0.890592`、RD带宽=`30.366827/27.971639`、零速能量=`0.560757/0.676832`、谱熵=`0.628636/0.519131`。6个错误Unknown跨越窄带近零速与宽带高熵两端，不存在由单一阈值支持的紧凑残差簇。
- 12个最终错误parent的最终置信度全部≥0.645257，中位数=`0.868866`，5个≥0.90，0个<0.60；现有“只关注低于0.60置信度”规则无法覆盖这些高置信错误。
- 创建并运行`src-v3/report/gate_residual_integrity.py`。12个parent的航迹、点迹、原始回波和全部声明MAT均存在；manifest/文件名标签码一致；长度、事件序、RD源帧唯一性、张量引用和V3逐点映射全部通过。
- 11个残差parent具有全RD点覆盖。`cq08|track|333`为22点Bird航迹，源物理帧索引的22帧只标到point 16–22；V3恰好映射这22帧且无重复，因此当前链路内是源级晚启动/部分覆盖，不是V3漏配。该结构一致性不能证明人工语义标签正确，人工标签复核和开放集Unknown审查均未执行。
- 残差特征报告原`formal_training=未执行`会混淆模型训练与审计动作，已修正为“V10正式训练已由用户完成；本描述性审计未启动训练”。没有修改数据、配置、checkpoint或指标。
- 残差特征报告 SHA-256=`6e3c1dc4610da70822492df3705e30551803ebcbb5020c46ef6cfd207a49ac30`；结构审计=`4585373ddd578b8cfba3259599c62a6140f010c7a4ec69865af0646e58d1aa8f`。

实际命令与验证：

- 在`src-v3`执行GPU `report.tasks` validation、`report.gate_paired_evaluation`、`report.gate_residual_features`和`report.gate_residual_integrity`；均退出码0。
- `py_compile report/gate_paired_evaluation.py report/gate_residual_features.py report/gate_residual_integrity.py`通过；`python -m compileall -q .`通过；`python -m smoke.static_isolation_check --root F:\\Project\\RadarInsight\\src-v3`通过，`violations=[]`。
- 一次组合PowerShell只读查询因管道语法解析失败，未写文件；拆分后通过。一次过宽`rg`只读检索在30秒超时并产生大量无关输出，未写文件；后续改用定向CSV/JSON读取完成核验。
- 新增或修改仅位于`src-v3/report/`、`src-v3/processed/report/`和三份跟踪文件；`data/`、`config/RADAR-LN3/`和正式artifact均未修改。

Stage 037当时判断：V10证实去除V9 dt可用性捷径后仍能获得更好的单seed validation Gate，但CU门槛未达，瓶颈集中于Bird/Unknown边界而非Clutter或液态单元复杂度。V10只保留为研究checkpoint。若人工标签/配对语义复核通过，下一候选仅允许增加“当前事件RD物理统计分支”作为V11单因素；液态单元、V3 split、TR增量、loss、class weight、采样、训练预算和checkpoint选择保持不变。该时点V11未执行；后续完整实施准备见Stage 041。

未执行：test、多seed、train parent-grouped OOF拒识校准、真实开放集Unknown评估、V10同版本DBB/CU specialists、正式三checkpoint联合评估、线上stable、P95和生产切换。

## 2026-08-07 - Stage 038 V10 Bird/Unknown残差人工复核包开始

用户确认：用户回复“可以”，按Stage 037推荐顺序先准备12个残差parent的人工语义复核证据，不直接创建V11或启动训练。

开始前边界与只读检查：

- 范围冻结为V10 validation最终误路由的6个BirdTarget和6个UnknownTarget；test、train阈值拟合、reject、stable、正式级联和P95不在本阶段。
- 读取Stage 037的paired、残差特征与结构完整性报告，V3 index、V10 `sample_detail_information.json`和报告入口；确认现有报告只有parent聚合统计，尚无面向人工审核的逐事件对齐包。
- 计划输出总览、逐事件证据和空白review模板；自动程序只允许填结构事实和风险提示，`review_decision`、`reviewer`、`review_notes`等人工字段必须保持空白。
- 本阶段不修改标签、V3 index、CQ08 manifest、RD张量、config或checkpoint，不创建V11版本文件，不运行训练。

实施与产物：

- 创建`src-v3/report/gate_residual_review.py`：只读取V3 index、V10 validation sample detail、Stage 037完整性报告、CQ08 manifest和现有物理RD tensor；对齐232个因果事件的原始TR字段、9维V10逐观测增量、7维RD物理统计、RD映射和`P(CU)`。
- 创建`src-v3/validate/gate_residual_review.py`：独立验证12 parent、232 event、6 Bird/6 Unknown、prefix连续性、概率范围、217个RD-present/15个RD-missing、输出文件存在性、人工字段空白和V11阻塞状态。
- 创建`src-v3/smoke/gate_residual_review_contract_test.py`及结果`src-v3/smoke/RADAR-LN3-V10-gate-residual-review-contract-result.json`，SHA-256=`27a2c8fbc55805636fcbabba0431bbe86b0f987854aa466219cd1504c23be014`。测试除验证真实包外，还在`src-v3/smoke`临时目录模拟人工填写，确认生成器拒绝覆盖。
- 生成`src-v3/processed/report/RADAR-LN3-V10-gate-residual-manual-review/`：包含`index.html`、`README.md`、`parent-index.csv`、`all-events.csv`、`manual-review.csv`、12份parent事件CSV、12份parent摘要JSON、12份TR/前缀演化图和12份RD物理证据图，共53个summary内记录的文件，约4.40 MB。
- 每份RD证据图包括prefix×物理速度的归一化谱演化和该parent所有V3映射帧的聚合range-Doppler相对功率；只用于离线描述，不作为模型输入、阈值或性能结论。
- 包summary SHA-256=`c542f1a2da8244ee6d7826b1502b0d9e1fe351e740e88fc5a440171a224d7a0c`；HTML=`8eac93d781c11878ab2ba13bef5d292993de79a894a60fbc7d38df735950d02f`；人工模板=`6a83d3cf7adf369a0342a6e66e0f577f52df17b64817231b077898c69876bfa0`。

实际验证：

- 报告入口最终运行退出码0：`parent_count=12`、`event_count=232`、Bird/Unknown各6、`review_status=awaiting_human_review`、`review_decisions_completed=0`、`automatic_label_changes=0`、`v11_release=blocked_pending_human_review`。
- 初始证据包生成时，独立验证器和focused防覆盖测试均退出码0；当时12行review决定、审核人、日期和备注全部为空。随后用户观察已按本Stage后续记录写入`review_notes`，决定/审核人/日期仍为空。
- 人工抽查track 333的TR/prefix图和RD图：前15个prefix呈灰色RD缺失、后7个prefix有映射谱，符合Stage 037的源级部分覆盖；抽查track 2862图可见其Unknown→DBB高置信演化。图像抽查不构成标签裁决。
- `py_compile`、全量`python -m compileall -q .`和静态隔离均通过，`violations=[]`。

实际失败与修复：

- 第一次生成在最终`summary.json`序列化时因`label_counts`含NumPy `int64`失败；此前生成的CSV/PNG/HTML没有人工内容。改为显式Python `int`后完整重跑通过。
- 一次组合预检查因`rg`无匹配返回码1，使并行工具整体显示失败；独立`py_compile`随后退出码0，代码无语法错误。
- 第一次RD图抽查路径遗漏track ID补零，文件查找失败且未修改任何内容；使用正确`track-0333-...`路径后图像读取正常。
- 两次跟踪文件收口补丁因Markdown反斜杠上下文不匹配而未应用；拆分为精确补丁后更新。代码、报告和数据不受影响。

用户观察记录与机器复核（同日继续）：

- 用户提供12个残差parent的曲线观察。统一用`P(CU)`复核后，将观察与机器值并列写入`manual-review.csv`的`review_notes`；明确标注为“非标签裁决”，没有填写`review_decision`、`reviewed_semantic_label`、审核人或日期。
- 5个持续Bird→CU：333/845/852/2367/2956全部prefix为CU；3581的`P(CU)=0.522–0.766`，始终在CU侧而非真实路由摇摆。2862和4171全部prefix为DBB；4171不存在用户视觉描述中的早期CU阶段，机器数据为`P(CU) 0.326→0.103`。
- 后期反转Unknown：2908前11个prefix为CU后一次跨界，最终`P(CU)=0.192`；3122前6个后跨界，最终0.159；3154前5个后跨界，最终0.312。1547不是简单单调下降：起始0.238、prefix 6峰值0.733、四次跨界、最终0.316。
- 跨界前后描述对照不拟合参数：2908后段相对前段RD带宽中位数`10.584→39.574 m/s`、零速能量`0.954→0.191`、谱熵`0.193→0.909`；3122和3154未呈现相同方向，不能将三者统一归因于液态遗忘或单一RD阈值。
- 残差特征对照显示，错误Bird的几何步长/路径直线度中位数高于正确Bird，错误Unknown也高度笔直；已确认“明显运动”不等价于“已知DBB身份”，不能继续用运动强度单轴解释DBB/CU。
- 检查Unknown capture：2862/2908/3122/3154集中于08:57–09:39，但同时间段也有正确Unknown 2907/2912/3123；只有validation描述证据，不足以断言批次泄漏或域偏移。

审核契约修正与实际验证：

- 修改`validate/gate_residual_review.py`，允许`review_notes/additional_evidence_required`保存非裁决观察；合法人工决定必须属于冻结词表且具备审核人和日期，非法词表或孤立审核字段会失败。校验同时核对review文件SHA-256/大小与summary。
- 修改focused测试，要求12条观察不得被提升为决定，并增加非法决定防护。最终`validate.gate_residual_review`和`smoke.gate_residual_review_contract_test`均通过：12 observations、0 decisions、0 automatic label changes、V11=`blocked_pending_human_review`；防覆盖和非法决定防护通过。
- 当前`manual-review.csv` SHA-256=`84f94c3704928c3abb5657e8ba8b5c2b704233bb2d45c2d4320a187ecdd114e4`，summary=`c36f9d66c02fd2f91430f72f62c1b3e1244ef22e20107d354bd8c060e4bb5911`，focused结果=`3a656d31713c7912ddafc302e33294984f87ba79941af78929a4165d2a9cd1ac`。
- 最终`py_compile`、全量`compileall -q .`和src-v3静态隔离通过，`violations=[]`。一次PowerShell哈希汇总因管道解析错误退出，未写文件；拆分为显式`$rows`后成功。

当前状态与放行：自动证据准备和用户观察记录已完成，但人工语义裁决未执行。`manual-review.csv`当前12行决定均为空，V11保持阻塞；没有修改标签、data、config、checkpoint或artifact，没有读取test，没有训练。

## 2026-08-08 - Stage 039 V10残差指标图与人工查看准备

用户选择方案A：先完成12条残差parent的人工语义复核准备，生成TR/RD指标图并提供需要比对的指标和合理范围；本记录只覆盖证据整理和人工查看，不自动完成标签裁决。

开始前只读核验：

- 已读取`src-v3/task_plan_v3.md`、`src-v3/process_v3.md`、`src-v3/findings_v3.md`和Stage 038审核包；确认`manual-review.csv`的12行`review_decision`、`reviewed_semantic_label`、`reviewer`和`review_date`仍为空，`review_notes`仅为用户提供的非裁决观察。
- 已核对Stage 038的`all-events.csv`为232个因果事件、6个BirdTarget和6个UnknownTarget，RD-present=217、RD-missing=15；V10残差结构审计已通过，test、V11、阈值搜索和训练均未执行。
- 已核对当前残差指标口径为V10逐观测TR增量和当前事件RD物理统计；本阶段参考范围必须从同一CQ08 V3 train parent分布计算，不能从12个validation残差反推阈值。

本阶段计划修改：

- 新增`src-v3/report/review_metrics.py`，生成只读指标比较表、类别参考范围、百分位热图、质量图和逐parent指标图；输出放入Stage 038目录下的独立`metrics/`子目录，不覆盖原审核包。
- 新增报告指南，明确P05-P95为CQ08 train同类parent中位数的描述性筛查区间；Q1-Q3为典型区间；超出区间只能说明“与训练参考分布不同”，不能说明误标、Unknown或数据损坏。
- 只更新`src-v3`代码、`src-v3/processed/report`报告和三份跟踪文件；`data/`、`config/RADAR-LN3/`、`artifacts/`、标签、checkpoint和原始数据不修改。

开始时尚未执行：报告代码实现、指标重算、PNG生成、图像人工查看、全量一致性审计、test、V11配置/数据契约、阈值校准、训练和正式级联；后续执行结果见下方同日Stage 039收口记录。

## 2026-08-08 - Stage 039指标报告、图像查看与收口

实际实现与命令：

- 新增`src-v3/report/review_metrics.py`，只读取V3 CQ08 index、V3映射RD tensor和Stage 038审核包；没有读取test、修改标签、写入data/config/artifacts或启动训练。
- 初次使用错误相对路径`..\\.venv\\Scripts\\python.exe`未执行成功；定位后使用实际解释器`F:\\Project\\RadarInsight\\.venv\\Scripts\\python.exe`重跑。
- 实际命令：`cd F:\\Project\\RadarInsight\\src-v3; F:\\Project\\RadarInsight\\.venv\\Scripts\\python.exe -m report.review_metrics --project-root F:\\Project\\RadarInsight`；退出码0，运行约71.8秒。
- 实际命令：`F:\\Project\\RadarInsight\\.venv\\Scripts\\python.exe -m py_compile report\\review_metrics.py`；退出码0。
- 实际命令：`F:\\Project\\RadarInsight\\.venv\\Scripts\\python.exe -m smoke.static_isolation_check --root F:\\Project\\RadarInsight\\src-v3`；退出码0，`violations=[]`。
- 实际命令：`F:\\Project\\RadarInsight\\.venv\\Scripts\\python.exe -m validate.gate_residual_review --project-root F:\\Project\\RadarInsight`；退出码0，12 parent/232 event、0 formal decisions、V11=`blocked_pending_human_review`。
- 实际命令：`F:\\Project\\RadarInsight\\.venv\\Scripts\\python.exe -m smoke.gate_residual_review_contract_test --project-root F:\\Project\\RadarInsight`；退出码0，package validation、human-input overwrite guard和invalid-decision guard均通过。
- 实际命令：`F:\\Project\\RadarInsight\\.venv\\Scripts\\python.exe -m compileall -q .`；退出码0。

实际统计：

- CQ08 V3 train参考：1081 parent、20684事件；其中BirdTarget=243 parent、UnknownTarget=70 parent；参考单位为同标签parent级中位数。
- V10残差：12 parent、232事件、BirdTarget=6、UnknownTarget=6；RD-present=217、RD-missing=15；审核表正式决定数仍为0。
- 生成`src-v3/processed/report/RADAR-LN3-V10-gate-residual-manual-review/metrics/`，包含`train-parent-metrics.csv`、`residual-parent-metrics.csv`、`reference-ranges.csv`、`residual-metric-percentiles.csv`、总体箱线图、百分位热图、质量/覆盖图、12个parent百分位图、HTML和指南；summary状态为`passed`，test/训练/阈值拟合/标签修改均为“未执行”。
- 已使用`view_image`查看总体3图、12个指标百分位图、12个TR/prefix图和12个RD证据图；人工查看记录写入`metrics/manual-view-notes.md`。

图像查看结论：

- 12例中TR步长、方向余弦和路径直线度与同类train分布大量重叠，运动明显不能单独证明DBB。
- RD存在近零速窄带、宽谱高熵和前后段谱形变化等异质形态；没有一条单一RD指标或固定范围可以安全区分标签、证明误标或实现Unknown拒识。
- track 333的7/22 RD覆盖是最明确的可比性风险；track 2908的后段RD变化与路由反转同步；track 2862和4171分别呈现近零速/集中与宽谱/高熵形态；这些是人工核对点，不是自动裁决。
- 未发现新的、足以支持`EXCLUDE_FOR_DATA_CORRUPTION`的自动结构证据；Stage 038的12/12结构审计结论保持有效。

修正与限制：

- 报告入口拒绝在`manual-review.csv`已有正式`review_decision`时继续运行，避免覆盖人工裁决；当前仅有用户非裁决观察备注，正式决定字段保持空白。
- `Q1-Q3`和`P05-P95`只用于同标签CQ08 train描述性比较，不是标签阈值、reject阈值或V11筛选规则。
- 发现并记录报告实现中的首次命令路径错误；没有产生代码或数据副作用。

Stage 039收口当时状态：指标计算、图像生成、图像查看和指南已完成；正式语义裁决当时仍待用户政策确认。该历史阻塞随后已由Stage 040的12条`KEEP_CURRENT_LABEL`裁决解除，原标签零改动；V11完整实施准备后续由Stage 041完成，正式训练、阈值搜索、test和正式级联仍未执行。

## 2026-08-08 - Stage 040残差标签边界与Track 333原始回波审计启动

用户确认：

- `UnknownTarget`包含风筝、昆虫等杂物，类内差异大，不能仅靠运动强度判为DBB。
- Bird可能静止在树上，低峰值速度不能作为否定Bird标签的依据。
- Track 333缺少部分RD，需要读取原始回波，可用Python `struct`和MATLAB双路径复核；MATLAB由用户手动运行。
- 原始采集记录已丢失，当前默认以现有标签为准；只有多方独立证据共同指向误标时，才能携带证据向用户申请，禁止自动改标。

启动时只读核实：

- `OriData/23/CQ-08/原始回波/333_Label_3.dat`存在，大小`2,694,512 bytes`。
- `OriData/23/CQ-08/MAT/`下Track 333有22个MAT，序号0001–0022；V3点级索引有22个TR事件。
- V3索引point 1–15为`rd_present=false`，point 16–18各映射4帧，point 19–21各映射2帧，point 22映射4帧，共22帧；现有映射事实本身不能证明前15点不可恢复。
- 已读取`src-v3/tests/processed/original_echo_audit.m`的帧格式：小端uint32头`FA55FA55`、长度字×4、尾`55FA55FA`、每目标4个uint32、CPI/PRT元数据和交错float32 I/Q。
- 现有12文件MATLAB抽样审计没有包含Track 333，因此本Stage需要聚焦复核。

本阶段允许修改：`src-v3`跟踪文件、审核报告元数据、`src-v3/report/`审计代码、`src-v3/tests/processed/`MATLAB脚本和`src-v3/processed/report/`审计产物。禁止修改：标签、`data/`、`config/RADAR-LN3/`、`artifacts/`、现有MAT和checkpoint。

启动时未执行：12条正式裁决写入、Python原始回波解析、MATLAB聚焦复核、数据重建、V11、训练、test、阈值搜索和部署切换。

## 2026-08-08 - Stage 040 Python审计、标签裁决与MATLAB脚本完成

12条标签裁决：

- 新增`src-v3/report/finalize_residual_review.py`，按照用户确认政策将12条正式决定写为`KEEP_CURRENT_LABEL`，`reviewed_semantic_label`逐行等于原`current_label`；BirdTarget=6、UnknownTarget=6，实际标签修改数=0。
- 审核说明明确：保留当前标签表示当前证据不足以推翻标签，不等于独立证明标签客观正确；Unknown异质性、运动强度、静止/低速Bird、模型置信度和任何单一TR/RD指标均不能单独触发改标。
- 改标重开条件冻结为多方独立证据共同冲突，并需携带证据向用户另行申请；程序不得自动修改标签。
- 裁决后`manual-review.csv` SHA-256=`5f8eff2749e5e32710d4bd3f9dc0397da4fe02c1b3a52a2bf1382cfe0b03195d`；summary SHA-256=`29dd794898e24b3df881caef14eab265f7d02dc879dbbd62d6f690a501596d19`。
- 更新独立验证器和focused测试：`KEEP_CURRENT_LABEL`必须保留原语义标签；12条裁决完成后状态为`ready_for_design_pending_explicit_authorization`，不会自动创建V11配置或启动训练。`validate.gate_residual_review`和`smoke.gate_residual_review_contract_test`均退出码0，12 decisions/12 observations/0 label changes；防覆盖和非法决定防护通过。

Python `struct`原始回波审计：

- 新增`src-v3/report/track_0333_original_echo_audit.py`，以小端结构从byte 0连续解析到EOF；遍历每帧全部target records，不假设每帧只有一个目标，也不把原始target identifier误写为CQ08 parent Track ID。
- 实际运行`F:\Project\RadarInsight\.venv\Scripts\python.exe -m report.track_0333_original_echo_audit --project-root F:\Project\RadarInsight`，退出码0。原始`.dat` SHA-256=`70bee1da3bf665d8436cd855d5d32ea97f9737fa05457388643b70ff60ef27a0`，大小/已解析字节均为`2,694,512`；从0到EOF恰有22个连续合法帧，header/footer全部有效，未解析payload=0，I/Q全部有限。
- raw原始point仅为16–22，帧数分布=`16:4,17:4,18:4,19:2,20:2,21:2,22:4`；point 1–15在该文件中不存在。
- 22/22 raw target record与MAT `debug_info.track_no_info_head`一致，point号、频率、CPI、PRT、data length和I/Q shape全部一致；复数I/Q逐元素最大绝对/相对误差均为0。
- raw逐point帧映射与22个MAT、物理帧索引和V3索引逐项完全一致。Python不执行RD数值对照，因为SciPy Taylor窗与MATLAB `taylorwin`归一化语义不同；MATLAB脚本负责权威RD重算。
- 报告写入`src-v3/processed/report/RADAR-LN3-track-0333-original-echo-audit/`；summary SHA-256=`6c23715ae697972bf8e024baf015df468fd5db4a023ab77bdb8871c34944fb9c`。没有写`data/`、config、artifact或MAT。

focused不可变输入契约：

- 新增并运行`src-v3/smoke/track_0333_original_echo_audit_test.py`；它重新执行解析，并对原始`.dat`、22个MAT、V3 index和物理帧索引共25个输入执行前后SHA-256比较。
- 测试退出码0：22帧、point 16–22、前15点不存在、raw/MAT I/Q完全一致、raw/MAT/物理/V3映射一致、25个输入hash不变；结果SHA-256=`f3c6f8f3130f832cf8e5b3641990df110dc04569aa976ccad39b025a33b9d3a8`。

MATLAB独立复核准备（当时状态；后续v2已通过）：

- 新增`src-v3/tests/processed/track_0333_original_echo_audit.m`，SHA-256=`6253946ca394387afae5b419867d63b74a197dc1fae1e950e28119b17abff4a6`。
- 脚本只读解析22帧，逐帧比较raw target record、MAT point号/元数据/IQ，并使用项目既有MATLAB公式`taylorwin(..., [], -30)`+慢时间FFT重算RD；输出CSV、JSON和首/中/末帧RD对照图到Track 333报告目录下的`matlab/`子目录，不覆盖现有MAT。
- 本准备记录形成时MATLAB尚未运行；后续用户已完成v1/v2，最终`all_rd_numeric_match=true`且误差为0，详见Stage 040收口记录。

最终静态验证：

- `python -m compileall -q .`退出码0。
- `python -m smoke.static_isolation_check --root F:\Project\RadarInsight\src-v3`退出码0，`violations=[]`。
- 一次只读搜索因工作目录误设为`F:\Project\RadarInsight\src-v3`且路径仍带`src-v3/`前缀而失败；未写任何文件，随后在项目根目录重跑成功。
- 最终并行一致性检查第一次从项目根目录直接运行`python -m smoke.static_isolation_check`，因`src-v3`不在该进程模块搜索根而报`ModuleNotFoundError: No module named 'smoke'`；未修改文件。随后以`F:\Project\RadarInsight\src-v3`为工作目录依次重跑审核验证、Track 333 focused测试、compileall和静态隔离，全部退出码0。

该时点的阶段结论：Python已证明Track 333前15个TR点没有可从指定`.dat`恢复的RD帧，MATLAB当时尚待独立复核。后续MATLAB v2与Python完全一致，最终结论见本文件Stage 040收口记录。该时点V11尚待授权；后续完整实施准备见Stage 041，正式训练和test未执行。

## 2026-08-08 - Stage 040首次MATLAB结果与coverage假阴性定位

用户实际运行`track_0333_original_echo_audit(pwd)`；MATLAB输出和落盘summary确认：raw frame=22、MAT=22、raw point=`[16 17 18 19 20 21 22]`、point 1–15不存在、22/22 target record/point/元数据/IQ/RD均匹配，I/Q与RD最大绝对和相对误差全部为0。原始`.dat` SHA-256仍为`70bee1da3bf665d8436cd855d5d32ea97f9737fa05457388643b70ff60ef27a0`。

首次summary却为`status=needs_review`，唯一失败字段是`all_point_coverage_raw_matches_mat=false`。读取实际`point-coverage.csv`后确认：point 16–22逐行`raw_matches_mat=1`；point 1–15的raw/mat frame count均为0，但raw空序列显示`[]`、MAT `find`空序列显示`zeros(1,0)`，逐行`raw_matches_mat=0`。根因是脚本使用`isequal(rawSequences, matSequences)`，MATLAB把两个内容相同的空集合表示为`0x0`和`1x0`不同形状。

该问题属于审计脚本假阴性，不是raw/MAT数据、帧内容、RD重算或V3映射冲突。修复范围冻结为把coverage内容比较规范化为方向无关的列向量，并统一空序列报告文本；不修改原始数据、MAT、索引、配置、标签、checkpoint或训练链路。首次MATLAB报告作为真实历史输出保留，修复后需用户原命令重跑生成新summary。

实际修复与静态验证：

- `buildCoverageRows`改为`isequal(rawSequences(:), matSequences(:))`，只消除行/列方向差异；新增`sequenceString`令任意空集合统一输出`[]`。原始帧解析、target匹配、元数据、I/Q和`taylorwin`+FFT RD公式均未修改。
- 修复后报告schema为`RADAR-LN3-track-0333-original-echo-matlab-audit-v2`，输出目录改为`.../matlab-v2/`；首次`.../matlab/`假阴性报告保留，不覆盖、不重命名。
- 新增`src-v3/validate/track_0333_matlab_audit.py`，第二次运行后将独立校验22帧、point 16–22分布、22个frame status、IQ/RD零误差、22个point coverage、原始文件hash和不可恢复结论。
- 修复后MATLAB脚本SHA-256=`0445762d7f6dc34421d143f494629c851fb30d7aeab165f3d26e778596ef7db4`；验证器SHA-256=`52b5b1669c60e84f0606106386b6c34f5196ca0a565a09e79c42e959ef6ac66f`。
- `py_compile validate\track_0333_matlab_audit.py`、全量compileall、静态隔离和`git diff --check -- src-v3`均退出码0；隔离结果`violations=[]`。
- 本修复记录形成时v2重跑尚未执行；后续用户已完成重跑，`validate.track_0333_matlab_audit`也已退出码0，详见下一节收口记录。

## 2026-08-08 - Stage 040 MATLAB v2交叉验证通过与收口

用户手动重跑修复后的MATLAB脚本，实际输出`status=passed`、schema=`RADAR-LN3-track-0333-original-echo-matlab-audit-v2`。落盘结果路径为`src-v3/processed/report/RADAR-LN3-track-0333-original-echo-audit/matlab-v2/`，没有覆盖首次`matlab/`历史报告。

实际结果：

- raw/MAT均为22帧，原始文件从byte 0连续解析至EOF；raw point仅`16–22`，帧数=`4/4/4/2/2/2/4`，point 1–15不存在。
- 22/22 raw target记录、point号、频率/CPI/PRT/data length、I/Q shape全部与MAT一致。
- 复数I/Q和MATLAB`taylorwin`+慢时间FFT重算RD逐元素完全一致；I/Q和RD最大绝对/相对误差全部为0。
- 22个TR point coverage全部`raw_matches_mat=true`：point 1–15双方均0帧，point 16–22双方帧数和序号一致。
- MATLAB v2 summary SHA-256=`2da9bc7a79ac990f6e23193765d0466e37cde2092424f49617ad34d4bcbdeb48`；frame CSV=`c33773366a76bc94545cd8a2af5df59179ee298bc2a51d928a83b26fd437c69f`；coverage CSV=`504cbc8152a3b84cfb8cbb6fe0d60b03775fce8b3ca10a07e8997485c275e7be`。

独立验证与图像检查：

- 运行`F:\Project\RadarInsight\.venv\Scripts\python.exe -m validate.track_0333_matlab_audit --project-root F:\Project\RadarInsight`，退出码0；验证器复核summary、22帧CSV、22点coverage、raw SHA-256和不可恢复结论，全部通过。
- 使用`view_image`查看`rd-comparison.png`：首帧point 16、中间帧point 18、末帧point 22的raw重算RD与MAT视觉重合，三张误差图标注最大值均为0。
- 查看`point-coverage.png`：point 1–15无柱，point 16–22的raw/MAT柱逐点重合并符合`4/4/4/2/2/2/4`。
- 生成`cross-validation-summary.json`，记录Python summary hash、MATLAB v2 summary hash、双路径一致性、首次假阴性历史和最终不可恢复结论。

Stage 040收口时结论：Track 333前15个TR点没有可从指定`.dat`恢复的RD；这是源端晚启动/部分采集，不是MAT转换、V3索引或配对丢帧。无需且不应重建/伪造前15点RD；模型继续使用显式missing mask。Track 333继续保留`BirdTarget`，RD缺失不构成改标证据。12条残差均保留当前标签；该时点V11仅可进入设计讨论。其后完整实施准备由Stage 041单独授权并完成，正式训练和test仍未执行。

最终全链路focused复跑：

- 更新`smoke.track_0333_original_echo_audit_test`，使其同时调用MATLAB v2结果验证器；最终结果明确记录`matlab_independent_audit=passed`、IQ/RD/coverage均为true。
- 最终focused结果SHA-256=`7217f60d2867dbbabaccb3cab73403db598723c9f2aa8e7f751e2aa99542d663`；交叉验证摘要SHA-256=`aee3aa4cb0f1a1e254d2a0b64db07a2438274a829ed56525386e44b696f86a3d`。
- 最终依次运行Track 333双路径契约、12条审核验证、审核防覆盖/非法决定契约、全量compileall和src-v3静态隔离，全部退出码0；`violations=[]`，`git diff --check -- .`通过。
- 一次组合复跑因用于确认“无过期表述”的`rg`零匹配返回码1而被工具整体标为失败；功能测试未因此失败或修改文件。随后拆分命令，所有功能检查均通过。

## 2026-08-08 Stage 041 - V11当前事件RD物理谱paired实施准备（启动记录）

用户授权：选择“完整实施准备，不启动正式训练”。允许创建V11派生缓存、manifest、数据/输入契约、正式配置及`src-v3`代码和有界Smoke；禁止代理启动6次正式训练、第二阶段OOF、test评估、阈值拟合、正式级联、P95或部署切换。

启动前只读核验：

- 已读取`task_plan_v3.md`、`process_v3.md`和`findings_v3.md`现有内容并保留Stage 008–040有效历史；Stage 041首次覆盖式登记于三个跟踪文件。
- 已执行`git status --short`；发现既有未跟踪报告、Smoke、MATLAB审计和`tmp/`测试目录，均视为用户历史工作并保留，不清理、不覆盖。
- 已列举`config/RADAR-LN3/`与`data/processed/`的RADAR-LN3文件；最大已用全局版本为V10，未发现V11命名文件。
- 已确认本轮继续只读复用V3 index/split、V2 train-only归一化统计、V10输入契约和canonical RD张量；V11只新增派生物理特征缓存与对应契约，不覆盖旧文件。

冻结实施语义：

- 四维特征顺序固定为`velocity_bandwidth_mps_over_85`、`zero_velocity_energy_fraction_abs_le_2_mps`、`velocity_spectral_entropy_over_log_384`、`top10_velocity_energy_fraction`。
- 功率权重固定为每帧`10 ** ((amplitude_db - frame_max_db) / 10)`，再跨当前事件的有效帧和距离维聚合速度谱；只使用规范化前RD幅度。
- 仅生成train=20684和val=4474事件缓存；共享V3索引中的test=4505行仅扫描`split_name`后排除，不打开test RD payload、不生成test缓存或指标。缓存字段为`sample_ids`、`values[N,4] float32`和`valid[N] bool`，sample ID只作对齐。
- Control为V10架构重训；Treatment只将Fusion输入127→131，新增四列权重零初始化；其余模型、数据、loss、预算和评估保持一致。
- seed为42/43/44，固定20 epoch、关闭early stopping；各自best为主，同时保存逐epoch恢复点、last和同epoch比较轨迹。仅固定初始化与DataLoader顺序，不开启CUDA严格确定性算法，不宣称位级确定性。
- 第一阶段门槛与离散Pareto已冻结；第二阶段五折grouped OOF当前不实现、不运行。预训练不需要。

实施代码与职责目录：

- 新增`src-v3/radar_ln3/rd_physical.py`：共享四维物理谱提取、速度轴验证、缺失语义和缓存加载契约。
- 修改`src-v3/radar_ln3/data.py`：可选V11缓存严格sample对齐、PointSequence/collate四维值和事件级valid；历史配置保持0维且不暴露V11 batch键。
- 修改`src-v3/radar_ln3/task_models.py`：V11配置字段、127/131维paired Fusion、共享初始化、在线同源入口、checkpoint身份与完整epoch边界恢复状态；V6–V10默认不启用。
- 修改`src-v3/train/tasks.py`：独立同seed generator、每epoch parent顺序hash、每epochcheckpoint、best/last、显式resume、正式artifact防覆盖和V11调试参数禁用。
- 新增`src-v3/build/rd_physical_paired.py`、`validate/rd_physical_paired.py`、`smoke/rd_physical_paired.py`、`smoke/rd_physical_paired_report_test.py`、`smoke/rd_physical_readiness.py`和`report/rd_physical_paired.py`。

V11版本化文件与hash：

- Train cache：`data/processed/RADAR-LN3-V11-cq08-train-rd-physical-event-cache.npz`，SHA-256=`7261431fe1e39465ad3eb11802f9972b09da62299c3465101dafcd03aac42ba5`。
- Validation cache：`data/processed/RADAR-LN3-V11-cq08-val-rd-physical-event-cache.npz`，SHA-256=`91776d1dfce1f4804d31e490a86339ada463647ffae3e65703e741f21817e7fc`。
- Cache manifest：`data/processed/RADAR-LN3-V11-cq08-rd-physical-cache-manifest.json`，SHA-256=`353222973278fcfa63d055d628c391f9fcaedbc8021d500757827237ce48b2e7`。
- 数据/输入契约：`data/processed/RADAR-LN3-V11-cq08-tr-rd-physical-gate-data-contract.json`，SHA-256=`3d0caa41de518ff4f49e42b42187803db03f31ca7f8197b3c7c472da327e54e4`。
- 六配置hash：Control seed42/43/44分别为`80cb5b45d33f0199ce67a5d3d31e24b930ae53906178c988dbfdaea748365571`、`ccee2ef1e049606733ba3b1c9343a1306ba324810b864f68c0bfd197aeab0cf5`、`e8ae9a38a72b785925c1c37b677bcb502ea4995d38d586915023ec4c3fd4111f`；Treatment seed42/43/44分别为`dbb867c88530e2290ddccdcf6ba09fa2a8bcf7aaca94bb36157d1cd76df5220c`、`a9cea4b2f1d1ae42f086880109a30f83c7a5f4c7dc1f57df77c30542fdd0659e`、`f4121afb3379146b5a93eb75642668b3a62b90107abd67c27c4fedd083877437`。
- 版本递增原因：V11实质增加派生缓存、四维输入schema、Fusion输入维度、checkpoint和paired训练语义；V3 index/split、V2统计、标签及canonical RD tensor保持只读复用。重复运行构建器时两缓存、manifest、契约和六配置全部为`reused_identical`，未创建新版本。

实际数据构建和全量验证：

- 首次1秒工具时限调用在写入前超时；确认无半成品后以合理时限重跑成功。Train共20684事件，其中valid=20482、missing/invalid=202、选中RD帧73938；Validation共4474事件，其中valid=4434、missing/invalid=40、选中RD帧15876；没有事件超过12帧而被截断。
- 验证器独立遍历全部25158个train/val事件并调用共享提取函数重算，sample集合/顺序完全一致、重复数0、`rd_present==valid`、最大逐值误差0；访问1313个train/val canonical RD tensor。
- Track 333保持22事件：point 1–15四维全零且valid=0，point 16–22 valid=7；没有回填或重建原始数据。
- 共享V3 index顺序扫描时只按`split_name`计数并排除4505个test行；test RD payload访问数=0、test cache不存在、test评估未执行。这里的“test封存”指不选择test事件、不解析/打开其RD payload、不生成特征或指标，不虚构为共享CSV字节完全未扫描。

模型、paired和在线契约结果：

- V10真实回归batch保持Fusion=127且不含V11 batch键；Treatment真实batch新增`rd_physical_features=[2,22,4]`和`rd_physical_valid=[2,22]`，Fusion=131。
- 三个seed均有42个共享state entry逐值一致，Treatment新增四列严格为0；用非零0.5四维探针时CPU初始logits逐位一致、最大差0。Control不加载或消费物理缓存。
- CUDA未开启严格确定性算法；seed42两次独立paired前向最大差`3.427267074584961e-07`，在`5e-6`容差内数值一致但非逐位一致。该事实符合用户选择，只能宣称固定共享初始化与采样顺序，不能宣称CUDA位级确定性。
- 缓存、共享提取器和模型在线准备逐值一致；离线/在线prefix最大logit差CPU=`4.76837158203125e-07`、CUDA=`1.4901161193847656e-07`；在线缺失事件全零/invalid，超过12帧明确拒绝。
- Treatment物理新列CPU/CUDA梯度范数分别为`2.780959129333496`和`2.7809600830078125`，一次Smoke优化后确实更新；这只证明链路可训练，不是性能提升。

checkpoint、训练和报告契约：

- 每epoch checkpoint保存模型、优化器、Python/NumPy/Torch CPU/CUDA RNG、DataLoader generator、global step、best selection和history；CPU/CUDA checkpoint恢复后模型logits、优化器、全部RNG、loader状态和run state均精确恢复。
- V11正式入口固定20 epoch、early stopping=0，禁止`--max-steps`、epoch覆盖或跳过validation；实际执行带`--max-steps 1`的护栏命令在创建artifact前按设计抛出ValueError。六个正式artifact路径最终均不存在。
- 训练后报告入口会独立加载六个best checkpoint重算validation，并核对20个epoch顺序hash、各自best、epoch20、双方best epoch、至少2/3 seed、三seed中位数和离散Pareto；synthetic focused测试证明通过场景放行、CU弱场景阻塞、顺序hash不一致被拒绝。

实际验证命令与结果：

- `python -m build.rd_physical_paired`首次成功创建，第二次全量重算后全部`reused_identical`。
- `python -m validate.rd_physical_paired`退出码0，全量逐值复算通过。
- `python -m smoke.rd_physical_paired --device cpu`和`--device cuda`退出码0；结果仅位于`src-v3/smoke/`。
- `python -m smoke.rd_physical_paired_report_test`、`python -m smoke.rd_physical_readiness`、V10 observation-increment回归、V9 motion回归、V7/V8修复回归、全量compileall和静态隔离均退出码0；隔离`violations=[]`。
- 实施中一次`break outside loop`编译错误已在任何Smoke/训练前修正；首次CUDA逐位相等断言暴露`3.43e-7`非严格数值差，随后按已冻结协议改为CPU逐位护栏+CUDA数值容差并如实记录。全工作区`git diff --check -- .`受用户历史`tmp/`不可读目录干扰；对本轮修改文件的限定检查退出码0，无trailing whitespace。

最终readiness：`src-v3/smoke/RADAR-LN3-V11-rd-physical-training-readiness.json`状态为passed。技术结论为“可以进入用户手动V11 Control/Treatment三seed正式训练”；不等于性能达标。预训练不需要。正式训练、正式validation性能、test、第二阶段OOF、阈值拟合、正式级联、P95和部署切换均未执行。

## 2026-08-08 Stage 042 - V11共享物理缓存跨账户读取修复（诊断与收口记录）

用户报告的实际失败：在交互式PowerShell中分别启动Treatment seed42和seed43，均在`src-v3/radar_ln3/task_models.py:592`计算train cache SHA-256时抛出`PermissionError [Errno 13]`；失败路径均为`data/processed/RADAR-LN3-V11-cq08-train-rd-physical-event-cache.npz`。代理未重跑正式训练。

只读诊断证据：

- seed42与seed43配置复用同一train cache、val cache及相同预期hash；train=`7261431fe1e39465ad3eb11802f9972b09da62299c3465101dafcd03aac42ba5`，val=`91776d1dfce1f4804d31e490a86339ada463647ffae3e65703e741f21817e7fc`。因此这是共享文件访问问题，与seed、模型初始化或训练逻辑无关；seed44 Treatment也会受同一问题影响。
- 两个NPZ均是普通文件，大小分别待最终记录；ACL均为`AreAccessRulesProtected=True`，仅显式允许OWNER RIGHTS、SYSTEM和Administrators。`data/processed`父目录及同目录V3 CSV、V11 JSON均为继承ACL，包含Authenticated Users/Users读取权限。
- 根因位于`src-v3/build/rd_physical_paired.py::_write_npz_new_or_identical`：缓存在`TemporaryDirectory`内创建并通过`temporary_path.rename(path)`移动到目标目录；Windows移动保留源文件的受保护安全描述符，没有自动采用目标父目录ACL。JSON文件直接在目标目录创建，因此没有该问题。

本Stage修复边界：只修改V11缓存发布方式和现有train/val缓存ACL元数据；不得修改NPZ字节、manifest、数据契约、六配置、模型、标签、split或版本号。修复前暂时撤回训练放行。

实际修复与验证：

- 将`_write_npz_new_or_identical`从“受保护临时目录内写文件后rename”改为“内存生成确定性ZIP/NPZ字节、`xb`直接创建最终文件、flush/fsync、即时round-trip”；保留版本化不覆盖和确定性字节语义。
- Windows受控工作区可能在直接创建时附加专用ACL，因此发布后统一调用精确文件目标的`icacls /reset`采用`data/processed`父目录DACL；内容相同的`reused_identical`路径也执行该规范化，使旧缓存可幂等修复。非Windows不执行平台动作。
- 首次尝试以PowerShell `Set-Acl`启用继承因缺少`SeSecurityPrivilege`失败，ACL和文件内容均未改变；随后用Windows原生ACL工具完成精确文件修复。一次性诊断探针证明`/reset`稳定产生9条继承规则，诊断文件随后删除。
- Train cache最终长度`346504`、SHA-256=`7261431fe1e39465ad3eb11802f9972b09da62299c3465101dafcd03aac42ba5`；Validation长度`75717`、SHA-256=`91776d1dfce1f4804d31e490a86339ada463647ffae3e65703e741f21817e7fc`。两者均`Protected=False`、继承规则数9，并包含Authenticated Users修改/读取与BUILTIN Users读取执行权限。文件字节、manifest、契约和配置均未变化，版本保持V11。
- 新增`src-v3/smoke/rd_physical_cache_access_test.py`。实际结果验证新文件`created`、重复写`reused_identical`、字节hash稳定、NPZ字段/数组一致；正式train/val缓存分别20684/4474事件；seed42/43/44三份Treatment数据契约全部passed；单parent真实Dataset shape为`[21,4]`、valid shape=`[21]`且finite。
- focused结果：`src-v3/smoke/RADAR-LN3-V11-rd-physical-cache-access-test-result.json`，SHA-256=`e48fbfc604c08a10bd94e4931750e5bbb084977f21608ff8eded637046d0175d`。临时ACL/NPZ探针均已删除。
- `python -m compileall -q .`通过；静态导入隔离通过且`violations=[]`；受影响文件的`git diff --check`通过。没有运行test、OOF、拒识校准、正式级联或P95。
- 将ACL focused结果加入`smoke.rd_physical_readiness`的后续硬依赖。训练开始后尝试重建readiness时，安全护栏因Control正式artifact已非空而按设计拒绝；这是训练前快照边界，不是ACL或模型失败，未修改/覆盖任何训练artifact。

用户侧最终验证：用户明确确认“训练正常进行，修复成功”。这证明交互式训练账户已越过原`task_models.py:592-593`的train/val缓存读取和hash校验阻塞。Stage 042完成并恢复训练放行；paired正式训练由用户手动进行中，完整六run validation性能仍未评估，test、OOF、阈值、正式级联和P95仍未执行。

## 2026-08-08 Stage 043 - V11三seed paired正式结果复核（启动记录）

用户确认训练已完成并授权自行评估分析。启动前只读核验六个Control/Treatment artifact：seed42/43/44各自均有20个`epoch-XXX.pt`、`best_model.pt`、`last_model.pt`、`training-summary.json`、`sample_detail_information.json`和`training.log`；六份日志均以`run=complete status=passed`结束。现有联合报告写于2026-08-08 23:19:15，晚于最后一个run完成时间23:12:38，初读状态为`completed_but_gate_failed`。本Stage仍将用当前报告入口独立重载六个best checkpoint重算validation，避免直接采信旧文件。test、OOF、拒识阈值、正式级联、P95和新训练均未执行。

正式paired复核：

- 在`src-v3`实际执行`F:\Project\RadarInsight\.venv\Scripts\python.exe -m report.rd_physical_paired --project-root F:\Project\RadarInsight`。六个best checkpoint均按配置在CUDA上独立重载，分别重算同一CQ08 V3 validation（232 parent、4474 causal-prefix window）；所有指标与对应`training-summary.json`逐值一致。
- 每个seed的Control/Treatment 20个epoch parent顺序SHA-256全部相等；三组初始CPU logits hash相等。该结果验证了paired初始化和采样顺序，不改变既有`cuda_deterministic_algorithms=false`边界。
- seed42/43/44 Treatment均未通过第一阶段，`passing_seed_count=0`。三seed中位Treatment：Accuracy=`0.9439655`、Macro-F1=`0.8568105`、DBB P/R/F1=`0.9803922/0.9569378/0.9685230`、CU P/R/F1=`0.6785714/0.8260870/0.7450980`、DBB→CU=`9`、CU→DBB=`4`。Control中位Macro-F1=`0.8615434`、CU P/R/F1=`0.6666667/0.8260870/0.7547170`。Treatment没有稳定Pareto收益，第一阶段失败，第二阶段状态=`blocked_by_stage_one`。
- Validation中DBB=209、CU=23（Clutter=8、Unknown=15）。若要CU P/R均≥0.90，离散计数上DBB→CU和CU→DBB基本都必须≤2；三Treatment best分别为`8/2`、`9/4`、`9/4`，瓶颈主要是CU Precision而非DBB整体门槛。

20 epoch与残差审查：

- 实际运行`F:\Project\RadarInsight\.venv\Scripts\python.exe -m report.rd_physical_residuals --project-root F:\Project\RadarInsight`。六条20 epoch轨迹共120个arm-epoch，无任何epoch同时达到DBB floor≥0.95和CU floor≥0.90。Treatment在DBB达标约束下的每seed最佳CU floor为`0.7241379/0.6785714/0.6785714`；因此失败不是仅由best checkpoint选错造成。
- 三seed合计Treatment修复6个parent并新增3个错误，但变化不稳定；Track 3154在seed42修复、seed43回退。Treatment三seed共同错误为Bird 2367/2956/333/3581/444/845/851/852，以及Unknown 2862/4171。Track 333 RD覆盖7/22，其余共同错误RD完整，RD缺失不是整体根因。
- 五类语义路由中，Drone和Balloon几乎/完全正确，Clutter 8/8在全部seed正确；主要矛盾是Bird→CU和Unknown→DBB。该Gate结果不是五分类specialist或最终cascade性能。
- 四维特征是CNN已消费RD的确定性压缩，只增加物理归纳偏置。Train final-event Unknown/Bird四个单特征directionless AUC约`0.5346–0.5785`；validation final-event约`0.5218–0.5513`。Clutter/Bird分离相对更强，解释Clutter路由较稳，但不能解决Unknown异质性。新列权重非零但相对基础Fusion列较小，不能把“已学习”解释为“有效改善”。

DBB-knownness和validation oracle诊断：

- 实际执行`F:\Project\RadarInsight\.venv\Scripts\python.exe -m report.liquid_knownness --project-root F:\Project\RadarInsight --batch-size 4`。仅用train的Drone/Bird/Balloon拟合三个液态最终状态中心；Unknown不建中心，Clutter不混入DBB knownness。validation不参与拟合。
- `P(CU)`的三seed中位AUC：CU/DBB=`0.97462`、Unknown/DBB=`0.96587`、Unknown/Bird=`0.88846`；最近DBB中心余弦距离分别约`0.97400/0.96619/0.87949`，没有提供明显独立排序增益。共同误拒Bird通常同时位于train DBB的`96–99%`高距离尾部，直接采用“大距离→CU/Unknown”会进一步拒绝这些困难Bird。
- 在报告v2中对`P(CU)`、Energy、余弦中心距离和标准化中心距离穷举validation离散切点，但不输出阈值。四种分数、三个seed全部`stage_one_feasible=false`。当CU Recall≥0.90时，`P(CU)`最少仍需DBB→CU=`8/18/13`；Energy为`29/18/62`；余弦距离为`9/20/15`。这仅是validation oracle可达性诊断，绝不能用作部署阈值或无偏性能。
- oracle诊断首次运行暴露“DBB floor≥0.95候选集合可能为空”的报告边界，已修为显式`DBB_floor_at_least_0_95_feasible=false`且最佳点为null；未修改模型/checkpoint/data/config。随后将含oracle字段的报告schema升级为`RADAR-LN3-V11-liquid-dbb-knownness-diagnostic-v2`并重跑通过。

统计解释与复现纪律（academic-research-suite experiment-agent validate）：

- 11/11谬误扫描已完成。未发现把聚合趋势反向覆盖分组趋势的Simpson反转；分组结果已单独报告。没有从parent级结果推断单window个体关系；window与parent指标分开。validation是固定CQ08 V3选定样本，存在选择/分布外推限制，但没有据此声称生产泛化。没有控制碰撞变量或进行因果回归。
- Base-rate风险明确：CU仅23/232，CU Precision受209个DBB基数强烈影响；不能只看CU Recall。没有pre/post极端组选样，回归均值不适用。六run均完整，无训练run幸存者筛除；但未运行test，不能把validation“幸存”结果外推。
- 多重查看/Look-Elsewhere与Forking Paths风险为High caution：比较3 seed×2 arm×20 epoch、多个指标和四种oracle分数；预冻结门槛和全轨迹报告避免挑最好点，但oracle只能标为探索诊断。实验是配对随机初始化的模型干预，可支持“在当前validation/协议下Treatment未形成稳定收益”，不能因果外推到所有RD物理特征或所有液态架构；反向因果不适用于该受控模型输入干预。
- 复现状态为`PARTIALLY_REPRODUCIBLE/VERIFIED_WITHIN_SCOPE`：best checkpoint前向指标精确复现、数据/顺序/hash一致；未重跑随机训练，且CUDA严格确定性关闭，因此不能宣称训练位级复现。三个seed共享同一validation，是优化重复而不是三个独立数据样本。

最终产物与hash：

- Paired报告：`src-v3/processed/report/RADAR-LN3-V11-rd-physical-paired-validation.json`，SHA-256=`dc36eb83478a69483689eb60ffd86176f4e21543d4291234beffc97fda6021b1`。
- 残差报告：`src-v3/processed/report/RADAR-LN3-V11-rd-physical-residual-analysis.json`，SHA-256=`f41fe799e94d718a102bf02585e0efc01c1de033dd8e2cb3b6af24e1944ce1db`。
- Parent变化表：`src-v3/processed/report/RADAR-LN3-V11-rd-physical-parent-transitions.csv`，SHA-256=`489c2f79912f0a6893c4a3ab274606215e3cce4d298259071de977a2849f0727`。
- Knownness/oracle报告：`src-v3/processed/report/RADAR-LN3-V11-liquid-dbb-knownness-diagnostic.json`，schema v2，SHA-256=`aead2e24458a508f75f2efe1e537dcd617108cfd0b5fa92a8d1c45911f33f34e`。

本Stage没有创建/修改data、config或checkpoint，没有启动新训练、OOF、阈值拟合、specialist/cascade评估、test、stable或P95。V11正式训练由用户此前完成；当前第二阶段不能放行。下一候选若获新授权，应只讨论“DBB三亚类辅助表征/度量目标”这一单因素，使困难Bird留在已知DBB流型，同时保持Unknown不建普通原型；不增加液态复杂度，不直接部署静态距离阈值。

最终工程校验：`python -m compileall -q .`、三个报告入口`py_compile`、`python -m smoke.static_isolation_check --root F:\Project\RadarInsight\src-v3`和本Stage受影响文件的`git diff --check`均退出码0；静态隔离`violations=[]`。Knownness v2 focused契约确认3 seed×4分数全部`stage_one_feasible=false`，`test=未执行`、`threshold_fitting=未执行`，报告SHA-256与上述记录一致。Stage 043已完成，结论为“不能进入第二阶段”。

## 2026-08-09 Stage 044 - V12 DBB三亚类辅助表征paired实施准备（启动记录）

用户要求根据既有实验和问题进入下一阶段。本阶段采用Stage 043唯一保留候选：V11 Control主链关闭四维RD物理Treatment，在液态序列状态增加Drone/Bird/Balloon三亚类辅助监督；实施边界为代码、V12配置、focused测试和短时Smoke，不由代理启动正式20 epoch训练。

启动只读核验：

- `git status --short`确认工作树包含用户和既有RADAR-LN3历史修改；本阶段不清理、不回退、不覆盖无关变更。`src-v3/radar_ln3/task_models.py`、`src-v3/train/tasks.py`已有V11未提交实现，后续修改必须在其上手术式叠加。
- 已完整读取三份跟踪文件、V11三seed Control/Treatment配置、V10配置、V8 DBB specialist配置、V11 paired/knownness报告结论和现有Smoke入口；全局已用最大版本为V11，V12尚不存在。
- CQ08 V3 parent分布实际重算：train为Drone=488、Bird=243、Balloon=245、Clutter=35、Unknown=70；val为105/52/52/8/15。为避免2:1:1的Drone基数弱化Bird/Balloon辅助约束，V12复用现有DBB specialist基于同一train parent计数的固定逆频率权重`[0.6666667,1.3388203,1.3278912]`；权重显式写入配置、加载器严格校验且正式objective直接消费，不进行权重搜索。
- V11 validation固定232 parent，其中DBB=209、CU=23。V11 Treatment三seed中位DBB P/R/F1=`0.980392/0.956938/0.968523`，CU=`0.678571/0.826087/0.745098`；共同残差为8个Bird→CU及2个Unknown→DBB。Clutter 8/8稳定，说明下一变量应保护DBB亚类表征，而不是继续强化Clutter物理规则。

冻结V12设计：

- 数据继续复用`data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-index.csv`、V10 observation-increment数据契约与V2 train-only归一化统计；不生成V12数据文件、不访问test payload。
- Control与Treatment均配置同构3-way辅助头并使用相同随机初始化；Control辅助权重固定0，Treatment固定0.20。除arm输出路径、counterpart引用和该权重外，模型、数据、主Gate class weight、学习率、batch、20 epoch、无early stopping、seed/DataLoader顺序、checkpoint选择和评估必须完全一致。
- 辅助目标仅来自已核实base label中的Drone/Bird/Balloon，并只监督每个parent最后一个合法因果状态。实现复用既有train-only parent计数488/243/245导出的固定逆频率权重`[0.6666667,1.3388203,1.3278912]`且不搜索；Clutter与Unknown不参与辅助loss，不建立原型，也不被当作第四个knownness类。
- 辅助头只用于训练表征约束；主Gate仍输出DBB/CU，validation与checkpoint选择仍只看parent级Gate弱类P/R及Macro-F1护栏；在线点接口不暴露辅助logits。
- 首轮不叠加center loss、SupCon、Energy、Mahalanobis、dropout、状态正则、阈值或V11四维RD分支；辅助权重不搜索。正式验收仍使用预冻结Stage 1门槛与三seed paired改善，不能用单seed、单epoch或validation oracle放行。

设计修正记录：初始草案曾考虑对全部合法prefix施加辅助CE；代码审查发现这会把完整parent亚类标签静默施加给证据不足的早期prefix，重新引入窗口目标冲突。已在创建V12配置和运行任何Smoke前改为final-state-only，并同步覆盖计划与发现。该修正没有运行或产生实验结果。

实施和验证结果：

- 在`src-v3/radar_ln3/task_models.py`实现V12 schema、同构三类辅助头、显式opt-in辅助logits、DBB final-state-only CE、CU `-1`排除、共用正式objective、三seed paired构造、严格checkpoint与epoch边界恢复。在线点接口保持原主Gate输出，不暴露辅助logits。
- 在`src-v3/train/tasks.py`接入V12正式objective；Control权重0时total严格等于main且辅助头无梯度/更新，Treatment权重固定0.20。日志分别输出total/main/dbb_aux、权重、实际DBB final state和排除CU parent数量；checkpoint选择仍只看主Gate parent弱类P/R和Macro-F1护栏。
- 创建六份V12配置，seed42/43/44各Control/Treatment一份。配置复用V3 index、V10 observation-increment contract和V2 train-only统计；V11四维RD物理缓存/输入关闭。六配置SHA-256依次为Control42=`5a43cb...e0`、Treatment42=`afa7ba...76`、Control43=`b9e150...52`、Treatment43=`446612...85`、Control44=`859384...05`、Treatment44=`378d06...b6`。
- CPU Smoke首次以60秒时限遍历全部1081 parent做顺序检查而超时；确认没有formal或Smoke半成品后，将顺序验证收敛为固定10个真实parent并使用五类各最短真实parent构造混合batch。第二次运行推进到checkpoint恢复时暴露`expected_contract`初始化顺序缺陷，修复后同时纠正V12误套V11 `rd_physical_*` checkpoint字段。两项均为实施期测试问题，不是性能结果。
- 最终CPU/CUDA Smoke均passed。真实五类batch辅助标签为`[2,1,-1,0,-1]`；CU梯度精确0，早期/padding状态梯度精确0；Control辅助头梯度/参数变化0；Treatment辅助-only梯度到达辅助头与共享TR/RD/液态表征。主/辅助shape分别`[5,15,2]`/`[5,15,3]`。
- 离线/在线主Gate最大logit误差CPU=`3.278255e-7`、CUDA=`1.192093e-7`，均在对应容差内；在线输出无辅助字段。checkpoint主/辅助logits逐值恢复，optimizer与next_epoch=2恢复通过。
- V11 Control baseline等价护栏：同seed下V12 Control的42个共享state初值、初始主logits、主objective及一次AdamW主任务更新后均与V11 Control逐值一致，V12辅助头保持不变。因此Control可承接既有baseline，而非另一个随机实验。
- 冻结`src-v3/report/dbb_auxiliary_paired.py`：训练后独立重载六best checkpoint重算validation，保留既有2/3 seed+三seed中位数主门槛、同epoch护栏和离散Pareto；另要求至少2/3 seed及中位Bird→CU减少≥1且Unknown→DBB不增加。合成focused测试通过，并证明危险Unknown接受增加时会拒绝；缺少正式artifact时报告入口失败关闭。
- 最终readiness=`ready_for_user_manual_paired_training`；六个artifact目标互不重复且全部为空。readiness SHA-256=`2ab6806f6531a5feee67e4785d2a4d5764143554e64cf71e18803223b2775509`；CPU/CUDA结果SHA-256分别=`9d7f6849...603f`、`2a946dc3...d7d`，报告门槛结果=`ce2c428c...c0c`。
- 最终`compileall`、V11 paired报告回归、logging回归、V12报告门槛、静态隔离和限定`git diff --check`均退出码0；隔离`violations=[]`。没有修改`src/`，没有生成或修改data文件，没有覆盖既有artifact。

版本记录：V12仅代表新的配置/模型训练语义，故全局版本由V11递增到V12；输入数据、index、split、特征schema和归一化未发生变化，继续复用V3/V10/V2，不创建虚假的V12数据版本。六份配置属于同一V12变更批次，分别记录seed与Control/Treatment用途。

当前状态（Stage 044结束时）：V12实施准备完成并达到用户手动训练状态；后续六次正式训练和评估已由Stage 045闭环。无需预训练。test、OOF阈值、正式级联、online stable和P95仍未执行。

## 2026-08-09 Stage 045 - V12 DBB三亚类辅助表征paired正式评估

用户确认六次V12训练已完成并授权开始评估。本Stage只读取V12 formal artifact和冻结CQ08 V3 validation；未访问test payload，未启动训练/重训、OOF、阈值拟合、级联、stable或P95。

产物完整性核验：

- 六个`artifacts/RADAR-LN3-V12-...-{control|treatment}-seed{42|43|44}-formal/`均存在，分别含`best_model.pt`、`last_model.pt`、`training-summary.json`、`training.log`和`sample_detail_information.json`；每份summary均`status=passed`、`stop_reason=epoch_limit`、history=20，且独立epoch目录含20个checkpoint。
- Control best epoch为seed42/43/44=`18/20/15`，Treatment为`18/14/15`。六份summary均记录`formal_pretraining=not_required_and_not_used`、`test_evaluation=not_executed`。
- 三个seed的Control/Treatment初始CPU logits hash逐seed相同，20个epoch的parent顺序SHA-256逐epoch完全相同；`cuda_deterministic_algorithms=false`，因此仅能声明共享初始化和采样顺序配对，不能声明训练位级复现。

冻结paired评估：

- 实际执行`python -m report.dbb_auxiliary_paired --project-root F:\Project\RadarInsight`，CUDA独立重载六个best checkpoint，对固定CQ08 V3 validation的232 parent/4474 causal-prefix window前向；六组best指标均与各自training summary逐值一致。
- seed42 Control→Treatment：DBB P/R/F1 `0.980583/0.966507/0.973494`→`0.975845/0.966507/0.971154`；CU `0.730769/0.826087/0.775510`→`0.720000/0.782609/0.750000`；错误`7/4`→`7/5`（DBB→CU/CU→DBB），Gate失败。
- seed43：DBB `0.985149/0.952153/0.968370`→`0.961905/0.966507/0.964200`；CU `0.666667/0.869565/0.754717`→`0.681818/0.652174/0.666667`；错误`10/3`→`7/8`，Gate失败。
- seed44：DBB `0.975490/0.952153/0.963680`→`0.975610/0.956938/0.966184`；CU `0.642857/0.782609/0.705882`→`0.666667/0.782609/0.720000`；错误`10/5`→`9/5`，Gate失败。
- 三seed中位Control→Treatment：DBB P/R/F1 `0.980583/0.952153/0.968370`→`0.975610/0.966507/0.966184`；CU `0.666667/0.826087/0.754717`→`0.681818/0.782609/0.720000`；Accuracy `0.943966`→`0.939655`，Macro-F1 `0.861543`→`0.843092`。中位DBB→CU由10降至7，但CU→DBB由4升至5。
- Treatment三个seed的DBB三指标均≥0.95，但CU三指标均未达到0.90；generic Gate通过`0/3`，中位门槛失败。目标机制通过`0/3`：Bird→CU减少分别`0/3/0`，Unknown→DBB减少分别`-1/-4/0`，中位为`0/-1`。因此`v12_retention_passed=false`、状态=`completed_but_gate_failed`。

全epoch与机制诊断：

- 首次临时PowerShell全轨迹读取使用了错误JSON层级并产生null/索引错误，该输出被明确丢弃，未写报告或跟踪结论。随后从六份summary实际`history[].validation.parent_metrics`失败关闭重算：120个arm-epoch中可行epoch为0。DBB floor≥0.95约束下每run最佳CU floor，Control为`0.730769/0.666667/0.642857`，Treatment为`0.720000/0.652174/0.666667`；失败不是best checkpoint选优偶然，禁止事后切换epoch。
- 新增只读`src-v3/report/dbb_auxiliary_diagnostics.py`。第一次设计同时扫描三seed train+val，在240秒工具时限内未完成且进程被终止，没有生成报告、没有修改checkpoint；随后收敛为三份Treatment best checkpoint的validation前向，并从既有summary读取训练辅助loss轨迹，42.6秒完成。
- 三seedvalidation DBB三亚类辅助头Accuracy=`0.913876/0.909091/0.904306`，Macro-F1=`0.913168/0.902351/0.896312`；中位Accuracy=`0.909091`、Macro-F1=`0.902351`。训练辅助loss由epoch1约`0.955/0.962/0.970`下降至best Gate epoch约`0.101/0.123/0.128`，但训练loss只说明优化，不代表开放集性能。
- 关键机制：Treatment seed42/43/44分别有`7/7/8`个Bird→CU，而辅助头对这些22个seed级错误全部仍预测为Bird。这证明共享状态/辅助头已保留明显Bird亚类证据，但二类Gate边界没有可靠利用该证据；继续简单提高辅助CE权重缺乏机制依据。
- Treatment seed42/43/44分别有`5/7/5`个Unknown→DBB；它们在三亚类辅助头中主要被强制投影为Bird，且部分置信度很高。由于Clutter/Unknown从未参与辅助loss，这不是knownness/拒识分数，不能用辅助softmax直接路由Unknown。
- 三seed共同残差：Treatment共同Bird→CU为Track `2367/2956/333/845/851/852`（6个），共同Unknown→DBB为`2862/3154/4171`（3个）；Clutter→DBB、Drone→CU和Balloon→CU无三seed共同错误。相较Control，共同Bird错误由7降6但共同Unknown错误由2升3，不构成安全Pareto改进。

统计/复现审查（academic-research-suite experiment-agent validate）：

- 11/11谬误扫描完成。未发现聚合结论掩盖方向反转；window、parent和语义残差分开报告。Base-rate风险显著：validation仅23个CU（8 Clutter、15 Unknown）对209个DBB，CU Precision对少数Bird误拒高度敏感。
- 三seed复用同一232-parent validation，是优化随机性重复而非三份独立样本；不能据此计算独立样本置信区间。六run×20 epoch与辅助诊断构成多重查看，冻结门槛和全轨迹否定性审计降低但不能消除Look-Elsewhere/Forking Paths风险。
- 该配对干预支持的结论仅为“固定CQ08 V3 validation和冻结协议下，权重0.20的final-state DBB辅助CE未形成保留收益”；不能外推为所有辅助学习、所有开放集方法或所有液态网络均无效。best checkpoint前向已精确复算，随机训练未重跑，复现状态为`VERIFIED_WITHIN_SCOPE/PARTIALLY_REPRODUCIBLE`。

最终产物：

- Paired报告`src-v3/processed/report/RADAR-LN3-V12-dbb-auxiliary-paired-validation.json`，SHA-256=`edc06c7e6d67b7670e467d05aeb988e9c171aeac0194dc26b2e9bd8bb185ec74`。
- 机制诊断`src-v3/processed/report/RADAR-LN3-V12-dbb-auxiliary-diagnostic.json`，SHA-256=`2c1bf5d9ade5ee0575b92b81e51ad2fec401d65b760863411a8a9ada43d4d955`。
- 新增`src-v3/report/dbb_auxiliary_diagnostics.py`；未修改data、config、checkpoint、`src/`或既有模型产物。

结论：V12 Treatment归档为失败的受控消融，不进入OOF、正式级联、test、stable、P95或部署。下一阶段不能仅提高辅助loss权重，也不能把三亚类softmax当拒识；若另行授权新实验，应围绕“已知亚类证据如何显式影响Gate，同时对Unknown保持开放集负证据/拒识边界”冻结一个新变量，而不是增加液态层数或复杂单元。

## 2026-08-09 Stage 046 - V12隐藏状态三亚类knownness只读诊断（启动记录）

用户同意继续。当前只授权诊断与实验依据判断，不授权V13代码、配置、训练、OOF、阈值拟合、test、级联或部署。本Stage使用academic-research-suite experiment-agent validate纪律，要求11/11统计谬误扫描并明确固定validation与小support边界。

启动核验：

- 已读取Stage 045 paired/辅助诊断、V11 `report/liquid_knownness.py`、V12六配置与artifact状态。现有V11 knownness已证明全局DBB中心距离没有优于`P(CU)`，但它没有对V12 Control/Treatment做同seed三亚类paired比较。
- 工作树仍包含用户及既有RADAR-LN3未提交修改；不清理、不回退、不覆盖。Stage 046新增内容只允许位于`src-v3/report`、`src-v3/smoke`和三个跟踪文件；不修改`src/`、模型、训练、data、config或checkpoint。
- 冻结输入单位为parent final causal state。每个run独立用train的Drone/Bird/Balloon拟合三类中心；Clutter/Unknown不建中心。Validation只评分且不进入任何拟合。Test继续封存。
- 冻结四分数及方向：`P(CU)`、Gate Energy、最近三亚类中心余弦距离、最近三亚类中心对角Mahalanobis距离，越高越偏CU/未知。对角尺度固定由train DBB三类各自去中心后的pooled per-dimension方差计算，只做数值floor，不搜索协方差收缩参数。
- 冻结报告比较为CU-vs-DBB、Unknown-vs-DBB、Unknown-vs-Bird、Clutter-vs-DBB、Clutter-vs-Bird AUC，以及Bird→CU和Unknown→DBB分数在train DBB参考中的百分位。不得输出validation阈值或oracle切点。
- 冻结设计证据门槛：距离候选须在至少2/3 seed及中位上，Unknown-vs-Bird AUC相对同seed Control与Treatment `P(CU)`均增加≥0.02；Treatment中位Unknown-vs-DBB AUC≥0.90；CU-vs-DBB AUC相对Treatment `P(CU)`降低≤0.01。通过只代表可以讨论一个单因素V13，不代表性能、拒识或部署放行。

实施与运行：

- 新增`src-v3/report/dbb_knownness_paired.py`和`src-v3/smoke/dbb_knownness_paired_test.py`。诊断对每个Control/Treatment run独立重载best checkpoint，扫描train=1081、val=232 parent；每run用train DBB三亚类`488/243/245`拟合中心及类内pooled对角方差，显式排除train CU=105。模型、数据、配置、checkpoint和`src/`均未修改。
- 合成focused test首先通过AUC方向/tie、CU对中心/方差零污染、距离方向、通过样例和不稳定样例门槛。为使报告可独立复算，后续新增统一CSV；旧报告尚无CSV时focused test按设计失败，属于过渡契约状态，不是数据/模型失败。
- 首次代理正式运行以batch=8在376.8秒完成，核心JSON状态passed且门槛失败。随后为增加统一CSV重跑时，在约304秒以无异常文本的退出码1结束且未发布CSV；只读核验确认旧JSON完整、无残留Python、无Windows应用崩溃事件、GPU无OOM证据。根据用户要求，耗时正式入口交由用户手动执行。
- 用户手动完成正式诊断；最终JSON时间为2026-08-09 08:41:08，统一CSV同步生成。报告SHA-256=`9673237abad74a81b73a9c74abedd1f56b6371f66a99a221a0a25967c9d62517`；CSV 1392行、SHA-256=`bfc64849382d7772d92848ec6d4342fc287f6d9ea3ac32aad9d454611557829b`。六组均232行且parent/arm/seed唯一，标签每组为105/52/52/8/15。
- 最终focused test读取正式JSON/CSV，验证train-only参考、两arm/三seed、无阈值/训练/test，独立复算`3 seed×2 arm×4 score×5 comparison=120`个AUC单元并与JSON逐值一致；状态passed，SHA-256=`5be29a75c1990dabc9f639e367963006c4810f282fb473c68bd8b64c88d224fb`。

正式结果：

- Control三seed中位AUC：`P(CU)`的CU-vs-DBB=`0.976285`、Unknown-vs-DBB=`0.967783`、Unknown-vs-Bird=`0.891026`；余弦三亚类距离=`0.975660/0.969059/0.888462`；对角Mahalanobis=`0.975660/0.970335/0.896154`。
- Treatment `P(CU)`中位=`0.968379/0.958214/0.876923`。Treatment余弦距离=`0.942168/0.924721/0.823077`，相对Treatment `P(CU)`在Unknown-vs-Bird下降`0.053846`、CU-vs-DBB下降`0.026212`；相对Control同分数的Unknown-vs-Bird中位下降`0.047436`。
- Treatment对角Mahalanobis=`0.941752/0.925359/0.821795`，相对Treatment `P(CU)`在Unknown-vs-Bird下降`0.055128`、CU-vs-DBB下降`0.026628`；相对Control同分数的Unknown-vs-Bird中位下降`0.026923`。
- 两个距离候选在三个seed的设计证据检查均`0/3`通过；`eligible_candidates=[]`、`v13_design_evidence_gate.passed=false`。尽管Treatment Unknown-vs-DBB中位仍>0.90，但最关键Unknown-vs-Bird增量和CU-vs-DBB护栏均失败。
- 困难Bird→CU在距离上通常处于train DBB的约95%–99%异常百分位；Unknown→DBB也常在约85%–97%且与其重叠。辅助头“最像Bird”是相对三选一方向，不意味着绝对接近典型Bird流型；单一中心距离无法同时救回困难Bird并拒绝异质Unknown。

统计解释（11/11谬误扫描）：未发现聚合/分组方向反转被隐藏，Control/Treatment和五种语义比较均分开报告；没有从parent结论推断window性能。Base-rate风险仍为High：每组仅15 Unknown、8 Clutter、52 Bird，AUC每一个parent都会产生可见变化。三seed共用同一validation，不是独立样本。四分数×五比较×六run属于多重查看；结果前冻结候选和门槛、完整报告全部分数，降低但不消除Look-Elsewhere/Forking Paths。该诊断是post-hoc表征分析，不能支持“距离导致失败/成功”的广义因果结论，也不能外推所有度量学习；只能拒绝当前V12隐藏状态上的两个预定义静态距离候选。

结论：不支持原型/距离耦合V13，不继续在同一隐藏状态上搜索更多距离或validation阈值。V13代码、配置、训练、OOF、test、stable、P95和部署均未执行。下一阶段如继续，应先回到输入与目标覆盖：分析Unknown内部子流型和困难Bird的时序/RD证据覆盖，设计不把Unknown当普通闭集类的训练目标；仍需另行冻结单一变量和用户授权。

最终工程收口：`python -m compileall -q .`退出码0；`python -m smoke.static_isolation_check --root F:\Project\RadarInsight\src-v3`返回`violations=[]`；focused导入搜索无`src`绕过；限定`git diff --check`和尾随空白检查通过（仅Git提示LF未来可能转换为CRLF，不影响当前内容）。最终报告/CSV/focused test SHA-256分别为`9673237abad74a81b73a9c74abedd1f56b6371f66a99a221a0a25967c9d62517`、`bfc64849382d7772d92848ec6d4342fc287f6d9ea3ac32aad9d454611557829b`、`5be29a75c1990dabc9f639e367963006c4810f282fb473c68bd8b64c88d224fb`；CSV 1392行，120个AUC单元全部独立复算一致。Stage 046已完成。

## 2026-08-09 Stage 047 - 三层软级联产品语义与关键Bug修复启动

用户要求暂停V13原型/距离knownness路线，转向Bug修复，并再次确认产品不是一次性五分类：Gate识别DBB、未接受者路由CU；DBB specialist条件识别Drone/Bird/Balloon；CU specialist识别Clutter、未接受者形成Unknown拒识动作。修复完成后必须停下请求确认，正式训练由用户手动执行。

只读检查已执行：读取`git status --short`、`src-v3/radar_ln3/{losses,task_models,rejection,stable,cascade,data,model}.py`、`src-v3/train/tasks.py`、`src-v3/report/cascade.py`、V8/V12配置、现有Smoke和三个跟踪文件；使用`rg`定位loss、checkpoint、reject、stable、dt和级联调用链。工作树已有用户/既有变更：`src-v3/report/dbb_knownness_paired.py`和两个历史Smoke结果为modified，V12 knownness JSON/CSV及多个`tmp/`目录为untracked/不可读；本Stage不清理、不回退、不覆盖这些内容。

核验结论：用户列出的10项中第1–6、8–10与实际代码一致；第7项需精确表述为“V3 index保留`tr_dt_reason`，但`PointSequence`/collate/模型未传递该字段，因此首点与秒级重复时间在dynamics中被同化”。`LiquidSoftCascade`当前确实使用Gate与两个specialist概率乘法形成五类soft posterior，但`OnlineLiquidSoftCascade`先做五类stable再做全局拒识；`RejectionPolicy`只有五类max probability；`evaluate_cascade`明确写`supports_validated_rejection=false`。因此现状是三模型软概率组合，不是产品定义的三层动作级联。

版本检查：`config/RADAR-LN3/`和`data/processed/`全局最大版本均为V12，V13不存在。Stage 046已明确禁止原型距离V13；本Stage若完成loss、dt输入、checkpoint与动作契约实质变化，将V13重新定义为“三层动作级联修复批次”，并明确不继承被否定的原型距离设计。计划复用V3 index、V10数据契约和V2 train-only normalization，不重建数据；创建任何V13配置前仍需再次核对路径/hash。

已冻结实施边界：保留五类soft posterior供概率审计；新增独立hierarchical action；真实Unknown标签、CU条件输出、Unknown拒识动作和pending状态字段分离；主损失对齐parent-final，prefix仅可作为显式低权重辅助；三任务同batch/epoch联合优化并使用产品级级联候选选择；Gate/DBB/CU保持条件监督；分层阈值仅接受train parent-grouped OOF校准，未校准时不得输出已验证拒识；dt reason仅进入dynamics控制，不进入普通分类Fusion。正式训练、OOF、threshold fitting、test、P95、部署、数据重建均未执行。

## 2026-08-09 Stage 047 - 三层软级联产品语义与关键Bug修复完成

实施范围严格限定于`src-v3/`、`config/RADAR-LN3/`和`data/processed/`中的V13语义契约；未修改`src/`，未清理工作树中的用户/历史修改，未读取不可访问的`tmp/`测试目录，未覆盖V8/V12 checkpoint或artifact。Stage 047收口时正式训练、OOF、test、P95和部署均未启动；用户随后在Stage 048手动完成正式监督训练，其他未执行项不变。

实际实现：

- 新增`hierarchical_final_state_nll`：每个parent只以最终合法因果状态承担主损失，Gate和实际路由内specialist形成分层NLL；Unknown→DBB specialist梯度为0，DBB→CU specialist梯度为0。prefix只作为权重0.10、最小3事件且不含final的显式辅助监督。
- 新增唯一V13正式入口`src-v3/train/cascade.py`：三个独立液态模型在同一全parent batch、同一optimizer step联合更新；每epoch每parent恰好一次。`train.tasks`明确拒绝V13，防止退回三个局部任务各自训练/选优。
- 产品checkpoint按Gate、DBB、CU、Drone/Bird安全floor及Unknown→DBB/Bird→CU错误方向排序，三task best只在同一产品候选胜出时一起写入。五类soft posterior仍输出供审计，但已从ranking key移除。
- checkpoint恢复新增V13 config semantic SHA-256、共同epoch/global step、同一产品selection及32字符`checkpoint_set_id`；同名但配置语义被改、缺selection、半套或跨run混拼均fail-closed。Joint checkpoint同时恢复三模型、optimizer、Python/NumPy/Torch/CUDA/DataLoader RNG和epoch边界run state。
- `tr_dt_reason`贯通V3 Dataset→collate→三个模型→online；编码为padding=0、首点=1、同秒重复=2、正间隔=3。Dataset校验每个parent事件索引从0连续、首点reason恰好一次、reason/valid/raw-dt一致；online新状态必须从first reason开始，已有状态禁止再次接收first reason。首点采用初始化更新、同秒采用有界观测、正dt采用连续时间更新；reason不进入分类Fusion。
- 新增强类型分层动作：Gate接受DBB或路由CU，DBB specialist条件输出Drone/Bird/Balloon，CU specialist输出`CLUTTER_ACCEPT_EVIDENCE/CLUTTER_REJECT_EVIDENCE`；只有train parent-grouped OOF校准后才允许Clutter接受或`UNKNOWN_REJECTED`。真实`UnknownTarget`训练标签只监督Clutter-reject evidence，不等于开放集泛化。
- 新增`DBB_PENDING/DBB_ACCEPTED/CU_PENDING/CLUTTER_ACCEPTED/UNKNOWN_REJECTED`状态机。pending或冲突动作立即撤销具体stable类别；Unknown拒识stable不携带class label。V13在线历史`window_label_*`字段置空，soft posterior迁至显式`soft_posterior_audit_*`字段，产品消费者只读取hierarchical action/stable字段。
- 校准artifact除要求`fit_split=train/grouping=parent/prediction_protocol=out_of_fold`及validation/test未参与外，还必须绑定V13三配置语义hash、三checkpoint文件hash、共同epoch/global step和product selection hash；同数据版本下旧模型阈值不能套用到新checkpoint。
- 评估分别输出soft posterior审计、diagnostic hierarchy、Gate、条件DBB、条件CU、calibrated action、stable action和dt reason。无真实校准时Coverage/FAR/FRR/Selective Risk/stable保持未评估。
- 修复动作指标分母：Accepted Macro只平均四个可接受known类；真实Unknown误接收仍计入对应known Precision假阳性及Accepted Accuracy。Drone/Bird Precision使用全体parent的预测假阳性，Recall将pending/rejected/接受为其他类全部计为假阴性；pending confidence为null。

版本与数据：

- 复用`data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-index.csv`，SHA-256=`a5d054f044709047938c88ef5601caf7d4b482ef980a5a61f19db87b6e649f87`；1546 parent、29663点，split parent为1081/232/233。
- 复用`data/processed/RADAR-LN3-V2-cq08-tr-rd-train-normalization-stats.json`，SHA-256=`584eddcaeb8d78677f0f4902d622c2aa313d9374b767a47982c12dc058f826b6`；train-only统计未改变。
- 新建`data/processed/RADAR-LN3-V13-cq08-hierarchical-cascade-data-contract.json`，最终SHA-256=`d7fe425b78d71784d9cbc3a8c800acab7fa5dc156a0d16481ae9c94e9877b4a1`。它以V6 causal-quality contract为只读上游，新增reason-aware dynamics、三层动作、联合训练、CU证据命名与soft-posterior审计语义；不改变index行、split、RD tensor或normalization。
- 三配置最终文件SHA-256：Gate=`e337a63dd285b22c38f123d1f39cd080d636852e7b2005d7bff4395419f4fba8`；DBB=`5bb20be19f50bdc3f8d8be53dcc82b5e050c5d48a5926f7f0e258f29a0f5c150`；CU=`b43f315c2065c0511bfb9929747783ae41f4539cf07fcec4261b0d865e943dff`。
- 三配置语义SHA-256：Gate=`cd0573ac2b33fe2b1a7b6603d48497ed6ed9686fc274172c44dc21ba670ca9ba`；DBB=`f92ed188a3b74b222f1ea45fa04819dfc877044e5837d59c30b68ff95f5b2eb7`；CU=`591fb4869375a395e03a7c958f9837d432f6f4b44ba4bf069cff0cda38bf45fc`。
- train标签parent为Drone/Bird/Balloon/Clutter/Unknown=`488/243/245/35/70`；validation=`105/52/52/8/15`。dt reason全量计数为首点1546、同秒9776、正间隔18341；1546个parent事件索引全部从0连续且无重复首点reason。29663行均有至少一个TR源值，因此当前`valid_point_count`定义为已接收indexed TR observation。

实际测试：

- `python -m smoke.hierarchical_contract_test`通过。主损失早期prefix梯度0，Unknown→DBB梯度0，DBB→CU梯度0；首点与同秒状态最大差`0.011045694351196289`；重复首点、fresh-state非首reason、continued-state再次首reason、reason/valid不一致均被拒绝。校准来源泄漏与不同checkpoint绑定均被拒绝；四known类全对时Accepted Macro-F1=1.0，Unknown→Bird假阳性及Bird拒识均进入正确分母。结果SHA-256=`f14bdc84d664dbaa38b8d79b01192aaa0d0865488dc9cad9dc3389f22c6ddcf2`。
- 真实CQ08有界CPU Smoke通过：Bird+Unknown两个train parent、各最多3事件、1次联合更新；三模型梯度/参数变化均非零；三task与joint checkpoint恢复最大差均0；离线/在线概率最大差`4.470348358154297e-08`；动作/stable逐事件完全一致；Stage 047执行Smoke时正式artifact尚未创建，后续Stage 048已由用户训练生成。结果SHA-256=`6595aa4b3b2db8c22eef11476ff75d5c36294c191589c7c1fcd573329eb6410a`。
- 真实CQ08有界CUDA Smoke通过，除浮点差异外同CPU契约；离线/在线概率最大差`8.940696716308594e-08`；结果SHA-256=`d2444adabfa6148655a7c0ce156aac71c87daf7023adfe63e0def8396804ccf7`。
- 两个Smoke均验证同名配置语义漂移checkpoint被拒、缺产品selection checkpoint被拒、历史online窗口分类字段在V13禁用、soft posterior仅从audit字段读取、动作记录包含概率/边际/质量/触发原因。
- V7/V8历史核心回归通过；六个现有V12 best checkpoint只读严格恢复通过。未重跑历史训练或评估，未修改历史checkpoint。
- 创建`src-v3/smoke/RADAR-LN3-V13-hierarchical-training-readiness.json`。它只放行用户确认后的监督训练，不放行OOF、test、拒识或部署。

Stage 047交付时未执行与边界（当前状态由Stage 048更新）：

- 正式V13 20 epoch监督训练：本Stage未执行；用户随后在Stage 048手动执行完成。
- 正式validation性能：本Stage未评估；Stage 048已独立复算。Smoke loss/梯度/合成动作指标始终不是分类性能。
- train parent-grouped OOF calibration、真实Gate/Clutter阈值、Coverage、Accepted Accuracy/Macro、FAR、FRR、Selective Risk、stable性能：未执行/未评估。
- test、session、P95、部署：未执行；当前verified index无session/capture字段。
- 跨split重复RD/capture provenance仍未裁决；允许受控监督训练，但阻塞任何新增泛化或部署结论。
- 用户确认前不输出正式训练命令、不创建正式artifact；用户确认后已使用`python -m train.cascade`三配置联合入口完成训练。

## 2026-08-09 Stage 048 - V13正式训练后冻结Validation评估

用户报告V13联合训练已运行完毕。本Stage只读取正式artifact、冻结三配置、V3 validation和既有人工审计报告；执行validation-only checkpoint恢复评估。未访问test、未拟合阈值、未运行train OOF、未启动重训、stable、P95或部署。

正式产物与训练完整性：

- `artifacts/RADAR-LN3-V13-cq08-liquid-hierarchical-cascade-formal/training-summary.json`记录`status=passed`、device=`cuda`、20 epoch、10820 global steps、elapsed=`3635.200574`秒；训练日志从`run=start`到`run=complete`闭合，精确扫描未发现Traceback、CUDA OOM或non-finite错误。
- best为epoch 13/global step 7033；Gate、DBB specialist、CU specialist和joint best共享`checkpoint_set_id=611c0817ef0f42afb9ccf9c8658e9602`。三task checkpoint SHA-256依次为`3e6c7cf52bba73a1d5528cb30a20e39ee2b98f4599a55fe685e99f46570a6dec`、`53075d1bc49f3e307964c46c0b80f6f57f6afe2b15f6390b2042f6a99fb090b5`、`b40dd82adc684041d6a52668c3428bc6bf9c19938fa1da536741e67f0cd435aa`；joint best为`9e322bac6c174de9bd733de958996350a1fb72d5b0ca518b14a15976bda60e22`。
- summary SHA-256=`99428aa4d3299ab79c5fef920b5a12ca96c04e5f595f826c35db64903ee9bb96`，training log SHA-256=`e1091b8957306e7124eb4ddea77ff0e63238c159c475a367e5ad5a7c61e07ffb`。没有覆盖V8/V12 artifact。

冻结validation独立复核：

- 首次以batch=8执行`python -m report.cascade`，三checkpoint严格加载并完成232 parent/4474 event的CUDA前向。所有分类指标与training summary一致；唯一报告差异是collate padding计数由416变为1334。
- 随后按正式配置batch=2再次执行同一入口。training summary的best validation与独立报告在schema、样本数、产品语义、diagnostic hierarchy、soft posterior、Gate/DBB/CU组件指标、checkpoint selection input、未校准动作、stable、dt reason和session状态上逐值一致；两侧规范化核心SHA-256均为`04b2905edf1b02c71d7f7a44ea6bbd83cf7217cd1e48418a22e8f72f8fd190da`。
- 规范报告为`src-v3/processed/report/RADAR-LN3-V13-hierarchical-cascade-validation-batch2/evaluation-val.json`，SHA-256=`9b8aae57050c3d1a6283a96a0f0e301758b67b58f553fc78c1a6a68835f51d77`；对应`sample_detail_information.json` SHA-256=`e93b3062ad5b763ab50757729fece8b1bbddf29990501fa0039b0b0c0db30fd5`。复现状态为`VERIFIED_WITHIN_FROZEN_FORWARD_SCOPE`；随机训练未重跑，不能声明训练位级复现。

真实监督结果：

- 未校准diagnostic parent-final五类：Accuracy=`0.840517`、Macro Precision=`0.670816`、Macro Recall=`0.709286`、Macro F1=`0.685081`。Drone P/R/F1=`0.989583/0.904762/0.945274`，Bird=`0.732143/0.788462/0.759259`，Balloon=`0.980392/0.961538/0.970874`，Clutter=`0.416667/0.625000/0.500000`，Unknown标签诊断=`0.235294/0.266667/0.250000`。该Unknown结果是未校准标签诊断，不是`UNKNOWN_REJECTED`动作。
- Gate parent矩阵为`[[195,14],[8,15]]`：DBB P/R/F1=`0.960591/0.933014/0.946602`，CU=`0.517241/0.652174/0.576923`。Gate总Accuracy=`0.905172`，但不满足DBB与CU各自P/R门槛。
- 条件DBB specialist矩阵为`[[95,10,0],[0,52,0],[1,1,50]]`，Macro P/R/F1=`0.938327/0.955433/0.943338`。Bird Recall=1.0但Precision=`0.825397`；因此“DBB整体较强”不能替代每类Precision门槛。
- 条件CU specialist矩阵为`[[7,1],[7,8]]`，Macro P/R/F1=`0.694444/0.704167/0.651515`。Clutter P/R/F1=`0.500000/0.875000/0.636364`，Unknown标签证据=`0.888889/0.533333/0.666667`；CU边界仍是主要监督瓶颈。
- best产品护栏floor为Gate=`0.517241`、DBB=`0.825397`、CU=`0.500000`、Drone/Bird=`0.732143`，通过`0/4`。20个epoch各自通过数均为0；全轨迹各护栏最高也仅`0.517241/0.852459/0.583333/0.732143`，不存在可合法事后换取达标的epoch。
- epoch 13最小归一化完成度=`0.544465`，高于epoch 15的`0.540541`；epoch 15虽diagnostic Macro-F1=`0.717829`高于epoch 13，但冻结排序优先最弱产品护栏，故不得事后切换。训练总loss从epoch 1的`1.451318`降至epoch 20的`0.344242`，而epoch 20最小完成度降至`0.138889`；已确认优化损失下降不等于产品路由改善，具体原因仍待训练集内归因，不能直接认定液态容量不足或经典过拟合。

方向错误与既有人工审计残差：

- Gate有14个DBB→CU：Bird 11、Drone 2、Balloon 1；8个CU→DBB：Unknown 6、Clutter 2。最终diagnostic错误37/232，其中Bird主要去Unknown标签诊断9个，Unknown主要去Bird 5个或Clutter 5个。
- 12条既有人工审计轨迹全部保持原标签。V13只纠正Track 852（Bird→DBB/Bird）和2862（Unknown→CU/Unknown标签诊断）；Track 333、845、2367、2956仍高置信Bird→CU，Track 3581仍Bird→Clutter，Track 1547、2908、3122、3154、4171仍Unknown→DBB。10/12持续残差说明三层目标修复没有消除这些具体边界问题。

统计与解释边界（academic-research-suite experiment-agent validate）：

- 完成11/11谬误扫描。Simpson：当前正式validation仅CQ08一个source，不能检验跨source反转，window/parent已分开；Ecological：结论限定parent，不从window总量外推目标；Berkson：冻结validation是既有筛分且被历代版本反复使用，存在选择风险；Collider：本轮未按质量变量筛样或回归控制；Base-rate：209 DBB对23 CU，风险显著；Regression-to-mean：非前后极值组设计；Survivorship：232 parent全部进入，无已知丢弃；Look-Elsewhere：20 epoch和历代validation反复查看构成高风险，但冻结选优阻止事后换epoch；Forking Paths：V8–V13已有多路线，当前仅作描述性复核；Correlation/Causation：V13同时改变目标、dt语义和动作契约，不能把指标差异归因给单一修复；Reverse Causality：不适用于当前监督分类描述。
- CU support极低：Gate CU Precision 95% Wilson区间约`[0.3443,0.6861]`、Recall约`[0.4489,0.8119]`；Clutter Recall=`0.875`但区间约`[0.5291,0.9776]`；Unknown标签Recall=`0.5333`但区间约`[0.3012,0.7519]`。不能用点估计宣称接近门槛。
- 该训练只有seed 42；V13不是与V8/V12共享单一变量的paired实验。正式结果可证明当前V13实现未达标，不能证明所有parent-final目标、reason-aware dt、液态网络或分层拒识均无效。

发现的报告口径问题：

- `dt_reason_counts_including_padding.padding`统计的是每个DataLoader batch内为对齐该batch最长序列而产生的collate位置，所以batch=2为416、batch=8为1334；分类输出完全一致。该字段名称容易被误解为数据集固有padding总数。
- 本Stage只记录该问题，未修改代码。后续应将其更名为明确的`collate_padding_positions`并记录batch size，或另算与batch无关的数据契约统计；它不影响mask后的V13当前分类指标。

当前决策：V13状态为“正式训练完成、技术契约通过、监督性能未放行”。由于四项基础护栏为0/4，且固定validation已反复使用，本Stage不继续train OOF阈值校准；阈值不能修复Bird→CU与Unknown→DBB的表征重叠，也不能把未达标监督模型包装为已验证拒识。下一阶段应先在train/parent-grouped OOF内部做错误覆盖、Unknown子流型、困难Bird和loss贡献归因，并冻结单一变量后再请求授权。test、calibrated action、stable、session、P95和部署均未执行。

## 2026-08-09 Stage 049 - V14严格Open-set启动与Git基线

用户确认：若当前不是严格open-set，则先按`.rules`保存Git记录，再拆分阶段并按`task_plan_v3.md`实施；只有当前已是严格open-set时才转入错误覆盖归因。只读代码核验已确认V13不是严格open-set：Gate把Unknown映射CU参与CE，CU specialist把Unknown作为第二目标学习，Gate接受DBB后DBB三分类argmax无二次拒识；真实train OOF calibration也尚不存在。因此本Stage进入V14严格open-set改造，不执行替代分支。

Git规则与基线：

- 已完整读取`.rules`。当前起点为受保护分支`main`，规则禁止直接提交；创建普通开发分支`codex/v13-strict-open-set`。Git成功切换分支，但仓库`.git/hooks/post-checkout`无法spawn导致命令退出码1；分支实际存在且当前分支核验通过，未修改用户hook。
- 提交前运行V12 knownness focused、V13 hierarchical focused、`python -m compileall -q .`和静态隔离，均通过；受限`tmp/`历史目录继续存在权限警告但未纳入提交。
- 基线提交`d4d39e2 保存V12已知性诊断基线`：仅包含V12 knownness诊断、逐parent分数、focused测试与结果；明确OOF/阈值/正式训练/test未执行。
- 基线提交`627950a 保存V13三层级联修复基线`：包含V13代码、三配置、V13数据契约、跟踪文件、focused/CPU/CUDA Smoke结果和规范batch=2 validation复核；未提交正式`.pt`、`tmp/`、`__pycache__`或batch=8重复报告。提交使用普通开发分支中文概述、空行和有序列表，符合`.rules`。

严格open-set污染审计：

- V13 train parent=`1081`，其中Unknown=`70`。`hierarchical_final_state_nll`对Unknown产生Gate CU CE和CU specialist第二类CE；Unknown还进入每epochvalidation与`unknown_to_dbb_count` checkpoint ranking。这是已确认的可学习与模型选择暴露。
- V13 Gate接受DBB后无`dbb_accept_known_threshold`或DBB knownness拒识，直接接受三分类argmax；Unknown误路由DBB时必被强制映射Drone/Bird/Balloon。
- V2 RD normalization使用全部1081 train parent，包含70个Unknown。即使不使用Unknown标签做CE，继续复用该统计也会让Unknown分布影响输入尺度，不满足本项目冻结的“Unknown零学习暴露”严格定义。
- V14 known-only train population应为Drone/Bird/Balloon/Clutter=`488/243/245/35`，总计1011；Unknown train 70必须从normalization、Dataset、optimizer、OOF、阈值和checkpoint selection全部排除。Validation Unknown=15只允许冻结development open-set审计；test仍封存。
- 当前全局最高版本为V13。训练population、normalization、label/loss、输出动作、calibration和checkpoint语义均实质变化，因此使用全局V14；复用V3 index/split/TR/RD张量，不覆盖V2/V13文件。

冻结V14动作设计：

- Gate只用DBB与Clutter known训练`DBB_ACCEPT_EVIDENCE/ROUTE_TO_CU_EVIDENCE`；Unknown不属于Gate训练CU类。
- DBB specialist只用Drone/Bird/Balloon训练三分类；运行时除亚类argmax外必须通过独立DBB knownness probability/margin边界，否则直接`UNKNOWN_REJECTED`。
- CU specialist不学习Unknown类，输出`CLUTTER_ACCEPT_EVIDENCE/NON_CLUTTER_KNOWN_EVIDENCE`；正样本Clutter，负样本只来自DBB known。Gate路由CU后仅Clutter接受条件通过才输出Clutter，否则`UNKNOWN_REJECTED`。
- 三个接受边界只允许由train known parent-grouped OOF校准，并以known false-rejection/coverage约束拟合；Unknown不能用于选择阈值。真实Unknown检测能力只能在冻结validation/test报告，不作为训练或调参依据。
- `UNKNOWN_REJECTED`是动作，不是模型class、中心或训练target。Unknown标签仅用于开放集评估真值及FAR/拒识Recall分母。

该段记录的是Stage 049启动时状态；V14实施与验证的当前事实由下方“Stage 049完成”覆盖。正式训练、完整OOF、真实阈值拟合、正式拒识、stable性能、development Unknown validation审计、test、P95和部署仍未执行。

## 2026-08-09 Stage 049 - V14严格Open-set实现、OOF准备与回归完成

本阶段只修改`src-v3/`、`config/RADAR-LN3/`和`data/processed/`中的V14实现/契约；未修改`src/`，未修改标签，未重建V3 index/split/TR/RD tensor，未访问test payload，未启动正式训练、完整OOF训练或真实阈值拟合。用户/历史未跟踪的V13报告和`tmp/`权限受限目录保持原样，不纳入本阶段提交。

严格open-set实现：

- 在`radar_ln3/task_models.py`新增V14强类型TaskSpec：Gate只学习DBB known与Clutter route evidence；DBB specialist只学习Drone/Bird/Balloon；CU specialist学习Clutter与DBB known非Clutter evidence。`UnknownTarget`不存在于任何V14 class mapping、prototype、概率输出或loss target。
- 在`data.py`为Dataset增加allowed/forbidden label及parent白名单；V14 train、known checkpoint validation和OOF均显式禁止Unknown。Unknown进入train Dataset、loss、checkpoint selection或calibration会fail-closed。
- 在`losses.py`实现`strict_open_set_final_state_nll`：known parent final state为主损失，prefix仅按既有0.10/最小3事件契约作辅助；Unknown label id=4立即报错。Clutter不会给DBB specialist产生梯度，DBB known进入CU的non-Clutter-known监督，Unknown梯度计数恒为0。
- `train/cascade.py`支持V13/V14 schema-aware联合训练。V14每epoch只使用217个known validation parent做产品checkpoint ranking，ranking key长度为8；Unknown development审计只在best checkpoint冻结后运行且不能返回selection输入。`train/tasks.py`继续拒绝三层V13/V14走单task正式入口。
- `rejection.py`实现三道known-only接受边界：Gate DBB route、DBB knownness、Clutter acceptance。Gate通过而DBB knownness失败与CU中Clutter未接受均输出`UNKNOWN_REJECTED`，accepted label为空；无真实OOF calibration时动作保持`NOT_CALIBRATED`。
- `cascade.py`分别输出known evidence、hierarchical action与stable状态；第五项命名为`UNRESOLVED_OPEN_SET_EVIDENCE`且明确不是Unknown posterior。V13/V14均将数据事件dt reason与collate padding拆分为`dt_reason_counts_data_events`、`collate_padding_positions`和`evaluation_batch_size`，修复历史batch依赖字段语义。
- `report/cascade.py`按V13/V14 schema加载对应校准，强制校准数据hash和正式三checkpoint部署身份一致；未授权test仍被拒绝。

版本化数据和配置：

- 只读复用V3 index，SHA-256=`a5d054f044709047938c88ef5601caf7d4b482ef980a5a61f19db87b6e649f87`；split与点行不变。
- 新建V14 known-only train normalization，路径`data/processed/RADAR-LN3-V14-cq08-known-only-train-normalization-stats.json`，SHA-256=`1484f58521be677b0a5f992e40fd8af558660da2c8f764669f151d9d969194b0`。只使用1011个known train parent，标签计数Drone/Bird/Balloon/Clutter=`488/243/245/35`；排除70个Unknown，selected/excluded RD tensor路径交集=0；center/scale=`55.9375/6.945900266082501`。
- 新建V14数据契约`RADAR-LN3-V14-cq08-strict-open-set-data-contract.json`，SHA-256=`062b67f7c9b33b682a55ce90a6b41e5d66bafada374e149ae46c02392a68f378`。V3 index、split和RD tensor均只读复用；训练population、normalization、loss、输出动作、checkpoint及calibration语义实质变化，因此使用全局V14。
- 三配置SHA-256：Gate=`9d1d708cc3955c1584357a3ea5207b1076d9ad3c59cd8b99d89648c7b0faa8b3`；DBB=`23d2f158b368003b3b1c927c6cd7b14cd10d86e09d5c1d431135d571e2ac2da0`；CU=`3beddc09d60a47edf15276d2be92b8345d0a9fb2e99fdcf1c9643f99252e3af6`。正式artifact路径与V13/V12隔离，检查时尚不存在V14正式或OOF artifact目录。
- `python -m build.strict_open_set`复跑结果为`reused_identical`；基础normalization、数据契约和三配置hash均保持不变。实施中发现normalization构建器曾会在未使用parent白名单时写入两个null字段并破坏基础文件幂等性，已改为仅在真实传入白名单时写入`allowed_parent_ids_count/sha256`。

Parent-grouped OOF准备：

- assignment manifest为`RADAR-LN3-V14-cq08-known-only-parent-grouped-oof-manifest.json`，SHA-256=`1aea914c9566a8495471590ea5ebf9d6b62da6a6788b25aa7b39c5e6ebf7a46f`。使用seed 42、按label内SHA-256顺序round-robin分5折；1011个known train parent各恰好一次held-out，Unknown OOF=0，validation/test parent=0。
- fold contract升级为v2，路径`RADAR-LN3-V14-cq08-known-only-parent-grouped-oof-fold-contract.json`，SHA-256=`8a6404fb7933bb8de29d561986494fde628f15c998c92fedd8b6b6b1088ce559`。每折固定20 epoch、关闭early stopping、held-out不参与checkpoint选择、只在最终epoch后单次预测。
- 为避免fold模型实际使用fold-train normalization却仍声称绑定基础1011-parent统计，新增5份fold-specific数据契约。Fold 00–04 contract SHA-256依次为`eb96b6eb8c49951f4459f9aebd1166e5ae9716759a3a1c8a3ee0b3c1a3c7b0bc`、`44626724546d82f48ef8620e3cd487754a63d33d6c2047435b6e22560de82e07`、`e5e6ee7a330a0fbf14db884d0fbbcb86b47f74075980dd846b1b21628ebbc9de`、`7180a2bbb94df9c369ea45eaaf7d7471b679f3c6984d12e14422181514e0fa4c`、`d8753c55e40709186888f7504ba48a9b76d71a63a507a111d298a06b1b410d4e`。
- 5份fold-train normalization SHA-256依次为`7e6c0aace0a77fe90e9e59e21ebcf067db57b56e1482ed13f8cbce5a147219ed`、`a223ae8f2f1d7cbebc0509e7bf57597e697dc217cbc74cc6244f3958f56ca2d2`、`72cfc0188371b3dfd8ee3fc93b439166e9c1b6b1e4aa503276b0416bb7fe06c3`、`0724a6552780b94906e0d45ad7644b0c9dfb7d5055fd3ce412632e19a2004f0f`、`f00e2423b96baf919db938273ce89b956d3166da4cfdb93065ef42f8ac1bd6d8`；fold 00–02 train/held-out=`808/203`，fold 03–04=`810/201`，所有fold Unknown/validation/test暴露均为0。
- 新增`train.strict_open_set_oof`：每次只训练指定fold，20 epoch内从不前向held-out；最终epoch保存校准支持checkpoint后对held-out执行一次parent-final预测。折模型明确不是正式部署checkpoint。
- 新增`report.strict_open_set_oof`：只有5份预测完整覆盖1011个known parent后才拟合Gate/DBB/Clutter边界；只优化known Precision/Recall、coverage和known false rejection，不声称优化Unknown检测。Calibration必须绑定正式V14三配置语义hash、三checkpoint hash、共同epoch/global step、selection hash及数据/OOF契约hash。

实际测试与回归：

- V14 focused契约测试通过，结果`src-v3/smoke/RADAR-LN3-V14-strict-open-set-focused-contract-result.json`，SHA-256=`9ea23d1e08590bd9c232e9fc163822af450ceef37771c7e10f4cc4b467627f80`。验证train known=1011、validation known=217、development audit=232/其中Unknown 15；Unknown进入loss/selection/calibration均被拒；V14 checkpoint round-trip通过；5折白名单合计恰好覆盖1011且互斥；synthetic known-only拟合通过，混入Unknown立即失败。
- 有界真实CQ08 CPU Smoke通过，结果SHA-256=`2dd012a828e10a1b4a558608e19c05a8390feb5777133a9bed27a14739a974af`；Bird+Clutter各最多3事件，三模型梯度/参数更新非零，checkpoint恢复最大差0，离线/在线evidence最大差`5.960464477539063e-08`。
- 有界真实CQ08 CUDA Smoke通过，结果SHA-256=`fe7aab7fd3b9aa188a2c56806c97ac9ece37477ae371c8d3f23cc4b80b16d546`；checkpoint恢复最大差0，离线/在线evidence最大差`4.470348358154297e-08`。
- 实施中发现`load_cascade`历史固定要求9维ranking key，V14实际为8维；已按schema分别校验V13=9、V14=8。V13正式三best checkpoint随后在CPU只读恢复成功，epoch=13、global step=7033、共同set ID=`611c0817ef0f42afb9ccf9c8658e9602`。
- V13 hierarchical focused通过；V13 bounded Smoke未重跑，因为其防覆盖护栏正确拒绝在正式V13 artifact已存在时执行。该情况不属于代码失败；正式三checkpoint只读恢复作为兼容性证据。
- 历史`smoke.causal_cascade_contract_test`通过；全`src-v3` compileall通过；静态搜索无`src/` import、`sys.path`、动态加载绕过；限定`git diff --check`通过。
- 第一次扩展focused测试时，新helper插入位置截断`_checkpoint_contract`导致输出`checkpoint=null`；测试结果未被采纳。函数结构修复后重新执行，checkpoint round-trip与全部V14/OOF断言完整通过。
- 创建`src-v3/smoke/RADAR-LN3-V14-strict-open-set-training-readiness.json`。该文件只证明等待用户确认后的手动正式known-only训练和5折OOF技术准备，不是性能或部署放行。

当前执行边界：

- V14正式20 epoch known-only联合训练：未执行。
- 5折OOF训练及held-out预测：未执行。
- 真实known-only OOF阈值拟合和calibration artifact：未执行。
- development Unknown validation审计、正式拒识指标、Coverage/FAR/FRR/Selective Risk、stable性能：未执行。
- test、session、P95和部署：未执行。
- 预训练：不需要且未使用。

当前结论：V14已经是本项目冻结定义下的严格known-only open-set实现，Unknown不是模型类、训练target、normalization/optimizer/OOF/threshold/checkpoint ranking population；运行时只能由三道known acceptance边界产生`UNKNOWN_REJECTED`动作。但真实拒识能力仍“未评估”，必须等待用户确认并手动完成正式训练、5折OOF和冻结校准后才能讨论。

## 2026-08-11 Stage 051 - V14 OOF断点续跑一键脚本

用户要求：不使用环境变量，在`src-v3/scripts/`创建一个可直接执行的脚本，串行完成剩余OOF、校准和一次冻结后的development validation五类开放集评估；任一步失败必须停止。长时间任务由用户手动执行，本阶段不得代跑完整OOF。

只读核验：

- V14正式联合训练产物存在且`training-summary.json status=passed`，best epoch=`12`、global step=`10120`。正式三checkpoint SHA-256为Gate=`b6fcff98e282709f0fb10d9d2ab447615436635f3b8e82d762a302f490ce462b`、DBB=`b60a2e35c150866073c5d652be93d580df144939182c6b5eff652232d1d33687`、CU=`e8ba8ba540a66759631b93cd6177a6d14602b9f6f7122b674e5477695a3da0c2`。
- OOF fold 00/01各已完成20 epoch并各写出203个held-out parent；脚本预检逐epoch验证每折train parent恰好消费一次、held-out最终前向次数=1、提前前向=0、Unknown/validation/test exposure=0。
- Fold 02尚无held-out prediction；只读加载`joint_last_model.pt`确认其为完整epoch=`5`、global step=`2020`、history count=`5`、checkpoint set ID=`49dc04b9152d4e8ea80be754cda001f8`。因此后续必须从该文件恢复，未完成epoch 6不计入。Fold 03/04未启动。
- Calibration文件与校准后validation报告均不存在；test和P95未执行。

实施：

- 新建`src-v3/scripts/v14_strict_open_set_oof_pipeline.ps1`。项目根目录从`$PSScriptRoot`解析；Python、config、checkpoint、OOF、calibration和report均使用直接路径，不读取或设置任何环境变量。
- 默认模式按fold 00→04串行：已有prediction先严格校验再跳过；缺prediction且有`joint_last_model.pt`则传`--resume-checkpoint`；全新fold从epoch 1开始；若目录非空又无prediction/resume checkpoint则立即停止，禁止覆盖。
- 每个Python子进程退出码非0立即停止。每折完成后校验schema、20 epoch、early stopping=false、held-out不选checkpoint、最终前单次访问策略、train/held-out parent/hash、fold normalization、三checkpoint文件hash和Unknown/validation/test零暴露。
- 五折聚合必须恰好覆盖1011个唯一known train parent且与assignment manifest逐ID一致，才运行`report.strict_open_set_oof`。Calibration必须来自`train_parent_grouped_oof`、Unknown拟合计数=0、validation/test未参与，并绑定当前正式三checkpoint。
- 最后仅执行`report.cascade --split val --batch-size 8`，写入`src-v3/processed/report/RADAR-LN3-V14-strict-open-set-calibrated-validation/`。脚本要求报告为217 known + 15 development Unknown，Unknown只映射到`UNKNOWN_REJECTED`动作，且development Unknown禁止阈值拟合/checkpoint选择。
- 增加独占锁避免重复启动；增加`-PreflightOnly`仅用于安全自检，默认不带参数仍执行完整流水线。
- 文件保存为UTF-8 BOM。原因是Windows PowerShell 5.1会把无BOM UTF-8中文脚本按系统ANSI解码；首次5.1兼容检查由此解析失败，加入BOM后问题消除，不涉及训练逻辑。

验证：

- 当前PowerShell AST解析通过，token count=`3781`。
- 当前PowerShell及Windows PowerShell 5.1均成功执行`-PreflightOnly`：fold 00/01严格校验通过，fold 02识别为可恢复，fold 03/04识别为未启动；退出码0。
- 静态搜索未发现`$env:`、环境变量赋值、`Start-Process`、`Invoke-Expression`、test或P95执行入口；脚本唯一评估split为`val`。
- `-PreflightOnly`未启动训练、校准或validation评估，未创建calibration或校准后报告。

该段记录的是脚本首次交付时的历史断点状态：当时正式V14联合训练已完成、OOF完成2/5、calibration尚未执行。此状态已被下方“完整OOF、Calibration生成与脚本后置断言修复”覆盖；当前为5/5 OOF和calibration已完成、冻结development validation待执行。

## 2026-08-11 Stage 051 - 完整OOF、Calibration生成与脚本后置断言修复

用户执行默认完整脚本后报告：fold 04已校验通过，随后脚本报“检索不到变量`$fieldName不在`”。只读核验确认该错误发生在Python calibration已成功写盘之后、冻结development validation开始之前。

实际完成状态：

- Fold 00/01/02均为20 epoch、各203个held-out parent；fold 03/04均为20 epoch、各201个held-out parent。五折status均为`passed`，Unknown OOF count、validation access和test access均为0。
- 增强预检聚合确认1011个known train parent各恰好一次OOF，无重复、无Unknown、无validation/test拟合暴露。
- Calibration已生成：`artifacts/RADAR-LN3-V14-cq08-liquid-strict-open-set-cascade-formal/strict-open-set-rejection-calibration.json`，SHA-256=`af53913420ab9293885467812d3a1f0990a74eb8a7e7826edd1d5f44e40cf4e1`。
- Calibration来源=`train_parent_grouped_oof`，known parent=`1011`，Unknown threshold fit=`0`，validation fit=`false`，test used=`false`；绑定正式deployment epoch=`12`、global step=`6072`及三checkpoint身份。
- 冻结边界为Gate probability/margin=`0.542160868644714/0.0843217372894287`，DBB knownness probability/margin=`1e-7/0`，Clutter probability/margin=`0.566431522369385/0.13286304473877`。这些是known-only OOF拟合结果，不代表Unknown性能。
- 校准后validation报告尚不存在，证明失败后没有部分执行或伪造冻结开放集结果。

根因与修复：

- 根因是PowerShell插值词法边界：消息字符串`"calibration字段$fieldName不在[0,1]"`和对应margin字符串被解析为变量`$fieldName不在`，在`Set-StrictMode -Version Latest`下触发未设置变量异常。OOF、checkpoint、calibration数值和Python入口均无故障。
- 两处改为明确边界`${fieldName}`。同时增强`-PreflightOnly`：若五折全部存在，则执行完整1011-parent聚合断言；若calibration/report已存在，则执行对应后置断言，但仍不启动训练、校准或评估。
- 当前PowerShell AST通过；Windows PowerShell 5.1增强预检退出码0，逐折验证5/5、五折聚合1011和calibration部署绑定全部通过。
- 重新执行默认脚本时，五折将只读校验并跳过；Python calibration入口会按“内容完全相同”复用既有文件；随后仅执行一次冻结development validation。无需重跑任何fold。

当前执行边界：完整OOF和真实known-only calibration已执行；冻结development validation、校准后五类开放集性能、test、P95和部署仍未执行。下一步由用户重新执行同一默认脚本；development Unknown结果不得用于反调本轮阈值。

## 2026-08-11 Stage 052 - V14冻结Development Open-set结果审计

用户报告一键脚本“全部完成”。本阶段只读回查正式V14训练、五份OOF held-out prediction、known-only calibration、冻结development validation及sample detail；未启动训练、未改阈值、未读取test payload、未执行P95或部署。

执行与产物核验：

- 冻结报告`src-v3/processed/report/RADAR-LN3-V14-strict-open-set-calibrated-validation/evaluation-val.json`存在且`status=passed`，SHA-256=`50a6387a18dc16d76a8cd1b700fac9e711cb427f08182b3cbaa7ba81e8367b18`；样本明细SHA-256=`cdbe8430f854c47d64a3c80f24ec483d9e413d753c6ca3cd556f362fb3343082`。
- 报告范围为CQ08 development validation共232个parent：217 known和15个仅审计Unknown；Unknown不是模型类别，训练暴露、OOF预测、阈值拟合和checkpoint选择暴露均为0。
- Calibration SHA-256仍为`af53913420ab9293885467812d3a1f0990a74eb8a7e7826edd1d5f44e40cf4e1`，来源为1011个known train parent的5折OOF，未使用validation/test，绑定正式best epoch 12三checkpoint；正式训练跑满20 epoch、总global step=`10120`，冻结best checkpoint的global step=`6072`，两种步数语义已分开记录。
- 五折held-out=`203/203/203/201/201`，1011个parent唯一覆盖。每折20 epoch、held-out最终单次前向、Unknown/validation/test暴露为0。

冻结development五动作结果：

- Parent-final Accuracy=`0.875000`，Macro Precision/Recall/F1=`0.770164/0.741996/0.733411`。类别P/R/F1/Support：Drone=`0.989474/0.895238/0.940000/105`；Bird=`0.714286/0.961538/0.819672/52`；Balloon=`0.980392/0.961538/0.970874/52`；Clutter=`0.500000/0.625000/0.555556/8`；`UNKNOWN_REJECTED`动作=`0.666667/0.266667/0.380952/15`。
- 混淆矩阵真实行为：Drone→Bird=`11`；Unknown→Bird=`7`、→Balloon=`1`、→Clutter=`3`，仅4/15 Unknown被拒识。Unknown FAR=`0.733333`，known FRR=`0.009217`，接受Coverage=`0.974138`，Accepted Accuracy=`0.880531`，Selective Risk=`0.119469`。
- Stable确认228/232，4个未确认。仅在228个确认样本上Accuracy=`0.885965`、Macro-F1=`0.729919`；若按产品验收把4个未确认保守计为未命中，Accuracy=`202/232=0.870690`、Macro-F1约`0.721353`、Unknown Recall=`3/15=0.200000`。因此stable没有修复开放集问题，且只报告确认样本会产生幸存者偏差。

OOF与校准复算：

- 使用冻结阈值直接从五份held-out records独立重算，合并known OOF Accuracy=`0.915925`、Macro-F1=`0.827326`、known coverage=`0.986152`，与calibration内置汇总逐值一致。类别P/R/F1：Drone=`0.985356/0.965164/0.975155`；Bird=`0.871795/0.839506/0.855346`；Balloon=`0.906977/0.955102/0.930417`；Clutter=`0.629630/0.485714/0.548387`。
- 分折Accuracy=`0.945813/0.940887/0.852217/0.935323/0.905473`，Macro-F1=`0.800196/0.895717/0.767150/0.904802/0.766726`；Fold 2明显较弱，证明折间异质性不可忽略。
- Gate阈值=`0.542161/0.084322`，OOF最弱P/R=`0.487805`；DBB knownness阈值=`1e-7/0`，最弱P/R=`0.907336`；Clutter阈值=`0.566432/0.132863`，最弱P/R=`0.542857`。三项`all_precision_recall_targets_passed`均为false；校准来源契约有效，但性能门槛不通过。

已确认瓶颈：

- DBB knownness阈值贴近数值下界，等价于几乎全放行。15个Unknown中8个先进入DBB路由，这8个全部被接受为Bird/Balloon；其DBB top probability中位=`0.978122`、最大=`0.990667`。Unknown错误不是简单低softmax置信度问题。
- Clutter train仅35、development仅8；OOF Clutter Recall=`0.485714`，折间波动大。当前证据支持样本覆盖/子流型和语义边界优先，不支持继续提高class weight或扩大液态容量。
- DBB specialist仍有11个Drone→Bird，Bird高Recall以Precision下降换取；这是DBB内部边界问题，不是Gate拒识单独造成。
- OOF折模型固定epoch 20，正式部署使用best epoch 12。该协议不构成数据泄漏，但可能产生分数尺度迁移；下一OOF需与正式checkpoint选择协议一致。
- Calibration和流水线当前只验证来源、数值范围与部署绑定，即使三项P/R目标均失败仍输出结构性`passed`。必须增加`deployment_eligible=false`或`audit_only`语义，避免把“文件有效”误读为“性能放行”。

统计审查：

- 按ARS experiment-agent validate流程完成11/11统计谬误扫描。主要限制为类别基率失衡、Clutter/Unknown低support、固定development validation被多版本反复查看、折间异质性、stable确认样本幸存者偏差和OOF/正式epoch协议差异。
- Unknown拒识Recall仅4/15，Wilson 95%区间约`[0.1090,0.5195]`；Clutter Recall仅5/8，约`[0.3057,0.8632]`。点估计和区间均不支持90%放行，也不能从15/8个样本外推部署泛化。
- 未发现需要用因果语言解释的实验设计；当前结论限定为该冻结CQ08 development范围。跨split RD/capture provenance旧风险仍未因本轮流水线自动消除。

当前执行边界：V14正式训练、5折known-only OOF、calibration、冻结development层级拒识和stable评估已执行；test、P95和部署未执行。V14冻结为“严格open-set协议正确、性能失败、不可部署”基线；不得使用这15个development Unknown反调本轮阈值、epoch或模型。

## 2026-08-11 Stage 053 - V14校准质量状态与部署Fail-closed

本阶段处理Finding F156，不改变模型性能或阈值。修改范围仅为`src-v3/radar_ln3/rejection.py`、`cascade.py`、两个report入口、focused/Smoke辅助代码、V14 PowerShell流水线和三份跟踪文件；未修改`src/`、数据、配置、标签、正式checkpoint、原calibration或冻结validation结果。

实现：

- V14 calibration schema新增v2。新产物必须显式写入`calibration_quality_status`、`deployment_eligible`和`quality_gate`；quality gate固定要求Gate、DBB knownness、Clutter acceptance三项OOF Precision/Recall门槛全部通过。
- Loader兼容v1/v2。v1不得因缺少新字段而默认为通过，必须从已有`selection_diagnostics.*.all_precision_recall_targets_passed`机械推导；v2字段必须与诊断逐项一致，否则拒绝。
- Loader新增用途边界，默认`deployment`。当前真实V14三项均失败，因此默认加载抛出`PermissionError`并列出`gate/dbb_knownness/clutter_acceptance_precision_recall_target_not_met`。只有显式`audit`可读取。
- 策略层增加第二道防线：audit-only校准默认不能启用；仅offline development审计可显式构造。Online cascade无论调用方是否显式构造audit策略都拒绝注入。
- `report.cascade`默认deployment；显式audit仅允许V14 validation。新审计报告分离`execution_status=passed`与`status=audit_only`，并写`deployment_eligible=false`、质量来源和失败原因。V13路径不接受audit purpose且原逻辑保持不变。
- `report.strict_open_set_oof`未来创建v2；若目标已有历史v1且阈值/provenance/诊断完全相同，则只读复用并返回`reused_legacy_v1_with_derived_quality_status`，不覆盖历史hash。
- PowerShell脚本支持v1/v2、独立推导质量状态。已有legacy报告可只读校验；新评估显式传`--calibration-purpose audit`。最终显示“执行完成、质量audit_only”，不再把执行成功写成部署通过。

验证：

- `python -m smoke.strict_open_set_contract_test`通过；新增断言覆盖合格v2 deployment、失败v2默认拒绝、显式audit、历史v1推导、v2字段篡改拒绝、策略默认拒绝和online注入拒绝。
- 真实历史calibration探针通过：默认deployment加载被拒绝；显式audit返回`status=audit_only`、`deployment_eligible=false`，三项失败原因完整。原SHA-256复核仍为`af53913420ab9293885467812d3a1f0990a74eb8a7e7826edd1d5f44e40cf4e1`。
- 真实`report.strict_open_set_oof`只读复算通过：1011 known parent、Unknown/validation/test拟合暴露0，返回`audit_only`并复用历史v1，无文件改写。
- 正式`report.cascade`在默认deployment用途下退出码1，错误明确为audit-only不可部署；失败发生在validation前向和报告写盘之前。
- Windows PowerShell 5.1 `-PreflightOnly`通过：5/5折、1011 parent、calibration与legacy报告全部校验，并显示`audit_only/deployment_eligible=False`。
- `python -m smoke.hierarchical_contract_test`通过，证明V13路径未回归；`compileall`、`smoke.static_isolation_check`和限定`git diff --check`通过。
- V14 CPU有界cascade Smoke未执行：其既有防护发现`artifacts/RADAR-LN3-V14-cq08-liquid-strict-open-set-cascade-formal`已存在，主动抛出`FileExistsError`以避免Smoke混入正式artifact。未删除、改名或绕过该安全保护；focused与真实loader/CLI集成测试覆盖了本次改动。

版本与执行边界：本次只改变calibration/report/部署加载语义，模型、数据、split、归一化、阈值和checkpoint不变，因此全局模型版本仍为V14；未来calibration schema使用v2，历史v1只读兼容。正式训练、OOF重训、test、P95和部署均未执行。

## 2026-08-11 Stage 054 - V15 DBB角度Knownness设计冻结

本阶段先完成只读调用链、版本与实验设计审查，尚未启动正式训练或OOF。Stage 053已单独提交为Git锚点`2628049ed842af35d5aa132024a81f398534b9f9`；提交后仓库缺失`.git/hooks/post-commit`可执行文件而输出hook错误，但commit对象已成功创建。V13/V14正式报告目录和历史浮点回归结果未纳入该提交，所有正式V14 checkpoint、calibration和冻结development报告继续只读。

已核实背景：

- V14 DBB knownness直接取DBB三分类softmax的top probability和top-two margin；没有独立embedding、类原型、类内紧致约束或可解释距离。真实OOF拟合得到threshold=`1e-7`、margin=`0`，冻结development进入DBB路由的8个Unknown全部被DBB接受。
- V14严格open-set数据边界本身正确：Unknown train source虽有70 parent，但normalization、optimizer、checkpoint selection、OOF和threshold fit暴露均为0；15个development Unknown只作为冻结评估真值。本轮继续保持该边界，不把Unknown加入prototype、center、hard negative或energy训练。
- V14折模型固定epoch 20，正式模型best epoch 12。阈值虽然绑定正式checkpoint身份，但分数由不同训练阶段的折模型产生，存在尺度迁移；该问题是测量协议缺陷，不是Unknown泄漏。
- 代码中严格open-set路径大量使用`schema_version == RADAR-LN3-V14`。V15若只改配置版本而不引入显式family判断，会退回历史TaskSpec、Dataset或loss路径，因此这是实施前必须修复的fail-closed契约问题。
- 当前`build_cascade()`按Gate→DBB→CU顺序依赖全局RNG构造模型。若Treatment新增/替换DBB头而继续共享全局流，CU初始权重会偏移；同seed不足以构成真正配对，必须改为逐task派生初始化并验证非干预参数逐位一致。

冻结实验设计：

- Control：V15严格open-set、线性DBB classifier；Treatment：只把DBB classifier替换为L2归一化embedding/类原型角度头。液态隐藏维度48、TR/RD输入、dt reason dynamics、Gate、CU、class weights、prefix weight 0.10、batch 2和20 epoch预算保持一致。
- Treatment训练对真实DBB标签使用固定角度margin；推理/OOF/online必须使用无margin cosine logits。knownness分数显式定义为top cosine及top-two cosine gap，不继续把普通softmax置信度冒充距离。
- 主要结果来自known train parent-grouped外层OOF：DBB knownness最弱Precision/Recall和known FRR。Gate、Clutter、DBB三类及Drone/Bird指标是不可退化护栏。development Unknown只在Control/Treatment、epoch协议和阈值全部冻结后各执行一次审计，不能参与胜负或下一轮参数搜索。
- 嵌套OOF：outer held-out从normalization、optimizer和epoch selection完全隔离；outer-train内部选择epoch，再从冻结初始状态用完整outer-train重训到该epoch；outer held-out只前向一次。不得继续用固定epoch 20替代正式best协议，也不得直接用outer held-out逐epoch选优。
- Clutter覆盖审计保持独立路线，不与角度头同时改变；若同时改变数据和模型，则无法归因本轮收益。

版本决策：全局最高模型版本为V14。DBB readout、logit/loss、knownness score、checkpoint contract及OOF execution contract发生实质变化，因此下一版本为V15。V3 index/split/TR/RD tensor及V14 known-only normalization内容不变时复用原路径和SHA-256；parent assignment内容不变时只读复用，但新建V15 OOF execution contract与artifact。配置仍只写`config/RADAR-LN3/`，代码仍只写`src-v3/`。

当前执行边界：Stage 054计划已写入；代码、配置、契约、focused测试和Smoke尚未完成。正式Control/Treatment训练、完整嵌套OOF、阈值拟合、development复评、test、P95和部署均未执行。

## 2026-08-11 Stage 055 - V15角度Knownness实施准备、OOF防重放与人工执行入口

本阶段继续V14冻结后的单因素V15路线，只在`src-v3/`和`config/RADAR-LN3/`实施；没有启动正式训练、嵌套OOF、阈值拟合、development Unknown审计、test、P95或部署。采用实验执行与可重复性约束：每个结论只来自真实代码/contract/smoke，Unknown仍不能作为训练或设计信号。

实际修改与核验：

- 将V14单值判断改为V14/V15严格open-set family；V15 Control保留线性DBB readout，Treatment仅替换为48维L2归一化embedding/3类原型、scale=`16`、训练期target margin=`0.20`。Gate、CU、TR/RD输入、液态动态、隐藏维度、loss权重、batch和20 epoch协议未改变。
- `strict_open_set_final_state_nll`仅在DBB已知类别final/prefix loss中施加margin；forward、online、OOF与校准均消费无标签的线性logit或raw cosine score。focused实测同一合成batch control/treatment loss=`2.95758056640625/6.071575164794922`，而forward调用前后推理score/evidence最大差异=`0`。
- `build_cascade`对V15按`sha256(paired_seed:task_name)`独立构造任务，恢复全局RNG；focused实测Control/Treatment Gate和CU state逐位一致，DBB所有同名同shape共享encoder/fusion/liquid参数逐位一致，只有读出结构不同。
- 新建V15 experiment contract：`data/processed/RADAR-LN3-V15-cq08-angular-knownness-experiment-contract.json`，SHA-256=`c86626be093e3d32b4fdfb268fdbd1ddc8c72f498e4802b6c4b05c5d2b71597`。六份V15配置均通过加载，Control/Treatment artifact路径均不存在，未覆盖V14。
- 新建/复核nested OOF contract：`data/processed/RADAR-LN3-V15-cq08-known-only-nested-oof-contract.json`，SHA-256保持`bbed0a99a8623660b30b5d823e82eb5eed6165a9d84b1e2eb97ac8ca43b20326`。5个outer held-out合计1011 known parent；inner selection为下一fold、inner train为其余三fold、outer retrain为除outer held-out外四fold。所有Unknown、development validation和test访问字段为0。
- 修复`build.angular_knownness_oof`二次执行失败：早期nested contract在stage条目中记录了`write_status`，此字段只描述文件是否创建/复用，导致完全相同的合同被误判不同。新代码只在内存兼容比较时忽略该运行时字段，不改写已有合同、路径或hash；复核返回`reused_legacy_runtime_status_only`。
- 新增outer-held-out forward claim：`train.angular_knownness_oof`在正式最终前向前持久化claim；若进程中断且claim存在而prediction缺失，下一次调用fail-closed，禁止自动重跑以避免无法证明的第二次held-out观察。完成prediction后claim绑定prediction SHA-256；聚合报告要求claim、derived config identity、fold ID、1011 parent唯一性和score语义全部一致。
- 新增`report.angular_knownness_oof`，将来只在两个formal三checkpoint均完成后聚合5折nested OOF，并为Control生成概率calibration schema、Treatment生成cosine calibration schema。报告入口在checkpoint不存在时以明确`FileNotFoundError`停止；没有伪造OOF或校准产物。
- 修正`train.cascade`：V15正式训练仅对known-only checkpoint-selection集合评估，结束时不访问development Unknown；摘要字段`development_unknown_access_count_during_formal_training=0`。V14保留历史训练结束development审计行为。
- 新增`smoke.angular_knownness_contract_test.py`。真实CQ08 known-only forward使用1011 parent集合，Treatment offline/online raw cosine最大差异=`8.940696716308594e-08`、open-set evidence最大差异=`1.7881393432617188e-07`。测试拒绝Unknown进入nested Dataset、错误held-out集合、Control/Treatment calibration混用、缺失cosine score和重复outer forward。
- 新增`src-v3/scripts/v15_angular_knownness_pipeline.ps1`。PowerShell 5.1 parser与`-PreflightOnly`通过；脚本不使用环境变量，所有路径从脚本位置解析，默认模式才依序formal→OOF→calibration。任何失败停止；对已有outer forward claim且无prediction的fold拒绝自动重试。

实际验证：

- `F:\Project\RadarInsight\.venv\Scripts\python.exe -m compileall -q src-v3`：通过。
- `python -m smoke.angular_knownness_contract_test --project-root F:\Project\RadarInsight`：通过；Unknown训练暴露=0、margin训练生效/推理不污染、paired初始化、online/offline和calibration binding均通过。
- `python -m smoke.strict_open_set_contract_test --project-root F:\Project\RadarInsight`：通过；V14严格open-set历史合同无回归。
- `python -m smoke.static_isolation_check --root F:\Project\RadarInsight\src-v3`：通过，`violations=[]`。
- `powershell -NoProfile -ExecutionPolicy Bypass -File ...\v15_angular_knownness_pipeline.ps1 -PreflightOnly`：通过；只做幂等build、focused与隔离检查，未创建V15正式artifact、checkpoint、OOF prediction或calibration。

当前执行边界：V15技术准备已闭合，但没有任何真实Control/Treatment训练、nested OOF、阈值拟合或development Unknown结果。test、P95和部署均未执行。V14性能失败结论保持不变，不能被V15 smoke覆盖或替代。

## 2026-08-11 Stage 055补充 - V15正式训练核验与OOF流水线阶段修复

用户报告以`-Stage oof -Arm both`执行V15脚本时，脚本在第一条预检命令`build.angular_knownness`失败。只读定位确认：该构建入口按设计拒绝在非空V15 formal artifact上运行；但PowerShell预检此前对所有阶段无条件调用它，导致已经完成正式训练后无法进入OOF。

本次仅修改`src-v3/build/angular_knownness.py`和`src-v3/scripts/v15_angular_knownness_pipeline.ps1`：

- 为构建入口增加`--verify-only`。该模式严格比较V15 experiment contract和六份配置的冻结内容，不创建、不覆写，也不检查或修改formal artifact目录。
- 脚本在`-Stage oof`或`-Stage calibrate`时，先核验每臂`training-summary.json`及三份task best/joint best checkpoint；再调用`build.angular_knownness --verify-only`。`formal/all`仍保留原有可写准备和非空artifact拒绝覆盖保护。
- 正式训练摘要改为`Get-Content -Encoding UTF8`读取。Windows PowerShell默认代码页会破坏摘要中的UTF-8中文状态文本，从而使`ConvertFrom-Json`报“Invalid object passed in”；文件内容本身经Python和UTF-8 PowerShell读取均为有效JSON。

只读正式训练核验：Control与Treatment摘要均`status=passed`、`schema_version=RADAR-LN3-V15`、history=`20`、final epoch=`20`、global step=`10120`；Control best epoch=`19`、摘要SHA-256=`9b1be7e2ac70014920ea6da13cf03ffd5a89498565aace16c4038d3fdfc28570`，Treatment best epoch=`6`、摘要SHA-256=`ac4d2b94aa8fc8639909b69bd588566a610a5eb3c29770b0d4823cdfc0091803`。两臂`unknown_training_exposure=0`且`development_unknown_access_count_during_formal_training=0`；本轮未读取development Unknown结果。

实际验证：`python -m compileall -q src-v3`通过；PowerShell AST解析通过；`powershell -NoProfile -ExecutionPolicy Bypass -File F:\Project\RadarInsight\src-v3\scripts\v15_angular_knownness_pipeline.ps1 -Stage oof -Arm both -PreflightOnly`退出码`0`。该预检只读复核了正式checkpoint、V15合同/配置、nested OOF合同、focused contract和静态隔离；没有启动任何fold训练、outer-held-out前向、calibration、development Unknown审计、test、P95或部署。

当前执行边界：正式两臂训练已完成；用户可继续手动运行V15 nested OOF。OOF、校准、development Unknown冻结审计、test、P95和部署仍为“未执行”；不得根据formal训练摘要或本次预检宣称性能提升。

## 2026-08-12 Stage 055补充 - V15 Control嵌套OOF中断与安全恢复核验

用户报告`v15_angular_knownness_pipeline.ps1 -Stage oof -Arm both`以`Python command failed with exit code -1`停止。本次只读审计脚本、artifact、训练日志和恢复checkpoint；未启动或恢复训练，未修改正式checkpoint、OOF prediction、阈值、数据、配置或模型代码。

实际状态：

- Control outer fold 00、01均已有`outer-held-out-predictions.json`和已完成的`outer-held-out-forward-claim.json`，脚本会只读跳过；Treatment nested OOF根目录尚不存在。
- Control outer fold 02的`inner-selection/stage-summary.json`已完成。`outer-retrain/stage-summary.json`不存在，但`outer-retrain/joint_last_model.pt`存在，SHA-256=`f410fc161321f32ec15f7b267d27e0c35202804de5147f2f87cce6bb8c6b6b5b`。
- 恢复checkpoint的`training_state.run_state`记录`next_epoch=12`、`global_step=4444`、`best_epoch=12`、`nested_oof_stage=outer_retrain`、`nested_oof_arm=control`、`nested_oof_outer_fold_id=2`，history仅含已完整保存的epoch 1–11。日志最后一行是epoch 12/12 step 0200/0404；没有Python traceback、CUDA OOM或held-out forward日志。
- fold 02尚未创建`outer-held-out-forward-claim.json`或prediction。因此没有发生或声称发生outer held-out观察；脚本的恢复分支会加载该checkpoint，仅从epoch 12重新训练，完成后再进行唯一一次held-out forward。
- 审计时无活动Python进程。`-1`只证明子进程被非正常终止；现有日志不足以归因于用户终止、宿主进程终止、GPU驱动或其他外部事件，不能把它写成训练代码异常。

恢复命令仅应使用`-Stage oof`，不可用`-Stage all`，以避免formal阶段的防覆盖保护。先单独恢复Control fold 02，完成后再继续其余fold；校准、development Unknown审计、test、P95和部署继续未执行。

当前执行边界：V15 nested OOF为部分完成状态，Control仅完成2/5折，尚无5折聚合或阈值校准；不得读取development Unknown作为本轮恢复或选择依据。

## 2026-08-12 Stage 056 - V15两臂嵌套OOF完成、known-only校准与归因审查

用户完成V15剩余nested OOF。本阶段先只读展开两臂10份outer-held-out prediction与forward claim；随后执行仅聚合known train OOF并绑定正式三checkpoint的`v15_angular_knownness_pipeline.ps1 -Stage calibrate -Arm both`。该命令没有训练、没有读取development Unknown、没有访问test，也没有执行stable、P95或部署。

实际完整性核验：

- Control/Treatment均为5/5折，fold parent数均为`203/203/203/201/201`；每臂record count=`1011`、unique parent count=`1011`、跨臂parent集合差异=`0`。
- 两臂每折`outer_held_out_prediction_pass_count=1`，`unknown_training_exposure=0`、`unknown_oof_prediction_count=0`、`validation_access_count=0`、`test_payload_access_count=0`。
- known OOF类别为Drone=`488`、Bird=`243`、Balloon=`245`、Clutter=`35`；Unknown没有进入OOF或阈值拟合。
- Control inner/outer epoch为`17/18/12/19/14`；Treatment为`18/7/19/17/14`。只有fold 04相同，正式best epoch分别为Control `19`、Treatment `6`。

校准产物与状态：

- Control calibration：`artifacts/RADAR-LN3-V15-cq08-liquid-strict-open-set-angular-knownness-control-formal/strict-open-set-rejection-calibration.json`，SHA-256=`f95b08ae0f14abe4715be70b42f07eb237480c5039d924ccdfcea02883145c4b`。
- Treatment calibration：`artifacts/RADAR-LN3-V15-cq08-liquid-strict-open-set-angular-knownness-treatment-formal/strict-open-set-rejection-calibration.json`，SHA-256=`d29b04e7a900e725329d1e29f22ecd381d3407ddaa7b5fadcc795458780fee06`。
- 两份产物均`source=train_parent_grouped_oof`、known parent=`1011`、Unknown threshold fit=`0`、validation/test fit=`false`，并绑定各自正式三checkpoint身份。
- 两份质量状态均为`audit_only`、`deployment_eligible=false`；Gate、DBB knownness、Clutter三组件都未全部达到各自P/R目标。

主要性能结果：

- Control DBB knownness：coverage=`1.000000`，Macro-F1=`0.955973`，最弱P/R=`0.930041`；Drone P/R=`0.983505/0.977459`，Bird=`0.930041/0.930041`，Balloon=`0.951613/0.963265`。
- Treatment DBB knownness：coverage=`0.983607`，Macro-F1=`0.950422`，最弱P/R=`0.915323`；Drone P/R=`0.987207/0.948770`，Bird=`0.915323/0.934156`，Balloon=`0.962963/0.955102`。
- 因此Treatment相对Control的预注册primary变化为最弱P/R `-0.014719`、Macro-F1 `-0.005550`；known FRR从`0.016815`升至`0.032641`。角度knownness主假设失败。
- 完整known action的次级变化：accepted accuracy `0.933602→0.941718`，Macro-F1 counting reject as error `0.793345→0.823156`，Clutter F1 `0.385965→0.500000`；但overall known accuracy `0.917903→0.910979`，Drone Recall `0.977459→0.948770`，未满足不可退化护栏。
- Treatment DBB knownness阈值拒绝16个DBB parent，其中10个raw预测错误、6个raw预测正确；其错误富集率为62.5%，但无法在保持known Recall的同时达到0.95 P/R。
- 不改阈值的排序诊断显示：Control低softmax-gap对raw DBB错误的ROC-AUC/AP=`0.919322/0.331366`；Treatment低cosine-gap为`0.918093/0.418002`。Treatment在最低gap 16个parent中捕获10/46个raw错误，Control对应为7/37；这表示Treatment提高最极端低-gap区域的precision，但没有提高整体ROC排序，不能覆盖预注册primary失败。

用户观察到的高分裁决：

- 精确指标`Accuracy=0.969388/Macro-F1=0.963904/support=196`只出现在Control outer-04 inner-selection `history[epoch=20]`，不是外层held-out聚合，也不是Treatment结果。
- 该快照Drone全召回，但Bird和Balloon Recall均为`0.938776`，按逐类P/R≥0.95的严格门槛仍未全通过。
- outer-04实际选择epoch 14。epoch 20虽然DBB更强，但Clutter acceptance Recall由epoch 14的`0.714286`降为`0.285714`，CU floor由`0.333333`降为`0.285714`，known parent Macro-F1也从`0.798485`降为`0.788536`；产品级选择没有漏掉该高分，而是拒绝用DBB局部提升交换Clutter退化。

配对归因限制：

- 初始化focused已证明Control/Treatment非干预参数逐位一致，但正式与inner OOF在相同epoch时Gate/CU全部参数均已分叉。
- 代码使用一个AdamW联合优化三模型，并在每个step对`model.parameters()`统一执行`clip_grad_norm_`。训练日志记录点中Control `grad>100`为`4/220`，Treatment为`39/220`，最大grad分别`197.469/600.513`；配置clip norm=`100`。
- 因此Treatment角度损失会通过全模型梯度裁剪改变Gate/CU更新尺度。Treatment的Gate CU和Clutter次级改善不能写成角度几何的直接因果收益；它是联合训练动力学和不同epoch选择后的系统级结果。

实验审查纪律：采用experiment validate流程检查11类统计/方法谬误。未发现可直接做显著性检验的预注册重复seed；主要风险为单折/单epochlook-elsewhere、Clutter低base-rate、accepted subset造成selection/survivorship偏差、4/5折不同epoch与联合梯度裁剪造成因果归因混杂。结论置信级别为`CAUTION`：OOF覆盖与来源验证充分，但多seed、development Unknown、test和部署泛化均未验证。

跟踪更新：覆盖更新`task_plan_v3.md`、`process_v3.md`和`findings_v3.md`。V15 experiment/nested OOF contract内容与SHA-256未改变；本轮新增的是两个正式artifact内的calibration文件，没有创建新数据、split、normalization或配置版本。

当前执行边界：V15不需要重复训练，且不可部署。development Unknown一次冻结审计仍未执行；即使后续获准，也只能描述失败候选的开放集行为，不得反向选择Control/Treatment或调margin/scale/threshold。test、stable、P95和部署继续未执行。

## 2026-08-12 Stage 057 - V15测试冻结与产品契约失效复核

用户暂停V15产品路线并提供代码审查。本阶段只读复核V15实现、两臂正式artifact、两臂nested OOF、V3 split与指定重复parent；随后更新跟踪文件，为按`.rules`提交V15研究快照做准备。没有修改标签、V15 checkpoint、calibration、V3 index/split、RD张量或原始文件；没有启动训练、OOF、test、stable、P95或部署。

V15执行完整性：

- Control/Treatment正式训练均完成20 epoch，best epoch=`19/6`；training-summary SHA-256=`9b1be7e2ac70014920ea6da13cf03ffd5a89498565aace16c4038d3fdfc28570/ac4d2b94aa8fc8639909b69bd588566a610a5eb3c29770b0d4823cdfc0091803`。
- 当前磁盘真实状态为两臂均5/5 nested OOF完成；每臂outer 00–04 prediction和forward claim均存在。早期“Treatment仅fold 00–02”的状态已过期，不能写入当前结论。
- Control/Treatment calibration SHA-256仍为`f95b08ae0f14abe4715be70b42f07eb237480c5039d924ccdfcea02883145c4b/d29b04e7a900e725329d1e29f22ecd381d3407ddaa7b5fadcc795458780fee06`，两者均`deployment_eligible=false`。

产品契约复核：

- `STRICT_OPEN_SET_TASK_SPECS`把V15 Gate限制为DBB versus Clutter route evidence，并把CU specialist定义为Clutter versus DBB-known non-Clutter；`strict_open_set_final_state_nll`显式拒绝Unknown ID 4，CU target等于DBB mask。因此V15没有学习Clutter/Unknown标签边界，其CU指标不是产品要求的CU二分类性能。
- `StrictOpenSetRejectionPolicy.decide`在Gate接受DBB、DBB knownness失败时直接返回`DBB_ROUTE_ACCEPTED→DBB_KNOWN_NOT_ACCEPTED→UNKNOWN_REJECTED`，没有回退CU。该路径与用户冻结的三级软级联动作不一致。
- V15 train/checkpoint selection/nested OOF/calibration全部known-only，Unknown exposure为0；这对其原研究问题是事实，但不能用于验证真实Unknown标签预测或Clutter/Unknown拒识动作。development Unknown、test、stable、P95和部署仍未执行。
- parent-final为主、prefix辅助权重0.10和parent-only checkpoint selection没有证明早期prefix动作安全；window/prefix、parent-final和stable必须在V16分开报告与验收。
- `train.cascade`摘要无条件写`status=passed`，但Control/Treatment best产品护栏分别只有`1/4`与`0/4`；V16必须改为流程状态与性能状态分离。

数据隔离复核：

- V3 split声明`group_unit=track_id`，没有capture/content provenance分组。CQ08 unified manifest实际为1550条记录，其中45个capture ID各出现2次。
- 独立hash核验：Track 1339/train与1490/val共享capture `20250616122328`、raw SHA-256=`76186016842f523cc2ec29742caa6a5e94d02131c0e5c30e285711413e6c87cb`、RD NPZ SHA-256=`9cb42c2782ac3f9a00bbd4e2517f63bbf83d0abd4e030a3fd08e51461e96c1a3`；Track 1343/val与1494/train共享capture `20250616122337`、raw SHA-256=`3bb1a50f12db89f454c4aaa42f62c65220eee2c291126e0d81bcf6340288f15c`、RD NPZ SHA-256=`7e29aba776e8d76d85534bbd07feb9bae952af036f17c1ed4c88250f54e1440b`。两组track/point文件hash不同，说明应按capture/raw/RD等价组件隔离，不能按文件名简单去重。
- 45个重复capture中有10个跨标签组；逐组复核raw与RD SHA-256后，10组均为raw不同且RD不同。当前没有跨标签完全等价内容证据，不修改或申请修改标签；V16仍把同capture视为同一采集组，整体放入一个split。
- offline与online均把`tr_dt_reason==2`累计进`gap_fraction`；V3 reason 2是合法同秒观测，必须在V16拆为`zero_dt_fraction`诊断，不得触发gap guard。

提交前验证：

- `python -m smoke.angular_knownness_contract_test --project-root F:\Project\RadarInsight`：通过。
- `python -m smoke.strict_open_set_contract_test --project-root F:\Project\RadarInsight`：通过。
- `python -m smoke.causal_cascade_contract_test`：通过。
- `python -m compileall -q src-v3`：通过。
- `python -m smoke.static_isolation_check --root F:\Project\RadarInsight\src-v3`：通过，违规数0。该检查前两次分别因从项目根无法解析`smoke`模块、误用`--project-root`参数而未执行；按脚本实际工作目录和`--root`参数重跑通过，不是实现失败。
- `v15_angular_knownness_pipeline.ps1 -Stage oof -Arm both -PreflightOnly`：通过；仅回读冻结配置、contract、正式checkpoint和现有OOF完整性，未启动训练、前向、校准或development/test。
- `git diff --check`：通过，仅有Git行尾转换提示。

版本结论：V15代码、配置、实验/OOF契约和结果保留为known-only角度knownness研究快照；因标签映射、split规则、normalization population、loss、checkpoint、动作和评估语义都将实质改变，下一实现必须使用全局新版本V16。V16不得加载V15 checkpoint/calibration或复用V14 known-only normalization。

当前执行边界：V15快照待Git提交；V16 provenance、index/split、normalization、数据契约、三任务独立训练代码和配置均未创建。正式三任务训练、完整级联、OOF、development/test、stable、P95和部署均未执行。

## 2026-08-12 Stage 058 - V16真实三任务局部训练链启动

V15诊断快照已按`.rules`提交，commit=`b6cd7e879fe0a78ac15022de781042b8a23aa140`。Git在commit后报告缺失`.git/hooks/post-commit`可执行文件，但commit对象、提交信息和43文件diff均已生成；未修改或暂存用户现有`todos.md`删除与三个历史大体积报告目录。

本Stage冻结的局部训练范围：

- Gate：Drone/Bird/Balloon→DBB，Clutter/Unknown→CU；目标最弱P/R≥0.95。
- DBB specialist：只训练Drone/Bird/Balloon；目标三类各自P/R/F1≥0.95。
- CU specialist：只训练Clutter/Unknown；目标两类各自P/R/F1≥0.90。
- 三模型分别优化和选择best，不执行联合optimizer、完整级联或三checkpoint组合选优。Unknown在本Stage是CU监督标签；运行时`UNKNOWN_REJECTED`动作仍是不同语义，正式开放集拒识未执行。

数据构建设计：为1546个实际入模parent计算capture、raw、RD、原始/规范track与point、模型输入语义指纹；任一同源键相同即并入同一连通组件。组件split采用`test > val > train`保守优先级，避免任何旧val/test parent进入新train。全量raw约9.746 GiB、RD NPZ约2.548 GiB，provenance为本Stage最耗时的只读步骤；训练不自动启动。

当前执行边界：仅计划与设计已落盘；V16构建器、数据产物、配置、代码修复和Smoke均未完成。

## 2026-08-12 Stage 058 - V16实现、数据重建与训练准备完成

V16全量provenance构建已执行。构建器读取1546个V3 parent、29663个真实事件，按capture ID、raw SHA-256、RD NPZ SHA-256、去ID规范track/point SHA-256和去标签/路径/ID的模型输入语义SHA-256建立连通组件。首次构建与第二次幂等复核均完成；第二次所有V16数据与配置均为`reused_identical`，未覆盖不同内容。

数据结果：

- 1546个parent合并为1501个provenance组件；45个组件包含多个parent，27个组件原先跨V3 split。
- 组件按`test > val > train`保守迁移，27个parent发生迁移；旧val/test进入V16 train数为0。
- V16 train/val/test parent=`1055/247/244`；五类support分别为train `476/239/238/33/69`、val `113/56/54/9/15`、test `109/53/58/8/16`，顺序为Drone/Bird/Balloon/Clutter/Unknown。
- capture/raw/RD/规范track/规范point/模型输入语义六类键跨split违反数均为0；跨标签content-equivalent冲突为0。10个跨标签capture仍整体同split保留原标签，因为raw/RD内容不等价且没有改标证据。
- 新index最长parent为29、最短12，不存在`max_sequence_length=29`截断parent-final。index/contract/normalization SHA-256分别为`0af7412f0ea5963ba2bcf42bc87a3d02ee1c6fcfedfc42d68f027a6922bf466c`、`ca99a5f1b2fa77d1a2348f56ae6e74f480daad1bba5bc9749b5311a382dca373`、`4b809a36dd1ad6885b2db4aa9041b5929c835033d969e3b61d43eaeba9388013`。
- normalization只使用1055个V16 train parent并包含全五类；validation/test未用于统计，excluded parent=0、selected/excluded RD路径交集=0。

模型与训练修复：

- 新增V16 product-local schema和三配置。Gate真实映射DBB/CU；DBB specialist只含Drone/Bird/Balloon；CU specialist只含Clutter/Unknown，Unknown train exposure=69，不再使用V15的DBB-known负类替代Unknown。
- V16主目标为每parent最终合法状态weighted CE；0.10 prefix辅助仅监督event≥3且排除final，先在parent内平均，再按weighted CE分母归一。focused梯度探针确认padding和event<3非final梯度均为0。
- `train.tasks`现传递`tr_dt_reason`，三任务分别选best并绑定完整config identity/data hash；正式V16 CLI拒绝`--max-steps`、epoch override和skip-validation，仅内部Smoke入口允许有界验证。
- 训练/报告摘要把流程`status=completed|failed`与`performance_status`分开；V16逐类P/R/F1门槛为Gate/DBB 0.95、CU 0.90，局部结果固定`deployment_eligible=false`。未来`train.cascade`也不再无条件写整体passed。
- V16 product-local三checkpoint被通用cascade组装器显式拒绝；完整级联必须在后续新版本中重新开放和验收。

时间与动作修复：

- 全量dt reason为first=1546、same-second zero=9776、positive=18341；raw zero占全部事件38.1688%，其中同秒事件占全部事件32.9569%，占非首转换34.7690%。真实来源是每parent首点无前驱和CQ08秒级时间精度下重复时间戳，不是padding或人为改写。
- 同秒reason仍执行bounded observation update，但只进入`zero_dt_fraction/same_second_fraction`；当前V16没有显式真实gap reason，因此`gap_fraction=0`。offline/online共用一个计算函数。
- `StrictOpenSetRejectionPolicy`已改为DBB knownness未接受后回退CU；CU接受Clutter则`CLUTTER_ACCEPTED`，否则`UNKNOWN_REJECTED`。旧V14/V15数据契约仍记录DBB直达Unknown，因此当前执行器fail-closed；历史checkpoint/calibration/报告未修改。

验证结果：

- V16 product-local focused：通过；真实三任务support、CU Unknown exposure、loss梯度、配置identity、DBB→CU回退、同秒非gap和局部性能状态语义均通过。
- V14 route-repair focused：通过；三份旧契约均因过时动作拓扑被拒绝，合成新路由探针通过。
- V13 hierarchical focused：通过；reason-aware和既有V13动作/指标接口未回归。
- 三任务CPU与CUDA（RTX 4060 Laptop）各1个optimizer step训练入口Smoke `contract-v3`：全部通过；checkpoint round-trip通过。Smoke未运行validation与checkpoint排名，不能作为分类性能。
- 全`src-v3` compileall通过；静态隔离通过且violations=0；V16构建器幂等复核通过；正式CLI缩短参数负向保护通过；`git diff --check`通过，仅有行尾提示。
- 中间调用问题：首次前台V16构建因工具短超时被终止，确认无残留进程/半成品后改为隐藏后台并成功完成；一次静态隔离误用`--project-root`、两次PowerShell/rg检查命令转义错误均按真实CLI重跑通过，不属于代码失败。

当前执行边界：V16已达到Gate、DBB specialist、CU specialist三条独立20 epoch正式训练的技术准备状态，但正式训练与独立validation尚未执行，因此没有V16性能数值。完整级联、三checkpoint组装选优、parent-grouped OOF、正式拒识、development/test、stable、session、P95和部署全部仍为“未执行”。

## 2026-08-13 Stage 059 - V17 DBB微多普勒摘要单因素配对准备

本Stage只处理DBB specialist的Drone/Bird/Balloon局部任务。Control和Treatment共同加入既有四维当前事件`rd_physical`，Treatment唯一模型变量是额外三维当前事件微多普勒摘要；Gate、CU、三层级联、拒识、stable、液态hidden size、loss和V16 grouped split均不改变。完整STFT张量被版本化保存为工程证据，但不是本轮模型输入。

原始回波处理口径：逐帧严格解析CQ08 `.dat`，I/Q按MATLAB列序reshape为`[31,PRTnum]`；用Taylor窗MTD和目标record多普勒±5、中心距离行16±5局部峰复现目标距离行；对该距离行按PRT与载频映射物理速度轴，执行Hann 50%重叠、FFT=128的STFT。三维摘要分别为峰频带能量时间CV、`-10 dB`速度谱展宽比例和自相关调制周期。每个TR事件仅聚合其V16 index明确映射的raw frames，禁止前后事件或parent级统计。

重要证据更正：早期临时探针的`Drone CV=0.52 vs Bird CV=0.13`使用了错误的Python I/Q重排，已失效。按MATLAB审计通过的Fortran列序，三条Smoke代表事件的归一化CV分别为Drone Track 1=`0.127563`、Bird Track 333=`0.181750`、Balloon Track 3010=`0.060699`。这三点不能证明类别分布；流水线在正式训练前必须先生成train-only全量分布报告。

为防止采样模式伪特征，STFT证据缓存额外保存逐帧`PRTnum`、PRT秒、载频和CPI count，这些字段均不进入模型。readiness在train-only上检查PRTnum/PRT秒/载频与三类标签的Cramér's V不超过0.50，并要求至少一个Drone/Bird均有不少于8个有效事件的共享采样模式内，至少一项robust effect绝对值仍达到0.10。CPI count只记录连续性/来源，不参与关联门槛或checkpoint选择。

为防止同一距离行的零速杂波峰替代目标峰，证据缓存同时记录局部MTD目标Doppler index/物理速度和STFT积分峰速度。9帧focused探针最大偏差约4.77 m/s，均落入冻结的±8 m/s峰频带；全量readiness要求train至少90%逐帧满足该覆盖，否则阻止训练。目标/峰诊断量不进入模型。

readiness不仅由PowerShell检查，`train.tasks`的V17路径也强制回读v3报告，核验两臂配置路径/SHA-256、data contract hashes、canonical frame-index SHA-256、所有check为true、validation未参与放行且test/正式训练仍未执行；训练摘要再绑定readiness文件SHA-256。直接运行单臂Python训练命令也不能绕过prepare。

数据/合同实现：

- V17构建器只处理V16 grouped index中的DBB train/val；test行仅用于确认split集合，raw/RD payload访问计数固定为0。
- 为每个split生成四维RD physical cache、三维摘要cache和ragged STFT evidence cache，并绑定source index、V16 contract/provenance/stats及所有cache SHA-256。
- 缓存采用只新建或内容完全相同才复用的策略；新NPZ写入后继承父目录ACL并立即逐数组回读。非空正式artifact只允许与同一生成配置完全绑定。
- Dataset要求summary sample ID与task split事件集合完全相同。Treatment还强制micro-Doppler valid与RD-physical valid逐值一致；两个valid均只门控，不拼入Fusion。

raw帧映射修复：审查确认`src/process/build_cq08_rd_physical_canonical_store.py`把`raw_frame_index`写成`enumerate(record.rd_mat_files)`的MAT列表序号，V16的`rd_raw_frame_indices_json`继续继承该历史字段，不能代表`.dat`完整帧号。全106026行canonical frame provenance审计显示46行不相等，全部属于Track 1010且真实raw-DAT序号比legacy ordinal大8。Track 1010 `.dat`有54帧，但manifest只有MAT采集帧9–54；旧构建器会把point 3的legacy 0错误索引到raw point 1并失败。

修复后以`parent_sample_id+canonical tensor index`回连`cq08_rd_physical_canonical_frame_index_v3_1.csv`，从`rd_mat_path`解析MAT一基采集帧号并减一得到raw-DAT零基帧号。每帧强制核验MAT filename track ID、canonical track point、tensor path与raw target record；evidence cache分别保存legacy MAT ordinal、canonical tensor index、MAT采集帧号和raw-DAT index。summary/STFT schema升级v2、paired protocol升级v2、checkpoint升级v2、readiness升级v3，旧缓存/报告/checkpoint不能混用。

配对模型/训练实现：两臂共享40个state entry；Fusion输入维度Control=`83`、Treatment=`86`，Treatment新增三列严格置零并复制其余全部参数。两臂seed和DataLoader seed均为42，固定20 epoch，关闭early stopping，每epoch保存checkpoint和parent顺序hash。各自best仍按V16的validation parent最弱Precision/Recall选择；paired报告要求Drone/Bird中较弱的P/R/F1 floor与Macro-F1严格改善，同时比较Control-best/Treatment-best/epoch20的不同same-epoch轨迹，重复epoch不能重复计票。

已执行验证：

- 全`src-v3` compileall通过。
- V17 focused Smoke v2通过：Drone/Bird/Balloon各3帧共9帧Python目标距离行与对应MATLAB `debug_info.detected_range_bin-1`完全一致；新增Track 1010反例确认legacy 0映射raw point 1且MAT I/Q最大误差=`277.504744`，修复映射为MAT frame 9/raw index 8/point 3且MAT I/Q误差=`0`。Track 1010全部46个保留MAT帧逐帧I/Q误差均为0且track point全部匹配；真实`_build_split`内存Smoke完整处理18事件、16个有效事件和46帧STFT，未写正式缓存。
- 配对初始化通过：初始logits逐位一致、新增三列梯度最大值=`0.666845`、offline/online首事件logits最大误差=0、Control与Treatment checkpoint保存/恢复均通过。
- V16 product-local focused回归通过：三任务support、真实CU标签、parent-final+prefix loss、同秒非gap、DBB失败回退CU及checkpoint identity均未回归。
- 静态隔离通过且violations=0；PowerShell脚本AST解析通过；`git diff --check`通过，仅有既有行尾转换提示。

未执行：V17全量train/val缓存构建、train-only readiness、两臂正式训练、paired validation、多seed、完整级联、OOF、test、正式拒识、stable、P95和部署。根据用户约束，这些耗时数据处理/训练只通过`src-v3/scripts/v17_micro_doppler_paired_pipeline.ps1`交由用户手动运行；本轮没有将Smoke指标写成分类性能。脚本对已完成arm只读跳过，对`status=failed`且存在完整epoch `last_model.pt`的arm恢复，其他非空artifact拒绝继续。

当前执行边界：已达到手动`-Stage prepare`状态，尚未达到正式训练状态。prepare若发现特征退化、调制周期可估率不足、采样模式强关联或共享采样模式内Drone/Bird效应不足，将返回非零并阻止train。

## 2026-08-13 Stage 059 - V17全量readiness阻断与目标锚定修复

用户首次执行`-Stage prepare`后，全量旧v2缓存、配置与readiness报告均已生成；脚本返回非零是readiness的预期fail-closed行为，不是PowerShell或Python异常。唯一失败检查为`train_stft_peak_band_contains_local_mtd_target_for_at_least_90_percent_frames=false`：train对齐`58038/65879=0.880979`，validation对齐`13486/15323=0.880115`。两split几乎相同，且99.977% train目标Doppler位于固定`[-85,85] m/s`轴内；Track 1010的46帧canonical/raw偏移审计仍通过，因此不支持把失败归因于旧帧映射、速度轴截断或validation选择。

根因是旧提取器在局部MTD确定的目标距离行上仍取“全局STFT积分峰”作为CV/周期频带中心；同一距离行内更强的零速杂波或其他峰会夺取中心。MATLAB交叉检查确认Python局部MTD目标Doppler本身正确：Drone/Bird/Balloon代表帧与MATLAB `detected_doppler_mps`最大误差分别约`0.01083/0.00423/0.00178 m/s`。因此没有降低冻结的90%门槛，也没有使用validation调参。

修复后的target-anchored v3在局部MTD目标Doppler±8 m/s候选区内选择最强积分STFT峰；CV与调制周期使用该目标局部峰±8 m/s频带，`-10 dB`展宽只计算围绕目标局部峰的连续谱瓣。无约束全局峰继续保存在证据缓存中，用于量化杂波夺峰，但不进入三维模型输入。合成强零速杂波探针中，全局峰位于`-1.3492 m/s`、距注入目标`20.8982 m/s`，目标局部峰位于`20.2381 m/s`、误差`0.6891 m/s`，验证新语义确实纠正目标而非仅改变门槛。为避免目标锚定后的几何覆盖检查近似按定义成立，readiness另要求至少90% train帧的目标局部峰/全局峰功率比≥0.10（−10 dB）；该物理下限复用既定谱展宽阈值，不由validation调参。旧v2 STFT证据只读重算得到train/val约`0.961445/0.964498`，仅用于实现前可行性检查，修订版仍必须重新全量计算并通过v4报告。

为阻止旧新数据混用，修订版使用独立`target-anchored-v3`缓存、contract、manifest、配置、artifact和报告路径；summary/STFT schema升级v3，paired protocol/checkpoint升级v3，readiness升级v4。旧v2全量缓存、旧配置和blocked readiness只读保留，不覆盖、不允许进入训练。

已执行：V17 target-anchored focused Smoke v3、V16 product-local回归、全相关模块compileall、静态隔离、Windows PowerShell 5.1 AST、脚本ASCII和`git diff --check`均通过。未执行：修订版全量prepare、正式训练、paired validation、完整级联、OOF、test、正式拒识、stable、P95和部署。

## 2026-08-13 Stage 059 - V17 target-anchored正式配对完成与错误簇归因

用户已串行完成target-anchored v3全量prepare、Control/Treatment各20 epoch正式训练和paired validation。train-only readiness全部检查通过：三维摘要非退化，调制周期raw frame可估率=`0.268234`，采样字段与DBB标签的最大Cramér's V低于冻结门槛，目标局部峰覆盖、目标/全局功率比、Track 1010真实raw帧映射与test payload零访问均通过。两臂20个epoch的parent顺序hash完全一致，初始CPU logits一致；因此正式结果没有复现旧I/Q reshape、全局杂波峰或MAT ordinal/raw frame错配问题。

冻结paired结论为`completed_but_gate_failed`。own-best Control epoch 15与Treatment epoch 20的parent-final结果如下：

- Control Accuracy/Macro-F1=`0.937220/0.932380`，Treatment=`0.932735/0.928712`，分别下降`0.45/0.37`个百分点。
- Bird Precision/Recall/F1从`0.925926/0.892857/0.909091`变为`0.879310/0.910714/0.894737`。Recall增加仅来自正确Bird由50增至51；预测为Bird的总数由54增至58，Drone→Bird由3增至6，因此Precision下降4.66个百分点、F1下降1.44个百分点。
- Drone正确数由107降至105，Bird由50升至51，Balloon保持52，总正确数由209降至208。Drone→Bird/Bird→Drone由`3/4`变为`6/3`，双向错误由7增至9。
- Treatment的Drone/Bird/Balloon最弱P/R/F1均未达到0.95；own-best Macro-F1、Drone→Bird与双向错误门槛失败，不进入多seed或任何产品级阶段。

parent-final逐ID对齐显示只有10条轨迹改变预测。Treatment修复4条Control错误：Bird 444（Balloon→Bird）、Bird 665/816（Drone→Bird）和Drone 1945（Bird→Drone）；破坏5条Control正确：Drone 1535/4237/4238（Drone→Bird）、Bird 531（Bird→Drone）和Bird 2687（Bird→Balloon）；Drone 3960由错误Balloon变为错误Bird。也就是5条由对变错、4条由错变对、1条错误目的类变化，净损失1条正确parent。50,000次固定seed 17042 parent bootstrap只作探索性区间：Bird Recall差约`[-5.88,+9.80]`个百分点、Bird Precision差约`[-12.03,+2.01]`个百分点、Macro-F1差约`[-3.24,+2.53]`个百分点；discordant correctness为5对4，McNemar exact `p=1.0`。单seed和56个Bird parent不足以支持稳定收益声明。

own-best还混合了checkpoint epoch。相同epoch 15时Treatment相对Control的Bird Recall/Macro-F1分别低`5.36/2.26`个百分点；相同epoch 20时分别低`3.57/1.31`个百分点，所以own-best的Bird Recall上升不能直接归因于微多普勒。Treatment epoch 19曾达到Macro-F1=`0.944209`、Bird P/R=`0.873016/0.982143`和Drone→Bird/Bird→Drone=`7/0`，但冻结`parent_min_precision_recall`在epoch 20得到略高最弱P/R=`0.879310`并选中后者，同时Macro-F1降至`0.928712`、双向错误升至9。epoch 19仅作为机制诊断，禁止在看过validation后追认best。20个same-epoch中Treatment的Bird F1/Macro-F1分别高于Control 13/12次，但Bird Precision/Recall只高9/10次，方向不稳定且epoch不独立；冻结same-epoch归因门槛未通过。

特征归因显示不是简单的末期液态状态漂移。Treatment改变的多数轨迹从首个可验收prefix起就保持另一类别，Track 4237/4238分别连续19/26个eligible状态判Bird，最终置信度为`0.8051/0.9010`；Track 1535连续18个eligible状态判Bird，最终为`0.9911`。这说明当前事件三维摘要提供持续同向证据，液态状态再沿轨迹累积放大，而不是只在final附近突然漂移。

根本限制是全局类间方向不具备采样模式不变性。train全局Drone−Bird robust effect在CV/展宽/周期上为`-0.517/-0.250/0.000`，validation为`-0.557/-0.375/+0.103`，即Bird通常有更高CV和展宽；但PRTnum=142的两个共享模式三项effect全部反向为正，约`+0.80至+1.10`。恰好新增错误Drone Track 4237/4238来自相邻两秒capture并全部采用该模式，其parent均值CV/展宽位于validation Drone约第99至100百分位。当前提取器对PRTnum<192使用32点窗、其余使用64点窗，STFT物理窗时长、hop、可分辨调制周期和谱估计方差随PRT/载频变化；采样元数据为避免捷径又不进入模型。因此readiness的“至少一个共享模式存在绝对效应”只能证明有信号，不能保证数值在模式间同义。调制周期还只有26.82% raw frame可估，不可估与真实低值均编码0，进一步形成语义混合。

本轮结论：三维摘要能强烈改变困难parent并纠正部分Bird，证明原始I/Q微动包含增量信息；但当前摘要/聚合/尺度把“Bird-like微动”扩大成非专属证据，增加Drone假阳性，未形成可放行的DBB收益。V17只否定当前三维target-anchored摘要契约，不否定完整Doppler-time STFT张量。后续若重开，优先变量应是按物理时间与速度分辨率统一的mode-invariant摘要及周期可估门控；在train-only先要求主要共享模式效应方向一致，再冻结新的checkpoint层级规则。不得在V17事后调class weight、阈值或挑epoch。

当前执行边界：V17局部正式训练与paired validation已完成并失败收口。完整五类级联、OOF、test、正式拒识、stable、P95和部署均未执行；没有修改标签、V16 split、normalization、历史checkpoint或正式artifact。

## 2026-08-13 Stage 060 - V18宽带多尺度微多普勒修复与训练准备

用户提出V17提取器没有保留Drone真正关键的宽带旋翼特征。复核确认传入链路并非主要问题：sample/cache对齐、valid门控、配对初始化、Fusion拼接、offline/online和checkpoint round-trip均已有通过证据；V17负结果主要来自三维物理摘要。需要对原判断作两项校正：PRT=142的原始脉冲采样Nyquist并非325 Hz，但其约4 ms单帧时长和train三类parent支持`4/3/0`使其无法校准；Bird 5–15 Hz在4–46 ms单帧内也不足一个15 Hz周期，不能声称被完整保留。

新实验递增为V18。Control继续使用四维`rd_physical`，Treatment最初设计39维宽带多尺度微多普勒摘要，冗余审查后形成21维候选，train-only采样模式一致性门禁又将正式模型输入收敛为8维`stable8`；完整23维逐帧物理证据始终保留。提取器在无标签局部MTD目标Doppler±8 m/s内细化主体频率，拟合并消除复相干主体正弦，然后统计相对主体正负8–30、30–60、60–90 m/s侧带能量、总能量、RMS/q90范围、对称性、Rayleigh分辨单元熵/集中度/峰数和SNR。噪声底不再从目标行宽带侧带区域估计，而从目标距离邻域十个非目标行按频率估计，避免把Drone宽带侧带自身扣除。

时间调制改为直接读取原始脉冲包络，提取300–600、600–1200和1200–2000 Hz幅度及高频占比、中心频率、集中度、熵和SNR，绕开V17 STFT hop的时间Nyquist。上限冻结为2 kHz，以保证所有候选PRT/PRI帧均满足Nyquist。V17 period被删除；Bird低频扑翼周期不在当前单帧合同中估计。每事件聚合23项中位数和16项q90；181点残差Doppler profile及64点包络调制profile只作审计，不进入模型。

包络调制三个频带的幅度按一侧Parseval带内能量直接换算RMS，不再使用“全包络RMS乘频带占比平方根”的近似，避免低频/宽带能量改变造成伪高频幅度。调制噪声底与Doppler侧带一致，来自邻近非目标距离行的逐频中位数估计；目标自身300–2000 Hz宽带不再承担噪声估计，防止旋翼调制越宽越被扣除。

相对Doppler按慢时间采样的PRF周期计算有符号循环频差。主体位于无模糊速度边缘时，跨Nyquist边界的真实侧带不会再因普通频差超过`±90 m/s`而被丢弃，也不会重复计数；focused合成信号覆盖主体靠近正边缘且正侧带折返负频率的情况，并要求正负`30–60 m/s`侧带同时保留。

相对侧带不再额外排除绝对速度`±2 m/s`。该截断会把运动主体的一条真实旋翼侧带在跨过零Doppler时系统性删除，破坏相对主体的平移不变性；V18由主体相消和邻距离行噪声底抑制杂波，并用跨绝对零速的合成信号验证正负`30–60 m/s`侧带仍同时保留。

为阻断采样模式捷径，逐帧可靠性不只看`PRTnum`，而按`脉冲数+PRI+载频`冻结8个完整采样模式，并要求每个模式有足够train三类parent支持、速度Nyquist≥90 m/s且调制Nyquist≥2000 Hz。现有V17只读证据核实8个模式最小parent支持仍为Drone/Bird/Balloon=`33/41/25`。PRT=142完整证据仍保存，但事件级模型干预严格置0；valid保持当前事件RD存在语义，PRT、载频、CPI、reliability均不进入分类器。全量构建器继续使用V17已审计的canonical tensor→MAT帧号→raw-DAT帧映射，保留Track 1010的46个`+8`反例，test raw/RD payload访问固定0。

train实际共有12个完整采样模式。除上述8个校准模式外，2个PRT=142模式为`3/3/0`与`4/3/0` parent支持；另有validation完全缺失的`226×48µs×15.6GHz`和`226×51µs×16.3GHz`，各仅`4/0/0`。后4个模式全部保留逐帧审计证据但模型干预精确为0；构建器/readiness要求train模式集合与冻结12项完全一致，避免新增模式被静默接收。

checkpoint排序在训练前冻结为：三类全部P/R/F1最弱值、Drone↔Bird双向错误最少、Macro-F1最高。2个百分点门槛只用于两臂完成后的采纳，不参与选epoch。Treatment需同时满足parent Macro-F1提升`>0.02`、Drone/Bird弱项floor提升`>0.02`、两个方向错误均不增加且双向总错误下降、三类各P/R/F1≥0.95，并至少有一个预冻结same-epoch锚点通过同一严格增益门槛。

冗余审查确认旧39维包含大量median/q90重复与强共线形状量。修订合同保留18项median：正负8–30/30–60、total、RMS/q90范围、top4 concentration、peak count、侧带SNR、包络RMS、三调制带、centroid、top3、调制熵/SNR；q90仅保留正负8–30和total。删除动作只作用于模型摘要，23维逐帧evidence及181/64点profile不变。调制三带已给出幅度，保留centroid表示频率位置并删除600–2000 fraction。每帧total与六方向带和严格相等，但median是非线性聚合；合成三帧反例中`median(total)=1.0`而六方向带median和为`0.0`，因此median total不能作为零信息列删除。

`pruned21` focused Smoke已通过：旧39维中恰好18项不再进入Fusion；真实三类代表事件输出21维finite向量；合成相干主体、宽带侧带和900 Hz包络调制被保留；跨Doppler Nyquist与跨绝对零速的正负侧带同时保留；PRT=142逐帧证据非零但事件干预向量精确为0；21维cache到Dataset/Collate逐值一致；合成readiness聚合覆盖12个观测模式、8个校准模式和4个零干预模式；Control/Treatment共享40个state entry，Fusion维度`83/104`，新增21列初始为0且梯度最大值`0.921369`；offline/online误差0；builder生成的两份JSON配置均可由正式加载器解析；checkpoint round-trip、readiness SHA绑定、Track 1010映射和PowerShell AST均通过。

prepare支持安全恢复：若上次中断后留下部分V18输出，重新执行会先完成全量重算，再仅复用逐数组/全文完全一致的文件；不同或损坏文件拒绝覆盖。任一正式V18 artifact非空后，prepare在读取原始数据前即阻止重建，防止训练checkpoint与变化后的缓存/配置混用。

中间阶段只达到手动全量`pruned21 prepare`状态。该prepare构建独立21维train/val缓存、manifest、data contract和两配置，然后执行train-only readiness：21维逐项非退化；按parent均衡均值至少4项同时达到方向无关AUC≥0.55与robust effect绝对值≥0.10，且在全部8个完整采样模式中方向不反转、至少两个模式绝对effect≥0.10；所有全局已达判别门槛的特征都必须满足该模式一致性，避免危险反转列仍进入Fusion；各模式三类parent support≥8；PRT=142支持和零干预保持；summary从逐帧证据精确重建。validation仅报告；门禁实际失败后没有降低阈值，也没有启动训练。

用户手动执行`pruned21 prepare`后，readiness唯一阻断项为全局有效特征未全部保持采样模式方向一致。16个全局有效特征中只有10个满足8模式零反转；median/q90 total在两个PRT=900模式由全局负效应翻为约`+0.46至+0.69`，600–1200 Hz调制幅度在两个PRT=226模式翻为约`+0.29至+0.32`，centroid和top3在PRT=900也发生强反转。q90正8–30的部分反向接近零，但q90分支没有一项满足完整模式一致性。该阻断证明readiness正在发现真实域条件反转，不是门槛实现错误，禁止绕过或只忽略反向模式。

只用train parent和既有23维frame evidence继续做候选审计：正负合并30–90、q90 envelope RMS、q90 modulation SNR虽可做到模式一致，但与已有安全median的Spearman相关约`0.92–0.997`；较独立的`q90-median modulation SNR`仍有一个弱反转模式。因此不新增派生列。10个安全候选内部，q90 abs offset与RMS相关约`0.946`，peak count与RMS约`0.901`，两者降为evidence-only，最终冻结8维模式稳定模型输入。

`stable8`实际train-only重算已经由focused Smoke完成：8项全局AUC约`0.685–0.927`、绝对robust effect约`0.581–1.180`，全部8个模式零反转；正8–30项有5个模式绝对effect≥0.10，其余7项至少8个或冻结要求数量的强同向模式。旧`pruned21` manifest SHA-256=`aa1b21bdb50430604fb9d05af6ccee6f47101b6bc443a046849c454fb1aae955`，blocked readiness SHA-256=`a35b3a54ec88ab6c30f9042cebbfd86605a51455d2e6e5de6e3dabf1e4267406`；新prepare将校验两者并重聚合train/val `18266/4288`事件。本次重聚合raw文件、raw帧和RD tensor访问均为0；报告另保留来源`pruned21`证据最初构建时train/val各`953/223`个raw文件与RD tensor的历史访问计数，二者不会混写。正式stable8 prepare、训练和paired validation仍未执行。

用户随后完成`stable8 prepare`和两臂各20 epoch正式训练。prepare的16项train-only检查全部通过，train 8/8特征全局有效且跨8模式零反转；validation仍为8/8全局有效，但正8–30、正30–60和包络RMS仅有5/8特征保持逐模式方向一致，严格作为report-only证据。Control/Treatment初始CPU logits逐值相同、20 epoch parent顺序SHA一致、各自best epoch为15/16。

paired validation结果为`completed_but_gate_failed`。Control→Treatment own-best parent Accuracy=`0.937220→0.955157`、Macro-F1=`0.932380→0.953929`，分别提升`0.017937/0.021549`；Drone/Bird弱项提升`0.020936`。Treatment逐类P/R/F1为Drone=`0.964286/0.955752/0.960000`、Bird=`0.913793/0.946429/0.929825`、Balloon=`0.981132/0.962963/0.971963`。Bird Precision未达0.95，Drone→Bird由3增至4，Bird→Drone由4降至3，双向总错仍为7。same-epoch 16虽同时取得Macro-F1 `+0.018576`、弱项`+0.038793`和双向错误减少4，但未严格超过冻结`+0.02` Macro-F1门槛，禁止事后降低门槛或追认采纳。

对Treatment best的4个Drone→Bird、3个Bird→Drone和1个Balloon→Bird进行了train尺度、validation正确近邻、逐事件特征中心和Control/Treatment概率归因。8条轨迹stable8有效事件率均为100%，且跨PRT 226/360/570与混合模式，不存在共同缺失或单一采样模式故障。Track 2083的8/8 parent特征更接近Bird、事件中心为Bird/Balloon=`14/9`；Track 3960的事件中心为Balloon/Bird/Drone=`23/1/1`且正确近邻混合，两者接受为当前stable8下的物理重叠或标签内异常流型。其余6条均不满足“特征相似不可分”：1908/2103分别22/23、23/24事件指向Drone，2589为15/16指向Bird，3027为14/17指向Balloon，531与731全部事件指向Bird；其中531由Control正确Bird `0.861274`被Treatment翻为Drone `0.522541`，是最直接的决策耦合反例。

V18最终冻结为：提取器与stable8包含有效信息，但将8维直接拼入共享Fusion并联合更新全部参数会错误放大部分边界。后续版本只允许改变决策耦合，不新增物理特征、不调validation阈值：冻结V18 Control主干，以train-only归一化后的stable8因果累计均值产生零初始化、有界、三类零和late residual，训练仅更新残差权重。完整级联、OOF、test、正式拒识、stable、P95和部署仍未执行；标签、V16 split/normalization及历史artifact均未修改。

## V19 stable8后置残差决策耦合实验

V19将V18归因结论落实为严格单因素实验。基础分类logits固定来自V18 Control best epoch 15；prepare阶段只在`torch.inference_mode`下对V16 grouped train/val DBB parent运行该冻结主干并生成版本化cache，正式训练不再加载、复制或更新主干参数。baseline checkpoint文件SHA、模型state SHA、V18 training summary、V18 paired report、V18数据contract及stable8 cache均写入V19 provenance；任一变化都会阻断训练。

stable8尺度只由953个train DBB parent拟合。每个parent先对其有效当前事件stable8取均值，使长轨迹不主导median/IQR；推理输入再逐事件使用train尺度归一化，并计算截至当前prefix的有效观测累计均值。当前事件缺失时保持先前累计证据，首次有效证据前为零，padding为零；valid count、PRT、载频、路径和标签均不进入残差模型。validation参与cache对齐和冻结Control指标重现，但`validation_used_for_readiness_gate=false`，不参与尺度、特征选择、阈值或启动裁决。

残差模型只有一个无bias线性层，权重shape为`3×8`。全零初始化保证combined logits与V18 Control逐值相等；每个事件的原始三类修正减去其均值后，再以共同系数缩放到绝对值不超过3.0，因此保留三类零和且任意两类最大相对修正不超过6.0。该共同缩放不会改变三类修正方向，也不会像逐类独立`tanh`那样破坏零和。训练仅更新24个权重，使用V18相同seed、parent顺序、AdamW `1e-4`、class weights、20 epoch、final主损失和0.10 prefix辅助；20个epoch的parent顺序SHA逐个与V18 Control历史值比对。

focused Smoke覆盖合成因果反例、无证据/缺失保持、零初始化、残差上界、零和、梯度、checkpoint/cache round-trip和完整临时端到端20 epoch。真实三类parent probe得到残差权重最大绝对梯度`0.167272`；一次optimizer step前后V18 Control state SHA均为`5d61933484a0f01dac629a5b8168b92b6733be304897a3dfac72e76e54780b7e`，所有主干参数gradient均为空；20/20 shuffled parent顺序hash与正式V18 Control一致。该Smoke只证明合同和可训练性，不提供V19性能结论。

正式采纳门槛保持V18冻结标准：Treatment三类各自Precision/Recall/F1均至少0.95，parent Macro-F1和Drone/Bird弱项相对固定V18 Control均严格提升超过0.02，Drone→Bird与Bird→Drone均不得增加，双向总错误至少减少1。V19不设validation阈值、不重新训练Control、不执行完整级联；prepare、正式训练、validation结果、OOF、test、正式拒识、stable、P95和部署当前均未执行。

## 2026-08-15 Stage 062 - V20 zero-delta skip诊断准备完成

根据V19之后的单因素实验纪律，新增V20 skip诊断并预留V20r density-weighted修订。两者均复用V16 capture/content grouped index、split、normalization、TR/RD输入和DBB specialist训练协议，只改变`liquid_invalid_dt_update_mode`；不重建数据、不新增字段、不访问test。Control为`bounded_observation`、固定gain=`0.25`；V20 Treatment为`skip`，同秒事件不吸收当前事件信息，但仍允许recurrent LayerNorm的确定性仿射变换；V20r Treatment未来按同秒因果序`k`使用`min(0.25×k,0.75)`。

已完成代码链路：`model.py`支持skip/density、offline按reason维护ordinal、`task_models.py`和`cascade.py`传递ordinal，`streaming.py`在线持有ordinal；配置解析、V20 paired model builder、train-only readiness和paired report均已建立。训练入口已修正为依据paired protocol选择对应readiness报告，V20常量/构建器也已加入公开导出。

已完成验证：V20 skip配置和readiness生成成功；train/val DBB parent为`953/223`，V16 contract hashes未改变；zero-delta正式统计排除first event，train/val event-level标签Cramér's V分别为`0.1264218055/0.0916738183`，仅作为报告诊断；test payload access=`0`，validation不参与readiness gate。Python compile、PowerShell AST、src-v3隔离检查、V20 focused Smoke和脚本`prepare`均通过；Smoke确认skip事件内容不影响候选状态、density离线/在线最大误差为0、配对初始logits逐值一致。

当前边界：代理未执行20 epoch正式训练。用户下一步运行V20 skip脚本的`train`或`all`阶段；训练完成后只做development validation paired report。V20r、Gate/CU、完整级联、OOF、test、正式拒识、stable、P95和部署继续记录为“未执行”。

## 2026-08-15 Stage 062结果与Stage 063 V20r启动

用户已完成V20 skip两臂20 epoch和paired validation。Control/Treatment best epoch为15/7；own-best parent Accuracy均为`0.923767`，Macro-F1为`0.916864→0.920911`，仅增加`0.004047`。Bird Precision由`0.892857`降至`0.877193`，Recall保持`0.892857`，F1降至`0.884956`；Drone→Bird由5增至6、Bird→Drone由2增至5，双向错误由7增至11。20个same-epoch结果方向混合，V20状态为`completed_but_gate_failed`，保留bounded Control且不采纳skip。

代码复核同时明确skip并未实现文字上的“当前观测可见但不累积”：Fusion事件只通过liquid state到分类头，`alpha=0`后当前zero事件不进入state或logits。因此V20只能解释为丢弃同秒当前观测的强消融，不能作为V20r的严格对称反向证据。V20r按用户确认作为独立正向实验启动。

V20r不重复训练Control，直接SHA绑定并复用已完成V20 skip Control。新增density readiness按parent因果推导ordinal：train `k1/k2/k≥3=11733/1677/4856`，其中Treatment对k2使用0.50、k≥3使用0.75；标签Cramér's V=`0.091077`。Bird/Drone/Balloon train parent均衡暴露率分别为`0.441576/0.357492/0.243624`，因此后续必须重点审计Bird边界和高k类别相关性，不允许仅凭总体Macro-F1放行。

V20r focused Smoke和prepare均已通过。合成增益验证k1/k2/k3为Control增量的1/2/3倍且k≥3饱和；padding不推进ordinal，offline/online误差0；真实Track 1010在首个density暴露前Control/Treatment logits相同、暴露后不同，训练loss和梯度finite且非零。配置/readiness复用V16 contract，validation只报告、test payload访问0。正式density Treatment训练等待用户手动执行；OOF、test、级联、正式拒识、stable、P95和部署未执行。
