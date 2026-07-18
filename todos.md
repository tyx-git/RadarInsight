# RadarInsight 一致性审查与历史模型资产治理执行计划

> **状态说明：** `[ ]` 未完成；`[x]` 已完成且有实现、验证和证据；`[!]` 阻塞或发现证据冲突，必须停止当前阶段。  
> **依据设计：** `docs/superpowers/specs/2026-07-18-code-consistency-and-model-governance-design.md`  
> **范围：** Python 源代码、测试、配置、模型 checkpoint、registry、评估结果和速度记录。  
> **排除：** `.history/`、原始数据、缓存、临时运行目录和未纳入资产治理的中间文件。

---

## 0. 全局执行协议

### 0.1 不可违反的约束

- [ ] 不读取、修改、暂存或提交 `.history/`；所有扫描命令必须显式排除该目录。
- [ ] 不运行全量 Expert1、System 或 MoE 训练；只允许 AST/静态检查、import、`py_compile`、单元测试、smoke 和固定小基准。
- [ ] 不得把 synthetic/zero RD 送入 TR+RD fusion；无真实 RD 必须走 trajectory-only。
- [ ] 不得使用 test 数据选择 checkpoint、校准参数、candidate/release/readiness/deployment/export 或 bottleneck/next action。
- [ ] 不使用 `git reset --hard`、`git restore`、`git clean`、强制 checkout 或 force push。
- [ ] 不覆盖用户已有的无关改动；禁止 `git add .` 和 `git add -A`。
- [ ] 任何模型删除必须先生成审计清单、SHA256、引用扫描、保留计划和删除日志。
- [ ] 仅自动删除同时满足“同一可比组、严格被支配、无引用、非保留角色、证据完整、哈希已记录”的模型。
- [ ] 指标缺失、NaN/Inf、评估协议不明、硬件条件不明或引用关系不明的资产默认保留。
- [ ] `[x]` 只能表示当前条目实现、回归/静态验证通过并已写入证据；“代码已修改但未验证”保持 `[ ]`。

### 0.2 每阶段固定记录

每个阶段完成前，在本文件对应位置填写：

```markdown
- 发现：<问题、证据等级 static/test/runtime-smoke/full-training-required/unknown>
- 修改文件：<路径与行号；若无修改写“仅审查”>
- 测试/命令：<完整命令>
- 结果：<通过/失败/阻塞与关键输出>
- 未覆盖风险：<不能由本阶段证明的内容>
- 状态：`[x]` 或 `[ ]`
```

### 0.3 初始基线

- [ ] 记录工作区状态，但不触碰无关改动：`git status --short`。
- [ ] 记录当前分支和最近提交：`git branch --show-current`、`git log -5 --oneline`。
- [ ] 确认目标文件不在 `.history/`，并确认待审查路径存在。
- [ ] 建立本轮审查工作目录清单，不把临时输出写入源代码目录。
- [ ] 所有命令结果和发现写入本文件，不以未保存的对话内容作为唯一证据。

**阶段门禁：** 若发现工作区存在用户未说明的删除、模型当前被训练进程占用、或审查路径包含 `.history/`，停止并记录 `[!]`，不得继续自动修改或删除。

**当前门禁状态：** `[x]` 2026-07-18：用户确认 `bug.md`、`finding.md`、`main.bak001` 与 `task_plan.md` 的删除，以及未跟踪 `.anthor/`，均为有意的无关工作区改动；保留原状且不暂存。

---

## P0：契约破坏、数据泄漏与安全边界

**目标：** 找出会改变模型语义、造成数据泄漏、绕过 RD 不变量或让不可信输入控制行为的高风险问题。  
**状态：** `[x]`

### P0-1 类别、标签和 tensor 契约

- [x] 检查类别定义、顺序、数量和 label mapping：
  - `src/core/data/labels.py`
  - `src/core/forwardlayer/labels.py`
  - `src/core/modeling/experts.py`
  - `src/core/modeling/systems.py`
  - `src/core/training/*.py`
  - `config/config.json` 及配置 dataclass。
- [x] 检查 dataset 输出、model 输入、head 输出、loss、metric、checkpoint 和 export 的 shape/dtype/device/mask/类别顺序。
- [x] 对每个发现建立最小回归测试，优先放入已有相关测试文件；不得仅以字符串搜索结果标记已修复。
- [x] 运行：
  ```bash
  python -m pytest tests/test_expert1_split_integrity.py
  python -m pytest tests/test_expert1_precision_controls_smoke.py
  python -m py_compile src/core/data/labels.py src/core/modeling/experts.py src/core/modeling/systems.py
  ```
- [x] 证据：列出不一致位置、输入输出契约、失败回归和修复后结果。

### P0-2 RD availability、窗口对齐与 fusion 隔离

- [x] 检查 `src/core/data/torch_dataset.py`、`src/core/data/rd_segment_store.py`、`src/process/build_expert1_rd_segment_store.py`、`src/core/expert1_data.py`、`src/core/expert1_trainer.py` 和 `src/core/modeling/experts.py`。
- [x] 证明无真实 RD 时不进入 TR+RD fusion；禁止用全零、伪造或仅由 tensor shape 推断的 RD 代替真实性。
- [x] 检查 trajectory window 是否只访问自身时间范围内的 RD segment；无法同步时必须显式降级 trajectory-only 并报告原因。
- [x] 增加/运行最小测试：有效 RD 进入 fusion、无有效 RD 不进入 fusion、窗口越界 segment 被拒绝。
- [x] 运行：
  ```bash
  python -m pytest tests/test_expert1_rd_segment_store_smoke.py
  python -m pytest tests/test_stage2_dataloader_throughput.py
  python -m py_compile src/core/data/torch_dataset.py src/core/data/rd_segment_store.py src/core/modeling/experts.py
  ```
- [x] 证据：记录 RD availability 来源、同步状态、coverage、降级原因和测试输出。

### P0-3 评估控制面与不可信边界

- [x] 检查 `src/core/expert1_trainer.py`、`src/core/reselect_expert1_precision.py` 及相关 checkpoint registry/export 代码。
- [x] 验证 checkpoint、nested config、registry record、caller lock 和 export payload 的 scale/bias/threshold 均经过严格 policy validation。
- [x] 验证 test 只写入 `test_reporting`，并强制包含：
  ```python
  {"report_scope": "test", "test_used_for_selection": False}
  ```
- [x] 为 malformed、non-finite、长度错误、未授权 finite bias 和 test-derived control variable 增加回归。
- [x] 运行：
  ```bash
  python -m pytest tests/test_expert1_precision_controls_smoke.py
  python -m py_compile src/core/expert1_trainer.py src/core/reselect_expert1_precision.py
  python -c "import src.core.expert1_trainer; import src.core.reselect_expert1_precision"
  ```
- [x] 证据：记录旧实现失败、修复后行为、test `_run_epoch` 调用次数和控制变量来源。

**P0 停止门禁：** 任一类别契约、RD 真实性、数据隔离或 test 控制面无法证明时，停止后续自动清理；模型资产只允许盘点，不允许删除。

**P0 执行记录：**

- 发现：`test_rd_shard_local_sampler_prefers_single_shard_batches` 失败（test）：小于每批 RD 配额的 shard 会从其他 shard 补样，破坏分片本地性；类别顺序、真实 RD 降级和 test 控制面通过现有 focused 回归确认。
- 修改文件：`src/core/expert1_data.py:191`；删除跨 shard 补样，改为仅对当前 shard 带权补样。
- 测试/命令：`.\\.venv\\Scripts\\python.exe -m pytest tests/test_expert1_rd_segment_store_smoke.py tests/test_stage2_dataloader_throughput.py -q`；`.\\.venv\\Scripts\\python.exe -m pytest tests/test_expert1_split_integrity.py -q`；`.\\.venv\\Scripts\\python.exe -m pytest tests/test_expert1_precision_controls_smoke.py -q -k "rejects_unauthorized or malformed or non_finite or invalid_precision_bias or posthoc_test_observations_do_not_override_validation_readiness or system_loader_restores_precision_checkpoint_joint_scale or final_test_checkpoint_prefers_highest_validation_score or final_test_checkpoint_uses_release_proxy_when_configured"`；`py_compile` 和 Expert1/reselector import。
- 结果：19 个 RD/吞吐测试、14 个 split 测试和 6 个评估控制面测试通过；完整 precision 模块仍有 2 个既有 P3 指标断言失败，未混入 P0 修复。
- 未覆盖风险：未运行全量训练；完整 precision 模块的两项 P3 指标断言失败仍需在 P3 单独处理；未读取或修改 `.history/`。
- 状态：`[x]`

---

## P1：参数、配置与调用链一致性

**目标：** 建立参数“定义—加载—传递—消费—持久化”矩阵，消除重复缩放、覆盖、忽略和别名冲突。  
**状态：** `[x]`

### P1-1 函数签名与调用方

- [x] 枚举 `src/**/*.py` 中函数、类方法和构造器的定义及调用方；优先检查：
  - `src/core/expert1_trainer.py`
  - `src/core/expert1_data.py`
  - `src/core/modeling/*.py`
  - `src/core/training/*.py`
  - `src/core/reselect_expert1_precision.py`。
- [x] 对每个参数记录名称、类型、默认值、单位、范围、调用方传值、最终消费位置。
- [x] 检查 positional/keyword 混用、旧参数名、重复默认值、传入后未读取和后续覆盖。
- [x] 不确定有动态调用、反射或外部脚本引用时，只登记问题，不直接删除参数。

### P1-2 配置字段与实际生效值

- [x] 建立 `config/config.json` → 配置 dataclass → trainer/model/data loader → checkpoint/registry 的对照表。
- [x] 检查 `scale`、`weight`、`threshold`、`temperature`、`margin`、`window`、`rd_*`、`stage2_*` 等参数是否重复定义或重复应用。
- [x] 识别配置声明但未消费、代码消费但配置不存在、缺失字段被错误解释为有效值的情况。
- [x] 对关键范围补配置/运行时测试，不使用完整训练验证参数生效。

### P1-3 重复归一化与参数二次应用

- [x] 静态检查 amplitude preparation、normalization、residual weight、deep reroute scale、calibration bias 是否被调用链重复执行。
- [x] 比较训练、validation、test、inference 四条路径的相同参数是否有相同语义。
- [x] 对已确认的重复应用先写失败测试，再最小修复；保留必要兼容逻辑并说明原因。

- [x] 运行：
  ```bash
  python -m pytest tests/ -k "config or parameter or calibration or expert1" --maxfail=1
  python -m py_compile $(find src -path '*/.history' -prune -o -name '*.py' -print)
  ```
  若测试选择结果为空，记录为“无匹配测试”，不得伪称通过。

**P1 停止门禁：** 若关键参数存在多个无法判定的来源，禁止进行行为性重构和模型清理；先登记冲突及所需证据。

**P1 执行记录：**

- 发现：`_clean_section()` 使用 `dict.update()` 将 `project` 的 `data_root`、`device`、`num_workers`、`seed` 覆盖分区显式值（test）；配置字段对照发现 `system.eval_fusion_candidate_notes`、`router.input`/`notes`/`selection_metric` 与 `pipeline.num_workers`/`seed` 为说明或不适用字段，未作为行为参数删除。
- 修改文件：`config/configmanager.py:50`；`tests/test_configmanager.py:1`。
- 测试/命令：`.\\.venv\\Scripts\\python.exe -m pytest tests/test_configmanager.py -q`；`.\\.venv\\Scripts\\python.exe -m pytest tests/ -k "config or parameter or calibration or expert1" --maxfail=1 -q`；`.\\.venv\\Scripts\\python.exe -m py_compile <rg --files src -g '*.py' -g '!**/.history/**'>`。
- 结果：新回归先失败后通过（2 passed）；全仓 `src` 编译通过；广泛 pytest 在 29 passed 后因既有 P3 `precision_candidate_joint_gain_gap` 断言失败停止，未发现 P1 失败。
- 未覆盖风险：未运行完整训练；动态外部调用无法由静态扫描穷尽；另一个既有 P3 RD bottleneck 断言仍待 P3 单独修复。
- 状态：`[x]`

---

## P2：分支语义与原始设计目的

**目标：** 证明每个高影响分支的进入条件、输出契约和设计目的没有漂移。  
**状态：** `[x]`

### P2-1 Expert1 分支

- [x] 梳理 trajectory-only、真实 TR+RD、低质量 RD、无 RD、fallback、reject、reroute 和 deep reroute 的流程图。
- [x] 对每条路径记录：输入条件、是否修改 logits、是否计算 auxiliary loss、是否允许 calibration、输出类别空间。
- [x] 检查 default/empty tensor 是否意外触发 fusion；检查异常处理是否吞掉 availability 或 quality 信息。
- [x] 用小型 fixture 验证每条路径的输出 shape、来源标记和不变量。

### P2-2 Router、Expert 和 System 分支

- [x] 检查 `src/core/modeling/routing.py`、`src/core/modeling/systems.py`、`src/core/training/router.py`、`src/core/training/system.py`。
- [x] 对 rule override、learned route、reroute、expert-only、routed-expert、final-system 分别记录 logits 来源、优先级和最终类别映射。
- [x] 检查 Aircraft、Ground、Unknown、Clutter 的开放集和 reject 逻辑是否在不同入口含义一致。
- [x] 检查训练/评估 quick/full validation 是否仅改变观测成本，不改变最终候选控制口径。

### P2-3 正反向回归

- [x] 为每个已确认高风险分支增加至少一个正确输入和一个边界/错误输入测试。
- [x] 验证 fallback 不绕过校准授权、RD 真实性和 split 隔离。
- [x] 运行相关 focused tests、import 和 compile；不运行全量训练。

**P2 停止门禁：** 若某分支输出契约与下游消费不一致，停止该分支后的清理工作；先修复并通过回归。

**P2 执行记录：**

- 发现：`LowAltitudeSuspectExpert.forward()` 在 `rd_is_multimodal=None` 且输入为任意四维 tensor 时推断 `has_rd=True`（test）；全零占位因此进入 TR+RD fusion，违反“真实性不得由 tensor shape 推断”的约束。
- 修改文件：`src/core/experts.py:953`；`tests/test_expert1_rd_segment_store_smoke.py:195`。
- 测试/命令：`.\\.venv\\Scripts\\python.exe -m pytest tests/test_expert1_rd_segment_store_smoke.py tests/test_stage2_dataloader_throughput.py tests/test_expert1_evidence.py tests/test_ground_candidate_arbitration.py -q`；`.\\.venv\\Scripts\\python.exe -m pytest tests/test_expert1_precision_controls_smoke.py -q -k "stage1_disabled_fusion_outputs_trajectory_final_even_with_rd or post_lock_test_reporting_is_separate_and_cannot_mutate_validation_readiness or test_reporting_path_cannot_export_or_select_calibration"`；focused compile/import。
- 结果：新测试先失败后通过；32 个 RD、路由和系统测试通过，3 个评估控制面测试通过；无可用性掩码默认 trajectory-only，显式掩码仍决定真实 TR+RD 路径。
- 未覆盖风险：未运行完整训练和真实部署数据；外部直接调用方若遗漏 `rd_is_multimodal` 将安全降级为 trajectory-only，须显式传入掩码才能启用 fusion。
- 状态：`[x]`

---

## P3：未使用、冗余与不可达实现

**目标：** 在不破坏动态调用、兼容格式和公开接口的前提下清理已证实无效代码。  
**状态：** `[x]`

### P3-1 生成候选清单

- [x] 使用 AST、import graph、字符串引用、配置 key 和脚本引用生成候选：未使用 import、函数、类、常量、配置字段、参数、返回字段、永真/永假条件和重复 helper。
- [x] 明确排除 `.history/`，并记录排除规则。
- [x] 将候选分为 `safe_remove`、`needs_runtime_evidence`、`public_or_compatibility`、`keep`。
- [x] 不因“搜索不到调用方”直接删除可能由反射、配置或外部脚本使用的符号。

### P3-2 逐项修复

- [x] 每个 `safe_remove` 项先补失败回归或确认现有回归覆盖。
- [x] 删除重复转换/不可达分支时保持输入输出契约不变。
- [x] 每次只处理一个逻辑小批次，运行对应 focused test，再继续下一个批次。
- [x] 将修改文件和未覆盖动态调用风险写入本文件。

### P3-3 清理后验证

- [x] 运行全仓 Python compile（排除 `.history/`）：
  ```bash
  python -m py_compile $(find src tests -path '*/.history' -prune -o -name '*.py' -print)
  ```
- [x] 运行与变更相关的 pytest；若全量 pytest 预计触发训练或外部数据，改用明确的 focused 集合并记录原因。
- [x] 检查导入错误、循环依赖、未定义名称和配置反序列化错误。

**P3 停止门禁：** 若清理导致任何 focused test 失败、import 失败、配置字段丢失或动态引用不确定，恢复仅本阶段新增的改动（不得使用 destructive git 命令），并标记 `[!]`。

**P3 执行记录：**

- 发现：AST 审计得到 91 个 import 候选，均为 `public_or_compatibility` 重导出或 `needs_runtime_evidence`，无 `safe_remove`；同时修复两项既有精度回归：候选阈值由提交 `28f47df` 放宽后测试 gap 未同步，且 proportional fallback 诊断错误复用候选容忍阈值而漏报任意负增益。
- 修改文件：`src/core/expert1_trainer.py:2440`；`tests/test_expert1_precision_controls_smoke.py:1058`。
- 测试/命令：标准库 AST 审计（仅扫描 `src/**/*.py`，显式排除 `.history/`）；`.\\.venv\\Scripts\\python.exe -m pytest tests/test_expert1_precision_controls_smoke.py -q`；`.\\.venv\\Scripts\\python.exe -m py_compile <rg --files src tests -g '*.py' -g '!**/.history/**'>`；trainer/reselector import。
- 结果：35 个 precision 测试和 20 个 subtests 通过；compile/import 通过；未删除候选，跨模块重复 helper 保持原位以避免破坏公开/脚本边界。
- 未覆盖风险：AST 无法证明反射或外部脚本调用；未运行完整训练；重复 helper 的统一抽取需要单独的公开接口迁移计划。
- 状态：`[x]`

---

## P4：跨模块架构、序列化与性能一致性

**目标：** 验证 data → model → trainer → evaluator → registry → exporter 的契约、持久化和性能控制。  
**状态：** `[x]`

### P4-1 跨模块往返

- [x] 使用小型 fixture 验证 dataset 输出可被 model 消费，model 输出可被 trainer/evaluator 消费。
- [x] 验证 checkpoint 保存—恢复—registry—导出的关键字段一致，尤其是类别顺序、scale/bias、provenance、selection flags。
- [x] 对 malformed payload、旧字段、缺失字段和非有限值验证安全降级或拒绝。
- [x] 确认 test reporting 不进入 selection/release/readiness/export。

### P4-2 性能风险审查

- [x] 检查 sampler、RD cache、worker、CPU/GPU 拷贝、逐 batch CPU 操作、重复模型执行和隐性全量扫描。
- [x] 只有静态证据显示性能风险后才运行固定小基准；不得运行全量训练。
- [x] 记录硬件、Python/PyTorch、dtype、batch、输入规格、warmup、重复次数、延迟/吞吐统计。
- [x] 证明 quick validation 优化不会替代最终完整 validation。

### P4-3 验证命令

- [x] 运行：
  ```bash
  python -m pytest tests/test_expert1_precision_controls_smoke.py
  python -m pytest tests/test_stage2_dataloader_throughput.py
  python -m py_compile src/core/expert1_trainer.py src/core/reselect_expert1_precision.py
  python -c "import src.core.expert1_trainer; import src.core.reselect_expert1_precision"
  ```
- [x] 运行小基准时保存原始结果，不将 smoke 延迟宣称为正式部署速度。

**P4 停止门禁：** 序列化往返不一致、test 控制面回流或性能优化改变评估语义时，停止资产角色判定和删除。

**P4 执行记录：**

- 发现：P4 控制面加固已在 `4c2e996` 完成；当前复验覆盖 checkpoint→registry→export 的精度锁、provenance、selection flags 和 test-report 隔离。静态审查显示 RD segment LRU cache、stage2 shard-local sampler、DataLoader worker/prefetch/pin-memory 与 non-blocking H2D 均有显式边界；未发现需要新建小基准的回归证据。
- 修改文件：仅阶段记录；无新增业务代码。
- 测试/命令：`.\\.venv\\Scripts\\python.exe -m pytest tests/test_expert1_precision_controls_smoke.py -q`；`.\\.venv\\Scripts\\python.exe -m pytest tests/test_stage2_dataloader_throughput.py -q`；focused compile/import；静态性能路径检索。
- 结果：35 个 precision 测试和 20 个 subtests 通过；9 个 stage2 dataloader 测试通过；compile/import 通过。未运行基准，故不记录硬件、延迟或吞吐，也未将 smoke 测试表述为部署性能；quick validation 仅由配置控制观测成本，完整 validation 保留独立的 interval/last-epoch 路径。
- 未覆盖风险：未对真实硬件、真实数据和生产 batch 执行性能基准；性能结论仅限静态控制路径与单元测试语义。
- 状态：`[x]`

---

## P5：模型资产盘点、Pareto 保留与自动清理

**目标：** 只删除明确被支配且无引用的历史差模型，同时保留不同目标下的代表模型和恢复基线。  
**状态：** `[ ]`

### P5-1 资产盘点（只读，不删除）

- [ ] 识别 checkpoint、registry、评估 JSON、速度 benchmark、导出清单和当前配置引用。
- [ ] 扫描命令必须排除 `.history/`、缓存和运行时目录；不得通过模糊 `find` 将 `.history/` 作为输入。
- [ ] 为每个资产记录：相对路径、SHA256、大小、mtime、架构版本、数据/标签版本、split/评估协议、硬件条件、registry/config 引用、指标证据和解析状态。
- [ ] 写入：`tmp/model_asset_audit.json`。
- [ ] 对无法解析、缺指标、NaN/Inf、协议不明、支持量不足的资产标记 `unclassified`，默认保留。

**盘点验证：**

```bash
python -m pytest tests/ -k "checkpoint or registry or export" --maxfail=1
python -m py_compile src/core/reselect_expert1_precision.py
```

若没有匹配测试，记录“无匹配测试”，不能记为通过。

### P5-2 可比性分组

- [ ] 依据数据/标签版本、架构、类别顺序、split/评估协议、calibration provenance、速度硬件/dtype/batch/warmup 建立可比组。
- [ ] 跨组模型只标记，不自动比较或删除。
- [ ] 检查每个 registry/config/export/训练续接引用，列入 `protected_refs`。
- [ ] 写入：`tmp/model_retention_plan.json` 的分组和引用字段。

### P5-3 角色模型识别

- [ ] 在每个可比组中识别：
  - `accuracy_best`
  - `macro_f1_best`
  - `low_altitude_recall_best`
  - `precision_best`
  - `speed_best`
  - `size_best`
  - `recommended`
  - `recovery_baseline`
- [ ] 统一所有指标方向；要求 finite、支持量和协议满足门槛。
- [ ] `recommended` 必须考虑低空五类、macro F1、source-balanced 指标和不确定性，不能只按总体 accuracy。
- [ ] 没有完整训练证据时使用 `candidate`，不得宣称达到 recall≥0.90。
- [ ] 任何角色若有并列，全部保留并记录并列原因。

### P5-4 Pareto 严格支配判定

- [ ] 仅在同一可比组比较。
- [ ] 只有模型 A 在全部有效维度不劣于 B，且至少一维严格优于 B，才允许标记 B 为 `dominated`。
- [ ] 再检查 B 不是角色模型、恢复基线或必要兼容资产。
- [ ] 再检查 B 无当前配置、训练续接、registry、导出清单或脚本引用。
- [ ] 任一条件不满足时标记 `retain` 或 `manual_review`，不能自动删除。
- [ ] 将每个维度对比、证据路径、支配模型和哈希写入 retention plan。

### P5-5 删除前门禁

- [ ] 确认 P0–P4 没有未解决的阻塞项；若有，资产只允许审计，不允许删除。
- [ ] 生成待删列表，包含路径、SHA256、支配模型路径/哈希、删除原因、引用扫描结果。
- [ ] 再次扫描当前配置、registry、导出清单、训练脚本和工作区引用。
- [ ] 确认待删文件没有被当前进程占用，且不属于用户新增或未登记资产。
- [ ] 记录“待删除数量”和“保留数量”；所有 `unclassified`/`manual_review` 资产保留。

### P5-6 执行自动删除与事后验证

- [ ] 仅删除 retention plan 中明确列出的严格被支配、无引用资产；禁止通配符删除、目录级删除和未列出的路径。
- [ ] 写入：`tmp/model_deletion_log.json`，包含删除前 SHA256、删除命令/工具版本、时间、原因和支配资产。
- [ ] 删除后重新生成资产扫描，确认删除清单与实际状态一致。
- [ ] 验证所有角色模型仍可定位，`recovery_baseline` 可加载，当前 registry/config 引用完整。
- [ ] 若事后校验失败，停止后续清理，不删除更多资产；将该阶段标记 `[!]`，保留证据。

**P5 资产治理证据要求：**

- `tmp/model_asset_audit.json`
- `tmp/model_retention_plan.json`
- `tmp/model_deletion_log.json`（仅实际发生删除时）
- 删除前后文件清单与 SHA256 对照
- 引用扫描结果
- 角色模型和不可比/人工复核资产清单

**P5 执行记录：**

- 发现：
- 修改文件：
- 测试/命令：
- 结果：
- 未覆盖风险：
- 状态：`[ ]`

---

## 6. 最终验收与提交纪律

### 6.1 全链路验证

- [ ] 运行所有本轮新增/修改相关 focused tests。
- [ ] 完成不读取 `.history/` 的 Python compile/import 检查。
- [ ] 检查 `git diff --check`。
- [ ] 对每个 `[x]` 条目确认有失败回归、修复说明、通过命令和未覆盖风险。
- [ ] 确认没有将 smoke 或静态结果写成完整训练性能结论。
- [ ] 确认测试报告明确 `test_used_for_selection=False`。
- [ ] 确认模型删除仅发生在 P5 门禁通过后，且 deletion log 与实际文件一致。

### 6.2 工作区与提交

- [ ] 重新运行 `git status --short`，确认无关改动未被覆盖。
- [ ] 逐文件检查 diff，不使用 `git add .` 或 `git add -A`。
- [ ] 代码修复、测试、审计清单和模型删除日志分开审阅。
- [ ] 只有用户明确指定提交范围时才逐文件 stage；不得推送。
- [ ] 若本轮仅完成审查和计划，不创建业务代码提交。

### 6.3 最终状态

- [ ] P0：契约、安全边界和泄漏防护
- [ ] P1：参数、配置和调用链
- [ ] P2：分支语义与设计目的
- [ ] P3：未使用、冗余和不可达实现
- [ ] P4：跨模块架构、序列化和性能
- [ ] P5：模型资产治理与严格 Pareto 清理

**最终未覆盖风险：**

- <填写仍需完整训练、重复 grouped split、真实硬件 benchmark 或人工复核的项目>

**最终结论：**

- `[ ]` 尚未达到可交付状态
- `[ ]` 代码审查完成，但模型性能仍需用户执行完整训练验证
- `[ ]` 资产治理完成，删除记录可审计
- `[ ]` 全部阶段完成
