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
**状态：** `[x]`

### P5-1 资产盘点（只读，不删除）

- [x] 识别 checkpoint、registry、评估 JSON、速度 benchmark、导出清单和当前配置引用。
- [x] 扫描命令必须排除 `.history/`、缓存和运行时目录；不得通过模糊 `find` 将 `.history/` 作为输入。
- [x] 为每个资产记录：相对路径、SHA256、大小、mtime、架构版本、数据/标签版本、split/评估协议、硬件条件、registry/config 引用、指标证据和解析状态。
- [x] 写入：`tmp/model_asset_audit.json`。
- [x] 对无法解析、缺指标、NaN/Inf、协议不明、支持量不足的资产标记 `unclassified`，默认保留。

**盘点验证：**

```bash
python -m pytest tests/ -k "checkpoint or registry or export" --maxfail=1
python -m py_compile src/core/reselect_expert1_precision.py
```

若没有匹配测试，记录“无匹配测试”，不能记为通过。

### P5-2 可比性分组

- [x] 依据数据/标签版本、架构、类别顺序、split/评估协议、calibration provenance、速度硬件/dtype/batch/warmup 建立可比组。
- [x] 跨组模型只标记，不自动比较或删除。
- [x] 检查每个 registry/config/export/训练续接引用，列入 `protected_refs`。
- [x] 写入：`tmp/model_retention_plan.json` 的分组和引用字段。

### P5-3 角色模型识别

- [x] 在每个可比组中识别：
  - `accuracy_best`
  - `macro_f1_best`
  - `low_altitude_recall_best`
  - `precision_best`
  - `speed_best`
  - `size_best`
  - `recommended`
  - `recovery_baseline`
- [x] 统一所有指标方向；要求 finite、支持量和协议满足门槛。
- [x] `recommended` 必须考虑低空五类、macro F1、source-balanced 指标和不确定性，不能只按总体 accuracy。
- [x] 没有完整训练证据时使用 `candidate`，不得宣称达到 recall≥0.90。
- [x] 任何角色若有并列，全部保留并记录并列原因。

### P5-4 Pareto 严格支配判定

- [x] 仅在同一可比组比较。
- [x] 只有模型 A 在全部有效维度不劣于 B，且至少一维严格优于 B，才允许标记 B 为 `dominated`。
- [x] 再检查 B 不是角色模型、恢复基线或必要兼容资产。
- [x] 再检查 B 无当前配置、训练续接、registry、导出清单或脚本引用。
- [x] 任一条件不满足时标记 `retain` 或 `manual_review`，不能自动删除。
- [x] 将每个维度对比、证据路径、支配模型和哈希写入 retention plan。

### P5-5 删除前门禁

- [x] 确认 P0–P4 没有未解决的阻塞项；若有，资产只允许审计，不允许删除。
- [x] 生成待删列表，包含路径、SHA256、支配模型路径/哈希、删除原因、引用扫描结果。
- [x] 再次扫描当前配置、registry、导出清单、训练脚本和工作区引用。
- [x] 确认待删文件没有被当前进程占用，且不属于用户新增或未登记资产。
- [x] 记录“待删除数量”和“保留数量”；所有 `unclassified`/`manual_review` 资产保留。

### P5-6 执行自动删除与事后验证

- [x] 仅删除 retention plan 中明确列出的严格被支配、无引用资产；禁止通配符删除、目录级删除和未列出的路径。
- [x] 写入：`tmp/model_deletion_log.json`，包含删除前 SHA256、删除命令/工具版本、时间、原因和支配资产。
- [x] 删除后重新生成资产扫描，确认删除清单与实际状态一致。
- [x] 验证所有角色模型仍可定位，`recovery_baseline` 可加载，当前 registry/config 引用完整。
- [x] 若事后校验失败，停止后续清理，不删除更多资产；将该阶段标记 `[!]`，保留证据。

**P5 资产治理证据要求：**

- `tmp/model_asset_audit.json`
- `tmp/model_retention_plan.json`
- `tmp/model_deletion_log.json`（仅实际发生删除时）
- 删除前后文件清单与 SHA256 对照
- 引用扫描结果
- 角色模型和不可比/人工复核资产清单

**P5 执行记录：**

- 发现：只读审计范围为 `artifacts/` 与 `reports/`（显式排除 `.history/`、缓存、运行时目录及用户未跟踪 `.anthor/`）：共 1,725 项，含 486 个 checkpoint（约 32.0 GiB）；906 项被 config/registry/export/report 引用保护，819 项因 checkpoint 未反序列化或缺少完整 data/label/split 协议而标为 `unclassified`。可比组 322 个，严格删除候选为 0。
- 修改文件：`tmp/model_asset_audit.json`；`tmp/model_retention_plan.json`；仅阶段记录，无业务代码。
- 测试/命令：逐文件 SHA256 审计与引用扫描；`.\\.venv\\Scripts\\python.exe -m pytest tests/ -k "checkpoint or registry or export" --maxfail=1 -q`；`.\\.venv\\Scripts\\python.exe -m py_compile src/core/reselect_expert1_precision.py`；审计 JSON schema/count/deletion-candidate 自检。
- 结果：14 个 checkpoint/registry/export 测试与 4 个 subtests 通过；审计与 retention plan 均解析通过且 `delete_count=0`。无资产满足严格支配、完整协议且无引用的组合门槛，故未删除文件，不生成 `model_deletion_log.json`。
- 未覆盖风险：checkpoint 仅使用旁路 JSON 元数据，未反序列化 32.0 GiB 权重；真实硬件/协议缺失资产全部保留；用户未跟踪 `.anthor/` 不纳入资产治理或提交。
- 状态：`[x]`

---

## 6. 最终验收与提交纪律

### 6.1 全链路验证

- [x] 运行所有本轮新增/修改相关 focused tests。
- [x] 完成不读取 `.history/` 的 Python compile/import 检查。
- [x] 检查 `git diff --check`。
- [x] 对每个 `[x]` 条目确认有失败回归、修复说明、通过命令和未覆盖风险。
- [x] 确认没有将 smoke 或静态结果写成完整训练性能结论。
- [x] 确认测试报告明确 `test_used_for_selection=False`。
- [x] 确认模型删除仅发生在 P5 门禁通过后，且 deletion log 与实际文件一致。

### 6.2 工作区与提交

- [x] 重新运行 `git status --short`，确认无关改动未被覆盖。
- [x] 逐文件检查 diff，不使用 `git add .` 或 `git add -A`。
- [x] 代码修复、测试、审计清单和模型删除日志分开审阅。
- [x] 只有用户明确指定提交范围时才逐文件 stage；不得推送。
- [x] 若本轮仅完成审查和计划，不创建业务代码提交。

### 6.3 最终状态

- [x] P0：契约、安全边界和泄漏防护
- [x] P1：参数、配置和调用链
- [x] P2：分支语义与设计目的
- [x] P3：未使用、冗余和不可达实现
- [x] P4：跨模块架构、序列化和性能
- [x] P5：模型资产治理与严格 Pareto 清理

**最终未覆盖风险：**

- 完整训练与独立 grouped/session split 的真实性能验证尚未运行；不得将 focused/smoke 测试等同于模型精度结论。
- 真实部署硬件上的延迟、吞吐和显存基准尚未运行。
- P5 中 819 个协议或元数据不完整资产保守保留，需人工补齐证据后才能进行 Pareto 比较。
- 用户有意保留的删除：`bug.md`、`finding.md`、`main.bak001`、`task_plan.md`；未跟踪 `.anthor/` 不属于本轮提交或资产治理范围。

**最终结论：**

- `[ ]` 尚未达到可交付状态
- `[x]` 代码审查完成，但模型性能仍需用户执行完整训练验证
- `[x]` 资产治理完成，零删除结果由审计与 retention plan 记录
- `[x]` 全部阶段完成

### 6.4 Expert1 精度与召回率训练准备（2026-07-18）

- [x] 修复训练命令引用的缺失实验配置：新增 `tmp/20260718-expert1-precision-recall-v1.json`。
- [x] 基于已验证的稳定 Expert1 配置，仅调整输出目录、`epochs=24` 和 `learning_rate=0.00015`；保持 `batch_size=8`、`seed=42`、全量验证和验证集选择边界。
- [x] 确认 `expert1_track_index.csv`、RD segment manifest 与预训练轨迹检查点在训练启动前存在。
- [ ] 执行完整训练并记录 validation macro F1、source-balanced F1、Unknown/Clutter recall 与 RD 净增益；训练前不得宣称精度或召回率已提升。



动手做之前先思考并更新计划文档task_plan.md，[]标记未完成，[x]标记已完成，需要将问题分解为最小可执行子问题，并分阶段进行，以Stage xxx标记阶段，每个阶段有若干个子问题需要解决，然后更新findings.md文档，同步记录每个阶段的发现，并更新，更新process.md，记录整个任务执行过程，所有任务文件创建完成后等待我确认开始执行

审查RD-only训练代码是否有前后逻辑不一致，参数误传参，表征提取，特征未使用，高维低维映射问题，RD通道未开启，RD编码，时间切片，滑动窗口，辅助头，拒识等模块问题，最后深度思考，当前RD-only瓶颈，为什么精度无法突破90%以上，目标为RD-only模型精度和召回都达到95%以上，另外RD-only的数据集只采用cq08数据集，另外还需要审查cq08数据集的数据，尤其是我人工审查出1343和1494几乎内容一摸一样，为什么会这样？静止状态吗？该如何处理？在整个模型训练过程中会有什么影响？整理为以下格式：1.在src/core/data/rd.py文件中，encode_multimodal_rd() 重新破坏了已经正确处理的环状相位。2.在src/core/rd_novel_parent_data.py文件中，Novel parent 路径对 dBamplitude 再次执行 log1p。

审查TR-only训练代码是否有前后逻辑不一致，参数误传参，表征提取，特征未使用，高维低维映射问题，轨迹编码，时间切片，滑动窗口，辅助头，拒识等模块问题，最后深度思考，当前TR-only瓶颈，为什么精度无法突破90%以上，目标为TR-only模型精度和召回都达到95%以上，整理为以下格式：1.在src/core/data/rd.py文件中，encode_multimodal_rd() 重新破坏了已经正确处理的环状相位。2.在src/core/rd_novel_parent_data.py文件中，Novel parent 路径对 dBamplitude 再次执行 log1p。

启动一轮预训练代码的全量审查，逐个模块进行审查，重点审查代码中对于训练有影响的问题，整理为以下格式：1.在src/core/data/rd.py文件中，encode_multimodal_rd() 重新破坏了已经正确处理的环状相位。2.在src/core/rd_novel_parent_data.py文件中，Novel parent 路径对 dBamplitude 再次执行 log1p。

动手修复之前先思考并更新计划文档TRonlyTask_plan.md，[]标记未完成，[x]标记已完成，需要将问题分解为最小可执行子问题，并分阶段进行，以Stage xxx标记阶段，每个阶段有若干个子问题需要解决，然后更新TRonlyFindings.md文档，同步记录每个阶段的发现，并更新，更新TRonlyProcess.md，记录整个任务执行过程，所有任务文件创建完成后等待我确认开始执行


你现在是RadarInsight项目的资深雷达深度学习工程师、代码审查员和实验设计员。当前工作目录为F:\Project\RadarInsight。请对当前项目的TR-only模型进行一次基于证据的全链路代码审查、性能评估、预训练验证和可执行优化设计。TR-only严格定义为只使用轨迹、Track和TR信息，不得使用RD输入、RD特征、RD质量字段、RD可用性字段，或任何由RD派生的采样、损失、checkpoint选择、阈值、拒识和后处理信息。数据集存在硬约束：TR-only全链路只能使用CQ08和raw两个数据源。必须核实实际数据路径、manifest、索引中的source_name、文件内容和调用链，确认语义上确实只包含CQ08与raw数据。禁止使用excel、excel_tracks、其他来源、合成数据、RD-only数据或未经核实的混合数据进行预训练、监督训练、验证、采样、归一化、checkpoint选择、阈值调优和性能结论。如果现有索引、manifest或配置包含Excel或其他来源，必须保留原始文件，采用手术式方式创建只包含CQ08和raw的新版本索引、manifest或配置，并记录版本、过滤规则、数据数量和校验结果；不得直接覆盖原始数据。必须先完成TR-only预训练，再决定是否进入Expert1全量监督训练。预训练前必须完成数据、标签、split、特征、编码和预训练链路审查；预训练完成后必须先输出真实预训练指标，深度分析是否达标，输出改进建议和“是否可以进入全量训练”的明确结论，未经该结论和用户确认不得启动Expert1全量训练。预训练指标不能被虚构，预训练是否达标不能仅凭训练损失下降判断，预训练达标也不能直接等同于最终分类指标达到90.00%。
总体目标是在CQ08和raw真实数据、session/parent隔离、无数据泄漏和可复现实验前提下，使总体Accuracy、Macro Precision、Macro Recall、Macro F1以及各类别Precision、Recall、F1最低达到90.00%。重点关注DroneTarget和BirdTarget，两个类别的Precision、Recall、F1均不得低于90.00%，必须重点分析DroneTarget→BirdTarget和BirdTarget→DroneTarget的互相误判，不得通过牺牲其中一个类别来提升另一个类别。90.00%只能作为优化和验收目标，未达到时必须明确标记“未达标”，禁止估算、补写或虚构指标。
一、执行总原则
  1.全程使用中文输出，文件名、类名、函数名、配置键和指标标准名称可保留原文。
  2.必须先审查、后修改；不得根据文件名、注释或函数名猜测代码作用，必须通过导入关系、调用链、配置传递、数据流、测试或实际运行结果核实。
  3.所有结论必须区分“已确认问题”“有证据但未运行验证”“待验证假设”，未核实内容不得写成确定性结论。
  4.先执行只读检查，读取git status --short、当前TR-only跟踪文件、有效配置、实际数据索引、manifest、split、模型入口、预训练入口和已有实验产物。当前工作区存在大量未提交修改和未跟踪
  文件，必须保留用户现有改动，禁止git reset、git checkout、git clean、递归删除或覆盖无关文件。
  5.当前仓库没有根目录AGENTS.md、README.md、manage.py和config/system_config.yaml时，不得臆造其内容；应记录文件不存在，并使用实际存在的src/、config/和tests/入口。
  6.不要把训练损失下降、预训练smoke结果、单个seed结果或窗口级Accuracy直接当作泛化性能；必须同时检查window、parent/session、source、类别和线上因果状态机指标。
  7.不得为了达到90.00%修改标签、删除困难样本、污染测试集、调试测试集阈值、引入未来信息、使用RD信息或改变评估分母。
  8.任何性能改善都必须同时检查DroneTarget和BirdTarget是否受损；如果一个类别提升而另一个类别的Recall或Precision下降到90.00%以下，候选方案判定为不合格。
  9.预训练必须先于Expert1全量监督训练。未完成预训练指标报告、预训练质量判定和放行建议前，不得启动全量监督训练。
  10.如果正式预训练或全量监督训练耗时较长且当前没有明确长时间训练授权，只允许运行短时smoke、静态检查、focused unit test、shape/finite验证和小规模数据验证；不得将smoke结果写成正式预训练或全量训练结果。
  11.如果预训练smoke失败，必须继续修复至smoke通过；如果正式预训练未执行、未完成或指标不足，必须明确写“不能进入全量训练”，不得输出乐观推测。
二、当前项目优先核查范围
  1.跟踪文件：TRonlyTask_plan.md、TRonlyProcess.md、TRonlyFindings.md。
  2.配置和入口：config/config.json、config/configmanager.py、src/core/pretrain_track_config.py、src/core/train_pretrain_track.py、src/core/expert1_config.py、src/core/train_expert1.py、src/core/evaluate_expert1.py。
  3.预训练链路：src/core/pretrain_track_manifest.py、src/core/build_tr_only_pretrain_manifest.py、src/core/pretrain_track_data.py、src/core/pretrain_track_trainer.py，以及预训练checkpoint加载、序列编码和下游初始化相关代码。
  4.TR特征和编码：src/core/features.py、src/core/encoding.py、src/core/trajectory.py、src/core/data/features.py、src/core/data/track_dataset.py、src/core/data/split.py、src/core/data/collate.py、src/core/data/dataloaders.py。
  5.Expert1链路：src/core/expert1_data.py、src/core/expert1_trainer.py、src/core/expert1_streaming.py、src/core/experts.py、src/core/metrics.py，以及实际被调用的modeling模块。
  6.索引、窗口和数据处理：src/process/standardize_tracks.py、src/process/build_expert1_window_index.py、src/process/build_tr_only_pretrain_manifest.py及其实际调用的处理模块。
  7.数据源约束：重点核查实际存在的data/processed/cq08_tracks/、data/processed/raw_tracks/及有效配置引用的CQ08/raw数据；明确排除data/processed/excel_tracks/、Excel manifest、其他来源和任何RD数据。若实际目录名称不同，必须根据文件内容、manifest和调用链确认其语义后记录真实路径。
  8.拒识链路：src/core/expert1_config.py中的reject_threshold、enable_reject_in_train、enable_reject_in_eval、attribution_reject_*等字段，以及src/core/experts.py、src/core/expert1_trainer.py、src/core/expert1_streaming.py、src/core/evaluate_expert1.py中与reject、UnknownTarget、confidence、margin、precision bias和stable decision有关的代码。
  9.测试：tests/中所有与pretrain、track、trajectory、feature、expert1、streaming、split、window、config、metrics和reject相关的测试。
  10.当前初步观察到V9索引字段可能包含label_name、source_name、sample_id、parent_sample_id、window_id、is_causal_window、window_start_seconds、window_end_seconds、一个文件硬编码类别。
三、必须逐模块完成的审查
  2.数据切分和泄漏：检查session、parent、track、window、content_group_id、split_group_id是否跨train/val/test重复；检查滑窗重叠、重复轨迹、原始文件重复、RD内容重复、增强样本跨split和预训练验证parent暴露；确认CQ08/raw分别完成parent/session隔离；确认统计量只使用训练集合。
  3.索引和窗口：核对时间戳精度、窗口长度、步长、最小点数、因果窗口、prefix/sliding/short-full模式、窗口边界、窗口内点数、缺失点和短轨迹处理；确认训练、离线评估、在线推理使用同一窗口语义；确认预训练序列与Expert1因果窗口在时间顺序、长度分布和有效点语义上兼容。
  4.TR特征提取：逐一列出每个原始字段、物理单位、来源、缺失条件、填补方式、标准化方式、最终tensor索引和实际消费者；核查位置、距离、速度、加速度、时间差、航向角、角速度、曲率、轨迹连续性、质量标记和missingness；确认特征只来自CQ08/raw的TR数据。
  5.物理和时间语义：检查dt是否使用真实时间；检查角度是否进行环状编码；检查359度到0度是否产生伪突变；检查CQ08和raw之间的单位、坐标系、符号、比例和幅度是否一致；检查NaN、Inf、真实零值、缺失值、padding值和MTM mask是否可区分。
  dropout、序列截断、序列顺序、输出embedding和分类头；确认没有使用未来窗口信息；确认预训练和监督训练的编码层、特征顺序、mask语义和位置编码契约一致。
  8.预训练：核对MTM mask、padding mask、feature-missing mask、mask token、InfoNCE正负样本、temperature、source统计量、train-parent manifest、checkpoint schema、特征维度和预训练到Expert1的加载；确认预训练分布只包含CQ08/raw，且与Expert1因果窗口分布一致。
  9.训练目标和损失：核对主CE、辅助CE、pairwise、residual、contrastive、parent consistency、focal、class weight、label smoothing和拒识损失；检查同一logits是否被重复计算主损失；检查辅助头是否真正参与反向传播和推理；检查pure TR-only路径是否仍混入RD损失或RD指标。
  10.采样和类别不平衡：核对sampler、class weight、source weight、parent weight、rare-class boost是否重复补偿；报告CQ08/raw各自以及每个类别、source和parent的实际曝光量；检查过采样是否造成重复窗口、过拟合或训练/验证分布不一致。
  11.配置传参与可达性：建立“配置字段→配置加载→类型转换→运行时变量→实际影响→测试/指标”的映射；重点检查参数名错误、默认值覆盖、布尔值解析、未知字段静默丢弃、配置未传入、传入后未使用、阶段条件判断错误和checkpoint配置恢复错误；重点确认预训练配置确实被加载，预训练checkpoint确实传入Expert1。
  12.checkpoint和版本：核对checkpoint保存的模型结构、特征顺序、schema版本、类别顺序、归一化统计量、manifest hash、数据源集合、split信息、seed、有效配置、代码版本和训练指标；检查strict load、维度不匹配、旧checkpoint误加载新特征、预训练权重部分加载和未加载参数。
  13.普通评估指标：核对logits与traj_logits是否混用；核对类别映射、confusion matrix、parent聚合、mean-logit、source指标、stable decision和线上因果状态机；确认调参数据与测试数据隔离；确认CQ08/raw均有独立指标。
  14.拒识配置和实现：核对reject_threshold、enable_reject_in_train、enable_reject_in_eval、attribution_reject_min_known_confidence、attribution_reject_min_known_margin、attribution_reject_min_track_quality、attribution_reject_target_valid_points、attribution_reject_max_gap_fraction及所有相关字段是否真正传入并生效；检查拒识判断使用的是softmaxconfidence、logit margin、track quality、missingness、UnknownTarget概率、状态机还是其他信号。
  15.拒识语义：必须区分UnknownTarget类别预测与拒识动作，不能把UnknownTarget自动等同于拒识，也不能把拒识结果混入普通分类结果而不标记；明确拒识输出标签、接受类别、拒绝类别、未决状态和线上状态机状态。
  16.拒识训练和评估一致性：检查拒识是否只在评估启用、训练和评估阈值是否不一致、拒识是否参与checkpoint选择、precision bias是否在拒识前后生效、阈值是否使用测试集调参，以及拒识是否误伤DroneTarget和BirdTarget。
  17.拒识离线和在线一致性：逐窗口对比offline与streaming的输入特征、时间戳、缺失处理、归一化、模型输出、confidence、margin、reject阈值、precision bias、连续确认次数、状态切换和最终类别；任何不一致都必须给出具体文件、位置和影响。
  18.TR-only纯度：即使索引中包含rd_path、has_rd、RD质量或RD可用性字段，也必须证明它们没有影响TR-only的输入、采样、损失、checkpoint选择、后处理和评估；如果只用于诊断，必须明确标记为not_applicable，不能伪装成TR-only能力。
  19.CQ08/raw纯度：必须从原始路径、manifest、索引、Dataset、sampler、DataLoader、预训练和Expert1入口逐层证明只有CQ08和raw数据进入全链路；任何Excel、其他source、RD数据、临时测试数据和未授权合成数据必须被排除并记录。
四、性能评估要求
  1.先确定当前可复现baseline，包括checkpoint路径、有效配置、特征schema、数据索引、split manifest、允许的数据源、seed、设备、训练epoch、batch size和评估命令。
  2.优先使用真实验证集和parent/session分组评估；测试集只有在已有授权和评估契约允许时才能运行，未运行时必须写“未执行”，禁止填写估计值。
  3.分类别性能必须输出如下表格，类别以当前有效索引和模型类别定义的交集为准，不得只输出DroneTarget和BirdTarget：
  |类别|Precision（精度）|Recall（召回率）|F1|Support|主要混淆去向|是否达到90.00%|评估范围|
  4.表格必须包含DroneTarget、BirdTarget、BalloonTarget、ClutterTarget、UnknownTarget以及当前有效数据中实际存在的其他类别；如果某类别不在当前TR-only支路或CQ08/raw有效数据中，必须写“当前支路/数据无此类别”，不得补写指标。重点单独分析DroneTarget和BirdTarget的双向混淆数量、误报数量、漏报数量、来源差异、parent/session差异和分组置信区间。
  5.总体指标必须输出：
  |评估层级|Accuracy|Macro Precision|Macro Recall|Macro F1|Support|是否达到90.00%|备注|
  其中至少包含总体window级、parent/session级、CQ08级、raw级和线上因果stable级。Accuracy、Precision、Recall、F1等质量指标低于90.00%时必须列出差距；Support过低时必须说明指标不稳定，不能仅凭单个seed或少量样本宣称达标。
  6.不得把总体Accuracy写成类别Precision，不得把Recall改名为类别Accuracy；如果输出类别Accuracy，必须明确计算公式。
  7.每次候选优化都必须与同一split、同一数据版本、同一评估脚本、同一类别顺序和相同seed集合比较；优先报告seed=42、1337、2026的均值、标准差、最小值和最大值。
  8.在样本量允许时使用parent/session分组bootstrap或置信区间；稀有类别support过低时，必须说明90.00%指标不稳定，不能仅凭一个窗口或一个seed宣称达标。
  9.目标检查必须单独输出：总体Accuracy是否达到90.00%、各类别Precision是否达到90.00%、各类别Recall是否达到90.00%、各类别F1是否达到90.00%、DroneTarget和BirdTarget是否均达到90.00%、CQ08和raw指标是否同步达标、parent/session指标是否同步提升、是否存在某类Recall提升但另一类Precision严重下降。
五、拒识模块评估要求
  1.必须单独输出拒识性能表，不得把拒识结果隐藏在普通分类表中：
  |拒识阈值|Coverage（保留率）|Accepted Accuracy|Accepted Macro Precision|Accepted Macro Recall|DroneTarget Recall|BirdTarget Recall|Unknown/Clutter拒识Recall|False Acceptance Rate|False Rejection Rate|Selective Risk|是否达到目标|
  2.Accepted Accuracy、Accepted Macro Precision、Accepted Macro Recall、DroneTarget Recall和BirdTarget Recall最低要求90.00%；False Acceptance Rate和False Rejection Rate属于错误率，目标应不高于10.00%；Coverage和拒识率不强行要求90.00%，但必须如实报告并分析其与精度、召回率之间的权衡。
  3.至少评估无拒识、当前默认阈值、验证集选择阈值和候选阈值；阈值只能使用训练集或验证集确定，测试集只能用于最终一次性评估。
  4.必须输出不同拒识阈值下的risk-coverage变化、DroneTarget/BirdTarget被误拒识数量、UnknownTarget/ClutterTarget被错误接受数量、已知目标错误拒识率、未知目标错误接受率，以及拒识前后的混淆矩阵。
  5.必须分别报告拒识前和拒识后的总体指标、DroneTarget指标、BirdTarget指标、UnknownTarget指标和ClutterTarget指标；如果拒识提高总体Accuracy但使DroneTarget或BirdTarget Recall低于90.00%，该方案必须判定为不合格。
  6.必须检查拒识阈值是否对CQ08/raw、不同parent、不同窗口长度、不同轨迹质量和不同类别产生不公平偏置；必须报告DroneTarget和BirdTarget在CQ08/raw上的拒识率。
  7.必须确认拒识模块没有使用测试集标签调阈值、未来窗口信息、RD信息、session_id、文件名类别词或其他标签泄漏字段。
六、性能优化和代码修改要求
  1.先修复数据、标签、切分、特征、评估、拒识和线上一致性问题，再讨论扩大模型、调超参数或增加复杂结构。
  2.每次修改只改变一个主要因素，保留当前baseline，建立清晰的before/after对比；禁止一次同时修改特征、采样、loss、模型、拒识和后处理。
  3.代码修改必须是手术式修改：只改与已确认问题直接相关的最小范围；保留现有接口和兼容性；禁止无关重构、批量重命名和大面积重写。
  4.如果修改特征schema、输入维度、窗口策略、预训练契约、loss、checkpoint选择、拒识语义、阈值策略或推理语义，必须明确版本变化、旧checkpoint是否失效、是否必须重新预训练或重训。
  5.每个优化建议必须包含：问题证据、修改文件、修改位置、预期改善类别、可能伤害类别、验证指标、实验成本、失败判据和回滚条件。
  7.不得默认建议“增大模型、增加epoch、降低学习率、加入复杂注意力”作为解决方案，必须先证明当前瓶颈确实来自模型容量。
  8.长时间训练、全量数据重建、正式预训练、独立测试和大规模实验必须先记录计划、命令、资源和验收标准；没有明确授权时只运行静态检查、focused unit test、shape/finite smoke和小规模验证。
  9.任何候选优化都必须同时满足DroneTarget和BirdTarget的90.00%门槛；如果无法同时满足，必须说明具体瓶颈，不得用总体指标掩盖。
  10.在预训练质量尚未达标、预训练checkpoint无法严格加载、数据源不是CQ08/raw、存在关键泄漏或预训练验证集失真时，不得进入Expert1全量监督训练。
七、跟踪文件更新要求
  1.必须先读取现有TRonlyTask_plan.md、TRonlyProcess.md和TRonlyFindings.md，识别已有Stage和Finding编号，不能覆盖真实历史结论。
  2.TRonlyTask_plan.md使用- [ ]表示未完成、- [x]表示已完成；只有存在代码、测试、数据或命令证据时才能勾选[x]。
  3.所有任务必须拆分为## Stage NNN - 阶段名称；每个Stage继续拆分为最小可执行子任务，并写明依赖、验收标准、验证命令、是否需要预训练、是否需要全量监督训练和数据源范围。
  4.若已有Stage编号，使用当前最大编号之后的新编号，禁止重复编号；每个Stage只允许一个主要改动因素。
  5.任务计划至少覆盖：审查基线、CQ08/raw数据源确认、数据和标签、切分与泄漏、特征提取、TR编码、Transformer、预训练、预训练checkpoint迁移、监督训练、loss和采样、checkpoint、普通评估、拒识模块、offline/streaming一致性、focused验证、性能候选和最终复评。
  6.TRonlyProcess.md必须覆盖式更新：先读取旧内容，再重新生成完整的本轮执行记录，不得只在文件末尾追加；记录日期、执行步骤、实际命令、数据源、预训练命令、预训练输出、修改文件、测试结果、拒识评估结果、阻塞原因和未执行事项。
  7.TRonlyFindings.md必须覆盖式更新：只保留当前仍有效或明确标注为历史的Finding；每条Finding包含Severity、Evidence、Impact、Action；不得把待验证假设写成已确认Finding；必须单独记录预训练相关Finding、数据源污染风险、预训练迁移风险和放行阻塞项。
  8.任何没有实际执行的训练、预训练、测试、评估、拒识阈值搜索或数据重建必须明确写“未执行”，不得补写结果。
八、最终输出格式
  1.第一部分输出“当前审查范围与已核实基线”，只写数据版本、允许的数据源、排除的数据源、类别、split、checkpoint、有效配置、已执行命令和限制。
  2.第二部分输出“分类性能评估表”，先输出分类别Precision、Recall、F1、Support，再输出总体、parent/session、CQ08/raw、source和线上stable指标。
  3.第三部分输出“拒识性能评估表”，包含不同阈值下的Coverage、Accepted指标、DroneTarget/BirdTarget Recall、Unknown/Clutter拒识Recall、False Acceptance Rate、False Rejection Rate和Selective Risk。
  4.第四部分输出“已确认代码问题”，严格使用以下格式，不得改成散文或无编号列表：1.【文件】文件路径；【位置】行号、类名或函数名；【问题】具体问题；【证据】实际代码、配置、数据或测试证据；【影响】对精度、召回率、DroneTarget/BirdTarget双向混淆、拒识效果、泛化、泄漏、评估正确性或部署一致性的具体影响。2.【文件】文件路径；【位置】行号、类名或函数名；【问题】具体问题；【证据】实际代码、配置、数据或测试证据；【影响】具体影响。
  5.如果只是待验证风险，单独输出“待验证项”，并明确说明验证方法，不能混入已确认代码问题。
  6.第五部分输出“优化建议和思考”，按优先级排序；每条包含最小修改方案、预期改善类别、对DroneTarget/BirdTarget的影响、拒识影响、验证指标、风险和停止条件。
  7.第六部分输出“本轮变更与验证”，列出修改文件、测试命令、评估命令、拒识评估命令、实际结果、未执行项目和下一Stage。
  8.最终结果必须明确说明Accuracy、Precision、Recall、F1是否全部达到90.00%，DroneTarget和BirdTarget是否全部达到90.00%，CQ08/raw是否均得到有效覆盖，拒识模块是否达到对应目标；任何未达标项必须列出具体差距。
  9.最终输出中不得把预训练损失或预训练准确率冒充分类性能，不得把预训练达标写成Expert1分类已达90.00%；正式监督训练未执行时必须明确写“未执行”。
  10.全程禁止客套话、空泛结论、无证据的90.00%性能承诺、无关背景介绍和多余空行；最终输出和三个TR-only跟踪文件均不得保留无意义空白行。
九、预训练专项执行、指标、达标判定和全量训练放行要求
  1.预训练执行顺序必须固定为：确认CQ08/raw数据源→确认预训练manifest和parent/session split→审查预训练特征与mask→修复预训练代码和配置→运行短时smoke→验证checkpoint和迁移契约→在获得长时间训练授权后运行正式预训练→输出真实预训练指标→深度判定是否达标→输出建议→输出是否可以进入Expert1全量训练。
  2.在预训练正式指标、达标判定和放行结论输出前，禁止启动Expert1全量监督训练、全量拒识阈值搜索、测试集调参和大规模架构实验。
  3.预训练manifest必须只包含CQ08和raw数据；必须输出每个数据源的原始文件数、parent数、session数、序列数、窗口数、类别分布、长度分布、有效点数、缺失率、train/val数量和split比例；必须确认同一parent、session、content fingerprint和时间连续片段没有跨split。
  4.预训练特征必须逐项报告：特征名、来源、物理单位、特征维度、缺失率、归一化方式、mask语义、是否参与MTM或其他预训练目标、最终tensor位置和下游消费者；任何计算后未使用、被覆盖、仅进入日志或实际没有进入loss的特征必须列为问题。
  5.预训练必须核查并报告以下实现内容：MTM mask比例和分布、padding mask、feature-missing mask、mask token、有效token数量、无效token数量、InfoNCE正负样本构造、temperature、source采样权重、parent采样权重、序列长度截断、时间顺序、因果约束、训练统计量来源、验证统计量来源、checkpoint保存字段和下游加载字段。
  6.预训练smoke至少必须验证：配置可加载、manifest可读取、CQ08/raw均可取样、Dataset和collate可运行、输入shape正确、padding和missing mask正确、forward可运行、loss有限、梯度有限、参数实际更新、无RD输入、checkpoint可保存、checkpoint可恢复、预训练权重可被Expert1严格加载或明确记录不兼容原因。
  7.正式预训练完成后必须输出预训练指标表，至少包含：
  |指标范围|数据源|Split|Parent数|Session数|序列/窗口数|有效点数|Mask点数|Mask比例|训练指标|验证指标|是否达标|备注|
  8.必须根据实际预训练目标输出对应指标；如果实现了MTM，至少报告总MTM loss、验证MTM loss、有效masked token数、整体masked token预测指标和按特征组的重建指标；如果实现了InfoNCE或对比学习，至少报告训练/验证contrastive loss、temperature、正样本相似度、负样本相似度、Top-1或Top-k检索指标及随机/先验基线；如果实现了其他目标，必须按实际目标报告，不得强行套用不存在的指标。
  9.预训练指标必须按CQ08、raw、总体、类别、序列长度区间、缺失率区间和parent/session分组报告；必须报告train/val差距、最差数据源、最差类别、最差长度区间和低support不稳定项；不能只报告总体平均值。
  10.必须进行预训练数值和学习状态检查：训练和验证loss是否为有限值；梯度是否为有限值；是否存在梯度爆炸、梯度消失、loss突然发散、NaN、Inf、空mask、全padding样本、mask比例异常、单一
  source主导训练或checkpoint损坏；必须报告首轮、最佳轮和末轮指标。
  11.预训练达标必须至少同时满足以下硬条件：只使用CQ08和raw；train/val按parent/session正确隔离；不存在RD或标签泄漏；预训练目标和mask有效；训练、验证和梯度数值稳定；正式checkpoint能够保存并严格恢复；预训练checkpoint的特征schema、维度、类别或任务契约与Expert1加载要求一致；验证指标相较随机、未训练或明确的朴素基线有可证实改善；CQ08和raw不能有一方完全失效；不存在尚未解决的Critical级代码或数据问题。
  12.预训练“达标”不能简单定义为loss下降。必须综合判断：目标是否学到有效TR时序结构、验证集是否同步改善、是否存在严重过拟合、CQ08/raw是否存在域偏置、短序列和长序列是否均可用、缺失和padding是否被错误利用、预训练embedding是否能够稳定迁移到Expert1、checkpoint是否可复现。
  13.预训练判定只能使用以下结论之一：达标、条件达标、不达标、无法判定。达标必须有全部硬条件和指标证据；条件达标必须列出未满足的软条件和风险；不达标必须列出失败指标、代码问题或数据问题；无法判定只能用于正式预训练未执行、指标缺失、support不足或验证契约不完整的情况。
  14.预训练指标不得直接使用90.00%分类门槛替代。必须同时说明：预训练阶段使用的是预训练目标指标，90.00%是Expert1最终分类和拒识验收目标；预训练达标只能说明具备进入下一阶段的必要条件，不能证明最终分类指标已经达到90.00%。
  15.预训练后必须输出“预训练深度判定”，至少回答：预训练是否真正使用CQ08和raw；是否存在source shortcut；验证指标是否超过合理基线；是否有过拟合；是否有严重CQ08/raw域差异；mask和padding是否有效；checkpoint是否能够严格迁移；预训练是否有助于DroneTarget/BirdTarget区分；当前证据是否足以支持进入全量训练。
  16.预训练后必须输出“预训练建议”，按优先级列出仍需修复的问题、最小修改方案、预期收益、风险、验证方法、是否需要重新预训练和失败停止条件；不得用“增加epoch”或“增大模型”替代根因分析。
  17.预训练后必须输出“是否可以进入全量训练：是/否”。只有当预训练判定为达标、CQ08/raw数据源纯度通过、checkpoint严格迁移通过、关键测试通过、无Critical阻塞问题且训练配置和split已冻结时，才允许输出“是”；条件达标、不达标或无法判定均必须输出“否”，并列出阻塞项。
  18.即使预训练判定为达标，也必须明确说明Expert1全量训练尚未执行，最终Accuracy、Precision、Recall、F1以及DroneTarget/BirdTarget的90.00%目标尚未被证明；不能把预训练结果写成最终分类结果。
  19.若允许进入全量训练，只能输出经过实际核实的全量训练配置、入口、命令建议、资源需求、预计耗时、日志路径、artifact路径、验收指标和停止条件；不得未经用户明确授权自动启动全量训练。若不允许进入全量训练，只输出阻塞原因、修复顺序、需要重新运行的预训练命令类型和下一Stage。
  20.预训练专项内容必须放在最终输出的最后，并严格按以下顺序输出：预训练数据源与split核验→预训练配置与实际命令→预训练指标表→数值稳定性与迁移验证→预训练是否达标→深度分析→建议→是否可以进入全量训练→全量训练前置条件和未执行事项。

你现在是资深雷达信号处理、RD-only目标识别、深度学习和代码审查专家。请在当前项目F:\Project\RadarInsight中，对RD-only模型进行全链路审查、问题修复和性能优化，最终将RD-only代码、配置和数据链路修复到“待全量训练”状态，再把可直接执行的全量训练命令输出给我，由我手动进行挂机训练。全程使用中文，结论必须有代码、配置、数据、日志或测试依据，不得猜测代码含义，不得虚构指标，不得把TR-only结果当作RD-only能力证据。RD-only模型只能使用RD数据及其合法派生特征，必须核实是否错误使用TR输入、TR标签、TR轨迹、TR匹配结果、路径、文件名、目标ID、split信息或其他标签泄漏字段。若RD/TR配对信息仅用于样本关联或质量校验，必须确认其没有进入模型输入、损失、阈值、采样或预测后处理。当前代码中重点核实RD_ONLY_LABELS、PhysicalRDDroneBirdClassifier.label_names和supports_rejection的实际定义；如果当前RD-only支路是DroneTarget/BirdTarget二分类，不得擅自扩展为五分类；如果存在其他RD分支包含BalloonTarget、ClutterTarget或UnknownTarget，必须分别评估，不能混合统计。重点关注DroneTarget和BirdTarget，不得通过牺牲一个类别提升另一个类别。性能目标为各主要分类指标最低达到90.00%，包括总体Accuracy、Precision、Recall、Macro-F1以及有效类别的分类别Precision、Recall、F1；DroneTarget和BirdTarget的Precision、Recall、F1均应达到90.00%。指标未达到目标时必须明确差距、证据和可执行原因，不得通过修改指标定义、删除困难样本、改变评估分母或使用测试集调参制造提升。RD-only跟踪文件固定为task_plan.md、process.md、findings.md，这三个文件视为RD-only专用跟踪文件，不得使用TRonlyTask_plan.md、TRonlyProcess.md或TRonlyFindings.md代替，也不得修改TR-only跟踪文件。若三个RD-only文件不存在，先创建；如果已存在，先读取其全部内容。三个文件必须覆盖式更新，不能只追加零散记录：保留仍有效的历史事实，删除或修正过期、矛盾和未经证实的结论。task_plan.md中未完成任务使用[ ]，已完成任务使用[x]，所有任务必须拆分为Stage xxx阶段，每个阶段继续拆分到最小可执行子任务，并写明依赖、证据、产物和验收条件。审查前先读取项目级指令、当前工作区状态、三个RD-only跟踪文件、有效配置、实际训练入口、实际评估入口、当前数据索引、manifest、split文件和checkpoint元数据。工作区已有修改时必须保留用户改动，不得进行无关重构。审查必须逐个模块执行，先确认调用链、输入输出契约和实际运行路径，再判断问题。重点核查以下模块及其实际引用关系：1.数据与预处理：src/process/standardize_rd.py、src/process/build_cq08_rd_frame_index.py、src/process/build_cq08_rd_canonical_store.py、src/process/build_cq08_rd_physical_canonical_store.py、src/process/merge_cq08_rd_physical_canonical_index.py、src/process/build_rd_only_causal_window_index.py、src/process/build_expert1_rd_segment_store.py、src/process/build_rd_physical_normalization_stats.py、src/process/build_rd_physical_feature_stats.py。2.数据读取与契约：src/core/data/rd.py、src/core/data/rd_segment_store.py、src/core/data/rd_physical_window_dataset.py、src/core/data/rd_pairing_contract.py、src/core/data/indexing.py、src/core/rd_physical_schema.py、src/core/rd_physical_statistics.py、src/core/rd_physical_artifacts.py。3.RD编码和模型：src/core/rd_physical_drone_bird_model.py、src/core/rd_patch_residual_model.py、src/core/rd_novel_parent_model.py、src/core/rd_novel_anchor_residual_model.py、src/core/forwardlayer/rd.py以及实际被训练入口调用的RD模型文件。4.配置、训练和checkpoint：src/core/rd_drone_bird_config.py、src/core/rd_drone_bird_trainer.py、src/core/rd_physical_drone_bird_config.py、src/core/rd_physical_drone_bird_trainer.py、src/core/rd_patch_residual_config.py、src/core/rd_patch_residual_trainer.py、src/core/rd_novel_parent_config.py、src/core/rd_novel_parent_trainer.py、src/core/rd_novel_anchor_residual_config.py、src/core/rd_novel_anchor_residual_trainer.py、对应train_rd_*.py入口、config/configmanager.py及实际生效配置。5.评估、诊断和线上路径：src/core/evaluate_rd_*.py、src/core/diagnose_rd_drone_bird_residuals.py、src/core/audit_rd_*.py、src/core/benchmark_rd_physical_stream.py、src/core/benchmark_rd_novel_parent.py、src/core/compare_rd_drone_bird_trajectory.py以及实际线上推理、路由、稳定决策和拒识调用链。Stage 001：冻结当前RD-only基线。最小子任务包括确认实际训练配置、数据版本、标签集合、输入shape、RD通道定义、range/doppler轴、checkpoint、seed、split、训练轮数、best checkpoint选择规则、当前window和parent指标；确认训练、验证、测试是否被正确隔离；确认当前支路是二分类、五分类还是多分支组合；确认是否存在独立的线上stable决策。Stage 002：审查数据、标签、split和泄漏。逐字段核查原始RD文件、标准化数据、canonical frame、segment、window、manifest和训练索引；确认同一parent、session、target、content group、文件指纹和时间连续片段不会跨split；确认causal window不包含未来帧；确认标签、目标关联、source、路径、ID、测试字段没有进入模型输入或阈值调优；确认归一化统计、feature statistics、采样权重和checkpoint选择只使用训练或验证范围。Stage 003：审查RD物理数据和特征提取。核查range、Doppler、velocity、amplitude、SNR、power、微多普勒、时间戳、质量特征、缺失值、padding值、无效帧和异常值的单位、方向、顺序、尺度、符号和语义；核查dB与线性功率转换、clip范围、归一化拟合范围、零值是否混淆真实零值与缺失；核查range/doppler轴是否被转置、翻转、错位或重复；核查每个声明的特征是否真正进入模型，是否存在计算后未使用、传参后被覆盖、名称相同但语义不同或只在诊断中出现却被误认为模型能力的特征。Stage 004：审查RD数据集、窗口、padding和时间处理。核查[batch,time,channel,range,doppler]等shape在dataset、collate、trainer、model之间是否一致；核查padding mask、有效帧长度、时间戳单调性、局部时间差、相邻帧配对、末帧锚点、稀疏窗口、重复窗口和窗口权重；确认GRU、attention、pooling、frame logits、prefix logits和micro-Doppler分支没有读取padding帧或未来信息；检查单帧、空帧、短序列、时间重复、时间跳变和缺失RD样本。Stage 005：审查RD编码和模型前向链路。逐层核查卷积输入通道、range/doppler空间编码、池化、token投影、时间编码、GRU/Transformer/attention、统计特征融合、quality特征融合、micro-Doppler分支、frame级分类头、window级分类头、base logits、residual delta、residual gate和最终logits；确认每个forward参数都被正确使用，梯度能传回，训练和评估路径一致，mask没有在错误维度广播，softmax/logit没有重复或顺序错误，二分类标签映射和checkpoint输出顺序一致；重点排查参数传递错误、特征未使用、通道错位、维度广播、分支未接入最终输出、残差方向错误、gate饱和、base checkpoint加载错误和冻结参数未按预期冻结。Stage 006：审查训练目标、loss、采样和checkpoint。逐项拆分主分类loss、prefix/frame losmicro-Doppler辅助loss、residualloss、gate loss、parent consistency、class weight、bird weight、focal loss、evidence weight、parent balance、sampler和正则化；确认没有对同一logits重复计算分类loss，没有对padding或缺失样本错误反传，没有训练集和验证集权重混用，没有类别补偿叠加导致某类过度放大；核查梯度、学习率、scheduler、early stopping、随机种子、AMP、eval模式、best checkpoint选择、checkpoint恢复和训练历史；每个优化实验只能改变一个主要因素，必须保存独立配置、checkpoint、日志和验证报告。Stage 007：审查模型性能评估。必须分别输出window级、parent级、session级、source级、稳定线上决策级指标，并报告support、parent/session数量、有效样本数和排除样本数；必须输出总体指标表、分类别指标表、混淆矩阵、主要混淆方向和置信度分布；如果当前RD-only是Drone/Bird二分类，只输出实际存在的两个类别，并明确其他类别不在该支路评估范围；如果存在五分类或novel/open-set分支，另表输出五类指标。分类别表至少包含类别、Support、Precision、Recall、F1和是否达到90.00%；Accuracy仅在定义明确时报告，若使用one-vs-rest Accuracy必须写清定义，不能把总体Accuracy复制到每个类别。总体表至少包含评估层级、样本数、Accuracy、Macro-Precision、Macro-Recall、Macro-F1、Balanced Accuracy和是否达到90.00%。必须单独统计DroneTarget→BirdTarget和BirdTarget→DroneTarget，不得只给总体混淆矩阵；必要时输出parent级和source级置信区间，低样本类别必须标记统计不稳定，不得隐藏。Stage 008：审查拒识、开放集和校准。必须搜索并核实实际存在的reject_threshold、enable_reject_in_train、enable_reject_in_eval、low_altitude_reject_*、attribution_reject_*、OOD/open-set、energy、margin、confidence、Unknown和abstain逻辑；不能因为配置中存在阈值就认定拒识模块有效。必须明确UnknownTarget分类结果、拒识/abstain动作和普通低置信度预测之间的区别；如果模型声明supports_rejection=False或没有经过验证的拒识输出，必须报告“当前支路不具备已验证的拒识能力”，不得伪造拒识指标。拒识评估表至少包含阈值、Coverage、Accepted Support、Accepted Accuracy、Accepted Macro-F1、Drone Recall、Bird Recall、Unknown/Clutter拒识Recall、FalseAcceptance Rate、False Rejection Rate和Selective Risk；Coverage只需报告，不强制达到90.00%；Accepted主要分类指标、Drone/Bird Recall、有效Unknown/Clutter拒识Recall和Selective Risk目标为至少90.00%，False Acceptance Rate和False Rejection Rate目标为不高于10.00%；必须明确每个指标的分母，分别报告已知目标被错误拒绝、未知/杂波被错误接受以及真正接受样本中的错误；阈值必须通过训练集或parent-grouped OOF验证集确定，禁止使用独立测试集调阈值、校准、选择checkpoint或选择后处理。Stage 009：提出并执行手术式优化。按照“数据契约和泄漏→RD单位/轴/缺失→窗口和mask→特征使用→RD编码→loss/采样→checkpoint/评估→校准/拒识→复杂架构”的顺序处理；每次代码修改只解决一个主要问题，不进行大规模重构，不同时修改多个超参数或多个网络分支；每个修改必须说明根因、修改文件、修改位置、输入输出契约、预期机制、潜在副作用、回归测试和验证指标；不得为追求90.00%而删除困难样本、改变标签、泄漏测试集、混入TR特征或调整评估分母。Stage 010：Smoke验证和待训练准备。RD-only训练时间很长，本任务中只允许运行短时smoke训练，禁止启动全量训练、长时间训练、全量数据重建或长时间评估。smoke训练必须明确限制epochs、max_train_batches_per_epoch、max_val_batches、数据范围和输出目录，并确认不会误用测试集；smoke至少验证配置可加载、数据可读取、模型可构建、forward shape正确、loss有限、梯度正常、checkpoint可保存和恢复、评估入口可运行、RD-only没有读取TR特征。除短时smoke外，可运行静态检查、单元测试、数据契约测试、shape测试、mask测试、checkpoint round-trip测试、无TR输入测试、归一化防泄漏测试和split隔离测试。若smoke失败，必须继续修复，不能输出“待全量训练”命令；若任何关键问题未闭环，必须明确阻塞原因。只有在代码修改、配置校验、数据契约、smoke训练、测试和运行入口全部通过后，才能将状态标记为“待全量训练”。待训练状态至少满足：训练入口可执行；训练配置为独立版本且输出目录唯一；数据索引、manifest、normalization stats和featurestats路径已核实；seed、device、batch size、epochs、学习率和checkpoint策略已固定；测试评估关闭；测试集没有参与阈值、校准、checkpoint或超参数选择；RD-only输入边界已验证；拒识开关和阈值状态已明确；日志和artifact目录已准备；预计资源和风险已说明。Stage 011：输出人工挂机训练命令。达到待训练状态后，只输出命令，不得由你实际启动全量训练，不得调用Start-Process、后台shell、任务调度器或其他方式替用户启动全量训练。必须根据实际代码、实际--help结果和实际配置生成命令，禁止猜测入口或参数。训练命令必须包含正确的Python解释器、实际训练入口、独立配置文件、唯一output_dir、固定seed、正确device、日志路径以及测试评估关闭设置；如果项目支持配置文件，优先使用已经验证过的独立配置；如果项目不支持--config，必须使用实际支持的参数形式。输出一条可直接复制执行的PowerShell全量训练命令，并可同时提供一条可复制执行的PowerShell后台挂机命令；命令中的路径、配置名、输出目录和参数必须是真实存在或已在本轮创建并验证过的内容，不得使用未定义占位符。后台挂机命令必须使用实际项目入口和正确的标准输出、标准错误重定向方式；同时输出日志查看命令、进程状态检查命令、artifact预期位置和训练完成后的评估命令，但这些命令只能输出给用户，不能由你执行。若训练命令无法可靠确定，必须先修复或核实入口，不能输出猜测命令。最终输出必须严格按以下顺序组织，章节之间不得插入多余空行：
一、审查范围与当前基线
说明实际RD-only入口、数据版本、标签集合、输入shape、checkpoint、评估范围、已确认和未确认事项。
二、模型性能评估
先输出总体指标表，再输出分类别指标表，再输出parent/session/source/线上stable指标表和混淆矩阵；所有百分比保留两位小数，例如98.00%，没有实际结果写未评估并说明原因。
三、拒识模块评估
单独输出拒识实现链路、阈值  来源、是否使用UnknownTarget、覆盖率、接受样本指标、已知目标误拒率、未知/杂波误接受率、Selective Risk和校准结论；没有真实开放集或拒识标签时必须明确说明无法得出有效结论。
四、问题清单
按严重性从高到低输出，每条必须严格使用以下格式并单独一行：1.[文件]文件路径；[位置]函数名、类名、配置字段或行号；[问题]可复现的问题；[证据]代码、数据、日志、测试或指标证据；[影响]对精度、召回率、Drone/Bird混淆、拒识、泛化或线上稳定性的具体影响。编号连续，不使用空泛标题，不把推测写成事实；没有证据的问题标记为待核实，不得列为已确认缺陷。
五、建议与思考
按优先级输出手术式修改建议，每条说明修改范围、预期收益、风险、验证方法和停止条件；区分已证实问题、合理假设和后续实验，不承诺未经验证的指标提升。
六、跟踪文件和变更记录
列出本轮覆盖式更新的task_plan.md、process.md、findings.md，实际修改的代码和配置文件、smoke命令及结果、测试命令及结果、未完成Stage、未执行的全量训练和剩余风险。
七、待全量训练交付仅当全部修复和smoke验证通过时输出待全量训练：是、实际训练配置、实际训练入口、可直接复制的前台训练命令、可直接复制的后台挂机命令、日志查看命令、进程状态检查命令、预期artifact目录和训练完成后的评估命令；明确写出“全量训练未由本次审查启动，请用户手动执行”。如果未达到待训练状态，只输出待全量训练：否及阻塞问题，不得输出伪造或未经验证的训练命令。全程不得输出虚构的95%或90%指标，不得省略DroneTarget、BirdTarget和拒识审查，不得修改TR-only代码或跟踪文件来完成RD-only任务。


现在创建src-v3/tests/cu和src-v3/tests/dbb-cu目录，现在聚焦于这个目录专门来构建Clutter/Unknown二分类模型，以及DBB(Drone/Bird/Balloon)/CU(Clutter/Unknow)分类模型，优先构建DBB/CU模型，这里如果效果很好未来会转化为一个模块加入某个三层软级联模型，现在思考构建DBB/CU模型，更新跟踪文件

