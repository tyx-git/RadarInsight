# RadarInsight

**低慢小目标雷达可辨识系统**：从 CQ08 原始 MAT/DAT 到 L1/L2/L3 级联推理与 R-RTSP 实时推流，全链路共享同一份数据身份。

```
原始数据 ─▶ 分层索引 / manifest ─▶ canonical RD 张量库 ─▶ 8 路特征缓存 ─▶ L1 ─▶ L2 ─▶ L3 ─▶ ONNX / R-RTSP
```

数据规模：**1,549** 个 parent、**29,724** 个 TR 事件、**106,026** 个 RD 帧；**8** 路特征缓存、**230+** 项自动化测试。

## 创新点

### 1. 全链路一份数据身份

- 一份 manifest + 统一 `DatasetView`：`view_hash` 同时绑定事件集合、split、标签域与顺序，特征缓存、模型、校准、审计都消费同一个视图，不再各自解释训练集。
- 数据身份与派生登记分层：`data_manifest_hash` 只覆盖索引/pairing/audit 主体，`product` registry、`audit.scope`、`audit.cache` 属于派生登记——新增缓存描述符或 L3 登记**不会**让已有特征缓存失效。
- 外部输入只在入口校验一次：manifest 的 size/hash 全量校验只做一遍，训练/校准/审计阶段直接复用；事件缓存、特征缓存各自一次校验并把结果记在对象上。

### 2. 物理可观测的微多普勒证据

- 23 维宽带帧证据：残余多普勒 8–90 m/s 六个能量分带、总能量占比、RMS/90 分位偏移、正负对称性、分辨单元熵与 top-4 集中度、分辨单元峰值计数、边带/包络 SNR、调制频率剖面。
- 只对"采样模式可观测且调制 Nyquist 足够"的**可靠帧**做逐维中位数，选出 8 维稳定证据（`STABLE8_INDICES = 1,2,3,7,11,13,14,22`），归一化统计量随缓存一起物化，训练与推理保持同一口径。
- L3 侧再派生 47 维证据摘要（23 维可靠帧均值 + 23 维标准差 + 有效帧比例，按事件因果构造），供 open-set 头直接消费。

### 3. Liquid V2 连续时间主干

- L1/L2 共用 Liquid V2 递归单元：以 raw memory `r` 为状态，只在校验 candidate / gate 时生成归一化视图，
  `r_next = (1-α)·r + α·(g·c + (1-g)·r)`。
- 逐事件消费 Δt、缺失掩码与缺失原因码；TR 14 维观测与 RD 空间编码（masked-mean）在同一主干内融合，任务分支各自持有 RD 编码器。

### 4. 增量 + 残差两级学习

- **L1 路由**：`ObservationIncrementAdapter` 把逐观测变化编码成 Liquid residual，叠加基础状态；配合 `course` 圆周不相干特征，学习"目标 vs 背景"。
- **L2 子类**：在冻结参考端点上做残差——`EvidenceResidual`（stable8 侧车）+ motion adapter，并以 baseline logits 为锚点；同一套权重支持 `residual / direct / direct_grouped` 三种推理模式与四组 arm（standard/robust × control/motion）。

### 5. 冻结端点上的独立 open-set 已知性

- L3 不读分类 logits、不改 L1/L2 端点：每个 route 一个独立二元 knownness logit（L1 侧 target/clutter 双路由，L2 侧共享 gate），输入是冻结端点已经产出的因果轨迹与特征，避免"用分类置信度冒充已知性"。
- 校准 = 温度缩放 + conformal 阈值 + Bird 恢复门（clutter / target 两路）；动作集 `Drone / Bird / Balloon / Clutter / UNKNOWN_REJECTED / PENDING_ABSTAIN`。阈值搜索同时受"已知类接受精度/覆盖下限"与"Unknown 误接受上限"约束，审计侧再用 Wilson 上界复核，结果封存为带 `artifact_sha256` 的 calibration。
- 质量事实（RD quality）只作为 knownness 输入与诊断记录，不参与语义动作判定，保证"分类置信度"与"已知性"两条通路互不污染。

### 6. 可审计、可复用的产物链

- 每个产物的 `file_size / sha256` 在落盘时一次算出并登记；calibration artifact hash、checkpoint `state_sha256`、审计证据 `data_binding` 形成可追溯链。
- 审计证据单独物化为 JSONL：每 parent 的 endpoint logits 与 knownness 分数，按六字段数据绑定（manifest / endpoint / feature cache / knownness source / head / parent 集合）；二次出报告直接复用，不重跑 forward。
- `formal_available` 提升在封存与登记之前完成，manifest 登记的 hash 与唯一一次落盘完全一致。

### 7. 大规模数据的缓存工程

- **canonical RD 张量库**：按 parent 落 npz（1,552 个文件、约 2.6 GB）+ 约 10 MB 的 compact 事件索引（tensor 帧号 ↔ MAT 帧号），事件读取零歧义。
- **事件引用缓存**：约 16 MB，文件级 `path/size/sha256` 按 `data_path` 去重（3,098 个文件），行内只留事件身份与 frame 映射。
- **8 路特征缓存**：约 12.6 MB 覆盖 29,724 个事件——tr-observation、increment、course、rd-quality、rd-physical4、micro-doppler evidence / stable8 frame / stable8 summary。

### 8. 端到端服务化

- ONNX 级联导出（单端点、全链、逐 parent 对拍脚本），PyTorch 权重与 ONNX 输出一致性可回归。
- R-RTSP 服务端：RTSP/RTP 会话、鉴权、流式批次与 ONNX 推理后端，配协议规范文档。

## 分层架构

| 层 | 职责 | 关键模块 |
|---|---|---|
| L0 数据层 | CQ08 MAT/DAT → 分层索引 → canonical RD → 事件引用缓存 → 特征缓存 | `radar_ln3/data`、`radar_ln3/cache`、`radar_ln3/features` |
| L1 路由 | 目标 / 背景，观测增量 residual + course | `model/l1`、`training/l1` |
| L2 子类 | Drone / Bird / Balloon，残差与直连三模式、baseline anchor | `model/l2`、`training/l2`、`training/l2_anchor` |
| L3 已知性 | 冻结端点表示 → 独立 knownness 头 → 校准 / 审计 / 动作 | `model/l3`、`training/l3`、`training/l3_heads`、`decision` |
| 服务 | ONNX 导出与级联推理、R-RTSP 推流 | `inference/onnx`、`core/server` |

## 快速开始

在项目根目录执行：

```powershell
# 全链路可续跑入口：cache → L1 → L2 → L3（自动跳过已完成阶段）
.\src\scripts\train.ps1 -Device cuda

# 只重建数据与缓存层
.\src\scripts\train.ps1 -Device cuda -RefreshData

# 只训练到 L1
.\src\scripts\train.ps1 -Device cuda -StopAfterL1
```

单阶段入口共享 Root 参数，便于隔离实验与正式产物：

```powershell
.\src\scripts\trainL1.ps1 -Root artifacts/radar_ln3 -Device cuda -Overwrite
.\src\scripts\trainL2.ps1 -Root artifacts/radar_ln3 -Device cuda -Overwrite
.\src\scripts\trainL3.ps1 -Root artifacts/radar_ln3 -Stage all -Device cuda -Overwrite
```

数据层单独重建（索引 → canonical RD → 特征缓存 → baseline anchor）：

```powershell
.\src\scripts\rebuild.ps1 -Device cuda
```

非空 Root 覆盖前会生成同级 zip 备份；非默认 Root 默认不登记产品 manifest，实验与正式登记天然分离。

## 服务后端（R-RTSP）

`src/` 是服务后端的独立源码边界，只保留雷达数据处理、开放集推理与 R-RTSP 服务逻辑，不包含前端界面。

- 读取 `.asd` 数据包并按 R-RTSP 协议回放、推流；
- 运行 L1 KnownGate → L2 DBB specialist → L3 semantic decision 开放集推理；
- 提供 transport-neutral 推理入口，输出 PT=96/97 观测与可选 PT=98 模型结果；
- 负责 L3 温度缩放、类别条件候选集、knownness、路由与语义动作解析；
- 负责模型加载、校准、评测、导出以及运行配置。

服务启动（在 `src/` 下执行，扫描一个 ASD 包目录，每个包完整推理一次并发送 PT=98）：

```powershell
python -m core.server --asd-dir <asd-package-dir> --host 0.0.0.0 --port 8554 --inference-mode package
```

R-RTSP 协议规范见 [`docs/R-RTSP协议规范-v1.md`](docs/R-RTSP协议规范-v1.md)。`core/cli.py` 是后端运行入口；`core/` 负责运行编排，`radar_ln3/` 负责雷达数据、特征、模型、开放集与决策逻辑。

## 配置与产物

| 配置 | 作用 |
|---|---|
| `config/model.json` | 主干、任务、残差、训练超参、特征维度 |
| `config/radar_ln3.json` | L1/L2 正式产物与缓存 binding |
| `config/radar_ln3_l3.json` | L3 Root、heads、calibration、`knownness_training` 超参 |
| `config/radar_ln3_policy.json` | 接受精度 / 覆盖 / 误接受约束 |
| `config/radar_ln3_decision.json` | 决策阈值与门限覆盖 |
| `config/manifests.json` | 数据身份 + 产品 registry |
| `config/source.json` / `runtime.json` | 数据源与运行时 |

产物布局（`<root>` 默认 `artifacts/radar_ln3`）：

```text
<root>/l1/l1.pt                 <root>/l2/l2.pt
<root>/l3/heads/                <root>/l3/representations/
<root>/l3/radar_ln3_calibration[.exploratory].json
<root>/l3/radar_ln3_audit_manifest.json   <root>/l3/radar_ln3_audit_report.json
<root>/l3/audit-evidence.jsonl  <root>/pipeline/{state,logs,commands}
```

L3 的 heads、representations、calibration、audit manifest/cache、证据与报告路径全部由 `radar_ln3.artifacts.l3_layout()` 从 `--root` 推导。

## 目录结构

```text
core/            产品入口（cli、cache 编排、推理管线、R-RTSP 服务）
radar_ln3/data/  索引、manifest、路径边界、canonical RD、CQ08/ASD 适配
radar_ln3/features/  物理特征编码、micro-Doppler 证据、特征缓存
radar_ln3/model/ L1/L2/L3 模型与配置
radar_ln3/training/  batch 装配、L1/L2 训练、L3 头训练与校准审计
radar_ln3/decision/  L3 决策、阈值策略、校准数据结构
radar_ln3/inference/ ONNX 运行器与级联推理
radar_ln3/liquid_v2/ Liquid V2 递归运行时
scripts/         编排入口（ps1）与分阶段工具（py）
config/          模型 / 数据 / 策略 / 决策 / 运行配置
tests/           数据契约、缓存、模型契约、校准审计与配置测试
docs/            R-RTSP 协议规范
```

## 质量保障

- 230+ 项测试覆盖数据契约、缓存绑定、模型契约、L3 校准/审计、配置口径与服务端。
- 数据层有三重完整性校验：索引/文件 hash、TR/RD 关系一致性、统一视图覆盖范围；特征是逐事件身份绑定的。
- 模型侧有 contract 测试与 ONNX 对拍脚本，checkpoint 记录 `model_state_sha256` 与训练配置 hash。
- 编排入口带 `state.json` 与命令快照，支持按阶段续跑与输入 hash 追踪。

## 开源协议

本项目基于 [MIT License](LICENSE) 开源。
