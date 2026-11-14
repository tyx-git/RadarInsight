# RadarInsight

**液态神经网络驱动的低慢小目标雷达辨识系统**：以 Liquid V2 连续时间递归主干为核心，从 CQ08 原始回波一路走到 L1/L2/L3 开放集决策、ONNX 级联导出与 R-RTSP 实时推流。

```text
CQ08 原始数据 ─▶ 分层索引 / manifest ─▶ canonical RD 张量库 ─▶ 逐事件特征
             ─▶ L1 路由 ─▶ L2 子类 ─▶ L3 已知性 ─▶ 语义动作 ─▶ ONNX / R-RTSP
```

数据规模：**1,549** 个 parent、**29,724** 个 TR 事件、**106,026** 个 RD 帧、**8** 类逐事件特征。

## 核心创新：液态神经网络在低慢小辨识中的应用

低慢小目标的回波不是等间隔帧序列，而是带 Δt 的事件流：相邻事件可能只隔几十毫秒，也可能隔数百毫秒；缺帧、重复时间戳、无效时间戳是常态。常规做法是先把事件对齐成固定帧率，再交给 RNN 或 Transformer——时间信息被压缩成帧号或位置编码，递归状态与真实物理时间脱钩。本项目把液态神经网络直接放进雷达事件流：状态按真实 Δt 连续演化，时间常数可学习且有界，异常时间步由显式时间策略处理。

### 1. 连续时间状态更新，而不是固定步长递归

- 每个隐藏维独立学习时间常数：`tau = tau_min + (tau_max - tau_min) * sigmoid(w)`，恒落在 `[0.05, 30]` 秒。
- `alpha = 1 - exp(-dt / tau)`，状态更新为 `r_next = (1 - alpha) * r_prev + alpha * m`。Δt 大，alpha 接近 1，旧状态自然衰减；Δt 小，状态更稳，不需要第二个 forget gate 与时间竞争。
- 首事件 `alpha = 1`：观测直接初始化状态，不虚构历史；Δt 统一截断到 120 s，被截断的行带 `dt_clamped_mask` 供训练与诊断使用。
- 同一单元对不同时间尺度自适应：慢目标积累长时证据，密集事件保留短时细节，不需要按帧率重采样。

### 2. 原始状态 + 计算用归一化视图

- 产品状态是 raw memory `r`：`n = LayerNorm(r)`，`(c, g) = F(x, n)`，`m = g * c + (1 - g) * r`；更新 `r_next` 后不再归一化写回。
- 归一化只服务于 candidate / gate 计算，递归状态保持原始尺度；长序列不会因反复归一化被压到同一量纲、丢掉幅度信息。
- padding 不调用 cell：raw memory 原样穿过，且 padding 必须是每条序列的后缀——训练、分块与续接共用同一契约。

### 3. 时间原因码进入状态转移，而不是简单 mask

每个事件携带 `dt_reason`（`FIRST_EVENT` / `POSITIVE_DELTA` / `ZERO_DELTA`；padding 不进入 cell）与 `dt_valid`，同一个 cell 按原因选择更新规则：

- 正 Δt：标准指数门控；
- 零 Δt：基础 alpha 用 1 ms 下限计算，再由 `invalid_dt_update_mode` 覆盖——默认 `bounded_observation` 使用 0.25 固定观测增益，避免同一时刻的重复事件被当作新信息；`density_weighted` 按零间隔序号加权（上限 3× 增益）；
- 无效 Δt：同样使用有界观测增益，坏时间戳不会打断状态；
- `strict_skip` 行走 `skipped_step`，不调用 candidate / gate，状态精确恒等——不伪造时间步，也不浪费算力。

这些规则位于 cell 内部的状态转移，而不是 loss 侧 mask，训练与推理口径一致。

## 系统链路

| 层 | 职责 | 关键模块 |
|---|---|---|
| L0 数据层 | CQ08 MAT/DAT → 分层索引 → canonical RD → 事件引用缓存 → 8 类特征缓存 | `radar_ln3/data`、`radar_ln3/cache`、`radar_ln3/features` |
| L1 路由 | 目标 / 背景；course 与观测增量适配器 | `model/l1`、`training/l1` |
| L2 子类 | Drone / Bird / Balloon；baseline 锚点与残差 / 直连推理模式 | `model/l2`、`training/l2`、`training/l2_anchor` |
| L3 已知性 | 冻结端点表示 → 独立 knownness 头 → 温度缩放 / conformal → 审计与动作 | `model/l3`、`training/l3`、`training/l3_heads`、`decision` |
| 服务 | ONNX 导出与级联推理、R-RTSP 推流 | `inference/onnx`、`core/server` |

数据层产物：

- **canonical RD**：按 parent 落 1,549 个 npz（约 2.63 GB）+ 约 10 MB compact 事件索引（tensor 帧号 ↔ MAT 帧号）。
- **事件引用缓存**：约 16 MB，按 `data_path` 去重文件级 `path / size / sha256`（3,098 个文件），行内只保留事件身份与帧映射。
- **逐事件特征**：约 12.6 MB 覆盖 29,724 个事件，tr-observation、increment、course、rd-quality、rd-physical4、micro-doppler evidence / stable8 frame / stable8 summary 在同一 bundle 物化。

L3 动作集：`Drone / Bird / Balloon / Clutter / UNKNOWN_REJECTED / PENDING_ABSTAIN`。knownness 头不读分类 logits、不回改 L1/L2 端点；校准由温度缩放、conformal 阈值与 Bird 恢复门组成，阈值同时受已知类接受精度 / 覆盖下限与 Unknown 误接受上限约束。

## 快速开始

环境：Windows + PowerShell 7、Python ≥ 3.10；CUDA 可选（`-Device cpu` 可全程 CPU 运行）。

```powershell
# 可选：把产品包装进当前环境
python -m pip install -e "./src[dev]"
```

### 一键全链路（可续跑）

在仓库根目录执行，按 `cache → L1 → L2 → L3` 顺序推进并自动跳过已完成阶段：

```powershell
.\src\scripts\train.ps1 -Device cuda

.\src\scripts\train.ps1 -Device cuda -RefreshData     # 只重建数据与缓存层
.\src\scripts\train.ps1 -Device cuda -StopAfterL1      # 只训练到 L1
.\src\scripts\train.ps1 -Device cuda -ResetState       # 清空编排状态后重跑
```

### 单阶段入口

三个阶段共享 `-Root`，便于隔离正式产物与本地验证：

```powershell
.\src\scripts\trainL1.ps1 -Root artifacts/radar_ln3 -Device cuda -Overwrite
.\src\scripts\trainL2.ps1 -Root artifacts/radar_ln3 -Device cuda -Overwrite
.\src\scripts\trainL3.ps1 -Root artifacts/radar_ln3 -Stage all -Device cuda -Overwrite
```

### 数据层单独重建

```powershell
.\src\scripts\rebuild.ps1 -Device cuda
```

非默认 Root 默认不登记产品 manifest，正式登记与本地实验天然分离。

### 测试

```powershell
python -m pytest src/tests -q
```

### 分阶段工具

| 工具 | 作用 |
|---|---|
| `materialize_canonical_rd.py` / `build_l3_cache.py` | 物化 canonical RD 与 L3 缓存 |
| `train_l3_heads.py` / `calibrate_l3.py` | 训练 knownness 头、生成 calibration |
| `audit_l3.py` / `freeze_l3_audit_manifest.py` / `report_l3_audit.py` | 证据物化、审计冻结与报告复用 |
| `ExportONNX.py` / `ExportFullCascadeONNX.py` | 单端点与全链 ONNX 导出 |
| `EvaluateCQ08ONNX.py` / `CompareASDAndCQ08ONNX.py` / `EvaluateASD.py` / `InferASD.py` | 对拍、评估与离线推理 |

## 服务后端（R-RTSP）

`src/` 是产品侧（数据处理、训练、评测、开放集推理与 R-RTSP 服务）的独立源码边界，不含前端界面。

- 读取 `.asd` 数据包并按 R-RTSP 协议回放、推流；
- 运行 L1 KnownGate → L2 DBB specialist → L3 semantic decision 开放集推理；
- 提供 transport-neutral 推理入口，输出 PT=96/97 观测与可选 PT=98 模型结果；
- 负责 L3 温度缩放、类别条件候选集、knownness、路由与语义动作解析；
- 负责模型加载、校准、评测、导出与运行配置。

在 `src/` 下启动（扫描一个 ASD 包目录，每个包完整推理一次并发送 PT=98）：

```powershell
python -m core.server --asd-dir <asd-package-dir> --host 0.0.0.0 --port 8554 --inference-mode package
```

`core/cli.py` 是后端命令入口；`core/` 负责运行编排，`radar_ln3/` 负责雷达数据、特征、模型、开放集与决策逻辑。协议细节见 [R-RTSP 协议规范](<src/docs/R-RTSP协议规范-v1.md>)。

## 配置与产物

| 配置 | 作用 |
|---|---|
| `src/config/model.json` | 主干、任务、液态单元时间常数、训练超参、特征维度 |
| `src/config/radar_ln3.json` | L1/L2 正式产物与缓存 binding |
| `src/config/radar_ln3_l3.json` | L3 Root、heads、calibration、`knownness_training` 超参 |
| `src/config/radar_ln3_policy.json` | 接受精度 / 覆盖 / 误接受约束 |
| `src/config/radar_ln3_decision.json` | 决策阈值与门限覆盖 |
| `src/config/manifests.json` | 数据身份 + 产品 registry |
| `src/config/source.json`、`src/config/runtime.json` | 数据源与运行时 |

产物布局（`<root>` 默认 `artifacts/radar_ln3`）：

```text
<root>/l1/l1.pt                          <root>/l2/l2.pt
<root>/l3/heads/                         <root>/l3/representations/
<root>/l3/radar_ln3_calibration[.exploratory].json
<root>/l3/radar_ln3_audit_manifest.json  <root>/l3/radar_ln3_audit_report.json
<root>/l3/audit-evidence.jsonl           <root>/pipeline/{state,logs,commands}
```

L3 的 heads、representations、calibration、audit manifest/cache、证据与报告路径全部由 `radar_ln3.artifacts.l3_layout()` 从 `--root` 推导，Root 决定全部产物位置。

## 仓库结构

```text
src/                    正式产品实现（唯一运行入口）
  core/                 运行编排：cli、缓存编排、推理管线、R-RTSP 服务
  radar_ln3/data/       索引、manifest、路径边界、canonical RD、CQ08/ASD 适配
  radar_ln3/features/   物理特征编码、micro-Doppler 证据、特征缓存
  radar_ln3/model/      L1/L2/L3 模型、配置与 ONNX 输入契约
  radar_ln3/training/   batch 装配、L1/L2 训练、L3 头训练与校准审计
  radar_ln3/decision/   L3 决策、阈值策略、校准数据结构
  radar_ln3/inference/  ONNX 运行器与级联推理
  radar_ln3/liquid_v2/  Liquid V2 递归运行时
  scripts/              编排入口（.ps1）与分阶段工具（.py）
  config/               模型 / 数据 / 策略 / 决策 / 运行配置
  tests/                数据契约、缓存、模型契约、校准审计与配置测试
  docs/                 R-RTSP 协议规范
data/                   CQ08 原始数据与派生缓存（原始数据与缓存不入库）
artifacts/              训练产物根目录（本地，不入库）
```

## 开源协议

本项目基于 [MIT License](<src/LICENSE>) 开源。
