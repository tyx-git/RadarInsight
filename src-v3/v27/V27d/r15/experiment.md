## Material Passport

- Origin Skill: `academic-research-suite / experiment-agent`
- Origin Mode: `run + validate`
- Origin Date: `2026-09-12`
- Verification Status: `ANALYZED`
- Version Label: `V27d-r15-post-fusion-residual-v3-seed42`

# V27d-r15：V10 observation increment 迁移到 V27 L1

## 当前有效版本：post-fusion residual v3

- 实验包：`V27d/r15`，使用独立的 v3 配置、manifest、cache 和 checkpoint 路径。
- 直接基座：V27 L1 known-gate epoch-14。
- 唯一变量：V10 `causal-tr-observation-increment-v1` 的 9 维逐观测变化量。
- 当前注入边界：`V27 原始 fusion -> V10 residual -> Liquid V2`。
- control：observation increment 输入全零；残差分支容量和初始化与 treatment 相同。
- treatment：输入 V10 observation increment；它表示相邻雷达报告之间的变化，不是按秒计算的变化率。
- seed：仅 `42`；两臂使用相同训练数据、batch size `2`、20 epochs、loss 和数据加载协议。
- 训练 parent：`986`；闭集 validation parent：`232`。
- Unknown 不进入训练；test payload 未访问；正式 L3、conformal 和 deployment 均关闭。

V10 的 9 个值为：reported speed delta、radial speed delta、course delta、geometric step distance、range delta、radial/range direction agreement、radial sign flip、consecutive-step direction cosine 和 causal path straightness。无 physical-time division；同秒事件仍按观测顺序推进状态，padding 不更新状态。

## 边界修正

此前 v2 实现把 observation embedding 拼接在 V27 fusion 的输入端，并重新构造 fusion Linear/LayerNorm。该实现与实验设计中“V27 fusion 后注入”的边界不一致，因此 v2 结果不与本版本合并。

当前实现保留并冻结原始 V27 fusion，先完成完整的 V27 fusion，再将 observation adapter 输出作为 48 维 residual 加到 fusion 输出，最后进入 Liquid V2。这样 control 的零输入可以严格回放 V27 L1 基座，同时 treatment 分支仍能获得梯度。

对应代码位置：

- `model.py`：`R15ObservationIncrementAdapter` 将 9 维输入编码为 48 维 residual；
- `model.py`：`base_events + encoded` 位于原始 V27 fusion 与 Liquid V2 之间；
- `protocol.py`：v3 配置、manifest 和 seed=42 约束；
- `smoke.py`：增加 `control_matches_v27_base` 边界合同检查。

contract smoke 已通过：

- post-fusion residual 输出宽度与 V27 `liquid_input_dim=48` 一致；
- control 零输入与 treatment 初始输出一致；
- control 初始输出与 V27 基座在有效事件上的差异不超过 `1e-6`；
- V10 形状、有限值、invalid=0、同秒语义、因果前缀、padding、Unknown/test 隔离均通过。

## 数据视图修复

open-set role 使用与其样本集合一致的 mixed course cache：`train`/`validation` 使用 known-only cache，`representation_fit`、`representation_development` 和 `calibration` 使用对应 mixed cache，并对路径和 SHA-256 做来源校验。这样修复了 Unknown parent `cq08|track|1546` 无法在 known-only cache 中匹配的问题。

## 训练结果

两臂均完成 `20/20` epochs，训练 loss 始终为有限值。

| arm | final train loss | mean epoch loss | clipped steps | max gradient norm |
| --- | ---: | ---: | ---: | ---: |
| control | 0.0257622 | 0.0380993 | 32 | 1746.8417 |
| V10 observation increment | 0.0070103 | 0.0224927 | 18 | 1439.3506 |

较低的 treatment train loss 只能说明训练集拟合更好，不能替代 validation 和 open-set 评估。

## 闭集分类结果

L1 的两个类别是 `KNOWN_TARGET_EVIDENCE`（Drone/Bird/Balloon）和 `KNOWN_CLUTTER_EVIDENCE`（Clutter）。validation support 为 `232`，其中 target `223`、clutter `9`。

| arm | accuracy | target P/R/F1 | clutter P/R/F1 | macro-F1 |
| --- | ---: | --- | --- | ---: |
| V27 L1 epoch-14 | 99.5690% | 99.5536% / 100.0000% / 99.7763% | 100.0000% / 88.8889% / 94.1176% | 96.9470% |
| control | 99.1379% | 99.5516% / 99.5516% / 99.5516% | 88.8889% / 88.8889% / 88.8889% | 94.2202% |
| post-fusion V10 treatment | 100.0000% | 100.0000% / 100.0000% / 100.0000% | 100.0000% / 100.0000% / 100.0000% | 100.0000% |

post-fusion treatment 相对 matched control：

- accuracy：`+0.8621 pp`；
- macro precision、macro recall、macro-F1、balanced accuracy：均为 `+5.7798 pp`；
- paired validation transition：`2` 个 `wrong -> correct`，没有 `correct -> wrong`。

按原始标签拆分的 recall：

| 类别 | control | treatment | 变化 |
| --- | ---: | ---: | ---: |
| DroneTarget | 113/113 = 100.0000% | 113/113 = 100.0000% | 0 pp |
| BirdTarget | 55/56 = 98.2143% | 56/56 = 100.0000% | +1.7857 pp |
| BalloonTarget | 54/54 = 100.0000% | 54/54 = 100.0000% | 0 pp |
| ClutterTarget | 8/9 = 88.8889% | 9/9 = 100.0000% | +11.1111 pp |

发生变化的 validation parent 为：`cq08|track|3581`（Bird，纠正）和 `cq08|track|2903`（Clutter，纠正）。

## Bird 与 Unknown 错误

Bird error 表示 `BirdTarget` 被预测为 `KNOWN_CLUTTER_EVIDENCE`；Unknown error 表示 `UnknownTarget` 被预测为 `KNOWN_TARGET_EVIDENCE`。错误率越低越好。

| role | arm | Bird error | Unknown error |
| --- | --- | ---: | ---: |
| representation_development baseline | V27 L1 epoch-14 | 3/114 = 2.6316% | 36/38 = 94.7368% |
| representation_development | control | 3/114 = 2.6316% | 35/38 = 92.1053% |
| representation_development | treatment | 1/114 = 0.8772% | 34/38 = 89.4737% |
| calibration baseline | V27 L1 epoch-14 | 0/56 = 0.0000% | 13/15 = 86.6667% |
| calibration | control | 1/56 = 1.7857% | 13/15 = 86.6667% |
| calibration | treatment | 0/56 = 0.0000% | 13/15 = 86.6667% |

在 `representation_development` 中，treatment 相对 control 少 `2` 个 Bird error、少 `1` 个 Unknown error；在 calibration 中，treatment 与 control 相同。该结果方向良好，但只有一个 seed，且 calibration 只有 `15` 个 Unknown，单个样本约等于 `6.6667 pp`，不能据此宣称稳定 open-set 能力。

## 结论与停止决定

当前 v3 单 seed 结果为 **`promising_but_unconfirmed`**，不是产品晋级结论。

在正确的 post-fusion residual 边界下，seed42 的 treatment 同时改善了闭集 validation、Bird 错误和 development Unknown 错误，且没有出现 paired regression。但该结论仅来自一个 seed，仍可能受随机初始化影响；旧 v2 的两 seed 结果不能用于补充稳定性，因为 v2 的注入边界不同。

建议后续至少补跑 seed44、seed45、seed46、seed47，并保持当前 v3 边界和配置不变；在多 seed 结果稳定前，不将该变量接入 V27 默认 L1 或正式 L3/deployment。

## 产物

- 配置：`config/V27d-r15-v10-observation-increment-l1-v3.json`
- manifest：`manifests/V27d-r15-manifest-v5.json`
- manifest SHA-256：`a3f5cd9aac08fbd8dd604a0524b4d847d2ff22e301ddf6fd34e2bdcfbcad55df`
- observation cache：`cache/V27d-r15-observation-increment-cache-v6.pt`
- cache manifest：`cache/V27d-r15-observation-increment-manifest-v6.json`
- cache payload SHA-256：`3f148b1566ec3d1681b913757f8fa34281d2faacc0b8673a3deae11b1b4be928`
- contract smoke：`reports/v5/V27d-r15-contract-smoke-v4.json`
- seed42 evaluation：`reports/v5/V27d-r15-seed42-evaluation.json`
- one-seed summary：`reports/v5/V27d-r15-one-seed-summary.json`
- checkpoint：`checkpoints/v5/V27d-r15-control-seed42.pt`
- checkpoint：`checkpoints/v5/V27d-r15-v10_observation_increment-seed42.pt`

checkpoint state hash：

- control：`24ca3a3277b8c0eb78702cd59263909eac91f01fc185b2d7c550fc4e704369b6`
- treatment：`064e7ee8d756c9e328e060c25f24455193a6026f9d12682127ea5b5641e46aa3`

## 实际执行与未执行

已执行命令：

```powershell
$env:PYTHONPATH = "src-v3"
$config = "src-v3/v27/V27d/r15/config/V27d-r15-v10-observation-increment-l1-v3.json"

.venv\Scripts\python.exe -B -m compileall -q src-v3/v27/V27d/r15
.venv\Scripts\python.exe -B -m v27.V27d.r15.cli --project-root F:\Project\RadarInsight --config $config prepare
.venv\Scripts\python.exe -B -m v27.V27d.r15.cli --project-root F:\Project\RadarInsight --config $config materialize
.venv\Scripts\python.exe -B -m v27.V27d.r15.cli --project-root F:\Project\RadarInsight --config $config smoke
.venv\Scripts\python.exe -B -m v27.V27d.r15.cli --project-root F:\Project\RadarInsight --config $config train --arm control --seed 42
.venv\Scripts\python.exe -B -m v27.V27d.r15.cli --project-root F:\Project\RadarInsight --config $config train --arm v10_observation_increment --seed 42
.venv\Scripts\python.exe -B -m v27.V27d.r15.cli --project-root F:\Project\RadarInsight --config $config evaluate --seed 42
.venv\Scripts\python.exe -B -m v27.V27d.r15.cli --project-root F:\Project\RadarInsight --config $config summarize
```

已执行：上述编译、manifest prepare、observation cache materialize、contract smoke、两臂 seed42 训练、checkpoint provenance/state hash 校验、seed42 evaluation 和 one-seed summarize。另对报告、cache、checkpoint 做了独立 sealed payload 复核，并确认没有残留 r15 CLI 进程。

未执行：seed44 及其他 seed、重新训练 V27 基座、Unknown 进入训练、test payload 评估、正式 L3/conformal/deployment、将 observation increment 接入产品默认模型。

## 统计解释边界

- 本报告没有 p-value、置信区间或多 seed 方差估计；seed42 只能作为单次诊断结果，不能证明变量具有稳定收益。
- validation 是 parent-level 样本，`223` 个 target 与 `9` 个 clutter 的类别比例不均衡；因此 accuracy 不能单独代表 Clutter 性能。
- `representation_development` 只有 `38` 个 Unknown，`calibration` 只有 `15` 个 Unknown；单个错误分别约对应 `2.6316 pp` 和 `6.6667 pp`，变化应视为诊断信号。
- 当前 treatment 与 control 使用相同的 seed42、数据、训练轮数和 loss，但 v3 与此前 v2 的注入边界不同；v2 结果不得合并到本报告。
- 该结果是描述性比较，不构成产品部署或正式开放集能力声明。

## 历史版本说明

此前 v2 版本的两 seed 训练和评估产物仍保留在 `checkpoints/`、`reports/`、`runs/` 的旧路径中，仅供审计。由于 v2 使用的是 fusion 输入端拼接，不代表当前 post-fusion residual 方案，不能与本 v3 结果合并计算。
