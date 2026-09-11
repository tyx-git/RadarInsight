# V27c：因果 TR Motion 对冻结 V27a L2 端点的独立增益验证

> 状态：设计已登记，尚未训练。  
> 实验类型：known-only L2 分类特征消融；不训练 knownness、不拟合 calibration/L3。  
> 目标端点：封存的 V27a Main epoch-20，而不是较早的 V27 L2 epoch-11。

## 1. 要回答的问题

`tr_motion_features` 目前由共享数据加载器生成，却没有进入 V27、V27a 或
V27b-r6 的模型输入。V27c 只回答一个窄而可归因的问题：

> 在已经使用 TR、RD spatial、RD physical4、stable8 summary，以及 V27a
> 可靠逐帧 stable8 late-residual 的 L2 分类端点上，9 维因果运动数值是否还能
> 为 Drone / Bird / Balloon 分类提供安全的、样本对齐的额外信息？

它不是“运动特征总体无效”的判决。若 V27c 的独立晚期分支失败，只能否定本实验的
**冻结端点 + motion-only late residual** 载体；早期融合进 Liquid V2 主干仍需作为
新的、单独登记的实验处理。

## 2. 冻结基座与动机

V27a Main 是当前已知分类端点：223 个 validation DBB parent 的 accuracy 为
96.41%、macro-F1 为 96.14%。其剩余主要问题是 4 个高置信 `Drone -> Bird`
错误，而非 Bird recall：

| 类别 | Precision | Recall | F1 | 主要混淆 |
|---|---:|---:|---:|---|
| Drone | 98.20% | 96.46% | 97.32% | 4 → Bird |
| Bird | 91.53% | 96.43% | 93.91% | 1 → Drone |
| Balloon | 98.11% | 96.30% | 97.20% | 1 → Drone |

因此 V27c 不回退到未使用 V27a 微多普勒残差的旧端点，也不重做一个混入多种新变量的
全主干训练。冻结 V27a Main 后，任何相对提升都必须来自 motion 分支，能直接回答
“它是否仍有独立增益”。

冻结输入必须逐项绑定：

- V27a Main checkpoint 文件 hash、state hash、model-contract hash、训练 config/
  manifest hash；
- V27a train/validation reference cache 的文件与 payload hash；
- 封存 V27a paired validation report 及其 `[[109,4,0],[1,54,1],[1,1,52]]`
  confusion matrix；
- CQ08 grouped index hash、V27a evidence audit 的 event-ID 顺序 hash、parent 顺序、label、length 与 padding；
- `causal-tr-motion-v1` schema、9 个字段名、scale、clip=5.0 与 mask mode。

目录迁移后，V27a 原 checkpoint 中仍保存旧的 `config_path`。V27c 不调用会按当前
路径重建 metadata 的旧 loader；它应采用 V27b-r6 已验证的迁移兼容读取方式：直接加载
state，再同时校验封存的 checkpoint identity、state hash、model contract 和 reference
cache payload hash。路径变化不是放宽 provenance 的理由。

## 3. 受检特征：只使用 V9 motion schema

V27c 的唯一新信息源为：

```text
schema: causal-tr-motion-v1
dimension: 9

reported_speed_acceleration_scaled
radial_acceleration_scaled
course_turn_rate_scaled
geometric_speed_scaled
reported_geometric_speed_abs_error_scaled
range_rate_scaled
radial_range_rate_abs_error_scaled
radial_sign_flip
causal_path_straightness
```

该 schema 以 parent/session 为边界重置，只读取当前及历史 TR 观测；其一部分导数
依赖正的有效 `dt`。无效分量的数值必须为零，`tr_motion_valid_mask` 只能用于数值门控，
**不得拼接为分类输入、coverage 输入或 learned quality 输入**。

V10 的 `causal-tr-observation-increment-v1` 是另一个、无 dt 的 9 维 schema；它在历史
审计中有更均衡的 validity，但不是本次变量，不能与 V9 motion 一起塞进 V27c。

### 当前可用性审计

在与 V27 L2 相同的 DBB train/validation parent 上，所有 parent 至少有一个 motion
有效事件，但“任一运动分量有效”的事件比例存在类别差异：

| Split | Drone | Bird | Balloon |
|---|---:|---:|---:|
| train | 58.28% | 51.34% | 68.27% |
| validation | 60.91% | 50.61% | 54.23% |

所有样本都来自 CQ08，故没有可用于跨 source 排除 shortcut 的 source 变化。此差异使
“把 validity/coverage 偷渡成类别输入”不可接受；V27c 必须通过零值与置换反证据确认
模型真的使用了运动**数值**。

## 4. 模型与对照

V27c 使用一个独立的、因果的 motion late-residual。Liquid V2、V27a Main、stable8
residual、分类头和所有共享参数全部冻结；motion 分支不读取 frozen logits、softmax、
margin、预测类别、parent ID、track ID、source ID 或任何 future event。

对每个事件：

\[
u_t = v_t \odot M_t
\]

\[
a_t = \mathbb{1}[\lVert u_t\rVert_\infty > 0]
\]

\[
z_t = \phi_{\mathrm{bias\text{-}free}}(u_t)
\]

\[
h_t =
\begin{cases}
\operatorname{GRUCell}(z_t, h_{t-1}), & a_t=1\\
h_{t-1}, & a_t=0
\end{cases}
\]

\[
\Delta_t = \operatorname{bounded\_zero\_sum}(W_{\mathrm{bias\text{-}free}}h_t)
\]

\[
\ell_t^{\mathrm{V27c}} = \ell_t^{\mathrm{V27a\ Main}} + \Delta_t
\]

其中 token projection 与输出 projection 均无 bias；`a_t` 仅由已经 mask 后的数值是否
为零决定，而非 `M_t.any()`。这使“所有 motion 数值置零、原 validity mask 保持不变”时，
整条分支从初态起严格输出零，不能把缺失率、有效事件数或 prefix 长度变成隐式类别证据。
有真实非零 motion 证据后，状态可以因果地保留至后续事件；padding 永不更新状态。

`bounded_zero_sum` 沿用 V27a 的安全形状约束：三类修正和为零、每维绝对值有登记上界
（初始工作值 3.0），从而不通过整体 logit 平移伪造 gain。它不是 open-set gate。

### 对照关系

| Arm | 输出 | 可训练参数 | 作用 |
|---|---|---|---|
| `frozen_control` | 原样 V27a Main logits | 无 | 精确、不可变的主基线 |
| `motion_treatment` | V27a Main + motion residual | 仅 V27c motion branch | 检验数值运动证据的独立增益 |

control 不训练一个“零 motion 分类器”：那会引入额外 residual 容量或全局类先验，反而混淆
对照。V27c 的必要反事实是 treatment 在全零 motion 下逐位退回 frozen control，而不是
另一个可借助 mask/偏置变化的伪 control。

## 5. 数据、训练与范围

- 训练/验证：只使用 V27a reference train/validation DBB parent；parent 集、顺序、标签、
  length 与 padding 必须逐项等同于冻结 cache。
- 分类标签：只含 Drone / Bird / Balloon；Unknown、Clutter、L1、knownness、quality、
  calibration、L3、test 均不读取、不训练、不拟合。
- motion sidecar：由相同 canonical index 与 `CQ08PointSequenceDataset` 在
  `tr_motion_schema="causal-tr-motion-v1"` 下重新生成；必须核验每个 parent 的事件顺序和
  length 与 reference cache 相同。
- loss：继续使用 parent-level `parent_final_fixed_parent_count_ce` 与冻结 V27 L2 的
  train-only class weights；禁止改回 batch 类组成归约。
- 训练：三个随机初始化 seed `42/43/44`，固定 dataloader seed、固定 20 epoch、固定
  primary epoch=20；不看 validation 后选最优 epoch。
- 优化器、batch size、weight decay、gradient clip 与 strict CUDA deterministic policy 需在
  config 中固定并写入 manifest；训练启动前必须设定
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`（或 `:16:8`），否则 fail closed；
  任何增量训练、主干解冻、V27a 重新训练或 V10 schema 都自动构成另一个实验。

## 6. 必须通过的合同 smoke

仅保留能防止错误结论的边界测试：

1. V9 schema 的维度、名字、clip、invalid=0、mask dtype、padding=0 与 parent reset；
2. sidecar 与 V27a train/validation cache 的 parent ID、label、length、padding 精确对齐；
3. 封存 V27a Main 的 sequence/final confusion matrix 精确重放；
4. future motion suffix 改动不改变独立计算的前缀输出；padding 不产生 GRU 更新；
5. 将全序列 motion 数值置零、但保留原 validity mask，输出必须与 frozen control 完全一致；
6. 改变一个有效且非零 motion token 必须能改变 treatment 输出；
7. checkpoint/cache/manifest 绑定 schema、frozen endpoint、index、代码和 parent registry hash。

不增加泛化的单元测试，也不以 smoke 代替真实训练评估。

## 7. 评价、反事实与晋级规则

主评价是 parent-final；eligible-prefix 为单独的 parent-balanced 安全诊断。每个 seed 都报告：

- accuracy、macro-F1、balanced accuracy、三类 P/R/F1、confusion matrix；
- `Drone -> Bird`、`Bird -> Drone` 及所有 DBB 双向混淆；
- treatment 相对 frozen control 的逐 parent correct/wrong 转移与 stratified paired bootstrap；
- 按 motion 非零事件比例、最终事件是否有非零 motion、轨迹长度分层的分类结果；这些字段
  只用于审计，不能进模型；
- 4 个既有高置信 `Drone -> Bird` 错例的逐事件 residual 和最终动作；
- value-zeroing、length-preserving、parent support-pattern-preserving，以及 component-support-
  preserving event-value permutation 四个 label-free deterministic 输入反事实。最后一个固定
  parent length、事件位置及九个分量的零/非零模式，只错配数值，专门排除 dt/缺失模式与
  coverage shortcut。置换只用于
  评估时确认 treatment 依赖正确对齐的数值，不能用于训练、选择 checkpoint 或产品推理。

### 预注册安全门

相对 frozen control（215/223 correct、4 个 `Drone -> Bird`、1 个 `Bird -> Drone`、
52/54 Balloon correct），任一 seed 出现以下情况即不晋级：

- parent-final correct 少于 215；
- 任一类别正确数低于 frozen control（Drone <109、Bird <54、Balloon <52）；
- zero-motion 反事实未严格回放 frozen control；
- length-preserving permutation 后效果几乎不变，表明声称的增益不能归因于正确对齐的
  motion；或 support-pattern-preserving permutation（固定非零 token 模式）后输出不变，
  表明分支可能只依赖有效率/coverage；或 component-support-preserving value permutation
  后数值优势仍保留，表明可能依赖逐分量 dt/缺失模式，而非 motion 数值。

三个 seed 必须全部通过安全门。“存在正向信号”的最低条件是：至少两个 seed 至少将四个
冻结 `Drone -> Bird` 错例中的一个真正纠正回 Drone、将 `Drone -> Bird` 从 4 降至至多
3、Bird precision 严格高于 91.53%，且 parent-final macro-F1 点估计相对 control 为正。
这两个或更多 seed 还必须在 parent support-pattern 和 component-support-preserving 置换下
失去该数值优势。另要求三 seed 平均 parent-final macro-F1 点估计增益为正，并且每个 seed
的 paired-bootstrap macro-F1 95% 下界不低于一个 validation parent 的影响量（`-1/223`）。
只有同时满足，才允许把 motion 视为可进入下一阶段的候选特征。

即使通过，V27c 也只允许提出“需要一个新的、受控的 early-fusion 主干实验”的建议；它
不自动改写 V27 baseline，不接入 V27b-r6 knownness，不拟合 L3，更不能声称 Unknown
拒识或跨 source 泛化改善。

## 8. 结果解释树

| 观察 | 正确结论 | 后续动作 |
|---|---|---|
| 多 seed 纠正 Drone→Bird 且安全门通过，zero/permutation 反事实支持 | V9 motion 在 frozen V27a endpoint 外仍有独立分类信息 | 另立 early-fusion 实验，重新训练 L2 与其 stable8 residual，不能直接替换产品 |
| 指标不变或安全退化 | 该 late-residual 载体无可采纳收益 | 不宣称 V9 motion 整体无效；若仍值得研究，单独比较 V10 无 dt 增量或早期融合 |
| gain 只在 validity/coverage 代理下存在，零值或置换反事实不支持 | 可能是缺失/对齐 shortcut，不是 motion 数值收益 | 失败；不得通过加入 mask、coverage 或 quality 头来“修复” |
| 仅已知分类改善但 Unknown 未评估 | 仍只是 known-only 分类结果 | 不外推到 knownness、拒识或 L3 |

## 9. 计划产物

实现后，V27c 应在本目录内拥有独立的 config、manifest、motion sidecar cache、checkpoint、
rolling training log、per-seed report、三 seed summary 与本实验的结果文档。大型冻结输入可
引用已封存 V27a Main lineage，但每次运行必须在 V27c manifest 中记录其文件及 payload hash。
