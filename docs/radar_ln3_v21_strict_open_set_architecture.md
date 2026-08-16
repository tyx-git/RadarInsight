# RADAR-LN3 V21严格Open-Set架构

状态：当前唯一有效的产品架构规范。V13-V20中的`dbb_cu_gate`、`cu_specialist`、五类soft posterior和历史拒识状态机均为复现实验语义，不再定义V21产品行为。

## 1. 三层职责

### L1：KnownGate

- 内部标识：`known_gate`。
- 学习类别：`KNOWN_TARGET_EVIDENCE`与`KNOWN_CLUTTER_EVIDENCE`。
- 已知目标训练人口：DroneTarget、BirdTarget、BalloonTarget。
- 已知背景训练人口：ClutterTarget。
- UnknownTarget不进入类别映射、loss、采样、normalization、checkpoint选择或静态阈值拟合。
- L1只提供目标/背景证据。没有OOF阈值时，二分类argmax仅是局部诊断，不是产品动作。

### L2：DBB specialist

- 内部标识：`dbb_specialist`。
- 学习类别：DroneTarget、BirdTarget、BalloonTarget。
- 只有L1目标证据达到接受条件后，L2子类证据才有资格生成最终目标类别。
- 子类证据不足时进入Unknown拒识候选；Unknown不是第四个DBB类别。

### L3：Semantic Decision Layer

- 内部标识：`semantic_decision_layer`。
- 不训练新的目标类别，不再存在CU专家模型。
- 只保存版本化评分器、train parent-grouped OOF阈值、质量下限、最小有效点数、连续确认、稳定状态规则、模型/config/hash绑定及动作原因。
- L3必须同时读取L1目标/背景证据、L2子类证据和观测质量；soft posterior可保留审计，但不能代替动作规则。

## 2. 互斥动作表

| 观测资格 | L1已知目标 | L1已知Clutter | L2子类 | 最终动作 |
|---|---:|---:|---|---|
| 合格 | 接受 | 不接受 | 接受Drone/Bird/Balloon | Drone/Bird/Balloon |
| 合格 | 不接受 | 接受 | 不适用 | Clutter |
| 合格 | 不接受 | 不接受 | 不适用 | UNKNOWN_REJECTED |
| 合格 | 接受 | 不接受 | 证据不足 | UNKNOWN_REJECTED |
| 不合格 | 任意 | 任意 | 任意 | PENDING_ABSTAIN |
| 合格但目标/Clutter规则冲突 | 接受 | 接受 | 任意 | PENDING_ABSTAIN |

`PENDING_ABSTAIN`不是真实Unknown预测，也不计作Unknown拒识成功。`UNKNOWN_REJECTED`是动作；数据中的UnknownTarget是评估真值。两者必须分字段记录。

## 3. V21局部实验边界

V21、V21a、V21b、V21c、V21d只回答KnownGate能否更好地区分已知目标和已知Clutter。它们不拟合拒识阈值，不执行Unknown动作评估，不修改DBB specialist，也不运行完整级联。

- Baseline：V16因果TR/RD液态主干，训练人口改为Known-only，并重算Known-only train normalization。
- V21a：只增加谱形时间一致性。
- V21b：只增加RD峰与TR radial的模态一致性。
- V21c：只增加目标range row能量占比iso。
- V21d：只增加MTD检测距离与TR滤波距离差值的当前事件标准差。

各证据字段采用逐特征缺失语义，不允许插值或伪造物理量：TR radial speed缺失只使V21b两项无效；TR range缺失或当前事件少于两帧只使V21d无效；V21a、V21c及其他具备输入的证据继续保留。无效值固定为0，并由内部valid门控，valid本身不进入分类器。

五个运行均从随机初始化完整训练20 epoch。Treatment复制Baseline全部共享初始参数，新增Fusion列精确为零；数据顺序、loss、checkpoint规则和seed保持一致。

## 4. 评估与封存

- 主单位：validation parent-final KnownGate二分类。
- 必报：Known target/Clutter P/R/F1、Macro-F1、混淆矩阵、Drone/Bird/Balloon目标接受Recall、Clutter support。
- Clutter validation仅9个parent，必须报告support并执行parent分层bootstrap 95%区间。
- test、parent-grouped OOF阈值、正式拒识、stable、P95、完整级联和部署在V21局部实验中均为“未执行”。
- 任一局部正向结果只允许晋级多seed和OOF确认，不得直接形成线上阈值或Unknown性能结论。

## 5. 历史兼容边界

以下名称只允许出现在旧配置、旧checkpoint和旧报告复现路径：

- `dbb_cu_gate`
- `cu_specialist`
- `ROUTE_TO_CU_SPECIALIST`
- 将Unknown作为普通二分类/五分类输出的历史语义

新代码、配置、artifact和报告必须使用`known_gate`与`semantic_decision_layer`。禁止把旧CU模型改名后继续充当L3。
