# RADAR-LN3 V21d 因果 d_range 稳定性实验

## 研究问题

在V21 KnownGate的已知目标/已知Clutter局部任务中，加入因果前缀距离残差稳定性，能否改善`BirdTarget/ClutterTarget`边界，同时不损害Drone、Balloon及Known Target总体接受率。

## 正式特征

对每个parent按事件因果顺序处理。每个已到达事件中的每个真实canonical帧生成：

`d_range = MTD detected_range_m - 当前事件TR filtered range_m`

截至当前事件，把全部已观察且两端有效的`d_range`放入前缀集合，使用MATLAB默认口径的样本标准差`std(ddof=1)`；至少2个残差观测后有效。模型输入为`clip(std_m / 75 m, 0, 1)`，valid只作门控，不拼入分类输入。

该定义只使用当前及历史事件。缺RD或缺TR range的当前事件不添加残差，但已有前缀统计可继续使用；padding不推进统计。final状态等价于整条已观察航迹的残差样本标准差。

## 被否决的旧草案

原`v21d`第6列在每个事件内计算`std(MTD range - 同一个TR range)`。由于同一事件的TR range是常数，它数学上退化为事件内`std(MTD range)`，无法表达检测距离相对航迹滤波距离的跨事件不稳定性。旧seed42配置保留追溯，但正式训练入口必须拒绝；正式内部标识为`v21d_prefix`，使用独立缓存、合同与artifact。

## 历史先验边界

历史MATLAB探针的`Bird≈12.4 m / Clutter≈1.9 m`已精确追溯为两个代表轨迹前9个MAT帧的样本标准差：Bird track 333约`12.414 m`，Clutter track 1868约`1.925 m`。它不是类别总体均值；Bird 333位于V16 validation，Clutter 1868位于V16 test，且“只取前9帧”不是正式在线统计规则。因此该数字只解释特征来源，不作为readiness门禁、性能目标或泛化证据。

为核实字段语义，本轮人工审计重新读取了已被历史探针暴露的test parent 1868全部54个MAT帧；未运行模型test评估，也未据此选择公式、clip或门槛。自动`prepare`仍为test payload访问0，但这不能表述为“该特征从未看过test”，原V16 test明确不是其盲holdout。

正式train-only诊断必须在986个known train parent上报告完整分布。若总体方向不复现历史代表轨迹，仍可执行预注册的单因素训练以检验与现有主干的互补性，但不得宣称“Bird总体必然更高”。

## 配对与裁决

- Control：V21 seed43 development Baseline。
- Treatment：从同一个seed43随机初态完整训练20 epoch；复制所有共享参数，只把新增Fusion列置零，不从Baseline best微调。
- 主假设标签：BirdTarget、ClutterTarget。
- 安全标签：DroneTarget、BalloonTarget。
- 主指标：development validation parent-final Macro-F1增量严格大于0.02。
- 方向守卫：Bird与Clutter合计净救回为正且各不为负；Drone/Balloon各不净受损；Known Target Recall与Clutter Recall均不下降。
- Clutter validation仅9个parent，必须报告support与5000次按原始四标签分层的paired parent bootstrap。

即使通过，也只能进入多seed与train parent-grouped OOF确认。Unknown不是本地模型类别；test、正式拒识、完整级联、stable、P95和部署均保持未执行。
