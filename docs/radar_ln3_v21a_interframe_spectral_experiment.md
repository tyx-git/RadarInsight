# RADAR-LN3 V21a 谱形时间一致性单因素实验

状态：训练前冻结协议。V21a只评估KnownGate局部已知目标/Clutter边界，不定义Unknown拒识、完整级联或部署性能。

## 1. 修正原因

原始train-only诊断将“谱形时间一致性”定义为相邻RD帧的速度谱相似与漂移。旧seed42候选误实现为单个原始回波帧内部相邻STFT窗的Bhattacharyya affinity；该量具有不同的物理时间尺度，并且minimum随当前事件帧数增加存在比较次数偏置。因此旧候选不得进入正式训练或与seed43 Baseline比较。

## 2. 唯一实验变量

正式内部标识为`v21a_interframe`，只增加一组六维当前事件因果证据：

1. `current_event_adjacent_rd_velocity_profile_cosine_mean`
2. `current_event_adjacent_rd_velocity_profile_cosine_minimum`
3. `current_event_adjacent_rd_peak_velocity_drift_mean_fraction`
4. `current_event_adjacent_rd_peak_velocity_drift_max_fraction`
5. `current_event_zero_velocity_energy_fraction_std`
6. `current_event_zero_velocity_energy_fraction_minimum`

每个当前事件最多带入上一有效事件的末个canonical RD帧作为桥接，然后将当前帧的dB幅度转换为相对线性功率，沿21个距离行求和得到384维速度谱；计算归属于当前事件的相邻帧速度谱余弦相似度mean/minimum、相邻峰速度漂移mean/max（按canonical速度跨度归一化），以及±2 m/s零速能量占比的std/minimum。RD缺失事件会重置桥接，禁止跨缺口伪造连续性；没有形成至少一对比较帧时六维置0并标记invalid。valid只作内部门控，不拼入分类器。

该特征不读取未来事件、parent-final统计、标签、路径、frame index或采样模式，不读取原始`.dat`。缓存只含train/validation已知类别事件；UnknownTarget和test payload均排除。

历史探索边界：最初用于提出该方向的临时诊断没有按split过滤，并在每个标签的`sample()`中独立重置`RandomState(3)`。逐字重建metadata后，108个探索track中train/val/test分别为77/17/14。因此当前prepare依旧保持test payload访问0，但原V16 test对V21a特征构思已经暴露，今后不得作为该特征家族的无偏最终holdout。V21a本轮只做development配对；正式泛化确认需使用train parent-grouped OOF、多seed以及新的独立holdout或新数据版本。

## 3. 配对协议

- Baseline：V21 development champion seed43。
- Treatment：seed43，从与Baseline相同的随机初始参数开始；复制所有共享参数和buffer，新增Fusion两列精确置零。
- 不加载Baseline best checkpoint做微调。
- 两臂均为全参数20 epoch、相同DataLoader seed、相同parent顺序、相同loss与checkpoint规则。
- prepare重建seed43随机Baseline并要求初始state SHA与正式seed43 summary逐值一致，防止代码变化导致“同seed但不同初态”。
- 原seed42 `v21a`配置只保留错误候选追溯，训练入口必须拒绝。

## 4. 评估与裁决

- 主单位：development validation parent-final KnownGate二分类。
- 主要正向信号：Macro-F1相对seed43 Baseline严格提升`>0.02`，Clutter/Bird/Balloon三类各自net rescue均不得为负且合计为正，同时Drone net rescue不得为负、Known-target Recall不得下降。
- 报告Known target与Clutter P/R/F1、Drone/Bird/Balloon逐类变化、混淆矩阵和5000次按原始四类分层的paired parent bootstrap。
- Clutter validation support固定为9，单点指标不得作为放行证据。
- train-only类别分布、当前事件RD帧数以及实际相邻比较对数与六维特征的相关性只作物理归因，不用于调公式、阈值或checkpoint。
- prepare的唯一train-only特征门槛预先固定为：Clutter-vs-Bird和Clutter-vs-Balloon各至少一维符合预期方向的parent AUC达到`0.65`；其余分布与相关性只作归因。门槛失败即停止，不允许用validation补证。
- 即使局部信号通过，也只允许进入多seed或train parent-grouped OOF确认。

原V16 test为“未执行且对V21a特征构思已暴露，禁止用于无偏最终结论”；OOF、完整级联、正式拒识、stable、P95和部署均为“未执行”。
