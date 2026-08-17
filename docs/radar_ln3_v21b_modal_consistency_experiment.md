# RADAR-LN3 V21b模态一致性单因素实验

状态：训练前冻结。V21b只回答当前事件RD峰值速度与TR径向速度的一致性，是否改善KnownGate中的`ClutterTarget/DroneTarget`边界。Bird与Balloon只作为安全守卫，不是预期受益类别。

## 单因素变量

Treatment相对V21 seed43 Baseline只新增两维当前事件证据：

1. `target_row_rd_peak_vs_tr_radial_abs_deviation_median_over_85mps`
2. `target_row_rd_peak_vs_tr_radial_abs_deviation_max_over_85mps`

每个真实canonical RD帧只在零距离偏移目标行寻找峰值速度，计算有符号RD峰值速度与当前TR `radial_speed_mps`之差的绝对值，再在当前事件内取median/max，除以85 m/s并截断到`[0,1]`。不读取前后事件、parent最终统计、标签、路径、source、PRT或未来信息。

TR径向速度缺失时两维值均为0并标记invalid；valid只用于内部门控，不拼入分类输入，不插值或复制邻点。V21冻结的train/validation缓存直接SHA复用，本实验不重新读取raw数据。

## 配对协议

- Control：V21 development champion seed43。
- Treatment：相同seed、DataLoader顺序、20 epoch、class weights、loss、prefix辅助、checkpoint选择和模型主干。
- 初始化：复制seed43 Baseline的全部共享随机初态，只将新增两列Fusion权重精确置零；不从训练完成的Baseline checkpoint微调。
- 全参数训练：主干与新增列均参与全部20 epoch训练。
- 旧seed42 V21b配置只保留追溯，禁止作为正式可比Treatment运行。

## 历史探索与正式口径

最初的临时探索不能作为正式正向证据。该脚本按标签从整个V16索引抽样，93个track中train/val/test为68/10/15；因此原V16 test已参与特征构思，未来不能再作为该特征的纯盲最终holdout。脚本还只取每3帧中的第1帧、最多8个事件，并使用了“每个距离行argmax索引再取最大索引”的表达式，而不是当前正式公式的零偏移目标行能量峰。历史Clutter/Drone AUC=`0.809/0.765`与当前两维不是同口径结果。

当前正式特征也不宣称等同于MAT `debug_info.detected_doppler_mps`或raw target-record Doppler；它严格测试已命名的“canonical RD零偏移目标行能量峰 vs 当前TR radial”代理量。

## 训练前物理审计

冻结train-only parent审计显示：Clutter相对Drone的两维均值较低，但方向无关parent AUC仅`0.567577/0.509008`。均值与秩方向并不一致，说明两类分布重叠且存在异质子群。因此本实验定位为“可能通过与主干条件交互产生互补收益”的验证，而不是已证明有效的特征。Clutter/Bird/Balloon统计只用于误伤审计，不能扩展V21b的主假设。

训练前不得依据validation重选符号、公式、clip、聚合方式或特征子集。Balloon与Known Target安全方向必须作为正式配对报告的守卫。

## 冻结验收

- 主单位：development validation parent-final KnownGate。
- 主要正向信号：Treatment Macro-F1相对Baseline严格提升`>0.02`。
- 主假设方向：Clutter与Drone合计净救回必须为正，且两类各自净救回不得为负。
- 安全要求：Bird、Balloon、总体Known Target Recall和Clutter Recall不得下降。
- 必报：两类P/R/F1、四原始标签净救回、混淆矩阵、20个same-epoch对照、5000次按parent和原始标签分层的paired bootstrap。
- Clutter validation仅9个parent，必须同时报告support与95%区间。

通过仅允许进入多seed或train parent-grouped OOF确认，并需要新的独立holdout/数据版本。Unknown不是训练类别，KnownGate argmax不是最终产品动作。test、OOF、完整级联、正式拒识、stable、P95和部署均保持“未执行”。
