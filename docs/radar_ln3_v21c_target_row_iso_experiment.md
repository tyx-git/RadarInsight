# RADAR-LN3 V21c目标行能量占比单因素实验

状态：训练前冻结。V21c只回答当前事件canonical RD目标行能量占比，是否改善KnownGate中的`ClutterTarget/DroneTarget`边界。Bird与Balloon只作为安全守卫，不是预期受益类别。

## 单因素变量

Treatment相对V21 seed43 Baseline只新增一维当前事件证据：

1. `target_row_linear_energy_fraction_mean`

每个真实canonical RD帧先将dB幅度按该帧最大值转换为相对线性功率，取零距离偏移目标行的全部384个Doppler bin功率和，除以完整`21×384`功率和，再在当前事件的真实RD帧间取均值。值域固定为`[0,1]`。目标行由canonical零偏移坐标确定，不按标签、全局能量峰、validation结果或parent最终状态选择。

事件没有canonical RD时该值为0并标记invalid；valid只用于内部门控，不拼入分类输入。TR radial/range缺失不影响iso。V21冻结的train/validation六维缓存直接按SHA复用，模型只选择iso一列，不重新读取raw数据或重建RD。

## 配对协议

- Control：V21 development champion seed43。
- Treatment：相同seed、DataLoader顺序、20 epoch、class weights、loss、prefix辅助、checkpoint选择和模型主干。
- 初始化：复制seed43 Baseline的全部共享随机初态，只将新增一列Fusion权重精确置零；不从训练完成的Baseline checkpoint微调。
- 全参数训练：主干与新增列均参与全部20 epoch训练。
- 旧seed42 V21c配置只保留追溯，禁止作为正式可比Treatment运行。

## 历史探索与正式口径

早期`tmp/v3_probe_archive/tmp_feature_mining.py`中的`Drone≈0.125、Clutter≈0.053`不能作为正式V21c数值。该脚本抽取的93个track跨train/val/test=`60/13/20`，使用原生`31×PRT`绝对幅度和、第一次能量峰选行后跨事件固定、每parent最多10个假定三帧事件。V21c则使用canonical `21×384`相对线性功率和每事件零偏移行；两者物理量、采样范围和聚合单位均不同。因此原V16 test已参与特征构思，不能再作为该特征概念的纯盲最终holdout。

正式train-only parent审计显示：Drone/Clutter的parent平均iso均值约为`0.2814/0.1307`，方向无关AUC约`0.7771`，支持“Drone目标行能量更集中”的预注册方向；Clutter/Bird AUC约`0.5183`，几乎没有单变量分离。该统计只冻结方向和风险，不是readiness性能门槛，也不允许据此调整公式、符号、阈值或validation选择。

## 冻结验收

- 主单位：development validation parent-final KnownGate。
- 主要正向信号：Treatment Macro-F1相对Baseline严格提升`>0.02`。
- 主假设方向：Clutter与Drone合计净救回必须为正，且两类各自净救回不得为负。
- 安全要求：Bird、Balloon、总体Known Target Recall和Clutter Recall不得下降。
- 必报：两类P/R/F1、四原始标签净救回、混淆矩阵、20个same-epoch对照、5000次按parent和原始标签分层的paired bootstrap。
- Clutter validation仅9个parent，必须同时报告support与95%区间。

如果硬指标不跨过冻结门槛，只允许把一致方向的概率变化记录为诊断，不得因“辅助特征”定位而降低2个百分点门槛。通过也只允许进入多seed或train parent-grouped OOF确认，并需要新的独立holdout/数据版本。Unknown不是训练类别，KnownGate argmax不是最终产品动作。test、OOF、完整级联、正式拒识、stable、P95和部署均保持“未执行”。
