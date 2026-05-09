# 阶段性目标：剩余优化任务

更新时间：2026-05-09

## 已从本文件移除的已落地内容

以下内容已经进入代码，不再作为待办项保留：

- 两层标签接口：`coarse_label` / `fine_label`。
- CQ-08 旋翼无人机细类映射与旧索引兼容。
- Expert1 正式模式：`dual_main` / `deep_reroute`。
- Expert1 第一轮 Unknown 作为 reject signal 的基础逻辑。
- System 推理期有方向 reroute graph。
- 测试报告中的 without / with reroute 基础对比。
- Expert1 routed / true-low-altitude 双口径报告。
- 回退路径、reject reason、deep 使用率、逐类别回退前后 recall 报告。
- 回退 helped / harmed 逐样本审计。
- Expert1 low reject score 组成项进入 prediction record。
- source-balanced macro F1、min source macro F1、source x label precision / recall / F1。
- System Aircraft / Expert2 仲裁诊断：Expert2 positive but final not Aircraft、final Aircraft but Expert2 not positive、Aircraft FP by source。
- 回退阈值从 `config` 进入 JSON / Markdown 报告。
- `calibrated_candidates` 作为 eval-only candidate probe 接入融合模式对比。
- 专家内部性能报告：
  - Expert1 trajectory / RD 本地指标。
  - Expert2 内部二分类混淆矩阵。
  - Expert3 高空规则命中正确率。
  - Expert4 地面内部混淆矩阵与 People -> Car 统计。
- RD 分支 CosFace 原型余弦相似度矩阵。
- 安全关键误判对回退前后变化。
- 训练损失组件曲线进入 `training_history.json`。

## 当前主线

继续围绕一个目标推进：

```text
提高系统 macro F1 与跨源泛化，同时保证 Aircraft recall / precision 不掉，
并让 Expert1 RD 主干在低空混淆对上产生稳定净收益。
```

短期目标仍是先稳定拉到：

```text
system macro F1 >= 0.75
Aircraft recall >= 0.80
Aircraft precision >= 0.80
Ground group F1 >= 0.60
Expert1 RD net gain >= 0
```

## 一、Expert1 deep_reroute 仍需增强

当前 `deep_reroute` 已有独立模式和更强 gate 策略，但仍主要复用同一套 RD 表征。下一步必须让它真正比 `dual_main` 更强。

### 待实现

- 更高分辨率 RD patch 路径。
- 更多 RD 帧采样策略。
- deep 专用局部 pair correction head。
- deep 专用 Unknown / Clutter residual head。
- deep 专用报告：
  - pass1 prediction
  - deep prediction
  - final prediction
  - deep helped / harmed
  - deep accept reason

### 验收

```text
deep_reroute 在触发样本上的 recovery_rate > harm_rate
Unknown / Clutter recall 不低于 dual_main
Drone / Bird 混淆减少
```

## 二、Expert1 RD 主干净收益

RD 已作为低空第二主证据参与，但仍需从“能参与”推进到“稳定贡献”。

### 待实现

- RD-only hard pair 分析固定进入报告：
  - DroneTarget <-> BirdTarget
  - UnknownTarget <-> ClutterTarget
  - BirdTarget <-> ClutterTarget
- RD harm 优先约束：
  - 优先压 `rd_harm_rate_on_rd`
  - 保持 `rd_rescue_count`
  - 不允许 final macro F1 因 RD 融合明显低于 trajectory
- Unknown 使用 residual / reject-style 建模继续强化。
- Clutter 使用局部纹理、能量分散、稀疏度、粗糙度特征继续强化。

### 验收

```text
rd_harm_rate_on_rd 持续下降
rd_net_gain_rate_on_rd >= 0
rd_unknown_clutter_recall_mean 提升
final_macro_f1 >= traj_macro_f1 - 可配置容忍值
```

## 三、跨专家回退继续收敛

基础 reroute graph 已落地，但仍需从“可运行”推进到“可调优、可审计、可部署”。

### 待实现

- Expert2 / Expert4 reject reason 细分，不只记录通用 reject。

### 验收

```text
with_reroute macro F1 >= without_reroute macro F1
reroute_harm_count 不明显上升
fallback_to_unknown 可解释
Aircraft recall / precision 不因 reroute 下滑
```

## 四、Ground / Expert4 仲裁继续优化

Ground 当前风险仍是被低空或 Aircraft 抢分，不能只堆 Expert4 head。

### 待实现

- System representation + Expert4 head 对照测试保留为常规测试。
- Ground candidate protection 只允许高置信触发：
  - groundness 高
  - ground logits margin 高
  - Router 没有强烈反对
- Ground diagnostics：
  - recovered_count
  - false_positive_introduced_count
  - ground_top1_margin
  - ground_candidate_changed_prediction

### 验收

```text
ground_group_f1 >= 0.60
Ship recall >= 0.60
Car recall >= 0.70
pred_ground_on_non_ground_count 不明显增加
```

## 五、Expert2 Open-Set 继续校准

Expert2 已朝 open-set / aircraft prototype 方向推进，但阈值和系统仲裁仍需继续验证。

### 待实现

- Expert2 独立阈值校准继续收紧：
  - aircraft_precision
  - aircraft_recall
  - false_aircraft_rate
  - raw_tr_rd false aircraft rate

### 验收

```text
Expert2 independent Aircraft precision >= 0.50
Expert2 independent Aircraft recall >= 0.60
System Aircraft precision >= 0.80
System Aircraft recall >= 0.80
```

## 六、跨源泛化与选模

当前不能只看整体 accuracy。弱源和 RD 源必须有硬约束。

### 待实现

- raw_tr_rd 作为重点弱源验证指标。
- source-aware sampling / weak-source replay 增强。

### 验收

```text
min_source_accuracy 不低于当前稳定基线
raw_tr_rd accuracy 不被 cq08 高分掩盖
source_balanced_macro_f1 随整体 macro F1 同步提升
```

## 七、性能测试清单剩余项

完整性能清单中仍需继续补齐：

- 推理延迟：
  - end-to-end
  - trajectory branch
  - RD branch
  - deep_reroute path
  - Expert2 / Expert3 / Expert4
  - RD 预处理
- Expert4 People -> background / 非 Ground 的专项安全误判拆分。

## 八、工程约束

- 不写冗余代码和冗余函数。
- 不新增没有训练意义的标签字段。
- 不复制训练器逻辑。
- 不让细类体系破坏当前大类输出。
- 不把 RD 退回弱辅助分支。
- 不在没有可靠标注时引入新细类。
