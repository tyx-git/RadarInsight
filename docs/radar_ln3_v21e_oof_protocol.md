# RADAR-LN3 V21e KnownGate五折OOF协议

状态：训练前冻结。该阶段只验证V21e因果航向圆周不一致性在known train parent上的外折泛化和相对V21 Baseline增益，不拟合阈值、不训练Unknown、不读取validation/test payload。

## 1. OOF含义

OOF（Out-of-Fold）把train分成五个互斥折。每次使用其中四折训练，只对从未进入该折优化器、归一化和checkpoint选择的第五折生成预测。五轮结束后，每个train parent恰好获得一次“模型未见过它”的预测。

它的作用是：

- 在不使用validation/test调参的前提下，估计固定训练协议的外样本表现；
- 为后续L3阈值校准提供没有同parent训练泄漏的分数；
- 用严格同折Control/Treatment判断V21e收益是否仍存在；
- 暴露只在完整train拟合或development错例上成立的特征。

OOF不是独立新采集，也不能替代test、新holdout或部署验证。

## 2. 分折与数据隔离

- 总人口固定为V16 train中的986个known parent：Drone 476、Bird 239、Balloon 238、Clutter 33；Unknown 69个完全排除。
- 分组单位为`provenance_component_id`，预测单位为parent。975个capture/content组件必须整组进入同一折，避免同capture或同内容近重复跨折。
- 冻结五折support：
  - fold 00：Drone/Bird/Balloon/Clutter=`95/48/48/6`；
  - fold 01：`95/48/48/6`；
  - fold 02：`95/48/48/7`；
  - fold 03：`96/47/47/7`；
  - fold 04：`95/48/47/7`。
- 每折RD amplitude normalization及二分类class weight仅从fold-train计算。V21e cache可按parent子集复用，因为其值逐parent因果生成、无标签且没有全train拟合统计。
- validation和test不参与分折、归一化、训练、选epoch、阈值或报告。

## 3. 配对训练

- 每折Control为V21 Baseline，Treatment为Baseline加一列V21e特征。
- 两臂使用seed43相同随机共享初态、相同fold-train parent顺序、batch、损失、fold class weight和20 epoch预算；Treatment新增Fusion列精确置零。
- 两臂均从头训练全部参数，禁止加载seed43正式best checkpoint微调。
- 外折不用于checkpoint选择；固定epoch 20后只前向一次。折模型仅为OOF支持模型，不是部署checkpoint。
- seed43正式best epoch 14继续封存为当前最佳development参考，不参与任何外折初始化。

## 4. 分阶段节省计算

流水线先完成五个Treatment折并汇总绝对护栏。若Treatment pooled OOF每类Precision/Recall/F1最低值小于0.90，则按预注册futility规则停止，不训练五个Control折，也不拟合阈值。

Treatment绝对护栏通过后才训练Control五折，并进行paired OOF比较。

## 5. 冻结决策

parent-final确认必须同时满足：

1. Treatment每类Precision/Recall/F1最低值不低于0.90；
2. Treatment−Control parent Macro-F1严格大于0.02；
3. DBB Recall和Clutter Recall均不下降；
4. Drone/Bird/Balloon/Clutter四原始标签net rescue均不为负；
5. 按capture/content组件分层的5000次paired bootstrap中，Macro-F1差值95%下界大于0。

eligible-prefix另设独立守卫：Macro-F1、DBB Recall和Clutter Recall均不得下降。该守卫失败不改写parent-final结论，但继续阻止在线KnownGate/stable采纳。

只有parent-final OOF确认通过，下一阶段才允许使用Treatment OOF分数拟合KnownGate/L3双接受边界。Stage 078本身的threshold、Unknown open-set audit、test、正式拒识、stable、P95和部署均为“未执行”。

## 6. 执行入口

唯一流水线：`src-v3/scripts/v21e_known_gate_oof_pipeline.ps1`。

`-Stage all`依次执行prepare、Treatment五折、futility gate、Control五折和paired report；支持按fold执行及从各折`last_model.pt`的epoch边界恢复。任一代码/合同错误立即停止，性能futility失败则保留Treatment结果并有意跳过Control。
