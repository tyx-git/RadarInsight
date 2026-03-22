# RadarInsight 模型优化路线状态

## 1. 当前总目标

当前阶段仍然坚持一个核心目标：**高精度优先**。

主方案不变：

```text
轨迹序列 -> Residual TCN -> Transformer Encoder ----------------------┐
                                                                    ├-> 双向 Cross-Attention Fusion -> 分类头
RD Clip 序列 -> 官方 X3D 风格 X3D-M ----------------------------------┘
```

即：

**`X3D-M + Residual TCN + Transformer + 双向 Cross-Attention`**

## 2. 当前总体判断

当前代码已经完成了真实数据链路、轨迹骨干、融合层以及多阶段 RD 强建模的主要实现。

但需要明确：

- 轨迹分支已经比较完整
- RD 微动链路仍然是主要优化重心
- 多分辨率 `DPL -> RD` 已完成实现，但当前不作为默认主路径

## 3. 已完成

### 3.1 真实数据主链路

- [x] `pro/raw -> pro/sessions -> data/sessions`
- [x] `TR/RD` 自动配对
- [x] 从 `TR` 提取时序轨迹数据
- [x] 从文件名提取弱监督标签
- [x] 生成标准 session 目录

### 3.2 模型主骨架

- [x] `X3D-M` 风格 RD 主骨干
- [x] `Residual TCN + Transformer` 轨迹骨干
- [x] 双向 `Cross-Attention`
- [x] `Track-only / RD-only` 辅助头

### 3.3 RD 强建模

- [x] 谐波敏感增强层
- [x] 谱轴精炼层
- [x] `background/local contrast` 并行增强支路
- [x] 复数感知 RD 表示
- [x] `real / imag / log amplitude / temporal difference / background suppressed / local contrast`

### 3.4 训练与评估策略

- [x] session 级 train/val/test 切分
- [x] 按类别分层的 session 切分
- [x] class-balanced loss
- [x] label smoothing
- [x] warmup + cosine scheduler
- [x] gradient clipping
- [x] macro-F1 监控
- [x] 独立评估与混淆矩阵导出

## 4. 进行中

### 4.1 RD 微动链路持续优化

- [ ] 继续验证 `phase difference / coherence` 的长期收益
- [ ] 继续验证并行增强支路对真实全量训练的稳定贡献
- [ ] 持续观察 `Bird / FixedWing / RotaryDrone` 的混淆矩阵变化

### 4.2 训练稳定性验证

- [ ] 不同 epoch 数下结果是否持续稳定
- [ ] 不同 session 划分下结果是否稳定
- [ ] `macro_f1` 是否持续优于旧版本

## 5. 待完成

### 5.1 多分辨率 `DPL -> RD`

这一项**已经实现**，但当前实验结果不优于单尺度主路径，因此现在明确保留为：

**`toTest` 分支，不作为默认训练路径。**

当前状态：

- [x] 已实现多分辨率 `DPL -> RD`
- [x] 已支持 `[16, 32, 64]` 多窗口
- [x] 已能进入训练链路
- [ ] 不作为默认主路径启用

后续还需要继续研究：

- [ ] 是否改成多尺度侧分支，而不是直接多通道主输入
- [ ] 是否存在更合适的窗口组合
- [ ] 是否能在更长训练周期内优于单尺度主路径

### 5.2 RD 专属辅助任务

- [ ] 对比学习式约束
- [ ] prototype / center loss
- [ ] 同类跨 session 聚拢、异类分离

### 5.3 更高质量标签体系

当前标签是 **session 级弱监督标签**。

后续真正决定上限的重要因素仍然是：

- [ ] clip 级标签
- [ ] 更细粒度时间局部标签
- [ ] 人工复核后的高质量标注集

### 5.4 RD 物理语义继续核验

- [ ] FFT 方向与参数是否最优
- [ ] `512/1024` 不同长度混用时的处理是否最合理
- [ ] 是否存在更适合当前设备的数据重排方式

## 6. 当前一句话总结

**当前路线已经进入“高精度实现中期版本”，而多分辨率 `DPL -> RD` 目前明确保留为 `toTest` 分支，不作为默认训练主路径。**
