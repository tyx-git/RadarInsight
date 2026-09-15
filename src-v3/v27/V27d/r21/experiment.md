[E]完成V27d-r21 radial adapter冻结主干实验
为了求证在冻结r20完整主干、只训练radial adapter的条件下，结构化radial sidecar能否提供独立L2收益，使用Excel TR-only数据、r20 zero_control初始化和固定adapter-only训练范围，比较zero_control与radial_decomposed两臂。
实验结论：两个验证臂Accuracy均79.76%、Macro-F1均52.96%，paired transition没有发生任何有效纠正；未产生离散分类收益，不支持当前默认接入，也不能推出开放集或跨源泛化结论。
