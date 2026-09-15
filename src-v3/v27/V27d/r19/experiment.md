[E]完成V27d-r19 missing-SNR通道消融实验
为了求证显式missing-SNR通道在Excel TR-only DBB分类中是否提供额外信息，固定r18的source-by-label划分和radial speed输入，比较保留SNR值加缺失标志与移除SNR通道的两臂训练。
实验结论：保留missing-SNR通道的验证Accuracy 66.67%、Macro-F1 45.27%，移除后为65.48%和43.90%；有小幅候选信号，但单seed且无跨源复现，暂不晋级。
