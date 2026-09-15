完成V27d-r16 RD帧真实局部时间连续递推实验
为了求证RD事件内部帧间时间顺序是否被masked mean丢失，使用每个TR事件独立重置的Liquid V2 frame cell，显式输入帧间Δt，并处理padding、单帧、缺失和非单调边界。
实验结论：工程链路通过，但两seed平均Accuracy下降约0.4484pp、Macro-F1下降约0.4387pp；不进入默认L2、knownness或L3。
