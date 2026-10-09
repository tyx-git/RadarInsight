# 产品编排核心

本目录负责最终产品的运行编排，不承载具体的雷达领域算法。

`cli.py` 负责数据处理命令；`training` 和 `evaluation` 分别负责训练与评估；
`inference` 提供 PyTorch、ONNX 和因果在线推理；`server` 提供当前自定义
R-RTSP 的后端回放服务。

完整链路为：

```text
.asd -> TR numeric/time -> canonical RD [21,384] -> RRTSP PT=97/PT=96
     -> 服务端 L1/L2/L3 Full Cascade -> PT=98 分类/置信度 -> AeroSight
```

PT=97 只传输雷达观测类别，PT=98 单独传输可选的模型结果。模型调用保持在
`core.server.model` 和 `core.inference`，可以替换 ONNX 模型而不改传输会话。
