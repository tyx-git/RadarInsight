# RadarInsight 服务后端

本目录是 RadarInsight 的**服务后端**独立源码边界，与 `src-v3` 和 `experiments` 相互隔离。这里只保留雷达数据处理、开放集推理和 R-RTSP 服务逻辑，不包含前端界面。

## 后端职责

- 读取 `.asd` 数据包并按 R-RTSP 协议循环回放、推流；
- 运行 L1 KnownGate → L2 DBB specialist → L3 semantic decision 开放集推理；
- 提供 transport-neutral 推理入口，输出 PT=96/97 观测与可选 PT=98 模型结果；
- 负责 L3 温度缩放、类别条件候选集、knownness、路由与语义动作解析；
- 负责模型加载、校准、评测、导出以及运行配置；
- 向兼容的 R-RTSP 客户端提供数据流和远端标签，不提供 UI。

AeroSight 客户端前端已从本仓库移除，不再维护。ASD 回放数据位于 `data/asd`，前端采集样例位于 `data/asd_samples`；R-RTSP 协议规范位于 [`docs/R-RTSP协议规范-v1.md`](docs/R-RTSP协议规范-v1.md)，独立推流工具位于 [`tools/rrtsp_tool/`](tools/rrtsp_tool/)。

服务后端统一采用 `radar_ln3` 核心包和 Liquid V2 主干。`core/cli.py` 规划为唯一后端运行入口；`core/` 负责运行编排，`radar_ln3/` 负责雷达数据、特征、模型、开放集和决策逻辑。配置、测试、报告和脚本分别放在对应目录中。

本目录不依赖历史实验包，也不把 Unknown 作为普通分类类别。当前严格 open-set 架构为 L1 KnownGate、L2 DBB specialist、L3 semantic decision layer。

`radar_ln3.decision` 提供 L3 温度缩放、类别条件候选集、独立 knownness head、路由和语义动作解析。当前 radar_ln3 不包含质量门；质量事实只作为 knownness 输入和诊断记录，不参与语义动作判定。transport-neutral 的 L1 → L2 → L3 服务入口位于 `core/inference`；当前 RRTSP 后端位于 `core/server`，PT=96/97 观测与可选 PT=98 模型结果分离。

服务启动示例（扫描 `data/asd`，每个 ASD 包完整推理一次并发送 PT=98）：

```text
python -m core.server --asd-dir data/asd --host 0.0.0.0 --port 8554 --inference-mode package
```

当前 L3 统一配置位于 `config/radar_ln3_l3.json`。正式 L1/L2 endpoint 直接绑定到 `artifacts/radar_ln3/l1/l1.pt` 和 `artifacts/radar_ln3/l2/l2.pt`；L3 head 与 calibration 需要在 endpoint 变化后重新拟合。当前状态仍为 formal package，不代表已获得部署认证。

当前 L3 head 使用冻结 endpoint 的 causal temporal features、L1 target/clutter 双路、L2 共享 DBB knownness 以及 44 维 RD/profile residual。新 head 的表示契约为 `RADAR-LN3-KNOWNNESS-HEAD-V2`，旧 head 不进入 active 产品加载路径。

## 开源协议

本项目基于 [MIT License](LICENSE) 开源。
