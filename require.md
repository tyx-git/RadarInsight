# RadarInsight 外部支持与依赖说明

本文档说明当前项目落地、调试和训练所需的外部支持、软件依赖与硬件条件。当前深度学习框架以 `CPU` 为默认优化目标，优先保证在普通开发机上的可运行性与稳定性，同时保留 `GPU` 接口，便于后续更换机器后直接切换。

## 1. 当前建议结论

- 当前训练默认使用 `CPU`，适合先完成数据链路打通、模型联调和小规模实验。
- 代码已保留 `GPU` 训练入口，无需改模型结构；未来只需安装对应 CUDA 版本的 `PyTorch` 并在命令行指定设备即可。
- 推荐先稳定数据质量、标签质量和特征链路，再升级到 GPU 做更大规模训练。

## 2. 必需的软件依赖

项目基础依赖定义在 `requirements.txt`，建议使用独立虚拟环境。

### 基础运行环境
- `Python 3.9+`，推荐 `Python 3.10` 或 `Python 3.11`
- `pip`
- Windows 10/11 或 Ubuntu 20.04+

### 核心 Python 依赖
- 数值与数据处理：`numpy`、`pandas`、`scipy`
- HDF5 读写：`h5py`
- 配置与元数据：`pyyaml`、`pydantic`
- 可视化与辅助工具：`matplotlib`、`tqdm`
- 测试：`pytest`
- 深度学习：`torch`

### 可选依赖
- `opencv-python`：图像/视频辅助处理
- `fastapi`、`uvicorn`：后续服务化接口
- `scapy`：原始网络报文解析

安装命令示例：

```bash
.\.venv\Scripts\pip.exe install -r requirements.txt
```

## 3. 深度学习框架要求

### 当前 CPU 优先策略
- 当前训练代码默认设备为 `cpu`
- 目标是优先优化开发期可移植性、依赖简洁度和调试效率
- 适合当前的轨迹分支、RD 分支联调与小中规模验证

### 保留 GPU 切换接口
训练命令已经保留 `--device` 参数：

```bash
.\.venv\Scripts\python.exe manage.py train final_debug --device cpu
```

未来切换到支持 CUDA 的机器后，可直接使用：

```bash
.\.venv\Scripts\python.exe manage.py train final_debug --device cuda
```

注意：切换到 GPU 时，需要安装与目标机器 CUDA 版本匹配的 `PyTorch` 构建版本。模型代码、数据集代码和训练入口本身无需重写。

## 4. 硬件与系统支持

### 开发与训练主机
- CPU：建议 8 核及以上
- 内存：建议 16GB 起，推荐 32GB
- 存储：建议 NVMe SSD，持续写入速度不低于 `300MB/s`
- 网卡：至少 1 个千兆网口；若 RD 数据吞吐量较大，建议支持 Jumbo Frame

### 可选 GPU 条件
- NVIDIA GPU
- 正确安装驱动、CUDA Runtime 与匹配版本的 `PyTorch`
- 仅在需要更大 batch、更快训练或更复杂模型时启用

## 5. 雷达侧外部支持

若要从真实设备接入，而不是仅使用预处理数据，还需要以下支持：

- 雷达 UDP 二进制协议文档，明确报文头、轨迹字段、RD 矩阵字段定义
- 厂商 SDK 或动态库（如有）
- 雷达参数配置接口或控制指令集
- 采集主机与雷达之间稳定的局域网环境

这些内容决定 `src/ingestion/` 是否能够从真实设备稳定解析并落盘。

## 6. 辅助设备与现场条件

- NTP 时间同步能力，用于雷达、主机、相机或 GNSS 的时钟对齐
- 相机或视频设备，用于可视化核验与人工复核
- GNSS/RTK 设备，用于生成更高质量真值标签

## 7. 使用顺序

1. 安装 Python 依赖并验证 `mock` / `process` 流程
2. 在 `CPU` 上跑通 `train` 命令，确认数据、标签、模型和输出文件完整
3. 确认更大规模训练需求后，再迁移到 `GPU` 机器并切换 `--device cuda`