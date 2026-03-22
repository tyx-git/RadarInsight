# XW/SR215-3200C 雷达对接接口文档 (Radar Interface Guide)

本文档旨在说明如何将 XW/SR215-3200C 雷达接入到本实验环境。

## 1. 物理连接与网络配置
- **雷达接口**: RJ45 百兆/千兆以太网口。
- **采集主机**: 建议配置固定 IP（例如 `192.168.0.101`），并确保与雷达（例如 `192.168.0.100`）在同一网段。
- **协议类型**: 通常为 UDP 广播或单播（请查阅具体产品说明书确认端口号）。

## 2. 数据流说明
本环境需要两类核心数据流：

### 2.1 轨迹级数据 (Track Data)
- **频率**: ≥ 2 Hz。
- **必选字段**:
  - `t` (Unix 时间戳或自启动时间)
  - `frame_id` (帧编号)
  - `track_id` (航迹 ID)
  - `range_m` (距离，单位：米)
  - `azimuth_deg` (方位角，单位：度)
  - `radial_speed_mps` (径向速度，单位：米/秒)
- **可选字段**:
  - `snr_db`, `amplitude`, `rcs` 等。

### 2.2 距离-多普勒数据 (RD Data)
- **形态**: 2D 矩阵（通常为复数或幅度值）。
- **坐标轴参数**:
  - 距离分辨率 (Range Resolution)
  - 速度分辨率 (Velocity Resolution)
  - 零多普勒位置 (Zero Doppler Index)

## 3. 对接步骤 (TODO)
开发者需在 `src/ingestion/radar_receiver.py` 中实现具体的解析逻辑。

### 实现建议
1. **UDP 监听**: 使用 Python `socket` 库监听雷达推送的原始二进制包。
2. **协议解析**: 根据雷达二进制协议文档，使用 `struct` 或 `construct` 库将字节流解析为 Python 字典或对象。
3. **数据对齐**: 将接收到的雷达时间转换为系统 UTC 时间。
4. **异步落盘**: 建议使用 `queue` 异步写入 CSV (Track) 或 HDF5 (RD)，避免阻塞主接收线程。

## 4. 示例代码入口
参见 [radar_receiver.py](src/ingestion/radar_receiver.py)。
