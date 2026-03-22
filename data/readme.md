# Data Storage

存放所有采集到的原始雷达数据、同步真值、标注文件及质检报告。

## 目录结构
- **sessions/**: 每次采集任务的独立会话目录，格式为 `YYYY-MM-DD_site_sessionID`。
  - **radar/**: `track` (CSV), `rd` (HDF5), `pcap` (Raw UDP packets).
  - **truth/**: `video`, `gnss`, `flightlog`.
  - **labels/**: 轨迹片段（Track Clip）和 RD 片段（RD Clip）的 JSONL 标注文件。
  - **qc/**: 采集后自动生成的质检图表与报告。
