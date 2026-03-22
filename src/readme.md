# Source Code

包含雷达数据采集与处理的核心逻辑。

## 模块介绍
- **ingestion/**: 雷达协议解析、Track 与 RD 实时/离线接收与落盘。
- **preprocessing/**: 数据切片（Track/RD Clip）、时间对齐、空间投影。
- **labeling/**: 标注逻辑、真值自动匹配算法、标注文件生成。
- **qc/**: 自动质检脚本（数据完整性、分布统计、同步校验）。
