# 产品编排核心

本目录负责最终产品的运行编排，不承载具体的雷达领域算法。

cli.py是规划中的唯一产品入口;
runtime处理配置、路径、哈希和 checkpoint;
logger负责产品日志;
training和evaluation分别负责训练与评估流程;
