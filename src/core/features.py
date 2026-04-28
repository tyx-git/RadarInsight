from __future__ import annotations

"""
轨迹特征列，包含时间秒数、方位角、距离、径向速度、仰角、高度、幅度、信噪比、航向角、速度等轨迹特征。
"""
TRACK_FEATURE_COLUMNS = [
    "time_seconds", # 时间秒数
    "azimuth_deg", # 方位角
    "range_m", # 距离
    "radial_speed_mps", # 径向速度
    "elevation_deg", # 仰角
    "height_m", # 高度
    "amplitude_db", # 幅度（dB）
    "snr_db", # 信噪比（dB）
    "course_deg", # 航向角
    "speed_mps", # 速度（m/s）
]

"""
轨迹元数据列，包含数据源名称、记录ID、对应ID、时间戳、帧索引、批次ID、原始标签名称、标签名称、标签ID、备注文本等信息。
"""
TRACK_META_COLUMNS = [
    "source_name", # 数据源名称
    "record_id", # 记录ID
    "pair_id", # 对应ID
    "timestamp", # 时间戳
    "frame_index", # 帧索引
    "batch_id", # 批次ID
    "label_name_raw", # 原始标签名称
    "label_name", # 标签名称
    "label_id", # 标签ID
    "note_text", # 备注文本
]
