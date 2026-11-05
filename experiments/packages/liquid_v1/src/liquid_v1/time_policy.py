"""第一版液态算法的 dt_reason 时间语义常量（与 radar_ln3 数据契约一致）。

来源：src-v3 @ fd34e46610165441d86469802e9ff461d49bd2aa，src-v3/radar_ln3/data.py 第 42-45 行。

zero_delta（同秒事件）在 CfCLiquidCausalEncoder 中按 invalid_dt_update_mode
选择五种策略之一：floor / bounded_observation / skip / strict_skip / density_weighted；
padding 不调用 cell。V1 未提供独立 contract 序列化函数。
"""

DT_REASON_PADDING = 0
DT_REASON_FIRST_EVENT = 1
DT_REASON_ZERO_DELTA = 2
DT_REASON_POSITIVE_DELTA = 3

__all__ = [
    "DT_REASON_PADDING",
    "DT_REASON_FIRST_EVENT",
    "DT_REASON_ZERO_DELTA",
    "DT_REASON_POSITIVE_DELTA",
]
