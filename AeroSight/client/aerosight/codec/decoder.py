# -*- coding: utf-8 -*-
"""雷达球坐标到客户端本地直角坐标的转换。"""
from __future__ import annotations

import numpy as np

# 角度约定：
# azimuth_deg / course_deg 均使用有符号角度；0° 为前方，顺时针（向右）为正。
# 客户端坐标：x=前方，y=左方，z=天顶。
COORDINATE_SYSTEM = {
    "angle_unit": "degree_signed",
    "azimuth_zero": "radar_forward",
    "azimuth_positive": "clockwise_right",
    "axes": {"x": "forward", "y": "left", "z": "up"},
}


def spherical_to_xyz(range_m, azimuth_deg, elevation_deg):
    """雷达球坐标 -> xyz。

    range_m: 斜距；azimuth_deg: 有符号方位角；elevation_deg: 俯仰角。
    正方位角应出现在雷达右侧，负方位角出现在左侧。
    """
    r = np.asarray(range_m, dtype=np.float64)
    az = np.deg2rad(np.asarray(azimuth_deg, dtype=np.float64))
    el = np.deg2rad(np.asarray(elevation_deg, dtype=np.float64))

    cos_el = np.cos(el)
    x = r * cos_el * np.cos(az)
    # 外部角度顺时针为正；客户端 y 轴向左，因此这里取负号。
    y = -r * cos_el * np.sin(az)
    z = r * np.sin(el)
    return np.stack([x, y, z], axis=-1).astype(np.float32)
