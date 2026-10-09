# -*- coding: utf-8 -*-
"""内置仿真航迹：二维预览使用平面距离，数据字段同时保留斜距。"""
from __future__ import annotations
import math
from aerosight.core.radar import RadarObservation


def _target(t, track_id, ground_radius, altitude):
    angular = 18.0 / ground_radius
    a = angular * t
    x = ground_radius * math.cos(a)
    y = ground_radius * math.sin(a)
    slant_range = math.hypot(ground_radius, altitude)
    elevation = math.degrees(math.atan2(altitude, ground_radius))
    # 外部方位角顺时针为正；客户端 y 轴向左，因此这里使用 -atan2(y, x)。
    azimuth = math.degrees(math.atan2(-y, x))
    speed = ground_radius * angular
    vx = -ground_radius * math.sin(a) * angular
    vy = ground_radius * math.cos(a) * angular
    course = math.degrees(math.atan2(-vy, vx))
    return RadarObservation(
        stream_id="simulation", track_id=track_id, state="confirmed",
        range_m=slant_range, azimuth_deg=azimuth, elevation_deg=elevation,
        ground_range_m=ground_radius, altitude_m=altitude,
        radial_speed_mps=0.0, speed_mps=speed, course_deg=course,
        snr_db=.82, amplitude_db=.75, x_m=x, y_m=y, z_m=altitude)


def build_simulation_observations(t):
    return [_target(t, 1, 1200, 120), _target(t, 2, 700, 70)]
