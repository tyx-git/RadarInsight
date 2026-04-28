from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


PHYSICAL_FEATURE_COLUMNS = [
    "height_mean", # 高度均值
    "height_std", # 高度标准差
    "height_slope", # 高度斜率
    "ground_speed_mean", # 地面速度均值
    "ground_speed_std", # 地面速度标准差
    "ground_speed_max", # 地面速度最大值
    "course_change_rate_mean", # 方位角变化率均值
    "trajectory_curvature_mean", # 轨迹曲率均值
    "acceleration_std", # 加速度标准差
    "hover_ratio", # 悬停比
    "radial_speed_std", # 径向速度标准差
    "radial_speed_flatness", # 径向速度谱平滑度
    "amplitude_mean", # 幅度均值
    "amplitude_std", # 幅度标准差
    "snr_mean", # 信噪比均值
    "height_missing_flag", # 高度缺失标记
    "phase_missing_flag", # 相位缺失标记
]


@dataclass(frozen=True) # 只读
class PhysicalFeatureVector:
    """
    物理特征向量，包含高度、地面速度、位角变化率、轨迹曲率、加速度、悬停比、径向速度谱平滑度、幅度、信噪比等物理特征。
    """

    values: np.ndarray # numpy 数组，包含物理特征值

    def as_dict(self) -> dict[str, float]: # 转换为字典，用于输出报告
        return {
            name: float(self.values[idx])
            for idx, name in enumerate(PHYSICAL_FEATURE_COLUMNS)
        }


def _safe_numeric(series: pd.Series) -> np.ndarray:
    """
    安全转换为数值类型，将 NaN 转换为 0.0。
    """
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float32, copy=True)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return values


def _linear_slope(values: np.ndarray) -> float:
    """
    计算线性拟合斜率，最小二乘法一元线性回归斜率。
    """
    if values.size < 2:
        return 0.0
    x = np.arange(values.size, dtype=np.float32)
    x_mean = float(x.mean())
    y_mean = float(values.mean())
    denom = float(((x - x_mean) ** 2).sum())
    if denom <= 0.0:
        return 0.0
    numer = float(((x - x_mean) * (values - y_mean)).sum())
    return numer / denom


def _spectral_flatness(values: np.ndarray) -> float:
    """
    计算谱平滑度。
    """
    if values.size == 0:
        return 0.0
    magnitude = np.abs(values.astype(np.float64)) + 1e-6
    geo = float(np.exp(np.log(magnitude).mean()))
    ari = float(magnitude.mean())
    if ari <= 0.0:
        return 0.0
    return geo / ari


def extract_physical_features(
    frame: pd.DataFrame,
    *,
    height_missing: bool = False,
    phase_missing: bool = False,
) -> PhysicalFeatureVector:
    """
    从数据框中提取物理特征向量。
    """
    if frame.empty:
        return PhysicalFeatureVector(values=np.zeros(len(PHYSICAL_FEATURE_COLUMNS), dtype=np.float32))

    height = _safe_numeric(frame["height_m"])
    speed = _safe_numeric(frame["speed_mps"])
    course = _safe_numeric(frame["course_deg"])
    radial_speed = _safe_numeric(frame["radial_speed_mps"])
    amplitude = _safe_numeric(frame["amplitude_db"])
    snr = _safe_numeric(frame["snr_db"])
    azimuth = _safe_numeric(frame["azimuth_deg"])
    track_range = _safe_numeric(frame["range_m"])

    speed_diff = np.diff(speed, prepend=speed[:1]) if speed.size else np.zeros(0, dtype=np.float32)
    acceleration_std = float(speed_diff.std()) if speed_diff.size else 0.0

    course_diff = np.diff(course, prepend=course[:1]) if course.size else np.zeros(0, dtype=np.float32)
    course_change_rate_mean = float(np.abs(course_diff).mean()) if course_diff.size else 0.0

    az_diff = np.diff(azimuth, prepend=azimuth[:1]) if azimuth.size else np.zeros(0, dtype=np.float32)
    range_diff = np.diff(track_range, prepend=track_range[:1]) if track_range.size else np.zeros(0, dtype=np.float32)
    curvature = np.sqrt(az_diff ** 2 + range_diff ** 2)

    hover_ratio = float((np.abs(speed) <= 1.0).mean()) if speed.size else 0.0 # 悬停比

    vector = np.array(
        [
            float(height.mean()) if height.size else 0.0, # 高度均值
            float(height.std()) if height.size else 0.0, # 高度标准差
            float(_linear_slope(height)), # 高度斜率
            float(speed.mean()) if speed.size else 0.0, # 地面速度均值
            float(speed.std()) if speed.size else 0.0, # 地面速度标准差
            float(speed.max()) if speed.size else 0.0, # 地面速度最大值
            float(course_change_rate_mean), # 方位角变化率均值
            float(curvature.mean()) if curvature.size else 0.0, # 轨迹曲率均值
            float(acceleration_std), # 加速度标准差
            float(hover_ratio), # 悬停比
            float(radial_speed.std()) if radial_speed.size else 0.0, # 径向速度标准差
            float(_spectral_flatness(radial_speed)), # 径向速度谱平滑度
            float(amplitude.mean()) if amplitude.size else 0.0, # 目标回波幅度均值
            float(amplitude.std()) if amplitude.size else 0.0, # 目标回波幅度标准差
            float(snr.mean()) if snr.size else 0.0, # 信噪比均值
            float(height_missing), # 高度缺失标记
            float(phase_missing), # 相位缺失标记
        ],
        dtype=np.float32,
    )
    return PhysicalFeatureVector(values=vector)
