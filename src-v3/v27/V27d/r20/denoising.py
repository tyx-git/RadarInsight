"""面向 V27d-r20 的密度-空间-可分性联合点集降噪。

该模块不依赖 sklearn，输入只需要包含 ``event_rows`` 的轨迹序列，输出每个
事件的 ``keep_weight`` 与可审计的噪声分数。设计目标是：

1. 用局部 kNN 密度识别孤立点和异常稀疏区域；
2. 用相邻事件的空间连续性识别跳点、速度不一致和轨迹断裂；
3. 用仅由训练集拟合的 Drone/Bird/Balloon 与 Clutter 原型保护可分目标、
   抑制杂波混叠区域；
4. 以目标类召回率约束阈值，避免把低密度 Bird/Drone 误删。

这里的 ``Noise`` Excel 列被视为不可信诊断字段，不进入降噪分数；其统计可在
外层报告中单独展示。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


TARGET_LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget")
PROTECTED_LABELS = ("DroneTarget", "BirdTarget")
X5A_FIELD_NAMES = tuple(f"u16_{index:02d}" for index in range(12))


@dataclass(frozen=True)
class DenoisingConfig:
    """可审计的降噪超参数。"""

    k_neighbors: int = 5
    target_retention_floor: float = 0.985
    score_temperature: float = 1.5
    density_weight: float = 0.24
    continuity_weight: float = 0.24
    clutter_weight: float = 0.34
    outlier_weight: float = 0.18
    min_event_keep_weight: float = 0.08
    range_scale_m: float = 5000.0
    vertical_scale_m: float = 1000.0
    radial_scale_mps: float = 25.0
    speed_scale_mps: float = 35.0
    height_scale_m: float = 1000.0
    amplitude_scale_db: float = 80.0
    # X5A 未知字段的保守空间尺度；仅对候选 u16 坐标接口生效。
    x5a_axis_scale: tuple[float, float, float] = (1000.0, 1000.0, 1000.0)
    x5a_saturation_penalty: float = 0.22
    x5a_zero_pattern_penalty: float = 0.06


@dataclass(frozen=True)
class DenoisingCalibration:
    """由训练 split 拟合的降噪统计量与类原型。"""

    config: DenoisingConfig
    feature_median: np.ndarray
    feature_scale: np.ndarray
    prototypes: Mapping[str, np.ndarray]
    density_center: float
    density_scale: float
    continuity_center: float
    continuity_scale: float
    threshold: float
    train_event_counts: Mapping[str, int]
    threshold_selection: Mapping[str, Any]

    def to_json(self) -> dict[str, Any]:
        """导出不含二进制数组的审计摘要。"""

        return {
            "config": {
                key: getattr(self.config, key)
                for key in self.config.__dataclass_fields__
            },
            "feature_median": self.feature_median.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "prototypes": {
                key: value.tolist() for key, value in self.prototypes.items()
            },
            "density_center": self.density_center,
            "density_scale": self.density_scale,
            "continuity_center": self.continuity_center,
            "continuity_scale": self.continuity_scale,
            "threshold": self.threshold,
            "train_event_counts": dict(self.train_event_counts),
            "threshold_selection": dict(self.threshold_selection),
        }


@dataclass(frozen=True)
class SequenceDenoisingResult:
    """单条轨迹的事件级降噪结果。"""

    remove_score: np.ndarray
    keep_weight: np.ndarray
    local_density: np.ndarray
    continuity_error: np.ndarray
    target_distance: np.ndarray
    clutter_affinity: np.ndarray
    removed_count: int

    def to_json(self) -> dict[str, Any]:
        return {
            "remove_score": self.remove_score.astype(float).tolist(),
            "keep_weight": self.keep_weight.astype(float).tolist(),
            "local_density": self.local_density.astype(float).tolist(),
            "continuity_error": self.continuity_error.astype(float).tolist(),
            "target_distance": self.target_distance.astype(float).tolist(),
            "clutter_affinity": self.clutter_affinity.astype(float).tolist(),
            "removed_count": int(self.removed_count),
        }


@dataclass(frozen=True)
class X5ARecordDenoisingResult:
    """X5A 单条变长记录的点级降噪结果。

    `u16_00..u16_11` 只表示二进制字段位置，不能解释为物理量。
    因此该接口只做协议安全的异常/稀疏性筛选，不使用类别原型。
    """

    remove_score: np.ndarray
    keep_weight: np.ndarray
    local_density: np.ndarray
    saturation_flag: np.ndarray
    zero_pattern_score: np.ndarray
    removed_count: int

    def to_json(self) -> dict[str, Any]:
        return {
            "remove_score": self.remove_score.astype(float).tolist(),
            "keep_weight": self.keep_weight.astype(float).tolist(),
            "local_density": self.local_density.astype(float).tolist(),
            "saturation_flag": self.saturation_flag.astype(int).tolist(),
            "zero_pattern_score": self.zero_pattern_score.astype(float).tolist(),
            "removed_count": int(self.removed_count),
        }


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _feature_vector(row: Mapping[str, Any], config: DenoisingConfig) -> tuple[np.ndarray, np.ndarray]:
    """生成稳健物理向量与米制局部坐标。"""

    range_m = max(0.0, _finite(row.get("range_m")))
    azimuth = radians(_finite(row.get("azimuth_deg")))
    elevation = radians(_finite(row.get("elevation_deg")))
    horizontal = range_m * cos(elevation)
    xyz = np.asarray(
        [
            horizontal * cos(azimuth),
            horizontal * sin(azimuth),
            range_m * sin(elevation),
        ],
        dtype=np.float64,
    )
    radial = _finite(row.get("radial_speed_mps"))
    speed = max(0.0, _finite(row.get("speed_mps")))
    height = _finite(row.get("height_m"))
    amplitude = max(0.0, _finite(row.get("amplitude_db")))
    feature = np.asarray(
        [
            range_m / config.range_scale_m,
            xyz[0] / config.range_scale_m,
            xyz[1] / config.range_scale_m,
            xyz[2] / config.vertical_scale_m,
            radial / config.radial_scale_mps,
            speed / config.speed_scale_mps,
            height / config.height_scale_m,
            amplitude / config.amplitude_scale_db,
            sin(elevation),
            cos(elevation),
        ],
        dtype=np.float64,
    )
    return feature, xyz


def _sequence_arrays(
    sequence: Any,
    config: DenoisingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = list(getattr(sequence, "event_rows", ()) or ())
    features: list[np.ndarray] = []
    coordinates: list[np.ndarray] = []
    for row in rows:
        feature, xyz = _feature_vector(row, config)
        features.append(feature)
        coordinates.append(xyz)
    if not features:
        return (
            np.zeros((0, 10), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )
    f = np.stack(features)
    xyz = np.stack(coordinates)
    density_distance = _local_knn_distance(xyz, config.k_neighbors)
    continuity_error = _continuity_error(rows, xyz)
    return f, xyz, density_distance, continuity_error


def _local_knn_distance(
    coordinates: np.ndarray,
    k_neighbors: int,
    *,
    scale: Sequence[float] | None = None,
    block_size: int = 1024,
) -> np.ndarray:
    """计算精确 kNN 平均距离，按 block 限制内存为 O(block_size*N)。"""

    if coordinates.shape[0] <= 1:
        return np.zeros(coordinates.shape[0], dtype=np.float64)
    axis_scale = np.asarray(
        tuple(scale) if scale is not None else (1000.0, 1000.0, 300.0),
        dtype=np.float64,
    )
    if axis_scale.shape != (coordinates.shape[1],) or bool((axis_scale <= 0.0).any()):
        raise ValueError("局部 kNN 坐标尺度必须为正且与维度一致")
    if isinstance(block_size, bool) or int(block_size) <= 0:
        raise ValueError("局部 kNN block_size 必须为正整数")
    normalized = np.asarray(coordinates, dtype=np.float64) / axis_scale
    k = max(1, min(int(k_neighbors), coordinates.shape[0] - 1))
    result = np.empty(coordinates.shape[0], dtype=np.float64)
    for start in range(0, coordinates.shape[0], int(block_size)):
        stop = min(coordinates.shape[0], start + int(block_size))
        delta = normalized[start:stop, None, :] - normalized[None, :, :]
        distances = np.sqrt(np.sum(delta * delta, axis=-1))
        local_rows = np.arange(stop - start)
        distances[local_rows, start + local_rows] = np.inf
        nearest = np.partition(distances, kth=k - 1, axis=1)[:, :k]
        result[start:stop] = np.mean(nearest, axis=1)
    return result


def _empirical_rank(values: np.ndarray) -> np.ndarray:
    """返回 [0,1] 的稳定经验分位秩，较大值对应较高异常度。"""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size <= 1:
        return np.zeros(values.size, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks / float(values.size - 1)


def score_x5a_record(
    point_values: np.ndarray,
    *,
    config: DenoisingConfig | None = None,
    remove_threshold: float = 0.995,
) -> X5ARecordDenoisingResult:
    """对一个 X5A 变长记录做协议安全的点级降噪。

    参数 ``point_values`` 必须是 ``(N, 12)`` 的 ``uint16``/数值矩阵，列语义
    仍未知。算法只使用三类可验证信号：候选前三列的局部 kNN 稀疏度、65535
    饱和标记、每点零值模式。默认 ``remove_threshold=0.995`` 只删除极端异常点；
    生产接入应在有标签 sidecar 后重新校准该阈值。
    """

    cfg = config or DenoisingConfig()
    values = np.asarray(point_values)
    if values.ndim != 2 or values.shape[1] != len(X5A_FIELD_NAMES):
        raise ValueError("X5A 点矩阵必须为 (N, 12)")
    if values.size and (not np.isfinite(values).all() or bool((values < 0).any())):
        raise ValueError("X5A 点矩阵必须为有限非负数")
    if not 0.0 < float(remove_threshold) < 1.0:
        raise ValueError("X5A 删除阈值必须在 (0,1) 内")
    numeric = values.astype(np.float64, copy=False)
    if numeric.shape[0] == 0:
        empty = np.zeros(0, dtype=np.float64)
        return X5ARecordDenoisingResult(empty, empty, empty, empty, 0)
    coordinates = numeric[:, :3]
    local_density = _local_knn_distance(
        coordinates,
        cfg.k_neighbors,
        scale=cfg.x5a_axis_scale,
    )
    sparse_score = _empirical_rank(np.log1p(local_density))
    saturation_flag = np.any(numeric >= 65535.0, axis=1).astype(np.float64)
    zero_fraction = np.mean(numeric == 0.0, axis=1)
    zero_rank = _empirical_rank(zero_fraction)
    zero_pattern_score = np.clip(np.maximum(zero_rank - 0.75, 0.0) * 4.0, 0.0, 1.0)
    remove_score = np.clip(
        cfg.density_weight * sparse_score
        + cfg.x5a_saturation_penalty * saturation_flag
        + cfg.x5a_zero_pattern_penalty * zero_pattern_score,
        0.0,
        1.0,
    )
    excess = np.maximum(remove_score - float(remove_threshold), 0.0)
    keep_weight = 1.0 - np.clip(
        excess / max(1.0 - float(remove_threshold), 1.0e-6),
        0.0,
        1.0,
    )
    keep_weight[remove_score <= float(remove_threshold)] = 1.0
    removed = int(np.sum(keep_weight <= 0.0))
    return X5ARecordDenoisingResult(
        remove_score=remove_score,
        keep_weight=keep_weight,
        local_density=local_density,
        saturation_flag=saturation_flag,
        zero_pattern_score=zero_pattern_score,
        removed_count=removed,
    )


def _continuity_error(rows: Sequence[Mapping[str, Any]], xyz: np.ndarray) -> np.ndarray:
    errors = np.zeros(len(rows), dtype=np.float64)
    if len(rows) <= 1:
        return errors
    for index in range(1, len(rows)):
        dt = max(0.0, _finite(rows[index].get("dt_seconds")))
        gap = float(np.linalg.norm(xyz[index] - xyz[index - 1]))
        speed = max(0.0, _finite(rows[index].get("speed_mps")))
        expected = max(8.0, speed * max(dt, 0.1))
        errors[index] = np.log1p(gap / expected)
    return errors


def _robust_center_scale(values: np.ndarray, floor: float = 1.0e-3) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 1.0
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = max(float(floor), 1.4826 * mad)
    return center, scale


def _standardize(features: np.ndarray, median: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (features - median) / np.maximum(scale, 1.0e-6)


def _sigmoid(value: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(value / max(float(temperature), 1.0e-6), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _score_from_parts(
    *,
    target_distance: np.ndarray,
    clutter_affinity: np.ndarray,
    density_error: np.ndarray,
    continuity_error: np.ndarray,
    config: DenoisingConfig,
) -> np.ndarray:
    outlier = np.clip(target_distance / 8.0, 0.0, 1.0)
    return np.clip(
        config.density_weight * density_error
        + config.continuity_weight * continuity_error
        + config.clutter_weight * clutter_affinity
        + config.outlier_weight * outlier,
        0.0,
        1.0,
    )


def _raw_scores(
    calibration: DenoisingCalibration,
    sequence: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features, _, density, continuity = _sequence_arrays(sequence, calibration.config)
    if not len(features):
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, empty, empty, empty, empty
    z = _standardize(features, calibration.feature_median, calibration.feature_scale)
    target_vectors = [
        calibration.prototypes[name]
        for name in TARGET_LABELS
        if name in calibration.prototypes
    ]
    if not target_vectors:
        raise ValueError("降噪校准缺少目标类原型")
    target_distances = np.min(
        np.stack([np.linalg.norm(z - proto[None, :], axis=1) for proto in target_vectors]),
        axis=0,
    )
    clutter_proto = calibration.prototypes.get("ClutterTarget")
    if clutter_proto is None:
        clutter_affinity = np.zeros(len(z), dtype=np.float64)
    else:
        clutter_distance = np.linalg.norm(z - clutter_proto[None, :], axis=1)
        clutter_affinity = _sigmoid(
            target_distances - clutter_distance,
            calibration.config.score_temperature,
        )
    density_log = np.log1p(density)
    density_error = np.clip(
        np.abs(density_log - calibration.density_center)
        / max(calibration.density_scale, 1.0e-6),
        0.0,
        1.0,
    )
    continuity_error = np.clip(
        np.abs(continuity - calibration.continuity_center)
        / max(3.0 * calibration.continuity_scale, 1.0e-6),
        0.0,
        1.0,
    )
    scores = _score_from_parts(
        target_distance=target_distances,
        clutter_affinity=clutter_affinity,
        density_error=density_error,
        continuity_error=continuity_error,
        config=calibration.config,
    )
    return scores, density, continuity_error, target_distances, clutter_affinity, z


def fit_denoiser(
    sequences: Iterable[Any],
    *,
    config: DenoisingConfig | None = None,
) -> DenoisingCalibration:
    """只使用训练 parent 拟合密度、连续性和类别原型。"""

    cfg = config or DenoisingConfig()
    sequence_list = list(sequences)
    if not sequence_list:
        raise ValueError("降噪校准序列为空")
    all_features: list[np.ndarray] = []
    class_features: dict[str, list[np.ndarray]] = {}
    all_density: list[np.ndarray] = []
    all_continuity: list[np.ndarray] = []
    counts: dict[str, int] = {}
    for sequence in sequence_list:
        features, _, density, continuity = _sequence_arrays(sequence, cfg)
        if not len(features):
            continue
        label = str(getattr(sequence, "label_name", ""))
        all_features.append(features)
        all_density.append(density)
        all_continuity.append(continuity)
        class_features.setdefault(label, []).append(features)
        counts[label] = counts.get(label, 0) + len(features)
    if not all_features:
        raise ValueError("降噪校准没有有效事件")
    target_features = np.concatenate(
        [np.concatenate(class_features[name]) for name in TARGET_LABELS if name in class_features],
        axis=0,
    )
    median = np.median(target_features, axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(target_features - median), axis=0), 1.0e-3)
    prototypes = {
        name: np.mean(_standardize(np.concatenate(values), median, scale), axis=0)
        for name, values in class_features.items()
        if name in (*TARGET_LABELS, "ClutterTarget") and values
    }
    # 按 parent 重新收集目标事件，避免空 parent 造成索引错位。
    target_continuity_parts: list[np.ndarray] = []
    target_density_parts: list[np.ndarray] = []
    for sequence in sequence_list:
        if str(getattr(sequence, "label_name", "")) not in TARGET_LABELS:
            continue
        _, _, density, continuity = _sequence_arrays(sequence, cfg)
        target_density_parts.append(np.log1p(density))
        target_continuity_parts.append(continuity)
    target_density = np.concatenate(target_density_parts) if target_density_parts else np.zeros(1)
    target_continuity = np.concatenate(target_continuity_parts) if target_continuity_parts else np.zeros(1)
    density_center, density_scale = _robust_center_scale(target_density)
    continuity_center, continuity_scale = _robust_center_scale(target_continuity)
    preliminary = DenoisingCalibration(
        config=cfg,
        feature_median=median,
        feature_scale=scale,
        prototypes=prototypes,
        density_center=density_center,
        density_scale=density_scale,
        continuity_center=continuity_center,
        continuity_scale=continuity_scale,
        threshold=0.99,
        train_event_counts=counts,
        threshold_selection={},
    )
    target_scores: dict[str, list[float]] = {name: [] for name in PROTECTED_LABELS}
    clutter_scores: list[float] = []
    for sequence in sequence_list:
        scores, *_ = _raw_scores(preliminary, sequence)
        label = str(getattr(sequence, "label_name", ""))
        if label in target_scores:
            target_scores[label].extend(scores.tolist())
        elif label == "ClutterTarget":
            clutter_scores.extend(scores.tolist())
    target_arrays = [np.asarray(values, dtype=np.float64) for values in target_scores.values() if values]
    combined_target_scores = np.concatenate(target_arrays)
    per_class_floor_thresholds = [
        float(np.quantile(values, cfg.target_retention_floor)) for values in target_arrays
    ]
    floor_threshold = max(per_class_floor_thresholds)
    candidate_values = np.unique(
        np.concatenate(
            [
                np.quantile(combined_target_scores, np.linspace(0.70, 0.9999, 256)),
                np.asarray(per_class_floor_thresholds + [floor_threshold]),
            ]
        )
    )
    best: tuple[float, float, float] | None = None
    best_threshold = floor_threshold
    for threshold in candidate_values:
        retentions = {
            name: float(np.mean(np.asarray(values) <= threshold))
            for name, values in target_scores.items()
            if values
        }
        if any(value + 1.0e-12 < cfg.target_retention_floor for value in retentions.values()):
            continue
        clutter_recall = (
            float(np.mean(np.asarray(clutter_scores) >= threshold)) if clutter_scores else 0.0
        )
        objective = (clutter_recall, min(retentions.values()), -float(threshold))
        if best is None or objective > best:
            best = objective
            best_threshold = float(threshold)
    selection = {
        "target_retention_floor": cfg.target_retention_floor,
        "target_retention": {
            name: float(np.mean(np.asarray(values) <= best_threshold))
            for name, values in target_scores.items()
            if values
        },
        "clutter_event_recall": float(np.mean(np.asarray(clutter_scores) >= best_threshold)) if clutter_scores else None,
        "candidate_count": int(len(candidate_values)),
        "selection_objective": list(best) if best is not None else None,
    }
    return DenoisingCalibration(
        config=cfg,
        feature_median=median,
        feature_scale=scale,
        prototypes=prototypes,
        density_center=density_center,
        density_scale=density_scale,
        continuity_center=continuity_center,
        continuity_scale=continuity_scale,
        threshold=best_threshold,
        train_event_counts=counts,
        threshold_selection=selection,
    )


def score_sequence(
    calibration: DenoisingCalibration,
    sequence: Any,
) -> SequenceDenoisingResult:
    """对一条轨迹计算可审计事件分数。"""

    scores, density, continuity, target_distance, clutter_affinity, _ = _raw_scores(
        calibration, sequence
    )
    threshold = float(calibration.threshold)
    excess = np.maximum(scores - threshold, 0.0)
    keep = 1.0 - np.clip(
        excess / max(1.0 - threshold, 1.0e-6),
        0.0,
        1.0,
    )
    keep[scores <= threshold] = 1.0
    keep[keep < calibration.config.min_event_keep_weight] = 0.0
    return SequenceDenoisingResult(
        remove_score=scores,
        keep_weight=keep,
        local_density=density,
        continuity_error=continuity,
        target_distance=target_distance,
        clutter_affinity=clutter_affinity,
        removed_count=int(np.sum(keep <= 0.0)),
    )


__all__ = [
    "DenoisingCalibration",
    "DenoisingConfig",
    "PROTECTED_LABELS",
    "SequenceDenoisingResult",
    "TARGET_LABELS",
    "X5A_FIELD_NAMES",
    "X5ARecordDenoisingResult",
    "fit_denoiser",
    "score_sequence",
    "score_x5a_record",
]
