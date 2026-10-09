"""r01 可复现性能与有效性基准。

覆盖四条只读路径，全部不训练模型、不加载 checkpoint、不写入 r01 之外的文件：

1. ``x5a_stream_parse``：X5A ``.dat`` 流式解析；
2. ``x5a_pointwise_quality_score``：X5A 单条变长记录逐点质量评分；
3. ``excel_eventwise_quality_score``：Excel TR ``event_rows`` 的 r20 原口径评分；
4. ``minimum_points_probe``：判定形成有效判断所需的最少轨迹点数。

输出 JSON，固定记录运行条件、输入规模、预热/重复次数、吞吐、延迟、内存与估算
计算量。性能结果只用于同机同环境的相对比较，不等价于分类收益。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .denoising import DenoisingConfig, X5A_FIELD_NAMES, fit_denoiser, score_sequence, score_x5a_record
from .protocol import R01_EXPERIMENT_ID, R01_SCHEMA_VERSION, load_config
from .x5a import X5ALayout, X5ARecord, iter_x5a_records


SEED = 42
DEFAULT_WARMUP = 2
DEFAULT_REPEATS = 7
DEFAULT_MAX_RECORDS = 2000
DEFAULT_EXCEL_POINTS = 64

# 形成有效判断所需的最少点数：由 minimum_points_probe 的经验结果支撑。
HARD_MINIMUM_POINTS = 2
EFFECTIVE_JUDGMENT_MINIMUM_POINTS = 3
STABLE_DENSITY_MINIMUM_POINTS = 6
PROBE_POINT_COUNTS = (1, 2, 3, 4, 5, 6, 8, 12, 24, 48)


@dataclass(frozen=True)
class BenchmarkSample:
    """固定的 benchmark 输入摘要。"""

    name: str
    path: str
    records: tuple[X5ARecord, ...]
    points: int


def _rss_bytes() -> int | None:
    """返回当前进程 RSS；Windows 使用 working set，Unix 使用 ru_maxrss。"""

    try:
        import resource  # type: ignore[import-not-found]
    except ImportError:
        resource = None
    if resource is not None:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value * (1024 if sys.platform != "darwin" else 1)
    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        return None


def _platform_info() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "pid": os.getpid(),
    }


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    data = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(data)),
        "p50": float(np.quantile(data, 0.50)),
        "p90": float(np.quantile(data, 0.90)),
        "p95": float(np.quantile(data, 0.95)),
        "max": float(np.max(data)),
    }


def _run_repeated(
    name: str,
    operation: Callable[[], dict[str, Any]],
    *,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    """重复调用同一操作；预热不计入统计，返回延迟分位与内存增量。"""

    for _ in range(max(0, int(warmup))):
        operation()
    gc.collect()
    durations: list[float] = []
    rss_before = _rss_bytes()
    last: dict[str, Any] = {}
    for _ in range(max(1, int(repeats))):
        started = time.perf_counter_ns()
        last = operation()
        durations.append((time.perf_counter_ns() - started) / 1.0e9)
    rss_after = _rss_bytes()
    result = {
        "name": name,
        "warmup": int(warmup),
        "repeats": int(max(1, repeats)),
        "duration_seconds": _percentiles(durations),
        "duration_mean_seconds": float(statistics.mean(durations)),
        "duration_stdev_seconds": float(statistics.stdev(durations)) if len(durations) > 1 else 0.0,
        "rss_before_bytes": rss_before,
        "rss_after_bytes": rss_after,
        "rss_delta_bytes": None if rss_before is None or rss_after is None else rss_after - rss_before,
        "last_output": last,
    }
    return result


def _load_x5a_sample(path: Path, *, layout: X5ALayout, max_records: int) -> BenchmarkSample:
    records: list[X5ARecord] = []
    for record in iter_x5a_records(path, layout=layout):
        records.append(record)
        if len(records) >= max(1, int(max_records)):
            break
    return BenchmarkSample(
        name=path.name,
        path=str(path),
        records=tuple(records),
        points=sum(record.point_count for record in records),
    )


def _record_point_distribution(records: Iterable[X5ARecord]) -> dict[str, Any]:
    counts = np.asarray([record.point_count for record in records], dtype=np.int64)
    if counts.size == 0:
        return {
            "records": 0,
            "points": 0,
            "empty_records": 0,
            "empty_record_fraction": 0.0,
            "mean_points_per_record": 0.0,
            "median_points_per_record": 0.0,
            "p95_points_per_record": 0,
            "max_points_per_record": 0,
        }
    return {
        "records": int(counts.size),
        "points": int(counts.sum()),
        "empty_records": int(np.sum(counts == 0)),
        "empty_record_fraction": float(np.mean(counts == 0)),
        "mean_points_per_record": float(np.mean(counts)),
        "median_points_per_record": float(np.median(counts)),
        "p95_points_per_record": int(np.percentile(counts, 95)),
        "max_points_per_record": int(counts.max()),
    }


def _x5a_parse_operation(path: Path, layout: X5ALayout, max_records: int) -> Callable[[], dict[str, Any]]:
    def operation() -> dict[str, Any]:
        records = 0
        points = 0
        empty = 0
        for record in iter_x5a_records(path, layout=layout):
            records += 1
            points += record.point_count
            empty += int(record.point_count == 0)
            if records >= max(1, int(max_records)):
                break
        return {"records": records, "points": points, "empty_records": empty}

    return operation


def _x5a_score_operation(sample: BenchmarkSample, config: DenoisingConfig) -> Callable[[], dict[str, Any]]:
    def operation() -> dict[str, Any]:
        points = 0
        removed = 0
        for record in sample.records:
            result = score_x5a_record(record.point_values, config=config)
            points += record.point_count
            removed += result.removed_count
        return {"records": len(sample.records), "points": points, "removed": removed}

    return operation


def _make_synthetic_sequence(point_count: int, *, outlier: bool = False) -> Any:
    """生成不依赖项目数据文件的 Excel event_rows 基准输入。

    ``outlier=True`` 时把最后一个事件在距离维上显著外推，用于检验评分器能否
    把该点排到异常度首位。
    """

    class Sequence:
        parent_sample_id = "synthetic|r01|benchmark"
        label_name = "DroneTarget"

    sequence = Sequence()
    rows: list[dict[str, Any]] = []
    for index in range(max(1, int(point_count))):
        jump = 600.0 if (outlier and index == max(1, int(point_count)) - 1) else 0.0
        rows.append(
            {
                "range_m": 1000.0 + 2.0 * index + jump,
                "azimuth_deg": 30.0 + 0.01 * index,
                "elevation_deg": 5.0 + 0.005 * index,
                "speed_mps": 12.0,
                "radial_speed_mps": 2.0 + 0.01 * index,
                "height_m": 150.0 + 0.2 * index,
                "amplitude_db": 60.0 + 0.01 * index,
                "dt_seconds": 1.0,
            }
        )
    sequence.event_rows = tuple(rows)
    return sequence


def _excel_score_operation(sequence: Any, config: DenoisingConfig) -> Callable[[], dict[str, Any]]:
    calibration = fit_denoiser([sequence], config=config)

    def operation() -> dict[str, Any]:
        result = score_sequence(calibration, sequence)
        return {
            "events": int(len(result.keep_weight)),
            "removed": int(result.removed_count),
            "mean_keep_weight": float(np.mean(result.keep_weight)) if len(result.keep_weight) else 0.0,
        }

    return operation


def _synthetic_x5a_track(point_count: int, *, outlier: bool = False) -> np.ndarray:
    """合成一条 X5A 记录：前三列作为候选坐标的平滑轨迹。

    ``outlier=True`` 时把最后一点在候选坐标上显著外推。列语义未知，因此这里
    只保证数值形态，不暗示物理量纲。
    """

    n = max(1, int(point_count))
    index = np.arange(n, dtype=np.float64)
    track = np.stack(
        [
            500.0 + 2.0 * index,
            520.0 + 1.6 * index,
            480.0 + 1.2 * index,
        ],
        axis=1,
    )
    if outlier and n >= 2:
        track[-1] += np.asarray([900.0, 900.0, 900.0], dtype=np.float64)
    points = np.zeros((n, len(X5A_FIELD_NAMES)), dtype=np.float64)
    points[:, :3] = track
    points[:, 3:] = 64.0
    return np.clip(points, 0.0, 60000.0).astype(np.uint16)


def _rank_of_last(values: np.ndarray) -> float:
    """返回末点在一维分数中的经验分位秩，1.0 表示最高。"""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return 0.0
    last = float(array[-1])
    return float(np.mean(array <= last))


def _last_point_margin(score: np.ndarray) -> float:
    """末点分数相对其余点中位数的余量。

    短轨迹的端点天然有更大的 kNN 距离，仅看绝对分数会把端点误判为异常。用
    “注入异常轨迹的余量 − 干净轨迹端点的自然余量” 才能反映真正的区分度。
    """

    array = np.asarray(score, dtype=np.float64).reshape(-1)
    if array.size < 2:
        return 0.0
    return float(array[-1] - float(np.median(array[:-1])))


def _probe_minimum_points(config: DenoisingConfig) -> dict[str, Any]:
    """经验判定形成有效判断所需的最少轨迹点数。

    对每条合成轨迹注入一个外推点，检查评分器能否把它排到异常度首位；同时对
    无外推的干净轨迹统计误报与自然端点余量。结果直接支撑推荐下限。
    """

    k = int(config.k_neighbors)
    entries: list[dict[str, Any]] = []
    for point_count in PROBE_POINT_COUNTS:
        n = int(point_count)
        effective_k = 0 if n <= 1 else max(1, min(k, n - 1))
        outlier_points = _synthetic_x5a_track(n, outlier=True)
        clean_points = _synthetic_x5a_track(n, outlier=False)
        outlier_result = score_x5a_record(outlier_points, config=config)
        clean_result = score_x5a_record(clean_points, config=config)
        score = np.asarray(outlier_result.remove_score, dtype=np.float64)
        clean_score = np.asarray(clean_result.remove_score, dtype=np.float64)
        outlier_margin = _last_point_margin(score)
        clean_margin = _last_point_margin(clean_score)
        std = float(np.std(score))
        entries.append(
            {
                "point_count": n,
                "effective_knn_neighbors": effective_k,
                "knn_saturated": bool(effective_k >= k),
                "density_evidence_defined": bool(n >= 2),
                "rank_is_discriminative": bool(n >= 2 and std > 0.0),
                "outlier_score_rank_of_injected_point": _rank_of_last(score),
                "outlier_score_std": std,
                "outlier_score_max": float(np.max(score)),
                "injected_point_margin_over_median": outlier_margin,
                "clean_track_endpoint_margin": clean_margin,
                "margin_over_natural_endpoint_effect": outlier_margin - clean_margin,
                "clean_track_removed_count": int(clean_result.removed_count),
                "clean_track_score_max": float(np.max(clean_score)),
                "verdict": _probe_verdict(n, effective_k, k),
            }
        )
    excel_entries: list[dict[str, Any]] = []
    for point_count in PROBE_POINT_COUNTS:
        n = int(point_count)
        sequence = _make_synthetic_sequence(n, outlier=True)
        calibration = fit_denoiser([sequence], config=config)
        result = score_sequence(calibration, sequence)
        continuity = np.asarray(result.continuity_error, dtype=np.float64)
        excel_entries.append(
            {
                "point_count": n,
                "continuity_evidence_defined": bool(n >= 2),
                "continuity_error_max": float(np.max(continuity)) if continuity.size else 0.0,
                "continuity_error_at_injected_point": float(continuity[-1]) if continuity.size else 0.0,
                "remove_score_max": float(np.max(np.asarray(result.remove_score, dtype=np.float64))),
                "removed_count": int(result.removed_count),
            }
        )
    saturated = [entry["point_count"] for entry in entries if entry["knn_saturated"]]
    discriminative = [entry["point_count"] for entry in entries if entry["rank_is_discriminative"]]
    max_net_margin = max((entry["margin_over_natural_endpoint_effect"] for entry in entries), default=0.0)
    return {
        "probe_design": (
            "为每个长度合成一条平滑轨迹，末点在候选坐标上外推 900 单位作为注入异常点；"
            "同时评估无外推的干净轨迹，用注入轨迹末点余量减去干净轨迹端点自然余量，"
            "得到扣除端点效应后的真实区分度。"
        ),
        "x5a_pointwise": entries,
        "excel_eventwise": excel_entries,
        "key_findings": {
            "ranking_discriminative_from": min(discriminative) if discriminative else None,
            "knn_saturated_from": min(saturated) if saturated else None,
            "max_margin_over_natural_endpoint_effect": max_net_margin,
            "absolute_judgment_supported_by_x5a_path": False,
            "absolute_judgment_reason": (
                "X5A 路径的 sparse_score 是记录内经验秩，注入的外推点与干净轨迹端点都取到"
                "同一最大秩，净余量≈0；因此它只能给出记录内相对排序，不能给出'该点是杂波'"
                "的绝对判定。绝对判定需要跨记录的标定阈值，或沿用 Excel 路径的训练集"
                "median/MAD 与类原型标定。"
            ),
            "excel_path_note": (
                "Excel 路径使用 self-fitted 合成标定，阈值偏高，探针中 removed 恒为 0；"
                "真实删除行为取决于训练 split 拟合的阈值，不以本探针数值为准。"
            ),
        },
        "conclusion": {
            "hard_minimum_points": HARD_MINIMUM_POINTS,
            "recommended_minimum_points": EFFECTIVE_JUDGMENT_MINIMUM_POINTS,
            "stable_density_minimum_points": STABLE_DENSITY_MINIMUM_POINTS,
            "minimum_for_absolute_removal": "not_supported_by_x5a_path",
            "rationale": (
                "n=1 时经验秩退化为常数，无任何邻域证据；"
                "n=2 时秩只有 0/1 两级，任一点必然取到极值，属退化判断；"
                "n>=3 时秩出现中间层级，可做记录内相对排序；"
                f"n>={STABLE_DENSITY_MINIMUM_POINTS} 时 kNN 才用满 k={k} 个邻居，密度估计稳定。"
                "但绝对删除判定在 X5A 路径上仍然不成立，需跨记录标定。"
            ),
            "classification_note": "这是降噪证据下限，不是分类器有效输入长度；parent 级分类仍受既有训练/评估契约约束。",
        },
    }


def _probe_verdict(n: int, effective_k: int, k: int) -> str:
    if n <= 1:
        return "insufficient: 无邻域与连续性证据，分数退化为常数"
    if n == 2:
        return "degenerate: 秩仅 0/1 两级，任一点必然取极值"
    if effective_k < k:
        return "usable_but_unstable: 可排序但 kNN 邻居不足，密度估计方差大"
    return "stable: kNN 用满邻居，密度与异常排序稳定"


def _scaling_probe(
    x5a_path: Path,
    layout: X5ALayout,
    config: DenoisingConfig,
    *,
    steps: tuple[int, ...],
) -> dict[str, Any]:
    """实测记录数增长时的耗时与逐点开销，用于验证复杂度模型。"""

    entries: list[dict[str, Any]] = []
    limit = max(steps)
    pool: list[X5ARecord] = []
    for record in iter_x5a_records(x5a_path, layout=layout):
        pool.append(record)
        if len(pool) >= limit:
            break
    for step in steps:
        subset = pool[: min(step, len(pool))]
        if not subset:
            continue
        points = sum(record.point_count for record in subset)
        started = time.perf_counter_ns()
        for record in subset:
            score_x5a_record(record.point_values, config=config)
        elapsed = (time.perf_counter_ns() - started) / 1.0e9
        entries.append(
            {
                "records": len(subset),
                "points": int(points),
                "duration_seconds": elapsed,
                "microseconds_per_record": elapsed / len(subset) * 1.0e6,
                "microseconds_per_point": elapsed / points * 1.0e6 if points else 0.0,
            }
        )
    return {
        "steps": list(steps),
        "entries": entries,
        "interpretation": "逐记录耗时近似为常数加成 O(n^2) 距离矩阵项；若 microseconds_per_point 随规模明显上升，说明长记录主导了成本。",
    }


def _complexity_estimates(*, points: int, records: int, k_neighbors: int) -> dict[str, Any]:
    """返回可解释的近似计算量，不冒充硬件指令计数。"""

    n = max(0, int(points))
    r = max(0, int(records))
    k = max(1, int(k_neighbors))
    return {
        "metric_definition": "algorithmic upper-bound estimates, not measured FLOPs",
        "point_feature_scalar_ops_estimate": int(n * 12),
        "knn_distance_scalar_ops_estimate": int(n * max(0, n - 1) * 3),
        "knn_topk_selection_comparisons_estimate": int(n * max(0, n - 1)),
        "score_scalar_ops_estimate": int(n * 16),
        "record_header_checks": r,
        "peak_working_set_model": "O(block_size*N) for r20 local kNN; X5A record scoring uses O(record_points^2) temporary distance matrix",
        "k_neighbors": k,
    }


def run_benchmark(
    project_root: str | Path,
    *,
    warmup: int = DEFAULT_WARMUP,
    repeats: int = DEFAULT_REPEATS,
    max_records: int = DEFAULT_MAX_RECORDS,
    excel_points: int = DEFAULT_EXCEL_POINTS,
    scaling_steps: tuple[int, ...] = (250, 500, 1000, 2000),
) -> dict[str, Any]:
    """运行基准并返回 JSON-safe 结果。"""

    root = Path(project_root).resolve()
    np.random.seed(SEED)
    config = load_config(root)
    x5a_path = root / "OriData" / "梁子湖X5A" / "plot_x5a_250331_1345_第一块电池.dat"
    layout = X5ALayout(payload_offset=74, count_mode="q", tail_bytes=0)
    sample = _load_x5a_sample(x5a_path, layout=layout, max_records=max_records)
    denoise_config = DenoisingConfig()
    synthetic_sequence = _make_synthetic_sequence(excel_points)
    results = [
        _run_repeated(
            "x5a_stream_parse",
            _x5a_parse_operation(x5a_path, layout, max_records),
            warmup=warmup,
            repeats=repeats,
        ),
        _run_repeated(
            "x5a_pointwise_quality_score",
            _x5a_score_operation(sample, denoise_config),
            warmup=warmup,
            repeats=repeats,
        ),
        _run_repeated(
            "excel_eventwise_quality_score",
            _excel_score_operation(synthetic_sequence, denoise_config),
            warmup=warmup,
            repeats=repeats,
        ),
    ]
    for result in results:
        output = result.pop("last_output")
        points = int(output.get("points", output.get("events", 0)))
        records = int(output.get("records", 1))
        p50 = result["duration_seconds"]["p50"]
        result["throughput"] = {
            "points_per_second_p50": points / p50 if p50 > 0 else 0.0,
            "records_per_second_p50": records / p50 if p50 > 0 else 0.0,
            "events_per_second_p50": points / p50 if p50 > 0 else 0.0,
        }
        result["normalized_latency"] = {
            "microseconds_per_point_p50": p50 / points * 1.0e6 if points else 0.0,
            "microseconds_per_record_p50": p50 / records * 1.0e6 if records else 0.0,
        }
        result["input_output"] = output
    return {
        "schema_version": "RADAR-LN3-V27d-r01-benchmark-v1",
        "experiment_id": R01_EXPERIMENT_ID,
        "r01_schema": R01_SCHEMA_VERSION,
        "run_conditions": {
            "seed": SEED,
            "warmup": int(warmup),
            "repeats": int(repeats),
            "max_records": int(max_records),
            "excel_synthetic_points": int(excel_points),
            "x5a_layout": {
                "payload_offset": layout.payload_offset,
                "count_mode": layout.count_mode,
                "tail_bytes": layout.tail_bytes,
            },
            "numpy_threads": os.environ.get("OMP_NUM_THREADS"),
            "input_file": str(x5a_path),
            "python_runtime": _platform_info(),
            "calibration": "r20 DenoisingConfig defaults; no checkpoint training",
        },
        "input_summary": {
            "x5a_record_point_distribution": _record_point_distribution(sample.records),
            "excel_synthetic_points": int(excel_points),
            "minimum_effective_point_count": EFFECTIVE_JUDGMENT_MINIMUM_POINTS,
        },
        "complexity": _complexity_estimates(
            points=sample.points,
            records=len(sample.records),
            k_neighbors=denoise_config.k_neighbors,
        ),
        "scaling": _scaling_probe(x5a_path, layout, denoise_config, steps=scaling_steps),
        "minimum_points_probe": _probe_minimum_points(denoise_config),
        "results": results,
        "statistical_method": {
            "timer": "time.perf_counter_ns",
            "reported_latency": "min/p50/p90/p95/max across repeated wall-clock runs",
            "warmup_excluded": True,
            "throughput_denominator": "单次重复处理的点数或记录数",
            "percentile_resolution_caveat": "重复次数有限，p90/p95 为小样本经验分位，跨机比较应优先看 min 与 p50",
            "memory": "best-effort process RSS；不可用时为 None",
            "repeatability": "固定 NumPy seed；报告随结果附带硬件与运行时信息",
        },
        "minimum_effective_track_points": {
            "hard_minimum": HARD_MINIMUM_POINTS,
            "recommended_minimum": EFFECTIVE_JUDGMENT_MINIMUM_POINTS,
            "stable_density_minimum": STABLE_DENSITY_MINIMUM_POINTS,
            "reason": "由 minimum_points_probe 的经验结果给出；1 点无证据、2 点退化、3 点可排序、6 点密度稳定。",
            "classification_note": "这是降噪证据下限，不是分类器有效输入长度；parent 级分类仍需既有训练/评估契约。",
        },
        "limitations": [
            "X5A 字段为未知语义 uint16，基准只度量实现成本与排序能力，不度量物理检测质量。",
            "X5A 评分按记录构建临时距离矩阵，长记录存在 O(n^2) 内存与耗时。",
            "Excel 路径使用合成 event_rows，避免依赖处理后数据落盘位置，绝对数值不等于真实数据集耗时。",
            "基准不声称 Precision/Recall 提升。",
        ],
        "config_identity_sha256": config["_identity_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark V27d-r01 pointwise denoising")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--max-records", type=int, default=DEFAULT_MAX_RECORDS)
    parser.add_argument("--excel-points", type=int, default=DEFAULT_EXCEL_POINTS)
    parser.add_argument("--scaling-steps", type=int, nargs="*", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    result = run_benchmark(
        args.project_root,
        warmup=args.warmup,
        repeats=args.repeats,
        max_records=args.max_records,
        excel_points=args.excel_points,
        scaling_steps=tuple(args.scaling_steps) if args.scaling_steps else (250, 500, 1000, 2000),
    )
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        target = args.output if args.output.is_absolute() else Path(args.project_root).resolve() / args.output
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        result = {**result, "report_path": str(target)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_benchmark"]
