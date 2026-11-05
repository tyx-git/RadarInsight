"""Build a deterministic multi-parent ASD set with AeroSight's writer.

The input is the CQ08 source tree. Each output package contains one CQ08
parent and keeps raw physical RD frames; the package is written by
``AeroSight.tools.rrtsp_tool.rrtsp.asd_format.AsdWriter``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from radar_ln3.data.cq08 import CQ08Adapter  # noqa: E402
from radar_ln3.data.paths import resolve_source_path  # noqa: E402
from radar_ln3.data.raw_reader import read_mat  # noqa: E402
from radar_ln3.data.rd_sanitizer import sanitize_rd_amplitude_db  # noqa: E402
from radar_ln3.data.source import resolve_registered_source_path  # noqa: E402
from radar_ln3.features.micro_doppler import (  # noqa: E402
    MICRO_DOPPLER_FRAME_CAP,
    MICRO_DOPPLER_FRAME_VALUE_DIM,
    STABLE8_INDICES,
    extract_current_event_broadband_micro_doppler,
)


ASD_ROOT = PROJECT_ROOT / "AeroSight"
if str(ASD_ROOT / "tools" / "rrtsp_tool") not in sys.path:
    sys.path.insert(0, str(ASD_ROOT / "tools" / "rrtsp_tool"))
from rrtsp.asd_format import AsdWriter  # noqa: E402


TR_FIELDS = (
    "range_m",
    "azimuth_deg",
    "elevation_deg",
    "radial_speed_mps",
    "amplitude_db",
    "snr_db",
    "speed_mps",
    "course_deg",
)
LABEL_ORDER = (
    "DroneTarget",
    "BirdTarget",
    "BalloonTarget",
    "ClutterTarget",
    "UnknownTarget",
)
LABEL_SLUGS = {
    "DroneTarget": "dronetarget",
    "BirdTarget": "birdtarget",
    "BalloonTarget": "balloontarget",
    "ClutterTarget": "cluttertarget",
    "UnknownTarget": "unknowntarget",
}


def _finite(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 不是数值: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} 不是有限数值: {value!r}")
    return result


def _resolve_frame_path(adapter: CQ08Adapter, raw_path: str) -> Path:
    if str(raw_path).replace("\\", "/").startswith("external/"):
        return resolve_source_path(
            adapter.project_root,
            raw_path,
            source_roots={"cq08": adapter.source_root},
            must_exist=True,
            allow_external=True,
        )
    return resolve_registered_source_path(
        adapter.project_root,
        raw_path,
        adapter.source_config,
        must_exist=True,
    )


def _debug_scalar(debug: Any, name: str, *, path: Path) -> float:
    value = np.asarray(getattr(debug, name, None)).reshape(-1)
    if value.size != 1:
        raise ValueError(f"MAT debug_info.{name} 不是标量: {path}")
    return _finite(value[0], name=f"{path}:debug_info.{name}")


def _raw_frame(
    path: Path,
    *,
    time_seconds: float,
    raw_frame_index: int,
    mat_frame_number: int,
    track_point_index: int,
    source_path: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    payload = read_mat(
        path,
        variable_names=(
            "data_proc_MTD_result_db",
            "data_proc_MTD_result",
            "Range_plot",
            "Vr",
            "debug_info",
        ),
    )
    if "data_proc_MTD_result_db" not in payload:
        raise ValueError(f"MAT 缺少 data_proc_MTD_result_db: {path}")
    complex_values = payload.get("data_proc_MTD_result")
    sanitized = sanitize_rd_amplitude_db(
        payload["data_proc_MTD_result_db"],
        complex_values,
        name=str(path),
    )
    amplitude = np.asarray(sanitized.values, dtype=np.float32)
    if complex_values is None:
        raise ValueError(f"MAT 缺少 phase 所需的 data_proc_MTD_result: {path}")
    complex_array = np.asarray(complex_values)
    if complex_array.shape != amplitude.shape:
        raise ValueError(f"MAT amplitude/complex shape 不一致: {path}")
    phase = np.asarray(np.angle(complex_array), dtype=np.float32)
    if not np.isfinite(phase).all():
        raise ValueError(f"MAT phase 包含非有限值: {path}")

    range_axis = np.asarray(payload["Range_plot"], dtype=np.float64).reshape(-1)
    velocity_axis = np.asarray(payload["Vr"], dtype=np.float64).reshape(-1)
    if amplitude.ndim != 2:
        raise ValueError(f"MAT amplitude 不是二维: {path}")
    if range_axis.shape != (amplitude.shape[0],):
        raise ValueError(f"MAT range axis 与 amplitude 不一致: {path}")
    if velocity_axis.shape != (amplitude.shape[1],):
        raise ValueError(f"MAT velocity axis 与 amplitude 不一致: {path}")
    if not np.isfinite(range_axis).all() or not np.isfinite(velocity_axis).all():
        raise ValueError(f"MAT 物理轴包含非有限值: {path}")
    debug = payload.get("debug_info")
    if debug is None:
        raise ValueError(f"MAT 缺少 debug_info: {path}")
    target_range_m = _debug_scalar(debug, "detected_range_m", path=path)
    axes = {
        "range_axis_m": range_axis.tolist(),
        "velocity_axis_mps": velocity_axis.tolist(),
        "target_range_m": target_range_m,
        "source_path": source_path,
        "raw_frame_index": int(raw_frame_index),
        "mat_frame_number": int(mat_frame_number),
        "track_point_index": int(track_point_index),
    }
    return amplitude, phase, axes


def _stable_evidence(adapter: CQ08Adapter, event: Any) -> dict[str, Any]:
    raw_frames = list(adapter.read_event_frames(event))
    micro_event = extract_current_event_broadband_micro_doppler(
        raw_frames,
        event_index=int(event.causal_event_index),
    )
    full_values = np.asarray(micro_event.frame_values, dtype=np.float32).copy()
    reliable = np.asarray(
        micro_event.frame_feature_reliable, dtype=np.bool_
    )
    full_values[~reliable] = 0.0
    stable_values = np.asarray(
        full_values[:, STABLE8_INDICES], dtype=np.float32
    )
    stable_values[~reliable] = 0.0
    return {
        "event_index": int(event.causal_event_index),
        "frame_cap": MICRO_DOPPLER_FRAME_CAP,
        "stable8_values": np.asarray(micro_event.values, dtype=np.float32).tolist(),
        "stable8_valid": bool(micro_event.valid),
        "stable8_frame_values": stable_values.tolist(),
        "stable8_frame_mask": reliable.tolist(),
        "micro_doppler_frame_values": full_values.tolist(),
        "micro_doppler_frame_mask": np.asarray(
            micro_event.frame_mask, dtype=np.bool_
        ).tolist(),
        "micro_doppler_reliable": reliable.tolist(),
        "raw_frame_indices": np.asarray(
            micro_event.raw_frame_indices, dtype=np.int32
        ).tolist(),
    }


def _build_parent(adapter: CQ08Adapter, parent: Any, output_path: Path) -> dict[str, Any]:
    events = adapter.discover_events(parent)
    if not events:
        raise ValueError(f"parent 没有事件: {parent.parent_sample_id}")
    raw_label_id = parent.metadata.get("raw_label_id")
    source = {
        "dataset": "CQ08",
        "session": "cq08",
        "track_id": str(parent.track_id),
        "raw_label_id": int(raw_label_id) if raw_label_id is not None else None,
        "mat_file_class": str(parent.label_name).replace("Target", ""),
        "parent_sample_id": str(parent.parent_sample_id),
        "source_track_path": str(parent.source_path),
        "source_point_path": str(parent.metadata.get("point_source_path", "")),
    }
    writer = AsdWriter(
        str(output_path),
        kind="track",
        label=parent.label_id,
        label_id=parent.label_id,
        label_name=parent.label_name,
        source=source,
        parent_sample_id=parent.parent_sample_id,
        tr_amplitude_semantics="linear_magnitude_legacy_cq08",
        rd_amplitude_mode="db",
    )
    elapsed = 0.0
    rd_frame_count = 0
    rd_event_count = 0
    for event in events:
        raw_values = dict(event.metadata.get("tr_values") or {})
        tr_values = {
            name: _finite(raw_values.get(name), name=f"{event.sample_id}:tr.{name}")
            for name in TR_FIELDS
        }
        timestamp_seconds = _finite(
            raw_values.get("time_seconds"),
            name=f"{event.sample_id}:timestamp_seconds",
        )
        raw_paths = list(event.metadata.get("rd_frame_paths") or [])
        frame_metadata = list(event.metadata.get("rd_frame_metadata") or [])
        rd_frames = []
        for position, raw_path in enumerate(raw_paths):
            source_meta = (
                dict(frame_metadata[position])
                if position < len(frame_metadata)
                and isinstance(frame_metadata[position], Mapping)
                else {}
            )
            duration = max(0.0, float(source_meta.get("duration_seconds") or 0.0))
            mat_frame_number = int(source_meta.get("frame_index", position + 1))
            path = _resolve_frame_path(adapter, str(raw_path))
            amplitude, phase, axes = _raw_frame(
                path,
                time_seconds=elapsed,
                raw_frame_index=mat_frame_number - 1,
                mat_frame_number=mat_frame_number,
                track_point_index=int(
                    event.metadata.get("track_point_index", event.causal_event_index + 1)
                ),
                source_path=str(raw_path),
            )
            axes["rd_observation_duration_seconds"] = duration
            rd_frames.append((elapsed, amplitude, phase, axes))
            elapsed += duration
            rd_frame_count += 1
        if rd_frames:
            rd_event_count += 1
        stable_evidence = _stable_evidence(adapter, event)
        writer.add_event(
            timestamp_seconds=timestamp_seconds,
            event_index=int(event.causal_event_index),
            track_point_index=int(
                event.metadata.get("track_point_index", event.causal_event_index + 1)
            ),
            tr=tr_values,
            rd_frames=rd_frames,
            stable_evidence=stable_evidence,
            source={
                "sample_id": str(event.sample_id),
                "source_name": "CQ08",
                "track_time_raw": event.metadata.get("track_time_raw"),
            },
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer.finish()
    return {
        "path": str(output_path),
        "parent_sample_id": str(parent.parent_sample_id),
        "track_id": str(parent.track_id),
        "label_id": int(parent.label_id) if parent.label_id is not None else None,
        "label_name": str(parent.label_name),
        "split_name": str(parent.split_name),
        "event_count": len(events),
        "rd_event_count": rd_event_count,
        "rd_frame_count": rd_frame_count,
    }


def _select_parents(
    parents: Sequence[Any],
    per_label: int | None,
    *,
    all_parents: bool = False,
) -> list[Any]:
    if all_parents:
        return [
            parent
            for parent in parents
            if int(parent.event_count or 0) > 0
        ]
    if per_label is None or per_label <= 0:
        raise ValueError("per_label 必须为正数")
    grouped: dict[str, list[Any]] = defaultdict(list)
    for parent in parents:
        grouped[str(parent.label_name)].append(parent)
    selected: list[Any] = []
    for label in LABEL_ORDER:
        candidates = [
            parent
            for parent in grouped.get(label, [])
            if int(parent.event_count or 0) > 0
            and int(parent.metadata.get("rd_mat_count", 0)) > 0
        ]
        # Prefer complete causal sequences and richer RD evidence. Track ID is
        # only a deterministic tie-breaker, so the balanced demo set is not
        # biased toward the lowest-numbered captures.
        candidates.sort(
            key=lambda parent: (
                -min(int(parent.event_count or 0), 29),
                -int(parent.metadata.get("rd_mat_count", 0)),
                -int(parent.metadata.get("raw_echo_file_size", 0)),
                int(parent.track_id),
            )
        )
        selected.extend(candidates[: int(per_label)])
    return selected


def build_set(
    *,
    project_root: Path,
    output_dir: Path,
    per_label: int | None,
    all_parents: bool = False,
    source_root: Path | None,
    overwrite: bool,
    skip_existing: bool = False,
    track_start: int | None = None,
    track_end: int | None = None,
) -> list[dict[str, Any]]:
    if not all_parents and (per_label is None or per_label <= 0):
        raise ValueError("per_label 必须为正数")
    if track_start is not None and track_start <= 0:
        raise ValueError("track_start 必须为正数")
    if track_end is not None and track_end <= 0:
        raise ValueError("track_end 必须为正数")
    if (
        track_start is not None
        and track_end is not None
        and track_start > track_end
    ):
        raise ValueError("track_start 不能大于 track_end")
    adapter = CQ08Adapter(
        project_root,
        source_root=source_root,
        verify_files=False,
    )
    parents = _select_parents(
        adapter.discover_parents(),
        per_label,
        all_parents=all_parents,
    )
    if track_start is not None or track_end is not None:
        lower = track_start if track_start is not None else 1
        upper = track_end if track_end is not None else 2**31 - 1
        parents = [
            parent
            for parent in parents
            if lower <= int(parent.track_id) <= upper
        ]
    if not parents:
        raise ValueError("没有可转换的 CQ08 parent")
    output_dir = output_dir if output_dir.is_absolute() else project_root / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for parent in parents:
        slug = LABEL_SLUGS[str(parent.label_name)]
        filename = f"cq08_track_{int(parent.track_id):04d}_{slug}.asd"
        target = output_dir / filename
        if target.exists() and skip_existing:
            continue
        if target.exists() and not overwrite:
            raise FileExistsError(f"输出已存在，请使用 --overwrite: {target}")
        results.append(_build_parent(adapter, parent, target))
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="使用 AeroSight AsdWriter 批量生成 CQ08 ASD")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("data/asd"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", dest="all_parents", action="store_true", help="转换全部 CQ08 parent")
    selection.add_argument("--per-label", type=int, default=4, help="每个类别转换的 parent 数量")
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--track-start", type=int, default=None, help="只处理大于等于该轨迹 ID 的 parent")
    parser.add_argument("--track-end", type=int, default=None, help="只处理小于等于该轨迹 ID 的 parent")
    write_mode = parser.add_mutually_exclusive_group()
    write_mode.add_argument("--overwrite", action="store_true")
    write_mode.add_argument("--skip-existing", action="store_true", help="已存在的 ASD 包直接跳过")
    args = parser.parse_args(argv)
    try:
        results = build_set(
            project_root=args.project_root.resolve(),
            output_dir=args.output_dir,
            per_label=None if args.all_parents else int(args.per_label),
            all_parents=bool(args.all_parents),
            source_root=args.source_root,
            overwrite=bool(args.overwrite),
            skip_existing=bool(args.skip_existing),
            track_start=args.track_start,
            track_end=args.track_end,
        )
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"ASD batch build failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"count": len(results), "packages": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
