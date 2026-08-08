from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from radar_ln3.features import (
    CausalTRObservationIncrementState,
    TR_OBSERVATION_INCREMENT_NAMES,
    parse_feature_json,
)
from report.gate_residual_features import RD_FEATURE_NAMES, _RDCache, _parse_indices, _rd_event_features


SCHEMA_VERSION = "RADAR-LN3-V10-gate-residual-manual-review-v1"
DEFAULT_INDEX = Path("data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-index.csv")
DEFAULT_DETAIL = Path(
    "artifacts/RADAR-LN3-V10-cq08-liquid-dbb-cu-gate-observation-increment-formal/"
    "sample_detail_information.json"
)
DEFAULT_INTEGRITY = Path(
    "src-v3/processed/report/RADAR-LN3-V10-gate-residual-integrity-audit.json"
)
DEFAULT_MANIFEST = Path("data/manifests/cq08_unified_manifest.json")
DEFAULT_OUTPUT = Path(
    "src-v3/processed/report/RADAR-LN3-V10-gate-residual-manual-review"
)
BASE_TR_FIELDS = (
    "range_m",
    "radial_speed_mps",
    "speed_mps",
    "course_deg",
    "azimuth_deg",
    "elevation_deg",
    "amplitude_db",
    "snr_db",
)
REVIEW_COLUMNS = (
    "parent_sample_id",
    "track_id",
    "current_label",
    "source_label_id",
    "V10_final_route",
    "V10_final_confidence",
    "event_count",
    "rd_coverage",
    "automatic_attention_flags",
    "review_decision",
    "reviewed_semantic_label",
    "reviewer",
    "review_date",
    "review_notes",
    "additional_evidence_required",
)
ALLOWED_REVIEW_DECISIONS = (
    "KEEP_CURRENT_LABEL",
    "SUSPECT_MISLABEL",
    "NEEDS_MORE_EVIDENCE",
    "EXCLUDE_FOR_DATA_CORRUPTION",
)


def _resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_path(root: Path, value: str) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _relative(path: Path, base: Path) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()


def _probability_cu(record: Mapping[str, Any]) -> float:
    pred_id = int(record["pred_id"])
    confidence = float(record["confidence"])
    if pred_id not in {0, 1} or not 0.0 <= confidence <= 1.0:
        raise ValueError("invalid Gate prediction record")
    return confidence if pred_id == 1 else 1.0 - confidence


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _automatic_flags(
    label: str,
    rd_fraction: float,
    probabilities: list[float],
    final_confidence: float,
) -> list[str]:
    flags: list[str] = []
    if rd_fraction < 1.0:
        flags.append("PARTIAL_RD_COVERAGE")
    if final_confidence >= 0.9:
        flags.append("HIGH_CONFIDENCE_ERROR")
    predictions = [int(value >= 0.5) for value in probabilities]
    if len(set(predictions)) == 1:
        flags.append("ROUTE_CONSTANT_FROM_FIRST_PREFIX")
    elif predictions[-1] != predictions[0]:
        flags.append("PREFIX_ROUTE_CHANGED")
    if label == "UnknownTarget":
        flags.append("UNKNOWN_CLOSED_SET_NOT_REJECT")
    return flags


def _plot_parent(frame: pd.DataFrame, path: Path, title: str) -> None:
    x = frame["window_length"].to_numpy(dtype=np.int64)
    figure, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True, constrained_layout=True)
    axes[0].plot(x, frame["probability_cu"], color="#8b1e3f", marker="o", markersize=3)
    axes[0].axhline(0.5, color="#555555", linestyle="--", linewidth=1)
    axes[0].set_ylabel("P(CU)")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, frame["speed_mps"], label="reported speed", color="#176b87")
    axes[1].plot(x, frame["radial_speed_mps"], label="radial speed", color="#d46b08")
    axes[1].set_ylabel("m/s")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.25)

    axes[2].plot(x, frame["course_deg"], label="course", color="#7251a2")
    axes[2].plot(x, frame["azimuth_deg"], label="azimuth", color="#2e7d32")
    axes[2].set_ylabel("degrees")
    axes[2].legend(loc="best")
    axes[2].grid(alpha=0.25)

    rd_present = frame["rd_present"].astype(bool).to_numpy()
    axes[3].plot(x, frame["rd_zero_velocity_energy_fraction"], label="zero-velocity energy", color="#9c2c2c")
    axes[3].plot(x, frame["rd_velocity_spectral_entropy"], label="spectral entropy", color="#005f73")
    axes[3].scatter(x[~rd_present], np.zeros((~rd_present).sum()), marker="x", color="black", label="RD missing")
    axes[3].set_ylabel("RD fraction")
    axes[3].set_xlabel("causal prefix length")
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].legend(loc="best")
    axes[3].grid(alpha=0.25)
    figure.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_rd_evidence(
    path: Path,
    title: str,
    event_profiles: list[np.ndarray | None],
    velocity_mps: np.ndarray | None,
    aggregate_power: np.ndarray | None,
    range_offset_m: np.ndarray | None,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
    if velocity_mps is None or aggregate_power is None or range_offset_m is None:
        for axis in axes:
            axis.text(0.5, 0.5, "No mapped RD frames", ha="center", va="center")
            axis.set_axis_off()
    else:
        profile_matrix = np.full((len(event_profiles), len(velocity_mps)), np.nan, dtype=np.float64)
        for event_index, profile in enumerate(event_profiles):
            if profile is None:
                continue
            normalized = np.asarray(profile, dtype=np.float64)
            normalized = normalized / max(float(normalized.max()), 1.0e-12)
            profile_matrix[event_index] = np.clip(10.0 * np.log10(np.maximum(normalized, 1.0e-6)), -40.0, 0.0)
        color_map = plt.colormaps["magma"].copy()
        color_map.set_bad("#d9d9d9")
        velocity_extent = [float(velocity_mps[0]), float(velocity_mps[-1]), len(event_profiles) + 0.5, 0.5]
        image = axes[0].imshow(
            profile_matrix,
            aspect="auto",
            interpolation="nearest",
            extent=velocity_extent,
            cmap=color_map,
            vmin=-40.0,
            vmax=0.0,
        )
        axes[0].axvline(0.0, color="white", linewidth=0.8, linestyle="--")
        axes[0].set_ylabel("causal prefix length")
        axes[0].set_xlabel("radial velocity (m/s)")
        axes[0].set_title("Per-event normalized velocity profile (grey = RD missing)")
        figure.colorbar(image, ax=axes[0], label="relative power (dB)")

        normalized_map = aggregate_power / max(float(aggregate_power.max()), 1.0e-12)
        map_db = np.clip(10.0 * np.log10(np.maximum(normalized_map, 1.0e-6)), -40.0, 0.0)
        rd_extent = [
            float(velocity_mps[0]),
            float(velocity_mps[-1]),
            float(range_offset_m[0]),
            float(range_offset_m[-1]),
        ]
        rd_image = axes[1].imshow(
            map_db,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=rd_extent,
            cmap="viridis",
            vmin=-40.0,
            vmax=0.0,
        )
        axes[1].axvline(0.0, color="white", linewidth=0.8, linestyle="--")
        axes[1].set_xlabel("radial velocity (m/s)")
        axes[1].set_ylabel("range offset (m)")
        axes[1].set_title("Aggregate mapped range-Doppler relative power")
        figure.colorbar(rd_image, ax=axes[1], label="relative power (dB)")
    figure.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _html_document(rows: list[dict[str, Any]], output_dir: Path) -> str:
    cards: list[str] = []
    for row in rows:
        flags = ", ".join(row["automatic_attention_flags"]) or "none"
        cards.append(
            "<article class='card'>"
            f"<h2>{html.escape(row['parent_sample_id'])}</h2>"
            f"<p><b>Current label:</b> {html.escape(row['current_label'])} &nbsp; "
            f"<b>V10 final route:</b> {row['V10_final_route']} &nbsp; "
            f"<b>confidence:</b> {row['V10_final_confidence']:.4f}</p>"
            f"<p><b>Events:</b> {row['event_count']} &nbsp; <b>RD:</b> "
            f"{row['rd_present_event_count']}/{row['event_count']} ({row['rd_coverage']})</p>"
            f"<p><b>Automatic attention flags:</b> {html.escape(flags)}</p>"
            f"<img src='{html.escape(row['plot_path'])}' alt='diagnostic plot'>"
            f"<img src='{html.escape(row['rd_plot_path'])}' alt='RD physical evidence'>"
            f"<p><a href='{html.escape(row['event_csv_path'])}'>event CSV</a> | "
            f"<a href='{html.escape(row['parent_json_path'])}'>parent JSON</a></p>"
            f"<details><summary>Source files</summary><pre>{html.escape(json.dumps(row['source_files'], ensure_ascii=False, indent=2))}</pre></details>"
            "</article>"
        )
    return """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>V10 Gate residual manual review</title>
<style>
body{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#f5f7fa;color:#17212b}
.notice{padding:14px;background:#fff3cd;border-left:5px solid #d39e00;margin-bottom:18px}
.card{background:white;padding:18px;margin:18px 0;border-radius:8px;box-shadow:0 1px 6px #ccd2d9}
img{max-width:100%;height:auto;border:1px solid #ddd} pre{white-space:pre-wrap;word-break:break-all}
a{color:#075985}
</style></head><body>
<h1>RADAR-LN3 V10 Gate Bird/Unknown residual manual review</h1>
<div class="notice"><b>重要：</b>本页面只汇总validation结构事实和描述性证据。模型预测、置信度、自动风险提示均不能作为改标签依据。请在manual-review.csv中由具备标签规范和采集背景的审核人填写决定。test未读取，开放集拒识未评估。</div>
<p>审核选项：KEEP_CURRENT_LABEL、SUSPECT_MISLABEL、NEEDS_MORE_EVIDENCE、EXCLUDE_FOR_DATA_CORRUPTION。KEEP_CURRENT_LABEL表示当前证据不足以推翻现标签，不等于独立证明标签客观正确；最后一项必须有独立的数据损坏证据，不能因分类困难而使用。</p>
""" + "\n".join(cards) + "\n</body></html>\n"


def run(
    project_root: Path,
    index_path: Path,
    detail_path: Path,
    integrity_path: Path,
    manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    existing_review = output_dir / "manual-review.csv"
    if existing_review.exists():
        existing = pd.read_csv(existing_review, keep_default_na=False, low_memory=False)
        human_columns = (
            "review_decision", "reviewed_semantic_label", "reviewer", "review_date",
            "review_notes", "additional_evidence_required",
        )
        if all(column in existing.columns for column in human_columns) and any(
            existing[column].astype(str).str.strip().ne("").any() for column in human_columns
        ):
            raise FileExistsError(
                "manual-review.csv contains human input; refusing to overwrite the review package"
            )
    detail = _load_json(detail_path)
    integrity = _load_json(integrity_path)
    if detail.get("split") != "val" or detail.get("task_name") != "dbb_cu_gate":
        raise ValueError("manual review requires V10 Gate validation detail")
    if integrity.get("status") != "passed" or not integrity.get("summary", {}).get("all_structural_checks_passed"):
        raise ValueError("manual review requires a passed residual integrity report")
    residual_parents = {str(row["parent_sample_id"]): dict(row) for row in integrity["parents"]}
    if len(residual_parents) != 12:
        raise ValueError("manual review requires exactly 12 V10 residual parents")

    window_records: dict[str, dict[int, dict[str, Any]]] = {parent: {} for parent in residual_parents}
    for raw in detail["records"]["window"]:
        parent = str(raw["parent_sample_id"])
        if parent in window_records:
            event = int(raw["event_index"])
            window_records[parent][event] = dict(raw)
    parent_records = {
        str(row["parent_sample_id"]): dict(row)
        for row in detail["records"]["parent"]
        if str(row["parent_sample_id"]) in residual_parents
    }
    if set(parent_records) != set(residual_parents):
        raise ValueError("parent predictions and residual integrity parent sets differ")

    columns = [
        "parent_sample_id", "track_id", "split_name", "label_name", "causal_event_index",
        "track_point_index", "tr_time_seconds", "tr_dt_seconds", "tr_dt_valid", "tr_dt_reason",
        "tr_missing_count", "tr_missing_fields_json", "tr_feature_values_json", "rd_present",
        "rd_frame_count", "rd_tensor_path", "rd_tensor_frame_indices_json", "rd_raw_frame_indices_json",
        "rd_parent_coverage_status", "track_csv_path", "cq08_track_file", "cq08_point_file",
        "cq08_raw_echo_file", "cq08_rd_mat_count",
    ]
    index = pd.read_csv(index_path, usecols=columns, low_memory=False)
    index = index[index["parent_sample_id"].astype(str).isin(residual_parents)].copy()
    index["causal_event_index"] = pd.to_numeric(index["causal_event_index"], errors="raise").astype(int)
    index = index.sort_values(["parent_sample_id", "causal_event_index"], kind="stable")

    manifest_payload = _load_json(manifest_path)
    track_ids = {str(row["track_id"]) for row in residual_parents.values()}
    manifests = {str(row["track_id"]): dict(row) for row in manifest_payload if str(row.get("track_id")) in track_ids}
    if set(manifests) != track_ids:
        raise ValueError("manual review manifest set is incomplete")

    output_dir.mkdir(parents=True, exist_ok=True)
    parents_dir = output_dir / "parents"
    plots_dir = output_dir / "plots"
    parents_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    cache = _RDCache(project_root)
    review_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    all_event_rows: list[dict[str, Any]] = []

    for parent_id in sorted(residual_parents, key=lambda value: int(value.rsplit("|", 1)[-1])):
        integrity_row = residual_parents[parent_id]
        track_id = str(integrity_row["track_id"])
        manifest_row = manifests[track_id]
        group = index[index["parent_sample_id"].astype(str).eq(parent_id)].copy()
        predictions = window_records[parent_id]
        if set(predictions) != set(group["causal_event_index"].tolist()):
            raise ValueError(f"prefix predictions do not cover every event: {parent_id}")
        state = CausalTRObservationIncrementState()
        previous_profile: np.ndarray | None = None
        event_profiles: list[np.ndarray | None] = []
        velocity_mps: np.ndarray | None = None
        range_offset_m: np.ndarray | None = None
        aggregate_power: np.ndarray | None = None
        event_rows: list[dict[str, Any]] = []
        probabilities: list[float] = []
        for row in group.itertuples(index=False):
            event_index = int(row.causal_event_index)
            raw_values = parse_feature_json(str(row.tr_feature_values_json))
            increment_values, increment_valid = state.update(raw_values)
            prediction = predictions[event_index]
            probability_cu = _probability_cu(prediction)
            probabilities.append(probability_cu)
            rd_values = np.zeros((len(RD_FEATURE_NAMES),), dtype=np.float64)
            rd_valid = np.zeros((len(RD_FEATURE_NAMES),), dtype=bool)
            rd_present = str(row.rd_present).strip().lower() in {"true", "1", "yes", "y"}
            if rd_present:
                arrays = cache.load(str(row.rd_tensor_path))
                indices = _parse_indices(row.rd_tensor_frame_indices_json, maximum=12)
                rd_values, rd_valid, profile = _rd_event_features(arrays, indices, previous_profile)
                previous_profile = profile
                event_profiles.append(profile.copy())
                if velocity_mps is None:
                    velocity_mps = np.asarray(arrays["velocity_mps"], dtype=np.float64)
                    tensor_path = _resolve(
                        project_root,
                        Path(str(row.rd_tensor_path).replace("\\", "/")),
                    )
                    with np.load(tensor_path, allow_pickle=False) as payload:
                        if "range_offset_m" not in payload.files:
                            raise ValueError(f"RD tensor lacks range_offset_m: {tensor_path}")
                        range_offset_m = np.asarray(payload["range_offset_m"], dtype=np.float64)
                    aggregate_power = np.zeros(arrays["amplitude"].shape[1:], dtype=np.float64)
                frame_amplitude = arrays["amplitude"][np.asarray(indices, dtype=np.int64)]
                relative_power = np.power(
                    10.0,
                    (frame_amplitude - frame_amplitude.max(axis=(1, 2), keepdims=True)) / 10.0,
                )
                if aggregate_power is None:
                    raise AssertionError("aggregate RD power was not initialized")
                aggregate_power += relative_power.sum(axis=0)
            else:
                previous_profile = None
                event_profiles.append(None)
            output: dict[str, Any] = {
                "parent_sample_id": parent_id,
                "track_id": int(track_id),
                "current_label": str(row.label_name),
                "event_index": event_index,
                "window_length": int(prediction["window_length"]),
                "track_point_index": int(row.track_point_index),
                "tr_time_seconds": float(row.tr_time_seconds),
                "tr_dt_seconds": float(row.tr_dt_seconds),
                "tr_dt_valid": str(row.tr_dt_valid).strip().lower() in {"true", "1", "yes", "y"},
                "tr_dt_reason": str(row.tr_dt_reason),
                "tr_missing_count": int(row.tr_missing_count),
                "tr_missing_fields_json": str(row.tr_missing_fields_json),
                "V10_pred_id": int(prediction["pred_id"]),
                "V10_route": "CU" if int(prediction["pred_id"]) == 1 else "DBB",
                "V10_confidence": float(prediction["confidence"]),
                "probability_cu": probability_cu,
                "rd_present": rd_present,
                "rd_frame_count": int(row.rd_frame_count),
                "rd_tensor_frame_indices_json": str(row.rd_tensor_frame_indices_json),
                "rd_raw_frame_indices_json": str(row.rd_raw_frame_indices_json),
            }
            for field in BASE_TR_FIELDS:
                output[field] = _finite_or_none(raw_values.get(field))
            for feature_index, feature_name in enumerate(TR_OBSERVATION_INCREMENT_NAMES):
                output[f"tr_{feature_name}"] = float(increment_values[feature_index])
                output[f"tr_{feature_name}_valid"] = bool(increment_valid[feature_index])
            for feature_index, feature_name in enumerate(RD_FEATURE_NAMES):
                output[f"rd_{feature_name}"] = (
                    float(rd_values[feature_index]) if rd_valid[feature_index] else None
                )
                output[f"rd_{feature_name}_valid"] = bool(rd_valid[feature_index])
            event_rows.append(output)
            all_event_rows.append(output)

        event_frame = pd.DataFrame(event_rows)
        if not np.isfinite(event_frame["probability_cu"].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"non-finite probability in manual review: {parent_id}")
        final_prediction = parent_records[parent_id]
        rd_fraction = float(event_frame["rd_present"].mean())
        flags = _automatic_flags(
            str(integrity_row["label_name"]), rd_fraction, probabilities, float(final_prediction["confidence"])
        )
        stem = f"track-{int(track_id):04d}-{str(integrity_row['label_name']).lower()}"
        event_csv = parents_dir / f"{stem}-events.csv"
        parent_json = parents_dir / f"{stem}-summary.json"
        plot_path = plots_dir / f"{stem}.png"
        rd_plot_path = plots_dir / f"{stem}-rd.png"
        event_frame.to_csv(event_csv, index=False, encoding="utf-8-sig")
        _plot_parent(event_frame, plot_path, f"{parent_id} | {integrity_row['label_name']} -> {'CU' if int(final_prediction['pred_id']) == 1 else 'DBB'}")
        _plot_rd_evidence(
            rd_plot_path,
            f"{parent_id} | mapped RD evidence",
            event_profiles,
            velocity_mps,
            aggregate_power,
            range_offset_m,
        )

        source_files = {
            "track_csv": str(_resolve(project_root, Path(str(group.iloc[0]["track_csv_path"]).replace("\\", "/")))),
            "cq08_track_file": str(_manifest_path(project_root, str(manifest_row["track_file"]))),
            "cq08_point_file": str(_manifest_path(project_root, str(manifest_row["point_file"]))),
            "cq08_raw_echo_file": str(_manifest_path(project_root, str(manifest_row["raw_echo_file"]))),
            "rd_tensor": str(_resolve(project_root, Path(str(group[group["rd_present"].astype(str).str.lower().isin({"true", "1", "yes", "y"})].iloc[0]["rd_tensor_path"]).replace("\\", "/")))) if bool(event_frame["rd_present"].any()) else "",
            "rd_mat_files": [str(_manifest_path(project_root, str(value))) for value in manifest_row.get("rd_mat_files", [])],
        }
        summary = {
            "schema_version": SCHEMA_VERSION,
            "parent_sample_id": parent_id,
            "track_id": int(track_id),
            "session_id": str(manifest_row.get("session_id", "")),
            "capture_id": str(manifest_row.get("capture_id", "")),
            "current_label": str(integrity_row["label_name"]),
            "source_label_id": int(integrity_row["source_label_id"]),
            "V10_final_route": "CU" if int(final_prediction["pred_id"]) == 1 else "DBB",
            "V10_final_confidence": float(final_prediction["confidence"]),
            "event_count": len(event_frame),
            "rd_present_event_count": int(event_frame["rd_present"].sum()),
            "rd_coverage_fraction": rd_fraction,
            "rd_coverage": str(integrity_row["rd_coverage"]),
            "probability_cu_minimum": float(event_frame["probability_cu"].min()),
            "probability_cu_median": float(event_frame["probability_cu"].median()),
            "probability_cu_maximum": float(event_frame["probability_cu"].max()),
            "automatic_attention_flags": flags,
            "automatic_interpretation_limit": "flags are review prompts, not label decisions",
            "source_files": source_files,
            "event_csv": str(event_csv),
            "plot": str(plot_path),
            "rd_plot": str(rd_plot_path),
            "manual_review": {
                "review_decision": "",
                "reviewed_semantic_label": "",
                "reviewer": "",
                "review_date": "",
                "review_notes": "",
                "additional_evidence_required": "",
            },
        }
        parent_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        index_rows.append({
            **{key: value for key, value in summary.items() if key not in {"source_files", "manual_review"}},
            "automatic_attention_flags": ";".join(flags),
            "plot_path": _relative(plot_path, output_dir),
            "rd_plot_path": _relative(rd_plot_path, output_dir),
            "event_csv_path": _relative(event_csv, output_dir),
            "parent_json_path": _relative(parent_json, output_dir),
            "source_files": source_files,
        })
        review_rows.append({
            "parent_sample_id": parent_id,
            "track_id": int(track_id),
            "current_label": str(integrity_row["label_name"]),
            "source_label_id": int(integrity_row["source_label_id"]),
            "V10_final_route": summary["V10_final_route"],
            "V10_final_confidence": float(final_prediction["confidence"]),
            "event_count": len(event_frame),
            "rd_coverage": str(integrity_row["rd_coverage"]),
            "automatic_attention_flags": ";".join(flags),
            "review_decision": "",
            "reviewed_semantic_label": "",
            "reviewer": "",
            "review_date": "",
            "review_notes": "",
            "additional_evidence_required": "",
        })

    index_csv = output_dir / "parent-index.csv"
    events_csv = output_dir / "all-events.csv"
    review_csv = output_dir / "manual-review.csv"
    html_path = output_dir / "index.html"
    summary_path = output_dir / "summary.json"
    pd.DataFrame(index_rows).drop(columns=["source_files"]).to_csv(index_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(all_event_rows).to_csv(events_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(review_rows, columns=REVIEW_COLUMNS).to_csv(review_csv, index=False, encoding="utf-8-sig")
    html_path.write_text(_html_document(index_rows, output_dir), encoding="utf-8")
    readme_path = output_dir / "README.md"
    readme_path.write_text(
        "# RADAR-LN3 V10 Gate Residual Manual Review\n\n"
        "范围：CQ08 V3 validation的12个V10最终Gate错误parent；test、训练、阈值和拒识均未执行。\n\n"
        "1. 打开`index.html`，依次查看TR/prefix图和RD物理证据图。\n"
        "2. 必要时查看对应event CSV与源航迹、点迹、原始回波和MAT文件。\n"
        "3. 在`manual-review.csv`填写审核决定；不要修改自动事实列。\n"
        "4. `KEEP_CURRENT_LABEL`表示当前证据不足以推翻现标签，但不等于独立证明标签客观正确；`SUSPECT_MISLABEL`表示有独立标注证据支持疑似误标；"
        "`NEEDS_MORE_EVIDENCE`表示无法裁决；`EXCLUDE_FOR_DATA_CORRUPTION`只能用于已证实的数据损坏，不能用于困难样本。\n\n"
        "模型预测和置信度不能作为改标签依据。UnknownTarget闭集标签不等同于开放集reject/abstain。\n",
        encoding="utf-8",
    )

    generated_files = sorted(
        [path for path in output_dir.rglob("*") if path.is_file() and path != summary_path],
        key=lambda path: path.as_posix(),
    )
    result = {
        "status": "passed",
        "schema_version": SCHEMA_VERSION,
        "scope": "12 V10 final Gate residual parents from CQ08 V3 validation only; test excluded",
        "parent_count": len(index_rows),
        "event_count": len(all_event_rows),
        "label_counts": {
            str(name): int(count)
            for name, count in pd.Series(
                [row["current_label"] for row in review_rows]
            ).value_counts().sort_index().items()
        },
        "review_status": "awaiting_human_review",
        "review_decisions_completed": 0,
        "human_observations_recorded": 0,
        "allowed_review_decisions": list(ALLOWED_REVIEW_DECISIONS),
        "automatic_label_changes": 0,
        "outputs": {
            "html": str(html_path),
            "parent_index": str(index_csv),
            "all_events": str(events_csv),
            "manual_review": str(review_csv),
            "readme": str(readme_path),
            "parent_directory": str(parents_dir),
            "plot_directory": str(plots_dir),
        },
        "inputs": {
            "index": {"path": str(index_path), "sha256": _sha256(index_path)},
            "detail": {"path": str(detail_path), "sha256": _sha256(detail_path)},
            "integrity": {"path": str(integrity_path), "sha256": _sha256(integrity_path)},
            "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        },
        "generated_files": [
            {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in generated_files
        ],
        "interpretation_limits": [
            "model predictions and confidence are not label evidence",
            "automatic attention flags are prompts for human review only",
            "review decisions are intentionally blank; review_notes may later record non-decisional human observations",
            "UnknownTarget is a closed-set label here and is not equivalent to reject/abstain",
            "no data, label, config or checkpoint was changed",
            "test, threshold fitting, training, stable, formal cascade and P95 were not executed",
        ],
        "v11_release": "blocked_pending_human_review",
        "test": "未执行",
        "training": "未执行",
        "output_path": str(summary_path),
    }
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the V10 Bird/Unknown residual manual-review package.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--detail", type=Path, default=DEFAULT_DETAIL)
    parser.add_argument("--integrity", type=Path, default=DEFAULT_INTEGRITY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = run(
        root,
        _resolve(root, args.index),
        _resolve(root, args.detail),
        _resolve(root, args.integrity),
        _resolve(root, args.manifest),
        _resolve(root, args.output_dir),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
