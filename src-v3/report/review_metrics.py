from __future__ import annotations

"""Build descriptive TR/RD comparison material for the V10 residual review.

This module is deliberately diagnostic-only.  It uses CQ08 train parent
medians to describe a reference distribution and never fits a classifier,
threshold, label change, or deployment rule.
"""

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

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
from report.gate_residual_features import (
    RD_FEATURE_NAMES,
    _RDCache,
    _parse_indices,
    _rd_event_features,
)


SCHEMA_VERSION = "RADAR-LN3-V10-gate-residual-review-metrics-v1"
DEFAULT_INDEX = Path("data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-index.csv")
DEFAULT_REVIEW_DIR = Path(
    "src-v3/processed/report/RADAR-LN3-V10-gate-residual-manual-review"
)

TR_METRICS = tuple(f"tr_{name}" for name in TR_OBSERVATION_INCREMENT_NAMES)
RD_METRICS = tuple(f"rd_{name}" for name in RD_FEATURE_NAMES)
MAIN_METRICS = TR_METRICS + RD_METRICS
QUALITY_METRICS = (
    "event_count",
    "sequence_duration_seconds",
    "rd_present_fraction",
    "tr_dt_valid_fraction",
    "tr_valid_feature_fraction",
    "rd_valid_feature_fraction",
    "tr_missing_fraction",
    "rd_frame_count_median",
)
ALL_METRICS = MAIN_METRICS + QUALITY_METRICS
BASE_TR_METRICS = (
    "range_m_median",
    "radial_speed_mps_median",
    "speed_mps_median",
    "snr_db_median",
)

DISPLAY_NAMES = {
    "tr_reported_speed_delta_scaled": "TR speed delta (scaled)",
    "tr_radial_speed_delta_scaled": "TR radial-speed delta (scaled)",
    "tr_course_delta_scaled": "TR course delta (scaled)",
    "tr_geometric_step_distance_scaled": "TR geometric step (scaled)",
    "tr_range_delta_scaled": "TR range delta (scaled)",
    "tr_radial_range_direction_agreement": "TR radial/range agreement",
    "tr_radial_sign_flip": "TR radial sign flip",
    "tr_consecutive_step_direction_cosine": "TR step direction cosine",
    "tr_causal_path_straightness": "TR causal path straightness",
    "rd_velocity_bandwidth_mps": "RD velocity bandwidth (m/s)",
    "rd_zero_velocity_energy_fraction": "RD zero-velocity energy",
    "rd_velocity_spectral_entropy": "RD velocity spectral entropy",
    "rd_top10_velocity_energy_fraction": "RD top-10 velocity energy",
    "rd_abs_peak_velocity_mps": "RD peak velocity (m/s)",
    "rd_phase_coherence": "RD phase coherence",
    "rd_velocity_profile_change": "RD profile change",
    "event_count": "Event count",
    "sequence_duration_seconds": "TR sequence duration (s)",
    "rd_present_fraction": "RD coverage fraction",
    "tr_dt_valid_fraction": "Valid TR-dt fraction",
    "tr_valid_feature_fraction": "Valid TR-feature fraction",
    "rd_valid_feature_fraction": "Valid RD-feature fraction",
    "tr_missing_fraction": "TR missing-field fraction",
    "rd_frame_count_median": "Median RD frames/event",
    "range_m_median": "Median range (m)",
    "radial_speed_mps_median": "Median radial speed (m/s)",
    "speed_mps_median": "Median reported speed (m/s)",
    "snr_db_median": "Median SNR (dB)",
}

UNITS = {
    "tr_geometric_step_distance_scaled": "dimensionless scaled by 100 m",
    "tr_range_delta_scaled": "dimensionless scaled by 100 m",
    "tr_reported_speed_delta_scaled": "dimensionless scaled by 10 m/s",
    "tr_radial_speed_delta_scaled": "dimensionless scaled by 10 m/s",
    "tr_course_delta_scaled": "dimensionless scaled by 45 deg",
    "tr_radial_range_direction_agreement": "dimensionless {-1, 1} diagnostic",
    "tr_radial_sign_flip": "dimensionless flag",
    "tr_consecutive_step_direction_cosine": "dimensionless cosine",
    "tr_causal_path_straightness": "dimensionless [0, 1]",
    "rd_velocity_bandwidth_mps": "m/s",
    "rd_zero_velocity_energy_fraction": "fraction",
    "rd_velocity_spectral_entropy": "normalized fraction",
    "rd_top10_velocity_energy_fraction": "fraction",
    "rd_abs_peak_velocity_mps": "m/s",
    "rd_phase_coherence": "fraction",
    "rd_velocity_profile_change": "total-variation fraction",
    "event_count": "events",
    "sequence_duration_seconds": "s from TR timestamps",
    "rd_present_fraction": "fraction",
    "tr_dt_valid_fraction": "fraction",
    "tr_valid_feature_fraction": "fraction",
    "rd_valid_feature_fraction": "fraction among mapped RD events",
    "tr_missing_fraction": "fraction of 11 raw TR fields",
    "rd_frame_count_median": "frames/event",
    "range_m_median": "m",
    "radial_speed_mps_median": "m/s",
    "speed_mps_median": "m/s",
    "snr_db_median": "dB",
}

REQUIRED_INDEX_COLUMNS = {
    "source_name",
    "split_name",
    "parent_sample_id",
    "label_name",
    "causal_event_index",
    "tr_feature_values_json",
    "tr_dt_seconds",
    "tr_dt_valid",
    "tr_missing_count",
    "rd_present",
    "rd_frame_count",
    "rd_tensor_path",
    "rd_tensor_frame_indices_json",
}


def _resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _median(values: Iterable[Any]) -> float | None:
    clean = [number for value in values if (number := _finite(value)) is not None]
    return float(np.median(np.asarray(clean, dtype=np.float64))) if clean else None


def _mean(values: Iterable[Any]) -> float | None:
    clean = [number for value in values if (number := _finite(value)) is not None]
    return float(np.mean(np.asarray(clean, dtype=np.float64))) if clean else None


def _metric_median(frame: pd.DataFrame, name: str) -> float | None:
    value_column = name
    valid_column = f"{name}_valid"
    if value_column not in frame.columns:
        return None
    values = pd.to_numeric(frame[value_column], errors="coerce")
    if valid_column in frame.columns:
        valid = frame[valid_column].map(_bool).to_numpy(dtype=bool)
        values = values[valid]
    values = values[np.isfinite(values.to_numpy(dtype=np.float64))]
    return float(values.median()) if len(values) else None


def _event_summary(event_frame: pd.DataFrame, label: str, parent_id: str) -> dict[str, Any]:
    if event_frame.empty:
        raise ValueError(f"empty event frame for {parent_id}")
    row: dict[str, Any] = {
        "parent_sample_id": parent_id,
        "label_name": label,
        "event_count": int(len(event_frame)),
        "rd_present_fraction": float(event_frame["rd_present"].map(_bool).mean()),
        "tr_dt_valid_fraction": float(event_frame["tr_dt_valid"].map(_bool).mean()),
        "tr_valid_feature_fraction": float(
            event_frame[
                [f"{name}_valid" for name in TR_METRICS]
            ].map(_bool).to_numpy(dtype=np.float64).mean()
        ),
        "rd_valid_feature_fraction": None,
        "tr_missing_fraction": float(
            pd.to_numeric(event_frame["tr_missing_count"], errors="coerce").fillna(0.0).mean()
            / 11.0
        ),
        "rd_frame_count_median": _median(event_frame["rd_frame_count"]),
    }
    times = pd.to_numeric(event_frame["tr_time_seconds"], errors="coerce").dropna()
    row["sequence_duration_seconds"] = (
        float(times.max() - times.min()) if len(times) else None
    )
    rd_present = event_frame["rd_present"].map(_bool).to_numpy(dtype=bool)
    if rd_present.any():
        rd_valid = event_frame.loc[
            rd_present, [f"{name}_valid" for name in RD_METRICS]
        ].map(_bool).to_numpy(dtype=np.float64)
        row["rd_valid_feature_fraction"] = float(rd_valid.mean())
    for name in MAIN_METRICS:
        row[name] = _metric_median(event_frame, name)
    for name in BASE_TR_METRICS:
        row[name] = _median(event_frame[name.removesuffix("_median")])
    return row


def _training_event_frame(
    group: pd.DataFrame,
    cache: _RDCache,
) -> pd.DataFrame:
    state = CausalTRObservationIncrementState()
    previous_profile: np.ndarray | None = None
    rows: list[dict[str, Any]] = []
    for item in group.sort_values("causal_event_index", kind="stable").itertuples(index=False):
        raw = parse_feature_json(str(item.tr_feature_values_json))
        tr_values, tr_valid = state.update(raw)
        output: dict[str, Any] = {
            "tr_dt_seconds": _finite(item.tr_dt_seconds),
            "tr_dt_valid": _bool(item.tr_dt_valid),
            "tr_missing_count": int(float(item.tr_missing_count)),
            "rd_present": _bool(item.rd_present),
            "rd_frame_count": int(float(item.rd_frame_count)),
            "tr_time_seconds": _finite(item.tr_time_seconds),
        }
        for index, name in enumerate(TR_METRICS):
            output[name] = float(tr_values[index])
            output[f"{name}_valid"] = bool(tr_valid[index])
        for name in ("range_m", "radial_speed_mps", "speed_mps", "snr_db"):
            output[name] = _finite(raw.get(name))
        rd_values = np.zeros((len(RD_FEATURE_NAMES),), dtype=np.float64)
        rd_valid = np.zeros((len(RD_FEATURE_NAMES),), dtype=bool)
        if output["rd_present"]:
            arrays = cache.load(str(item.rd_tensor_path))
            indices = _parse_indices(item.rd_tensor_frame_indices_json, maximum=12)
            rd_values, rd_valid, previous_profile = _rd_event_features(
                arrays, indices, previous_profile
            )
        else:
            previous_profile = None
        for index, name in enumerate(RD_METRICS):
            output[name] = float(rd_values[index]) if rd_valid[index] else None
            output[f"{name}_valid"] = bool(rd_valid[index])
        rows.append(output)
    return pd.DataFrame(rows)


def _load_training_parent_metrics(project_root: Path, index_path: Path) -> pd.DataFrame:
    header = pd.read_csv(index_path, nrows=0).columns
    missing = sorted(REQUIRED_INDEX_COLUMNS - set(header))
    if missing:
        raise ValueError(f"V3 index lacks required columns: {missing}")
    usecols = sorted(REQUIRED_INDEX_COLUMNS | {"tr_time_seconds"})
    frame = pd.read_csv(index_path, usecols=usecols, low_memory=False)
    if frame["source_name"].astype(str).ne("CQ08").any():
        raise ValueError("review reference contains a non-CQ08 source")
    frame = frame[frame["split_name"].astype(str).eq("train")].copy()
    if frame.empty:
        raise ValueError("CQ08 train reference is empty")
    frame["causal_event_index"] = pd.to_numeric(
        frame["causal_event_index"], errors="raise"
    ).astype(int)
    if frame.groupby("parent_sample_id")["label_name"].nunique().gt(1).any():
        raise ValueError("CQ08 train parent has multiple labels")
    if frame.groupby("parent_sample_id")["split_name"].nunique().gt(1).any():
        raise ValueError("CQ08 train parent crosses splits")
    cache = _RDCache(project_root)
    rows: list[dict[str, Any]] = []
    for parent_id, group in frame.groupby("parent_sample_id", sort=True):
        event_frame = _training_event_frame(group, cache)
        rows.append(_event_summary(event_frame, str(group["label_name"].iloc[0]), str(parent_id)))
    result = pd.DataFrame(rows)
    if result["parent_sample_id"].duplicated().any():
        raise AssertionError("training parent metrics are duplicated")
    return result


def _load_residual_metrics(review_dir: Path) -> pd.DataFrame:
    event_path = review_dir / "all-events.csv"
    parent_path = review_dir / "parent-index.csv"
    if not event_path.exists() or not parent_path.exists():
        raise FileNotFoundError("Stage 038 review package is incomplete")
    events = pd.read_csv(event_path, low_memory=False)
    parents = pd.read_csv(parent_path, low_memory=False)
    events["event_index"] = pd.to_numeric(events["event_index"], errors="raise").astype(int)
    events = events.sort_values(["parent_sample_id", "event_index"], kind="stable")
    rows: list[dict[str, Any]] = []
    for parent_id, group in events.groupby("parent_sample_id", sort=False):
        label = str(group["current_label"].iloc[0])
        row = _event_summary(group, label, str(parent_id))
        parent_row = parents[parents["parent_sample_id"].astype(str).eq(str(parent_id))]
        if len(parent_row) != 1:
            raise ValueError(f"residual parent index mismatch: {parent_id}")
        source = parent_row.iloc[0]
        route = str(source["V10_final_route"])
        row.update(
            {
                "track_id": int(source["track_id"]),
                "V10_final_route": route,
                "V10_final_confidence": float(source["V10_final_confidence"]),
                "rd_coverage": str(source["rd_coverage"]),
                "prediction_status": (
                    "wrong"
                    if (label == "BirdTarget" and route == "CU")
                    or (label == "UnknownTarget" and route == "DBB")
                    else "correct"
                ),
            }
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != 12 or set(result["label_name"]) != {"BirdTarget", "UnknownTarget"}:
        raise ValueError("residual review package must contain 12 Bird/Unknown parents")
    if result["prediction_status"].ne("wrong").any():
        raise ValueError("review package contains a non-residual parent")
    return result


def _percentile(value: float, reference: np.ndarray) -> float:
    less = float(np.sum(reference < value))
    equal = float(np.sum(reference == value))
    return 100.0 * (less + 0.5 * equal) / float(len(reference))


def _reference_table(train: pd.DataFrame, labels: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label in sorted(set(labels)):
        subset = train[train["label_name"].astype(str).eq(label)]
        if subset.empty:
            raise ValueError(f"missing CQ08 train label reference: {label}")
        for metric in ALL_METRICS + BASE_TR_METRICS:
            values = pd.to_numeric(subset[metric], errors="coerce").dropna().to_numpy(dtype=np.float64)
            if not len(values):
                continue
            rows.append(
                {
                    "reference_label": label,
                    "metric": metric,
                    "metric_display": DISPLAY_NAMES.get(metric, metric),
                    "unit": UNITS.get(metric, "dimensionless"),
                    "parent_count": int(len(values)),
                    "p05": float(np.quantile(values, 0.05)),
                    "q1": float(np.quantile(values, 0.25)),
                    "median": float(np.quantile(values, 0.50)),
                    "q3": float(np.quantile(values, 0.75)),
                    "p95": float(np.quantile(values, 0.95)),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def _percentile_table(
    residual: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in residual.itertuples(index=False):
        label = str(item.label_name)
        for metric in ALL_METRICS + BASE_TR_METRICS:
            value = _finite(getattr(item, metric))
            stats = reference[
                reference["reference_label"].eq(label) & reference["metric"].eq(metric)
            ]
            if value is None or len(stats) != 1:
                continue
            stat = stats.iloc[0]
            train_values = pd.to_numeric(
                reference.loc[
                    reference["reference_label"].eq(label)
                    & reference["metric"].eq(metric),
                    "median",
                ],
                errors="coerce",
            )
            # The empirical percentile needs the parent-level values, which are
            # attached by the caller in ``_add_empirical_percentiles``.
            rows.append(
                {
                    "parent_sample_id": str(item.parent_sample_id),
                    "track_id": int(item.track_id),
                    "label_name": label,
                    "V10_final_route": str(item.V10_final_route),
                    "metric": metric,
                    "metric_display": DISPLAY_NAMES.get(metric, metric),
                    "value": value,
                    "reference_label": label,
                    "reference_parent_count": int(stat["parent_count"]),
                    "reference_p05": float(stat["p05"]),
                    "reference_q1": float(stat["q1"]),
                    "reference_median": float(stat["median"]),
                    "reference_q3": float(stat["q3"]),
                    "reference_p95": float(stat["p95"]),
                    "unit": UNITS.get(metric, "dimensionless"),
                }
            )
    return pd.DataFrame(rows)


def _add_empirical_percentiles(
    table: pd.DataFrame,
    train: pd.DataFrame,
) -> pd.DataFrame:
    if table.empty:
        return table
    output = table.copy()
    percentiles: list[float] = []
    for item in output.itertuples(index=False):
        values = pd.to_numeric(
            train.loc[
                train["label_name"].astype(str).eq(str(item.reference_label)),
                str(item.metric),
            ],
            errors="coerce",
        ).dropna().to_numpy(dtype=np.float64)
        if not len(values):
            percentiles.append(float("nan"))
        else:
            percentiles.append(_percentile(float(item.value), values))
    output["empirical_percentile"] = percentiles
    output["within_iqr"] = output["value"].between(
        output["reference_q1"], output["reference_q3"], inclusive="both"
    )
    output["within_p05_p95"] = output["value"].between(
        output["reference_p05"], output["reference_p95"], inclusive="both"
    )
    output["outlier_direction"] = np.select(
        [output["value"] < output["reference_p05"], output["value"] > output["reference_p95"]],
        ["LOW", "HIGH"],
        default="INSIDE_P05_P95",
    )
    return output


def _plot_box_reference(
    train: pd.DataFrame,
    residual: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(4, 4, figsize=(22, 17), constrained_layout=True)
    colors = {"BirdTarget": "#176b87", "UnknownTarget": "#9c2c2c"}
    for axis, metric in zip(axes.flat, MAIN_METRICS):
        groups: list[np.ndarray] = []
        labels: list[str] = []
        positions: list[int] = []
        for position, label in enumerate(("BirdTarget", "UnknownTarget"), start=1):
            values = pd.to_numeric(
                train.loc[train["label_name"].eq(label), metric], errors="coerce"
            ).dropna().to_numpy(dtype=np.float64)
            groups.append(values)
            labels.append("Bird" if label.startswith("Bird") else "Unknown")
            positions.append(position)
        box = axis.boxplot(
            groups,
            positions=positions,
            widths=0.45,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),
        )
        for patch, label in zip(box["boxes"], ("BirdTarget", "UnknownTarget")):
            patch.set_facecolor(colors[label])
            patch.set_alpha(0.25)
        for label, position in zip(("BirdTarget", "UnknownTarget"), positions):
            points = residual[residual["label_name"].eq(label)]
            y = pd.to_numeric(points[metric], errors="coerce")
            valid = y.notna()
            jitter = np.linspace(-0.12, 0.12, int(valid.sum())) if valid.sum() else []
            axis.scatter(
                np.asarray(position, dtype=np.float64) + np.asarray(jitter),
                y[valid].to_numpy(dtype=np.float64),
                s=28,
                color=colors[label],
                edgecolor="black",
                linewidth=0.35,
                zorder=4,
                label="residual" if position == 1 else None,
            )
        axis.set_title(DISPLAY_NAMES.get(metric, metric), fontsize=9)
        axis.set_xticks(positions, labels, fontsize=8)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="y", labelsize=8)
    figure.suptitle(
        "V10 residual parent medians vs CQ08 train same-label parent reference\n"
        "box whiskers=P05/P95; points=12 residuals; descriptive only",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def _plot_percentile_heatmap(table: pd.DataFrame, output_path: Path) -> None:
    row_order = (
        table[["parent_sample_id", "track_id", "label_name"]]
        .drop_duplicates()
        .sort_values(["label_name", "track_id"])
    )
    matrix = np.full((len(row_order), len(MAIN_METRICS)), np.nan, dtype=np.float64)
    row_index = {str(row.parent_sample_id): index for index, row in enumerate(row_order.itertuples())}
    col_index = {name: index for index, name in enumerate(MAIN_METRICS)}
    for item in table.itertuples(index=False):
        if item.metric not in col_index:
            continue
        matrix[row_index[str(item.parent_sample_id)], col_index[item.metric]] = float(
            item.empirical_percentile
        )
    figure, axis = plt.subplots(figsize=(22, 9), constrained_layout=True)
    masked = np.ma.masked_invalid(matrix)
    image = axis.imshow(masked, aspect="auto", cmap="coolwarm", vmin=0.0, vmax=100.0)
    axis.set_xticks(
        np.arange(len(MAIN_METRICS)),
        [DISPLAY_NAMES[name].replace(" (scaled)", "") for name in MAIN_METRICS],
        rotation=65,
        ha="right",
        fontsize=8,
    )
    axis.set_yticks(
        np.arange(len(row_order)),
        [f"{int(row.track_id):04d} {row.label_name.replace('Target', '')}" for row in row_order.itertuples()],
        fontsize=8,
    )
    for row_index_value in range(matrix.shape[0]):
        for col_index_value in range(matrix.shape[1]):
            value = matrix[row_index_value, col_index_value]
            if np.isfinite(value):
                axis.text(
                    col_index_value,
                    row_index_value,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="black" if 25 <= value <= 75 else "white",
                )
    axis.axvline(len(TR_METRICS) - 0.5, color="black", linewidth=1.5)
    axis.set_title(
        "V10 residual empirical percentile in same-label CQ08 train parent distribution\n"
        "0=low tail, 50=median, 100=high tail; this is not a label threshold",
        fontsize=13,
    )
    figure.colorbar(image, ax=axis, label="empirical percentile")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_quality(residual: pd.DataFrame, reference: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    metrics = (
        "rd_present_fraction",
        "tr_dt_valid_fraction",
        "event_count",
        "sequence_duration_seconds",
    )
    colors = {"BirdTarget": "#176b87", "UnknownTarget": "#9c2c2c"}
    for axis, metric in zip(axes.flat, metrics):
        row_positions = np.arange(len(residual))
        values = pd.to_numeric(residual[metric], errors="coerce").to_numpy(dtype=np.float64)
        labels = [f"{int(row.track_id):04d}" for row in residual.itertuples()]
        bar_colors = [colors[str(label)] for label in residual["label_name"]]
        axis.barh(row_positions, values, color=bar_colors, alpha=0.75)
        for label in ("BirdTarget", "UnknownTarget"):
            stats = reference[reference["reference_label"].eq(label) & reference["metric"].eq(metric)]
            if len(stats) == 1:
                stat = stats.iloc[0]
                axis.axvspan(
                    float(stat["p05"]),
                    float(stat["p95"]),
                    color=colors[label],
                    alpha=0.08,
                    label=f"{label.replace('Target', '')} train P05-P95",
                )
        axis.set_yticks(row_positions, labels)
        axis.set_title(DISPLAY_NAMES[metric])
        axis.grid(axis="x", alpha=0.25)
        axis.legend(fontsize=7, loc="best")
    figure.suptitle(
        "Residual data-quality and sequence coverage comparison\n"
        "bars=12 residuals; shaded bands=same-label CQ08 train P05-P95",
        fontsize=13,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def _plot_parent(
    parent_id: str,
    label: str,
    route: str,
    table: pd.DataFrame,
    output_path: Path,
) -> None:
    subset = table[
        table["parent_sample_id"].astype(str).eq(parent_id)
        & table["metric"].isin(MAIN_METRICS)
    ].copy()
    ordered = [metric for metric in MAIN_METRICS if metric in set(subset["metric"])]
    values = [float(subset.loc[subset["metric"].eq(metric), "empirical_percentile"].iloc[0]) for metric in ordered]
    names = [DISPLAY_NAMES[metric].replace("TR ", "").replace("RD ", "") for metric in ordered]
    figure, axis = plt.subplots(figsize=(14, 8), constrained_layout=True)
    y = np.arange(len(ordered))
    axis.axvspan(5, 95, color="#d9e2ec", alpha=0.75, label="train P05-P95")
    axis.axvspan(25, 75, color="#9fbad0", alpha=0.6, label="train Q1-Q3")
    axis.axvline(50, color="#555555", linestyle="--", linewidth=1.0, label="train median")
    axis.scatter(values, y, color="#9c2c2c" if label == "UnknownTarget" else "#176b87", s=55, zorder=3)
    axis.set_xlim(0, 100)
    axis.set_yticks(y, names, fontsize=9)
    axis.set_xlabel("same-label CQ08 train empirical percentile")
    axis.set_title(
        f"{parent_id} | label={label} | V10 route={route}\n"
        "point outside the band is a comparison prompt, not a semantic verdict"
    )
    axis.grid(axis="x", alpha=0.25)
    axis.legend(loc="lower right", fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def _fmt(value: Any) -> str:
    number = _finite(value)
    if number is None:
        return "未统计"
    if abs(number) >= 100:
        return f"{number:.2f}"
    return f"{number:.4f}"


def _write_guide(
    output_path: Path,
    train: pd.DataFrame,
    residual: pd.DataFrame,
    reference: pd.DataFrame,
) -> None:
    lines = [
        "# RADAR-LN3 V10残差TR/RD指标人工比对指南",
        "",
        "范围：CQ08 V3 validation中的12个V10 Gate残差parent；参考分布为同一V3索引、同一CQ08 source、同一train split的parent级中位数。test未读取，未拟合阈值、分类器、拒识规则或标签修改。",
        "",
        "## 先看什么",
        "",
        "1. 先核对源文件和标签码：航迹/点迹/raw文件是否存在，文件名Label编码、manifest标签和当前标签是否一致。结构异常才可能支持数据损坏或标签冲突；模型路由不属于标签证据。",
        "2. 再看覆盖和时间：`rd_present_fraction`、事件数、序列时长、`tr_dt_valid_fraction`。track 333的RD部分覆盖必须和其源MAT/RD映射一起解释，不能拿RD缺失当作Unknown语义。",
        "3. 再看TR：步长、range delta、方向余弦、路径直线度和角度增量。运动明显只能说明运动形态，不能单独证明DBB；运动弱也不能单独证明Clutter或Unknown。",
        "4. 再看RD：速度带宽、零速能量、谱熵、峰值速度、相位一致性和profile change。RD指标必须结合速度轴、映射帧和覆盖查看；单一RD阈值不支持Unknown裁决。",
        "5. 最后看V10 P(CU)/路由曲线。它只用于定位模型残差模式，不用于改标签；Unknown当前是闭集标签，不等同于开放集reject/abstain。",
        "",
        "## 区间怎么用",
        "",
        "- `Q1-Q3`是同标签CQ08 train parent中位数的典型中间50%范围。",
        "- `P05-P95`是同标签CQ08 train parent中位数的描述性90%范围；超出它只表示该parent与训练参考分布不同，不表示误标、数据损坏或开放集。",
        "- 百分位图中50接近训练中位数，0/100接近低/高尾部；多个相互独立的结构、TR和RD证据一致时才值得升级为`NEEDS_MORE_EVIDENCE`或在有外部标注证据时考虑`SUSPECT_MISLABEL`。",
        "- `EXCLUDE_FOR_DATA_CORRUPTION`只能在文件截断、解析失败、帧错位、重复/乱序且无法恢复等事实被独立证实后使用；不能因为难分类或超出P05-P95使用。",
        "",
        "## 实际参考样本",
        "",
        f"CQ08 train parent数：{int(train['parent_sample_id'].nunique())}；残差parent数：{len(residual)}；残差事件数：{int(residual['event_count'].sum())}。",
        "",
        "|类别|CQ08 train parent数|残差parent数|",
        "|---|---:|---:|",
    ]
    for label in ("BirdTarget", "UnknownTarget"):
        lines.append(
            f"|{label}|{int(train['label_name'].eq(label).sum())}|{int(residual['label_name'].eq(label).sum())}|"
        )
    lines.extend(["", "## 同标签参考范围（parent级中位数）", "", "|指标|单位|Bird P05–P95|Bird Q1–Q3|Unknown P05–P95|Unknown Q1–Q3|", "|---|---|---:|---:|---:|---:|"])
    for metric in MAIN_METRICS + QUALITY_METRICS:
        values: list[str] = []
        for label in ("BirdTarget", "UnknownTarget"):
            stats = reference[reference["reference_label"].eq(label) & reference["metric"].eq(metric)]
            if len(stats) != 1:
                values.extend(["未统计", "未统计"])
            else:
                row = stats.iloc[0]
                values.extend([
                    f"{_fmt(row['p05'])}–{_fmt(row['p95'])}",
                    f"{_fmt(row['q1'])}–{_fmt(row['q3'])}",
                ])
        lines.append(
            f"|{DISPLAY_NAMES.get(metric, metric)}|{UNITS.get(metric, 'dimensionless')}|{values[0]}|{values[1]}|{values[2]}|{values[3]}|"
        )
    lines.extend([
        "",
        "## 输出文件",
        "",
        "- `reference-ranges.csv`：同标签CQ08 train的P05/Q1/median/Q3/P95/min/max。",
        "- `residual-parent-metrics.csv`：12个parent的事件、覆盖、TR/RD中位数和V10路由事实。",
        "- `residual-metric-percentiles.csv`：每个parent指标在同标签train分布中的经验百分位。",
        "- `reference-boxplots.png`：训练参考箱线图和12个残差点。",
        "- `residual-percentile-heatmap.png`：12个parent×16个主TR/RD指标的百分位热图。",
        "- `quality-and-coverage.png`：长度、时间有效性和RD覆盖比较。",
        "- `parents/`：每个parent的百分位图。",
        "",
        "## 本报告不能证明什么",
        "",
        "本报告不能证明标签正确性、Unknown是开放集未知、Bird/Drone身份、Clutter来源或液态状态漂移；这些需要人工标注规范、采集上下文、源记录或独立外部样本。没有人工审核决定时，不创建V11配置、不改标签、不重建训练数据、不训练。",
    ])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_html(output_path: Path, residual: pd.DataFrame, metrics_dir: Path) -> None:
    rows = []
    for item in residual.sort_values(["label_name", "track_id"]).itertuples(index=False):
        stem = f"track-{int(item.track_id):04d}"
        rows.append(
            "<tr>"
            f"<td>{html.escape(stem)}</td><td>{html.escape(str(item.label_name))}</td>"
            f"<td>{html.escape(str(item.V10_final_route))}</td>"
            f"<td>{float(item.V10_final_confidence):.4f}</td>"
            f"<td>{int(item.event_count)}</td><td>{float(item.rd_present_fraction):.3f}</td>"
            f"<td><a href='parents/{stem}-metrics.png'>metrics</a></td>"
            "</tr>"
        )
    document = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>RADAR-LN3 V10 residual metrics</title>
<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#17212b;background:#f5f7fa}table{border-collapse:collapse;background:white}th,td{padding:8px 10px;border:1px solid #d7dde5}th{background:#e9eef5}a{color:#075985}.notice{padding:14px;background:#fff3cd;border-left:5px solid #d39e00;margin-bottom:18px}</style></head>
<body><h1>RADAR-LN3 V10残差TR/RD指标对比</h1>
<div class="notice">仅为CQ08 train同标签parent分布的描述性比较。模型路由、置信度和百分位不能自动裁决标签，也不能等同于开放集拒识。</div>
<p><a href="review-metrics-guide.md">人工比对指南</a> | <a href="reference-boxplots.png">参考箱线图</a> | <a href="residual-percentile-heatmap.png">百分位热图</a> | <a href="quality-and-coverage.png">质量/覆盖图</a></p>
<table><thead><tr><th>track</th><th>当前标签</th><th>V10 route</th><th>confidence</th><th>events</th><th>RD coverage</th><th>parent metrics</th></tr></thead><tbody>
""" + "\n".join(rows) + """</tbody></table></body></html>
"""
    output_path.write_text(document, encoding="utf-8")


def run(project_root: Path, index_path: Path, review_dir: Path) -> dict[str, Any]:
    if not review_dir.exists():
        raise FileNotFoundError(f"review directory does not exist: {review_dir}")
    manual_review = review_dir / "manual-review.csv"
    if not manual_review.exists():
        raise FileNotFoundError(f"manual review template does not exist: {manual_review}")
    review_frame = pd.read_csv(manual_review, keep_default_na=False, low_memory=False)
    if review_frame["review_decision"].astype(str).str.strip().ne("").any():
        raise RuntimeError("review metrics refuses to run after a human decision was entered")
    train = _load_training_parent_metrics(project_root, index_path)
    residual = _load_residual_metrics(review_dir)
    labels = sorted(set(residual["label_name"].astype(str)))
    reference = _reference_table(train, labels)
    percentile = _add_empirical_percentiles(_percentile_table(residual, reference), train)

    metrics_dir = review_dir / "metrics"
    parent_dir = metrics_dir / "parents"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    parent_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(metrics_dir / "train-parent-metrics.csv", index=False, encoding="utf-8-sig")
    residual.to_csv(metrics_dir / "residual-parent-metrics.csv", index=False, encoding="utf-8-sig")
    reference.to_csv(metrics_dir / "reference-ranges.csv", index=False, encoding="utf-8-sig")
    percentile.to_csv(metrics_dir / "residual-metric-percentiles.csv", index=False, encoding="utf-8-sig")
    _plot_box_reference(train, residual, metrics_dir / "reference-boxplots.png")
    _plot_percentile_heatmap(percentile, metrics_dir / "residual-percentile-heatmap.png")
    _plot_quality(residual, reference, metrics_dir / "quality-and-coverage.png")
    for item in residual.itertuples(index=False):
        _plot_parent(
            str(item.parent_sample_id),
            str(item.label_name),
            str(item.V10_final_route),
            percentile,
            parent_dir / f"track-{int(item.track_id):04d}-metrics.png",
        )
    _write_guide(metrics_dir / "review-metrics-guide.md", train, residual, reference)
    _write_html(metrics_dir / "index.html", residual, metrics_dir)

    generated = sorted(
        path for path in metrics_dir.rglob("*") if path.is_file() and path.name not in {"summary.json", "files.sha256.json"}
    )
    file_manifest = [
        {"path": str(path), "sha256": _sha256(path), "bytes": int(path.stat().st_size)}
        for path in generated
    ]
    (metrics_dir / "files.sha256.json").write_text(
        json.dumps(file_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "passed",
        "schema_version": SCHEMA_VERSION,
        "scope": "CQ08 V3 train parent descriptive reference vs 12 V10 validation residual parents; test excluded",
        "index_path": str(index_path),
        "review_dir": str(review_dir),
        "train_parent_count": int(train["parent_sample_id"].nunique()),
        "train_event_count": int(sum(int(value) for value in train["event_count"])),
        "residual_parent_count": int(len(residual)),
        "residual_event_count": int(sum(int(value) for value in residual["event_count"])),
        "residual_label_counts": {
            str(key): int(value) for key, value in residual["label_name"].value_counts().sort_index().items()
        },
        "residual_rd_present_event_count": int(round(float(residual["rd_present_fraction"].mul(residual["event_count"]).sum()))),
        "manual_decisions": int(review_frame["review_decision"].astype(str).str.strip().ne("").sum()),
        "reference_unit": "parent median; same-label CQ08 train only",
        "threshold_fitting": "未执行",
        "label_changes": "未执行",
        "v11": "未创建",
        "training": "未执行",
        "test": "未执行",
        "interpretation_limits": [
            "P05-P95 and empirical percentiles are descriptive comparison ranges, not label or reject thresholds",
            "model route/confidence is not semantic label evidence",
            "UnknownTarget remains a closed-set label and is not equivalent to open-set rejection",
            "manual semantic decisions remain external-human-review work",
            "RD metrics are computed only on V3 mapped frames and do not recover missing wall-clock time",
        ],
        "outputs": {
            "guide": str(metrics_dir / "review-metrics-guide.md"),
            "html": str(metrics_dir / "index.html"),
            "summary": str(metrics_dir / "summary.json"),
            "files": str(metrics_dir / "files.sha256.json"),
        },
    }
    (metrics_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build descriptive V10 residual TR/RD review metrics.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = run(root, _resolve(root, args.index), _resolve(root, args.review_dir))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
