from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from radar_ln3.micro_doppler_broadband import (
    BROADBAND_CANDIDATE_SAMPLING_MODES,
    BROADBAND_MICRO_DOPPLER_NAMES,
    broadband_sampling_mode,
    broadband_sampling_mode_name,
)
from radar_ln3.task_models import (
    LiquidTaskClassifier,
    build_task_dataloader,
    build_task_dataset,
    load_task_checkpoint,
    load_task_config,
    task_device,
    verify_task_data_contract,
)


CLASS_NAMES = ("DroneTarget", "BirdTarget", "BalloonTarget")
CONTROL_CONFIG = Path(
    "config/RADAR-LN3/RADAR-LN3-V18-cq08-liquid-dbb-specialist-"
    "rd-physical4-broadband-multiscale-micro-doppler8-stable-control-seed42-config.json"
)
TREATMENT_CONFIG = Path(
    "config/RADAR-LN3/RADAR-LN3-V18-cq08-liquid-dbb-specialist-"
    "rd-physical4-broadband-multiscale-micro-doppler8-stable-treatment-seed42-config.json"
)
PAIRED_REPORT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V18-broadband-multiscale-micro-doppler8-stable-paired-validation.json"
)
OUTPUT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V18-broadband-multiscale-micro-doppler8-stable-error-attribution.json"
)
PARENT_CSV = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V18-broadband-multiscale-micro-doppler8-stable-error-parent-summary.csv"
)
EVENT_CSV = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V18-broadband-multiscale-micro-doppler8-stable-error-event-detail.csv"
)
NEIGHBOR_CSV = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V18-broadband-multiscale-micro-doppler8-stable-error-neighbors.csv"
)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _forward_inputs(model: LiquidTaskClassifier, tensors: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "tr_features": tensors["tr_features"],
        "tr_missing_mask": tensors["tr_missing_mask"],
        "tr_dt_seconds": tensors["tr_dt_seconds"],
        "tr_dt_valid": tensors["tr_dt_valid"],
        "tr_dt_reason": tensors.get("tr_dt_reason"),
        "tr_motion_features": tensors.get("tr_motion_features"),
        "tr_motion_valid_mask": tensors.get("tr_motion_valid_mask"),
        "rd_physical_features": tensors.get("rd_physical_features"),
        "rd_physical_valid": tensors.get("rd_physical_valid"),
        "micro_doppler_features": (
            tensors.get("micro_doppler_features") if model.micro_doppler_dim else None
        ),
        "micro_doppler_valid": (
            tensors.get("micro_doppler_valid") if model.micro_doppler_dim else None
        ),
        "rd_frames": tensors["rd_frames"],
        "rd_frame_mask": tensors["rd_frame_mask"],
        "rd_quality": tensors["rd_quality"],
        "tr_padding_mask": tensors["tr_padding_mask"],
    }


def _infer_arm(root: Path, config_path: Path) -> dict[str, Any]:
    config = load_task_config(root / config_path)
    if tuple(config.task.class_names) != CLASS_NAMES or config.task.name != "dbb_specialist":
        raise ValueError(f"V18 attribution requires the frozen DBB specialist: {config_path}")
    contract = verify_task_data_contract(root, config)
    dataset = build_task_dataset(
        root, config, split_name=config.data.validation_split_name
    )
    loader = build_task_dataloader(
        dataset, batch_size=config.train.batch_size, shuffle=False
    )
    device = task_device(config)
    model = LiquidTaskClassifier(config).to(device)
    checkpoint_path = root / config.train.checkpoint_path
    checkpoint = load_task_checkpoint(
        checkpoint_path,
        model,
        expected_schema=config.train.checkpoint_schema,
        map_location=device,
    )
    model.eval()
    parents: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        for batch in loader:
            tensors = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            output = model(**_forward_inputs(model, tensors))
            probabilities = torch.softmax(output["sequence_logits"], dim=-1).cpu().numpy()
            for batch_index, parent_id_value in enumerate(batch["parent_sample_ids"]):
                parent_id = str(parent_id_value)
                length = int(batch["lengths"][batch_index].item())
                values = np.asarray(probabilities[batch_index, :length], dtype=np.float64)
                predictions = values.argmax(axis=1).astype(int)
                parents[parent_id] = {
                    "parent_sample_id": parent_id,
                    "track_id": str(batch["track_ids"][batch_index]),
                    "label_name": str(batch["label_names"][batch_index]),
                    "true_id": int(batch["labels"][batch_index].item()),
                    "length": length,
                    "event_probabilities": values.tolist(),
                    "event_predictions": predictions.tolist(),
                    "final_probabilities": values[-1].tolist(),
                    "final_prediction": int(predictions[-1]),
                    "final_confidence": float(values[-1, predictions[-1]]),
                    "prediction_transition_count": int(np.count_nonzero(np.diff(predictions))),
                }
    if len(parents) != len(dataset):
        raise ValueError("V18 attribution inference did not cover every validation parent")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "config": config,
        "config_path": root / config_path,
        "checkpoint_path": checkpoint_path,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "contract_hashes": contract["hashes"],
        "parents": parents,
    }


def _split_cache_paths(root: Path, config: Any, split: str) -> tuple[Path, Path | None]:
    if split == "train":
        summary = config.data.train_micro_doppler_cache_path
        evidence = config.data.train_micro_doppler_stft_cache_path
    elif split == "val":
        summary = config.data.validation_micro_doppler_cache_path
        evidence = config.data.validation_micro_doppler_stft_cache_path
    else:
        raise ValueError(f"unsupported attribution split: {split}")
    if summary is None:
        raise ValueError(f"V18 {split} summary cache is not configured")
    return _resolve(root, summary), None if evidence is None else _resolve(root, evidence)


def _parent_feature_records(
    root: Path,
    config: Any,
    *,
    split: str,
    include_evidence: bool,
) -> dict[str, dict[str, Any]]:
    summary_path, evidence_path = _split_cache_paths(root, config, split)
    summary = _load_npz(summary_path)
    required = {"sample_ids", "values", "valid"}
    if set(summary) != required:
        raise ValueError(f"V18 {split} summary cache fields differ: {set(summary)}")
    sample_ids = summary["sample_ids"].astype(str)
    values = np.asarray(summary["values"], dtype=np.float64)
    valid = np.asarray(summary["valid"], dtype=bool)
    if values.shape != (len(sample_ids), len(BROADBAND_MICRO_DOPPLER_NAMES)):
        raise ValueError(f"V18 {split} summary cache shape is invalid: {values.shape}")
    index_path = _resolve(root, config.data.index_path)
    index = pd.read_csv(index_path, low_memory=False)
    rows = index.set_index(index["sample_id"].astype(str), drop=False).loc[sample_ids].copy()
    if not np.array_equal(rows["sample_id"].astype(str).to_numpy(), sample_ids):
        raise ValueError(f"V18 {split} summary/index sample order differs")
    if rows["split_name"].astype(str).ne(split).any():
        raise ValueError(f"V18 attribution {split} cache contains another split")
    evidence: dict[str, np.ndarray] | None = None
    if include_evidence:
        if evidence_path is None:
            raise ValueError("V18 validation evidence cache is not configured")
        evidence = _load_npz(evidence_path)
        if not np.array_equal(evidence["sample_ids"].astype(str), sample_ids):
            raise ValueError("V18 summary/evidence sample order differs")
        if evidence["event_offsets"].shape != (len(sample_ids) + 1,):
            raise ValueError("V18 evidence event offsets are invalid")
    parent_indices: dict[str, list[int]] = {}
    for position, parent_id in enumerate(rows["parent_sample_id"].astype(str)):
        parent_indices.setdefault(parent_id, []).append(position)
    parents: dict[str, dict[str, Any]] = {}
    calibrated_names = {
        broadband_sampling_mode_name(mode) for mode in BROADBAND_CANDIDATE_SAMPLING_MODES
    }
    for parent_id, positions_list in parent_indices.items():
        positions = np.asarray(positions_list, dtype=np.int64)
        parent_rows = rows.iloc[positions]
        if parent_rows["label_name"].astype(str).nunique() != 1:
            raise ValueError(f"V18 attribution parent has multiple labels: {parent_id}")
        parent_values = values[positions]
        parent_valid = valid[positions]
        valid_values = parent_values[parent_valid]
        valid_mean = (
            valid_values.mean(axis=0)
            if len(valid_values)
            else np.zeros((len(BROADBAND_MICRO_DOPPLER_NAMES),), dtype=np.float64)
        )
        record: dict[str, Any] = {
            "parent_sample_id": parent_id,
            "track_id": str(parent_rows.iloc[0]["track_id"]),
            "label_name": str(parent_rows.iloc[0]["label_name"]),
            "event_count": int(len(positions)),
            "valid_event_count": int(parent_valid.sum()),
            "valid_event_fraction": float(parent_valid.mean()),
            "valid_event_mean": valid_mean,
            "all_event_mean": parent_values.mean(axis=0),
            "valid_event_std": (
                valid_values.std(axis=0)
                if len(valid_values)
                else np.zeros((len(BROADBAND_MICRO_DOPPLER_NAMES),), dtype=np.float64)
            ),
            "event_sample_ids": sample_ids[positions].tolist(),
            "event_values": parent_values,
            "event_valid": parent_valid,
        }
        if evidence is not None:
            all_modes: Counter[str] = Counter()
            reliable_modes: Counter[str] = Counter()
            event_modes: list[dict[str, Any]] = []
            offsets = np.asarray(evidence["event_offsets"], dtype=np.int64)
            for event_position in positions:
                start = int(offsets[event_position])
                stop = int(offsets[event_position + 1])
                frame_modes: Counter[str] = Counter()
                reliable_frame_modes: Counter[str] = Counter()
                for frame_index in range(start, stop):
                    mode = broadband_sampling_mode(
                        int(evidence["frame_prt_num"][frame_index]),
                        float(evidence["frame_prt_seconds"][frame_index]),
                        float(evidence["frame_frequency_hz"][frame_index]),
                    )
                    name = broadband_sampling_mode_name(mode)
                    frame_modes[name] += 1
                    all_modes[name] += 1
                    if bool(evidence["frame_feature_reliable"][frame_index]):
                        reliable_frame_modes[name] += 1
                        reliable_modes[name] += 1
                event_modes.append(
                    {
                        "all_frame_modes": dict(sorted(frame_modes.items())),
                        "reliable_frame_modes": dict(
                            sorted(reliable_frame_modes.items())
                        ),
                    }
                )
            record.update(
                {
                    "all_frame_sampling_mode_counts": dict(sorted(all_modes.items())),
                    "reliable_frame_sampling_mode_counts": dict(
                        sorted(reliable_modes.items())
                    ),
                    "calibrated_reliable_frame_count": int(
                        sum(
                            count
                            for name, count in reliable_modes.items()
                            if name in calibrated_names
                        )
                    ),
                    "event_sampling_modes": event_modes,
                }
            )
        parents[parent_id] = record
    return parents


def _robust_reference(
    train_parents: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    ids = sorted(train_parents)
    matrix = np.stack(
        [np.asarray(train_parents[parent_id]["valid_event_mean"]) for parent_id in ids]
    )
    labels = np.asarray([str(train_parents[parent_id]["label_name"]) for parent_id in ids])
    median = np.median(matrix, axis=0)
    q25 = np.quantile(matrix, 0.25, axis=0)
    q75 = np.quantile(matrix, 0.75, axis=0)
    spread = q75 - q25
    standard = matrix.std(axis=0)
    scale = np.where(spread > 1.0e-6, spread, np.maximum(standard, 1.0e-6))
    z = (matrix - median) / scale
    class_centers: dict[str, np.ndarray] = {}
    class_radius: dict[str, dict[str, float]] = {}
    class_feature_values: dict[str, np.ndarray] = {}
    for class_name in CLASS_NAMES:
        selected = z[labels == class_name]
        raw_selected = matrix[labels == class_name]
        center = np.median(selected, axis=0)
        distances = np.sqrt(np.mean(np.square(selected - center), axis=1))
        class_centers[class_name] = center
        class_feature_values[class_name] = raw_selected
        class_radius[class_name] = {
            "median": float(np.quantile(distances, 0.50)),
            "q90": float(np.quantile(distances, 0.90)),
            "q95": float(np.quantile(distances, 0.95)),
        }
    return {
        "ids": ids,
        "matrix": matrix,
        "labels": labels,
        "median": median,
        "scale": scale,
        "class_centers": class_centers,
        "class_radius": class_radius,
        "class_feature_values": class_feature_values,
    }


def _rms_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(left - right))))


def _percentile(values: np.ndarray, value: float) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64) <= float(value)))


def _trajectory_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    probabilities = np.asarray(record["event_probabilities"], dtype=np.float64)
    predictions = np.asarray(record["event_predictions"], dtype=np.int64)
    final_prediction = int(predictions[-1])
    eligible = predictions[2:] if len(predictions) > 2 else predictions
    counts = Counter(CLASS_NAMES[int(value)] for value in eligible)
    matching = np.flatnonzero(predictions == final_prediction)
    return {
        "length": int(len(predictions)),
        "final_probabilities": {
            class_name: float(probabilities[-1, class_id])
            for class_id, class_name in enumerate(CLASS_NAMES)
        },
        "final_prediction": CLASS_NAMES[final_prediction],
        "final_confidence": float(probabilities[-1, final_prediction]),
        "final_margin_over_runner_up": float(
            np.sort(probabilities[-1])[-1] - np.sort(probabilities[-1])[-2]
        ),
        "eligible_prediction_counts": dict(sorted(counts.items())),
        "eligible_final_prediction_fraction": float(
            np.mean(eligible == final_prediction)
        ),
        "first_event_with_final_prediction": int(matching[0]) if len(matching) else None,
        "prediction_transition_count": int(np.count_nonzero(np.diff(predictions))),
        "last_five_probabilities": [
            {
                "event_index": int(event_index),
                **{
                    class_name: float(probabilities[event_index, class_id])
                    for class_id, class_name in enumerate(CLASS_NAMES)
                },
            }
            for event_index in range(max(0, len(predictions) - 5), len(predictions))
        ],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty attribution CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    paired_path = root / PAIRED_REPORT
    paired = json.loads(paired_path.read_text(encoding="utf-8"))
    if (
        paired.get("status") != "completed_but_gate_failed"
        or paired.get("test") != "未执行"
        or paired.get("OOF") != "未执行"
    ):
        raise ValueError("V18 attribution requires the frozen failed paired report")

    control = _infer_arm(root, CONTROL_CONFIG)
    treatment = _infer_arm(root, TREATMENT_CONFIG)
    if set(control["parents"]) != set(treatment["parents"]):
        raise ValueError("V18 Control/Treatment validation parents differ")
    treatment_config = treatment["config"]
    train_features = _parent_feature_records(
        root, treatment_config, split="train", include_evidence=False
    )
    val_features = _parent_feature_records(
        root, treatment_config, split="val", include_evidence=True
    )
    if set(val_features) != set(treatment["parents"]):
        raise ValueError("V18 attribution feature and inference parents differ")
    reference = _robust_reference(train_features)
    robust_median = np.asarray(reference["median"])
    robust_scale = np.asarray(reference["scale"])

    val_ids = sorted(val_features)
    val_z = {
        parent_id: (
            np.asarray(val_features[parent_id]["valid_event_mean"]) - robust_median
        )
        / robust_scale
        for parent_id in val_ids
    }
    correctly_classified = [
        parent_id
        for parent_id in val_ids
        if treatment["parents"][parent_id]["final_prediction"]
        == treatment["parents"][parent_id]["true_id"]
    ]
    requested_pairs = {
        ("DroneTarget", "BirdTarget"),
        ("BirdTarget", "DroneTarget"),
        ("BalloonTarget", "BirdTarget"),
    }
    targets = [
        parent_id
        for parent_id in val_ids
        if (
            str(treatment["parents"][parent_id]["label_name"]),
            CLASS_NAMES[int(treatment["parents"][parent_id]["final_prediction"])],
        )
        in requested_pairs
    ]
    pair_counts = Counter(
        (
            str(treatment["parents"][parent_id]["label_name"]),
            CLASS_NAMES[int(treatment["parents"][parent_id]["final_prediction"])],
        )
        for parent_id in targets
    )
    expected_counts = {
        ("DroneTarget", "BirdTarget"): 4,
        ("BirdTarget", "DroneTarget"): 3,
        ("BalloonTarget", "BirdTarget"): 1,
    }
    if dict(pair_counts) != expected_counts:
        raise ValueError(f"V18 requested attribution error counts changed: {pair_counts}")

    parent_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    neighbor_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    class_centers = reference["class_centers"]
    class_radius = reference["class_radius"]
    for parent_id in targets:
        feature = val_features[parent_id]
        treatment_record = treatment["parents"][parent_id]
        control_record = control["parents"][parent_id]
        true_name = str(treatment_record["label_name"])
        predicted_name = CLASS_NAMES[int(treatment_record["final_prediction"])]
        z_value = val_z[parent_id]
        center_distances = {
            class_name: _rms_distance(z_value, class_centers[class_name])
            for class_name in CLASS_NAMES
        }
        center_ranking = sorted(center_distances, key=center_distances.get)
        distances = sorted(
            (
                _rms_distance(z_value, val_z[candidate]),
                candidate,
                str(val_features[candidate]["label_name"]),
            )
            for candidate in correctly_classified
            if candidate != parent_id
        )
        nearest = distances[:15]
        local_counts = Counter(label for _, _, label in nearest)
        nearest_by_class: dict[str, Any] = {}
        for class_name in CLASS_NAMES:
            selected = [row for row in distances if row[2] == class_name]
            nearest_by_class[class_name] = (
                None
                if not selected
                else {
                    "distance": float(selected[0][0]),
                    "parent_sample_id": selected[0][1],
                    "track_id": str(val_features[selected[0][1]]["track_id"]),
                }
            )
        feature_details: list[dict[str, Any]] = []
        closer_counts = Counter()
        raw_vector = np.asarray(feature["valid_event_mean"])
        event_values = np.asarray(feature["event_values"])
        event_valid = np.asarray(feature["event_valid"], dtype=bool)
        event_feature_center_predictions: list[str | None] = []
        event_feature_center_distances: list[dict[str, float] | None] = []
        event_feature_center_counts: Counter[str] = Counter()
        for event_index in range(int(feature["event_count"])):
            if not bool(event_valid[event_index]):
                event_feature_center_predictions.append(None)
                event_feature_center_distances.append(None)
                continue
            event_z = (event_values[event_index] - robust_median) / robust_scale
            distances_by_class = {
                class_name: _rms_distance(event_z, class_centers[class_name])
                for class_name in CLASS_NAMES
            }
            nearest_class = min(distances_by_class, key=distances_by_class.get)
            event_feature_center_predictions.append(nearest_class)
            event_feature_center_distances.append(distances_by_class)
            event_feature_center_counts[nearest_class] += 1
        for feature_index, feature_name in enumerate(BROADBAND_MICRO_DOPPLER_NAMES):
            class_medians = {
                class_name: float(
                    np.median(reference["class_feature_values"][class_name][:, feature_index])
                )
                for class_name in CLASS_NAMES
            }
            class_percentiles = {
                class_name: _percentile(
                    reference["class_feature_values"][class_name][:, feature_index],
                    float(raw_vector[feature_index]),
                )
                for class_name in CLASS_NAMES
            }
            true_difference = abs(
                float(z_value[feature_index] - class_centers[true_name][feature_index])
            )
            predicted_difference = abs(
                float(z_value[feature_index] - class_centers[predicted_name][feature_index])
            )
            closer = (
                true_name
                if true_difference < predicted_difference
                else predicted_name
                if predicted_difference < true_difference
                else "tie"
            )
            closer_counts[closer] += 1
            feature_details.append(
                {
                    "feature": feature_name,
                    "value": float(raw_vector[feature_index]),
                    "train_class_medians": class_medians,
                    "train_class_percentiles": class_percentiles,
                    "closer_center_between_true_and_predicted": closer,
                    "absolute_robust_distance_to_true_center": true_difference,
                    "absolute_robust_distance_to_predicted_center": predicted_difference,
                }
            )
        true_inside = center_distances[true_name] <= class_radius[true_name]["q95"]
        predicted_inside = (
            center_distances[predicted_name] <= class_radius[predicted_name]["q95"]
        )
        local_majority = local_counts.most_common(1)[0][0]
        predicted_neighbor_fraction = float(local_counts[predicted_name] / len(nearest))
        true_neighbor_fraction = float(local_counts[true_name] / len(nearest))
        if feature["valid_event_fraction"] < 0.50:
            advisory = "insufficient_valid_micro_doppler_evidence"
        elif (
            true_inside
            and predicted_inside
            and true_neighbor_fraction >= 0.20
            and predicted_neighbor_fraction >= 0.20
        ):
            advisory = "true_and_predicted_classes_overlap_in_stable8_space"
        elif (
            center_distances[predicted_name] < center_distances[true_name]
            and local_majority == predicted_name
        ):
            advisory = "stable8_features_support_predicted_class_or_label_is_atypical"
        elif (
            center_distances[true_name] < center_distances[predicted_name]
            and local_majority == true_name
        ):
            advisory = "model_coupling_error_despite_true_class_feature_support"
        else:
            advisory = "mixed_feature_evidence"

        treatment_trajectory = _trajectory_summary(treatment_record)
        control_trajectory = _trajectory_summary(control_record)
        record = {
            "parent_sample_id": parent_id,
            "track_id": str(feature["track_id"]),
            "true_class": true_name,
            "treatment_predicted_class": predicted_name,
            "control_predicted_class": CLASS_NAMES[
                int(control_record["final_prediction"])
            ],
            "event_count": int(feature["event_count"]),
            "valid_event_count": int(feature["valid_event_count"]),
            "valid_event_fraction": float(feature["valid_event_fraction"]),
            "treatment_trajectory": treatment_trajectory,
            "control_trajectory": control_trajectory,
            "treatment_minus_control_final_probabilities": {
                class_name: float(
                    treatment_record["final_probabilities"][class_id]
                    - control_record["final_probabilities"][class_id]
                )
                for class_id, class_name in enumerate(CLASS_NAMES)
            },
            "stable8_parent_valid_event_mean": {
                feature_name: float(raw_vector[index])
                for index, feature_name in enumerate(BROADBAND_MICRO_DOPPLER_NAMES)
            },
            "stable8_parent_valid_event_std": {
                feature_name: float(feature["valid_event_std"][index])
                for index, feature_name in enumerate(BROADBAND_MICRO_DOPPLER_NAMES)
            },
            "train_robust_center_distances": center_distances,
            "train_class_q95_center_radius": {
                class_name: float(class_radius[class_name]["q95"])
                for class_name in CLASS_NAMES
            },
            "inside_true_class_train_q95_radius": bool(true_inside),
            "inside_predicted_class_train_q95_radius": bool(predicted_inside),
            "feature_only_nearest_train_center_class": center_ranking[0],
            "nearest_correct_validation_by_class": nearest_by_class,
            "nearest_15_correct_validation_label_counts": dict(
                sorted(local_counts.items())
            ),
            "nearest_15_true_class_fraction": true_neighbor_fraction,
            "nearest_15_predicted_class_fraction": predicted_neighbor_fraction,
            "nearest_15_majority_class": local_majority,
            "per_feature_closer_center_counts": dict(sorted(closer_counts.items())),
            "valid_event_feature_only_center_prediction_counts": dict(
                sorted(event_feature_center_counts.items())
            ),
            "valid_event_feature_only_true_class_fraction": float(
                event_feature_center_counts[true_name]
                / max(1, int(feature["valid_event_count"]))
            ),
            "valid_event_feature_only_predicted_class_fraction": float(
                event_feature_center_counts[predicted_name]
                / max(1, int(feature["valid_event_count"]))
            ),
            "per_feature_details": feature_details,
            "reliable_frame_sampling_mode_counts": feature.get(
                "reliable_frame_sampling_mode_counts", {}
            ),
            "all_frame_sampling_mode_counts": feature.get(
                "all_frame_sampling_mode_counts", {}
            ),
            "advisory_feature_attribution": advisory,
        }
        records.append(record)
        parent_rows.append(
            {
                "parent_sample_id": parent_id,
                "track_id": feature["track_id"],
                "true_class": true_name,
                "treatment_predicted_class": predicted_name,
                "control_predicted_class": record["control_predicted_class"],
                "event_count": feature["event_count"],
                "valid_event_count": feature["valid_event_count"],
                "valid_event_fraction": feature["valid_event_fraction"],
                "treatment_confidence": treatment_trajectory["final_confidence"],
                "treatment_margin": treatment_trajectory[
                    "final_margin_over_runner_up"
                ],
                "control_confidence": control_trajectory["final_confidence"],
                "true_center_distance": center_distances[true_name],
                "predicted_center_distance": center_distances[predicted_name],
                "inside_true_q95": true_inside,
                "inside_predicted_q95": predicted_inside,
                "nearest_true_correct_distance": nearest_by_class[true_name][
                    "distance"
                ],
                "nearest_predicted_correct_distance": nearest_by_class[
                    predicted_name
                ]["distance"],
                "nearest_15_true_fraction": true_neighbor_fraction,
                "nearest_15_predicted_fraction": predicted_neighbor_fraction,
                "feature_only_center_class": center_ranking[0],
                "nearest_15_majority_class": local_majority,
                "event_feature_center_true_fraction": record[
                    "valid_event_feature_only_true_class_fraction"
                ],
                "event_feature_center_predicted_fraction": record[
                    "valid_event_feature_only_predicted_class_fraction"
                ],
                "advisory": advisory,
            }
        )
        for rank, (distance, neighbor_id, neighbor_label) in enumerate(
            nearest[:10], start=1
        ):
            neighbor_rows.append(
                {
                    "parent_sample_id": parent_id,
                    "true_class": true_name,
                    "predicted_class": predicted_name,
                    "neighbor_rank": rank,
                    "neighbor_parent_sample_id": neighbor_id,
                    "neighbor_track_id": val_features[neighbor_id]["track_id"],
                    "neighbor_label": neighbor_label,
                    "robust_rms_distance": distance,
                }
            )
        event_modes = list(feature.get("event_sampling_modes") or [])
        control_probabilities = np.asarray(control_record["event_probabilities"])
        treatment_probabilities = np.asarray(treatment_record["event_probabilities"])
        for event_index in range(int(feature["event_count"])):
            row: dict[str, Any] = {
                "parent_sample_id": parent_id,
                "track_id": feature["track_id"],
                "true_class": true_name,
                "final_predicted_class": predicted_name,
                "event_index": event_index,
                "micro_doppler_valid": bool(event_valid[event_index]),
                "control_prediction": CLASS_NAMES[
                    int(control_probabilities[event_index].argmax())
                ],
                "treatment_prediction": CLASS_NAMES[
                    int(treatment_probabilities[event_index].argmax())
                ],
                "event_feature_only_center_prediction": (
                    event_feature_center_predictions[event_index]
                ),
                "reliable_frame_modes": json.dumps(
                    event_modes[event_index]["reliable_frame_modes"],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
            for class_index, class_name in enumerate(CLASS_NAMES):
                row[f"control_probability_{class_name}"] = float(
                    control_probabilities[event_index, class_index]
                )
                row[f"treatment_probability_{class_name}"] = float(
                    treatment_probabilities[event_index, class_index]
                )
                row[f"event_feature_center_distance_{class_name}"] = (
                    None
                    if event_feature_center_distances[event_index] is None
                    else float(
                        event_feature_center_distances[event_index][class_name]
                    )
                )
            for feature_index, feature_name in enumerate(
                BROADBAND_MICRO_DOPPLER_NAMES
            ):
                row[feature_name] = float(event_values[event_index, feature_index])
            event_rows.append(row)

    advisory_counts = Counter(
        str(record["advisory_feature_attribution"]) for record in records
    )
    report = {
        "status": "completed",
        "schema_version": (
            "RADAR-LN3-V18-broadband-multiscale-micro-doppler8-stable-"
            "error-attribution-v1"
        ),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": (
            "V18 Treatment best epoch validation only; requested 4 Drone->Bird, "
            "3 Bird->Drone and 1 Balloon->Bird parents"
        ),
        "method": {
            "model_probabilities": (
                "offline causal sequence inference from frozen Control/Treatment best checkpoints"
            ),
            "feature_unit": (
                "mean stable8 current-event summaries over valid events within each parent"
            ),
            "feature_scaling": (
                "train-parent-only median and IQR; validation never fits a threshold or scale"
            ),
            "distance": "RMS Euclidean distance in train-robust-scaled stable8 space",
            "neighbors": "correctly classified Treatment validation parents only",
            "class_support_radius": "95th percentile train-parent distance to train class center",
            "advisory_not_label_review": True,
        },
        "feature_names": list(BROADBAND_MICRO_DOPPLER_NAMES),
        "target_counts": {
            f"{true_name}->{predicted_name}": int(count)
            for (true_name, predicted_name), count in sorted(pair_counts.items())
        },
        "advisory_counts": dict(sorted(advisory_counts.items())),
        "train_class_radius": class_radius,
        "records": records,
        "provenance": {
            "paired_report": str(paired_path),
            "paired_report_sha256": _sha256(paired_path),
            "control_config": str(control["config_path"]),
            "control_config_sha256": _sha256(control["config_path"]),
            "control_checkpoint": str(control["checkpoint_path"]),
            "control_checkpoint_sha256": _sha256(control["checkpoint_path"]),
            "control_checkpoint_epoch": control["checkpoint_epoch"],
            "treatment_config": str(treatment["config_path"]),
            "treatment_config_sha256": _sha256(treatment["config_path"]),
            "treatment_checkpoint": str(treatment["checkpoint_path"]),
            "treatment_checkpoint_sha256": _sha256(treatment["checkpoint_path"]),
            "treatment_checkpoint_epoch": treatment["checkpoint_epoch"],
            "contract_hashes": treatment["contract_hashes"],
        },
        "interpretation_limits": [
            "This is post-hoc validation attribution and cannot tune a new threshold or select a checkpoint.",
            "Feature similarity cannot prove a label is wrong or that raw phenomenology is identical.",
            "No test, OOF, full cascade, rejection, stable or P95 evaluation was executed.",
        ],
        "test": "未执行",
        "OOF": "未执行",
        "full_cascade": "未执行",
        "formal_rejection": "未执行",
        "stable": "未执行",
        "P95": "未执行",
    }
    output = root / OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(root / PARENT_CSV, parent_rows)
    _write_csv(root / EVENT_CSV, event_rows)
    _write_csv(root / NEIGHBOR_CSV, neighbor_rows)
    report["report_path"] = str(output)
    report["parent_csv"] = str(root / PARENT_CSV)
    report["event_csv"] = str(root / EVENT_CSV)
    report["neighbor_csv"] = str(root / NEIGHBOR_CSV)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attribute the eight requested V18 stable8 DBB validation errors."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    result = run(args.project_root.resolve())
    print(
        json.dumps(
            {
                "status": result["status"],
                "schema_version": result["schema_version"],
                "target_counts": result["target_counts"],
                "advisory_counts": result["advisory_counts"],
                "report_path": result["report_path"],
                "parent_csv": result["parent_csv"],
                "event_csv": result["event_csv"],
                "neighbor_csv": result["neighbor_csv"],
                "test": result["test"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
