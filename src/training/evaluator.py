import csv
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.fusion_classifier import FusionRadarClassifier
from src.training.config import TrainingConfig
from src.training.session_manager import build_dataset
from src.training.trainer import apply_logit_bias
from src.utils.checkpoint import load_checkpoint


def _apply_pairwise_calibration(
    logits: torch.Tensor,
    pair_calibration: dict[str, Any] | None,
) -> torch.Tensor:
    if not pair_calibration:
        return logits
    adjusted = logits.clone()
    probabilities = torch.softmax(adjusted, dim=-1)
    for payload in pair_calibration.values():
        pair_indices = payload.get('pair_indices') or []
        pair_bias = payload.get('pair_bias') or []
        gate_threshold = float(payload.get('gate_threshold', 0.55))
        if len(pair_indices) != 2 or len(pair_bias) != 2:
            continue
        first_index = int(pair_indices[0])
        second_index = int(pair_indices[1])
        gate_mask = (probabilities[:, first_index] + probabilities[:, second_index]) >= gate_threshold
        if gate_mask.any():
            adjusted[gate_mask, first_index] = adjusted[gate_mask, first_index] + float(pair_bias[0])
            adjusted[gate_mask, second_index] = adjusted[gate_mask, second_index] + float(pair_bias[1])
            probabilities = torch.softmax(adjusted, dim=-1)
    return adjusted


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


def _compute_classification_metrics(confusion_matrix: np.ndarray, labels: list[str]) -> dict[str, Any]:
    total = int(confusion_matrix.sum())
    accuracy = float(np.trace(confusion_matrix) / total) if total else 0.0
    per_class: dict[str, dict[str, float | int]] = {}
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0

    for index, label in enumerate(labels):
        tp = float(confusion_matrix[index, index])
        fp = float(confusion_matrix[:, index].sum() - tp)
        fn = float(confusion_matrix[index, :].sum() - tp)
        support = int(confusion_matrix[index, :].sum())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    class_count = max(len(labels), 1)
    return {
        "accuracy": accuracy,
        "macro_precision": precision_sum / class_count,
        "macro_recall": recall_sum / class_count,
        "macro_f1": f1_sum / class_count,
        "per_class": per_class,
    }


def _derive_session_id(sample_id: str) -> str:
    parts = str(sample_id).split("_")
    if len(parts) >= 3:
        return "_".join(parts[:-2])
    return str(sample_id)


def _split_sample_id(sample_id: str) -> tuple[str, str, str]:
    parts = str(sample_id).split("_")
    if len(parts) >= 3:
        return "_".join(parts[:-2]), parts[-2], parts[-1]
    return str(sample_id), "", ""


def _write_csv_rows(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with output_path.open("w", encoding="utf-8", newline="") as file:
            file.write("")
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_csv_rows(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _to_float(value: Any, default: float = 0.0) -> float:
    if value in {None, "", "None"}:
        return default
    return float(value)


def _to_int(value: Any, default: int = 0) -> int:
    if value in {None, "", "None"}:
        return default
    return int(float(value))


def _build_suspicious_error_score(row: dict[str, Any]) -> float:
    base = _to_float(row.get("top1_prob"))
    margin = _to_float(row.get("margin_top1_top2"))
    router_surface = _to_float(row.get("router_surface_prob"))
    ship_drone_ship = _to_float(row.get("ship_drone_ship_prob"))
    ship_bird_ship = _to_float(row.get("ship_bird_ship_prob"))
    bird_surface = _to_float(row.get("bird_surface_prob"))
    true_label = str(row.get("true_label", ""))
    predicted_label = str(row.get("predicted_label", ""))

    score = base + 0.25 * margin
    if true_label == "ShipTarget" and predicted_label == "DroneTarget":
        score += 0.35 + 0.15 * (1.0 - ship_drone_ship)
    if true_label == "ShipTarget" and predicted_label == "BirdTarget":
        score += 0.3 + 0.15 * (1.0 - ship_bird_ship)
    if true_label == "BirdTarget" and predicted_label == "ShipTarget":
        score += 0.35 + 0.15 * router_surface + 0.1 * (1.0 - bird_surface)
    if true_label == "BirdTarget" and predicted_label == "AircraftTarget":
        score += 0.2
    if true_label == "BirdTarget" and predicted_label == "DroneTarget":
        score += 0.2
    return float(score)


def _analyze_clip_rows(clip_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    suspicious_clips: list[dict[str, Any]] = []
    by_session: dict[str, list[dict[str, Any]]] = {}

    for row in clip_rows:
        if str(row.get("predicted_label", "")) == str(row.get("true_label", "")):
            continue
        enriched = dict(row)
        enriched["clip_token_int"] = _to_int(row.get("clip_token"))
        enriched["suspicious_score"] = _build_suspicious_error_score(row)
        suspicious_clips.append(enriched)
        by_session.setdefault(str(row.get("session_id", "")), []).append(enriched)

    suspicious_clips.sort(key=lambda item: (-float(item["suspicious_score"]), item["session_id"], item["clip_token_int"]))

    suspicious_spans: list[dict[str, Any]] = []
    session_summaries: list[dict[str, Any]] = []
    for session_id, rows in by_session.items():
        ordered_rows = sorted(rows, key=lambda item: item["clip_token_int"])
        clip_tokens = [row["clip_token_int"] for row in ordered_rows]
        step = 4
        if len(clip_tokens) >= 2:
            deltas = [b - a for a, b in zip(clip_tokens[:-1], clip_tokens[1:]) if b > a]
            if deltas:
                step = max(1, int(round(float(np.median(np.asarray(deltas, dtype=np.float32))))))
        max_gap = max(step, 4)

        current_span: list[dict[str, Any]] = []
        for row in ordered_rows:
            if not current_span:
                current_span = [row]
                continue
            previous_token = current_span[-1]["clip_token_int"]
            if row["clip_token_int"] - previous_token <= max_gap:
                current_span.append(row)
            else:
                suspicious_spans.append(_summarize_span(session_id, current_span))
                current_span = [row]
        if current_span:
            suspicious_spans.append(_summarize_span(session_id, current_span))

        session_summaries.append(
            {
                "session_id": session_id,
                "num_suspicious_clips": len(ordered_rows),
                "max_suspicious_score": float(max(row["suspicious_score"] for row in ordered_rows)),
                "mean_suspicious_score": float(np.mean([row["suspicious_score"] for row in ordered_rows])),
            }
        )

    suspicious_spans.sort(key=lambda item: (-int(item["num_clips"]), -float(item["mean_suspicious_score"]), item["session_id"], item["start_clip_token"]))
    session_summaries.sort(key=lambda item: (-int(item["num_suspicious_clips"]), -float(item["mean_suspicious_score"]), item["session_id"]))
    analysis_summary = {
        "num_suspicious_clips": len(suspicious_clips),
        "num_suspicious_spans": len(suspicious_spans),
        "session_summaries": session_summaries,
    }
    return suspicious_clips, suspicious_spans, analysis_summary


def _summarize_span(session_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    true_label = str(rows[0].get("true_label", ""))
    predicted_labels = [str(row.get("predicted_label", "")) for row in rows]
    predicted_label_counts: dict[str, int] = {}
    for label_name in predicted_labels:
        predicted_label_counts[label_name] = predicted_label_counts.get(label_name, 0) + 1
    dominant_predicted_label = max(predicted_label_counts.items(), key=lambda item: item[1])[0]
    clip_tokens = [row["clip_token_int"] for row in rows]
    return {
        "session_id": session_id,
        "true_label": true_label,
        "dominant_predicted_label": dominant_predicted_label,
        "start_clip_token": min(clip_tokens),
        "end_clip_token": max(clip_tokens),
        "num_clips": len(rows),
        "mean_top1_prob": float(np.mean([_to_float(row.get("top1_prob")) for row in rows])),
        "mean_margin_top1_top2": float(np.mean([_to_float(row.get("margin_top1_top2")) for row in rows])),
        "mean_suspicious_score": float(np.mean([float(row["suspicious_score"]) for row in rows])),
        "max_suspicious_score": float(max(float(row["suspicious_score"]) for row in rows)),
        "predicted_label_counts": json.dumps(predicted_label_counts, ensure_ascii=False),
    }


def _build_session_level_report(
    sample_ids: list[str],
    labels: list[int],
    logits_list: list[np.ndarray],
    ordered_labels: list[str],
) -> dict[str, Any]:
    session_to_logits: dict[str, list[np.ndarray]] = {}
    session_to_true: dict[str, int] = {}

    for sample_id, label, logits in zip(sample_ids, labels, logits_list):
        session_id = _derive_session_id(sample_id)
        session_to_logits.setdefault(session_id, []).append(np.asarray(logits, dtype=np.float32))
        session_to_true[session_id] = int(label)

    confusion_matrix = np.zeros((len(ordered_labels), len(ordered_labels)), dtype=np.int64)
    session_predictions: list[dict[str, Any]] = []
    for session_id, logits_blocks in sorted(session_to_logits.items()):
        mean_logits = np.mean(np.stack(logits_blocks, axis=0), axis=0)
        true_index = int(session_to_true[session_id])
        pred_index = int(mean_logits.argmax())
        confusion_matrix[true_index, pred_index] += 1
        session_predictions.append(
            {
                "session_id": session_id,
                "true_label": ordered_labels[true_index],
                "predicted_label": ordered_labels[pred_index],
                "num_clips": len(logits_blocks),
            }
        )

    metrics = _compute_classification_metrics(confusion_matrix, ordered_labels)
    return {
        "num_sessions": len(session_to_logits),
        "aggregation": "mean_logits",
        "confusion_matrix": confusion_matrix.tolist(),
        "predictions": session_predictions,
        **metrics,
    }


def _build_peak_weighted_session_level_report(
    sample_ids: list[str],
    labels: list[int],
    logits_list: list[np.ndarray],
    ordered_labels: list[str],
    top_ratio: float = 0.2,
) -> dict[str, Any]:
    session_to_logits: dict[str, list[np.ndarray]] = {}
    session_to_true: dict[str, int] = {}

    for sample_id, label, logits in zip(sample_ids, labels, logits_list):
        session_id = _derive_session_id(sample_id)
        session_to_logits.setdefault(session_id, []).append(np.asarray(logits, dtype=np.float32))
        session_to_true[session_id] = int(label)

    confusion_matrix = np.zeros((len(ordered_labels), len(ordered_labels)), dtype=np.int64)
    session_predictions: list[dict[str, Any]] = []
    ratio = float(min(max(top_ratio, 0.05), 1.0))
    for session_id, logits_blocks in sorted(session_to_logits.items()):
        logits_array = np.stack(logits_blocks, axis=0)
        probabilities = _softmax_numpy(logits_array)
        aggregated_scores = np.zeros((probabilities.shape[1],), dtype=np.float32)
        for class_index in range(probabilities.shape[1]):
            class_scores = probabilities[:, class_index]
            top_k = max(1, int(np.ceil(len(class_scores) * ratio)))
            top_values = np.sort(class_scores)[-top_k:]
            aggregated_scores[class_index] = float(top_values.mean())
        pred_index = int(aggregated_scores.argmax())
        true_index = int(session_to_true[session_id])
        confusion_matrix[true_index, pred_index] += 1
        session_predictions.append(
            {
                "session_id": session_id,
                "true_label": ordered_labels[true_index],
                "predicted_label": ordered_labels[pred_index],
                "num_clips": len(logits_blocks),
                "aggregation_scores": {ordered_labels[index]: float(aggregated_scores[index]) for index in range(len(ordered_labels))},
            }
        )

    metrics = _compute_classification_metrics(confusion_matrix, ordered_labels)
    return {
        "num_sessions": len(session_to_logits),
        "aggregation": "topk_mean_prob",
        "top_ratio": ratio,
        "confusion_matrix": confusion_matrix.tolist(),
        "predictions": session_predictions,
        **metrics,
    }


def _softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(axis=-1, keepdims=True), a_min=1e-8, a_max=None)




def _infer_num_notes_from_checkpoint(checkpoint: dict[str, Any], fallback: int) -> int:
    explicit = checkpoint.get('num_notes')
    if explicit is not None:
        return int(explicit)
    state_dict = checkpoint.get('model_state_dict', {})
    notes_weight = state_dict.get('notes_head.weight')
    if notes_weight is not None:
        return int(notes_weight.shape[0])
    return int(fallback)

def _load_manifest(manifest_path: Path) -> dict[str, Any] | None:
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _build_coarse_taxonomy(config: TrainingConfig, ordered_labels: Sequence[str]) -> dict[str, int]:
    coarse_name_to_index: dict[str, int] = {}
    for label in ordered_labels:
        coarse_name = str(config.coarse_label_map.get(label, label))
        if coarse_name not in coarse_name_to_index:
            coarse_name_to_index[coarse_name] = len(coarse_name_to_index)
    return coarse_name_to_index


def run_evaluation(
    checkpoint_path: str,
    config_values: dict[str, Any] | None = None,
    session_ids: Sequence[str] | None = None,
    data_root: str | None = None,
    output_dir: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    resolved_device = torch.device(device or "cpu")
    checkpoint = load_checkpoint(checkpoint_path, map_location=resolved_device)
    checkpoint_config = TrainingConfig.from_dict(checkpoint.get("config", {}))

    override_config = dict(config_values or {})
    if data_root is not None:
        override_config["data_root"] = data_root
    merged_config = checkpoint_config.to_dict()
    merged_config.update(override_config)
    config = TrainingConfig.from_dict(merged_config)

    label_to_index = checkpoint.get("label_to_index", {})
    if not label_to_index:
        raise ValueError("Checkpoint does not contain label_to_index.")

    ordered_labels = [label for label, _ in sorted(label_to_index.items(), key=lambda item: item[1])]

    manifest_path = Path(checkpoint_path).parent / "training_session_manifest.json"
    manifest = _load_manifest(manifest_path)
    resolved_session_ids = list(session_ids) if session_ids else None
    if resolved_session_ids is None and manifest is not None:
        test_sessions = manifest.get("test_sessions") or []
        if test_sessions:
            resolved_session_ids = list(test_sessions)

    config.include_track = _env_flag("TRACK", bool(getattr(config, "include_track", True)))
    config.include_rd = _env_flag("RD", bool(config.include_rd))
    dataset = build_dataset(data_root=config.data_root, config=config, session_ids=resolved_session_ids, label_order=ordered_labels)
    track_feature_center = checkpoint.get("track_feature_center")
    track_feature_scale = checkpoint.get("track_feature_scale")
    if track_feature_center is not None and track_feature_scale is not None:
        dataset.set_track_normalization(
            np.asarray(track_feature_center, dtype=np.float32),
            np.asarray(track_feature_scale, dtype=np.float32),
        )
    track_global_center = checkpoint.get("track_global_center")
    track_global_scale = checkpoint.get("track_global_scale")
    if track_global_center is not None and track_global_scale is not None:
        dataset.set_track_global_normalization(
            np.asarray(track_global_center, dtype=np.float32),
            np.asarray(track_global_scale, dtype=np.float32),
        )
    resolved_num_workers = int(getattr(config, "num_workers", 0))
    if os.name == "nt" and resolved_num_workers > 0:
        resolved_num_workers = 0
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=resolved_num_workers)

    coarse_to_index = checkpoint.get("coarse_to_index")
    if not coarse_to_index:
        coarse_to_index = _build_coarse_taxonomy(config, ordered_labels)

    model = FusionRadarClassifier(
        track_input_dim=config.feature_dim,
        track_global_feature_dim=config.global_feature_dim,
        num_classes=len(label_to_index),
        num_coarse_classes=len(coarse_to_index),
        include_track=_env_flag("TRACK", bool(getattr(config, "include_track", True))),
        include_rd=_env_flag("RD", bool(config.include_rd)),
        track_projection_dim=config.track_projection_dim,
        tcn_channels=config.tcn_channels,
        transformer_layers=config.transformer_layers,
        transformer_heads=config.transformer_heads,
        transformer_dropout=config.transformer_dropout,
        max_sequence_length=config.track_long_length,
        short_sequence_length=config.track_short_length,
        mid_sequence_length=config.track_mid_length,
        long_sequence_length=config.track_long_length,
        x3d_in_channels=config.rd_input_channels,
        x3d_stem_channels=config.x3d_stem_channels,
        x3d_stage_channels=config.x3d_stage_channels,
        x3d_stage_depths=config.x3d_stage_depths,
        x3d_spatial_strides=config.x3d_spatial_strides,
        x3d_temporal_strides=config.x3d_temporal_strides,
        x3d_bottleneck_expansion=config.x3d_bottleneck_expansion,
        x3d_se_ratio=config.x3d_se_ratio,
        cross_attention_layers=config.cross_attention_layers,
        cross_attention_heads=config.cross_attention_heads,
        cross_attention_dropout=config.cross_attention_dropout,
        fusion_hidden_size=config.fusion_hidden_size,
        detail_logit_residual_weight=config.detail_logit_residual_weight,
        detail_cosine_scale=config.detail_cosine_scale,
        num_notes=_infer_num_notes_from_checkpoint(checkpoint, getattr(dataset, "note_dim", 0)),
        track_backbone_type=config.track_backbone_type,
        temporal_2d_branch_channels=getattr(config, "temporal_2d_branch_channels", 96),
        bird_class_index=dataset.label_to_index.get("BirdTarget"),
        surface_class_index=dataset.label_to_index.get("ShipTarget"),
        aircraft_class_index=dataset.label_to_index.get("AircraftTarget"),
        drone_class_index=dataset.label_to_index.get("DroneTarget"),
        bird_surface_logit_residual_weight=float(getattr(config, "bird_surface_logit_residual_weight", 0.2)),
        ship_specialist_logit_residual_weight=float(getattr(config, "ship_specialist_logit_residual_weight", 0.0)),
        rd_dominant_fusion_bias=float(getattr(config, "rd_dominant_fusion_bias", 0.35)),
        rd_unimodal_logit_residual_weight=float(getattr(config, "rd_unimodal_logit_residual_weight", 0.18)),
        min_rd_fusion_weight=float(getattr(config, "min_rd_fusion_weight", 0.15)),
        ambiguous_rd_fusion_boost=float(getattr(config, "ambiguous_rd_fusion_boost", 0.15)),
        surface_rd_fusion_boost=float(getattr(config, "surface_rd_fusion_boost", 0.1)),
        use_weighted_fusion_feature=bool(getattr(config, "use_weighted_fusion_feature", False)),
    ).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    logit_bias = checkpoint.get("logit_bias")
    resolved_logit_bias = None if logit_bias is None else torch.tensor(logit_bias, dtype=torch.float32, device=resolved_device)
    pair_calibration = None
    calibration_payload = checkpoint.get('calibration')
    if isinstance(calibration_payload, dict):
        pair_calibration = calibration_payload.get('pair_calibration')
    total_loss = 0.0
    total_samples = 0
    confusion_matrix = np.zeros((len(ordered_labels), len(ordered_labels)), dtype=np.int64)
    collected_sample_ids: list[str] = []
    collected_labels: list[int] = []
    collected_logits: list[np.ndarray] = []

    with torch.no_grad():
        for batch in data_loader:
            track_features = batch["track_features"].to(device=resolved_device, dtype=torch.float32)
            track_global_features = batch["track_global_features"].to(device=resolved_device, dtype=torch.float32)
            track_short_features = batch.get("track_short_features")
            if track_short_features is not None:
                track_short_features = track_short_features.to(device=resolved_device, dtype=torch.float32)
            track_long_features = batch.get("track_long_features")
            if track_long_features is not None:
                track_long_features = track_long_features.to(device=resolved_device, dtype=torch.float32)
            rd_features = None
            rd_mask = None
            if config.include_rd:
                rd_features = batch["rd_features"].to(device=resolved_device, dtype=torch.float32)
                rd_mask = batch["has_rd"].to(device=resolved_device, dtype=torch.float32).view(-1, 1)
            labels = batch["label"].to(device=resolved_device, dtype=torch.long)

            outputs = model(track_features, track_global_features, track_short_features, track_long_features, rd_features, rd_mask)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            logits = apply_logit_bias(logits, resolved_logit_bias)
            logits = _apply_pairwise_calibration(logits, pair_calibration)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=-1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            for expected, predicted in zip(labels.cpu().tolist(), predictions.cpu().tolist()):
                confusion_matrix[expected, predicted] += 1
            collected_sample_ids.extend([str(item) for item in batch["sample_id"]])
            collected_labels.extend(labels.cpu().tolist())
            collected_logits.extend(logits.detach().cpu().numpy())

    metrics = _compute_classification_metrics(confusion_matrix, ordered_labels)
    session_metrics = _build_session_level_report(
        sample_ids=collected_sample_ids,
        labels=collected_labels,
        logits_list=collected_logits,
        ordered_labels=ordered_labels,
    )
    session_peak_metrics = _build_peak_weighted_session_level_report(
        sample_ids=collected_sample_ids,
        labels=collected_labels,
        logits_list=collected_logits,
        ordered_labels=ordered_labels,
    )
    result = {
        "checkpoint_path": checkpoint_path,
        "evaluated_sessions": resolved_session_ids,
        "num_samples": total_samples,
        "loss": total_loss / max(total_samples, 1),
        "labels": ordered_labels,
        "confusion_matrix": confusion_matrix.tolist(),
        "session_level": session_metrics,
        "session_level_peak": session_peak_metrics,
        **metrics,
    }

    export_dir = Path(output_dir) if output_dir is not None else Path(checkpoint_path).parent
    export_dir.mkdir(parents=True, exist_ok=True)
    report_path = export_dir / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    result["report_path"] = str(report_path)
    return result


def run_session_inference(
    checkpoint_path: str,
    config_values: dict[str, Any] | None = None,
    session_ids: Sequence[str] | None = None,
    data_root: str | None = None,
    output_dir: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    evaluation_result = run_evaluation(
        checkpoint_path=checkpoint_path,
        config_values=config_values,
        session_ids=session_ids,
        data_root=data_root,
        output_dir=output_dir,
        device=device,
    )
    export_dir = Path(output_dir) if output_dir is not None else Path(checkpoint_path).parent
    export_dir.mkdir(parents=True, exist_ok=True)
    session_result = {
        "checkpoint_path": evaluation_result["checkpoint_path"],
        "evaluated_sessions": evaluation_result["evaluated_sessions"],
        "labels": evaluation_result["labels"],
        "clip_level": {
            "num_samples": evaluation_result["num_samples"],
            "loss": evaluation_result["loss"],
            "accuracy": evaluation_result["accuracy"],
            "macro_precision": evaluation_result["macro_precision"],
            "macro_recall": evaluation_result["macro_recall"],
            "macro_f1": evaluation_result["macro_f1"],
            "confusion_matrix": evaluation_result["confusion_matrix"],
            "per_class": evaluation_result["per_class"],
        },
        "session_level": evaluation_result["session_level"],
        "session_level_peak": evaluation_result.get("session_level_peak"),
    }
    session_report_path = export_dir / "session_inference_report.json"
    with open(session_report_path, "w", encoding="utf-8") as file:
        json.dump(session_result, file, indent=2, ensure_ascii=False)
    session_result["report_path"] = str(session_report_path)
    return session_result


def run_session_diagnosis(
    checkpoint_path: str,
    config_values: dict[str, Any] | None = None,
    session_ids: Sequence[str] | None = None,
    data_root: str | None = None,
    output_dir: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    resolved_device = torch.device(device or "cpu")
    checkpoint = load_checkpoint(checkpoint_path, map_location=resolved_device)
    checkpoint_config = TrainingConfig.from_dict(checkpoint.get("config", {}))

    override_config = dict(config_values or {})
    if data_root is not None:
        override_config["data_root"] = data_root
    merged_config = checkpoint_config.to_dict()
    merged_config.update(override_config)
    config = TrainingConfig.from_dict(merged_config)

    label_to_index = checkpoint.get("label_to_index", {})
    if not label_to_index:
        raise ValueError("Checkpoint does not contain label_to_index.")
    ordered_labels = [label for label, _ in sorted(label_to_index.items(), key=lambda item: item[1])]
    index_to_label = {index: label for index, label in enumerate(ordered_labels)}

    manifest_path = Path(checkpoint_path).parent / "training_session_manifest.json"
    manifest = _load_manifest(manifest_path)
    resolved_session_ids = list(session_ids) if session_ids else None
    if resolved_session_ids is None and manifest is not None:
        test_sessions = manifest.get("test_sessions") or []
        if test_sessions:
            resolved_session_ids = list(test_sessions)

    config.include_track = _env_flag("TRACK", bool(getattr(config, "include_track", True)))
    config.include_rd = _env_flag("RD", bool(config.include_rd))
    dataset = build_dataset(data_root=config.data_root, config=config, session_ids=resolved_session_ids, label_order=ordered_labels)
    track_feature_center = checkpoint.get("track_feature_center")
    track_feature_scale = checkpoint.get("track_feature_scale")
    if track_feature_center is not None and track_feature_scale is not None:
        dataset.set_track_normalization(
            np.asarray(track_feature_center, dtype=np.float32),
            np.asarray(track_feature_scale, dtype=np.float32),
        )
    track_global_center = checkpoint.get("track_global_center")
    track_global_scale = checkpoint.get("track_global_scale")
    if track_global_center is not None and track_global_scale is not None:
        dataset.set_track_global_normalization(
            np.asarray(track_global_center, dtype=np.float32),
            np.asarray(track_global_scale, dtype=np.float32),
        )

    resolved_num_workers = int(getattr(config, "num_workers", 0))
    if os.name == "nt" and resolved_num_workers > 0:
        resolved_num_workers = 0
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=resolved_num_workers)

    coarse_to_index = checkpoint.get("coarse_to_index")
    if not coarse_to_index:
        coarse_to_index = _build_coarse_taxonomy(config, ordered_labels)

    model = FusionRadarClassifier(
        track_input_dim=config.feature_dim,
        track_global_feature_dim=config.global_feature_dim,
        num_classes=len(label_to_index),
        num_coarse_classes=len(coarse_to_index),
        include_track=_env_flag("TRACK", bool(getattr(config, "include_track", True))),
        include_rd=_env_flag("RD", bool(config.include_rd)),
        track_projection_dim=config.track_projection_dim,
        tcn_channels=config.tcn_channels,
        transformer_layers=config.transformer_layers,
        transformer_heads=config.transformer_heads,
        transformer_dropout=config.transformer_dropout,
        max_sequence_length=config.track_long_length,
        short_sequence_length=config.track_short_length,
        mid_sequence_length=config.track_mid_length,
        long_sequence_length=config.track_long_length,
        x3d_in_channels=config.rd_input_channels,
        x3d_stem_channels=config.x3d_stem_channels,
        x3d_stage_channels=config.x3d_stage_channels,
        x3d_stage_depths=config.x3d_stage_depths,
        x3d_spatial_strides=config.x3d_spatial_strides,
        x3d_temporal_strides=config.x3d_temporal_strides,
        x3d_bottleneck_expansion=config.x3d_bottleneck_expansion,
        x3d_se_ratio=config.x3d_se_ratio,
        cross_attention_layers=config.cross_attention_layers,
        cross_attention_heads=config.cross_attention_heads,
        cross_attention_dropout=config.cross_attention_dropout,
        fusion_hidden_size=config.fusion_hidden_size,
        detail_logit_residual_weight=config.detail_logit_residual_weight,
        detail_cosine_scale=config.detail_cosine_scale,
        num_notes=_infer_num_notes_from_checkpoint(checkpoint, getattr(dataset, "note_dim", 0)),
        track_backbone_type=config.track_backbone_type,
        temporal_2d_branch_channels=getattr(config, "temporal_2d_branch_channels", 96),
        bird_class_index=dataset.label_to_index.get("BirdTarget"),
        surface_class_index=dataset.label_to_index.get("ShipTarget"),
        aircraft_class_index=dataset.label_to_index.get("AircraftTarget"),
        drone_class_index=dataset.label_to_index.get("DroneTarget"),
        bird_surface_logit_residual_weight=float(getattr(config, "bird_surface_logit_residual_weight", 0.2)),
        ship_specialist_logit_residual_weight=float(getattr(config, "ship_specialist_logit_residual_weight", 0.0)),
        rd_dominant_fusion_bias=float(getattr(config, "rd_dominant_fusion_bias", 0.35)),
        rd_unimodal_logit_residual_weight=float(getattr(config, "rd_unimodal_logit_residual_weight", 0.18)),
    ).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    logit_bias = checkpoint.get("logit_bias")
    resolved_logit_bias = None if logit_bias is None else torch.tensor(logit_bias, dtype=torch.float32, device=resolved_device)
    pair_calibration = None
    calibration_payload = checkpoint.get('calibration')
    if isinstance(calibration_payload, dict):
        pair_calibration = calibration_payload.get('pair_calibration')

    clip_rows: list[dict[str, Any]] = []
    collected_sample_ids: list[str] = []
    collected_labels: list[int] = []
    collected_logits: list[np.ndarray] = []

    with torch.no_grad():
        for batch in data_loader:
            track_features = batch["track_features"].to(device=resolved_device, dtype=torch.float32)
            track_global_features = batch["track_global_features"].to(device=resolved_device, dtype=torch.float32)
            track_short_features = batch.get("track_short_features")
            if track_short_features is not None:
                track_short_features = track_short_features.to(device=resolved_device, dtype=torch.float32)
            track_long_features = batch.get("track_long_features")
            if track_long_features is not None:
                track_long_features = track_long_features.to(device=resolved_device, dtype=torch.float32)
            rd_features = None
            rd_mask = None
            if config.include_rd:
                rd_features = batch["rd_features"].to(device=resolved_device, dtype=torch.float32)
                rd_mask = batch["has_rd"].to(device=resolved_device, dtype=torch.float32).view(-1, 1)
            labels = batch["label"].to(device=resolved_device, dtype=torch.long)

            outputs = model(track_features, track_global_features, track_short_features, track_long_features, rd_features, rd_mask)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            logits = apply_logit_bias(logits, resolved_logit_bias)
            logits = _apply_pairwise_calibration(logits, pair_calibration)
            probabilities = torch.softmax(logits, dim=-1)
            predictions = probabilities.argmax(dim=-1)
            track_router_logits = outputs.get("track_router_logits")
            track_router_prob = None if track_router_logits is None else torch.softmax(track_router_logits, dim=-1)
            rd_ship_drone_logits = outputs.get("rd_ship_drone_logits")
            rd_ship_drone_prob = None if rd_ship_drone_logits is None else torch.softmax(rd_ship_drone_logits, dim=-1)
            rd_ship_bird_logits = outputs.get("rd_ship_bird_logits")
            rd_ship_bird_prob = None if rd_ship_bird_logits is None else torch.softmax(rd_ship_bird_logits, dim=-1)
            rd_ship_air_logits = outputs.get("rd_ship_air_logits")
            rd_ship_air_prob = None if rd_ship_air_logits is None else torch.softmax(rd_ship_air_logits, dim=-1)
            bird_surface_logits = outputs.get("bird_surface_logits")
            bird_surface_prob = None if bird_surface_logits is None else torch.sigmoid(bird_surface_logits.view(-1))
            bird_rd_logits = outputs.get("bird_rd_logits")
            bird_rd_prob = None if bird_rd_logits is None else torch.sigmoid(bird_rd_logits.view(-1))
            bird_surface_rd_logits = outputs.get("bird_surface_rd_logits")
            bird_surface_rd_prob = None if bird_surface_rd_logits is None else torch.sigmoid(bird_surface_rd_logits.view(-1))
            track_weight = outputs.get("track_weight")
            rd_weight = outputs.get("rd_weight")

            probabilities_cpu = probabilities.detach().cpu().numpy()
            logits_cpu = logits.detach().cpu().numpy()
            predictions_cpu = predictions.detach().cpu().numpy()
            labels_cpu = labels.detach().cpu().numpy()

            for row_index, sample_id in enumerate(batch["sample_id"]):
                sample_id_text = str(sample_id)
                session_id, track_token, clip_token = _split_sample_id(sample_id_text)
                probability_row = probabilities_cpu[row_index]
                sorted_indices = np.argsort(-probability_row)
                top1_index = int(sorted_indices[0])
                top2_index = int(sorted_indices[1]) if len(sorted_indices) > 1 else top1_index
                clip_row = {
                    "session_id": session_id,
                    "sample_id": sample_id_text,
                    "track_token": track_token,
                    "clip_token": clip_token,
                    "true_label": index_to_label[int(labels_cpu[row_index])],
                    "predicted_label": index_to_label[int(predictions_cpu[row_index])],
                    "top1_label": index_to_label[top1_index],
                    "top1_prob": float(probability_row[top1_index]),
                    "top2_label": index_to_label[top2_index],
                    "top2_prob": float(probability_row[top2_index]),
                    "margin_top1_top2": float(probability_row[top1_index] - probability_row[top2_index]),
                    "track_weight": None if track_weight is None else float(track_weight[row_index].detach().cpu().view(-1)[0]),
                    "rd_weight": None if rd_weight is None else float(rd_weight[row_index].detach().cpu().view(-1)[0]),
                    "router_surface_prob": None if track_router_prob is None else float(track_router_prob[row_index, 0].detach().cpu()),
                    "router_aircraft_prob": None if track_router_prob is None else float(track_router_prob[row_index, 1].detach().cpu()),
                    "router_ambiguous_prob": None if track_router_prob is None else float(track_router_prob[row_index, 2].detach().cpu()),
                    "ship_drone_ship_prob": None if rd_ship_drone_prob is None else float(rd_ship_drone_prob[row_index, 1].detach().cpu()),
                    "ship_bird_ship_prob": None if rd_ship_bird_prob is None else float(rd_ship_bird_prob[row_index, 1].detach().cpu()),
                    "ship_air_ship_prob": None if rd_ship_air_prob is None else float(rd_ship_air_prob[row_index, 1].detach().cpu()),
                    "bird_surface_prob": None if bird_surface_prob is None else float(bird_surface_prob[row_index].detach().cpu()),
                    "bird_rd_prob": None if bird_rd_prob is None else float(bird_rd_prob[row_index].detach().cpu()),
                    "bird_surface_rd_prob": None if bird_surface_rd_prob is None else float(bird_surface_rd_prob[row_index].detach().cpu()),
                }
                for class_index, label_name in enumerate(ordered_labels):
                    clip_row[f"prob_{label_name}"] = float(probability_row[class_index])
                    clip_row[f"logit_{label_name}"] = float(logits_cpu[row_index, class_index])
                clip_rows.append(clip_row)
                collected_sample_ids.append(sample_id_text)
                collected_labels.append(int(labels_cpu[row_index]))
                collected_logits.append(logits_cpu[row_index])

    session_rows_map: dict[str, dict[str, Any]] = {}
    for clip_row in clip_rows:
        session_id = str(clip_row["session_id"])
        session_entry = session_rows_map.get(session_id)
        if session_entry is None:
            session_entry = {
                "session_id": session_id,
                "true_label": clip_row["true_label"],
                "num_clips": 0,
                "num_correct": 0,
                "pred_counts": {label_name: 0 for label_name in ordered_labels},
                "prob_sums": {label_name: 0.0 for label_name in ordered_labels},
                "margin_sum": 0.0,
            }
            session_rows_map[session_id] = session_entry
        session_entry["num_clips"] += 1
        session_entry["num_correct"] += int(clip_row["true_label"] == clip_row["predicted_label"])
        session_entry["pred_counts"][str(clip_row["predicted_label"])] += 1
        for label_name in ordered_labels:
            session_entry["prob_sums"][label_name] += float(clip_row[f"prob_{label_name}"])
        session_entry["margin_sum"] += float(clip_row["margin_top1_top2"])

    session_summary_rows: list[dict[str, Any]] = []
    for session_id, session_entry in session_rows_map.items():
        num_clips = max(int(session_entry["num_clips"]), 1)
        mean_probs = {label_name: session_entry["prob_sums"][label_name] / num_clips for label_name in ordered_labels}
        mean_pred_label = max(mean_probs.items(), key=lambda item: item[1])[0]
        clip_vote_label = max(session_entry["pred_counts"].items(), key=lambda item: item[1])[0]
        error_count = num_clips - int(session_entry["num_correct"])
        summary_row = {
            "session_id": session_id,
            "true_label": session_entry["true_label"],
            "num_clips": num_clips,
            "num_correct_clips": int(session_entry["num_correct"]),
            "num_error_clips": error_count,
            "clip_accuracy": float(session_entry["num_correct"] / num_clips),
            "clip_vote_label": clip_vote_label,
            "mean_logit_label": mean_pred_label,
            "mean_margin_top1_top2": float(session_entry["margin_sum"] / num_clips),
        }
        for label_name in ordered_labels:
            summary_row[f"pred_count_{label_name}"] = int(session_entry["pred_counts"][label_name])
            summary_row[f"mean_prob_{label_name}"] = float(mean_probs[label_name])
        session_summary_rows.append(summary_row)

    session_summary_rows.sort(key=lambda item: (item["clip_accuracy"], item["num_error_clips"], item["session_id"]))
    session_mean_metrics = _build_session_level_report(
        sample_ids=collected_sample_ids,
        labels=collected_labels,
        logits_list=collected_logits,
        ordered_labels=ordered_labels,
    )
    session_peak_metrics = _build_peak_weighted_session_level_report(
        sample_ids=collected_sample_ids,
        labels=collected_labels,
        logits_list=collected_logits,
        ordered_labels=ordered_labels,
    )

    export_dir = Path(output_dir) if output_dir is not None else Path(checkpoint_path).parent
    export_dir.mkdir(parents=True, exist_ok=True)
    clip_csv_path = export_dir / "session_clip_predictions.csv"
    session_csv_path = export_dir / "session_summary.csv"
    report_path = export_dir / "session_diagnosis_report.json"
    _write_csv_rows(clip_csv_path, clip_rows)
    _write_csv_rows(session_csv_path, session_summary_rows)

    diagnosis_result = {
        "checkpoint_path": checkpoint_path,
        "evaluated_sessions": resolved_session_ids,
        "labels": ordered_labels,
        "num_clips": len(clip_rows),
        "num_sessions": len(session_summary_rows),
        "session_level": session_mean_metrics,
        "session_level_peak": session_peak_metrics,
        "session_summaries": session_summary_rows,
        "hardest_sessions": session_summary_rows[: min(10, len(session_summary_rows))],
        "clip_csv_path": str(clip_csv_path),
        "session_csv_path": str(session_csv_path),
    }
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(diagnosis_result, file, indent=2, ensure_ascii=False)
    diagnosis_result["report_path"] = str(report_path)
    return diagnosis_result


def run_hard_session_mining(
    checkpoint_path: str,
    config_values: dict[str, Any] | None = None,
    session_ids: Sequence[str] | None = None,
    data_root: str | None = None,
    output_dir: str | None = None,
    device: str | None = None,
    top_k: int = 3,
    min_error_clips: int = 1,
) -> dict[str, Any]:
    export_dir = Path(output_dir) if output_dir is not None else Path(checkpoint_path).parent / "hard_session_mining"
    export_dir.mkdir(parents=True, exist_ok=True)

    base_dir = export_dir / "all_sessions"
    diagnosis_result = run_session_diagnosis(
        checkpoint_path=checkpoint_path,
        config_values=config_values,
        session_ids=session_ids,
        data_root=data_root,
        output_dir=str(base_dir),
        device=device,
    )

    summaries = list(diagnosis_result.get("session_summaries", []))
    ranked_sessions = [
        row for row in summaries
        if int(row.get("num_error_clips", 0)) >= int(min_error_clips)
    ]
    ranked_sessions.sort(key=lambda item: (-int(item.get("num_error_clips", 0)), float(item.get("clip_accuracy", 0.0)), str(item.get("session_id", ""))))
    selected_rows = ranked_sessions[: max(int(top_k), 0)]
    selected_session_ids = [str(row["session_id"]) for row in selected_rows]

    focused_result = None
    focused_dir = export_dir / "focused_sessions"
    if selected_session_ids:
        focused_result = run_session_diagnosis(
            checkpoint_path=checkpoint_path,
            config_values=config_values,
            session_ids=selected_session_ids,
            data_root=data_root,
            output_dir=str(focused_dir),
            device=device,
        )

    suspicious_clips_csv = None
    suspicious_spans_csv = None
    suspicious_report_path = None
    suspicious_analysis = None
    if focused_result is not None:
        focused_clip_rows = _load_csv_rows(Path(str(focused_result["clip_csv_path"])))
        suspicious_clips, suspicious_spans, suspicious_analysis = _analyze_clip_rows(focused_clip_rows)
        suspicious_clips_csv = focused_dir / "suspicious_clips.csv"
        suspicious_spans_csv = focused_dir / "suspicious_spans.csv"
        suspicious_report_path = focused_dir / "suspicious_analysis_report.json"
        _write_csv_rows(suspicious_clips_csv, suspicious_clips)
        _write_csv_rows(suspicious_spans_csv, suspicious_spans)
        with suspicious_report_path.open("w", encoding="utf-8") as file:
            json.dump(suspicious_analysis, file, indent=2, ensure_ascii=False)

    mining_result = {
        "checkpoint_path": checkpoint_path,
        "top_k": int(top_k),
        "min_error_clips": int(min_error_clips),
        "selected_sessions": selected_rows,
        "all_sessions_report": diagnosis_result.get("report_path"),
        "all_sessions_csv": diagnosis_result.get("session_csv_path"),
        "all_clips_csv": diagnosis_result.get("clip_csv_path"),
        "focused_sessions_report": None if focused_result is None else focused_result.get("report_path"),
        "focused_sessions_csv": None if focused_result is None else focused_result.get("session_csv_path"),
        "focused_clips_csv": None if focused_result is None else focused_result.get("clip_csv_path"),
        "focused_suspicious_clips_csv": None if suspicious_clips_csv is None else str(suspicious_clips_csv),
        "focused_suspicious_spans_csv": None if suspicious_spans_csv is None else str(suspicious_spans_csv),
        "focused_suspicious_report": None if suspicious_report_path is None else str(suspicious_report_path),
        "focused_suspicious_summary": suspicious_analysis,
    }
    report_path = export_dir / "hard_session_mining_report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(mining_result, file, indent=2, ensure_ascii=False)
    mining_result["report_path"] = str(report_path)
    return mining_result
