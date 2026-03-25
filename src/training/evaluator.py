import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.fusion_classifier import FusionRadarClassifier
from src.training.config import TrainingConfig
from src.training.dataset import RadarFusionDataset
from src.training.trainer import apply_logit_bias


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
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
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

    dataset = RadarFusionDataset(
        data_root=config.data_root,
        session_ids=resolved_session_ids,
        label_order=ordered_labels,
        sequence_length=config.sequence_length,
        rd_frames=config.rd_frames,
        rd_height=config.rd_height,
        rd_width=config.rd_width,
        rd_input_channels=config.rd_input_channels,
        feature_columns=config.track_feature_columns,
        global_feature_columns=config.track_global_feature_columns,
        include_rd=config.include_rd,
    )
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
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    coarse_to_index = checkpoint.get("coarse_to_index")
    if not coarse_to_index:
        coarse_to_index = _build_coarse_taxonomy(config, ordered_labels)

    model = FusionRadarClassifier(
        track_input_dim=config.feature_dim,
        track_global_feature_dim=config.global_feature_dim,
        num_classes=len(label_to_index),
        num_coarse_classes=len(coarse_to_index),
        include_rd=config.include_rd,
        track_projection_dim=config.track_projection_dim,
        tcn_channels=config.tcn_channels,
        transformer_layers=config.transformer_layers,
        transformer_heads=config.transformer_heads,
        transformer_dropout=config.transformer_dropout,
        max_sequence_length=config.sequence_length,
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
    ).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    logit_bias = checkpoint.get("logit_bias")
    resolved_logit_bias = None if logit_bias is None else torch.tensor(logit_bias, dtype=torch.float32, device=resolved_device)
    total_loss = 0.0
    total_samples = 0
    confusion_matrix = np.zeros((len(ordered_labels), len(ordered_labels)), dtype=np.int64)

    with torch.no_grad():
        for batch in data_loader:
            track_features = batch["track_features"].to(device=resolved_device, dtype=torch.float32)
            track_global_features = batch["track_global_features"].to(device=resolved_device, dtype=torch.float32)
            rd_features = None
            rd_mask = None
            if config.include_rd:
                rd_features = batch["rd_features"].to(device=resolved_device, dtype=torch.float32)
                rd_mask = batch["has_rd"].to(device=resolved_device, dtype=torch.float32).view(-1, 1)
            labels = batch["label"].to(device=resolved_device, dtype=torch.long)

            outputs = model(track_features, track_global_features, rd_features, rd_mask)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            logits = apply_logit_bias(logits, resolved_logit_bias)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=-1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            for expected, predicted in zip(labels.cpu().tolist(), predictions.cpu().tolist()):
                confusion_matrix[expected, predicted] += 1

    metrics = _compute_classification_metrics(confusion_matrix, ordered_labels)
    result = {
        "checkpoint_path": checkpoint_path,
        "evaluated_sessions": resolved_session_ids,
        "num_samples": total_samples,
        "loss": total_loss / max(total_samples, 1),
        "labels": ordered_labels,
        "confusion_matrix": confusion_matrix.tolist(),
        **metrics,
    }

    export_dir = Path(output_dir) if output_dir is not None else Path(checkpoint_path).parent
    export_dir.mkdir(parents=True, exist_ok=True)
    report_path = export_dir / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    result["report_path"] = str(report_path)
    return result
