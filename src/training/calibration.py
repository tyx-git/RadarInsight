import json
import os
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.fusion_classifier import FusionRadarClassifier
from src.training.config import TrainingConfig
from src.training.session_manager import build_dataset
from src.training.trainer import build_coarse_taxonomy, compute_classification_metrics, optimize_group_logit_bias


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}




def _infer_num_notes_from_checkpoint(checkpoint: dict[str, Any], fallback: int) -> int:
    explicit = checkpoint.get('num_notes')
    if explicit is not None:
        return int(explicit)
    state_dict = checkpoint.get('model_state_dict', {})
    notes_weight = state_dict.get('notes_head.weight')
    if notes_weight is not None:
        return int(notes_weight.shape[0])
    return int(fallback)

def _load_json(path: Path) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def _build_dataset(
    config: TrainingConfig,
    checkpoint: dict[str, Any],
    ordered_labels: Sequence[str],
    session_ids: Sequence[str],
) -> Any:
    config.include_track = _env_flag("TRACK", bool(getattr(config, "include_track", True)))
    config.include_rd = _env_flag("RD", bool(config.include_rd))
    dataset = build_dataset(data_root=config.data_root, config=config, session_ids=session_ids, label_order=ordered_labels)
    track_feature_center = checkpoint.get('track_feature_center')
    track_feature_scale = checkpoint.get('track_feature_scale')
    if track_feature_center is not None and track_feature_scale is not None:
        dataset.set_track_normalization(
            np.asarray(track_feature_center, dtype=np.float32),
            np.asarray(track_feature_scale, dtype=np.float32),
        )
    track_global_center = checkpoint.get('track_global_center')
    track_global_scale = checkpoint.get('track_global_scale')
    if track_global_center is not None and track_global_scale is not None:
        dataset.set_track_global_normalization(
            np.asarray(track_global_center, dtype=np.float32),
            np.asarray(track_global_scale, dtype=np.float32),
        )
    return dataset


def _build_model(
    config: TrainingConfig,
    checkpoint: dict[str, Any],
    num_classes: int,
    num_coarse_classes: int,
    num_notes: int,
    device: torch.device,
) -> FusionRadarClassifier:
    model = FusionRadarClassifier(
        track_input_dim=config.feature_dim,
        track_global_feature_dim=config.global_feature_dim,
        num_classes=num_classes,
        num_coarse_classes=num_coarse_classes,
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
        num_notes=_infer_num_notes_from_checkpoint(checkpoint, num_notes),
        track_backbone_type=config.track_backbone_type,
        temporal_2d_branch_channels=getattr(config, "temporal_2d_branch_channels", 96),
        bird_class_index=checkpoint.get("label_to_index", {}).get("BirdTarget"),
        surface_class_index=checkpoint.get("label_to_index", {}).get("ShipTarget"),
        aircraft_class_index=checkpoint.get("label_to_index", {}).get("AircraftTarget"),
        drone_class_index=checkpoint.get("label_to_index", {}).get("DroneTarget"),
        bird_surface_logit_residual_weight=float(getattr(config, "bird_surface_logit_residual_weight", 0.2)),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    return model


def _collect_logits(
    model: FusionRadarClassifier,
    data_loader: DataLoader,
    device: torch.device,
    include_rd: bool,
) -> tuple[np.ndarray, np.ndarray]:
    logits_blocks: list[np.ndarray] = []
    labels: list[int] = []
    with torch.no_grad():
        for batch in data_loader:
            track_features = batch['track_features'].to(device=device, dtype=torch.float32)
            track_global_features = batch['track_global_features'].to(device=device, dtype=torch.float32)
            rd_features = None
            rd_mask = None
            if include_rd:
                rd_features = batch['rd_features'].to(device=device, dtype=torch.float32)
                rd_mask = batch['has_rd'].to(device=device, dtype=torch.float32).view(-1, 1)
            track_short_features = batch.get("track_short_features")
            if track_short_features is not None:
                track_short_features = track_short_features.to(device=device, dtype=torch.float32)
            track_long_features = batch.get("track_long_features")
            if track_long_features is not None:
                track_long_features = track_long_features.to(device=device, dtype=torch.float32)
            outputs = model(track_features, track_global_features, track_short_features, track_long_features, rd_features, rd_mask)
            logits_blocks.append(outputs['logits'].detach().cpu().numpy())
            labels.extend(batch['label'].detach().cpu().tolist())
    return np.concatenate(logits_blocks, axis=0), np.asarray(labels, dtype=np.int64)


def run_group_calibration(
    checkpoint_path: str,
    output_dir: str,
    data_root: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    source_checkpoint_path = Path(checkpoint_path)
    source_artifact_dir = source_checkpoint_path.parent
    target_artifact_dir = Path(output_dir)
    target_artifact_dir.mkdir(parents=True, exist_ok=True)

    resolved_device = torch.device(device or 'cpu')
    checkpoint = torch.load(source_checkpoint_path, map_location=resolved_device)
    config = TrainingConfig.from_dict(checkpoint.get('config', {}))
    if data_root is not None:
        config = TrainingConfig.from_dict({**config.to_dict(), 'data_root': data_root})

    label_to_index = checkpoint.get('label_to_index', {})
    if not label_to_index:
        raise ValueError('Checkpoint does not contain label_to_index.')
    ordered_labels = [label for label, _ in sorted(label_to_index.items(), key=lambda item: item[1])]

    coarse_to_index, fine_to_coarse_index = build_coarse_taxonomy(config, ordered_labels)
    manifest_path = source_artifact_dir / 'training_session_manifest.json'
    manifest = _load_json(manifest_path)
    val_sessions = manifest.get('val_sessions') or []
    if not val_sessions:
        raise ValueError('Calibration requires validation sessions in the manifest.')

    val_dataset = _build_dataset(config, checkpoint, ordered_labels, val_sessions)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = _build_model(
        config=config,
        checkpoint=checkpoint,
        num_classes=len(label_to_index),
        num_coarse_classes=len(coarse_to_index),
        num_notes=int(checkpoint.get("num_notes", 0)),
        device=resolved_device,
    )
    val_logits, val_labels = _collect_logits(model, val_loader, resolved_device, config.include_rd)

    base_val_metrics = compute_classification_metrics(
        val_labels.tolist(),
        val_logits.argmax(axis=1).tolist(),
        num_classes=len(label_to_index),
    )
    logit_bias = optimize_group_logit_bias(
        labels=val_labels,
        logits=val_logits,
        fine_to_coarse_index=fine_to_coarse_index.numpy(),
    )
    biased_val_metrics = compute_classification_metrics(
        val_labels.tolist(),
        (val_logits + logit_bias.reshape(1, -1)).argmax(axis=1).tolist(),
        num_classes=len(label_to_index),
    )

    for filename in ['training_history.json', 'training_metadata.json', 'training_session_manifest.json']:
        source_path = source_artifact_dir / filename
        if source_path.exists():
            shutil.copy2(source_path, target_artifact_dir / filename)

    checkpoint['coarse_to_index'] = coarse_to_index
    checkpoint['fine_to_coarse_index'] = fine_to_coarse_index.tolist()
    checkpoint['logit_bias'] = logit_bias.tolist()
    checkpoint['calibration'] = {
        'source_artifact': str(source_artifact_dir),
        'validation_sessions': list(val_sessions),
        'bias': logit_bias.tolist(),
        'base_val_metrics': base_val_metrics,
        'biased_val_metrics': biased_val_metrics,
    }
    torch.save(checkpoint, target_artifact_dir / 'best_fusion_model.pt')

    metadata_path = target_artifact_dir / 'training_metadata.json'
    if metadata_path.exists():
        metadata = _load_json(metadata_path)
    else:
        metadata = {'config': config.to_dict(), 'labels': {index: label for index, label in enumerate(ordered_labels)}}
    metadata['coarse_to_index'] = coarse_to_index
    metadata['fine_to_coarse_index'] = fine_to_coarse_index.tolist()
    metadata['calibration'] = checkpoint['calibration']
    with open(metadata_path, 'w', encoding='utf-8') as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    calibration_report = {
        'output_dir': str(target_artifact_dir),
        'bias': logit_bias.tolist(),
        'base_val_metrics': base_val_metrics,
        'biased_val_metrics': biased_val_metrics,
    }
    with open(target_artifact_dir / 'calibration_report.json', 'w', encoding='utf-8') as file:
        json.dump(calibration_report, file, indent=2, ensure_ascii=False)

    return calibration_report
