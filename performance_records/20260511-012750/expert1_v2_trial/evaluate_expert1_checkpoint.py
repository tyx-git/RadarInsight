from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.configmanager import load_expert1_config
from src.core.device import resolve_cuda_device
from src.core.expert1_data import build_expert1_dataloaders
from src.core.expert1_trainer import (
    _build_class_weights,
    _build_rd_class_weights,
    _count_trainable_parameters,
    _rd_coverage,
    _run_epoch,
)
from src.core.experts import LOW_ALTITUDE_CLASSES, LowAltitudeSuspectExpert
from src.core.physical import PHYSICAL_FEATURE_COLUMNS
from src.core.trajectory import TrackRepresentationBuilder


def _tensor_to_json(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _tensor_to_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_tensor_to_json(item) for item in value]
    return value


def _public_metrics(metrics: dict[str, object]) -> dict[str, object]:
    excluded = {
        "logits",
        "traj_logits",
        "rd_logits",
        "labels",
        "sample_ids",
        "source_names",
        "label_names",
        "has_rd",
        "rd_activated",
    }
    return {key: _tensor_to_json(value) for key, value in metrics.items() if key not in excluded}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["val", "test", "both"], default="both")
    args = parser.parse_args()

    config = load_expert1_config(args.config)
    device, device_info = resolve_cuda_device(config.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)

    loaders = build_expert1_dataloaders(
        data_root=config.data_root,
        batch_size=config.batch_size,
        seed=config.seed,
        num_workers=config.num_workers,
        rd_sample_boost=config.rd_sample_boost,
        raw_source_boost=config.raw_source_boost,
        excel_source_boost=config.excel_source_boost,
        cq08_source_boost=config.cq08_source_boost,
        source_balance_power=config.source_balance_power,
        source_label_balance_power=config.source_label_balance_power,
        max_sampler_weight=config.max_sampler_weight,
        unknown_sample_boost=config.unknown_sample_boost,
        clutter_sample_boost=config.clutter_sample_boost,
        bird_sample_boost=config.bird_sample_boost,
    )
    representation_builder = TrackRepresentationBuilder(
        sequence_input_dim=10,
        physical_input_dim=len(PHYSICAL_FEATURE_COLUMNS),
        model_dim=128,
        dropout=config.dropout,
    ).to(device)
    rd_dim = 16
    rd_proj = nn.Sequential(
        nn.Linear(6, rd_dim),
        nn.GELU(),
        nn.Dropout(config.dropout),
        nn.Linear(rd_dim, rd_dim),
        nn.LayerNorm(rd_dim),
    ).to(device)
    model = LowAltitudeSuspectExpert(
        input_dim=128 + 32,
        rd_input_dim=4,
        rd_patch_input_dim=3,
        rd_context_dim=rd_dim,
        use_rd_patch_cnn=config.use_rd_patch_cnn,
        dropout=config.dropout,
        cosface_scale=config.cosface_scale,
        cosface_margin=config.cosface_margin,
        reject_threshold=config.reject_threshold,
        rd_activation_threshold=config.rd_activation_threshold,
        rd_bird_logit_scale=config.rd_bird_logit_scale,
        rd_unknown_logit_scale=config.rd_unknown_logit_scale,
        rd_clutter_logit_scale=config.rd_clutter_logit_scale,
        rd_residual_logit_scale=config.rd_residual_logit_scale,
        rd_unknown_clutter_logit_scale=config.rd_unknown_clutter_logit_scale,
        rd_clutter_morphology_logit_scale=config.rd_clutter_morphology_logit_scale,
        initial_rd_gate_bias=config.low_altitude_initial_rd_gate_bias,
        rd_class_activation_threshold=config.low_altitude_rd_class_activation_threshold,
    ).to(device)

    representation_builder.load_state_dict(checkpoint["representation_builder_state_dict"])
    rd_proj.load_state_dict(checkpoint["rd_proj_state_dict"])
    model.load_state_dict(checkpoint["model_state_dict"])

    class_weights = _build_class_weights(loaders.train_loader).to(device)
    rd_class_weights = _build_rd_class_weights(loaders.train_loader).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    source_feature_stats = checkpoint.get("source_feature_stats")

    split_map = {}
    if args.split in {"val", "both"}:
        split_map["val"] = loaders.val_loader
    if args.split in {"test", "both"}:
        split_map["test"] = loaders.test_loader

    metrics = {}
    for split_name, loader in split_map.items():
        raw = _run_epoch(
            representation_builder,
            rd_proj,
            model,
            loader,
            criterion,
            optimizer=None,
            device=device,
            train=False,
            config=config,
            stage2_enabled=True,
            stage2_scale=1.0,
            stage2_epoch=max(1, int(config.epochs)),
            rd_class_weights=rd_class_weights,
            source_feature_stats=source_feature_stats,
        )
        metrics[split_name] = _public_metrics(raw)

    output = {
        "checkpoint": str(Path(args.checkpoint).as_posix()),
        "config": str(Path(args.config).as_posix()),
        "device": device_info,
        "labels": LOW_ALTITUDE_CLASSES,
        "checkpoint_selection_reason": checkpoint.get("selection_reason"),
        "checkpoint_selection_eligible": checkpoint.get("selection_eligible"),
        "checkpoint_config": checkpoint.get("config", {}),
        "trainable_parameters": _count_trainable_parameters(representation_builder)
        + _count_trainable_parameters(rd_proj)
        + _count_trainable_parameters(model),
        "split_sizes": {
            "train": len(loaders.split.train_ids),
            "val": len(loaders.split.val_ids),
            "test": len(loaders.split.test_ids),
        },
        "rd_coverage": {
            "train": _rd_coverage(loaders.train_loader),
            "val": _rd_coverage(loaders.val_loader),
            "test": _rd_coverage(loaders.test_loader),
        },
        "metrics": metrics,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output_path.as_posix()),
        "checkpoint": output["checkpoint"],
        "metrics": {
            name: {
                "accuracy": item["accuracy"],
                "macro_f1": item["macro_f1"],
                "traj_macro_f1": item["traj_macro_f1"],
                "joint_macro_f1_on_rd": item["joint_macro_f1_on_rd"],
                "traj_macro_f1_on_rd": item["traj_macro_f1_on_rd"],
                "joint_minus_traj_macro_f1_on_rd": item["joint_minus_traj_macro_f1_on_rd"],
                "final_net_gain_rate_on_rd": item["final_net_gain_rate_on_rd"],
            }
            for name, item in metrics.items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
