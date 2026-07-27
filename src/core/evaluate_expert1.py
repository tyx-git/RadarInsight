from __future__ import annotations

import argparse
import json
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .expert1_config import Expert1TrainConfig
from .expert1_trainer import _build_class_weights_from_index
from .expert1_trainer import _build_expert1_loaders
from .expert1_trainer import _build_expert1_modules
from .expert1_trainer import _build_rd_class_weights_from_index
from .expert1_trainer import _checkpoint_eval_runtime
from .expert1_trainer import _load_checkpoint_into_modules
from .expert1_trainer import _parent_weak_class_metrics
from .expert1_trainer import _parent_window_aggregation_metrics
from .expert1_trainer import _rd_joint_bottleneck_summary
from .expert1_trainer import _run_epoch
from .expert1_trainer import _load_window_metadata
from .expert1_trainer import resolve_device
from .metrics import compute_classification_metrics
from .experts import LOW_ALTITUDE_CLASSES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one Expert1 checkpoint using its checkpoint-saved runtime configuration. "
            "The command never selects a checkpoint, calibration, or release status."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=["val", "test", "both"], default="val")
    parser.add_argument("--device", default="", help="Optional device override, for example cuda or cpu.")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=0,
        help="Grouped bootstrap replicate count for evaluation uncertainty; 0 disables it.",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def _config_from_checkpoint(
    checkpoint_path: Path,
    *,
    device_override: str = "",
    num_workers: int | None = None,
) -> Expert1TrainConfig:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    raw_config = checkpoint.get("config", {})
    if not isinstance(raw_config, dict):
        raise ValueError("checkpoint config is missing or invalid")
    allowed = {item.name for item in fields(Expert1TrainConfig)}
    values = {key: value for key, value in raw_config.items() if key in allowed}
    config = Expert1TrainConfig(**values)
    if num_workers is not None:
        normalized_workers = max(0, int(num_workers))
        config = replace(
            config,
            num_workers=normalized_workers,
            eval_num_workers=normalized_workers,
            persistent_workers=bool(config.persistent_workers and normalized_workers > 0),
        )
    if device_override:
        config = replace(config, device=str(device_override))
    return config


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    return value


def _parent_group_id(sample_id: str) -> str:
    normalized = str(sample_id)
    return normalized.split("|win", 1)[0] if "|win" in normalized else normalized


def _safe_subset_accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    if logits.numel() == 0 or labels.numel() == 0 or not bool(mask.any()):
        return 0.0
    return float((logits[mask].argmax(dim=-1) == labels[mask]).float().mean().item())


def _per_class_recall(logits: torch.Tensor, labels: torch.Tensor, class_index: int) -> float:
    mask = labels == int(class_index)
    if not bool(mask.any()):
        return 0.0
    return float((logits[mask].argmax(dim=-1) == int(class_index)).float().mean().item())


def _bootstrap_interval(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _grouped_bootstrap_summary(
    metrics: dict[str, object],
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    if replicates <= 0:
        return {"enabled": False, "replicates": 0}
    required = ("logits", "traj_logits", "labels", "sample_ids", "source_names", "has_rd")
    if any(key not in metrics for key in required):
        raise ValueError("metrics are missing grouped bootstrap inputs")
    logits = metrics["logits"]
    traj_logits = metrics["traj_logits"]
    labels = metrics["labels"]
    sample_ids = metrics["sample_ids"]
    source_names = metrics["source_names"]
    has_rd = metrics["has_rd"]
    if not isinstance(logits, torch.Tensor) or not isinstance(traj_logits, torch.Tensor):
        raise TypeError("bootstrap logits must be tensors")
    if not isinstance(labels, torch.Tensor):
        raise TypeError("bootstrap labels must be a tensor")
    if not isinstance(sample_ids, list) or not isinstance(source_names, list) or not isinstance(has_rd, list):
        raise TypeError("bootstrap metadata must be lists")
    sample_count = int(labels.shape[0])
    if not (
        logits.shape[0] == sample_count
        and traj_logits.shape[0] == sample_count
        and len(sample_ids) == sample_count
        and len(source_names) == sample_count
        and len(has_rd) == sample_count
    ):
        raise ValueError("bootstrap metadata length mismatch")
    groups: dict[tuple[str, str], list[int]] = {}
    for index, (sample_id, source_name) in enumerate(zip(sample_ids, source_names)):
        key = (str(source_name), _parent_group_id(str(sample_id)))
        groups.setdefault(key, []).append(index)
    group_indices = list(groups.values())
    if not group_indices:
        return {"enabled": False, "replicates": 0, "reason": "no_groups"}
    random = np.random.default_rng(int(seed))
    unknown_index = LOW_ALTITUDE_CLASSES.index("UnknownTarget")
    clutter_index = LOW_ALTITUDE_CLASSES.index("ClutterTarget")
    distributions: dict[str, list[float]] = {
        "accuracy": [],
        "macro_f1": [],
        "macro_recall": [],
        "source_balanced_macro_f1": [],
        "without_rd_accuracy": [],
        "final_unknown_recall": [],
        "final_clutter_recall": [],
        "final_net_gain_rate_on_rd": [],
        "final_harm_rate_on_rd": [],
    }
    group_count = len(group_indices)
    for _ in range(int(replicates)):
        selected_group_indices = random.integers(0, group_count, size=group_count)
        selected_indices = [
            item
            for group_index in selected_group_indices.tolist()
            for item in group_indices[int(group_index)]
        ]
        index_tensor = torch.tensor(selected_indices, dtype=torch.long)
        sampled_logits = logits[index_tensor]
        sampled_traj_logits = traj_logits[index_tensor]
        sampled_labels = labels[index_tensor]
        sampled_has_rd = torch.tensor([bool(has_rd[index]) for index in selected_indices], dtype=torch.bool)
        sampled_sources = [str(source_names[index]) for index in selected_indices]
        final_metrics = compute_classification_metrics(
            sampled_logits,
            sampled_labels,
            num_classes=len(LOW_ALTITUDE_CLASSES),
        )
        distributions["accuracy"].append(float(final_metrics.accuracy))
        distributions["macro_f1"].append(float(final_metrics.macro_f1))
        distributions["macro_recall"].append(float(final_metrics.macro_recall))
        source_f1_values = []
        for source_name in sorted(set(sampled_sources)):
            source_mask = torch.tensor(
                [source == source_name for source in sampled_sources],
                dtype=torch.bool,
            )
            source_f1_values.append(
                compute_classification_metrics(
                    sampled_logits[source_mask],
                    sampled_labels[source_mask],
                    num_classes=len(LOW_ALTITUDE_CLASSES),
                ).macro_f1
            )
        distributions["source_balanced_macro_f1"].append(
            float(np.mean(source_f1_values)) if source_f1_values else 0.0
        )
        distributions["without_rd_accuracy"].append(
            _safe_subset_accuracy(sampled_logits, sampled_labels, ~sampled_has_rd)
        )
        distributions["final_unknown_recall"].append(
            _per_class_recall(sampled_logits, sampled_labels, unknown_index)
        )
        distributions["final_clutter_recall"].append(
            _per_class_recall(sampled_logits, sampled_labels, clutter_index)
        )
        if bool(sampled_has_rd.any()):
            final_pred = sampled_logits.argmax(dim=-1)
            traj_pred = sampled_traj_logits.argmax(dim=-1)
            final_correct = final_pred == sampled_labels
            traj_correct = traj_pred == sampled_labels
            rescues = int((final_correct & ~traj_correct & sampled_has_rd).sum().item())
            harms = int((~final_correct & traj_correct & sampled_has_rd).sum().item())
            rd_support = int(sampled_has_rd.sum().item())
            distributions["final_net_gain_rate_on_rd"].append((rescues - harms) / rd_support)
            distributions["final_harm_rate_on_rd"].append(harms / rd_support)
        else:
            distributions["final_net_gain_rate_on_rd"].append(0.0)
            distributions["final_harm_rate_on_rd"].append(0.0)
    source_group_counts: dict[str, int] = {}
    for source_name, _parent_id in groups:
        source_group_counts[source_name] = source_group_counts.get(source_name, 0) + 1
    return {
        "enabled": True,
        "replicates": int(replicates),
        "seed": int(seed),
        "group_key": "source_name|parent_sample_id",
        "parent_group_count": int(group_count),
        "source_parent_group_counts": source_group_counts,
        "window_sample_count": sample_count,
        "metrics": {
            name: _bootstrap_interval(values)
            for name, values in distributions.items()
        },
    }


def _public_metrics(metrics: dict[str, object]) -> dict[str, object]:
    excluded = {
        "logits",
        "traj_logits",
        "joint_logits",
        "rd_logits",
        "labels",
        "sample_ids",
        "source_names",
        "label_names",
        "has_rd",
        "tr_rd_active",
        "rd_activated",
        "traj_cosine",
        "rd_cosine",
        "rd_pair_fusion_weights",
        "rd_alignment_quality",
    }
    return {
        key: _to_jsonable(value)
        for key, value in metrics.items()
        if key not in excluded
    }


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    split: str = "val",
    device_override: str = "",
    num_workers: int | None = None,
    max_batches: int = 0,
    bootstrap_replicates: int = 0,
    bootstrap_seed: int = 42,
) -> dict[str, object]:
    config = _config_from_checkpoint(
        checkpoint_path,
        device_override=device_override,
        num_workers=num_workers,
    )
    device, device_info = resolve_device(config.device)
    loaders = _build_expert1_loaders(config, validate_csv_files=False)
    representation_builder, rd_proj, model = _build_expert1_modules(config, device)
    class_weights = _build_class_weights_from_index(loaders.train_loader.dataset.index_df).to(device)
    rd_class_weights = _build_rd_class_weights_from_index(loaders.train_loader.dataset.index_df).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    checkpoint = _load_checkpoint_into_modules(
        checkpoint_path,
        representation_builder,
        rd_proj,
        model,
        device,
    )
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    stage2_enabled, stage2_scale, stage2_epoch = _checkpoint_eval_runtime(
        checkpoint_epoch,
        config,
    )
    split_loaders = {
        "val": loaders.val_loader,
        "test": loaders.test_loader,
    }
    requested_splits = ["val", "test"] if split == "both" else [split]
    window_metadata = _load_window_metadata(config.data_root, config.index_name)
    split_metrics: dict[str, dict[str, object]] = {}
    for split_name in requested_splits:
        raw_metrics = _run_epoch(
            representation_builder,
            rd_proj,
            model,
            split_loaders[split_name],
            criterion,
            optimizer=None,
            device=device,
            train=False,
            config=config,
            stage2_enabled=stage2_enabled,
            stage2_scale=stage2_scale,
            stage2_epoch=stage2_epoch,
            rd_class_weights=rd_class_weights,
            source_feature_stats=checkpoint.get("source_feature_stats"),
            max_batches=max(0, int(max_batches)),
            compute_aux_loss=config.compute_eval_aux_loss,
        )
        raw_metrics["selected_joint_logit_delta_scale"] = float(model.joint_logit_delta_scale)
        parent_aggregation = _parent_window_aggregation_metrics(raw_metrics, window_metadata)
        parent_mean_logit = parent_aggregation.get("mean_logit", {})
        if isinstance(parent_mean_logit, dict):
            raw_metrics["parent_mean_logit_macro_f1"] = float(parent_mean_logit.get("macro_f1", 0.0))
        raw_metrics.update(_parent_weak_class_metrics(parent_aggregation))
        raw_metrics["rd_joint_bottleneck_summary"] = _rd_joint_bottleneck_summary(raw_metrics, config)
        split_metrics[split_name] = {
            "metrics": _public_metrics(raw_metrics),
            "parent_aggregation": _to_jsonable(parent_aggregation),
            "grouped_bootstrap": _grouped_bootstrap_summary(
                raw_metrics,
                replicates=max(0, int(bootstrap_replicates)),
                seed=int(bootstrap_seed),
            ),
        }

    return {
        "purpose": "independent_checkpoint_evaluation",
        "checkpoint": str(checkpoint_path.as_posix()),
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_selection_reason": checkpoint.get("selection_reason", ""),
        "checkpoint_selection_eligible": bool(checkpoint.get("selection_eligible", False)),
        "checkpoint_candidate_role": str(checkpoint.get("candidate_role", "unknown")),
        "checkpoint_locked_precision_parameters": _to_jsonable(
            checkpoint.get("locked_precision_parameters", {})
        ),
        "runtime_joint_logit_delta_scale": float(model.joint_logit_delta_scale),
        "runtime_stage2": {
            "enabled": bool(stage2_enabled),
            "scale": float(stage2_scale),
            "epoch": int(stage2_epoch),
        },
        "effective_config": _to_jsonable(config.__dict__),
        "device": device_info,
        "split_sizes": {
            "train": len(loaders.split.train_ids),
            "val": len(loaders.split.val_ids),
            "test": len(loaders.split.test_ids),
        },
        "evaluation": split_metrics,
        "test_used_for_selection": False,
        "selection_or_calibration_modified": False,
    }


def main() -> None:
    args = parse_args()
    result = evaluate_checkpoint(
        args.checkpoint,
        split=args.split,
        device_override=args.device,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        split_name: {
            "accuracy": metrics["metrics"].get("accuracy", 0.0),
            "macro_f1": metrics["metrics"].get("macro_f1", 0.0),
            "traj_macro_f1": metrics["metrics"].get("traj_macro_f1", 0.0),
            "final_net_gain_rate_on_rd": metrics["metrics"].get("final_net_gain_rate_on_rd", 0.0),
            "final_harm_rate_on_rd": metrics["metrics"].get("final_harm_rate_on_rd", 0.0),
        }
        for split_name, metrics in result["evaluation"].items()
    }
    print(json.dumps({"output": str(args.output.as_posix()), "metrics": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
