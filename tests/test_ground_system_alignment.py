from __future__ import annotations

import json
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from config.configmanager import load_expert4_config, load_system_config
from src.core.expert_other_trainer import _apply_source_feature_norm, _groundness_details, _load_shared_representation_builder
from src.core.experts import GroundTargetExpert
from src.core.forwardlayer.dataloaders import build_dataloaders
from src.core.forwardlayer.labels import DEFAULT_VOCAB
from src.core.physical import PHYSICAL_FEATURE_COLUMNS
from src.core.system import RadarClassificationSystem
from src.core.system_trainer import _project_to_global_logits
from src.core.trajectory import TrackRepresentationBuilder


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "tests" / "ground_system_alignment_report.json"
GROUND_LABELS = ["PeopleTarget", "CarTarget", "ShipTarget"]


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this diagnostic")
    return torch.device("cuda")


def _load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint_path = ROOT / Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device, weights_only=True)


def _latest_system_checkpoint() -> Path:
    candidates = sorted(
        (ROOT / "artifacts").glob("**/system_config_smoke/best_model.pt"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("no system_config_smoke/best_model.pt checkpoint found")
    return candidates[0]


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _source_stats_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, dict[str, torch.Tensor]] | None:
    stats = checkpoint.get("source_feature_stats")
    if not stats:
        return None
    return {
        str(source_name): {
            "center": item["center"].detach().cpu() if isinstance(item["center"], torch.Tensor) else torch.tensor(item["center"], dtype=torch.float32),
            "scale": item["scale"].detach().cpu() if isinstance(item["scale"], torch.Tensor) else torch.tensor(item["scale"], dtype=torch.float32),
        }
        for source_name, item in stats.items()
    }


def _ground_global_indices() -> dict[int, str]:
    return {DEFAULT_VOCAB.label_to_index[name]: name for name in GROUND_LABELS}


def _ground_group_metrics(pred_ground: torch.Tensor, true_ground: torch.Tensor) -> dict[str, float | int]:
    tp = int((pred_ground & true_ground).sum().item())
    fp = int((pred_ground & ~true_ground).sum().item())
    fn = int((~pred_ground & true_ground).sum().item())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "support": int(true_ground.sum().item()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_ground_on_non_ground_count": fp,
    }


def _local_ground_metrics(
    ground_logits: torch.Tensor,
    global_labels: torch.Tensor,
    source_names: list[str],
    gate_by_groundness: torch.Tensor | None = None,
) -> dict[str, Any]:
    ground_indices = _ground_global_indices()
    true_ground = torch.tensor([int(label) in ground_indices for label in global_labels.tolist()], dtype=torch.bool)
    local_labels = torch.full_like(global_labels, fill_value=-1)
    for local_idx, label_name in enumerate(GROUND_LABELS):
        local_labels[global_labels == DEFAULT_VOCAB.label_to_index[label_name]] = local_idx
    local_preds = ground_logits.argmax(dim=-1)
    if gate_by_groundness is not None:
        pred_ground = gate_by_groundness.bool()
    else:
        pred_ground = torch.ones_like(true_ground, dtype=torch.bool)
    ground_group = _ground_group_metrics(pred_ground, true_ground)
    per_class: dict[str, dict[str, float | int]] = {}
    for local_idx, label_name in enumerate(GROUND_LABELS):
        mask = local_labels == local_idx
        pred_mask = (local_preds == local_idx) & pred_ground
        tp = int((mask & pred_mask).sum().item())
        fp = int((~mask & pred_mask).sum().item())
        fn = int((mask & ~pred_mask).sum().item())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        per_class[label_name] = {
            "support": int(mask.sum().item()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    ground_mask = true_ground
    local_accuracy = float((local_preds[ground_mask] == local_labels[ground_mask]).float().mean().item()) if bool(ground_mask.any()) else 0.0
    buckets: dict[str, list[int]] = defaultdict(list)
    for idx, source_name in enumerate(source_names):
        buckets[str(source_name)].append(idx)
    by_source: dict[str, dict[str, float | int]] = {}
    for source_name, indices in buckets.items():
        index_tensor = torch.tensor(indices, dtype=torch.long)
        source_ground = true_ground[index_tensor]
        if bool(source_ground.any()):
            by_source[source_name] = {
                "count": len(indices),
                "ground_support": int(source_ground.sum().item()),
                "local_ground_accuracy": float((local_preds[index_tensor][source_ground] == local_labels[index_tensor][source_ground]).float().mean().item()),
            }
    return {
        "local_ground_accuracy_on_true_ground": local_accuracy,
        "ground_group": ground_group,
        "per_class": per_class,
        "by_source": by_source,
        "prediction_distribution": dict(Counter(GROUND_LABELS[int(idx)] for idx in local_preds[true_ground].tolist())),
    }


def _build_system_model(config, checkpoint: dict[str, Any], device: torch.device) -> RadarClassificationSystem:
    model = RadarClassificationSystem(
        sequence_input_dim=10,
        physical_input_dim=len(PHYSICAL_FEATURE_COLUMNS),
        model_dim=config.hidden_dim,
        dropout=config.dropout,
        low_altitude_cosface_scale=config.low_altitude_cosface_scale,
        low_altitude_cosface_margin=config.low_altitude_cosface_margin,
        low_altitude_reject_threshold=config.low_altitude_reject_threshold,
        low_altitude_rd_activation_threshold=config.low_altitude_rd_activation_threshold,
        low_altitude_use_rd_patch_cnn=config.low_altitude_use_rd_patch_cnn,
        low_altitude_initial_rd_gate_bias=config.low_altitude_initial_rd_gate_bias,
    ).to(device)
    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    allowed_missing_prefixes = ("low_altitude_expert.deep_",)
    disallowed_missing = [
        key
        for key in load_result.missing_keys
        if not key.startswith(allowed_missing_prefixes)
    ]
    if disallowed_missing or load_result.unexpected_keys:
        raise RuntimeError(
            "system checkpoint is incompatible with ground alignment diagnostic: "
            f"missing={disallowed_missing}, unexpected={list(load_result.unexpected_keys)}"
        )
    model.eval()
    return model


def _build_expert4(config, checkpoint: dict[str, Any], device: torch.device) -> tuple[TrackRepresentationBuilder, GroundTargetExpert]:
    dropout = float(checkpoint.get("config", {}).get("dropout", config.dropout))
    representation = TrackRepresentationBuilder(
        sequence_input_dim=10,
        physical_input_dim=len(PHYSICAL_FEATURE_COLUMNS),
        model_dim=128,
        dropout=dropout,
    ).to(device)
    _load_shared_representation_builder(representation, ROOT / str(checkpoint["shared_representation_checkpoint"]), device)
    model = GroundTargetExpert(
        deep_dim=128,
        raw_physical_dim=10,
        dropout=dropout,
        groundness_threshold=float(checkpoint.get("groundness_threshold", 0.5)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return representation, model


def _collect_ground_head(
    representation_builder: TrackRepresentationBuilder,
    ground_head: GroundTargetExpert,
    loader,
    source_stats: dict[str, dict[str, torch.Tensor]] | None,
    device: torch.device,
) -> dict[str, Any]:
    logits_all: list[torch.Tensor] = []
    probabilities_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    source_names_all: list[str] = []
    representation_builder.eval()
    ground_head.eval()
    with torch.no_grad():
        for batch in loader:
            features = batch.features.to(device)
            if source_stats is not None:
                features = _apply_source_feature_norm(features, batch.source_names, source_stats)
            rep = representation_builder(
                sequence_features=features,
                physical_features=batch.physical_features.to(device),
                padding_mask=batch.padding_mask.to(device),
            )
            output = ground_head(rep.deep_features, rep.raw_physical_features)
            logits_all.append(output.logits.detach().cpu())
            probabilities_all.append(output.auxiliary["groundness_probability"].detach().cpu())
            labels_all.append(batch.labels.detach().cpu())
            source_names_all.extend(str(name) for name in batch.source_names)
    return {
        "ground_logits": torch.cat(logits_all, dim=0),
        "groundness_probability": torch.cat(probabilities_all, dim=0),
        "labels": torch.cat(labels_all, dim=0),
        "source_names": source_names_all,
    }


def _collect_system_final(
    model: RadarClassificationSystem,
    loader,
    source_stats: dict[str, dict[str, torch.Tensor]] | None,
    config,
    device: torch.device,
) -> dict[str, Any]:
    logits_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    source_names_all: list[str] = []
    route_logits_all: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            features = batch.features.to(device)
            if source_stats is not None:
                features = _apply_source_feature_norm(features, batch.source_names, source_stats)
            output = model(
                sequence_features=features,
                physical_features=batch.physical_features.to(device),
                rd_features=batch.rd_features.to(device),
                rd_patch_features=batch.rd_patch_features.to(device),
                low_altitude_targets=None,
                rd_is_multimodal=batch.rd_is_multimodal.to(device),
                padding_mask=batch.padding_mask.to(device),
                rd_padding_mask=batch.rd_padding_mask.to(device),
                rd_patch_padding_mask=batch.rd_patch_padding_mask.to(device),
                enable_low_altitude_reject=config.enable_low_altitude_reject_in_eval,
            )
            logits = _project_to_global_logits(
                batch.labels.to(device),
                output,
                len(DEFAULT_VOCAB.labels),
                device,
                config,
                source_names=batch.source_names,
                apply_source_aircraft_policy=True,
            )
            logits_all.append(logits.detach().cpu())
            labels_all.append(batch.labels.detach().cpu())
            route_logits_all.append(output.routing.route_logits.detach().cpu())
            source_names_all.extend(str(name) for name in batch.source_names)
    class_logits = torch.cat(logits_all, dim=0)
    labels = torch.cat(labels_all, dim=0)
    preds = class_logits.argmax(dim=-1)
    buckets: dict[str, list[int]] = defaultdict(list)
    for idx, source_name in enumerate(source_names_all):
        buckets[str(source_name)].append(idx)
    return {
        "class_logits": class_logits,
        "labels": labels,
        "source_names": source_names_all,
        "by_source": {
            source_name: {
                "count": len(indices),
                "accuracy": float((preds[torch.tensor(indices)] == labels[torch.tensor(indices)]).float().mean().item()),
            }
            for source_name, indices in buckets.items()
        },
        "route_logits": torch.cat(route_logits_all, dim=0) if route_logits_all else torch.empty((0, len(DEFAULT_VOCAB.labels))),
    }


def _head_report(collected: dict[str, Any], threshold: float) -> dict[str, Any]:
    labels = collected["labels"]
    probs = collected["groundness_probability"]
    true_ground = torch.tensor([DEFAULT_VOCAB.decode(int(label)) in GROUND_LABELS for label in labels.tolist()], dtype=torch.bool)
    groundness_labels = true_ground.long()
    gated = probs >= float(threshold)
    return {
        "local_ground": _local_ground_metrics(
            collected["ground_logits"],
            labels,
            collected["source_names"],
            gate_by_groundness=gated,
        ),
        "local_ground_without_groundness_gate": _local_ground_metrics(
            collected["ground_logits"],
            labels,
            collected["source_names"],
            gate_by_groundness=None,
        ),
        "groundness": _groundness_details(probs, groundness_labels, threshold),
        "groundness_probability_mean_true_ground": float(probs[true_ground].mean().item()) if bool(true_ground.any()) else 0.0,
        "groundness_probability_mean_non_ground": float(probs[~true_ground].mean().item()) if bool((~true_ground).any()) else 0.0,
    }


def _final_report(collected: dict[str, Any]) -> dict[str, Any]:
    logits = collected["class_logits"]
    labels = collected["labels"]
    preds = logits.argmax(dim=-1)
    ground_indices = {DEFAULT_VOCAB.label_to_index[name] for name in GROUND_LABELS}
    true_ground = torch.tensor([int(label) in ground_indices for label in labels.tolist()], dtype=torch.bool)
    pred_ground = torch.tensor([int(label) in ground_indices for label in preds.tolist()], dtype=torch.bool)
    return {
        "accuracy": float((preds == labels).float().mean().item()),
        "ground_group": _ground_group_metrics(pred_ground, true_ground),
        "prediction_distribution": dict(Counter(DEFAULT_VOCAB.decode(int(idx)) for idx in preds.tolist())),
    }


class GroundSystemAlignmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = _device()
        cls.system_config = load_system_config(ROOT / "config" / "config.json")
        cls.expert4_config = load_expert4_config(ROOT / "config" / "config.json")
        cls.system_checkpoint_path = _latest_system_checkpoint()
        cls.system_checkpoint = _load_checkpoint(cls.system_checkpoint_path, cls.device)
        cls.expert4_checkpoint = _load_checkpoint(cls.system_config.expert4_checkpoint, cls.device)
        cls.system_model = _build_system_model(cls.system_config, cls.system_checkpoint, cls.device)
        cls.expert4_representation, cls.expert4_head = _build_expert4(cls.expert4_config, cls.expert4_checkpoint, cls.device)
        loaders = build_dataloaders(
            data_root=ROOT / cls.system_config.data_root,
            batch_size=cls.system_config.batch_size,
            seed=cls.system_config.seed,
            num_workers=cls.system_config.num_workers,
        )
        system_source_stats = _source_stats_from_checkpoint(cls.system_checkpoint)
        expert4_source_stats = _source_stats_from_checkpoint(cls.expert4_checkpoint)
        threshold = float(cls.expert4_checkpoint.get("groundness_threshold", 0.5))
        standalone = _collect_ground_head(
            cls.expert4_representation,
            cls.expert4_head,
            loaders.test_loader,
            expert4_source_stats,
            cls.device,
        )
        system_rep = _collect_ground_head(
            cls.system_model.representation_builder,
            cls.expert4_head,
            loaders.test_loader,
            system_source_stats,
            cls.device,
        )
        final = _collect_system_final(
            cls.system_model,
            loaders.test_loader,
            system_source_stats,
            cls.system_config,
            cls.device,
        )
        report = {
            "system_checkpoint": str(cls.system_checkpoint_path.relative_to(ROOT)),
            "expert4_checkpoint": cls.system_config.expert4_checkpoint,
            "expert4_standalone_on_system_test_split": _head_report(standalone, threshold),
            "system_representation_plus_expert4_head": _head_report(system_rep, threshold),
            "system_final_global": _final_report(final),
        }
        standalone_f1 = float(report["expert4_standalone_on_system_test_split"]["local_ground"]["ground_group"]["f1"])
        system_rep_f1 = float(report["system_representation_plus_expert4_head"]["local_ground"]["ground_group"]["f1"])
        final_f1 = float(report["system_final_global"]["ground_group"]["f1"])
        if system_rep_f1 < standalone_f1 - 0.10:
            primary = "representation_drift"
        elif final_f1 < system_rep_f1 - 0.10:
            primary = "global_arbitration_suppression"
        else:
            primary = "expert4_head_or_data_limit"
        report["diagnosis"] = {
            "primary": primary,
            "standalone_ground_f1": standalone_f1,
            "system_rep_ground_f1": system_rep_f1,
            "system_final_ground_f1": final_f1,
        }
        REPORT_PATH.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
        cls.report = report

    def test_report_written(self) -> None:
        self.assertTrue(REPORT_PATH.exists())

    def test_alignment_has_diagnosis(self) -> None:
        self.assertIn(self.report["diagnosis"]["primary"], {
            "representation_drift",
            "global_arbitration_suppression",
            "expert4_head_or_data_limit",
        })

    def test_print_report_for_logs(self) -> None:
        print(json.dumps(_jsonable(self.report["diagnosis"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
