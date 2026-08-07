from __future__ import annotations

import math
from dataclasses import dataclass

from .data.rd_physical_five_dataset import RD_FIVE_LABELS, RD_FIVE_QUALITY_NAMES
from .data.rd_physical_window_dataset import PHYSICAL_RD_INPUT_CHANNELS
from .data.rd_physical_specialist_dataset import specialist_label_names


@dataclass(frozen=True)
class PhysicalRDSpecialistConfig:
    kind: str = "parent_gate"
    index_name: str = "expert1_rd_only_causal_window_index_v1_physical_v3_1.csv"
    normalization_stats_name: str = "expert1_rd_physical_train_db_normalization_five_v1.json"
    feature_stats_name: str = "expert1_rd_physical_feature_stats_five_v1.json"
    output_dir: str = "artifacts/rd-only-specialist"
    epochs: int = 18
    batch_size: int = 4
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    dropout: float = 0.1
    input_channels: int = PHYSICAL_RD_INPUT_CHANNELS
    token_dim: int = 256
    quality_dim: int = len(RD_FIVE_QUALITY_NAMES)
    prefix_aux_weight: float = 0.10
    micro_aux_weight: float = 0.10
    prefix_max_loss_weight: float = 0.5
    evidence_target_frames: int = 24
    focal_gamma: float = 0.0
    class_weights: tuple[float, ...] = (1.0, 1.0)
    reference_metrics: dict[str, dict[str, float]] | None = None
    reference_validation_report: str = ""
    reference_validation_report_sha256: str = ""
    reference_checkpoint_sha256: str = ""
    reference_predictions_csv: str = ""
    reference_predictions_csv_sha256: str = ""
    reference_route_recalls: dict[str, float] | None = None
    non_degradation_tolerance: float = 0.0
    enforce_non_degradation: bool = True
    # Legacy specialist experiments always emitted a ``best_*`` fallback even
    # when it failed the configured guards.  New safety-critical experiments
    # can require a qualified checkpoint without changing historical config
    # behavior or old checkpoint loading.
    require_qualified_checkpoint: bool = False
    # Checkpoint selection is validation-only.  The weak-class policy is
    # useful for imbalanced parent routing: when no epoch satisfies every
    # reference guard, select the epoch with the best bottleneck recall
    # instead of silently falling back to macro-F1.
    selection_policy: str = "eligible_macro_f1"
    tr_entry_preserved: bool = True
    allow_tr_inputs: bool = False
    max_train_batches_per_epoch: int = 0
    max_val_batches: int = 0
    log_interval_batches: int = 25
    num_workers: int = 0
    seed: int = 42
    device: str = "cuda"
    run_test_evaluation: bool = False

    def __post_init__(self) -> None:
        if self.kind not in {"parent_gate", "dbb", "cu"}:
            raise ValueError("kind must be parent_gate, dbb, or cu")
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("invalid specialist optimization settings")
        if not 0.0 <= self.dropout < 1.0 or self.token_dim < 32:
            raise ValueError("invalid specialist model dimensions")
        if self.input_channels != PHYSICAL_RD_INPUT_CHANNELS or self.quality_dim != len(RD_FIVE_QUALITY_NAMES):
            raise ValueError("specialist RD input schema mismatch")
        if self.prefix_aux_weight < 0.0 or self.micro_aux_weight < 0.0 or self.focal_gamma < 0.0 or not 0.0 < self.prefix_max_loss_weight <= 1.0 or self.evidence_target_frames < 1:
            raise ValueError("specialist auxiliary weights must be non-negative")
        if self.non_degradation_tolerance < 0.0:
            raise ValueError("non_degradation_tolerance must be non-negative")
        if self.selection_policy not in {"eligible_macro_f1", "eligible_weak_class_guard", "eligible_reference_guard"}:
            raise ValueError(
                "selection_policy must be eligible_macro_f1, eligible_weak_class_guard, or eligible_reference_guard"
            )
        expected = len(specialist_label_names(self.kind))
        weights = tuple(float(value) for value in self.class_weights)
        if len(weights) != expected or any(value <= 0.0 for value in weights):
            raise ValueError(f"class_weights must contain {expected} positive values")
        if not self.tr_entry_preserved or self.allow_tr_inputs:
            raise ValueError("RD specialist preserves the TR entry but forbids TR inputs")
        if self.max_train_batches_per_epoch < 0 or self.max_val_batches < 0 or self.num_workers < 0 or self.log_interval_batches < 1:
            raise ValueError("invalid specialist batch limits")
        if self.run_test_evaluation:
            raise ValueError("RD specialist training forbids test evaluation")
        object.__setattr__(self, "class_weights", weights)
        if self.reference_metrics is not None:
            if not isinstance(self.reference_metrics, dict):
                raise ValueError("reference_metrics must be a mapping")
            for label, values in self.reference_metrics.items():
                if label not in specialist_label_names(self.kind) or not isinstance(values, dict):
                    raise ValueError("reference_metrics labels must match specialist labels")
                if any(name not in {"precision", "recall", "f1"} or not math.isfinite(float(value)) or float(value) < 0.0 or float(value) > 1.0 for name, value in values.items()):
                    raise ValueError("reference_metrics values must be finite [0,1] precision/recall/f1")
        provenance = (
            self.reference_validation_report,
            self.reference_validation_report_sha256,
            self.reference_checkpoint_sha256,
        )
        if any(provenance) and not all(provenance):
            raise ValueError("reference provenance requires report path, report SHA-256, and checkpoint SHA-256")
        for name, value in (
            ("reference_validation_report_sha256", self.reference_validation_report_sha256),
            ("reference_checkpoint_sha256", self.reference_checkpoint_sha256),
            ("reference_predictions_csv_sha256", self.reference_predictions_csv_sha256),
        ):
            if value and (len(value) != 64 or any(character not in "0123456789abcdefABCDEF" for character in value)):
                raise ValueError(f"{name} must be a 64-character SHA-256")
        route_provenance = (
            self.reference_predictions_csv,
            self.reference_predictions_csv_sha256,
            self.reference_route_recalls,
        )
        if any(value not in (None, "") for value in route_provenance) and not all(
            value not in (None, "") for value in route_provenance
        ):
            raise ValueError("route-recall provenance requires predictions CSV path/hash and route recalls")
        if self.reference_route_recalls is not None:
            if self.kind != "parent_gate" or set(self.reference_route_recalls) != set(RD_FIVE_LABELS):
                raise ValueError("reference_route_recalls must cover all five original labels for parent_gate")
            if any(not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0 for value in self.reference_route_recalls.values()):
                raise ValueError("reference_route_recalls values must be finite [0,1]")
        if self.selection_policy == "eligible_reference_guard":
            if not self.enforce_non_degradation:
                raise ValueError("eligible_reference_guard requires enforce_non_degradation=true")
            if not self.require_qualified_checkpoint:
                raise ValueError("eligible_reference_guard requires require_qualified_checkpoint=true")
            if not self.reference_metrics or not all(provenance) or self.reference_route_recalls is None:
                raise ValueError(
                    "eligible_reference_guard requires verified metric, report/checkpoint, and original-class route-recall references"
                )


__all__ = ["PhysicalRDSpecialistConfig"]
