"""Single assembly boundary for the current strict-open-set product.

The module is intentionally small: versioned experiments own training and
artifact loading, while this boundary owns only the product topology and
action invariants.  Unknown is an action produced by L3, never a model class.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .known_gate import KnownGateClassifier, KnownGateConfig
from .open_set_architecture import (
    DBB_ACTIONS,
    DBB_SPECIALIST_TASK_NAME,
    KNOWN_GATE_TASK_NAME,
    ProductAction,
    SemanticDecision,
    resolve_semantic_action,
)
from .semantic_decision_registry import (
    FITTED_CALIBRATION_STATUS,
    verify_semantic_decision_registry_artifacts,
)
from .task_models import (
    LiquidTaskClassifier,
    LiquidTaskConfig,
    task_config_identity_sha256,
)
from .known_gate import known_gate_config_identity_sha256
from .runtime import resolve_project_path, sha256_file


@dataclass(frozen=True)
class CurrentProduct:
    """Frozen L1/L2 models plus an explicitly validated L3 registry."""

    l1: KnownGateClassifier
    l2: LiquidTaskClassifier
    registry: Mapping[str, Any]
    project_root: Path | None = None

    def __post_init__(self) -> None:
        if self.l1.task_name != KNOWN_GATE_TASK_NAME:
            raise ValueError("current product L1 must be KnownGate")
        if self.l2.task_name != DBB_SPECIALIST_TASK_NAME:
            raise ValueError("current product L2 must be the DBB specialist")
        if tuple(self.l2.class_names) != (
            "DroneTarget",
            "BirdTarget",
            "BalloonTarget",
        ):
            raise ValueError("current product L2 must remain a three-class DBB specialist")
        if self.project_root is None:
            raise ValueError(
                "current product construction requires project_root for artifact verification"
            )
        validate_product_model_configs(self.l1.config, self.l2.config)
        validated = validate_product_artifact_bindings(
            project_root=self.project_root,
            l1_config=self.l1.config,
            l2_config=self.l2.config,
            registry=self.registry,
        )
        if validated.get("calibration", {}).get("status") != FITTED_CALIBRATION_STATUS:
            raise PermissionError(
                "an unfitted L3 registry may be used for audit, but cannot build the current product"
            )
        if validated.get("final_actions_enabled") is not True:
            raise PermissionError("current product requires explicitly enabled L3 actions")

    def decide(
        self,
        *,
        observation_eligible: bool,
        known_target_accepted: bool,
        known_clutter_accepted: bool,
        dbb_subclass: ProductAction | None = None,
        dbb_subclass_accepted: bool = False,
    ) -> SemanticDecision:
        """Resolve one mutually exclusive action through the frozen L3 rules."""

        decision = resolve_semantic_action(
            observation_eligible=observation_eligible,
            known_target_accepted=known_target_accepted,
            known_clutter_accepted=known_clutter_accepted,
            dbb_subclass=dbb_subclass,
            dbb_subclass_accepted=dbb_subclass_accepted,
        )
        if decision.action in DBB_ACTIONS and not known_target_accepted:
            raise AssertionError("DBB action escaped without an accepted L1 target route")
        if not observation_eligible and decision.action != ProductAction.PENDING_ABSTAIN:
            raise AssertionError("ineligible observations must resolve to Pending")
        return decision


def validate_product_model_configs(
    l1_config: KnownGateConfig,
    l2_config: LiquidTaskConfig,
) -> None:
    """Validate the non-negotiable topology before model construction."""

    if l1_config.task_name != KNOWN_GATE_TASK_NAME:
        raise ValueError("L1 config is not a KnownGate config")
    if l2_config.task.name != DBB_SPECIALIST_TASK_NAME:
        raise ValueError("L2 config is not a DBB specialist config")
    if "UnknownTarget" in l2_config.task.class_names:
        raise ValueError("UnknownTarget must not be an L2 learned class")
    if tuple(l2_config.task.class_names) != (
        "DroneTarget",
        "BirdTarget",
        "BalloonTarget",
    ):
        raise ValueError("L2 config class order is not the canonical DBB order")


def _config_task_name(config: KnownGateConfig | LiquidTaskConfig) -> str:
    if isinstance(config, KnownGateConfig):
        return config.task_name
    return config.task.name


def _config_identity(config: KnownGateConfig | LiquidTaskConfig) -> str:
    if isinstance(config, KnownGateConfig):
        return known_gate_config_identity_sha256(config)
    return task_config_identity_sha256(config)


def _verify_component_binding(
    *,
    project_root: Path,
    component_name: str,
    binding: Mapping[str, Any],
    config: KnownGateConfig | LiquidTaskConfig,
) -> None:
    """Verify registry metadata against the actual config/checkpoint pair.

    The registry hash check proves file integrity; this second check proves
    that the files are the files for the models passed to the product.  A
    valid hash on a different task/config must not be enough to assemble a
    product accidentally.
    """

    config_path = resolve_project_path(
        project_root,
        str(binding["config_path"]),
        must_exist=True,
    )
    checkpoint_path = resolve_project_path(
        project_root,
        str(binding["checkpoint_path"]),
        must_exist=True,
    )
    summary_path = resolve_project_path(
        project_root,
        str(binding["training_summary_path"]),
        must_exist=True,
    )
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    if config_payload.get("config_name") != config.config_name:
        raise ValueError(
            f"current product {component_name} config_name does not match the registry binding"
        )
    if _config_task_name(config) != binding.get("task_name"):
        raise ValueError(
            f"current product {component_name} task does not match the registry binding"
        )
    if sha256_file(config_path).lower() != str(binding["config_sha256"]).lower():
        raise ValueError(f"current product {component_name} config hash changed during verification")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"current product {component_name} checkpoint payload is invalid")
    if checkpoint.get("task_name") != _config_task_name(config):
        raise ValueError(
            f"current product {component_name} checkpoint task does not match its config"
        )
    if checkpoint.get("config_name") != config.config_name:
        raise ValueError(
            f"current product {component_name} checkpoint config does not match its config"
        )
    expected_schema = getattr(config.train, "checkpoint_schema", None)
    if expected_schema is not None and checkpoint.get("checkpoint_schema") != expected_schema:
        raise ValueError(
            f"current product {component_name} checkpoint schema does not match its config"
        )
    expected_identity = _config_identity(config)
    if checkpoint.get("config_identity_sha256") != expected_identity:
        raise ValueError(
            f"current product {component_name} checkpoint config identity does not match"
        )
    if component_name == "l1_known_gate":
        if checkpoint.get("config_file_sha256") != sha256_file(config_path):
            raise ValueError("current product L1 checkpoint config file hash does not match")
        if checkpoint.get("unknown_is_model_class") is not False:
            raise ValueError("current product L1 checkpoint violates strict Unknown semantics")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("config_name") not in {None, config.config_name}:
        raise ValueError(
            f"current product {component_name} training summary config does not match"
        )
    if summary.get("task_name") not in {None, _config_task_name(config)}:
        raise ValueError(
            f"current product {component_name} training summary task does not match"
        )
    best_checkpoint = summary.get("best_checkpoint")
    if best_checkpoint:
        summary_checkpoint = resolve_project_path(
            project_root,
            str(best_checkpoint),
            must_exist=True,
        )
        if summary_checkpoint != checkpoint_path:
            raise ValueError(
                f"current product {component_name} summary best checkpoint is not the bound checkpoint"
            )


def validate_product_artifact_bindings(
    *,
    project_root: str | Path,
    l1_config: KnownGateConfig,
    l2_config: LiquidTaskConfig,
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless registry, configs, summaries and checkpoints agree."""

    root = Path(project_root).resolve()
    validated = verify_semantic_decision_registry_artifacts(registry, root)
    components = dict(validated["component_bindings"])
    _verify_component_binding(
        project_root=root,
        component_name="l1_known_gate",
        binding=components["l1_known_gate"],
        config=l1_config,
    )
    _verify_component_binding(
        project_root=root,
        component_name="l2_dbb_specialist",
        binding=components["l2_dbb_specialist"],
        config=l2_config,
    )
    return validated


def build_current_product(
    *,
    l1_config: KnownGateConfig,
    l2_config: LiquidTaskConfig,
    registry: Mapping[str, Any],
    project_root: str | Path,
) -> CurrentProduct:
    """Construct the only supported current product topology."""

    validate_product_model_configs(l1_config, l2_config)
    root = Path(project_root).resolve()
    validate_product_artifact_bindings(
        project_root=root,
        l1_config=l1_config,
        l2_config=l2_config,
        registry=registry,
    )
    return CurrentProduct(
        l1=KnownGateClassifier(l1_config),
        l2=LiquidTaskClassifier(l2_config),
        registry=dict(registry),
        project_root=root,
    )


__all__ = [
    "CurrentProduct",
    "build_current_product",
    "validate_product_artifact_bindings",
    "validate_product_model_configs",
]
